"""Sync before read — keeps a Light install's index fresh with nothing running.

A Light install has no watcher, so the index is only as fresh as the last
ingest. `openaugi kb` calls `sync_before_read` before answering: if the last
refresh is older than `[kb] sync_after_seconds` (default 60), it runs the same
incremental ingest as `openaugi ingest` — only changed files are re-read, about
a second on a vault of several thousand notes — and embeds the changed blocks
when embeddings are configured. A burst of agent calls pays that once.

What it never does:
- Block the answer. Any failure is logged and the read goes ahead on the
  index as it is.
- Race another sync. A non-blocking file lock next to the database means one
  caller syncs; concurrent callers read the current index without waiting.
- Compete with `openaugi up`. When `up` holds its lock, its watcher is
  already keeping the index fresh, so this does nothing.
- Launch anything. New `zzz:` lines are queued exactly as `openaugi ingest`
  queues them (so the next `ingest` doesn't mistake them for old ones), and
  they wait for `openaugi tasks run` like any Light task.

State is two files beside the database: `<db>.kb-sync` (its mtime is the last
successful refresh) and `<db>.kb-sync.lock`.
"""

from __future__ import annotations

import fcntl
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SYNC_AFTER_SECONDS = 60.0


@dataclass(frozen=True)
class SyncOutcome:
    """What sync_before_read did. `status` is one of:

    synced · fresh (refreshed recently) · off (`[kb] auto_sync = false`) ·
    no-vault · up-running (the watcher owns freshness) · busy (another call is
    syncing) · failed (logged; the read goes ahead).
    """

    status: str
    changed: int = 0
    embedded: int = 0
    seconds: float = 0.0


def _stamp_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".kb-sync")


def _settings(config: dict) -> tuple[bool, float]:
    kb = config.get("kb", {}) or {}
    enabled = bool(kb.get("auto_sync", True))
    try:
        after = float(kb.get("sync_after_seconds", DEFAULT_SYNC_AFTER_SECONDS))
    except (TypeError, ValueError):
        after = DEFAULT_SYNC_AFTER_SECONDS
    return enabled, max(after, 0.0)


def is_fresh(db_path: Path, sync_after_seconds: float, now: float) -> bool:
    stamp = _stamp_path(db_path)
    return stamp.exists() and now - stamp.stat().st_mtime < sync_after_seconds


def _embed_changed(store, config: dict) -> int:
    """Embed blocks that need it, when embeddings are configured and installed."""
    from openaugi.models import embeddings_enabled, get_embedding_model, provider_installed

    embedding_cfg = config.get("models", {}).get("embedding")
    if not embeddings_enabled(embedding_cfg):
        return 0
    if not provider_installed((embedding_cfg or {}).get("provider", "")):
        return 0
    from openaugi.pipeline.embed import run_embed

    return run_embed(store, get_embedding_model(embedding_cfg))


def _queue_zzz(result: dict, store, vault_path: str, config: dict, first_ingest: bool) -> None:
    """Queue `zzz:` lines exactly as `openaugi ingest` does.

    Without this, a line indexed here would look old to the next `ingest` and
    never become a task. Nothing launches: in a Light install tasks wait for
    `openaugi tasks run`, and under `up` this sync does not run at all.
    """
    from openaugi.pipeline.dispatch import queue_from_ingest
    from openaugi.pipeline.watcher import _zzz_settle

    try:
        queue_from_ingest(result, store, vault_path, _zzz_settle(config), first_ingest)
    except Exception:
        logger.warning("kb sync: zzz queueing skipped", exc_info=True)


def sync_before_read(
    db_path: str | Path,
    vault_path: str | None,
    config: dict,
    *,
    up_running: Callable[[], bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> SyncOutcome:
    """Refresh the index if it is stale. Never raises; see the module docstring."""
    clock = clock or time.time
    enabled, after = _settings(config)
    if not enabled:
        return SyncOutcome("off")
    if not vault_path or not Path(vault_path).is_dir():
        return SyncOutcome("no-vault")

    db = Path(db_path)
    if is_fresh(db, after, clock()):
        return SyncOutcome("fresh")

    if up_running is None:
        from openaugi.singleton import is_held

        def _up_holds_lock() -> bool:
            return is_held("up")

        up_running = _up_holds_lock

    if up_running():
        return SyncOutcome("up-running")

    db.parent.mkdir(parents=True, exist_ok=True)
    with open(db.with_name(db.name + ".kb-sync.lock"), "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return SyncOutcome("busy")
        try:
            # Another caller may have finished a sync while we waited for nothing.
            if is_fresh(db, after, clock()):
                return SyncOutcome("fresh")
            return _sync(db, vault_path, config, clock)
        except Exception:
            logger.warning("kb sync failed; answering from the current index", exc_info=True)
            return SyncOutcome("failed")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _sync(db: Path, vault_path: str, config: dict, clock: Callable[[], float]) -> SyncOutcome:
    from openaugi.pipeline.runner import run_layer0_from_config
    from openaugi.store.sqlite import SQLiteStore

    started = clock()
    store = SQLiteStore(db)
    try:
        first_ingest = store.get_stats()["total_blocks"] == 0
        result = run_layer0_from_config(vault_path, store, config)
        _queue_zzz(result, store, vault_path, config, first_ingest)
        try:
            embedded = _embed_changed(store, config)
        except Exception:
            # Keyword search already sees the new text; embeddings catch up next time.
            logger.warning("kb sync: embedding skipped", exc_info=True)
            embedded = 0
    finally:
        store.close()
    _stamp_path(db).touch()
    changed = int(result.get("blocks_added", 0)) + int(result.get("blocks_removed", 0))
    outcome = SyncOutcome("synced", changed=changed, embedded=embedded, seconds=clock() - started)
    logger.debug("kb sync: %s", outcome)
    return outcome
