"""File watcher — triggers incremental ingest on vault changes.

Watches an Obsidian vault directory for .md file changes using watchdog.
Debounces rapid saves (e.g., Obsidian autosave) before triggering Layer 0 + 1.

Designed to run as a separate long-lived process alongside `openaugi serve`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """Collects .md file events and fires a callback after a quiet period."""

    def __init__(
        self,
        debounce_seconds: float,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self.debounce_seconds = debounce_seconds
        self.exclude_patterns = exclude_patterns or []
        self._changed = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending_paths: set[str] = set()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = str(event.src_path)
        if not src.endswith(".md"):
            return
        if self._is_excluded(src):
            return

        with self._lock:
            self._pending_paths.add(src)
        self._changed.set()

    def _is_excluded(self, path: str) -> bool:
        """Check if path matches any exclude pattern."""
        from fnmatch import fnmatch

        for pattern in self.exclude_patterns:
            if fnmatch(path, f"*/{pattern}") or fnmatch(path, pattern):
                return True
        return False

    def drain(self) -> set[str]:
        """Return and clear pending changed paths."""
        with self._lock:
            paths = self._pending_paths
            self._pending_paths = set()
            self._changed.clear()
        return paths

    def wait_for_change(self, timeout: float | None = None) -> bool:
        """Block until a change is detected. Returns False if stopped."""
        return self._changed.wait(timeout=timeout)

    def stop(self) -> None:
        self._stop.set()
        self._changed.set()  # unblock any wait

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


def _run_ingest_cycle(
    vault_path: Path,
    db_path: str,
    config: dict[str, Any],
    changed_paths: set[str],
) -> None:
    """Run one Layer 0 + Layer 1 cycle, then dispatch any zzz instructions."""
    from openaugi.pipeline.runner import run_layer0
    from openaugi.store.sqlite import SQLiteStore

    n = len(changed_paths)
    logger.info(f"Detected {n} changed file(s), running incremental ingest")

    store = SQLiteStore(db_path)
    try:
        exclude = config.get("vault", {}).get("exclude_patterns")
        workers = config.get("vault", {}).get("max_workers", 4)
        source_rules = config.get("vault", {}).get("source_rules")
        provenance_rules = config.get("vault", {}).get("provenance_rules")

        result = run_layer0(
            vault_path,
            store,
            exclude_patterns=exclude,
            max_workers=workers,
            source_rules=source_rules,
            provenance_rules=provenance_rules,
        )
        stats = result["stats"]
        logger.info(
            f"Layer 0 done: {result['blocks_added']} added, "
            f"{result['blocks_kept']} kept, {result['blocks_removed']} removed "
            f"({stats['total_blocks']} total blocks)"
        )

        # Layer 1: embedding — graceful fallback
        try:
            from openaugi.models import get_embedding_model
            from openaugi.pipeline.embed import run_embed

            model = get_embedding_model(config.get("models", {}).get("embedding"))
            count = run_embed(store, model)
            if count:
                logger.info(f"Embedded {count} blocks")
        except Exception as e:
            logger.warning(f"Embedding skipped: {e}")
            logger.info("Blocks saved without embeddings — will retry on next cycle")

        # Post-ingest: queue zzz instructions, superseding any they replace.
        # Nothing dispatches here — the queue drains once a block has settled,
        # so a half-typed instruction never becomes a task. See dispatch.py.
        new_blocks = result.get("new_data_blocks", [])
        try:
            from openaugi.pipeline.dispatch import record_zzz_changes

            record_zzz_changes(
                new_blocks,
                result.get("removed_data_blocks", []),
                store,
                vault_path,
            )
        except Exception as e:
            logger.error(f"ZZZ queueing failed: {e}", exc_info=True)

        drain_zzz_queue(vault_path, store, config)

        if new_blocks:
            # Proactive echo: surface older thinking that bears on what was
            # just written. Never fails the cycle — it is an extra, not a step.
            try:
                from openaugi.models import get_embedding_model
                from openaugi.pipeline.echo import run_echo

                run_echo(
                    new_blocks,
                    vault_path,
                    store,
                    get_embedding_model(config.get("models", {}).get("embedding")),
                    config,
                )
            except Exception as e:
                logger.error(f"Proactive echo failed: {e}", exc_info=True)

            # Routing: ask where each new human block lives. One row per
            # block in the same log; nothing applies until the master box.
            try:
                from openaugi.models import get_embedding_model
                from openaugi.pipeline.route import run_routing

                run_routing(
                    new_blocks,
                    vault_path,
                    store,
                    get_embedding_model(config.get("models", {}).get("embedding")),
                    config,
                )
            except Exception as e:
                logger.error(f"Routing pass failed: {e}", exc_info=True)

        # Janitor: act on any checkbox ticked in an Augi Log
        try:
            from openaugi.pipeline.echo_janitor import process_changed

            process_changed(changed_paths, vault_path)
        except Exception as e:
            logger.error(f"Echo janitor failed: {e}", exc_info=True)

        # Janitor: apply a day's routing rows once its master box is ticked
        try:
            from openaugi.pipeline.routing_janitor import process_changed as process_routing

            process_routing(changed_paths, vault_path, store, config)
        except Exception as e:
            logger.error(f"Routing janitor failed: {e}", exc_info=True)

        # Janitor: act on any checkbox ticked on a currency board
        try:
            from openaugi.pipeline.board_janitor import process_changed as process_boards

            process_boards(changed_paths, vault_path)
        except Exception as e:
            logger.error(f"Board janitor failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Ingest cycle failed: {e}", exc_info=True)
    finally:
        store.close()


def _zzz_settle(config: dict[str, Any]) -> float:
    """Seconds a zzz block must sit unchanged before it becomes a task."""
    from openaugi.pipeline.dispatch import DEFAULT_ZZZ_SETTLE

    return float(config.get("tasks", {}).get("zzz_settle_seconds", DEFAULT_ZZZ_SETTLE))


def drain_zzz_queue(vault_path: Path, store: Any, config: dict[str, Any]) -> None:
    """Turn settled zzz blocks into task files. Never fails the caller."""
    try:
        from openaugi.pipeline.dispatch import drain_zzz_queue as _drain

        _drain(store, vault_path, settle_seconds=_zzz_settle(config))
    except Exception as e:
        logger.error(f"ZZZ dispatch failed: {e}", exc_info=True)


def _drain_tick(vault_path: Path, db_path: str, config: dict[str, Any]) -> None:
    """Drain the zzz queue outside an ingest cycle.

    Without this, an instruction written just before the vault goes quiet
    would sit queued until the next file change — the settle window would
    become "wait for the next edit", which is not a window at all.
    """
    from openaugi.store.sqlite import SQLiteStore

    store = SQLiteStore(db_path)
    try:
        drain_zzz_queue(vault_path, store, config)
    finally:
        store.close()


def _watch_loop(
    vault_path: Path,
    db_path: str,
    config: dict[str, Any],
    handler: _DebouncedHandler,
    debounce_seconds: float,
) -> None:
    """Internal watch loop — runs until handler.stop() is called."""
    import time

    drain_every = max(5.0, _zzz_settle(config) / 4)
    last_drain = time.monotonic()

    while not handler.stopped:
        if not handler.wait_for_change(timeout=1.0):
            if time.monotonic() - last_drain >= drain_every:
                last_drain = time.monotonic()
                _drain_tick(vault_path, db_path, config)
            continue

        # Debounce: wait for quiet period after last change
        while True:
            handler._changed.clear()
            if handler._changed.wait(timeout=debounce_seconds):
                continue  # Another change during debounce — reset
            else:
                break  # Quiet period elapsed

        if handler.stopped:
            break

        changed = handler.drain()
        if changed:
            _run_ingest_cycle(vault_path, db_path, config, changed)
        last_drain = time.monotonic()


def watch_vault(
    vault_path: str | Path,
    db_path: str,
    config: dict[str, Any],
    debounce_seconds: float = 30.0,
) -> None:
    """Watch vault for .md changes and run incremental ingest.

    Blocks until interrupted (Ctrl+C). Designed to run as a standalone process.
    """
    vault_path = Path(vault_path).resolve()
    if not vault_path.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")

    exclude = config.get("vault", {}).get("exclude_patterns", [])
    handler = _DebouncedHandler(debounce_seconds, exclude_patterns=exclude)

    observer = Observer()
    observer.schedule(handler, str(vault_path), recursive=True)
    observer.start()

    logger.info(f"Watching {vault_path} (debounce={debounce_seconds}s)")
    logger.info("Press Ctrl+C to stop")

    try:
        _watch_loop(vault_path, db_path, config, handler, debounce_seconds)
    except KeyboardInterrupt:
        logger.info("Stopping watcher")
    finally:
        handler.stop()
        observer.stop()
        observer.join()


def start_watcher_thread(
    vault_path: str | Path,
    db_path: str,
    config: dict[str, Any],
    debounce_seconds: float = 30.0,
) -> None:
    """Start file watcher as a daemon thread (for embedding in other processes)."""
    vault_path = Path(vault_path).resolve()
    if not vault_path.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")

    exclude = config.get("vault", {}).get("exclude_patterns", [])
    handler = _DebouncedHandler(debounce_seconds, exclude_patterns=exclude)

    observer = Observer()
    observer.schedule(handler, str(vault_path), recursive=True)
    observer.daemon = True
    observer.start()

    thread = threading.Thread(
        target=_watch_loop,
        args=(vault_path, db_path, config, handler, debounce_seconds),
        daemon=True,
        name="openaugi-watcher",
    )
    thread.start()

    logger.info(f"Watcher thread started for {vault_path} (debounce={debounce_seconds}s)")
