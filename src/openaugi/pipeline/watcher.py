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
    """Run one Layer 0 + Layer 1 cycle."""
    from openaugi.pipeline.runner import run_layer0
    from openaugi.store.sqlite import SQLiteStore

    n = len(changed_paths)
    logger.info(f"Detected {n} changed file(s), running incremental ingest")

    store = SQLiteStore(db_path)
    try:
        exclude = config.get("vault", {}).get("exclude_patterns")
        workers = config.get("vault", {}).get("max_workers", 4)

        result = run_layer0(vault_path, store, exclude_patterns=exclude, max_workers=workers)
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
    except Exception as e:
        logger.error(f"Ingest cycle failed: {e}", exc_info=True)
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
    while not handler.stopped:
        if not handler.wait_for_change(timeout=1.0):
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
