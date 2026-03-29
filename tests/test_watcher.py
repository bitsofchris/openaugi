"""Tests for the file watcher — debounce logic and ingest triggering."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from openaugi.pipeline.watcher import _DebouncedHandler, _run_ingest_cycle


class TestDebouncedHandler:
    def test_ignores_non_md_files(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/vault/notes/image.png"
        handler.on_any_event(event)
        assert handler.drain() == set()

    def test_collects_md_changes(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/vault/notes/daily.md"
        handler.on_any_event(event)
        assert handler.drain() == {"/vault/notes/daily.md"}

    def test_drain_clears_pending(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/vault/test.md"
        handler.on_any_event(event)
        handler.drain()
        assert handler.drain() == set()

    def test_deduplicates_same_file(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/vault/test.md"
        handler.on_any_event(event)
        handler.on_any_event(event)
        handler.on_any_event(event)
        assert handler.drain() == {"/vault/test.md"}

    def test_excludes_patterns(self):
        handler = _DebouncedHandler(
            debounce_seconds=1.0,
            exclude_patterns=[".obsidian/**", ".git/**"],
        )
        event = MagicMock()
        event.is_directory = False

        event.src_path = "/vault/.obsidian/workspace.md"
        handler.on_any_event(event)

        event.src_path = "/vault/.git/HEAD.md"
        handler.on_any_event(event)

        event.src_path = "/vault/notes/real.md"
        handler.on_any_event(event)

        assert handler.drain() == {"/vault/notes/real.md"}

    def test_ignores_directories(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/vault/notes/"
        handler.on_any_event(event)
        assert handler.drain() == set()

    def test_wait_for_change_signals(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/vault/test.md"
        handler.on_any_event(event)
        assert handler.wait_for_change(timeout=0.1) is True

    def test_wait_for_change_times_out(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        assert handler.wait_for_change(timeout=0.05) is False

    def test_stop_unblocks_wait(self):
        handler = _DebouncedHandler(debounce_seconds=1.0)
        handler.stop()
        assert handler.wait_for_change(timeout=0.1) is True
        assert handler.stopped is True


class TestRunIngestCycle:
    @patch("openaugi.store.sqlite.SQLiteStore")
    @patch("openaugi.pipeline.runner.run_layer0")
    def test_runs_layer0(self, mock_run_layer0, mock_store_cls, tmp_path: Path):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_run_layer0.return_value = {
            "blocks_added": 5,
            "blocks_kept": 10,
            "blocks_removed": 0,
            "stats": {"total_blocks": 15},
        }

        _run_ingest_cycle(
            vault_path=tmp_path,
            db_path=str(tmp_path / "test.db"),
            config={"vault": {"exclude_patterns": [".obsidian/**"]}},
            changed_paths={str(tmp_path / "note.md")},
        )

        mock_run_layer0.assert_called_once()
        mock_store.close.assert_called_once()

    @patch("openaugi.store.sqlite.SQLiteStore")
    @patch("openaugi.pipeline.runner.run_layer0")
    def test_embedding_failure_does_not_crash(
        self, mock_run_layer0, mock_store_cls, tmp_path: Path
    ):
        """If embedding fails, Layer 0 results are still persisted."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_run_layer0.return_value = {
            "blocks_added": 3,
            "blocks_kept": 0,
            "blocks_removed": 0,
            "stats": {"total_blocks": 3},
        }

        # Embedding import will succeed but model creation will fail
        with patch(
            "openaugi.models.get_embedding_model",
            side_effect=Exception("No API key configured"),
        ):
            # Should not raise
            _run_ingest_cycle(
                vault_path=tmp_path,
                db_path=str(tmp_path / "test.db"),
                config={},
                changed_paths={str(tmp_path / "note.md")},
            )

        # Layer 0 still ran
        mock_run_layer0.assert_called_once()
        mock_store.close.assert_called_once()
