"""Tests for the file watcher — debounce logic and ingest triggering."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from openaugi.pipeline.watcher import _DebouncedHandler, _drain_tick, _run_ingest_cycle


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

    def test_the_heartbeat_view_never_wakes_the_watcher(self):
        # Whatever config says — the file the tick writes must not trigger the
        # ingest that writes it.
        handler = _DebouncedHandler(debounce_seconds=1.0, exclude_patterns=[])
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/vault/OpenAugi/Views/View - System Heartbeat.md"
        handler.on_any_event(event)
        event.src_path = "/vault/OpenAugi/Views/View - Dashboard.md"
        handler.on_any_event(event)

        assert handler.drain() == {"/vault/OpenAugi/Views/View - Dashboard.md"}

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


class TestDrainTick:
    """The drain tick is now also the lens heartbeat — no cron, no daemon."""

    LENS = (
        "---\nname: currency-board\ndescription: >-\n  Where every thread left off.\n"
        "scope: >-\n  every container head.\ntrigger: every 1d\ntarget: >-\n"
        "  view — overwrite View - Board.md\n---\n\n# currency-board\n"
    )

    def _vault_with_a_due_lens(self, tmp_path: Path) -> Path:
        lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
        lens_dir.mkdir(parents=True)
        (lens_dir / "currency-board.md").write_text(self.LENS, encoding="utf-8")
        return tmp_path

    def test_an_open_gate_writes_the_due_task(self, tmp_path: Path):
        vault = self._vault_with_a_due_lens(tmp_path)

        _drain_tick(vault, str(tmp_path / "test.db"), {"tasks": {"schedule_lenses": True}})

        (task,) = (vault / "OpenAugi" / "Tasks").glob("TASK-*-currency-board.md")
        assert "apply lens currency-board" in task.read_text(encoding="utf-8")

    def test_the_gate_is_off_by_default(self, tmp_path: Path):
        vault = self._vault_with_a_due_lens(tmp_path)

        _drain_tick(vault, str(tmp_path / "test.db"), {})

        assert not (vault / "OpenAugi" / "Tasks").exists()

    def test_a_second_tick_does_not_write_a_second_task(self, tmp_path: Path):
        vault = self._vault_with_a_due_lens(tmp_path)
        config = {"tasks": {"schedule_lenses": True}}
        _drain_tick(vault, str(tmp_path / "test.db"), config)

        _drain_tick(vault, str(tmp_path / "test.db"), config)

        assert len(list((vault / "OpenAugi" / "Tasks").glob("TASK-*.md"))) == 1

    def test_the_tick_leaves_a_heartbeat(self, tmp_path: Path):
        from openaugi.pipeline.heartbeat import read_heartbeat

        vault = self._vault_with_a_due_lens(tmp_path)
        _drain_tick(vault, str(tmp_path / "test.db"), {})

        beat = read_heartbeat(vault)
        assert beat is not None
        assert [row["name"] for row in beat["lenses"]] == ["currency-board"]

    def test_a_failing_heartbeat_does_not_take_the_tick_down(self, tmp_path: Path):
        vault = self._vault_with_a_due_lens(tmp_path)
        with patch(
            "openaugi.pipeline.heartbeat.write_heartbeat", side_effect=Exception("disk full")
        ):
            _drain_tick(vault, str(tmp_path / "test.db"), {"tasks": {"schedule_lenses": True}})

        (task,) = (vault / "OpenAugi" / "Tasks").glob("TASK-*-currency-board.md")
        assert task.exists()

    def test_a_failing_scheduler_does_not_take_the_tick_down(self, tmp_path: Path):
        vault = self._vault_with_a_due_lens(tmp_path)
        with patch(
            "openaugi.pipeline.schedule.run_due_lenses", side_effect=Exception("registry on fire")
        ):
            _drain_tick(vault, str(tmp_path / "test.db"), {"tasks": {"schedule_lenses": True}})

        assert not (vault / "OpenAugi" / "Tasks").exists()
