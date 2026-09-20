"""`openaugi doctor` — the terminal answer to "is anything actually running?"

The clock is always passed in; the lock directory is always a tmp_path. The
CLI test drives the real command end to end against a scratch vault and
database, because the exit code is the contract a janitor fires on.
"""

from datetime import timedelta
from unittest.mock import patch

from typer.testing import CliRunner

from openaugi.cli.main import app
from openaugi.doctor import RESTART, diagnose, render
from openaugi.pipeline.heartbeat import STALE_AFTER, write_heartbeat
from openaugi.pipeline.schedule import record_run
from openaugi.service_version import SERVICE_STATE_COLLECTION, UP_RECORD_ID
from openaugi.singleton import acquire
from openaugi.store.sqlite import SQLiteStore
from tests.test_schedule import NOW, write_lens

runner = CliRunner()


class TestDiagnose:
    def test_no_heartbeat_is_stale_and_unhealthy(self, tmp_path, store):
        report = diagnose(tmp_path, store, NOW, lock_dir=tmp_path)
        assert report["last_tick"] is None
        assert report["stale"] is True
        assert report["healthy"] is False
        assert report["watcher_pid"] is None

    def test_a_fresh_tick_is_healthy(self, tmp_path, store):
        write_heartbeat(tmp_path, store, NOW, pid=7)
        report = diagnose(tmp_path, store, NOW + timedelta(minutes=3), lock_dir=tmp_path)
        assert report["tick_age"] == timedelta(minutes=3)
        assert report["stale"] is False
        assert report["healthy"] is True
        assert report["tick_pid"] == 7

    def test_a_tick_older_than_ten_minutes_is_stale(self, tmp_path, store):
        write_heartbeat(tmp_path, store, NOW, pid=7)
        assert timedelta(minutes=10) == STALE_AFTER
        fine = diagnose(tmp_path, store, NOW + timedelta(minutes=10), lock_dir=tmp_path)
        late = diagnose(tmp_path, store, NOW + timedelta(minutes=11), lock_dir=tmp_path)
        assert fine["healthy"] is True
        assert late["healthy"] is False

    def test_the_watcher_pid_comes_from_the_held_lock(self, tmp_path, store):
        acquire("up", tmp_path)
        import os

        report = diagnose(tmp_path, store, NOW, lock_dir=tmp_path)
        assert report["watcher_pid"] == str(os.getpid())

    def test_a_released_lock_reads_as_not_running(self, tmp_path, store):
        (tmp_path / "up.lock").write_text("12345")  # left behind by a dead process
        assert diagnose(tmp_path, store, NOW, lock_dir=tmp_path)["watcher_pid"] is None

    def test_every_scheduled_lens_is_reported(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        write_lens(tmp_path, "substack-batch", trigger="every 7d")
        record_run(store, "currency-board", NOW - timedelta(hours=1), "t.md")

        report = diagnose(tmp_path, store, NOW, lock_dir=tmp_path)

        by_name = {row["name"]: row for row in report["lenses"]}
        assert by_name["currency-board"]["next_due"] == NOW + timedelta(hours=23)
        assert by_name["substack-batch"]["last_run"] is None

    def test_code_drift_is_carried_through(self, tmp_path, store):
        store.write_record(
            SERVICE_STATE_COLLECTION,
            UP_RECORD_ID,
            {"sha": "0" * 40, "pid": 1, "started_at": NOW.isoformat()},
            NOW.isoformat(),
        )
        with patch("openaugi.service_version.head_sha", return_value="f" * 40):
            report = diagnose(tmp_path, store, NOW, lock_dir=tmp_path)
        assert report["drift"]["running_sha"] == "0" * 40


class TestRender:
    def test_a_stale_report_names_the_restart(self, tmp_path, store):
        text = render(diagnose(tmp_path, store, NOW, lock_dir=tmp_path))
        assert "NOT RUNNING" in text
        assert "never" in text
        assert RESTART in text

    def test_a_healthy_report_is_quiet(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        write_heartbeat(tmp_path, store, NOW, pid=7)
        text = render(diagnose(tmp_path, store, NOW + timedelta(minutes=1), lock_dir=tmp_path))
        assert "1m ago) ok" in text
        assert RESTART not in text
        assert "currency-board" in text

    def test_an_overdue_lens_is_called_out(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        record_run(store, "currency-board", NOW - timedelta(days=3), "t.md")
        text = render(diagnose(tmp_path, store, NOW, lock_dir=tmp_path))
        assert "OVERDUE" in text


class TestCommand:
    def _db(self, tmp_path):
        db = tmp_path / "test.db"
        SQLiteStore(db).close()
        return db

    def test_exits_non_zero_when_the_tick_is_stale(self, tmp_path):
        db = self._db(tmp_path)
        result = runner.invoke(app, ["doctor", "--path", str(tmp_path), "--db", str(db)])
        assert result.exit_code == 1
        assert "never" in result.output

    def test_exits_zero_when_the_tick_is_fresh(self, tmp_path):
        db = self._db(tmp_path)
        store = SQLiteStore(db)
        write_heartbeat(tmp_path, store, pid=7)
        store.close()
        result = runner.invoke(app, ["doctor", "--path", str(tmp_path), "--db", str(db)])
        assert result.exit_code == 0, result.output
        assert "tick:" in result.output

    def test_a_missing_database_is_an_error(self, tmp_path):
        result = runner.invoke(
            app, ["doctor", "--path", str(tmp_path), "--db", str(tmp_path / "nope.db")]
        )
        assert result.exit_code == 1
