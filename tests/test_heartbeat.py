"""The heartbeat view — written by the tick, read by whatever notices silence.

Every test drives the clock explicitly. The file is the only state, so a
fake `now` is all a throttle test needs.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from openaugi.adapters.vault import HEARTBEAT_VIEW, SYSTEM_EXCLUDE_PATTERNS, parse_vault
from openaugi.pipeline.heartbeat import (
    HEARTBEAT_INTERVAL,
    STALE_AFTER,
    is_stale,
    read_heartbeat,
    tick_age,
    write_heartbeat,
)
from openaugi.pipeline.schedule import lens_status, record_run
from openaugi.service_version import SERVICE_STATE_COLLECTION, UP_RECORD_ID
from tests.test_schedule import NOW, write_lens


class TestWrite:
    def test_the_first_tick_writes_the_view(self, tmp_path, store):
        path = write_heartbeat(tmp_path, store, NOW, pid=4242)
        assert path == tmp_path / HEARTBEAT_VIEW
        beat = read_heartbeat(tmp_path)
        assert beat["last_tick"] == NOW
        assert beat["pid"] == 4242

    def test_the_running_commit_is_the_one_up_stamped(self, tmp_path, store):
        store.write_record(
            SERVICE_STATE_COLLECTION, UP_RECORD_ID, {"sha": "abc123", "pid": 1}, NOW.isoformat()
        )
        write_heartbeat(tmp_path, store, NOW, pid=1)
        assert read_heartbeat(tmp_path)["commit"] == "abc123"

    def test_a_tick_inside_the_interval_is_throttled(self, tmp_path, store):
        write_heartbeat(tmp_path, store, NOW, pid=1)
        assert write_heartbeat(tmp_path, store, NOW + timedelta(minutes=4), pid=1) is None
        assert read_heartbeat(tmp_path)["last_tick"] == NOW

    def test_a_tick_past_the_interval_rewrites(self, tmp_path, store):
        write_heartbeat(tmp_path, store, NOW, pid=1)
        later = NOW + HEARTBEAT_INTERVAL
        assert write_heartbeat(tmp_path, store, later, pid=1) is not None
        assert read_heartbeat(tmp_path)["last_tick"] == later

    def test_force_ignores_the_throttle(self, tmp_path, store):
        write_heartbeat(tmp_path, store, NOW, pid=1)
        assert write_heartbeat(tmp_path, store, NOW + timedelta(seconds=1), pid=1, force=True)

    def test_a_clock_set_back_still_writes(self, tmp_path, store):
        # A file from the "future" must not silence the heartbeat for hours.
        write_heartbeat(tmp_path, store, NOW + timedelta(hours=2), pid=1)
        assert write_heartbeat(tmp_path, store, NOW, pid=1) is not None

    def test_each_scheduled_lens_is_a_row_in_frontmatter_and_body(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        write_lens(tmp_path, "substack-batch", trigger="every 7d")
        write_lens(tmp_path, "distill", trigger="on-demand")
        record_run(store, "currency-board", NOW - timedelta(hours=6), "t.md")

        text = write_heartbeat(tmp_path, store, NOW, pid=1).read_text(encoding="utf-8")
        beat = read_heartbeat(tmp_path)

        names = [row["name"] for row in beat["lenses"]]
        assert names == ["currency-board", "substack-batch"]
        board = beat["lenses"][0]
        assert board["period_seconds"] == 86400
        assert board["overdue"] is False
        assert board["next_due"] == NOW + timedelta(hours=18)  # YAML reads ISO stamps as dates
        assert "| currency-board | `every 1d` |" in text
        assert "distill" not in text

    def test_an_anchored_lens_shows_its_anchor(self, tmp_path, store):
        write_lens(
            tmp_path, "substack-batch", trigger="every 7d", run="\n## Run\n\nat: 06:30\non: Fri\n"
        )
        text = write_heartbeat(tmp_path, store, NOW, pid=1).read_text(encoding="utf-8")
        (row,) = read_heartbeat(tmp_path)["lenses"]
        assert row["anchor"] == "06:30 Fri"  # a string, not YAML's sexagesimal 390
        assert "| substack-batch | `every 7d` | 06:30 Fri |" in text

    def test_an_overdue_lens_is_flagged(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        record_run(store, "currency-board", NOW - timedelta(days=3), "t.md")

        text = write_heartbeat(tmp_path, store, NOW, pid=1).read_text(encoding="utf-8")

        assert read_heartbeat(tmp_path)["lenses"][0]["overdue"] is True
        assert "**overdue**" in text


class TestRead:
    def test_no_file_is_no_heartbeat(self, tmp_path):
        assert read_heartbeat(tmp_path) is None
        assert tick_age(tmp_path, NOW) is None
        assert is_stale(tmp_path, NOW) is True

    def test_age_and_staleness_follow_the_clock(self, tmp_path, store):
        write_heartbeat(tmp_path, store, NOW, pid=1)
        assert tick_age(tmp_path, NOW + timedelta(minutes=3)) == timedelta(minutes=3)
        assert is_stale(tmp_path, NOW + STALE_AFTER) is False
        assert is_stale(tmp_path, NOW + STALE_AFTER + timedelta(seconds=1)) is True

    def test_an_unreadable_stamp_reads_as_absent(self, tmp_path, caplog):
        path = tmp_path / HEARTBEAT_VIEW
        path.parent.mkdir(parents=True)
        path.write_text("---\nlast_tick: whenever\n---\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert read_heartbeat(tmp_path) is None
        assert is_stale(tmp_path, NOW) is True

    def test_a_file_without_frontmatter_reads_as_absent(self, tmp_path):
        path = tmp_path / HEARTBEAT_VIEW
        path.parent.mkdir(parents=True)
        path.write_text("# nothing here\n", encoding="utf-8")
        assert read_heartbeat(tmp_path) is None


class TestNeverIngested:
    """The write must not re-trigger the ingest that performs it."""

    def test_the_view_is_a_system_exclude(self):
        assert HEARTBEAT_VIEW in SYSTEM_EXCLUDE_PATTERNS

    def test_parse_vault_skips_it_whatever_config_says(self, tmp_path, store):
        note = tmp_path / "note.md"
        note.write_text("# A note\n\nSome thought.\n", encoding="utf-8")
        write_heartbeat(tmp_path, store, NOW, pid=1)

        # A config that names its own excludes replaces the defaults wholesale;
        # the heartbeat has to survive that.
        blocks, _ = parse_vault(tmp_path, exclude_patterns=[".git/**"])

        sources = {b.metadata.get("source_path", "") for b in blocks}
        assert any(s.endswith("note.md") for s in sources)
        assert not any("Heartbeat" in s for s in sources)


class TestLensStatus:
    def test_a_never_run_lens_is_due_now_and_not_overdue(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        (row,) = lens_status(tmp_path, store, NOW)
        assert row["last_run"] is None
        assert row["next_due"] == NOW
        assert row["overdue"] is False

    def test_unschedulable_lenses_are_not_rows(self, tmp_path, store):
        write_lens(tmp_path, "distill", trigger="on-demand")
        write_lens(tmp_path, "broken", trigger='"every: fortnight"')
        assert lens_status(tmp_path, store, NOW) == []

    def test_overdue_means_a_whole_missed_period(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        record_run(store, "currency-board", NOW - timedelta(days=2), "t.md")
        assert lens_status(tmp_path, store, NOW)[0]["overdue"] is False
        record_run(store, "currency-board", NOW - timedelta(days=2, minutes=1), "t.md")
        assert lens_status(tmp_path, store, NOW)[0]["overdue"] is True


def test_heartbeat_path_is_under_views(tmp_path: Path):
    assert Path(HEARTBEAT_VIEW).parts[:2] == ("OpenAugi", "Views")


def test_now_defaults_to_utc(tmp_path, store):
    write_heartbeat(tmp_path, store, pid=1)
    tick = read_heartbeat(tmp_path)["last_tick"]
    assert tick.tzinfo is not None
    assert abs(datetime.now(UTC) - tick) < timedelta(minutes=1)
