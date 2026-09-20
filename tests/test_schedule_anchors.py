"""`at:` / `on:` — a cadence that names a local time, not a UTC interval.

Every test freezes the clock and passes a fake zone. The zone is a real
`ZoneInfo`, because the point of the anchor is what happens on the far side
of a DST change, and a fixed offset cannot have one.
"""

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from openaugi.pipeline.schedule import (
    Anchor,
    MalformedAnchor,
    anchor_slot,
    describe_anchor,
    due_lenses,
    last_run,
    next_anchor,
    parse_anchor,
    record_run,
    run_due_lenses,
    schedule_timezone,
)
from tests.test_schedule import write_lens

NY = ZoneInfo("America/New_York")
CONFIG = {"tasks": {"schedule_lenses": True, "timezone": "America/New_York"}}


def local(y, m, d, hh, mm=0):
    """A wall-clock moment in the fake zone."""
    return datetime(y, m, d, hh, mm, tzinfo=NY)


BOARD_RUN = """
## Run

Read `OpenAugi/Board/.board-state.json` before building.

at: 06:00          # local wall clock, honored across DST
dedupe: OpenAugi/Board/{date} - Board.md
"""

BATCH_RUN = """
## Run

at: 06:30
on: Fri            # weekday pin, only meaningful with `every 7d`
"""


class TestParseAnchor:
    def test_no_anchor_lines_is_no_anchor(self):
        assert parse_anchor("Read the state first.\n\ndedupe: x/{date}.md") is None

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("at: 06:00", Anchor(at=time(6, 0))),
            ("at: 6:05", Anchor(at=time(6, 5))),
            ("- at: 22:30", Anchor(at=time(22, 30))),
            ("at: 06:00   # local wall clock", Anchor(at=time(6, 0))),
            ("on: Fri", Anchor(on=4)),
            ("on: friday", Anchor(on=4)),
            ("on: SUNDAY", Anchor(on=6)),
            ("at: 06:30\non: Fri", Anchor(at=time(6, 30), on=4)),
            ("prose first\n\nat: 05:50\n\ndedupe: a/{date}.md", Anchor(at=time(5, 50))),
        ],
    )
    def test_anchor_lines_parse(self, text, expected):
        assert parse_anchor(text) == expected

    @pytest.mark.parametrize(
        "text", ["at: 6am", "at: 24:00", "at: 06:60", "at:", "on: Funday", "on: F", "on:"]
    )
    def test_malformed_lines_raise(self, text):
        with pytest.raises(MalformedAnchor):
            parse_anchor(text)

    def test_one_bad_line_spoils_the_pair(self):
        with pytest.raises(MalformedAnchor):
            parse_anchor("at: 06:00\non: Funday")

    def test_a_weekday_alone_anchors_at_midnight(self):
        assert Anchor(on=4).clock == time(0, 0)

    @pytest.mark.parametrize(
        "anchor,text",
        [
            (None, ""),
            (Anchor(at=time(6)), "06:00"),
            (Anchor(on=6), "Sun"),
            (Anchor(at=time(6, 30), on=4), "06:30 Fri"),
        ],
    )
    def test_describe_anchor_is_the_one_line_form(self, anchor, text):
        assert describe_anchor(anchor) == text


class TestTimezone:
    def test_config_names_the_zone(self):
        assert schedule_timezone(CONFIG) is NY or str(schedule_timezone(CONFIG)) == str(NY)

    def test_an_unknown_zone_is_logged_and_falls_back(self, caplog):
        with caplog.at_level("WARNING"):
            tz = schedule_timezone({"tasks": {"timezone": "Mars/Olympus_Mons"}})
        assert tz is not None
        assert "Mars/Olympus_Mons" in caplog.text

    def test_no_config_means_the_system_zone(self):
        assert schedule_timezone({}) is not None


class TestAnchorSlot:
    def test_past_todays_anchor_is_today(self):
        assert anchor_slot(Anchor(at=time(6)), local(2026, 9, 14, 8), NY) == local(2026, 9, 14, 6)

    def test_before_todays_anchor_is_yesterday(self):
        assert anchor_slot(Anchor(at=time(6)), local(2026, 9, 14, 5, 59), NY) == local(
            2026, 9, 13, 6
        )

    def test_a_weekday_pin_walks_back_to_that_day(self):
        # 2026-09-20 is a Sunday; the most recent Friday 06:30 is the 18th.
        assert anchor_slot(Anchor(at=time(6, 30), on=4), local(2026, 9, 20, 8), NY) == local(
            2026, 9, 18, 6, 30
        )

    def test_on_the_pinned_day_before_the_hour_is_last_week(self):
        assert anchor_slot(Anchor(at=time(6, 30), on=4), local(2026, 9, 18, 6, 29), NY) == local(
            2026, 9, 11, 6, 30
        )

    def test_the_slot_is_local_whatever_zone_now_is_in(self):
        now = local(2026, 9, 14, 8).astimezone(UTC)
        assert anchor_slot(Anchor(at=time(6)), now, NY) == local(2026, 9, 14, 6)

    def test_next_anchor_counts_local_days(self):
        # 2026-10-31 06:00 EDT → 2026-11-01 06:00 EST is 25 hours and one day.
        nxt = next_anchor(Anchor(at=time(6)), local(2026, 10, 31, 6), timedelta(days=1), NY)
        assert nxt == local(2026, 11, 1, 6)
        # Subtract in UTC: two datetimes sharing one ZoneInfo subtract naively.
        assert nxt.astimezone(UTC) - local(2026, 10, 31, 6).astimezone(UTC) == timedelta(hours=25)

    def test_next_anchor_honours_the_weekday(self):
        nxt = next_anchor(
            Anchor(at=time(6, 30), on=4), local(2026, 9, 18, 6, 30), timedelta(days=7), NY
        )
        assert nxt == local(2026, 9, 25, 6, 30)


class TestDueWithAnchor:
    def test_before_the_anchor_is_not_due_even_if_the_period_elapsed(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        record_run(store, "currency-board", local(2026, 9, 13, 6), "t.md")
        # 25 hours later, but 05:59 local — the day's anchor has not come.
        assert due_lenses(tmp_path, store, local(2026, 9, 14, 5, 59), tz=NY) == []

    def test_at_the_anchor_is_due(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        record_run(store, "currency-board", local(2026, 9, 13, 6), "t.md")
        (spec,) = due_lenses(tmp_path, store, local(2026, 9, 14, 6), tz=NY)
        assert spec["slot"] == local(2026, 9, 14, 6)

    def test_already_ran_this_slot_is_not_due(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        record_run(store, "currency-board", local(2026, 9, 14, 6), "t.md")
        assert due_lenses(tmp_path, store, local(2026, 9, 14, 9), tz=NY) == []

    def test_a_never_run_anchored_lens_waits_for_its_first_anchor(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        # Nothing in the ledger: due, but for the most recent anchor, so the
        # grid starts on the clock and not on the tick.
        (spec,) = due_lenses(tmp_path, store, local(2026, 9, 14, 8), tz=NY)
        assert spec["slot"] == local(2026, 9, 14, 6)

    def test_a_run_forced_40_minutes_late_does_not_move_the_fire_time(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        record_run(store, "currency-board", local(2026, 9, 13, 6), "t.md")

        (path,) = run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 14, 6, 40))

        assert path.exists()
        assert last_run(store, "currency-board") == local(2026, 9, 14, 6)
        assert run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 15, 5, 59)) == []
        assert len(run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 15, 6))) == 1

    def test_the_board_still_fires_at_0600_local_after_dst(self, tmp_path, store):
        """2026-11-01: EDT → EST. A 10:00Z interval would become 05:00 local."""
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        record_run(store, "currency-board", local(2026, 10, 31, 6), "t.md")  # 10:00Z

        # The morning of the change: 06:00 EST is 11:00Z, 25 hours on.
        assert run_due_lenses(tmp_path, store, CONFIG, local(2026, 11, 1, 5, 59)) == []
        (path,) = run_due_lenses(tmp_path, store, CONFIG, local(2026, 11, 1, 6))
        assert last_run(store, "currency-board") == local(2026, 11, 1, 6)
        assert last_run(store, "currency-board").astimezone(UTC).hour == 11

        # The day after: still 06:00 local, not 05:00.
        assert run_due_lenses(tmp_path, store, CONFIG, local(2026, 11, 2, 5, 0)) == []
        assert run_due_lenses(tmp_path, store, CONFIG, local(2026, 11, 2, 5, 59)) == []
        (path,) = run_due_lenses(tmp_path, store, CONFIG, local(2026, 11, 2, 6))
        assert path.name == "TASK-2026-11-02-currency-board.md"

    def test_spring_forward_is_still_one_day(self, tmp_path, store):
        """2027-03-14: EST → EDT. 23 hours pass, and the board is still due."""
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        record_run(store, "currency-board", local(2027, 3, 13, 6), "t.md")
        (spec,) = due_lenses(tmp_path, store, local(2027, 3, 14, 6), tz=NY)
        elapsed = spec["slot"].astimezone(UTC) - local(2027, 3, 13, 6).astimezone(UTC)
        assert elapsed == timedelta(hours=23)

    def test_substack_batch_still_fires_on_a_friday_after_a_three_day_sleep(self, tmp_path, store):
        """One catch-up run when the machine wakes; the grid stays on Friday."""
        write_lens(tmp_path, "substack-batch", trigger="every 7d", run=BATCH_RUN)
        record_run(store, "substack-batch", local(2026, 9, 11, 6, 30), "t.md")  # a Friday

        # Asleep from Thursday the 17th, awake Sunday the 20th at 08:00.
        (path,) = run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 20, 8))
        assert last_run(store, "substack-batch") == local(2026, 9, 18, 6, 30)  # the Friday
        assert run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 20, 8, 5)) == []

        # Not Sunday, not Thursday: Friday, at 06:30.
        assert run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 24, 8)) == []
        assert run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 25, 6, 29)) == []
        (path,) = run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 25, 6, 30))
        assert path.name == "TASK-2026-09-25-substack-batch.md"
        assert last_run(store, "substack-batch").weekday() == 4

    def test_a_long_sleep_produces_one_run_per_lens_not_one_per_missed_slot(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        record_run(store, "currency-board", local(2026, 9, 13, 6), "t.md")

        written = run_due_lenses(tmp_path, store, CONFIG, local(2026, 9, 16, 9))

        assert len(written) == 1
        assert last_run(store, "currency-board") == local(2026, 9, 16, 6)

    def test_habit_parse_is_ordered_before_the_board(self, tmp_path, store):
        """05:50 precedes 06:00: alone on its own tick, and first on a catch-up."""
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        write_lens(tmp_path, "habit-parse", trigger="every 1d", run="\n## Run\n\nat: 05:50\n")
        record_run(store, "currency-board", local(2026, 9, 13, 6), "t.md")
        record_run(store, "habit-parse", local(2026, 9, 13, 5, 50), "t.md")

        # A normal morning: the tick at 05:52 sees only the habit parse.
        assert [
            s["name"] for s in due_lenses(tmp_path, store, local(2026, 9, 14, 5, 52), tz=NY)
        ] == ["habit-parse"]
        # A catch-up tick sees both — habit-parse first, though "c" < "h".
        assert [s["name"] for s in due_lenses(tmp_path, store, local(2026, 9, 14, 8), tz=NY)] == [
            "habit-parse",
            "currency-board",
        ]

    def test_a_malformed_anchor_falls_back_to_the_interval(self, tmp_path, store, caplog):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run="\n## Run\n\nat: 6am\n")
        # A late run, stamped verbatim at 10:00. An anchored lens would be due
        # at 09:00 the next day (past 06:00, one local day on); the plain
        # interval it falls back to is not due until 10:00.
        record_run(store, "currency-board", local(2026, 9, 13, 10), "t.md")
        with caplog.at_level("WARNING"):
            assert due_lenses(tmp_path, store, local(2026, 9, 14, 9), tz=NY) == []
            assert len(due_lenses(tmp_path, store, local(2026, 9, 14, 10), tz=NY)) == 1
        assert "currency-board" in caplog.text and "6am" in caplog.text

    def test_an_anchor_on_a_sub_day_cadence_is_ignored_with_a_warning(
        self, tmp_path, store, caplog
    ):
        write_lens(tmp_path, "pulse", trigger="every 12h", run="\n## Run\n\nat: 06:00\n")
        record_run(store, "pulse", local(2026, 9, 13, 6), "t.md")
        with caplog.at_level("WARNING"):
            assert len(due_lenses(tmp_path, store, local(2026, 9, 13, 18), tz=NY)) == 1
        assert "pulse" in caplog.text

    def test_the_run_prose_still_carries_the_anchor_lines(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d", run=BOARD_RUN)
        (spec,) = due_lenses(tmp_path, store, local(2026, 9, 14, 8), tz=NY)
        assert "at: 06:00" in spec["run"]
        assert spec["dedupe"] == "OpenAugi/Board/{date} - Board.md"
