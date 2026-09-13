"""The trigger field, made real — periods, due-ness, dedupe, and the task file.

The fixture vault is built per test so each one states the whole world it
depends on: a registry of lens files, a records store, and a fixed `now`.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from openaugi.pipeline.schedule import (
    LENS_SCHEDULE_COLLECTION,
    build_lens_task,
    due_lenses,
    expand,
    last_run,
    parse_period,
    read_run_section,
    record_run,
    run_due_lenses,
    scheduling_enabled,
    task_filename,
    write_lens_task,
)

NOW = datetime(2026, 9, 13, 6, 0, tzinfo=UTC)

LENS = """---
name: {name}
description: >-
  {description}
scope: >-
  {scope}
trigger: {trigger}
target: >-
  view — overwrite View - Board.md
---

# {name}

## Intent

The question this lens answers.
{run}"""

RUN_SECTION = """
## Run

Read `OpenAugi/Board/.board-state.json` before building.

dedupe: OpenAugi/Board/{date} - Board.md
"""


def write_lens(
    vault: Path,
    name: str,
    *,
    trigger: str = "every 1d",
    description: str = "Where every thread left off.",
    scope: str = "every container head, plus the last 7 days.",
    run: str = "",
) -> Path:
    lens_dir = vault / "OpenAugi" / "AGENT" / "lenses"
    lens_dir.mkdir(parents=True, exist_ok=True)
    path = lens_dir / f"{name}.md"
    path.write_text(
        LENS.format(name=name, description=description, scope=scope, trigger=trigger, run=run),
        encoding="utf-8",
    )
    return path


class TestParsePeriod:
    @pytest.mark.parametrize(
        "trigger,expected",
        [
            ("every 1d", timedelta(days=1)),
            ("every 7d", timedelta(days=7)),
            ("every: 7d", timedelta(days=7)),  # the quoted-colon form the contract allows
            ("every 12h", timedelta(hours=12)),
            ("every 30m", timedelta(minutes=30)),
            ("every 2w", timedelta(weeks=2)),
            ("EVERY 1D", timedelta(days=1)),
            ("  every 1d  ", timedelta(days=1)),
        ],
    )
    def test_cadences_parse(self, trigger, expected):
        assert parse_period(trigger) == expected

    @pytest.mark.parametrize(
        "trigger",
        ["on-demand", "on-pass", "", "weekly", "every day", "every 1y", "every 0d", "every -1d"],
    )
    def test_everything_else_is_not_a_cadence(self, trigger):
        assert parse_period(trigger) is None


class TestRunSection:
    def test_absent_section_reads_as_nothing(self, tmp_path):
        write_lens(tmp_path, "nuggets")
        assert read_run_section(tmp_path, {"name": "nuggets", "file": "nuggets.md"}) == ("", None)

    def test_prose_and_dedupe_are_both_read(self, tmp_path):
        write_lens(tmp_path, "currency-board", run=RUN_SECTION)
        run, dedupe = read_run_section(
            tmp_path, {"name": "currency-board", "file": "currency-board.md"}
        )
        assert ".board-state.json" in run
        assert dedupe == "OpenAugi/Board/{date} - Board.md"

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert read_run_section(tmp_path, {"name": "gone", "file": "gone.md"}) == ("", None)

    def test_expand_substitutes_the_run_date(self):
        assert (
            expand("OpenAugi/Board/{date} - Board.md", NOW)
            == "OpenAugi/Board/2026-09-13 - Board.md"
        )

    def test_task_filename_is_deterministic_per_day(self):
        assert task_filename("currency-board", NOW) == "TASK-2026-09-13-currency-board.md"


class TestDueLenses:
    def test_a_never_run_cadence_is_due(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        assert [s["name"] for s in due_lenses(tmp_path, store, NOW)] == ["currency-board"]

    def test_within_the_period_is_not_due(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        record_run(store, "currency-board", NOW - timedelta(hours=6), "t.md")
        assert due_lenses(tmp_path, store, NOW) == []

    def test_past_the_period_is_due_again(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        record_run(store, "currency-board", NOW - timedelta(days=1, minutes=1), "t.md")
        assert [s["name"] for s in due_lenses(tmp_path, store, NOW)] == ["currency-board"]

    @pytest.mark.parametrize("trigger", ["on-demand", "on-pass"])
    def test_unscheduled_triggers_never_fire(self, tmp_path, store, trigger):
        write_lens(tmp_path, "distill", trigger=trigger)
        assert due_lenses(tmp_path, store, NOW) == []

    def test_a_malformed_trigger_is_skipped_and_logged(self, tmp_path, store, caplog):
        write_lens(tmp_path, "substack-batch", trigger='"every: fortnight"')
        with caplog.at_level("WARNING"):
            assert due_lenses(tmp_path, store, NOW) == []
        assert "substack-batch" in caplog.text

    def test_a_contract_violation_is_skipped_not_guessed_at(self, tmp_path, store, caplog):
        lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
        lens_dir.mkdir(parents=True)
        (lens_dir / "half.md").write_text(
            "---\nname: half\ndescription: >-\n  No scope, no target.\ntrigger: every 1d\n---\n"
        )
        with caplog.at_level("WARNING"):
            assert due_lenses(tmp_path, store, NOW) == []
        assert "half" in caplog.text

    def test_an_existing_task_file_for_today_holds_it_back(self, tmp_path, store):
        write_lens(tmp_path, "currency-board")
        tasks = tmp_path / "OpenAugi" / "Tasks"
        tasks.mkdir(parents=True)
        (tasks / task_filename("currency-board", NOW)).write_text("already queued")
        assert due_lenses(tmp_path, store, NOW) == []

    def test_an_existing_dedupe_output_holds_it_back(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", run=RUN_SECTION)
        board = tmp_path / "OpenAugi" / "Board" / "2026-09-13 - Board.md"
        board.parent.mkdir(parents=True)
        board.write_text("today's board, built by hand")
        assert due_lenses(tmp_path, store, NOW) == []

    def test_yesterdays_output_does_not_hold_it_back(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", run=RUN_SECTION)
        board = tmp_path / "OpenAugi" / "Board" / "2026-09-12 - Board.md"
        board.parent.mkdir(parents=True)
        board.write_text("yesterday's board")
        assert [s["name"] for s in due_lenses(tmp_path, store, NOW)] == ["currency-board"]

    def test_an_empty_registry_is_not_an_error(self, tmp_path, store):
        assert due_lenses(tmp_path, store, NOW) == []

    def test_due_specs_carry_what_the_task_file_needs(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", run=RUN_SECTION)
        (spec,) = due_lenses(tmp_path, store, NOW)
        assert spec["period"] == timedelta(days=1)
        assert spec["scope"].startswith("every container head")
        assert ".board-state.json" in spec["run"]


class TestLedger:
    def test_a_lens_that_never_ran_has_no_last_run(self, store):
        assert last_run(store, "currency-board") is None

    def test_a_run_is_recorded_once_per_lens(self, store):
        record_run(store, "currency-board", NOW - timedelta(days=1), "old.md")
        record_run(store, "currency-board", NOW, "new.md")
        rows = store.list_records(LENS_SCHEDULE_COLLECTION)
        assert len(rows) == 1
        assert rows[0]["task"] == "new.md"
        assert last_run(store, "currency-board") == NOW

    def test_an_unreadable_stamp_reads_as_never_run(self, store, caplog):
        store.write_record(
            LENS_SCHEDULE_COLLECTION,
            "currency-board",
            {"lens": "currency-board", "last_run": "whenever"},
            NOW.isoformat(),
        )
        with caplog.at_level("WARNING"):
            assert last_run(store, "currency-board") is None
        assert "currency-board" in caplog.text


class TestTaskFile:
    def _spec(self, tmp_path, **kw):
        write_lens(tmp_path, "currency-board", **kw)
        return {
            "name": "currency-board",
            "file": "currency-board.md",
            "description": "Where every thread left off.",
            "scope": "every container head, plus the last 7 days.",
            "trigger": "every 1d",
            "run": read_run_section(
                tmp_path, {"name": "currency-board", "file": "currency-board.md"}
            )[0],
        }

    def test_the_file_follows_the_task_contract(self, tmp_path):
        text = build_lens_task(self._spec(tmp_path, run=RUN_SECTION), tmp_path, NOW)
        assert text.startswith("---\nstatus: pending\n")
        assert f"working_dir: {tmp_path}" in text
        assert "lens: currency-board" in text
        for heading in (
            "## Context",
            "## User instruction",
            "## Task",
            "## Human Todo",
            "## Results",
        ):
            assert f"\n{heading}\n" in text

    def test_the_prompt_points_at_the_lens_and_carries_its_run_prose(self, tmp_path):
        text = build_lens_task(self._spec(tmp_path, run=RUN_SECTION), tmp_path, NOW)
        assert "> apply lens currency-board" in text
        assert "OpenAugi/AGENT/lenses/currency-board.md" in text
        assert "every container head" in text
        assert ".board-state.json" in text

    def test_a_lens_without_a_run_section_still_builds(self, tmp_path):
        text = build_lens_task(self._spec(tmp_path), tmp_path, NOW)
        assert "## Task" in text and "dedupe" not in text

    def test_write_creates_the_folder_and_names_the_file_by_day(self, tmp_path):
        path = write_lens_task(self._spec(tmp_path), tmp_path, NOW)
        assert path == tmp_path / "OpenAugi" / "Tasks" / "TASK-2026-09-13-currency-board.md"
        assert path.read_text(encoding="utf-8").startswith("---")

    def test_write_never_overwrites_an_existing_task(self, tmp_path):
        spec = self._spec(tmp_path)
        write_lens_task(spec, tmp_path, NOW)
        assert write_lens_task(spec, tmp_path, NOW) is None


class TestTheTick:
    def test_the_gate_is_off_by_default(self, tmp_path, store):
        write_lens(tmp_path, "currency-board")
        assert scheduling_enabled({}) is False
        assert run_due_lenses(tmp_path, store, {}, NOW) == []
        assert not (tmp_path / "OpenAugi" / "Tasks").exists()

    def test_an_open_gate_writes_the_task_and_records_the_run(self, tmp_path, store):
        write_lens(tmp_path, "currency-board")
        config = {"tasks": {"schedule_lenses": True}}

        (path,) = run_due_lenses(tmp_path, store, config, NOW)

        assert path.name == "TASK-2026-09-13-currency-board.md"
        assert last_run(store, "currency-board") == NOW

    def test_a_second_tick_the_same_day_writes_nothing(self, tmp_path, store):
        write_lens(tmp_path, "currency-board")
        config = {"tasks": {"schedule_lenses": True}}
        run_due_lenses(tmp_path, store, config, NOW)

        assert run_due_lenses(tmp_path, store, config, NOW + timedelta(minutes=5)) == []

    def test_the_next_day_writes_again(self, tmp_path, store):
        write_lens(tmp_path, "currency-board")
        config = {"tasks": {"schedule_lenses": True}}
        run_due_lenses(tmp_path, store, config, NOW)

        (path,) = run_due_lenses(tmp_path, store, config, NOW + timedelta(days=1))
        assert path.name == "TASK-2026-09-14-currency-board.md"

    def test_only_due_lenses_are_written(self, tmp_path, store):
        write_lens(tmp_path, "currency-board", trigger="every 1d")
        write_lens(tmp_path, "substack-batch", trigger="every 7d")
        write_lens(tmp_path, "distill", trigger="on-demand")
        record_run(store, "substack-batch", NOW - timedelta(days=2), "t.md")

        written = run_due_lenses(tmp_path, store, {"tasks": {"schedule_lenses": True}}, NOW)

        assert [p.name for p in written] == ["TASK-2026-09-13-currency-board.md"]
