"""Board janitor — checkbox answers become state the next board honors."""

from __future__ import annotations

import json

import pytest

from openaugi.pipeline.board_janitor import (
    PRUNE_DAYS,
    board_notes,
    decisions_from_log,
    declined_proposals,
    load_state,
    open_items,
    parse_items,
    parse_proposals,
    previous_board_summary,
    rebuild_state,
    retired_items,
    sync_board,
)

BOARD = """---
type: document
cssclasses:
  - board
---

# Board — 2026-09-02 (Wednesday)

> [!board-lane]+ Self · Money
> *Left off: sizing done, blocked on three portal numbers.*
>
> - **Pull the three numbers from the benefits portal** `quick · 5m`
>     - [ ] done
>     - [ ] not doing
>     - [ ] someday
>     <!-- item:insurance-numbers -->
>
> - **Stand up the Kalshi ingestion job** `build · 60m`
>     - [ ] done
>     - [ ] not doing
>     - [ ] someday
>     <!-- item:kalshi-ingestion -->

> [!board-judgment]+ Needs your judgment (1 of 3)
> - **Archive three finished sessions** — idle and complete; unarchive is one click
>     - [ ] done
>     - [ ] not doing
>     - [ ] someday
>     <!-- item:archive-sessions -->

> [!board-note]- Notes to augi
> *Anything that isn't a checkbox.*
> <!-- board-note -->
>
"""


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "OpenAugi" / "Board").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def board(vault):
    path = vault / "OpenAugi" / "Board" / "2026-09-02 - Board.md"
    path.write_text(BOARD, encoding="utf-8")
    return path


def _tick(path, label, key, reason=None):
    """Tick one box for the item whose marker follows it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = next(i for i, line in enumerate(lines) if f"item:{key}" in line)
    for i in range(marker - 1, max(marker - 8, -1), -1):
        if lines[i].strip().endswith(f"[ ] {label}"):
            lines[i] = lines[i].replace(f"[ ] {label}", f"[x] {label}")
            if reason:
                lines.insert(marker, f">     aaa: {reason}")
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parses_every_item_with_title_and_lane(board):
    items = parse_items(board.read_text(encoding="utf-8"))
    assert set(items) == {"insurance-numbers", "kalshi-ingestion", "archive-sessions"}
    assert items["insurance-numbers"]["title"].startswith("Pull the three numbers")
    assert items["insurance-numbers"]["lane"] == "Self · Money"
    assert items["archive-sessions"]["lane"] == "Needs your judgment (1 of 3)"
    assert all(item["answer"] is None for item in items.values())


def test_recording_is_idempotent_per_board_date(board, vault):
    sync_board(board, vault)
    sync_board(board, vault)
    sync_board(board, vault)
    items = load_state(vault)["items"]
    assert items["insurance-numbers"]["appearances"] == 1
    assert items["insurance-numbers"]["first_seen"] == "2026-09-02"
    assert load_state(vault)["last_run"] == "2026-09-02"


def test_age_accumulates_across_boards(board, vault):
    sync_board(board, vault)
    later = vault / "OpenAugi" / "Board" / "2026-09-03 - Board.md"
    later.write_text(BOARD.replace("2026-09-02", "2026-09-03"), encoding="utf-8")
    sync_board(later, vault)
    assert load_state(vault)["items"]["insurance-numbers"]["appearances"] == 2


def test_ticking_retires_the_item_and_stores_the_reason(board, vault):
    sync_board(board, vault)
    _tick(board, "not doing", "insurance-numbers", reason="waiting on open enrollment")
    assert sync_board(board, vault) == 1

    record = load_state(vault)["items"]["insurance-numbers"]
    assert record["state"] == "not-doing"
    assert record["reason"] == "waiting on open enrollment"
    assert record["answered_on"] == "2026-09-02"

    assert "insurance-numbers" not in open_items(vault)
    assert "insurance-numbers" in retired_items(vault)
    assert "kalshi-ingestion" in open_items(vault)


def test_answered_boxes_become_a_confirmation_and_are_not_reprocessed(board, vault):
    _tick(board, "done", "kalshi-ingestion")
    assert sync_board(board, vault) == 1

    text = board.read_text(encoding="utf-8")
    assert "- ✓ done" in text
    # The losing boxes for that item are gone; other items keep theirs.
    assert text.count("[ ] someday") == 2
    assert "[x]" not in text
    # A second pass finds nothing new to do.
    assert sync_board(board, vault) == 0


def test_feedback_is_appended_to_the_shared_stream(board, vault):
    _tick(board, "someday", "archive-sessions", reason="not this week")
    sync_board(board, vault)

    log = (vault / "OpenAugi" / "Capture" / "feedback-log.ndjson").read_text(encoding="utf-8")
    record = json.loads(log.strip().splitlines()[-1])
    assert record["source"] == "currency-board"
    assert record["item"] == "archive-sessions"
    assert record["signal"] == "someday"
    assert record["reason"] == "not this week"
    assert record["board"] == "2026-09-02"


def test_free_text_note_about_the_board_is_logged_and_marked_read(board, vault):
    text = board.read_text(encoding="utf-8")
    marker = text.index("<!-- board-note -->") + len("<!-- board-note -->")
    board.write_text(
        text[:marker]
        + '\n> the "work doc" item was too vague — no idea which doc\n'
        + text[marker:],
        encoding="utf-8",
    )
    sync_board(board, vault)

    record = json.loads(
        (vault / "OpenAugi" / "Capture" / "feedback-log.ndjson")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()[-1]
    )
    assert record["source"] == "currency-board-note"
    assert "too vague" in record["reason"]

    # Marked read, and a second pass does not log it again.
    assert "✓ noted 2026-09-02" in board.read_text(encoding="utf-8")
    before = (vault / "OpenAugi" / "Capture" / "feedback-log.ndjson").read_text(encoding="utf-8")
    sync_board(board, vault)
    assert (vault / "OpenAugi" / "Capture" / "feedback-log.ndjson").read_text(
        encoding="utf-8"
    ) == before


def test_empty_note_callout_logs_nothing(board, vault):
    sync_board(board, vault)
    log = vault / "OpenAugi" / "Capture" / "feedback-log.ndjson"
    assert not log.exists() or "currency-board-note" not in log.read_text(encoding="utf-8")


def test_missing_board_and_unreadable_state_are_survivable(vault):
    assert sync_board(vault / "OpenAugi" / "Board" / "nope.md", vault) == 0
    (vault / "OpenAugi" / "Board" / ".board-state.json").write_text("{{{", encoding="utf-8")
    assert load_state(vault) == {"last_run": None, "items": {}}


# --- state as a projection (2026-09-03) -------------------------------------
#
# `.board-state.json` used to be mutated in place by this module *and* by
# board-build agent sessions. The counters drifted to 3 on a two-day-old board
# and falsely tripped the "third board" staleness flag. State is now projected
# from the board notes plus the decision log, so it cannot drift.


def _second_board(vault, day="2026-09-03"):
    path = vault / "OpenAugi" / "Board" / f"{day} - Board.md"
    path.write_text(BOARD.replace("2026-09-02", day), encoding="utf-8")
    return path


def test_appearances_counts_boards_rather_than_incrementing(board, vault):
    sync_board(board, vault)
    assert load_state(vault)["items"]["kalshi-ingestion"]["appearances"] == 1

    second = _second_board(vault)
    sync_board(second, vault)
    item = load_state(vault)["items"]["kalshi-ingestion"]
    assert item["appearances"] == 2
    assert item["first_seen"] == "2026-09-02"
    assert item["last_seen"] == "2026-09-03"


def test_inflated_counter_is_repaired_by_the_next_rebuild(board, vault):
    """The exact corruption found on 2026-09-03: a counter bumped past truth."""
    sync_board(board, vault)
    state = load_state(vault)
    state["items"]["kalshi-ingestion"]["appearances"] = 3
    state["items"]["kalshi-ingestion"]["last_seen"] = "1999-01-01"
    (vault / "OpenAugi" / "Board" / ".board-state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    rebuild_state(vault)
    item = load_state(vault)["items"]["kalshi-ingestion"]
    assert item["appearances"] == 1
    assert item["last_seen"] == "2026-09-02"


def test_state_rebuilds_from_scratch_after_deletion(board, vault):
    _tick(board, "not doing", "kalshi-ingestion", reason="not until October")
    sync_board(board, vault)
    before = load_state(vault)["items"]

    (vault / "OpenAugi" / "Board" / ".board-state.json").unlink()
    rebuild_state(vault)

    assert load_state(vault)["items"] == before


def test_unknown_state_is_dropped_on_load(board, vault):
    sync_board(board, vault)
    path = vault / "OpenAugi" / "Board" / ".board-state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["items"]["drift-content-lane"] = {"state": "withdrawn", "title": "x"}
    path.write_text(json.dumps(state), encoding="utf-8")

    assert "drift-content-lane" not in load_state(vault)["items"]


def test_latest_log_entry_wins_when_he_changes_his_mind(board, vault):
    _tick(board, "not doing", "kalshi-ingestion")
    sync_board(board, vault)
    assert load_state(vault)["items"]["kalshi-ingestion"]["state"] == "not-doing"

    second = _second_board(vault)
    _tick(second, "done", "kalshi-ingestion")
    sync_board(second, vault)
    assert load_state(vault)["items"]["kalshi-ingestion"]["state"] == "done"


def test_decision_binds_even_when_its_board_note_is_deleted(board, vault):
    _tick(board, "not doing", "kalshi-ingestion", reason="not until October")
    sync_board(board, vault)
    board.unlink()

    rebuild_state(vault)
    item = load_state(vault)["items"]["kalshi-ingestion"]
    assert item["state"] == "not-doing"
    assert item["reason"] == "not until October"


def _age_decision(vault, key, days):
    """Backdate one logged decision so the prune can reach it."""
    from datetime import UTC, datetime, timedelta

    old = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()
    path = vault / "OpenAugi" / "Capture" / "feedback-log.ndjson"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get("item") == key:
            row["board"] = old
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_prune_drops_only_aged_bare_retirements(board, vault):
    _tick(board, "done", "insurance-numbers")
    _tick(board, "not doing", "kalshi-ingestion", reason="not until October")
    _tick(board, "someday", "archive-sessions")
    sync_board(board, vault)

    for key in ("insurance-numbers", "kalshi-ingestion", "archive-sessions"):
        _age_decision(vault, key, PRUNE_DAYS + 1)
    board.unlink()  # the notes are gone too, as they would be after tidying
    rebuild_state(vault)

    items = load_state(vault)["items"]
    assert "insurance-numbers" not in items  # bare `done`, aged out
    assert items["kalshi-ingestion"]["reason"] == "not until October"  # durable
    assert items["archive-sessions"]["state"] == "someday"  # parks never prune


def test_recent_retirement_survives_the_prune(board, vault):
    _tick(board, "done", "insurance-numbers")
    sync_board(board, vault)
    _age_decision(vault, "insurance-numbers", PRUNE_DAYS - 1)
    rebuild_state(vault)
    assert "insurance-numbers" in load_state(vault)["items"]


def test_board_notes_and_decisions_read_the_append_only_sources(board, vault):
    _tick(board, "done", "insurance-numbers")
    sync_board(board, vault)
    _second_board(vault)

    assert [day for day, _ in board_notes(vault)] == ["2026-09-02", "2026-09-03"]
    assert decisions_from_log(vault)["insurance-numbers"]["state"] == "done"


# ── The plain-markdown format ──────────────────────────────────────────────
#
# Boards moved off callouts on 2026-09-04: inside a callout every line carries
# a `> ` prefix, so in Obsidian there is nowhere to tap and type an `aaa:` line
# without fighting the editor. Headings leave ordinary paragraphs underneath.
# Both shapes are parsed — the callout boards already on disk still answer.

PLAIN_BOARD = """---
type: document
cssclasses:
  - board
---

# Board — 2026-09-05 (Friday)

*Previous board [[2026-09-02 - Board]]*

## Self · Money

*Left off: sizing done, blocked on three portal numbers.*

- **Pull the three numbers from the benefits portal** `quick · 5m`
    ↳ [[AMOC - Finances]]
    - [ ] done
    - [ ] not doing
    - [ ] someday
    <!-- item:insurance-numbers -->

## Needs your judgment (1 of 3)

- **Archive three finished sessions** — idle and complete
    - [ ] done
    - [ ] not doing
    - [ ] someday
    <!-- item:archive-sessions -->

## Augi could run these

- **Sweep the Board folder for stale drift flags** `build · 20m`
    ↳ Read every board in `OpenAugi/Board/` and list the drift flags never
      answered, with the dates they were raised.
    - [ ] do
    - [ ] no
    <!-- propose:drift-sweep -->

- **Refresh the Sessions Index** `quick · 5m`
    ↳ Run `scripts/session_cards.py`; it was last regenerated 2026-08-29.
    - [ ] do
    - [ ] no
    <!-- propose:sessions-index -->

## Notes to augi

<!-- board-note -->

*Everything else stays append-only truth underneath.*
"""


@pytest.fixture
def plain(vault):
    path = vault / "OpenAugi" / "Board" / "2026-09-05 - Board.md"
    path.write_text(PLAIN_BOARD, encoding="utf-8")
    return path


def test_plain_markdown_lanes_are_read_from_headings(plain):
    items = parse_items(plain.read_text(encoding="utf-8"))
    assert items["insurance-numbers"]["lane"] == "Self · Money"
    assert items["archive-sessions"]["lane"] == "Needs your judgment (1 of 3)"
    assert items["insurance-numbers"]["title"].startswith("Pull the three numbers")


def test_plain_markdown_ticks_and_aaa_work_without_a_callout(plain, vault):
    text = plain.read_text(encoding="utf-8").replace(
        "    - [ ] not doing\n    - [ ] someday\n    <!-- item:insurance-numbers -->",
        "    - [x] not doing\n    - [ ] someday\n    aaa: no idea what I need yet\n"
        "    <!-- item:insurance-numbers -->",
    )
    plain.write_text(text, encoding="utf-8")
    assert sync_board(plain, vault) == 1

    record = load_state(vault)["items"]["insurance-numbers"]
    assert record["state"] == "not-doing"
    assert record["reason"] == "no idea what I need yet"
    assert "- ✓ not-doing" in plain.read_text(encoding="utf-8")


def test_plain_markdown_board_note_is_logged_without_quote_prefixes(plain, vault):
    text = plain.read_text(encoding="utf-8").replace(
        "<!-- board-note -->\n", "<!-- board-note -->\nthe drift section was noise today\n"
    )
    plain.write_text(text, encoding="utf-8")
    sync_board(plain, vault)

    record = json.loads(
        (vault / "OpenAugi" / "Capture" / "feedback-log.ndjson")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()[-1]
    )
    assert record["source"] == "currency-board-note"
    assert record["reason"] == "the drift section was noise today"
    # Marked read in place, and the closing line below it is untouched.
    after = plain.read_text(encoding="utf-8")
    assert "✓ noted 2026-09-05" in after
    assert "*Everything else stays append-only truth underneath.*" in after
    assert sync_board(plain, vault) == 0


# ── Proposals ──────────────────────────────────────────────────────────────


def _tick_proposal(path, label, key):
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = next(i for i, line in enumerate(lines) if f"propose:{key}" in line)
    for i in range(marker - 1, max(marker - 8, -1), -1):
        if lines[i].strip().endswith(f"[ ] {label}"):
            lines[i] = lines[i].replace(f"[ ] {label}", f"[x] {label}")
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_proposals_carry_their_own_brief(plain):
    proposals = parse_proposals(plain.read_text(encoding="utf-8"))
    assert set(proposals) == {"drift-sweep", "sessions-index"}
    assert proposals["drift-sweep"]["title"].startswith("Sweep the Board folder")
    assert "never" in proposals["drift-sweep"]["body"]
    assert proposals["drift-sweep"]["body"].startswith("↳ Read every board")
    assert all(p["answer"] is None for p in proposals.values())


def test_ticking_do_writes_a_pending_task_the_watcher_can_pick_up(plain, vault):
    _tick_proposal(plain, "do", "drift-sweep")
    assert sync_board(plain, vault) == 1

    task = vault / "OpenAugi" / "Tasks" / "board-2026-09-05-drift-sweep.md"
    body = task.read_text(encoding="utf-8")
    assert "status: pending" in body
    assert f"working_dir: {vault}" in body
    assert "proposal: drift-sweep" in body
    # The brief he read is the brief the agent gets.
    assert "Sweep the Board folder for stale drift flags" in body
    assert "Read every board" in body
    assert "[[2026-09-05 - Board]]" in body

    assert "- ✓ do → `board-2026-09-05-drift-sweep.md`" in plain.read_text(encoding="utf-8")
    assert load_state(vault)["proposals"]["drift-sweep"]["state"] == "dispatched"
    # The untouched proposal keeps both of its buttons.
    assert plain.read_text(encoding="utf-8").count("] no\n") == 1


def test_dispatch_never_writes_the_same_task_twice(plain, vault):
    _tick_proposal(plain, "do", "drift-sweep")
    sync_board(plain, vault)
    task = vault / "OpenAugi" / "Tasks" / "board-2026-09-05-drift-sweep.md"
    task.write_text("hydrated by the task watcher", encoding="utf-8")
    # Re-tick the confirmation line as if the board were rebuilt over it.
    plain.write_text(
        plain.read_text(encoding="utf-8").replace(
            "- ✓ do → `board-2026-09-05-drift-sweep.md`", "- [x] do"
        ),
        encoding="utf-8",
    )
    sync_board(plain, vault)
    assert task.read_text(encoding="utf-8") == "hydrated by the task watcher"


def test_declining_a_proposal_retires_it_and_writes_no_task(plain, vault):
    _tick_proposal(plain, "no", "sessions-index")
    assert sync_board(plain, vault) == 1

    assert not (vault / "OpenAugi" / "Tasks").exists()
    assert "- ✓ no" in plain.read_text(encoding="utf-8")
    assert "sessions-index" in declined_proposals(vault)
    assert "drift-sweep" not in declined_proposals(vault)

    record = json.loads(
        (vault / "OpenAugi" / "Capture" / "feedback-log.ndjson")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()[-1]
    )
    assert record["source"] == "currency-board-proposal"
    assert record["signal"] == "declined"


def test_proposal_state_survives_a_rebuild_from_scratch(plain, vault):
    _tick_proposal(plain, "no", "sessions-index")
    sync_board(plain, vault)
    (vault / "OpenAugi" / "Board" / ".board-state.json").unlink()
    rebuilt = rebuild_state(vault)
    assert rebuilt["proposals"]["sessions-index"]["state"] == "declined"
    assert rebuilt["proposals"]["drift-sweep"]["state"] == "open"


# ── Comparing today to yesterday ───────────────────────────────────────────


def test_previous_board_summary_splits_answered_from_carried(board, vault, plain):
    _tick(board, "done", "kalshi-ingestion")
    sync_board(board, vault)

    summary = previous_board_summary(vault, before="2026-09-05")
    assert summary["date"] == "2026-09-02"
    assert summary["note"] == "2026-09-02 - Board"
    assert set(summary["answered"]) == {"kalshi-ingestion"}
    assert summary["answered"]["kalshi-ingestion"]["state"] == "done"
    assert set(summary["carried"]) == {"insurance-numbers", "archive-sessions"}
    # Two boards carry it — the one summarised and today's, both on disk.
    assert summary["carried"]["insurance-numbers"]["appearances"] == 2


def test_previous_board_summary_is_none_when_there_is_no_earlier_board(board, vault):
    sync_board(board, vault)
    assert previous_board_summary(vault, before="2026-09-02") is None
