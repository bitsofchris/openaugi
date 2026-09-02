"""Board janitor — checkbox answers become state the next board honors."""

from __future__ import annotations

import json

import pytest

from openaugi.pipeline.board_janitor import (
    load_state,
    open_items,
    parse_items,
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


def test_missing_board_and_unreadable_state_are_survivable(vault):
    assert sync_board(vault / "OpenAugi" / "Board" / "nope.md", vault) == 0
    (vault / "OpenAugi" / "Board" / ".board-state.json").write_text("{{{", encoding="utf-8")
    assert load_state(vault) == {"last_run": None, "items": {}}
