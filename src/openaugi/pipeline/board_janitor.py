"""Board janitor — turns the checkboxes on a currency board into state.

The board (see `OpenAugi/AGENT/lenses/currency-board.md`) offers three boxes
under every move, judgment item and drift option: done / not doing / someday.
Ticking one IS the command — the same precedent the echo janitor follows.
An optional `aaa:` line under the boxes is the comment channel: free text
explaining *why*, which the next board must honor literally.

Two jobs, both idempotent, both run from `sync_board`:

1. **Record** — every item on the board gets its `appearances` counter bumped
   once per board date. That counter is what lets the next board say "third
   board — do it or say not doing" instead of nagging blindly.
2. **Process** — ticked boxes retire the item (permanently for `not-doing`,
   until weekly reflection for `someday`), append a signal to the shared
   `feedback-log.ndjson` stream, and rewrite the answered lines into a
   confirmation so nothing is ever processed twice.

State lives in `OpenAugi/Board/.board-state.json`, which the lens reads
before building the next board. No LLM calls in this module.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

BOARD_FOLDER = "OpenAugi/Board"
STATE_FILE = f"{BOARD_FOLDER}/.board-state.json"
FEEDBACK_LOG = "OpenAugi/Capture/feedback-log.ndjson"

#: `<!-- item:some-stable-key -->`, optionally inside a callout (`> ` prefix).
_ITEM_RE = re.compile(r"^\s*>?\s*<!-- item:(?P<key>[a-z0-9][a-z0-9-]*) -->\s*$")
#: `- [x] not doing` at any indent, optionally inside a callout.
_BOX_RE = re.compile(
    r"^(?P<pre>\s*>?\s*)- \[(?P<mark>[ xX])\] (?P<label>done|not doing|someday)\s*$"
)
#: `aaa: because it can wait until October`
_AAA_RE = re.compile(r"^\s*>?\s*aaa:\s*(?P<reason>.*?)\s*$", re.IGNORECASE)
#: `- **Pull the three numbers from the benefits portal** \`quick · 5m\``
_TITLE_RE = re.compile(r"^\s*>?\s*- \*\*(?P<title>.+?)\*\*")
#: `> [!board-lane]+ Self · Money`
_LANE_RE = re.compile(r"^\s*>?\s*\[!board-(?P<kind>[a-z]+)\][+-]?\s*(?P<label>.*?)\s*$")
#: `2026-09-02 - Board.md`
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
#: The free-text channel for feedback about the board itself.
_NOTE_MARKER_RE = re.compile(r"^\s*>?\s*<!-- board-note -->\s*$")
#: A line inside the note callout, once its `> ` prefix is stripped.
_NOTE_LINE_RE = re.compile(r"^\s*>\s?(?P<text>.*?)\s*$")

_STATES = {"done": "done", "not doing": "not-doing", "someday": "someday"}
#: How far back an item may be scanned for its title/lane before giving up.
_LOOKBACK = 12


def _now() -> str:
    return datetime.now(UTC).isoformat()


def board_date(path: Path) -> str:
    """The board's own date, taken from its filename."""
    match = _DATE_RE.search(path.name)
    return match.group(1) if match else datetime.now(UTC).date().isoformat()


def load_state(vault_path: Path) -> dict:
    """Read the board state, or an empty one on first run / unreadable file."""
    path = vault_path / STATE_FILE
    if not path.exists():
        return {"last_run": None, "items": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Board state unreadable ({e}); starting fresh")
        return {"last_run": None, "items": {}}
    state.setdefault("items", {})
    state.setdefault("last_run", None)
    return state


def save_state(vault_path: Path, state: dict) -> None:
    path = vault_path / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_feedback(vault_path: Path, record: dict) -> None:
    path = vault_path / FEEDBACK_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _context_for(lines: list[str], idx: int) -> tuple[str, str]:
    """Walk back from an item marker to find its title and enclosing lane."""
    title = ""
    for offset in range(1, min(_LOOKBACK, idx) + 1):
        if match := _TITLE_RE.match(lines[idx - offset]):
            title = match.group("title")
            break
    lane = ""
    for back in range(idx, -1, -1):
        if match := _LANE_RE.search(lines[back]):
            lane = match.group("label") or match.group("kind")
            break
    return title, lane


def parse_items(text: str) -> dict[str, dict]:
    """Every item on a board: its key, title, lane, ticked answer and reason.

    Line indices are returned so the caller can rewrite exactly the lines the
    user answered, leaving the rest of the board untouched.
    """
    lines = text.splitlines()
    items: dict[str, dict] = {}
    for idx, line in enumerate(lines):
        match = _ITEM_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        title, lane = _context_for(lines, idx)
        item = {
            "title": title,
            "lane": lane,
            "answer": None,
            "reason": "",
            "box_lines": [],
            "answer_line": None,
            "indent": "",
        }
        # The boxes and the aaa: line sit between the title and this marker.
        for offset in range(1, min(_LOOKBACK, idx) + 1):
            pos = idx - offset
            if box := _BOX_RE.match(lines[pos]):
                item["box_lines"].append(pos)
                if box.group("mark").lower() == "x" and item["answer"] is None:
                    item["answer"] = _STATES[box.group("label")]
                    item["answer_line"] = pos
                    # Captured here so the rewrite never re-matches the line.
                    item["indent"] = box.group("pre")
            elif aaa := _AAA_RE.match(lines[pos]):
                item["reason"] = aaa.group("reason")
            elif _TITLE_RE.match(lines[pos]):
                break
        items[key] = item
    return items


def _process_board_note(lines: list[str], vault_path: Path, day: str) -> bool:
    """Log free text left in the `Notes to augi` callout, then mark it read.

    This is the channel for feedback about the *board* rather than about one
    item: what was wrong, missing, too vague, or noise. Returns True when
    something was found, so the caller knows to rewrite the file.
    """
    # Answered items leave None tombstones behind; skip them.
    start = next(
        (i for i, line in enumerate(lines) if line is not None and _NOTE_MARKER_RE.match(line)),
        None,
    )
    if start is None:
        return False

    body: list[tuple[int, str]] = []
    for pos in range(start + 1, len(lines)):
        if lines[pos] is None:
            continue
        match = _NOTE_LINE_RE.match(lines[pos])
        if not match:  # the callout ended
            break
        text = match.group("text")
        if text.startswith("✓ noted"):  # already processed
            return False
        if text:
            body.append((pos, text))

    if not body:
        return False

    _append_feedback(
        vault_path,
        {
            "ts": _now(),
            "source": "currency-board-note",
            "board": day,
            "signal": "note",
            "reason": " ".join(text for _, text in body),
        },
    )
    for pos, _ in body[1:]:
        lines[pos] = None  # type: ignore[call-overload]
    lines[body[0][0]] = f"> ✓ noted {day}"
    logger.info(f"Board janitor: board note recorded for {day}")
    return True


def sync_board(board_path: Path, vault_path: Path) -> int:
    """Record appearances and act on ticked boxes for one board. Idempotent.

    Returns the number of items answered on this call (0 when the board is
    merely being recorded, or has already been fully processed).
    """
    if not board_path.exists():
        return 0
    text = board_path.read_text(encoding="utf-8")
    items = parse_items(text)
    if not items:
        return 0

    day = board_date(board_path)
    state = load_state(vault_path)
    known = state["items"]
    lines = text.splitlines()
    answered = 0

    for key, item in items.items():
        record = known.setdefault(
            key,
            {
                "state": "open",
                "title": item["title"],
                "lane": item["lane"],
                "first_seen": day,
                "last_seen": None,
                "appearances": 0,
                "reason": "",
            },
        )
        # Keep the human-readable fields fresh; the key is the stable identity.
        record["title"] = item["title"] or record.get("title", "")
        record["lane"] = item["lane"] or record.get("lane", "")

        # 1. Record — once per board date, however often this runs.
        if record.get("last_seen") != day:
            record["last_seen"] = day
            record["appearances"] = record.get("appearances", 0) + 1

        # 2. Process — a ticked box retires the item and rewrites its lines.
        if item["answer"] is None:
            continue
        record["state"] = item["answer"]
        record["answered_on"] = day
        if item["reason"]:
            record["reason"] = item["reason"]
        _append_feedback(
            vault_path,
            {
                "ts": _now(),
                "source": "currency-board",
                "board": day,
                "item": key,
                "title": item["title"],
                "lane": item["lane"],
                "signal": item["answer"],
                "reason": item["reason"],
            },
        )
        # Confirmation replaces the answered box; the unanswered boxes go.
        answer_line = item["answer_line"]
        lines[answer_line] = f"{item['indent']}- ✓ {item['answer']}"
        for pos in item["box_lines"]:
            if pos != answer_line:
                lines[pos] = None  # type: ignore[call-overload]
        answered += 1

    noted = _process_board_note(lines, vault_path, day)

    if answered or noted:
        board_path.write_text(
            "\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8"
        )
        logger.info(f"Board janitor: {answered} item(s) answered on {board_path.name}")

    state["items"] = known
    state["last_run"] = max(filter(None, [state.get("last_run"), day]))
    save_state(vault_path, state)
    return answered


def open_items(vault_path: Path) -> dict[str, dict]:
    """Items the next board may carry — everything not retired or parked."""
    return {
        key: item
        for key, item in load_state(vault_path)["items"].items()
        if item.get("state") == "open"
    }


def retired_items(vault_path: Path) -> dict[str, dict]:
    """Items the next board must NOT re-propose, with the reasons given."""
    return {
        key: item
        for key, item in load_state(vault_path)["items"].items()
        if item.get("state") in {"done", "not-doing", "someday"}
    }


def process_changed(changed_paths: set[str], vault_path: Path) -> int:
    """Janitor entry point for the watcher — handles any touched board note."""
    total = 0
    for raw in changed_paths:
        path = Path(raw)
        if path.suffix == ".md" and path.parent.name == "Board" and "Board" in path.stem:
            try:
                total += sync_board(path, vault_path)
            except Exception as e:
                logger.error(f"Board janitor failed on {path}: {e}", exc_info=True)
    return total
