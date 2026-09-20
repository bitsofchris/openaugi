"""Board janitor — turns the checkboxes on a currency board into state.

The board (see `OpenAugi/AGENT/lenses/currency-board.md`) offers three boxes
under every move, judgment item and drift option: done / not doing / someday.
Ticking one IS the command — the same precedent the echo janitor follows.
An optional `aaa:` line under the boxes is the comment channel: free text
explaining *why*, which the next board must honor literally.

A board also *proposes*: at the bottom sit two-button offers (`do` / `no`) for
tasks augi would run for the user. Ticking `do` writes a pending task file into
`OpenAugi/Tasks/`, which the task watcher hydrates and launches like any zzz
dispatch; ticking `no` retires the offer permanently. Nothing else about the
board differs — the same `aaa:` line is the same comment channel.

`sync_board` does one job: **process** ticked boxes. A tick retires the item
(permanently for `not-doing`, until weekly reflection for `someday`), appends
a signal to the shared `feedback-log.ndjson` stream, and rewrites the answered
lines into a confirmation so nothing is ever processed twice.

State is then *rebuilt*, never mutated.

`OpenAugi/Board/.board-state.json` is a **pure projection** over two
append-only sources, both of which are files you can read:

- the dated board notes in `OpenAugi/Board/` — who was proposed, and when.
  `appearances`, `first_seen` and `last_seen` are *counted from these*, not
  stored as counters.
- `feedback-log.ndjson` — every decision, with its reason.

Nothing else may write this file. It was mutated in place until 2026-09-03,
by both this module and by board-build agent sessions; the counters drifted
to 3 on a two-day-old board and falsely tripped the staleness flag. A
projection cannot drift: if the file is ever wrong, delete it and call
`rebuild_state`. That replay is the whole resilience story.

Pruning keeps it bounded without losing anything reachable. A bare
`done`/`not-doing` older than `PRUNE_DAYS` is dropped — the board's window is
72 hours, so nothing that old can be re-proposed from fresh writing. Anything
`someday`, and anything carrying a `reason` ("not until October"), is durable
and never pruned.

The weekly reflection (`Research/Weekly Reflection - <date>.md`) is the one
other surface with `do` / `no` offers — "Propose next week" and "Apply", the
two ticks of the Sunday pass. `sync_reflection` runs only the proposal half
on it: same task file shape, its own log source, no state projection.

No LLM calls in this module.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path

from openaugi.pipeline.writeback import (
    aaa_re,
    append_feedback,
    box_re,
    now,
    read_feedback,
    ticked,
)

logger = logging.getLogger(__name__)

BOARD_FOLDER = "OpenAugi/Board"
STATE_FILE = f"{BOARD_FOLDER}/.board-state.json"

#: `<!-- item:some-stable-key -->`, optionally inside a callout (`> ` prefix).
_ITEM_RE = re.compile(r"^\s*>?\s*<!-- item:(?P<key>[a-z0-9][a-z0-9-]*) -->\s*$")
#: `- [x] not doing` at any indent, optionally inside a callout.
_BOX_RE = box_re("done", "not doing", "someday", callout=True)
#: `aaa: because it can wait until October`
_AAA_RE = aaa_re(callout=True)
#: `- **Pull the three numbers from the benefits portal** \`quick · 5m\``
_TITLE_RE = re.compile(r"^\s*>?\s*- \*\*(?P<title>.+?)\*\*")
#: `> [!board-lane]+ Self · Money` — the old callout heading, still parsed so
#: every board already on disk keeps working.
_LANE_RE = re.compile(r"^\s*>?\s*\[!board-(?P<kind>[a-z]+)\][+-]?\s*(?P<label>.*?)\s*$")
#: `## Self · Money` — the plain-markdown lane heading that replaced it. Inside
#: a callout every line is quoted, so there is nowhere to tap and type an
#: `aaa:` line; a heading leaves ordinary paragraphs underneath it.
_HEADING_RE = re.compile(r"^\s{0,3}#{2,4}\s+(?P<label>.+?)\s*$")
#: `2026-09-02 - Board.md`
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
#: The free-text channel for feedback about the board itself.
_NOTE_MARKER_RE = re.compile(r"^\s*>?\s*<!-- board-note -->\s*$")
#: A line in the note section — inside a callout (`> text`) or plain markdown.
_NOTE_LINE_RE = re.compile(r"^\s*(?:>\s?)?(?P<text>.*?)\s*$")
#: What ends the plain-markdown note section: the next heading, a rule, or the
#: board's closing line.
_NOTE_END_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s|---\s*$|\*Everything else)")

#: `<!-- propose:sweep-stale-drift -->` — a task augi offers to run.
_PROPOSE_RE = re.compile(r"^\s*>?\s*<!-- propose:(?P<key>[a-z0-9][a-z0-9-]*) -->\s*$")
#: `- [x] do` / `- [ ] no` — the two buttons under a proposal.
_PROPOSE_BOX_RE = box_re("do", "no", callout=True)

_STATES = {"done": "done", "not doing": "not-doing", "someday": "someday"}
_PROPOSE_STATES = {"do": "dispatched", "no": "declined"}
#: The only proposal states. `dispatched` means a task file was written and the
#: task watcher owns it from there; `declined` means never offer this again.
_VALID_PROPOSAL_STATES = {"open", "dispatched", "declined"}
#: Log sources. Item ticks and proposal ticks share the stream but not the
#: vocabulary, so they are projected separately.
BOARD_SOURCE = "currency-board"
# Free text left in the `Notes to augi` section — feedback about the board
# itself. Logged under its own source so it never mixes with item ticks.
NOTE_SOURCE = "currency-board-note"
PROPOSAL_SOURCE = "currency-board-proposal"
#: Where a dispatched proposal lands. The task watcher hydrates and renames it.
TASKS_FOLDER = "OpenAugi/Tasks"
#: The second surface that carries `do` / `no` proposals: the Sunday
#: reflection (`OpenAugi/AGENT/lenses/weekly-reflection.md`). Its two offers —
#: "Propose next week" and "Apply" — are the whole Sunday pass, so a tick there
#: must dispatch exactly like a tick on a board. It has no items, no lanes and
#: no state file; only the proposal half of this module applies to it.
REFLECTION_FOLDER = "OpenAugi/Research"
REFLECTION_SOURCE = "weekly-reflection-proposal"
_REFLECTION_STEM_RE = re.compile(r"^Weekly Reflection - \d{4}-\d{2}-\d{2}$")
#: The only states this module recognises. Anything else in a state file was
#: written by something that had no business writing it, and is discarded.
_VALID_STATES = {"open", "done", "not-doing", "someday"}
#: Retirements with no reason attached age out after this many days. The board
#: reads a 72h window, so nothing this old is reachable from fresh writing.
PRUNE_DAYS = 90
#: How far back an item may be scanned for its title/lane before giving up.
_LOOKBACK = 12


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
    state.setdefault("proposals", {})
    state.setdefault("last_run", None)
    # The file is a projection; a hand-edit or a stray writer cannot be trusted
    # to have used the vocabulary. Drop what we do not recognise and say so —
    # `rebuild_state` will put back anything the ground truth still supports.
    for section, vocabulary in (("items", _VALID_STATES), ("proposals", _VALID_PROPOSAL_STATES)):
        bad = [k for k, v in state[section].items() if v.get("state") not in vocabulary]
        for key in bad:
            logger.warning(
                f"Board state: {section[:-1]} {key!r} has unknown state "
                f"{state[section][key].get('state')!r}; dropping it"
            )
            del state[section][key]
    return state


def save_state(vault_path: Path, state: dict) -> None:
    path = vault_path / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _context_for(lines: list[str], idx: int) -> tuple[str, str]:
    """Walk back from an item marker to find its title and enclosing lane."""
    title = ""
    for offset in range(1, min(_LOOKBACK, idx) + 1):
        if match := _TITLE_RE.match(lines[idx - offset]):
            title = match.group("title")
            break
    lane = ""
    for back in range(idx, -1, -1):
        # Either shape of section header counts — callout boards predate the
        # plain-markdown ones and both are still on disk.
        if match := _LANE_RE.search(lines[back]):
            lane = match.group("label") or match.group("kind")
            break
        if match := _HEADING_RE.match(lines[back]):
            lane = match.group("label")
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
                if ticked(box) and item["answer"] is None:
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


def _note_lines(text: str) -> list[str]:
    """The free-text lines left in the `Notes to augi` section.

    Reads the board as written — the section is never rewritten, so this is a
    pure read. Legacy `✓ noted` receipt lines (written by an older janitor
    that overwrote the first line) are skipped, not treated as a terminator:
    a receipt above the text must not hide the text below it.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _NOTE_MARKER_RE.match(line)), None)
    if start is None:
        return []

    # A quoted marker means this board still uses the callout format, where the
    # section ends at the first unquoted line. A plain one runs until the next
    # heading, rule or checkbox.
    quoted = lines[start].lstrip().startswith(">")

    body: list[str] = []
    for pos in range(start + 1, len(lines)):
        line = lines[pos]
        if quoted and not line.lstrip().startswith(">"):
            break  # the callout ended
        if _NOTE_END_RE.match(line) or _BOX_RE.match(line) or _ITEM_RE.match(line):
            break  # the next section started
        entry = _NOTE_LINE_RE.match(line).group("text")  # type: ignore[union-attr]
        if not entry or entry.startswith("✓ noted"):
            continue
        body.append(entry)
    return body


def _logged_note_lines(vault_path: Path, day: str) -> set[str]:
    """Note lines already logged for this board — the read marker.

    The append-only log is where "we have seen this" lives, so the board note
    itself needs no receipt and never gets edited. Rows written before the
    `lines` field existed fall back to their joined `reason`.
    """
    seen: set[str] = set()
    for row in read_feedback(vault_path, source=NOTE_SOURCE):
        if row.get("board") != day:
            continue
        logged = row.get("lines")
        if isinstance(logged, list):
            seen.update(str(entry) for entry in logged)
        elif row.get("reason"):
            seen.add(str(row["reason"]))
    return seen


def _log_board_note(text: str, vault_path: Path, day: str) -> int:
    """Log free text left in the `Notes to augi` callout. Never edits it.

    This is the channel for feedback about the *board* rather than about one
    item: what was wrong, missing, too vague, or noise. The text belongs to
    whoever wrote it — the janitor reads it and remembers that it read it, and
    that is all. Adding a
    line later logs the new line only; nothing already logged repeats.

    Returns the number of lines newly logged.
    """
    body = _note_lines(text)
    if not body:
        return 0
    seen = _logged_note_lines(vault_path, day)
    fresh = [entry for entry in body if entry not in seen]
    if not fresh:
        return 0

    append_feedback(
        vault_path,
        {
            "ts": now(),
            "source": NOTE_SOURCE,
            "board": day,
            "signal": "note",
            "lines": fresh,
            "reason": " ".join(fresh),
        },
    )
    logger.info(f"Board janitor: {len(fresh)} board-note line(s) recorded for {day}")
    return len(fresh)


def _unquote(line: str) -> str:
    """One board line as prose — callout prefix and leading indent removed."""
    return re.sub(r"^\s*>\s?", "", line).strip()


def _proposal_title(lines: list[str], idx: int) -> tuple[str, int | None]:
    """The title line of the proposal whose marker sits at `idx`.

    Unbounded, unlike an item's: a `Worth keeping` proposal carries a whole
    drafted note between its title and its boxes, and a fixed lookback
    silently truncates it. On 2026-09-11 a 14-line proposal parsed with an
    empty title and an empty body, and dispatched a task with no brief in it.
    The scan stops at the previous structural boundary instead — the lane
    heading, or the marker of whatever was proposed before this one.
    """
    for pos in range(idx - 1, -1, -1):
        if found := _TITLE_RE.match(lines[pos]):
            return found.group("title"), pos
        if (
            _HEADING_RE.match(lines[pos])
            or _ITEM_RE.match(lines[pos])
            or _PROPOSE_RE.match(lines[pos])
        ):
            break
    return "", None


def parse_proposals(text: str) -> dict[str, dict]:
    """Every proposed task on a board: what augi offered, and the answer.

    A proposal is a task augi would run *for* the user, offered with two
    buttons instead of three: `do` writes the task file, `no` retires the
    offer. The lines between the title and the boxes are the proposal's own
    description, and they become the task's context — so what was read is what
    the agent gets, with nothing regenerated in between.
    """
    lines = text.splitlines()
    out: dict[str, dict] = {}
    for idx, line in enumerate(lines):
        match = _PROPOSE_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        title, title_line = _proposal_title(lines, idx)
        item = {
            "title": title,
            "body": "",
            "answer": None,
            "reason": "",
            "box_lines": [],
            "answer_line": None,
            "indent": "",
        }
        first_box = idx
        # Everything belonging to this proposal lies between its title and its
        # marker, so that is the scan range — falling back to the item lookback
        # only when there is no title to bound it.
        floor = title_line if title_line is not None else max(idx - _LOOKBACK - 1, -1)
        seen_aaa = False
        for pos in range(idx - 1, floor, -1):
            if box := _PROPOSE_BOX_RE.match(lines[pos]):
                item["box_lines"].append(pos)
                first_box = min(first_box, pos)
                if ticked(box) and item["answer"] is None:
                    item["answer"] = _PROPOSE_STATES[box.group("label")]
                    item["answer_line"] = pos
                    item["indent"] = box.group("pre")
            elif (aaa := _AAA_RE.match(lines[pos])) and not seen_aaa:
                # The comment channel is the line between the boxes and the
                # marker; a scan that now reaches the body must not be fooled
                # by prose that happens to start `aaa:`.
                item["reason"], seen_aaa = aaa.group("reason"), True
        if title_line is not None:
            body = [_unquote(raw) for raw in lines[title_line + 1 : first_box]]
            item["body"] = "\n".join(part for part in body if part).strip()
        out[key] = item
    return out


def build_proposal_task(
    key: str,
    proposal: dict,
    *,
    vault_path: Path,
    day: str,
    board_stem: str,
    source: str = BOARD_SOURCE,
    origin: str = "the currency board",
) -> str:
    """The pending task file for an accepted proposal.

    Deliberately the same shape as a zzz dispatch (`pipeline/dispatch.py`) so
    the task watcher needs to know nothing about boards: it finds a pending
    file, hydrates it, and launches a session like any other.
    """
    aaa = f"\n\nNote on it: {proposal['reason']}" if proposal["reason"] else ""
    return f"""---
status: pending
working_dir: {vault_path}
source: {source}
board: {day}
proposal: {key}
---

# {proposal["title"]}

## Context

Proposed by {origin} on {day} and accepted from [[{board_stem}]].

{proposal["body"]}{aaa}

## User instruction

> {proposal["title"]}

## Task

Do the work described above. The proposal text is the brief — it is what the
user read before ticking `do`, so treat it as the instruction and do not widen
the scope beyond it. If it turns out to be underspecified, say so in `## Results`
and set `status: needs-input` rather than guessing.

## Human Todo

## Results
"""


def _process_proposals(
    lines: list[str | None],
    text: str,
    board_path: Path,
    vault_path: Path,
    day: str,
    *,
    task_prefix: str = "board",
    source: str = PROPOSAL_SOURCE,
    task_source: str = BOARD_SOURCE,
    origin: str = "the currency board",
) -> int:
    """Act on ticked proposal boxes: `do` writes a task, `no` retires the offer.

    The keyword arguments name the surface: a board writes `board-<day>-<key>`
    under the board sources, the reflection writes `reflection-<day>-<key>`
    under its own, so the two never share a log projection.

    Returns the number of proposals answered on this call.
    """
    answered = 0
    for key, proposal in parse_proposals(text).items():
        if proposal["answer"] is None:
            continue
        task_name = ""
        if proposal["answer"] == "dispatched":
            tasks_dir = vault_path / TASKS_FOLDER
            tasks_dir.mkdir(parents=True, exist_ok=True)
            task_path = tasks_dir / f"{task_prefix}-{day}-{key}.md"
            if task_path.exists():
                # Already written and possibly already hydrated into a TASK-*
                # file; writing again would launch the same agent twice.
                logger.info(f"Board janitor: proposal {key!r} already dispatched")
            else:
                task_path.write_text(
                    build_proposal_task(
                        key,
                        proposal,
                        vault_path=vault_path,
                        day=day,
                        board_stem=board_path.stem,
                        source=task_source,
                        origin=origin,
                    ),
                    encoding="utf-8",
                )
                logger.info(f"Board janitor: proposal {key!r} dispatched → {task_path.name}")
            task_name = task_path.name
        append_feedback(
            vault_path,
            {
                "ts": now(),
                "source": source,
                "board": day,
                "item": key,
                "title": proposal["title"],
                "signal": proposal["answer"],
                "reason": proposal["reason"],
                "task_file": task_name,
            },
        )
        confirmation = (
            f"- ✓ do → `{task_name}`" if proposal["answer"] == "dispatched" else "- ✓ no"
        )
        lines[proposal["answer_line"]] = f"{proposal['indent']}{confirmation}"
        for pos in proposal["box_lines"]:
            if pos != proposal["answer_line"]:
                lines[pos] = None
        answered += 1
    return answered


def board_notes(vault_path: Path) -> list[tuple[str, Path]]:
    """Every dated board note on disk, oldest first. Ground truth for age."""
    folder = vault_path / BOARD_FOLDER
    if not folder.is_dir():
        return []
    found = [
        (match.group(1), path)
        for path in folder.glob("*.md")
        if (match := _DATE_RE.search(path.name))
    ]
    return sorted(found)


def decisions_from_log(
    vault_path: Path,
    *,
    source: str = BOARD_SOURCE,
    vocabulary: set[str] | None = None,
) -> dict[str, dict]:
    """The latest decision per item key, read from the append-only log.

    The log is the record of what was actually ticked. A later entry for the
    same key wins, so changing your mind is just another append. Item ticks and
    proposal ticks share the stream but not the vocabulary, so each is read
    with its own `source` and set of legal signals.
    """
    vocabulary = vocabulary or _VALID_STATES
    decisions: dict[str, dict] = {}
    for row in read_feedback(vault_path, source=source):
        key, signal = row.get("item"), row.get("signal")
        if not key or signal not in vocabulary:
            continue
        decisions[key] = {
            "state": signal,
            "reason": row.get("reason") or "",
            "answered_on": row.get("board") or "",
        }
    return decisions


def _prunable(item: dict, today: str) -> bool:
    """A retirement nothing can reach any more, carrying nothing worth keeping."""
    if item["state"] not in {"done", "not-doing"}:
        return False  # `open` is live; `someday` is a park that must persist
    if item.get("reason"):
        return False  # a stated reason is the durable half of a decision
    answered = item.get("answered_on") or ""
    if not answered:
        return False
    try:
        age = (date.fromisoformat(today) - date.fromisoformat(answered)).days
    except ValueError:
        return False
    return age > PRUNE_DAYS


def _project_proposals(notes: list[tuple[str, Path]], vault_path: Path) -> dict[str, dict]:
    """Proposal state, projected the same way item state is: notes plus log.

    A `declined` proposal is the one thing the next board must never repeat —
    offering the same agent task twice after a `no` is exactly the failure
    that costs a surface its trust.
    """
    answers = decisions_from_log(
        vault_path, source=PROPOSAL_SOURCE, vocabulary=_VALID_PROPOSAL_STATES
    )
    proposals: dict[str, dict] = {}
    for day, path in notes:  # oldest first, so the newest title wins
        try:
            parsed = parse_proposals(path.read_text(encoding="utf-8"))
        except OSError as e:
            logger.warning(f"Board state: cannot read {path.name} ({e}); skipping")
            continue
        for key, parsed_proposal in parsed.items():
            record = proposals.setdefault(key, {"first_seen": day, "title": ""})
            record["last_seen"] = day
            if parsed_proposal["title"]:
                record["title"] = parsed_proposal["title"]
    for key, record in proposals.items():
        answer = answers.get(key)
        record["state"] = answer["state"] if answer else "open"
        record["reason"] = answer["reason"] if answer else ""
        if answer and answer["answered_on"]:
            record["answered_on"] = answer["answered_on"]
    # A decision whose board note was deleted still binds us.
    for key, answer in answers.items():
        proposals.setdefault(
            key,
            {
                "title": "",
                "first_seen": answer["answered_on"],
                "last_seen": answer["answered_on"],
                "state": answer["state"],
                "reason": answer["reason"],
                "answered_on": answer["answered_on"],
            },
        )
    return proposals


def declined_proposals(vault_path: Path) -> dict[str, dict]:
    """Proposals the next board must NOT offer again, with the reasons given."""
    return {
        key: proposal
        for key, proposal in load_state(vault_path).get("proposals", {}).items()
        if proposal.get("state") == "declined"
    }


def previous_board_summary(vault_path: Path, *, before: str | None = None) -> dict | None:
    """What the last board proposed, and what became of it.

    The board is the one surface that promises currency, and "how does today
    compare to yesterday" was unanswerable from the note alone. This is the
    cheap answer: the previous board's own items, split into what was answered
    and what is still carried, so today's board can link back and say so.

    `before` limits the search to boards strictly older than that date — pass
    today's date while building today's board.
    """
    notes = [(day, path) for day, path in board_notes(vault_path) if not before or day < before]
    if not notes:
        return None
    day, path = notes[-1]
    try:
        parsed = parse_items(path.read_text(encoding="utf-8"))
    except OSError as e:
        logger.warning(f"Board state: cannot read {path.name} ({e})")
        return None
    known = load_state(vault_path)["items"]
    answered, carried = {}, {}
    for key, item in parsed.items():
        record = known.get(key, {})
        row = {
            "title": item["title"] or record.get("title", ""),
            "state": record.get("state", "open"),
            "reason": record.get("reason", ""),
            "appearances": record.get("appearances", 1),
        }
        (carried if row["state"] == "open" else answered)[key] = row
    return {"date": day, "note": path.stem, "answered": answered, "carried": carried}


def rebuild_state(vault_path: Path, *, save: bool = True) -> dict:
    """Project the state file from the board notes and the decision log.

    This is the only function that writes `.board-state.json`. It reads no
    prior state, so a corrupt file repairs itself: delete it and call this.
    """
    notes = board_notes(vault_path)
    decisions = decisions_from_log(vault_path)
    today = datetime.now(UTC).date().isoformat()
    proposals = _project_proposals(notes, vault_path)

    items: dict[str, dict] = {}
    for day, path in notes:  # oldest first, so last write wins for the title
        try:
            parsed = parse_items(path.read_text(encoding="utf-8"))
        except OSError as e:
            logger.warning(f"Board state: cannot read {path.name} ({e}); skipping")
            continue
        for key, parsed_item in parsed.items():
            record = items.setdefault(key, {"first_seen": day, "appearances": 0, "boards": []})
            if day not in record["boards"]:
                record["boards"].append(day)
            if parsed_item["title"]:
                record["title"] = parsed_item["title"]

    for key, record in items.items():
        boards = record.pop("boards")
        # Derived, never incremented — this is the pair that used to drift.
        record["appearances"] = len(boards)
        record["last_seen"] = boards[-1]
        record.setdefault("title", "")
        decision = decisions.get(key)
        record["state"] = decision["state"] if decision else "open"
        record["reason"] = decision["reason"] if decision else ""
        if decision and decision["answered_on"]:
            record["answered_on"] = decision["answered_on"]

    # A decision whose board note has been deleted still binds us — otherwise
    # tidying old boards would silently un-retire what was already ruled on.
    for key, decision in decisions.items():
        if key not in items:
            items[key] = {
                "title": "",
                "first_seen": decision["answered_on"],
                "last_seen": decision["answered_on"],
                "appearances": 0,
                "state": decision["state"],
                "reason": decision["reason"],
                "answered_on": decision["answered_on"],
            }

    pruned = [k for k, v in items.items() if _prunable(v, today)]
    for key in pruned:
        del items[key]
    if pruned:
        logger.info(f"Board state: pruned {len(pruned)} aged retirement(s)")

    state = {
        "items": items,
        "proposals": proposals,
        "last_run": notes[-1][0] if notes else None,
    }
    if save:
        save_state(vault_path, state)
    return state


def sync_board(board_path: Path, vault_path: Path) -> int:
    """Act on the ticked boxes of one board, then rebuild state. Idempotent.

    Ticks are appended to the decision log and the answered lines are rewritten
    into confirmations, so a second run finds nothing to do. State is projected
    from scratch afterwards — this function never edits it in place.

    Returns the number of items answered on this call (0 when the board is
    merely being recorded, or has already been fully processed).
    """
    if not board_path.exists():
        return 0
    text = board_path.read_text(encoding="utf-8")
    items = parse_items(text)
    if not items and not parse_proposals(text):
        return 0

    day = board_date(board_path)
    lines: list[str | None] = list(text.splitlines())
    answered = 0

    for key, item in items.items():
        # A ticked box is the command. The log is where it lands; the state
        # file is rebuilt from that log below, never written to here.
        if item["answer"] is None:
            continue
        append_feedback(
            vault_path,
            {
                "ts": now(),
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
                lines[pos] = None
        answered += 1

    answered += _process_proposals(lines, text, board_path, vault_path, day)
    # Read-only: the note is logged, never rewritten, so it never contributes
    # a reason to touch the file.
    _log_board_note(text, vault_path, day)

    if answered:
        board_path.write_text(
            "\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8"
        )
        logger.info(f"Board janitor: {answered} item(s) answered on {board_path.name}")

    # Project state from the board notes and the log — including this call's
    # appends. Nothing here depends on what the previous file happened to say.
    rebuild_state(vault_path)
    return answered


def is_reflection_note(path: Path) -> bool:
    """`OpenAugi/Research/Weekly Reflection - 2026-09-20.md`, and nothing else."""
    return (
        path.suffix == ".md"
        and path.parent.name == Path(REFLECTION_FOLDER).name
        and bool(_REFLECTION_STEM_RE.match(path.stem))
    )


def sync_reflection(note_path: Path, vault_path: Path) -> int:
    """Act on the ticked `do` / `no` boxes of one weekly reflection. Idempotent.

    The reflection carries proposals only — no three-box items, no lanes, no
    note channel — and it never touches `.board-state.json`: the board's state
    is projected from board notes and board log rows, and a Sunday tick is
    neither. An answered box is rewritten into a confirmation and its task
    file is never written twice, exactly as on a board.

    Returns the number of proposals answered on this call.
    """
    if not note_path.exists():
        return 0
    text = note_path.read_text(encoding="utf-8")
    if not parse_proposals(text):
        return 0
    day = board_date(note_path)
    lines: list[str | None] = list(text.splitlines())
    answered = _process_proposals(
        lines,
        text,
        note_path,
        vault_path,
        day,
        task_prefix="reflection",
        source=REFLECTION_SOURCE,
        task_source="weekly-reflection",
        origin="the weekly reflection",
    )
    if answered:
        note_path.write_text(
            "\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8"
        )
        logger.info(f"Board janitor: {answered} proposal(s) answered on {note_path.name}")
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
    """Janitor entry point for the watcher — any touched board or reflection."""
    total = 0
    for raw in changed_paths:
        path = Path(raw)
        if path.suffix == ".md" and path.parent.name == "Board" and "Board" in path.stem:
            try:
                total += sync_board(path, vault_path)
            except Exception as e:
                logger.error(f"Board janitor failed on {path}: {e}", exc_info=True)
        elif is_reflection_note(path):
            try:
                total += sync_reflection(path, vault_path)
            except Exception as e:
                logger.error(f"Board janitor failed on {path}: {e}", exc_info=True)
    return total
