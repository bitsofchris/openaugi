"""Routing janitor — applies a day's routing rows once the master box is ticked.

The Augi Log's `## Routing` section (see `route.py`) is answered with
checkboxes and `aaa:` lines, but nothing happens until the user ticks the
section's master box, `- [ ] process this log`. That tick IS the command
(the echo and board janitors' precedent). Then, for every row in the log:

- a ticked box is the answer; a non-empty `aaa:` line overrides it;
- an untouched row takes the bold suggestion when the ledger marked the
  proposal confident, otherwise `memory`, which writes nothing.

What each verb writes:

    file under / link   a `routed_to` link in the DB (the review-pass primitive)
    extend [[X]]        the block's text inserted into X, newest-first, plus the link
    new note            OpenAugi/Notes/<slug>.md holding the text, plus the link
    memory / hold       ledger state only

Every answered row is rewritten to a `✓` confirmation with an `undo` box, so
nothing is processed twice and every apply can be reversed. Every resolution
lands in `OpenAugi/Capture/feedback-log.ndjson` with the proposal, the choice
and the block's features — the history later proposals are biased by.

Extend is the only verb that touches one of the user's own notes. It is
append-only (an insert under a dated heading, wrapped in markers so undo
removes exactly what was inserted) and can be disabled with
`[routing] extend_writes_note = false`, which falls back to the link.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openaugi.model.link import Link
from openaugi.pipeline import augi_log, route
from openaugi.pipeline.writeback import aaa_re, append_feedback, box_re, now

logger = logging.getLogger(__name__)

NOTES_FOLDER = "OpenAugi/Notes"

_MASTER_RE = re.compile(r"^- \[(?P<mark>[ xX])\] process this log\s*$", re.MULTILINE)
_ROW_RE = re.compile(
    r"<!-- route:(?P<bid>[0-9a-f]+) -->\n(?P<body>.*?)(?=\n<!-- route:|\n## |\n<!-- heartbeat|\Z)",
    re.DOTALL,
)
_BOX_RE = box_re(suffix=True, multiline=True)
_UNDO_RE = re.compile(r"^- \[[xX]\] undo\s*$", re.MULTILINE)
_AAA_RE = aaa_re(multiline=True)
_LABEL_RE = re.compile(r"^(?P<verb>extend|link|file under) \[\[(?P<target>[^\]|]+)\]\]$")
_DATED_H3_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2})\b.*$", re.MULTILINE)
_ANY_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)
_JOURNAL_H1_RE = re.compile(r"^# Journal\s*$", re.MULTILINE)
_SOURCE_LINE_RE = re.compile(r"^\*\[\[(?P<note>[^\]]+)\]\]")


@dataclass
class Choice:
    verb: str
    target: str | None = None
    by: str = "you"  # you | aaa | auto


@dataclass
class Row:
    block_id: str
    body: str
    ticked: Choice | None = None
    aaa: str = ""
    resolved: bool = False
    undo: bool = False
    daily_note: str = ""
    boxes: list[str] = field(default_factory=list)


# ── Parsing ────────────────────────────────────────────────────────


def _parse_label(label: str) -> Choice | None:
    label = label.strip().strip("*").strip()
    if label in ("new note", "memory", "hold"):
        return Choice(label)
    if match := _LABEL_RE.match(label):
        return Choice(match.group("verb"), match.group("target"))
    return None


def _parse_aaa(text: str, registry: dict[str, route.Container], store) -> Choice | None:
    """An `aaa:` answer: a verb word and/or a [[target]] or a container title."""
    if not text:
        return None
    targets = route._WIKILINK_RE.findall(text)
    if not targets:
        lowered = text.lower()
        targets = sorted((t for t in registry if t.lower() in lowered), key=len, reverse=True)[:1]
    verb = route._verb_from(text, "")
    if verb in ("memory", "hold", "new note"):
        return Choice(verb, None, "aaa")
    if not targets:
        return Choice("hold", None, "aaa") if not verb else None
    target = targets[0]
    if not verb:
        verb = "file under" if target in registry else "extend"
    elif (
        verb == "file under" and target not in registry and route._note_title_exists(store, target)
    ):
        verb = "extend"
    return Choice(verb, target, "aaa")


def parse_rows(text: str, registry: dict[str, route.Container], store) -> list[Row]:
    rows: list[Row] = []
    for match in _ROW_RE.finditer(text):
        body = match.group("body")
        row = Row(block_id=match.group("bid"), body=body)
        if "- ✓ " in body:
            row.resolved = True
            row.undo = bool(_UNDO_RE.search(body))
        for line in body.splitlines():
            if src := _SOURCE_LINE_RE.match(line):
                row.daily_note = src.group("note")
        for box in _BOX_RE.finditer(body):
            row.boxes.append(box.group(0))
            if box.group("mark").lower() == "x" and row.ticked is None:
                row.ticked = _parse_label(box.group("label"))
        if aaa := _AAA_RE.search(body):
            row.aaa = aaa.group("reason")
        rows.append(row)
    if rows and registry is not None:
        for row in rows:
            if row.aaa and not row.resolved:
                override = _parse_aaa(row.aaa, registry, store)
                if override:
                    row.ticked = override
    return rows


# ── Applying ───────────────────────────────────────────────────────


def _doc_id(store, title: str) -> str | None:
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (title,),
    ).fetchone()
    return row[0] if row else None


def _doc_path(store, title: str) -> str | None:
    row = store.conn.execute(
        "SELECT json_extract(metadata, '$.source_path') FROM blocks "
        "WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (title,),
    ).fetchone()
    return row[0] if row else None


def _link(store, block_id: str, title: str) -> bool:
    target = _doc_id(store, title)
    if target is None:
        return False
    if store.get_contains_parent_id(block_id) == target:
        return True  # already home
    store.insert_links([Link(from_id=block_id, to_id=target, kind="routed_to")])
    return True


def _unlink(store, block_id: str, title: str) -> None:
    target = _doc_id(store, title)
    if target is not None:
        store.delete_link(block_id, target, "routed_to")


def _block_text(store, block_id: str) -> str | None:
    block = store.get_block(block_id)
    if block is None:
        return None
    return route._AAA_RE.sub("", block.content or "").strip()


def _wrapped(block_id: str, text: str, daily_note: str) -> str:
    src = f"\n— from [[{daily_note}]]" if daily_note else ""
    return f"<!-- augi:routed {block_id} -->\n{text}{src}\n<!-- /augi:routed -->\n"


def insert_extend(note_text: str, day: str, block_id: str, text: str, daily_note: str) -> str:
    """Insert under `### <day>`, newest-first, anchored on `# Journal` or the dated H3s.

    The rule that set this (2026-09-03): "look for the Journal H1 or the other H3s — I like the
    most recent entry to be on top."
    """
    payload = _wrapped(block_id, text, daily_note)
    headings = list(_DATED_H3_RE.finditer(note_text))
    for h in headings:
        if h.group(1) == day:
            # end of that day's section: before the next heading, or EOF
            nxt = _ANY_HEADING_RE.search(note_text, h.end())
            at = nxt.start() if nxt else len(note_text)
            head = note_text[:at].rstrip("\n") + "\n\n"
            tail = note_text[at:]
            return head + payload + ("\n" + tail if tail else "")
    section = f"### {day}\n\n{payload}\n"
    if journal := _JOURNAL_H1_RE.search(note_text):
        at = journal.end()
        return note_text[:at].rstrip("\n") + "\n\n" + section + note_text[at:].lstrip("\n")
    if headings:
        at = headings[0].start()
        return note_text[:at] + section + note_text[at:]
    return note_text.rstrip("\n") + "\n\n# Journal\n\n" + section


def remove_extend(note_text: str, block_id: str) -> str:
    pattern = re.compile(
        rf"<!-- augi:routed {re.escape(block_id)} -->\n.*?<!-- /augi:routed -->\n?", re.DOTALL
    )
    out = pattern.sub("", note_text)
    # a dated heading left with nothing under it goes too
    out = re.sub(r"^### \d{4}-\d{2}-\d{2}\s*\n(?=\s*(?:#|\Z))", "", out, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", out)


def _slug(text: str) -> str:
    topic = " ".join(text.split())[:60]
    slug = re.sub(r"[^\w\s-]", "", topic)
    return re.sub(r"[\s_]+", "-", slug).strip("-")[:50]


def _new_note(vault_path: Path, store, block_id: str, text: str, daily_note: str, day: str) -> str:
    title = _slug(text) or f"routed-{block_id[:8]}"
    path = vault_path / NOTES_FOLDER / f"{title}.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\ntype: document\n"
            f"description: Routed from the Augi Log on {day} as a new note.\n"
            f"created: {day}\n---\n\n"
            f"# {title}\n\n#human-review\n\n"
            f"### {day}\n\n{_wrapped(block_id, text, daily_note)}",
            encoding="utf-8",
        )
    return title


def apply_choice(
    store,
    vault_path: Path,
    row: Row,
    choice: Choice,
    day: str,
    settings: dict[str, Any],
) -> tuple[str, str]:
    """Do the write. Returns (confirmation label, ledger status)."""
    text = _block_text(store, row.block_id)
    if choice.verb in ("memory", "hold"):
        return choice.verb, "memory" if choice.verb == "memory" else "held"
    if text is None:
        return "block gone — nothing applied", "gone"
    target = choice.target or ""
    if choice.verb == "new note":
        title = _new_note(vault_path, store, row.block_id, text, row.daily_note, day)
        return f"new note → [[{title}]]", "applied"
    if choice.verb in ("file under", "link"):
        if not _link(store, row.block_id, target):
            return f"{choice.verb} [[{target}]] — note not found", "failed"
        return f"{choice.verb} [[{target}]]", "applied"
    if choice.verb == "extend":
        if not _link(store, row.block_id, target):
            return f"extend [[{target}]] — note not found", "failed"
        rel = _doc_path(store, target)
        if not settings.get("extend_writes_note", True) or not rel:
            return f"link [[{target}]] (extend disabled)", "applied"
        note = vault_path / rel
        try:
            note.write_text(
                insert_extend(
                    note.read_text(encoding="utf-8"), day, row.block_id, text, row.daily_note
                ),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Routing extend failed on {rel}: {e}")
            return f"link [[{target}]] (extend failed: {e})", "applied"
        return f"extend [[{target}]]", "applied"
    return f"{choice.verb} — unknown verb", "failed"


def undo_choice(store, vault_path: Path, row: Row, record: dict) -> str:
    choice = record.get("chosen") or {}
    verb, target = choice.get("verb"), choice.get("target")
    if verb in ("file under", "link", "extend") and target:
        _unlink(store, row.block_id, target)
    if verb == "extend" and target and (rel := _doc_path(store, target)):
        note = vault_path / rel
        if note.exists():
            note.write_text(
                remove_extend(note.read_text(encoding="utf-8"), row.block_id), encoding="utf-8"
            )
    if verb == "new note" and target:
        _unlink(store, row.block_id, target)
        return f"undone — [[{target}]] left in place, unlinked"
    return "undone"


# ── The pass ───────────────────────────────────────────────────────


def _record(store, block_id: str) -> dict:
    """The row's ledger record, or {} — records are keyed by block id."""
    row = store.conn.execute(
        "SELECT data FROM records WHERE collection = ? AND id = ?",
        (route.ROUTING_COLLECTION, block_id),
    ).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def _rewrite_row(text: str, row: Row, confirmation: str, undo: bool = True) -> str:
    """Replace the row's boxes with one ✓ line (and an undo box)."""
    body = row.body
    lines = [ln for ln in body.splitlines() if not _BOX_RE.match(ln)]
    lines = [ln for ln in lines if not _AAA_RE.match(ln) or ln.split(":", 1)[1].strip()]
    lines = [ln.rstrip() for ln in lines]
    while lines and not lines[-1]:
        lines.pop()
    lines.append("")
    lines.append(f"- ✓ {confirmation}")
    if undo:
        lines.append("- [ ] undo")
    lines.append("")
    return text.replace(body, "\n".join(lines))


def _signal(choice: Choice, top: dict | None) -> str:
    """What the answer teaches. `by` in the record says who chose it."""
    if choice.verb in ("memory", "hold"):
        return choice.verb
    if choice.by == "auto":
        return "auto"
    if top and top.get("verb") == choice.verb and top.get("target") == choice.target:
        return "accepted"
    return "corrected"


def process_log(log_path: Path, vault_path: Path, store, config: dict[str, Any]) -> int:
    """Act on one Augi Log's routing section. Returns actions taken."""
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8")
    if "<!-- route:" not in text:
        return 0
    settings = config.get("routing", {})
    registry = route.load_registry(store, vault_path)
    rows = parse_rows(text, registry, store)
    master = _MASTER_RE.search(text)
    day = augi_log.split(text)[0]
    day_match = re.search(r"# Augi Log — (\d{4}-\d{2}-\d{2})", day)
    day = day_match.group(1) if day_match else datetime.now(UTC).date().isoformat()
    actions = 0
    stamp = now()

    # Undo boxes work whether or not the log was processed already.
    for row in rows:
        if not (row.resolved and row.undo):
            continue
        record = _record(store, row.block_id)
        confirmation = undo_choice(store, vault_path, row, record)
        store.update_record(
            route.ROUTING_COLLECTION, row.block_id, {"status": "undone", "undone_at": stamp}, stamp
        )
        append_feedback(
            vault_path,
            {
                "ts": stamp,
                "source": "routing",
                "block_id": row.block_id,
                "day": day,
                "signal": "undo",
                "chosen": record.get("chosen"),
                "proposed": (record.get("proposed") or [None])[0],
                "features": record.get("features"),
            },
        )
        text = _rewrite_row(text, row, confirmation, undo=False)
        actions += 1

    if not master or master.group("mark").lower() != "x":
        if actions:
            log_path.write_text(text, encoding="utf-8")
        return actions

    for row in rows:
        if row.resolved:
            continue
        record = _record(store, row.block_id)
        proposed = record.get("proposed") or []
        top = proposed[0] if proposed else None
        choice = row.ticked
        if choice is None:
            if top and record.get("confident") and not record.get("memory"):
                choice = Choice(top["verb"], top.get("target"), "auto")
            else:
                choice = Choice("memory", None, "auto")
        label, status = apply_choice(store, vault_path, row, choice, day, settings)
        by = {"you": "you", "aaa": "your aaa:", "auto": "auto"}[choice.by]
        confirmation = f"{label} ({by})"
        chosen = {"verb": choice.verb, "target": choice.target}
        store.update_record(
            route.ROUTING_COLLECTION,
            row.block_id,
            {"status": status, "chosen": chosen, "by": choice.by, "resolved_at": stamp},
            stamp,
        )
        append_feedback(
            vault_path,
            {
                "ts": stamp,
                "source": "routing",
                "block_id": row.block_id,
                "day": day,
                "signal": _signal(choice, top),
                "by": choice.by,
                "proposed": top,
                "chosen": chosen,
                "features": record.get("features"),
            },
        )
        text = _rewrite_row(text, row, confirmation, undo=status == "applied")
        actions += 1

    text = _MASTER_RE.sub(f"- ✓ processed {stamp[:16].replace('T', ' ')}", text, count=1)
    store.update_record(
        route.ROUTING_COLLECTION,
        route.log_record_id(day),
        {"status": "processed", "processed_at": stamp},
        stamp,
    )
    log_path.write_text(text, encoding="utf-8")
    logger.info(f"Routing janitor: {actions} row(s) applied in {log_path.name}")
    return actions


def process_changed(
    changed_paths: set[str], vault_path: Path, store, config: dict[str, Any]
) -> int:
    """Janitor entry point for the watcher — handles any touched Augi Log."""
    total = 0
    for raw in changed_paths:
        path = Path(raw)
        if path.name == augi_log.LOG_NAME:
            try:
                total += process_log(path, vault_path, store, config)
            except Exception as e:
                logger.error(f"Routing janitor failed on {path}: {e}", exc_info=True)
    return total
