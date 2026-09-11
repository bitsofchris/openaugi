"""Bring highlights back from Reader onto the note that produced them.

Highlights are themselves documents in `/api/v3/list/` — `category:
"highlight"`, a `parent_id` pointing at the document, `content` (the marked
span) and `notes` (the inline note). So the whole return leg is the Reader API
and nothing else.

The mapping is free: the parent's `source_url` is the fabricated URL we pushed,
which carries the sha8 of the vault path. Highlights land as a
`## Read in Reader — <date>` section *on the source note in `OpenAugi/`*, where
the idea already lives — one artifact per idea, not a detached mirror in a
reference folder.

The official Readwise plugin will also sync augi's own documents back into
`_private/2-Reference/Readwise/`, where the graph would classify augi's output
as external reading material. That is suppressed with an exclude glob in
`~/.openaugi/config.toml`, not here; see docs/reference/reading-queue.md.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openaugi.reading.note import key_from_url
from openaugi.reading.push import _LEDGER_META, COLLECTION
from openaugi.reading.reader_api import ReaderAPI

logger = logging.getLogger(__name__)

STATE_COLLECTION = "reading_queue_state"
STATE_ID = "harvest"
SECTION_PREFIX = "## Read in Reader — "


@dataclass
class HarvestResult:
    notes_updated: list[str] = field(default_factory=list)
    highlights_appended: int = 0
    unmatched_parents: list[str] = field(default_factory=list)
    """Highlight parents whose source_url is not one of ours — i.e. the user's
    own reading. Counted, never touched."""

    @property
    def summary(self) -> str:
        return (
            f"{self.highlights_appended} highlights → {len(self.notes_updated)} notes "
            f"· {len(self.unmatched_parents)} other documents ignored"
        )


def last_run(store: Any) -> str | None:
    rows = store.list_records(STATE_COLLECTION, limit=1)
    return rows[0].get("last_run") if rows else None


def harvest(
    vault: str | Path,
    store: Any,
    client: ReaderAPI,
    *,
    since: str | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> HarvestResult:
    """Append every new highlight on an augi-pushed document to its note."""
    moment = now or datetime.now()
    vault_path = Path(vault)
    window = since if since is not None else last_run(store)
    result = HarvestResult()

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for highlight in client.list_documents(category="highlight", updated_after=window):
        parent_id = highlight.get("parent_id")
        if parent_id:
            by_parent[str(parent_id)].append(highlight)

    ledger = {row["id"]: row for row in store.list_records(COLLECTION, limit=10_000)}

    for parent_id, highlights in by_parent.items():
        parent = next(iter(client.list_documents(document_id=parent_id)), None)
        key = key_from_url((parent or {}).get("source_url"))
        record = ledger.get(key) if key else None
        if not record:
            result.unmatched_parents.append(parent_id)
            continue

        seen = set(record.get("harvested_ids") or [])
        fresh = [h for h in _ordered(highlights) if str(h.get("id")) not in seen]
        if not fresh:
            continue

        note_path = vault_path / str(record.get("path", ""))
        if not _writable(note_path, vault_path):
            logger.warning("Refusing to write outside OpenAugi/: %s", note_path)
            continue

        if not dry_run:
            _append_section(note_path, fresh, moment)
            store.write_record(
                COLLECTION,
                str(key),
                {
                    **{k: v for k, v in record.items() if k not in _LEDGER_META},
                    "harvested_ids": sorted(seen | {str(h.get("id")) for h in fresh}),
                    "last_harvest": moment.isoformat(timespec="seconds"),
                    "reading_progress": (parent or {}).get("reading_progress"),
                },
                moment.isoformat(timespec="seconds"),
            )
        result.notes_updated.append(str(record.get("path")))
        result.highlights_appended += len(fresh)

    if not dry_run:
        store.write_record(
            STATE_COLLECTION,
            STATE_ID,
            {"last_run": moment.isoformat(timespec="seconds")},
            moment.isoformat(timespec="seconds"),
        )
    return result


def render_section(highlights: list[dict[str, Any]], when: datetime) -> str:
    """The markdown appended to a note. Kept separate so tests assert on shape."""
    lines = [f"{SECTION_PREFIX}{when.date().isoformat()}", ""]
    for highlight in highlights:
        content = (highlight.get("content") or "").strip()
        if not content:
            continue
        for line in content.splitlines():
            lines.append(f"> {line}".rstrip())
        note = (highlight.get("notes") or "").strip()
        if note:
            lines.append(f"> — note: {note}")
        lines.append("")
    return "\n".join(lines)


def _append_section(path: Path, highlights: list[dict[str, Any]], when: datetime) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    path.write_text(existing + separator + render_section(highlights, when), encoding="utf-8")


def _ordered(highlights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reader returns highlights unordered; `created_at` puts them back in the
    order they were marked, which is the order they were read."""
    return sorted(highlights, key=lambda h: (str(h.get("created_at") or ""), str(h.get("id"))))


def _writable(path: Path, vault: Path) -> bool:
    """The hard rule, enforced in code: agents only write under `OpenAugi/`."""
    try:
        return path.resolve().relative_to(vault.resolve()).parts[0] == "OpenAugi"
    except ValueError:
        return False
