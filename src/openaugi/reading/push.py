"""Push flagged vault notes into Readwise Reader.

The gate is `reading_queue: true` in a note's frontmatter (note.py). This
module adds the two things that keep the queue from becoming another dead
inbox:

- **A cap.** Two pushes a day, default. A reading queue that grows faster than
  it is read is the failure mode of every queue ever abandoned.
- **Idempotence.** The fabricated `url` is stable per note, so re-saving the
  same note updates the Reader document in place. We skip unchanged notes
  entirely (content hash in the ledger) and re-push edited ones.

The ledger is the `reading_queue` records collection (docs/reference/records.md):
one row per note, keyed by sha8, holding the vault path, the last pushed
content hash and the Reader id. It is machinery — delete it and the only cost
is one redundant re-push per note.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openaugi.reading.note import TITLE_PREFIX, ReadingNote, load_note, to_html
from openaugi.reading.reader_api import ReaderAPI

logger = logging.getLogger(__name__)

COLLECTION = "reading_queue"
DEFAULT_CAP = 2
AGENT_FOLDER = "OpenAugi"
# Agent scratch and machinery: never reading material even if flagged by hand.
SKIP_FOLDERS = {"Archive", "Tasks", "Sessions", "Compiled", "Context", "render"}


@dataclass
class PushResult:
    """What one run did, in the shape the CLI and the board line need."""

    pushed: list[str] = field(default_factory=list)
    skipped_unchanged: list[str] = field(default_factory=list)
    deferred_over_cap: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"pushed {len(self.pushed)} · unchanged {len(self.skipped_unchanged)} "
            f"· over cap {len(self.deferred_over_cap)} · failed {len(self.failed)}"
        )


def find_flagged_notes(vault: str | Path) -> list[ReadingNote]:
    """Every note under `OpenAugi/` carrying `reading_queue: true`.

    Sorted by path so a capped run is deterministic rather than
    filesystem-order.
    """
    vault_path = Path(vault)
    root = vault_path / AGENT_FOLDER
    if not root.is_dir():
        return []

    notes: list[ReadingNote] = []
    for path in sorted(root.rglob("*.md")):
        if SKIP_FOLDERS & set(path.relative_to(root).parts[:-1]):
            continue
        try:
            note = load_note(path, vault_path)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if note.flagged:
            notes.append(note)
    return notes


def build_payload(note: ReadingNote, tags: list[str] | None = None) -> dict[str, Any]:
    """The `POST /save/` body for one note.

    `location: "later"` on purpose — augi's output goes into the queue, not to
    the front of it, so it never jumps the things the user deliberately saved.
    """
    payload: dict[str, Any] = {
        "url": note.url,
        "html": to_html(note.body),
        "title": f"{TITLE_PREFIX}{note.title}",
        "author": "augi",
        "category": "article",
        "location": "later",
        "tags": tags or ["augi"],
    }
    if note.description:
        payload["summary"] = note.description
    return payload


def push_notes(
    vault: str | Path,
    store: Any,
    client: ReaderAPI | None = None,
    *,
    cap: int = DEFAULT_CAP,
    dry_run: bool = False,
    now: datetime | None = None,
) -> PushResult:
    """Push flagged, changed notes — at most `cap` of them.

    `client` may be None only when `dry_run` is set; that is the mode the
    30-minute taste test uses to see exactly what would ship.
    """
    moment = now or datetime.now()
    today = moment.date().isoformat()
    result = PushResult()

    ledger = {row["id"]: row for row in store.list_records(COLLECTION, limit=10_000)}
    remaining = max(0, cap - sum(1 for r in ledger.values() if _pushed_on(r) == today))

    for note in find_flagged_notes(vault):
        existing = ledger.get(note.key)
        if existing and existing.get("content_hash") == note.content_hash:
            result.skipped_unchanged.append(note.rel_path)
            continue
        if remaining <= 0:
            result.deferred_over_cap.append(note.rel_path)
            continue

        if dry_run:
            result.pushed.append(note.rel_path)
            remaining -= 1
            continue

        if client is None:
            raise ValueError("push_notes needs a client unless dry_run is set")
        try:
            saved = client.save(build_payload(note))
        except Exception as exc:  # noqa: BLE001 — one bad note must not kill the run
            logger.warning("Push failed for %s: %s", note.rel_path, exc)
            result.failed.append((note.rel_path, str(exc)))
            continue

        store.write_record(
            COLLECTION,
            note.key,
            {
                **{k: v for k, v in (existing or {}).items() if k not in _LEDGER_META},
                "path": note.rel_path,
                "title": note.title,
                "content_hash": note.content_hash,
                "pushed_at": moment.isoformat(timespec="seconds"),
                "reader_id": saved.get("id"),
                "reader_url": saved.get("url"),
            },
            moment.isoformat(timespec="seconds"),
        )
        result.pushed.append(note.rel_path)
        remaining -= 1

    return result


# Fields the store adds on read; carrying them back in would duplicate them.
_LEDGER_META = {"id", "created_at", "updated_at"}


def _pushed_on(record: dict[str, Any]) -> str:
    pushed_at = record.get("pushed_at")
    return pushed_at[:10] if isinstance(pushed_at, str) else ""
