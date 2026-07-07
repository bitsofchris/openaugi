"""Context pack — the machine-readable sidecar for mobile capture assist.

Part of the M3 read contract (docs/plans/master-plan.md): the review pass
emits `<vault>/OpenAugi/context-pack.json`; the mobile bridge serves it to
the phone at `GET /context-pack`. The shape is pinned by the mobile repo's
`shared/contract.ts` `ContextPack` type:

    { agentFile: str, taxonomy: [str],
      recentConcepts: [{title, path}], noteTitles: [str] }

Extra keys (`generatedAt`) are additive-safe for the TS consumer.

Sources, in trust order:
- taxonomy: curated inline tags from `OpenAugi/AGENT/My Taxonomy.md`,
  then top-used DB tags appended (capped) — curation first, usage second.
- recentConcepts: route targets (AMOC/PMOC containers) by most recent
  routing activity — this is what the DB knows that a vault glob doesn't.
- noteTitles: containers first, then documents by recency (capped).
- agentFile: `OpenAugi/AGENT/capture-conventions.md` if present, else a
  built-in default.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

OUTPUT_RELPATH = "OpenAugi/context-pack.json"
TAXONOMY_NOTE = "OpenAugi/AGENT/My Taxonomy.md"
CONVENTIONS_NOTE = "OpenAugi/AGENT/capture-conventions.md"

MAX_TAXONOMY = 40
MAX_RECENT_CONCEPTS = 5
MAX_NOTE_TITLES = 500

# Like the mobile bridge's taxonomy parser (server/vaultContext.ts), plus
# backtick — the real taxonomy note writes tags as `#status/active`.
_INLINE_TAG_RE = re.compile(r"(?:^|[\s(`])#([\w/-]+)")

DEFAULT_AGENT_FILE = (
    "# OpenAugi capture conventions\n\n"
    "Tag thoughts with facet tags (`#area/*`, `#todo`, `#idea`, `#question`).\n"
    "Grammar: `qqq` splits blocks · `zzz:` dispatches a task · `aaa:` gives the\n"
    "review pass a routing/parsing instruction. Keep each capture to one idea."
)


def _parse_taxonomy_note(vault: Path) -> list[str]:
    """Unique inline #tags from the curated taxonomy note, in order. [] if absent."""
    note = vault / TAXONOMY_NOTE
    if not note.is_file():
        return []
    tags: list[str] = []
    for m in _INLINE_TAG_RE.finditer(note.read_text()):
        tag = f"#{m.group(1)}"
        if tag not in tags:
            tags.append(tag)
    return tags


def _read_agent_file(vault: Path) -> str:
    note = vault / CONVENTIONS_NOTE
    if note.is_file():
        return note.read_text()
    return DEFAULT_AGENT_FILE


def build_context_pack(store: SQLiteStore, vault_path: str | Path) -> dict:
    """Assemble the ContextPack dict from the DB + curated vault notes."""
    vault = Path(vault_path)

    # Taxonomy: curated note first, then top-used DB tags not already listed.
    taxonomy = _parse_taxonomy_note(vault)
    seen_tags = set(taxonomy)
    for tag in store.get_tag_details(limit=MAX_TAXONOMY):
        if len(taxonomy) >= MAX_TAXONOMY:
            break
        name = tag.get("tag_name") or ""
        if not name:
            continue
        hashed = name if name.startswith("#") else f"#{name}"
        if hashed not in seen_tags:
            taxonomy.append(hashed)
            seen_tags.add(hashed)

    # Recent concepts: containers by most recent routing activity.
    containers = store.get_recent_route_targets(limit=MAX_RECENT_CONCEPTS)
    recent_concepts = [
        {"title": b.title or "", "path": b.metadata.get("source_path", "")}
        for b in containers
        if b.title
    ]

    # Note titles: containers first, then documents by recency. Task files
    # are ephemeral dispatch artifacts — not useful wikilink targets.
    titles = [c["title"] for c in recent_concepts]
    seen_titles = set(titles)
    for doc in store.get_recent_documents(limit=MAX_NOTE_TITLES):
        if len(titles) >= MAX_NOTE_TITLES:
            break
        if not doc.title or doc.title in seen_titles:
            continue
        if doc.metadata.get("source_path", "").startswith("OpenAugi/Tasks/"):
            continue
        titles.append(doc.title)
        seen_titles.add(doc.title)

    return {
        "agentFile": _read_agent_file(vault),
        "taxonomy": taxonomy,
        "recentConcepts": recent_concepts,
        "noteTitles": titles,
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_context_pack(store: SQLiteStore, vault_path: str | Path) -> Path:
    """Write `<vault>/OpenAugi/context-pack.json`. Returns the output path."""
    vault = Path(vault_path)
    pack = build_context_pack(store, vault)
    out = vault / OUTPUT_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote context pack: %s (%d tags, %d concepts, %d titles)",
        out,
        len(pack["taxonomy"]),
        len(pack["recentConcepts"]),
        len(pack["noteTitles"]),
    )
    return out
