"""Pipeline runner — orchestrates Layer 0 and Layer 1 processing.

Layer 0 (FREE): ingest → split → extract tags/links/dates → dedup → FTS
Layer 1 (near-free): embed → hub scoring (query-time)

Incremental ingestion uses two levels of hashing:
- Level 1 (file): skip entirely if file content hash unchanged
- Level 2 (block): within a changed file, diff entry content hashes —
  keep unchanged entries (preserving embeddings), insert new, delete removed

See docs/plans/m0.md § Incremental Ingestion Strategy.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path

from openaugi.adapters.vault import parse_vault_incremental
from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

# Minimum content similarity (difflib ratio) for an edited block to inherit
# the old block's routing and agent classification. Below this, the edit is
# treated as a genuinely new block and agent state dies with the old one.
IDENTITY_MATCH_RATIO = 0.5


def run_layer0(
    vault_path: str | Path,
    store: SQLiteStore,
    exclude_patterns: list[str] | None = None,
    max_workers: int = 4,
    source_rules: dict[str, str] | None = None,
) -> dict:
    """Run Layer 0 pipeline: ingest vault → blocks + links → store.

    Two-level incremental ingestion:
    1. File-level: skip unchanged files entirely
    2. Block-level: within changed files, diff entries by content_hash —
       unchanged entries keep their embeddings, only new/modified get re-embedded
    """
    known_hashes = _get_known_doc_hashes(store)
    logger.info(f"Tracking {len(known_hashes)} previously ingested files")

    # Parse vault (file-level incremental — only parses changed/new files)
    blocks, links, current_hashes, deleted_paths = parse_vault_incremental(
        vault_path=vault_path,
        known_doc_hashes=known_hashes,
        exclude_patterns=exclude_patterns,
        max_workers=max_workers,
        source_rules=source_rules,
    )

    # Handle deleted files — CASCADE removes entries and their links
    deleted_count = 0
    for rel_path in deleted_paths:
        doc_id = Block.make_document_id(rel_path)
        if store.delete_block(doc_id):
            deleted_count += 1
    if deleted_count:
        logger.info(f"Removed {deleted_count} deleted document blocks")

    # Group new blocks by document for block-level diffing
    doc_blocks: dict[str, Block] = {}  # source_path → document block
    entry_blocks_by_doc: dict[str, list[Block]] = {}  # source_path → entry blocks
    tag_blocks: list[Block] = []
    other_blocks: list[Block] = []

    for b in blocks:
        if b.kind == "context_block:document":
            source_path = b.metadata.get("source_path", "")
            doc_blocks[source_path] = b
        elif b.kind == "data_block":
            source_path = b.metadata.get("source_path", "")
            entry_blocks_by_doc.setdefault(source_path, []).append(b)
        elif b.kind == "context_block:tag":
            tag_blocks.append(b)
        else:
            other_blocks.append(b)

    # Block-level incremental: diff entries within each changed file
    blocks_to_insert: list[Block] = []
    migrations: list[dict] = []  # agent state carried across block edits
    blocks_kept = 0
    blocks_removed = 0
    blocks_added = 0

    for source_path, doc_block in doc_blocks.items():
        doc_id = doc_block.id
        new_entries = entry_blocks_by_doc.get(source_path, [])
        new_hashes = {e.content_hash for e in new_entries}

        # Check if document already exists (changed file vs new file)
        existing_doc = store.get_block(doc_id)
        if existing_doc:
            # Changed file — diff entries
            old_entries = store.get_entries_for_document(doc_id)
            old_hashes = {e.content_hash for e in old_entries}

            # Entries to keep (hash in both old and new) — don't touch them
            kept_hashes = old_hashes & new_hashes
            blocks_kept += len(kept_hashes)

            # Entries to remove (hash in old but not new)
            removed_hashes = old_hashes - new_hashes
            removed_entries = [e for e in old_entries if e.content_hash in removed_hashes]
            added_entries = [e for e in new_entries if e.content_hash not in kept_hashes]

            # Block identity is a content hash, so an edit = delete old +
            # insert new — which would CASCADE away agent state (routed_to
            # links, augi_tags). Match removed→added by content similarity
            # and carry that state over BEFORE deleting.
            migrations.extend(_collect_identity_migrations(store, removed_entries, added_entries))

            for old_entry in removed_entries:
                store.delete_block(old_entry.id)
                blocks_removed += 1

            # Entries to add (hash in new but not old)
            blocks_to_insert.extend(added_entries)
            blocks_added += len(added_entries)

            # Update document block's file hash
            if doc_block.content_hash:
                store.update_block_hash(doc_id, doc_block.content_hash)
        else:
            # New file — insert everything
            blocks_to_insert.append(doc_block)
            blocks_to_insert.extend(new_entries)
            blocks_added += len(new_entries)

    # Insert tag blocks (always idempotent via INSERT OR IGNORE)
    blocks_to_insert.extend(tag_blocks)
    blocks_to_insert.extend(other_blocks)

    if blocks_to_insert:
        count = store.insert_blocks(blocks_to_insert)
        logger.info(f"Inserted {count} blocks")

    # Re-attach migrated agent state now that the new block rows exist
    if migrations:
        _apply_identity_migrations(store, migrations)

    if blocks_kept or blocks_removed:
        logger.info(
            f"Block-level diff: {blocks_kept} kept, {blocks_added} added, {blocks_removed} removed"
        )

    # Insert links — filter to valid endpoints
    if links:
        all_block_ids = {b.id for b in blocks_to_insert}
        # Also need IDs of kept blocks (they exist in store but weren't re-inserted)
        # Plus any other existing blocks that links point to
        existing_ids: set[str] = set()
        for link in links:
            for bid in (link.from_id, link.to_id):
                if bid not in all_block_ids and store.get_block(bid):
                    existing_ids.add(bid)

        valid_ids = all_block_ids | existing_ids
        valid_links = [lnk for lnk in links if lnk.from_id in valid_ids and lnk.to_id in valid_ids]

        link_count = store.insert_links(valid_links)
        logger.info(
            f"Inserted {link_count} links "
            f"({len(links) - len(valid_links)} skipped — missing endpoints)"
        )

    stats = store.get_stats()
    logger.info(f"Store: {stats['total_blocks']} blocks, {stats['total_links']} links")

    # Collect newly added data_block blocks for post-ingest hooks (e.g. zzz dispatch)
    new_data_blocks = [b for b in blocks_to_insert if b.kind == "data_block"]

    return {
        "blocks_inserted": len(blocks_to_insert),
        "blocks_kept": blocks_kept,
        "blocks_removed": blocks_removed,
        "blocks_added": blocks_added,
        "links_inserted": len(links),
        "files_changed": len(doc_blocks),
        "files_deleted": len(deleted_paths),
        "stats": stats,
        "new_data_blocks": new_data_blocks,
    }


def _collect_identity_migrations(
    store: SQLiteStore,
    removed_entries: list[Block],
    added_entries: list[Block],
) -> list[dict]:
    """Match removed blocks to their edited successors and collect agent state.

    Only removed blocks that HAVE agent state (routed_to links or augi_tags)
    are considered — everything else re-derives from text. Matching is greedy
    best-first on difflib content ratio ≥ IDENTITY_MATCH_RATIO; a removed
    block with no similar successor is a real deletion and its state dies.

    Returns migration dicts: {new_id, routed_to: [container_ids], augi_tags}.
    Collected BEFORE the old rows are deleted (CASCADE drops their links).
    """
    if not removed_entries or not added_entries:
        return []

    # Agent state per removed block
    stateful: list[tuple[Block, list[str], list]] = []
    for old in removed_entries:
        routes = [
            r[0]
            for r in store.conn.execute(
                "SELECT to_id FROM links WHERE from_id = ? AND kind = 'routed_to'",
                (old.id,),
            ).fetchall()
        ]
        augi_tags = old.metadata.get("augi_tags") or []
        if routes or augi_tags:
            stateful.append((old, routes, augi_tags))

    if not stateful:
        return []

    # Greedy best-first content matching, one-to-one
    pairs: list[tuple[float, int, int]] = []
    for i, (old, _r, _t) in enumerate(stateful):
        for j, new in enumerate(added_entries):
            ratio = difflib.SequenceMatcher(None, old.content or "", new.content or "").ratio()
            if ratio >= IDENTITY_MATCH_RATIO:
                pairs.append((ratio, i, j))
    pairs.sort(reverse=True)

    migrations: list[dict] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for _ratio, i, j in pairs:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        old, routes, augi_tags = stateful[i]
        migrations.append(
            {"new_id": added_entries[j].id, "routed_to": routes, "augi_tags": augi_tags}
        )
        logger.info(
            "Block edit detected: migrating agent state %s → %s (%d routes, %d augi_tags)",
            old.id,
            added_entries[j].id,
            len(routes),
            len(augi_tags),
        )
    return migrations


def _apply_identity_migrations(store: SQLiteStore, migrations: list[dict]) -> None:
    """Re-attach migrated routed_to links and augi_tags to the new block rows."""
    links = [
        Link(from_id=m["new_id"], to_id=container_id, kind="routed_to")
        for m in migrations
        for container_id in m["routed_to"]
    ]
    if links:
        store.insert_links(links)
    for m in migrations:
        if m["augi_tags"]:
            store.conn.execute(
                """UPDATE blocks
                   SET metadata = json_set(COALESCE(metadata, '{}'), '$.augi_tags', json(?))
                   WHERE id = ?""",
                (json.dumps(m["augi_tags"]), m["new_id"]),
            )
    store.conn.commit()
    logger.info("Applied %d identity migrations", len(migrations))


def _get_known_doc_hashes(store: SQLiteStore) -> dict[str, str]:
    """Get {relative_path: content_hash} for all document blocks."""
    docs = store.get_blocks_by_kind("context_block:document", limit=100_000)
    return {
        b.metadata.get("source_path", ""): b.content_hash
        for b in docs
        if b.content_hash and b.metadata.get("source_path")
    }
