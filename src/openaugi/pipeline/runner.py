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

import logging
from pathlib import Path

from openaugi.adapters.vault import parse_vault_incremental
from openaugi.model.block import Block
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


def run_layer0(
    vault_path: str | Path,
    store: SQLiteStore,
    exclude_patterns: list[str] | None = None,
    max_workers: int = 4,
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
        if b.kind == "document":
            source_path = b.metadata.get("source_path", "")
            doc_blocks[source_path] = b
        elif b.kind == "entry":
            source_path = b.metadata.get("source_path", "")
            entry_blocks_by_doc.setdefault(source_path, []).append(b)
        elif b.kind == "tag":
            tag_blocks.append(b)
        else:
            other_blocks.append(b)

    # Block-level incremental: diff entries within each changed file
    blocks_to_insert: list[Block] = []
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
            for old_entry in old_entries:
                if old_entry.content_hash in removed_hashes:
                    store.delete_block(old_entry.id)
                    blocks_removed += 1

            # Entries to add (hash in new but not old)
            for new_entry in new_entries:
                if new_entry.content_hash not in kept_hashes:
                    blocks_to_insert.append(new_entry)
                    blocks_added += 1

            # Update document block's file hash
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

    if blocks_kept or blocks_removed:
        logger.info(
            f"Block-level diff: {blocks_kept} kept, {blocks_added} added, "
            f"{blocks_removed} removed"
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
        valid_links = [
            lnk for lnk in links
            if lnk.from_id in valid_ids and lnk.to_id in valid_ids
        ]

        link_count = store.insert_links(valid_links)
        logger.info(
            f"Inserted {link_count} links "
            f"({len(links) - len(valid_links)} skipped — missing endpoints)"
        )

    stats = store.get_stats()
    logger.info(
        f"Store: {stats['total_blocks']} blocks, {stats['total_links']} links"
    )

    return {
        "blocks_inserted": len(blocks_to_insert),
        "blocks_kept": blocks_kept,
        "blocks_removed": blocks_removed,
        "blocks_added": blocks_added,
        "links_inserted": len(links),
        "files_changed": len(doc_blocks),
        "files_deleted": len(deleted_paths),
        "stats": stats,
    }


def _get_known_doc_hashes(store: SQLiteStore) -> dict[str, str]:
    """Get {relative_path: content_hash} for all document blocks."""
    docs = store.get_blocks_by_kind("document", limit=100_000)
    return {
        b.metadata.get("source_path", ""): b.content_hash
        for b in docs
        if b.content_hash and b.metadata.get("source_path")
    }
