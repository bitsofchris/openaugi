"""Pipeline runner — orchestrates Layer 0 and Layer 1 processing.

Layer 0 (FREE): ingest → split → extract tags/links/dates → dedup → FTS
Layer 1 (near-free): embed → hub scoring (query-time)

See docs/plans/m0.md § Pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openaugi.adapters.vault import parse_vault_incremental
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


def run_layer0(
    vault_path: str | Path,
    store: SQLiteStore,
    exclude_patterns: list[str] | None = None,
    max_workers: int = 4,
) -> dict:
    """Run Layer 0 pipeline: ingest vault → blocks + links → store.

    Supports incremental ingestion via file-level content hashing.
    """
    # Get known document hashes for change detection
    known_hashes = _get_known_doc_hashes(store)
    logger.info(f"Tracking {len(known_hashes)} previously ingested files")

    # Parse vault (incremental)
    blocks, links, current_hashes, deleted_paths = parse_vault_incremental(
        vault_path=vault_path,
        known_doc_hashes=known_hashes,
        exclude_patterns=exclude_patterns,
        max_workers=max_workers,
    )

    # Handle deleted files — remove their document blocks (CASCADE cleans links)
    deleted_count = 0
    for rel_path in deleted_paths:
        from openaugi.model.block import Block

        doc_id = Block.make_document_id(rel_path)
        if store.delete_block(doc_id):
            deleted_count += 1

    if deleted_count:
        logger.info(f"Removed {deleted_count} deleted document blocks")

    # Delete existing blocks for changed files before re-inserting
    changed_paths = set()
    for b in blocks:
        if b.kind == "document" and b.metadata.get("source_path"):
            changed_paths.add(b.metadata["source_path"])

    for rel_path in changed_paths:
        from openaugi.model.block import Block

        doc_id = Block.make_document_id(rel_path)
        store.delete_block(doc_id)

    # Insert new blocks and links
    if blocks:
        block_count = store.insert_blocks(blocks)
        logger.info(f"Inserted {block_count} blocks")

    if links:
        # Filter links to only those where both endpoints exist
        valid_block_ids = {b.id for b in blocks}
        # Also include existing blocks in store
        existing_ids = set()
        for link in links:
            for bid in (link.from_id, link.to_id):
                if bid not in valid_block_ids and store.get_block(bid):
                    existing_ids.add(bid)

        valid_ids = valid_block_ids | existing_ids
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
        "blocks_inserted": len(blocks),
        "links_inserted": len(links),
        "files_changed": len(changed_paths),
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
