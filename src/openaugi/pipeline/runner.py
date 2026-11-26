"""Pipeline runner — orchestrates Layer 0 and Layer 1 processing.

Layer 0 (FREE): ingest → split → extract tags/links/dates → dedup → FTS
Layer 1 (near-free): embed → hub scoring (query-time)

Incremental ingestion uses two levels of hashing:
- Level 1 (file): skip entirely if file content hash unchanged
- Level 2 (block): within a changed file, diff entry content hashes —
  keep unchanged entries (preserving embeddings), insert new, delete removed

See docs/plans/m0.md (in git history) § Incremental Ingestion Strategy.
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
    source_rules: dict[str, str] | None = None,
    provenance_rules: dict[str, str] | None = None,
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
        provenance_rules=provenance_rules,
    )

    # Handle deleted files: the document and the entries it contains. The
    # document is found by its stored source_path, not by hashing the path (a
    # note with an augi_id is keyed on that). A document this run re-parsed
    # under another path was moved or renamed, not deleted: it keeps its
    # identity and its edges, and the diff below records the new path.
    parsed_doc_ids = {b.id for b in blocks if b.kind == "context_block:document"}
    ids_by_path = store.get_document_ids_by_source_path() if deleted_paths else {}
    deleted_count = 0
    for rel_path in deleted_paths:
        for doc_id in ids_by_path.get(rel_path, []):
            if doc_id in parsed_doc_ids:
                continue
            if store.delete_document(doc_id):
                deleted_count += 1
    if deleted_count:
        logger.info(f"Removed {deleted_count} deleted documents and their entries")

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

    _fold_rekeyed_documents(store, doc_blocks)

    # Block-level incremental: diff entries within each changed file
    blocks_to_insert: list[Block] = []
    # Entries deleted this cycle. An edit is a delete + an insert, so this is
    # the only place the pipeline can still see a block's predecessor — the
    # zzz dispatch hook needs it to tell "instruction was edited" apart from
    # "new instruction written". See pipeline/dispatch.py.
    removed_entries: list[Block] = []
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

            # Entries to keep (hash in both old and new) — don't touch them.
            # A moved note (same augi_id, new path) keeps none: an entry's id
            # and metadata carry its path, so every entry is re-created.
            moved = existing_doc.metadata.get("source_path") != source_path
            if moved:
                store.update_document_path(doc_id, source_path, doc_block.title)
            kept_hashes = set() if moved else old_hashes & new_hashes
            blocks_kept += len(kept_hashes)

            # Entries to remove (hash in old but not new). An edited block IS
            # a removal + an insertion — its agent state (routed_to links,
            # augi_tags) CASCADEs away by design: the new block re-enters the
            # review queue and the next pass re-decides (aaa: instructions in
            # the text carry the durable human intent). DB is a projection;
            # decisions are re-derived, never fuzzily migrated.
            removed_hashes = old_hashes - kept_hashes
            dropped_routes = 0
            for old_entry in old_entries:
                if old_entry.content_hash in removed_hashes:
                    dropped_routes += store.conn.execute(
                        "SELECT COUNT(*) FROM links WHERE from_id = ? AND kind = 'routed_to'",
                        (old_entry.id,),
                    ).fetchone()[0]
                    store.delete_block(old_entry.id)
                    removed_entries.append(old_entry)
                    blocks_removed += 1
            if dropped_routes:
                logger.info(
                    "%s: %d routed_to link(s) dropped with edited/removed blocks — "
                    "successors re-enter the review queue",
                    source_path,
                    dropped_routes,
                )

            # Entries to add (hash in new but not old)
            added_entries = [e for e in new_entries if e.content_hash not in kept_hashes]
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
        valid_links: list = []
        skipped_links: list = []
        for lnk in links:
            ok = lnk.from_id in valid_ids and lnk.to_id in valid_ids
            (valid_links if ok else skipped_links).append(lnk)

        link_count = store.insert_links(valid_links)
        logger.info(f"Inserted {link_count} links")
        _report_skipped_links(skipped_links, set(current_hashes))

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
        "removed_data_blocks": [b for b in removed_entries if b.kind == "data_block"],
    }


def _report_skipped_links(skipped: list, vault_paths: set[str]) -> dict[str, int]:
    """Say out loud which edges the valid-endpoint filter dropped.

    One kind of drop is expected: a wikilink whose target is not a note in
    the vault (`vault_paths`, every included file). Every other drop means an
    edge was keyed by one rule while its endpoint was keyed by another, and
    the graph silently lost it; that is a WARNING, with examples. Returns the
    two counts (for tests).
    """
    stems = {Path(p).stem for p in vault_paths}
    unresolved: list = []
    unexpected: list = []
    for lnk in skipped:
        title = lnk.metadata.get("target_title", "")
        expected = (
            lnk.kind == "links_to" and title not in stems and f"{title}.md" not in vault_paths
        )
        (unresolved if expected else unexpected).append(lnk)
    if unresolved:
        logger.info("%d wikilink(s) to notes not in the vault — no edge stored", len(unresolved))
    if unexpected:
        logger.warning(
            "%d edge(s) dropped: an endpoint is missing from the store (e.g. %s). "
            "Run `openaugi check-graph` to see where the graph drifted.",
            len(unexpected),
            ", ".join(f"{lnk.kind} {lnk.from_id}->{lnk.to_id}" for lnk in unexpected[:3]),
        )
    return {"unresolved": len(unresolved), "unexpected": len(unexpected)}


def _fold_rekeyed_documents(store: SQLiteStore, doc_blocks: dict[str, Block]) -> None:
    """Fold a changed note's old document into its new one when its id changed.

    A note that gains (or loses) an `augi_id` gets a new document id while its
    old document still sits under the same source_path. Left alone, the old
    one keeps every edge into the note (backlinks from notes that did not
    change and so are never re-parsed, routes, recaps) and title lookups
    find two documents. Folding it in before the diff moves those edges and
    lets the diff below see, and retire, the old entries.
    """
    if not doc_blocks:
        return
    ids_by_path = store.get_document_ids_by_source_path()
    for source_path, doc_block in doc_blocks.items():
        stale = [i for i in ids_by_path.get(source_path, []) if i != doc_block.id]
        if not stale:
            continue
        if not store.get_block(doc_block.id):
            store.insert_blocks([doc_block])
        for old_id in stale:
            moved = store.merge_document(old_id, doc_block.id)
            logger.info(
                "%s: document re-keyed %s -> %s, edges moved: %s",
                source_path,
                old_id,
                doc_block.id,
                moved,
            )


def _get_known_doc_hashes(store: SQLiteStore) -> dict[str, str]:
    """Get {relative_path: content_hash} for all document blocks."""
    docs = store.get_blocks_by_kind("context_block:document", limit=100_000)
    return {
        b.metadata.get("source_path", ""): b.content_hash
        for b in docs
        if b.content_hash and b.metadata.get("source_path")
    }


def run_layer0_from_config(vault_path: str | Path, store: SQLiteStore, config: dict) -> dict:
    """run_layer0 with the config's [vault] rules — what every caller passes.

    Shared by `openaugi ingest`, `openaugi up` and `openaugi kb`'s sync, so an
    index refreshed from any of them honours the same exclude/source/provenance
    rules.
    """
    vault_cfg = config.get("vault", {})
    return run_layer0(
        vault_path,
        store,
        exclude_patterns=vault_cfg.get("exclude_patterns"),
        max_workers=vault_cfg.get("max_workers", 4),
        source_rules=vault_cfg.get("source_rules"),
        provenance_rules=vault_cfg.get("provenance_rules"),
    )
