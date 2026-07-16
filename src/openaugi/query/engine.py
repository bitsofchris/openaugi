"""The query engine — deterministic read semantics, full blocks, no transport.

Every *rule* that used to live inline in mcp/server.py tool bodies executes
here: mode dispatch, the after_ingested bound, the has_task + bronze filter,
path exclusion, reference-document grouping, semantic overfetch. Adapters
(MCP, HTTP, CLI) shape the results — truncation, docstrings, envelopes are
presentation and stay out of this module.

Import rule (the boundary test): this module imports store/models/config —
never `mcp.*` or any web framework.

Results carry full `Block` models plus envelope metadata whose quirks
(pre-filter `has_more` in browse mode, the semantic k+1 break) are
deliberately preserved — the golden harness (tests/test_query_golden.py)
pins the MCP wire format built on top of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from openaugi.model.block import Block
from openaugi.store.sqlite import normalize_utc_timestamp

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

# User-demoted scaffolding (mobile curation demote). Stored without the `#`,
# like every parsed tag. See [layers] bronze_weight in config.py.
BRONZE_TAG = "layer/bronze"


class EmptyQuerySpec(ValueError):
    """Raised when a spec has no text mode and no filter — nothing to run."""


class RunResult(BaseModel):
    """Engine output for `run` — full blocks + envelope metadata.

    `blocks` is post-filter, cut to k, in rank order. `scores` is populated
    in semantic mode only ({block_id: similarity}). Browse mode adds
    `total` (pre-pagination SQL count), `next_offset`, and the collapsed
    `reference_documents` groups.
    """

    mode: str
    blocks: list[Block]
    scores: dict[str, float] = Field(default_factory=dict)
    has_more: bool = False
    total: int | None = None
    next_offset: int | None = None
    reference_documents: list[dict] = Field(default_factory=list)
    reference_block_count: int = 0


def run(store: SQLiteStore, spec, embedding_model=None) -> RunResult:
    """Execute a QuerySpec. Raises EmptyQuerySpec when nothing was asked.

    embedding_model is required for semantic mode (callers own model
    caching); pass None otherwise.
    """
    if spec.is_empty():
        raise EmptyQuerySpec(
            "Provide at least one of: query (semantic), keyword (FTS), title, or filters."
        )

    # Normalize once so every mode compares in the storage format; a bad
    # timestamp fails loudly here instead of silently matching nothing.
    ingested_bound = normalize_utc_timestamp(spec.after_ingested) if spec.after_ingested else None

    def _ingested_too_old(block: Block) -> bool:
        return ingested_bound is not None and (block.ingested_at or "") < ingested_bound

    def _fails_task_filter(block: Block) -> bool:
        """has_task=True: keep only user-marked tasks; bronze never counts."""
        if spec.has_task is not True:
            return False
        all_tags = {t.lstrip("#") for t in block.tags + block.metadata.get("augi_tags", [])}
        if BRONZE_TAG in all_tags:
            return True
        return not (block.metadata.get("has_open_task") or "type/task" in all_tags)

    k = spec.k

    if spec.mode in ("title", "keyword"):
        fts_query = f"title:{spec.title}" if spec.mode == "title" else spec.keyword
        results = store.search_fts(fts_query, limit=k + 1)
        results = [
            b
            for b in results
            if not _path_excluded(b, spec.exclude_path_prefix)
            and not _ingested_too_old(b)
            and not _fails_task_filter(b)
        ]
        has_more = len(results) > k
        return RunResult(mode=spec.mode, blocks=results[:k], has_more=has_more)

    if spec.mode == "semantic":
        if embedding_model is None:
            raise ValueError("semantic mode requires an embedding model")
        query_vec = embedding_model.embed_query(spec.query)
        hits = store.semantic_search(query_vec, k=k * 3)

        hit_ids = [block_id for block_id, _ in hits]
        blocks_map = store.get_blocks_by_ids(hit_ids)

        kept: list[Block] = []
        scores: dict[str, float] = {}
        for block_id, distance in hits:
            block = blocks_map.get(block_id)
            if block is None:
                continue
            if spec.kind and block.kind != spec.kind:
                continue
            if spec.source and block.source != spec.source:
                continue
            if spec.tags and not set(spec.tags).intersection(
                block.tags + block.metadata.get("augi_tags", [])
            ):
                continue
            if spec.after and (block.block_time or "") < spec.after:
                continue
            if spec.before and (block.block_time or "") > spec.before:
                continue
            if _ingested_too_old(block):
                continue
            if _path_excluded(block, spec.exclude_path_prefix):
                continue
            if _fails_task_filter(block):
                continue
            kept.append(block)
            scores[block_id] = round(1.0 - distance, 4)
            if len(kept) > k:
                break

        has_more = len(kept) > k
        return RunResult(mode="semantic", blocks=kept[:k], scores=scores, has_more=has_more)

    # Browse mode — SQL-filtered and paginated
    blocks, total = store.get_blocks_filtered(
        kind=spec.kind or "data_block",
        source=spec.source,
        after=spec.after,
        before=spec.before,
        after_ingested=spec.after_ingested,
        limit=k + 1,
        offset=spec.offset,
        exclude_path_prefix=spec.exclude_path_prefix,
    )
    # Tags filtering happens in Python (not pushed to SQL)
    kept = []
    reference_blocks: list[Block] = []
    for b in blocks:
        if spec.tags and not set(spec.tags).intersection(b.tags + b.metadata.get("augi_tags", [])):
            continue
        if _fails_task_filter(b):
            continue
        if _reference_source_tags(b):
            reference_blocks.append(b)
        else:
            kept.append(b)

    # has_more reflects the SQL overfetch (k+1), not the post-filter count —
    # historical envelope semantics the golden tests pin.
    has_more = len(blocks) > k
    return RunResult(
        mode="browse",
        blocks=kept[:k],
        has_more=has_more,
        total=total,
        next_offset=spec.offset + k if has_more else None,
        reference_documents=_group_reference_documents(reference_blocks),
        reference_block_count=len(reference_blocks),
    )


# ── Shared rules ────────────────────────────────────────────────────


def _path_excluded(block: Block, prefix: str | None) -> bool:
    """True if the block's source_path falls under an excluded prefix."""
    return bool(prefix) and block.metadata.get("source_path", "").startswith(prefix)


def _reference_source_tags(block: Block) -> list[str]:
    """The block's source/* tags (set by [vault.source_rules] for synced
    reference material like Readwise or Snipd). Empty list = user capture."""
    all_tags = block.tags + block.metadata.get("augi_tags", [])
    return [t for t in all_tags if t.startswith("source/")]


def _group_reference_documents(blocks: list[Block]) -> list[dict[str, Any]]:
    """Collapse reference-source blocks into one entry per source document.

    Reference material is one artifact: the review pass routes the document
    once (apply_routing with document_id), never its individual blocks.
    """
    groups: dict[str, dict] = {}
    for b in blocks:
        source_path = b.metadata.get("source_path", "")
        g = groups.get(source_path)
        if g is None:
            g = groups[source_path] = {
                "source_path": source_path,
                "document_id": Block.make_document_id(source_path),
                "title": Path(source_path).stem if source_path else b.title,
                "source_tags": [],
                "block_count": 0,
                "first_block_time": b.block_time,
                "last_block_time": b.block_time,
                "snippet": (b.content or "")[:200],
            }
        g["block_count"] += 1
        for t in _reference_source_tags(b):
            if t not in g["source_tags"]:
                g["source_tags"].append(t)
        if b.block_time:
            if not g["first_block_time"] or b.block_time < g["first_block_time"]:
                g["first_block_time"] = b.block_time
            if not g["last_block_time"] or b.block_time > g["last_block_time"]:
                g["last_block_time"] = b.block_time
    return list(groups.values())
