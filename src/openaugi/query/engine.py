"""The query engine — deterministic read semantics, full blocks, no transport.

Every *rule* that used to live inline in mcp/server.py tool bodies executes
here: mode dispatch, the after_ingested bound, the has_task filter,
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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from openaugi.model.block import Block
from openaugi.store.sqlite import normalize_utc_timestamp

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


# A keyword hit we cannot score on the cosine scale (no embedding yet). High,
# because a literal match is strong evidence, but not 1.0 — that would make it
# an unbeatable outlier in any distribution-based ranking.
FTS_FALLBACK_SCORE = 0.75


def _similarity(distance: float) -> float:
    """Cosine similarity from an L2 distance over unit vectors: 1 − d²/2.

    vec_blocks stores normalized embeddings and vec0 MATCH returns L2, so the
    old `1 − distance` formula clamped every pair with cosine < 0.5 (L2 > 1)
    to a meaningless ≤ 0 — which the salience gate then dropped wholesale.
    This keeps the full (−1, 1] similarity scale intact.
    """
    return round(1.0 - (distance * distance) / 2.0, 4)


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
        """has_task=True: keep only user-marked tasks."""
        if spec.has_task is not True:
            return False
        all_tags = {t.lstrip("#") for t in block.tags + block.metadata.get("augi_tags", [])}
        return not (block.metadata.get("has_open_task") or "type/task" in all_tags)

    k = spec.k

    if spec.mode in ("title", "keyword"):
        fts_query = f"title:{spec.title}" if spec.mode == "title" else spec.keyword
        results = store.search_fts(fts_query, limit=k + 1)
        results = [
            b
            for b in results
            if not _path_excluded(b, spec.exclude_path_prefix)
            and not _path_not_included(b, spec.include_path_prefix)
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
            if _path_not_included(block, spec.include_path_prefix):
                continue
            if _fails_task_filter(block):
                continue
            kept.append(block)
            scores[block_id] = _similarity(distance)
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
        include_path_prefix=spec.include_path_prefix,
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


# ── Block fetch ─────────────────────────────────────────────────────


class FetchResult(BaseModel):
    """Ordered found blocks + their routed-container titles + missing ids."""

    blocks: list[Block]
    routes: dict[str, list[str]] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)


def fetch(store: SQLiteStore, block_ids: list[str]) -> FetchResult:
    """Full blocks by id, in request order, with routed_to container titles."""
    found = store.get_blocks_by_ids(block_ids)
    ordered = [found[bid] for bid in block_ids if bid in found]
    missing = [bid for bid in block_ids if bid not in found]
    routes = store.get_routed_container_titles(list(found)) if found else {}
    return FetchResult(blocks=ordered, routes=routes, missing=missing)


# ── Graph ───────────────────────────────────────────────────────────


class RelatedItem(BaseModel):
    block: Block
    link_kind: str
    direction: str  # "out" | "in"


def related(
    store: SQLiteStore,
    block_id: str,
    kind: str | None = None,
    direction: str = "both",
    limit: int = 50,
) -> list[RelatedItem]:
    """One-hop link follow — out links first, then in, each capped at limit."""
    out_links = (
        store.get_links_from(block_id, kind=kind)[:limit] if direction in ("out", "both") else []
    )
    in_links = (
        store.get_links_to(block_id, kind=kind)[:limit] if direction in ("in", "both") else []
    )

    needed_ids = [lnk.to_id for lnk in out_links] + [lnk.from_id for lnk in in_links]
    blocks_map = store.get_blocks_by_ids(needed_ids) if needed_ids else {}

    items: list[RelatedItem] = []
    for lnk in out_links:
        target = blocks_map.get(lnk.to_id)
        if target:
            items.append(RelatedItem(block=target, link_kind=lnk.kind, direction="out"))
    for lnk in in_links:
        source_block = blocks_map.get(lnk.from_id)
        if source_block:
            items.append(RelatedItem(block=source_block, link_kind=lnk.kind, direction="in"))
    return items


class TraverseItem(BaseModel):
    block: Block
    depth: int


def traverse(
    store: SQLiteStore,
    start_id: str,
    max_hops: int = 2,
    link_kinds: list[str] | None = None,
    limit: int = 50,
) -> list[TraverseItem]:
    """Breadth-first multi-hop walk; each item carries distance from start."""
    visited: set[str] = {start_id}
    items: list[TraverseItem] = []
    current_level = [start_id]

    for depth in range(1, max_hops + 1):
        if not current_level or len(items) >= limit:
            break

        next_ids: list[str] = []
        for cid in current_level:
            links_out = store.get_links_from(cid)
            links_in = store.get_links_to(cid)
            for lnk in links_out + links_in:
                nid = lnk.to_id if lnk.from_id == cid else lnk.from_id
                if nid not in visited:
                    if link_kinds and lnk.kind not in link_kinds:
                        continue
                    next_ids.append(nid)
                    visited.add(nid)

        if not next_ids:
            break

        blocks_map = store.get_blocks_by_ids(next_ids)
        for nid in next_ids:
            block = blocks_map.get(nid)
            if block:
                items.append(TraverseItem(block=block, depth=depth))
                if len(items) >= limit:
                    break

        current_level = next_ids

    return items


# ── Recency ─────────────────────────────────────────────────────────


def recent(
    store: SQLiteStore,
    k: int = 20,
    kind: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
) -> list[Block]:
    """Recently ingested blocks, newest first; overfetch absorbs tag filtering."""
    blocks, _ = store.get_blocks_filtered(
        kind=kind or "data_block",
        source=source,
        order_by="ingested_at",
        limit=k * 3,
    )

    results: list[Block] = []
    for b in blocks:
        if tags and not set(tags).intersection(b.tags + b.metadata.get("augi_tags", [])):
            continue
        results.append(b)
        if len(results) >= k:
            break
    return results


# ── Containers and views ────────────────────────────────────────────


class ContainerNotFound(LookupError):
    """The named container note doesn't exist as a document block."""


class Member(BaseModel):
    block: Block
    membership: str  # "contained" | "routed" | "both"


class MembersResult(BaseModel):
    container_id: str
    members: list[Member]  # the requested page
    total: int


def members(
    store: SQLiteStore, container_title: str, limit: int = 100, offset: int = 0
) -> MembersResult:
    """Container membership under the unified rule (routed_to ∪ contains)."""
    container_id = _container_id(store, container_title)
    all_members = store.get_container_members(container_id)
    page = all_members[offset : offset + limit]
    return MembersResult(
        container_id=container_id,
        members=[Member(block=b, membership=m) for b, m in page],
        total=len(all_members),
    )


class ViewResult(BaseModel):
    container_id: str
    recap: dict | None = None  # {recap_md, generated_at, stale}
    members: list[Member]
    member_count: int


def view(store: SQLiteStore, container_title: str, member_limit: int = 50) -> ViewResult:
    """A container's view: live membership + cached recap with staleness."""
    container_id = _container_id(store, container_title)
    all_members = store.get_container_members(container_id)
    recap_row = store.get_recap(container_id)
    recap = None
    if recap_row is not None:
        recap = {
            "recap_md": recap_row["recap_md"],
            "generated_at": recap_row["generated_at"],
            "stale": recap_row["membership_hash"] != store.membership_hash(container_id),
        }
    return ViewResult(
        container_id=container_id,
        recap=recap,
        members=[Member(block=b, membership=m) for b, m in all_members[:member_limit]],
        member_count=len(all_members),
    )


def views(store: SQLiteStore) -> list[dict]:
    """The render list: every container with a cached recap, newest first."""
    out: list[dict] = []
    for r in store.list_recaps():
        out.append(
            {
                "container": r["title"],
                "container_id": r["container_id"],
                "generated_at": r["generated_at"],
                "stale": r["membership_hash"] != store.membership_hash(r["container_id"]),
            }
        )
    return out


def review_state(store: SQLiteStore) -> dict:
    """Review-pass high-water mark: {'last_run', 'last_summary'}."""
    return store.get_review_state()


def _container_id(store: SQLiteStore, container_title: str) -> str:
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (container_title,),
    ).fetchone()
    if not row:
        raise ContainerNotFound(container_title)
    return row[0]


# ── Context (retrieve → dedupe → rerank → expand) ───────────────────


class ContextEntry(BaseModel):
    block: Block
    extras: dict  # direct: {score, source}; expanded: {expanded_from, link_kind}


class ContextResult(BaseModel):
    """Mechanics output of the get_context pipeline.

    `seen` is every surfaced block in presentation order — direct results
    first, then expanded neighbors (historical get_context behavior fills
    direct_results from this combined ordering). `had_candidates` is False
    when retrieval found nothing at all (adapters return a short envelope).
    """

    seen: list[ContextEntry] = Field(default_factory=list)
    expanded: list[ContextEntry] = Field(default_factory=list)
    had_candidates: bool = True
    min_score: float | None = None


def context(
    store: SQLiteStore,
    query: str,
    k: int = 10,
    expand: bool = True,
    purpose: str | None = None,
    embedding_model=None,
    config: dict | None = None,
) -> ContextResult:
    """FTS + semantic retrieval → MMR rerank → link expand.

    Deterministic — the rerank is cosine/MMR math (pipeline.rerank), no LLM.
    `config` supplies [retrieval]/[salience]; callers pass their
    loaded config so test monkeypatching stays at the adapter.
    """
    import numpy as np

    from openaugi.pipeline.rerank import rerank as _rerank

    if config is None:
        from openaugi.config import load_config

        config = load_config()

    retrieval = config.get("retrieval", {})
    overfetch_ratio = retrieval.get("overfetch_ratio", 3)
    group_threshold = retrieval.get("group_threshold", 0.15)
    mmr_lambda = retrieval.get("mmr_lambda", 0.5)
    representative = retrieval.get("representative", "centroid")

    fetch_limit = k * overfetch_ratio

    # Collect candidate IDs and their best relevance scores
    candidate_scores: dict[str, float] = {}

    # Prong 1: FTS — keyword hits enter the pool; they are scored below,
    # alongside semantic hits, on the one comparable scale (cosine). Pinning
    # them at a constant 1.0 (pre-2026-08-29) made every keyword hit an
    # automatic outlier, which is fine for a top-k list but breaks any caller
    # that reasons about the score distribution — see FTS_FALLBACK_SCORE.
    fts_results = store.search_fts(query, limit=fetch_limit)
    fts_only_ids = [b.id for b in fts_results]
    for b in fts_results:
        candidate_scores[b.id] = 0.0

    # Prong 2: semantic search
    query_vec: list[float] | None = None
    try:
        if embedding_model is None:
            raise ValueError("no embedding model provided")
        vec: list[float] = embedding_model.embed_query(query)
        query_vec = vec
        hits = store.semantic_search(vec, k=fetch_limit)
        for block_id, distance in hits:
            score = _similarity(distance)
            candidate_scores[block_id] = max(candidate_scores.get(block_id, 0.0), score)
    except Exception:
        logger.warning("Semantic search unavailable in context", exc_info=True)

    if not candidate_scores:
        return ContextResult(had_candidates=False)

    all_ids = list(candidate_scores.keys())

    # Batch-fetch embeddings for all candidates (single query, no full block load)
    emb_map = store.get_embeddings_for_ids(all_ids)

    # Score the keyword hits on the same cosine scale as the semantic ones, so
    # every candidate is comparable. A keyword hit with no embedding keeps a
    # high constant — it matched literally, so it belongs near the top.
    if query_vec is not None:
        query_arr = np.array(query_vec, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_arr)) or 1.0
        for block_id in fts_only_ids:
            if candidate_scores.get(block_id, 0.0) > 0.0:
                continue  # semantic search already scored it
            blob = emb_map.get(block_id)
            if blob is None:
                candidate_scores[block_id] = FTS_FALLBACK_SCORE
                continue
            vec_arr = np.frombuffer(blob, dtype=np.float32)
            denom = query_norm * (float(np.linalg.norm(vec_arr)) or 1.0)
            candidate_scores[block_id] = round(float(query_arr @ vec_arr) / denom, 4)
    else:
        for block_id in fts_only_ids:
            candidate_scores[block_id] = FTS_FALLBACK_SCORE

    # Build candidates for reranker: (block_id, embedding_blob_or_None, score)
    candidates = [(bid, emb_map.get(bid), candidate_scores[bid]) for bid in all_ids]

    # Rerank if we have a query embedding; fall back to score order otherwise
    if query_vec is not None:
        query_blob = np.array(query_vec, dtype=np.float32).tobytes()
        final_ids = _rerank(
            candidates,
            query_blob,
            k,
            group_threshold=group_threshold,
            mmr_lambda=mmr_lambda,
            representative=representative,
        )
    else:
        final_ids = sorted(all_ids, key=lambda bid: candidate_scores[bid], reverse=True)[:k]

    # Salience gate — purpose-based min-score policy owned here, not by
    # callers (mobile resurfacing today, push notifications later). Drops
    # low-salience results after reranking; the scores themselves are
    # untouched.
    min_score: float | None = None
    if purpose is not None:
        threshold = config.get("salience", {}).get(purpose)
        if isinstance(threshold, int | float) and not isinstance(threshold, bool):
            min_score = float(threshold)
            final_ids = [bid for bid in final_ids if candidate_scores[bid] >= min_score]

    # Batch-fetch full blocks for the final k IDs
    final_blocks = store.get_blocks_by_ids(final_ids)
    fts_ids = {b.id for b in fts_results}
    seen: dict[str, ContextEntry] = {}
    for block_id in final_ids:
        block = final_blocks.get(block_id)
        if block:
            seen[block_id] = ContextEntry(
                block=block,
                extras={
                    "score": candidate_scores.get(block_id, 0.0),
                    "source": "fts" if block_id in fts_ids else "semantic",
                },
            )

    # Expand: follow links from top results, batch-fetch targets
    expanded: list[ContextEntry] = []
    if expand:
        expand_targets: list[tuple[str, str, str]] = []  # (to_id, from_id, link_kind)
        for block_id in final_ids:
            links = store.get_links_from(block_id)
            for lnk in links[:5]:
                if lnk.to_id not in seen:
                    expand_targets.append((lnk.to_id, block_id, lnk.kind))

        if expand_targets:
            expand_ids = list({t[0] for t in expand_targets})
            expand_blocks = store.get_blocks_by_ids(expand_ids)
            for to_id, from_id, link_kind in expand_targets:
                target = expand_blocks.get(to_id)
                if target and to_id not in seen:
                    entry = ContextEntry(
                        block=target,
                        extras={"expanded_from": from_id, "link_kind": link_kind},
                    )
                    expanded.append(entry)
                    seen[to_id] = entry

    return ContextResult(
        seen=list(seen.values()),
        expanded=expanded,
        had_candidates=True,
        min_score=min_score,
    )


# ── Shared rules ────────────────────────────────────────────────────


def _path_excluded(block: Block, prefix: str | None) -> bool:
    """True if the block's source_path falls under an excluded prefix."""
    return bool(prefix) and block.metadata.get("source_path", "").startswith(prefix)


def _path_not_included(block: Block, prefix: str | None) -> bool:
    """True if an include prefix was given and the block falls outside it.

    The mirror of _path_excluded. A block with no source_path can't be under
    the requested folder, so it fails the filter too."""
    return bool(prefix) and not block.metadata.get("source_path", "").startswith(prefix)


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
