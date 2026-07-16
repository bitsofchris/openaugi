"""OpenAugi MCP Server — query + write tools for Claude.

Read tools (readOnlyHint):
- search: semantic (sqlite-vec KNN) + title (FTS5 title-only) + keyword (FTS5) + filters
  (tag filter matches both user tags and augi_tags)
- get_block: full block content by ID
- get_blocks: batch fetch multiple blocks by ID (up to 50)
- get_related: follow links from a block
- traverse: multi-hop graph walk
- get_context: compound search → expand → structured result
- recent: recently created blocks
- get_members: a container's members under the unified rule (contained ∪ routed)
- get_view: render a container's view from the DB (membership log + cached recap)
- list_views: the render list — every container with a cached recap

Write tools:
- write_document: create a markdown note in OpenAugi/{subfolder}/
- tag_block: stamp AI-classified augi_tags onto a block
- write_recap: cache a container's recap (the synthesis half of its view)

Review pass tools:
- apply_routing: the route CRUD tool — batch add/remove routed_to links + tag,
  one call for a whole pass's decisions or a single correction
- get_review_state: read the review-pass high-water mark
- mark_review_complete: advance the high-water mark after a pass

Resources:
- vault://note/{title}: all entries for a note + hub context
"""

from __future__ import annotations

import functools
import json
import logging
import os
from datetime import UTC
from pathlib import Path
from typing import Literal

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from openaugi.config import load_config
from openaugi.models import get_embedding_model
from openaugi.pipeline.rerank import rerank as _rerank
from openaugi.query import QuerySpec, engine
from openaugi.query.engine import BRONZE_TAG
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

mcp = FastMCP("openaugi")


# ── State (initialized lazily) ─────────────────────────────────────

_store: SQLiteStore | None = None
_embedding_model = None
_db_path: str | None = None


def _get_db_path() -> str:
    return os.environ.get("OPENAUGI_DB", str(Path.home() / ".openaugi" / "openaugi.db"))


def _get_vault_path() -> str | None:
    """Resolve vault path: env var > config.toml > None."""
    vault = os.environ.get("OPENAUGI_VAULT_PATH")
    if vault:
        return vault
    config = load_config()
    return config.get("vault", {}).get("default_path")


def _get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore(_get_db_path())
    return _store


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        config = load_config()
        _embedding_model = get_embedding_model(config.get("models", {}).get("embedding"))
    return _embedding_model


def _release_conn(fn):
    """Close SQLite connection after each tool call (auto-reconnects on next use)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            if _store is not None:
                _store.close()

    return wrapper


def _json(data) -> str:
    return json.dumps(data, indent=2, default=str)


# ── Read Tools ─────────────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def search(
    query: str | None = None,
    keyword: str | None = None,
    title: str | None = None,
    k: int = 100,
    offset: int = 0,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    after_ingested: str | None = None,
    kind: str | None = None,
    source: str | None = None,
    exclude_path_prefix: str | None = None,
    has_task: bool | None = None,
) -> str:
    """Search the knowledge base. Returns block summaries (not full content).

    Use this as your first step to find relevant blocks. Four modes:
    - Title: provide 'title' to search note/document titles only (e.g. title="meeting notes")
    - Semantic: provide 'query' for vector similarity search (best for concepts/questions)
    - Keyword: provide 'keyword' for FTS5 full-text search (best for exact terms)
    - Browse: provide only filters (tags, after, before, after_ingested, kind, source)

    Do NOT use this to get full block content — use get_block or get_blocks for that.
    For a complete research workflow (search + deduplicate + expand), use get_context instead.

    Prefer 'title' over 'keyword' when you know the note name.
    Must provide at least one of: query, keyword, title, or a filter.

    Browse mode (date-range queries): results are paginated. Use offset to fetch the next
    page. Response includes 'total' so you know how many pages to expect.
    Example: search(after="2026-04-05", before="2026-04-12") returns all blocks in that week.
    Call again with offset=100 if has_more is true.
    Dates use ISO format: after="2025-01-01", before="2025-06-01".

    after/before filter on block_time — the CONTENT date, which may be
    date-only and doesn't change when a note is edited. after_ingested
    filters on when the block entered the DB (full UTC timestamp): use it
    for "what's new since <timestamp>" queues like the review pass, where
    it catches same-day date-only blocks and re-ingested edits that
    after= would silently miss. The two can combine but usually don't.

    exclude_path_prefix drops blocks whose source_path starts with the given
    prefix (e.g. exclude_path_prefix="OpenAugi/" keeps derived artifacts out
    of a review queue). Works in every mode.

    has_task=True keeps only blocks the user marked as a task — an open
    `- [ ] …` checkbox (metadata has_open_task, extracted at ingest) or a
    type/task tag — and always excludes layer/bronze (demoted scaffolding
    carries no signal). Deterministic: this is the Dashboard task-shelf
    query (e.g. search(has_task=True, after=<14 days ago>)). Works in
    every mode.

    Browse mode groups reference material: blocks carrying a source/* tag
    (Readwise, Snipd, and other synced imports) are collapsed into
    'reference_documents' — one entry per source document with document_id,
    block_count, and time range. Route the DOCUMENT (apply_routing with
    document_id), never its individual blocks."""
    spec = QuerySpec(
        query=query,
        keyword=keyword,
        title=title,
        tags=tags,
        after=after,
        before=before,
        after_ingested=after_ingested,
        kind=kind,
        source=source,
        exclude_path_prefix=exclude_path_prefix,
        has_task=has_task,
        k=k,
        offset=offset,
    )
    if spec.is_empty():
        return _json(
            {
                "error": "No search parameters provided.",
                "hint": "Provide at least one of: query (semantic), keyword (FTS), "
                "title, or filters (tags, after, before, after_ingested, kind, source). "
                "Example: search(keyword='project plan')",
            }
        )

    model = _get_embedding_model() if spec.mode == "semantic" else None
    result = engine.run(_get_store(), spec, embedding_model=model)

    if result.mode == "semantic":
        results = []
        for b in result.blocks:
            summary = _block_summary(b)
            summary["score"] = result.scores[b.id]
            results.append(summary)
        return _json(
            {
                "results": results,
                "count": len(results),
                "has_more": result.has_more,
                "mode": "semantic",
            }
        )

    if result.mode in ("title", "keyword"):
        return _json(
            {
                "results": [_block_summary(b) for b in result.blocks],
                "count": len(result.blocks),
                "has_more": result.has_more,
                "mode": result.mode,
            }
        )

    return _json(
        {
            "results": [_block_summary(b) for b in result.blocks],
            "count": len(result.blocks),
            "reference_documents": result.reference_documents,
            "reference_block_count": result.reference_block_count,
            "total": result.total,
            "has_more": result.has_more,
            "next_offset": result.next_offset,
            "mode": "browse",
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_block(block_id: str) -> str:
    """Get full block content and metadata by ID.

    Use after search/get_context to read the complete content of a specific block.
    For multiple blocks, use get_blocks instead — one call vs. many.
    Do NOT use this in a loop — use get_blocks with a list of IDs."""
    store = _get_store()
    block = store.get_block(block_id)
    if block is None:
        return _json(
            {
                "error": f"Block not found: {block_id}",
                "hint": "This ID may be stale or incorrect. Use search(keyword=...) or "
                "search(title=...) to find valid block IDs.",
            }
        )
    routes = store.get_routed_container_titles([block_id]).get(block_id, [])
    return _json(_block_full(block, routes))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_blocks(block_ids: list[str]) -> str:
    """Get full content and metadata for multiple blocks in one call.

    Use this instead of calling get_block in a loop — saves round trips.
    Accepts up to 50 IDs. Missing IDs are listed in the 'missing' array (not errors).
    Block order in the response matches the order of block_ids."""
    if len(block_ids) > 50:
        return _json(
            {
                "error": f"Too many IDs ({len(block_ids)}). Maximum is 50 per request.",
                "hint": "Split into multiple get_blocks calls of 50 or fewer IDs each.",
            }
        )
    store = _get_store()
    found = store.get_blocks_by_ids(block_ids)
    missing = [bid for bid in block_ids if bid not in found]
    routes = store.get_routed_container_titles(list(found))
    return _json(
        {
            "blocks": [
                _block_full(found[bid], routes.get(bid, [])) for bid in block_ids if bid in found
            ],
            "count": len(found),
            "missing": missing,
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_related(
    block_id: str,
    kind: str | None = None,
    direction: str = "both",
    limit: int = 50,
) -> str:
    """Follow links from/to a block. Returns connected block summaries.

    Use to explore the graph around a known block — find its tags, parent document,
    wikilinked notes, or derived content.
    Do NOT use this for broad discovery — use search or get_context instead.

    - direction: 'out' (from block), 'in' (to block), or 'both' (default)
    - kind: filter by link kind. Common kinds: 'contains' (data_block→context_block:document),
      'groups' (data_block→context_block:tag), 'links_to' (wikilink between notes)
    - limit: max results (default 50)"""
    store = _get_store()

    out_links = (
        store.get_links_from(block_id, kind=kind)[:limit] if direction in ("out", "both") else []
    )
    in_links = (
        store.get_links_to(block_id, kind=kind)[:limit] if direction in ("in", "both") else []
    )

    # Batch-fetch all linked blocks in one query
    needed_ids = [lnk.to_id for lnk in out_links] + [lnk.from_id for lnk in in_links]
    blocks_map = store.get_blocks_by_ids(needed_ids) if needed_ids else {}

    results = []
    for lnk in out_links:
        target = blocks_map.get(lnk.to_id)
        if target:
            results.append(
                {"block": _block_summary(target), "link_kind": lnk.kind, "direction": "out"}
            )
    for lnk in in_links:
        source_block = blocks_map.get(lnk.from_id)
        if source_block:
            results.append(
                {"block": _block_summary(source_block), "link_kind": lnk.kind, "direction": "in"}
            )

    return _json({"block_id": block_id, "related": results, "count": len(results)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def traverse(
    start_id: str,
    max_hops: int = 2,
    link_kinds: list[str] | None = None,
    limit: int = 50,
) -> str:
    """Multi-hop graph walk from a starting block. Returns block summaries with depth.

    Use to map the neighborhood around a block — e.g. find everything connected to a
    topic tag within 2 hops. Each result includes 'depth' (distance from start).
    For single-hop exploration, prefer get_related (simpler, shows link kinds).
    Do NOT use for broad search — use search or get_context instead.

    - max_hops: how many link-steps to follow (default 2, max recommended 3)
    - link_kinds: restrict to specific link types (e.g. ['links_to', 'groups'])
    - limit: max results (default 50)"""
    store = _get_store()

    visited: set[str] = {start_id}
    results: list[dict] = []
    current_level = [start_id]

    for depth in range(1, max_hops + 1):
        if not current_level or len(results) >= limit:
            break

        # Gather all neighbor IDs for this frontier level
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

        # Batch-fetch all blocks for this level
        blocks_map = store.get_blocks_by_ids(next_ids)
        for nid in next_ids:
            block = blocks_map.get(nid)
            if block:
                results.append({**_block_summary(block), "depth": depth})
                if len(results) >= limit:
                    break

        current_level = next_ids

    return _json(
        {
            "start_id": start_id,
            "results": results,
            "count": len(results),
            "max_hops": max_hops,
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_context(
    query: str,
    k: int = 10,
    expand: bool = True,
    purpose: str | None = None,
) -> str:
    """Primary research tool. Use this as the default for answering questions against the
    knowledge base — it runs a full retrieval pipeline in one call:
    semantic search + keyword search → deduplicate → diversity re-rank (MMR) → expand via links.

    Returns 'direct_results' (top blocks) and 'expanded' (linked blocks for extra context).
    Prefer this over manual search → get_block → get_related chains.
    Use plain 'search' only when you need specific search modes (title-only, browse filters)
    or fine-grained control over results.

    - k: number of final results (default 10). Internally overfetches 3x for dedup.
    - expand: follow links from top results for richer context (default true)
    - purpose: optional salience gate for proactive surfaces (e.g. 'resurface').
      Applies the min-score from config [salience] and drops results below it —
      scores themselves are unchanged. Unknown purpose or no config key = no gate.
      Regular research calls should omit this.

    Blocks tagged #layer/bronze (user-demoted scaffolding) are down-weighted by
    config [layers] bronze_weight before reranking, and excluded entirely when
    purpose is set — demoted thoughts never resurface proactively."""
    store = _get_store()
    config = load_config()
    retrieval = config.get("retrieval", {})
    overfetch_ratio = retrieval.get("overfetch_ratio", 3)
    group_threshold = retrieval.get("group_threshold", 0.15)
    mmr_lambda = retrieval.get("mmr_lambda", 0.5)
    representative = retrieval.get("representative", "centroid")

    fetch_limit = k * overfetch_ratio

    # Collect candidate IDs and their best relevance scores
    candidate_scores: dict[str, float] = {}

    # Prong 1: FTS — score 1.0 (keyword match = high relevance)
    fts_results = store.search_fts(query, limit=fetch_limit)
    for b in fts_results:
        candidate_scores[b.id] = 1.0

    # Prong 2: semantic search
    query_vec: list[float] | None = None
    try:
        query_vec = _get_embedding_model().embed_query(query)
        hits = store.semantic_search(query_vec, k=fetch_limit)
        for block_id, distance in hits:
            score = round(1.0 - distance, 4)
            candidate_scores[block_id] = max(candidate_scores.get(block_id, 0.0), score)
    except Exception:
        logger.warning("Semantic search unavailable in get_context", exc_info=True)

    if not candidate_scores:
        return _json({"query": query, "direct_results": [], "expanded": [], "total_blocks": 0})

    all_ids = list(candidate_scores.keys())

    # Bronze layer — #layer/bronze marks user-demoted scaffolding (mobile
    # curation). Kept in the DB (raw data is truth) but down-weighted here so
    # full-weight thinking outranks it; proactive surfaces (purpose=...)
    # exclude it outright below, like the source/* firewall.
    raw_weight = config.get("layers", {}).get("bronze_weight", 1.0)
    bronze_weight = (
        float(raw_weight)
        if isinstance(raw_weight, int | float) and not isinstance(raw_weight, bool)
        else 1.0
    )
    bronze_ids: set[str] = set()
    if purpose is not None or bronze_weight < 1.0:
        tags_map = store.get_tags_for_ids(all_ids)
        bronze_ids = {bid for bid, tags in tags_map.items() if BRONZE_TAG in tags}
    if bronze_weight < 1.0:
        for bid in bronze_ids:
            candidate_scores[bid] = round(candidate_scores[bid] * bronze_weight, 4)

    # Batch-fetch embeddings for all candidates (single query, no full block load)
    emb_map = store.get_embeddings_for_ids(all_ids)

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

    # Salience gate — purpose-based min-score policy owned here, not by callers
    # (mobile resurfacing today, push notifications later). Drops low-salience
    # results after reranking; the scores themselves are untouched. Proactive
    # surfaces never resurface bronze: the user already demoted it.
    min_score: float | None = None
    if purpose is not None:
        final_ids = [bid for bid in final_ids if bid not in bronze_ids]
        threshold = config.get("salience", {}).get(purpose)
        if isinstance(threshold, int | float) and not isinstance(threshold, bool):
            min_score = float(threshold)
            final_ids = [bid for bid in final_ids if candidate_scores[bid] >= min_score]

    # Batch-fetch full blocks for the final k IDs
    final_blocks = store.get_blocks_by_ids(final_ids)
    fts_ids = {b.id for b in fts_results}
    blocks_seen: dict[str, dict] = {}
    for block_id in final_ids:
        block = final_blocks.get(block_id)
        if block:
            entry = {**_block_summary(block), "score": candidate_scores.get(block_id, 0.0)}
            entry["source"] = "fts" if block_id in fts_ids else "semantic"
            blocks_seen[block_id] = entry

    # Expand: follow links from top results, batch-fetch targets
    expanded: list[dict] = []
    if expand:
        expand_targets: list[tuple[str, str, str]] = []  # (to_id, from_id, link_kind)
        for block_id in final_ids:
            links = store.get_links_from(block_id)
            for lnk in links[:5]:
                if lnk.to_id not in blocks_seen:
                    expand_targets.append((lnk.to_id, block_id, lnk.kind))

        if expand_targets:
            expand_ids = list({t[0] for t in expand_targets})
            expand_blocks = store.get_blocks_by_ids(expand_ids)
            for to_id, from_id, link_kind in expand_targets:
                target = expand_blocks.get(to_id)
                if target and to_id not in blocks_seen:
                    entry = {
                        **_block_summary(target),
                        "expanded_from": from_id,
                        "link_kind": link_kind,
                    }
                    expanded.append(entry)
                    blocks_seen[to_id] = entry

    result = {
        "query": query,
        "direct_results": list(blocks_seen.values())[:k],
        "expanded": expanded[:k],
        "total_blocks": len(blocks_seen),
    }
    if purpose is not None:
        result["salience"] = {"purpose": purpose, "min_score": min_score}
    return _json(result)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def recent(
    k: int = 20,
    kind: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Recently created blocks, ordered newest first. Returns block summaries.

    Use to see what's new in the knowledge base or catch up on recent activity.
    Do NOT use this for topic-based search — use search or get_context instead.

    - k: max results (default 20)
    - kind: block kind to filter by (default 'data_block'). Common kinds:
      'data_block', 'context_block:document', 'context_block:tag'
    - source: filter by source (e.g. 'vault')
    - tags: filter to blocks matching any of these tags"""
    store = _get_store()
    blocks, _ = store.get_blocks_filtered(
        kind=kind or "data_block",
        source=source,
        order_by="ingested_at",
        limit=k * 3,  # overfetch to absorb tag filtering below
    )

    results = []
    for b in blocks:
        if tags and not set(tags).intersection(b.tags + b.metadata.get("augi_tags", [])):
            continue
        results.append(_block_summary(b))
        if len(results) >= k:
            break

    return _json({"results": results, "count": len(results), "has_more": False})


# ── Classification Tools ───────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
@_release_conn
def tag_block(block_id: str, augi_tags: list[str]) -> str:
    """Stamp AI-classified tags onto a block.

    Writes augi_tags to block.metadata["augi_tags"]. Overwrites any prior
    classification — call once per block after classifying. Used by the
    augi-agent to persist area/type/status classifications.

    augi_tags: list of tag strings, e.g. ["area/work", "type/task", "status/active"]
    """
    store = _get_store()
    found = store.update_block_metadata(block_id, {"augi_tags": augi_tags})
    if not found:
        return _json({"status": "error", "reason": f"Block {block_id} not found."})
    return _json({"status": "ok", "block_id": block_id, "augi_tags": augi_tags})


# ── Review Pass Tools ──────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
@_release_conn
def apply_routing(decisions: list[dict]) -> str:
    """Modify block→container routing — THE route write tool (add, remove, tag).

    Membership is a routed_to LINK from a block to a container note
    (AMOC/PMOC/MOC); a block may belong to several containers. Each decision:

      {"block_id": str,
       "add": [container titles],      # route into ("containers" is an alias)
       "remove": [container titles],   # un-route from (correct a bad route)
       "augi_tags": [taxonomy tags]}   # overwrite classification (like tag_block)

    Any subset of add/remove/augi_tags per decision. "Move out of A into B"
    is one decision with both add and remove. Works for a whole pass's batch
    or a single correction. Both add and remove are idempotent: duplicate
    adds and removes of absent routes are no-ops (no-op removes are counted
    in 'routes_not_found', not errors).

    Home by construction: a block is already a member of the note it
    physically lives in — adding a route to the block's own source note is
    a no-op counted in 'already_home' (no edge is written), and removing
    that containment is an error (move or delete the text itself instead).

    Container titles must match exactly; unknown containers or block_ids fail
    that decision only (reported in 'errors'), the rest still apply. Current
    membership is visible via get_members (containment + routes unified),
    get_block/get_blocks ('routed_to'), or get_related(kind="routed_to")."""
    from openaugi.model.link import Link

    store = _get_store()

    block_ids: list[str] = [d["block_id"] for d in decisions if isinstance(d.get("block_id"), str)]
    known_blocks = store.get_blocks_by_ids(block_ids)

    container_titles = {t for d in decisions for t in [*_decision_adds(d), *d.get("remove", [])]}
    container_ids: dict[str, str] = {}
    for title in container_titles:
        row = store.conn.execute(
            "SELECT id FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
            (title,),
        ).fetchone()
        if row:
            container_ids[title] = row[0]

    links: list[Link] = []
    routed = 0
    removed = 0
    removes_not_found = 0
    already_home = 0
    tagged = 0
    errors: list[dict] = []
    for d in decisions:
        block_id = d.get("block_id")
        if not block_id or block_id not in known_blocks:
            errors.append({"block_id": block_id, "reason": "Block not found."})
            continue
        adds = _decision_adds(d)
        removes = d.get("remove", [])
        if not adds and not removes and not d.get("augi_tags"):
            errors.append(
                {
                    "block_id": block_id,
                    "reason": "Decision has none of add/remove/augi_tags.",
                }
            )
            continue
        home_id = store.get_contains_parent_id(block_id) if (adds or removes) else None
        for title in adds:
            container_id = container_ids.get(title)
            if container_id is None:
                errors.append(
                    {
                        "block_id": block_id,
                        "reason": f"Container note not found: {title}",
                        "hint": "Use search(title=...) to find the exact note title.",
                    }
                )
                continue
            if container_id == home_id:
                already_home += 1
                continue
            links.append(Link(from_id=block_id, to_id=container_id, kind="routed_to"))
            routed += 1
        for title in removes:
            container_id = container_ids.get(title)
            if container_id is None:
                errors.append(
                    {
                        "block_id": block_id,
                        "reason": f"Container note not found: {title}",
                        "hint": "Use search(title=...) to find the exact note title.",
                    }
                )
                continue
            if container_id == home_id:
                errors.append(
                    {
                        "block_id": block_id,
                        "reason": f"Block physically lives in '{title}' — containment "
                        "can't be removed by unrouting. Move or delete the text itself.",
                    }
                )
                continue
            if store.delete_link(block_id, container_id, "routed_to"):
                removed += 1
            else:
                removes_not_found += 1
        if d.get("augi_tags"):
            store.update_block_metadata(block_id, {"augi_tags": d["augi_tags"]})
            tagged += 1

    if links:
        store.insert_links(links)

    return _json(
        {
            "status": "ok" if not errors else "partial",
            "decisions": len(decisions),
            "routes_applied": routed,
            "routes_removed": removed,
            "routes_not_found": removes_not_found,
            "already_home": already_home,
            "blocks_tagged": tagged,
            "errors": errors,
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_members(container_title: str, limit: int = 100, offset: int = 0) -> str:
    """Members of a container note under the unified membership rule.

    A block is a member iff it is routed to the container (routed_to link)
    OR physically lives in it (the user pasted/wrote it there) — the two are
    the same fact seen from different sides. Each member carries a
    'membership' field: 'contained' (lives there), 'routed' (linked there),
    or 'both'. Newest first.

    This is THE query views render: use it to build a container's membership
    log instead of assembling routed_to links by hand."""
    store = _get_store()
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (container_title,),
    ).fetchone()
    if not row:
        return _json(
            {
                "status": "error",
                "reason": f"Container note not found: {container_title}",
                "hint": "Use search(title=...) to find the exact note title.",
            }
        )
    members = store.get_container_members(row[0])
    page = members[offset : offset + limit]
    return _json(
        {
            "container": container_title,
            "container_id": row[0],
            "members": [{**_block_summary(b), "membership": membership} for b, membership in page],
            "count": len(page),
            "total": len(members),
            "has_more": offset + limit < len(members),
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_view(container_title: str, member_limit: int = 50) -> str:
    """Render a container's view from the DB — the saved query, not a file.

    A view = membership log (computed live via the unified rule, always
    fresh) + the cached recap (LLM synthesis written by write_recap, with
    staleness visible: 'stale' is true when membership changed since the
    recap was generated). This is what rendered surfaces (plugin pane,
    mobile bridge, markdown export) consume; prefer it over reading
    View - *.md files."""
    store = _get_store()
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (container_title,),
    ).fetchone()
    if not row:
        return _json(
            {
                "status": "error",
                "reason": f"Container note not found: {container_title}",
                "hint": "Use search(title=...) to find the exact note title.",
            }
        )
    container_id = row[0]
    members = store.get_container_members(container_id)
    recap_row = store.get_recap(container_id)
    recap = None
    if recap_row is not None:
        recap = {
            "recap_md": recap_row["recap_md"],
            "generated_at": recap_row["generated_at"],
            "stale": recap_row["membership_hash"] != store.membership_hash(container_id),
        }
    return _json(
        {
            "container": container_title,
            "container_id": container_id,
            "recap": recap,
            "members": [
                {**_block_summary(b), "membership": membership}
                for b, membership in members[:member_limit]
            ],
            "member_count": len(members),
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def list_views() -> str:
    """The render list: every container with a cached recap, newest first.

    A container appears here once the review pass has written its recap row
    (write_recap) — that row IS the "this container has a view" bit.
    Containers the user curates entirely by hand (recap off, e.g. a dream
    journal) never show up. Surfaces iterate this list and call get_view
    per container."""
    store = _get_store()
    views = []
    for r in store.list_recaps():
        views.append(
            {
                "container": r["title"],
                "container_id": r["container_id"],
                "generated_at": r["generated_at"],
                "stale": r["membership_hash"] != store.membership_hash(r["container_id"]),
            }
        )
    return _json({"views": views, "count": len(views)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
@_release_conn
def write_recap(container_title: str, recap_md: str) -> str:
    """Cache a container's recap (the LLM synthesis half of its view).

    Write this at the end of a review pass for every container whose
    membership changed materially (Q2 decision: recaps refresh on pass
    only). The current membership hash is stored alongside, so get_view
    can show staleness instead of hiding it. Overwrites any prior recap —
    a recap is a cache row, never an archive."""
    from datetime import datetime

    store = _get_store()
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (container_title,),
    ).fetchone()
    if not row:
        return _json(
            {
                "status": "error",
                "reason": f"Container note not found: {container_title}",
                "hint": "Use search(title=...) to find the exact note title.",
            }
        )
    now = datetime.now(UTC).isoformat()
    store.upsert_recap(row[0], recap_md, now, store.membership_hash(row[0]))
    return _json(
        {
            "status": "ok",
            "container": container_title,
            "generated_at": now,
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_review_state() -> str:
    """Get the review-pass high-water mark.

    Returns last_run (ISO timestamp of the last completed review pass) and
    last_summary (its one-line summary). Call at the start of a review pass
    to scope which blocks are new: search(after_ingested=last_run) — filter
    on ingest time, NOT after=, which compares content dates and misses
    same-day date-only blocks and re-ingested edits. If last_run is null
    this is the first run — backfill from a sensible date instead.
    """
    store = _get_store()
    return _json(store.get_review_state())


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
@_release_conn
def mark_review_complete(summary: str) -> str:
    """Mark a review pass complete — advances the high-water mark to now.

    Call ONCE at the end of a successful review pass with a one-line summary
    of the run (e.g. "routed 42 blocks; regenerated 4 views + Dashboard").
    The next pass will only process blocks newer than this point.
    """
    from datetime import datetime

    now = datetime.now(UTC).isoformat()
    store = _get_store()
    store.set_review_state(now, summary)
    return _json({"status": "ok", "last_run": now, "summary": summary})


# ── Write Tools ────────────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def write_document(
    title: str,
    description: str,
    content: str,
    subfolder: str = "Notes",
    overwrite: bool = False,
    extra_frontmatter: dict[str, str] | None = None,
) -> str:
    """Save something to the user's vault under OpenAugi/{subfolder}/.

    Use any time the user says "save this", "write this to augi / openaugi / auggie",
    "save this to my vault", or explicitly asks to persist something.
    Also use for substantial agent output: notes, research, summaries, drafts.

    - title: Note title (becomes the filename). Must be a valid Obsidian title.
    - description: One-line summary — goes in frontmatter, used for scanning.
    - content: Markdown body. Frontmatter (type, description, created) is auto-generated.
    - subfolder: Where to write under OpenAugi/. Infer from content:
        'Notes' for raw ideas or captures (default),
        'Docs' for structured reference output,
        'Research' for investigation results,
        'Views' for regenerable derived views (review pass output).
      Cannot escape the OpenAugi/ root.
    - overwrite: Replace an existing file. Use ONLY for regenerable derived
      views (subfolder='Views'); never overwrite notes.
    - extra_frontmatter: Optional machine-readable frontmatter keys, e.g.
      {"lens": "echoes"} when a lens writes its output — this is what keeps
      the Dashboard's Lenses section reconstructible from disk.
      Reserved keys (type/description/created) are ignored.

    Requires vault path configured via 'openaugi init' or OPENAUGI_VAULT_PATH env var."""
    from openaugi.mcp.doc_writer import VaultWriter

    vault_path = _get_vault_path()
    if not vault_path:
        return _json(
            {
                "status": "error",
                "reason": (
                    "No vault path configured. "
                    "Run 'openaugi init' to set a default vault, "
                    "or set OPENAUGI_VAULT_PATH environment variable."
                ),
            }
        )

    writer = VaultWriter(vault_path)
    return _json(
        writer.write_document(
            title,
            description,
            content,
            subfolder=subfolder,
            overwrite=overwrite,
            extra_frontmatter=extra_frontmatter,
        )
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
@_release_conn
def write_context_pack() -> str:
    """Regenerate <vault>/OpenAugi/context-pack.json — the mobile capture-assist sidecar.

    Call once near the end of a review pass (after routing, alongside the
    Dashboard regeneration). Assembles taxonomy (curated note + top DB tags),
    recentConcepts (containers by routing recency), and noteTitles from the
    DB, then writes the JSON file the mobile bridge serves to the phone.

    Requires vault path configured via 'openaugi init' or OPENAUGI_VAULT_PATH env var."""
    from openaugi.pipeline.context_pack import write_context_pack as _write_pack

    vault_path = _get_vault_path()
    if not vault_path:
        return _json(
            {
                "status": "error",
                "reason": (
                    "No vault path configured. "
                    "Run 'openaugi init' to set a default vault, "
                    "or set OPENAUGI_VAULT_PATH environment variable."
                ),
            }
        )

    out = _write_pack(_get_store(), vault_path)
    return _json({"status": "ok", "path": str(out)})


# ── Resources ──────────────────────────────────────────────────────


@mcp.resource("vault://note/{title}")
@_release_conn
def get_note_resource(title: str) -> str:
    """All entries for a note by title, plus hub context.

    Shows up in Claude Code's @ autocomplete as @openaugi:vault://note/Title.
    Use to deep-read a specific note after finding it via search or hubs."""
    store = _get_store()

    # Find document block by title
    rows = store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (title,),
    ).fetchall()

    if not rows:
        # Fall back to FTS search on title
        fts = store.search_fts(title, limit=5)
        doc_blocks = [b for b in fts if b.kind == "context_block:document"]
        if not doc_blocks:
            return _json(
                {
                    "error": f"Note not found: {title}",
                    "hint": "Use search(title=...) to find notes by partial title match.",
                }
            )
        doc_id = doc_blocks[0].id
    else:
        doc_id = rows[0][0]

    entries = store.get_entries_for_document(doc_id)
    hub_links_in = store.get_links_to(doc_id)
    hub_links_out = store.get_links_from(doc_id)
    entry_routes = store.get_routed_container_titles([e.id for e in entries])

    return _json(
        {
            "note_title": title,
            "doc_id": doc_id,
            "entries": [_block_full(e, entry_routes.get(e.id, [])) for e in entries],
            "entry_count": len(entries),
            "inbound_links": len(hub_links_in),
            "outbound_links": len(hub_links_out),
        }
    )


# ── Helpers ────────────────────────────────────────────────────────


def _decision_adds(d: dict) -> list[str]:
    """Containers to route into — "add" plus its legacy alias "containers"."""
    return [*d.get("add", []), *d.get("containers", [])]


def _block_summary(block) -> dict:
    return {
        "id": block.id,
        "kind": block.kind,
        "title": block.title,
        "content": (block.content or "")[:500],
        "tags": block.tags,
        "augi_tags": block.metadata.get("augi_tags", []),
        "block_time": block.block_time,
        "source": block.source,
        "source_path": block.metadata.get("source_path", ""),
    }


def _block_full(block, routed_to: list[str] | None = None) -> dict:
    return {
        "id": block.id,
        "kind": block.kind,
        "title": block.title,
        "content": block.content,
        "summary": block.summary,
        "tags": block.tags,
        "augi_tags": block.metadata.get("augi_tags", []),
        "routed_to": routed_to or [],
        "block_time": block.block_time,
        "occurred_at": block.occurred_at,
        "source": block.source,
        "metadata": block.metadata,
        "content_hash": block.content_hash,
        "ingested_at": block.ingested_at,
    }


# ── Entry point ────────────────────────────────────────────────────


def run_server(
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8787,
    auth_provider: str | None = None,
):
    """Start the MCP server.

    Args:
        transport: "stdio" for Claude Desktop/Code, "streamable-http" for remote access.
        host: HTTP host (only used with streamable-http transport).
        port: HTTP port (only used with streamable-http transport).
        auth_provider: Optional auth provider (e.g. "cloudflare"). Only for HTTP transport.
    """
    if transport != "stdio":
        mcp.settings.host = host
        mcp.settings.port = port
        # Allow tunnel hostnames through DNS rebinding protection.
        # OPENAUGI_ALLOWED_HOSTS is a comma-separated list of hostnames
        # that can reach this server (e.g. via Cloudflare Tunnel).
        allowed = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
        extra_hosts = os.environ.get("OPENAUGI_ALLOWED_HOSTS", "")
        if extra_hosts:
            allowed.extend(h.strip() for h in extra_hosts.split(",") if h.strip())
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed,
        )
        logger.info("Starting MCP server on http://%s:%d/mcp", host, port)

    if auth_provider:
        from openaugi.auth import configure_auth

        config = load_config()
        configure_auth(mcp, auth_provider, config)
        logger.info("Auth provider configured: %s", auth_provider)

    mcp.run(transport=transport)


if __name__ == "__main__":
    run_server()
