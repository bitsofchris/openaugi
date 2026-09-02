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

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from openaugi.config import load_config, resolve_vault_path
from openaugi.http_api import register_api_routes
from openaugi.models import get_embedding_model
from openaugi.query import QuerySpec, engine, saved
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
    """Resolve vault path: env var > config.toml > None (with ~ expanded)."""
    vault = os.environ.get("OPENAUGI_VAULT_PATH")
    if vault:
        return resolve_vault_path(vault)
    return resolve_vault_path(config=load_config())


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
    include_path_prefix: str | None = None,
    has_task: bool | None = None,
    provenance: list[str] | None = None,
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
    of a review queue). include_path_prefix is the mirror — keep ONLY blocks
    under that prefix. Both work in every mode.

    Use them as a PAIR, in two queries, to reach one folder inside an
    otherwise-excluded tree. The review pass does exactly this:

        search(after_ingested=since, exclude_path_prefix="OpenAugi/")
        search(after_ingested=since, include_path_prefix="OpenAugi/Capture/")

    Everything under OpenAugi/ is generated output EXCEPT Capture/, which is
    the user's mobile capture stream — their voice notes, aaa: instructions,
    and Dashboard answers. Excluding the tree and then naming the one folder
    of truth means a new generated folder is excluded automatically, instead
    of silently leaking into the queue.

    provenance keeps only blocks written by the named authors, any of
    "human" (the user's own writing), "ai" (model output — everything under
    OpenAugi/ except Capture/, plus notes tagged as AI summaries or pasted
    chats), "reference" (imported material: Readwise, Snipd, web clips,
    gdrive). Set at ingest from [vault.provenance_rules] and tags. Works in
    every mode. When omitted, SEMANTIC mode drops the config default
    [retrieval] exclude_provenance (reference, out of the box) so imported
    quotes do not crowd out the user's thinking; pass
    provenance=["human","ai","reference"] to search everything. Ask for
    provenance=["human"] when the question is what the user themselves
    thought, so model-written reflections are not quoted back as theirs.

    has_task=True keeps only blocks the user marked as a task — an open
    `- [ ] …` checkbox (metadata has_open_task, extracted at ingest) or a
    type/task tag. Deterministic: this is the Dashboard task-shelf
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
        include_path_prefix=include_path_prefix,
        has_task=has_task,
        provenance=provenance,
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
    result = engine.run(_get_store(), spec, embedding_model=model, config=load_config())
    return _json(_render_run_result(result))


def _with_routes(summaries: list[dict]) -> list[dict]:
    """Stamp `routed_to` onto a page of block summaries.

    One batched lookup for the whole page (never N+1). Membership is the
    question every reader eventually asks — "which context blocks does this
    belong to, and does it belong to any?" — and without it a consumer can't
    tell a routed block from an unrouted one, which is exactly the queue a
    review surface needs to show.
    """
    ids = [s["id"] for s in summaries if isinstance(s.get("id"), str)]
    routes = _get_store().get_routed_container_titles(ids) if ids else {}
    for summary in summaries:
        summary["routed_to"] = routes.get(summary["id"], [])
    return summaries


def _render_run_result(result: engine.RunResult) -> dict:
    """Agent-shaped envelope for an engine RunResult — shared by search and
    run_query. Key order is part of the golden wire format; don't reorder."""
    if result.mode == "semantic":
        results = []
        for b in result.blocks:
            summary = _block_summary(b)
            summary["score"] = result.scores[b.id]
            results.append(summary)
        return {
            "results": _with_routes(results),
            "count": len(results),
            "has_more": result.has_more,
            "mode": "semantic",
        }

    if result.mode in ("title", "keyword"):
        return {
            "results": _with_routes([_block_summary(b) for b in result.blocks]),
            "count": len(result.blocks),
            "has_more": result.has_more,
            "mode": result.mode,
        }

    return {
        "results": _with_routes([_block_summary(b) for b in result.blocks]),
        "count": len(result.blocks),
        "reference_documents": result.reference_documents,
        "reference_block_count": result.reference_block_count,
        "total": result.total,
        "has_more": result.has_more,
        "next_offset": result.next_offset,
        "mode": "browse",
    }


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_block(block_id: str) -> str:
    """Get full block content and metadata by ID.

    Use after search/get_context to read the complete content of a specific block.
    For multiple blocks, use get_blocks instead — one call vs. many.
    Do NOT use this in a loop — use get_blocks with a list of IDs."""
    result = engine.fetch(_get_store(), [block_id])
    if not result.blocks:
        return _json(
            {
                "error": f"Block not found: {block_id}",
                "hint": "This ID may be stale or incorrect. Use search(keyword=...) or "
                "search(title=...) to find valid block IDs.",
            }
        )
    block = result.blocks[0]
    return _json(_block_full(block, result.routes.get(block.id, [])))


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
    result = engine.fetch(_get_store(), block_ids)
    return _json(
        {
            "blocks": [_block_full(b, result.routes.get(b.id, [])) for b in result.blocks],
            "count": len(result.blocks),
            "missing": result.missing,
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
    items = engine.related(_get_store(), block_id, kind=kind, direction=direction, limit=limit)
    results = [
        {"block": _block_summary(i.block), "link_kind": i.link_kind, "direction": i.direction}
        for i in items
    ]
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
    items = engine.traverse(
        _get_store(), start_id, max_hops=max_hops, link_kinds=link_kinds, limit=limit
    )
    results = [{**_block_summary(i.block), "depth": i.depth} for i in items]
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
    after: str | None = None,
    before: str | None = None,
    tags: list[str] | None = None,
    exclude_path_prefix: str | None = None,
    include_path_prefix: str | None = None,
    provenance: list[str] | None = None,
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

    When
    purpose is set — demoted thoughts never resurface proactively.

    Filters — same meaning as on search, applied to the candidate pool
    BEFORE rerank, so the answer is ranked within the scope, not trimmed
    after the fact:
    - after / before: block_time (content date) bounds, ISO dates. Use these
      for "what was I thinking about X in March": get_context(query="X",
      after="2026-03-01", before="2026-03-31").
    - tags: match any of these (user tags + augi_tags).
    - exclude_path_prefix / include_path_prefix: source_path scoping, e.g.
      exclude_path_prefix="OpenAugi/" to keep generated artifacts out.
    - provenance: any of "human", "ai", "reference" (see search). Omitted =
      the config default, which drops "reference"; ask for ["human"] when the
      question is what the user themselves wrote."""
    try:
        model = _get_embedding_model()
    except Exception:
        # Engine falls back to FTS-only, matching historical get_context
        # behavior when the embedding model can't be constructed.
        logger.warning("Embedding model unavailable in get_context", exc_info=True)
        model = None
    ctx = engine.context(
        _get_store(),
        query,
        k=k,
        expand=expand,
        purpose=purpose,
        embedding_model=model,
        config=load_config(),
        after=after,
        before=before,
        tags=tags,
        exclude_path_prefix=exclude_path_prefix,
        include_path_prefix=include_path_prefix,
        provenance=provenance,
    )
    if not ctx.had_candidates:
        return _json({"query": query, "direct_results": [], "expanded": [], "total_blocks": 0})

    result = {
        "query": query,
        "direct_results": [{**_block_summary(e.block), **e.extras} for e in ctx.seen[:k]],
        "expanded": [{**_block_summary(e.block), **e.extras} for e in ctx.expanded[:k]],
        "total_blocks": len(ctx.seen),
    }
    if purpose is not None:
        result["salience"] = {"purpose": purpose, "min_score": ctx.min_score}
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
    blocks = engine.recent(_get_store(), k=k, kind=kind, source=source, tags=tags)
    results = [_block_summary(b) for b in blocks]
    return _json({"results": results, "count": len(results), "has_more": False})


# ── Saved Queries ──────────────────────────────────────────────────


def _no_vault_error() -> str:
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


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def list_queries() -> str:
    """List saved queries — named, user-editable QuerySpecs.

    A saved query is a markdown file at OpenAugi/AGENT/queries/<name>.md:
    frontmatter carries a description plus the query definition, with
    relative-date tokens ("-14d", "today", "$review-mark") resolved when
    it runs — the lens principle applied to retrieval. Execute one with
    run_query(name); edit the file in Obsidian to change what it returns."""
    vault_path = _get_vault_path()
    if not vault_path:
        return _no_vault_error()
    queries = saved.list_saved(vault_path)
    return _json(
        {
            "queries": [
                {
                    "name": q.name,
                    "description": q.description,
                    "spec": q.spec.model_dump(exclude_none=True),
                }
                for q in queries
            ],
            "count": len(queries),
        }
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def run_query(name: str) -> str:
    """Execute a saved query by name (see list_queries).

    Resolves the spec's relative-date tokens ("-14d", "today" against
    today's date; "$review-mark" against the review-pass high-water mark),
    runs it through the same engine as search, and returns the search
    envelope prefixed with the query's name, description, and the resolved
    spec. This is how recurring product queries (Dashboard task shelf,
    review queue) stay data instead of hard-coded conventions."""
    vault_path = _get_vault_path()
    if not vault_path:
        return _no_vault_error()
    try:
        sq = saved.load_saved(vault_path, name)
    except saved.SavedQueryNotFound:
        return _json(
            {
                "error": f"Saved query not found: {name}",
                "hint": "Use list_queries() to see available saved queries.",
            }
        )
    except saved.SavedQueryError as e:
        return _json({"error": f"Saved query '{name}' is invalid: {e}"})

    store = _get_store()
    resolved = saved.resolve_spec(sq.spec, store=store)
    model = _get_embedding_model() if resolved.mode == "semantic" else None
    try:
        result = engine.run(store, resolved, embedding_model=model)
    except engine.EmptyQuerySpec:
        return _json({"error": f"Saved query '{name}' resolves to an empty spec."})
    return _json(
        {
            "query": name,
            "description": sq.description,
            "resolved_spec": resolved.model_dump(exclude_none=True),
            **_render_run_result(result),
        }
    )


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
    try:
        result = engine.members(_get_store(), container_title, limit=limit, offset=offset)
    except engine.ContainerNotFound:
        return _json(
            {
                "status": "error",
                "reason": f"Container note not found: {container_title}",
                "hint": "Use search(title=...) to find the exact note title.",
            }
        )
    return _json(
        {
            "container": container_title,
            "container_id": result.container_id,
            "members": [
                {**_block_summary(m.block), "membership": m.membership} for m in result.members
            ],
            "count": len(result.members),
            "total": result.total,
            "has_more": offset + limit < result.total,
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
    try:
        result = engine.view(_get_store(), container_title, member_limit=member_limit)
    except engine.ContainerNotFound:
        return _json(
            {
                "status": "error",
                "reason": f"Container note not found: {container_title}",
                "hint": "Use search(title=...) to find the exact note title.",
            }
        )
    return _json(
        {
            "container": container_title,
            "container_id": result.container_id,
            "recap": result.recap,
            "members": [
                {**_block_summary(m.block), "membership": m.membership} for m in result.members
            ],
            "member_count": result.member_count,
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
    views = engine.views(_get_store())
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
    return _json(engine.review_state(_get_store()))


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
    """The read-tool wire row.

    `anchor_id` + `ingested_at` are projected so consumers can collapse
    superseded versions of an entry. Blocks are append-only and identity is a
    content hash, so editing a note in Obsidian leaves the previous version in
    the store: two rows sharing `source_path` and `anchor_id` (the Obsidian
    block anchor — the entry's stable identity) with different content hashes.
    The newest `ingested_at` is current. Without these fields a reader can only
    guess by content similarity, which hides genuinely distinct blocks.
    """
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
        "anchor_id": block.metadata.get("anchor_id"),
        "ingested_at": block.ingested_at,
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


# ── HTTP API (/api/*) ───────────────────────────────────────────────
# The HTTP read adapter mounts on this same daemon (one process, one
# store handle). Routes are registered unconditionally — they are only
# reachable when serving with --transport streamable-http; stdio never
# opens a socket. Auth posture matches /mcp (see auth/cloudflare.py).

register_api_routes(
    mcp,
    get_store=_get_store,
    get_embedding_model=_get_embedding_model,
    release_store=lambda: _store.close() if _store is not None else None,
    get_vault_path=_get_vault_path,
)


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


# -------------------------------------------------------------------- records
# A collection store for agent workflow state (docs/reference/records.md).
#
# openaugi knows nothing about what a collection means. "proposals",
# "routings", "passes" are names a *prompt* chose; their shape lives in that
# prompt, and their policy lives in the caller's config. Nothing about any
# one user's conventions is compiled in here.
#
# That boundary is the reason these three tools replaced eight named ones: the
# eight encoded one person's review workflow — including a hardcoded list of
# which routing rules were legitimate — into a general library that other
# vaults are meant to use. Policy in config, mechanism in tools.


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
@_release_conn
def write_record(collection: str, record_id: str, data: dict) -> str:
    """Store one record in a named collection. Creates or replaces.

    A collection is just a name you choose — openaugi does not validate it and
    has no opinion about what it holds. Use this for workflow state an agent
    needs to survive between runs: what a pass did, what is awaiting a human's
    approval, anything a later step or surface has to read back.

    **Use a stable id derived from the subject** (`promote-silver-notes`,
    `pass-2026-08-20:block-abc`) rather than a random one. Re-recording the
    same subject then updates in place instead of stacking duplicates, which
    is how an approval queue avoids re-asking the same question every run.

    This is NOT for knowledge. Blocks and notes are the knowledge layer and
    live in the vault; records are ephemeral machinery that can be dropped
    without losing anything a human wrote.
    """
    from datetime import datetime

    if not collection.strip() or not record_id.strip():
        return _json({"status": "error", "reason": "collection and record_id are required"})
    _get_store().write_record(
        collection.strip(), record_id.strip(), data or {}, datetime.now(UTC).isoformat()
    )
    return _json({"status": "ok", "collection": collection, "record_id": record_id})


@mcp.tool()
@_release_conn
def list_records(
    collection: str,
    where: dict | None = None,
    order: str = "created_at",
    desc: bool = False,
    limit: int = 100,
) -> str:
    """Read a collection, optionally filtered by exact matches on `data` fields.

    `where` matches top-level fields only, by equality — e.g.
    `{"state": "proposed"}` or `{"pass_id": "pass-2026-08-20"}`. Filtering is
    deliberately equality-only: a general store that grows a query language
    becomes a database with a worse dialect. Anything richer, do in the caller.

    `order` is `created_at` (default), `updated_at`, or `id`. Oldest first
    unless `desc` — for a queue of things awaiting a human, oldest first is
    usually right, so a decision that has waited three runs is not buried
    under one raised this morning.
    """
    records = _get_store().list_records(
        collection, where=where, order=order, desc=desc, limit=limit
    )
    return _json({"collection": collection, "records": records, "count": len(records)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
@_release_conn
def update_record(collection: str, record_id: str, patch: dict) -> str:
    """Merge fields into an existing record. Absent keys are left alone.

    Use for state transitions on something already stored — answering a queued
    proposal, marking an action reversed. Returns an error rather than
    creating the record if it is missing: updating something that vanished
    usually means a stale client, and silently creating it hides that.
    """
    from datetime import datetime

    ok = _get_store().update_record(
        collection, record_id, patch or {}, datetime.now(UTC).isoformat()
    )
    if not ok:
        return _json({"status": "error", "reason": f"no such record: {collection}/{record_id}"})
    return _json({"status": "ok", "collection": collection, "record_id": record_id})
