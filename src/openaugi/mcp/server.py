"""OpenAugi MCP Server — query + write tools for Claude.

Read tools (readOnlyHint):
- search: semantic (sqlite-vec KNN) + title (FTS5 title-only) + keyword (FTS5) + filters
- get_block: full block content by ID
- get_blocks: batch fetch multiple blocks by ID (up to 50)
- get_related: follow links from a block
- traverse: multi-hop graph walk
- get_context: compound search → expand → structured result
- recent: recently created blocks

Write tools:
- write_document: create a markdown note in OpenAugi/{subfolder}/
- write_snip: save a curated snippet to OpenAugi/Snips/

Resources:
- vault://note/{title}: all entries for a note + hub context
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from openaugi.config import load_config
from openaugi.models import get_embedding_model
from openaugi.pipeline.rerank import rerank as _rerank
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
        _store = SQLiteStore(_get_db_path(), read_only=True)
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
    k: int = 20,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    kind: str | None = None,
    source: str | None = None,
) -> str:
    """Search the knowledge base. Returns block summaries (not full content).

    Use this as your first step to find relevant blocks. Four modes:
    - Title: provide 'title' to search note/document titles only (e.g. title="meeting notes")
    - Semantic: provide 'query' for vector similarity search (best for concepts/questions)
    - Keyword: provide 'keyword' for FTS5 full-text search (best for exact terms)
    - Browse: provide only filters (tags, after, before, kind, source)

    Do NOT use this to get full block content — use get_block or get_blocks for that.
    For a complete research workflow (search + deduplicate + expand), use get_context instead.

    Prefer 'title' over 'keyword' when you know the note name.
    Must provide at least one of: query, keyword, title, or a filter.
    Results are capped at k (default 20). If count == k, there may be more — increase k or
    narrow your filters.
    Dates use ISO format: after="2025-01-01", before="2025-06-01"."""
    if not query and not keyword and not title and not any([tags, after, before, kind, source]):
        return _json(
            {
                "error": "No search parameters provided.",
                "hint": "Provide at least one of: query (semantic), keyword (FTS), "
                "title, or filters (tags, after, before, kind, source). "
                "Example: search(keyword='project plan')",
            }
        )

    store = _get_store()

    if title:
        results = store.search_fts(f"title:{title}", limit=k + 1)
        has_more = len(results) > k
        results = results[:k]
        return _json(
            {
                "results": [_block_summary(b) for b in results],
                "count": len(results),
                "has_more": has_more,
                "mode": "title",
            }
        )

    if keyword:
        results = store.search_fts(keyword, limit=k + 1)
        has_more = len(results) > k
        results = results[:k]
        return _json(
            {
                "results": [_block_summary(b) for b in results],
                "count": len(results),
                "has_more": has_more,
                "mode": "keyword",
            }
        )

    if query:
        query_vec = _get_embedding_model().embed_query(query)
        hits = store.semantic_search(query_vec, k=k * 3)

        # Batch-fetch all hit blocks
        hit_ids = [block_id for block_id, _ in hits]
        blocks_map = store.get_blocks_by_ids(hit_ids)

        results = []
        for block_id, distance in hits:
            block = blocks_map.get(block_id)
            if block is None:
                continue
            if kind and block.kind != kind:
                continue
            if source and block.source != source:
                continue
            if tags and not set(tags).intersection(block.tags):
                continue
            if after and (block.timestamp or "") < after:
                continue
            if before and (block.timestamp or "") > before:
                continue
            summary = _block_summary(block)
            summary["score"] = round(1.0 - distance, 4)
            results.append(summary)
            if len(results) > k:
                break

        has_more = len(results) > k
        results = results[:k]
        return _json(
            {
                "results": results,
                "count": len(results),
                "has_more": has_more,
                "mode": "semantic",
            }
        )

    # Browse mode — filter blocks
    blocks = store.get_blocks_by_kind(kind or "entry", limit=k * 3)
    results = []
    for b in blocks:
        if source and b.source != source:
            continue
        if tags and not set(tags).intersection(b.tags):
            continue
        if after and (b.timestamp or "") < after:
            continue
        if before and (b.timestamp or "") > before:
            continue
        results.append(_block_summary(b))
        if len(results) > k:
            break

    has_more = len(results) > k
    results = results[:k]
    return _json(
        {
            "results": results,
            "count": len(results),
            "has_more": has_more,
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
    return _json(_block_full(block))


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
    return _json(
        {
            "blocks": [_block_full(found[bid]) for bid in block_ids if bid in found],
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
    - kind: filter by link kind. Common kinds: 'split_from' (entry→document),
      'tagged' (entry→tag), 'links_to' (wikilink between notes)
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
    - link_kinds: restrict to specific link types (e.g. ['links_to', 'tagged'])
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
) -> str:
    """Primary research tool. Use this as the default for answering questions against the
    knowledge base — it runs a full retrieval pipeline in one call:
    semantic search + keyword search → deduplicate → diversity re-rank (MMR) → expand via links.

    Returns 'direct_results' (top blocks) and 'expanded' (linked blocks for extra context).
    Prefer this over manual search → get_block → get_related chains.
    Use plain 'search' only when you need specific search modes (title-only, browse filters)
    or fine-grained control over results.

    - k: number of final results (default 10). Internally overfetches 3x for dedup.
    - expand: follow links from top results for richer context (default true)"""
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

    return _json(
        {
            "query": query,
            "direct_results": list(blocks_seen.values())[:k],
            "expanded": expanded[:k],
            "total_blocks": len(blocks_seen),
        }
    )


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
    - kind: block kind to filter by (default 'entry'). Common kinds: 'entry', 'document', 'tag'
    - source: filter by source (e.g. 'vault')
    - tags: filter to blocks matching any of these tags"""
    store = _get_store()
    target_kind = kind or "entry"
    blocks = store.get_blocks_by_kind(target_kind, limit=k * 3)

    results = []
    for b in blocks:
        if source and b.source != source:
            continue
        if tags and not set(tags).intersection(b.tags):
            continue
        results.append(_block_summary(b))
        if len(results) > k:
            break

    has_more = len(results) > k
    results = results[:k]
    return _json({"results": results, "count": len(results), "has_more": has_more})


# ── Write Tools ────────────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def write_document(
    title: str,
    description: str,
    content: str,
    subfolder: str = "Notes",
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
        'Research' for investigation results.
      Cannot escape the OpenAugi/ root.

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
    return _json(writer.write_document(title, description, content, subfolder=subfolder))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def write_thread(
    topic: str,
    description: str,
    content: str,
) -> str:
    """Save a distilled session note to OpenAugi/Threads/.

    Use when the user says "save this thread", "log this session",
    "save our conversation", or when finishing a non-trivial task and
    wanting to capture the output. Not a transcript — synthesize:
    what was the intent, what was decided, what was learned, what's still open.

    - topic: Short title for the session (becomes part of the filename).
    - description: One-line summary — goes in frontmatter, used for scanning.
    - content: Markdown body. Write whatever structure fits the session.

    Writes to OpenAugi/Threads/YYYY-MM-DD - {topic}.md.
    Handles collisions by appending -2, -3, etc.

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
    return _json(writer.write_thread(topic, description, content))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def write_snip(
    title: str,
    content: str,
    description: str = "",
    stream: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Save a curated snippet to OpenAugi/Snips/.

    Use when the user highlights text and says "save this", "snip this",
    or when capturing a key insight from a conversation. Snips are atomic
    captures — distilled ideas, not raw transcripts.

    - title: Short descriptive title (becomes filename)
    - content: The captured text, optionally refined/distilled
    - description: One-line summary for scanning
    - stream: Workstream this snip belongs to (optional)
    - tags: Tags for categorization (optional)

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
    return _json(writer.write_snip(title, content, description, stream=stream, tags=tags))


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
        "SELECT id FROM blocks WHERE kind = 'document' AND title = ? LIMIT 1",
        (title,),
    ).fetchall()

    if not rows:
        # Fall back to FTS search on title
        fts = store.search_fts(title, limit=5)
        doc_blocks = [b for b in fts if b.kind == "document"]
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

    return _json(
        {
            "note_title": title,
            "doc_id": doc_id,
            "entries": [_block_full(e) for e in entries],
            "entry_count": len(entries),
            "inbound_links": len(hub_links_in),
            "outbound_links": len(hub_links_out),
        }
    )


# ── Helpers ────────────────────────────────────────────────────────


def _block_summary(block) -> dict:
    return {
        "id": block.id,
        "kind": block.kind,
        "title": block.title,
        "content": (block.content or "")[:500],
        "tags": block.tags,
        "timestamp": block.timestamp,
        "source": block.source,
    }


def _block_full(block) -> dict:
    return {
        "id": block.id,
        "kind": block.kind,
        "title": block.title,
        "content": block.content,
        "summary": block.summary,
        "tags": block.tags,
        "timestamp": block.timestamp,
        "occurred_at": block.occurred_at,
        "source": block.source,
        "metadata": block.metadata,
        "content_hash": block.content_hash,
        "created_at": block.created_at,
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
