"""OpenAugi MCP Server — 6 query tools for Claude.

Port of v1's 13-tool MCP server to the blocks+links data model.
All tools are read-only. Connection released after each call.

Tools:
- search: semantic (FAISS) + keyword (FTS5) + filters
- get_block: full block content by ID
- get_related: follow links from a block
- traverse: multi-hop graph walk
- get_context: compound search → expand → structured result
- recent: recently created blocks
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from openaugi.config import load_config
from openaugi.models import get_embedding_model
from openaugi.pipeline.embed import build_faiss_index
from openaugi.store.sqlite import SQLiteStore

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)

mcp = FastMCP("openaugi")

# ── State (initialized lazily) ─────────────────────────────────────

_store: SQLiteStore | None = None
_faiss_index = None
_embedding_model = None
_db_path: str | None = None
_db_mtime: float = 0


def _get_db_path() -> str:
    """Resolve DB path from env or default."""
    return os.environ.get("OPENAUGI_DB", str(Path.home() / ".openaugi" / "openaugi.db"))


def _get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore(_get_db_path(), read_only=True)
    return _store


def _check_freshness():
    """Invalidate FAISS if DB was modified."""
    global _db_mtime, _faiss_index
    db_path = _get_db_path()
    if os.path.exists(db_path):
        current = os.path.getmtime(db_path)
        if current > _db_mtime:
            _faiss_index = None
            _db_mtime = current


def _get_faiss():
    global _faiss_index, _embedding_model
    if _faiss_index is None:
        store = _get_store()
        config = load_config()
        if _embedding_model is None:
            _embedding_model = get_embedding_model(
                config.get("models", {}).get("embedding")
            )
        _faiss_index = build_faiss_index(store, dim=_embedding_model.dimensions)
    return _faiss_index


def _release_conn(fn):
    """Close SQLite connection after each tool call."""

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


# ── Tools ──────────────────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def search(
    query: str | None = None,
    keyword: str | None = None,
    k: int = 20,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    kind: str | None = None,
    source: str | None = None,
) -> str:
    """Search knowledge base. Three modes:
    - Semantic: provide 'query' for FAISS cosine similarity
    - Keyword: provide 'keyword' for FTS5 full-text search
    - Browse: provide filters (tags, after, before, kind, source)

    Must provide at least one of: query, keyword, or a filter."""
    if not query and not keyword and not any([tags, after, before, kind, source]):
        return _json({"error": "Provide query, keyword, or at least one filter"})

    _check_freshness()
    store = _get_store()

    if keyword:
        results = store.search_fts(keyword, limit=k)
        return _json({
            "results": [_block_summary(b) for b in results],
            "count": len(results),
            "mode": "keyword",
        })

    if query:
        global _embedding_model
        config = load_config()
        if _embedding_model is None:
            _embedding_model = get_embedding_model(
                config.get("models", {}).get("embedding")
            )
        faiss_index = _get_faiss()
        query_vec = _embedding_model.embed_query(query)
        hits = faiss_index.search(query_vec, k=k * 3)

        results = []
        for block_id, score in hits:
            block = store.get_block(block_id)
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
            summary["score"] = score
            results.append(summary)
            if len(results) >= k:
                break

        return _json({"results": results, "count": len(results), "mode": "semantic"})

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
        if len(results) >= k:
            break

    return _json({"results": results, "count": len(results), "mode": "browse"})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_block(block_id: str) -> str:
    """Get full block content and metadata by ID."""
    store = _get_store()
    block = store.get_block(block_id)
    if block is None:
        return _json({"error": f"Block not found: {block_id}"})
    return _json(_block_full(block))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_related(
    block_id: str,
    kind: str | None = None,
    direction: str = "both",
    limit: int = 50,
) -> str:
    """Follow links from/to a block. Returns connected blocks.
    - direction: 'out' (from block), 'in' (to block), or 'both'
    - kind: filter by link kind (split_from, tagged, links_to, etc.)"""
    store = _get_store()

    results = []
    if direction in ("out", "both"):
        links = store.get_links_from(block_id, kind=kind)
        for lnk in links[:limit]:
            target = store.get_block(lnk.to_id)
            if target:
                results.append({
                    "block": _block_summary(target),
                    "link_kind": lnk.kind,
                    "direction": "out",
                })

    if direction in ("in", "both"):
        links = store.get_links_to(block_id, kind=kind)
        for lnk in links[:limit]:
            source = store.get_block(lnk.from_id)
            if source:
                results.append({
                    "block": _block_summary(source),
                    "link_kind": lnk.kind,
                    "direction": "in",
                })

    return _json({"block_id": block_id, "related": results, "count": len(results)})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def traverse(
    start_id: str,
    max_hops: int = 2,
    link_kinds: list[str] | None = None,
    limit: int = 50,
) -> str:
    """Multi-hop graph walk from a starting block.
    Follows links up to max_hops deep. Returns all reachable blocks."""
    store = _get_store()

    visited: set[str] = set()
    results: list[dict] = []
    frontier = [(start_id, 0)]

    while frontier and len(results) < limit:
        current_id, depth = frontier.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        block = store.get_block(current_id)
        if block and current_id != start_id:
            results.append({**_block_summary(block), "depth": depth})

        if depth < max_hops:
            links_out = store.get_links_from(current_id)
            links_in = store.get_links_to(current_id)
            for lnk in links_out + links_in:
                next_id = lnk.to_id if lnk.from_id == current_id else lnk.from_id
                if next_id not in visited:
                    if link_kinds and lnk.kind not in link_kinds:
                        continue
                    frontier.append((next_id, depth + 1))

    return _json({
        "start_id": start_id,
        "results": results,
        "count": len(results),
        "max_hops": max_hops,
    })


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def get_context(
    query: str,
    k: int = 10,
    expand: bool = True,
) -> str:
    """Power tool: multi-prong search → expand via links → structured context.
    Combines semantic + keyword search, then follows links from top results
    to build a rich context package for Claude."""
    _check_freshness()
    store = _get_store()

    blocks_seen: dict[str, dict] = {}

    # Prong 1: keyword/FTS
    fts_results = store.search_fts(query, limit=k)
    for b in fts_results:
        if b.id not in blocks_seen:
            blocks_seen[b.id] = {**_block_summary(b), "source": "fts"}

    # Prong 2: semantic (if FAISS available)
    try:
        global _embedding_model
        config = load_config()
        if _embedding_model is None:
            _embedding_model = get_embedding_model(
                config.get("models", {}).get("embedding")
            )
        faiss_index = _get_faiss()
        if faiss_index.size > 0:
            query_vec = _embedding_model.embed_query(query)
            hits = faiss_index.search(query_vec, k=k)
            for block_id, score in hits:
                if block_id not in blocks_seen:
                    block = store.get_block(block_id)
                    if block:
                        blocks_seen[block_id] = {
                            **_block_summary(block),
                            "score": score,
                            "source": "semantic",
                        }
    except Exception:
        pass  # Semantic search is optional

    # Expand: follow links from top results
    expanded: list[dict] = []
    if expand:
        for block_id in list(blocks_seen.keys())[:k]:
            links = store.get_links_from(block_id)
            for lnk in links[:5]:
                if lnk.to_id not in blocks_seen:
                    target = store.get_block(lnk.to_id)
                    if target:
                        entry = {
                            **_block_summary(target),
                            "expanded_from": block_id,
                            "link_kind": lnk.kind,
                        }
                        expanded.append(entry)
                        blocks_seen[lnk.to_id] = entry

    return _json({
        "query": query,
        "direct_results": list(blocks_seen.values())[:k],
        "expanded": expanded[:k],
        "total_blocks": len(blocks_seen),
    })


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
@_release_conn
def recent(
    k: int = 20,
    kind: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Recently created blocks, filtered by kind/source/tags."""
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
        if len(results) >= k:
            break

    return _json({"results": results, "count": len(results)})


# ── Helpers ────────────────────────────────────────────────────────


def _block_summary(block) -> dict:
    """Compact block representation for tool results."""
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
    """Full block representation."""
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

def run_server():
    """Start the MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
