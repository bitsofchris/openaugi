"""OpenAugi MCP Server — query + write tools for Claude.

Read tools (readOnlyHint):
- search: semantic (sqlite-vec KNN) + title (FTS5 title-only) + keyword (FTS5) + filters
- get_block: full block content by ID
- get_related: follow links from a block
- traverse: multi-hop graph walk
- get_context: compound search → expand → structured result
- recent: recently created blocks

Write tools:
- write_document: create a markdown note in OpenAugi/{subfolder}/

Resources:
- vault://note/{title}: all entries for a note + hub context
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from openaugi.config import load_config
from openaugi.models import get_embedding_model
from openaugi.pipeline.rerank import rerank as _rerank
from openaugi.store.sqlite import SQLiteStore

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
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
        _embedding_model = get_embedding_model(
            config.get("models", {}).get("embedding")
        )
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
    """Search knowledge base. Four modes:
    - Title: provide 'title' to search note/document titles only — use this when
      looking for a specific note by name (e.g. title="meeting notes", title="project plan")
    - Semantic: provide 'query' for vector similarity search
    - Keyword: provide 'keyword' for FTS5 full-text search (title + content + tags)
    - Browse: provide filters (tags, after, before, kind, source)

    Prefer 'title' over 'keyword' when you know the note name or want to navigate by title.
    Must provide at least one of: query, keyword, title, or a filter."""
    if not query and not keyword and not title and not any([tags, after, before, kind, source]):
        return _json({"error": "Provide query, keyword, title, or at least one filter"})

    store = _get_store()

    if title:
        results = store.search_fts(f"title:{title}", limit=k)
        return _json({
            "results": [_block_summary(b) for b in results],
            "count": len(results),
            "mode": "title",
        })

    if keyword:
        results = store.search_fts(keyword, limit=k)
        return _json({
            "results": [_block_summary(b) for b in results],
            "count": len(results),
            "mode": "keyword",
        })

    if query:
        query_vec = _get_embedding_model().embed_query(query)
        hits = store.semantic_search(query_vec, k=k * 3)

        results = []
        for block_id, distance in hits:
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
            summary["score"] = round(1.0 - distance, 4)
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
            source_block = store.get_block(lnk.from_id)
            if source_block:
                results.append({
                    "block": _block_summary(source_block),
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
    """Power tool: multi-prong search → deduplicate → diversity re-rank → expand via links.
    Combines semantic + keyword search, deduplicates redundant chunks via embedding
    similarity grouping, re-ranks for diversity (MMR), then follows links from top
    results to build a rich context package for Claude."""
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

    # Fetch full blocks only for the final k IDs
    blocks_seen: dict[str, dict] = {}
    for block_id in final_ids:
        block = store.get_block(block_id)
        if block:
            entry = {**_block_summary(block), "score": candidate_scores.get(block_id, 0.0)}
            entry["source"] = "fts" if block_id in {b.id for b in fts_results} else "semantic"
            blocks_seen[block_id] = entry

    # Expand: follow links from top results
    expanded: list[dict] = []
    if expand:
        for block_id in final_ids:
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
        return _json({
            "status": "error",
            "reason": (
                "No vault path configured. "
                "Run 'openaugi init' to set a default vault, "
                "or set OPENAUGI_VAULT_PATH environment variable."
            ),
        })

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
        return _json({
            "status": "error",
            "reason": (
                "No vault path configured. "
                "Run 'openaugi init' to set a default vault, "
                "or set OPENAUGI_VAULT_PATH environment variable."
            ),
        })

    writer = VaultWriter(vault_path)
    return _json(writer.write_thread(topic, description, content))


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
            return _json({"error": f"Note not found: {title}"})
        doc_id = doc_blocks[0].id
    else:
        doc_id = rows[0][0]

    entries = store.get_entries_for_document(doc_id)
    hub_links_in = store.get_links_to(doc_id)
    hub_links_out = store.get_links_from(doc_id)

    return _json({
        "note_title": title,
        "doc_id": doc_id,
        "entries": [_block_full(e) for e in entries],
        "entry_count": len(entries),
        "inbound_links": len(hub_links_in),
        "outbound_links": len(hub_links_out),
    })


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

def run_server():
    """Start the MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
