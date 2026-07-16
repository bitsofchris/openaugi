"""HTTP read adapter — plain JSON /api/* routes on the MCP daemon.

The third adapter over the query engine (docs/plans/query-layer.md §3):
UIs (mobile bridge /views, the block explorer) get the SAME semantics as
MCP without the session handshake, and with FULL block content — the
500-char summary truncation is agent presentation and never applies here.

Mounted via FastMCP.custom_route on the existing daemon, so there is one
process and one store handle. Routes exist only under
`openaugi serve --transport streamable-http` (stdio has no HTTP at all).

Read-only by design: every route is a query; the command side stays MCP
tools + file contracts. Auth posture is identical to /mcp — when
--auth cloudflare is configured, the same Bearer middleware guards /api/*
(see auth/cloudflare.py), and the localhost default binding applies
otherwise.

Presentation rules here (NOT engine semantics):
- full content, never truncated (contract-tested)
- k / limit are capped at HTTP_MAX_K (500) — UIs page, agents shouldn't
- errors are JSON {"error": ...} with proper status codes
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from openaugi.model.block import Block
from openaugi.query import QuerySpec, engine

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from starlette.requests import Request
    from starlette.responses import Response

    from openaugi.store.sqlite import SQLiteStore

# UIs can page but shouldn't need five round trips for a browse view;
# agents keep the MCP default (k=100) — this ceiling is HTTP-only.
HTTP_MAX_K = 500


def register_api_routes(
    mcp_server: FastMCP,
    get_store: Callable[[], SQLiteStore],
    get_embedding_model: Callable[[], Any],
    release_store: Callable[[], None],
) -> None:
    """Mount /api/* on the FastMCP daemon.

    Dependencies are injected (store/model getters live in mcp/server.py —
    the daemon owns process state; this module owns HTTP shape only).
    """
    from starlette.responses import JSONResponse

    def _json_response(data: dict, status_code: int = 200) -> Response:
        return JSONResponse(data, status_code=status_code)

    def _error(message: str, status_code: int) -> Response:
        return _json_response({"error": message}, status_code=status_code)

    def _run_released(fn: Callable[[], Response]) -> Response:
        """Execute a handler body, releasing the SQLite handle after
        (same discipline as the MCP tools' _release_conn)."""
        try:
            return fn()
        finally:
            release_store()

    # ── /api/search + /api/query ───────────────────────────────────

    def _execute_spec(spec: QuerySpec) -> Response:
        spec.k = min(spec.k, HTTP_MAX_K)
        if spec.is_empty():
            return _error(
                "Empty query: provide query, keyword, title, or at least one filter.", 400
            )
        store = get_store()
        model = get_embedding_model() if spec.mode == "semantic" else None
        result = engine.run(store, spec, embedding_model=model)

        routes = store.get_routed_container_titles([b.id for b in result.blocks])
        results = []
        for b in result.blocks:
            item = _block_json(b, routes.get(b.id, []))
            if result.mode == "semantic":
                item["score"] = result.scores[b.id]
            results.append(item)

        payload: dict[str, Any] = {
            "results": results,
            "count": len(results),
            "has_more": result.has_more,
            "mode": result.mode,
        }
        if result.mode == "browse":
            payload["total"] = result.total
            payload["next_offset"] = result.next_offset
            payload["reference_documents"] = result.reference_documents
            payload["reference_block_count"] = result.reference_block_count
        return _json_response(payload)

    @mcp_server.custom_route("/api/search", methods=["GET"])
    async def api_search(request: Request) -> Response:
        try:
            spec = _spec_from_params(request.query_params)
        except (ValidationError, ValueError) as e:
            return _error(f"Bad query parameters: {e}", 400)
        return _run_released(lambda: _execute_spec(spec))

    @mcp_server.custom_route("/api/query", methods=["POST"])
    async def api_query(request: Request) -> Response:
        try:
            body = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            return _error("Body must be JSON.", 400)
        if not isinstance(body, dict):
            return _error("Body must be a JSON object (QuerySpec or {'saved': name}).", 400)
        if "saved" in body:
            # Saved queries land in query-layer step 5.
            return _error("Saved queries are not available yet.", 501)
        try:
            spec = QuerySpec.model_validate(body)
        except ValidationError as e:
            return _error(f"Invalid QuerySpec: {e}", 400)
        return _run_released(lambda: _execute_spec(spec))

    # ── /api/blocks ────────────────────────────────────────────────

    @mcp_server.custom_route("/api/blocks", methods=["GET"])
    async def api_blocks(request: Request) -> Response:
        ids_param = request.query_params.get("ids", "")
        block_ids = [i.strip() for i in ids_param.split(",") if i.strip()]
        if not block_ids:
            return _error("Provide ?ids=a,b,c", 400)
        if len(block_ids) > HTTP_MAX_K:
            return _error(f"Too many ids ({len(block_ids)}); maximum {HTTP_MAX_K}.", 400)

        def body() -> Response:
            result = engine.fetch(get_store(), block_ids)
            return _json_response(
                {
                    "blocks": [_block_json(b, result.routes.get(b.id, [])) for b in result.blocks],
                    "count": len(result.blocks),
                    "missing": result.missing,
                }
            )

        return _run_released(body)

    # ── /api/related/{id} ──────────────────────────────────────────

    @mcp_server.custom_route("/api/related/{block_id}", methods=["GET"])
    async def api_related(request: Request) -> Response:
        block_id = request.path_params["block_id"]
        direction = request.query_params.get("direction", "both")
        kind = request.query_params.get("kind")
        try:
            limit = _int_param(request.query_params, "limit", 50)
        except ValueError as e:
            return _error(str(e), 400)

        def body() -> Response:
            items = engine.related(
                get_store(), block_id, kind=kind, direction=direction, limit=limit
            )
            return _json_response(
                {
                    "block_id": block_id,
                    "related": [
                        {
                            "block": _block_json(i.block),
                            "link_kind": i.link_kind,
                            "direction": i.direction,
                        }
                        for i in items
                    ],
                    "count": len(items),
                }
            )

        return _run_released(body)

    # ── /api/views ─────────────────────────────────────────────────

    @mcp_server.custom_route("/api/views", methods=["GET"])
    async def api_views(request: Request) -> Response:
        def body() -> Response:
            views = engine.views(get_store())
            return _json_response({"views": views, "count": len(views)})

        return _run_released(body)

    @mcp_server.custom_route("/api/views/{title}", methods=["GET"])
    async def api_view(request: Request) -> Response:
        title = request.path_params["title"]
        try:
            member_limit = _int_param(request.query_params, "member_limit", 50)
        except ValueError as e:
            return _error(str(e), 400)

        def body() -> Response:
            try:
                result = engine.view(get_store(), title, member_limit=member_limit)
            except engine.ContainerNotFound:
                return _error(f"Container note not found: {title}", 404)
            return _json_response(
                {
                    "container": title,
                    "container_id": result.container_id,
                    "recap": result.recap,
                    "members": [
                        {**_block_json(m.block), "membership": m.membership}
                        for m in result.members
                    ],
                    "member_count": result.member_count,
                }
            )

        return _run_released(body)


# ── Serialization (HTTP presentation) ───────────────────────────────


def _block_json(block: Block, routed_to: list[str] | None = None) -> dict:
    """Full block for UIs — same fields as the MCP full shape, no truncation."""
    out = {
        "id": block.id,
        "kind": block.kind,
        "title": block.title,
        "content": block.content,
        "summary": block.summary,
        "tags": block.tags,
        "augi_tags": block.metadata.get("augi_tags", []),
        "block_time": block.block_time,
        "occurred_at": block.occurred_at,
        "source": block.source,
        "source_path": block.metadata.get("source_path", ""),
        "metadata": block.metadata,
        "content_hash": block.content_hash,
        "ingested_at": block.ingested_at,
    }
    if routed_to is not None:
        out["routed_to"] = routed_to
    return out


def _spec_from_params(params) -> QuerySpec:
    """QuerySpec from query-string params. Repeatable or comma-separated tags."""
    tags: list[str] = []
    for raw in params.getlist("tags"):
        tags.extend(t.strip() for t in raw.split(",") if t.strip())

    has_task: bool | None = None
    if "has_task" in params:
        has_task = params["has_task"].lower() in ("1", "true", "yes")

    return QuerySpec(
        query=params.get("query"),
        keyword=params.get("keyword"),
        title=params.get("title"),
        tags=tags or None,
        after=params.get("after"),
        before=params.get("before"),
        after_ingested=params.get("after_ingested"),
        kind=params.get("kind"),
        source=params.get("source"),
        exclude_path_prefix=params.get("exclude_path_prefix"),
        has_task=has_task,
        k=_int_param(params, "k", 100),
        offset=_int_param(params, "offset", 0),
    )


def _int_param(params, name: str, default: int) -> int:
    raw = params.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return min(value, HTTP_MAX_K)
