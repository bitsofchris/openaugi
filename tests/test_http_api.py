"""HTTP adapter contract tests — /api/* on the MCP daemon.

Pins the adapter contract from docs/plans/query-layer.md step 4:
- route ↔ engine parity (same spec → same ids)
- /api/* NEVER truncates content (the 500-char cut is MCP presentation)
- read-only surface (no mutating routes exist)
- auth parity: when the Cloudflare middleware is installed, /api/* is
  guarded exactly like /mcp
- k/limit ceiling (HTTP_MAX_K) and JSON error shapes
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from openaugi.model.block import Block
from openaugi.query import QuerySpec, engine
from openaugi.store.sqlite import SQLiteStore
from tests.query_golden_corpus import FakeEmbedder, build_store

LONG_CONTENT = "long-form zeppelin content " * 60  # ~1600 chars, well past 500


def _build_http_db(path: Path) -> Path:
    db = path / "http.db"
    build_store(db)
    store = SQLiteStore(db)
    store.insert_blocks(
        [
            Block(
                id="b-long-content",
                kind="data_block",
                content=LONG_CONTENT,
                source="vault",
                title="Long Note",
                block_time="2026-06-11",
                metadata={"source_path": "Daily/2026-06-11.md"},
                ingested_at="2026-06-11T10:00:00.000Z",
            )
        ]
    )
    store.close()
    return db


@pytest.fixture(scope="module")
def http_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_http_db(tmp_path_factory.mktemp("http"))


@pytest.fixture(scope="module")
def client(http_db: Path):
    """TestClient over the real daemon app (server module's FastMCP)."""
    import openaugi.mcp.server as srv

    old_db = os.environ.get("OPENAUGI_DB")
    os.environ["OPENAUGI_DB"] = str(http_db)
    srv._store = None
    srv._embedding_model = FakeEmbedder()

    app = srv.mcp.streamable_http_app()
    with TestClient(app) as c:
        yield c

    srv._store = None
    srv._embedding_model = None
    if old_db is None:
        os.environ.pop("OPENAUGI_DB", None)
    else:
        os.environ["OPENAUGI_DB"] = old_db


class TestSearchRoute:
    def test_keyword_search_full_blocks(self, client: TestClient):
        r = client.get("/api/search", params={"keyword": "quantum"})
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "keyword"
        assert data["count"] > 0
        assert all("routed_to" in item for item in data["results"])

    def test_never_truncates_content(self, client: TestClient):
        r = client.get("/api/search", params={"keyword": "zeppelin"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 1
        assert results[0]["content"] == LONG_CONTENT  # full, not [:500]

    def test_semantic_carries_scores(self, client: TestClient):
        r = client.get("/api/search", params={"query": "quantum garden", "k": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "semantic"
        assert all("score" in item for item in data["results"])

    def test_browse_envelope(self, client: TestClient):
        r = client.get("/api/search", params={"after": "2026-01-01", "k": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "browse"
        assert data["total"] > 3
        assert data["has_more"] is True
        assert data["next_offset"] == 3
        assert "reference_documents" in data

    def test_empty_query_is_400(self, client: TestClient):
        r = client.get("/api/search")
        assert r.status_code == 400
        assert "error" in r.json()

    def test_bad_int_param_is_400(self, client: TestClient):
        r = client.get("/api/search", params={"keyword": "quantum", "k": "lots"})
        assert r.status_code == 400

    def test_k_is_capped_not_rejected(self, client: TestClient):
        r = client.get("/api/search", params={"after": "2026-01-01", "k": 99999})
        assert r.status_code == 200


class TestQueryRoute:
    def test_post_spec_matches_get_search(self, client: TestClient, http_db: Path):
        spec = {"has_task": True, "after": "2026-01-01"}
        via_post = client.post("/api/query", json=spec)
        via_get = client.get("/api/search", params={"has_task": "true", "after": "2026-01-01"})
        assert via_post.status_code == via_get.status_code == 200
        post_ids = [b["id"] for b in via_post.json()["results"]]
        get_ids = [b["id"] for b in via_get.json()["results"]]
        assert post_ids == get_ids

        # …and both match the engine directly (route ↔ engine parity).
        store = SQLiteStore(http_db, read_only=True)
        try:
            engine_ids = [b.id for b in engine.run(store, QuerySpec(**spec)).blocks]
        finally:
            store.close()
        assert post_ids == engine_ids

    def test_saved_placeholder_501(self, client: TestClient):
        r = client.post("/api/query", json={"saved": "dashboard-task-shelf"})
        assert r.status_code == 501

    def test_invalid_spec_400(self, client: TestClient):
        r = client.post("/api/query", json={"k": "many"})
        assert r.status_code == 400

    def test_non_json_body_400(self, client: TestClient):
        r = client.post("/api/query", content=b"not json")
        assert r.status_code == 400


class TestBlocksAndRelated:
    def test_blocks_batch_with_missing(self, client: TestClient):
        r = client.get("/api/blocks", params={"ids": "b1-quantum-idea,nope-000,b2-open-task"})
        assert r.status_code == 200
        data = r.json()
        assert [b["id"] for b in data["blocks"]] == ["b1-quantum-idea", "b2-open-task"]
        assert data["missing"] == ["nope-000"]
        assert data["blocks"][0]["routed_to"] == ["MOC - Alpha Project"]

    def test_blocks_requires_ids(self, client: TestClient):
        assert client.get("/api/blocks").status_code == 400

    def test_related(self, client: TestClient):
        r = client.get("/api/related/b1-quantum-idea", params={"direction": "out"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] > 0
        assert all(i["direction"] == "out" for i in data["related"])


class TestViews:
    def test_views_list(self, client: TestClient):
        r = client.get("/api/views")
        assert r.status_code == 200
        containers = {v["container"] for v in r.json()["views"]}
        assert "MOC - Alpha Project" in containers

    def test_view_detail(self, client: TestClient):
        r = client.get("/api/views/MOC - Alpha Project")
        assert r.status_code == 200
        data = r.json()
        assert data["recap"]["stale"] is False
        assert data["member_count"] == 3

    def test_view_missing_404(self, client: TestClient):
        assert client.get("/api/views/MOC - Nope").status_code == 404


class TestReadOnlySurface:
    def test_no_mutating_api_routes_exist(self, client: TestClient, http_db: Path):
        """The /api surface is read-only: GET everywhere, POST only on
        /api/query (which executes a QuerySpec — a read)."""
        import openaugi.mcp.server as srv

        for route in srv.mcp.streamable_http_app().routes:
            path = getattr(route, "path", "")
            if not path.startswith("/api"):
                continue
            methods = set(getattr(route, "methods", set()) or set()) - {"HEAD", "OPTIONS"}
            if path == "/api/query":
                assert methods == {"POST"}
            else:
                assert methods == {"GET"}, f"{path} allows {methods}"

    def test_post_to_get_route_is_405(self, client: TestClient):
        assert client.post("/api/views").status_code == 405


class TestAuthParity:
    def test_cloudflare_middleware_guards_api(self, http_db: Path):
        """With auth installed, /api/* rejects missing/bad tokens and admits
        good ones — same posture as /mcp, pinned so a tunnel can't expose
        reads unauthenticated."""
        from mcp.server.fastmcp import FastMCP

        from openaugi.auth.cloudflare import _register_auth_middleware
        from openaugi.http_api import register_api_routes

        store_holder: dict = {}

        def get_store():
            if "s" not in store_holder:
                store_holder["s"] = SQLiteStore(http_db, read_only=True)
            return store_holder["s"]

        def release_store():
            # Runs in the handler's thread — sqlite objects are
            # thread-bound, so this is where the close must happen.
            s = store_holder.pop("s", None)
            if s is not None:
                s.close()

        fresh = FastMCP("authed-test")
        register_api_routes(
            fresh,
            get_store=get_store,
            get_embedding_model=FakeEmbedder,
            release_store=release_store,
        )

        class FakeVerifier:
            def verify(self, token: str) -> dict | None:
                return {"sub": "chris"} if token == "good-token" else None

        _register_auth_middleware(fresh, FakeVerifier())  # type: ignore[arg-type]

        with TestClient(fresh.streamable_http_app()) as c:
            assert c.get("/api/views").status_code == 401
            assert c.get("/api/views", headers={"Authorization": "Bearer nope"}).status_code == 401
            assert (
                c.get("/api/views", headers={"Authorization": "Bearer good-token"}).status_code
                == 200
            )
