"""Saved queries — markdown+frontmatter QuerySpecs in the vault.

Covers query/saved.py (parsing, listing, token resolution with a frozen
clock), the shipped seed templates, the MCP tools (list_queries /
run_query), and the HTTP routes (/api/queries*, POST /api/query saved).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from openaugi.query import QuerySpec, saved
from openaugi.store.sqlite import SQLiteStore
from tests.query_golden_corpus import build_store

TODAY = date(2026, 7, 16)


def _write_query(vault: Path, name: str, body: str) -> Path:
    folder = saved.queries_dir(vault)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.md"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    return v


class TestParsing:
    def test_parse_frontmatter_query(self, vault: Path):
        _write_query(
            vault,
            "task-shelf",
            "---\ndescription: Open tasks.\nquery:\n  has_task: true\n"
            '  after: "-14d"\n---\n\nProse body ignored.\n',
        )
        sq = saved.load_saved(vault, "task-shelf")
        assert sq.name == "task-shelf"
        assert sq.description == "Open tasks."
        assert sq.spec == QuerySpec(has_task=True, after="-14d")

    def test_yaml_date_coerced_to_string(self, vault: Path):
        _write_query(vault, "dated", "---\nquery:\n  after: 2026-06-01\n---\n")
        sq = saved.load_saved(vault, "dated")
        assert sq.spec.after == "2026-06-01"

    def test_missing_query_mapping_is_error(self, vault: Path):
        _write_query(vault, "broken", "---\ndescription: no spec here\n---\n")
        with pytest.raises(saved.SavedQueryError):
            saved.load_saved(vault, "broken")

    def test_unknown_name_raises_not_found(self, vault: Path):
        with pytest.raises(saved.SavedQueryNotFound):
            saved.load_saved(vault, "nope")

    def test_list_skips_unparseable_lenient(self, vault: Path):
        _write_query(vault, "good", "---\nquery:\n  keyword: quantum\n---\n")
        _write_query(vault, "bad", "no frontmatter at all\n")
        names = [q.name for q in saved.list_saved(vault)]
        assert names == ["good"]

    def test_list_empty_when_no_dir(self, vault: Path):
        assert saved.list_saved(vault) == []


class TestTokenResolution:
    def test_relative_days_and_today(self):
        spec = QuerySpec(after="-14d", before="today", keyword="x")
        resolved = saved.resolve_spec(spec, today=TODAY)
        assert resolved.after == "2026-07-02"
        assert resolved.before == "2026-07-16"
        # original untouched — tokens never stored resolved
        assert spec.after == "-14d"

    def test_absolute_dates_pass_through(self):
        spec = QuerySpec(after="2026-01-01")
        assert saved.resolve_spec(spec, today=TODAY).after == "2026-01-01"

    def test_review_mark_resolves_from_store(self, tmp_path: Path):
        db = tmp_path / "s.db"
        build_store(db)  # sets review mark to 2026-06-05T00:00:00+00:00
        store = SQLiteStore(db)
        try:
            spec = QuerySpec(after_ingested=saved.REVIEW_MARK_TOKEN)
            resolved = saved.resolve_spec(spec, store=store, today=TODAY)
            assert resolved.after_ingested == "2026-06-05T00:00:00+00:00"
        finally:
            store.close()

    def test_review_mark_without_mark_is_epoch_backfill(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "empty.db")
        try:
            spec = QuerySpec(after_ingested=saved.REVIEW_MARK_TOKEN)
            resolved = saved.resolve_spec(spec, store=store, today=TODAY)
            assert resolved.after_ingested == "1970-01-01T00:00:00Z"
        finally:
            store.close()

    def test_review_mark_only_valid_on_after_ingested(self):
        with pytest.raises(saved.SavedQueryError):
            saved.resolve_spec(QuerySpec(after=saved.REVIEW_MARK_TOKEN), today=TODAY)


class TestSeedTemplates:
    def test_shipped_seeds_parse_and_resolve(self, tmp_path: Path):
        """The templates 'openaugi init' copies must be valid saved queries."""
        import importlib.resources

        templates = importlib.resources.files("openaugi") / "templates" / "queries"
        vault = tmp_path / "vault"
        folder = saved.queries_dir(vault)
        folder.mkdir(parents=True)
        for entry in templates.iterdir():
            (folder / entry.name).write_text(entry.read_text(encoding="utf-8"))

        queries = saved.list_saved(vault)
        names = {q.name for q in queries}
        assert names == {"dashboard-task-shelf", "review-queue", "today"}
        assert all(q.description for q in queries)

        db = tmp_path / "seed.db"
        build_store(db)
        store = SQLiteStore(db)
        try:
            for q in queries:
                resolved = saved.resolve_spec(q.spec, store=store, today=TODAY)
                assert not resolved.is_empty()
        finally:
            store.close()

    def test_task_shelf_seed_matches_dashboard_convention(self, tmp_path: Path):
        import importlib.resources

        text = (
            importlib.resources.files("openaugi")
            / "templates"
            / "queries"
            / "dashboard-task-shelf.md"
        ).read_text(encoding="utf-8")
        vault = tmp_path / "v"
        _write_query(vault, "dashboard-task-shelf", text)
        sq = saved.load_saved(vault, "dashboard-task-shelf")
        assert sq.spec.has_task is True
        assert sq.spec.after == "-14d"


@pytest.fixture
def mcp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Golden DB + vault with seed queries, wired into the server module."""
    import importlib.resources

    import openaugi.mcp.server as srv

    db = tmp_path / "mcp.db"
    build_store(db)
    vault = tmp_path / "vault"
    folder = saved.queries_dir(vault)
    folder.mkdir(parents=True)
    templates = importlib.resources.files("openaugi") / "templates" / "queries"
    for entry in templates.iterdir():
        (folder / entry.name).write_text(entry.read_text(encoding="utf-8"))

    monkeypatch.setenv("OPENAUGI_DB", str(db))
    monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(vault))

    # Freeze "today" inside token resolution so "-14d" lands inside the
    # corpus's June 2026 window regardless of when the test runs.
    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 6, 15)

    monkeypatch.setattr(saved, "date", _FrozenDate)
    srv._store = None
    yield db, vault
    srv._store = None


class TestMCPTools:
    def test_list_queries(self, mcp_env):
        from openaugi.mcp.server import list_queries

        data = json.loads(list_queries())
        assert data["count"] == 3
        names = {q["name"] for q in data["queries"]}
        assert "dashboard-task-shelf" in names
        shelf = next(q for q in data["queries"] if q["name"] == "dashboard-task-shelf")
        assert shelf["spec"] == {"has_task": True, "after": "-14d", "k": 100, "offset": 0}

    def test_run_query_task_shelf(self, mcp_env):
        """The saved shelf returns exactly what the equivalent search does —
        the docstring convention is now a file."""
        from openaugi.mcp.server import run_query, search

        data = json.loads(run_query("dashboard-task-shelf"))
        assert data["query"] == "dashboard-task-shelf"
        assert data["resolved_spec"]["has_task"] is True
        assert data["resolved_spec"]["after"] != "-14d"  # resolved to a date
        ids = [r["id"] for r in data["results"]]

        equivalent = json.loads(search(has_task=True, after=data["resolved_spec"]["after"]))
        assert ids == [r["id"] for r in equivalent["results"]]
        assert ids  # task blocks exist in the corpus

    def test_run_query_review_queue_uses_mark(self, mcp_env):
        from openaugi.mcp.server import run_query

        data = json.loads(run_query("review-queue"))
        assert data["resolved_spec"]["after_ingested"] == "2026-06-05T00:00:00+00:00"
        ids = {r["id"] for r in data["results"]}
        assert "b6-old-ingest" not in ids  # ingested before the mark

    def test_run_query_missing_name(self, mcp_env):
        from openaugi.mcp.server import run_query

        data = json.loads(run_query("does-not-exist"))
        assert "error" in data
        assert "list_queries" in data["hint"]

    def test_run_query_no_vault(self, mcp_env, monkeypatch):
        import openaugi.mcp.server as srv
        from openaugi.mcp.server import run_query

        monkeypatch.delenv("OPENAUGI_VAULT_PATH")
        monkeypatch.setattr(srv, "load_config", lambda: {})
        data = json.loads(run_query("today"))
        assert data["status"] == "error"


class TestHTTPRoutes:
    @pytest.fixture
    def client(self, mcp_env):
        from starlette.testclient import TestClient

        import openaugi.mcp.server as srv

        # The session manager's run() is once-per-instance; build fresh.
        srv.mcp._session_manager = None
        with TestClient(srv.mcp.streamable_http_app()) as c:
            yield c

    def test_api_queries_list(self, client):
        r = client.get("/api/queries")
        assert r.status_code == 200
        assert r.json()["count"] == 3

    def test_api_query_results(self, client):
        r = client.get("/api/queries/dashboard-task-shelf/results")
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "dashboard-task-shelf"
        assert {b["id"] for b in data["results"]} == {"b2-open-task", "b4-tagged-task"}
        # HTTP invariant holds for saved queries too: full content
        assert all("- [ ]" in b["content"] for b in data["results"] if b["id"] == "b2-open-task")

    def test_post_saved_matches_get_results(self, client):
        via_post = client.post("/api/query", json={"saved": "dashboard-task-shelf"})
        via_get = client.get("/api/queries/dashboard-task-shelf/results")
        assert via_post.status_code == 200
        assert [b["id"] for b in via_post.json()["results"]] == [
            b["id"] for b in via_get.json()["results"]
        ]

    def test_saved_not_found_404(self, client):
        assert client.get("/api/queries/nope/results").status_code == 404
        assert client.post("/api/query", json={"saved": "nope"}).status_code == 404
