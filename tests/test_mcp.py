"""Smoke tests for MCP server tools.

Tests the tool functions directly (not via MCP transport) after
ingesting the fixture vault. Verifies the tools return valid JSON
with expected structure.
"""

import json
from pathlib import Path

import pytest

from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore


@pytest.fixture
def populated_db(tmp_path: Path, vault_path: Path) -> Path:
    """Ingest fixture vault into a temp DB, return the DB path."""
    db_path = tmp_path / "mcp_test.db"
    store = SQLiteStore(db_path)
    run_layer0(vault_path, store)
    store.close()
    return db_path


@pytest.fixture(autouse=True)
def _set_db_env(populated_db: Path, monkeypatch: pytest.MonkeyPatch):
    """Point MCP server at the test DB."""
    monkeypatch.setenv("OPENAUGI_DB", str(populated_db))
    # Reset module-level state so it picks up the new DB
    import openaugi.mcp.server as srv

    srv._store = None
    srv._faiss_index = None
    srv._embedding_model = None
    srv._db_mtime = 0


class TestMCPTools:
    def test_search_keyword(self):
        from openaugi.mcp.server import search

        result = json.loads(search(keyword="career"))
        assert result["count"] > 0
        assert result["mode"] == "keyword"
        assert "results" in result
        # Results should have block structure
        first = result["results"][0]
        assert "id" in first
        assert "content" in first
        assert "kind" in first

    def test_search_browse_by_tags(self):
        from openaugi.mcp.server import search

        result = json.loads(search(tags=["project"]))
        assert "results" in result

    def test_search_has_more_flag(self):
        from openaugi.mcp.server import search

        # Request k=1 — if there's more than 1 result, has_more should be True
        result = json.loads(search(keyword="career", k=1))
        assert "has_more" in result
        if result["count"] == 1:
            # We know fixture vault has multiple career entries
            assert result["has_more"] is True

    def test_search_no_args_returns_error_with_hint(self):
        from openaugi.mcp.server import search

        result = json.loads(search())
        assert "error" in result
        assert "hint" in result

    def test_get_block(self):
        from openaugi.mcp.server import get_block, search

        # First find a block via search
        search_result = json.loads(search(keyword="career"))
        block_id = search_result["results"][0]["id"]

        result = json.loads(get_block(block_id))
        assert result["id"] == block_id
        assert "content" in result
        assert "metadata" in result

    def test_get_blocks_batch(self):
        from openaugi.mcp.server import get_blocks, search

        # Find some block IDs via search
        search_result = json.loads(search(keyword="career"))
        ids = [r["id"] for r in search_result["results"][:3]]

        result = json.loads(get_blocks(ids))
        assert result["count"] == len(ids)
        assert len(result["blocks"]) == len(ids)
        assert result["missing"] == []
        # Each block should have full content
        for block in result["blocks"]:
            assert "content" in block
            assert "metadata" in block

    def test_get_blocks_with_missing(self):
        from openaugi.mcp.server import get_blocks, search

        search_result = json.loads(search(keyword="career"))
        valid_id = search_result["results"][0]["id"]

        result = json.loads(get_blocks([valid_id, "nonexistent_id"]))
        assert result["count"] == 1
        assert len(result["blocks"]) == 1
        assert result["missing"] == ["nonexistent_id"]

    def test_get_blocks_too_many(self):
        from openaugi.mcp.server import get_blocks

        result = json.loads(get_blocks([f"id_{i}" for i in range(51)]))
        assert "error" in result

    def test_get_block_not_found(self):
        from openaugi.mcp.server import get_block

        result = json.loads(get_block("nonexistent"))
        assert "error" in result
        assert "hint" in result

    def test_get_related(self):
        from openaugi.mcp.server import get_related, search

        # Find an entry block
        search_result = json.loads(search(keyword="career"))
        block_id = search_result["results"][0]["id"]

        result = json.loads(get_related(block_id))
        assert "related" in result
        assert result["count"] >= 0
        # Entry should have at least a split_from link
        if result["count"] > 0:
            assert "link_kind" in result["related"][0]
            assert "block" in result["related"][0]

    def test_traverse(self):
        from openaugi.mcp.server import search, traverse

        # Find an entry, traverse from it
        search_result = json.loads(search(keyword="career"))
        block_id = search_result["results"][0]["id"]

        result = json.loads(traverse(block_id, max_hops=2))
        assert "results" in result
        assert result["count"] >= 0
        if result["count"] > 0:
            assert "depth" in result["results"][0]

    def test_get_context_keyword_only(self):
        from openaugi.mcp.server import get_context

        result = json.loads(get_context("career direction"))
        assert "direct_results" in result
        assert "expanded" in result
        assert result["total_blocks"] > 0

    def test_recent(self):
        from openaugi.mcp.server import recent

        result = json.loads(recent(k=5))
        assert "results" in result
        assert result["count"] > 0
        assert result["count"] <= 5

    def test_recent_filtered_by_kind(self):
        from openaugi.mcp.server import recent

        result = json.loads(recent(kind="document"))
        assert "results" in result
        for r in result["results"]:
            assert r["kind"] == "document"

    def test_write_document_no_vault(self, monkeypatch):
        import openaugi.mcp.server as srv
        from openaugi.mcp.server import write_document

        monkeypatch.delenv("OPENAUGI_VAULT_PATH", raising=False)
        monkeypatch.setattr(srv, "_get_vault_path", lambda: None)
        result = json.loads(write_document("Test Note", "a test note", "content"))
        assert result["status"] == "error"
        assert "vault path" in result["reason"].lower()

    def test_write_document_creates_file(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_document

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(
            write_document("My Research", "research on X", "# Hello\nSome content.", "Docs")
        )
        assert result["status"] == "created"
        path = tmp_path / "OpenAugi" / "Docs" / "My Research.md"
        assert path.exists()
        assert "description: research on X" in path.read_text()

    def test_write_document_collision(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_document

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        write_document("Duplicate Note", "first note", "first", "Docs")
        result = json.loads(write_document("Duplicate Note", "second note", "second", "Docs"))
        assert result["status"] == "error"
        assert "already exists" in result["reason"]

    def test_write_document_subfolder_escape_blocked(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_document

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(write_document("Escape Note", "desc", "content", "../../../etc"))
        assert result["status"] == "error"

    def test_write_document_custom_subfolder(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_document

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(write_document("Summary", "a summary", "content", "Research"))
        assert result["status"] == "created"
        assert (tmp_path / "OpenAugi" / "Research" / "Summary.md").exists()

    def test_write_thread_creates_file(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_thread

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(
            write_thread(
                "MCP design session",
                "Designing write-back tools for OpenAugi MCP",
                "## Decisions\n- Keep write_thread interface simple",
            )
        )
        assert result["status"] == "created"
        threads_dir = tmp_path / "OpenAugi" / "Threads"
        files = list(threads_dir.glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text()
        assert "type: thread" in text
        assert "description: Designing write-back tools" in text
        assert "# MCP design session" in text

    def test_write_thread_collision(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_thread

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        write_thread("Same Topic", "first", "content one")
        write_thread("Same Topic", "second", "content two")
        threads_dir = tmp_path / "OpenAugi" / "Threads"
        files = list(threads_dir.glob("*.md"))
        assert len(files) == 2
        names = {f.name for f in files}
        assert any("-2.md" in n for n in names)

    def test_write_thread_no_vault(self, monkeypatch):
        import openaugi.mcp.server as srv
        from openaugi.mcp.server import write_thread

        monkeypatch.delenv("OPENAUGI_VAULT_PATH", raising=False)
        monkeypatch.setattr(srv, "_get_vault_path", lambda: None)
        result = json.loads(write_thread("topic", "desc", "content"))
        assert result["status"] == "error"
        assert "vault path" in result["reason"].lower()

    def test_write_snip_creates_file(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_snip

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(
            write_snip(
                "Key Insight",
                "This is an important distilled idea.",
                description="Insight from research session",
                stream="product-management",
                tags=["insight", "product"],
            )
        )
        assert result["status"] == "created"
        path = tmp_path / "OpenAugi" / "Snips" / "Key Insight.md"
        assert path.exists()
        text = path.read_text()
        assert "type: snip" in text
        assert "stream: product-management" in text
        assert "tags: [insight, product]" in text
        assert "description: Insight from research session" in text
        assert "# Key Insight" in text

    def test_write_snip_minimal(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_snip

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(write_snip("Bare Snip", "Just content."))
        assert result["status"] == "created"
        text = (tmp_path / "OpenAugi" / "Snips" / "Bare Snip.md").read_text()
        assert "type: snip" in text
        # Optional fields should not appear when empty
        assert "stream:" not in text
        assert "tags:" not in text
        assert "source_session:" not in text
        assert "description:" not in text

    def test_write_snip_collision(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_snip

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        write_snip("Same Title", "first content")
        result = json.loads(write_snip("Same Title", "second content"))
        assert result["status"] == "created"
        snips_dir = tmp_path / "OpenAugi" / "Snips"
        files = list(snips_dir.glob("*.md"))
        assert len(files) == 2
        names = {f.name for f in files}
        assert "Same Title.md" in names
        assert "Same Title-2.md" in names

    def test_write_snip_no_vault(self, monkeypatch):
        import openaugi.mcp.server as srv
        from openaugi.mcp.server import write_snip

        monkeypatch.delenv("OPENAUGI_VAULT_PATH", raising=False)
        monkeypatch.setattr(srv, "_get_vault_path", lambda: None)
        result = json.loads(write_snip("title", "content"))
        assert result["status"] == "error"
        assert "vault path" in result["reason"].lower()

    def test_write_snip_empty_title(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_snip

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(write_snip("", "content"))
        assert result["status"] == "error"
        assert "empty" in result["reason"].lower()

    def test_write_snip_invalid_title(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_snip

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        result = json.loads(write_snip("bad/title", "content"))
        assert result["status"] == "error"
        assert "invalid" in result["reason"].lower()
