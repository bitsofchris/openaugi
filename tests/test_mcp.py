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


def _config_with(salience: dict) -> dict:
    """Default config with a [salience] override, as load_config would merge it."""
    from openaugi.config import DEFAULT_CONFIG, _merge

    return _merge(DEFAULT_CONFIG, {"salience": salience})


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

    def test_summaries_include_source_path(self):
        from openaugi.mcp.server import search

        result = json.loads(search(keyword="career"))
        assert result["count"] > 0
        for block in result["results"]:
            assert "source_path" in block
        # Fixture-vault ingestion sets real paths — at least one non-empty
        assert any(b["source_path"] for b in result["results"])

    def test_search_exclude_path_prefix_browse(self):
        from openaugi.mcp.server import search

        everything = json.loads(search(after="2000-01-01"))
        assert everything["count"] > 0
        target = next(b["source_path"] for b in everything["results"] if b["source_path"])
        filtered = json.loads(search(after="2000-01-01", exclude_path_prefix=target))
        assert all(not b["source_path"].startswith(target) for b in filtered["results"])
        assert filtered["count"] < everything["count"]

    def test_search_exclude_path_prefix_keyword(self):
        from openaugi.mcp.server import search

        unfiltered = json.loads(search(keyword="career"))
        target = next((b["source_path"] for b in unfiltered["results"] if b["source_path"]), None)
        assert target is not None
        filtered = json.loads(search(keyword="career", exclude_path_prefix=target))
        assert all(b["source_path"] != target for b in filtered["results"])

    def test_browse_groups_reference_documents(self, populated_db: Path):
        from openaugi.mcp.server import search
        from openaugi.model.block import Block

        store = SQLiteStore(populated_db)
        store.insert_blocks(
            [
                Block(
                    id="ref1",
                    kind="data_block",
                    content="podcast chunk one",
                    block_time="2031-01-01",
                    tags=["source/snipd"],
                    metadata={"source_path": "Reference/Snipd/Episode.md"},
                ),
                Block(
                    id="ref2",
                    kind="data_block",
                    content="podcast chunk two",
                    block_time="2031-01-02",
                    tags=["source/snipd"],
                    metadata={"source_path": "Reference/Snipd/Episode.md"},
                ),
                Block(
                    id="cap1",
                    kind="data_block",
                    content="my own thought",
                    block_time="2031-01-03",
                    metadata={"source_path": "Journal/2031-01-03.md"},
                ),
            ]
        )
        store.close()

        result = json.loads(search(after="2030-12-31"))
        # The user's capture stays a normal result; reference blocks collapse
        assert [b["id"] for b in result["results"]] == ["cap1"]
        assert result["reference_block_count"] == 2
        assert len(result["reference_documents"]) == 1
        doc = result["reference_documents"][0]
        assert doc["source_path"] == "Reference/Snipd/Episode.md"
        assert doc["title"] == "Episode"
        assert doc["block_count"] == 2
        assert doc["source_tags"] == ["source/snipd"]
        assert doc["first_block_time"] == "2031-01-01"
        assert doc["last_block_time"] == "2031-01-02"
        assert doc["document_id"] == Block.make_document_id("Reference/Snipd/Episode.md")

    def test_apply_routing_batch(self, populated_db: Path):
        from openaugi.mcp.server import apply_routing, search
        from openaugi.model.block import Block

        store = SQLiteStore(populated_db)
        store.insert_blocks(
            [
                Block(
                    id="doc-container",
                    kind="context_block:document",
                    content="",
                    title="AMOC - Test Area",
                ),
            ]
        )
        store.close()

        found = json.loads(search(keyword="career"))
        b1, b2 = found["results"][0]["id"], found["results"][1]["id"]

        result = json.loads(
            apply_routing(
                [
                    {"block_id": b1, "containers": ["AMOC - Test Area"]},
                    {
                        "block_id": b2,
                        "containers": ["AMOC - Test Area"],
                        "augi_tags": ["area/work"],
                    },
                    {"block_id": "nonexistent", "containers": ["AMOC - Test Area"]},
                    {"block_id": b1, "containers": ["No Such Container"]},
                ]
            )
        )
        assert result["status"] == "partial"
        assert result["routes_applied"] == 2
        assert result["blocks_tagged"] == 1
        assert len(result["errors"]) == 2

        store = SQLiteStore(populated_db)
        rows = store.conn.execute(
            "SELECT from_id FROM links WHERE to_id = 'doc-container' AND kind = 'routed_to'"
        ).fetchall()
        tagged = store.get_block(b2)
        store.close()
        assert {r[0] for r in rows} == {b1, b2}
        assert tagged.metadata["augi_tags"] == ["area/work"]

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
        # data_block should have at least a contains link
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

    def test_get_context_purpose_gates_low_scores(self, monkeypatch):
        """purpose applies the [salience] min-score: results below it are dropped."""
        import openaugi.mcp.server as srv
        from openaugi.mcp.server import get_context

        # Fixture-vault candidates score 1.0 (FTS prong); a gate above that
        # silences everything, proving the filter path works end to end.
        monkeypatch.setattr(srv, "load_config", lambda: _config_with({"resurface": 2.0}))
        gated = json.loads(get_context("career direction", purpose="resurface"))
        assert gated["direct_results"] == []
        assert gated["expanded"] == []
        assert gated["salience"] == {"purpose": "resurface", "min_score": 2.0}

        # A gate below the scores passes everything through — identical to ungated.
        # expand=False so direct_results holds only scored direct hits.
        monkeypatch.setattr(srv, "load_config", lambda: _config_with({"resurface": 0.5}))
        passed = json.loads(get_context("career direction", expand=False, purpose="resurface"))
        ungated = json.loads(get_context("career direction", expand=False))
        assert passed["total_blocks"] == ungated["total_blocks"]
        assert all(r["score"] >= 0.5 for r in passed["direct_results"])
        assert "salience" not in ungated

    def test_get_context_unknown_purpose_no_gate(self, monkeypatch):
        """A purpose with no [salience] key applies no gate (and reports none)."""
        import openaugi.mcp.server as srv
        from openaugi.mcp.server import get_context

        monkeypatch.setattr(srv, "load_config", lambda: _config_with({"resurface": 2.0}))
        result = json.loads(get_context("career direction", purpose="mystery"))
        ungated = json.loads(get_context("career direction"))
        assert result["total_blocks"] == ungated["total_blocks"]
        assert result["total_blocks"] > 0
        assert result["salience"] == {"purpose": "mystery", "min_score": None}

    def test_get_context_salience_config_override(self, monkeypatch):
        """config.toml [salience] overrides the default threshold per purpose."""
        import openaugi.mcp.server as srv
        from openaugi.config import DEFAULT_CONFIG
        from openaugi.mcp.server import get_context

        # Defaults ship the calibrated gates (2026-07-07): resurface permissive,
        # push reserved stricter.
        assert DEFAULT_CONFIG["salience"]["resurface"] == 0.06
        assert DEFAULT_CONFIG["salience"]["push"] == 0.15

        monkeypatch.setattr(srv, "load_config", lambda: _config_with({"push": 3.0}))
        result = json.loads(get_context("career direction", purpose="push"))
        assert result["salience"]["min_score"] == 3.0
        assert result["direct_results"] == []

    def test_recent(self):
        from openaugi.mcp.server import recent

        result = json.loads(recent(k=5))
        assert "results" in result
        assert result["count"] > 0
        assert result["count"] <= 5

    def test_recent_filtered_by_kind(self):
        from openaugi.mcp.server import recent

        result = json.loads(recent(kind="context_block:document"))
        assert "results" in result
        for r in result["results"]:
            assert r["kind"] == "context_block:document"

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


class TestReviewPassTools:
    def test_get_review_state_initially_empty(self):
        from openaugi.mcp.server import get_review_state

        result = json.loads(get_review_state())
        assert result["last_run"] is None
        assert result["last_summary"] is None

    def test_mark_review_complete_advances_mark(self):
        from openaugi.mcp.server import get_review_state, mark_review_complete

        result = json.loads(mark_review_complete(summary="routed 5 blocks; 2 views"))
        assert result["status"] == "ok"
        assert result["last_run"] is not None

        state = json.loads(get_review_state())
        assert state["last_run"] == result["last_run"]
        assert state["last_summary"] == "routed 5 blocks; 2 views"


class TestWriteDocumentOverwrite:
    def test_overwrite_replaces_view(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_document

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        write_document("openaugi", "view of OpenAugi area", "old head", "Views")
        result = json.loads(
            write_document(
                "openaugi", "view of OpenAugi area", "new head", "Views", overwrite=True
            )
        )
        assert result["status"] == "updated"
        path = tmp_path / "OpenAugi" / "Views" / "openaugi.md"
        text = path.read_text()
        assert "new head" in text
        assert "old head" not in text

    def test_no_overwrite_still_errors(self, tmp_path, monkeypatch):
        from openaugi.mcp.server import write_document

        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(tmp_path))
        write_document("openaugi", "view", "first", "Views")
        result = json.loads(write_document("openaugi", "view", "second", "Views"))
        assert result["status"] == "error"
        assert "already exists" in result["reason"]


class TestRoutingCrud:
    """apply_routing is the single route write tool: add, remove, move."""

    def _block_and_container(self):
        from openaugi.mcp.server import get_related, search

        blocks = json.loads(search(keyword="career"))["results"]
        block_id = next(b["id"] for b in blocks if b["kind"] == "data_block")
        related = json.loads(get_related(block_id, direction="out", kind="contains"))
        home_title = related["related"][0]["block"]["title"]
        docs = json.loads(search(kind="context_block:document", k=10))["results"]
        foreign = [d["title"] for d in docs if d["title"] != home_title]
        return block_id, foreign[0], foreign[1]

    def test_add_creates_link_and_is_readable(self):
        from openaugi.mcp.server import apply_routing, get_block, get_related

        block_id, title, _ = self._block_and_container()
        result = json.loads(apply_routing([{"block_id": block_id, "add": [title]}]))
        assert result["status"] == "ok"
        assert result["routes_applied"] == 1

        related = json.loads(get_related(block_id, direction="out", kind="routed_to"))
        assert any(r["block"]["title"] == title for r in related["related"])
        block = json.loads(get_block(block_id))
        assert title in block["routed_to"]

    def test_containers_alias_still_works(self):
        from openaugi.mcp.server import apply_routing

        block_id, title, _ = self._block_and_container()
        result = json.loads(apply_routing([{"block_id": block_id, "containers": [title]}]))
        assert result["status"] == "ok"
        assert result["routes_applied"] == 1

    def test_remove_deletes_link(self):
        from openaugi.mcp.server import apply_routing, get_block

        block_id, title, _ = self._block_and_container()
        apply_routing([{"block_id": block_id, "add": [title]}])
        result = json.loads(apply_routing([{"block_id": block_id, "remove": [title]}]))
        assert result["status"] == "ok"
        assert result["routes_removed"] == 1
        assert json.loads(get_block(block_id))["routed_to"] == []

    def test_move_between_containers_in_one_decision(self):
        from openaugi.mcp.server import apply_routing, get_block

        block_id, wrong, right = self._block_and_container()
        apply_routing([{"block_id": block_id, "add": [wrong]}])
        result = json.loads(
            apply_routing([{"block_id": block_id, "add": [right], "remove": [wrong]}])
        )
        assert result["status"] == "ok"
        assert result["routes_applied"] == 1
        assert result["routes_removed"] == 1
        assert json.loads(get_block(block_id))["routed_to"] == [right]

    def test_remove_absent_route_is_noop_not_error(self):
        from openaugi.mcp.server import apply_routing

        block_id, title, _ = self._block_and_container()
        result = json.loads(apply_routing([{"block_id": block_id, "remove": [title]}]))
        assert result["status"] == "ok"
        assert result["routes_removed"] == 0
        assert result["routes_not_found"] == 1

    def test_unknown_container_fails_that_decision_only(self):
        from openaugi.mcp.server import apply_routing

        block_id, title, _ = self._block_and_container()
        result = json.loads(
            apply_routing(
                [
                    {"block_id": block_id, "add": [title]},
                    {"block_id": block_id, "remove": ["No Such Note Title"]},
                ]
            )
        )
        assert result["status"] == "partial"
        assert result["routes_applied"] == 1
        assert len(result["errors"]) == 1

    def test_add_idempotent(self):
        from openaugi.mcp.server import apply_routing, get_block

        block_id, title, _ = self._block_and_container()
        apply_routing([{"block_id": block_id, "add": [title]}])
        result = json.loads(apply_routing([{"block_id": block_id, "add": [title]}]))
        assert result["status"] == "ok"
        assert json.loads(get_block(block_id))["routed_to"] == [title]

    def _block_and_home(self):
        """A data block plus the title of the document it physically lives in."""
        from openaugi.mcp.server import get_related, search

        blocks = json.loads(search(keyword="career"))["results"]
        block_id = next(b["id"] for b in blocks if b["kind"] == "data_block")
        related = json.loads(get_related(block_id, direction="out", kind="contains"))
        home_title = related["related"][0]["block"]["title"]
        return block_id, home_title

    def test_add_to_own_source_note_is_already_home(self):
        from openaugi.mcp.server import apply_routing, get_block

        block_id, home_title = self._block_and_home()
        result = json.loads(apply_routing([{"block_id": block_id, "add": [home_title]}]))
        assert result["status"] == "ok"
        assert result["already_home"] == 1
        assert result["routes_applied"] == 0
        # no redundant edge was written
        assert json.loads(get_block(block_id))["routed_to"] == []

    def test_remove_containment_is_an_error(self):
        from openaugi.mcp.server import apply_routing

        block_id, home_title = self._block_and_home()
        result = json.loads(apply_routing([{"block_id": block_id, "remove": [home_title]}]))
        assert result["status"] == "partial"
        assert result["routes_removed"] == 0
        assert "physically lives" in result["errors"][0]["reason"]


class TestGetMembers:
    def test_unified_membership(self):
        from openaugi.mcp.server import apply_routing, get_members, get_related, search

        # the container is a real source document; its own blocks are 'contained'
        blocks = json.loads(search(keyword="career"))["results"]
        block_id = next(b["id"] for b in blocks if b["kind"] == "data_block")
        related = json.loads(get_related(block_id, direction="out", kind="contains"))
        home_title = related["related"][0]["block"]["title"]

        # route a foreign block in from another document
        others = json.loads(search(kind="data_block", k=50))["results"]
        foreign = next(
            b["id"]
            for b in others
            if b["id"] != block_id and b["source_path"] != blocks[0].get("source_path", "")
        )
        apply_routing([{"block_id": foreign, "add": [home_title]}])

        result = json.loads(get_members(home_title))
        by_id = {m["id"]: m["membership"] for m in result["members"]}
        assert by_id[block_id] == "contained"
        assert by_id[foreign] in ("routed", "both")
        assert result["total"] == len(result["members"])

    def test_unknown_container(self):
        from openaugi.mcp.server import get_members

        result = json.loads(get_members("No Such Container"))
        assert result["status"] == "error"


class TestRenderedViews:
    """§2 of views-as-rendered-queries: get_view + write_recap recap cache."""

    def _container(self):
        from openaugi.mcp.server import get_related, search

        blocks = json.loads(search(keyword="career"))["results"]
        block_id = next(b["id"] for b in blocks if b["kind"] == "data_block")
        related = json.loads(get_related(block_id, direction="out", kind="contains"))
        return related["related"][0]["block"]["title"]

    def test_view_without_recap(self):
        from openaugi.mcp.server import get_view

        title = self._container()
        result = json.loads(get_view(title))
        assert result["container"] == title
        assert result["recap"] is None
        assert result["member_count"] > 0
        assert all(m["membership"] in ("contained", "routed", "both") for m in result["members"])

    def test_write_recap_then_view_is_fresh(self):
        from openaugi.mcp.server import get_view, write_recap

        title = self._container()
        result = json.loads(write_recap(title, "## TLDR\nAll quiet."))
        assert result["status"] == "ok"

        view = json.loads(get_view(title))
        assert view["recap"]["recap_md"] == "## TLDR\nAll quiet."
        assert view["recap"]["stale"] is False

    def test_membership_change_makes_recap_stale(self):
        from openaugi.mcp.server import apply_routing, get_view, search, write_recap

        title = self._container()
        write_recap(title, "recap before the change")
        assert json.loads(get_view(title))["recap"]["stale"] is False

        # route in a foreign block → membership hash changes → recap is stale
        docs_members = {m["id"] for m in json.loads(get_view(title))["members"]}
        others = json.loads(search(kind="data_block", k=50))["results"]
        foreign = next(b["id"] for b in others if b["id"] not in docs_members)
        apply_routing([{"block_id": foreign, "add": [title]}])

        assert json.loads(get_view(title))["recap"]["stale"] is True

    def test_recap_overwrites(self):
        from openaugi.mcp.server import get_view, write_recap

        title = self._container()
        write_recap(title, "first")
        write_recap(title, "second")
        assert json.loads(get_view(title))["recap"]["recap_md"] == "second"

    def test_unknown_container(self):
        from openaugi.mcp.server import get_view, write_recap

        assert json.loads(get_view("No Such Container"))["status"] == "error"
        assert json.loads(write_recap("No Such Container", "x"))["status"] == "error"


class TestWriteContextPack:
    def test_requires_vault_path(self, monkeypatch: pytest.MonkeyPatch):
        from openaugi.mcp.server import write_context_pack

        monkeypatch.delenv("OPENAUGI_VAULT_PATH", raising=False)
        monkeypatch.setattr("openaugi.mcp.server.load_config", lambda: {})
        result = json.loads(write_context_pack())
        assert result["status"] == "error"

    def test_writes_pack_to_vault(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from openaugi.mcp.server import write_context_pack

        vault = tmp_path / "vault-out"
        vault.mkdir()
        monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(vault))
        result = json.loads(write_context_pack())
        assert result["status"] == "ok"
        pack = json.loads((vault / "OpenAugi" / "context-pack.json").read_text())
        assert set(pack) >= {"agentFile", "taxonomy", "recentConcepts", "noteTitles"}
        assert pack["noteTitles"]  # fixture vault has documents
