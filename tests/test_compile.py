"""Tests for the compile pipeline — context block materialization.

Tests Layer 0 (hub summaries, recent activity, index) and Layer 1
(concept pages, graph health) against the fixture vault.
"""

from pathlib import Path

import pytest

from openaugi.pipeline.compile import COMPILED_SOURCE, run_compile
from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore


@pytest.fixture
def ingested_store(vault_path: Path, store: SQLiteStore) -> SQLiteStore:
    """Store with fixture vault already ingested."""
    run_layer0(vault_path, store)
    return store


class TestCompileLayer0:
    def test_compile_creates_context_blocks(self, ingested_store: SQLiteStore):
        """Compile should create context blocks in the store."""
        result = run_compile(ingested_store, layer=0)

        assert result["context_blocks"] > 0
        # Should have hub summaries + recent_activity + index
        assert result["by_type"].get("hub_summary", 0) > 0
        assert result["by_type"].get("recent_activity", 0) == 1
        assert result["by_type"].get("index", 0) == 1

    def test_context_blocks_have_correct_kind_and_source(self, ingested_store: SQLiteStore):
        """All context blocks should have kind=context and source=compiled."""
        run_compile(ingested_store, layer=0)

        context_blocks = ingested_store.get_blocks_by_kind("context", limit=200)
        assert len(context_blocks) > 0

        for block in context_blocks:
            assert block.kind == "context"
            assert block.source == COMPILED_SOURCE
            assert "context_type" in block.metadata
            assert "compile_layer" in block.metadata

    def test_hub_summary_content(self, ingested_store: SQLiteStore):
        """Hub summary blocks should contain entry counts and co-occurring tags."""
        run_compile(ingested_store, layer=0)

        context_blocks = ingested_store.get_blocks_by_kind("context", limit=200)
        hub_summaries = [
            b for b in context_blocks if b.metadata.get("context_type") == "hub_summary"
        ]
        assert len(hub_summaries) > 0

        # Each hub summary should have structured content
        for hub in hub_summaries:
            assert hub.content is not None
            assert "Entries:" in hub.content
            assert "Hub score:" in hub.content
            assert hub.metadata.get("entry_count", 0) > 0

    def test_recent_activity_block(self, ingested_store: SQLiteStore):
        """Recent activity block should exist and contain counts."""
        run_compile(ingested_store, layer=0)

        context_blocks = ingested_store.get_blocks_by_kind("context", limit=200)
        recent = [b for b in context_blocks if b.metadata.get("context_type") == "recent_activity"]
        assert len(recent) == 1
        assert recent[0].content is not None
        assert "Last 7 Days" in recent[0].content
        assert "Last 30 Days" in recent[0].content

    def test_index_block_references_all_context_blocks(self, ingested_store: SQLiteStore):
        """Index block should list all other context blocks."""
        run_compile(ingested_store, layer=0)

        context_blocks = ingested_store.get_blocks_by_kind("context", limit=200)
        index = [b for b in context_blocks if b.metadata.get("context_type") == "index"]
        assert len(index) == 1

        idx = index[0]
        assert idx.content is not None
        assert "Top Hubs" in idx.content
        assert "Recent Activity" in idx.content
        # Count should include all blocks
        assert idx.metadata.get("context_block_count", 0) == len(context_blocks)

    def test_recompile_replaces_old_blocks(self, ingested_store: SQLiteStore):
        """Running compile twice should replace, not duplicate, context blocks."""
        result1 = run_compile(ingested_store, layer=0)
        count1 = result1["context_blocks"]

        result2 = run_compile(ingested_store, layer=0)
        count2 = result2["context_blocks"]

        # Same number of blocks each time
        assert count1 == count2

        # Total context blocks in store should equal one run's worth
        context_blocks = ingested_store.get_blocks_by_kind("context", limit=500)
        assert len(context_blocks) == count2

    def test_hub_summary_deterministic_ids(self, ingested_store: SQLiteStore):
        """Context block IDs should be deterministic across runs."""
        run_compile(ingested_store, layer=0)
        blocks1 = ingested_store.get_blocks_by_kind("context", limit=200)
        ids1 = {b.id for b in blocks1}

        run_compile(ingested_store, layer=0)
        blocks2 = ingested_store.get_blocks_by_kind("context", limit=200)
        ids2 = {b.id for b in blocks2}

        assert ids1 == ids2


class TestCompileLayer1:
    def test_concept_pages_created(self, ingested_store: SQLiteStore):
        """Layer 1 should create concept pages for top hubs."""
        result = run_compile(ingested_store, layer=1)

        assert result["by_type"].get("concept", 0) > 0

    def test_concept_page_has_entries(self, ingested_store: SQLiteStore):
        """Concept pages should list entries with timestamps."""
        run_compile(ingested_store, layer=1)

        context_blocks = ingested_store.get_blocks_by_kind("context", limit=200)
        concepts = [b for b in context_blocks if b.metadata.get("context_type") == "concept"]
        assert len(concepts) > 0

        for concept in concepts:
            assert concept.content is not None
            assert "Entries" in concept.content
            assert concept.metadata.get("entry_count", 0) > 0

    def test_concept_pages_have_summarizes_links(self, ingested_store: SQLiteStore):
        """Concept blocks should link to their source entries via summarizes."""
        result = run_compile(ingested_store, layer=1)

        assert result["context_links"] > 0

        context_blocks = ingested_store.get_blocks_by_kind("context", limit=200)
        concepts = [b for b in context_blocks if b.metadata.get("context_type") == "concept"]

        for concept in concepts:
            links = ingested_store.get_links_from(concept.id, kind="summarizes")
            # Each concept should link to at least one entry
            assert len(links) > 0
            # Target should be an existing block
            for link in links:
                target = ingested_store.get_block(link.to_id)
                assert target is not None

    def test_graph_health_block(self, ingested_store: SQLiteStore):
        """Graph health block should report orphans and stale tags."""
        run_compile(ingested_store, layer=1)

        context_blocks = ingested_store.get_blocks_by_kind("context", limit=200)
        health = [b for b in context_blocks if b.metadata.get("context_type") == "graph_health"]
        assert len(health) == 1

        h = health[0]
        assert h.content is not None
        assert "Orphan Entries" in h.content
        assert "Stale Tags" in h.content
        assert "orphan_count" in h.metadata
        assert "stale_tag_count" in h.metadata


class TestVaultRendering:
    def test_render_creates_files(self, ingested_store: SQLiteStore, tmp_path: Path):
        """Compile with vault_path should create .md files."""
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()

        run_compile(ingested_store, vault_path=vault_dir, layer=1)

        compiled_dir = vault_dir / "OpenAugi" / "Compiled"
        assert compiled_dir.exists()
        assert (compiled_dir / "INDEX.md").exists()
        assert (compiled_dir / "RECENT.md").exists()
        assert (compiled_dir / "HEALTH.md").exists()

        # Should have hub and concept subdirectories
        hub_files = (
            list((compiled_dir / "hubs").glob("*.md")) if (compiled_dir / "hubs").exists() else []
        )
        concept_files = (
            list((compiled_dir / "concepts").glob("*.md"))
            if (compiled_dir / "concepts").exists()
            else []
        )
        assert len(hub_files) > 0 or len(concept_files) > 0

    def test_render_index_content(self, ingested_store: SQLiteStore, tmp_path: Path):
        """Rendered INDEX.md should have readable content."""
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()

        run_compile(ingested_store, vault_path=vault_dir, layer=0)

        index_content = (vault_dir / "OpenAugi" / "Compiled" / "INDEX.md").read_text()
        assert "OpenAugi Index" in index_content
        assert "Top Hubs" in index_content

    def test_rerender_cleans_old_files(self, ingested_store: SQLiteStore, tmp_path: Path):
        """Re-rendering should clean old files first."""
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()

        # First compile
        run_compile(ingested_store, vault_path=vault_dir, layer=1)

        # Plant a rogue file
        rogue = vault_dir / "OpenAugi" / "Compiled" / "concepts" / "old-concept.md"
        rogue.parent.mkdir(parents=True, exist_ok=True)
        rogue.write_text("should be cleaned")

        # Second compile
        run_compile(ingested_store, vault_path=vault_dir, layer=1)

        assert not rogue.exists()


class TestStoreCompileHelpers:
    """Test the new store methods used by compile."""

    def test_delete_blocks_by_source(self, ingested_store: SQLiteStore):
        """Should delete all blocks with a given source."""
        from openaugi.model.block import Block

        # Insert some test blocks with source=compiled
        blocks = [
            Block(id=f"test-{i}", kind="context", content=f"test {i}", source="compiled")
            for i in range(3)
        ]
        ingested_store.insert_blocks(blocks)

        deleted = ingested_store.delete_blocks_by_source("compiled")
        assert deleted == 3

        # Should be gone
        for b in blocks:
            assert ingested_store.get_block(b.id) is None

    def test_get_tag_details(self, ingested_store: SQLiteStore):
        """Should return tag details with entry counts."""
        tags = ingested_store.get_tag_details(limit=20)
        assert len(tags) > 0

        for tag in tags:
            assert "tag_id" in tag
            assert "tag_name" in tag
            assert "entry_count" in tag
            assert "hub_score" in tag

    def test_get_co_occurring_tags(self, ingested_store: SQLiteStore):
        """Should find tags that appear on the same entries."""
        tags = ingested_store.get_tag_details(limit=5)
        if tags:
            # Try the top tag
            co = ingested_store.get_co_occurring_tags(tags[0]["tag_id"])
            # May or may not have co-occurring tags depending on fixture
            assert isinstance(co, list)

    def test_get_entries_for_tag(self, ingested_store: SQLiteStore):
        """Should return entries tagged with a given tag."""
        tags = ingested_store.get_tag_details(limit=5)
        if tags:
            entries = ingested_store.get_entries_for_tag(tags[0]["tag_id"])
            assert len(entries) > 0
            assert all(e.kind == "entry" for e in entries)

    def test_get_orphan_block_ids(self, ingested_store: SQLiteStore):
        """Should find entry blocks with no meaningful links."""
        orphans = ingested_store.get_orphan_block_ids(kind="entry")
        # Result depends on fixture; just verify it runs
        assert isinstance(orphans, list)
