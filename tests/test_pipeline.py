"""Integration tests for the pipeline runner.

Tests the full flow: vault → blocks + links → SQLite store → queries.
"""

from pathlib import Path

from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore


class TestPipelineIntegration:
    def test_full_ingest(self, vault_path: Path, store: SQLiteStore):
        """End-to-end: ingest fixture vault → verify store contents."""
        result = run_layer0(vault_path, store)

        assert result["blocks_inserted"] > 0
        assert result["stats"]["total_blocks"] > 0

        # Should have documents, data blocks, and tags
        stats = result["stats"]
        assert stats["blocks_by_kind"].get("context_block:document", 0) >= 5
        assert stats["blocks_by_kind"].get("data_block", 0) >= 5
        assert stats["blocks_by_kind"].get("context_block:tag", 0) >= 3

        # Should have links
        assert stats["total_links"] > 0
        assert stats["links_by_kind"].get("contains", 0) > 0
        assert stats["links_by_kind"].get("groups", 0) > 0

    def test_incremental_second_run_is_noop(self, vault_path: Path, store: SQLiteStore):
        """Second ingest with no changes should not insert new blocks."""
        result1 = run_layer0(vault_path, store)
        blocks_after_first = result1["stats"]["total_blocks"]

        result2 = run_layer0(vault_path, store)
        blocks_after_second = result2["stats"]["total_blocks"]

        # Should be the same — no new blocks
        assert blocks_after_second == blocks_after_first

    def test_fts_search_after_ingest(self, vault_path: Path, store: SQLiteStore):
        """FTS should work after ingestion."""
        run_layer0(vault_path, store)

        # Search for content we know exists
        results = store.search_fts("career")
        assert len(results) > 0

        results = store.search_fts("architecture")
        assert len(results) > 0

    def test_hub_scoring_after_ingest(self, vault_path: Path, store: SQLiteStore):
        """Hub scoring should work after ingestion."""
        run_layer0(vault_path, store)

        scores = store.get_hub_scores(limit=10)
        assert len(scores) > 0
        # At least one hub should have entries
        assert any(s["entry_count"] > 0 for s in scores)

    def test_link_traversal_after_ingest(self, vault_path: Path, store: SQLiteStore):
        """Should be able to traverse links after ingestion."""
        run_layer0(vault_path, store)

        # Get a data_block
        entries = store.get_blocks_by_kind("data_block", limit=1)
        assert len(entries) > 0

        entry = entries[0]

        # Should have a contains link to its document
        links = store.get_links_from(entry.id, kind="contains")
        assert len(links) == 1

        # The target should be a context_block:document block
        doc = store.get_block(links[0].to_id)
        assert doc is not None
        assert doc.kind == "context_block:document"
