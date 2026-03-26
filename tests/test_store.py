"""Tests for SQLiteStore — block CRUD, link CRUD, FTS, cascade delete, vector search."""

import numpy as np

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.store.sqlite import SQLiteStore


class TestBlockCRUD:
    def test_insert_and_get_block(self, store: SQLiteStore):
        b = Block(id="b1", kind="entry", content="Hello", source="vault", title="Test")
        store.insert_block(b)
        store.conn.commit()

        result = store.get_block("b1")
        assert result is not None
        assert result.id == "b1"
        assert result.content == "Hello"
        assert result.kind == "entry"

    def test_insert_block_ignore_duplicate(self, store: SQLiteStore):
        b1 = Block(id="b1", kind="entry", content="First")
        b2 = Block(id="b1", kind="entry", content="Second")
        store.insert_block(b1)
        store.insert_block(b2)
        store.conn.commit()

        result = store.get_block("b1")
        assert result.content == "First"  # second insert ignored

    def test_insert_blocks_batch(self, store: SQLiteStore):
        blocks = [Block(id=f"b{i}", kind="entry", content=f"Content {i}") for i in range(5)]
        count = store.insert_blocks(blocks)
        assert count == 5

        for i in range(5):
            assert store.get_block(f"b{i}") is not None

    def test_delete_block(self, store: SQLiteStore):
        b = Block(id="b1", kind="entry", content="Hello")
        store.insert_block(b)
        store.conn.commit()

        assert store.delete_block("b1")
        assert store.get_block("b1") is None

    def test_delete_nonexistent_block(self, store: SQLiteStore):
        assert not store.delete_block("nonexistent")

    def test_get_blocks_by_kind(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="Entry 1"),
                Block(id="e2", kind="entry", content="Entry 2"),
                Block(id="t1", kind="tag", title="career"),
                Block(id="d1", kind="document", title="daily.md"),
            ]
        )
        entries = store.get_blocks_by_kind("entry")
        assert len(entries) == 2

        tags = store.get_blocks_by_kind("tag")
        assert len(tags) == 1

    def test_tags_roundtrip(self, store: SQLiteStore):
        b = Block(id="b1", kind="entry", content="Test", tags=["career", "ai", "nested/tag"])
        store.insert_block(b)
        store.conn.commit()

        result = store.get_block("b1")
        assert result.tags == ["career", "ai", "nested/tag"]

    def test_metadata_roundtrip(self, store: SQLiteStore):
        b = Block(
            id="b1",
            kind="entry",
            content="Test",
            metadata={"h3_date": "2024-03-15", "section_index": 0},
        )
        store.insert_block(b)
        store.conn.commit()

        result = store.get_block("b1")
        assert result.metadata["h3_date"] == "2024-03-15"
        assert result.metadata["section_index"] == 0


class TestLinkCRUD:
    def test_insert_and_get_link(self, store: SQLiteStore):
        # Must create blocks first (foreign key)
        store.insert_blocks(
            [
                Block(id="a", kind="entry", content="A"),
                Block(id="b", kind="document", title="B"),
            ]
        )
        lnk = Link(from_id="a", to_id="b", kind="split_from")
        store.insert_link(lnk)
        store.conn.commit()

        links = store.get_links_from("a")
        assert len(links) == 1
        assert links[0].to_id == "b"
        assert links[0].kind == "split_from"

    def test_get_links_to(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="E1"),
                Block(id="e2", kind="entry", content="E2"),
                Block(id="t1", kind="tag", title="career"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="e1", to_id="t1", kind="tagged"),
                Link(from_id="e2", to_id="t1", kind="tagged"),
            ]
        )

        links = store.get_links_to("t1")
        assert len(links) == 2

    def test_get_links_filtered_by_kind(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="a", kind="entry", content="A"),
                Block(id="b", kind="document", title="B"),
                Block(id="c", kind="tag", title="C"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="a", to_id="b", kind="split_from"),
                Link(from_id="a", to_id="c", kind="tagged"),
            ]
        )

        split_links = store.get_links_from("a", kind="split_from")
        assert len(split_links) == 1

        tag_links = store.get_links_from("a", kind="tagged")
        assert len(tag_links) == 1

    def test_link_ignore_duplicate(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="a", kind="entry", content="A"),
                Block(id="b", kind="tag", title="B"),
            ]
        )
        lnk = Link(from_id="a", to_id="b", kind="tagged")
        store.insert_link(lnk)
        store.insert_link(lnk)  # duplicate — should be ignored
        store.conn.commit()

        links = store.get_links_from("a")
        assert len(links) == 1


class TestCascadeDelete:
    def test_delete_block_cascades_links(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="doc1", kind="document", title="Daily"),
                Block(id="e1", kind="entry", content="Entry 1"),
                Block(id="t1", kind="tag", title="career"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="e1", to_id="doc1", kind="split_from"),
                Link(from_id="e1", to_id="t1", kind="tagged"),
            ]
        )

        # Deleting entry should cascade its links
        store.delete_block("e1")

        links_from = store.get_links_from("e1")
        assert len(links_from) == 0

        # Doc and tag should still exist
        assert store.get_block("doc1") is not None
        assert store.get_block("t1") is not None


class TestFTSSearch:
    def test_fts_search_content(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="Thinking about career direction"),
                Block(id="e2", kind="entry", content="Notes on architecture review"),
                Block(id="e3", kind="entry", content="Weekend hiking plans"),
            ]
        )

        results = store.search_fts("career")
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_fts_search_title(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="d1", kind="document", title="Project Alpha", content="Overview"),
                Block(id="d2", kind="document", title="Team Meetings", content="Notes"),
            ]
        )

        results = store.search_fts("Alpha")
        assert len(results) == 1
        assert results[0].id == "d1"

    def test_fts_search_tags(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="A note", tags=["career", "growth"]),
                Block(id="e2", kind="entry", content="Another note", tags=["cooking"]),
            ]
        )

        results = store.search_fts("career")
        assert len(results) >= 1
        assert any(r.id == "e1" for r in results)

    def test_fts_no_results(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="Hello world"),
            ]
        )
        results = store.search_fts("nonexistent")
        assert len(results) == 0


class TestEmbeddingHelpers:
    def test_get_blocks_needing_embeddings(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="Needs embedding"),
                Block(id="e2", kind="entry", content="Also needs"),
                Block(id="e3", kind="entry", content="Has embedding", embedding=b"\x00" * 16),
            ]
        )

        needing = store.get_blocks_needing_embeddings()
        assert len(needing) == 2

    def test_update_embeddings(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="Test"),
            ]
        )
        store.update_embeddings({"e1": b"\x01" * 16})

        result = store.get_block("e1")
        assert result.embedding == b"\x01" * 16

    def test_get_blocks_with_embeddings(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="Has it", embedding=b"\x01" * 16),
                Block(id="e2", kind="entry", content="No embedding"),
            ]
        )

        with_emb = store.get_blocks_with_embeddings()
        assert len(with_emb) == 1
        assert with_emb[0].id == "e1"


class TestVectorSearch:
    def _make_blob(self, vec: list[float]) -> bytes:
        return np.array(vec, dtype=np.float32).tobytes()

    def test_ensure_vec_table_creates_table(self, store: SQLiteStore):
        store.ensure_vec_table(4)
        assert store._vec_table_exists()

    def test_semantic_search_returns_empty_without_vec_table(self, store: SQLiteStore):
        results = store.semantic_search([1.0, 0.0, 0.0, 0.0], k=5)
        assert results == []

    def test_semantic_search_finds_similar(self, store: SQLiteStore):
        store.ensure_vec_table(4)
        store.insert_blocks(
            [
                Block(id="e1", kind="entry", content="cats"),
                Block(id="e2", kind="entry", content="dogs"),
                Block(id="e3", kind="entry", content="cars"),
            ]
        )
        store.update_embeddings(
            {
                "e1": self._make_blob([1.0, 0.0, 0.0, 0.0]),
                "e2": self._make_blob([0.9, 0.1, 0.0, 0.0]),
                "e3": self._make_blob([0.0, 0.0, 1.0, 0.0]),
            }
        )
        # Query close to e1 and e2
        results = store.semantic_search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert "e1" in ids
        assert "e2" in ids
        # e3 should not appear
        assert "e3" not in ids

    def test_populate_vec_from_blocks(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(
                    id="e1",
                    kind="entry",
                    content="a",
                    embedding=self._make_blob([1.0, 0.0, 0.0, 0.0]),
                ),
                Block(
                    id="e2",
                    kind="entry",
                    content="b",
                    embedding=self._make_blob([0.0, 1.0, 0.0, 0.0]),
                ),
            ]
        )
        count = store.populate_vec_from_blocks(dim=4)
        assert count == 2
        results = store.semantic_search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0][0] == "e1"


class TestHubScoring:
    def test_hub_scores(self, store: SQLiteStore):
        # Create docs with entries and cross-links
        store.insert_blocks(
            [
                Block(
                    id="doc1",
                    kind="document",
                    title="Hub Note",
                    metadata={"source_path": "hub.md"},
                ),
                Block(id="e1", kind="entry", content="Entry 1"),
                Block(id="e2", kind="entry", content="Entry 2"),
                Block(id="e3", kind="entry", content="Entry 3"),
                Block(
                    id="doc2",
                    kind="document",
                    title="Other Note",
                    metadata={"source_path": "other.md"},
                ),
                Block(id="e4", kind="entry", content="Entry 4"),
            ]
        )
        store.insert_links(
            [
                # doc1 has 3 entries
                Link(from_id="e1", to_id="doc1", kind="split_from"),
                Link(from_id="e2", to_id="doc1", kind="split_from"),
                Link(from_id="e3", to_id="doc1", kind="split_from"),
                # doc2 has 1 entry
                Link(from_id="e4", to_id="doc2", kind="split_from"),
                # External link pointing TO doc1 (in_link for doc1)
                Link(from_id="e4", to_id="doc1", kind="links_to"),
            ]
        )

        scores = store.get_hub_scores(limit=10)
        assert len(scores) >= 1
        # doc1 should have highest score (3 entries + 1 in_link)
        assert scores[0]["doc_id"] == "doc1"
        assert scores[0]["entry_count"] == 3
        assert scores[0]["in_links"] == 1
        assert scores[0]["hub_score"] > 0


class TestStats:
    def test_get_stats(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="d1", kind="document", title="Doc"),
                Block(id="e1", kind="entry", content="Entry", embedding=b"\x00"),
                Block(id="t1", kind="tag", title="career"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="e1", to_id="d1", kind="split_from"),
                Link(from_id="e1", to_id="t1", kind="tagged"),
            ]
        )

        stats = store.get_stats()
        assert stats["total_blocks"] == 3
        assert stats["total_links"] == 2
        assert stats["blocks_by_kind"]["entry"] == 1
        assert stats["embedded_blocks"] == 1
