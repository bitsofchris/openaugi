"""Tests for SQLiteStore — block CRUD, link CRUD, FTS, cascade delete, vector search."""

import numpy as np

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.store.sqlite import SQLiteStore


class TestBlockCRUD:
    def test_insert_and_get_block(self, store: SQLiteStore):
        b = Block(id="b1", kind="data_block", content="Hello", source="vault", title="Test")
        store.insert_block(b)
        store.conn.commit()

        result = store.get_block("b1")
        assert result is not None
        assert result.id == "b1"
        assert result.content == "Hello"
        assert result.kind == "data_block"

    def test_insert_block_ignore_duplicate(self, store: SQLiteStore):
        b1 = Block(id="b1", kind="data_block", content="First")
        b2 = Block(id="b1", kind="data_block", content="Second")
        store.insert_block(b1)
        store.insert_block(b2)
        store.conn.commit()

        result = store.get_block("b1")
        assert result.content == "First"  # second insert ignored

    def test_insert_blocks_batch(self, store: SQLiteStore):
        blocks = [Block(id=f"b{i}", kind="data_block", content=f"Content {i}") for i in range(5)]
        count = store.insert_blocks(blocks)
        assert count == 5

        for i in range(5):
            assert store.get_block(f"b{i}") is not None

    def test_delete_block(self, store: SQLiteStore):
        b = Block(id="b1", kind="data_block", content="Hello")
        store.insert_block(b)
        store.conn.commit()

        assert store.delete_block("b1")
        assert store.get_block("b1") is None

    def test_delete_nonexistent_block(self, store: SQLiteStore):
        assert not store.delete_block("nonexistent")

    def test_get_blocks_by_ids(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="b1", kind="data_block", content="First"),
                Block(id="b2", kind="data_block", content="Second"),
                Block(id="b3", kind="data_block", content="Third"),
            ]
        )
        result = store.get_blocks_by_ids(["b1", "b3", "nonexistent"])
        assert len(result) == 2
        assert "b1" in result
        assert "b3" in result
        assert "nonexistent" not in result
        assert result["b1"].content == "First"
        assert result["b3"].content == "Third"

    def test_get_blocks_by_ids_empty(self, store: SQLiteStore):
        result = store.get_blocks_by_ids([])
        assert result == {}

    def test_get_blocks_by_kind(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="Entry 1"),
                Block(id="e2", kind="data_block", content="Entry 2"),
                Block(id="t1", kind="context_block:tag", title="career"),
                Block(id="d1", kind="context_block:document", title="daily.md"),
            ]
        )
        entries = store.get_blocks_by_kind("data_block")
        assert len(entries) == 2

        tags = store.get_blocks_by_kind("context_block:tag")
        assert len(tags) == 1

    def test_tags_roundtrip(self, store: SQLiteStore):
        b = Block(id="b1", kind="data_block", content="Test", tags=["career", "ai", "nested/tag"])
        store.insert_block(b)
        store.conn.commit()

        result = store.get_block("b1")
        assert result.tags == ["career", "ai", "nested/tag"]

    def test_metadata_roundtrip(self, store: SQLiteStore):
        b = Block(
            id="b1",
            kind="data_block",
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
                Block(id="a", kind="data_block", content="A"),
                Block(id="b", kind="context_block:document", title="B"),
            ]
        )
        lnk = Link(from_id="a", to_id="b", kind="contains")
        store.insert_link(lnk)
        store.conn.commit()

        links = store.get_links_from("a")
        assert len(links) == 1
        assert links[0].to_id == "b"
        assert links[0].kind == "contains"

    def test_get_links_to(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="E1"),
                Block(id="e2", kind="data_block", content="E2"),
                Block(id="t1", kind="context_block:tag", title="career"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="e1", to_id="t1", kind="groups"),
                Link(from_id="e2", to_id="t1", kind="groups"),
            ]
        )

        links = store.get_links_to("t1")
        assert len(links) == 2

    def test_get_links_filtered_by_kind(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="a", kind="data_block", content="A"),
                Block(id="b", kind="context_block:document", title="B"),
                Block(id="c", kind="context_block:tag", title="C"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="a", to_id="b", kind="contains"),
                Link(from_id="a", to_id="c", kind="groups"),
            ]
        )

        split_links = store.get_links_from("a", kind="contains")
        assert len(split_links) == 1

        tag_links = store.get_links_from("a", kind="groups")
        assert len(tag_links) == 1

    def test_link_ignore_duplicate(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="a", kind="data_block", content="A"),
                Block(id="b", kind="context_block:tag", title="B"),
            ]
        )
        lnk = Link(from_id="a", to_id="b", kind="groups")
        store.insert_link(lnk)
        store.insert_link(lnk)  # duplicate — should be ignored
        store.conn.commit()

        links = store.get_links_from("a")
        assert len(links) == 1


class TestCascadeDelete:
    def test_delete_block_cascades_links(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="doc1", kind="context_block:document", title="Daily"),
                Block(id="e1", kind="data_block", content="Entry 1"),
                Block(id="t1", kind="context_block:tag", title="career"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="e1", to_id="doc1", kind="contains"),
                Link(from_id="e1", to_id="t1", kind="groups"),
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
                Block(id="e1", kind="data_block", content="Thinking about career direction"),
                Block(id="e2", kind="data_block", content="Notes on architecture review"),
                Block(id="e3", kind="data_block", content="Weekend hiking plans"),
            ]
        )

        results = store.search_fts("career")
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_fts_search_title(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(
                    id="d1",
                    kind="context_block:document",
                    title="Project Alpha",
                    content="Overview",
                ),
                Block(
                    id="d2", kind="context_block:document", title="Team Meetings", content="Notes"
                ),
            ]
        )

        results = store.search_fts("Alpha")
        assert len(results) == 1
        assert results[0].id == "d1"

    def test_fts_search_tags(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="A note", tags=["career", "growth"]),
                Block(id="e2", kind="data_block", content="Another note", tags=["cooking"]),
            ]
        )

        results = store.search_fts("career")
        assert len(results) >= 1
        assert any(r.id == "e1" for r in results)

    def test_fts_no_results(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="Hello world"),
            ]
        )
        results = store.search_fts("nonexistent")
        assert len(results) == 0

    def test_fts_special_chars_dont_crash(self, store: SQLiteStore):
        """FTS5 operators in user input should not cause SQL errors."""
        store.insert_blocks(
            [
                Block(
                    id="d1",
                    kind="context_block:document",
                    title="MOC - Advice on Finding Your Niche",
                    content="Overview of niche advice",
                ),
            ]
        )
        # Title search with dashes and spaces — previously crashed with
        # "no such column: Advice" because FTS5 parsed `-` as NOT operator
        results = store.search_fts("title:MOC - Advice on Finding Your Niche")
        assert len(results) == 1
        assert results[0].id == "d1"

    def test_fts_keyword_with_special_chars(self, store: SQLiteStore):
        """Plain keyword queries with FTS5 operators should be safe."""
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="pros and cons of React vs Vue"),
            ]
        )
        # Bare `-` and other operators should not crash
        results = store.search_fts("React - pros")
        assert len(results) >= 0  # no crash is the test

    def test_fts_commas_in_query(self, store: SQLiteStore):
        """Commas in user queries should not cause FTS5 syntax errors."""
        store.insert_blocks(
            [
                Block(
                    id="e2",
                    kind="data_block",
                    content="finding work that feels like play and niche evolution",
                ),
            ]
        )
        # Commas previously caused: fts5: syntax error near ","
        results = store.search_fts(
            "finding work that feels like play, niche evolution, specific knowledge"
        )
        assert len(results) >= 0  # no crash is the test


class TestEmbeddingHelpers:
    def test_get_blocks_needing_embeddings(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="Needs embedding"),
                Block(id="e2", kind="data_block", content="Also needs"),
                Block(id="e3", kind="data_block", content="Has embedding", embedding=b"\x00" * 16),
            ]
        )

        needing = store.get_blocks_needing_embeddings()
        assert len(needing) == 2

    def test_update_embeddings(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="Test"),
            ]
        )
        store.update_embeddings({"e1": b"\x01" * 16})

        result = store.get_block("e1")
        assert result.embedding == b"\x01" * 16

    def test_get_blocks_with_embeddings(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="e1", kind="data_block", content="Has it", embedding=b"\x01" * 16),
                Block(id="e2", kind="data_block", content="No embedding"),
            ]
        )

        with_emb = store.get_blocks_with_embeddings()
        assert len(with_emb) == 1
        assert with_emb[0].id == "e1"


class TestGetTagsForIds:
    def test_combines_user_tags_and_augi_tags(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(id="t1", kind="data_block", content="a", tags=["layer/bronze", "idea"]),
                Block(id="t2", kind="data_block", content="b", metadata={"augi_tags": ["area/x"]}),
                Block(id="t3", kind="data_block", content="c"),
            ]
        )

        tags = store.get_tags_for_ids(["t1", "t2", "t3", "missing"])
        assert tags["t1"] == ["layer/bronze", "idea"]
        assert tags["t2"] == ["area/x"]
        assert tags["t3"] == []
        assert "missing" not in tags

    def test_empty_ids(self, store: SQLiteStore):
        assert store.get_tags_for_ids([]) == {}


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
                Block(id="e1", kind="data_block", content="cats"),
                Block(id="e2", kind="data_block", content="dogs"),
                Block(id="e3", kind="data_block", content="cars"),
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
                    kind="data_block",
                    content="a",
                    embedding=self._make_blob([1.0, 0.0, 0.0, 0.0]),
                ),
                Block(
                    id="e2",
                    kind="data_block",
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
                    kind="context_block:document",
                    title="Hub Note",
                    metadata={"source_path": "hub.md"},
                ),
                Block(id="e1", kind="data_block", content="Entry 1"),
                Block(id="e2", kind="data_block", content="Entry 2"),
                Block(id="e3", kind="data_block", content="Entry 3"),
                Block(
                    id="doc2",
                    kind="context_block:document",
                    title="Other Note",
                    metadata={"source_path": "other.md"},
                ),
                Block(id="e4", kind="data_block", content="Entry 4"),
            ]
        )
        store.insert_links(
            [
                # doc1 has 3 entries
                Link(from_id="e1", to_id="doc1", kind="contains"),
                Link(from_id="e2", to_id="doc1", kind="contains"),
                Link(from_id="e3", to_id="doc1", kind="contains"),
                # doc2 has 1 entry
                Link(from_id="e4", to_id="doc2", kind="contains"),
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
                Block(id="d1", kind="context_block:document", title="Doc"),
                Block(id="e1", kind="data_block", content="Entry", embedding=b"\x00"),
                Block(id="t1", kind="context_block:tag", title="career"),
            ]
        )
        store.insert_links(
            [
                Link(from_id="e1", to_id="d1", kind="contains"),
                Link(from_id="e1", to_id="t1", kind="groups"),
            ]
        )

        stats = store.get_stats()
        assert stats["total_blocks"] == 3
        assert stats["total_links"] == 2
        assert stats["blocks_by_kind"]["data_block"] == 1
        assert stats["embedded_blocks"] == 1


class TestGetBlocksFiltered:
    def _seed(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(
                    id="capture1",
                    kind="data_block",
                    content="a thought",
                    block_time="2026-07-01",
                    metadata={"source_path": "Journal/2026-07-01.md"},
                ),
                Block(
                    id="capture2",
                    kind="data_block",
                    content="another thought",
                    block_time="2026-07-02",
                    metadata={"source_path": "Journal/2026-07-02.md"},
                ),
                Block(
                    id="derived1",
                    kind="data_block",
                    content="a derived view",
                    block_time="2026-07-03",
                    metadata={"source_path": "OpenAugi/Views/Dashboard.md"},
                ),
                Block(
                    id="nopath",
                    kind="data_block",
                    content="no source path",
                    block_time="2026-07-04",
                    metadata={},
                ),
            ]
        )

    def test_exclude_path_prefix(self, store: SQLiteStore):
        self._seed(store)
        blocks, total = store.get_blocks_filtered(
            kind="data_block", exclude_path_prefix="OpenAugi/"
        )
        ids = {b.id for b in blocks}
        assert ids == {"capture1", "capture2", "nopath"}
        assert total == 3

    def test_include_path_prefix(self, store: SQLiteStore):
        """The review-pass second query: reach into an excluded tree.

        Query 1 excludes OpenAugi/ wholesale; query 2 names the one folder
        under it that is truth rather than generated output.
        """
        self._seed(store)
        store.insert_blocks(
            [
                Block(
                    id="phone",
                    kind="data_block",
                    content="a mobile capture",
                    block_time="2026-07-05",
                    metadata={"source_path": "OpenAugi/Capture/2026-07-05.md"},
                ),
            ]
        )
        blocks, total = store.get_blocks_filtered(
            kind="data_block", include_path_prefix="OpenAugi/Capture/"
        )
        assert {b.id for b in blocks} == {"phone"}
        assert total == 1  # the count honours the include filter too

    def test_include_drops_blocks_without_a_source_path(self, store: SQLiteStore):
        """A block with no source_path can't be under the requested folder."""
        self._seed(store)
        blocks, total = store.get_blocks_filtered(
            kind="data_block", include_path_prefix="Journal/"
        )
        assert {b.id for b in blocks} == {"capture1", "capture2"}
        assert "nopath" not in {b.id for b in blocks}
        assert total == 2

    def test_include_wildcards_match_literally(self, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(
                    id="under",
                    kind="data_block",
                    content="x",
                    metadata={"source_path": "My_Dir/note.md"},
                ),
                Block(
                    id="lookalike",
                    kind="data_block",
                    content="y",
                    metadata={"source_path": "MyXDir/note.md"},
                ),
            ]
        )
        blocks, total = store.get_blocks_filtered(kind="data_block", include_path_prefix="My_Dir/")
        assert {b.id for b in blocks} == {"under"}
        assert total == 1

    def test_include_and_exclude_compose(self, store: SQLiteStore):
        """Both applied: include narrows to a folder, exclude carves out of it."""
        self._seed(store)
        store.insert_blocks(
            [
                Block(
                    id="phone",
                    kind="data_block",
                    content="a mobile capture",
                    block_time="2026-07-05",
                    metadata={"source_path": "OpenAugi/Capture/2026-07-05.md"},
                ),
            ]
        )
        blocks, total = store.get_blocks_filtered(
            kind="data_block",
            include_path_prefix="OpenAugi/",
            exclude_path_prefix="OpenAugi/Views/",
        )
        assert {b.id for b in blocks} == {"phone"}
        assert total == 1

    def test_no_prefix_returns_all(self, store: SQLiteStore):
        self._seed(store)
        blocks, total = store.get_blocks_filtered(kind="data_block")
        assert total == 4

    def test_prefix_wildcards_match_literally(self, store: SQLiteStore):
        # An underscore in the prefix must not act as a LIKE wildcard
        store.insert_blocks(
            [
                Block(
                    id="under",
                    kind="data_block",
                    content="x",
                    metadata={"source_path": "My_Dir/note.md"},
                ),
                Block(
                    id="notunder",
                    kind="data_block",
                    content="y",
                    metadata={"source_path": "MyXDir/note.md"},
                ),
            ]
        )
        blocks, total = store.get_blocks_filtered(kind="data_block", exclude_path_prefix="My_Dir/")
        ids = {b.id for b in blocks}
        assert ids == {"notunder"}
        assert total == 1


class TestReviewState:
    def test_default_state_is_empty(self, store: SQLiteStore):
        state = store.get_review_state()
        assert state == {"last_run": None, "last_summary": None}

    def test_set_and_get_roundtrip(self, store: SQLiteStore):
        store.set_review_state("2026-07-06T12:00:00+00:00", "routed 10 blocks")
        state = store.get_review_state()
        assert state["last_run"] == "2026-07-06T12:00:00+00:00"
        assert state["last_summary"] == "routed 10 blocks"

    def test_empty_summary_preserves_previous(self, store: SQLiteStore):
        store.set_review_state("2026-07-06T12:00:00+00:00", "first run")
        store.set_review_state("2026-07-07T12:00:00+00:00")
        state = store.get_review_state()
        assert state["last_run"] == "2026-07-07T12:00:00+00:00"
        assert state["last_summary"] == "first run"

    def test_persists_across_reconnect(self, store: SQLiteStore):
        store.set_review_state("2026-07-06T12:00:00+00:00", "persisted")
        store.close()
        state = store.get_review_state()
        assert state["last_run"] == "2026-07-06T12:00:00+00:00"
        assert state["last_summary"] == "persisted"


class TestAfterIngested:
    """The review-pass queue filter — ingest time, not content date.

    Regression coverage for the pass-#5 bug (2026-07-12): date-only
    block_time sorts before any same-day timestamp, and edited blocks
    re-ingest keeping their note's old date, so after= (block_time)
    silently dropped both from the queue.
    """

    def _seed(self, store: SQLiteStore):
        store.insert_blocks(
            [
                # Captured today as a daily note: date-only block_time,
                # ingested after the mark.
                Block(
                    id="today",
                    kind="data_block",
                    content="this morning's capture",
                    block_time="2026-07-12",
                    ingested_at="2026-07-12T12:21:52.009852Z",
                ),
                # Yesterday's note edited late: re-derived block keeps the
                # old content date but a fresh ingested_at.
                Block(
                    id="reingested",
                    kind="data_block",
                    content="edited last night",
                    block_time="2026-07-11",
                    ingested_at="2026-07-12T12:21:52.017928Z",
                ),
                # Processed by the previous pass — must stay out of the queue.
                Block(
                    id="old",
                    kind="data_block",
                    content="already routed",
                    block_time="2026-07-11",
                    ingested_at="2026-07-11T22:49:05.932347Z",
                ),
            ]
        )

    # The high-water mark as mark_review_complete stores it:
    # datetime.now(UTC).isoformat() — microseconds, "+00:00" suffix.
    MARK = "2026-07-12T00:29:50.356550+00:00"

    def test_after_block_time_misses_same_day_date_only(self, store: SQLiteStore):
        # Documents the bug: the old filter returns nothing for this window.
        self._seed(store)
        blocks, total = store.get_blocks_filtered(kind="data_block", after=self.MARK)
        assert total == 0

    def test_after_ingested_catches_new_and_reingested(self, store: SQLiteStore):
        self._seed(store)
        blocks, total = store.get_blocks_filtered(kind="data_block", after_ingested=self.MARK)
        assert {b.id for b in blocks} == {"today", "reingested"}
        assert total == 2

    def test_after_ingested_z_suffix_input(self, store: SQLiteStore):
        self._seed(store)
        blocks, _ = store.get_blocks_filtered(
            kind="data_block", after_ingested="2026-07-12T00:29:50Z"
        )
        assert {b.id for b in blocks} == {"today", "reingested"}

    def test_after_ingested_combines_with_exclude_prefix(self, store: SQLiteStore):
        self._seed(store)
        store.insert_blocks(
            [
                Block(
                    id="derived",
                    kind="data_block",
                    content="agent output",
                    ingested_at="2026-07-12T12:30:00.000000Z",
                    metadata={"source_path": "OpenAugi/Views/View - X.md"},
                )
            ]
        )
        blocks, _ = store.get_blocks_filtered(
            kind="data_block", after_ingested=self.MARK, exclude_path_prefix="OpenAugi/"
        )
        assert {b.id for b in blocks} == {"today", "reingested"}


class TestNormalizeUtcTimestamp:
    def test_offset_suffix_to_storage_format(self):
        from openaugi.store.sqlite import normalize_utc_timestamp

        assert (
            normalize_utc_timestamp("2026-07-12T00:29:50.356550+00:00")
            == "2026-07-12T00:29:50.356550Z"
        )

    def test_date_only_becomes_midnight_utc(self):
        from openaugi.store.sqlite import normalize_utc_timestamp

        assert normalize_utc_timestamp("2026-07-12") == "2026-07-12T00:00:00.000000Z"

    def test_non_utc_offset_is_converted(self):
        from openaugi.store.sqlite import normalize_utc_timestamp

        assert (
            normalize_utc_timestamp("2026-07-12T08:00:00-04:00") == "2026-07-12T12:00:00.000000Z"
        )

    def test_naive_input_assumed_utc(self):
        from openaugi.store.sqlite import normalize_utc_timestamp

        assert normalize_utc_timestamp("2026-07-12T05:00:00") == "2026-07-12T05:00:00.000000Z"

    def test_garbage_raises(self):
        import pytest

        from openaugi.store.sqlite import normalize_utc_timestamp

        with pytest.raises(ValueError):
            normalize_utc_timestamp("not a timestamp")
