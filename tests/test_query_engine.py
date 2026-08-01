"""Unit tests for the query engine — the rules, tested at the engine boundary.

The golden harness pins the MCP wire format; these tests pin the engine
semantics directly (typed results, full blocks) so HTTP/CLI adapters get
the same guarantees without going through MCP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openaugi.query import QuerySpec, engine
from openaugi.store.sqlite import SQLiteStore
from tests.query_golden_corpus import FakeEmbedder, build_store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> SQLiteStore:
    db = tmp_path_factory.mktemp("engine") / "engine.db"
    build_store(db)
    s = SQLiteStore(db)
    yield s
    s.close()


class TestQuerySpec:
    def test_mode_precedence_title_first(self):
        spec = QuerySpec(title="t", keyword="k", query="q")
        assert spec.mode == "title"
        assert QuerySpec(keyword="k", query="q").mode == "keyword"
        assert QuerySpec(query="q").mode == "semantic"
        assert QuerySpec(after="2026-01-01").mode == "browse"

    def test_is_empty(self):
        assert QuerySpec().is_empty()
        assert QuerySpec(has_task=False).is_empty()  # falsy = not provided
        assert not QuerySpec(has_task=True).is_empty()
        assert not QuerySpec(tags=["idea"]).is_empty()

    def test_round_trips_json(self):
        spec = QuerySpec(has_task=True, after="-14d", k=25)
        again = QuerySpec.model_validate_json(spec.model_dump_json())
        assert again == spec


class TestEngineRun:
    def test_empty_spec_raises(self, store: SQLiteStore):
        with pytest.raises(engine.EmptyQuerySpec):
            engine.run(store, QuerySpec())

    def test_semantic_without_model_raises(self, store: SQLiteStore):
        with pytest.raises(ValueError, match="embedding model"):
            engine.run(store, QuerySpec(query="quantum"))

    def test_keyword_returns_full_blocks(self, store: SQLiteStore):
        result = engine.run(store, QuerySpec(keyword="quantum"))
        assert result.mode == "keyword"
        assert result.blocks
        # Engine returns FULL blocks — content beyond any truncation limit
        # is an adapter concern, never lost here.
        assert all(b.content for b in result.blocks)

    def test_has_task_excludes_bronze(self, store: SQLiteStore):
        result = engine.run(store, QuerySpec(has_task=True, after="2026-01-01"))
        ids = [b.id for b in result.blocks]
        assert "b2-open-task" in ids  # open checkbox
        assert "b4-tagged-task" in ids  # type/task tag
        assert "b3-bronze-task" not in ids  # bronze never counts

    def test_path_exclusion_all_modes(self, store: SQLiteStore):
        browse = engine.run(store, QuerySpec(after="2026-01-01", exclude_path_prefix="OpenAugi/"))
        assert all(
            not b.metadata.get("source_path", "").startswith("OpenAugi/") for b in browse.blocks
        )
        keyword = engine.run(store, QuerySpec(keyword="quantum", exclude_path_prefix="OpenAugi/"))
        assert all(
            not b.metadata.get("source_path", "").startswith("OpenAugi/") for b in keyword.blocks
        )

    def test_path_inclusion_all_modes(self, store: SQLiteStore):
        """include_path_prefix is the mirror: keep only what's under the folder.

        This is the review pass's second query — after excluding OpenAugi/
        wholesale, it names OpenAugi/Capture/ to pick the phone stream back up.
        """
        browse = engine.run(store, QuerySpec(after="2026-01-01", include_path_prefix="OpenAugi/"))
        assert browse.blocks
        assert all(
            b.metadata.get("source_path", "").startswith("OpenAugi/") for b in browse.blocks
        )

        keyword = engine.run(store, QuerySpec(keyword="quantum", include_path_prefix="OpenAugi/"))
        assert [b.id for b in keyword.blocks] == ["b5-derived"]

        # A folder nothing lives under returns empty, not everything
        none = engine.run(store, QuerySpec(keyword="quantum", include_path_prefix="Nowhere/"))
        assert none.blocks == []

    def test_include_prefix_alone_is_a_valid_query(self):
        """ "Everything under this folder" is a complete question; a bare
        exclusion is not."""
        assert not QuerySpec(include_path_prefix="OpenAugi/Capture/").is_empty()
        assert QuerySpec(exclude_path_prefix="OpenAugi/").is_empty()

    def test_after_ingested_bound(self, store: SQLiteStore):
        result = engine.run(store, QuerySpec(after_ingested="2026-05-01T00:00:00Z"))
        ids = [b.id for b in result.blocks]
        assert "b6-old-ingest" not in ids  # ingested 2026-01-01
        assert "b1-quantum-idea" in ids

    def test_browse_pagination_envelope(self, store: SQLiteStore):
        p1 = engine.run(store, QuerySpec(after="2026-01-01", k=3))
        assert p1.has_more is True
        assert p1.next_offset == 3
        assert p1.total is not None and p1.total > 3
        p2 = engine.run(store, QuerySpec(after="2026-01-01", k=3, offset=3))
        assert {b.id for b in p1.blocks}.isdisjoint({b.id for b in p2.blocks})

    def test_reference_documents_collapsed(self, store: SQLiteStore):
        result = engine.run(store, QuerySpec(after="2026-01-01"))
        assert result.reference_block_count == 2
        assert len(result.reference_documents) == 1
        group = result.reference_documents[0]
        assert group["block_count"] == 2
        assert group["source_tags"] == ["source/readwise"]
        assert group["title"] == "Zebra Protocol Article"
        # grouped blocks are not in the main results
        assert all(b.id not in ("b7-ref-one", "b8-ref-two") for b in result.blocks)

    def test_semantic_scores_and_filters(self, store: SQLiteStore):
        result = engine.run(
            store,
            QuerySpec(query="quantum garden", k=5, tags=["idea"]),
            embedding_model=FakeEmbedder(),
        )
        assert result.mode == "semantic"
        assert [b.id for b in result.blocks] == ["b1-quantum-idea"]
        # score = round(1 - distance, 4); fake L2 distances can exceed 1,
        # so only the shape is asserted here (real similarity ∈ (0, 1]).
        assert isinstance(result.scores["b1-quantum-idea"], float)
        assert result.scores["b1-quantum-idea"] <= 1

    def test_engine_never_imports_transport(self):
        """The boundary rule from the plan doc: semantics, not presentation."""
        import openaugi.query.engine as eng
        import openaugi.query.spec as spec_mod

        for mod in (eng, spec_mod):
            source = Path(mod.__file__).read_text(encoding="utf-8")
            assert "import mcp" not in source
            assert "starlette" not in source
            assert "fastapi" not in source
