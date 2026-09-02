"""Provenance filtering and get_context filters at the engine boundary.

Complements test_query_engine (golden-corpus semantics) with a small
purpose-built store where provenance is stamped, so both the explicit
`provenance=[...]` keep-list and the `[retrieval] exclude_provenance`
default can be pinned. See docs/plans/query-provenance-and-dates.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.query import QuerySpec, engine
from openaugi.store.sqlite import SQLiteStore
from tests.query_golden_corpus import DIM, FakeEmbedder, _blob

NO_DEFAULT = {"retrieval": {"exclude_provenance": []}}
DROP_REF = {"retrieval": {"exclude_provenance": ["reference"]}}


def _block(id: str, content: str, *, day: str, path: str, prov: str | None, tags=None) -> Block:
    md: dict = {"source_path": path}
    if prov:
        md["provenance"] = prov
    return Block(
        id=id,
        kind="data_block",
        content=content,
        source="vault",
        title=Path(path).stem,
        tags=tags or [],
        block_time=day,
        metadata=md,
        ingested_at=f"{day}T10:00:00.000Z",
    )


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> SQLiteStore:
    db = tmp_path_factory.mktemp("filters") / "f.db"
    s = SQLiteStore(db)
    blocks = [
        _block(
            "h1", "zebra thinking, my own", day="2026-03-05", path="J/2026-03-05.md", prov="human"
        ),
        _block(
            "h2", "zebra again in June", day="2026-06-05", path="J/2026-06-05.md", prov="human"
        ),
        _block(
            "a1", "zebra summary by a model", day="2026-03-06", path="OpenAugi/N/x.md", prov="ai"
        ),
        _block(
            "r1",
            "zebra quote from a podcast",
            day="2026-03-07",
            path="_private/2-Reference/q.md",
            prov="reference",
            tags=["source/podcast"],
        ),
        # Pre-provenance row: no stamp at all. Must behave as human.
        _block(
            "u1", "zebra, unstamped legacy row", day="2026-03-08", path="J/legacy.md", prov=None
        ),
        _block(
            "t1",
            "zebra tagged idea",
            day="2026-03-09",
            path="J/idea.md",
            prov="human",
            tags=["idea"],
        ),
    ]
    s.insert_blocks(blocks)
    s.insert_links([Link(from_id="h1", to_id="h2", kind="links_to")])
    s.ensure_vec_table(DIM)
    s.update_embeddings({b.id: _blob(b.content or "") for b in blocks})
    yield s
    s.close()


def _ids(blocks) -> set[str]:
    return {b.id for b in blocks}


class TestRunProvenance:
    def test_keyword_keep_list(self, store):
        r = engine.run(store, QuerySpec(keyword="zebra", provenance=["human"]))
        assert _ids(r.blocks) == {"h1", "h2", "u1", "t1"}

    def test_keyword_no_default_exclusion(self, store):
        """Keyword is precise by construction: the config default does not apply."""
        r = engine.run(store, QuerySpec(keyword="zebra"), config=DROP_REF)
        assert "r1" in _ids(r.blocks)

    def test_browse_keep_list(self, store):
        r = engine.run(store, QuerySpec(after="2026-01-01", provenance=["ai", "reference"]))
        assert _ids(r.blocks) | {"r1"} == {"a1", "r1"}  # r1 lands in reference_documents
        assert r.reference_block_count == 1

    def test_semantic_default_drops_reference(self, store):
        r = engine.run(
            store, QuerySpec(query="zebra"), embedding_model=FakeEmbedder(), config=DROP_REF
        )
        assert "r1" not in _ids(r.blocks)
        assert {"h1", "a1", "u1"} <= _ids(r.blocks)

    def test_semantic_explicit_list_overrides_default(self, store):
        r = engine.run(
            store,
            QuerySpec(query="zebra", provenance=["reference"]),
            embedding_model=FakeEmbedder(),
            config=DROP_REF,
        )
        assert _ids(r.blocks) == {"r1"}

    def test_semantic_empty_default_keeps_everything(self, store):
        r = engine.run(
            store, QuerySpec(query="zebra"), embedding_model=FakeEmbedder(), config=NO_DEFAULT
        )
        assert "r1" in _ids(r.blocks)

    def test_provenance_alone_is_a_complete_query(self):
        assert not QuerySpec(provenance=["human"]).is_empty()


class TestContextFilters:
    def _ctx(self, store, **kw):
        kw.setdefault("config", NO_DEFAULT)
        return engine.context(
            store, "zebra", k=10, expand=False, embedding_model=FakeEmbedder(), **kw
        )

    def test_no_filters_returns_all(self, store):
        ctx = self._ctx(store)
        assert _ids(e.block for e in ctx.seen) == {"h1", "h2", "a1", "r1", "u1", "t1"}

    def test_date_window(self, store):
        ctx = self._ctx(store, after="2026-03-01", before="2026-03-31")
        assert "h2" not in _ids(e.block for e in ctx.seen)
        assert "h1" in _ids(e.block for e in ctx.seen)

    def test_tags(self, store):
        ctx = self._ctx(store, tags=["idea"])
        assert _ids(e.block for e in ctx.seen) == {"t1"}

    def test_path_prefixes(self, store):
        excl = self._ctx(store, exclude_path_prefix="OpenAugi/")
        assert "a1" not in _ids(e.block for e in excl.seen)
        incl = self._ctx(store, include_path_prefix="OpenAugi/")
        assert _ids(e.block for e in incl.seen) == {"a1"}

    def test_provenance_keep_list_and_unstamped_counts_as_human(self, store):
        ctx = self._ctx(store, provenance=["human"])
        assert _ids(e.block for e in ctx.seen) == {"h1", "h2", "u1", "t1"}

    def test_default_exclusion_applies(self, store):
        ctx = self._ctx(store, config=DROP_REF)
        assert "r1" not in _ids(e.block for e in ctx.seen)

    def test_filters_can_empty_the_pool(self, store):
        ctx = self._ctx(store, after="2030-01-01")
        assert not ctx.had_candidates

    def test_expand_still_works_with_filters(self, store):
        ctx = engine.context(
            store,
            "zebra",
            k=10,
            expand=True,
            embedding_model=FakeEmbedder(),
            config=NO_DEFAULT,
            before="2026-03-31",
        )
        # h2 is outside the window but reachable by link from h1; expansion is
        # unfiltered on purpose (it is context for a result, not a result).
        assert "h2" in _ids(e.block for e in ctx.expanded)

    def test_overfetch_doubles_when_filtered(self, store, monkeypatch):
        calls: list[int] = []
        real = store.semantic_search

        def spy(vec, k):
            calls.append(k)
            return real(vec, k=k)

        monkeypatch.setattr(store, "semantic_search", spy)
        self._ctx(store)
        self._ctx(store, after="2026-01-01")
        assert calls[1] == calls[0] * 2


def test_similarity_helper_is_unchanged():
    """Guard: the filter work must not touch scoring math."""
    assert engine._similarity(0.0) == pytest.approx(1.0)
    assert np.isfinite(engine._similarity(0.5))
