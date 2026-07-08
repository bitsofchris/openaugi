"""Tests for the idea-lineage pre-compute.

Uses a fake embedding model and synthetic vectors in a tmp store —
no API calls, no real vault data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openaugi.model.block import Block
from openaugi.pipeline.lineage import (
    _era_of,
    _era_sequence,
    _slugify,
    compute_lineage,
    render_lineage_markdown,
    write_lineage_sidecar,
)
from openaugi.store.sqlite import SQLiteStore

DIM = 8


class FakeEmbedder:
    """Embeds every query as the first basis vector."""

    def embed_query(self, query: str) -> list[float]:
        vec = np.zeros(DIM, dtype=np.float32)
        vec[0] = 1.0
        return vec.tolist()


def _vec(similar: bool) -> bytes:
    """A unit vector near (similar) or orthogonal to (not) the query vector."""
    v = np.zeros(DIM, dtype=np.float32)
    if similar:
        v[0], v[1] = 1.0, 0.15
    else:
        v[3] = 1.0
    return (v / np.linalg.norm(v)).astype(np.float32).tobytes()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "test.db")
    s.ensure_vec_table(DIM)
    blocks = [
        # The idea across three eras, with a two-quarter gap in 2025
        Block(
            id="b1",
            kind="data_block",
            title="first spark",
            content="idea appears",
            block_time="2024-11-05",
            embedding=_vec(True),
            metadata={"source_path": "Journal/2024-11-05.md"},
        ),
        Block(
            id="b2",
            kind="data_block",
            title="revision",
            content="idea grows",
            block_time="2025-01-20",
            embedding=_vec(True),
            metadata={"source_path": "Journal/2025-01-20.md"},
        ),
        Block(
            id="b3",
            kind="data_block",
            title="return",
            content="idea returns changed",
            block_time="2025-10-02",
            embedding=_vec(True),
            metadata={"source_path": "Journal/2025-10-02.md"},
        ),
        # Third-party evidence — matched but flagged
        Block(
            id="b4",
            kind="data_block",
            title="a book highlight",
            content="related quote",
            block_time="2025-10-05",
            embedding=_vec(True),
            tags=["source/readwise"],
            metadata={"source_path": "Reference/Readwise/book.md"},
        ),
        # Derived artifact — excluded
        Block(
            id="b5",
            kind="data_block",
            title="derived view",
            content="agent output",
            block_time="2025-10-06",
            embedding=_vec(True),
            metadata={"source_path": "OpenAugi/Views/View - X.md"},
        ),
        # Unrelated — beyond max_distance
        Block(
            id="b6",
            kind="data_block",
            title="noise",
            content="unrelated",
            block_time="2025-10-07",
            embedding=_vec(False),
            metadata={"source_path": "Journal/2025-10-07.md"},
        ),
    ]
    s.insert_blocks(blocks)
    rows = [(b.id, b.embedding) for b in blocks]
    s.conn.executemany("INSERT INTO vec_blocks(block_id, embedding) VALUES (?, ?)", rows)
    s.conn.commit()
    return s


# ── Pure helpers ───────────────────────────────────────────────────


def test_era_of():
    assert _era_of("2025-01-20") == "2025-Q1"
    assert _era_of("2025-12-31") == "2025-Q4"


def test_era_sequence_crosses_years():
    assert _era_sequence("2024-Q4", "2025-Q2") == ["2024-Q4", "2025-Q1", "2025-Q2"]


def test_slugify():
    assert _slugify("Dopamine & focus!") == "dopamine-focus"


# ── compute_lineage ────────────────────────────────────────────────


def test_lineage_time_ordered_eras(store: SQLiteStore):
    report = compute_lineage(store, FakeEmbedder(), "the idea")
    assert report["first_mention"]["title"] == "first spark"
    assert report["last_mention"]["date"] == "2025-10-05"
    eras = [e["era"] for e in report["eras"]]
    assert eras == sorted(eras)
    assert "2024-Q4" in eras and "2025-Q4" in eras


def test_lineage_excludes_derived_and_noise(store: SQLiteStore):
    report = compute_lineage(store, FakeEmbedder(), "the idea")
    titles = {b["title"] for e in report["eras"] for b in e["blocks"]}
    assert "derived view" not in titles  # OpenAugi/ artifacts excluded
    assert "noise" not in titles  # beyond max_distance
    assert report["total_matches"] == 4


def test_lineage_flags_third_party(store: SQLiteStore):
    report = compute_lineage(store, FakeEmbedder(), "the idea")
    by_title = {b["title"]: b for e in report["eras"] for b in e["blocks"]}
    assert by_title["a book highlight"]["third_party"] is True
    assert by_title["first spark"]["third_party"] is False


def test_lineage_detects_dormant_gap(store: SQLiteStore):
    """2025-Q2 and Q3 have no matches — a 2-quarter dormancy."""
    report = compute_lineage(store, FakeEmbedder(), "the idea")
    assert report["gaps"] == [{"from": "2025-Q2", "to": "2025-Q3", "quarters": 2}]


def test_lineage_empty_store(tmp_path: Path):
    s = SQLiteStore(tmp_path / "empty.db")
    report = compute_lineage(s, FakeEmbedder(), "anything")
    assert report["total_matches"] == 0
    assert report["eras"] == []


# ── Rendering + sidecar ────────────────────────────────────────────


def test_render_markdown_smoke(store: SQLiteStore):
    md = render_lineage_markdown(compute_lineage(store, FakeEmbedder(), "the idea"))
    assert "Idea lineage" in md
    assert "2024-11-05" in md
    assert "[third-party]" in md
    assert "dormant 2025-Q2" in md


def test_write_sidecar(store: SQLiteStore, tmp_path: Path):
    report = compute_lineage(store, FakeEmbedder(), "The Idea!")
    out = write_lineage_sidecar(report, tmp_path)
    assert out == tmp_path / "OpenAugi" / "lineage" / "the-idea.json"
    loaded = json.loads(out.read_text())
    assert loaded["total_matches"] == 4
