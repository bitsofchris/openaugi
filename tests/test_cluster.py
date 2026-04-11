"""Tests for the clustering pipeline.

Uses synthetic embeddings (3 tight clusters in 8D space) stored in a
tmp SQLite DB. Tests plumbing and correctness without touching real vault data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hdbscan", reason="hdbscan not installed; skipping cluster tests")

from openaugi.model.block import Block
from openaugi.pipeline.cluster import (
    ClusterPassConfig,
    _topological_sort,
    load_embeddings,
    parse_cluster_passes,
    run_cluster_dag,
    truncate_and_normalize,
)
from openaugi.store.sqlite import SQLiteStore

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path: Path) -> SQLiteStore:
    """Fresh SQLite store in a tmp directory."""
    return SQLiteStore(tmp_path / "test.db")


def _make_embedding(center: np.ndarray, n: int, noise: float = 0.05) -> list[np.ndarray]:
    """n vectors tightly clustered around center (normalized)."""
    rng = np.random.default_rng(42)
    vecs = []
    for _ in range(n):
        v = center + rng.normal(0, noise, size=center.shape)
        v = v / np.linalg.norm(v)
        vecs.append(v.astype(np.float32))
    return vecs


@pytest.fixture
def store_with_embeddings(tmp_store: SQLiteStore) -> SQLiteStore:
    """Store with 3 tight clusters of 20 data_blocks each (8D embeddings)."""
    # Three well-separated centers in 8D
    centers = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    ]

    blocks = []
    for cluster_idx, center in enumerate(centers):
        vecs = _make_embedding(center, 20)
        for i, vec in enumerate(vecs):
            bid = f"b{cluster_idx}_{i}"
            blocks.append(
                Block(
                    id=bid,
                    kind="data_block",
                    content=f"Block {i} in cluster {cluster_idx}",
                    title=f"Note {cluster_idx}",
                    source="test",
                    block_time=f"2024-0{cluster_idx + 1}-01",
                    embedding=vec.tobytes(),
                    content_hash=bid,
                )
            )

    tmp_store.insert_blocks(blocks)
    return tmp_store


# ── Config parsing ─────────────────────────────────────────────────


def test_parse_cluster_passes_empty():
    passes = parse_cluster_passes({})
    assert passes == []


def test_parse_cluster_passes_valid():
    config = {
        "clustering": {
            "passes": [
                {"id": "coarse", "dims": 4, "scope": "all", "min_cluster_size": 5},
                {
                    "id": "fine",
                    "dims": 8,
                    "scope": "within",
                    "parent_pass": "coarse",
                    "min_cluster_size": 3,
                },
            ]
        }
    }
    passes = parse_cluster_passes(config)
    assert len(passes) == 2
    assert passes[0].id == "coarse"
    assert passes[1].parent_pass == "coarse"


def test_parse_cluster_passes_unknown_parent():
    config = {
        "clustering": {
            "passes": [
                {
                    "id": "fine",
                    "dims": 8,
                    "scope": "within",
                    "parent_pass": "nonexistent",
                    "min_cluster_size": 5,
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="unknown parent_pass"):
        parse_cluster_passes(config)


def test_pass_config_within_requires_parent():
    with pytest.raises(ValueError, match="requires parent_pass"):
        ClusterPassConfig(id="x", dims=8, scope="within", min_cluster_size=5)


# ── Topological sort ───────────────────────────────────────────────


def test_topological_sort_independent():
    a = ClusterPassConfig(id="a", dims=4, scope="all", min_cluster_size=5)
    b = ClusterPassConfig(id="b", dims=8, scope="all", min_cluster_size=5)
    order = _topological_sort([b, a])
    # Both independent — order may vary but both present
    assert {p.id for p in order} == {"a", "b"}


def test_topological_sort_parent_before_child():
    parent = ClusterPassConfig(id="parent", dims=4, scope="all", min_cluster_size=5)
    child = ClusterPassConfig(
        id="child", dims=8, scope="within", parent_pass="parent", min_cluster_size=3
    )
    order = _topological_sort([child, parent])
    ids = [p.id for p in order]
    assert ids.index("parent") < ids.index("child")


# ── load_embeddings + truncate_and_normalize ───────────────────────


def test_load_embeddings(store_with_embeddings: SQLiteStore):
    vecs = load_embeddings(store_with_embeddings)
    assert len(vecs) == 60  # 3 clusters × 20 blocks
    for vec in vecs.values():
        assert vec.dtype == np.float32
        assert vec.shape == (8,)


def test_truncate_and_normalize():
    vecs = {"a": np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)}
    out = truncate_and_normalize(vecs, dims=2)
    assert out["a"].shape == (2,)
    norm = np.linalg.norm(out["a"])
    assert abs(norm - 1.0) < 1e-5


# ── Store methods ──────────────────────────────────────────────────


def test_delete_cluster_blocks_by_pass(tmp_store: SQLiteStore):
    b = Block(
        id="cb1",
        kind="context_block:cluster",
        source="test",
        content_hash="cb1",
        metadata={"pass_id": "my_pass"},
    )
    tmp_store.insert_block(b)
    assert tmp_store.get_block("cb1") is not None

    deleted = tmp_store.delete_cluster_blocks_by_pass("my_pass")
    assert deleted == 1
    assert tmp_store.get_block("cb1") is None


def test_delete_cluster_blocks_only_matching_pass(tmp_store: SQLiteStore):
    for i, pid in enumerate(["pass_a", "pass_b"]):
        tmp_store.insert_block(
            Block(
                id=f"cb{i}",
                kind="context_block:cluster",
                source="test",
                content_hash=f"cb{i}",
                metadata={"pass_id": pid},
            )
        )
    tmp_store.delete_cluster_blocks_by_pass("pass_a")
    assert tmp_store.get_block("cb0") is None
    assert tmp_store.get_block("cb1") is not None


def test_batch_update_cluster_assignments(tmp_store: SQLiteStore):
    b = Block(id="d1", kind="data_block", content="x", source="test", content_hash="d1")
    tmp_store.insert_block(b)
    tmp_store.batch_update_cluster_assignments("life_areas", [("d1", "3")])

    updated = tmp_store.get_block("d1")
    assert updated is not None
    assignments = updated.metadata.get("cluster_assignments", {})
    assert assignments.get("life_areas") == "3"


def test_batch_update_cluster_assignments_preserves_other_metadata(tmp_store: SQLiteStore):
    b = Block(
        id="d1",
        kind="data_block",
        content="x",
        source="test",
        content_hash="d1",
        metadata={"existing_key": "existing_value"},
    )
    tmp_store.insert_block(b)
    tmp_store.batch_update_cluster_assignments("pass_a", [("d1", "5")])

    updated = tmp_store.get_block("d1")
    assert updated is not None
    assert updated.metadata.get("existing_key") == "existing_value"
    assert updated.metadata["cluster_assignments"]["pass_a"] == "5"


def test_batch_update_cluster_assignments_multiple_passes(tmp_store: SQLiteStore):
    b = Block(id="d1", kind="data_block", content="x", source="test", content_hash="d1")
    tmp_store.insert_block(b)
    tmp_store.batch_update_cluster_assignments("pass_a", [("d1", "1")])
    tmp_store.batch_update_cluster_assignments("pass_b", [("d1", "7")])

    updated = tmp_store.get_block("d1")
    assert updated is not None
    assignments = updated.metadata["cluster_assignments"]
    assert assignments["pass_a"] == "1"
    assert assignments["pass_b"] == "7"


# ── Full DAG run ───────────────────────────────────────────────────


def _single_pass_config(dims: int = 4, min_cluster_size: int = 5) -> ClusterPassConfig:
    return ClusterPassConfig(
        id="test_pass",
        dims=dims,
        scope="all",
        min_cluster_size=min_cluster_size,
        store_centroid=True,
    )


def test_run_cluster_dag_dry_run(store_with_embeddings: SQLiteStore):
    """dry_run=True prints stats but writes no cluster blocks."""
    cfg = _single_pass_config(dims=4, min_cluster_size=5)
    run_cluster_dag(store_with_embeddings, [cfg], dry_run=True)
    # No cluster blocks written
    clusters = store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
    assert len(clusters) == 0


def test_run_cluster_dag_writes_cluster_blocks(store_with_embeddings: SQLiteStore):
    """Clustering 3 well-separated clusters should produce 3 context_block:cluster blocks."""
    cfg = _single_pass_config(dims=4, min_cluster_size=5)
    results = run_cluster_dag(store_with_embeddings, [cfg], dry_run=False)

    assert "test_pass" in results
    clusters = store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
    assert len(clusters) == 3

    for cb in clusters:
        assert cb.metadata["pass_id"] == "test_pass"
        assert cb.metadata["member_count"] > 0


def test_run_cluster_dag_writes_groups_links(store_with_embeddings: SQLiteStore):
    """Every cluster block must have groups links to its members."""
    cfg = _single_pass_config(dims=4, min_cluster_size=5)
    run_cluster_dag(store_with_embeddings, [cfg], dry_run=False)

    clusters = store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
    total_members = 0
    for cb in clusters:
        links = store_with_embeddings.get_links_from(cb.id, kind="groups")
        assert len(links) > 0
        total_members += len(links)

    # All non-noise blocks should be linked; 60 total, some may be noise
    assert total_members <= 60
    assert total_members > 0


def test_run_cluster_dag_updates_data_block_metadata(store_with_embeddings: SQLiteStore):
    """data_blocks get cluster_assignments updated after clustering."""
    cfg = _single_pass_config(dims=4, min_cluster_size=5)
    run_cluster_dag(store_with_embeddings, [cfg], dry_run=False)

    # Pick any data_block that's a cluster member
    clusters = store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
    first_cluster = clusters[0]
    links = store_with_embeddings.get_links_from(first_cluster.id, kind="groups")
    assert links

    member = store_with_embeddings.get_block(links[0].to_id)
    assert member is not None
    assert "cluster_assignments" in member.metadata
    assert "test_pass" in member.metadata["cluster_assignments"]


def test_run_cluster_dag_idempotent(store_with_embeddings: SQLiteStore):
    """Re-running same pass replaces old cluster blocks — count stays the same."""
    cfg = _single_pass_config(dims=4, min_cluster_size=5)
    run_cluster_dag(store_with_embeddings, [cfg], dry_run=False)
    count_first = len(
        store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
    )
    run_cluster_dag(store_with_embeddings, [cfg], dry_run=False)
    count_second = len(
        store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
    )
    assert count_first == count_second


def test_run_cluster_dag_two_pass_hierarchy(store_with_embeddings: SQLiteStore):
    """scope=within pass creates sub-clusters linked to parent clusters via contains."""
    coarse = ClusterPassConfig(id="coarse", dims=4, scope="all", min_cluster_size=5)
    fine = ClusterPassConfig(
        id="fine", dims=8, scope="within", parent_pass="coarse", min_cluster_size=3
    )
    run_cluster_dag(store_with_embeddings, [coarse, fine], dry_run=False)

    coarse_clusters = [
        b
        for b in store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
        if b.metadata.get("pass_id") == "coarse"
    ]
    fine_clusters = [
        b
        for b in store_with_embeddings.get_blocks_by_kind("context_block:cluster", limit=1000)
        if b.metadata.get("pass_id") == "fine"
    ]

    assert len(coarse_clusters) >= 1
    # Fine pass may or may not produce sub-clusters depending on data density
    # but contains links should exist if fine clusters were created
    for fc in fine_clusters:
        links = store_with_embeddings.get_links_to(fc.id, kind="contains")
        assert len(links) == 1, (
            f"Fine cluster {fc.id} should have exactly one parent contains link"
        )
