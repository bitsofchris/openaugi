"""Tests for cluster weather — snapshots, cross-run matching, weather report.

Uses synthetic snapshots and a tmp SQLite store; no real vault data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hdbscan", reason="hdbscan not installed; skipping cluster tests")

from openaugi.model.block import Block
from openaugi.pipeline.cluster import ClusterPassConfig, PassResult, run_cluster_dag
from openaugi.pipeline.cluster_weather import (
    _jaccard,
    _match_clusters,
    compute_weather,
    load_snapshots,
    render_weather_markdown,
    snapshot_cluster_run,
)
from openaugi.store.sqlite import SQLiteStore

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "test.db")


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _fake_pass_result(pass_id: str, clusters: dict[int, list[str]]) -> PassResult:
    """Build a PassResult from {label: member_ids} without running clustering."""
    from openaugi.pipeline.cluster import ClusterResult

    block_ids = [bid for members in clusters.values() for bid in members]
    labels = np.array([lbl for lbl, members in clusters.items() for _ in members], dtype=np.int64)
    cfg = ClusterPassConfig(id=pass_id, dims=8, scope="all", type="kmeans", n_clusters=2)
    return PassResult(
        pass_cfg=cfg,
        results=[
            ClusterResult(
                pass_id=pass_id,
                parent_cluster_label=None,
                block_ids=block_ids,
                labels=labels,
                centroids={},
            )
        ],
        cluster_block_ids={},
    )


def _snapshot_at(store: SQLiteStore, days_ago: int, clusters: dict[int, list[str]]) -> str:
    run_at = _iso(datetime.now(UTC) - timedelta(days=days_ago))
    return snapshot_cluster_run(
        store, {"concepts": _fake_pass_result("concepts", clusters)}, run_at=run_at
    )


# ── Matching primitives ────────────────────────────────────────────


def test_jaccard():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"c"}) == 0.0
    assert _jaccard({"a", "b", "c"}, {"a", "b", "d"}) == pytest.approx(0.5)
    assert _jaccard(set(), set()) == 0.0


def test_match_clusters_same_members_matches():
    cur = [{"label": "0", "title": "p_0", "member_count": 3, "members": ["a", "b", "c"]}]
    base = [{"label": "7", "title": "p_7", "member_count": 3, "members": ["a", "b", "c"]}]
    matches = _match_clusters(cur, base)
    assert len(matches) == 1
    assert matches[0][0] is cur[0]
    assert matches[0][1] is base[0]


def test_match_clusters_label_drift_still_matches():
    """K-means labels move between runs — matching is by membership, not label."""
    cur = [
        {"label": "0", "title": "p_0", "member_count": 4, "members": ["a", "b", "c", "d"]},
        {"label": "1", "title": "p_1", "member_count": 2, "members": ["x", "y"]},
    ]
    base = [
        {"label": "1", "title": "p_1", "member_count": 3, "members": ["a", "b", "c"]},
        {"label": "0", "title": "p_0", "member_count": 2, "members": ["x", "y"]},
    ]
    matched = {(c["title"], b["title"]) for c, b in _match_clusters(cur, base) if c and b}
    assert ("p_0", "p_1") in matched
    assert ("p_1", "p_0") in matched


def test_match_clusters_by_centroid_when_membership_churned():
    """K-means reshuffles can drop Jaccard below threshold for the same cluster
    (found in the wild 2026-07-08) — a near-identical centroid still matches."""
    from openaugi.pipeline.cluster_weather import _encode_centroid

    c1 = _encode_centroid(np.array([1.0, 0.05, 0.0, 0.0], dtype=np.float32))
    c2 = _encode_centroid(np.array([1.0, 0.0, 0.05, 0.0], dtype=np.float32))
    # Only 1 of 4 members overlap → jaccard 1/7 ≈ 0.14, containment 0.25
    cur = [
        {
            "label": "0",
            "title": "p_0",
            "member_count": 4,
            "members": ["a", "x", "y", "z"],
            "centroid": c1,
        }
    ]
    base = [
        {
            "label": "3",
            "title": "p_3",
            "member_count": 4,
            "members": ["a", "b", "c", "d"],
            "centroid": c2,
        }
    ]
    matches = _match_clusters(cur, base)
    assert (cur[0], base[0]) in matches


def test_match_clusters_centroid_below_threshold_stays_born_died():
    from openaugi.pipeline.cluster_weather import _encode_centroid

    c1 = _encode_centroid(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    c2 = _encode_centroid(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    cur = [
        {"label": "0", "title": "p_0", "member_count": 2, "members": ["x", "y"], "centroid": c1}
    ]
    base = [
        {"label": "0", "title": "p_0", "member_count": 2, "members": ["a", "b"], "centroid": c2}
    ]
    matched = [(c, b) for c, b in _match_clusters(cur, base) if c and b]
    assert matched == []


def test_match_clusters_missing_centroid_falls_back_to_members():
    """Snapshots written before centroids were stored still diff correctly."""
    cur = [{"label": "0", "title": "p_0", "member_count": 3, "members": ["a", "b", "c"]}]
    base = [{"label": "1", "title": "p_1", "member_count": 3, "members": ["a", "b", "c"]}]
    matched = [(c, b) for c, b in _match_clusters(cur, base) if c and b]
    assert len(matched) == 1


def test_snapshot_stores_centroid(tmp_store: SQLiteStore):
    from openaugi.pipeline.cluster import ClusterResult
    from openaugi.pipeline.cluster_weather import _centroid_cosine

    cfg = ClusterPassConfig(id="p", dims=8, scope="all", type="kmeans", n_clusters=1)
    centroid = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pr = PassResult(
        pass_cfg=cfg,
        results=[
            ClusterResult(
                pass_id="p",
                parent_cluster_label=None,
                block_ids=["a", "b"],
                labels=np.array([0, 0], dtype=np.int64),
                centroids={0: centroid},
            )
        ],
        cluster_block_ids={},
    )
    snapshot_cluster_run(tmp_store, {"p": pr})
    snap = load_snapshots(tmp_store)[0]
    stored = snap["passes"]["p"][0]["centroid"]
    assert stored
    # Round-trips as a unit vector identical to itself
    assert _centroid_cosine(stored, stored) == pytest.approx(1.0)


def test_match_clusters_born_and_died():
    cur = [{"label": "0", "title": "p_0", "member_count": 2, "members": ["n1", "n2"]}]
    base = [{"label": "0", "title": "p_0", "member_count": 2, "members": ["o1", "o2"]}]
    matches = _match_clusters(cur, base)
    statuses = {(c is not None, b is not None) for c, b in matches}
    assert (True, False) in statuses  # born
    assert (False, True) in statuses  # died


# ── Snapshots ──────────────────────────────────────────────────────


def test_snapshot_writes_block_and_loads_back(tmp_store: SQLiteStore):
    _snapshot_at(tmp_store, 0, {0: ["a", "b"], 1: ["c"]})
    snaps = load_snapshots(tmp_store)
    assert len(snaps) == 1
    clusters = snaps[0]["passes"]["concepts"]
    assert {c["title"] for c in clusters} == {"concepts_0", "concepts_1"}
    assert sum(c["member_count"] for c in clusters) == 3


def test_snapshots_survive_cluster_repass(tmp_store: SQLiteStore):
    """delete_cluster_blocks_by_pass must not touch cluster_run history."""
    _snapshot_at(tmp_store, 0, {0: ["a"]})
    tmp_store.delete_cluster_blocks_by_pass("concepts")
    assert len(load_snapshots(tmp_store)) == 1


def test_snapshots_ordered_newest_first(tmp_store: SQLiteStore):
    _snapshot_at(tmp_store, 20, {0: ["a"]})
    _snapshot_at(tmp_store, 0, {0: ["a", "b"]})
    snaps = load_snapshots(tmp_store)
    assert len(snaps) == 2
    assert snaps[0]["run_at"] > snaps[1]["run_at"]


# ── Weather report ─────────────────────────────────────────────────


def test_weather_no_snapshots(tmp_store: SQLiteStore):
    report = compute_weather(tmp_store)
    assert "note" in report
    assert report["passes"] == {}


def test_weather_single_snapshot_reports_stable(tmp_store: SQLiteStore):
    _snapshot_at(tmp_store, 0, {0: ["a", "b"]})
    report = compute_weather(tmp_store, window_days=14)
    pdata = report["passes"]["concepts"]
    assert pdata["baseline_run"] is None
    assert all(c["status"] == "stable" for c in pdata["clusters"])


def test_weather_growth_detected(tmp_store: SQLiteStore):
    # Titles for new members so new_member_titles resolves
    tmp_store.insert_blocks(
        [Block(id=f"n{i}", kind="data_block", title=f"New note {i}") for i in range(3)]
    )
    _snapshot_at(tmp_store, 15, {0: ["a", "b", "c", "d"]})
    _snapshot_at(tmp_store, 0, {0: ["a", "b", "c", "d", "n0", "n1", "n2"]})
    report = compute_weather(tmp_store, window_days=14)
    pdata = report["passes"]["concepts"]
    assert pdata["baseline_run"] is not None
    grown = [c for c in pdata["clusters"] if c["status"] == "grew"]
    assert len(grown) == 1
    assert grown[0]["delta"] == 3
    assert set(grown[0]["new_member_titles"]) == {"New note 0", "New note 1", "New note 2"}


def test_weather_born_and_died(tmp_store: SQLiteStore):
    _snapshot_at(tmp_store, 15, {0: ["a", "b", "c"]})
    _snapshot_at(tmp_store, 0, {5: ["x", "y", "z"]})
    report = compute_weather(tmp_store, window_days=14)
    statuses = {c["status"] for c in report["passes"]["concepts"]["clusters"]}
    assert statuses == {"born", "died"}


def test_weather_recent_activity_from_live_cluster_blocks(tmp_store: SQLiteStore):
    """recent_activity counts block_timestamps inside the window."""
    recent = (datetime.now(UTC) - timedelta(days=2)).strftime("%Y-%m-%d")
    tmp_store.insert_blocks(
        [
            Block(
                id="cluster_live",
                kind="context_block:cluster",
                title="concepts_0",
                source="pipeline:cluster",
                metadata={
                    "pass_id": "concepts",
                    "temporal": {"block_timestamps": ["2024-01-01", recent, recent]},
                },
            )
        ]
    )
    _snapshot_at(tmp_store, 0, {0: ["a", "b"]})
    report = compute_weather(tmp_store, window_days=14)
    cluster = report["passes"]["concepts"]["clusters"][0]
    assert cluster["recent_activity"] == 2
    assert cluster["last_block"] == recent


def test_weather_baseline_prefers_full_window(tmp_store: SQLiteStore):
    """With snapshots at 30d, 5d, and now: baseline is 30d (the one past the window)."""
    _snapshot_at(tmp_store, 30, {0: ["a"]})
    _snapshot_at(tmp_store, 5, {0: ["a", "b"]})
    _snapshot_at(tmp_store, 0, {0: ["a", "b", "c"]})
    report = compute_weather(tmp_store, window_days=14)
    pdata = report["passes"]["concepts"]
    grown = [c for c in pdata["clusters"] if c["status"] == "grew"]
    assert grown and grown[0]["delta"] == 2  # vs 30d-old snapshot, not 5d


def test_render_markdown_smoke(tmp_store: SQLiteStore):
    _snapshot_at(tmp_store, 15, {0: ["a", "b"]})
    _snapshot_at(tmp_store, 0, {0: ["a", "b", "c"]})
    md = render_weather_markdown(compute_weather(tmp_store, window_days=14))
    assert "Cluster weather" in md
    assert "concepts_0" in md
    assert "+1" in md


# ── Integration: run_cluster_dag writes a snapshot ─────────────────


def _store_with_embeddings(store: SQLiteStore) -> None:
    rng = np.random.default_rng(7)
    centers = [np.eye(8, dtype=np.float32)[i] for i in range(3)]
    blocks = []
    for ci, center in enumerate(centers):
        for i in range(20):
            v = center + rng.normal(0, 0.05, size=8)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            blocks.append(
                Block(
                    id=f"b_{ci}_{i}",
                    kind="data_block",
                    title=f"note {ci}-{i}",
                    content=f"content {ci}-{i}",
                    embedding=v.tobytes(),
                )
            )
    store.insert_blocks(blocks)


def test_run_cluster_dag_records_snapshot(tmp_store: SQLiteStore):
    _store_with_embeddings(tmp_store)
    cfg = ClusterPassConfig(
        id="areas", dims=8, scope="all", type="kmeans", n_clusters=3, input_level="block"
    )
    run_cluster_dag(tmp_store, [cfg], dry_run=False)
    snaps = load_snapshots(tmp_store)
    assert len(snaps) == 1
    assert sum(c["member_count"] for c in snaps[0]["passes"]["areas"]) == 60


def test_run_cluster_dag_dry_run_no_snapshot(tmp_store: SQLiteStore):
    _store_with_embeddings(tmp_store)
    cfg = ClusterPassConfig(
        id="areas", dims=8, scope="all", type="kmeans", n_clusters=3, input_level="block"
    )
    run_cluster_dag(tmp_store, [cfg], dry_run=True)
    assert load_snapshots(tmp_store) == []
