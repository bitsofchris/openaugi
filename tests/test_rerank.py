"""Tests for pipeline.rerank — grouping, MMR, edge cases."""

from __future__ import annotations

import numpy as np

from openaugi.pipeline.rerank import (
    _group_by_similarity,
    _mmr,
    _pick_representatives,
    rerank,
)

# ── Helpers ────────────────────────────────────────────────────────


def _blob(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _unit(vec: list[float]) -> list[float]:
    a = np.array(vec, dtype=np.float32)
    return (a / np.linalg.norm(a)).tolist()


# ── rerank() public interface ───────────────────────────────────────


class TestRerankEdgeCases:
    def test_empty_candidates(self):
        assert rerank([], _blob([1.0, 0.0]), k=5) == []

    def test_fewer_candidates_than_k(self):
        query = _blob([1.0, 0.0])
        candidates = [("a", _blob([1.0, 0.0]), 1.0)]
        result = rerank(candidates, query, k=10)
        assert result == ["a"]

    def test_no_embeddings_falls_back_to_score_order(self):
        query = _blob([1.0, 0.0])
        candidates = [
            ("low", None, 0.3),
            ("high", None, 0.9),
            ("mid", None, 0.6),
        ]
        result = rerank(candidates, query, k=3)
        # No embeddings — returns first k in original order (not score-sorted)
        assert set(result) == {"low", "high", "mid"}
        assert len(result) == 3

    def test_single_candidate_returned(self):
        query = _blob([1.0, 0.0])
        candidates = [("only", _blob([1.0, 0.0]), 1.0)]
        result = rerank(candidates, query, k=5)
        assert result == ["only"]

    def test_returns_at_most_k(self):
        query = _blob(_unit([1.0, 0.0]))
        candidates = [(f"id{i}", _blob(_unit([float(i), 1.0])), 0.5) for i in range(20)]
        result = rerank(candidates, query, k=5)
        assert len(result) <= 5

    def test_no_duplicate_ids_in_result(self):
        query = _blob(_unit([1.0, 0.0]))
        candidates = [(f"id{i}", _blob(_unit([float(i % 3), 1.0])), 0.5) for i in range(12)]
        result = rerank(candidates, query, k=6)
        assert len(result) == len(set(result))

    def test_embedding_less_blocks_appended_when_needed(self):
        """If <k blocks have embeddings, embedding-less blocks fill the gap."""
        query = _blob([1.0, 0.0])
        candidates = [
            ("emb1", _blob([1.0, 0.0]), 1.0),
            ("noemb", None, 0.9),
        ]
        result = rerank(candidates, query, k=2)
        assert "emb1" in result
        assert "noemb" in result


# ── Deduplication (grouping) ────────────────────────────────────────


class TestGroupBySimilarity:
    def test_identical_vectors_group_together(self):
        embs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        groups = _group_by_similarity(embs, threshold=0.15)
        # All three are identical — should collapse to one group
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_orthogonal_vectors_stay_separate(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        groups = _group_by_similarity(embs, threshold=0.15)
        assert len(groups) == 2

    def test_near_duplicate_merges(self):
        # cos distance ≈ 0 between these
        embs = np.array(
            [
                _unit([1.0, 0.01]),
                _unit([1.0, 0.02]),
            ],
            dtype=np.float32,
        )
        groups = _group_by_similarity(embs, threshold=0.15)
        assert len(groups) == 1

    def test_threshold_controls_merge(self):
        # cos distance between [1,0] and [0.9, 0.436] ≈ 0.1
        embs = np.array(
            [
                [1.0, 0.0],
                _unit([0.9, 0.436]),
            ],
            dtype=np.float32,
        )
        strict = _group_by_similarity(embs, threshold=0.05)
        loose = _group_by_similarity(embs, threshold=0.20)
        assert len(strict) == 2
        assert len(loose) == 1

    def test_single_item(self):
        embs = np.array([[1.0, 0.0]], dtype=np.float32)
        groups = _group_by_similarity(embs, threshold=0.15)
        assert groups == [[0]]


# ── Representative selection ────────────────────────────────────────


class TestPickRepresentatives:
    def _setup(self):
        embs = np.array(
            [
                _unit([1.0, 0.0]),  # idx 0 — closest to centroid
                _unit([0.99, 0.1]),  # idx 1
                _unit([0.98, 0.2]),  # idx 2
            ],
            dtype=np.float32,
        )
        scores = np.array([0.3, 0.9, 0.5], dtype=np.float32)
        return embs, scores

    def test_centroid_strategy(self):
        embs, scores = self._setup()
        groups = [[0, 1, 2]]
        reps = _pick_representatives(groups, embs, scores, "centroid")
        assert len(reps) == 1
        # idx 1 is closest to the centroid of this tight cluster (middle vector)
        assert reps[0] == 1

    def test_score_strategy_picks_highest_score(self):
        embs, scores = self._setup()
        groups = [[0, 1, 2]]
        reps = _pick_representatives(groups, embs, scores, "score")
        assert reps[0] == 1  # highest score = 0.9

    def test_singleton_group(self):
        embs = np.array([[1.0, 0.0]], dtype=np.float32)
        scores = np.array([0.5], dtype=np.float32)
        reps = _pick_representatives([[0]], embs, scores, "centroid")
        assert reps == [0]

    def test_one_rep_per_group(self):
        embs = np.array([_unit([float(i), 1.0]) for i in range(6)], dtype=np.float32)
        scores = np.ones(6, dtype=np.float32)
        groups = [[0, 1], [2, 3], [4, 5]]
        reps = _pick_representatives(groups, embs, scores, "centroid")
        assert len(reps) == 3
        assert all(r in range(6) for r in reps)


# ── MMR diversity re-ranking ────────────────────────────────────────


class TestMMR:
    def test_returns_k_items(self):
        embs = np.array([_unit([float(i), 1.0]) for i in range(10)], dtype=np.float32)
        query = np.array(_unit([1.0, 0.0]), dtype=np.float32)
        rep_indices = list(range(10))
        result = _mmr(rep_indices, embs, query, k=5, mmr_lambda=0.5)
        assert len(result) == 5

    def test_no_duplicate_selections(self):
        embs = np.array([_unit([float(i), 1.0]) for i in range(8)], dtype=np.float32)
        query = np.array(_unit([1.0, 0.0]), dtype=np.float32)
        rep_indices = list(range(8))
        result = _mmr(rep_indices, embs, query, k=8, mmr_lambda=0.5)
        assert len(result) == len(set(result))

    def test_pure_relevance_picks_highest_similarity_first(self):
        # With mmr_lambda=1.0, MMR = pure relevance
        embs = np.array(
            [
                _unit([1.0, 0.0]),  # most similar to query
                _unit([0.5, 0.866]),  # less similar
                _unit([0.0, 1.0]),  # least similar
            ],
            dtype=np.float32,
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        rep_indices = [0, 1, 2]
        result = _mmr(rep_indices, embs, query, k=1, mmr_lambda=1.0)
        assert result == [0]  # most relevant picked first

    def test_empty_reps(self):
        embs = np.zeros((0, 2), dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        result = _mmr([], embs, query, k=5, mmr_lambda=0.5)
        assert result == []

    def test_diversity_spreads_selections(self):
        # Two clusters: {0,1} near [1,0] and {2,3} near [0,1]
        embs = np.array(
            [
                _unit([1.0, 0.01]),
                _unit([1.0, 0.02]),
                _unit([0.01, 1.0]),
                _unit([0.02, 1.0]),
            ],
            dtype=np.float32,
        )
        query = np.array(_unit([0.5, 0.5]), dtype=np.float32)
        rep_indices = [0, 1, 2, 3]
        # With high diversity weight, second pick should come from the other cluster
        result = _mmr(rep_indices, embs, query, k=2, mmr_lambda=0.3)
        assert len(result) == 2
        # First pick from one cluster, second from the other
        first_cluster = result[0] in (0, 1)
        second_cluster = result[1] in (2, 3) if first_cluster else result[1] in (0, 1)
        assert second_cluster


# ── Integration: full pipeline ──────────────────────────────────────


class TestRerankIntegration:
    def test_deduplicates_near_identical_chunks(self):
        """Near-duplicate blocks should collapse — only one representative returned."""
        query = _blob(_unit([1.0, 0.0]))
        # Three near-duplicates (all point roughly the same direction)
        near_dup = [
            ("dup1", _blob(_unit([1.0, 0.01])), 1.0),
            ("dup2", _blob(_unit([1.0, 0.02])), 0.9),
            ("dup3", _blob(_unit([1.0, 0.03])), 0.8),
            # Distinct block
            ("other", _blob(_unit([0.0, 1.0])), 0.5),
        ]
        result = rerank(near_dup, query, k=4, group_threshold=0.15)
        # The three near-dupes should collapse to one representative
        dup_hits = [r for r in result if r in ("dup1", "dup2", "dup3")]
        assert len(dup_hits) == 1

    def test_diverse_blocks_all_surface(self):
        """Clearly distinct blocks should all appear in top-k."""
        query = _blob(_unit([1.0, 0.0]))
        candidates = [
            ("a", _blob(_unit([1.0, 0.0])), 1.0),
            ("b", _blob(_unit([0.0, 1.0])), 0.9),
            ("c", _blob(_unit([-1.0, 0.0])), 0.8),
        ]
        result = rerank(candidates, query, k=3, group_threshold=0.15)
        assert set(result) == {"a", "b", "c"}

    def test_score_representative_strategy(self):
        """With representative='score', highest-score block wins within a group."""
        query = _blob(_unit([1.0, 0.0]))
        candidates = [
            ("low_score", _blob(_unit([1.0, 0.01])), 0.2),
            ("high_score", _blob(_unit([1.0, 0.02])), 0.95),
        ]
        result = rerank(candidates, query, k=2, group_threshold=0.15, representative="score")
        assert "high_score" in result
        assert "low_score" not in result
