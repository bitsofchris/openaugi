"""Retrieval deduplication + diversity re-ranking for get_context.

Pipeline (all pure numpy, no LLM calls):
  1. Candidates already over-fetched (k * overfetch_ratio) upstream.
  2. Group by embedding similarity — greedy agglomerative, average linkage.
  3. Pick one representative per group (centroid or highest score).
  4. MMR re-rank over representatives to enforce diversity.
  5. Return top-k block IDs.
"""

from __future__ import annotations

import numpy as np


def rerank(
    candidates: list[tuple[str, bytes | None, float]],
    query_embedding: bytes,
    k: int,
    group_threshold: float = 0.15,
    mmr_lambda: float = 0.5,
    representative: str = "centroid",
) -> list[str]:
    """Return up to k block_ids, deduplicated and diversity-ranked.

    Args:
        candidates: (block_id, embedding_blob, relevance_score).
            embedding_blob may be None (e.g. FTS-only blocks without embeddings).
        query_embedding: query vector as float32 bytes.
        k: max results to return.
        group_threshold: cosine distance below which two blocks merge into one group.
        mmr_lambda: 1.0 = pure relevance, 0.0 = pure diversity.
        representative: "centroid" (closest to group mean) or "score" (highest relevance).

    Returns:
        List of block_ids, length <= k.
    """
    if not candidates:
        return []

    # Partition by embedding availability
    with_emb = [(bid, blob, score) for bid, blob, score in candidates if blob is not None]
    without_emb_ids = [bid for bid, blob, _score in candidates if blob is None]

    if not with_emb:
        # No embeddings — return by relevance score order
        return [bid for bid, _, _ in candidates][:k]

    ids = [bid for bid, _, _ in with_emb]
    raw_embs = np.array(
        [np.frombuffer(blob, dtype=np.float32) for _, blob, _ in with_emb]
    )
    scores = np.array([score for _, _, score in with_emb], dtype=np.float32)

    embs_norm = _normalize_rows(raw_embs)

    query_arr = np.frombuffer(query_embedding, dtype=np.float32).copy()
    q_norm = np.linalg.norm(query_arr)
    if q_norm > 0:
        query_arr = query_arr / q_norm

    # Steps 2 + 3: group → representative indices into `ids`
    groups = _group_by_similarity(embs_norm, group_threshold)
    rep_indices = _pick_representatives(groups, embs_norm, scores, representative)

    # Step 4: MMR over representatives
    mmr_order = _mmr(rep_indices, embs_norm, query_arr, k, mmr_lambda)

    result_ids = [ids[rep_indices[j]] for j in mmr_order]

    # Append embedding-less blocks if we still have room
    if len(result_ids) < k and without_emb_ids:
        seen = set(result_ids)
        for bid in without_emb_ids:
            if bid not in seen:
                result_ids.append(bid)
                if len(result_ids) >= k:
                    break

    return result_ids[:k]


# ── Internal helpers ────────────────────────────────────────────────


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Rows with zero norm are left as-is."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _group_by_similarity(
    embs_norm: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Greedy agglomerative grouping using centroid comparison.

    Each new item is assigned to the nearest existing group whose centroid
    is within `threshold` cosine distance; otherwise it starts a new group.
    Group centroids are updated (and re-normalised) after each merge.

    Returns: list of groups, each group is a list of original row indices.
    """
    groups: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for i in range(len(embs_norm)):
        emb = embs_norm[i]
        best_g = -1
        best_dist = threshold  # must be strictly less than threshold to merge

        for g_idx, centroid in enumerate(centroids):
            dist = 1.0 - float(np.dot(emb, centroid))
            if dist < best_dist:
                best_dist = dist
                best_g = g_idx

        if best_g == -1:
            groups.append([i])
            centroids.append(emb.copy())
        else:
            groups[best_g].append(i)
            # Recompute centroid as mean of all members, then renormalise
            member_embs = embs_norm[groups[best_g]]
            centroid = member_embs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            centroids[best_g] = centroid / c_norm if c_norm > 0 else centroid

    return groups


def _pick_representatives(
    groups: list[list[int]],
    embs_norm: np.ndarray,
    scores: np.ndarray,
    strategy: str,
) -> list[int]:
    """Return one representative index per group."""
    reps: list[int] = []
    for group in groups:
        if len(group) == 1:
            reps.append(group[0])
            continue

        if strategy == "score":
            reps.append(max(group, key=lambda i: scores[i]))
        else:  # centroid (default)
            group_embs = embs_norm[group]
            centroid = group_embs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            dists = 1.0 - (group_embs @ centroid)
            reps.append(group[int(np.argmin(dists))])

    return reps


def _mmr(
    rep_indices: list[int],
    embs_norm: np.ndarray,
    query_arr: np.ndarray,
    k: int,
    mmr_lambda: float,
) -> list[int]:
    """Maximal Marginal Relevance over a set of representative indices.

    Returns: ordered list of local positions into rep_indices (length <= k).
    """
    if not rep_indices:
        return []

    # Similarity of each representative to the query
    rep_embs = embs_norm[rep_indices]  # (n_reps, dim)
    query_sims = rep_embs @ query_arr  # (n_reps,)

    selected: list[int] = []
    remaining = list(range(len(rep_indices)))

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda j: query_sims[j])
        else:
            selected_embs = embs_norm[[rep_indices[s] for s in selected]]
            best = -1
            best_score = -np.inf
            for j in remaining:
                emb = embs_norm[rep_indices[j]]
                relevance = float(query_sims[j])
                max_sim = float(np.max(selected_embs @ emb))
                score = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim
                if score > best_score:
                    best_score = score
                    best = j

        selected.append(best)
        remaining.remove(best)

    return selected
