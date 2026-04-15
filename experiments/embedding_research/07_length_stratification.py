#!/usr/bin/env python3
"""Phase 7: Length stratification — does block length confound cluster quality?

Hypothesis: long blocks span multiple ideas, producing "average" embeddings
that sit between clusters rather than inside one. If true, cluster quality
metrics will be systematically better for short blocks at every dimension —
meaning poor clustering of long blocks is a chunking problem, not a dimension
problem.

Method:
  Split all data_blocks into tertiles by content length:
    short  — bottom third  (~50–200 chars)
    medium — middle third  (~200–450 chars)
    long   — top third     (~450+ chars)

  For each tertile × dim combination, run k-means (k=5,8,12) and measure:
    - ARI stability (5 runs)
    - Silhouette score
    - Mean pairwise cosine similarity within clusters

  Compare: if short >>> long at every dim, chunking is the confound.
  If they converge at some dim, that dim resolves length-induced noise.

Outputs:
  output/length_results.json
  output/length_ari_comparison.png   — ARI per tertile vs dim
  output/length_distributions.png    — block length distribution + tertile cuts

Usage:
    python 07_length_stratification.py
    python 07_length_stratification.py --db ~/.openaugi/openaugi.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

DEFAULT_DB = Path.home() / ".openaugi" / "openaugi.db"
OUT_DIR = Path(__file__).parent / "output"

DIMS = [32, 64, 128, 256, 512, 768, 1536, 3072]
K_VALUES = [5, 8, 12]
ARI_RUNS = 5
N_PER_TERTILE = 300  # cap per tertile — keeps runtime predictable


# ── DB ────────────────────────────────────────────────────────────────────────


def load_blocks_with_length(db_path: Path, n_per_tertile: int) -> dict[str, np.ndarray]:
    """Load all data_blocks, split into length tertiles, return matrices."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, content, embedding, length(content) as char_len
        FROM blocks
        WHERE kind = 'data_block'
          AND embedding IS NOT NULL
          AND length(content) > 50
        ORDER BY char_len
        """
    ).fetchall()
    conn.close()

    lengths = [r["char_len"] for r in rows]
    t33 = int(np.percentile(lengths, 33))
    t67 = int(np.percentile(lengths, 67))

    buckets: dict[str, list] = {"short": [], "medium": [], "long": []}
    for row in rows:
        cl = row["char_len"]
        if cl <= t33:
            buckets["short"].append(row)
        elif cl <= t67:
            buckets["medium"].append(row)
        else:
            buckets["long"].append(row)

    print(f"  Length tertile cuts: short ≤{t33}  medium ≤{t67}  long >{t67}")
    print(
        "  Raw counts: "
        f"short={len(buckets['short'])}  "
        f"medium={len(buckets['medium'])}  "
        f"long={len(buckets['long'])}"
    )

    rng = np.random.default_rng(42)
    result = {}
    for name, bucket_rows in buckets.items():
        sampled = rng.choice(
            len(bucket_rows), size=min(n_per_tertile, len(bucket_rows)), replace=False
        )
        vecs = [
            np.frombuffer(bucket_rows[int(i)]["embedding"], dtype=np.float32).copy()
            for i in sampled
        ]
        result[name] = np.stack(vecs)
        print(f"  Sampled {name}: {result[name].shape[0]} blocks")

    return result, t33, t67


# ── Math ──────────────────────────────────────────────────────────────────────


def truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    t = matrix[:, :dims].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norms == 0, 1.0, norms)


def cluster_stability(X: np.ndarray, k: int) -> dict:
    if len(X) < k * 2:
        return {"ari": 0.0, "silhouette": 0.0}
    runs = [
        KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=s).fit_predict(X)
        for s in range(ARI_RUNS)
    ]
    pairs = list(combinations(range(ARI_RUNS), 2))
    ari = float(np.mean([adjusted_rand_score(runs[a], runs[b]) for a, b in pairs]))
    sil = float(silhouette_score(X, runs[0], metric="cosine")) if len(set(runs[0])) > 1 else 0.0
    return {"ari": ari, "silhouette": sil}


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_ari_comparison(results: dict, out_path: Path) -> None:
    """ARI vs dim, one line per tertile, one panel per k."""
    dims_list = DIMS
    tertiles = ["short", "medium", "long"]
    colors = {"short": "#16a34a", "medium": "#2563eb", "long": "#dc2626"}

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        "Cluster stability (ARI) by block length tertile vs. dimension\n"
        "If short >> long at every dim → chunking is the confound, not dimension choice",
        fontsize=11,
        y=1.02,
    )

    for ax, k in zip(axes, K_VALUES, strict=False):
        for tertile in tertiles:
            aris = [
                next(r["ari"] for r in results[tertile] if r["dims"] == d and r["k"] == k)
                for d in dims_list
            ]
            ax.plot(dims_list, aris, "o-", color=colors[tertile], lw=2, ms=5, label=tertile)

        ax.set_xscale("log", base=2)
        ax.set_xticks(dims_list)
        ax.set_xticklabels([str(d) for d in dims_list], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"k={k}", fontsize=11)
        ax.set_xlabel("Dims (log₂)", fontsize=10)
        ax.grid(True, alpha=0.25, ls="--")
        ax.set_ylim(0, 1.0)
        if ax == axes[0]:
            ax.set_ylabel(f"Mean pairwise ARI ({ARI_RUNS} runs)", fontsize=10)
            ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def plot_distributions(db_path: Path, cuts: tuple[int, int], out_path: Path) -> None:
    """Histogram of block lengths with tertile cut lines."""
    conn = sqlite3.connect(str(db_path))
    lengths = [
        r[0]
        for r in conn.execute(
            "SELECT length(content) FROM blocks WHERE kind='data_block' AND length(content) > 50"
        ).fetchall()
    ]
    conn.close()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lengths, bins=80, color="#6366f1", alpha=0.7, edgecolor="none")
    ax.axvline(cuts[0], color="#16a34a", ls="--", lw=1.5, label=f"short/med cut ({cuts[0]})")
    ax.axvline(cuts[1], color="#dc2626", ls="--", lw=1.5, label=f"med/long cut ({cuts[1]})")
    ax.set_xlim(0, min(2000, max(lengths)))
    ax.set_xlabel("Block length (chars)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Block length distribution with tertile cuts", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)

    print(f"Loading blocks from {db_path}...")
    tertile_matrices, t33, t67 = load_blocks_with_length(db_path, N_PER_TERTILE)

    total = len(tertile_matrices) * len(DIMS) * len(K_VALUES)
    print(f"\nRunning {total} combinations (single-threaded)...\n")
    print(f"  {'tertile':<8}  {'dims':>5}  {'k':>3}  {'ari':>6}  {'sil':>6}")
    print("  " + "-" * 38)

    results: dict[str, list[dict]] = {t: [] for t in tertile_matrices}
    done = 0
    for tertile, full_matrix in tertile_matrices.items():
        for dims in DIMS:
            X = truncate_normalize(full_matrix, dims)
            for k in K_VALUES:
                metrics = cluster_stability(X, k)
                results[tertile].append({"dims": dims, "k": k, **metrics})
                done += 1
                print(
                    f"  {tertile:<8}  {dims:>5}d  {k:>3}  "
                    f"{metrics['ari']:>6.3f}  {metrics['silhouette']:>6.3f}"
                    f"  [{done}/{total}]"
                )
                time.sleep(0.1)

    # Save
    results_path = out_dir / "length_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "tertile_cuts": {"short_max": t33, "medium_max": t67},
                "n_per_tertile": N_PER_TERTILE,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results → {results_path.name}")

    plot_distributions(db_path, (t33, t67), out_dir / "length_distributions.png")
    plot_ari_comparison(results, out_dir / "length_ari_comparison.png")

    # Summary: mean ARI per tertile across all dims+k
    print("\nMean ARI per tertile (all dims, all k):")
    for tertile, rows in results.items():
        mean_ari = float(np.mean([r["ari"] for r in rows]))
        print(f"  {tertile:<8}  {mean_ari:.3f}")

    print("\nIf short >> long: chunking is the confound.")
    print("If similar: dimension is the lever, not chunk size.")


if __name__ == "__main__":
    main()
