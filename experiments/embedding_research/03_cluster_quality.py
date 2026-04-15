#!/usr/bin/env python3
"""Phase 3: Cluster quality metrics across truncation dimensions and k values.

For each (dims, k) combination runs k-means and records:
  - Silhouette score        — how well-separated clusters are (-1..1, higher better)
  - Davies-Bouldin index    — ratio of within/between scatter (lower better)
  - Calinski-Harabasz index — between/within variance ratio (higher better)
  - ARI stability           — run k-means 5x with different seeds, measure agreement

Key signal: the dimension where ARI stability plateaus = real cluster structure
emerging vs. noise. Cross-reference with Phase 2 sweet spot (128-256d).

Single-threaded throughout — no worker pools, no n_jobs parallelism.

Outputs:
  output/cluster_results.json   — raw metrics table
  output/cluster_heatmaps.png   — silhouette + ARI heatmaps (dims x k)
  output/cluster_stability.png  — ARI stability vs dims (per k)

Usage:
    python 03_cluster_quality.py
    python 03_cluster_quality.py --labeled output/labeled.json
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Smaller grid than the full probe — enough to see the shape without thrashing
DIMS = [32, 64, 128, 256, 512, 768, 1024, 1536, 3072]
K_VALUES = [3, 5, 7, 10, 15, 20]
ARI_RUNS = 5  # k-means reruns per (dims, k) for stability
OUT_DIR = Path(__file__).parent / "output"


# ── Data ──────────────────────────────────────────────────────────────────────


def load_embeddings(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    clean = [b for b in data if b.get("embedding_3072")]
    print(f"  {len(clean)} blocks loaded")
    return np.array([b["embedding_3072"] for b in clean], dtype=np.float32)


def truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    t = matrix[:, :dims].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return t / norms


# ── Metrics ───────────────────────────────────────────────────────────────────


def run_kmeans_once(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = KMeans(
        n_clusters=k,
        n_init=3,  # fewer inits per run — stability comes from ARI_RUNS
        max_iter=100,
        random_state=seed,
    )
    return km.fit_predict(X)


def cluster_metrics(X: np.ndarray, k: int) -> dict:
    """Run k-means ARI_RUNS times, compute quality metrics on the best run."""
    label_runs = [run_kmeans_once(X, k, seed=s) for s in range(ARI_RUNS)]

    # ARI stability: mean pairwise ARI across all run pairs
    pairs = list(combinations(range(ARI_RUNS), 2))
    ari_scores = [adjusted_rand_score(label_runs[a], label_runs[b]) for a, b in pairs]
    ari_mean = float(np.mean(ari_scores))

    # Use the first run for intrinsic metrics (consistent reference)
    labels = label_runs[0]

    sil = float(silhouette_score(X, labels, metric="cosine"))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "ari_stability": ari_mean,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_heatmaps(results: list[dict], out_path: Path) -> None:
    dims_list = sorted(set(r["dims"] for r in results))
    k_list = sorted(set(r["k"] for r in results))

    def make_grid(metric: str) -> np.ndarray:
        grid = np.zeros((len(k_list), len(dims_list)))
        for r in results:
            i = k_list.index(r["k"])
            j = dims_list.index(r["dims"])
            grid[i, j] = r[metric]
        return grid

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label, cmap in [
        (axes[0], "silhouette", "Silhouette score\n(higher = better)", "YlGn"),
        (
            axes[1],
            "ari_stability",
            f"ARI stability ({ARI_RUNS} runs)\n(higher = more consistent)",
            "YlOrRd",
        ),
    ]:
        grid = make_grid(metric)
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=grid.min(), vmax=grid.max())
        ax.set_xticks(range(len(dims_list)))
        ax.set_xticklabels([str(d) for d in dims_list], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(k_list)))
        ax.set_yticklabels([f"k={k}" for k in k_list], fontsize=9)
        ax.set_xlabel("Embedding dims")
        ax.set_title(label, fontsize=11, pad=10)
        for i in range(len(k_list)):
            for j in range(len(dims_list)):
                ax.text(
                    j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black"
                )
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Cluster quality metrics — k-means across Matryoshka dims", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmaps saved → {out_path.name}")


def plot_stability(results: list[dict], out_path: Path) -> None:
    dims_list = sorted(set(r["dims"] for r in results))
    k_list = sorted(set(r["k"] for r in results))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(k_list)))

    for k, color in zip(k_list, colors, strict=False):
        ari_per_dim = [
            next(r["ari_stability"] for r in results if r["dims"] == d and r["k"] == k)
            for d in dims_list
        ]
        ax.plot(
            dims_list, ari_per_dim, "o-", label=f"k={k}", color=color, linewidth=1.8, markersize=5
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims_list)
    ax.set_xticklabels([str(d) for d in dims_list], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Embedding dims (log₂ scale)", fontsize=11)
    ax.set_ylabel(
        f"ARI stability (mean of {len(list(combinations(range(ARI_RUNS), 2)))} pairs)", fontsize=10
    )
    ax.set_title(
        "Cluster stability vs. dimension — where does structure emerge?", fontsize=13, pad=14
    )
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stability plot saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labeled", default=str(OUT_DIR / "labeled.json"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    labeled_path = Path(args.labeled)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labeled_path.exists():
        print(f"ERROR: {labeled_path} not found — run 01_sample_and_label.py first")
        raise SystemExit(1)

    print(f"Loading embeddings from {labeled_path}...")
    embeddings = load_embeddings(labeled_path)

    total = len(DIMS) * len(K_VALUES)
    print(
        f"\nRunning {total} (dims × k) combinations — single-threaded, {ARI_RUNS} runs each...\n"
    )
    print(f"  {'dims':>5}  {'k':>3}  {'silhouette':>10}  {'davies_bouldin':>14}")
    print(f"  {'':>5}  {'':>3}  {'':>10}  {'':>14}  {'ARI stability':>13}")
    print("  " + "-" * 58)

    results = []
    n = 0
    for dims in DIMS:
        X = truncate_normalize(embeddings, dims)
        for k in K_VALUES:
            n += 1
            metrics = cluster_metrics(X, k)
            row = {"dims": dims, "k": k, **metrics}
            results.append(row)
            print(
                f"  {dims:>5}d  {k:>3}  "
                f"{metrics['silhouette']:>10.3f}  "
                f"{metrics['davies_bouldin']:>14.3f}  "
                f"{metrics['ari_stability']:>13.3f}  "
                f"  [{n}/{total}]"
            )
            time.sleep(0.3)  # breathe between fits

    # Save
    results_path = out_dir / "cluster_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {"dims_tested": DIMS, "k_tested": K_VALUES, "ari_runs": ARI_RUNS, "results": results},
            f,
            indent=2,
        )
    print(f"\n  Results saved → {results_path.name}")

    plot_heatmaps(results, out_dir / "cluster_heatmaps.png")
    plot_stability(results, out_dir / "cluster_stability.png")

    # Summary: best silhouette per dims
    print("\nBest silhouette per dimension (across k values):")
    for dims in DIMS:
        dim_results = [r for r in results if r["dims"] == dims]
        best = max(dim_results, key=lambda r: r["silhouette"])
        print(f"  {dims:>5}d  k={best['k']}  sil={best['silhouette']:.3f}")
        print(f"           ari={best['ari_stability']:.3f}")

    print("\nNext: run 04_visualize.py")


if __name__ == "__main__":
    main()
