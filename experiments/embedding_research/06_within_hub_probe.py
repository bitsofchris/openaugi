#!/usr/bin/env python3
"""Phase 6: Within-hub sub-cluster stability — the idea-level sweet spot.

The Phase 5 hubs were too broad (parenting, Jung, health) — easy to separate
at any dimension. This script asks the harder question: within a single hub,
at what dimension does the embedding best separate *fine ideas*?

Method:
  For each hub independently:
    For each truncation dim:
      Run k-means (k=3,5,7) within just that hub's blocks, N times.
      Measure ARI stability across runs.
      Measure silhouette score.

The dimension where within-hub ARI plateaus = the idea-level sweet spot.
Because coarse signal is removed (all blocks are from the same hub), only
fine structure remains. This is also the practical retrieval scenario —
you'd already be scoped to a topic before doing similarity search.

Outputs:
  output/within_hub_results.json
  output/within_hub_stability.png   — ARI per hub per dim (grid of small plots)
  output/within_hub_summary.png     — mean ARI across hubs vs dim

Usage:
    python 06_within_hub_probe.py
    python 06_within_hub_probe.py --db ~/.openaugi/openaugi.db
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

SELECTED_HUBS = [
    ("4431923b0c175b67", "parenting"),
    ("ed8926f2d3c1712d", "jung_self"),
    ("a0cec5e1c943e556", "niche_knowledge"),
    ("71d5cd32bf327ef2", "physical_health"),
    ("e69205dbec0bf2c1", "embedding_research"),
    ("e830e953b10d6a86", "priorities_vision"),
    ("8629784a8dcf5bf8", "victory_feedback"),
    ("982ec76c691201c1", "overthinking"),
]

DIMS = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 3072]
K_VALUES = [3, 5, 7]
ARI_RUNS = 6


# ── DB ────────────────────────────────────────────────────────────────────────


def load_hub_blocks(db_path: Path, hubs: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Return {hub_label: embeddings_matrix} — one matrix per hub."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    hub_ids = [h[0] for h in hubs]
    id_to_label = {h[0]: h[1] for h in hubs}
    placeholders = ",".join("?" * len(hub_ids))

    rows = conn.execute(
        f"""
        SELECT DISTINCT b.id, b.embedding, l.to_id AS hub_id
        FROM links l
        JOIN blocks b ON l.from_id = b.id
        WHERE l.kind = 'links_to'
          AND l.to_id IN ({placeholders})
          AND b.kind = 'data_block'
          AND b.embedding IS NOT NULL
          AND length(b.content) > 50
        """,
        hub_ids,
    ).fetchall()
    conn.close()

    # Group by hub, deduplicate blocks that appear in multiple hubs
    seen_blocks: set[str] = set()
    hub_vecs: dict[str, list[np.ndarray]] = {lbl: [] for _, lbl in hubs}
    for row in rows:
        bid = row["id"]
        if bid in seen_blocks:
            continue
        seen_blocks.add(bid)
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        hub_vecs[id_to_label[row["hub_id"]]].append(emb)

    return {lbl: np.stack(vecs) for lbl, vecs in hub_vecs.items() if vecs}


# ── Math ──────────────────────────────────────────────────────────────────────


def truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    t = matrix[:, :dims].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norms == 0, 1.0, norms)


def within_hub_metrics(X: np.ndarray, k: int) -> dict:
    """Run k-means ARI_RUNS times, return mean pairwise ARI + silhouette."""
    if len(X) < k * 2:
        return {"ari": 0.0, "silhouette": 0.0, "n": len(X)}

    runs = [
        KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=s).fit_predict(X)
        for s in range(ARI_RUNS)
    ]
    pairs = list(combinations(range(ARI_RUNS), 2))
    ari = float(np.mean([adjusted_rand_score(runs[a], runs[b]) for a, b in pairs]))

    sil = float(silhouette_score(X, runs[0], metric="cosine")) if len(set(runs[0])) > 1 else 0.0
    return {"ari": ari, "silhouette": sil, "n": len(X)}


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_hub_grid(all_results: dict, out_path: Path) -> None:
    """Small-multiple ARI plots — one panel per hub."""
    hubs = list(all_results.keys())
    n_hubs = len(hubs)
    cols = 4
    rows = (n_hubs + cols - 1) // cols

    colors = {"3": "#2563eb", "5": "#dc2626", "7": "#16a34a"}
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.2))
    axes = axes.flatten()

    for i, hub in enumerate(hubs):
        ax = axes[i]
        hub_data = all_results[hub]
        dims_list = sorted(set(r["dims"] for r in hub_data))
        n_blocks = hub_data[0]["n"] if hub_data else 0

        for k in K_VALUES:
            aris = [
                next(r["ari"] for r in hub_data if r["dims"] == d and r["k"] == k)
                for d in dims_list
            ]
            ax.plot(dims_list, aris, "o-", color=colors[str(k)], lw=1.5, ms=3.5, label=f"k={k}")

        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 128, 512, 3072])
        ax.set_xticklabels(["32", "128", "512", "3k"], fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{hub}\n(n={n_blocks})", fontsize=8, pad=4)
        ax.grid(True, alpha=0.2, ls="--")
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Within-hub cluster stability (ARI) vs. dim — idea-level sweet spot",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def plot_summary(all_results: dict, out_path: Path) -> None:
    """Mean ARI across all hubs vs dim, one line per k."""
    hubs = list(all_results.keys())
    dims_list = sorted(set(r["dims"] for r in next(iter(all_results.values()))))
    colors = {"3": "#2563eb", "5": "#dc2626", "7": "#16a34a"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for k in K_VALUES:
        mean_aris = []
        for d in dims_list:
            hub_aris = [
                next(r["ari"] for r in all_results[hub] if r["dims"] == d and r["k"] == k)
                for hub in hubs
            ]
            mean_aris.append(float(np.mean(hub_aris)))
        ax.plot(dims_list, mean_aris, "o-", color=colors[str(k)], lw=2, ms=6, label=f"k={k}")

        # Mark peak
        peak_i = int(np.argmax(mean_aris))
        ax.annotate(
            f"{dims_list[peak_i]}d",
            xy=(dims_list[peak_i], mean_aris[peak_i]),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color=colors[str(k)],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims_list)
    ax.set_xticklabels([str(d) for d in dims_list], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Embedding dims (log₂ scale)", fontsize=11)
    ax.set_ylabel(f"Mean within-hub ARI ({ARI_RUNS} runs)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Within-hub cluster stability — mean across all hubs\n"
        "Peak = dimension where fine idea structure emerges",
        fontsize=12,
        pad=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, ls="--")
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

    print(f"Loading hub blocks from {args.db}...")
    hub_matrices = load_hub_blocks(Path(args.db), SELECTED_HUBS)
    for lbl, mat in hub_matrices.items():
        print(f"  {lbl:<25} {mat.shape[0]} blocks")

    total = len(hub_matrices) * len(DIMS) * len(K_VALUES)
    print(f"\nRunning {total} combinations (single-threaded)...\n")

    all_results: dict[str, list[dict]] = {lbl: [] for lbl in hub_matrices}
    done = 0
    for hub_label, full_matrix in hub_matrices.items():
        for dims in DIMS:
            X = truncate_normalize(full_matrix, dims)
            for k in K_VALUES:
                metrics = within_hub_metrics(X, k)
                all_results[hub_label].append({"dims": dims, "k": k, **metrics})
                done += 1
                print(
                    f"  [{done:>3}/{total}]  {hub_label:<22}  {dims:>5}d  k={k}"
                    f"  ari={metrics['ari']:.3f}  sil={metrics['silhouette']:.3f}"
                )
                time.sleep(0.1)

    # Save
    results_path = out_dir / "within_hub_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "hubs": SELECTED_HUBS,
                "dims": DIMS,
                "k_values": K_VALUES,
                "ari_runs": ARI_RUNS,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results → {results_path.name}")

    plot_hub_grid(all_results, out_dir / "within_hub_stability.png")
    plot_summary(all_results, out_dir / "within_hub_summary.png")

    # Find peak dim per hub
    print("\nPeak within-hub ARI per hub (best k):")
    overall_peaks = []
    for hub, rows in all_results.items():
        best = max(rows, key=lambda r: r["ari"])
        overall_peaks.append(best["dims"])
        print(f"  {hub:<25}  {best['dims']:>5}d  k={best['k']}  ari={best['ari']:.3f}")
    print(f"\nMedian peak dim across hubs: {int(np.median(overall_peaks))}d")
    print("Next: run 07_length_stratification.py")


if __name__ == "__main__":
    main()
