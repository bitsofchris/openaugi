#!/usr/bin/env python3
"""Phase 4: Visualization — combines Phase 2 + 3 outputs into a final report figure.

Generates:
  output/viz_probe_and_stability.png  — dual-panel: probe accuracy + ARI stability
  output/viz_umap_triptych.png        — UMAP at 64d / 256d / 3072d coloured by coarse label
  output/viz_cosine_heatmap.png       — cosine similarity matrix at sweet-spot dim (256d)

Single-threaded. UMAP can be slow on first run but is fast after numba JIT warms up.

Usage:
    python 04_visualize.py
    python 04_visualize.py --sweet-spot 128  # override sweet-spot dim for UMAP highlight
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "output"
SWEET_SPOT = 256  # from Phase 2+3 convergence — override with --sweet-spot
UMAP_DIMS = None  # set in main after parsing args: [32, SWEET_SPOT, 3072]


# ── Helpers ───────────────────────────────────────────────────────────────────


def truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    t = matrix[:, :dims].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return t / norms


def run_umap(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """2D UMAP projection. Returns (N, 2) array."""
    from umap import UMAP

    reducer = UMAP(n_components=2, random_state=seed, n_jobs=1, verbose=False)
    return reducer.fit_transform(X)


def label_colors(labels: list[str]) -> tuple[list[str], dict[str, str]]:
    """Assign a distinct color to each unique label."""
    unique = sorted(set(labels))
    palette = list(mcolors.TABLEAU_COLORS.values())[: len(unique)]
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique)}
    return [color_map[label] for label in labels], color_map


# ── Plot 1: Probe accuracy + ARI stability (side by side) ────────────────────


def plot_probe_and_stability(
    probe_path: Path,
    cluster_path: Path,
    out_path: Path,
    sweet_spot: int,
) -> None:
    with open(probe_path) as f:
        probe = json.load(f)
    with open(cluster_path) as f:
        cluster = json.load(f)

    # Probe data
    p_dims = [r["dims"] for r in probe["results"]]
    coarse_acc = [r["coarse_accuracy"] for r in probe["results"]]
    fine_acc = [r["fine_accuracy"] for r in probe["results"]]

    # ARI stability at k=3 (most stable k across all dims)
    c_dims = sorted(set(r["dims"] for r in cluster["results"]))
    ari_k3 = [
        next(r["ari_stability"] for r in cluster["results"] if r["dims"] == d and r["k"] == 3)
        for d in c_dims
    ]
    ari_k5 = [
        next(r["ari_stability"] for r in cluster["results"] if r["dims"] == d and r["k"] == 5)
        for d in c_dims
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Matryoshka embedding probe — Phase 2 + 3 summary", fontsize=14, y=1.01)

    # ── Left: probe accuracy ──
    ax1.plot(p_dims, coarse_acc, "o-", color="#2563eb", lw=2, ms=6, label="Coarse accuracy")
    ax1.plot(p_dims, fine_acc, "s--", color="#dc2626", lw=2, ms=6, label="Fine accuracy")
    ax1.axvline(sweet_spot, color="#9ca3af", ls=":", lw=1.5)
    ax1.annotate(
        f"sweet spot\n{sweet_spot}d",
        xy=(sweet_spot, 0.05),
        fontsize=8,
        color="#6b7280",
        ha="left",
        xytext=(sweet_spot * 1.1, 0.05),
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(p_dims)
    ax1.set_xticklabels([str(d) for d in p_dims], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Embedding dims (log₂)", fontsize=10)
    ax1.set_ylabel("CV accuracy (3-fold)", fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_title("Linear probe accuracy", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, ls="--")

    # ── Right: ARI stability ──
    ax2.plot(c_dims, ari_k3, "o-", color="#7c3aed", lw=2, ms=6, label="k=3")
    ax2.plot(c_dims, ari_k5, "s--", color="#c026d3", lw=2, ms=6, label="k=5")
    ax2.axvline(sweet_spot, color="#9ca3af", ls=":", lw=1.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(c_dims)
    ax2.set_xticklabels([str(d) for d in c_dims], rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Embedding dims (log₂)", fontsize=10)
    ax2.set_ylabel("Mean pairwise ARI (5 runs)", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Cluster stability (ARI)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, ls="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Plot 2: UMAP triptych ─────────────────────────────────────────────────────


def plot_umap_triptych(
    embeddings: np.ndarray,
    coarse_labels: list[str],
    dims_list: list[int],
    out_path: Path,
) -> None:
    colors, color_map = label_colors(coarse_labels)

    fig, axes = plt.subplots(1, len(dims_list), figsize=(6 * len(dims_list), 6))
    fig.suptitle(
        "UMAP projections at different Matryoshka truncation levels\ncoloured by coarse label",
        fontsize=13,
    )

    for ax, dims in zip(axes, dims_list, strict=False):
        print(f"    UMAP at {dims}d...", end="", flush=True)
        X = truncate_normalize(embeddings, dims)
        coords = run_umap(X)
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.7, linewidths=0)
        ax.set_title(f"{dims}d", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        print(" done")

    # Legend (shared)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=lbl)
        for lbl, c in color_map.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Plot 3: Cosine similarity heatmap at sweet-spot dim ──────────────────────


def plot_cosine_heatmap(
    embeddings: np.ndarray,
    coarse_labels: list[str],
    dims: int,
    out_path: Path,
) -> None:
    X = truncate_normalize(embeddings, dims)

    # Sort blocks by coarse label so same-label blocks are adjacent
    order = sorted(range(len(coarse_labels)), key=lambda i: coarse_labels[i])
    X_sorted = X[order]
    labels_sorted = [coarse_labels[i] for i in order]

    # Cosine similarity = dot product on L2-normalised vectors
    sim = X_sorted @ X_sorted.T
    sim = np.clip(sim, -1, 1)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Cosine similarity")

    # Draw label-boundary lines
    label_changes = [
        i for i in range(1, len(labels_sorted)) if labels_sorted[i] != labels_sorted[i - 1]
    ]
    for pos in label_changes:
        ax.axhline(pos - 0.5, color="white", lw=0.8, alpha=0.7)
        ax.axvline(pos - 0.5, color="white", lw=0.8, alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Cosine similarity matrix at {dims}d — sorted by coarse label\n"
        "Green diagonal blocks = within-label similarity",
        fontsize=11,
        pad=12,
    )

    # Label annotations on y-axis
    unique_labels = []
    prev = None
    for i, lbl in enumerate(labels_sorted):
        if lbl != prev:
            unique_labels.append((i, lbl))
            prev = lbl
    for pos, lbl in unique_labels:
        ax.text(-2, pos, lbl[:28], fontsize=6.5, va="top", ha="right", color="#374151")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labeled", default=str(OUT_DIR / "labeled.json"))
    parser.add_argument("--probe-results", default=str(OUT_DIR / "probe_results.json"))
    parser.add_argument("--cluster-results", default=str(OUT_DIR / "cluster_results.json"))
    parser.add_argument("--sweet-spot", type=int, default=SWEET_SPOT)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    umap_dims = [64, args.sweet_spot, 3072]

    # ── Load labeled data ────────────────────────────────────────────────────
    print(f"Loading {args.labeled}...")
    with open(args.labeled) as f:
        data = json.load(f)
    clean = [
        b
        for b in data
        if b.get("embedding_3072")
        and b.get("coarse_label")
        and b["coarse_label"] != "NEEDS_REVIEW"
    ]
    embeddings = np.array([b["embedding_3072"] for b in clean], dtype=np.float32)
    coarse_labels = [b["coarse_label"] for b in clean]
    print(f"  {len(clean)} blocks")

    # ── Plot 1: Summary panel ────────────────────────────────────────────────
    print("\n[1/3] Probe + stability summary panel...")
    plot_probe_and_stability(
        Path(args.probe_results),
        Path(args.cluster_results),
        out_dir / "viz_probe_and_stability.png",
        args.sweet_spot,
    )

    # ── Plot 2: UMAP triptych ────────────────────────────────────────────────
    print("\n[2/3] UMAP triptych (3 projections)...")
    plot_umap_triptych(embeddings, coarse_labels, umap_dims, out_dir / "viz_umap_triptych.png")

    # ── Plot 3: Cosine heatmap ───────────────────────────────────────────────
    print(f"\n[3/3] Cosine similarity heatmap at {args.sweet_spot}d...")
    plot_cosine_heatmap(
        embeddings, coarse_labels, args.sweet_spot, out_dir / "viz_cosine_heatmap.png"
    )

    print("\nDone. Outputs:")
    for f in sorted(out_dir.glob("viz_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
