#!/usr/bin/env python3
"""Phase 2: Linear probe — logistic regression accuracy vs. truncation dimension.

For each Matryoshka truncation level, trains a logistic regression classifier
(5-fold stratified CV) to predict coarse and fine labels. Plots accuracy vs.
log-dimension to find where representation quality saturates.

Key signal: the dimension where the coarse curve plateaus but the fine curve
is still climbing is the idea-level sweet spot.

Outputs:
  output/probe_results.json   — raw accuracy table
  output/probe_plot.png       — dual-curve accuracy plot

Usage:
    python 02_linear_probe.py
    python 02_linear_probe.py --labeled output/labeled.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

DIMS = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]
CV_FOLDS = 3  # 3-fold: 39 total fits instead of 65, plenty for 200 samples
OUT_DIR = Path(__file__).parent / "output"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_labeled(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Returns (embeddings [N x 3072], coarse_labels, fine_labels).

    Skips blocks with NEEDS_REVIEW labels so they don't pollute the probe.
    """
    with open(path) as f:
        data = json.load(f)

    clean = [
        b
        for b in data
        if b.get("coarse_label")
        and b["coarse_label"] != "NEEDS_REVIEW"
        and b.get("fine_label")
        and b["fine_label"] != "NEEDS_REVIEW"
        and b.get("embedding_3072")
    ]

    skipped = len(data) - len(clean)
    if skipped:
        print(f"  Skipped {skipped} blocks with NEEDS_REVIEW labels")

    embeddings = np.array([b["embedding_3072"] for b in clean], dtype=np.float32)
    coarse = [b["coarse_label"] for b in clean]
    fine = [b["fine_label"] for b in clean]
    return embeddings, coarse, fine


def truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    """Slice to dims and L2-normalize each row."""
    t = matrix[:, :dims].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return t / norms


# ── Probing ───────────────────────────────────────────────────────────────────


def probe(X: np.ndarray, y_encoded: np.ndarray, n_classes: int) -> float:
    """5-fold stratified CV accuracy with logistic regression."""
    # For many classes with small N, reduce max_iter and use a fast solver
    clf = LogisticRegression(
        max_iter=200,  # saga converges fast on small N
        solver="saga",  # stochastic — low memory, single-threaded, no worker pool
        C=1.0,
        n_jobs=1,  # never spawn workers
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="accuracy", n_jobs=1)
    return float(scores.mean())


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_results(results: list[dict], out_path: Path) -> None:
    dims_list = [r["dims"] for r in results]
    coarse_acc = [r["coarse_accuracy"] for r in results]
    fine_acc = [r["fine_accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        dims_list,
        coarse_acc,
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=6,
        label="Coarse accuracy",
    )
    ax.plot(
        dims_list,
        fine_acc,
        "s--",
        color="#dc2626",
        linewidth=2,
        markersize=6,
        label="Fine accuracy",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims_list)
    ax.set_xticklabels([str(d) for d in dims_list], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Embedding dimensions (log₂ scale)", fontsize=11)
    ax.set_ylabel("CV accuracy (5-fold)", fontsize=11)
    ax.set_title("Linear probe accuracy vs. Matryoshka truncation dimension", fontsize=13, pad=14)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Annotate the knee: first dim where coarse plateaus (derivative drops below threshold)
    coarse_deltas = [coarse_acc[i + 1] - coarse_acc[i] for i in range(len(coarse_acc) - 1)]
    plateau_idx = next(
        (i + 1 for i, d in enumerate(coarse_deltas) if d < 0.01),
        len(dims_list) - 1,
    )
    ax.axvline(
        dims_list[plateau_idx],
        color="#9ca3af",
        linestyle=":",
        linewidth=1.5,
        label=f"Coarse plateau ≈ {dims_list[plateau_idx]}d",
    )
    ax.annotate(
        f"coarse plateau\n~{dims_list[plateau_idx]}d",
        xy=(dims_list[plateau_idx], coarse_acc[plateau_idx]),
        xytext=(dims_list[plateau_idx] * 1.15, coarse_acc[plateau_idx] - 0.06),
        fontsize=8,
        color="#6b7280",
        arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path.name}")


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

    print(f"Loading labeled blocks from {labeled_path}...")
    embeddings, coarse_labels, fine_labels = load_labeled(labeled_path)
    N = len(embeddings)
    print(f"  {N} clean blocks")

    coarse_enc = LabelEncoder().fit(coarse_labels)
    fine_enc = LabelEncoder().fit(fine_labels)
    y_coarse = coarse_enc.transform(coarse_labels)
    y_fine = fine_enc.transform(fine_labels)

    n_coarse = len(coarse_enc.classes_)
    n_fine = len(fine_enc.classes_)
    print(f"  Coarse classes: {n_coarse}  →  {list(coarse_enc.classes_)}")
    print(f"  Fine classes:   {n_fine}")
    print(f"\nRunning {CV_FOLDS}-fold CV probe across {len(DIMS)} truncation levels...\n")

    results = []
    header = f"{'dims':>6}  {'coarse acc':>10}  {'fine acc':>10}"
    print(header)
    print("-" * len(header))

    for dims in DIMS:
        X = truncate_normalize(embeddings, dims)
        coarse_acc = probe(X, y_coarse, n_coarse)
        fine_acc = probe(X, y_fine, n_fine)
        results.append({"dims": dims, "coarse_accuracy": coarse_acc, "fine_accuracy": fine_acc})
        print(f"  {dims:>4}d    {coarse_acc:>9.1%}    {fine_acc:>9.1%}")
        time.sleep(0.5)  # brief pause between dims — let CPU cool

    # Write results
    results_path = out_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "n_blocks": N,
                "cv_folds": CV_FOLDS,
                "coarse_classes": list(coarse_enc.classes_),
                "fine_classes": list(fine_enc.classes_),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved → {results_path.name}")

    plot_results(results, out_dir / "probe_plot.png")

    # Summary
    best_coarse = max(results, key=lambda r: r["coarse_accuracy"])
    best_fine = max(results, key=lambda r: r["fine_accuracy"])
    print(
        f"\nBest coarse accuracy: {best_coarse['coarse_accuracy']:.1%} at {best_coarse['dims']}d"
    )
    print(f"Best fine accuracy:   {best_fine['fine_accuracy']:.1%} at {best_fine['dims']}d")

    # Estimate sweet spot: last dim where fine is still gaining meaningfully
    fine_acc_list = [r["fine_accuracy"] for r in results]
    fine_deltas = [fine_acc_list[i + 1] - fine_acc_list[i] for i in range(len(fine_acc_list) - 1)]
    sweet_idx = next(
        (i + 1 for i, d in enumerate(reversed(fine_deltas)) if d > 0.005),
        len(DIMS) - 1,
    )
    sweet_idx = len(DIMS) - 1 - sweet_idx
    print(f"\nEstimated sweet spot: {DIMS[sweet_idx]}d (last dim with meaningful fine-label gain)")
    print("\nNext: run 03_cluster_quality.py")


if __name__ == "__main__":
    main()
