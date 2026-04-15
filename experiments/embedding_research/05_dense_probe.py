#!/usr/bin/env python3
"""Phase 5: Dense hub probe — uses existing vault links as ground truth clusters.

Instead of LLM-labeled random blocks, we use hub notes as cluster labels.
Every data_block that links_to a hub IS a member of that hub's topic cluster —
those links were made deliberately. This gives us clean, human-curated ground
truth without any labeling step.

Method:
  1. Pull data_blocks linked to each selected hub (links_to relation)
  2. Deduplicate: blocks linking to multiple selected hubs are assigned to the
     hub they link to most, or dropped if ambiguous
  3. For each Matryoshka truncation dimension, run:
       a. Linear probe  — can logistic regression predict hub membership?
       b. Within vs. between cosine similarity — do same-hub blocks score
          higher similarity than cross-hub blocks? By how much?
  4. Plot: accuracy + similarity gap vs. dimension

The similarity gap (within_mean - between_mean) is the cleanest signal:
it doesn't depend on label count balance and answers directly "does the
embedding know these ideas are related?"

Outputs:
  output/dense_probe_results.json
  output/dense_probe_accuracy.png     — probe accuracy per hub + overall
  output/dense_similarity_gap.png     — within vs. between cosine per dim

Usage:
    python 05_dense_probe.py
    python 05_dense_probe.py --db ~/.openaugi/openaugi.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

DEFAULT_DB = Path.home() / ".openaugi" / "openaugi.db"
OUT_DIR = Path(__file__).parent / "output"

# Hubs chosen for topical distinctness + block count (≥40 linked blocks with embeddings)
# Format: (hub_id, short_label)
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
CV_FOLDS = 3


# ── DB ────────────────────────────────────────────────────────────────────────


def load_hub_blocks(db_path: Path, hubs: list[tuple[str, str]]) -> list[dict]:
    """Pull data_blocks linked to each hub. Deduplicate cross-hub members."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    hub_ids = [h[0] for h in hubs]
    id_to_label = {h[0]: h[1] for h in hubs}

    # For each data_block, find which selected hubs it links to + how many times
    placeholders = ",".join("?" * len(hub_ids))
    rows = conn.execute(
        f"""
        SELECT
            b.id,
            b.content,
            b.title,
            b.block_time,
            json_extract(b.metadata, '$.source_path') AS source_path,
            b.embedding,
            l.to_id AS hub_id,
            count(*) as link_count
        FROM links l
        JOIN blocks b ON l.from_id = b.id
        WHERE l.kind = 'links_to'
          AND l.to_id IN ({placeholders})
          AND b.kind = 'data_block'
          AND b.embedding IS NOT NULL
          AND length(b.content) > 50
        GROUP BY b.id, l.to_id
        ORDER BY b.id, link_count DESC
        """,
        hub_ids,
    ).fetchall()
    conn.close()

    # Group by block_id — assign to the hub it links to most (first = highest count)
    seen: dict[str, dict] = {}
    for row in rows:
        bid = row["id"]
        if bid in seen:
            # Already assigned to a hub — skip (dedup)
            continue
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        seen[bid] = {
            "block_id": bid,
            "text": (row["content"] or "").strip(),
            "source_path": row["source_path"] or "",
            "date": row["block_time"][:7] if row["block_time"] else None,
            "hub_label": id_to_label[row["hub_id"]],
            "embedding_3072": emb,
        }

    return list(seen.values())


# ── Math ──────────────────────────────────────────────────────────────────────


def truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    t = matrix[:, :dims].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norms == 0, 1.0, norms)


def probe_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=200, solver="saga", C=1.0, n_jobs=1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(scores.mean())


def similarity_gap(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (within_mean, between_mean) cosine similarities.

    Samples pairs rather than computing full N² matrix — safe for large N.
    """
    rng = np.random.default_rng(42)
    n = len(X)
    n_pairs = min(5000, n * (n - 1) // 2)

    within, between = [], []
    attempts = 0
    while (len(within) < n_pairs or len(between) < n_pairs) and attempts < n_pairs * 10:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            attempts += 1
            continue
        sim = float(X[i] @ X[j])  # already L2-normalised → cosine similarity
        if labels[i] == labels[j]:
            if len(within) < n_pairs:
                within.append(sim)
        else:
            if len(between) < n_pairs:
                between.append(sim)
        attempts += 1

    return float(np.mean(within)) if within else 0.0, float(np.mean(between)) if between else 0.0


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_accuracy(results: list[dict], out_path: Path) -> None:
    dims_list = [r["dims"] for r in results]
    acc = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dims_list, acc, "o-", color="#2563eb", lw=2.5, ms=7)

    for r in results:
        ax.annotate(
            f"{r['accuracy']:.0%}",
            (r["dims"], r["accuracy"]),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7.5,
            ha="center",
            color="#374151",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims_list)
    ax.set_xticklabels([str(d) for d in dims_list], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Embedding dims (log₂ scale)", fontsize=11)
    ax.set_ylabel(f"CV accuracy ({CV_FOLDS}-fold)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Hub-membership probe accuracy vs. Matryoshka truncation\n"
        f"(ground truth = vault links — {len(SELECTED_HUBS)} hubs)",
        fontsize=12,
        pad=12,
    )
    ax.grid(True, alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def plot_similarity_gap(results: list[dict], out_path: Path) -> None:
    dims_list = [r["dims"] for r in results]
    within = [r["within_cosine"] for r in results]
    between = [r["between_cosine"] for r in results]
    gap = [w - b for w, b in zip(within, between, strict=False)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Within-hub vs. between-hub cosine similarity\n"
        "Gap = how well the embedding knows ideas in the same hub belong together",
        fontsize=12,
    )

    ax1.plot(dims_list, within, "o-", color="#16a34a", lw=2, ms=6, label="Within-hub (same topic)")
    ax1.plot(
        dims_list,
        between,
        "s--",
        color="#dc2626",
        lw=2,
        ms=6,
        label="Between-hub (different topics)",
    )
    ax1.set_ylabel("Mean cosine similarity", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, ls="--")
    ax1.set_ylim(0, 1.0)

    ax2.fill_between(dims_list, gap, alpha=0.25, color="#7c3aed")
    ax2.plot(dims_list, gap, "o-", color="#7c3aed", lw=2.5, ms=6, label="Gap (within − between)")
    # Mark peak
    peak_idx = int(np.argmax(gap))
    ax2.annotate(
        f"peak gap\n{dims_list[peak_idx]}d  ({gap[peak_idx]:.3f})",
        xy=(dims_list[peak_idx], gap[peak_idx]),
        xytext=(dims_list[peak_idx], gap[peak_idx] + 0.015),
        fontsize=9,
        ha="center",
        color="#5b21b6",
        arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1),
    )
    ax2.set_xlabel("Embedding dims (log₂ scale)", fontsize=11)
    ax2.set_ylabel("Similarity gap", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, ls="--")

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims_list)
        ax.set_xticklabels([str(d) for d in dims_list], rotation=45, ha="right", fontsize=9)

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

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading hub-linked blocks from {args.db}...")
    blocks = load_hub_blocks(Path(args.db), SELECTED_HUBS)

    counts = Counter(b["hub_label"] for b in blocks)
    print(f"  {len(blocks)} blocks across {len(counts)} hubs:")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<25} {n}")

    embeddings = np.stack([b["embedding_3072"] for b in blocks])
    hub_labels = [b["hub_label"] for b in blocks]
    le = LabelEncoder().fit(hub_labels)
    y = le.transform(hub_labels)

    print(f"\nRunning probe across {len(DIMS)} dims (single-threaded)...\n")
    print(f"  {'dims':>5}  {'accuracy':>9}  {'within_cos':>10}  {'between_cos':>11}  {'gap':>6}")
    print("  " + "-" * 55)

    results = []
    for dims in DIMS:
        X = truncate_normalize(embeddings, dims)
        acc = probe_accuracy(X, y)
        w, b = similarity_gap(X, y)
        results.append(
            {
                "dims": dims,
                "accuracy": acc,
                "within_cosine": w,
                "between_cosine": b,
                "gap": w - b,
            }
        )
        print(f"  {dims:>5}d  {acc:>9.1%}  {w:>10.3f}  {b:>11.3f}  {w - b:>6.3f}")
        time.sleep(0.3)

    # Save
    results_path = out_dir / "dense_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "hubs": [{"id": hid, "label": lbl} for hid, lbl in SELECTED_HUBS],
                "block_counts": dict(counts),
                "total_blocks": len(blocks),
                "cv_folds": CV_FOLDS,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results → {results_path.name}")

    plot_accuracy(results, out_dir / "dense_probe_accuracy.png")
    plot_similarity_gap(results, out_dir / "dense_similarity_gap.png")

    # Summary
    best_acc = max(results, key=lambda r: r["accuracy"])
    best_gap = max(results, key=lambda r: r["gap"])
    print(f"\nPeak accuracy:      {best_acc['accuracy']:.1%} at {best_acc['dims']}d")
    print(f"Peak similarity gap: {best_gap['gap']:.3f} at {best_gap['dims']}d")
    print(f"\nSweet spot: {best_gap['dims']}d")
    print("Next: update FINDINGS.md with results")


if __name__ == "__main__":
    main()
