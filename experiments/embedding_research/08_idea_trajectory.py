#!/usr/bin/env python3
"""Phase 8: Idea trajectory — finding loops and deepening in real notes.

Two parts:

  PART 1 — Synthetic demo with real sentences
    Contrived journal-style entries that clearly demonstrate:
      • Deepening  — each entry builds on the last (idea advancing)
      • Circling   — same core insight rephrased (idea looping)
    Embedded with text-embedding-3-large, shown with actual text labels.
    You can read the entries and verify the geometry matches.

  PART 2 — Real loop detection (parenting hub, 2 years)
    Finds actual pairs of blocks that are:
      • Far apart in time (≥ 6 months)
      • Semantically similar (cosine sim > threshold)
    These are the loops — ideas you've written twice without realising.
    Shows the actual block text for each pair.

Outputs:
  output/trajectory_synthetic.png   — deepening vs circling with text labels
  output/trajectory_loops.png       — real loop pairs visualised
  output/loop_pairs.json            — all detected loops with text + dates

Usage:
    python 08_idea_trajectory.py
    python 08_idea_trajectory.py --db ~/.openaugi/openaugi.db
    python 08_idea_trajectory.py --sim-threshold 0.82 --min-gap-days 90
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DB = Path.home() / ".openaugi" / "openaugi.db"
OUT_DIR = Path(__file__).parent / "output"

HUB_ID = "4431923b0c175b67"  # parenting — widest time range (2yr)
HUB_LABEL = "parenting"
EMBED_DIMS = 192  # sweet spot for this hub (Phase 6)
SIM_THRESHOLD = 0.82
MIN_GAP_DAYS = 180


# ── Synthetic sentences ───────────────────────────────────────────────────────

DEEPENING = [
    "Exercise is good for me.",
    "Morning exercise seems to set the tone for my whole day.",
    "Morning walks before checking my phone give me a clearer head.",
    "The walk works not because of movement but because it's a protected window "
    "before the day's demands.",
    "The walk is the last moment before I become reactive. I need to protect it.",
    "Protecting the walk is protecting my identity — "
    "who I am before the world tells me what to do.",
]

CIRCLING = [
    "I keep overcommitting and then feeling overwhelmed.",
    "I say yes to too many things and end up stressed.",
    "My calendar is always too full and I feel stretched thin.",
    "I take on more than I can handle and it exhausts me.",
    "The problem is I can't say no and I end up burnt out.",
    "I overload myself with commitments and then crash.",
]


# ── Embedding helper ──────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a short list of texts via OpenAI text-embedding-3-large."""
    try:
        import openai
    except ImportError:
        print("ERROR: openai package needed for synthetic demo. pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set — needed to embed synthetic sentences")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
    vecs = np.array([r.embedding for r in resp.data], dtype=np.float32)
    return vecs


def truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    t = matrix[:, :dims].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norms == 0, 1.0, norms)


def run_umap_2d(X: np.ndarray) -> np.ndarray:
    from umap import UMAP

    return UMAP(n_components=2, random_state=42, n_jobs=1, verbose=False).fit_transform(X)


# ── Part 1: Synthetic ─────────────────────────────────────────────────────────


def plot_synthetic(out_path: Path) -> None:
    print("Embedding synthetic sentences via text-embedding-3-large...")

    all_texts = DEEPENING + CIRCLING
    vecs = embed_texts(all_texts)
    vecs_64 = truncate_normalize(vecs, 64)  # low dim for clean 2D projection
    coords = run_umap_2d(vecs_64)

    d_coords = coords[: len(DEEPENING)]
    c_coords = coords[len(DEEPENING) :]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Synthetic demo — same embedding model, contrived sentences\n"
        "Read the labels to verify the geometry makes sense",
        fontsize=12,
        y=1.01,
    )

    for ax, pts, texts, color, title, desc in [
        (
            axes[0],
            d_coords,
            DEEPENING,
            "#2563eb",
            "Deepening",
            "Each entry builds on the last.\nIdea advances — should see drift in one direction.",
        ),
        (
            axes[1],
            c_coords,
            CIRCLING,
            "#dc2626",
            "Circling",
            "Same insight rephrased six ways.\nIdea loops — should see tight cluster, no drift.",
        ),
    ]:
        n = len(pts)
        alphas = np.linspace(0.35, 1.0, n)

        # Arrows between consecutive points
        for i in range(n - 1):
            ax.annotate(
                "",
                xy=pts[i + 1],
                xytext=pts[i],
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.3, mutation_scale=14),
            )

        # Points + text labels
        for i, (pt, txt, alpha) in enumerate(zip(pts, texts, alphas, strict=False)):
            ax.scatter(
                *pt, color=color, s=90, alpha=alpha, zorder=5, edgecolors="white", linewidths=0.8
            )
            # Wrap text at ~40 chars
            wrapped = txt if len(txt) < 45 else txt[:42] + "…"
            ax.annotate(
                f"{i + 1}. {wrapped}",
                pt,
                fontsize=7.5,
                color="#222",
                xytext=(8, 4),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="none"),
            )

        # Coherence
        deltas = np.diff(pts, axis=0)
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        unit_d = deltas / np.where(norms < 1e-8, 1e-8, norms)
        coh = float(np.mean([unit_d[i] @ unit_d[i + 1] for i in range(len(unit_d) - 1)]))

        ax.set_title(f"{title}  (coherence: {coh:+.2f})", fontsize=12, pad=8)
        ax.text(
            0.5,
            -0.06,
            desc,
            transform=ax.transAxes,
            fontsize=9,
            ha="center",
            color="#555",
            style="italic",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#e5e5e5")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Part 2: Real loop detection ───────────────────────────────────────────────


def load_hub_blocks_dated(db_path: Path, hub_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT DISTINCT b.id, b.content, b.block_time, b.embedding
        FROM links l
        JOIN blocks b ON l.from_id = b.id
        WHERE l.kind = 'links_to'
          AND l.to_id = ?
          AND b.kind = 'data_block'
          AND b.embedding IS NOT NULL
          AND b.block_time IS NOT NULL
          AND length(b.content) > 50
        ORDER BY b.block_time ASC
        """,
        (hub_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "date": r["block_time"][:10],
            "text": (r["content"] or "").strip(),
            "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy(),
        }
        for r in rows
    ]


def find_loops(
    blocks: list[dict],
    X: np.ndarray,
    sim_threshold: float,
    min_gap_days: int,
) -> list[dict]:
    """Find pairs far apart in time but semantically similar.

    Returns list of {i, j, date_i, date_j, gap_days, similarity, text_i, text_j}
    sorted by similarity descending.
    """
    n = len(blocks)
    loops = []

    for i in range(n):
        for j in range(i + 1, n):
            date_i = datetime.fromisoformat(blocks[i]["date"])
            date_j = datetime.fromisoformat(blocks[j]["date"])
            gap = (date_j - date_i).days
            if gap < min_gap_days:
                continue
            sim = float(X[i] @ X[j])
            if sim >= sim_threshold:
                loops.append(
                    {
                        "i": i,
                        "j": j,
                        "date_i": blocks[i]["date"],
                        "date_j": blocks[j]["date"],
                        "gap_days": gap,
                        "similarity": sim,
                        "text_i": blocks[i]["text"],
                        "text_j": blocks[j]["text"],
                    }
                )

    loops.sort(key=lambda x: x["similarity"], reverse=True)
    return loops


def plot_loops(
    blocks: list[dict],
    coords: np.ndarray,
    loops: list[dict],
    out_path: Path,
    top_n: int = 8,
) -> None:
    n = len(blocks)
    time_idx = np.linspace(0, 1, n)

    fig, (ax_map, ax_pairs) = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    fig.suptitle(
        f"Loop detection — {HUB_LABEL} hub\n"
        f"Same idea written ≥{MIN_GAP_DAYS} days apart  (similarity ≥ {SIM_THRESHOLD})",
        fontsize=12,
        y=1.01,
    )

    # ── Left: UMAP map with loop arcs ────────────────────────────────────────
    ax_map.scatter(
        coords[:, 0],
        coords[:, 1],
        c=time_idx,
        cmap="plasma",
        s=30,
        alpha=0.5,
        edgecolors="none",
        zorder=3,
    )

    # Draw arcs for top loops
    loop_colors = plt.cm.Reds(np.linspace(0.5, 0.95, min(top_n, len(loops))))
    for lp, lc in zip(loops[:top_n], loop_colors, strict=False):
        pi, pj = coords[lp["i"]], coords[lp["j"]]
        ax_map.plot([pi[0], pj[0]], [pi[1], pj[1]], color=lc, lw=1.5, alpha=0.7, zorder=4)
        ax_map.scatter(*pi, color=lc, s=60, zorder=5, edgecolors="white", linewidths=0.8)
        ax_map.scatter(*pj, color=lc, s=60, zorder=5, edgecolors="white", linewidths=0.8)

    ax_map.set_title(f"UMAP at {EMBED_DIMS}d — red lines = loops", fontsize=10)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([blocks[0]["date"], blocks[-1]["date"]], fontsize=8)

    # ── Right: Pair text panel ────────────────────────────────────────────────
    ax_pairs.axis("off")
    show = loops[: min(top_n, 5)]  # top 5 for readability
    y = 0.97
    ax_pairs.text(
        0,
        y,
        f"Top {len(show)} loops by similarity",
        fontsize=10,
        fontweight="bold",
        transform=ax_pairs.transAxes,
        va="top",
    )
    y -= 0.05

    for k, lp in enumerate(show):
        sim_pct = lp["similarity"] * 100
        gap_mo = lp["gap_days"] // 30
        header = (
            f"#{k + 1}  sim={sim_pct:.0f}%  ·  "
            f"{lp['date_i']}  →  {lp['date_j']}  ({gap_mo} mo apart)"
        )
        ax_pairs.text(
            0,
            y,
            header,
            fontsize=8.5,
            fontweight="bold",
            color="#dc2626",
            transform=ax_pairs.transAxes,
            va="top",
        )
        y -= 0.04

        # Truncate each to ~200 chars
        t_i = lp["text_i"].replace("\n", " ")[:200]
        t_j = lp["text_j"].replace("\n", " ")[:200]
        ax_pairs.text(
            0.02,
            y,
            f"A: {t_i}…",
            fontsize=7.5,
            color="#1e40af",
            transform=ax_pairs.transAxes,
            va="top",
            wrap=True,
            style="italic",
        )
        y -= 0.06
        ax_pairs.text(
            0.02,
            y,
            f"B: {t_j}…",
            fontsize=7.5,
            color="#166534",
            transform=ax_pairs.transAxes,
            va="top",
            wrap=True,
            style="italic",
        )
        y -= 0.07
        ax_pairs.plot(
            [0, 1], [y + 0.01, y + 0.01], color="#e5e5e5", lw=0.8, transform=ax_pairs.transAxes
        )
        y -= 0.01

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
    parser.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD)
    parser.add_argument("--min-gap-days", type=int, default=MIN_GAP_DAYS)
    parser.add_argument(
        "--skip-synthetic", action="store_true", help="Skip Part 1 (no OPENAI_API_KEY needed)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Part 1 ────────────────────────────────────────────────────────────────
    if not args.skip_synthetic:
        print("=" * 60)
        print("PART 1 — Synthetic (contrived sentences, real embeddings)")
        print("=" * 60)
        plot_synthetic(out_dir / "trajectory_synthetic.png")

    # ── Part 2 ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"PART 2 — Real loop detection: {HUB_LABEL} hub")
    print("=" * 60)

    print("Loading blocks...")
    blocks = load_hub_blocks_dated(Path(args.db), HUB_ID)
    print(f"  {len(blocks)} blocks  ({blocks[0]['date']} → {blocks[-1]['date']})")

    full_matrix = np.stack([b["embedding"] for b in blocks])
    X = truncate_normalize(full_matrix, EMBED_DIMS)

    print(f"Finding loops (sim ≥ {args.sim_threshold}, gap ≥ {args.min_gap_days} days)...")
    loops = find_loops(blocks, X, args.sim_threshold, args.min_gap_days)
    print(f"  Found {len(loops)} loop pairs")

    if not loops:
        print("  No loops found — try lowering --sim-threshold")
        return

    # Save all loops
    loops_path = out_dir / "loop_pairs.json"
    with open(loops_path, "w") as f:
        json.dump(
            {
                "hub": HUB_LABEL,
                "embed_dims": EMBED_DIMS,
                "sim_threshold": args.sim_threshold,
                "min_gap_days": args.min_gap_days,
                "total_loops": len(loops),
                "pairs": [
                    {k: v for k, v in lp.items() if k not in ("i", "j")} for lp in loops[:50]
                ],
            },
            f,
            indent=2,
        )
    print(f"  Saved → {loops_path.name}")

    # Print top 5 to terminal
    print("\nTop 5 loops (same idea, months apart):\n")
    for k, lp in enumerate(loops[:5]):
        print(
            f"  #{k + 1}  sim={lp['similarity']:.3f}  ·  "
            f"{lp['date_i']} → {lp['date_j']}  ({lp['gap_days'] // 30} mo)"
        )
        print(f"    A: {lp['text_i'].replace(chr(10), ' ')[:120]}...")
        print(f"    B: {lp['text_j'].replace(chr(10), ' ')[:120]}...")
        print()

    print("Running UMAP for loop map...")
    coords = run_umap_2d(X)
    plot_loops(blocks, coords, loops, out_dir / "trajectory_loops.png")


if __name__ == "__main__":
    main()
