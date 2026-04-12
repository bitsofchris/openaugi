#!/usr/bin/env python3
"""Plot cached UMAP projections for all dims and save as images.

Usage:
    python scripts/plot_umaps.py
    python scripts/plot_umaps.py --db /path/to/custom.db
    python scripts/plot_umaps.py --out docs.local/umaps
"""

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
backend_dir = repo_root / "experiments" / "knowledge-explorer" / "backend"
sys.path.insert(0, str(backend_dir))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from server import (  # noqa: E402
    _normalize_coords,
    _truncate_normalize,
    compute_umap,
    load_data_blocks,
    open_db,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIMS_OPTIONS = [32, 64, 96, 128, 256, 512, 1024, 1280, 1536, 2048, 2560, 3072]
DEFAULT_DB = Path.home() / ".openaugi" / "openaugi.db"


def plot_umap(coords: np.ndarray, dims: int, n_blocks: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#0d0d14")
    ax.set_facecolor("#0d0d14")

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=1.5,
        alpha=0.5,
        linewidths=0,
        c="#818cf8",
    )

    ax.set_title(
        f"UMAP  —  dims={dims}  ({n_blocks:,} blocks)",
        color="#e2e8f0",
        fontsize=16,
        pad=16,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default=str(repo_root / "docs.local"))
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading blocks from {db_path}")
    conn = open_db(str(db_path))
    try:
        blocks = load_data_blocks(conn)
    finally:
        conn.close()

    print(f"Loaded {len(blocks):,} blocks\n")

    full_matrix = np.stack(
        [np.frombuffer(b["embedding"], dtype=np.float32).copy() for b in blocks]
    )

    for dims in DIMS_OPTIONS:
        print(f"dims={dims} …", end="", flush=True)
        matrix = _truncate_normalize(full_matrix, dims)
        coords = _normalize_coords(compute_umap(matrix))
        out_path = out_dir / f"umap_{dims:04d}.png"
        plot_umap(coords, dims, len(blocks), out_path)

    print(f"\nDone. {len(DIMS_OPTIONS)} images saved to {out_dir}/")


if __name__ == "__main__":
    main()
