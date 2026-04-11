"""
Knowledge Explorer backend — reads from openaugi SQLite DB,
runs UMAP on data_block embeddings, and serves the explorer JSON.

Requires: pip install fastapi uvicorn umap-learn numpy

Usage:
    python backend/server.py [--db ~/.openaugi/openaugi.db]
"""

import argparse
import hashlib
import json
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_DB = Path.home() / ".openaugi" / "openaugi.db"

app = FastAPI(title="OpenAugi Knowledge Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB queries ────────────────────────────────────────────────────────────────


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_data_blocks(conn: sqlite3.Connection) -> list[dict]:
    """Load data_blocks that have embeddings. Embedding blob stays as bytes."""
    rows = conn.execute("""
        SELECT id,
               content,
               title,
               block_time,
               embedding,
               json_extract(metadata, '$.source_path') AS source_path,
               json_extract(metadata, '$.cluster_assignments') AS cluster_assignments_json
        FROM blocks
        WHERE kind = 'data_block'
          AND embedding IS NOT NULL
        ORDER BY id
    """).fetchall()
    result = []
    for r in rows:
        ca_raw = r["cluster_assignments_json"]
        try:
            ca = json.loads(ca_raw) if ca_raw else {}
        except (json.JSONDecodeError, TypeError):
            ca = {}
        result.append(
            {
                "id": r["id"],
                "content": r["content"] or "",
                "title": r["title"] or "",
                "block_time": r["block_time"],
                "embedding": r["embedding"],  # raw float32 bytes
                "source_path": r["source_path"] or r["title"] or r["id"],
                "cluster_assignments": ca,
            }
        )
    return result


def load_source_content(conn: sqlite3.Connection, source_paths: set[str]) -> dict[str, str]:
    """
    Load a snippet of each source note's content from context_block:document blocks.
    Falls back to empty string if not found.
    """
    if not source_paths:
        return {}
    rows = conn.execute("""
        SELECT json_extract(metadata, '$.source_path') AS sp, substr(content, 1, 500) AS snippet
        FROM blocks
        WHERE kind = 'context_block:document'
          AND json_extract(metadata, '$.source_path') IS NOT NULL
    """).fetchall()
    return {r["sp"]: r["snippet"] or "" for r in rows if r["sp"] in source_paths}


def load_cluster_blocks(conn: sqlite3.Connection) -> list[dict]:
    """Load context_block:cluster blocks with their metadata."""
    rows = conn.execute("""
        SELECT id, content, title, metadata
        FROM blocks
        WHERE kind = 'context_block:cluster'
        ORDER BY id
    """).fetchall()
    result = []
    for r in rows:
        try:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        temporal_raw = meta.get("temporal")
        result.append(
            {
                "id": r["id"],
                "content": r["content"],
                "title": r["title"] or "",
                "pass_id": meta.get("pass_id", ""),
                "parent_pass_id": meta.get("parent_pass_id"),
                "cluster_label": meta.get("cluster_label"),
                "dims": meta.get("dims", 0),
                "min_cluster_size": meta.get("min_cluster_size"),
                "member_count": meta.get("member_count", 0),
                "noise_count": meta.get("noise_count", 0),
                "temporal": temporal_raw,
            }
        )
    return result


def discover_passes(cluster_blocks: list[dict]) -> list[dict]:
    """
    Derive ordered list of passes from cluster blocks.
    Sorted: scope=all (no parent) first by dims asc, then scope=within by dims asc.
    """
    seen: dict[str, dict] = {}  # pass_id → {id, dims, parent_pass_id}
    for cb in cluster_blocks:
        pid = cb["pass_id"]
        if pid and pid not in seen:
            seen[pid] = {
                "id": pid,
                "description": "",
                "dims": cb["dims"],
                "scope": "within" if cb.get("parent_pass_id") else "all",
                "parent_pass": cb.get("parent_pass_id"),
            }
    passes = list(seen.values())
    # Topological sort: parents before children, then by dims asc within each level
    root_passes = sorted([p for p in passes if not p["parent_pass"]], key=lambda p: p["dims"])
    child_passes = sorted([p for p in passes if p["parent_pass"]], key=lambda p: p["dims"])
    return root_passes + child_passes


UMAP_CACHE_DIR = Path.home() / ".openaugi" / "umap_cache"


def _matrix_hash(matrix: np.ndarray) -> str:
    """SHA-1 of the raw embedding bytes — changes when any embedding is updated."""
    return hashlib.sha1(matrix.tobytes()).hexdigest()[:16]  # noqa: S324


def compute_umap(matrix: np.ndarray) -> np.ndarray:
    """UMAP to 2D with disk cache.

    Cache key is a hash of the full embedding matrix, so it invalidates
    automatically whenever embeddings are re-run. Cache lives in
    ~/.openaugi/umap_cache/.
    """
    UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _matrix_hash(matrix)
    cache_file = UMAP_CACHE_DIR / f"umap_{cache_key}_{len(matrix)}.pkl"

    if cache_file.exists():
        log.info("UMAP cache hit — loading from %s", cache_file)
        with cache_file.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    log.info("UMAP cache miss — computing (this takes ~30-60s for large vaults)…")
    try:
        import umap  # type: ignore
    except ImportError:
        raise HTTPException(500, "umap-learn not installed: pip install umap-learn") from None
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(matrix) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    coords = reducer.fit_transform(matrix)

    with cache_file.open("wb") as f:
        pickle.dump(coords, f)
    log.info("UMAP result cached to %s", cache_file)
    return coords


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/data")
def get_data(db: str = str(DEFAULT_DB)):
    db_path = Path(db)
    if not db_path.exists():
        raise HTTPException(404, f"Database not found: {db_path}")

    conn = open_db(str(db_path))
    try:
        log.info("Loading data blocks from %s", db_path)
        data_blocks = load_data_blocks(conn)
        if not data_blocks:
            raise HTTPException(
                404, "No data_blocks with embeddings found. Run 'openaugi ingest' first."
            )

        log.info("Loaded %d blocks with embeddings", len(data_blocks))

        # Extract embedding matrix
        matrix = np.stack(
            [np.frombuffer(b["embedding"], dtype=np.float32).copy() for b in data_blocks]
        )

        log.info("Running UMAP on %d × %d matrix", *matrix.shape)
        coords = compute_umap(matrix)

        # Normalize coords to [-1, 1]
        for axis in range(2):
            col = coords[:, axis]
            col_range = col.max() - col.min()
            if col_range > 0:
                coords[:, axis] = (col - col.min()) / col_range * 2 - 1

        # Load cluster blocks
        cluster_blocks = load_cluster_blocks(conn)

        # Load source note content for blocks that have a source_path
        source_paths = {b["source_path"] for b in data_blocks if b["source_path"]}
        source_content_map = load_source_content(conn, source_paths)

        # Build clusters dict (cluster_block_id → ClusterInfo)
        clusters: dict[str, dict] = {}
        clusters_by_pass: dict[str, list[str]] = {}
        for cb in cluster_blocks:
            bid = cb["id"]
            pid = cb["pass_id"]
            label = str(cb["cluster_label"]) if cb["cluster_label"] is not None else ""
            clusters[bid] = {
                "id": bid,
                "pass_id": pid,
                "label": label,
                "dims": cb["dims"],
                "summary": cb["content"] or None,
                "member_count": cb["member_count"],
                "temporal": cb["temporal"],
            }
            clusters_by_pass.setdefault(pid, []).append(bid)

        # Discover pass order
        passes = discover_passes(cluster_blocks)

        # Build output blocks
        output_blocks = []
        for i, b in enumerate(data_blocks):
            # Parse date from block_time
            block_date = None
            if b["block_time"]:
                try:
                    dt = datetime.fromisoformat(b["block_time"][:19])
                    block_date = dt.strftime("%Y-%m")
                except ValueError:
                    pass

            # Map numeric cluster labels to label strings that match ClusterInfo.label
            # cluster_assignments stored as {"pass_id": "3"} or {"pass_id": "3_7"}
            output_blocks.append(
                {
                    "id": b["id"],
                    "content": b["content"],
                    "source_path": b["source_path"],
                    "source_content": source_content_map.get(b["source_path"], ""),
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                    "date": block_date,
                    "cluster_assignments": b["cluster_assignments"],
                }
            )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "block_count": len(output_blocks),
            "passes": passes,
            "blocks": output_blocks,
            "clusters": clusters,
            "clusters_by_pass": clusters_by_pass,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to openaugi SQLite DB")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
