"""
Knowledge Explorer backend — reads from openaugi SQLite DB,
runs UMAP on data_block embeddings, and serves the explorer JSON.

Requires: pip install fastapi uvicorn umap-learn numpy hdbscan scikit-learn

Usage:
    python backend/server.py [--db ~/.openaugi/openaugi.db]
"""

import argparse
import contextlib
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
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_DB = Path.home() / ".openaugi" / "openaugi.db"
UMAP_CACHE_DIR = Path.home() / ".openaugi" / "umap_cache"

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
    root_passes = sorted([p for p in passes if not p["parent_pass"]], key=lambda p: p["dims"])
    child_passes = sorted([p for p in passes if p["parent_pass"]], key=lambda p: p["dims"])
    return root_passes + child_passes


# ── UMAP helpers ──────────────────────────────────────────────────────────────


def _matrix_hash(matrix: np.ndarray) -> str:
    """SHA-1 of the raw embedding bytes — changes when any embedding is updated."""
    return hashlib.sha1(matrix.tobytes()).hexdigest()[:16]  # noqa: S324


def _run_umap(matrix: np.ndarray) -> np.ndarray:
    """Pure UMAP computation, no caching."""
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
    return reducer.fit_transform(matrix)


def compute_umap(matrix: np.ndarray) -> np.ndarray:
    """UMAP to 2D with disk cache keyed on matrix hash.

    Cache lives in ~/.openaugi/umap_cache/ and invalidates automatically
    whenever embeddings change (hash of the matrix bytes changes).
    """
    UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _matrix_hash(matrix)
    cache_file = UMAP_CACHE_DIR / f"umap_{cache_key}_{len(matrix)}.pkl"

    if cache_file.exists():
        log.info("UMAP cache hit — loading from %s", cache_file)
        with cache_file.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    log.info("UMAP cache miss — computing (this takes ~30-60s for large vaults)…")
    coords = _run_umap(matrix)

    with cache_file.open("wb") as f:
        pickle.dump(coords, f)
    log.info("UMAP result cached to %s", cache_file)
    return coords


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize 2D coords to [-1, 1] per axis."""
    coords = coords.copy()
    for axis in range(2):
        col = coords[:, axis]
        col_range = col.max() - col.min()
        if col_range > 0:
            coords[:, axis] = (col - col.min()) / col_range * 2 - 1
    return coords


def _truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    """Truncate to dims and L2-normalize (important after Matryoshka truncation)."""
    dims = min(dims, matrix.shape[1])
    m = matrix[:, :dims].copy()
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return m / norms


# ── Explore endpoint ──────────────────────────────────────────────────────────


class ExploreRequest(BaseModel):
    algo: str = "hdbscan"
    dims: int = 64
    min_cluster_size: int = 20
    min_samples: int = 5
    k: int = 10
    block_ids: list[str] | None = None


def _explore_cache_key(req: ExploreRequest, block_ids_sorted: list[str]) -> str:
    ids_hash = hashlib.sha1(",".join(block_ids_sorted).encode()).hexdigest()[:12]  # noqa: S324
    parts = f"{req.algo}:{req.dims}:{req.min_cluster_size}:{req.min_samples}:{req.k}:{ids_hash}"
    return hashlib.sha1(parts.encode()).hexdigest()[:16]  # noqa: S324


@app.post("/api/explore")
def post_explore(req: ExploreRequest, db: str = str(DEFAULT_DB)):
    db_path = Path(db)
    if not db_path.exists():
        raise HTTPException(404, f"Database not found: {db_path}")

    conn = open_db(str(db_path))
    try:
        all_blocks = load_data_blocks(conn)
    finally:
        conn.close()

    if not all_blocks:
        raise HTTPException(404, "No data_blocks with embeddings found.")

    # Filter to requested subset
    if req.block_ids:
        id_set = set(req.block_ids)
        blocks = [b for b in all_blocks if b["id"] in id_set]
    else:
        blocks = all_blocks

    if len(blocks) < 5:
        raise HTTPException(400, f"Need at least 5 blocks to cluster, got {len(blocks)}.")

    # Check full-result cache
    sorted_ids = sorted(b["id"] for b in blocks)
    cache_key = _explore_cache_key(req, sorted_ids)
    cache_file = UMAP_CACHE_DIR / f"explore_{cache_key}.pkl"
    UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        log.info("Explore cache hit: %s", cache_key)
        with cache_file.open("rb") as f:
            result = pickle.load(f)  # noqa: S301
        result["cached"] = True
        return result

    log.info("Explore: %d blocks, algo=%s dims=%d", len(blocks), req.algo, req.dims)

    # Build + truncate embedding matrix
    matrix = np.stack([np.frombuffer(b["embedding"], dtype=np.float32).copy() for b in blocks])
    matrix = _truncate_normalize(matrix, req.dims)

    # Run clustering
    if req.algo == "hdbscan":
        try:
            import hdbscan as hdbscan_lib  # type: ignore
        except ImportError:
            raise HTTPException(500, "hdbscan not installed: pip install hdbscan") from None
        mcs = max(2, min(req.min_cluster_size, len(blocks) // 2))
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=req.min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(matrix)
    else:
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except ImportError:
            raise HTTPException(
                500, "scikit-learn not installed: pip install scikit-learn"
            ) from None
        k = min(req.k, len(blocks))
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(matrix)

    # UMAP 2D projection (cached by matrix hash — subset+dims → unique hash)
    log.info("Running UMAP on explore subset: %d × %d", *matrix.shape)
    coords = _normalize_coords(compute_umap(matrix))

    # Build cluster stats
    stats: dict[str, dict] = {}
    for lbl in set(labels):
        if lbl == -1:
            continue
        members = [blocks[i] for i, label in enumerate(labels) if label == lbl]
        stats[str(lbl)] = {
            "count": len(members),
            "sample_titles": [b["source_path"].split("/")[-1] for b in members[:6]],
        }

    # Build output blocks
    output_blocks = []
    for i, b in enumerate(blocks):
        lbl = int(labels[i])
        date = None
        if b["block_time"]:
            with contextlib.suppress(ValueError):
                date = datetime.fromisoformat(b["block_time"][:19]).strftime("%Y-%m")
        output_blocks.append(
            {
                "id": b["id"],
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "label": str(lbl),
                "content": b["content"],
                "source_path": b["source_path"],
                "date": date,
            }
        )

    noise_count = int(sum(1 for label in labels if label == -1))
    result = {
        "blocks": output_blocks,
        "stats": stats,
        "noise_count": noise_count,
        "cluster_count": len(stats),
        "cached": False,
        "params": {
            "algo": req.algo,
            "dims": req.dims,
            "min_cluster_size": req.min_cluster_size,
            "min_samples": req.min_samples,
            "k": req.k,
        },
    }

    with cache_file.open("wb") as f:
        pickle.dump(result, f)

    return result


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
        coords = _normalize_coords(compute_umap(matrix))

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
            block_date = None
            if b["block_time"]:
                try:
                    dt = datetime.fromisoformat(b["block_time"][:19])
                    block_date = dt.strftime("%Y-%m")
                except ValueError:
                    pass

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
