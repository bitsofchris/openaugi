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
import threading
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
EXPLORER_CONFIG = Path.home() / ".openaugi" / "explorer_config.json"


def load_explorer_config() -> dict:
    """Load explorer_config.json from ~/.openaugi/. Returns {} if missing."""
    if EXPLORER_CONFIG.exists():
        try:
            with EXPLORER_CONFIG.open() as f:
                cfg = json.load(f)
            log.info("Explorer config: %s", EXPLORER_CONFIG)
            return cfg
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read explorer config: %s", e)
    return {}


_EXPLORER_CONFIG = load_explorer_config()
_INCLUDE_FOLDERS: list[str] = _EXPLORER_CONFIG.get("include_folders", [])

# One UMAP runs at a time — Numba's workqueue is not re-entrant.
# A second request hitting the same cache key will wait, then get a cache hit.
_umap_lock = threading.Lock()

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


def _in_include_folders(source_path: str | None) -> bool:
    """Return True if source_path starts with one of the configured include folders.

    If _INCLUDE_FOLDERS is empty, all blocks pass (no filter).
    source_path is vault-relative, e.g. "1-notes/MyNote.md".
    """
    if not _INCLUDE_FOLDERS or not source_path:
        return True
    # Normalise: strip leading slash, compare first path component
    sp = source_path.lstrip("/")
    return any(sp == folder or sp.startswith(folder + "/") for folder in _INCLUDE_FOLDERS)


def load_data_blocks(conn: sqlite3.Connection) -> list[dict]:
    """Load data_blocks that have embeddings, filtered by include_folders config."""
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
    skipped = 0
    for r in rows:
        sp = r["source_path"] or r["title"] or r["id"]
        if not _in_include_folders(r["source_path"]):
            skipped += 1
            continue
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
                "embedding": r["embedding"],
                "source_path": sp,
                "cluster_assignments": ca,
            }
        )
    if skipped:
        log.info(
            "Folder filter: kept %d blocks, skipped %d (not in include_folders)",
            len(result),
            skipped,
        )
    return result


def load_source_content(conn: sqlite3.Connection, source_paths: set[str]) -> dict[str, str]:
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
                "temporal": meta.get("temporal"),
            }
        )
    return result


def discover_passes(cluster_blocks: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
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


# ── Embedding helpers ─────────────────────────────────────────────────────────


def _matrix_hash(matrix: np.ndarray) -> str:
    """SHA-1 of raw bytes — unique per (content × dims) combination."""
    return hashlib.sha1(matrix.tobytes()).hexdigest()[:16]  # noqa: S324


def _truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    """Slice to dims and L2-normalize (required after Matryoshka truncation)."""
    d = min(dims, matrix.shape[1])
    m = matrix[:, :d].copy()
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return m / norms


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize 2D projection to [-1, 1] per axis."""
    coords = coords.copy()
    for axis in range(2):
        col = coords[:, axis]
        col_range = col.max() - col.min()
        if col_range > 0:
            coords[:, axis] = (col - col.min()) / col_range * 2 - 1
    return coords


# ── UMAP (cached, serialized) ─────────────────────────────────────────────────


def _run_umap(matrix: np.ndarray) -> np.ndarray:
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
    """UMAP to 2D, cached to disk, serialized via lock (Numba is not re-entrant).

    Cache key = SHA-1 of the truncated+normalized matrix bytes, so it is unique
    per (block set × dims). A second concurrent request waits on the lock, then
    gets a cache hit instead of crashing Numba.
    """
    UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _matrix_hash(matrix)
    cache_file = UMAP_CACHE_DIR / f"umap_{cache_key}_{len(matrix)}.pkl"

    # Fast path: cache hit before acquiring lock
    if cache_file.exists():
        log.info("UMAP cache hit — %s", cache_file.name)
        with cache_file.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    with _umap_lock:
        # Re-check inside lock — another thread may have computed while we waited
        if cache_file.exists():
            log.info("UMAP cache hit (post-lock) — %s", cache_file.name)
            with cache_file.open("rb") as f:
                return pickle.load(f)  # noqa: S301

        log.info("UMAP computing: %d × %d (this takes a while first time)…", *matrix.shape)
        coords = _run_umap(matrix)
        with cache_file.open("wb") as f:
            pickle.dump(coords, f)
        log.info("UMAP cached → %s", cache_file.name)
        return coords


# ── Clustering helpers ────────────────────────────────────────────────────────


def _run_kmeans(matrix: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except ImportError:
        raise HTTPException(500, "scikit-learn not installed: pip install scikit-learn") from None
    k = min(k, len(matrix))
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(matrix)


def _run_hdbscan(matrix: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    try:
        import hdbscan as hdbscan_lib  # type: ignore
    except ImportError:
        raise HTTPException(500, "hdbscan not installed: pip install hdbscan") from None
    mcs = max(2, min(min_cluster_size, len(matrix) // 2))
    return hdbscan_lib.HDBSCAN(
        min_cluster_size=mcs, min_samples=min_samples, metric="euclidean"
    ).fit_predict(matrix)


def _cluster_stats(blocks: list[dict], labels: np.ndarray) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for lbl in set(labels):
        if lbl == -1:
            continue
        members = [blocks[i] for i, lab in enumerate(labels) if lab == lbl]
        stats[str(lbl)] = {
            "count": len(members),
            "sample_titles": [b["source_path"].split("/")[-1] for b in members[:6]],
        }
    return stats


# ── Explore endpoints ─────────────────────────────────────────────────────────


class UmapRequest(BaseModel):
    dims: int = 64
    block_ids: list[str] | None = None


class ClusterRequest(BaseModel):
    algo: str = "kmeans"
    dims: int = 64
    k: int = 10
    min_cluster_size: int = 20
    min_samples: int = 5
    block_ids: list[str] | None = None


def _load_blocks_for_request(block_ids: list[str] | None, db_path: Path) -> list[dict]:
    conn = open_db(str(db_path))
    try:
        all_blocks = load_data_blocks(conn)
    finally:
        conn.close()
    if not all_blocks:
        raise HTTPException(404, "No data_blocks with embeddings found.")
    if block_ids:
        id_set = set(block_ids)
        blocks = [b for b in all_blocks if b["id"] in id_set]
    else:
        blocks = all_blocks
    if len(blocks) < 5:
        raise HTTPException(400, f"Need at least 5 blocks, got {len(blocks)}.")
    return blocks


@app.post("/api/explore/umap")
def post_explore_umap(req: UmapRequest, db: str = str(DEFAULT_DB)):
    """Compute (or load cached) UMAP projection for a given dims + block subset.

    Returns {id, x, y, date} for each block. XY positions are stable for a given
    (dims, block_set) — they only change when dims or the block set changes.
    Call this once when dims changes; reuse coords when only k changes.
    """
    db_path = Path(db)
    if not db_path.exists():
        raise HTTPException(404, f"Database not found: {db_path}")

    blocks = _load_blocks_for_request(req.block_ids, db_path)

    full_matrix = np.stack(
        [np.frombuffer(b["embedding"], dtype=np.float32).copy() for b in blocks]
    )
    matrix = _truncate_normalize(full_matrix, req.dims)
    coords = _normalize_coords(compute_umap(matrix))

    points = []
    for i, b in enumerate(blocks):
        date = None
        if b["block_time"]:
            with contextlib.suppress(ValueError):
                date = datetime.fromisoformat(b["block_time"][:19]).strftime("%Y-%m")
        points.append(
            {
                "id": b["id"],
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "date": date,
            }
        )

    return {"points": points, "block_count": len(points), "dims": req.dims}


@app.post("/api/explore/cluster")
def post_explore_cluster(req: ClusterRequest, db: str = str(DEFAULT_DB)):
    """Run k-means or HDBSCAN on truncated embeddings. Fast — no UMAP.

    Returns {id → label} assignments and per-cluster stats.
    Combine with cached UMAP coords on the frontend to recolor the scatter.
    """
    db_path = Path(db)
    if not db_path.exists():
        raise HTTPException(404, f"Database not found: {db_path}")

    blocks = _load_blocks_for_request(req.block_ids, db_path)

    full_matrix = np.stack(
        [np.frombuffer(b["embedding"], dtype=np.float32).copy() for b in blocks]
    )
    matrix = _truncate_normalize(full_matrix, req.dims)

    # Check cluster cache
    UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids_hash = hashlib.sha1(  # noqa: S324
        ",".join(sorted(b["id"] for b in blocks)).encode()
    ).hexdigest()[:12]
    cache_key = hashlib.sha1(  # noqa: S324
        f"{req.algo}:{req.dims}:{req.k}:{req.min_cluster_size}:{req.min_samples}:{ids_hash}".encode()
    ).hexdigest()[:16]
    cache_file = UMAP_CACHE_DIR / f"cluster_{cache_key}.pkl"

    if cache_file.exists():
        log.info("Cluster cache hit: %s", cache_key)
        with cache_file.open("rb") as f:
            result = pickle.load(f)  # noqa: S301
        result["cached"] = True
        return result

    log.info("Clustering: algo=%s dims=%d k=%d n=%d", req.algo, req.dims, req.k, len(blocks))

    if req.algo == "hdbscan":
        labels = _run_hdbscan(matrix, req.min_cluster_size, req.min_samples)
    else:
        labels = _run_kmeans(matrix, req.k)

    stats = _cluster_stats(blocks, labels)
    noise_count = int(sum(1 for lab in labels if lab == -1))

    result = {
        "labels": {blocks[i]["id"]: str(int(labels[i])) for i in range(len(blocks))},
        "stats": stats,
        "noise_count": noise_count,
        "cluster_count": len(stats),
        "cached": False,
    }

    with cache_file.open("wb") as f:
        pickle.dump(result, f)

    return result


# ── Browse endpoint ───────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    """Return active explorer configuration (include_folders filter, etc.)."""
    return {
        "include_folders": _INCLUDE_FOLDERS,
        "config_path": str(EXPLORER_CONFIG),
    }


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
        full_matrix = np.stack(
            [np.frombuffer(b["embedding"], dtype=np.float32).copy() for b in data_blocks]
        )
        # Truncate+normalize at full dims so cache key matches explore mode prewarm
        matrix = _truncate_normalize(full_matrix, full_matrix.shape[1])

        log.info("Running UMAP on %d × %d matrix", *matrix.shape)
        coords = _normalize_coords(compute_umap(matrix))

        cluster_blocks = load_cluster_blocks(conn)
        source_paths = {b["source_path"] for b in data_blocks if b["source_path"]}
        source_content_map = load_source_content(conn, source_paths)

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

        passes = discover_passes(cluster_blocks)

        output_blocks = []
        for i, b in enumerate(data_blocks):
            block_date = None
            if b["block_time"]:
                with contextlib.suppress(ValueError):
                    block_date = datetime.fromisoformat(b["block_time"][:19]).strftime("%Y-%m")
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
