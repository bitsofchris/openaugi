"""Clustering pipeline — config-driven HDBSCAN cluster context blocks.

Each named pass in [[clustering.passes]] produces a set of
context_block:cluster nodes linked to their member data_blocks.
Passes can scope to "all" data_blocks or "within" the member set
of a parent pass's clusters, forming a hierarchy.

See docs/plans/hierarchical-embeddings.md for full design.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────


@dataclass
class ClusterPassConfig:
    """One clustering pass parsed from [[clustering.passes]] config."""

    id: str
    dims: int
    scope: str  # "all" | "within"
    # Required for hdbscan; unused for kmeans (set to 0 if omitted in config).
    min_cluster_size: int = 0
    description: str = ""
    # "hdbscan" — density-based, produces noise (-1); good for fine passes and
    #             cross-domain where noise itself is signal (bridge blocks).
    # "kmeans"  — centroid-based, assigns every point; good for coarse life-area
    #             passes where every document should belong to some area.
    type: str = "hdbscan"
    parent_pass: str | None = None
    min_samples: int | None = None
    store_centroid: bool = True
    bridge_detection: bool = False
    # "block"    — cluster individual data_block embeddings (default).
    # "document" — mean-pool all blocks per source document first, then cluster
    #              one vector per document. Better for coarse life-area passes:
    #              a 144-chunk podcast contributes one topical vector instead of
    #              144 near-identical vectors that dominate HDBSCAN density.
    input_level: str = "block"
    # Required for kmeans; unused for hdbscan.
    n_clusters: int | None = None
    # Which embedding column to use. "embedding" = title-prepended (default,
    # powers retrieval). "content_only_embedding" = pure content signal, better
    # for concept/idea clustering where title noise causes within-doc grouping.
    embedding_col: str = "embedding"

    def __post_init__(self) -> None:
        if self.scope == "within" and not self.parent_pass:
            raise ValueError(f"Pass '{self.id}': scope='within' requires parent_pass")
        if self.bridge_detection and self.scope == "within":
            raise ValueError(f"Pass '{self.id}': bridge_detection requires scope='all'")
        if self.input_level not in ("block", "document"):
            raise ValueError(f"Pass '{self.id}': input_level must be 'block' or 'document'")
        if self.type == "kmeans" and not self.n_clusters:
            raise ValueError(f"Pass '{self.id}': type='kmeans' requires n_clusters")
        if self.type == "hdbscan" and not self.min_cluster_size:
            raise ValueError(f"Pass '{self.id}': type='hdbscan' requires min_cluster_size")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterPassConfig:
        return cls(
            id=d["id"],
            dims=d["dims"],
            scope=d.get("scope", "all"),
            min_cluster_size=d.get("min_cluster_size", 0),
            description=d.get("description", ""),
            type=d.get("type", "hdbscan"),
            parent_pass=d.get("parent_pass"),
            min_samples=d.get("min_samples"),
            store_centroid=d.get("store_centroid", True),
            bridge_detection=d.get("bridge_detection", False),
            input_level=d.get("input_level", "block"),
            n_clusters=d.get("n_clusters"),
            embedding_col=d.get("embedding_col", "embedding"),
        )


def parse_cluster_passes(config: dict[str, Any]) -> list[ClusterPassConfig]:
    """Parse [[clustering.passes]] from config dict. Returns validated list."""
    raw = config.get("clustering", {}).get("passes", [])
    passes = [ClusterPassConfig.from_dict(d) for d in raw]
    ids = {p.id for p in passes}
    for p in passes:
        if p.parent_pass and p.parent_pass not in ids:
            raise ValueError(f"Pass '{p.id}' references unknown parent_pass '{p.parent_pass}'")
    return passes


# ── Data structures ────────────────────────────────────────────────


@dataclass
class ClusterResult:
    """Output of one HDBSCAN run (one scope=all pass, or one parent-cluster subset)."""

    pass_id: str
    parent_cluster_label: int | None  # None for scope=all
    block_ids: list[str]  # ordered to match labels
    labels: np.ndarray  # HDBSCAN labels; -1 = noise
    centroids: dict[int, np.ndarray]  # cluster_label -> centroid vector


@dataclass
class PassResult:
    """All results from one complete clustering pass."""

    pass_cfg: ClusterPassConfig
    results: list[ClusterResult]
    # cluster_title -> block_id; used by child passes to create contains links
    cluster_block_ids: dict[str, str]


# ── Core computations ──────────────────────────────────────────────


def load_embeddings(
    store: SQLiteStore,
    kind: str = "data_block",
    embedding_col: str = "embedding",
) -> dict[str, np.ndarray]:
    """Load all embedding blobs for blocks of kind. Returns {block_id: float32 vec}.

    embedding_col: "embedding" (title-prepended, default) or
                   "content_only_embedding" (content-only, better for clustering).
    """
    col = (
        embedding_col if embedding_col in ("embedding", "content_only_embedding") else "embedding"
    )
    rows = store.conn.execute(
        f"SELECT id, {col} FROM blocks WHERE kind = ? AND {col} IS NOT NULL",
        (kind,),
    ).fetchall()
    out: dict[str, np.ndarray] = {}
    for bid, blob in rows:
        if blob:
            out[bid] = np.frombuffer(blob, dtype=np.float32).copy()
    logger.info("Loaded %d embeddings (kind=%s, col=%s)", len(out), kind, col)
    return out


def load_document_embeddings(
    store: SQLiteStore,
    embedding_col: str = "embedding",
) -> dict[str, np.ndarray]:
    """Load one mean embedding per source document.

    Groups all data_block embeddings by metadata.source_path, mean-pools them,
    and returns {context_block:document_id: mean_vec}.  Keying by document ID
    means groups links written by _write_pass point to context_block:document
    blocks, which carry the document title and source path.

    Why mean-pool instead of just using the first block: a 144-chunk podcast
    transcript should contribute one vector representing the overall topic, not
    144 near-duplicate vectors that dominate HDBSCAN density estimation.
    """
    from openaugi.model.block import Block as BlockModel

    col = (
        embedding_col if embedding_col in ("embedding", "content_only_embedding") else "embedding"
    )
    rows = store.conn.execute(
        f"""SELECT id, {col}, metadata FROM blocks
            WHERE kind = 'data_block' AND {col} IS NOT NULL""",
    ).fetchall()
    source_vecs: dict[str, list[np.ndarray]] = {}
    for _bid, blob, meta_json in rows:
        if blob:
            meta = json.loads(meta_json) if meta_json else {}
            source_path = meta.get("source_path", "")
            if source_path:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
                source_vecs.setdefault(source_path, []).append(vec)

    # Only include documents whose context_block:document actually exists in the DB.
    # Data ingested before document blocks were added, or from deleted files, may
    # have data_blocks with no corresponding document block — skip those to avoid
    # FK violations when writing groups links.
    existing_doc_ids: set[str] = {
        r[0]
        for r in store.conn.execute(
            "SELECT id FROM blocks WHERE kind = 'context_block:document'"
        ).fetchall()
    }

    result: dict[str, np.ndarray] = {}
    skipped = 0
    for source_path, vecs in source_vecs.items():
        doc_id = BlockModel.make_document_id(source_path)
        if doc_id in existing_doc_ids:
            result[doc_id] = np.mean(np.stack(vecs), axis=0).astype(np.float32)
        else:
            skipped += 1

    if skipped:
        logger.debug("Skipped %d source paths with no matching context_block:document", skipped)
    logger.info(
        "Loaded %d document embeddings (mean-pooled from %d blocks, skipped %d orphaned sources)",
        len(result),
        sum(len(v) for v in source_vecs.values()),
        skipped,
    )
    return result


def truncate_and_normalize(vecs: dict[str, np.ndarray], dims: int) -> dict[str, np.ndarray]:
    """Slice each vector to [:dims] and L2-normalize in place. Returns new dict."""
    out: dict[str, np.ndarray] = {}
    for bid, vec in vecs.items():
        v = vec[:dims].copy()
        norm = np.linalg.norm(v)
        out[bid] = v / norm if norm > 0 else v
    return out


def _run_kmeans(
    ids: list[str],
    vecs: dict[str, np.ndarray],
    n_clusters: int,
) -> tuple[list[str], np.ndarray]:
    """Run faiss Kmeans. Every point gets a label — no noise (-1).

    faiss Kmeans is significantly faster than sklearn MiniBatchKMeans for
    large vector sets, particularly at higher dimensions, due to its BLAS-
    optimized distance computation and GPU-ready architecture.
    """
    import faiss

    valid_ids = [bid for bid in ids if bid in vecs]
    if not valid_ids:
        return [], np.array([], dtype=int)

    matrix = np.stack([vecs[bid] for bid in valid_ids]).astype(np.float32)
    d = matrix.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, seed=42, verbose=False)
    kmeans.train(matrix)
    _, assign = kmeans.index.search(matrix, 1)  # type: ignore[call-arg]
    labels = assign.reshape(-1).astype(np.int64)
    return valid_ids, labels


def _run_hdbscan(
    ids: list[str],
    vecs: dict[str, np.ndarray],
    min_cluster_size: int,
    min_samples: int | None,
) -> tuple[list[str], np.ndarray]:
    """Run HDBSCAN on the subset of ids present in vecs. Returns (ordered_ids, labels)."""
    from hdbscan import HDBSCAN

    valid_ids = [bid for bid in ids if bid in vecs]
    if not valid_ids:
        return [], np.array([], dtype=int)

    matrix = np.stack([vecs[bid] for bid in valid_ids])
    kwargs: dict[str, Any] = {"min_cluster_size": min_cluster_size, "metric": "euclidean"}
    if min_samples is not None:
        kwargs["min_samples"] = min_samples

    labels: np.ndarray = HDBSCAN(**kwargs).fit_predict(matrix)
    return valid_ids, labels


def _compute_centroids(
    block_ids: list[str],
    vecs: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Mean vector per cluster label. Skips noise (-1)."""
    buckets: dict[int, list[np.ndarray]] = {}
    for bid, label in zip(block_ids, labels, strict=False):
        if label == -1:
            continue
        buckets.setdefault(int(label), []).append(vecs[bid])
    return {lbl: np.mean(np.stack(vs), axis=0) for lbl, vs in buckets.items()}


def _compute_temporal(block_ids: list[str], store: SQLiteStore) -> dict[str, Any]:
    """Individual block timestamps for the given block IDs.

    Returns a sorted list of ISO date strings — one entry per block — so
    consumers can plot each block as an event on a timeline (scatter, KDE,
    frequency histogram at any resolution) rather than being locked into
    a pre-bucketed monthly granularity.
    """
    if not block_ids:
        return {}
    placeholders = ",".join("?" * len(block_ids))
    rows = store.conn.execute(
        f"SELECT block_time FROM blocks WHERE id IN ({placeholders})",
        block_ids,
    ).fetchall()

    dates = sorted(r[0][:10] for r in rows if r[0])
    if not dates:
        return {}

    return {
        "first_block": dates[0],
        "last_block": dates[-1],
        "block_timestamps": dates,
    }


def _detect_bridges(
    noise_ids: list[str],
    noise_vecs: list[np.ndarray],
    centroids: dict[int, np.ndarray],
    pass_id: str,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Surface noise points close to 2+ cluster centroids (cross-domain bridges)."""
    if not noise_ids or len(centroids) < 2:
        return []

    clabels = list(centroids.keys())
    centroid_matrix = np.stack([centroids[lbl] for lbl in clabels])  # (n_clusters, dims)
    bridges = []

    for bid, vec in zip(noise_ids, noise_vecs, strict=False):
        sims = centroid_matrix @ vec  # cosine sims (vecs normalized)
        top2_idx = np.argsort(sims)[-2:][::-1]
        top2_sims = sims[top2_idx]
        if top2_sims[1] >= threshold:
            bridges.append(
                {
                    "block_id": bid,
                    "pass_id": pass_id,
                    "near_clusters": [
                        {"label": clabels[top2_idx[0]], "similarity": float(top2_sims[0])},
                        {"label": clabels[top2_idx[1]], "similarity": float(top2_sims[1])},
                    ],
                }
            )
    return bridges


# ── ID / title helpers ─────────────────────────────────────────────


def _cluster_block_id(pass_id: str, label: int, parent_label: int | None = None) -> str:
    key = (
        f"cluster:{pass_id}:{parent_label}_{label}"
        if parent_label is not None
        else f"cluster:{pass_id}:{label}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _cluster_title(pass_id: str, label: int, parent_label: int | None = None) -> str:
    return (
        f"{pass_id}_{parent_label}_{label}" if parent_label is not None else f"{pass_id}_{label}"
    )


# ── DAG computation ────────────────────────────────────────────────


def _run_clustering(
    pass_cfg: ClusterPassConfig,
    ids: list[str],
    working_vecs: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Dispatch to the configured clustering algorithm."""
    if pass_cfg.type == "kmeans":
        return _run_kmeans(ids, working_vecs, pass_cfg.n_clusters)  # type: ignore[arg-type]
    return _run_hdbscan(ids, working_vecs, pass_cfg.min_cluster_size, pass_cfg.min_samples)


def _compute_pass(
    pass_cfg: ClusterPassConfig,
    all_vecs: dict[str, np.ndarray],
    parent_result: PassResult | None,
    doc_to_blocks: dict[str, list[str]] | None = None,
) -> list[ClusterResult]:
    """Run clustering for one pass. Pure computation — no DB writes.

    doc_to_blocks: required when the parent pass used input_level='document'
    (keyed by context_block:document IDs) but this pass uses input_level='block'
    (keyed by data_block IDs). Maps doc_id → [data_block_id, ...] so we can
    expand parent member IDs into the correct block-level IDs.
    """
    working_vecs = truncate_and_normalize(all_vecs, pass_cfg.dims)

    if pass_cfg.scope == "all":
        ids = list(all_vecs.keys())
        ordered_ids, labels = _run_clustering(pass_cfg, ids, working_vecs)
        centroids = _compute_centroids(ordered_ids, working_vecs, labels)
        n_clusters = len(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))
        logger.info(
            "Pass '%s': %d clusters, %d noise (%.1f%%)",
            pass_cfg.id,
            n_clusters,
            n_noise,
            100 * n_noise / max(len(labels), 1),
        )
        return [
            ClusterResult(
                pass_id=pass_cfg.id,
                parent_cluster_label=None,
                block_ids=ordered_ids,
                labels=labels,
                centroids=centroids,
            )
        ]

    # scope == "within"
    assert parent_result is not None
    sub_results: list[ClusterResult] = []

    for parent_cr in parent_result.results:
        parent_labels = sorted(set(parent_cr.labels.tolist()) - {-1})
        for parent_label in parent_labels:
            doc_ids = [
                bid
                for bid, lbl in zip(parent_cr.block_ids, parent_cr.labels, strict=False)
                if lbl == parent_label
            ]

            # Expand document IDs → block IDs when parent clustered at document
            # level but this pass clusters at block level.
            if doc_to_blocks is not None:
                member_ids = [
                    block_id for doc_id in doc_ids for block_id in doc_to_blocks.get(doc_id, [])
                ]
            else:
                member_ids = doc_ids

            if len(member_ids) < pass_cfg.min_cluster_size:
                logger.debug(
                    "Pass '%s': parent cluster %d has %d members (< min_cluster_size %d), skipping",  # noqa: E501
                    pass_cfg.id,
                    parent_label,
                    len(member_ids),
                    pass_cfg.min_cluster_size,
                )
                continue

            ordered_ids, labels = _run_clustering(pass_cfg, member_ids, working_vecs)
            centroids = _compute_centroids(ordered_ids, working_vecs, labels)
            n_clusters = len(set(labels) - {-1})
            logger.info(
                "Pass '%s' parent=%d: %d sub-clusters from %d blocks",
                pass_cfg.id,
                parent_label,
                n_clusters,
                len(member_ids),
            )
            sub_results.append(
                ClusterResult(
                    pass_id=pass_cfg.id,
                    parent_cluster_label=int(parent_label),
                    block_ids=ordered_ids,
                    labels=labels,
                    centroids=centroids,
                )
            )

    return sub_results


def _build_doc_to_blocks(store: SQLiteStore) -> dict[str, list[str]]:
    """Build a mapping from context_block:document ID → [data_block IDs].

    Used when a document-level parent pass (input_level='document') feeds a
    block-level child pass (input_level='block'). The parent's member IDs are
    context_block:document IDs; the child needs data_block IDs.

    Link direction: data_block --contains--> context_block:document
    """
    rows = store.conn.execute(
        """SELECT l.from_id AS block_id, l.to_id AS doc_id
           FROM links l
           JOIN blocks b ON b.id = l.from_id
           WHERE l.kind = 'contains'
             AND b.kind = 'data_block'"""
    ).fetchall()
    mapping: dict[str, list[str]] = {}
    for block_id, doc_id in rows:
        mapping.setdefault(doc_id, []).append(block_id)
    return mapping


def _propagate_doc_assignments(
    store: SQLiteStore,
    pass_id: str,
    doc_assignments: list[tuple[str, str]],
) -> None:
    """Propagate cluster labels from document IDs to their constituent data_blocks.

    document-level passes cluster context_block:document blocks, but downstream
    consumers (fine pass, retrieval) expect cluster_assignments on data_blocks.
    Walk the data_block --contains--> context_block:document links to find the
    right blocks and write the assignment to each.
    """
    # Build doc_id → label map, then fetch all matching data_blocks in one query.
    doc_label_map = {doc_id: label for doc_id, label in doc_assignments}
    placeholders = ",".join("?" * len(doc_label_map))
    rows = store.conn.execute(
        f"""SELECT l.from_id, l.to_id FROM links l
            JOIN blocks b ON b.id = l.from_id
            WHERE l.to_id IN ({placeholders})
              AND l.kind = 'contains'
              AND b.kind = 'data_block'""",
        list(doc_label_map.keys()),
    ).fetchall()
    block_assignments = [(from_id, doc_label_map[to_id]) for from_id, to_id in rows]
    if block_assignments:
        store.batch_update_cluster_assignments(pass_id, block_assignments)


def _write_pass(
    store: SQLiteStore,
    pass_cfg: ClusterPassConfig,
    results: list[ClusterResult],
    parent_cluster_block_ids: dict[str, str] | None,
    working_vecs: dict[str, np.ndarray],
) -> PassResult:
    """Write cluster blocks, links, and metadata updates to DB. Idempotent."""
    import base64

    deleted = store.delete_cluster_blocks_by_pass(pass_cfg.id)
    if deleted:
        logger.info("Deleted %d existing cluster blocks for pass '%s'", deleted, pass_cfg.id)

    cluster_blocks: list[Block] = []
    cluster_links: list[Link] = []
    cluster_block_ids: dict[str, str] = {}
    assignments: list[tuple[str, str]] = []  # (block_id, label_str)

    for result in results:
        unique_labels = sorted(set(result.labels.tolist()) - {-1})

        for label in unique_labels:
            member_ids = [
                bid
                for bid, lbl in zip(result.block_ids, result.labels, strict=False)
                if lbl == label
            ]
            block_id = _cluster_block_id(pass_cfg.id, label, result.parent_cluster_label)
            title = _cluster_title(pass_cfg.id, label, result.parent_cluster_label)
            cluster_block_ids[title] = block_id

            centroid_b64: str | None = None
            if pass_cfg.store_centroid and label in result.centroids:
                centroid_b64 = base64.b64encode(
                    result.centroids[label].astype(np.float32).tobytes()
                ).decode()

            temporal = _compute_temporal(member_ids, store)
            noise_count = int(np.sum(result.labels == -1))

            metadata: dict[str, Any] = {
                "pass_id": pass_cfg.id,
                "parent_pass_id": pass_cfg.parent_pass,
                "parent_cluster_label": result.parent_cluster_label,
                "cluster_label": int(label),
                "dims": pass_cfg.dims,
                "min_cluster_size": pass_cfg.min_cluster_size,
                "min_samples": pass_cfg.min_samples,
                "member_count": len(member_ids),
                "noise_count": noise_count,
                "centroid": centroid_b64,
                "temporal": temporal,
            }

            cluster_blocks.append(
                Block(
                    id=block_id,
                    kind="context_block:cluster",
                    title=title,
                    source="pipeline:cluster",
                    metadata=metadata,
                )
            )

            # groups links: cluster → data_block
            for mid in member_ids:
                cluster_links.append(Link(from_id=block_id, to_id=mid, kind="groups"))

            # contains link: parent cluster → this cluster
            if result.parent_cluster_label is not None and parent_cluster_block_ids:
                parent_title = _cluster_title(
                    pass_cfg.parent_pass,  # type: ignore[arg-type]
                    result.parent_cluster_label,
                )
                if parent_bid := parent_cluster_block_ids.get(parent_title):
                    cluster_links.append(Link(from_id=parent_bid, to_id=block_id, kind="contains"))

            # cluster assignment for each member data_block
            label_str = (
                f"{result.parent_cluster_label}_{label}"
                if result.parent_cluster_label is not None
                else str(label)
            )
            for mid in member_ids:
                assignments.append((mid, label_str))

    store.insert_blocks(cluster_blocks)
    store.insert_links(cluster_links)

    # For document-level passes the member IDs are context_block:document IDs,
    # not data_block IDs.  Propagate the assignment to each document's data_blocks
    # so the fine pass (which filters data_blocks by cluster_assignment) works.
    if pass_cfg.input_level == "document":
        _propagate_doc_assignments(store, pass_cfg.id, assignments)
    else:
        store.batch_update_cluster_assignments(pass_cfg.id, assignments)

    # Bridge detection (scope=all passes only)
    if pass_cfg.bridge_detection and results:
        result = results[0]
        noise_ids = [
            bid for bid, lbl in zip(result.block_ids, result.labels, strict=False) if lbl == -1
        ]
        if noise_ids:
            trunc_vecs = truncate_and_normalize(
                {bid: working_vecs[bid] for bid in noise_ids if bid in working_vecs},
                pass_cfg.dims,
            )
            noise_vecs_list = [trunc_vecs[bid] for bid in noise_ids if bid in trunc_vecs]
            valid_noise_ids = [bid for bid in noise_ids if bid in trunc_vecs]
            bridge_candidates = _detect_bridges(
                valid_noise_ids, noise_vecs_list, result.centroids, pass_cfg.id
            )
            if bridge_candidates:
                _write_bridges(store, bridge_candidates, cluster_block_ids)

    logger.info(
        "Pass '%s': wrote %d cluster blocks, %d links",
        pass_cfg.id,
        len(cluster_blocks),
        len(cluster_links),
    )

    return PassResult(
        pass_cfg=pass_cfg,
        results=results,
        cluster_block_ids=cluster_block_ids,
    )


def _write_bridges(
    store: SQLiteStore,
    bridge_candidates: list[dict[str, Any]],
    cluster_block_ids: dict[str, str],
) -> None:
    bridge_blocks: list[Block] = []
    for bc in bridge_candidates:
        bid = bc["block_id"]
        bridge_id = hashlib.sha256(f"bridge:{bc['pass_id']}:{bid}".encode()).hexdigest()[:16]
        near = []
        for n in bc["near_clusters"]:
            title = f"{bc['pass_id']}_{n['label']}"
            if cid := cluster_block_ids.get(title):
                near.append({"cluster_block_id": cid, "similarity": n["similarity"]})
        bridge_blocks.append(
            Block(
                id=bridge_id,
                kind="context_block:bridge",
                title=f"bridge_{bc['pass_id']}_{bid[:8]}",
                source="pipeline:cluster",
                metadata={
                    "source_block_id": bid,
                    "pass_id": bc["pass_id"],
                    "near_clusters": near,
                },
            )
        )
    store.insert_blocks(bridge_blocks)
    logger.info("Wrote %d bridge blocks", len(bridge_blocks))


# ── Topological sort ───────────────────────────────────────────────


def _topological_sort(passes: list[ClusterPassConfig]) -> list[ClusterPassConfig]:
    pass_map = {p.id: p for p in passes}
    order: list[ClusterPassConfig] = []
    visited: set[str] = set()

    def visit(p: ClusterPassConfig) -> None:
        if p.id in visited:
            return
        if p.parent_pass:
            visit(pass_map[p.parent_pass])
        visited.add(p.id)
        order.append(p)

    for p in passes:
        visit(p)
    return order


# ── Stats printing ─────────────────────────────────────────────────


def print_pass_stats(pass_cfg: ClusterPassConfig, results: list[ClusterResult]) -> None:
    all_labels: list[int] = []
    for r in results:
        all_labels.extend(r.labels.tolist())

    labels_arr = np.array(all_labels)
    unique_clusters = sorted(set(labels_arr.tolist()) - {-1})
    n_noise = int(np.sum(labels_arr == -1))
    total = len(labels_arr)
    noise_pct = 100 * n_noise / max(total, 1)

    sizes = []
    for r in results:
        for label in sorted(set(r.labels.tolist()) - {-1}):
            sizes.append(int(np.sum(r.labels == label)))

    print(
        f"\nPass: {pass_cfg.id}  "
        f"(dims={pass_cfg.dims}, min_cluster_size={pass_cfg.min_cluster_size})"
    )
    if pass_cfg.description:
        print(f"  {pass_cfg.description}")
    print(f"  Clusters : {len(unique_clusters)}")
    print(f"  Noise    : {n_noise} blocks ({noise_pct:.1f}%)")
    if sizes:
        print(f"  Sizes    : min={min(sizes)}  median={int(np.median(sizes))}  max={max(sizes)}")
    if pass_cfg.scope == "within":
        print(f"  Parents  : {len(results)} parent clusters processed")


# ── Grid exploration ───────────────────────────────────────────────


def explore_kmeans_grid(
    store: SQLiteStore,
    dims_list: list[int],
    k_list: list[int],
    n_samples: int = 8,
    input_level: str = "document",
) -> None:
    """Try every (dims, k) combination and print cluster summaries with sample titles.

    Pure exploration — writes nothing to DB. Use this to find a good (dims, k)
    before committing a config. Sample titles come from context_block:document
    for document-level, or data_block titles for block-level.

    Args:
        dims_list:   Matryoshka truncation sizes to try, e.g. [64, 96, 128, 256, 512].
        k_list:      Cluster counts to try, e.g. [5, 8, 10, 12, 15].
        n_samples:   Document titles to show per cluster (default 8).
        input_level: "document" (mean-pooled) or "block".
    """
    import faiss

    # Load embeddings once
    if input_level == "document":
        all_vecs = load_document_embeddings(store)
    else:
        all_vecs = load_embeddings(store)

    if not all_vecs:
        print("No embeddings found.")
        return

    # Fetch titles for all IDs in one query
    id_list = list(all_vecs.keys())
    placeholders = ",".join("?" * len(id_list))
    rows = store.conn.execute(
        f"SELECT id, title FROM blocks WHERE id IN ({placeholders})",
        id_list,
    ).fetchall()
    titles: dict[str, str] = {r[0]: r[1] or "" for r in rows}

    ids = list(all_vecs.keys())
    total = len(ids)
    W = 72  # column width

    print(f"\n{'═' * W}")
    print(f"  K-MEANS GRID EXPLORATION  |  {total} vectors ({input_level}-level)")
    print(f"{'═' * W}")

    for dims in dims_list:
        working_vecs = truncate_and_normalize(all_vecs, dims)
        matrix = np.stack([working_vecs[bid] for bid in ids]).astype(np.float32)
        d = matrix.shape[1]

        for k in k_list:
            if k >= total:
                print(f"\n[dims={dims}, k={k}] Skipped — k >= total vectors ({total})")
                continue

            kmeans = faiss.Kmeans(d, k, niter=20, seed=42, verbose=False)
            kmeans.train(matrix)
            _, assign = kmeans.index.search(matrix, 1)  # type: ignore[call-arg]
            labels = assign.reshape(-1).astype(int)

            # Build cluster → member IDs (in original order for stable samples)
            clusters: dict[int, list[str]] = {}
            for bid, label in zip(ids, labels, strict=False):
                clusters.setdefault(int(label), []).append(bid)

            sizes = [len(v) for v in clusters.values()]
            size_summary = (
                f"min={min(sizes)}  median={int(np.median(sizes))}"
                f"  max={max(sizes)}  total={total}"
            )

            print(f"\n{'─' * W}")
            print(f"  dims={dims}  k={k}  |  {size_summary}")
            print(f"{'─' * W}")

            for label in sorted(clusters.keys()):
                members = clusters[label]
                pct = 100 * len(members) / total
                print(f"\n  Cluster {label}  ({len(members)} docs, {pct:.1f}%)")
                for bid in members[:n_samples]:
                    t = titles.get(bid, bid[:12])
                    print(f"    · {t}")


def explore_fine_cluster(
    store: SQLiteStore,
    parent_pass_id: str,
    cluster_label: str,
    dims_list: list[int],
    hdbscan_min_sizes: list[int],
    k_list: list[int],
    n_samples: int = 8,
    embedding_col: str = "embedding",
) -> None:
    """Explore fine clustering within one coarse cluster — HDBSCAN and k-means at multiple dims.

    Loads data_block embeddings for members of a specific coarse cluster, then
    tries both HDBSCAN (noise = recurring-idea signal) and k-means (every block
    assigned) at each dims setting. Pure exploration — writes nothing to DB.

    Args:
        parent_pass_id:    Pass id of the coarse clustering (e.g. "life_areas").
        cluster_label:     Label string to match in cluster_assignments (e.g. "7").
        dims_list:         Dims to try, e.g. [1536, 3072].
        hdbscan_min_sizes: min_cluster_size values to try for HDBSCAN.
        k_list:            k values to try for k-means.
        n_samples:         Titles to show per cluster.
        embedding_col:     "embedding" (title-prepended) or "content_only_embedding".
    """
    import faiss

    # Load block IDs whose cluster_assignment for parent_pass matches the label
    rows = store.conn.execute(
        """SELECT id FROM blocks
           WHERE kind = 'data_block'
             AND json_extract(metadata, '$.cluster_assignments.' || ?) = ?""",
        (parent_pass_id, cluster_label),
    ).fetchall()
    member_ids = [r[0] for r in rows]

    if not member_ids:
        print(
            f"No data_blocks found with cluster_assignments.{parent_pass_id} = '{cluster_label}'"
        )
        return

    # Load embeddings for these blocks only
    col = (
        embedding_col if embedding_col in ("embedding", "content_only_embedding") else "embedding"
    )
    placeholders = ",".join("?" * len(member_ids))
    emb_rows = store.conn.execute(
        f"SELECT id, {col}, title FROM blocks WHERE id IN ({placeholders})",
        member_ids,
    ).fetchall()

    vecs: dict[str, np.ndarray] = {}
    titles: dict[str, str] = {}
    for bid, emb, title in emb_rows:
        if emb:
            vecs[bid] = np.frombuffer(emb, dtype=np.float32).copy()
        titles[bid] = title or ""

    logger.info(
        "explore_fine_cluster: %d members, %d with %s, cluster=%s",
        len(member_ids),
        len(vecs),
        col,
        cluster_label,
    )

    ids = [bid for bid in member_ids if bid in vecs]
    total = len(ids)
    W = 72

    print(f"\n{'═' * W}")
    print(
        f"  FINE CLUSTER EXPLORE  |  {parent_pass_id}[{cluster_label}]"
        f"  |  {total} blocks  |  {col}"
    )
    print(f"{'═' * W}")

    for dims in dims_list:
        working_vecs = truncate_and_normalize(vecs, dims)
        matrix = np.stack([working_vecs[bid] for bid in ids]).astype(np.float32)

        # ── HDBSCAN ───────────────────────────────────────────────
        for min_size in hdbscan_min_sizes:
            try:
                from hdbscan import HDBSCAN

                labels: np.ndarray = HDBSCAN(
                    min_cluster_size=min_size, metric="euclidean"
                ).fit_predict(matrix)

                clusters: dict[int, list[str]] = {}
                for bid, lbl in zip(ids, labels, strict=False):
                    clusters.setdefault(int(lbl), []).append(bid)

                n_clusters = len([lbl for lbl in clusters if lbl != -1])
                n_noise = len(clusters.get(-1, []))
                noise_pct = 100 * n_noise / max(total, 1)
                sizes = sorted([len(v) for lbl, v in clusters.items() if lbl != -1], reverse=True)
                size_summary = f"clusters={n_clusters}  noise={n_noise} ({noise_pct:.0f}%)" + (
                    f"  sizes {min(sizes)}–{max(sizes)}" if sizes else ""
                )

                print(f"\n{'─' * W}")
                print(f"  HDBSCAN  dims={dims}  min_cluster_size={min_size}  |  {size_summary}")
                print(f"{'─' * W}")

                for label in sorted(k for k in clusters if k != -1):
                    members = clusters[label]
                    pct = 100 * len(members) / total
                    print(f"\n  Cluster {label}  ({len(members)} blocks, {pct:.1f}%)")
                    for bid in members[:n_samples]:
                        print(f"    · {titles.get(bid, bid[:12])}")
                if -1 in clusters:
                    noise = clusters[-1]
                    print(f"\n  [noise]  ({len(noise)} blocks, {100 * len(noise) / total:.1f}%)")
                    for bid in noise[:n_samples]:
                        print(f"    · {titles.get(bid, bid[:12])}")

            except Exception as e:
                print(f"\n  HDBSCAN dims={dims} min_size={min_size}: ERROR — {e}")

        # ── k-means ───────────────────────────────────────────────
        for k in k_list:
            if k >= total:
                print(f"\n  k-means dims={dims} k={k}: Skipped — k >= total ({total})")
                continue

            kmeans = faiss.Kmeans(dims, k, niter=20, seed=42, verbose=False)
            kmeans.train(matrix)
            _, assign_km = kmeans.index.search(matrix, 1)  # type: ignore[call-arg]
            km_labels = assign_km.reshape(-1).astype(int)

            clusters_km: dict[int, list[str]] = {}
            for bid, lbl in zip(ids, km_labels, strict=False):
                clusters_km.setdefault(int(lbl), []).append(bid)

            sizes_km = sorted([len(v) for v in clusters_km.values()], reverse=True)
            size_summary_km = (
                f"min={min(sizes_km)}  median={int(np.median(sizes_km))}  max={max(sizes_km)}"
            )

            print(f"\n{'─' * W}")
            print(f"  k-means  dims={dims}  k={k}  |  {size_summary_km}")
            print(f"{'─' * W}")

            for label in sorted(clusters_km.keys()):
                members = clusters_km[label]
                pct = 100 * len(members) / total
                print(f"\n  Cluster {label}  ({len(members)} blocks, {pct:.1f}%)")
                for bid in members[:n_samples]:
                    print(f"    · {titles.get(bid, bid[:12])}")


# ── Entry point ────────────────────────────────────────────────────


def run_cluster_dag(
    store: SQLiteStore,
    passes: list[ClusterPassConfig],
    dry_run: bool = False,
) -> dict[str, PassResult]:
    """Execute all passes in topological order. Returns {pass_id: PassResult}.

    dry_run=True: runs clustering math and prints stats but writes nothing to DB.
    """
    sorted_passes = _topological_sort(passes)

    # Cache embeddings keyed by (input_level, embedding_col).  Most runs use
    # one combination but mixed configs (e.g. coarse=title-prepend,
    # fine=content-only) are valid and each needs its own loaded array.
    _vec_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def _get_vecs(input_level: str, embedding_col: str) -> dict[str, np.ndarray]:
        key = (input_level, embedding_col)
        if key not in _vec_cache:
            if input_level == "document":
                _vec_cache[key] = load_document_embeddings(store, embedding_col=embedding_col)
            else:
                _vec_cache[key] = load_embeddings(store, embedding_col=embedding_col)
        return _vec_cache[key]

    # Verify at least one pass has data
    first_pass = sorted_passes[0] if sorted_passes else None
    if first_pass and not _get_vecs(first_pass.input_level, first_pass.embedding_col):
        logger.warning("No embeddings found — run 'openaugi re-embed' first")
        return {}

    # Build doc→blocks mapping once if any scope=within pass mixes document-level
    # parent with block-level child.  Lazy: only if actually needed.
    _doc_to_blocks: dict[str, list[str]] | None = None

    def _get_doc_to_blocks() -> dict[str, list[str]]:
        nonlocal _doc_to_blocks
        if _doc_to_blocks is None:
            _doc_to_blocks = _build_doc_to_blocks(store)
        return _doc_to_blocks

    pass_results: dict[str, PassResult] = {}

    for pass_cfg in sorted_passes:
        all_vecs = _get_vecs(pass_cfg.input_level, pass_cfg.embedding_col)
        logger.info(
            "Computing pass: %s (scope=%s, dims=%d, input=%s)",
            pass_cfg.id,
            pass_cfg.scope,
            pass_cfg.dims,
            pass_cfg.input_level,
        )
        parent_result = pass_results.get(pass_cfg.parent_pass) if pass_cfg.parent_pass else None

        # Resolve level mismatch: parent clustered at document level, this pass
        # needs block IDs.  Pass the mapping so _compute_pass can expand.
        doc_to_blocks: dict[str, list[str]] | None = None
        if (
            pass_cfg.scope == "within"
            and parent_result is not None
            and parent_result.pass_cfg.input_level == "document"
            and pass_cfg.input_level == "block"
        ):
            doc_to_blocks = _get_doc_to_blocks()

        results = _compute_pass(pass_cfg, all_vecs, parent_result, doc_to_blocks)

        print_pass_stats(pass_cfg, results)

        if not dry_run:
            working_vecs = truncate_and_normalize(all_vecs, pass_cfg.dims)
            parent_block_ids = (
                pass_results[pass_cfg.parent_pass].cluster_block_ids
                if pass_cfg.parent_pass
                else None
            )
            pass_result = _write_pass(store, pass_cfg, results, parent_block_ids, working_vecs)
        else:
            # dry_run: build a PassResult without DB writes for child passes to reference
            pass_result = PassResult(
                pass_cfg=pass_cfg,
                results=results,
                cluster_block_ids={},
            )

        pass_results[pass_cfg.id] = pass_result

    if dry_run:
        print("\n[dry-run] No data written to DB.")

    return pass_results
