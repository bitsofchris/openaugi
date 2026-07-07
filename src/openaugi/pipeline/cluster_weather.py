"""Cluster weather — deterministic pre-compute for the cluster-weather lens.

Two pieces:

1. ``snapshot_cluster_run(store, pass_results)`` — called after each committed
   ``openaugi cluster`` run. Writes one ``context_block:cluster_run`` block
   whose metadata records, per pass, each cluster's label, member IDs, and
   member count. Cluster blocks themselves are *replaced* on every run;
   snapshots are the append-only history that makes diffing possible.

2. ``compute_weather(store, window_days)`` — diffs the latest snapshot of each
   pass against the most recent snapshot older than the window. Clusters are
   matched across runs by member overlap (Jaccard), because k-means labels are
   not stable between runs. Per-cluster recent activity comes from the live
   cluster blocks' ``temporal.block_timestamps``.

The output is deterministic data (counts, deltas, titles). Interpretation —
which movements are worth a Dashboard nomination — belongs to the
cluster-weather lens (an agent), not this module.

See docs/clustering.md ("Cluster weather") and the lens spec at
<vault>/OpenAugi/AGENT/lenses/cluster-weather.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from openaugi.model.block import Block
from openaugi.store.sqlite import SQLiteStore

if TYPE_CHECKING:
    from openaugi.pipeline.cluster import PassResult

logger = logging.getLogger(__name__)

# Two clusters from different runs are "the same cluster" when membership
# overlaps enough: Jaccard for the general case, containment for clusters that
# grew or shrank a lot (a small old cluster fully inside a bigger new one is
# growth, not death + birth). Below both thresholds → died + born.
JACCARD_THRESHOLD = 0.5
CONTAINMENT_THRESHOLD = 0.7


# ── Snapshot (write side) ──────────────────────────────────────────


def snapshot_cluster_run(
    store: SQLiteStore,
    pass_results: dict[str, PassResult],
    run_at: str | None = None,
) -> str:
    """Persist one cluster_run snapshot block for a committed cluster run.

    Metadata shape:
        {"run_at": iso, "passes": {pass_id: [
            {"label": "5_3", "title": "concepts_5_3",
             "member_count": 90, "members": [block_or_doc_ids...]}
        ]}}

    Members are whatever IDs the pass clustered (data_block IDs for
    block-level passes, context_block:document IDs for document-level ones);
    diffing only needs them to be consistent between runs of the same pass.
    Returns the snapshot block ID.
    """
    run_at = run_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    passes_snapshot: dict[str, list[dict[str, Any]]] = {}
    for pass_id, pr in pass_results.items():
        clusters: list[dict[str, Any]] = []
        for result in pr.results:
            labels_set = sorted(set(result.labels.tolist()) - {-1})
            for label in labels_set:
                members = [
                    bid
                    for bid, lbl in zip(result.block_ids, result.labels, strict=False)
                    if lbl == label
                ]
                label_str = (
                    f"{result.parent_cluster_label}_{label}"
                    if result.parent_cluster_label is not None
                    else str(label)
                )
                clusters.append(
                    {
                        "label": label_str,
                        "title": f"{pass_id}_{label_str}",
                        "member_count": len(members),
                        "members": members,
                    }
                )
        passes_snapshot[pass_id] = clusters

    block_id = hashlib.sha256(f"cluster_run:{run_at}".encode()).hexdigest()[:16]
    store.insert_blocks(
        [
            Block(
                id=block_id,
                kind="context_block:cluster_run",
                title=f"cluster_run_{run_at}",
                source="pipeline:cluster",
                block_time=run_at,
                metadata={"run_at": run_at, "passes": passes_snapshot},
            )
        ]
    )
    logger.info(
        "Snapshot %s: %s",
        block_id,
        {p: len(c) for p, c in passes_snapshot.items()},
    )
    return block_id


def load_snapshots(store: SQLiteStore) -> list[dict[str, Any]]:
    """All cluster_run snapshots, newest first. Each: {run_at, passes}."""
    rows = store.conn.execute(
        """SELECT metadata FROM blocks
           WHERE kind = 'context_block:cluster_run'
           ORDER BY block_time DESC"""
    ).fetchall()
    return [json.loads(r[0]) for r in rows if r[0]]


# ── Weather (read side) ────────────────────────────────────────────


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _match_clusters(
    current: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Greedy best-first matching by member Jaccard.

    Returns (current, baseline) pairs; (cur, None) = born, (None, base) = died.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, cur in enumerate(current):
        cur_members = set(cur["members"])
        for j, base in enumerate(baseline):
            base_members = set(base["members"])
            jac = _jaccard(cur_members, base_members)
            overlap = len(cur_members & base_members)
            containment = overlap / min(len(cur_members), len(base_members) or 1) if overlap else 0
            if jac >= JACCARD_THRESHOLD or containment >= CONTAINMENT_THRESHOLD:
                pairs.append((jac, i, j))
    pairs.sort(reverse=True)

    matched_cur: set[int] = set()
    matched_base: set[int] = set()
    out: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    for _sim, i, j in pairs:
        if i in matched_cur or j in matched_base:
            continue
        matched_cur.add(i)
        matched_base.add(j)
        out.append((current[i], baseline[j]))

    out.extend((current[i], None) for i in range(len(current)) if i not in matched_cur)
    out.extend((None, baseline[j]) for j in range(len(baseline)) if j not in matched_base)
    return out


def _load_live_cluster_meta(store: SQLiteStore) -> dict[str, dict[str, Any]]:
    """Live cluster block metadata keyed by cluster title (e.g. 'concepts_5_3')."""
    rows = store.conn.execute(
        "SELECT title, metadata FROM blocks WHERE kind = 'context_block:cluster'"
    ).fetchall()
    return {r[0]: json.loads(r[1]) for r in rows if r[0] and r[1]}


def _resolve_titles(store: SQLiteStore, ids: list[str]) -> dict[str, str]:
    """Map block/document IDs to titles (chunked to stay under SQLite param limits)."""
    titles: dict[str, str] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = store.conn.execute(
            f"SELECT id, title FROM blocks WHERE id IN ({placeholders})",
            chunk,
        ).fetchall()
        titles.update({r[0]: r[1] or "" for r in rows})
    return titles


def compute_weather(
    store: SQLiteStore,
    window_days: int = 14,
    sample_titles: int = 5,
) -> dict[str, Any]:
    """Deterministic weather report over all clustering passes.

    For each pass: take its latest snapshot as "current", and as "baseline"
    the most recent snapshot at least window_days older than current (falling
    back to the oldest available). Report per cluster:

    - status: grew | shrank | stable | born | died
    - delta / added / removed (member counts vs baseline)
    - recent_activity: member data_blocks written inside the window
      (from live cluster block temporal.block_timestamps)
    - last_block, new_member_titles, sample_titles (for the lens to name it)

    With a single snapshot there is no baseline: every cluster reports
    status "stable" with no delta, but recent_activity still works — the
    first run already yields usable weather.
    """
    snapshots = load_snapshots(store)
    live_meta = _load_live_cluster_meta(store)
    now = datetime.now(UTC)
    cutoff_date = (now - timedelta(days=window_days)).strftime("%Y-%m-%d")

    report: dict[str, Any] = {
        "window_days": window_days,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "passes": {},
    }
    if not snapshots:
        report["note"] = "No cluster_run snapshots — run 'openaugi cluster' first."
        return report

    all_pass_ids = {p for s in snapshots for p in s.get("passes", {})}

    for pass_id in sorted(all_pass_ids):
        with_pass = [s for s in snapshots if pass_id in s.get("passes", {})]
        current_snap = with_pass[0]  # newest first
        current_dt = datetime.strptime(current_snap["run_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=UTC
        )
        window_edge = current_dt - timedelta(days=window_days)
        older = [
            s
            for s in with_pass[1:]
            if datetime.strptime(s["run_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            <= window_edge
        ]
        # Prefer a baseline a full window back; otherwise the oldest we have.
        baseline_snap = older[0] if older else (with_pass[-1] if len(with_pass) > 1 else None)

        current = current_snap["passes"][pass_id]
        baseline = baseline_snap["passes"][pass_id] if baseline_snap else []
        matches = (
            _match_clusters(current, baseline) if baseline_snap else [(c, None) for c in current]
        )

        # One title-lookup for everything this pass may print
        ids_to_resolve: list[str] = []
        for cur, base in matches:
            if cur:
                ids_to_resolve.extend(cur["members"][:sample_titles])
                if base:
                    added = set(cur["members"]) - set(base["members"])
                    ids_to_resolve.extend(list(added)[:10])
        titles = _resolve_titles(store, ids_to_resolve)

        clusters_out: list[dict[str, Any]] = []
        for cur, base in matches:
            if cur is None and base is not None:
                clusters_out.append(
                    {
                        "cluster": base["title"],
                        "status": "died",
                        "member_count": 0,
                        "delta": -base["member_count"],
                        "recent_activity": 0,
                    }
                )
                continue
            assert cur is not None

            entry: dict[str, Any] = {
                "cluster": cur["title"],
                "member_count": cur["member_count"],
            }
            if base is None and baseline_snap is not None:
                entry["status"] = "born"
                entry["delta"] = cur["member_count"]
            elif base is None:
                entry["status"] = "stable"  # no baseline yet — first run
                entry["delta"] = 0
            else:
                added = set(cur["members"]) - set(base["members"])
                removed = set(base["members"]) - set(cur["members"])
                delta = cur["member_count"] - base["member_count"]
                entry["delta"] = delta
                entry["added"] = len(added)
                entry["removed"] = len(removed)
                entry["status"] = "grew" if delta > 0 else ("shrank" if delta < 0 else "stable")
                if added:
                    entry["new_member_titles"] = [
                        titles[a] for a in list(added)[:10] if titles.get(a)
                    ]

            meta = live_meta.get(cur["title"], {})
            timestamps: list[str] = meta.get("temporal", {}).get("block_timestamps", []) or []
            entry["recent_activity"] = sum(1 for t in timestamps if t >= cutoff_date)
            if timestamps:
                entry["last_block"] = timestamps[-1]
            entry["sample_titles"] = [
                titles[m] for m in cur["members"][:sample_titles] if titles.get(m)
            ]
            clusters_out.append(entry)

        clusters_out.sort(
            key=lambda c: (abs(c.get("delta", 0)), c["recent_activity"]), reverse=True
        )
        report["passes"][pass_id] = {
            "current_run": current_snap["run_at"],
            "baseline_run": baseline_snap["run_at"] if baseline_snap else None,
            "clusters": clusters_out,
        }

    return report


def render_weather_markdown(report: dict[str, Any], top: int = 10) -> str:
    """Human-readable report — the CLI's default output."""
    lines = [
        f"# Cluster weather — {report['generated_at']}",
        f"Window: {report['window_days']} days",
    ]
    if note := report.get("note"):
        lines.append(f"\n{note}")
        return "\n".join(lines)

    for pass_id, pdata in report["passes"].items():
        baseline = pdata["baseline_run"] or "none (first run — deltas unavailable)"
        lines.append(f"\n## {pass_id}  (run {pdata['current_run']}, baseline {baseline})")
        for c in pdata["clusters"][:top]:
            delta = c.get("delta", 0)
            delta_s = f"{delta:+d}" if delta else "±0"
            lines.append(
                f"\n- **{c['cluster']}** [{c['status']}] "
                f"{c['member_count']} members ({delta_s}), "
                f"{c['recent_activity']} blocks in window"
                + (f", last {c['last_block']}" if c.get("last_block") else "")
            )
            if c.get("new_member_titles"):
                lines.append(f"  - new: {'; '.join(c['new_member_titles'][:5])}")
            if c.get("sample_titles"):
                lines.append(f"  - sample: {'; '.join(c['sample_titles'][:3])}")
    return "\n".join(lines)
