"""Idea lineage — deterministic pre-compute for the idea-lineage lens.

Given a topic query: semantic search across ALL history, order the hits by
block_time, and bucket them into eras (quarters) — first mention, activity
per era, dormant gaps, last mention. The output is deterministic evidence;
the narrative (revisions, dead branches, current strongest form) belongs to
the lens (an agent) following Chris's "Persistent Memory Artifact" shape.

The JSON report is also the mobile timeline payload: `--write` drops it at
`<vault>/OpenAugi/lineage/<slug>.json`, next to context-pack.json in the
read contract, so the app can render an idea's timeline without new APIs.

See <vault>/OpenAugi/AGENT/lenses/idea-lineage.md and docs/reference/lenses.md.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openaugi.model.protocols import EmbeddingModel
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

# vec_blocks stores L2-normalized vectors, so distance ∈ [0, 2] with
# 0 = identical. Beyond ~1.2 the match is topical noise, not the same idea.
DEFAULT_MAX_DISTANCE = 1.2


def _slugify(query: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug[:60] or "lineage"


def _era_of(date: str) -> str:
    """ISO date → quarter label, e.g. '2025-05-05' → '2025-Q2'."""
    year, month = int(date[:4]), int(date[5:7])
    return f"{year}-Q{(month - 1) // 3 + 1}"


def _era_sequence(first: str, last: str) -> list[str]:
    """All quarter labels from first to last inclusive."""
    y, q = int(first[:4]), int(first[6])
    ly, lq = int(last[:4]), int(last[6])
    out = []
    while (y, q) <= (ly, lq):
        out.append(f"{y}-Q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def compute_lineage(
    store: SQLiteStore,
    embedding_model: EmbeddingModel,
    query: str,
    k: int = 150,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    snippet_len: int = 240,
    per_era: int = 6,
) -> dict[str, Any]:
    """Time-ordered semantic evidence for one idea.

    Returns {query, first_mention, last_mention, total_matches, eras, gaps}.
    Eras are quarters with matched blocks: count + strongest blocks (by
    similarity), each with date/title/snippet/source_path and a third_party
    flag (block carries a source/* tag — cited evidence, not Chris's voice).
    Gaps are quarter runs with zero matches between first and last mention.
    """
    query_vec = embedding_model.embed_query(query)
    hits = store.semantic_search(query_vec, k=k)
    if not hits:
        return {"query": query, "total_matches": 0, "eras": [], "gaps": []}

    ids = [bid for bid, dist in hits if dist <= max_distance]
    dist_by_id = dict(hits)

    placeholders = ",".join("?" * len(ids))
    rows = store.conn.execute(
        f"""SELECT id, title, content, block_time, tags,
                   json_extract(metadata, '$.source_path')
            FROM blocks
            WHERE id IN ({placeholders}) AND kind = 'data_block'""",
        ids,
    ).fetchall()

    entries: list[dict[str, Any]] = []
    undated = 0
    for bid, title, content, block_time, tags_json, source_path in rows:
        source_path = source_path or ""
        if source_path.startswith("OpenAugi/"):
            continue  # derived artifacts are not lineage evidence
        if not block_time:
            undated += 1
            continue
        tags = json.loads(tags_json) if tags_json else []
        entries.append(
            {
                "date": block_time[:10],
                "title": title or "",
                "snippet": (content or "")[:snippet_len],
                "distance": round(dist_by_id.get(bid, 0.0), 4),
                "source_path": source_path,
                "third_party": any(t.startswith("source/") for t in tags),
                "block_id": bid,
            }
        )

    entries.sort(key=lambda e: e["date"])
    if not entries:
        return {"query": query, "total_matches": 0, "undated": undated, "eras": [], "gaps": []}

    # Bucket into quarters
    buckets: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        buckets.setdefault(_era_of(e["date"]), []).append(e)

    eras = []
    for era in sorted(buckets):
        blocks = sorted(buckets[era], key=lambda e: e["distance"])[:per_era]
        eras.append(
            {
                "era": era,
                "count": len(buckets[era]),
                "blocks": sorted(blocks, key=lambda e: e["date"]),
            }
        )

    # Dormant gaps: quarters with no matches between first and last mention
    full = _era_sequence(eras[0]["era"], eras[-1]["era"])
    gaps: list[dict[str, Any]] = []
    run: list[str] = []
    for era in full:
        if era in buckets:
            if len(run) >= 2:  # a single quiet quarter is noise, not dormancy
                gaps.append({"from": run[0], "to": run[-1], "quarters": len(run)})
            run = []
        else:
            run.append(era)

    return {
        "query": query,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "first_mention": entries[0],
        "last_mention": entries[-1],
        "total_matches": len(entries),
        "undated": undated,
        "eras": eras,
        "gaps": gaps,
    }


def render_lineage_markdown(report: dict[str, Any]) -> str:
    """Human-readable evidence timeline — the CLI's default output."""
    lines = [f"# Idea lineage — {report['query']}"]
    if not report.get("total_matches"):
        lines.append("\nNo dated matches found.")
        return "\n".join(lines)

    first, last = report["first_mention"], report["last_mention"]
    lines.append(f"{report['total_matches']} matched blocks, {first['date']} → {last['date']}")
    for gap in report["gaps"]:
        lines.append(f"- dormant {gap['from']} → {gap['to']} ({gap['quarters']} quarters)")

    for era in report["eras"]:
        lines.append(f"\n## {era['era']}  ({era['count']} blocks)")
        for b in era["blocks"]:
            marker = " [third-party]" if b["third_party"] else ""
            snippet = b["snippet"].replace("\n", " ")[:120]
            lines.append(f"- **{b['date']}** — {b['title']}{marker}: {snippet}")
    return "\n".join(lines)


def write_lineage_sidecar(report: dict[str, Any], vault_path: str | Path) -> Path:
    """Write the JSON payload to <vault>/OpenAugi/lineage/<slug>.json.

    Same transport pattern as context-pack.json: the file is the API; the
    mobile bridge serves it, a future HTTP endpoint would serve the same
    builder's output.
    """
    out_dir = Path(vault_path) / "OpenAugi" / "lineage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slugify(report['query'])}.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote lineage sidecar: %s", out_path)
    return out_path
