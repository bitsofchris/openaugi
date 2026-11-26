"""Test helpers for the graph-integrity invariants.

- `assert_graph_integrity(store, vault)` fails with every error-level finding
  of `openaugi.pipeline.integrity.check_graph_integrity`. conftest runs it
  after every test that ingests into the `store` fixture, so a new ingest
  test exercises it for free (opt out with `@pytest.mark.graph_drift_expected`).
- `graph_snapshot(store)` is a path-level normal form of the graph, for
  "incremental ingest equals a fresh full ingest" comparisons: ids differ
  between two stores only where they should not, so everything is expressed
  in source paths and content hashes.

See docs/reference/graph-integrity.md.
"""

from __future__ import annotations

from pathlib import Path

from openaugi.pipeline.integrity import check_graph_integrity, errors
from openaugi.store.sqlite import SQLiteStore


def assert_graph_integrity(
    store: SQLiteStore,
    vault_root: str | Path | None = None,
    exclude_patterns: list[str] | None = None,
) -> None:
    found = errors(check_graph_integrity(store, vault_root, exclude_patterns))
    assert not found, "graph drifted from the vault:\n" + "\n".join(str(f) for f in found)


def graph_snapshot(store: SQLiteStore) -> dict[str, set]:
    """The graph as paths and hashes, comparable across two stores.

    - documents: (source_path, document id). The id is included on purpose:
      identity is part of what must agree.
    - entries: (source_path, content_hash, containing document's source_path)
    - edges: (kind, from note path, from content_hash, to note path) for
      links_to / routed_to, the target named by its document's source_path.
    """
    c = store.conn
    documents = set(
        c.execute(
            """SELECT json_extract(metadata, '$.source_path'), id FROM blocks
               WHERE kind = 'context_block:document' AND source = 'vault'"""
        ).fetchall()
    )
    entries = set(
        c.execute(
            """SELECT json_extract(b.metadata, '$.source_path'), b.content_hash,
                      json_extract(d.metadata, '$.source_path')
               FROM blocks b
               LEFT JOIN links l ON l.from_id = b.id AND l.kind = 'contains'
               LEFT JOIN blocks d ON d.id = l.to_id
               WHERE b.kind = 'data_block' AND b.source = 'vault'"""
        ).fetchall()
    )
    edges = set(
        c.execute(
            """SELECT l.kind, json_extract(a.metadata, '$.source_path'), a.content_hash,
                      json_extract(b.metadata, '$.source_path')
               FROM links l
               JOIN blocks a ON a.id = l.from_id
               JOIN blocks b ON b.id = l.to_id
               WHERE l.kind IN ('links_to', 'routed_to')"""
        ).fetchall()
    )
    return {"documents": documents, "entries": entries, "edges": edges}


def snapshot_diff(got: dict[str, set], want: dict[str, set]) -> str:
    """Readable difference between two snapshots ('' when equal)."""
    lines: list[str] = []
    for key in want:
        extra, missing = got[key] - want[key], want[key] - got[key]
        for row in sorted(extra, key=str):
            lines.append(f"  {key} only in incremental: {row}")
        for row in sorted(missing, key=str):
            lines.append(f"  {key} only in full: {row}")
    return "\n".join(lines)
