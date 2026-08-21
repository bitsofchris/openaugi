#!/usr/bin/env python3
"""Re-key container documents from path-derived ids to `augi_id`-derived ones.

Run this ONCE, alongside the adapter change that made a document's id come
from its frontmatter `augi_id` rather than its file path.

**Why it is needed.** Every edge into a container — `contains` and `routed_to`
alike — is keyed on the document block's id. Before the adapter change that id
was `sha256("doc:" + rel_path)`, so a note that was renamed or moved got a new
id and every edge into it was orphaned: no error, no repair, the container just
looked emptier than it should. After the change the id is
`sha256("doc:augi:" + augi_id)`, which travels in the note.

**Which means the change itself re-keys every existing container**, orphaning
exactly the edges it exists to protect. This script does the remap, so run it
in the same step — before anything re-ingests the vault. Without it, routing
history survives in the DB but points at ids nothing resolves.

Safe to re-run: a container already carrying its new id is skipped.

    python scripts/migrate_container_augi_id.py --vault ~/vault           # dry run
    python scripts/migrate_container_augi_id.py --vault ~/vault --apply

Take a database snapshot first:

    sqlite3 ~/.openaugi/openaugi.db ".backup 'openaugi-pre-migration.db'"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openaugi.adapters.splitter import _extract_augi_id  # noqa: E402


def document_id(name: str) -> str:
    """Mirror of `Block.make_document_id` — kept local so the migration is
    readable on its own and cannot drift silently if that helper moves."""
    return hashlib.sha256(f"doc:{name}".encode()).hexdigest()[:16]


def plan(vault: Path, conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Every note carrying an `augi_id` whose document is still path-keyed.

    Returns (old_id, new_id, rel_path). A note whose new id is already in the
    blocks table is skipped: the migration has already run for it.
    """
    out: list[tuple[str, str, str]] = []
    for path in vault.rglob("*.md"):
        if any(part.startswith(".") for part in path.relative_to(vault).parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        augi_id = _extract_augi_id(content)
        if not augi_id:
            continue
        rel = str(path.relative_to(vault))
        old, new = document_id(rel), document_id(f"augi:{augi_id}")
        if old == new:
            continue
        if conn.execute("SELECT 1 FROM blocks WHERE id = ?", (new,)).fetchone():
            continue  # already migrated
        if not conn.execute("SELECT 1 FROM blocks WHERE id = ?", (old,)).fetchone():
            continue  # never ingested; the adapter will key it correctly
        out.append((old, new, rel))
    return out


def edge_count(conn: sqlite3.Connection, block_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM links WHERE to_id = ? OR from_id = ?", (block_id, block_id)
    ).fetchone()[0]


def migrate(conn: sqlite3.Connection, rows: list[tuple[str, str, str]]) -> None:
    """Re-key in one transaction, then verify every edge survived.

    Foreign keys are disabled for the rewrite: `links` and `recaps` reference
    `blocks(id)` with ON DELETE CASCADE and no ON UPDATE, so updating the
    parent id first would otherwise be refused. The verification below is what
    actually guarantees the result, not the constraint.
    """
    before = {old: edge_count(conn, old) for old, _, _ in rows}

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        for old, new, _ in rows:
            conn.execute("UPDATE blocks SET id = ? WHERE id = ?", (new, old))
            conn.execute("UPDATE links SET to_id = ? WHERE to_id = ?", (new, old))
            conn.execute("UPDATE links SET from_id = ? WHERE from_id = ?", (new, old))
            conn.execute("UPDATE recaps SET container_id = ? WHERE container_id = ?", (new, old))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    # The FTS index is external-content over blocks.rowid, which the rewrite
    # does not change — but it stores the id column, so it goes stale.
    conn.execute("INSERT INTO blocks_fts(blocks_fts) VALUES('rebuild')")
    conn.commit()

    failures = []
    for old, new, rel in rows:
        if edge_count(conn, new) != before[old]:
            failures.append(f"{rel}: {before[old]} edges before, {edge_count(conn, new)} after")
        if edge_count(conn, old) != 0:
            failures.append(f"{rel}: {edge_count(conn, old)} edges still on the old id")
    orphans = conn.execute(
        """SELECT COUNT(*) FROM links l
           LEFT JOIN blocks a ON a.id = l.from_id
           LEFT JOIN blocks b ON b.id = l.to_id
           WHERE a.id IS NULL OR b.id IS NULL"""
    ).fetchone()[0]
    if orphans:
        failures.append(f"{orphans} links now point at a block that does not exist")
    if failures:
        raise SystemExit("MIGRATION VERIFY FAILED:\n  " + "\n  ".join(failures))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault", required=True, type=Path)
    ap.add_argument("--db", type=Path, default=Path(os.path.expanduser("~/.openaugi/openaugi.db")))
    ap.add_argument("--apply", action="store_true", help="without this, prints the plan and exits")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = plan(args.vault.expanduser(), conn)
    if not rows:
        print("nothing to migrate — every augi_id container is already re-keyed")
        return

    total = 0
    for old, new, rel in rows:
        n = edge_count(conn, old)
        total += n
        print(f"{old} -> {new}  {n:>5} edges  {rel}")
    print(f"\n{len(rows)} containers, {total} edges")

    if not args.apply:
        print("\ndry run — pass --apply to write (snapshot the DB first)")
        return

    migrate(conn, rows)
    print("\nmigrated and verified: every edge preserved, no orphans")


if __name__ == "__main__":
    main()
