#!/usr/bin/env python3
"""Remove blocks ingested from a nested Claude Code worktree.

WHY THIS EXISTS (2026-09-03): a Claude Code session created a git worktree at
`<vault>/.claude/worktrees/<name>/`, which checked out a second full copy of
the vault *inside* the vault. `exclude_patterns` covered `.obsidian/**` and
`.git/**` but not `.claude/**`, so ingest treated 6011 duplicated files as
new. Blocks went 32947 -> 63826, every historical `zzz:` marker resurfaced as
a pending task, and the task watcher launched 104 headless agents.

WHY SURGICAL DELETE AND NOT A REBUILD: re-ingesting from the vault would
rebuild blocks and embeddings, but `routed_to` edges are decisions recorded
nowhere else — a rebuild silently drops them. Deleting only the blocks whose
`metadata.source_path` sits under `.claude/` leaves every routing decision
intact. The script asserts that afterwards rather than trusting it.

WHAT NEEDS EXPLICIT HANDLING:
  - `links` declares ON DELETE CASCADE, but SQLite ships `foreign_keys=OFF`,
    so cascade does NOT happen unless we ask for it. We delete links directly.
  - `blocks_fts` IS maintained by AFTER DELETE triggers on `blocks`.
  - `vec_blocks` (sqlite-vec) has NO trigger — its rows must be deleted by
    hand or the vector index keeps returning dead block ids.

Usage:
    purge_worktree_blocks.py            # dry run: report only, no writes
    purge_worktree_blocks.py --apply    # snapshot, then delete
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path.home() / ".openaugi" / "openaugi.db"
BACKUP_DIR = Path.home() / ".openaugi" / "backups"
PREFIX = ".claude/%"

TARGET_SQL = """
SELECT id FROM blocks
WHERE json_extract(metadata, '$.source_path') LIKE ?
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"!! could not load sqlite-vec ({e}); vec_blocks cannot be cleaned", file=sys.stderr)
        # Without sqlite-vec the vector rows cannot be deleted, and deleting
        # blocks alone would leave the index pointing at rows that no longer
        # exist. Refuse rather than half-clean.
        raise SystemExit(2) from e
    return conn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="perform the deletion")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args()

    conn = connect(args.db)
    cur = conn.cursor()

    # A temp table rather than bound parameters: there are ~30k target ids and
    # SQLite caps host variables well below that ("too many SQL variables").
    # It also lets every later statement reuse one definition of "target".
    cur.execute("CREATE TEMP TABLE purge_ids AS " + TARGET_SQL, (PREFIX,))
    n_targets = cur.execute("SELECT COUNT(*) FROM purge_ids").fetchone()[0]

    total = cur.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
    routed_before = cur.execute("SELECT COUNT(*) FROM links WHERE kind='routed_to'").fetchone()[0]

    if not n_targets:
        print("No .claude/* blocks found — nothing to purge.")
        return 0

    link_count = cur.execute(
        "SELECT COUNT(*) FROM links WHERE from_id IN (SELECT id FROM purge_ids) "
        "OR to_id IN (SELECT id FROM purge_ids)"
    ).fetchone()[0]
    routed_hit = cur.execute(
        "SELECT COUNT(*) FROM links WHERE kind='routed_to' "
        "AND (from_id IN (SELECT id FROM purge_ids) OR to_id IN (SELECT id FROM purge_ids))"
    ).fetchone()[0]

    print(f"blocks total          : {total}")
    print(f"blocks to delete      : {n_targets}")
    print(f"blocks remaining      : {total - n_targets}")
    print(f"links to delete       : {link_count}")
    print(f"routed_to total       : {routed_before}")
    print(f"routed_to affected    : {routed_hit}")

    if routed_hit:
        print("\n!! routed_to edges would be lost. Refusing — investigate first.", file=sys.stderr)
        return 1

    if not args.apply:
        print("\nDry run. Re-run with --apply to delete.")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    snap = BACKUP_DIR / f"openaugi-{datetime.now():%Y-%m-%d}-pre-worktree-purge.db"
    snap.unlink(missing_ok=True)
    print(f"\nsnapshotting to {snap.name} ...")
    conn.execute("VACUUM INTO ?", (str(snap),))

    print("deleting ...")
    with conn:
        conn.execute(
            "DELETE FROM links WHERE from_id IN (SELECT id FROM purge_ids) "
            "OR to_id IN (SELECT id FROM purge_ids)"
        )
        conn.execute("DELETE FROM vec_blocks WHERE block_id IN (SELECT id FROM purge_ids)")
        conn.execute("DELETE FROM blocks WHERE id IN (SELECT id FROM purge_ids)")

    remaining = cur.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
    routed_after = cur.execute("SELECT COUNT(*) FROM links WHERE kind='routed_to'").fetchone()[0]
    orphans = cur.execute(
        "SELECT COUNT(*) FROM blocks WHERE json_extract(metadata,'$.source_path') LIKE ?",
        (PREFIX,),
    ).fetchone()[0]

    print(f"blocks now            : {remaining}")
    print(f"routed_to now         : {routed_after}  (was {routed_before})")
    print(f"residual .claude rows : {orphans}")

    assert routed_after == routed_before, "routed_to edges changed — restore the snapshot"
    assert orphans == 0, "residual .claude blocks remain"

    print("compacting ...")
    conn.execute("VACUUM")
    conn.close()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
