"""SQLite backend — blocks + links + FTS5.

Two tables. That's the whole store. WAL mode for concurrent reads
while the pipeline writes.

See docs/plans/m0.md § Data Model for the canonical schema.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from openaugi.model.block import Block
from openaugi.model.link import Link

logger = logging.getLogger(__name__)

# Schema version — bump when migrations are needed
SCHEMA_VERSION = 1

_BLOCKS_DDL = """
CREATE TABLE IF NOT EXISTS blocks (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    embedding BLOB,
    source TEXT,
    title TEXT,
    tags TEXT,                  -- JSON array
    timestamp TEXT,             -- ISO-8601
    occurred_at TEXT,
    metadata TEXT,              -- JSON object
    content_hash TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_LINKS_DDL = """
CREATE TABLE IF NOT EXISTS links (
    from_id TEXT NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    to_id TEXT NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    weight REAL,
    metadata TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (from_id, to_id, kind)
);
"""

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
    id UNINDEXED,
    title,
    content,
    tags,
    content='blocks',
    content_rowid='rowid'
);
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS blocks_ai AFTER INSERT ON blocks BEGIN
    INSERT INTO blocks_fts(rowid, id, title, content, tags)
    VALUES (new.rowid, new.id, COALESCE(new.title, ''), COALESCE(new.content, ''), COALESCE(new.tags, ''));
END;

CREATE TRIGGER IF NOT EXISTS blocks_ad AFTER DELETE ON blocks BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, id, title, content, tags)
    VALUES ('delete', old.rowid, old.id, COALESCE(old.title, ''), COALESCE(old.content, ''), COALESCE(old.tags, ''));
END;

CREATE TRIGGER IF NOT EXISTS blocks_au AFTER UPDATE ON blocks BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, id, title, content, tags)
    VALUES ('delete', old.rowid, old.id, COALESCE(old.title, ''), COALESCE(old.content, ''), COALESCE(old.tags, ''));
    INSERT INTO blocks_fts(rowid, id, title, content, tags)
    VALUES (new.rowid, new.id, COALESCE(new.title, ''), COALESCE(new.content, ''), COALESCE(new.tags, ''));
END;
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_blocks_kind ON blocks(kind);",
    "CREATE INDEX IF NOT EXISTS idx_blocks_source ON blocks(source);",
    "CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_blocks_content_hash ON blocks(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_id);",
    "CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_id);",
    "CREATE INDEX IF NOT EXISTS idx_links_kind ON links(kind);",
]


class SQLiteStore:
    """SQLite storage for blocks + links.

    Connection is lazy — opened on first access, can be closed and
    will auto-reopen. This allows read-only consumers (MCP server) to
    release the file lock between queries.
    """

    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = str(db_path)
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None

        if not read_only:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy connection — auto-reconnects after close()."""
        if self._conn is None:
            if self.read_only:
                uri = f"file:{self.db_path}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True)
            else:
                self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def close(self):
        """Close connection and release file lock."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _initialize(self):
        """Create tables, indexes, FTS, and triggers."""
        c = self.conn
        c.executescript(_BLOCKS_DDL)
        c.executescript(_LINKS_DDL)
        c.executescript(_FTS_DDL)
        c.executescript(_FTS_TRIGGERS)
        for idx_sql in _INDEXES:
            c.execute(idx_sql)
        c.commit()

    # ── Block CRUD ─────────────────────────────────────────────────

    def insert_block(self, block: Block) -> None:
        """Insert a single block. Ignores if ID already exists."""
        self.conn.execute(
            """INSERT OR IGNORE INTO blocks
               (id, kind, content, summary, embedding, source, title,
                tags, timestamp, occurred_at, metadata, content_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                block.id,
                block.kind,
                block.content,
                block.summary,
                block.embedding,
                block.source,
                block.title,
                block.tags_json(),
                block.timestamp,
                block.occurred_at,
                block.metadata_json(),
                block.content_hash,
                block.created_at,
            ),
        )

    def insert_blocks(self, blocks: list[Block]) -> int:
        """Batch insert blocks. Returns count inserted."""
        if not blocks:
            return 0
        self.conn.executemany(
            """INSERT OR IGNORE INTO blocks
               (id, kind, content, summary, embedding, source, title,
                tags, timestamp, occurred_at, metadata, content_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    b.id,
                    b.kind,
                    b.content,
                    b.summary,
                    b.embedding,
                    b.source,
                    b.title,
                    b.tags_json(),
                    b.timestamp,
                    b.occurred_at,
                    b.metadata_json(),
                    b.content_hash,
                    b.created_at,
                )
                for b in blocks
            ],
        )
        self.conn.commit()
        return len(blocks)

    def get_block(self, block_id: str) -> Block | None:
        """Fetch a block by ID."""
        row = self.conn.execute(
            """SELECT id, kind, content, summary, embedding, source, title,
                      tags, timestamp, occurred_at, metadata, content_hash, created_at
               FROM blocks WHERE id = ?""",
            (block_id,),
        ).fetchone()
        if not row:
            return None
        return _row_to_block(row)

    def get_blocks_by_kind(self, kind: str, limit: int = 100) -> list[Block]:
        """Fetch blocks by kind."""
        rows = self.conn.execute(
            """SELECT id, kind, content, summary, embedding, source, title,
                      tags, timestamp, occurred_at, metadata, content_hash, created_at
               FROM blocks WHERE kind = ? ORDER BY created_at DESC LIMIT ?""",
            (kind, limit),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    def delete_block(self, block_id: str) -> bool:
        """Delete a block by ID. CASCADE deletes its links."""
        cursor = self.conn.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_blocks_by_source_path(self, source_path: str) -> int:
        """Delete all blocks from a source path (via metadata). Returns count."""
        # Document blocks have source_path in metadata
        # Entry blocks link to doc via split_from — CASCADE handles them
        cursor = self.conn.execute(
            "DELETE FROM blocks WHERE id IN ("
            "  SELECT id FROM blocks WHERE kind = 'document' "
            "  AND json_extract(metadata, '$.source_path') = ?"
            ")",
            (source_path,),
        )
        self.conn.commit()
        return cursor.rowcount

    # ── Link CRUD ──────────────────────────────────────────────────

    def insert_link(self, link: Link) -> None:
        """Insert a single link. Ignores if already exists."""
        self.conn.execute(
            """INSERT OR IGNORE INTO links
               (from_id, to_id, kind, weight, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                link.from_id,
                link.to_id,
                link.kind,
                link.weight,
                link.metadata_json(),
                link.created_at,
            ),
        )

    def insert_links(self, links: list[Link]) -> int:
        """Batch insert links. Returns count inserted."""
        if not links:
            return 0
        self.conn.executemany(
            """INSERT OR IGNORE INTO links
               (from_id, to_id, kind, weight, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    lnk.from_id,
                    lnk.to_id,
                    lnk.kind,
                    lnk.weight,
                    lnk.metadata_json(),
                    lnk.created_at,
                )
                for lnk in links
            ],
        )
        self.conn.commit()
        return len(links)

    def get_links_from(self, block_id: str, kind: str | None = None) -> list[Link]:
        """Get outgoing links from a block, optionally filtered by kind."""
        if kind:
            rows = self.conn.execute(
                "SELECT from_id, to_id, kind, weight, metadata, created_at "
                "FROM links WHERE from_id = ? AND kind = ?",
                (block_id, kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT from_id, to_id, kind, weight, metadata, created_at "
                "FROM links WHERE from_id = ?",
                (block_id,),
            ).fetchall()
        return [_row_to_link(r) for r in rows]

    def get_links_to(self, block_id: str, kind: str | None = None) -> list[Link]:
        """Get incoming links to a block, optionally filtered by kind."""
        if kind:
            rows = self.conn.execute(
                "SELECT from_id, to_id, kind, weight, metadata, created_at "
                "FROM links WHERE to_id = ? AND kind = ?",
                (block_id, kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT from_id, to_id, kind, weight, metadata, created_at "
                "FROM links WHERE to_id = ?",
                (block_id,),
            ).fetchall()
        return [_row_to_link(r) for r in rows]

    # ── FTS5 Search ────────────────────────────────────────────────

    def search_fts(self, query: str, limit: int = 20) -> list[Block]:
        """Full-text search across blocks via FTS5."""
        rows = self.conn.execute(
            """SELECT b.id, b.kind, b.content, b.summary, b.embedding, b.source,
                      b.title, b.tags, b.timestamp, b.occurred_at, b.metadata,
                      b.content_hash, b.created_at
               FROM blocks_fts f
               JOIN blocks b ON f.id = b.id
               WHERE blocks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    # ── Embedding helpers ──────────────────────────────────────────

    def get_blocks_needing_embeddings(self, kind: str = "entry") -> list[Block]:
        """Get blocks that need embeddings (embedding IS NULL)."""
        rows = self.conn.execute(
            """SELECT id, kind, content, summary, embedding, source, title,
                      tags, timestamp, occurred_at, metadata, content_hash, created_at
               FROM blocks WHERE kind = ? AND embedding IS NULL AND content IS NOT NULL""",
            (kind,),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    def update_embedding(self, block_id: str, embedding: bytes) -> None:
        """Update the embedding blob for a block."""
        self.conn.execute(
            "UPDATE blocks SET embedding = ? WHERE id = ?",
            (embedding, block_id),
        )

    def update_embeddings(self, embeddings: dict[str, bytes]) -> None:
        """Batch update embeddings. {block_id: blob}."""
        if not embeddings:
            return
        self.conn.executemany(
            "UPDATE blocks SET embedding = ? WHERE id = ?",
            [(blob, bid) for bid, blob in embeddings.items()],
        )
        self.conn.commit()

    def get_blocks_with_embeddings(self, kind: str = "entry") -> list[Block]:
        """Get all blocks with embeddings for FAISS index building."""
        rows = self.conn.execute(
            """SELECT id, kind, content, summary, embedding, source, title,
                      tags, timestamp, occurred_at, metadata, content_hash, created_at
               FROM blocks WHERE kind = ? AND embedding IS NOT NULL""",
            (kind,),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    # ── Hub scoring ────────────────────────────────────────────────

    def get_hub_scores(
        self,
        weights: dict[str, float] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Compute hub scores via link aggregation. Pure SQL, query-time.

        Hub score = w_in * ln(1+in_links) + w_out * ln(1+out_links) + w_ent * ln(1+entry_count)
        """
        w = weights or {"in_links": 0.5, "out_links": 0.3, "entry_count": 0.2}
        w_in = w.get("in_links", 0.5)
        w_out = w.get("out_links", 0.3)
        w_ent = w.get("entry_count", 0.2)

        # SQLite lacks ln(), so we compute hub scores in Python
        return self._compute_hub_scores_python(w_in, w_out, w_ent, limit)

    def _compute_hub_scores_python(
        self, w_in: float, w_out: float, w_ent: float, limit: int
    ) -> list[dict]:
        """Compute hub scores in Python (SQLite lacks ln())."""
        import math

        rows = self.conn.execute(
            """
            SELECT
                b.id AS doc_id,
                b.title,
                b.source,
                COALESCE(json_extract(b.metadata, '$.source_path'), '') AS source_path,
                (SELECT COUNT(*) FROM links l WHERE l.to_id = b.id AND l.kind != 'split_from') AS in_links,
                (SELECT COUNT(*) FROM links l WHERE l.from_id = b.id) AS out_links,
                (SELECT COUNT(*) FROM links l WHERE l.to_id = b.id AND l.kind = 'split_from') AS entry_count
            FROM blocks b
            WHERE b.kind = 'document'
            """
        ).fetchall()

        results = []
        for doc_id, title, source, source_path, in_l, out_l, ent_c in rows:
            score = (
                w_in * math.log(1 + in_l)
                + w_out * math.log(1 + out_l)
                + w_ent * math.log(1 + ent_c)
            )
            results.append({
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "source_path": source_path,
                "in_links": in_l,
                "out_links": out_l,
                "entry_count": ent_c,
                "hub_score": score,
            })

        results.sort(key=lambda x: x["hub_score"], reverse=True)
        return results[:limit]

    # ── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get block and link counts by kind."""
        block_counts = self.conn.execute(
            "SELECT kind, COUNT(*) FROM blocks GROUP BY kind"
        ).fetchall()
        link_counts = self.conn.execute(
            "SELECT kind, COUNT(*) FROM links GROUP BY kind"
        ).fetchall()
        total_blocks = sum(c for _, c in block_counts)
        total_links = sum(c for _, c in link_counts)
        embedded = self.conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        return {
            "total_blocks": total_blocks,
            "total_links": total_links,
            "blocks_by_kind": {k: c for k, c in block_counts},
            "links_by_kind": {k: c for k, c in link_counts},
            "embedded_blocks": embedded,
        }


# ── Row conversion helpers ─────────────────────────────────────────

def _row_to_block(row: tuple) -> Block:
    """Convert a SQLite row tuple to a Block."""
    (
        id_, kind, content, summary, embedding, source, title,
        tags_json_str, timestamp, occurred_at, metadata_json_str,
        content_hash, created_at,
    ) = row

    tags = json.loads(tags_json_str) if tags_json_str else []
    metadata = json.loads(metadata_json_str) if metadata_json_str else {}

    return Block(
        id=id_,
        kind=kind,
        content=content,
        summary=summary,
        embedding=embedding,
        source=source,
        title=title,
        tags=tags,
        timestamp=timestamp,
        occurred_at=occurred_at,
        metadata=metadata,
        content_hash=content_hash,
        created_at=created_at or "",
    )


def _row_to_link(row: tuple) -> Link:
    """Convert a SQLite row tuple to a Link."""
    from_id, to_id, kind, weight, metadata_json_str, created_at = row
    metadata = json.loads(metadata_json_str) if metadata_json_str else {}
    return Link(
        from_id=from_id,
        to_id=to_id,
        kind=kind,
        weight=weight,
        metadata=metadata,
        created_at=created_at or "",
    )
