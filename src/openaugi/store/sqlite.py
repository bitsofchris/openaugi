"""SQLite backend — blocks + links + FTS5 + sqlite-vec vector search.

Two core tables (blocks, links) plus a vec0 virtual table for semantic search.
WAL mode for concurrent reads while the pipeline writes.

See docs/plans/m0.md § Data Model for the canonical schema.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

import numpy as np

from openaugi.model.block import Block
from openaugi.model.link import Link

logger = logging.getLogger(__name__)

# Schema version — bump when migrations are needed
SCHEMA_VERSION = 2

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

_META_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
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
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        return self._conn

    def close(self):
        """Close connection and release file lock."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _initialize(self):
        """Create tables, indexes, FTS, triggers, and meta table."""
        c = self.conn
        c.executescript(_BLOCKS_DDL)
        c.executescript(_LINKS_DDL)
        c.executescript(_FTS_DDL)
        c.executescript(_FTS_TRIGGERS)
        c.executescript(_META_DDL)
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

    def get_entries_for_document(self, doc_id: str) -> list[Block]:
        """Get entry blocks linked to a document via split_from."""
        rows = self.conn.execute(
            """SELECT b.id, b.kind, b.content, b.summary, b.embedding, b.source,
                      b.title, b.tags, b.timestamp, b.occurred_at, b.metadata,
                      b.content_hash, b.created_at
               FROM blocks b
               JOIN links l ON l.from_id = b.id
               WHERE l.to_id = ? AND l.kind = 'split_from'""",
            (doc_id,),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    def update_block_hash(self, block_id: str, content_hash: str) -> None:
        """Update a block's content_hash (used for document block file-level hash)."""
        self.conn.execute(
            "UPDATE blocks SET content_hash = ? WHERE id = ?",
            (content_hash, block_id),
        )
        self.conn.commit()

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
        """Full-text search across blocks via FTS5.

        User input is quoted to prevent FTS5 syntax errors (e.g., bare words
        being interpreted as column names).
        """
        safe_query = _sanitize_fts_query(query)
        rows = self.conn.execute(
            """SELECT b.id, b.kind, b.content, b.summary, b.embedding, b.source,
                      b.title, b.tags, b.timestamp, b.occurred_at, b.metadata,
                      b.content_hash, b.created_at
               FROM blocks_fts f
               JOIN blocks b ON f.id = b.id
               WHERE blocks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, limit),
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
        """Batch update embeddings. {block_id: blob}. Also writes to vec_blocks if it exists."""
        if not embeddings:
            return
        self.conn.executemany(
            "UPDATE blocks SET embedding = ? WHERE id = ?",
            [(blob, bid) for bid, blob in embeddings.items()],
        )
        if self._vec_table_exists():
            self.conn.executemany(
                "INSERT OR REPLACE INTO vec_blocks(block_id, embedding) VALUES (?, ?)",
                [(bid, _normalize_blob(blob)) for bid, blob in embeddings.items()],
            )
        self.conn.commit()

    def get_blocks_by_ids(self, block_ids: list[str]) -> dict[str, Block]:
        """Fetch full blocks for a list of block IDs in a single query.

        Returns a dict {block_id: Block} — missing IDs are omitted.
        """
        if not block_ids:
            return {}
        placeholders = ",".join("?" * len(block_ids))
        rows = self.conn.execute(
            f"""SELECT id, kind, content, summary, embedding, source, title,
                       tags, timestamp, occurred_at, metadata, content_hash, created_at
                FROM blocks WHERE id IN ({placeholders})""",
            block_ids,
        ).fetchall()
        return {row[0]: _row_to_block(row) for row in rows}

    def get_embeddings_for_ids(self, block_ids: list[str]) -> dict[str, bytes | None]:
        """Fetch embedding blobs for a list of block IDs.

        Returns a dict {block_id: embedding_blob_or_None}.
        More efficient than get_block() when only embeddings are needed.
        """
        if not block_ids:
            return {}
        placeholders = ",".join("?" * len(block_ids))
        rows = self.conn.execute(
            f"SELECT id, embedding FROM blocks WHERE id IN ({placeholders})",
            block_ids,
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_blocks_with_embeddings(self, kind: str = "entry") -> list[Block]:
        """Get all blocks with embeddings (raw blobs for migration purposes)."""
        rows = self.conn.execute(
            """SELECT id, kind, content, summary, embedding, source, title,
                      tags, timestamp, occurred_at, metadata, content_hash, created_at
               FROM blocks WHERE kind = ? AND embedding IS NOT NULL""",
            (kind,),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    # ── Vector search ──────────────────────────────────────────────

    def ensure_vec_table(self, dim: int) -> None:
        """Create vec0 virtual table for the given dimension.

        Safe to call repeatedly — no-op if already exists with matching dim.
        Drops and recreates if dimension changed (model swap).
        """
        existing = self._get_meta("vec_dim")
        if existing == str(dim):
            return
        if existing is not None:
            logger.warning(
                "Embedding dimension changed %s→%d, recreating vec_blocks", existing, dim
            )
            self.conn.execute("DROP TABLE IF EXISTS vec_blocks")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE vec_blocks USING vec0(block_id TEXT PRIMARY KEY, embedding float[{dim}])"
        )
        self._set_meta("vec_dim", str(dim))
        self.conn.commit()
        logger.info("Created vec_blocks table (dim=%d)", dim)

    def semantic_search(self, query_vec: list[float], k: int = 20) -> list[tuple[str, float]]:
        """KNN search over vec_blocks. Returns (block_id, distance) sorted ascending.

        Lower distance = more similar. Returns empty list if vec table doesn't exist.
        """
        if not self._vec_table_exists():
            return []
        normalized = _normalize_blob(np.array(query_vec, dtype=np.float32).tobytes())
        rows = self.conn.execute(
            "SELECT block_id, distance FROM vec_blocks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            [normalized, k],
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def populate_vec_from_blocks(self, dim: int) -> int:
        """Migrate existing embedding blobs from blocks table into vec_blocks.

        Used for one-time migration of existing databases. Returns count inserted.
        """
        self.ensure_vec_table(dim)
        blocks = self.get_blocks_with_embeddings()
        if not blocks:
            return 0
        self.conn.executemany(
            "INSERT OR REPLACE INTO vec_blocks(block_id, embedding) VALUES (?, ?)",
            [(b.id, _normalize_blob(b.embedding)) for b in blocks if b.embedding],
        )
        self.conn.commit()
        count = len([b for b in blocks if b.embedding])
        logger.info("Migrated %d embeddings to vec_blocks", count)
        return count

    def _vec_table_exists(self) -> bool:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_blocks'"
        ).fetchone()
        return row is not None

    def _get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

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
            results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source": source,
                    "source_path": source_path,
                    "in_links": in_l,
                    "out_links": out_l,
                    "entry_count": ent_c,
                    "hub_score": score,
                }
            )

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


# ── Internal helpers ───────────────────────────────────────────────


def _normalize_blob(blob: bytes) -> bytes:
    """Normalize a float32 embedding blob to unit length (for cosine similarity via L2 distance)."""
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


# ── Row conversion helpers ─────────────────────────────────────────


def _sanitize_fts_query(query: str) -> str:
    """Make a user query safe for FTS5 MATCH.

    FTS5 has its own syntax: bare `-` means NOT, unquoted words matching column
    names (title, content, tags) are treated as column prefixes, and special chars
    like `*`, `"`, `(` have meaning.

    Strategy: quote each word individually so FTS5 treats them as literal terms
    joined by implicit AND (not as a phrase). Column-prefixed queries from our
    own code (e.g., `title:foo`) get the value part quoted as a phrase.
    """
    # Column-prefixed query from our own code (e.g., "title:Some Title")
    if ":" in query:
        col, _, value = query.partition(":")
        col = col.strip()
        # Only allow known FTS5 columns
        if col in ("title", "content", "tags"):
            escaped = value.strip().replace('"', '""')
            return f'{col}:"{escaped}"'

    # Strip FTS5 operators that cause syntax errors even when quoted
    cleaned = re.sub(r"[,()\*\+\^~]", " ", query)
    # Plain user query — quote each word individually (implicit AND, not phrase)
    tokens = cleaned.split()
    quoted = [f'"{t.replace(chr(34), chr(34) + chr(34))}"' for t in tokens if t.strip("-")]
    return " ".join(quoted) if quoted else f'"{query}"'


def _row_to_block(row: tuple) -> Block:
    """Convert a SQLite row tuple to a Block."""
    (
        id_,
        kind,
        content,
        summary,
        embedding,
        source,
        title,
        tags_json_str,
        timestamp,
        occurred_at,
        metadata_json_str,
        content_hash,
        created_at,
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
