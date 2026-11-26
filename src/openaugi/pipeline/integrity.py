"""Graph integrity — does the stored graph still say what the vault says?

Read-only. The class of bug this exists for is silent drift: an edge keyed by
one identity rule while its endpoint is keyed by another (path vs `augi_id`),
an incremental ingest ending somewhere a fresh ingest would not, the
valid-endpoint filter dropping edges without a word, records left behind by a
rename or a delete. None of these raise; the graph just answers with less than
the vault holds. See docs/reference/graph-integrity.md.

Two tiers:

- **Store-only** checks need nothing but the DB: one document per
  source_path, every edge has both endpoints, every vault data_block is
  contained by exactly one document, and that document is its own note's.
- **Vault** checks (when `vault_root` is given) re-parse the vault with the
  ingest parser and compare: every stored document still has a file and the
  id the identity rule gives it, and the `links_to` edges of each note match a
  fresh parse of it. Only notes whose file is unchanged since the last ingest
  (content hash equal) are compared, so a note edited a second ago is not
  reported as drift.

Findings have a severity: `error` is drift; `info` is worth knowing and not
wrong (wikilinks whose target is not a note in the vault).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from openaugi.store.sqlite import SQLiteStore

# Edges whose endpoints are blocks the ingest owns. `groups` also covers
# cluster → document edges; `routed_to` is written by the review pass.
_CHECKED_KINDS = ("contains", "links_to", "routed_to", "groups")
_EXAMPLES = 5


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "info"
    count: int
    detail: str
    examples: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        ex = "".join(f"\n    - {e}" for e in self.examples)
        return f"[{self.severity}] {self.check}: {self.count} — {self.detail}{ex}"


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def check_graph_integrity(
    store: SQLiteStore,
    vault_root: str | Path | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Finding]:
    """Every integrity finding for `store`, vault checks included when given."""
    findings = _store_checks(store)
    if vault_root is not None:
        findings += _vault_checks(store, Path(vault_root).expanduser(), exclude_patterns)
    return findings


# ── Store-only ─────────────────────────────────────────────────────


def _add(out: list[Finding], check: str, items: list[str], detail: str, severity="error"):
    if items:
        out.append(Finding(check, severity, len(items), detail, items[:_EXAMPLES]))


def _store_checks(store: SQLiteStore) -> list[Finding]:
    c = store.conn
    out: list[Finding] = []

    dup = c.execute(
        """SELECT json_extract(metadata, '$.source_path') sp, COUNT(*) n
           FROM blocks WHERE kind = 'context_block:document' AND source = 'vault'
           GROUP BY sp HAVING n > 1"""
    ).fetchall()
    _add(
        out,
        "duplicate_document",
        [f"{sp} ({n} documents)" for sp, n in dup],
        "more than one document block for one source_path",
    )

    placeholders = ",".join("?" * len(_CHECKED_KINDS))
    dangling = c.execute(
        f"""SELECT l.kind, l.from_id, l.to_id FROM links l
            LEFT JOIN blocks a ON a.id = l.from_id
            LEFT JOIN blocks b ON b.id = l.to_id
            WHERE l.kind IN ({placeholders}) AND (a.id IS NULL OR b.id IS NULL)""",
        _CHECKED_KINDS,
    ).fetchall()
    _add(
        out,
        "dangling_edge",
        [f"{k} {f} -> {t}" for k, f, t in dangling],
        "edge with a missing endpoint",
    )

    rows = c.execute(
        """SELECT b.id, json_extract(b.metadata, '$.source_path'),
                  COUNT(l.to_id),
                  SUM(CASE WHEN json_extract(d.metadata, '$.source_path')
                                = json_extract(b.metadata, '$.source_path')
                           THEN 1 ELSE 0 END)
           FROM blocks b
           LEFT JOIN links l ON l.from_id = b.id AND l.kind = 'contains'
           LEFT JOIN blocks d ON d.id = l.to_id
           WHERE b.kind = 'data_block' AND b.source = 'vault'
           GROUP BY b.id"""
    ).fetchall()
    _add(
        out,
        "orphan_data_block",
        [f"{sp} {bid}" for bid, sp, n, _ in rows if n == 0],
        "vault data_block contained by no document",
    )
    _add(
        out,
        "multiple_contains",
        [f"{sp} {bid} ({n})" for bid, sp, n, _ in rows if n > 1],
        "vault data_block contained by more than one document",
    )
    _add(
        out,
        "contains_wrong_document",
        [f"{sp} {bid}" for bid, sp, n, own in rows if n >= 1 and not own],
        "data_block's document is not the one for its own source_path",
    )

    wrong_target = c.execute(
        """SELECT l.kind, l.from_id, l.to_id, b.kind FROM links l
           JOIN blocks b ON b.id = l.to_id
           WHERE l.kind IN ('links_to', 'routed_to')
             AND b.kind != 'context_block:document'"""
    ).fetchall()
    _add(
        out,
        "edge_target_not_document",
        [f"{k} {f} -> {t} ({tk})" for k, f, t, tk in wrong_target],
        "links_to / routed_to edge into a block that is not a document",
    )
    return out


# ── Against the vault ──────────────────────────────────────────────


def _vault_checks(
    store: SQLiteStore, vault: Path, exclude_patterns: list[str] | None
) -> list[Finding]:
    from openaugi.adapters.vault import parse_vault

    blocks, links = parse_vault(vault, exclude_patterns=exclude_patterns)
    parsed_docs = {
        b.metadata["source_path"]: b for b in blocks if b.kind == "context_block:document"
    }
    entry_path = {b.id: b.metadata["source_path"] for b in blocks if b.kind == "data_block"}
    parsed_doc_ids = {b.id for b in parsed_docs.values()}

    stored = store.conn.execute(
        """SELECT id, json_extract(metadata, '$.source_path'), content_hash
           FROM blocks WHERE kind = 'context_block:document' AND source = 'vault'"""
    ).fetchall()
    # A note is "in sync" when the stored document carries the hash of the
    # file on disk: the last ingest saw exactly this content.
    in_sync: set[str] = set()
    out: list[Finding] = []
    no_file: list[str] = []
    wrong_id: list[str] = []
    for doc_id, sp, content_hash in stored:
        parsed = parsed_docs.get(sp)
        if parsed is None:
            no_file.append(f"{sp} {doc_id}")
            continue
        if parsed.content_hash != content_hash:
            continue
        in_sync.add(sp)
        if parsed.id != doc_id:
            wrong_id.append(f"{sp}: stored {doc_id}, identity rule gives {parsed.id}")
    _add(out, "document_without_file", no_file, "document whose source_path is not in the vault")
    _add(out, "document_identity", wrong_id, "document id differs from the identity rule")

    existing = {r[0] for r in store.conn.execute("SELECT id FROM blocks")}
    expected: set[tuple[str, str]] = set()
    unresolved: Counter[str] = Counter()
    for lnk in links:
        if lnk.kind != "links_to" or entry_path.get(lnk.from_id) not in in_sync:
            continue
        if lnk.to_id not in parsed_doc_ids:
            unresolved[lnk.metadata.get("target_title", lnk.to_id)] += 1
            continue
        expected.add((lnk.from_id, lnk.to_id))

    stored_links: set[tuple[str, str]] = set()
    for from_id, to_id, sp in store.conn.execute(
        """SELECT l.from_id, l.to_id, json_extract(b.metadata, '$.source_path')
           FROM links l JOIN blocks b ON b.id = l.from_id
           WHERE l.kind = 'links_to' AND b.kind = 'data_block'"""
    ):
        if sp in in_sync:
            stored_links.add((from_id, to_id))

    titles = {d.id: d.title or d.id for d in parsed_docs.values()}

    def _fmt(pair: tuple[str, str]) -> str:
        return f"{entry_path.get(pair[0], pair[0])} -> {titles.get(pair[1], pair[1])}"

    missing = sorted(expected - stored_links)
    _add(
        out,
        "links_to_missing",
        [_fmt(p) for p in missing if p[0] in existing and p[1] in existing],
        "wikilink in an unchanged note with no stored edge",
    )
    _add(
        out,
        "links_to_extra",
        [_fmt(p) for p in sorted(stored_links - expected)],
        "stored edge a fresh parse of an unchanged note does not produce",
    )
    _add(
        out,
        "unresolved_wikilink",
        [f"[[{t}]] ×{n}" for t, n in unresolved.most_common()],
        "wikilink target is not a note in the vault (no edge is stored)",
        severity="info",
    )
    return out
