"""Tests for block provenance — who wrote it: human, ai, or reference.

Resolution order (first match wins): explicit `provenance/*` tag in the text,
[vault.provenance_rules] path globs, the closed set of AI/source tags, then
`human`. Backfill applies the same rule to rows already in the DB and reports,
without relabelling, blocks whose title matches a pattern the user names.
See docs/plans/query-provenance-and-dates.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openaugi.adapters.vault import (
    PROVENANCE_AI,
    PROVENANCE_HUMAN,
    PROVENANCE_REFERENCE,
    _normalize_provenance_rules,
    backfill_provenance,
    parse_vault,
    resolve_provenance,
)
from openaugi.model.block import Block
from openaugi.store.sqlite import SQLiteStore

RULES = {
    "OpenAugi/Capture/**": "human",  # narrower first: the capture stream is truth
    "OpenAugi/**": "ai",
    "_private/2-Reference/**": "reference",
}


# ── Pure logic ─────────────────────────────────────────────────────


def test_normalize_keeps_order_and_lowercases():
    rules = _normalize_provenance_rules({"A/**": "AI", "B/**": "human"})
    assert rules == [("A/**", "ai"), ("B/**", "human")]


def test_normalize_empty():
    assert _normalize_provenance_rules(None) is None
    assert _normalize_provenance_rules({}) is None


def test_normalize_rejects_unknown_value():
    """A typo in config would silently mislabel a whole folder — fail loudly."""
    with pytest.raises(ValueError, match="not one of"):
        _normalize_provenance_rules({"OpenAugi/**": "robot"})


def test_explicit_tag_wins_over_rule():
    rules = _normalize_provenance_rules(RULES)
    got = resolve_provenance("OpenAugi/Notes/x.md", ["provenance/human"], rules)
    assert got == PROVENANCE_HUMAN


def test_explicit_tag_with_unknown_value_is_ignored():
    rules = _normalize_provenance_rules(RULES)
    assert resolve_provenance("OpenAugi/Notes/x.md", ["provenance/robot"], rules) == PROVENANCE_AI


def test_first_matching_rule_wins():
    rules = _normalize_provenance_rules(RULES)
    assert resolve_provenance("OpenAugi/Capture/2026-09-01.md", [], rules) == PROVENANCE_HUMAN
    assert resolve_provenance("OpenAugi/Research/Lens - X.md", [], rules) == PROVENANCE_AI
    assert resolve_provenance("_private/2-Reference/Snipd/x.md", [], rules) == PROVENANCE_REFERENCE


def test_ai_tags_without_rules():
    path = "_private/0-Inbox/x.md"
    assert resolve_provenance(path, ["note-type/ai-summary"], None) == PROVENANCE_AI
    assert resolve_provenance(path, ["source/ai-chat"], None) == PROVENANCE_AI


def test_source_tags_are_reference_except_capture():
    assert resolve_provenance("x.md", ["source/readwise"], None) == PROVENANCE_REFERENCE
    assert resolve_provenance("x.md", ["source/podcast", "career"], None) == PROVENANCE_REFERENCE
    assert resolve_provenance("x.md", ["source/capture"], None) == PROVENANCE_HUMAN


def test_default_is_human():
    assert resolve_provenance("_private/0-Inbox/x.md", ["career"], None) == PROVENANCE_HUMAN


def test_rule_beats_tag_rule():
    """Path rules are the user's explicit statement; tag heuristics fill gaps."""
    rules = _normalize_provenance_rules({"OpenAugi/Capture/**": "human"})
    tags = ["source/readwise"]  # would be reference by tag alone
    assert resolve_provenance("OpenAugi/Capture/x.md", tags, rules) == PROVENANCE_HUMAN


# ── Parse integration ──────────────────────────────────────────────


def test_parse_vault_stamps_provenance(tmp_path: Path):
    (tmp_path / "OpenAugi" / "Notes").mkdir(parents=True)
    (tmp_path / "OpenAugi" / "Notes" / "Lens - X.md").write_text("Generated analysis.\n")
    (tmp_path / "Journal").mkdir()
    (tmp_path / "Journal" / "2026-09-01.md").write_text("I wrote this myself.\n")
    (tmp_path / "Journal" / "pasted.md").write_text("Model output #note-type/ai-summary\n")

    blocks, _ = parse_vault(tmp_path, provenance_rules=RULES)
    by_path = {
        b.metadata["source_path"]: b.metadata.get("provenance")
        for b in blocks
        if b.kind == "data_block"
    }
    assert by_path["OpenAugi/Notes/Lens - X.md"] == PROVENANCE_AI
    assert by_path["Journal/2026-09-01.md"] == PROVENANCE_HUMAN
    assert by_path["Journal/pasted.md"] == PROVENANCE_AI


def test_parse_vault_without_rules_still_stamps(tmp_path: Path):
    (tmp_path / "note.md").write_text("plain\n")
    blocks, _ = parse_vault(tmp_path)
    data = [b for b in blocks if b.kind == "data_block"]
    assert data and data[0].metadata["provenance"] == PROVENANCE_HUMAN


# ── Backfill ───────────────────────────────────────────────────────


def _block(id: str, path: str, title: str, tags: list[str] | None = None, prov=None) -> Block:
    md: dict = {"source_path": path}
    if prov:
        md["provenance"] = prov
    return Block(
        id=id,
        kind="data_block",
        content="c",
        source="vault",
        title=title,
        tags=tags or [],
        metadata=md,
    )


@pytest.fixture
def store_unlabelled(store: SQLiteStore) -> SQLiteStore:
    store.insert_blocks(
        [
            _block("gen1", "OpenAugi/Research/Lens - Y.md", "Lens - Y"),
            _block("cap1", "OpenAugi/Capture/2026-09-01.md", "2026-09-01"),
            _block(
                "jung1", "_private/0-Inbox/2025-07-27 - Jung - fate.md", "2025-07-27 - Jung - fate"
            ),
            _block("done1", "_private/0-Inbox/x.md", "x", prov="human"),  # already stamped
            _block("ref1", "_private/2-Reference/Snipd/q.md", "q", tags=["source/podcast"]),
        ]
    )
    return store


def _prov(store: SQLiteStore, block_id: str) -> str | None:
    row = store.conn.execute("SELECT metadata FROM blocks WHERE id = ?", (block_id,)).fetchone()
    return json.loads(row[0]).get("provenance")


def test_backfill_stamps_and_counts(store_unlabelled: SQLiteStore):
    out = backfill_provenance(store_unlabelled, RULES)
    assert out["updated"] == {"ai": 1, "human": 2, "reference": 1}
    assert out["unchanged"] == 1
    assert _prov(store_unlabelled, "gen1") == PROVENANCE_AI
    assert _prov(store_unlabelled, "cap1") == PROVENANCE_HUMAN
    assert _prov(store_unlabelled, "ref1") == PROVENANCE_REFERENCE


def test_backfill_is_idempotent(store_unlabelled: SQLiteStore):
    backfill_provenance(store_unlabelled, RULES)
    again = backfill_provenance(store_unlabelled, RULES)
    assert again["updated"] == {}
    assert again["unchanged"] == 5


def test_backfill_dry_run_writes_nothing(store_unlabelled: SQLiteStore):
    out = backfill_provenance(store_unlabelled, RULES, dry_run=True)
    assert sum(out["updated"].values()) == 4
    assert _prov(store_unlabelled, "gen1") is None


def test_backfill_reports_title_candidates_without_relabelling(store_unlabelled: SQLiteStore):
    """A pasted-in AI reflection lives in a human folder; a rule cannot see it.
    Title patterns surface it for the user to tag — they never decide."""
    out = backfill_provenance(store_unlabelled, RULES, title_patterns=[" - Jung - "])
    assert out["candidates"] == [("jung1", "2025-07-27 - Jung - fate")]
    assert _prov(store_unlabelled, "jung1") == PROVENANCE_HUMAN
