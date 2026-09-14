"""Tests for source attribution rules — [vault.source_rules] config.

Folder-glob rules stamp source/* tags at parse time; explicit source/*
tags in note text win; backfill applies the same logic to existing DB rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openaugi.adapters.vault import (
    _apply_source_rules,
    _normalize_source_rules,
    backfill_source_tags,
    parse_vault,
)
from openaugi.model.block import Block
from openaugi.store.sqlite import SQLiteStore

RULES = {
    "Reference/Readwise/**": "source/readwise",
    "Reference/AI Conversations/**": "source/ai-chat",
    "Reference/Articles/**": "#source/webclip",  # leading '#' tolerated
}


# ── Pure logic ─────────────────────────────────────────────────────


def test_normalize_strips_hash_and_keeps_order():
    rules = _normalize_source_rules(RULES)
    assert rules is not None
    assert rules[0] == ("Reference/Readwise/**", "source/readwise")
    assert rules[2] == ("Reference/Articles/**", "source/webclip")


def test_normalize_empty():
    assert _normalize_source_rules(None) is None
    assert _normalize_source_rules({}) is None


def test_apply_matching_folder_stamps_tag():
    rules = _normalize_source_rules(RULES)
    tags = _apply_source_rules("Reference/Readwise/Some Article.md", ["books"], rules)
    assert tags == ["books", "source/readwise"]


def test_apply_non_matching_path_unchanged():
    rules = _normalize_source_rules(RULES)
    tags = _apply_source_rules("Journal/2026-07-07.md", ["career"], rules)
    assert tags == ["career"]


def test_apply_explicit_source_tag_wins():
    """Text is truth — a source/* tag written in the note beats the folder rule."""
    rules = _normalize_source_rules(RULES)
    tags = _apply_source_rules("Reference/Readwise/clip.md", ["source/notebookLM"], rules)
    assert tags == ["source/notebookLM"]


def test_apply_first_matching_rule_wins():
    rules = _normalize_source_rules(
        {"Reference/**": "source/webclip", "Reference/Readwise/**": "source/readwise"}
    )
    tags = _apply_source_rules("Reference/Readwise/clip.md", [], rules)
    assert tags == ["source/webclip"]


# ── Parse-time integration ─────────────────────────────────────────


@pytest.fixture
def rules_vault(tmp_path: Path) -> Path:
    (tmp_path / "Reference" / "Readwise").mkdir(parents=True)
    (tmp_path / "Journal").mkdir()
    (tmp_path / "Reference" / "Readwise" / "highlight.md").write_text(
        "Saved highlight about #focus from a book."
    )
    (tmp_path / "Reference" / "Readwise" / "tagged.md").write_text(
        "Already attributed #source/notebookLM content."
    )
    (tmp_path / "Journal" / "2026-07-07.md").write_text("My own thought about focus.")
    return tmp_path


def test_parse_vault_stamps_source_rules(rules_vault: Path):
    # DEFAULT_EXCLUDE_PATTERNS skips **/Readwise/** — a user who ingests
    # Readwise (like the user) overrides exclude_patterns, so mirror that here.
    blocks, _links = parse_vault(
        rules_vault, exclude_patterns=[".obsidian/**"], source_rules=RULES
    )
    by_path = {b.metadata["source_path"]: b for b in blocks if b.kind == "data_block"}
    assert "source/readwise" in by_path["Reference/Readwise/highlight.md"].tags
    assert by_path["Reference/Readwise/tagged.md"].tags == ["source/notebookLM"]
    assert by_path["Journal/2026-07-07.md"].tags == []
    # tag block + groups link created like any other tag
    tag_titles = {b.title for b in blocks if b.kind == "context_block:tag"}
    assert "source/readwise" in tag_titles


def test_parse_vault_no_rules_unchanged(rules_vault: Path):
    blocks, _links = parse_vault(rules_vault)
    for b in blocks:
        if b.kind == "data_block":
            assert not any(t == "source/readwise" for t in b.tags)


# ── Backfill ───────────────────────────────────────────────────────


@pytest.fixture
def store_with_unattributed(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "test.db")
    store.insert_blocks(
        [
            Block(
                id="rw1",
                kind="data_block",
                content="a readwise highlight",
                tags=["books"],
                metadata={"source_path": "Reference/Readwise/h.md"},
            ),
            Block(
                id="own1",
                kind="data_block",
                content="my own words",
                metadata={"source_path": "Journal/2026-07-07.md"},
            ),
            Block(
                id="pre1",
                kind="data_block",
                content="explicitly tagged",
                tags=["source/notebookLM"],
                metadata={"source_path": "Reference/Readwise/x.md"},
            ),
        ]
    )
    return store


def test_backfill_updates_matching_blocks(store_with_unattributed: SQLiteStore):
    stats = backfill_source_tags(store_with_unattributed, RULES)
    assert stats == {"source/readwise": 1}
    row = store_with_unattributed.conn.execute(
        "SELECT tags FROM blocks WHERE id = 'rw1'"
    ).fetchone()
    assert json.loads(row[0]) == ["books", "source/readwise"]
    # groups link to the tag block exists
    tag_id = Block.make_tag_id("source/readwise")
    link = store_with_unattributed.conn.execute(
        "SELECT 1 FROM links WHERE from_id='rw1' AND to_id=? AND kind='groups'",
        (tag_id,),
    ).fetchone()
    assert link


def test_backfill_respects_explicit_tags(store_with_unattributed: SQLiteStore):
    backfill_source_tags(store_with_unattributed, RULES)
    row = store_with_unattributed.conn.execute(
        "SELECT tags FROM blocks WHERE id = 'pre1'"
    ).fetchone()
    assert json.loads(row[0]) == ["source/notebookLM"]


def test_backfill_idempotent(store_with_unattributed: SQLiteStore):
    backfill_source_tags(store_with_unattributed, RULES)
    stats2 = backfill_source_tags(store_with_unattributed, RULES)
    assert stats2 == {}


def test_backfill_dry_run_writes_nothing(store_with_unattributed: SQLiteStore):
    stats = backfill_source_tags(store_with_unattributed, RULES, dry_run=True)
    assert stats == {"source/readwise": 1}
    row = store_with_unattributed.conn.execute(
        "SELECT tags FROM blocks WHERE id = 'rw1'"
    ).fetchone()
    assert json.loads(row[0]) == ["books"]
