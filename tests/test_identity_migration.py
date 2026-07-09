"""Tests for agent-state durability across block edits.

Block identity is a content hash, so editing a block = delete + insert.
Before this fix, CASCADE silently dropped routed_to links and augi_tags
(found in the wild 2026-07-08: the trading MOC lost all its routed blocks
within 48h). run_layer0 now migrates that state to the edited successor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore

CONTAINER_ID = "container-amoc-1"


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "daily.md").write_text(
        "## Trading thoughts\n\n"
        "Considering NVDA calls because earnings momentum looks strong this quarter.\n\n"
        "## Unrelated\n\nGroceries and errands today.\n"
    )
    return v


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "test.db")
    s.insert_blocks(
        [Block(id=CONTAINER_ID, kind="context_block:document", title="AMOC - Trading")]
    )
    return s


def _route_and_tag_block(store: SQLiteStore, content_fragment: str) -> str:
    """Simulate the review pass: route + classify the block matching the fragment."""
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind='data_block' AND content LIKE ?",
        (f"%{content_fragment}%",),
    ).fetchone()
    assert row, f"no block containing {content_fragment!r}"
    block_id = row[0]
    store.insert_links([Link(from_id=block_id, to_id=CONTAINER_ID, kind="routed_to")])
    store.conn.execute(
        "UPDATE blocks SET metadata = json_set(metadata, '$.augi_tags', json(?)) WHERE id = ?",
        (json.dumps(["area/finance", "type/idea"]), block_id),
    )
    store.conn.commit()
    return block_id


def _routed_block_ids(store: SQLiteStore) -> set[str]:
    return {
        r[0]
        for r in store.conn.execute(
            "SELECT from_id FROM links WHERE kind='routed_to' AND to_id = ?",
            (CONTAINER_ID,),
        ).fetchall()
    }


def test_route_survives_block_edit(vault: Path, store: SQLiteStore):
    run_layer0(vault, store)
    old_id = _route_and_tag_block(store, "NVDA calls")

    # Edit the routed block — typo fix + a new sentence (same idea)
    (vault / "daily.md").write_text(
        "## Trading thoughts\n\n"
        "Considering NVDA calls because earnings momentum looks strong this quarter. "
        "Position sized small.\n\n"
        "## Unrelated\n\nGroceries and errands today.\n"
    )
    run_layer0(vault, store)

    routed = _routed_block_ids(store)
    assert routed, "routing was dropped by the edit"
    new_id = routed.pop()
    assert new_id != old_id  # identity did change — state was migrated
    meta = json.loads(
        store.conn.execute("SELECT metadata FROM blocks WHERE id=?", (new_id,)).fetchone()[0]
    )
    assert meta.get("augi_tags") == ["area/finance", "type/idea"]


def test_unedited_block_untouched(vault: Path, store: SQLiteStore):
    """Editing one block must not disturb another block's routing."""
    run_layer0(vault, store)
    routed_id = _route_and_tag_block(store, "NVDA calls")

    # Edit only the OTHER section
    (vault / "daily.md").write_text(
        "## Trading thoughts\n\n"
        "Considering NVDA calls because earnings momentum looks strong this quarter.\n\n"
        "## Unrelated\n\nGroceries, errands, and a haircut today.\n"
    )
    run_layer0(vault, store)
    assert _routed_block_ids(store) == {routed_id}


def test_rewrite_beyond_recognition_drops_state(vault: Path, store: SQLiteStore):
    """A block replaced by unrelated content is a real deletion, not an edit."""
    run_layer0(vault, store)
    _route_and_tag_block(store, "NVDA calls")

    (vault / "daily.md").write_text(
        "## Trading thoughts\n\n"
        "Completely different topic now: meal prep plan for the week ahead.\n\n"
        "## Unrelated\n\nGroceries and errands today.\n"
    )
    run_layer0(vault, store)
    assert _routed_block_ids(store) == set()


def test_deleted_block_drops_state(vault: Path, store: SQLiteStore):
    run_layer0(vault, store)
    _route_and_tag_block(store, "NVDA calls")

    (vault / "daily.md").write_text("## Unrelated\n\nGroceries and errands today.\n")
    run_layer0(vault, store)
    assert _routed_block_ids(store) == set()


def test_blocks_without_agent_state_skip_matching(vault: Path, store: SQLiteStore):
    """Edits to never-routed blocks don't produce migrations (no-op path)."""
    run_layer0(vault, store)
    (vault / "daily.md").write_text(
        "## Trading thoughts\n\n"
        "Considering NVDA calls because earnings momentum looks strong this quarter.\n\n"
        "## Unrelated\n\nGroceries and errands today, plus laundry.\n"
    )
    result = run_layer0(vault, store)
    assert result["blocks_added"] >= 1  # the edit itself ingested fine
    assert _routed_block_ids(store) == set()
