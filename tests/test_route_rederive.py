"""Tests for the route re-derive contract (decided 2026-07-09, the user's call).

Routing is a decision the review pass makes, recorded as routed_to links in
the DB — a projection, not truth. Editing a routed block changes its
content-hash identity: the old row (and its routes) is deleted BY DESIGN,
and the edited block re-enters the review queue as a new block for the next
pass to re-decide (aaa: lines in the text carry durable human intent).
No similarity matching, no fuzzy migration — hashes stay deterministic.
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
        "Considering NVDA calls because earnings momentum looks strong. "
        "aaa: file under trading\n\n"
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


def _route_block(store: SQLiteStore, content_fragment: str) -> str:
    """Simulate the review pass routing the block matching the fragment."""
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind='data_block' AND content LIKE ?",
        (f"%{content_fragment}%",),
    ).fetchone()
    assert row, f"no block containing {content_fragment!r}"
    store.insert_links([Link(from_id=row[0], to_id=CONTAINER_ID, kind="routed_to")])
    return row[0]


def _routed_block_ids(store: SQLiteStore) -> set[str]:
    return {
        r[0]
        for r in store.conn.execute(
            "SELECT from_id FROM links WHERE kind='routed_to' AND to_id = ?",
            (CONTAINER_ID,),
        ).fetchall()
    }


def test_edit_drops_route_and_requeues_block(vault: Path, store: SQLiteStore):
    """The core contract: edit → route gone, successor is a NEW block for the pass."""
    run_layer0(vault, store)
    old_id = _route_block(store, "NVDA calls")

    (vault / "daily.md").write_text(
        "## Trading thoughts\n\n"
        "Considering NVDA calls because earnings momentum looks strong. "
        "Position sized small. aaa: file under trading\n\n"
        "## Unrelated\n\nGroceries and errands today.\n"
    )
    result = run_layer0(vault, store)

    # Route dropped by design — the projection follows the text
    assert _routed_block_ids(store) == set()
    # Old identity is gone entirely
    assert store.get_block(old_id) is None
    # The edited block re-enters the review queue as a new data block,
    # with the aaa: instruction still riding in its text (text is truth)
    new_contents = [b.content or "" for b in result["new_data_blocks"]]
    assert any("aaa: file under trading" in c for c in new_contents)


def test_unedited_blocks_keep_routes(vault: Path, store: SQLiteStore):
    """Editing one block must not disturb another block's routing."""
    run_layer0(vault, store)
    routed_id = _route_block(store, "NVDA calls")

    (vault / "daily.md").write_text(
        "## Trading thoughts\n\n"
        "Considering NVDA calls because earnings momentum looks strong. "
        "aaa: file under trading\n\n"
        "## Unrelated\n\nGroceries, errands, and a haircut today.\n"
    )
    result = run_layer0(vault, store)

    assert _routed_block_ids(store) == {routed_id}
    # And the untouched block did NOT re-enter the queue
    assert all("NVDA" not in (b.content or "") for b in result["new_data_blocks"])


def test_untouched_file_keeps_routes(vault: Path, store: SQLiteStore):
    run_layer0(vault, store)
    routed_id = _route_block(store, "NVDA calls")
    run_layer0(vault, store)  # no changes at all
    assert _routed_block_ids(store) == {routed_id}


def test_deleted_block_drops_route(vault: Path, store: SQLiteStore):
    run_layer0(vault, store)
    _route_block(store, "NVDA calls")
    (vault / "daily.md").write_text("## Unrelated\n\nGroceries and errands today.\n")
    run_layer0(vault, store)
    assert _routed_block_ids(store) == set()


def test_no_orphaned_route_links_after_edit(vault: Path, store: SQLiteStore):
    """CASCADE must leave zero dangling routed_to rows."""
    run_layer0(vault, store)
    _route_block(store, "NVDA calls")
    (vault / "daily.md").write_text(
        "## Trading thoughts\n\nTotally rewritten thought.\n\n"
        "## Unrelated\n\nGroceries and errands today.\n"
    )
    run_layer0(vault, store)
    orphans = store.conn.execute(
        """SELECT COUNT(*) FROM links l
           WHERE l.kind='routed_to'
             AND NOT EXISTS (SELECT 1 FROM blocks b WHERE b.id = l.from_id)"""
    ).fetchone()[0]
    assert orphans == 0


def test_augi_tags_also_drop_with_identity(vault: Path, store: SQLiteStore):
    """Agent classification is projection state too — same re-derive contract."""
    run_layer0(vault, store)
    row = store.conn.execute(
        "SELECT id FROM blocks WHERE kind='data_block' AND content LIKE '%NVDA%'"
    ).fetchone()
    store.conn.execute(
        "UPDATE blocks SET metadata = json_set(metadata, '$.augi_tags', json(?)) WHERE id = ?",
        (json.dumps(["area/finance"]), row[0]),
    )
    store.conn.commit()

    (vault / "daily.md").write_text(
        "## Trading thoughts\n\nEdited NVDA thought, same idea.\n\n"
        "## Unrelated\n\nGroceries and errands today.\n"
    )
    run_layer0(vault, store)
    tagged = store.conn.execute(
        "SELECT COUNT(*) FROM blocks WHERE json_extract(metadata,'$.augi_tags') IS NOT NULL"
        " AND kind='data_block'"
    ).fetchone()[0]
    assert tagged == 0
