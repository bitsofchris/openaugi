"""Tests for the zzz dispatch pipeline.

Covers:
- dispatch_zzz_blocks writes task files for blocks with zzz_instructions
- Blocks without zzz_instructions are skipped
- Non-data_block blocks are skipped
- Task file content follows the expected format
- build_task_file produces valid frontmatter and sections
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.pipeline.dispatch import (
    DISPATCHED,
    QUEUED,
    SUPERSEDED,
    ZZZ_QUEUE_COLLECTION,
    build_task_file,
    dispatch_zzz_blocks,
    drain_zzz_queue,
    record_zzz_changes,
    resolve_anchor_refs,
)
from openaugi.store.sqlite import SQLiteStore


def _make_block(
    id_: str,
    content: str,
    source_path: str = "journal.md",
    zzz: list[str] | None = None,
    kind: str = "data_block",
) -> Block:
    metadata: dict = {"source_path": source_path}
    if zzz:
        metadata["zzz_instructions"] = zzz
    return Block(
        id=id_,
        kind=kind,
        content=content,
        source="vault",
        title="journal",
        content_hash=id_,
        metadata=metadata,
    )


class TestBuildTaskFile:
    def test_includes_zzz_instructions(self):
        block = _make_block("a" * 16, "some content", zzz=["research deep learning"])
        result = build_task_file(block)
        assert "research deep learning" in result
        assert "some content" in result

    def test_has_required_sections(self):
        block = _make_block("a" * 16, "content", zzz=["task fix readme"])
        result = build_task_file(block)
        assert "status: pending" in result
        assert f"source_block_id: {'a' * 16}" in result
        assert "## Context" in result
        assert "## User instruction" in result
        assert "## Task" in result
        assert "## Results" in result

    def test_multiple_zzz_instructions(self):
        block = _make_block("a" * 16, "content", zzz=["research X", "also tag Y"])
        result = build_task_file(block)
        assert "> research X" in result
        assert "> also tag Y" in result

    def test_derives_title_from_zzz(self):
        block = _make_block("a" * 16, "content", zzz=["research deep learning models"])
        result = build_task_file(block)
        assert "# research deep learning models" in result

    def test_derives_title_from_content_when_zzz_short(self):
        block = _make_block("a" * 16, "My thoughts on embeddings", zzz=["go"])
        result = build_task_file(block)
        assert "# My thoughts on embeddings" in result


class TestDispatchZzzBlocks:
    def test_writes_task_file_for_zzz_block(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        blocks = [_make_block("a" * 16, "content", zzz=["research this"])]

        written = dispatch_zzz_blocks(blocks, vault)

        assert len(written) == 1
        assert written[0].exists()
        text = written[0].read_text()
        assert "status: pending" in text
        assert "research this" in text

    def test_skips_blocks_without_zzz(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        blocks = [_make_block("a" * 16, "no instructions here")]

        written = dispatch_zzz_blocks(blocks, vault)

        assert len(written) == 0

    def test_skips_non_data_blocks(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        blocks = [
            _make_block(
                "t" * 16, "tag content", kind="context_block:tag", zzz=["should be skipped"]
            )
        ]

        written = dispatch_zzz_blocks(blocks, vault)

        assert len(written) == 0

    def test_creates_tasks_folder(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        blocks = [_make_block("a" * 16, "content", zzz=["do this"])]

        dispatch_zzz_blocks(blocks, vault)

        assert (vault / "OpenAugi" / "Tasks").is_dir()

    def test_multiple_blocks_multiple_files(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        blocks = [
            _make_block("a" * 16, "first", zzz=["task one"]),
            _make_block("b" * 16, "second", zzz=["task two"]),
            _make_block("c" * 16, "no zzz"),
        ]

        written = dispatch_zzz_blocks(blocks, vault)

        assert len(written) == 2


def _capture_note(vault: Path, date: str, entries: dict[str, str]) -> Path:
    """Write a mobile-style daily note: `HH:MM — text` entries + anchor lines."""
    note = vault / "OpenAugi" / "Capture" / f"{date}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {date}"]
    for anchor, text in entries.items():
        parts.append(f"09:15 — {text}\n^{anchor}")
    note.write_text("\n\n".join(parts) + "\n")
    return note


LENS_CONTENT = (
    "gathered 2 blocks:\n"
    "[[2026-07-14#^augi-aaaa1111]] [[2026-07-14#^augi-bbbb2222]]\n"
    "zzz: apply lens nuggets"
)


class TestAnchorRefResolution:
    def test_resolves_refs_from_capture_daily_note(self, tmp_path: Path):
        vault = tmp_path / "vault"
        _capture_note(
            vault,
            "2026-07-14",
            {"augi-aaaa1111": "first gathered thought", "augi-bbbb2222": "second thought"},
        )

        resolved = resolve_anchor_refs(LENS_CONTENT, vault)

        assert resolved == [
            ("[[2026-07-14#^augi-aaaa1111]]", "09:15 — first gathered thought"),
            ("[[2026-07-14#^augi-bbbb2222]]", "09:15 — second thought"),
        ]

    def test_multiline_entry_resolved_whole(self, tmp_path: Path):
        vault = tmp_path / "vault"
        note = vault / "OpenAugi" / "Capture" / "2026-07-14.md"
        note.parent.mkdir(parents=True)
        note.write_text(
            "# 2026-07-14\n\n09:15 — line one\nline two\n^augi-aaaa1111\n",
        )

        resolved = resolve_anchor_refs("[[2026-07-14#^augi-aaaa1111]]", vault)

        assert resolved == [("[[2026-07-14#^augi-aaaa1111]]", "09:15 — line one\nline two")]

    def test_dangling_ref_is_none(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()

        resolved = resolve_anchor_refs("[[2026-07-14#^augi-gone0000]]", vault)

        assert resolved == [("[[2026-07-14#^augi-gone0000]]", None)]

    def test_duplicate_refs_resolved_once(self, tmp_path: Path):
        vault = tmp_path / "vault"
        _capture_note(vault, "2026-07-14", {"augi-aaaa1111": "thought"})
        content = "[[2026-07-14#^augi-aaaa1111]] [[2026-07-14#^augi-aaaa1111]]"

        resolved = resolve_anchor_refs(content, vault)

        assert len(resolved) == 1

    def test_falls_back_to_daily_note_outside_capture_folder(self, tmp_path: Path):
        vault = tmp_path / "vault"
        note = vault / "Daily" / "2026-07-14.md"
        note.parent.mkdir(parents=True)
        note.write_text("# 2026-07-14\n\nsome thought\n^augi-aaaa1111\n")

        resolved = resolve_anchor_refs("[[2026-07-14#^augi-aaaa1111]]", vault)

        assert resolved == [("[[2026-07-14#^augi-aaaa1111]]", "some thought")]

    def test_build_task_file_inlines_referenced_blocks(self, tmp_path: Path):
        vault = tmp_path / "vault"
        _capture_note(
            vault,
            "2026-07-14",
            {"augi-aaaa1111": "first gathered thought", "augi-bbbb2222": "second thought"},
        )
        block = _make_block("a" * 16, LENS_CONTENT, zzz=["apply lens nuggets"])

        result = build_task_file(block, vault_path=vault)

        assert "### Referenced blocks" in result
        assert "first gathered thought" in result
        assert "second thought" in result
        # Resolved context lands in ## Context, before ## User instruction
        assert result.index("first gathered thought") < result.index("## User instruction")

    def test_build_task_file_marks_unresolved_refs(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        block = _make_block("a" * 16, LENS_CONTENT, zzz=["apply lens nuggets"])

        result = build_task_file(block, vault_path=vault)

        assert "[[2026-07-14#^augi-aaaa1111]]: (unresolved)" in result

    def test_build_task_file_without_refs_unchanged(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        block = _make_block("a" * 16, "plain content", zzz=["do this"])

        assert "### Referenced blocks" not in build_task_file(block, vault_path=vault)

    def test_dispatch_writes_resolved_context(self, tmp_path: Path):
        vault = tmp_path / "vault"
        _capture_note(vault, "2026-07-14", {"augi-aaaa1111": "the gathered idea"})
        content = "gathered 1 block:\n[[2026-07-14#^augi-aaaa1111]]\nzzz: apply lens nuggets"
        blocks = [_make_block("a" * 16, content, zzz=["apply lens nuggets"])]

        written = dispatch_zzz_blocks(blocks, vault)

        assert len(written) == 1
        assert "the gathered idea" in written[0].read_text()


class TestReviewCLI:
    def test_review_writes_task_file(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from openaugi.cli.main import app

        monkeypatch.setattr("openaugi.config.load_config", lambda: {})
        runner = CliRunner()
        result = runner.invoke(app, ["review", "--path", str(tmp_path)])
        assert result.exit_code == 0
        tasks = list((tmp_path / "OpenAugi" / "Tasks").glob("run-the-review-pass-*.md"))
        assert len(tasks) == 1
        text = tasks[0].read_text()
        assert "status: pending" in text
        assert "run the review pass" in text

    def test_review_dashboard_only(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from openaugi.cli.main import app

        monkeypatch.setattr("openaugi.config.load_config", lambda: {})
        runner = CliRunner()
        result = runner.invoke(app, ["review", "--path", str(tmp_path), "--dashboard-only"])
        assert result.exit_code == 0
        tasks = list((tmp_path / "OpenAugi" / "Tasks").glob("process-the-dashboard-*.md"))
        assert len(tasks) == 1


# ── The zzz queue ──────────────────────────────────────────────────────────
#
# Regression cover for the 2026-09-01 double dispatch: one instruction, typed
# in two passes, produced two tasks. See dispatch.py's module docstring.


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(str(tmp_path / "test.db"))
    yield s
    s.close()


def _ledger(store: SQLiteStore, block_id: str) -> dict:
    rows = store.list_records(ZZZ_QUEUE_COLLECTION, limit=100)
    return next(r for r in rows if r["id"] == block_id)


def _tasks(vault: Path) -> list[Path]:
    return sorted((vault / "OpenAugi" / "Tasks").glob("*.md"))


def test_zzz_does_not_dispatch_before_it_settles(tmp_path: Path, store: SQLiteStore):
    block = _make_block("aaa1", "note", zzz=["read this voice"])
    store.insert_blocks([block])

    record_zzz_changes([block], [], store, tmp_path)
    assert _ledger(store, "aaa1")["status"] == QUEUED

    # Settle window still open — nothing written.
    assert drain_zzz_queue(store, tmp_path, settle_seconds=3600) == []
    assert _tasks(tmp_path) == []

    # Window elapsed — one task, once.
    assert len(drain_zzz_queue(store, tmp_path, settle_seconds=0)) == 1
    assert _ledger(store, "aaa1")["status"] == DISPATCHED
    assert drain_zzz_queue(store, tmp_path, settle_seconds=0) == []
    assert len(_tasks(tmp_path)) == 1


def test_instruction_edited_while_settling_dispatches_once(tmp_path: Path, store: SQLiteStore):
    """The 2026-09-01 bug, caught inside the settle window."""
    draft = _make_block("aaa1", "note", zzz=["read this voice"])
    store.insert_blocks([draft])
    record_zzz_changes([draft], [], store, tmp_path)

    # Sentence finished: the old block is deleted, a new one inserted.
    final = _make_block("aaa2", "note", zzz=["read this voice note, then invert"])
    store.delete_block("aaa1")
    store.insert_blocks([final])
    record_zzz_changes([final], [draft], store, tmp_path)

    assert _ledger(store, "aaa1")["status"] == SUPERSEDED
    written = drain_zzz_queue(store, tmp_path, settle_seconds=0)
    assert len(written) == 1
    assert "then invert" in written[0].read_text()


def test_edit_after_dispatch_supersedes_the_launched_task(tmp_path: Path, store: SQLiteStore):
    """The 2026-09-01 bug as it actually happened — 11 minutes apart, so the
    draft had already become a task. The successor retires it."""
    draft = _make_block("aaa1", "note", zzz=["read this voice"])
    store.insert_blocks([draft])
    record_zzz_changes([draft], [], store, tmp_path)
    (first,) = drain_zzz_queue(store, tmp_path, settle_seconds=0)

    final = _make_block("aaa2", "note", zzz=["read this voice note, then invert"])
    store.delete_block("aaa1")
    store.insert_blocks([final])
    record_zzz_changes([final], [draft], store, tmp_path)

    assert "status: superseded" in first.read_text()
    assert "Superseded:" in first.read_text()

    (second,) = drain_zzz_queue(store, tmp_path, settle_seconds=0)
    assert "then invert" in second.read_text()
    # Two files on disk, but only one live task — the draft is retired.
    assert len(_tasks(tmp_path)) == 2


def test_deleting_a_zzz_line_before_dispatch_writes_no_task(tmp_path: Path, store: SQLiteStore):
    block = _make_block("aaa1", "note", zzz=["never mind"])
    store.insert_blocks([block])
    record_zzz_changes([block], [], store, tmp_path)

    store.delete_block("aaa1")
    record_zzz_changes([], [block], store, tmp_path)

    assert _ledger(store, "aaa1")["status"] == SUPERSEDED
    assert drain_zzz_queue(store, tmp_path, settle_seconds=0) == []
    assert _tasks(tmp_path) == []


def test_block_that_vanishes_between_cycles_is_dropped(tmp_path: Path, store: SQLiteStore):
    """Belt and braces: if the removal is never reported, the drain still
    refuses to dispatch an instruction that is no longer in the vault."""
    block = _make_block("aaa1", "note", zzz=["read this voice"])
    store.insert_blocks([block])
    record_zzz_changes([block], [], store, tmp_path)
    store.delete_block("aaa1")

    assert drain_zzz_queue(store, tmp_path, settle_seconds=0) == []
    assert _ledger(store, "aaa1")["reason"] == "vanished"


def test_two_separate_instructions_in_one_note_both_dispatch(tmp_path: Path, store: SQLiteStore):
    a = _make_block("aaa1", "one", zzz=["first thing"])
    b = _make_block("bbb1", "two", zzz=["second thing"])
    store.insert_blocks([a, b])
    record_zzz_changes([a, b], [], store, tmp_path)

    assert len(drain_zzz_queue(store, tmp_path, settle_seconds=0)) == 2


def test_unrelated_removal_does_not_supersede(tmp_path: Path, store: SQLiteStore):
    """A block removed in the same cycle that never owed us a task must not
    consume the pairing slot of a genuinely new instruction."""
    plain = _make_block("old1", "some prose with no instruction")
    fresh = _make_block("aaa1", "note", zzz=["do the thing"])
    store.insert_blocks([fresh])
    record_zzz_changes([fresh], [plain], store, tmp_path)

    assert _ledger(store, "aaa1")["status"] == QUEUED
    assert len(drain_zzz_queue(store, tmp_path, settle_seconds=0)) == 1


def test_supersede_finds_the_task_after_the_watcher_renames_it(tmp_path: Path, store: SQLiteStore):
    """Hydration renames the file to TASK-<id>.md, so the name in the ledger
    goes stale. `source_block_id` is what we match on."""
    from openaugi.agents.task_watcher import hydrate_note

    draft = _make_block("aaa1", "note", zzz=["read this voice"])
    store.insert_blocks([draft])
    record_zzz_changes([draft], [], store, tmp_path)
    (first,) = drain_zzz_queue(store, tmp_path, settle_seconds=0)

    _, _, renamed = hydrate_note(first)
    assert renamed != first and renamed.name.startswith("TASK-")

    final = _make_block("aaa2", "note", zzz=["read this voice note, then invert"])
    store.delete_block("aaa1")
    store.insert_blocks([final])
    record_zzz_changes([final], [draft], store, tmp_path)

    assert "status: superseded" in renamed.read_text()
