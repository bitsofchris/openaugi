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

from openaugi.model.block import Block
from openaugi.pipeline.dispatch import (
    build_task_file,
    dispatch_zzz_blocks,
    resolve_anchor_refs,
)


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
