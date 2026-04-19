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
from openaugi.pipeline.dispatch import build_task_file, dispatch_zzz_blocks


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
