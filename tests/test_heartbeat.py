"""Tests for the heartbeat pipeline.

Covers:
- Timestamp persistence (get/set last heartbeat)
- find_new_blocks filtering by block_time
- build_prompt formatting (skill file ref, zzz instructions, log path)
- run_heartbeat orchestration with a mocked agent launch
- Skill file missing → loud failure
- Empty window still advances the timestamp
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.pipeline import heartbeat as hb
from openaugi.store.sqlite import SQLiteStore


@pytest.fixture
def tmp_state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the heartbeat state file to a tmp path so tests don't clobber each other."""
    state = tmp_path / "last_heartbeat"
    monkeypatch.setattr(hb, "HEARTBEAT_STATE_FILE", state)
    return state


@pytest.fixture
def vault_with_skill(tmp_path: Path) -> Path:
    """A tmp vault with a heartbeat skill file already in place."""
    vault = tmp_path / "vault"
    (vault / "OpenAugi").mkdir(parents=True)
    (vault / "OpenAugi" / "heartbeat-skill.md").write_text(
        "# Heartbeat Skill\n\n## Workstreams\n- openaugi\n- self\n",
        encoding="utf-8",
    )
    return vault


def _make_entry(
    id_: str,
    content: str,
    source_path: str = "journal.md",
    zzz: list[str] | str | None = None,
) -> Block:
    metadata: dict = {"source_path": source_path}
    if zzz:
        # Accept a single string for test ergonomics; normalize to list.
        metadata["zzz_instructions"] = [zzz] if isinstance(zzz, str) else list(zzz)
    return Block(
        id=id_,
        kind="data_block",
        content=content,
        source="vault",
        title="journal",
        content_hash=id_,
        metadata=metadata,
    )


class TestHeartbeatState:
    def test_no_state_file_returns_none(self, tmp_state_file: Path):
        assert hb.get_last_heartbeat() is None

    def test_set_then_get(self, tmp_state_file: Path):
        hb.set_last_heartbeat("2026-04-08T06:14:00Z")
        assert hb.get_last_heartbeat() == "2026-04-08T06:14:00Z"

    def test_set_creates_parent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        state = tmp_path / "deep" / "nested" / "last_heartbeat"
        monkeypatch.setattr(hb, "HEARTBEAT_STATE_FILE", state)
        hb.set_last_heartbeat("2026-04-08T06:14:00Z")
        assert state.exists()


class TestFindNewBlocks:
    def test_first_run_returns_all(self, store: SQLiteStore):
        store.insert_blocks([_make_entry("a" * 16, "first")])
        store.insert_blocks([_make_entry("b" * 16, "second")])
        blocks = hb.find_new_blocks(store, since=None)
        assert len(blocks) == 2

    def test_filters_by_block_time(self, store: SQLiteStore):
        # Insert two entries with distinct block_time values
        older = Block(
            id="a" * 16,
            kind="data_block",
            content="first",
            source="vault",
            title="journal",
            content_hash="a" * 16,
            metadata={"source_path": "journal.md"},
            block_time="2024-01-01",
        )
        newer = Block(
            id="b" * 16,
            kind="data_block",
            content="second",
            source="vault",
            title="journal",
            content_hash="b" * 16,
            metadata={"source_path": "journal.md"},
            block_time="2024-06-01",
        )
        store.insert_blocks([older, newer])

        # Filter: only entries with block_time > "2024-01-01"
        blocks = hb.find_new_blocks(store, since="2024-01-01")
        ids = [b.id for b in blocks]
        assert "b" * 16 in ids
        assert "a" * 16 not in ids

    def test_respects_max_blocks(self, store: SQLiteStore):
        store.insert_blocks([_make_entry(f"{i:016x}", f"entry {i}") for i in range(5)])
        blocks = hb.find_new_blocks(store, since=None, max_blocks=3)
        assert len(blocks) == 3

    def test_only_returns_data_block_kind(self, store: SQLiteStore):
        store.insert_blocks(
            [
                _make_entry("a" * 16, "entry content"),
                Block(id="t" * 16, kind="context_block:tag", title="research", source="vault"),
            ]
        )
        blocks = hb.find_new_blocks(store, since=None)
        assert all(b.kind == "data_block" for b in blocks)
        assert len(blocks) == 1


class TestBuildPrompt:
    def test_prompt_includes_skill_file_reference(self, tmp_path: Path):
        skill = tmp_path / "skill.md"
        log = tmp_path / "log.md"
        blocks = [_make_entry("a" * 16, "some content")]
        prompt = hb.build_prompt(
            blocks=blocks,
            skill_file=skill,
            heartbeat_log=log,
            since=None,
            now="2026-04-08T06:14:00Z",
        )
        assert str(skill) in prompt
        assert str(log) in prompt

    def test_prompt_surfaces_zzz_instructions(self, tmp_path: Path):
        blocks = [
            _make_entry(
                "a" * 16,
                "Had a thought about embeddings",
                zzz=["research this more"],
            )
        ]
        prompt = hb.build_prompt(
            blocks=blocks,
            skill_file=tmp_path / "skill.md",
            heartbeat_log=tmp_path / "log.md",
            since=None,
            now="2026-04-08T06:14:00Z",
        )
        assert "User instructions:" in prompt
        assert "research this more" in prompt
        assert "embeddings" in prompt

    def test_prompt_surfaces_multiple_zzz_instructions(self, tmp_path: Path):
        blocks = [
            _make_entry(
                "a" * 16,
                "Thinking about workstreams",
                zzz=["research this", "also tag openaugi/design"],
            )
        ]
        prompt = hb.build_prompt(
            blocks=blocks,
            skill_file=tmp_path / "skill.md",
            heartbeat_log=tmp_path / "log.md",
            since=None,
            now="2026-04-08T06:14:00Z",
        )
        # Both instructions should appear as separate bullet lines.
        block_section = prompt.split("### Block 1")[1].split("##")[0]
        assert "User instructions:" in block_section
        assert "- research this" in block_section
        assert "- also tag openaugi/design" in block_section

    def test_prompt_handles_no_zzz(self, tmp_path: Path):
        blocks = [_make_entry("a" * 16, "plain content")]
        prompt = hb.build_prompt(
            blocks=blocks,
            skill_file=tmp_path / "skill.md",
            heartbeat_log=tmp_path / "log.md",
            since=None,
            now="2026-04-08T06:14:00Z",
        )
        # The block's metadata section should have no user-instruction bullet.
        block_section = prompt.split("### Block 1")[1].split("##")[0]
        assert "User instructions:" not in block_section
        assert "plain content" in block_section

    def test_prompt_truncates_long_content(self, tmp_path: Path):
        long_content = "x" * 2000
        blocks = [_make_entry("a" * 16, long_content)]
        prompt = hb.build_prompt(
            blocks=blocks,
            skill_file=tmp_path / "skill.md",
            heartbeat_log=tmp_path / "log.md",
            since=None,
            now="2026-04-08T06:14:00Z",
        )
        # Preview cap is 800 chars + ellipsis
        assert "…" in prompt


class TestRunHeartbeat:
    def test_missing_skill_file_raises(
        self, store: SQLiteStore, tmp_path: Path, tmp_state_file: Path
    ):
        vault = tmp_path / "vault"
        vault.mkdir()
        with pytest.raises(FileNotFoundError, match="Heartbeat skill file"):
            hb.run_heartbeat(store=store, vault_path=vault)

    def test_empty_window_advances_timestamp(
        self,
        store: SQLiteStore,
        vault_with_skill: Path,
        tmp_state_file: Path,
    ):
        result = hb.run_heartbeat(store=store, vault_path=vault_with_skill)
        assert result["block_count"] == 0
        assert result["launched"] is False
        assert tmp_state_file.exists()

    def test_dry_run_does_not_launch(
        self,
        store: SQLiteStore,
        vault_with_skill: Path,
        tmp_state_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        store.insert_blocks([_make_entry("a" * 16, "content", zzz="research this")])

        called = {"count": 0}

        def fake_launch(prompt: str) -> int:
            called["count"] += 1
            return 0

        monkeypatch.setattr(hb, "launch_agent", fake_launch)
        result = hb.run_heartbeat(store=store, vault_path=vault_with_skill, dry_run=True)
        assert called["count"] == 0
        assert result["launched"] is False
        assert result["block_count"] == 1
        assert "research this" in result["prompt"]
        # Dry run should NOT advance the timestamp
        assert not tmp_state_file.exists()

    def test_successful_agent_advances_timestamp(
        self,
        store: SQLiteStore,
        vault_with_skill: Path,
        tmp_state_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        store.insert_blocks([_make_entry("a" * 16, "content")])
        monkeypatch.setattr(hb, "launch_agent", lambda prompt: 0)

        result = hb.run_heartbeat(store=store, vault_path=vault_with_skill)
        assert result["launched"] is True
        assert result["return_code"] == 0
        assert tmp_state_file.exists()
        assert hb.get_last_heartbeat() == result["now"]

    def test_failed_agent_leaves_timestamp(
        self,
        store: SQLiteStore,
        vault_with_skill: Path,
        tmp_state_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        store.insert_blocks([_make_entry("a" * 16, "content")])
        monkeypatch.setattr(hb, "launch_agent", lambda prompt: 2)

        result = hb.run_heartbeat(store=store, vault_path=vault_with_skill)
        assert result["return_code"] == 2
        # Timestamp NOT advanced so next run retries
        assert not tmp_state_file.exists()

    def test_batch_cap_flagged(
        self,
        store: SQLiteStore,
        vault_with_skill: Path,
        tmp_state_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        store.insert_blocks([_make_entry(f"{i:016x}", f"entry {i}") for i in range(5)])
        monkeypatch.setattr(hb, "launch_agent", lambda prompt: 0)

        result = hb.run_heartbeat(store=store, vault_path=vault_with_skill, max_blocks=3)
        assert result["block_count"] == 3
        assert result["batch_capped"] is True
