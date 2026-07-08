"""Tests for the context-pack writer — the mobile capture-assist sidecar.

The output shape is pinned by private-augi-mobile's shared/contract.ts
ContextPack type: {agentFile, taxonomy, recentConcepts, noteTitles}.
"""

import json
from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.pipeline.context_pack import (
    DEFAULT_AGENT_FILE,
    OUTPUT_RELPATH,
    build_context_pack,
    write_context_pack,
)
from openaugi.store.sqlite import SQLiteStore


def _doc(title: str, path: str, block_time: str | None = None) -> Block:
    return Block(
        id=Block.make_document_id(path),
        kind="context_block:document",
        title=title,
        source="vault",
        block_time=block_time,
        metadata={"source_path": path},
    )


def _tag(name: str) -> Block:
    return Block(id=Block.make_tag_id(name), kind="context_block:tag", title=name, source="vault")


def _entry(content: str, path: str) -> Block:
    return Block(
        id=Block.make_id(path, Block.hash_content(content)),
        kind="data_block",
        content=content,
        source="vault",
        metadata={"source_path": path},
    )


@pytest.fixture
def populated(store: SQLiteStore) -> SQLiteStore:
    """Store with two containers, three docs, tags, and routing activity."""
    amoc = _doc("AMOC - OpenAugi Main", "OpenAugi/AMOC - OpenAugi Main.md", "2026-07-01")
    pmoc = _doc("PMOC - Season 2", "OpenAugi/PMOC - Season 2.md", "2026-07-02")
    note = _doc("Local-first capture", "Notes/Local-first capture.md", "2026-07-05")
    task = _doc("TASK-2026-07-06-x", "OpenAugi/Tasks/TASK-2026-07-06-x.md", "2026-07-06")
    entry_a = _entry("thought a", "Daily/2026-07-05.md")
    entry_b = _entry("thought b", "Daily/2026-07-06.md")
    tag_todo = _tag("todo")
    tag_area = _tag("area/openaugi")
    store.insert_blocks([amoc, pmoc, note, task, entry_a, entry_b, tag_todo, tag_area])
    store.insert_links(
        [
            # pmoc routed into more recently than amoc (link rows share ingest
            # time resolution, so route amoc only via the older entry).
            Link(from_id=entry_a.id, to_id=amoc.id, kind="routed_to"),
            Link(from_id=entry_a.id, to_id=tag_todo.id, kind="groups"),
            Link(from_id=entry_b.id, to_id=tag_area.id, kind="groups"),
            Link(from_id=entry_b.id, to_id=pmoc.id, kind="routed_to"),
        ]
    )
    return store


class TestBuildContextPack:
    def test_contract_shape(self, populated: SQLiteStore, tmp_path: Path):
        pack = build_context_pack(populated, tmp_path)
        assert set(pack) >= {"agentFile", "taxonomy", "recentConcepts", "noteTitles"}
        assert isinstance(pack["taxonomy"], list)
        assert all({"title", "path"} <= set(c) for c in pack["recentConcepts"])
        assert all(isinstance(t, str) for t in pack["noteTitles"])

    def test_recent_concepts_are_route_targets_with_paths(
        self, populated: SQLiteStore, tmp_path: Path
    ):
        pack = build_context_pack(populated, tmp_path)
        by_title = {c["title"]: c["path"] for c in pack["recentConcepts"]}
        assert by_title == {
            "AMOC - OpenAugi Main": "OpenAugi/AMOC - OpenAugi Main.md",
            "PMOC - Season 2": "OpenAugi/PMOC - Season 2.md",
        }

    def test_note_titles_containers_first_tasks_excluded(
        self, populated: SQLiteStore, tmp_path: Path
    ):
        pack = build_context_pack(populated, tmp_path)
        titles = pack["noteTitles"]
        assert set(titles[:2]) == {"AMOC - OpenAugi Main", "PMOC - Season 2"}
        assert "Local-first capture" in titles
        assert "TASK-2026-07-06-x" not in titles
        assert len(titles) == len(set(titles))

    def test_taxonomy_from_db_tags_when_no_note(self, populated: SQLiteStore, tmp_path: Path):
        pack = build_context_pack(populated, tmp_path)
        assert "#todo" in pack["taxonomy"]
        assert "#area/openaugi" in pack["taxonomy"]

    def test_taxonomy_note_comes_first(self, populated: SQLiteStore, tmp_path: Path):
        agent_dir = tmp_path / "OpenAugi" / "AGENT"
        agent_dir.mkdir(parents=True)
        (agent_dir / "My Taxonomy.md").write_text("Facets: #area/content #question and #todo.")
        pack = build_context_pack(populated, tmp_path)
        assert pack["taxonomy"][:3] == ["#area/content", "#question", "#todo"]
        # DB tags not in the note are appended, not lost
        assert "#area/openaugi" in pack["taxonomy"]

    def test_taxonomy_note_backticked_tags(self, populated: SQLiteStore, tmp_path: Path):
        # The real taxonomy note writes tags in table cells as `#status/active`.
        agent_dir = tmp_path / "OpenAugi" / "AGENT"
        agent_dir.mkdir(parents=True)
        (agent_dir / "My Taxonomy.md").write_text(
            "| `#status/active` | In flight |\n\n# Heading is not a tag\n"
        )
        pack = build_context_pack(populated, tmp_path)
        assert pack["taxonomy"][0] == "#status/active"
        assert "#Heading" not in pack["taxonomy"]

    def test_agent_file_default_and_override(self, populated: SQLiteStore, tmp_path: Path):
        assert build_context_pack(populated, tmp_path)["agentFile"] == DEFAULT_AGENT_FILE
        agent_dir = tmp_path / "OpenAugi" / "AGENT"
        agent_dir.mkdir(parents=True)
        (agent_dir / "capture-conventions.md").write_text("# My conventions\n")
        assert build_context_pack(populated, tmp_path)["agentFile"] == "# My conventions\n"

    def test_lenses_from_registry(self, populated: SQLiteStore, tmp_path: Path):
        lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
        lens_dir.mkdir(parents=True)
        (lens_dir / "nuggets.md").write_text(
            "---\nname: nuggets\ndescription: Find stand-alone insights.\n"
            "scope: recent writing\ntrigger: on-demand\ntarget: dashboard\n---\n\nIntent.\n"
        )
        (lens_dir / "no-frontmatter.md").write_text("just prose")
        pack = build_context_pack(populated, tmp_path)
        assert pack["lenses"] == [{"name": "nuggets", "description": "Find stand-alone insights."}]

    def test_lenses_empty_without_registry(self, populated: SQLiteStore, tmp_path: Path):
        assert build_context_pack(populated, tmp_path)["lenses"] == []

    def test_broken_yaml_lens_is_salvaged_not_dropped(
        self, populated: SQLiteStore, tmp_path: Path
    ):
        # The exact failure shipped on 2026-07-07: a description starting with
        # a quoted phrase is invalid YAML. The lens must still reach the pack.
        lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
        lens_dir.mkdir(parents=True)
        (lens_dir / "echoes.md").write_text(
            '---\nname: echoes\ndescription: "Have I thought this before?" — lineage: growth\n'
            "trigger: on-demand\n---\n\nIntent.\n"
        )
        pack = build_context_pack(populated, tmp_path)
        assert len(pack["lenses"]) == 1
        assert pack["lenses"][0]["name"] == "echoes"
        assert "Have I thought this before?" in pack["lenses"][0]["description"]

    def test_read_lens_specs_flags_errors(self, tmp_path: Path):
        from openaugi.pipeline.context_pack import read_lens_specs

        lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
        lens_dir.mkdir(parents=True)
        (lens_dir / "good.md").write_text(
            "---\nname: good\ndescription: >-\n  Fine.\nscope: >-\n  Recent writing.\n"
            "trigger: on-pass\ntarget: >-\n  dashboard\n---\nBody.\n"
        )
        (lens_dir / "bad.md").write_text(
            '---\nname: bad\ndescription: "quoted" — and: broken\n---\nBody.\n'
        )
        specs = {s["name"]: s for s in read_lens_specs(tmp_path)}
        assert "error" not in specs["good"]
        assert specs["good"]["trigger"] == "on-pass"
        assert "error" in specs["bad"]
        assert specs["bad"]["file"] == "bad.md"

    def test_empty_store(self, store: SQLiteStore, tmp_path: Path):
        pack = build_context_pack(store, tmp_path)
        assert pack["taxonomy"] == []
        assert pack["recentConcepts"] == []
        assert pack["noteTitles"] == []
        assert pack["agentFile"] == DEFAULT_AGENT_FILE


class TestWriteContextPack:
    def test_writes_json_at_contract_path(self, populated: SQLiteStore, tmp_path: Path):
        out = write_context_pack(populated, tmp_path)
        assert out == tmp_path / OUTPUT_RELPATH
        pack = json.loads(out.read_text())
        assert set(pack) >= {"agentFile", "taxonomy", "recentConcepts", "noteTitles"}
        assert pack["generatedAt"]

    def test_overwrites_on_rerun(self, populated: SQLiteStore, tmp_path: Path):
        first = write_context_pack(populated, tmp_path).read_text()
        second = write_context_pack(populated, tmp_path).read_text()
        assert json.loads(first)["noteTitles"] == json.loads(second)["noteTitles"]
