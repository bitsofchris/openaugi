"""The stored graph must say what the vault says — for every variant of drift.

The augi_id link bug (docs/reference/data-model.md § Document identity) was one
instance of a class: an edge keyed by one identity rule while its endpoint is
keyed by another, an incremental ingest ending where a fresh ingest would not,
the valid-endpoint filter dropping edges silently, records left behind. This
file tests the class, not the instance:

1. The integrity checker itself: each check detects the defect it names.
2. An oracle: links derived from the markdown by a deliberately simple,
   independent parser, compared to the stored `links_to` edges per wikilink form.
3. Incremental equals full: lifecycle steps, each followed by an incremental
   ingest, must end in the same graph as one fresh ingest of the final vault.
4. The runtime guard: unexpected dropped edges are a WARNING.

Known-open drift is `xfail(strict=True)` with its reason, so it stays visible
and turns into an XPASS failure the day it is fixed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.pipeline.integrity import check_graph_integrity, errors
from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore
from tests.graph_integrity import assert_graph_integrity, graph_snapshot, snapshot_diff


def _write(vault: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return vault


def _checks(store: SQLiteStore, vault: Path | None = None) -> set[str]:
    return {f.check for f in errors(check_graph_integrity(store, vault))}


# ── 1. The checker ─────────────────────────────────────────────────


BASE = {
    "Hub.md": "### 2022-01-01\nHub idea.\n",
    "Linker.md": "### 2022-01-02\nSee [[Hub]].\n",
    "Other.md": "### 2022-01-03\nAlso [[Hub]] and [[Linker]].\n",
}


@pytest.fixture
def ingested(tmp_path: Path, store: SQLiteStore) -> Path:
    vault = _write(tmp_path / "v", BASE)
    run_layer0(vault, store)
    return vault


def _entry(store: SQLiteStore, rel: str) -> str:
    return store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'data_block' "
        "AND json_extract(metadata, '$.source_path') = ?",
        (rel,),
    ).fetchone()[0]


class TestChecker:
    def test_fixture_vault_is_clean(self, vault_path: Path, store: SQLiteStore):
        run_layer0(vault_path, store)
        assert errors(check_graph_integrity(store, vault_path)) == []

    def test_clean_after_ingest(self, ingested: Path, store: SQLiteStore):
        assert errors(check_graph_integrity(store, ingested)) == []

    @pytest.mark.graph_drift_expected(reason="plants each defect on purpose")
    def test_duplicate_document(self, ingested: Path, store: SQLiteStore):
        store.insert_blocks(
            [
                Block(
                    id="dup",
                    kind="context_block:document",
                    source="vault",
                    metadata={"source_path": "Hub.md"},
                )
            ]
        )
        assert "duplicate_document" in _checks(store)

    @pytest.mark.graph_drift_expected(reason="plants each defect on purpose")
    def test_orphan_and_wrong_document(self, ingested: Path, store: SQLiteStore):
        hub_entry = _entry(store, "Hub.md")
        hub_doc = Block.make_document_id("Hub.md")
        store.delete_link(hub_entry, hub_doc, "contains")
        assert "orphan_data_block" in _checks(store)
        store.insert_links(
            [Link(from_id=hub_entry, to_id=Block.make_document_id("Other.md"), kind="contains")]
        )
        assert "contains_wrong_document" in _checks(store)
        store.insert_links([Link(from_id=hub_entry, to_id=hub_doc, kind="contains")])
        assert "multiple_contains" in _checks(store)

    @pytest.mark.graph_drift_expected(reason="plants each defect on purpose")
    def test_dangling_edge(self, ingested: Path, store: SQLiteStore):
        store.conn.execute("PRAGMA foreign_keys = OFF")
        store.conn.execute(
            "INSERT INTO links (from_id, to_id, kind) VALUES (?, 'nowhere', 'links_to')",
            (_entry(store, "Linker.md"),),
        )
        store.conn.commit()
        assert "dangling_edge" in _checks(store)

    @pytest.mark.graph_drift_expected(reason="plants each defect on purpose")
    def test_edge_into_a_non_document(self, ingested: Path, store: SQLiteStore):
        store.insert_links(
            [
                Link(
                    from_id=_entry(store, "Linker.md"),
                    to_id=_entry(store, "Hub.md"),
                    kind="links_to",
                )
            ]
        )
        assert "edge_target_not_document" in _checks(store)

    @pytest.mark.graph_drift_expected(reason="plants each defect on purpose")
    def test_missing_and_extra_links(self, ingested: Path, store: SQLiteStore):
        linker, hub = _entry(store, "Linker.md"), Block.make_document_id("Hub.md")
        store.delete_link(linker, hub, "links_to")
        assert "links_to_missing" in _checks(store, ingested)
        store.insert_links(
            [Link(from_id=linker, to_id=Block.make_document_id("Other.md"), kind="links_to")]
        )
        assert "links_to_extra" in _checks(store, ingested)

    @pytest.mark.graph_drift_expected(reason="plants each defect on purpose")
    def test_document_identity_and_file(self, ingested: Path, store: SQLiteStore):
        # Hub gains an id on disk; the DB is told the file did not change.
        text = "---\naugi_id: hub-1\n---\n\n" + BASE["Hub.md"]
        (ingested / "Hub.md").write_text(text)
        from openaugi.adapters.vault import _hash_file

        store.update_block_hash(Block.make_document_id("Hub.md"), _hash_file(ingested / "Hub.md"))
        assert "document_identity" in _checks(store, ingested)
        (ingested / "Other.md").unlink()
        assert "document_without_file" in _checks(store, ingested)

    def test_changed_notes_are_not_drift(self, ingested: Path, store: SQLiteStore):
        """A note edited after the last ingest is skipped, not reported."""
        (ingested / "Linker.md").write_text("### 2022-01-02\nSee [[Other]] instead.\n")
        assert _checks(store, ingested) == set()

    def test_unresolved_links_are_info(self, tmp_path: Path, store: SQLiteStore):
        vault = _write(tmp_path / "v", {"A.md": "Points at [[Ghost]].\n"})
        run_layer0(vault, store)
        findings = check_graph_integrity(store, vault)
        assert errors(findings) == []
        (info,) = [f for f in findings if f.check == "unresolved_wikilink"]
        assert info.severity == "info" and info.examples == ["[[Ghost]] ×1"]


# ── 2. The oracle ──────────────────────────────────────────────────
#
# How Obsidian resolves a wikilink, written independently of the adapter:
# `[[target|alias]]`, `[[target#heading]]`, `[[target#^block]]`, `![[embed]]`;
# a target with a `/` is a vault path; otherwise the note whose file name
# matches, case-insensitively, the shortest path winning. Links inside fenced
# code are text, not links. Frontmatter property links count.

_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
_FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def oracle_links(vault: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """({(from note, to note)}, {(from note, raw target)} that resolve to nothing)."""
    notes = sorted(
        (p.relative_to(vault).as_posix() for p in vault.rglob("*.md")),
        key=lambda r: (r.count("/"), r),
    )
    resolved: set[tuple[str, str]] = set()
    unresolved: set[tuple[str, str]] = set()
    for src in notes:
        text = _FENCE.sub("", (vault / src).read_text(encoding="utf-8"))
        for raw in _WIKILINK.findall(text):
            target = raw.split("|")[0].split("#")[0].strip()
            if not target:
                continue  # [[#heading]] — a link within the same note
            target = target.removesuffix(".md")
            if "/" in target:
                hits = [n for n in notes if n == f"{target}.md"]
            else:
                hits = [n for n in notes if Path(n).stem.lower() == target.lower()]
            if hits:
                resolved.add((src, hits[0]))
            else:
                unresolved.add((src, raw))
    return resolved, unresolved


def stored_links(store: SQLiteStore) -> set[tuple[str, str]]:
    return set(
        store.conn.execute(
            """SELECT json_extract(a.metadata, '$.source_path'),
                      json_extract(b.metadata, '$.source_path')
               FROM links l
               JOIN blocks a ON a.id = l.from_id AND a.kind = 'data_block'
               JOIN blocks b ON b.id = l.to_id
               WHERE l.kind = 'links_to'"""
        ).fetchall()
    )


def test_oracle_on_the_fixture_vault(vault_path: Path, store: SQLiteStore):
    run_layer0(vault_path, store)
    want, unresolved = oracle_links(vault_path)
    assert stored_links(store) == want
    # Unresolved targets are pinned, not silently equal: a new one is a change.
    assert {raw for _, raw in unresolved} == {
        "Architecture Review",
        "Security Audit",
        "Personal Growth",
        "Book Notes - Thinking in Systems",
    }


def _xfail(reason: str):
    return pytest.mark.xfail(strict=True, reason=reason)


TARGETS = {
    "Target.md": "### 2022-01-01\nThe target.\n\n#### Heading\nUnder a heading. ^blk\n",
    "deep/Nested.md": "### 2022-01-01\nNested.\n",
    "Keyed.md": "---\naugi_id: keyed-1\n---\n\n### 2022-01-01\nKeyed.\n",
    "Twin.md": "### 2022-01-01\nThe root twin.\n",
    "folder/Twin.md": "### 2022-01-01\nThe nested twin.\n",
}

LINK_FORMS = [
    pytest.param("See [[Target]].", id="plain"),
    pytest.param("See [[Target|an alias]].", id="alias"),
    pytest.param("See ![[Target]].", id="embed"),
    pytest.param("See [[Keyed]].", id="augi_target"),
    pytest.param("See [[Twin]].", id="same_title_two_folders"),
    pytest.param(
        "See [[Target#Heading]].",
        id="heading",
        marks=_xfail("[[note#heading]] is looked up as a file named 'note#heading'"),
    ),
    pytest.param(
        "See [[Target#^blk]].",
        id="block_ref",
        marks=_xfail("[[note#^block]] is looked up as a file named 'note#^block'"),
    ),
    pytest.param("See [[deep/Nested]].", id="path_form"),
    pytest.param(
        "See [[target]].",
        id="case_insensitive",
        marks=_xfail("file-name lookup is case-sensitive; Obsidian's is not"),
    ),
    pytest.param(
        "See [[Target.md]].",
        id="md_suffix",
        marks=_xfail("[[note.md]] is looked up as a file named 'note.md.md'"),
    ),
    pytest.param(
        "Text.\n\n```\n[[Target]]\n```\n",
        id="inside_code_fence",
        marks=_xfail("a wikilink inside fenced code is stored as an edge"),
    ),
    pytest.param(
        '---\nrelated:\n  - "[[Target]]"\n---\n\nText.',
        id="frontmatter_property",
        marks=_xfail("frontmatter is stripped before links are extracted"),
    ),
]


@pytest.mark.parametrize("body", LINK_FORMS)
def test_oracle_per_link_form(tmp_path: Path, store: SQLiteStore, body: str):
    """Each wikilink form, alone in one note, against the independent oracle."""
    text = body if body.startswith("---") else f"### 2022-01-02\n{body}\n"
    vault = _write(tmp_path / "v", {**TARGETS, "Linker.md": text})
    run_layer0(vault, store)
    want, _ = oracle_links(vault)
    got = stored_links(store)
    assert {p for p in got if p[0] == "Linker.md"} == {p for p in want if p[0] == "Linker.md"}


# ── 3. Incremental equals full ─────────────────────────────────────

Step = Callable[[Path], None]


def write(rel: str, text: str) -> Step:
    return lambda v: _write(v, {rel: text}) and None


def delete(rel: str) -> Step:
    return lambda v: (v / rel).unlink()


def move(src: str, dst: str) -> Step:
    def step(v: Path) -> None:
        (v / dst).parent.mkdir(parents=True, exist_ok=True)
        (v / src).rename(v / dst)

    return step


AUGI_HUB = "---\naugi_id: hub-1\n---\n\n" + BASE["Hub.md"]

DELETE_LEAVES_ENTRIES = (
    "a deleted (or renamed, or moved) note's data_blocks outlive its document: "
    "deleting the document cascades its contains edges, not its entries"
)
AUGI_PATH_IS_STALE = (
    "an augi-keyed document keeps its first source_path: a move or rename is "
    "never recorded and a delete never finds the document (it looks up the path id)"
)
LINK_BEFORE_TARGET = (
    "a link to a note that does not exist yet is dropped, and the linking note "
    "is never re-parsed when the target appears"
)

SCENARIOS = [
    pytest.param([write("New.md", "### 2022-02-01\nNew, about [[Hub]].\n")], id="add_note"),
    pytest.param(
        [write("Linker.md", "### 2022-01-02\nSee [[Hub]] again.\n\n### 2022-02-02\n[[Other]].\n")],
        id="edit_note",
    ),
    pytest.param(
        [
            write(
                "Other.md", "### 2022-01-03\nAlso [[Hub]] and [[Linker]].\n\n### 2022-02-03\nx\n"
            ),
            write("Other.md", "### 2022-02-03\nx\n"),
        ],
        id="add_then_remove_section",
    ),
    pytest.param([write("Hub.md", AUGI_HUB)], id="add_augi_id_to_linked_note"),
    pytest.param(
        [write("Hub.md", AUGI_HUB), write("Hub.md", BASE["Hub.md"])],
        id="add_then_remove_augi_id",
    ),
    pytest.param(
        [write("Hub.md", AUGI_HUB), write("Linker.md", "### 2022-01-02\n[[Hub]], edited.\n")],
        id="augi_id_then_linker_edit",
    ),
    pytest.param(
        [write("sub/Hub.md", "### 2022-02-01\nA second Hub.\n")],
        id="same_title_two_folders",
    ),
    pytest.param(
        [write("Linker.md", "### 2022-01-02\n[[Hub|the hub]], [[Hub#Heading]].\n")],
        id="alias_and_heading_forms",
    ),
    pytest.param(
        [delete("Other.md")], id="delete_plain_note", marks=_xfail(DELETE_LEAVES_ENTRIES)
    ),
    pytest.param(
        [move("Hub.md", "Hub Renamed.md")],
        id="rename_plain_note",
        marks=_xfail(DELETE_LEAVES_ENTRIES),
    ),
    pytest.param(
        [write("Hub.md", AUGI_HUB), move("Hub.md", "Hub Renamed.md")],
        id="rename_augi_note",
        marks=_xfail(AUGI_PATH_IS_STALE),
    ),
    pytest.param(
        [write("Hub.md", AUGI_HUB), move("Hub.md", "sub/Hub.md")],
        id="move_augi_note",
        marks=_xfail(AUGI_PATH_IS_STALE),
    ),
    pytest.param(
        [write("Hub.md", AUGI_HUB), delete("Hub.md")],
        id="delete_augi_note",
        marks=_xfail(AUGI_PATH_IS_STALE),
    ),
    pytest.param(
        [
            write("Linker.md", "### 2022-01-02\nSee [[Hub]] and [[Later]].\n"),
            write("Later.md", "### 2022-02-01\nArrived.\n"),
        ],
        id="link_before_target_exists",
        marks=_xfail(LINK_BEFORE_TARGET),
    ),
]


@pytest.mark.graph_drift_expected(reason="checks integrity itself, step by step")
@pytest.mark.parametrize("steps", SCENARIOS)
def test_incremental_equals_full(tmp_path: Path, store: SQLiteStore, steps: list[Step]):
    vault = _write(tmp_path / "v", BASE)
    run_layer0(vault, store)
    for step in steps:
        step(vault)
        run_layer0(vault, store)
        assert_graph_integrity(store, vault)

    full = SQLiteStore(tmp_path / "full.db")
    try:
        run_layer0(vault, full)
        assert_graph_integrity(full, vault)
        diff = snapshot_diff(graph_snapshot(store), graph_snapshot(full))
        assert not diff, "incremental ingest differs from a fresh one:\n" + diff
    finally:
        full.close()


# ── 4. The runtime guard ───────────────────────────────────────────


class TestSkippedLinkReport:
    def test_link_to_a_missing_note_is_not_a_warning(
        self, tmp_path: Path, store: SQLiteStore, caplog
    ):
        vault = _write(tmp_path / "v", {"A.md": "Points at [[Ghost]].\n"})
        with caplog.at_level(logging.INFO, logger="openaugi.pipeline.runner"):
            run_layer0(vault, store)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("not in the vault" in r.getMessage() for r in caplog.records)

    @pytest.mark.graph_drift_expected(reason="reproduces the drop on purpose")
    def test_misresolved_link_is_a_warning(
        self, tmp_path: Path, store: SQLiteStore, caplog, monkeypatch
    ):
        """The augi_id bug, replayed: the edge's target id is nobody's."""
        from openaugi.adapters import vault as vault_mod

        monkeypatch.setattr(
            vault_mod,
            "_resolve_document_id",
            lambda t, idx, _ids: Block.make_document_id(vault_mod._resolve_wikilink(t, idx)),
        )
        vault = _write(tmp_path / "v", {"Hub.md": AUGI_HUB, "Linker.md": BASE["Linker.md"]})
        with caplog.at_level(logging.INFO, logger="openaugi.pipeline.runner"):
            run_layer0(vault, store)
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "1 edge(s) dropped" in warning.getMessage()


class TestCheckGraphCli:
    def _run(self, store: SQLiteStore, vault: Path):
        from typer.testing import CliRunner

        from openaugi.cli.main import app

        return CliRunner().invoke(
            app, ["check-graph", "--db", str(store.db_path), "--path", str(vault)]
        )

    def test_clean_graph_exits_zero(self, ingested: Path, store: SQLiteStore):
        result = self._run(store, ingested)
        assert result.exit_code == 0, result.output
        assert "No findings" in result.output

    @pytest.mark.graph_drift_expected(reason="plants a defect on purpose")
    def test_drift_exits_one_and_writes_nothing(self, ingested: Path, store: SQLiteStore):
        store.delete_link(_entry(store, "Linker.md"), Block.make_document_id("Hub.md"), "links_to")
        before = store.get_stats()
        store.close()
        result = self._run(store, ingested)
        assert result.exit_code == 1, result.output
        assert "links_to_missing" in result.output
        assert store.get_stats() == before
