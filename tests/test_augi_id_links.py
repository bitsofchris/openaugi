"""Wikilinks into a note that carries an `augi_id` land on its real document.

A note with `augi_id` in its frontmatter is keyed on it, not on its path
(docs/reference/data-model.md § Document identity). Wikilinks into such a note
used to be resolved by path, so the edge pointed at an id the note did not
have: dropped at insert when nothing had that id, or attached to a stale
path-keyed document left from before the note was given its id. Either way
the note lost its backlinks without an error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openaugi.adapters.vault import parse_vault
from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.pipeline.runner import run_layer0
from openaugi.query import engine
from openaugi.store.sqlite import SQLiteStore

A_FM = "---\naugi_id: id-a\n---\n\n"
A_BODY = "### 2022-01-01\nFirst thought.\n\n### 2022-01-02\nSecond thought.\n"
B_BODY = "### 2022-01-03\nThis builds on [[Hub A]].\n"

A_AUGI = Block.make_document_id("augi:id-a")
A_PATH = Block.make_document_id("Hub A.md")


def _write_vault(root: Path, a_has_id: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "Hub A.md").write_text((A_FM if a_has_id else "") + A_BODY, encoding="utf-8")
    (root / "Note B.md").write_text(B_BODY, encoding="utf-8")
    return root


def _b_block_id(store: SQLiteStore) -> str:
    rows = store.conn.execute(
        "SELECT id FROM blocks WHERE kind = 'data_block' "
        "AND json_extract(metadata, '$.source_path') = 'Note B.md'"
    ).fetchall()
    assert len(rows) == 1
    return rows[0][0]


def _backlinks(store: SQLiteStore, doc_id: str) -> set[str]:
    return {i.block.id for i in engine.related(store, doc_id, kind="links_to", direction="in")}


def _docs_for(store: SQLiteStore, source_path: str) -> list[str]:
    return store.get_document_ids_by_source_path().get(source_path, [])


class TestParse:
    def test_link_targets_the_augi_keyed_document(self, tmp_path: Path):
        blocks, links = parse_vault(_write_vault(tmp_path / "v"))
        targets = {lnk.to_id for lnk in links if lnk.kind == "links_to"}
        assert targets == {A_AUGI}

    def test_link_to_a_plain_note_still_uses_the_path(self, tmp_path: Path):
        _, links = parse_vault(_write_vault(tmp_path / "v", a_has_id=False))
        targets = {lnk.to_id for lnk in links if lnk.kind == "links_to"}
        assert targets == {A_PATH}

    def test_link_to_a_missing_note_uses_the_path(self, tmp_path: Path):
        v = tmp_path / "v"
        v.mkdir()
        (v / "B.md").write_text("See [[Nowhere]].\n", encoding="utf-8")
        _, links = parse_vault(v)
        assert {lnk.to_id for lnk in links if lnk.kind == "links_to"} == {
            Block.make_document_id("Nowhere.md")
        }


class TestIngest:
    def test_get_related_in_returns_the_linking_block(self, tmp_path: Path, store: SQLiteStore):
        run_layer0(_write_vault(tmp_path / "v"), store)
        assert _backlinks(store, A_AUGI) == {_b_block_id(store)}
        assert store.get_block(A_PATH) is None

    def test_link_survives_an_incremental_run_of_the_linking_note(
        self, tmp_path: Path, store: SQLiteStore
    ):
        """B changes after A exists: its new edge must still find A."""
        v = _write_vault(tmp_path / "v")
        run_layer0(v, store)
        (v / "Note B.md").write_text(B_BODY + "\n### 2022-01-04\nAgain [[Hub A]].\n")
        run_layer0(v, store)
        assert len(_backlinks(store, A_AUGI)) == 2


class TestTargetGainsAnAugiId:
    """The incremental case: B does not change, A is given an augi_id later.

    B is never re-parsed, so its edge still points at A's old path-keyed
    document. Ingest folds that document into the new one.
    """

    def _ingest_then_add_id(self, tmp_path: Path, store: SQLiteStore) -> Path:
        v = _write_vault(tmp_path / "v", a_has_id=False)
        run_layer0(v, store)
        assert _backlinks(store, A_PATH) == {_b_block_id(store)}
        (v / "Hub A.md").write_text(A_FM + A_BODY, encoding="utf-8")
        return v

    def test_backlinks_follow_to_the_new_document(self, tmp_path: Path, store: SQLiteStore):
        v = self._ingest_then_add_id(tmp_path, store)
        run_layer0(v, store)
        assert _backlinks(store, A_AUGI) == {_b_block_id(store)}
        assert _docs_for(store, "Hub A.md") == [A_AUGI]

    def test_routes_and_recap_follow(self, tmp_path: Path, store: SQLiteStore):
        v = self._ingest_then_add_id(tmp_path, store)
        b_id = _b_block_id(store)
        store.insert_links([Link(from_id=b_id, to_id=A_PATH, kind="routed_to")])
        store.upsert_recap(A_PATH, "recap", "2022-01-05", "h")
        run_layer0(v, store)
        assert {lnk.from_id for lnk in store.get_links_to(A_AUGI, kind="routed_to")} == {b_id}
        assert store.get_recap(A_AUGI) is not None

    def test_entries_are_kept_and_removed_ones_retired(self, tmp_path: Path, store: SQLiteStore):
        v = self._ingest_then_add_id(tmp_path, store)
        before = {e.id: e for e in store.get_entries_for_document(A_PATH)}
        assert len(before) == 2
        # Same change also drops the second section.
        (v / "Hub A.md").write_text(A_FM + "### 2022-01-01\nFirst thought.\n")
        result = run_layer0(v, store)
        after = store.get_entries_for_document(A_AUGI)
        assert len(after) == 1 and after[0].id in before
        assert result["blocks_removed"] == 1


class TestMergeDocument:
    def _doc(self, store: SQLiteStore, doc_id: str) -> None:
        store.insert_blocks(
            [
                Block(
                    id=doc_id,
                    kind="context_block:document",
                    title="Hub",
                    metadata={"source_path": "Hub.md"},
                )
            ]
        )

    def test_moves_edges_and_drops_duplicates(self, store: SQLiteStore):
        self._doc(store, "old")
        self._doc(store, "new")
        store.insert_blocks(
            [Block(id=f"e{i}", kind="data_block", content=f"x{i}") for i in (1, 2)]
        )
        store.insert_links(
            [
                Link(from_id="e1", to_id="old", kind="links_to", metadata={"target_title": "Hub"}),
                Link(from_id="e1", to_id="new", kind="links_to"),  # duplicate after merge
                Link(from_id="e2", to_id="old", kind="links_to"),
                Link(from_id="e2", to_id="old", kind="routed_to"),
            ]
        )
        moved = store.merge_document("old", "new")
        assert moved == {"links_to": 1, "routed_to": 1}
        assert store.get_block("old") is None
        kinds = sorted((lnk.from_id, lnk.kind) for lnk in store.get_links_to("new"))
        assert kinds == [("e1", "links_to"), ("e2", "links_to"), ("e2", "routed_to")]

    def test_new_document_keeps_its_own_recap(self, store: SQLiteStore):
        self._doc(store, "old")
        self._doc(store, "new")
        store.upsert_recap("old", "stale", "2022-01-01", "h1")
        store.upsert_recap("new", "fresh", "2022-01-02", "h2")
        store.merge_document("old", "new")
        recap = store.get_recap("new")
        assert recap is not None and recap["recap_md"] == "fresh"


class TestClusterDocumentEmbeddings:
    def test_mean_vector_is_keyed_on_the_augi_document(self, tmp_path: Path, store: SQLiteStore):
        from openaugi.pipeline.cluster import load_document_embeddings

        run_layer0(_write_vault(tmp_path / "v"), store)
        vec = np.ones(4, dtype=np.float32).tobytes()
        store.conn.execute("UPDATE blocks SET embedding = ? WHERE kind = 'data_block'", (vec,))
        store.conn.commit()
        assert A_AUGI in load_document_embeddings(store)


class TestBackfillDocumentLinks:
    """Repair a DB written by the old resolver (see the module docstring)."""

    def _stale_db(self, tmp_path: Path, store: SQLiteStore, monkeypatch) -> Path:
        """A gained an augi_id (and lost a section) under the old pipeline:
        a new augi-keyed document, the old one keeping B's backlink."""
        from openaugi.pipeline import runner

        v = _write_vault(tmp_path / "v", a_has_id=False)
        run_layer0(v, store)
        store.insert_links([Link(from_id=_b_block_id(store), to_id=A_PATH, kind="routed_to")])
        (v / "Hub A.md").write_text(A_FM + "### 2022-01-01\nFirst thought.\n")
        monkeypatch.setattr(runner, "_fold_rekeyed_documents", lambda *_: None)
        run_layer0(v, store)
        monkeypatch.undo()
        assert sorted(_docs_for(store, "Hub A.md")) == sorted([A_PATH, A_AUGI])
        assert _backlinks(store, A_AUGI) == set()
        return v

    def test_folds_the_stale_document(self, tmp_path: Path, store: SQLiteStore, monkeypatch):
        from openaugi.adapters.vault import backfill_document_links

        v = self._stale_db(tmp_path, store, monkeypatch)
        out = backfill_document_links(store, v)
        (m,) = out["merged"]
        assert (m["old_id"], m["new_id"]) == (A_PATH, A_AUGI)
        assert m["moved"] == {"links_to": 1, "routed_to": 1}
        assert m["orphans_removed"] == 1
        assert _docs_for(store, "Hub A.md") == [A_AUGI]
        assert _backlinks(store, A_AUGI) == {_b_block_id(store)}
        assert len(store.get_entries_for_document(A_AUGI)) == 1

    def test_dry_run_writes_nothing_and_reports_the_same(
        self, tmp_path: Path, store: SQLiteStore, monkeypatch
    ):
        from openaugi.adapters.vault import backfill_document_links

        v = self._stale_db(tmp_path, store, monkeypatch)
        before = store.get_stats()
        dry = backfill_document_links(store, v, dry_run=True)
        assert store.get_stats() == before
        real = backfill_document_links(store, v)
        assert dry == real

    def test_idempotent(self, tmp_path: Path, store: SQLiteStore, monkeypatch):
        from openaugi.adapters.vault import backfill_document_links

        v = self._stale_db(tmp_path, store, monkeypatch)
        backfill_document_links(store, v)
        again = backfill_document_links(store, v)
        assert again["merged"] == [] and again["relinked_total"] == 0

    def test_restores_links_the_old_resolver_dropped(
        self, tmp_path: Path, store: SQLiteStore, monkeypatch
    ):
        from openaugi.adapters import vault
        from openaugi.adapters.vault import backfill_document_links

        v = _write_vault(tmp_path / "v")
        monkeypatch.setattr(
            vault,
            "_resolve_document_id",
            lambda t, idx, _ids: Block.make_document_id(vault._resolve_wikilink(t, idx)),
        )
        run_layer0(v, store)
        monkeypatch.undo()
        assert _backlinks(store, A_AUGI) == set()

        out = backfill_document_links(store, v)
        assert out["relinked"] == {"Hub A": 1}
        assert _backlinks(store, A_AUGI) == {_b_block_id(store)}

    def test_other_missing_links_only_with_all_links(self, tmp_path: Path, store: SQLiteStore):
        """A link written before its target existed is never revisited by ingest."""
        from openaugi.adapters.vault import backfill_document_links

        v = _write_vault(tmp_path / "v")
        (v / "Note B.md").write_text(B_BODY + "Also [[Later]].\n")
        run_layer0(v, store)
        (v / "Later.md").write_text("Arrived.\n")
        run_layer0(v, store)
        later = Block.make_document_id("Later.md")
        assert _backlinks(store, later) == set()

        out = backfill_document_links(store, v)
        assert out["other_missing_links"] == 1 and out["relinked_total"] == 0
        out = backfill_document_links(store, v, all_links=True)
        assert out["relinked"] == {"Later": 1}
        assert _backlinks(store, later) == {_b_block_id(store)}

    @pytest.mark.graph_drift_expected(reason="a dry run leaves the stale DB it reports on")
    def test_cli_dry_run(self, tmp_path: Path, store: SQLiteStore, monkeypatch):
        from typer.testing import CliRunner

        from openaugi.cli.main import app

        v = self._stale_db(tmp_path, store, monkeypatch)
        before = store.get_stats()
        result = CliRunner().invoke(
            app,
            ["backfill-document-links", "--db", str(store.db_path), "--path", str(v), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "Would fold 1 stale document" in result.output
        assert "links_to 1" in result.output
        assert store.get_stats() == before
