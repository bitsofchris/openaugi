"""Tests for block-level incremental ingestion.

Uses a mutable temp vault (copied from fixtures) to test:
- First ingest populates store correctly
- No-change re-ingest is a noop
- Adding a section to a file → only new entry inserted, old entries kept
- Removing a section → entry deleted, others kept
- Modifying a section → old entry removed, new entry inserted, unchanged entries kept
- Embeddings on unchanged entries survive re-ingest
- Deleted file → all its blocks removed via CASCADE
"""

import shutil
from pathlib import Path

from openaugi.model.block import Block
from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore


def _make_temp_vault(tmp_path: Path, vault_path: Path) -> Path:
    """Copy the fixture vault to a temp dir so we can mutate it."""
    dest = tmp_path / "vault"
    shutil.copytree(vault_path, dest)
    return dest


class TestBlockLevelIncremental:
    """Test the two-level incremental ingestion strategy."""

    def test_first_ingest(self, vault_path: Path, store: SQLiteStore):
        """First run should populate the store."""
        result = run_layer0(vault_path, store)
        assert result["stats"]["total_blocks"] > 0
        assert result["stats"]["total_links"] > 0

    def test_noop_second_run(self, vault_path: Path, store: SQLiteStore):
        """Second run with no changes should insert nothing."""
        run_layer0(vault_path, store)
        stats_after_first = store.get_stats()

        result = run_layer0(vault_path, store)
        stats_after_second = store.get_stats()

        assert stats_after_second["total_blocks"] == stats_after_first["total_blocks"]
        assert result["blocks_kept"] == 0  # no changed files → no diffing
        assert result["blocks_added"] == 0
        assert result["blocks_removed"] == 0

    def test_add_section_keeps_existing(
        self, tmp_path: Path, vault_path: Path, store: SQLiteStore
    ):
        """Adding a new H3 section to a file should insert only the new entry,
        keeping existing entries (and their embeddings) intact."""
        temp_vault = _make_temp_vault(tmp_path, vault_path)

        # First ingest
        run_layer0(temp_vault, store)

        # Get entry IDs for Team Meetings before modification
        team_doc_id = Block.make_document_id("Team Meetings.md")
        entries_before = store.get_entries_for_document(team_doc_id)
        ids_before = {e.id for e in entries_before}

        # Simulate embedding on existing entries
        fake_embedding = b"\x42" * 16
        for entry in entries_before:
            store.update_embedding(entry.id, fake_embedding)
        store.conn.commit()

        # Add a new section to Team Meetings.md
        meetings_file = temp_vault / "Team Meetings.md"
        original = meetings_file.read_text()
        meetings_file.write_text(
            original + "\n\n### 2024-03-17\n\nNew standup about #deployment.\n"
        )

        # Re-ingest
        result = run_layer0(temp_vault, store)

        # Old entries should be kept (their IDs still exist)
        entries_after = store.get_entries_for_document(team_doc_id)
        ids_after = {e.id for e in entries_after}

        assert ids_before.issubset(ids_after), "Existing entries should survive"
        assert len(ids_after) == len(ids_before) + 1, "Should have one new entry"

        # Embeddings on old entries should be preserved
        for entry in entries_after:
            if entry.id in ids_before:
                assert entry.embedding == fake_embedding, (
                    f"Embedding lost on kept entry {entry.id}"
                )

        # The result should report the diff
        assert result["blocks_kept"] >= len(ids_before)
        assert result["blocks_added"] >= 1

    def test_remove_section_deletes_entry(
        self, tmp_path: Path, vault_path: Path, store: SQLiteStore
    ):
        """Removing an H3 section should delete that entry, keep others."""
        temp_vault = _make_temp_vault(tmp_path, vault_path)

        # First ingest
        run_layer0(temp_vault, store)

        team_doc_id = Block.make_document_id("Team Meetings.md")
        entries_before = store.get_entries_for_document(team_doc_id)
        count_before = len(entries_before)
        assert count_before >= 3, "Team Meetings should have 3+ entries"

        # Remove the last H3 section (2024-02-25)
        meetings_file = temp_vault / "Team Meetings.md"
        content = meetings_file.read_text()
        # Cut off everything from "### 2024-02-25" onward
        cut_point = content.index("### 2024-02-25")
        meetings_file.write_text(content[:cut_point])

        # Re-ingest
        result = run_layer0(temp_vault, store)

        entries_after = store.get_entries_for_document(team_doc_id)
        assert len(entries_after) == count_before - 1

        # Removed entry should be gone
        hashes_after = {e.content_hash for e in entries_after}
        for old_entry in entries_before:
            if "2024-02-25" in (old_entry.metadata.get("h3_date") or ""):
                assert old_entry.content_hash not in hashes_after

        assert result["blocks_removed"] >= 1

    def test_modify_section_replaces_entry(
        self, tmp_path: Path, vault_path: Path, store: SQLiteStore
    ):
        """Modifying a section's content should remove the old entry
        and insert a new one, while keeping unchanged sections."""
        temp_vault = _make_temp_vault(tmp_path, vault_path)

        # First ingest
        run_layer0(temp_vault, store)

        team_doc_id = Block.make_document_id("Team Meetings.md")
        entries_before = store.get_entries_for_document(team_doc_id)
        hashes_before = {e.content_hash for e in entries_before}

        # Add embedding to all entries
        fake_embedding = b"\x42" * 16
        for entry in entries_before:
            store.update_embedding(entry.id, fake_embedding)
        store.conn.commit()

        # Modify one section — change "Sprint planning" to "Sprint retrospective"
        meetings_file = temp_vault / "Team Meetings.md"
        content = meetings_file.read_text()
        old_text = "Sprint planning for next two weeks."
        new_text = "Sprint retrospective for last two weeks."
        content = content.replace(old_text, new_text)
        meetings_file.write_text(content)

        # Re-ingest
        run_layer0(temp_vault, store)

        entries_after = store.get_entries_for_document(team_doc_id)
        hashes_after = {e.content_hash for e in entries_after}

        # Same count (one removed, one added)
        assert len(entries_after) == len(entries_before)

        # The modified entry should have a different hash
        assert hashes_before != hashes_after

        # Unchanged entries should still have embeddings
        unchanged_entries = [
            e for e in entries_after if e.content_hash in hashes_before
        ]
        for entry in unchanged_entries:
            assert entry.embedding == fake_embedding, (
                f"Embedding lost on unchanged entry {entry.id}"
            )

        # The new entry should NOT have an embedding (needs re-embed)
        new_entries = [
            e for e in entries_after if e.content_hash not in hashes_before
        ]
        assert len(new_entries) == 1
        assert new_entries[0].embedding is None

    def test_delete_file_cascades(
        self, tmp_path: Path, vault_path: Path, store: SQLiteStore
    ):
        """Deleting a file should remove its document, entries, and links."""
        temp_vault = _make_temp_vault(tmp_path, vault_path)

        # First ingest
        run_layer0(temp_vault, store)
        stats_before = store.get_stats()

        # Delete a file
        (temp_vault / "no-dates-no-h3.md").unlink()

        # Re-ingest
        result = run_layer0(temp_vault, store)
        stats_after = store.get_stats()

        assert result["files_deleted"] >= 1
        assert stats_after["total_blocks"] < stats_before["total_blocks"]

        # The document block should be gone
        doc_id = Block.make_document_id("no-dates-no-h3.md")
        assert store.get_block(doc_id) is None

    def test_new_file_added(
        self, tmp_path: Path, vault_path: Path, store: SQLiteStore
    ):
        """Adding a new file should create new document + entry blocks."""
        temp_vault = _make_temp_vault(tmp_path, vault_path)

        # First ingest
        run_layer0(temp_vault, store)
        stats_before = store.get_stats()

        # Add a new file
        new_file = temp_vault / "brand-new-note.md"
        new_file.write_text("# Brand New\n\nThis is new content about #testing.\n")

        # Re-ingest
        run_layer0(temp_vault, store)
        stats_after = store.get_stats()

        assert stats_after["total_blocks"] > stats_before["total_blocks"]

        doc_id = Block.make_document_id("brand-new-note.md")
        doc = store.get_block(doc_id)
        assert doc is not None
        assert doc.kind == "document"

        entries = store.get_entries_for_document(doc_id)
        assert len(entries) == 1
        assert "Brand New" in (entries[0].content or "")
