"""Tests for the Obsidian vault adapter.

Uses the fixture vault at tests/fixtures/vault/ which contains:
- daily-2024-03-15.md (H3 dates, frontmatter tags, wikilinks, inline tags)
- Project Alpha.md (frontmatter tags, wikilinks, inline tags, no H3 dates)
- Team Meetings.md (H3 dates, wikilinks, inline tags)
- 2024-01-15 Book Notes.md (filename date, wikilinks, inline tags)
- empty-note.md (frontmatter only, stub)
- no-dates-no-h3.md (single entry, no dates)
- subdir/nested-note.md (subdirectory, nested tags)
- .obsidian/config.json (should be excluded)
"""

from pathlib import Path

import pytest

from openaugi.adapters.vault import (
    _extract_links,
    _extract_tags,
    _matches_pattern,
    _split_by_h3_dates,
    _strip_frontmatter,
    parse_vault,
    parse_vault_incremental,
)


class TestRegexExtraction:
    def test_extract_tags(self):
        text = "Working on #career and #ai/research today"
        tags = _extract_tags(text)
        assert "career" in tags
        assert "ai/research" in tags

    def test_extract_tags_deduped(self):
        text = "#career is important. Also #career."
        tags = _extract_tags(text)
        assert tags.count("career") == 1

    def test_extract_links(self):
        text = "See [[Project Alpha]] and [[Team Meetings|meetings]]"
        links = _extract_links(text)
        assert "Project Alpha" in links
        assert "Team Meetings" in links
        assert len(links) == 2

    def test_extract_links_deduped(self):
        text = "[[Alpha]] and [[Alpha]] again"
        links = _extract_links(text)
        assert len(links) == 1


class TestFrontmatter:
    def test_strip_frontmatter_with_list_tags(self):
        content = "---\ntags:\n  - career\n  - project\n---\n\n# Content"
        body, tags = _strip_frontmatter(content)
        assert "# Content" in body
        assert "career" in tags
        assert "project" in tags

    def test_strip_frontmatter_no_frontmatter(self):
        content = "# Just content\nNo frontmatter here."
        body, tags = _strip_frontmatter(content)
        assert body == content
        assert tags == []

    def test_strip_frontmatter_inline_tags(self):
        content = '---\ntags: [career, "project"]\n---\n\nBody'
        body, tags = _strip_frontmatter(content)
        assert "career" in tags
        assert "project" in tags


class TestSplitByH3:
    def test_split_with_h3_dates(self):
        content = "Preamble\n\n### 2024-03-15\nSection 1\n\n### 2024-03-14\nSection 2"
        sections = _split_by_h3_dates(content)
        assert len(sections) == 3  # preamble + 2 dated sections
        assert sections[0][1] is None  # preamble has no date
        assert sections[1][1] == "2024-03-15"
        assert sections[2][1] == "2024-03-14"

    def test_split_no_h3(self):
        content = "# Title\n\nJust a regular note."
        sections = _split_by_h3_dates(content)
        assert len(sections) == 1
        assert sections[0][1] is None

    def test_split_empty_preamble_skipped(self):
        content = "### 2024-03-15\nFirst section"
        sections = _split_by_h3_dates(content)
        assert len(sections) == 1
        assert sections[0][1] == "2024-03-15"


class TestExcludePatterns:
    def test_obsidian_dir_excluded(self):
        assert _matches_pattern(".obsidian/config.json", ".obsidian/**")

    def test_trash_excluded(self):
        assert _matches_pattern(".trash/deleted.md", ".trash/**")

    def test_normal_file_not_excluded(self):
        assert not _matches_pattern("notes/daily.md", ".obsidian/**")

    def test_nested_path_excluded(self):
        assert _matches_pattern(
            ".obsidian/plugins/something.json", ".obsidian/**"
        )


class TestParseVault:
    def test_parse_vault_returns_blocks_and_links(self, vault_path: Path):
        blocks, links = parse_vault(vault_path)

        assert len(blocks) > 0
        assert len(links) > 0

        # Should have document blocks
        doc_blocks = [b for b in blocks if b.kind == "document"]
        assert len(doc_blocks) >= 5  # at least our 6 .md files minus empty

        # Should have entry blocks
        entry_blocks = [b for b in blocks if b.kind == "entry"]
        assert len(entry_blocks) >= 5

        # Should have tag blocks
        tag_blocks = [b for b in blocks if b.kind == "tag"]
        assert len(tag_blocks) >= 3

    def test_obsidian_dir_excluded(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)
        # .obsidian/ files should not produce any blocks
        for b in blocks:
            if b.kind == "document":
                assert ".obsidian" not in b.metadata.get("source_path", "")

    def test_h3_splitting(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        # daily-2024-03-15.md has 2 H3 dates → should produce 2+ entries
        daily_entries = [
            b for b in blocks
            if b.kind == "entry"
            and b.metadata.get("parent_note_title") == "daily-2024-03-15"
        ]
        assert len(daily_entries) >= 2

    def test_frontmatter_tags_extracted(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        # daily-2024-03-15.md has frontmatter tags
        daily_entries = [
            b for b in blocks
            if b.kind == "entry"
            and b.metadata.get("parent_note_title") == "daily-2024-03-15"
        ]
        # At least one entry should have the frontmatter tag
        all_tags = set()
        for e in daily_entries:
            all_tags.update(e.tags)
        assert "note-type/daily-journal" in all_tags

    def test_wikilinks_produce_links_to(self, vault_path: Path):
        _, links = parse_vault(vault_path)

        links_to = [lnk for lnk in links if lnk.kind == "links_to"]
        assert len(links_to) > 0

        # At least one should target Project Alpha
        targets = [lnk.metadata.get("target_title") for lnk in links_to]
        assert "Project Alpha" in targets

    def test_split_from_links(self, vault_path: Path):
        _, links = parse_vault(vault_path)

        split_links = [lnk for lnk in links if lnk.kind == "split_from"]
        assert len(split_links) > 0

    def test_tagged_links(self, vault_path: Path):
        _, links = parse_vault(vault_path)

        tagged_links = [lnk for lnk in links if lnk.kind == "tagged"]
        assert len(tagged_links) > 0

    def test_nested_directory(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        nested = [
            b for b in blocks
            if b.kind == "document"
            and "subdir/" in b.metadata.get("source_path", "")
        ]
        assert len(nested) == 1

    def test_filename_date_resolution(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        # 2024-01-15 Book Notes.md should have timestamp from filename
        book_entries = [
            b for b in blocks
            if b.kind == "entry"
            and b.metadata.get("parent_note_title", "").startswith(
                "2024-01-15"
            )
        ]
        assert len(book_entries) >= 1
        assert book_entries[0].timestamp == "2024-01-15"

    def test_single_entry_for_no_h3(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        simple_entries = [
            b for b in blocks
            if b.kind == "entry"
            and b.metadata.get("parent_note_title") == "no-dates-no-h3"
        ]
        assert len(simple_entries) == 1

    def test_block_ids_deterministic(self, vault_path: Path):
        blocks1, _ = parse_vault(vault_path)
        blocks2, _ = parse_vault(vault_path)

        ids1 = sorted(b.id for b in blocks1 if b.kind == "entry")
        ids2 = sorted(b.id for b in blocks2 if b.kind == "entry")
        assert ids1 == ids2


class TestIncrementalParsing:
    def test_incremental_no_changes(self, vault_path: Path):
        # First run — get all hashes
        blocks1, _, hashes1, _ = parse_vault_incremental(
            vault_path, known_doc_hashes={}
        )
        assert len(blocks1) > 0

        # Second run with same hashes — should return nothing
        blocks2, links2, _, deleted = parse_vault_incremental(
            vault_path, known_doc_hashes=hashes1
        )
        assert len(blocks2) == 0
        assert len(links2) == 0
        assert len(deleted) == 0

    def test_incremental_detects_deleted(self, vault_path: Path):
        fake_hashes = {"nonexistent.md": "abc123"}
        _, _, _, deleted = parse_vault_incremental(
            vault_path, known_doc_hashes=fake_hashes
        )
        assert "nonexistent.md" in deleted


class TestVaultValidation:
    def test_nonexistent_vault_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            parse_vault(tmp_path / "nonexistent")

    def test_nonexistent_vault_incremental_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            parse_vault_incremental(tmp_path / "nonexistent", {})

    def test_unreadable_vault_raises(self, tmp_path: Path):
        """Vault dir exists but can't be listed → PermissionError."""
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        blocked.chmod(0o000)
        try:
            with pytest.raises(PermissionError, match="Cannot read"):
                parse_vault(blocked)
        finally:
            blocked.chmod(0o755)


class TestEdgeCases:
    def test_unicode_content(self, tmp_path: Path):
        """Files with emoji and non-ASCII parse without error."""
        note = tmp_path / "unicode.md"
        note.write_text(
            "---\ntags:\n  - café\n---\n\n# 日本語 🎉\n\nContent with émojis 🚀\n",
            encoding="utf-8",
        )
        blocks, links = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "entry"]
        assert len(entries) == 1
        assert "🚀" in entries[0].content

    def test_frontmatter_only_file(self, vault_path: Path):
        """A file with only frontmatter and no body produces no entries."""
        blocks, _ = parse_vault(vault_path)
        empty_entries = [
            b for b in blocks
            if b.kind == "entry"
            and b.metadata.get("parent_note_title") == "empty-note"
        ]
        assert len(empty_entries) == 0

    def test_special_chars_in_filename(self, tmp_path: Path):
        """Filenames with parentheses and ampersands parse fine."""
        note = tmp_path / "Q&A (draft).md"
        note.write_text("Some content here\n")
        blocks, _ = parse_vault(tmp_path)
        docs = [b for b in blocks if b.kind == "document"]
        assert any("Q&A (draft)" in (b.title or "") for b in docs)

    def test_non_utf8_file_skipped_gracefully(self, tmp_path: Path):
        """A Latin-1 file is skipped with a warning, not a crash."""
        good = tmp_path / "good.md"
        good.write_text("Normal content\n", encoding="utf-8")
        bad = tmp_path / "bad.md"
        bad.write_bytes(b"caf\xe9 r\xe9sum\xe9\n")  # Latin-1
        blocks, _ = parse_vault(tmp_path)
        # good.md produces a doc + entry; bad.md produces a doc but entry may fail
        # The key assertion: no crash, and good.md content is present
        entries = [b for b in blocks if b.kind == "entry"]
        assert any("Normal content" in (b.content or "") for b in entries)
