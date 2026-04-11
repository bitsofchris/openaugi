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
    _code_fence_ranges,
    _extract_links,
    _extract_tags,
    _extract_wk_date,
    _extract_zzz_instructions,
    _has_meaningful_content,
    _matches_pattern,
    _split_by_headings,
    _split_by_qqq,
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


class TestZzzInstructions:
    def test_no_zzz_returns_content_unchanged(self):
        text = "Just a normal thought about something."
        clean, instructions = _extract_zzz_instructions(text)
        assert clean == text
        assert instructions == []

    def test_zzz_colon_form(self):
        text = "Had an idea\nzzz: research this more"
        clean, instructions = _extract_zzz_instructions(text)
        assert clean == "Had an idea"
        assert instructions == ["research this more"]

    def test_zzz_no_colon(self):
        text = "Need to fix onboarding\nzzz task - do this in openaugi repo"
        clean, instructions = _extract_zzz_instructions(text)
        assert clean == "Need to fix onboarding"
        assert instructions == ["task - do this in openaugi repo"]

    def test_multiple_zzz_lines_as_separate_items(self):
        text = "A thought\nzzz research this\nmore content\nzzz also tag personal"
        clean, instructions = _extract_zzz_instructions(text)
        assert "zzz" not in clean
        assert "A thought" in clean
        assert "more content" in clean
        # Each zzz line is its own list item, in document order
        assert instructions == ["research this", "also tag personal"]

    def test_zzz_line_stripped_from_middle(self):
        text = "line one\nzzz just log this\nline three"
        clean, instructions = _extract_zzz_instructions(text)
        assert "zzz" not in clean
        assert "line one" in clean
        assert "line three" in clean
        assert instructions == ["just log this"]

    def test_zzz_case_insensitive_upper(self):
        text = "Thought\nZZZ: do research"
        _, instructions = _extract_zzz_instructions(text)
        assert instructions == ["do research"]

    def test_zzz_case_insensitive_mixed(self):
        """Any case combo on the zzz prefix should match: zZz, Zzz, zZZ, etc."""
        text = "Thought\nzZz: mixed one\nZzZ: mixed two\nzZZ: mixed three"
        _, instructions = _extract_zzz_instructions(text)
        assert instructions == ["mixed one", "mixed two", "mixed three"]

    def test_bare_zzz_line_no_body(self):
        """A line with just `zzz` and nothing after produces no instruction."""
        text = "content\nzzz"
        clean, instructions = _extract_zzz_instructions(text)
        assert clean == "content"
        assert instructions == []

    def test_zzz_not_matched_inside_word(self):
        """Words containing 'zzz' like 'buzzzing' should not match."""
        text = "The buzzzing sound was loud"
        clean, instructions = _extract_zzz_instructions(text)
        assert clean == text
        assert instructions == []


class TestSplitByQqq:
    def test_no_qqq_returns_content_unchanged(self):
        content = "Just a single block with no split markers."
        assert _split_by_qqq(content) == [content]

    def test_single_qqq_splits_into_two(self):
        content = "First thought\nqqq\nSecond thought"
        segments = _split_by_qqq(content)
        assert len(segments) == 2
        assert "First thought" in segments[0]
        assert "Second thought" in segments[1]
        # qqq marker itself is not in either segment
        assert "qqq" not in segments[0]
        assert "qqq" not in segments[1]

    def test_multiple_qqq_splits(self):
        content = "A\nqqq\nB\nqqq\nC"
        segments = _split_by_qqq(content)
        assert len(segments) == 3
        assert "A" in segments[0]
        assert "B" in segments[1]
        assert "C" in segments[2]

    def test_qqq_case_insensitive(self):
        """qqq, QQQ, qQq should all act as splitters."""
        content = "one\nqqq\ntwo\nQQQ\nthree\nqQq\nfour"
        segments = _split_by_qqq(content)
        assert len(segments) == 4
        assert "one" in segments[0]
        assert "two" in segments[1]
        assert "three" in segments[2]
        assert "four" in segments[3]

    def test_qqq_with_surrounding_whitespace_still_matches(self):
        content = "a\n  qqq  \nb"
        segments = _split_by_qqq(content)
        assert len(segments) == 2
        assert "a" in segments[0]
        assert "b" in segments[1]

    def test_qqq_inside_word_not_matched(self):
        """qqq inside a word (like 'qqq!') should not split."""
        content = "Talking about qqq! in the middle of a sentence"
        segments = _split_by_qqq(content)
        assert len(segments) == 1
        assert segments[0] == content

    def test_consecutive_qqq_produces_empty_middle_segment(self):
        """Two qqq lines in a row produce an empty segment between them."""
        content = "one\nqqq\nqqq\ntwo"
        segments = _split_by_qqq(content)
        assert len(segments) == 3
        # Middle segment is empty/whitespace; caller is responsible for dropping it
        assert segments[1].strip() == ""
        assert "one" in segments[0]
        assert "two" in segments[2]

    def test_leading_and_trailing_qqq(self):
        """qqq at the very start or end produces empty edge segments."""
        content = "qqq\nmiddle\nqqq"
        segments = _split_by_qqq(content)
        assert len(segments) == 3
        assert segments[0].strip() == ""
        assert "middle" in segments[1]
        assert segments[2].strip() == ""


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


class TestSplitByHeadings:
    def test_split_with_date_h3s(self):
        content = "Preamble\n\n### 2024-03-15\nSection 1\n\n### 2024-03-14\nSection 2"
        sections = _split_by_headings(content)
        # preamble + 2 date sections
        assert len(sections) == 3
        assert sections[0][1] is None  # preamble: no date
        assert sections[0][2] is None  # preamble: no heading
        assert sections[1][1] == "2024-03-15"
        assert sections[1][2] == "2024-03-15"
        assert sections[2][1] == "2024-03-14"

    def test_split_any_heading(self):
        content = "## Overview\nsome text\n\n## Goals\nmore text"
        sections = _split_by_headings(content)
        assert len(sections) == 2
        assert sections[0][2] == "Overview"
        assert sections[1][2] == "Goals"
        assert sections[0][1] is None  # no date
        assert sections[1][1] is None  # no date

    def test_date_flows_down_to_subheadings(self):
        content = "### 2024-03-15\ncontent\n#### HW\npersonal stuff"
        sections = _split_by_headings(content)
        assert len(sections) == 2
        assert sections[0][1] == "2024-03-15"  # date heading
        assert sections[1][1] == "2024-03-15"  # inherited
        assert sections[1][2] == "HW"

    def test_section_content_excludes_heading_line(self):
        content = "### 2024-03-15\nhello world"
        sections = _split_by_headings(content)
        assert len(sections) == 1
        assert "hello world" in sections[0][0]
        assert "###" not in sections[0][0]

    def test_heading_only_section_has_empty_content(self):
        content = "# Title\n\n### 2024-03-15\nreal content"
        sections = _split_by_headings(content)
        # # Title section has only whitespace content
        assert sections[0][0].strip() == ""
        # Date section has real content
        assert "real content" in sections[1][0]

    def test_no_headings_returns_full_content(self):
        content = "Just a plain note with no headings."
        sections = _split_by_headings(content)
        assert len(sections) == 1
        assert sections[0][0] == content
        assert sections[0][1] is None
        assert sections[0][2] is None

    def test_hash_inside_code_fence_not_treated_as_heading(self):
        """# comments inside ``` fences must not split the block."""
        content = (
            "Typical:\n```\n# Start services\ndocker-compose up -d\n\n"
            "# Stop services\ndocker-compose down\n```"
        )
        sections = _split_by_headings(content)
        # The whole content is one section (no real headings)
        assert len(sections) == 1
        assert "docker-compose up" in sections[0][0]
        assert "docker-compose down" in sections[0][0]

    def test_hash_in_fence_and_real_heading_split_correctly(self):
        """Real headings split normally; # inside fences are ignored."""
        content = (
            "```\n# comment\n```\n\n"
            "### 2024-03-15\n"
            "real content\n"
            "```\n# another comment\ndocker run\n```"
        )
        sections = _split_by_headings(content)
        # Preamble (the code block) + one real heading section
        assert len(sections) == 2
        assert sections[1][1] == "2024-03-15"
        assert "real content" in sections[1][0]
        assert "# another comment" in sections[1][0]

    def test_tilde_fence_hash_ignored(self):
        """~~~ fences are also respected."""
        content = "~~~\n# not a heading\n~~~\n# real heading\nsome text"
        sections = _split_by_headings(content)
        # preamble + one heading section
        assert len(sections) == 2
        assert sections[1][2] == "real heading"

    def test_empty_preamble_before_first_heading_skipped(self):
        content = "### 2024-03-15\nFirst section"
        sections = _split_by_headings(content)
        assert len(sections) == 1
        assert sections[0][1] == "2024-03-15"


class TestCodeFenceRanges:
    def test_no_fences_returns_empty(self):
        assert _code_fence_ranges("just plain text") == []

    def test_single_backtick_fence(self):
        content = "before\n```\ncode\n```\nafter"
        ranges = _code_fence_ranges(content)
        assert len(ranges) == 1
        start, end = ranges[0]
        # The fenced region covers the opening ``` through the closing ```
        assert content[start:end].startswith("```")
        assert content[start:end].endswith("```")
        assert "code" in content[start:end]

    def test_hash_inside_fence_is_within_range(self):
        content = "```\n# comment\n```"
        ranges = _code_fence_ranges(content)
        assert len(ranges) == 1
        start, end = ranges[0]
        # The # comment falls inside the range
        hash_pos = content.index("# comment")
        assert start <= hash_pos < end

    def test_tilde_fence(self):
        content = "~~~\ncode here\n~~~"
        ranges = _code_fence_ranges(content)
        assert len(ranges) == 1

    def test_fence_with_language_specifier(self):
        content = "```bash\n# a comment\n```"
        ranges = _code_fence_ranges(content)
        assert len(ranges) == 1

    def test_multiple_fences(self):
        content = "```\nblock1\n```\nmiddle\n```\nblock2\n```"
        ranges = _code_fence_ranges(content)
        assert len(ranges) == 2

    def test_unclosed_fence_extends_to_eof(self):
        content = "```\nunclosed code"
        ranges = _code_fence_ranges(content)
        assert len(ranges) == 1
        assert ranges[0][1] == len(content)

    def test_heading_outside_fence_not_in_range(self):
        content = "```\n# inside\n```\n# outside"
        ranges = _code_fence_ranges(content)
        outside_pos = content.rindex("# outside")
        assert not any(start <= outside_pos < end for start, end in ranges)


class TestMeaningfulContent:
    def test_real_text_is_meaningful(self):
        assert _has_meaningful_content("Some actual content here")

    def test_empty_string_is_not_meaningful(self):
        assert not _has_meaningful_content("")

    def test_only_horizontal_rule_not_meaningful(self):
        assert not _has_meaningful_content("---")

    def test_multiple_horizontal_rules_not_meaningful(self):
        assert not _has_meaningful_content("---\n---\n---")

    def test_empty_checkbox_not_meaningful(self):
        assert not _has_meaningful_content("- [ ]")

    def test_empty_checkbox_with_trailing_space_not_meaningful(self):
        assert not _has_meaningful_content("- [ ]   ")

    def test_mix_of_rule_and_empty_checkboxes_not_meaningful(self):
        assert not _has_meaningful_content("---\n- [ ]\n---")

    def test_checkbox_with_text_is_meaningful(self):
        assert _has_meaningful_content("- [ ] Buy groceries")

    def test_completed_checkbox_is_meaningful(self):
        assert _has_meaningful_content("- [x] Done task")

    def test_only_blank_lines_not_meaningful(self):
        assert not _has_meaningful_content("   \n\n  \n")

    def test_rule_plus_real_text_is_meaningful(self):
        assert _has_meaningful_content("---\nSome notes below the divider")


class TestExcludePatterns:
    def test_obsidian_dir_excluded(self):
        assert _matches_pattern(".obsidian/config.json", ".obsidian/**")

    def test_trash_excluded(self):
        assert _matches_pattern(".trash/deleted.md", ".trash/**")

    def test_normal_file_not_excluded(self):
        assert not _matches_pattern("notes/daily.md", ".obsidian/**")

    def test_nested_path_excluded(self):
        assert _matches_pattern(".obsidian/plugins/something.json", ".obsidian/**")

    def test_double_glob_matches_dir_anywhere_in_path(self):
        assert _matches_pattern("_private/4-Tech Notes/Docker.md", "**/4-Tech Notes/**")

    def test_double_glob_matches_dir_at_root(self):
        assert _matches_pattern("4-Tech Notes/Docker.md", "**/4-Tech Notes/**")

    def test_double_glob_does_not_match_unrelated_path(self):
        assert not _matches_pattern("notes/daily.md", "**/4-Tech Notes/**")


class TestParseVault:
    def test_parse_vault_returns_blocks_and_links(self, vault_path: Path):
        blocks, links = parse_vault(vault_path)

        assert len(blocks) > 0
        assert len(links) > 0

        # Should have context_block:document blocks
        doc_blocks = [b for b in blocks if b.kind == "context_block:document"]
        assert len(doc_blocks) >= 5  # at least our 6 .md files minus empty

        # Should have data_block blocks
        data_blocks = [b for b in blocks if b.kind == "data_block"]
        assert len(data_blocks) >= 5

        # Should have context_block:tag blocks
        tag_blocks = [b for b in blocks if b.kind == "context_block:tag"]
        assert len(tag_blocks) >= 3

    def test_obsidian_dir_excluded(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)
        # .obsidian/ files should not produce any blocks
        for b in blocks:
            if b.kind == "context_block:document":
                assert ".obsidian" not in b.metadata.get("source_path", "")

    def test_h3_splitting(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        # daily-2024-03-15.md has 2 H3 dates → should produce 2+ data_blocks
        daily_entries = [
            b
            for b in blocks
            if b.kind == "data_block" and b.metadata.get("parent_note_title") == "daily-2024-03-15"
        ]
        assert len(daily_entries) >= 2

    def test_frontmatter_tags_extracted(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        # daily-2024-03-15.md has frontmatter tags
        daily_entries = [
            b
            for b in blocks
            if b.kind == "data_block" and b.metadata.get("parent_note_title") == "daily-2024-03-15"
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

    def test_contains_links(self, vault_path: Path):
        _, links = parse_vault(vault_path)

        contains_links = [lnk for lnk in links if lnk.kind == "contains"]
        assert len(contains_links) > 0

    def test_groups_links(self, vault_path: Path):
        _, links = parse_vault(vault_path)

        groups_links = [lnk for lnk in links if lnk.kind == "groups"]
        assert len(groups_links) > 0

    def test_nested_directory(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        nested = [
            b
            for b in blocks
            if b.kind == "context_block:document"
            and "subdir/" in b.metadata.get("source_path", "")
        ]
        assert len(nested) == 1

    def test_filename_date_resolution(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        # 2024-01-15 Book Notes.md should have timestamp from filename
        book_entries = [
            b
            for b in blocks
            if b.kind == "data_block"
            and b.metadata.get("parent_note_title", "").startswith("2024-01-15")
        ]
        assert len(book_entries) >= 1
        assert book_entries[0].block_time == "2024-01-15"

    def test_single_entry_for_no_h3(self, vault_path: Path):
        blocks, _ = parse_vault(vault_path)

        simple_entries = [
            b
            for b in blocks
            if b.kind == "data_block" and b.metadata.get("parent_note_title") == "no-dates-no-h3"
        ]
        assert len(simple_entries) == 1

    def test_block_ids_deterministic(self, vault_path: Path):
        blocks1, _ = parse_vault(vault_path)
        blocks2, _ = parse_vault(vault_path)

        ids1 = sorted(b.id for b in blocks1 if b.kind == "data_block")
        ids2 = sorted(b.id for b in blocks2 if b.kind == "data_block")
        assert ids1 == ids2


class TestIncrementalParsing:
    def test_incremental_no_changes(self, vault_path: Path):
        # First run — get all hashes
        blocks1, _, hashes1, _ = parse_vault_incremental(vault_path, known_doc_hashes={})
        assert len(blocks1) > 0

        # Second run with same hashes — should return nothing
        blocks2, links2, _, deleted = parse_vault_incremental(vault_path, known_doc_hashes=hashes1)
        assert len(blocks2) == 0
        assert len(links2) == 0
        assert len(deleted) == 0

    def test_incremental_detects_deleted(self, vault_path: Path):
        fake_hashes = {"nonexistent.md": "abc123"}
        _, _, _, deleted = parse_vault_incremental(vault_path, known_doc_hashes=fake_hashes)
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


class TestWeeklyNotes:
    def test_extract_wk_date_standard(self, tmp_path: Path):
        f = tmp_path / "WK - 25-11-09.md"
        assert _extract_wk_date(f) == "2025-11-09"

    def test_extract_wk_date_different_format(self, tmp_path: Path):
        f = tmp_path / "WK - 24-03-01.md"
        assert _extract_wk_date(f) == "2024-03-01"

    def test_extract_wk_date_no_match(self, tmp_path: Path):
        f = tmp_path / "regular-note.md"
        assert _extract_wk_date(f) is None

    def test_extract_wk_date_ignores_four_digit_year(self, tmp_path: Path):
        """Files already matched by FILENAME_DATE_PATTERN are not re-matched."""
        f = tmp_path / "2024-03-15 daily.md"
        assert _extract_wk_date(f) is None

    def test_wk_note_produces_single_block(self, tmp_path: Path):
        """WK notes are not split by headings — the whole body is one block."""
        note = tmp_path / "WK - 25-11-09.md"
        note.write_text(
            "## How did this week go?\nIt was productive.\n\n"
            "## What would I do differently?\nStart earlier.\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        content = entries[0].content or ""
        assert "productive" in content
        assert "Start earlier" in content

    def test_wk_note_block_time_from_filename(self, tmp_path: Path):
        """The block timestamp comes from the WK date in the filename."""
        note = tmp_path / "WK - 25-11-09.md"
        note.write_text("Weekly reflection content.\n", encoding="utf-8")
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        assert entries[0].block_time == "2025-11-09"

    def test_wk_note_gets_document_granularity(self, tmp_path: Path):
        """Single-block WK notes are tagged granularity=document."""
        note = tmp_path / "WK - 25-11-09.md"
        note.write_text(
            "## Reflection\nGood week.\n\n## Goals\nKeep going.\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        assert entries[0].metadata.get("granularity") == "document"


class TestEdgeCases:
    def test_unicode_content(self, tmp_path: Path):
        """Files with emoji and non-ASCII parse without error."""
        note = tmp_path / "unicode.md"
        note.write_text(
            "---\ntags:\n  - café\n---\n\n# 日本語 🎉\n\nContent with émojis 🚀\n",
            encoding="utf-8",
        )
        blocks, links = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        assert "🚀" in entries[0].content

    def test_frontmatter_only_file(self, vault_path: Path):
        """A file with only frontmatter and no body produces no entries."""
        blocks, _ = parse_vault(vault_path)
        empty_entries = [
            b
            for b in blocks
            if b.kind == "data_block" and b.metadata.get("parent_note_title") == "empty-note"
        ]
        assert len(empty_entries) == 0

    def test_special_chars_in_filename(self, tmp_path: Path):
        """Filenames with parentheses and ampersands parse fine."""
        note = tmp_path / "Q&A (draft).md"
        note.write_text("Some content here\n")
        blocks, _ = parse_vault(tmp_path)
        docs = [b for b in blocks if b.kind == "context_block:document"]
        assert any("Q&A (draft)" in (b.title or "") for b in docs)

    def test_zzz_instructions_captured_in_metadata(self, tmp_path: Path):
        """Full-parse flow: zzz line is stripped from content and stored as a list in metadata."""
        note = tmp_path / "journal.md"
        note.write_text(
            "Had a thought about matryoshka embeddings\n"
            "zzz research this - find papers in my vault\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        entry = entries[0]
        assert "zzz" not in (entry.content or "")
        assert "matryoshka" in (entry.content or "")
        assert entry.metadata.get("zzz_instructions") == [
            "research this - find papers in my vault"
        ]

    def test_multiple_zzz_instructions_in_one_block(self, tmp_path: Path):
        """Each zzz line in a block becomes its own metadata entry."""
        note = tmp_path / "journal.md"
        note.write_text(
            "Thinking about workstream routing\n"
            "zzz: research this\n"
            "zzz: also tag as openaugi/design\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        assert entries[0].metadata.get("zzz_instructions") == [
            "research this",
            "also tag as openaugi/design",
        ]

    def test_zzz_only_block_skipped(self, tmp_path: Path):
        """A section containing only a zzz line produces no entry."""
        note = tmp_path / "scratch.md"
        note.write_text("zzz just a note to self\n", encoding="utf-8")
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 0

    def test_zzz_changes_block_identity(self, tmp_path: Path):
        """Adding a zzz to a block changes its content_hash so it shows up as new."""
        note = tmp_path / "note.md"
        note.write_text("An idea about X\n", encoding="utf-8")
        blocks_before, _ = parse_vault(tmp_path)
        entry_before = next(b for b in blocks_before if b.kind == "data_block")

        note.write_text("An idea about X\nzzz research this\n", encoding="utf-8")
        blocks_after, _ = parse_vault(tmp_path)
        entry_after = next(b for b in blocks_after if b.kind == "data_block")

        assert entry_before.id != entry_after.id
        assert entry_after.metadata.get("zzz_instructions") == ["research this"]
        assert entry_after.content == "An idea about X"

    def test_qqq_splits_a_note_without_h3(self, tmp_path: Path):
        """A note with qqq markers and no H3 dates produces one block per qqq segment."""
        note = tmp_path / "scratch.md"
        note.write_text(
            "First thought\nqqq\nSecond thought\nqqq\nThird thought\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 3
        contents = sorted((e.content or "").strip() for e in entries)
        assert contents == ["First thought", "Second thought", "Third thought"]

    def test_qqq_subsplits_an_h3_section(self, tmp_path: Path):
        """qqq markers inside an H3 date section split that section further."""
        note = tmp_path / "daily-2026-04-08.md"
        note.write_text(
            "### 2026-04-08\nFirst thought of the day\nqqq\nSecond thought of the day\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [
            b
            for b in blocks
            if b.kind == "data_block" and b.metadata.get("parent_note_title") == "daily-2026-04-08"
        ]
        assert len(entries) == 2
        # Both sub-blocks should inherit the date from the heading
        for entry in entries:
            assert entry.metadata.get("section_date") == "2026-04-08"

    def test_qqq_and_zzz_combined(self, tmp_path: Path):
        """A note mixing qqq splits and per-block zzz instructions."""
        note = tmp_path / "journal.md"
        note.write_text(
            "First thought about X\n"
            "zzz: research this\n"
            "qqq\n"
            "Second thought about Y\n"
            "zzz: task — fix tomorrow\n"
            "zzz: also log this\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = sorted(
            (b for b in blocks if b.kind == "data_block"),
            key=lambda b: b.content or "",
        )
        assert len(entries) == 2
        first, second = entries
        assert (first.content or "").strip() == "First thought about X"
        assert first.metadata.get("zzz_instructions") == ["research this"]
        assert (second.content or "").strip() == "Second thought about Y"
        assert second.metadata.get("zzz_instructions") == [
            "task — fix tomorrow",
            "also log this",
        ]

    def test_qqq_case_insensitive_in_parse_vault(self, tmp_path: Path):
        """QQQ / qQq variants are all recognized as splitters."""
        note = tmp_path / "mixed.md"
        note.write_text("one\nQQQ\ntwo\nqQq\nthree\n", encoding="utf-8")
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 3

    def test_note_with_no_delimiters_is_single_block(self, tmp_path: Path):
        """A note with no H3 and no qqq produces one block for the whole content."""
        note = tmp_path / "focused.md"
        note.write_text(
            "This is a focused note about one topic.\n"
            "It has several lines.\n"
            "But no headers and no qqq.\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        content = entries[0].content or ""
        assert "focused note" in content
        assert "several lines" in content
        assert "no qqq" in content

    def test_single_block_note_gets_document_granularity(self, tmp_path: Path):
        """A note with no headings and no qqq produces one block tagged granularity=document."""
        note = tmp_path / "atomic.md"
        note.write_text("A focused thought with no structure.\n", encoding="utf-8")
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        assert entries[0].metadata.get("granularity") == "document"

    def test_h3_split_note_gets_section_granularity(self, tmp_path: Path):
        """A note split by H3 date headers produces blocks tagged granularity=section."""
        note = tmp_path / "journal.md"
        note.write_text(
            "### 2024-01-01\nFirst entry.\n\n### 2024-01-02\nSecond entry.\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 2
        for entry in entries:
            assert entry.metadata.get("granularity") == "section"

    def test_qqq_split_note_gets_section_granularity(self, tmp_path: Path):
        """A note split by qqq markers produces blocks tagged granularity=section."""
        note = tmp_path / "ideas.md"
        note.write_text("First idea.\nqqq\nSecond idea.\nqqq\nThird idea.\n", encoding="utf-8")
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 3
        for entry in entries:
            assert entry.metadata.get("granularity") == "section"

    def test_excalidraw_files_excluded(self, tmp_path: Path):
        """Files ending in .excalidraw.md are skipped entirely."""
        drawing = tmp_path / "diagram.excalidraw.md"
        drawing.write_text('{"type":"excalidraw","version":2}\n', encoding="utf-8")
        real = tmp_path / "real-note.md"
        real.write_text("Actual content\n", encoding="utf-8")
        blocks, _ = parse_vault(tmp_path)
        source_paths = [b.metadata.get("source_path", "") for b in blocks]
        assert not any("excalidraw" in p for p in source_paths)
        assert any("real-note" in p for p in source_paths)

    def test_empty_checkbox_section_produces_no_block(self, tmp_path: Path):
        """A heading section whose only content is an empty checkbox is skipped."""
        note = tmp_path / "template.md"
        note.write_text("### Next\n- [ ]\n\n### Notes\nActual content here\n", encoding="utf-8")
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        assert len(entries) == 1
        assert "Actual content" in (entries[0].content or "")

    def test_horizontal_rule_only_section_produces_no_block(self, tmp_path: Path):
        """A heading section whose only content is a horizontal rule is skipped."""
        note = tmp_path / "project.md"
        note.write_text(
            "Scope: build the thing\n\n# Resources\n---\n\n# Notes\nReal notes here\n",
            encoding="utf-8",
        )
        blocks, _ = parse_vault(tmp_path)
        entries = [b for b in blocks if b.kind == "data_block"]
        contents = [e.content or "" for e in entries]
        assert any("Real notes" in c for c in contents)
        assert not any(c.strip() == "---" for c in contents)

    def test_non_utf8_file_skipped_gracefully(self, tmp_path: Path):
        """A Latin-1 file is skipped with a warning, not a crash."""
        good = tmp_path / "good.md"
        good.write_text("Normal content\n", encoding="utf-8")
        bad = tmp_path / "bad.md"
        bad.write_bytes(b"caf\xe9 r\xe9sum\xe9\n")  # Latin-1
        blocks, _ = parse_vault(tmp_path)
        # good.md produces a doc + entry; bad.md produces a doc but entry may fail
        # The key assertion: no crash, and good.md content is present
        entries = [b for b in blocks if b.kind == "data_block"]
        assert any("Normal content" in (b.content or "") for b in entries)
