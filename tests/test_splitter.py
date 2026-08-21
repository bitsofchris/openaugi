"""Tests for the shared splitter — the deterministic split() primitive.

Covers the public API (`split_text`, `split_file`, `Segment`, `SplitResult`).
Regex- and helper-level coverage lives in tests/test_vault_adapter.py since
those helpers are re-exported from the vault module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openaugi.adapters.splitter import (
    Segment,
    SplitResult,
    _extract_augi_id,
    split_file,
    split_text,
)


class TestSplitText:
    def test_empty_input(self):
        assert split_text("") == []

    def test_no_headings_single_segment(self):
        segs = split_text("just a plain thought\nwith two lines")
        assert len(segs) == 1
        assert segs[0].granularity == "document"
        assert segs[0].clean_content == "just a plain thought\nwith two lines"
        assert segs[0].section_heading is None
        assert segs[0].section_date is None

    def test_multiple_headings_become_section_segments(self):
        text = "# H1\nalpha content\n## H2\nbeta content\n## H3\ngamma content"
        segs = split_text(text)
        assert len(segs) == 3
        assert [s.section_heading for s in segs] == ["H1", "H2", "H3"]
        assert all(s.granularity == "section" for s in segs)

    def test_qqq_splits_inside_section(self):
        segs = split_text("# Day\nfirst thought\nqqq\nsecond thought\nqqq\nthird thought")
        assert len(segs) == 3
        assert [s.clean_content for s in segs] == [
            "first thought",
            "second thought",
            "third thought",
        ]

    def test_zzz_extracted_and_stripped(self):
        segs = split_text("# H\na real thought\nzzz research this later")
        assert len(segs) == 1
        assert segs[0].clean_content == "a real thought"
        assert segs[0].zzz_instructions == ["research this later"]
        # raw_hash is computed on the pre-strip content
        assert segs[0].raw_hash == segs[0].raw_hash  # stable
        # clean_content differs from raw content
        assert "zzz" not in segs[0].clean_content

    def test_zzz_behind_daily_note_timestamp_prefix(self):
        """The mobile daily-note writer inlined `HH:MM — ` in front of a
        block's first line (pre-2026-07-15 notes), hiding a leading zzz from
        dispatch. The pattern tolerates the prefix so historical notes still
        dispatch."""
        segs = split_text("# 2026-07-14\n21:42 — zzz: recap my two feature prompts")
        assert len(segs) == 1
        assert segs[0].zzz_instructions == ["recap my two feature prompts"]
        assert "zzz" not in segs[0].clean_content

    def test_plain_timestamp_line_is_not_a_zzz(self):
        segs = split_text("# 2026-07-14\n21:42 — a plain thought, no directive")
        assert segs[0].zzz_instructions == []
        assert "21:42 — a plain thought" in segs[0].clean_content

    def test_date_flows_down_sections(self):
        text = "# 2026-04-08 Monday\nmorning\n## Evening\nsomething\n# 2026-04-09 Tuesday\nnext"
        segs = split_text(text)
        dates = [s.section_date for s in segs]
        assert dates == ["2026-04-08", "2026-04-08", "2026-04-09"]

    def test_tags_and_links_extracted(self):
        segs = split_text("# H\nTalked with [[Sam]] about #career and #focus.")
        assert segs[0].tags == ["career", "focus"]
        assert segs[0].links == ["Sam"]

    def test_structural_only_section_dropped(self):
        segs = split_text("# Real\nhello\n## Empty\n- [ ]\n---\n## Real2\nworld")
        headings = [s.section_heading for s in segs]
        assert "Empty" not in headings
        assert headings == ["Real", "Real2"]

    def test_zzz_only_block_kept_as_directive(self):
        """A sub-section containing only a zzz directive is still meaningful —
        the directive itself IS the block, dispatched as an agent task."""
        segs = split_text("# H\nreal\nqqq\nzzz just a directive\nqqq\nalso real")
        # All three subs survive: "real", zzz-only (empty clean), "also real"
        cleans = [s.clean_content for s in segs]
        assert "real" in cleans
        assert "also real" in cleans
        # zzz-only segment has empty clean_content but carries the directive
        zzz_only = [s for s in segs if not s.clean_content]
        assert len(zzz_only) == 1
        assert zzz_only[0].zzz_instructions == ["just a directive"]

    def test_frontmatter_stripped_in_split_text(self):
        segs = split_text("---\ntags: [a, b]\n---\n# H\nhello")
        assert len(segs) == 1
        # tags from frontmatter are not returned by split_text (use split_file)
        assert segs[0].tags == []

    def test_returns_segment_objects(self):
        segs = split_text("hello")
        assert isinstance(segs[0], Segment)
        dumped = segs[0].model_dump()
        for key in ("content", "clean_content", "zzz_instructions", "raw_hash", "granularity"):
            assert key in dumped

    def test_code_fence_hashes_are_not_headings(self):
        text = "# Real\nhere is code\n```python\n# not a heading\n```\nmore text"
        segs = split_text(text)
        assert len(segs) == 1
        assert segs[0].section_heading == "Real"


class TestAnchorSplitting:
    """A line that is only an Obsidian block anchor (`^id`) closes the current
    segment — the anchor names the content above it. Format-native: applies to
    any note, not just mobile capture daily notes."""

    def test_anchored_entries_become_per_entry_segments(self):
        text = (
            "# 2026-07-08\n\n"
            "09:15 — first thought\n^augi-aaaa1111\n\n"
            "10:30 — second thought\n^augi-bbbb2222\n"
        )
        segs = split_text(text)
        assert len(segs) == 2
        assert [s.anchor_id for s in segs] == ["augi-aaaa1111", "augi-bbbb2222"]
        assert [s.clean_content for s in segs] == [
            "09:15 — first thought",
            "10:30 — second thought",
        ]
        assert all(s.section_heading == "2026-07-08" for s in segs)

    def test_anchor_kept_in_raw_content_stripped_from_clean(self):
        segs = split_text("a hand-anchored paragraph\n^my-ref")
        assert segs[0].anchor_id == "my-ref"
        assert segs[0].content.endswith("^my-ref")  # hash identity covers the anchor
        assert "^my-ref" not in segs[0].clean_content

    def test_multi_paragraph_entry_stays_one_segment(self):
        text = "first paragraph of the entry\n\nsecond paragraph, same entry\n^augi-cccc3333\n"
        segs = split_text(text)
        assert len(segs) == 1
        assert segs[0].anchor_id == "augi-cccc3333"
        assert "second paragraph" in segs[0].clean_content

    def test_unanchored_trailing_text_is_own_segment(self):
        segs = split_text("an anchored entry\n^augi-dddd4444\n\na human annotation added later\n")
        assert [s.anchor_id for s in segs] == ["augi-dddd4444", None]
        assert segs[1].clean_content == "a human annotation added later"

    def test_inline_caret_does_not_split(self):
        """Only a line that is SOLELY an anchor closes a segment — inline
        `^ref` text and block references like [[note#^ref]] don't."""
        segs = split_text("see the earlier point ^not-alone here\nand [[note#^some-ref]] too")
        assert len(segs) == 1
        assert segs[0].anchor_id is None

    def test_qqq_and_anchors_mixed(self):
        text = "alpha\n^a1\nbeta\nqqq\ngamma\n^a2\n"
        segs = split_text(text)
        assert [(s.clean_content, s.anchor_id) for s in segs] == [
            ("alpha", "a1"),
            ("beta", None),
            ("gamma", "a2"),
        ]

    def test_anchor_only_piece_dropped(self):
        """A dangling anchor with no content above it names nothing — dropped."""
        segs = split_text("real entry\n^a1\n^a2\n")
        assert len(segs) == 1
        assert segs[0].anchor_id == "a1"

    def test_zzz_attaches_to_its_own_entry(self):
        text = (
            "# 2026-07-08\n\n"
            "09:15 — plain thought\n^augi-aaaa1111\n\n"
            "14:20 — try a lens\nzzz: apply lens distill\n^augi-bbbb2222\n"
        )
        segs = split_text(text)
        assert segs[0].zzz_instructions == []
        assert segs[1].zzz_instructions == ["apply lens distill"]
        assert "zzz" not in segs[1].clean_content

    def test_entry_time_parsed_from_anchored_lead(self):
        segs = split_text("9:15 — early thought\n^a1\n\n16:45 —\nzzz: recap today\n^a2\n")
        assert [s.entry_time for s in segs] == ["09:15", "16:45"]

    def test_entry_time_ignored_when_invalid_or_unanchored(self):
        segs = split_text("29:99 — not a clock\n^a1\n\n10:30 — unanchored trailing text\n")
        assert [s.entry_time for s in segs] == [None, None]

    def test_single_anchored_note_is_document_granularity(self):
        segs = split_text("the whole note is one anchored block\n^solo")
        assert len(segs) == 1
        assert segs[0].granularity == "document"
        assert segs[0].anchor_id == "solo"


class TestSplitFile:
    def test_split_file_returns_result_with_metadata(self, tmp_path: Path):
        p = tmp_path / "2026-04-08-journal.md"
        p.write_text("---\ntags: [personal, journal]\n---\n# Morning\nsomething\n")
        result = split_file(p)
        assert isinstance(result, SplitResult)
        assert result.filename_date == "2026-04-08"
        assert result.frontmatter_tags == ["personal", "journal"]
        assert len(result.segments) == 1
        assert result.segments[0].section_date == "2026-04-08"  # filename fallback applied

    def test_wk_filename_keeps_file_as_single_segment(self, tmp_path: Path):
        p = tmp_path / "WK - 25-11-09.md"
        p.write_text("# Weekly\n## Q1\nanswer one\n## Q2\nanswer two\n")
        result = split_file(p)
        assert result.filename_date == "2025-11-09"
        # WK notes are kept as one segment (no heading split) so the
        # question/answer pairs aren't fragmented.
        assert len(result.segments) == 1
        assert result.segments[0].section_date == "2025-11-09"

    def test_doc_hash_stable_and_changes_with_content(self, tmp_path: Path):
        p = tmp_path / "x.md"
        p.write_text("hello")
        h1 = split_file(p).doc_hash
        p.write_text("hello world")
        h2 = split_file(p).doc_hash
        assert h1 != h2
        p.write_text("hello")
        assert split_file(p).doc_hash == h1

    def test_heading_date_overrides_filename_date(self, tmp_path: Path):
        p = tmp_path / "2026-04-08-journal.md"
        p.write_text("# 2026-04-01 Backdated\nold thought\n# Unrelated\nnew\n")
        segs = split_file(p).segments
        assert segs[0].section_date == "2026-04-01"
        # section with no date of its own falls back to filename date? No —
        # the previous date flows down until the next date-headed section
        assert segs[1].section_date == "2026-04-01"


class TestCLISplitCommand:
    def test_cli_emits_json(self, tmp_path: Path):
        from typer.testing import CliRunner

        from openaugi.cli.main import app

        p = tmp_path / "note.md"
        p.write_text("# H\nhello\n## H2\nworld")
        result = CliRunner().invoke(app, ["split", str(p), "--format", "json"])
        assert result.exit_code == 0, result.output
        import json

        payload = json.loads(result.output)
        assert payload["source_path"] == str(p)
        assert len(payload["segments"]) == 2

    def test_cli_ndjson_streams_one_per_line(self, tmp_path: Path):
        from typer.testing import CliRunner

        from openaugi.cli.main import app

        p = tmp_path / "note.md"
        p.write_text("# A\nalpha\n# B\nbeta")
        result = CliRunner().invoke(app, ["split", str(p), "--format", "ndjson"])
        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 2
        import json

        for line in lines:
            assert "clean_content" in json.loads(line)

    def test_cli_rejects_unknown_format(self, tmp_path: Path):
        from typer.testing import CliRunner

        from openaugi.cli.main import app

        p = tmp_path / "n.md"
        p.write_text("hi")
        result = CliRunner().invoke(app, ["split", str(p), "--format", "xml"])
        assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestOpenTaskFlag:
    """`has_open_task` — deterministic ingest-time fact feeding the Dashboard's
    14-day task shelf (rendered query, no agent judgment)."""

    def test_open_checkbox_with_text_sets_flag(self):
        segs = split_text("a thought\n- [ ] call the plumber\n^augi-aaaa1111")
        assert len(segs) == 1
        assert segs[0].has_open_task is True

    def test_completed_checkbox_does_not_count(self):
        segs = split_text("a thought\n- [x] already shipped")
        assert segs[0].has_open_task is False

    def test_plain_prose_has_no_flag(self):
        segs = split_text("just thinking out loud, no task here")
        assert segs[0].has_open_task is False

    def test_bare_empty_checkbox_is_still_structural_noise(self):
        # `- [ ]` with no text stays meaningless — no segment survives.
        segs = split_text("# H\n- [ ]")
        assert segs == []

    def test_star_and_plus_bullets_count(self):
        assert split_text("x\n* [ ] star task")[0].has_open_task is True
        assert split_text("x\n+ [ ] plus task")[0].has_open_task is True


class TestExtractAugiId:
    """A container note's identity, read from its frontmatter."""

    def test_reads_a_plain_value(self):
        assert _extract_augi_id("---\naugi_id: 898c3199-596c\n---\nbody") == "898c3199-596c"

    def test_reads_a_quoted_value(self):
        assert _extract_augi_id('---\naugi_id: "quoted-id"\n---\n') == "quoted-id"
        assert _extract_augi_id("---\naugi_id: 'sq-id'\n---\n") == "sq-id"

    def test_none_without_frontmatter(self):
        assert _extract_augi_id("augi_id: not-in-frontmatter\n") is None

    def test_none_when_absent(self):
        assert _extract_augi_id("---\ndescription: x\n---\nbody") is None

    def test_ignores_a_body_line_that_looks_like_one(self):
        assert _extract_augi_id("---\ndescription: x\n---\naugi_id: nope\n") is None

    def test_takes_any_opaque_token_not_only_uuids(self):
        """It is a name, not a structure — validating it as a UUID would only
        break vaults that name their notes some other way."""
        assert _extract_augi_id("---\naugi_id: moc.mindfulness:v1\n---\n") == "moc.mindfulness:v1"
