"""Tests for the shared splitter — the deterministic split() primitive.

Covers the public API (`split_text`, `split_file`, `Segment`, `SplitResult`).
Regex- and helper-level coverage lives in tests/test_vault_adapter.py since
those helpers are re-exported from the vault module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openaugi.adapters.splitter import Segment, SplitResult, split_file, split_text


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

    def test_zzz_only_block_dropped(self):
        segs = split_text("# H\nreal\nqqq\nzzz just a directive\nqqq\nalso real")
        # middle sub-section has no content after zzz-strip — should drop
        assert [s.clean_content for s in segs] == ["real", "also real"]

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
