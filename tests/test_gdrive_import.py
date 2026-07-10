"""Tests for the Google Drive → vault markdown converter (scripts/gdrive_import.py).

Covers the pure transform functions: note-type mapping, frontmatter assembly,
granularity detection/splitting, and the full stamp. IO (pandoc, rclone) is
not exercised here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the script module (it lives in scripts/, not the package).
_spec = importlib.util.spec_from_file_location(
    "gdrive_import",
    Path(__file__).resolve().parent.parent / "scripts" / "gdrive_import.py",
)
assert _spec and _spec.loader
gdrive_import = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gdrive_import  # dataclass needs the module registered
_spec.loader.exec_module(gdrive_import)


class TestNoteTypeFor:
    def test_journal_maps_to_reflection(self):
        assert gdrive_import.note_type_for("Self/Journal") == "note-type/reflection"

    def test_deep_subfolder_inherits(self):
        assert gdrive_import.note_type_for("Self/Journal/2015") == "note-type/reflection"

    def test_physical_maps_to_reflection(self):
        assert gdrive_import.note_type_for("Self/Physical") == "note-type/reflection"

    def test_loose_self_typed_reflection(self):
        # exact rule: loose Self docs (therapy, personal) → reflection
        assert gdrive_import.note_type_for("Self") == "note-type/reflection"

    def test_reference_and_notes_untyped(self):
        # exact "Self" rule must NOT leak into Self subfolders
        assert gdrive_import.note_type_for("Self/Reference & Notes") is None

    def test_people_untyped(self):
        assert gdrive_import.note_type_for("Self/People") is None

    def test_ideas_untyped(self):
        # note-type/idea is proposed for approval, NOT auto-applied
        assert gdrive_import.note_type_for("Ideas／ Someday/Book Ideas") is None

    def test_unknown_untyped(self):
        assert gdrive_import.note_type_for("Coding/Courses") is None


class TestCleanExportArtifacts:
    def test_unescapes_punctuation(self):
        out = gdrive_import.clean_export_artifacts(r"\- bullet \* star \[link\]")
        assert out == "- bullet * star [link]"

    def test_drops_hardbreak_backslashes(self):
        out = gdrive_import.clean_export_artifacts("line one\\\nline two")
        assert out == "line one\nline two"

    def test_removes_standalone_backslash_lines(self):
        out = gdrive_import.clean_export_artifacts("para\n\n\\\n\nmore")
        assert "\\" not in out
        assert "para" in out and "more" in out

    def test_preserves_real_text(self):
        out = gdrive_import.clean_export_artifacts("A normal sentence about C++ & things.")
        assert out == "A normal sentence about C++ & things."

    def test_strips_span_keeps_inner_text(self):
        out = gdrive_import.clean_export_artifacts(
            'be present<span class="Apple-converted-space">  </span>now'
        )
        assert "<span" not in out and "</span>" not in out
        assert "be present" in out and "now" in out

    def test_strips_sup_and_u(self):
        out = gdrive_import.clean_export_artifacts("x<sup>2</sup> and <u>under</u>")
        assert "x2" in out and "under" in out
        assert "<sup>" not in out and "<u>" not in out

    def test_removes_data_uri_image(self):
        out = gdrive_import.clean_export_artifacts(
            "before ![img](data:image/png;base64,AAAA) after"
        )
        assert "data:image" not in out
        assert "before" in out and "after" in out

    def test_preserves_autolinks_and_literal_brackets(self):
        out = gdrive_import.clean_export_artifacts("see <https://example.com> and <Your Name>")
        assert "<https://example.com>" in out
        assert "<Your Name>" in out


class TestDateOnly:
    def test_truncates_iso(self):
        assert gdrive_import.date_only("2024-07-30T00:29:26.235Z") == "2024-07-30"

    def test_date_passthrough(self):
        assert gdrive_import.date_only("2015-03-12") == "2015-03-12"

    def test_none_safe(self):
        assert gdrive_import.date_only(None) is None


class TestDriveUrl:
    def test_native_doc_url(self):
        assert (
            gdrive_import.drive_url("ABC123", True) == "https://docs.google.com/document/d/ABC123"
        )

    def test_uploaded_file_url(self):
        assert gdrive_import.drive_url("ABC123", False) == "https://drive.google.com/file/d/ABC123"


class TestBuildFrontmatter:
    def test_full_with_note_type(self):
        fm = gdrive_import.build_frontmatter(
            block_date="2023-10-01",
            gdrive_created="2013-05-02",
            gdrive_modified="2023-10-01",
            gdrive_path="Self/Journal",
            gdrive_url_="https://docs.google.com/document/d/X",
            note_type="note-type/reflection",
        )
        assert fm.startswith("---\n")
        assert fm.endswith("\n---")
        assert "created: 2023-10-01" in fm
        assert "gdrive_created: 2013-05-02" in fm
        assert "gdrive_modified: 2023-10-01" in fm
        assert "gdrive_path: Self/Journal" in fm
        assert "tags:\n  - note-type/reflection" in fm

    def test_untyped_omits_tags(self):
        fm = gdrive_import.build_frontmatter(
            block_date="2023-10-01",
            gdrive_created="2013-05-02",
            gdrive_modified="2023-10-01",
            gdrive_path="Self/Reference & Notes",
            gdrive_url_="https://docs.google.com/document/d/X",
            note_type=None,
        )
        assert "tags:" not in fm

    def test_path_with_colon_is_quoted(self):
        fm = gdrive_import.build_frontmatter(
            block_date=None,
            gdrive_created=None,
            gdrive_modified=None,
            gdrive_path="Self/Notes: misc",
            gdrive_url_="u",
            note_type=None,
        )
        assert 'gdrive_path: "Self/Notes: misc"' in fm


class TestGranularity:
    def test_short_doc_no_split(self):
        assert gdrive_import.needs_granularity_split("A short note.") is False

    def test_long_headerless_needs_split(self):
        body = "word " * 2000
        assert gdrive_import.needs_granularity_split(body) is True

    def test_long_with_heading_no_split(self):
        body = "## Section\n\n" + "word " * 2000
        assert gdrive_import.needs_granularity_split(body) is False

    def test_hashtag_not_a_heading(self):
        # an inline #tag is not an ATX heading
        body = "#personal stuff " + "word " * 2000
        assert gdrive_import.needs_granularity_split(body) is True

    def test_insert_qqq_creates_breaks(self):
        paras = ["word " * 200 for _ in range(5)]  # ~1000 words, 5 paras
        body = "\n\n".join(paras)
        out = gdrive_import.insert_qqq(body, target_words=400)
        assert "\nqqq\n" in out
        # regrouped into ~3 chunks → 2 delimiters
        assert out.count("qqq") >= 2

    def test_insert_qqq_preserves_text(self):
        body = "alpha beta\n\ngamma delta"
        out = gdrive_import.insert_qqq(body, target_words=1)
        assert "alpha beta" in out and "gamma delta" in out


class TestStampDocument:
    def test_frontmatter_prepended_and_body_kept(self):
        out = gdrive_import.stamp_document(
            "  My reflection.  ",
            block_date="2020-01-01",
            gdrive_created="2019-01-01",
            gdrive_modified="2020-01-01",
            gdrive_path="Self/Journal",
            gdrive_url_="u",
            note_type="note-type/reflection",
        )
        assert out.startswith("---\n")
        assert "My reflection." in out
        assert out.rstrip().endswith("My reflection.")

    def test_long_headerless_gets_qqq(self):
        body = "word " * 2000
        out = gdrive_import.stamp_document(
            body,
            block_date="2020-01-01",
            gdrive_created=None,
            gdrive_modified="2020-01-01",
            gdrive_path="Self/Reference & Notes",
            gdrive_url_="u",
            note_type=None,
        )
        assert "qqq" in out


class TestLoadInventory:
    def test_indexes_by_dir_and_stem(self, tmp_path: Path):
        inv = tmp_path / "inv.json"
        inv.write_text(
            '[{"Path":"Self/Journal/Foo.docx","ID":"X","bucket":"gdoc",'
            '"created":"2015-01-01T00:00:00Z","modified":"2016-01-01T00:00:00Z"}]'
        )
        index = gdrive_import.load_inventory(inv)
        meta = index[("Self/Journal", "Foo")]
        assert meta.file_id == "X"
        assert meta.bucket == "gdoc"
        assert meta.created.startswith("2015")
