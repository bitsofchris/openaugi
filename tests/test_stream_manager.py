"""Tests for StreamManager — workstream file CRUD.

Unit tests for the StreamManager class and helpers. MCP tool integration
tests for stream tools are in test_mcp.py.
"""

from datetime import date

from openaugi.mcp.stream_manager import (
    StreamManager,
    _parse_stream_file,
    _serialize_stream,
    slugify,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Product Management") == "product-management"

    def test_special_chars(self):
        assert slugify("API v2 — Design!") == "api-v2-design"

    def test_already_slug(self):
        assert slugify("product-management") == "product-management"

    def test_leading_trailing_stripped(self):
        assert slugify("  Hello World  ") == "hello-world"

    def test_empty_after_strip(self):
        assert slugify("!!!") == ""


class TestParseStreamFile:
    def test_full_file(self):
        text = (
            "---\n"
            "stream: Product Management\n"
            "status: active\n"
            "last_active: '2026-04-03'\n"
            "linked_sessions:\n"
            "  - abc123\n"
            "---\n"
            "\n"
            "## Context\n"
            "Q2 roadmap planning.\n"
            "\n"
            "## LEFT OFF\n"
            "Reviewing ICE scoring.\n"
            "\n"
            "## Log\n"
            "- 2026-04-03: Started planning\n"
        )
        parsed = _parse_stream_file(text)
        assert parsed["frontmatter"]["stream"] == "Product Management"
        assert parsed["frontmatter"]["status"] == "active"
        assert parsed["frontmatter"]["linked_sessions"] == ["abc123"]
        assert parsed["context"] == "Q2 roadmap planning."
        assert parsed["left_off"] == "Reviewing ICE scoring."
        assert "Started planning" in parsed["log"]

    def test_empty_sections(self):
        text = "---\nstream: Empty\nstatus: active\n---\n\n## Context\n\n## LEFT OFF\n\n## Log\n"
        parsed = _parse_stream_file(text)
        assert parsed["context"] == ""
        assert parsed["left_off"] == ""
        assert parsed["log"] == ""

    def test_no_frontmatter(self):
        text = "## Context\nSome context\n\n## LEFT OFF\n\n## Log\n"
        parsed = _parse_stream_file(text)
        assert parsed["frontmatter"] == {}
        assert parsed["context"] == "Some context"


class TestSerializeStream:
    def test_roundtrip(self):
        fm = {"stream": "Test", "status": "active", "last_active": "2026-04-03"}
        text = _serialize_stream(fm, "My context", "Where I am", "- 2026-04-03: Entry")
        parsed = _parse_stream_file(text)
        assert parsed["frontmatter"]["stream"] == "Test"
        assert parsed["context"] == "My context"
        assert parsed["left_off"] == "Where I am"
        assert "Entry" in parsed["log"]


class TestMakeStream:
    def test_creates_file(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        result = manager.make_stream("Product Management", context="Q2 planning")
        assert result["status"] == "created"
        assert result["slug"] == "product-management"

        path = tmp_path / "OpenAugi" / "Streams" / "product-management.md"
        assert path.exists()
        text = path.read_text()
        assert "stream: Product Management" in text
        assert "Q2 planning" in text

    def test_duplicate_returns_error(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Test Stream")
        result = manager.make_stream("Test Stream")
        assert result["status"] == "error"
        assert "already exists" in result["reason"]

    def test_empty_name_returns_error(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        result = manager.make_stream("")
        assert result["status"] == "error"

    def test_sets_last_active_to_today(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Today Stream")
        path = tmp_path / "OpenAugi" / "Streams" / "today-stream.md"
        parsed = _parse_stream_file(path.read_text())
        assert parsed["frontmatter"]["last_active"] == date.today().isoformat()


class TestListStreams:
    def test_empty_dir(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        result = manager.list_streams()
        assert result["count"] == 0

    def test_lists_created_streams(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Alpha", context="First")
        manager.make_stream("Beta", context="Second")
        result = manager.list_streams()
        assert result["count"] == 2
        slugs = {s["slug"] for s in result["streams"]}
        assert slugs == {"alpha", "beta"}

    def test_filter_by_status(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Active One")
        manager.make_stream("Done One", status="done")
        result = manager.list_streams(status="active")
        assert result["count"] == 1
        assert result["streams"][0]["slug"] == "active-one"

    def test_left_off_preview(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Preview Test")
        manager.update_stream("preview-test", left_off="Working on the thing")
        result = manager.list_streams()
        assert result["streams"][0]["left_off_preview"] == "Working on the thing"


class TestGetStreamContext:
    def test_by_slug(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("My Stream", context="Context here")
        result = manager.get_stream_context("my-stream")
        assert result["slug"] == "my-stream"
        assert result["context"] == "Context here"

    def test_by_display_name(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Product Management")
        result = manager.get_stream_context("Product Management")
        assert result["slug"] == "product-management"

    def test_not_found(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        result = manager.get_stream_context("nonexistent")
        assert result["status"] == "error"

    def test_case_insensitive_name_match(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Product Management")
        result = manager.get_stream_context("product management")
        assert result["slug"] == "product-management"


class TestUpdateStream:
    def test_update_left_off(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Update Test")
        result = manager.update_stream("update-test", left_off="New state")
        assert result["status"] == "updated"

        ctx = manager.get_stream_context("update-test")
        assert ctx["left_off"] == "New state"

    def test_left_off_replaces(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Replace Test")
        manager.update_stream("replace-test", left_off="First")
        manager.update_stream("replace-test", left_off="Second")
        ctx = manager.get_stream_context("replace-test")
        assert ctx["left_off"] == "Second"
        assert "First" not in ctx["left_off"]

    def test_update_context(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Ctx Test", context="Original")
        manager.update_stream("ctx-test", context="Updated scope")
        ctx = manager.get_stream_context("ctx-test")
        assert ctx["context"] == "Updated scope"

    def test_log_appends(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Log Test")
        manager.update_stream("log-test", log="First entry")
        manager.update_stream("log-test", log="Second entry")
        ctx = manager.get_stream_context("log-test")
        assert "First entry" in ctx["log"]
        assert "Second entry" in ctx["log"]

    def test_link_session_deduped(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Session Test")
        manager.update_stream("session-test", session_id="abc123")
        manager.update_stream("session-test", session_id="abc123")
        manager.update_stream("session-test", session_id="def456")
        ctx = manager.get_stream_context("session-test")
        assert ctx["linked_sessions"] == ["abc123", "def456"]

    def test_update_status(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Status Test")
        manager.update_stream("status-test", status="done")
        ctx = manager.get_stream_context("status-test")
        assert ctx["status"] == "done"

    def test_combined_update(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Combo Test")
        result = manager.update_stream(
            "combo-test",
            left_off="At step 3",
            log="Finished step 2",
            session_id="sess-1",
        )
        assert result["status"] == "updated"
        ctx = manager.get_stream_context("combo-test")
        assert ctx["left_off"] == "At step 3"
        assert "Finished step 2" in ctx["log"]
        assert "sess-1" in ctx["linked_sessions"]

    def test_not_found(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        result = manager.update_stream("nonexistent", left_off="x")
        assert result["status"] == "error"

    def test_updates_last_active(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Active Test")
        manager.update_stream("active-test", log="ping")
        ctx = manager.get_stream_context("active-test")
        assert ctx["last_active"] == date.today().isoformat()

    def test_fuzzy_name_match(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Product Management")
        result = manager.update_stream("Product Management", left_off="Updated via name")
        assert result["status"] == "updated"


class TestAppendToLog:
    def test_appends(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Log Append")
        assert manager.append_to_log("log-append", "test entry") is True
        ctx = manager.get_stream_context("log-append")
        assert "test entry" in ctx["log"]

    def test_nonexistent_returns_false(self, tmp_path):
        manager = StreamManager(str(tmp_path))
        assert manager.append_to_log("nope", "entry") is False


class TestWriteSnipBacklink:
    """Test that write_snip appends a wikilink to the stream's Log."""

    def test_snip_with_stream_backlinks(self, tmp_path):
        from openaugi.mcp.doc_writer import VaultWriter

        # Create a stream first
        manager = StreamManager(str(tmp_path))
        manager.make_stream("Dev Work")

        # Write a snip linked to that stream
        writer = VaultWriter(str(tmp_path))
        result = writer.write_snip(
            "Key Insight",
            "Important finding",
            stream="dev-work",
        )
        assert result["status"] == "created"

        # Check that the stream's Log has the wikilink
        ctx = manager.get_stream_context("dev-work")
        assert "[[Key Insight]]" in ctx["log"]

    def test_snip_without_stream_no_backlink(self, tmp_path):
        from openaugi.mcp.doc_writer import VaultWriter

        writer = VaultWriter(str(tmp_path))
        result = writer.write_snip("Standalone", "No stream")
        assert result["status"] == "created"
        # No crash, no stream file needed

    def test_snip_with_nonexistent_stream_no_crash(self, tmp_path):
        from openaugi.mcp.doc_writer import VaultWriter

        writer = VaultWriter(str(tmp_path))
        result = writer.write_snip("Orphan", "content", stream="nonexistent")
        assert result["status"] == "created"
        # Backlink silently fails (stream doesn't exist), snip still saved
