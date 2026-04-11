"""Tests for the embedding pipeline — content cleaning and text building."""

from openaugi.pipeline.embed import _clean_for_embedding


class TestCleanForEmbedding:
    def test_plain_text_unchanged(self):
        text = "Just a normal thought about something."
        assert _clean_for_embedding(text) == text

    def test_image_embed_stripped(self):
        text = "Some text\n![rw-book-cover](https://example.com/cover.png)\nmore text"
        result = _clean_for_embedding(text)
        assert "![" not in result
        assert "example.com" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_raw_url_stripped(self):
        text = "Check this out: https://github.com/some/repo for details"
        result = _clean_for_embedding(text)
        assert "https://" not in result
        assert "Check this out:" in result
        assert "for details" in result

    def test_markdown_link_keeps_text(self):
        text = "Read [the docs](https://docs.example.com) for more"
        result = _clean_for_embedding(text)
        assert "the docs" in result
        assert "https://" not in result
        assert "[" not in result

    def test_bold_markers_stripped(self):
        assert _clean_for_embedding("This is **important** stuff") == "This is important stuff"

    def test_italic_markers_stripped(self):
        assert _clean_for_embedding("This is *emphasized* text") == "This is emphasized text"

    def test_checkbox_marker_stripped_keeps_text(self):
        text = "- [ ] Buy groceries\n- [x] Done task"
        result = _clean_for_embedding(text)
        assert "[ ]" not in result
        assert "[x]" not in result
        assert "Buy groceries" in result
        assert "Done task" in result

    def test_blockquote_marker_stripped(self):
        text = "> A quoted thought\n> continuing here"
        result = _clean_for_embedding(text)
        assert ">" not in result
        assert "A quoted thought" in result
        assert "continuing here" in result

    def test_horizontal_rule_stripped(self):
        text = "Before\n---\nAfter"
        result = _clean_for_embedding(text)
        assert "---" not in result
        assert "Before" in result
        assert "After" in result

    def test_wikilinks_preserved(self):
        text = "Working on [[Project Alpha]] today"
        result = _clean_for_embedding(text)
        assert "[[Project Alpha]]" in result

    def test_inline_tags_preserved(self):
        text = "Thinking about #career and #ai/research"
        result = _clean_for_embedding(text)
        assert "#career" in result
        assert "#ai/research" in result

    def test_readwise_block_cleaned(self):
        """Typical Readwise import block: title + cover image."""
        text = (
            "What Every Successful Person Knows, but Never Says\n"
            "![rw-book-cover](https://readwise-assets.s3.amazonaws.com/static/images/article3.5c705a01b476.png)"
        )
        result = _clean_for_embedding(text)
        assert "What Every Successful Person Knows" in result
        assert "readwise-assets" not in result
        assert "![" not in result

    def test_multiple_blank_lines_collapsed(self):
        text = "line one\n\n\n\nline two"
        result = _clean_for_embedding(text)
        assert "\n\n\n" not in result
        assert "line one" in result
        assert "line two" in result

    def test_empty_string_returns_empty(self):
        assert _clean_for_embedding("") == ""

    def test_image_only_block_returns_empty(self):
        text = "![cover](https://example.com/img.png)"
        assert _clean_for_embedding(text) == ""
