"""Tests for Block model."""

from openaugi.model.block import Block


class TestBlockModel:
    def test_create_entry_block(self):
        b = Block(id="abc123", kind="entry", content="Hello world", source="vault")
        assert b.id == "abc123"
        assert b.kind == "entry"
        assert b.content == "Hello world"
        assert b.content_hash is not None  # auto-computed

    def test_content_hash_auto_computed(self):
        b = Block(id="a", kind="entry", content="test content")
        assert b.content_hash == Block.hash_content("test content")

    def test_content_hash_not_overwritten(self):
        b = Block(id="a", kind="entry", content="test", content_hash="explicit")
        assert b.content_hash == "explicit"

    def test_make_id_deterministic(self):
        id1 = Block.make_id("notes/daily.md", "abc123")
        id2 = Block.make_id("notes/daily.md", "abc123")
        assert id1 == id2

    def test_make_id_different_for_different_content(self):
        id1 = Block.make_id("notes/daily.md", "hash1")
        id2 = Block.make_id("notes/daily.md", "hash2")
        assert id1 != id2

    def test_make_tag_id_deterministic(self):
        id1 = Block.make_tag_id("career")
        id2 = Block.make_tag_id("career")
        assert id1 == id2

    def test_make_document_id_deterministic(self):
        id1 = Block.make_document_id("notes/daily.md")
        id2 = Block.make_document_id("notes/daily.md")
        assert id1 == id2

    def test_tags_json(self):
        b = Block(id="a", kind="entry", tags=["career", "project"])
        assert '"career"' in b.tags_json()
        assert '"project"' in b.tags_json()

    def test_metadata_json(self):
        b = Block(id="a", kind="entry", metadata={"h3_date": "2024-03-15"})
        assert '"h3_date"' in b.metadata_json()

    def test_block_no_content_no_hash(self):
        b = Block(id="a", kind="tag", title="career")
        assert b.content is None
        assert b.content_hash is None

    def test_hash_content_stable(self):
        h1 = Block.hash_content("same content")
        h2 = Block.hash_content("same content")
        assert h1 == h2
        assert len(h1) == 16
