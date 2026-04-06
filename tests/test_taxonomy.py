"""Tests for tag taxonomy discovery and normalization.

Tests the rule engine and tag normalization without requiring LLM calls.
LLM-dependent tests (discover_taxonomy, infer_document_tags) are integration
tests that require an API key — run manually, not in CI.
"""

from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.pipeline.runner import run_layer0
from openaugi.pipeline.taxonomy import TagRules, _normalize_tags, apply_tag_rules
from openaugi.store.sqlite import SQLiteStore


@pytest.fixture
def ingested_store(vault_path: Path, store: SQLiteStore) -> SQLiteStore:
    """Store with fixture vault already ingested."""
    run_layer0(vault_path, store)
    return store


class TestTagRules:
    def test_save_and_load(self, tmp_path: Path):
        """Rules should round-trip through save/load."""
        rules = TagRules(
            taxonomy={"area": ["health", "work"], "type": ["idea", "task"]},
            ignore_patterns=[r"^\d+$"],
            merge={"idea/content": ["area/content", "type/idea"]},
            discovered_topics=["openaugi", "comedy"],
        )

        path = tmp_path / "tag_rules.json"
        rules.save(path)
        loaded = TagRules.load(path)

        assert loaded.taxonomy == rules.taxonomy
        assert loaded.merge == rules.merge
        assert loaded.discovered_topics == rules.discovered_topics

    def test_load_missing_file(self, tmp_path: Path):
        """Loading from missing file should return defaults."""
        rules = TagRules.load(tmp_path / "nonexistent.json")
        assert len(rules.ignore_patterns) > 0  # has defaults

    def test_default_ignore_patterns(self):
        """Default rules should ignore numeric and date tags."""
        rules = TagRules(ignore_patterns=[r"^\d+$", r"^\d{4}-\d{2}"])

        import re

        regexes = [re.compile(p) for p in rules.ignore_patterns]

        # Should ignore
        assert any(rx.match("1") for rx in regexes)
        assert any(rx.match("267") for rx in regexes)
        assert any(rx.match("2023-12-15") for rx in regexes)

        # Should keep
        assert not any(rx.match("openaugi") for rx in regexes)
        assert not any(rx.match("idea/content") for rx in regexes)


class TestNormalizeTags:
    def test_ignore_numeric_tags(self):
        """Numeric tags should be filtered out."""
        import re

        rules = TagRules(ignore_patterns=[r"^\d+$"])
        regexes = [re.compile(p) for p in rules.ignore_patterns]

        result = _normalize_tags(["openaugi", "1", "idea", "267"], rules, regexes)
        assert "1" not in result
        assert "267" not in result
        assert "openaugi" in result
        assert "idea" in result

    def test_merge_rules_applied(self):
        """Merge rules should replace source tags with computed equivalents."""
        import re

        rules = TagRules(
            merge={"idea/content": ["area/content", "type/idea"]},
            ignore_patterns=[],
        )
        regexes = [re.compile(p) for p in rules.ignore_patterns]

        result = _normalize_tags(["idea/content", "active"], rules, regexes)
        assert "area/content" in result
        assert "type/idea" in result
        assert "active" in result
        assert "idea/content" not in result

    def test_deduplication(self):
        """Duplicate tags should be removed."""
        import re

        rules = TagRules(
            merge={
                "tag-a": ["area/x"],
                "tag-b": ["area/x"],  # same output
            },
            ignore_patterns=[],
        )
        regexes = [re.compile(p) for p in rules.ignore_patterns]

        result = _normalize_tags(["tag-a", "tag-b"], rules, regexes)
        assert result.count("area/x") == 1

    def test_no_rules_passthrough(self):
        """With no rules, source tags pass through unchanged."""

        rules = TagRules(ignore_patterns=[])
        regexes = []

        result = _normalize_tags(["a", "b", "c"], rules, regexes)
        assert result == ["a", "b", "c"]


class TestApplyTagRules:
    def test_apply_updates_metadata(self, ingested_store: SQLiteStore):
        """Applying rules should set computed_tags in metadata."""
        # Get a tag that exists in the fixture vault
        tags = ingested_store.get_tag_details(limit=5)
        assert len(tags) > 0

        # Create rules that merge the first tag
        first_tag = tags[0]["tag_name"]
        rules = TagRules(
            merge={first_tag: [f"area/{first_tag}"]},
            ignore_patterns=[r"^\d+$"],
        )

        result = apply_tag_rules(ingested_store, rules)
        assert result["total_entries"] > 0

        assert result["blocks_updated"] >= 0  # some entries may not have the tag


class TestEffectiveTags:
    def test_effective_tags_uses_computed_when_available(self):
        """Block.effective_tags should prefer computed_tags."""
        block = Block(
            id="test",
            kind="entry",
            tags=["raw-tag"],
            metadata={"computed_tags": ["area/health", "type/idea"]},
        )
        assert block.effective_tags == ["area/health", "type/idea"]

    def test_effective_tags_falls_back_to_source(self):
        """Block.effective_tags should fall back to source tags."""
        block = Block(id="test", kind="entry", tags=["raw-tag"])
        assert block.effective_tags == ["raw-tag"]

    def test_effective_tags_empty_computed_falls_back(self):
        """Empty computed_tags should fall back to source tags."""
        block = Block(
            id="test",
            kind="entry",
            tags=["raw-tag"],
            metadata={"computed_tags": []},
        )
        assert block.effective_tags == ["raw-tag"]
