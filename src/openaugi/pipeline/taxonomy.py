"""Taxonomy discovery and tag normalization.

Two-phase approach:
1. Discover: LLM analyzes existing tags → proposes faceted taxonomy + rules
2. Apply: Rules normalize source tags → computed_tags on blocks

Source tags are never modified. Computed tags live in metadata["computed_tags"].
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from openaugi.model.protocols import LLMModel
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

# Default taxonomy priors — the agent refines these, doesn't start from scratch
DEFAULT_TAXONOMY = {
    "area": {
        "description": "Domain of life or work",
        "examples": ["health", "family", "work", "learning", "content", "self"],
    },
    "type": {
        "description": "What kind of thought or artifact",
        "examples": ["idea", "task", "insight", "memory", "question", "reference", "journal"],
    },
    "status": {
        "description": "Lifecycle state (only for tasks and workstreams)",
        "examples": ["active", "done", "parked", "review"],
    },
    "topic": {
        "description": "Subject matter (open-ended, discovered from data)",
        "examples": [],
    },
}

# Patterns that are almost always garbage tags
DEFAULT_IGNORE_PATTERNS = [
    r"^\d+$",  # pure numbers: 1, 2, 267
    r"^\d{4}-\d{2}",  # date strings: 2023-12-15
]


class TagRules(BaseModel):
    """Tag normalization rules — saved to config, applied mechanically."""

    taxonomy: dict[str, list[str]] = Field(default_factory=dict)
    ignore_patterns: list[str] = Field(default_factory=list)
    merge: dict[str, list[str]] = Field(default_factory=dict)
    discovered_topics: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2), encoding="utf-8")
        logger.info("Saved tag rules to %s", path)

    @classmethod
    def load(cls, path: Path) -> TagRules:
        if not path.exists():
            return cls(ignore_patterns=DEFAULT_IGNORE_PATTERNS)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


class TaxonomyProposal(BaseModel):
    """LLM output: proposed taxonomy from analyzing existing tags."""

    facets: dict[str, list[str]] = Field(
        description="Facet name → list of values. e.g. {'area': ['health', 'work', ...]}"
    )
    merge_rules: dict[str, list[str]] = Field(description="Source tag → list of computed tags")
    ignore_tags: list[str] = Field(description="Tags to ignore (garbage, dates, numbers)")
    topics: list[str] = Field(description="Discovered topic tags (open-ended subject matter)")


def discover_taxonomy(store: SQLiteStore, llm: LLMModel) -> TagRules:
    """Analyze existing tags and propose a faceted taxonomy.

    Uses LLM to classify existing tags into facets and propose merge rules.
    Returns TagRules ready for human review.
    """
    tag_details = store.get_tag_details(limit=300)

    # Build tag inventory for the LLM
    tag_lines = []
    for t in tag_details:
        tag_lines.append(f"- {t['tag_name']} ({t['entry_count']} entries)")

    tag_inventory = "\n".join(tag_lines)

    taxonomy_desc = "\n".join(
        f"- **{facet}**: {info['description']}. Examples: {', '.join(info['examples'])}"
        for facet, info in DEFAULT_TAXONOMY.items()
    )

    prompt = (
        "Analyze these tags from a personal knowledge base "
        "and organize them into a faceted taxonomy.\n\n"
        "## Existing Tags (by usage)\n\n"
        f"{tag_inventory}\n\n"
        "## Taxonomy Priors\n\n"
        f"Organize tags into these facets:\n{taxonomy_desc}\n\n"
        "## Instructions\n\n"
        "1. **Facets**: For each facet (area, type, status, topic), "
        "list the values. Status is ONLY for tasks/workstreams. "
        "Topic is open-ended, discovered from data.\n\n"
        "2. **Merge rules**: Map existing tags to faceted equivalents.\n"
        '   e.g. "idea/content/bits-of-chris" → '
        '["area/content", "type/idea", "topic/bits-of-chris"]\n'
        "   Only include tags that need mapping.\n\n"
        "3. **Ignore tags**: List garbage tags "
        "(pure numbers, dates, accidental, meaningless).\n\n"
        "4. **Topics**: List real subject-matter topics "
        'from the tags (e.g. "openaugi", "comedy").\n\n'
        "Be pragmatic. Don't over-categorize. "
        "Garbage tags should be ignored. "
        "Well-structured tags should be kept."
    )

    system = (
        "You are a data engineer organizing a personal knowledge base's tag system. "
        "Be pragmatic and concise. Output valid JSON."
    )

    result = llm.structured_output(prompt, TaxonomyProposal, system=system)
    proposal: TaxonomyProposal = result  # type: ignore[assignment]

    return TagRules(
        taxonomy=proposal.facets,
        ignore_patterns=DEFAULT_IGNORE_PATTERNS,
        merge={k: v for k, v in proposal.merge_rules.items()},
        discovered_topics=proposal.topics,
    )


def apply_tag_rules(store: SQLiteStore, rules: TagRules) -> dict:
    """Apply tag rules to all blocks: normalize source tags → computed_tags.

    Computed tags are stored in metadata["computed_tags"].
    Source tags are never modified.
    """
    ignore_regexes = [re.compile(p) for p in rules.ignore_patterns]

    entries = store.get_blocks_by_kind("entry", limit=100_000)
    updated = 0

    for entry in entries:
        computed = _normalize_tags(entry.tags, rules, ignore_regexes)
        if computed != entry.metadata.get("computed_tags"):
            entry.metadata["computed_tags"] = computed
            store.conn.execute(
                "UPDATE blocks SET metadata = ? WHERE id = ?",
                (entry.metadata_json(), entry.id),
            )
            updated += 1

    if updated:
        store.conn.commit()

    logger.info("Applied tag rules: %d blocks updated", updated)
    return {"blocks_updated": updated, "total_entries": len(entries)}


def _normalize_tags(
    source_tags: list[str],
    rules: TagRules,
    ignore_regexes: list[re.Pattern],
) -> list[str]:
    """Normalize a list of source tags using rules."""
    computed: list[str] = []

    for tag in source_tags:
        # Skip ignored tags
        if any(rx.match(tag) for rx in ignore_regexes):
            continue

        # Apply merge rules
        if tag in rules.merge:
            computed.extend(rules.merge[tag])
        else:
            # Keep the original tag if no merge rule
            computed.append(tag)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for t in computed:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped
