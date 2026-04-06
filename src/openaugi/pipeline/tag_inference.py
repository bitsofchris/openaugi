"""Document-level tag inference via cheap LLM.

Classifies documents against the discovered taxonomy.
Tags trickle down from documents to their entries.

Uses batched calls (20 docs per request) for efficiency.
~200 tokens per doc → ~4K tokens per batch → cheap even at scale.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from openaugi.model.protocols import LLMModel
from openaugi.pipeline.taxonomy import TagRules
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
CONTENT_PREVIEW_CHARS = 500


class DocClassification(BaseModel):
    """LLM output: tags for a batch of documents."""

    classifications: dict[str, list[str]] = Field(description="doc_id → list of computed tags")
    flagged: list[str] = Field(
        default_factory=list,
        description="doc_ids that are ambiguous or need human review",
    )


def infer_document_tags(
    store: SQLiteStore,
    llm: LLMModel,
    rules: TagRules,
) -> dict:
    """Classify documents against taxonomy, trickle tags to entries.

    Returns stats dict.
    """
    docs = store.get_blocks_by_kind("document", limit=100_000)

    # Only classify docs that don't already have computed_tags
    untagged = [d for d in docs if not d.metadata.get("computed_tags")]
    logger.info("Found %d documents, %d need classification", len(docs), len(untagged))

    if not untagged:
        return {"classified": 0, "flagged": 0, "entries_updated": 0}

    # Build taxonomy context for the prompt
    taxonomy_str = _format_taxonomy(rules)

    # Process in batches
    total_classified = 0
    total_flagged: list[str] = []
    total_entries_updated = 0

    for i in range(0, len(untagged), BATCH_SIZE):
        batch = untagged[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(untagged) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info("Classifying batch %d/%d (%d docs)", batch_num, total_batches, len(batch))

        try:
            result = _classify_batch(batch, taxonomy_str, llm)
        except Exception:
            logger.exception("Batch %d failed, skipping", batch_num)
            continue

        # Apply classifications to documents
        for doc in batch:
            tags = result.classifications.get(doc.id, [])
            if tags:
                doc.metadata["computed_tags"] = tags
                store.conn.execute(
                    "UPDATE blocks SET metadata = ? WHERE id = ?",
                    (doc.metadata_json(), doc.id),
                )
                total_classified += 1

                # Trickle to entries
                entries = store.get_entries_for_document(doc.id)
                for entry in entries:
                    # Merge doc-level tags with any existing computed_tags
                    existing = entry.metadata.get("computed_tags", [])
                    merged = _merge_tags(existing, tags)
                    if merged != existing:
                        entry.metadata["computed_tags"] = merged
                        store.conn.execute(
                            "UPDATE blocks SET metadata = ? WHERE id = ?",
                            (entry.metadata_json(), entry.id),
                        )
                        total_entries_updated += 1

        total_flagged.extend(result.flagged)

        # Commit each batch
        store.conn.commit()

    stats = {
        "classified": total_classified,
        "flagged": len(total_flagged),
        "flagged_ids": total_flagged[:50],  # cap for display
        "entries_updated": total_entries_updated,
        "total_docs": len(docs),
        "skipped": len(docs) - len(untagged),
    }
    logger.info(
        "Classification complete: %d classified, %d flagged, %d entries updated",
        total_classified,
        len(total_flagged),
        total_entries_updated,
    )
    return stats


def _classify_batch(
    docs: list[Any],
    taxonomy_str: str,
    llm: LLMModel,
) -> DocClassification:
    """Classify a batch of documents."""
    doc_descriptions = []
    for doc in docs:
        folder = doc.metadata.get("source_path", "")
        preview = (doc.content or "")[:CONTENT_PREVIEW_CHARS]
        source_tags = doc.tags

        doc_descriptions.append(
            f"### {doc.id}\n"
            f"- Title: {doc.title or '(untitled)'}\n"
            f"- Folder: {folder}\n"
            f"- Existing tags: {', '.join(source_tags) if source_tags else 'none'}\n"
            f"- Preview: {preview[:200]}\n"
        )

    docs_text = "\n".join(doc_descriptions)

    prompt = f"""Classify these documents from a personal knowledge base.

## Taxonomy

{taxonomy_str}

## Documents

{docs_text}

## Instructions

For each document (by ID), assign 2-5 tags from the taxonomy.
Use the format "facet/value" (e.g. "area/health", "type/idea", "topic/openaugi").

Use title, folder path, existing tags, and content preview as signals.
If a document is ambiguous or very large, add its ID to the flagged list.

Classify every document — even if uncertain, give your best guess.
Only flag ones that are genuinely ambiguous (could be multiple very different areas)."""

    system = (
        "You are classifying personal knowledge base documents. "
        "Be pragmatic. Most documents have a clear area and type. "
        "Output valid JSON."
    )

    result = llm.structured_output(prompt, DocClassification, system=system)
    return result  # type: ignore[return-value]


def _format_taxonomy(rules: TagRules) -> str:
    """Format taxonomy for the LLM prompt."""
    lines = []
    for facet, values in rules.taxonomy.items():
        if values:
            lines.append(f"- **{facet}/**: {', '.join(values)}")
        else:
            lines.append(f"- **{facet}/**: (open-ended)")

    if rules.discovered_topics:
        lines.append(f"- **Known topics**: {', '.join(rules.discovered_topics)}")

    return "\n".join(lines)


def _merge_tags(existing: list[str], new: list[str]) -> list[str]:
    """Merge two tag lists, deduplicating."""
    seen: set[str] = set()
    result: list[str] = []
    for t in existing + new:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result
