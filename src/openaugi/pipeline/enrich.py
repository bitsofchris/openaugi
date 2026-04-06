"""Tag enrichment — export inventory, launch agent or API, apply results.

Two paths:
1. Agent (default): Export data → launch claude CLI → agent writes results → apply
2. API (--model): Direct LLM calls for headless/automated use

The agent path requires no API key — it uses the user's existing Claude Code session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openaugi.pipeline.taxonomy import TagRules, apply_tag_rules
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

ENRICH_DIR_NAME = "enrich"


def get_enrich_dir() -> Path:
    d = Path.home() / ".openaugi" / ENRICH_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_inventory(store: SQLiteStore) -> dict[str, Path]:
    """Export tag inventory + doc summaries for agent analysis.

    Returns paths to the exported files.
    """
    enrich_dir = get_enrich_dir()

    # Tag inventory
    tags = store.get_tag_details(limit=500)
    tag_lines = []
    for t in tags:
        tag_lines.append(
            f"- {t['tag_name']} | entries: {t['entry_count']} | "
            f"last: {t['last_active'] or 'unknown'} | score: {t['hub_score']:.1f}"
        )

    tag_file = enrich_dir / "tag_inventory.md"
    tag_file.write_text(
        f"# Tag Inventory\n\n{len(tags)} tags found.\n\n" + "\n".join(tag_lines),
        encoding="utf-8",
    )

    # Document summaries (title + folder + source tags + preview)
    docs = store.get_blocks_by_kind("document", limit=100_000)
    doc_lines = []
    for doc in docs:
        folder = doc.metadata.get("source_path", "")
        src_tags = ", ".join(doc.tags) if doc.tags else "none"
        preview = (doc.content or "")[:150].replace("\n", " ")
        doc_lines.append(
            f"- **{doc.title or '(untitled)'}** | folder: {folder} | "
            f"tags: {src_tags} | preview: {preview}"
        )

    doc_file = enrich_dir / "doc_inventory.md"
    doc_file.write_text(
        f"# Document Inventory\n\n{len(docs)} documents found.\n\n" + "\n".join(doc_lines),
        encoding="utf-8",
    )

    # Write the output template so the agent knows where to write
    output_template = enrich_dir / "README.md"
    output_template.write_text(
        "# Enrichment Working Directory\n\n"
        "## Input files (read these)\n"
        "- `tag_inventory.md` — all tags with usage counts\n"
        "- `doc_inventory.md` — all documents with titles, folders, tags\n\n"
        "## Output files (agent writes these)\n"
        "- `tag_rules.json` — taxonomy + merge rules + ignore patterns\n"
        "- `doc_tags.json` — document classifications {doc_title: [tags]}\n\n"
        "## Format for tag_rules.json\n"
        "```json\n"
        "{\n"
        '  "taxonomy": {"area": ["health", "work", ...], "type": [...], ...},\n'
        '  "ignore_patterns": ["^\\\\d+$", "^\\\\d{4}-\\\\d{2}"],\n'
        '  "merge": {"old-tag": ["area/new", "type/new"], ...},\n'
        '  "discovered_topics": ["openaugi", "comedy", ...]\n'
        "}\n"
        "```\n\n"
        "## Format for doc_tags.json\n"
        "```json\n"
        "{\n"
        '  "Document Title": ["area/health", "type/journal"],\n'
        '  "Another Doc": ["area/work", "type/idea", "topic/openaugi"]\n'
        "}\n"
        "```\n",
        encoding="utf-8",
    )

    logger.info(
        "Exported inventory: %d tags, %d docs to %s",
        len(tags),
        len(docs),
        enrich_dir,
    )
    return {
        "tag_inventory": tag_file,
        "doc_inventory": doc_file,
        "enrich_dir": enrich_dir,
    }


def build_agent_prompt(enrich_dir: Path) -> str:
    """Build the prompt for the Claude agent to analyze and classify."""
    return f"""You are enriching a personal knowledge base with a clean tag taxonomy.

## Your task

1. Read the tag inventory at {enrich_dir}/tag_inventory.md
2. Read the document inventory at {enrich_dir}/doc_inventory.md
3. Analyze the existing tags and propose a faceted taxonomy
4. Classify documents against the taxonomy
5. Write results to {enrich_dir}/tag_rules.json and {enrich_dir}/doc_tags.json

## Taxonomy priors

Organize into these facets:
- **area/** — domain of life (health, family, work, learning, content, self, etc.)
- **type/** — what kind of artifact (idea, task, insight, memory, question, reference, journal)
- **status/** — lifecycle, ONLY for tasks/workstreams (active, done, parked, review)
- **topic/** — subject matter, open-ended (discovered from the data)

## Rules

- Be pragmatic. Don't over-categorize.
- Ignore garbage tags (pure numbers, date strings, accidental tags).
- Map existing tags to faceted equivalents where the mapping is clear.
- Tags already in good shape can stay as-is.
- For documents: use title, folder, and tags as signals. Assign 2-5.
- Work in batches. Focus on patterns by folder and title similarity.

## Output

Write two files:
1. `{enrich_dir}/tag_rules.json` — see README.md for format
2. `{enrich_dir}/doc_tags.json` — see README.md for format

When done, say "Enrichment complete" so the user knows to run `openaugi enrich --apply`."""


def launch_agent(enrich_dir: Path) -> bool:
    """Launch claude CLI with the enrichment prompt.

    Returns True if claude was found and launched, False otherwise.
    """
    import shutil
    import subprocess

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False

    prompt = build_agent_prompt(enrich_dir)

    subprocess.run(
        [claude_bin, "-p", prompt, "--allowedTools", "Read,Write,Glob,Grep"],
        check=False,
    )
    return True


def apply_results(store: SQLiteStore) -> dict:
    """Apply agent-generated tag rules and document classifications.

    Reads tag_rules.json and doc_tags.json from the enrich directory.
    """
    enrich_dir = get_enrich_dir()
    stats = {"rules_applied": False, "docs_classified": 0, "entries_updated": 0}

    # Apply tag rules
    rules_path = enrich_dir / "tag_rules.json"
    if rules_path.exists():
        rules = TagRules.load(rules_path)
        result = apply_tag_rules(store, rules)
        stats["rules_applied"] = True
        stats["entries_updated"] = result["blocks_updated"]

        # Also save to canonical location
        canonical = Path.home() / ".openaugi" / "tag_rules.json"
        rules.save(canonical)
    else:
        logger.warning("No tag_rules.json found in %s", enrich_dir)

    # Apply document classifications
    doc_tags_path = enrich_dir / "doc_tags.json"
    if doc_tags_path.exists():
        doc_tags = json.loads(doc_tags_path.read_text(encoding="utf-8"))
        docs_classified = _apply_doc_tags(store, doc_tags)
        stats["docs_classified"] = docs_classified
    else:
        logger.warning("No doc_tags.json found in %s", enrich_dir)

    return stats


def _apply_doc_tags(store: SQLiteStore, doc_tags: dict[str, list[str]]) -> int:
    """Apply document-level tags and trickle down to entries."""
    docs = store.get_blocks_by_kind("document", limit=100_000)

    # Build title → doc lookup
    title_to_doc = {d.title: d for d in docs if d.title}

    classified = 0
    for title, tags in doc_tags.items():
        doc = title_to_doc.get(title)
        if not doc:
            continue

        # Set computed_tags on document
        doc.metadata["computed_tags"] = tags
        store.conn.execute(
            "UPDATE blocks SET metadata = ? WHERE id = ?",
            (doc.metadata_json(), doc.id),
        )

        # Trickle to entries
        entries = store.get_entries_for_document(doc.id)
        for entry in entries:
            existing = entry.metadata.get("computed_tags", [])
            merged = list(dict.fromkeys(existing + tags))  # dedup, preserve order
            entry.metadata["computed_tags"] = merged
            store.conn.execute(
                "UPDATE blocks SET metadata = ? WHERE id = ?",
                (entry.metadata_json(), entry.id),
            )

        classified += 1

    store.conn.commit()
    return classified
