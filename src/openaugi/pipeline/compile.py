"""Compile pipeline — materialize context blocks from data blocks.

Context blocks are the navigational metadata layer. They give agents a map
of the knowledge graph without requiring them to read every raw block.

Compile layers:
- Layer 0 (FREE): hub_summary, recent_activity, index — pure SQL aggregation
- Layer 1 (FREE): concept pages, graph_health — SQL + templates
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

COMPILED_SOURCE = "compiled"


def run_compile(
    store: SQLiteStore,
    vault_path: str | Path | None = None,
    layer: int = 1,
    top_n: int = 50,
    force: bool = False,
) -> dict:
    """Run the compile pipeline. Materializes context blocks in the store.

    Args:
        store: SQLiteStore instance.
        vault_path: If provided, render context blocks as .md to vault.
        layer: Max compile layer (0 or 1). Layer 2 = LLM, deferred.
        top_n: Number of top hubs to compile.
        force: Recompile even if context blocks exist.

    Returns:
        Stats dict with counts of context blocks created.
    """
    # Clear old context blocks (full recompile each time — they're cheap)
    deleted = store.delete_blocks_by_source(COMPILED_SOURCE)
    if deleted:
        logger.info("Cleared %d old context blocks", deleted)

    context_blocks: list[Block] = []
    context_links: list[Link] = []

    # Layer 0: pure SQL aggregation
    hub_summaries = _compile_hub_summaries(store, top_n)
    context_blocks.extend(hub_summaries)

    recent = _compile_recent_activity(store)
    context_blocks.append(recent)

    # Layer 1: templates + deeper queries
    if layer >= 1:
        concepts = _compile_concept_pages(store, top_n)
        for concept_block, concept_links in concepts:
            context_blocks.append(concept_block)
            context_links.extend(concept_links)

        health = _compile_graph_health(store)
        context_blocks.append(health)

    # Index — always last, references all other context blocks
    index = _compile_index(context_blocks)
    context_blocks.append(index)

    # Insert context blocks and links
    store.insert_blocks(context_blocks)
    if context_links:
        store.insert_links(context_links)

    logger.info("Compiled %d context blocks, %d links", len(context_blocks), len(context_links))

    # Vault rendering
    rendered_files = 0
    if vault_path:
        rendered_files = _render_to_vault(context_blocks, vault_path)
        logger.info("Rendered %d files to vault", rendered_files)

    return {
        "context_blocks": len(context_blocks),
        "context_links": len(context_links),
        "rendered_files": rendered_files,
        "by_type": _count_by_type(context_blocks),
    }


# ── Layer 0: Hub Summaries ────────────────────────────────────────


def _compile_hub_summaries(store: SQLiteStore, top_n: int) -> list[Block]:
    """Create a context block for each top hub (tag)."""
    tags = store.get_tag_details(limit=top_n)
    blocks = []

    for tag in tags:
        if tag["entry_count"] == 0:
            continue

        co_tags = store.get_co_occurring_tags(tag["tag_id"], limit=5)
        co_tag_str = ", ".join(f"{ct['tag']} ({ct['count']})" for ct in co_tags)

        content = (
            f"# Hub: {tag['tag_name']}\n\n"
            f"- **Entries:** {tag['entry_count']}\n"
            f"- **Inbound links:** {tag['in_links']}\n"
            f"- **Outbound links:** {tag['out_links']}\n"
            f"- **Last active:** {tag['last_active'] or 'unknown'}\n"
            f"- **Hub score:** {tag['hub_score']:.2f}\n"
            f"- **Co-occurring tags:** {co_tag_str or 'none'}\n"
        )

        block = _make_context_block(
            context_type="hub_summary",
            scope=f"tag:{tag['tag_name']}",
            title=f"Hub: {tag['tag_name']}",
            content=content,
            compile_layer=0,
            extra_metadata={
                "tag_id": tag["tag_id"],
                "entry_count": tag["entry_count"],
                "hub_score": tag["hub_score"],
                "last_active": tag["last_active"],
            },
        )
        blocks.append(block)

    return blocks


# ── Layer 0: Recent Activity ─────────────────────────────────────


def _compile_recent_activity(store: SQLiteStore) -> Block:
    """Create a context block summarizing recent activity."""
    recent_7 = store.get_recent_blocks(days=7, kind="entry")
    recent_30 = store.get_recent_blocks(days=30, kind="entry")

    # Group by top tags
    tag_counts_7: dict[str, int] = {}
    for b in recent_7:
        for t in b.tags:
            tag_counts_7[t] = tag_counts_7.get(t, 0) + 1

    tag_counts_30: dict[str, int] = {}
    for b in recent_30:
        for t in b.tags:
            tag_counts_30[t] = tag_counts_30.get(t, 0) + 1

    # Group by source file
    source_counts_7: dict[str, int] = {}
    for b in recent_7:
        src = b.metadata.get("source_path", "unknown")
        source_counts_7[src] = source_counts_7.get(src, 0) + 1

    top_tags_7 = sorted(tag_counts_7.items(), key=lambda x: -x[1])[:10]
    top_tags_30 = sorted(tag_counts_30.items(), key=lambda x: -x[1])[:10]
    top_sources = sorted(source_counts_7.items(), key=lambda x: -x[1])[:10]

    content = "# Recent Activity\n\n"
    content += "## Last 7 Days\n\n"
    content += f"- **Entries:** {len(recent_7)}\n"
    if top_tags_7:
        content += f"- **Top tags:** {', '.join(f'{t} ({c})' for t, c in top_tags_7)}\n"
    if top_sources:
        content += f"- **Active files:** {', '.join(f'{s} ({c})' for s, c in top_sources)}\n"

    content += "\n## Last 30 Days\n\n"
    content += f"- **Entries:** {len(recent_30)}\n"
    if top_tags_30:
        content += f"- **Top tags:** {', '.join(f'{t} ({c})' for t, c in top_tags_30)}\n"

    return _make_context_block(
        context_type="recent_activity",
        scope="global",
        title="Recent Activity",
        content=content,
        compile_layer=0,
        extra_metadata={
            "entries_7d": len(recent_7),
            "entries_30d": len(recent_30),
        },
    )


# ── Layer 1: Concept Pages ───────────────────────────────────────


def _compile_concept_pages(store: SQLiteStore, top_n: int) -> list[tuple[Block, list[Link]]]:
    """Create concept context blocks with entry listings and links."""
    tags = store.get_tag_details(limit=top_n)
    results = []

    for tag in tags:
        if tag["entry_count"] == 0:
            continue

        entries = store.get_entries_for_tag(tag["tag_id"], limit=50)
        co_tags = store.get_co_occurring_tags(tag["tag_id"], limit=10)

        # Build entry listing
        entry_lines = []
        for e in entries:
            first_line = ""
            if e.content:
                # Get first non-empty, non-header line
                for line in e.content.split("\n"):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        first_line = stripped[:120]
                        break
            ts = e.timestamp or "no date"
            entry_lines.append(f"- [{ts}] {first_line}")

        content = f"# {tag['tag_name']}\n\n"
        content += f"**{tag['entry_count']} entries** | "
        content += f"Last active: {tag['last_active'] or 'unknown'}\n\n"

        if co_tags:
            content += "## Related Tags\n\n"
            for ct in co_tags:
                content += f"- {ct['tag']} ({ct['count']} shared entries)\n"
            content += "\n"

        content += "## Entries\n\n"
        content += "\n".join(entry_lines) if entry_lines else "No entries found.\n"

        block = _make_context_block(
            context_type="concept",
            scope=f"tag:{tag['tag_name']}",
            title=f"Concept: {tag['tag_name']}",
            content=content,
            compile_layer=1,
            extra_metadata={
                "tag_id": tag["tag_id"],
                "entry_count": tag["entry_count"],
            },
        )

        # Links from concept block to source entries
        links = [Link(from_id=block.id, to_id=e.id, kind="summarizes") for e in entries]

        results.append((block, links))

    return results


# ── Layer 1: Graph Health ────────────────────────────────────────


def _compile_graph_health(store: SQLiteStore) -> Block:
    """Create a context block with graph health metrics."""
    stats = store.get_stats()
    orphans = store.get_orphan_block_ids(kind="entry")
    stale = store.get_stale_tags(weeks=4)

    content = "# Graph Health\n\n"
    content += "## Overview\n\n"
    for kind, count in sorted(stats["blocks_by_kind"].items(), key=lambda x: -x[1]):
        if kind == "context":
            continue  # Don't count context blocks in health
        content += f"- **{kind}:** {count}\n"
    content += f"- **Total links:** {stats['total_links']}\n"
    content += f"- **Embedded:** {stats['embedded_blocks']}\n"

    content += "\n## Orphan Entries\n\n"
    content += f"**{len(orphans)} entries** with no tag or wikilink connections.\n"
    if orphans:
        content += "IDs: " + ", ".join(orphans[:20])
        if len(orphans) > 20:
            content += f" ... and {len(orphans) - 20} more"
        content += "\n"

    content += "\n## Stale Tags\n\n"
    content += f"**{len(stale)} tags** with no activity in 4+ weeks.\n"
    for s in stale[:20]:
        content += f"- {s['tag_name']} ({s['entry_count']} entries, last: {s['last_active']})\n"

    return _make_context_block(
        context_type="graph_health",
        scope="global",
        title="Graph Health",
        content=content,
        compile_layer=1,
        extra_metadata={
            "orphan_count": len(orphans),
            "stale_tag_count": len(stale),
            "total_blocks": stats["total_blocks"],
            "total_links": stats["total_links"],
        },
    )


# ── Index ─────────────────────────────────────────────────────────


def _compile_index(context_blocks: list[Block]) -> Block:
    """Create the master index context block listing all other context blocks."""
    # Group by type
    by_type: dict[str, list[Block]] = {}
    for b in context_blocks:
        ct = b.metadata.get("context_type", "unknown")
        by_type.setdefault(ct, []).append(b)

    content = "# OpenAugi Index\n\n"
    content += "This is the navigational map of your knowledge graph. "
    content += (
        "Use this to find relevant context blocks, then drill into specific hubs or entries.\n\n"
    )

    # Recent activity (always first)
    if "recent_activity" in by_type:
        for b in by_type["recent_activity"]:
            entries_7d = b.metadata.get("entries_7d", 0)
            entries_30d = b.metadata.get("entries_30d", 0)
            content += "## Recent Activity\n\n"
            content += f"{entries_7d} entries in last 7 days, {entries_30d} in last 30 days.\n\n"

    # Hub summaries
    if "hub_summary" in by_type:
        hubs = sorted(
            by_type["hub_summary"],
            key=lambda b: b.metadata.get("hub_score", 0),
            reverse=True,
        )
        content += f"## Top Hubs ({len(hubs)})\n\n"
        for b in hubs:
            name = b.metadata.get("scope", "").replace("tag:", "")
            entries = b.metadata.get("entry_count", 0)
            last = b.metadata.get("last_active", "?")
            score = b.metadata.get("hub_score", 0)
            content += f"- **{name}** — {entries} entries, last active {last}, score {score:.1f}\n"
        content += "\n"

    # Concepts
    if "concept" in by_type:
        content += f"## Concept Pages ({len(by_type['concept'])})\n\n"
        content += "Detailed pages with entry listings for each hub.\n\n"

    # Graph health
    if "graph_health" in by_type:
        for b in by_type["graph_health"]:
            orphans = b.metadata.get("orphan_count", 0)
            stale = b.metadata.get("stale_tag_count", 0)
            content += "## Graph Health\n\n"
            content += f"{orphans} orphan entries, {stale} stale tags.\n"

    return _make_context_block(
        context_type="index",
        scope="global",
        title="OpenAugi Index",
        content=content,
        compile_layer=0,
        extra_metadata={
            "context_block_count": len(context_blocks) + 1,  # +1 for this index
            "types": {k: len(v) for k, v in by_type.items()},
        },
    )


# ── Vault Rendering ──────────────────────────────────────────────


def _render_to_vault(context_blocks: list[Block], vault_path: str | Path) -> int:
    """Render context blocks as .md files to OpenAugi/Compiled/ in vault."""
    compiled_dir = Path(vault_path) / "OpenAugi" / "Compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    # Clean old rendered files
    _clean_compiled_dir(compiled_dir)

    rendered = 0
    for block in context_blocks:
        ct = block.metadata.get("context_type", "unknown")

        if ct == "index":
            path = compiled_dir / "INDEX.md"
        elif ct == "recent_activity":
            path = compiled_dir / "RECENT.md"
        elif ct == "graph_health":
            path = compiled_dir / "HEALTH.md"
        elif ct == "hub_summary":
            name = _safe_filename(block.metadata.get("scope", "unknown").replace("tag:", ""))
            path = compiled_dir / "hubs" / f"{name}.md"
        elif ct == "concept":
            name = _safe_filename(block.metadata.get("scope", "unknown").replace("tag:", ""))
            path = compiled_dir / "concepts" / f"{name}.md"
        else:
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block.content or "", encoding="utf-8")
        rendered += 1

    return rendered


def _clean_compiled_dir(compiled_dir: Path) -> None:
    """Remove all .md files from the compiled directory."""
    if not compiled_dir.exists():
        return
    for md_file in compiled_dir.rglob("*.md"):
        md_file.unlink()
    # Remove empty subdirectories
    for subdir in sorted(compiled_dir.rglob("*"), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()


# ── Helpers ──────────────────────────────────────────────────────


def _make_context_block(
    context_type: str,
    scope: str,
    title: str,
    content: str,
    compile_layer: int,
    extra_metadata: dict | None = None,
) -> Block:
    """Create a context block with standard metadata."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    metadata = {
        "context_type": context_type,
        "scope": scope,
        "compile_layer": compile_layer,
        "compiled_at": now,
        **(extra_metadata or {}),
    }

    # Deterministic ID from context_type + scope
    block_id = hashlib.sha256(f"context:{context_type}:{scope}".encode()).hexdigest()[:16]

    return Block(
        id=block_id,
        kind="context",
        content=content,
        source=COMPILED_SOURCE,
        title=title,
        metadata=metadata,
        timestamp=now[:10],  # just the date portion
        created_at=now,
    )


def _count_by_type(blocks: list[Block]) -> dict[str, int]:
    """Count context blocks by context_type."""
    counts: dict[str, int] = {}
    for b in blocks:
        ct = b.metadata.get("context_type", "unknown")
        counts[ct] = counts.get(ct, 0) + 1
    return counts


def _safe_filename(name: str) -> str:
    """Convert a tag/hub name to a safe filename."""
    # Replace path separators and special chars
    safe = name.replace("/", "-").replace("\\", "-")
    safe = "".join(c for c in safe if c.isalnum() or c in "-_ ")
    return safe.strip() or "unnamed"
