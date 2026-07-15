"""ZZZ dispatch — writes task files for blocks with zzz instructions.

Post-ingest hook. After the watcher ingests changed files, this module
checks for blocks that carry `zzz_instructions` in their metadata and
writes a pending task file to `OpenAugi/Tasks/` for each one.

The task watcher (`agents/task_watcher.py`) picks up pending files and
launches Claude Code sessions in tmux. This module is the bridge between
passive ingest and active agent work.

No LLM calls. No classification. Just deterministic file creation.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from openaugi.model.block import Block

logger = logging.getLogger(__name__)

DEFAULT_TASKS_FOLDER = "OpenAugi/Tasks"
DEFAULT_CAPTURE_FOLDER = "OpenAugi/Capture"  # mobile daily-note writer (server/dailyNote.ts)

# Obsidian block link into a capture daily note: [[YYYY-MM-DD#^augi-<id8>]].
# Written by mobile distill (curation.md) — provenance refs to gathered blocks.
ANCHOR_REF_RE = re.compile(r"\[\[(\d{4}-\d{2}-\d{2})#\^(augi-[A-Za-z0-9]+)\]\]")


def _slugify(text: str, max_len: int = 50) -> str:
    """Convert text to a filename-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-").lower()
    return slug[:max_len]


def _derive_title(block: Block) -> str:
    """Derive a short task title from block content or zzz instructions."""
    zzz = block.metadata.get("zzz_instructions", [])
    # Use first zzz instruction as title basis if it's descriptive enough
    if zzz and len(zzz[0]) > 5:
        return zzz[0][:80]
    # Fall back to first line of content
    content = (block.content or "").strip()
    first_line = content.split("\n")[0].strip()
    return first_line[:80] if first_line else "untitled task"


def _extract_anchor_entry(text: str, anchor: str) -> str | None:
    """The daily-note entry ending at `^<anchor>`, without the anchor line.

    Mirrors the mobile writer's entry shape (server/dailyNote.ts): an entry
    spans from the previous blank line to its anchor line. Returns None if
    the anchor isn't in the text.
    """
    lines = text.split("\n")
    anchor_line = f"^{anchor}"
    for i, line in enumerate(lines):
        if line.strip() == anchor_line:
            start = i
            while start > 0 and lines[start - 1].strip() != "":
                start -= 1
            entry = "\n".join(lines[start:i]).strip()
            return entry or None
    return None


def resolve_anchor_refs(
    content: str,
    vault_path: str | Path,
    capture_folder: str = DEFAULT_CAPTURE_FOLDER,
) -> list[tuple[str, str | None]]:
    """Resolve [[YYYY-MM-DD#^augi-<id8>]] refs to their block content.

    Looks in `<vault>/<capture_folder>/<date>.md` first (where the mobile
    writer puts them), then falls back to any `<date>.md` in the vault —
    Obsidian links resolve by basename, so the ref may point elsewhere.

    Returns [(ref_text, entry_or_None), ...] — unique refs in order of first
    appearance; None marks a dangling ref (kept so the reader sees the gap).
    """
    vault = Path(vault_path)
    file_cache: dict[Path, str | None] = {}

    def _read(path: Path) -> str | None:
        if path not in file_cache:
            try:
                file_cache[path] = path.read_text(encoding="utf-8") if path.is_file() else None
            except OSError:
                file_cache[path] = None
        return file_cache[path]

    resolved: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for m in ANCHOR_REF_RE.finditer(content):
        ref, date, anchor = m.group(0), m.group(1), m.group(2)
        if ref in seen:
            continue
        seen.add(ref)

        entry: str | None = None
        primary = vault / capture_folder / f"{date}.md"
        text = _read(primary)
        if text is not None:
            entry = _extract_anchor_entry(text, anchor)
        if entry is None:
            for candidate in sorted(vault.rglob(f"{date}.md")):
                if candidate == primary:
                    continue
                if any(p.startswith(".") for p in candidate.relative_to(vault).parts):
                    continue
                text = _read(candidate)
                if text is not None:
                    entry = _extract_anchor_entry(text, anchor)
                    if entry is not None:
                        break
        if entry is None:
            logger.warning("Dangling anchor ref in zzz block: %s", ref)
        resolved.append((ref, entry))
    return resolved


def build_task_file(block: Block, vault_path: str | Path | None = None) -> str:
    """Build a pending task file from a block with zzz instructions.

    Follows the task file contract in templates/task-template.md.
    The zzz instructions become both the user instruction and the task
    body — the agent in the tmux session interprets them.

    When vault_path is given, [[YYYY-MM-DD#^augi-<id8>]] anchor refs in the
    block (mobile distill-with-lens) are resolved and their content inlined
    into `## Context` — the gathered blocks become the context the lens runs
    over, the same contract as M3b "Distill selection".
    """
    zzz_list: list[str] = block.metadata.get("zzz_instructions", [])
    source_path = block.metadata.get("source_path", "")
    source_title = Path(source_path).stem if source_path else "unknown"
    content = (block.content or "").strip()
    title = _derive_title(block)

    if vault_path is not None:
        sections = []
        for ref, entry in resolve_anchor_refs(content, vault_path):
            sections.append(f"{ref}:\n{entry}" if entry is not None else f"{ref}: (unresolved)")
        if sections:
            content += "\n\n### Referenced blocks\n\n" + "\n\n".join(sections)

    # Build frontmatter
    fm_lines = [
        "---",
        "status: pending",
        f"source_block_id: {block.id}",
        f'source_note: "[[{source_title}]]"',
        "---",
    ]

    # Build body
    zzz_text = "\n".join(f"> {z}" for z in zzz_list)
    body = f"""
# {title}

## Context

{content}

## User instruction

{zzz_text}

## Task

Process the user instruction(s) above in the context of the source block.
The instruction text is the user's own words — interpret and execute accordingly.

## Human Todo

## Results
"""

    return "\n".join(fm_lines) + body


def dispatch_zzz_blocks(
    blocks: list[Block],
    vault_path: str | Path,
    tasks_folder: str = DEFAULT_TASKS_FOLDER,
) -> list[Path]:
    """Write task files for blocks that have zzz instructions.

    Args:
        blocks: Newly ingested blocks (from run_layer0).
        vault_path: Path to the vault root.
        tasks_folder: Relative path to the tasks folder in the vault.

    Returns:
        List of task file paths written.
    """
    vault = Path(vault_path)
    tasks_dir = vault / tasks_folder
    tasks_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for block in blocks:
        if block.kind != "data_block":
            continue
        zzz = block.metadata.get("zzz_instructions")
        if not zzz:
            continue

        title = _derive_title(block)
        slug = _slugify(title) or "zzz-task"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{slug}-{timestamp}.md"
        filepath = tasks_dir / filename

        task_content = build_task_file(block, vault_path=vault)
        filepath.write_text(task_content, encoding="utf-8")
        logger.info("Dispatched zzz task: %s → %s", block.id[:12], filepath.name)
        written.append(filepath)

    if written:
        logger.info("Dispatched %d zzz task(s)", len(written))

    return written
