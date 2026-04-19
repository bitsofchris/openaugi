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


def build_task_file(block: Block) -> str:
    """Build a pending task file from a block with zzz instructions.

    Follows the task file contract in templates/task-template.md.
    The zzz instructions become both the user instruction and the task
    body — the agent in the tmux session interprets them.
    """
    zzz_list: list[str] = block.metadata.get("zzz_instructions", [])
    source_path = block.metadata.get("source_path", "")
    source_title = Path(source_path).stem if source_path else "unknown"
    content = (block.content or "").strip()
    title = _derive_title(block)

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

        task_content = build_task_file(block)
        filepath.write_text(task_content, encoding="utf-8")
        logger.info("Dispatched zzz task: %s → %s", block.id[:12], filepath.name)
        written.append(filepath)

    if written:
        logger.info("Dispatched %d zzz task(s)", len(written))

    return written
