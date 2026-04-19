"""Heartbeat — dumb script, smart agent.

Finds new blocks since the last heartbeat run, builds a prompt describing
them (with any per-block `zzz:` instructions), and spawns a Claude Code
session with openaugi MCP tools to process them.

The Python side is deliberately dumb: it does not classify, it does not
decide what goes where. It only:
  1. Reads the last heartbeat timestamp from ~/.openaugi/last_heartbeat
  2. Queries data_block blocks created since that timestamp
  3. Builds a prompt listing blocks + zzz instructions + skill file ref
  4. Shells out to `claude -p <prompt> --allowedTools ...`
  5. Updates the timestamp on success

The agent does the reasoning. The skill file at
`<vault>/OpenAugi/heartbeat-skill.md` contains the rules. Users iterate on
the skill file, not the Python. See docs/plans/heartbeat.md.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from openaugi.model.block import Block
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

HEARTBEAT_STATE_FILE = Path.home() / ".openaugi" / "last_heartbeat"
SKILL_FILE_RELATIVE = "OpenAugi/heartbeat-skill.md"
HEARTBEAT_LOG_RELATIVE = "OpenAugi/Heartbeat"
DEFAULT_MAX_BLOCKS = 50


def get_last_heartbeat() -> str | None:
    """Read the ISO timestamp of the last successful heartbeat run."""
    if not HEARTBEAT_STATE_FILE.exists():
        return None
    try:
        return HEARTBEAT_STATE_FILE.read_text().strip() or None
    except OSError as e:
        logger.warning("Failed to read %s: %s", HEARTBEAT_STATE_FILE, e)
        return None


def set_last_heartbeat(timestamp: str) -> None:
    """Persist the heartbeat timestamp for the next run."""
    HEARTBEAT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_STATE_FILE.write_text(timestamp + "\n")


def find_new_blocks(
    store: SQLiteStore,
    since: str | None,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    ignore_sources: list[str] | None = None,
    ignore_headings: list[str] | None = None,
) -> list[Block]:
    """Fetch data_block blocks created since the last heartbeat, oldest first.

    Capped at `max_blocks` to keep prompts bounded. If more exist, the
    caller should surface that so the user can rerun or raise the cap.

    `ignore_sources`: fnmatch glob patterns on `source_path` (file path).
    `ignore_headings`: exact heading texts to exclude (case-insensitive),
      matched against `section_heading` metadata — e.g. ["HW", "private"].
    """
    import fnmatch

    blocks = store.get_blocks_since(since=since, kind="data_block", limit=max_blocks)
    if ignore_sources:
        blocks = [
            b
            for b in blocks
            if not any(
                fnmatch.fnmatch(b.metadata.get("source_path", ""), pat) for pat in ignore_sources
            )
        ]
    if ignore_headings:
        ignore_lower = {h.lower() for h in ignore_headings}
        blocks = [
            b
            for b in blocks
            if (b.metadata.get("section_heading") or "").lower() not in ignore_lower
        ]
    return blocks


def build_prompt(
    blocks: list[Block],
    skill_file: Path,
    heartbeat_log: Path,
    since: str | None,
    now: str,
) -> str:
    """Build the prompt handed to the Claude Code agent session.

    The prompt is intentionally small: it points the agent at the skill
    file, lists the new blocks with their per-block zzz instructions, and
    tells it where to write the heartbeat log. Everything else — workstream
    rules, defaults, task handling — lives in the skill file the user
    maintains.
    """
    header = f"""You are the OpenAugi heartbeat agent.

Read your skill file first:
  {skill_file}

It contains the area taxonomy (area/type/status facets), default
classification rules, task/research handling, and what to write back.
Follow it.

## Context

- Window: blocks created {"since " + since if since else "on first run"} through {now}
- Blocks in this batch: {len(blocks)}
- You have access to these openaugi MCP tools for lookups:
    - mcp__openaugi__search            — keyword + semantic search
    - mcp__openaugi__get_context       — FTS + semantic with dedup/MMR
    - mcp__openaugi__get_block / get_blocks — fetch full content by id
    - mcp__openaugi__get_related       — follow links from/to a block
    - mcp__openaugi__traverse          — multi-hop graph walk
    - mcp__openaugi__recent            — recent blocks
    - mcp__openaugi__tag_block         — stamp area/type/status classification onto a block
    - mcp__openaugi__list_streams / get_stream_context — workstream CRUD

Use them to chain decisions: if block 1 surfaces a connection, let that
inform how you process block 2. Notice related blocks in this batch and
handle them together when it helps.

## Per-block `zzz:` instructions

Each block below may have a `User instruction:` line. That is the user's
per-block directive written inline. Honor it. If there is no instruction,
fall back to the defaults in the skill file.

## Blocks to process
"""

    block_sections: list[str] = []
    for i, block in enumerate(blocks, start=1):
        source_path = block.metadata.get("source_path", "(unknown)")
        zzz_list = block.metadata.get("zzz_instructions") or []
        tags = ", ".join(block.tags) if block.tags else "(none)"
        content = (block.content or "").strip()
        preview = content if len(content) <= 800 else content[:800] + "…"
        section = (
            f"\n### Block {i} — id {block.id}\n"
            f"- **Source:** {source_path}\n"
            f"- **Tags:** {tags}\n"
            f"- **Timestamp:** {block.block_time or '(none)'}\n"
        )
        if zzz_list:
            section += "- **User instructions:**\n"
            for instruction in zzz_list:
                section += f"  - {instruction}\n"
        section += f"\n{preview}\n"
        block_sections.append(section)

    footer = f"""

## What to write back

Write a heartbeat log to:
  {heartbeat_log}

Summarize what you processed, what you classified, what connections you
found, what actions you took (tasks created, research written), and
anything you flagged for human review. Use the skill file's "What to write
back" section as the source of truth for format.

When the log is written, the session is done. Do not modify the user's raw
notes — everything system-generated lives under OpenAugi/ in the vault.
"""

    return header + "".join(block_sections) + footer


def launch_agent(prompt: str) -> int:
    """Spawn the claude CLI with the heartbeat prompt.

    Returns the subprocess return code. Raises FileNotFoundError if the
    `claude` binary is not on PATH — the CLI caller turns that into a
    helpful error for the user.
    """
    from openaugi.agents.task_watcher import detect_claude

    try:
        claude_bin = detect_claude()
    except FileNotFoundError as err:
        raise FileNotFoundError(
            "The `claude` CLI was not found. Install Claude Code "
            "(https://claude.com/claude-code) or run the prompt manually."
        ) from err

    allowed_tools = ",".join(
        [
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "mcp__openaugi__search",
            "mcp__openaugi__get_context",
            "mcp__openaugi__get_block",
            "mcp__openaugi__get_blocks",
            "mcp__openaugi__get_related",
            "mcp__openaugi__traverse",
            "mcp__openaugi__recent",
            "mcp__openaugi__tag_block",
            "mcp__openaugi__list_streams",
            "mcp__openaugi__get_stream_context",
        ]
    )

    logger.info("Launching claude agent (allowed tools: %s)", allowed_tools)
    result = subprocess.run(
        [claude_bin, "-p", prompt, "--allowedTools", allowed_tools],
        check=False,
    )
    return result.returncode


def run_heartbeat(
    store: SQLiteStore,
    vault_path: str | Path,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    dry_run: bool = False,
    ignore_sources: list[str] | None = None,
    ignore_headings: list[str] | None = None,
    skill_file_path: str | None = None,
) -> dict:
    """Run one heartbeat cycle.

    Steps:
      1. Resolve the skill file path and fail loudly if missing.
      2. Read last heartbeat timestamp.
      3. Query data_block blocks created since that timestamp.
      4. Build a prompt and (unless dry_run) spawn the claude agent.
      5. On success, update the timestamp.

    Returns a dict with the prompt, block count, skill file path, log path,
    whether the agent was launched, and its return code.
    """
    vault = Path(vault_path)
    skill_file = Path(skill_file_path) if skill_file_path else vault / SKILL_FILE_RELATIVE
    if not skill_file.exists():
        raise FileNotFoundError(
            f"Heartbeat skill file not found: {skill_file}\n"
            "Create it with workstreams, default rules, and what to write "
            "back. Or set [heartbeat] skill_file in config.toml to an absolute path."
        )

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    since = get_last_heartbeat()
    blocks = find_new_blocks(
        store,
        since=since,
        max_blocks=max_blocks,
        ignore_sources=ignore_sources,
        ignore_headings=ignore_headings,
    )

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    heartbeat_log = vault / HEARTBEAT_LOG_RELATIVE / f"{today}.md"

    result: dict = {
        "since": since,
        "now": now,
        "skill_file": skill_file,
        "heartbeat_log": heartbeat_log,
        "block_count": len(blocks),
        "max_blocks": max_blocks,
        "batch_capped": len(blocks) >= max_blocks,
        "launched": False,
        "return_code": None,
        "prompt": "",
    }

    if not blocks:
        logger.info("Heartbeat: no new blocks since %s", since or "(first run)")
        # Still advance the timestamp so we don't re-query the same empty window.
        if not dry_run:
            set_last_heartbeat(now)
        return result

    prompt = build_prompt(
        blocks=blocks,
        skill_file=skill_file,
        heartbeat_log=heartbeat_log,
        since=since,
        now=now,
    )
    result["prompt"] = prompt

    if dry_run:
        logger.info("Dry run — skipping agent launch for %d blocks", len(blocks))
        return result

    # Ensure the heartbeat log directory exists so the agent can write.
    heartbeat_log.parent.mkdir(parents=True, exist_ok=True)

    return_code = launch_agent(prompt)
    result["launched"] = True
    result["return_code"] = return_code

    if return_code == 0:
        set_last_heartbeat(now)
        logger.info("Heartbeat complete. Advanced last_heartbeat to %s", now)
    else:
        logger.warning(
            "Agent returned %s — leaving last_heartbeat unchanged so next run retries",
            return_code,
        )

    return result
