"""ZZZ dispatch — writes task files for blocks with zzz instructions.

Post-ingest hook. After the watcher ingests changed files, this module
checks for blocks that carry `zzz_instructions` in their metadata and
writes a pending task file to `OpenAugi/Tasks/` for each one.

The task watcher (`agents/task_watcher.py`) picks up pending files and
launches Claude Code sessions in tmux. This module is the bridge between
passive ingest and active agent work.

No LLM calls. No classification. Just deterministic file creation.

## Why dispatch is queued rather than immediate

A block's identity is the hash of its raw text *including* the `zzz` line
(adapters/splitter.py). So finishing a half-written instruction is not an
update — it deletes one block and inserts another, and a hook that fires on
"new block with a zzz" fires twice for one instruction. That is exactly what
happened on 2026-09-01: `read this voice` dispatched at 20:41 while the
sentence was still being typed, and the finished sentence dispatched again at
20:52, giving two agents the same voice note with different instructions.

Two mechanisms fix it, and both are needed because they cover different gaps:

1. **Settle window.** A zzz block is queued, not dispatched. It only becomes a
   task once it has survived `settle` seconds unchanged. This absorbs typing
   and dictation pauses, where the discarded draft never becomes a task at all.
2. **Supersession.** Past the settle window the draft has already launched, so
   waiting cannot help. `run_layer0` reports the entries it deleted this cycle,
   which is the one place a block's predecessor is still visible: a document
   that drops a queued/dispatched zzz block and adds a new one in the same
   cycle has *edited* an instruction, not written a second one. The successor
   supersedes it — the old task is marked and its tmux session killed, so the
   instruction you finished writing is the one that runs.

3. **Carry-forward.** Supersession alone still re-dispatches, because a
   successor block is a new block. But an edit to the *prose* around a `zzz`
   line — finishing the thought the instruction is about — rewrites the block
   without touching the instruction. When the successor's zzz text is
   byte-identical to the predecessor's and that predecessor already
   dispatched, the successor inherits its ledger row and task file instead of
   getting its own. That is what makes dispatch idempotent per source block:
   one instruction, one task, however many times the paragraph is edited.
   (On 2026-09-09 one research `zzz` dispatched three times this way.) A
   *changed* instruction is a real edit and still supersedes.

The ledger lives in the `records` table under the `zzz_queue` collection
(docs/reference/records.md) — workflow state, droppable, pruned once settled.
It also makes dispatch idempotent across restarts and re-ingests: a block id
that has been dispatched once never dispatches again.
"""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from openaugi.model.block import Block

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

DEFAULT_TASKS_FOLDER = "OpenAugi/Tasks"

# Ledger of every zzz instruction this pipeline has seen, keyed by block id.
ZZZ_QUEUE_COLLECTION = "zzz_queue"
# Seconds a zzz block must sit unchanged before it becomes a task. Long enough
# to cover a dictation pause mid-sentence, short enough that a deliberate zzz
# still feels immediate.
DEFAULT_ZZZ_SETTLE = 120.0
# Settled ledger rows are kept this long for auditing, then pruned.
LEDGER_RETENTION_DAYS = 30

# Ledger statuses.
QUEUED = "queued"  # seen, waiting out the settle window
DISPATCHED = "dispatched"  # task file written
SUPERSEDED = "superseded"  # the instruction was edited or deleted before/after dispatch
DEFAULT_CAPTURE_FOLDER = "OpenAugi/Capture"  # mobile daily-note writer (server/dailyNote.ts)

# Obsidian block link into a capture daily note: [[YYYY-MM-DD#^augi-<id8>]].
# Written by mobile distill (curation.md) — provenance refs to gathered blocks.
ANCHOR_REF_RE = re.compile(r"\[\[(\d{4}-\d{2}-\d{2})#\^(augi-[A-Za-z0-9]+)\]\]")


def _instructions_of(block: Block) -> list[str]:
    """The block's zzz instructions, normalised for comparison."""
    return [str(z).strip() for z in block.metadata.get("zzz_instructions", [])]


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

    for block in zzz_blocks(blocks):
        task_content = build_task_file(block, vault_path=vault)
        filepath = _write_task_file(tasks_dir, _derive_title(block), task_content)
        logger.info("Dispatched zzz task: %s → %s", block.id[:12], filepath.name)
        written.append(filepath)

    if written:
        logger.info("Dispatched %d zzz task(s)", len(written))

    return written


def zzz_blocks(blocks: list[Block]) -> list[Block]:
    """The data blocks in `blocks` that carry zzz instructions, in order."""
    return [b for b in blocks if b.kind == "data_block" and b.metadata.get("zzz_instructions")]


def _write_task_file(tasks_dir: Path, title: str, task_content: str) -> Path:
    """Write one pending task file, named from its title plus a timestamp."""
    slug = _slugify(title) or "zzz-task"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = tasks_dir / f"{slug}-{timestamp}.md"
    filepath.write_text(task_content, encoding="utf-8")
    return filepath


# ── The zzz queue ──────────────────────────────────────────────────────────
#
# See the module docstring for why dispatch is queued instead of immediate.


def _now() -> datetime:
    return datetime.now()


def _load_ledger(store: SQLiteStore) -> dict[str, dict]:
    """Every zzz ledger row, keyed by block id."""
    rows = store.list_records(ZZZ_QUEUE_COLLECTION, limit=100_000)
    return {r["id"]: r for r in rows}


def record_zzz_changes(
    new_blocks: list[Block],
    removed_blocks: list[Block],
    store: SQLiteStore,
    vault_path: str | Path,
    tasks_folder: str = DEFAULT_TASKS_FOLDER,
) -> None:
    """Queue this cycle's new zzz instructions, superseding the ones they replace.

    `removed_blocks` are the entries `run_layer0` deleted in the same cycle.
    Within one document, a dropped zzz block paired with a fresh one is an
    *edit* of a single instruction — the pair is matched in document order, so
    the common case (one instruction, edited) matches exactly. Unpaired blocks
    on either side are genuinely new or genuinely deleted.
    """
    vault = Path(vault_path)
    tasks_dir = vault / tasks_folder
    ledger = _load_ledger(store)
    now = _now()
    stamp = now.isoformat(timespec="seconds")

    def _by_doc(blocks: list[Block]) -> dict[str, list[Block]]:
        out: dict[str, list[Block]] = {}
        for b in blocks:
            out.setdefault(b.metadata.get("source_path", ""), []).append(b)
        return out

    fresh = _by_doc(zzz_blocks(new_blocks))
    # A removed block was a zzz block iff we have a ledger row for it. That
    # avoids depending on metadata surviving the store round-trip, and it is
    # the same question we actually care about: did this block owe us a task?
    gone = _by_doc([b for b in removed_blocks if b.id in ledger])

    superseded: set[str] = set()
    for source_path, olds in gone.items():
        news = fresh.get(source_path, [])
        for old, new in zip(olds, news, strict=False):
            if _carry_forward(store, ledger, old.id, new, stamp):
                superseded.add(old.id)
                continue
            _supersede(store, ledger, tasks_dir, old.id, new.id, stamp)
            superseded.add(old.id)
        for old in olds[len(news) :]:
            # The zzz line was deleted outright, not rewritten.
            row = ledger[old.id]
            if row.get("status") == QUEUED:
                # Never became a task — drop it silently.
                store.update_record(
                    ZZZ_QUEUE_COLLECTION,
                    old.id,
                    {"status": SUPERSEDED, "reason": "deleted"},
                    stamp,
                )
                logger.info("zzz %s dropped before dispatch — instruction deleted", old.id[:12])
            else:
                # Already running. Deleting the line after the fact is not a
                # cancel signal, so the task is left alone.
                logger.info(
                    "zzz %s deleted after dispatch — leaving its task running", old.id[:12]
                )
            superseded.add(old.id)

    for source_path, news in fresh.items():
        for block in news:
            if block.id in ledger:
                continue  # already seen — never re-dispatch the same block
            store.write_record(
                ZZZ_QUEUE_COLLECTION,
                block.id,
                {
                    "status": QUEUED,
                    "source_path": source_path,
                    "title": _derive_title(block),
                    "instructions": _instructions_of(block),
                    "task_content": build_task_file(block, vault_path=vault),
                },
                stamp,
            )
            logger.info(
                "zzz %s queued from %s (settling)", block.id[:12], source_path or "unknown"
            )


def _carry_forward(
    store: SQLiteStore,
    ledger: dict[str, dict],
    old_id: str,
    new: Block,
    stamp: str,
) -> bool:
    """Inherit a dispatched row when only the prose around the zzz changed.

    A block's identity is the hash of its whole raw text, so appending a
    sentence to the paragraph that carries a `zzz` line deletes the block and
    inserts a new one — with the instruction byte-for-byte unchanged. Without
    this, every such edit looked like a brand-new instruction and dispatched
    the same task again (2026-09-09: one research zzz, three agents).

    So: same instruction, already dispatched → the successor inherits the row
    and the task file. Nothing new is written, the running session is left
    alone, and the chain stays intact for the next edit. A *changed*
    instruction is a real edit and falls through to supersession.

    Returns True when the row was carried forward.
    """
    row = ledger.get(old_id, {})
    if row.get("status") != DISPATCHED:
        return False
    previous = row.get("instructions")
    if not isinstance(previous, list) or [str(z) for z in previous] != _instructions_of(new):
        # Either the instruction really changed, or this row predates the
        # `instructions` field and we cannot tell — supersede, as before.
        return False

    store.update_record(
        ZZZ_QUEUE_COLLECTION,
        old_id,
        {"status": SUPERSEDED, "reason": "text-edited", "superseded_by": new.id},
        stamp,
    )
    carried = {
        **row,
        "status": DISPATCHED,
        "carried_from": old_id,
        "instructions": _instructions_of(new),
    }
    carried.pop("id", None)
    carried.pop("created_at", None)
    carried.pop("updated_at", None)
    store.write_record(ZZZ_QUEUE_COLLECTION, new.id, carried, stamp)
    ledger[new.id] = {"id": new.id, **carried}
    logger.info(
        "zzz %s re-hashed as %s with the same instruction — not dispatching again",
        old_id[:12],
        new.id[:12],
    )
    return True


def _supersede(
    store: SQLiteStore,
    ledger: dict[str, dict],
    tasks_dir: Path,
    old_id: str,
    new_id: str,
    stamp: str,
) -> None:
    """Retire the ledger row and task for `old_id`, replaced by `new_id`."""
    row = ledger.get(old_id, {})
    store.update_record(
        ZZZ_QUEUE_COLLECTION,
        old_id,
        {"status": SUPERSEDED, "reason": "edited", "superseded_by": new_id},
        stamp,
    )
    if row.get("status") != DISPATCHED:
        logger.info(
            "zzz %s superseded by %s before dispatch — no task was written",
            old_id[:12],
            new_id[:12],
        )
        return
    path = _find_task_file(tasks_dir, row.get("task_file"), old_id)
    if path is None:
        logger.warning("No task file found for superseded zzz %s", old_id[:12])
        return
    _retire_task_file(path, new_id)


def _find_task_file(tasks_dir: Path, filename: str | None, block_id: str) -> Path | None:
    """Locate a dispatched task file.

    The name it was written under is only a hint: the task watcher renames the
    file to `TASK-<id>.md` when it hydrates it. `source_block_id` is the field
    that survives, so that is what we match on when the name has moved.
    """
    from openaugi.agents.task_watcher import parse_note

    if filename:
        direct = tasks_dir / filename
        if direct.exists():
            return direct
    if not tasks_dir.is_dir():
        return None
    for candidate in sorted(tasks_dir.glob("*.md")):
        try:
            fm, _ = parse_note(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if fm.get("source_block_id") == block_id:
            return candidate
    return None


def _retire_task_file(path: Path, new_id: str) -> None:
    """Mark a launched task superseded and kill its tmux session.

    The Claude transcript stays on disk and the file records why it stopped —
    only the live session is torn down, so the instruction that replaced this
    one is the only one still running.
    """
    from openaugi.agents.task_watcher import detect_tmux, parse_note, rebuild_note

    if not path.exists():
        logger.warning("Superseded task file is gone: %s", path)
        return
    try:
        fm, body = parse_note(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Could not read task file %s: %s", path, e)
        return

    if fm.get("status") == "done":
        logger.info("Task %s already done — not retiring it", path.name)
        return

    session = fm.get("tmux_session")
    fm["status"] = "superseded"
    fm["superseded_by_block"] = new_id
    body = body.rstrip() + (
        f"\n\n> Superseded: the `zzz` instruction behind this task was edited "
        f"before it finished. Block `{new_id[:12]}` carries the final wording and "
        f"was dispatched as its own task.\n"
    )
    path.write_text(rebuild_note(fm, body), encoding="utf-8")

    if session:
        try:
            tmux = detect_tmux()
        except FileNotFoundError:
            return
        if (
            subprocess.run([tmux, "has-session", "-t", session], capture_output=True).returncode
            == 0
        ):
            subprocess.run([tmux, "kill-session", "-t", session], check=False)
            logger.info("Killed superseded task session: %s", session)


def drain_zzz_queue(
    store: SQLiteStore,
    vault_path: str | Path,
    settle_seconds: float = DEFAULT_ZZZ_SETTLE,
    tasks_folder: str = DEFAULT_TASKS_FOLDER,
) -> list[Path]:
    """Write task files for queued zzz blocks that have settled.

    Safe to call on a timer with nothing new to do — an instruction written
    just before the vault went quiet still matures without another file event.
    """
    vault = Path(vault_path)
    tasks_dir = vault / tasks_folder
    now = _now()
    stamp = now.isoformat(timespec="seconds")
    written: list[Path] = []

    for row in store.list_records(ZZZ_QUEUE_COLLECTION, where={"status": QUEUED}, limit=1000):
        block_id = row["id"]
        if store.get_block(block_id) is None:
            # Edited or deleted between cycles and we never saw the removal.
            store.update_record(
                ZZZ_QUEUE_COLLECTION, block_id, {"status": SUPERSEDED, "reason": "vanished"}, stamp
            )
            logger.info("zzz %s left the vault before settling — dropped", block_id[:12])
            continue
        if (now - _parse_stamp(row["created_at"])).total_seconds() < settle_seconds:
            continue

        tasks_dir.mkdir(parents=True, exist_ok=True)
        filepath = _write_task_file(tasks_dir, row.get("title") or "zzz task", row["task_content"])
        store.update_record(
            ZZZ_QUEUE_COLLECTION,
            block_id,
            {"status": DISPATCHED, "task_file": filepath.name},
            stamp,
        )
        logger.info("Dispatched zzz task: %s → %s", block_id[:12], filepath.name)
        written.append(filepath)

    if written:
        logger.info("Dispatched %d zzz task(s)", len(written))
    _prune_ledger(store, now)
    return written


def _parse_stamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        # Unreadable timestamp — treat as ancient so the row settles rather
        # than sticking in the queue forever.
        return datetime.min


def _prune_ledger(store: SQLiteStore, now: datetime) -> int:
    """Drop settled ledger rows past the retention window."""
    cutoff = (now - timedelta(days=LEDGER_RETENTION_DAYS)).isoformat(timespec="seconds")
    dropped = 0
    for status in (DISPATCHED, SUPERSEDED):
        for row in store.list_records(
            ZZZ_QUEUE_COLLECTION, where={"status": status}, limit=100_000
        ):
            if row["updated_at"] < cutoff:
                dropped += store.delete_record(ZZZ_QUEUE_COLLECTION, row["id"])
    if dropped:
        logger.debug("Pruned %d settled zzz ledger row(s)", dropped)
    return dropped
