"""Proactive echo — "you thought this before", written to the daily Augi Log.

Post-ingest hook, sibling of `dispatch.py`. For each newly ingested daily-note
block it retrieves older related blocks, asks the LLM whether any of them would
genuinely help the thought being written right now, and appends an echo to
`OpenAugi/YYYY/MM/DD/Augi Log.md`.

Design: OpenAugi/2026/08/29 - Design - Proactive Echo (Watcher + Daily Augi Log).

Three properties are load-bearing and should survive refactors:

1. **Silence is the default.** Most blocks produce nothing. The log is
   read-optional — if it is never opened, nothing breaks and no queue builds.
2. **The intent/judgment step is the gate, not the score.** A 7-day replay
   (2026-08-29) found real similarity scores cluster 0.53-0.68 whatever the
   block is — a family journal entry outscored an architecture note. Salience
   is only a noise floor; judgment decides.
3. **Every echo is idempotent.** An `<!-- echo:<block_id> -->` marker means a
   block is never echoed twice, however often the file is re-ingested.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from openaugi.model.block import Block

logger = logging.getLogger(__name__)

DAILY_PREFIX = "_private/0-Fleeting-Inbox/"
LOG_FOLDER_FMT = "OpenAugi/{y}/{m:02d}/{d:02d}"
LOG_NAME = "Augi Log.md"

# Deterministic filters (the replay's two mandatory additions)
MIN_PROSE_CHARS = 40
_WIKILINK_RE = re.compile(r"\[\[[^\]]*\]\]")
_MD_NOISE_RE = re.compile(r"[#>*_`\-\[\]()]|https?://\S+")
_INSTRUCTION_RE = re.compile(r"^\s*zzz\s*:", re.IGNORECASE | re.MULTILINE)

JUDGE_SYSTEM = """You decide whether a personal-knowledge agent should speak.

The user is writing in their daily note. You get the block they just wrote and
older blocks from their own vault. Surface an older block ONLY when it would
change what they do next — it shows they already worked this out, already named
this idea, or already hit this pattern.

SPEAK when the new block is a thought, idea, plan, or question about their work,
and an older block adds something they would want to re-open:
  new: "look at my autowiki structure... gold level... bronze are the default"
  old: "the bronze silver gold medallion architecture" (6 weeks earlier)
  -> echo. They are re-deriving something they already named.

STAY SILENT when:
- the block logs life: kids, meals, workouts, chores, moods, family, errands.
  Retrieving another day where a kid also played on a laptop helps nobody.
- the block is an instruction, a task, or a bare link.
- the older blocks are merely on a similar topic and add no new information.
- you would be restating what the new block already says.

Everything in this vault is written by one person, so almost everything looks
similar. Topical similarity is NEVER a reason to speak. Only usefulness is.
When genuinely unsure, stay silent — a missed echo costs nothing, a noisy one
costs trust.

Return ONLY JSON:
{"echo": true|false,
 "lines": [{"title": "<exact older block title>", "why": "<one clause, max 15 words>"}]}
At most 3 lines. Prefer 1 excellent line over 3 weak ones. The "why" says what
the older block contributes, never a summary of it."""


def _prose_len(content: str) -> int:
    """Length of real prose, ignoring wikilinks and markdown punctuation."""
    stripped = _WIKILINK_RE.sub("", content or "")
    return len(_MD_NOISE_RE.sub("", stripped).strip())


def is_echo_eligible(block: Block) -> bool:
    """Deterministic pre-filter — cheap checks before any retrieval or LLM."""
    path = (block.metadata or {}).get("source_path") or ""
    if not path.startswith(DAILY_PREFIX):
        return False
    content = block.content or ""
    if _prose_len(content) < MIN_PROSE_CHARS:
        return False  # link-only / stub blocks (a bare [[link]] scored 1.000 in replay)
    # dispatch owns zzz blocks; they are asks, not thoughts
    return not ((block.metadata or {}).get("zzz_instructions") or _INSTRUCTION_RE.search(content))


def _candidates(block: Block, store, model, config: dict[str, Any], k: int = 6) -> list[Block]:
    """Older, non-derived blocks related to this one."""
    from openaugi.query import engine

    day = (block.block_time or "")[:10]
    result = engine.context(
        store,
        (block.content or "")[:600],
        k=k * 3,
        expand=False,
        purpose="resurface",
        embedding_model=model,
        config=config,
    )
    out: list[Block] = []
    for entry in result.seen:
        cand = entry.block
        cpath = (cand.metadata or {}).get("source_path") or ""
        if cand.id == block.id:
            continue
        if day and (cand.block_time or "")[:10] >= day:
            continue  # same-day or newer is not an echo
        if cpath.startswith("OpenAugi/"):
            continue  # never echo our own artifacts back
        if _prose_len(cand.content or "") < MIN_PROSE_CHARS:
            continue
        out.append(cand)
        if len(out) >= k:
            break
    return out


def _judge(block: Block, candidates: list[Block], llm) -> list[dict]:
    """Ask the LLM which candidates are worth surfacing. [] means stay silent."""
    listing = "\n\n".join(
        f"[{i + 1}] {c.title} ({(c.block_time or '')[:10]})\n{(c.content or '')[:400]}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"NEW BLOCK (being written now):\n{(block.content or '')[:1200]}\n\n"
        f"OLDER BLOCKS FROM THEIR VAULT:\n{listing}"
    )
    try:
        raw = llm.complete(prompt, system=JUDGE_SYSTEM).strip()
    except Exception as e:
        logger.warning(f"Echo judge failed: {e}")
        return []
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not verdict.get("echo"):
        return []
    titles = {c.title for c in candidates if c.title}
    return [
        line
        for line in (verdict.get("lines") or [])[:3]
        if isinstance(line, dict) and line.get("title") in titles
    ]


def _log_path(vault_path: Path, day: str) -> Path:
    y, m, d = int(day[:4]), int(day[5:7]), int(day[8:10])
    return vault_path / LOG_FOLDER_FMT.format(y=y, m=m, d=d) / LOG_NAME


def _ensure_log(path: Path, day: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "---\ntype: document\n"
        f"description: Proactive echoes augi surfaced while you wrote on {day}. "
        "Ephemeral — delete freely, nothing depends on it.\n"
        f"created: {day}\n---\n\n"
        f"# Augi Log — {day}\n\n#human-review\n\n"
        "*Appended as you write. Read when you want; ignoring it costs nothing.*\n"
    )
    path.write_text(header, encoding="utf-8")
    return header


def _render(block: Block, lines: list[dict], by_title: dict[str, Block]) -> str:
    snippet = " ".join((block.content or "").split())[:70]
    out = [f"\n<!-- echo:{block.id} -->", f'\n### echo on "{snippet}…"']
    src = (block.metadata or {}).get("source_path", "")
    if src:
        out.append(f"*While writing in [[{Path(src).stem}]]:*\n")
    for line in lines:
        cand = by_title.get(line["title"])
        when = (cand.block_time or "")[:10] if cand else ""
        out.append(f"- [[{line['title']}]]{f' ({when})' if when else ''} — {line['why']}")
    out.append("")
    out.append("- [ ] promote → new note")
    out.append("- [ ] good match")
    out.append("- [ ] bad match")
    out.append("")
    return "\n".join(out)


def run_echo(
    new_blocks: list[Block],
    vault_path: Path,
    store,
    model,
    config: dict[str, Any],
) -> dict[str, int]:
    """Echo pass over newly ingested blocks. Returns {watched, spoke, quiet}."""
    from openaugi.models import get_llm_model

    llm = get_llm_model(config.get("models", {}).get("llm"))
    if llm is None:
        logger.info("Proactive echo skipped — no [models.llm] configured")
        return {"watched": 0, "spoke": 0, "quiet": 0}

    eligible = [b for b in new_blocks if is_echo_eligible(b)]
    stats = {"watched": len(eligible), "spoke": 0, "quiet": 0}

    for block in eligible:
        day = (block.block_time or "")[:10] or date.today().isoformat()
        path = _log_path(vault_path, day)
        existing = _ensure_log(path, day)
        if f"<!-- echo:{block.id} -->" in existing:
            continue  # already spoken about this block
        candidates = _candidates(block, store, model, config)
        lines = _judge(block, candidates, llm) if candidates else []
        if not lines:
            stats["quiet"] += 1
            continue
        by_title = {c.title: c for c in candidates if c.title}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(_render(block, lines, by_title))
        stats["spoke"] += 1

    if stats["watched"]:
        logger.info(
            f"Echo pass: watched {stats['watched']} · "
            f"spoke {stats['spoke']} · quiet {stats['quiet']}"
        )
        _write_heartbeat(vault_path, stats)
    return stats


def _write_heartbeat(vault_path: Path, stats: dict[str, int]) -> None:
    """Keep a running per-day tally at the end of the log.

    Silence has to be legible: the pain-points sweep found that a system which
    never reports its own quiet is indistinguishable from one that is broken.
    """
    day = date.today().isoformat()
    path = _log_path(vault_path, day)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    prior = re.search(r"<!-- heartbeat (\d+) (\d+) (\d+) -->", text)
    watched, spoke, quiet = stats["watched"], stats["spoke"], stats["quiet"]
    if prior:
        watched += int(prior.group(1))
        spoke += int(prior.group(2))
        quiet += int(prior.group(3))
        text = re.sub(r"\n?<!-- heartbeat .*?-->\n.*?\n?$", "\n", text, flags=re.DOTALL)
    text = text.rstrip("\n") + (
        f"\n\n<!-- heartbeat {watched} {spoke} {quiet} -->\n"
        f"*watched {watched} · spoke {spoke} · quiet {quiet}*\n"
    )
    path.write_text(text, encoding="utf-8")
