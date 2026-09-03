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
from openaugi.pipeline import augi_log
from openaugi.pipeline.augi_log import (  # noqa: F401 — re-exported for callers and tests
    DAILY_PREFIX,
    MIN_PROSE_CHARS,
    QUIET_HEADING,
)

logger = logging.getLogger(__name__)

# A block reached by walking one link from a hit inherits its parent's score,
# discounted. Graph proximity is real evidence — it is how "the note this
# belongs to" surfaces — but it is weaker than being retrieved directly.
LINK_HOP_DISCOUNT = 0.9

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
the older block contributes, never a summary of it.

When a block is marked "you have returned to this note Nx", that recurrence is
itself the finding — say so ("a thread you keep coming back to"), because a
pattern across time is worth more than any single match."""


def _prose_len(content: str) -> int:
    return augi_log.prose_len(content)


def is_echo_eligible(block: Block) -> bool:
    """Deterministic pre-filter — the shared capture gate; echo adds nothing yet."""
    return augi_log.is_capture_block(block)


def _candidates(block: Block, store, model, config: dict[str, Any], k: int = 6):
    """Older related blocks, stratified and ranked relative to their own pool."""
    from openaugi.pipeline.echo_rank import rank
    from openaugi.query import engine

    day = (block.block_time or "")[:10]
    result = engine.context(
        store,
        (block.content or "")[:600],
        k=k * 4,
        expand=True,
        purpose="resurface",
        embedding_model=model,
        config=config,
    )

    # Direct hits carry a score; link-expanded neighbours do not — they were
    # reached by graph traversal, not similarity. Give them their parent's
    # score, discounted, so one hop of the graph competes on the same scale
    # instead of being scored 0 and cut as noise.
    direct_scores = {
        e.block.id: float(e.extras.get("score") or 0.0) for e in result.seen if "score" in e.extras
    }
    scored: list[tuple[Block, float]] = []
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
        if "score" in entry.extras:
            score = float(entry.extras.get("score") or 0.0)
        else:
            parent = direct_scores.get(entry.extras.get("expanded_from", ""), 0.0)
            score = parent * LINK_HOP_DISCOUNT
        scored.append((cand, score))

    ranked = rank(scored)
    if ranked.dropped_external or ranked.dropped_noise:
        logger.debug(
            f"Echo rank: -{ranked.dropped_external} external, "
            f"-{ranked.dropped_noise} below z (pool {ranked.pool_mean:.3f}"
            f"±{ranked.pool_stdev:.3f})"
        )
    ranked.candidates = ranked.candidates[:k]
    return ranked


def _judge(block: Block, ranked, llm) -> list[dict]:
    """Ask the LLM which candidates are worth surfacing. [] means stay silent."""
    recurring = {t.title: t for t in ranked.recurring}
    parts = []
    for i, (cand, _score, _z) in enumerate(ranked.candidates):
        thread = recurring.get(cand.title or "")
        note = ""
        if thread:
            first, last = thread.span
            note = f" [you have returned to this note {thread.recurrence}x, {first}…{last}]"
        parts.append(
            f"[{i + 1}] {cand.title} ({(cand.block_time or '')[:10]}){note}\n"
            f"{(cand.content or '')[:400]}"
        )
    prompt = (
        f"NEW BLOCK (being written now):\n{(block.content or '')[:1200]}\n\n"
        f"OLDER BLOCKS FROM THEIR VAULT:\n" + "\n\n".join(parts)
    )
    try:
        # temperature=0: the same block must not echo on one pass and stay
        # silent on the next. Judgment is the gate, so it has to be repeatable.
        raw = llm.complete(prompt, system=JUDGE_SYSTEM, temperature=0).strip()
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
    titles = {c.title for c, _, _ in ranked.candidates if c.title}
    return [
        line
        for line in (verdict.get("lines") or [])[:3]
        if isinstance(line, dict) and line.get("title") in titles
    ]


def _log_path(vault_path: Path, day: str) -> Path:
    return augi_log.log_path(vault_path, day)


def _ensure_log(path: Path, day: str) -> str:
    return augi_log.ensure_log(path, day)


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


def _render_quiet(block: Block, ranked) -> str:
    """One compact line per silent block: what was closest, and why it lost.

    Silence is the default and most of the day is silence, so this is the only
    window into whether the judge is calibrated. Two boxes, not three — the
    question here is simply whether it should have spoken.
    """
    snippet = " ".join((block.content or "").split())[:60]
    out = [f"\n<!-- quiet:{block.id} -->", f'\n**"{snippet}…"**']
    if not ranked.candidates:
        out.append("- nothing retrieved (no older block cleared the floor)")
    else:
        for cand, score, z in ranked.candidates[:2]:
            when = (cand.block_time or "")[:10]
            out.append(f"- closest: [[{cand.title}]] ({when}) · score {score:.2f}, z {z:+.2f}")
    out.append("- [ ] should have surfaced")
    out.append("- [ ] correctly quiet")
    out.append("")
    return "\n".join(out)


def _write_sections(path: Path, echo_md: str = "", quiet_md: str = "") -> None:
    """Splice into the shared log: echoes under `## Echoes`, quiet under `## Quiet`."""
    if echo_md:
        augi_log.append_to_section(path, augi_log.ECHO_HEADING, echo_md)
    if quiet_md:
        augi_log.append_to_section(path, QUIET_HEADING, quiet_md, intro=augi_log.QUIET_INTRO)


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
        if f"<!-- echo:{block.id} -->" in existing or f"<!-- quiet:{block.id} -->" in existing:
            continue  # already handled this block, either way
        ranked = _candidates(block, store, model, config)
        lines = _judge(block, ranked, llm) if ranked.candidates else []
        if not lines:
            stats["quiet"] += 1
            _write_sections(path, quiet_md=_render_quiet(block, ranked))
            continue
        by_title = {c.title: c for c in ranked.blocks if c.title}
        _write_sections(path, echo_md=_render(block, lines, by_title))
        stats["spoke"] += 1

    if stats["watched"]:
        logger.info(
            f"Echo pass: watched {stats['watched']} · "
            f"spoke {stats['spoke']} · quiet {stats['quiet']}"
        )
        _write_heartbeat(vault_path, stats)
    return stats


def _write_heartbeat(vault_path: Path, stats: dict[str, int]) -> None:
    augi_log.write_heartbeat(_log_path(vault_path, date.today().isoformat()), stats)
