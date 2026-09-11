"""Routing rows — "where does this block live?", asked in the Augi Log.

Post-ingest hook, sibling of `echo.py`. For every new human daily-note block
it proposes where the block belongs and appends one routing row to the
day's Augi Log under `## Routing`. The user answers with checkboxes and
`aaa:` lines; nothing is applied until they tick the day's master box
(`- [ ] process this log`), which `routing_janitor.py` acts on.

Design: docs/plans/augi-log-routing.md. The framing that drove it (2026-09-02): "the
augi log — it's the routing decisions as I go ... you can just suggest where
to stick things or what to merge with based on my hints and I confirm it
there. Here's how you learn me."

Verbs (one word each, the row's grammar and the `aaa:` override grammar):

    extend [[X]]     belongs in X's running log       (janitor inserts, newest-first)
    link [[X]]       related, lives where it is       (routed_to link only)
    file under [[X]] belongs to an area / project     (routed_to link to a container)
    new note         starts a note of its own         (OpenAugi/Notes/<slug>.md)
    memory           life-log, stays in the daily note (ledger only)
    hold             working thought, not ready       (re-proposed later)

Proposals are deterministic first — an `aaa:` hint, then an explicit
wikilink, then where the nearest older writing lives — and an optional
temperature-0 judge (the echo LLM) picks the verb and writes the one-clause
why. Every row is idempotent on its `<!-- route:<block_id> -->` marker.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from openaugi.model.block import Block
from openaugi.pipeline import augi_log
from openaugi.pipeline.writeback import aaa_re, now, read_feedback

logger = logging.getLogger(__name__)

ROUTING_COLLECTION = "routing_queue"
MASTER_BOX = "- [ ] process this log"
ROUTING_INTRO = (
    f"{MASTER_BOX}\n"
    "*Tick a box per row, or answer in its `aaa:` line. Nothing is applied until "
    "the box above is ticked; untouched rows then take the bold suggestion when "
    "augi is confident, otherwise memory.*\n"
)

VERBS = ("extend", "link", "file under", "new note", "memory", "hold")
CONTAINER_TAGS = ("note-type/amoc", "note-type/moc")
PROJECT_TAG = "note-type/pmoc"
ACTIVE_TAG = "status/active"
MAX_SUGGESTIONS = 3

_AAA_RE = aaa_re(indent=True, multiline=True, require_text=True, spaced_colon=True)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---", re.DOTALL)
_DESCRIPTION_RE = re.compile(r"^description:\s*(?P<d>.+?)\s*$", re.MULTILINE)
_DAILY_TITLE_RE = re.compile(r"\d{4}-\d{2}-\d{2}.*")
_VERB_WORDS = (
    ("new note", "new note"),
    ("extend", "extend"),
    ("merge", "extend"),
    ("append", "extend"),
    ("file under", "file under"),
    ("route to", "file under"),
    ("route", "file under"),
    ("link", "link"),
    ("memory", "memory"),
    ("hold", "hold"),
)

# Evidence strengths, on one scale so an aaa: hint always beats retrieval.
SCORE_AAA = 1.0
SCORE_LINK_CONTAINER = 0.9
SCORE_LINK_NOTE = 0.7
#: Retrieval evidence is a z-score from echo_rank; squash it under the link tier.
Z_CAP = 3.0
DEFAULT_CONFIDENT_MARGIN = 1.0


@dataclass
class Container:
    title: str
    path: str
    description: str
    kind: str  # amoc | pmoc | moc


@dataclass
class Suggestion:
    verb: str
    target: str | None
    why: str
    score: float
    source: str  # aaa | link | nearest | judge


@dataclass
class Proposal:
    suggestions: list[Suggestion] = field(default_factory=list)
    memory: bool = False  # the judge (or nothing found) says this is life-log
    why: str = ""
    nearest: list[str] = field(default_factory=list)
    had_aaa: bool = False

    @property
    def top(self) -> Suggestion | None:
        return self.suggestions[0] if self.suggestions else None


# ── Priors: what the history says about where things go ──────────

#: Largest bonus or penalty a prior may add. Priors only ever reorder
#: retrieval-sourced suggestions (scores 0..0.7); they never lift one above
#: the user's own link (0.7+) or hint (1.0).
PRIOR_WEIGHT = 0.1
#: A rate is trusted in proportion to how many decisions back it, up to this many.
PRIOR_FULL_TRUST_AT = 5


@dataclass
class Priors:
    """Acceptance rates from the routing feedback log. Deterministic, dumpable."""

    targets: dict[str, tuple[int, int]] = field(default_factory=dict)  # title → (chosen, seen)
    verbs: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)  # (folder, verb)
    signals: dict[str, int] = field(default_factory=dict)
    decisions: int = 0

    @staticmethod
    def _rate(chosen: int, seen: int) -> float | None:
        return None if seen == 0 else chosen / seen

    def target_rate(self, title: str) -> tuple[float | None, int]:
        chosen, seen = self.targets.get(title, (0, 0))
        return self._rate(chosen, seen), seen

    def verb_rate(self, folder: str, verb: str) -> tuple[float | None, int]:
        chosen, seen = self.verbs.get((folder, verb), (0, 0))
        return self._rate(chosen, seen), seen

    def bonus(self, folder: str, s: Suggestion) -> float:
        """Bounded, evidence-weighted nudge for one suggestion."""
        total = 0.0
        for rate, seen in (
            self.target_rate(s.target) if s.target else (None, 0),
            self.verb_rate(folder, s.verb),
        ):
            if rate is None:
                continue
            trust = min(seen, PRIOR_FULL_TRUST_AT) / PRIOR_FULL_TRUST_AT
            total += (rate - 0.5) * trust
        return max(-PRIOR_WEIGHT, min(PRIOR_WEIGHT, total * PRIOR_WEIGHT))


def _tally(table: dict, key, chosen_key, sign: int = 1) -> None:
    if key is None or (isinstance(key, tuple) and key[1] is None):
        return
    c, n = table.get(key, (0, 0))
    table[key] = (c + (sign if key == chosen_key else 0), n + sign)


def load_priors(vault_path: Path) -> Priors:
    """Read every routing decision on record. Undo reverses the one it undoes."""
    priors = Priors()
    for rec in read_feedback(vault_path, source="routing"):
        signal = rec.get("signal", "")
        priors.signals[signal] = priors.signals.get(signal, 0) + 1
        proposed, chosen = rec.get("proposed") or {}, rec.get("chosen") or {}
        folder = (rec.get("features") or {}).get("folder", "")
        if signal == "undo":
            # reverse the choice it undoes: that target/verb was not, after all, chosen
            _tally(priors.targets, chosen.get("target"), chosen.get("target"), -1)
            _tally(priors.verbs, (folder, chosen.get("verb")), (folder, chosen.get("verb")), -1)
            continue
        priors.decisions += 1
        # every candidate that was on the table counts as seen once; the chosen one as chosen
        for title in {proposed.get("target"), chosen.get("target")} - {None}:
            _tally(priors.targets, title, chosen.get("target"))
        for verb in {proposed.get("verb"), chosen.get("verb")} - {None}:
            _tally(priors.verbs, (folder, verb), (folder, chosen.get("verb")))
    return priors


# ── Eligibility ────────────────────────────────────────────────────


def is_routing_eligible(block: Block) -> bool:
    """A human capture block. AI and reference blocks are never routed."""
    provenance = (block.metadata or {}).get("provenance") or "human"
    return provenance == "human" and augi_log.is_capture_block(block)


# ── Registry: notes that are registered routing targets ────────────

_registry_cache: dict[str, tuple[float, Container | None]] = {}


def _parse_container(title: str, rel_path: str, text: str) -> Container | None:
    """The review-pass rule: a container tag AND a filled description."""
    fm = _FRONTMATTER_RE.match(text)
    desc = _DESCRIPTION_RE.search(fm.group(1)) if fm else None
    if not desc or not desc.group("d").strip():
        return None
    tagged = {t for t in re.findall(r"#([\w/-]+)", text)}
    if any(t in tagged for t in CONTAINER_TAGS):
        kind = "amoc" if "note-type/amoc" in tagged else "moc"
    elif PROJECT_TAG in tagged and ACTIVE_TAG in tagged:
        kind = "pmoc"
    else:
        return None
    return Container(title=title, path=rel_path, description=desc.group("d").strip(), kind=kind)


def load_registry(store, vault_path: Path) -> dict[str, Container]:
    """Registered containers by title, verified on disk, cached by mtime.

    Document blocks carry no tags or description, so the DB only prefilters
    (titles containing "MOC" outside OpenAugi/); the file decides.
    """
    rows = store.conn.execute(
        "SELECT title, json_extract(metadata, '$.source_path') FROM blocks "
        "WHERE kind = 'context_block:document' AND title LIKE '%MOC%'"
    ).fetchall()
    registry: dict[str, Container] = {}
    for title, rel_path in rows:
        if not rel_path or rel_path.startswith("OpenAugi/"):
            continue
        path = vault_path / rel_path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        cached = _registry_cache.get(rel_path)
        if cached and cached[0] == mtime:
            container = cached[1]
        else:
            try:
                container = _parse_container(title, rel_path, path.read_text(encoding="utf-8"))
            except OSError:
                container = None
            _registry_cache[rel_path] = (mtime, container)
        if container:
            registry[title] = container
    return registry


# ── Candidates ─────────────────────────────────────────────────────


def _verb_from(text: str, default: str) -> str:
    lowered = text.lower()
    for word, verb in _VERB_WORDS:
        if word in lowered:
            return verb
    return default


def _note_title_exists(store, title: str) -> bool:
    row = store.conn.execute(
        "SELECT 1 FROM blocks WHERE kind = 'context_block:document' AND title = ? LIMIT 1",
        (title,),
    ).fetchone()
    return row is not None


def _from_aaa(block: Block, store, registry: dict[str, Container]) -> list[Suggestion]:
    out: list[Suggestion] = []
    for match in _AAA_RE.finditer(block.content or ""):
        hint = match.group("reason")
        targets = _WIKILINK_RE.findall(hint)
        if not targets:
            # a bare title mentioned in the hint, longest registered match wins
            lowered = hint.lower()
            targets = sorted((t for t in registry if t.lower() in lowered), key=len, reverse=True)[
                :1
            ]
        verb = _verb_from(hint, "file under")
        if verb in ("memory", "hold", "new note"):
            out.append(Suggestion(verb, None, "your aaa: hint", SCORE_AAA, "aaa"))
            continue
        for target in targets:
            if verb == "file under" and target not in registry:
                verb = "extend" if _note_title_exists(store, target) else verb
            out.append(Suggestion(verb, target, "your aaa: hint", SCORE_AAA, "aaa"))
    return out


def _from_links(block: Block, store, registry: dict[str, Container]) -> list[Suggestion]:
    out: list[Suggestion] = []
    content = _AAA_RE.sub("", block.content or "")
    for target in dict.fromkeys(_WIKILINK_RE.findall(content)):
        if target in registry:
            out.append(
                Suggestion("file under", target, "you linked it", SCORE_LINK_CONTAINER, "link")
            )
        elif _note_title_exists(store, target) and not _DAILY_TITLE_RE.fullmatch(target):
            out.append(Suggestion("link", target, "you linked it", SCORE_LINK_NOTE, "link"))
    return out


def _nearest(block: Block, store, model, config: dict[str, Any]) -> list[tuple[Block, float]]:
    """Older related blocks as (block, z). Isolated so tests can stub it."""
    from openaugi.pipeline.echo import _candidates

    ranked = _candidates(block, store, model, config)
    return [(cand, z) for cand, _score, z in ranked.candidates]


def _from_nearest(
    nearest: list[tuple[Block, float]], store, registry: dict[str, Container]
) -> tuple[list[Suggestion], list[str]]:
    """Where the nearest older writing lives: its container, or its note."""
    best: dict[tuple[str, str], Suggestion] = {}
    titles: list[str] = []
    ids = [b.id for b, _ in nearest]
    routes = store.get_routed_container_titles(ids) if ids else {}
    for cand, z in nearest:
        score = max(0.0, min(z, Z_CAP)) / Z_CAP * SCORE_LINK_NOTE  # never beats a link
        src = (cand.metadata or {}).get("source_path") or ""
        parent = (cand.metadata or {}).get("parent_note_title") or cand.title or ""
        if parent and parent not in titles:
            titles.append(parent)
        found: list[tuple[str, str]] = []
        for container in routes.get(cand.id, []):
            found.append(("file under", container))
        if parent in registry:
            found.append(("file under", parent))
        elif (
            parent
            and not _DAILY_TITLE_RE.fullmatch(parent)  # daily notes are never a home
            and not src.startswith("OpenAugi/")
            and (cand.metadata or {}).get("provenance", "human") == "human"
        ):
            found.append(("extend", parent))
        for verb, target in found:
            key = (verb, target)
            when = (cand.block_time or "")[:10]
            why = f"your nearest older writing on this lives there ({when})"
            if key not in best or best[key].score < score:
                best[key] = Suggestion(verb, target, why, score, "nearest")
    return sorted(best.values(), key=lambda s: s.score, reverse=True), titles


JUDGE_SYSTEM = """You decide where a block of someone's daily note belongs.

You get the block and candidate homes found from their own hints, links and
earlier writing. Answer two things:
1. Is this LIFE-LOG (kids, meals, workouts, chores, moods, family, errands —
   a memory to keep in the daily note) or a WORKING THOUGHT about their
   projects, ideas, work, learning, money, health plans?
2. If a working thought, which candidate is its home, if any. Prefer the
   candidate whose description matches. Never invent a target that is not
   in the list.

Return ONLY JSON:
{"memory": true|false, "pick": <1-based candidate index or 0 for none>,
 "why": "<one clause, max 12 words, what the block adds to that home>"}"""


def _judge(block: Block, proposal: Proposal, registry: dict[str, Container], llm) -> None:
    """Let the LLM reorder and explain. Mutates the proposal; silent on failure."""
    parts = []
    for i, s in enumerate(proposal.suggestions):
        desc = registry[s.target].description if s.target in registry else ""
        parts.append(f"[{i + 1}] {s.verb} [[{s.target}]]" + (f" — {desc}" if desc else ""))
    prompt = f"BLOCK:\n{(block.content or '')[:1200]}\n\nCANDIDATES:\n" + (
        "\n".join(parts) if parts else "(none found)"
    )
    try:
        raw = llm.complete(prompt, system=JUDGE_SYSTEM, temperature=0).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        verdict = json.loads(match.group(0)) if match else {}
    except Exception as e:  # the judge is optional; the deterministic order stands
        logger.warning(f"Routing judge failed: {e}")
        return
    why = str(verdict.get("why") or "").strip()
    if verdict.get("memory") and not any(s.source == "aaa" for s in proposal.suggestions):
        proposal.memory = True
        proposal.why = why or "reads as life-log"
        return
    pick = verdict.get("pick")
    if isinstance(pick, int) and 1 <= pick <= len(proposal.suggestions):
        chosen = proposal.suggestions.pop(pick - 1)
        if why:
            chosen.why = why
        chosen.source = "judge" if chosen.source == "nearest" else chosen.source
        proposal.suggestions.insert(0, chosen)


def propose(
    block: Block,
    store,
    model,
    config: dict[str, Any],
    registry: dict[str, Container],
    llm=None,
    nearest: list[tuple[Block, float]] | None = None,
    priors: Priors | None = None,
) -> Proposal:
    """Rank the candidate homes for one block. Deterministic unless `llm` is given.

    `priors` may nudge retrieval-sourced suggestions by past decisions; it
    never touches an aaa: or link suggestion, so the user's own words always
    rank first.
    """
    proposal = Proposal(had_aaa=bool(_AAA_RE.search(block.content or "")))
    found = _from_aaa(block, store, registry) + _from_links(block, store, registry)
    if nearest is None:
        try:
            nearest = _nearest(block, store, model, config)
        except Exception as e:
            logger.warning(f"Routing retrieval failed: {e}")
            nearest = []
    from_nearest, proposal.nearest = _from_nearest(nearest, store, registry)
    if priors is not None:
        folder = str(Path((block.metadata or {}).get("source_path", "")).parent)
        for s in from_nearest:
            s.score = max(0.0, min(SCORE_LINK_NOTE, s.score + priors.bonus(folder, s)))
    found += from_nearest
    best: dict[tuple[str, str | None], Suggestion] = {}
    for s in found:
        key = (s.verb, s.target)
        if key not in best or best[key].score < s.score:
            best[key] = s
    proposal.suggestions = sorted(best.values(), key=lambda s: s.score, reverse=True)[
        :MAX_SUGGESTIONS
    ]
    if llm is not None:
        _judge(block, proposal, registry, llm)
    return proposal


#: Verbs that only write a DB link. Retrieval evidence alone may apply these;
#: it may never write into one of the user's notes (extend) or mint one (new note).
DB_ONLY_VERBS = ("file under", "link")


def is_confident(proposal: Proposal, margin: float = DEFAULT_CONFIDENT_MARGIN) -> bool:
    """May the top suggestion apply without a tick? Only on strong evidence.

    The user's own hint or link: yes, whatever the verb. Retrieval (or the judge
    picking among retrieval hits): only for DB-only verbs, and only when the
    top candidate's z-margin over the runner-up clears `margin`.
    """
    top = proposal.top
    if top is None or proposal.memory:
        return False
    if top.source in ("aaa", "link"):
        return True
    if top.verb not in DB_ONLY_VERBS:
        return False
    runner_up = proposal.suggestions[1].score if len(proposal.suggestions) > 1 else 0.0
    return (top.score - runner_up) * Z_CAP / SCORE_LINK_NOTE >= margin


# ── Rendering ──────────────────────────────────────────────────────


def _label(s: Suggestion) -> str:
    return f"{s.verb} [[{s.target}]]" if s.target else s.verb


def render_row(block: Block, proposal: Proposal) -> str:
    snippet = " ".join((block.content or "").split())[:70]
    src = (block.metadata or {}).get("source_path", "")
    day = (block.block_time or "")[:10]
    out = [f"\n<!-- route:{block.id} -->", f'\n### route "{snippet}…"']
    if src:
        out.append(f"*[[{Path(src).stem}]]{f' · {day}' if day else ''}*\n")
    if proposal.memory:
        out.append(f"*reads as a memory — {proposal.why}*")
    for i, s in enumerate(proposal.suggestions):
        label = f"**{_label(s)}**" if i == 0 and not proposal.memory else _label(s)
        out.append(f"- [ ] {label} — {s.why}")
    if not any(s.verb == "new note" for s in proposal.suggestions):
        out.append("- [ ] new note")
    if not any(s.verb == "memory" for s in proposal.suggestions):
        out.append("- [ ] memory")
    if not any(s.verb == "hold" for s in proposal.suggestions):
        out.append("- [ ] hold")
    out.append("aaa:")
    out.append("")
    return "\n".join(out)


# ── Ledger ─────────────────────────────────────────────────────────


def log_record_id(day: str) -> str:
    return f"log:{day}"


def _record_row(store, block: Block, proposal: Proposal, day: str, confident: bool) -> None:
    store.write_record(
        ROUTING_COLLECTION,
        block.id,
        {
            "kind": "row",
            "status": "proposed",
            "day": day,
            "source_path": (block.metadata or {}).get("source_path", ""),
            "proposed": [asdict(s) for s in proposal.suggestions],
            "memory": proposal.memory,
            "confident": confident,
            "features": {
                "folder": str(Path((block.metadata or {}).get("source_path", "")).parent),
                "tags": list(block.tags or []),
                "nearest": proposal.nearest[:5],
                "had_aaa": proposal.had_aaa,
            },
            "proposed_at": now(),
        },
        now(),
    )


def _record_log(store, day: str, rel_log: str) -> None:
    existing = store.list_records(ROUTING_COLLECTION, where={"kind": "log", "day": day}, limit=1)
    if existing:
        return
    store.write_record(
        ROUTING_COLLECTION,
        log_record_id(day),
        {"kind": "log", "status": "waiting", "day": day, "path": rel_log, "opened_at": now()},
        now(),
    )


def waiting_logs(store) -> list[dict]:
    """Logs whose master box is still unticked — what the board reports."""
    return store.list_records(
        ROUTING_COLLECTION, where={"kind": "log", "status": "waiting"}, limit=1000
    )


# ── Entry point ────────────────────────────────────────────────────


def run_routing(
    new_blocks: list[Block],
    vault_path: Path,
    store,
    model,
    config: dict[str, Any],
) -> dict[str, int]:
    """Routing pass over newly ingested blocks. Returns {watched, proposed}."""
    settings = config.get("routing", {})
    if not settings.get("enabled", True):
        return {"watched": 0, "proposed": 0}
    eligible = [b for b in new_blocks if is_routing_eligible(b)]
    stats = {"watched": len(eligible), "proposed": 0}
    if not eligible:
        return stats

    from openaugi.models import get_llm_model

    llm = get_llm_model(config.get("models", {}).get("llm"))
    margin = float(settings.get("confident_margin", DEFAULT_CONFIDENT_MARGIN))
    registry = load_registry(store, vault_path)
    priors = load_priors(vault_path)

    for block in eligible:
        day = (block.block_time or "")[:10] or date.today().isoformat()
        path = augi_log.log_path(vault_path, day)
        existing = augi_log.ensure_log(path, day)
        if f"<!-- route:{block.id} -->" in existing:
            continue
        proposal = propose(block, store, model, config, registry, llm=llm, priors=priors)
        augi_log.append_to_section(
            path, augi_log.ROUTING_HEADING, render_row(block, proposal), intro=ROUTING_INTRO
        )
        _record_row(store, block, proposal, day, is_confident(proposal, margin))
        _record_log(store, day, str(path.relative_to(vault_path)))
        stats["proposed"] += 1

    if stats["watched"]:
        logger.info(f"Routing pass: watched {stats['watched']} · proposed {stats['proposed']}")
    return stats
