"""Stratify-then-cluster ranking for proactive echo.

Global similarity does not work in a single-author vault. A 7-day replay
(2026-08-29) found top-scores clustered at 0.53-0.68 whether the block was an
architecture note or a journal entry about training wheels: cosine mostly
measures "this is Chris writing" — his voice, his recurring nouns — not "this
is the same thought." No global threshold can separate those.

So we apply Chris's own method (`2026-03-02 - Stratify then cluster`, already
the Contextgraph Rule in AGENT/routing.md): sort roughly first, then sort
within the pile. Three steps, using only signal available for every block:

1. **Stratify by the taxonomy's own facets** — `source` (who wrote it) and
   `note-type` (what kind of note it is), from `AGENT/My Taxonomy.md`. Both are
   read from tags when present and inferred from the folder when not, which
   matters: only 26% of blocks carry tags and 2.4% carry routing, so a
   tag-only stratifier would gut recall. External sources (podcast, readwise,
   webclip, ai-chat, notebookLM, gdrive) are *someone else's words* — you never
   "already thought" a podcast, so they never echo.

2. **Score relative, not absolute.** Rank by z-score against the pool's own
   distribution. A 0.63 in a pool of 0.60±0.02 is noise; a 0.63 in a pool of
   0.52±0.03 is a spike. Same primitive as the day-job framing: surprise under
   your own model's predicted distribution.

3. **Cluster by source note.** Group survivors and count recurrences, so the
   agent can say "a thread you've returned to 5 times" rather than "here is one
   similar block" — which is what makes pattern-tier echoes real.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from openaugi.model.block import Block

logger = logging.getLogger(__name__)

# ── Facet: source (AGENT/My Taxonomy.md) ────────────────────────────
# `source/capture` — Chris's own writing; assumed when no source tag is present.
CAPTURE = "source/capture"
EXTERNAL_SOURCES = {
    "source/ai-chat",
    "source/webclip",
    "source/readwise",
    "source/podcast",
    "source/notebookLM",
    "source/gdrive",
}
# Folder → source, for the 74% of blocks carrying no tag at all.
SOURCE_BY_FOLDER = (
    ("_private/2-Reference/", "source/webclip"),
    ("_sources/gdrive/", "source/gdrive"),
    ("_sources/", "source/webclip"),
)

# ── Facet: note-type (same taxonomy) ────────────────────────────────
NOTE_TYPE_BY_FOLDER = (
    ("_private/0-Fleeting-Inbox/", "note-type/daily-journal"),
    ("_private/0-Inbox/", "note-type/daily-journal"),
    ("_private/5-Journals/", "note-type/reflection"),
    ("_private/3-MOCs and Projects/", "note-type/moc"),
    ("_private/1-Notes/", "note-type/learning"),
    ("_private/2-Reference/", "note-type/reference"),
)
UNKNOWN_NOTE_TYPE = "note-type/unknown"

# A candidate must beat the pool by this many standard deviations to survive.
# Permissive on purpose: this is a noise cut before judgment, not the judgment.
DEFAULT_Z_MIN = 0.5
# Below this, a z-score is meaningless — keep everything and let judgment decide.
MIN_POOL_FOR_Z = 4


def source_facet(block: Block) -> str:
    """Who wrote it. Tag wins; folder infers; `source/capture` is the default."""
    for tag in block.tags or []:
        normalized = tag.lstrip("#")
        if normalized in EXTERNAL_SOURCES or normalized == CAPTURE:
            return normalized
    path = (block.metadata or {}).get("source_path") or ""
    for prefix, source in SOURCE_BY_FOLDER:
        if prefix in path:
            return source
    return CAPTURE


def note_type(block: Block) -> str:
    """What kind of note it is. Tag wins; folder infers."""
    for tag in block.tags or []:
        normalized = tag.lstrip("#")
        if normalized.startswith("note-type/"):
            return normalized
    path = (block.metadata or {}).get("source_path") or ""
    for prefix, kind in NOTE_TYPE_BY_FOLDER:
        if prefix in path:
            return kind
    return UNKNOWN_NOTE_TYPE


def is_own_writing(block: Block) -> bool:
    """True when Chris wrote it — the only stratum an echo may draw from."""
    return source_facet(block) == CAPTURE


@dataclass
class Thread:
    """Candidates grouped by the note they came from."""

    title: str
    note_type: str = UNKNOWN_NOTE_TYPE
    blocks: list[Block] = field(default_factory=list)
    best_score: float = 0.0
    best_z: float = 0.0

    @property
    def recurrence(self) -> int:
        return len(self.blocks)

    @property
    def span(self) -> tuple[str, str]:
        times = sorted((b.block_time or "")[:10] for b in self.blocks if b.block_time)
        return (times[0], times[-1]) if times else ("", "")


@dataclass
class Ranked:
    """Ranking output: what survived, how it clustered, and why."""

    candidates: list[tuple[Block, float, float]] = field(default_factory=list)  # block, score, z
    threads: list[Thread] = field(default_factory=list)
    pool_mean: float = 0.0
    pool_stdev: float = 0.0
    dropped_external: int = 0
    dropped_noise: int = 0

    @property
    def blocks(self) -> list[Block]:
        return [b for b, _, _ in self.candidates]

    @property
    def recurring(self) -> list[Thread]:
        """Threads seen more than once — the pattern signal."""
        return [t for t in self.threads if t.recurrence > 1]


def rank(
    scored: list[tuple[Block, float]],
    z_min: float = DEFAULT_Z_MIN,
    own_writing_only: bool = True,
) -> Ranked:
    """Stratify, score relatively, then cluster. Pure function — easy to test."""
    result = Ranked()
    if not scored:
        return result

    # 1. Stratify — an echo can only come from Chris's own writing.
    if own_writing_only:
        kept = [(b, s) for b, s in scored if is_own_writing(b)]
        result.dropped_external = len(scored) - len(kept)
    else:
        kept = list(scored)
    if not kept:
        return result

    # 2. Relative scoring — a spike against this pool's own distribution.
    scores = [s for _, s in kept]
    result.pool_mean = statistics.fmean(scores)
    result.pool_stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    if len(kept) >= MIN_POOL_FOR_Z and result.pool_stdev > 0:
        z_scored = [(b, s, (s - result.pool_mean) / result.pool_stdev) for b, s in kept]
        survivors = [(b, s, z) for b, s, z in z_scored if z >= z_min]
        result.dropped_noise = len(z_scored) - len(survivors)
        if not survivors:  # never return nothing on a technicality
            survivors = sorted(z_scored, key=lambda t: t[1], reverse=True)[:1]
            result.dropped_noise = len(z_scored) - 1
    else:
        survivors = [(b, s, 0.0) for b, s in kept]

    result.candidates = sorted(survivors, key=lambda t: t[2], reverse=True)

    # 3. Cluster by source note — recurrence is the pattern signal.
    threads: dict[str, Thread] = {}
    for block, score, z in result.candidates:
        key = block.title or (block.metadata or {}).get("source_path") or block.id
        thread = threads.setdefault(key, Thread(title=key, note_type=note_type(block)))
        thread.blocks.append(block)
        thread.best_score = max(thread.best_score, score)
        thread.best_z = max(thread.best_z, z)
    result.threads = sorted(threads.values(), key=lambda t: (t.recurrence, t.best_z), reverse=True)
    return result
