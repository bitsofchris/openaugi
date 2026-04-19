"""Deterministic block splitter — the shared primitive.

One function, one contract: given markdown text, return a list of `Segment`s
using the same deterministic rules OpenAugi's vault ingest uses. No LLM,
no database, no side effects.

Any agent, script, skill, or adapter that needs "cut this note into blocks
the way OpenAugi does" should import from here instead of re-implementing
heading/qqq/zzz parsing. The `openaugi split` CLI is a thin wrapper over
`split_file`.

See [docs/splitter.md](../../../docs/splitter.md) for usage from agents and
[docs/plans/zzz-instructions.md](../../../docs/plans/zzz-instructions.md) for
the `zzz:` convention.

## Public API

- `split_text(text) -> list[Segment]` — pure, for in-memory content.
- `split_file(path) -> SplitResult` — reads a file, resolves a filename
  date, returns segments plus file-level metadata.
- `Segment`, `SplitResult` — Pydantic models; call `.model_dump()` for JSON.

## Splitting rules (in order)

1. Strip YAML frontmatter; capture `tags:` from it.
2. Split on ANY markdown heading (`#`–`######`). Heading-like lines inside
   fenced code blocks are ignored.
3. Within each section, split on standalone `qqq` marker lines.
4. For each sub-section:
   - Extract `zzz[:] body` lines → `zzz_instructions` list, strip from content.
   - Drop if the remaining content is empty or structurally meaningless
     (horizontal rules, empty checkboxes, URL-only lines, dataview blocks).
5. A YYYY-MM-DD prefix on a heading sets the date for itself and subsequent
   sections until the next date-headed section.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ── Regex patterns ────────────────────────────────────────────────

ANY_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)$", re.MULTILINE)
TAG_PATTERN = re.compile(r"(?<!\w)#([a-zA-Z0-9_/\-]+)")
LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
FILENAME_DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})")
# Two-digit year date found anywhere in the stem, e.g. "WK - 25-11-09".
# Negative lookbehind prevents matching the last two digits of a 4-digit year.
WK_DATE_PATTERN = re.compile(r"(?<!\d)(\d{2})-(\d{2})-(\d{2})(?!\d)")
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Atomic thought delimiter — a line containing only `qqq` (case-insensitive).
QQQ_PATTERN = re.compile(r"^[ \t]*[qQ]{3}[ \t]*$", re.MULTILINE)
DATAVIEW_BLOCK_PATTERN = re.compile(r"```dataview\b.*?```", re.DOTALL | re.IGNORECASE)
# Per-block agent instructions — lines starting with `zzz` (case-insensitive),
# optionally followed by a colon.
ZZZ_PATTERN = re.compile(r"^[ \t]*[zZ]{3}\b[:\s]*(.*?)\s*$", re.MULTILINE)


# ── Public types ──────────────────────────────────────────────────


class Segment(BaseModel):
    """One deterministic split of a note.

    `content` is the raw sub-section (including any `zzz` lines); `clean_content`
    has the `zzz` lines stripped. The raw form is what you hash for identity —
    editing a `zzz` instruction should produce a new block.
    """

    content: str  # raw, pre-zzz-strip
    clean_content: str  # zzz lines removed, ready to store/display
    zzz_instructions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    section_heading: str | None = None
    section_date: str | None = None  # YYYY-MM-DD inherited from nearest date-headed ancestor
    granularity: Literal["document", "section"] = "section"
    raw_hash: str  # sha256(content)[:16] — stable identity for this segment

    def __str__(self) -> str:  # debug-friendly
        head = f"[{self.section_heading}] " if self.section_heading else ""
        return f"{head}{self.clean_content[:80]}"


class SplitResult(BaseModel):
    """Result of splitting a single file."""

    source_path: str  # path as provided (absolute or relative to caller's cwd)
    doc_hash: str  # sha256 of full file content [:16]
    filename_date: str | None = None  # YYYY-MM-DD extracted from stem, if any
    frontmatter_tags: list[str] = Field(default_factory=list)
    segments: list[Segment]


# ── Public API ────────────────────────────────────────────────────


def split_text(text: str) -> list[Segment]:
    """Split markdown text into segments using OpenAugi's deterministic rules.

    Pure function — no filesystem, no network. Frontmatter is stripped if
    present at the top of `text`, but any `tags:` it contains are discarded
    (use `split_file` if you need frontmatter tags).
    """
    body, _fm_tags = _strip_frontmatter(text)
    return _segments_from_body(body)


def split_file(path: str | Path) -> SplitResult:
    """Read a markdown file and return its segments plus file metadata.

    Date resolution priority for the file as a whole:
    `filename 4-digit date > filename 2-digit "WK" date`. Per-segment dates
    from heading prefixes still take precedence inside each segment.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    body, fm_tags = _strip_frontmatter(raw)

    title_date = _extract_filename_date(p)
    wk_date = _extract_wk_date(p)
    effective_title_date = title_date or wk_date

    # Weekly-reflection notes (2-digit year in stem) are kept as a single
    # segment — heading-splitting would fragment question/answer pairs.
    if wk_date:
        segments = _segments_from_single_section(body, wk_date, None)
    else:
        segments = _segments_from_body(body)

    # If a segment has no date of its own, fall back to the filename date.
    if effective_title_date:
        for s in segments:
            if s.section_date is None:
                s.section_date = effective_title_date

    return SplitResult(
        source_path=str(path),
        doc_hash=_hash(raw),
        filename_date=effective_title_date,
        frontmatter_tags=fm_tags,
        segments=segments,
    )


# ── Internals ─────────────────────────────────────────────────────


def _segments_from_body(body: str) -> list[Segment]:
    sections = _split_by_headings(body)
    out: list[Segment] = []
    for section_content, section_date, section_heading in sections:
        out.extend(_segments_from_single_section(section_content, section_date, section_heading))

    # Granularity: if the whole file collapses to one segment, it's document-level.
    if len(out) == 1:
        out[0].granularity = "document"
    return out


def _segments_from_single_section(
    section_content: str, section_date: str | None, section_heading: str | None
) -> list[Segment]:
    results: list[Segment] = []
    for sub in _split_by_qqq(section_content):
        stripped = sub.strip()
        if not stripped or not _has_meaningful_content(stripped):
            continue
        clean, zzz = _extract_zzz_instructions(stripped)
        if not clean:
            continue
        results.append(
            Segment(
                content=stripped,
                clean_content=clean,
                zzz_instructions=zzz,
                tags=_extract_tags(clean),
                links=_extract_links(clean),
                section_heading=section_heading,
                section_date=section_date,
                granularity="section",
                raw_hash=_hash(stripped),
            )
        )
    return results


def _code_fence_ranges(content: str) -> list[tuple[int, int]]:
    """Return (start, end) character ranges for all fenced code blocks.

    Used to exclude `#` lines inside code blocks from heading detection.
    """
    ranges: list[tuple[int, int]] = []
    fence_re = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
    fence_matches = list(fence_re.finditer(content))
    i = 0
    while i < len(fence_matches):
        opener = fence_matches[i]
        fence_char = opener.group(1)[0]
        fence_len = len(opener.group(1))
        j = i + 1
        while j < len(fence_matches):
            closer = fence_matches[j]
            if closer.group(1)[0] == fence_char and len(closer.group(1)) >= fence_len:
                ranges.append((opener.start(), closer.end()))
                i = j + 1
                break
            j += 1
        else:
            ranges.append((opener.start(), len(content)))
            break
    return ranges


def _split_by_headings(content: str) -> list[tuple[str, str | None, str | None]]:
    """Split content by any markdown heading.

    Returns (section_content, date_str, heading_text). A YYYY-MM-DD prefix on
    a heading establishes the date for itself and all subsequent sections
    until another date-headed heading appears.
    """
    all_matches = list(ANY_HEADING_PATTERN.finditer(content))
    fence_ranges = _code_fence_ranges(content)
    matches = [
        m for m in all_matches if not any(start <= m.start() < end for start, end in fence_ranges)
    ]

    if not matches:
        return [(content, None, None)]

    sections: list[tuple[str, str | None, str | None]] = []
    current_date: str | None = None

    if matches[0].start() > 0:
        preamble = content[: matches[0].start()]
        if preamble.strip():
            sections.append((preamble, None, None))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", heading_text)
        if date_match:
            current_date = date_match.group(1)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((content[content_start:content_end], current_date, heading_text))

    return sections


def _split_by_qqq(content: str) -> list[str]:
    """Split a section on standalone `qqq` lines (case-insensitive).

    Returns `[content]` unchanged if there are no markers, so callers who
    don't use qqq see no behavior change.
    """
    matches = list(QQQ_PATTERN.finditer(content))
    if not matches:
        return [content]
    segments: list[str] = []
    cursor = 0
    for match in matches:
        segments.append(content[cursor : match.start()])
        cursor = match.end()
    segments.append(content[cursor:])
    return segments


def _has_meaningful_content(text: str) -> bool:
    """False if text contains only structural markdown with no real content."""
    cleaned = DATAVIEW_BLOCK_PATTERN.sub("", text)
    for line in cleaned.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == "---":
            continue
        if re.match(r"^-\s+\[\s*\]\s*$", s):
            continue
        if re.match(r"^-\s*$", s):
            continue
        if re.match(r"^-\s+[xX]\s*$", s):
            continue
        if re.match(r"^-?\s*https?://\S+\s*$", s):
            continue
        return True
    return False


def _strip_frontmatter(content: str) -> tuple[str, list[str]]:
    """Remove YAML frontmatter, extract tags from it (no yaml dep)."""
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return content, []

    fm_text = match.group(1)
    body = content[match.end() :]

    tags: list[str] = []
    in_tags = False
    for line in fm_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("tags:"):
            in_tags = True
            rest = stripped[5:].strip()
            if rest.startswith("["):
                tags.extend(
                    t.strip().strip("'\"") for t in rest.strip("[]").split(",") if t.strip()
                )
                in_tags = False
            continue
        if in_tags:
            if stripped.startswith("- "):
                tags.append(stripped[2:].strip().strip("'\""))
            elif stripped and not stripped.startswith("-"):
                in_tags = False

    return body, tags


def _extract_tags(text: str) -> list[str]:
    return _unique_ordered(TAG_PATTERN.findall(text))


def _extract_links(text: str) -> list[str]:
    return _unique_ordered(LINK_PATTERN.findall(text))


def _extract_zzz_instructions(text: str) -> tuple[str, list[str]]:
    """Extract `zzz` lines. Returns (clean_content, instructions)."""
    instructions: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body:
            instructions.append(body)
        return ""

    stripped = ZZZ_PATTERN.sub(_capture, text)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip("\n")
    return stripped, instructions


def _extract_filename_date(file_path: Path) -> str | None:
    match = FILENAME_DATE_PATTERN.match(file_path.stem)
    return match.group(1) if match else None


def _extract_wk_date(file_path: Path) -> str | None:
    """YYYY-MM-DD from a 2-digit-year stem (e.g. 'WK - 25-11-09'). Assumes 20YY."""
    if FILENAME_DATE_PATTERN.match(file_path.stem):
        return None
    match = WK_DATE_PATTERN.search(file_path.stem)
    if match:
        yy, mm, dd = match.group(1), match.group(2), match.group(3)
        return f"20{yy}-{mm}-{dd}"
    return None


def _parse_date(date_str: str) -> str | None:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except (ValueError, TypeError):
        return None


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _unique_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
