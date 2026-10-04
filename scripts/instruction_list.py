#!/usr/bin/env python3
"""instruction_list.py — list every typed instruction marker in a date window.

A user types an instruction to the agent as a line carrying a marker, by
default `aaa:` (any case), anywhere in their own notes. A model asked to
"find the week's instructions" misses some. This script does not: it greps,
dates each hit, and prints all of them, so the weekly pass can rule on every
one.

A hit is in the window when either holds:

  1. the note's filename carries a date (YYYY-MM-DD) inside the window — a
     daily note or a dated side note; or
  2. the hit sits under a dated level-3 heading (`### YYYY-MM-DD ...`) inside
     the window — a journal entry in a longer-lived note.

A dated heading above the hit wins over the filename date. Lines inside code
fences are skipped, and so is a marker written in backticks (prose *about*
the marker). A hit whose line ends in `(done ...)` is reported as done.

Which folders hold the user's own writing is the vault's business: pass
--exclude for folders of generated output. The script only lists; it never
decides what an instruction means or whether anything acted on it.

Usage:
    python3 scripts/instruction_list.py --days 7                  # vault from config
    python3 scripts/instruction_list.py --vault ~/vault --start 2026-09-28 --end 2026-10-04
    python3 scripts/instruction_list.py --days 7 --exclude OpenAugi --json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

DEFAULT_MARKER = "aaa"
DEFAULT_EXCLUDE = ("OpenAugi",)
SKIP_DIRS = {".git", ".obsidian", ".claude", ".trash", "node_modules"}
CONTEXT_CHARS = 240

DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
DONE_RE = re.compile(r"\(done\b[^)]*\)\s*$", re.IGNORECASE)


@dataclass
class Hit:
    date: str
    path: str
    line: int
    text: str
    context: str
    done: bool


def marker_re(marker: str) -> re.Pattern[str]:
    """The marker followed by a colon, not inside a word and not in backticks."""
    return re.compile(rf"(?<![\w`]){re.escape(marker)}:\s*(?P<text>.*)$", re.IGNORECASE)


def parse_date(text: str) -> date | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _context(lines: list[str], idx: int) -> str:
    """The nearest non-empty line above the hit that is not itself a heading."""
    for j in range(idx - 1, -1, -1):
        prev = lines[j].strip()
        if not prev:
            continue
        if HEADING_RE.match(prev):
            return ""
        return prev[:CONTEXT_CHARS]
    return ""


def scan_text(text: str, rel_path: str, file_date: date | None, marker: str) -> list[Hit]:
    """Every marker line in one note, dated by heading first, filename second.

    Hits with no date at all are returned with an empty date; the caller
    drops them when filtering by window.
    """
    pattern = marker_re(marker)
    hits: list[Hit] = []
    lines = text.splitlines()
    heading_date: date | None = None
    in_fence = False
    for idx, raw in enumerate(lines):
        if FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        h = HEADING_RE.match(raw)
        if h:
            level = len(h.group(1))
            if level == 3:
                heading_date = parse_date(h.group(2))
            elif level < 3:
                heading_date = None
            continue
        m = pattern.search(raw)
        if not m:
            continue
        when = heading_date or file_date
        body = m.group("text").strip()
        hits.append(
            Hit(
                date=when.isoformat() if when else "",
                path=rel_path,
                line=idx + 1,
                text=body,
                context=_context(lines, idx),
                done=bool(DONE_RE.search(body)),
            )
        )
    return hits


def iter_notes(vault: Path, exclude: tuple[str, ...]):
    excluded = [vault / e for e in exclude]
    for path in sorted(vault.rglob("*.md")):
        parts = set(path.relative_to(vault).parts)
        if parts & SKIP_DIRS:
            continue
        if any(ex == path or ex in path.parents for ex in excluded):
            continue
        yield path


def collect(
    vault: Path,
    start: date,
    end: date,
    marker: str = DEFAULT_MARKER,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
) -> list[Hit]:
    """All hits dated inside [start, end], oldest first."""
    hits: list[Hit] = []
    needle = f"{marker}:".lower()
    for path in iter_notes(vault, exclude):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if needle not in text.lower():
            continue
        rel = path.relative_to(vault).as_posix()
        for hit in scan_text(text, rel, parse_date(path.stem), marker):
            if hit.date and start.isoformat() <= hit.date <= end.isoformat():
                hits.append(hit)
    hits.sort(key=lambda h: (h.date, h.path, h.line))
    return hits


def render(hits: list[Hit], start: date, end: date, marker: str) -> str:
    out = [f"{len(hits)} `{marker}:` lines, {start.isoformat()} to {end.isoformat()}", ""]
    for n, h in enumerate(hits, 1):
        state = " [done]" if h.done else ""
        out.append(f"{n}. {h.date} · {h.path}:{h.line}{state}")
        out.append(f"   {marker}: {h.text}")
        if h.context:
            out.append(f"   under: {h.context}")
    return "\n".join(out)


def resolve_vault(explicit: str | None) -> Path | None:
    """The vault root: the flag if given, else the openaugi config's default_path."""
    if explicit:
        return Path(explicit).expanduser()
    try:
        from openaugi.config import load_config, resolve_vault_path
    except ImportError:
        return None
    resolved = resolve_vault_path(None, load_config())
    return Path(resolved) if resolved else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--vault",
        help="vault root (default: [vault] default_path from the openaugi config)",
    )
    ap.add_argument("--days", type=int, default=7, help="window length ending at --end")
    ap.add_argument("--start", help="first day of the window, YYYY-MM-DD (overrides --days)")
    ap.add_argument("--end", help="last day of the window, YYYY-MM-DD (default: today)")
    ap.add_argument("--marker", default=DEFAULT_MARKER, help="the instruction marker word")
    ap.add_argument(
        "--exclude",
        action="append",
        help="folder to skip, relative to the vault; repeatable "
        f"(default: {', '.join(DEFAULT_EXCLUDE)})",
    )
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    vault = resolve_vault(args.vault)
    if vault is None:
        ap.error("no vault: pass --vault or set [vault] default_path in the openaugi config")
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days - 1)
    exclude = tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDE
    hits = collect(vault, start, end, args.marker, exclude)
    if args.json:
        print(json.dumps([asdict(h) for h in hits], indent=1, ensure_ascii=False))
    else:
        print(render(hits, start, end, args.marker))


if __name__ == "__main__":
    main()
