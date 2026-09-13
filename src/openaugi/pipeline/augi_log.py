"""The Augi Log file — one per day, shared by every proactive pass.

`OpenAugi/YYYY/MM/DD/Augi Log.md` is written by proactive echo (echo.py) and
by routing (route.py), and read back by their janitors. This module owns the
things they must agree on so neither has to know the other's layout:

- the path and the header,
- the capture eligibility gate (which daily-note blocks any pass may act on),
- the section order and a writer that splices into the right section,
- the per-day heartbeat that keeps silence legible.

The file shape is fixed:

    [header]
    ## Routing
    ## Echoes
    ## Quiet — closest match, not surfaced
    <!-- heartbeat -->

Sections are created on first write and always re-assembled in this order,
whatever order the passes ran in. Logs written before the sections existed
(echoes directly under the header) keep working: their old rows stay in the
header region and new rows go under the headings.
"""

from __future__ import annotations

import re
from pathlib import Path

from openaugi.model.block import Block

DAILY_PREFIX = "_private/0-Fleeting-Inbox/"
LOG_FOLDER_FMT = "OpenAugi/{y}/{m:02d}/{d:02d}"
LOG_NAME = "Augi Log.md"

ROUTING_HEADING = "## Routing"
ECHO_HEADING = "## Echoes"
QUIET_HEADING = "## Quiet — closest match, not surfaced"
#: Canonical order. A section absent from the file is simply omitted.
SECTIONS = (ROUTING_HEADING, ECHO_HEADING, QUIET_HEADING)

QUIET_INTRO = "*Debug view — what was closest when augi stayed silent. Tick a box to teach it.*"

# ── Eligibility (deterministic, shared by every pass) ──────────────
MIN_PROSE_CHARS = 40
_WIKILINK_RE = re.compile(r"\[\[[^\]]*\]\]")
_MD_NOISE_RE = re.compile(r"[#>*_`\-\[\]()]|https?://\S+")
_INSTRUCTION_RE = re.compile(r"^\s*zzz\s*:", re.IGNORECASE | re.MULTILINE)


def prose_len(content: str) -> int:
    """Length of real prose, ignoring wikilinks and markdown punctuation."""
    stripped = _WIKILINK_RE.sub("", content or "")
    return len(_MD_NOISE_RE.sub("", stripped).strip())


def is_capture_block(block: Block) -> bool:
    """A daily-note block with real prose that is not a `zzz:` instruction.

    Cheap checks before any retrieval or LLM. A bare [[link]] scored 1.000
    against another bare link in the echo replay, hence the prose floor;
    dispatch owns zzz blocks — they are asks, not thoughts.
    """
    path = (block.metadata or {}).get("source_path") or ""
    if not path.startswith(DAILY_PREFIX):
        return False
    content = block.content or ""
    if prose_len(content) < MIN_PROSE_CHARS:
        return False
    return not ((block.metadata or {}).get("zzz_instructions") or _INSTRUCTION_RE.search(content))


# ── Path and header ────────────────────────────────────────────────


def log_path(vault_path: Path, day: str) -> Path:
    y, m, d = int(day[:4]), int(day[5:7]), int(day[8:10])
    return vault_path / LOG_FOLDER_FMT.format(y=y, m=m, d=d) / LOG_NAME


def ensure_log(path: Path, day: str) -> str:
    """Create the day's log with its header if missing. Returns the text."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "---\ntype: document\n"
        f"description: Routing rows and proactive echoes from what you wrote on {day}. "
        "Answer with checkboxes; tick 'process this log' to apply. Safe to delete — "
        "an unprocessed log is simply never applied.\n"
        f"created: {day}\n---\n\n"
        f"# Augi Log — {day}\n\n- [ ] seen\n\n"
        "*Appended as you write. Read when you want; an unanswered log is never applied.*\n"
    )
    path.write_text(header, encoding="utf-8")
    return header


# ── Sections ───────────────────────────────────────────────────────

_HEARTBEAT_RE = re.compile(r"\n<!-- heartbeat .*", re.DOTALL)


def split(text: str) -> tuple[str, dict[str, str], str]:
    """(head, {heading: body}, heartbeat). Bodies exclude their heading line."""
    hb = _HEARTBEAT_RE.search(text)
    body_text, heartbeat = (text[: hb.start()], text[hb.start() :]) if hb else (text, "")
    positions: list[tuple[int, int, str]] = []
    for heading in SECTIONS:
        match = re.search(rf"(?m)^{re.escape(heading)}[ \t]*$", body_text)
        if match:
            positions.append((match.start(), match.end(), heading))
    positions.sort()
    head = body_text[: positions[0][0]] if positions else body_text
    sections: dict[str, str] = {}
    for i, (_start, end, heading) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(body_text)
        sections[heading] = body_text[end:stop]
    return head, sections, heartbeat


def assemble(head: str, sections: dict[str, str], heartbeat: str) -> str:
    out = head.rstrip("\n") + "\n"
    for heading in SECTIONS:
        if heading in sections:
            out += f"\n{heading}\n\n{sections[heading].strip(chr(10))}\n"
    if heartbeat.strip():
        out += "\n" + heartbeat.strip("\n") + "\n"
    return out


def append_to_section(path: Path, heading: str, md: str, intro: str = "") -> None:
    """Splice `md` into `heading`, creating the section (with `intro`) if new.

    The heartbeat stays last and the other sections keep their order, however
    many passes have written since the file was created.
    """
    if heading not in SECTIONS:
        raise ValueError(f"Unknown Augi Log section: {heading}")
    head, sections, heartbeat = split(path.read_text(encoding="utf-8"))
    if heading in sections:
        sections[heading] = sections[heading].rstrip("\n") + "\n" + md
    else:
        sections[heading] = (intro + "\n" if intro else "") + md
    path.write_text(assemble(head, sections, heartbeat), encoding="utf-8")


# ── Heartbeat ──────────────────────────────────────────────────────

_HEARTBEAT_COUNTS_RE = re.compile(r"<!-- heartbeat (\d+) (\d+) (\d+) -->")


def write_heartbeat(path: Path, stats: dict[str, int]) -> None:
    """Keep a running per-day tally at the very end of the log.

    Silence has to be legible: a system that never reports its own quiet is
    indistinguishable from one that is broken.
    """
    if not path.exists():
        return
    head, sections, heartbeat = split(path.read_text(encoding="utf-8"))
    watched, spoke, quiet = stats["watched"], stats["spoke"], stats["quiet"]
    if prior := _HEARTBEAT_COUNTS_RE.search(heartbeat):
        watched += int(prior.group(1))
        spoke += int(prior.group(2))
        quiet += int(prior.group(3))
    heartbeat = (
        f"<!-- heartbeat {watched} {spoke} {quiet} -->\n"
        f"*watched {watched} · spoke {spoke} · quiet {quiet}*\n"
    )
    path.write_text(assemble(head, sections, heartbeat), encoding="utf-8")
