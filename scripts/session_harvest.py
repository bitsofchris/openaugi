#!/usr/bin/env python3
"""session_harvest.py — pull one day of *my own* turns out of AI chat transcripts.

`session_cards.py` answers "which sessions did I have, and where did each leave
off". This answers a different question: **what did I actually say yesterday?**
It walks the same local Claude Code (~/.claude/projects) and Codex
(~/.codex/sessions) transcript stores, keeps only human turns whose timestamp
falls inside a local-time day window, and prints them grouped by session with a
short slice of the reply that followed each one.

It extracts; it does not judge. Deciding which turns are worth keeping in the
second brain is the `chat-harvest` lens's job
(`<vault>/OpenAugi/AGENT/lenses/chat-harvest.md`), which reads this output.

Stdlib only. Usage:

    python3 scripts/session_harvest.py                     # yesterday, local time
    python3 scripts/session_harvest.py --day 2026-09-07
    python3 scripts/session_harvest.py --days 3 --json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path

# First human turn of a session dispatched by the OpenAugi task watcher — augi
# talking to itself, not Chris thinking out loud. Matched against the session's
# opening turn only, so a session that merely mentions the phrase is kept.
AGENT_PROMPT_MARKERS = ("Read your skill file first",)

# Turns that are mechanically uninteresting no matter what they say.
TRIVIAL_TURNS = re.compile(
    r"^(y|n|yes|no|ok|okay|k|sure|thanks|ty|go|do it|continue|next|stop|"
    r"yep|yeah|nope|please|proceed|good|great|nice|perfect|done|hm+|\?+|\.+)$",
    re.IGNORECASE,
)


@dataclass
class Turn:
    """One human message inside the window, plus the reply it drew.

    `reply` is the LONGEST assistant message before the next human turn, not the
    first — the first is usually a one-line preamble ("let me check X"), while
    the substance lands a few tool calls later.
    """

    timestamp: str
    text: str
    reply: str = ""


@dataclass
class SessionSlice:
    """The part of one session that falls inside the window."""

    tool: str
    session_id: str
    source_path: str
    project: str = ""
    title: str = ""
    opening_turn: str = ""
    total_human_turns: int = 0
    turns: list[Turn] = field(default_factory=list)

    @property
    def project_name(self) -> str:
        return Path(self.project).name if self.project else "?"

    @property
    def is_agent_dispatched(self) -> bool:
        opening = self.opening_turn.lstrip()
        return any(opening.startswith(m) for m in AGENT_PROMPT_MARKERS)

    @property
    def substantive_turns(self) -> list[Turn]:
        return [t for t in self.turns if not TRIVIAL_TURNS.match(t.text.strip())]

    @property
    def resume_command(self) -> str:
        if self.tool == "claude":
            return f'cd "{self.project or "."}" && claude --resume {self.session_id}'
        return f"codex resume {self.session_id}"


# --- transcript parsing ----------------------------------------------------


def _text_of(content) -> str:
    """Extract plain text from a message content field (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _is_noise(text: str) -> bool:
    """Harness-injected user turns that aren't the human speaking."""
    t = text.lstrip()
    return (
        not t
        or t.startswith("<")  # <command-name>, <system-reminder>, <task-notification>
        or t.startswith("Caveat:")
    )


def _local(ts: str) -> datetime | None:
    """Parse a transcript timestamp into local time. None if unparseable."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:  # naive timestamps are already local
        return parsed
    return parsed.astimezone().replace(tzinfo=None)


def _iter_records(path: Path):
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _add_turn(s: SessionSlice, text: str, when: datetime | None, start, end) -> None:
    s.total_human_turns += 1
    if not s.opening_turn:
        s.opening_turn = text
    if when is not None and start <= when < end:
        s.turns.append(Turn(timestamp=when.strftime("%Y-%m-%d %H:%M"), text=text))


def parse_claude_slice(path: Path, start: datetime, end: datetime) -> SessionSlice | None:
    s = SessionSlice(tool="claude", session_id=path.stem, source_path=str(path))
    open_turn: Turn | None = None
    for obj in _iter_records(path):
        if obj.get("type") == "ai-title" and obj.get("aiTitle"):
            s.title = str(obj["aiTitle"])
            continue
        if obj.get("isSidechain"):  # subagent chatter, never the human
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        when = _local(obj.get("timestamp"))
        if obj.get("type") == "user" and msg.get("role") == "user":
            origin = obj.get("origin")
            if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
                continue
            text = _text_of(msg.get("content"))
            if _is_noise(text):
                continue
            if not s.project:
                s.project = obj.get("cwd", "") or s.project
            before = len(s.turns)
            _add_turn(s, text, when, start, end)
            open_turn = s.turns[-1] if len(s.turns) > before else None
        elif obj.get("type") == "assistant" and msg.get("role") == "assistant":
            if open_turn is not None:
                reply = _text_of(msg.get("content")).strip()
                if len(reply) > len(open_turn.reply):
                    open_turn.reply = reply
    return s if s.turns else None


def parse_codex_slice(path: Path, start: datetime, end: datetime) -> SessionSlice | None:
    s = SessionSlice(tool="codex", session_id=path.stem, source_path=str(path))
    open_turn: Turn | None = None
    for obj in _iter_records(path):
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        when = _local(obj.get("timestamp"))
        if obj.get("type") == "session_meta":
            s.session_id = payload.get("session_id") or payload.get("id") or s.session_id
            s.project = payload.get("cwd", "") or s.project
            continue
        text = ""
        if payload.get("type") == "message" and payload.get("role") == "user":
            text = _text_of(payload.get("content"))
        elif payload.get("type") == "user_message":
            text = _text_of(payload.get("message"))
        if text and not _is_noise(text):
            before = len(s.turns)
            _add_turn(s, text, when, start, end)
            open_turn = s.turns[-1] if len(s.turns) > before else None
        elif payload.get("type") == "message" and payload.get("role") == "assistant":
            if open_turn is not None:
                reply = _text_of(payload.get("content")).strip()
                if len(reply) > len(open_turn.reply):
                    open_turn.reply = reply
    return s if s.turns else None


# --- collection and rendering ----------------------------------------------


def collect(
    start: datetime,
    end: datetime,
    claude_dir: Path,
    codex_dir: Path,
) -> list[SessionSlice]:
    """Every session slice overlapping [start, end), newest turn last."""
    # A session file is only worth opening if it was touched at or after the
    # window opened; mtime is the cheap pre-filter, timestamps are the truth.
    cutoff = start.timestamp()
    slices: list[SessionSlice] = []
    if claude_dir.is_dir():
        for f in sorted(claude_dir.glob("*/*.jsonl")):
            if f.stat().st_mtime < cutoff:
                continue
            parsed = parse_claude_slice(f, start, end)
            if parsed:
                slices.append(parsed)
    if codex_dir.is_dir():
        for f in sorted(codex_dir.rglob("rollout-*.jsonl")):
            if f.stat().st_mtime < cutoff:
                continue
            parsed = parse_codex_slice(f, start, end)
            if parsed:
                slices.append(parsed)
    return sorted(slices, key=lambda s: s.turns[0].timestamp)


def _flatten(text: str) -> str:
    """One line, words untouched — a paragraph break becomes ` / `.

    Turns are often dictated or pasted, so they arrive full of blank lines. The
    digest is read as markdown bullets; keeping the newlines would shred it.
    """
    text = re.sub(r"\n\s*\n+", " / ", text.strip())
    return re.sub(r"\s+", " ", text)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _trim(text: str, max_len: int) -> str:
    text = text.strip()
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def select(
    slices: list[SessionSlice],
    *,
    min_turns: int,
    include_agent_sessions: bool,
) -> list[SessionSlice]:
    kept = []
    for s in slices:
        if not include_agent_sessions and s.is_agent_dispatched:
            continue
        if len(s.substantive_turns) < min_turns:
            continue
        kept.append(s)
    return kept


def render_markdown(
    slices: list[SessionSlice],
    start: datetime,
    end: datetime,
    *,
    turn_chars: int,
    reply_chars: int,
) -> str:
    window = f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} (local)"
    turn_total = sum(len(s.substantive_turns) for s in slices)
    lines = [
        f"# Chat harvest — {window}",
        "",
        f"{len(slices)} session(s), {turn_total} substantive human turn(s). "
        "Human turns verbatim (trimmed); replies are a slice for context only.",
        "",
    ]
    if not slices:
        lines.append("*No sessions in this window.*")
        return "\n".join(lines) + "\n"
    for s in slices:
        title = s.title or _trim(_first_line(s.opening_turn), 70)
        lines += [
            f"## {title or s.session_id[:8]}",
            "",
            f"`{s.project_name}` · {s.tool} · session `{s.session_id[:8]}` · "
            f"{len(s.substantive_turns)} turn(s) in window of {s.total_human_turns} total",
            "",
        ]
        for t in s.substantive_turns:
            lines.append(f"- **{t.timestamp}** — {_trim(_flatten(t.text), turn_chars)}")
            if reply_chars > 0 and t.reply:
                lines.append(f"    ↳ *reply:* {_trim(_flatten(t.reply), reply_chars)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_json(
    slices: list[SessionSlice],
    start: datetime,
    end: datetime,
    *,
    turn_chars: int,
    reply_chars: int,
) -> str:
    payload = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "sessions": [
            {
                "tool": s.tool,
                "session_id": s.session_id,
                "project": s.project,
                "project_name": s.project_name,
                "title": s.title,
                "source_path": s.source_path,
                "total_human_turns": s.total_human_turns,
                "resume": s.resume_command,
                "turns": [
                    {
                        "timestamp": t.timestamp,
                        "text": _trim(_flatten(t.text), turn_chars),
                        "reply": _trim(_flatten(t.reply), reply_chars) if reply_chars > 0 else "",
                    }
                    for t in s.substantive_turns
                ],
            }
            for s in slices
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def resolve_window(
    day: str | None, days: int, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """Local-time window. `--day` pins one calendar day; `--days N` ends today."""
    now = now or datetime.now()
    if day:
        anchor = datetime.strptime(day, "%Y-%m-%d").date()
        start = datetime.combine(anchor, time.min)
        return start, start + timedelta(days=1)
    end = datetime.combine(now.date(), time.min)  # today 00:00 — yesterday and back
    return end - timedelta(days=days), end


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", help="one local calendar day, YYYY-MM-DD (overrides --days)")
    ap.add_argument("--days", type=int, default=1, help="how many days back from today 00:00")
    ap.add_argument("--claude-dir", default="~/.claude/projects")
    ap.add_argument("--codex-dir", default="~/.codex/sessions")
    ap.add_argument("--turn-chars", type=int, default=1500, help="max chars per human turn")
    ap.add_argument(
        "--reply-chars", type=int, default=350, help="max chars of reply; 0 drops replies"
    )
    ap.add_argument(
        "--min-turns", type=int, default=2, help="skip sessions with fewer substantive turns"
    )
    ap.add_argument(
        "--include-agent-sessions",
        action="store_true",
        help="keep sessions augi dispatched to itself (excluded by default)",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="write to this path instead of stdout")
    args = ap.parse_args()

    start, end = resolve_window(args.day, args.days)
    slices = collect(
        start,
        end,
        Path(args.claude_dir).expanduser(),
        Path(args.codex_dir).expanduser(),
    )
    slices = select(
        slices,
        min_turns=args.min_turns,
        include_agent_sessions=args.include_agent_sessions,
    )
    render = render_json if args.json else render_markdown
    text = render(slices, start, end, turn_chars=args.turn_chars, reply_chars=args.reply_chars)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
        print(f"wrote {len(slices)} session(s) -> {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
