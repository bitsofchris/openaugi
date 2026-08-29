"""Echo janitor — processes the checkboxes Chris ticks in an Augi Log.

The log offers three boxes under every echo: promote / good match / bad match.
Ticking one IS the command (the review-pass precedent, 2026-07-15): the janitor
acts on it, then rewrites the line as a confirmation so it is never re-processed.

Feedback lands in the same `OpenAugi/Capture/feedback-log.ndjson` stream the
mobile app wrote, so the one proactive feature that already had signal keeps
accumulating it.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

FEEDBACK_LOG = "OpenAugi/Capture/feedback-log.ndjson"
PROMOTE_FOLDER = "OpenAugi/Notes"

_ECHO_BLOCK_RE = re.compile(
    r"<!-- echo:(?P<bid>[0-9a-f]+) -->\n(?P<body>.*?)(?=\n<!-- echo:|\n<!-- heartbeat|\Z)",
    re.DOTALL,
)
_CHECKED_RE = re.compile(r"^- \[x\] (promote → new note|good match|bad match)\s*$", re.MULTILINE)
_LINK_RE = re.compile(r"^- \[\[([^\]]+)\]\]", re.MULTILINE)


def _append_feedback(vault_path: Path, record: dict) -> None:
    path = vault_path / FEEDBACK_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _promote(vault_path: Path, block_id: str, body: str, day: str) -> str | None:
    """Write the promoted note: context header + append-only dated log body."""
    links = _LINK_RE.findall(body)
    if not links:
        return None
    heading = re.search(r'### echo on "(.*?)…?"', body)
    topic = (heading.group(1) if heading else links[0])[:60].strip()
    slug = re.sub(r"[^\w\s-]", "", topic)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")[:50] or f"echo-{block_id[:8]}"
    path = vault_path / PROMOTE_FOLDER / f"{slug}.md"
    if path.exists():
        return path.stem
    path.parent.mkdir(parents=True, exist_ok=True)
    linked = "\n".join(f"- [[{link}]]" for link in links)
    path.write_text(
        "---\ntype: document\n"
        f"description: Promoted from a proactive echo on {day} — {topic}\n"
        f"created: {day}\n---\n\n"
        f"# {topic}\n\n#human-review\n\n"
        "## Context\n\n"
        f"Promoted from the Augi Log on {day}. The echo connected what was being "
        "written that day to earlier thinking:\n\n"
        f"{linked}\n\n"
        "## Log\n\n"
        f"### {day}\n\n"
        f"{body.strip()}\n",
        encoding="utf-8",
    )
    logger.info(f"Promoted echo {block_id[:8]} → {path.name}")
    return path.stem


def process_log(log_path: Path, vault_path: Path) -> int:
    """Act on every ticked checkbox in one Augi Log. Returns actions taken."""
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8")
    day = log_path.parent.name and "-".join(
        [log_path.parent.parent.parent.name, log_path.parent.parent.name, log_path.parent.name]
    )
    actions = 0
    stamp = datetime.now(UTC).isoformat()

    for match in list(_ECHO_BLOCK_RE.finditer(text)):
        body, bid = match.group("body"), match.group("bid")
        checked = _CHECKED_RE.findall(body)
        if not checked:
            continue
        new_body = body
        for label in checked:
            if label == "promote → new note":
                name = _promote(vault_path, bid, body, day)
                confirm = (
                    f"- ✓ promoted → [[{name}]]" if name else "- ✓ promote skipped (no links)"
                )
            else:
                _append_feedback(
                    vault_path,
                    {
                        "ts": stamp,
                        "source": "proactive-echo",
                        "block_id": bid,
                        "signal": "liked" if label == "good match" else "disliked",
                        "links": _LINK_RE.findall(body),
                    },
                )
                confirm = f"- ✓ feedback recorded ({label})"
            new_body = new_body.replace(f"- [x] {label}", confirm)
            actions += 1
        # drop the remaining unticked boxes for this echo — it has been answered
        new_body = re.sub(
            r"^- \[ \] (?:promote → new note|good match|bad match)\s*$\n?",
            "",
            new_body,
            flags=re.MULTILINE,
        )
        text = text.replace(body, new_body)

    if actions:
        log_path.write_text(text, encoding="utf-8")
        logger.info(f"Echo janitor: {actions} action(s) in {log_path.name}")
    return actions


def process_changed(changed_paths: set[str], vault_path: Path) -> int:
    """Janitor entry point for the watcher — handles any touched Augi Log."""
    total = 0
    for raw in changed_paths:
        path = Path(raw)
        if path.name == "Augi Log.md":
            try:
                total += process_log(path, vault_path)
            except Exception as e:
                logger.error(f"Echo janitor failed on {path}: {e}", exc_info=True)
    return total
