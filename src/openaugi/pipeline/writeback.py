"""Write-back — the one place the janitors agree on how a tick is read and logged.

Every surface that asks a question in the vault answers it the same way: a
checkbox under the thing, an optional `aaa:` line beneath it, and an append to
`OpenAugi/Capture/feedback-log.ndjson`. The board, the Augi Log's echoes and
its routing rows each grew that machinery independently, so the log path was
spelled out in four modules, `append_feedback` was byte-identical in three,
`now()` in four, and the tolerant ndjson read loop in three more.

This module owns all of it. What it deliberately does **not** own is the
*vocabulary*: `done / not doing / someday` belongs to the board, `promote /
good match / bad match` to the echo log, the routing verbs to routing. Each
surface passes its own labels in.

The grammar builders take strictness as arguments rather than imposing one
pattern, because the surfaces really do differ — a board is written inside a
callout where every line carries a `> ` prefix, a routing row is plain
markdown. Sharing the *shape* removes the duplication; forcing one regex would
quietly widen what each surface matches.

No LLM calls in this module.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: The single append-only stream every surface's decisions land in. Rows are
#: distinguished by their `source` field, never by living in separate files.
FEEDBACK_LOG = "OpenAugi/Capture/feedback-log.ndjson"


def now() -> str:
    """The timestamp every log row is stamped with: UTC, ISO 8601."""
    return datetime.now(UTC).isoformat()


# ── The log ────────────────────────────────────────────────────────


def append_feedback(vault_path: Path, record: dict) -> None:
    """Append one decision to the feedback log, creating the folder if needed."""
    path = vault_path / FEEDBACK_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def read_feedback(vault_path: Path, *, source: str | None = None) -> Iterator[dict]:
    """Every row in the log, oldest first, skipping what cannot be read.

    The log is shared with the mobile app and is edited by hand often enough
    that a blank or truncated line must not stop the replay — a foreign line is
    not our business. Pass `source` to read only one surface's rows.
    """
    path = vault_path / FEEDBACK_LOG
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if source is not None and row.get("source") != source:
            continue
        yield row


# ── The tick grammar ───────────────────────────────────────────────

#: Leading callout marker and indent — `> `, `  `, `  > `, or nothing.
_CALLOUT = r"(?P<pre>\s*>?\s*)"
#: A trailing ` — why this one` annotation the surface renders but never reads.
_SUFFIX = r"(?: — .*)?"


def box_re(
    *labels: str,
    callout: bool = False,
    suffix: bool = False,
    multiline: bool = False,
) -> re.Pattern[str]:
    """`- [x] done` — one checkbox, with the caller's vocabulary.

    Groups: `pre` (the indent/callout prefix, empty when `callout` is off),
    `mark` (a space or an `x` in either case) and `label`. With no `labels`
    the pattern accepts any label, which is what routing needs — its verbs
    carry `[[targets]]` and cannot be enumerated.

    `callout` allows the `> ` prefix a board line carries; `suffix` allows a
    trailing ` — …` annotation; `multiline` anchors per line for `finditer`
    over a block of text rather than a single line.
    """
    pre = _CALLOUT if callout else r"(?P<pre>)"
    label = "|".join(re.escape(one) for one in labels) if labels else ".+?"
    pattern = rf"^{pre}- \[(?P<mark>[ xX])\] (?P<label>{label})"
    pattern += (_SUFFIX if suffix else "") + r"\s*$"
    return re.compile(pattern, re.MULTILINE if multiline else 0)


def ticked(match: re.Match[str]) -> bool:
    """True when this box was ticked. `[x]` and `[X]` both count."""
    return match.group("mark").lower() == "x"


def aaa_re(
    *,
    callout: bool = False,
    indent: bool = False,
    multiline: bool = False,
    require_text: bool = False,
    spaced_colon: bool = False,
) -> re.Pattern[str]:
    """`aaa: because it can wait until October` — the comment channel.

    Always case-insensitive and always exposes the comment as `reason`; an
    empty `aaa:` placeholder matches with an empty reason unless `require_text`
    is set. `callout` allows a board's `> ` prefix (and implies `indent`);
    `spaced_colon` also accepts `aaa :`, which is what a phone keyboard tends
    to produce.
    """
    pre = _CALLOUT if callout else (r"\s*" if indent else "")
    colon = r"\s*:" if spaced_colon else ":"
    text = ".+?" if require_text else ".*?"
    pattern = rf"^{pre}aaa{colon}\s*(?P<reason>{text})\s*$"
    flags = re.IGNORECASE | (re.MULTILINE if multiline else 0)
    return re.compile(pattern, flags)
