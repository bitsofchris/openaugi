"""Agent files — the vault's ``OpenAugi/AGENT/`` folder and the templates that seed it.

The engine / personal line, made executable. Every file under the vault's
``AGENT/`` folder declares ``kind: engine`` or ``kind: personal`` in its
frontmatter:

* **engine** — part of the operating system anyone who installs OpenAugi runs.
  It has a template twin under ``src/openaugi/templates/`` (written by
  ``scripts/sync_templates.py``, copied into a fresh vault by ``openaugi init``)
  and it must not name, gender, or describe its user.
* **personal** — the user's own configuration: their taxonomy, context, repos,
  and the lenses only their life needs. Never shipped.

An engine file may still carry the user's own rulings — the dated quote that
explains why a rule exists, the table naming their areas. Those stay in the
vault copy inside a **personal region** and are stripped from the template::

    %% personal %%
    Ruling, 2026-08-20: *"I don't want you to guess anything…"*
    %% /personal %%

The markers are Obsidian comments, so the region is invisible in reading view
and plain text to an agent reading the raw file.
"""

from __future__ import annotations

import re
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

KINDS = ("engine", "personal")
ENGINE = "engine"
PERSONAL = "personal"

PERSONAL_OPEN = "%% personal %%"
PERSONAL_CLOSE = "%% /personal %%"

#: Files under ``templates/`` that are engine data but not agent files — they
#: are read by the code (``dispatch.py`` hydrates the task template) and never
#: copied into a vault's ``AGENT/`` folder.
NON_AGENT_TEMPLATES = frozenset({"task-template.md"})

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_KIND_RE = re.compile(r"^kind:[ \t]*([A-Za-z]+)[ \t]*(?:#.*)?$", re.MULTILINE)
_SEEN_LINE_RE = re.compile(r"\A- \[[ xX]\] seen[ \t]*(?:\r?\n|\Z)")


class PersonalRegionError(ValueError):
    """The personal markers in a file do not pair up."""


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Return ``(frontmatter_body, rest)``; frontmatter is None when absent."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end() :]


def read_kind(text: str) -> str | None:
    """The ``kind:`` declared in a file's frontmatter, or None when it has none."""
    fm, _ = split_frontmatter(text)
    if fm is None:
        return None
    m = _KIND_RE.search(fm)
    return m.group(1).lower() if m else None


def strip_personal(text: str) -> str:
    """Remove every ``%% personal %% … %% /personal %%`` region, markers included.

    Regions are whole lines. Nesting and dangling markers are errors — a
    template with a half-stripped ruling in it is worse than a failed sync.
    """
    out: list[str] = []
    depth = 0
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        stripped = line.strip()
        if stripped == PERSONAL_OPEN:
            if depth:
                raise PersonalRegionError(f"line {number}: nested {PERSONAL_OPEN}")
            depth = 1
            continue
        if stripped == PERSONAL_CLOSE:
            if not depth:
                raise PersonalRegionError(f"line {number}: {PERSONAL_CLOSE} without an open")
            depth = 0
            continue
        if not depth:
            out.append(line)
    if depth:
        raise PersonalRegionError(f"{PERSONAL_OPEN} never closed")
    joined = "".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)  # a stripped region leaves no gap
    return joined.rstrip("\n") + "\n"  # exactly one trailing newline


def to_template(text: str) -> str:
    """The shipped form of an engine file: personal regions and the review tick gone."""
    kind = read_kind(text)
    if kind != ENGINE:
        raise ValueError(f"only kind: engine files become templates (got {kind!r})")
    fm, rest = split_frontmatter(text)
    rest = rest.lstrip("\r\n")
    rest = _SEEN_LINE_RE.sub("", rest).lstrip("\r\n")  # only a leading tick is vault state
    return strip_personal(f"---\n{fm}\n---\n\n{rest}")


def template_description(text: str) -> str:
    """The first line of the frontmatter ``description:``, for ``init`` output."""
    fm, _ = split_frontmatter(text)
    if fm is None:
        return ""
    m = re.search(r"^description:[ \t]*(.*)$", fm, re.MULTILINE)
    if not m:
        return ""
    first = m.group(1).strip()
    if first in ("", ">", ">-", "|", "|-"):
        # Folded scalar — the description starts on the next indented line.
        after = fm[m.end() :].lstrip("\r\n")
        first = after.splitlines()[0].strip() if after else ""
    return first.strip("\"'")


def _walk(root: Path | Traversable, prefix: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for entry in root.iterdir():
        rel = f"{prefix}{entry.name}"
        if entry.is_dir():
            if entry.name == "__pycache__":
                continue
            found.extend(_walk(entry, f"{rel}/"))
        elif entry.name.endswith(".md"):
            found.append((rel, entry.read_text(encoding="utf-8")))
    return found


def templates_root() -> Traversable:
    """The packaged ``templates/`` folder (a real directory in an editable install)."""
    return resources.files("openaugi") / "templates"


def iter_templates(root: Path | Traversable | None = None) -> list[tuple[str, str]]:
    """Every markdown template as ``(relative_path, text)``, sorted by path."""
    return sorted(_walk(root if root is not None else templates_root()))


def engine_templates(
    root: Path | Traversable | None = None,
) -> list[tuple[str, str]]:
    """The templates ``openaugi init`` copies into ``<vault>/OpenAugi/AGENT/``."""
    return [(rel, text) for rel, text in iter_templates(root) if read_kind(text) == ENGINE]
