#!/usr/bin/env python3
"""fix_frontmatter_links.py — find (and fix) frontmatter that puts several links in one value.

Obsidian links a property only when its whole value is one link. A value like
``sources: "[[a]] · [[b]]"`` is read as one link to a note named
``a]] · [[b``, which then shows up in every ``[[`` suggestion; written
unquoted (``related: [[a]], [[b]]``) it is not even valid YAML. The fix is a
list, one link per item (see ``openaugi.frontmatter_links``).

    .venv/bin/python scripts/fix_frontmatter_links.py            # report, exit 1 if any
    .venv/bin/python scripts/fix_frontmatter_links.py --write    # rewrite them as lists
    .venv/bin/python scripts/fix_frontmatter_links.py --under OpenAugi

Only single-line top-level properties whose value starts with ``[[`` and ends
with ``]]`` are touched (prose that merely mentions a link is fine); prose
between the links is dropped, and the report shows each value before and after. The vault path
comes from the openaugi config unless ``--vault`` is given.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from openaugi.config import load_config, resolve_vault_path
from openaugi.frontmatter_links import WIKILINK_RE, needs_link_list, property_lines

_PROPERTY_RE = re.compile(r"^([A-Za-z0-9_./-]+):[ \t]+(\S.*)$")


def fix_text(text: str) -> tuple[str, list[tuple[str, str, list[str]]]]:
    """Return (new text, [(key, old value, links)]) for one note's content."""
    if not text.startswith("---\n"):
        return text, []
    end = text.find("\n---", 4)
    if end < 0:
        return text, []
    lines = text[4:end].split("\n")
    out: list[str] = []
    changes: list[tuple[str, str, list[str]]] = []
    for line in lines:
        m = _PROPERTY_RE.match(line)
        if m and needs_link_list(m.group(2)):
            key, value = m.group(1), m.group(2)
            links = WIKILINK_RE.findall(value)
            out.extend(property_lines(key, links))
            changes.append((key, value, links))
        else:
            out.append(line)
    if not changes:
        return text, []
    return "---\n" + "\n".join(out) + text[end:], changes


def scan(root: Path) -> list[Path]:
    notes = []
    for path in sorted(root.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        notes.append(path)
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vault", default=None, help="vault root (default: openaugi config)")
    ap.add_argument("--under", default="", help="only notes under this vault-relative folder")
    ap.add_argument("--write", action="store_true", help="rewrite the values as lists")
    args = ap.parse_args()

    raw = resolve_vault_path(args.vault, load_config())
    if not raw:
        ap.error("no vault: pass --vault or set [vault] default_path in the openaugi config")
    vault = Path(raw).expanduser()
    root = vault / args.under

    found = 0
    for path in scan(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new, changes = fix_text(text)
        if not changes:
            continue
        found += len(changes)
        print(path.relative_to(vault))
        for key, old, links in changes:
            print(f"  {key}: {old}")
            print(f"  -> {key}: {links}")
        if args.write:
            path.write_text(new, encoding="utf-8")

    verb = "fixed" if args.write else "found"
    print(f"{found} multi-link frontmatter value(s) {verb}.")
    return 1 if found and not args.write else 0


if __name__ == "__main__":
    sys.exit(main())
