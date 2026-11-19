"""Frontmatter values that hold wikilinks, written the way Obsidian reads them.

Obsidian treats a property as a link only when the whole value is one link.
A string holding several links, such as ``"[[a]] · [[b]]"``, starts with
``[[`` and ends with ``]]``, so Obsidian reads it as ONE link to a note named
``a]] · [[b``. That note does not exist, so it shows up in every link
suggestion. Written unquoted (``key: [[a]], [[b]]``) it is not valid YAML.

The shape that works is a list, one quoted link per item::

    sources:
      - "[[a]]"
      - "[[b]]"

``property_lines`` is the one place that turns a key and value into
frontmatter lines; ``write_document`` and ``scripts/fix_frontmatter_links.py``
both go through it.
"""

from __future__ import annotations

import json
import re

WIKILINK_RE = re.compile(r"\[\[[^\[\]\n]+\]\]")

# A plain scalar that YAML could misread: a leading indicator character, a
# `: ` or ` #` inside, or surrounding whitespace.
_UNSAFE_START = tuple("[]{}>|*&!%@`'\"#,?-:")


def _quote(text: str) -> str:
    # A JSON string is a valid YAML double-quoted scalar.
    return json.dumps(text, ensure_ascii=False)


def scalar(text: str) -> str:
    """One-line YAML scalar for `text`, quoted only when a bare value would break."""
    flat = " ".join(str(text).split())
    if not flat or flat.startswith(_UNSAFE_START) or ": " in flat or " #" in flat:
        return _quote(flat)
    return flat


def needs_link_list(value: str) -> bool:
    """True when Obsidian would read `value` as one bogus link.

    That is a value that starts with ``[[`` and ends with ``]]`` but is not a
    single link. Prose that mentions a link (``see [[a]] for more``) is
    plain text to Obsidian and stays a scalar.
    """
    text = value.strip().strip("\"'").strip()
    if not (text.startswith("[[") and text.endswith("]]")):
        return False
    links = WIKILINK_RE.findall(text)
    return not (len(links) == 1 and links[0] == text)


def property_lines(key: str, value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Frontmatter lines for one property.

    - A list becomes a YAML list, one quoted item per entry.
    - A string Obsidian would misread as one link (starts with ``[[``, ends
      with ``]]``, holds more than one) becomes a list of just its links; any
      prose between them is dropped, since Obsidian could not link it anyway.
    - Anything else is a single scalar, quoted when it needs to be.
    """
    if isinstance(value, list | tuple):
        items = [" ".join(str(v).split()) for v in value]
        items = [i for i in items if i]
        if not items:
            return [f"{key}: []"]
        return [f"{key}:"] + [f"  - {_quote(i)}" for i in items]
    text = str(value)
    if needs_link_list(text):
        return property_lines(key, WIKILINK_RE.findall(text))
    return [f"{key}: {scalar(text)}"]
