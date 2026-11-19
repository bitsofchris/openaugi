"""frontmatter_links + scripts/fix_frontmatter_links.py.

Obsidian links a property only when the whole value is one link; several
links in one string become one link to a note that does not exist.
"""

import importlib.util
from pathlib import Path

import yaml

from openaugi.frontmatter_links import needs_link_list, property_lines, scalar

_SPEC = importlib.util.spec_from_file_location(
    "fix_frontmatter_links", Path(__file__).parent.parent / "scripts" / "fix_frontmatter_links.py"
)
fx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fx)


def test_needs_link_list():
    assert needs_link_list('"[[a]] · [[b]]"')
    assert needs_link_list("[[a]], [[b]]")
    assert needs_link_list("[[a]] line, routed by [[b]]")
    assert not needs_link_list("see [[a]] and [[b]] for more")  # prose, plain text
    assert not needs_link_list('"[[a]] block 37cb"')
    assert not needs_link_list('"[[a]]"')
    assert not needs_link_list("[[a]]")
    assert not needs_link_list("plain text")


def test_property_lines_round_trip_through_yaml():
    lines = property_lines("sources", '"[[a]] · [[b|B]]"')
    assert yaml.safe_load("\n".join(lines)) == {"sources": ["[[a]]", "[[b|B]]"]}
    assert yaml.safe_load("\n".join(property_lines("k", "x: y #z"))) == {"k": "x: y #z"}
    assert property_lines("k", []) == ["k: []"]


def test_scalar_quotes_only_when_needed():
    assert scalar("a summary") == "a summary"
    assert scalar("line one\nline two") == "line one line two"
    assert scalar("[[a]]") == '"[[a]]"'
    assert scalar("- dash") == '"- dash"'


def test_fix_text_rewrites_only_multi_link_values():
    note = (
        "---\n"
        "created: 2022-01-01\n"
        'sources: "[[a]] · [[b]]"\n'
        "related: [[c]], [[d]]\n"
        'parent: "[[e]]"\n'
        "---\n"
        "body with [[a]] · [[b]]\n"
    )
    new, changes = fx.fix_text(note)
    assert [c[0] for c in changes] == ["sources", "related"]
    front = yaml.safe_load(new.split("---\n")[1])
    assert front == {
        "created": __import__("datetime").date(2022, 1, 1),
        "sources": ["[[a]]", "[[b]]"],
        "related": ["[[c]]", "[[d]]"],
        "parent": "[[e]]",
    }
    assert new.endswith("---\nbody with [[a]] · [[b]]\n")  # body untouched


def test_fix_text_leaves_clean_notes_alone():
    note = '---\nsources:\n  - "[[a]]"\n---\nbody\n'
    assert fx.fix_text(note) == (note, [])
    assert fx.fix_text("no frontmatter [[a]] [[b]]") == ("no frontmatter [[a]] [[b]]", [])
