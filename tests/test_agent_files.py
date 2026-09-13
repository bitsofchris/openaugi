"""The engine / personal boundary — every agent file is on one side of it.

Two things are enforced on the shipped tree: every template under
``src/openaugi/templates/`` declares ``kind: engine`` (the one exception is the
task-file contract, which the code reads and never copies into a vault), and no
template still carries a personal region marker — a half-stripped ruling would
mean the sync script and the vault disagree about where the line is.
"""

from pathlib import Path

import pytest

from openaugi.agent_files import (
    ENGINE,
    NON_AGENT_TEMPLATES,
    PERSONAL_CLOSE,
    PERSONAL_OPEN,
    PersonalRegionError,
    engine_templates,
    iter_templates,
    read_kind,
    split_frontmatter,
    strip_personal,
    template_description,
    to_template,
)

TEMPLATES = Path(__file__).parent.parent / "src" / "openaugi" / "templates"
SHIPPED = iter_templates(TEMPLATES)

ENGINE_FILE = (
    "---\nname: sample\nkind: engine\ndescription: >-\n  A rule file.\n---\n"
    '- [ ] seen\n\n# Sample\n\nThe rule.\n\n%% personal %%\nChris, 2026-08-20: *"the ruling"*\n'
    "%% /personal %%\n\nMore rule.\n"
)


# ── read_kind ────────────────────────────────────────────────────────────────


def test_read_kind_engine_and_personal():
    assert read_kind("---\nkind: engine\n---\nbody") == ENGINE
    assert read_kind("---\nname: x\nkind: personal\n---\n") == "personal"


def test_read_kind_missing_or_no_frontmatter():
    assert read_kind("---\nname: x\n---\nbody") is None
    assert read_kind("# no frontmatter\nkind: engine\n") is None


# ── strip_personal ───────────────────────────────────────────────────────────


def test_strip_personal_removes_region_and_markers():
    out = strip_personal(ENGINE_FILE)
    assert "ruling" not in out
    assert PERSONAL_OPEN not in out and PERSONAL_CLOSE not in out
    assert "The rule.\n\nMore rule.\n" in out  # no triple blank left behind


def test_strip_personal_no_regions_is_identity():
    text = "---\nkind: engine\n---\n\nplain\n"
    assert strip_personal(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "%% personal %%\nnever closed\n",
        "%% /personal %%\nclosed first\n",
        "%% personal %%\n%% personal %%\nnested\n%% /personal %%\n",
    ],
)
def test_strip_personal_rejects_unbalanced(text):
    with pytest.raises(PersonalRegionError):
        strip_personal(text)


# ── to_template ──────────────────────────────────────────────────────────────


def test_to_template_drops_seen_tick_and_regions():
    out = to_template(ENGINE_FILE)
    assert out.startswith("---\nname: sample\nkind: engine\n")
    assert "seen" not in out
    assert "ruling" not in out
    assert "# Sample" in out


def test_to_template_refuses_personal_and_unkinded():
    with pytest.raises(ValueError):
        to_template("---\nkind: personal\n---\nmine\n")
    with pytest.raises(ValueError):
        to_template("---\nname: x\n---\nnobody said\n")


def test_to_template_is_idempotent():
    once = to_template(ENGINE_FILE)
    assert to_template(once) == once


# ── template_description ─────────────────────────────────────────────────────


def test_template_description_plain_and_folded():
    assert template_description("---\ndescription: One line.\n---\n") == "One line."
    folded = "---\nname: x\ndescription: >-\n  First line\n  second line\n---\n"
    assert template_description(folded) == "First line"
    assert template_description("no frontmatter") == ""


# ── the shipped tree ─────────────────────────────────────────────────────────


def test_shipped_templates_found():
    assert len(SHIPPED) > 10
    assert any(rel == "augi-agent.md" for rel, _ in SHIPPED)


@pytest.mark.parametrize("rel,text", SHIPPED, ids=[rel for rel, _ in SHIPPED])
def test_every_shipped_template_declares_kind_engine(rel, text):
    if Path(rel).name in NON_AGENT_TEMPLATES:
        assert read_kind(text) is None, f"{rel} is code-read data, not an agent file"
        return
    assert read_kind(text) == ENGINE, f"{rel}: templates ship only engine files"


@pytest.mark.parametrize("rel,text", SHIPPED, ids=[rel for rel, _ in SHIPPED])
def test_no_personal_region_survives_in_a_template(rel, text):
    assert PERSONAL_OPEN not in text and PERSONAL_CLOSE not in text, rel
    _, body = split_frontmatter(text)
    assert not body.lstrip().startswith("- ["), f"{rel}: a review tick is vault state"


def test_init_copies_exactly_the_engine_templates():
    copied = {rel for rel, _ in engine_templates(TEMPLATES)}
    assert "augi-agent.md" in copied
    assert "lenses/distill.md" in copied
    assert not copied & NON_AGENT_TEMPLATES
