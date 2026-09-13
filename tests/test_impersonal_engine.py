"""The engine does not know whose vault it is — the criterion, made executable.

From the decision brief *Where a Personal Surface's Code Lives*: a surface's
trigger, scope and prompt are the user's configuration and belong in the vault
as a lens file. Generic mechanism belongs in the engine. The plainest tell that
a module has drifted the wrong way is prose written *about one person* — a
first name, or "he / his / him" standing in for whoever installed this.

Scope note: second-person "you" is deliberately allowed. It is the product's
own voice — config examples, the LLM prompts, and the lens templates shipped
into a vault all address whoever is reading, and that is generic. What this
test forbids is the engine naming, or gendering, its user.
"""

import re
from pathlib import Path

import pytest

from openaugi import __file__ as package_file

SRC = Path(package_file).parent

#: The name the repo was written around, and the pronouns that stood in for it.
_PERSONAL_RE = re.compile(r"\b(chris(?:'s)?|he|his|him)\b", re.IGNORECASE)


def _shipped_files() -> list[Path]:
    """Every source and template file the package ships."""
    return sorted(
        path
        for suffix in ("*.py", "*.md", "*.toml", "*.json")
        for path in SRC.rglob(suffix)
        if "__pycache__" not in path.parts
    )


def test_there_are_files_to_check():
    """Guard against the glob silently finding nothing and the suite passing."""
    files = _shipped_files()
    assert len(files) > 50
    assert any(path.name == "board_janitor.py" for path in files)


@pytest.mark.parametrize("path", _shipped_files(), ids=lambda p: p.name)
def test_no_file_names_or_genders_its_user(path):
    offenders = [
        f"{path.relative_to(SRC)}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _PERSONAL_RE.search(line)
    ]
    assert not offenders, (
        "The engine is written about one person. Rewrite for whoever installs "
        "it — 'the user', 'they', or second person:\n  " + "\n  ".join(offenders)
    )
