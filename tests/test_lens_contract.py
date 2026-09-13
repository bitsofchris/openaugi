"""The lens contract test — writers and readers agree on lens-template.md.

Every lens template this repo ships MUST validate clean through
read_lens_specs (the reader the context pack and `openaugi lenses` use).
If this test fails, either fix the lens file or — if the contract itself
is changing — update src/openaugi/templates/lens-template.md and the
reader together.
"""

import shutil
from pathlib import Path

import pytest

from openaugi.agent_files import ENGINE, read_kind
from openaugi.pipeline.context_pack import LENS_REQUIRED_KEYS, read_lens_specs

TEMPLATES = Path(__file__).parent.parent / "src" / "openaugi" / "templates"
SHIPPED_LENSES = sorted((TEMPLATES / "lenses").glob("*.md"))


def _registry_with(tmp_path: Path, *files: Path) -> Path:
    lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
    lens_dir.mkdir(parents=True)
    for f in files:
        shutil.copy(f, lens_dir / f.name)
    return tmp_path


def test_shipped_lens_templates_exist():
    assert SHIPPED_LENSES, "no lens templates shipped — templates/lenses/ is empty"


@pytest.mark.parametrize("lens_file", SHIPPED_LENSES, ids=lambda f: f.name)
def test_shipped_lens_honors_contract(lens_file: Path, tmp_path: Path):
    """The regression that would have caught the 2026-07-07 YAML bug."""
    vault = _registry_with(tmp_path, lens_file)
    (spec,) = read_lens_specs(vault)
    assert "error" not in spec, f"{lens_file.name}: {spec.get('error')}"
    for key in ("name", "description", "trigger", "target"):
        assert spec[key], f"{lens_file.name}: empty `{key}`"


@pytest.mark.parametrize("lens_file", SHIPPED_LENSES, ids=lambda f: f.name)
def test_shipped_lens_is_an_engine_file(lens_file: Path):
    """Only `kind: engine` files ship; a lens without it is on nobody's side of the line."""
    assert read_kind(lens_file.read_text(encoding="utf-8")) == ENGINE, lens_file.name


def test_lens_template_itself_validates(tmp_path: Path):
    vault = _registry_with(tmp_path, TEMPLATES / "lens-template.md")
    (spec,) = read_lens_specs(vault)
    assert "error" not in spec, spec.get("error")


def test_missing_required_keys_flagged(tmp_path: Path):
    vault = tmp_path
    lens_dir = vault / "OpenAugi" / "AGENT" / "lenses"
    lens_dir.mkdir(parents=True)
    (lens_dir / "bare.md").write_text("---\nname: bare\ndescription: >-\n  Just a name.\n---\n")
    (spec,) = read_lens_specs(vault)
    assert "error" in spec
    for key in ("scope", "trigger", "target"):
        assert f"missing `{key}`" in spec["error"]
    assert set(LENS_REQUIRED_KEYS) == {"name", "description", "scope", "trigger", "target"}


def test_bad_trigger_flagged(tmp_path: Path):
    lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
    lens_dir.mkdir(parents=True)
    (lens_dir / "weird.md").write_text(
        "---\nname: weird\ndescription: >-\n  X.\nscope: >-\n  Y.\n"
        "trigger: whenever-i-feel-like-it\ntarget: >-\n  dashboard\n---\n"
    )
    (spec,) = read_lens_specs(tmp_path)
    assert "error" in spec and "trigger" in spec["error"]


def test_scheduled_trigger_forms_accepted(tmp_path: Path):
    lens_dir = tmp_path / "OpenAugi" / "AGENT" / "lenses"
    lens_dir.mkdir(parents=True)
    # `every 7d` is the documented form; the colon form works only quoted.
    for i, trig in enumerate(("on-demand", "on-pass", "every 7d", '"every: 7d"')):
        (lens_dir / f"l{i}.md").write_text(
            f"---\nname: l{i}\ndescription: >-\n  X.\nscope: >-\n  Y.\n"
            f"trigger: {trig}\ntarget: >-\n  dashboard\n---\n"
        )
    assert all("error" not in s for s in read_lens_specs(tmp_path))
