"""scripts/sync_templates.py — the vault's engine files become the templates."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "sync_templates", Path(__file__).parent.parent / "scripts" / "sync_templates.py"
)
assert _SPEC and _SPEC.loader
sync = importlib.util.module_from_spec(_SPEC)
sys.modules["sync_templates"] = sync
_SPEC.loader.exec_module(sync)


def _vault(tmp_path: Path, files: dict[str, str]) -> Path:
    vault = tmp_path / "vault"
    for rel, text in files.items():
        p = vault / "OpenAugi" / "AGENT" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return vault


ENGINE = (
    "---\nname: a\nkind: engine\n---\n\n- [ ] seen\n\nrule\n"
    "%% personal %%\nmine\n%% /personal %%\n"
)
PERSONAL = "---\nname: p\nkind: personal\n---\n\nmy taxonomy\n"


def test_plan_classifies_every_outcome(tmp_path: Path):
    vault = _vault(
        tmp_path,
        {
            "a.md": ENGINE,
            "p.md": PERSONAL,
            "lenses/nokind.md": "---\nname: n\n---\n\n?",
            "lenses/broken.md": "---\nkind: engine\n---\n%% personal %%\nnever closed\n",
        },
    )
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "orphan.md").write_text("---\nkind: engine\n---\nold\n")
    (templates / "task-template.md").write_text("---\nstatus: pending\n---\n")
    report = sync.plan(vault, templates)
    assert report["new"] == ["a.md"]
    assert report["personal"] == ["p.md"]
    assert report["no_kind"] == ["lenses/nokind.md"]
    assert report["bad_region"] and report["bad_region"][0].startswith("lenses/broken.md")
    assert report["orphan_template"] == ["orphan.md"]


def test_write_then_check_is_clean(tmp_path: Path):
    vault = _vault(tmp_path, {"a.md": ENGINE, "p.md": PERSONAL})
    templates = tmp_path / "templates"
    assert sync.write(vault, templates) == ["a.md"]
    shipped = (templates / "a.md").read_text()
    assert "mine" not in shipped and "seen" not in shipped and "rule" in shipped
    assert not (templates / "p.md").exists()
    report = sync.plan(vault, templates)
    assert report["unchanged"] == ["a.md"] and not report["drift"] and not report["new"]
    assert sync.write(vault, templates) == []  # nothing to do the second time


def test_edit_in_vault_shows_as_drift(tmp_path: Path):
    vault = _vault(tmp_path, {"a.md": ENGINE})
    templates = tmp_path / "templates"
    sync.write(vault, templates)
    (vault / "OpenAugi" / "AGENT" / "a.md").write_text(ENGINE.replace("rule", "new rule"))
    assert sync.plan(vault, templates)["drift"] == ["a.md"]


def test_main_check_exits_nonzero_on_missing_kind(tmp_path: Path, monkeypatch, capsys):
    vault = _vault(tmp_path, {"a.md": "---\nname: a\n---\nno kind\n"})
    monkeypatch.setattr(sync, "TEMPLATES", tmp_path / "templates")
    (tmp_path / "templates").mkdir()
    assert sync.main(["--check", "--vault", str(vault)]) == 1
    assert "no_kind" in capsys.readouterr().out
