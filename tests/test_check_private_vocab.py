"""Tests for scripts/check_private_vocab.py — the private-vocabulary and notebook-output guard.

The word list under test is a throwaway (`zebra`, a regex); the real one
lives in the vault and never in this repo.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_private_vocab.py"
spec = importlib.util.spec_from_file_location("check_private_vocab", SCRIPT)
cpv = importlib.util.module_from_spec(spec)
sys.modules["check_private_vocab"] = cpv
spec.loader.exec_module(cpv)


def _denylist(tmp_path: Path) -> Path:
    p = tmp_path / "words.txt"
    p.write_text("# comment line\n\nzebra\nre:\\bqu+x\\b\n")
    return p


def test_load_rules_skips_comments_and_blanks_and_reads_regex(tmp_path):
    rules = cpv.load_rules(_denylist(tmp_path))
    assert [r.pattern for r in rules] == ["zebra", r"\bqu+x\b"]
    assert rules[0].search("A ZEBRA crossed")  # case-insensitive
    assert rules[1].search("quuux here") and not rules[1].search("quuxes")


def test_scan_file_masks_the_match(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("fine line\nthe Zebra and the quux met\n")
    rules = cpv.load_rules(_denylist(tmp_path))
    out = cpv.scan_file(f, rules, label="note.md")
    assert out == [f"note.md:2: the {cpv.MASK} and the {cpv.MASK} met"]
    assert "zebra" not in out[0].lower()


def test_scan_file_skips_binary_and_missing(tmp_path):
    b = tmp_path / "blob.bin"
    b.write_bytes(b"zebra\0zebra")
    rules = cpv.load_rules(_denylist(tmp_path))
    assert cpv.scan_file(b, rules) == []
    assert cpv.scan_file(tmp_path / "missing.md", rules) == []


def _notebook(path: Path, outputs: list) -> Path:
    path.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": ["1+1"], "outputs": outputs}]})
    )
    return path


def test_notebook_with_outputs_is_refused_even_without_a_word_list(tmp_path):
    with_out = _notebook(tmp_path / "a.ipynb", [{"output_type": "stream", "text": ["2"]}])
    clean = _notebook(tmp_path / "b.ipynb", [])
    assert cpv.scan_file(with_out, []) == [
        f"{with_out}: notebook has cell outputs — clear them before committing"
    ]
    assert cpv.scan_file(clean, []) == []


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(cpv.ENV_VAR, raising=False)
    words = _denylist(tmp_path)
    (tmp_path / "ok.md").write_text("nothing to see\n")
    (tmp_path / "bad.md").write_text("a zebra\n")
    assert cpv.main(["ok.md", "--denylist", str(words)]) == 0
    assert cpv.main(["ok.md", "bad.md", "--denylist", str(words)]) == 1
    assert cpv.MASK in capsys.readouterr().out


def test_env_var_names_the_list_and_missing_list_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.md").write_text("a zebra\n")
    monkeypatch.setenv(cpv.ENV_VAR, str(_denylist(tmp_path)))
    assert cpv.main(["bad.md"]) == 1
    monkeypatch.setenv(cpv.ENV_VAR, str(tmp_path / "nope.txt"))
    assert cpv.main(["bad.md"]) == 0
    assert "vocabulary check skipped" in capsys.readouterr().err


def test_all_walks_tracked_files_and_skips_scratch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "docs" / "scratch").mkdir(parents=True)
    (tmp_path / "docs" / "scratch" / "dump.md").write_text("zebra\n")
    (tmp_path / "src.py").write_text("x = 1\n")
    (tmp_path / "untracked.md").write_text("zebra\n")
    subprocess.run(["git", "add", "src.py", "docs/scratch/dump.md"], check=True)
    words = _denylist(tmp_path)
    assert cpv.main(["--all", "--denylist", str(words)]) == 0
    (tmp_path / "src.py").write_text("x = 'zebra'\n")
    assert cpv.main(["--all", "--denylist", str(words)]) == 1
