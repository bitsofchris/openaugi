"""`openaugi init` — the guided setup (cli/setup.py)."""

from __future__ import annotations

import stat
import tomllib
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import openaugi.cli.setup as setup
from openaugi.cli.main import app
from openaugi.config import _merge, auto_dispatch_enabled
from openaugi.models import embeddings_enabled

runner = CliRunner()
NONE, LOCAL, OPENAI = setup.SEARCH_MODES


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return h


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    (v / ".obsidian").mkdir(parents=True)
    (v / "a.md").write_text("# A\n")
    return v


def _config(home: Path) -> dict:
    return tomllib.loads((home / ".openaugi" / "config.toml").read_text())


# ── pure helpers ───────────────────────────────────────────────────────────


def test_config_toml_round_trips_and_means_what_it_says():
    text = setup.build_config_toml("/v", NONE, auto_dispatch=False, today="2022-01-01")
    cfg = tomllib.loads(text)
    assert cfg["vault"]["default_path"] == "/v"
    assert cfg["models"]["embedding"] == {"provider": "none"}
    assert cfg["tasks"]["auto_dispatch"] is False

    from openaugi.config import DEFAULT_CONFIG

    merged = _merge(DEFAULT_CONFIG, cfg)
    assert not embeddings_enabled(merged["models"]["embedding"])
    assert not auto_dispatch_enabled(merged)


def test_config_toml_escapes_awkward_paths():
    path = 'C:\\Users\\someone\\My "Vault"'
    cfg = tomllib.loads(setup.build_config_toml(path, OPENAI, auto_dispatch=True))
    assert cfg["vault"]["default_path"] == path
    assert cfg["models"]["embedding"] == {"provider": "openai", "model": "text-embedding-3-small"}
    assert cfg["tasks"]["auto_dispatch"] is True


def test_merge_env_updates_in_place_and_keeps_the_rest():
    existing = "# keys\nREADWISE=abc\nOPENAI_API_KEY=old\n"
    out = setup.merge_env(existing, {"OPENAI_API_KEY": "new", "OTHER": "1"})
    assert out == "# keys\nREADWISE=abc\nOPENAI_API_KEY=new\nOTHER=1\n"


def test_count_notes_skips_dot_folders(vault: Path):
    (vault / ".obsidian" / "x.md").write_text("plugin")
    (vault / "sub").mkdir()
    (vault / "sub" / "b.md").write_text("# B")
    (vault / "sub" / "img.png").write_bytes(b"")
    assert setup.count_notes(vault) == 2


def test_tilde(home: Path):
    assert setup.tilde(home / "Documents") == "~/Documents"
    assert setup.tilde("/elsewhere") == "/elsewhere"


def test_backup_path_is_dated(tmp_path: Path):
    b = setup.backup_path(tmp_path / "config.toml", datetime(2022, 3, 4, 5, 6, 7))
    assert b.name == "config.toml.bak-2022-03-04-050607"


def test_next_steps_name_the_missing_extra(monkeypatch):
    monkeypatch.setattr(setup, "provider_installed", lambda p: False)
    cmds = [cmd for cmd, _ in setup.next_steps(LOCAL, auto=False)]
    assert cmds[0] == 'pip install "openaugi[local]"'
    assert "openaugi ingest" in cmds
    assert "openaugi up" in [cmd for cmd, _ in setup.next_steps(NONE, auto=True)]


# ── the command ────────────────────────────────────────────────────────────


def test_unattended_setup(home: Path, vault: Path):
    result = runner.invoke(
        app, ["init", "--vault", str(vault), "--search", "none", "--mode", "light", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "You're set up" in result.output
    # Light: the agent gets the CLI, not an MCP server.
    assert "openaugi kb get-context" in result.output
    assert "claude mcp add" not in result.output
    cfg = _config(home)
    assert cfg["vault"]["default_path"] == str(vault.resolve())
    assert cfg["models"]["embedding"]["provider"] == "none"
    assert cfg["tasks"]["auto_dispatch"] is False
    assert (vault / "OpenAugi" / "AGENT" / "augi-agent.md").exists()


def test_interactive_defaults_are_the_safe_ones(home: Path, vault: Path):
    # Vault path, then Enter for search and Enter for tasks.
    result = runner.invoke(app, ["init"], input=f"{vault}\n\n\n")
    assert result.exit_code == 0, result.output
    cfg = _config(home)
    assert cfg["models"]["embedding"]["provider"] == "none"
    assert cfg["tasks"]["auto_dispatch"] is False


def test_a_choice_off_the_menu_asks_again(home: Path, vault: Path):
    # 9 is not on the search menu; Enter then takes the default.
    result = runner.invoke(app, ["init"], input=f"{vault}\n9\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Pick a number from 1 to" in result.output
    assert _config(home)["models"]["embedding"]["provider"] == "none"


def test_missing_vault_asks_again(home: Path, vault: Path):
    result = runner.invoke(app, ["init"], input=f"{vault}/nope\n{vault}\n\n\n")
    assert result.exit_code == 0, result.output
    assert "doesn't exist" in result.output
    assert _config(home)["vault"]["default_path"] == str(vault.resolve())


def test_missing_vault_flag_exits(home: Path, tmp_path: Path):
    result = runner.invoke(app, ["init", "--vault", str(tmp_path / "nope"), "--yes"])
    assert result.exit_code == 1


def test_yes_needs_a_vault_the_first_time(home: Path):
    result = runner.invoke(app, ["init", "--yes"])
    assert result.exit_code == 1
    assert "--vault" in result.output


def test_existing_config_is_kept_unless_you_say_replace(home: Path, vault: Path):
    cfg_dir = home / ".openaugi"
    cfg_dir.mkdir()
    original = '[vault]\ndefault_path = "/old"\nexclude_patterns = ["x/**"]\n'
    (cfg_dir / "config.toml").write_text(original)

    kept = runner.invoke(app, ["init", "--vault", str(vault)], input="\n\nn\n")
    assert kept.exit_code == 0, kept.output
    assert (cfg_dir / "config.toml").read_text() == original
    assert "kept your existing" in kept.output
    assert "auto_dispatch = false" in kept.output  # the snippet to apply by hand
    assert not list(cfg_dir.glob("config.toml.bak-*"))

    replaced = runner.invoke(app, ["init", "--vault", str(vault)], input="\n\ny\n")
    assert replaced.exit_code == 0, replaced.output
    backups = list(cfg_dir.glob("config.toml.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    assert _config(home)["vault"]["default_path"] == str(vault.resolve())


def test_full_mode_prints_the_mcp_hookup(home: Path, vault: Path):
    result = runner.invoke(app, ["init", "--vault", str(vault), "--mode", "full", "--yes"])
    assert result.exit_code == 0, result.output
    assert "claude mcp add" in result.output
    assert "openaugi up" in result.output
    assert _config(home)["tasks"]["auto_dispatch"] is True


@pytest.mark.parametrize("old,auto", [("manual", False), ("auto", True)])
def test_old_tasks_flag_still_works(home: Path, vault: Path, old: str, auto: bool):
    result = runner.invoke(app, ["init", "--vault", str(vault), "--tasks", old, "--yes"])
    assert result.exit_code == 0, result.output
    assert _config(home)["tasks"]["auto_dispatch"] is auto


def test_rerun_offers_current_choices_as_defaults(home: Path, vault: Path):
    runner.invoke(app, ["init", "--vault", str(vault), "--mode", "full", "--yes"])
    assert _config(home)["tasks"]["auto_dispatch"] is True
    # Enter through everything (vault default, search, tasks), then replace.
    result = runner.invoke(app, ["init"], input="\n\n\ny\n")
    assert result.exit_code == 0, result.output
    assert _config(home)["tasks"]["auto_dispatch"] is True


def test_openai_key_from_env_is_saved_owner_only(home: Path, vault: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(setup, "provider_installed", lambda p: True)
    result = runner.invoke(app, ["init", "--vault", str(vault), "--search", "openai", "--yes"])
    assert result.exit_code == 0, result.output
    env = home / ".openaugi" / ".env"
    assert "OPENAI_API_KEY=sk-test-not-a-real-key" in env.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert "sk-test-not-a-real-key" not in result.output


def test_choosing_an_uninstalled_provider_says_how_to_install(
    home: Path, vault: Path, monkeypatch
):
    monkeypatch.setattr(setup, "provider_installed", lambda p: False)
    result = runner.invoke(app, ["init", "--vault", str(vault), "--search", "local", "--yes"])
    assert result.exit_code == 0, result.output
    assert 'pip install "openaugi[local]"' in result.output
