"""Config loading and vault path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from openaugi.config import resolve_vault_path


class TestResolveVaultPath:
    def test_expands_tilde_from_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = {"vault": {"default_path": "~/my_vault"}}
        assert resolve_vault_path(config=config) == str(tmp_path / "my_vault")

    def test_expands_tilde_from_explicit_arg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_vault_path("~/my_vault") == str(tmp_path / "my_vault")

    def test_explicit_arg_wins_over_config(self, tmp_path: Path):
        config = {"vault": {"default_path": "/from/config"}}
        assert resolve_vault_path("/from/flag", config) == "/from/flag"

    def test_absolute_path_passes_through(self):
        assert resolve_vault_path("/absolute/vault") == "/absolute/vault"

    def test_relative_path_is_left_relative(self):
        """Relative paths stay relative — resolving them would change cwd semantics."""
        assert resolve_vault_path("some/vault") == "some/vault"

    def test_accepts_path_object(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_vault_path(Path("~/my_vault")) == str(tmp_path / "my_vault")

    def test_none_when_nothing_configured(self):
        assert resolve_vault_path(None, {}) is None
        assert resolve_vault_path() is None

    def test_none_when_config_has_empty_path(self):
        assert resolve_vault_path(None, {"vault": {"default_path": ""}}) is None

    def test_missing_vault_section(self):
        assert resolve_vault_path(None, {"models": {}}) is None


class TestMCPVaultPath:
    """The MCP server resolves its own vault path — it must expand ~ too."""

    def test_env_var_tilde_expands(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from openaugi.mcp import server

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OPENAUGI_VAULT_PATH", "~/my_vault")
        assert server._get_vault_path() == str(tmp_path / "my_vault")

    def test_falls_back_to_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from openaugi.mcp import server

        monkeypatch.delenv("OPENAUGI_VAULT_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            server, "load_config", lambda: {"vault": {"default_path": "~/from_config"}}
        )
        assert server._get_vault_path() == str(tmp_path / "from_config")
