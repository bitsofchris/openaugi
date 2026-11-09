"""Shared test fixtures for OpenAugi.

Provides:
- tmp_db: temporary SQLite database (fresh per test)
- store: SQLiteStore instance backed by tmp_db
- vault_path: path to the fixture vault
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from openaugi.store.sqlite import SQLiteStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VAULT_DIR = FIXTURES_DIR / "vault"

_HERMETIC_UNSET = (
    "OPENAUGI_DB",
    "OPENAUGI_VAULT_PATH",
    "OPENAUGI_VAULT",
    "OPENAI_API_KEY",
    "GITHUB_ACTIONS",
    "FORCE_COLOR",
    "CI",
)
_saved_env: dict[str, str | None] = {}
_fake_home: str | None = None


def pytest_configure(config):
    """Every test sees the same machine, on a laptop and in CI.

    - HOME is an empty folder, so no test reads the developer's
      ~/.openaugi/config.toml or .env. A test that silently depended on a
      configured embedding provider would otherwise pass locally and fail in CI.
    - OPENAUGI_* overrides and API keys from the shell are dropped.
    - Colour is off and CI detection is removed: GitHub Actions makes rich
      colour some output, which splits text like `--block-id` with escape codes.

    Done here, before test modules are imported, because the CLI builds its
    rich Consoles at import time and reads the colour settings then. Tests
    that need any of these set them with monkeypatch.
    """
    global _fake_home
    _fake_home = tempfile.mkdtemp(prefix="openaugi-test-home-")
    overrides: dict[str, str | None] = {name: None for name in _HERMETIC_UNSET}
    overrides.update({"HOME": _fake_home, "NO_COLOR": "1", "TERM": "dumb"})
    for name, value in overrides.items():
        _saved_env[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def pytest_unconfigure(config):
    for name, value in _saved_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if _fake_home:
        shutil.rmtree(_fake_home, ignore_errors=True)


@pytest.fixture
def vault_path() -> Path:
    """Path to the test fixture vault."""
    return VAULT_DIR


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Temporary SQLite database path (unique per test)."""
    return tmp_path / "test.db"


@pytest.fixture
def store(tmp_db: Path) -> SQLiteStore:
    """Fresh SQLiteStore instance for each test."""
    s = SQLiteStore(tmp_db)
    yield s
    s.close()
