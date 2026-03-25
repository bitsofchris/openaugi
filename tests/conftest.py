"""Shared test fixtures for OpenAugi.

Provides:
- tmp_db: temporary SQLite database (fresh per test)
- store: SQLiteStore instance backed by tmp_db
- vault_path: path to the fixture vault
"""

from pathlib import Path

import pytest

from openaugi.store.sqlite import SQLiteStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VAULT_DIR = FIXTURES_DIR / "vault"


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
