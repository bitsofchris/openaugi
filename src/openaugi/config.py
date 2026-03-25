"""TOML config loader + env var loading.

Config: ~/.openaugi/config.toml or project-local openaugi.toml.
Keys: ~/.openaugi/.env (loaded into os.environ on first config load).
Falls back to sensible defaults (local sentence-transformers, no LLM).

API keys follow the industry standard: env vars are primary,
.env files are the convenience/persistence layer.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_env_loaded = False

DEFAULT_CONFIG = {
    "models": {
        "embedding": {
            "provider": "sentence-transformers",
            "model": "all-MiniLM-L6-v2",
        },
    },
    "vault": {
        "exclude_patterns": [
            ".obsidian/**",
            ".git/**",
            ".smart-env/**",
            ".trash/**",
            "templates/**",
        ],
        "max_workers": 4,
    },
    "hub": {
        "weights": {"in_links": 0.5, "out_links": 0.3, "entry_count": 0.2},
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config from TOML file, merging with defaults.

    Also loads API keys from ~/.openaugi/.env into os.environ (once).

    Search order:
    1. Explicit path (if provided)
    2. ./openaugi.toml (project-local)
    3. ~/.openaugi/config.toml (user-level)
    4. Defaults only
    """
    _load_env()
    if config_path:
        path = Path(config_path)
        if path.exists():
            return _merge(DEFAULT_CONFIG, _load_toml(path))
        logger.warning(f"Config file not found: {path}")

    # Project-local
    local = Path("openaugi.toml")
    if local.exists():
        return _merge(DEFAULT_CONFIG, _load_toml(local))

    # User-level
    user = Path.home() / ".openaugi" / "config.toml"
    if user.exists():
        return _merge(DEFAULT_CONFIG, _load_toml(user))

    return dict(DEFAULT_CONFIG)


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_env() -> None:
    """Load ~/.openaugi/.env into os.environ (once, won't overwrite existing vars).

    Simple parser — no dependency on python-dotenv. Handles KEY=VALUE lines,
    ignores comments and blank lines.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True

    env_path = Path.home() / ".openaugi" / ".env"
    if not env_path.exists():
        return

    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")  # strip optional quotes
            # Don't overwrite existing env vars (env takes priority over file)
            if key not in os.environ:
                os.environ[key] = value
        logger.debug(f"Loaded env from {env_path}")
    except Exception as e:
        logger.warning(f"Failed to load {env_path}: {e}")
