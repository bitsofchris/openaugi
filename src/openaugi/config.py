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
    "retrieval": {
        "overfetch_ratio": 3,  # fetch k*ratio candidates before reranking
        "group_threshold": 0.15,  # cosine distance below which chunks are considered duplicates
        "mmr_lambda": 0.5,  # 1.0 = pure relevance, 0.0 = pure diversity
        "representative": "centroid",  # "centroid" | "score"
        # Provenance values dropped from get_context and semantic search when
        # the caller does not name a provenance. Imported material (Readwise,
        # Snipd, gdrive) otherwise drowns a single-author vault: one Hollis
        # podcast came back as twenty near-identical hits. Keyword and browse
        # modes are unaffected — keyword is precise by construction and browse
        # already groups reference documents. Pass provenance=[...] to opt in.
        "exclude_provenance": ["reference"],
    },
    "salience": {
        # Min retrieval score per purpose. Scores are cosine similarity as of
        # 2026-08-29 (was 1−L2, which clamped cosine<0.5 to ~0). Recalibrated the
        # same day against a 7-day replay: real scores cluster 0.53-0.68 whatever
        # the block is, so this is a NOISE FLOOR — the caller's intent classifier
        # is what actually decides whether a proactive surface should speak.
        "resurface": 0.50,  # in-app resurfacing (mobile bridge, proactive echo)
        "push": 0.62,  # reserved: push notifications need a stricter floor (no consumer yet)
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


def resolve_vault_path(
    explicit: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Resolve the vault path from an explicit value or config, with "~" expanded.

    Precedence: explicit (a --path flag) > [vault] default_path in config.

    Expansion happens here, once, because the raw config string reaches a dozen
    consumers (parser, watcher, dispatcher, context pack, renderers) and only
    some of them would survive a literal "~". Note that Path.resolve() does NOT
    expand "~" — it resolves the tilde as a relative directory name against the
    cwd — so a caller that only resolves still fails on a "~/vault" config.

    Returns None when neither source supplies a path; callers own that error
    message since it differs per command.
    """
    raw = explicit or (config or {}).get("vault", {}).get("default_path")
    if not raw:
        return None
    return str(Path(raw).expanduser())


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
