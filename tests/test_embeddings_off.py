"""Keyword-only installs: `[models.embedding] provider = "none"`.

A bare `pip install openaugi` has no embedding package, and `openaugi init`
sets it up with embeddings off. Everything must still answer: semantic
requests run as keyword searches and say so, and nothing that needs vectors
crashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openaugi.models import (
    EmbeddingsDisabled,
    embeddings_enabled,
    get_embedding_model,
    provider_installed,
)
from openaugi.pipeline.runner import run_layer0
from openaugi.query import QuerySpec, engine
from openaugi.store.sqlite import SQLiteStore

OFF = {"provider": "none"}


# ── factory ─────────────────────────────────────────────────────────────────


def test_provider_none_raises_embeddings_disabled():
    with pytest.raises(EmbeddingsDisabled, match="keyword"):
        get_embedding_model(OFF)


@pytest.mark.parametrize(
    "config,expected",
    [
        (OFF, False),
        ({"provider": "openai", "model": "x"}, True),
        ({}, True),  # no config → the sentence-transformers default
        (None, True),
    ],
)
def test_embeddings_enabled(config, expected):
    assert embeddings_enabled(config) is expected


def test_provider_installed_reads_the_import(monkeypatch):
    import openaugi.models as models

    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: None)
    assert provider_installed("openai") is False
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: object())
    assert provider_installed("openai") is True
    assert provider_installed("no-such-provider") is False


# ── resolve_semantic ────────────────────────────────────────────────────────


def _disabled():
    raise EmbeddingsDisabled("off")


def test_semantic_spec_becomes_keyword_when_embeddings_are_off():
    spec, model, note = engine.resolve_semantic(QuerySpec(query="career"), _disabled)
    assert spec.mode == "keyword"
    assert spec.keyword == "career"
    assert spec.query is None
    assert model is None
    assert note == engine.SEMANTIC_OFF_NOTE


def test_semantic_spec_keeps_its_model_when_embeddings_are_on():
    sentinel = object()
    spec, model, note = engine.resolve_semantic(QuerySpec(query="career"), lambda: sentinel)
    assert spec.mode == "semantic"
    assert model is sentinel
    assert note is None


def test_non_semantic_spec_never_builds_a_model():
    def explode():
        raise AssertionError("model must not be built for a keyword spec")

    spec, model, note = engine.resolve_semantic(QuerySpec(keyword="career"), explode)
    assert (spec.mode, model, note) == ("keyword", None, None)


def test_other_model_errors_still_surface():
    def broken():
        raise ImportError("sentence_transformers missing")

    with pytest.raises(ImportError):
        engine.resolve_semantic(QuerySpec(query="career"), broken)


# ── MCP server with embeddings off ──────────────────────────────────────────


@pytest.fixture
def keyword_only_server(tmp_path: Path, vault_path: Path, monkeypatch):
    from openaugi.config import DEFAULT_CONFIG, _merge

    db_path = tmp_path / "kw.db"
    store = SQLiteStore(db_path)
    run_layer0(vault_path, store)
    store.close()

    import openaugi.mcp.server as srv

    monkeypatch.setenv("OPENAUGI_DB", str(db_path))
    monkeypatch.setattr(
        srv, "load_config", lambda: _merge(DEFAULT_CONFIG, {"models": {"embedding": OFF}})
    )
    srv._store = None
    srv._embedding_model = None
    yield srv
    srv._store = None
    srv._embedding_model = None


def test_mcp_semantic_search_answers_as_keyword(keyword_only_server):
    result = json.loads(keyword_only_server.search(query="career"))
    assert result["mode"] == "keyword"
    assert result["count"] > 0
    assert result["note"] == engine.SEMANTIC_OFF_NOTE


def test_mcp_keyword_search_has_no_note(keyword_only_server):
    result = json.loads(keyword_only_server.search(keyword="career"))
    assert "note" not in result


def test_mcp_get_context_works_without_embeddings(keyword_only_server, caplog):
    result = json.loads(keyword_only_server.get_context(query="career"))
    assert "error" not in result
    # No model is expected in a keyword-only install, so nothing to warn about
    # (each `openaugi kb get-context` call would print a traceback otherwise).
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_re_embed_refuses_before_wiping_vectors(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import openaugi.config as config_mod
    from openaugi.cli.main import app

    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: {"models": {"embedding": OFF}})
    db_path = tmp_path / "never-created.db"
    result = CliRunner().invoke(app, ["re-embed", "--db", str(db_path)])
    assert result.exit_code == 1
    assert "Embeddings are off" in result.output
    assert not db_path.exists()
