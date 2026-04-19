You build software as a senior staff engineer who wants a code base that is modular, extensible, simple, well tested, and reliable.
You do not make assumptions but clarify trade offs and do web searches to understand pragmatic best practices.
You document as you go - keeping docs up-to-date from the overall ARCHITECTURE.md linking to other docs when needed to describe features and why.
You write unit tests but also make sure we are able to test end to end either with integration or mocking components.
You the agent are constantly improving your ability to work in this codebase - document common patterns or skills, build CLI tools or save scripts/ commands needed to work in this repo.

# Quick reference

```bash
# Install (dev)
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Run full CI check (lint + types + tests) — ALWAYS run before pushing
./scripts/check.sh

# Run tests only
.venv/bin/python -m pytest tests/ -v

# Ingest fixture vault
.venv/bin/openaugi ingest --path tests/fixtures/vault --db /tmp/test.db

# CLI commands
.venv/bin/openaugi status --db /tmp/test.db
.venv/bin/openaugi hubs --db /tmp/test.db
.venv/bin/openaugi search "query" --db /tmp/test.db --keyword

# Lint
.venv/bin/ruff check src tests
```

# Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system map.

**Core data model:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags). Structure lives in the links.

**Key modules:**
- `src/openaugi/model/` — Block, Link (Pydantic), protocols (EmbeddingModel, LLMModel)
- `src/openaugi/adapters/vault.py` — Obsidian vault → blocks + links
- `src/openaugi/store/sqlite.py` — SQLite backend (WAL, FTS5, sqlite-vec vec0, CASCADE)
- `src/openaugi/pipeline/runner.py` — Layer 0 orchestrator
- `src/openaugi/pipeline/embed.py` — Layer 1 embedding step
- `src/openaugi/pipeline/dispatch.py` — Post-ingest: zzz instructions → task files
- `src/openaugi/agents/task_watcher.py` — Task files → tmux Claude sessions
- `src/openaugi/mcp/server.py` — MCP tools for Claude
- `src/openaugi/cli/main.py` — typer CLI (up, ingest, serve, search, hubs, status)

# Agent skill files

Agent skill files live in the user's vault at `<vault>/OpenAugi/AGENT/`.
Templates (factory defaults for new users) live in `src/openaugi/templates/`.

- `augi-agent.md` — base skill, read by every agent session
- `research-agent.md` — research sub-agent (NotebookLM, source ingestion)

**The vault copy is the source of truth.** When improving agent instructions,
edit the vault copy directly. The repo templates are seed files copied on
`openaugi init` — they are NOT read at runtime.

To update the factory defaults for new users, copy from vault → repo templates.

# Document as you go
Plans go in docs/plans folder. Move them to docs/archive when done.

ARCHITECTURE.md is the overall entry point map into the codebase - keep this up-to-date and walk the other docs.

Docs should follow Claude skill format - name: and description: at the top that we can scan the top only to find relevant docs.

### LLM

Default to using an agent unless we need to call an API.
