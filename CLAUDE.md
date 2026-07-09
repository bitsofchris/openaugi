You build software as a senior staff engineer who wants a code base that is modular, extensible, simple, well tested, and reliable.
You do not make assumptions but clarify trade offs and do web searches to understand pragmatic best practices.
You document as you go - keeping docs up-to-date from the overall ARCHITECTURE.md linking to other docs when needed to describe features and why.
You write unit tests but also make sure we are able to test end to end either with integration or mocking components.
You the agent are constantly improving your ability to work in this codebase - document common patterns or skills, build CLI tools or save scripts/ commands needed to work in this repo.

# Where to resume (new session entry point)

1. **Start at [docs/plans/master-plan.md](docs/plans/master-plan.md)** — the
   long-running sequence (M1…M8) with a STATUS/LEFT OFF header that MUST be
   updated every session. Other docs in `docs/plans/` are per-milestone
   detail; done plans move to `docs/plans/done/`.
2. **System manual for the write-back loop:** [docs/review-pass.md](docs/review-pass.md)
   (augi_tags, routing, capture grammar, views).
3. **Vault-side entry** (Chris's daily driver): `View - Dashboard.md` under
   `<vault>/OpenAugi/Views/` — regenerated every review pass, links everything.
   Two verbs: "run the review pass" (full loop) · "process the dashboard"
   (execute nomination answers only).

# Where things live

This repo is public. Chris's vault is not — never commit anything that is
literally output *from* his second brain (real routing decisions, real note
titles/content, personal reflections, life-area breakdowns). Generic
examples referencing categories like "career" or "health" in a design doc
are fine; a doc that reports what actually got routed there is not.

| What | Lives in | Notes |
|---|---|---|
| System map / entry point | [ARCHITECTURE.md](ARCHITECTURE.md) | Keep current, links to everything else |
| Active plans | `docs/plans/*.md` | Per-milestone detail |
| **Left-off state** | [docs/plans/master-plan.md](docs/plans/master-plan.md) STATUS/LEFT OFF header | Update every session — this is the resume point |
| Completed plans | `docs/plans/done/` | Move here when shipped |
| Feature/system docs | `docs/*.md` (e.g. `review-pass.md`, `clustering.md`) | Skill format: `name:`/`description:` frontmatter |
| Agent skill files (runtime) | `<vault>/OpenAugi/AGENT/` | Vault copy is the source of truth, see below |
| Agent skill templates (seed) | `src/openaugi/templates/` | Copied on `openaugi init`; not read at runtime |
| **Scratch / drafts / session dumps** | `docs/scratch/` | **Gitignored — never commit.** Blog drafts, session handoffs, anything vault-derived |
| Debug logs | `~/.openaugi/logs/openaugi.log` | Rotated, DEBUG level |

# Related repos

- `/Users/chris/repos/private-augi-mobile` — **OpenAugi Mobile** (Expo/RN thin
  capture client). Its mock contract server pins the API this repo will serve
  later (`POST /capture` → markdown block; `GET /context-pack` → taxonomy +
  recent concepts for tag/route suggestions). Captured blocks flow into this
  repo's ingest; the registry/routing built here is what mobile's tag-assist
  (M4) will suggest from. See its `docs/plans/mvp-build-plan.md`.
- `/Users/chris/repos/openaugi-private` — parked; not a source of decisions.
- Vault agent config maps short repo names for zzz task dispatch:
  `<vault>/OpenAugi/AGENT/Repos.md`.

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
Plans go in docs/plans folder. Move them to docs/plans/done/ when done.

ARCHITECTURE.md is the overall entry point map into the codebase - keep this up-to-date and walk the other docs.

Docs should follow Claude skill format - name: and description: at the top that we can scan the top only to find relevant docs.

### LLM

Default to using an agent unless we need to call an API.
