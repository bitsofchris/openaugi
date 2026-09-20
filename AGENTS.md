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
2. **System manual for the write-back loop:** [docs/reference/review-pass.md](docs/reference/review-pass.md)
   (augi_tags, routing, capture grammar, views).
3. **Vault-side entry** (the user's daily driver): `View - Dashboard.md` under
   `<vault>/OpenAugi/Views/` — regenerated every review pass, links everything.
   Two verbs: "run the review pass" (full loop) · "process the dashboard"
   (execute nomination answers only).

# Where things live

This repo is public. The user's vault is not. Before committing any doc, ask:
does this contain actual content from the vault, or just describe the
system that operates on it?

**Fine to commit:** category/taxonomy names, tag names, schema fields,
config shapes — the system-level vocabulary (e.g. "an `area/health`
container exists," "clustering produces life-area buckets").

**Never commit:** real note titles or content, real routing/nomination
decisions, actual cluster contents or counts, personal reflections, post
drafts, session handoffs that quote real vault data — i.e. anything that
is *output from* the second brain rather than a description *of* it. If a
doc is reporting what specifically got routed/clustered/decided on the user's
real data, it belongs in `docs/scratch/` (gitignored), not tracked.

**Privacy checklist — every file, every commit message, every time.** Each
rule below exists because it was broken once and the history had to be
rewritten to fix it. Mechanism lives in the repo; everything that is *yours*
lives in the vault.

1. **No names.** Not the maintainer's, not family, not friends, not the
   therapist. Write "the user" / "they"; plans record rulings as *"the user,
   2026-09-04: …"*. The public author identity appears only in LICENSE,
   NOTICE and `pyproject.toml`.
2. **No personal vocabulary or schema in code.** A script takes its kinds,
   keys and targets as flags and discovers values from the data. Field names,
   value lists and schedules belong in the vault's `OpenAugi/AGENT/` files
   (lenses, plans) that invoke the script.
3. **Examples are placeholders.** `mood=`, `place=`, `alice`,
   `/Users/someone`, a 2022 date. Never paste a real line from a daily note
   into a test, a docstring or a doc, not even "just to show the shape".
4. **No paths that are yours.** `~/…` and `<vault>/…`, never
   `/Users/<name>/…`. The vault's folder name comes from the openaugi config
   or `$OPENAUGI_VAULT`, never a literal in a script.
5. **No notebook outputs, no data dumps.** Clear outputs before committing.
   Exports (chat history, Readwise, sqlite files) never enter the tree, not
   even under `experiments/`.
6. **No vault output.** Cluster contents, routing decisions, real note
   titles, board items, session handoffs → `docs/scratch/` (gitignored).
7. **The hook has the last word.** `scripts/check_private_vocab.py` runs at
   commit, commit-msg and push against
   `<vault>/OpenAugi/AGENT/private-vocabulary.txt` — a vault file, so the
   guard itself can never leak — and refuses notebooks with outputs. When
   you coin a private word, add it to that list *before* you write the code.
   Never `--no-verify`. Install the three stages once per clone:
   `.venv/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push`.
   Details: [docs/reference/privacy-guard.md](docs/reference/privacy-guard.md).

**When something leaks anyway:** deleting it in a new commit is not enough;
history keeps it. Back up every ref to a bundle, rewrite with
`git filter-repo` (replace-text for words, invert-paths for files), verify
over `--branches --tags` (a fetch would bring the old remote history back into
view), force-push, prune, re-clone every other copy, add the word to the
list. The incident itself is vault or `docs/scratch/` material, not a commit
message.

| What | Lives in | Notes |
|---|---|---|
| System map / entry point | [ARCHITECTURE.md](ARCHITECTURE.md) | Keep current, links to everything else |
| Active plans | `docs/plans/*.md` | Per-milestone detail |
| **Left-off state** | [docs/plans/master-plan.md](docs/plans/master-plan.md) STATUS/LEFT OFF header | Update every session — this is the resume point |
| Completed plans | `docs/plans/done/` | Move here when shipped |
| Feature/system reference docs | `docs/reference/*.md` (e.g. `review-pass.md`, `clustering.md`, `lenses.md`) | Durable "how it works" manuals. **New reference docs go here, never `docs/` root.** Skill format: `name:`/`description:` frontmatter |
| Agent skill files (runtime) | `<vault>/OpenAugi/AGENT/` | Vault copy is the source of truth, see below |
| Agent skill templates (seed) | `src/openaugi/templates/` | Written by `scripts/sync_templates.py` from the vault's `kind: engine` files; copied on `openaugi init`; not read at runtime |
| **Scratch / drafts / session dumps** | `docs/scratch/` | **Gitignored — never commit.** Blog drafts, session handoffs, anything vault-derived |
| Debug logs | `~/.openaugi/logs/openaugi.log` | Rotated, DEBUG level |

**Docs layout (convention — keep it this way).** Everything under `docs/`
is one of three tiers: **`reference/`** = durable manuals for how a
subsystem works (the tier ARCHITECTURE.md links); **`plans/`** = design
records and active plans, moved to `plans/done/` when shipped;
**`scratch/`** = gitignored throwaway. When you add a doc, pick the tier
first — a "how X works" manual is `reference/`, a "here's what we'll
build" is `plans/`. Never drop a reference doc at `docs/` root.

# After shipping a feature: update the Command Deck (required)

The user is still learning their own system, so `<vault>/OpenAugi/Docs/OpenAugi
Command Deck.md` is his entry point: the one screen that answers "what can I
actually do with this thing right now." A feature he cannot find there does not
exist to him.

**Any change to OpenAugi or the Obsidian plugin that adds, alters, or retires
something the user can use or must know about ends with an edit to that file** —
in the same session as the change, not batched later. That includes new
commands and capture grammar, new lenses or agents, background behavior, new
config keys, changed defaults, and anything retired.

Write two things:

1. **Where it belongs in the body** — the command to type, or a short "how you
   use it" for background behavior. Say how to turn it off and how to tune it.
   Link the design doc and the code files.
2. **A row at the top of the Changelog table** — date, one line on what
   changed, and a link into the deck section or design note.

Written for a user, not a reviewer: what to type, what shows up, where. Keep it
one screen — trim stale entries rather than letting the deck grow unbounded.
The deck is vault-side, so it may name real notes; the same content must not
leak into this public repo (see "Where things live").

# Related repos

- `~/repos/private-augi-mobile` — **OpenAugi Mobile** (Expo/RN thin
  capture client). Its mock contract server pins the API this repo will serve
  later (`POST /capture` → markdown block; `GET /context-pack` → taxonomy +
  recent concepts for tag/route suggestions). Captured blocks flow into this
  repo's ingest; the registry/routing built here is what mobile's tag-assist
  (M4) will suggest from. See its `docs/plans/mvp-build-plan.md`.
- `~/repos/openaugi-private` — parked; not a source of decisions.
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

**The vault copy is the source of truth.** When improving agent instructions,
edit the vault copy directly. The repo templates are seed files copied on
`openaugi init` — they are NOT read at runtime.

## Engine vs personal — the line every agent file sits on

Every file under the vault's `AGENT/` folder declares one of two kinds in its
frontmatter, and `scripts/sync_templates.py --check` refuses to run until it
does:

- **`kind: engine`** — part of the operating system anyone who installs
  OpenAugi runs (`augi-agent.md`, `review-pass.md`, `kanban.md`, `pmoc.md`,
  `routing.md`, the generic lenses). It has a template twin at the same
  relative path under `src/openaugi/templates/`, and it never names, genders,
  or describes its user — `tests/test_impersonal_engine.py` enforces that on
  every shipped file, `tests/test_agent_files.py` that every template declares
  `kind: engine`.
- **`kind: personal`** — the user's own configuration: their taxonomy, `Slowly
  Changing Context`, `Repos.md`, and the lenses only their life needs. Never
  shipped.

An engine file may still carry the user's rulings — the dated quote that says
why a rule exists, the table naming their areas. Those stay in the vault copy
inside a **personal region**, which the sync strips from the template:

```
%% personal %%
Ruling, 2026-08-20: *"…"*
%% /personal %%
```

The markers are Obsidian comments (hidden in reading view, plain text to an
agent). The engine sentence that the ruling justifies stays outside the region,
reworded for whoever installs this ("the user", "they").

**When a personal process changes, the repo does not.** The user's workflow
(the Sunday pass, the boards, what a lens asks) lives in vault lens files and
changes there first. Repo code changes only when the engine is actually broken
(a bug anyone would hit) or when the process needs a primitive it does not
have — and then the agent says so before touching `src/`, because a primitive
built for one iteration of a personal process is usually dead a day later
(2026-09-20: the reflection do-box dispatch, added and removed the same day).
Try the lens-only version for a week before asking the engine for anything.

**Workflow.** Edit the vault file → `python3 scripts/sync_templates.py --write`
(vault path from the openaugi config, or `--vault`) → commit the template.
`--check` reports drift and is the pre-push habit. Which files `init` copies is
decided by the templates themselves: every `.md` under `templates/` that
declares `kind: engine` (`src/openaugi/agent_files.py`). The one template
without a kind is `task-template.md`, which the code hydrates and never copies.

**Reference docs follow the same line.** `docs/reference/` describes the engine
in its current state; dated rulings and vault-specific incidents stay in the
vault's AGENT files (inside personal regions) or in `docs/scratch/`.

# Document as you go
Plans go in docs/plans folder. Move them to docs/plans/done/ when done.

ARCHITECTURE.md is the overall entry point map into the codebase - keep this up-to-date and walk the other docs.

Docs should follow skill format - name: and description: at the top that we can scan the top only to find relevant docs.

### LLM

Default to using an agent unless we need to call an API.
