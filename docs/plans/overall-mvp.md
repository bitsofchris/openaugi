---
name: overall-mvp
description: Master milestone plan for open-sourcing OpenAugi — porting augi-engine-v1 to blocks+links data model
---

# OpenAugi — Milestones to Ship
*Created: 2026-03-19 | Updated: 2026-03-25*

---

## The Goal

A self-hostable, open source personal intelligence engine. One `pip install`. One SQLite file. One MCP server. Works with Claude out of the box.

Take augi-engine-v1 (working Obsidian engine, private) → port to the blocks+links data model (architecture-v5) → clean open source repo.

**Not a SaaS app. Not a framework. A working tool that solves a real problem.**

---

## Architecture

See [ARCHITECTURE.md](../../ARCHITECTURE.md) for the canonical data model and system design.

**Core idea:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags, summaries, extractions, lenses). Everything connectable is a link. Structure lives in the links, not in the schema.

**Storage:** SQLite in WAL mode + FAISS for vector search + FTS5 for keyword search.

**Processing layers:**
- Layer 0 (FREE) — split, tag/link extract, FTS, dedup. No models.
- Layer 1 (~$0) — embed (local sentence-transformers by default), hub scoring.
- Layer 2 (selective) — hub summaries, entity extraction. LLM required. M2.

---

## Milestone Overview

| # | Milestone | Goal | Status |
|---|-----------|------|--------|
| **M0** | **Port & Foundation** | Port v1 to blocks+links, SQLite+FAISS, vault adapter, Layer 0+1, model abstraction, 6 MCP read tools, CLI. | **✅ Done** |
| **M1** | **MCP Write + Local Setup** | Write tools (`write_document`), `reload_index`, vault resource, Claude Desktop/Code registration docs. | **✅ Done** |
| M2 | Enrichment | Layer 2: hub summaries (LLM → summary blocks), `get_summary` MCP tool, `openaugi enrich` CLI. Entity extraction as stretch. | **Next** |
| M3 | Temporal Intelligence | Hub velocity, recurrence, dead streams. New MCP tools. Lens protocol. | |
| M4 | Public Launch | README as problem statement, v0.1 tag, PyPI, first post. | |
| M5 | Multi-Source | ChatGPT adapter, adapter base class, cross-source hubs. | Last |

---

## M0 — Port & Foundation ✅

*Detailed plan: [done/m0.md](done/m0.md)*

**End state:** `pip install openaugi`, point at your Obsidian vault, ingest → embed → query via MCP.

### What Shipped

- **Data model**: Block + Link as Pydantic models
- **Store**: SQLite (WAL, FTS5, CASCADE) + FAISS index
- **Vault adapter**: Obsidian `.md` → blocks + links (H3 split, tags, wikilinks, dates, incremental)
- **Pipeline Layer 0+1**: Ingest → split → extract → dedup → embed → hub scoring
- **Model abstraction**: EmbeddingModel + LLMModel protocols, factory, TOML config
- **MCP server**: 6 read tools (search, get_block, get_related, traverse, get_context, recent)
- **CLI**: ingest, serve, search, hubs, status, init

---

## M1 — MCP Write + Local Setup ✅

*Detailed plan: [done/m1.md](done/m1.md)*

**End state:** Claude can read your vault and write documents back to it. Registered in Claude Desktop and Claude Code.

### What Shipped

- **`write_document` MCP tool**: Create `.md` notes in `OpenAugi/{subfolder}/` in the vault. Agent picks subfolder (`Docs`, `Notes`, `Research`, etc.), constrained to `OpenAugi/` root.
- **`reload_index` MCP tool**: Force-refresh FAISS after ingest.
- **`vault://note/{title}` resource**: `@openaugi:vault://note/Title` autocomplete in Claude Code.
- **Vault path config**: `OPENAUGI_VAULT_PATH` env var + `config.toml [vault] default_path`. Set via `openaugi init`.
- **Setup docs**: [docs/MCP_SERVER.md](../MCP_SERVER.md) — Claude Desktop config, Claude Code `claude mcp add`, env vars reference.
- **Refactor**: Extracted `_get_embedding_model()` helper, fixed silent exception swallowing in `get_context`.

---

## M2 — Enrichment

Layer 2 processing: LLM-powered summarization of top hubs. The goal is restoring v1's most-used feature (`get_summary`) and making "what have I been thinking about?" a genuinely good answer.

**End state:** `openaugi enrich` runs against your ingested vault, generates summary blocks for top hubs, and Claude can retrieve them via `get_summary`.

### Core Scope

- **`openaugi enrich` CLI**: Run Layer 2 on demand. Takes `--top-n` (default 50), `--model` override. Shows progress.
- **Hub summaries**: For each top-N hub by link count, call LLM with the hub's entries → `Block(kind=summary)` + `Link(kind=summarizes")`. Prompt structure from v1 (themes, tensions, decisions). Incremental — skip hubs with a fresh summary unless `--force`.
- **`get_summary` MCP tool**: Look up summary block by document title or ID. Falls back gracefully if no summary exists.
- **LLM config**: `openaugi init` prompts for LLM provider (OpenAI default, Ollama option). Stored in `config.toml [models.llm]`.

### Stretch (include if clean, else M3)

- **Entity extraction**: LLM call per hub → extract named entities (people, projects, concepts) as `Block(kind=entity)` + `mentions` links. Useful for graph density but adds complexity.
- **Block-level summaries**: Short (1-2 sentence) self-description per entry, generated on-demand via `summarize_block` MCP tool.

### What Does NOT Ship in M2

- Clustering (HDBSCAN on embeddings) — deferred to M3 or later
- Streaming / async enrichment — batch is fine for personal use
- Automatic re-enrichment on ingest — always manual (`openaugi enrich`)

---

## M3 — Temporal Intelligence

The moat. Time-aware analysis over the graph. Mostly free SQL aggregations over data that already exists.

**End state:** Claude can answer "what have I been thinking about?" and "what threads did I drop?" with grounded, time-aware answers.

**Scope:**
- Hub velocity (rising vs declining topics over sliding windows)
- Recurrence detection (topics that keep coming back)
- Dead stream detection (threads you dropped)
- Centroid drift (your thinking is shifting on this topic)
- New MCP tools for temporal queries (`hub_velocity`, `dead_streams`, `recurrence`)
- Lens protocol + first built-in lens (WeeklyReview)

---

## M4 — Public Launch

Ship it. Embarrassingly early is the right time.

**Outcome:** Live on PyPI, GitHub, first post published.

**Scope:**
- README as problem statement (not feature list)
- v0.1.0 tag
- GitHub Actions release → PyPI
- First public post: architecture + why nothing else solves this

---

## M5 — Multi-Source

Add ChatGPT JSON export as a second source. Deferred until after launch — one working source is enough to prove the thesis.

**Outcome:** Vault + ChatGPT history in one graph. Adapter base class extracted. Cross-source hubs work.

**Scope:**
- ChatGPT adapter: parse export JSON → document blocks → entry blocks (turn-level split)
- Extract adapter base class from vault + chatgpt implementations
- Validate hub scoring across sources
- Maybe: Claude export adapter, Readwise

---

## Repo Strategy

| Repo | Role |
|------|------|
| `openaugi` (public) | The trunk. MIT. This is what ships. |
| `openaugi-private` | Dev workspace. augi-engine-v1 lives here. Experiments stay here. |
| `openaugi-ai-chat-to-second-brain` | ChatGPT adapter source material (M5). |

---

## What Does NOT Ship in These Milestones

- Cloud deployment / SaaS
- Web frontend
- Postgres / Redis / DuckDB (SQLite only)
- OAuth flows for Google, Notion, Slack
- Obsidian plugin
- Mobile-specific features (covered by MCP + remote tunnel)
- Task system / thread management (private workflow)
- Custom lenses (framework ships in M3, custom lenses are personal)
