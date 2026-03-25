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

See [architecture-v5.md](architecture-v5.md) for the canonical data model and system design.

**Core idea:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags, summaries, extractions, lenses). Everything connectable is a link. Structure lives in the links, not in the schema.

**Storage:** SQLite in WAL mode + FAISS for vector search + FTS5 for keyword search.

**Processing layers:**
- Layer 0 (FREE) — split, tag/link extract, FTS, dedup. No models.
- Layer 1 (~$1/50k) — embed (local sentence-transformers by default), hub scoring, clustering.
- Layer 2 (selective) — entity extraction, hub summaries. LLM required. Deferred to M2.

---

## What We're Porting

augi-engine-v1 is ~1500 lines across 7 files. It works. The port maps it to the new data model:

| v1 Component | Port To | What Changes | What Stays |
|---|---|---|---|
| `Entry` dataclass | `Block` (Pydantic, kind=entry) | Schema shape, field names | Splitting logic, timestamp resolution, tag/link extraction |
| `ObsidianEntryParser` | Vault adapter | Output type (Block instead of Entry), tags become tag blocks + edges | H3 splitting, regex patterns, date inference, frontmatter parsing |
| DuckDB (4 tables) | SQLite (2 tables + FTS5) | Storage engine, schema | Incremental hashing, query patterns |
| `notes` table + hub_score | Link-count aggregation queries | Hub scoring from dedicated table → edge aggregation | Logarithmic formula |
| OpenAI embeddings (hardcoded) | Model abstraction (EmbeddingModel protocol) | Provider pluggable, default local | FAISS index, cosine search |
| FAISS IndexFlatIP | Stays | — | Cosine similarity, normalized vectors |
| MCP server (9 read tools + 4 write tools) | 6 core tools (v5 spec) | Simplified tool surface | Core search patterns |
| `hub_summaries` table | `Block(kind=summary)` + links | Summaries become blocks | LLM prompts (deferred to M2) |

**Not ported (stays in private repo for now):** Task system (scan_tasks, write_thread, update_task, write_document). See [future-work.md](future-work.md).

---

## Milestone Overview

| # | Milestone | Goal | Status |
|---|-----------|------|--------|
| **M0** | **Port & Foundation** | Port v1 to blocks+links, SQLite+FAISS, vault adapter, Layer 0+1 pipeline, model abstraction, 6 MCP tools, clean repo. | **✅ Done** |
| M1 | Multi-Source | Second source (ChatGPT JSON export). Adapter abstraction emerges. Cross-source hub scoring. | Next |
| M2 | Enrichment | Layer 2: entity extraction, hub summaries as summary blocks, clustering. | |
| M3 | Temporal Intelligence | Hub velocity, recurrence, dead streams. New MCP tools. | |
| M4 | Public Launch | README as problem statement, v0.1 tag, PyPI, first post. | |

---

## M0 — Port & Foundation

*Detailed plan: [m0.md](m0.md)*

The only milestone that matters right now. Port augi-engine-v1 into the public repo with the new data model and a clean architecture.

**End state:** `pip install openaugi`, point at your Obsidian vault, ingest → embed → query via MCP. Works.

### What Ships

- **Data model**: Block + Link as Pydantic models, matching architecture-v5
- **Store**: SQLite (blocks + links tables, WAL mode) + FAISS index + FTS5
- **Vault adapter**: Read Obsidian .md files, split by headings, extract tags/links/dates
- **Pipeline Layer 0**: Ingest → split → extract tags (as tag blocks + links) → extract wikilinks → dedup hash → FTS index
- **Pipeline Layer 1**: Embed (pluggable model, default sentence-transformers) → hub scoring via link aggregation
- **Model abstraction**: EmbeddingModel + LLMModel protocols, factory functions, TOML config. Ship sentence-transformers + OpenAI embedding adapters. LLM adapters defined but not invoked (Layer 2 deferred).
- **Incremental ingestion**: File-level hash to skip unchanged files. Block-level content hash to preserve unchanged entries within changed files.
- **MCP server**: 6 tools (search, get_block, get_related, traverse, get_context, recent)
- **CLI**: `openaugi ingest --path ~/vault`, `openaugi serve`, `openaugi search`, `openaugi hubs`, `openaugi status`
- **Repo scaffolding**: `src/` layout, pyproject.toml (hatchling+uv), MIT license, CI, CLAUDE.md, CONTRIBUTING.md, issue/PR templates

### What Does NOT Ship in M0

- Adapter protocol / registry / entry points (M1 — emerges from second source)
- ChatGPT / Claude / Readwise adapters (M1+)
- Layer 2 processing: entity extraction, hub summaries, clustering (M2)
- Lenses (M3+)
- Daemon / file watcher / scheduler
- Task system (private repo)
- Config wizard / onboarding CLI

---

## M1 — Multi-Source

Add ChatGPT JSON export as the second source. The adapter abstraction emerges naturally from having two concrete implementations. If hub scoring surfaces real cross-source connections, the thesis is proven.

**Outcome:** Vault + ChatGPT history in one graph. Adapter base class extracted. Cross-source hubs work.

**Scope:**
- ChatGPT adapter: parse export JSON → document blocks → entry blocks (turn-level split)
- Extract adapter base class from vault + chatgpt implementations
- Validate hub scoring across sources (entries with no wikilinks still get hub edges via tags + clusters)
- Maybe: Claude export adapter, Google Drive static ingest

---

## M2 — Enrichment

Layer 2 processing. LLM-powered extraction and summarization, applied selectively to top-N hubs.

**Outcome:** Entity extraction creates extraction blocks + entity links. Hub summaries become summary blocks in the graph. Clusters group semantically similar entries.

**Scope:**
- Entity extraction via LLM (ontology-guided, structured output)
- Hub summaries as `Block(kind=summary)` + `summarizes` links
- Clustering (HDBSCAN on embeddings → cluster blocks + member_of links)
- Block-level summaries (short self-description per entry, on-demand)

---

## M3 — Temporal Intelligence

The moat. Time-aware analysis over the graph. Mostly free SQL aggregations over data that already exists.

**Outcome:** Claude can answer "what have I been thinking about?" and "what threads did I drop?" with grounded, time-aware answers.

**Scope:**
- Hub velocity (rising vs declining topics)
- Recurrence detection (topics that keep coming back)
- Dead stream detection (threads you dropped)
- Centroid drift (your thinking is shifting on this topic)
- New MCP tools for temporal queries
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

## Repo Strategy

| Repo | Role |
|------|------|
| `openaugi` (public) | The trunk. Clean orphan branch, MIT, port v1 in. This is what ships. |
| `openaugi-private` | Dev workspace. augi-engine-v1 lives here. Experiments stay here. |
| `openaugi-ai-chat-to-second-brain` | Extract ChatGPT adapter from here for M1. |

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
