---
name: m2-feature-roadmap
description: Post-launch feature roadmap — phased plan from public launch through context block layer, adapters, enrichment, and temporal intelligence
---

# Feature Roadmap — Post-Launch

*Created: 2026-04-05 | Updated: 2026-04-08*

---

## Context

OpenAugi shipped as a working personal intelligence engine: one `pip install`, one SQLite file, one MCP server. The foundation is solid — blocks+links data model, incremental vault ingest, semantic + keyword search, MCP read/write tools, file watcher, HTTP transport.

The next phase builds the **context block layer** — the LLM's representation of your data. Not a copy, not an export. A synthesized, deduplicated, linked layer that sits above your raw blocks and gives agents (and humans) a map to navigate by.

This architecture aligns with the [context engineering as index design](https://bitsofchris.com/p/context-engineering-is-index-design) thesis:

| Concept | OpenAugi Today | What We're Building |
|---|---|---|
| **Data Blocks** | `blocks` (document, entry) — raw content + metadata | Already done |
| **Context Blocks** | Tags (partial), hub scores (query-time) | Materialized routing signals — hub summaries, concept pages, stream status |
| **Context Graph** | `links` table (split_from, tagged, links_to) | Links between context blocks, not just data blocks |
| **Context Snapshots** | Nothing | Daily snapshots of context blocks for temporal tracking |

---

## Current State

| Milestone | Status |
|-----------|--------|
| M0 — Port & Foundation (blocks+links, SQLite, vault adapter, MCP read tools) | Done |
| M1 — MCP Write + Local Setup (write_document, vault resource, Claude registration) | Done |
| M1.5 — Ship & Run (HTTP transport, file watcher, launchd service, `openaugi up`) | Done |
| v0.1.0 — Public Launch (README, GETTING_STARTED.md, PyPI) | Done |
| Phase 2 — Context Block Layer (compile, context blocks, vault rendering, `get_index` MCP tool) | Done |

---

## Phases

### Phase 1: Ship ✓

v0.1.0 live on PyPI. `pip install openaugi && openaugi init && openaugi up` works.

---

### Phase 2: Show — Context Block Layer ✓

**Status:** Shipped. [src/openaugi/pipeline/compile.py](../../src/openaugi/pipeline/compile.py) implements all Layer 0 + Layer 1 context block types (`hub_summary`, `recent_activity`, `index`, `concept`, `graph_health`). Vault rendering writes to `OpenAugi/Compiled/`. The `get_index` MCP tool is live at [src/openaugi/mcp/server.py:557](../../src/openaugi/mcp/server.py#L557). Layer 0 compile runs automatically after ingest in `openaugi up`.

Key decisions (as shipped):
- Context blocks live in SQLite as `Block(kind=context)` with `metadata["context_type"]` differentiating subtypes
- `openaugi compile` materializes context blocks from data blocks at progressive cost layers (free SQL → templates)
- Compile always renders to vault — `OpenAugi/Compiled/` is excluded from ingest via path filter
- LLM-powered context block types (`hub_narrative`, `connection_map`) deferred to Phase 4

---

### Phase 3: Adapt — New Ingest Sources

**Goal:** Ingest beyond Obsidian. Research collections, AI chat exports, PDFs, web articles.

**Why:** Your knowledge isn't just in Obsidian. Each new source makes the context graph richer and the context blocks more useful.

**Plan:** [docs/plans/phase3-adapters.md](phase3-adapters.md)

Adapters to build (decided 2026-04-05):
1. ChatGPT export → message-level blocks tagged by role (user messages are first-class)
2. Readwise API → highlight-level entries, incremental via API timestamp
3. Research output → saved AI research as `.md`, with citation extraction (papers, URLs as reference blocks)
4. LlamaIndex bridge → optional dependency, translates any LlamaIndex reader output to our blocks (ecosystem access without coupling)

Architecture direction:
- `SourceAdapter` protocol in `protocols.py` — each adapter is a black box returning `(blocks, links, current_ids, deleted_ids)`
- `metadata["source_id"]` replaces `source_path` as canonical identity key across all adapters
- Config-driven: `[sources.*]` in `config.toml`, `openaugi ingest` runs all configured sources
- `openaugi init` walks through source setup
- Per-adapter config options with sensible defaults (pulled from Phase 7)
- Shared helpers for tag extraction, markdown splitting, wikilinks
- Cross-source dedup deferred to compile (semantic clustering, not ingest-time)

---

### Phase 4: Deepen — LLM Enrichment

**Goal:** LLM-powered context blocks. Hub narrative summaries, entity extraction, connection maps. Restore v1's most-used feature (`get_summary`) in the context block framework.

**Why:** Free SQL context blocks give structure. LLM context blocks give meaning. "What have I been thinking about?" deserves a genuinely good answer.

**Plan:** [docs/plans/capture-tag-stream-loop.md](capture-tag-stream-loop.md) — the capture → tag → stream loop is the concrete Phase 4 plan.

Partially landed: [src/openaugi/pipeline/enrich.py](../../src/openaugi/pipeline/enrich.py), [tag_inference.py](../../src/openaugi/pipeline/tag_inference.py), [taxonomy.py](../../src/openaugi/pipeline/taxonomy.py) exist; `openaugi enrich` CLI runs batch taxonomy discovery + tag inference via the agent or API flow. The missing piece is wiring tag inference incrementally into the ingest pipeline.

Key pieces:
- Incremental tag inference in `runner.py` after Layer 0 ingest — new blocks get `computed_tags` automatically
- `openaugi enrich --incremental` flag for re-tag passes on existing data
- `openaugi stream <tag>` CLI to prove the "tag = stream" claim at the CLI level
- Compile reads `computed_tags` in addition to source tags for hub scoring and concept pages
- LLM hub narratives — `Block(kind=context, context_type=hub_narrative)` — extraction not summarization, atomic ideas not TLDRs (lesson from v1)
- Entity extraction as stretch
- Agent-generated content always marked `source=agent` to prevent echo-chamber retrieval

---

### Phase 5: Differentiate — Workstreams + Temporal Intelligence

**Goal:** Two capabilities nobody else has. Workstreams answer "what am I working on and where did I leave off?" Temporal intelligence answers "how has my thinking changed?"

**Why:** A knowledge base answers "what do I know?" That's table stakes. Workstreams and temporal awareness are the moat.

**Plan:** [docs/plans/phase5-differentiate.md](phase5-differentiate.md) (to be created)

#### Workstreams

- Formalize streams as a concept in the data model (likely context blocks with stream-specific metadata: status, LEFT OFF, session history)
- Stream timeline — activity over time, entries, edits, gaps
- Active streams dashboard — all active streams at a glance
- Stream routing — auto-detect which stream a journal entry belongs to via tag matching
- Builds on existing MCP tools (`list_streams`, `make_stream`, `update_stream`)

#### Temporal Intelligence

- **Context snapshots** — daily snapshots of context blocks, tracking movement in embedding space over time
- `hub_velocity` — rising vs declining topics over sliding windows
- `recurrence` — topics that go quiet then return
- `centroid_drift` — how thinking on a topic shifts over time (embedding distance between snapshots)
- Hub snapshots table — periodic centroid embeddings for drift analysis
- `openaugi trends` CLI command

---

### Phase 6: Lenses — Proactive Analysis

**Goal:** Automated analysis that looks at your data and offers insights. The janitor that keeps the graph clean, the advisor that spots patterns you're blind to.

**Why:** Agents are smart enough to reason over your context graph. Give them the right prompts and they become proactive collaborators.

**Plan:** [docs/plans/phase6-lenses.md](phase6-lenses.md) (to be created)

Layers:
- **Free lenses** (SQL): orphan blocks, stale streams, hub health, tag inconsistencies
- **LLM lenses**: consistency checks, connection suggestions, missing links, duplicate thinking detection
- **Personal rule lenses** (configurable, private): WIP limit violations, tornado detection, season checks — the framework ships open source, the personal rules stay in user config

Architecture: lenses are functions that read the store, apply logic, return results. Callable via MCP tools and CLI (`openaugi lint`). Users can add custom lenses later.

---

### Phase 7: Expand — Extensibility

**Goal:** Make the system extensible beyond built-in adapters and lenses.

**Why:** By this point we'll have multiple lens types and enrichment options. Per-adapter config shipped in Phase 3. What remains is the community/extensibility layer.

Direction:
- Custom lens framework (users write their own analysis functions)
- Community adapter registry/discovery
- `openaugi init` full wizard (model + adapters + lenses + enrichment in one flow)
- Pipeline orchestration (parallel adapters, dependency ordering) — if needed

Note: Per-adapter pipeline config (extraction depth, tagging rules) moved to Phase 3 — ships with each adapter.

This phase is NOT a prerequisite for anything above. It emerges naturally from building Phases 2-6.

---

## Cross-Cutting: Write-Back Flywheel

Every useful interaction should enrich the knowledge base by default.

- Compile output (context blocks) feeds back into the graph — richer index → better retrieval → richer compile
- Agent-generated content marked `source=agent`, weighted below original content in retrieval
- Session summaries as opt-in context blocks
- Prompt engineering in MCP tool descriptions encouraging write-back

This is mostly prompt engineering and MCP tool defaults, shipped incrementally across all phases.

---

## What We Are NOT Building

- Cloud deployment / SaaS (local-first, always)
- Web frontend (Obsidian + Claude are the interfaces)
- Premature framework / `pipeline.toml` before we have concrete adapters
- Personal productivity rules baked into the open-source tool (framework yes, rules no)

---

## Sequencing Summary

```
Done       → Phase 1: Ship v0.1.0
Done       → Phase 2: Context block layer + compile + vault rendering
Next       → Phase 3: New ingest adapters (ChatGPT, Readwise, Research, LlamaIndex bridge)
Next       → Phase 4: Capture → tag → stream loop + LLM enrichment
Later      → Phase 5: Workstreams + temporal intelligence
Later      → Phase 6: Proactive lenses
Ongoing    → Phase 7: Extensibility emerges from concrete implementations
Throughout → Write-back flywheel, incremental
```

---

## The Positioning

| What Others Have | What OpenAugi Has |
|---|---|
| Hacky scripts (Karpathy) | Productionized engine, one `pip install` |
| Static wiki / compiled markdown | Living context graph with materialized context blocks |
| Brute-force RAG | Navigational metadata layer — agents read a map before they retrieve |
| No temporal awareness | Context snapshots, drift detection, workstream tracking |
| No write-back | Explorations compound — every Q&A enriches the graph |
| Closed-source, cloud-locked | Open source, local-first, model-agnostic |

His is a library. Ours is a second brain with a pulse.
