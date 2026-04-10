---
name: context-block-architecture
description: Align OpenAugi's data model terminology with the Context Engineering post, then extend to richer context blocks (cluster and hub summaries).
---

# Context Block Architecture

Two phases: first align language, then enrich the context layer.

---

## Phase 1 — Terminology Alignment ✓ Implemented

Current impl already has the right structure. This phase makes the code and docs speak the same language as the post so the architecture is legible.

> **Migration complete** — all source code, tests, and docs updated to use new kind strings.

### Rename block kinds

| Current `kind` | New `kind` | What it is |
|---|---|---|
| `entry` | `data_block` | Raw content unit — the thing agents eventually read |
| `document` | `context_block:document` | Identity + path — routes agent to a file |
| `tag` | `context_block:tag` | Tag node — routes agent to a concept cluster |

**Migration:** add a `kind` prefix for context blocks, keep `data_block` as the primary embeddable kind. `get_blocks_needing_embeddings` and heartbeat continue filtering on the data block kind only.

### Rename link kinds

| Current `kind` | New `kind` | Meaning |
|---|---|---|
| `split_from` | `contains` | context_block:document → data_block |
| `tagged` | `groups` | data_block → context_block:tag |
| `links_to` | `links_to` | lateral wikilink between data_blocks (keep as-is) |

### Docs + MCP tool descriptions

- Update `docs/data-model.md` to use data block / context block / context graph vocabulary
- Update MCP tool docstrings in `server.py` to name the block kinds correctly
- Update `ARCHITECTURE.md` intro paragraph to reference the four-layer framing

### What does NOT change

- Schema (two tables, `blocks` + `links`) — no migration needed
- Embedding pipeline — still only embeds data blocks
- Heartbeat — still queries data blocks by timestamp
- All retrieval modes (semantic, keyword, graph, time, lookup) — unchanged

---

## Phase 2 — Richer Context Blocks

Context blocks today are thin identity nodes. This phase generates richer routing signals so agents can make scoping decisions before touching data blocks.

Two new context block subtypes, both generated offline (not on ingest):

### 2a — Cluster summaries (embedding space) → see [hierarchical-embeddings.md](hierarchical-embeddings.md)

Full design moved to [docs/plans/hierarchical-embeddings.md](hierarchical-embeddings.md).

Summary: multi-level `context_block:cluster` hierarchy generated from agglomerative/HDBSCAN clustering over data block embeddings. Includes LLM-generated summaries, temporal analysis, and `contains` + `groups` links forming a navigable hierarchy. Re-embed pipeline (title-prepend + `text-embedding-3-large`) is shipped; clustering exploration is the next step.

### 2b — Hub summaries (link graph) ← outstanding

- Identify hub documents/tags by inbound link count (query infrastructure already in `hubs` CLI command)
- For each hub above a threshold: generate a `context_block:hub` with a summary — what connects here, what topics dominate, recent activity
- Link: `context_block:hub --summarizes--> context_block:document|tag`
- Refresh: incremental — only regenerate hubs whose link topology changed since last run

This is graph topology, not embedding space — independent of Phase 2a and can be done separately.

### MCP tools (spans both phases)

- `get_context_map` — return all context blocks (clusters + hubs) for agent orientation pass; fully useful only after both 2a and 2b are complete
- `get_cluster` — return a cluster context block + its member data block IDs (Phase 2a)
- `get_hub` — return a hub context block + its top connected nodes (Phase 2b)

### Agent query flow (post Phase 2)

```
1. get_context_map        → agent sees clusters + hubs, picks where to look
2. get_cluster / get_hub  → agent scopes to relevant area
3. search / traverse      → agent retrieves specific data blocks
4. get_block              → agent reads raw content
```

This is the Map → Assess → Drill → Retrieve pattern from the post, implemented directly.

---

## Out of scope (future)

- Context snapshots / time travel — track context block state over time
- Cross-source context blocks (non-vault data)
- Streaming / incremental cluster updates
