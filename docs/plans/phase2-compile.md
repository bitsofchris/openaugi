---
name: phase2-compile
description: Plan for the context block layer — materializing navigational metadata from raw data blocks via openaugi compile
---

# Phase 2: Compile — The Context Block Layer

*Created: 2026-04-05 | Updated: 2026-04-05 (decisions locked)*

---

## What Is Compile?

Compile is the process of building the **navigational metadata layer** from raw data blocks.

Your raw data (notes, entries, tags) lives in blocks. Compile produces **context blocks** — synthesized, deduplicated, linked representations that give agents a map to navigate by and give humans a browsable view of their knowledge.

This is NOT a rendering step. It's not "export to markdown." It's the core operation that transforms raw blocks into the LLM's representation of your data — separate from what you write, but built from it.

The vault markdown rendering (`OpenAugi/Compiled/`) is one **output format** of context blocks. MCP tools querying context blocks is another.

### The Karpathy Parallel

Karpathy describes: `raw/ → LLM "compiles" wiki → .md files in directory structure`

Our version:
```
data blocks (your notes, entries, tags)
  → compile (SQL aggregation + templates + optional LLM)
  → context blocks in SQLite (the primary home, queryable via MCP)
  → render to vault as browsable .md (OpenAugi/Compiled/)
```

The key difference: his "compile" is a monolithic LLM pass. Ours is layered — free SQL context blocks first, LLM enrichment later. You get value before spending tokens.

---

## Decisions (Locked)

| Decision | Choice | Rationale |
|---|---|---|
| MCP approach | New `get_index` tool, leave `get_context` unchanged | Learn how agents use it before changing core retrieval. Less risk. |
| Compile on `up` | Layer 0 auto-runs after ingest, before MCP server starts | It's fast SQL (<1s). Agent always has a fresh map. |
| Vault rendering | Always render on compile (not a separate flag) | Rendering is trivial once context blocks exist. Ship the full "Show." |
| Feedback loop | Option A — exclude `OpenAugi/Compiled/` from ingest | Simple. No edge cases. Revisit if users want compiled content in MCP search. |
| Vault structure | Flat current view, overwritten on each compile | Timestamp/snapshot hierarchy deferred to Phase 5 (temporal intelligence). |
| `get_context` | No changes in Phase 2 | Add context-block-aware routing later once we see usage patterns. |

---

## What Is a Context Block?

A context block is a `Block(kind=context)` that contains **routing signals** about one or more data blocks. It doesn't duplicate raw content. It answers: "Should the agent look here? What will it find?"

Context blocks have a `context_type` field in metadata that describes what kind of context block it is.

### Context Block Types (Phase 2)

| context_type | What It Describes | Compile Layer | Cost |
|---|---|---|---|
| `index` | Master map listing all other context blocks with one-line descriptions | Layer 0 | FREE |
| `hub_summary` | A top tag/hub: entry count, link count, last active, co-occurring tags, velocity | Layer 0 | FREE |
| `recent_activity` | Blocks created/modified in last 7/30 days, grouped by source and top tags | Layer 0 | FREE |
| `concept` | All entries under a hub + their links + related hubs, rendered as navigable page | Layer 1 | FREE |
| `graph_health` | Orphan blocks, stale tags, hub health metrics | Layer 1 | FREE |

Layer 0 = pure SQL aggregation. Layer 1 = SQL + templates + embedding queries.

### Future Context Block Types (Later Phases)

| context_type | Phase | Cost |
|---|---|---|
| `stream_status` | Phase 5 (Workstreams) | FREE |
| `hub_narrative` | Phase 4 (Enrichment) | $$$ |
| `connection_map` | Phase 4 (Enrichment) | $$$ |
| `snapshot` | Phase 5 (Temporal) | FREE |

---

## Data Model

Context blocks use the existing blocks+links schema. No new tables.

```
Block(
    kind="context",
    content="<the synthesized content>",
    metadata={
        "context_type": "hub_summary",
        "scope": "tag:context-engineering",
        "compile_layer": 0,
        "compiled_at": "2026-04-05T10:00:00",
        "source_count": 42,
    },
    source="compiled",
    ...
)

Link(from_id=context_block_id, to_id=data_block_id, kind="summarizes")
Link(from_id=context_block_id, to_id=other_context_block_id, kind="links_to")
```

- **`kind=context`** — one block kind, differentiated by `context_type` in metadata.
- **`source=compiled`** — all context blocks marked. Filtered from compile inputs. Weighted below original content in retrieval.
- **`summarizes` links** — every context block links to the data blocks it was built from. Full provenance.
- **Context blocks link to each other** — hub_summary blocks link to related hub_summaries via `links_to`. This IS the context graph.

---

## What `openaugi compile` Produces

### Layer 0 (pure SQL)

**`hub_summary` context blocks** (one per top-N hub, default N=50):

Content includes: entry count, inbound/outbound link count, last active timestamp, co-occurring tags (top 5), entry velocity (entries per week over last 4 weeks).

**`recent_activity` context block** (one, refreshed each compile):

Content includes: blocks created/modified in last 7 and 30 days, grouped by source file and top tags, total counts, most active areas.

**`index` context block** (one, the master map):

Lists all other context blocks with one-line descriptions. This is what the agent reads first via `get_index`.

### Layer 1 (SQL + templates)

**`concept` context blocks** (one per top-N hub):

Template-rendered page listing all entries under a hub, their first lines, tag co-occurrences, related hubs by shared entries, link counts. No LLM — structured data in a readable format.

**`graph_health` context block** (one):

Orphan blocks (zero links), stale tags (no new blocks in 4+ weeks), hub health metrics.

---

## Vault Rendering

Compile always renders context blocks as `.md` to the vault:

```
OpenAugi/Compiled/
├── INDEX.md              # Master index — the agent's (and human's) entry point
├── RECENT.md             # Recent activity — last 7/30 days
├── HEALTH.md             # Graph health — orphans, stale tags, metrics
└── concepts/
    ├── context-engineering.md
    ├── augmented-intelligence.md
    └── ...               # One per top-N hub
```

`OpenAugi/Compiled/` is excluded from ingest via path filter in the vault adapter. Overwritten on each compile.

### Phase 5 Evolution

When temporal intelligence ships, the structure evolves:

```
OpenAugi/Compiled/
├── latest/               # current view (what Phase 2 produces)
│   ├── INDEX.md
│   ├── concepts/...
│   └── ...
└── snapshots/            # periodic snapshots for browsing history
    ├── 2026-04-05/
    └── 2026-04-12/
```

---

## CLI Interface

```bash
openaugi compile                    # Layer 0 + Layer 1 + render to vault
openaugi compile --layer 0          # Layer 0 only (fast, what `up` runs)
openaugi compile --hub "topic"      # Compile specific hub only
openaugi compile --force            # Skip freshness check, recompile all
```

### Integration

- **`openaugi up`** runs Layer 0 compile after ingest, before starting MCP server
- **File watcher** triggers Layer 0 recompile after each ingest cycle (debounced)

---

## MCP Tool: `get_index`

New tool. Returns the master index context block — the agent's entry point to navigate the knowledge graph.

```
get_index() → content of the index context block
```

The agent calls `get_index` first, reads the map of hubs/concepts/recent activity, then uses existing tools (`search`, `get_block`, `get_related`, `traverse`) to drill in.

`get_context` remains unchanged. We observe how agents use `get_index` before deciding whether to add context-block routing to `get_context`.

---

## Implementation Sequence

### Step 1: Context block infrastructure
- [ ] Add `kind=context` to block model (add to `BlockKind` enum or validation)
- [ ] Add `summarizes` to link kind validation
- [ ] Add `source=compiled` filter to store query methods
- [ ] Add `OpenAugi/Compiled/` to vault adapter ignore list

### Step 2: Layer 0 compile
- [ ] Hub summary SQL query → `hub_summary` context blocks (top 50)
- [ ] Recent activity SQL query → `recent_activity` context block
- [ ] Index builder → `index` context block referencing all others
- [ ] `pipeline/compile.py` — orchestrator for compile steps
- [ ] `openaugi compile` CLI command

### Step 3: Layer 1 compile
- [ ] Concept page templates → `concept` context blocks
- [ ] Graph health query → `graph_health` context block

### Step 4: Vault rendering
- [ ] Render context blocks as `.md` to `OpenAugi/Compiled/`
- [ ] Run rendering as final step of compile

### Step 5: MCP + integration
- [ ] `get_index` MCP tool
- [ ] Layer 0 compile in `openaugi up` (after ingest, before MCP server)
- [ ] Layer 0 recompile in watcher cycle (after ingest, debounced)

---

## Testing

### Unit tests
- Hub summary query returns correct counts, velocities, co-occurring tags for fixture vault
- Recent activity query returns blocks within time windows
- Index builder aggregates all context blocks into structured index
- Graph health identifies known orphans and stale tags in fixture vault
- Concept template renders expected structure

### Integration tests
- Full cycle: ingest fixture vault → compile → verify context blocks in SQLite with correct metadata
- Verify `summarizes` links connect context blocks to source data blocks
- Verify `OpenAugi/Compiled/` contains expected `.md` files with correct content
- Verify `OpenAugi/Compiled/` files are excluded from re-ingest
- `get_index` MCP tool returns the index context block

### Manual testing / showcase
- Run on Chris's real vault (~5,000 notes)
- Screenshot `OpenAugi/Compiled/` in Obsidian — browsable, linkable, shareable
- Demo: `openaugi up` → compile runs → `get_index` in Claude → agent navigates hubs → drills into entries
- Benchmark: compile time on real vault (target: <5s for Layer 0, <30s for Layer 0+1)
