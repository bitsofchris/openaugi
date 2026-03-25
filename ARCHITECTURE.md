---
name: architecture
description: System architecture for OpenAugi — blocks+links data model, processing layers, module map
---

# OpenAugi Architecture

A self-hostable personal intelligence engine. One `pip install`. One SQLite file. One MCP server.

## Data Model

Two tables. That's the whole store.

```
blocks (id, kind, content, summary, embedding, source, title, tags, timestamp, metadata, content_hash)
links  (from_id, to_id, kind, weight, metadata)  — PK: (from_id, to_id, kind)
```

**Block kinds:** document, entry, tag, cluster (M2), summary (M2)
**Link kinds:** split_from, tagged, links_to, member_of (M2), summarizes (M2)

Everything is a block. Structure lives in the links, not in the schema.

## Processing Layers

| Layer | Cost | What | Requires |
|-------|------|------|----------|
| **Layer 0** | FREE | Split, tag/link extract, FTS, dedup hash | Python + SQLite |
| **Layer 1** | ~$0 | Embed (local default), hub scoring (SQL) | sentence-transformers (local) |
| **Layer 2** | $$$ | Entity extraction, summaries, clustering | LLM (deferred to M2) |

## Module Map

```
src/openaugi/
├── model/
│   ├── block.py          # Block Pydantic model
│   ├── link.py           # Link Pydantic model
│   └── protocols.py      # EmbeddingModel, LLMModel protocols
├── adapters/
│   └── vault.py          # Obsidian vault → blocks + links
├── pipeline/
│   ├── runner.py          # Layer 0 orchestrator (incremental ingestion)
│   └── embed.py           # Layer 1 embedding step + FAISS build
├── store/
│   ├── sqlite.py          # SQLite backend (WAL, FTS5, CASCADE)
│   └── faiss.py           # FAISS vector index wrapper
├── models/
│   ├── __init__.py        # Factory: get_embedding_model(), get_llm_model()
│   └── embeddings/
│       ├── sentence_transformer.py  # Local default (free)
│       └── openai.py               # OpenAI API adapter
├── mcp/
│   └── server.py          # 6 MCP tools for Claude
├── cli/
│   └── main.py            # typer CLI
└── config.py              # TOML config loader
```

## Key Flows

### Ingest (Layer 0 + 1)

```
Vault .md files
  → parse_vault_incremental()  [adapters/vault.py]
    → file hash check (skip unchanged)
    → split by H3 dates → entry blocks
    → extract tags → tag blocks + tagged links
    → extract [[wikilinks]] → links_to links
    → document block + split_from links
  → insert blocks + links  [store/sqlite.py]
  → FTS5 auto-indexed via triggers
  → run_embed()  [pipeline/embed.py]
    → embed blocks where embedding IS NULL
    → build FAISS index
```

### Query (MCP)

```
Claude → MCP tool call → server.py
  → search: FAISS (semantic) or FTS5 (keyword) + filters
  → get_block: full content by ID
  → get_related: follow links from/to a block
  → traverse: multi-hop graph walk
  → get_context: compound search → expand → structured result
  → recent: recently created blocks
```

### Hub Scoring

Pure SQL aggregation at query time (no stored table):
```
hub_score = w_in * ln(1 + inbound_links) + w_out * ln(1 + outbound_links) + w_ent * ln(1 + entry_count)
```

## Design Decisions

See [docs/plans/m0.md](docs/plans/m0.md) § Key Design Decisions for full rationale.

- **SQLite over DuckDB**: WAL mode concurrent writes. DuckDB is single-writer.
- **Content hash as block identity**: `hash(source_path + content_hash)` — stable across section reordering.
- **Tags as blocks**: First-class graph nodes. Hub scoring, traversal, entity resolution work uniformly.
- **Default local embeddings**: sentence-transformers, no API key. Users upgrade via config.
- **6 MCP tools (down from v1's 13)**: `get_context` is the power tool. `get_related` + `traverse` replace several v1 tools.

## Plans

- [docs/plans/overall-mvp.md](docs/plans/overall-mvp.md) — Milestones M0–M4
- [docs/plans/m0.md](docs/plans/m0.md) — M0 detailed plan (current)
- [docs/plans/future-work.md](docs/plans/future-work.md) — Deferred features
