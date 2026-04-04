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
│   ├── embed.py           # Layer 1 embedding step → vec_blocks (sqlite-vec)
│   ├── rerank.py          # Dedup + MMR re-ranking for get_context
│   └── watcher.py         # File watcher — debounced incremental ingest on vault changes
├── store/
│   └── sqlite.py          # SQLite backend (WAL, FTS5, sqlite-vec vec0, CASCADE)
├── models/
│   ├── __init__.py        # Factory: get_embedding_model(), get_llm_model()
│   └── embeddings/
│       ├── sentence_transformer.py  # Local default (free)
│       └── openai.py               # OpenAI API adapter
├── mcp/
│   ├── server.py          # MCP tools (read + write + streams), stdio + streamable-http transport
│   ├── doc_writer.py      # VaultWriter — writes .md to OpenAugi/ in vault
│   └── stream_manager.py  # StreamManager — workstream CRUD (OpenAugi/Streams/)
├── cli/
│   └── main.py            # typer CLI (up, ingest, serve, watch, search, hubs, status, service)
└── config.py              # TOML config loader + .env loader
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
    → write float32 blobs to blocks.embedding + vec_blocks (sqlite-vec)
```

### Query (MCP)

```
Claude → MCP tool call → server.py
  → search: sqlite-vec KNN (semantic) or FTS5 (keyword) + filters
  → get_block / get_blocks: full content by ID (single or batch)
  → get_related: follow links from/to a block
  → traverse: multi-hop graph walk
  → get_context: FTS + semantic (3× overfetch)
                 → deduplicate (cosine grouping, rerank.py)
                 → MMR re-rank
                 → expand via links
  → recent: recently created blocks
  → write_document / write_thread / write_snip: save notes to vault
  → list_streams / get_stream_context / make_stream / update_stream: workstream CRUD
```

### Hub Scoring

Pure SQL aggregation at query time (no stored table):
```
hub_score = w_in * ln(1 + inbound_links) + w_out * ln(1 + outbound_links) + w_ent * ln(1 + entry_count)
```

## Design Decisions

See [docs/plans/m0.md](docs/plans/m0.md) § Key Design Decisions for full rationale.

- **SQLite over DuckDB**: WAL mode concurrent writes. DuckDB is single-writer.
- **sqlite-vec over FAISS**: KNN via `vec0` virtual table — everything in one file, no separate index management. Embeddings normalized on write so L2 distance ≡ cosine.
- **Content hash as block identity**: `hash(source_path + content_hash)` — stable across section reordering.
- **Tags as blocks**: First-class graph nodes. Hub scoring, traversal, entity resolution work uniformly.
- **Default local embeddings**: sentence-transformers, no API key. Users upgrade via config.
- **`get_context` dedup + MMR**: Over-fetches 3× candidates, collapses near-duplicates via cosine grouping, re-ranks for diversity before returning. See [docs/MCP_SERVER.md](docs/MCP_SERVER.md) for tuning.

## Running

### Quick start (one command)

```bash
openaugi init          # one-time: configure embedding model, API key, vault path
openaugi up            # daily: sync vault + file watcher + MCP server
```

`openaugi up` is the single command to run OpenAugi:

1. **Incremental ingest** — syncs vault to SQLite (skips unchanged files via content hash)
2. **File watcher** — daemon thread watches for `.md` changes, debounces (default 30s), re-ingests
3. **MCP server** — foreground, stdio or HTTP transport

Embedding is attempted with the user's configured model. If it fails, blocks are saved without embeddings and retried on the next watcher cycle. SQLite WAL mode handles concurrent reads (MCP) and writes (watcher) without locking.

### Individual commands

| Command | What |
|---------|------|
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only (incremental ingest on vault changes) |
| `openaugi up` | Both in one process |

### Transports

Two transport modes — see [docs/REMOTE_ACCESS.md](docs/REMOTE_ACCESS.md) for full setup.

| Transport | Command | Use Case |
|-----------|---------|----------|
| stdio (default) | `openaugi up` | Claude Desktop/Code on same machine |
| streamable-http | `openaugi up --transport http` | Remote clients, Claude mobile via Cloudflare Tunnel |

Service management (macOS): `openaugi service install/uninstall/status` — launchd plist, starts on boot.

## Plans

- [docs/plans/overall-mvp.md](docs/plans/overall-mvp.md) — Milestones M0–M5
- [docs/plans/future-work.md](docs/plans/future-work.md) — Deferred features
