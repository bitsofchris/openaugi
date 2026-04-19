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

**Block kinds:** data_block, context_block:document, context_block:tag
**Link kinds:** contains, groups, links_to

Everything is a block. Structure lives in the links, not in the schema.

See [docs/data-model.md](docs/data-model.md) for the full data model philosophy.

## Processing Layers

| Layer | Plane | Cost | What | Requires |
|-------|-------|------|------|----------|
| **Layer 0** | Data | FREE | Split, tag/link extract, FTS, dedup hash, zzz dispatch | Python + SQLite |
| **Layer 1** | Data | ~$0 | Embed (local default), hub scoring (SQL) | sentence-transformers (local) |
| **Agent** | Agent | per-task | Execute zzz-dispatched tasks in tmux Claude sessions | Claude Code CLI + tmux |

## Two Planes

The codebase separates two fundamentally different kinds of work, both run by `openaugi up`:

**Data plane (`pipeline/`)** — passive transforms on blocks. Ingest, embed, watch for file changes, dispatch zzz instructions as task files, re-rank search results. All in-process Python, no external processes.

**Agent plane (`agents/`)** — task dispatch watches `OpenAugi/Tasks/` for pending task files and launches Claude Code sessions in tmux. The agent's behavior is governed by `templates/augi-agent.md` (copied to `<vault>/OpenAugi/augi-agent.md` on init).

The two planes share the store and the block/link data model. The bridge between them is `pipeline/dispatch.py`: when ingest finds blocks with `zzz:` instructions, it writes task files that the agent plane picks up. The MCP server (`mcp/`) sits alongside both as the read/write API surface that Claude calls.

## Module Map

```
src/openaugi/
├── model/
│   ├── block.py          # Block Pydantic model
│   ├── link.py           # Link Pydantic model
│   └── protocols.py      # EmbeddingModel, LLMModel protocols
├── adapters/
│   ├── splitter.py       # Deterministic block splitter — shared primitive (see docs/splitter.md)
│   └── vault.py          # Obsidian vault → blocks + links (wraps splitter)
├── pipeline/              # Data plane — transforms on blocks + zzz dispatch
│   ├── runner.py          # Layer 0 orchestrator (incremental ingestion)
│   ├── embed.py           # Layer 1 embedding step → vec_blocks (sqlite-vec)
│   ├── dispatch.py        # Post-ingest: zzz instructions → task files in OpenAugi/Tasks/
│   ├── rerank.py          # Dedup + MMR re-ranking for get_context
│   ├── vault_render.py    # Vault rendering — write blocks as .md to OpenAugi/Compiled/ (future)
│   └── watcher.py         # File watcher — debounced incremental ingest + zzz dispatch
├── agents/                # Agent plane — launches Claude Code sessions
│   └── task_watcher.py    # Task dispatch — OpenAugi/Tasks/ → tmux-hosted Claude sessions
├── store/
│   └── sqlite.py          # SQLite backend (WAL, FTS5, sqlite-vec vec0, CASCADE)
├── models/
│   ├── __init__.py        # Factory: get_embedding_model(), get_llm_model()
│   ├── embeddings/
│   │   ├── sentence_transformer.py  # Local default (free)
│   │   └── openai.py               # OpenAI API adapter
│   └── llms/
│       └── openai.py               # OpenAI-compatible LLM (gpt-5.4-nano default)
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
    → split by H3 dates → data_block blocks
    → extract tags → context_block:tag blocks + groups links
    → extract [[wikilinks]] → links_to links
    → context_block:document block + contains links
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

### ZZZ Dispatch (zzz → task file → agent)

```
you write `zzz: <instruction>` in a vault note
  → file watcher detects change (30s debounce)
  → ingest: parse, split, extract blocks + tags + links
  → pipeline/dispatch.py: blocks with zzz_instructions
    → write task file to OpenAugi/Tasks/<slug>.md (status: pending)
  → agents/task_watcher.py picks it up (5s poll, 30s settle)
    → hydrate: assign task_id, flip status→active, inject ## Session
    → resolve working dir via OpenAugi/Repos.md
    → build prompt: augi-agent skill file + task body + linked notes
    → launch tmux: detached session + `claude "$(cat ctx)"`
  → agent reads skill file, uses MCP tools, does the work
  → writes output to OpenAugi/ tagged #human-review
  → marks task file status: done
```

Per-block `zzz:` lines are extracted by the vault adapter into
`metadata["zzz_instructions"]` (a list, one item per line; stripped from
the clean content). Blocks are split on `###` headers and `qqq` markers
(case-insensitive) — see [docs/plans/zzz-instructions.md](docs/plans/zzz-instructions.md).

The agent's behavior is governed by `templates/augi-agent.md` (source of
truth, copied to `<vault>/OpenAugi/augi-agent.md`). Edit that file to
change how the agent handles tasks — not the Python code.

The **task file format is a single contract** defined in
`src/openaugi/templates/task-template.md` and enforced by
`test_task_template_hydrates_cleanly`. Both `dispatch.py` (writer)
and `task_watcher.py` (reader) point at that template.

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

### Daily use — one command

```
openaugi up     ← ingest + watcher + zzz dispatch + task agent + MCP server
```

### All commands

| Command | What |
|---------|------|
| `openaugi up` | Ingest + watcher + zzz dispatch + task agent + MCP server |
| `openaugi up --no-agent` | Same but without task dispatch (no tmux agent sessions) |
| `openaugi task-dispatch` | Watch `OpenAugi/Tasks/` and launch pending tasks in tmux (standalone) |
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only (incremental ingest on vault changes) |
| `openaugi re-embed` | Reset + re-embed all data blocks with current model (use after model switch) |
| `openaugi cluster` | Run HDBSCAN clustering DAG → write `context_block:cluster` nodes to DB |
| `openaugi cluster --dry-run` | Compute clusters + print stats, no DB writes (use for param tuning) |

### Transports

Two transport modes — see [docs/REMOTE_ACCESS.md](docs/REMOTE_ACCESS.md) for full setup.

| Transport | Command | Use Case |
|-----------|---------|----------|
| stdio (default) | `openaugi up` | Claude Desktop/Code on same machine |
| streamable-http | `openaugi up --transport http` | Remote clients, Claude mobile via Cloudflare Tunnel |

Service management (macOS): `openaugi service install/uninstall/status` — launchd plist, starts on boot.

## Plans

- [docs/plans/m2-feature-roadmap.md](docs/plans/m2-feature-roadmap.md) — Post-launch roadmap (Ship → Show → Adapt → Deepen → Differentiate → Lenses → Expand)
- [docs/plans/phase3-adapters.md](docs/plans/phase3-adapters.md) — Phase 3: multi-source ingest adapters (ChatGPT, Readwise, Research, LlamaIndex bridge)
- [docs/plans/done/heartbeat.md](docs/plans/done/heartbeat.md) — (shipped, then replaced by zzz dispatch) Heartbeat design history
- [docs/plans/capture-tag-stream-loop.md](docs/plans/capture-tag-stream-loop.md) — Phase 4: capture → tag → stream incremental pipeline
- [docs/plans/from-capture-to-jarvis.md](docs/plans/from-capture-to-jarvis.md) — Longer-horizon vision (layers 1–4)
- [docs/plans/future-work.md](docs/plans/future-work.md) — Deferred features
- [docs/clustering.md](docs/clustering.md) — Clustering feature: config format, data model, SQL queries, param tuning guide
- [docs/plans/hierarchical-embeddings.md](docs/plans/hierarchical-embeddings.md) — Design rationale: two-pass strategy, matryoshka truncation, bridge detection
- [docs/plans/done/](docs/plans/done/) — Shipped milestone plans (M0, M1)
