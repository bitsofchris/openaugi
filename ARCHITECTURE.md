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

**Block kinds:** document, entry, tag
**Link kinds:** split_from, tagged, links_to

Everything is a block. Structure lives in the links, not in the schema.

See [docs/data-model.md](docs/data-model.md) for the full data model philosophy.

## Processing Layers

| Layer | Cost | What | Requires |
|-------|------|------|----------|
| **Layer 0** | FREE | Split, tag/link extract, FTS, dedup hash | Python + SQLite |
| **Layer 1** | ~$0 | Embed (local default), hub scoring (SQL) | sentence-transformers (local) |
| **Heartbeat** | FREE | Per-block classification → `augi_tags` on blocks, task dispatch | Claude Code agent |

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
│   ├── heartbeat.py       # Heartbeat orchestrator — new blocks → Claude Code agent session
│   ├── rerank.py          # Dedup + MMR re-ranking for get_context
│   ├── vault_render.py    # Vault rendering — write blocks as .md to OpenAugi/Compiled/ (future)
│   ├── watcher.py         # File watcher — debounced incremental ingest on vault changes
│   └── task_watcher.py    # Task dispatch — OpenAugi/Tasks/ → tmux-hosted Claude sessions (optional)
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
│   └── main.py            # typer CLI (up, ingest, serve, watch, heartbeat, search, hubs, status, service)
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

### Heartbeat (dumb script, smart agent)

See [docs/plans/heartbeat.md](docs/plans/heartbeat.md) for the full design.

```
openaugi heartbeat → pipeline/heartbeat.py
  → run incremental ingest (only if --ingest; default assumes `up` is running)
  → read ~/.openaugi/last_heartbeat timestamp
  → store.get_blocks_created_since(since) → entry blocks (capped at --max-blocks)
  → build prompt: skill file ref + per-block content + zzz_instructions metadata
  → spawn `claude -p <prompt>` with openaugi MCP tools allowed
  → agent classifies each block → calls tag_block MCP tool → augi_tags on block
  → agent dispatches tasks → writes OpenAugi/Tasks/<slug>.md
  → agent writes heartbeat log → OpenAugi/Heartbeat/YYYY-MM-DD.md
  → on success: advance ~/.openaugi/last_heartbeat
```

The Python side is deliberately dumb: it does not classify. The reasoning
lives in `<vault>/OpenAugi/heartbeat-skill.md`, a user-maintained markdown
file (template: `src/openaugi/templates/heartbeat-skill.md`). Per-block
`zzz:` lines are extracted by the vault adapter into
`metadata["zzz_instructions"]` (a list, one item per line; stripped from
the clean content) so the agent can honor each as an independent per-block
directive. Blocks are split on `###` headers and `qqq` markers
(case-insensitive) — see [docs/plans/zzz-instructions.md](docs/plans/zzz-instructions.md).

The tag taxonomy the skill applies — `area/*` evolution streams, `type/task`
for actionable items, and `status/*` task states — is documented in
[docs/taxonomy.md](docs/taxonomy.md). That doc is the single source of truth
for what `workstream:` means in task-dispatch frontmatter and for the
disambiguation between the block-level `status/*` tag facet and the
task-file `status:` frontmatter lifecycle.

### Task Dispatch (optional add-on)

See [docs/task-dispatch.md](docs/task-dispatch.md) for the full feature doc.

```
openaugi task-dispatch → pipeline/task_watcher.py
  poll loop (default every 5s):
  ├── scan_pending(OpenAugi/Tasks/, settle=30s)
  ├── hydrate_note(file) — assign task_id, flip status→active, inject ## Session
  ├── resolve_working_dir — OpenAugi/Repos.md short-name → absolute path
  ├── build_prompt — small wrapper around the task body
  └── launch_tmux — detached `tmux new-session` + send-keys `claude "$(cat ctx)"`
```

Optional add-on: if you don't run `openaugi task-dispatch`, task files from
heartbeat (or hand-written) just sit in `OpenAugi/Tasks/`. Opt in by
running the watcher. The **task file format is a single contract**
defined in `src/openaugi/templates/task-template.md` and enforced by
`test_task_template_hydrates_cleanly`. Both the heartbeat skill (writer)
and `task_watcher.py` (reader) point at that template rather than
redefine the format — see the contract diagram in [docs/task-dispatch.md](docs/task-dispatch.md).

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

### Daily use — two entry points

```
openaugi up     ← Claude Desktop MCP config (vault sync + MCP server)
openaugi agent  ← one terminal window (heartbeat every 5m + task dispatch)
```

### All commands

| Command | What |
|---------|------|
| `openaugi up` | Ingest + watcher + MCP server in one process |
| `openaugi agent` | Heartbeat loop (every `--interval` min) + task dispatch |
| `openaugi heartbeat` | One-shot: find new blocks → spawn Claude Code agent |
| `openaugi task-dispatch` | Watch `OpenAugi/Tasks/` and launch pending tasks in tmux |
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only (incremental ingest on vault changes) |

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
- [docs/plans/heartbeat.md](docs/plans/heartbeat.md) — Heartbeat: dumb script spawns a Claude Code agent session to process new blocks
- [docs/plans/capture-tag-stream-loop.md](docs/plans/capture-tag-stream-loop.md) — Phase 4: capture → tag → stream incremental pipeline
- [docs/plans/from-capture-to-jarvis.md](docs/plans/from-capture-to-jarvis.md) — Longer-horizon vision (layers 1–4)
- [docs/plans/future-work.md](docs/plans/future-work.md) — Deferred features
- [docs/plans/done/](docs/plans/done/) — Shipped milestone plans (M0, M1)
