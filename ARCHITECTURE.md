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

Sidecars: `meta` (key-value, review-pass high-water mark), `recaps`
(per-container view-synthesis cache, staleness-hashed — see
[docs/plans/views-as-rendered-queries.md](docs/plans/views-as-rendered-queries.md)),
`vec_blocks`/`blocks_fts` (derived indexes).

**Block kinds:** data_block, context_block:document, context_block:tag
**Link kinds:** contains, groups, links_to

Everything is a block. Structure lives in the links, not in the schema.

See [docs/reference/data-model.md](docs/reference/data-model.md) for the full data model philosophy.

## Processing Layers

| Layer | Plane | Cost | What | Requires |
|-------|-------|------|------|----------|
| **Layer 0** | Data | FREE | Split, tag/link extract, FTS, dedup hash, zzz dispatch | Python + SQLite |
| **Layer 1** | Data | ~$0 | Embed (local default), hub scoring (SQL) | sentence-transformers (local) |
| **Agent** | Agent | per-task | Execute zzz-dispatched tasks in tmux Claude sessions | Claude Code CLI + tmux |

## Two Planes

The codebase separates two fundamentally different kinds of work, run by `openaugi up` (the service) and `openaugi serve` (the MCP interface):

**Data plane (`pipeline/`)** — passive transforms on blocks. Ingest, embed, watch for file changes, dispatch zzz instructions as task files, re-rank search results. All in-process Python, no external processes.

**Agent plane (`agents/`)** — task dispatch watches `OpenAugi/Tasks/` for pending task files and launches Claude Code sessions in tmux. The agent's behavior is governed by the vault's `OpenAugi/AGENT/` files; `templates/` holds their shipped twins (`kind: engine`), copied on init.

The two planes share the store and the block/link data model. The bridge between them is `pipeline/dispatch.py`: when ingest finds blocks with `zzz:` instructions, it writes task files that the agent plane picks up. The MCP server (`mcp/`) sits alongside both as the read/write API surface that Claude calls.

## Module Map

```
src/openaugi/
├── model/
│   ├── block.py          # Block Pydantic model
│   ├── link.py           # Link Pydantic model
│   └── protocols.py      # EmbeddingModel, LLMModel protocols
├── adapters/
│   ├── splitter.py       # Deterministic block splitter — shared primitive (see docs/reference/splitter.md)
│   └── vault.py          # Obsidian vault → blocks + links (wraps splitter)
├── pipeline/              # Data plane — transforms on blocks + zzz dispatch
│   ├── runner.py          # Layer 0 orchestrator (incremental ingestion)
│   ├── embed.py           # Layer 1 embedding step → vec_blocks (sqlite-vec)
│   ├── dispatch.py        # Post-ingest: zzz instructions → task files in OpenAugi/Tasks/
│   ├── augi_log.py        # The shared per-day Augi Log: sections, eligibility gate, heartbeat
│   ├── writeback.py       # Shared write-back: the feedback log path/read/append, and the box + `aaa:` grammar
│   ├── route.py           # Post-ingest: one routing row per new human block in the Augi Log (docs/reference/augi-log-routing.md)
│   ├── routing_janitor.py # Applies a log's routing rows once its master box is ticked; undo; feedback
│   ├── rerank.py          # Dedup + MMR re-ranking for get_context
│   ├── context_pack.py    # OpenAugi/context-pack.json — mobile capture-assist sidecar + lens list (docs/reference/lenses.md)
│   ├── board_janitor.py   # Currency board write-back — checkboxes → feedback log → projected .board-state.json; also the weekly reflection's `do` offers (docs/reference/currency-board.md)
│   ├── vault_render.py    # Vault rendering — write blocks as .md to OpenAugi/Compiled/ (future)
│   ├── schedule.py        # Lens triggers → due lenses → task files (docs/reference/lenses.md)
│   └── watcher.py         # File watcher — debounced incremental ingest + zzz dispatch + lens tick
├── render/                # M6 — static HTML surfaces from the DB (no server)
│   └── lifestream.py      # Merged chronological stream + heat strip → OpenAugi/render/
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
├── query/                 # THE query engine — read semantics shared by every adapter (docs/reference/query-layer.md)
│   ├── spec.py            # QuerySpec — serializable query object (also the saved-query file format)
│   ├── engine.py          # run/fetch/related/traverse/recent/members/view/context — full blocks, no transport imports
│   └── saved.py           # Saved queries: OpenAugi/AGENT/queries/*.md + relative-date tokens
├── mcp/
│   ├── server.py          # MCP adapter — agent presentation (summaries, docstrings) over query/; write + review tools
│   └── doc_writer.py      # VaultWriter — writes .md to OpenAugi/ in vault
├── http_api.py            # HTTP adapter — /api/* JSON routes on the daemon (full blocks, read-only)
├── cli/
│   └── main.py            # typer CLI (up, ingest, serve, watch, search, query, hubs, status, service)
└── config.py              # TOML config loader + .env loader + vault path resolution
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

### Query (engine + adapters)

All read semantics live in ONE place — `query/engine.py` — and three thin
adapters present it (docs/reference/query-layer.md):

```
                       query/engine.py  (full blocks, deterministic,
                        never imports mcp/web — semantics only)
                      ↗        ↑         ↖
   mcp/server.py         http_api.py        cli/main.py
   (agent shape:         (/api/* JSON:      (openaugi search /
    500-char summaries,   full blocks,       openaugi query,
    docstrings,           no handshake,      terminal output)
    pagination hints)     k ≤ 500)

  engine functions:
  → run(QuerySpec): title/keyword/semantic/browse dispatch + filters
    (after_ingested bound, has_task filter, path exclusion,
     reference-document grouping, pagination envelope)
  → fetch / related / traverse / recent / members / view / views
  → context: FTS + semantic (3× overfetch)
             → MMR re-rank (rerank.py) → salience gate → expand via links
  → saved queries: OpenAugi/AGENT/queries/*.md (QuerySpec in frontmatter,
    tokens "-14d"/"today"/"$review-mark" resolved at run time)

  write side (MCP only — the command half of the CQRS split):
  → write_document / write_recap / tag_block / apply_routing
  → write_record / list_records / update_record (generic collection store for
    agent workflow state — docs/reference/records.md; no schema, no policy)
  → get_review_state / mark_review_complete: review-pass high-water mark
```

The Streams subsystem (StreamManager + `make_stream`/`update_stream`/
`get_stream_context`/`list_streams`) and the chat-capture tools
(`write_snip`/`write_thread`) were removed on 2026-07-06 — superseded by the
review-pass derived views (see [docs/plans/review-pass-v1.md](docs/plans/review-pass-v1.md)).

### ZZZ Dispatch (zzz → task file → agent)

```
you write `zzz: <instruction>` in a vault note
  → file watcher detects change (30s debounce)
  → ingest: parse, split, extract blocks + tags + links
  → pipeline/dispatch.py: blocks with zzz_instructions
    → queue them in the `zzz_queue` ledger (records table)
    → supersede any instruction this cycle's edit replaced
    → carry a dispatched row forward when only the prose around it changed
    → drain: blocks unchanged for 120s become tasks
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

**Dispatch is queued, not immediate.** A block's identity is the hash of its
raw text *including* the `zzz` line, so finishing a half-written instruction
deletes one block and inserts another — and a hook that fires on "new block
with a zzz" fires twice for one instruction. (It did, on 2026-09-01: a task
launched on `read this voice` at 20:41 and another on the finished sentence at
20:52.) Three mechanisms, covering different gaps:

- **Settle window** (`tasks.zzz_settle_seconds`, default 120) — a zzz block is
  queued and only becomes a task once it has survived unchanged. Drafts
  abandoned inside the window never become tasks. The watcher also drains on a
  timer, so an instruction written just before the vault goes quiet still fires.
- **Supersession** — past the window the draft has already launched, so waiting
  cannot help. `run_layer0` reports the entries it deleted, which is the one
  place a block's predecessor is still visible; a document that drops a known
  zzz block and adds a new one in the same cycle has *edited* an instruction.
  The old task is marked `status: superseded` and its tmux session killed, so
  the wording you finished is the only one still running. The transcript stays
  on disk.
- **Carry-forward** — supersession alone still re-dispatches, because the
  successor is a new block. But editing the *prose* around a `zzz` line
  rewrites the block with the instruction untouched. When the successor's zzz
  text is byte-identical and its predecessor already dispatched, it inherits
  that ledger row and task file: no second task, no retirement notice, no
  session killed. That is what makes dispatch idempotent per source block —
  one instruction, one task, however many times the paragraph is edited. (On
  2026-09-09 one research `zzz` dispatched three times over five hours this
  way.) A *changed* instruction is a real edit and still supersedes.

The ledger is the `zzz_queue` collection in the `records` table
([docs/reference/records.md](docs/reference/records.md)) — droppable workflow
state, pruned 30 days after a row settles. It also makes dispatch idempotent
across restarts: a block id dispatched once never dispatches again.

Per-block `zzz:` lines are extracted by the vault adapter into
`metadata["zzz_instructions"]` (a list, one item per line; stripped from
the clean content). Blocks are split on `###` headers and `qqq` markers
(case-insensitive) — see [docs/plans/zzz-instructions.md](docs/plans/zzz-instructions.md).

The agent's behavior is governed by the vault's `OpenAugi/AGENT/` files (the
source of truth), not the Python code. `src/openaugi/templates/` holds the
shipped twin of every `kind: engine` file, written by
`scripts/sync_templates.py` and copied on `openaugi init`; `kind: personal`
files never ship. The kind, the personal-region markers and the template walk
live in `src/openaugi/agent_files.py`; the rule is in AGENTS.md "Engine vs
personal".

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
- **Proactive echo** *(PAUSED 2026-09-04 — see [docs/plans/augi-log-paused.md](docs/plans/augi-log-paused.md))*: the one pass that runs unasked — new daily-note blocks are matched against the user's own prior writing and, when it would genuinely help, appended to a dated Augi Log with promote/feedback checkboxes. Post-ingest hook in the watcher, sibling of zzz dispatch. See [docs/reference/proactive-echo.md](docs/reference/proactive-echo.md).
- **`get_context` dedup + MMR**: Over-fetches 3× candidates, collapses near-duplicates via cosine grouping, re-ranks for diversity before returning. See [docs/reference/MCP_SERVER.md](docs/reference/MCP_SERVER.md) for tuning.
- **Augi Log routing** *(PAUSED 2026-09-04 — see [docs/plans/augi-log-paused.md](docs/plans/augi-log-paused.md))*: the capture/routing surface. Every new human daily-note block gets one row proposing where it lives (extend / link / file under / new note / memory / hold); the user answers with checkboxes and `aaa:` lines, ticks the day's master box, and `routing_janitor.py` applies the log — DB links, or an append-only newest-first insert into the target note for extend. Every answer is logged to teach later proposals. See [docs/reference/augi-log-routing.md](docs/reference/augi-log-routing.md).

- **Currency board**: the one surface that promises to be current — everything else stays append-only truth. A scheduled daily board (left off → next moves → ≤3 judgment items → drift), answered with done/not-doing/someday checkboxes that `board_janitor.py` appends to the feedback log and projects into the state the next board must honor (the board build only ever reads it). Its last build step is the **chat harvest**: `scripts/session_harvest.py` slices yesterday's human turns out of the local Claude/Codex transcripts and the `chat-harvest` lens offers at most one note worth keeping, as a two-box proposal carrying the drafted note in full. See [docs/reference/currency-board.md](docs/reference/currency-board.md).

## Running

### Quick start (one command)

```bash
openaugi init          # one-time: configure embedding model, API key, vault path
openaugi up            # daily: sync vault + file watcher + zzz dispatch
```

`openaugi up` is the background service; `openaugi serve` is what MCP clients talk to:

1. **Incremental ingest** — syncs vault to SQLite (skips unchanged files via content hash)
2. **File watcher** — daemon thread watches for `.md` changes, debounces (default 30s), re-ingests
3. **MCP server** — foreground, stdio or HTTP transport

Embedding is attempted with the user's configured model. If it fails, blocks are saved without embeddings and retried on the next watcher cycle. SQLite WAL mode handles concurrent reads (MCP) and writes (watcher) without locking.

### Daily use — one command

```
openaugi up      ← ingest + watcher + zzz dispatch + task agent   (one per vault, locked)
openaugi serve   ← MCP tools for one client                       (one per client)
```

### All commands

| Command | What |
|---------|------|
| `openaugi up` | Ingest + watcher + zzz dispatch + task agent. **One per vault** — takes a lock. Does not serve MCP |
| `openaugi up --no-agent` | Same but without task dispatch (no tmux agent sessions) |
| `openaugi task-dispatch` | Watch `OpenAugi/Tasks/` and launch pending tasks in tmux (standalone) |
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only (incremental ingest on vault changes) |
| `openaugi re-embed` | Reset + re-embed all data blocks with current model (use after model switch) |
| `openaugi cluster` | Run clustering DAG → write `context_block:cluster` nodes + a `cluster_run` snapshot to DB |
| `openaugi cluster --dry-run` | Compute clusters + print stats, no DB writes (use for param tuning) |
| `openaugi cluster-weather` | Growth/death report over cluster snapshots (feeds the cluster-weather lens; `--json`) |
| `openaugi lineage "<topic>"` | Time-ordered semantic evidence for one idea (feeds the idea-lineage lens; `--write` emits the mobile timeline sidecar) |
| `openaugi backfill-source-tags` | Apply `[vault.source_rules]` attribution to existing DB rows (once after adding rules) |

### Transports

Two transport modes — see [docs/REMOTE_ACCESS.md](docs/REMOTE_ACCESS.md) for full setup.

| Transport | Command | Use Case |
|-----------|---------|----------|
| stdio (default) | `openaugi serve` | Claude Desktop/Code on same machine |
| streamable-http | `openaugi serve --transport streamable-http` | Remote clients, Claude mobile via Cloudflare Tunnel |

Service management (macOS): `openaugi service install/uninstall/status` — launchd plist, starts on boot.

## Related repos

- **private-augi-mobile** (`~/repos/private-augi-mobile`) — the mobile capture client (Expo/RN, iOS-first). Thin client over this repo's data model: its mock contract server (`POST /capture`, `GET /context-pack`) pins the API a future FastAPI server in this repo will implement. Mobile captures become blocks here; the container registry + routing built here feed its tag/link-assist suggestions (its milestone M4).
- **openaugi-obsidian-plugin** — Obsidian-side capture/context tooling.
- **openaugi-private** — parked; not a source of decisions.

## Docs & plans

Docs live in three tiers: **`docs/reference/`** — durable "how it works"
manuals (the reference docs linked below); **`docs/plans/`** — design
records and active plans (→ `docs/plans/done/` when shipped);
**`docs/scratch/`** — gitignored throwaway. Reference docs use the skill
format (`name:`/`description:` frontmatter) so they're scannable.

- [docs/reference/core-principles.md](docs/reference/core-principles.md) — **The skeleton (read first when designing):** capture grammar, truth/index/cache/render layer model, trust model, promotion — the four invariants everything else hangs on.
- [docs/reference/agentic-kb-field-guide.md](docs/reference/agentic-kb-field-guide.md) — The portable ruleset: what building this hardened or simplified from the "agent + janitor + flat folder" starting advice; transplantable to any agentic knowledge base.
- [docs/reference/reading-queue.md](docs/reference/reading-queue.md) — Reading queue: notes flagged `reading_queue: true` pushed to Readwise Reader under a daily cap, highlights harvested back onto the note that produced them. Manual commands, nothing scheduled.
- [docs/reference/pings.md](docs/reference/pings.md) — Pings: phone check-ins appended to the daily note as `- [HH:MM] <kind>: key=value …` lines, `scripts/ping_stats.py` cross-tabs them (kinds, target key and vocabulary come from the vault lens that invokes it, never from the code)
- [docs/reference/privacy-guard.md](docs/reference/privacy-guard.md) — Privacy guard: the pre-commit hook that refuses private vocabulary (word list in the vault, never the repo) and notebooks with outputs
- [docs/reference/user-guide.md](docs/reference/user-guide.md) — Day-to-day manual: entry points, the loop, trust rules, triggering a pass, lens system in brief. Chronological build history stays in this file's STATUS header, not there.
- [docs/reference/augi-log-routing.md](docs/reference/augi-log-routing.md) — **Routing rows in the Augi Log (PAUSED 2026-09-04; replaced the review pass for daily capture):** verbs, the master box, what each verb writes, undo, the feedback record. Design record: [docs/plans/augi-log-routing.md](docs/plans/augi-log-routing.md)
- [docs/reference/review-pass.md](docs/reference/review-pass.md) — **The write-back loop (superseded for daily capture by Augi Log routing, 2026-09-03; `apply_routing` and the registry rule still apply):** augi_tags, capture grammar (qqq/zzz/aaa), running a pass. Per-container view FILES were retired 2026-08-17 — recaps are `write_recap` rows. Design record: [docs/plans/review-pass-v1.md](docs/plans/review-pass-v1.md)
- [docs/reference/records.md](docs/reference/records.md) — **The collection store + the test for a new MCP tool:** three generic tools for agent workflow state. Schemas live in the caller's prompt, policy in the caller's config, only mechanism in a tool. Read before adding any tool.
- [docs/reference/recap-spec.md](docs/reference/recap-spec.md) — **What a recap contains:** only what scrolling can't give you — cross-month patterns, contradictions, unanswered questions, what's gone quiet. Not what-moved, not member lists.
- [docs/plans/m2-feature-roadmap.md](docs/plans/m2-feature-roadmap.md) — Post-launch roadmap (Ship → Show → Adapt → Deepen → Differentiate → Lenses → Expand)
- [docs/plans/phase3-adapters.md](docs/plans/phase3-adapters.md) — Phase 3: multi-source ingest adapters (ChatGPT, Readwise, Research, LlamaIndex bridge)
- [docs/plans/done/heartbeat.md](docs/plans/done/heartbeat.md) — (shipped, then replaced by zzz dispatch) Heartbeat design history
- [docs/plans/capture-tag-stream-loop.md](docs/plans/capture-tag-stream-loop.md) — Phase 4: capture → tag → stream incremental pipeline
- [docs/plans/from-capture-to-jarvis.md](docs/plans/from-capture-to-jarvis.md) — Longer-horizon vision (layers 1–4)
- [docs/plans/future-work.md](docs/plans/future-work.md) — Deferred features
- [docs/reference/clustering.md](docs/reference/clustering.md) — Clustering feature: config format, data model, cluster weather (snapshots + diffs), SQL queries, param tuning guide
- [docs/plans/hierarchical-embeddings.md](docs/plans/hierarchical-embeddings.md) — Design rationale: two-pass strategy, matryoshka truncation, bridge detection
- [docs/plans/done/](docs/plans/done/) — Shipped milestone plans (M0, M1)
