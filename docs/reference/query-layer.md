---
name: query-layer
description: The query engine and its three adapters — QuerySpec fields, saved-query files, relative-date tokens, the HTTP /api route table, and the engine/adapter boundary rule.
---

# Query Layer

One engine, thin adapters. Every deterministic read — SQLite filters, FTS5, sqlite-vec KNN, link walks, the get_context pipeline — executes in `src/openaugi/query/engine.py`. MCP, HTTP, and the CLI are presentation over it: same question, same answer, whichever door you knock on.

## When to use

- You're adding or changing **what a query returns** → edit `query/engine.py` (all three surfaces pick it up).
- You're adding or changing **how results look** on one surface (truncation, docstrings, JSON envelopes, terminal output) → edit that adapter (`mcp/server.py`, `http_api.py`, `cli/main.py`).
- You want a recurring query as **data** the user can edit in Obsidian → a saved query file (below).
- A UI needs reads without MCP session overhead → the HTTP routes (below).

**The boundary rule:** the engine never imports `mcp.*` or any web framework (pinned by a test). If code needs those, it's presentation and belongs in an adapter. The canonical example: the 500-char summary truncation is MCP context-window courtesy — it lives in `mcp/server.py` and never applies to HTTP or the engine.

## QuerySpec

`query/spec.py` — the serializable query object. Mode is derived, never stored: `title` → title search, `keyword` → FTS, `query` → semantic, none of those → browse.

| Field | Meaning |
|---|---|
| `query` / `keyword` / `title` | semantic / FTS / title-only text search |
| `tags` | match any of these (user tags + augi_tags) |
| `after` / `before` | `block_time` bounds — the CONTENT date |
| `after_ingested` | ingest-time bound — the review-queue axis (catches same-day and re-ingested blocks `after` misses) |
| `kind` / `source` | block kind (browse defaults to `data_block`) / source |
| `exclude_path_prefix` | drop blocks whose `source_path` starts with this (e.g. `OpenAugi/`) |
| `include_path_prefix` | keep **only** blocks whose `source_path` starts with this (e.g. `OpenAugi/Capture/`). The mirror of the above — pair them across two queries to reach one folder inside an excluded tree |
| `has_task` | only user-marked tasks (open `- [ ]` or `type/task`) |
| `k` / `offset` | page size / browse offset |

The same JSON shape works everywhere: MCP `search` arguments, `POST /api/query` body, saved-query frontmatter, `engine.run` input.

**Where each filter runs.** `kind`, `source`, `after`, `before`, `after_ingested`, and `exclude_path_prefix` are pushed into SQL, so pagination and `total` are correct. `tags` and `has_task` are applied **in Python, after the page is fetched** (`engine.run`, browse branch) — they can only see the rows in the current page, and `total` does not reflect them. A filtered page can therefore come back empty with a large `total`. Callers must paginate; the durable fix is to push both into the SQL `WHERE` clause.

## Saved queries

A saved query is a markdown file at `<vault>/OpenAugi/AGENT/queries/<name>.md` — the lens principle applied to retrieval, and the format views-as-rendered-queries step 5 converges lenses onto:

```markdown
---
description: Open tasks from the last two weeks.
query:
  has_task: true
  after: "-14d"
---
Optional prose about when to use this.
```

Relative-date tokens resolve at **run** time (never stored resolved): `-<N>d` (N days ago), `today`, and `$review-mark` (`after_ingested` only — the review-pass high-water mark; epoch before the first pass, so the first run is a backfill). Seeds shipped by `openaugi init`: `dashboard-task-shelf`, `review-queue`, `today`.

Execute via MCP `run_query(name)` / `list_queries()`, HTTP `GET /api/queries/{name}/results` or `POST /api/query {"saved": name}`, CLI `openaugi query <name>` (bare `openaugi query` lists). The reader is lenient: unparseable files are skipped with a warning, never a crash.

## HTTP routes

Mounted on the daemon (`openaugi serve --transport streamable-http`) — one process, one store handle; stdio mode opens no socket. **Full blocks, never truncated. Read-only.** `k`/`limit` cap at 500 (UIs page; agents keep the MCP default). Auth posture matches `/mcp`: with `--auth cloudflare`, the same Bearer middleware guards `/api/*`; default binding is localhost.

| Route | Returns |
|---|---|
| `GET /api/search?…QuerySpec fields…` | search results (repeatable or comma-separated `tags`) |
| `POST /api/query` | body = QuerySpec JSON, or `{"saved": "<name>"}` |
| `GET /api/blocks?ids=a,b,c` | full blocks, batch, with `routed_to` + `missing` |
| `GET /api/related/{id}?direction=&kind=` | one-hop links |
| `GET /api/queries` | saved-query list (name, description, spec) |
| `GET /api/queries/{name}/results` | execute a saved query |
| `GET /api/views` | the render list (containers with recaps + staleness) |
| `GET /api/views/{title}` | one view: recap + live membership |

## Testing contract

- `tests/test_query_golden.py` pins the MCP wire format **byte-for-byte** over a deterministic corpus (`tests/query_golden_corpus.py`; regenerate only on intentional format changes via `scripts/gen_query_golden.py`).
- `tests/test_query_engine.py` pins engine rules directly (including the no-transport-imports rule).
- `tests/test_http_api.py` pins the HTTP contract: engine parity, no truncation, read-only surface, auth parity.
- `tests/test_saved_queries.py` / `tests/test_cli_query.py` cover saved-query parsing + tokens and the CLI.

## Related

- [docs/plans/query-layer.md](../plans/query-layer.md) — the design record and implementation ledger
- [views-as-rendered-queries](../plans/views-as-rendered-queries.md) — step 5 lands on the saved-query format
- [MCP_SERVER.md](MCP_SERVER.md) — the agent-facing tool reference
