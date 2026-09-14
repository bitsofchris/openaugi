---
name: query-layer
description: Design — extract the deterministic query engine out of mcp/server.py into a shared query/ package with typed QuerySpecs; MCP, HTTP, and CLI become thin adapters over it; saved queries become data. The read-side substrate for views-as-rendered-queries step 5.
---

# The Query Layer — one engine, thin adapters

**Status: SHIPPED 2026-07-16** (proposed the same day from the
mobile/explorer design session; adopted with the user's answers to the open
questions — saved queries are **markdown+frontmatter**, HTTP `k` caps at
**500**, merge to local main when green. Implementation ledger below;
reference doc: [docs/reference/query-layer.md](../reference/query-layer.md).)

**Deviations from the proposal, decided during implementation:**

- `get_review_state` (omitted from the step-3 list) moved to the engine
  with the other read tools.
- The `review-queue` seed can't be a static value — new `$review-mark`
  token resolves `after_ingested` from the review-pass high-water mark at
  run time (epoch before the first pass = first run is a backfill).
- The Cloudflare auth middleware only guarded `/mcp`; it now guards
  `/api/*` too (auth parity is pinned by a test with a fake verifier).
- Semantic golden cases run on a deterministic hash-derived fake embedder
  (`tests/query_golden_corpus.py`) — no network in the harness.
- `/api/*` exists only under `--transport streamable-http`; stdio opens
  no socket (Claude Code local is unaffected; the bridge runs the HTTP
  daemon).

**Related:**
[views-as-rendered-queries.md](views-as-rendered-queries.md) (ADOPTED) —
its step 5 ("converge lenses onto saved queries") lands on the saved-query
format defined here; its step 3 (bridge renders from daemon queries) is the
first consumer of the HTTP adapter. Also the CQRS framing in the master
plan: this is the **read side** made first-class. Write tools
(`apply_routing`, `write_document`, `tag_block`, `write_recap`) are the
command side and are **out of scope**.

**Supersedes:** nothing — this is a refactor of `mcp/server.py` internals
plus two new adapters. MCP tool signatures and wire format do not change.

---

## Motivation

**The principle.** MCP is a transport, not the engine. Retrieval here is
deterministic — SQLite filters, FTS5, sqlite-vec KNN, link walks. No agent
or token is required to execute a query; agents are just one *caller*.
Today the query *semantics* are trapped inside the MCP adapter, so every
other caller either pays agent ergonomics or re-implements the semantics.

**The evidence — drift is already real, inside this repo:**

1. `mcp/server.py` is ~1,300 lines: half MCP adapter, half query engine.
   Inline in tool bodies live: the four-mode dispatch
   (title/keyword/semantic/browse), `after_ingested` normalization, the
   `has_task` rule ("this is the Dashboard task-shelf query" — a product
   query hard-coded as a parameter convention in a docstring), the
   `layer/bronze` exclusion, `exclude_path_prefix`, reference-document
   collapsing, semantic 3× overfetch + rerank, and the summary/full
   response shaping (`_block_summary` / `_block_full`).
2. `cli/main.py::search` (~line 1085) is a **second implementation**:
   semantic/keyword straight against the store with *none* of the above —
   no tags, no time bounds, no bronze rule, no task filter. Same question,
   different answer depending on the door you knock on.
3. The first non-agent UI consumers exist (augi-mobile bridge `/views`;
   the Datadog-style block-explorer prototype). Over MCP they pay:
   session handshake + JSON-RPC envelope, summaries truncated at 500
   chars (`_block_summary`) forcing a second `get_blocks` round trip for
   full content, and pagination shaped for agent context windows.

The truncation point is the tell: 500-char summaries are **presentation
for agents** (context-window courtesy), not query semantics. A UI wants
full blocks in one call. Presentation belongs in adapters; semantics
belong in one engine.

---

## Target architecture

```
store/sqlite.py            primitives (get_blocks_filtered, search_fts,
                           semantic_search, get_links_from/to, ...)
        ↑
query/                     THE ENGINE — all query semantics, typed in/out,
                           full blocks, deterministic, no MCP/HTTP imports
        ↑                ↑                ↑
  mcp/server.py      HTTP routes        cli/main.py
  (agent ergonomics: (full blocks,      (rewired search/
   truncation,        no handshake,      new query cmd)
   docstrings,        same daemon)
   pagination hints)
```

### 1. `query/` package — the engine

```
src/openaugi/query/
├── spec.py      # QuerySpec — the serializable query object
├── engine.py    # execution: run(spec), related(), traverse(), recent(),
│                #   members(), view(), context()
└── saved.py     # saved-query loading + run_saved(name)
```

**`spec.py`** — a Pydantic model capturing exactly the current `search`
tool surface, no more:

```python
class QuerySpec(BaseModel):
    # mode is derived: query → semantic, keyword → fts, title → title,
    # none of those → browse
    query: str | None = None
    keyword: str | None = None
    title: str | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None
    after_ingested: str | None = None
    kind: str | None = None
    source: str | None = None
    exclude_path_prefix: str | None = None
    has_task: bool | None = None
    k: int = 100
    offset: int = 0
```

Serializable to/from JSON — **the same object is the saved-query file
format** (§2). Round-trips exactly, so a saved query and an ad-hoc query
are indistinguishable to the engine.

**`engine.py`** — functions take a store + spec (or explicit args for the
graph ops), return typed results carrying **full `Block` models** plus a
result envelope (`total`, `has_more`, `next_offset`, `mode`,
`reference_documents`). Everything that is a *rule* moves here from
`mcp/server.py`:

| Rule (today: inline in mcp/server.py) | Engine home |
|---|---|
| mode dispatch title/keyword/semantic/browse | `engine.run(spec)` |
| `normalize_utc_timestamp` bound + `_ingested_too_old` | `engine.run` |
| `_fails_task_filter` (has_task + bronze exclusion) | `engine.run` |
| `_path_excluded` | `engine.run` |
| semantic 3× overfetch + `pipeline.rerank` (deterministic MMR/cosine — no LLM) | `engine.run` / `engine.context` |
| `_group_reference_documents` (source/* collapsing) | `engine.run` (returns groups; adapters decide rendering) |
| `recent`, `get_related`, `traverse`, `get_members`, `get_view` bodies | 1:1 engine functions |
| `_block_summary` 500-char truncation | **does NOT move** — MCP presentation |

The engine imports `store`, `models` (embedding), `pipeline.rerank`,
`config`. It never imports `mcp.*` or any web framework. That constraint
is the test for "is this semantics or presentation."

### 2. Saved queries — specs as data

A saved query is a named, described `QuerySpec` on disk. This is the
lens principle applied to retrieval, and the format
views-as-rendered-queries step 5 converges lenses onto.

- **Format:** JSON file = `{"name", "description", "spec": {…QuerySpec…}}`.
- **Location:** `OpenAugi/AGENT/queries/*.json` in the vault — per the M3
  file-contracts principle (vault filesystem is the API), sitting beside
  `AGENT/lenses/`. Alternative considered: repo/config dir — rejected
  because saved queries are *user* curation, not code.
- **Execution:** `saved.run_saved(store, name)`; relative dates supported
  via small tokens resolved at run time (`"after": "-14d"`, `"after":
  "today"`) so a saved query stays meaningful tomorrow.
- **Seed queries** (ship with the feature, as files):
  - `dashboard-task-shelf` — `{has_task: true, after: "-14d"}`; the
    `search` docstring's hard-coded description then *points at the file*.
  - `review-queue` — `{after_ingested: <high-water>, exclude_path_prefix:
    "OpenAugi/"}` shape used by the review pass.
  - `today` — `{after: "today"}`.
- MCP gets one new read tool `run_query(name)` (+ `list_queries()`); HTTP
  gets `GET /api/queries` and `GET /api/queries/{name}/results`.

### 3. Adapters

**MCP (`mcp/server.py`)** — becomes thin: parse tool args → `QuerySpec` →
engine → agent-shaped presentation. Keeps: 500-char summaries, docstring
guidance, pagination hints, reference-document presentation, `_json`
envelope. **Wire format byte-compatible with today** — enforced by golden
tests (below). Expected size after extraction: roughly half.

**HTTP** — FastMCP is Starlette-based; mount plain JSON routes **on the
same daemon** (`openaugi serve --transport streamable-http`) so there is
still one process and one store handle. No session handshake, full
blocks, no truncation:

```
GET  /api/search?…QuerySpec fields as params…
POST /api/query            # body = QuerySpec JSON (or {"saved": name})
GET  /api/blocks?ids=a,b,c # full content, batch
GET  /api/related/{id}?direction=&kind=
GET  /api/queries          # saved-query list
GET  /api/views            # render list (list_views)
GET  /api/views/{title}    # get_view
```

Read-only routes only (command side stays MCP/file-contract). Bind and
auth posture identical to the MCP endpoint (reuse `auth/` —
`configure_auth`); localhost by default.

**CLI (`cli/main.py`)** — `search` is rewired to build a `QuerySpec` and
call the engine (gaining tags/time/task filters for free and retiring the
duplicate implementation); add `openaugi query <name>` to execute a saved
query. Output formatting stays in the CLI.

### 4. What does not change

No schema change. No new state. No new daemon or process. No MCP tool
renames, signature changes, or response-shape changes. Write tools
untouched. `get_context`'s rerank stays deterministic (it already is —
MMR/cosine in `pipeline/rerank.py`). The explorer/bridge (private repo)
migrates to `/api/*` on its own schedule; MCP keeps working for it
meanwhile.

---

## Implementation plan

Each step ships green on its own; sequence is dependency order.

1. **Golden harness first.** On the fixture DB, capture current outputs of
   every read tool across the mode/filter matrix (title, keyword,
   semantic, browse; each filter; has_task; exclude_path_prefix;
   pagination edges). These are the contract for steps 2–3.
2. **Extract `query/spec.py` + `engine.run`** from the `search` tool body;
   `search` delegates. Golden tests must not change.
3. **Move the remaining read tools** (`get_block(s)`, `get_related`,
   `traverse`, `recent`, `get_members`, `get_view`, `list_views`,
   `get_context`) onto engine functions. Golden tests must not change.
4. **HTTP adapter** mounted on the daemon + contract tests (including the
   invariant: `/api/*` never truncates content).
5. **Saved queries**: `saved.py`, relative-date tokens, seed files,
   `run_query`/`list_queries` MCP tools, `GET /api/queries*`.
6. **CLI rewire** (`search` → engine; new `query` command; delete the
   duplicate store-direct path).
7. **Docs pass** (same PRs as the code they describe — see checklist).

### Docs to update (the checklist for the implementing agent)

- `ARCHITECTURE.md` Module Map: add `query/` with one-line-per-file
  comments in the existing style.
- `ARCHITECTURE.md` "Key Flows → Query (MCP)": retitle **"Query (engine +
  adapters)"**; show the engine box with MCP/HTTP/CLI fan-in; note that
  tool list semantics now live in `query/engine.py`.
- New `docs/reference/query-layer.md` in skill format (name, description,
  when to use, how it works): the QuerySpec fields, saved-query file
  format + location, relative-date tokens, the HTTP route table, and the
  engine/adapter boundary rule ("engine never imports mcp/web; truncation
  is presentation").
- `docs/plans/views-as-rendered-queries.md` implementation ledger: point
  step 5 at this doc's saved-query format.
- `docs/plans/master-plan.md` STATUS block per its own "update every
  session" rule.
- This doc's ledger (below) as steps land.

### Testing requirements

- **Golden regression** (step 1) — MCP wire outputs byte-stable through
  steps 2–3.
- **Engine unit tests** per rule: has_task/bronze, path exclusion,
  ingested bound, mode dispatch precedence, pagination envelope,
  reference-doc grouping, relative-date token resolution (frozen clock).
- **HTTP contract tests**: route ↔ engine parity (same spec → same ids,
  full content), auth posture, read-only (no mutating route exists).
- **CLI smoke**: `openaugi search`/`query` against the fixture DB.

## Implementation ledger

| Step | State | Shipped as |
|---|---|---|
| 1. Golden harness over read tools | ✅ 2026-07-16 | `tests/test_query_golden.py` + `query_golden_corpus.py` + `scripts/gen_query_golden.py` (40 cases, byte-pinned) |
| 2. `query/spec.py` + `engine.run`; `search` delegates | ✅ 2026-07-16 | `src/openaugi/query/{spec,engine}.py`; golden unchanged |
| 3. Remaining read tools on engine | ✅ 2026-07-16 | engine `fetch/related/traverse/recent/members/view/views/context/review_state`; server.py 1309→~1000 lines |
| 4. HTTP adapter on the daemon | ✅ 2026-07-16 | `src/openaugi/http_api.py` + auth-parity fix in `auth/cloudflare.py`; `tests/test_http_api.py` |
| 5. Saved queries (files, tokens, MCP tools, routes) | ✅ 2026-07-16 | `query/saved.py`, seeds in `templates/queries/`, `run_query`/`list_queries`, `/api/queries*` |
| 6. CLI rewire | ✅ 2026-07-16 | `openaugi search` on the engine (+filters), new `openaugi query`; duplicate path deleted |
| 7. Docs pass complete | ✅ 2026-07-16 | ARCHITECTURE map + Query flow, `docs/reference/query-layer.md`, this ledger, views ledger pointer, master-plan STATUS |

## Open questions — all resolved 2026-07-16 with the user

1. **Saved-query file format** → **markdown-with-frontmatter.**
   Obsidian-editable, matches `AGENT/lenses/`, avoids a format migration
   when views-as-rendered-queries step 5 converges lenses onto saved
   queries. `description:` + `query:` mapping in frontmatter, prose body.
2. **`get_context` splitting** → as proposed: `engine.context` owns the
   mechanics (FTS+semantic prongs, bronze weighting, MMR rerank, salience
   gate, expand); the MCP docstring keeps the agent workflow framing.
3. **HTTP pagination** → higher ceiling for HTTP only: `k`/`limit` cap at
   **500** (`http_api.HTTP_MAX_K`); MCP default unchanged.
