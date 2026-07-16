---
name: mcp-server
description: Setup and registration guide for the OpenAugi MCP server — Claude Desktop, Claude Code, tools reference
---

# OpenAugi MCP Server

The OpenAugi MCP server exposes your knowledge graph as structured tools for Claude.
All reads hit SQLite (FTS5 + sqlite-vec vector search). Writes go directly to markdown files in your vault.

## Architecture

```
openaugi serve  (stdio transport)
├── SQLiteStore (read-only, lazy connection)
│   ├── FTS5 virtual table  (keyword search)
│   └── vec0 virtual table  (semantic vector search via sqlite-vec)
└── VaultWriter (writes .md files to OpenAugi/ in vault)
```

**No startup needed.** Claude starts the server as a child process on first use (stdio transport).
It stays alive for the session and exits when Claude exits.

## Setup

### 1. Ingest your vault

```bash
openaugi init          # configure vault path + embedding model (one time)
openaugi ingest        # run Layer 0 + Layer 1 pipeline
```

### 2. Register with Claude

#### Claude Code (CLI)

```bash
claude mcp add --transport stdio --scope user openaugi -- \
  /path/to/.venv/bin/openaugi serve
```

If you need to point at a non-default DB:

```bash
claude mcp add --transport stdio --scope user openaugi \
  --env OPENAUGI_DB=/path/to/openaugi.db \
  -- /path/to/.venv/bin/openaugi serve
```

Verify: run `/mcp` in Claude Code to check server status.

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "openaugi": {
      "command": "/path/to/.venv/bin/openaugi",
      "args": ["serve"],
      "env": {
        "OPENAUGI_DB": "/Users/you/.openaugi/openaugi.db"
      }
    }
  }
}
```

Restart Claude Desktop after editing.

### 3. Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAUGI_DB` | Path to SQLite database | `~/.openaugi/openaugi.db` |
| `OPENAUGI_VAULT_PATH` | Path to Obsidian vault (required for write tools) | From `config.toml [vault] default_path` |

Both are optional if you've run `openaugi init` — the config file is the default.

## Tools

### Read tools

| Tool | Purpose |
|------|---------|
| `search` | Semantic (sqlite-vec KNN), keyword (FTS5), or browse with filters; paginated for date-range queries |
| `get_block` | Full block content + metadata by ID |
| `get_blocks` | Batch fetch up to 50 blocks by ID — prefer over calling `get_block` in a loop |
| `get_related` | Follow links from/to a block (tags, wikilinks, derivations) |
| `get_members` | A container's members under the unified membership rule (containment ∪ routing) — each member marked `contained`/`routed`/`both`, newest first. The query views render. |
| `get_view` | Render a container's view from the DB: live membership log + cached recap with visible staleness (`stale: true` when membership changed since the recap was written). What rendered surfaces consume instead of `View - *.md` files. |
| `list_views` | The render list: every container with a cached recap row, newest first, with per-view staleness. A recap row IS the "this container has a view" bit — hand-curated containers (recap off) never appear. |
| `traverse` | Multi-hop graph walk from a starting block |
| `get_context` | Power tool: semantic + keyword → deduplicate → MMR re-rank → expand via links; optional `purpose` applies a `[salience]` min-score gate for proactive surfaces |
| `recent` | Recently ingested blocks, filtered by kind/source/tags |

### `search` — browse mode and date-range queries

When no `query`, `keyword`, or `title` is given, `search` runs in **browse mode** — returning
all blocks that match the provided filters. Date filtering is pushed to SQL, so results are
accurate even across large vaults.

```
search(after="2026-04-05", before="2026-04-12")
→ { results: [...], count: 100, total: 247, has_more: true, next_offset: 100 }
```

- `total` — full result set size before pagination; use to plan how many calls are needed
- `has_more` + `next_offset` — call again with `offset=next_offset` to get the next page
- Default `k` is 100; for a typical week (~200 blocks) you'll need at most 2 calls
- Tags filtering still happens in Python — `total` reflects pre-tag counts
- `exclude_path_prefix="OpenAugi/"` — drop blocks by `source_path` prefix at the SQL
  level (keeps derived artifacts out of a review queue); works in every mode
- `after_ingested=<iso timestamp>` — filter on when the block entered the DB rather
  than its content date; works in every mode. `after`/`before` compare `block_time`,
  which is often date-only (`"2026-07-12"` sorts before any same-day timestamp) and is
  kept when an edited block re-ingests — so ingest-order queues (the review pass,
  "what's new since X") must use `after_ingested`, not `after`
- Block summaries include `source_path`, so derived-vs-capture is explicit
- **Reference grouping**: blocks carrying a `source/*` tag (Readwise, Snipd — set via
  `[vault.source_rules]`) are collapsed into `reference_documents`, one entry per source
  document (`document_id`, `block_count`, time range). Route the document, not its blocks.

Use this for workflows like weekly reflection where you want **every block in a time window**,
not just top-k by relevance.

### `get_context` — retrieval pipeline

`get_context` is the primary tool for Claude to answer questions against your knowledge
base. It runs a multi-stage pipeline before returning results:

```
1. FTS keyword search  →  k × overfetch_ratio candidates
2. Semantic KNN search →  k × overfetch_ratio candidates
3. Merge by block_id (deduplicates across prongs)
4. Group semantically similar chunks (greedy agglomerative, cosine distance)
5. Pick one representative per group (centroid or highest score)
6. MMR re-rank representatives for diversity
7. Fetch full content for final k blocks
8. Expand via links (unchanged)
```

Steps 4–6 eliminate near-duplicate chunks — the same idea phrased multiple times — before
the results reach Claude, saving context window and improving reasoning quality.

#### Tuning via `config.toml`

Add a `[retrieval]` section to `~/.openaugi/config.toml` (or `./openaugi.toml`).
All keys are optional; defaults are shown:

```toml
[retrieval]
overfetch_ratio = 3     # fetch k*3 candidates before dedup (more = better recall, slower)
group_threshold = 0.15  # cosine distance below which two chunks are considered duplicates
mmr_lambda = 0.5        # 1.0 = pure relevance, 0.0 = pure diversity
representative = "centroid"  # "centroid" | "score"
```

**`group_threshold`** — controls how aggressively near-duplicates are collapsed:

| Value | Effect |
|-------|--------|
| `0.10` | Conservative — only near-identical chunks merge |
| `0.15` | Default — balanced deduplication |
| `0.20` | Aggressive — topically similar chunks merge |

Raise this if you're still seeing redundant results. Lower it if unrelated chunks
are being collapsed.

**`mmr_lambda`** — balances relevance vs. diversity in the final ranking:

| Value | Effect |
|-------|--------|
| `0.3` | Diversity-focused — maximises topical breadth |
| `0.5` | Default — balanced |
| `0.7` | Relevance-focused — stays close to query |

Lower this when you want Claude to survey a broad range of your notes. Raise it
when you want tight focus on the most relevant content.

**`representative`** — which chunk survives when a group is collapsed:

| Value | Effect |
|-------|--------|
| `"centroid"` | Default — keeps the chunk whose embedding is closest to the group mean (most "typical") |
| `"score"` | Keeps the chunk with the highest original retrieval score (preserves FTS/KNN ranking signal) |

**`overfetch_ratio`** — multiplier on `k` for the initial candidate pool. Higher values
give the deduplication step more to work with, at the cost of a slightly larger DB query.
Rarely needs changing.

#### Salience gating (`purpose` parameter)

Proactive surfaces — things that speak up *unprompted*, like mobile resurfacing —
need a precision bias that a research query doesn't: a wrong resurface costs trust,
a missing one costs nothing. `get_context(purpose="resurface")` applies a per-purpose
minimum score and drops results below it. Scores are computed exactly as before —
the gate only filters; it never changes ranking or scoring math.

```toml
[salience]
resurface = 0.06  # in-app resurfacing (mobile bridge). Calibrated live 2026-07-07:
                  # mundane captures ~0, weak associations 0.01-0.05, real hits 0.06-0.14
push = 0.15       # reserved — push notifications need a stricter gate (no consumer yet)
```

- Omitting `purpose` (the default) applies no gate — regular research calls are
  unaffected. The parameter is additive; existing callers need no changes.
- An unknown purpose (no matching `[salience]` key) applies no gate.
- When `purpose` is passed, the response carries a `salience: {purpose, min_score}`
  field so callers can see which gate ran.

#### The bronze layer (`#layer/bronze` down-weighting)

Mobile curation (2026-07-14, `private-augi-mobile` docs/systems/curation.md) lets the
user demote a block to scaffolding by tagging it `#layer/bronze` — a tag, not a move,
so ingest keeps the block untouched (raw data is truth). `get_context` honors the
demotion the same way the `source/*` firewall handles third-party material:

```toml
[layers]
bronze_weight = 0.5  # retrieval-score multiplier for #layer/bronze blocks; 1.0 disables
```

- Candidate scores for bronze blocks are multiplied by `bronze_weight` *before*
  reranking, so full-weight thinking outranks demoted scaffolding but bronze is
  still reachable by a direct query.
- When `purpose` is set (proactive surfaces: resurfacing, push), bronze blocks are
  excluded outright regardless of score — the user already demoted them; they never
  resurface unprompted.
- Bronze is by tag alone. Capture daily notes ingest per anchored entry (the
  splitter's anchor rule, see [splitter.md](../reference/splitter.md) and
  [docs/plans/anchor-segmentation.md](../plans/anchor-segmentation.md)), so the
  tag lands exactly on the demoted entry's block — a demoted entry never dims
  the rest of its day.
- Promote = the user removes the tag; the next ingest of the file restores full weight.
- Review-pass policy (route bronze, never nominate or feature it) lives in the
  agent prompt: `OpenAugi/AGENT/review-pass.md` / `src/openaugi/templates/review-pass.md`.

### Write tools

| Tool | Purpose |
|------|---------|
| `write_document` | Save anything to `OpenAugi/{subfolder}/` — triggered by "save this", "write this to augi", or explicit save requests. Agent infers subfolder from content (`Notes`, `Docs`, `Research`, `Views`). Supports `overwrite=true` for regenerable derived views. |
| `tag_block` | Stamp AI-classified `augi_tags` onto a block's metadata. Used by the augi-agent for area/type/status classification. |
| `write_recap` | Cache a container's recap (the LLM synthesis half of its rendered view) as a DB row, stamped with the membership hash it was generated against. Written by the review pass (recaps refresh on pass only); read back through `get_view`. |

`write_document` takes a `description` field — a one-liner that goes in frontmatter for scanning.

**Write scope**: All writes are constrained to `{vault_path}/OpenAugi/`.
The agent picks the subfolder but cannot escape the `OpenAugi/` root.
This keeps agent output separate from your own notes.

After writing, run `openaugi ingest` to pick up new notes into the knowledge graph.

### Review pass tools

| Tool | Purpose |
|------|---------|
| `apply_routing` | THE route write tool: a list of `{block_id, add, remove, augi_tags}` decisions in one call (`containers` is a legacy alias for `add`). Adds and removes `routed_to` links — a wrong route is corrected with one decision carrying both `add` and `remove`. Idempotent both ways; per-decision errors don't block the rest. Home by construction: adding a route to the block's own source note is a counted no-op (`already_home`); removing containment is an error. Current membership is readable via `get_members`, `get_block`/`get_blocks` (`routed_to` field), or `get_related(kind="routed_to")`. |
| `get_review_state` | Read the review-pass high-water mark (`last_run` timestamp + `last_summary`). Called at the start of a pass to scope new blocks. |
| `mark_review_complete` | Advance the high-water mark to now with a one-line run summary. Called once at the end of a successful pass. |

> **Removed 2026-07-06:** the Streams subsystem (`make_stream`, `update_stream`,
> `get_stream_context`, `list_streams`) and the chat-capture tools (`write_snip`,
> `write_thread`) were superseded by the review-pass derived views
> (see [docs/plans/review-pass-v1.md](../plans/review-pass-v1.md)). Use `write_document`
> for all vault writes.

## Resources

`vault://note/{title}` — dynamic resource template. Returns all entries for a note
plus inbound/outbound link counts. Shows up in Claude Code's `@` autocomplete:

```
@openaugi:vault://note/My Note Title
```

## Remote Access (Advanced)

The MCP server supports HTTP transport with optional OAuth authentication for remote access (e.g. from Claude mobile). This is an advanced, optional feature — most users should use the default stdio transport with Claude Desktop/Code.

```bash
# HTTP transport (local network only, no auth)
openaugi serve --transport http

# HTTP transport with Cloudflare Access OAuth
openaugi serve --transport http --auth cloudflare
```

Requires `pip install openaugi[remote]` for auth dependencies.

Setup involves a Cloudflare account, domain, tunnel, and Access configuration. See the local docs for detailed instructions.

## Troubleshooting

- **Server not showing in `/mcp`**: Run `claude mcp list` to check registration
- **Import errors**: Verify the venv path in your registration command
- **write_document fails with "No vault path"**: Run `openaugi init` to set a default vault path, or set `OPENAUGI_VAULT_PATH`
- **Semantic search returns no results**: Run `openaugi ingest` to embed blocks into vec_blocks
