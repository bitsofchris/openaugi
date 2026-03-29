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
| `search` | Semantic (sqlite-vec KNN), keyword (FTS5), or browse with filters |
| `get_block` | Full block content + metadata by ID |
| `get_blocks` | Batch fetch up to 50 blocks by ID — prefer over calling `get_block` in a loop |
| `get_related` | Follow links from/to a block (tags, wikilinks, derivations) |
| `traverse` | Multi-hop graph walk from a starting block |
| `get_context` | Power tool: semantic + keyword → deduplicate → MMR re-rank → expand via links |
| `recent` | Recently created blocks, filtered by kind/source/tags |

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

### Write tools

| Tool | Purpose |
|------|---------|
| `write_document` | Save anything to `OpenAugi/{subfolder}/` — triggered by "save this", "write this to augi", or explicit save requests. Agent infers subfolder from content (`Notes`, `Docs`, `Research`). |
| `write_thread` | Save a distilled session note to `OpenAugi/Threads/YYYY-MM-DD - {topic}.md`. Triggered by "save this thread", "log this session". Not a transcript — synthesize intent, decisions, and what was learned. |

Both tools take a `description` field — a one-liner that goes in frontmatter for scanning.

**Write scope**: All writes are constrained to `{vault_path}/OpenAugi/`.
The agent picks the subfolder but cannot escape the `OpenAugi/` root.
This keeps agent output separate from your own notes.

After writing, run `openaugi ingest` to pick up new notes into the knowledge graph.

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
