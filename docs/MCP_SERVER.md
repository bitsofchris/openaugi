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
| `get_related` | Follow links from/to a block (tags, wikilinks, derivations) |
| `traverse` | Multi-hop graph walk from a starting block |
| `get_context` | Power tool: semantic + keyword → expand via links → structured context |
| `recent` | Recently created blocks, filtered by kind/source/tags |

### Write tools

| Tool | Purpose |
|------|---------|
| `write_document` | Create a markdown note in `OpenAugi/{subfolder}/` in your vault |

**Write scope**: All writes are constrained to `{vault_path}/OpenAugi/{subfolder}/`.
The agent picks the subfolder (`Docs`, `Notes`, `Research`, etc.) but cannot escape
the `OpenAugi/` root. This keeps agent output separate from your own notes.

After writing, run `openaugi ingest` to pick up new notes into the knowledge graph.

## Upgrading from a previous version

If you have an existing database created before the sqlite-vec migration, run once:

```bash
openaugi migrate-vec --db /path/to/openaugi.db
```

This copies existing embedding blobs from the `blocks` table into the `vec_blocks` virtual table.
No re-embedding is needed — it's a local data migration only.

## Resources

`vault://note/{title}` — dynamic resource template. Returns all entries for a note
plus inbound/outbound link counts. Shows up in Claude Code's `@` autocomplete:

```
@openaugi:vault://note/My Note Title
```

## Troubleshooting

- **Server not showing in `/mcp`**: Run `claude mcp list` to check registration
- **Import errors**: Verify the venv path in your registration command
- **write_document fails with "No vault path"**: Run `openaugi init` to set a default vault path, or set `OPENAUGI_VAULT_PATH`
- **Semantic search returns no results**: Run `openaugi ingest` (or `openaugi migrate-vec` if upgrading from an older DB)
