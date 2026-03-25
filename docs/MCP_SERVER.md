---
name: mcp-server
description: Setup and registration guide for the OpenAugi MCP server — Claude Desktop, Claude Code, tools reference
---

# OpenAugi MCP Server

The OpenAugi MCP server exposes your knowledge graph as structured tools for Claude.
All reads hit SQLite + FAISS. Writes go directly to markdown files in your vault.

## Architecture

```
openaugi serve  (stdio transport)
├── SQLiteStore (read-only, lazy connection)
├── FAISS index (lazy, auto-reloads when DB changes)
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
| `search` | Semantic (FAISS), keyword (FTS5), or browse with filters |
| `get_block` | Full block content + metadata by ID |
| `get_related` | Follow links from/to a block (tags, wikilinks, derivations) |
| `traverse` | Multi-hop graph walk from a starting block |
| `get_context` | Power tool: semantic + keyword → expand via links → structured context |
| `recent` | Recently created blocks, filtered by kind/source/tags |
| `reload_index` | Force-refresh FAISS (auto-reloads on DB change; this is the manual override) |

### Write tools

| Tool | Purpose |
|------|---------|
| `write_document` | Create a markdown note in `OpenAugi/{subfolder}/` in your vault |

**Write scope**: All writes are constrained to `{vault_path}/OpenAugi/{subfolder}/`.
The agent picks the subfolder (`Docs`, `Notes`, `Research`, etc.) but cannot escape
the `OpenAugi/` root. This keeps agent output separate from your own notes.

After writing, run `openaugi ingest` to pick up new notes into the knowledge graph.

## Resources

`vault://note/{title}` — dynamic resource template. Returns all entries for a note
plus inbound/outbound link counts. Shows up in Claude Code's `@` autocomplete:

```
@openaugi:vault://note/My Note Title
```

## Auto-Refresh

The FAISS index auto-reloads when the DB file changes (detected via mtime after
`openaugi ingest`). The index rebuilds lazily on the next semantic search.

Use `reload_index` to force an immediate refresh without waiting for a search.

## Troubleshooting

- **Server not showing in `/mcp`**: Run `claude mcp list` to check registration
- **Import errors**: Verify the venv path in your registration command
- **write_document fails with "No vault path"**: Run `openaugi init` to set a default vault path, or set `OPENAUGI_VAULT_PATH`
- **Stale search results after ingest**: Call `reload_index` or wait for the next search (auto-detects DB change)
