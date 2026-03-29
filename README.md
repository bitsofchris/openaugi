# OpenAugi

This is an evolving personal project, submit issues with ideas or bugs, but expect nothing as I figure out what OpenAugi is :)

Today it's:

Self-hostable personal intelligence engine. One SQLite file. One MCP server. Works with Claude out of the box.

> **Status:** M1.5 complete — local install from source. PyPI package coming in M4.

## What It Does

Point OpenAugi at your Obsidian vault. It builds a knowledge graph (blocks + links) in a single SQLite file, then exposes it to Claude via MCP. Claude can search, traverse, and understand your notes — semantically, by keyword, by tag, by time.

```
Obsidian Vault → split → extract → embed → SQLite → MCP Server → Claude
```

## Install

Not on PyPI yet. Install from source:

```bash
git clone https://github.com/bitsofchris/openaugi.git
cd openaugi
python3 -m venv .venv

# Core + OpenAI embeddings (recommended)
.venv/bin/pip install -e ".[openai]"

# Or core + local embeddings (free, no API key)
.venv/bin/pip install -e ".[local]"

# Or everything
.venv/bin/pip install -e ".[all]"
```

## Quick Start

### 1. Configure (one time)

```bash
openaugi init
```

Interactive setup — choose embedding model (OpenAI, local, or none), set API key if needed, set your vault path.

### 2. Run

```bash
openaugi up
```

That's it. `up` does everything:

1. **Syncs your vault** — incremental ingest (skips unchanged files via content hash, fast if already up-to-date)
2. **Starts file watcher** — watches for `.md` changes, debounces (30s default), re-ingests automatically
3. **Starts MCP server** — stdio transport for Claude Desktop/Code

Embedding is attempted with your configured model. If it fails (no API key, model unavailable), blocks are saved without embeddings and retried on the next watcher cycle.

You can also run the pieces separately if needed:

```bash
openaugi ingest            # one-off ingest without serving
openaugi serve             # MCP server only
openaugi watch             # file watcher only
```

### 3. Register with Claude

**Claude Code:**

```bash
claude mcp add --transport stdio --scope user openaugi -- \
  /path/to/openaugi/.venv/bin/openaugi up
```

**Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "openaugi": {
      "command": "/path/to/openaugi/.venv/bin/openaugi",
      "args": ["up"]
    }
  }
}
```

### Remote access (Claude mobile)

The MCP server is always local. For remote access (e.g., Claude mobile), run the HTTP transport and put a Cloudflare Tunnel in front of it:

```bash
openaugi up --transport http --port 8787
# Then: cloudflared tunnel route to localhost:8787
```

See [docs/REMOTE_ACCESS.md](docs/REMOTE_ACCESS.md) for the full Cloudflare Tunnel + Access setup.

## CLI Reference

| Command | What |
|---------|------|
| `openaugi init` | Configure embedding model, API key, vault path |
| `openaugi ingest` | Run full Layer 0 + Layer 1 pipeline |
| `openaugi up` | MCP server + file watcher (the daily driver) |
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only (incremental ingest on vault changes) |
| `openaugi search "query"` | Search from terminal (semantic or `--keyword`) |
| `openaugi hubs` | Top connected notes by link count |
| `openaugi status` | Block/link/embedding counts |
| `openaugi service install` | Run as macOS launchd service (starts on boot) |

## How It Works

**Data model:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags). Structure lives in the links.

**Processing layers:**
- **Layer 0** (free) — split by headings, extract tags/links, FTS5 index, dedup hash
- **Layer 1** (~$0) — embed with OpenAI or local sentence-transformers, hub scoring via link aggregation
- **Layer 2** (coming) — entity extraction, summaries. LLM required.

**MCP tools:**

| Tool | What |
|------|------|
| `search` | Semantic (sqlite-vec KNN) + keyword (FTS5) + tag/time filters |
| `get_block` | Full block content by ID |
| `get_related` | Follow links from a block |
| `traverse` | Multi-hop graph walk |
| `get_context` | Compound search → expand → structured context |
| `recent` | Recently created blocks |
| `write_document` | Create a markdown note in your vault |
| `write_thread` | Save a conversation thread as a note |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run full CI check (lint + types + tests)
./scripts/check.sh

# Or individually
.venv/bin/python -m pytest tests/ -v
.venv/bin/ruff check src tests
.venv/bin/pyright src
```

## License

MIT
