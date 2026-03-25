# OpenAugi

This is an evolving personal project, submit issues with ideas or bugs, but expect nothing as I figure out what OpenAugi is :)

Today it's:

Self-hostable personal intelligence engine. One SQLite file. One MCP server. Works with Claude out of the box.

> **Status:** M0 complete — local install from source. PyPI package coming in M4.

## What It Does

Point OpenAugi at your Obsidian vault. It builds a knowledge graph (blocks + links) in a single SQLite file, then exposes it to Claude via MCP. Claude can search, traverse, and understand your notes — semantically, by keyword, by tag, by time.

```
Obsidian Vault → split → extract → embed → SQLite + FAISS → MCP Server → Claude
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

```bash
# First-time setup — choose model, set API key, set vault path
.venv/bin/openaugi init

# Ingest your vault (uses default path from init)
.venv/bin/openaugi ingest

# Search from terminal
.venv/bin/openaugi search "what have I been thinking about?" --keyword
.venv/bin/openaugi hubs        # top connected notes
.venv/bin/openaugi status      # block/link counts

# Start MCP server for Claude
.venv/bin/openaugi serve
```

### Register with Claude

```bash
# Claude Code (after activating venv)
claude mcp add --transport stdio --scope user openaugi -- \
  /path/to/openaugi/.venv/bin/openaugi serve

# Claude Desktop: add to ~/Library/Application Support/Claude/claude_desktop_config.json
```

## How It Works

**Data model:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags). Structure lives in the links.

**Processing layers:**
- **Layer 0** (free) — split by headings, extract tags/links, FTS5 index, dedup hash
- **Layer 1** (~$0) — embed with OpenAI or local sentence-transformers, hub scoring via link aggregation
- **Layer 2** (coming) — entity extraction, summaries. LLM required.

**MCP tools (6):**

| Tool | What |
|------|------|
| `search` | Semantic (FAISS) + keyword (FTS5) + tag/time filters |
| `get_block` | Full block content by ID |
| `get_related` | Follow links from a block |
| `traverse` | Multi-hop graph walk |
| `get_context` | Compound search → expand → structured context |
| `recent` | Recently created blocks |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

## Development

```bash
# Dev setup
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
