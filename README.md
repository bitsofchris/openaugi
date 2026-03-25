# OpenAugi

Self-hostable personal intelligence engine. One `pip install`. One SQLite file. One MCP server.

## What It Does

Point OpenAugi at your Obsidian vault. It builds a knowledge graph (blocks + links) in a single SQLite file, then exposes it to Claude via MCP. Claude can search, traverse, and understand your notes — semantically, by keyword, by tag, by time.

```
Obsidian Vault → split → extract → embed → SQLite + FAISS → MCP Server → Claude
```

## Quick Start

```bash
# Install
pip install openaugi[local]    # includes sentence-transformers + FAISS

# Ingest your vault
openaugi ingest --path ~/your-vault

# Start MCP server (Claude discovers tools automatically)
openaugi serve

# Or search from terminal
openaugi search "what have I been thinking about?"
openaugi hubs        # top connected notes
openaugi status      # block/link counts
```

### Register with Claude

```bash
# Claude Code
claude mcp add --transport stdio --scope user openaugi -- openaugi serve

# Claude Desktop: add to ~/Library/Application Support/Claude/claude_desktop_config.json
```

## How It Works

**Data model:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags). Structure lives in the links.

**Processing layers:**
- **Layer 0**  — split by headings, extract tags/links, FTS5 index, dedup hash
- **Layer 1**  — embed with local sentence-transformers, hub scoring via link aggregation
- **Layer 2**  — entity extraction, summaries. LLM required. Coming later.

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

# Tests
.venv/bin/python -m pytest tests/ -v

# Lint
.venv/bin/ruff check src tests
```

## License

MIT
