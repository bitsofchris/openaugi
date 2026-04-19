---
name: getting-started
description: Full install guide, CLI reference, MCP tools, and Claude registration for OpenAugi
---

# Getting Started with OpenAugi

Everything you need to install, configure, and run OpenAugi with Claude.

## Install

```bash
pip install openaugi
```

For local embeddings (free, no API key needed):

```bash
pip install "openaugi[local]"
```

### Install from source (development)

```bash
git clone https://github.com/bitsofchris/openaugi.git
cd openaugi
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Configure

```bash
openaugi init
```

Interactive setup — choose embedding model (local or OpenAI), set API key if needed, set your vault path. Config is saved to `~/.openaugi/config.toml`.

## Run

One command does everything:

```bash
openaugi up
```

Add to your Claude Desktop MCP config (see [Register with Claude](#register-with-claude)) and it starts automatically. Does four things:

1. **Syncs your vault** — incremental ingest (skips unchanged files via content hash)
2. **Starts file watcher** — watches for `.md` changes, debounces (30s default), re-ingests automatically
3. **Dispatches zzz instructions** — blocks with `zzz:` lines get written as task files to `OpenAugi/Tasks/`
4. **Launches agents** — task watcher picks up pending task files and launches Claude Code agents in named tmux sessions
5. **Starts MCP server** — stdio transport so Claude can query your vault

```bash
openaugi up --no-agent     # disable task dispatch (watcher + MCP only)
openaugi up --debounce 10  # faster watcher response (default 30s)
```

Embedding is attempted with your configured model. If it fails (no API key, model unavailable), blocks are saved without embeddings and retried on the next watcher cycle.

You can also run the pieces separately:

```bash
openaugi ingest            # one-off ingest without serving
openaugi serve             # MCP server only
openaugi watch             # file watcher only
openaugi task-dispatch     # task dispatch only (standalone)
```

## Register with Claude

### Claude Code

```bash
claude mcp add --transport stdio --scope user openaugi -- \
  /path/to/openaugi/.venv/bin/openaugi up
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

### Remote access (advanced)

The MCP server supports HTTP transport with OAuth authentication for remote access from Claude mobile. Requires a Cloudflare account, domain, and tunnel setup. See [Remote Access](local.docs/REMOTE_ACCESS.md).

## CLI Reference

### All commands

| Command | What |
|---------|------|
| `openaugi init` | Configure embedding model, API key, vault path |
| `openaugi up` | Ingest + watcher + zzz dispatch + task agent + MCP server |
| `openaugi up --no-agent` | Same but without task dispatch (no tmux agent sessions) |
| `openaugi task-dispatch` | Task dispatch only (standalone) |
| `openaugi ingest` | One-off ingest without serving |
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only |
| `openaugi search "query"` | Search from terminal (semantic or `--keyword`) |
| `openaugi hubs` | Top connected notes by link count |
| `openaugi status` | Block/link/embedding counts |
| `openaugi service install` | Run as macOS launchd service (starts on boot) |

## ZZZ Dispatch — trigger agents from your notes

Write `zzz: <instruction>` anywhere in your notes to dispatch work to a Claude
Code agent. Any capitalization works — `zzz`, `ZZZ`, `Zzz`, `zZz` are all
recognized. The colon is optional (`zzz research this` works too).

### How it works

```
you write `zzz: research deep learning` in a daily note
  → file watcher detects change (30s debounce)
  → ingest: parses the file, splits into blocks, extracts tags/links
  → zzz dispatch: sees the instruction, writes a task file to OpenAugi/Tasks/
  → task watcher: picks up the pending file, launches claude in a tmux session
  → agent reads the augi-agent skill file, uses MCP tools, does the work
  → writes output to OpenAugi/ tagged #human-review
  → marks task file status: done
```

The `zzz:` line is stripped from the block content and stored as metadata — it
won't appear in search results or rendered views. Adding or editing a `zzz:`
line changes the block's content hash, so the next ingest cycle picks it up.

### Examples

```markdown
Need to fix the README onboarding flow before Thursday
zzz: task - do this in the openaugi repo

Had a thought about matryoshka embeddings for multi-res matching
ZZZ: research this, find papers, write a summary note

Interesting thread on context engineering
zzz look into what's in my vault about this
```

### The agent skill file

The agent's behavior is governed by `<vault>/OpenAugi/AGENT/augi-agent.md`. This is
the source of truth for how the agent handles tasks — edit it in Obsidian to
change behavior. The template lives at `src/openaugi/templates/augi-agent.md`
and is copied to the vault by `openaugi init`.

The skill file tells the agent:
- What MCP tools are available (search, traverse, get_context, etc.)
- How to handle different instruction types (research, task, freeform)
- Hard rules (never modify raw notes, write to OpenAugi/, tag #human-review)

### Watching agent sessions

Task files land in `<vault>/OpenAugi/Tasks/`. Each gets a tmux session:

```bash
tmux ls                          # list active sessions
tmux attach -t TASK-2026-04-19-research-embeddings  # watch it work
```

When the agent finishes, it fills in `## Results` in the task file and sets
`status: done`. If it gets stuck, it sets `status: needs-input` and adds
items to `## Human Todo`.

## MCP Tools

These are the tools Claude gets when connected to OpenAugi:

### Read Tools

| Tool | What |
|------|------|
| `search` | Semantic (sqlite-vec KNN) + keyword (FTS5) + tag/time filters |
| `get_block` | Full block content by ID |
| `get_blocks` | Batch fetch up to 50 blocks by ID |
| `get_related` | Follow links from a block |
| `traverse` | Multi-hop graph walk |
| `get_context` | Compound search -> deduplicate -> MMR re-rank -> expand via links |
| `recent` | Recently created blocks |
| `tag_block` | Stamp AI-classified `augi_tags` onto a block |

### Write Tools

| Tool | What |
|------|------|
| `write_document` | Create a markdown note in your vault |
| `write_thread` | Save a conversation thread as a note |
| `write_snip` | Save a curated snippet to `OpenAugi/Snips/` |

### Workstream Tools

| Tool | What |
|------|------|
| `list_streams` | List workstreams with status and left-off preview |
| `get_stream_context` | Load full workstream state for resuming work |
| `make_stream` | Create a new workstream |
| `update_stream` | Update workstream left-off, context, log, or status |

See [MCP Server docs](MCP_SERVER.md) for tool parameters and tuning.

## How It Works

**Data model:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags). Structure lives in the links.

**Processing layers:**
- **Layer 0** (free) — split by headings, extract tags/links, FTS5 index, dedup hash
- **Layer 1** (~$0) — embed with OpenAI or local sentence-transformers, hub scoring via link aggregation
- **Layer 2** (coming) — entity extraction, summaries. LLM required.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full system design, module map, and design decisions.

## Logs & Debugging

Logs are at `~/.openaugi/logs/openaugi.log` (rotated, DEBUG level). Check here when the MCP server or ingest crashes.
