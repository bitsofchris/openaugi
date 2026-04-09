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

OpenAugi has two entry points that run alongside each other:

**1. `openaugi up` — data + MCP (run via Claude Desktop)**

Add to your Claude Desktop MCP config (see [Register with Claude](#register-with-claude)) and it starts automatically. Does three things:

1. **Syncs your vault** — incremental ingest (skips unchanged files via content hash)
2. **Starts file watcher** — watches for `.md` changes, debounces (30s default), re-ingests automatically
3. **Starts MCP server** — stdio transport so Claude can query your vault

**2. `openaugi agent` — heartbeat + task dispatch (run in a terminal)**

```bash
openaugi agent
```

Runs alongside `up` in a terminal window. Does two things:

1. **Heartbeat every 5 minutes** — finds new blocks, hands them to a Claude Code agent that classifies, chases connections, and honors `zzz:` instructions
2. **Task dispatch** — watches `OpenAugi/Tasks/` and launches pending task files as Claude Code agents in named tmux sessions

```bash
openaugi agent --interval 10              # run heartbeat every 10 minutes
openaugi agent --ignore-source 'HW/*'    # skip blocks from this source path pattern
openaugi agent --ingest                  # run ingest too (if `up` is not running)
```

Embedding is attempted with your configured model. If it fails (no API key, model unavailable), blocks are saved without embeddings and retried on the next watcher cycle.

You can also run the pieces separately:

```bash
openaugi ingest            # one-off ingest without serving
openaugi serve             # MCP server only
openaugi watch             # file watcher only
openaugi heartbeat         # one-shot heartbeat (useful for testing)
openaugi task-dispatch     # task dispatch only
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

### Daily use (two entry points)

| Entry point | How it runs | What it does |
|-------------|-------------|--------------|
| `openaugi up` | Claude Desktop MCP config | Vault sync + file watcher + MCP server |
| `openaugi agent` | One terminal window | Heartbeat every 5m + task dispatch |

### All commands

| Command | What |
|---------|------|
| `openaugi init` | Configure embedding model, API key, vault path |
| `openaugi up` | Ingest + file watcher + MCP server (run via Claude Desktop) |
| `openaugi agent` | Heartbeat loop + task dispatch (run in terminal alongside `up`) |
| `openaugi heartbeat` | One-shot heartbeat (useful for testing/debugging) |
| `openaugi task-dispatch` | Task dispatch only (standalone, no heartbeat loop) |
| `openaugi ingest` | One-off ingest without serving |
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only |
| `openaugi search "query"` | Search from terminal (semantic or `--keyword`) |
| `openaugi hubs` | Top connected notes by link count |
| `openaugi status` | Block/link/embedding counts |
| `openaugi service install` | Run as macOS launchd service (starts on boot) |

## Heartbeat — process new blocks with a Claude Code agent

`openaugi heartbeat` is a "dumb script, smart agent" loop for reviewing what
you've captured since the last run. The Python side finds entry blocks added
since the last heartbeat and spawns a Claude Code session with your OpenAugi
MCP tools. The agent classifies each block (`area/`, `type/`, `status/` tags
written back to the block via `tag_block`), dispatches actionable items as
task files, and writes an audit log. One command, no config system — the
rules live in a markdown file you edit by hand.

### One-time setup: write the skill file

Create `<vault>/OpenAugi/heartbeat-skill.md` with your workstreams and default
rules. This is the config — edit it in Obsidian like any other note. The
command fails loudly if it's missing. A minimal example:

```markdown
# Heartbeat Skill

## Areas (customize for your vault)
- area/openaugi — the tool, the project, the code
- area/work — day job
- area/self — journal, reflection, life

## Defaults (when no zzz: instruction)
- Classify by folder if the signal is clear (path beats content)
- type/task for actionable items only — everything else just gets an area
- Unsure? Tag what you're confident about and flag the rest.

## What to write back
- Always call tag_block to stamp the classification onto the block.
- Write task files to OpenAugi/Tasks/ for anything actionable.
- Heartbeat log at OpenAugi/Heartbeat/YYYY-MM-DD.md.
- Never modify raw source notes.
```

See [docs/plans/heartbeat.md](plans/heartbeat.md) for the full rationale and a
more complete example.

### Per-block `zzz:` instructions

You can steer the agent on a per-block basis by writing a line starting with
`zzz` inline in your notes. The vault adapter strips these from the clean
content and attaches them to the block as metadata — the heartbeat prompt then
surfaces them per-block so the agent honors each one.

```markdown
Need to fix the README onboarding flow before Thursday
zzz task - do this in the openaugi repo

Feeling stuck on the direction of openaugi today
zzz just log this, tag area/self

Had a thought about matryoshka embeddings for multi-res matching
zzz task - research this, find papers, write a summary note
```

Adding or editing a `zzz` line changes the block's content hash, so the next
heartbeat run will pick it up.

### Running heartbeat

```bash
openaugi heartbeat                    # ingest + find new blocks + spawn agent
openaugi heartbeat --dry-run          # build the prompt and print it (no agent)
openaugi heartbeat --max-blocks 20    # cap the batch size (default 50)
openaugi heartbeat --ingest           # run ingest first (use when `up` is not running)
```

Output goes to `<vault>/OpenAugi/Heartbeat/YYYY-MM-DD.md`. The last-run
timestamp is stored at `~/.openaugi/last_heartbeat` and only advances on a
successful agent run — a failed run retries the same window next time.

### Running heartbeat alongside `openaugi up`

Heartbeat skips ingest by default — it assumes `up` is running and keeping
the DB current. If `up` is not running, pass `--ingest` to do a one-off
incremental ingest first. Both are safe to run together (SQLite WAL mode
serializes writes, block IDs are deterministic, re-inserts are idempotent)
but the default avoids redundant file hashing when `up` is already watching.

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
| `tag_block` | Stamp AI-classified `augi_tags` onto a block (used by heartbeat agent) |

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
