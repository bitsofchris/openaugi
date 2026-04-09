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

```bash
openaugi up
```

That's it. `up` does everything:

1. **Syncs your vault** — incremental ingest (skips unchanged files via content hash)
2. **Starts file watcher** — watches for `.md` changes, debounces (30s default), re-ingests automatically
3. **Starts MCP server** — stdio transport for Claude Desktop/Code

Embedding is attempted with your configured model. If it fails (no API key, model unavailable), blocks are saved without embeddings and retried on the next watcher cycle.

You can also run the pieces separately:

```bash
openaugi ingest            # one-off ingest without serving
openaugi serve             # MCP server only
openaugi watch             # file watcher only
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

| Command | What |
|---------|------|
| `openaugi init` | Configure embedding model, API key, vault path |
| `openaugi up` | MCP server + file watcher (the daily driver) |
| `openaugi ingest` | Run full Layer 0 + Layer 1 pipeline |
| `openaugi serve` | MCP server only (stdio or HTTP) |
| `openaugi watch` | File watcher only (incremental ingest on vault changes) |
| `openaugi heartbeat` | One-shot: ingest → hand new blocks to a Claude Code agent (see below) |
| `openaugi search "query"` | Search from terminal (semantic or `--keyword`) |
| `openaugi hubs` | Top connected notes by link count |
| `openaugi status` | Block/link/embedding counts |
| `openaugi service install` | Run as macOS launchd service (starts on boot) |

## Heartbeat — process new blocks with a Claude Code agent

`openaugi heartbeat` is a "dumb script, smart agent" loop for reviewing what
you've captured since the last run. The Python side finds entry blocks added
since the last heartbeat and spawns a Claude Code session with your OpenAugi
MCP tools. The agent classifies, looks up connections, follows any inline
`zzz:` instructions you wrote, and writes a log. One command, no config
system — the rules live in a markdown file you edit by hand.

### One-time setup: write the skill file

Create `<vault>/OpenAugi/heartbeat-skill.md` with your workstreams and default
rules. This is the config — edit it in Obsidian like any other note. The
command fails loudly if it's missing. A minimal example:

```markdown
# Heartbeat Skill

## Workstreams
- openaugi — the tool, the project, the code
- work — day job
- self — journal, reflection, life

## Defaults (when no zzz: instruction)
- Classify by folder if the signal is clear
- Tag with facets: type/idea, type/task, type/insight
- Unsure? Tag and move on. Don't block.

## What to write back
- Heartbeat log at OpenAugi/Heartbeat/YYYY-MM-DD.md summarizing
  blocks processed, classifications, connections, actions taken.
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
Had a thought about matryoshka embeddings for multi-res matching
zzz research this more - find papers in my vault

Need to fix the README onboarding flow before Thursday
zzz task - do this in the openaugi repo

Feeling stuck on the direction of openaugi today
zzz just log this, tag personal/reflection
```

Adding or editing a `zzz` line changes the block's content hash, so the next
heartbeat run will pick it up.

### Running heartbeat

```bash
openaugi heartbeat                    # ingest + find new blocks + spawn agent
openaugi heartbeat --dry-run          # build the prompt and print it (no agent)
openaugi heartbeat --max-blocks 20    # cap the batch size (default 50)
openaugi heartbeat --skip-ingest      # skip ingest (use when running alongside `up`)
```

Output goes to `<vault>/OpenAugi/Heartbeat/YYYY-MM-DD.md`. The last-run
timestamp is stored at `~/.openaugi/last_heartbeat` and only advances on a
successful agent run — a failed run retries the same window next time.

### Running heartbeat alongside `openaugi up`

`up` and `heartbeat` both call incremental ingest, but they're safe to run
together: SQLite WAL mode serializes writes, block IDs are deterministic, and
re-inserts are idempotent. The only cost of running both is duplicated file
hashing. Pass `--skip-ingest` to `heartbeat` when `up` is already running and
its watcher has the vault up to date.

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
