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

A bare install has **no embedding provider**: it searches by keyword, title,
tag, date and links, and nothing leaves your machine. Add semantic search with
one extra:

```bash
pip install "openaugi[local]"    # sentence-transformers, on-device (pulls in PyTorch)
pip install "openaugi[openai]"   # OpenAI embeddings; note text is sent to OpenAI
```

From a source checkout with uv: `uv sync --extra openai` (or `--extra local`).

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

Three questions, each with a safe default (press Enter):

1. 📁 **Vault** — where your notes are. It counts them so you know it's the right folder.
2. 🔎 **Search** — 🔤 keyword + links (built in, default), 💻 local semantic
   (`openaugi[local]`), or ☁️ OpenAI semantic (`openaugi[openai]` + API key; note
   text goes to OpenAI). It shows which are installed and how to install the rest.
3. 🪶 **How it runs** — 🪶 Light (default): the CLI only. Agents call
   `openaugi kb …`; no server, no watcher, nothing in the background, and
   `zzz:` tasks wait for `openaugi tasks run`. ⚡ Full: `openaugi up` keeps the
   index fresh and launches an agent for each `zzz:`, and Claude Code connects
   to `openaugi serve` over MCP. Both have every tool.

It writes `~/.openaugi/config.toml`, copies the agent files into
`<vault>/OpenAugi/AGENT/` (never overwriting yours), and prints the next
commands for what you chose. Re-run it any time: your current choices are the
defaults, and an existing config is only replaced if you say so, with a dated
backup (`config.toml.bak-…`) kept next to it. Replacing drops settings `init`
doesn't ask about (exclude patterns, source rules) — copy them back from the
backup.

Unattended: `openaugi init --vault ~/notes --search none --mode light --yes`.
(0.3.0's `--tasks manual|auto` still works and means `--mode light|full`.)

The vault path may use `~` (`~/Documents/vault`) — it is expanded to an absolute path everywhere it is used, whether it comes from `init`, a `--path` flag, or `[vault] default_path` edited by hand.

## Run

Pick one of two ways. `openaugi init` asks which you want.

### 🪶 Light (default) — the CLI only

```bash
openaugi ingest                                  # index; re-run whenever you want it fresh
openaugi kb get-context --query "budget"         # what an agent runs: JSON out, then exit
openaugi kb --help                               # the agent's guide and every tool
```

No server, no watcher, no daemon: nothing is running between calls. Every MCP
tool is an `openaugi kb` command with the same arguments and the same JSON —
`search`, `get-context`, `get-block(s)`, `get-related`, `traverse`, `recent`,
`run-query`, the view and review tools, and the writes (`write-document`,
`write-record`, …). Arguments map one to one: `--block-id X`; a list repeats
its option (`--tags a --tags b`); a JSON argument is a string (`--data
'{"k": "v"}'`); `-` reads a value from stdin (`--content - < note.md`). Exit 0 is
an answer, 1 an error envelope from the tool, 2 bad arguments.

Tell your agent once — `init` prints this for `~/.claude/CLAUDE.md`:

```markdown
## My notes (OpenAugi)
My notes vault is indexed by OpenAugi. Before grep/find, use its CLI:
run `openaugi kb --help` once for the tools, start research with
`openaugi kb get-context --query "..."`. Output is JSON.
```

`zzz:` lines still become task files on `ingest`; an agent starts only if you
run `openaugi tasks run`. See
[Task dispatch § On demand](task-dispatch.md#on-demand-openaugi-tasks).

### ⚡ Full — background service `openaugi up` + MCP

```bash
openaugi up        # the service — leave it running
```

Does four things:

1. **Syncs your vault** — incremental ingest (skips unchanged files via content hash)
2. **Starts file watcher** — watches for `.md` changes, debounces (30s default), re-ingests automatically
3. **Dispatches zzz instructions** — blocks with `zzz:` lines get written as task files to `OpenAugi/Tasks/`
4. **Launches agents** — only with `[tasks] auto_dispatch = true` (or `--agent`): the task watcher picks up pending task files and launches Claude Code agents in named tmux sessions. Otherwise they wait for `openaugi tasks run`

**It does not serve MCP.** That is `openaugi serve`, and it is what you
register with your client (see [Register with Claude](#register-with-claude)).

### Why two commands

They need opposite cardinality:

- **`up` is one per vault.** Two watchers see the same `zzz:` land and both
  dispatch it, so one instruction becomes two agents racing over one database.
  It takes a lock (`~/.openaugi/up.lock`) and a second one exits rather than
  competing.
- **`serve` is one per client.** Every stdio MCP client spawns its own — that
  is how the transport works. It takes no lock and any number can run.

Registering `up` as your MCP server conflates the two: your client spawns a
new one each time it starts, and only the first can hold the lock.

```bash
openaugi up --serve        # both, for a single terminal with one client
openaugi up --agent        # launch agents this run, whatever auto_dispatch says
openaugi up --no-agent     # never launch agents this run (watcher only)
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
  /path/to/openaugi/.venv/bin/openaugi serve
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "openaugi": {
      "command": "/path/to/openaugi/.venv/bin/openaugi",
      "args": ["serve"]
    }
  }
}
```

Keep `openaugi up` running separately — as a launchd/systemd service, or in a
terminal. If you previously registered `up` here, change it to `serve`.

### Remote access (advanced)

The MCP server supports HTTP transport with OAuth authentication for remote access from Claude mobile. Requires a Cloudflare account, domain, and tunnel setup. See [Remote Access](../local.docs/REMOTE_ACCESS.md).

## CLI Reference

### All commands

| Command | What |
|---------|------|
| `openaugi init` | Guided setup: vault path, search mode, Light or Full |
| `openaugi kb <tool>` | Every MCP tool as a command, JSON out, no server (`kb --help` lists them) |
| `openaugi tasks` | Sync the vault and list `zzz:` tasks; `tasks show <#>`, `tasks run <#>` |
| `openaugi up` | **The service, one per vault:** ingest + watcher + zzz → task files; launches agents only with `[tasks] auto_dispatch = true`. Does not serve MCP |
| `openaugi up --serve` | Also serve MCP in the foreground (single terminal, one client) |
| `openaugi up --agent` / `--no-agent` | Override `auto_dispatch` for this run |
| `openaugi task-dispatch` | Task launcher only (standalone, foreground) |
| `openaugi ingest` | One-off ingest without serving |
| `openaugi serve` | **The MCP interface, one per client** (stdio or HTTP). Register this with Claude |
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
| `write_document` | Create a markdown note in your vault (`OpenAugi/{subfolder}/`); `overwrite=true` only for regenerable `Views/` |

### Review Pass Tools

| Tool | What |
|------|------|
| `apply_routing` | Route blocks into (or out of) container notes (AMOC/PMOC/MOC) as `routed_to` links — batch `add`/`remove` per decision; membership is links, tags stay taxonomy-only |
| `get_review_state` | Read the review-pass high-water mark |
| `mark_review_complete` | Advance the high-water mark after a completed pass |

The review pass is the write-back loop: it routes new blocks, regenerates
derived view notes under `OpenAugi/Views/`, and surfaces nominations on a
Dashboard. Capture grammar: `qqq` (block delimiter) · `zzz:` (dispatch a
task) · `aaa:` (routing instruction to the review pass). See
[review-pass.md](review-pass.md).

See [MCP Server docs](MCP_SERVER.md) for tool parameters and tuning.

## How It Works

**Data model:** Two tables — `blocks` and `links`. Everything is a block (documents, entries, tags). Structure lives in the links.

**Processing layers:**
- **Layer 0** (free) — split by headings, extract tags/links, FTS5 index, dedup hash
- **Layer 1** (~$0) — embed with OpenAI or local sentence-transformers, hub scoring via link aggregation
- **Layer 2** (coming) — entity extraction, summaries. LLM required.

See [ARCHITECTURE.md](../../ARCHITECTURE.md) for the full system design, module map, and design decisions.

## Logs & Debugging

Logs are at `~/.openaugi/logs/openaugi.log` (rotated, DEBUG level). Check here when the MCP server or ingest crashes.
