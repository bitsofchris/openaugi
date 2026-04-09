# OpenAugi

Your augmented knowledge base for Agentic work.

## Human Context for Agents

You've become the bottleneck.

Your notes are scattered. Your AI can't reach them. Every conversation starts from zero.

Claude and ChatGPT memory keep you stuck in a weird bubble.

You've built years of thinking in Obsidian, Google, ChatGPT — ideas, decisions, threads you've followed and dropped. But when you talk to your agent, none of that context exists. You repeat yourself. You lose threads. The AI that's supposed to help you think doesn't know what you've been thinking about.

**OpenAugi fixes this.** It turns your personal data vault turned into a knowledge graph that agents can search, traverse, and understand — semantically, by keyword, by tag, by time. One SQLite file. One MCP server. Everything stays on your machine.

> **Status:** Alpha. `pip install openaugi` and go. Evolving — expect rough edges. [MIT licensed.](LICENSE)

---

## Quick Start

```bash
pip install openaugi

# Configure and run
openaugi init    # one-time: vault path, embedding model, API key
openaugi up      # sync vault + start MCP server + watch for changes
```

Then [register with Claude](docs/GETTING_STARTED.md#register-with-claude) and start asking questions about your notes.

---

## What It Actually Does

```
Obsidian Vault --> split --> extract --> embed --> SQLite --> MCP Server --> Claude
```

**Ingest:** Splits your vault by headings, extracts tags and links, builds a graph of blocks and links in SQLite. Embeds everything for semantic search. Watches for changes and re-ingests automatically.

**Query:** Claude gets MCP tools to search (semantic + keyword), traverse your knowledge graph, fetch full context, and understand how your ideas connect. It can also write notes back to your vault.

**Two entry points:**

```
openaugi up     ← runs via Claude Desktop (ingest + file watcher + MCP server)
openaugi agent  ← run in a terminal (heartbeat every 5m + task dispatch)
```

**Process:** `openaugi agent` runs heartbeat on an interval — finds new blocks since the last run, hands them to a Claude Code agent that classifies them, dispatches tasks, and honors inline `zzz:` instructions you write in your notes. Rules live in a markdown skill file you edit in Obsidian. See [Getting Started](docs/GETTING_STARTED.md).

**Dispatch:** The agent also watches `OpenAugi/Tasks/` — when heartbeat writes a task file, it launches a Claude Code agent in a named tmux session automatically. Attach any time with `tmux attach -t <task_id>`. See [Task Dispatch](docs/task-dispatch.md).

---

## Why This Exists

Most "AI + notes" tools are cloud services that want your data. Or they're RAG demos that chunk your files and call it a day.

OpenAugi is different:

- **Your data stays yours.** One SQLite file on your machine. No cloud. No account.
- **Graph, not chunks.** Tags, links, and documents are first-class nodes. Claude can follow connections, not just match keywords.
- **Time-aware.** Your notes have history. OpenAugi preserves it — recently created, hub velocity, threads you dropped.
- **Composable.** MCP tools that Claude calls directly. No middle layer, no wrapper app.

This started as a personal tool to make Claude useful with a large Obsidian vault. It works well enough that it might be useful to others.

---

## Values

- **Privacy as foundation** — your data stays on your machine
- **Open by default** — MIT licensed, all code public
- **Augment, stay human** — amplify your thinking, don't replace it
- **Composable ecosystem** — building blocks that work together

---

## Data Model: Give Agents a Map

Most agent systems do brute-force retrieval — semantic search that stuffs the context window with raw documents. That's a magnifying glass in a warehouse. Agents need a map.

OpenAugi's data model is two tables — `blocks` and `links`:

- **Blocks** — raw content (documents, entries, tags) with deterministic identity and `augi_tags` from heartbeat classification
- **Links** — typed edges (split_from, tagged, links_to) that let agents traverse connections they wouldn't find through search alone

Five retrieval modes — semantic, keyword, graph traversal, time-based, direct lookup — all operating on the same graph.

**[Read the full data model](docs/data-model.md)** | Based on [Context Engineering is Index Design](https://bitsofchris.com/p/context-engineering-is-index-design)

---

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** — full install guide, CLI reference, MCP tools, Claude registration
- **[Architecture](ARCHITECTURE.md)** — data model, processing layers, module map, design decisions
- **[Data Model](docs/data-model.md)** — philosophy, block kinds, navigation pattern, four-layer architecture
- **[MCP Server](docs/MCP_SERVER.md)** — tool reference and tuning
- **[Task Dispatch](docs/task-dispatch.md)** — optional Obsidian → tmux dispatch: write a task, watcher launches a Claude Code agent in a named session
- **[Remote Access](docs/local.docs/REMOTE_ACCESS.md)** — Cloudflare Tunnel setup for Claude mobile

---

## Development

```bash
.venv/bin/pip install -e ".[dev]"
./scripts/check.sh          # lint + types + tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system map.

---

## License

MIT
