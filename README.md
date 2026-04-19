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

**One command:**

```
openaugi up     ← ingest + file watcher + zzz dispatch + task agent + MCP server
```

**ZZZ dispatch:** Write `zzz: <instruction>` anywhere in your notes — any capitalization works (`zzz`, `ZZZ`, `Zzz`). The file watcher detects changes, ingests the block, and writes a task file to `OpenAugi/Tasks/`. The task watcher picks it up and launches a Claude Code agent in a named tmux session. Attach any time with `tmux attach -t <task_id>`. The agent's behavior is governed by a skill file you edit in Obsidian. See [Getting Started](docs/GETTING_STARTED.md).

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

- **Blocks** — raw content (documents, entries, tags) with deterministic identity and optional `augi_tags` from agent classification
- **Links** — typed edges (split_from, tagged, links_to) that let agents traverse connections they wouldn't find through search alone

Five retrieval modes — semantic, keyword, graph traversal, time-based, direct lookup — all operating on the same graph.

**[Read the full data model](docs/data-model.md)** | Based on [Context Engineering is Index Design](https://bitsofchris.com/p/context-engineering-is-index-design)

---

## Clustering

After embedding, run `openaugi cluster` to generate a hierarchical map of your vault:

```bash
openaugi cluster --dry-run    # tune params, no writes
openaugi cluster              # write context_block:cluster nodes to DB
```

Configured as named passes in `~/.openaugi/config.toml`. Each pass runs HDBSCAN at a different embedding dimensionality — coarse pass (dim-64) surfaces life areas, fine pass (full-dim within each area) surfaces specific recurring ideas. Cross-domain pass finds connections across areas you wouldn't consciously link.

Cluster assignments land in each data_block's metadata (`cluster_assignments.{pass_id}`), making them queryable and renderable without joins. See **[Clustering](docs/clustering.md)** for config, SQL queries, and param tuning.

---

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** — full install guide, CLI reference, MCP tools, Claude registration
- **[Architecture](ARCHITECTURE.md)** — data model, processing layers, module map, design decisions
- **[Data Model](docs/data-model.md)** — philosophy, block kinds, navigation pattern, four-layer architecture
- **[Clustering](docs/clustering.md)** — HDBSCAN clustering: config format, data model, SQL queries, param tuning
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
