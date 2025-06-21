# AugiMCP

Personal augmented intelligence.

Your private data layer - organized, de-duplicated, and fully connected.

## How it Works

### Vision in a Nutshell

**AugiMCP** is a **local-first**, Python-powered FastAPI service that: 

> 1. **Ingests** all your personal data (Obsidian MD, Drive docs, ChatGPT/Anthropic exports, MCP streams…)
> 2. **Stores** raw chunks + vector embeddings + timestamps in an embedded DuckDB database
> 3. **Distills** those chunks into “concept nodes” via clustering + LLM summarization
> 4. **Exposes** a simple JSON API (ingest, list concepts, retrieve, evolve) that any client (Obsidian plugin, CLI, web UI) can call
> 5. **Grows** over time with plug-in adapters, hierarchical summarization, type tagging (idea vs. journal vs. ref), and “personal intelligence” analytics

All your context—projects, goals, reflections, chat logs—lives in one portable, versioned, time-indexed vault layer you control.


### Core Architecture

```text
┌─────────────────────┐
│  Source Adapters    │
├─────────────────────┤
│ • Obsidian MD       │
│ • Google Drive API  │
│ • Chat exports      │
│ • MCP streams       │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Ingest Pipeline    │
├─────────────────────┤
│ • Chunk content     │
│ • Generate embeddings│
│ • Store in DuckDB   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Distillation Loop   │
├─────────────────────┤
│ • Cluster embeddings│
│ • Summarize clusters│
│ • Create concepts   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  FastAPI Server     │
├─────────────────────┤
│ POST /ingest        │
│ POST /distill       │
│ GET  /concepts      │
│ GET  /retrieve      │
│ GET  /evolution     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│      Clients        │
├─────────────────────┤
│ • Obsidian plugin   │
│ • CLI tools         │
│ • Web dashboard     │
└─────────────────────┘
```


## Milestones & Version Roadmap

### 🎯 **V0: Chat-First Distiller** (1–2 days)

- **Python FastAPI + DuckDB**: bootstrap schema for `raw_chunks`, `concept_nodes`, `links`
- **Chat adapter**: ingest a folder of ChatGPT/Anthropic JSON exports into `raw_chunks`
- **Embed + cluster + summarize**: implement the simple loop (no agent frameworks)
- **API endpoints**:

  - `POST /ingest/chats`
  - `POST /distill/chats`
  - `GET /concepts`
- **Proof**: CLI or minimal Obsidian plugin to list “today’s chat concepts”

### 🚀 **V1: Obsidian Integration & Local MD Ingest** (1 week)

- **MD adapter**: read vault `.md`, chunk by heading or simple splitter
- **Incremental ingest**: file-watcher or manual command to pick up new/changed notes
- **Obsidian plugin** (JS/TS + duckdb-wasm):

  - UI command “Sync with Augi MCP”
  - Panel showing concept summaries + links back to vault files
- **Time queries**: simple `/retrieve?since=2025-06-01` endpoint

### 🔌 **V2: Multi-Source & Modular Adapters** (2–3 weeks)

- **Google Drive adapter**: pull docs, convert to text, ingest
- **MCP stream adapter**: hook into live chat streams for real-time ingest
- **Plugin settings**: enable/disable sources, set paths/API keys
- **Vector store tuning**: experiment with DuckDB’s vector extension or a FAISS backend

### 🏗 **V3: Hierarchical Summarization & Type-Tagging** (3–4 weeks)

- **LLM-guided chunking**: swap simple splitter for LlamaIndex text-splitter
- **Concept hierarchy**: summarize concept\_nodes into “meta-concepts” (two-level summarization)
- **Node classification**: run an LLM classifier to tag each concept as `idea | journal | reference | highlight`
- **Enhanced API**:

  - `GET /concepts?type=journal`
  - `GET /hierarchy`

### 🧠 **V4: Personal Intelligence & Analytics** (Ongoing)

- **Evolution endpoint**: `GET /evolution?concept_id=…` → timeline of updates & embedding drift
- **Dashboards**: charts of “new concepts/week,” “stale concepts not touched in 30d,” “repeated patterns”
- **Agent/Memory frameworks (opt-in)**: integrate Mem0 or GraphRAG for dynamic memory retrieval if users want more powerful chains
- **Packaging & Distribution**: PyPI package, `mcp-server` CLI, Docker image, optional PyInstaller desktop binary
