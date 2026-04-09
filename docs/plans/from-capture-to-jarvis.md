# OpenAugi Roadmap — From Capture to Jarvis

---

## Core Architecture Principle

**Graph structure and taxonomy are separate things.**

- **Links** = connections between ideas. "This relates to that." They build the web. They do NOT imply ownership or classification.
- **Taxonomy** = where something lives. Workstreams and concepts. Minimum structure to stratify data for retrieval.
- **Concepts** = emergent, not assigned. They're summaries of hubs (notes with many inbound links) or clusters (blocks close in embedding space). The system discovers them, you confirm them.
- **Workstreams** = the minimum taxonomy. Assigned by context (which journal are you writing in?) or by LLM classification against a known set. This is what the agent learns on initialization and reviews with you.

---

## Layer 1: Always-On Sense-Making (Build Now)

### 1A — Deterministic Pipeline (This Week)

- [ ] File watcher detects changes in vault (Obsidian sync as trigger)
- [ ] Block splitter: split on `qqq` delimiter, extract individual blocks
- [ ] Block identity: content hash per block to detect new vs. changed vs. unchanged
- [ ] Deterministic extraction per block:
  - Links (`[[...]]`) → create graph edges (connections, NOT classification)
  - Tags (`#...`) → classification hints for LLM
  - `zzz` token → optional LLM instruction metadata (stripped from clean version)
  - `- [ ]` → type = task
  - Date headers → type = journal
  - URL-heavy → type = reference
- [ ] Source context: which file did this block come from?
  - Area journal (e.g. OpenAugi journal) → workstream = that area
  - Daily note → workstream needs LLM classification
- [ ] Index new/changed blocks into OpenAugi's SQLite graph
- [ ] Embed new blocks for semantic search

### 1B — Workstream Classification (Next Week)

The minimum taxonomy. The coarsest level of "what is this about."

- [ ] On initialization: agent reads vault structure, proposes workstream taxonomy based on existing areas/journals/MOCs
- [ ] User reviews and confirms the workstream list (5-10 top-level streams max)
- [ ] For blocks from area journals: workstream assigned deterministically (you're writing in the OpenAugi journal → it's OpenAugi)
- [ ] For blocks from daily notes or unscoped files: LLM classifies against the known workstream list
- [ ] Tags and content steer classification, NOT links
- [ ] Links between blocks in different workstreams are cross-stream connections — valuable signal, not a classification error
- [ ] New workstream proposals flagged for user review

### 1C — Concept Discovery (Weeks 3-4)

Concepts are emergent, not assigned. The system finds them, you name them.

- [ ] Hub detection: identify notes/blocks with many inbound links — these are natural concept nodes
- [ ] Cluster detection: find groups of blocks that are semantically close in embedding space, even without explicit links
- [ ] Concept page generation: for each hub or cluster, generate a summary page in OpenAugi/Concepts/
  - What blocks belong to this cluster
  - What the connecting theme is
  - Key links to other concepts
- [ ] Concept naming: LLM proposes a name, you confirm or rename
- [ ] Concept hierarchy: some concepts naturally nest under workstreams, some span multiple workstreams — that's fine, don't force a tree
- [ ] Periodic re-clustering: as new blocks arrive, clusters shift. Rerun discovery on a schedule.

### 1D — Rendered View (Weeks 3-4, parallel with 1C)

- [ ] OpenAugi/ directory in vault with system-maintained markdown files
- [ ] Taxonomy.md — the living index:
  - Workstreams with block counts
  - Discovered concepts with hub scores
  - Recent additions
  - Pending review items (new concept proposals, ambiguous classifications)
- [ ] Stream pages (OpenAugi/Streams/openaugi.md) — blocks in this workstream, organized by discovered concepts
- [ ] Concept pages (OpenAugi/Concepts/retrieval-engineering.md) — hub summary + all related blocks + cross-stream connections
- [ ] Raw notes never modified — append-only, you own them completely

### Layer 1 — Open Questions

- How many workstreams is the right starting set? (Recommend: let the agent propose from vault structure, you prune to 5-10)
- Cluster detection: pure embedding similarity, or hybrid with link co-occurrence? (Start with embeddings, add link signal later)
- How often to rerun concept discovery? (Weekly? On-demand? When block count since last run exceeds threshold?)
- Concept pages vs. just taxonomy entries — how much rendered content is useful vs. noise?
- Blocks that span multiple workstreams — tag both, or force one?

---

## Layer 2: Proactive Agents & Dispatch (Weeks 5-7)

The system acts on what it finds.

### 2A — Task Dispatch

- [ ] Blocks typed as "task" spawn agent tasks
- [ ] Agent pulls relevant context from OpenAugi graph (search + traverse) before executing
- [ ] Task results written back to OpenAugi/Threads/ and linked to source block
- [ ] Task status tracked (pending, in-progress, done)

### 2B — Proactive Processing

- [ ] Periodic lint pass: orphan blocks, missing connections, contradictions
- [ ] Connection suggestions: "this new block is semantically close to these 3 blocks from different workstreams"
- [ ] Concept evolution alerts: "the retrieval-engineering concept has grown significantly — review the summary"
- [ ] Stale concept detection: concepts with no new blocks in 30+ days

### 2C — Content Shipping Pipeline

- [ ] Public second brain as living OpenAugi showcase
- [ ] Content ideas auto-queued with context assembled from related concepts
- [ ] Draft → publish pipeline with less friction

### 2D — OpenClaw / AugiRouter Integration

- [ ] Obsidian sync as dispatch trigger
- [ ] Agent management: see running tasks, queued work
- [ ] Notifications: task complete, review needed, interesting connection found

---

## Layer 3: The Interface (Weeks 8+)

Your second brain as the primary surface, not a chat window.

### 3A — Workstream View

- [ ] See streams, discovered concepts within them, active agent tasks
- [ ] Not chat UI — knowledge UI with chat as a tool inside it
- [ ] Context pulling like Cursor — you type, relevant blocks surface

### 3B — Session Management

- [ ] Spawn chat sessions from any block or concept
- [ ] Sessions link back to source block
- [ ] Ephemeral by default — snip/save to persist
- [ ] Chat history attached to blocks, not floating in separate apps

### 3C — Multi-Agent Orchestration

- [ ] Multiple agents for different workstreams
- [ ] Agent-to-agent context passing through shared graph

---

## Layer 4: Network & Scale (Future)

### 4A — Advanced Retrieval

- [ ] Matryoshka embeddings for multi-resolution matching
- [ ] Progressive disclosure: coarse → drill-down → fine-grained
- [ ] Hub scoring improvements

### 4B — Federation & Social

- [ ] Multi-vault support
- [ ] Cross-vault federation
- [ ] Second brain social network

### 4C — Remote & Mobile

- [ ] Phone capture → desktop processing
- [ ] Mobile app routing to OpenAugi
- [ ] Cloud option for non-local users

---

## The Data Model in Plain English

```
You write messy notes. You split blocks with qqq.

DETERMINISTIC:
  Block is from OpenAugi journal → workstream = openaugi
  Block has - [ ] → type = task
  Block has [[Muness]] → graph edge to Muness (NOT classification)
  Block has #openaugi/features → classification hint

LLM (async, batched):
  Block is from daily note, mentions retrieval → workstream = openaugi
  Block is ambiguous → classify against known workstream list

EMERGENT:
  15 blocks all link to "agentic retrieval" → hub detected → concept page
  20 blocks cluster in embedding space around "capture workflows"
    → cluster detected → concept proposed → you name it

RENDERED:
  Taxonomy.md: your workstreams, discovered concepts, counts
  Stream pages: blocks organized by concept within each workstream
  Concept pages: hub summaries with all related blocks and cross-links

Your raw notes: untouched, forever.
```

---

## This Week

1. Ship block splitter + deterministic extraction (1A)
2. Workstream initialization: agent reads vault, proposes taxonomy, you confirm
3. Clean up README / onboarding flow
4. Write and publish Thursday post
