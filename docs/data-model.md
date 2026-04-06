---
name: data-model
description: OpenAugi's data model philosophy — blocks, links, context blocks, and why agents need a map, not a magnifying glass
---

# Data Model

OpenAugi's data model is built on one premise: **agents need a map, not a magnifying glass.**

Most agent systems today do brute-force retrieval. Semantic search stuffs the context window with raw documents. GraphRAG extracts granular entities at high cost. Both miss the middle path: lightweight structure that lets an agent navigate before it retrieves.

This is [context engineering as index design](https://bitsofchris.com/p/context-engineering-is-index-design).

## Philosophy

Agents are smart. Intelligence is commoditizing. The bottleneck isn't the model or the prompt — it's the data representation.

Without a map, agents pick one retrieval mode, grab a bunch of data, and hope for the best. With a map, they decompose a query, route each piece to the right mode, and scope every call. They explore widely and go deep only when the map shows something is worth it.

The query pattern that works:

```
Map  -->  Assess  -->  Drill  -->  Retrieve
```

The agent reads a lightweight map of what exists, assesses relevance, drills into promising areas, then retrieves only what it needs. This pattern appears independently across RAPTOR, GraphRAG, LazyGraphRAG, HippoRAG, and A-RAG — different teams, different domains, same conclusion.

OpenAugi implements this pattern with two primitives and one compile step.

## Two Tables

The entire store is two tables. That's it.

```sql
blocks (id, kind, content, summary, embedding, source, title, tags, timestamp, metadata, content_hash)
links  (from_id, to_id, kind, weight, metadata)  -- PK: (from_id, to_id, kind)
```

Everything is a block. Structure lives in the links, not in the schema.

## Data Blocks

A data block is the raw thing — a document, a journal entry, a tag. It has content and whatever inherent metadata comes with it: timestamps, file paths, source, tags.

**Block kinds:**

| Kind | What it holds | Example |
|------|--------------|---------|
| `document` | A full source file | An Obsidian note |
| `entry` | A section split from a document | An H3-dated journal entry |
| `tag` | A first-class tag node | `#project/openaugi` |
| `context` | Generated navigational metadata | Hub summary, concept page, index |

Data blocks are like parquet files — raw data with a footer full of statistics. The agent should rarely read these directly as a first step. It should read the map first.

### Block identity

Block IDs are deterministic: `hash(source_path + content_hash)`. This gives stable identity across section reordering and incremental re-ingestion. If the content hasn't changed, the block doesn't change.

### Tags as blocks

Tags are first-class graph nodes, not string annotations. This means hub scoring, traversal, and entity resolution work uniformly across all block kinds. A tag like `#data-engineering` is a block that connects to every document and entry tagged with it — making it a natural navigation hub.

## Links

Links are typed, weighted edges between blocks. The composite primary key `(from_id, to_id, kind)` allows multiple relationship types between the same pair of blocks.

**Link kinds:**

| Kind | Meaning | Example |
|------|---------|---------|
| `split_from` | Entry was extracted from document | Journal entry -> parent note |
| `tagged` | Block has this tag | Entry -> tag block |
| `links_to` | Explicit reference (wikilink) | Note A -> Note B |
| `summarizes` | Context block summarizes source | Hub summary -> tag block |

The graph that emerges from these links is what the agent actually navigates.

## Context Blocks

Context blocks are the map. They're generated metadata about data blocks — enough for an agent to make a routing decision without reading the underlying content.

A context block answers one question: **"Should the agent look here?"**

OpenAugi generates context blocks through the [compile step](../ARCHITECTURE.md#compile):

| Context block | What it provides | How it's built |
|---------------|-----------------|----------------|
| **Hub summaries** | Top tags by connectivity, what they link to | SQL aggregation over links |
| **Concept pages** | Topic clusters with key entries | Semantic grouping |
| **Index** | Navigational map of the entire graph | Aggregation of all context blocks |

Context blocks are stored as `Block(kind="context", source="compiled")` — same table, same query interface, but serving a fundamentally different purpose. They're the table of contents, not the chapters.

### Scope is flexible

A context block might describe a single document, a semantic cluster, a time window, or an entire data source. What matters is that it contains signals extracted from content — topic labels, entity mentions, temporal ranges, connectivity scores — not the content itself.

## The Navigation Pattern

With this architecture, agents have five retrieval modes available:

| Mode | When | OpenAugi implementation |
|------|------|------------------------|
| **Semantic search** | Fuzzy conceptual matching | sqlite-vec KNN on block embeddings |
| **Keyword search** | Known terms | FTS5 full-text search |
| **Graph traversal** | Discovering what you didn't know to search for | `get_related` / `traverse` following links |
| **Time-based** | Recent activity, temporal patterns | `recent` blocks by timestamp |
| **Direct lookup** | Fetch specific block by ID | `get_block` / `get_blocks` |

The [`get_context`](MCP_SERVER.md) tool combines these: semantic + keyword with 3x overfetch, deduplication via cosine grouping, MMR re-ranking for diversity, then link expansion. The [`get_index`](MCP_SERVER.md) tool returns the compiled navigational map — the agent's starting point.

The context blocks route between these modes automatically. Entity mentions signal graph traversal. Topic labels signal semantic search. Temporal ranges signal time-based retrieval. The context block is a routing table for retrieval.

## How This Maps to the Four-Layer Architecture

OpenAugi's implementation maps to a [general architecture for context engineering](https://bitsofchris.com/p/context-engineering-is-index-design):

| Concept | OpenAugi implementation |
|---------|------------------------|
| **Data blocks** | `Block(kind=document\|entry\|tag)` — raw content with metadata |
| **Context blocks** | `Block(kind=context)` — compiled hub summaries, concepts, index |
| **Context graph** | `links` table — typed edges the agent traverses |
| **Context snapshots** | Incremental compile — content-hash based change detection |

The key insight from building this on 5,000+ Obsidian notes: the agents are smart enough. We just need to give them a better data representation. Standard RAG missed the structure. Full GraphRAG was too expensive and too granular. The middle path — lightweight navigational metadata over a link graph — is what works.

## Hub Scoring

Hubs are the most-connected nodes in the graph. They're computed at query time via pure SQL aggregation (no stored table):

```
hub_score = w_in * ln(1 + inbound_links) + w_out * ln(1 + outbound_links) + w_ent * ln(1 + entry_count)
```

Hub scores surface the tags and topics that are most central to the knowledge graph — natural starting points for agent navigation.

## Design Decisions

- **SQLite over DuckDB**: WAL mode for concurrent reads (MCP) and writes (watcher). DuckDB is single-writer.
- **sqlite-vec over FAISS**: KNN via `vec0` virtual table — everything in one file. Embeddings normalized on write so L2 distance = cosine.
- **Content hash as identity**: Stable across section reordering. Incremental ingestion skips unchanged files.
- **Tags as blocks**: First-class graph nodes. Hub scoring, traversal, entity resolution work uniformly.
- **Default local embeddings**: sentence-transformers, no API key required. Upgrade via config.
- **`get_context` dedup + MMR**: Over-fetches 3x, collapses near-duplicates, re-ranks for diversity. See [MCP_SERVER.md](MCP_SERVER.md) for tuning.

## Further Reading

- [Context Engineering is Index Design](https://bitsofchris.com/p/context-engineering-is-index-design) — the philosophy behind this architecture
- [ARCHITECTURE.md](../ARCHITECTURE.md) — system architecture, module map, key flows
- [MCP_SERVER.md](MCP_SERVER.md) — tool reference and query tuning
- [enrichment.md](enrichment.md) — tag taxonomy and document classification
