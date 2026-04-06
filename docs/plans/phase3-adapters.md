---
name: phase3-adapters
description: Plan for multi-source ingest — adapter protocol, ChatGPT/Readwise/Research adapters, LlamaIndex bridge, config-driven pipeline
---

# Phase 3: Adapt — Multi-Source Ingest

*Created: 2026-04-05 | Updated: 2026-04-05*

---

## What Is Phase 3?

Your knowledge isn't just in Obsidian. Phase 3 makes OpenAugi ingest from multiple sources — ChatGPT exports, Readwise highlights, AI research output — through a clean adapter protocol. Each adapter's job: parse a source into blocks + links. The existing pipeline handles the rest.

The vault adapter stays as-is. New adapters plug in alongside it.

---

## Decisions (Locked)

| Decision | Choice | Rationale |
|---|---|---|
| Adapter contract | Protocol in `protocols.py` + shared helpers module | Loose coupling. Vault stays functional. New adapters can be classes. Runner dispatches by protocol. |
| Block identity key | `metadata["source_id"]` (replaces vault's `source_path` as canonical) | Generic across file-based and API-based sources. Vault uses relative path, ChatGPT uses conversation ID, Readwise uses highlight ID. |
| Adapters to build | ChatGPT export, Readwise API, Research output, LlamaIndex bridge | These are Chris's actual data sources. Skip speculative adapters. |
| LlamaIndex | Optional bridge adapter, not a core dependency | `pip install openaugi[llamaindex]` pulls it in. Core stays lean. Bridge translates LlamaIndex `Document` → our blocks. |
| Config model | `[sources.*]` sections in `config.toml`, setup via `openaugi init` | Config-driven ingest from the start. `openaugi ingest` runs all enabled sources. |
| CLI | `openaugi ingest` (all sources) / `openaugi ingest --source <name>` (one) | Abstract away source selection. Config is the source of truth. |
| Per-adapter config | Sensible defaults + overridable options (pulled from Phase 7) | Ship with good defaults. Users can tune extraction depth, tagging rules per adapter. |
| Cross-source dedup | Deferred to compile (Phase 2/4) | Ingest stores everything. Compile handles semantic dedup across sources — that's where LLM + embeddings live. |
| Feedback loop | `source` field on blocks filters compiled/agent content from ingest inputs | Same pattern as Phase 2's `source=compiled`. Each adapter sets its own `source` string. |

---

## Architecture

### The Pipeline

```
Source (files, API, export JSON)
  → Adapter (source-specific parsing)
  → (list[Block], list[Link])
  → Runner (generic: dedup, diff, insert, embed)
  → SQLiteStore
```

Each adapter is a black box that returns blocks and links. The runner doesn't know or care where they came from.

### Adapter Protocol

```python
# model/protocols.py

class IngestResult(NamedTuple):
    blocks: list[Block]
    links: list[Link]
    current_ids: dict[str, str]   # {source_id: content_hash} for incremental
    deleted_ids: list[str]        # source_ids no longer present

class SourceAdapter(Protocol):
    """Contract for all ingest adapters."""

    source_name: str  # "vault", "chatgpt", "readwise", etc.

    def ingest_incremental(
        self,
        config: dict[str, Any],
        known_hashes: dict[str, str],
    ) -> IngestResult: ...
```

**Key:** `known_hashes` is `{source_id: content_hash}`. The runner fetches this per-source from the store. Each adapter returns updated hashes and deletions.

### Runner Refactor

`run_layer0` becomes source-agnostic:

```python
def run_ingest(store: SQLiteStore, config: dict) -> dict:
    """Run all configured source adapters."""
    results = {}
    for source_name, source_config in config.get("sources", {}).items():
        adapter = get_adapter(source_config["type"])
        known = _get_known_hashes(store, source=source_name)
        result = adapter.ingest_incremental(source_config, known)
        _apply_result(store, result, source_name)
        results[source_name] = result
    return results
```

The vault adapter gets wrapped to conform to the protocol. Its internals don't change.

### Shared Helpers

`adapters/helpers.py` — reusable across adapters:

- `extract_tags(text) → list[str]` — `#tag` and `[[wikilink]]` extraction (already in vault adapter)
- `extract_wikilinks(text) → list[str]` — `[[link]]` extraction
- `make_entry_blocks(sections, source, source_id_prefix) → (list[Block], list[Link])` — common pattern of splitting content into entries
- `markdown_to_blocks(md_text, source, source_id) → (list[Block], list[Link])` — parse markdown into document + entry blocks. This is the "markdown is the interchange format" idea — adapters that produce markdown can use this directly.

---

## Config Model

```toml
# ~/.openaugi/config.toml

[sources.vault]
type = "vault"
path = "~/Documents/Obsidian"
exclude_patterns = [".obsidian/**", ".git/**", ".trash/**", "templates/**"]

[sources.chatgpt]
type = "chatgpt"
path = "~/exports/chatgpt"          # folder of export ZIPs or unpacked JSON
include_user_messages = true         # default: true
include_assistant_messages = true    # default: true
min_turns = 2                        # skip single-turn throwaway chats

[sources.readwise]
type = "readwise"
api_key_env = "READWISE_API_KEY"    # env var name (loaded from .env)
sync_since = "2024-01-01"           # optional: only sync highlights after this date

[sources.research]
type = "research"
path = "~/research"                  # folder of saved AI research outputs
extract_citations = true             # parse and link cited papers/URLs

[sources.llamaindex]
type = "llamaindex"
reader = "SimpleDirectoryReader"     # any LlamaIndex reader class
reader_args = { input_dir = "~/documents" }
```

`openaugi init` adds `[sources.*]` sections interactively. Adapter-specific options have sensible defaults — users only override what they care about.

---

## Adapter Designs

### 1. ChatGPT Export Adapter

**Source:** ChatGPT data export — folder of `conversations-NNN.json` files (from "Export your data" in ChatGPT settings). Each JSON file is a list of ~100 conversation objects.

**Real export structure (verified):**
- Conversation keys: `id`, `title`, `create_time`, `update_time`, `default_model_slug`, `mapping`, `is_archived`, `is_starred`
- `mapping` is a dict of `{node_id: {message, parent, children}}` — a tree, not a flat list (supports branching/editing)
- Message keys: `id` (UUID), `author.role`, `content.content_type`, `content.parts[]`, `create_time`, `metadata.model_slug`
- Content types: `text` (primary), `multimodal_text` (has image parts — extract text only), `code`, `execution_output`, `tether_quote`, `tether_browsing_display`, `system_error`
- Roles: `user`, `assistant`, `system`, `tool`
- Per-conversation folders (UUID-named) contain `audio/` and image files — **ignore by default**

**Parsing:**
- One **document block** per conversation (`source_id` = conversation UUID)
- One **entry block** per message with `content_type` in `{text, multimodal_text}` (`source_id` = message UUID)
- Filter: only `user` and `assistant` roles by default. Skip `system` and `tool` messages.
- For `multimodal_text`, extract only string parts from `content.parts[]`, skip image/file dicts
- Walk the mapping tree from root to leaves to get message ordering (don't rely on dict order)
- **Tag blocks** from conversation title keywords and any `#tags` in messages
- **Links:** `split_from` (message → conversation), `tagged` (message → tags), `follows` (sequential messages within a conversation thread)

**Metadata per entry:**
- `role`: user | assistant
- `model`: gpt-4, gpt-4o, etc. (from `metadata.model_slug`)
- `conversation_title`: parent conversation title
- `content_type`: original content_type from export
- `source_id`: message UUID from export

**Incremental:** Hash all conversations in a JSON file → file-level hash. Within changed files, hash per conversation UUID. If conversation hash unchanged, skip.

**Config options:**
- `include_user_messages` (default: true) — user messages are blocks
- `include_assistant_messages` (default: true) — assistant messages are blocks
- `min_turns` (default: 2) — skip trivially short conversations
- `content_types` (default: `["text", "multimodal_text"]`) — which content types to ingest
- `exclude_models` — skip conversations with specific models

### 2. Readwise Adapter

**Source:** Readwise API v2 (`/highlights/`, `/books/`)

**Real API structure (verified):**
- Highlights: `id`, `text`, `note` (user annotation), `location`, `highlighted_at`, `book_id`, `tags[]`, `url`, `external_id`
- Books: `id`, `title`, `author`, `category` (articles|books|podcasts|tweets), `source` (reader|kindle|etc), `source_url`, `num_highlights`, `tags[]`, `document_note`
- Pagination: `count`, `next`, `previous`, `results[]`
- Incremental: `updated` field on both — use `?updated__gt=` param
- Chris has 2,581 highlights across 235 sources

**Ingest modes** (configurable via `mode`):

| Mode | What it ingests | Use case |
|---|---|---|
| `highlights` (default) | Highlights as entry blocks, source metadata on document block only | Lightweight — just your annotations and notes |
| `full` | Highlights + fetches full article/document text via Readwise Reader API | Complete source text alongside highlights. Document block gets full content. |

In `highlights` mode, the document block has title/author/URL metadata but no `content` — it's a container for highlight entries. In `full` mode, the document block has the full source text as `content`, which gets split into sections like vault entries.

**Parsing (both modes):**
- One **document block** per book/article (`source_id` = Readwise book ID as string)
- One **entry block** per highlight (`source_id` = Readwise highlight ID as string)
- If highlight has a `note`, store as `content` with `metadata.highlight_text` for the passage. If no note, the highlight text is the `content`.
- **Tag blocks** from Readwise tags + book category
- **Links:** `split_from` (highlight → source), `tagged` (highlight → tags)

**Metadata per document block:**
- `title`, `author`, `category`, `source` (reader|kindle|etc)
- `source_url`: original article/book URL
- `num_highlights`: count
- `readwise_url`: link back to Readwise

**Metadata per entry block:**
- `highlight_text`: the highlighted passage
- `note`: user's annotation (if any)
- `location`: position in source
- `highlighted_at`: ISO timestamp
- `source_title`: parent book/article title
- `source_author`: author
- `readwise_url`: link back to highlight in Readwise

**Incremental:** Store last sync timestamp in `metadata["last_sync"]` on a sentinel block or in config state. On next ingest, fetch `?updated__gt={last_sync}`. For `full` mode, only re-fetch full text if the source's `updated` is newer.

**Config options:**
- `mode` (default: `"highlights"`) — `"highlights"` or `"full"`
- `sync_since` — initial sync start date (ISO, e.g. `"2024-01-01"`)
- `source_types` — filter by category (e.g., `["articles", "books"]`)
- `api_key_env` (default: `"READWISE_API_KEY"`) — env var name for API token

### 3. Research Output Adapter

**Source:** Saved AI research outputs (Claude, GPT, Perplexity deep research) as `.md` files in a folder

**Parsing:**
- One **document block** per research file (`source_id` = relative path)
- **Entry blocks** from H2/H3 sections within the research
- **Citation blocks** (kind=`reference`) extracted from markdown links, footnotes, and bibliography sections — each cited paper/URL becomes a block
- **Links:** `split_from` (section → document), `cites` link kind (entry → reference block), `tagged`

**Citation extraction:**
- Parse markdown links `[title](url)` — create reference blocks for academic URLs (arxiv, doi, scholar)
- Parse footnote-style citations
- Extract paper titles, authors, years from citation text where parseable
- Reference blocks get `source=research`, `metadata.url`, `metadata.cited_by`

**Metadata per entry:**
- `research_tool`: claude | chatgpt | perplexity (inferred from content patterns or folder structure)
- `topic`: inferred from filename or first heading

**Incremental:** Same as vault — file hash comparison.

**Config options:**
- `extract_citations` (default: true) — parse and link cited sources
- `section_depth` (default: 2) — H2 vs H3 splitting

### 4. LlamaIndex Bridge Adapter

**Source:** Any LlamaIndex reader's output

**How it works:**
- User configures a LlamaIndex reader class + args in config
- Bridge instantiates the reader, calls `load_data()` → `list[LlamaIndexDocument]`
- Translates each `LlamaIndexDocument` into our Block model:
  - `content` = doc text
  - `metadata` = LlamaIndex metadata dict merged into our metadata
  - `source_id` = LlamaIndex `doc_id` or hash
  - `source` = `"llamaindex:{reader_name}"`
- Runs `markdown_to_blocks()` on the content to split into entries
- Tags and links extracted by shared helpers

**Optional dependency:** `pip install openaugi[llamaindex]` — core never imports LlamaIndex.

**Config options:**
- `reader` — LlamaIndex reader class name
- `reader_args` — dict passed to reader constructor
- `reader_module` — optional, if reader isn't in default namespace

---

## `source_id` Migration

The vault adapter currently uses `metadata["source_path"]`. To unify:

1. Add `metadata["source_id"]` to vault adapter output (= relative path, same value as `source_path`)
2. Runner queries by `source_id` instead of `source_path`
3. Keep `source_path` on vault blocks for backwards compat (it's still useful metadata)
4. Store helper: `get_known_hashes(source: str) → dict[str, str]` filters by `Block.source` field

---

## CLI Changes

```bash
openaugi ingest                     # all configured sources
openaugi ingest --source vault      # just vault
openaugi ingest --source chatgpt    # just chatgpt
openaugi init                       # interactive: add/configure sources
```

`openaugi up` calls `run_ingest(store, config)` instead of `run_layer0(vault_path, store)`.

---

## Implementation Sequence

### Step 1: Adapter protocol + runner refactor
- [ ] Define `SourceAdapter` protocol and `IngestResult` in `model/protocols.py`
- [ ] Add `metadata["source_id"]` to vault adapter (= relative path)
- [ ] Create `adapters/helpers.py` with shared extraction utilities
- [ ] Wrap vault adapter as `VaultAdapter` class conforming to protocol
- [ ] Refactor `run_layer0` → `run_ingest` — dispatch by adapter, loop over configured sources
- [ ] Update `_get_known_doc_hashes` → `_get_known_hashes(store, source)` — filter by source
- [ ] Update CLI `ingest` command with `--source` flag
- [ ] Update `openaugi up` to use new runner
- [ ] Tests: vault ingest still works identically through new abstraction

### Step 2: ChatGPT adapter
- [ ] `adapters/chatgpt.py` — parse conversations.json → blocks + links
- [ ] Handle both ZIP and unpacked export formats
- [ ] Entry block per message, tagged by role
- [ ] Config options: `include_user_messages`, `min_turns`
- [ ] Unit tests with fixture ChatGPT export JSON
- [ ] Integration test: ingest → verify blocks in store with correct metadata and links

### Step 3: Readwise adapter
- [ ] `adapters/readwise.py` — Readwise API client → blocks + links
- [ ] Two ingest modes: `highlights` (default) and `full` (fetches source text via Reader API)
- [ ] Incremental via `updated__gt` timestamp tracking
- [ ] Document block per source, entry block per highlight
- [ ] Config options: `mode`, `sync_since`, `source_types`, `api_key_env`
- [ ] Unit tests with mocked API responses (synthetic fixture matching real schema)
- [ ] Integration test with fixture data

### Step 4: Research output adapter
- [ ] `adapters/research.py` — parse research markdown → blocks + links + citations
- [ ] Citation extraction: markdown links, footnotes, bibliography patterns
- [ ] Reference blocks (kind=`reference`) with `cites` links
- [ ] Config options: `extract_citations`, `section_depth`
- [ ] Unit tests with fixture research outputs
- [ ] Integration test: verify citation blocks and link graph

### Step 5: LlamaIndex bridge
- [ ] `adapters/llamaindex_bridge.py` — translate LlamaIndex Documents → blocks
- [ ] Optional import guard (graceful error if not installed)
- [ ] `pyproject.toml` optional dependency: `openaugi[llamaindex]`
- [ ] Unit test with mock LlamaIndex Documents
- [ ] Doc: how to use any LlamaIndex reader with OpenAugi

### Step 6: Config + init
- [ ] Add `[sources.*]` to config schema and `DEFAULT_CONFIG`
- [ ] `openaugi init` interactive source setup (extend existing init flow)
- [ ] Validate source configs on load (type exists, required fields present)
- [ ] Doc: multi-source configuration guide

### Step 7: Integration + docs
- [ ] End-to-end test: multiple sources → single store → verify blocks from all sources coexist
- [ ] Update ARCHITECTURE.md with adapter protocol and new module map
- [ ] Update phase2-compile.md if compile needs awareness of new source types
- [ ] Verify `openaugi up` works with multi-source config

---

## Testing Strategy

**Principle:** All committed test fixtures are synthetic — realistic structure matching the real schemas, but no real user data in the repo. Real data is used for local manual testing only.

### Fixture design
- **ChatGPT:** Synthetic `conversations.json` with 2-3 fake conversations covering: multi-turn text, multimodal_text with image parts to skip, short conversation (filtered by min_turns), branching message tree. Structure matches the real export format exactly.
- **Readwise:** Synthetic API response JSON matching real `/highlights/` and `/books/` schemas. Covers: highlight with note, highlight without note, multiple source types (article, book), pagination, tags.
- **Research:** Synthetic `.md` files with H2/H3 sections, markdown links to arxiv/doi URLs, footnote citations, bibliography section.

### Unit tests
- Each adapter: synthetic fixture input → verify correct blocks, links, metadata, source_id
- ChatGPT: conversation splitting, role tagging, min_turns filtering, multimodal_text text-only extraction, message tree walking
- Readwise: highlight → entry mapping, mode=highlights vs mode=full, incremental timestamp tracking, note vs no-note content handling
- Research: citation extraction from various markdown patterns
- LlamaIndex bridge: Document → Block translation
- Shared helpers: tag extraction, wikilink extraction, markdown splitting

### Integration tests
- Multi-source ingest into single store — no ID collisions, correct source filtering
- Incremental ingest per source — only changed data re-processed
- `openaugi ingest --source X` runs only that adapter
- Config validation — missing required fields, unknown adapter type
- Readwise API tests mock the HTTP layer (no real API calls in CI)

### Manual testing (local only, not committed)
- Ingest Chris's real ChatGPT export (`~/Downloads/chatgpt-history/`)
- Ingest Chris's Readwise highlights (API key from `keys.env`)
- Run compile after multi-source ingest — verify context blocks span sources
- Demo: `openaugi up` with vault + chatgpt configured → both ingested → search finds results from both

---

## New Link Kind: `cites`

The research adapter introduces a new link kind: `cites` (entry → reference block). Add to link kind validation.

---

## New Block Kind: `reference`

Citation/source blocks from research output. `Block(kind="reference")` with metadata: `url`, `title`, `authors`, `year`, `cited_by`.

---

## What This Does NOT Include

- **Cross-source semantic dedup** — that's compile/Phase 4 territory (LLM sees duplicates across sources and merges)
- **Real-time API sync** (Readwise webhook, etc.) — poll-based for now
- **Pipeline orchestration framework** — just a for-loop over adapters
- **UI for source management** — CLI + config.toml, UI is a future layer

---

## Phase 7 Updates

Per decision to pull basic adapter configurability into Phase 3, Phase 7's scope narrows to:
- Custom lens framework
- Community adapter registry/discovery
- `openaugi init` full wizard (model + adapters + lenses + enrichment in one flow)
- Pipeline orchestration (parallel adapters, dependency ordering) — if needed

The per-adapter config options (extraction depth, tagging rules) now ship with each adapter in Phase 3.
