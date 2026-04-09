---
name: Capture → Classify → Render Loop
description: Deterministic-first, LLM-surgical pipeline. Workstreams (not facets) as minimum taxonomy. Concepts emerge from graph/embeddings. Rendered OpenAugi/ directory is the interface. Karpathy-style — LLM is the product, code is plumbing.
status: draft
created: 2026-04-08
updated: 2026-04-08
---

# Capture → Classify → Render Loop

## What changed from the first draft

This plan was initially "wire LLM tag inference into the pipeline." After reading [from-capture-to-jarvis.md](from-capture-to-jarvis.md), the framing sharpens significantly:

- **Links and taxonomy are separate things.** Links are the idea graph — "this relates to that." They don't imply classification. Taxonomy is where something lives. Don't mix them.
- **Workstreams, not facets.** The minimum taxonomy is 5-10 top-level workstreams, user-confirmed once. The existing `area/type/status/topic` facet system in [taxonomy.py](../../src/openaugi/pipeline/taxonomy.py) is over-engineered for what's actually needed.
- **Concepts are emergent, not assigned.** The LLM doesn't classify things *as* concepts. Concepts come from hub detection + embedding clusters. The LLM just names and summarizes what the graph already reveals.
- **Deterministic-first, LLM surgical.** If a block is in `journals/openaugi/`, workstream = openaugi. No LLM call needed. Only ambiguous blocks go to the LLM. This is the Karpathy angle.
- **Rendered vault IS the interface.** The tool becomes valuable when you open `OpenAugi/Streams/openaugi.md` in Obsidian and see what you've been thinking about. Not context blocks in SQLite. The vault itself, rendered.
- **Raw notes never modified.** Hard rule. Everything the system writes goes in `OpenAugi/`.

## Core principles

### 1. Graph structure ≠ taxonomy

- **Links** (`[[wiki]]`) → edges in the idea graph. Connections, not classification.
- **Workstreams** → the coarsest "where does this live?" — a short user-confirmed list.
- **Concepts** → emergent summaries of hubs and clusters. The system discovers them, you confirm or rename.
- **Raw notes** → append-only, untouched. Everything generated goes in `OpenAugi/`.

### 2. Deterministic-first, LLM surgical (the Karpathy angle)

Most classification should be free:

- Block is in `journals/openaugi/` → workstream = openaugi (path lookup)
- Block has `- [ ]` → type = task (regex)
- Block has `[[link]]` → graph edge (regex)
- Block has date header → type = journal (regex)
- Block has `#openaugi/feature` → classification hint (regex)

LLM is called **only** for:

- **Ambiguous blocks** (daily notes, unscoped files) → classify against the known workstream list
- **Hub/cluster summarization** → name and describe what the graph already shows
- **`zzz:` instructions** → per-block inline LLM actions the user requested

The LLM is a sharp knife, not a hammer. Prompts do the real work. The code is plumbing around a small number of well-aimed LLM calls.

**The Karpathy test.** If you removed all the pipeline code and just had:
1. A prompt that classifies a block against a workstream list
2. A prompt that summarizes a hub/cluster into a concept page
3. A prompt that executes `zzz:` instructions

...could you run the whole system as a series of shell scripts? Yes. That's the target shape. Every time you're tempted to add complexity, ask: "could a prompt do this instead?"

### 3. Blocks are thought-sized, not header-sized

The current [vault adapter](../../src/openaugi/adapters/vault.py) splits on `###` date headers. A single day's journal has 5-15 different thoughts inside one H3 section. Split finer:

- **Primary delimiter:** `###` headers (keep existing behavior)
- **Secondary delimiter:** `qqq` markers within a section (new)
- **Result:** each block is one thought — the unit of classification, embedding, and rendering

### 4. The rendered vault IS the interface

The tool becomes valuable when you can open Obsidian and see:

```
OpenAugi/
  Taxonomy.md              # workstreams + concept index
  Streams/
    openaugi.md            # all openaugi blocks, chronological, grouped by concept
    work.md
    content.md
    self.md
  Concepts/
    retrieval-engineering.md   # hub summary + related blocks
    append-only-capture.md
```

Not a web UI. Not a clippy agent. The vault itself, through Obsidian. This is the MVP UX.

## The updated pipeline

```
CAPTURE
  Vault adapter
    → split on ### headers (existing)
    → split on qqq within sections (NEW)
    → extract zzz: instructions to metadata (NEW, optional)
    → blocks in SQLite

CLASSIFY (the missing link)
  For each new block:
    deterministic (covers ~80% of blocks):
      source_path → workstream lookup
      regex → type (task / journal / reference)
      regex → links and tags
    LLM fallback (only ambiguous blocks):
      classify against workstream list (narrow, batched)
      execute zzz: instructions if present

DISCOVER (periodic, not per-ingest)
  Hub detection on links
  Cluster detection on embeddings
  LLM names and summarizes each hub/cluster → concept page

RENDER (materialize the interface)
  Write OpenAugi/Taxonomy.md
  Write OpenAugi/Streams/<workstream>.md
  Write OpenAugi/Concepts/<concept>.md
  Raw notes: never modified
```

## What already exists vs. what's needed

| Piece | Status |
|---|---|
| [vault.py](../../src/openaugi/adapters/vault.py) `###` splitting | Works |
| [runner.py](../../src/openaugi/pipeline/runner.py) incremental ingest | Works |
| Context block compile ([compile.py](../../src/openaugi/pipeline/compile.py)) | Shipped — reframe as renderer |
| `qqq` secondary splitting in vault adapter | **NEW** |
| `zzz:` instruction extraction | **NEW** |
| `~/.openaugi/workstreams.json` + `openaugi init` | **NEW** |
| `classify.py` — deterministic source path → workstream | **NEW** |
| Narrow LLM fallback for ambiguous blocks | **REFACTOR** from [tag_inference.py](../../src/openaugi/pipeline/tag_inference.py) — strip the facet system, classify against a 5-10 item list |
| Hub/cluster concept discovery | **NEW** (builds on existing hub scoring) |
| Renderer: write `OpenAugi/Streams/*.md` and `OpenAugi/Concepts/*.md` | **NEW** |
| [taxonomy.py](../../src/openaugi/pipeline/taxonomy.py) facet discovery | **KEEP for reference, don't use in main path** |
| [enrich.py](../../src/openaugi/pipeline/enrich.py) agent flow | **KEEP as one-off tool** |

## What to build now — revised minimum useful version

### Step 1 — `openaugi init` and workstreams.json

One-time interactive: read vault folder structure + top-level MOC note titles, propose 5-10 workstreams, let user prune. Write to `~/.openaugi/workstreams.json`:

```json
{
  "workstreams": [
    {"slug": "openaugi", "name": "OpenAugi", "paths": ["journals/openaugi/", "OpenAugi/"]},
    {"slug": "work",     "name": "Work",     "paths": ["journals/work/", "Work/"]},
    {"slug": "content",  "name": "Content",  "paths": ["Content/"]},
    {"slug": "self",     "name": "Self",     "paths": ["journals/daily/"]}
  ]
}
```

This covers most blocks deterministically. Nothing else works without this.

### Step 2 — Deterministic classifier

New module `src/openaugi/pipeline/classify.py`:

```python
def classify_block(block: Block, workstreams: list[Workstream]) -> str | None:
    """Return workstream slug if source path matches, else None."""
    for ws in workstreams:
        if any(block.metadata.get("source_path", "").startswith(p) for p in ws.paths):
            return ws.slug
    return None
```

Runs after ingest. Writes result to `metadata["workstream"]`. No LLM call. Covers ~80% of blocks for free.

### Step 3 — `qqq` secondary splitting in vault adapter

Extend `_split_by_h3_dates` in [vault.py](../../src/openaugi/adapters/vault.py) to also split on `qqq` lines within a section. Each fragment becomes its own block with its own content hash. Preserves the existing H3-date grouping as parent metadata.

### Step 4 — Renderer: `OpenAugi/Streams/<workstream>.md`

Simplest possible renderer. After classification, write one markdown file per workstream, blocks ordered by date:

```markdown
# OpenAugi Stream

> System-generated. Do not edit — edits will be overwritten.
> Raw source: your original notes (untouched).

## 2026-04-08
- [block content preview]  — [[source note]]
- [block content preview]  — [[source note]]

## 2026-04-07
- [block content preview]  — [[source note]]
```

This is the MVP UX. Open it in Obsidian, see your workstream. That's the whole value prop for now.

### Step 5 — Narrow LLM fallback for ambiguous blocks

Refactor [tag_inference.py](../../src/openaugi/pipeline/tag_inference.py) into a narrow prompt:

```
Classify this block against these known workstreams: [openaugi, work, content, self].
Return exactly one slug, or "unscoped" if truly ambiguous.
[block content]
```

Batch 20 blocks per call. Only runs on blocks where deterministic classification returned None. Defaults off until the deterministic path is proven and you can measure what's actually left over.

### Defer to Phase 2 of this plan

- `zzz:` instruction execution (parsing can land in Step 3; execution deferred)
- Hub/cluster concept discovery + rendered concept pages
- `Taxonomy.md` index file
- File watcher trigger (CLI is fine for now)
- Tag review UI
- Self-improving taxonomy loop

## What NOT to build (sharpened)

- **No facet taxonomy.** The existing `area/type/status/topic` facet system is richer than needed. Workstreams only in the main path. Type can stay deterministic (regex).
- **No merge rules or discovery.** Workstream list is user-confirmed once in `openaugi init`, not inferred from tag usage.
- **No UI.** The vault IS the UI. Obsidian renders `OpenAugi/` as browseable markdown.
- **No janitor loop yet.** Run classification per-ingest, concept discovery periodically, by hand. No self-improving taxonomy until the basic loop works and is being used.
- **No LLM for non-ambiguous blocks.** If the source path deterministically maps to a workstream, that's enough. Don't second-guess it with an LLM call.
- **Don't modify user's raw notes.** Everything system-generated goes in `OpenAugi/`.

## How this relates to the rest of the roadmap

- **[from-capture-to-jarvis.md](from-capture-to-jarvis.md) Layer 1A** — this plan IS Layer 1A, concrete.
- **Layer 1B (workstream classification)** — Steps 1, 2, 5 above.
- **Layer 1C (concept discovery)** — deferred to Phase 2 of this plan.
- **Layer 1D (rendered view)** — Step 4 above is the MVP.
- **Phase 2 compile** (shipped, [compile.py](../../src/openaugi/pipeline/compile.py)) — reframe. Compile becomes the renderer. Context blocks are less important than `OpenAugi/Streams/*.md` and `OpenAugi/Concepts/*.md` that you can browse in Obsidian. The renderer consumes the same classified blocks.
- **[phase3-adapters.md](phase3-adapters.md)** — new adapters drop in with zero changes to classify/render. They produce blocks, same path downstream.

## Open questions

- **Block-level vs doc-level classification.** Daily notes: each block is likely a different workstream (classify per block). Focused notes (task files, workstream notes): all blocks share the workstream (classify per doc, trickle down). Heuristic: if source path deterministically maps to a workstream, use it for all blocks in that doc; else classify per block with LLM fallback.
- **When to render.** After every ingest? On demand? Start with manual `openaugi render`, automate later.
- **Unscoped blocks.** Daily notes cover multiple workstreams by nature. Start with: classify per block via LLM fallback, land unclassified ones in `OpenAugi/Streams/unscoped.md` for periodic review.
- **Concept discovery frequency.** Weekly? After N new blocks? Start with manual `openaugi discover`, automate later.
- **Legacy code.** The existing [taxonomy.py](../../src/openaugi/pipeline/taxonomy.py) and [tag_inference.py](../../src/openaugi/pipeline/tag_inference.py) facet system isn't in the main path anymore. Keep the code in place, but mark as legacy / one-off tooling via `openaugi enrich`.
- **`qqq` vs `###` as parent.** When a block is split by `qqq` within an H3 section, it should inherit the date from the H3 parent. Need to preserve this in block metadata for chronological rendering.

## Success criteria

1. `openaugi init` produces a workstream list you're happy with after 2 minutes of editing.
2. `openaugi up` assigns every new block a workstream — deterministic for most, LLM fallback for the rest.
3. `openaugi render` produces `OpenAugi/Streams/openaugi.md` that you can open in Obsidian and actually find useful.
4. **The real test:** you find yourself opening `OpenAugi/Streams/openaugi.md` in Obsidian to see what you've been thinking about this week — and it's accurate, organized, and surfaces things you'd forgotten.
5. Total LLM cost per daily ingest stays under $0.10 (Haiku, batched, only ambiguous blocks).

## Next action

In order, smallest end-to-end proof first:

1. **`openaugi init`** CLI that proposes workstreams from folder/MOC structure and writes `~/.openaugi/workstreams.json`.
2. **`classify.py`** with deterministic source_path → workstream lookup, wired into [runner.py](../../src/openaugi/pipeline/runner.py) post-ingest.
3. **Minimal renderer** — one markdown file per workstream, blocks in date order, written to `OpenAugi/Streams/`.

Stop there and *use it*. Open `OpenAugi/Streams/openaugi.md` in Obsidian for a week. See what's missing. Only then add `qqq` splitting, the LLM fallback, and concept discovery — based on what actually feels broken when you're using it, not based on the plan.
