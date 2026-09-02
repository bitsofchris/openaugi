---
name: query-provenance-and-dates
description: Fixes surfaced by the 2026-09-01 high-note analysis run (34 sessions, 8 extraction agents, 8 lens agents over the MCP surface). Four small changes shipping now (get_context filters, block provenance, creation-time dates, reference exclusion by default) and three larger designs for Chris to decide on (derived tables from lens schemas, an activity tape, incremental extraction).
---

# Query provenance and dates

**Status: IN PROGRESS 2026-09-02.** Branch `feat/query-provenance-dates`.

## Where this came from

On 2026-09-01 Claude ran a multi-agent analysis over every "high note" in the
vault: extract structured proposals from 47 notes, map the vault evidence in the
10 days around each of 34 sessions, load it into SQLite, then run eight lenses.
The run worked but exposed how the MCP surface fails an agent doing real
analysis. This plan is the triage of that run. Chris's reactions are recorded
inline so the design does not drift from what he asked for.

## Triage

| # | Finding from the run | Chris's call | Disposition |
|---|---|---|---|
| 1 | Agents fell back to filename grep and MOC scraping for "what was he doing between two dates." `search` has `after`/`before`; `get_context`, the tool the docs call primary, has no time or path filters at all. | "We do have time window queries. Why did you not see them?" | **Ship now.** Add `after`, `before`, `tags`, `exclude_path_prefix`, `include_path_prefix`, `provenance` to `get_context`. The gap was real but in the wrong tool. |
| 2 | Forty AI-written Jung sessions in `_private/0-Inbox/` came back from search indistinguishable from Chris's own writing and were quoted back to him as "your vault." | "I do have a tag for AI generated, and the whole OpenAugi folder is AI generated. We need to encode that." | **Ship now.** A derived `provenance` field on every data block (`human`, `ai`, `reference`), set at ingest from path rules and tags, filterable in every read tool, backfillable. |
| 3 | Block dates inherit file mtime when nothing else is dated. Dated bullets inside MOCs carry the file's date. | "Block date should inherit the date from the note created." | **Ship now** for creation time: use `st_birthtime` where the OS has it, mtime as fallback. **Ask** before changing bullet-level date inheritance (it changes splitting semantics). |
| 4 | Twenty near-identical Hollis quotes from Snipd came back in one search. | "As part of search you should be ignoring the reference folder." | **Ship now.** `[retrieval] exclude_provenance` config, default `["reference"]`, applied to `get_context` and semantic search unless the caller names a provenance explicitly. Browse mode already groups reference docs; unchanged. |
| 5 | The run had to create three ad-hoc SQLite tables by hand because the records store is untyped JSON with equality-only filters. | "I like lenses being able to declare a schema and emit normalized rows. A really cool way to do data analysis on myself." | **Design below, needs a decision.** |
| 6 | "Shipped" and "working on" were inferred by hand from daily notes. Session cards, git history, task files, and the Augi Log exist but nothing rolls them into a daily activity record. | "Now that you have my code sessions and repository history, look at those." | **Design below, depends on 5.** |
| 7 | Extraction cost eight agents and ~1.7M tokens for 2,240 rows, and hit the session limit. | "Not sure on incremental extraction, but we could be better there." | **Deferred.** Becomes a lens with an `emits` schema once 5 exists. |

## Shipping now

### 1. `get_context` filters

`engine.context()` gains the same filter fields as `QuerySpec` (`after`,
`before`, `tags`, `exclude_path_prefix`, `include_path_prefix`, `provenance`).
Filters apply to the candidate pool before rerank. When any filter is set the
overfetch multiplier doubles so a tight window does not starve the pool. The
MCP tool passes them through. Wire format of the response is unchanged, so the
golden corpus does not need regenerating.

### 2. Provenance

A derived field, `metadata["provenance"]`, one of:

- `human`: the user wrote it
- `ai`: a model wrote it, whether inside OpenAugi's own folders or pasted in by the user
- `reference`: someone else wrote it and it was imported (Readwise, Snipd, web clips, gdrive)

Resolution order, first match wins, mirrors `[vault.source_rules]`:

1. An explicit `provenance/<value>` tag in the note text. Text is truth.
2. `[vault.provenance_rules]` path globs in config, in file order.
3. Tag rules: `note-type/ai-summary`, `note-type/ai-response`, `source/ai-chat` give `ai`; any other `source/*` tag except `source/capture` gives `reference`.
4. Default `human`.

Chris's config will carry:

```toml
[vault.provenance_rules]
"OpenAugi/Capture/**" = "human"
"OpenAugi/**" = "ai"
"_private/2-Reference/**" = "reference"
"_sources/**" = "reference"
```

The pasted-in Jung sessions in `_private/0-Inbox/` need either a `provenance/ai`
tag or a rule. A rule cannot see inside the file, so this plan adds a one-shot
`openaugi provenance --backfill` that also reports blocks whose title matches a
configurable pattern (`[vault.provenance_title_patterns]`, e.g. `" - Jung - "`)
as candidates. It never guesses from content.

Every read tool that takes filters accepts `provenance: list[str]`. Block
summaries do not yet carry the field (that changes the wire format; a follow-up
once the golden corpus is regenerated deliberately).

### 3. Creation time

`_get_file_created_time` prefers `st_birthtime` and falls back to `st_mtime`.
Filename and heading dates still win, as today. A git checkout resets birthtime,
so on a freshly cloned vault this is no worse than before.

### 4. Reference exclusion by default

```toml
[retrieval]
exclude_provenance = ["reference"]
```

Applied in `get_context` and in semantic `search` when the caller does not pass
`provenance`. Passing `provenance=["reference"]` or `provenance=["human","reference"]`
opts back in. Keyword and browse modes are unchanged: keyword is precise by
construction and browse already collapses reference material into
`reference_documents`.

## Designs that need a decision

### 5. Derived tables from lens schemas

The high-note run wanted exactly this: an agent extracts rows from notes, rows
land somewhere queryable with SQL, later agents and lenses read them back.

Proposal: a lens may declare an `emits` block in its frontmatter:

```yaml
emits:
  table: vocational_pills
  key: pill_id
  columns:
    pill_id: text
    session_id: text
    note_date: date
    category: text
    inflation: int
    concreteness: int
    proposal: text
```

`openaugi tables sync` reads every lens with `emits` and creates or migrates a
real SQLite table `derived_<table>` (additive migrations only). A new write
tool, `write_rows(table, rows)`, validates rows against the declared columns and
upserts by key. Reads stay SQL: `openaugi query --sql` for humans, and a
read-only `run_sql(sql)` MCP tool restricted to `derived_*` tables for agents.
Tables are machinery, like records: droppable and regenerable, never the place a
human's words live.

What this deliberately does not do: grow the records store a query language.
Records stay untyped workflow state. Derived tables are typed analysis output.

Open questions for Chris: (a) is `run_sql` acceptable on the MCP surface, given
the records boundary in `docs/reference/records.md`? (b) should a lens with
`emits` be allowed to also write a note, or is the table its only artifact?

### 6. Activity tape

A derived table (`activity`, one row per day per domain) fed by four
deterministic sources already on disk: git log across every repo in
`OpenAugi/AGENT/Repos.md`, session cards from `scripts/session_cards.py`, task
files in `OpenAugi/Tasks/`, and vault blocks by day and area tag. No LLM. This
is the "tape" every lens wanted and none had. Builds on 5.

### 7. Incremental extraction

Once 5 exists, the pill extractor becomes a lens with `emits`, triggered on new
notes under a scope, run on a cheap model. Not before.

## Tests

- `test_query_engine.py`: context filters (date, path, tags, provenance), overfetch doubling.
- `test_provenance.py`: resolution order, explicit tag wins, config rules, tag rules, default, backfill idempotent, title-pattern candidates reported not applied.
- `test_vault_adapter.py`: birthtime preferred, mtime fallback.
- `test_mcp.py`: `get_context` accepts the new params; `search` `provenance` filter; default reference exclusion and opt-in.
- Golden corpus unchanged (assert by running it).

## Left off / next

**2026-09-02.** Items 1 through 4 are on the branch as four commits: creation
time, ingest-side provenance plus `backfill-provenance`, the query-side
provenance filter and `get_context` filters, and a normalizer fix. Suite
green (722), pyright clean, golden corpus untouched. Chris's config carries
`[vault.provenance_rules]` and `provenance_title_patterns`; the live DB was
backed up to `backups/openaugi-2026-09-02-pre-provenance.db` and backfilled:
15,707 human, 5,659 ai, 5,530 reference. 618 human-labelled blocks match a
title pattern and were reported, not relabelled; the `"Claude"` pattern is
noisy (it catches posts *about* Claude) and can be dropped from config.

**Next, in order:**
1. Chris merges the branch and restarts `openaugi serve`; until then the live
   server ignores the stamped field and has no `provenance` parameter.
2. Chris tags the pasted-in AI reflections (the ` - Jung - ` and
   ` - distilled - ` candidates) with `#provenance/ai`, or adds a folder for
   them and a rule.
3. Decide item 5 (derived tables from lens schemas) — the two open questions
   above. Item 6 (activity tape) and item 7 (incremental extraction) wait on it.
4. Decide whether dated bullets inside MOCs should set block dates (item 3,
   second half).
5. Follow-up: put `provenance` in the block summary wire row and regenerate
   the golden corpus deliberately.
