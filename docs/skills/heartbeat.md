---

# Heartbeat Skill — Rules for Processing New Blocks

## Workstreams (classify every block against these)

Customize this list. Five to ten top-level workstreams works well. These are the coarsest "where does this live" partitions.

- **openaugi** — the tool, the project, the code, the roadmap
- **work** — day job, projects, meetings
- **content** — writing, posts, creator work, drafts
- **self** — journal, reflection, life, family, personal
- **reference** — memories, quotes, facts, things to keep findable

## Defaults (when there is no `zzz:` instruction on a block)

- If the block source path starts with `journals/openaugi/` or `OpenAugi/` → workstream = openaugi
- If the block source path starts with `journals/work/` or `Work/` → workstream = work
- If the block is from a daily note → classify by content
- Tags: use facets when you see clear signals — `type/idea`, `type/task`, `type/insight`, `type/question`
- Unsure? Tag it and move on. Don't block on ambiguity.

## Supported `zzz:` instructions

The user writes per-block instructions inline as `zzz: <text>` (case-insensitive — `ZZZ:`, `zZz:` all work). Match on **intent**, not exact wording. "Research this," "look into this more," and "find what I already have on this" all map to the same action.

Multiple `zzz:` lines in one block are separate instructions — handle each independently. Every instruction is always scoped to the containing block; never re-target to another block or to this skill file.

### `research`
Search the vault using `openaugi:search` and `openaugi:get_context`. Summarize what's already known about the topic. List open questions and what to read next. Write the summary to `OpenAugi/Research/<slug>.md`. Link the research note from the heartbeat log entry for this block.

**Always** create a task at `OpenAugi/Tasks/TASK-YYYY-MM-DD-<slug>.md` with `status: pending` that includes:
- A link to the research note
- What vault connections were found
- What web/external research still needs to happen (Perplexity, X, specific URLs, etc.)

Web research is always a follow-up step — the heartbeat does vault synthesis, the task captures what's left for you or a web-enabled agent to finish.

### `task`
Create a task entry at `OpenAugi/Tasks/TASK-YYYY-MM-DD-<slug>.md` with the block content as context. Link back to the source block. Set status = `active`. Do not execute the task — just capture it so the user can pick it up.

### `log`
Log the block in the heartbeat entry with the specified tags (e.g. `zzz: log, tag self/reflection`). No further action. Useful for reflections and personal notes you want captured but not acted on.

### `remember`
Promote the block to a reference note at `OpenAugi/Reference/<slug>.md`. Use for quotes, facts, and decisions the user wants easily findable later. Link back to the source block.

### anything else
Use best judgment based on the natural-language instruction. Log exactly what you did in the heartbeat entry for that block so the user can review and either accept or update this skill file.

## Task handling (when there is no explicit `zzz: task` instruction)

- If a block mentions a deadline → create a task entry
- If a block has a `- [ ]` checkbox → note it in the heartbeat log but don't auto-create a task
- Never modify the raw source note

## Research handling (when there is no explicit `zzz: research` instruction)

- Only run research when explicitly requested via `zzz: research` — don't speculatively search the vault for every block
- When you do research, always use `openaugi:search` and `get_context` before writing anything, so you don't duplicate existing notes

## What to write back

Write a heartbeat log to `OpenAugi/Heartbeat/YYYY-MM-DD.md` summarizing:

- Blocks processed (count + one-line summary each)
- Classifications made (workstream + tags per block)
- Connections found (links to related blocks from `openaugi:search`/`get_context`)
- Actions taken (tasks created, research written, reference notes promoted)
- `zzz:` instructions honored — list each one and what you did
- Anything flagged for human review (ambiguous workstream, unclear instruction, etc.)

Format each block as its own `###` section with:

```
### Block N — "<first ~60 chars of content>"
- **Source:** <source path>
- **Block ID:** <id>
- **Workstream:** <slug>
- **Tags:** <tag list>
- **User instructions:** <zzz: texts if any, else "(none)">
- **Actions:**
  - <what you did>
  - <links to created files>
```

## Hard rules

- **Always use the entry block ID, not the document block.** Each block you receive has its own `id` — use that ID when calling `tag_block`, logging, or referencing a block. Never substitute the parent document's block ID. The document block is a navigation node; the entry block is the actual content unit.
- **Never modify the user's raw notes.** Everything system-generated lives under `OpenAugi/` in the vault.
- **Use the openaugi MCP tools for lookups.** Don't try to search the vault via file reads when `openaugi:search` and `get_context` are available — they use the indexed graph and embeddings.
- **If uncertain, log and move on.** Don't block the whole heartbeat run on a single ambiguous block.
- **The heartbeat log is the audit trail.** The user reads it to see what you did and to decide whether to update this skill file.
