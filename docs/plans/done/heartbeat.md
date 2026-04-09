---
name: Heartbeat — Dumb Script, Smart Agent
description: Pivot from narrow API classification to agent sessions. A deterministic script splits/indexes new blocks, then spawns a Claude Code session with MCP tools and a skill file. The agent reasons through new blocks using openaugi MCP tools, honors per-block `zzz:` instructions, and writes a heartbeat log. One command — `openaugi heartbeat`.
status: draft
created: 2026-04-08
---

# Heartbeat — Dumb Script, Smart Agent

## The pivot

Earlier plans (capture-tag-stream-loop, from-capture-to-jarvis) assumed classification happens via narrow LLM API calls: "here's a block, return JSON." That's stateless. It can't look anything up. You have to pre-stuff all context into the prompt.

**Better architecture:** hand new blocks to a Claude Code agent session with MCP tools. The agent reads a skill file, uses `openaugi:search` / `get_context` / `traverse` to look up connections, honors per-block `zzz:` instructions from the user, and writes its work to a heartbeat log. One command: `openaugi heartbeat`.

Why this is better:

- **The agent chains decisions.** Block 1 finds a connection from March → that context informs how it processes block 2.
- **The agent can notice related blocks in a batch** and handle them together instead of classifying each in isolation.
- **The agent can go get more context** when it needs it (it has tools), instead of failing on ambiguity.
- **`zzz:` becomes natural inline instructions** — you write guidance in pen, the agent reads it as a per-block directive.
- **The skill file is just a markdown file** in your vault. No config system. No YAML. Edit it in Obsidian.
- **The openaugi MCP tools already exist.** The agent doesn't need new plumbing — it reuses what's already built.

## The architecture

```
openaugi heartbeat
  │
  ├── DUMB SCRIPT (deterministic, boring, fast)
  │     1. Run incremental ingest (existing)
  │     2. Find blocks added since last heartbeat (by timestamp or marker)
  │     3. Extract per-block metadata: links, tags, zzz: instructions
  │     4. Build a prompt listing those blocks + their zzz: instructions
  │     5. Spawn claude CLI with:
  │          - The prompt
  │          - Reference to the skill file (OpenAugi/heartbeat-skill.md)
  │          - MCP access to openaugi tools
  │          - Allowed tools: Read, Write, MCP
  │     6. Record the heartbeat timestamp for next run
  │
  └── SMART AGENT SESSION (one Claude Code run, reasoning + tools)
        - Reads OpenAugi/heartbeat-skill.md for rules + workstream list
        - For each new block:
            · Uses openaugi:get_context to find connections
            · Uses openaugi:search to check prior thinking
            · Reads the zzz: instruction if present; follows it
            · Classifies workstream, proposes tags
            · If zzz says "task" → writes to OpenAugi/Tasks/
            · If zzz says "research" → searches + summarizes
            · If no zzz → uses skill file defaults
        - Writes OpenAugi/Heartbeat/YYYY-MM-DD.md summarizing what it did
        - Session ends
```

The script doesn't need to know what the agent did. It only checks that the heartbeat log was written.

## The `zzz:` convention

Per-block agent instructions, written inline by the user in pen-mode. Stripped from the clean block content, captured as metadata.

```
Had a thought about matryoshka embeddings for multi-res matching
zzz research this more - find papers in my vault and summarize what I already know
qqq
Need to fix the README onboarding flow before Thursday
zzz task - do this in the openaugi repo
qqq
Feeling stuck on the direction of openaugi today
zzz just log this, tag personal/reflection
```

The script extracts `zzz:` lines and attaches them to the block metadata. The agent prompt surfaces them per-block:

> **Block 3** (from Journal/2026-04-08.md): "Feeling stuck on the direction of openaugi today"
> **User instruction:** "just log this, tag personal/reflection"

The agent follows the instruction. If there's no `zzz:`, the agent uses the skill file defaults.

## The skill file

Just a markdown file the user maintains in their vault. No special format beyond markdown headings.

**Location:** `OpenAugi/heartbeat-skill.md`

**Example:**

```markdown
# Heartbeat Skill — Rules for Processing New Blocks

## Workstreams (classify against these)
- openaugi — the tool, the project, the code
- work — Datadog stuff, topology, metrics
- content — writing, posts, creator work
- self — journal, reflection, life, kids, personal
- reference — memories, quotes, things to keep

## Defaults (when no zzz: instruction)
- If block is from journals/openaugi/ → workstream = openaugi
- If block is from journals/work/ → workstream = work
- If block is from a daily note → classify by content
- Tags: use facets if you see clear signals (type/idea, type/task, type/insight)
- Unsure? Tag it and move on. Don't block.

## Task handling
- If a block mentions a deadline → create a task entry in OpenAugi/Tasks/
- If a block has a checkbox → note it but don't auto-act
- Never modify the raw source note

## Research handling
- Search the vault first using openaugi:search and get_context
- Summarize what's already known about the topic
- List what's missing, what to read next
- Write results to OpenAugi/Research/<topic>.md

## What to write back
- Heartbeat log at OpenAugi/Heartbeat/YYYY-MM-DD.md summarizing:
  - Blocks processed
  - Classifications made
  - Connections found
  - Actions taken
  - Anything flagged for review
- Never modify raw notes. Everything system-generated lives in OpenAugi/.
```

The user edits this in Obsidian like any other note. Change the workstream list, change the rules, change what the agent does. No code changes required.

## The heartbeat log

Output artifact. One file per run, named by date.

**Location:** `OpenAugi/Heartbeat/2026-04-08.md`

**Example:**

```markdown
# Heartbeat — 2026-04-08 06:14

## Blocks processed: 7

### Block 1 — "Had a thought about matryoshka embeddings..."
- **Source:** Journal/2026-04-08.md
- **Workstream:** openaugi
- **Instruction followed:** "research this more - find papers in my vault"
- **Actions:**
  - Searched vault, found 3 related blocks: [[2025-11-17 - Matryoshka embeddings]], [[Research - Multi-res retrieval]], [[PMOC - Embedding strategies]]
  - Wrote summary to OpenAugi/Research/matryoshka-embeddings.md

### Block 2 — "Need to fix the README onboarding flow..."
- **Source:** OpenAugi Journal.md
- **Workstream:** openaugi
- **Instruction followed:** "task - do this in the openaugi repo"
- **Actions:**
  - Created task entry OpenAugi/Tasks/TASK-2026-04-08-README-onboarding.md
  - Linked to source block

### Block 3 — "Feeling stuck on the direction of openaugi..."
- **Source:** Journal/2026-04-08.md
- **Workstream:** self
- **Instruction followed:** "just log this, tag personal/reflection"
- **Actions:** tagged, logged, no further action

...
```

This is the audit trail. When you come back, you can see what the agent did. If something went wrong, you read the log and correct the skill file.

## What to build first — minimum useful version

Keep scope tight. Four things, in order.

### Step 1 — `openaugi heartbeat` command that spawns the agent

A new CLI command in [src/openaugi/cli/main.py](../../src/openaugi/cli/main.py):

1. Run incremental ingest (existing `openaugi up` logic)
2. Query blocks added since last heartbeat timestamp (stored in `~/.openaugi/last_heartbeat`)
3. Build a prompt listing those blocks (title, source, content preview, tags, links)
4. Shell out to `claude -p <prompt> --allowedTools Read,Write,mcp__openaugi__*`
5. Update `last_heartbeat` timestamp on success

No `zzz:` extraction yet. No skill file logic beyond a path reference. The agent handles everything itself using the existing MCP tools and a skill file you write by hand.

### Step 2 — Write the skill file

Manual. Create `OpenAugi/heartbeat-skill.md` with your workstreams and default rules. You can see the example above.

The heartbeat command should fail loudly if the skill file is missing, with a message telling you to create it.

### Step 3 — `zzz:` extraction in the vault adapter

Extend [vault.py](../../src/openaugi/adapters/vault.py) to capture lines starting with `zzz` in a block. Store them in `block.metadata["zzz_instructions"]` (list — one item per `zzz:` line, see [zzz-instructions.md](zzz-instructions.md)). Strip them from the clean content stored in `block.content`.

Then update the heartbeat prompt builder to surface these per-block:

```
## Block 3 (from Journal/2026-04-08.md)
Content: "Feeling stuck on the direction of openaugi today"
User instruction: just log this, tag personal/reflection
```

### Step 4 — First real run, iterate on the skill file

Write a daily note with a few `zzz:` instructions. Run `openaugi heartbeat`. Read the heartbeat log. See what the agent did.

If it did the wrong thing: update the skill file. Don't touch the code.

This is the feedback loop that matters. The skill file is where you iterate, not the Python.

## What NOT to build

- **No file watcher.** Run it on demand first. Automate later when it feels stable.
- **No `qqq` splitting yet.** Existing `###` splits are fine for the first run. Add `qqq` when you feel the block granularity is wrong.
- **No custom config system.** The skill file IS the config. It's a markdown file you edit in Obsidian.
- **No classifier code.** The agent classifies. Don't duplicate that in Python.
- **No renderer (Streams/Concepts pages).** The heartbeat log is the first artifact. Add Streams/Concepts rendering later if the heartbeat itself doesn't cover it.
- **No review UI.** The heartbeat log IS the review surface.
- **No agent memory beyond the skill file.** The agent starts fresh each run. If you want it to remember something, write it into the skill file.

## How this relates to prior plans

| Prior plan | What survives | What changes |
|---|---|---|
| [capture-tag-stream-loop.md](capture-tag-stream-loop.md) | Principles (deterministic-first, raw notes untouched, rendered output in `OpenAugi/`) | Classification moves from narrow API call → agent session. The `classify.py` module becomes unnecessary. The LLM fallback in [tag_inference.py](../../src/openaugi/pipeline/tag_inference.py) becomes unnecessary. |
| [from-capture-to-jarvis.md](from-capture-to-jarvis.md) | Layer 1 vision (sense-making pipeline, concepts are emergent, workstreams as minimum taxonomy) | The "deterministic pipeline + LLM classification" step becomes "deterministic script + agent session." `zzz:` tokens go from "extra metadata" to "the primary per-block user directive." |
| [phase2-compile.md](phase2-compile.md) (shipped in [compile.py](../../src/openaugi/pipeline/compile.py)) | Compile produces context blocks that MCP tools read | Context blocks feed the agent's searches. The renderer path (OpenAugi/Streams/*.md) can wait. |
| [phase3-adapters.md](phase3-adapters.md) | New adapters just produce blocks | Heartbeat works on blocks regardless of source. No adapter-specific code. |

## Open questions

- **How to detect "new since last heartbeat."** Timestamp on `~/.openaugi/last_heartbeat`? Block `created_at` column? Both? Probably: filter blocks where `created_at > last_heartbeat_timestamp`.
- **Prompt size limits.** If you haven't run heartbeat in a week, there could be hundreds of new blocks. Options: batch the agent into multiple sessions, cap at N blocks per run, or let Claude Code's large context handle it. Start with a cap (e.g. 50 blocks per heartbeat).
- **Skill file location.** `OpenAugi/heartbeat-skill.md` in the vault (user-editable, synced) vs `~/.openaugi/heartbeat-skill.md` (private). Vault is better — you edit it like any other note and it syncs via Obsidian.
- **Error handling.** If the agent crashes mid-session, do we retry? Mark those blocks as "seen" anyway? Probably: don't update `last_heartbeat` on failure, so the next run retries.
- **Cost visibility.** Claude Code sessions can get expensive on big batches. Log token usage per heartbeat run for observability.

## Success criteria

1. `openaugi heartbeat` runs end-to-end: ingest → find new blocks → spawn agent → agent writes log.
2. The heartbeat log at `OpenAugi/Heartbeat/YYYY-MM-DD.md` contains useful summaries of what the agent found and did.
3. **`zzz:` instructions are honored.** You write "zzz research this" on a block, the agent does research.
4. **You edit the skill file instead of the code** when the agent does the wrong thing.
5. **You open the heartbeat log in the morning** and it tells you something worth knowing.

## Next action

In order:

1. Create `OpenAugi/heartbeat-skill.md` in the vault with workstreams + default rules. Write it by hand first — this forces clarity about what you actually want the agent to do.
2. Add `openaugi heartbeat` command that: runs ingest → queries new blocks since `~/.openaugi/last_heartbeat` → builds a prompt → shells out to `claude -p <prompt> --allowedTools Read,Write,mcp__openaugi__*` → updates timestamp.
3. Run it once. Read the log. Edit the skill file. Run again.
4. Only after that feedback loop is working: add `zzz:` extraction to the vault adapter.

Nothing else until you've done a week of real heartbeat runs and know what's actually missing.
