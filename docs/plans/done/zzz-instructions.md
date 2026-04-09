---
name: zzz — Inline Block Instructions
description: Per-block instructions written inline in pen-mode. Python parser is dumb (finds `zzz:` lines and attaches them as metadata). The skill file defines what instructions are valid and what they mean. One signal, one target (the containing block), flexible vocabulary owned by the user.
status: draft
created: 2026-04-08
---

# zzz — Inline Block Instructions

## Context

[heartbeat.md](heartbeat.md) is shipped: the dumb script + smart agent architecture works. This plan adds the inline instruction layer so you can steer the agent while writing in pen, without switching notes.

The goal: you're writing in a daily note, you have a thought, you want the agent to *do something specific* with it (research it, make a task, just log it), so you write a `zzz:` line right under the thought. Next heartbeat run picks it up and does the thing.

## Design principles

1. **Non-confusing** — always starts with `zzz:`, always on its own line, always about the containing block. No namespaces, no cross-block references, no meta-targets. One rule.
2. **Flexible vocabulary** — the list of valid instructions lives in the skill file, not in code. You extend by editing the skill file in Obsidian. Python parser never needs updating.
3. **LLM-leveraged** — the parser only extracts raw instruction strings. The agent, guided by the skill file, interprets intent. "research this," "find papers on this," and "look into this more" all map to the same action if the skill file says so. You don't need exact syntax.

## Block definition (canonical)

A **block** is the smallest chunk between delimiters, in this priority order:

1. `###` header — provides date/section context as parent metadata
2. `qqq` marker — finer-grained atomic thought breaks (**case-insensitive**: `qqq`, `QQQ`, `qQq`, `Qqq` all match)

A block runs from one delimiter to the next, whichever comes first.

**Fallback:** if a note has **no `###` headers and no `qqq` markers**, the entire note is a single block. This is important for focused notes (a task file, a single-topic note) where you don't want to split at all.

Examples:

```markdown
# Journal 2026-04-08           ← file title, not a block delimiter

### Morning pages               ← H3 starts a block
First thought
qqq                             ← qqq ends the block, starts a new one
Second thought
qqq                             ← another block
Third thought

### Evening reflection          ← new H3, new block
Fourth thought
```

→ 4 blocks: "First thought", "Second thought", "Third thought", "Fourth thought"

```markdown
# OpenAugi Brand Vision         ← no ### or qqq anywhere

Some thoughts about positioning...
More thoughts about the business model...
```

→ 1 block: the entire note content

## zzz: syntax

```
zzz: <instruction text>
```

- Must start at the beginning of a line (after optional whitespace).
- **Case-insensitive**: `zzz:`, `ZZZ:`, `zZz:`, `Zzz:` all match. The user writes however feels natural in pen.
- Must be inside a block (if not, it attaches to the enclosing block per the block rules above).
- Instruction text is free-form natural language — no required keywords.
- **Multiple `zzz:` lines in one block are treated separately**, as a list of instructions all scoped to that block.
- `zzz:` lines are **stripped from clean block content** so they don't pollute embeddings/search, but preserved in block metadata.

Example:

```markdown
### Morning pages
Thinking about matryoshka embeddings for multi-res matching
zzz: research this, find what I have in the vault
zzz: remember — I also talked about this with Ethan last week
qqq
Need to fix README onboarding before Thursday
zzz: make this a task
```

→ Block 1: content "Thinking about matryoshka..." with 2 zzz instructions
→ Block 2: content "Need to fix README..." with 1 zzz instruction

## Parser responsibilities (Python, dumb)

The vault adapter does exactly these things, no more:

1. Split into blocks per the canonical definition above (including the fallback). Both delimiters are **case-insensitive**:
   - `###` header (unchanged)
   - `qqq` marker on its own line — match `^\s*[qQ]{3}\s*$`
2. For each block, find every line matching `^\s*[zZ]{3}:\s*(.+)$` — **case-insensitive** on the `zzz` prefix.
3. Store the extracted texts as a list on block metadata:
   ```python
   block.metadata["zzz_instructions"] = [
       "research this, find what I have in the vault",
       "remember — I also talked about this with Ethan last week",
   ]
   ```
4. Strip the `zzz:` lines from `block.content`.
5. Compute content hash on the stripped content — so adding or removing a `zzz:` line after the agent has already acted on it doesn't re-trigger processing for the same text. (Open question below.)

That's it. The parser has no knowledge of what any instruction means.

Reference regex constants (put these in a shared place so the adapter and any tests agree):

```python
import re

QQQ_LINE = re.compile(r"^\s*[qQ]{3}\s*$")
ZZZ_LINE = re.compile(r"^\s*[zZ]{3}:\s*(.+?)\s*$")
```

Using character classes `[qQ]`/`[zZ]` makes the case-insensitivity explicit and scoped to just those letters — safer than a global `re.IGNORECASE` flag on the whole file.

## Skill file responsibilities (user-editable, owns the vocabulary)

The canonical template lives in the repo at [src/openaugi/templates/heartbeat-skill.md](../../src/openaugi/templates/heartbeat-skill.md). At runtime, the heartbeat command reads it from `<vault>/OpenAugi/heartbeat-skill.md` (copied into the user's vault on first setup).

The template now includes a `## Supported zzz: instructions` section alongside the existing workstreams, defaults, and task/research rules from [heartbeat.md](heartbeat.md). The four starter instructions:

- **research** — search the vault using openaugi MCP tools, summarize what's known, write to `OpenAugi/Research/<slug>.md`, link from the heartbeat log.
- **task** — create a task entry at `OpenAugi/Tasks/TASK-YYYY-MM-DD-<slug>.md` with the block content as context. Don't execute, just capture.
- **log** — log only, no further action. For reflections and personal notes you just want captured.
- **remember** — promote the block to a reference note at `OpenAugi/Reference/<slug>.md`. For quotes, facts, decisions.
- **anything else** — agent uses best judgment and logs what it did, so the user can review and either accept or update the skill file.

Key property: **the list of supported instructions lives in the skill file, not in code**. Add new ones by editing the markdown file. The agent reads the skill file every heartbeat run, so changes take effect immediately with no Python changes.

The template is a starting point — the user copies it to their vault and customizes the workstreams and rules. Code never reads the repo template at runtime; it only reads `<vault>/OpenAugi/heartbeat-skill.md`.

## Agent behavior

For each new block the heartbeat agent processes:

1. Handle the block normally (classify workstream, extract links, etc.) using the existing rules in the skill file.
2. If `block.metadata["zzz_instructions"]` is non-empty, for each instruction string:
   - Match intent against the "Supported zzz instructions" section of the skill file.
   - If matched → perform the corresponding action.
   - If not matched → use best judgment and log what it did.
3. Log every instruction and the action taken in the heartbeat entry for that block so you can review.

**Always about the current block.** The agent never re-targets an instruction to a different block or to the skill file. If you want to update the skill file, edit it directly in Obsidian.

## Example end-to-end

Daily note entry you write:

```markdown
### Morning
Had a thought about matryoshka embeddings for multi-res matching
zzz: research this, find what I have
qqq
Feeling stuck on openaugi direction today
zzz: log only, tag self/reflection
qqq
Need to fix README onboarding before Thursday
zzz: make this a task
```

Next heartbeat run:

- **Block 1** → workstream=openaugi, 1 zzz instruction. Agent calls `openaugi:search` for "matryoshka embeddings", finds 3 related blocks, writes summary to `OpenAugi/Research/matryoshka-embeddings.md`, links from heartbeat log.
- **Block 2** → workstream=self, 1 zzz instruction. Agent logs with tag self/reflection, takes no other action.
- **Block 3** → workstream=openaugi, 1 zzz instruction. Agent creates `OpenAugi/Tasks/TASK-2026-04-08-readme-onboarding.md`, links back to source block.

Heartbeat log records all three actions so you can review in the morning.

## What to build (minimum useful version)

Four small steps. Each independently testable.

### Step 1 — Canonical block definition + fallback

Extend [vault.py](../../src/openaugi/adapters/vault.py) so the splitter:
- Splits on `###` headers (existing)
- Splits on `qqq` markers within or outside sections, **case-insensitive** — match `^\s*[qQ]{3}\s*$` (new)
- **If a file contains neither `###` nor `qqq`, treats the entire note content as one block** (new — verify this path exists; it probably doesn't)

Update content hashing so block IDs are stable under the new split rules.

### Step 2 — `zzz:` extraction

In the block-building path of [vault.py](../../src/openaugi/adapters/vault.py), after content is finalized per block:
1. Regex find all lines matching `^\s*[zZ]{3}:\s*(.+?)\s*$` — **case-insensitive** on the prefix.
2. Store matches as `block.metadata["zzz_instructions"] = [...]`.
3. Strip those lines from `block.content` before hashing.

Unit tests:
- Block with zero `zzz:` → empty list
- Block with one `zzz:` → list of one
- Block with multiple → list in order
- **Case variants:** `zzz:`, `ZZZ:`, `zZz:`, `Zzz:` all captured
- **`qqq` case variants:** `qqq`, `QQQ`, `qQq` all split
- `zzz:` mid-line (not line-start) → ignored
- `zzz:` in a code fence → still captured for now; can refine later if it's a problem
- Note with no delimiters → single block containing full content

### Step 3 — Skill file template

The canonical template is in the repo at [src/openaugi/templates/heartbeat-skill.md](../../src/openaugi/templates/heartbeat-skill.md). It already includes the "Supported `zzz:` instructions" section with the 4 starters (research, task, log, remember) and the "best judgment" fallback.

Copy the template to `<vault>/OpenAugi/heartbeat-skill.md` and customize the workstreams for your own vault. The heartbeat command reads the vault copy, not the repo template — the repo version is a starting point that ships with the package.

A future `openaugi init` command can automate this copy; for now it's manual.

### Step 4 — Surface instructions in the heartbeat prompt

In the heartbeat command's prompt builder, include `zzz_instructions` per block:

```
## Block 3 (from: Journal/2026-04-08.md)
Content: "Need to fix README onboarding before Thursday"
Workstream: openaugi (from source path)
User instructions:
  - make this a task
```

The agent already reads the skill file, so it knows what "make this a task" means. No code changes in the agent path — just surface the data.

## What NOT to build

- **No namespaces.** No `zzz skill:`, no `zzz task:`, no per-instruction target prefixes. One form: `zzz: <text>`. If it turns out a second target is needed later, add it then — not speculatively.
- **No Python-side instruction matching.** The parser extracts strings; it does not know what "research" or "task" mean. That knowledge lives in the skill file.
- **No hardcoded instruction list.** Don't ship a fixed vocabulary in code. The skill file owns it.
- **No retroactive handling.** If you add a `zzz:` to an old block, the next heartbeat will pick it up only if the block is considered "new since last heartbeat" by the existing logic. Don't build a re-process mechanism yet.
- **No auto-expansion or macro system.** `zzz: research this` is a natural-language instruction, not a templated macro. Keep the LLM doing the work.
- **No line-level targeting inside a block.** Instructions always scope to the whole containing block. Don't try to let users attach instructions to a specific paragraph inside a block.

## Open questions

- **Content hash and stripping.** If the stripped-content hash is what drives incremental ingest, then adding a `zzz:` line to a previously-processed block won't change the hash and the heartbeat won't see the new instruction. Options:
  - (a) hash the stripped content + a hash of the zzz instructions → the block is "changed" when zzz changes, so the agent re-runs
  - (b) hash stripped content only → zzz changes are silent unless you also change the block content
  - Leaning (a) — that way you can add a `zzz:` to an old thought and have the agent act on it next run.
- **Code fences.** A `zzz:` inside a markdown code fence will be captured by the naive regex. Probably fine for now (you're unlikely to write code examples with `zzz:` at line-start). Revisit if it becomes noise.
- **Empty-block instructions.** A block that's *only* a `zzz:` line with no other content — is that valid? Suggest: yes. It becomes a standalone instruction block that the agent processes and logs. Useful for "zzz: summarize this week's openaugi entries" type directives, if we later allow that kind of cross-block instruction. For the first cut, keep the "always about the current block" rule and accept that a content-less block's instruction is a no-op unless the skill file defines how to handle it.
- **Fallback block for note-level instructions.** If the entire note is a single block (no `###`, no `qqq`) and you write `zzz: remember` at the bottom, that attaches to the whole-note block. Works by construction. Good.
- **Stripping and embeddings.** Should `zzz:` lines be part of the embedded content (so semantic search surfaces them) or not? Suggest: strip them from embeddings too. They're instructions, not thoughts. You search your thoughts, not your instructions to the agent.

## Success criteria

1. You write `zzz: research this` under a thought, run `openaugi heartbeat`, and the agent writes a research summary to `OpenAugi/Research/`.
2. You add a new instruction type to the skill file (e.g. "draft" — write a blog draft from this block), use it in a block, and it works **without any Python changes**.
3. You can open a block's heartbeat entry and see exactly which `zzz:` instructions were honored and what the agent did for each.
4. A note with no `###` or `qqq` is treated as a single block, and its `zzz:` lines attach to that block.
5. A block with multiple `zzz:` lines has every instruction handled independently.

## Next action

1. Add failing tests for the vault adapter:
   - A fixture note with no `###` or `qqq` should produce one block containing the full content.
   - A fixture note with `qqq`, `QQQ`, and `qQq` mixed should split on all three.
   - A fixture note with `zzz:`, `ZZZ:`, and `zZz:` should capture all three into `metadata["zzz_instructions"]`.
2. Fix the vault adapter:
   - Add the "no delimiters → one block" fallback path.
   - Add case-insensitive `qqq` splitting within and outside headers.
   - Add case-insensitive `zzz:` extraction + content stripping.
3. Update the heartbeat prompt builder in [heartbeat.py](../../src/openaugi/pipeline/heartbeat.py) to surface `zzz_instructions` (the new list field) per block instead of the current single `zzz_instruction`.
4. Copy [src/openaugi/templates/heartbeat-skill.md](../../src/openaugi/templates/heartbeat-skill.md) to `<vault>/OpenAugi/heartbeat-skill.md` and customize the workstreams.
5. Write a real daily note with 2-3 `zzz:` instructions, run `openaugi heartbeat`, read the log, iterate on the skill file based on what the agent did wrong.

Stop there. No namespaces, no meta-targets, no instruction registry. See what the loop feels like after a few real runs before adding anything.
