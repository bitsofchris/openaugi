---
kind: engine
name: pmoc
description: How a PMOC is made and kept — the two kinds (a task note on the Backlog, a PMOC when active), the line between a task and a project, the note format, and the five-step pass (intent, PMOC check, context, related, first entry). Read before creating or reviving any PMOC or task note.
created: 2026-09-13
consumers: [weekly-reflection, currency-board, kanban, task-dispatch]
---

# PMOC

## What a PMOC is, and what it is not

A **PMOC is an outcome that takes more than one sitting.** It accumulates dated entries; the newest entry is where they left off. It ends when the outcome sentence is true. It is a card on the Board (a thing that ends), it carries `#status/active` while the user is working it — the tag is their marker for the Dashboard query, not a mirror of the Board — and it is never deleted: an inactive PMOC is the memory the PMOC check reads before anything similar gets a new note.

A **TASK is one dispatched unit of work** — a `zzz:` or a board `do`, hydrated by the task watcher, finished inside one session, with a `## Results` section. A task may *belong* to a PMOC (it links it) but is never one.

Rule of thumb: **if it will have a second dated entry, it is a PMOC.** If it is done when the session ends, it is a task. An **AMOC** is an area, never active, never a card.

## Two kinds: task note and PMOC

| Kind | Note | Lives | Card | Tag |
|---|---|---|---|---|
| **Task note** — a thread with an end state but not yet a project | `OpenAugi/Notes/YYYY-MM-DD - <what it is>.md` — end state, their words, what exists, what is missing, smallest next step. No Journal, no header. | OpenAugi/Notes | Backlog, its AMOC column | none |
| **PMOC** — active, will have a second dated entry | `_private/3-MOCs and Projects/PMOC - <Outcome>.md` in the format below | 3-MOCs and Projects | Board (or Backlog if inactive) | `#status/active` / `inactive` |

The same pass makes both: clarify the end state, run the PMOC check, assemble context, find related, write the next step. A task note **promotes** to a PMOC the day it gets pulled onto the Board and needs a second entry: the PMOC is created with the dated note as its first Context source, the note stays where it is, the card is moved. *Task note* is the note; *task file* (`OpenAugi/Tasks/TASK-*.md`) is the dispatched unit that may work on it. Never rewrite the dated note into a PMOC in place; the date is the record of when the thread started. Examples: [[2026-09-13 - Link notes to coding sessions]] is a task note; [[PMOC - Lens Scheduler - The Augi Task Loop]] is a PMOC because it will take several entries.

## Format

Path: `_private/3-MOCs and Projects/PMOC - <Name>.md`. Name says the outcome, not the topic (*Kanban per Area - Triage, Park, Elevate*, not *Kanban*).

```markdown
---
description: <one line the PMOC check will read: the problem, and what done looks like>
outcome: <the sentence that, when true, closes this PMOC>
---
#note-type/pmoc #area/<one area> #status/active

[[AMOC - <parent>]] · [[<the one or two notes this grows out of>]]

<anything they want to say to themselves goes here, above the Journal — their space, never written by an agent>

# Journal

### YYYY-MM-DD — created

<The synthesized next step: the smallest thing that moves toward the outcome, and the one open question if there is one. This is the first left-off line the currency board will read.>

*(Augi: this block was #ai-generated)*

# Context

## Intent

<The problem or intent in their words, dated and quoted. Then the outcome sentence again, plainly. Two short paragraphs at most.>

## What exists (assembled at creation)

<What already exists so nothing is rebuilt: prior notes and their rulings, code and branches, earlier attempts and why they stopped, constraints they have written. Bold lead-ins, one paragraph each. Written once and not maintained; the journal carries change.>

## Not in scope

<Two to four lines. What this PMOC deliberately does not do, with the note where that thing lives if it lives anywhere.>

## Links

<Every note read during creation, one line each, grouped loosely.>
```

**Journal comes first.** Everything assembled at creation sits under one `# Context` H1 with H2 sections, below the journal, so it never gets in the way of the left-off. Newest entry on top. Each later entry is `### YYYY-MM-DD` (their) or `### YYYY-MM-DD — <what happened>` with the Augi marker (agent). The closing entry is `### YYYY-MM-DD — done` and states which outcome sentence became true; they flip the tag to `#status/inactive` and ticks the Board card.

## The creation pass (five steps, one sitting)

1. **Clarify intent and outcome.** From their words. Write the outcome sentence first. If it cannot be written, this is not a PMOC yet — it is a Backlog card or a question back to them, one line.
2. **The PMOC check.** One `search` over `#note-type/pmoc` with three or four keywords; read the `description:` of the top hits. A match means append and revive, not create (augi-agent.md, How to work item 4).
3. **Assemble the context.** Vault search — graph, semantic, keyword — plus the repo state when code is involved (branch, last commits, open task files). Read, do not summarize from memory. Everything that would otherwise be rebuilt goes in *Context*.
4. **Find related.** Links to the AMOC, the notes it grows out of, sibling PMOCs, and anything the Board or Backlog already holds on it. A Backlog card that this PMOC absorbs gets ticked.
5. **Synthesize the first entry.** The smallest next step and the one open question. Write the note. Add one card to the Board in its AMOC column (`- [ ] [[PMOC - Name]] — <clause>`); if that column is at cap, the card goes to the Backlog and the tag is `#status/inactive` until they pull it.

Agents run this pass only when they ask for a PMOC or a task note, or when the weekly reflection's triage proposes one and they tick it. Never spawn either from inference. A task note is agent output and goes under `OpenAugi/Notes`; a PMOC is their and goes where their PMOCs live, which is why it needs their ask.

## Keeping it

- **Left-off is the newest entry.** The currency board reads it with the coding sessions; nothing else needs to be maintained for the board to be right.
- **Context is not updated;** the journal is. If the context section is badly wrong, one dated entry says so.
- **The space between the links line and `# Journal` is theirs.** Notes to themselves, pinned reminders, whatever. Agents never write there.
- **Dashboard and Board stay in sync** through the tag (kanban.md, Sunday pass step 5).
- **Revival** is a dated entry that says why, the tag flipped, the card back on the Board. Not a new note.
