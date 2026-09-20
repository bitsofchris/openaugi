---
kind: engine
name: kanban
description: How the two kanban boards work — file format the Obsidian plugin needs, columns, cards, who moves what, and the Sunday pass. Read before touching _private/0-Current Focus/Kanban.md (the Board) or Backlog.md (the Backlog).
created: 2026-09-13
consumers: [currency-board, weekly-reflection, review-pass, habit-parse]
---

# Kanban

## What it is

Two boards in `_private/0-Current Focus/`, opened by the Obsidian Kanban plugin: **`Kanban.md` is the Board** (in progress and next, cap 3 per AMOC column) and **`Backlog.md` is the Backlog** (never lost, no cap). System note: [[MOC - My Operating System]]. A **column is an AMOC** from [[My Taxonomy]], plus Triage (Board only) and Household / Other (both). A **card is a link** to the note that holds the work, plus at most one clause. The board holds; it never explains. If a card needs a paragraph, the paragraph goes in the linked note and the card stays one line.

The boards are updated together on Sundays. Design history: [[PMOC - Kanban per Area - Triage, Park, Elevate]].

## File format (the plugin is strict)

```markdown
---

kanban-plugin: board

---

## <Column name>

- [ ] <card: [[link]] — one clause>
- [ ] <card>

## <Next column>

- [ ] <card>

***

## Archive

- [ ] <cards the plugin moved here on "archive">

%% kanban:settings
```
{"kanban-plugin":"board","list-collapse":[false,false]}
```
%%
```

- Frontmatter must contain `kanban-plugin: board`. Extra frontmatter keys (like `description`) are fine.
- Every `## ` heading is a column, in file order. No prose between columns — the plugin treats stray text as a card or drops it.
- Cards are `- [ ]` list items. Ticking a card marks it done; the plugin's "archive" moves it under `***` / `## Archive`. Done is not a column.
- `list-collapse` in the settings block has one boolean per column; keep it in sync when adding a column.
- Edit with plain text writes. Never rewrite the whole file from a template; read it, change the lines, write it back, so their manual moves survive.

## Columns (current)

Same on both boards, in this order. `list-collapse` needs one boolean per column.

One column per AMOC in the user's taxonomy, plus Triage (Board only), a
Creative sub-list (Backlog only), and Household / Other. The Board caps each
AMOC column at 3.

A Board card in OpenAugi or Research Engineer is a PMOC. The Board is the record of what is active; `#status/active` is a tag the user sets for the Dashboard's query and is not kept in sync with the Board — a card without the tag is not drift. Backlog sub-lists are allowed where an area has real sub-kinds, a few, never on the Board. A new column only when they name it.

## Card rules

- **A card is a thing that ends.** A task or a project — something that can be ticked done. A reference note, a system description, an AMOC, a habit, an idea with no next action: never a card. A reference belongs as a link on the area's AMOC journal or the Dashboard. If a card cannot be finished, it is misfiled.
- **One card, one link, one clause.** `- [ ] [[Note]] — why it is here`. If there is no note, the clause is the whole card and it should be short enough to do in an hour, otherwise it needs a note first.
- **Order within a column is theirs, and it means something: the top card of each column is this week's focus** ([[Slowly Changing Context]] §4). Agents append at the bottom and never reorder, except the Sunday session performing a reorder they ruled.
- **No duplicates.** Search the board (all columns and Archive) before adding. A card that exists gets its clause updated, not a twin.
- **Source stays with the card** when an agent adds it: `(from [[2026-09-11]])` at the end of the clause, so they can see why it appeared.
- **Nothing lane-shaped.** If it belongs to an active PMOC, it is a left-off line in that PMOC, not a card here. Habits are not cards ([[Slowly Changing Context]] §7).

## Who moves what

- **The user moves cards.** Between columns and between boards, ticks done. Always. Ticked cards are hidden by the plugin; nobody archives on a schedule.
- **The plugin cannot move a card between files.** So a cross-board move is either the user cutting and pasting the line, or an `aaa:` on the card in plain words ("move to top of backlog", "this is done, the video gets its own backlog card"). There is no special syntax. The Sunday pass reads every card `aaa:`, does what it says — move, split, re-clause — and appends `(done <date>)` to the instruction so it is not re-run; the words themselves are never deleted. Weekday boards do not act on card `aaa:` lines.
- **Agents add cards** at the bottom of a Backlog column (or Triage), with a source, when they have named the thing in their own writing (the daily parse, a `[todo]` in the reflection). Never straight into an AMOC column on the Board, never by inference, never from an AI-generated note.
- **Agents propose** as checkboxes in the weekly reflection's *Triage outside the slots* section, never by editing a column: a destination for each Triage card; a pull from the Backlog only when they ask or a column has room on Sunday; one line when a column is over cap, for discussion. **No push suggestions** — a quiet card stays until they move it.
- **The daily board does not read this file.** The Household column is theirs to tick, and the weekly reflection is the only pass that proposes changes to it.

## The Sunday pass (together)

Step 3 of the Sunday session (`lenses/weekly-reflection.md` § The Sunday
session): one question per card decision, in this order, each performed
before the next is asked:

0. **Card instructions.** Every card carrying an `aaa:` on either board, first: one box that does what the words ask.
1. **Triage.** Each card in the Board's Triage column gets one proposed destination.
2. **Add.** At most three of the week's candidates go to the Backlog with their source, one question each. An add to a Board column when they approve it in the session.
3. **Over cap.** Any AMOC column past 3: one line naming the cards, then talk. Nothing nominated.
4. **Pull.** Only if they ask to fill a slot: the Backlog cards their writing touched this week.

## Open design questions

Tracked in [[PMOC - Kanban per Area - Triage, Park, Elevate]]. The two that gate anything bigger: one file versus one per area, and whether this board is a view over `.board-state.json` or its own store. Today it is its own store, one file.
