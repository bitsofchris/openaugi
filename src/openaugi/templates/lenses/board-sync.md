---
kind: engine
name: board-sync
description: >-
  "Does my board still match the work?" — compares every open card with what its note, its Agent Log and its pull requests say happened, and proposes the edits that would make the board true again.
scope: >-
  The board note (every open card above the archive that links a note, and its linked sub-items), each of those notes (Outcome, Verify, Next, the latest Journal entries), and each one's Agent Log (## Context for links, ## Chats for what was worked on). Every pull-request link found in those is checked for its state. When the draft names an anchor and the instruction says "this one", only that card and its sub-items. No semantic search: this lens reads state, it does not go looking for ideas.
trigger: on-demand
target: >-
  note — the board note and the Agent Logs of the notes it links, edited in place, and only for the rows the user says yes to in chat: a sub-item or card ticked, a card's clause, a card moved on a board whose columns are stages, a pull-request link added to an Agent Log. Never a new card and never a reorder.
---

# Board Sync

## Intent

A board drifts. Work moves in chats and pull requests while the card stays
where it was put. This lens reads what happened and lists the smallest edits
that would make the board say it. It proposes, the user rules, and only then
does anything change.

## Process

1. Read `OpenAugi/AGENT/kanban.md` when it is there: its file format and
   card rules bind every edit below. Then read the board. A **card** is an
   open item above the archive that links a note; an indented checkbox under
   it that links a note is one of its **sub-items**.
2. For each card and sub-item, gather what happened, newest first:
   - the note: the last few Journal entries (their dates and what they say
     moved), whether its Verify reads as met, its open `# Next` items;
   - its Agent Log, `Agent Log - <note name>.md` in the Agent Log folder
     (`OpenAugi/Agent-Logs` unless the draft names another): the links under
     `## Context`, and under `## Chats` what was worked on and when;
   - every pull-request link in those or in the card's clause: its state
     (open, in review, merged, closed) and title. Use
     `gh pr view <url> --json state,title,mergedAt,reviewDecision` when `gh`
     is installed and signed in, else whatever GitHub tools the chat has.
     With neither, keep the link and mark it unchecked.
3. For each, either it is **in step** (say nothing about it) or propose
   edits, from this list only:
   - **tick**: a sub-item whose work is finished, because its pull request
     merged or the Journal says its Verify was met;
   - **done?**: a top-level card whose sub-items are all ticked and whose
     Verify is met. Ask; never tick a top-level card on evidence alone;
   - **clause**: the card's clause no longer says why it is there. Give the
     new one, short (`— in review, PR 12`);
   - **link**: a pull request named in a Journal entry or a chat that is not
     under the Agent Log's `## Context`. Add it there as
     ``- url: `<link>` · PR: <title>``, which pins it to the work;
   - **move**: only on a board whose columns are stages (their headings say
     so: To do, Doing, Review, Done). Name the column the evidence points
     to. On a board whose columns are areas, never propose a move.
4. Answer in this shape, rows numbered, only cards with an edit:

```
# Board sync — YYYY-MM-DD

| # | Card | What happened | Proposed edit |
|---|---|---|---|
| 1 | [[note]] | PR 12 merged 2026-01-04 | tick the sub-item |

In step: N cards. Unchecked links: N (no GitHub access).
Which rows? (all, numbers, or none)
```

5. On the answer, make exactly those edits. On the board change only the
   lines named, as plain text edits (never rewrite the file); add the links
   to the Agent Logs. Then say what changed, one line each.

## Hard rules

- Propose first, write after a yes. A yes covers the rows named and nothing
  else.
- Never add a card, delete a card, reorder a column, or touch the archive or
  the board plugin's settings block.
- Evidence or nothing. Each row names the Journal date, the pull-request
  state or the Verify line it rests on. "Looks done" is not evidence, and a
  card that has only gone quiet gets no row.
- Read only on GitHub: never comment, approve, merge or close.
- The user's notes are theirs: this lens writes to the board and to Agent
  Logs, never into a note's Outcome, Verify, Next or Journal.
- One screen. Past twenty rows, show the twenty with the strongest evidence
  and count the rest.
