---
kind: engine
name: jira-pull
description: >-
  "What is on my plate in Jira, and where does each issue belong on my board?" — lists the open issues assigned to the user, matches them to the task notes already there, and proposes for each one: nothing, an update, a link, or a new task and where its card would go.
scope: >-
  Jira, read only: the open issues assigned to the user (in the project the draft or the instruction names, else all of theirs) with summary, status, parent or epic, and last update. The board note and the notes it links, for the issue each already carries (a `Jira: KEY` line or `jira:` in frontmatter) and for titles to match against. No semantic search.
trigger: on-demand
target: >-
  note — task notes and the board note, edited or made only for the rows the user says yes to in chat: a `Jira:` line added to a note, a card ticked or its clause updated, a new task note from the task template with its card on the board. Jira itself is never written.
---

# Jira Pull

## Intent

The tracker says what the user owes; the board says what they are working
on. This lens lines the two up: every open issue is either already a task
here, the same work as a task here under another name, or missing. It says
which, proposes where a missing one would land, and writes nothing until the
user picks.

## Process

1. Read `OpenAugi/AGENT/kanban.md` and the task template
   (`OpenAugi/AGENT/templates/Task.md`) when they are there: they decide the
   shape of every card and note below. Read the board and the notes it
   links, archive included, and map each Jira key already carried (a
   `Jira: KEY` line or `jira:` in frontmatter) to its note.
2. Ask Jira for the user's open issues, with the Jira MCP server the draft
   names, else whatever Jira tools the chat has. With none, say so and
   stop. The search is theirs and not done:
   `assignee = currentUser() AND statusCategory != Done ORDER BY updated DESC`,
   narrowed to the project when one is named, at most fifty. For each key a
   note carries that the search did not return, read that one issue too: it
   may have been finished or handed on.
3. Put each issue in one group:
   - **linked**: a note carries its key. Compare the issue's status with
     where the card stands. In step, count it and say nothing. Otherwise
     propose the edit: a clause that names the status (`— in review`), or a
     tick when the issue is done.
   - **likely**: no note carries the key, but one is plainly the same work
     by its title or Outcome. Propose the link: `Jira: KEY` under the note's
     `Up:` line, or `jira:` in its frontmatter when the note has frontmatter.
   - **new**: nothing matches. Propose a task and where it lands, with the
     reason: a sub-item under the card whose note carries the issue's parent
     or epic, or whose work it plainly belongs to; else its own card at the
     bottom of the column that fits; else say "not sure" and ask.
4. Answer in this shape, rows numbered:

```
# Jira pull — YYYY-MM-DD

| # | Issue | Jira status | Here | Proposal |
|---|---|---|---|---|
| 1 | KEY-12 · summary | In Review | [[note]], card open | clause: — in review |
| 2 | KEY-15 · summary | To Do | none | new task, sub-item under [[parent note]] (same epic) |

In step: N issues.
Which rows? (all, numbers, or none)
```

5. On the answer, make exactly those rows:
   - a **link** or an **update** changes one line in the note or on the board;
   - a **new task** is a note made from the task template in the folder the
     board's other task notes are in, titled with the issue's summary
     trimmed to a file name, with `Jira: KEY` under its `Up:` line
     (`Up: [[parent]]` for a sub-item; no `Up:` line for a card of its own).
     Acceptance criteria the issue states go under Verify, quoted; Outcome
     stays empty for the user to write. Then its card: `- [ ] [[Title]]` at
     the bottom of the column agreed, or indented as the last sub-item under
     the parent card.
   Then say what was made and changed, one line each.

## Hard rules

- Read only in Jira: never create, comment on, assign or move an issue.
- Propose first, write after a yes. A yes covers the rows named and nothing
  else.
- No twins. Before making a note or a card, search the board (archive
  included) and the notes folder for the key and for the title.
- Bring over only the key, the summary and stated acceptance criteria.
  Descriptions, comments and people's names stay in Jira.
- Never reorder a column, and edit the board as plain text, the lines named
  and no others.
- A guess is labeled. "Likely" and "not sure" rows say what the match rests
  on, so a wrong one is easy to refuse.
