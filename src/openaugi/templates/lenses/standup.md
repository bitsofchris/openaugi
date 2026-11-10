---
kind: engine
name: standup
description: >-
  "What did I do since the last standup, what is next, and what is in the way?" — three short lists, in plain text, ready to paste where the team reads it.
scope: >-
  Today's daily note and the previous working day's (the lines logged in them and the boxes ticked), the chats started in that window in the Agent Logs (## Chats lines dated then, with their goals), and the Journal entries dated in that window in the notes those logs belong to. The window is given at apply time when it is not "since the last working day". Nothing older, and no semantic search.
trigger: on-demand
target: >-
  view — overwrite OpenAugi/Views/View - Standup.md each run (a standup is read once and replaced the next day)
---

# Standup

## Intent

A standup update someone can paste into a channel without editing: what
got done, what is next, what is blocked. It is assembled from what was
logged during the day, so its worth depends on the log. Where the log is
thin it says so; it never fills the gap with plausible work.

## Process

1. Fix the window: from the start of the previous working day to now,
   unless the instruction names another.
2. Read today's daily note and the previous working day's: the logged
   lines (`- HH:MM …`), the boxes ticked, and the boxes still open.
3. Read the Agent Logs changed in the window: under `## Chats`, each
   line is a chat with the day it started and its goal; a line ending
   `(done)` is finished. Read the Journal entries dated in the window in
   the notes those logs belong to.
4. Group what was found by project (the anchor's note), then write:

```
Done
- <project>: <what changed, as an outcome, one line>

Next
- <project>: <the next step, from today's open boxes or a note's Next>

Blocked
- <project>: <what is in the way and who or what it is waiting on>
```

## Hard rules

- Plain text. No wikilinks, no file paths, no note names that mean
  nothing outside the vault, no agent or chat ids.
- At most eight lines in all. Merge small items of one project into one.
- Outcomes, not activity: "the export handles empty files", not "worked
  on the export". Use the words already in the log where they fit.
- Omit Blocked when nothing is. If the window holds nothing logged, say
  "Nothing logged for <day>" and stop.
- Work only: a line in the daily note that is plainly personal stays out.
- Read only. Applied from a chat, answer in the chat.
