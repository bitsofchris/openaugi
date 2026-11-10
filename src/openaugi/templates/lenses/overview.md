---
kind: engine
name: overview
description: >-
  "What is open across everything I am working on, and what is next?" — every active anchor on the board with its open chats, its open tasks and its one next step, on one screen.
scope: >-
  The board note (every open card above the archive that links a note, and its linked sub-items), each of those notes (Outcome, Next, the latest Journal entry), and each one's Agent Log (## Chats, ## Questions). [[Slowly Changing Context]] for what this season is for, and today's daily note when the instruction or the draft names one. No semantic search: this lens reads state, it does not go looking for ideas.
trigger: on-demand
target: >-
  view — overwrite OpenAugi/Views/View - Overview.md each run (an overview is a cache of the current state, never an archive)
---

# Overview

## Intent

One screen that answers "what am I in the middle of, and what do I do
next?" for someone running several agent chats on several things at once.
It is a mirror of the state already written down: the board, the notes,
the Agent Logs. It never invents a priority that the notes do not state.

## Process

1. Read the board. An **anchor** is an open card above the archive that
   links a note; an indented checkbox under it that links a note is one
   of its **tasks**. Keep the column each anchor sits in.
2. For each anchor and each of its tasks, read:
   - the note: its Outcome (a heading or `outcome:`), its open `# Next`
     checkboxes when it has that section, and the date of its latest
     Journal entry;
   - its Agent Log, `Agent Log - <note name>.md` in the Agent Log folder
     (`OpenAugi/Agent-Logs` unless the draft names another): under
     `## Chats`, each line is one chat with the day it started and its
     goal, and a line ending `(done)` or `(hidden)` is finished; under
     `## Questions`, an unticked line is an open question.
3. With more than twelve anchors, keep the ones whose note or Agent Log
   changed in the last 14 days and count the rest in one closing line.
4. Read [[Slowly Changing Context]] for the season's focus, and today's
   daily note when one was named, for what the day was meant for.
5. Answer in this shape:

```
# Overview — YYYY-MM-DD

**Next three:** the three smallest next steps across everything, each
naming its anchor and, in a few words, why it is first (it is in today's
note, it serves the season's focus, a chat is waiting on it).

| Anchor | Outcome | Open chats | Open tasks | Next step |
|---|---|---|---|---|
| [[note]] | one line | N: their goals | N | the first open Next item, or the newest chat's goal |

**Waiting on you:** open questions in the Agent Logs, each with its anchor.

**Gone quiet:** anchors with open chats or tasks and nothing written in
their note or log for 7 days or more, oldest first.

**Loose ends:** open chats still titled `chat` (no goal recorded), and
tasks whose Outcome is empty.
```

## Hard rules

- One screen: the table plus at most twelve more lines. Omit a section
  that would be empty.
- Every anchor is a link to its note. Every claim comes from a file read
  in step 2; when a note has no Outcome or no Next, say "not written"
  instead of guessing one.
- Mirror, not coach. "Next three" are picked from next steps the notes
  already state; rank them, do not author them, and do not moralize
  about what has gone quiet.
- Read only. Applied from a chat, answer in the chat; the view is written
  only by a scheduled or review-pass run.
