---
kind: engine
name: chat-state
description: >-
  "Where does this chat stand?" — what it is doing, what the last few turns did, the next step and whether it is waiting on the user, in four lines.
scope: >-
  This conversation and nothing else: no files are read and no search is run.
trigger: on-demand
target: >-
  note — under this chat's own entry in the folded Chats list of the anchor note's Agent Log section, replaced each time; with no anchor the answer stays in the pane.
---

# chat-state

Say where this chat stands, for me coming back to it after working in other chats. Go by the last few turns, not the whole history. Answer with exactly these four lines, each one short, and nothing before or after them:

Doing: what this chat is working on right now
Last: what the last few turns did or found, with the result if one was checked
Next: the one next step
Waiting on me: yes and what for, or no

Say only what the conversation shows. Mark anything that is a guess "(unverified)".
