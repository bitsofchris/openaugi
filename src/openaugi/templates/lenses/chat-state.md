---
kind: engine
name: chat-state
description: >-
  "Where does the work in this chat stand?" — goal, last verified result, next step and whether it is waiting on the user, in four lines.
scope: >-
  This conversation and nothing else: no files are read and no search is run.
trigger: on-demand
target: >-
  note — the State block at the top of the anchor note's Agent Log section, overwritten each time; with no anchor the answer stays in the pane.
---

# chat-state

Say where the work in this conversation stands right now, for someone coming back to it cold. Answer with exactly these four lines, each one short, and nothing before or after them:

Goal: what this chat is trying to get done
Verified: the last result that was actually checked (a test run, a command and its output), or "nothing yet"
Next: the one next step
Waiting on me: yes and what for, or no

Say only what the conversation shows. Mark anything that is a guess "(unverified)".
