---
kind: engine
name: recall-question
description: >-
  "What should I be able to explain about what this chat just did?" — one recall question about the last turn or two, with the claims a complete answer makes.
scope: >-
  This conversation and nothing else: no files are read and no search is run.
trigger: on-demand
target: >-
  note — the Recall card in the Augi pane; once graded, a row in the anchor's learning record (the explain-to-learn format).
---

# recall-question

Write one recall question for me about what the last turn or two of this chat did, so I learn the work instead of only watching it.

- Ask about the why or the mechanism: why a change was made, what was ruled out, how the piece works, what would break if it were missing. Not a file name, a flag or a number to memorise.
- It must be answerable in one or two sentences from this conversation alone.
- Then list the two to four claims a complete answer makes, each one short.

Answer with exactly these lines and nothing before or after them:

Question: the question
Claim: one claim a complete answer makes
Claim: another claim
