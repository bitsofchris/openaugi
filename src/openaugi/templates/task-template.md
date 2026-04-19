---
status: pending
workstream: openaugi
repo: openaugi
source_block_id: 0123456789abcdef
source_note: "[[Journal 2026-04-09]]"
---

# Fix the README onboarding flow

## Context

<!-- The source block content, verbatim. This is what the user wrote that
triggered the task. The remote agent reads this to understand the trigger.
Keep it as-is — do not paraphrase. -->

Need to fix the README onboarding flow before Thursday — users are getting
stuck on the initial install step.

## User instruction

<!-- The exact text of the zzz: task instruction on the source block.
Block-quoted so the agent sees it as a literal user directive. -->

> task in openaugi repo — fix the README onboarding flow before Thursday

## Task

<!-- The self-contained description of what the remote agent should do.
Written by the zzz dispatch hook (or the user) based on the block + the
user's zzz: instruction. Include: what to change, how to verify, and
what "done" looks like. The agent in the tmux session uses THIS section
as the real prompt, so it needs to stand on its own. -->

Review README.md in the openaugi repo. Identify the point where a new
user is most likely to get stuck during initial setup. Rewrite that
section to be clearer, with explicit copy-pasteable commands and a
"what success looks like" sentence after each step. Verify by following
the steps yourself in a clean clone.

## Human Todo

<!-- Empty to start. If the remote agent hits something that requires
a human (deploy approvals, manual testing on a device, a decision the
agent can't make), it appends a checklist of items here and flips
`status: needs-input`. -->

## Results

<!-- Empty to start. When the remote agent finishes, it fills this in
with a summary of what changed and any follow-ups, then sets
`status: done` in frontmatter. -->
