---
name: save-thread
user_invocable: true
description: Synthesize and save the current conversation as a thread note in OpenAugi/Threads/.
  Use when the user says "save this thread", "save our conversation", "log this session",
  "save this to augi", or wants to capture what was discussed and learned.
---

# Save Thread

Synthesize the current conversation and write it to OpenAugi via `write_thread`.

## What to capture

Not the transcript — the signal:

- **What was the intent?** The original question or goal behind the conversation
- **How did thinking evolve?** Key pivots, realizations, or direction changes
- **What was decided?** Choices made, approaches agreed on, things ruled out
- **What was learned?** Concrete discoveries, non-obvious insights
- **What's still open?** Unresolved questions, next steps, things to revisit
- **What was produced?** Any documents written, code created, plans made

## Format

Write `content` as markdown. Use headers and bullets. No fixed schema — structure it to fit the session. A short session might be a few bullets. A long one might have sections.

Example structure (adapt as needed):

```markdown
## Intent
[What the user was trying to figure out or accomplish]

## What happened
[How the conversation evolved — key pivots or realizations]

## Decisions
- [Choice made]

## Learned
- [Non-obvious thing discovered]

## Open
- [Unresolved question or next step]

## Produced
- [Title of any document or artifact created]
```

## Steps

1. Re-read the conversation to identify intent, evolution, decisions, and output
2. Write a one-line `description` (goes in frontmatter — make it scannable)
3. Choose a short `topic` that names the session (e.g. "write-back MCP tools design")
4. Call `write_thread` with topic, description, and content
5. Confirm the path written
