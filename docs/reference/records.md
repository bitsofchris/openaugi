---
name: records
description: The generic collection store — three tools (write_record, list_records, update_record) for agent workflow state. openaugi stores it and knows nothing about what it means; schemas live in the caller's prompt, policy in the caller's config.
---

# Records — workflow state, without opinions

**When to use:** an agent workflow needs state that survives between runs —
what a run did, what is awaiting a human's approval, what was already
declined. Also read this before adding any MCP tool, as the test of whether
the tool belongs.

## The boundary

openaugi is a **general** knowledge layer. Other people's vaults use it, with
other conventions.

So there are exactly three tools, and they carry no vocabulary:

| Tool | Does |
|---|---|
| `write_record(collection, record_id, data)` | create or replace |
| `list_records(collection, where?, order?, desc?, limit?)` | read, filtered by exact field matches |
| `update_record(collection, record_id, patch)` | merge fields into an existing record |

A **collection** is a name the caller chose. openaugi does not validate it,
does not know what it holds, and has no schema for it. `"proposals"`,
`"routings"`, `"passes"` are words a *prompt* picked.

> **Schemas live in the prompt. Policy lives in config. Only mechanism lives
> in a tool.**

### Why this exists — a mistake worth not repeating

This replaced eight named tools (`record_pass`, `record_routing`,
`write_proposal`, `list_proposals`, `list_passes`, `list_routings`,
`answer_proposal`, `undo_routing`) added for one user's review workflow. They
were 30% of the entire MCP surface, and one of them validated a hardcoded list
of which *routing rules* were legitimate — compiling one person's routing
policy into a library other vaults are meant to use.

The tell was visible at the time: **eight tools in one sitting, for one
feature.** If a change adds more than one or two tools, it is probably
encoding a workflow rather than extending a capability.

`test_records.py` asserts those eight are gone, so the boundary can't quietly
be recrossed.

## What belongs here, and what does not

| Belongs | Does not |
|---|---|
| Run logs — what an agent did, when, and why | Anything a human wrote |
| Approval queues — proposed, accepted, declined | Blocks, notes, containers |
| Reversal markers — this action was undone | Recaps (they have `recaps`) |
| Anything droppable without losing knowledge | Anything you'd mourn |

**Records are machinery, not knowledge.** Delete the whole table and nothing a
human wrote is lost — that is the test for whether something belongs here.
Knowledge lives in the vault; the vault is the durable artifact.

## Conventions that make it work

**Derive ids from the subject, never randomly.** `promote-silver-notes`, or
`pass-2026-08-20:block-abc`. Writes upsert, so a stable id means re-recording
the same subject updates in place instead of stacking — which is how an
approval queue avoids re-asking the same question on every run. Random ids
turn a queue into a pile.

**Oldest first is usually right** for anything a human answers, so a decision
that has waited three runs isn't buried under one raised this morning. That is
the default; pass `desc` for a log, where newest first is what you want.

**Filtering is equality-only, on top-level fields.** A general store that
grows a query language becomes a database with a worse dialect. Anything
richer, do in the caller — it already has the rows.

**`created_at` survives a rewrite; `updated_at` moves.** So "when was this
first raised" stays answerable after an update, which is what makes the
oldest-first ordering meaningful.

**Absent records error rather than being created.** Updating something that
vanished usually means a stale client; silently creating it hides the bug.

## Worked example — the review pass

The mobile app's review pass uses three collections. **None of this is known
to openaugi**; it lives in the review-pass prompt
(`OpenAugi/AGENT/review-pass.md`) and in the app's own config.

```
collection "passes"      id: pass-2026-08-20
  { window_from, window_to, scanned, left_alone }

collection "routings"    id: pass-2026-08-20:<block_id>:<container>
  { pass_id, block_id, container, rule, undone_at? }

collection "proposals"   id: promote-<subject-slug>
  { pass_id, kind, block_ids, target, payload, why, state }
```

`routed` and `proposed` counts are **not stored** — they are `list_records`
counts. Storing a number you can count is how a number goes wrong.

The set of legitimate values for `rule` is that workflow's policy and is
enforced by *its* config and *its* prompt, not here. openaugi will happily
store `rule: "anything"`, and that is correct: a different vault might route
by folder, by frontmatter, or by five rules instead of three.

## The test for a new MCP tool

Before adding one, ask: **would a different vault, with different
conventions, want this?**

- *Yes* — it's a capability. `get_context`, `apply_routing`, `write_recap`
  all pass.
- *No* — it's a workflow. Give it a collection and put its meaning in a
  prompt.
