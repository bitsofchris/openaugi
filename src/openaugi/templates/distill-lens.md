---
name: distill-lens (template)
description: >
  TEMPLATE — copied to <vault>/OpenAugi/AGENT/distill-lens.md on `openaugi init`.
  The vault copy is the live version the agent reads. Edit there, not here.
  On-command, topic-scoped distillation: gather everything about X (agentic
  search + links + clusters, or user-selected context), apply a stated intent,
  and write ONE derived note with full provenance. Use when the user says
  "distill X", "synthesize my thinking on X", "make the concept note for X",
  or provides selected context to distill. NOT for summaries-in-chat
  (just answer those) and NOT the recurring review pass (see review-pass.md).
---

# Distill Lens

A lens = intent applied to a scope. This lens produces a **distillation** —
a curated knowledge artifact, different from a view:

- **View** (review pass): regenerable cache, no review, overwritten each run.
- **Distillation** (this): created once, reviewed by the user, then owned by
  them. It graduates toward silver/gold.

**When NOT to persist:** if the user just wants understanding in the moment,
answer in chat — just-in-time distillation is free and is the default.
Persist only on explicit command. The signal for the user to run this lens
is repetition: re-deriving the same synthesis a third time means save it once.

## Process

1. **State the intent.** One line, from the user's ask: "current
   understanding of X", "the evolution of X over time", "extract the
   nuggets from this working note". If the ask is bare ("distill X"),
   default to *current understanding* — don't interrogate.
2. **Scope.** Gather source blocks:
   - `get_context` / `search` (semantic + keyword) on the topic
   - membership: blocks routed to relevant containers (`get_related`,
     `routed_to`, direction=in), wikilink neighborhoods (`traverse`)
   - or: the user hands you selected context (e.g. from the Obsidian
     plugin) — then that IS the scope; don't expand it uninvited
   - search first for an existing distillation of X — if one exists,
     propose updating it instead of duplicating
3. **Distill without loss of provenance.** Condense to the useful core.
   Every claim traceable. **Voice rules:**
   - The user's own words/thoughts (source/capture, source/chat,
     source/gdrive) may be synthesized freely — that is their voice.
   - Third-party material (source/readwise, source/notebook, web) must be
     attributed inline ("Wang argues…") — never blended into the user's
     voice. External ideas may link and inspire; they may not impersonate.
4. **Write ONE note** via `write_document` to `OpenAugi/Notes/` (concepts)
   or `OpenAugi/Research/` (topic research):
   - frontmatter description = the intent it answers
   - `#status/needs-review` tag in the body
   - wikilinks to every source note (these become graph edges at ingest)
   - footer: `*Distilled YYYY-MM-DD from N blocks. Source block IDs: …*`
   - a **placement nomination** line: "Suggest linking from [[<gold/silver
     note>]] — say the word and I'll draft the link line for you to paste."
     (Never edit the user's notes; placement is theirs.)

## Hard rules

- One topic per run. No vault-wide batch distillation, ever — history is
  harvested incrementally, pulled by live threads (see review-pass.md
  promotion flow), or by this lens one topic at a time.
- Never edit notes outside `OpenAugi/`. Never overwrite an existing
  distillation without being asked to update it.
- Tags only from the user's taxonomy (OpenAugi/AGENT/My Taxonomy.md). Provenance is mandatory.
- The raw sources are never marked, moved, or edited — backlinks from the
  distillation are the "processed" signal.
