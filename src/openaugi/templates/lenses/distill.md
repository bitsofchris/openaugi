---
kind: engine
name: distill
description: Gather everything about a topic (or user-selected context), apply a stated intent, write ONE curated note with full provenance. Created once, reviewed by the user, graduates toward silver/gold.
scope: at apply time — a topic (agentic search + links + clusters) or context the user hands over (e.g. a selection from the plugin). Never expand a handed scope uninvited.
trigger: on-demand
target: note
---

# Distill

## Intent

Produce a **distillation** — a curated knowledge artifact, different from
a view:

- **View** (review pass): regenerable cache, no review, overwritten each run.
- **Distillation** (this): created once, reviewed by the user, then owned
  by them. It graduates toward silver/gold.

**When NOT to persist:** if the user just wants understanding in the
moment, answer in chat — just-in-time distillation is free and is the
default. Persist only on explicit command. The signal to run this lens is
repetition: re-deriving the same synthesis a third time means save it once.

## Process

1. **State the intent.** One line, from the user's ask: "current
   understanding of X", "the evolution of X over time". If the ask is bare
   ("distill X"), default to *current understanding* — don't interrogate.
2. **Scope.** Gather source blocks:
   - `get_context` / `search` (semantic + keyword) on the topic
   - membership: blocks routed to relevant containers (`get_related`,
     `routed_to`, direction=in), wikilink neighborhoods (`traverse`)
   - or: the user hands you selected context — then that IS the scope
   - search first for an existing distillation of X — if one exists,
     propose updating it instead of duplicating
3. **Distill without loss of provenance.** Condense to the useful core.
   Every claim traceable. **Voice rules:**
   - The user's own words (source/capture, source/chat, source/gdrive)
     may be synthesized freely — that is their voice.
   - Third-party material (source/readwise, source/notebook, web) must be
     attributed inline ("Wang argues…") — never blended into the user's
     voice.
4. **Write ONE note** via `write_document` to `OpenAugi/Notes/` (concepts)
   or `OpenAugi/Research/` (topic research):
   - frontmatter description = the intent it answers
   - `- [ ] seen` checkbox as the first line of the body
   - wikilinks to every source note (graph edges at ingest)
   - footer: `*Distilled YYYY-MM-DD from N blocks. Source block IDs: …*`
   - a **placement nomination** on the Dashboard (standard grammar):
     "Suggest linking from [[<gold/silver note>]]" — never edit the
     user's notes; placement is theirs.

## Hard rules

- One topic per run. No vault-wide batch distillation, ever.
- Never edit notes outside `OpenAugi/`. Never overwrite an existing
  distillation without being asked to update it.
- Tags only from [[My Taxonomy]]. Provenance is mandatory.
- The raw sources are never marked, moved, or edited — backlinks from the
  distillation are the "processed" signal.
