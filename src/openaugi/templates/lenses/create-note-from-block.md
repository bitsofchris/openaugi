---
kind: engine
name: create-note-from-block
description: Promote a block (or a selection) into a canonical silver note — gather related blocks around it, review everything, then write one note of dated entries that links back to every source. The bronze→silver transition as a single reviewed gesture.
scope: the seed block(s) plus semantically related blocks from anywhere in the vault. Excludes nothing by path — a canonical note may legitimately gather from OpenAugi/ and _private/ alike.
trigger: on-demand   # from the phone's selection bar, or from an `aaa:` instruction
target: >-
  note — OpenAugi/Notes/<slug>.md (one canonical silver note per apply,
  named for the idea it states)
---

# Create note from block

## Intent

Turn scattered restatements of one idea into a note that states it once and
links back to every block that said it.

This is the **bronze → silver** transition. Bronze is the untagged default —
most blocks. Silver is what you get when nuggets are aggregated into something
canonical, and it is the layer at which a note gets a `description` and starts
*attracting* blocks instead of merely holding them.

## The rule that makes it safe

**This lens writes.** Every other lens returns a reading. Creating a note is a
structure change, and structure changes are never autonomous — so the whole
design is: *propose everything, write nothing, until the user confirms.*

- The seed blocks are **never edited**.
- Related blocks are **never edited** — they are *routed* to the new note,
  which is a reversible edge.
- Nothing exists in the vault until the confirm step.

## How it runs

1. **Seed** — the blocks the user selected, verbatim.
2. **Related** — semantic candidates around the seed. Nothing accepted by
   default; the user checks what belongs. **Adopt-before-create sits above
   this list**: if a note already covers the idea, offer to append a dated
   entry instead. A lens that skips this fragments the vault one note at a
   time.
3. **Title + description** — proposed, editable. The description is the
   routing rule; a note without one is a dead file rather than a target.
4. **Confirm** — the path, the entry count, the blocks that will be routed.

## What it writes

```markdown
---
description: <the routing rule>
---
- [ ] seen

#note-type/moc #ai-generated

### 2026-08-05

<block content>

— [[2026-08-05]]

### 2026-01-02

<block content>

— [[AMOC - Open Augi Journal]]
```

Newest entries append at the end, so the note reads chronologically and grows
as the idea recurs. Every entry links back to the note the block really lives
in — the silver note is a *view over* blocks, never a replacement for them.

## Three answers, always: yes / no / discuss

**Creating a note is a structure change, so it takes the same three answers
the Dashboard takes.** Never just do it — even when the `aaa:` reads like an
instruction, offer the three and say which you'd pick:

- **Yes** — write it, as above.
- **Discuss** — hold it, and say what the open question is. Carry the
  assembled proposal (title, description, the blocks it would gather) so the
  conversation starts from the work, not from scratch. *"Worth discussing" is
  not "no"* — it's the answer with the most information in it, and dropping it
  is how a review becomes a rubber stamp.
- **No** — record it in the declined list with a reason, so it is not
  renominated without new evidence.

The mobile sheet offers exactly these three on its confirm screen. Match it.

## Where deduplication lives — the head, not the entries

**Do not merge entries.** Five blocks that say the same thing become five
entries, and that is correct: the entry log is append-only ground truth and
editing it would destroy provenance to produce something you get for free
elsewhere.

The deduplicated statement is the note's **head** — its recap, which you
maintain over the whole log with `write_recap` and which the app renders above
the entries. See `docs/reference/recap-spec.md`: section 1, "current
understanding", *is* the deduplicated representation of the thread, and for a
concept note you re-derive it from the container's whole membership rather
than just this pass's arrivals.

So: the entries are the back of the index card, the recap is the front. When
you create a note with this lens, **write its first recap in the same pass** —
a silver note without a head is only half of one.

## From an `aaa:` instruction

The same operation, typed:

| Said in capture | Means |
|---|---|
| `aaa: this belongs in [[X]]` | adopt — append a dated entry to X, route the block |
| `aaa: I know there's a note like this` | adopt-before-create; surface candidates and ask |
| `aaa: make this canonical` | the full flow above |

**Use the same promote path the app uses** (`POST /promote` on the mobile
bridge, or `write_document` + `apply_routing` directly with the exact shape
above). Two implementations of "create a silver note" will drift within a
month, and the one that drifts is the one that stops linking back.
