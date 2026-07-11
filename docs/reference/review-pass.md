---
name: review-pass
description: The write-back loop — routes new blocks into containers via augi_tags (DB-only), regenerates derived view notes in OpenAugi/Views/, and nominates structure changes on the Dashboard. Covers how augi_tags work, the capture grammar (qqq/zzz/aaa), how to run a pass, and what the human does.
---

# Review Pass

## When to use this doc

- You want to know what `augi_tags` are and where classifications live
- You're about to run (or debug) a review pass
- You forgot the capture grammar or what the Dashboard nominations mean

**Source-of-truth hierarchy (don't duplicate procedure across these):**

1. [src/openaugi/templates/review-pass.md](../../src/openaugi/templates/review-pass.md)
   — authoritative for pass *behavior* (the step-by-step, view shape,
   nomination format). Copied to the vault by `openaugi init`.
2. `<vault>/OpenAugi/AGENT/review-pass.md` — the live prompt the agent
   actually reads: the template plus the user's own registry seeds.
3. This doc — *concepts only*: why the pass works the way it does. If you
   find procedure here that's also in the template, delete it here.

Design record with full rationale: [docs/plans/review-pass-v1.md](../plans/review-pass-v1.md);
v2 refinements (unified registry rule, tiers, reference handling):
[docs/plans/review-pass-v2-workstreams.md](../plans/review-pass-v2-workstreams.md).

## The frame in one paragraph

Event sourcing applied to a vault. **Truth** is the user's own writing — append-only,
never edited by agents. **Views** are agent-generated markdown files under
`OpenAugi/Views/` — derived, regenerable caches that need no review (a wrong view
is a stale cache entry, not damage). Human review gates only **structure changes**:
new tags/areas, promoting a block cluster to a new note, merges. The agent
nominates on the Dashboard; the human commands; the agent assembles.

## Tags vs. routing — two mechanisms, one taxonomy

**Classification is tags; membership is links. Never conflate them.**

**Tags — one closed vocabulary, two authors.** There is exactly one taxonomy
(the user's `My Taxonomy`). What differs is who applied a tag:

| Layer | Where it lives | Who writes it |
|---|---|---|
| `block.tags` | Parsed from the user's markdown (`#area/self` etc.) | The user — ground truth |
| `block.metadata["augi_tags"]` | SQLite only — **never written into notes** | The agent, via `tag_block`, using the *same* taxonomy vocabulary |

`augi_tags` is not a second tagging system — it is "tags the agent applied,"
kept out of the user's files. The agent never invents a tag or facet, and
never re-tags a block the user already tagged; it only fills gaps.
`search(tags=[...])` matches both layers. **Untagged is a valid state** —
life-log blocks (daily memories) usually carry no tags at all; tag only what
you'd query.

**Routing — `routed_to` links.** The `apply_routing` MCP tool records "this
block belongs to that container" as a link in the DB — and removes the link
when a route was wrong (a block can route to many containers; each decision
carries `add` and/or `remove` container lists). Views distill a container's
routed blocks. Unrouted is also a valid state — route only what a view should
distill.

**Membership = containment ∪ routing.** A block the user pasted into a
container and a block the agent routed there are the same fact: member of
that container. Routing a block to its own source note is a no-op
(`already_home`), and `get_members` returns the unified list with each
member marked contained/routed. Design record:
[views-as-rendered-queries.md](../plans/views-as-rendered-queries.md).

Both live in the DB only, which is why routing costs nothing and is always
correctable: the user's files never change.

## The registry is the routing map

**One rule: a note is a registered routing target iff it has a container
tag AND a filled `description` frontmatter.** Container tags:
`#note-type/amoc` (areas), `#note-type/pmoc` + `#status/active` (projects),
`#note-type/moc` (concept notes — a facet of an area/project, an evolving
idea; the permanent home for "I've said this before" captures). The registry
is discovered per pass by tag search — no hand-maintained list anywhere.

The `description` says *when to route here* — skill-file style. Human-owned
frontmatter on container notes = the map the router reads; generated
frontmatter under `OpenAugi/Views/` = agent output. A tagged note without a
description is not an inference target (explicit `aaa:`/link routing still
works); the pass nominates a **drafted description as a paste-line** —
pasting it is what registers the note.

**Adopt before create:** when a block cluster earns promotion, the pass
first searches for an existing note that already is the canonical home and
upgrades it (tag + description paste-line) rather than minting a duplicate;
only if nothing exists does it create a concept note fresh. Either way it
sweeps in older related blocks — resurfacing is repeated, not one-shot.

**Reference material** (Snipd, Readwise, and other synced imports) routes at
document granularity — one artifact, one `routed_to` link; never per-block
decisions over a transcript, and reference files are never moved or edited.

## Capture grammar

Three tokens, written anywhere in a note:

- `qqq` — block delimiter (splits a note into blocks at ingest)
- `zzz: <instruction>` — agent dispatch: spawn a task (see [task-dispatch.md](task-dispatch.md))
- `aaa: <instruction>` — routing instruction to the review pass:
  "aaa: route to OpenAugi Mobile", "aaa: find my note on X and link this".
  **Not processed at ingest** — unlike `zzz:`, no watcher acts on it. It
  stays as plain text in the block and is read only when a review pass runs.

Routing precedence: `aaa:` > explicit `[[MOC link]]`/`#area/*` in the block >
location (a block in a MOC's journal is home) > inference (most specific
container wins) > unrouted (Dashboard). Routing ≠ surfacing — every block
routes, views surface only what's salient.

## Running a pass

The pass is an agent skill, not a pipeline. In any Claude session with the
openaugi MCP server: say **"run the review pass"** (or dispatch
`zzz: run the review pass`). Saying **"process the dashboard"** executes
your nomination answers without advancing the high-water mark.

The loop in one line: process Dashboard answers → pull new blocks since the
high-water mark → route the batch (`apply_routing`) → regenerate touched
views + Dashboard → refresh the context pack → advance the mark.

**The step-by-step procedure, view shape, and nomination format live in the
[template](../../src/openaugi/templates/review-pass.md)** — that file is the
behavior spec; this doc doesn't restate it.

Repo-side pointers the template doesn't carry: the high-water mark is the
`meta` table keys `review_pass_last_run` / `review_pass_last_summary`; the
context pack's shape is pinned by the mobile repo's `shared/contract.ts`
(`ContextPack`), builder in `src/openaugi/pipeline/context_pack.py`.

Cadence: manual, attached to the Sunday weekly plan. Schedule only after it's
boringly reliable.

## Lenses: views vs. distillations

A **lens** = an intent applied to a scope, producing a derived artifact.
Two lens families exist, with different trust levels:

| | View (review pass) | Distillation (distill lens) |
|---|---|---|
| Trigger | Scheduled / "run the review pass" | On command: "distill X" |
| Scope | Container's routed blocks since last run | Topic (agentic search + links) or user-selected context |
| Output | `OpenAugi/Views/` — regenerable cache, no review | One note in `OpenAugi/Notes|Research/`, `#human-review`, wikilinked provenance |
| Lifecycle | Overwritten every run | Created once, user reviews, graduates toward their curated notes |

Default is neither: **just-in-time distillation in chat** (retrieve + answer,
persist nothing). Persist a distillation only on reuse — the signal is
re-deriving the same synthesis repeatedly. No vault-wide batch distillation:
history is harvested incrementally, pulled by live threads (promotion flow)
or one topic at a time (distill lens). Agent instructions:
`<vault>/OpenAugi/AGENT/distill-lens.md`.

## Obsidian setup (one-time)

1. **Transclude each container's view** into the container note:
   `![[View - <container title>]]` (agent-created containers get this at
   birth; hand-made ones need the paste once). The Dashboard can be embedded
   too — e.g. into a home/Current-Focus note via `![[View - Dashboard]]`.
2. **Hide the Views folder from search/switcher:** Settings → Files and links
   → Excluded files → add `OpenAugi/Views/`. Views remain browsable in the
   file explorer, but stop appearing in Quick Switcher and search — they
   exist to be embedded, not visited. Invariant: **every view is transcluded
   somewhere**; a view with no embed home is a smell the pass should flag.
3. **Renames:** if a container note is renamed, the next pass regenerates its
   view under the new title and deletes the stale view file (views are
   caches — deleting them is always safe).

## Triggering — the task file is the API

Any of these fire a pass; they all converge on the same mechanism:

- Say **"run the review pass"** / **"process the dashboard"** in a Claude
  session with the openaugi MCP.
- Write `zzz: run the review pass` in any note — dispatch writes a task file
  to `OpenAugi/Tasks/`, the task watcher launches a tmux Claude session.
- Run `openaugi review` — the CLI writes that same task file.
- (In progress, plugin repo) **Obsidian plugin commands** — also just write
  the task file. Anything that can create a markdown file can trigger the
  system; the task-file contract is the integration point. Mobile needs no
  trigger surface of its own: a `zzz:` line in a captured block dispatches
  after ingest.

## What the human does

- **One-time:** the Obsidian setup above.
- **Weekly:** read `View - Dashboard.md`; answer the gravity nominations
  (inline note or zzz) — that's the entire review burden.
- **Anytime:** correct a bad route by telling the agent ("that block doesn't
  belong in Meta — it's Content"); the agent fixes it with one `apply_routing`
  decision (`remove` the wrong container, `add` the right one). Use `aaa:` on
  future captures to pre-empt. Wrong routes are tuning signal.
