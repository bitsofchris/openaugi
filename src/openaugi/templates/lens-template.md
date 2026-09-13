---
kind: engine   # engine ships as a template on `openaugi init`; personal never does (repo AGENTS.md)
name: lens-template
description: >-
  THE LENS CONTRACT — the authoritative, annotated spec every lens file
  must follow. Copy this file to start a new lens. Writers (agents via
  "new lens", humans by hand) and readers (context pack, openaugi lenses,
  the apply-lens engine) agree on THIS format; change it here and
  test_lens_contract breaks until everything keeps up.
scope: >-
  REQUIRED. The default retrieval recipe, plain prose: what data this
  lens reads (whose writing, which tags/containers, what time window,
  what to exclude). Overridable at apply time — an explicit scope in the
  instruction wins. INPUT AXIS: if the lens is meaningless without a
  subject (distill, echoes, idea-lineage), scope MUST say "topic given
  at apply time" — surfaces use this to know a bare "apply lens NAME"
  isn't a complete instruction.
trigger: on-demand
target: >-
  REQUIRED. Where the artifact lands AND what happens to it on re-run —
  one of three families, three persistence behaviors:
  "dashboard" (MERGE — nominations upserted into View - Dashboard.md by
  ^nom-* anchor; no standalone file; same-anchor rule makes re-runs
  idempotent) · "note — <path pattern>" (ARTIFACT — a NEW dated
  `- [ ] seen` note per run; accumulates, never overwrites) · "view —
  overwrite <View - X.md>" (CACHE — same file overwritten every run;
  only the latest run exists; the only legal overwrite=True). Prose
  after the keyword is welcome. Pick by the question's shape: recurring
  status → view · durable one-off answer → note · structure change →
  dashboard.
---

# <Lens Name>

<!--
CONTRACT RULES (the annotations; delete comments in real lenses):

FRONTMATTER — all five keys REQUIRED:
- name: kebab-case, matches the filename stem. Bare scalar.
- description: ONE line answering "what question does this lens
  answer?" — surfaces (mobile chips, `openaugi lenses`) display it.
- scope / target: see above.
- trigger: `on-demand` | `on-pass` | `every <period>` (e.g. `every 7d` —
  note NO colon: `every: 7d` written bare is invalid YAML; quote it if
  you must use the colon). `every <period>` FIRES on the watcher's tick
  once `tasks.schedule_lenses` is on in config: units `s m h d w`, and a
  period that will not parse is skipped and logged, never guessed at.
  `on-demand` (the default) never auto-fires; `on-pass` is the review
  pass's. A trailing `# comment` is fine.

YAML SAFETY — description/scope/target MUST be folded scalars
(`key: >-`, text indented on the next line) whenever the text contains
a colon+space or starts with a quote. When in doubt, always use `>-`.
`name` and `trigger` stay bare. Validate: `openaugi lenses --check`.

BODY — the intent prose the agent follows. Recommended sections:
## Intent (the question + the bar), ## Process (numbered), ## Hard rules.

## Run — OPTIONAL, and only for a scheduled lens. The two things a
run with nobody to ask needs: prose naming the state to read before
starting (copied verbatim into the task file), and one line
`dedupe: <path>` naming the output that proves today's run already
happened (`{date}` expands to the run date). See docs/reference/lenses.md.

EVERY lens inherits the augi-agent hard rules: never edit notes outside
OpenAugi/ · dashboard/note output is nominate-or-`seen` · only
view targets regenerate silently · one artifact per apply.
-->

## Intent

What this lens finds and the bar it holds.

## Process

1. Resolve scope (spec default unless the apply instruction overrides).
2. Retrieve, judge, condense.
3. Write to the target in its family's grammar.

## Hard rules

- The lens-specific constraints that make its output trustworthy.
