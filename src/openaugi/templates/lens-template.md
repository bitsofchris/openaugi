---
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
  instruction wins.
trigger: on-demand
target: >-
  REQUIRED. Where the artifact lands, one of three families —
  "dashboard" (nominations in checkbox+anchor grammar) · "note — <path
  pattern>" (one #human-review note) · "view — overwrite <View - X.md>"
  (regenerable cache). Prose after the keyword is welcome.
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
  you must use the colon). Scheduled triggers stay DORMANT until the
  review pass activates them. A trailing `# comment` is fine.

YAML SAFETY — description/scope/target MUST be folded scalars
(`key: >-`, text indented on the next line) whenever the text contains
a colon+space or starts with a quote. When in doubt, always use `>-`.
`name` and `trigger` stay bare. Validate: `openaugi lenses --check`.

BODY — the intent prose the agent follows. Recommended sections:
## Intent (the question + the bar), ## Process (numbered), ## Hard rules.

EVERY lens inherits the augi-agent hard rules: never edit notes outside
OpenAugi/ · dashboard/note output is nominate-or-#human-review · only
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
