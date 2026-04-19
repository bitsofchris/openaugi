---
name: taxonomy
description: The three-facet tag taxonomy (area/type/status) for classifying captured blocks. Single source of truth — referenced by augi-agent.md and task-dispatch.md. Explains what each facet is for, the classification rules, and disambiguates the block-level `status/*` tag from the task-file `status:` frontmatter.
---

# OpenAugi Taxonomy

A tiny, load-bearing tag taxonomy for captured blocks. Three facets, six values total in the default template. Everything else lives in the link graph and embeddings.

## When to use this doc

Read this when you're:

- Customizing `<vault>/OpenAugi/AGENT/augi-agent.md` for your own practice
- Working on task-dispatch and need to know what `workstream:` means or why `status:` and `status/*` aren't the same thing
- Tempted to add a new tag facet (answer: probably not)

## The point of the taxonomy

**Areas are evolution streams, not life domains.** The taxonomy exists to enable two queries the graph can't answer on its own:

1. *"Show me the evolution log of X over time"* → `area/*`
2. *"Show me all active tasks"* → `type/task` + `status/active`

That's it. Topic, theme, and relationship live in the link graph and embeddings, where they're handled better than any tag could.

**Rule of thumb:** if you're not going to query `area/X` later, `area/X` shouldn't exist. If you're not going to filter by `type/Y`, `type/Y` shouldn't exist. The graph handles everything else.

## The three facets

### `area/*` — evolution streams

The stories you're actively following over time. Five or fewer. Add one when you catch yourself wanting its timeline; delete one when you stop caring.

The default template ships with none — areas are personal. A typical set:

- `area/openaugi` — building the tool
- `area/research` — learning, reading, domain research
- `area/self` — reflections and personal evolution
- `area/content` — things going toward publication

Areas can overlap. A block about "OpenAugi research" is legitimately both `area/openaugi` and `area/research`. Prefer one; allow two when both stories genuinely want the block.

### `type/*` — only for things with a lifecycle

One value in the default:

- `type/task` — an actionable item

Everything else is "a captured thought" — an idea, an insight, a reflection, a quote. The graph tells you what it is better than a tag ever would. Don't add `type/idea` or `type/insight` unless you find yourself regularly filtering on them and failing.

### `status/*` — only on `type/task` blocks

Three states:

- `status/active` — you're working on it now
- `status/parked` — looked at, not doing yet (archive-ish)
- `status/done` — completed

Absence of a status tag on a task means *"queued, not yet triaged."* That's a fine default — don't invent `status/next` or `status/someday` unless you feel actual pain from the ambiguity.

**Not the same as the task-file `status:` frontmatter field.** See [disambiguation](#disambiguation--block-status-tag-vs-task-file-status-field) below.

## Classification precedence

When classifying blocks (manually or via the agent), apply these rules in order. **Path beats tags beats content beats guess** — cheapest, most reliable signals first.

1. **Area from source path.** `journals/work/` → `area/work`. Stop if a folder rule matches.
2. **Respect existing facet tags on the block.** If the user already tagged it, trust them.
3. **Type from content shape.** Is it an actionable task? Tag `type/task`. Otherwise, no type tag.
4. **Area from content, as a last resort.** Only when path and tags aren't decisive.
5. **Status only if type is task, and only when obvious.** Default to no status tag. Never guess `parked` or `done`.
6. **Unsure? Tag what you're confident about and flag the rest.** One solid tag > three weak ones.

Agents never modify source notes — classifications are stored as `augi_tags` in block metadata via the `tag_block` MCP tool.

## Disambiguation — block `status/*` tag vs task-file `status:` field

There are **two `status` concepts** in OpenAugi and they live in different places. Don't confuse them.

| Where | Field | Values | Owner | Meaning |
|---|---|---|---|---|
| A block in your notes | `status/*` tag | `active`, `parked`, `done` | You, manually in Obsidian | Your tracking of the work represented by this thought |
| A task file in `OpenAugi/Tasks/` | `status:` frontmatter | `pending`, `active`, `done`, `needs-input` | Watcher + remote agent | Dispatch lifecycle: has this been picked up, is it running, did it finish |

They describe different things:

- The **tag facet** on blocks is your personal, manual tracking for any `type/task` block, whether or not it ever gets dispatched as an agent task.
- The **frontmatter field** on task files is the watcher's state machine for dispatched task files only.

A single task can exist in both places at once — a block in your journal tagged `#type/task #status/active`, linked via `source_block_id` to a task file in `OpenAugi/Tasks/` with frontmatter `status: active` (meaning the agent is currently running in tmux). The two values overlap on vocabulary but not on meaning. The zzz dispatch hook does **not** modify block tags when it writes a task file — source notes are never touched. You manage block-level `status/*` yourself.

## `workstream:` in task frontmatter

The `workstream:` field in [task-template.md](../src/openaugi/templates/task-template.md) holds the **area slug without the `area/` prefix** — e.g., `openaugi`, not `area/openaugi`.

When the zzz dispatch hook writes a task file, you can fill `workstream:` from the block's area. If you write a task file by hand, put your area slug there.

See [task-dispatch.md](task-dispatch.md) for the full dispatch flow.

## What this explicitly gives up

- `type/idea`, `type/insight`, `type/question`, `type/reference`, `type/reflection` — use the graph and semantic search instead.
- `status/next`, `status/should-do`, `status/someday`, `status/blocked` — the GTD priority ladder. Most people don't actually change behavior based on these distinctions.
- `channel/*`, `action/*` — workflow tags you apply manually if at all, not agent-time classifications.

If you later find yourself wishing you had one of these, add it back. But start lean.

## Customizing for your vault

Your personal `<vault>/OpenAugi/AGENT/augi-agent.md` is where you customize agent behavior. The agent reads this skill file every time it runs a task, so changes take effect immediately.
