---
name: task-dispatch
description: ZZZ dispatch + task watcher. Blocks with zzz instructions become task files that get launched as Claude Code agents in named tmux sessions.
---

# Task Dispatch — ZZZ to tmux

Write `zzz: <instruction>` in your notes. The file watcher ingests the block, the dispatch hook writes a task file, and the task watcher launches a Claude Code agent in a named tmux session. Attach any time with `tmux attach -t <task_id>` to watch it work.

Task dispatch runs as part of `openaugi up` by default. Pass `--no-agent` to disable it. You can also run it standalone with `openaugi task-dispatch`.

## When to use this

- You want a "go do this" capture path from Obsidian (phone or desktop) that triggers agent work automatically.
- You want `zzz:` instructions to actually run, not just be recorded.
- You want parallel agent work — each task lives in its own tmux window, isolated from the others.

## When NOT to use this

- You don't want subprocesses spawning `claude` sessions on your machine unattended — pass `--no-agent` to `openaugi up`.
- You don't have `tmux` or the `claude` CLI installed — the watcher fails loudly at startup if either is missing.

## The flow

```
you write `zzz: research deep learning` in a note
  → file watcher detects change (30s debounce)
  → ingest: parse → split → blocks + tags + links → SQLite
  → pipeline/dispatch.py: block has zzz_instructions
    → writes OpenAugi/Tasks/<slug>-<timestamp>.md (status: pending)
  → task watcher picks it up (5s poll, 30s settle):
    → hydrate: assign task_id, flip status→active, inject ## Session
    → resolve working dir via OpenAugi/Repos.md
    → build prompt: augi-agent skill + task body + linked notes
    → launch tmux: detached session + `claude "$(cat ctx)"`
  → agent reads skill file, uses MCP tools, does the work
  → writes output to OpenAugi/ tagged #human-review
  → marks task file status: done
```

## The single-source contract

The format of a task file is the **one contract** between the writer (`pipeline/dispatch.py`, or you manually, or a mobile capture) and the reader (the task watcher). It lives in exactly one place:

```
src/openaugi/templates/task-template.md
```

Change that file and `test_task_template_hydrates_cleanly` in [tests/test_task_watcher.py](../tests/test_task_watcher.py) breaks until the watcher keeps up.

```
┌──────────────────────────────────────────────────────┐
│  src/openaugi/templates/task-template.md             │ ◀── THE contract
│  (annotated, literal example)                        │
└──────────────────────────────────────────────────────┘
         ▲                              ▲
         │ references                   │ references
         │                              │
┌────────┴───────────┐       ┌──────────┴──────────────┐
│ task_watcher.py    │       │ pipeline/dispatch.py    │
│  (reader side)     │       │  (writer side)          │
└────────────────────┘       └─────────────────────────┘
         ▲
         │ enforced by
         │
┌────────┴─────────────────────────────────────────────┐
│ test_task_template_hydrates_cleanly                  │
│ (reads the template, runs the watcher, asserts OK)   │
└──────────────────────────────────────────────────────┘
```

## What a task file looks like

Minimal pending task, before hydration:

```markdown
---
status: pending
source_block_id: 0123456789abcdef
source_note: "[[Journal 2026-04-09]]"
---

# Fix the README onboarding flow

## Context
Need to fix the README onboarding flow before Thursday — users are
getting stuck on the initial install step.

## User instruction
> task in openaugi repo — fix the README onboarding flow before Thursday

## Task
Process the user instruction(s) above in the context of the source block.

## Human Todo
<!-- Empty to start. Agent appends items here if you're needed. -->

## Results
<!-- Empty to start. Agent fills this in when finished. -->
```

**Required frontmatter:**
- `status: pending` — the watcher only picks up pending files. Flows `pending → active → done` (or `needs-input`). **This is the dispatch lifecycle for the task file, not the same as `status/*` tag facet on blocks** — see [taxonomy.md § disambiguation](taxonomy.md#disambiguation--block-status-tag-vs-task-file-status-field).

**Optional frontmatter:**
- `workstream` — area slug without `area/` prefix (e.g., `openaugi`)
- `repo` — short name resolved via `OpenAugi/Repos.md` to a working directory
- `working_dir` / `working-dir` — absolute path, for one-off tasks not in the repo map
- `source_block_id`, `source_note` — provenance back to the block that triggered the task
- `task_id`, `created`, `tmux_session` — added by the watcher during hydration (don't set these yourself)

See [src/openaugi/templates/task-template.md](../src/openaugi/templates/task-template.md) for the annotated authoritative version.

## The Repos.md map

Short names in `repo:` get resolved to absolute paths by `<vault>/OpenAugi/Repos.md`:

```markdown
---
repos:
  openaugi: /Users/me/repos/openaugi
  my-site: /Users/me/repos/my-site
  notes: /Users/me/notes
---

# Repos

Short-name → absolute-path mapping used by the task watcher.
```

Benefits of short names over absolute paths:
- Task files don't need to know your directory layout — just write `repo: my-project`
- Moving a repo is a one-line edit in `Repos.md` instead of a vault-wide find/replace
- Your task files are portable across machines if you keep `Repos.md` per-machine

If the task doesn't need a specific repo (it's editing notes in the vault itself), omit both `repo` and `working_dir`. The watcher will run claude in the vault directory.

## The agent skill file

The agent's behavior is governed by `<vault>/OpenAugi/AGENT/augi-agent.md`. This is a markdown file you edit in Obsidian — it tells the agent what MCP tools it has, how to handle different task types (research, code work, freeform), and hard rules (never modify raw notes, tag everything with #human-review).

The template lives at `src/openaugi/templates/augi-agent.md` in the repo and is copied to the vault by `openaugi init`. The task watcher injects a reference to this file at the top of every agent prompt.

## How the agent finishes a task

The prompt tells the remote agent:

> When done, edit `OpenAugi/Tasks/<task_id>.md`: fill in `## Results` and set frontmatter `status: done`.

That's the whole protocol. No custom MCP tool, no callback, no "completion signal." The agent edits the task file directly — which both records the outcome and takes the file out of the `pending` pool on any future scans. If the agent hits something it can't handle, it appends items to `## Human Todo` and sets `status: needs-input`.

## Running it

### Prerequisites

- `tmux` on PATH (`brew install tmux` on macOS)
- `claude` CLI on PATH ([Claude Code](https://claude.com/claude-code))
- An ingested vault (`openaugi init` + `openaugi up`)
- Optional: `OpenAugi/AGENT/augi-agent.md` in your vault (created by `openaugi init`)
- Optional: `OpenAugi/Repos.md` in your vault with your short-name → path map

### Via `openaugi up` (default)

Task dispatch runs automatically as part of `openaugi up`. No extra terminal window needed.

### Standalone

```bash
openaugi task-dispatch --path "/path/to/your/vault"
```

Options:

| Flag | Default | What |
|------|---------|------|
| `--path, -p` | config default | Vault path |
| `--tasks-folder` | `OpenAugi/Tasks` | Relative folder to watch |
| `--repos-note` | `OpenAugi/Repos.md` | Relative path of the repos map |
| `--interval` | `5.0` | Poll interval in seconds |
| `--settle` | `30.0` | Seconds a file must be unchanged before dispatching |
| `--verbose, -v` | off | Debug logging |

The `--settle` window exists so Obsidian Sync (or any other file-based writer) has time to finish before the watcher grabs a file mid-write. For local-only flows you can lower it.

### End-to-end example

1. Write a daily note block:
   ```
   Need to fix the README onboarding flow before Thursday
   zzz: task in openaugi repo
   ```
2. The file watcher detects the change, ingests the block, and `dispatch.py` writes `OpenAugi/Tasks/task-in-openaugi-repo-20260419-143022.md` with `status: pending`.
3. The task watcher picks it up after 30s, hydrates it to `TASK-2026-04-19-task-in-openaugi-repo.md`, flips to `status: active`, and launches `claude` in a detached tmux session.
4. Attach with `tmux attach -t TASK-2026-04-19-task-in-openaugi-repo` to watch the agent work.
5. When the agent finishes, it edits `## Results` and sets `status: done`. Detach from tmux any time — the session keeps running regardless.

## Writing a task by hand (no zzz)

You don't need zzz instructions. Any `.md` file you drop into `<vault>/OpenAugi/Tasks/` with `status: pending` and a `## Task` section will work. Minimum viable hand-written task:

```markdown
---
status: pending
repo: my-project
---

# Clean up the test suite

## Task
Remove the three skipped tests in `tests/test_old.py` that have been
dead since the refactor. Make sure `./scripts/check.sh` still passes.
```

Save it, wait ~30s (the settle window), and it dispatches. Useful for mobile capture via Obsidian Sync: open the Tasks folder on your phone, write the file, close the app, and by the time you're back at your desk it's already running.

## Design notes

**Why polling instead of inotify/fswatch?** Polling is simpler because we already need `settle` logic (inotify fires on every write of a syncing file). A 5s poll with a 30s settle is roughly equivalent to "dispatch once the file has stopped changing for half a minute."

**Why tmux?** Detached sessions survive if the watcher or your terminal dies. Named sessions make attaching trivial (`tmux attach -t TASK-...`). You can have ten agent sessions running in parallel, switch between them, and close/kill any individual one.

**Why no defaults for workstream/priority?** The writer (zzz dispatch or you) owns those fields. If `workstream` is missing, the task is logged un-scoped — which is honest.

## Related

- [docs/taxonomy.md](taxonomy.md) — the taxonomy that defines `workstream` values and disambiguates the `status:` frontmatter field from the `status/*` tag facet on blocks
- [src/openaugi/agents/task_watcher.py](../src/openaugi/agents/task_watcher.py) — the watcher implementation
- [src/openaugi/pipeline/dispatch.py](../src/openaugi/pipeline/dispatch.py) — the zzz dispatch hook (writer side)
- [src/openaugi/templates/task-template.md](../src/openaugi/templates/task-template.md) — the authoritative task file format
- [src/openaugi/templates/augi-agent.md](../src/openaugi/templates/augi-agent.md) — the agent skill file template (factory default, copied to vault on init)
- [tests/test_task_watcher.py](../tests/test_task_watcher.py) — unit + contract tests
