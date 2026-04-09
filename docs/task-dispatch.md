---
name: task-dispatch
description: Obsidian → tmux task dispatch. Optional watcher that picks up `status: pending` task files and launches Claude Code agents in named tmux sessions to work on them.
---

# Task Dispatch — Obsidian to tmux

Write a task in Obsidian (on desktop or mobile). A watcher picks it up and launches a Claude Code agent in a named tmux session to do the work. Attach any time with `tmux attach -t <task_id>` to watch it happen.

This is an **optional add-on**. If you don't run the watcher, `zzz: task` instructions from heartbeat simply land as files in `OpenAugi/Tasks/` and sit there. Nothing bad happens. You opt in by running `openaugi task-dispatch`.

## When to use this

- You want a "go do this" capture path from phone → desktop that doesn't require you to be at your terminal to kick it off.
- You want heartbeat-generated `zzz: task` instructions to actually run, not just be recorded.
- You want parallel agent work — each task lives in its own tmux window, isolated from the others.

## When NOT to use this

- You don't want subprocesses spawning `claude` sessions on your machine unattended — skip the watcher, use heartbeat logs as notes instead.
- You don't have `tmux` or the `claude` CLI installed — the watcher fails loudly at startup if either is missing.

## The single-source contract

The format of a task file is the **one contract** between the writer (heartbeat agent, or you manually, or a mobile capture) and the reader (the watcher). It lives in exactly one place:

```
src/openaugi/templates/task-template.md
```

Change that file and `test_task_template_hydrates_cleanly` in [tests/test_task_watcher.py](../tests/test_task_watcher.py) breaks until the watcher keeps up. Both the heartbeat skill and the watcher docstring point at that template rather than redefine the format.

```
┌──────────────────────────────────────────────────────┐
│  src/openaugi/templates/task-template.md             │ ◀── THE contract
│  (annotated, literal example)                        │
└──────────────────────────────────────────────────────┘
         ▲                              ▲
         │ references                   │ references
         │                              │
┌────────┴───────────┐       ┌──────────┴──────────────┐
│ task_watcher.py    │       │ heartbeat-skill.md      │
│  docstring         │       │  task section           │
│  (reader side)     │       │  (writer side)          │
└────────────────────┘       └─────────────────────────┘
         ▲                              ▲
         │ enforced by                  │ enforced by
         │                              │
┌────────┴──────────────────────────────┴──────────────┐
│ test_task_template_hydrates_cleanly                  │
│ (reads the template, runs the watcher, asserts OK)   │
└──────────────────────────────────────────────────────┘
```

One file defines the format. The watcher and the skill both reference it. One test enforces that whatever's in the template is actually handleable by the watcher. If you change the template in a way the watcher doesn't understand, CI breaks — the contract can't silently drift.

## What a task file looks like

Minimal pending task, before hydration:

```markdown
---
status: pending
workstream: openaugi
repo: openaugi
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
Review README.md in the openaugi repo. Identify where a new user gets
stuck. Rewrite with explicit copy-pasteable commands and a "what success
looks like" line after each step. Verify by following the steps in a
clean clone.

## Human Todo
<!-- Empty to start. Agent appends items here if you're needed. -->

## Results
<!-- Empty to start. Agent fills this in when finished. -->
```

**Required frontmatter:**
- `status: pending` — the watcher only picks up pending files. Flows `pending → active → done` (or `needs-input`). **This is the dispatch lifecycle for the task file, not the same as `status/*` tag facet on blocks** — see [taxonomy.md § disambiguation](taxonomy.md#disambiguation--block-status-tag-vs-task-file-status-field).
- `workstream` — the writer owns this; the watcher does not default it. Holds the **area slug without the `area/` prefix** (e.g., `openaugi`, not `area/openaugi`). See [taxonomy.md](taxonomy.md) for the full area list.

**Optional frontmatter:**
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
- The heartbeat agent doesn't need to know your directory layout — it just writes `repo: my-project`
- Moving a repo is a one-line edit in `Repos.md` instead of a vault-wide find/replace
- Your task files are portable across machines if you keep `Repos.md` per-machine

If the task doesn't need a specific repo (it's editing notes in the vault itself), omit both `repo` and `working_dir`. The watcher will run claude in the vault directory.

## The full flow

```
openaugi task-dispatch → pipeline/task_watcher.py
  poll loop (default every 5s):
  │
  ├── scan_pending(tasks_dir, settle=30s)
  │     → find .md files with `status: pending` stable for 30+ seconds
  │
  ├── for each pending file:
  │   │
  │   ├── hydrate_note(file)
  │   │   → assign task_id (TASK-YYYY-MM-DD-<slug>)
  │   │   → add `created`, `tmux_session` frontmatter
  │   │   → flip `status: pending` → `active`
  │   │   → inject `## Session` block with tmux attach command
  │   │   → inject `## Results` section if missing
  │   │   → rename file to <task_id>.md
  │   │
  │   ├── resolve_working_dir(fm, repo_paths)
  │   │   → check `working_dir` / `working-dir` / `repo` frontmatter
  │   │   → look up short names in `OpenAugi/Repos.md`
  │   │   → fall through to absolute paths
  │   │
  │   ├── build_prompt(task_id, body, links)
  │   │   → small wrapper around the task body itself
  │   │   → adds task_id and finish instructions
  │   │
  │   ├── write_context_file(task_id, prompt) → /tmp/openaugi-tasks/
  │   │
  │   └── launch_tmux(tmux, claude, session_name, context_file, cwd)
  │       → create detached session with `-c <cwd>`
  │       → wait for shell prompt to settle
  │       → send-keys: `claude "$(cat /tmp/openaugi-tasks/task-X-context.md)"`
  │
  └── sleep(poll_interval) and repeat
```

The Python side is deliberately simple. It does not talk to any LLM — the remote Claude session in the tmux window does all the reasoning. The task body IS the prompt; the watcher just wraps it with the task id and a one-line finish instruction.

## How the agent finishes a task

The watcher's prompt tells the remote agent:

> When done, edit `OpenAugi/Tasks/<task_id>.md`: fill in `## Results` and set frontmatter `status: done`.

That's the whole protocol. No custom MCP tool, no callback, no "completion signal." The agent edits the task file directly — which both records the outcome and takes the file out of the `pending` pool on any future scans. If the agent hits something it can't handle, it appends items to `## Human Todo` and sets `status: needs-input`.

## Running it

### Prerequisites

- `tmux` on PATH (`brew install tmux` on macOS)
- `claude` CLI on PATH ([Claude Code](https://claude.com/claude-code))
- An ingested vault (`openaugi init` + `openaugi up`)
- A `heartbeat-skill.md` in your vault if you want the heartbeat agent to generate tasks (see [src/openaugi/templates/heartbeat-skill.md](../src/openaugi/templates/heartbeat-skill.md))
- Optional: `OpenAugi/Repos.md` in your vault with your short-name → path map

### Start the watcher

```bash
# in its own terminal (or a long-lived tmux window)
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
2. Run `openaugi heartbeat --path /path/to/vault`. The heartbeat agent writes `OpenAugi/Tasks/readme-onboarding-fix.md` with `status: pending`, `repo: openaugi`, and a self-contained task body.
3. The watcher (already running) picks it up after 30s, hydrates it to `TASK-2026-04-09-readme-onboarding-fix.md`, flips to `status: active`, and launches `claude` in a detached tmux session named after the task id.
4. Attach with `tmux attach -t TASK-2026-04-09-readme-onboarding-fix` to watch the agent work.
5. When the agent finishes, it edits `## Results` and sets `status: done`. Detach from tmux any time — the session keeps running regardless.

## Writing a task by hand (no heartbeat)

You don't need the heartbeat agent. Any `.md` file you drop into `<vault>/OpenAugi/Tasks/` with `status: pending` and a `## Task` section will work. Minimum viable hand-written task:

```markdown
---
status: pending
workstream: self
repo: my-project
---

# Clean up the test suite

## Task
Remove the three skipped tests in `tests/test_old.py` that have been
dead since the refactor. Make sure `./scripts/check.sh` still passes.
```

Save it, wait ~30s (the settle window), and it dispatches. Useful for mobile capture via Obsidian Sync: open the Tasks folder on your phone, write the file, close the app, and by the time you're back at your desk it's already running.

## Design notes

**Why polling instead of inotify/fswatch?** The watcher uses the same `watchdog` library as [pipeline/watcher.py](../src/openaugi/pipeline/watcher.py) for vault ingest. Polling is simpler here because we already need `settle` logic (inotify fires on every write of a syncing file, which for Obsidian Sync happens dozens of times per save). A 5s poll with a 30s settle is roughly equivalent to "dispatch once the file has stopped changing for half a minute."

**Why tmux?** Detached sessions survive if the watcher or your terminal dies. Named sessions make attaching trivial (`tmux attach -t TASK-...`). You can have ten agent sessions running in parallel, switch between them, and close/kill any individual one. The alternative — a single CI-style log file per task — is harder to interact with if the agent wants to ask a question.

**Why an optional add-on instead of bundled with heartbeat?** Heartbeat is in-process Python reading/writing SQLite and markdown. The task watcher spawns subprocesses with user binaries. Keeping them separate means you can use heartbeat without opting into subprocess dispatch. The heartbeat log still shows you what tasks were generated; you can run them manually by starting the watcher any time.

**Why no defaults for workstream/priority?** V1 of this watcher set `workstream: openaugi` and `priority: now` as defaults during hydration. That's an opinion the public repo shouldn't bake in. The writer (heartbeat agent or you) owns those fields. If `workstream` is missing, the task is logged un-scoped — which is honest.

## Related

- [docs/taxonomy.md](taxonomy.md) — the taxonomy that defines `workstream` values and disambiguates the `status:` frontmatter field from the `status/*` tag facet on blocks
- [src/openaugi/pipeline/task_watcher.py](../src/openaugi/pipeline/task_watcher.py) — the watcher implementation
- [src/openaugi/templates/task-template.md](../src/openaugi/templates/task-template.md) — the authoritative task file format
- [src/openaugi/templates/heartbeat-skill.md](../src/openaugi/templates/heartbeat-skill.md) — the writer side: how the heartbeat agent knows to emit task files
- [tests/test_task_watcher.py](../tests/test_task_watcher.py) — unit + contract tests
- [docs/plans/done/heartbeat.md](plans/done/heartbeat.md) — the "dumb script, smart agent" design that produces tasks in the first place
