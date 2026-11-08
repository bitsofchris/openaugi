---
name: augi-ui
description: augi-ui, the Claude Code pane over an OpenAugi vault — anchor on a board card, pick the context a chat gets, run lenses, and keep an Agent Log per anchor. Drafts prompts only; the user presses enter.
---

# augi-ui — the pane beside your agent chat

## When to use

You work with an agent (Claude Code, in the terminal or the desktop app) and
your projects, context and lenses live in a folder of markdown notes. augi-ui
keeps that folder one click away from the chat: pick what you are working on,
choose what the agent should read, run a saved lens, and see the chats and
questions that belong to that project.

It does not need the OpenAugi engine. With OpenAugi installed, chats can also
search the notes through its MCP tools.

## Install

```
/plugin install augi-ui --marketplace bitsofchris/openaugi
```

Then `/augi` opens the pane.

## How it works

**Setup.** The first `/augi` walks through settings (stored by Claude Code, also
in `/config` under `augi-ui`; `/augi setup` runs it again):

| Setting | What it is | Default |
|---|---|---|
| `notesFolder` | The notes folder; Obsidian's known vaults are offered as buttons | — |
| `editor` / `editorCommand` | What opens a note: obsidian, vscode, cursor, system, or a custom command with `{file}` | `obsidian` |
| `boardNote` | The Kanban note whose linked cards are the anchors | `Kanban.md` |
| `contextNote` | A note about the user and this season | `OpenAugi/AGENT/Slowly Changing Context.md` |
| `lensFolder` | One markdown file per lens | `OpenAugi/AGENT/lenses` |
| `logFolder` | One Agent Log per anchor | `OpenAugi/Agent-Logs` |
| `lensPicks` | Lenses listed first, comma separated; empty lists all | — |

Each missing file can be created as a starter, pointed at an existing note, or
skipped. Starters are never written over an existing file.

**The pane.**

- **Anchor** (`a`): a linked card from the board — a project or a task. Its
  open sub-items show as tasks; `p` opens the note you write, `k` the Agent Log.
- **Context**: what the next chat is told to read — the anchor's notes, pinned
  notes, files, links and coding sessions (`h` pins a recent Claude or Codex
  session), the context note and board (`m`), and OpenAugi search (`o`).
- **Draft** (`n`): fills the prompt with that context. Nothing is sent until
  the user presses enter.
- **Lenses** (`l`): run a lens with the same context; `g` drafts a new one.
  Every lens in the lens folder is listed; the `lensPicks` setting narrows and orders
  the menu. Lenses follow the OpenAugi lens contract (`OpenAugi/AGENT/lens-template.md`).
- **Agent Log**: per anchor, a companion note that holds the pinned context,
  open questions and the chats started from the pane. The user's own note is
  never written; `⧉` copies an embed line to paste into it.

## Developing

The authority is Claude Code's own guide to writing plugins, the built-in
`plugin-authoring` skill (ask Claude to load it). It ships with Claude Code and
matches the installed version, so it is not copied here. What it says, applied
to this folder:

- **Load the working folder.** `claude --plugin-dir augi-ui` in a terminal. The
  desktop app takes no flag: set `CLAUDE_CODE_PLUGIN_DIRS` to the folder's
  absolute path in the `env` block of `~/.claude/settings.json`, plus
  `CLAUDE_CODE_PLUGIN_DIR_WATCH=1` so desktop sessions reload on save
  (terminal sessions always do). Settings are read under `pluginConfigs`
  `augi-ui@inline`.
- **Check.** `claude plugin validate augi-ui` (manifest and module, as the
  engine reads them; run it at the repo root to check the marketplace file
  too), `claude plugin test augi-ui`, and `claude --debug` for any hook the
  engine skipped and why.
- **Type-check.** `tsc -p augi-ui`, against the types the engine writes to
  `augi-ui/.claude-plugin/types/` when it loads the plugin (git-ignored by
  the engine's own `.gitignore` there).
- **Before sharing.** The guide's secret scan over the folder, then a read of
  every shipped file for keys, internal URLs, other people's data or employer
  code. The repo's privacy hook also runs on every commit
  (`docs/reference/privacy-guard.md`).
- **Release.** Bump `version` in `augi-ui/.claude-plugin/plugin.json` and push
  `main`; an installed copy changes only with a new version or commit.

## Where the code lives

- `augi-ui/hooks/register.tsx` — the whole plugin: pure functions
  (settings, board, Agent Log, sessions, draft) and the Claude Code hooks and
  pane that call them.
- `augi-ui/hooks/register.test.tsx` — unit tests for the pure functions and
  mounted-pane tests for the flows. Run `claude plugin test augi-ui`.
- `.claude-plugin/marketplace.json` — makes this repo a plugin marketplace.

The pure functions are the part a future non-Claude interface (a standalone
terminal UI) would share; they move to their own module when that second
interface is real.
