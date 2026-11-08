---
name: augi-pane
description: Augi, the Claude Code pane over an OpenAugi vault — anchor on a board card, pick the context a chat gets, run lenses, and keep an Agent Log per anchor. Drafts prompts only; the user presses enter.
---

# Augi — the pane beside your agent chat

## When to use

You work with an agent (Claude Code, in the terminal or the desktop app) and
your projects, context and lenses live in a folder of markdown notes. Augi
keeps that folder one click away from the chat: pick what you are working on,
choose what the agent should read, run a saved lens, and see the chats and
questions that belong to that project.

It does not need the OpenAugi engine. With OpenAugi installed, chats can also
search the notes through its MCP tools.

## Install

```
/plugin install augi --marketplace bitsofchris/openaugi
```

Then `/augi` opens the pane. For development, load the folder directly:
`claude --plugin-dir ui/claude`, or set `CLAUDE_CODE_PLUGIN_DIRS` to that path
in `~/.claude/settings.json` so the desktop app loads it too.

## How it works

**Setup.** The first `/augi` walks through settings (stored by Claude Code, also
in `/config` under `augi`; `/augi setup` runs it again):

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
  Lenses follow the OpenAugi lens contract (`OpenAugi/AGENT/lens-template.md`):
  with no picks set, every lens with `trigger: on-demand` is listed.
- **Agent Log**: per anchor, a companion note that holds the pinned context,
  open questions and the chats started from the pane. The user's own note is
  never written; `⧉` copies an embed line to paste into it.

## Where the code lives

- `ui/claude/hooks/register.tsx` — the whole plugin: pure functions
  (settings, board, Agent Log, sessions, draft) and the Claude Code hooks and
  pane that call them.
- `ui/claude/hooks/register.test.tsx` — unit tests for the pure functions and
  mounted-pane tests for the flows. Run `claude plugin test ui/claude`.
- `.claude-plugin/marketplace.json` — makes this repo a plugin marketplace.

The pure functions are the part a future non-Claude interface (a standalone
terminal UI) would share; they move to their own module when that second
interface is real.
