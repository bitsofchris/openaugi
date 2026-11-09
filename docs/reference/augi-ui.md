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
| `fileEditor` | What opens a file that is not a note (code, docs outside the notes folder): vscode, cursor or system | `vscode` |
| `boardNote` | The Kanban note whose linked cards are the anchors | `Kanban.md` |
| `contextNote` | A note about the user and this season | `OpenAugi/AGENT/Slowly Changing Context.md` |
| `lensFolder` | One markdown file per lens | `OpenAugi/AGENT/lenses` |
| `logFolder` | One Agent Log per anchor | `OpenAugi/Agent-Logs` |
| `taskFolder` | The one flat folder every new task note goes to; empty is the notes folder's root. Tasks are organized by their `Up:` link, never by folder | — |
| `jiraProject` | Optional. The Jira project key or board (`PROJ`), quoted into the Jira drafts | — |
| `jiraSite` | Optional. The Jira site's base URL (`https://example.atlassian.net`), only to open a linked issue in the browser | — |
| `jiraServer` | Optional. The Jira MCP server's name as `/mcp` lists it; empty looks for a likely one | — |
| `lensPicks` | Lenses listed first, comma separated; empty lists all | — |

Each missing file can be created as a starter, pointed at an existing note, or
skipped. Starters are never written over an existing file.

**The pane.**

The pane has two sections. **This chat** holds what belongs to the anchor this
chat is attached to: the anchor card, its questions, chats and saved notes, the
files this chat touched, and the scratchpad. **Anywhere** holds what is not
tied to one chat: the context
picker (pull notes, files, links and sessions from anywhere in the notes
folder), the lenses, and setup.

- **Anchor** (`a`): a linked card from the board — a project or a task. The
  menu lists each card's linked sub-items (`\t- [ ] [[Child task]]`) indented
  under it, and any of them can be the anchor too: it anchors in its parent's
  column with `under <parent>` as its detail and no tasks of its own. A stored
  anchor that is neither a card nor a linked sub-item (a pasted draft named it,
  or it left the board) still anchors, bare, while a note of that name is in the
  notes folder; with no such note the pane shows no anchor. The
  anchor belongs to the chat session: a new session opens with none, and a
  reload or `claude --resume` of the session keeps it (`$.store`, keyed by
  session id). Choosing one records this chat in its Agent Log. The card shows
  only the links (note, Agent Log, artifacts, Jira, and under them a row of
  `📄` buttons that open its newest five pinned files and notes, however they
  were pinned; see **Files**) until `▸ outcome and next (N)`
  is opened: then the card's detail, the note's `🎯` outcome, and its open
  sub-items and the note's `# Next` checkboxes as tasks; one that links a
  note (`\t- [ ] [[Child task]]`) is a button that opens it, which is the whole
  parent/child story: the child's `Up:` line points back. A `jira:` key in the
  note's frontmatter shows as `🔗 PROJ-123` (see **Jira**). `p` opens the note
  you write, `k` the Agent Log, and `◇` buttons open its newest three pinned
  artifacts.
- **New task** (two buttons in the Anchor card): `＋ new task` (always) and
  `＋ sub-task of <anchor>` (with an anchor). No field that makes a file is open
  until one is pressed: it opens a title field and `cancel`; an empty title or
  `cancel` makes nothing. A title may carry a Jira key or issue link at either
  end. The note goes into `taskFolder`, one flat folder whatever the anchor
  (hierarchy is the `Up:` link, never where the file sits; with `taskFolder`
  empty, the notes folder's root, and the toast says Setup changes it), from
  `OpenAugi/AGENT/templates/Task.md` in the notes folder (else a built-in copy
  of the shipped one: date and title filled, a `Jira: KEY` line under `Up:`).
  A **new task** is standalone: no `Up:` line, a top-level `- [ ] [[<title>]]`
  card at the end of the first column. A **sub-task** hangs off the anchor:
  `Up: [[<anchor>]]`, a linked sub-item `\t- [ ] [[<title>]]` as the last of
  the anchor card's indented lines (`insertSubItem`; an anchor that is itself a
  sub-item gets it beside it under the same card; an anchor not on the board
  gets a top-level card at the end of the first column, and the toast says
  which), and the anchor pinned, ticked, in the child's Agent Log so the two
  share context. Either way this chat's transcript is pinned there too, the
  board is read back into the pane, and **the chat's anchor and prompt box are
  left alone**. A `✓ just made` row then offers: `open` (the note in the
  editor), `define here` (an unsent draft in this chat's box asking it to fill
  in the note's Outcome and Verify and the Agent Log's `## Context`, and with
  Jira on and no key typed to find the task's issue; a half-typed prompt goes to
  the clipboard first), `copy for a new chat` (a context draft whose project
  line names the new task, so pasting it into a fresh chat anchors that chat on
  it), `anchor on it`, and `dismiss`. The row is this chat's: `/clear` and a
  resume drop it. It never writes over a note that already exists.
- **Jira** (optional): the pane never calls Jira itself, because Jira MCP servers
  differ in name and tools. It does the deterministic part and drafts the rest
  for the chat, which uses whatever Jira tools it has; nothing is created or
  posted in Jira until the user sends a draft and says yes. Jira is on when a
  Jira MCP server is found among the session's tools (`jiraServer`, else a
  likely name such as `atlassian` or `jira`), `jiraProject` or `jiraSite` is
  set, or the anchor already has a `jira:` key; otherwise the pane shows nothing
  about it. With an anchor: unlinked, paste a key or issue link into `🔗` to
  write the key into the note (one line, read fresh, on the user's press: a
  `Jira: KEY` line under `Up:`, or `jira:` in the frontmatter when the note has
  frontmatter), or `find in Jira` to draft a search that lists likely issues
  and links the one the user picks. Linked, `🔗 KEY` opens the issue when
  `jiraSite` is set, `unlink` removes the line, and `↑ update Jira` drafts a
  status update from the note and Agent Log, shown first and posted as a comment
  only on a yes. A `jira-update.md` lens in the lens folder replaces that
  built-in wording.
- **Context**: what the next chat is told to read — the anchor's notes, pinned
  artifacts (listed first), notes, files, links and coding sessions (a
  collapsed menu pins a recent Claude or Codex session), the context note and
  board (`m`), and OpenAugi search. An artifact the chat publishes is pinned
  ticked; one published before an anchor was chosen is pinned when it is.
- **Draft** (`n`): fills the prompt with that context. Nothing is sent until
  the user presses enter. `⧉ copy it for a new chat` copies the same text instead
  (the prompt box is left alone) to start a fresh chat on the anchor, say a
  manager chat that reads the Agent Log's list of every chat on it. The draft's
  `- Project [[<note>]]:` line names the anchor, so when it is sent in a chat
  with no anchor, that chat is anchored on it first and lands in the same Agent
  Log; a chat that already has an anchor keeps its own. A draft with the
  project context unticked names no anchor and attaches nothing.
- **Questions**: parked with `?` in the Anchor card, listed under `On <anchor>`.
  Pressing one drafts it as the instruction. `aside` asks it on the side: with
  Claude Code's `/btw` command in the session (checked when the pane loads) it
  fills `/btw <question>`, unsent; without it, a low-effort fork of this chat
  answers from what the chat already holds, shown under the question with `keep`
  (written under the question in the Agent Log as `  - aside: …`, the question
  left open; a second keep replaces it) and `drop`. Unkept answers live in
  session state only.
- **Lenses (saved prompts)** (`l`): run a lens with the same context; `+ new lens`
  takes a name, writes a skeleton file in the lens folder (never over one that's
  there) and opens it in your editor to paste the prompt.
  Every lens in the lens folder is listed; the `lensPicks` setting narrows and orders
  the menu. Lenses follow the OpenAugi lens contract (`OpenAugi/AGENT/lens-template.md`).
- **Files** (`📄 Files this chat touched · N`, in This chat, drawn once there is
  one): every file the chat wrote or edited (`Write`, `Edit`, `NotebookEdit`
  calls that ran; a denied or failed one is not listed), newest first, one row
  per path, the newest five until `▸ show all (N)`, at most 30, kept in session
  state (a `/clear` starts a new list). Agent Logs and dot-folders of the notes
  folder are left out. A row shows the path made short (relative to the notes
  folder or the session's working folder, else `~/…`), a dim `new` when the
  chat created the file, and `open`: a markdown note in the notes folder opens
  with `editor`, anything else with `fileEditor`; a file that is gone says so.
  With an anchor, `pin` adds it to the Agent Log's `## Context` unticked (a
  markdown file in the notes folder as `[[note]]`, the rest as `file:`; a
  link, not prompt context, until ticked there) and the row then reads
  `pinned`. Every pinned file or note, however it was pinned, is a `📄` button
  in the Anchor card. A new note the chat writes in the notes folder is still
  pinned unticked by itself.
- **Scratchpad**: notes to self for this chat only, kept in session state (gone
  with the session). `→ notes` adds one to the anchor's own note as `- HH:MM text`
  under its `# Journal`, beneath that day's dated heading (made if missing); with
  no anchor the entries just stay on the pad. This is the one place the pane
  writes the user's own note: only on their press, only their own words.
- **Agent Log**: per anchor, a companion note that holds the pinned context,
  open questions (with any kept side answers indented under them) and every
  chat anchored on it (titled `chat` until a draft
  sent from the pane names it). Its first line links the user's own note (so
  the note shows it as a backlink); agents never write that note (the pane does only for a
  saved scratchpad entry). An older log is tidied (title and seen box dropped) when the pane next writes it.

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
