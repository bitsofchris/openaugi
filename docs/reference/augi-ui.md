---
name: augi-ui
description: augi-ui, the Claude Code pane over an OpenAugi vault — anchor on a board card, tell the chat which note it is on, pick the context a chat gets, keep today's note in view, run lenses, and keep an Agent Log per anchor. Drafts prompts only; the user presses enter.
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
| `dailyNote` | Today's note, shown in the You card: a path inside the notes folder with the date in it (`Daily/{{date:YYYY-MM-DD}}.md`). Empty reads Obsidian's own daily-note settings (`.obsidian/daily-notes.json` for the folder and name format; with no folder named there, the default new-note folder from `.obsidian/app.json`, as Obsidian does); `none` turns it off | — |
| `dailyHeading` | The heading in today's note that lines logged from the pane go under (made when missing); empty adds them at the end of the note | — |
| `taskFolder` | The one flat folder every new task note goes to; empty is the notes folder's root (Setup warns while it is). Tasks are organized by their `Up:` link, never by folder | — |
| `taskColumn` | The board column a new standalone task goes to when the chat has no anchor, by its heading; with an anchor it goes to the anchor's column. Empty is the board's first column | — |
| `jiraProject` | Optional. The Jira project key or board (`PROJ`), quoted into the Jira drafts | — |
| `jiraSite` | Optional. The Jira site's base URL (`https://example.atlassian.net`), only to open a linked issue in the browser | — |
| `jiraServer` | Optional. The Jira MCP server's name as `/mcp` lists it; empty looks for a likely one | — |
| `lensPicks` | Lenses listed first, comma separated; empty lists all | — |
| `openOnStart` | `yes` opens the pane when a new interactive chat starts; `no` keeps it closed until `/augi` | `yes` |
| `tellChat` | `yes` tells each chat's own model its anchor (the note, its Agent Log, today's note) beside your next prompt, so "update the task note" needs no path; `no` keeps the anchor in the pane only | `yes` |

Each missing file can be created as a starter, pointed at an existing note, or
skipped. Starters are never written over an existing file.

**The pane.**

It opens with each new chat, once a session (turn that off with `openOnStart`,
or the switch under Setup): in a terminal when the session starts; in the
desktop app and the editor when their window attaches (`session.attach`),
since those start a session with no surface to draw on yet. A window that
reconnects does not reopen a pane you closed, and a headless run or a phone
joining opens nothing. An open pane follows its anchor's files: every ten
seconds, and when a prompt is sent, it checks whether the board, the anchor's
note or its Agent Log changed and reads them again if so, so two chats on one
anchor see each other's pins and tasks without a refresh.

The pane has two sections. **This chat** holds what belongs to the anchor this
chat is attached to: the anchor card, its
pins with the draft button under them, the files this chat touched, and the
scratchpad. **Always** holds what is not tied to one chat: the You card (the
season line from the context note, `✏️ my context`, `📋 the board` and today's
note: see **Today** below), the saved prompts, and setup. Setup holds paths
and switches only.

Both sections and every card fold: each title row ends in a `▾ hide` /
`▸ show` button, and a folded card keeps its count in the title. Lenses and
Setup start folded, everything else open. The fold state is kept for the
session.

Layout rules, so the pane reads the same on the terminal and the desktop
(where every Button, `plain` or not, is drawn as a native button): a title is
text, never a button; a list row puts what it is on the left and its actions
together at the right edge (the `line` helper); indentation is a Box's
`paddingLeft`, never spaces in a label; rows space their children with `gap`,
not divider characters; secondary actions (`unpin`, `done`, `drop`, `cancel`)
are `dimColor`, and the one primary button is the draft.

- **Anchor** (`a`): a linked card from the board — a project or a task. The
  menu lists each card's linked sub-items (`\t- [ ] [[Child task]]`) indented
  under it, and any of them can be the anchor too: it anchors in its parent's
  column with `under <parent>` as its detail and no tasks of its own. A stored
  anchor that is neither a card nor a linked sub-item (a pasted draft named it,
  or it left the board) still anchors, bare, while a note of that name is in the
  notes folder; with no such note the pane shows no anchor. The
  anchor belongs to the chat session, and a reload or `claude --resume` of the
  session keeps it (`$.store`, keyed by session id). **A new chat does not
  start by picking one**: it starts on the anchor last chosen in its working
  folder (a Claude Code worktree counts as its repository), and the card says
  so with `keep`, `change` and `no anchor`. Only a session that began in the
  last ten minutes is given one, so an older chat that never had an anchor is
  left alone; a headless run is never given one, and neither is any chat when
  the folder's anchor names a note that is no longer on the board or in the
  notes folder; `no anchor` sticks for the session; a draft pasted from another
  anchor wins over one the chat only started on. A chat that is given its
  anchor this way is written to the Agent Log with its first prompt, not
  before, so a chat opened and abandoned leaves nothing there. Every anchored
  chat is titled in the log by its first prompt: a pane draft's lens and
  instruction, else the prompt's first line (a slash command titles nothing).
  Choosing an anchor by hand records the chat at once, as `chat` until then.
  **The chat itself knows its anchor.** The pane's state is not something the
  model can read, so the plugin says it: the first prompt sent on an anchor
  carries a short block beside it for the model, not shown in the chat, naming
  the note and its path, the Agent Log's path, today's note when there is one,
  and the rule for writing (what the agent saves goes in the Agent Log; the
  user's note gets a dated Journal entry only when they ask). After that
  "update the task note", "log this" or "save this to the task" need no path.
  It is sent once, and again only when something changed: the whole block for
  another anchor or after a compaction (which may have summarized it away), one
  line when the day's note is a new one, and one line when the anchor is
  cleared. A slash command carries nothing. It goes beside a prompt, not into
  the system prompt, because a chat's system prompt stays as first sent and
  would not follow a change of anchor. It reads the session's stored anchor, so
  it works with the pane closed. `tellChat: no` (⚙️ Setup, *tell each chat its
  anchor*) turns it off. The card shows
  only the links (note, Agent Log, Jira; pins are in their own card below, not
  repeated here) until `▸ outcome and next (N)`
  is opened: then the card's detail, the note's `🎯` outcome, and its open
  sub-items and the note's `# Next` checkboxes as tasks; one that links a
  note (`\t- [ ] [[Child task]]`) is a button that opens it, which is the whole
  parent/child story: the child's `Up:` line points back. A `jira:` key in the
  note's frontmatter shows as `🔗 PROJ-123` (see **Jira**). `p` opens the note
  you write, `k` the Agent Log.
- **Chats.** A piece of an anchor's work gets a chat, not a note. A new chat
  needs no button: open one in the same folder and it starts on the anchor, is
  filed in the Agent Log under `## Chats` with its first prompt as its title,
  and its model is told the anchor. The pane keeps no list of those chats: the
  Agent Log is the record (`k` opens it), and the `overview` lens reads it.
  A line there is `- <when> · <title> · \`<id>\``; one ending in `(done)` (or
  the older `(hidden)`) reads as finished, which is the agent's or the user's
  to write.
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
  card at the end of the anchor's column (the chat is working there), else of
  the `taskColumn` setting, else of the board's first column. A **sub-task** hangs off the anchor:
  `Up: [[<anchor>]]`, a linked sub-item `\t- [ ] [[<title>]]` as the last of
  the anchor card's indented lines (`insertSubItem`; an anchor that is itself a
  sub-item gets it beside it under the same card; an anchor not on the board
  gets a top-level card at the end of the `taskColumn` setting's column, else the first, and the toast says
  which), and the anchor pinned in the child's Agent Log so the two
  share context. Either way this chat's transcript is pinned there too and the
  board is read back into the pane. **A new task moves the chat onto it**: it
  is this chat starting something of its own, so the task becomes the chat's
  anchor (and the folder's, for the next new chat), the chat is filed in its
  Agent Log, and the next draft carries its context. **A sub-task leaves the
  chat on its anchor**: it is a piece of that work. The prompt box is left
  alone either way. A `✓ just made` row says where on the board it went and
  offers: `open` (the note in the editor), `open the board`, `define here` (an unsent draft in this chat's box asking it to fill
  in the note's Outcome and Verify and the Agent Log's `## Context`, and with
  Jira on and no key typed to find the task's issue; a half-typed prompt goes to
  the clipboard first), `anchor on it` (for a sub-task; a new task already is the anchor), and `dismiss`. The row is this chat's: `/clear` and a
  resume drop it. It never writes over a note that already exists.
  **The template.** The notes folder's own `OpenAugi/AGENT/templates/Task.md`
  wins over the built-in one, so a folder set up before the template changed
  keeps making tasks in the old shape. Setup names which is in use (*yours* or
  *built-in*). One that still has an Agent Log heading or embeds the log is
  flagged, counts as a setup item that needs you, and `use the current
  template` replaces it with the shipped one; the old file is kept beside it as
  `Task.md.<date>.bak`.
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
- **Today** (in the You card, under Always): the note for today, found from
  `dailyNote` or, with that empty, from Obsidian's daily-note settings. The
  date in the path is written with Obsidian's tokens (`YYYY YY MMMM MMM MM M
  DD D dddd ddd`, `[text]` kept). With a note for today the card has
  `☀️ today's note` (opens it), lists its first five open checkboxes so the
  day's priorities stay in view, and follows the file: ticking one in the
  editor, or midnight passing, shows within seconds. `+ log` adds
  `- HH:MM text` to the note, under `dailyHeading` when one is set, else at
  its end; a scratchpad entry has `→ today` for the same. Whether the note
  goes into every draft is a box under Setup (see **Context**).
  **The note is made only on a press, never by itself**: with none for today
  the card says where it would be and has `＋ make today's note`, which writes
  it and opens it. It is made from the template Obsidian's daily notes name,
  when that file is there, with Obsidian's own placeholders filled (`{{date}}`,
  `{{time}}`, `{{title}}`; a scripting plugin's code is copied as written and
  not run); else empty. A note that exists is never written over. Like the
  scratchpad's `→ notes`, a logged line is the user's own words written on
  their press. With no daily note found and none set, the card says where to
  set one. A standup or a plan for the day is a lens the user writes that
  reads this note, not code.
- **Context**: what the next chat is told to read is the anchor's pins plus
  four things that go in without being listed, each a checkbox under Setup →
  *What goes into every draft*, all on by default: the context note and board
  (`m`), the anchor's human and agent notes, today's note when there is one,
  and a search of the notes with OpenAugi.
- **Pins** (`📌 Pinned to <anchor> · N`, in This chat): the one place to pin.
  The list is the anchor's own, kept under `## Context` in its Agent Log:
  artifacts (listed first), notes, files, links and coding sessions. Paste a
  `[[note]]`, a file path or a link in the card to pin it. An artifact the
  chat publishes is pinned by itself (one published before an anchor was
  chosen, when it is), and so is a new note the chat writes in the notes
  folder. A pin row is its name and `✕ unpin`: pressing the name opens the pin
  (a session copies its resume command). Every pin goes into the anchor's
  drafts; there is no second state. **On the log a pin is a plain bullet**:
  `- [[note]] · why it matters`, or `- file: \`path\` · label` (also
  `session:`, `artifact:`, `url:`), so a line a chat adds there in that shape is
  a pin too, and any other bullet is a kept fact the pane leaves alone.
  **Unpinning ticks the line off** (`- [x] …`) instead of deleting it: it
  leaves the card and the drafts and stays on the log as a record, and pinning
  the same thing again turns the line back into a bullet where it was. A log
  written before pins were bullets ticked what it pinned (`- [x]` in drafts,
  `- [ ]` a link only): every such line reads as a pin, and the pane rewrites
  them as bullets, with the intro line that says so, the next time it writes
  that log (`tidyTrail`; the intro line is how a log's form is told). With no
  anchor the card is `📌 Context`, with nothing to pin to and the draft button.
- **Draft** (`n`, the button under the pins): fills the prompt with that
  context. Nothing is sent until
  the user presses enter. It is the one way to draft: the context is that of
  the chat's anchor: a new task just made already is the anchor, a sub-task is one `anchor on it` away. The draft's
  `- Project [[<note>]]:` line names the anchor, so when it is sent in a chat
  with no anchor, that chat is anchored on it first and lands in the same Agent
  Log; a chat that already has an anchor keeps its own. A draft with the
  project context unticked names no anchor and attaches nothing.
- **Questions**: the pane has one place to type a thought, the scratchpad
  (below), and `→ ask` there drafts an entry as the instruction. Questions an
  agent left under `## Questions` in the Agent Log stay in the log (`k` opens
  it); the pane neither lists nor writes that section.
- **Saved prompts** (`l`): the pane's name for lenses, since the card is a
  row of buttons that each run a saved prompt; the files, the folder and the
  settings still say lens. Run one with the same context; `+ new prompt`
  takes a name, writes a skeleton file in the lens folder (never over one that's
  there) and opens it in your editor to paste the prompt.
  Every lens in the lens folder is listed; the `lensPicks` setting narrows and orders
  the menu. Lenses follow the OpenAugi lens contract (`OpenAugi/AGENT/lens-template.md`).
- **Files** (`📄 Files this chat touched · N`, in This chat, drawn once there is
  one): every file the chat wrote or edited (`Write`, `Edit`, `NotebookEdit`
  calls that ran; a denied or failed one is not listed), newest first, one row
  per path, the newest five until `▸ show all (N)`, at most 30, kept in session
  state (a `/clear` starts a new list). Agent Logs and dot-folders of the notes
  folder are left out. A row shows the file's name first and then the folder
  it sits in (`register.tsx · augi-ui/hooks`; the folder relative to the notes
  folder or the session's working folder, else `~/…`, its last two parts
  kept), so a row cut short at its end still names the file, a dim
  `new` when the
  chat created the file, and `open`: a markdown note in the notes folder opens
  with `editor`, anything else with `fileEditor`; a file that is gone says so.
  With an anchor, `pin` adds it to the Agent Log's `## Context` (a
  markdown file in the notes folder as `[[note]]`, the rest as `file:`) and
  the row then reads `pinned`.
- **Usage log** (on by default; ⚙️ Setup → *count which controls I use* turns it
  off, as does `trackUsage: no`): one hook on `ui.press` and one on `ui.input`
  append a JSON line to `~/.openaugi/usage/augi-ui.jsonl` for every Button
  pressed and Input submitted in the pane (typing in a field is not logged), and one more (`draft-sent`) when a
  pane draft is sent. Nothing leaves the computer and there is no library: the
  line is appended by `sh`, so two chats can log at once. A line is
  `{ at, kind, what, lens?, anchored, surface }`: an ISO time; `press`, `input`
  or `sent`; the control's name; the lens's name when a lens was run; whether
  the chat had an anchor; `terminal` or `desktop`. `what` is the element's key
  with anything of the user's taken off (`card-<note>` is `card`,
  `pin-open-<kind>-<target>` is `pin-open`; `usageName`), so the log holds no
  note name, path, chat id or typed text. `/augi usage` prints the report, and
  Setup has `copy my usage report` and `clear the usage log`: counts for the
  last 30 days, lenses run, and every control never pressed in the whole log.
  **A new control is logged without any change**: its key is its name. When the
  key ends in something of the user's, add its prefix to `USAGE_PREFIXES`; add
  its name to `USAGE_FEATURES` either way, which `tests/test_augi_ui_usage.py`
  enforces by reading the Button and Input keys out of the source. SQLite was
  not used: the plugin has no database access, and the openaugi database is the
  notes index, rebuilt by ingest.
- **Scratchpad**: notes to self for this chat only, kept in session state (gone
  with the session). What is typed and not yet kept stays in the field: it is
  saved on every change, so a click away, a redraw or a reload does not lose
  it, and Enter turns it into an entry. `→ ask` drafts an entry as the
  instruction (unsent), so a question or side quest typed here is asked with
  one press. `→ notes` adds one to the anchor's own note as `- HH:MM text`
  under its `# Journal`, beneath that day's dated heading (made if missing); with
  no anchor the entries just stay on the pad. `→ today` adds one to today's
  note instead (see **Today**). These two, the You card's `+ log` and `＋ make today's note` and a Jira
  link are the only writes the pane makes to the user's own notes: only on
  their press, only their own words.
- **Agent Log**: per anchor, a companion note that holds the pinned context,
  open questions and every
  chat anchored on it (titled by its first prompt). Its first line links the user's own note (so
  the note shows it as a backlink); agents never write that note (the pane does only for a
  saved scratchpad entry). An older log is tidied (title and seen box dropped, ticked pins turned into bullets) when the pane next writes it.

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
