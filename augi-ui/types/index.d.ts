export type Lens = { name: string; path: string; description: string }
export type Session = { path: string; tool: 'claude' | 'codex'; repo: string; title: string; mtimeMs: number }
// An open checkbox under a card or a note's Next heading. `note` is the first
// wikilink's target, so a sub-item that links a child task opens it.
export type SubItem = { text: string; note: string | null }
// A Kanban card that links a note: the project or task an anchor points at.
export type Card = { column: string; note: string; detail: string; tasks: SubItem[] }
// A chat anchored here: when it was filed, its goal (its first prompt, or what a child
// chat was started for), its session id, and whether it is finished.
export type Chat = { id: string; when: string; title: string; done: boolean }
// A question an agent (or an older pane) left on the trail under ## Questions.
export type Question = { when: string; text: string; done: boolean }
// Something kept as context for an anchor, under ## Context on its trail. `on` is whether
// it goes into the next draft; a note's target is its name, the rest a path or URL.
// `path` is set on search hits only: where the note was found.
export type PinKind = 'note' | 'file' | 'session' | 'artifact' | 'url'
export type Pin = { on: boolean; kind: PinKind; target: string; label: string; path?: string }
// The anchor's Agent Log (once called its trail): the companion note agents write.
export type Trail = { questions: Question[]; chats: Chat[]; saved: string[]; pins: Pin[] }
// One thing setup checks: a setting, the path it resolves to, and whether it's there.
export type SetupItem = { field: string; label: string; path: string; ok: boolean; starter: boolean }
// Today's note: the pattern its path comes from (empty for none), the path for `day`
// (YYYY-MM-DD), whether the file is there, its first open checkboxes, and when it last changed.
export type Today = { pattern: string; path: string; day: string; exists: boolean; todo: string[]; mtime: number }
export type Swatch = {
  setup: SetupItem[]
  // Obsidian vaults found on this computer, offered on the setup screen.
  vaults: string[]
  // What setup found in the notes for the step it's on, by setting.
  found: Record<string, string[]>
  // Setup steps skipped for now.
  skipped: string[]
  season: string
  cards: Card[]
  anchor: Card | null
  notePath: string
  outcome: string
  // Open checkboxes under the note's Next heading.
  next: SubItem[]
  // The anchor note's `jira:` frontmatter, a Jira key, or empty.
  jira: string
  // The Jira MCP server found among this session's tools, or empty.
  jiraServer: string
  // Which task template new tasks are made from; `old` is the notes folder's own from
  // before the Agent Log left the note.
  taskTemplate: 'built-in' | 'yours' | 'old'
  today: Today
  trail: Trail
  // The anchor's open chats whose transcripts are on this computer, by chat id: the file,
  // and when it was last written. `seenAt` is when those were read.
  chatFiles: Record<string, string>
  chatSeen: Record<string, number>
  seenAt: number
  // Where each pinned note was found, by name.
  pinPaths: Record<string, string>
  // Notes a pane search found, to pin with a press.
  hits: Pin[]
  // Pins that go into every chat's draft, whatever its anchor; kept in the store.
  everywhere: Pin[]
  lenses: Lens[]
  // The lenses that get a button where they are used, by path; empty when the lens folder has none.
  quick: { overview: string; standup: string }
  sessions: Session[]
  loadedAt: number
}
// An artifact this chat published: its URL and the title it was published under.
export type MadeArtifact = { url: string; title: string }
// What this chat has published, for the session `session` (a /clear starts a new list).
export type Made = { session: string; artifacts: MadeArtifact[] }
// A file this chat wrote or edited; `made` when it did not exist before the chat's first write.
export type TouchedFile = { path: string; made: boolean }
// What this chat has touched, newest first, for the session `session` (a /clear starts a new list).
export type Touched = { session: string; files: TouchedFile[] }
// A note to self on this chat's scratchpad; `when` as `stamp()` writes it.
export type PadEntry = { when: string; text: string }
// What goes into every draft: `today` is today's note, when there is one.
export type Picks = { aboutMe: boolean; openAugi: boolean; project: boolean; today: boolean }
export type View = {
  // The two sections and each card under them, folded or open.
  chatOpen: boolean
  anywhereOpen: boolean
  anchorOpen: boolean
  trailOpen: boolean
  pinsOpen: boolean
  touchedOpen: boolean
  padOpen: boolean
  // The You card under Always: the season line, the context note and the board.
  youOpen: boolean
  // The Today card under Always, drawn when there is a daily note.
  todayOpen: boolean
  contextOpen: boolean
  anchorsOpen: boolean
  sessionsOpen: boolean
  lensesOpen: boolean
  // The finished chats of the anchor, folded away by default.
  chatsDoneOpen: boolean
  setupOpen: boolean
  inOpen: boolean
  doneOpen: boolean
  // The Anchor card's detail, outcome and open tasks, folded away by default.
  detailsOpen: boolean
  // The Files card lists every touched file, not just the newest five.
  filesOpen: boolean
  // Which field is open in the Anchor card: none, a standalone task, or a sub-task of the anchor.
  // `child` asks for a child chat's goal; `promote` turns the chat `promote` into a task.
  taskMode: '' | 'new' | 'sub' | 'child' | 'promote'
  promote: string
  // How many lines the Today card has logged this session: it empties the log field after each.
  logged: number
}
// The task this chat just made, for the Anchor card's "just made" row; an empty `note` is none.
// `jira` is the key typed with its title, if any; `where` says where on the board it went.
export type JustMade = { note: string; path: string; jira: string; where: string }
// The anchor `note` this chat (`session`) started on by itself, carried over from the last
// chat in its folder: empty once the user keeps or changes it or the chat is filed.
// `named` is `<session>|<anchor>` once this chat is in that anchor's log under a real title.
export type Inherited = { session: string; note: string; named: string }
// The session id the pane last opened by itself for; `-` before any.
export type Opened = { session: string }
declare module 'claude-code' {
  interface PluginState {
    'augi-ui': { swatch: Shaped<Swatch>; picks: Shaped<Picks>; view: Shaped<View>; made: Shaped<Made>; touched: Shaped<Touched>; pad: Shaped<PadEntry[]>; justMade: Shaped<JustMade>; opened: Shaped<Opened>; padDraft: Shaped<string>; inherited: Shaped<Inherited> }
  }
}
