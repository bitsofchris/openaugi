export type Lens = { name: string; path: string; description: string }
export type Session = { path: string; tool: 'claude' | 'codex'; repo: string; title: string; mtimeMs: number }
// An open checkbox under a card or a note's Next heading. `note` is the first
// wikilink's target, so a sub-item that links a child task opens it.
export type SubItem = { text: string; note: string | null }
// A Kanban card that links a note: the project or task an anchor points at.
export type Card = { column: string; note: string; detail: string; tasks: SubItem[] }
export type Chat = { id: string; when: string; title: string; hidden: boolean }
// A question or side quest captured from the pane, kept on the trail under ## Questions.
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
  trail: Trail
  // Where each pinned note was found, by name.
  pinPaths: Record<string, string>
  // Notes a pane search found, to pin with a press.
  hits: Pin[]
  lenses: Lens[]
  sessions: Session[]
  loadedAt: number
}
// An artifact this chat published: its URL and the title it was published under.
export type MadeArtifact = { url: string; title: string }
// What this chat has published, for the session `session` (a /clear starts a new list).
export type Made = { session: string; artifacts: MadeArtifact[] }
// A note to self on this chat's scratchpad; `when` as `stamp()` writes it.
export type PadEntry = { when: string; text: string }
export type Picks = { aboutMe: boolean; openAugi: boolean; project: boolean }
export type View = {
  anchorsOpen: boolean
  contextOpen: boolean
  sessionsOpen: boolean
  lensesOpen: boolean
  showHidden: boolean
  setupOpen: boolean
  inOpen: boolean
  doneOpen: boolean
  // The Anchor card's detail, outcome and open tasks, folded away by default.
  detailsOpen: boolean
  // Which new-task field is open in the Anchor card: none, a standalone task, or a sub-task of the anchor.
  taskMode: '' | 'new' | 'sub'
}
// The task this chat just made, for the Anchor card's "just made" row; an empty `note` is none.
// `jira` is the key typed with its title, if any.
export type JustMade = { note: string; path: string; jira: string }
// A side answer to a parked question, by the question's `when` and text: `pending`
// while the fork runs, then the answer until it is kept in the Agent Log or dropped.
export type AsideAnswer = { when: string; text: string; answer: string; pending: boolean }
// `btw`: whether this session has a /btw command, checked when the pane loads.
export type Asides = { btw: boolean; answers: AsideAnswer[] }

declare module 'claude-code' {
  interface PluginState {
    'augi-ui': { swatch: Shaped<Swatch>; picks: Shaped<Picks>; view: Shaped<View>; made: Shaped<Made>; pad: Shaped<PadEntry[]>; asides: Shaped<Asides>; justMade: Shaped<JustMade> }
  }
}
