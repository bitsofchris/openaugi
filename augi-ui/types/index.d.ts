export type Lens = { name: string; path: string; description: string }
// An open checkbox under a card or a note's Next heading. `note` is the first
// wikilink's target, so a sub-item that links a child task opens it.
export type SubItem = { text: string; note: string | null }
// A Kanban card that links a note: the project or task an anchor points at.
export type Card = { column: string; note: string; detail: string; tasks: SubItem[] }
// A chat anchored here: when it was filed, its goal (its first prompt), its session id,
// and whether it is marked finished.
export type Chat = { id: string; when: string; title: string; done: boolean }
// Something pinned to an anchor, a line of its note's Agent Log section. `on` is whether
// it is still pinned (listed in the pane, named in the next draft): unpinning ticks the line
// off and keeps it. A note's target is its name, the rest a path or URL; `path` is where a
// pinned note was found.
export type PinKind = 'note' | 'file' | 'session' | 'artifact' | 'url'
export type Pin = { on: boolean; kind: PinKind; target: string; label: string; path?: string }
// The anchor's Agent Log (once called its trail): the section at the end of its note that
// holds what the pane keeps for it. What the pane reads of it: the chats filed there,
// where each last stood (`states`, by chat id, with `when` it was said), and the pins.
export type Trail = { chats: Chat[]; pins: Pin[]; states: Record<string, { when: string; text: string }> }
// The task offered to a chat with no anchor: a board note to link, or a title (and the
// Jira key its branch names) for a new one, with why; `done` once taken or turned down.
export type Offer = { session: string; note: string; title: string; jira: string; why: string; busy: boolean; error: string; done: boolean }
// Where a chat's work stands, as its own model last said when asked: see `chatState`.
export type ChatState = { session: string; text: string; when: string; at: number; busy: boolean; error: string }
// A recall question due again, from a note's Recall table: the note's name and path, the
// question's id there, its claims, and the day it fell due.
export type Due = { topic: string; path: string; id: string; question: string; rubric: string[]; due: string }
// What the grader said about an answer, and the day the question is due again.
export type RecallResult = { grade: 'recalled' | 'partial' | 'missed' | 'peeked'; missing: string; misconception: string; feedback: string; followUp: string; next: string }
// The Recall card's question, this chat's (`session`): from the chat's own model
// (`from: 'chat'`) or a due one from a note's Recall table (`from: 'due'`, with its `id`).
// `topic` is the note it is kept in (the anchor, empty for none) and `path` where that note
// is, when known; `at` is when it was asked (it paces the automatic asks); `busy` while
// asking or grading; `result` once graded.
export type Recall = { session: string; topic: string; path: string; id: string; question: string; rubric: string[]; from: 'chat' | 'due'; at: number; busy: '' | 'asking' | 'grading'; error: string; result: RecallResult | null }
// One thing setup checks: a setting, the path it resolves to, and whether it's there.
export type SetupItem = { field: string; label: string; path: string; ok: boolean; starter: boolean }
// Today's note: the pattern its path comes from (empty for none), the path for `day`
// (YYYY-MM-DD), whether the file is there, its first open checkboxes, and when it last changed.
// Today's note, or this week's when the weekNote setting names one (`week`): where it is,
// whether it is there, what is open in it, and for a week its workstreams (`goals`).
export type Today = { pattern: string; path: string; day: string; exists: boolean; todo: string[]; mtime: number; week: boolean; goals: { text: string; note: string | null; done: boolean; line: string }[] }
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
  // Which task template new tasks are made from: the built-in one or the notes folder's own.
  taskTemplate: 'built-in' | 'yours'
  today: Today
  trail: Trail
  // Where each pinned note was found, by name.
  pinPaths: Record<string, string>
  lenses: Lens[]
  // Recall questions due today or earlier, across the learning records, oldest first.
  due: Due[]
  loadedAt: number
}
// A note to self on this chat's scratchpad; `when` as `stamp()` writes it.
export type PadEntry = { when: string; text: string }
// What goes into every draft: `today` is today's note, when there is one.
export type Picks = { aboutMe: boolean; openAugi: boolean; project: boolean; today: boolean }
export type View = {
  // The two sections and each card under them, folded or open.
  chatOpen: boolean
  anywhereOpen: boolean
  anchorOpen: boolean
  stateOpen: boolean
  recallOpen: boolean
  // The State card's list of saved prompts to ask with.
  statePickOpen: boolean
  pinsOpen: boolean
  padOpen: boolean
  // The You card under Always: the season line, the context note, the board and today's note.
  youOpen: boolean
  anchorsOpen: boolean
  lensesOpen: boolean
  setupOpen: boolean
  inOpen: boolean
  // The Anchor card's detail, outcome and open tasks, folded away by default.
  detailsOpen: boolean
  // Which field is open in the Anchor card: none or a new task.
  taskMode: '' | 'new'
  // How many lines the You card has logged to today's note this session: it empties the log field after each.
  logged: number
}
// The task this chat just made, for the Anchor card's "just made" row; an empty `note` is none.
// `jira` is the key typed with its title, if any; `where` says where on the board it went.
export type JustMade = { note: string; path: string; jira: string; where: string }
// The anchor `note` this chat (`session`) started on by itself, carried over from the last
// chat in its folder: empty once the user keeps or changes it or the chat is filed.
// `named` is `<session>|<anchor>` once this chat is in that anchor's log under a real title.
export type Inherited = { session: string; note: string; named: string }
// What this chat's model was last told about its anchor, beside one of the user's prompts:
// the session it was told in, the anchor (empty for none), today's note's path (empty for
// none), and `again` once a compaction may have dropped it from what the model reads.
export type Told = { session: string; note: string; today: string; again: boolean }
// The session id the pane last opened by itself for; `-` before any.
export type Opened = { session: string }
declare module 'claude-code' {
  interface PluginState {
    'augi-ui': { swatch: Shaped<Swatch>; picks: Shaped<Picks>; view: Shaped<View>; pad: Shaped<PadEntry[]>; justMade: Shaped<JustMade>; opened: Shaped<Opened>; padDraft: Shaped<string>; inherited: Shaped<Inherited>; told: Shaped<Told>; chatState: Shaped<ChatState>; offer: Shaped<Offer>; recall: Shaped<Recall>; recallDraft: Shaped<string> }
  }
}
