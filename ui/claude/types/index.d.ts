export type Lens = { name: string; path: string; description: string }
export type Session = { path: string; tool: 'claude' | 'codex'; repo: string; title: string; mtimeMs: number }
// A Kanban card that links a note: the project or task an anchor points at.
export type Card = { column: string; note: string; detail: string; tasks: string[] }
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
  next: string[]
  // Whether the note already embeds its Agent Log.
  hasEmbed: boolean
  trail: Trail
  // Where each pinned note was found, by name.
  pinPaths: Record<string, string>
  // Notes a pane search found, to pin with a press.
  hits: Pin[]
  lenses: Lens[]
  sessions: Session[]
  loadedAt: number
}
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
}

declare module 'claude-code' {
  interface PluginState {
    augi: { swatch: Shaped<Swatch>; picks: Shaped<Picks>; view: Shaped<View> }
  }
}
