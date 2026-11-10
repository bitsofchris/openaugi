import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, RenderChildren, ToolCallResult } from 'claude-code'

import type { Card, Chat, Inherited, JustMade, Lens, Made, MadeArtifact, Opened, PadEntry, Picks, Pin, PinKind, Question, Session, SetupItem, SubItem, Swatch, Today, Touched, TouchedFile, Trail, View } from '../types'

// Everything machine-specific comes from the settings: the manifest's userConfig,
// set at install, in /config, or from the pane's setup menu.
export type Options = {
  notesFolder: string
  editor: string
  editorCommand: string
  // What opens a file that is not a note: vscode, cursor or system.
  fileEditor: string
  contextNote: string
  boardNote: string
  lensFolder: string
  logFolder: string
  // The old name for logFolder, still read.
  trailFolder: string
  // The one flat folder every new task note goes to; empty means the notes folder's root.
  taskFolder: string
  // The board column a new standalone task goes to when the chat has no anchor; empty means the first.
  taskColumn: string
  // Today's note: a path inside the notes folder with the date in it, `{{date:YYYY-MM-DD}}`.
  // Empty reads Obsidian's daily-note settings; `none` turns the Today card off.
  dailyNote: string
  // The heading in today's note that log lines go under; empty adds them at its end.
  dailyHeading: string
  lensPicks: string
  // `no` keeps the pane closed until /augi; anything else opens it with each new chat.
  openOnStart: string
  // Jira, all optional: the project key or board quoted into drafts, the site's base URL
  // (only to open an issue in the browser), and the MCP server's name as /mcp lists it.
  jiraProject: string
  jiraSite: string
  jiraServer: string
  // `no` stops the usage log; anything else keeps it.
  trackUsage: string
}
export type Paths = {
  home: string
  notes: string
  context: string
  board: string
  lenses: string
  trails: string
  // Where every new task note goes: the task folder, else the notes folder's root.
  tasks: string
  // The board column for a new standalone task from a chat with no anchor, or empty.
  taskColumn: string
  // The notes folder's own task template, which the pane prefers over its built-in one.
  taskTemplate: string
  // The daily note setting as typed (a pattern, `none` or empty) and the log heading in it.
  daily: string
  dailyHeading: string
  // Where Obsidian keeps its daily-note folder and name format for this notes folder.
  obsidianDaily: string
  claudeDir: string
  codexDir: string
  // The usage log: one JSON line per pane press, on this computer only.
  usage: string
  editor: string
  editorCommand: string
  fileEditor: string
  picks: string[]
  jiraProject: string
  jiraSite: string
  jiraServer: string
}

const expand = (path: string, home: string): string => path.trim().replace(/^~(?=\/|$)/, home).replace(/\/+$/, '')

// A setting inside the notes folder may also be an absolute path.
const within = (notes: string, path: string, home: string): string => {
  const clean = expand(path, home)

  return clean.startsWith('/') ? clean : `${notes}/${clean}`
}

export const resolvePaths = (options: Partial<Options>, home: string): Paths => {
  const notes = expand(options.notesFolder ?? '', home)

  return {
    home,
    notes,
    context: within(notes, options.contextNote || 'OpenAugi/AGENT/Slowly Changing Context.md', home),
    board: within(notes, options.boardNote || 'Kanban.md', home),
    lenses: within(notes, options.lensFolder || 'OpenAugi/AGENT/lenses', home),
    trails: within(notes, options.logFolder || options.trailFolder || 'OpenAugi/Agent-Logs', home),
    tasks: options.taskFolder?.trim() ? within(notes, options.taskFolder, home) : notes,
    taskColumn: (options.taskColumn ?? '').trim(),
    taskTemplate: `${notes}/OpenAugi/AGENT/templates/Task.md`,
    daily: (options.dailyNote ?? '').trim(),
    dailyHeading: (options.dailyHeading ?? '').trim().replace(/^#+\s*/, ''),
    obsidianDaily: `${notes}/.obsidian/daily-notes.json`,
    claudeDir: `${home}/.claude/projects`,
    codexDir: `${home}/.codex/sessions`,
    usage: `${home}/.openaugi/usage/augi-ui.jsonl`,
    editor: options.editor || 'obsidian',
    editorCommand: options.editorCommand ?? '',
    fileEditor: options.fileEditor || 'vscode',
    picks: (options.lensPicks ?? '').split(',').map(one => one.trim()).filter(Boolean),
    jiraProject: (options.jiraProject ?? '').trim(),
    jiraSite: (options.jiraSite ?? '').trim().replace(/\/+$/, ''),
    jiraServer: (options.jiraServer ?? '').trim(),
  }
}

let P: Paths = resolvePaths({}, '')
// Sets the paths the module works from; register calls it once HOME is known.
export const configure = (paths: Paths): void => {
  P = paths
}

// ---- Usage: which pane controls get pressed, kept in a file on this computer ----

// One line of the usage log. `what` is a control's name with anything of the user's
// (a note, a path, a chat id) taken off; `lens` names the lens a lens press ran.
export type UsageEvent = { at: string; kind: 'press' | 'input' | 'sent'; what: string; lens?: string; anchored: boolean; surface: string }

// Keys that end in something of the user's: `card-<note>`, `pin-open-<kind>-<target>`.
const USAGE_PREFIXES = [
  'card-sub', 'card', 'task', 'ask', 'done', 'reopen',
  'saved', 'chat', 'chat-task', 'chat-done', 'chat-reopen', 'pin-tick', 'pin-open', 'pin-drop', 'pin-paste', 'touched-open', 'touched-pin',
  'pad-input', 'pad-save', 'pad-ask', 'pad-today', 'pad-drop', 'today-log', 'all-tick', 'all-open', 'all-drop', 'all-paste', 'offer-anchor', 'offer-all',
  'edit', 'new-lens', 'starter', 'vault', 'found', 'editor', 'file-editor',
].sort((one, other) => other.length - one.length)
// Whole keys that only look like one of those.
const USAGE_WHOLE = ['task-new', 'task-sub', 'task-child', 'task-cancel', 'task-title-new', 'task-title-sub', 'task-title-child', 'task-title-promote', 'pad-save-all', 'edit-context']

// A control's name for the log. A key nobody listed is logged as written unless it
// looks like it carries a name or a path, so a new button is tracked without a change here.
export const usageName = (element: string): string => {
  if (USAGE_WHOLE.includes(element)) {
    return element
  }
  const prefix = USAGE_PREFIXES.find(one => element.startsWith(`${one}-`))
  if (prefix) {
    return prefix
  }

  return /[\s/\\:.]/.test(element) || element.length > 40 ? 'other' : element
}

// Every control the pane draws, by its log name: what a report checks for never used.
// tests/test_augi_ui_usage.py reads this file's Button and Input keys and fails when one is missing here.
export const USAGE_FEATURES = [
  'anchor-menu', 'card', 'card-sub', 'anchor-clear', 'inherit-keep', 'inherit-change', 'inherit-none', 'open-note', 'open-trail', 'anchor-details', 'task',
  'jira-open', 'jira-update', 'jira-unlink', 'jira-link', 'jira-find', 'task-new', 'task-sub', 'task-child', 'task-title-new', 'task-title-sub', 'task-title-child', 'task-title-promote', 'task-cancel',
  'made-open', 'made-board', 'made-define', 'made-copy', 'made-anchor', 'made-dismiss', 'template-update',
  'ask', 'done', 'show-done', 'reopen', 'saved', 'chat', 'chat-task', 'chat-done', 'chat-reopen', 'chats-done',
  'pin-tick', 'pin-open', 'pin-drop', 'pin-paste',
  'touched-open', 'touched-pin', 'files-all', 'pad-input', 'pad-save', 'pad-ask', 'pad-today', 'pad-drop', 'pad-save-all',
  'today-log', 'today-open', 'pick-today', 'run-overview', 'run-standup',
  'all-tick', 'all-open', 'all-drop', 'all-paste', 'pin-search', 'offer-anchor', 'offer-all', 'hits-clear', 'sessions-menu', 'draft', 'copy-draft', 'draft-sent',
  'lens', 'edit', 'new-lens', 'all-lenses', 'lens-menu', 'setup-menu',
  'about-me', 'pick-project', 'openaugi', 'edit-context', 'open-board', 'set-openOnStart', 'set-trackUsage', 'usage-report', 'usage-clear', 'start-over', 'refresh',
  'head-chatOpen', 'head-anywhereOpen', 'head-anchorOpen', 'head-trailOpen', 'head-pinsOpen', 'head-touchedOpen', 'head-padOpen', 'head-youOpen', 'head-todayOpen', 'head-contextOpen',
]

// The log's lines as events; a line that is not one is skipped.
export const parseUsage = (text: string): UsageEvent[] => {
  const events: UsageEvent[] = []
  for (const line of text.split('\n')) {
    try {
      const one = JSON.parse(line) as Partial<UsageEvent> | null
      if (one && typeof one.at === 'string' && typeof one.what === 'string') {
        events.push({ at: one.at, kind: one.kind ?? 'press', what: one.what, ...(typeof one.lens === 'string' ? { lens: one.lens } : {}), anchored: Boolean(one.anchored), surface: one.surface ?? '' })
      }
    } catch {
      // Not a JSON line: skip it.
    }
  }

  return events
}

const tally = (names: string[]): [string, number][] => {
  const counts = new Map<string, number>()
  for (const name of names) {
    counts.set(name, (counts.get(name) ?? 0) + 1)
  }

  return [...counts].sort((one, other) => other[1] - one[1] || one[0].localeCompare(other[0]))
}

// What was used in the last `days` days, most first, then what never was.
export const usageReport = (events: UsageEvent[], now: number, days = 30): string => {
  const since = new Date(now - days * 86_400_000).toISOString()
  const recent = events.filter(one => one.at >= since)
  const used = tally(recent.map(one => one.what))
  const lenses = tally(recent.filter(one => one.lens).map(one => one.lens ?? ''))
  const never = USAGE_FEATURES.filter(name => !events.some(one => one.what === name))
  const count = ([name, times]: [string, number]): string => `${String(times).padStart(5)}  ${name}`

  return [
    `Augi pane usage, last ${days} days: ${recent.length} events (${events.length} in the log, since ${events[0]?.at.slice(0, 10) ?? 'never'}).`,
    '',
    'Used:',
    ...(used.length > 0 ? used.map(count) : ['  nothing yet']),
    ...(lenses.length > 0 ? ['', 'Lenses run:', ...lenses.map(count)] : []),
    '',
    `Never used, in the whole log (${never.length}):`,
    never.length > 0 ? `  ${never.join(', ')}` : '  nothing: every control has been pressed at least once',
  ].join('\n')
}

export const EDITORS = ['obsidian', 'vscode', 'cursor', 'system', 'custom'] as const

// The command that opens a note in the chosen editor.
export const editorArgv = (editor: string, command: string, path: string): string[] => {
  if (editor === 'vscode') {
    return ['code', path]
  }
  if (editor === 'cursor') {
    return ['cursor', path]
  }
  if (editor === 'system') {
    return ['open', path]
  }
  if (editor === 'custom') {
    const parts = command.trim().split(/\s+/).filter(Boolean)
    if (parts.length === 0) {
      return ['open', path]
    }

    return parts.some(one => one.includes('{file}')) ? parts.map(one => one.split('{file}').join(path)) : [...parts, path]
  }

  return ['open', `obsidian://open?path=${encodeURIComponent(path)}`]
}

export const FILE_EDITORS = ['vscode', 'cursor', 'system'] as const

// A markdown note inside the notes folder; anything else is a file.
export const isVaultNote = (path: string, notes: string): boolean => Boolean(notes) && path.startsWith(`${notes}/`) && path.endsWith('.md')

// The command that opens a path: a note with the note editor, any other file with the file editor.
export const openArgv = (paths: Paths, path: string): string[] =>
  isVaultNote(path, paths.notes) ? editorArgv(paths.editor, paths.editorCommand, path) : editorArgv(paths.fileEditor, '', path)

// A path made short to read: relative to the notes folder or the session's root, else
// ~/… under home; then only the file's name and the two folders it sits in, `…/` standing
// for what was cut.
export const shortPath = (path: string, notes: string, home: string, root = '', kept = 3): string => {
  const base = [notes, root].find(one => one && path.startsWith(`${one}/`))
  const relative = base ? path.slice(base.length + 1) : home && path.startsWith(`${home}/`) ? `~${path.slice(home.length)}` : path
  const parts = relative.split('/')

  return parts.length > kept ? `…/${parts.slice(-kept).join('/')}` : relative
}

const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

// A date written the way Obsidian's daily notes name it: YYYY YY MMMM MMM MM M DD D dddd
// ddd, with `[text]` kept as written. Anything else stays as it is.
export const formatDate = (format: string, ms: number): string => {
  const date = new Date(ms)
  const two = (n: number): string => String(n).padStart(2, '0')
  const parts: Record<string, string> = {
    YYYY: String(date.getFullYear()),
    YY: String(date.getFullYear()).slice(-2),
    MMMM: MONTHS[date.getMonth()] ?? '',
    MMM: (MONTHS[date.getMonth()] ?? '').slice(0, 3),
    MM: two(date.getMonth() + 1),
    M: String(date.getMonth() + 1),
    DD: two(date.getDate()),
    D: String(date.getDate()),
    dddd: WEEKDAYS[date.getDay()] ?? '',
    ddd: (WEEKDAYS[date.getDay()] ?? '').slice(0, 3),
  }

  return format.replace(/\[([^\]]*)\]|YYYY|YY|MMMM|MMM|MM|M|DD|D|dddd|ddd/g, (token, kept: string | undefined) => (kept !== undefined ? kept : parts[token] ?? token))
}

// The daily note setting as a pattern: what was typed; else, from Obsidian's own
// daily-note settings (`obsidian` is that file's text: folder and name format, each with
// Obsidian's default); else none. `none` typed turns it off.
export const dailyPattern = (typed: string, obsidian: string): string => {
  if (typed.toLowerCase() === 'none') {
    return ''
  }
  if (typed) {
    return typed
  }
  if (!obsidian.trim()) {
    return ''
  }
  try {
    const { folder, format } = JSON.parse(obsidian) as { folder?: unknown; format?: unknown }
    const inFolder = typeof folder === 'string' ? folder.trim().replace(/^\/+|\/+$/g, '') : ''

    return `${inFolder ? `${inFolder}/` : ''}{{date:${typeof format === 'string' && format.trim() ? format.trim() : 'YYYY-MM-DD'}}}.md`
  } catch {
    return ''
  }
}

// Today's note for a pattern: each `{{date:FORMAT}}` (or `{{date}}`) filled for the day,
// `.md` added when missing, inside the notes folder unless absolute. Empty for no pattern.
export const dailyNotePath = (pattern: string, ms: number, notes: string, home: string): string => {
  if (!pattern.trim()) {
    return ''
  }
  const filled = pattern.trim().replace(/\{\{date(?::([^}]*))?\}\}/g, (_, format: string | undefined) => formatDate(format?.trim() || 'YYYY-MM-DD', ms))

  return within(notes, filled.endsWith('.md') ? filled : `${filled}.md`, home)
}

// The first few open checkboxes of a note, as text: what today's note says is still to do.
export const openBoxes = (text: string, limit = 5): string[] =>
  text.split('\n').map(line => line.match(/^\s*[-*] \[ \] (.*\S.*)$/)?.[1]).filter((one): one is string => Boolean(one)).map(plain).slice(0, limit)

// Adds a line to today's note: `- HH:MM text` as the last line under the heading named
// `heading` (any level, any case; made at the end when the note has none), or as the
// note's last line when no heading is named. Every other line stays as it was.
export const addDailyLine = (text: string, heading: string, one: PadEntry): string => {
  const entry = journalLines(one)
  if (entry.length === 0) {
    return text
  }
  const atEnd = (body: string, added: string[]): string => `${body}${!body || body.endsWith('\n') ? '' : '\n'}${added.join('\n')}\n`
  if (!heading) {
    return atEnd(text, entry)
  }
  const lines = text.split('\n')
  const headingAt = (at: number): { level: number; title: string } | null => {
    const found = lines[at]?.match(/^(#{1,6})\s+(.*?)\s*$/)

    return found ? { level: (found[1] ?? '').length, title: found[2] ?? '' } : null
  }
  const at = lines.findIndex((_, index) => headingAt(index)?.title.toLowerCase() === heading.toLowerCase())
  if (at < 0) {
    return atEnd(text, [...(text.trim() ? [''] : []), `## ${heading}`, ...entry])
  }
  const level = headingAt(at)?.level ?? 1
  let end = at + 1
  while (end < lines.length && (headingAt(end)?.level ?? 7) > level) {
    end += 1
  }
  let last = end - 1
  while (last > at && !lines[last]?.trim()) {
    last -= 1
  }
  lines.splice(last + 1, 0, ...entry)
  // The note's last line still ends with a newline.
  if (last + 1 + entry.length === lines.length) {
    lines.push('')
  }

  return lines.join('\n')
}

// What setup still needs: the notes folder first, then the files the pane reads.
export const setupItems = (paths: Paths, found: Set<string>): SetupItem[] => [
  { field: 'notesFolder', label: 'notes folder', path: paths.notes, ok: Boolean(paths.notes) && found.has(paths.notes), starter: false },
  { field: 'contextNote', label: 'context note', path: paths.context, ok: found.has(paths.context), starter: true },
  { field: 'boardNote', label: 'board', path: paths.board, ok: found.has(paths.board), starter: true },
  { field: 'lensFolder', label: 'lens folder', path: paths.lenses, ok: found.has(paths.lenses), starter: true },
]

// Obsidian keeps its vault list in obsidian.json: the open vault first, then the most recently used.
export const vaultsFrom = (text: string): string[] => {
  try {
    const vaults = Object.values((JSON.parse(text) as { vaults?: Record<string, { path?: string; ts?: number; open?: boolean }> }).vaults ?? {})

    return vaults
      .filter(one => typeof one.path === 'string' && one.path.length > 0)
      .sort((a, b) => Number(Boolean(b.open)) - Number(Boolean(a.open)) || (b.ts ?? 0) - (a.ts ?? 0))
      .map(one => one.path as string)
  } catch {
    return []
  }
}

// Where obsidian.json lives on macOS, Linux (plain and Flatpak) and Windows.
export const obsidianConfigs = (home: string, appData: string): string[] => [
  `${home}/Library/Application Support/obsidian/obsidian.json`,
  `${home}/.config/obsidian/obsidian.json`,
  `${home}/.var/app/md.obsidian.Obsidian/config/obsidian/obsidian.json`,
  ...(appData ? [`${appData.replace(/\\/g, '/')}/obsidian/obsidian.json`] : []),
]

// The setup steps after the folder, in order, with what each one is for.
export const STEPS: { field: string; title: string; why: string; starter: string }[] = [
  { field: 'boardNote', title: 'Your board', why: 'The Kanban note your projects live on. Its cards become the anchors you pick from.', starter: 'Kanban.md' },
  { field: 'contextNote', title: 'Your context note', why: 'A note about you and this season: what you are focused on and how you like to work.', starter: 'OpenAugi/AGENT/Slowly Changing Context.md' },
  { field: 'lensFolder', title: 'Your lens folder', why: 'A folder of saved prompts, one note each. They show under Lenses.', starter: 'OpenAugi/AGENT/lenses' },
]

// Search output (one path per line, newest first) as paths inside the notes folder, the first few.
export const candidatesFrom = (stdout: string, notes: string, limit = 5): string[] =>
  [...new Set(stdout.split('\n').map(line => line.trim()).filter(Boolean))]
    .map(path => (path.startsWith(`${notes}/`) ? path.slice(notes.length + 1) : path))
    .filter(path => !path.startsWith('/') && !path.split('/').some(part => part.startsWith('.')))
    .slice(0, limit)

export const starterContext = (today: string): string =>
  [
    '---',
    "description: Who I am, this season's focus, and my projects. Augi reads it.",
    `created: ${today}`,
    '---',
    '',
    '# Context',
    '',
    '## Who I am',
    '',
    'A line or two: your role and what you care about.',
    '',
    "## This season's focus",
    '',
    '**Primary: the one thing that matters most right now.**',
    '',
    'Why it matters, and what done looks like.',
    '',
    '## Projects',
    '',
    '- ',
    '',
  ].join('\n')

export const starterBoard = (): string =>
  [
    '---',
    '',
    'kanban-plugin: board',
    '',
    '---',
    '',
    '## Doing',
    '',
    '- [ ] [[My first project]] — what it is for',
    '\t- [ ] the next step',
    '',
    '',
    '## Next',
    '',
    '',
    '',
  ].join('\n')

export const starterLens = (): string =>
  [
    '---',
    'name: focus-check',
    "description: Am I working on the thing that matters, and what's the next step?",
    "scope: The anchor's note and Agent Log, my context note, and any coding session handed over.",
    'trigger: on-demand',
    'target: chat — answers in the conversation and writes nothing.',
    'kind: personal',
    '---',
    '',
    '# Focus check',
    '',
    "Read the context handed over: the anchor's note, my context note, and any coding session.",
    '',
    'Answer in at most six lines:',
    "1. What I'm working on right now, in one line.",
    "2. Whether it serves the anchor's outcome or my season focus: yes, partly or no, and why.",
    '3. The one next step that moves it forward.',
    "4. Anything I'm drifting into that can wait.",
    '',
  ].join('\n')

const SESSION_KEYS = 'abcde'
// No `p`: it opens the anchor's note while the lens menu is open.
const EDIT_KEYS = 'qwtyui'
const PANE = 'augi'
// How often an open pane checks whether another chat changed its anchor's files.
const SYNC_MS = 10000
// Augi's face (an otter, for old times' sake) on surfaces that draw Svg (desktop, editor, phone); the terminal skips it.
const AUGI_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="14" cy="16" r="7" fill="#8a5a3b"/><circle cx="50" cy="16" r="7" fill="#8a5a3b"/><circle cx="14" cy="16" r="3.5" fill="#5e3b25"/><circle cx="50" cy="16" r="3.5" fill="#5e3b25"/><circle cx="32" cy="34" r="24" fill="#a06a45"/><ellipse cx="32" cy="43" rx="15" ry="11" fill="#e8cfb0"/><circle cx="23" cy="30" r="3.2" fill="#2b1a10"/><circle cx="41" cy="30" r="3.2" fill="#2b1a10"/><circle cx="24" cy="29" r="1" fill="#fff"/><circle cx="42" cy="29" r="1" fill="#fff"/><ellipse cx="32" cy="38" rx="4.5" ry="3" fill="#2b1a10"/><path d="M32 41v3M28 46q4 3 8 0" stroke="#2b1a10" stroke-width="1.5" fill="none" stroke-linecap="round"/><path d="M20 41l-9-2M20 44l-9 1M44 41l9-2M44 44l9 1" stroke="#5e3b25" stroke-width="1" stroke-linecap="round"/></svg>'
const DRAFT_HEAD = 'Context for this chat:'
const HEAD_BYTES = '65536'
// Each chat session's anchor, as [session id, note] pairs, oldest first. The anchor
// belongs to the session: a new one opens with none; a reload or a resume keeps it.
const ANCHORS_KEY = 'anchors'
const ANCHORS_KEPT = 200
// The anchor last chosen in each working folder, as [folder, note] pairs, oldest first:
// what a new chat there starts on.
const FOLDERS_KEY = 'folders'
const FOLDERS_KEPT = 100
// How long after it began a session still counts as new, for carrying an anchor over. An
// older chat that never had one (a resume, a chat open across an update) is left as it is.
const NEW_SESSION_MS = 10 * 60_000
// The title a chat gets in ## Chats when anchoring records it, until its first prompt names it.
export const CHAT_PLACEHOLDER = 'chat'
// The setup steps someone chose to skip; /augi setup clears it.
const SKIPPED_KEY = 'skipped'
// Pins that go into every chat's draft, whatever its anchor: kept in the store, newest last.
const PINS_KEY = 'pins'
const PINS_KEPT = 50
export const EMPTY_TRAIL: Trail = { questions: [], chats: [], saved: [], pins: [] }
// No daily note: the Today card is not drawn.
const NO_TODAY: Today = { pattern: '', path: '', day: '', exists: false, todo: [], mtime: 0 }


const swatch = atom(
  { plugin: 'augi-ui', key: 'swatch' } as const,
  {
    setup: [],
    vaults: [],
    found: {},
    skipped: [],
    season: '',
    cards: [],
    anchor: null,
    notePath: '',
    outcome: '',
    next: [],
    jira: '',
    jiraServer: '',
    taskTemplate: 'built-in',
    today: NO_TODAY,
    trail: EMPTY_TRAIL,
    chatFiles: {},
    chatSeen: {},
    seenAt: 0,
    pinPaths: {},
    hits: [],
    everywhere: [],
    lenses: [],
    quick: { overview: '', standup: '' },
    sessions: [],
    loadedAt: 0,
  } as Swatch,
  { shape: 'swatch-16' },
)
// The two sections and the cards inside them start open; Lenses and Setup start folded.
const CARDS_OPEN = { chatOpen: true, anywhereOpen: true, anchorOpen: true, trailOpen: true, pinsOpen: true, touchedOpen: true, padOpen: true, youOpen: true, todayOpen: true, contextOpen: true }
const picks = atom(
  { plugin: 'augi-ui', key: 'picks' } as const,
  { aboutMe: true, openAugi: true, project: true, today: true } as Picks,
  { shape: 'picks-4' },
)
const view = atom(
  { plugin: 'augi-ui', key: 'view' } as const,
  { ...CARDS_OPEN, anchorsOpen: false, sessionsOpen: false, lensesOpen: false, chatsDoneOpen: false, setupOpen: false, inOpen: true, doneOpen: false, detailsOpen: false, filesOpen: false, taskMode: '', promote: '', logged: 0 } as View,
  { shape: 'view-11' },
)
const made = atom(
  { plugin: 'augi-ui', key: 'made' } as const,
  { session: '', artifacts: [] } as Made,
  { shape: 'made-1' },
)
// The files this chat wrote or edited, tagged with its session like `made`.
const touched = atom(
  { plugin: 'augi-ui', key: 'touched' } as const,
  { session: '', files: [] } as Touched,
  { shape: 'touched-1' },
)

// This chat's scratchpad, newest first: session state, so each chat has its own and it
// outlives a redraw or a reload of this module, but not the session.
const pad = atom({ plugin: 'augi-ui', key: 'pad' } as const, [] as PadEntry[], { shape: 'pad-1' })

// What is typed in the scratchpad's field and not yet kept: saved on every change, so it
// is still there after a click away, a redraw or a reload. No drawing reads the atom (a
// redraw per keystroke would fight the typing); the pane draws `typedPad`, which
// session.start fills from it after a reload.
const padDraft = atom({ plugin: 'augi-ui', key: 'padDraft' } as const, '' as string, { shape: 'pad-draft-1' })
let typedPad = ''

// The session the pane was opened for by itself: it opens once a session, so a window
// that attaches again does not reopen a pane the user closed.
const opened = atom({ plugin: 'augi-ui', key: 'opened' } as const, { session: '-' } as Opened, { shape: 'opened-1' })

// The anchor this chat started on by itself (carried over from the last chat in its
// folder), until the user keeps or changes it or the chat says something. `named` is the
// session and anchor whose log already has this chat under a real title.
const NOT_INHERITED: Inherited = { session: '', note: '', named: '' }
const inherited = atom({ plugin: 'augi-ui', key: 'inherited' } as const, NOT_INHERITED, { shape: 'inherited-1' })

// The task this chat just made: session state, cleared by its dismiss and on clear or resume.
const NO_TASK: JustMade = { note: '', path: '', jira: '', where: '' }
const justMade = atom({ plugin: 'augi-ui', key: 'justMade' } as const, NO_TASK, { shape: 'just-made-2' })

// The first bold line under a heading that names the season or focus,
// e.g. "Primary: the metrics data factory."
export const anchorFrom = (text: string): string => {
  const season = text.split(/^#+ /m).find(part => /^[^\n]*(season|focus)/i.test(part))
  const line = season?.split('\n').find(one => one.startsWith('**'))

  return line?.match(/\*\*(.+?)\*\*/)?.[1] ?? 'No focus line in your context note yet.'
}

export const descriptionFrom = (text: string): string => {
  const found = text.match(/^description:\s*(?:>-?\s*\n\s+)?(.+)$/m)?.[1] ?? ''

  return found.trim()
}

// The first line of text under a heading named Outcome (any level), up to the next heading.
const underOutcome = (body: string): string => {
  let isIn = false
  for (const line of body.split('\n')) {
    const heading = line.match(/^#{1,6}\s+(.+?)\s*$/)
    if (heading) {
      if (isIn) {
        return ''
      }
      isIn = /^outcome\b/i.test(heading[1] ?? '')
      continue
    }
    if (isIn && line.trim() && !/^\s*(?:<!--|%%)/.test(line)) {
      return plain(line.replace(/^\s*(?:[-*]\s+)?(?:\[[ x]\]\s+)?/, ''))
    }
  }

  return ''
}

// The outcome: the text under the note's Outcome heading (where a task template keeps
// it), else `outcome:` in the frontmatter, else an `Outcome:` line in the body
// (`**Outcome:**` too). An empty one falls through to the next.
export const outcomeFrom = (text: string): string => {
  const front = text.match(/^---\n([\s\S]*?)\n---/)
  const fromFront = (front?.[1] ?? '').match(/^outcome:[ \t]*(?:>-?[ \t]*\n[ \t]+)?(.+)$/m)?.[1] ?? ''
  const body = front ? text.slice(front[0].length) : text
  const fromBody = body.match(/^[ \t]*(?:\*\*)?outcome(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?[ \t]*(.+)$/im)?.[1] ?? ''

  return (underOutcome(body) || fromFront.trim() || fromBody.trim()).replace(/^"|"$/g, '')
}

// A scratchpad entry as a Journal line: `- HH:MM text`, further lines indented under it.
const journalLines = (one: PadEntry): string[] => {
  const lines = one.text.trim().split('\n').map(line => line.trimEnd()).filter(line => line.trim())

  return lines.map((line, at) => (at === 0 ? `- ${one.when.slice(11, 16)} ${line}` : `  ${line}`))
}

// Adds a scratchpad entry to the user's note: under its Journal heading (any level), under
// the dated heading for the entry's day, as that day's last line. A missing day heading is
// made at the level the note's dated headings use, first when they run newest first, else
// last. A note with no Journal gets one above its Agent Log heading, else at the end.
// Every other line, the frontmatter included, is left as it was.
export const addJournalEntry = (text: string, one: PadEntry): string => {
  const entry = journalLines(one)
  if (entry.length === 0) {
    return text
  }
  const day = one.when.slice(0, 10)
  const lines = text.split('\n')
  const front = lines[0] === '---' ? lines.indexOf('---', 1) + 1 : 0
  const headingAt = (at: number): { level: number; title: string } | null => {
    const found = at >= front ? lines[at]?.match(/^(#{1,6})\s+(.*?)\s*$/) : null

    return found ? { level: (found[1] ?? '').length, title: found[2] ?? '' } : null
  }
  const lastFilled = (from: number, to: number): number => {
    let at = to - 1
    while (at > from && !lines[at]?.trim()) {
      at -= 1
    }

    return at
  }
  const journal = lines.findIndex((_, at) => /^journal$/i.test(headingAt(at)?.title ?? ''))
  if (journal < 0) {
    const block = ['# Journal', '', `### ${day}`, ...entry]
    const log = lines.findIndex((_, at) => /^agent log$/i.test(headingAt(at)?.title ?? ''))
    if (log >= 0) {
      lines.splice(log, 0, ...(log > 0 && lines[log - 1]?.trim() ? [''] : []), ...block, '')

      return lines.join('\n')
    }
    if (!text.trim()) {
      return `${block.join('\n')}\n`
    }

    return `${text}${text.endsWith('\n\n') ? '' : text.endsWith('\n') ? '\n' : '\n\n'}${block.join('\n')}\n`
  }
  const level = headingAt(journal)?.level ?? 1
  let end = journal + 1
  while (end < lines.length && (headingAt(end)?.level ?? 7) > level) {
    end += 1
  }
  const dated: { at: number; level: number; date: string }[] = []
  for (let at = journal + 1; at < end; at += 1) {
    const heading = headingAt(at)
    const date = heading?.title.match(/^(\d{4}-\d{2}-\d{2})\b/)?.[1]
    if (heading && date && heading.level > level) {
      dated.push({ at, level: heading.level, date })
    }
  }
  const dayLevel = dated[0]?.level ?? Math.min(6, Math.max(3, level + 1))
  const days = dated.filter(one => one.level === dayLevel)
  const same = days.find(one => one.date === day)
  if (same) {
    let close = same.at + 1
    while (close < end && (headingAt(close)?.level ?? 7) > dayLevel) {
      close += 1
    }
    lines.splice(lastFilled(same.at, close) + 1, 0, ...entry)

    return lines.join('\n')
  }
  const heading = `${'#'.repeat(dayLevel)} ${day}`
  const first = days[0]
  const last = days[days.length - 1]
  if (first && last && first.date > last.date) {
    lines.splice(first.at, 0, heading, ...entry, '')
  } else {
    lines.splice(lastFilled(journal, end) + 1, 0, '', heading, ...entry)
  }

  return lines.join('\n')
}

// The open checkboxes under the note's Next heading (# or ##), up to the next heading
// at that level or above.
export const nextFrom = (text: string): SubItem[] => {
  const tasks: SubItem[] = []
  let level = 0

  for (const line of text.split('\n')) {
    const heading = line.match(/^(#{1,6})\s+(.+?)\s*$/)
    if (heading) {
      if (level && heading[1].length <= level) {
        break
      }
      if (!level && /^next$/i.test(heading[2])) {
        level = heading[1].length
      }
      continue
    }
    const task = level ? line.match(/^\s*- \[ \] (.*\S.*)$/) : null
    if (task) {
      tasks.push(subItem(task[1] ?? ''))
    }
  }

  return tasks
}

// The lens menu: the picks in their order (the lens picks setting is the only filter).
// With no picks, every lens in the folder.
export const lensNames = (files: string[]): string[] =>
  P.picks.length > 0 ? P.picks.filter(name => files.includes(name)) : [...files].sort()

// A typed lens name as a file name: no path or link characters, no `.md`, dashes for spaces.
export const lensFileName = (typed: string): string =>
  typed
    .trim()
    .replace(/\.md$/i, '')
    .replace(/[\s\\/:*?"<>|#^[\]]+/g, '-')
    .replace(/^[-.]+|-+$/g, '')

// A new lens file, in the contract starterLens follows, waiting for its prompt.
export const newLensSkeleton = (name: string): string =>
  [
    '---',
    `name: ${name}`,
    'description: What this lens answers, in one line.',
    "scope: The anchor's note and Agent Log, my context note, and any coding session handed over.",
    'trigger: on-demand',
    'target: chat — answers in the conversation and writes nothing.',
    'kind: personal',
    '---',
    '',
    `# ${name}`,
    '',
    'Paste the prompt here.',
    '',
  ].join('\n')

const linkName = (text: string): string | null => {
  const found = text.match(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/)?.[1]

  return found ? found.trim() : null
}

const plain = (text: string): string =>
  text.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, '$2').replace(/\[\[([^\]]+)\]\]/g, '$1').replace(/\s+/g, ' ').trim()

// A checkbox's text, keeping the note its first wikilink points at.
// A sub-item keeps its first link; the Kanban block id at its end is dropped.
const subItem = (text: string): SubItem => ({ text: plain(text.replace(/\s\^\w+\s*$/, '')), note: linkName(text) })

// The open cards above the Archive that link a note, each with its open sub-items.
export const kanbanCards = (text: string): Card[] => {
  const cards: Card[] = []
  let column = ''
  let current: Card | null = null

  for (const line of (text.split(/^\*\*\*$/m)[0] ?? '').split('\n')) {
    const heading = line.match(/^## (.+)$/)
    if (heading) {
      column = heading[1].trim()
      current = null
      continue
    }
    const top = line.match(/^- \[ \] (.*)$/)
    if (top) {
      const note = linkName(top[1])
      const detail = plain(top[1].replace(/\[\[[^\]]+\]\]/, '').replace(/\^\w+\s*$/, '').replace(/^\s*[—-]\s*/, ''))
      current = note ? { column, note, detail, tasks: [] } : null
      if (current) {
        cards.push(current)
      }
      continue
    }
    if (/^- /.test(line)) {
      current = null
      continue
    }
    const task = line.match(/^\t- \[ \] (.+)$/)
    if (current && task) {
      current.tasks.push(subItem(task[1] ?? ''))
    }
  }

  return cards
}

// The note's Jira key: `jira:` in its frontmatter, else a `Jira: KEY` line in the body. Or empty.
export const jiraFrom = (text: string): string => {
  const front = text.match(/^---\n([\s\S]*?)\n---/)
  const fromFront = ((front?.[1] ?? '').match(/^jira:[ \t]*(.*)$/m)?.[1] ?? '').trim().replace(/^["']|["']$/g, '')
  const body = front ? text.slice(front[0].length) : text

  return fromFront || (body.match(/^Jira:[ \t]*(\S+)[ \t]*$/m)?.[1] ?? '')
}

// A body line that links the note's issue: a key and nothing else.
const JIRA_LINE = /^Jira:[ \t]*\S+[ \t]*$/
const JIRA_KEY = /^[A-Z][A-Z0-9]+-\d+$/
const JIRA_URL = /^https?:\/\/\S*\/browse\/([A-Z][A-Z0-9]+-\d+)(?:[/?#]\S*)?$/

// A Jira key from one word: a bare key (`PROJ-123`, capitals only) or an issue URL
// (`…/browse/PROJ-123`), brackets and trailing punctuation dropped. Else empty.
export const jiraKeyFrom = (text: string): string => {
  const bare = text.trim().replace(/^[([<]+|[)\]>:,;.]+$/g, '')

  return JIRA_KEY.test(bare) ? bare : (bare.match(JIRA_URL)?.[1] ?? '')
}

// What was typed for a new task: a Jira key or issue URL at its start or end is
// taken out of the title and kept as the key.
export const parseNewTask = (typed: string): { title: string; jira: string } => {
  const words = typed.trim().split(/\s+/).filter(Boolean)
  const keyOf = (word: string | undefined): string => (word ? jiraKeyFrom(word) : '')
  let jira = keyOf(words[words.length - 1])
  if (jira) {
    words.pop()
  } else {
    jira = keyOf(words[0])
    if (jira) {
      words.shift()
    }
  }

  return { title: words.join(' ').replace(/^[\s:—–-]+|[\s:—–-]+$/g, ''), jira }
}

// A title as a note name: nothing a file name or a wikilink can't hold.
export const taskNoteName = (title: string): string =>
  title.replace(/[\\/:]/g, '-').replace(/[*?"<>|#^[\]]/g, '').replace(/\s+/g, ' ').trim().replace(/^[.\s]+/, '')

// The shipped task template (templates/Task.md), for a notes folder without its own.
export const TASK_TEMPLATE = [
  'Up: [[PMOC - ]]',
  '',
  '### Outcome',
  '',
  '',
  '### Verify',
  '',
  '',
  '# Journal',
  '',
  '### {{date:YYYY-MM-DD}}',
  '',
  '',
].join('\n')

// Which task template a new task is made from: the built-in one (the notes folder has
// none), the notes folder's own, or its own from before the Agent Log left the note: one
// that still has an Agent Log heading or embeds the log.
export const taskTemplateState = (text: string): 'built-in' | 'yours' | 'old' => {
  if (!text.trim()) {
    return 'built-in'
  }

  return /^#{1,6}\s+Agent Log\s*$/im.test(text) || /!\[\[Agent Log - /.test(text) ? 'old' : 'yours'
}

// A new task note from a template: date and title filled, `Up:` pointing at the anchor
// (the line gone without one), and the Jira key when there is one (see withJira).
export const taskNoteText = (template: string, name: string, today: string, up: string | null, jira: string): string => {
  const lines = template.replace(/\{\{date(?::YYYY-MM-DD)?\}\}/g, today).replace(/\{\{title\}\}/g, name).split('\n')
  const at = lines.findIndex(line => /^Up:/.test(line))
  if (at >= 0 && up) {
    lines[at] = `Up: [[${up}]]`
  } else if (at >= 0) {
    // With a blank line on both sides, one of them goes too.
    lines.splice(at, lines[at + 1] === '' && (at === 0 || lines[at - 1] === '') ? 2 : 1)
  }
  const text = lines.join('\n')

  return jira ? withJira(text, jira) : text
}

// The note with its Jira key set to `key`, or removed when `key` is empty. A note with
// frontmatter keeps it there as `jira:` (replaced, or added last). A note without
// frontmatter gets no frontmatter made for it: the key is a `Jira: KEY` line in the body,
// under the `Up:` line when there is one, else first. Every other byte stays as it was.
export const withJira = (text: string, key: string): string => {
  const front = text.match(/^---\n([\s\S]*?\n)?---(?=\n|$)/)
  let fields = front?.[1] ?? ''
  const hasField = Boolean(front) && /^jira:.*$/m.test(fields)
  const lines = text.slice(front ? front[0].length : 0).split('\n')
  // Only a line that is a key and nothing else is the link; `Jira: waiting on…` is the user's prose.
  const at = lines.findIndex(line => JIRA_LINE.test(line))
  if (!key) {
    fields = fields.replace(/^jira:.*\n/m, '')
    if (at >= 0) {
      // With a blank line on both sides (or first, then a blank), one blank goes too.
      lines.splice(at, lines[at + 1] === '' && (at === 0 || lines[at - 1] === '') ? 2 : 1)
    }
  } else if (hasField) {
    fields = fields.replace(/^jira:.*$/m, `jira: ${key}`)
  } else if (at >= 0) {
    lines[at] = `Jira: ${key}`
  } else if (front) {
    fields = `${fields}jira: ${key}\n`
  } else if (!text) {
    return `Jira: ${key}\n`
  } else {
    const up = lines.findIndex(line => /^Up:/.test(line))
    lines.splice(up + 1, 0, ...(up >= 0 ? [`Jira: ${key}`] : [`Jira: ${key}`, '']))
  }

  return `${front ? `---\n${fields}---` : ''}${lines.join('\n')}`
}

// Where an issue opens in the browser.
export const jiraBrowseUrl = (site: string, key: string): string => `${site.replace(/\/+$/, '')}/browse/${key}`

// Names a Jira MCP server often goes by, for finding one when the setting is empty.
export const JIRA_SERVERS = ['atlassian', 'Atlassian', 'claude.ai Atlassian', 'jira']

const serverKey = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]+/g, '_')

// The Jira MCP server among the session's tools (`mcp__<server>__<tool>`): the one the
// setting names, else the first of the likely names. Empty when none is there.
export const jiraServerFrom = (tools: { name: string; mcp: boolean }[], wanted: string): string => {
  const servers = tools
    .filter(tool => tool.mcp && tool.name.startsWith('mcp__'))
    .map(tool => tool.name.slice(5).split('__')[0] ?? '')
    .filter(Boolean)
  const has = (name: string): string | undefined => servers.find(one => serverKey(one) === serverKey(name))
  if (wanted.trim()) {
    return has(wanted) ? wanted.trim() : ''
  }
  for (const name of JIRA_SERVERS) {
    const found = has(name)
    if (found) {
      return found
    }
  }

  return ''
}

// How a draft tells the chat which Jira to use.
const jiraTools = (project: string, server: string): string =>
  `${project ? ` in project ${project}` : ''}${server ? ` using the ${server} MCP server` : ' with whatever Jira tools you have; if you have none, say so and stop'}`

// The find-or-offer-to-create steps, shared by "find in Jira" and a new task's draft.
const jiraFindSteps = (what: string, project: string, server: string): string[] => [
  `- Search Jira${jiraTools(project, server)} for issues that match ${what}. List up to five likely matches, one line each: KEY · summary · status. Do not create or change anything in Jira.`,
  `- When I pick one, link it in the note: add the line \`Jira: <KEY>\` under its \`Up:\` line, or set \`jira:\` in its frontmatter if the note has frontmatter (this once you may write that line: I'm asking). If nothing matches, offer to create an issue and wait for my yes.`,
]

// What Jira the drafts name: the project key and the MCP server, either may be empty.
export type JiraAsk = { project: string; server: string }

// The prompt "find in Jira" leaves in the box, unsent: the chat searches Jira for the
// anchor's issue and links the one the user picks.
export const findInJiraDraft = (notePath: string, note: string, ask: JiraAsk): string =>
  [
    DRAFT_HEAD,
    `- Anchor note: ${notePath ? `"${notePath}"` : `[[${note}]] (find it in "${P.notes}")`} (${note}). Read it; it has no Jira issue linked yet.`,
    ...jiraFindSteps('this note', ask.project, ask.server),
    `Instruction: find the Jira issue for ${note}`,
  ].join('\n')

// The prompt "↑ update Jira" leaves in the box, unsent, when the lens folder has no
// jira-update lens: a short status update, shown first, posted only on a yes.
export const jiraUpdateDraft = (notePath: string, note: string, logPath: string, key: string, ask: JiraAsk): string =>
  [
    DRAFT_HEAD,
    `- Anchor note: ${notePath ? `"${notePath}"` : `[[${note}]] (find it in "${P.notes}")`} (${note}): read its Outcome, Verify and Journal sections, and its Agent Log "${logPath}".`,
    `- Read Jira issue ${key} and its recent comments${jiraTools('', ask.server)}, so the update covers only what is new since the last one.`,
    '- Write a status update of at most six lines: what moved, what is next, what is blocked. Show it to me first.',
    `- Post it as a comment on ${key} only after I say yes. Change the issue's status only if I ask. Don't change any files.`,
    `Instruction: update Jira ${key}`,
  ].join('\n')

// Puts `- [ ] [[note]]` on the board as a top-level card: after the anchor's card and
// its sub-items; else at the end of the column named `into` (by its heading, any case);
// else at the end of the first column's cards. Never in the archive; every other line
// stays as it was.
export const insertCard = (board: string, note: string, anchor: string | null, into = ''): string => {
  const lines = board.split('\n')
  // Cards end at the archive, or at the Kanban plugin's settings block on a board with no archive.
  const archive = lines.findIndex(line => line === '***' || /^%% kanban:settings/.test(line))
  const end = archive >= 0 ? archive : lines.length
  const card = `- [ ] [[${note}]]`
  const at = anchor === null ? -1 : lines.findIndex((line, index) => index < end && /^- \[.\] /.test(line) && linkName(line) === anchor)
  if (at >= 0) {
    let after = at + 1
    while (after < end && /^\s+\S/.test(lines[after] ?? '')) {
      after += 1
    }
    lines.splice(after, 0, card)

    return lines.join('\n')
  }
  const isColumn = (line: string, index: number): boolean => index < end && line.startsWith('## ')
  const wanted = into.trim().toLowerCase()
  const named = wanted ? lines.findIndex((line, index) => isColumn(line, index) && line.slice(3).trim().toLowerCase() === wanted) : -1
  const column = named >= 0 ? named : lines.findIndex(isColumn)
  if (column < 0) {
    lines.splice(end, 0, ...(end > 0 && lines[end - 1] !== '' ? [''] : []), card, '')

    return lines.join('\n')
  }
  let last = column + 1
  while (last < end && !lines[last]?.startsWith('## ')) {
    last += 1
  }
  last -= 1
  while (last > column && (lines[last] ?? '').trim() === '') {
    last -= 1
  }
  if (last === column) {
    // An empty column: the card goes under the blank line after its heading.
    if (lines[column + 1] === '') {
      lines.splice(column + 2, 0, card)
    } else {
      lines.splice(column + 1, 0, '', card)
    }

    return lines.join('\n')
  }
  lines.splice(last + 1, 0, card)

  return lines.join('\n')
}

// Puts `\t- [ ] [[note]]` on the board as a linked sub-item of `parent`: the last of the
// parent card's indented lines. A parent that is itself a linked sub-item of a card gets
// the new one beside it, under the same card, after the parent's own deeper lines. A
// parent not on the board (or only in the archive) changes nothing: the caller decides.
// Never in the archive or the Kanban settings block; every other line stays as it was.
export const insertSubItem = (board: string, note: string, parent: string): string => {
  const lines = board.split('\n')
  const archive = lines.findIndex(line => line === '***' || /^%% kanban:settings/.test(line))
  const end = archive >= 0 ? archive : lines.length
  const item = `\t- [ ] [[${note}]]`
  const card = lines.findIndex((line, index) => index < end && /^- \[.\] /.test(line) && linkName(line) === parent)
  if (card >= 0) {
    let after = card + 1
    while (after < end && /^\s+\S/.test(lines[after] ?? '')) {
      after += 1
    }
    lines.splice(after, 0, item)

    return lines.join('\n')
  }
  // A linked sub-item: an indented checkbox under a top-level card.
  let isUnderCard = false
  for (let at = 0; at < end; at += 1) {
    const line = lines[at] ?? ''
    if (/^- /.test(line) || /^#/.test(line)) {
      isUnderCard = /^- \[.\] /.test(line)
      continue
    }
    if (isUnderCard && /^\t- \[.\] /.test(line) && linkName(line) === parent) {
      let after = at + 1
      while (after < end && /^\t\s+\S/.test(lines[after] ?? '')) {
        after += 1
      }
      lines.splice(after, 0, item)

      return lines.join('\n')
    }
  }

  return board
}

// What a note anchors as: its top-level card; else, for a linked sub-item of a card, a
// card-shaped value in the parent's column (`under <parent>`, no tasks of its own); else null.
export const anchorCard = (cards: Card[], note: string): Card | null => {
  const card = cards.find(one => one.note === note)
  if (card) {
    return card
  }
  const parent = cards.find(one => one.tasks.some(task => task.note === note))

  return parent ? subCard(parent, note) : null
}

// A linked sub-item of `parent` as an anchor.
export const subCard = (parent: Card, note: string): Card => ({ column: parent.column, note, detail: `under ${parent.note}`, tasks: [] })

// The prompt a new task leaves in the box, unsent: the chat fills in the note's
// Outcome and Verify from what it has done, and keeps the context in the Agent Log.
// With `jira` (Jira is on and no key was typed) it also looks for the task's issue.
export const newTaskDraft = (notePath: string, logPath: string, name: string, jira: JiraAsk | null = null): string =>
  [
    DRAFT_HEAD,
    `- A task note was just made from this chat: "${notePath}".`,
    "- From what this chat has done, fill in what goes under the note's Outcome and Verify headings (the outcome as a state of the world, and how we will know it is reached). Add no headings and no frontmatter. This once you may write that note: I'm asking.",
    `- Put the context worth keeping (decisions, files, links) under ## Context in its Agent Log "${logPath}", one line each on why it matters.`,
    ...(jira ? jiraFindSteps('this task', jira.project, jira.server) : []),
    'Reply in a few lines with what you wrote.',
    `Instruction: fill in the new task ${name}`,
  ].join('\n')

// One Agent Log per anchor, 1:1 by name. (Functions keep the old word, trail.)
const safe = (note: string): string => note.replace(/[\\/:]/g, '-')
export const logName = (note: string): string => `Agent Log - ${safe(note)}`
export const trailPath = (note: string): string => `${P.trails}/${logName(note)}.md`
// Where v0.6 and earlier kept it, for moving it over.
export const oldTrailPath = (note: string): string => `${P.trails}/Trail - ${safe(note)}.md`

export const trailTemplate = (note: string, today: string): string =>
  [
    '---',
    'type: document',
    `description: The Agent Log for ${note}: context worth keeping, questions, chats started on it and notes saved from them. Agents write here; ${note} is the person's own.`,
    `created: ${today}`,
    '---',
    '',
    logIntro(note),
    '',
    '## Context',
    '',
    '## Questions',
    '',
    '## Chats',
    '',
    '## Saved',
    '',
  ].join('\n')

// The log's first body line links the human note, so the note shows it as a backlink.
const logIntro = (note: string): string =>
  `Human note: [[${note}]]. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Each chat is one line: when it started, its goal, its id; one that is finished stays listed, marked \`(done)\`.`
// The intros earlier versions wrote, replaced with the current one when the pane writes the log.
const oldLogIntros = (note: string): string[] => [
  `Project: [[${note}]], embedded there under # Agent Log. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Hidden chats stay listed, marked \`(hidden)\`.`,
  `Human note: [[${note}]]. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Hidden chats stay listed, marked \`(hidden)\`.`,
]

// An older log in the current shape: no title H1 (Agent Log or Trail), no seen box,
// the human-note link first. Removes only those exact lines and the blank each leaves.
export const tidyTrail = (text: string, note: string): string => {
  const dropped = new Set([`# ${logName(note)}`, `# Trail - ${safe(note)}`, '- [ ] seen', '- [x] seen'])
  const lines: string[] = []
  let gap = false
  for (const line of text.split('\n')) {
    if (dropped.has(line)) {
      gap = gap || lines.length === 0 || lines[lines.length - 1] === ''
      continue
    }
    if (gap && line === '') {
      gap = false
      continue
    }
    gap = false
    lines.push(oldLogIntros(note).includes(line) ? logIntro(note) : line)
  }
  if (!lines.some(line => line.startsWith(`Human note: [[${note}]]`))) {
    // Under the frontmatter (one blank between), else at the top.
    const close = lines[0] === '---' ? lines.indexOf('---', 1) : -1
    if (close < 0) {
      lines.splice(0, 0, logIntro(note), '')
    } else if (lines[close + 1] === '') {
      lines.splice(close + 2, 0, logIntro(note), '')
    } else {
      lines.splice(close + 1, 0, '', logIntro(note), '')
    }
  }

  return lines.join('\n')
}

// A chat's line under ## Chats. `(hidden)` is what a finished chat was marked before `(done)`.
const CHAT_LINE = /^- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) · (.*) · `([^`]+)`( \((?:done|hidden)\))?$/

const QUESTION_LINE = /^- \[([ x])\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) · (.*)$/

export const parseTrail = (text: string): Trail => {
  const questions: Question[] = []
  const chats: Chat[] = []
  const saved: string[] = []
  const pins: Pin[] = []
  let section = ''

  for (const line of text.split('\n')) {
    const heading = line.match(/^## (.+)$/)
    if (heading) {
      section = heading[1].trim()
      continue
    }
    if (section === 'Questions') {
      const found = line.match(QUESTION_LINE)
      if (found) {
        questions.push({ done: found[1] === 'x', when: found[2], text: found[3] })
      }
    }
    if (section === 'Chats') {
      const found = line.match(CHAT_LINE)
      if (found) {
        chats.push({ when: found[1], title: found[2], id: found[3], done: Boolean(found[4]) })
      }
    }
    if (section === 'Context') {
      const pin = parsePin(line)
      if (pin) {
        pins.push(pin)
      }
    }
    if (section === 'Saved' && line.startsWith('- ')) {
      const name = linkName(line)
      if (name) {
        saved.push(name)
      }
    }
  }

  return { questions, chats, saved, pins }
}

const questionLine = (one: Question): string => `- [${one.done ? 'x' : ' '}] ${one.when} · ${one.text.replace(/\s+/g, ' ').trim()}`

// Ticks a question done, or opens it again.
export const setDone = (text: string, one: Question, done: boolean): string =>
  text
    .split('\n')
    .map(line => {
      const found = line.match(QUESTION_LINE)

      return found && found[2] === one.when && found[3] === one.text ? questionLine({ ...one, done }) : line
    })
    .join('\n')

// ## Context lines: `- [x] [[Note]]` for a note, `- [ ] kind: \`target\` · label` for the rest.
const PIN_LINE = /^- \[([ x])\] (?:\[\[([^\]|]+)(?:\|[^\]]*)?\]\]|(file|session|artifact|url): `([^`]+)`)(?: · (.*))?\s*$/

export const parsePin = (line: string): Pin | null => {
  const found = line.match(PIN_LINE)
  if (!found) {
    return null
  }
  const on = found[1] === 'x'
  if (found[2]) {
    return { on, kind: 'note', target: found[2].trim(), label: (found[5] ?? '').trim() || found[2].trim() }
  }

  return { on, kind: found[3] as PinKind, target: found[4], label: (found[5] ?? '').trim() || found[4] }
}

const clean = (text: string): string => text.replace(/[`\n]/g, ' ').replace(/\s+/g, ' ').trim()

export const pinLine = (pin: Pin): string => {
  const box = `- [${pin.on ? 'x' : ' '}]`
  if (pin.kind === 'note') {
    return pin.label && pin.label !== pin.target ? `${box} [[${pin.target}]] · ${clean(pin.label)}` : `${box} [[${pin.target}]]`
  }
  const label = clean(pin.label)

  return label && label !== pin.target ? `${box} ${pin.kind}: \`${pin.target}\` · ${label}` : `${box} ${pin.kind}: \`${pin.target}\``
}

const samePin = (a: Pin, b: Pin): boolean => a.kind === b.kind && a.target === b.target

// Adds a pin at the end of ## Context; one already there stays as it is. A trail
// without the section gets it above ## Questions, or at the end.
export const addPin = (text: string, pin: Pin): string => {
  if (parseTrail(text).pins.some(one => samePin(one, pin))) {
    return text
  }
  const lines = text.split('\n')
  const at = lines.findIndex(line => line.trim() === '## Context')
  if (at >= 0) {
    let end = at + 1
    while (end < lines.length && !lines[end].startsWith('## ')) {
      end += 1
    }
    let last = end - 1
    while (last > at && lines[last].trim() === '') {
      last -= 1
    }
    const added = last === at ? ['', pinLine(pin)] : [pinLine(pin)]
    const after = last + 1 + added.length
    lines.splice(last + 1, 0, ...added)
    if (lines[after]?.startsWith('## ')) {
      lines.splice(after, 0, '')
    }

    return lines.join('\n')
  }
  const questions = lines.findIndex(line => line.trim() === '## Questions')
  if (questions >= 0) {
    lines.splice(questions, 0, '## Context', '', pinLine(pin), '')

    return lines.join('\n')
  }

  return `${text.trimEnd()}\n\n## Context\n\n${pinLine(pin)}\n`
}

const changePin = (text: string, pin: Pin, change: (one: Pin) => Pin | null): string => {
  let section = ''
  const out: string[] = []
  for (const line of text.split('\n')) {
    const heading = line.match(/^## (.+)$/)
    if (heading) {
      section = heading[1].trim()
    }
    const found = section === 'Context' ? parsePin(line) : null
    if (found && samePin(found, pin)) {
      const next = change(found)
      if (next) {
        out.push(pinLine(next))
      }
      continue
    }
    out.push(line)
  }

  return out.join('\n')
}

export const setPinOn = (text: string, pin: Pin, on: boolean): string => changePin(text, pin, one => ({ ...one, on }))

export const removePin = (text: string, pin: Pin): string => changePin(text, pin, () => null)

const baseName = (path: string): string => path.split('/').pop() ?? path

// What you paste becomes a pin: a URL, a [[note]], a note in the vault, a session or any file.
export const pinFrom = (input: string, notes: string, home: string): Pin | null => {
  const value = input.trim().replace(/^["']|["']$/g, '')
  if (!value) {
    return null
  }
  if (/^https?:\/\//.test(value)) {
    const isArtifact = /^https:\/\/claude\.ai\/(?:code\/)?artifact\//.test(value)
    const label = isArtifact ? 'artifact' : value.replace(/^https?:\/\//, '').slice(0, 60)

    return { on: true, kind: isArtifact ? 'artifact' : 'url', target: value, label }
  }
  const wiki = value.match(/^\[\[([^\]|]+)/)
  if (wiki) {
    return { on: true, kind: 'note', target: wiki[1].trim(), label: wiki[1].trim() }
  }
  const path = value.startsWith('~/') ? `${home}${value.slice(1)}` : value
  if (notes && path.startsWith(`${notes}/`) && path.endsWith('.md')) {
    const name = baseName(path).replace(/\.md$/, '')

    return { on: true, kind: 'note', target: name, label: name, path }
  }
  if (path.endsWith('.jsonl') && (path.includes('/.claude/projects/') || path.includes('/.codex/'))) {
    return { on: true, kind: 'session', target: path, label: baseName(path) }
  }

  return { on: true, kind: 'file', target: path, label: baseName(path) }
}

export const sessionPin = (one: Session): Pin => ({ on: true, kind: 'session', target: one.path, label: `${one.repo} · ${one.title}` })

// The command that picks a session back up: claude's by id, codex's by path.
const resumeOf = (path: string): string =>
  !path.includes('/.codex/') && path.endsWith('.jsonl') ? `claude --resume ${baseName(path).replace(/\.jsonl$/, '')}` : path

// The paths in an OpenAugi search answer, vault-relative, once each. Its text can
// be the JSON itself or a { result } wrapping it as a string.
export const pathsFromSearch = (text: string): string[] => {
  let value: unknown = text
  for (let depth = 0; depth < 3 && typeof value === 'string'; depth += 1) {
    try {
      value = JSON.parse(value)
    } catch {
      return []
    }
    if (value && typeof value === 'object' && typeof (value as { result?: unknown }).result === 'string') {
      value = (value as { result: string }).result
    }
  }
  const results = (value as { results?: unknown })?.results
  const paths: string[] = []
  for (const one of Array.isArray(results) ? results : []) {
    const path = (one as { source_path?: unknown })?.source_path
    if (typeof path === 'string' && path.endsWith('.md') && !paths.includes(path)) {
      paths.push(path)
    }
  }

  return paths
}

// Exactly what a draft from the pane puts in, in the order the draft names it.
export type ContextItem = { key: string; label: string; path: string; kind: 'note' | 'pin' | 'search' }

export const contextItems = (pins: Pin[], choice: Picks, anchor: Anchor | null, everywhere: Pin[] = [], today = ''): ContextItem[] => {
  const items: ContextItem[] = []
  if (anchor && choice.project) {
    items.push({ key: 'in-note', label: anchor.note, path: anchor.notePath, kind: 'note' })
    items.push({ key: 'in-trail', label: `Trail - ${anchor.note}`, path: anchor.trail, kind: 'note' })
  }
  for (const pin of anchor ? pins.filter(one => one.on) : []) {
    items.push({ key: `in-pin-${pin.kind}-${pin.target}`, label: pin.label, path: pin.kind === 'note' ? pin.path ?? '' : pin.target, kind: 'pin' })
  }
  for (const pin of extraPins(pins, everywhere, Boolean(anchor))) {
    items.push({ key: `in-all-${pin.kind}-${pin.target}`, label: pin.label, path: pin.kind === 'note' ? pin.path ?? '' : pin.target, kind: 'pin' })
  }
  if (choice.aboutMe) {
    items.push({ key: 'in-context', label: noteName(P.context), path: P.context, kind: 'note' })
    items.push({ key: 'in-board', label: noteName(P.board), path: P.board, kind: 'note' })
  }
  if (today && choice.today) {
    items.push({ key: 'in-today', label: noteName(today), path: today, kind: 'note' })
  }
  if (choice.openAugi) {
    items.push({ key: 'in-search', label: 'notes the chat finds by searching (it names them in its answer)', path: '', kind: 'search' })
  }

  return items
}

// One draft line for a ticked pin.
const pinDraftLine = (pin: Pin): string => {
  if (pin.kind === 'note') {
    return pin.path ? `- Note: "${pin.path}" (${pin.target}).` : `- Note: [[${pin.target}]] (find it in "${P.notes}").`
  }
  if (pin.kind === 'session') {
    return `- Coding session (${pin.label}): "${pin.target}". It is large JSONL, so start with tail -c 60000.`
  }
  if (pin.kind === 'artifact') {
    return `- Artifact (${pin.label}): ${pin.target}. Read it with the Artifact tool's read action.`
  }

  return pin.kind === 'url' ? `- Link (${pin.label}): ${pin.target}` : `- File: "${pin.target}".`
}

const noteName = (path: string): string => (path.split('/').pop() ?? path).replace(/\.md$/, '')

// A Claude session file is named by its id, so a click can copy the resume command.
export const resumeCommand = (session: Session): string => (session.tool === 'claude' ? resumeOf(session.path) : session.path)

const chatLine = (chat: Chat): string =>
  `- ${chat.when} · ${chat.title.replace(/·|`/g, ' ').trim()} · \`${chat.id}\`${chat.done ? ' (done)' : ''}`

// Adds a chat at the top of ## Chats, newest first; a chat already listed stays as it is.
export const addChat = (text: string, chat: Chat): string => {
  if (parseTrail(text).chats.some(one => one.id === chat.id)) {
    return text
  }
  const lines = text.split('\n')
  const at = lines.findIndex(line => line.trim() === '## Chats')
  if (at < 0) {
    return `${text.trimEnd()}\n\n## Chats\n\n${chatLine(chat)}\n`
  }
  lines.splice(at + 2, 0, chatLine(chat))

  return lines.join('\n')
}

// Marks a chat finished, or open again; its line stays where it is.
export const setChatDone = (text: string, id: string, done: boolean): string =>
  text
    .split('\n')
    .map(line => {
      const found = line.match(CHAT_LINE)

      return found && found[3] === id ? chatLine({ when: found[1] ?? '', title: found[2] ?? '', id, done }) : line
    })
    .join('\n')

// Takes a chat's line out of the log (it moved to another anchor's); every other line stays.
export const removeChat = (text: string, id: string): string =>
  text
    .split('\n')
    .filter(line => line.match(CHAT_LINE)?.[3] !== id)
    .join('\n')

// When a chat was last active, short: the time today, else the month and day. Empty when unknown.
export const lastActive = (mtimeMs: number, now: number): string => {
  if (!mtimeMs) {
    return ''
  }
  const at = stamp(mtimeMs)

  return at.slice(0, 10) === stamp(now).slice(0, 10) ? at.slice(11) : at.slice(5, 10)
}

// Names a chat recorded with the placeholder title; a real title stays as it is.
export const retitleChat = (text: string, id: string, title: string): string =>
  text
    .split('\n')
    .map(line => {
      const found = line.match(CHAT_LINE)

      return found && found[3] === id && (found[2] ?? '').trim() === CHAT_PLACEHOLDER
        ? chatLine({ when: found[1] ?? '', title, id, done: Boolean(found[4]) })
        : line
    })
    .join('\n')

// The session → anchor pairs with this session's set (or dropped, for null), most
// recent last, the oldest let go past `kept`. An empty note is kept: it says the session
// chose to have no anchor, so none is carried over to it.
export const withAnchor = (pairs: [string, string][], id: string, note: string | null, kept = ANCHORS_KEPT): [string, string][] => {
  const others = pairs.filter(([one]) => one !== id)

  return (note === null ? others : [...others, [id, note] as [string, string]]).slice(-kept)
}

// The folder a chat's anchor is remembered under: its working folder, a Claude Code
// worktree counted as the repository it belongs to.
export const folderKey = (cwd: string): string => cwd.replace(/\/+$/, '').replace(/\/\.claude\/worktrees\/[^/]+$/, '')

// What a chat is called in its anchor's log when no pane draft names it: its prompt's
// first line. Empty for a slash command or a line that is only markup.
export const promptTitle = (text: string): string => {
  const first = text.split('\n').map(line => line.trim()).find(Boolean) ?? ''

  return first.startsWith('/') ? '' : first.replace(/<[^>]+>/g, ' ').replace(/[·`]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 70)
}

const PIN_KINDS: PinKind[] = ['note', 'file', 'session', 'artifact', 'url']

// What the store holds under the pins key, read defensively.
export const storedPins = (value: unknown): Pin[] =>
  Array.isArray(value)
    ? value.filter((one): one is Pin => {
        const pin = one as Partial<Pin> | null
        return Boolean(pin) && typeof pin?.on === 'boolean' && typeof pin.target === 'string' && typeof pin.label === 'string' && PIN_KINDS.includes(pin.kind as PinKind)
      })
    : []

// The everywhere pins a draft adds: ticked, and not already going in from the anchor.
export const extraPins = (pins: Pin[], everywhere: Pin[], hasAnchor: boolean): Pin[] =>
  everywhere.filter(one => one.on && !(hasAnchor && pins.some(other => other.on && samePin(one, other))))

// What the store holds under the anchors key, read defensively.
export const anchorPairs = (value: unknown): [string, string][] =>
  Array.isArray(value)
    ? value.filter((one): one is [string, string] => Array.isArray(one) && typeof one[0] === 'string' && typeof one[1] === 'string')
    : []

// This session's published artifacts with one more: a URL already listed takes the new title.
export const withArtifact = (list: MadeArtifact[], one: MadeArtifact): MadeArtifact[] =>
  list.some(old => old.url === one.url) ? list.map(old => (old.url === one.url ? one : old)) : [...list, one]

export const TOUCHED_KEPT = 30

// This chat's touched files with one more on top: a path already listed moves up and stays made if it was.
export const withTouched = (list: TouchedFile[], one: TouchedFile, kept = TOUCHED_KEPT): TouchedFile[] => {
  const was = list.find(old => old.path === one.path)

  return [{ path: one.path, made: one.made || Boolean(was?.made) }, ...list.filter(old => old.path !== one.path)].slice(0, kept)
}

// Whether a touched path is listed: not the Agent Logs, nothing in a dot-folder of the notes folder.
export const isListedTouch = (path: string, notes: string, trails: string): boolean =>
  Boolean(path) && !(trails && path.startsWith(`${trails}/`)) && !(notes && path.startsWith(`${notes}/`) && path.slice(notes.length).includes('/.'))

// A touched file as a pin, unticked (a link, not prompt context): a vault note by name, else the file.
export const touchedPin = (path: string, notes: string): Pin => {
  if (isVaultNote(path, notes)) {
    const name = baseName(path).replace(/\.md$/, '')

    return { on: false, kind: 'note', target: name, label: name, path }
  }

  return { on: false, kind: 'file', target: path, label: baseName(path) }
}

// Pins as the context card lists them: artifacts first, each group in its order.
export const artifactsFirst = (pins: Pin[]): Pin[] => [...pins.filter(pin => pin.kind === 'artifact'), ...pins.filter(pin => pin.kind !== 'artifact')]

// What a submitted draft was about: the lens and the instruction. Null if it isn't ours.
export const chatTitleFrom = (text: string): string | null => {
  if (!text.startsWith(DRAFT_HEAD)) {
    return null
  }
  const lens = text.match(/^- Lens: read "[^"]*\/([^"/]+)\.md"/m)?.[1]
  const said = (text.match(/^Instruction: ([\s\S]*)$/m)?.[1] ?? '').replace(/\s+/g, ' ').trim()
  const title = [lens, said].filter(Boolean).join(': ') || 'chat'

  return title.slice(0, 70)
}

export const stamp = (ms: number): string => {
  const date = new Date(ms)
  const pad = (n: number): string => String(n).padStart(2, '0')

  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`
}

const repoFrom =(cwd: string): string => cwd.split('/').filter(Boolean).pop() ?? 'unknown'

const oneLine = (text: string): string =>
  text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 50)

const firstUserText = (content: unknown): string => {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    const block = content.find(one => one?.type === 'text')

    return typeof block?.text === 'string' ? block.text : ''
  }

  return ''
}

// Reads the start of a Claude Code transcript. Null for what isn't a person's
// coding session: headless runs, and tasks the OpenAugi watcher dispatched.
export const parseClaudeHead = (raw: string): { repo: string; title: string } | null => {
  let cwd = ''
  let prompt = ''
  let title = ''

  for (const line of raw.split('\n')) {
    let row: Record<string, any>
    try {
      row = JSON.parse(line)
    } catch {
      continue
    }
    if (row.type === 'custom-title' && typeof row.customTitle === 'string') {
      title = row.customTitle
    }
    if (row.type === 'user' && !row.isMeta && !row.isSidechain && !prompt) {
      if (row.entrypoint === 'sdk-cli') {
        return null
      }
      const text = firstUserText(row.message?.content)
      // A slash command's own rows are not what the session is about.
      if (text && !text.startsWith('<')) {
        prompt = text
        cwd = typeof row.cwd === 'string' ? row.cwd : ''
      }
    }
  }
  // An Augi chat opens with our own draft, so it isn't a coding session either.
  if (!prompt || prompt.startsWith('Read your skill file first') || prompt.startsWith(DRAFT_HEAD)) {
    return null
  }

  return { repo: repoFrom(cwd), title: oneLine(title || prompt) }
}

// A Codex session_meta line can outrun the bytes read, so this matches text.
// Null for a thread an agent spawned rather than a person.
export const parseCodexHead = (raw: string): { repo: string; title: string } | null => {
  if (!raw.includes('"thread_source":"user"')) {
    return null
  }
  const cwd = raw.match(/"cwd":"([^"]+)"/)?.[1] ?? ''

  return { repo: repoFrom(cwd), title: 'Codex session' }
}

export const ago = (now: number, then: number): string => {
  const minutes = Math.max(0, Math.round((now - then) / 60000))
  if (minutes < 60) {
    return `${minutes}m`
  }
  const hours = Math.round(minutes / 60)

  return hours < 48 ? `${hours}h` : `${Math.round(hours / 24)}d`
}

export type Anchor = { note: string; notePath: string; trail: string }

// `pins` are the anchor's pinned context, a note's `path` filled where it was found;
// `everywhere` the pins that go into every chat; `today` the path of today's note when
// there is one.
export const buildDraft = (
  pins: Pin[],
  choice: Picks,
  lens: Lens | null,
  typed: string,
  anchor: Anchor | null = null,
  everywhere: Pin[] = [],
  today = '',
): string => {
  const lines = [DRAFT_HEAD]

  if (anchor && choice.project) {
    // The note's name in [[ ]] lets a pasted draft anchor a new chat (anchorFromDraft).
    const where = anchor.notePath ? `"${anchor.notePath}"` : '(find it in the vault)'
    lines.push(`- Project [[${anchor.note}]]: ${where}, my note (outcome, Next, Journal), and its Agent Log "${anchor.trail}": the context, questions, chats and notes saved on it so far. Read both.`)
  }
  for (const pin of anchor ? pins.filter(one => one.on) : []) {
    lines.push(pinDraftLine(pin))
  }
  for (const pin of extraPins(pins, everywhere, Boolean(anchor))) {
    lines.push(pinDraftLine(pin))
  }
  if (choice.aboutMe) {
    lines.push(`- About me and my projects: read "${P.context}" and "${P.board}".`)
  }
  if (today && choice.today) {
    lines.push(`- Today: read "${today}", my note for today: what I set out to do and what I have logged so far.`)
  }
  if (choice.openAugi) {
    lines.push(`- Search my notes in "${P.notes}" for what is relevant to this (with the OpenAugi tools if you have them), and name the notes you used.`)
  }
  if (lens) {
    lines.push(`- Lens: read "${lens.path}" and apply it to the context above. Answer here in chat instead of writing its output file.`)
  }
  lines.push('Reply in at most 10 lines: what you found or did, then anything you need from me. No walls of text.')
  lines.push("Don't change any files unless my instruction asks you to.")
  if (anchor) {
    lines.push(`Write only to the Agent Log, never to my note. If you write a note for this, add "- [[<its name>]]" under ## Saved there; if you find something worth keeping for this project, add "- [ ] [[<note>]] · <why it matters>" under ## Context.`)
  }
  lines.push(`Instruction: ${typed}`)

  return lines.join('\n')
}

// The anchor a draft names on its `- Project [[<note>]]:` line, exactly as written;
// null for text that isn't ours or names no anchor.
export const anchorFromDraft = (text: string): string | null => {
  if (!text.startsWith(DRAFT_HEAD)) {
    return null
  }
  const note = text.match(/^- Project \[\[(.+?)\]\]: /m)?.[1]

  return note?.trim() ? note : null
}

export const typedFrom = (box: string): string => {
  if (!box.startsWith(DRAFT_HEAD)) {
    return box
  }

  return box.match(/^Instruction: ([\s\S]*)$/m)?.[1] ?? ''
}

export const toggled = (list: string[], item: string): string[] =>
  list.includes(item) ? list.filter(one => one !== item) : [...list, item]

// Codex multi-agent runs write several rollouts in the same second, all marked as
// the person's. Keep the newest of each, named by the timestamp in its file name.
export const onePerCodexRun = (sessions: Session[]): Session[] => {
  const seen = new Set<string>()

  return sessions.filter(one => {
    const run = one.tool === 'codex' ? one.path.match(/rollout-([\dT-]+?)-[0-9a-f]{8}-/)?.[1] : undefined
    if (!run) {
      return true
    }
    if (seen.has(run)) {
      return false
    }
    seen.add(run)

    return true
  })
}

async function headOf($: EngineInterface, path: string): Promise<string> {
  const { stdout } = await $.process.run(['head', '-c', HEAD_BYTES, path])

  return stdout
}

async function claudeCandidates($: EngineInterface): Promise<Session[]> {
  const found: Session[] = []

  if (!(await $.fs.exists(P.claudeDir))) {
    return found
  }
  for (const dir of await $.fs.list(P.claudeDir)) {
    if (dir.kind === 'dir' && !dir.name.startsWith('-private-tmp')) {
      const files = await $.fs.list(`${P.claudeDir}/${dir.name}`)
      for (const file of files) {
        if (file.name.endsWith('.jsonl')) {
          found.push({
            path: `${P.claudeDir}/${dir.name}/${file.name}`,
            tool: 'claude',
            repo: '',
            title: '',
            mtimeMs: file.mtimeMs,
          })
        }
      }
    }
  }

  return found
}

// Codex keeps sessions under year/month/day, so the newest two days cover "recent".
async function codexCandidates($: EngineInterface): Promise<Session[]> {
  const found: Session[] = []
  if (!(await $.fs.exists(P.codexDir))) {
    return found
  }
  const days: string[] = []

  for (const year of await $.fs.list(P.codexDir)) {
    for (const month of await $.fs.list(`${P.codexDir}/${year.name}`)) {
      for (const day of await $.fs.list(`${P.codexDir}/${year.name}/${month.name}`)) {
        days.push(`${P.codexDir}/${year.name}/${month.name}/${day.name}`)
      }
    }
  }
  for (const day of days.sort().slice(-2)) {
    for (const file of await $.fs.list(day)) {
      if (file.name.endsWith('.jsonl')) {
        found.push({ path: `${day}/${file.name}`, tool: 'codex', repo: '', title: '', mtimeMs: file.mtimeMs })
      }
    }
  }

  return found
}

async function recentSessions($: EngineInterface): Promise<Session[]> {
  const own = await $.session.id()
  const all = onePerCodexRun(
    [...(await claudeCandidates($)), ...(await codexCandidates($))]
      .filter(one => !own || !one.path.includes(own))
      .sort((a, b) => b.mtimeMs - a.mtimeMs),
  ).slice(0, 400)
  const kept: Session[] = []

  for (const one of all) {
    if (kept.length === SESSION_KEYS.length) {
      break
    }
    const raw = await headOf($, one.path)
    const parsed = one.tool === 'claude' ? parseClaudeHead(raw) : parseCodexHead(raw)
    if (parsed) {
      kept.push({ ...one, ...parsed })
    }
  }

  return kept
}


async function notePathOf($: EngineInterface, note: string): Promise<string> {
  const { stdout } = await $.process.run([
    'find', P.notes, '-name', `${note}.md`, '-not', '-path', '*/.git/*', '-not', '-path', '*/.obsidian/*', '-not', '-path', '*/.claude/*',
  ])

  return stdout.split('\n').find(Boolean) ?? ''
}

async function readOr($: EngineInterface, path: string): Promise<string> {
  return (await $.fs.exists(path)) ? $.fs.read(path) : ''
}

// Renames a v0.6 trail note to its Agent Log name, once, tidied (its title line goes).
async function moveOldTrail($: EngineInterface, note: string): Promise<void> {
  const from = oldTrailPath(note)
  const to = trailPath(note)
  if ((await $.fs.exists(from)) && !(await $.fs.exists(to))) {
    const text = tidyTrail(await $.fs.read(from), note)
    await $.fs.write(to, text)
    await $.process.run(['rm', from])
  }
}

// Reads the anchored card's note and Agent Log; the rest of the pane stays as it was.
async function loadAnchor($: EngineInterface, cards: Card[]): Promise<Pick<Swatch, 'anchor' | 'notePath' | 'outcome' | 'next' | 'jira' | 'trail' | 'chatFiles' | 'chatSeen' | 'seenAt' | 'pinPaths' | 'hits'>> {
  const chosen = await sessionAnchor($)
  let anchor = chosen ? anchorCard(cards, chosen) : null
  // A note not on the board (a pasted draft named it, or it left the board) still
  // anchors, bare, while a note of that name is in the notes folder.
  const bare = chosen && !anchor ? await notePathOf($, chosen) : ''
  if (chosen && bare) {
    anchor = { column: '', note: chosen, detail: '', tasks: [] }
  }
  if (!anchor) {
    return { anchor: null, notePath: '', outcome: '', next: [], jira: '', trail: EMPTY_TRAIL, chatFiles: {}, chatSeen: {}, seenAt: 0, pinPaths: {}, hits: [] }
  }
  const notePath = bare || (await notePathOf($, anchor.note))
  const noteText = notePath ? await $.fs.read(notePath) : ''
  await moveOldTrail($, anchor.note)
  const trail = parseTrail(await readOr($, trailPath(anchor.note)))
  const pinPaths: Record<string, string> = {}
  for (const pin of trail.pins.filter(one => one.kind === 'note')) {
    pinPaths[pin.target] = await notePathOf($, pin.target)
  }

  const chatFiles = await transcriptsOf($, trail.chats.filter(one => !one.done).map(one => one.id))

  return {
    anchor,
    notePath,
    outcome: outcomeFrom(noteText),
    next: nextFrom(noteText),
    jira: jiraFrom(noteText),
    trail,
    chatFiles,
    chatSeen: await seenOf($, chatFiles),
    seenAt: await $.clock.now(),
    pinPaths,
    hits: [],
  }
}

// How many open chats the pane looks up transcripts for.
const CHATS_FOLLOWED = 12

// Where the transcripts of these chats are, by chat id: one search of the Claude projects
// folder for the newest few. A chat whose transcript is not on this computer has none.
async function transcriptsOf($: EngineInterface, ids: string[]): Promise<Record<string, string>> {
  const wanted = ids.slice(0, CHATS_FOLLOWED).filter(id => /^[\w-]+$/.test(id))
  const files: Record<string, string> = {}
  try {
    if (wanted.length === 0 || !(await $.fs.exists(P.claudeDir))) {
      return files
    }
    const names = wanted.flatMap((id, at) => [...(at > 0 ? ['-o'] : []), '-name', `${id}.jsonl`])
    const { stdout } = await $.process.run(['find', P.claudeDir, '-maxdepth', '2', '(', ...names, ')'])
    for (const path of stdout.split('\n').filter(Boolean)) {
      const id = baseName(path).replace(/\.jsonl$/, '')
      if (wanted.includes(id) && path.startsWith(`${P.claudeDir}/`)) {
        files[id] = path
      }
    }
  } catch {
    // No transcripts found: the chats list without a last-active time.
  }

  return files
}

// When each of those transcripts was last written, by chat id.
async function seenOf($: EngineInterface, files: Record<string, string>): Promise<Record<string, number>> {
  const seen: Record<string, number> = {}
  for (const [id, path] of Object.entries(files)) {
    try {
      seen[id] = (await $.fs.stat(path)).mtimeMs
    } catch {
      // Gone since: no time shown.
    }
  }

  return seen
}

// Follows the open chats' transcripts between full reads: a stat each, and a redraw only
// when one of them was written since.
async function resyncChats($: EngineInterface): Promise<void> {
  const { chatFiles, chatSeen } = await read($, swatch)
  if (Object.keys(chatFiles ?? {}).length === 0) {
    return
  }
  const seen = await seenOf($, chatFiles)
  if (Object.keys(seen).some(id => seen[id] !== chatSeen[id])) {
    const seenAt = await $.clock.now()
    await update($, swatch, old => ({ ...old, chatSeen: seen, seenAt }))
  }
}

// Today's note as the pane shows it: where it is, whether it is there, what is still open in it.
async function todayNow($: EngineInterface, pattern: string): Promise<Today> {
  const now = await $.clock.now()
  const path = dailyNotePath(pattern, now, P.notes, P.home)
  if (!path) {
    return NO_TODAY
  }
  const day = stamp(now).slice(0, 10)
  if (!(await $.fs.exists(path))) {
    return { pattern, path, day, exists: false, todo: [], mtime: 0 }
  }
  let mtime = 0
  try {
    mtime = (await $.fs.stat(path)).mtimeMs
  } catch {
    mtime = 0
  }

  return { pattern, path, day, exists: true, todo: openBoxes(await $.fs.read(path)), mtime }
}

// Reads today's note again when the day rolled over, the note appeared, or it changed
// (ticked in the editor, say). One stat when nothing did.
async function resyncToday($: EngineInterface): Promise<void> {
  const { today } = await read($, swatch)
  if (!today.pattern) {
    return
  }
  const path = dailyNotePath(today.pattern, await $.clock.now(), P.notes, P.home)
  let mtime = 0
  try {
    mtime = (await $.fs.exists(path)) ? (await $.fs.stat(path)).mtimeMs : 0
  } catch {
    mtime = 0
  }
  if (path === today.path && mtime === today.mtime && today.exists === mtime > 0) {
    return
  }
  const next = await todayNow($, today.pattern)
  await update($, swatch, old => ({ ...old, today: next }))
}

// A line of the user's own, on their press, added to today's note (read fresh: it may be
// open in their editor). False when there is no note for today or it could not be written.
async function writeToday($: EngineInterface, entries: PadEntry[]): Promise<boolean> {
  const { today } = await read($, swatch)
  if (!today.path || entries.length === 0) {
    return false
  }
  try {
    if (!(await $.fs.exists(today.path))) {
      await $.ui.toast("No note for today yet: make it in your editor first, and it shows here.")

      return false
    }
    const old = await $.fs.read(today.path)
    await $.fs.write(today.path, [...entries].reverse().reduce((text, one) => addDailyLine(text, P.dailyHeading, one), old))
  } catch (error) {
    await $.ui.toast(`Couldn't write today's note: ${String(error)}`)

    return false
  }
  const next = await todayNow($, today.pattern)
  await update($, swatch, old => ({ ...old, today: next }))

  return true
}

// What was typed in the Today card's log field goes into today's note, stamped now.
async function logToday($: EngineInterface, text: string): Promise<void> {
  if (!text.trim()) {
    return
  }
  if (await writeToday($, [{ when: stamp(await $.clock.now()), text: text.trim() }])) {
    await update($, view, old => ({ ...old, logged: (old.logged ?? 0) + 1 }))
    await $.ui.toast("Logged to today's note")
  }
}

// Scratchpad entries go into today's note, oldest first, and come off the pad.
async function padToToday($: EngineInterface, entries: PadEntry[]): Promise<void> {
  if (await writeToday($, entries)) {
    await unjot($, entries)
    await $.ui.toast("Saved to today's note")
  }
}

// The settings as the module was loaded with them; a change reloads the module.
let settings: Partial<Options> = {}

async function ready($: EngineInterface): Promise<void> {
  configure(resolvePaths(settings, (await $.env.get('HOME')) ?? ''))
}

async function setupNow($: EngineInterface): Promise<SetupItem[]> {
  const found = new Set<string>()
  for (const path of [P.notes, P.context, P.board, P.lenses]) {
    if (path && (await $.fs.exists(path))) {
      found.add(path)
    }
  }

  return setupItems(P, found)
}

// Obsidian's vaults that still exist on disk.
async function obsidianVaults($: EngineInterface): Promise<string[]> {
  for (const path of obsidianConfigs(P.home, (await $.env.get('APPDATA')) ?? '')) {
    if (await $.fs.exists(path)) {
      const vaults: string[] = []
      for (const vault of vaultsFrom(await $.fs.read(path))) {
        if (await $.fs.exists(vault)) {
          vaults.push(vault)
        }
      }

      return vaults
    }
  }

  return []
}

// Runs a search over the notes folder, newest first.
async function newestFirst($: EngineInterface, argv: string[]): Promise<string> {
  const hits = (await $.process.run(argv)).stdout.split('\n').filter(Boolean).slice(0, 40)
  if (hits.length === 0) {
    return ''
  }

  return (await $.process.run(['ls', '-td', '--', ...hits])).stdout
}

// Notes in the folder that look like what a setup step wants.
async function lookFor($: EngineInterface, field: string): Promise<string[]> {
  const notes = P.notes
  const skipHidden = ['-path', '*/.*', '-prune', '-o']
  if (field === 'boardNote') {
    return candidatesFrom(await newestFirst($, ['grep', '-rl', '-m1', '--include=*.md', '--exclude-dir=.*', '--exclude-dir=node_modules', 'kanban-plugin', notes]), notes)
  }
  if (field === 'contextNote') {
    return candidatesFrom(await newestFirst($, ['find', notes, ...skipHidden, '-type', 'f', '(', '-iname', '*context.md', '-o', '-iname', '*about me*.md', '-o', '-iname', '*profile.md', ')', '-print']), notes)
  }
  if (field === 'lensFolder') {
    return candidatesFrom(await newestFirst($, ['find', notes, ...skipHidden, '-type', 'd', '(', '-iname', '*lens*', '-o', '-iname', '*prompt*', ')', '-print']), notes)
  }

  return []
}

async function skippedNow($: EngineInterface): Promise<string[]> {
  const value = await $.store.get(SKIPPED_KEY)

  return Array.isArray(value) ? value.filter((one): one is string => typeof one === 'string') : []
}

async function load($: EngineInterface): Promise<void> {
  const setup = await setupNow($)
  if (!setup[0]?.ok) {
    const vaults = await obsidianVaults($)
    await update($, swatch, old => ({ ...old, setup, vaults, loadedAt: 1 }))

    return
  }
  const skipped = await skippedNow($)
  const step = STEPS.find(one => !skipped.includes(one.field) && !setup.find(item => item.field === one.field)?.ok)
  if (step) {
    const found = { [step.field]: await lookFor($, step.field) }
    await update($, swatch, old => ({ ...old, setup, skipped, found, loadedAt: 1 }))

    return
  }
  await inheritAnchor($)
  const cards = kanbanCards(await readOr($, P.board))
  const next: Swatch = {
    setup,
    vaults: [],
    found: {},
    skipped,
    season: anchorFrom(await readOr($, P.context)),
    cards,
    ...(await loadAnchor($, cards)),
    jiraServer: await jiraServerNow($),
    taskTemplate: taskTemplateState(await readOr($, P.taskTemplate)),
    today: await todayNow($, dailyPattern(P.daily, P.daily ? '' : await readOr($, P.obsidianDaily))),
    everywhere: storedPins(await $.store.get(PINS_KEY)),
    lenses: [],
    quick: { overview: '', standup: '' },
    sessions: await recentSessions($),
    loadedAt: await $.clock.now(),
  }

  const texts = new Map<string, string>()
  for (const file of (await $.fs.exists(P.lenses)) ? await $.fs.list(P.lenses) : []) {
    if (file.name.endsWith('.md')) {
      texts.set(file.name.slice(0, -3), await $.fs.read(`${P.lenses}/${file.name}`))
    }
  }
  for (const name of lensNames([...texts.keys()])) {
    next.lenses.push({ name, path: `${P.lenses}/${name}.md`, description: descriptionFrom(texts.get(name) ?? '') })
  }
  // Two lenses get a button of their own where they are used, when the lens folder has
  // them, whatever the lens picks list: the wording stays a note the user can edit.
  next.quick = {
    overview: texts.has('overview') ? `${P.lenses}/overview.md` : '',
    standup: texts.has('standup') ? `${P.lenses}/standup.md` : '',
  }

  await update($, swatch, () => next)
}

// The Jira MCP server this session has, read from its tool list (a plain read: nothing
// connects, signs in or asks). `$.mcp.connect` can't tell: it connects only servers
// this plugin's own manifest lists.
async function jiraServerNow($: EngineInterface): Promise<string> {
  try {
    return jiraServerFrom(await $.tool.list(), P.jiraServer)
  } catch {
    return ''
  }
}

// What the Jira drafts name, or null when Jira is off: on with a server found or named,
// a project or site set, or a key on the anchor.
async function jiraAsk($: EngineInterface): Promise<JiraAsk | null> {
  const { jiraServer, jira } = await read($, swatch)
  const server = jiraServer || P.jiraServer

  return server || P.jiraProject || P.jiraSite || jira ? { project: P.jiraProject, server } : null
}

// Sets or clears `jira:` in the anchor note's frontmatter, read fresh from disk (it may
// be open in the user's editor). Like the scratchpad's save, this is the user's own press
// changing one line of their own note; nothing else in it changes, and Jira is not called.
async function writeJira($: EngineInterface, key: string): Promise<void> {
  const { anchor, notePath } = await read($, swatch)
  if (!anchor) {
    return
  }
  if (!notePath) {
    await $.ui.toast(`No note found for ${anchor.note}: nothing linked.`)

    return
  }
  try {
    const old = await $.fs.read(notePath)
    const text = withJira(old, key)
    if (text !== old) {
      await $.fs.write(notePath, text)
    }
  } catch (error) {
    await $.ui.toast(`Couldn't write ${anchor.note}: ${String(error)}`)

    return
  }
  await update($, swatch, old => (old.anchor?.note === anchor.note ? { ...old, jira: key } : old))
}

// A pasted key or issue link links the anchor. A site in the link is not saved as a setting.
async function linkJira($: EngineInterface, typed: string): Promise<void> {
  const key = jiraKeyFrom(typed)
  if (!key) {
    await $.ui.toast(`Not a Jira key or issue link: ${typed.trim()}`)

    return
  }
  await writeJira($, key)
}

// Drafts (unsent) a search for the anchor's issue; the chat uses its own Jira tools.
async function findInJira($: EngineInterface): Promise<void> {
  const { anchor, notePath } = await read($, swatch)
  if (anchor) {
    await fillOver($, findInJiraDraft(notePath, anchor.note, (await jiraAsk($)) ?? { project: P.jiraProject, server: P.jiraServer }))
  }
}

// Drafts (unsent) a status update for the linked issue: a jira-update lens in the lens
// folder when there is one, so the user can make the wording their own, else the built-in.
async function updateJira($: EngineInterface): Promise<void> {
  const { anchor, notePath, jira } = await read($, swatch)
  if (!anchor || !jira) {
    return
  }
  const lensPath = `${P.lenses}/jira-update.md`
  if (await $.fs.exists(lensPath)) {
    await draft($, { name: 'jira-update', path: lensPath, description: '' }, `update Jira ${jira}`)

    return
  }
  const ask = (await jiraAsk($)) ?? { project: P.jiraProject, server: P.jiraServer }
  await fillOver($, jiraUpdateDraft(notePath, anchor.note, trailPath(anchor.note), jira, ask))
}

// The note this chat session is anchored on, if any.
async function sessionAnchor($: EngineInterface): Promise<string | null> {
  const id = await $.session.id()

  return anchorPairs(await $.store.get(ANCHORS_KEY)).find(([one]) => one === id)?.[1] || null
}

async function setSessionAnchor($: EngineInterface, note: string | null): Promise<void> {
  const id = await $.session.id()
  if (id) {
    await $.store.set(ANCHORS_KEY, withAnchor(anchorPairs(await $.store.get(ANCHORS_KEY)), id, note))
  }
}

// The session's working folder as the key its anchor is remembered under, or empty.
async function folderNow($: EngineInterface): Promise<string> {
  try {
    return folderKey(await $.session.cwd())
  } catch {
    return ''
  }
}

// A chosen anchor becomes what the next new chat in this folder starts on.
async function rememberFolder($: EngineInterface, note: string): Promise<void> {
  const folder = await folderNow($)
  if (folder) {
    await $.store.set(FOLDERS_KEY, withAnchor(anchorPairs(await $.store.get(FOLDERS_KEY)), folder, note, FOLDERS_KEPT))
  }
}

// A new chat that has not chosen starts on the anchor last chosen in its working folder,
// so starting a chat does not begin with picking one. Only a session that just began: an
// older one that never had an anchor keeps none. Nothing is written to the anchor's log
// until the chat says something (nameChat), so an abandoned chat leaves no trace there.
// True when it anchored the session.
async function inheritAnchor($: EngineInterface): Promise<boolean> {
  try {
    const id = await $.session.id()
    const pairs = anchorPairs(await $.store.get(ANCHORS_KEY))
    if (!id || pairs.some(([one]) => one === id)) {
      return false
    }
    const folder = await folderNow($)
    const note = folder ? anchorPairs(await $.store.get(FOLDERS_KEY)).find(([one]) => one === folder)?.[1] : ''
    if (!note || (await $.clock.now()) - (await $.session.usage()).startedAt > NEW_SESSION_MS) {
      return false
    }
    await $.store.set(ANCHORS_KEY, withAnchor(pairs, id, note))
    await update($, inherited, () => ({ session: id, note, named: '' }))

    return true
  } catch {
    return false
  }
}

// Whether this chat is still on the anchor it started on by itself, unconfirmed.
async function isInherited($: EngineInterface, note: string): Promise<boolean> {
  const one = await read($, inherited)

  return one.note === note && one.session === (await $.session.id())
}

// The user is fine with the anchor the chat started on: the note about it goes.
async function keepInherited($: EngineInterface): Promise<void> {
  await update($, inherited, old => ({ ...old, session: '', note: '' }))
}

// Anchors this session on `note`: the pane (when it has loaded) shows it, and what the
// chat already published is pinned to it, ticked. The caller records the chat.
// When the board, the anchor's note and its Agent Log were last changed, as one string.
async function anchorStamp($: EngineInterface, note: string, notePath: string): Promise<string> {
  const stamps: string[] = []
  for (const path of [P.board, trailPath(note), notePath]) {
    let at = 0
    try {
      at = path && (await $.fs.exists(path)) ? (await $.fs.stat(path)).mtimeMs : 0
    } catch {
      at = 0
    }
    stamps.push(String(at))
  }

  return stamps.join('|')
}

// What the pane last saw of those files. Another chat on the same anchor writes the same
// Agent Log; this is how this chat's pane notices and reads it again.
let seenStamp = ''
let anchorWatch: { cancel: () => void } | null = null

// Reads the board and the anchor again when one of its files changed since the pane
// last looked. Three stats when nothing changed.
export async function resync($: EngineInterface): Promise<void> {
  const { anchor, notePath, loadedAt } = await read($, swatch)
  if (loadedAt <= 1) {
    return
  }
  await resyncToday($)
  if (!anchor) {
    // A chat cleared or resumed in this process is a new session: it starts on the folder's anchor too.
    if (await inheritAnchor($)) {
      const { cards } = await read($, swatch)
      const found = await loadAnchor($, cards)
      await update($, swatch, old => ({ ...old, ...found }))
    }

    return
  }
  const stamp = await anchorStamp($, anchor.note, notePath)
  if (stamp === seenStamp) {
    await resyncChats($)

    return
  }
  seenStamp = stamp
  const cards = kanbanCards(await readOr($, P.board))
  const found = await loadAnchor($, cards)
  await update($, swatch, old => (old.anchor?.note === anchor.note ? { ...old, cards, ...found, hits: old.hits } : old))
}

// Checks the anchor's files every few seconds while the pane is open in this process.
function watchAnchor($: EngineInterface): void {
  try {
    anchorWatch?.cancel()
    anchorWatch = $.clock.every(SYNC_MS, () => {
      resync($).catch(() => undefined)
    })
  } catch {
    anchorWatch = null
  }
}

// Opens the pane beside a new chat, once per session, unless that is turned off. Never
// in the way of the session: a failure here only leaves the pane closed for /augi.
async function openOnce($: EngineInterface): Promise<void> {
  try {
    if (settings.openOnStart === 'no') {
      return
    }
    const id = (await $.session.id()) ?? ''
    if ((await read($, opened)).session === id) {
      return
    }
    await update($, opened, () => ({ session: id }))
    await load($)
    await $.ui.open({ id: PANE, title: 'Augi' })
    watchAnchor($)
  } catch {
    // The pane stays closed; /augi opens it.
  }
}

async function anchorSession($: EngineInterface, note: string): Promise<void> {
  await setSessionAnchor($, note)
  await rememberFolder($, note)
  await keepInherited($)
  const { cards, loadedAt } = await read($, swatch)
  if (loadedAt > 0) {
    const found = await loadAnchor($, cards)
    await update($, swatch, old => ({ ...old, ...found }))
  }
  for (const one of await sessionArtifacts($)) {
    await writeTrail($, note, text => addPin(text, artifactPin(one)))
  }
}

// Anchoring puts this chat in the anchor's ## Chats and pins what it already published, ticked.
async function chooseAnchor($: EngineInterface, card: Card): Promise<void> {
  await anchorSession($, card.note)
  await recordChat($, CHAT_PLACEHOLDER)
  await update($, view, old => ({ ...old, anchorsOpen: false }))
}

// A draft pasted into a chat with no anchor (copied for a new chat) anchors it on the
// draft's anchor. A chat that chose one keeps it, whatever the draft says; one the chat
// only started on by itself gives way to the draft.
async function anchorFromPaste($: EngineInterface, text: string): Promise<void> {
  const note = anchorFromDraft(text)
  const mine = await sessionAnchor($)
  if (note && note !== mine && (!mine || (await isInherited($, mine)))) {
    await anchorSession($, note)
  }
}

// No anchor, by choice: remembered for the session, so none is carried over to it again.
async function clearAnchor($: EngineInterface): Promise<void> {
  await setSessionAnchor($, '')
  await keepInherited($)
  await update($, swatch, old => ({ ...old, anchor: null, notePath: '', outcome: '', next: [], jira: '', trail: EMPTY_TRAIL, chatFiles: {}, chatSeen: {}, seenAt: 0, pinPaths: {}, hits: [] }))
  await update($, view, old => ({ ...old, anchorsOpen: false }))
}

async function anchorNow($: EngineInterface): Promise<Anchor | null> {
  const { anchor, notePath } = await read($, swatch)

  return anchor ? { note: anchor.note, notePath, trail: trailPath(anchor.note) } : null
}

// `about` replaces what's typed in the prompt box: a captured question, for one.
async function draftText($: EngineInterface, lens: Lens | null, about?: string): Promise<string> {
  const { trail, pinPaths, everywhere, today } = await read($, swatch)
  const choice = await read($, picks)
  const typed = about ?? typedFrom((await $.prompt.read()).text)

  return buildDraft(withPaths(trail.pins ?? [], pinPaths), choice, lens, typed, await anchorNow($), everywhere ?? [], today.exists ? today.path : '')
}

// Fills the prompt box with a pane draft. A prompt the user was typing is not lost:
// it goes to the clipboard first.
async function fillOver($: EngineInterface, text: string): Promise<void> {
  let had = ''
  try {
    had = (await $.prompt.read()).text ?? ''
  } catch {
    had = ''
  }
  if (had.trim() && !had.startsWith(DRAFT_HEAD)) {
    await $.ui.copy({ text: had })
    await $.ui.toast('Your unsent prompt was copied to the clipboard.')
  }
  await $.prompt.fill({ text, mode: 'replace' })
}

async function draft($: EngineInterface, lens: Lens | null, about?: string): Promise<void> {
  await $.prompt.fill({ text: await draftText($, lens, about), mode: 'replace' })
}

// The same draft, copied to start a new chat with; the prompt box is left alone.
async function copyDraft($: EngineInterface): Promise<void> {
  const text = await draftText($, null)
  const copied = await $.ui.copy({ text })
  const anchors = anchorFromDraft(text) ? 'Sent there, it anchors that chat here too.' : 'It names no anchor (the project notes are unticked or none is picked), so that chat stays unanchored.'
  await $.ui.toast(copied.isCopied ? `Copied: paste it into a new chat. ${anchors}` : `Couldn't copy the draft (${copied.reason}).`)
}

const withPaths = (pins: Pin[], pinPaths: Record<string, string>): Pin[] =>
  pins.map(pin => (pin.kind === 'note' && pinPaths[pin.target] ? { ...pin, path: pinPaths[pin.target] } : pin))

// Pins something on a trail. A note keeps where it was found, for drafts and links.
async function pinIt($: EngineInterface, note: string, pin: Pin): Promise<void> {
  const { path, ...kept } = pin
  if (pin.kind === 'note') {
    const found = path || (await notePathOf($, pin.target))
    await update($, swatch, old => ({ ...old, pinPaths: { ...old.pinPaths, [pin.target]: found } }))
  }
  await writeTrail($, note, text => addPin(text, kept))
  await update($, swatch, old => ({ ...old, hits: old.hits.filter(one => !(one.kind === pin.kind && one.target === pin.target)) }))
}

// Rewrites the pins that go into every chat, in the store and in the pane.
async function setEverywhere($: EngineInterface, change: (pins: Pin[]) => Pin[]): Promise<void> {
  const pins = change(storedPins(await $.store.get(PINS_KEY))).slice(-PINS_KEPT)
  await $.store.set(PINS_KEY, pins)
  await update($, swatch, old => ({ ...old, everywhere: pins }))
}

// Pins something for every chat. A note keeps where it was found, as on a trail.
async function pinEverywhere($: EngineInterface, pin: Pin): Promise<void> {
  const found = pin.kind === 'note' ? pin.path || (await notePathOf($, pin.target)) : ''
  const kept: Pin = found ? { ...pin, path: found } : pin
  await setEverywhere($, pins => (pins.some(one => samePin(one, pin)) ? pins : [...pins, kept]))
  await update($, swatch, old => ({ ...old, hits: old.hits.filter(one => !samePin(one, pin)) }))
}

// A pasted link, [[note]] or path, pinned to the anchor or for every chat.
async function pinTyped($: EngineInterface, value: string, where: 'anchor' | 'everywhere' = 'anchor'): Promise<void> {
  const { anchor } = await read($, swatch)
  const pin = pinFrom(value, P.notes, P.home)
  if (!pin || (where === 'anchor' && !anchor)) {
    return
  }
  if ((pin.kind === 'file' || pin.kind === 'session') && !(await $.fs.exists(pin.target))) {
    await $.ui.toast(`Not found: ${pin.target}`)
    return
  }
  if (where === 'everywhere' || !anchor) {
    await pinEverywhere($, pin)
    return
  }
  await pinIt($, anchor.note, pin)
}

// OpenAugi's semantic search first; a file-name match when it finds nothing or isn't there.
async function searchNotes($: EngineInterface, query: string): Promise<void> {
  const asked = query.trim()
  if (!asked) {
    await update($, swatch, old => ({ ...old, hits: [] }))
    return
  }
  let paths: string[] = []
  try {
    const answer = await $.mcp.call('openaugi', 'search', { query: asked, k: 20 })
    const text = (answer.content ?? []).map(block => (block.type === 'text' ? block.text ?? '' : '')).join('')
    paths = pathsFromSearch(text).map(rel => `${P.notes}/${rel}`)
  } catch {
    paths = []
  }
  if (paths.length === 0) {
    const pattern = `*${asked.replace(/[*?[\]]/g, '')}*.md`
    const { stdout } = await $.process.run(['find', P.notes, '-path', '*/.*', '-prune', '-o', '-type', 'f', '-iname', pattern, '-print'])
    paths = stdout.split('\n').filter(Boolean)
  }
  const { trail, everywhere } = await read($, swatch)
  const hits: Pin[] = []
  for (const path of paths) {
    const name = baseName(path).replace(/\.md$/, '')
    const isPinned = [...(trail.pins ?? []), ...(everywhere ?? [])].some(one => one.kind === 'note' && one.target === name)
    if (!isPinned && !hits.some(one => one.target === name)) {
      hits.push({ on: true, kind: 'note', target: name, label: name, path })
    }
  }
  await update($, swatch, old => ({ ...old, hits: hits.slice(0, 5) }))
  if (hits.length === 0) {
    await $.ui.toast(`No notes found for "${asked}".`)
  }
}

async function copyText($: EngineInterface, text: string, said = 'Copied'): Promise<void> {
  await $.ui.copy({ text })
  await $.ui.toast(`${said}: ${text}`)
}

async function openPin($: EngineInterface, pin: Pin): Promise<void> {
  if (pin.kind === 'note') {
    return pin.path ? openNote($, pin.path) : openByName($, pin.target)
  }
  if (pin.kind === 'session') {
    return copyText($, resumeOf(pin.target), 'A session reopens in a terminal. Copied')
  }
  if (pin.kind === 'file') {
    return openPath($, pin.target)
  }
  await launch($, ['open', pin.target])
}

// Opens a file the way its kind wants (a vault note in the note editor, the rest in the
// file editor); one that is gone says so and opens nothing.
async function openPath($: EngineInterface, path: string): Promise<void> {
  if (!(await $.fs.exists(path))) {
    await $.ui.toast(`Not found: ${path}`)

    return
  }
  await launch($, openArgv(P, path))
}

// Whether a file is there before a write or edit; unsure reads as there (nothing marked new).
async function existsBefore($: EngineInterface, path: string): Promise<boolean> {
  try {
    return await $.fs.exists(path)
  } catch {
    return true
  }
}

// After a write or edit that ran (not denied, no error), the file goes on this chat's list.
async function afterTouch($: EngineInterface, path: string, existed: boolean, ran: ToolCallResult): Promise<void> {
  try {
    if (ran.deny === undefined && !ran.isError) {
      await rememberTouch($, path, !existed)
    }
  } catch {
    // the list is a convenience; the call's result stands
  }
}

// The files this chat touched; a list from an earlier session reads as none.
async function sessionTouched($: EngineInterface): Promise<TouchedFile[]> {
  const { session, files } = await read($, touched)

  return session && session === (await $.session.id()) ? files : []
}

// A file a write or edit of this chat's changed: listed unless it is the pane's own or hidden.
async function rememberTouch($: EngineInterface, path: string, isMade: boolean): Promise<void> {
  if (!isListedTouch(path, P.notes, P.trails)) {
    return
  }
  const id = await $.session.id()
  await update($, touched, old => ({ session: id, files: withTouched(old.session === id ? old.files : [], { path, made: isMade }) }))
}

// The anchor whose Agent Log holds this session's chat: anchoring records it, so any anchor.
async function paneChatAnchor($: EngineInterface): Promise<string | null> {
  return sessionAnchor($)
}

// The artifacts this chat session published; a list from an earlier session reads as none.
async function sessionArtifacts($: EngineInterface): Promise<MadeArtifact[]> {
  const { session, artifacts } = await read($, made)

  return session && session === (await $.session.id()) ? artifacts : []
}

async function rememberArtifact($: EngineInterface, one: MadeArtifact): Promise<void> {
  const id = await $.session.id()
  await update($, made, old => ({ session: id, artifacts: withArtifact(old.session === id ? old.artifacts : [], one) }))
}

// An artifact the chat published, pinned ticked: it sticks to the anchor as context.
const artifactPin = (one: MadeArtifact): Pin => ({ on: true, kind: 'artifact', target: one.url, label: one.title })

export const ARTIFACT_URL = /https:\/\/claude\.ai\/(?:code\/)?artifact\/[\w-]+/

// Writes the lens file (never over one that's there), opens it to paste the prompt, lists it.
async function newLens($: EngineInterface, typed: string): Promise<void> {
  const name = lensFileName(typed)
  if (!name) {
    return
  }
  const path = `${P.lenses}/${name}.md`

  await writeNew($, path, newLensSkeleton(name))
  await openNote($, path)
  await load($)
}

// Tidies an older log as it's written: only logs the pane touches change.
async function writeTrail($: EngineInterface, note: string, change: (text: string) => string): Promise<void> {
  const path = trailPath(note)
  const old = await readOr($, path)
  const base = old ? tidyTrail(old, note) : trailTemplate(note, stamp(await $.clock.now()).slice(0, 10))
  const text = change(base)

  if (text !== old) {
    // Only a first write needs the folder: nothing sits between an existing log's read and its write.
    if (!old) {
      await $.process.run(['mkdir', '-p', P.trails])
    }
    await $.fs.write(path, text)
  }
  await update($, swatch, current => (current.anchor?.note === note ? { ...current, trail: parseTrail(text) } : current))
}

// Marks a chat on this anchor finished or open again.
async function chatDone($: EngineInterface, note: string, id: string, done: boolean): Promise<void> {
  await writeTrail($, note, text => setChatDone(text, id, done))
}

// Puts this chat in its anchor's Agent Log, or names it there if it went in as the
// placeholder. Nothing happens with no anchor.
async function recordChat($: EngineInterface, title: string): Promise<void> {
  const note = await sessionAnchor($)
  const id = await $.session.id()
  if (!note || !id) {
    return
  }
  const when = stamp(await $.clock.now())

  await writeTrail($, note, text => retitleChat(addChat(text, { id, when, title, done: false }), id, title))
}

// Once an anchored chat says something it is in its anchor's log under a real title: a
// pane draft's, else the prompt's first line. The log is read for this once per session
// and anchor, not on every prompt. A slash command names nothing.
async function nameChat($: EngineInterface, drafted: string | null, text: string): Promise<void> {
  const note = await sessionAnchor($)
  const id = await $.session.id()
  const title = drafted ?? promptTitle(text)
  if (!note || !id || !title) {
    return
  }
  const key = `${id}|${note}`
  if (!drafted && (await read($, inherited)).named === key) {
    return
  }
  await recordChat($, title)
  // It is filed now: the anchor it started on by itself is its anchor.
  await update($, inherited, () => ({ session: '', note: '', named: key }))
}

// Remembers what is typed in the scratchpad's field, unsent.
async function typePad($: EngineInterface, text: string): Promise<void> {
  typedPad = text
  await update($, padDraft, () => text)
}

// Keeps a note to self on this chat's scratchpad, newest first. Nothing is written.
async function jot($: EngineInterface, text: string): Promise<void> {
  if (!text.trim()) {
    return
  }
  const when = stamp(await $.clock.now())

  await typePad($, '')
  await update($, pad, old => [{ when, text: text.trim() }, ...old])
}

const samePad = (a: PadEntry, b: PadEntry): boolean => a.when === b.when && a.text === b.text

// Takes the given entries off the pad, each once.
async function unjot($: EngineInterface, gone: PadEntry[]): Promise<void> {
  await update($, pad, old => {
    const left = [...gone]

    return old.filter(one => {
      const at = left.findIndex(other => samePad(one, other))
      if (at >= 0) {
        left.splice(at, 1)
      }

      return at < 0
    })
  })
}

// The one place the pane writes the user's own note, and only on their press, only with
// their own words: scratchpad entries go into its Journal, oldest first, read fresh from
// disk (it may be open in their editor). On a failed write the entries stay on the pad.
async function saveToNotes($: EngineInterface, entries: PadEntry[]): Promise<void> {
  const { anchor, notePath } = await read($, swatch)
  if (!anchor || !notePath || entries.length === 0) {
    return
  }
  try {
    const old = await $.fs.read(notePath)
    const text = [...entries].reverse().reduce(addJournalEntry, old)
    await $.fs.write(notePath, text)
  } catch (error) {
    await $.ui.toast(`Couldn't save to ${anchor.note}: ${String(error)}`)

    return
  }
  await unjot($, entries)
  await $.ui.toast(`Saved to ${anchor.note}'s Journal`)
}

// This chat's own transcript, if it is under the Claude projects folder.
async function ownTranscript($: EngineInterface): Promise<string> {
  const id = await $.session.id()
  if (!id || !(await $.fs.exists(P.claudeDir))) {
    return ''
  }
  for (const dir of await $.fs.list(P.claudeDir)) {
    const path = `${P.claudeDir}/${dir.name}/${id}.jsonl`
    if (dir.kind === 'dir' && (await $.fs.exists(path))) {
      return path
    }
  }

  return ''
}

// Writes a task note from the template and puts it on the board: a top-level card (no
// `parent`: standalone, no `Up:`, at the end of the column `into`, else of the first) or a
// linked sub-item under the parent's card (`Up:` the parent; a parent not on the board gets
// the task its own card so it is not lost). The board is read back into the pane. Null,
// after a toast saying why, when nothing was made: never over a note that is there.
async function makeTask($: EngineInterface, typed: string, parent: string | null, into: string): Promise<{ name: string; path: string; jira: string; where: string } | null> {
  const { title, jira } = parseNewTask(typed)
  const name = taskNoteName(title)
  if (!name) {
    await $.ui.toast('No title: no task made.')

    return null
  }
  if (!(await $.fs.exists(P.board))) {
    await $.ui.toast(`No board at ${P.board}: set one up first.`)

    return null
  }
  // One flat folder: hierarchy is the `Up:` link, never where the file sits.
  const path = `${P.tasks}/${name}.md`
  if ((await $.fs.exists(path)) || (await notePathOf($, name))) {
    await $.ui.toast(`A note named ${name} is already there: nothing written.`)

    return null
  }
  const template = (await readOr($, P.taskTemplate)) || TASK_TEMPLATE
  await writeNew($, path, taskNoteText(template, name, stamp(await $.clock.now()).slice(0, 10), parent, jira))
  const old = await $.fs.read(P.board)
  let board = parent ? insertSubItem(old, name, parent) : old
  let where = ''
  const columnOf = (text: string): string => kanbanCards(text).find(one => one.note === name)?.column || 'the first column'
  if (!parent) {
    board = insertCard(old, name, null, into)
    where = `a card at the end of ${columnOf(board)}`
  } else if (board === old) {
    // A bare anchor (not on the board): its own card, so the task is not lost.
    board = insertCard(old, name, null, P.taskColumn)
    where = `${parent} is not on the board, so a card at the end of ${columnOf(board)}`
  } else {
    const holder = kanbanCards(old).find(one => one.note === parent || one.tasks.some(task => task.note === parent))
    where = !holder || holder.note === parent ? `under ${parent}'s card` : `under ${holder.note}'s card, beside ${parent}`
  }
  await $.fs.write(P.board, board)
  const cards = kanbanCards(board)
  await update($, swatch, current => ({ ...current, cards, anchor: current.anchor ? (anchorCard(cards, current.anchor.note) ?? current.anchor) : null }))

  return { name, path, jira, where }
}

// What every new task ends with: the just-made row and a toast saying where it went.
async function madeTask($: EngineInterface, one: { name: string; path: string; jira: string; where: string }, also = ''): Promise<void> {
  await update($, justMade, () => ({ note: one.name, path: one.path, jira: one.jira, where: one.where }))
  const folder = settings.taskFolder?.trim() ? '' : " New tasks go to the notes folder's root: Setup → new task folder changes it."
  await $.ui.toast(`Made ${one.name} at ${one.path}; on the board: ${one.where}.${also}${folder}`)
}

// A new task from this chat, on the user's press: standalone ('new': a card in the
// anchor's column, else the task column setting's, else the first) or a sub-task of the
// anchor ('sub': the anchor pinned in its Agent Log so the two share context). This chat's
// transcript is pinned to it too. The chat's anchor and its prompt box are left alone:
// the just-made row offers the next step.
async function newTask($: EngineInterface, typed: string, mode: 'new' | 'sub'): Promise<void> {
  await update($, view, old => ({ ...old, taskMode: '' as const }))
  const anchored = (await read($, swatch)).anchor
  const parent = mode === 'sub' ? (anchored?.note ?? null) : null
  if (mode === 'sub' && !parent) {
    await $.ui.toast('Pick an anchor first: a sub-task hangs off it. Nothing made.')

    return
  }
  // A standalone task belongs where the chat is working: its anchor's column.
  const made = await makeTask($, typed, parent, anchored?.column || P.taskColumn)
  if (!made) {
    return
  }
  const transcript = await ownTranscript($)
  await writeTrail($, made.name, text => {
    const withParent = parent ? addPin(text, { on: true, kind: 'note', target: parent, label: parent }) : text

    return transcript ? addPin(withParent, { on: true, kind: 'session', target: transcript, label: 'the chat this task came from' }) : withParent
  })
  await madeTask($, made)
}

// A child chat of the anchor: a piece of its work that gets a chat of its own, not a
// note. The goal is one line; the draft for a new chat is copied (the pane starts no
// session), and once sent there that chat is anchored here and listed with the goal as
// its title. Nothing is written until then. The prompt box is left alone.
async function childChat($: EngineInterface, typed: string): Promise<void> {
  await update($, view, old => ({ ...old, taskMode: '' as const }))
  const goal = typed.replace(/\s+/g, ' ').trim()
  const anchor = await anchorNow($)
  if (!goal || !anchor) {
    await $.ui.toast(goal ? 'Pick an anchor first: a child chat works on it.' : 'No goal: nothing copied.')

    return
  }
  const { trail, pinPaths, everywhere, today } = await read($, swatch)
  const choice = { ...(await read($, picks)), project: true }
  const text = buildDraft(withPaths(trail.pins ?? [], pinPaths), choice, null, goal, anchor, everywhere ?? [], today.exists ? today.path : '')
  const copied = await $.ui.copy({ text })
  await $.ui.toast(copied.isCopied ? `Copied: paste it into a new chat and send. It then shows under Chats on ${anchor.note} as "${goal.slice(0, 40)}".` : `Couldn't copy the draft (${copied.reason}).`)
}

// A chat on the anchor that outgrew a chat becomes a task, on one press: the note (`Up:`
// the anchor), a linked sub-item under the anchor's card, and its own Agent Log with the
// anchor and the chat's transcript pinned. The chat's line moves from the anchor's log to
// the task's, and the chat is anchored on the task from then on, wherever it runs.
async function promoteChat($: EngineInterface, typed: string, id: string): Promise<void> {
  await update($, view, old => ({ ...old, taskMode: '' as const, promote: '' }))
  const { anchor, trail, chatFiles } = await read($, swatch)
  const chat = trail.chats.find(one => one.id === id)
  if (!anchor || !chat) {
    await $.ui.toast('That chat is no longer listed here: nothing made.')

    return
  }
  const made = await makeTask($, typed, anchor.note, '')
  if (!made) {
    return
  }
  const own = await $.session.id()
  const transcript = chatFiles?.[id] ?? (id === own ? await ownTranscript($) : '')
  await writeTrail($, made.name, text => {
    const withParent = addPin(text, { on: true, kind: 'note', target: anchor.note, label: anchor.note })
    const withChat = addChat(withParent, { ...chat, title: chat.title === CHAT_PLACEHOLDER ? made.name : chat.title, done: false })

    return transcript ? addPin(withChat, { on: true, kind: 'session', target: transcript, label: 'the chat this task came from' }) : withChat
  })
  await writeTrail($, anchor.note, text => removeChat(text, id))
  // That chat's anchor is the task now. Another session's pane follows on its next sync.
  await $.store.set(ANCHORS_KEY, withAnchor(anchorPairs(await $.store.get(ANCHORS_KEY)), id, made.name))
  if (id === own) {
    await anchorSession($, made.name)
  }
  await madeTask($, made, id === own ? ' This chat is anchored on it now.' : ' That chat is anchored on it now.')
}

// Replaces the notes folder's task template with the shipped one, on the user's press. The
// one it replaces is kept beside it, named for the day, so nothing of theirs is lost.
async function useCurrentTemplate($: EngineInterface): Promise<void> {
  const old = await readOr($, P.taskTemplate)
  if (old) {
    await $.fs.write(`${P.taskTemplate}.${stamp(await $.clock.now()).slice(0, 10)}.bak`, old)
  }
  await $.fs.write(P.taskTemplate, TASK_TEMPLATE)
  await update($, swatch, current => ({ ...current, taskTemplate: 'yours' as const }))
  await $.ui.toast(`New tasks now use the current template (Outcome, Verify, Journal). Your old one is kept beside it as a .bak file.`)
}

// The just-made task's draft for this chat's prompt box (unsent): fill in the note.
async function defineHere($: EngineInterface): Promise<void> {
  const one = await read($, justMade)
  if (!one.note) {
    return
  }
  // A typed key is already linked: the draft asks for no search then.
  const ask = one.jira ? null : await jiraAsk($)
  await fillOver($, newTaskDraft(one.path, trailPath(one.note), one.note, ask))
}

// A context draft anchored on the just-made task, copied: pasted into a new chat, it
// anchors that chat on the task (anchorFromDraft). The prompt box is left alone.
async function copyForNewChat($: EngineInterface): Promise<void> {
  const one = await read($, justMade)
  if (!one.note) {
    return
  }
  const pins = parseTrail(await readOr($, trailPath(one.note))).pins
  const choice = { ...(await read($, picks)), project: true }
  const { everywhere, today } = await read($, swatch)
  const text = buildDraft(pins, choice, null, '', { note: one.note, notePath: one.path, trail: trailPath(one.note) }, everywhere ?? [], today.exists ? today.path : '')
  const copied = await $.ui.copy({ text })
  await $.ui.toast(copied.isCopied ? `Copied: paste it into a new chat to work on ${one.note} there; it anchors that chat on it.` : `Couldn't copy the draft (${copied.reason}).`)
}

// Makes the just-made task this chat's anchor.
async function anchorOnMade($: EngineInterface): Promise<void> {
  const one = await read($, justMade)
  if (!one.note) {
    return
  }
  const { cards } = await read($, swatch)
  await chooseAnchor($, anchorCard(cards, one.note) ?? { column: '', note: one.note, detail: '', tasks: [] })
}

// `open` is macOS; elsewhere the same argument goes to xdg-open.
async function launch($: EngineInterface, argv: string[]): Promise<void> {
  let ran: { exitCode: number | null } | null = null
  try {
    ran = await $.process.run(argv)
  } catch {
    ran = null
  }
  if (ran?.exitCode === 0) {
    return
  }
  if (argv[0] === 'open') {
    try {
      const again = await $.process.run(['xdg-open', ...argv.slice(1)])
      if (again.exitCode === 0) {
        return
      }
    } catch {
      // fall through to the toast
    }
  }
  await $.ui.toast(`Couldn't run ${argv[0]}. Check the editor in Augi's setup.`)
}

async function openNote($: EngineInterface, path: string): Promise<void> {
  await launch($, editorArgv(P.editor, P.editorCommand, path))
}

async function openFolder($: EngineInterface, path: string): Promise<void> {
  await launch($, ['open', path])
}

async function openByName($: EngineInterface, name: string): Promise<void> {
  const path = await notePathOf($, name)
  if (path) {
    await openNote($, path)
  } else {
    await $.ui.toast(`No note named ${name} in your notes folder.`)
  }
}

// Writes a starter file, never over one that's there.
async function writeNew($: EngineInterface, path: string, text: string): Promise<void> {
  if (!(await $.fs.exists(path))) {
    await $.process.run(['mkdir', '-p', path.slice(0, path.lastIndexOf('/'))])
    await $.fs.write(path, text)
  }
}

async function writeStarter($: EngineInterface, field: string): Promise<void> {
  // The Agent Log folder comes with the starters, so agents' notes have a home from the start.
  await $.process.run(['mkdir', '-p', P.trails])
  if (field === 'contextNote') {
    await writeNew($, P.context, starterContext(stamp(await $.clock.now()).slice(0, 10)))
  }
  if (field === 'boardNote') {
    await writeNew($, P.board, starterBoard())
  }
  if (field === 'lensFolder') {
    await writeNew($, `${P.lenses}/focus-check.md`, starterLens())
  }
}

async function makeStarter($: EngineInterface, field: string): Promise<void> {
  await writeStarter($, field)
  await load($)
}

// A setup step uses the starter: the default path, with the starter written there.
async function useStarter($: EngineInterface, field: string): Promise<void> {
  if (settings[field as keyof Options]) {
    await setOption($, field, '')
    await $.store.set('starter', field)

    return
  }
  await makeStarter($, field)
}

async function skipStep($: EngineInterface, field: string): Promise<void> {
  await $.store.set(SKIPPED_KEY, [...(await skippedNow($)), field])
  await load($)
}

// Back to the first screen: the folder, then each step again.
async function startOver($: EngineInterface): Promise<void> {
  await $.store.delete(SKIPPED_KEY)
  await setOption($, 'notesFolder', '')
}

// Writes one of Augi's settings. A change reloads the module with it.
async function setOption($: EngineInterface, field: string, value: string): Promise<void> {
  const rows = await $.config.list()
  const row = rows.find(one => one.key === `augi-ui.${field}`) ?? rows.find(one => one.key.startsWith('augi-ui') && one.key.endsWith(`.${field}`))
  if (!row) {
    await $.ui.toast(`Couldn't find Augi's ${field} setting. Set it in /config.`)

    return
  }
  await $.config.set({ key: row.key, value })
}

// Appends one event to the usage log. It never fails a press: a log that can't be
// written is a press that isn't counted.
async function track($: EngineInterface, kind: UsageEvent['kind'], element: string, surface: string): Promise<void> {
  if (settings.trackUsage === 'no') {
    return
  }
  try {
    const { anchor, lenses } = await read($, swatch)
    const isLens = kind === 'press' && lenses.some(one => one.name === element)
    const event: UsageEvent = {
      at: new Date(await $.clock.now()).toISOString(),
      kind,
      what: isLens ? 'lens' : usageName(element),
      ...(isLens ? { lens: element } : {}),
      anchored: Boolean(anchor),
      surface,
    }
    // One short line appended by the shell: two chats can log at once.
    await $.process.run(['sh', '-c', 'mkdir -p "$(dirname "$2")" && printf \'%s\\n\' "$1" >> "$2"', 'sh', JSON.stringify(event), P.usage])
  } catch {
    // Not counted.
  }
}

async function usageNow($: EngineInterface): Promise<string> {
  return usageReport(parseUsage(await readOr($, P.usage)), await $.clock.now())
}

async function copyResume($: EngineInterface, chat: Chat): Promise<void> {
  await $.ui.copy({ text: `claude --resume ${chat.id}` })
  await $.ui.toast(`Copied: claude --resume ${chat.id}`)
}

const byColumn = (cards: Card[]): [string, Card[]][] => {
  const groups = new Map<string, Card[]>()
  for (const card of cards) {
    groups.set(card.column, [...(groups.get(card.column) ?? []), card])
  }

  return [...groups]
}

export const register: Register = (on, options) => {
  settings = options as Partial<Options>

  on('session.start', async ($, e, next) => {
    await ready($)
    await $.command.register({ name: 'augi', description: 'Show Augi pane: your anchor, its context and chats, and the lenses' })
    // A reload starts this module's variables over: the unsent scratchpad text comes back.
    typedPad = await read($, padDraft)
    // A settings change reloads the module; redraw an open pane with the new paths.
    const starter = await $.store.get('starter')
    if (typeof starter === 'string') {
      await $.store.delete('starter')
      await writeStarter($, starter)
    }
    if ((await read($, swatch)).loadedAt > 0) {
      await load($)
      // The reload dropped this module's timer; an open pane goes on following its anchor.
      watchAnchor($)
    }
    // A new interactive chat opens with the pane beside it, unless that is turned off.
    // Never in the way of the session starting: a failure here only skips the pane.
    if (e.isInteractive && e.surface) {
      await openOnce($)
    }

    return next(e)
  })

  // The desktop app and the editor start a session through the SDK: at session.start
  // nothing draws yet, so the pane opens when their window attaches instead. Once a
  // session: a reconnect does not bring back a pane the user closed. A phone joining
  // opens nothing.
  on('session.attach', async ($, e, next) => {
    const joined = await next(e)
    if (e.surface === 'desktop' || e.surface === 'vscode') {
      await ready($)
      await openOnce($)
    }

    return joined
  })

  // A /clear goes on in this process under a new session id with no session.start: the
  // pane drops the old session's anchor, and the next sync starts the new session on the
  // folder's anchor as any new chat would.
  on('session.end', async ($, e, next) => {
    const ended = await next(e)
    if (e.reason === 'clear' || e.reason === 'resume') {
      await update($, swatch, old => ({ ...old, anchor: null, notePath: '', outcome: '', next: [], jira: '', trail: EMPTY_TRAIL, chatFiles: {}, chatSeen: {}, seenAt: 0, pinPaths: {}, hits: [] }))
      // The scratchpad was that chat's; it doesn't follow into the next.
      await update($, pad, () => [])
      await typePad($, '')
      await update($, touched, () => ({ session: '', files: [] }))
      await update($, justMade, () => NO_TASK)
      await update($, view, old => ({ ...old, taskMode: '' as const, promote: '' }))
    }

    return ended
  })

  on('command.run', { command: 'augi' }, async ($, e) => {
    await ready($)
    if (e.args.trim() === 'usage') {
      return { text: `${await usageNow($)}\n\nThe log is ${P.usage}. Turn it off under Setup in the pane.` }
    }
    if (e.args.trim() === 'setup') {
      await $.store.delete(SKIPPED_KEY)
      await update($, view, old => ({ ...old, setupOpen: true }))
    }
    await load($)
    await $.ui.open({ id: PANE, title: 'Augi' })
    watchAnchor($)

    return { text: 'Augi is open. Pick an anchor, tick its context, then draft a prompt or run a lens. It only drafts; you press enter.' }
  })

  // An anchored chat goes on its anchor's log with its first prompt: titled by a pane
  // draft, else by the prompt's first line. A draft pasted into a chat with no anchor of
  // its own choosing anchors it first, so a new chat lands in the same Agent Log.
  on('prompt.submit', async ($, e, next) => {
    const title = chatTitleFrom(e.text)
    try {
      await ready($)
      if (title) {
        await anchorFromPaste($, e.text)
      }
      await nameChat($, title, e.text)
      if (title) {
        await track($, 'sent', 'draft-sent', '')
      }
    } catch (error) {
      await $.ui.toast(`Augi couldn't record this chat: ${String(error)}`)
    }

    // Sending a prompt is a good moment to catch up with what other chats wrote.
    resync($).catch(() => undefined)

    return next(e)
  })

  // Every press and submit in the pane is counted, by the control's name alone: what
  // was typed and whose note it was on are not kept. A control added later is counted too.
  on('ui.press', async ($, e, next) => {
    const ran = await next(e)
    if (e.plugin === 'augi-ui' && e.requestId === PANE) {
      await track($, 'press', e.element, e.surface)
    }

    return ran
  })

  on('ui.input', async ($, e, next) => {
    const ran = await next(e)
    // A submit is a use; a change is typing, which is not counted.
    if (e.plugin === 'augi-ui' && e.requestId === PANE && e.kind === 'submit') {
      await track($, 'input', e.element, e.surface)
    }

    return ran
  })

  // What an anchored chat makes goes on its anchor's context: an artifact it publishes,
  // ticked, and a new note it writes in the notes folder, unticked. The session keeps
  // its artifacts either way, so anchoring later pins them.
  on('tool.call', { tool: 'Artifact' }, async ($, e, next) => {
    const ran = await next(e)
    try {
      await ready($)
      // Only a page published; reads, lists and asset uploads name artifacts too.
      const isPublish = (e.action === undefined || e.action === 'publish') && !e.asset
      const url = isPublish ? JSON.stringify(ran).match(ARTIFACT_URL)?.[0] : undefined
      if (url && ran.deny === undefined && !ran.isError) {
        const one = { url, title: typeof e.title === 'string' && e.title ? e.title : 'artifact' }
        await rememberArtifact($, one)
        const note = await paneChatAnchor($)
        if (note) {
          await writeTrail($, note, text => addPin(text, artifactPin(one)))
        }
      }
    } catch (error) {
      await $.ui.toast(`Augi couldn't pin this artifact: ${String(error)}`)
    }

    return ran
  })

  // Every file the chat writes or edits is listed under Files this chat touched; made
  // is whether it was there before. Neither step throws into the tool call.
  on('tool.call', { tool: 'Edit' }, async ($, e, next) => {
    const existed = await existsBefore($, e.file_path)
    const ran = await next(e)
    await afterTouch($, e.file_path, existed, ran)

    return ran
  })

  on('tool.call', { tool: 'NotebookEdit' }, async ($, e, next) => {
    const existed = await existsBefore($, e.notebook_path)
    const ran = await next(e)
    await afterTouch($, e.notebook_path, existed, ran)

    return ran
  })

  on('tool.call', { tool: 'Write' }, async ($, e, next) => {
    await ready($)
    const path = e.file_path
    const existed = await existsBefore($, path)
    const ran = await next(e)
    await afterTouch($, path, existed, ran)
    try {
      const isNewNote = !existed && isVaultNote(path, P.notes) && !path.startsWith(`${P.trails}/`) && !path.includes('/.')
      const note = isNewNote ? await paneChatAnchor($) : null
      if (note && ran.deny === undefined && !ran.isError) {
        const name = baseName(path).replace(/\.md$/, '')
        await update($, swatch, old => ({ ...old, pinPaths: { ...old.pinPaths, [name]: path } }))
        await writeTrail($, note, text => addPin(text, { on: false, kind: 'note', target: name, label: name }))
      }
    } catch (error) {
      await $.ui.toast(`Augi couldn't pin this note: ${String(error)}`)
    }

    return ran
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Button, Input, Text } = $.ui.resolve(e)
    // Svg is drawn by the desktop, editor and phone, not the terminal.
    const Svg = e.surface === 'terminal' ? null : (($.ui.resolve(e) as unknown as { Svg?: typeof Text }).Svg ?? null)
    const { setup, vaults, found, skipped, season, cards, anchor, notePath, outcome, next, jira, jiraServer, taskTemplate, today, quick, trail, chatSeen, seenAt, pinPaths, hits, lenses, sessions, loadedAt } = await read($, swatch)
    const everywhere = (await read($, swatch)).everywhere ?? []
    // Jira shows only when it is on: a server found or named, a project or site set, or a key on the anchor.
    const isJiraOn = Boolean(jiraServer || P.jiraServer || P.jiraProject || P.jiraSite || jira)
    const choice = await read($, picks)
    const shown = await read($, view)
    const ownArtifacts = await sessionArtifacts($)
    const ownFiles = await sessionTouched($)
    let root = ''
    if (ownFiles.length > 0) {
      try {
        root = await $.session.cwd()
      } catch {
        root = ''
      }
    }
    const shownFiles = shown.filesOpen ? ownFiles : ownFiles.slice(0, 5)
    const jotted = await read($, pad)
    const madeTask = await read($, justMade)
    // Drawing never depends on it: with no id, no row is marked as this chat.
    const ownId = await $.session.id().catch(() => '')
    const carried = await read($, inherited)
    const isCarried = Boolean(anchor) && carried.note === anchor?.note && carried.session === ownId
    // Every field but a standalone task's needs an anchor; with none it reads as closed.
    const taskMode = shown.taskMode !== 'new' && !anchor ? '' : (shown.taskMode ?? '')
    const mark = (isOn: boolean): string => (isOn ? '☑' : '☐')
    const arrow = (isOpen: boolean): string => (isOpen ? '▾' : '▸')
    const flip = (key: keyof View) => () => update($, view, old => ({ ...old, [key]: !old[key] }))
    const openChats = trail.chats.filter(one => !one.done)
    const doneChats = trail.chats.filter(one => one.done)
    const promoting = trail.chats.find(one => one.id === shown.promote)
    // A chat's row: its goal, then when it was last active (its transcript, when it is on
    // this computer; else when it was filed).
    const chatLabel = (chat: Chat): string => `◦ ${chat.title}${chat.id === ownId ? ' (this chat)' : ''} · ${lastActive(chatSeen?.[chat.id] ?? 0, seenAt ?? 0) || chat.when.slice(5)}`
    const pins = withPaths(trail.pins ?? [], pinPaths ?? {})
    const contextCount = contextItems(pins, choice, anchor ? { note: anchor.note, notePath, trail: trailPath(anchor.note) } : null, everywhere, today.exists ? today.path : '').length
    const PIN_ICON: Record<Pin['kind'], string> = { note: '▤', file: '📄', session: '◦', artifact: '◇', url: '🔗' }
    const pinKey = (pin: Pin): string => `${pin.kind}-${pin.target}`
    const onPins = (change: (text: string) => string) => (anchor ? writeTrail($, anchor.note, change) : undefined)
    // The note's Next items, then the card's sub-items not already among them.
    const openTasks = anchor ? [...(next ?? []), ...anchor.tasks.filter(task => !(next ?? []).some(one => one.text === task.text))] : []
    type Tone = 'claude' | 'success' | 'remember' | 'suggestion'
    // A card's title row: the title as text, its fold button at the trailing edge.
    const head = (key: keyof View, title: string, tone?: Tone, button: string = `head-${key}`, hotkey?: string) => (
      <Box key={`title-${key}`} flexDirection="row" justifyContent="space-between" alignItems="center" gap={1}>
        <Text bold color={tone}>{title}</Text>
        <Button key={button} plain dimColor label={shown[key] ? '▾ hide' : '▸ show'} hotkey={hotkey} onPress={flip(key)} />
      </Box>
    )
    // A section's title row, larger than a card's.
    const section = (key: keyof View, title: string) => (
      <Box key={`title-${key}`} flexDirection="row" justifyContent="space-between" alignItems="center" gap={1} marginBottom={1}>
        <Text bold size="heading">{title}</Text>
        <Button key={`head-${key}`} plain dimColor label={shown[key] ? '▾ hide' : '▸ show'} onPress={flip(key)} />
      </Box>
    )
    // A list row: what it is on the left, what you can do with it lined up on the right.
    const line = (key: string, main: RenderChildren, actions: RenderChildren) => (
      <Box key={key} flexDirection="row" justifyContent="space-between" alignItems="center" gap={1}>
        <Box flexDirection="row" alignItems="center" flexShrink={1} gap={1}>{main}</Box>
        <Box flexDirection="row" alignItems="center" flexShrink={0} gap={1}>{actions}</Box>
      </Box>
    )
    // What every draft carries without being listed: switched under Setup.
    const standing = [anchor && choice.project ? `notes for ${anchor.note}` : '', choice.aboutMe ? 'your overall context' : '', choice.today && today.exists ? "today's note" : '', choice.openAugi ? 'a search of your notes' : ''].filter(Boolean)
    // A search hit or recent session, offered to the anchor or to every chat.
    const offer = (pin: Pin, label: string) => line(
      `offer-${pinKey(pin)}`,
      <Text key={`offer-label-${pinKey(pin)}`} wrap="truncate-end">{`+ ${label}`}</Text>,
      [
        anchor && <Button key={`offer-anchor-${pinKey(pin)}`} label="pin to anchor" onPress={() => pinIt($, anchor.note, pin)} />,
        <Button key={`offer-all-${pinKey(pin)}`} label="pin everywhere" onPress={() => pinEverywhere($, pin)} />,
      ],
    )
    const CARD = { flexDirection: 'column', borderStyle: 'round', paddingX: 1, marginBottom: 1 } as const
    const openQuestions = trail.questions.filter(one => !one.done)
    const doneQuestions = trail.questions.filter(one => one.done)
    const gaps = setup.filter(item => !item.ok).length + (taskTemplate === 'old' ? 1 : 0)
    const short = (path: string): string => (path.startsWith(`${P.notes}/`) ? path.slice(P.notes.length + 1) : path.startsWith(P.home) ? `~${path.slice(P.home.length)}` : path)
    const step = setup[0]?.ok ? STEPS.find(one => !skipped.includes(one.field) && !setup.find(item => item.field === one.field)?.ok) : undefined
    const editorPicker = (
      <Box key="editor-picker" flexDirection="column">
        <Text>Open notes (markdown in the notes folder) with:</Text>
        <Box flexDirection="row">
          {EDITORS.map(name => (
            <Button key={`editor-${name}`} plain label={`${P.editor === name ? '●' : '○'} ${name} `} onPress={() => setOption($, 'editor', name)} />
          ))}
        </Box>
        {P.editor === 'custom' && (
          <Input key="set-editorCommand" label="Command" placeholder="zed {file}" value={settings.editorCommand ?? ''} onSubmit={value => setOption($, 'editorCommand', value)} />
        )}
        <Text>Open other files (code, docs outside the notes) with:</Text>
        <Box flexDirection="row">
          {FILE_EDITORS.map(name => (
            <Button key={`file-editor-${name}`} plain label={`${P.fileEditor === name ? '●' : '○'} ${name} `} onPress={() => setOption($, 'fileEditor', name)} />
          ))}
        </Box>
      </Box>
    )

    if (setup.length > 0 && !setup[0]?.ok) {
      return (
        <Box flexDirection="column">
          <Text bold>Set up Augi</Text>
          <Text>Augi works from a folder of markdown notes: a context note about you, a board of projects to anchor on, and lenses (saved prompts). Start with the folder.</Text>
          <Text> </Text>
          {vaults.length > 0 && <Text>Your Obsidian vaults:</Text>}
          {vaults.slice(0, 9).map((vault, index) => (
            <Button key={`vault-${index}`} label={`${vault.startsWith(P.home) ? `~${vault.slice(P.home.length)}` : vault}`} hotkey={String(index + 1)} onPress={() => setOption($, 'notesFolder', vault)} />
          ))}
          {vaults.length > 0 && <Text dimColor>Or type another folder:</Text>}
          {vaults.length === 0 ? (
            <Input key="set-notesFolder" label="Notes folder" placeholder="~/Documents/Notes" value={settings.notesFolder ?? ''} autoFocus onSubmit={value => setOption($, 'notesFolder', value)} />
          ) : (
            <Input key="set-notesFolder" label="Notes folder" placeholder="~/Documents/Notes" value={settings.notesFolder ?? ''} onSubmit={value => setOption($, 'notesFolder', value)} />
          )}
          {Boolean(setup[0]?.path) && <Text dimColor>Not found: {setup[0]?.path}</Text>}
          <Text> </Text>
          {editorPicker}
          <Text> </Text>
          <Text dimColor>Next you point Augi at your context note, board and lenses, or let it create starters. All of it is also in /config under augi.</Text>
        </Box>
      )
    }

    if (step) {
      const index = STEPS.indexOf(step)
      const hits = found[step.field] ?? []

      return (
        <Box flexDirection="column">
          <Text bold>Set up Augi · step {index + 2} of {STEPS.length + 1}</Text>
          <Text dimColor>✓ Notes folder: {short(P.notes)}</Text>
          {STEPS.slice(0, index).map(done => (
            <Text key={`done-${done.field}`} dimColor>{`${setup.find(item => item.field === done.field)?.ok ? '✓' : '–'} ${done.title}: ${short(setup.find(item => item.field === done.field)?.path ?? '')}`}</Text>
          ))}
          <Text> </Text>
          <Text bold>{step.title}</Text>
          <Text>{step.why}</Text>
          <Text> </Text>
          {hits.length > 0 && <Text>Found in your notes:</Text>}
          {hits.map((hit, at) => (
            <Button key={`found-${at}`} label={`${hit}`} hotkey={String(at + 1)} onPress={() => setOption($, step.field, hit)} />
          ))}
          {hits.length === 0 && <Text dimColor>Nothing like it found in your notes.</Text>}
          <Button key="use-starter" label={`create a starter at ${step.starter}`} hotkey="v" onPress={() => useStarter($, step.field)} />
          <Input key={`set-${step.field}`} label="or type a path" placeholder={step.starter} value="" onSubmit={value => setOption($, step.field, value)} />
          <Button key="skip-step" plain label="skip for now" hotkey="e" onPress={() => skipStep($, step.field)} />
          <Text> </Text>
          <Button key="start-over" plain label="← change the notes folder" onPress={() => startOver($)} />
          <Text dimColor>Saved in Claude Code's settings. Come back with /augi setup, or ⚙️ Setup in the pane.</Text>
        </Box>
      )
    }

    return (
      <Box flexDirection="column">
        {gaps > 0 && <Text dimColor>{gaps} setup {gaps === 1 ? 'item needs' : 'items need'} you: open setup below.</Text>}
        {Svg && <Svg key="augi-face" source={AUGI_SVG} alt="Augi" width={40} height={40} />}
        <Box key="section-chat" flexDirection="column">{section('chatOpen', `This chat${anchor ? ` · ${anchor.note}` : ''}`)}</Box>
        {shown.chatOpen && (
          <Box key="chat-body" flexDirection="column">
            <Box key="card-anchor" {...CARD} borderColor="claude">
              {head('anchorOpen', `⚓ Anchor${anchor && !shown.anchorOpen ? ` · ${anchor.note}` : ''}`, 'claude')}
              {shown.anchorOpen && (
                <Box key="anchor-body" flexDirection="column">
                  <Button key="anchor-menu" label={`${arrow(shown.anchorsOpen)} ${anchor ? anchor.note : 'pick a project or task'}`} hotkey="a" onPress={flip('anchorsOpen')} />
                  {shown.anchorsOpen && (
                    <Box key="anchor-list" flexDirection="column" paddingLeft={2} marginBottom={1}>
                      {byColumn(cards).map(([column, list]) => (
                        <Box key={`column-${column}`} flexDirection="column">
                          <Text dimColor>{column}</Text>
                          {list.map(card => (
                            <Box key={`card-box-${card.note}`} flexDirection="column" paddingLeft={2}>
                              <Button key={`card-${card.note}`} plain label={`${anchor?.note === card.note ? '●' : '○'} ${card.note}`} onPress={() => chooseAnchor($, card)} />
                              {card.tasks.map(task => task.note).filter((note): note is string => Boolean(note)).map(note => (
                                <Box key={`card-sub-box-${card.note}-${note}`} flexDirection="column" paddingLeft={2}>
                                  <Button key={`card-sub-${card.note}-${note}`} plain label={`${anchor?.note === note ? '●' : '○'} ${note}`} onPress={() => chooseAnchor($, subCard(card, note))} />
                                </Box>
                              ))}
                            </Box>
                          ))}
                        </Box>
                      ))}
                      {anchor && <Button key="anchor-clear" plain dimColor label="○ no anchor" onPress={() => clearAnchor($)} />}
                    </Box>
                  )}
                  {isCarried && (
                    <Box key="anchor-carried" flexDirection="column">
                      <Text dimColor wrap="wrap">↩ From your last chat in this folder. It is filed here with its first prompt.</Text>
                      <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                        <Button key="inherit-keep" label="keep" onPress={() => keepInherited($)} />
                        <Button key="inherit-change" label="change" onPress={() => update($, view, old => ({ ...old, anchorsOpen: true }))} />
                        <Button key="inherit-none" dimColor label="no anchor" onPress={() => clearAnchor($)} />
                      </Box>
                    </Box>
                  )}
                  {anchor && (
                    <Box key="anchor-links" flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                      <Button key="open-note" label="🧠 Human Notes" hotkey="p" onPress={() => (notePath ? openNote($, notePath) : openByName($, anchor.note))} />
                      <Button key="open-trail" label="🤖 Agent Log" hotkey="k" onPress={() => writeTrail($, anchor.note, text => text).then(() => openNote($, trailPath(anchor.note)))} />
                    </Box>
                  )}
                  {anchor && jira && (
                    <Box key="jira-linked" flexDirection="row" alignItems="center" gap={1}>
                      {P.jiraSite
                        ? <Button key="jira-open" plain label={`🔗 ${jira}`} onPress={() => launch($, ['open', jiraBrowseUrl(P.jiraSite, jira)])} />
                        : <Text key="jira-key" dimColor>{`🔗 ${jira}`}</Text>}
                      <Button key="jira-update" label="↑ update Jira" onPress={() => updateJira($)} />
                      <Button key="jira-unlink" plain dimColor label="unlink" onPress={() => writeJira($, '')} />
                    </Box>
                  )}
                  {anchor && !jira && isJiraOn && (
                    <Box key="jira-unlinked" flexDirection="row" alignItems="center" gap={1}>
                      <Input key="jira-link" label="🔗" placeholder="paste a Jira key or link: Enter links it to this anchor" submitLabel="link" onSubmit={value => linkJira($, value)} />
                      <Button key="jira-find" label="find in Jira" onPress={() => findInJira($)} />
                    </Box>
                  )}
                  {anchor && (
                    <Button key="anchor-details" plain dimColor label={`${arrow(shown.detailsOpen)} outcome and next (${openTasks.length})`} onPress={flip('detailsOpen')} />
                  )}
                  {anchor && shown.detailsOpen && (
                    <Box key="anchor-detail" flexDirection="column" paddingLeft={2} marginBottom={1}>
                      {anchor.detail && <Text dimColor>{anchor.detail}</Text>}
                      {outcome && <Text>🎯 {outcome}</Text>}
                      {openTasks.map(task => (
                        task.note
                          ? <Button key={`task-${task.text}`} plain label={`☐ ${task.text} ↗`} onPress={() => openByName($, task.note ?? '')} />
                          : <Text key={`task-${task.text}`}>☐ {task.text}</Text>
                      ))}
                    </Box>
                  )}
                  {taskMode === '' && (
                    <Box key="task-buttons" flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                      {anchor && <Button key="task-child" label="＋ child chat" onPress={() => update($, view, old => ({ ...old, taskMode: 'child' as const }))} />}
                      {anchor && (
                        <Button key="task-sub" label={`＋ sub-task of ${anchor.note.length <= 32 ? anchor.note : `${anchor.note.slice(0, 31)}…`}`} onPress={() => update($, view, old => ({ ...old, taskMode: 'sub' as const }))} />
                      )}
                      <Button key="task-new" label="＋ new task" onPress={() => update($, view, old => ({ ...old, taskMode: 'new' as const }))} />
                    </Box>
                  )}
                  {(taskMode === 'new' || taskMode === 'sub') && (
                    <Box key="task-form" flexDirection="column">
                      <Input
                        key={`task-title-${taskMode}`}
                        label={taskMode === 'sub' ? '＋ sub-task' : '＋ new task'}
                        placeholder={taskMode === 'sub' ? `title (add PROJ-123 to link Jira): Enter makes it under ${anchor?.note ?? 'the anchor'}` : 'title (add PROJ-123 to link Jira): Enter makes a standalone task'}
                        submitLabel="create"
                        autoFocus
                        onSubmit={value => newTask($, value, taskMode)}
                      />
                      <Button key="task-cancel" plain dimColor label="cancel" onPress={() => update($, view, old => ({ ...old, taskMode: '' as const }))} />
                    </Box>
                  )}
                  {taskMode === 'child' && (
                    <Box key="task-form" flexDirection="column">
                      <Text dimColor wrap="wrap">A piece of this work in a chat of its own: no note, no card. It is listed under Chats with this goal.</Text>
                      <Input
                        key="task-title-child"
                        label="＋ child chat"
                        placeholder="its goal in one line: Enter copies a draft to paste into a new chat"
                        submitLabel="copy"
                        autoFocus
                        onSubmit={value => childChat($, value)}
                      />
                      <Button key="task-cancel" plain dimColor label="cancel" onPress={() => update($, view, old => ({ ...old, taskMode: '' as const }))} />
                    </Box>
                  )}
                  {taskMode === 'promote' && promoting && (
                    <Box key="task-form" flexDirection="column">
                      <Text dimColor wrap="wrap">{`Make this chat a task of ${anchor?.note ?? 'the anchor'}: a note, a line under its card, and the chat moves to the task's log.`}</Text>
                      <Input
                        key={`task-title-promote`}
                        label="→ task"
                        placeholder="the task's title: Enter makes it"
                        value={promoting.title === CHAT_PLACEHOLDER ? '' : promoting.title}
                        submitLabel="create"
                        autoFocus
                        onSubmit={value => promoteChat($, value, promoting.id)}
                      />
                      <Button key="task-cancel" plain dimColor label="cancel" onPress={() => update($, view, old => ({ ...old, taskMode: '' as const, promote: '' }))} />
                    </Box>
                  )}
                  {madeTask.note && (
                    <Box key="just-made" flexDirection="column" marginTop={1}>
                      <Text dimColor>{`✓ just made: ${madeTask.note}`}</Text>
                      {madeTask.where && <Text key="made-where" dimColor wrap="wrap">{`on the board: ${madeTask.where}`}</Text>}
                      <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                        <Button key="made-open" label="open" onPress={() => openNote($, madeTask.path)} />
                        <Button key="made-board" label="open the board" onPress={() => openNote($, P.board)} />
                        <Button key="made-define" label="define here" onPress={() => defineHere($)} />
                        <Button key="made-copy" label="copy for a new chat" onPress={() => copyForNewChat($)} />
                        {anchor?.note !== madeTask.note && <Button key="made-anchor" label="anchor on it" onPress={() => anchorOnMade($)} />}
                        <Button key="made-dismiss" plain dimColor label="dismiss" onPress={() => update($, justMade, () => NO_TASK)} />
                      </Box>
                    </Box>
                  )}
                </Box>
              )}
            </Box>

            {anchor && (
              <Box key="card-trail" {...CARD} borderColor="success">
                {head('trailOpen', `🌱 Chats on ${anchor.note} · ${openChats.length} open${openQuestions.length > 0 ? `, ${openQuestions.length} ${openQuestions.length === 1 ? 'question' : 'questions'}` : ''}`, 'success')}
                {shown.trailOpen && (
                  <Box key="trail-body" flexDirection="column">
                    {openChats.map(chat => line(
                      `chat-row-${chat.id}`,
                      <Button key={`chat-${chat.id}`} plain label={chatLabel(chat)} onPress={() => copyResume($, chat)} />,
                      [
                        <Button key={`chat-task-${chat.id}`} label="→ task" onPress={() => update($, view, old => ({ ...old, anchorOpen: true, taskMode: 'promote' as const, promote: chat.id }))} />,
                        <Button key={`chat-done-${chat.id}`} dimColor label="done" onPress={() => chatDone($, anchor.note, chat.id, true)} />,
                      ],
                    ))}
                    {openChats.length === 0 && <Text key="trail-empty" dimColor wrap="wrap">No open chats. A chat anchored here is listed with its first prompt; ＋ child chat starts one with a goal.</Text>}
                    {doneChats.length > 0 && (
                      <Button key="chats-done" plain dimColor label={`${arrow(shown.chatsDoneOpen)} done chats (${doneChats.length})`} hotkey="x" onPress={flip('chatsDoneOpen')} />
                    )}
                    {shown.chatsDoneOpen && doneChats.map(chat => line(
                      `chat-row-${chat.id}`,
                      <Button key={`chat-${chat.id}`} plain dimColor label={`✓ ${chat.title} · ${chat.when.slice(5, 10)}`} onPress={() => copyResume($, chat)} />,
                      <Button key={`chat-reopen-${chat.id}`} dimColor label="reopen" onPress={() => chatDone($, anchor.note, chat.id, false)} />,
                    ))}
                    {openQuestions.length > 0 && <Text key="trail-questions" dimColor>Questions</Text>}
                    {openQuestions.map(one => line(
                      `question-row-${one.when}-${one.text}`,
                      <Button key={`ask-${one.when}`} plain label={`? ${one.text}`} onPress={() => draft($, null, one.text)} />,
                      <Button key={`done-${one.when}`} dimColor label="done" onPress={() => writeTrail($, anchor.note, text => setDone(text, one, true))} />,
                    ))}
                    {doneQuestions.length > 0 && (
                      <Button key="show-done" plain dimColor label={`${arrow(shown.doneOpen)} done questions (${doneQuestions.length})`} onPress={flip('doneOpen')} />
                    )}
                    {shown.doneOpen && doneQuestions.map(one => line(
                      `question-${one.when}-${one.text}`,
                      <Text key={`done-text-${one.when}`} dimColor>{`✓ ${one.text}`}</Text>,
                      <Button key={`reopen-${one.when}`} dimColor label="reopen" onPress={() => writeTrail($, anchor.note, text => setDone(text, one, false))} />,
                    ))}
                    {trail.saved.length > 0 && <Text key="trail-saved" dimColor>Saved notes</Text>}
                    {trail.saved.map(name => (
                      <Button key={`saved-${name}`} plain label={`▤ ${name}`} onPress={() => openByName($, name)} />
                    ))}
                  </Box>
                )}
              </Box>
            )}

            {anchor && (
              <Box key="card-pins" {...CARD} borderColor="remember">
                {head('pinsOpen', `📌 Pinned to ${anchor.note} · ${pins.length}`, 'remember')}
                {shown.pinsOpen && <Text key="pins-hint" dimColor>{pins.length === 0 ? 'Nothing pinned yet. ' : ''}☑ goes into this anchor's drafts · ☐ is a link only: press a pin to switch it.</Text>}
                {shown.pinsOpen && artifactsFirst(pins).map(pin => line(
                  `pin-${pinKey(pin)}`,
                  <Button key={`pin-tick-${pinKey(pin)}`} plain label={`${mark(pin.on)} ${PIN_ICON[pin.kind]} ${pin.label}`} onPress={() => onPins(text => setPinOn(text, pin, !pin.on))} />,
                  [
                    <Button key={`pin-open-${pinKey(pin)}`} label={pin.kind === 'session' ? 'resume' : 'open'} onPress={() => openPin($, pin)} />,
                    <Button key={`pin-drop-${pinKey(pin)}`} dimColor label="✕ unpin" onPress={() => onPins(text => removePin(text, pin))} />,
                  ],
                ))}
                {shown.pinsOpen && (
                  <Input key={`pin-paste-${pins.length}`} label="⧉" placeholder="paste an artifact link, [[note]] or file path: Enter pins it to this anchor" submitLabel="pin" onSubmit={value => pinTyped($, value)} />
                )}
              </Box>
            )}

            {ownFiles.length > 0 && (
              <Box key="card-files" {...CARD} borderColor="subtle">
                {head('touchedOpen', `📄 Files this chat touched · ${ownFiles.length}`)}
                {shown.touchedOpen && shownFiles.map(one => {
                  const pin = touchedPin(one.path, P.notes)
                  const isPinned = pins.some(other => other.kind === pin.kind && other.target === pin.target)

                  return line(
                    `touched-${one.path}`,
                    [
                      <Text key={`touched-path-${one.path}`} wrap="truncate-start">{shortPath(one.path, P.notes, P.home, root)}</Text>,
                      one.made && <Text key={`touched-new-${one.path}`} dimColor>new</Text>,
                    ],
                    [
                      <Button key={`touched-open-${one.path}`} label="open" onPress={() => openPath($, one.path)} />,
                      anchor && (isPinned
                        ? <Text key={`touched-pinned-${one.path}`} dimColor>pinned</Text>
                        : <Button key={`touched-pin-${one.path}`} label="pin" onPress={() => pinIt($, anchor.note, pin)} />),
                    ],
                  )
                })}
                {shown.touchedOpen && ownFiles.length > 5 && (
                  <Button key="files-all" plain dimColor label={`${arrow(shown.filesOpen)} ${shown.filesOpen ? 'show fewer' : `show all (${ownFiles.length})`}`} onPress={flip('filesOpen')} />
                )}
                {shown.touchedOpen && !anchor && <Text dimColor>Pick an anchor to pin these to it, as links.</Text>}
              </Box>
            )}

            <Box key="card-pad" {...CARD} borderColor="subtle">
              {head('padOpen', `📝 Scratchpad · this chat${jotted.length > 0 ? ` · ${jotted.length}` : ''}`)}
              {shown.padOpen && (
                <Box key="pad-body" flexDirection="column">
                  <Input
                    key={`pad-input-${jotted.length}`}
                    placeholder="a note, a question, a side quest: Enter keeps it here for this chat"
                    submitLabel="keep"
                    rows={2}
                    value={typedPad}
                    onInput={value => typePad($, value)}
                    onSubmit={value => jot($, value)}
                  />
                  {jotted.map((one, at) => line(
                    `pad-row-${at}`,
                    <Text key={`pad-text-${at}`} wrap="wrap">{`${one.when.slice(11, 16)} ${one.text}`}</Text>,
                    [
                      <Button key={`pad-ask-${at}`} label="→ ask" onPress={() => draft($, null, one.text)} />,
                      anchor && notePath && <Button key={`pad-save-${at}`} label="→ notes" onPress={() => saveToNotes($, [one])} />,
                      today.exists && <Button key={`pad-today-${at}`} label="→ today" onPress={() => padToToday($, [one])} />,
                      <Button key={`pad-drop-${at}`} dimColor label="drop" onPress={() => unjot($, [one])} />,
                    ],
                  ))}
                  {anchor && notePath && jotted.length >= 2 && (
                    <Button key="pad-save-all" plain label="→ save all to Human Notes" onPress={() => saveToNotes($, jotted)} />
                  )}
                  {!(anchor && notePath) && <Text dimColor>{anchor ? `No note found for ${anchor.note}: entries stay here.` : 'Pick an anchor to save these to its Human Notes.'}</Text>}
                </Box>
              )}
            </Box>
          </Box>
        )}

        <Box key="section-anywhere" flexDirection="column">{section('anywhereOpen', 'Always · you, your context, lenses and setup')}</Box>
        {shown.anywhereOpen && (
          <Box key="anywhere-body" flexDirection="column">
            <Box key="card-you" {...CARD} borderColor="claude">
              {head('youOpen', '🧭 You', 'claude')}
              {shown.youOpen && (
                <Box key="you-body" flexDirection="column">
                  <Text key="season" wrap="wrap">{`Season: ${season}`}</Text>
                  <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                    <Button key="edit-context" label="✏️ my context" hotkey="s" onPress={() => openNote($, P.context)} />
                    <Button key="open-board" label="📋 the board" hotkey="d" onPress={() => openNote($, P.board)} />
                    {quick?.overview && <Button key="run-overview" label="🔭 what's open, what's next" onPress={() => draft($, { name: 'overview', path: quick.overview, description: '' })} />}
                  </Box>
                </Box>
              )}
            </Box>

            {today.path && (
              <Box key="card-today" {...CARD} borderColor="success">
                {head('todayOpen', `☀️ Today · ${today.day}`, 'success')}
                {shown.todayOpen && (
                  <Box key="today-body" flexDirection="column">
                    {!today.exists && <Text key="today-missing" dimColor wrap="wrap">{`No note for today yet (${short(today.path)}). Make it in your editor and it shows here.`}</Text>}
                    {today.todo.map((one, at) => (
                      <Text key={`today-todo-${at}`} wrap="truncate-end">{`☐ ${one}`}</Text>
                    ))}
                    {today.exists && today.todo.length === 0 && <Text key="today-clear" dimColor>Nothing open in today's note.</Text>}
                    {today.exists && (
                      <Input key={`today-log-${shown.logged ?? 0}`} label="+ log" placeholder="what you just did or decided: Enter adds it to today's note" submitLabel="log" onSubmit={value => logToday($, value)} />
                    )}
                    {today.exists && (
                      <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                        <Button key="today-open" label="open" onPress={() => openNote($, today.path)} />
                        {quick?.standup && <Button key="run-standup" label="draft my standup" onPress={() => draft($, { name: 'standup', path: quick.standup, description: '' })} />}
                        <Button key="pick-today" plain label={`${mark(choice.today)} goes into every draft`} onPress={() => update($, picks, old => ({ ...old, today: !old.today }))} />
                      </Box>
                    )}
                  </Box>
                )}
              </Box>
            )}

            <Box key="card-context" {...CARD} borderColor="remember">
              {head('contextOpen', `📎 Context · ${contextCount} going in`, 'remember')}
              {shown.contextOpen && (
                <Box key="context-body" flexDirection="column" marginBottom={1}>
                  <Text dimColor>Pins here go into every chat's draft, whatever its anchor (☑ on, ☐ kept but left out). An anchor's own pins are under This chat.</Text>
                  {standing.length > 0 && <Text key="standing" dimColor>{`+ ${standing.join(', ')} (switch these under Setup)`}</Text>}
                  {!anchor && ownArtifacts.map(one => (
                    <Text key={`made-${one.url}`} dimColor>{`◇ ${one.title}`}</Text>
                  ))}
                  {!anchor && ownArtifacts.length > 0 && <Text dimColor>Artifacts this chat published: picking an anchor keeps them with it.</Text>}
                  {everywhere.map(pin => line(
                    `all-${pinKey(pin)}`,
                    <Button key={`all-tick-${pinKey(pin)}`} plain label={`${mark(pin.on)} ${PIN_ICON[pin.kind]} ${pin.label}`} onPress={() => setEverywhere($, list => list.map(one => (samePin(one, pin) ? { ...one, on: !one.on } : one)))} />,
                    [
                      <Button key={`all-open-${pinKey(pin)}`} label={pin.kind === 'session' ? 'resume' : 'open'} onPress={() => openPin($, pin)} />,
                      <Button key={`all-drop-${pinKey(pin)}`} dimColor label="✕ unpin" onPress={() => setEverywhere($, list => list.filter(one => !samePin(one, pin)))} />,
                    ],
                  ))}
                  <Input key="pin-search" label="🔎" placeholder="search notes to pin (OpenAugi): Enter searches" submitLabel="search" onSubmit={value => searchNotes($, value)} />
                  {(hits ?? []).map(hit => offer(hit, hit.label))}
                  {(hits ?? []).length > 0 && <Button key="hits-clear" plain dimColor label="clear results" onPress={() => update($, swatch, old => ({ ...old, hits: [] }))} />}
                  <Input key={`all-paste-${everywhere.length}`} label="⧉" placeholder="paste an artifact link, [[note]] or file path: Enter pins it for every chat" submitLabel="pin" onSubmit={value => pinTyped($, value, 'everywhere')} />
                  <Button key="sessions-menu" plain dimColor label={`${arrow(shown.sessionsOpen)} pin a recent coding session`} onPress={flip('sessionsOpen')} />
                  {shown.sessionsOpen && sessions.length === 0 && <Text dimColor>No recent coding sessions found.</Text>}
                  {shown.sessionsOpen && sessions.map(one => offer(sessionPin(one), `${one.repo} · ${ago(loadedAt, one.mtimeMs)} · ${one.title}`))}
                </Box>
              )}
              <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                <Button key="draft" variant="primary" label="💬 draft a prompt with this context" hotkey="n" onPress={() => draft($, null)} />
                <Button key="copy-draft" label="⧉ copy it for a new chat" onPress={() => copyDraft($)} />
              </Box>
            </Box>

            <Box key="card-lenses" {...CARD} borderColor="suggestion">
              {head('lensesOpen', `🔍 Lenses (saved prompts) · ${lenses.length}`, 'suggestion', 'lens-menu', 'l')}
              {shown.lensesOpen && lenses.map((lens, index) => line(
                `lens-row-${lens.name}`,
                <Button key={lens.name} plain label={lens.name} hotkey={index < 9 ? String(index + 1) : undefined} onPress={() => draft($, lens)} />,
                <Button key={`edit-${lens.name}`} dimColor label="edit" hotkey={EDIT_KEYS[index]} onPress={() => openNote($, lens.path)} />,
              ))}
              {shown.lensesOpen && (
                <Input key={`new-lens-${lenses.length}`} label="+ new lens" placeholder="name it: Enter opens it in your editor" submitLabel="create" onSubmit={value => newLens($, value)} />
              )}
              {shown.lensesOpen && <Button key="all-lenses" plain dimColor label="open the lens folder" hotkey="f" onPress={() => openFolder($, P.lenses)} />}
            </Box>

            <Box key="card-setup" {...CARD} borderColor="subtle">
              {head('setupOpen', `⚙️ Setup${gaps > 0 ? ` · ${gaps} to fix` : ''}`, undefined, 'setup-menu', 'z')}
              {shown.setupOpen && (
                <Box key="setup-body" flexDirection="column">
                  <Text bold>What goes into every draft</Text>
                  <Button key="about-me" plain label={`${mark(choice.aboutMe)} my context note and board`} hotkey="m" onPress={() => update($, picks, old => ({ ...old, aboutMe: !old.aboutMe }))} />
                  <Button key="pick-project" plain label={`${mark(choice.project)} the anchor's 🧠 human and 🤖 agent notes`} onPress={() => update($, picks, old => ({ ...old, project: !old.project }))} />
                  <Button key="openaugi" plain label={`${mark(choice.openAugi)} a search of my notes (with OpenAugi)`} onPress={() => update($, picks, old => ({ ...old, openAugi: !old.openAugi }))} />
                  <Box key="setup-files" flexDirection="column" marginTop={1}>
                    <Text bold>Files and folders</Text>
                    {setup.map(item => (
                      <Box key={`setup-${item.field}`} flexDirection="column">
                        <Text>{`${item.ok ? '✓' : '✗'} ${item.label}: ${item.path.startsWith(`${P.notes}/`) ? item.path.slice(P.notes.length + 1) : item.path}`}</Text>
                        {!item.ok && item.starter && <Button key={`starter-${item.field}`} label="create a starter" onPress={() => makeStarter($, item.field)} />}
                        <Input
                          key={`set-${item.field}`}
                          label="path"
                          value={String(settings[item.field as keyof Options] ?? '')}
                          onSubmit={value => setOption($, item.field, value)}
                        />
                      </Box>
                    ))}
                    <Input key="set-logFolder" label="agent log folder" value={settings.logFolder || settings.trailFolder || ''} placeholder="OpenAugi/Agent-Logs" onSubmit={value => setOption($, 'logFolder', value)} />
                    <Input key="set-dailyNote" label="daily note" value={settings.dailyNote ?? ''} placeholder="Daily/{{date:YYYY-MM-DD}}.md; empty reads Obsidian's setting, none turns it off" onSubmit={value => setOption($, 'dailyNote', value)} />
                    <Text key="daily-now" dimColor wrap="wrap">{today.path ? `Today's note: ${short(today.path)}${today.exists ? '' : ' (not made yet)'}` : 'No daily note: the Today card is off.'}</Text>
                    <Input key="set-dailyHeading" label="log heading in it" value={settings.dailyHeading ?? ''} placeholder="the end of the note" onSubmit={value => setOption($, 'dailyHeading', value)} />
                    <Input key="set-taskFolder" label="new task folder" value={settings.taskFolder ?? ''} placeholder="the notes folder's root" onSubmit={value => setOption($, 'taskFolder', value)} />
                    {!settings.taskFolder?.trim() && <Text key="task-folder-unset" dimColor wrap="wrap">⚠ No task folder set: new task notes go to the notes folder's root.</Text>}
                    <Input key="set-taskColumn" label="new task column" value={settings.taskColumn ?? ''} placeholder="the anchor's column, else the board's first" onSubmit={value => setOption($, 'taskColumn', value)} />
                    <Text key="task-template" wrap="wrap">{taskTemplate === 'old' ? '✗ task template: yours still has an Agent Log section. The Agent Log links the note now, so new tasks get a section nothing fills.' : `✓ task template: ${taskTemplate === 'yours' ? short(P.taskTemplate) : 'built-in (Outcome, Verify, Journal)'}`}</Text>
                    {taskTemplate === 'old' && <Button key="template-update" label="use the current template" onPress={() => useCurrentTemplate($)} />}
                  </Box>
                  <Box key="setup-jira" flexDirection="column" marginTop={1}>
                    <Text bold>Jira (optional)</Text>
                    <Text dimColor>{`${jiraServer ? `Jira MCP server found: ${jiraServer}.` : 'No Jira MCP server found in this session.'} The pane links notes and drafts prompts; the chat talks to Jira.`}</Text>
                    <Input key="set-jiraProject" label="project or board" value={settings.jiraProject ?? ''} placeholder="PROJ" onSubmit={value => setOption($, 'jiraProject', value)} />
                    <Input key="set-jiraSite" label="site" value={settings.jiraSite ?? ''} placeholder="https://example.atlassian.net" onSubmit={value => setOption($, 'jiraSite', value)} />
                    <Input key="set-jiraServer" label="MCP server" value={settings.jiraServer ?? ''} placeholder="as /mcp lists it; empty looks for one" onSubmit={value => setOption($, 'jiraServer', value)} />
                  </Box>
                  <Box key="setup-pane" flexDirection="column" marginTop={1}>
                    <Text bold>The pane</Text>
                    <Button key="set-openOnStart" plain label={`${mark(settings.openOnStart !== 'no')} open Augi with every new chat`} onPress={() => setOption($, 'openOnStart', settings.openOnStart === 'no' ? 'yes' : 'no')} />
                    <Input key="set-lensPicks" label="lenses to list first" value={settings.lensPicks ?? ''} placeholder="all of them" onSubmit={value => setOption($, 'lensPicks', value)} />
                    <Button key="set-trackUsage" plain label={`${mark(settings.trackUsage !== 'no')} count which controls I use (kept on this computer only)`} onPress={() => setOption($, 'trackUsage', settings.trackUsage === 'no' ? 'yes' : 'no')} />
                    <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                      <Button key="usage-report" label="copy my usage report" onPress={() => usageNow($).then(text => $.ui.copy({ text })).then(() => $.ui.toast('Usage report copied. /augi usage prints it in the chat.'))} />
                      <Button key="usage-clear" plain dimColor label="clear the usage log" onPress={() => $.fs.write(P.usage, '').then(() => $.ui.toast('Usage log cleared.'))} />
                    </Box>
                    {editorPicker}
                    <Button key="start-over" plain dimColor label="← change the notes folder and run setup again" onPress={() => startOver($)} />
                  </Box>
                </Box>
              )}
            </Box>
          </Box>
        )}
        <Button key="refresh" plain dimColor label="↻ refresh" hotkey="r" onPress={() => load($)} />
      </Box>
    )
  })
}
