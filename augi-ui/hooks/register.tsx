import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, RenderChildren } from 'claude-code'

import type { Card, Chat, Inherited, ChatState, JustMade, Lens, Offer, Opened, PadEntry, Picks, Pin, PinKind, SetupItem, SubItem, Swatch, Today, Told, Trail, View } from '../types'

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
  // This week's note, which replaces today's in the pane when set: a path with a date in
  // it, `{{date:YYYY-MM-DD}}`, that may be any day of the week (Monday's when the pane makes it).
  weekNote: string
  // The heading in the week's note whose checkboxes are the week's workstreams; empty is `This week`.
  weekHeading: string
  // The heading each day's logged lines go under in the week's note, with the date in it
  // and its level as `#`s; empty is `### {{date:YYYY-MM-DD}}`, the dated entry the user
  // writes there too.
  dayHeading: string
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
  // `no` keeps the anchor out of the chat's own context; anything else tells the chat.
  tellChat: string
  // The saved prompt that asks a chat's model where its work stands; empty is `chat-state`.
  stateLens: string
  // `yes` asks it by itself after a turn, at most every STATE_EVERY_MS; anything else waits for a press.
  stateAuto: string
  // `no` hides the one-line row above the prompt; anything else shows it.
  band: string
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
  // This week's note setting (a pattern, or empty for none), its workstream heading, and
  // the per-day heading pattern log lines go under.
  week: string
  weekHeading: string
  dayHeading: string
  // Where Obsidian keeps its daily-note folder and name format for this notes folder.
  obsidianDaily: string
  // Obsidian's own settings for this notes folder: where it puts a new note by default.
  obsidianApp: string
  claudeDir: string
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
    week: (options.weekNote ?? '').trim(),
    weekHeading: (options.weekHeading ?? '').trim().replace(/^#+\s*/, '') || 'This week',
    dayHeading: (options.dayHeading ?? '').trim() || '### {{date:YYYY-MM-DD}}',
    obsidianDaily: `${notes}/.obsidian/daily-notes.json`,
    obsidianApp: `${notes}/.obsidian/app.json`,
    claudeDir: `${home}/.claude/projects`,
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
  'card-sub', 'card', 'task',
  'pin-open', 'pin-drop', 'pin-paste',
  'pad-input', 'pad-save', 'pad-drop', 'today-log', 'state-pick', 'week-goal', 'week-anchor',
  'edit', 'new-lens', 'starter', 'vault', 'found', 'editor', 'file-editor',
].sort((one, other) => other.length - one.length)
// Whole keys that only look like one of those.
const USAGE_WHOLE = ['task-new', 'task-cancel', 'task-title-new', 'pad-save-all', 'edit-context']

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
  'anchor-menu', 'card', 'card-sub', 'anchor-clear', 'inherit-keep', 'inherit-change', 'inherit-none', 'open-note', 'anchor-details', 'task',
  'jira-open', 'jira-update', 'jira-unlink', 'jira-link', 'jira-find', 'task-new', 'task-title-new', 'task-cancel',
  'made-open', 'made-board', 'made-define', 'made-anchor', 'made-dismiss',
  'pin-open', 'pin-drop', 'pin-paste',
  'state-update', 'state-auto', 'state-prompt', 'state-edit', 'state-pick',
  'offer-ask', 'offer-link', 'offer-make', 'offer-dismiss',
  'pad-input', 'pad-save', 'pad-drop', 'pad-save-all',
  'today-log', 'today-open', 'today-make', 'pick-today', 'week-goal', 'week-anchor',
  'sync-week',
  'close-out', 'close-yes', 'close-cancel', 'band-open',
  'draft', 'draft-sent',
  'lens', 'edit', 'new-lens', 'all-lenses', 'lens-menu', 'setup-menu',
  'about-me', 'pick-project', 'openaugi', 'edit-context', 'open-board', 'set-openOnStart', 'set-trackUsage', 'usage-report', 'usage-clear', 'start-over', 'refresh',
  'head-chatOpen', 'head-anywhereOpen', 'head-anchorOpen', 'head-stateOpen', 'head-pinsOpen', 'head-padOpen', 'head-youOpen',
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
    ...(lenses.length > 0 ? ['', 'Saved prompts run:', ...lenses.map(count)] : []),
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

// The command that opens a path: a note with the note editor, a web page with whatever
// the system opens pages with (the browser), any other file with the file editor.
export const openArgv = (paths: Paths, path: string): string[] => {
  if (isVaultNote(path, paths.notes)) {
    return editorArgv(paths.editor, paths.editorCommand, path)
  }

  return /\.html?$/i.test(path) ? editorArgv('system', '', path) : editorArgv(paths.fileEditor, '', path)
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
// daily-note settings (`obsidian` is that file's text: folder and name format); else none.
// `none` typed turns it off. Obsidian's defaults apply as it applies them: the name is
// YYYY-MM-DD, and with no folder named the note goes where Obsidian puts any new note
// (`app` is its app settings' text: a set folder, else the vault's root).
export const dailyPattern = (typed: string, obsidian: string, app = ''): string => {
  if (typed.toLowerCase() === 'none') {
    return ''
  }
  if (typed) {
    return typed
  }
  if (!obsidian.trim()) {
    return ''
  }
  const tidy = (value: unknown): string => (typeof value === 'string' ? value.trim().replace(/^\/+|\/+$/g, '') : '')
  try {
    const { folder, format } = JSON.parse(obsidian) as { folder?: unknown; format?: unknown }
    let inFolder = tidy(folder)
    if (!inFolder && app.trim()) {
      try {
        const { newFileLocation, newFileFolderPath } = JSON.parse(app) as { newFileLocation?: unknown; newFileFolderPath?: unknown }
        inFolder = newFileLocation === 'folder' ? tidy(newFileFolderPath) : ''
      } catch {
        inFolder = ''
      }
    }

    return `${inFolder ? `${inFolder}/` : ''}{{date:${typeof format === 'string' && format.trim() ? format.trim() : 'YYYY-MM-DD'}}}.md`
  } catch {
    return ''
  }
}

// The template Obsidian makes a daily note from, as a path in the notes folder, or empty
// when its settings name none.
export const dailyTemplatePath = (obsidian: string, notes: string): string => {
  try {
    const { template } = JSON.parse(obsidian) as { template?: unknown }
    const named = typeof template === 'string' ? template.trim().replace(/^\/+/, '') : ''

    return named ? `${notes}/${named.endsWith('.md') ? named : `${named}.md`}` : ''
  } catch {
    return ''
  }
}

// A template's text for a note made now: Obsidian's own placeholders filled, `{{date}}`
// and `{{time}}` (each with an optional `:FORMAT`) and `{{title}}`. Anything else in it,
// a scripting plugin's code included, is left exactly as written.
export const fillTemplate = (text: string, ms: number, title: string): string => {
  const date = new Date(ms)
  const clock = (format: string): string =>
    format.replace(/HH|mm/g, token => String(token === 'HH' ? date.getHours() : date.getMinutes()).padStart(2, '0'))

  return text
    .replace(/\{\{date(?::([^}]*))?\}\}/g, (_, format: string | undefined) => formatDate(format?.trim() || 'YYYY-MM-DD', ms))
    .replace(/\{\{time(?::([^}]*))?\}\}/g, (_, format: string | undefined) => clock(format?.trim() || 'HH:mm'))
    .replace(/\{\{title\}\}/g, title)
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

// The seven days of the week `ms` falls in, Monday first, each at noon (clear of a clock
// change at midnight).
export const weekDays = (ms: number): number[] => {
  const date = new Date(ms)
  const back = (date.getDay() + 6) % 7

  return [0, 1, 2, 3, 4, 5, 6].map(add => new Date(date.getFullYear(), date.getMonth(), date.getDate() - back + add, 12).getTime())
}

// Where this week's note may be: the pattern filled for each day of the week, Monday
// first, a path that repeats (a pattern with no day in it) listed once. The note may be
// named for any of its days; the pane makes Monday's.
export const weekNotePaths = (pattern: string, ms: number, notes: string, home: string): string[] =>
  pattern.trim() ? [...new Set(weekDays(ms).map(day => dailyNotePath(pattern, day, notes, home)))] : []

// The lines right under `heading` (any level, any case): from the line after it to the
// next heading of any level, so a day's heading below never reads as part of the week's
// list. Null when the note has no such heading.
const sectionRange = (lines: string[], heading: string): { start: number; end: number } | null => {
  const isHeading = (line: string): boolean => /^#{1,6}\s/.test(line)
  const at = lines.findIndex(line => isHeading(line) && line.replace(/^#+\s*/, '').trim().toLowerCase() === heading.toLowerCase())
  if (at < 0) {
    return null
  }
  let end = at + 1
  while (end < lines.length && !isHeading(lines[end] ?? '')) {
    end += 1
  }

  return { start: at + 1, end }
}

// One of the week's workstreams: a top-level checkbox under the week heading. `note` is
// the note it links, when it links one; `line` is the line as written.
export type Goal = { text: string; note: string | null; done: boolean; line: string }

// The week's workstreams: the top-level checkboxes under `heading`, in order. An indented
// checkbox is a detail of the one above it, not a workstream of its own.
export const weekGoals = (text: string, heading: string): Goal[] => {
  const lines = text.split('\n')
  const range = sectionRange(lines, heading)
  if (!range) {
    return []
  }

  return lines.slice(range.start, range.end).flatMap(line => {
    const found = line.match(/^[-*] \[([ xX])\] (.*\S.*)$/)

    return found ? [{ text: plain(found[2] ?? ''), note: linkName(found[2] ?? ''), done: found[1] !== ' ', line }] : []
  })
}

// A new week's note: the workstream heading with last week's open workstreams carried
// over (each with its indented lines), or one empty checkbox when there are none.
export const newWeekNote = (lastWeek: string, heading: string): string => {
  const lines = lastWeek.split('\n')
  const range = sectionRange(lines, heading)
  const carried: string[] = []
  let isCarrying = false
  for (const line of range ? lines.slice(range.start, range.end) : []) {
    if (/^[-*] \[[ xX]\] /.test(line)) {
      isCarrying = /^[-*] \[ \] \S/.test(line)
    } else if (!/^\s+\S/.test(line)) {
      isCarrying = false
    }
    if (isCarrying) {
      carried.push(line)
    }
  }

  return `# ${heading}\n\n${(carried.length > 0 ? carried : ['- [ ] ']).join('\n')}\n`
}

// The first few open checkboxes of a note, as text: what today's note says is still to do.
export const openBoxes = (text: string, limit = 5): string[] =>
  text.split('\n').map(line => line.match(/^\s*[-*] \[ \] (.*\S.*)$/)?.[1]).filter((one): one is string => Boolean(one)).map(plain).slice(0, limit)

// Adds a line to today's note: `- HH:MM text` as the last line under the heading named
// `heading` (any level, any case; made at the end when the note has none, at `level`), or
// as the note's last line when no heading is named. With `isPrefix`, a heading that starts
// with the name and goes on after a space counts too (`### 2022-01-05 hinges` for
// `2022-01-05`): the dated entry the user wrote that day. Every other line stays as it was.
export const addDailyLine = (text: string, heading: string, one: PadEntry, level = 2, isPrefix = false): string => {
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
  const wanted = heading.toLowerCase()
  const isIt = (title: string): boolean => title.toLowerCase() === wanted || (isPrefix && title.toLowerCase().startsWith(`${wanted} `))
  const at = lines.findIndex((_, index) => isIt(headingAt(index)?.title ?? '\n'))
  if (at < 0) {
    return atEnd(text, [...(text.trim() ? [''] : []), `${'#'.repeat(Math.min(6, Math.max(1, level)))} ${heading}`, ...entry])
  }
  const found = headingAt(at)?.level ?? 1
  let end = at + 1
  while (end < lines.length && (headingAt(end)?.level ?? 7) > found) {
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
  { field: 'lensFolder', label: 'saved prompts folder', path: paths.lenses, ok: found.has(paths.lenses), starter: true },
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
  { field: 'lensFolder', title: 'Your saved prompts folder', why: 'A folder of saved prompts, one note each. They show under Saved prompts.', starter: 'OpenAugi/AGENT/lenses' },
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

// No `p`: it opens the anchor's note while the lens menu is open.
const EDIT_KEYS = 'qwtyui'
const PANE = 'augi'
// How often an open pane checks whether another chat changed its anchor's files.
const SYNC_MS = 10000
// Augi's face (an otter, for old times' sake) on surfaces that draw Svg (desktop, editor, phone); the terminal skips it.
const AUGI_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="14" cy="16" r="7" fill="#8a5a3b"/><circle cx="50" cy="16" r="7" fill="#8a5a3b"/><circle cx="14" cy="16" r="3.5" fill="#5e3b25"/><circle cx="50" cy="16" r="3.5" fill="#5e3b25"/><circle cx="32" cy="34" r="24" fill="#a06a45"/><ellipse cx="32" cy="43" rx="15" ry="11" fill="#e8cfb0"/><circle cx="23" cy="30" r="3.2" fill="#2b1a10"/><circle cx="41" cy="30" r="3.2" fill="#2b1a10"/><circle cx="24" cy="29" r="1" fill="#fff"/><circle cx="42" cy="29" r="1" fill="#fff"/><ellipse cx="32" cy="38" rx="4.5" ry="3" fill="#2b1a10"/><path d="M32 41v3M28 46q4 3 8 0" stroke="#2b1a10" stroke-width="1.5" fill="none" stroke-linecap="round"/><path d="M20 41l-9-2M20 44l-9 1M44 41l9-2M44 44l9 1" stroke="#5e3b25" stroke-width="1" stroke-linecap="round"/></svg>'
const DRAFT_HEAD = 'Context for this chat:'
// Each chat session's anchor, as [session id, note] pairs, oldest first. The anchor
// belongs to the session: a new one opens with none; a reload or a resume keeps it.
const ANCHORS_KEY = 'anchors'
const ANCHORS_KEPT = 200
// The title a chat gets in ## Chats when anchoring records it, until its first prompt names it.
export const CHAT_PLACEHOLDER = 'chat'
// The setup steps someone chose to skip; /augi setup clears it.
const SKIPPED_KEY = 'skipped'
// Pins that go into every chat's draft, whatever its anchor: kept in the store, newest last.
export const EMPTY_TRAIL: Trail = { chats: [], pins: [], states: {} }
// No daily note: the Today card is not drawn.
const NO_TODAY: Today = { pattern: '', path: '', day: '', exists: false, todo: [], mtime: 0, week: false, goals: [], entries: 0 }


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
    pinPaths: {},
    lenses: [],
    loadedAt: 0,
  } as Swatch,
  { shape: 'swatch-23' },
)
// The two sections and the cards inside them start open; Lenses and Setup start folded.
const CARDS_OPEN = { chatOpen: true, anywhereOpen: true, anchorOpen: true, stateOpen: true, pinsOpen: true, padOpen: true, youOpen: true }
const picks = atom(
  { plugin: 'augi-ui', key: 'picks' } as const,
  { aboutMe: true, openAugi: true, project: true, today: true } as Picks,
  { shape: 'picks-4' },
)
const view = atom(
  { plugin: 'augi-ui', key: 'view' } as const,
  { ...CARDS_OPEN, anchorsOpen: false, statePickOpen: false, lensesOpen: false, setupOpen: false, inOpen: true, detailsOpen: false, taskMode: '', logged: 0, closing: false } as View,
  { shape: 'view-18' },
)
// Where this chat's work stands, as its own model last said when asked (updateState): for
// the session `session`, the answer and when it came (`at`, in ms, paces the automatic
// asks), `busy` while it is being asked, and why the last ask got no answer.
const NO_STATE: ChatState = { session: '', text: '', when: '', at: 0, busy: false, error: '' }
const chatState = atom({ plugin: 'augi-ui', key: 'chatState' } as const, NO_STATE, { shape: 'chatState-1' })


// The task the pane offers a chat that has no anchor (suggestTask): for the session
// `session`, a note on the board to link (`note`) or, with none that fits, a title for a
// new one (`title`, with the Jira key its branch names), and why. `done` once the user
// took it or turned it down.
const NO_OFFER: Offer = { session: '', note: '', title: '', jira: '', why: '', busy: false, error: '', done: false }
const offer = atom({ plugin: 'augi-ui', key: 'offer' } as const, NO_OFFER, { shape: 'offer-1' })

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

// What this chat's model was last told about its anchor, so it is told once and again only
// when that changes. `again` is set by a compaction, which may have summarized it away.
const NOT_TOLD: Told = { session: '', note: '', today: '', again: false }
const told = atom({ plugin: 'augi-ui', key: 'told' } as const, NOT_TOLD, { shape: 'told-1' })

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
  '# Agent Log',
  '',
].join('\n')

// Which task template a new task is made from: the built-in one (the notes folder has
// none) or the notes folder's own.
export const taskTemplateState = (text: string): 'built-in' | 'yours' => (text.trim() ? 'yours' : 'built-in')

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
export const jiraUpdateDraft = (notePath: string, note: string, key: string, ask: JiraAsk): string =>
  [
    DRAFT_HEAD,
    `- Anchor note: ${notePath ? `"${notePath}"` : `[[${note}]] (find it in "${P.notes}")`} (${note}): read its Outcome, Verify, Journal and Agent Log sections.`,
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

// ---- Board sync: the board brought in line with the week's workstreams, on a press ----

// What syncing the board would do, worked out from the week's workstreams and the board's
// open cards. `add`: open linked workstreams with no open card or linked sub-item. `tick`:
// done linked workstreams whose card is still open. `stale`: open cards that are not this
// week's (nor hold one as a sub-item). `unlinked`: open workstreams that link no note yet.
export type BoardSync = { add: string[]; tick: string[]; stale: string[]; unlinked: Goal[] }

export const boardSync = (goals: Goal[], cards: Card[]): BoardSync => {
  const onBoard = new Set(cards.flatMap(card => [card.note, ...card.tasks.map(task => task.note).filter((note): note is string => Boolean(note))]))
  const linked = goals.filter(goal => goal.note)
  const weekNotes = new Set(linked.map(goal => goal.note ?? ''))
  const isTopCard = new Set(cards.map(card => card.note))

  return {
    add: [...new Set(linked.filter(goal => !goal.done && !onBoard.has(goal.note ?? '')).map(goal => goal.note ?? ''))],
    tick: [...new Set(linked.filter(goal => goal.done && isTopCard.has(goal.note ?? '') && !goals.some(one => !one.done && one.note === goal.note)).map(goal => goal.note ?? ''))],
    stale: cards
      .filter(card => !weekNotes.has(card.note) && !card.tasks.some(task => task.note && weekNotes.has(task.note)))
      .map(card => card.note),
    unlinked: goals.filter(goal => !goal.done && !goal.note),
  }
}

// Where the board's cards end: at the archive (`***`), else at the Kanban settings block.
const cardsEnd = (lines: string[]): number => {
  const at = lines.findIndex(line => line === '***' || /^%% kanban:settings/.test(line))

  return at >= 0 ? at : lines.length
}

// The board with the note's open top-level card ticked (`- [x]`), where it is; with no
// such card, its open linked sub-item under a card. Every other line as it was; a note
// with neither changes nothing.
export const tickCard = (board: string, note: string): string => {
  const lines = board.split('\n')
  const end = cardsEnd(lines)
  const open = (pattern: RegExp): number => lines.findIndex((line, index) => index < end && pattern.test(line) && linkName(line) === note)
  const top = open(/^- \[ \] /)
  const at = top >= 0 ? top : open(/^\t- \[ \] /)
  if (at < 0) {
    return board
  }
  lines[at] = lines[at]?.replace(/- \[ \] /, '- [x] ') ?? ''

  return lines.join('\n')
}

// The week's note with one workstream's line linking its note: `- [ ] text` becomes
// `- [ ] [[name]]`, the text kept after a `·` when the note's name is not the same words
// (case aside). Only that line changes; a line no longer there changes nothing.
export const linkGoal = (week: string, line: string, name: string): string => {
  const lines = week.split('\n')
  const at = lines.indexOf(line)
  const found = line.match(/^([-*] \[[ xX]\] )(.*\S)\s*$/)
  if (at < 0 || !found) {
    return week
  }
  const text = found[2] ?? ''
  lines[at] = `${found[1]}[[${name}]]${text.toLowerCase() === name.toLowerCase() ? '' : ` · ${text}`}`

  return lines.join('\n')
}

// The week's note with the note's open workstream line(s) ticked, under `heading` only.
export const tickGoal = (week: string, heading: string, note: string): string => {
  const lines = week.split('\n')
  const range = sectionRange(lines, heading)
  if (!range) {
    return week
  }
  for (let at = range.start; at < range.end; at += 1) {
    if (/^[-*] \[ \] /.test(lines[at] ?? '') && linkName(lines[at] ?? '') === note) {
      lines[at] = (lines[at] ?? '').replace(/\[ \]/, '[x]')
    }
  }

  return lines.join('\n')
}

// Open checkboxes in the week's note outside its workstream list, as written, each with
// the heading it sits under: priorities the user wrote while journaling in a day's entry.
export const entryBoxes = (week: string, heading: string): { under: string; line: string }[] => {
  const lines = week.split('\n')
  const range = sectionRange(lines, heading)
  let under = ''

  return lines.flatMap((line, at) => {
    if (/^#{1,6}\s/.test(line)) {
      under = line.replace(/^#+\s*/, '').trim()

      return []
    }
    const isListed = range !== null && at >= range.start && at < range.end

    return !isListed && /^\s*[-*] \[ \] \S/.test(line) ? [{ under, line: line.trim() }] : []
  })
}

// What one press of `⇅ sync this week` leaves for the chat: the parts that need judgement.
export type WeekLeft = {
  // The week's note, and the edits that need no judgement, each one row (`sure`).
  path: string
  sure: string[]
  // Workstreams that link no note; checkboxes in the day entries; open cards not this week's.
  unlinked: Goal[]
  entries: { under: string; line: string }[]
  stale: string[]
  // Linked open workstreams whose note has no Jira key, with the note's path; Jira's names.
  noKey: { note: string; path: string }[]
  jira: JiraAsk | null
  // Where a new card goes (a column's heading) and how a new task note starts.
  column: string
  template: string
}

// The prompt `⇅ sync this week` leaves in the box, unsent: every edit the week needs,
// the sure ones as they are and the rest for the chat to resolve, in one numbered table.
// Nothing is changed until the user answers; then exactly what was agreed is written.
export const weekSyncDraft = (left: WeekLeft): string => {
  const lines = [
    DRAFT_HEAD,
    `- This week's note: "${left.path}". The checkboxes right under its "${P.weekHeading}" heading are my workstreams; below it I journal in a dated entry per day.`,
    `- My board: "${P.board}" (Kanban; cards above "***", the archive below it).`,
  ]
  lines.push('Rows for the table, one each:')
  if (left.sure.length > 0) {
    lines.push('- Sure, worked out by the pane: propose them as they are.')
    lines.push(...left.sure.map(one => `  - ${one}`))
  }
  if (left.unlinked.length > 0) {
    lines.push('- Workstreams with no note yet. Find the note each one is: a card on the board first, then a search of my notes (with the OpenAugi tools if you have them). If none fits, propose a new task note with a short name.')
    lines.push(...left.unlinked.map(goal => `  - ${goal.line}`))
  }
  if (left.entries.length > 0) {
    lines.push(`- Checkboxes I wrote in my day entries. Say for each whether it is a workstream for the week (a new line under "${P.weekHeading}", or part of one already there) or just a to-do for that day, which stays where it is.`)
    lines.push(...left.entries.map(one => `  - ${one.under ? `under "${one.under}": ` : ''}${one.line}`))
  }
  if (left.stale.length > 0) {
    lines.push('- Open cards on the board that are not this week\'s. Propose done (tick it where it is), archive (move it with its indented lines to the end of the archive, ticked) or keep (leave it).')
    lines.push(...left.stale.map(note => `  - [[${note}]]`))
  }
  if (left.jira && left.noKey.length > 0) {
    lines.push(`- Workstreams whose note has no Jira key. Search Jira${jiraTools(left.jira.project, left.jira.server)} for the issue each most likely is, up to three candidates (KEY · summary · status), or none.`)
    lines.push(...left.noKey.map(one => `  - [[${one.note}]] ${one.path ? `"${one.path}"` : ''}`.trimEnd()))
  }
  lines.push(
    'Show it all as one numbered table: item, what you propose, why in a few words. Change nothing until I answer `all`, row numbers, or `none`.',
    `Then do exactly what I agreed. A workstream line becomes "- [ ] [[Note]]", my words kept after " · " when they differ from the name. A new note goes in "${P.tasks}" from ${left.template}, never over a note that exists, and gets a card "- [ ] [[Note]]" at the end of the "${left.column}" column. ${left.jira ? 'A Jira key goes in its note as `jira: KEY` in the frontmatter, or a `Jira: KEY` line under its `Up:` line when it has no frontmatter. Change nothing in Jira. ' : ''}Leave every other line of my notes and the board exactly as it is.`,
    'Instruction: sync this week',
  )

  return lines.join('\n')
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
export const newTaskDraft = (notePath: string, name: string, jira: JiraAsk | null = null): string =>
  [
    DRAFT_HEAD,
    `- A task note was just made from this chat: "${notePath}".`,
    "- From what this chat has done, fill in what goes under the note's Outcome and Verify headings (the outcome as a state of the world, and how we will know it is reached). Add no headings and no frontmatter. This once you may write that note: I'm asking.",
    '- Put the context worth keeping (decisions, files, links) under the note\'s "# Agent Log" heading, at its end, one line each on why it matters.',
    ...(jira ? jiraFindSteps('this task', jira.project, jira.server) : []),
    'Reply in a few lines with what you wrote.',
    `Instruction: fill in the new task ${name}`,
  ].join('\n')

// An anchor's Agent Log used to be a file of its own, 1:1 by name. These say where it
// was, for moving what it holds into the note (`withOldLog`).
const safe = (note: string): string => note.replace(/[\\/:]/g, '-')
export const logName = (note: string): string => `Agent Log - ${safe(note)}`
export const trailPath = (note: string): string => `${P.trails}/${logName(note)}.md`
// Where v0.6 and earlier kept it, for moving it over.
export const oldTrailPath = (note: string): string => `${P.trails}/Trail - ${safe(note)}.md`

// What the intro says of ## Context since pins became bullets. A log that carries it is in
// today's form; one that doesn't still ticks what it pins (see `tidyTrail`).
const PINS_ARE_BULLETS = 'Context lines are `- [[note]] · why it matters`'
// Whether a log is from before that: every pin line in it is a pin, whatever its box.
const isOldLog = (text: string): boolean => !text.includes(PINS_ARE_BULLETS)

// The first body line of a log file linked the human note.
const logIntro = (note: string): string =>
  `Human note: [[${note}]]. ${PINS_ARE_BULLETS}: each is pinned and goes into the next draft; one ticked off (\`- [x]\`) was unpinned. Each chat is one line: when it started, its goal, its id; one that is finished stays listed, marked \`(done)\`.`
// The intros earlier versions wrote, replaced with the current one when the pane writes the log.
const oldLogIntros = (note: string): string[] => [
  `Human note: [[${note}]]. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Each chat is one line: when it started, its goal, its id; one that is finished stays listed, marked \`(done)\`.`,
  `Project: [[${note}]], embedded there under # Agent Log. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Hidden chats stay listed, marked \`(hidden)\`.`,
  `Human note: [[${note}]]. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Hidden chats stay listed, marked \`(hidden)\`.`,
]

// A log file in its last shape, as the step before moving it into the note: no title H1
// (Agent Log or Trail), no seen box, the human-note link first, its pins as bullets (in a
// log from before bullets, ticked or not, each was a pin).
export const tidyTrail = (text: string, note: string): string => {
  const dropped = new Set([`# ${logName(note)}`, `# Trail - ${safe(note)}`, '- [ ] seen', '- [x] seen'])
  const isOld = isOldLog(text)
  const lines: string[] = []
  let gap = false
  let section = ''
  for (const each of text.split('\n')) {
    section = each.match(/^## (.+)$/)?.[1]?.trim() ?? section
    const pin = isOld && section === 'Context' ? parsePin(each, true) : null
    const line = pin ? pinLine(pin) : each
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

// A chat's line in the Chats callout. `(hidden)` is what a finished chat was marked before `(done)`.
const CHAT_LINE = /^- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) · (.*) · `([^`]+)`( \((?:done|hidden)\))?$/

// A pin's line: `- [[Note]] · label` for a note, `- kind: \`target\` · label` for the
// rest. A pin is a plain bullet (an open box reads the same); `- [x]` is one that was
// unpinned: kept on the log, out of the pane and the drafts.
const PIN_LINE = /^- (?:\[([ x])\] )?(?:\[\[([^\]|]+)(?:\|[^\]]*)?\]\]|(file|session|artifact|url): `([^`]+)`)(?: · (.*))?\s*$/

// `isOld` reads a line from a log file of before bullets, where a tick meant pinned.
export const parsePin = (line: string, isOld = false): Pin | null => {
  const found = line.match(PIN_LINE)
  if (!found) {
    return null
  }
  const on = isOld || found[1] !== 'x'
  if (found[2]) {
    return { on, kind: 'note', target: found[2].trim(), label: (found[5] ?? '').trim() || found[2].trim() }
  }

  return { on, kind: found[3] as PinKind, target: found[4], label: (found[5] ?? '').trim() || found[4] }
}

const clean = (text: string): string => text.replace(/[`\n]/g, ' ').replace(/\s+/g, ' ').trim()

export const pinLine = (pin: Pin): string => {
  const box = pin.on ? '-' : '- [x]'
  if (pin.kind === 'note') {
    return pin.label && pin.label !== pin.target ? `${box} [[${pin.target}]] · ${clean(pin.label)}` : `${box} [[${pin.target}]]`
  }
  const label = clean(pin.label)

  return label && label !== pin.target ? `${box} ${pin.kind}: \`${pin.target}\` · ${label}` : `${box} ${pin.kind}: \`${pin.target}\``
}

const samePin = (a: Pin, b: Pin): boolean => a.kind === b.kind && a.target === b.target

// ── The Agent Log ──
// One section at the bottom of the anchor's own note, under `# Agent Log`: what is pinned
// to the work and whatever agents append, then the chats on it in a callout that is folded
// until opened. Everything above the heading is the user's. (Functions and state keep the
// log's first name, trail.)
const LOG_HEAD = '# Agent Log'
const CHATS_HEAD = '> [!abstract]- Chats'
const isLogHead = (line: string): boolean => /^# Agent Log\s*$/i.test(line)
// For a day the section opened with one State callout for the whole task. A chat's state
// is kept under its own entry in the chat list now; a callout still there is dropped.
const OLD_STATE_HEAD = /^> \[!\w+\][-+]? State(?: · .*)?\s*$/i
export const STATE_LINES = 8
const isChatsHead = (line: string): boolean => /^> \[!\w+\][-+]? Chats\s*$/i.test(line)
// What the section held while the log was a file of its own: a callout embedding that file.
const isLogEmbed = (line: string): boolean => /^> \[!\w+\][-+]? Agent Log\s*$/i.test(line) || /^> !\[\[Agent Log - [^\]]*\]\]\s*$/.test(line)

const trimBlank = (lines: string[]): string[] => {
  let from = 0
  let to = lines.length
  while (from < to && (lines[from] ?? '').trim() === '') {
    from += 1
  }
  while (to > from && (lines[to - 1] ?? '').trim() === '') {
    to -= 1
  }

  return lines.slice(from, to)
}

// A note in five parts: the lines above the section's heading; the section's own lines
// (`body`: pins and whatever was saved there, blank edges trimmed); the lines of its Chats
// callout without their `> ` (a chat's line, then indented under it where that chat last
// stood); the rows of a `## Recall` table (`recall`: kept as written, from the retired Recall card);
// and what follows the section (a later level-1 heading and on). `has` is whether the note
// has the section; the last `# Agent Log` is the one.
export type LogParts = { before: string[]; body: string[]; chats: string[]; recall: string[]; after: string[]; has: boolean }
export const RECALL_HEAD = '## Recall'
const isRecallHead = (line: string): boolean => /^## Recall\s*$/i.test(line)

export const splitLog = (text: string): LogParts => {
  const lines = text.split('\n')
  let at = -1
  lines.forEach((line, index) => {
    if (isLogHead(line)) {
      at = index
    }
  })
  if (at < 0) {
    return { before: lines, body: [], chats: [], recall: [], after: [], has: false }
  }
  let end = at + 1
  while (end < lines.length && !/^# \S/.test(lines[end] ?? '')) {
    end += 1
  }
  const body: string[] = []
  const chats: string[] = []
  const recall: string[] = []
  // Which block the line is in: the Chats list, an old State block (dropped), or the Recall
  // table. The blank lines around one only set it apart: they are not the body's. Under
  // `## Recall` only the table's rows are the table's: any other line there is the body's.
  let inside: '' | 'chats' | 'old' | 'recall' = ''
  let isJustPast = false
  for (const line of lines.slice(at + 1, end)) {
    const isOld = OLD_STATE_HEAD.test(line)
    if (isOld || isChatsHead(line)) {
      inside = isOld ? 'old' : 'chats'
      while (body.length > 0 && (body[body.length - 1] ?? '').trim() === '') {
        body.pop()
      }
    } else if (isRecallHead(line)) {
      inside = 'recall'
      while (body.length > 0 && (body[body.length - 1] ?? '').trim() === '') {
        body.pop()
      }
    } else if (inside === 'recall' && (line.trim().startsWith('|') || line.trim() === '')) {
      if (line.trim()) {
        recall.push(line.trim())
      }
    } else if (inside && inside !== 'recall' && line.startsWith('>')) {
      if (inside === 'chats') {
        chats.push(line.replace(/^> ?/, ''))
      }
    } else {
      isJustPast = (inside !== '' || isJustPast) && line.trim() === ''
      inside = ''
      if (!isJustPast && !isLogEmbed(line)) {
        body.push(line)
      }
    }
  }

  return { before: lines.slice(0, at), body: trimBlank(body), chats: chats.filter(line => line.trim()), recall, after: lines.slice(end), has: true }
}

// The note again from its parts. A note that had no section gets it at its end, one blank
// line down; what is above an existing section, and after it, stays byte for byte. The
// Chats callout comes after everything else but the Recall table, which is last, so a line
// appended after either moves up beside the others.
const joinLog = ({ before, body, chats, recall, after, has }: LogParts): string => {
  let last = before.length
  while (!has && last > 0 && (before[last - 1] ?? '').trim() === '') {
    last -= 1
  }
  const top = has ? before : last > 0 ? [...before.slice(0, last), ''] : []
  const blocks = [body, chats.length > 0 ? [CHATS_HEAD, ...chats.map(line => `> ${line}`)] : [], recall.length > 0 ? [RECALL_HEAD, ...recall] : []].filter(one => one.length > 0)
  const section = [LOG_HEAD, ...blocks.flatMap(one => ['', ...one])]

  return [...top, ...section, '', ...after].join('\n')
}

// What a note's Agent Log section says: the chats on the anchor, where each last stood
// (by chat id), and its pins.
export const parseTrail = (text: string): Trail => {
  const { body, chats } = splitLog(text)
  const states: Trail['states'] = {}
  let id = ''
  for (const line of chats) {
    const chat = line.match(CHAT_LINE)
    const under = chat ? null : line.match(/^\s+- (.*)$/)
    if (chat) {
      id = chat[3] ?? ''
    } else if (under && id) {
      const when = under[1]?.match(/^as of (.+)$/)?.[1]
      const was = states[id] ?? { when: '', text: '' }
      states[id] = when ? { ...was, when: when.trim() } : { ...was, text: was.text ? `${was.text}\n${under[1] ?? ''}` : (under[1] ?? '') }
    }
  }

  return {
    states,
    chats: chats.flatMap(line => {
      const found = line.match(CHAT_LINE)

      return found ? [{ when: found[1] ?? '', title: found[2] ?? '', id: found[3] ?? '', done: Boolean(found[4]) }] : []
    }),
    pins: body.flatMap(line => {
      const pin = parsePin(line)

      return pin ? [pin] : []
    }),
  }
}

// A model's answer as plain lines: no quote or list marks of its own, no blank lines, at
// most STATE_LINES.
export const stateLines = (reply: string): string[] =>
  reply.split('\n').map(line => line.replace(/^\s*(?:>+\s?|[-*]\s+)/, '').trim()).filter(Boolean).slice(0, STATE_LINES)

// Where one chat last stood, kept under that chat's own entry in the Chats list: when it
// was said, then the lines, replacing what was under the entry before. Each chat has its
// own, so two chats on one task do not write over each other; an empty answer takes it
// away. A chat that is not in the list changes nothing.
export const setChatState = (text: string, id: string, when: string, reply: string): string => {
  const parts = splitLog(text)
  const at = parts.chats.findIndex(line => line.match(CHAT_LINE)?.[3] === id)
  if (at < 0) {
    return text
  }
  let end = at + 1
  while (end < parts.chats.length && /^\s/.test(parts.chats[end] ?? '')) {
    end += 1
  }
  const lines = stateLines(reply)
  const under = lines.length > 0 ? [`  - as of ${when}`, ...lines.map(line => `  - ${line}`)] : []

  return joinLog({ ...parts, chats: [...parts.chats.slice(0, at + 1), ...under, ...parts.chats.slice(end)] })
}

// The saved prompt asked for a chat's state when the lens folder has none of that name.
export const STATE_LENS = 'chat-state'
export const STATE_PROMPT = [
  'Say where this chat stands, for me coming back to it after working in other chats. Go by the last few turns, not the whole history. Answer with exactly these four lines, each one short, and nothing before or after them:',
  '',
  'Doing: what this chat is working on right now',
  'Last: what the last few turns did or found, with the result if one was checked',
  'Next: the one next step',
  'Waiting on me: yes and what for, or no',
  '',
  'Say only what the conversation shows. Mark anything that is a guess "(unverified)".',
].join('\n')

// The lens file the pane writes for that prompt, so it can be edited like any saved prompt.
export const stateLensFile = (name: string): string =>
  [
    '---',
    'kind: engine',
    `name: ${name}`,
    'description: >-',
    '  "Where does this chat stand?" — what it is doing, what the last few turns did, the next step and whether it is waiting on the user, in four lines.',
    'scope: >-',
    '  This conversation and nothing else: no files are read and no search is run.',
    'trigger: on-demand',
    'target: >-',
    '  note — under this chat\'s own entry in the folded Chats list of the anchor note\'s Agent Log section, replaced each time; with no anchor the answer stays in the pane.',
    '---',
    `# ${name}`,
    '',
    STATE_PROMPT,
    '',
  ].join('\n')

// What a lens file asks: everything under its frontmatter and its title line.
export const lensBody = (text: string): string => {
  const lines = text.split('\n')
  const close = lines[0] === '---' ? lines.indexOf('---', 1) : -1
  const body = lines.slice(close + 1)
  while (body.length > 0 && ((body[0] ?? '').trim() === '' || /^# \S/.test(body[0] ?? ''))) {
    body.shift()
  }

  return body.join('\n').trim()
}

// The side question a chat's model is asked: the lens's words, held to this conversation.
export const stateAsk = (body: string): string =>
  ['Augi side question. Answer it from this conversation alone, in plain text, with no preamble.', '', body.trim() || STATE_PROMPT].join('\n')

// Every note a chat could be anchored on: the board's cards and their linked sub-items, once each.
export const offerCandidates = (cards: Card[]): string[] => {
  const names: string[] = []
  for (const card of cards) {
    for (const name of [card.note, ...card.tasks.map(task => task.note ?? '')]) {
      if (name && !names.includes(name)) {
        names.push(name)
      }
    }
  }

  return names
}

// The first Jira key in a branch name or a line of text (`feature/proj-123-fix` names PROJ-123), or empty.
export const jiraKeyIn = (text: string): string => text.toUpperCase().match(/(?:^|[^A-Z0-9])([A-Z][A-Z0-9]+-\d+)(?![A-Z0-9])/)?.[1] ?? ''

// The side question that matches a chat to a task: the model picks one name from the
// list or says none fits and titles a new one. `hint` is what the pane knows already.
export const offerAsk = (candidates: string[], hint = ''): string =>
  [
    'Augi side question. Answer it from this conversation alone, in plain text, with no preamble.',
    '',
    'Which of my tasks is this conversation working on? Pick one only if the work here is plainly part of it. Answer with exactly these three lines:',
    '',
    'Task: one name from the list below, copied exactly, or NONE',
    'New: if NONE, a title for a new task in at most eight words; otherwise leave this empty',
    'Why: one short line',
    ...(hint ? ['', hint] : []),
    '',
    'My tasks:',
    ...(candidates.length > 0 ? candidates.map(name => `- ${name}`) : ['(none yet)']),
  ].join('\n')

// The model's answer to that question: a note only when it names one on the list, else
// the title it gave for a new task (never a path, never more than a line).
export const parseOffer = (reply: string, candidates: string[]): { note: string; title: string; why: string } => {
  const said = (label: string): string =>
    (reply.match(new RegExp(`^\\s*${label}:[ \\t]*(.*)$`, 'im'))?.[1] ?? '').replace(/^\[\[|\]\]$/g, '').replace(/^["'“`]+|["'”`.]+$/g, '').trim()
  const named = said('Task').toLowerCase()
  const note = candidates.find(name => name.toLowerCase() === named) ?? ''
  const title = note || /^none$/i.test(said('New')) ? '' : said('New').replace(/[\\/:]/g, ' ').replace(/\s+/g, ' ').slice(0, 80).trim()

  return { note, title, why: said('Why').slice(0, 160) }
}

// Pins something: a line at the end of the section, which is made when the note has none.
// One already there stays where it is, pinned again if it had been unpinned.
export const addPin = (text: string, pin: Pin): string => {
  const parts = splitLog(text)
  const at = parts.body.findIndex(line => {
    const one = parsePin(line)

    return one !== null && samePin(one, pin)
  })
  const had = at >= 0 ? parsePin(parts.body[at] ?? '') : null
  if (had?.on) {
    return text
  }
  const body = had ? parts.body.map((line, index) => (index === at ? pinLine({ ...had, on: true }) : line)) : [...parts.body, pinLine(pin)]

  return joinLog({ ...parts, body })
}

// Unpins: the line stays where it is, ticked off, as a record of what was once context.
export const removePin = (text: string, pin: Pin): string => {
  const parts = splitLog(text)
  const body = parts.body.map(line => {
    const one = parsePin(line)

    return one?.on && samePin(one, pin) ? pinLine({ ...one, on: false }) : line
  })

  return body.some((line, index) => line !== parts.body[index]) ? joinLog({ ...parts, body }) : text
}

// Files a chat at the top of the Chats callout, newest first; one already listed stays as it is.
export const addChat = (text: string, chat: Chat): string => {
  const parts = splitLog(text)
  if (parts.chats.some(line => line.match(CHAT_LINE)?.[3] === chat.id)) {
    return text
  }

  return joinLog({ ...parts, chats: [chatLine(chat), ...parts.chats] })
}

// Marks a chat finished on its log line, `(done)`, as an agent or the user would. A chat
// not listed, or already done, changes nothing.
export const markChatDone = (text: string, id: string): string => {
  const parts = splitLog(text)
  const chats = parts.chats.map(line => {
    const found = line.match(CHAT_LINE)

    return found && found[3] === id && !found[4] ? chatLine({ when: found[1] ?? '', title: found[2] ?? '', id, done: true }) : line
  })

  return chats.some((line, index) => line !== parts.chats[index]) ? joinLog({ ...parts, chats }) : text
}

// Names a chat recorded with the placeholder title; a real title stays as it is.
export const retitleChat = (text: string, id: string, title: string): string => {
  const parts = splitLog(text)
  const chats = parts.chats.map(line => {
    const found = line.match(CHAT_LINE)

    return found && found[3] === id && (found[2] ?? '').trim() === CHAT_PLACEHOLDER
      ? chatLine({ when: found[1] ?? '', title, id, done: Boolean(found[4]) })
      : line
  })

  return chats.some((line, index) => line !== parts.chats[index]) ? joinLog({ ...parts, chats }) : text
}

// The note with what its old log file held: pins and kept lines from ## Context, the
// questions as they were (a dated checkbox is no pin), each saved note marked `saved:` so
// it is a record and not a pin, and the chats, after any the note already lists. A line
// the section already has is not added twice, so moving the same file again changes nothing.
export const withOldLog = (text: string, old: string, note: string): string => {
  const parts = splitLog(text)
  const body = [...parts.body]
  const chats = [...parts.chats]
  let section = ''
  for (const line of tidyTrail(old, note).split('\n')) {
    const heading = line.match(/^## (.+)$/)?.[1]
    if (heading) {
      section = heading.trim()
    } else if (section === 'Chats') {
      if (CHAT_LINE.test(line) && !chats.includes(line)) {
        chats.push(line)
      }
    } else if (line.trim() && ['Context', 'Questions', 'Saved'].includes(section)) {
      const kept = section === 'Saved' && line.startsWith('- ') ? `- saved: ${line.slice(2)}` : line
      if (!body.includes(kept)) {
        body.push(kept)
      }
    }
  }

  return joinLog({ ...parts, body, chats })
}

// A log file whose content went into its note says so near its top and is left alone after.
const MOVED = 'Moved into [['
export const isMovedLog = (text: string): boolean => text.split('\n').some(line => line.startsWith(MOVED))

export const markMoved = (text: string, note: string, day: string): string => {
  const lines = text.split('\n')
  const said = `${MOVED}${note}]] on ${day}: that note's # Agent Log section is the log now. Nothing reads or writes this file any more.`
  const close = lines[0] === '---' ? lines.indexOf('---', 1) : -1
  lines.splice(close + 1, 0, ...(close < 0 ? [said, ''] : ['', said]))

  return lines.join('\n')
}

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

// The command that picks a session back up: claude's by id, codex's by path.
const resumeOf = (path: string): string =>
  !path.includes('/.codex/') && path.endsWith('.jsonl') ? `claude --resume ${baseName(path).replace(/\.jsonl$/, '')}` : path

// One draft line for a pin.
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

const chatLine = (chat: Chat): string =>
  `- ${chat.when} · ${chat.title.replace(/·|`/g, ' ').trim()} · \`${chat.id}\`${chat.done ? ' (done)' : ''}`

// The session → anchor pairs with this session's set (or dropped, for null), most
// recent last, the oldest let go past `kept`. An empty note is kept: it says the session
// chose to have no anchor, so none is carried over to it.
export const withAnchor = (pairs: [string, string][], id: string, note: string | null, kept = ANCHORS_KEPT): [string, string][] => {
  const others = pairs.filter(([one]) => one !== id)

  return (note === null ? others : [...others, [id, note] as [string, string]]).slice(-kept)
}

// What a chat is called in its anchor's log when no pane draft names it: its prompt's
// first line. Empty for a slash command or a line that is only markup.
export const promptTitle = (text: string): string => {
  const first = text.split('\n').map(line => line.trim()).find(Boolean) ?? ''

  return first.startsWith('/') ? '' : first.replace(/<[^>]+>/g, ' ').replace(/[·`]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 70)
}

// What the store holds under the anchors key, read defensively.
export const anchorPairs = (value: unknown): [string, string][] =>
  Array.isArray(value)
    ? value.filter((one): one is [string, string] => Array.isArray(one) && typeof one[0] === 'string' && typeof one[1] === 'string')
    : []

// Pins as the pins card lists them: artifacts first, each group in its order.
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

export type Anchor = { note: string; notePath: string }

// `pins` are the anchor's pinned context, a note's `path` filled where it was found;
// `today` the path of today's note when there is one.
export const buildDraft = (
  pins: Pin[],
  choice: Picks,
  lens: Lens | null,
  typed: string,
  anchor: Anchor | null = null,
  today = '',
  // A lens whose job is to write (a page, rows in the note): run as written, files and all,
  // without the short-answer and no-files rules a lens answered in chat gets.
  writes = false,
): string => {
  const lines = [DRAFT_HEAD]

  if (anchor && choice.project) {
    // The note's name in [[ ]] lets a pasted draft anchor a new chat (anchorFromDraft).
    const where = anchor.notePath ? `"${anchor.notePath}"` : '(find it in the vault)'
    lines.push(`- Project [[${anchor.note}]]: ${where}, my note. Its outcome, Next and Journal are mine; the "# Agent Log" section at its end holds what is pinned to this work and, folded, the chats on it with where each last stood. Read it.`)
  }
  for (const pin of anchor ? pins.filter(one => one.on) : []) {
    lines.push(pinDraftLine(pin))
  }
  if (choice.aboutMe) {
    lines.push(`- About me and my projects: read "${P.context}" and "${P.board}".`)
  }
  if (today && choice.today && P.week) {
    lines.push(`- This week: read "${today}", my note for this week: the workstreams under its "${P.weekHeading}" heading and what I logged under each day.`)
  } else if (today && choice.today) {
    lines.push(`- Today: read "${today}", my note for today: what I set out to do and what I have logged so far.`)
  }
  if (choice.openAugi) {
    lines.push(`- Search my notes in "${P.notes}" for what is relevant to this (with the OpenAugi tools if you have them), and name the notes you used.`)
  }
  if (lens && writes) {
    lines.push(`- Lens: read "${lens.path}" and run it as written, step by step, writing the files and the lines it asks for.`)
  } else if (lens) {
    lines.push(`- Lens: read "${lens.path}" and apply it to the context above. Answer here in chat instead of writing its output file.`)
  }
  if (!writes) {
    lines.push('Reply in at most 10 lines: what you found or did, then anything you need from me. No walls of text.')
    lines.push("Don't change any files unless my instruction asks you to.")
  }
  if (anchor) {
    lines.push(`Write in my note only when I ask. What I ask you to save goes under its "# Agent Log" heading, never above it, one line with why; "- [[<note>]] · <why it matters>" pins a note.`)
  }
  lines.push(`Instruction: ${typed}`)

  return lines.join('\n')
}

// The row above the prompt: the chat's anchor, then from its state what waits on the user
// (when that is not "no"), else its next step. Empty with no anchor.
export const bandLine = (anchor: string, state: string): string => {
  // A state line is `Label: text`, maybe bulleted or bold; the first one with text wins.
  const said = (label: string): string =>
    state.split('\n').map(line => line.replace(/\*/g, '').replace(/^[-\s]+/, '')).map(line => (line.toLowerCase().startsWith(`${label.toLowerCase()}:`) ? line.slice(label.length + 1).trim() : '')).find(Boolean) ?? ''
  const waiting = said('Waiting on me')
  const next = said('Next')
  const parts = [
    anchor ? `⚓ ${anchor}` : '',
    waiting && !/^(no|nothing|none)\b/i.test(waiting) ? `⏳ ${waiting}` : next ? `→ ${next}` : '',
  ].filter(Boolean)

  return anchor ? parts.join('  ·  ') : ''
}

// What an anchored chat's model is told beside the user's next prompt, unseen by them:
// which note the chat is on and where its writing goes, so "update the task note" needs
// no path. `was` is the anchor it was told before. It says where; it asks for nothing.
export const anchorBrief = (anchor: Anchor, today = '', was = ''): string => {
  const where = anchor.notePath ? `"${anchor.notePath}"` : '(find it in my notes by name)'
  const lines = [
    was && was !== anchor.note
      ? `Augi: this chat is now anchored to [[${anchor.note}]], no longer to [[${was}]].`
      : `Augi: this chat is anchored to [[${anchor.note}]], the task or project it is working on.`,
    `- My note: ${where}. Everything above its "# Agent Log" heading is mine: its outcome, what is next and its Journal.`,
    '- Its "# Agent Log" section, at the end of that note, is the pane\'s: what is pinned to this work, and a folded list of the chats on it with where each last stood. It is not a log for you to keep. Add a line there only when I ask you to save something, above the Chats list, one line with why; "- [[<note>]] · <why>" pins a note.',
  ]
  if (today) {
    lines.push(P.week ? `- This week's note: "${today}". Its "${P.weekHeading}" checkboxes are what I am working on this week.` : `- Today's note: "${today}".`)
  }
  lines.push('When I say "the task note", "the project note", "log this" or "save this to the task" without naming a file, I mean this note. What I ask you to save goes under "# Agent Log" (add the heading at the note\'s end if it is missing). Write above that heading only when I ask for that ("update the task note"), as a dated entry at the top of its Journal: "### YYYY-MM-DD — <what happened>".')
  lines.push('This says where things go. It is not a request: write nothing there until I ask.')

  return lines.join('\n')
}

// How a brief reads in the conversation the model is sent: a hook's context is a text
// block of the user's message, wrapped as a reminder. A brief quoted anywhere else (a
// tool result, a pasted file) is not such a block.
const BRIEF_SAID = 'prompt.submit hook additional context: Augi: this chat is '

// What a conversation's briefs say its chat is anchored to, the last one deciding: a
// note's name, '' when the last word was that it has none, null when it holds no brief.
// A forked chat starts with its parent's conversation, so this is how a fork keeps its
// parent's anchor.
export const anchorFromMessages = (messages: readonly { role?: unknown; content?: unknown }[]): string | null => {
  let said: string | null = null
  for (const message of messages) {
    for (const block of message.role === 'user' && Array.isArray(message.content) ? message.content : []) {
      const text: string = block?.type === 'text' && typeof block.text === 'string' ? block.text : ''
      const at = text.startsWith('<system-reminder>') ? text.indexOf(BRIEF_SAID) : -1
      if (at < 0) {
        continue
      }
      const rest = text.slice(at + BRIEF_SAID.length)
      const note = rest.match(/^(?:now )?anchored to \[\[([^\]]+)\]\]/)?.[1]
      if (note) {
        said = note
      } else if (rest.startsWith('no longer anchored to ')) {
        said = ''
      }
    }
  }

  return said
}

// The same, for a chat whose anchor was taken away after it was told one.
export const unanchoredBrief = (was: string): string =>
  `Augi: this chat is no longer anchored to [[${was}]]. Leave its note alone unless I name it.`

// And for a chat that already knows its anchor, when the day's note is another one.
export const todayBrief = (today: string): string => `Augi: ${P.week ? "this week's" : "today's"} note is "${today}".`

// What to tell the chat now, given what it was told (`before`): the whole brief for a new
// anchor or after a compaction, one line when only today's note moved, null for no news.
export const anchorNews = (before: Told, session: string, anchor: Anchor | null, today: string): { text: string; next: Told } | null => {
  const last = before.session === session ? before : { session, note: '', today: '', again: false }
  if (!anchor) {
    return last.note ? { text: unanchoredBrief(last.note), next: { session, note: '', today: '', again: false } } : null
  }
  const next = { session, note: anchor.note, today, again: false }
  if (last.note !== anchor.note || last.again) {
    return { text: anchorBrief(anchor, today, last.note), next }
  }

  return today && today !== last.today ? { text: todayBrief(today), next } : null
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
// A log file whose content has not gone into its note yet goes in now: the first time the
// pane reads or writes an anchor that still has one. The file is marked as moved, never
// deleted, and not read again. `text` is the note as just read; the note as it is now
// comes back.
async function withMovedLog($: EngineInterface, note: string, path: string, text: string): Promise<string> {
  await moveOldTrail($, note)
  const from = trailPath(note)
  if (!(await $.fs.exists(from))) {
    return text
  }
  const old = await $.fs.read(from)
  if (isMovedLog(old)) {
    return text
  }
  const merged = withOldLog(text, old, note)
  await $.fs.write(path, merged)
  await $.fs.write(from, markMoved(old, note, stamp(await $.clock.now()).slice(0, 10)))

  return merged
}

async function moveOldTrail($: EngineInterface, note: string): Promise<void> {
  const from = oldTrailPath(note)
  const to = trailPath(note)
  if ((await $.fs.exists(from)) && !(await $.fs.exists(to))) {
    const text = tidyTrail(await $.fs.read(from), note)
    await $.fs.write(to, text)
    await $.process.run(['rm', from])
  }
}

const NO_ANCHOR: Pick<Swatch, 'anchor' | 'notePath' | 'outcome' | 'next' | 'jira' | 'trail' | 'pinPaths'> = { anchor: null, notePath: '', outcome: '', next: [], jira: '', trail: EMPTY_TRAIL, pinPaths: {} }

// Reads the anchored card's note, its Agent Log section with it; the rest of the pane stays as it was.
async function loadAnchor($: EngineInterface, cards: Card[]): Promise<Pick<Swatch, 'anchor' | 'notePath' | 'outcome' | 'next' | 'jira' | 'trail' | 'pinPaths'>> {
  const chosen = await sessionAnchor($)
  let anchor = chosen ? anchorCard(cards, chosen) : null
  // A note not on the board (a pasted draft named it, or it left the board) still
  // anchors, bare, while a note of that name is in the notes folder.
  const bare = chosen && !anchor ? await notePathOf($, chosen) : ''
  if (chosen && bare) {
    anchor = { column: '', note: chosen, detail: '', tasks: [] }
  }
  if (!anchor) {
    return NO_ANCHOR
  }
  const notePath = bare || (await notePathOf($, anchor.note))
  const noteText = notePath ? await withMovedLog($, anchor.note, notePath, await $.fs.read(notePath)) : ''
  const trail = parseTrail(noteText)
  const pinPaths: Record<string, string> = {}
  for (const pin of trail.pins.filter(one => one.on && one.kind === 'note')) {
    pinPaths[pin.target] = await notePathOf($, pin.target)
  }

  return {
    anchor,
    notePath,
    outcome: outcomeFrom(noteText),
    next: nextFrom(noteText),
    jira: jiraFrom(noteText),
    trail,
    pinPaths,
  }
}

// Which note the pane keeps in view: this week's when the weekNote setting names one, else
// today's, from the dailyNote setting or Obsidian's own daily-note settings.
async function periodPattern($: EngineInterface): Promise<{ pattern: string; week: boolean }> {
  if (P.week) {
    return { pattern: P.week, week: true }
  }

  return { pattern: dailyPattern(P.daily, P.daily ? '' : await readOr($, P.obsidianDaily), P.daily ? '' : await readOr($, P.obsidianApp)), week: false }
}

// Where the note is now: today's path; for a week, the first of its days' paths that is
// there, else Monday's (where the pane would make it). Empty for no pattern.
async function periodPath($: EngineInterface, pattern: string, week: boolean, now: number): Promise<string> {
  if (!week) {
    return dailyNotePath(pattern, now, P.notes, P.home)
  }
  const paths = weekNotePaths(pattern, now, P.notes, P.home)
  for (const path of paths) {
    if (await $.fs.exists(path)) {
      return path
    }
  }

  return paths[0] ?? ''
}

// Today's (or this week's) note as the pane shows it: where it is, whether it is there,
// what is still open in it; for a week, its workstreams.
async function todayNow($: EngineInterface, pattern: string, week = false): Promise<Today> {
  const now = await $.clock.now()
  const path = await periodPath($, pattern, week, now)
  if (!path) {
    return NO_TODAY
  }
  const day = stamp(now).slice(0, 10)
  if (!(await $.fs.exists(path))) {
    return { pattern, path, day, exists: false, todo: [], mtime: 0, week, goals: [], entries: 0 }
  }
  let mtime = 0
  try {
    mtime = (await $.fs.stat(path)).mtimeMs
  } catch {
    mtime = 0
  }
  const text = await $.fs.read(path)
  const goals = week ? weekGoals(text, P.weekHeading) : []

  return { pattern, path, day, exists: true, todo: week ? goals.filter(one => !one.done).map(one => one.text) : openBoxes(text), mtime, week, goals, entries: week ? entryBoxes(text, P.weekHeading).length : 0 }
}

// Reads the note again when the day (or week) rolled over, the note appeared, or it
// changed (ticked in the editor, say). A stat or two when nothing did.
async function resyncToday($: EngineInterface): Promise<void> {
  const { today } = await read($, swatch)
  if (!today.pattern) {
    return
  }
  const path = await periodPath($, today.pattern, today.week, await $.clock.now())
  let mtime = 0
  try {
    mtime = (await $.fs.exists(path)) ? (await $.fs.stat(path)).mtimeMs : 0
  } catch {
    mtime = 0
  }
  if (path === today.path && mtime === today.mtime && today.exists === mtime > 0) {
    return
  }
  const next = await todayNow($, today.pattern, today.week)
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
      await $.ui.toast("No note for today yet: make it first.")

      return false
    }
    const old = await $.fs.read(today.path)
    // In a week's note each line goes under its own day's heading.
    // A day's heading in the week's note keeps its level from the setting (`### …`).
    const day = fillTemplate(P.dayHeading, await $.clock.now(), '')
    const level = today.week ? (day.match(/^#+/)?.[0].length ?? 3) : 2
    const heading = today.week ? day.replace(/^#+\s*/, '') : P.dailyHeading
    await $.fs.write(today.path, [...entries].reverse().reduce((text, one) => addDailyLine(text, heading, one, level, today.week), old))
  } catch (error) {
    await $.ui.toast(`Couldn't write today's note: ${String(error)}`)

    return false
  }
  const next = await todayNow($, today.pattern, today.week)
  await update($, swatch, old => ({ ...old, today: next }))

  return true
}

// Makes today's note, on the user's press and never by itself: from the template
// Obsidian's daily notes use when its settings name one that is there, else empty. A
// week's note is named for its Monday and starts with last week's open workstreams carried
// over. Never over a note that exists. Then it opens.
async function makeToday($: EngineInterface): Promise<void> {
  const { today } = await read($, swatch)
  if (!today.pattern) {
    return
  }
  const now = await $.clock.now()
  const path = await periodPath($, today.pattern, today.week, now)
  try {
    if (!(await $.fs.exists(path))) {
      let text = ''
      if (today.week) {
        const lastWeek = await periodPath($, today.pattern, true, now - 7 * 24 * 3600_000)
        text = newWeekNote(await readOr($, lastWeek), P.weekHeading)
      } else {
        const template = dailyTemplatePath(await readOr($, P.obsidianDaily), P.notes)
        text = template ? fillTemplate(await readOr($, template), now, baseName(path).replace(/\.md$/, '')) : ''
      }
      await writeNew($, path, text)
    }
  } catch (error) {
    await $.ui.toast(`Couldn't make ${today.week ? "this week's" : "today's"} note: ${String(error)}`)

    return
  }
  const next = await todayNow($, today.pattern, today.week)
  await update($, swatch, old => ({ ...old, today: next }))
  await openNote($, path)
}

// `✓ close out`, confirmed: the anchor's card ticked on the board, its line ticked in this
// week's note, and this chat marked done in its log. The chat keeps its anchor, so a
// wrap-up (an update to Jira, say) still lands there.
async function closeOut($: EngineInterface): Promise<void> {
  await update($, view, old => ({ ...old, closing: false }))
  const { anchor, today, jira } = await read($, swatch)
  if (!anchor) {
    return
  }
  const done: string[] = []
  const old = await readOr($, P.board)
  const board = tickCard(old, anchor.note)
  if (board !== old) {
    await writeBoard($, board)
    done.push('its card ticked')
  }
  if (today.week && today.exists) {
    const week = await $.fs.read(today.path)
    const ticked = tickGoal(week, P.weekHeading, anchor.note)
    if (ticked !== week) {
      await $.fs.write(today.path, ticked)
      const next = await todayNow($, today.pattern, today.week)
      await update($, swatch, current => ({ ...current, today: next }))
      done.push("ticked in this week's note")
    }
  }
  const id = await $.session.id().catch(() => '')
  const { trail } = await read($, swatch)
  if (id && trail.chats.some(chat => chat.id === id && !chat.done)) {
    await writeTrail($, anchor.note, text => markChatDone(text, id))
    done.push('this chat marked done in its log')
  }
  await $.ui.toast(done.length > 0
    ? `Closed out ${anchor.note}: ${done.join(', ')}.${jira ? ' ↑ update Jira drafts the wrap-up.' : ''}`
    : `Nothing left open for ${anchor.note}: no open card, week line or chat entry.`)
}

// ---- Board sync, on the user's press ----

// Writes the board and reads its cards back into the pane.
async function writeBoard($: EngineInterface, board: string): Promise<void> {
  await $.fs.write(P.board, board)
  const cards = kanbanCards(board)
  await update($, swatch, current => ({ ...current, cards, anchor: current.anchor ? (anchorCard(cards, current.anchor.note) ?? current.anchor) : null }))
}

// The week's workstreams and the board, both read fresh: either may be open in the editor.
async function syncNow($: EngineInterface): Promise<{ goals: Goal[]; board: string; week: string } | null> {
  const { today } = await read($, swatch)
  if (!today.week || !today.exists || !(await $.fs.exists(P.board))) {
    await $.ui.toast(today.week && today.exists ? `No board at ${P.board}: set one up first.` : "No note for this week yet: make it first.")

    return null
  }
  const week = await $.fs.read(today.path)

  return { goals: weekGoals(week, P.weekHeading), board: await $.fs.read(P.board), week }
}

// `⇅ sync this week`, the one button. It writes nothing: it reads the week's note and the
// board and drafts every edit they need for the chat to show as one table. The pane works
// out the sure rows itself: a plain workstream whose words are a note's name (a board
// card's, any case, else a note file's) to link to it; a card for each open linked
// workstream with none; a tick for each done one's card. The rows that need judgement are
// left to the chat: which note a plain workstream is, checkboxes in the day entries, cards
// that are not this week's, Jira keys.
async function syncWeek($: EngineInterface): Promise<void> {
  const now = await syncNow($)
  if (!now) {
    return
  }
  const { today, taskTemplate } = await read($, swatch)
  const cards = kanbanCards(now.board)
  // The week as it reads once the sure links are made: what the board rows follow.
  let week = now.week
  const sure: string[] = []
  for (const goal of now.goals.filter(one => !one.done && !one.note)) {
    const card = cards.find(one => one.note.toLowerCase() === goal.text.toLowerCase())
    // A name `find` would read as a pattern, or as a path, is never looked up.
    const found = card ? '' : /[*?[\]/\\]/.test(goal.text) ? '' : await notePathOf($, goal.text)
    const name = card?.note ?? (found ? baseName(found).replace(/\.md$/, '') : '')
    if (name) {
      week = linkGoal(week, goal.line, name)
      sure.push(`link "${goal.line}" to [[${name}]], an existing ${card ? 'card' : 'note'}`)
    }
  }
  const goals = weekGoals(week, P.weekHeading)
  const plan = boardSync(goals, cards)
  const column = P.taskColumn || (now.board.split(/^\*\*\*$/m)[0]?.match(/^## (.+)$/m)?.[1]?.trim() ?? 'the first')
  sure.push(...plan.add.map(note => `add a card "- [ ] [[${note}]]" at the end of the "${column}" column`))
  sure.push(...plan.tick.map(note => `tick the card for [[${note}]] where it is: its line is ticked in the week's note`))

  const jira = await jiraAsk($)
  const noKey: { note: string; path: string }[] = []
  for (const note of jira ? [...new Set(goals.filter(goal => !goal.done && goal.note).map(goal => goal.note ?? ''))] : []) {
    const path = await notePathOf($, note)
    if (!jiraFrom(path ? await readOr($, path) : '')) {
      noKey.push({ note, path })
    }
  }
  const entries = entryBoxes(now.week, P.weekHeading)
  if (sure.length + plan.unlinked.length + entries.length + plan.stale.length + noKey.length === 0) {
    await $.ui.toast('This week and the board match: nothing to sync.')

    return
  }
  await fillOver($, weekSyncDraft({
    path: today.path,
    sure,
    unlinked: plan.unlinked,
    entries,
    stale: plan.stale,
    noKey,
    jira,
    column,
    template: taskTemplate === 'yours' ? `the template "${P.taskTemplate}"` : 'the sections "### Outcome", "### Verify", "# Journal" and "# Agent Log"',
  }))
  await $.ui.toast("Sync drafted: press enter to see every edit as one table. Nothing is changed until you answer.")
}

// What was typed in the You card's log field goes into today's note, stamped now.
async function logToday($: EngineInterface, text: string): Promise<void> {
  if (!text.trim()) {
    return
  }
  if (await writeToday($, [{ when: stamp(await $.clock.now()), text: text.trim() }])) {
    await update($, view, old => ({ ...old, logged: (old.logged ?? 0) + 1 }))
    const { today } = await read($, swatch)
    await $.ui.toast(today.week ? "Logged under today in this week's note" : "Logged to today's note")
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

// A part of the pane's load, or `fallback` when it fails.
const part = <T,>(work: Promise<T>, fallback: T): Promise<T> => work.catch(() => fallback)

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
  await inheritAnchor($).catch(() => false)
  // Each part on its own: one that fails (a note gone, a server down) leaves its part
  // empty and the rest of the pane, the board and the anchor above all, still loads.
  const cards = await part(readOr($, P.board).then(kanbanCards), [])
  const next: Swatch = {
    setup,
    vaults: [],
    found: {},
    skipped,
    season: await part(readOr($, P.context).then(anchorFrom), ''),
    cards,
    ...(await part(loadAnchor($, cards), NO_ANCHOR)),
    jiraServer: await part(jiraServerNow($), ''),
    taskTemplate: await part(readOr($, P.taskTemplate).then(taskTemplateState), taskTemplateState('')),
    today: await part(periodPattern($).then(({ pattern, week }) => todayNow($, pattern, week)), NO_TODAY),
    lenses: [],
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
  await fillOver($, jiraUpdateDraft(notePath, anchor.note, jira, ask))
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

// The session whose conversation was already read for a brief: once is enough, since what
// a fork starts with is there from the start, and a long conversation is not read again
// on every sync.
let askedOf = ''

// What this chat's conversation says it is anchored to (`anchorFromMessages`), asked once
// a session; null after that, with no brief in it, or where the conversation can't be read.
async function toldInConversation($: EngineInterface, id: string): Promise<string | null> {
  if (askedOf === id) {
    return null
  }
  askedOf = id
  try {
    const messages = await $.session.messages({ as: 'api' })

    return Array.isArray(messages) ? anchorFromMessages(messages) : null
  } catch {
    return null
  }
}

// A new chat starts with no anchor. The one exception is a fork: it opens with its
// parent's conversation, and the brief in it names the parent's anchor, so the fork starts
// on that (a parent that had none leaves the fork with none). Nothing else carries an
// anchor over: not the folder's last pick, which used to file every task run in the
// vault into whatever note was picked last. Nothing is written to the anchor's log until
// the chat says something (nameChat), so an abandoned fork leaves no trace there. True
// when it anchored the session.
async function inheritAnchor($: EngineInterface): Promise<boolean> {
  try {
    const id = await $.session.id()
    const pairs = anchorPairs(await $.store.get(ANCHORS_KEY))
    if (!id || pairs.some(([one]) => one === id)) {
      return false
    }
    const note = await toldInConversation($, id)
    if (!note) {
      return false
    }
    // Never a headless run: nobody is there to see, keep or change it.
    if ((await $.session.surfaces()).length === 0) {
      return false
    }
    // Nor onto a note that was renamed or deleted since: the chat would be filed in a log for nothing.
    if (!anchorCard(kanbanCards(await readOr($, P.board)), note) && !(await notePathOf($, note))) {
      return false
    }
    // Read again: a pick made while the checks above ran is the session's choice.
    const now = anchorPairs(await $.store.get(ANCHORS_KEY))
    if (now.some(([one]) => one === id)) {
      return false
    }
    await $.store.set(ANCHORS_KEY, withAnchor(now, id, note))
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
// When the board and the anchor's note were last changed, as one string.
async function anchorStamp($: EngineInterface, note: string, notePath: string): Promise<string> {
  const stamps: string[] = []
  for (const path of [P.board, notePath]) {
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

// How many times this pane has written an Agent Log: a sync under way checks it (resync).
let logWrites = 0
// What the pane last saw of those files. Another chat on the same anchor writes the same
// note; this is how this chat's pane notices and reads it again.
let seenStamp = ''
// A stored anchor whose note was not found, so the sync does not search for it every time.
let unfound = ''
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
    // A chat cleared or resumed in this process is another session: it shows its own stored
    // anchor, or, a fork, its parent's; any other new chat starts with none.
    const stored = await sessionAnchor($)
    if ((stored && stored !== unfound) || (await inheritAnchor($))) {
      const { cards } = await read($, swatch)
      const found = await loadAnchor($, cards)
      if (found.anchor) {
        await update($, swatch, old => ({ ...old, ...found }))
      } else {
        // Its note is gone: not looked for again on every sync.
        unfound = stored ?? ''
      }
    }

    return
  }
  const stamp = await anchorStamp($, anchor.note, notePath)
  if (stamp === seenStamp) {
    return
  }
  seenStamp = stamp
  const cards = kanbanCards(await readOr($, P.board))
  // A pin or a chat written while the note was being read would be lost from the pane until
  // the next sync: read again until no write landed in between.
  let wrote = logWrites
  let found = await loadAnchor($, cards)
  while (wrote !== logWrites) {
    wrote = logWrites
    found = await loadAnchor($, cards)
  }
  await update($, swatch, old => (old.anchor?.note === anchor.note ? { ...old, cards, ...found } : old))
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
      // Opened already: but an update that changed the shape of what the pane keeps
      // starts it over, empty, and an open pane would stay empty until /augi.
      if (!(await read($, swatch)).loadedAt) {
        await load($)
        watchAnchor($)
      }
      return
    }
    await load($)
    await $.ui.open({ id: PANE, title: 'Augi' })
    // Marked only once it has opened: a failed open is tried again when a window attaches.
    await update($, opened, () => ({ session: id }))
    watchAnchor($)
  } catch {
    // The pane stays closed; /augi opens it.
  }
}

async function anchorSession($: EngineInterface, note: string): Promise<void> {
  await setSessionAnchor($, note)
  await keepInherited($)
  // A field opened for the anchor it leaves (a sub-task, a chat to make a task) closes.
  await update($, view, old => ({ ...old, taskMode: '' as const }))
  const { cards, loadedAt } = await read($, swatch)
  if (loadedAt > 0) {
    const found = await loadAnchor($, cards)
    await update($, swatch, old => ({ ...old, ...found }))
  }
}

// Anchoring puts this chat in the anchor's ## Chats.
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
  await update($, swatch, old => ({ ...old, anchor: null, notePath: '', outcome: '', next: [], jira: '', trail: EMPTY_TRAIL, pinPaths: {} }))
  await update($, view, old => ({ ...old, anchorsOpen: false, taskMode: '' as const }))
}

async function anchorNow($: EngineInterface): Promise<Anchor | null> {
  const { anchor, notePath } = await read($, swatch)

  return anchor ? { note: anchor.note, notePath } : null
}

// `about` replaces what's typed in the prompt box: what a Jira update is for, for one.
async function draftText($: EngineInterface, lens: Lens | null, about?: string, writes = false): Promise<string> {
  const { trail, pinPaths, today } = await read($, swatch)
  const choice = await read($, picks)
  const typed = about ?? typedFrom((await $.prompt.read()).text)

  return buildDraft(withPaths(trail.pins ?? [], pinPaths), choice, lens, typed, await anchorNow($), today.exists ? today.path : '', writes)
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

async function draft($: EngineInterface, lens: Lens | null, about?: string, writes = false): Promise<void> {
  await $.prompt.fill({ text: await draftText($, lens, about, writes), mode: 'replace' })
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
  if (!(await writeTrail($, note, text => addPin(text, kept)))) {
    await $.ui.toast(`No note found for ${note}: pins are kept in its Agent Log section.`)
  }
}

// A pasted link, [[note]] or path, pinned to the anchor.
async function pinTyped($: EngineInterface, value: string): Promise<void> {
  const { anchor } = await read($, swatch)
  const pin = pinFrom(value, P.notes, P.home)
  if (!pin || !anchor) {
    return
  }
  if ((pin.kind === 'file' || pin.kind === 'session') && !(await $.fs.exists(pin.target))) {
    await $.ui.toast(`Not found: ${pin.target}`)
    return
  }
  await pinIt($, anchor.note, pin)
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

// The anchor whose Agent Log holds this session's chat: anchoring records it, so any anchor.
async function paneChatAnchor($: EngineInterface): Promise<string | null> {
  return sessionAnchor($)
}

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

// Changes the anchor's Agent Log: the section at the end of its own note (`at`, when the
// caller knows the note's path). What an old log file held goes in first. False, with
// nothing written, when the note is not found.
async function writeTrail($: EngineInterface, note: string, change: (text: string) => string, at = ''): Promise<boolean> {
  const shown = await read($, swatch)
  const path = at || (shown.anchor?.note === note && shown.notePath ? shown.notePath : await notePathOf($, note))
  if (!path) {
    return false
  }
  let was = ''
  try {
    was = await $.fs.read(path)
  } catch {
    // The note went away since it was found: nothing to write the section in.
    return false
  }
  const base = await withMovedLog($, note, path, was)
  const text = change(base)
  if (text !== base) {
    await $.fs.write(path, text)
  }
  logWrites += 1
  await update($, swatch, current => (current.anchor?.note === note ? { ...current, trail: parseTrail(text) } : current))

  return true
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

// The path of today's note when there is one, read fresh enough for a prompt: the pane's
// own copy once it has loaded, else worked out from the settings.
async function todayPathNow($: EngineInterface): Promise<string> {
  if ((await read($, swatch)).loadedAt > 0) {
    await resyncToday($)
    const { today } = await read($, swatch)

    return today.exists ? today.path : ''
  }
  const { pattern, week } = await periodPattern($)
  const today = await todayNow($, pattern, week)

  return today.exists ? today.path : ''
}

// What this chat's model should hear about its anchor with the prompt being sent, or null
// when it already knows. The session's stored anchor decides, so a closed pane tells too.
async function tellNow($: EngineInterface): Promise<{ text: string; next: Told } | null> {
  if (settings.tellChat === 'no') {
    return null
  }
  const id = await $.session.id()
  if (!id) {
    return null
  }
  const note = await sessionAnchor($)
  const before = await read($, told)
  if (!note) {
    return anchorNews(before, id, null, '')
  }
  const today = await todayPathNow($)
  // Nothing new: the note is not looked up again.
  if (!anchorNews(before, id, { note, notePath: '' }, today)) {
    return null
  }
  const shown = await read($, swatch)
  const notePath = shown.anchor?.note === note && shown.notePath ? shown.notePath : await notePathOf($, note)

  return anchorNews(before, id, { note, notePath }, today)
}

// One suggestion at a time.
let isAskingOffer = false

// The note among `candidates` that carries this Jira key (`Jira: KEY` or `jira: KEY`), or empty.
async function noteWithJira($: EngineInterface, key: string, candidates: string[]): Promise<string> {
  const { stdout } = await $.process.run(['grep', '-rlE', '--include=*.md', `^(jira|Jira): *${key}([^0-9]|$)`, P.notes])
  const found = stdout.split('\n').filter(Boolean).map(path => baseName(path).replace(/\.md$/, ''))

  return candidates.find(name => found.includes(name)) ?? ''
}

// For a chat with no anchor: works out which task it belongs to and offers it in the
// Anchor card, where one press links the chat or makes the task. Nothing is linked, made
// or written here. A Jira key in the git branch that a board note carries settles it with
// no model call; otherwise the chat's own model is asked, with the board's notes to pick from.
async function suggestTask($: EngineInterface, by: 'press' | 'auto'): Promise<void> {
  const id = await $.session.id()
  if (!id || isAskingOffer || (await sessionAnchor($))) {
    return
  }
  isAskingOffer = true
  let error = ''
  try {
    await update($, offer, () => ({ ...NO_OFFER, session: id, busy: true }))
    const shown = await read($, swatch)
    const candidates = offerCandidates(shown.loadedAt > 1 ? shown.cards : kanbanCards(await readOr($, P.board)))
    let branch = ''
    try {
      branch = (await $.process.run(['git', '-C', await $.session.cwd(), 'branch', '--show-current'])).stdout.trim()
    } catch {
      branch = ''
    }
    const named = jiraKeyIn(branch)
    const linked = named ? await noteWithJira($, named, candidates) : ''
    if (linked) {
      await update($, offer, () => ({ ...NO_OFFER, session: id, note: linked, jira: named, why: `The branch ${branch} names ${named}, and this note carries it.` }))

      return
    }
    // A branch can look like a key and not be one (`release-2026-11`): with no note carrying
    // it, it counts as one only in the Jira project the settings name.
    const jira = P.jiraProject && named.startsWith(`${P.jiraProject.toUpperCase()}-`) ? named : ''
    const reply = await $.model.fork({ prompt: offerAsk(candidates, jira ? `The git branch is "${branch}", which names the Jira issue ${jira}; no task of mine carries that key yet.` : ''), effort: 'low' })
    const found = reply.isAnswered ? parseOffer(reply.text, candidates) : { note: '', title: '', why: '' }
    if (found.note || found.title) {
      await update($, offer, () => ({ ...NO_OFFER, session: id, ...found, jira }))

      return
    }
    error = reply.isAnswered ? 'No task suggested for this chat.' : reply.reason === 'nothing-to-fork' ? 'Nothing to go on yet: send a prompt first.' : `No answer (${reply.reason}).`
  } catch (failure) {
    error = `No answer: ${String(failure)}`
  } finally {
    isAskingOffer = false
  }
  // Asked and unanswered counts as asked: auto does not try again in this chat.
  await update($, offer, () => ({ ...NO_OFFER, session: id, error, done: true }))
  if (by === 'press') {
    await $.ui.toast(error)
  }
}

// Takes the offer: links this chat to the note, or makes the new task (a standalone one,
// which moves the chat onto it; the branch's Jira key goes with its title).
async function takeOffer($: EngineInterface): Promise<void> {
  const one = await read($, offer)
  await update($, offer, old => ({ ...old, done: true }))
  if (one.note) {
    const { cards } = await read($, swatch)
    await chooseAnchor($, anchorCard(cards, one.note) ?? { column: '', note: one.note, detail: '', tasks: [] })
  } else if (one.title) {
    await newTask($, one.jira ? `${one.title} ${one.jira}` : one.title, 'new')
  }
}

// After a turn, with auto on: a chat that has no anchor, has not said it wants none, and
// has not been offered a task yet is offered one, once.
async function autoOffer($: EngineInterface): Promise<void> {
  const id = await $.session.id()
  if (settings.stateAuto !== 'yes' || !id || (await read($, offer)).session === id) {
    return
  }
  if (anchorPairs(await $.store.get(ANCHORS_KEY)).some(([one]) => one === id)) {
    return
  }
  await suggestTask($, 'auto')
}

const stateLensName = (): string => (settings.stateLens ?? '').trim() || STATE_LENS

// One ask at a time: a press while one is out does nothing.
let isAskingState = false
// The least time between two automatic asks in one chat.
export const STATE_EVERY_MS = 10 * 60_000

// Asks this chat's own model where the chat stands: a side question over the conversation
// as it is (`$.model.fork`: no tools, nothing added to the chat), worded by the state
// lens. The answer is this chat's alone: it shows in the pane and, with an anchor, goes
// under this chat's entry in the note's chat list, replacing what was there. No answer
// changes nothing but the reason shown.
async function updateState($: EngineInterface, by: 'press' | 'auto'): Promise<void> {
  const id = await $.session.id()
  if (!id || isAskingState) {
    return
  }
  isAskingState = true
  let error = ''
  try {
    await update($, chatState, old => ({ ...(old.session === id ? old : NO_STATE), session: id, busy: true, error: '' }))
    const ask = stateAsk(lensBody(await readOr($, `${P.lenses}/${stateLensName()}.md`)))
    const reply = await $.model.fork({ prompt: ask, effort: 'low' })
    const text = reply.isAnswered ? stateLines(reply.text).join('\n') : ''
    if (text) {
      const now = await $.clock.now()
      const when = stamp(now)
      await update($, chatState, () => ({ session: id, text, when, at: now, busy: false, error: '' }))
      const note = await sessionAnchor($)
      if (note) {
        // A chat that is not in the list yet (it has an anchor it never spoke on) goes in first.
        await writeTrail($, note, was => setChatState(addChat(was, { id, when, title: CHAT_PLACEHOLDER, done: false }), id, when, text))
      }

      return
    }
    error = reply.isAnswered ? 'The answer was empty.' : reply.reason === 'nothing-to-fork' ? 'Nothing to go on yet: send a prompt first.' : `No answer (${reply.reason}).`
  } catch (failure) {
    error = `No answer: ${String(failure)}`
  } finally {
    isAskingState = false
  }
  // A failed ask is paced like an answered one, so an error is not asked again every turn.
  const at = await $.clock.now()
  await update($, chatState, old => ({ ...old, session: id, at, busy: false, error }))
  if (by === 'press') {
    await $.ui.toast(error)
  }
}

// After a turn, with auto on: asks again when this chat was never asked or not for STATE_EVERY_MS.
async function autoState($: EngineInterface): Promise<void> {
  if (settings.stateAuto !== 'yes') {
    return
  }
  await ready($)
  await autoOffer($)
  const id = await $.session.id()
  const last = await read($, chatState)
  if (last.session === id && (await $.clock.now()) - last.at < STATE_EVERY_MS) {
    return
  }
  await updateState($, 'auto')
}

// Opens the state lens to edit; one that is not in the lens folder yet is written first,
// from the built-in prompt.
async function editStatePrompt($: EngineInterface): Promise<void> {
  const path = `${P.lenses}/${stateLensName()}.md`
  await writeNew($, path, stateLensFile(stateLensName()))
  await openNote($, path)
  await load($)
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

// Writes a task note from the template and puts it on the board as a top-level card at
// the end of the column `into` (else of the first). The board is read back into the pane.
// Null, after a toast saying why, when nothing was made: never over a note that is there.
async function makeTask($: EngineInterface, typed: string, into: string): Promise<{ name: string; path: string; jira: string; where: string } | null> {
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
  await writeNew($, path, taskNoteText(template, name, stamp(await $.clock.now()).slice(0, 10), null, jira))
  const board = insertCard(await $.fs.read(P.board), name, null, into)
  const column = kanbanCards(board).find(one => one.note === name)?.column || 'the first column'
  await $.fs.write(P.board, board)
  const cards = kanbanCards(board)
  await update($, swatch, current => ({ ...current, cards, anchor: current.anchor ? (anchorCard(cards, current.anchor.note) ?? current.anchor) : null }))

  return { name, path, jira, where: `a card at the end of ${column}` }
}

// What every new task ends with: the just-made row and a toast saying where it went.
async function madeTask($: EngineInterface, one: { name: string; path: string; jira: string; where: string }, also = ''): Promise<void> {
  await update($, justMade, () => ({ note: one.name, path: one.path, jira: one.jira, where: one.where }))
  const folder = settings.taskFolder?.trim() ? '' : " New tasks go to the notes folder's root: Setup → new task folder changes it."
  await $.ui.toast(`Made ${one.name} at ${one.path}; on the board: ${one.where}.${also}${folder}`)
}

// A new task from this chat, on the user's press: a card in the anchor's column, else the
// task column setting's, else the first. This chat's transcript is pinned to it. A new task
// is this chat starting something of its own, so the chat moves onto it: the next draft
// carries the new task's context. The prompt box is left alone. A piece of an anchor's work
// is a chat (a fork keeps the anchor), not a sub-task note: there is no sub-task button.
async function newTask($: EngineInterface, typed: string): Promise<void> {
  await update($, view, old => ({ ...old, taskMode: '' as const }))
  const anchored = (await read($, swatch)).anchor
  // A standalone task belongs where the chat is working: its anchor's column.
  const made = await makeTask($, typed, anchored?.column || P.taskColumn)
  if (!made) {
    return
  }
  const transcript = await ownTranscript($)
  // Written even with nothing to pin: the first write drops a template's old log embed.
  await writeTrail($, made.name, text => (transcript ? addPin(text, { on: true, kind: 'session', target: transcript, label: 'the chat this task came from' }) : text), made.path)
  await madeTask($, made, ' This chat is anchored on it now.')
  await anchorOnMade($)
}

async function defineHere($: EngineInterface): Promise<void> {
  const one = await read($, justMade)
  if (!one.note) {
    return
  }
  // A typed key is already linked: the draft asks for no search then.
  const ask = one.jira ? null : await jiraAsk($)
  await fillOver($, newTaskDraft(one.path, one.note, ask))
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
    await $.command.register({ name: 'augi', description: 'Show Augi pane: your anchor, its context, and your saved prompts' })
    // A reload starts this module's variables over: the unsent scratchpad text comes back.
    typedPad = await read($, padDraft)
    // A settings change reloads the module; redraw an open pane with the new paths.
    const starter = await $.store.get('starter')
    if (typeof starter === 'string') {
      await $.store.delete('starter')
      await writeStarter($, starter)
    }
    // Loaded before, or opened in this session and then reset by an update: load again.
    const id = await $.session.id().catch(() => '')
    if ((await read($, swatch)).loadedAt > 0 || (id && (await read($, opened)).session === id)) {
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
  // pane drops the old session's anchor, and the new session starts with none.
  on('session.end', async ($, e, next) => {
    const ended = await next(e)
    if (e.reason === 'clear' || e.reason === 'resume') {
      await update($, swatch, old => ({ ...old, anchor: null, notePath: '', outcome: '', next: [], jira: '', trail: EMPTY_TRAIL, pinPaths: {} }))
      // The scratchpad was that chat's; it doesn't follow into the next.
      await update($, pad, () => [])
      await typePad($, '')
      await update($, justMade, () => NO_TASK)
      await update($, view, old => ({ ...old, taskMode: '' as const, closing: false }))
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

    return { text: 'Augi is open. Pick an anchor, tick its context, then draft a prompt or run a saved prompt. It only drafts; you press enter.' }
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

    // The chat's own model learns its anchor here: a block beside the prompt, not shown
    // to the user, sent once and again only when the anchor changes. The system prompt
    // would be the wrong place: a thread's stays as first sent. Never beside a slash
    // command or a prompt the user did not type.
    let news: Awaited<ReturnType<typeof tellNow>> = null
    if (!e.command && !e.isMeta && !e.text.trim().startsWith('/')) {
      news = await tellNow($).catch(() => null)
    }

    // Sending a prompt is a good moment to catch up with what other chats wrote.
    resync($).catch(() => undefined)

    if (!news) {
      return next(e)
    }
    const entered = await next({ ...e, context: [...(e.context ?? []), news.text] })
    if (!entered.drop) {
      const said = news.next
      await update($, told, () => said)
    }

    return entered
  })

  // A compaction may summarize the anchor away: the next prompt says it again.
  // With auto on, the end of a main-thread turn that answered may ask for the chat's state.
  // Not awaited: the ask is a model call, and the turn does not wait on it.
  on('turn.complete', async ($, e, next) => {
    const done = await next(e)
    if (e.agentId === undefined && !e.isAborted && e.reason === 'answer') {
      void autoState($).catch(() => undefined)
    }

    return done
  })

  on('session.compact', async ($, e, next) => {
    const compacted = await next(e)
    if (!e.agentId && !compacted.skip) {
      await update($, told, old => ({ ...old, again: true }))
    }

    return compacted
  })

  // Every press and submit in the pane is counted, by the control's name alone: what
  // was typed and whose note it was on are not kept. A control added later is counted too.
  on('ui.press', async ($, e, next) => {
    const ran = await next(e)
    if (e.plugin === 'augi-ui' && (e.requestId === PANE || e.element === 'band-open')) {
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

  // A new note an anchored chat writes in the notes folder is pinned to its anchor.
  // Nothing else it makes is: a file or an artifact is pinned by pasting it in the card.
  on('tool.call', { tool: 'Write' }, async ($, e, next) => {
    await ready($)
    const path = e.file_path
    const existed = await existsBefore($, path)
    const ran = await next(e)
    try {
      const isNewNote = !existed && isVaultNote(path, P.notes) && !path.startsWith(`${P.trails}/`) && !path.includes('/.')
      const note = isNewNote ? await paneChatAnchor($) : null
      if (note && ran.deny === undefined && !ran.isError) {
        const name = baseName(path).replace(/\.md$/, '')
        await update($, swatch, old => ({ ...old, pinPaths: { ...old.pinPaths, [name]: path } }))
        await writeTrail($, note, text => addPin(text, { on: true, kind: 'note', target: name, label: name }))
      }
    } catch (error) {
      await $.ui.toast(`Augi couldn't pin this note: ${String(error)}`)
    }

    return ran
  })

  // One line above the prompt, where the eyes already are: the anchor, what waits on the
  // user or comes next. Pressing it opens the pane. Nothing until the pane
  // has loaded this session, and nothing over a survey; `band: no` turns it off.
  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey || settings.band === 'no') {
      return next(e)
    }
    const { anchor, trail, loadedAt } = await read($, swatch)
    if (!loadedAt) {
      return next(e)
    }
    const ownId = await $.session.id().catch(() => '')
    const asked = await read($, chatState)
    const state = asked.session === ownId && asked.text ? asked.text : (anchor && trail.states?.[ownId]?.text) || ''
    const text = bandLine(anchor?.note ?? '', state)
    if (!text) {
      return next(e)
    }
    const { Box, Button } = $.ui.resolve(e)

    return (
      <Box flexDirection="row">
        <Button key="band-open" plain dimColor label={text} onPress={() => $.ui.open({ id: PANE, title: 'Augi' }).then(() => undefined)} />
      </Box>
    )
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Button, Input, Text } = $.ui.resolve(e)
    // Svg is drawn by the desktop, editor and phone, not the terminal.
    const Svg = e.surface === 'terminal' ? null : (($.ui.resolve(e) as unknown as { Svg?: typeof Text }).Svg ?? null)
    const { setup, vaults, found, skipped, season, cards, anchor, notePath, outcome, next, jira, jiraServer, taskTemplate, today, trail, pinPaths, lenses } = await read($, swatch)
    // Jira shows only when it is on: a server found or named, a project or site set, or a key on the anchor.
    const isJiraOn = Boolean(jiraServer || P.jiraServer || P.jiraProject || P.jiraSite || jira)
    const choice = await read($, picks)
    const shown = await read($, view)
    const jotted = await read($, pad)
    const asked = await read($, chatState)
    const offered = await read($, offer)
    const madeTask = await read($, justMade)
    // Drawing never depends on it: with no id, no row is marked as this chat.
    const ownId = await $.session.id().catch(() => '')
    const carried = await read($, inherited)
    // The state shown is this chat's own: what its model last said, or, when the pane has
    // not asked in this process (a resumed chat), what the anchor's note holds for it.
    const isMine = asked.session === ownId
    const standing = isMine && asked.text ? { when: asked.when, text: asked.text } : (anchor && trail.states?.[ownId]) || { when: '', text: '' }
    // An offer is this chat's, and live until taken or turned down.
    const isOffered = offered.session === ownId && !offered.done && !offered.busy && Boolean(offered.note || offered.title)
    const isCarried = Boolean(anchor) && carried.note === anchor?.note && carried.session === ownId
    // A mode this pane no longer has (an older one's sub-task or chat-to-task form) is no form.
    const taskMode = shown.taskMode === 'new' ? shown.taskMode : ''
    const mark = (isOn: boolean): string => (isOn ? '☑' : '☐')
    const arrow = (isOpen: boolean): string => (isOpen ? '▾' : '▸')
    const flip = (key: keyof View) => () => update($, view, old => ({ ...old, [key]: !old[key] }))
    // What syncing the board would do now, from the week's workstreams and the open cards.
    const plan = today.week && today.exists ? boardSync(today.goals, cards) : null
    // What one sync press has to look at: board changes, plain workstreams, day-entry boxes.
    const toSync = plan ? plan.add.length + plan.tick.length + plan.stale.length + plan.unlinked.length + (today.entries ?? 0) : 0
    const weekAnchors = today.week ? [...new Set(today.goals.filter(goal => !goal.done && goal.note).map(goal => goal.note ?? ''))] : []
    // What is pinned now: a line ticked off on the log was unpinned.
    const pins = withPaths((trail.pins ?? []).filter(pin => pin.on), pinPaths ?? {})
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
    const CARD = { flexDirection: 'column', borderStyle: 'round', paddingX: 1, marginBottom: 1 } as const
    const gaps = setup.filter(item => !item.ok).length
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
          <Text>Augi works from a folder of markdown notes: a context note about you, a board of projects to anchor on, and saved prompts. Start with the folder.</Text>
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
          <Text dimColor>Next you point Augi at your context note, board and saved prompts, or let it create starters. All of it is also in /config under augi.</Text>
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
                      {weekAnchors.length > 0 && (
                        <Box key="column-this-week" flexDirection="column">
                          <Text dimColor>{P.weekHeading}</Text>
                          {weekAnchors.map(note => (
                            <Box key={`week-anchor-box-${note}`} flexDirection="column" paddingLeft={2}>
                              <Button key={`week-anchor-${note}`} plain label={`${anchor?.note === note ? '●' : '○'} ${note}`} onPress={() => chooseAnchor($, anchorCard(cards, note) ?? { column: '', note, detail: '', tasks: [] })} />
                            </Box>
                          ))}
                        </Box>
                      )}
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
                      <Text dimColor wrap="wrap">↩ Carried over: from the chat this one was forked from. It is filed here with its first prompt.</Text>
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
                    </Box>
                  )}
                  {anchor && !shown.closing && (
                    <Button key="close-out" plain dimColor label="✓ close out" onPress={() => update($, view, old => ({ ...old, closing: true }))} />
                  )}
                  {anchor && shown.closing && (
                    <Box key="close-confirm" flexDirection="column" marginBottom={1}>
                      <Text key="close-what" wrap="wrap">{`Close out ${anchor.note}? Ticks its card on the board${today.week && today.exists ? " and its line in this week's note" : ''}, and marks this chat done in its log. The chat stays on it.`}</Text>
                      <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                        <Button key="close-yes" label="yes, close it out" onPress={() => closeOut($)} />
                        <Button key="close-cancel" plain dimColor label="cancel" onPress={() => update($, view, old => ({ ...old, closing: false }))} />
                      </Box>
                    </Box>
                  )}
                  {anchor && (
                    <Box key="anchor-pins" flexDirection="column">
                      <Button key="head-pinsOpen" plain dimColor label={`${arrow(shown.pinsOpen)} 📌 pinned (${pins.length})`} onPress={flip('pinsOpen')} />
                      {shown.pinsOpen && (
                        <Box key="pins-body" flexDirection="column" paddingLeft={2}>
                          {artifactsFirst(pins).map(pin => line(
                            `pin-${pinKey(pin)}`,
                            <Button key={`pin-open-${pinKey(pin)}`} plain label={`${PIN_ICON[pin.kind]} ${pin.kind === 'note' && pin.label !== pin.target ? `${pin.target} · ${pin.label}` : pin.label}`} onPress={() => openPin($, pin)} />,
                            <Button key={`pin-drop-${pinKey(pin)}`} dimColor label="✕ unpin" onPress={() => onPins(text => removePin(text, pin))} />,
                          ))}
                          <Input key={`pin-paste-${pins.length}`} label="⧉" placeholder="paste a [[note]], file path or link: Enter pins it" submitLabel="pin" onSubmit={value => pinTyped($, value)} />
                        </Box>
                      )}
                    </Box>
                  )}
                  {!anchor && isOffered && (
                    <Box key="offer" flexDirection="column" marginBottom={1}>
                      <Text key="offer-what" wrap="wrap">{offered.note ? `💡 This chat looks like ${offered.note}.` : `💡 No task on the board fits. New task: “${offered.title}”${offered.jira ? ` (${offered.jira})` : ''}`}</Text>
                      {offered.why && <Text key="offer-why" dimColor wrap="wrap">{offered.why}</Text>}
                      <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                        {offered.note
                          ? <Button key="offer-link" label="link this chat to it" onPress={() => takeOffer($)} />
                          : <Button key="offer-make" label="make the task" onPress={() => takeOffer($)} />}
                        <Button key="offer-dismiss" plain dimColor label="not a task" onPress={() => update($, offer, old => ({ ...old, done: true }))} />
                      </Box>
                    </Box>
                  )}
                  {!anchor && !isOffered && (
                    <Button key="offer-ask" plain dimColor label={offered.session === ownId && offered.busy ? '… asking' : '💡 suggest a task for this chat'} onPress={() => suggestTask($, 'press')} />
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
                      <Button key="task-new" plain dimColor label="＋ new task" onPress={() => update($, view, old => ({ ...old, taskMode: 'new' as const }))} />
                    </Box>
                  )}
                  {taskMode === 'new' && (
                    <Box key="task-form" flexDirection="column">
                      <Input
                        key="task-title-new"
                        label="＋ new task"
                        placeholder="title (add PROJ-123 to link Jira): Enter makes a task and anchors this chat on it"
                        submitLabel="create"
                        autoFocus
                        onSubmit={value => newTask($, value)}
                      />
                      <Button key="task-cancel" plain dimColor label="cancel" onPress={() => update($, view, old => ({ ...old, taskMode: '' as const }))} />
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
                        {anchor?.note !== madeTask.note && <Button key="made-anchor" label="anchor on it" onPress={() => anchorOnMade($)} />}
                        <Button key="made-dismiss" plain dimColor label="dismiss" onPress={() => update($, justMade, () => NO_TASK)} />
                      </Box>
                    </Box>
                  )}
                </Box>
              )}
            </Box>

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
                  {jotted.map((one, at) => (
                    <Box key={`pad-row-${at}`} flexDirection="column" marginTop={1}>
                      <Text key={`pad-text-${at}`} wrap="wrap">{`${one.when.slice(11, 16)} ${one.text}`}</Text>
                      <Box flexDirection="row" alignItems="center" gap={2}>
                        {anchor && notePath && <Button key={`pad-save-${at}`} plain label="→ save to notes" onPress={() => saveToNotes($, [one])} />}
                        <Button key={`pad-drop-${at}`} plain dimColor label="✕ drop" onPress={() => unjot($, [one])} />
                      </Box>
                    </Box>
                  ))}
                  {anchor && notePath && jotted.length >= 2 && (
                    <Button key="pad-save-all" plain label="→ save all to Human Notes" onPress={() => saveToNotes($, jotted)} />
                  )}
                  {!(anchor && notePath) && <Text dimColor>{anchor ? `No note found for ${anchor.note}: entries stay here.` : 'Pick an anchor to save these to its Human Notes.'}</Text>}
                </Box>
              )}
            </Box>

            <Box key="card-state" {...CARD} borderColor="subtle">
              {head('stateOpen', `📍 State${standing.when ? ` · ${standing.when.slice(5)}` : ''}`)}
              {shown.stateOpen && (
                <Box key="state-body" flexDirection="column">
                  {standing.text
                    ? standing.text.split('\n').map((one, at) => <Text key={`state-line-${at}`} wrap="wrap">{one}</Text>)
                    : <Text key="state-none" dimColor wrap="wrap">{`Not asked yet. Update asks this chat's model what the last few turns did and what is next, and keeps the answer ${anchor ? "with this chat's entry in the anchor's note" : 'here'}.`}</Text>}
                  {isMine && asked.error && <Text key="state-error" dimColor wrap="wrap">{asked.error}</Text>}
                  <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                    <Button key="state-update" label={isMine && asked.busy ? '… asking' : '↻ update'} onPress={() => updateState($, 'press')} />
                    <Button key="state-auto" plain label={`${mark(settings.stateAuto === 'yes')} auto after a turn`} onPress={() => setOption($, 'stateAuto', settings.stateAuto === 'yes' ? 'no' : 'yes')} />
                  </Box>
                  <Box flexDirection="row" flexWrap="wrap" alignItems="center" gap={1}>
                    <Button key="state-prompt" plain dimColor label={`${arrow(shown.statePickOpen)} prompt: ${stateLensName()}`} onPress={flip('statePickOpen')} />
                    <Button key="state-edit" dimColor label="edit prompt" onPress={() => editStatePrompt($)} />
                  </Box>
                  {shown.statePickOpen && (
                    <Box key="state-picks" flexDirection="column" paddingLeft={2}>
                      {lenses.map(lens => (
                        <Button key={`state-pick-${lens.name}`} plain label={`${stateLensName() === lens.name ? '●' : '○'} ${lens.name}`} onPress={() => setOption($, 'stateLens', lens.name)} />
                      ))}
                      {lenses.length === 0 && <Text dimColor>No saved prompts yet: edit prompt makes this one.</Text>}
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          </Box>
        )}

        <Box key="section-anywhere" flexDirection="column">{section('anywhereOpen', 'Always · you, saved prompts and setup')}</Box>
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
                    {today.exists && <Button key="today-open" label={today.week ? "🗓 this week's note" : "☀️ today's note"} onPress={() => openNote($, today.path)} />}
                    {today.path && !today.exists && <Button key="today-make" label={today.week ? "＋ make this week's note" : "＋ make today's note"} onPress={() => makeToday($)} />}
                  </Box>
                  {today.path && !today.exists && <Text key="today-missing" dimColor wrap="wrap">{today.week ? `No note for this week yet: it would be ${short(today.path)}, with last week's open "${P.weekHeading}" checkboxes carried over.` : `No note for ${today.day} yet: it would be ${short(today.path)}.`}</Text>}
                  {!today.path && P.daily.toLowerCase() !== 'none' && <Text key="today-unset" dimColor wrap="wrap">No daily note set: name one under Setup (daily note) to keep today's note here.</Text>}
                  {today.exists && (
                    <Box key="today-body" flexDirection="column" marginTop={1}>
                      <Text key="today-title" dimColor>{today.week ? `${P.weekHeading} · ${baseName(today.path).replace(/\.md$/, '')}` : `Today · ${today.day}`}</Text>
                      {!today.week && today.todo.map((one, at) => (
                        <Text key={`today-todo-${at}`} wrap="truncate-end">{`☐ ${one}`}</Text>
                      ))}
                      {!today.week && today.todo.length === 0 && <Text key="today-clear" dimColor>Nothing open in today's note.</Text>}
                      {today.week && today.goals.map((goal, at) => (
                        goal.note
                          ? <Button key={`week-goal-${at}`} plain dimColor={goal.done} label={`${goal.done ? '☑' : '☐'} ${goal.text} ↗`} onPress={() => openByName($, goal.note ?? '')} />
                          : <Text key={`week-goal-text-${at}`} dimColor={goal.done} wrap="truncate-end">{`${goal.done ? '☑' : '☐'} ${goal.text}`}</Text>
                      ))}
                      {today.week && today.goals.length === 0 && <Text key="week-none" dimColor wrap="wrap">{`No checkboxes under "${P.weekHeading}" yet: each one is a workstream for the week, a [[note]] for the ones with a note.`}</Text>}
                      {plan && <Button key="sync-week" label={`⇅ sync this week${toSync > 0 ? ` (${toSync})` : ' ✓'}`} onPress={() => syncWeek($)} />}
                      <Input key={`today-log-${shown.logged ?? 0}`} label="+ log" placeholder={today.week ? "what you just did or decided: Enter adds it under today in this week's note" : "what you just did or decided: Enter adds it to today's note"} submitLabel="log" onSubmit={value => logToday($, value)} />
                    </Box>
                  )}
                </Box>
              )}
            </Box>

            <Box key="card-lenses" {...CARD} borderColor="suggestion">
              {head('lensesOpen', `🔍 Saved prompts · ${lenses.length}`, 'suggestion', 'lens-menu', 'l')}
              {shown.lensesOpen && (
                <Button key="draft" plain label={anchor ? '💬 no prompt: just this context' : '💬 no prompt: just my context'} hotkey="n" onPress={() => draft($, null)} />
              )}
              {shown.lensesOpen && lenses.map((lens, index) => line(
                `lens-row-${lens.name}`,
                <Button key={lens.name} plain label={lens.name} hotkey={index < 9 ? String(index + 1) : undefined} onPress={() => draft($, lens)} />,
                <Button key={`edit-${lens.name}`} dimColor label="edit" hotkey={EDIT_KEYS[index]} onPress={() => openNote($, lens.path)} />,
              ))}
              {shown.lensesOpen && (
                <Input key={`new-lens-${lenses.length}`} label="+ new prompt" placeholder="name it: Enter opens it in your editor" submitLabel="create" onSubmit={value => newLens($, value)} />
              )}
              {shown.lensesOpen && <Button key="all-lenses" plain dimColor label="open the prompts folder" hotkey="f" onPress={() => openFolder($, P.lenses)} />}
            </Box>

            <Box key="card-setup" {...CARD} borderColor="subtle">
              {head('setupOpen', `⚙️ Setup${gaps > 0 ? ` · ${gaps} to fix` : ''}`, undefined, 'setup-menu', 'z')}
              {shown.setupOpen && (
                <Box key="setup-body" flexDirection="column">
                  <Text bold>What goes into every draft</Text>
                  <Button key="about-me" plain label={`${mark(choice.aboutMe)} my context note and board`} hotkey="m" onPress={() => update($, picks, old => ({ ...old, aboutMe: !old.aboutMe }))} />
                  <Button key="pick-project" plain label={`${mark(choice.project)} the anchor's note`} onPress={() => update($, picks, old => ({ ...old, project: !old.project }))} />
                  <Button key="pick-today" plain label={`${mark(choice.today)} today's note, when there is one`} onPress={() => update($, picks, old => ({ ...old, today: !old.today }))} />
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
                    <Input key="set-dailyNote" label="daily note" value={settings.dailyNote ?? ''} placeholder="Daily/{{date:YYYY-MM-DD}}.md; empty reads Obsidian's setting, none turns it off" onSubmit={value => setOption($, 'dailyNote', value)} />
                    <Text key="daily-now" dimColor wrap="wrap">{today.path ? `Today's note: ${short(today.path)}${today.exists ? '' : ' (not made yet)'}` : 'No daily note: nothing about today is shown.'}</Text>
                    <Input key="set-dailyHeading" label="log heading in it" value={settings.dailyHeading ?? ''} placeholder="the end of the note" onSubmit={value => setOption($, 'dailyHeading', value)} />
                    <Input key="set-weekNote" label="week note" value={settings.weekNote ?? ''} placeholder="Weekly/{{date:YYYY-MM-DD}} - week.md (any day of the week); empty shows today's note instead" onSubmit={value => setOption($, 'weekNote', value)} />
                    {P.week && <Input key="set-weekHeading" label="workstreams under" value={settings.weekHeading ?? ''} placeholder="This week" onSubmit={value => setOption($, 'weekHeading', value)} />}
                    {P.week && <Input key="set-dayHeading" label="each day's log under" value={settings.dayHeading ?? ''} placeholder="### {{date:YYYY-MM-DD}}" onSubmit={value => setOption($, 'dayHeading', value)} />}
                    <Input key="set-taskFolder" label="new task folder" value={settings.taskFolder ?? ''} placeholder="the notes folder's root" onSubmit={value => setOption($, 'taskFolder', value)} />
                    {!settings.taskFolder?.trim() && <Text key="task-folder-unset" dimColor wrap="wrap">⚠ No task folder set: new task notes go to the notes folder's root.</Text>}
                    <Input key="set-taskColumn" label="new task column" value={settings.taskColumn ?? ''} placeholder="the anchor's column, else the board's first" onSubmit={value => setOption($, 'taskColumn', value)} />
                    <Text key="task-template" wrap="wrap">{`✓ task template: ${taskTemplate === 'yours' ? short(P.taskTemplate) : 'built-in (Outcome, Verify, Journal, Agent Log)'}`}</Text>
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
                    <Input key="set-lensPicks" label="prompts to list first" value={settings.lensPicks ?? ''} placeholder="all of them" onSubmit={value => setOption($, 'lensPicks', value)} />
                    <Button key="set-band" plain label={`${mark(settings.band !== 'no')} a line above the prompt: anchor and what's next`} onPress={() => setOption($, 'band', settings.band === 'no' ? 'yes' : 'no')} />
                    <Button key="set-tellChat" plain label={`${mark(settings.tellChat !== 'no')} tell each chat its anchor (so "update the task note" needs no path)`} onPress={() => setOption($, 'tellChat', settings.tellChat === 'no' ? 'yes' : 'no')} />
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
