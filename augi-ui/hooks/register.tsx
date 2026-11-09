import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { Card, Chat, Lens, Made, MadeArtifact, PadEntry, Picks, Pin, PinKind, Question, Session, SetupItem, Swatch, Trail, View } from '../types'

// Everything machine-specific comes from the settings: the manifest's userConfig,
// set at install, in /config, or from the pane's setup menu.
export type Options = {
  notesFolder: string
  editor: string
  editorCommand: string
  contextNote: string
  boardNote: string
  lensFolder: string
  logFolder: string
  // The old name for logFolder, still read.
  trailFolder: string
  lensPicks: string
}
export type Paths = {
  home: string
  notes: string
  context: string
  board: string
  lenses: string
  trails: string
  claudeDir: string
  codexDir: string
  editor: string
  editorCommand: string
  picks: string[]
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
    claudeDir: `${home}/.claude/projects`,
    codexDir: `${home}/.codex/sessions`,
    editor: options.editor || 'obsidian',
    editorCommand: options.editorCommand ?? '',
    picks: (options.lensPicks ?? '').split(',').map(one => one.trim()).filter(Boolean),
  }
}

let P: Paths = resolvePaths({}, '')
// Sets the paths the module works from; register calls it once HOME is known.
export const configure = (paths: Paths): void => {
  P = paths
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
// Augi's face (an otter, for old times' sake) on surfaces that draw Svg (desktop, editor, phone); the terminal skips it.
const AUGI_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="14" cy="16" r="7" fill="#8a5a3b"/><circle cx="50" cy="16" r="7" fill="#8a5a3b"/><circle cx="14" cy="16" r="3.5" fill="#5e3b25"/><circle cx="50" cy="16" r="3.5" fill="#5e3b25"/><circle cx="32" cy="34" r="24" fill="#a06a45"/><ellipse cx="32" cy="43" rx="15" ry="11" fill="#e8cfb0"/><circle cx="23" cy="30" r="3.2" fill="#2b1a10"/><circle cx="41" cy="30" r="3.2" fill="#2b1a10"/><circle cx="24" cy="29" r="1" fill="#fff"/><circle cx="42" cy="29" r="1" fill="#fff"/><ellipse cx="32" cy="38" rx="4.5" ry="3" fill="#2b1a10"/><path d="M32 41v3M28 46q4 3 8 0" stroke="#2b1a10" stroke-width="1.5" fill="none" stroke-linecap="round"/><path d="M20 41l-9-2M20 44l-9 1M44 41l9-2M44 44l9 1" stroke="#5e3b25" stroke-width="1" stroke-linecap="round"/></svg>'
const DRAFT_HEAD = 'Context for this chat:'
const HEAD_BYTES = '65536'
// Each chat session's anchor, as [session id, note] pairs, oldest first. The anchor
// belongs to the session: a new one opens with none; a reload or a resume keeps it.
const ANCHORS_KEY = 'anchors'
const ANCHORS_KEPT = 200
// The title a chat gets in ## Chats when anchoring records it, until a draft names it.
export const CHAT_PLACEHOLDER = 'chat'
// The setup steps someone chose to skip; /augi setup clears it.
const SKIPPED_KEY = 'skipped'
export const EMPTY_TRAIL: Trail = { questions: [], chats: [], saved: [], pins: [] }


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
    trail: EMPTY_TRAIL,
    pinPaths: {},
    hits: [],
    lenses: [],
    sessions: [],
    loadedAt: 0,
  } as Swatch,
  { shape: 'swatch-10' },
)
const picks = atom(
  { plugin: 'augi-ui', key: 'picks' } as const,
  { aboutMe: true, openAugi: true, project: true } as Picks,
  { shape: 'picks-3' },
)
const view = atom(
  { plugin: 'augi-ui', key: 'view' } as const,
  { anchorsOpen: false, contextOpen: false, sessionsOpen: false, lensesOpen: false, showHidden: false, setupOpen: false, inOpen: true, doneOpen: false } as View,
  { shape: 'view-4' },
)
const made = atom(
  { plugin: 'augi-ui', key: 'made' } as const,
  { session: '', artifacts: [] } as Made,
  { shape: 'made-1' },
)

// This chat's scratchpad, newest first: session state, so each chat has its own and it
// outlives a redraw or a reload of this module, but not the session.
const pad = atom({ plugin: 'augi-ui', key: 'pad' } as const, [] as PadEntry[], { shape: 'pad-1' })

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

// The outcome: `outcome:` in the frontmatter, else an `Outcome:` line in the body
// (`**Outcome:**` too). An empty frontmatter value falls through to the body.
export const outcomeFrom = (text: string): string => {
  const front = text.match(/^---\n([\s\S]*?)\n---/)
  const fromFront = (front?.[1] ?? '').match(/^outcome:[ \t]*(?:>-?[ \t]*\n[ \t]+)?(.+)$/m)?.[1] ?? ''
  const body = front ? text.slice(front[0].length) : text
  const fromBody = body.match(/^[ \t]*(?:\*\*)?outcome(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?[ \t]*(.+)$/im)?.[1] ?? ''

  return (fromFront.trim() || fromBody.trim()).replace(/^"|"$/g, '')
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
export const nextFrom = (text: string): string[] => {
  const tasks: string[] = []
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
      tasks.push(plain(task[1]))
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
      current.tasks.push(plain(task[1]))
    }
  }

  return cards
}

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
  `Human note: [[${note}]]. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Hidden chats stay listed, marked \`(hidden)\`.`
const oldLogIntro = (note: string): string =>
  `Project: [[${note}]], embedded there under # Agent Log. Context lines are \`- [x] [[note]] · why it matters\` (ticked goes into the next draft). Hidden chats stay listed, marked \`(hidden)\`.`

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
    lines.push(line === oldLogIntro(note) ? logIntro(note) : line)
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

const CHAT_LINE = /^- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) · (.*) · `([^`]+)`( \(hidden\))?$/

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
        chats.push({ when: found[1], title: found[2], id: found[3], hidden: Boolean(found[4]) })
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

// Adds a question at the top of ## Questions, newest first. An older trail without
// the section gets it above ## Chats.
export const addQuestion = (text: string, one: Question): string => {
  if (!one.text.trim()) {
    return text
  }
  const lines = text.split('\n')
  const at = lines.findIndex(line => line.trim() === '## Questions')
  if (at >= 0) {
    let first = at + 1
    while (lines[first] === '') {
      first += 1
    }
    if (QUESTION_LINE.test(lines[first] ?? '')) {
      lines.splice(first, 0, questionLine(one))
    } else {
      lines.splice(at + 1, first - at - 1, '', questionLine(one), '')
    }

    return lines.join('\n')
  }
  const chats = lines.findIndex(line => line.trim() === '## Chats')
  if (chats >= 0) {
    lines.splice(chats, 0, '## Questions', '', questionLine(one), '')

    return lines.join('\n')
  }

  return `${text.trimEnd()}\n\n## Questions\n\n${questionLine(one)}\n`
}

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

// What "link" copies: a note's full path, a file's path, a session's resume command, a URL.
export const pinLink = (pin: Pin, path = ''): string => {
  if (pin.kind === 'note') {
    return path || `[[${pin.target}]]`
  }

  return pin.kind === 'session' ? resumeOf(pin.target) : pin.target
}

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

export const contextItems = (pins: Pin[], choice: Picks, anchor: Anchor | null): ContextItem[] => {
  const items: ContextItem[] = []
  if (anchor && choice.project) {
    items.push({ key: 'in-note', label: anchor.note, path: anchor.notePath, kind: 'note' })
    items.push({ key: 'in-trail', label: `Trail - ${anchor.note}`, path: anchor.trail, kind: 'note' })
  }
  for (const pin of anchor ? pins.filter(one => one.on) : []) {
    items.push({ key: `in-pin-${pin.kind}-${pin.target}`, label: pin.label, path: pin.kind === 'note' ? pin.path ?? '' : pin.target, kind: 'pin' })
  }
  if (choice.aboutMe) {
    items.push({ key: 'in-context', label: noteName(P.context), path: P.context, kind: 'note' })
    items.push({ key: 'in-board', label: noteName(P.board), path: P.board, kind: 'note' })
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
  `- ${chat.when} · ${chat.title.replace(/·|`/g, ' ').trim()} · \`${chat.id}\`${chat.hidden ? ' (hidden)' : ''}`

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

export const setHidden = (text: string, id: string, hidden: boolean): string =>
  text
    .split('\n')
    .map(line => {
      const found = line.match(CHAT_LINE)

      return found && found[3] === id ? chatLine({ when: found[1], title: found[2], id, hidden }) : line
    })
    .join('\n')

// Names a chat recorded with the placeholder title; a real title stays as it is.
export const retitleChat = (text: string, id: string, title: string): string =>
  text
    .split('\n')
    .map(line => {
      const found = line.match(CHAT_LINE)

      return found && found[3] === id && found[2].trim() === CHAT_PLACEHOLDER
        ? chatLine({ when: found[1], title, id, hidden: Boolean(found[4]) })
        : line
    })
    .join('\n')

// The session → anchor pairs with this session's set (or dropped, for null), most
// recent last, the oldest let go past `kept`.
export const withAnchor = (pairs: [string, string][], id: string, note: string | null, kept = ANCHORS_KEPT): [string, string][] => {
  const others = pairs.filter(([one]) => one !== id)

  return (note ? [...others, [id, note] as [string, string]] : others).slice(-kept)
}

// What the store holds under the anchors key, read defensively.
export const anchorPairs = (value: unknown): [string, string][] =>
  Array.isArray(value)
    ? value.filter((one): one is [string, string] => Array.isArray(one) && typeof one[0] === 'string' && typeof one[1] === 'string')
    : []

// This session's published artifacts with one more: a URL already listed takes the new title.
export const withArtifact = (list: MadeArtifact[], one: MadeArtifact): MadeArtifact[] =>
  list.some(old => old.url === one.url) ? list.map(old => (old.url === one.url ? one : old)) : [...list, one]

// The anchor's pinned artifacts for its card, newest first.
export const anchorArtifacts = (pins: Pin[], limit = 3): Pin[] => pins.filter(pin => pin.kind === 'artifact').reverse().slice(0, limit)

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

// `pins` are the anchor's pinned context, a note's `path` filled where it was found.
export const buildDraft = (
  pins: Pin[],
  choice: Picks,
  lens: Lens | null,
  typed: string,
  anchor: Anchor | null = null,
): string => {
  const lines = [DRAFT_HEAD]

  if (anchor && choice.project) {
    const where = anchor.notePath ? `"${anchor.notePath}"` : `[[${anchor.note}]] (find it in the vault)`
    lines.push(`- Project: ${where}, my note (outcome, Next, Journal), and its Agent Log "${anchor.trail}": the context, questions, chats and notes saved on it so far. Read both.`)
  }
  for (const pin of anchor ? pins.filter(one => one.on) : []) {
    lines.push(pinDraftLine(pin))
  }
  if (choice.aboutMe) {
    lines.push(`- About me and my projects: read "${P.context}" and "${P.board}".`)
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
async function loadAnchor($: EngineInterface, cards: Card[]): Promise<Pick<Swatch, 'anchor' | 'notePath' | 'outcome' | 'next' | 'trail' | 'pinPaths' | 'hits'>> {
  const chosen = await sessionAnchor($)
  const anchor = cards.find(card => card.note === chosen) ?? null
  if (!anchor) {
    return { anchor: null, notePath: '', outcome: '', next: [], trail: EMPTY_TRAIL, pinPaths: {}, hits: [] }
  }
  const notePath = await notePathOf($, anchor.note)
  const noteText = notePath ? await $.fs.read(notePath) : ''
  await moveOldTrail($, anchor.note)
  const trail = parseTrail(await readOr($, trailPath(anchor.note)))
  const pinPaths: Record<string, string> = {}
  for (const pin of trail.pins.filter(one => one.kind === 'note')) {
    pinPaths[pin.target] = await notePathOf($, pin.target)
  }

  return {
    anchor,
    notePath,
    outcome: outcomeFrom(noteText),
    next: nextFrom(noteText),
    trail,
    pinPaths,
    hits: [],
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
  const cards = kanbanCards(await readOr($, P.board))
  const next: Swatch = {
    setup,
    vaults: [],
    found: {},
    skipped,
    season: anchorFrom(await readOr($, P.context)),
    cards,
    ...(await loadAnchor($, cards)),
    lenses: [],
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

  await update($, swatch, () => next)
}

// The note this chat session is anchored on, if any.
async function sessionAnchor($: EngineInterface): Promise<string | null> {
  const id = await $.session.id()

  return anchorPairs(await $.store.get(ANCHORS_KEY)).find(([one]) => one === id)?.[1] ?? null
}

async function setSessionAnchor($: EngineInterface, note: string | null): Promise<void> {
  const id = await $.session.id()
  if (id) {
    await $.store.set(ANCHORS_KEY, withAnchor(anchorPairs(await $.store.get(ANCHORS_KEY)), id, note))
  }
}

// Anchoring puts this chat in the anchor's ## Chats and pins what it already published, ticked.
async function chooseAnchor($: EngineInterface, card: Card): Promise<void> {
  await setSessionAnchor($, card.note)
  const { cards } = await read($, swatch)
  const found = await loadAnchor($, cards)
  await update($, swatch, old => ({ ...old, ...found }))
  await recordChat($, CHAT_PLACEHOLDER)
  for (const one of await sessionArtifacts($)) {
    await writeTrail($, card.note, text => addPin(text, artifactPin(one)))
  }
  await update($, view, old => ({ ...old, anchorsOpen: false }))
}

async function clearAnchor($: EngineInterface): Promise<void> {
  await setSessionAnchor($, null)
  await update($, swatch, old => ({ ...old, anchor: null, notePath: '', outcome: '', next: [], trail: EMPTY_TRAIL, pinPaths: {}, hits: [] }))
  await update($, view, old => ({ ...old, anchorsOpen: false }))
}

async function anchorNow($: EngineInterface): Promise<Anchor | null> {
  const { anchor, notePath } = await read($, swatch)

  return anchor ? { note: anchor.note, notePath, trail: trailPath(anchor.note) } : null
}

// `about` replaces what's typed in the prompt box: a captured question, for one.
async function draft($: EngineInterface, lens: Lens | null, about?: string): Promise<void> {
  const { trail, pinPaths } = await read($, swatch)
  const choice = await read($, picks)
  const typed = about ?? typedFrom((await $.prompt.read()).text)

  await $.prompt.fill({ text: buildDraft(withPaths(trail.pins ?? [], pinPaths), choice, lens, typed, await anchorNow($)), mode: 'replace' })
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

async function pinTyped($: EngineInterface, value: string): Promise<void> {
  const { anchor } = await read($, swatch)
  const pin = pinFrom(value, P.notes, P.home)
  if (!anchor || !pin) {
    return
  }
  if ((pin.kind === 'file' || pin.kind === 'session') && !(await $.fs.exists(pin.target))) {
    await $.ui.toast(`Not found: ${pin.target}`)
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
  const { trail } = await read($, swatch)
  const hits: Pin[] = []
  for (const path of paths) {
    const name = baseName(path).replace(/\.md$/, '')
    const isPinned = (trail.pins ?? []).some(one => one.kind === 'note' && one.target === name)
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
    return copyText($, pinLink(pin), 'A session reopens in a terminal. Copied')
  }
  if (pin.kind === 'file' && pin.target.endsWith('.md')) {
    return openNote($, pin.target)
  }
  await launch($, ['open', pin.target])
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
    await $.process.run(['mkdir', '-p', P.trails])
    await $.fs.write(path, text)
  }
  await update($, swatch, current => (current.anchor?.note === note ? { ...current, trail: parseTrail(text) } : current))
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

  await writeTrail($, note, text => retitleChat(addChat(text, { id, when, title, hidden: false }), id, title))
}

// Parks a question or side quest on the anchor's trail without leaving the step you're in.
async function capture($: EngineInterface, text: string): Promise<void> {
  const { anchor } = await read($, swatch)
  if (!text.trim()) {
    return
  }
  if (!anchor) {
    await $.ui.toast('Pick an anchor first: a captured question goes in its Agent Log.')

    return
  }
  const when = stamp(await $.clock.now())

  await writeTrail($, anchor.note, old => addQuestion(old, { when, text: text.trim(), done: false }))
}

// Keeps a note to self on this chat's scratchpad, newest first. Nothing is written.
async function jot($: EngineInterface, text: string): Promise<void> {
  if (!text.trim()) {
    return
  }
  const when = stamp(await $.clock.now())

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
    // A settings change reloads the module; redraw an open pane with the new paths.
    const starter = await $.store.get('starter')
    if (typeof starter === 'string') {
      await $.store.delete('starter')
      await writeStarter($, starter)
    }
    if ((await read($, swatch)).loadedAt > 0) {
      await load($)
    }

    return next(e)
  })

  // A /clear goes on in this process under a new session id with no session.start:
  // the pane drops the old session's anchor, as a new session opens with none.
  on('session.end', async ($, e, next) => {
    const ended = await next(e)
    if (e.reason === 'clear' || e.reason === 'resume') {
      await update($, swatch, old => ({ ...old, anchor: null, notePath: '', outcome: '', next: [], trail: EMPTY_TRAIL, pinPaths: {}, hits: [] }))
    }

    return ended
  })

  on('command.run', { command: 'augi' }, async ($, e) => {
    await ready($)
    if (e.args.trim() === 'setup') {
      await $.store.delete(SKIPPED_KEY)
      await update($, view, old => ({ ...old, setupOpen: true }))
    }
    await load($)
    await $.ui.open({ id: PANE, title: 'Augi' })

    return { text: 'Augi is open. Pick an anchor, tick its context, then draft a prompt or run a lens. It only drafts; you press enter.' }
  })

  // A draft from the pane, once sent, goes on the anchor's trail.
  on('prompt.submit', async ($, e, next) => {
    const title = chatTitleFrom(e.text)
    if (title) {
      try {
        await ready($)
        await recordChat($, title)
      } catch (error) {
        await $.ui.toast(`Augi couldn't record this chat: ${String(error)}`)
      }
    }

    return next(e)
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

  on('tool.call', { tool: 'Write' }, async ($, e, next) => {
    await ready($)
    const path = e.file_path
    const isNewNote = Boolean(P.notes) && path.startsWith(`${P.notes}/`) && path.endsWith('.md')
      && !path.startsWith(`${P.trails}/`) && !path.includes('/.') && !(await $.fs.exists(path))
    const ran = await next(e)
    try {
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
    const { setup, vaults, found, skipped, season, cards, anchor, notePath, outcome, next, trail, pinPaths, hits, lenses, sessions, loadedAt } = await read($, swatch)
    const choice = await read($, picks)
    const shown = await read($, view)
    const ownArtifacts = await sessionArtifacts($)
    const jotted = await read($, pad)
    const mark = (isOn: boolean): string => (isOn ? '☑' : '☐')
    const arrow = (isOpen: boolean): string => (isOpen ? '▾' : '▸')
    const flip = (key: keyof View) => () => update($, view, old => ({ ...old, [key]: !old[key] }))
    const chats = trail.chats.filter(one => shown.showHidden || !one.hidden)
    const hiddenCount = trail.chats.filter(one => one.hidden).length
    const pins = withPaths(trail.pins ?? [], pinPaths ?? {})
    const contextCount = contextItems(pins, choice, anchor ? { note: anchor.note, notePath, trail: trailPath(anchor.note) } : null).length
    const PIN_ICON: Record<Pin['kind'], string> = { note: '▤', file: '📄', session: '◦', artifact: '◇', url: '🔗' }
    const pinKey = (pin: Pin): string => `${pin.kind}-${pin.target}`
    const onPins = (change: (text: string) => string) => (anchor ? writeTrail($, anchor.note, change) : undefined)
    const openQuestions = trail.questions.filter(one => !one.done)
    const doneQuestions = trail.questions.filter(one => one.done)
    const gaps = setup.filter(item => !item.ok).length
    const short = (path: string): string => (path.startsWith(`${P.notes}/`) ? path.slice(P.notes.length + 1) : path.startsWith(P.home) ? `~${path.slice(P.home.length)}` : path)
    const step = setup[0]?.ok ? STEPS.find(one => !skipped.includes(one.field) && !setup.find(item => item.field === one.field)?.ok) : undefined
    const editorPicker = (
      <Box key="editor-picker" flexDirection="column">
        <Text>Open notes with:</Text>
        <Box flexDirection="row">
          {EDITORS.map(name => (
            <Button key={`editor-${name}`} plain label={`${P.editor === name ? '●' : '○'} ${name} `} onPress={() => setOption($, 'editor', name)} />
          ))}
        </Box>
        {P.editor === 'custom' && (
          <Input key="set-editorCommand" label="Command" placeholder="zed {file}" value={settings.editorCommand ?? ''} onSubmit={value => setOption($, 'editorCommand', value)} />
        )}
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
            <Button key={`vault-${index}`} label={`  ${vault.startsWith(P.home) ? `~${vault.slice(P.home.length)}` : vault}`} hotkey={String(index + 1)} onPress={() => setOption($, 'notesFolder', vault)} />
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
            <Button key={`found-${at}`} label={`  ${hit}`} hotkey={String(at + 1)} onPress={() => setOption($, step.field, hit)} />
          ))}
          {hits.length === 0 && <Text dimColor>Nothing like it found in your notes.</Text>}
          <Button key="use-starter" label={`  create a starter at ${step.starter}`} hotkey="v" onPress={() => useStarter($, step.field)} />
          <Input key={`set-${step.field}`} label="  or type a path" placeholder={step.starter} value="" onSubmit={value => setOption($, step.field, value)} />
          <Button key="skip-step" plain label="  skip for now" hotkey="e" onPress={() => skipStep($, step.field)} />
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
        <Box key="card-anchor" flexDirection="column" borderStyle="round" paddingX={1} marginBottom={1} borderColor="claude">
        <Text bold color="claude">⚓ Anchor</Text>
        <Button key="anchor-menu" label={`${arrow(shown.anchorsOpen)} ${anchor ? anchor.note : 'pick a project or task'}`} hotkey="a" onPress={flip('anchorsOpen')} />
        {shown.anchorsOpen && byColumn(cards).map(([column, list]) => (
          <Box key={`column-${column}`} flexDirection="column">
            <Text dimColor>  {column}</Text>
            {list.map(card => (
              <Button key={`card-${card.note}`} plain label={`    ${anchor?.note === card.note ? '●' : '○'} ${card.note}`} onPress={() => chooseAnchor($, card)} />
            ))}
          </Box>
        ))}
        {shown.anchorsOpen && anchor && <Button key="anchor-clear" plain label="    ○ no anchor" onPress={() => clearAnchor($)} />}
        {!anchor && <Text dimColor>🧭 Season: {season}</Text>}
        {anchor && anchor.detail && <Text dimColor>{anchor.detail}</Text>}
        {anchor && outcome && <Text>🎯 {outcome}</Text>}
        {anchor && [...(next ?? []), ...anchor.tasks.filter(task => !(next ?? []).includes(task))].map(task => <Text key={`task-${task}`}>☐ {task}</Text>)}
        {anchor && (
          <Box flexDirection="row">
            <Button key="open-note" label="🧠 Human Notes" hotkey="p" onPress={() => (notePath ? openNote($, notePath) : openByName($, anchor.note))} />
            <Text dimColor> | </Text>
            <Button key="open-trail" label="🤖 Agent Log" hotkey="k" onPress={() => writeTrail($, anchor.note, text => text).then(() => openNote($, trailPath(anchor.note)))} />
            {anchorArtifacts(pins).map(pin => (
              <Box key={`anchor-artifact-box-${pin.target}`} flexDirection="row">
                <Text dimColor> | </Text>
                <Button key={`anchor-artifact-${pin.target}`} label={`◇ ${pin.label}`} onPress={() => openPin($, pin)} />
              </Box>
            ))}
          </Box>
        )}
        {anchor && (
          <Input
            key={`capture-${trail.questions.length}`}
            label="? "
            placeholder="a question or side quest: Enter parks it in the Agent Log"
            submitLabel="park it"
            onSubmit={value => capture($, value)}
          />
        )}
        </Box>

        <Box key="card-context" flexDirection="column" borderStyle="round" paddingX={1} marginBottom={1} borderColor="remember">
        <Text bold color="remember">📎 Project context · {contextCount} going in</Text>
        <Text dimColor>Ticked items go into the drafted prompt. Claude reads them after you press enter.</Text>
        {!anchor && <Text dimColor>  Pick an anchor to keep context with it.</Text>}
        {!anchor && ownArtifacts.map(one => (
          <Text key={`made-${one.url}`} dimColor>{`  ◇ ${one.title}`}</Text>
        ))}
        {!anchor && ownArtifacts.length > 0 && <Text dimColor>  Artifacts this chat published: picking an anchor keeps them with it.</Text>}
        {anchor && (
          <Button key="pick-project" plain label={`  ${mark(choice.project)} 🧠 human and 🤖 agent notes for this ${anchor.note.startsWith('TASK') ? 'task' : 'project'}`} onPress={() => update($, picks, old => ({ ...old, project: !old.project }))} />
        )}
        {anchor && artifactsFirst(pins).map(pin => (
          <Box key={`pin-${pinKey(pin)}`} flexDirection="row">
            <Button key={`pin-tick-${pinKey(pin)}`} plain label={`  ${mark(pin.on)} ${PIN_ICON[pin.kind]} ${pin.label} `} onPress={() => onPins(text => setPinOn(text, pin, !pin.on))} />
            <Button key={`pin-open-${pinKey(pin)}`} label={pin.kind === 'session' ? 'resume' : 'open'} onPress={() => openPin($, pin)} />
            <Button key={`pin-link-${pinKey(pin)}`} label="link" onPress={() => copyText($, pinLink(pin, pin.path))} />
            <Button key={`pin-drop-${pinKey(pin)}`} label="unpin" onPress={() => onPins(text => removePin(text, pin))} />
          </Box>
        ))}
        {anchor && (
          <Input key="pin-search" label="  🔎 " placeholder="search notes to pin (OpenAugi): Enter searches" submitLabel="search" onSubmit={value => searchNotes($, value)} />
        )}
        {anchor && (hits ?? []).map(hit => (
          <Button key={`hit-${hit.target}`} plain label={`      + ${hit.label}`} onPress={() => pinIt($, anchor.note, hit)} />
        ))}
        {anchor && (hits ?? []).length > 0 && <Button key="hits-clear" plain label="      clear results" onPress={() => update($, swatch, old => ({ ...old, hits: [] }))} />}
        {anchor && (
          <Input key={`pin-paste-${pins.length}`} label="  ⧉ " placeholder="paste an artifact link, [[note]] or file path: Enter pins it" submitLabel="pin" onSubmit={value => pinTyped($, value)} />
        )}
        {anchor && (
          <Button key="sessions-menu" plain label={`  ${arrow(shown.sessionsOpen)} ◦ pin a recent coding session`} onPress={flip('sessionsOpen')} />
        )}
        {anchor && shown.sessionsOpen && sessions.length === 0 && <Text dimColor>      No recent coding sessions found.</Text>}
        {anchor && shown.sessionsOpen && sessions.map(one => (
          <Button
            key={`session-${one.path}`}
            plain
            label={`      + ${one.repo} · ${ago(loadedAt, one.mtimeMs)} · ${one.title}`}
            onPress={() => pinIt($, anchor.note, sessionPin(one))}
          />
        ))}
        <Button key="openaugi" plain label={`  ${mark(choice.openAugi)} also let the chat search my notes with OpenAugi`} onPress={() => update($, picks, old => ({ ...old, openAugi: !old.openAugi }))} />
        {choice.aboutMe && <Text dimColor>  + your overall context (change it under Setup)</Text>}
        <Button key="draft" variant="primary" label="💬 draft a prompt with this context" hotkey="n" onPress={() => draft($, null)} />
        </Box>

        <Box key="card-trail" flexDirection="column" borderStyle="round" paddingX={1} marginBottom={1} borderColor="success" display={anchor ? 'flex' : 'none'}>
        {anchor && <Text bold color="success">🌱 On this {anchor.note.startsWith('TASK') ? 'task' : 'project'}</Text>}
        {anchor && openQuestions.map(one => (
          <Box key={`question-${one.when}-${one.text}`} flexDirection="row">
            <Button key={`ask-${one.when}`} plain label={`  ? ${one.text}`} onPress={() => draft($, null, one.text)} />
            <Button key={`done-${one.when}`} label="done" onPress={() => writeTrail($, anchor.note, text => setDone(text, one, true))} />
          </Box>
        ))}
        {anchor && doneQuestions.length > 0 && (
          <Button key="show-done" plain label={`  ${arrow(shown.doneOpen)} done questions (${doneQuestions.length})`} onPress={flip('doneOpen')} />
        )}
        {anchor && shown.doneOpen && doneQuestions.map(one => (
          <Box key={`question-${one.when}-${one.text}`} flexDirection="row">
            <Text dimColor>{`    ✓ ${one.text}`}</Text>
            <Button key={`reopen-${one.when}`} label="reopen" onPress={() => writeTrail($, anchor.note, text => setDone(text, one, false))} />
          </Box>
        ))}
        {anchor && trail.saved.map(name => (
          <Button key={`saved-${name}`} plain label={`  ▤ ${name}`} onPress={() => openByName($, name)} />
        ))}
        {anchor && chats.map(chat => (
          <Box key={`chat-${chat.id}`} flexDirection="row">
            <Button key={`chat-${chat.id}`} plain label={`  ◦ ${chat.when.slice(5)} ${chat.title}${chat.hidden ? ' (hidden)' : ''}`} onPress={() => copyResume($, chat)} />
            <Button
              key={`hide-${chat.id}`}
              label={chat.hidden ? 'show' : 'hide'}
              onPress={() => writeTrail($, anchor.note, text => setHidden(text, chat.id, !chat.hidden))}
            />
          </Box>
        ))}
        {anchor && trail.saved.length === 0 && chats.length === 0 && trail.questions.length === 0 && <Text dimColor>  Nothing yet. Questions you park, chats you start and notes they save show up here.</Text>}
        {anchor && hiddenCount > 0 && (
          <Button key="show-hidden" label={`${shown.showHidden ? 'hide' : 'show'} hidden chats (${hiddenCount})`} hotkey="x" onPress={flip('showHidden')} />
        )}
        </Box>

        <Box key="card-pad" flexDirection="column" borderStyle="round" paddingX={1} marginBottom={1} borderColor="subtle">
        <Text bold>📝 Scratchpad · this chat</Text>
        <Input
          key={`pad-input-${jotted.length}`}
          label="  "
          placeholder="a note to self: Enter keeps it here for this chat"
          submitLabel="keep"
          rows={2}
          onSubmit={value => jot($, value)}
        />
        {jotted.map((one, at) => (
          <Box key={`pad-row-${at}`} flexDirection="row">
            <Text key={`pad-text-${at}`} dimColor>{`  ${one.when.slice(11, 16)} ${one.text} `}</Text>
            {anchor && notePath && <Button key={`pad-save-${at}`} label="→ notes" onPress={() => saveToNotes($, [one])} />}
            <Button key={`pad-drop-${at}`} label="drop" onPress={() => unjot($, [one])} />
          </Box>
        ))}
        {anchor && notePath && jotted.length >= 2 && (
          <Button key="pad-save-all" plain label="  → save all to Human Notes" onPress={() => saveToNotes($, jotted)} />
        )}
        {!(anchor && notePath) && <Text dimColor>{anchor ? `  No note found for ${anchor.note}: entries stay here.` : '  Pick an anchor to save these to its Human Notes.'}</Text>}
        </Box>
        <Box key="card-lenses" flexDirection="column" borderStyle="round" paddingX={1} marginBottom={1} borderColor="suggestion">
        <Button key="lens-menu" label={`${arrow(shown.lensesOpen)} 🔍 Lenses (saved prompts) · ${lenses.length}`} hotkey="l" onPress={flip('lensesOpen')} />
        {shown.lensesOpen && lenses.map((lens, index) => (
          <Box key={lens.name} flexDirection="row">
            <Button key={lens.name} label={`  ${lens.name}`} hotkey={index < 9 ? String(index + 1) : undefined} onPress={() => draft($, lens)} />
            <Button key={`edit-${lens.name}`} label="edit" hotkey={EDIT_KEYS[index]} onPress={() => openNote($, lens.path)} />
          </Box>
        ))}
        {shown.lensesOpen && (
          <Input key={`new-lens-${lenses.length}`} label="  + new lens " placeholder="name it: Enter opens it in your editor" submitLabel="create" onSubmit={value => newLens($, value)} />
        )}
        {shown.lensesOpen && <Button key="all-lenses" label="  open the lens folder" hotkey="f" onPress={() => openFolder($, P.lenses)} />}
        </Box>

        <Box key="card-setup" flexDirection="column" borderStyle="round" paddingX={1} marginBottom={1} borderColor="subtle">
        <Button key="setup-menu" label={`${arrow(shown.setupOpen)} ⚙️ Setup · overall context${gaps > 0 ? ` (${gaps} to fix)` : ''}`} hotkey="z" onPress={flip('setupOpen')} />
        {shown.setupOpen && <Text bold>  Overall context</Text>}
        {shown.setupOpen && (
          <Button key="about-me" plain label={`  ${mark(choice.aboutMe)} put my context note and board in every chat`} hotkey="m" onPress={() => update($, picks, old => ({ ...old, aboutMe: !old.aboutMe }))} />
        )}
        {shown.setupOpen && (
          <Box flexDirection="row">
            <Button key="edit-context" label="✏️ edit my context" hotkey="s" onPress={() => openNote($, P.context)} />
            <Button key="open-board" label="📋 open the board" hotkey="d" onPress={() => openNote($, P.board)} />
          </Box>
        )}
        {shown.setupOpen && <Text bold>  Files and folders</Text>}
        {shown.setupOpen && setup.map(item => (
          <Box key={`setup-${item.field}`} flexDirection="column">
            <Text>{`  ${item.ok ? '✓' : '✗'} ${item.label}: ${item.path.startsWith(`${P.notes}/`) ? item.path.slice(P.notes.length + 1) : item.path}`}</Text>
            {!item.ok && item.starter && <Button key={`starter-${item.field}`} label="    create a starter" onPress={() => makeStarter($, item.field)} />}
            <Input
              key={`set-${item.field}`}
              label="    path"
              value={String(settings[item.field as keyof Options] ?? '')}
              onSubmit={value => setOption($, item.field, value)}
            />
          </Box>
        ))}
        {shown.setupOpen && (
          <Input key="set-logFolder" label="  agent log folder" value={settings.logFolder || settings.trailFolder || ''} placeholder="OpenAugi/Agent-Logs" onSubmit={value => setOption($, 'logFolder', value)} />
        )}
        {shown.setupOpen && (
          <Input key="set-lensPicks" label="  lenses to list first" value={settings.lensPicks ?? ''} placeholder="all of them" onSubmit={value => setOption($, 'lensPicks', value)} />
        )}
        {shown.setupOpen && editorPicker}
        {shown.setupOpen && <Button key="start-over" plain label="  ← change the notes folder and run setup again" onPress={() => startOver($)} />}
        </Box>
        <Button key="refresh" label="↻ refresh" hotkey="r" onPress={() => load($)} />
      </Box>
    )
  })
}
