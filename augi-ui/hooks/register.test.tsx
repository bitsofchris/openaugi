import type { On } from 'claude-code'
import type { Pin } from '../types'
import { expect, test } from 'claude-code/testing'

import { parseUsage, usageName, usageReport } from './register'
import { addDailyLine, dailyNotePath, dailyPattern, dailyTemplatePath, fillTemplate, formatDate, openBoxes } from './register'
import { bandLine } from './register'
import { boardSync, entryBoxes, linkGoal, markChatDone, weekSyncDraft, newWeekNote, tickCard, tickGoal, weekDays, weekGoals, weekNotePaths } from './register'
import { promptTitle } from './register'
import { anchorBrief, anchorFromMessages, anchorNews, unanchoredBrief } from './register'
import { openArgv } from './register'
import { anchorCard, findInJiraDraft, insertCard, jiraBrowseUrl, jiraFrom, jiraKeyFrom, jiraServerFrom, jiraUpdateDraft, newTaskDraft, parseNewTask, TASK_TEMPLATE, taskNoteName, taskNoteText, taskTemplateState, withJira } from './register'
import { addChat, addJournalEntry, anchorFromDraft, anchorPairs, artifactsFirst, retitleChat, withAnchor, logName, nextFrom, addPin, parsePin, pinFrom, pinLine, removePin, configure, editorArgv, obsidianConfigs, resolvePaths, setupItems, vaultsFrom, anchorFrom, buildDraft, chatTitleFrom, descriptionFrom, kanbanCards, lensFileName, lensNames, newLensSkeleton, outcomeFrom, parseTrail, tidyTrail, trailPath, toggled, typedFrom } from './register'
import { isMovedLog, markMoved, splitLog, withOldLog } from './register'
import { jiraKeyIn, offerAsk, offerCandidates, parseOffer } from './register'
import { addDays, dueRecalls, EXPLAIN_LENS, EXPLAIN_PROMPT, explainLensFile, GRADER_SYSTEM, gradeAsk, isKnown, nextRung, parseGrade, parseRecall, parseRecallRows, RECALL_EVERY_MS, RECALL_PROMPT, recallAsk, recallLensFile, recordRecall, rungAfter } from './register'
import { lensBody, setChatState, STATE_EVERY_MS, STATE_PROMPT, stateAsk, stateLensFile, stateLines } from './register'

const PANE = {
  plugin: 'augi-ui',
  component: 'Pane',
  requestId: 'augi',
  props: {
    title: 'Augi',
    isFocused: true,
    bodyColumns: 60,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 40 },
    view: {},
  },
} as const

const row = (fields: object): string => JSON.stringify(fields)
// The plain context draft is the first row of the Saved prompts card, which starts folded.
type Mounted = { find: (query: { key: string }) => Promise<unknown>; press: (query: { key: string }) => Promise<unknown> }
const pressDraft = async (ui: Mounted): Promise<void> => {
  if (!(await ui.find({ key: 'draft' }))) {
    await ui.press({ key: 'lens-menu' })
  }
  await ui.press({ key: 'draft' })
}
// An Agent Log with one more open question on top, as an agent would write it there.
const withQuestion = (log: string, when: string, text: string): string => log.replace('## Questions\n', `## Questions\n\n- [ ] ${when} · ${text}`)
// An Agent Log file as the pane wrote it while the log was a file of its own.
const oldLogFile = (note: string, day: string): string =>
  [
    '---', 'type: document',
    `description: The Agent Log for ${note}: context worth keeping, questions, chats started on it and notes saved from them. Agents write here; ${note} is the person's own.`,
    `created: ${day}`, '---', '',
    `Human note: [[${note}]]. Context lines are \`- [[note]] · why it matters\`: each is pinned and goes into the next draft; one ticked off (\`- [x]\`) was unpinned. Each chat is one line: when it started, its goal, its id; one that is finished stays listed, marked \`(done)\`.`,
    '', '## Context', '', '## Questions', '', '## Chats', '', '## Saved', '',
  ].join('\n')
const human = row({ type: 'user', entrypoint: 'cli', cwd: '/r/garden', message: { content: 'fix the sprinkler' } })
const OPTIONS = {
  notesFolder: '/v',
  contextNote: 'AGENT/Slowly Changing Context.md',
  boardNote: 'Kanban.md',
  lensFolder: 'lenses',
  logFolder: 'Notes',
  lensPicks: 'execution-focus, echoes',
  editor: 'obsidian',
  editorCommand: '',
  // Off here so the tests that count what the pane runs see only that; the usage tests turn it on.
  trackUsage: 'no',
}
configure(resolvePaths(OPTIONS, '/Users/t'))

const ALL_ON = { aboutMe: true, openAugi: true, project: true, today: true }
const ANCHOR = { note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md' }
const SESSION_PIN = { on: true, kind: 'session', target: '/p/a.jsonl', label: 'garden · fix the sprinkler' } as const

test('the anchor is the first bold line of the season section', async () => {
  const text = '## 2. Lanes\n**Not this.**\n## 3. This season\n> quote\n**Primary: the garden.** more\n## 4. Weekly'
  expect(anchorFrom(text)).toBe('Primary: the garden.')
})

test('a description reads as folded or plain', async () => {
  expect(descriptionFrom('name: a\ndescription: >-\n  "Have I?" yes\nscope: b')).toBe('"Have I?" yes')
  expect(descriptionFrom('description: Gather everything\n')).toBe('Gather everything')
})

test('the draft carries what is pinned and switched on, and leaves out the rest', async () => {
  const full = buildDraft([SESSION_PIN], ALL_ON, null, '', ANCHOR)
  expect(full.includes('/p/a.jsonl')).toBe(true)
  expect(full.includes('Slowly Changing Context')).toBe(true)
  expect(full.includes('Search my notes in "/v"')).toBe(true)
  expect(full.includes('No walls of text')).toBe(true)
  expect(full.endsWith('Instruction: ')).toBe(true)

  const bare = buildDraft([{ ...SESSION_PIN, on: false }], { aboutMe: false, openAugi: false, project: false, today: false }, null, 'hi', ANCHOR)
  expect(bare.includes('/p/a.jsonl')).toBe(false)
  expect(bare.includes('Slowly Changing Context')).toBe(false)
  expect(bare.includes('Search my notes')).toBe(false)
  expect(bare.endsWith('Instruction: hi')).toBe(true)
})

test('typed words survive, an earlier draft does not', async () => {
  expect(typedFrom('why is this slow')).toBe('why is this slow')
  expect(typedFrom(buildDraft([SESSION_PIN], ALL_ON, null, 'keep me'))).toBe('keep me')
})

test('the menu is the picks in order; with none, every lens', async () => {
  const files = ['zeta', 'echoes', 'execution-focus', 'weekly-reflection']
  expect(lensNames(files)).toEqual(['execution-focus', 'echoes'])
  configure(resolvePaths({ ...OPTIONS, lensPicks: '' }, '/Users/t'))
  expect(lensNames(files)).toEqual(['echoes', 'execution-focus', 'weekly-reflection', 'zeta'])
  configure(resolvePaths(OPTIONS, '/Users/t'))
})

test('a new lens is a file-safe name and a skeleton in the lens contract', async () => {
  expect(lensFileName('  check my tests.md ')).toBe('check-my-tests')
  expect(lensFileName('a/b: [[c]]#d')).toBe('a-b-c-d')
  expect(lensFileName('  ..  ')).toBe('')
  const text = newLensSkeleton('check-my-tests')
  expect(text.startsWith('---\nname: check-my-tests\ndescription: ')).toBe(true)
  for (const line of ['trigger: on-demand', 'target: chat — answers in the conversation and writes nothing.', 'kind: personal', '# check-my-tests', 'Paste the prompt here.']) {
    expect(text.split('\n').includes(line)).toBe(true)
  }
  expect(text.includes('scope: ')).toBe(true)
})

test('toggled adds and removes', async () => {
  expect(toggled(['a'], 'b')).toEqual(['a', 'b'])
  expect(toggled(['a', 'b'], 'a')).toEqual(['b'])
})

const BOARD_TEXT = [
  '---', 'kanban-plugin: board', '---', '',
  '## Ideas', '',
  '- [ ] [[2022-01-04 - Sketch]] ^abc123',
  '\t[[Some related note]]',
  '- [ ] Plain card with no link',
  '\t- [ ] not a task of any card',
  '', '## Work', '',
  '- [ ] [[PMOC - Garden|the garden]] — the main project',
  '\t- [ ] Plant the beds',
  '\t- [x] done already',
  '\t- [ ] [[PMOC - Birdhouse]]',
  '', '***', '', '## Archive', '', '- [ ] [[Archived thing]]',
].join('\n')

test('anchors are the open linked cards above the archive, with their open sub-items', async () => {
  const cards = kanbanCards(BOARD_TEXT)
  expect(cards.map(card => card.note)).toEqual(['2022-01-04 - Sketch', 'PMOC - Garden'])
  expect(cards[0]?.column).toBe('Ideas')
  expect(cards[0]?.tasks).toEqual([])
  expect(cards[1]?.detail).toBe("the main project")
  expect(cards[1]?.tasks).toEqual([{ text: 'Plant the beds', note: null }, { text: 'PMOC - Birdhouse', note: 'PMOC - Birdhouse' }])
})

const MANY_COLUMNS = [
  '---', '', 'kanban-plugin: board', '', '---', '',
  '## Doing', '',
  '- [ ] [[PMOC - Garden]] — the main project ^abc123',
  '\t- [ ] Plant the beds ^def456',
  '\t[[Some related note]]',
  '- [ ] [[PMOC - Birdhouse]]',
  '', '', '## Next', '',
  '- [ ] [[PMOC - Pond]]',
  '\t- [ ] [[Dig the hole]]',
  '', '', '***', '', '## Archive', '', '- [ ] [[PMOC - Garden]] old copy', '- [ ] [[Archived thing]]', '',
].join('\n')

// The board with the one inserted line taken out again, to prove nothing else moved.
const without = (text: string, line: string): string => {
  const lines = text.split('\n')
  lines.splice(lines.indexOf(line), 1)

  return lines.join('\n')
}

test('a new card goes after its anchor card and sub-items, else at the end of the first column, never in the archive', async () => {
  const card = '- [ ] [[Fix the gate]]'
  const after = insertCard(MANY_COLUMNS, 'Fix the gate', 'PMOC - Garden')
  expect(after.split('\n').indexOf(card)).toBe(MANY_COLUMNS.split('\n').indexOf('\t[[Some related note]]') + 1)
  expect(without(after, card)).toBe(MANY_COLUMNS)
  expect(kanbanCards(after).map(one => one.note)).toEqual(['PMOC - Garden', 'Fix the gate', 'PMOC - Birdhouse', 'PMOC - Pond'])
  expect(kanbanCards(after)[0]?.tasks.map(one => one.text)).toEqual(['Plant the beds'])

  const inNext = insertCard(MANY_COLUMNS, 'Fix the gate', 'PMOC - Pond')
  expect(inNext.split('\n').indexOf(card)).toBe(MANY_COLUMNS.split('\n').indexOf('\t- [ ] [[Dig the hole]]') + 1)
  expect(without(inNext, card)).toBe(MANY_COLUMNS)

  for (const anchor of [null, 'Archived thing', 'Not on the board']) {
    const first = insertCard(MANY_COLUMNS, 'Fix the gate', anchor)
    expect(first.split('\n').indexOf(card)).toBe(MANY_COLUMNS.split('\n').indexOf('- [ ] [[PMOC - Birdhouse]]') + 1)
    expect(without(first, card)).toBe(MANY_COLUMNS)
    expect(first.slice(first.indexOf('***'))).toBe(MANY_COLUMNS.slice(MANY_COLUMNS.indexOf('***')))
  }

  // With no anchor card to follow, a named column takes it at its end, whatever the case;
  // a column the board does not have falls back to the first.
  for (const into of ['Next', 'next ', 'NEXT']) {
    const named = insertCard(MANY_COLUMNS, 'Fix the gate', null, into)
    expect(named.split('\n').indexOf(card)).toBe(MANY_COLUMNS.split('\n').indexOf('\t- [ ] [[Dig the hole]]') + 1)
    expect(without(named, card)).toBe(MANY_COLUMNS)
    expect(kanbanCards(named).find(one => one.note === 'Fix the gate')?.column).toBe('Next')
  }
  expect(insertCard(MANY_COLUMNS, 'Fix the gate', null, 'Someday')).toBe(insertCard(MANY_COLUMNS, 'Fix the gate', null))
  // The archive is never a column to add to, even by name.
  expect(insertCard(MANY_COLUMNS, 'Fix the gate', null, 'Archive')).toBe(insertCard(MANY_COLUMNS, 'Fix the gate', null))
  // An anchor card on the board still wins over the named column.
  expect(insertCard(MANY_COLUMNS, 'Fix the gate', 'PMOC - Garden', 'Next')).toBe(after)

  // An empty first column takes the card under the blank line after its heading.
  const empty = '---\n\nkanban-plugin: board\n\n---\n\n## Todo\n\n\n\n## Doing\n\n- [ ] [[A]]\n'
  const filled = insertCard(empty, 'Fix the gate', null)
  expect(filled).toBe('---\n\nkanban-plugin: board\n\n---\n\n## Todo\n\n- [ ] [[Fix the gate]]\n\n\n## Doing\n\n- [ ] [[A]]\n')
  expect(kanbanCards(filled)[0]).toEqual({ column: 'Todo', note: 'Fix the gate', detail: '', tasks: [] })
})

test('an anchor is a top-level card, else a linked sub-item shaped as a card under its parent', async () => {
  const cards = kanbanCards(MANY_COLUMNS)
  expect(anchorCard(cards, 'PMOC - Pond')).toBe(cards[2])
  expect(anchorCard(cards, 'Dig the hole')).toEqual({ column: 'Next', note: 'Dig the hole', detail: 'under PMOC - Pond', tasks: [] })
  expect(anchorCard(cards, 'Plant the beds')).toBe(null)
  expect(anchorCard(cards, 'Archived thing')).toBe(null)
})

test('a typed task gives its title and any Jira key or issue link at either end', async () => {
  expect(parseNewTask('Fix the gate PROJ-123')).toEqual({ title: 'Fix the gate', jira: 'PROJ-123' })
  expect(parseNewTask('PROJ-123: Fix the gate')).toEqual({ title: 'Fix the gate', jira: 'PROJ-123' })
  expect(parseNewTask('Fix the gate (PROJ-123)')).toEqual({ title: 'Fix the gate', jira: 'PROJ-123' })
  expect(parseNewTask('Fix the gate — https://example.atlassian.net/browse/PROJ-7?focus=1')).toEqual({ title: 'Fix the gate', jira: 'PROJ-7' })
  expect(parseNewTask('  Fix the gate  ')).toEqual({ title: 'Fix the gate', jira: '' })
  expect(parseNewTask('Fix PROJ-1 then the gate')).toEqual({ title: 'Fix PROJ-1 then the gate', jira: '' })
  expect(taskNoteName('Gate: fix/oil [it] #now?')).toBe('Gate- fix-oil it now')
})

test('a task note fills the template: date, title, Up to the anchor or no Up line, and the Jira key under it', async () => {
  const linked = taskNoteText(TASK_TEMPLATE, 'Fix the gate', '2022-01-04', 'PMOC - Garden', 'PROJ-123')
  expect(linked.startsWith('Up: [[PMOC - Garden]]\nJira: PROJ-123\n\n### Outcome\n\n\n### Verify\n\n\n# Journal\n\n### 2022-01-04\n')).toBe(true)
  expect(linked.includes('Up: [[PMOC - Garden]]\n')).toBe(true)
  expect(linked.includes('### 2022-01-04')).toBe(true)
  // The note ends with the heading agents write under, empty.
  expect(linked.endsWith('### 2022-01-04\n\n\n# Agent Log\n')).toBe(true)
  expect(parseTrail(linked)).toEqual({ chats: [], pins: [], states: {} })
  expect(nextFrom(linked)).toEqual([])
  expect(outcomeFrom(linked)).toBe('')
  expect(jiraFrom(linked)).toBe('PROJ-123')

  const bare = taskNoteText(TASK_TEMPLATE, 'Fix the gate', '2022-01-04', null, '')
  expect(bare.includes('Up:')).toBe(false)
  expect(bare.includes('jira:')).toBe(false)
  expect(bare.startsWith('### Outcome\n')).toBe(true)

  expect(taskNoteText('# {{title}}\n', 'X', '2022-01-04', null, 'PROJ-1')).toBe('Jira: PROJ-1\n\n# X\n')
  expect(taskNoteText('---\njira: \n---\n', 'X', '2022-01-04', null, 'PROJ-1')).toBe('---\njira: PROJ-1\n---\n')
  expect(jiraFrom('---\njira: "PROJ-9"\n---\n')).toBe('PROJ-9')
  expect(jiraFrom('jira: PROJ-9\n')).toBe('')
})

test('a new-task draft asks the chat to fill in the note and its Agent Log section, and is titled as ours', async () => {
  const text = newTaskDraft('/v/Fix the gate.md', 'Fix the gate')
  expect(text.startsWith('Context for this chat:\n')).toBe(true)
  expect(text.includes('"/v/Fix the gate.md"')).toBe(true)
  expect(text.includes("under the note's Outcome and Verify headings")).toBe(true)
  expect(text.includes('no frontmatter')).toBe(true)
  expect(text.includes('under the note\'s "# Agent Log" heading, at its end')).toBe(true)
  expect(chatTitleFrom(text)).toBe('fill in the new task Fix the gate')
})

test('the outcome comes from the note frontmatter', async () => {
  expect(outcomeFrom('---\noutcome: "By the trip: it loads."\n---\nbody')).toBe('By the trip: it loads.')
  expect(outcomeFrom('no frontmatter\noutcome: no')).toBe('no')
  expect(outcomeFrom('---\noutcome: \nstatus: active\n---\n**Outcome:** the sprinkler serves a day in 1s\n')).toBe('the sprinkler serves a day in 1s')
  expect(outcomeFrom('#area/x\n\nOutcome: a mod I use daily.\n# Journal')).toBe('a mod I use daily.')
  expect(outcomeFrom('# Journal\nno outcome here')).toBe('')
  // A template that keeps the outcome under a heading wins over the frontmatter line.
  expect(outcomeFrom('---\noutcome: old\n---\n### Outcome\n\nThe gate latches.\n\n### Verify\n\nPush it.\n')).toBe('The gate latches.')
  expect(outcomeFrom('##### Outcome\n- [[The gate]] latches\n# Next')).toBe('The gate latches')
  expect(outcomeFrom('---\noutcome: kept\n---\n### Outcome\n\n### Verify\nPush it.\n')).toBe('kept')
})

// A task note as the user keeps it: theirs down to the Journal, with no Agent Log section yet.
const TASK_NOTE = 'Up: [[PMOC - Garden]]\n\n### Outcome\n\nThe gate closes.\n\n# Journal\n\n### 2022-01-04\n- hung it\n'

test('the Agent Log is a section at the end of the note: made on the first write, everything above it left byte for byte', async () => {
  expect(parseTrail(TASK_NOTE)).toEqual({ chats: [], pins: [], states: {} })
  expect(splitLog(TASK_NOTE).has).toBe(false)

  const pinned = addPin(TASK_NOTE, { on: true, kind: 'note', target: 'A', label: 'A' })
  expect(pinned).toBe(`${TASK_NOTE}\n# Agent Log\n\n- [[A]]\n`)
  // A note with no newline at its end, and an empty one, get the same section.
  expect(addPin(TASK_NOTE.trimEnd(), { on: true, kind: 'note', target: 'A', label: 'A' })).toBe(pinned)
  expect(addPin('', { on: true, kind: 'note', target: 'A', label: 'A' })).toBe('# Agent Log\n\n- [[A]]\n')

  // Chats go in a callout that is folded until opened, newest first, once each, under the pins.
  const one = addChat(pinned, { id: 's1', when: '2026-11-08 10:00', title: 'echoes: why slow', done: false })
  const two = addChat(one, { id: 's2', when: '2026-11-08 11:00', title: 'second', done: false })
  expect(addChat(two, { id: 's1', when: '2026-11-08 12:00', title: 'again', done: false })).toBe(two)
  expect(two).toBe(`${TASK_NOTE}\n# Agent Log\n\n- [[A]]\n\n> [!abstract]- Chats\n> - 2026-11-08 11:00 · second · \`s2\`\n> - 2026-11-08 10:00 · echoes: why slow · \`s1\`\n`)
  expect(parseTrail(two).chats.map(chat => chat.id)).toEqual(['s2', 's1'])
  expect(parseTrail(two).pins.map(pin => pin.target)).toEqual(['A'])
  // A chat filed on a note with nothing pinned: the callout is the whole section.
  expect(addChat(TASK_NOTE, { id: 's1', when: '2026-11-08 10:00', title: 'chat', done: false }).endsWith('\n\n# Agent Log\n\n> [!abstract]- Chats\n> - 2026-11-08 10:00 · chat · `s1`\n')).toBe(true)
})

test('what an agent appends anywhere in the section is kept; a pin it writes after the Chats callout is a pin, and the callout goes back to the end', async () => {
  const start = addChat(addPin(TASK_NOTE, { on: true, kind: 'note', target: 'A', label: 'A' }), { id: 's1', when: '2026-11-08 10:00', title: 'chat', done: false })
  // An agent appends at the end of the file, as agents do: a fact, a question, a pin.
  const appended = `${start}- the valve sticks below 4 degrees\n- [ ] 2026-11-08 10:05 · does the timer survive a power cut?\n- [[Valve Notes]] · the part numbers\n`
  expect(parseTrail(appended).pins.map(pin => pin.target)).toEqual(['A', 'Valve Notes'])
  expect(parseTrail(appended).chats.length).toBe(1)
  // The pane's next write moves them up beside the other lines and changes none of them.
  const next = addPin(appended, { on: true, kind: 'file', target: '/r/a.ts', label: 'a.ts' })
  expect(next.endsWith('# Agent Log\n\n- [[A]]\n- the valve sticks below 4 degrees\n- [ ] 2026-11-08 10:05 · does the timer survive a power cut?\n- [[Valve Notes]] · the part numbers\n- file: `/r/a.ts` · a.ts\n\n> [!abstract]- Chats\n> - 2026-11-08 10:00 · chat · `s1`\n')).toBe(true)
  expect(next.startsWith(TASK_NOTE)).toBe(true)
  // A level-1 heading after the section ends it: what follows is not the log's.
  const followed = `${start}\n# Scratch\n- [[Not a pin]]\n`
  expect(parseTrail(followed).pins.map(pin => pin.target)).toEqual(['A'])
  expect(addPin(followed, { on: true, kind: 'note', target: 'B', label: 'B' }).endsWith('> - 2026-11-08 10:00 · chat · `s1`\n\n# Scratch\n- [[Not a pin]]\n')).toBe(true)
})

test('a pin is a plain bullet and one that was unpinned is ticked off; each kind round-trips', async () => {
  const pins = [
    { on: true, kind: 'note', target: 'Sprinkler Benchmarks', label: 'Sprinkler Benchmarks' },
    { on: false, kind: 'session', target: '/p/a b.jsonl', label: 'garden · fix the sprinkler' },
    { on: true, kind: 'artifact', target: 'https://claude.ai/artifact/abc-1', label: 'Cache explainer' },
    { on: true, kind: 'url', target: 'https://example.com/x', label: 'https://example.com/x' },
    { on: false, kind: 'file', target: '/r/docs/ARCH.md', label: 'ARCH.md' },
  ] as const
  for (const pin of pins) {
    expect(parsePin(pinLine(pin))).toEqual(pin)
  }
  expect(pinLine(pins[0])).toBe('- [[Sprinkler Benchmarks]]')
  expect(pinLine(pins[1])).toBe('- [x] session: `/p/a b.jsonl` · garden · fix the sprinkler')
  expect(parsePin('- [[Note|alias]] · why it matters')).toEqual({ on: true, kind: 'note', target: 'Note', label: 'why it matters' })
  // An open box an agent wrote is a pin too; a question, or a sentence that only starts with a link, is not.
  expect(parsePin('- [ ] [[Note]]')?.on).toBe(true)
  expect(parsePin('- [ ] 2026-11-08 10:00 · a question')).toBe(null)
  expect(parsePin('- [[Note]] and [[Other]] disagree on the valve')).toBe(null)
  expect(parsePin('- the valve sticks below 4 degrees')).toBe(null)
})

test('a chat\'s state is kept under its own entry in the chat list: replaced for that chat, untouched for the others', async () => {
  const filed = addChat(addChat(addPin(TASK_NOTE, { on: true, kind: 'note', target: 'A', label: 'A' }), { id: 's1', when: '2026-11-08 10:00', title: 'first', done: false }), { id: 's2', when: '2026-11-08 11:00', title: 'second', done: false })
  const one = setChatState(filed, 's1', '2026-11-08 10:05', 'Doing: hanging the gate\nLast: bought hinges\nNext: fit them\nWaiting on me: no')
  expect(one).toBe(`${TASK_NOTE}\n# Agent Log\n\n- [[A]]\n\n> [!abstract]- Chats\n> - 2026-11-08 11:00 · second · \`s2\`\n> - 2026-11-08 10:00 · first · \`s1\`\n>   - as of 2026-11-08 10:05\n>   - Doing: hanging the gate\n>   - Last: bought hinges\n>   - Next: fit them\n>   - Waiting on me: no\n`)
  expect(parseTrail(one).states).toEqual({ s1: { when: '2026-11-08 10:05', text: 'Doing: hanging the gate\nLast: bought hinges\nNext: fit them\nWaiting on me: no' } })
  // The chats and pins read as before: a state is not a chat and not a pin.
  expect(parseTrail(one).chats.map(chat => chat.id)).toEqual(['s2', 's1'])
  expect(parseTrail(one).pins.map(pin => pin.target)).toEqual(['A'])

  // The other chat on the task has its own, and neither writes over the other.
  const two = setChatState(one, 's2', '2026-11-08 11:30', 'Doing: painting\nNext: second coat')
  expect(parseTrail(two).states).toEqual({
    s1: { when: '2026-11-08 10:05', text: 'Doing: hanging the gate\nLast: bought hinges\nNext: fit them\nWaiting on me: no' },
    s2: { when: '2026-11-08 11:30', text: 'Doing: painting\nNext: second coat' },
  })
  // Said again, a chat's state is replaced, not added to.
  const again = setChatState(two, 's1', '2026-11-08 12:00', 'Doing: hanging the gate\nNext: oil it')
  expect(again.split('as of').length).toBe(3)
  expect(again.includes('bought hinges')).toBe(false)
  expect(parseTrail(again).states.s1).toEqual({ when: '2026-11-08 12:00', text: 'Doing: hanging the gate\nNext: oil it' })
  expect(parseTrail(again).states.s2?.text).toBe('Doing: painting\nNext: second coat')
  // It stays with its chat when another is filed or the chat is named, and an empty answer takes it away.
  const later = retitleChat(addChat(again, { id: 's3', when: '2026-11-08 13:00', title: 'chat', done: false }), 's3', 'third')
  expect(parseTrail(later).chats.map(chat => chat.id)).toEqual(['s3', 's2', 's1'])
  expect(parseTrail(later).states.s1?.text).toBe('Doing: hanging the gate\nNext: oil it')
  expect(parseTrail(later).states.s3).toBe(undefined)
  expect(setChatState(again, 's2', '2026-11-08 14:00', ' \n')).toBe(again.replace('>   - as of 2026-11-08 11:30\n>   - Doing: painting\n>   - Next: second coat\n', ''))
  // A chat that is not in the list changes nothing.
  expect(setChatState(again, 'nope', '2026-11-08 14:00', 'Doing: x')).toBe(again)
})

test('a State block from the one day the task had a single state is dropped on the next write', async () => {
  const old = `${TASK_NOTE}\n# Agent Log\n\n> [!info] State · 2026-11-14 09:19\n> Goal: x\n> Next: y\n\n- [[A]]\n\n> [!abstract]- Chats\n> - 2026-11-08 10:00 · first · \`s1\`\n`
  expect(parseTrail(old).pins.map(pin => pin.target)).toEqual(['A'])
  expect(parseTrail(old).states).toEqual({})
  expect(addPin(old, { on: true, kind: 'note', target: 'B', label: 'B' })).toBe(`${TASK_NOTE}\n# Agent Log\n\n- [[A]]\n- [[B]]\n\n> [!abstract]- Chats\n> - 2026-11-08 10:00 · first · \`s1\`\n`)
})

test('a state answer is kept as plain lines, and the question asked is the lens\'s own words', async () => {
  expect(stateLines('> Doing: a\n\n- Next: b  \n\n  * Waiting on me: no\n')).toEqual(['Doing: a', 'Next: b', 'Waiting on me: no'])
  expect(stateLines(Array.from({ length: 20 }, (_, at) => `line ${at}`).join('\n')).length).toBe(8)
  // A lens file's frontmatter and title are not part of what is asked.
  expect(lensBody(stateLensFile('chat-state'))).toBe(STATE_PROMPT)
  expect(lensBody('---\nname: x\n---\n\n# X\n\nJust say hello.\n')).toBe('Just say hello.')
  expect(lensBody('No frontmatter here.')).toBe('No frontmatter here.')
  expect(stateAsk('Just say hello.')).toBe('Augi side question. Answer it from this conversation alone, in plain text, with no preamble.\n\nJust say hello.')
  // An empty or missing lens asks the built-in question.
  expect(stateAsk('').endsWith(STATE_PROMPT)).toBe(true)
  expect(STATE_PROMPT).toContain('Go by the last few turns, not the whole history.')
  expect(STATE_PROMPT).toContain('Waiting on me: yes and what for, or no')
})

test('pins go in the section once each; unpinning ticks the line off where it is and pinning it again brings it back', async () => {
  const note = { on: true, kind: 'note', target: 'A', label: 'A' } as const
  const one = addPin(TASK_NOTE, note)
  const two = addPin(one, SESSION_PIN)
  expect(addPin(two, note)).toBe(two)
  expect(two.endsWith('# Agent Log\n\n- [[A]]\n- session: `/p/a.jsonl` · garden · fix the sprinkler\n')).toBe(true)
  expect(parseTrail(two).pins.map(pin => pin.target)).toEqual(['A', '/p/a.jsonl'])

  const off = removePin(two, note)
  expect(off.endsWith('# Agent Log\n\n- [x] [[A]]\n- session: `/p/a.jsonl` · garden · fix the sprinkler\n')).toBe(true)
  expect(parseTrail(off).pins.map(pin => pin.on)).toEqual([false, true])
  expect(removePin(off, note)).toBe(off)
  expect(addPin(off, note)).toBe(two)
  expect(removePin(TASK_NOTE, note)).toBe(TASK_NOTE)
})

test('an old Agent Log file goes into its note once: pins, kept lines and questions as they were, saved notes as a record, chats folded, and the embed of the file gone', async () => {
  // A project note from the template of the time: its Agent Log heading holds an embed of the file.
  const project = '---\noutcome: It grows.\n---\n# Next\n- [ ] Plant the beds\n\n# Journal\n\n# Agent Log\n\n> [!abstract]- Agent Log\n> ![[Agent Log - PMOC - Garden]]\n'
  const file = oldLogFile('PMOC - Garden', '2026-11-08')
    .replace('## Context\n', '## Context\n\n- [[A]] · the plan\n- [x] file: `/r/docs/ARCH.md` · ARCH.md\n- the valve sticks below 4 degrees\n')
    .replace('## Questions\n', '## Questions\n\n- [ ] 2026-11-08 10:00 · does the cache survive restarts?\n')
    .replace('## Chats\n', '## Chats\n\n- 2026-11-08 11:00 · second · `s2`\n- 2026-11-08 10:00 · first · `s1` (done)\n')
    .replace('## Saved\n', '## Saved\n\n- [[Valve Notes]]\n')
  const moved = withOldLog(project, file, 'PMOC - Garden')
  expect(moved).toBe([
    '---', 'outcome: It grows.', '---', '# Next', '- [ ] Plant the beds', '', '# Journal', '',
    '# Agent Log', '',
    '- [[A]] · the plan', '- [x] file: `/r/docs/ARCH.md` · ARCH.md', '- the valve sticks below 4 degrees',
    '- [ ] 2026-11-08 10:00 · does the cache survive restarts?', '- saved: [[Valve Notes]]', '',
    '> [!abstract]- Chats', '> - 2026-11-08 11:00 · second · `s2`', '> - 2026-11-08 10:00 · first · `s1` (done)', '',
  ].join('\n'))
  // One pin, one unpinned record; the question and the saved note are lines, not pins.
  expect(parseTrail(moved).pins.map(pin => [pin.target, pin.on])).toEqual([['A', true], ['/r/docs/ARCH.md', false]])
  expect(parseTrail(moved).chats.map(chat => [chat.id, chat.done])).toEqual([['s2', false], ['s1', true]])
  // Moving the same file again changes nothing; a chat the note already lists stays first.
  expect(withOldLog(moved, file, 'PMOC - Garden')).toBe(moved)
  const filed = addChat(TASK_NOTE, { id: 's3', when: '2026-11-09 09:00', title: 'third', done: false })
  expect(parseTrail(withOldLog(filed, file, 'PMOC - Garden')).chats.map(chat => chat.id)).toEqual(['s3', 's2', 's1'])

  // A file from before pins were bullets ticked what it pinned: each such line comes over as a pin.
  const before = file
    .replace(/^Human note: .*$/m, 'Human note: [[PMOC - Garden]]. Context lines are `- [x] [[note]] · why it matters` (ticked goes into the next draft). Each chat is one line: when it started, its goal, its id; one that is finished stays listed, marked `(done)`.')
  expect(parseTrail(withOldLog(TASK_NOTE, before, 'PMOC - Garden')).pins.map(pin => [pin.target, pin.on])).toEqual([['A', true], ['/r/docs/ARCH.md', true]])

  // The file is marked, under its frontmatter, and a marked file is not moved again.
  expect(isMovedLog(file)).toBe(false)
  const marked = markMoved(file, 'PMOC - Garden', '2026-11-13')
  expect(isMovedLog(marked)).toBe(true)
  expect(marked.split('\n').slice(4, 7)).toEqual(['---', '', "Moved into [[PMOC - Garden]] on 2026-11-13: that note's # Agent Log section is the log now. Nothing reads or writes this file any more."])
  expect(markMoved('## Chats\n', 'X', '2026-11-13').startsWith('Moved into [[X]] on 2026-11-13')).toBe(true)
})

test('a log file from before bullets is tidied on the way: its ticked and open pin lines become bullets', async () => {
  const before = oldLogFile('X', '2026-11-08')
    .replace(/^Human note: .*$/m, 'Human note: [[X]]. Context lines are `- [x] [[note]] · why it matters` (ticked goes into the next draft). Each chat is one line: when it started, its goal, its id; one that is finished stays listed, marked `(done)`.')
    .replace('## Context\n', '## Context\n\n- [x] [[A]] · the plan\n- [ ] file: `/r/docs/ARCH.md` · ARCH.md\n- the valve sticks below 4 degrees\n')
  const tidy = tidyTrail(before, 'X')
  expect(tidy.includes('## Context\n\n- [[A]] · the plan\n- file: `/r/docs/ARCH.md` · ARCH.md\n- the valve sticks below 4 degrees\n')).toBe(true)
  expect(tidy.includes('ticked goes into the next draft')).toBe(false)
  expect(tidyTrail(tidy, 'X')).toBe(tidy)
})

test('what you paste becomes the right kind of pin', async () => {
  expect(pinFrom('https://claude.ai/artifact/abc-1', '/v', '/Users/t')?.kind).toBe('artifact')
  expect(pinFrom('https://claude.ai/code/artifact/abc-1', '/v', '/Users/t')?.kind).toBe('artifact')
  expect(pinFrom('https://example.com/a', '/v', '/Users/t')?.kind).toBe('url')
  expect(pinFrom('[[Sprinkler Benchmarks]]', '/v', '/Users/t')).toEqual({ on: true, kind: 'note', target: 'Sprinkler Benchmarks', label: 'Sprinkler Benchmarks' })
  expect(pinFrom('"/v/Notes/Sprinkler Benchmarks.md"', '/v', '/Users/t')).toEqual({ on: true, kind: 'note', target: 'Sprinkler Benchmarks', label: 'Sprinkler Benchmarks', path: '/v/Notes/Sprinkler Benchmarks.md' })
  expect(pinFrom('~/.claude/projects/-r/abc.jsonl', '/v', '/Users/t')?.kind).toBe('session')
  expect(pinFrom('~/repos/x/README.md', '/v', '/Users/t')).toEqual({ on: true, kind: 'file', target: '/Users/t/repos/x/README.md', label: 'README.md' })
  expect(pinFrom('   ', '/v', '/Users/t')).toBe(null)
})

test('a chat marked (done) in the log reads as done, and so does an older (hidden) mark', async () => {
  const text = addChat(addChat(oldLogFile('X', '2026-11-08'), { id: 's1', when: '2026-11-08 10:00', title: 'fix the gate', done: false }), { id: 's2', when: '2026-11-08 11:00', title: 'paint it', done: false })
  const done = text.replace('`s1`', '`s1` (done)')
  expect(parseTrail(done).chats.map(chat => [chat.id, chat.done])).toEqual([['s2', false], ['s1', true]])
  expect(parseTrail(text.replace('`s1`', '`s1` (hidden)')).chats.find(chat => chat.id === 's1')?.done).toBe(true)
})

test('a sent draft is titled by its lens and instruction; anything else is not ours', async () => {
  const lens = { name: 'echoes', path: '/v/lenses/echoes.md', description: '' }
  expect(chatTitleFrom(buildDraft([], ALL_ON, lens, 'why slow'))).toBe('echoes: why slow')
  expect(chatTitleFrom(buildDraft([], ALL_ON, null, 'just ask'))).toBe('just ask')
  expect(chatTitleFrom('hello')).toBe(null)
})

test('with an anchor, the draft carries its note and says where in it agents write', async () => {
  const anchor = { note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md' }
  const text = buildDraft([], ALL_ON, null, 'go', anchor)
  expect(text.includes('/v/PMOC - Garden.md')).toBe(true)
  expect(text.includes('the "# Agent Log" section at its end holds what is pinned to this work and, folded, the chats on it with where each last stood')).toBe(true)
  // Nothing is logged unasked: a chat writes in the note only on request, and then under the heading.
  expect(text.includes('Write in my note only when I ask. What I ask you to save goes under its "# Agent Log" heading, never above it')).toBe(true)
  // No second file: the log is in the note.
  expect(text.includes('Agent Log - PMOC - Garden')).toBe(false)
  expect(buildDraft([], { ...ALL_ON, project: false }, null, 'go', anchor).includes('/v/PMOC - Garden.md')).toBe(false)
})

test('pick an anchor, start a chat, send it: the chat lands on the trail', { options: OPTIONS }, async ($, on) => {
  on('env.get', () => ({ value: '/Users/t' }))
  const filled: string[] = []
  const written = new Map<string, string>()
  const store = new Map<string, unknown>()
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.id', () => ({ value: 'own-session' }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/Users/t/.claude/projects'].includes(e.path) || e.path.endsWith('/echoes.md') || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({
    value: written.get(e.path)
      ?? (e.path.endsWith('/echoes.md') ? 'description: Have I\n'
        : e.path.endsWith('Kanban.md') ? BOARD_TEXT
        : e.path.endsWith('Garden.md') ? '---\noutcome: It loads fast.\n---\n# Next\n- [ ] Plant the beds\n# Journal\n'
        : '## 3. This season\n**Primary: the garden.**\n'),
  }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', (_, e) => {
    const file = (name: string, mtimeMs: number) => ({ name, kind: 'file', size: 1, mtimeMs, isLink: false })
    const dir = (name: string) => ({ name, kind: 'dir', size: 0, mtimeMs: 0, isLink: false })
    if (e.path.endsWith('/lenses')) {
      return { value: [file('echoes.md', 0)] }
    }
    if (e.path.endsWith('/.claude/projects')) {
      return { value: [dir('-r-garden')] }
    }
    if (e.path.endsWith('-r-garden')) {
      return { value: [file('new.jsonl', 2000)] }
    }

    return { value: [] }
  })
  on('process.run', (_, e) => ({
    value: {
      exitCode: 0,
      stdout: e.argv[0] === 'find' ? '/v/PMOC - Garden.md\n' : human,
      stderr: '', isStdoutTruncated: false, isStderrTruncated: false,
    },
  }))
  on('prompt.read', () => ({ value: { text: 'is the sprinkler the bottleneck', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'card-PMOC - Garden' })).toBeFalsy()
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-PMOC - Garden' })
  expect(store.get('anchors')).toEqual([['own-session', 'PMOC - Garden']])
  expect(await ui.find({ key: 'copy-embed' })).toBeFalsy()
  // Outcome and tasks sit behind one toggle, closed by default.
  expect(await ui.find({ text: 'Plant the beds' })).toBeFalsy()
  expect(await ui.find({ text: '🎯 It loads fast.' })).toBeFalsy()
  expect((await ui.find({ key: 'anchor-details' }))?.text).toMatch(/▸ outcome and next \(2\)/)
  await ui.press({ key: 'anchor-details' })
  expect(await ui.find({ text: 'Plant the beds' })).toBeTruthy()
  expect(await ui.find({ text: '🎯 It loads fast.' })).toBeTruthy()
  await ui.press({ key: 'anchor-details' })
  expect(await ui.find({ text: 'Plant the beds' })).toBeFalsy()
  await ui.press({ key: 'anchor-details' })
  expect([...written.keys()].some(path => path.includes('Trail - '))).toBe(false)
  // The anchor card opens the note; its Agent Log is a section of it, so there is no second
  // button. Who you are sits in the You card under Always, with the season line, never behind Setup.
  expect(await ui.find({ key: 'open-note' })).toBeTruthy()
  expect(await ui.find({ key: 'open-trail' })).toBeFalsy()
  expect(await ui.find({ key: 'edit-context' })).toBeTruthy()
  expect(await ui.find({ key: 'open-board' })).toBeTruthy()
  expect(await ui.find({ text: 'Season: Primary: the garden.' })).toBeTruthy()
  await ui.press({ key: 'head-youOpen' })
  expect(await ui.find({ key: 'edit-context' })).toBeFalsy()
  await ui.press({ key: 'head-youOpen' })
  expect(await ui.find({ key: 'about-me' })).toBeFalsy()
  // What every draft carries is switched under Setup, and listed nowhere else.
  expect(await ui.find({ key: 'pick-project' })).toBeFalsy()
  expect(await ui.find({ key: 'openaugi' })).toBeFalsy()
  expect(await ui.find({ key: 'pick-today' })).toBeFalsy()
  expect(await ui.find({ text: /Context · \d+ going in/ })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })
  // They are real checkboxes, on by default: pressing one flips it in place.
  expect((await ui.find({ key: 'openaugi' }))?.text).toMatch(/^\s*☑/)
  expect((await ui.find({ key: 'pick-project' }))?.text).toMatch(/^\s*☑/)
  await ui.press({ key: 'pick-project' })
  expect((await ui.find({ key: 'pick-project' }))?.text).toMatch(/^\s*☐/)
  await ui.press({ key: 'pick-project' })
  await ui.press({ key: 'setup-menu' })
  await ui.press({ key: 'lens-menu' })
  await ui.press({ key: 'echoes' })

  expect(filled.length).toBe(1)
  expect(filled[0]?.includes('/v/PMOC - Garden.md')).toBe(true)

  await $.prompt.submit({ text: filled[0] ?? '' })
  const trail = written.get('/v/PMOC - Garden.md') ?? ''
  expect(parseTrail(trail).chats).toEqual([
    { id: 'own-session', when: '2026-11-08 09:30', title: 'echoes: is the sprinkler the bottleneck', done: false },
  ])
  // The chat is filed in the log; the pane keeps no list of chats.
  expect(await ui.find({ key: 'chat-own-session' })).toBeFalsy()
  expect(await ui.find({ text: /Chats on PMOC - Garden/ })).toBeFalsy()

  // The pane has one place to type, the scratchpad: no question field in the Anchor card.
  expect((await ui.findAll({ type: 'Input' })).some(one => /^capture/.test(String(one.key ?? '')))).toBe(false)
  // A question an agent left in the log stays in the log: the pane lists none.
  written.set('/v/PMOC - Garden.md', `${trail}- [ ] 2026-11-08 09:30 · does the cache survive restarts?\n`)
  await ui.press({ key: 'refresh' })
  expect(await ui.find({ text: '? does the cache survive restarts?' })).toBeFalsy()
  await ui.unmount()
})

test('settings resolve inside the notes folder, with ~ and absolute paths', async () => {
  const paths = resolvePaths({ notesFolder: '~/Notes/', boardNote: '/elsewhere/Board.md', lensPicks: ' a, ,b ' }, '/Users/t')
  expect(paths.notes).toBe('/Users/t/Notes')
  expect(paths.context).toBe('/Users/t/Notes/OpenAugi/AGENT/Slowly Changing Context.md')
  expect(paths.board).toBe('/elsewhere/Board.md')
  expect(paths.picks).toEqual(['a', 'b'])
  expect(paths.editor).toBe('obsidian')
})

test('each editor opens a note its own way', async () => {
  expect(editorArgv('obsidian', '', '/v/a b.md')).toEqual(['open', 'obsidian://open?path=%2Fv%2Fa%20b.md'])
  expect(editorArgv('vscode', '', '/v/a.md')).toEqual(['code', '/v/a.md'])
  expect(editorArgv('custom', 'zed --wait {file}', '/v/a b.md')).toEqual(['zed', '--wait', '/v/a b.md'])
  expect(editorArgv('custom', 'subl', '/v/a.md')).toEqual(['subl', '/v/a.md'])
  expect(editorArgv('custom', '', '/v/a.md')).toEqual(['open', '/v/a.md'])
})

test('setup lists what is missing, the notes folder first', async () => {
  const paths = resolvePaths({ notesFolder: '/n' }, '/h')
  const none = setupItems(paths, new Set())
  expect(none.map(item => item.ok)).toEqual([false, false, false, false])
  const some = setupItems(paths, new Set(['/n', '/n/OpenAugi/AGENT/lenses']))
  expect(some.filter(item => !item.ok).map(item => item.field)).toEqual(['contextNote', 'boardNote'])
  expect(setupItems(resolvePaths({}, '/h'), new Set(['']))[0]?.ok).toBe(false)
})

test('with no notes folder the pane is the setup screen, and entering one writes the setting', async ($, on) => {
  const set: { key: string; value: unknown }[] = []
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.id', () => ({ value: 'own' }))
  on('clock.now', () => ({ value: 0 }))
  on('store.get', () => ({ value: undefined }))
  on('store.set', () => ({ value: undefined }))
  on('fs.exists', () => ({ value: false }))
  on('config.list', () => ({ value: [{ key: 'augi-ui.notesFolder', label: 'Notes folder', kind: 'text', value: '', provider: { plugin: 'augi-ui', tier: 'user' }, isLocked: false }] }))
  on('config.set', (_, e) => {
    set.push({ key: e.key, value: e.value })
    return { value: e.value }
  })

  await $.command.run({ command: 'augi', args: '' })
  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface })
    expect(await ui.find({ key: 'set-notesFolder' })).toBeTruthy()
    expect(await ui.find({ key: 'anchor-menu' })).toBeFalsy()
    await ui.unmount()
  }
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.input({ key: 'set-notesFolder', text: '~/Notes' })
  expect(set).toEqual([{ key: 'augi-ui.notesFolder', value: '~/Notes' }])
  await ui.unmount()
})

const CONFIG = '/Users/t/Library/Application Support/obsidian/obsidian.json'

test('obsidian vaults come open first, then most recent, and bad files give none', async () => {
  const text = JSON.stringify({ vaults: {
    a: { path: '/old', ts: 1 },
    b: { path: '/open', ts: 0, open: true },
    c: { path: '/newer', ts: 5 },
    d: { ts: 9 },
  } })
  expect(vaultsFrom(text)).toEqual(['/open', '/newer', '/old'])
  expect(vaultsFrom('not json')).toEqual([])
  expect(vaultsFrom('{}')).toEqual([])
  expect(obsidianConfigs('/h', '')[0]).toBe('/h/Library/Application Support/obsidian/obsidian.json')
  expect(obsidianConfigs('/h', 'C:\\Users\\t\\AppData\\Roaming').pop()).toBe('C:/Users/t/AppData/Roaming/obsidian/obsidian.json')
})

test('the setup screen offers the Obsidian vaults that still exist, and picking one sets the folder', async ($, on) => {
  const set: { key: string; value: unknown }[] = []
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/Users/t' : undefined }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('clock.now', () => ({ value: 0 }))
  on('store.get', () => ({ value: undefined }))
  on('fs.exists', (_, e) => ({ value: [CONFIG, '/Users/t/Notes'].includes(e.path) }))
  on('fs.read', () => ({ value: JSON.stringify({ vaults: { a: { path: '/Users/t/Gone', ts: 2 }, b: { path: '/Users/t/Notes', ts: 1, open: true } } }) }))
  on('config.list', () => ({ value: [{ key: 'augi-ui.notesFolder', label: 'Notes folder', kind: 'text', value: '', provider: { plugin: 'augi-ui', tier: 'user' }, isLocked: false }] }))
  on('config.set', (_, e) => {
    set.push({ key: e.key, value: e.value })
    return { value: e.value }
  })

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'vault-0' })).toBeTruthy()
  expect(await ui.find({ key: 'vault-1' })).toBeFalsy()
  await ui.press({ key: 'vault-0' })
  expect(set).toEqual([{ key: 'augi-ui.notesFolder', value: '/Users/t/Notes' }])
  await ui.unmount()
})

test('after the folder, setup walks the steps: found notes, a starter, or skip', { options: { ...OPTIONS, boardNote: '', contextNote: '', lensFolder: '', logFolder: '' } }, async ($, on) => {
  const written = new Map<string, string>()
  const store = new Map<string, unknown>()
  const set: { key: string; value: unknown }[] = []
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/Users/t' : undefined }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.id', () => ({ value: 'own' }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('store.delete', (_, e) => {
    store.delete(e.key)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: e.path === '/v' || written.has(e.path) || (e.path === '/v/OpenAugi/AGENT/lenses' && written.has('/v/OpenAugi/AGENT/lenses/focus-check.md')) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? '' }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('config.list', () => ({ value: ['boardNote', 'contextNote', 'lensFolder', 'notesFolder'].map(field => ({ key: `augi-ui.${field}`, label: field, kind: 'text', value: '', provider: { plugin: 'augi-ui', tier: 'user' }, isLocked: false })) }))
  on('config.set', (_, e) => {
    set.push({ key: e.key, value: e.value })
    return { value: e.value }
  })
  const run = (stdout: string) => ({ value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } })
  const made: string[] = []
  on('process.run', (_, e) => {
    if (e.argv[0] === 'mkdir') {
      made.push(e.argv[2] ?? '')
    }
    if (e.argv[0] === 'grep') {
      return run('/v/old/Board.md\n/v/Plans/Kanban.md\n/v/.trash/Kanban.md\n')
    }
    if (e.argv[0] === 'ls') {
      return run(e.argv.slice(3).reverse().join('\n'))
    }

    return run('')
  })

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'anchor-menu' })).toBeFalsy()
  expect(await ui.find({ key: 'found-1' })).toBeTruthy()
  expect(await ui.find({ key: 'found-2' })).toBeFalsy()
  await ui.press({ key: 'found-0' })
  expect(set).toEqual([{ key: 'augi-ui.boardNote', value: 'Plans/Kanban.md' }])

  await ui.press({ key: 'skip-step' })
  expect(store.get('skipped')).toEqual(['boardNote'])
  expect(await ui.find({ key: 'set-contextNote' })).toBeTruthy()
  expect(await ui.find({ key: 'found-0' })).toBeFalsy()
  await ui.press({ key: 'use-starter' })
  expect([...written.keys()]).toEqual(['/v/OpenAugi/AGENT/Slowly Changing Context.md'])
  // No folder is made for Agent Logs: each is a section of its note.
  expect(made.includes('/v/OpenAugi/Agent-Logs')).toBe(false)
  expect(await ui.find({ key: 'set-lensFolder' })).toBeTruthy()
  await ui.press({ key: 'use-starter' })
  expect(await ui.find({ key: 'anchor-menu' })).toBeTruthy()

  await $.command.run({ command: 'augi', args: 'setup' })
  expect(store.has('skipped')).toBe(false)
  expect(await ui.find({ key: 'set-boardNote' })).toBeTruthy()
  await ui.unmount()
})

test('pins on an anchor: paste to pin, press the name to open, unpin ticks the line off; a note the chat writes is pinned, an artifact it publishes is not', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  const store = new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]], ['pins', [{ on: true, kind: 'file', target: '/r/old/EVERYWHERE.md', label: 'EVERYWHERE.md' }]]])
  const opened: string[][] = []
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', () => ({ value: undefined }))
  on('session.id', () => ({ value: 'own-session' }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/r/docs/ARCH.md', '/v/Notes/Sprinkler Benchmarks.md'].includes(e.path) || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => {
    if (e.argv[0] !== 'find' && e.argv[0] !== 'mkdir') {
      opened.push(e.argv)
    }
    const found = e.argv[0] !== 'find' ? '' : e.argv.includes('PMOC - Garden.md') ? '/v/PMOC - Garden.md\n' : e.argv.includes('Sprinkler Benchmarks.md') ? '/v/Notes/Sprinkler Benchmarks.md\n' : ''

    return { value: { exitCode: 0, stdout: found, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('tool.call', () => ({ result: { url: 'https://claude.ai/artifact/abc-1' } as never, text: 'published https://claude.ai/artifact/abc-1' }))
  const filled: string[] = []
  on('prompt.read', () => ({ value: { text: 'explain the cache', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))
  const logNow = () => written.get('/v/PMOC - Garden.md') ?? ''
  const pinned = () => parseTrail(logNow()).pins.filter(pin => pin.on)

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'head-pinsOpen' }))?.text).toMatch(/📌 pinned \(0\)/)

  await ui.input({ key: 'pin-paste-0', text: '/r/docs/ARCH.md' })
  expect(pinned()).toEqual([{ on: true, kind: 'file', target: '/r/docs/ARCH.md', label: 'ARCH.md' }])
  await ui.input({ key: 'pin-paste-1', text: '[[Sprinkler Benchmarks]]' })
  expect(pinned().map(pin => pin.target)).toEqual(['/r/docs/ARCH.md', 'Sprinkler Benchmarks'])
  // In the note's Agent Log section a pin is a plain bullet, not a ticked box.
  expect(logNow().includes('# Agent Log\n\n- file: `/r/docs/ARCH.md` · ARCH.md\n- [[Sprinkler Benchmarks]]\n')).toBe(true)
  expect((await ui.find({ key: 'head-pinsOpen' }))?.text).toMatch(/📌 pinned \(2\)/)

  // A pin row is its name, which opens it, and unpin: no tick box, and no line explaining one.
  expect((await ui.find({ key: 'pin-open-note-Sprinkler Benchmarks' }))?.text).toMatch(/Sprinkler Benchmarks/)
  expect((await ui.find({ key: 'pin-drop-note-Sprinkler Benchmarks' }))?.text).toMatch(/unpin/)
  expect(await ui.find({ key: 'pin-tick-note-Sprinkler Benchmarks' })).toBeFalsy()
  expect(await ui.find({ text: /goes into this anchor's drafts/ })).toBeFalsy()
  await ui.press({ key: 'pin-open-note-Sprinkler Benchmarks' })
  expect(opened.pop()).toEqual(['open', 'obsidian://open?path=%2Fv%2FNotes%2FSprinkler%20Benchmarks.md'])
  expect(pinned().length).toBe(2)

  // Unpinned, it leaves the card and the draft; its line stays on the log, ticked off.
  await ui.press({ key: 'pin-drop-file-/r/docs/ARCH.md' })
  expect(pinned().map(pin => pin.target)).toEqual(['Sprinkler Benchmarks'])
  expect(logNow().includes('- [x] file: `/r/docs/ARCH.md` · ARCH.md\n- [[Sprinkler Benchmarks]]\n')).toBe(true)
  expect(await ui.find({ key: 'pin-open-file-/r/docs/ARCH.md' })).toBeFalsy()
  expect((await ui.find({ key: 'head-pinsOpen' }))?.text).toMatch(/📌 pinned \(1\)/)
  await pressDraft(ui)
  expect(filled[0]?.includes('- Note: "/v/Notes/Sprinkler Benchmarks.md"')).toBe(true)
  expect(filled[0]?.includes('ARCH.md')).toBe(false)
  // A pin an older pane kept for every chat is in no draft: there is one place to pin.
  expect(filled[0]?.includes('EVERYWHERE.md')).toBe(false)
  // Pinned again, the same line is a bullet again, where it was.
  await ui.input({ key: 'pin-paste-1', text: '/r/docs/ARCH.md' })
  expect(logNow().includes('# Agent Log\n\n- file: `/r/docs/ARCH.md` · ARCH.md\n- [[Sprinkler Benchmarks]]\n')).toBe(true)
  await ui.press({ key: 'pin-drop-file-/r/docs/ARCH.md' })

  // An artifact the chat publishes is not pinned by itself; its link pasted in the card is,
  // and is listed first.
  await $.prompt.submit({ text: filled[0] ?? '' })
  await $.tool.call({ tool: 'Artifact', file_path: '/s/page.html', title: 'Cache explainer' })
  expect(pinned().map(pin => pin.target)).toEqual(['Sprinkler Benchmarks'])
  await ui.input({ key: 'pin-paste-1', text: 'https://claude.ai/artifact/abc-1' })
  expect(pinned().slice(1)).toEqual([{ on: true, kind: 'artifact', target: 'https://claude.ai/artifact/abc-1', label: 'artifact' }])
  expect(await ui.find({ key: 'pin-open-artifact-https://claude.ai/artifact/abc-1' })).toBeTruthy()
  // A new note it writes in the vault is pinned; a file outside the vault or an edit of one there is not.
  await $.tool.call({ tool: 'Write', file_path: '/v/Ideas/New Idea.md', content: 'x' })
  await $.tool.call({ tool: 'Write', file_path: '/tmp/scratch.md', content: 'x' })
  await $.tool.call({ tool: 'Write', file_path: '/v/AGENT/Slowly Changing Context.md', content: 'x' })
  await $.tool.call({ tool: 'Edit', file_path: '/r/src/a.ts', old_string: 'a', new_string: 'b' })
  expect(pinned().slice(2)).toEqual([{ on: true, kind: 'note', target: 'New Idea', label: 'New Idea' }])
  // The pane keeps no list of the files a chat wrote or edited.
  expect(await ui.find({ text: /Files this chat touched/ })).toBeFalsy()

  // The pane has no second place to pin, no note search and no session list.
  for (const key of ['pin-search', 'all-paste-0', 'all-paste-1', 'sessions-menu', 'head-contextOpen']) {
    expect(await ui.find({ key })).toBeFalsy()
  }

  // Every card and both sections fold from their title, and open again. The pins are in
  // the Anchor card, under its note button, and fold on their own and with it.
  for (const [head, inside] of [['head-pinsOpen', 'pin-paste-3'], ['head-anchorOpen', 'open-note'], ['head-anchorOpen', 'pin-paste-3'], ['head-padOpen', 'pad-input-0']] as const) {
    expect(await ui.find({ key: inside })).toBeTruthy()
    await ui.press({ key: head })
    expect(await ui.find({ key: inside })).toBeFalsy()
    await ui.press({ key: head })
    expect(await ui.find({ key: inside })).toBeTruthy()
  }
  await ui.press({ key: 'head-chatOpen' })
  expect(await ui.find({ key: 'head-anchorOpen' })).toBeFalsy()
  expect(await ui.find({ key: 'open-note' })).toBeFalsy()
  expect(await ui.find({ key: 'edit-context' })).toBeTruthy()
  await ui.press({ key: 'head-anywhereOpen' })
  expect(await ui.find({ key: 'edit-context' })).toBeFalsy()
  expect(await ui.find({ key: 'refresh' })).toBeTruthy()
  await ui.unmount()
})

test('Next lists the open checkboxes under the Next heading only', async () => {
  const note = [
    '---', 'outcome: x', '---', '# Next', '', '- [ ] Plant the beds', '- [x] done already', '\t- [ ] [[PMOC - Birdhouse]]', '- [ ] ',
    '## Next detail', '- [ ] still under Next', '# Journal', '### 2026-11-08', '- [ ] a journal checkbox',
  ].join('\n')
  expect(nextFrom(note)).toEqual([
    { text: 'Plant the beds', note: null },
    { text: 'PMOC - Birdhouse', note: 'PMOC - Birdhouse' },
    { text: 'still under Next', note: null },
  ])
  expect(nextFrom('## Next\n- [ ] a\n## Later\n- [ ] b')).toEqual([{ text: 'a', note: null }])
  expect(nextFrom('# Next\n- [ ] wire [[Child|the child]] up')).toEqual([{ text: 'wire the child up', note: 'Child' }])
  expect(nextFrom('# Journal\n- [ ] a')).toEqual([])
})

test('the old Agent Log file was named for its note, in the log folder', async () => {
  expect(logName('PMOC - A/B')).toBe('Agent Log - PMOC - A-B')
  expect(trailPath('PMOC - A/B')).toBe('/v/Notes/Agent Log - PMOC - A-B.md')
})

const OLD_INTRO = 'Project: [[PMOC - Garden]], embedded there under # Agent Log. Context lines are `- [x] [[note]] · why it matters` (ticked goes into the next draft). Hidden chats stay listed, marked `(hidden)`.'
// The intro the log had while a finished chat was called hidden.
const HIDDEN_INTRO = 'Human note: [[PMOC - Garden]]. Context lines are `- [x] [[note]] · why it matters` (ticked goes into the next draft). Hidden chats stay listed, marked `(hidden)`.'

test('an older Agent Log loses its title and seen box, gains the note link, turns its pins into bullets, and keeps its sections', async () => {
  const sections = ['## Context', '', '- [x] [[A note]] · why', '', '## Questions', '', '- [x] 2022-01-04 10:00 · q', '', '## Chats', '', '## Saved', '']
  const old = ['---', 'type: document', 'description: d', 'created: 2022-01-04', '---', '', '# Agent Log - PMOC - Garden', '', '- [x] seen', '', OLD_INTRO, '', ...sections].join('\n')
  const tidy = tidyTrail(old, 'PMOC - Garden')
  // Its ticked pin is a bullet now; the ticked question under ## Questions is as it was.
  const tidied = sections.map(line => (line === '- [x] [[A note]] · why' ? '- [[A note]] · why' : line))
  expect(tidy).toBe(oldLogFile('PMOC - Garden', '2022-01-04').replace(/description: .*/, 'description: d').split('## Context')[0] + tidied.join('\n'))
  expect(tidyTrail(tidy, 'PMOC - Garden')).toBe(tidy)
  expect(parseTrail(tidy)).toEqual(parseTrail(old))
  expect(tidyTrail(oldLogFile('PMOC - Garden', '2022-01-04'), 'PMOC - Garden')).toBe(oldLogFile('PMOC - Garden', '2022-01-04'))
  // A log from the version before this one takes the current intro, which says done, and nothing else changes.
  const current = oldLogFile('PMOC - Garden', '2022-01-04')
  const previous = current.replace(/^Human note: .*$/m, HIDDEN_INTRO)
  expect(previous).not.toBe(current)
  expect(tidyTrail(previous, 'PMOC - Garden')).toBe(current)
  expect(current.includes('marked `(done)`')).toBe(true)

  // A moved trail and a log with no link line get the link line; a seen line inside a section stays.
  const trail = tidyTrail('# Trail - PMOC - Garden\n\n## Chats\n- [ ] seen it\n', 'PMOC - Garden')
  expect(trail.startsWith('Human note: [[PMOC - Garden]]. ')).toBe(true)
  expect(trail.endsWith('\n\n## Chats\n- [ ] seen it\n')).toBe(true)
  const bare = tidyTrail('---\ntype: document\n---\n## Chats\n', 'PMOC - Garden')
  expect(bare.startsWith('---\ntype: document\n---\n\nHuman note: [[PMOC - Garden]]. ')).toBe(true)
  expect(bare.endsWith('\n\n## Chats\n')).toBe(true)
})

test('a new lens is written once from its name, opened in the editor, and listed', { options: { ...OPTIONS, lensPicks: '' } }, async ($, on) => {
  const written = new Map<string, string>()
  const ran: string[][] = []
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', () => ({ value: undefined }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', () => ({ value: undefined }))
  on('session.id', () => ({ value: 'own-session' }))
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/lenses/echoes.md', '/v/AGENT/Slowly Changing Context.md'].includes(e.path) || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : 'description: Have I\n') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', (_, e) => {
    const file = (name: string) => ({ name, kind: 'file', size: 1, mtimeMs: 0, isLink: false })
    const names = ['echoes.md', ...[...written.keys()].filter(path => path.startsWith('/v/lenses/')).map(path => path.slice('/v/lenses/'.length))]

    return { value: e.path.endsWith('/lenses') ? names.map(file) : [] }
  })
  on('process.run', (_, e) => {
    ran.push(e.argv)
    return { value: { exitCode: 0, stdout: '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('prompt.fill', () => ({ isFilled: true }))

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ text: /Saved prompts · 1/ })).toBeTruthy()
  await ui.press({ key: 'lens-menu' })
  await ui.input({ key: 'new-lens-1', text: 'check my tests' })
  expect(written.get('/v/lenses/check-my-tests.md')).toBe(newLensSkeleton('check-my-tests'))
  expect(ran.some(argv => argv.join(' ') === `open obsidian://open?path=${encodeURIComponent('/v/lenses/check-my-tests.md')}`)).toBe(true)
  expect(await ui.find({ key: 'check-my-tests' })).toBeTruthy()

  // A name that's there already just opens that lens; nothing is written over it.
  const writes = written.size
  await ui.input({ key: 'new-lens-2', text: 'echoes' })
  expect(written.size).toBe(writes)
  expect(written.has('/v/lenses/echoes.md')).toBe(false)
  expect(ran.some(argv => argv.join(' ') === `open obsidian://open?path=${encodeURIComponent('/v/lenses/echoes.md')}`)).toBe(true)
  await ui.unmount()
})

test('a placeholder chat title takes the draft title once; a real title stays', async () => {
  const text = addChat(oldLogFile('X', '2026-11-08'), { id: 's1', when: '2026-11-08 10:00', title: 'chat', done: false })
  const named = retitleChat(text, 's1', 'echoes: why slow')
  expect(parseTrail(named).chats[0]?.title).toBe('echoes: why slow')
  expect(retitleChat(named, 's1', 'later')).toBe(named)
  expect(retitleChat(text, 's2', 'other')).toBe(text)
  const hidden = retitleChat(text.replace('`s1`', '`s1` (done)'), 's1', 'named')
  expect(parseTrail(hidden).chats[0]).toEqual({ id: 's1', when: '2026-11-08 10:00', title: 'named', done: true })
})

test('session anchors keep the latest per session, the most recent last, capped', async () => {
  const one = withAnchor([], 'a', 'PMOC - Garden')
  const two = withAnchor(one, 'b', 'PMOC - Birdhouse')
  expect(withAnchor(two, 'a', 'TASK - Plant')).toEqual([['b', 'PMOC - Birdhouse'], ['a', 'TASK - Plant']])
  expect(withAnchor(two, 'a', null)).toEqual([['b', 'PMOC - Birdhouse']])
  expect(withAnchor(two, 'c', 'X', 2)).toEqual([['b', 'PMOC - Birdhouse'], ['c', 'X']])
  expect(anchorPairs('PMOC - Garden')).toEqual([])
  expect(anchorPairs([['a', 'N'], ['b'], 3])).toEqual([['a', 'N']])
})

test('the pin list puts artifacts first', async () => {
  const art = (n: number) => ({ on: true, kind: 'artifact', target: `https://claude.ai/artifact/${n}`, label: String(n) }) as const
  const note = { on: true, kind: 'note', target: 'N', label: 'N' } as const
  expect(artifactsFirst([note, art(1), SESSION_PIN, art(2)]).map(pin => pin.label)).toEqual(['1', '2', 'N', SESSION_PIN.label])
})

test('a new session opens with no anchor; anchoring logs the chat and pins nothing it published, and a sent draft names the chat', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  // An old global anchor and another session's anchor are not this session's.
  const store = new Map<string, unknown>([['anchor', 'PMOC - Garden'], ['anchors', [['other-session', 'PMOC - Garden']]]])
  let session = 'new-session'
  let url = 'https://claude.ai/artifact/early-1'
  const opened: string[][] = []
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', () => ({ value: undefined }))
  on('session.id', () => ({ value: session }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md'].includes(e.path) || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => {
    if (e.argv[0] === 'open') {
      opened.push(e.argv)
    }
    return { value: { exitCode: 0, stdout: e.argv[0] === 'find' && e.argv.includes('PMOC - Garden.md') ? '/v/PMOC - Garden.md\n' : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('tool.call', () => ({ result: { url } as never, text: `published ${url}` }))
  const filled: string[] = []
  on('prompt.read', () => ({ value: { text: 'explain the cache', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))
  const logNow = () => parseTrail(written.get('/v/PMOC - Garden.md') ?? '')

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(await ui.find({ key: 'open-trail' })).toBeFalsy()
  expect(await ui.find({ key: 'add-this-chat' })).toBeFalsy()

  // With no anchor there is nothing to pin to, so no Pins card; the Anchor card still drafts a prompt.
  expect(await ui.find({ key: 'head-pinsOpen' })).toBeFalsy()
  expect((await ui.findAll({ type: 'Input' })).some(one => String(one.key ?? '').startsWith('pin-paste'))).toBe(false)
  await ui.press({ key: 'lens-menu' })
  expect((await ui.find({ key: 'draft' }))?.text).toMatch(/just my context/)

  // An artifact published before anchoring is written nowhere, then or on anchoring.
  await $.tool.call({ tool: 'Artifact', file_path: '/s/early.html', title: 'Early plan' })
  expect(written.size).toBe(0)

  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-PMOC - Garden' })
  expect(store.get('anchors')).toEqual([['other-session', 'PMOC - Garden'], ['new-session', 'PMOC - Garden']])
  expect(logNow().chats).toEqual([{ id: 'new-session', when: '2026-11-08 09:30', title: 'chat', done: false }])
  expect(logNow().pins).toEqual([])
  expect((await ui.find({ key: 'draft' }))?.text).toMatch(/just this context/)

  // Published after anchoring: not pinned either. Its link pasted in the Pins card pins it,
  // and pressing its name opens it.
  url = 'https://claude.ai/artifact/late-2'
  await $.tool.call({ tool: 'Artifact', file_path: '/s/late.html', title: 'Late plan' })
  expect(logNow().pins).toEqual([])
  await ui.input({ key: 'pin-paste-0', text: 'https://claude.ai/artifact/late-2' })
  expect(logNow().pins.map(pin => [pin.target, pin.on])).toEqual([['https://claude.ai/artifact/late-2', true]])
  await ui.press({ key: 'pin-open-artifact-https://claude.ai/artifact/late-2' })
  expect(opened).toEqual([['open', 'https://claude.ai/artifact/late-2']])

  // A draft sent later names the chat; the pinned artifact goes in it.
  await pressDraft(ui)
  expect(filled[0]?.includes('https://claude.ai/artifact/late-2')).toBe(true)
  await $.prompt.submit({ text: filled[0] ?? '' })
  expect(logNow().chats.map(chat => chat.title)).toEqual(['explain the cache'])

  // Another session opens with no anchor; this one's comes back when the session does.
  session = 'third-session'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  session = 'new-session'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  await ui.unmount()
})

test('a scratchpad entry lands under its day in the Journal, every other line as it was', async () => {
  const at = (when: string, text: string) => ({ when, text })
  // Oldest first: a new day goes last; the same day takes the entry as its last line.
  const oldestFirst = ['---', 'outcome: x', '# Journal', '---', '# Next', '- [ ] a', '', '# Journal', '', '### 2022-01-03', '- one', '', '### 2022-01-04', '- two', '', '# Agent Log', '']
  const once = addJournalEntry(oldestFirst.join('\n'), at('2022-01-05 09:30', 'water the beds'))
  expect(once).toBe([...oldestFirst.slice(0, 14), '', '### 2022-01-05', '- 09:30 water the beds', '', '# Agent Log', ''].join('\n'))
  const twice = addJournalEntry(once, at('2022-01-05 10:15', 'then the pots'))
  expect(twice.split('\n').filter(line => line === '### 2022-01-05').length).toBe(1)
  expect(twice.includes('- 09:30 water the beds\n- 10:15 then the pots\n\n# Agent Log')).toBe(true)
  expect(addJournalEntry(oldestFirst.join('\n'), at('2022-01-03 08:00', 'early'))).toBe(
    [...oldestFirst.slice(0, 11), '- 08:00 early', ...oldestFirst.slice(11)].join('\n'),
  )
  // Newest first: a new day goes on top, at the level the note uses.
  const newestFirst = ['## journal', '#### 2022-01-04 Tue', '- two', '#### 2022-01-03', '- one']
  expect(addJournalEntry(newestFirst.join('\n'), at('2022-01-05 09:30', 'x'))).toBe(
    ['## journal', '#### 2022-01-05', '- 09:30 x', '', ...newestFirst.slice(1)].join('\n'),
  )
  // A dated heading after the Journal section is not the Journal's.
  const after = ['# Journal', '', '# Later', '### 2022-01-05', '- not this']
  expect(addJournalEntry(after.join('\n'), at('2022-01-05 09:30', 'x'))).toBe(
    ['# Journal', '', '### 2022-01-05', '- 09:30 x', '', '# Later', '### 2022-01-05', '- not this'].join('\n'),
  )
  // An empty day heading (the template's) takes the line right under it.
  expect(addJournalEntry('# Journal\n\n### 2022-01-05\n\n\n# Agent Log', at('2022-01-05 09:30', 'x'))).toBe('# Journal\n\n### 2022-01-05\n- 09:30 x\n\n\n# Agent Log')
  // No Journal: one goes above the Agent Log heading, else at the end.
  expect(addJournalEntry('---\na: 1\n---\n# Next\n- [ ] a\n# Agent Log\n> embed', at('2022-01-05 09:30', 'x'))).toBe(
    '---\na: 1\n---\n# Next\n- [ ] a\n\n# Journal\n\n### 2022-01-05\n- 09:30 x\n\n# Agent Log\n> embed',
  )
  expect(addJournalEntry('# Next\n- [ ] a\n', at('2022-01-05 09:30', 'x'))).toBe('# Next\n- [ ] a\n\n# Journal\n\n### 2022-01-05\n- 09:30 x\n')
  expect(addJournalEntry('', at('2022-01-05 09:30', 'x'))).toBe('# Journal\n\n### 2022-01-05\n- 09:30 x\n')
  // Several lines stay one entry; an empty one changes nothing.
  expect(addJournalEntry('# Journal', at('2022-01-05 09:30', 'first\n\nsecond  '))).toBe('# Journal\n\n### 2022-01-05\n- 09:30 first\n  second')
  expect(addJournalEntry('# Journal', at('2022-01-05 09:30', '  \n'))).toBe('# Journal')
})

// The engine calls a pane with no anchor or one anchored on PMOC - Garden; the clock moves a minute a call.
const padHooks = (on: On, store: Map<string, unknown>, written: Map<string, string>, toasts: string[]): void => {
  let minute = 30
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', (_, e) => {
    toasts.push(e.text)
    return { value: undefined }
  })
  on('session.id', () => ({ value: 'own-session' }))
  on('clock.now', () => ({ value: new Date(2022, 0, 5, 9, minute++).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md'].includes(e.path) || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => ({
    value: { exitCode: 0, stdout: e.argv[0] === 'find' && e.argv.includes('PMOC - Garden.md') ? '/v/PMOC - Garden.md\n' : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false },
  }))
}

test('the scratchpad keeps notes for this chat, newest first; with no anchor nothing can be saved', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  const filled: string[] = []
  padHooks(on, new Map(), written, [])
  on('prompt.read', () => ({ value: { text: 'something half typed', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.input({ key: 'pad-input-0', text: 'check the hose' })
  await ui.input({ key: 'pad-input-1', text: 'buy seeds' })
  expect((await ui.find({ key: 'pad-row-0' }))?.text).toMatch(/buy seeds/)
  expect((await ui.find({ key: 'pad-row-1' }))?.text).toMatch(/check the hose/)
  expect(await ui.find({ key: 'pad-save-0' })).toBeFalsy()
  expect(await ui.find({ key: 'pad-save-all' })).toBeFalsy()
  expect(await ui.find({ text: 'Pick an anchor to save these to its Human Notes.' })).toBeTruthy()
  // An entry has two things to press at most, under its text: save (with an anchor) and drop.
  expect(await ui.find({ key: 'pad-ask-1' })).toBeFalsy()
  expect(await ui.find({ key: 'pad-today-1' })).toBeFalsy()
  expect((await ui.find({ key: 'pad-row-1' }))?.text).toMatch(/check the hose\s*✕ drop$/)
  expect(filled.length).toBe(0)
  await ui.press({ key: 'pad-drop-0' })
  expect((await ui.find({ key: 'pad-row-0' }))?.text).toMatch(/check the hose/)
  expect(await ui.find({ key: 'pad-row-1' })).toBeFalsy()
  expect(written.size).toBe(0)
  await ui.unmount()
})

test('with an anchor, save to notes puts the entry in the note\'s Journal and takes it off the pad', { options: OPTIONS }, async ($, on) => {
  const note = '---\noutcome: x\n---\n# Next\n- [ ] a\n\n# Journal\n\n### 2022-01-04\n- 08:00 older\n\n# Agent Log\n'
  const written = new Map<string, string>([['/v/PMOC - Garden.md', note]])
  const toasts: string[] = []
  padHooks(on, new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]]]), written, toasts)

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.input({ key: 'pad-input-0', text: 'keep this one' })
  await ui.input({ key: 'pad-input-1', text: 'a passing thought' })
  expect(await ui.find({ key: 'pad-save-all' })).toBeTruthy()
  // The user edits the note in their editor meanwhile: the save reads it fresh.
  const edited = note.replace('- 08:00 older', '- 08:00 older\n- 08:05 typed in the editor')
  written.set('/v/PMOC - Garden.md', edited)
  const time = (await ui.find({ key: 'pad-row-1' }))?.text.match(/\d\d:\d\d/)?.[0]
  expect((await ui.find({ key: 'pad-row-1' }))?.text).toMatch(/keep this one\s*→ save to notes\s*✕ drop$/)
  await ui.press({ key: 'pad-save-1' })
  expect(written.get('/v/PMOC - Garden.md')).toBe(
    edited.replace('- 08:05 typed in the editor\n', `- 08:05 typed in the editor\n\n### 2022-01-05\n- ${time} keep this one\n`),
  )
  expect(toasts).toEqual(["Saved to PMOC - Garden's Journal"])
  expect((await ui.find({ key: 'pad-row-0' }))?.text).toMatch(/a passing thought/)
  expect(await ui.find({ key: 'pad-row-1' })).toBeFalsy()
  expect(await ui.find({ key: 'pad-save-all' })).toBeFalsy()
  await ui.unmount()
})

// A vault anchored on PMOC - Garden, with this chat's transcript; `notes` are files by path,
// found by name. `session.id` reads `session.id` each call.
const taskHooks = (on: On, anchors: [string, string][], notes: Record<string, string>, session = { id: 'own-session' }) => {
  const transcript = '/Users/t/.claude/projects/-r-garden/own-session.jsonl'
  const written = new Map<string, string>()
  const store = new Map<string, unknown>([['anchors', anchors]])
  // `said` is the conversation the chat starts with: a fork's holds its parent's messages.
  const got = { written, store, transcript, said: [] as { role: string; content: unknown }[], filled: [] as string[], copied: [] as string[], ran: [] as string[][], toasts: [] as string[], beside: [] as string[][] }
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', (_, e) => {
    got.toasts.push(e.text)
    return { value: undefined }
  })
  on('ui.copy', (_, e) => {
    got.copied.push(e.text)
    return { value: { isCopied: true } }
  })
  on('session.id', () => ({ value: session.id }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/v/Kanban.md', '/Users/t/.claude/projects', transcript].includes(e.path) || written.has(e.path) || e.path in notes }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path === '/v/Kanban.md' ? BOARD_TEXT : notes[e.path] ?? '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', (_, e) => ({ value: e.path === '/Users/t/.claude/projects' ? [{ name: '-r-garden', kind: 'dir', size: 0, mtimeMs: 0, isLink: false }] : [] }))
  on('process.run', (_, e) => {
    got.ran.push(e.argv)
    const name = e.argv[0] === 'find' ? e.argv[e.argv.indexOf('-name') + 1] ?? '' : ''
    const found = name ? [...Object.keys(notes), ...written.keys()].find(path => path.endsWith(`/${name}`)) ?? '' : ''

    return { value: { exitCode: 0, stdout: found ? `${found}\n` : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('session.messages', () => ({ value: got.said as never }))
  on('prompt.read', () => ({ value: { text: 'half a thought', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    got.filled.push(e.text)
    return { isFilled: true }
  })
  // What entered beside each prompt for the model, unseen by the user.
  on('prompt.submit', (_, e) => {
    got.beside.push([...(e.context ?? [])])
    return { text: e.text }
  })

  return got
}

const GARDEN_WITH_CHILD = '---\noutcome: It loads fast.\njira: PROJ-1\n---\n# Next\n- [ ] Plant the beds\n- [ ] [[Child task]] wire it\n# Journal\n'
const GARDEN_PLAIN = '---\noutcome: It loads fast.\n---\n# Next\n- [ ] Plant the beds\n'

test('a sub-item that links a note opens it; one without a link is plain text', { options: OPTIONS }, async ($, on) => {
  const got = taskHooks(on, [['own-session', 'PMOC - Garden']], { '/v/PMOC - Garden.md': GARDEN_WITH_CHILD, '/v/Child task.md': '' })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ text: '🔗 PROJ-1' })).toBeTruthy()
  await ui.press({ key: 'anchor-details' })
  expect(await ui.find({ text: 'Plant the beds' })).toBeTruthy()
  await ui.press({ key: 'task-Child task wire it' })
  expect(got.ran.some(argv => argv[0] === 'open' && argv[1] === `obsidian://open?path=${encodeURIComponent('/v/Child task.md')}`)).toBe(true)
  await ui.unmount()
})

test('no task field is open until a button opens it; ＋ new task makes a standalone note and card, leaving the anchor and the prompt box; the just-made row', { options: OPTIONS }, async ($, on) => {
  const got = taskHooks(on, [['own-session', 'PMOC - Garden']], { '/v/PMOC - Garden.md': GARDEN_PLAIN })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'task-new' })).toBeTruthy()
  // A piece of the anchor's work is a fork of this chat, not a sub-task note: no such button.
  expect(await ui.find({ key: 'task-sub' })).toBeFalsy()
  expect((await ui.findAll({ type: 'Input' })).some(one => /^task-title/.test(String(one.key ?? '')))).toBe(false)
  // Typing a thought where the old field was makes nothing.
  try {
    await ui.input({ key: 'task-title-new', text: 'a stray thought' })
  } catch {
    // no such field
  }
  expect(got.written.size).toBe(0)

  // cancel, and an empty title, make nothing.
  await ui.press({ key: 'task-new' })
  expect(await ui.find({ key: 'task-title-new' })).toBeTruthy()
  await ui.press({ key: 'task-cancel' })
  expect(await ui.find({ key: 'task-title-new' })).toBeFalsy()
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: '   ' })
  expect(got.written.size).toBe(0)
  expect(await ui.find({ key: 'task-title-new' })).toBeFalsy()

  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Fix the gate PROJ-123' })
  const note = got.written.get('/v/Fix the gate.md') ?? ''
  expect(note.includes('Up:')).toBe(false)
  expect(note.includes('Jira: PROJ-123')).toBe(true)
  expect(note.includes('### 2026-11-08')).toBe(true)
  const board = got.written.get('/v/Kanban.md') ?? ''
  expect(without(board, '- [ ] [[Fix the gate]]')).toBe(BOARD_TEXT)
  // A standalone task lands where the chat is working: at the end of its anchor's column, not the board's first.
  expect(board.split('\n').indexOf('- [ ] [[Fix the gate]]')).toBe(BOARD_TEXT.split('\n').indexOf('\t- [ ] [[PMOC - Birdhouse]]') + 1)
  expect(kanbanCards(board).find(card => card.note === 'Fix the gate')?.column).toBe('Work')
  expect(kanbanCards(board).find(card => card.note === 'PMOC - Garden')?.tasks.length).toBe(2)
  expect(parseTrail(got.written.get('/v/Fix the gate.md') ?? '').pins).toEqual([
    { on: true, kind: 'session', target: got.transcript, label: 'the chat this task came from' },
  ])
  // A standalone task is this chat starting something of its own: the chat moves onto it,
  // is filed in its log, and the next draft carries the new task's context.
  expect(anchorOf(got.store, 'own-session')).toBe('Fix the gate')
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/Fix the gate/)
  expect(parseTrail(got.written.get('/v/Fix the gate.md') ?? '').chats.map(chat => chat.id)).toEqual(['own-session'])
  expect(got.filled).toEqual([])
  expect(await ui.find({ key: 'task-title-new' })).toBeFalsy()
  expect(got.toasts.some(text => text.includes('/v/Fix the gate.md') && text.includes('This chat is anchored on it now.') && text.includes('Setup → new task folder'))).toBe(true)

  // The just-made row says where on the board it went and opens the board.
  expect(await ui.find({ text: '✓ just made: Fix the gate' })).toBeTruthy()
  expect(await ui.find({ text: 'on the board: a card at the end of Work' })).toBeTruthy()
  expect(got.toasts.some(text => text.includes('a card at the end of Work'))).toBe(true)
  await ui.press({ key: 'made-board' })
  expect(got.ran.some(argv => argv[0] === 'open' && argv[1] === `obsidian://open?path=${encodeURIComponent('/v/Kanban.md')}`)).toBe(true)
  await ui.press({ key: 'made-open' })
  expect(got.ran.some(argv => argv[0] === 'open' && argv[1] === `obsidian://open?path=${encodeURIComponent('/v/Fix the gate.md')}`)).toBe(true)
  await ui.press({ key: 'made-define' })
  expect(got.filled).toEqual([newTaskDraft('/v/Fix the gate.md', 'Fix the gate')])
  // The half-typed prompt went to the clipboard first.
  expect(got.copied).toEqual(['half a thought'])
  // One way to draft: no copy-for-a-new-chat beside it.
  expect(await ui.find({ key: 'made-copy' })).toBeFalsy()
  // Already on it: nothing to press to get there, and the draft button has its context.
  expect(await ui.find({ key: 'made-anchor' })).toBeFalsy()
  await pressDraft(ui)
  expect(anchorFromDraft(got.filled[1] ?? '')).toBe('Fix the gate')
  expect(got.filled[1]?.includes('"/v/Fix the gate.md"')).toBe(true)
  await ui.press({ key: 'made-dismiss' })
  expect(await ui.find({ text: '✓ just made: Fix the gate' })).toBeFalsy()

  // The same title again writes nothing.
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Fix the gate' })
  expect(got.toasts.some(text => text.includes('already there'))).toBe(true)
  expect(got.written.get('/v/Kanban.md')).toBe(board)
  await ui.unmount()
})

test('a linked sub-item chosen from the anchor menu becomes the anchor, its outcome shown; a stored anchor off the board still shows', { options: OPTIONS }, async ($, on) => {
  const notes = { '/v/PMOC - Garden.md': GARDEN_PLAIN, '/v/PMOC - Birdhouse.md': '---\noutcome: Birds move in.\n---\n', '/v/Loose note.md': '### Outcome\n\nIt is tidy.\n' }
  const session = { id: 'own-session' }
  const got = taskHooks(on, [['own-session', 'PMOC - Garden'], ['loose', 'Loose note'], ['gone', 'Gone note']], notes, session)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'anchor-menu' })
  expect((await ui.find({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' }))?.text).toMatch(/○ PMOC - Birdhouse/)
  expect(await ui.find({ key: 'card-sub-PMOC - Garden-Plant the beds' })).toBeFalsy()
  await ui.press({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' })
  expect(anchorOf(got.store, 'own-session')).toBe('PMOC - Birdhouse')
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Birdhouse/)
  await ui.press({ key: 'anchor-details' })
  expect(await ui.find({ text: '🎯 Birds move in.' })).toBeTruthy()
  expect(await ui.find({ text: 'under PMOC - Garden' })).toBeTruthy()
  expect(parseTrail(got.written.get('/v/PMOC - Birdhouse.md') ?? '').chats.map(chat => chat.id)).toEqual(['own-session'])

  // Not on the board, but a note of that name exists: a bare anchor.
  session.id = 'loose'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/Loose note/)
  expect(await ui.find({ text: '🎯 It is tidy.' })).toBeTruthy()
  expect(await ui.find({ key: 'open-note' })).toBeTruthy()
  expect(anchorOf(got.store, 'loose')).toBe('Loose note')
  // No such note anywhere: no anchor.
  session.id = 'gone'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await ui.unmount()
})

test('a Jira key comes from a bare key or an issue link; lowercase and other text are not keys', async () => {
  expect(jiraKeyFrom('PROJ-123')).toBe('PROJ-123')
  expect(jiraKeyFrom('  (PROJ-123), ')).toBe('PROJ-123')
  expect(jiraKeyFrom('https://example.atlassian.net/browse/PROJ-7?focus=1')).toBe('PROJ-7')
  expect(jiraKeyFrom('https://example.atlassian.net/browse/PROJ-7')).toBe('PROJ-7')
  expect(jiraKeyFrom('proj-123')).toBe('')
  expect(jiraKeyFrom('PROJ')).toBe('')
  expect(jiraKeyFrom('https://example.com/issues/PROJ-7')).toBe('')
  expect(jiraBrowseUrl('https://example.atlassian.net/', 'PROJ-7')).toBe('https://example.atlassian.net/browse/PROJ-7')
})

test('jira: in the frontmatter is added, replaced or removed, every other byte as it was', async () => {
  const note = '---\noutcome: It loads fast.\nstatus: active\n---\n# Next\n- [ ] jira: not this one\n'
  const added = withJira(note, 'PROJ-1')
  expect(added).toBe('---\noutcome: It loads fast.\nstatus: active\njira: PROJ-1\n---\n# Next\n- [ ] jira: not this one\n')
  expect(withJira(added, 'PROJ-2')).toBe(added.replace('jira: PROJ-1', 'jira: PROJ-2'))
  expect(withJira(added, '')).toBe(note)
  const middle = '---\njira: PROJ-1\noutcome: x\n---\nbody'
  expect(withJira(middle, 'PROJ-9')).toBe('---\njira: PROJ-9\noutcome: x\n---\nbody')
  expect(withJira(middle, '')).toBe('---\noutcome: x\n---\nbody')
  // No frontmatter: none is made. The key is a body line, under Up: when there is one, else first.
  expect(withJira('# Next\n', 'PROJ-1')).toBe('Jira: PROJ-1\n\n# Next\n')
  expect(withJira('# Next\n', '')).toBe('# Next\n')
  expect(withJira('', 'PROJ-1')).toBe('Jira: PROJ-1\n')
  // A body rule (---) is not frontmatter.
  expect(withJira('# A\n---\nb\n', 'PROJ-1')).toBe('Jira: PROJ-1\n\n# A\n---\nb\n')
  const flat = 'Up: [[PMOC - Garden]]\n\n### Outcome\n'
  expect(withJira(flat, 'PROJ-1')).toBe('Up: [[PMOC - Garden]]\nJira: PROJ-1\n\n### Outcome\n')
  expect(withJira(withJira(flat, 'PROJ-1'), 'PROJ-2')).toBe('Up: [[PMOC - Garden]]\nJira: PROJ-2\n\n### Outcome\n')
  expect(withJira(withJira(flat, 'PROJ-1'), '')).toBe(flat)
  expect(withJira(withJira('# Next\n', 'PROJ-1'), '')).toBe('# Next\n')
  expect(jiraFrom(withJira(flat, 'PROJ-1'))).toBe('PROJ-1')
  // A line of the user's own that starts with Jira: is not the link, and is never written over.
  const prose = 'Up: [[PMOC - Garden]]\n\n# Journal\nJira: waiting on access\n'
  expect(jiraFrom(prose)).toBe('')
  expect(withJira(prose, 'PROJ-1')).toBe('Up: [[PMOC - Garden]]\nJira: PROJ-1\n\n# Journal\nJira: waiting on access\n')
  expect(withJira(withJira(prose, 'PROJ-1'), '')).toBe(prose)
  // Frontmatter without the field but a key line in the body: that line is the link.
  const mixed = '---\na: 1\n---\nUp: [[X]]\nJira: PROJ-1\n'
  expect(withJira(mixed, 'PROJ-2')).toBe('---\na: 1\n---\nUp: [[X]]\nJira: PROJ-2\n')
  expect(withJira(mixed, '')).toBe('---\na: 1\n---\nUp: [[X]]\n')
  expect(taskNoteName('.NET: upgrade')).toBe('NET- upgrade')
  // A one-column board with no archive: the card stays above the Kanban settings block.
  const settings = '## Doing\n\n- [ ] [[A]]\n\n\n%% kanban:settings\n```\n{}\n```\n%%'
  expect(insertCard(settings, 'B', null)).toBe('## Doing\n\n- [ ] [[A]]\n- [ ] [[B]]\n\n\n%% kanban:settings\n```\n{}\n```\n%%')
  expect(jiraFrom(withJira(note, 'PROJ-1'))).toBe('PROJ-1')
})

test('the Jira server is the one the setting names, else a likely name among the MCP tools', async () => {
  const tools = [
    { name: 'Read', mcp: false },
    { name: 'mcp__openaugi__search', mcp: true },
    { name: 'mcp__claude_ai_Atlassian__getJiraIssue', mcp: true },
  ]
  expect(jiraServerFrom(tools, '')).toBe('claude_ai_Atlassian')
  expect(jiraServerFrom(tools, 'claude.ai Atlassian')).toBe('claude.ai Atlassian')
  expect(jiraServerFrom(tools, 'jira')).toBe('')
  expect(jiraServerFrom([{ name: 'mcp__jira__search', mcp: true }], '')).toBe('jira')
  expect(jiraServerFrom(tools.slice(0, 2), '')).toBe('')
})

test('find in Jira drafts a search that changes nothing in Jira, with and without a project and server', async () => {
  const both = findInJiraDraft('/v/PMOC - Garden.md', 'PMOC - Garden', { project: 'PROJ', server: 'atlassian' })
  expect(both.startsWith('Context for this chat:\n- Anchor note: "/v/PMOC - Garden.md" (PMOC - Garden).')).toBe(true)
  expect(both.includes('Search Jira in project PROJ using the atlassian MCP server for issues')).toBe(true)
  expect(both.includes('up to five likely matches, one line each: KEY · summary · status')).toBe(true)
  expect(both.includes('Do not create or change anything in Jira.')).toBe(true)
  expect(both.includes("add the line `Jira: <KEY>` under its `Up:` line")).toBe(true)
  expect(both.includes('offer to create an issue and wait for my yes')).toBe(true)
  expect(both.endsWith('\nInstruction: find the Jira issue for PMOC - Garden')).toBe(true)
  expect(chatTitleFrom(both)).toBe('find the Jira issue for PMOC - Garden')
  const neither = findInJiraDraft('', 'PMOC - Garden', { project: '', server: '' })
  expect(neither.includes('[[PMOC - Garden]] (find it in "/v")')).toBe(true)
  expect(neither.includes('Search Jira with whatever Jira tools you have; if you have none, say so and stop for issues')).toBe(true)
  expect(neither.includes('in project')).toBe(false)
})

test('update Jira drafts a short update shown first and posted only on a yes', async () => {
  const text = jiraUpdateDraft('/v/PMOC - Garden.md', 'PMOC - Garden', 'PROJ-1', { project: 'PROJ', server: 'atlassian' })
  expect(text.startsWith('Context for this chat:\n')).toBe(true)
  for (const part of ['read its Outcome, Verify, Journal and Agent Log sections', 'Read Jira issue PROJ-1 and its recent comments using the atlassian MCP server', 'only what is new since the last one', 'at most six lines: what moved, what is next, what is blocked', 'Show it to me first', 'comment on PROJ-1 only after I say yes', 'status only if I ask']) {
    expect(text.includes(part)).toBe(true)
  }
  expect(text.endsWith('\nInstruction: update Jira PROJ-1')).toBe(true)
  const bare = jiraUpdateDraft('/v/PMOC - Garden.md', 'PMOC - Garden', 'PROJ-1', { project: '', server: '' })
  expect(bare.includes('with whatever Jira tools you have; if you have none, say so and stop')).toBe(true)
})

test('a new-task draft looks for the Jira issue only when asked', async () => {
  const plain = newTaskDraft('/v/Fix the gate.md', 'Fix the gate')
  expect(plain.includes('Jira')).toBe(false)
  const asked = newTaskDraft('/v/Fix the gate.md', 'Fix the gate', { project: 'PROJ', server: '' })
  expect(asked.includes('Search Jira in project PROJ with whatever Jira tools you have')).toBe(true)
  expect(asked.includes('offer to create an issue and wait for my yes')).toBe(true)
  expect(asked.split('\n').filter(line => !line.includes('Jira') && !line.includes('jira'))).toEqual(plain.split('\n'))
  expect(asked.endsWith('Instruction: fill in the new task Fix the gate')).toBe(true)
})

const GARDEN_IN_FOLDER = '/v/Projects/PMOC - Garden.md'

// A vault with the anchor's note in a subfolder; `tools` is what $.tool.list answers.
const jiraHooks = (on: On, note: string, tools: { name: string; description: string; mcp: boolean }[], extra: string[] = []) => {
  const written = new Map<string, string>([[GARDEN_IN_FOLDER, note]])
  const store = new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]]])
  const got = { written, store, filled: [] as string[], ran: [] as string[][], toasts: [] as string[], sent: [] as string[] }
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', (_, e) => {
    got.toasts.push(e.text)
    return { value: undefined }
  })
  on('session.id', () => ({ value: 'own-session' }))
  on('clock.now', () => ({ value: new Date(2022, 0, 5, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('tool.list', () => ({ value: tools }))
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/v/Kanban.md', ...extra].includes(e.path) || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path === '/v/Kanban.md' ? BOARD_TEXT : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => {
    got.ran.push([...e.argv])
    const name = e.argv[0] === 'find' ? e.argv[e.argv.indexOf('-name') + 1] ?? '' : ''
    const found = name ? [...written.keys()].find(path => path.endsWith(`/${name}`)) ?? '' : ''

    return { value: { exitCode: 0, stdout: found ? `${found}\n` : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('prompt.read', () => ({ value: { text: '', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    got.filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => {
    got.sent.push(e.text)
    return { text: e.text }
  })

  return got
}

const GARDEN_NOTE = '---\noutcome: It loads fast.\n---\n# Next\n- [ ] Plant the beds\n'

test('with no Jira settings, no server and an unlinked anchor, the pane draws nothing about Jira', { options: OPTIONS }, async ($, on) => {
  jiraHooks(on, GARDEN_NOTE, [{ name: 'mcp__openaugi__search', description: 'search', mcp: true }])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'open-note' })).toBeTruthy()
  for (const key of ['jira-link', 'jira-find', 'jira-update', 'jira-unlink', 'jira-open', 'jira-key']) {
    expect(await ui.find({ key })).toBeFalsy()
  }
  await ui.unmount()
})

for (const surface of ['terminal', 'desktop'] as const) {
  test(`with a Jira project set (${surface}): a pasted link links the anchor, find and update fill the prompt and send nothing, unlink removes it`, { options: { ...OPTIONS, jiraProject: 'PROJ' } }, async ($, on) => {
    const got = jiraHooks(on, GARDEN_NOTE, [])
    await $.command.run({ command: 'augi', args: '' })
    const ui = await $.ui.mount({ ...PANE, surface })
    expect(await ui.find({ key: 'jira-link' })).toBeTruthy()

    await ui.press({ key: 'jira-find' })
    expect(got.filled).toEqual([findInJiraDraft(GARDEN_IN_FOLDER, 'PMOC - Garden', { project: 'PROJ', server: '' })])

    // The note was edited meanwhile: the link reads it fresh and changes only its frontmatter.
    const edited = `${GARDEN_NOTE}- [ ] typed in the editor\n`
    got.written.set(GARDEN_IN_FOLDER, edited)
    await ui.input({ key: 'jira-link', text: 'https://example.atlassian.net/browse/PROJ-42' })
    expect(got.written.get(GARDEN_IN_FOLDER)).toBe(withJira(edited, 'PROJ-42'))
    expect(await ui.find({ text: '🔗 PROJ-42' })).toBeTruthy()
    expect(await ui.find({ key: 'jira-link' })).toBeFalsy()
    expect(await ui.find({ key: 'jira-open' })).toBeFalsy()

    await ui.press({ key: 'jira-update' })
    expect(got.filled[1]).toBe(jiraUpdateDraft(GARDEN_IN_FOLDER, 'PMOC - Garden', 'PROJ-42', { project: 'PROJ', server: '' }))

    await ui.press({ key: 'jira-unlink' })
    expect(got.written.get(GARDEN_IN_FOLDER)).toBe(edited)
    expect(await ui.find({ key: 'jira-link' })).toBeTruthy()

    // Not a key: nothing written.
    await ui.input({ key: 'jira-link', text: 'proj-42' })
    expect(got.written.get(GARDEN_IN_FOLDER)).toBe(edited)
    expect(got.toasts.some(text => text.startsWith('Not a Jira key'))).toBe(true)
    expect(got.sent).toEqual([])
    await ui.unmount()
  })
}

test('with a Jira site set, the linked key opens the issue in the browser', { options: { ...OPTIONS, jiraSite: 'https://example.atlassian.net/' } }, async ($, on) => {
  const got = jiraHooks(on, '---\njira: PROJ-1\n---\n# Next\n', [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'jira-open' })
  expect(got.ran.some(argv => argv[0] === 'open' && argv[1] === 'https://example.atlassian.net/browse/PROJ-1')).toBe(true)
  await ui.unmount()
})

test('a jira-update lens in the lens folder writes the update draft; a found server names itself in drafts', { options: OPTIONS }, async ($, on) => {
  const got = jiraHooks(on, '---\njira: PROJ-1\n---\n# Next\n', [{ name: 'mcp__atlassian__getJiraIssue', description: 'get', mcp: true }], ['/v/lenses/jira-update.md'])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'jira-update' })
  expect(got.filled[0]?.includes('- Lens: read "/v/lenses/jira-update.md"')).toBe(true)
  expect(got.filled[0]?.endsWith('Instruction: update Jira PROJ-1')).toBe(true)
  expect(chatTitleFrom(got.filled[0] ?? '')).toBe('jira-update: update Jira PROJ-1')

  await ui.press({ key: 'jira-unlink' })
  await ui.press({ key: 'jira-find' })
  expect(got.filled[1]?.includes('using the atlassian MCP server')).toBe(true)
  expect(got.sent).toEqual([])
  await ui.unmount()
})

test('a new task goes to the task folder even when the anchor note sits in a subfolder, and its draft looks for its issue', { options: { ...OPTIONS, taskFolder: 'Tasks', jiraProject: 'PROJ' } }, async ($, on) => {
  const got = jiraHooks(on, GARDEN_NOTE, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Fix the gate' })
  expect(got.written.has('/v/Tasks/Fix the gate.md')).toBe(true)
  expect(got.written.get('/v/Tasks/Fix the gate.md')?.includes('Up: ')).toBe(false)
  expect(got.toasts.some(text => text.includes('Setup → new task folder'))).toBe(false)
  expect(got.filled).toEqual([])
  await ui.press({ key: 'made-define' })
  expect(got.filled).toEqual([newTaskDraft('/v/Tasks/Fix the gate.md', 'Fix the gate', { project: 'PROJ', server: '' })])
  await ui.unmount()
})

test('with no task folder, a new task goes to the notes root, never beside the anchor note', { options: OPTIONS }, async ($, on) => {
  const got = jiraHooks(on, GARDEN_NOTE, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Fix the gate' })
  expect(got.written.has('/v/Fix the gate.md')).toBe(true)
  expect(got.written.has('/v/Projects/Fix the gate.md')).toBe(false)
  await ui.press({ key: 'made-define' })
  expect(got.filled).toEqual([newTaskDraft('/v/Fix the gate.md', 'Fix the gate')])
  await ui.unmount()
})

test('a chat with no anchor puts a standalone task in the task column setting, else in the first column', { options: { ...OPTIONS, taskColumn: 'work' } }, async ($, on) => {
  const got = taskHooks(on, [], {})
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Fix the gate' })
  const board = got.written.get('/v/Kanban.md') ?? ''
  expect(kanbanCards(board).find(card => card.note === 'Fix the gate')?.column).toBe('Work')
  expect(without(board, '- [ ] [[Fix the gate]]')).toBe(BOARD_TEXT)
  expect(await ui.find({ text: 'on the board: a card at the end of Work' })).toBeTruthy()
  await ui.unmount()
})

test('a chat with no anchor and no task column puts a standalone task in the first column, and says which', { options: OPTIONS }, async ($, on) => {
  const got = taskHooks(on, [], {})
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Fix the gate' })
  expect(kanbanCards(got.written.get('/v/Kanban.md') ?? '').find(card => card.note === 'Fix the gate')?.column).toBe('Ideas')
  expect(await ui.find({ text: 'on the board: a card at the end of Ideas' })).toBeTruthy()
  await ui.unmount()
})

// The task template as it shipped before the Agent Log left the note.
const OLD_TASK_TEMPLATE = '---\noutcome: \nstatus: active\n---\n\nUp: [[PMOC - ]]\n\n# Next\n\n- [ ] \n\n# Journal\n\n### {{date:YYYY-MM-DD}}\n\n# Agent Log\n\n> [!abstract]- Agent Log\n> ![[Agent Log - {{title}}]]\n'

test('a task template is the built-in one or the notes folder\'s own, whatever its own holds', async () => {
  expect(taskTemplateState('')).toBe('built-in')
  expect(taskTemplateState('  \n')).toBe('built-in')
  expect(taskTemplateState(TASK_TEMPLATE)).toBe('yours')
  expect(taskTemplateState(OLD_TASK_TEMPLATE)).toBe('yours')
})

test('an anchor that still has an Agent Log file: opening the pane on it moves the file into the note, once, and marks the file', { options: OPTIONS }, async ($, on) => {
  const file = '/v/Notes/Agent Log - PMOC - Garden.md'
  const old = oldLogFile('PMOC - Garden', '2026-11-01')
    .replace('## Context\n', '## Context\n\n- [[PMOC - Birdhouse]] · the sister project\n- the valve sticks below 4 degrees\n')
    .replace('## Chats\n', '## Chats\n\n- 2026-11-01 10:00 · fix the valve · `earlier`\n')
  const project = `${GARDEN_PLAIN}\n# Agent Log\n\n> [!abstract]- Agent Log\n> ![[Agent Log - PMOC - Garden]]\n`
  const got = taskHooks(on, [['own-session', 'PMOC - Garden']], { '/v/PMOC - Garden.md': project, '/v/PMOC - Birdhouse.md': '', [file]: old })
  expect(trailPath('PMOC - Garden')).toBe(file)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  // The pane shows what the file pinned, read from the note it now lives in.
  expect((await ui.find({ key: 'pin-open-note-PMOC - Birdhouse' }))?.text).toMatch(/the sister project/)
  expect(got.written.get('/v/PMOC - Garden.md')).toBe(
    `${GARDEN_PLAIN}\n# Agent Log\n\n- [[PMOC - Birdhouse]] · the sister project\n- the valve sticks below 4 degrees\n\n> [!abstract]- Chats\n> - 2026-11-01 10:00 · fix the valve · \`earlier\`\n`,
  )
  // The file is kept, marked, and left alone from then on.
  const marked = got.written.get(file) ?? ''
  expect(marked.includes('Moved into [[PMOC - Garden]] on 2026-11-08')).toBe(true)
  expect(marked.includes('- [[PMOC - Birdhouse]] · the sister project')).toBe(true)
  // The only way to the log is the note: one button.
  expect(await ui.find({ key: 'open-note' })).toBeTruthy()
  expect(await ui.find({ key: 'open-trail' })).toBeFalsy()

  // A later write goes to the note alone, and the file's lines are not brought over again.
  await ui.input({ key: 'pin-paste-1', text: 'https://example.com/valves' })
  const note = got.written.get('/v/PMOC - Garden.md') ?? ''
  expect(note.split('- [[PMOC - Birdhouse]]').length).toBe(2)
  expect(note.includes('- url: `https://example.com/valves` · example.com/valves\n\n> [!abstract]- Chats\n')).toBe(true)
  expect(got.written.get(file)).toBe(marked)
  await ui.press({ key: 'refresh' })
  expect(got.written.get('/v/PMOC - Garden.md')).toBe(note)
  await ui.unmount()
})

test('a task template that still embeds an Agent Log file is what new tasks get, without the embed; Setup has nothing to say about it', { options: OPTIONS }, async ($, on) => {
  const template = '/v/OpenAugi/AGENT/templates/Task.md'
  const got = taskHooks(on, [['own-session', 'PMOC - Garden']], { '/v/PMOC - Garden.md': GARDEN_PLAIN, [template]: OLD_TASK_TEMPLATE })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ text: /setup items? needs? you/ })).toBeFalsy()
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Old shape' })
  const made = got.written.get('/v/Old shape.md') ?? ''
  // The template is theirs: its frontmatter and sections are kept, its empty Up line gone.
  // Its Agent Log heading now holds the log itself (this chat pinned), not an embed of a
  // file that will never exist.
  expect(made.startsWith('---\noutcome: \nstatus: active\n---\n\n# Next\n')).toBe(true)
  expect(made.includes('![[Agent Log - ')).toBe(false)
  expect(made.includes('\n# Agent Log\n\n- session: `')).toBe(true)
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: '✓ task template: OpenAugi/AGENT/templates/Task.md' })).toBeTruthy()
  expect(await ui.find({ key: 'template-update' })).toBeFalsy()
  expect(got.written.has(template)).toBe(false)
  await ui.unmount()
})

test('Setup names the built-in task template when the notes folder has none, and warns while no task folder is set', { options: OPTIONS }, async ($, on) => {
  taskHooks(on, [], {})
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ text: /setup items? needs? you/ })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: '✓ task template: built-in (Outcome, Verify, Journal, Agent Log)' })).toBeTruthy()
  // The log is in the note now: there is no folder for it to set.
  expect(await ui.find({ key: 'set-logFolder' })).toBeFalsy()
  expect(await ui.find({ text: /No task folder set: new task notes go to the notes folder's root/ })).toBeTruthy()
  expect(await ui.find({ key: 'set-taskColumn' })).toBeTruthy()
  await ui.unmount()
})

test('with a task folder set, Setup does not warn about it', { options: { ...OPTIONS, taskFolder: 'Tasks' } }, async ($, on) => {
  taskHooks(on, [], {})
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: /No task folder set/ })).toBeFalsy()
  await ui.unmount()
})

test('a date is written the way Obsidian names a daily note, with [text] kept', async () => {
  const day = new Date(2022, 0, 5, 9, 30).getTime()
  expect(formatDate('YYYY-MM-DD', day)).toBe('2022-01-05')
  expect(formatDate('YYYY/MM/YYYY-MM-DD dddd', day)).toBe('2022/01/2022-01-05 Wednesday')
  expect(formatDate('D MMM YY (ddd)', day)).toBe('5 Jan 22 (Wed)')
  expect(formatDate('MMMM D, YYYY', day)).toBe('January 5, 2022')
  expect(formatDate('[Daily] YYYY-MM-DD', day)).toBe('Daily 2022-01-05')
  expect(formatDate('[DD is day] DD', day)).toBe('DD is day 05')
  expect(formatDate('M-D', new Date(2022, 10, 23).getTime())).toBe('11-23')
})

test('the daily note pattern is what was typed, else Obsidian\'s own setting, else none; none typed turns it off', async () => {
  expect(dailyPattern('Daily/{{date:YYYY-MM-DD}}.md', '{"folder":"Elsewhere"}')).toBe('Daily/{{date:YYYY-MM-DD}}.md')
  expect(dailyPattern('', '{"folder":"Journal/Daily","format":"YYYY/MM/YYYY-MM-DD"}')).toBe('Journal/Daily/{{date:YYYY/MM/YYYY-MM-DD}}.md')
  // Obsidian writes only what differs from its defaults: the vault root and YYYY-MM-DD.
  expect(dailyPattern('', '{"template":"Templates/Daily"}')).toBe('{{date:YYYY-MM-DD}}.md')
  expect(dailyPattern('', '{"folder":"/Daily/","format":"  "}')).toBe('Daily/{{date:YYYY-MM-DD}}.md')
  // With no folder named, Obsidian puts the daily note where it puts any new note.
  const inbox = '{"newFileLocation":"folder","newFileFolderPath":"Inbox/Fleeting"}'
  expect(dailyPattern('', '{"template":"Templates/Daily"}', inbox)).toBe('Inbox/Fleeting/{{date:YYYY-MM-DD}}.md')
  expect(dailyPattern('', '{"format":"YYYY-MM-DD dddd"}', inbox)).toBe('Inbox/Fleeting/{{date:YYYY-MM-DD dddd}}.md')
  // A folder named for daily notes wins over it; so does anything typed.
  expect(dailyPattern('', '{"folder":"Daily"}', inbox)).toBe('Daily/{{date:YYYY-MM-DD}}.md')
  expect(dailyPattern('Mine/{{date}}', '{}', inbox)).toBe('Mine/{{date}}')
  // New notes in the vault root, beside the current file, or settings that do not parse: the root.
  expect(dailyPattern('', '{}', '{"newFileLocation":"root","newFileFolderPath":"Old"}')).toBe('{{date:YYYY-MM-DD}}.md')
  expect(dailyPattern('', '{}', '{"newFileLocation":"current"}')).toBe('{{date:YYYY-MM-DD}}.md')
  expect(dailyPattern('', '{}', 'not json')).toBe('{{date:YYYY-MM-DD}}.md')
  expect(dailyPattern('', '', inbox)).toBe('')
  expect(dailyPattern('', '')).toBe('')
  expect(dailyPattern('', 'not json')).toBe('')
  expect(dailyPattern('none', '{"folder":"Daily"}')).toBe('')
  expect(dailyPattern('None', '')).toBe('')
})

test('a daily note made from the pane uses the template Obsidian names, its own placeholders filled and nothing else touched', async () => {
  expect(dailyTemplatePath('{"template":"Templates/Daily"}', '/v')).toBe('/v/Templates/Daily.md')
  expect(dailyTemplatePath('{"template":"/Templates/Daily.md"}', '/v')).toBe('/v/Templates/Daily.md')
  expect(dailyTemplatePath('{"folder":"Daily"}', '/v')).toBe('')
  expect(dailyTemplatePath('', '/v')).toBe('')
  expect(dailyTemplatePath('not json', '/v')).toBe('')
  const at = new Date(2022, 0, 5, 9, 7).getTime()
  expect(fillTemplate('# {{title}}\n{{date}} · {{date:dddd D MMMM}} · {{time}} · {{time:HH}}h\n- [ ] \n', at, '2022-01-05')).toBe('# 2022-01-05\n2022-01-05 · Wednesday 5 January · 09:07 · 09h\n- [ ] \n')
  // A scripting plugin's code and unknown placeholders stay exactly as written.
  expect(fillTemplate('<% tp.date.now() %> {{mood}} {{ date }}', at, 't')).toBe('<% tp.date.now() %> {{mood}} {{ date }}')
})

test('today\'s note path fills the date, adds .md, and sits in the notes folder unless absolute', async () => {
  const day = new Date(2022, 0, 5, 9, 30).getTime()
  expect(dailyNotePath('Daily/{{date:YYYY-MM-DD}}.md', day, '/v', '/Users/t')).toBe('/v/Daily/2022-01-05.md')
  expect(dailyNotePath('Daily/{{date}}', day, '/v', '/Users/t')).toBe('/v/Daily/2022-01-05.md')
  expect(dailyNotePath('{{date:YYYY}}/{{date:MM}}/{{date:YYYY-MM-DD dddd}}', day, '/v', '/Users/t')).toBe('/v/2022/01/2022-01-05 Wednesday.md')
  expect(dailyNotePath('~/notes/{{date:YYYY-MM-DD}}.md', day, '/v', '/Users/t')).toBe('/Users/t/notes/2022-01-05.md')
  // A folder named like a token is left alone: only what is inside {{date:…}} is a date.
  expect(dailyNotePath('DD Diary/{{date:YYYY-MM-DD}}', day, '/v', '/Users/t')).toBe('/v/DD Diary/2022-01-05.md')
  expect(dailyNotePath('', day, '/v', '/Users/t')).toBe('')
  expect(dailyNotePath('  ', day, '/v', '/Users/t')).toBe('')
})

test('what is still open in a note is its first open checkboxes, links read as text', async () => {
  const text = '# Priorities\n- [ ] ship [[PMOC - Garden|the garden]]\n- [x] done\n  - [ ] nested one\n* [ ] star box\n- plain\n- [ ] \n- [ ] a\n- [ ] b\n- [ ] c\n- [ ] d\n'
  expect(openBoxes(text)).toEqual(['ship the garden', 'nested one', 'star box', 'a', 'b'])
  expect(openBoxes(text, 2)).toEqual(['ship the garden', 'nested one'])
  expect(openBoxes('nothing here')).toEqual([])
})

test('a day\'s line can go under a heading that starts with the date, made at the level asked when missing; a longer date is not the same day', async () => {
  const one = { when: '2022-01-05T09:30', text: 'hung it' }
  expect(addDailyLine('# This week\n- [ ] a\n', '2022-01-05', one, 3, true)).toBe('# This week\n- [ ] a\n\n### 2022-01-05\n- 09:30 hung it\n')
  expect(addDailyLine('### 2022-01-05 gate day\nnotes\n', '2022-01-05', one, 3, true)).toBe('### 2022-01-05 gate day\nnotes\n- 09:30 hung it\n')
  // Without isPrefix only the exact title counts; a different date never does.
  expect(addDailyLine('### 2022-01-05 gate day\n', '2022-01-05', one, 3)).toBe('### 2022-01-05 gate day\n\n### 2022-01-05\n- 09:30 hung it\n')
  expect(addDailyLine('### 2022-01-050\n', '2022-01-05', one, 3, true)).toBe('### 2022-01-050\n\n### 2022-01-05\n- 09:30 hung it\n')
})

test('a logged line goes last under the named heading, made when missing, or at the end of the note; every other line as it was', async () => {
  const one = { when: '2022-01-05 09:30', text: 'fixed the gate\nand oiled it' }
  // No heading named: the note's last line, with a newline between when the note had none.
  expect(addDailyLine('# Today\n- [ ] a\n', '', one)).toBe('# Today\n- [ ] a\n- 09:30 fixed the gate\n  and oiled it\n')
  expect(addDailyLine('# Today\n- [ ] a', '', one)).toBe('# Today\n- [ ] a\n- 09:30 fixed the gate\n  and oiled it\n')
  expect(addDailyLine('', '', one)).toBe('- 09:30 fixed the gate\n  and oiled it\n')
  // Under the heading, after its last line and before the next heading of its level; any case, any level.
  const note = '---\ntags: daily\n---\n# Priorities\n- [ ] a\n\n## Log\n- 08:00 coffee\n\n### sub\n- 08:10 deeper\n\n## Later\n- x\n'
  expect(addDailyLine(note, 'log', one)).toBe(note.replace('- 08:10 deeper\n', '- 08:10 deeper\n- 09:30 fixed the gate\n  and oiled it\n'))
  // An empty section takes it right under the heading.
  expect(addDailyLine('## Log\n\n## Later\n', 'Log', one)).toBe('## Log\n- 09:30 fixed the gate\n  and oiled it\n\n## Later\n')
  // The heading is the note's last section: the note still ends with a newline.
  expect(addDailyLine('# Priorities\n\n## Log\n- 08:00 coffee\n', 'Log', one)).toBe('# Priorities\n\n## Log\n- 08:00 coffee\n- 09:30 fixed the gate\n  and oiled it\n')
  expect(addDailyLine('# Priorities\n\n## Log', 'Log', one)).toBe('# Priorities\n\n## Log\n- 09:30 fixed the gate\n  and oiled it\n')
  // No such heading: made at the end.
  expect(addDailyLine('# Priorities\n- [ ] a\n', 'Log', one)).toBe('# Priorities\n- [ ] a\n\n## Log\n- 09:30 fixed the gate\n  and oiled it\n')
  expect(addDailyLine('', 'Log', one)).toBe('## Log\n- 09:30 fixed the gate\n  and oiled it\n')
  expect(addDailyLine(note, 'Log', { when: '2022-01-05 09:30', text: '  ' })).toBe(note)
})

test('the draft names today\'s note when there is one and it is ticked', async () => {
  const withToday = buildDraft([], ALL_ON, null, 'go', ANCHOR, '/v/2022-01-05.md')
  expect(withToday.includes('- Today: read "/v/2022-01-05.md"')).toBe(true)
  expect(buildDraft([], { ...ALL_ON, today: false }, null, 'go', ANCHOR, '/v/2022-01-05.md').includes('Today:')).toBe(false)
  expect(buildDraft([], ALL_ON, null, 'go', ANCHOR, '').includes('Today:')).toBe(false)
  // What a draft asks the chat to keep is written as a bullet, which pins it.
  expect(withToday.includes('"- [[<note>]] · <why it matters>" pins a note')).toBe(true)
})

// padHooks, with today's note found from Obsidian's daily-note settings; `clock` can be moved.
const todayHooks = (on: On, written: Map<string, string>, toasts: string[], clock = { now: new Date(2022, 0, 5, 9, 30).getTime() }) => {
  const filled: string[] = []
  padHooks(except(on, ['clock.now']), new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]]]), written, toasts)
  on('clock.now', () => ({ value: clock.now }))
  on('fs.stat', (_, e) => ({ value: { kind: 'file', size: 1, mtimeMs: (written.get(e.path) ?? '').length + 1, isLink: false } }))
  on('prompt.read', () => ({ value: { text: 'what next', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))

  return { filled, clock }
}

test('with no daily note setting and no Obsidian daily notes, the You card says where to set one and shows nothing about today', { options: OPTIONS }, async ($, on) => {
  todayHooks(on, new Map(), [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'today-open' })).toBeFalsy()
  expect(await ui.find({ key: 'today-make' })).toBeFalsy()
  expect(await ui.find({ text: /No daily note set: name one under Setup/ })).toBeTruthy()
  // There is no separate Today card: today's note lives in the You card.
  expect(await ui.find({ key: 'head-todayOpen' })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: 'No daily note: nothing about today is shown.' })).toBeTruthy()
  await ui.unmount()
})

test('today\'s note in the You card: found where Obsidian puts new notes, it lists what is open, logs a line under the heading, and goes into drafts until switched off under Setup', { options: { ...OPTIONS, dailyHeading: '## Log' } }, async ($, on) => {
  const note = '# Priorities\n- [ ] ship the gate\n- [x] water\n- [ ] call the plumber\n\n## Log\n- 08:00 coffee\n'
  // Daily notes name a template but no folder; new notes go to Inbox.
  const written = new Map<string, string>([
    ['/v/.obsidian/daily-notes.json', '{"template":"Templates/Daily"}'],
    ['/v/.obsidian/app.json', '{"newFileLocation":"folder","newFileFolderPath":"Inbox"}'],
    ['/v/Inbox/2022-01-05.md', note],
    ['/v/PMOC - Garden.md', '# Journal\n'],
  ])
  const today = '/v/Inbox/2022-01-05.md'
  const toasts: string[] = []
  const got = todayHooks(on, written, toasts)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  // One card: context, board and today's note side by side, then what today's note still has open.
  expect((await ui.find({ key: 'today-open' }))?.text).toMatch(/today's note/)
  expect(await ui.find({ key: 'today-make' })).toBeFalsy()
  expect(await ui.find({ key: 'edit-context' })).toBeTruthy()
  expect(await ui.find({ text: 'Today · 2022-01-05' })).toBeTruthy()
  expect(await ui.find({ text: '☐ ship the gate' })).toBeTruthy()
  expect(await ui.find({ text: '☐ call the plumber' })).toBeTruthy()
  expect(await ui.find({ text: '☐ water' })).toBeFalsy()
  // No button for the overview lens: it is run from the lens list like any other.
  expect(await ui.find({ key: 'run-overview' })).toBeFalsy()

  // + log writes the user's line under the heading, read fresh, and empties the field.
  written.set(today, note.replace('- [ ] call the plumber', '- [x] call the plumber'))
  await ui.input({ key: 'today-log-0', text: 'gate is fixed' })
  expect(written.get(today)).toBe(note.replace('- [ ] call the plumber', '- [x] call the plumber').replace('- 08:00 coffee\n', '- 08:00 coffee\n- 09:30 gate is fixed\n'))
  expect(toasts).toEqual(["Logged to today's note"])
  expect(await ui.find({ key: 'today-log-1' })).toBeTruthy()
  expect(await ui.find({ key: 'today-log-0' })).toBeFalsy()
  // The card read the note again: what was ticked in the editor is gone from it.
  expect(await ui.find({ text: '☐ call the plumber' })).toBeFalsy()
  await ui.input({ key: 'today-log-1', text: '   ' })
  expect(toasts.length).toBe(1)

  // A scratchpad entry is saved to the anchor's note, not to today's: + log is how a line gets here.
  await ui.input({ key: 'pad-input-0', text: 'ordered the hinges' })
  expect(await ui.find({ key: 'pad-today-0' })).toBeFalsy()
  await ui.press({ key: 'pad-drop-0' })

  // It goes into every draft until switched off: the box is under Setup with the others,
  // not in the You card.
  await pressDraft(ui)
  expect(got.filled[0]?.includes(`- Today: read "${today}"`)).toBe(true)
  expect(await ui.find({ key: 'pick-today' })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })
  expect((await ui.find({ key: 'pick-today' }))?.text).toMatch(/☑ today's note, when there is one/)
  await ui.press({ key: 'pick-today' })
  expect((await ui.find({ key: 'pick-today' }))?.text).toMatch(/☐/)
  await ui.press({ key: 'setup-menu' })
  await pressDraft(ui)
  expect(got.filled[1]?.includes('Today:')).toBe(false)

  // The You card folds with everything in it.
  await ui.press({ key: 'head-youOpen' })
  expect(await ui.find({ key: 'today-open' })).toBeFalsy()
  expect(await ui.find({ key: 'today-log-1' })).toBeFalsy()
  await ui.press({ key: 'head-youOpen' })
  expect(await ui.find({ key: 'today-open' })).toBeTruthy()
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: "Today's note: Inbox/2022-01-05.md" })).toBeTruthy()
  await ui.unmount()
})

test('a note for today that is not there is made only on a press: empty, never over one that exists, and the card follows the day', { options: { ...OPTIONS, dailyNote: 'Daily/{{date:YYYY-MM-DD}}' } }, async ($, on) => {
  const written = new Map<string, string>()
  const toasts: string[] = []
  const got = todayHooks(on, written, toasts)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  // Nothing is made by opening the pane, drafting, or typing on the scratchpad.
  expect(await ui.find({ text: 'No note for 2022-01-05 yet: it would be Daily/2022-01-05.md.' })).toBeTruthy()
  expect(await ui.find({ key: 'today-make' })).toBeTruthy()
  expect(await ui.find({ key: 'today-open' })).toBeFalsy()
  expect((await ui.findAll({ type: 'Input' })).some(one => /^today-log/.test(String(one.key ?? '')))).toBe(false)
  await pressDraft(ui)
  expect(got.filled[0]?.includes('Today:')).toBe(false)
  await $.prompt.submit({ text: 'hello' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect([...written.keys()].some(path => path.startsWith('/v/Daily/'))).toBe(false)

  // One press makes it, empty (no template is named), and the card has it at once.
  await ui.press({ key: 'today-make' })
  expect(written.get('/v/Daily/2022-01-05.md')).toBe('')
  expect(await ui.find({ key: 'today-make' })).toBeFalsy()
  expect(await ui.find({ key: 'today-open' })).toBeTruthy()
  expect(await ui.find({ key: 'today-log-0' })).toBeTruthy()
  expect(await ui.find({ text: "Nothing open in today's note." })).toBeTruthy()

  // The user writes in it in their editor: the next sync shows it.
  written.set('/v/Daily/2022-01-05.md', '- [ ] one thing\n')
  await $.prompt.submit({ text: 'hello' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(await ui.find({ text: '☐ one thing' })).toBeTruthy()
  written.set('/v/Daily/2022-01-05.md', '- [x] one thing\n- [ ] and another\n')
  await $.prompt.submit({ text: 'hello' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(await ui.find({ text: '☐ one thing' })).toBeFalsy()
  expect(await ui.find({ text: '☐ and another' })).toBeTruthy()

  // Past midnight the card is the new day's, whose note is not made, and yesterday's is untouched.
  got.clock.now = new Date(2022, 0, 6, 0, 5).getTime()
  await $.prompt.submit({ text: 'hello' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(await ui.find({ text: 'No note for 2022-01-06 yet: it would be Daily/2022-01-06.md.' })).toBeTruthy()
  expect(await ui.find({ key: 'today-make' })).toBeTruthy()
  expect(written.has('/v/Daily/2022-01-06.md')).toBe(false)
  expect(written.get('/v/Daily/2022-01-05.md')).toBe('- [x] one thing\n- [ ] and another\n')
  await ui.unmount()
})

test('make today\'s note fills the template Obsidian names, and a note made meanwhile in the editor is left as it is', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>([
    ['/v/.obsidian/daily-notes.json', '{"folder":"Daily","template":"Templates/Daily"}'],
    ['/v/Templates/Daily.md', '# {{title}}\n\n## Priorities\n- [ ] \n\n## Log\n<% keep this %>\n'],
  ])
  todayHooks(on, written, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'today-make' })
  expect(written.get('/v/Daily/2022-01-05.md')).toBe('# 2022-01-05\n\n## Priorities\n- [ ] \n\n## Log\n<% keep this %>\n')
  expect(await ui.find({ key: 'today-open' })).toBeTruthy()
  await ui.unmount()

  // Another day: the pane still shows the button, but the note was made in the editor since.
  written.delete('/v/Daily/2022-01-05.md')
  await $.command.run({ command: 'augi', args: '' })
  const again = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect(await again.find({ key: 'today-make' })).toBeTruthy()
  written.set('/v/Daily/2022-01-05.md', '- [ ] written in the editor\n')
  await again.press({ key: 'today-make' })
  expect(written.get('/v/Daily/2022-01-05.md')).toBe('- [ ] written in the editor\n')
  expect(await again.find({ text: '☐ written in the editor' })).toBeTruthy()
  await again.unmount()
})

test('with dailyNote none, Obsidian\'s daily notes are ignored and the You card says nothing about today', { options: { ...OPTIONS, dailyNote: 'none' } }, async ($, on) => {
  todayHooks(on, new Map([['/v/.obsidian/daily-notes.json', '{}'], ['/v/2022-01-05.md', '- [ ] x\n']]), [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'today-open' })).toBeFalsy()
  expect(await ui.find({ key: 'today-make' })).toBeFalsy()
  expect(await ui.find({ text: /No daily note set/ })).toBeFalsy()
  expect(await ui.find({ key: 'edit-context' })).toBeTruthy()
  await ui.unmount()
})

test('a session\'s anchor pair: null drops it, a chosen none is kept', async () => {
  // null drops a session's pair; an empty note stays, as "no anchor, by choice".
  expect(withAnchor([['a', 'X'], ['b', 'Y']], 'a', null)).toEqual([['b', 'Y']])
  expect(withAnchor([['a', 'X'], ['b', 'Y']], 'a', '')).toEqual([['b', 'Y'], ['a', '']])
})

test('a chat is titled by its prompt\'s first line; a slash command titles nothing', async () => {
  expect(promptTitle('fix the sprinkler valve\nit leaks at the joint')).toBe('fix the sprinkler valve')
  expect(promptTitle('\n\n  why is `this` slow · today  ')).toBe('why is this slow today')
  expect(promptTitle('see <pasted_content id="1">the log</pasted_content> please')).toBe('see the log please')
  expect(promptTitle('/augi usage')).toBe('')
  expect(promptTitle('   \n ')).toBe('')
  expect(promptTitle('x'.repeat(200)).length).toBe(70)
})

// A vault whose folder /r/garden last worked on PMOC - Garden (a store key older versions
// started every new chat on; it must not anchor anything now). `state` can be changed
// mid-test: the session id, its folder, and how long ago it began.
const inheritHooks = (on: On, folders: [string, string][] = [['/r/garden', 'PMOC - Garden']], anchors: [string, string][] = []) => {
  const state = { id: 'fresh', cwd: '/r/garden', age: 60_000, surfaces: ['terminal'] as ('terminal' | 'desktop')[] }
  const now = new Date(2026, 10, 8, 9, 30).getTime()
  const got = taskHooks(on, anchors, { '/v/PMOC - Garden.md': GARDEN_PLAIN, '/v/PMOC - Birdhouse.md': '### Outcome\n\nBirds move in.\n' }, state)
  got.store.set('folders', folders)
  on('session.cwd', () => ({ value: state.cwd }))
  on('session.surfaces', () => ({ value: state.surfaces }))
  on('session.usage', () => ({ value: { startedAt: now - state.age, context: { window: 200000 }, rateLimits: [] } as never }))
  // When each file was last written, by path; a test moves a transcript's time forward.
  const seen: Record<string, number> = {}
  on('fs.stat', (_, e) => ({ value: { kind: 'file', size: 1, mtimeMs: seen[e.path] ?? 1, isLink: false } }))

  return { ...got, state, seen }
}

test('a new chat starts with no anchor, whatever was picked last in its folder, and files nothing', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(anchorOf(got.store, 'fresh')).toBe(undefined)
  // A task run's prompt is filed nowhere, and nothing is told to its model.
  expect(await sentWith(got, () => $.prompt.submit({ text: 'Read your skill file first: do the task' }))).toEqual([])
  expect(got.written.size).toBe(0)
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await ui.unmount()
})

test('a fork starts on its parent\'s anchor, is filed there with its first prompt, and says where the anchor came from', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  got.said.push(toldBeside(anchorBrief(GARDEN_ANCHOR)))
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  // No picking: the pane is on the parent's anchor, and says where it came from.
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  expect(await ui.find({ text: /Carried over: from the chat this one was forked from/ })).toBeTruthy()
  expect(anchorOf(got.store, 'fresh')).toBe('PMOC - Garden')
  // Nothing is written to the anchor's log for a chat that has not said anything.
  expect(got.written.size).toBe(0)

  // A slash command is not the chat saying something.
  await $.prompt.submit({ text: '/augi usage' })
  expect(got.written.size).toBe(0)
  // Its first prompt files it under that prompt's first line, and the note about where it came from goes.
  await $.prompt.submit({ text: 'fix the sprinkler valve\nit leaks at the joint' })
  expect(parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').chats).toEqual([{ id: 'fresh', when: '2026-11-08 09:30', title: 'fix the sprinkler valve', done: false }])
  expect(await ui.find({ text: /Carried over: from the chat this one was forked from/ })).toBeFalsy()
  // Later prompts do not rename it, and do not read the log again to find that out.
  const log = got.written.get('/v/PMOC - Garden.md')
  got.written.set('/v/PMOC - Garden.md', `${log}\n<!-- an edit made elsewhere -->\n`)
  await $.prompt.submit({ text: 'and the timer too' })
  expect(got.written.get('/v/PMOC - Garden.md')).toBe(`${log}\n<!-- an edit made elsewhere -->\n`)
  await ui.unmount()
})

const GARDEN_ANCHOR = { note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md' }
const NOT_TOLD = { session: '', note: '', today: '', again: false }

test('anchorBrief names the note, its Agent Log section and today, says where writing goes, and asks for nothing', () => {
  const text = anchorBrief(GARDEN_ANCHOR, '/v/Daily/2022-03-04.md')
  expect(text.split('\n')[0]).toBe('Augi: this chat is anchored to [[PMOC - Garden]], the task or project it is working on.')
  expect(text).toContain('- My note: "/v/PMOC - Garden.md".')
  expect(text).toContain('- Its "# Agent Log" section, at the end of that note, is the pane\'s')
  expect(text).toContain('It is not a log for you to keep. Add a line there only when I ask you to save something')
  expect(text).not.toContain('Agent Log - PMOC - Garden')
  expect(text).toContain('a folded list of the chats on it with where each last stood')
  expect(text).toContain('- Today\'s note: "/v/Daily/2022-03-04.md".')
  expect(text).toMatch(/"the task note"/)
  expect(text).toMatch(/What I ask you to save goes under "# Agent Log" \(add the heading at the note's end if it is missing\)\. Write above that heading only when I ask/)
  expect(text.split('\n').at(-1)).toMatch(/not a request: write nothing there until I ask/)

  // No daily note: no line for one. A note not found is looked for by name.
  expect(anchorBrief(GARDEN_ANCHOR)).not.toMatch(/Today/)
  expect(anchorBrief({ ...GARDEN_ANCHOR, notePath: '' })).toContain('- My note: (find it in my notes by name).')
  // A change of anchor says which one it replaces.
  expect(anchorBrief(GARDEN_ANCHOR, '', 'PMOC - Birdhouse').split('\n')[0]).toBe('Augi: this chat is now anchored to [[PMOC - Garden]], no longer to [[PMOC - Birdhouse]].')
  expect(anchorBrief(GARDEN_ANCHOR, '', 'PMOC - Garden').split('\n')[0]).toMatch(/^Augi: this chat is anchored to/)
})

test('anchorNews tells a chat its anchor once, again when it changes or after a compaction, and one line for a new day', () => {
  // Never told, no anchor: nothing to say.
  expect(anchorNews(NOT_TOLD, 's1', null, '')).toBeNull()
  const first = anchorNews(NOT_TOLD, 's1', GARDEN_ANCHOR, '/v/Daily/2022-03-04.md')
  expect(first?.text).toBe(anchorBrief(GARDEN_ANCHOR, '/v/Daily/2022-03-04.md'))
  expect(first?.next).toEqual({ session: 's1', note: 'PMOC - Garden', today: '/v/Daily/2022-03-04.md', again: false })
  const knows = first?.next ?? NOT_TOLD
  // Told already: silence.
  expect(anchorNews(knows, 's1', GARDEN_ANCHOR, '/v/Daily/2022-03-04.md')).toBeNull()
  // The day's note moved: one line, not the whole brief. A day with no note yet says nothing.
  expect(anchorNews(knows, 's1', GARDEN_ANCHOR, '/v/Daily/2022-03-05.md')?.text).toBe('Augi: today\'s note is "/v/Daily/2022-03-05.md".')
  expect(anchorNews(knows, 's1', GARDEN_ANCHOR, '')).toBeNull()
  // Another anchor: the brief, naming the one it replaces.
  const other = { note: 'PMOC - Birdhouse', notePath: '/v/PMOC - Birdhouse.md' }
  expect(anchorNews(knows, 's1', other, '')?.text).toBe(anchorBrief(other, '', 'PMOC - Garden'))
  // No anchor any more: one line, then silence.
  const dropped = anchorNews(knows, 's1', null, '')
  expect(dropped?.text).toBe(unanchoredBrief('PMOC - Garden'))
  expect(anchorNews(dropped?.next ?? knows, 's1', null, '')).toBeNull()
  // A compaction may have summarized it away: the whole brief again, as if new.
  expect(anchorNews({ ...knows, again: true }, 's1', GARDEN_ANCHOR, '/v/Daily/2022-03-04.md')?.text).toBe(first?.text)
  // What another session was told (before a /clear) is not what this one knows.
  expect(anchorNews(knows, 's2', GARDEN_ANCHOR, '')?.text).toBe(anchorBrief(GARDEN_ANCHOR))
  expect(anchorNews(knows, 's2', null, '')).toBeNull()
})

// What entered beside the prompt `send` submits, for the model: empty when nothing did.
const sentWith = async (got: { beside: string[][] }, send: () => Promise<unknown>): Promise<string[]> => {
  const before = got.beside.length
  await send()

  return got.beside.length > before ? got.beside[got.beside.length - 1] ?? [] : []
}

test('an anchored chat is told its anchor beside its first prompt, unseen, and again only when the anchor changes', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  got.said.push(toldBeside(anchorBrief(GARDEN_ANCHOR)))
  on('session.compact', (_, e) => ({ messages: e.messages }))
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })

  // A slash command carries nothing, and does not use up the telling.
  expect(await sentWith(got, () => $.prompt.submit({ text: '/augi usage' }))).toEqual([])
  // The first prompt carries the brief: the note, its log, and where writing goes.
  const first = await sentWith(got, () => $.prompt.submit({ text: 'fix the sprinkler valve' }))
  expect(first).toHaveLength(1)
  expect(first[0]).toBe(anchorBrief({ note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md' }))
  // The log it names is there by then: the same prompt filed the chat in it.
  expect(got.written.has('/v/PMOC - Garden.md')).toBe(true)
  // The next prompt carries nothing: the chat knows.
  expect(await sentWith(got, () => $.prompt.submit({ text: 'and the timer too' }))).toEqual([])

  // Another anchor picked in the pane: the next prompt says so, once.
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' })
  const moved = await sentWith(got, () => $.prompt.submit({ text: 'now the birdhouse' }))
  expect(moved).toEqual([anchorBrief({ note: 'PMOC - Birdhouse', notePath: '/v/PMOC - Birdhouse.md' }, '', 'PMOC - Garden')])
  expect(await sentWith(got, () => $.prompt.submit({ text: 'paint it' }))).toEqual([])

  // A compaction may drop it from what the model reads: said again, in full.
  await $.session.compact({ trigger: 'manual', messages: [{ role: 'user', text: 'where the sprinkler leaks', toolUses: [], toolResults: [] }] } as never)
  expect((await sentWith(got, () => $.prompt.submit({ text: 'where were we' })))[0]).toMatch(/^Augi: this chat is anchored to \[\[PMOC - Birdhouse\]\]/)
  expect(await sentWith(got, () => $.prompt.submit({ text: 'right' }))).toEqual([])

  // No anchor, by choice: one line, then nothing.
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'anchor-clear' })
  expect(await sentWith(got, () => $.prompt.submit({ text: 'something else' }))).toEqual([unanchoredBrief('PMOC - Birdhouse')])
  expect(await sentWith(got, () => $.prompt.submit({ text: 'more of it' }))).toEqual([])
  await ui.unmount()
})

test('a chat with no anchor carries nothing, and a pasted draft that anchors it carries the brief with it', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [])
  await $.command.run({ command: 'augi', args: '' })
  // No anchor in this folder: nothing beside the prompt.
  expect(await sentWith(got, () => $.prompt.submit({ text: 'hello' }))).toEqual([])
  // A draft pasted from another chat anchors this one, and the same prompt carries the brief.
  const draft = buildDraft([], { aboutMe: false, openAugi: false, project: true, today: false }, null, 'prune the roses', { note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md' })
  expect((await sentWith(got, () => $.prompt.submit({ text: draft })))[0]).toMatch(/^Augi: this chat is anchored to \[\[PMOC - Garden\]\]/)
  expect(anchorOf(got.store, 'fresh')).toBe('PMOC - Garden')
})

test('tellChat: no keeps the anchor out of the chat', { options: { ...OPTIONS, tellChat: 'no' } }, async ($, on) => {
  const got = inheritHooks(on, [], [['fresh', 'PMOC - Garden']])
  await $.command.run({ command: 'augi', args: '' })
  expect(await sentWith(got, () => $.prompt.submit({ text: 'fix the sprinkler valve' }))).toEqual([])
})

test('keep confirms a carried-over anchor; change opens the menu; no anchor sticks for the session', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  got.said.push(toldBeside(anchorBrief(GARDEN_ANCHOR)))
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'inherit-keep' })
  expect(await ui.find({ text: /Carried over: from the chat this one was forked from/ })).toBeFalsy()
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  await ui.unmount()

  // Another fork of it: change opens the anchor list, and picking one is a choice.
  got.state.id = 'second'
  await $.command.run({ command: 'augi', args: '' })
  const two = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await two.find({ text: /Carried over: from the chat this one was forked from/ })).toBeTruthy()
  expect(await two.find({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' })).toBeFalsy()
  await two.press({ key: 'inherit-change' })
  await two.press({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' })
  expect(anchorOf(got.store, 'second')).toBe('PMOC - Birdhouse')
  expect(await two.find({ text: /Carried over: from the chat this one was forked from/ })).toBeFalsy()
  // The choice is this chat's alone: the store's old folder key is not touched.
  expect(got.store.get('folders')).toEqual([['/r/garden', 'PMOC - Garden']])
  await two.unmount()

  // A third fork: no anchor, by choice. A refresh does not bring one back.
  got.state.id = 'third'
  await $.command.run({ command: 'augi', args: '' })
  const three = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  await three.press({ key: 'inherit-none' })
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await three.press({ key: 'refresh' })
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await $.prompt.submit({ text: 'an unanchored question' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  // Nothing was filed anywhere for the chats that only looked.
  expect(got.written.has('/v/PMOC - Garden.md')).toBe(false)
  await three.unmount()
})

// A user message as the model is sent it, with what the pane told the model beside the prompt.
const toldBeside = (brief: string, prompt = 'go on'): { role: string; content: unknown } => ({
  role: 'user',
  content: [{ type: 'text', text: `<system-reminder>\nprompt.submit hook additional context: ${brief}\n</system-reminder>` }, { type: 'text', text: prompt }],
})

test('a conversation says what its chat is anchored to: the last brief decides, and a brief quoted in a prompt or a tool result is not one', () => {
  const garden = toldBeside(anchorBrief(GARDEN_ANCHOR, '/v/2022-01-05.md'))
  const moved = toldBeside(anchorBrief({ ...GARDEN_ANCHOR, note: 'PMOC - Birdhouse' }, '', 'PMOC - Garden'))
  const reply = { role: 'assistant', content: [{ type: 'text', text: 'done' }] }
  expect(anchorFromMessages([garden, reply])).toBe('PMOC - Garden')
  expect(anchorFromMessages([garden, reply, moved])).toBe('PMOC - Birdhouse')
  expect(anchorFromMessages([garden, toldBeside(unanchoredBrief('PMOC - Garden'))])).toBe('')
  // Today's note changing is not a word about the anchor.
  expect(anchorFromMessages([garden, toldBeside('Augi: today\'s note is "/v/2022-01-06.md".')])).toBe('PMOC - Garden')
  expect(anchorFromMessages([])).toBe(null)
  expect(anchorFromMessages([{ role: 'user', content: 'a plain prompt' }, reply])).toBe(null)
  const quoted = `prompt.submit hook additional context: ${anchorBrief(GARDEN_ANCHOR)}`
  expect(anchorFromMessages([{ role: 'user', content: [{ type: 'text', text: quoted }, { type: 'tool_result', content: `<system-reminder>\n${quoted}` }] }])).toBe(null)
  expect(anchorFromMessages([{ role: 'assistant', content: garden.content }])).toBe(null)
})

test('a forked chat starts on the anchor its parent was told, however old the parent is; a parent with none leaves it with none', { options: OPTIONS }, async ($, on) => {
  // The folder last worked on the garden; the chat this one was forked from was on the birdhouse.
  const got = inheritHooks(on)
  got.state.id = 'fork-one'
  got.state.age = 5 * 3_600_000
  got.said.push(toldBeside(anchorBrief({ ...GARDEN_ANCHOR, note: 'PMOC - Birdhouse', notePath: '/v/PMOC - Birdhouse.md' })))
  await $.command.run({ command: 'augi', args: '' })
  let ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Birdhouse/)
  expect(anchorOf(got.store, 'fork-one')).toBe('PMOC - Birdhouse')
  expect(await ui.find({ text: /Carried over: from the chat this one was forked from/ })).toBeTruthy()
  expect(await ui.find({ key: 'inherit-change' })).toBeTruthy()
  expect(got.written.size).toBe(0)
  await ui.unmount()

  // Another fork, of a parent last told it has no anchor: it starts with none, new as it is.
  got.state.id = 'fork-two'
  got.state.age = 5_000
  got.said.push(toldBeside(unanchoredBrief('PMOC - Birdhouse')))
  await $.command.run({ command: 'augi', args: '' })
  ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(anchorOf(got.store, 'fork-two')).toBe(undefined)
  await ui.unmount()
})

test('no anchor is carried over to an older chat, a chat in another folder, or a worktree; one that already chose keeps its own', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [['/r/garden', 'PMOC - Garden']], [['chose', 'PMOC - Birdhouse']])
  // A chat that began two hours ago and never had an anchor (a resume, a chat open across an update).
  got.state.id = 'old'
  got.state.age = 2 * 3_600_000
  await $.command.run({ command: 'augi', args: '' })
  let ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(anchorOf(got.store, 'old')).toBe(undefined)
  await ui.unmount()

  // A new chat in a folder nothing was anchored in.
  got.state.id = 'elsewhere'
  got.state.age = 5_000
  got.state.cwd = '/r/pond'
  await $.command.run({ command: 'augi', args: '' })
  ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await ui.unmount()

  // A chat with its own anchor keeps it.
  got.state.id = 'chose'
  got.state.cwd = '/r/garden'
  await $.command.run({ command: 'augi', args: '' })
  ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Birdhouse/)
  expect(await ui.find({ text: /Carried over: from the chat this one was forked from/ })).toBeFalsy()
  await ui.unmount()

  // A new chat in a worktree of the repository starts with none too.
  got.state.id = 'tree'
  got.state.cwd = '/r/garden/.claude/worktrees/fix-gate'
  await $.command.run({ command: 'augi', args: '' })
  ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await ui.unmount()
})

test('no anchor is carried over to a headless run, or onto a note that is gone; a session resumed in this process shows its own stored anchor', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [['/r/garden', 'PMOC - Garden']], [['earlier', 'PMOC - Birdhouse']])
  on('session.end', (_, e) => ({ sessionId: e.sessionId }) as never)
  // A forked run that draws nowhere: nobody could see, keep or change the anchor.
  got.said.push(toldBeside(anchorBrief(GARDEN_ANCHOR)))
  got.state.surfaces = []
  await $.command.run({ command: 'augi', args: '' })
  expect(anchorOf(got.store, 'fresh')).toBe(undefined)
  // A fork whose parent's anchor names a note that is on no board and in no folder now.
  got.state.surfaces = ['terminal']
  got.state.id = 'pond'
  got.said.push(toldBeside(anchorBrief({ note: 'Renamed since', notePath: '' })))
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(anchorOf(got.store, 'pond')).toBe(undefined)
  await $.prompt.submit({ text: 'a first prompt' })
  expect(got.written.size).toBe(0)

  // /resume in this process: the pane drops the old session's anchor, then shows the resumed one's.
  await $.session.end({ reason: 'resume', sessionId: 'pond' } as never)
  got.state.id = 'earlier'
  got.state.age = 5 * 3_600_000
  await $.prompt.submit({ text: '/augi' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Birdhouse/)
  expect(await ui.find({ text: /Carried over: from the chat this one was forked from/ })).toBeFalsy()
  await ui.unmount()
})

test('a draft copied for a new chat wins over an anchor the chat only started on by itself', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  got.said.push(toldBeside(anchorBrief(GARDEN_ANCHOR)))
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(anchorOf(got.store, 'fresh')).toBe('PMOC - Garden')
  const pasted = buildDraft([], ALL_ON, null, 'hang the birdhouse', { note: 'PMOC - Birdhouse', notePath: '/v/PMOC - Birdhouse.md' })
  await $.prompt.submit({ text: pasted })
  expect(anchorOf(got.store, 'fresh')).toBe('PMOC - Birdhouse')
  expect(parseTrail(got.written.get('/v/PMOC - Birdhouse.md') ?? '').chats.map(chat => chat.title)).toEqual(['hang the birdhouse'])
  // The anchor it started on was never written to.
  expect(got.written.has('/v/PMOC - Garden.md')).toBe(false)
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Birdhouse/)
  await ui.unmount()
})

test('after /clear the pane is the new session\'s: no anchor, and none on the next sync either', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [['/r/garden', 'PMOC - Garden']], [['fresh', 'PMOC - Garden']])
  on('session.end', (_, e) => ({ sessionId: e.sessionId }) as never)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  await $.session.end({ reason: 'clear', sessionId: 'fresh' } as never)
  got.state.id = 'cleared'
  got.state.age = 0
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await $.prompt.submit({ text: '/augi' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(anchorOf(got.store, 'cleared')).toBe(undefined)
  await ui.unmount()
})

test('a chat on an anchor it chose is titled by its first prompt too, replacing the placeholder once', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-PMOC - Garden' })
  const chats = () => parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').chats.map(chat => chat.title)
  expect(chats()).toEqual(['chat'])
  // Picking is this chat's choice alone: no folder remembers it for the next chat.
  expect(got.store.get('folders')).toEqual([])
  await $.prompt.submit({ text: 'plan the beds for spring' })
  expect(chats()).toEqual(['plan the beds for spring'])
  await $.prompt.submit({ text: 'something else entirely' })
  expect(chats()).toEqual(['plan the beds for spring'])
  await ui.unmount()
})

test('a pane draft pasted into a new chat anchors it and files it in the anchor\'s log under its instruction; the pane lists no chats', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [], [['parent', 'PMOC - Garden']])
  got.state.id = 'parent'
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  // One way to draft, and no form or button for a chat of its own.
  expect(await ui.find({ key: 'task-child' })).toBeFalsy()
  expect(await ui.find({ key: 'copy-draft' })).toBeFalsy()
  await pressDraft(ui)
  const draft = (got.filled[0] ?? '').replace(/Instruction: .*$/, 'Instruction: wire the sprinkler timer')
  expect(anchorFromDraft(draft)).toBe('PMOC - Garden')
  expect(draft.includes('"/v/PMOC - Garden.md"')).toBe(true)
  // Drafting writes nothing.
  expect(got.written.size).toBe(0)

  // The same text sent in another chat in the folder: that chat is on the anchor, in its log.
  got.state.id = 'child-1'
  await $.prompt.submit({ text: draft })
  expect(anchorOf(got.store, 'child-1')).toBe('PMOC - Garden')
  expect(parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').chats).toEqual([{ id: 'child-1', when: '2026-11-08 09:30', title: 'wire the sprinkler timer', done: false }])
  // Filing it wrote the anchor's note, in its Agent Log section, and no other.
  expect([...got.written.keys()].filter(path => path.endsWith('.md'))).toEqual(['/v/PMOC - Garden.md'])
  expect(got.written.get('/v/PMOC - Garden.md')?.startsWith(GARDEN_PLAIN)).toBe(true)

  // Back in the first chat the log has it; the pane keeps no list of chats.
  got.state.id = 'parent'
  await ui.press({ key: 'refresh' })
  expect(await ui.find({ key: 'chat-child-1' })).toBeFalsy()
  expect(await ui.find({ text: /Chats on PMOC - Garden/ })).toBeFalsy()
  await ui.unmount()
})

test('a draft names its anchor so a pasted copy can attach a new chat; anything else names none', async () => {
  expect(anchorFromDraft(buildDraft([], ALL_ON, null, 'go', ANCHOR))).toBe('PMOC - Garden')
  const odd = { note: 'Plan: a/b', notePath: '/v/Plan- a-b.md' }
  expect(anchorFromDraft(buildDraft([], ALL_ON, null, 'go', odd))).toBe('Plan: a/b')
  const lost = { ...ANCHOR, notePath: '' }
  const text = buildDraft([], ALL_ON, null, 'go', lost)
  expect(text.includes('- Project [[PMOC - Garden]]: (find it in the vault), my note')).toBe(true)
  expect(anchorFromDraft(text)).toBe('PMOC - Garden')
  expect(buildDraft([], ALL_ON, null, 'go', ANCHOR).includes('- Project [[PMOC - Garden]]: "/v/PMOC - Garden.md", my note')).toBe(true)
  expect(anchorFromDraft('hello\n- Project [[PMOC - Garden]]: x')).toBe(null)
  expect(anchorFromDraft(buildDraft([], ALL_ON, null, 'go'))).toBe(null)
  expect(anchorFromDraft(buildDraft([], { ...ALL_ON, project: false }, null, 'go', ANCHOR))).toBe(null)
  expect(anchorFromDraft(findInJiraDraft('/v/PMOC - Garden.md', 'PMOC - Garden', { project: '', server: '' }))).toBe(null)
})

// A vault with a question already in PMOC - Garden's Agent Log section; `session.id` is read each call.
const chatHooks = (on: On, session: { id: string }, anchors: [string, string][]) => {
  const question = '- [ ] 2026-11-08 09:00 · does the cache survive restarts?'
  const written = new Map<string, string>([['/v/PMOC - Garden.md', `${GARDEN_NOTE}\n# Agent Log\n\n${question}\n`]])
  const store = new Map<string, unknown>([['anchors', anchors]])
  const got = { written, store, question, filled: [] as string[], copied: [] as string[], toasts: [] as string[] }
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', (_, e) => {
    got.toasts.push(e.text)
    return { value: undefined }
  })
  on('ui.copy', (_, e) => {
    got.copied.push(e.text)
    return { value: { isCopied: true } }
  })
  on('session.id', () => ({ value: session.id }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/v/Kanban.md'].includes(e.path) || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path === '/v/Kanban.md' ? BOARD_TEXT : e.path === '/v/PMOC - Garden.md' ? GARDEN_NOTE : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => {
    const name = e.argv[0] === 'find' ? e.argv[e.argv.indexOf('-name') + 1] ?? '' : ''
    const found = name === 'PMOC - Garden.md' || name === 'PMOC - Birdhouse.md' ? `/v/${name}` : ''

    return { value: { exitCode: 0, stdout: found ? `${found}\n` : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('prompt.read', () => ({ value: { text: 'explain the cache', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    got.filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))

  return got
}

const anchorOf = (store: Map<string, unknown>, id: string): string | undefined => anchorPairs(store.get('anchors')).find(([one]) => one === id)?.[1]

test('a pane draft sent in a chat with no anchor anchors that chat; one with an anchor keeps it', { options: OPTIONS }, async ($, on) => {
  const session = { id: 'first' }
  const got = chatHooks(on, session, [['first', 'PMOC - Garden'], ['other', 'PMOC - Birdhouse']])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await pressDraft(ui)
  expect(got.filled.length).toBe(1)
  expect(anchorFromDraft(got.filled[0] ?? '')).toBe('PMOC - Garden')

  // A new chat with no anchor: the pasted draft anchors it, and the pane shows it.
  session.id = 'second'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await $.prompt.submit({ text: got.filled[0] ?? '' })
  expect(anchorOf(got.store, 'second')).toBe('PMOC - Garden')
  expect(parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').chats.map(chat => [chat.id, chat.title])).toEqual([['second', 'explain the cache']])
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)

  // A chat anchored elsewhere keeps its anchor; the chat goes in its own anchor's log.
  session.id = 'other'
  await $.prompt.submit({ text: got.filled[0] ?? '' })
  expect(anchorOf(got.store, 'other')).toBe('PMOC - Birdhouse')
  expect(parseTrail(got.written.get('/v/PMOC - Birdhouse.md') ?? '').chats.map(chat => chat.id)).toEqual(['other'])
  expect(parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').chats.map(chat => chat.id)).toEqual(['second'])
  await ui.unmount()
})

test('a vault note opens in the note editor, a web page in the browser, any other file in the file editor', async () => {
  const paths = resolvePaths({ notesFolder: '/v', editor: 'obsidian' }, '/Users/t')
  expect(paths.fileEditor).toBe('vscode')
  expect(openArgv(paths, '/v/A.md')).toEqual(['open', 'obsidian://open?path=%2Fv%2FA.md'])
  expect(openArgv(paths, '/v/pic.png')).toEqual(['code', '/v/pic.png'])
  expect(openArgv(paths, '/r/notes.md')).toEqual(['code', '/r/notes.md'])
  expect(openArgv({ ...paths, fileEditor: 'cursor' }, '/r/a.ts')).toEqual(['cursor', '/r/a.ts'])
  expect(openArgv({ ...paths, fileEditor: 'system' }, '/r/a.ts')).toEqual(['open', '/r/a.ts'])
  // A web page is for looking at: the system opens it, which is the browser, whatever the file editor is.
  expect(openArgv(paths, '/r/report.html')).toEqual(['open', '/r/report.html'])
  expect(openArgv({ ...paths, fileEditor: 'cursor' }, '/v/exports/Page.HTM')).toEqual(['open', '/v/exports/Page.HTM'])
  expect(openArgv(paths, '/r/html.ts')).toEqual(['code', '/r/html.ts'])
})

// padHooks without the events a test wants to answer itself.
const except = (on: On, names: string[]): On => ((name: string, ...rest: unknown[]) => (names.includes(name) ? undefined : (on as unknown as (...args: unknown[]) => unknown)(name, ...rest))) as unknown as On

test('another chat writes the anchor\'s note: this pane reads it again when the file changed, and not otherwise', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  const store = new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]]])
  padHooks(except(on, ['fs.read']), store, written, [])
  let mtime = 1
  let reads = 0
  on('fs.stat', () => ({ value: { kind: 'file', size: 1, mtimeMs: mtime, isLink: false } }))
  on('prompt.submit', (_, e) => ({ text: e.text }))
  on('fs.read', (_, e) => {
    reads += e.path === '/v/PMOC - Garden.md' ? 1 : 0
    return { value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }
  })

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  const log = '/v/PMOC - Garden.md'
  // Another chat on the same anchor pins something in the note's Agent Log section.
  written.set(log, addPin('### Outcome\n\nIt grows.\n', { on: true, kind: 'url', target: 'https://example.com/from-the-other-chat', label: 'from the other chat' }))
  mtime = 2
  await $.prompt.submit({ text: 'hello' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect((await ui.find({ key: 'pin-open-url-https://example.com/from-the-other-chat' }))?.text).toMatch(/from the other chat/)
  // Nothing changed since: the log is not read again.
  const before = reads
  await $.prompt.submit({ text: 'hello again' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(reads).toBe(before)
  await ui.unmount()
})

test('a new interactive chat opens the pane; openOnStart no, or a headless start, leaves it closed', { options: OPTIONS }, async ($, on) => {
  let opened = 0
  padHooks(except(on, ['ui.open']), new Map(), new Map(), [])
  on('ui.open', () => {
    opened += 1
    return { value: { isPlaced: true } }
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  await $.session.start({ cwd: '/r', surface: null, isInteractive: false })
  expect(opened).toBe(0)
  await $.session.start({ cwd: '/r', surface: 'terminal', isInteractive: true })
  expect(opened).toBe(1)
})

test('with openOnStart no, a new chat does not open the pane', { options: { ...OPTIONS, openOnStart: 'no' } }, async ($, on) => {
  let opened = 0
  padHooks(except(on, ['ui.open']), new Map(), new Map(), [])
  on('ui.open', () => {
    opened += 1
    return { value: { isPlaced: true } }
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.attach', (_, e) => ({ clientId: e.clientId }))
  await $.session.start({ cwd: '/r', surface: 'terminal', isInteractive: true })
  expect(opened).toBe(0)
  await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })
  expect(opened).toBe(0)
})

// A desktop chat starts through the SDK (no surface, not interactive) and its window attaches after.
test('a desktop chat opens the pane when its window attaches, once a session: a reconnect or a second start does not reopen it', { options: OPTIONS }, async ($, on) => {
  let opened = 0
  const session = { id: 'desk-1' }
  padHooks(except(on, ['ui.open', 'session.id']), new Map(), new Map(), [])
  on('session.id', () => ({ value: session.id }))
  on('ui.open', () => {
    opened += 1
    return { value: { isPlaced: true } }
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.attach', (_, e) => ({ clientId: e.clientId }))
  await $.session.start({ cwd: '/r', surface: null, isInteractive: false })
  expect(opened).toBe(0)
  // A phone joining opens nothing.
  await $.session.attach({ surface: 'mobile', clientId: 'mobile:default' })
  expect(opened).toBe(0)
  const joined = await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })
  expect(joined.clientId).toBe('desktop:default')
  expect(opened).toBe(1)
  // The pane drew from loaded state: it is the real pane, not an empty one.
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect(await ui.find({ key: 'anchor-menu' })).toBeTruthy()
  await ui.unmount()
  // The window reconnects, and the module reloads (a settings change runs session.start again).
  await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })
  await $.session.start({ cwd: '/r', surface: null, isInteractive: false })
  await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })
  expect(opened).toBe(1)
  // The editor's window counts too, in a session of its own.
  session.id = 'code-1'
  await $.session.attach({ surface: 'vscode', clientId: 'vscode:default' })
  expect(opened).toBe(2)
})

test('a terminal chat that opened its pane at start does not open it again when a window attaches', { options: OPTIONS }, async ($, on) => {
  let opened = 0
  padHooks(except(on, ['ui.open']), new Map(), new Map(), [])
  on('ui.open', () => {
    opened += 1
    return { value: { isPlaced: true } }
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.attach', (_, e) => ({ clientId: e.clientId }))
  await $.session.start({ cwd: '/r', surface: 'terminal', isInteractive: true })
  await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })
  expect(opened).toBe(1)
})

test('a failed open at attach never fails the attach, and is tried again when the window attaches next', { options: OPTIONS }, async ($, on) => {
  let tries = 0
  padHooks(except(on, ['ui.open']), new Map(), new Map(), [])
  on('ui.open', () => {
    tries += 1
    if (tries === 1) {
      throw new Error('no surface')
    }
    return { value: { isPlaced: true } }
  })
  on('session.attach', (_, e) => ({ clientId: e.clientId }))
  expect((await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })).clientId).toBe('desktop:default')
  expect(tries).toBe(1)
  await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })
  expect(tries).toBe(2)
  // It opened: a third attach leaves it alone.
  await $.session.attach({ surface: 'desktop', clientId: 'desktop:default' })
  expect(tries).toBe(2)
})

test('the scratchpad keeps what is typed and not yet kept: across a redraw, a remount and a reload; a submit clears it', { options: OPTIONS }, async ($, on) => {
  padHooks(on, new Map(), new Map(), [])
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  await $.command.run({ command: 'augi', args: '' })
  let ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  const typed = async (): Promise<string> => String((await ui.find({ key: 'pad-input-0' }))?.props?.value ?? '')
  expect(await typed()).toBe('')
  await ui.input({ key: 'pad-input-0', text: 'half a thou', kind: 'change' })
  await ui.input({ key: 'pad-input-0', text: 'half a thought\nsecond line', kind: 'change' })
  // Nothing was kept as an entry: it is still only typed.
  expect(await ui.find({ key: 'pad-row-0' })).toBeFalsy()
  // Another press redraws the pane: the field still holds the text.
  await ui.press({ key: 'head-padOpen' })
  await ui.press({ key: 'head-padOpen' })
  expect(await typed()).toBe('half a thought\nsecond line')
  // The pane is closed and drawn again (a click away and back).
  await ui.unmount()
  ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect(await typed()).toBe('half a thought\nsecond line')
  // A reload of the module runs session.start again.
  await $.session.start({ cwd: '/r', surface: null, isInteractive: false })
  expect(await typed()).toBe('half a thought\nsecond line')
  // Enter keeps it as an entry and empties the field.
  await ui.input({ key: 'pad-input-0', text: 'half a thought\nsecond line' })
  expect((await ui.find({ key: 'pad-row-0' }))?.text).toMatch(/half a thought/)
  expect(String((await ui.find({ key: 'pad-input-1' }))?.props?.value ?? '')).toBe('')
  await ui.unmount()
})

test('a control is logged by its name alone: a note, a path or an id in its key is dropped', () => {
  expect(usageName('draft')).toBe('draft')
  expect(usageName('card-PMOC - Garden')).toBe('card')
  expect(usageName('card-sub-PMOC - Garden-Fix the gate')).toBe('card-sub')
  expect(usageName('pin-open-file-/r/docs/ARCH.md')).toBe('pin-open')
  expect(usageName('task-Plant the beds')).toBe('task')
  expect(usageName('task-new')).toBe('task-new')
  expect(usageName('pad-save-all')).toBe('pad-save-all')
  expect(usageName('pad-save-0')).toBe('pad-save')
  expect(usageName('edit-echoes')).toBe('edit')
  expect(usageName('edit-context')).toBe('edit-context')
  // A button added later is logged as written, unless its key looks like a name or a path.
  expect(usageName('new-button')).toBe('new-button')
  expect(usageName('Sprinkler Benchmarks')).toBe('other')
  expect(usageName('/r/docs/ARCH.md')).toBe('other')
})

test('the usage report counts the last 30 days, names the lenses run, and lists what was never pressed', () => {
  const now = Date.parse('2022-03-31T12:00:00Z')
  const log = [
    JSON.stringify({ at: '2022-01-02T09:00:00.000Z', kind: 'press', what: 'refresh', anchored: false, surface: 'desktop' }),
    JSON.stringify({ at: '2022-03-30T09:00:00.000Z', kind: 'press', what: 'draft', anchored: true, surface: 'desktop' }),
    JSON.stringify({ at: '2022-03-30T09:01:00.000Z', kind: 'press', what: 'draft', anchored: true, surface: 'desktop' }),
    JSON.stringify({ at: '2022-03-30T09:02:00.000Z', kind: 'press', what: 'lens', lens: 'echoes', anchored: true, surface: 'terminal' }),
    'not a line', '', '{"at": 5}',
  ].join('\n')
  const events = parseUsage(log)
  expect(events.length).toBe(4)
  const report = usageReport(events, now)
  expect(report).toContain('last 30 days: 3 events (4 in the log, since 2022-01-02)')
  expect(report).toMatch(/Used:\n +2 {2}draft\n +1 {2}lens\n/)
  expect(report).toMatch(/Saved prompts run:\n +1 {2}echoes/)
  const never = report.split('Never used')[1] ?? ''
  expect(never).toContain('made-anchor')
  // Pressed once, long ago: not in the last 30 days, and not never used either.
  expect(never).not.toMatch(/\brefresh\b/)
  expect(never).not.toMatch(/ draft,/)
  expect(usageReport([], now)).toContain('nothing yet')
})

test('with the usage log on, a press, a submit and a sent draft each append one line that names the control and nothing of the user\'s', { options: { ...OPTIONS, trackUsage: 'yes' } }, async ($, on) => {
  const written = new Map<string, string>()
  const lines: string[] = []
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.id', () => ({ value: 'own-session' }))
  on('clock.now', () => ({ value: Date.parse('2022-03-30T09:00:00Z') }))
  const store = new Map<string, unknown>()
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('ui.toast', () => ({ value: undefined }))
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md'].includes(e.path) || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => {
    if (e.argv[0] === 'sh' && e.argv[e.argv.length - 1] === '/Users/t/.openaugi/usage/augi-ui.jsonl') {
      lines.push(e.argv[e.argv.length - 2] ?? '')
    }
    return { value: { exitCode: 0, stdout: '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  const filled: string[] = []
  on('prompt.read', () => ({ value: { text: 'explain the cache', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-PMOC - Garden' })
  // Typing is not a use: only the submit is counted.
  await ui.input({ key: 'pad-input-0', text: 'a secret half', kind: 'change' })
  await ui.input({ key: 'pad-input-0', text: 'does the secret cache survive?', kind: 'change' })
  await ui.input({ key: 'pad-input-0', text: 'does the secret cache survive?' })
  await pressDraft(ui)
  await $.prompt.submit({ text: filled[0] ?? '' })

  const events = parseUsage(lines.join('\n'))
  expect(events.map(one => `${one.kind} ${one.what}`)).toEqual(['press anchor-menu', 'press card', 'input pad-input', 'press lens-menu', 'press draft', 'sent draft-sent'])
  expect(events.map(one => one.anchored)).toEqual([false, true, true, true, true, true])
  expect(events[0]?.surface).toBe('desktop')
  expect(events[0]?.at).toBe('2022-03-30T09:00:00.000Z')
  expect(lines.join('\n')).not.toMatch(/Garden|secret|cache/)

  // /augi usage prints the report from the log file.
  written.set('/Users/t/.openaugi/usage/augi-ui.jsonl', lines.join('\n'))
  const said = await $.command.run({ command: 'augi', args: 'usage' })
  expect(JSON.stringify(said)).toContain('6 events')
  await ui.unmount()
})

// taskHooks with a clock a test can move, the settings rows the State card writes, and a
// model that answers the side question; `asked` is every question it was put.
const stateHooks = (on: On, anchors: [string, string][], skip: string[] = [], session = { id: 'own-session' }) => {
  const clock = { now: new Date(2026, 10, 8, 9, 30).getTime() }
  const got = taskHooks(except(on, ['clock.now', 'fs.list', ...skip]), anchors, { '/v/PMOC - Garden.md': GARDEN_PLAIN, '/v/lenses/echoes.md': '---\nname: echoes\n---\n# echoes\n\nSay what this echoes, in one line.\n' }, session)
  const asked: string[] = []
  const set: [string, unknown][] = []
  const model = { reply: { isAnswered: true, text: 'Doing: fixing the valve\nLast: nothing yet\nNext: find the leak\nWaiting on me: no' } as unknown }
  on('clock.now', () => ({ value: clock.now }))
  on('fs.list', (_, e) => ({ value: e.path === '/v/lenses' ? [{ name: 'echoes.md', kind: 'file', size: 1, mtimeMs: 0, isLink: false }] : [] }))
  on('config.list', () => ({ value: ['stateAuto', 'stateLens'].map(field => ({ key: `augi-ui.${field}`, label: field, kind: 'text', value: '', provider: { plugin: 'augi-ui', tier: 'user' }, isLocked: false })) as never }))
  on('config.set', (_, e) => {
    set.push([e.key, e.value])
    return { value: e.value }
  })
  on('model.fork', (_, e) => {
    asked.push(e.prompt)
    return { value: model.reply as never }
  })
  on('turn.complete', (_, e) => ({ text: e.answer }))

  return { ...got, clock, asked, set, model }
}
const turnEnds = { answer: 'done', durationMs: 1, isAborted: false, turnId: 't1', reason: 'answer' } as const
const settle = () => new Promise(resolve => setTimeout(resolve, 20))

test('update asks this chat\'s model where the chat stands: shown in the pane, kept under this chat\'s entry in the anchor\'s note, and nothing is asked unpressed', { options: OPTIONS }, async ($, on) => {
  const session = { id: 'own-session' }
  const got = stateHooks(on, [['own-session', 'PMOC - Garden']], [], session)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ text: /Not asked yet\. Update asks this chat's model what the last few turns did and what is next, and keeps the answer with this chat's entry in the anchor's note\./ })).toBeTruthy()
  expect((await ui.find({ key: 'state-auto' }))?.text).toMatch(/☐ auto after a turn/)
  // Auto is off: a turn ending asks nothing.
  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked).toEqual([])

  await ui.press({ key: 'state-update' })
  // The built-in question, since the lens folder has no chat-state lens.
  expect(got.asked).toEqual([stateAsk(STATE_PROMPT)])
  expect(await ui.find({ text: 'Doing: fixing the valve' })).toBeTruthy()
  expect(await ui.find({ text: 'Waiting on me: no' })).toBeTruthy()
  expect(await ui.find({ text: '📍 State · 11-08 09:30' })).toBeTruthy()
  // In the note it sits under this chat's own entry in the folded chat list; the chat is filed for it.
  expect(got.written.get('/v/PMOC - Garden.md')).toBe(`${GARDEN_PLAIN}\n# Agent Log\n\n> [!abstract]- Chats\n> - 2026-11-08 09:30 · chat · \`own-session\`\n>   - as of 2026-11-08 09:30\n>   - Doing: fixing the valve\n>   - Last: nothing yet\n>   - Next: find the leak\n>   - Waiting on me: no\n`)

  // Asked again later, this chat's state is replaced, not added to.
  got.clock.now += 60 * 60_000
  got.model.reply = { isAnswered: true, text: 'Doing: fixing the valve\nNext: test under pressure' }
  await ui.press({ key: 'state-update' })
  const note = got.written.get('/v/PMOC - Garden.md') ?? ''
  expect(note.split('as of').length).toBe(2)
  expect(note.includes('>   - as of 2026-11-08 10:30\n>   - Doing: fixing the valve\n>   - Next: test under pressure\n')).toBe(true)
  expect(await ui.find({ text: 'Next: find the leak' })).toBeFalsy()

  // No answer changes nothing in the note and says why.
  got.model.reply = { isAnswered: false, reason: 'nothing-to-fork' }
  await ui.press({ key: 'state-update' })
  expect(got.written.get('/v/PMOC - Garden.md')).toBe(note)
  expect(await ui.find({ text: 'Nothing to go on yet: send a prompt first.' })).toBeTruthy()
  expect(got.toasts.at(-1)).toBe('Nothing to go on yet: send a prompt first.')
  expect(await ui.find({ text: 'Next: test under pressure' })).toBeTruthy()

  await ui.unmount()

  // Another chat on the same task has its own state: the card shows none of the first one's,
  // and its answer goes under its own entry, leaving the first chat's where it was.
  got.store.set('anchors', [['own-session', 'PMOC - Garden'], ['second-chat', 'PMOC - Garden']])
  session.id = 'second-chat'
  got.model.reply = { isAnswered: true, text: 'Doing: painting the gate\nNext: second coat' }
  const other = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await other.press({ key: 'refresh' })
  expect(await other.find({ text: 'Next: test under pressure' })).toBeFalsy()
  expect(await other.find({ text: /Not asked yet/ })).toBeTruthy()
  await other.press({ key: 'state-update' })
  expect(await other.find({ text: 'Doing: painting the gate' })).toBeTruthy()
  expect(parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').states).toEqual({
    'own-session': { when: '2026-11-08 10:30', text: 'Doing: fixing the valve\nNext: test under pressure' },
    'second-chat': { when: '2026-11-08 10:30', text: 'Doing: painting the gate\nNext: second coat' },
  })
  // Back in the first chat, a pane that has not asked in this process shows what the note holds for it.
  session.id = 'own-session'
  await other.press({ key: 'refresh' })
  expect(await other.find({ text: 'Next: test under pressure' })).toBeTruthy()
  expect(await other.find({ text: 'Doing: painting the gate' })).toBeFalsy()
  await other.unmount()
})

test('with no anchor the state stays in the pane; the prompt is a saved prompt that can be picked or edited', { options: OPTIONS }, async ($, on) => {
  const got = stateHooks(on, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'state-update' })
  expect(await ui.find({ text: 'Doing: fixing the valve' })).toBeTruthy()
  expect(got.written.size).toBe(0)

  // The saved prompts are listed to pick from; picking one writes the setting.
  expect(await ui.find({ key: 'state-pick-echoes' })).toBeFalsy()
  await ui.press({ key: 'state-prompt' })
  expect((await ui.find({ key: 'state-pick-echoes' }))?.text).toMatch(/○ echoes/)
  await ui.press({ key: 'state-pick-echoes' })
  expect(got.set).toEqual([['augi-ui.stateLens', 'echoes']])
  // The auto box writes its setting too.
  await ui.press({ key: 'state-auto' })
  expect(got.set.at(-1)).toEqual(['augi-ui.stateAuto', 'yes'])

  // Edit prompt writes the chat-state lens from the built-in question, once, and opens it.
  await ui.press({ key: 'state-edit' })
  expect(got.written.get('/v/lenses/chat-state.md')).toBe(stateLensFile('chat-state'))
  expect(got.ran.some(argv => argv.join(' ').includes('chat-state.md'))).toBe(true)
  // From then on the file's words are what is asked.
  got.written.set('/v/lenses/chat-state.md', '---\nname: chat-state\n---\n# chat-state\n\nOne line: what is this chat for?\n')
  await ui.press({ key: 'state-update' })
  expect(got.asked.at(-1)).toBe(stateAsk('One line: what is this chat for?'))
  await ui.unmount()
})

test('with stateLens set, that saved prompt is the question', { options: { ...OPTIONS, stateLens: 'echoes' } }, async ($, on) => {
  const got = stateHooks(on, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'state-prompt' }))?.text).toMatch(/prompt: echoes/)
  await ui.press({ key: 'state-update' })
  expect(got.asked).toEqual([stateAsk('Say what this echoes, in one line.')])
  await ui.press({ key: 'state-prompt' })
  expect((await ui.find({ key: 'state-pick-echoes' }))?.text).toMatch(/● echoes/)
  await ui.unmount()
})

test('auto asks after a main-thread turn that answered, at most every ten minutes, and never for a subagent or an interrupted turn', { options: { ...OPTIONS, stateAuto: 'yes' } }, async ($, on) => {
  const got = stateHooks(on, [['own-session', 'PMOC - Garden']])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'state-auto' }))?.text).toMatch(/☑ auto after a turn/)
  await $.turn.complete({ ...turnEnds, agentId: 'sub-1' })
  await $.turn.complete({ ...turnEnds, isAborted: true, reason: 'aborted' })
  await settle()
  expect(got.asked.length).toBe(0)

  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked.length).toBe(1)
  expect(parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').states['own-session']?.text.startsWith('Doing: fixing the valve')).toBe(true)
  // Another turn five minutes on asks nothing; one past ten minutes does.
  got.clock.now += 5 * 60_000
  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked.length).toBe(1)
  got.clock.now += STATE_EVERY_MS
  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked.length).toBe(2)
  // An ask that got no answer is paced the same, not repeated every turn.
  got.model.reply = { isAnswered: false, reason: 'api-error', status: 529, error: 'overloaded' }
  got.clock.now += STATE_EVERY_MS
  await $.turn.complete(turnEnds)
  await settle()
  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked.length).toBe(3)
  await ui.unmount()
})

const BOARD_NOTES = ['2022-01-04 - Sketch', 'PMOC - Garden', 'PMOC - Birdhouse']

test('a task offer is a name from the board or a title for a new task, never anything else', () => {
  expect(offerCandidates(kanbanCards(BOARD_TEXT))).toEqual(BOARD_NOTES)
  expect(parseOffer('Task: pmoc - garden\nNew:\nWhy: it is about the beds', BOARD_NOTES)).toEqual({ note: 'PMOC - Garden', title: '', why: 'it is about the beds' })
  expect(parseOffer('Task: [[PMOC - Birdhouse]]\nNew: ignored when a task is named\nWhy: x', BOARD_NOTES)).toEqual({ note: 'PMOC - Birdhouse', title: '', why: 'x' })
  // A name that is not on the board is not a match: the title, if any, is what is offered.
  expect(parseOffer('Task: PMOC - Pond\nNew: "Dig the pond."\nWhy: nothing fits', BOARD_NOTES)).toEqual({ note: '', title: 'Dig the pond', why: 'nothing fits' })
  expect(parseOffer('Task: NONE\nNew: Fix: the gate/latch\nWhy: y', BOARD_NOTES).title).toBe('Fix the gate latch')
  expect(parseOffer('Task: NONE\nNew: NONE\nWhy: a quick question', BOARD_NOTES)).toEqual({ note: '', title: '', why: 'a quick question' })
  expect(parseOffer('I could not tell.', BOARD_NOTES)).toEqual({ note: '', title: '', why: '' })

  const ask = offerAsk(BOARD_NOTES, 'The git branch is "x".')
  expect(ask).toContain('Task: one name from the list below, copied exactly, or NONE')
  expect(ask.endsWith('The git branch is "x".\n\nMy tasks:\n- 2022-01-04 - Sketch\n- PMOC - Garden\n- PMOC - Birdhouse')).toBe(true)
  expect(offerAsk([]).endsWith('My tasks:\n(none yet)')).toBe(true)

  expect(jiraKeyIn('feature/proj-123-fix-the-gate')).toBe('PROJ-123')
  expect(jiraKeyIn('PROJ-9')).toBe('PROJ-9')
  expect(jiraKeyIn('main')).toBe('')
  expect(jiraKeyIn('v2-fixes')).toBe('')
})

test('a chat with no anchor can be offered its task: the model picks from the board, one press links it, and nothing is written before that', { options: OPTIONS }, async ($, on) => {
  const got = stateHooks(on, [])
  got.model.reply = { isAnswered: true, text: 'Task: PMOC - Garden\nNew:\nWhy: the chat is fixing the sprinkler valve' }
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'offer-link' })).toBeFalsy()
  await ui.press({ key: 'offer-ask' })
  expect(got.asked).toEqual([offerAsk(BOARD_NOTES)])
  expect(await ui.find({ text: '💡 This chat looks like PMOC - Garden.' })).toBeTruthy()
  expect(await ui.find({ text: 'the chat is fixing the sprinkler valve' })).toBeTruthy()
  expect(await ui.find({ key: 'offer-ask' })).toBeFalsy()
  // An offer is only an offer.
  expect(got.written.size).toBe(0)
  expect(anchorOf(got.store, 'own-session')).toBe(undefined)

  await ui.press({ key: 'offer-link' })
  expect(anchorOf(got.store, 'own-session')).toBe('PMOC - Garden')
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  expect(parseTrail(got.written.get('/v/PMOC - Garden.md') ?? '').chats.map(chat => chat.id)).toEqual(['own-session'])
  expect(await ui.find({ key: 'offer-link' })).toBeFalsy()
  expect(await ui.find({ key: 'offer-ask' })).toBeFalsy()
  await ui.unmount()
})

test('when nothing on the board fits, the offer is a new task: one press makes it and moves the chat onto it; not a task turns it down', { options: OPTIONS }, async ($, on) => {
  const got = stateHooks(on, [])
  got.model.reply = { isAnswered: true, text: 'Task: NONE\nNew: Dig the pond\nWhy: nothing on the list is about a pond' }
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'offer-ask' })
  expect(await ui.find({ text: '💡 No task on the board fits. New task: “Dig the pond”' })).toBeTruthy()
  // Turned down: the offer goes, nothing was made, and it can be asked for again.
  await ui.press({ key: 'offer-dismiss' })
  expect(await ui.find({ key: 'offer-make' })).toBeFalsy()
  expect(got.written.size).toBe(0)
  await ui.press({ key: 'offer-ask' })
  await ui.press({ key: 'offer-make' })
  expect(got.written.has('/v/Dig the pond.md')).toBe(true)
  expect(got.written.get('/v/Kanban.md')?.includes('- [ ] [[Dig the pond]]')).toBe(true)
  expect(anchorOf(got.store, 'own-session')).toBe('Dig the pond')

  // No suggestion says so and offers nothing.
  got.store.set('anchors', [])
  got.model.reply = { isAnswered: true, text: 'Task: NONE\nNew: NONE\nWhy: just a question' }
  await ui.press({ key: 'refresh' })
  await ui.press({ key: 'offer-ask' })
  expect(got.toasts.at(-1)).toBe('No task suggested for this chat.')
  expect(await ui.find({ key: 'offer-make' })).toBeFalsy()
  await ui.unmount()
})

test('a Jira key in the git branch that a board note carries settles the offer with no model call; one no note carries goes with the new task when it is in the Jira project set', { options: { ...OPTIONS, jiraProject: 'proj' } }, async ($, on) => {
  const got = stateHooks(on, [], ['process.run'])
  const repo = { branch: 'feature/proj-7-sprinkler', carries: '/v/PMOC - Garden.md\n' }
  on('session.cwd', () => ({ value: '/r/garden' }))
  on('process.run', (_, e) => {
    got.ran.push(e.argv)
    const name = e.argv[0] === 'find' ? e.argv[e.argv.indexOf('-name') + 1] ?? '' : ''
    const found = name ? ['/v/PMOC - Garden.md', ...got.written.keys()].find(path => path.endsWith(`/${name}`)) ?? '' : ''
    const stdout = e.argv[0] === 'git' ? `${repo.branch}\n` : e.argv[0] === 'grep' ? repo.carries : found ? `${found}\n` : ''

    return { value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'offer-ask' })
  expect(got.asked).toEqual([])
  expect(await ui.find({ text: '💡 This chat looks like PMOC - Garden.' })).toBeTruthy()
  expect(await ui.find({ text: 'The branch feature/proj-7-sprinkler names PROJ-7, and this note carries it.' })).toBeTruthy()
  expect(got.ran.some(argv => argv[0] === 'grep' && argv.includes('^(jira|Jira): *PROJ-7([^0-9]|$)'))).toBe(true)
  await ui.press({ key: 'offer-dismiss' })

  // No note carries the key: the model is told the branch, and a new task takes the key.
  repo.carries = ''
  got.model.reply = { isAnswered: true, text: 'Task: NONE\nNew: Replace the sprinkler valve\nWhy: new work' }
  await ui.press({ key: 'offer-ask' })
  expect(got.asked.at(-1)).toContain('The git branch is "feature/proj-7-sprinkler", which names the Jira issue PROJ-7; no task of mine carries that key yet.')
  expect(await ui.find({ text: '💡 No task on the board fits. New task: “Replace the sprinkler valve” (PROJ-7)' })).toBeTruthy()
  await ui.press({ key: 'offer-make' })
  expect(jiraFrom(got.written.get('/v/Replace the sprinkler valve.md') ?? '')).toBe('PROJ-7')
  await ui.unmount()
})

test('with auto on, a chat with no anchor is offered a task once after a turn; one that chose no anchor is not', { options: { ...OPTIONS, stateAuto: 'yes' } }, async ($, on) => {
  const got = stateHooks(on, [['own-session', '']])
  got.model.reply = { isAnswered: true, text: 'Task: PMOC - Garden\nNew:\nWhy: valve work' }
  const offers = () => got.asked.filter(one => one.includes('Which of my tasks'))
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  // This chat said it wants no anchor: a turn ending offers nothing.
  await $.turn.complete(turnEnds)
  await settle()
  expect(offers()).toEqual([])

  // A chat that never chose is offered one after its turn, and only once.
  got.store.set('anchors', [])
  await $.turn.complete(turnEnds)
  await settle()
  expect(offers()).toEqual([offerAsk(BOARD_NOTES)])
  expect(await ui.find({ key: 'offer-link' })).toBeTruthy()
  got.clock.now += 2 * STATE_EVERY_MS
  await $.turn.complete(turnEnds)
  await settle()
  expect(offers().length).toBe(1)
  await ui.unmount()
})

// ── Recall ──

test('a recall question and its claims are read from the model\'s answer, list marks and bold allowed; no question or no claim is none', async () => {
  expect(parseRecall('Question: Why does the pane re-read the note every ten seconds?\nClaim: another chat may have written it\nClaim: the file changed on disk')).toEqual({
    question: 'Why does the pane re-read the note every ten seconds?',
    rubric: ['another chat may have written it', 'the file changed on disk'],
  })
  expect(parseRecall('**Question:** Why?\n- Claim 1: one\n2. Claim: two · three')?.rubric).toEqual(['one', 'two', 'three'])
  expect(parseRecall('Claim: one\nClaim: two\nClaim: three\nClaim: four\nClaim: five')).toBe(null)
  expect(parseRecall('Question: only a question')).toBe(null)
  expect(parseRecall('Question: q\nClaim: 1\nClaim: 2\nClaim: 3\nClaim: 4\nClaim: 5')?.rubric.length).toBe(4)
  expect(recallAsk('')).toBe(recallAsk(RECALL_PROMPT))
  expect(recallAsk('Ask me about the tests.')).toContain('Ask me about the tests.')
  // The lens file the pane writes reads back as the built-in prompt, in the lens contract.
  expect(lensBody(recallLensFile('recall-question'))).toBe(RECALL_PROMPT)
  expect(recallLensFile('recall-question')).toMatch(/^---\nkind: engine\nname: recall-question\ndescription: >-\n/)
})

test('the grader is shown the question, the claims and the answer only, and its grade is read back; "none" reads as empty', async () => {
  const ask = gradeAsk('Why a fork?', ['it has the conversation', 'it adds nothing to the chat'], ' because it can see the chat ')
  expect(ask).toContain('Question: Why a fork?\nRubric (the claims a complete answer makes):\n- it has the conversation\n- it adds nothing to the chat\n\nAnswer, verbatim:\nbecause it can see the chat\n')
  expect(GRADER_SYSTEM).toContain('No praise')
  expect(parseGrade('Grade: **partial**\nMissing: "it adds nothing to the chat"\nMisconception: none\nFeedback: Say what the fork leaves alone.\nFollow-up: What does the chat see of it?')).toEqual({
    grade: 'partial', missing: '"it adds nothing to the chat"', misconception: '', feedback: 'Say what the fork leaves alone.', followUp: 'What does the chat see of it?',
  })
  expect(parseGrade('Grade: Missed.\nMisconception: thinks the fork runs tools')?.misconception).toBe('thinks the fork runs tools')
  expect(parseGrade('I think it was fine')).toBe(null)
})

test('the ladder: recalled moves up a rung, partial repeats it, missed and peeked go back to a day; a new question starts on the first', async () => {
  expect(nextRung('recalled', null)).toBe(3)
  expect(nextRung('partial', null)).toBe(1)
  expect(nextRung('missed', null)).toBe(1)
  expect(nextRung('recalled', 3)).toBe(7)
  expect(nextRung('recalled', 21)).toBe(21)
  expect(nextRung('partial', 7)).toBe(7)
  expect(nextRung('peeked', 21)).toBe(1)
  expect(addDays('2026-11-30', 1)).toBe('2026-12-01')
  expect(addDays('2026-12-25', 21)).toBe('2027-01-15')
  expect(addDays('2028-02-28', 1)).toBe('2028-02-29')
})

const graded = (grade: 'recalled' | 'partial' | 'missed' | 'peeked', misconception = '') => ({ grade, missing: '', misconception, feedback: '', followUp: '' })

test('the rung after a run of grades replays the ladder from the first rung', async () => {
  expect(rungAfter(['partial'])).toBe(1)
  expect(rungAfter(['recalled'])).toBe(3)
  expect(rungAfter(['missed', 'recalled', 'recalled'])).toBe(7)
  expect(rungAfter(['recalled', 'recalled', 'recalled', 'recalled'])).toBe(21)
  expect(rungAfter(['recalled', 'recalled', 'peeked'])).toBe(1)
})

test('a grade is kept as a row of the Recall table, the last block of the note\'s Agent Log section; everything else in the note stays as it was', async () => {
  const note = '---\noutcome: It loads fast.\n---\n# Next\n- [ ] Plant the beds\n'
  const one = recordRecall(note, { id: '', question: 'Why | a fork?', rubric: ['it has the chat', 'it adds nothing'] }, '2026-11-08', graded('partial', 'thinks the fork runs tools'))
  expect(one.id).toBe('q1')
  expect(one.due).toBe('2026-11-09')
  expect(one.text).toBe(`${note}\n# Agent Log\n\n## Recall\n| id | question | full answer | grades | due | mix-up |\n|---|---|---|---|---|---|\n| q1 | Why / a fork? | it has the chat · it adds nothing | 2026-11-08 partial | 2026-11-09 | thinks the fork runs tools |\n`)
  // The same question again goes on its row; the ladder runs over every grade; a recall clears the mix-up.
  const two = recordRecall(one.text, { id: '', question: 'Why | a fork?', rubric: ['other'] }, '2026-11-09', graded('recalled'))
  expect(two.id).toBe('q1')
  expect(two.text).toContain('| q1 | Why / a fork? | it has the chat · it adds nothing | 2026-11-08 partial, 2026-11-09 recalled | 2026-11-12 |  |')
  // A new question gets the next id.
  const three = recordRecall(two.text, { id: '', question: 'What breaks?', rubric: ['x'] }, '2026-11-09', graded('missed'))
  expect(three.id).toBe('q2')
  expect(parseRecallRows(three.text.split('## Recall\n')[1]?.split('\n') ?? []).map(row => row.id)).toEqual(['q1', 'q2'])
  // Pins and the chat list keep their places, and the table stays last.
  const busy = `${note}\n# Agent Log\n\n- [[A note]] · why\n\n> [!abstract]- Chats\n> - 2026-11-08 09:30 · fix it · \`s1\`\n`
  const kept = recordRecall(busy, { id: '', question: 'Q?', rubric: ['c'] }, '2026-11-08', graded('recalled')).text
  expect(kept).toBe(`${note}\n# Agent Log\n\n- [[A note]] · why\n\n> [!abstract]- Chats\n> - 2026-11-08 09:30 · fix it · \`s1\`\n\n## Recall\n| id | question | full answer | grades | due | mix-up |\n|---|---|---|---|---|---|\n| q1 | Q? | c | 2026-11-08 recalled | 2026-11-11 |  |\n`)
  expect(parseTrail(kept).pins.map(pin => pin.target)).toEqual(['A note'])
  expect(parseTrail(kept).chats.map(chat => chat.id)).toEqual(['s1'])
  // A pin a chat appends after the table is the body's, and the table goes back to the end.
  const appended = `${kept}- [[Later note]] · added at the end\n`
  expect(parseTrail(appended).pins.map(pin => pin.target)).toEqual(['A note', 'Later note'])
  expect(splitLog(appended).recall.length).toBe(3)
  const rewritten = recordRecall(appended, { id: 'q1', question: 'Q?', rubric: ['c'] }, '2026-11-11', graded('recalled')).text
  expect(rewritten.indexOf('[[Later note]]')).toBeLessThan(rewritten.indexOf('## Recall'))
  expect(rewritten.trimEnd().endsWith('| q1 | Q? | c | 2026-11-08 recalled, 2026-11-11 recalled | 2026-11-18 |  |')).toBe(true)
})

test('what is due: each row due today or earlier across notes, oldest first; a question recalled on two days is known', async () => {
  let a = recordRecall('# Agent Log\n', { id: '', question: 'qa1', rubric: ['x'] }, '2026-11-10', graded('missed')).text
  a = recordRecall(a, { id: 'q1', question: 'qa1', rubric: ['x'] }, '2026-11-11', graded('recalled')).text
  a = recordRecall(a, { id: '', question: 'qa2', rubric: ['y', 'z'] }, '2026-11-12', graded('recalled')).text
  a = recordRecall(a, { id: 'q2', question: 'qa2', rubric: ['y', 'z'] }, '2026-11-15', graded('recalled')).text
  a = recordRecall(a, { id: '', question: 'qa3', rubric: ['w'] }, '2026-11-17', graded('partial')).text
  const b = recordRecall('# Agent Log\n', { id: '', question: 'qb1', rubric: ['v'] }, '2026-11-11', graded('peeked')).text
  // A Recall heading above the Agent Log section is the user's, not the pane's.
  const mine = '## Recall\n| id | question | full answer | grades | due | mix-up |\n|---|---|---|---|---|---|\n| q1 | not ours | x | 2026-11-01 missed | 2026-11-02 |  |\n'
  const due = dueRecalls([{ topic: 'A', path: '/v/A.md', text: a }, { topic: 'B', path: '/v/B.md', text: b }, { topic: 'C', path: '/v/C.md', text: mine }], '2026-11-17')
  expect(due).toEqual([
    { topic: 'B', path: '/v/B.md', id: 'q1', question: 'qb1', rubric: ['v'], due: '2026-11-12' },
    { topic: 'A', path: '/v/A.md', id: 'q1', question: 'qa1', rubric: ['x'], due: '2026-11-14' },
  ])
  expect(isKnown(parseRecallRows(splitLog(a).recall)[1]!)).toBe(true)
  expect(dueRecalls([{ topic: 'A', path: '', text: a }], '2026-12-31').map(one => one.question)).toEqual(['qa1', 'qa3'])
})

test('explain-back is a saved prompt in the lens contract: the chat\'s own work, traced in the code, its grades kept in the Recall table', async () => {
  expect(lensBody(explainLensFile(EXPLAIN_LENS))).toBe(EXPLAIN_PROMPT)
  expect(explainLensFile(EXPLAIN_LENS)).toMatch(/^---\nkind: engine\nname: explain-back\ndescription: >-\n/)
  expect(EXPLAIN_PROMPT).toContain('`| id | question | full answer | grades | due | mix-up |`')
  expect(EXPLAIN_PROMPT).toContain('the diff')
  expect(EXPLAIN_PROMPT).not.toMatch(/explain-to-learn/)
})

// stateHooks with a grep that finds the notes holding a Recall table, and a grader that answers.
const recallHooks = (on: On, anchors: [string, string][], notes: Record<string, string> = {}, session = { id: 'own-session' }) => {
  const got = stateHooks(except(on, ['config.list', 'process.run', 'fs.read', 'fs.exists']), anchors, [], session)
  const all = { '/v/PMOC - Garden.md': GARDEN_PLAIN, ...notes }
  const graded: { model: string; system: string; prompt: string }[] = []
  const grader = { reply: { isAnswered: true, text: 'Grade: partial\nMissing: "it adds nothing to the chat"\nMisconception: thinks the fork runs tools\nFeedback: Say what the fork leaves alone.\nFollow-up: What does the chat see of it?' } as unknown }
  const textOf = (path: string): string => got.written.get(path) ?? (path === '/v/Kanban.md' ? BOARD_TEXT : (all as Record<string, string>)[path] ?? '')
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/v/Kanban.md', '/Users/t/.claude/projects', got.transcript].includes(e.path) || got.written.has(e.path) || e.path in all }))
  on('fs.read', (_, e) => ({ value: textOf(e.path) }))
  on('process.run', (_, e) => {
    got.ran.push(e.argv)
    const paths = [...new Set([...Object.keys(all), ...got.written.keys()])]
    let stdout = ''
    if (e.argv[0] === 'grep') {
      stdout = paths.filter(path => path.endsWith('.md') && /^## Recall\s*$/m.test(textOf(path))).map(path => `${path}\n`).join('')
    } else if (e.argv[0] === 'find') {
      const name = e.argv[e.argv.indexOf('-name') + 1] ?? ''
      const found = paths.find(path => path.endsWith(`/${name}`))
      stdout = found ? `${found}\n` : ''
    }

    return { value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('config.list', () => ({ value: ['stateAuto', 'stateLens', 'recallAuto', 'recallLens'].map(field => ({ key: `augi-ui.${field}`, label: field, kind: 'text', value: '', provider: { plugin: 'augi-ui', tier: 'user' }, isLocked: false })) as never }))
  on('model.complete', (_, e) => {
    graded.push({ model: e.model, system: String(e.system ?? ''), prompt: String(e.prompt) })
    return { value: grader.reply as never }
  })

  return { ...got, graded, grader }
}

test('quiz me asks this chat\'s model for a question; the answer is graded blind by a fresh model and kept in the anchor note\'s Recall table', { options: OPTIONS }, async ($, on) => {
  const got = recallHooks(on, [['own-session', 'PMOC - Garden']])
  got.model.reply = { isAnswered: true, text: 'Question: Why does the pane ask a fork, not the chat?\nClaim: the fork has the conversation\nClaim: it adds nothing to the chat' }
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  // The Recall card starts folded: it is there when wanted, out of the way of the chat's own cards.
  expect(await ui.find({ key: 'recall-ask' })).toBeFalsy()
  await ui.press({ key: 'head-recallOpen' })
  expect(await ui.find({ text: /^No question yet\. .*Grades are kept in PMOC - Garden, under Agent Log → Recall\.$/ })).toBeTruthy()
  expect((await ui.find({ key: 'recall-auto' }))?.text).toMatch(/☐ ask after each turn/)
  // Off by default: a turn ending asks nothing.
  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked).toEqual([])

  await ui.press({ key: 'recall-ask' })
  expect(got.asked).toEqual([recallAsk(RECALL_PROMPT)])
  expect(await ui.find({ text: 'Why does the pane ask a fork, not the chat?' })).toBeTruthy()
  // The claims stay hidden until it is answered.
  expect(await ui.find({ text: '• the fork has the conversation' })).toBeFalsy()

  // An empty answer grades nothing.
  await ui.input({ key: `recall-answer-${got.clock.now}`, text: '  ' })
  expect(got.graded).toEqual([])
  expect(got.toasts.at(-1)).toBe('Write an answer first, in your own words, or peek.')

  await ui.input({ key: `recall-answer-${got.clock.now}`, text: 'it can see the whole chat' })
  // The grader sees the question, the claims and the answer: nothing of the chat.
  expect(got.graded).toEqual([{ model: 'sonnet', system: GRADER_SYSTEM, prompt: gradeAsk('Why does the pane ask a fork, not the chat?', ['the fork has the conversation', 'it adds nothing to the chat'], 'it can see the whole chat') }])
  expect(await ui.find({ text: '◐ partial · again on 2026-11-09' })).toBeTruthy()
  expect(await ui.find({ text: 'Say what the fork leaves alone.' })).toBeTruthy()
  expect(await ui.find({ text: 'Mix-up: thinks the fork runs tools' })).toBeTruthy()
  expect(await ui.find({ text: '• it adds nothing to the chat' })).toBeTruthy()
  expect(await ui.find({ text: 'Try next: What does the chat see of it?' })).toBeTruthy()

  // Kept in the anchor's own note, and nowhere else.
  const note = got.written.get('/v/PMOC - Garden.md') ?? ''
  expect(note).toBe(`${GARDEN_PLAIN}\n# Agent Log\n\n## Recall\n| id | question | full answer | grades | due | mix-up |\n|---|---|---|---|---|---|\n| q1 | Why does the pane ask a fork, not the chat? | the fork has the conversation · it adds nothing to the chat | 2026-11-08 partial | 2026-11-09 | thinks the fork runs tools |\n`)
  expect([...got.written.keys()].filter(path => path.endsWith('.md'))).toEqual(['/v/PMOC - Garden.md'])
  // The table is one press away.
  await ui.press({ key: 'recall-open' })
  expect(got.ran.some(argv => argv[0] === 'open' && argv[1] === `obsidian://open?path=${encodeURIComponent('/v/PMOC - Garden.md')}`)).toBe(true)

  // Peeking at the next one grades nothing and counts as a miss.
  got.clock.now += 60_000
  got.model.reply = { isAnswered: true, text: 'Question: What would break without the ten-second check?\nClaim: two chats would not see each other\'s pins' }
  await ui.press({ key: 'recall-ask' })
  await ui.press({ key: 'recall-peek' })
  expect(got.graded.length).toBe(1)
  expect(await ui.find({ text: '👁 peeked · again on 2026-11-09' })).toBeTruthy()
  expect(got.written.get('/v/PMOC - Garden.md')).toContain('| q2 | What would break without the ten-second check? | two chats would not see each other\'s pins | 2026-11-08 peeked | 2026-11-09 |  |')

  // A fork with nothing to go on says why.
  got.model.reply = { isAnswered: false, reason: 'nothing-to-fork' }
  await ui.press({ key: 'recall-ask' })
  expect(got.toasts.at(-1)).toBe('Nothing to go on yet: send a prompt first.')
  await ui.unmount()
})

test('with no anchor a grade shows and is not kept anywhere', { options: OPTIONS }, async ($, on) => {
  const got = recallHooks(on, [])
  got.model.reply = { isAnswered: true, text: 'Question: Why?\nClaim: because' }
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'head-recallOpen' })
  expect(await ui.find({ text: /With no anchor, grades are not kept\.$/ })).toBeTruthy()
  await ui.press({ key: 'recall-ask' })
  await ui.input({ key: `recall-answer-${got.clock.now}`, text: 'because' })
  expect(await ui.find({ text: '◐ partial · again on 2026-11-09' })).toBeTruthy()
  expect(await ui.find({ text: 'No anchor: this grade is not kept. Anchor the chat to keep its questions in its note.' })).toBeTruthy()
  expect([...got.written.keys()].filter(path => path.endsWith('.md'))).toEqual([])
  await ui.unmount()
})

test('with recall on, a turn ending asks a question, at most every two minutes, and never over an answer being typed', { options: { ...OPTIONS, recallAuto: 'yes' } }, async ($, on) => {
  const got = recallHooks(on, [['own-session', 'PMOC - Garden']])
  got.model.reply = { isAnswered: true, text: 'Question: First?\nClaim: one' }
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'head-recallOpen' })
  expect((await ui.find({ key: 'recall-auto' }))?.text).toMatch(/☑ ask after each turn/)
  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked).toEqual([recallAsk(RECALL_PROMPT)])
  expect(await ui.find({ text: 'First?' })).toBeTruthy()

  // Too soon: nothing new.
  got.model.reply = { isAnswered: true, text: 'Question: Second?\nClaim: two' }
  await $.turn.complete(turnEnds)
  await settle()
  expect(got.asked.length).toBe(1)

  // An answer being typed is never replaced, however long ago the question came.
  got.clock.now += RECALL_EVERY_MS
  await ui.input({ key: `recall-answer-${got.clock.now - RECALL_EVERY_MS}`, text: 'half an answer', kind: 'change' } as never)
  await $.turn.complete(turnEnds)
  await settle()
  expect(await ui.find({ text: 'First?' })).toBeTruthy()
  await ui.unmount()
})

test('a due recall comes back from a note\'s Recall table: the You card counts it, pressing it puts it in the card, and the grade goes on its row', { options: OPTIONS }, async ($, on) => {
  const old = recordRecall('# Next\n', { id: '', question: 'Why a fork?', rubric: ['it has the chat'] }, '2026-11-05', graded('recalled')).text
  const got = recallHooks(on, [['own-session', 'PMOC - Garden']], { '/v/Projects/PMOC - Pond.md': old })
  got.grader.reply = { isAnswered: true, text: 'Grade: recalled\nMissing: none\nMisconception: none\nFeedback: All claims met.\nFollow-up: none' }
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'head-recallOpen' })
  expect(await ui.find({ text: '🎯 Recall · 1 due' })).toBeTruthy()
  await ui.press({ key: 'you-due' })
  expect(await ui.find({ text: 'Why a fork?' })).toBeTruthy()
  expect(await ui.find({ text: 'due again · PMOC - Pond' })).toBeTruthy()
  // No model was asked for a question: it came from the note.
  expect(got.asked).toEqual([])
  await ui.input({ key: `recall-answer-${got.clock.now}`, text: 'it reads the chat' })
  expect(await ui.find({ text: '✓ recalled · again on 2026-11-15' })).toBeTruthy()
  // Kept on its own row in its own note, not the anchor's.
  const after = got.written.get('/v/Projects/PMOC - Pond.md') ?? ''
  expect(after).toContain('| q1 | Why a fork? | it has the chat | 2026-11-05 recalled, 2026-11-08 recalled | 2026-11-15 |  |')
  expect(got.written.has('/v/PMOC - Garden.md')).toBe(false)
  expect(await ui.find({ key: 'you-due' })).toBeFalsy()
  await ui.unmount()
})

test('explain it back writes the explain-back lens when the folder has none, then drafts it with this chat\'s context', { options: OPTIONS }, async ($, on) => {
  const got = recallHooks(on, [['own-session', 'PMOC - Garden']])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'head-recallOpen' })
  await ui.press({ key: 'recall-explain' })
  expect(got.written.get('/v/lenses/explain-back.md')).toBe(explainLensFile(EXPLAIN_LENS))
  expect(got.filled.length).toBe(1)
  expect(got.filled[0]).toContain('- Lens: read "/v/lenses/explain-back.md" and run it as written, step by step, writing the files and the lines it asks for.')
  // A lens that writes is not held to a short answer in chat with no files changed.
  expect(got.filled[0]).not.toMatch(/at most 10 lines|Don't change any files|Answer here in chat/)
  // A sent draft is titled by its lens, as any lens draft is.
  expect(chatTitleFrom(got.filled[0] ?? '')).toMatch(/^explain-back/)
  expect(got.filled[0]).toContain('[[PMOC - Garden]]')
  // An edited lens is never written over.
  got.written.set('/v/lenses/explain-back.md', '---\nname: explain-back\n---\n# explain-back\n\nMine.\n')
  await ui.press({ key: 'recall-explain' })
  expect(got.written.get('/v/lenses/explain-back.md')).toContain('Mine.')
  await ui.unmount()
})

// ---- The week's note ----

const WEDNESDAY = new Date(2022, 0, 5, 9, 30).getTime()
const WEEK_PATTERN = 'Inbox/{{date:YYYY-MM-DD}} - week'

test('a week runs Monday to Sunday, and its note may be named for any of its days, Monday first', () => {
  expect(weekDays(WEDNESDAY).map(ms => formatDate('YYYY-MM-DD ddd', ms))).toEqual([
    '2022-01-03 Mon', '2022-01-04 Tue', '2022-01-05 Wed', '2022-01-06 Thu', '2022-01-07 Fri', '2022-01-08 Sat', '2022-01-09 Sun',
  ])
  // A Sunday night is still the week that began the Monday before; a week can start in the year before.
  expect(formatDate('YYYY-MM-DD', weekDays(new Date(2022, 0, 9, 23, 59).getTime())[0] ?? 0)).toBe('2022-01-03')
  expect(formatDate('YYYY-MM-DD', weekDays(new Date(2022, 0, 1, 8).getTime())[0] ?? 0)).toBe('2021-12-27')
  expect(weekNotePaths(WEEK_PATTERN, WEDNESDAY, '/v', '/Users/t')).toEqual(
    ['03', '04', '05', '06', '07', '08', '09'].map(day => `/v/Inbox/2022-01-${day} - week.md`),
  )
  // A pattern with no day in it is one note; no pattern, none.
  expect(weekNotePaths('Weekly/current', WEDNESDAY, '/v', '/Users/t')).toEqual(['/v/Weekly/current.md'])
  expect(weekNotePaths('', WEDNESDAY, '/v', '/Users/t')).toEqual([])
})

const WEEK_NOTE = [
  '# This week',
  '- [ ] [[PMOC - Garden]] · beds planted',
  '\t- [ ] buy soil',
  '- [x] [[PMOC - Birdhouse]]',
  '- [ ] call the plumber',
  '',
  '## Monday 2022-01-03',
  '- [ ] a box under a day is not a workstream',
  '- 09:00 coffee',
  '',
].join('\n')

test('the week\'s workstreams are the top-level checkboxes right under its heading', () => {
  expect(weekGoals(WEEK_NOTE, 'this week')).toEqual([
    { text: 'PMOC - Garden · beds planted', note: 'PMOC - Garden', done: false, line: '- [ ] [[PMOC - Garden]] · beds planted' },
    { text: 'PMOC - Birdhouse', note: 'PMOC - Birdhouse', done: true, line: '- [x] [[PMOC - Birdhouse]]' },
    { text: 'call the plumber', note: null, done: false, line: '- [ ] call the plumber' },
  ])
  expect(weekGoals(WEEK_NOTE, 'Next week')).toEqual([])
  expect(weekGoals('## This week\n', 'This week')).toEqual([])
})

test('a new week\'s note carries over last week\'s open workstreams with their details, and nothing else', () => {
  expect(newWeekNote(WEEK_NOTE, 'This week')).toBe('# This week\n\n- [ ] [[PMOC - Garden]] · beds planted\n\t- [ ] buy soil\n- [ ] call the plumber\n')
  expect(newWeekNote('', 'This week')).toBe('# This week\n\n- [ ] \n')
  expect(newWeekNote('# This week\n- [x] done\n', 'Focus')).toBe('# Focus\n\n- [ ] \n')
})

test('with a week note set, the You card keeps this week\'s note: made on a press for Monday with last week carried over, found under any day\'s name, its workstreams listed, a line logged under today', { options: { ...OPTIONS, weekNote: WEEK_PATTERN } }, async ($, on) => {
  const lastWeek = '/v/Inbox/2021-12-29 - week.md'
  const written = new Map<string, string>([[lastWeek, WEEK_NOTE], ['/v/PMOC - Garden.md', '# Journal\n']])
  const toasts: string[] = []
  const got = todayHooks(on, written, toasts)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'today-make' }))?.text).toMatch(/make this week's note/)
  expect(await ui.find({ text: /No note for this week yet: it would be Inbox\/2022-01-03 - week\.md/ })).toBeTruthy()
  // Nothing is made until the press; then Monday's note, last week's open workstreams in it.
  expect([...written.keys()].some(path => path.includes('2022-01-0'))).toBe(false)
  await ui.press({ key: 'today-make' })
  const monday = '/v/Inbox/2022-01-03 - week.md'
  expect(written.get(monday)).toBe('# This week\n\n- [ ] [[PMOC - Garden]] · beds planted\n\t- [ ] buy soil\n- [ ] call the plumber\n')
  expect((await ui.find({ key: 'today-open' }))?.text).toMatch(/this week's note/)
  expect(await ui.find({ text: 'This week · 2022-01-03 - week' })).toBeTruthy()
  // A linked workstream opens its note; a plain one is text.
  expect((await ui.find({ key: 'week-goal-0' }))?.text).toMatch(/☐ PMOC - Garden · beds planted ↗/)
  expect(await ui.find({ text: '☐ call the plumber' })).toBeTruthy()

  // + log goes under today's dated entry, an H3 made at the end when missing.
  await ui.input({ key: 'today-log-0', text: 'gate is fixed' })
  expect(written.get(monday)?.endsWith('- [ ] call the plumber\n\n### 2022-01-05\n- 09:30 gate is fixed\n')).toBe(true)
  expect(toasts).toEqual(["Logged under today in this week's note"])
  // An entry the user wrote that day, words after the date, is the one it goes under.
  written.set(monday, `${written.get(monday)?.replace('### 2022-01-05\n- 09:30 gate is fixed\n', '### 2022-01-05 hinges came\nthey fit\n\n### 2022-01-04\n- older\n')}`)
  await ui.input({ key: 'today-log-1', text: 'oiled them' })
  expect(written.get(monday)?.endsWith('### 2022-01-05 hinges came\nthey fit\n- 09:30 oiled them\n\n### 2022-01-04\n- older\n')).toBe(true)
  // The dated entries below the list are never workstreams.
  expect(await ui.find({ key: 'week-goal-2' })).toBeFalsy()

  // The draft names the week's note.
  await pressDraft(ui)
  expect(got.filled[0]?.includes(`- This week: read "${monday}", my note for this week`)).toBe(true)

  // Named for another day of the week, it is still this week's note.
  written.delete(monday)
  const thursday = '/v/Inbox/2022-01-06 - week.md'
  written.set(thursday, '# This week\n- [x] [[PMOC - Garden]]\n')
  await ui.press({ key: 'refresh' })
  expect(await ui.find({ text: 'This week · 2022-01-06 - week' })).toBeTruthy()
  expect((await ui.find({ key: 'week-goal-0' }))?.text).toMatch(/☑ PMOC - Garden/)
  await ui.press({ key: 'week-goal-0' })
  await ui.unmount()
  // The paths the next tests see are the shared ones again.
  configure(resolvePaths(OPTIONS, '/Users/t'))
})

// ---- Board sync from the week's workstreams ----

const goal = (line: string) => weekGoals(`# This week\n${line}\n`, 'This week')[0] ?? { text: '', note: null, done: false, line }

test('sync works out what the board lacks: a card for each open workstream not on it, a tick for each done one, the cards that are not this week, and the workstreams with no note', () => {
  const cards = kanbanCards(MANY_COLUMNS)
  const goals = [
    goal('- [ ] [[PMOC - Garden]] · on the board already'),
    goal('- [x] [[PMOC - Birdhouse]]'),
    goal('- [ ] [[Dig the hole]] · a sub-item counts as on the board'),
    goal('- [ ] [[New thing]]'),
    goal('- [ ] call the plumber'),
    goal('- [x] a done plain one'),
  ]
  expect(boardSync(goals, cards)).toEqual({
    add: ['New thing'],
    tick: ['PMOC - Birdhouse'],
    // Pond holds a workstream as its sub-item: it is this week's too.
    stale: [],
    unlinked: [goals[4]],
  })
  // Only Garden this week: the others are listed.
  expect(boardSync([goals[0] ?? goal('')], cards).stale).toEqual(['PMOC - Birdhouse', 'PMOC - Pond'])
  // A note both open and done this week is not ticked.
  expect(boardSync([goal('- [x] [[PMOC - Garden]]'), goal('- [ ] [[PMOC - Garden]] · still going')], cards).tick).toEqual([])
})

test('a card is ticked where it is, the open one above the archive; nothing else moves', () => {
  expect(tickCard(MANY_COLUMNS, 'PMOC - Birdhouse')).toBe(MANY_COLUMNS.replace('- [ ] [[PMOC - Birdhouse]]', '- [x] [[PMOC - Birdhouse]]'))
  expect(tickCard(MANY_COLUMNS, 'PMOC - Garden')).toBe(MANY_COLUMNS.replace('- [ ] [[PMOC - Garden]] — the main project', '- [x] [[PMOC - Garden]] — the main project'))
  expect(tickCard(MANY_COLUMNS, 'Archived thing')).toBe(MANY_COLUMNS)
})

test('a workstream line links its new note, its words kept when they differ from the name; only that line changes', () => {
  const week = '# This week\n- [ ] call the plumber\n- [ ] fix: the gate\n'
  expect(linkGoal(week, '- [ ] call the plumber', 'call the plumber')).toBe('# This week\n- [ ] [[call the plumber]]\n- [ ] fix: the gate\n')
  expect(linkGoal(week, '- [ ] fix: the gate', 'fix- the gate')).toBe('# This week\n- [ ] call the plumber\n- [ ] [[fix- the gate]] · fix: the gate\n')
  expect(linkGoal(week, '- [ ] gone', 'gone')).toBe(week)
  // Case aside, the same words are the name: nothing kept after it.
  expect(linkGoal('- [ ] pmoc - garden\n', '- [ ] pmoc - garden', 'PMOC - Garden')).toBe('- [ ] [[PMOC - Garden]]\n')
})

test('the open boxes in the day entries are listed with the heading they sit under; the workstream list and ticked boxes are not', () => {
  const week = '# This week\n- [ ] [[A]]\n\n### 2022-01-04\nvented a while\n- [ ] fix the eval pipeline\n\t- [ ] a detail\n- [x] done already\n\n### 2022-01-05 calmer\n- [ ] call the plumber\n'
  expect(entryBoxes(week, 'This week')).toEqual([
    { under: '2022-01-04', line: '- [ ] fix the eval pipeline' },
    { under: '2022-01-04', line: '- [ ] a detail' },
    { under: '2022-01-05 calmer', line: '- [ ] call the plumber' },
  ])
  expect(entryBoxes('# This week\n- [ ] [[A]]\n', 'This week')).toEqual([])
})

test('the sync draft lists the sure edits as rows and the rest to resolve, and changes nothing until the user answers', () => {
  const left = {
    path: '/v/Inbox/2022-01-03 - week.md',
    sure: ['add a card "- [ ] [[PMOC - Pond]]" at the end of the "Ideas" column'],
    unlinked: [goal('- [ ] call the plumber')],
    entries: [{ under: '2022-01-04', line: '- [ ] fix the eval pipeline' }],
    stale: ['2022-01-04 - Sketch'],
    noKey: [{ note: 'PMOC - Garden', path: '/v/PMOC - Garden.md' }],
    jira: { project: 'PROJ', server: 'atlassian' },
    column: 'Ideas',
    template: 'the template "/v/OpenAugi/AGENT/templates/Task.md"',
  }
  const draft = weekSyncDraft(left)
  for (const part of [
    '- This week\'s note: "/v/Inbox/2022-01-03 - week.md".',
    '  - add a card "- [ ] [[PMOC - Pond]]" at the end of the "Ideas" column',
    '  - - [ ] call the plumber',
    '  - under "2022-01-04": - [ ] fix the eval pipeline',
    '  - [[2022-01-04 - Sketch]]',
    '  - [[PMOC - Garden]] "/v/PMOC - Garden.md"',
    'in project PROJ using the atlassian MCP server',
    'Change nothing until I answer `all`, row numbers, or `none`.',
    'at the end of the "Ideas" column',
    'Change nothing in Jira.',
  ]) {
    expect(draft.includes(part)).toBe(true)
  }
  // Nothing about a kind of item that has none, and nothing about Jira when it is off.
  const bare = weekSyncDraft({ ...left, sure: [], entries: [], stale: [], jira: null })
  expect(bare.includes('Sure, worked out by the pane')).toBe(false)
  expect(bare.includes('day entries')).toBe(false)
  expect(bare.includes('not this week')).toBe(false)
  expect(bare.includes('Jira key')).toBe(false)
  expect(anchorFromDraft(draft)).toBe(null)
})

test('one press drafts the week\'s sync and writes nothing: the sure edits as rows (a plain line that names a note, a missing card, a done card) and the rest for the chat; with nothing to do, no draft', { options: { ...OPTIONS, weekNote: WEEK_PATTERN } }, async ($, on) => {
  const monday = '/v/Inbox/2022-01-03 - week.md'
  const week = '# This week\n- [ ] pmoc - garden\n- [ ] [[PMOC - Pond]]\n- [x] [[2022-01-04 - Sketch]]\n- [ ] call the plumber\n\n### 2022-01-04\n- [ ] fix the eval pipeline\n'
  const written = new Map<string, string>([[monday, week], ['/v/PMOC - Garden.md', '# Journal\n']])
  const toasts: string[] = []
  const got = todayHooks(on, written, toasts)
  const copied: string[] = []
  on('ui.copy', (_, e) => {
    copied.push(e.text)
    return { value: { isCopied: true } }
  })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  // One button. Pond to add, Sketch to tick, two plain workstreams, one box in a day entry,
  // and Garden's card, not this week's while its line is plain.
  expect((await ui.find({ key: 'sync-week' }))?.text).toMatch(/⇅ sync this week \(6\)/)
  await ui.press({ key: 'sync-week' })

  // Nothing is written: not the week's note, not the board.
  expect(written.get(monday)).toBe(week)
  expect(written.has('/v/Kanban.md')).toBe(false)
  // One draft. The sure rows: the plain line that is a card's name (any case), Pond's card,
  // Sketch's tick. Garden is then this week's, so no stale card is left; Jira is off.
  expect(got.filled.length).toBe(1)
  const draft = got.filled[0] ?? ''
  for (const part of [
    '  - link "- [ ] pmoc - garden" to [[PMOC - Garden]], an existing card',
    '  - add a card "- [ ] [[PMOC - Pond]]" at the end of the "Ideas" column',
    '  - tick the card for [[2022-01-04 - Sketch]] where it is',
    '  - - [ ] call the plumber',
    '  - under "2022-01-04": - [ ] fix the eval pipeline',
  ]) {
    expect(draft.includes(part)).toBe(true)
  }
  expect(draft.includes('not this week')).toBe(false)
  expect(toasts.at(-1)).toBe('Sync drafted: press enter to see every edit as one table. Nothing is changed until you answer.')
  // What was half-typed in the prompt box went to the clipboard first.
  expect(copied).toEqual(['what next'])

  // Once the chat made the edits, a press drafts nothing and says so.
  written.set(monday, '# This week\n- [ ] [[PMOC - Garden]]\n- [ ] [[PMOC - Pond]]\n\n### 2022-01-04\n- [x] fix the eval pipeline\n')
  written.set('/v/Kanban.md', BOARD_TEXT.replace('- [ ] [[2022-01-04 - Sketch]] ^abc123', '- [ ] [[PMOC - Pond]]\n- [x] [[2022-01-04 - Sketch]] ^abc123'))
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'sync-week' }))?.text).toMatch(/⇅ sync this week ✓/)
  await ui.press({ key: 'sync-week' })
  expect(got.filled.length).toBe(1)
  expect(toasts.at(-1)).toBe('This week and the board match: nothing to sync.')
  await ui.unmount()
  configure(resolvePaths(OPTIONS, '/Users/t'))
})

// ---- Close out ----

test('closing out ticks a linked sub-item when the note has no card of its own, the week\'s line under its heading only, and the chat\'s log line', () => {
  expect(tickCard(MANY_COLUMNS, 'Dig the hole')).toBe(MANY_COLUMNS.replace('\t- [ ] [[Dig the hole]]', '\t- [x] [[Dig the hole]]'))
  expect(tickGoal(WEEK_NOTE, 'This week', 'PMOC - Garden')).toBe(WEEK_NOTE.replace('- [ ] [[PMOC - Garden]] · beds planted', '- [x] [[PMOC - Garden]] · beds planted'))
  expect(tickGoal(WEEK_NOTE, 'This week', 'call the plumber')).toBe(WEEK_NOTE)
  expect(tickGoal(WEEK_NOTE, 'Next week', 'PMOC - Garden')).toBe(WEEK_NOTE)
  const log = addChat('# Journal\n', { when: '2022-01-05 09:30', title: 'fix the gate', id: 'abc', done: false })
  const done = markChatDone(log, 'abc')
  expect(parseTrail(done).chats).toEqual([{ when: '2022-01-05 09:30', title: 'fix the gate', id: 'abc', done: true }])
  expect(markChatDone(done, 'abc')).toBe(done)
  expect(markChatDone(log, 'other')).toBe(log)
})

test('close out asks first, then ticks the anchor\'s card and its line in this week\'s note and marks this chat done; the chat keeps its anchor', { options: { ...OPTIONS, weekNote: WEEK_PATTERN } }, async ($, on) => {
  const monday = '/v/Inbox/2022-01-03 - week.md'
  const week = '# This week\n- [ ] [[PMOC - Garden]]\n- [ ] [[PMOC - Pond]]\n'
  const written = new Map<string, string>([[monday, week], ['/v/PMOC - Garden.md', '# Journal\n']])
  const toasts: string[] = []
  todayHooks(on, written, toasts)
  await $.command.run({ command: 'augi', args: '' })
  await $.prompt.submit({ text: 'fix the gate' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(parseTrail(written.get('/v/PMOC - Garden.md') ?? '').chats.map(chat => chat.done)).toEqual([false])
  // The first press only asks; cancel changes nothing.
  await ui.press({ key: 'close-out' })
  expect(await ui.find({ text: /^Close out PMOC - Garden\? Ticks its card on the board and its line in this week's note/ })).toBeTruthy()
  await ui.press({ key: 'close-cancel' })
  expect(written.has('/v/Kanban.md')).toBe(false)
  await ui.press({ key: 'close-out' })
  await ui.press({ key: 'close-yes' })
  expect(written.get('/v/Kanban.md')).toBe(BOARD_TEXT.replace('- [ ] [[PMOC - Garden|the garden]]', '- [x] [[PMOC - Garden|the garden]]'))
  expect(written.get(monday)).toBe(week.replace('- [ ] [[PMOC - Garden]]', '- [x] [[PMOC - Garden]]'))
  expect(parseTrail(written.get('/v/PMOC - Garden.md') ?? '').chats.map(chat => chat.done)).toEqual([true])
  expect(toasts).toContain("Closed out PMOC - Garden: its card ticked, ticked in this week's note, this chat marked done in its log.")
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  // Again: nothing left to close.
  await ui.press({ key: 'close-out' })
  await ui.press({ key: 'close-yes' })
  expect(toasts.at(-1)).toBe('Nothing left open for PMOC - Garden: no open card, week line or chat entry.')
  await ui.unmount()
  configure(resolvePaths(OPTIONS, '/Users/t'))
})

// ---- The line above the prompt ----

test('the line above the prompt names the anchor, what waits on the user or else the next step, and recalls due; with neither anchor nor due, nothing', () => {
  const state = 'Doing: wiring the gate\nLast: hung it\nNext: oil the hinges\nWaiting on me: no'
  expect(bandLine('PMOC - Garden', state, 0)).toBe('⚓ PMOC - Garden  ·  → oil the hinges')
  expect(bandLine('PMOC - Garden', state.replace('Waiting on me: no', '- **Waiting on me:** pick a paint'), 2)).toBe('⚓ PMOC - Garden  ·  ⏳ pick a paint  ·  🎯 2 due')
  expect(bandLine('PMOC - Garden', '', 0)).toBe('⚓ PMOC - Garden')
  expect(bandLine('', '', 3)).toBe('🎯 3 due')
  expect(bandLine('', state, 0)).toBe('')
})

const BAND = {
  plugin: 'augi-ui',
  component: 'AbovePrompt',
  props: { hasSurvey: false, isCollapsed: false, isWorking: false, maxRows: 6, bodyColumns: 80, scroll: { offset: 0, bodyRows: 6 }, view: {} },
} as const

test('an anchored chat shows its line above the prompt once the pane has loaded; pressing it opens the pane; band no hides it', { options: OPTIONS }, async ($, on) => {
  const opened: string[] = []
  todayHooks(except(on, ['ui.open']), new Map([['/v/PMOC - Garden.md', '# Journal\n']]), [])
  on('ui.open', (_, e) => {
    opened.push(e.id)
    return { value: { isPlaced: true } }
  })
  const before = await $.ui.mount({ ...BAND, surface: 'terminal' })
  expect(await before.find({ key: 'band-open' })).toBeFalsy()
  await before.unmount()
  await $.command.run({ command: 'augi', args: '' })
  const band = await $.ui.mount({ ...BAND, surface: 'terminal' })
  expect((await band.find({ key: 'band-open' }))?.text).toMatch(/⚓ PMOC - Garden/)
  opened.length = 0
  await band.press({ key: 'band-open' })
  expect(opened).toEqual(['augi'])
  await band.unmount()
})

test('with band no there is no line above the prompt', { options: { ...OPTIONS, band: 'no' } }, async ($, on) => {
  todayHooks(on, new Map([['/v/PMOC - Garden.md', '# Journal\n']]), [])
  await $.command.run({ command: 'augi', args: '' })
  const band = await $.ui.mount({ ...BAND, surface: 'terminal' })
  expect(await band.find({ key: 'band-open' })).toBeFalsy()
  await band.unmount()
})
