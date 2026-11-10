import type { On } from 'claude-code'
import type { Pin } from '../types'
import { expect, test } from 'claude-code/testing'

import { parseUsage, usageName, usageReport } from './register'
import { addDailyLine, dailyNotePath, dailyPattern, dailyTemplatePath, fillTemplate, formatDate, openBoxes } from './register'
import { folderKey, promptTitle } from './register'
import { extraPins, isListedTouch, storedPins, openArgv, shortPath, touchedPin, withTouched } from './register'
import { anchorCard, findInJiraDraft, insertCard, insertSubItem, jiraBrowseUrl, jiraFrom, jiraKeyFrom, jiraServerFrom, jiraUpdateDraft, newTaskDraft, parseNewTask, TASK_TEMPLATE, taskNoteName, taskNoteText, taskTemplateState, withJira } from './register'
import { addChat, addJournalEntry, anchorFromDraft, anchorPairs, artifactsFirst, retitleChat, withAnchor, withArtifact, logName, nextFrom, addPin, contextItems, parsePin, pathsFromSearch, pinFrom, pinLine, removePin, sessionPin, setPinOn, resumeCommand, setDone, configure, editorArgv, obsidianConfigs, resolvePaths, setupItems, vaultsFrom, ago, anchorFrom, buildDraft, chatTitleFrom, descriptionFrom, kanbanCards, lensFileName, lensNames, newLensSkeleton, onePerCodexRun, outcomeFrom, parseTrail, removeChat, setChatDone, lastActive, tidyTrail, trailPath, trailTemplate, parseClaudeHead, parseCodexHead, toggled, typedFrom } from './register'

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
// An Agent Log with one more open question on top, as an agent would write it there.
const withQuestion = (log: string, when: string, text: string): string => log.replace('## Questions\n', `## Questions\n\n- [ ] ${when} · ${text}`)
const human = row({ type: 'user', entrypoint: 'cli', cwd: '/r/garden', message: { content: 'fix the sprinkler' } })
const SESSION = { path: '/p/a.jsonl', tool: 'claude', repo: 'garden', title: 'fix the sprinkler', mtimeMs: 0 } as const
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
const ANCHOR = { note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md', trail: '/v/Notes/Agent Log - PMOC - Garden.md' }
const SESSION_PIN = { on: true, kind: 'session', target: '/p/a.jsonl', label: 'garden · fix the sprinkler' } as const

test('the anchor is the first bold line of the season section', async () => {
  const text = '## 2. Lanes\n**Not this.**\n## 3. This season\n> quote\n**Primary: the garden.** more\n## 4. Weekly'
  expect(anchorFrom(text)).toBe('Primary: the garden.')
})

test('a description reads as folded or plain', async () => {
  expect(descriptionFrom('name: a\ndescription: >-\n  "Have I?" yes\nscope: b')).toBe('"Have I?" yes')
  expect(descriptionFrom('description: Gather everything\n')).toBe('Gather everything')
})

test('a person session is kept, titled, and named by its repo', async () => {
  expect(parseClaudeHead(human)).toEqual({ repo: 'garden', title: 'fix the sprinkler' })
  const titled = `${human}\n${row({ type: 'custom-title', customTitle: 'Sprinkler fix' })}`
  expect(parseClaudeHead(titled)?.title).toBe('Sprinkler fix')
})

test('headless runs and dispatched tasks are not coding sessions', async () => {
  expect(parseClaudeHead(row({ type: 'user', entrypoint: 'sdk-cli', cwd: '/r', message: { content: 'x' } }))).toBe(null)
  expect(parseClaudeHead(row({ type: 'user', cwd: '/r', message: { content: 'Read your skill file first: y' } }))).toBe(null)
  expect(parseClaudeHead('{"type":"queue-operation"}\nnot json')).toBe(null)
  expect(parseClaudeHead(row({ type: 'user', cwd: '/r', message: { content: 'Context for this chat:\n- x' } }))).toBe(null)
})

test('pasted-content tags stay out of the title', async () => {
  const pasted = row({ type: 'user', cwd: '/r/x', message: { content: 'see <pasted_content id="1">the log</pasted_content>' } })
  expect(parseClaudeHead(pasted)?.title).toBe('see the log')
})

test('a meta row is not the opening prompt', async () => {
  const meta = row({ type: 'user', isMeta: true, cwd: '/r/x', message: { content: 'caveat' } })
  expect(parseClaudeHead(`${meta}\n${human}`)?.title).toBe('fix the sprinkler')
})

test('a slash command row is not the opening prompt', async () => {
  const command = row({ type: 'user', cwd: '/r/x', message: { content: '<command-name>/augi</command-name>' } })
  expect(parseClaudeHead(`${command}\n${human}`)?.title).toBe('fix the sprinkler')
  expect(parseClaudeHead(command)).toBe(null)
})

test('one codex multi-agent run shows once', async () => {
  const codex = (id: string, stamp: string) => ({ ...SESSION, tool: 'codex' as const, path: `/d/rollout-${stamp}-${id}-f011-7942.jsonl` })
  const kept = onePerCodexRun([
    codex('01a174fe', '2026-10-25T17-37-07'),
    codex('01a174ff', '2026-10-25T17-37-07'),
    codex('01a17500', '2026-10-25T18-00-00'),
    SESSION,
  ])
  expect(kept.length).toBe(3)
})

test('codex keeps only threads a person started, even when the line is cut', async () => {
  expect(parseCodexHead('{"payload":{"cwd":"/r/craft","thread_source":"user","base_instr')?.repo).toBe('craft')
  expect(parseCodexHead('{"payload":{"cwd":"/r/craft","thread_source":"subagent"')).toBe(null)
})

test('ago reads minutes, hours, days', async () => {
  expect(ago(600000, 0)).toBe('10m')
  expect(ago(3 * 3600000, 0)).toBe('3h')
  expect(ago(72 * 3600000, 0)).toBe('3d')
})

test('the draft carries what is ticked and leaves out what is not', async () => {
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

test('a sub-item goes last under its parent card, beside a parent that is a sub-item, nowhere for a parent off the board', async () => {
  const item = '\t- [ ] [[Fix the gate]]'
  const under = insertSubItem(MANY_COLUMNS, 'Fix the gate', 'PMOC - Garden')
  expect(under.split('\n').indexOf(item)).toBe(MANY_COLUMNS.split('\n').indexOf('\t[[Some related note]]') + 1)
  expect(without(under, item)).toBe(MANY_COLUMNS)
  expect(under.includes('- [ ] [[PMOC - Garden]] — the main project ^abc123\n\t- [ ] Plant the beds ^def456\n')).toBe(true)
  expect(kanbanCards(under)[0]?.tasks).toEqual([{ text: 'Plant the beds', note: null }, { text: 'Fix the gate', note: 'Fix the gate' }])

  // A card with no sub-items takes it right under its line.
  const none = insertSubItem(MANY_COLUMNS, 'Fix the gate', 'PMOC - Birdhouse')
  expect(none.split('\n').indexOf(item)).toBe(MANY_COLUMNS.split('\n').indexOf('- [ ] [[PMOC - Birdhouse]]') + 1)
  expect(without(none, item)).toBe(MANY_COLUMNS)

  // A parent that is a linked sub-item: beside it, under the same card.
  const beside = insertSubItem(MANY_COLUMNS, 'Fix the gate', 'Dig the hole')
  expect(beside.split('\n').indexOf(item)).toBe(MANY_COLUMNS.split('\n').indexOf('\t- [ ] [[Dig the hole]]') + 1)
  expect(without(beside, item)).toBe(MANY_COLUMNS)
  expect(kanbanCards(beside).find(card => card.note === 'PMOC - Pond')?.tasks.map(task => task.note)).toEqual(['Dig the hole', 'Fix the gate'])

  // Off the board, or only in the archive: nothing changes.
  for (const parent of ['Archived thing', 'Not on the board']) {
    expect(insertSubItem(MANY_COLUMNS, 'Fix the gate', parent)).toBe(MANY_COLUMNS)
  }
  for (const board of [under, none, beside]) {
    expect(board.slice(board.indexOf('***'))).toBe(MANY_COLUMNS.slice(MANY_COLUMNS.indexOf('***')))
  }
  // Above the Kanban settings block on a board with no archive.
  const settings = '## Doing\n\n- [ ] [[A]]\n\n\n%% kanban:settings\n```\n{}\n```\n%%'
  expect(insertSubItem(settings, 'B', 'A')).toBe('## Doing\n\n- [ ] [[A]]\n\t- [ ] [[B]]\n\n\n%% kanban:settings\n```\n{}\n```\n%%')
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
  expect(linked.includes('Agent Log')).toBe(false)
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

test('a new-task draft asks the chat to fill in the note and its Agent Log, and is titled as ours', async () => {
  const text = newTaskDraft('/v/Fix the gate.md', '/v/Notes/Agent Log - Fix the gate.md', 'Fix the gate')
  expect(text.startsWith('Context for this chat:\n')).toBe(true)
  expect(text.includes('"/v/Fix the gate.md"')).toBe(true)
  expect(text.includes("under the note's Outcome and Verify headings")).toBe(true)
  expect(text.includes('no frontmatter')).toBe(true)
  expect(text.includes('## Context in its Agent Log "/v/Notes/Agent Log - Fix the gate.md"')).toBe(true)
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

test('a trail note starts empty, takes chats newest first, once each', async () => {
  const empty = trailTemplate('PMOC - Garden', '2026-11-08')
  expect(empty.includes('seen')).toBe(false)
  expect(empty.includes('\n# ')).toBe(false)
  expect(empty.split('---\n')[2]?.split('\n')[1]?.startsWith('Human note: [[PMOC - Garden]]. ')).toBe(true)
  expect(parseTrail(empty)).toEqual({ questions: [], chats: [], saved: [], pins: [] })

  const one = addChat(empty, { id: 's1', when: '2026-11-08 10:00', title: 'echoes: why slow', done: false })
  const two = addChat(one, { id: 's2', when: '2026-11-08 11:00', title: 'second', done: false })
  expect(addChat(two, { id: 's1', when: '2026-11-08 12:00', title: 'again', done: false })).toBe(two)
  expect(parseTrail(two).chats.map(chat => chat.id)).toEqual(['s2', 's1'])

  const saved = two.replace('## Saved\n', '## Saved\n- [[A note]] from a chat\n')
  expect(parseTrail(saved).saved).toEqual(['A note'])
})

test('questions an agent left under ## Questions are read newest first and can be ticked done', async () => {
  const one = withQuestion(trailTemplate('X', '2026-11-08'), '2026-11-08 10:00', 'does the sprinkler cache?')
  const two = withQuestion(one, '2026-11-08 11:00', 'side quest: try arrow')
  expect(parseTrail(two).questions.map(q => q.text)).toEqual(['side quest: try arrow', 'does the sprinkler cache?'])

  const done = setDone(two, { when: '2026-11-08 10:00', text: 'does the sprinkler cache?', done: false }, true)
  expect(parseTrail(done).questions.map(q => q.done)).toEqual([false, true])
  expect(setDone(done, { when: '2026-11-08 10:00', text: 'does the sprinkler cache?', done: true }, false)).toBe(two)
})

test('what goes in lists exactly the paths the draft names, and toggles take them out', async () => {
  const anchor = { note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md', trail: trailPath('PMOC - Garden') }
  const note = { on: true, kind: 'note', target: 'Sprinkler Benchmarks', label: 'Sprinkler Benchmarks', path: '/v/Notes/Sprinkler Benchmarks.md' } as const
  const off = { on: false, kind: 'file', target: '/r/docs/ARCH.md', label: 'ARCH.md' } as const
  const pins = [SESSION_PIN, note, off]
  const items = contextItems(pins, ALL_ON, anchor)
  const text = buildDraft(pins, ALL_ON, null, 'go', anchor)
  expect(items.map(item => item.key)).toEqual(['in-note', 'in-trail', 'in-pin-session-/p/a.jsonl', 'in-pin-note-Sprinkler Benchmarks', 'in-context', 'in-board', 'in-search'])
  for (const item of items.filter(one => one.path)) {
    expect(text.includes(item.path)).toBe(true)
  }
  expect(text.includes('/r/docs/ARCH.md')).toBe(false)
  expect(contextItems(pins, { project: false, aboutMe: false, openAugi: false, today: false }, anchor).map(item => item.kind)).toEqual(['pin', 'pin'])
  expect(contextItems(pins, ALL_ON, null).some(item => item.kind === 'pin' || item.key === 'in-note')).toBe(false)
})

test('a pin line round-trips for each kind, ticked or not', async () => {
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
  expect(pinLine(pins[0])).toBe('- [x] [[Sprinkler Benchmarks]]')
  expect(pinLine(pins[1])).toBe('- [ ] session: `/p/a b.jsonl` · garden · fix the sprinkler')
  expect(parsePin('- [x] [[Note|alias]]')?.target).toBe('Note')
  expect(parsePin('- [ ] 2026-11-08 10:00 · a question')).toBe(null)
})

test('pins go under ## Context once each, tick off and unpin in place', async () => {
  const empty = trailTemplate('PMOC - Garden', '2026-11-08')
  const note = { on: true, kind: 'note', target: 'A', label: 'A' } as const
  const one = addPin(empty, note)
  const two = addPin(one, SESSION_PIN)
  expect(addPin(two, { ...note, on: false })).toBe(two)
  expect(two.includes('## Context\n\n- [x] [[A]]\n- [x] session: `/p/a.jsonl` · garden · fix the sprinkler\n\n## Questions')).toBe(true)
  expect(parseTrail(two).pins.map(pin => pin.target)).toEqual(['A', '/p/a.jsonl'])

  const off = setPinOn(two, note, false)
  expect(parseTrail(off).pins.map(pin => pin.on)).toEqual([false, true])
  expect(parseTrail(removePin(off, SESSION_PIN)).pins.map(pin => pin.target)).toEqual(['A'])

  // A trail from before ## Context gets the section above ## Questions.
  const old = '# Trail\n\n## Questions\n\n## Chats\n'
  const added = addPin(old, note)
  expect(added.indexOf('## Context') < added.indexOf('## Questions')).toBe(true)
  expect(parseTrail(added).pins.length).toBe(1)
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

test('an OpenAugi search answer gives its note paths once each, wrapped or not', async () => {
  const inner = JSON.stringify({ results: [{ source_path: 'N/A.md' }, { source_path: 'N/A.md' }, { source_path: 'N/B.md' }, { source_path: 'x.pdf' }] })
  expect(pathsFromSearch(inner)).toEqual(['N/A.md', 'N/B.md'])
  expect(pathsFromSearch(JSON.stringify({ result: inner }))).toEqual(['N/A.md', 'N/B.md'])
  expect(pathsFromSearch('not json')).toEqual([])
})

test('a claude session copies its resume command; codex its path', async () => {
  expect(resumeCommand({ path: '/p/-r/abc-123.jsonl', tool: 'claude', repo: 'r', title: 't', mtimeMs: 0 })).toBe('claude --resume abc-123')
  expect(resumeCommand({ path: '/c/rollout.jsonl', tool: 'codex', repo: 'r', title: 't', mtimeMs: 0 })).toBe('/c/rollout.jsonl')
})

test('a finished chat is marked done on its line and stays listed; an older (hidden) mark reads as done; a moved chat\'s line is removed', async () => {
  const text = addChat(addChat(trailTemplate('X', '2026-11-08'), { id: 's1', when: '2026-11-08 10:00', title: 'fix the gate', done: false }), { id: 's2', when: '2026-11-08 11:00', title: 'paint it', done: false })
  const done = setChatDone(text, 's1', true)
  expect(done.includes('- 2026-11-08 10:00 · fix the gate · `s1` (done)')).toBe(true)
  expect(parseTrail(done).chats.map(chat => [chat.id, chat.done])).toEqual([['s2', false], ['s1', true]])
  expect(setChatDone(done, 's1', false)).toBe(text)
  // A log written before done was the word: the chat reads as done, and is written back as done.
  const older = text.replace('`s1`', '`s1` (hidden)')
  expect(parseTrail(older).chats.find(chat => chat.id === 's1')?.done).toBe(true)
  expect(setChatDone(older, 's1', true)).toBe(done)
  // Moved to another anchor's log: only that line goes.
  expect(removeChat(text, 's1')).toBe(text.replace('- 2026-11-08 10:00 · fix the gate · `s1`\n', ''))
  expect(removeChat(text, 'nobody')).toBe(text)
  expect(parseTrail(removeChat(done, 's1')).chats.map(chat => chat.id)).toEqual(['s2'])
})

test('a chat\'s last activity reads as the time today, the month and day before, nothing when unknown', async () => {
  const now = new Date(2022, 0, 5, 15, 0).getTime()
  expect(lastActive(new Date(2022, 0, 5, 9, 7).getTime(), now)).toBe('09:07')
  expect(lastActive(new Date(2022, 0, 3, 23, 59).getTime(), now)).toBe('01-03')
  expect(lastActive(0, now)).toBe('')
})

test('a sent draft is titled by its lens and instruction; anything else is not ours', async () => {
  const lens = { name: 'echoes', path: '/v/lenses/echoes.md', description: '' }
  expect(chatTitleFrom(buildDraft([], ALL_ON, lens, 'why slow'))).toBe('echoes: why slow')
  expect(chatTitleFrom(buildDraft([], ALL_ON, null, 'just ask'))).toBe('just ask')
  expect(chatTitleFrom('hello')).toBe(null)
})

test('with an anchor, the draft carries its note and trail', async () => {
  const anchor = { note: 'PMOC - Garden', notePath: '/v/PMOC - Garden.md', trail: trailPath('PMOC - Garden') }
  const text = buildDraft([], ALL_ON, null, 'go', anchor)
  expect(text.includes('/v/PMOC - Garden.md')).toBe(true)
  expect(text.includes('Agent Log - PMOC - Garden.md')).toBe(true)
  expect(text.includes('never to my note')).toBe(true)
  expect(buildDraft([], { ...ALL_ON, project: false }, null, 'go', anchor).includes('/v/PMOC - Garden.md')).toBe(false)
})

test('pick an anchor, start a chat, send it: the chat lands on the trail and sessions stay unticked', { options: OPTIONS }, async ($, on) => {
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
  // The anchor card opens the human and agent notes. Who you are sits in the You card
  // under Always, with the season line, never behind Setup.
  expect(await ui.find({ key: 'open-note' })).toBeTruthy()
  expect(await ui.find({ key: 'open-trail' })).toBeTruthy()
  expect(await ui.find({ key: 'edit-context' })).toBeTruthy()
  expect(await ui.find({ key: 'open-board' })).toBeTruthy()
  expect(await ui.find({ text: 'Season: Primary: the garden.' })).toBeTruthy()
  await ui.press({ key: 'head-youOpen' })
  expect(await ui.find({ key: 'edit-context' })).toBeFalsy()
  await ui.press({ key: 'head-youOpen' })
  expect(await ui.find({ key: 'about-me' })).toBeFalsy()
  // What every draft carries is switched under Setup, not listed in the Context card.
  expect(await ui.find({ key: 'pick-project' })).toBeFalsy()
  expect(await ui.find({ key: 'openaugi' })).toBeFalsy()
  expect(await ui.find({ text: 'notes for PMOC - Garden, your overall context, a search of your notes' })).toBeTruthy()
  await ui.press({ key: 'setup-menu' })
  // They are real checkboxes, on by default: pressing one flips it in place.
  expect((await ui.find({ key: 'openaugi' }))?.text).toMatch(/^\s*☑/)
  expect((await ui.find({ key: 'pick-project' }))?.text).toMatch(/^\s*☑/)
  await ui.press({ key: 'pick-project' })
  expect((await ui.find({ key: 'pick-project' }))?.text).toMatch(/^\s*☐/)
  expect(await ui.find({ text: 'notes for PMOC - Garden' })).toBeFalsy()
  await ui.press({ key: 'pick-project' })
  await ui.press({ key: 'setup-menu' })
  await ui.press({ key: 'lens-menu' })
  await ui.press({ key: 'echoes' })

  expect(filled.length).toBe(1)
  expect(filled[0]?.includes('/v/PMOC - Garden.md')).toBe(true)
  expect(filled[0]?.includes('new.jsonl')).toBe(false)

  await $.prompt.submit({ text: filled[0] ?? '' })
  const trail = written.get(trailPath('PMOC - Garden')) ?? ''
  expect(parseTrail(trail).chats).toEqual([
    { id: 'own-session', when: '2026-11-08 09:30', title: 'echoes: is the sprinkler the bottleneck', done: false },
  ])
  expect(await ui.find({ key: 'chat-own-session' })).toBeTruthy()

  expect(await ui.find({ key: `in-session-${'/Users/t/.claude/projects/-r-garden/new.jsonl'}` })).toBeFalsy()

  // The pane has one place to type, the scratchpad: no question field in the Anchor card.
  expect((await ui.findAll({ type: 'Input' })).some(one => /^capture/.test(String(one.key ?? '')))).toBe(false)
  // A question an agent left in the log is listed: pressing it drafts it, done ticks it.
  written.set(trailPath('PMOC - Garden'), withQuestion(trail, '2026-11-08 09:30', 'does the cache survive restarts?'))
  await ui.press({ key: 'refresh' })
  expect(await ui.find({ text: '? does the cache survive restarts?' })).toBeTruthy()
  await ui.press({ key: 'ask-2026-11-08 09:30' })
  expect(filled[1]?.endsWith('Instruction: does the cache survive restarts?')).toBe(true)
  await ui.press({ key: 'done-2026-11-08 09:30' })
  expect(parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').questions[0]?.done).toBe(true)
  expect(await ui.find({ key: 'ask-2026-11-08 09:30' })).toBeFalsy()

  // The chat is a row under Chats: done folds it away, reopen brings it back.
  expect((await ui.find({ key: 'chat-own-session' }))?.text).toMatch(/◦ echoes: is the sprinkler the bottleneck \(this chat\) · 11-08 09:30/)
  expect(await ui.find({ text: /🌱 Chats on PMOC - Garden · 1 open/ })).toBeTruthy()
  await ui.press({ key: 'chat-done-own-session' })
  expect(parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').chats[0]?.done).toBe(true)
  expect(written.get(trailPath('PMOC - Garden'))?.includes('`own-session` (done)')).toBe(true)
  expect(await ui.find({ key: 'chat-own-session' })).toBeFalsy()
  expect(await ui.find({ text: /🌱 Chats on PMOC - Garden · 0 open/ })).toBeTruthy()
  await ui.press({ key: 'chats-done' })
  expect((await ui.find({ key: 'chat-own-session' }))?.text).toMatch(/✓ echoes: is the sprinkler the bottleneck/)
  await ui.press({ key: 'chat-reopen-own-session' })
  expect(parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').chats[0]?.done).toBe(false)
  expect(await ui.find({ key: 'chat-done-own-session' })).toBeTruthy()
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
  expect(made.includes('/v/OpenAugi/Agent-Logs')).toBe(true)
  expect(await ui.find({ key: 'set-lensFolder' })).toBeTruthy()
  await ui.press({ key: 'use-starter' })
  expect(await ui.find({ key: 'anchor-menu' })).toBeTruthy()

  await $.command.run({ command: 'augi', args: 'setup' })
  expect(store.has('skipped')).toBe(false)
  expect(await ui.find({ key: 'set-boardNote' })).toBeTruthy()
  await ui.unmount()
})

test('pin context on an anchor: paste, search, tick, copy a link, unpin, and what a pane chat publishes', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  const store = new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]]])
  const copied: string[] = []
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', () => ({ value: undefined }))
  on('ui.copy', (_, e) => {
    copied.push(e.text)
    return { value: { isCopied: true } }
  })
  on('session.id', () => ({ value: 'own-session' }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/r/docs/ARCH.md'].includes(e.path) || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => ({
    value: { exitCode: 0, stdout: e.argv[0] === 'find' && e.argv.includes('PMOC - Garden.md') ? '/v/PMOC - Garden.md\n' : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false },
  }))
  on('mcp.call', () => ({
    value: { isError: false, content: [{ type: 'text', text: JSON.stringify({ result: JSON.stringify({ results: [{ source_path: 'Notes/Sprinkler Benchmarks.md' }] }) }) }] },
  }))
  on('tool.call', () => ({ result: { url: 'https://claude.ai/artifact/abc-1' } as never, text: 'published https://claude.ai/artifact/abc-1' }))
  const filled: string[] = []
  on('prompt.read', () => ({ value: { text: 'explain the cache', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))
  const pinsNow = () => parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').pins

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })

  await ui.input({ key: 'pin-paste-0', text: '/r/docs/ARCH.md' })
  expect(pinsNow()).toEqual([{ on: true, kind: 'file', target: '/r/docs/ARCH.md', label: 'ARCH.md' }])

  await ui.input({ key: 'pin-search', text: 'sprinkler' })
  await ui.press({ key: 'offer-anchor-note-Sprinkler Benchmarks' })
  expect(pinsNow().map(pin => pin.target)).toEqual(['/r/docs/ARCH.md', 'Sprinkler Benchmarks'])
  expect(await ui.find({ key: 'offer-anchor-note-Sprinkler Benchmarks' })).toBeFalsy()
  // A pin row has two actions, open and unpin, and the card says what the tick means.
  expect(await ui.find({ key: 'pin-link-note-Sprinkler Benchmarks' })).toBeFalsy()
  expect(await ui.find({ key: 'pin-all-note-Sprinkler Benchmarks' })).toBeFalsy()
  expect((await ui.find({ key: 'pin-drop-note-Sprinkler Benchmarks' }))?.text).toMatch(/unpin/)
  expect(await ui.find({ key: 'pin-open-note-Sprinkler Benchmarks' })).toBeTruthy()
  expect(await ui.find({ text: /☑ goes into this anchor's drafts · ☐ is a link only/ })).toBeTruthy()

  await ui.press({ key: 'pin-tick-file-/r/docs/ARCH.md' })
  expect(pinsNow()[0]?.on).toBe(false)
  expect((await ui.find({ key: 'pin-tick-file-/r/docs/ARCH.md' }))?.text).toMatch(/^\s*☐/)
  await ui.press({ key: 'pin-drop-file-/r/docs/ARCH.md' })
  expect(pinsNow().map(pin => pin.target)).toEqual(['Sprinkler Benchmarks'])

  // An anchored chat's published artifact pins ticked; a read of another does not pin.
  await ui.press({ key: 'draft' })
  expect(filled[0]?.includes('- Note: "/v/Notes/Sprinkler Benchmarks.md"')).toBe(true)
  await $.prompt.submit({ text: filled[0] ?? '' })
  await $.tool.call({ tool: 'Artifact', file_path: '/s/page.html', title: 'Cache explainer' })
  await $.tool.call({ tool: 'Artifact', action: 'read', url: 'https://claude.ai/artifact/zzz-9' })
  expect(pinsNow().slice(1)).toEqual([{ on: true, kind: 'artifact', target: 'https://claude.ai/artifact/abc-1', label: 'Cache explainer' }])
  // Artifacts sort first in the context card.
  expect((await ui.find({ key: 'pin-tick-artifact-https://claude.ai/artifact/abc-1' }))).toBeTruthy()
  // A new note it writes in the vault pins too; a file outside the vault or an edit of one there does not.
  await $.tool.call({ tool: 'Write', file_path: '/v/Ideas/New Idea.md', content: 'x' })
  await $.tool.call({ tool: 'Write', file_path: '/tmp/scratch.md', content: 'x' })
  await $.tool.call({ tool: 'Write', file_path: '/v/AGENT/Slowly Changing Context.md', content: 'x' })
  expect(pinsNow().slice(2)).toEqual([{ on: false, kind: 'note', target: 'New Idea', label: 'New Idea' }])

  // Pins for every chat live in the store, not on the anchor: pasted in the Context card.
  // One pinned in both lists goes into the draft once; they untick and unpin in place.
  await ui.input({ key: 'all-paste-0', text: '/r/docs/ARCH.md' })
  expect(store.get('pins')).toEqual([{ on: true, kind: 'file', target: '/r/docs/ARCH.md', label: 'ARCH.md' }])
  expect(pinsNow().some(pin => pin.target === '/r/docs/ARCH.md')).toBe(false)
  await ui.input({ key: 'all-paste-1', text: '[[Sprinkler Benchmarks]]' })
  expect(pinsNow().some(pin => pin.target === 'Sprinkler Benchmarks')).toBe(true)
  expect(storedPins(store.get('pins')).map(pin => pin.target)).toEqual(['/r/docs/ARCH.md', 'Sprinkler Benchmarks'])
  await ui.press({ key: 'draft' })
  const last = filled[filled.length - 1] ?? ''
  expect(last.split('ARCH.md').length).toBe(2)
  expect(last.split('Sprinkler Benchmarks').length).toBe(2)
  await ui.press({ key: 'all-tick-file-/r/docs/ARCH.md' })
  expect(storedPins(store.get('pins'))[0]?.on).toBe(false)
  await ui.press({ key: 'all-drop-file-/r/docs/ARCH.md' })
  expect(storedPins(store.get('pins')).map(pin => pin.target)).toEqual(['Sprinkler Benchmarks'])

  // Every card and both sections fold from their title, and open again.
  for (const [head, inside] of [['head-pinsOpen', 'pin-paste-3'], ['head-contextOpen', 'pin-search'], ['head-anchorOpen', 'anchor-menu'], ['head-padOpen', 'pad-input-0'], ['head-trailOpen', 'chat-own-session']] as const) {
    expect(await ui.find({ key: inside })).toBeTruthy()
    await ui.press({ key: head })
    expect(await ui.find({ key: inside })).toBeFalsy()
    await ui.press({ key: head })
    expect(await ui.find({ key: inside })).toBeTruthy()
  }
  await ui.press({ key: 'head-chatOpen' })
  expect(await ui.find({ key: 'head-anchorOpen' })).toBeFalsy()
  expect(await ui.find({ key: 'pin-search' })).toBeTruthy()
  await ui.press({ key: 'head-anywhereOpen' })
  expect(await ui.find({ key: 'pin-search' })).toBeFalsy()
  expect(await ui.find({ key: 'draft' })).toBeFalsy()
  expect(await ui.find({ key: 'refresh' })).toBeTruthy()
  await ui.unmount()
})

test('pins for every chat: the store is read defensively, and a draft adds the ticked ones once, with or without an anchor', () => {
  const arch: Pin = { on: true, kind: 'file', target: '/r/docs/ARCH.md', label: 'ARCH.md' }
  const off: Pin = { on: false, kind: 'url', target: 'https://example.com', label: 'example' }
  expect(storedPins([arch, off, { on: 'yes', kind: 'file', target: 'x', label: 'x' }, { on: true, kind: 'mood', target: 'x', label: 'x' }, null, 'x'])).toEqual([arch, off])
  expect(storedPins(undefined)).toEqual([])
  expect(extraPins([], [arch, off], false)).toEqual([arch])
  expect(extraPins([arch], [arch], true)).toEqual([])
  expect(extraPins([{ ...arch, on: false }], [arch], true)).toEqual([arch])
  const none = { aboutMe: false, openAugi: false, project: false, today: false }
  expect(buildDraft([], none, null, '', null, [arch, off]).split('ARCH.md').length).toBe(2)
  expect(buildDraft([], none, null, '', null, [arch, off]).includes('example.com')).toBe(false)
  expect(contextItems([], none, null, [arch, off]).map(item => item.key)).toEqual(['in-all-file-/r/docs/ARCH.md'])
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

test('the Agent Log is named for its note and links it first, with no title or seen box', async () => {
  expect(logName('PMOC - A/B')).toBe('Agent Log - PMOC - A-B')
  const log = trailTemplate('PMOC - Garden', '2026-11-08')
  expect(log.includes('# Agent Log - PMOC - Garden')).toBe(false)
  expect(log.includes('type: document\n')).toBe(true)
  expect(log.includes('created: 2026-11-08\n---\n\nHuman note: [[PMOC - Garden]]. Context lines')).toBe(true)
})

const OLD_INTRO = 'Project: [[PMOC - Garden]], embedded there under # Agent Log. Context lines are `- [x] [[note]] · why it matters` (ticked goes into the next draft). Hidden chats stay listed, marked `(hidden)`.'
// The intro the log had while a finished chat was called hidden.
const HIDDEN_INTRO = 'Human note: [[PMOC - Garden]]. Context lines are `- [x] [[note]] · why it matters` (ticked goes into the next draft). Hidden chats stay listed, marked `(hidden)`.'

test('an older Agent Log loses its title and seen box, gains the note link, and keeps its sections', async () => {
  const sections = ['## Context', '', '- [x] [[A note]] · why', '', '## Questions', '', '- [x] 2022-01-04 10:00 · q', '', '## Chats', '', '## Saved', '']
  const old = ['---', 'type: document', 'description: d', 'created: 2022-01-04', '---', '', '# Agent Log - PMOC - Garden', '', '- [x] seen', '', OLD_INTRO, '', ...sections].join('\n')
  const tidy = tidyTrail(old, 'PMOC - Garden')
  expect(tidy).toBe(trailTemplate('PMOC - Garden', '2022-01-04').replace(/description: .*/, 'description: d').split('## Context')[0] + sections.join('\n'))
  expect(tidyTrail(tidy, 'PMOC - Garden')).toBe(tidy)
  expect(parseTrail(tidy)).toEqual(parseTrail(old))
  expect(tidyTrail(trailTemplate('PMOC - Garden', '2022-01-04'), 'PMOC - Garden')).toBe(trailTemplate('PMOC - Garden', '2022-01-04'))
  // A log from the version before this one takes the current intro, which says done, and nothing else changes.
  const current = trailTemplate('PMOC - Garden', '2022-01-04')
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
  expect(await ui.find({ text: /Lenses \(saved prompts\) · 1/ })).toBeTruthy()
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
  const text = addChat(trailTemplate('X', '2026-11-08'), { id: 's1', when: '2026-11-08 10:00', title: 'chat', done: false })
  const named = retitleChat(text, 's1', 'echoes: why slow')
  expect(parseTrail(named).chats[0]?.title).toBe('echoes: why slow')
  expect(retitleChat(named, 's1', 'later')).toBe(named)
  expect(retitleChat(text, 's2', 'other')).toBe(text)
  const hidden = retitleChat(setChatDone(text, 's1', true), 's1', 'named')
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

test('artifacts: a republished url takes its new title; the pin list puts them first', async () => {
  const a = { url: 'https://claude.ai/artifact/a', title: 'A' }
  expect(withArtifact(withArtifact([a], { url: 'https://claude.ai/artifact/b', title: 'B' }), { ...a, title: 'A2' }).map(one => one.title)).toEqual(['A2', 'B'])
  const art = (n: number) => ({ on: true, kind: 'artifact', target: `https://claude.ai/artifact/${n}`, label: String(n) }) as const
  const note = { on: true, kind: 'note', target: 'N', label: 'N' } as const
  expect(artifactsFirst([note, art(1), SESSION_PIN, art(2)]).map(pin => pin.label)).toEqual(['1', '2', 'N', SESSION_PIN.label])
})

test('a new session opens with no anchor; anchoring logs the chat, pins its artifacts ticked, and a sent draft names the chat', { options: OPTIONS }, async ($, on) => {
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
  const logNow = () => parseTrail(written.get(trailPath('PMOC - Garden')) ?? '')

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(await ui.find({ key: 'open-trail' })).toBeFalsy()
  expect(await ui.find({ key: 'add-this-chat' })).toBeFalsy()

  // Published before anchoring: remembered and shown dimly, written nowhere.
  await $.tool.call({ tool: 'Artifact', file_path: '/s/early.html', title: 'Early plan' })
  expect(written.size).toBe(0)
  expect(await ui.find({ text: '◇ Early plan' })).toBeTruthy()

  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-PMOC - Garden' })
  expect(store.get('anchors')).toEqual([['other-session', 'PMOC - Garden'], ['new-session', 'PMOC - Garden']])
  expect(logNow().chats).toEqual([{ id: 'new-session', when: '2026-11-08 09:30', title: 'chat', done: false }])
  expect(logNow().pins).toEqual([{ on: true, kind: 'artifact', target: 'https://claude.ai/artifact/early-1', label: 'Early plan' }])
  // The Anchor card does not repeat the pins: they are opened from the Pins card below it.
  expect(await ui.find({ key: 'anchor-artifact-https://claude.ai/artifact/early-1' })).toBeFalsy()
  await ui.press({ key: 'pin-open-artifact-https://claude.ai/artifact/early-1' })
  expect(opened).toEqual([['open', 'https://claude.ai/artifact/early-1']])

  // Published after anchoring: pinned ticked.
  url = 'https://claude.ai/artifact/late-2'
  await $.tool.call({ tool: 'Artifact', file_path: '/s/late.html', title: 'Late plan' })
  expect(logNow().pins.map(pin => [pin.target, pin.on])).toEqual([['https://claude.ai/artifact/early-1', true], ['https://claude.ai/artifact/late-2', true]])
  expect((await ui.find({ key: 'pin-tick-artifact-https://claude.ai/artifact/late-2' }))?.text).toMatch(/◇ Late plan/)

  // A draft sent later names the chat; the ticked artifacts go in it.
  await ui.press({ key: 'draft' })
  expect(filled[0]?.includes('https://claude.ai/artifact/late-2')).toBe(true)
  await $.prompt.submit({ text: filled[0] ?? '' })
  expect(logNow().chats.map(chat => chat.title)).toEqual(['explain the cache'])

  // Another session opens with no anchor; this one's comes back when the session does.
  session = 'third-session'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(await ui.find({ text: '◇ Early plan' })).toBeFalsy()
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
  // → ask drafts an entry as the instruction, unsent; the entry stays on the pad.
  await ui.press({ key: 'pad-ask-1' })
  expect(filled.length).toBe(1)
  expect(filled[0]?.endsWith('Instruction: check the hose')).toBe(true)
  expect((await ui.find({ key: 'pad-row-1' }))?.text).toMatch(/check the hose/)
  await ui.press({ key: 'pad-drop-0' })
  expect((await ui.find({ key: 'pad-row-0' }))?.text).toMatch(/check the hose/)
  expect(await ui.find({ key: 'pad-row-1' })).toBeFalsy()
  expect(written.size).toBe(0)
  await ui.unmount()
})

test('with an anchor, → notes puts the entry in the note\'s Journal and takes it off the pad', { options: OPTIONS }, async ($, on) => {
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
  const got = { written, store, transcript, filled: [] as string[], copied: [] as string[], ran: [] as string[][], toasts: [] as string[] }
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
  on('prompt.read', () => ({ value: { text: 'half a thought', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    got.filled.push(e.text)
    return { isFilled: true }
  })
  on('prompt.submit', (_, e) => ({ text: e.text }))

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
  expect((await ui.find({ key: 'task-sub' }))?.text).toMatch(/sub-task of PMOC - Garden/)
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
  expect(parseTrail(got.written.get(trailPath('Fix the gate')) ?? '').pins).toEqual([
    { on: true, kind: 'session', target: got.transcript, label: 'the chat this task came from' },
  ])
  expect(anchorOf(got.store, 'own-session')).toBe('PMOC - Garden')
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  expect(got.filled).toEqual([])
  expect(await ui.find({ key: 'task-title-new' })).toBeFalsy()
  expect(got.toasts.some(text => text.includes('/v/Fix the gate.md') && text.includes('Setup → new task folder'))).toBe(true)

  // The just-made row says where on the board it went and opens the board.
  expect(await ui.find({ text: '✓ just made: Fix the gate' })).toBeTruthy()
  expect(await ui.find({ text: 'on the board: a card at the end of Work' })).toBeTruthy()
  expect(got.toasts.some(text => text.includes('a card at the end of Work'))).toBe(true)
  await ui.press({ key: 'made-board' })
  expect(got.ran.some(argv => argv[0] === 'open' && argv[1] === `obsidian://open?path=${encodeURIComponent('/v/Kanban.md')}`)).toBe(true)
  await ui.press({ key: 'made-open' })
  expect(got.ran.some(argv => argv[0] === 'open' && argv[1] === `obsidian://open?path=${encodeURIComponent('/v/Fix the gate.md')}`)).toBe(true)
  await ui.press({ key: 'made-define' })
  expect(got.filled).toEqual([newTaskDraft('/v/Fix the gate.md', trailPath('Fix the gate'), 'Fix the gate')])
  // The half-typed prompt went to the clipboard first.
  expect(got.copied).toEqual(['half a thought'])
  await ui.press({ key: 'made-copy' })
  expect(got.filled.length).toBe(1)
  expect(anchorFromDraft(got.copied[1] ?? '')).toBe('Fix the gate')
  expect(got.copied[1]?.includes('"/v/Fix the gate.md"')).toBe(true)
  expect(got.copied[1]?.includes(got.transcript)).toBe(true)
  await ui.press({ key: 'made-anchor' })
  expect(anchorOf(got.store, 'own-session')).toBe('Fix the gate')
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/Fix the gate/)
  expect(await ui.find({ key: 'made-anchor' })).toBeFalsy()
  await ui.press({ key: 'made-dismiss' })
  expect(await ui.find({ text: '✓ just made: Fix the gate' })).toBeFalsy()

  // The same title again writes nothing.
  await ui.press({ key: 'task-new' })
  await ui.input({ key: 'task-title-new', text: 'Fix the gate' })
  expect(got.toasts.some(text => text.includes('already there'))).toBe(true)
  expect(got.written.get('/v/Kanban.md')).toBe(board)
  await ui.unmount()
})

test('＋ sub-task makes a note Up the anchor, a linked sub-item under its card, the anchor pinned in its log; the anchor stays', { options: OPTIONS }, async ($, on) => {
  const got = taskHooks(on, [['own-session', 'PMOC - Garden']], { '/v/PMOC - Garden.md': GARDEN_PLAIN })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'task-sub' })
  await ui.press({ key: 'task-cancel' })
  expect(got.written.size).toBe(0)
  await ui.press({ key: 'task-sub' })
  await ui.input({ key: 'task-title-sub', text: 'Fix the gate' })
  expect(got.written.get('/v/Fix the gate.md')?.startsWith('Up: [[PMOC - Garden]]\n')).toBe(true)
  const item = '\t- [ ] [[Fix the gate]]'
  const board = got.written.get('/v/Kanban.md') ?? ''
  expect(board.split('\n').indexOf(item)).toBe(BOARD_TEXT.split('\n').indexOf('\t- [ ] [[PMOC - Birdhouse]]') + 1)
  expect(without(board, item)).toBe(BOARD_TEXT)
  expect(parseTrail(got.written.get(trailPath('Fix the gate')) ?? '').pins).toEqual([
    { on: true, kind: 'note', target: 'PMOC - Garden', label: 'PMOC - Garden' },
    { on: true, kind: 'session', target: got.transcript, label: 'the chat this task came from' },
  ])
  expect(anchorOf(got.store, 'own-session')).toBe('PMOC - Garden')
  expect(got.filled).toEqual([])
  expect(got.toasts.some(text => text.includes("under PMOC - Garden's card"))).toBe(true)
  // The board was read back: the new sub-item is among the anchor's open tasks.
  await ui.press({ key: 'anchor-details' })
  expect(await ui.find({ key: 'task-Fix the gate' })).toBeTruthy()
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
  expect(parseTrail(got.written.get(trailPath('PMOC - Birdhouse')) ?? '').chats.map(chat => chat.id)).toEqual(['own-session'])

  // Not on the board, but a note of that name exists: a bare anchor.
  session.id = 'loose'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/Loose note/)
  expect(await ui.find({ text: '🎯 It is tidy.' })).toBeTruthy()
  expect(await ui.find({ key: 'open-note' })).toBeTruthy()
  // A sub-task of an anchor off the board: Up the anchor, its own card at the end of the first column, and the toast says so.
  await ui.press({ key: 'task-sub' })
  await ui.input({ key: 'task-title-sub', text: 'Sweep up' })
  expect(got.written.get('/v/Sweep up.md')?.startsWith('Up: [[Loose note]]\n')).toBe(true)
  expect(without(got.written.get('/v/Kanban.md') ?? '', '- [ ] [[Sweep up]]')).toBe(BOARD_TEXT)
  expect(got.toasts.some(text => text.includes('Loose note is not on the board'))).toBe(true)
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
  const text = jiraUpdateDraft('/v/PMOC - Garden.md', 'PMOC - Garden', '/v/Notes/Agent Log - PMOC - Garden.md', 'PROJ-1', { project: 'PROJ', server: 'atlassian' })
  expect(text.startsWith('Context for this chat:\n')).toBe(true)
  for (const part of ['Outcome, Verify and Journal sections', 'Agent Log "/v/Notes/Agent Log - PMOC - Garden.md"', 'Read Jira issue PROJ-1 and its recent comments using the atlassian MCP server', 'only what is new since the last one', 'at most six lines: what moved, what is next, what is blocked', 'Show it to me first', 'comment on PROJ-1 only after I say yes', 'status only if I ask']) {
    expect(text.includes(part)).toBe(true)
  }
  expect(text.endsWith('\nInstruction: update Jira PROJ-1')).toBe(true)
  const bare = jiraUpdateDraft('/v/PMOC - Garden.md', 'PMOC - Garden', '/v/x.md', 'PROJ-1', { project: '', server: '' })
  expect(bare.includes('with whatever Jira tools you have; if you have none, say so and stop')).toBe(true)
})

test('a new-task draft looks for the Jira issue only when asked', async () => {
  const plain = newTaskDraft('/v/Fix the gate.md', '/v/Notes/Agent Log - Fix the gate.md', 'Fix the gate')
  expect(plain.includes('Jira')).toBe(false)
  const asked = newTaskDraft('/v/Fix the gate.md', '/v/Notes/Agent Log - Fix the gate.md', 'Fix the gate', { project: 'PROJ', server: '' })
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
    expect(got.filled[1]).toBe(jiraUpdateDraft(GARDEN_IN_FOLDER, 'PMOC - Garden', trailPath('PMOC - Garden'), 'PROJ-42', { project: 'PROJ', server: '' }))

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
  await ui.press({ key: 'task-sub' })
  await ui.input({ key: 'task-title-sub', text: 'Fix the gate' })
  expect(got.written.has('/v/Tasks/Fix the gate.md')).toBe(true)
  expect(got.written.get('/v/Tasks/Fix the gate.md')?.includes('Up: [[PMOC - Garden]]')).toBe(true)
  expect(got.toasts.some(text => text.includes('Setup → new task folder'))).toBe(false)
  expect(got.filled).toEqual([])
  await ui.press({ key: 'made-define' })
  expect(got.filled).toEqual([newTaskDraft('/v/Tasks/Fix the gate.md', trailPath('Fix the gate'), 'Fix the gate', { project: 'PROJ', server: '' })])
  await ui.unmount()
})

test('with no task folder, a new task goes to the notes root, never beside the anchor note', { options: OPTIONS }, async ($, on) => {
  const got = jiraHooks(on, GARDEN_NOTE, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'task-sub' })
  await ui.input({ key: 'task-title-sub', text: 'Fix the gate' })
  expect(got.written.has('/v/Fix the gate.md')).toBe(true)
  expect(got.written.has('/v/Projects/Fix the gate.md')).toBe(false)
  await ui.press({ key: 'made-define' })
  expect(got.filled).toEqual([newTaskDraft('/v/Fix the gate.md', trailPath('Fix the gate'), 'Fix the gate')])
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

test('a task template is the built-in one, the notes folder\'s own, or its own from before the Agent Log left the note', async () => {
  expect(taskTemplateState('')).toBe('built-in')
  expect(taskTemplateState('  \n')).toBe('built-in')
  expect(taskTemplateState(TASK_TEMPLATE)).toBe('yours')
  expect(taskTemplateState('Up: [[PMOC - ]]\n\n### Outcome\n\nMention of the Agent Log in a sentence.\n')).toBe('yours')
  expect(taskTemplateState(OLD_TASK_TEMPLATE)).toBe('old')
  expect(taskTemplateState('### Outcome\n\n## agent log\n')).toBe('old')
  expect(taskTemplateState('### Outcome\n\n![[Agent Log - {{title}}]]\n')).toBe('old')
})

test('an old task template in the notes folder is what new tasks get, Setup says so, and one press replaces it keeping the old one', { options: OPTIONS }, async ($, on) => {
  const template = '/v/OpenAugi/AGENT/templates/Task.md'
  const got = taskHooks(on, [['own-session', 'PMOC - Garden']], { '/v/PMOC - Garden.md': GARDEN_PLAIN, [template]: OLD_TASK_TEMPLATE })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  // It counts as a setup item that needs the user, shown before Setup is opened.
  expect(await ui.find({ text: '1 setup item needs you: open setup below.' })).toBeTruthy()
  // Until they act, the notes folder's template is theirs to keep: a new task is made from it.
  await ui.press({ key: 'task-sub' })
  await ui.input({ key: 'task-title-sub', text: 'Old shape' })
  expect(got.written.get('/v/Old shape.md')?.includes('# Agent Log')).toBe(true)
  expect(got.written.get('/v/Old shape.md')?.includes('![[Agent Log - Old shape]]')).toBe(true)

  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: /✗ task template: yours still has an Agent Log section/ })).toBeTruthy()
  await ui.press({ key: 'template-update' })
  expect(got.written.get(template)).toBe(TASK_TEMPLATE)
  expect(got.written.get(`${template}.2026-11-08.bak`)).toBe(OLD_TASK_TEMPLATE)
  expect(await ui.find({ key: 'template-update' })).toBeFalsy()
  expect(await ui.find({ text: '✓ task template: OpenAugi/AGENT/templates/Task.md' })).toBeTruthy()
  expect(await ui.find({ text: '1 setup item needs you: open setup below.' })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })

  await ui.press({ key: 'task-sub' })
  await ui.input({ key: 'task-title-sub', text: 'New shape' })
  const made = got.written.get('/v/New shape.md') ?? ''
  expect(made.includes('Agent Log')).toBe(false)
  expect(made.startsWith('Up: [[PMOC - Garden]]\n\n### Outcome\n')).toBe(true)
  // A refresh reads the template again: still fine.
  await ui.press({ key: 'refresh' })
  expect(await ui.find({ text: '1 setup item needs you: open setup below.' })).toBeFalsy()
  await ui.unmount()
})

test('Setup names the built-in task template when the notes folder has none, and warns while no task folder is set', { options: OPTIONS }, async ($, on) => {
  taskHooks(on, [], {})
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ text: /setup items? needs? you/ })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: '✓ task template: built-in (Outcome, Verify, Journal)' })).toBeTruthy()
  expect(await ui.find({ key: 'template-update' })).toBeFalsy()
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
  const withToday = buildDraft([], ALL_ON, null, 'go', ANCHOR, [], '/v/2022-01-05.md')
  expect(withToday.includes('- Today: read "/v/2022-01-05.md"')).toBe(true)
  expect(buildDraft([], { ...ALL_ON, today: false }, null, 'go', ANCHOR, [], '/v/2022-01-05.md').includes('Today:')).toBe(false)
  expect(buildDraft([], ALL_ON, null, 'go', ANCHOR, [], '').includes('Today:')).toBe(false)
  expect(contextItems([], ALL_ON, null, [], '/v/2022-01-05.md').map(item => item.key)).toContain('in-today')
  expect(contextItems([], { ...ALL_ON, today: false }, null, [], '/v/2022-01-05.md').map(item => item.key)).not.toContain('in-today')
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
  await ui.input({ key: 'pad-input-0', text: 'a thought' })
  expect(await ui.find({ key: 'pad-today-0' })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ text: 'No daily note: nothing about today is shown.' })).toBeTruthy()
  await ui.unmount()
})

test('today\'s note in the You card: found where Obsidian puts new notes, it lists what is open, logs a line under the heading, takes scratchpad entries, and goes into drafts until unticked', { options: { ...OPTIONS, dailyHeading: '## Log' } }, async ($, on) => {
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

  // A scratchpad entry goes there too, and comes off the pad.
  await ui.input({ key: 'pad-input-0', text: 'ordered the hinges' })
  await ui.press({ key: 'pad-today-0' })
  expect(written.get(today)?.endsWith('- 09:30 gate is fixed\n- 09:30 ordered the hinges\n')).toBe(true)
  expect(await ui.find({ key: 'pad-row-0' })).toBeFalsy()
  expect(toasts[1]).toBe("Saved to today's note")

  // It goes into every draft, named in the Context card's line, until unticked.
  expect(await ui.find({ text: /your overall context, today's note, a search of your notes/ })).toBeTruthy()
  await ui.press({ key: 'draft' })
  expect(got.filled[0]?.includes(`- Today: read "${today}"`)).toBe(true)
  expect((await ui.find({ key: 'pick-today' }))?.text).toMatch(/☑ today's note goes into every draft/)
  await ui.press({ key: 'pick-today' })
  expect((await ui.find({ key: 'pick-today' }))?.text).toMatch(/☐/)
  await ui.press({ key: 'draft' })
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
  await ui.input({ key: 'pad-input-0', text: 'a thought' })
  expect(await ui.find({ key: 'pad-today-0' })).toBeFalsy()
  await ui.press({ key: 'draft' })
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
  expect(await ui.find({ key: 'pad-today-0' })).toBeTruthy()

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

test('an anchor is remembered per working folder, a worktree counted as its repository; a chosen none is kept', async () => {
  expect(folderKey('/r/garden')).toBe('/r/garden')
  expect(folderKey('/r/garden/')).toBe('/r/garden')
  expect(folderKey('/r/garden/.claude/worktrees/fix-gate')).toBe('/r/garden')
  expect(folderKey('/r/garden/.claude/worktrees/fix-gate/src')).toBe('/r/garden/.claude/worktrees/fix-gate/src')
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

// A vault whose folder /r/garden last worked on PMOC - Garden. `state` can be changed
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

test('a new chat starts on the anchor last chosen in its folder, is filed there with its first prompt, and keep, change or no anchor is one press', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  // No picking: the pane is on the folder's anchor, and says where it came from.
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  expect(await ui.find({ text: /From your last chat in this folder/ })).toBeTruthy()
  expect(anchorOf(got.store, 'fresh')).toBe('PMOC - Garden')
  // Nothing is written to the anchor's log for a chat that has not said anything.
  expect(got.written.size).toBe(0)

  // A slash command is not the chat saying something.
  await $.prompt.submit({ text: '/augi usage' })
  expect(got.written.size).toBe(0)
  // Its first prompt files it under that prompt's first line, and the note about where it came from goes.
  await $.prompt.submit({ text: 'fix the sprinkler valve\nit leaks at the joint' })
  expect(parseTrail(got.written.get(trailPath('PMOC - Garden')) ?? '').chats).toEqual([{ id: 'fresh', when: '2026-11-08 09:30', title: 'fix the sprinkler valve', done: false }])
  expect(await ui.find({ text: /From your last chat in this folder/ })).toBeFalsy()
  expect(await ui.find({ key: 'chat-fresh' })).toBeTruthy()
  // Later prompts do not rename it, and do not read the log again to find that out.
  const log = got.written.get(trailPath('PMOC - Garden'))
  got.written.set(trailPath('PMOC - Garden'), `${log}\n<!-- an edit made elsewhere -->\n`)
  await $.prompt.submit({ text: 'and the timer too' })
  expect(got.written.get(trailPath('PMOC - Garden'))).toBe(`${log}\n<!-- an edit made elsewhere -->\n`)
  await ui.unmount()
})

test('keep confirms a carried-over anchor; change opens the menu; no anchor sticks for the session', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'inherit-keep' })
  expect(await ui.find({ text: /From your last chat in this folder/ })).toBeFalsy()
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  await ui.unmount()

  // Another new chat in the folder: change opens the anchor list, and picking one is a choice.
  got.state.id = 'second'
  await $.command.run({ command: 'augi', args: '' })
  const two = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await two.find({ text: /From your last chat in this folder/ })).toBeTruthy()
  expect(await two.find({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' })).toBeFalsy()
  await two.press({ key: 'inherit-change' })
  await two.press({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' })
  expect(anchorOf(got.store, 'second')).toBe('PMOC - Birdhouse')
  expect(await two.find({ text: /From your last chat in this folder/ })).toBeFalsy()
  // The choice is what the next new chat here starts on.
  expect(got.store.get('folders')).toEqual([['/r/garden', 'PMOC - Birdhouse']])
  await two.unmount()

  // A third: no anchor, by choice. A refresh does not bring one back, and the folder keeps its last.
  got.state.id = 'third'
  await $.command.run({ command: 'augi', args: '' })
  const three = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Birdhouse/)
  await three.press({ key: 'inherit-none' })
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await three.press({ key: 'refresh' })
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await $.prompt.submit({ text: 'an unanchored question' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect((await three.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  expect(got.store.get('folders')).toEqual([['/r/garden', 'PMOC - Birdhouse']])
  // Nothing was filed anywhere for the chats that only looked.
  expect([...got.written.keys()].filter(path => path.includes('Agent Log - PMOC - Garden'))).toEqual([])
  await three.unmount()
})

test('no anchor is carried over to an older chat, a chat in another folder, or one that already chose; a worktree counts as its repository', { options: OPTIONS }, async ($, on) => {
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
  expect(await ui.find({ text: /From your last chat in this folder/ })).toBeFalsy()
  await ui.unmount()

  // A new chat in a worktree of the repository starts on the repository's anchor.
  got.state.id = 'tree'
  got.state.cwd = '/r/garden/.claude/worktrees/fix-gate'
  await $.command.run({ command: 'augi', args: '' })
  ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  await ui.unmount()
})

test('no anchor is carried over to a headless run, or onto a note that is gone; a session resumed in this process shows its own stored anchor', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [['/r/garden', 'PMOC - Garden'], ['/r/pond', 'Renamed since']], [['earlier', 'PMOC - Birdhouse']])
  on('session.end', (_, e) => ({ sessionId: e.sessionId }) as never)
  // A run that draws nowhere: nobody could see, keep or change the anchor.
  got.state.surfaces = []
  await $.command.run({ command: 'augi', args: '' })
  expect(anchorOf(got.store, 'fresh')).toBe(undefined)
  // The folder's anchor names a note that is on no board and in no folder now.
  got.state.surfaces = ['terminal']
  got.state.id = 'pond'
  got.state.cwd = '/r/pond'
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
  expect(await ui.find({ text: /From your last chat in this folder/ })).toBeFalsy()
  await ui.unmount()
})

test('a → task field left open closes when its chat leaves the log or the anchor changes: the buttons are never lost', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [], [['parent', 'PMOC - Garden']])
  got.state.id = 'parent'
  const log = addChat(trailTemplate('PMOC - Garden', '2026-11-08'), { id: 'child-1', when: '2026-11-08 09:10', title: 'wire the sprinkler timer', done: false })
  got.written.set(trailPath('PMOC - Garden'), log)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'chat-task-child-1' })
  expect(await ui.find({ key: 'task-title-promote' })).toBeTruthy()
  expect(await ui.find({ key: 'task-new' })).toBeFalsy()
  // Another chat made it a task meanwhile (its line left this log): the field goes, the buttons return.
  got.written.set(trailPath('PMOC - Garden'), removeChat(log, 'child-1'))
  await ui.press({ key: 'refresh' })
  expect(await ui.find({ key: 'task-title-promote' })).toBeFalsy()
  expect(await ui.find({ key: 'task-new' })).toBeTruthy()
  expect(await ui.find({ key: 'task-child' })).toBeTruthy()

  // Open again, then pick another anchor: the field was for the anchor it left.
  got.written.set(trailPath('PMOC - Garden'), log)
  await ui.press({ key: 'refresh' })
  await ui.press({ key: 'chat-task-child-1' })
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-sub-PMOC - Garden-PMOC - Birdhouse' })
  expect(await ui.find({ key: 'task-title-promote' })).toBeFalsy()
  expect(await ui.find({ key: 'task-new' })).toBeTruthy()
  // And with a field open, no anchor leaves only what needs none.
  await ui.press({ key: 'task-sub' })
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'anchor-clear' })
  expect(await ui.find({ key: 'task-title-sub' })).toBeFalsy()
  expect(await ui.find({ key: 'task-new' })).toBeTruthy()
  await ui.unmount()
})

test('a draft copied for a new chat wins over an anchor the chat only started on by itself', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(anchorOf(got.store, 'fresh')).toBe('PMOC - Garden')
  const pasted = buildDraft([], ALL_ON, null, 'hang the birdhouse', { note: 'PMOC - Birdhouse', notePath: '/v/PMOC - Birdhouse.md', trail: trailPath('PMOC - Birdhouse') })
  await $.prompt.submit({ text: pasted })
  expect(anchorOf(got.store, 'fresh')).toBe('PMOC - Birdhouse')
  expect(parseTrail(got.written.get(trailPath('PMOC - Birdhouse')) ?? '').chats.map(chat => chat.title)).toEqual(['hang the birdhouse'])
  // The anchor it started on was never written to.
  expect(got.written.has(trailPath('PMOC - Garden'))).toBe(false)
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Birdhouse/)
  await ui.unmount()
})

test('after /clear the pane is the new session\'s: no anchor at first, then the folder\'s on the next sync', { options: OPTIONS }, async ($, on) => {
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
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  expect(await ui.find({ text: /From your last chat in this folder/ })).toBeTruthy()
  expect(anchorOf(got.store, 'cleared')).toBe('PMOC - Garden')
  await ui.unmount()
})

test('a chat on an anchor it chose is titled by its first prompt too, replacing the placeholder once', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-PMOC - Garden' })
  const chats = () => parseTrail(got.written.get(trailPath('PMOC - Garden')) ?? '').chats.map(chat => chat.title)
  expect(chats()).toEqual(['chat'])
  expect(got.store.get('folders')).toEqual([['/r/garden', 'PMOC - Garden']])
  await $.prompt.submit({ text: 'plan the beds for spring' })
  expect(chats()).toEqual(['plan the beds for spring'])
  await $.prompt.submit({ text: 'something else entirely' })
  expect(chats()).toEqual(['plan the beds for spring'])
  await ui.unmount()
})

test('＋ child chat copies a draft with its goal and writes nothing; sent in a new chat, that chat is listed on the parent with the goal, its last activity, and done', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [], [['parent', 'PMOC - Garden']])
  got.state.id = 'parent'
  const { seen } = got
  const child = '/Users/t/.claude/projects/-r-garden/child-1.jsonl'
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect(await ui.find({ text: /No open chats/ })).toBeTruthy()

  // The goal field opens on a press; cancel and an empty goal copy nothing.
  expect(await ui.find({ key: 'task-title-child' })).toBeFalsy()
  await ui.press({ key: 'task-child' })
  await ui.press({ key: 'task-cancel' })
  await ui.press({ key: 'task-child' })
  await ui.input({ key: 'task-title-child', text: '   ' })
  expect(got.copied).toEqual([])
  await ui.press({ key: 'task-child' })
  await ui.input({ key: 'task-title-child', text: '  wire the  sprinkler timer ' })
  expect(got.copied.length).toBe(1)
  const draft = got.copied[0] ?? ''
  expect(anchorFromDraft(draft)).toBe('PMOC - Garden')
  expect(draft.endsWith('Instruction: wire the sprinkler timer')).toBe(true)
  expect(draft.includes('"/v/PMOC - Garden.md"')).toBe(true)
  // No note, no card, no log line, and this chat's prompt box untouched.
  expect(got.written.size).toBe(0)
  expect(got.filled).toEqual([])
  expect(await ui.find({ key: 'task-title-child' })).toBeFalsy()
  expect(got.toasts.some(text => text.includes('shows under Chats on PMOC - Garden as "wire the sprinkler timer"'))).toBe(true)

  // The draft is pasted into a new chat in the same folder and sent.
  got.state.id = 'child-1'
  got.written.set(child, '')
  await $.prompt.submit({ text: draft })
  expect(anchorOf(got.store, 'child-1')).toBe('PMOC - Garden')
  expect(parseTrail(got.written.get(trailPath('PMOC - Garden')) ?? '').chats).toEqual([{ id: 'child-1', when: '2026-11-08 09:30', title: 'wire the sprinkler timer', done: false }])
  expect([...got.written.keys()].filter(path => path.endsWith('.md') && !path.includes('Agent Log'))).toEqual([])

  // Back in the parent chat, the pane reads the log again and lists the child with its goal.
  got.state.id = 'parent'
  seen[child] = new Date(2026, 10, 8, 9, 42).getTime()
  await ui.press({ key: 'refresh' })
  expect(await ui.find({ text: /🌱 Chats on PMOC - Garden · 1 open/ })).toBeTruthy()
  expect((await ui.find({ key: 'chat-child-1' }))?.text).toBe('◦ wire the sprinkler timer · 09:42')
  // The child keeps working: its row follows its transcript without a full reload.
  seen[child] = new Date(2026, 10, 8, 10, 5).getTime()
  await $.prompt.submit({ text: 'still here' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect((await ui.find({ key: 'chat-child-1' }))?.text).toMatch(/· 10:05$/)
  // The parent chat itself is listed too, marked as this chat, by its own first prompt.
  expect((await ui.find({ key: 'chat-parent' }))?.text).toMatch(/◦ still here \(this chat\)/)

  // Pressing a chat copies how to resume it; done folds it away.
  await ui.press({ key: 'chat-child-1' })
  expect(got.copied[1]).toBe('claude --resume child-1')
  await ui.press({ key: 'chat-done-child-1' })
  expect(await ui.find({ key: 'chat-done-child-1' })).toBeFalsy()
  expect(await ui.find({ text: /🌱 Chats on PMOC - Garden · 1 open/ })).toBeTruthy()
  expect(await ui.find({ key: 'chats-done' })).toBeTruthy()
  await ui.unmount()
})

test('→ task turns a chat on the anchor into a task in one press: note, sub-item, its own log with the chat moved there, and that chat anchored on it', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [], [['parent', 'PMOC - Garden'], ['child-1', 'PMOC - Garden']])
  got.state.id = 'parent'
  const child = '/Users/t/.claude/projects/-r-garden/child-1.jsonl'
  got.written.set(child, '')
  const log = addChat(addChat(trailTemplate('PMOC - Garden', '2026-11-08'), { id: 'parent', when: '2026-11-08 09:00', title: 'plan the beds', done: false }), { id: 'child-1', when: '2026-11-08 09:10', title: 'wire the sprinkler timer', done: false })
  got.written.set(trailPath('PMOC - Garden'), log)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect(await ui.find({ text: /🌱 Chats on PMOC - Garden · 2 open/ })).toBeTruthy()
  // One search finds the open chats' transcripts, by id, two folders deep and no further.
  expect(got.ran.find(argv => argv[0] === 'find' && argv.includes('-maxdepth'))).toEqual(
    ['find', '/Users/t/.claude/projects', '-maxdepth', '2', '(', '-name', 'child-1.jsonl', '-o', '-name', 'parent.jsonl', ')'],
  )

  // The title field opens with the chat's goal; cancel makes nothing.
  await ui.press({ key: 'chat-task-child-1' })
  expect((await ui.find({ key: 'task-title-promote' }))?.props?.value).toBe('wire the sprinkler timer')
  await ui.press({ key: 'task-cancel' })
  expect(await ui.find({ key: 'task-title-promote' })).toBeFalsy()
  expect(got.written.has('/v/wire the sprinkler timer.md')).toBe(false)

  await ui.press({ key: 'chat-task-child-1' })
  await ui.input({ key: 'task-title-promote', text: 'Wire the sprinkler timer PROJ-7' })
  // The note, from the template, Up the anchor, with the Jira key typed.
  const note = got.written.get('/v/Wire the sprinkler timer.md') ?? ''
  expect(note.startsWith('Up: [[PMOC - Garden]]\nJira: PROJ-7\n')).toBe(true)
  expect(note.includes('### Outcome')).toBe(true)
  // A linked sub-item under the anchor's card, nothing else on the board moved.
  const item = '\t- [ ] [[Wire the sprinkler timer]]'
  const board = got.written.get('/v/Kanban.md') ?? ''
  expect(board.split('\n').indexOf(item)).toBe(BOARD_TEXT.split('\n').indexOf('\t- [ ] [[PMOC - Birdhouse]]') + 1)
  expect(without(board, item)).toBe(BOARD_TEXT)
  // Its own log: the anchor and the chat's transcript pinned, and the chat itself, as it was.
  const made = parseTrail(got.written.get(trailPath('Wire the sprinkler timer')) ?? '')
  expect(made.pins).toEqual([
    { on: true, kind: 'note', target: 'PMOC - Garden', label: 'PMOC - Garden' },
    { on: true, kind: 'session', target: child, label: 'the chat this task came from' },
  ])
  expect(made.chats).toEqual([{ id: 'child-1', when: '2026-11-08 09:10', title: 'wire the sprinkler timer', done: false }])
  // The parent's log lost only that chat's line.
  expect(got.written.get(trailPath('PMOC - Garden'))).toBe(removeChat(log, 'child-1'))
  expect(await ui.find({ key: 'chat-child-1' })).toBeFalsy()
  expect(await ui.find({ text: /🌱 Chats on PMOC - Garden · 1 open/ })).toBeTruthy()
  // That chat is on the task now; this one stays where it was, and so does the folder's anchor.
  expect(anchorOf(got.store, 'child-1')).toBe('Wire the sprinkler timer')
  expect(anchorOf(got.store, 'parent')).toBe('PMOC - Garden')
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  expect(got.store.get('folders')).toEqual([])
  expect(await ui.find({ text: '✓ just made: Wire the sprinkler timer' })).toBeTruthy()
  expect(got.toasts.some(text => text.includes("under PMOC - Garden's card") && text.includes('That chat is anchored on it now.'))).toBe(true)
  // The new task is among the anchor's open tasks.
  await ui.press({ key: 'anchor-details' })
  expect(await ui.find({ key: 'task-Wire the sprinkler timer' })).toBeTruthy()

  // The child chat's own pane follows on its next sync: it is on the task.
  got.state.id = 'child-1'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/Wire the sprinkler timer/)
  expect((await ui.find({ key: 'chat-child-1' }))?.text).toMatch(/wire the sprinkler timer \(this chat\)/)
  await ui.unmount()
})

test('→ task on this chat itself anchors it on the new task; a name already taken makes nothing and moves nothing', { options: OPTIONS }, async ($, on) => {
  const got = inheritHooks(on, [], [['parent', 'PMOC - Garden']])
  got.state.id = 'parent'
  const log = addChat(trailTemplate('PMOC - Garden', '2026-11-08'), { id: 'parent', when: '2026-11-08 09:00', title: 'chat', done: false })
  got.written.set(trailPath('PMOC - Garden'), log)
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'chat-task-parent' })
  // A chat still titled by the placeholder offers no title: the user names the task.
  expect((await ui.find({ key: 'task-title-promote' }))?.props?.value ?? '').toBe('')
  // A note of that name exists: nothing is written, the chat stays where it is.
  await ui.input({ key: 'task-title-promote', text: 'PMOC - Birdhouse' })
  expect(got.toasts.some(text => text.includes('already there'))).toBe(true)
  expect(got.written.get(trailPath('PMOC - Garden'))).toBe(log)
  expect(anchorOf(got.store, 'parent')).toBe('PMOC - Garden')

  await ui.press({ key: 'chat-task-parent' })
  await ui.input({ key: 'task-title-promote', text: 'Mend the fence' })
  expect(got.written.get('/v/Mend the fence.md')?.startsWith('Up: [[PMOC - Garden]]\n')).toBe(true)
  expect(anchorOf(got.store, 'parent')).toBe('Mend the fence')
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/Mend the fence/)
  // The placeholder title becomes the task's name in its log; this chat's transcript is pinned.
  const made = parseTrail(got.written.get(trailPath('Mend the fence')) ?? '')
  expect(made.chats).toEqual([{ id: 'parent', when: '2026-11-08 09:00', title: 'Mend the fence', done: false }])
  expect(made.pins.map(pin => pin.kind)).toEqual(['note'])
  expect(parseTrail(got.written.get(trailPath('PMOC - Garden')) ?? '').chats).toEqual([])
  expect(got.toasts.some(text => text.includes('This chat is anchored on it now.'))).toBe(true)
  // The task this chat is now on is what the next new chat in the folder starts on.
  expect(got.store.get('folders')).toEqual([['/r/garden', 'Mend the fence']])
  await ui.unmount()
})

test('a draft names its anchor so a pasted copy can attach a new chat; anything else names none', async () => {
  expect(anchorFromDraft(buildDraft([], ALL_ON, null, 'go', ANCHOR))).toBe('PMOC - Garden')
  const odd = { note: 'Plan: a/b', notePath: '/v/Plan- a-b.md', trail: trailPath('Plan: a/b') }
  expect(trailPath('Plan: a/b').endsWith('Agent Log - Plan- a-b.md')).toBe(true)
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

// A vault with a question already in PMOC - Garden's log; `session.id` is read each call.
const chatHooks = (on: On, session: { id: string }, anchors: [string, string][]) => {
  const question = '- [ ] 2026-11-08 09:00 · does the cache survive restarts?'
  const written = new Map<string, string>([[trailPath('PMOC - Garden'), withQuestion(trailTemplate('PMOC - Garden', '2026-11-08'), '2026-11-08 09:00', 'does the cache survive restarts?')]])
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
    const found = name === 'PMOC - Garden.md' ? '/v/PMOC - Garden.md' : ''

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

test('copy it for a new chat copies the draft and leaves the box; sent in a chat with no anchor, it anchors that chat; one with an anchor keeps it', { options: OPTIONS }, async ($, on) => {
  const session = { id: 'first' }
  const got = chatHooks(on, session, [['first', 'PMOC - Garden'], ['other', 'PMOC - Birdhouse']])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'draft' })
  await ui.press({ key: 'copy-draft' })
  expect(got.filled.length).toBe(1)
  expect(got.copied).toEqual([got.filled[0]])
  expect(anchorFromDraft(got.copied[0] ?? '')).toBe('PMOC - Garden')
  expect(got.toasts.some(text => text.startsWith('Copied: paste it into a new chat'))).toBe(true)

  // A new chat with no anchor: the pasted draft anchors it, and the pane shows it.
  session.id = 'second'
  await ui.press({ key: 'refresh' })
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/pick a project or task/)
  await $.prompt.submit({ text: got.copied[0] ?? '' })
  expect(anchorOf(got.store, 'second')).toBe('PMOC - Garden')
  expect(parseTrail(got.written.get(trailPath('PMOC - Garden')) ?? '').chats.map(chat => [chat.id, chat.title])).toEqual([['second', 'explain the cache']])
  expect((await ui.find({ key: 'anchor-menu' }))?.text).toMatch(/PMOC - Garden/)
  expect(await ui.find({ key: 'chat-second' })).toBeTruthy()

  // A chat anchored elsewhere keeps its anchor; the chat goes in its own anchor's log.
  session.id = 'other'
  await $.prompt.submit({ text: got.copied[0] ?? '' })
  expect(anchorOf(got.store, 'other')).toBe('PMOC - Birdhouse')
  expect(parseTrail(got.written.get(trailPath('PMOC - Birdhouse')) ?? '').chats.map(chat => chat.id)).toEqual(['other'])
  expect(parseTrail(got.written.get(trailPath('PMOC - Garden')) ?? '').chats.map(chat => chat.id)).toEqual(['second'])
  await ui.unmount()
})

test('touched files: newest first, one row per path, made sticks, capped', async () => {
  const one = (path: string, made = false) => ({ path, made })
  const list = withTouched(withTouched(withTouched([], one('/r/a.ts', true)), one('/r/b.ts')), one('/r/a.ts'))
  expect(list).toEqual([one('/r/a.ts', true), one('/r/b.ts')])
  let many: { path: string; made: boolean }[] = []
  for (let at = 0; at < 40; at += 1) {
    many = withTouched(many, one(`/r/${at}.ts`))
  }
  expect(many.length).toBe(30)
  expect(many[0]?.path).toBe('/r/39.ts')
  expect(withTouched([one('/r/a.ts')], one('/r/b.ts'), 1)).toEqual([one('/r/b.ts')])
})

test('a touched path is short: notes-relative, then root-relative, then ~, and never more than the name and two folders', async () => {
  expect(shortPath('/v/Ideas/A.md', '/v', '/Users/t', '/Users/t/r')).toBe('Ideas/A.md')
  expect(shortPath('/Users/t/r/src/a.ts', '/v', '/Users/t', '/Users/t/r')).toBe('src/a.ts')
  expect(shortPath('/Users/t/other/a.ts', '/v', '/Users/t', '/Users/t/r')).toBe('~/other/a.ts')
  expect(shortPath('/tmp/a.ts', '/v', '/Users/t', '')).toBe('/tmp/a.ts')
  // A deep path keeps its last three parts, wherever it is.
  expect(shortPath('/Users/t/r/packages/web/src/hooks/a.ts', '/v', '/Users/t', '/Users/t/r')).toBe('…/src/hooks/a.ts')
  expect(shortPath('/v/Areas/Garden/Beds/North/A.md', '/v', '/Users/t', '')).toBe('…/Beds/North/A.md')
  expect(shortPath('/Users/t/other/deep/down/a.ts', '/v', '/Users/t', '/Users/t/r')).toBe('…/deep/down/a.ts')
  expect(shortPath('/opt/tools/share/lib/a.ts', '/v', '/Users/t', '')).toBe('…/share/lib/a.ts')
  // Exactly three parts is not cut; a file at the root has no folder to show.
  expect(shortPath('/Users/t/r/src/hooks/a.ts', '/v', '/Users/t', '/Users/t/r')).toBe('src/hooks/a.ts')
  expect(shortPath('/Users/t/r/a.ts', '/v', '/Users/t', '/Users/t/r')).toBe('a.ts')
})

test('a vault note opens in the note editor, any other file in the file editor', async () => {
  const paths = resolvePaths({ notesFolder: '/v', editor: 'obsidian' }, '/Users/t')
  expect(paths.fileEditor).toBe('vscode')
  expect(openArgv(paths, '/v/A.md')).toEqual(['open', 'obsidian://open?path=%2Fv%2FA.md'])
  expect(openArgv(paths, '/v/pic.png')).toEqual(['code', '/v/pic.png'])
  expect(openArgv(paths, '/r/notes.md')).toEqual(['code', '/r/notes.md'])
  expect(openArgv({ ...paths, fileEditor: 'cursor' }, '/r/a.ts')).toEqual(['cursor', '/r/a.ts'])
  expect(openArgv({ ...paths, fileEditor: 'system' }, '/r/a.ts')).toEqual(['open', '/r/a.ts'])
})

test('Agent Logs and dot-folders in the notes are not listed; the pin is a note by name or the file, unticked', async () => {
  expect(isListedTouch('/r/src/a.ts', '/v', '/v/Notes')).toBe(true)
  expect(isListedTouch('/r/.github/ci.yml', '/v', '/v/Notes')).toBe(true)
  expect(isListedTouch('/v/Notes/Agent Log - X.md', '/v', '/v/Notes')).toBe(false)
  expect(isListedTouch('/v/.obsidian/app.json', '/v', '/v/Notes')).toBe(false)
  expect(touchedPin('/v/Ideas/A.md', '/v')).toEqual({ on: false, kind: 'note', target: 'A', label: 'A', path: '/v/Ideas/A.md' })
  expect(touchedPin('/r/README.md', '/v')).toEqual({ on: false, kind: 'file', target: '/r/README.md', label: 'README.md' })
})

test('files this chat touched: listed newest first, opened in the right editor, pinned as unticked links that show in the Pins card', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  const store = new Map<string, unknown>()
  const onDisk = new Set(['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/r/README.md', '/r/src/b.ts'])
  const ran: string[][] = []
  const toasts: string[] = []
  let answer: object = { result: {}, text: 'ok' }
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', (_, e) => {
    toasts.push(e.text)
    return { value: undefined }
  })
  on('session.id', () => ({ value: 'own-session' }))
  on('session.cwd', () => ({ value: '/r' }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: onDisk.has(e.path) || e.path.endsWith('Kanban.md') || written.has(e.path) }))
  on('fs.read', (_, e) => ({ value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', () => ({ value: [] }))
  on('process.run', (_, e) => {
    if (e.argv[0] !== 'find' && e.argv[0] !== 'mkdir') {
      ran.push(e.argv)
    }
    return { value: { exitCode: 0, stdout: e.argv[0] === 'find' && e.argv.includes('PMOC - Garden.md') ? '/v/PMOC - Garden.md\n' : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  // The tool runs as told: the file is on disk after, unless the call is denied or fails.
  on('tool.call', (_, e) => {
    const input = e as { file_path?: string; notebook_path?: string }
    if (!('deny' in answer) && !('isError' in answer)) {
      onDisk.add(input.file_path ?? input.notebook_path ?? '')
    }
    return answer as never
  })
  const pinsNow = () => parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').pins

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ key: 'card-files' })).toBeFalsy()

  await $.tool.call({ tool: 'Write', file_path: '/r/src/new.ts', content: 'x' })
  await $.tool.call({ tool: 'Edit', file_path: '/r/README.md', old_string: 'a', new_string: 'b' })
  // The Agent Log folder is the pane's own; a denied or failed call changed nothing.
  await $.tool.call({ tool: 'Write', file_path: '/v/Notes/Agent Log - X.md', content: 'x' })
  answer = { deny: 'no' }
  await $.tool.call({ tool: 'Edit', file_path: '/r/src/denied.ts', old_string: 'a', new_string: 'b' })
  answer = { result: 'boom', text: 'boom', isError: true }
  await $.tool.call({ tool: 'Write', file_path: '/r/src/failed.ts', content: 'x' })
  answer = { result: {}, text: 'ok' }

  // A row's text with its spacing dropped: the path, `new`, then its buttons.
  const rowOf = async (path: string) => ((await ui.find({ key: `touched-${path}` }))?.text ?? '').replace(/\s+/g, '')
  expect(await ui.find({ text: '📄 Files this chat touched · 2' })).toBeTruthy()
  expect((await ui.findAll({ type: 'Box', text: /open/ })).map(one => one.key).filter(key => key?.startsWith('touched-'))).toEqual(['touched-/r/README.md', 'touched-/r/src/new.ts'])
  expect(await rowOf('/r/README.md')).toBe('README.mdopen')
  expect(await rowOf('/r/src/new.ts')).toBe('src/new.tsnewopen')
  expect(await rowOf('/v/Notes/Agent Log - X.md')).toBe('')
  expect(await rowOf('/r/src/denied.ts')).toBe('')
  expect(await rowOf('/r/src/failed.ts')).toBe('')
  expect(await ui.find({ key: 'touched-pin-/r/README.md' })).toBeFalsy()
  expect(await ui.find({ text: /Pick an anchor to pin these/ })).toBeTruthy()

  await ui.press({ key: 'touched-open-/r/README.md' })
  expect(ran.pop()).toEqual(['code', '/r/README.md'])
  await $.tool.call({ tool: 'Write', file_path: '/v/Ideas/Fresh.md', content: 'x' })
  expect(await rowOf('/v/Ideas/Fresh.md')).toBe('Ideas/Fresh.mdnewopen')
  await ui.press({ key: 'touched-open-/v/Ideas/Fresh.md' })
  expect(ran.pop()).toEqual(['open', 'obsidian://open?path=%2Fv%2FIdeas%2FFresh.md'])
  onDisk.delete('/r/src/new.ts')
  await ui.press({ key: 'touched-open-/r/src/new.ts' })
  expect(toasts.pop()).toBe('Not found: /r/src/new.ts')
  expect(ran.length).toBe(0)

  // Anchored: pin writes an unticked pin, the row says pinned, the Anchor card links it.
  await ui.press({ key: 'anchor-menu' })
  await ui.press({ key: 'card-PMOC - Garden' })
  await ui.press({ key: 'touched-pin-/r/README.md' })
  expect(pinsNow()).toEqual([{ on: false, kind: 'file', target: '/r/README.md', label: 'README.md' }])
  expect(await ui.find({ key: 'touched-pin-/r/README.md' })).toBeFalsy()
  expect(await rowOf('/r/README.md')).toMatch(/openpinned$/)
  expect((await ui.find({ key: 'pin-tick-file-/r/README.md' }))?.text).toMatch(/☐ 📄 README.md/)
  await ui.press({ key: 'pin-open-file-/r/README.md' })
  expect(ran.pop()).toEqual(['code', '/r/README.md'])
  await ui.press({ key: 'touched-pin-/v/Ideas/Fresh.md' })
  expect(pinsNow()[1]).toEqual({ on: false, kind: 'note', target: 'Fresh', label: 'Fresh' })
  await ui.press({ key: 'pin-open-note-Fresh' })
  expect(ran.pop()).toEqual(['open', 'obsidian://open?path=%2Fv%2FIdeas%2FFresh.md'])

  // A new vault note written now is still pinned unticked by itself, and listed as pinned.
  await $.tool.call({ tool: 'Write', file_path: '/v/Ideas/Later.md', content: 'x' })
  expect(pinsNow()[2]).toEqual({ on: false, kind: 'note', target: 'Later', label: 'Later' })
  expect(await rowOf('/v/Ideas/Later.md')).toBe('Ideas/Later.mdnewopenpinned')

  // Past five, the rest wait behind show all.
  await $.tool.call({ tool: 'NotebookEdit', notebook_path: '/r/nb.ipynb', new_source: 'x' })
  await $.tool.call({ tool: 'Edit', file_path: '/r/src/b.ts', old_string: 'a', new_string: 'b' })
  expect(await ui.find({ text: '📄 Files this chat touched · 6' })).toBeTruthy()
  expect(await rowOf('/r/src/new.ts')).toBe('')
  expect(await rowOf('/r/nb.ipynb')).toMatch(/^nb\.ipynbnewopen/)
  await ui.press({ key: 'files-all' })
  expect(await rowOf('/r/src/new.ts')).toMatch(/^src\/new\.tsnewopen/)
  await ui.unmount()
})

// padHooks without the events a test wants to answer itself.
const except = (on: On, names: string[]): On => ((name: string, ...rest: unknown[]) => (names.includes(name) ? undefined : (on as unknown as (...args: unknown[]) => unknown)(name, ...rest))) as unknown as On

test('another chat writes the anchor log: this pane reads it again when the file changed, and not otherwise', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  const store = new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]]])
  padHooks(except(on, ['fs.read']), store, written, [])
  let mtime = 1
  let reads = 0
  on('fs.stat', () => ({ value: { kind: 'file', size: 1, mtimeMs: mtime, isLink: false } }))
  on('prompt.submit', (_, e) => ({ text: e.text }))
  on('fs.read', (_, e) => {
    reads += e.path.includes('Agent Log - ') ? 1 : 0
    return { value: written.get(e.path) ?? (e.path.endsWith('Kanban.md') ? BOARD_TEXT : '') }
  })

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  const log = trailPath('PMOC - Garden')
  // Another chat on the same anchor leaves a question in its log.
  written.set(log, withQuestion(trailTemplate('PMOC - Garden', '2022-01-05'), '2022-01-05 10:00', 'from the other chat'))
  mtime = 2
  await $.prompt.submit({ text: 'hello' })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(await ui.find({ text: 'from the other chat' })).toBeTruthy()
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
  expect(usageName('chat-own-session')).toBe('chat')
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
  expect(report).toMatch(/Lenses run:\n +1 {2}echoes/)
  const never = report.split('Never used')[1] ?? ''
  expect(never).toContain('copy-draft')
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
  await ui.press({ key: 'draft' })
  await $.prompt.submit({ text: filled[0] ?? '' })

  const events = parseUsage(lines.join('\n'))
  expect(events.map(one => `${one.kind} ${one.what}`)).toEqual(['press anchor-menu', 'press card', 'input pad-input', 'press draft', 'sent draft-sent'])
  expect(events.map(one => one.anchored)).toEqual([false, true, true, true, true])
  expect(events[0]?.surface).toBe('desktop')
  expect(events[0]?.at).toBe('2022-03-30T09:00:00.000Z')
  expect(lines.join('\n')).not.toMatch(/Garden|secret|cache/)

  // /augi usage prints the report from the log file.
  written.set('/Users/t/.openaugi/usage/augi-ui.jsonl', lines.join('\n'))
  const said = await $.command.run({ command: 'augi', args: 'usage' })
  expect(JSON.stringify(said)).toContain('5 events')
  await ui.unmount()
})
