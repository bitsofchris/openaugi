import type { On } from 'claude-code'
import { expect, test } from 'claude-code/testing'

import { findInJiraDraft, insertCard, jiraBrowseUrl, jiraFrom, jiraKeyFrom, jiraServerFrom, jiraUpdateDraft, newTaskDraft, parseNewTask, TASK_TEMPLATE, taskNoteName, taskNoteText, withJira } from './register'
import { addAnswer, addChat, addJournalEntry, anchorFromDraft,anchorArtifacts, anchorPairs, artifactsFirst, retitleChat, withAnchor, withArtifact, logName, nextFrom, addPin, addQuestion, contextItems, parsePin, pathsFromSearch, pinFrom, pinLine, pinLink, removePin, sessionPin, setPinOn, resumeCommand, setDone, configure, editorArgv, obsidianConfigs, resolvePaths, setupItems, vaultsFrom, ago, anchorFrom, buildDraft, chatTitleFrom, descriptionFrom, kanbanCards, lensFileName, lensNames, newLensSkeleton, onePerCodexRun, outcomeFrom, parseTrail, setHidden, tidyTrail, trailPath, trailTemplate, parseClaudeHead, parseCodexHead, toggled, typedFrom } from './register'

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
}
configure(resolvePaths(OPTIONS, '/Users/t'))

const ALL_ON = { aboutMe: true, openAugi: true, project: true }
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

  const bare = buildDraft([{ ...SESSION_PIN, on: false }], { aboutMe: false, openAugi: false, project: false }, null, 'hi', ANCHOR)
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

  // An empty first column takes the card under the blank line after its heading.
  const empty = '---\n\nkanban-plugin: board\n\n---\n\n## Todo\n\n\n\n## Doing\n\n- [ ] [[A]]\n'
  const filled = insertCard(empty, 'Fix the gate', null)
  expect(filled).toBe('---\n\nkanban-plugin: board\n\n---\n\n## Todo\n\n- [ ] [[Fix the gate]]\n\n\n## Doing\n\n- [ ] [[A]]\n')
  expect(kanbanCards(filled)[0]).toEqual({ column: 'Todo', note: 'Fix the gate', detail: '', tasks: [] })
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

  const one = addChat(empty, { id: 's1', when: '2026-11-08 10:00', title: 'echoes: why slow', hidden: false })
  const two = addChat(one, { id: 's2', when: '2026-11-08 11:00', title: 'second', hidden: false })
  expect(addChat(two, { id: 's1', when: '2026-11-08 12:00', title: 'again', hidden: false })).toBe(two)
  expect(parseTrail(two).chats.map(chat => chat.id)).toEqual(['s2', 's1'])

  const saved = two.replace('## Saved\n', '## Saved\n- [[A note]] from a chat\n')
  expect(parseTrail(saved).saved).toEqual(['A note'])
})

test('a captured question goes on top of ## Questions and can be ticked done', async () => {
  const empty = trailTemplate('X', '2026-11-08')
  const one = addQuestion(empty, { when: '2026-11-08 10:00', text: 'does the sprinkler cache?', done: false })
  const two = addQuestion(one, { when: '2026-11-08 11:00', text: 'side quest:  try  arrow', done: false })
  expect(parseTrail(two).questions.map(q => q.text)).toEqual(['side quest: try arrow', 'does the sprinkler cache?'])
  expect(two.includes('## Questions\n\n- [ ] 2026-11-08 11:00 · side quest: try arrow\n- [ ] 2026-11-08 10:00 · does the sprinkler cache?\n\n## Chats')).toBe(true)
  expect(addQuestion(one, { when: '2026-11-08 12:00', text: '  ', done: false })).toBe(one)

  const done = setDone(two, { when: '2026-11-08 10:00', text: 'does the sprinkler cache?', done: false }, true)
  expect(parseTrail(done).questions.map(q => q.done)).toEqual([false, true])

  const old = '# Trail\n\n## Chats\n\n## Saved\n'
  const added = addQuestion(old, { when: '2026-11-08 10:00', text: 'q', done: false })
  expect(added.indexOf('## Questions') < added.indexOf('## Chats')).toBe(true)
  expect(parseTrail(added).questions.length).toBe(1)
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
  expect(contextItems(pins, { project: false, aboutMe: false, openAugi: false }, anchor).map(item => item.kind)).toEqual(['pin', 'pin'])
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

test('what you paste becomes the right kind of pin, and link copies something usable', async () => {
  expect(pinFrom('https://claude.ai/artifact/abc-1', '/v', '/Users/t')?.kind).toBe('artifact')
  expect(pinFrom('https://claude.ai/code/artifact/abc-1', '/v', '/Users/t')?.kind).toBe('artifact')
  expect(pinFrom('https://example.com/a', '/v', '/Users/t')?.kind).toBe('url')
  expect(pinFrom('[[Sprinkler Benchmarks]]', '/v', '/Users/t')).toEqual({ on: true, kind: 'note', target: 'Sprinkler Benchmarks', label: 'Sprinkler Benchmarks' })
  expect(pinFrom('"/v/Notes/Sprinkler Benchmarks.md"', '/v', '/Users/t')).toEqual({ on: true, kind: 'note', target: 'Sprinkler Benchmarks', label: 'Sprinkler Benchmarks', path: '/v/Notes/Sprinkler Benchmarks.md' })
  expect(pinFrom('~/.claude/projects/-r/abc.jsonl', '/v', '/Users/t')?.kind).toBe('session')
  expect(pinFrom('~/repos/x/README.md', '/v', '/Users/t')).toEqual({ on: true, kind: 'file', target: '/Users/t/repos/x/README.md', label: 'README.md' })
  expect(pinFrom('   ', '/v', '/Users/t')).toBe(null)

  expect(pinLink({ on: true, kind: 'note', target: 'A', label: 'A' }, '/v/A.md')).toBe('/v/A.md')
  expect(pinLink({ on: true, kind: 'note', target: 'A', label: 'A' })).toBe('[[A]]')
  expect(pinLink({ on: true, kind: 'session', target: '/Users/t/.claude/projects/-r/abc-1.jsonl', label: '' })).toBe('claude --resume abc-1')
  expect(pinLink(sessionPin({ ...SESSION, tool: 'codex', path: '/Users/t/.codex/sessions/r.jsonl' }))).toBe('/Users/t/.codex/sessions/r.jsonl')
  expect(pinLink({ on: true, kind: 'artifact', target: 'https://claude.ai/artifact/abc-1', label: '' })).toBe('https://claude.ai/artifact/abc-1')
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

test('hiding a chat marks its line and keeps it listed', async () => {
  const text = addChat(trailTemplate('X', '2026-11-08'), { id: 's1', when: '2026-11-08 10:00', title: 't', hidden: false })
  const hidden = setHidden(text, 's1', true)
  expect(hidden.includes('`s1` (hidden)')).toBe(true)
  expect(parseTrail(hidden).chats[0]?.hidden).toBe(true)
  expect(parseTrail(setHidden(hidden, 's1', false)).chats[0]?.hidden).toBe(false)
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
  // The anchor card opens the human and agent notes; overall context lives under Setup.
  expect(await ui.find({ key: 'open-note' })).toBeTruthy()
  expect(await ui.find({ key: 'open-trail' })).toBeTruthy()
  expect(await ui.find({ key: 'edit-context' })).toBeFalsy()
  expect(await ui.find({ key: 'about-me' })).toBeFalsy()
  await ui.press({ key: 'setup-menu' })
  expect(await ui.find({ key: 'edit-context' })).toBeTruthy()
  expect(await ui.find({ key: 'open-board' })).toBeTruthy()
  await ui.press({ key: 'setup-menu' })
  // Project context ticks are real checkboxes: pressing one flips it in place.
  expect((await ui.find({ key: 'pick-project' }))?.text).toMatch(/^\s*☑/)
  await ui.press({ key: 'pick-project' })
  expect((await ui.find({ key: 'pick-project' }))?.text).toMatch(/^\s*☐/)
  await ui.press({ key: 'pick-project' })
  await ui.press({ key: 'lens-menu' })
  await ui.press({ key: 'echoes' })

  expect(filled.length).toBe(1)
  expect(filled[0]?.includes('/v/PMOC - Garden.md')).toBe(true)
  expect(filled[0]?.includes('new.jsonl')).toBe(false)

  await $.prompt.submit({ text: filled[0] ?? '' })
  const trail = written.get(trailPath('PMOC - Garden')) ?? ''
  expect(parseTrail(trail).chats).toEqual([
    { id: 'own-session', when: '2026-11-08 09:30', title: 'echoes: is the sprinkler the bottleneck', hidden: false },
  ])
  expect(await ui.find({ key: 'chat-own-session' })).toBeTruthy()

  expect((await ui.find({ key: 'pick-project' }))?.text).toMatch(/^\s*☑/)
  expect(await ui.find({ key: `in-session-${'/Users/t/.claude/projects/-r-garden/new.jsonl'}` })).toBeFalsy()

  await ui.input({ key: 'capture-0', text: 'does the cache survive restarts?' })
  expect(parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').questions[0]?.text).toBe('does the cache survive restarts?')
  await ui.press({ key: 'ask-2026-11-08 09:30' })
  expect(filled[1]?.endsWith('Instruction: does the cache survive restarts?')).toBe(true)
  await ui.press({ key: 'done-2026-11-08 09:30' })
  expect(parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').questions[0]?.done).toBe(true)
  expect(await ui.find({ key: 'ask-2026-11-08 09:30' })).toBeFalsy()

  await ui.press({ key: 'hide-own-session' })
  expect(parseTrail(written.get(trailPath('PMOC - Garden')) ?? '').chats[0]?.hidden).toBe(true)
  expect(await ui.find({ key: 'chat-own-session' })).toBeFalsy()
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
  await ui.press({ key: 'hit-Sprinkler Benchmarks' })
  expect(pinsNow().map(pin => pin.target)).toEqual(['/r/docs/ARCH.md', 'Sprinkler Benchmarks'])
  expect(await ui.find({ key: 'hit-Sprinkler Benchmarks' })).toBeFalsy()
  await ui.press({ key: 'pin-link-note-Sprinkler Benchmarks' })
  expect(copied).toEqual(['/v/Notes/Sprinkler Benchmarks.md'])

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

test('the Agent Log is named for its note and links it first, with no title or seen box', async () => {
  expect(logName('PMOC - A/B')).toBe('Agent Log - PMOC - A-B')
  const log = trailTemplate('PMOC - Garden', '2026-11-08')
  expect(log.includes('# Agent Log - PMOC - Garden')).toBe(false)
  expect(log.includes('type: document\n')).toBe(true)
  expect(log.includes('created: 2026-11-08\n---\n\nHuman note: [[PMOC - Garden]]. Context lines')).toBe(true)
})

const OLD_INTRO = 'Project: [[PMOC - Garden]], embedded there under # Agent Log. Context lines are `- [x] [[note]] · why it matters` (ticked goes into the next draft). Hidden chats stay listed, marked `(hidden)`.'

test('an older Agent Log loses its title and seen box, gains the note link, and keeps its sections', async () => {
  const sections = ['## Context', '', '- [x] [[A note]] · why', '', '## Questions', '', '- [x] 2022-01-04 10:00 · q', '', '## Chats', '', '## Saved', '']
  const old = ['---', 'type: document', 'description: d', 'created: 2022-01-04', '---', '', '# Agent Log - PMOC - Garden', '', '- [x] seen', '', OLD_INTRO, '', ...sections].join('\n')
  const tidy = tidyTrail(old, 'PMOC - Garden')
  expect(tidy).toBe(trailTemplate('PMOC - Garden', '2022-01-04').replace(/description: .*/, 'description: d').split('## Context')[0] + sections.join('\n'))
  expect(tidyTrail(tidy, 'PMOC - Garden')).toBe(tidy)
  expect(parseTrail(tidy)).toEqual(parseTrail(old))
  expect(tidyTrail(trailTemplate('PMOC - Garden', '2022-01-04'), 'PMOC - Garden')).toBe(trailTemplate('PMOC - Garden', '2022-01-04'))

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
  expect((await ui.find({ key: 'lens-menu' }))?.text).toMatch(/Lenses \(saved prompts\) · 1/)
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
  const text = addChat(trailTemplate('X', '2026-11-08'), { id: 's1', when: '2026-11-08 10:00', title: 'chat', hidden: false })
  const named = retitleChat(text, 's1', 'echoes: why slow')
  expect(parseTrail(named).chats[0]?.title).toBe('echoes: why slow')
  expect(retitleChat(named, 's1', 'later')).toBe(named)
  expect(retitleChat(text, 's2', 'other')).toBe(text)
  const hidden = retitleChat(setHidden(text, 's1', true), 's1', 'named')
  expect(parseTrail(hidden).chats[0]).toEqual({ id: 's1', when: '2026-11-08 10:00', title: 'named', hidden: true })
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

test('artifacts: a republished url takes its new title; the anchor card shows the newest three; the context list puts them first', async () => {
  const a = { url: 'https://claude.ai/artifact/a', title: 'A' }
  expect(withArtifact(withArtifact([a], { url: 'https://claude.ai/artifact/b', title: 'B' }), { ...a, title: 'A2' }).map(one => one.title)).toEqual(['A2', 'B'])
  const art = (n: number) => ({ on: true, kind: 'artifact', target: `https://claude.ai/artifact/${n}`, label: String(n) }) as const
  const note = { on: true, kind: 'note', target: 'N', label: 'N' } as const
  expect(anchorArtifacts([art(1), note, art(2), art(3), art(4)]).map(pin => pin.label)).toEqual(['4', '3', '2'])
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
  expect(logNow().chats).toEqual([{ id: 'new-session', when: '2026-11-08 09:30', title: 'chat', hidden: false }])
  expect(logNow().pins).toEqual([{ on: true, kind: 'artifact', target: 'https://claude.ai/artifact/early-1', label: 'Early plan' }])
  expect(await ui.find({ key: 'anchor-artifact-https://claude.ai/artifact/early-1' })).toBeTruthy()
  await ui.press({ key: 'anchor-artifact-https://claude.ai/artifact/early-1' })
  expect(opened).toEqual([['open', 'https://claude.ai/artifact/early-1']])

  // Published after anchoring: pinned ticked, and first in the anchor card.
  url = 'https://claude.ai/artifact/late-2'
  await $.tool.call({ tool: 'Artifact', file_path: '/s/late.html', title: 'Late plan' })
  expect(logNow().pins.map(pin => [pin.target, pin.on])).toEqual([['https://claude.ai/artifact/early-1', true], ['https://claude.ai/artifact/late-2', true]])
  expect((await ui.find({ key: 'anchor-artifact-https://claude.ai/artifact/late-2' }))?.text).toMatch(/◇ Late plan/)

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
  padHooks(on, new Map(), written, [])

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.input({ key: 'pad-input-0', text: 'check the hose' })
  await ui.input({ key: 'pad-input-1', text: 'buy seeds' })
  expect((await ui.find({ key: 'pad-row-0' }))?.text).toMatch(/buy seeds/)
  expect((await ui.find({ key: 'pad-row-1' }))?.text).toMatch(/check the hose/)
  expect(await ui.find({ key: 'pad-save-0' })).toBeFalsy()
  expect(await ui.find({ key: 'pad-save-all' })).toBeFalsy()
  expect(await ui.find({ text: 'Pick an anchor to save these to its Human Notes.' })).toBeTruthy()
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

test('a new task from the pane: note from the template, card after the anchor, chat pinned, anchored, draft in the box; linked sub-items open their note', { options: OPTIONS }, async ($, on) => {
  const written = new Map<string, string>()
  const store = new Map<string, unknown>([['anchors', [['own-session', 'PMOC - Garden']]]])
  const filled: string[] = []
  const ran: string[][] = []
  const toasts: string[] = []
  const transcript = '/Users/t/.claude/projects/-r-garden/own-session.jsonl'
  const known: Record<string, string> = { 'PMOC - Garden.md': '/v/PMOC - Garden.md', 'Child task.md': '/v/Child task.md' }
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.toast', (_, e) => {
    toasts.push(e.text)
    return { value: undefined }
  })
  on('session.id', () => ({ value: 'own-session' }))
  on('clock.now', () => ({ value: new Date(2026, 10, 8, 9, 30).getTime() }))
  on('store.get', (_, e) => ({ value: store.get(e.key) }))
  on('store.set', (_, e) => {
    store.set(e.key, e.value)
    return { value: undefined }
  })
  on('fs.exists', (_, e) => ({ value: ['/v', '/v/lenses', '/v/AGENT/Slowly Changing Context.md', '/v/Kanban.md', '/Users/t/.claude/projects', transcript].includes(e.path) || written.has(e.path) }))
  on('fs.read', (_, e) => ({
    value: written.get(e.path)
      ?? (e.path === '/v/Kanban.md' ? BOARD_TEXT
        : e.path === '/v/PMOC - Garden.md' ? '---\noutcome: It loads fast.\njira: PROJ-1\n---\n# Next\n- [ ] Plant the beds\n- [ ] [[Child task]] wire it\n# Journal\n'
        : ''),
  }))
  on('fs.write', (_, e) => {
    written.set(e.path, e.text)
    return { value: undefined }
  })
  on('fs.list', (_, e) => ({ value: e.path === '/Users/t/.claude/projects' ? [{ name: '-r-garden', kind: 'dir', size: 0, mtimeMs: 0, isLink: false }] : [] }))
  on('process.run', (_, e) => {
    ran.push(e.argv)
    const name = e.argv[0] === 'find' ? e.argv[e.argv.indexOf('-name') + 1] ?? '' : ''
    const found = name ? known[name] ?? [...written.keys()].find(path => path.endsWith(`/${name}`)) ?? '' : ''

    return { value: { exitCode: 0, stdout: found ? `${found}\n` : '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('prompt.read', () => ({ value: { text: '', cursor: 0 } }))
  on('prompt.fill', (_, e) => {
    filled.push(e.text)
    return { isFilled: true }
  })

  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ text: '🔗 PROJ-1' })).toBeTruthy()

  // A sub-item that links a note opens it; one without a link is plain text.
  await ui.press({ key: 'anchor-details' })
  expect(await ui.find({ text: 'Plant the beds' })).toBeTruthy()
  await ui.press({ key: 'task-Child task wire it' })
  expect(ran.some(argv => argv[0] === 'open' && argv[1] === `obsidian://open?path=${encodeURIComponent('/v/Child task.md')}`)).toBe(true)

  await ui.input({ key: 'new-task-2', text: 'Fix the gate PROJ-123' })
  const note = written.get('/v/Fix the gate.md') ?? ''
  expect(note.includes('Jira: PROJ-123')).toBe(true)
  expect(note.includes('Up: [[PMOC - Garden]]')).toBe(true)
  expect(note.includes('### 2026-11-08')).toBe(true)
  expect(note.includes('### Outcome')).toBe(true)
  const board = written.get('/v/Kanban.md') ?? ''
  expect(board.split('\n').indexOf('- [ ] [[Fix the gate]]')).toBe(BOARD_TEXT.split('\n').indexOf('\t- [ ] [[PMOC - Birdhouse]]') + 1)
  expect(without(board, '- [ ] [[Fix the gate]]')).toBe(BOARD_TEXT)
  expect(store.get('anchors')).toEqual([['own-session', 'Fix the gate']])
  expect((await ui.find({ key: 'anchor-menu' }))?.text?.includes('Fix the gate')).toBe(true)
  expect(parseTrail(written.get(trailPath('Fix the gate')) ?? '').pins).toEqual([
    { on: true, kind: 'session', target: transcript, label: 'the chat this task came from' },
  ])
  expect(filled).toEqual([newTaskDraft('/v/Fix the gate.md', trailPath('Fix the gate'), 'Fix the gate')])

  // The same title again writes nothing.
  await ui.input({ key: 'new-task-3', text: 'Fix the gate' })
  expect(toasts.some(text => text.includes('already there'))).toBe(true)
  expect(written.get('/v/Kanban.md')).toBe(board)
  expect(filled.length).toBe(1)
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
  await ui.input({ key: 'new-task-2', text: 'Fix the gate' })
  expect(got.written.has('/v/Tasks/Fix the gate.md')).toBe(true)
  expect(got.written.get('/v/Tasks/Fix the gate.md')?.includes('Up: [[PMOC - Garden]]')).toBe(true)
  expect(got.filled).toEqual([newTaskDraft('/v/Tasks/Fix the gate.md', trailPath('Fix the gate'), 'Fix the gate', { project: 'PROJ', server: '' })])
  await ui.unmount()
})

test('with no task folder, a new task goes to the notes root, never beside the anchor note', { options: OPTIONS }, async ($, on) => {
  const got = jiraHooks(on, GARDEN_NOTE, [])
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.input({ key: 'new-task-2', text: 'Fix the gate' })
  expect(got.written.has('/v/Fix the gate.md')).toBe(true)
  expect(got.written.has('/v/Projects/Fix the gate.md')).toBe(false)
  expect(got.filled).toEqual([newTaskDraft('/v/Fix the gate.md', trailPath('Fix the gate'), 'Fix the gate')])
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

test('a kept side answer goes under its question, indented; the questions parse as before and a second keep replaces it', async () => {
  const older = { when: '2022-01-05 09:30', text: 'does the cache survive restarts?', done: false }
  const newer = { when: '2022-01-05 09:45', text: 'who owns the hose?', done: false }
  const log = addQuestion(addQuestion(trailTemplate('X', '2022-01-05'), older), newer)
  const kept = addAnswer(log, older, 'Yes, it is on disk.\n\n  Restarts read it back.  \n')
  expect(kept).toBe(log.replace(
    '- [ ] 2022-01-05 09:30 · does the cache survive restarts?',
    '- [ ] 2022-01-05 09:30 · does the cache survive restarts?\n  - aside: Yes, it is on disk.\n      Restarts read it back.',
  ))
  expect(parseTrail(kept)).toEqual(parseTrail(log))
  expect(addAnswer(kept, older, 'No.')).toBe(addAnswer(log, older, 'No.'))
  // The top question's answer goes between it and the next question.
  const top = addAnswer(kept, newer, 'The neighbour.')
  expect(top.includes('who owns the hose?\n  - aside: The neighbour.\n- [ ] 2022-01-05 09:30')).toBe(true)
  expect(parseTrail(top).questions).toEqual(parseTrail(log).questions)
  // Ticking done keeps the answer under it.
  const done = setDone(kept, older, true)
  expect(done.includes('- [x] 2022-01-05 09:30 · does the cache survive restarts?\n  - aside: Yes, it is on disk.')).toBe(true)
  expect(parseTrail(done).questions.find(one => one.when === older.when)?.done).toBe(true)
  // A question not in the log, or an empty answer, changes nothing.
  expect(addAnswer(log, { ...older, when: '2022-01-04 08:00' }, 'x')).toBe(log)
  expect(addAnswer(log, older, ' \n ')).toBe(log)
  // Lines someone else put under a question are not an aside: a keep leaves them.
  const noted = '## Questions\n\n- [ ] 2022-01-05 09:30 · q\n\t- tried the hose\n'
  expect(addAnswer(noted, { when: '2022-01-05 09:30', text: 'q', done: false }, 'No.')).toBe('## Questions\n\n- [ ] 2022-01-05 09:30 · q\n  - aside: No.\n\t- tried the hose\n')
})

// A vault with PMOC - Garden (a Next item and a parked question) and PMOC - Birdhouse on the board.
const asideHooks = (on: On, session: { id: string }, anchors: [string, string][], extra: { btw?: boolean; fork?: (prompt: string) => unknown } = {}) => {
  const question = '- [ ] 2026-11-08 09:00 · does the cache survive restarts?'
  const written = new Map<string, string>([[trailPath('PMOC - Garden'), addQuestion(trailTemplate('PMOC - Garden', '2026-11-08'), { when: '2026-11-08 09:00', text: 'does the cache survive restarts?', done: false })]])
  const store = new Map<string, unknown>([['anchors', anchors]])
  const got = { written, store, question, filled: [] as string[], copied: [] as string[], toasts: [] as string[], forks: [] as { prompt: string; effort?: string }[] }
  on('env.get', () => ({ value: '/Users/t' }))
  on('command.register', () => ({ value: { command: 'augi' } }))
  on('command.list', () => ({ value: extra.btw ? [{ name: 'clear', description: '', source: 'builtin' }, { name: 'btw', description: 'ask aside', source: 'builtin' }] : [{ name: 'clear', description: '', source: 'builtin' }] }))
  on('model.fork', (_, e) => {
    got.forks.push({ prompt: e.prompt, effort: e.effort })
    return { value: (extra.fork ?? (() => ({ isAnswered: false, reason: 'nothing-to-fork' })))(e.prompt) as never }
  })
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
  const got = asideHooks(on, session, [['first', 'PMOC - Garden'], ['other', 'PMOC - Birdhouse']])
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

test('with /btw in this session, aside fills /btw and the question unsent; pressing the question still drafts it', { options: OPTIONS }, async ($, on) => {
  const got = asideHooks(on, { id: 'own' }, [['own', 'PMOC - Garden']], { btw: true })
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.press({ key: 'aside-2026-11-08 09:00' })
  expect(got.filled).toEqual(['/btw does the cache survive restarts?'])
  expect(got.forks).toEqual([])
  await ui.press({ key: 'ask-2026-11-08 09:00' })
  expect(got.filled[1]?.endsWith('Instruction: does the cache survive restarts?')).toBe(true)
  await ui.unmount()
})

test('without /btw, aside forks the chat: the answer shows under the question, keep writes it under the question, drop clears it', { options: OPTIONS }, async ($, on) => {
  let replies = 0
  const got = asideHooks(on, { id: 'own' }, [['own', 'PMOC - Garden']], {
    fork: () => {
      replies += 1
      return replies === 1
        ? { isAnswered: false, reason: 'nothing-to-fork' }
        : { isAnswered: true, text: `Yes, it is on disk (${replies}).`, usage: { input_tokens: 1, output_tokens: 1, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 } }
    },
  })
  const log = () => got.written.get(trailPath('PMOC - Garden')) ?? ''
  const before = log()
  await $.command.run({ command: 'augi', args: '' })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })

  // Nothing to fork yet: a toast, the question as it was.
  await ui.press({ key: 'aside-2026-11-08 09:00' })
  expect(got.forks[0]?.effort).toBe('low')
  expect(got.forks[0]?.prompt.startsWith('does the cache survive restarts?\n')).toBe(true)
  expect(got.toasts.some(text => text.startsWith('No side answer: nothing to ask yet'))).toBe(true)
  expect(await ui.find({ key: 'aside-keep-2026-11-08 09:00' })).toBeFalsy()
  expect(await ui.find({ key: 'aside-2026-11-08 09:00' })).toBeTruthy()

  // Answered: shown under the question; drop clears it and writes nothing.
  await ui.press({ key: 'aside-2026-11-08 09:00' })
  expect(await ui.find({ text: 'Yes, it is on disk (2).' })).toBeTruthy()
  await ui.press({ key: 'aside-drop-2026-11-08 09:00' })
  expect(await ui.find({ text: 'Yes, it is on disk (2).' })).toBeFalsy()
  expect(log()).toBe(before)
  expect(got.filled).toEqual([])

  // Answered again and kept: under the question in the Agent Log, the question still open.
  await ui.press({ key: 'aside-2026-11-08 09:00' })
  await ui.press({ key: 'aside-keep-2026-11-08 09:00' })
  expect(log()).toBe(before.replace(got.question, `${got.question}\n  - aside: Yes, it is on disk (3).`))
  expect(parseTrail(log()).questions).toEqual(parseTrail(before).questions)
  expect(await ui.find({ text: 'Yes, it is on disk (3).' })).toBeFalsy()
  expect(await ui.find({ key: 'aside-keep-2026-11-08 09:00' })).toBeFalsy()
  expect(await ui.find({ key: 'done-2026-11-08 09:00' })).toBeTruthy()
  await ui.unmount()
})
