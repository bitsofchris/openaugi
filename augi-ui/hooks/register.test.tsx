import type { On } from 'claude-code'
import { expect, test } from 'claude-code/testing'

import { addChat, addJournalEntry, embedFor, hasEmbedOf, logName, nextFrom, addPin, addQuestion, contextItems, parsePin, pathsFromSearch, pinFrom, pinLine, pinLink, removePin, sessionPin, setPinOn, resumeCommand, setDone, configure, editorArgv, obsidianConfigs, resolvePaths, setupItems, vaultsFrom, ago, anchorFrom, buildDraft, chatTitleFrom, descriptionFrom, kanbanCards, lensNames, newLensDraft, onePerCodexRun, outcomeFrom, parseTrail, setHidden, trailPath, trailTemplate, parseClaudeHead, parseCodexHead, toggled, typedFrom } from './register'

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

test('a new-lens draft asks for the lens contract and keeps what was typed', async () => {
  const text = newLensDraft('check my tests')
  expect(text.includes('trigger: on-demand')).toBe(true)
  expect(text.includes('augi: true')).toBe(false)
  expect(text.endsWith('What it should do: check my tests')).toBe(true)
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
  expect(cards[1]?.tasks).toEqual(['Plant the beds', 'PMOC - Birdhouse'])
})

test('the outcome comes from the note frontmatter', async () => {
  expect(outcomeFrom('---\noutcome: "By the trip: it loads."\n---\nbody')).toBe('By the trip: it loads.')
  expect(outcomeFrom('no frontmatter\noutcome: no')).toBe('no')
  expect(outcomeFrom('---\noutcome: \nstatus: active\n---\n**Outcome:** the sprinkler serves a day in 1s\n')).toBe('the sprinkler serves a day in 1s')
  expect(outcomeFrom('#area/x\n\nOutcome: a mod I use daily.\n# Journal')).toBe('a mod I use daily.')
  expect(outcomeFrom('# Journal\nno outcome here')).toBe('')
})

test('a trail note starts empty, takes chats newest first, once each', async () => {
  const empty = trailTemplate('PMOC - Garden', '2026-11-08')
  expect(empty.includes('- [ ] seen')).toBe(true)
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
  expect(store.get('anchor')).toBe('PMOC - Garden')
  expect(await ui.find({ key: 'copy-embed' })).toBeTruthy()
  expect(await ui.find({ text: 'Plant the beds' })).toBeTruthy()
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
  const store = new Map<string, unknown>([['anchor', 'PMOC - Garden']])
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

  // Not a pane chat yet: a published artifact is not pinned.
  await $.tool.call({ tool: 'Artifact', file_path: '/s/page.html', title: 'Cache explainer' })
  expect(pinsNow().length).toBe(1)
  // Once a draft from the pane is sent, this session is on the trail and its artifacts pin, unticked.
  await ui.press({ key: 'draft' })
  expect(filled[0]?.includes('- Note: "/v/Notes/Sprinkler Benchmarks.md"')).toBe(true)
  await $.prompt.submit({ text: filled[0] ?? '' })
  await $.tool.call({ tool: 'Artifact', file_path: '/s/page.html', title: 'Cache explainer' })
  await $.tool.call({ tool: 'Artifact', action: 'read', url: 'https://claude.ai/artifact/zzz-9' })
  expect(pinsNow().slice(1)).toEqual([{ on: false, kind: 'artifact', target: 'https://claude.ai/artifact/abc-1', label: 'Cache explainer' }])
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
  expect(nextFrom(note)).toEqual(['Plant the beds', 'PMOC - Birdhouse', 'still under Next'])
  expect(nextFrom('## Next\n- [ ] a\n## Later\n- [ ] b')).toEqual(['a'])
  expect(nextFrom('# Journal\n- [ ] a')).toEqual([])
})

test('the Agent Log is named for its note, and the embed is a collapsed callout the note can hold', async () => {
  expect(logName('PMOC - A/B')).toBe('Agent Log - PMOC - A-B')
  const embed = embedFor('PMOC - Garden')
  expect(embed).toBe('# Agent Log\n\n> [!abstract]- Agent Log\n> ![[Agent Log - PMOC - Garden]]\n')
  expect(hasEmbedOf(`# Journal\n${embed}`, 'PMOC - Garden')).toBe(true)
  expect(hasEmbedOf('# Journal', 'PMOC - Garden')).toBe(false)
  expect(trailTemplate('PMOC - Garden', '2026-11-08').includes('# Agent Log - PMOC - Garden')).toBe(true)
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
  padHooks(on, new Map<string, unknown>([['anchor', 'PMOC - Garden']]), written, toasts)

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
