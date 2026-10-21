import { expect, test } from 'vitest'
import { buildFixtureTrace, makeRunRecord } from '../test/fixtures'
import { parseRunFile, serializeRun } from './runFiles'

test('serialize → parse round-trips meta and events', () => {
  const record = makeRunRecord(3, { endedAt: Date.UTC(2026, 7, 28) })
  const { filename, json } = serializeRun(record)
  expect(filename).toBe('tsumugi-run-the-cat-20260828.json')
  const parsed = parseRunFile(json)
  expect(parsed.ok).toBe(true)
  if (parsed.ok) {
    expect(parsed.meta.prompt).toBe('The cat')
    expect(parsed.meta.endedAt).toBe(Date.UTC(2026, 7, 28))
    expect(parsed.events).toEqual(record.events)
  }
})

test('rejects invalid JSON, wrong version, and missing meta', () => {
  expect(parseRunFile('{nope')).toEqual({ ok: false, error: 'not valid JSON' })
  const record = makeRunRecord(1)
  const wrongVersion = JSON.stringify({ formatVersion: 2, meta: record.meta, events: record.events })
  expect(parseRunFile(wrongVersion)).toMatchObject({ ok: false, error: expect.stringContaining('format version') })
  const noMeta = JSON.stringify({ formatVersion: 1, events: record.events })
  expect(parseRunFile(noMeta)).toMatchObject({ ok: false, error: expect.stringContaining('metadata') })
})

test('rejects a file whose trace fails validation', () => {
  const record = makeRunRecord(1)
  const broken = JSON.stringify({ formatVersion: 1, meta: record.meta,
    events: [...record.events].reverse() })
  expect(parseRunFile(broken)).toMatchObject({ ok: false, error: expect.stringContaining('invalid trace') })
})

test('an events array holding null or a non-object is rejected, not thrown on', () => {
  const record = makeRunRecord(1)
  for (const bad of [null, 7, 'x']) {
    const text = JSON.stringify({ formatVersion: 1, meta: record.meta,
      events: [record.events[0], bad, ...record.events.slice(1)] })
    expect(() => parseRunFile(text)).not.toThrow()
    expect(parseRunFile(text)).toMatchObject({ ok: false, error: expect.stringContaining('invalid trace') })
  }
})

test('slug falls back for an unusable prompt', () => {
  const record = makeRunRecord(1, { prompt: '   ', endedAt: Date.UTC(2026, 7, 28) })
  expect(serializeRun(record).filename).toBe('tsumugi-run-run-20260828.json')
})

test('an import with a broken neurons field loads without it and says so', () => {
  const events = buildFixtureTrace({ mode: 'real', mlp: true }).map((e) =>
    e.type === 'edits' ? { ...e, neurons: { neurons: 4, invRms: 0.05, activations: 'AAAA', candidates: [] } } : e)
  const text = JSON.stringify({ formatVersion: 1, meta: { prompt: 'p', params: { temperature: 0.8, topK: 10, maxNewTokens: 2 },
    mode: 'real', endedAt: 1, reason: 'max-tokens' }, events })
  const parsed = parseRunFile(text)
  expect(parsed.ok).toBe(true)
  if (!parsed.ok) return
  expect(parsed.note).toBe('Neuron data of 2 cycles was damaged and left out.')
  expect(parsed.events.some((e) => e.type === 'edits' && e.neurons)).toBe(false)
})

const runText = (events: unknown[]) => JSON.stringify({ formatVersion: 1, meta: { prompt: 'p',
  params: { temperature: 0.8, topK: 10, maxNewTokens: 2 }, mode: 'real', endedAt: 1, reason: 'max-tokens' }, events })

test('an import with only broken step data keeps the neuron data and names the step data', () => {
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true }).map((e) =>
    e.type === 'edits' && e.neurons ? { ...e, neurons: { ...e.neurons, steps: { ...e.neurons.steps!, gate: 'AAAA' } } } : e)
  const parsed = parseRunFile(runText(events))
  expect(parsed.ok).toBe(true)
  if (!parsed.ok) return
  expect(parsed.note).toBe('Step-by-step data of 2 cycles was damaged and left out.')
  expect(parsed.events.filter((e) => e.type === 'edits').every((e) => e.type === 'edits' && e.neurons && !e.neurons.steps)).toBe(true)
})

test('an import with one broken neuron field and one broken step field names both, singular', () => {
  let n = 0
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true }).map((e) => {
    if (e.type !== 'edits' || !e.neurons) return e
    return n++ === 0 ? { ...e, neurons: { ...e.neurons, activations: 'AAAA' } }
      : { ...e, neurons: { ...e.neurons, steps: { ...e.neurons.steps!, up: 'AAAA' } } }
  })
  const parsed = parseRunFile(runText(events))
  expect(parsed.ok && parsed.note).toBe('Neuron data of 1 cycle was damaged and left out. Step-by-step data of 1 cycle was damaged and left out.')
})

test('a chat run file keeps its format; a malformed format is dropped and the run loads as text', () => {
  const events = buildFixtureTrace({ chat: true, mode: 'real' })
  const meta = { prompt: 'What is the capital of France?', params: { temperature: 0, topK: 1, maxNewTokens: 8 },
    mode: 'real', endedAt: 1, reason: 'eos' }
  const good = parseRunFile(JSON.stringify({ formatVersion: 1, events, meta: { ...meta, format: { kind: 'chat', system: null } } }))
  expect(good.ok && good.meta.format).toEqual({ kind: 'chat', system: null })
  const bad = parseRunFile(JSON.stringify({ formatVersion: 1, events, meta: { ...meta, format: { kind: 'chat', system: 3 } } }))
  expect(bad.ok && bad.meta.format).toBeUndefined()
  const old = parseRunFile(JSON.stringify({ formatVersion: 1, events, meta }))
  expect(old.ok && old.meta.format).toBeUndefined()
})

test('a tokenize event with a damaged chat span loads without chat and says so; a valid one is kept', () => {
  const meta = { prompt: 'What is the capital of France?', params: { temperature: 0, topK: 1, maxNewTokens: 8 },
    mode: 'real', endedAt: 1, reason: 'eos' }
  const load = (chat: unknown) => {
    const events = buildFixtureTrace({ chat: true, mode: 'real' }).map((e) => e.type === 'tokenize' ? { ...e, chat } : e)
    return parseRunFile(JSON.stringify({ formatVersion: 1, events, meta }))
  }
  const n = buildFixtureTrace({ chat: true, mode: 'real' }).find((e) => e.type === 'tokenize')!
  const len = n.type === 'tokenize' ? n.tokens.length : 0
  const orig = n.type === 'tokenize' ? n.chat : undefined
  const kept = load(orig)
  expect(kept.ok && kept.events.find((e) => e.type === 'tokenize' && e.chat)).toBeTruthy()
  expect(kept.ok && kept.note).toBeFalsy()
  const bad: unknown[] = [
    { system: [1, 3] }, 'x', null, { user: [3, 3, 3] }, { user: [4, 2] }, { user: [0, len + 1] },
    { user: [1.5, 2] }, { user: [3, 4], system: [3, 3] }, { user: [3, 4], system: [0, 2] }, { user: [3, 4], system: [1, 9] },
  ]
  for (const chat of bad) {
    const r = load(chat)
    expect(r.ok).toBe(true)
    if (r.ok) {
      const t = r.events.find((e) => e.type === 'tokenize')
      expect(t && 'chat' in t).toBe(false)
      expect(r.note).toContain('Chat template data of 1 cycle was damaged')
    }
  }
})
