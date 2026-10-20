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
