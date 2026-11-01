import { expect, test } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../test/fixtures'
import type { KnockoutSweep } from '../trace/types'
import { runSummary } from './runSummary'

// makeFixtureTrace(): 0 run-start, 1 tokenize, cycle 0 = 2..10 (embed 2, layers 3-5,
// attention 6, logits 7, softmax 8, sample 9, append 10), cycle 1 = 11..19, 20 run-end

test('no trace, no summary', () => {
  expect(runSummary([], -1)).toBeNull()
})

test('summarises the run at the viewed cycle', () => {
  const r = runSummary(makeFixtureTrace(), 8)!
  expect(r.mode).toBe('sim')
  expect(r.modelId).toBe('fixture')
  expect(r.params).toEqual({ temperature: 0.8, topK: 10, maxNewTokens: 2 })
  expect(r.tokens.map((t) => t.text)).toEqual(['The', ' cat'])
  expect(r.cycle).toBe(0)
  expect(r.cycles).toBe(2)
  expect(r.stream).toEqual({ seqLen: 2, dims: 576, layers: 3 })
  expect(r.vocabSize).toBe(49152)
  expect(r.softmax?.temperature).toBe(0.8)
  expect(r.softmax?.top[0]).toEqual({ id: 100, text: ' sat', logit: 9.1, prob: 0.7 })
})

test('follows the cursor into the next cycle', () => {
  const events = makeFixtureTrace()
  const softmax1 = events.findIndex((e) => e.type === 'softmax' && e.cycle === 1)
  const r = runSummary(events, softmax1)!
  expect(r.cycle).toBe(1)
  expect(r.stream?.seqLen).toBe(3)
  expect(r.softmax?.top[0].text).toBe(' on')
})

test('before the first softmax there is no softmax block', () => {
  const r = runSummary(buildFixtureTrace({ temperature: 1 }), 2)!
  expect(r.softmax).toBeUndefined()
  expect(r.stream).toEqual({ seqLen: 2, dims: 576, layers: 3 })
})

test('MLP facts: the last token’s largest MLP edit, and the MLP layer that added most to the top candidate', () => {
  const t = buildFixtureTrace({ mlp: true, mode: 'real' })
  const s = runSummary(t, 4)!
  expect(s.edits?.token).toEqual({ id: 11, text: ' cat' })
  expect(s.edits?.ratio).toBeGreaterThan(0)
  expect(s.votes?.candidate).toEqual({ id: 100, text: ' sat' })
  expect(s.votes?.layer).toBeGreaterThanOrEqual(0)
  expect(runSummary(buildFixtureTrace(), 4)!.edits).toBeUndefined()
})

test('a finished run says how many tokens it wrote and why it stopped', () => {
  const r = runSummary(makeFixtureTrace(), 8)!
  expect(r.generated).toBe(2)
  expect(r.ended).toBe('max-tokens')
  const chat = buildFixtureTrace({ chat: true })
  expect(runSummary(chat, 2)!.ended).toBe('eos')
})

test('a run still in flight has no stop reason yet', () => {
  const events = makeFixtureTrace().slice(0, -1)
  const r = runSummary(events, events.length - 1)!
  expect(r.ended).toBeUndefined()
  expect(r.generated).toBe(2)
})

const lay = (layer: number, prob: number, flips = false) => ({ layer, prob, top: { id: 1, text: 'x', prob }, flips })
const sweepAt = (cycle: number): KnockoutSweep => ({ cycle, revision: 'v7', chosen: { id: 100, text: ' sat' }, baseline: 0.7,
  layers: [lay(0, 0.01, true), lay(1, 0.69), lay(2, 0.2), lay(3, 0.75), lay(4, 0.1, true)] })

test('without sweeps, or none for the viewed cycle, there is no knockout summary', () => {
  expect(runSummary(makeFixtureTrace(), 8)!.knockout).toBeUndefined()
  expect(runSummary(makeFixtureTrace(), 8, [sweepAt(1)])!.knockout).toBeUndefined()
})

test('the sweep of the viewed cycle is summarised: flips, largest drop and largest rise', () => {
  const k = runSummary(makeFixtureTrace(), 8, [sweepAt(1), sweepAt(0)])!.knockout!
  expect(k).toMatchObject({ chosen: { id: 100 }, baseline: 0.7, layers: 5, flipped: 2, drop: { layer: 0 } })
  expect(k.drop!.delta).toBeCloseTo(0.69)
  expect(k.rise!.layer).toBe(3)
  expect(k.rise!.delta).toBeCloseTo(0.05)
})

test('a sweep with no rise has no rise entry', () => {
  const sw = { ...sweepAt(0), layers: [lay(0, 0.2), lay(1, 0.7)] }
  const k = runSummary(makeFixtureTrace(), 8, [sw])!.knockout!
  expect(k.rise).toBeUndefined()
  expect(k.drop).toMatchObject({ layer: 0 })
})
