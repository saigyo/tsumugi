import { expect, test } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../test/fixtures'
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
  expect(r.softmax?.top[0]).toEqual({ text: ' sat', logit: 9.1, prob: 0.7 })
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
  expect(s.edits?.token).toBe(' cat')
  expect(s.edits?.ratio).toBeGreaterThan(0)
  expect(s.votes?.candidate).toBe(' sat')
  expect(s.votes?.layer).toBeGreaterThanOrEqual(0)
  expect(runSummary(buildFixtureTrace(), 4)!.edits).toBeUndefined()
})
