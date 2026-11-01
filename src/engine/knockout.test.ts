import { expect, test } from 'vitest'
import { forcedToken, mlpScale, sweepEntry } from './knockout'
import { sampleIndex, softmax, topK } from './math'

test('mlpScale with no knockout is all ones', () => {
  const s = mlpScale(30, null)
  expect(s).toBeInstanceOf(Float32Array)
  expect(s).toHaveLength(30)
  expect(Array.from(s).every((v) => v === 1)).toBe(true)
})

test('mlpScale zeroes only the knocked-out layer', () => {
  const s = mlpScale(30, 15)
  expect(s[15]).toBe(0)
  expect(Array.from(s).filter((v) => v === 0)).toHaveLength(1)
})

test('mlpScale ignores an out-of-range layer', () => {
  expect(Array.from(mlpScale(4, 9)).every((v) => v === 1)).toBe(true)
})

test('forcedToken returns the copied prefix then null', () => {
  expect(forcedToken([5, 9], 0)).toBe(5)
  expect(forcedToken([5, 9], 1)).toBe(9)
  expect(forcedToken([5, 9], 2)).toBeNull()
})

const P = { temperature: 1, topK: 3, maxNewTokens: 5 }
const logits = Float32Array.from([0.5, 3, 1, 2, -1])  // ranking: 1, 3, 2

test('sweepEntry: chosen at the top keeps its softmax share and does not flip', () => {
  const e = sweepEntry(logits, P, 1, 0.01)
  const probs = softmax([3, 2, 1], 1)
  expect(e.topId).toBe(1)
  expect(e.prob).toBeCloseTo(probs[0], 12)
  expect(e.topProb).toBeCloseTo(probs[0], 12)
  expect(e.flips).toBe(false)
})

test('sweepEntry: chosen outside top-k has prob 0 and flips', () => {
  const e = sweepEntry(logits, P, 4, 0.5)
  expect(e.prob).toBe(0)
  expect(e.flips).toBe(true)
})

test('sweepEntry: temperature 0 uses 1 for prob and the top id for flips', () => {
  const g = { ...P, temperature: 0 }
  const e = sweepEntry(logits, g, 3, 0.99)
  expect(e.prob).toBeCloseTo(softmax([3, 2, 1], 1)[1], 12)
  expect(e.flips).toBe(true)
  expect(sweepEntry(logits, g, 1, 0.99).flips).toBe(false)
})

test('sweepEntry: with temperature the draw decides flips like sampleIndex', () => {
  const t = { ...P, temperature: 0.8 }
  const top = topK(logits, 3)
  const probs = softmax(top.map((c) => c.logit), 0.8)
  for (const draw of [0.01, 0.5, 0.95]) {
    const idx = sampleIndex(probs, () => draw)
    expect(sweepEntry(logits, t, 1, draw).flips).toBe(top[idx].id !== 1)
  }
})
