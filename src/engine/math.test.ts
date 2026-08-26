import { expect, test } from 'vitest'
import { sampleIndex, softmax, topK } from './math'

test('softmax sums to 1 and preserves order', () => {
  const p = softmax([3, 1, 0.5], 1)
  expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  expect(p[0]).toBeGreaterThan(p[1])
})

test('temperature 0 is one-hot argmax', () => {
  expect(softmax([1, 5, 2], 0)).toEqual([0, 1, 0])
})

test('low temperature sharpens the distribution', () => {
  const sharp = softmax([3, 1], 0.5)
  const soft = softmax([3, 1], 2)
  expect(sharp[0]).toBeGreaterThan(soft[0])
})

test('topK returns k best ids sorted desc', () => {
  expect(topK([0.1, 9, 3, 7], 2)).toEqual([
    { id: 1, logit: 9 },
    { id: 3, logit: 7 },
  ])
})

test('sampleIndex picks by cumulative probability', () => {
  expect(sampleIndex([0.2, 0.5, 0.3], () => 0.1)).toBe(0)
  expect(sampleIndex([0.2, 0.5, 0.3], () => 0.6)).toBe(1)
  expect(sampleIndex([0.2, 0.5, 0.3], () => 0.99)).toBe(2)
})
