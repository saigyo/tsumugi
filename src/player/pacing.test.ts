import { expect, test } from 'vitest'
import { BASE_MS, delayFor } from './pacing'

test('layer events are much quicker than sample events', () => {
  const layer = delayFor({ type: 'layer', cycle: 0, index: 0, total: 12 }, 1)
  const sample = delayFor({ type: 'sample', cycle: 0, chosen: { id: 1, text: 'a' }, method: 'top-k' }, 1)
  expect(sample).toBeGreaterThan(layer * 4)
})

test('speed divides delay', () => {
  const e = { type: 'tokenize', tokens: [] } as const
  expect(delayFor(e, 2)).toBeCloseTo(delayFor(e, 1) / 2)
})

test('undefined event gets base delay', () => {
  expect(delayFor(undefined, 1)).toBe(BASE_MS)
})
