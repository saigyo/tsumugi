import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { activeStage, cycleTickIndices, eventAt, latestOfType, visibleTokens } from './selectors'

const trace = makeFixtureTrace()  // 2 cycles, 3 layers

test('activeStage maps events to stage cards', () => {
  expect(activeStage({ type: 'tokenize', tokens: [] })).toBe('tokenizer')
  expect(activeStage(eventAt(trace, 2))).toBe('embeddings')
  expect(activeStage({ type: 'softmax', cycle: 0, temperature: 1, topK: [] })).toBe('logits')
  expect(activeStage({ type: 'append', cycle: 0, token: { id: 1, text: 'x' } })).toBe('sampler')
  expect(activeStage(undefined)).toBe(null)
  expect(activeStage({ type: 'run-end', reason: 'eos' })).toBe(null)
})

test('visibleTokens grows with cursor', () => {
  expect(visibleTokens(trace, 0).prompt).toHaveLength(0)          // before tokenize
  expect(visibleTokens(trace, 1).prompt).toHaveLength(2)          // after tokenize
  expect(visibleTokens(trace, 1).generated).toHaveLength(0)
  expect(visibleTokens(trace, trace.length - 1).generated).toHaveLength(2)
})

test('latestOfType finds most recent event at or before cursor', () => {
  const last = trace.length - 1
  expect(latestOfType(trace, last, 'softmax')?.cycle).toBe(1)
  expect(latestOfType(trace, 5, 'tokenize')?.type).toBe('tokenize')
  expect(latestOfType(trace, 0, 'softmax')).toBeUndefined()
})

test('cycleTickIndices marks append events', () => {
  const ticks = cycleTickIndices(trace)
  expect(ticks).toHaveLength(2)
  expect(trace[ticks[0]].type).toBe('append')
})
