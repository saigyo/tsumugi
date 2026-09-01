import { expect, test } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace, makeGridEvent } from '../test/fixtures'
import { validateTrace } from './validate'
import type { TraceEvent } from './types'

test('fixture trace is valid', () => {
  expect(validateTrace(makeFixtureTrace())).toEqual([])
})

test('missing run-start is flagged', () => {
  expect(validateTrace(makeFixtureTrace().slice(1)).length).toBeGreaterThan(0)
})

test('out-of-order layer indices are flagged', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((e) => e.type === 'layer')
  const j = t.findIndex((e, k) => e.type === 'layer' && k > i)
  ;[t[i], t[j]] = [t[j], t[i]]
  expect(validateTrace(t).some((v) => v.includes('layer'))).toBe(true)
})

test('softmax probs must sum to 1', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'softmax')
  if (e?.type === 'softmax') e.topK = e.topK.map((c) => ({ ...c, prob: c.prob * 2 }))
  expect(validateTrace(t).some((v) => v.includes('softmax'))).toBe(true)
})

test('attention rows must be causal and sum to 1', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'attention')
  if (e?.type === 'attention') e.heads[0].matrix[1] = [0.9, 0.9]
  expect(validateTrace(t).some((v) => v.includes('attention'))).toBe(true)
})

test('attention outside the post-layer slot is flagged', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((e) => e.type === 'attention')
  const [att] = t.splice(i, 1)
  t.splice(2, 0, att)  // before embed
  expect(validateTrace(t).some((v) => v.includes('attention'))).toBe(true)
})

test('default fixture has no attention-grid event', () => {
  expect(makeFixtureTrace().some((e) => e.type === 'attention-grid')).toBe(false)
})

test('attention-grid directly before run-end is valid', () => {
  const t = makeFixtureTrace()
  t.splice(t.length - 1, 0, makeGridEvent())
  expect(validateTrace(t)).toEqual([])
})

test('attention-grid away from run-end is flagged', () => {
  const t = makeFixtureTrace()
  t.splice(2, 0, makeGridEvent())
  expect(validateTrace(t).some((v) => v.includes('attention-grid'))).toBe(true)
})

test('attention-grid cell count must be layers × heads', () => {
  const t = makeFixtureTrace()
  const g = makeGridEvent(2, 2)
  g.cells.pop()
  t.splice(t.length - 1, 0, g)
  expect(validateTrace(t).some((v) => v.includes('cells'))).toBe(true)
})

test('attention-grid thumbs must be ≤12×12 with values in [0,1]', () => {
  const t = makeFixtureTrace()
  const g = makeGridEvent(2, 2)
  g.cells[0].thumb[0][0] = 1.5
  g.cells[1].thumb = Array.from({ length: 13 }, () => Array.from({ length: 13 }, () => 0))
  t.splice(t.length - 1, 0, g)
  const errs = validateTrace(t)
  expect(errs.some((v) => v.includes('[0, 1]'))).toBe(true)
  expect(errs.some((v) => v.includes('12'))).toBe(true)
})

test('embed rows must be dims-long finite numbers', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'embed')
  if (e?.type === 'embed') { e.source = 'model'; e.rows = [[0.1, 0.2]] }   // dims is 576
  expect(validateTrace(t).some((v) => v.includes('embed'))).toBe(true)
})

test('model-source embed without rows is flagged', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'embed')
  if (e?.type === 'embed') e.source = 'model'
  expect(validateTrace(t).some((v) => v.includes('embed'))).toBe(true)
})

test('model-source embed with well-formed rows is valid', () => {
  expect(validateTrace(buildFixtureTrace({ embedRows: true }))).toEqual([])
})

test('legacy embed events carrying preview and no source still validate', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((x) => x.type === 'embed')
  t[i] = { type: 'embed', cycle: 0, seqLen: 2, dims: 576, preview: [[0.1, 0.2]] } as unknown as TraceEvent
  expect(validateTrace(t)).toEqual([])
})
