import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { validateTrace } from './validate'

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
