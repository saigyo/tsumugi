import { expect, test } from 'vitest'
import { buildFixtureTrace } from '../../test/fixtures'
import type { TraceEvent } from '../../trace/types'
import { alignRuns, combinedDistribution, pairedDistributions, pairedHeads } from './compareSelectors'

test('identical runs: same prompt, no fork, cycle counts', () => {
  const r = alignRuns(buildFixtureTrace(), buildFixtureTrace())
  expect(r.samePrompt).toBe(true)
  expect(r.forkCycle).toBeNull()
  expect(r.maxCycles).toBe(2)
  expect(r.chosenA.map((t) => t.text)).toEqual([' sat', ' on'])
})

test('divergent chosen tokens: fork at the first differing cycle', () => {
  const a = buildFixtureTrace()
  const b = buildFixtureTrace({ chosen: [{ id: 100, text: ' sat' }, { id: 999, text: ' off' }] })
  const r = alignRuns(a, b)
  expect(r.samePrompt).toBe(true)
  expect(r.forkCycle).toBe(1)
})

test('different prompts: no fork marker even when outputs differ', () => {
  const b = buildFixtureTrace({ prompt: 'A dog', promptTokens: [{ id: 20, text: 'A' }, { id: 21, text: ' dog' }],
    chosen: [{ id: 500, text: ' ran' }, { id: 501, text: ' far' }] })
  const r = alignRuns(buildFixtureTrace(), b)
  expect(r.samePrompt).toBe(false)
  expect(r.forkCycle).toBeNull()
})

test('length mismatch: maxCycles covers the longer run, no fork on an equal prefix', () => {
  const r = alignRuns(buildFixtureTrace(), buildFixtureTrace({ cycles: 1 }))
  expect(r.maxCycles).toBe(2)
  expect(r.chosenB).toHaveLength(1)
  expect(r.forkCycle).toBeNull()
})

test('pairedHeads unions by (layer, head) in order', () => {
  const attn = (cycle: number, heads: Array<[number, number]>): TraceEvent => ({
    type: 'attention', cycle,
    heads: heads.map(([layer, head]) => ({ layer, head, label: 'previous-token' as const, matrix: [[1]] })),
  })
  const a: TraceEvent[] = [attn(0, [[0, 3], [2, 1]])]
  const b: TraceEvent[] = [attn(0, [[2, 1], [5, 0]])]
  const pairs = pairedHeads(a, b, 0)
  expect(pairs.map((p) => [p.layer, p.head])).toEqual([[0, 3], [2, 1], [5, 0]])
  expect(pairs[0].a).toBeDefined(); expect(pairs[0].b).toBeUndefined()
  expect(pairs[1].a).toBeDefined(); expect(pairs[1].b).toBeDefined()
  expect(pairs[2].a).toBeUndefined(); expect(pairs[2].b).toBeDefined()
})

test('pairedDistributions returns per-side data, undefined past a run end', () => {
  const short = buildFixtureTrace({ cycles: 1 })
  const d = pairedDistributions(buildFixtureTrace(), short, 1)
  expect(d.a?.sample.chosen.text).toBe(' on')
  expect(d.b).toBeUndefined()
})

test('combinedDistribution merges identical top-k lists with zero deltas', () => {
  const c = combinedDistribution(buildFixtureTrace(), buildFixtureTrace(), 0)
  expect(c).toBeDefined()
  if (!c) return
  expect(c.rows.map((r) => [r.text, r.pA, r.pB, r.delta, r.approx])).toEqual([
    [' sat', 0.7, 0.7, 0, false], [' ran', 0.2, 0.2, 0, false], [' was', 0.1, 0.1, 0, false],
  ])
  expect(c.chosenA).toBe(100)
  expect(c.chosenB).toBe(100)
})

test('combinedDistribution marks one-sided tokens approximate with 0 on the missing side', () => {
  const b = buildFixtureTrace({ chosen: [{ id: 100, text: ' sat' }, { id: 999, text: ' off' }] })
  const c = combinedDistribution(buildFixtureTrace(), b, 1)
  expect(c).toBeDefined()
  if (!c) return
  const byId = new Map(c.rows.map((r) => [r.id, r]))
  expect(byId.get(101)).toMatchObject({ pA: 0.7, pB: null, delta: 0.7, approx: true })   // ' on', A only
  expect(byId.get(999)).toMatchObject({ pA: null, pB: 0.7, delta: -0.7, approx: true })  // ' off', B only
  expect(byId.get(200)).toMatchObject({ pA: 0.2, pB: 0.2, delta: 0, approx: false })
  expect(c.chosenA).toBe(101)
  expect(c.chosenB).toBe(999)
})

test('combinedDistribution is undefined when either side lacks the cycle', () => {
  expect(combinedDistribution(buildFixtureTrace(), buildFixtureTrace({ cycles: 1 }), 1)).toBeUndefined()
})
