import { expect, test } from 'vitest'
import type { TokenInfo } from '../../trace/types'
import { createAccumulator } from './attentionAccum'
import { headStats, selectShowcaseHeads } from './attentionStats'

const toks = (...t: string[]): TokenInfo[] => t.map((text, id) => ({ id, text }))

function fill(rowsFor: (i: number) => number[], n: number) {
  return Array.from({ length: n }, (_, i) => rowsFor(i))
}
const diagRow = (i: number) => i === 0 ? [1] : [...Array(i - 1).fill(0), 1, 0]
const sinkRow = (i: number) => i === 0 ? [1] : [1, ...Array(i).fill(0)]

test('scores identify diagonal and sink heads', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = fill(diagRow, 5)
  acc.rows[1][0] = fill(sinkRow, 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  expect(stats[0].prevTokenScore).toBe(1)
  expect(stats[1].sinkScore).toBe(1)
  expect(stats[0].inductionScore).toBeNull()   // no repeated tokens
})

test('induction score measured only on repeat rows', () => {
  const acc = createAccumulator(1, 1)
  // tokens: a b a — row 2 repeats row 0's token; induction target = col 1
  acc.rows[0][0] = [[1], [0.5, 0.5], [0, 1, 0]]
  const stats = headStats(acc, toks('a', ' b', 'a'))
  expect(stats[0].inductionScore).toBe(1)
})

test('whitespace-only tokens yield no induction targets', () => {
  const acc = createAccumulator(1, 1)
  // tokens: '\n' ' ' '\n' — whitespace collapses to '' under trim(), which
  // must not be treated as a repeat (previously '\n'.trim() === ' '.trim()
  // both being '' made every whitespace row an induction "hit").
  acc.rows[0][0] = [[1], [0.5, 0.5], [0.34, 0.33, 0.33]]
  const stats = headStats(acc, toks('\n', ' ', '\n'))
  expect(stats[0].inductionScore).toBeNull()
})

test('selectShowcaseHeads picks top head per label above threshold', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = fill(diagRow, 5)
  acc.rows[1][0] = fill(sinkRow, 5)
  const heads = selectShowcaseHeads(headStats(acc, toks('a', ' b', ' c', ' d', ' e')), acc)
  expect(heads.map((h) => h.label).sort()).toEqual(['attention-sink', 'previous-token'])
  const prev = heads.find((h) => h.label === 'previous-token')!
  expect(prev.layer).toBe(0)
  expect(prev.score).toBe(1)
  expect(prev.matrix).toHaveLength(5)
})

test('heads below threshold are not selected', () => {
  const acc = createAccumulator(1, 1)
  // Create rows with low scores: scatter attention evenly across positions
  acc.rows[0][0] = [
    [1],
    [0.2, 0.8],
    [0.2, 0.2, 0.6],
    [0.2, 0.2, 0.2, 0.4],
    [0.2, 0.2, 0.2, 0.2, 0.2],
  ]
  const heads = selectShowcaseHeads(headStats(acc, toks('a', ' b', ' c', ' d', ' e')), acc)
  expect(heads).toHaveLength(0)
})
