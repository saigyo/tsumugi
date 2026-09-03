import { expect, test } from 'vitest'
import type { TokenInfo } from '../../trace/types'
import { createAccumulator } from './attentionAccum'
import { headStats, resolveHeadLabel, selectShowcaseHeads, type ShowcasePrev } from './attentionStats'

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

test('distinctive score: template heads low, focused-untemplated head high, uniform head low', () => {
  const acc = createAccumulator(3, 1)
  // template: perfect previous-token head
  acc.rows[0][0] = fill(diagRow, 6)
  // focused but untemplated: rows 3.. lock onto position 1 (not prev, not sink)
  acc.rows[1][0] = [
    [1], [0.5, 0.5], [1 / 3, 1 / 3, 1 / 3],
    [0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0],
  ]
  // uniform: attention spread evenly
  acc.rows[2][0] = Array.from({ length: 6 }, (_, i) =>
    Array.from({ length: i + 1 }, () => 1 / (i + 1)))
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e', ' f'))
  expect(stats[0].distinctiveScore).toBe(0)                 // templateMax = 1
  expect(stats[1].distinctiveScore).toBeGreaterThan(0.4)    // ≈ 0.83 × 0.6
  expect(stats[2].distinctiveScore).toBeLessThan(0.01)      // uniformity ≈ 1
})

test('distinctive score is 0 for single-row heads', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = [[1]]
  expect(headStats(acc, toks('a'))[0].distinctiveScore).toBe(0)
})

test('fourth chip: a distinctive head above 0.25 is selected', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = [
    [1], [0.5, 0.5], [1 / 3, 1 / 3, 1 / 3],
    [0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0],
  ]
  const heads = selectShowcaseHeads(headStats(acc, toks('a', ' b', ' c', ' d', ' e', ' f')), acc)
  const d = heads.find((h) => h.label === 'distinctive')
  expect(d).toBeDefined()
  expect(d!.score).toBeGreaterThan(0.25)
})

test('hysteresis: incumbent sticks under a <0.05 challenger lead', () => {
  const acc = createAccumulator(2, 1)
  // incumbent L0: prev-token score 0.96; challenger L1: 0.99 — lead 0.03 < 0.05
  const nearDiag = (p: number) => (i: number) =>
    i === 0 ? [1] : [...Array(Math.max(0, i - 1)).fill(0), p, 1 - p]
  acc.rows[0][0] = fill(nearDiag(0.96), 5)
  acc.rows[1][0] = fill(nearDiag(0.99), 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  const prev: ShowcasePrev = { 'previous-token': { layer: 0, head: 0 } }
  const kept = selectShowcaseHeads(stats, acc, 0.3, prev).find((h) => h.label === 'previous-token')
  expect(kept!.layer).toBe(0)
  // without prev, argmax wins
  const argmax = selectShowcaseHeads(stats, acc).find((h) => h.label === 'previous-token')
  expect(argmax!.layer).toBe(1)
})

test('hysteresis: challenger wins at a >=0.05 lead', () => {
  const acc = createAccumulator(2, 1)
  const nearDiag = (p: number) => (i: number) =>
    i === 0 ? [1] : [...Array(Math.max(0, i - 1)).fill(0), p, 1 - p]
  acc.rows[0][0] = fill(nearDiag(0.9), 5)
  acc.rows[1][0] = fill(nearDiag(0.96), 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  const prev: ShowcasePrev = { 'previous-token': { layer: 0, head: 0 } }
  const winner = selectShowcaseHeads(stats, acc, 0.3, prev).find((h) => h.label === 'previous-token')
  expect(winner!.layer).toBe(1)
})

test('hysteresis: incumbent below threshold falls back to argmax', () => {
  const acc = createAccumulator(2, 1)
  // incumbent decayed to uniform: prevTokenScore = mean(1/2..1/6) ≈ 0.29 < 0.3
  acc.rows[0][0] = Array.from({ length: 6 }, (_, i) =>
    Array.from({ length: i + 1 }, () => 1 / (i + 1)))
  // challenger at 0.32: above threshold, but its lead over 0.29 is < 0.05 —
  // it must still win, because a sub-threshold incumbent loses its seat
  const nearDiag = (p: number) => (i: number) =>
    i === 0 ? [1] : [...Array(Math.max(0, i - 1)).fill(0), p, 1 - p]
  acc.rows[1][0] = fill(nearDiag(0.32), 6)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e', ' f'))
  const prev: ShowcasePrev = { 'previous-token': { layer: 0, head: 0 } }
  const winner = selectShowcaseHeads(stats, acc, 0.3, prev).find((h) => h.label === 'previous-token')
  expect(winner!.layer).toBe(1)
})

test('resolveHeadLabel returns the best template at or above 0.3', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = fill(diagRow, 5)
  acc.rows[1][0] = fill(sinkRow, 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  expect(resolveHeadLabel(stats, 0, 0)).toEqual({ label: 'previous-token', score: 1 })
  expect(resolveHeadLabel(stats, 1, 0)).toEqual({ label: 'attention-sink', score: 1 })
})

test('resolveHeadLabel returns nulls below threshold or for unknown heads', () => {
  const acc = createAccumulator(1, 1)
  // NOTE: brief's original fixture here used the uniform 1/(i+1) rows (as in
  // the "distinctive score is 0 for uniform" test), but that yields
  // prevTokenScore = sinkScore ≈ 0.3208 — ABOVE the 0.3 threshold, so
  // resolveHeadLabel correctly returns a non-null label for it. Reused
  // instead the low, evenly-scattered rows from "heads below threshold are
  // not selected" above, which genuinely score 0.2 on every template.
  acc.rows[0][0] = [
    [1],
    [0.2, 0.8],
    [0.2, 0.2, 0.6],
    [0.2, 0.2, 0.2, 0.4],
    [0.2, 0.2, 0.2, 0.2, 0.2],
  ]
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  expect(resolveHeadLabel(stats, 0, 0)).toEqual({ label: null, score: null })
  expect(resolveHeadLabel(stats, 7, 7)).toEqual({ label: null, score: null })
})

// --- coreference template (backlog #5, docs/research/2026-09-03-coreference-head-spike.md)

// rows 0..3 previous-token; row 4 (" it") peaks on " cat" at column 1
const corefRows = (pronounRow: number[]) => [
  [1], [1, 0], [0, 1, 0], [0, 0, 1, 0], pronounRow,
]
const corefToks = toks('The', ' cat', ' sat', ' because', ' it')

test('corefScore: the pronoun row peaked on an earlier word token', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = corefRows([0.05, 0.8, 0.05, 0.1, 0])
  const [s] = headStats(acc, corefToks)
  expect(s.corefScore).toBeCloseTo(0.8)
})

test('corefScore ignores the sink, previous-token and self columns', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = corefRows([0.5, 0, 0, 0.5, 0])   // sink + prev only
  acc.rows[1][0] = corefRows([0, 0, 0, 0, 1])       // self-attention
  const stats = headStats(acc, corefToks)
  expect(stats[0].corefScore).toBe(0)
  expect(stats[1].corefScore).toBe(0)
})

test('corefScore ignores punctuation and single-letter columns', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = corefRows([0, 0.1, 0.6, 0.3, 0])
  const [s] = headStats(acc, toks('The', ' cat', ',', ' a', ' it'))
  expect(s.corefScore).toBeCloseTo(0.1)
})

test('corefScore is null without a pronoun row at index >= 2, and is case-insensitive', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = fill(diagRow, 5)
  expect(headStats(acc, toks('a', ' b', ' c', ' d', ' e'))[0].corefScore).toBeNull()
  expect(headStats(acc, toks('It', ' was', ' c', ' d', ' e'))[0].corefScore).toBeNull()
  acc.rows[0][0] = corefRows([0, 0.7, 0.3, 0, 0])
  expect(headStats(acc, toks('The', ' cat', ' sat', '.', ' It'))[0].corefScore).toBeCloseTo(0.7)
})

test('corefScore averages over several pronoun rows', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = [[1], [1, 0], [0, 1, 0], [0, 0.6, 0.4, 0], [0, 0.2, 0.8, 0, 0]]
  // rows 3 (" it") and 4 (" she"): peaks 0.6 on " cat" and 0.8 on " sat"
  const [s] = headStats(acc, toks('The', ' cat', ' sat', ' it', ' she'))
  expect(s.corefScore).toBeCloseTo(0.7)
})

test('coreference is a template: it suppresses the distinctive score', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = [
    [1], [0.5, 0.5], [1 / 3, 1 / 3, 1 / 3],
    [0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0],
  ]
  const [s] = headStats(acc, toks('The', ' cat', ' sat', ' it', ' it', ' it'))
  expect(s.corefScore).toBe(1)
  expect(s.distinctiveScore).toBe(0)
})

test('showcase selects and labels a coreference head; resolveHeadLabel agrees', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = fill(sinkRow, 5)
  acc.rows[1][0] = corefRows([0.05, 0.8, 0.05, 0.1, 0])
  const stats = headStats(acc, corefToks)
  const coref = selectShowcaseHeads(stats, acc).find((h) => h.label === 'coreference')
  expect(coref).toBeDefined()
  expect(coref!.layer).toBe(1)
  expect(coref!.score).toBe(0.8)
  expect(resolveHeadLabel(stats, 1, 0)).toEqual({ label: 'coreference', score: 0.8 })
})
