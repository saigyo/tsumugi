import { expect, test } from 'vitest'
import type { TokenInfo } from '../../trace/types'
import { inductionRow, prevTokenRow, proceduralHeads, sinkRow } from './attention'

const toks = (...texts: string[]): TokenInfo[] => texts.map((text, id) => ({ id, text }))

const sumsToOne = (row: number[]) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9

test('all row builders are causal and row-stochastic', () => {
  const tokens = toks('one', ' two', ' three', ' one', ' two')
  for (let i = 0; i < tokens.length; i++) {
    for (const row of [prevTokenRow(i), sinkRow(i), inductionRow(i, tokens)]) {
      expect(row).toHaveLength(i + 1)
      expect(sumsToOne(row)).toBe(true)
    }
  }
})

test('previous-token row concentrates on i-1', () => {
  const row = prevTokenRow(4)
  expect(row[3]).toBeGreaterThan(0.5)
})

test('sink row concentrates on position 0', () => {
  const row = sinkRow(4)
  expect(row[0]).toBeGreaterThan(0.5)
})

test('induction row attends to the token after the previous occurrence', () => {
  const tokens = toks('one', ' two', ' three', 'one')  // idx 3 repeats idx 0
  const row = inductionRow(3, tokens)
  expect(row[1]).toBeGreaterThan(0.5)  // " two" followed the first "one"
})

test('induction row falls back locally when no repeat exists', () => {
  const tokens = toks('a', ' b', ' c')
  const row = inductionRow(2, tokens)
  expect(row[1]).toBeGreaterThanOrEqual(0.5)  // previous token dominates
})

test('proceduralHeads returns three labeled heads with full matrices', () => {
  const tokens = toks('The', ' cat', ' sat')
  const heads = proceduralHeads(tokens)
  expect(heads.map((h) => h.label).sort()).toEqual(['attention-sink', 'induction', 'previous-token'])
  for (const h of heads) {
    expect(h.matrix).toHaveLength(3)
    h.matrix.forEach((row, i) => expect(row).toHaveLength(i + 1))
  }
})
