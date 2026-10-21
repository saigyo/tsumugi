import { expect, test } from 'vitest'
import { foldMatrix } from './foldMatrix'

// 5 positions; row i spreads its weight evenly over 0..i
const M = Array.from({ length: 5 }, (_, i) => Array.from({ length: i + 1 }, () => 1 / (i + 1)))

test('without a span the matrix is unchanged, one column per position', () => {
  const f = foldMatrix(M, undefined)
  expect(f.rows).toEqual([0, 1, 2, 3, 4])
  expect(f.cols).toEqual([0, 1, 2, 3, 4].map((i) => ({ from: i, to: i + 1 })))
  expect(f.values).toEqual(M)
})

test('the span’s rows are hidden and its columns become one summed column after column 0', () => {
  const f = foldMatrix(M, [1, 3])
  expect(f.rows).toEqual([0, 3, 4])
  expect(f.cols).toEqual([{ from: 0, to: 1 }, { from: 1, to: 3 }, { from: 3, to: 4 }, { from: 4, to: 5 }])
  expect(f.values[0]).toEqual([1])                       // row 0 sees only column 0
  expect(f.values[1]).toEqual([0.25, 0.5, 0.25])         // row 3: col 0, cols 1–2 summed, col 3
  expect(f.values[2].reduce((a, b) => a + b, 0)).toBeCloseTo(1)
})

test('a matrix shorter than the span is not folded and never read past its rows', () => {
  const short = M.slice(0, 2)
  const f = foldMatrix(short, [1, 3])
  expect(f.rows).toEqual([0, 1])
  expect(f.values).toEqual(short)
})
