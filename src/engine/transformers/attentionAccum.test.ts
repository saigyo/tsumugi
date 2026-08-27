import { expect, test } from 'vitest'
import { addAttentionOutput, createAccumulator } from './attentionAccum'

test('prefill appends the full causal triangle', () => {
  const acc = createAccumulator(1, 2)
  // batch 1, 2 heads, 3 query rows, 3 kv — rows may carry zeros above the diagonal
  const head0 = [1, 0, 0, /**/ 0.5, 0.5, 0, /**/ 0.2, 0.3, 0.5]
  const head1 = head0.map((v) => v)
  addAttentionOutput(acc, 0, [1, 2, 3, 3], Float32Array.from([...head0, ...head1]))
  expect(acc.rows[0][0]).toHaveLength(3)
  expect(acc.rows[0][0][0]).toEqual([1])            // causal truncation
  expect(acc.rows[0][0][2][0]).toBeCloseTo(0.2)
  expect(acc.rows[0][0][2][1]).toBeCloseTo(0.3)
  expect(acc.rows[0][0][2][2]).toBeCloseTo(0.5)
})

test('a decode step appends one row per head', () => {
  const acc = createAccumulator(1, 1)
  addAttentionOutput(acc, 0, [1, 1, 2, 2], Float32Array.from([1, 0, 0.4, 0.6]))
  addAttentionOutput(acc, 0, [1, 1, 1, 3], Float32Array.from([0.1, 0.2, 0.7]))
  expect(acc.rows[0][0]).toHaveLength(3)
  expect(acc.rows[0][0][2][0]).toBeCloseTo(0.1)
  expect(acc.rows[0][0][2][1]).toBeCloseTo(0.2)
  expect(acc.rows[0][0][2][2]).toBeCloseTo(0.7)
})

test('no-cache full matrices are consumed by keeping only new rows', () => {
  const acc = createAccumulator(1, 1)
  addAttentionOutput(acc, 0, [1, 1, 2, 2], Float32Array.from([1, 0, 0.4, 0.6]))
  // Approach B: next step re-sends the FULL 3×3 matrix; only row 2 is new
  addAttentionOutput(acc, 0, [1, 1, 3, 3], Float32Array.from([1, 0, 0, 0.4, 0.6, 0, 0.1, 0.2, 0.7]))
  expect(acc.rows[0][0]).toHaveLength(3)
  expect(acc.rows[0][0][2][0]).toBeCloseTo(0.1)
  expect(acc.rows[0][0][2][1]).toBeCloseTo(0.2)
  expect(acc.rows[0][0][2][2]).toBeCloseTo(0.7)
})
