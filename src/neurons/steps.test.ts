import { expect, test } from 'vitest'
import { contribution, editRatio, gateCounts, gateDomain, normalize, productSummary, rms, silu } from './steps'
import type { NeuronLayer } from './format'

test('SiLU passes large values, squashes negatives to at least −0.28', () => {
  expect(silu(10)).toBeCloseTo(9.9995, 3)
  expect(silu(0)).toBe(0)
  expect(silu(-1.278)).toBeCloseTo(-0.2785, 3)
  expect(silu(-20)).toBeCloseTo(0, 6)
})

test('normalize divides by the root mean square (with eps) and applies the learned scale', () => {
  const { n, rms: r } = normalize(Float32Array.from([3, -4]), Float32Array.from([1, 2]), 0)
  expect(r).toBeCloseTo(Math.sqrt(12.5), 6)
  expect(Array.from(n).map((v) => +v.toFixed(4))).toEqual([+(3 / r).toFixed(4), +((-4 / r) * 2).toFixed(4)])
  expect(rms([3, -4])).toBeCloseTo(Math.sqrt(12.5), 6)
})

test('the gate axis always shows both sides of zero, padded', () => {
  const [lo, hi] = gateDomain(Float32Array.from([-4.1, -2, 1.7]))
  expect(lo).toBeLessThan(-4.1); expect(hi).toBeGreaterThan(1.7)
  const [lo2, hi2] = gateDomain(Float32Array.from([-3, -2.5]))     // all negative: zero and +1 still on the axis
  expect(lo2).toBeLessThan(-3); expect(hi2).toBeGreaterThanOrEqual(1)
  expect(gateCounts(Float32Array.from([-1, 0.5, -0.2, 3]))).toEqual({ negative: 2, positive: 2 })
})

test('the product summary orders by size and counts the neurons carrying half and 90 % of the square', () => {
  const h = Float32Array.from([0.1, -3, 0.2, 4, 1, -0.05])   // squares 0.01 9 0.04 16 1 0.0025, total 26.0525
  const s = productSummary(h, 3)
  expect(s.order).toEqual([3, 1, 4])
  expect(s.half).toBe(1)        // 16 / 26.05 = 0.61
  expect(s.ninety).toBe(2)      // 25 / 26.05 = 0.96
  expect(s.restMax).toBeCloseTo(0.2, 6)
  expect(productSummary(h, 10).restMax).toBeNull()
})

test('a contribution is the dequantized column times the activation', () => {
  const layer = { scales: Float32Array.from([0.5, 2]), columns: Int8Array.from([1, -2, 3, 4]),
    labelIds: new Uint16Array(), thresholds: new Float32Array(), normWeight: new Float32Array(2) } as NeuronLayer
  expect(Array.from(contribution(layer, 2, 1, -1))).toEqual([-6, -8])
  expect(editRatio(Float32Array.from([3, 4]), Float32Array.from([0, 10]))).toBeCloseTo(0.5, 6)
})
