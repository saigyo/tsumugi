import { expect, test } from 'vitest'
import type { NeuronLayer, NeuronManifest } from './format'
import { behindTheVote, firingMost, layerActivations, reconcile } from './math'

// 4 neurons over 3 dims. Dequantized columns:
//   W0 = [2.54, 0, 0]   W1 = [0, 2.54, 0]   W2 = [0, 0, 3.81]   W3 = [1.27, 1.27, 0]
const m = { neurons: 4, dims: 3, labelsPerSign: 2, fractions: [0.001, 0.01, 0.1] } as NeuronManifest
const T = [0.9, 0.6, 0.3]
const layer: NeuronLayer = {
  scales: Float32Array.from([0.02, 0.02, 0.03, 0.01]),
  columns: Int8Array.from([127, 0, 0, 0, 127, 0, 0, 0, 127, 127, 127, 0]),
  labelIds: Uint16Array.from(Array.from({ length: 16 }, (_, k) => 10 * Math.floor(k / 4) + 5 * (Math.floor(k / 2) % 2) + (k % 2))),
  thresholds: Float32Array.from([0.5, 0.4, 0.2, ...T, ...T, ...T, ...T, ...T, ...T, ...T]),
}
const g = Float32Array.from([1, 1, 1])
const row = Float32Array.from([1, -1, 0.5])           // u = g ⊙ row
const acts = Float32Array.from([2, 1, 4, -3])
const INV = 0.5
// votes = a · (W·u) · inv: [2·2.54·.5, 1·(−2.54)·.5, 4·1.905·.5, −3·0·.5] = [2.54, −1.27, 3.81, 0]

test('votes rank by size; the rest line is the total minus the rows shown', () => {
  const t = behindTheVote(m, layer, acts, row, g, INV, 2)
  expect(t.rows.map((r) => r.neuron)).toEqual([2, 0])
  expect(t.rows[0].vote).toBeCloseTo(3.81, 4)
  expect(t.rows[1].vote).toBeCloseTo(2.54, 4)
  expect(t.total).toBeCloseTo(5.08, 4)
  expect(t.rest).toBeCloseTo(-1.27, 4)
})

test('labels and badges follow the sign the neuron fired with, or the opposite list when it votes away', () => {
  const byNeuron = Object.fromEntries(behindTheVote(m, layer, acts, row, g, INV, 4).rows.map((r) => [r.neuron, r]))
  // n0: fired +, cos(W0, u) = 1/1.5 ≥ 0.5 (its own thresholds) → top 0.1%
  expect(byNeuron[0]).toMatchObject({ away: false, badge: 'top 0.1%', labels: [0, 1] })
  // n1: fired +, votes away → negative list; cos(−W1, u) = 0.667 ≥ 0.6 → top 1%
  expect(byNeuron[1]).toMatchObject({ away: true, badge: 'top 1%', labels: [15, 16] })
  // n2: cos(W2, u) = 0.333 ≥ 0.3 → top 10%
  expect(byNeuron[2]).toMatchObject({ away: false, badge: 'top 10%', labels: [20, 21] })
  // n3: fired −, vote 0 → its negative list; cos(−W3, u) = 0 → not in top 10%
  expect(byNeuron[3]).toMatchObject({ away: false, badge: 'not in top 10%', labels: [35, 36] })
})

test('a negative activation with a positive vote uses the negative-sign list and thresholds', () => {
  const t = behindTheVote(m, layer, Float32Array.from([-2, 0, 0, 0]), Float32Array.from([-1, 0, 0]), g, INV, 1)
  expect(t.rows[0]).toMatchObject({ neuron: 0, away: false, labels: [5, 6], badge: 'top 0.1%' })
  expect(t.rows[0].vote).toBeCloseTo(2.54, 4)
})

test('the final norm weight scales the candidate row before the comparison', () => {
  const t = behindTheVote(m, layer, acts, row, Float32Array.from([2, 1, 1]), INV, 4)
  expect(t.rows.find((r) => r.neuron === 0)!.vote).toBeCloseTo(5.08, 4)
})

test('firing most ranks by |activation| × column length, labels from the fired sign', () => {
  const rows = firingMost(m, layer, acts, 2)
  expect(rows.map((r) => r.neuron)).toEqual([2, 3])
  expect(rows[0].size).toBeCloseTo(15.24, 3)
  expect(rows[1].size).toBeCloseTo(3 * 1.27 * Math.SQRT2, 3)
  expect(rows[1].labels).toEqual([35, 36])
})

test('reconcile adopts the votes card value within tolerance, keeps the computed total outside it', () => {
  const t = behindTheVote(m, layer, acts, row, g, INV, 2)
  const close = reconcile(t, 5.1)
  expect(close.exact).toBe(true)
  expect(close.total).toBe(5.1)
  expect(close.rest).toBeCloseTo(5.1 - 6.35, 4)
  const far = reconcile(t, 6)
  expect(far.exact).toBe(false)
  expect(far.total).toBeCloseTo(5.08, 4)
})

test('layerActivations slices one layer out of the layer-major array', () => {
  expect([...layerActivations(Float32Array.from([1, 2, 3, 4, 5, 6]), 1, 3)]).toEqual([4, 5, 6])
})
