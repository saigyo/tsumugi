import { expect, test } from 'vitest'
import { unpackFloat32, unpackHalf } from '../../trace/neuronPack'
import { extractNeurons } from './neuronData'

const L = 2, N = 3, D = 4
const outputs = () => ({
  mlp_activations: { dims: [1, L, N], data: Float32Array.from([1, -2, 3, 0.5, 0, -28.25]) },
  final_norm_inv_rms: { dims: [1, 1, 1], data: Float32Array.from([0.04]) },
  topk_ids: { dims: [1, 3], data: BigInt64Array.from([7n, 9n, 11n]) },
  topk_embeds: { dims: [1, 3, D], data: Float32Array.from(Array.from({ length: 12 }, (_, i) => i / 4)) },
})

test('packs activations, 1/rms and the rows of the requested candidates by id', () => {
  const r = extractNeurons(outputs(), L, D, [9, 7])
  expect(r.status).toBe('ok')
  if (r.status !== 'ok') return
  expect(r.data.neurons).toBe(N)
  expect(r.data.invRms).toBeCloseTo(0.04, 6)
  expect([...unpackHalf(r.data.activations)]).toEqual([1, -2, 3, 0.5, 0, -28.25])
  expect(r.data.candidates.map((c) => c.id)).toEqual([9, 7])
  expect([...unpackFloat32(r.data.candidates[0].row)]).toEqual([1, 1.25, 1.5, 1.75])
})

test('absent when the export has no neuron outputs (v4 and older)', () => {
  const { topk_ids, topk_embeds } = outputs()
  expect(extractNeurons({ topk_ids, topk_embeds }, L, D, [7]).status).toBe('absent')
})

test('bad shape: wrong layer count, missing 1/rms, unknown candidate, bad 1/rms', () => {
  const o = outputs()
  expect(extractNeurons({ ...o, mlp_activations: { dims: [1, 3, 2], data: new Float32Array(6) } }, L, D, [7]))
    .toMatchObject({ status: 'bad-shape', detail: expect.stringMatching(/mlp_activations has shape \[1, 3, 2\]/) })
  expect(extractNeurons({ ...o, final_norm_inv_rms: undefined }, L, D, [7]))
    .toMatchObject({ status: 'bad-shape', detail: expect.stringMatching(/incomplete/) })
  expect(extractNeurons(o, L, D, [42])).toMatchObject({ status: 'bad-shape', detail: 'candidate 42 is not among topk_ids' })
  expect(extractNeurons({ ...o, final_norm_inv_rms: { dims: [1, 1, 1], data: Float32Array.from([0]) } }, L, D, [7]))
    .toMatchObject({ status: 'bad-shape', detail: expect.stringMatching(/1\/rms/) })
})
