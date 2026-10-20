import { expect, test } from 'vitest'
import { ATTN_MODEL_REVISION } from '../engine/tokenizer'
import { unpackHalf } from '../trace/neuronPack'
import { encodeNeuronFixture, fixtureNeuronData, NF } from './neuronFixture'

test('the fixture manifest is for the revision the app pins', () => {
  expect(encodeNeuronFixture().manifest.revision).toBe(ATTN_MODEL_REVISION)
})

test('fixture steps are consistent: SiLU(gate) · up is the activation', () => {
  const d = fixtureNeuronData(1, [100], () => Array(576).fill(0.1))
  const g = unpackHalf(d.steps!.gate), u = unpackHalf(d.steps!.up), a = unpackHalf(d.activations)
  for (let i = 0; i < NF.layers * NF.neurons; i++) expect(a[i]).toBeCloseTo((g[i] / (1 + Math.exp(-g[i]))) * u[i], 2)
  expect(unpackHalf(d.steps!.mlpIn)).toHaveLength(NF.layers * NF.dims)
})
