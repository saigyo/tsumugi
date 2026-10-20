import { expect, test } from 'vitest'
import { buildFixtureTrace } from '../test/fixtures'
import { effectiveMlpLayer, selectedLayer, useMlpLayer } from './mlpLayerSelection'
import type { VotedCandidate } from './selectors'

test('a pick holds for its run and falls back for another run', () => {
  const a = buildFixtureTrace()[0], b = buildFixtureTrace()[0]
  useMlpLayer.getState().select(a, 2)
  const pick = useMlpLayer.getState().pick
  expect(selectedLayer(pick, a)).toBe(2)
  expect(selectedLayer(pick, b)).toBeNull()
  expect(selectedLayer(null, a)).toBeNull()
})

test('the effective layer is the pick for this run, else the largest MLP vote of the selected candidate', () => {
  const run = buildFixtureTrace()[0]
  const cands = [{ votes: { embed: 0, attn: [0, 0, 0], mlp: [1, -5, 2] } }, { votes: { embed: 0, attn: [0, 0, 0], mlp: [9, 0, 0] } }] as unknown as VotedCandidate[]
  expect(effectiveMlpLayer(null, run, cands, 0)).toBe(1)
  expect(effectiveMlpLayer(null, run, cands, 1)).toBe(0)
  expect(effectiveMlpLayer({ run, layer: 2 }, run, cands, 0)).toBe(2)
  expect(effectiveMlpLayer({ run: buildFixtureTrace()[0], layer: 2 }, run, cands, 0)).toBe(1)   // another run's pick
  expect(effectiveMlpLayer(null, run, undefined, 0)).toBeNull()
})
