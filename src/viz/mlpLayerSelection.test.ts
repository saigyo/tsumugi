import { expect, test } from 'vitest'
import { buildFixtureTrace } from '../test/fixtures'
import { selectedLayer, useMlpLayer } from './mlpLayerSelection'

test('a pick holds for its run and falls back for another run', () => {
  const a = buildFixtureTrace()[0], b = buildFixtureTrace()[0]
  useMlpLayer.getState().select(a, 2)
  const pick = useMlpLayer.getState().pick
  expect(selectedLayer(pick, a)).toBe(2)
  expect(selectedLayer(pick, b)).toBeNull()
  expect(selectedLayer(null, a)).toBeNull()
})
