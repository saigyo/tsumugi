import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../../test/fixtures'
import { makeGeometryAsset } from '../../test/geometryFixture'
import type { GeometryState } from '../../geometry/useGeometry'
import { LogitsDetail } from './LogitsDetail'

const ready = (): GeometryState => ({ status: 'ready', asset: makeGeometryAsset(), retry: () => {} })
let geo: GeometryState = ready()
vi.mock('../../geometry/useGeometry', () => ({ useGeometry: () => geo }))
afterEach(() => { cleanup(); geo = ready() })

// fixture indices: cycle 0 logits = 7, softmax = 8; cycle 1 logits = 16

test('real fixture: bench sum equals the top candidate’s logit; formula and copy present', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true })} cursor={7} />)
  expect(screen.getByTestId('logits-formula')).toHaveTextContent('[1×576]')
  expect(screen.getByTestId('detail-logits')).toHaveTextContent(/dot product/i)
  expect(screen.getByTestId('readout-sum')).toHaveTextContent('9.1')
})

test('the logits card always shows scores, never percentages — softmax has its own card', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true })} cursor={8} />)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('Logits')
  expect(screen.getAllByTestId('readout-candidate')[0]).toHaveTextContent('9.1')
  expect(screen.queryByTestId('softmax-formula')).toBeNull()
  expect(screen.queryByTestId('softmax-explainer')).toBeNull()
})

test('selection resets when the cursor moves to another cycle', () => {
  const t = buildFixtureTrace({ readout: true })
  const { rerender } = render(<LogitsDetail events={t} cursor={16} />)
  fireEvent.click(screen.getAllByTestId('readout-candidate')[2])
  expect(screen.getAllByTestId('readout-candidate')[2].dataset.selected).toBe('true')
  rerender(<LogitsDetail events={t} cursor={7} />)
  expect(screen.getAllByTestId('readout-candidate')[0].dataset.selected).toBe('true')
})

test('sim fixture resolves candidate rows from the geometry asset', () => {
  render(<LogitsDetail events={makeFixtureTrace()} cursor={7} />)
  expect(screen.getByTestId('readout-x-missing')).toBeInTheDocument()
  expect(screen.getByTestId('bench-row').querySelectorAll('rect')).toHaveLength(4)
  expect(screen.getByTestId('readout-sum')).toHaveTextContent('PCA-reduced')
})

test('asset row lookup is guarded for ids outside the vocabulary', () => {
  const t = makeFixtureTrace()
  const logits = t[7]
  if (logits.type !== 'logits') throw new Error('fixture index moved')
  logits.topK[0].id = 99999   // fallback hash tokenizer ids can exceed the asset vocab (256 in the fixture)
  render(<LogitsDetail events={t} cursor={7} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('unavailable')
})

test('geometry loading and error states show their notes', () => {
  geo = { status: 'loading', retry: () => {} }
  render(<LogitsDetail events={makeFixtureTrace()} cursor={7} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent(/loading/i)
  cleanup()
  geo = { status: 'error', error: 'offline', retry: () => {} }
  render(<LogitsDetail events={makeFixtureTrace()} cursor={7} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('unavailable')
})
