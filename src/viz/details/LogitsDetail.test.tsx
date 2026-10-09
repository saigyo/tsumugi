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

test('softmax cursor keeps the readout and shows percentages', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true })} cursor={8} />)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('Softmax')
  expect(screen.getAllByTestId('readout-candidate')[0]).toHaveTextContent('70%')
  expect(screen.getAllByTestId('readout-product').length).toBeGreaterThan(0)
})

test('softmax cursor shows both formula lines and explains the derivation', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true })} cursor={8} />)
  expect(screen.getByTestId('logits-formula')).toHaveTextContent('x_last · Eᵀ')          // step 1 stays
  const sm = screen.getByTestId('softmax-formula')
  expect(sm).toHaveTextContent('exp(logit / T)')
  expect(sm).toHaveTextContent('3 of 49 152')                                            // fixture keeps 3 candidates
  const why = screen.getByTestId('softmax-explainer')
  expect(why).toHaveTextContent(/divided by the temperature/)
  // the ratio is e^(1/T): at the fixture's T = 0.8 a 1-point lead is 3.5×, 2 points 12.2× (2.7× holds only at T = 1)
  expect(why).toHaveTextContent(/at T = 0\.8 a lead of 1 point is a 3\.5× lead/)
  expect(why).toHaveTextContent(/2 points is 12\.2×/)
  expect(why).not.toHaveTextContent(/2\.7×/)
  expect(why).toHaveTextContent(/shares among the 3 shown/)
})

test('the ratio sentence follows the run’s temperature', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true, temperature: 1 })} cursor={8} />)
  expect(screen.getByTestId('softmax-explainer')).toHaveTextContent(/at T = 1 a lead of 1 point is a 2\.7× lead/)
})

test('logits cursor shows neither the softmax formula nor the explainer', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true })} cursor={7} />)
  expect(screen.queryByTestId('softmax-formula')).toBeNull()
  expect(screen.queryByTestId('softmax-explainer')).toBeNull()
})

test('greedy run (T=0) explains that the top score takes everything', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true, temperature: 0 })} cursor={8} />)
  expect(screen.getByTestId('softmax-explainer')).toHaveTextContent(/greedy/)
  expect(screen.getByTestId('softmax-explainer')).not.toHaveTextContent(/lead of 1 point/)
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
