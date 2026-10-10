import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../../test/fixtures'
import { SoftmaxDetail } from './SoftmaxDetail'
import { resetGlossaryStore, useGlossaryStore } from '../../glossary/store'
import { texOf } from '../../test/tex'

afterEach(() => { cleanup(); resetGlossaryStore() })

// fixture indices: cycle 0 logits = 7, softmax = 8; candidates sat 9.1 / ran 7.2 / was 5.0 → 70% / 20% / 10%

test('shows both formula lines as the two-step derivation, then the explanation', () => {
  render(<SoftmaxDetail events={makeFixtureTrace()} cursor={8} />)
  expect(screen.getByTestId('detail-softmax')).toHaveTextContent('Softmax (T=0.8)')
  expect(texOf(screen.getByTestId('logits-formula'))).toContain('x_{\\text{last}} \\cdot E^{\\top}')   // step 1
  const sm = texOf(screen.getByTestId('softmax-formula'))                                         // step 2
  expect(sm).toContain('\\frac{\\exp(\\text{logit}/T)}')
  expect(sm).toContain('3 \\text{ of } 49\\,152')
  const why = screen.getByTestId('softmax-explainer')
  expect(why).toHaveTextContent(/divided by the temperature/)
  // the ratio is e^(1/T): at T = 0.8 a 1-point lead is 3.5×, 2 points 12.2× (2.7× holds only at T = 1)
  expect(why).toHaveTextContent(/at T = 0\.8 a lead of 1 point is a 3\.5× lead/)
  expect(why).toHaveTextContent(/2 points is 12\.2×/)
  expect(why).toHaveTextContent(/shares among the 3 shown/)
})

test('the ratio sentence follows the run’s temperature', () => {
  render(<SoftmaxDetail events={buildFixtureTrace({ temperature: 1 })} cursor={8} />)
  expect(screen.getByTestId('softmax-explainer')).toHaveTextContent(/at T = 1 a lead of 1 point is a 2\.7× lead/)
})

test('greedy run (T=0) says the top score takes everything', () => {
  render(<SoftmaxDetail events={buildFixtureTrace({ temperature: 0 })} cursor={8} />)
  expect(screen.getByTestId('softmax-explainer')).toHaveTextContent(/greedy/)
  expect(screen.getByTestId('softmax-explainer')).not.toHaveTextContent(/lead of 1 point/)
})

test('one bar per candidate with its percentage and a derivation on its own numbers', () => {
  render(<SoftmaxDetail events={makeFixtureTrace()} cursor={8} />)
  const rows = screen.getAllByTestId('softmax-candidate')
  expect(rows).toHaveLength(3)
  expect(rows[0]).toHaveTextContent('sat')
  expect(rows[0]).toHaveTextContent('70%')
  const titles = screen.getAllByTestId('prob-bar').map((b) => b.querySelector('title')?.textContent)
  expect(titles[0]).toMatch(/top score/)
  expect(titles[0]).toMatch(/70%/)
  // 9.1 − 7.2 = 1.9; exp(−1.9 / 0.8) = 0.093
  expect(titles[1]).toMatch(/1\.9 below the top score/)
  expect(titles[1]).toMatch(/exp\(−1\.9 \/ 0\.8\) = 0\.09/)
  expect(titles[1]).toMatch(/20%/)
})

test('T=0 bars say the top score takes everything', () => {
  render(<SoftmaxDetail events={buildFixtureTrace({ temperature: 0 })} cursor={8} />)
  const titles = screen.getAllByTestId('prob-bar').map((b) => b.querySelector('title')?.textContent)
  expect(titles[0]).toMatch(/T = 0/)
})

test('callouts open the Softmax entry at exp, temperature and top-k', () => {
  render(<SoftmaxDetail events={makeFixtureTrace()} cursor={8} />)
  const callouts = screen.getAllByTestId('embed-callout')
  expect(callouts.map((c) => c.textContent)).toEqual(['ⓘ why exponentiate', 'ⓘ temperature', 'ⓘ top-k only'])
  expect(callouts.map((c) => c.dataset.section)).toEqual(['exp', 'temperature', 'top-k'])
  fireEvent.click(callouts[1])
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'softmax', section: 'temperature' })
})

test('renders nothing before the cycle’s softmax event', () => {
  const { container } = render(<SoftmaxDetail events={makeFixtureTrace()} cursor={7} />)
  expect(container).toBeEmptyDOMElement()
})
