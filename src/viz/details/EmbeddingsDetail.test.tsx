import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../../test/fixtures'
import { makeGeometryAsset } from '../../test/geometryFixture'
import type { GeometryState } from '../../geometry/useGeometry'
import { EmbeddingsDetail } from './EmbeddingsDetail'

const ready = (): GeometryState => ({ status: 'ready', asset: makeGeometryAsset(), retry: () => {} })
let geo: GeometryState = ready()
vi.mock('../../geometry/useGeometry', () => ({ useGeometry: () => geo }))
afterEach(() => { cleanup(); geo = ready() })

// fixture indices: 2 = cycle-0 embed (The, cat), 11 = cycle-1 embed (The, cat, sat)

test('lists the visible tokens as chips with the newest selected', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={11} />)
  const chips = screen.getAllByTestId('embed-token')
  expect(chips).toHaveLength(3)
  expect(chips[2].dataset.selected).toBe('true')
  expect(screen.getByTestId('detail-embeddings')).toHaveTextContent('49 152 × 576')
})

test('clicking a chip selects that row', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={11} />)
  fireEvent.click(screen.getAllByTestId('embed-token')[0])
  expect(screen.getAllByTestId('embed-token')[0].dataset.selected).toBe('true')
  expect(screen.getByTestId('embed-lookup')).toHaveTextContent('row 10')
})

test('model-source rows render a 96-cell pooled strip', () => {
  render(<EmbeddingsDetail events={buildFixtureTrace({ embedRows: true })} cursor={2} />)
  expect(screen.getAllByTestId('embed-strip-cell')).toHaveLength(96)
  expect(screen.getByTestId('embed-lookup')).toHaveTextContent('mean-pooled into 96 cells')
})

test('asset-source rows come from the geometry asset (fixture vectors have 4 dims)', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getAllByTestId('embed-strip-cell')).toHaveLength(4)
  expect(screen.getByTestId('embed-lookup')).toHaveTextContent('PCA-reduced')
})

test('asset source without geometry shows the offline placeholder', () => {
  geo = { status: 'error', error: 'offline', retry: () => {} }
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getByTestId('embed-strip-missing')).toHaveTextContent('unavailable offline')
})

test('asset source while geometry loads says so', () => {
  geo = { status: 'loading', retry: () => {} }
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getByTestId('embed-strip-missing')).toHaveTextContent(/loading/i)
})

test('three callouts carry their explanations', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  const callouts = screen.getAllByTestId('embed-callout')
  expect(callouts.map((c) => c.textContent)).toEqual(['ⓘ learned, not designed', 'ⓘ no position here', 'ⓘ tied with Logits'])
  expect(callouts[1].getAttribute('title')).toMatch(/rotary/)
})

test('stack has one row per token with the newest marked', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={11} />)
  const rows = screen.getAllByTestId('embed-stack-row')
  expect(rows).toHaveLength(3)
  expect(rows[2].dataset.newest).toBe('true')
})
