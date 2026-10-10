import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { READOUT_CELLS, type LogitCandidate } from '../trace/types'
import { fixtureLastHidden, fixtureReadout } from '../test/fixtures'
import { LogitsReadout, type LogitsReadoutProps } from './LogitsReadout'

afterEach(cleanup)

const real: LogitCandidate[] = [
  { id: 100, text: ' sat', logit: 9.1, ...fixtureReadout(100, 9.1) },
  { id: 200, text: ' ran', logit: 7.2, ...fixtureReadout(200, 7.2) },
  { id: 201, text: ' was', logit: 5.0 },                      // no graph row (tie at rank 16)
]
const sim: LogitCandidate[] = real.map(({ id, text, logit }) => ({ id, text, logit }))

function props(over: Partial<LogitsReadoutProps> = {}): LogitsReadoutProps {
  return {
    candidates: real, lastHidden: fixtureLastHidden(0), dims: 576, vocabSize: 49152,
    selected: 0, onSelect: vi.fn(), assetRowFor: (id) => (id < 256 ? [1, 2, 3, 4] : undefined),
    assetNote: 'vector values unavailable', ...over,
  }
}

test('real run: x strip, one product strip per candidate with a row, bench sum equals the logit', () => {
  render(<LogitsReadout {...props()} />)
  expect(screen.getByTestId('readout-x').querySelectorAll('rect')).toHaveLength(READOUT_CELLS)
  expect(screen.getAllByTestId('readout-product')).toHaveLength(2)
  expect(screen.getAllByTestId('readout-candidate')).toHaveLength(3)
  expect(screen.getAllByTestId('logit-bar')).toHaveLength(3)
  expect(screen.getByTestId('readout-sum')).toHaveTextContent('9.1')
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('sat')
})

test('candidate without dot renders without a product strip and gets the asset bench', () => {
  render(<LogitsReadout {...props({ selected: 2 })} />)
  const bench = screen.getByTestId('readout-bench')
  expect(bench).toHaveTextContent('PCA-reduced')
  // the reader IS in a real run here: say why this one candidate has no products, don't send them to real mode
  expect(bench).toHaveTextContent(/not among the 16 rows the model exported/)
  expect(bench).not.toHaveTextContent(/real-mode generation/)
  expect(bench.querySelector('[data-testid="bench-product"]')).toBeNull()
})

test('clicking a candidate reports its index', () => {
  const onSelect = vi.fn()
  render(<LogitsReadout {...props({ onSelect })} />)
  fireEvent.click(screen.getAllByTestId('readout-candidate')[1])
  expect(onSelect).toHaveBeenCalledWith(1)
})

test('sim/legacy run: placeholder x, asset rows, no products, one-liner', () => {
  render(<LogitsReadout {...props({ candidates: sim, lastHidden: undefined })} />)
  expect(screen.getByTestId('readout-x-missing')).toHaveTextContent('[1×576]')
  expect(screen.queryAllByTestId('readout-product')).toHaveLength(0)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent(/real-mode generation/)
  expect(screen.getByTestId('bench-row').querySelectorAll('rect')).toHaveLength(4)   // fixture asset vectors have 4 dims
})

test('asset row unavailable shows the note instead of a strip', () => {
  render(<LogitsReadout {...props({ candidates: sim, lastHidden: undefined, assetRowFor: () => undefined })} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('vector values unavailable')
})

test('bars carry the logit value, no derivation tooltip — that belongs to the Softmax card', () => {
  render(<LogitsReadout {...props()} />)
  expect(screen.getAllByTestId('readout-candidate')[1]).toHaveTextContent('7.2')
  expect(screen.getAllByTestId('logit-bar')[1].querySelector('title')).toBeNull()
})

test('matrix sketch marks every candidate and the callouts explain the tie', () => {
  render(<LogitsReadout {...props()} />)
  const matrix = screen.getByTestId('readout-matrix')
  expect(matrix.querySelectorAll('[data-testid="readout-row-mark"]')).toHaveLength(3)
  // the box is labelled Eᵀ, so its shape must read transposed — the same shape the formula line shows
  expect(matrix).toHaveTextContent('Eᵀ')
  expect(matrix).toHaveTextContent('[576 × 49 152]')
  const callouts = screen.getAllByTestId('embed-callout')
  expect(callouts.map((c) => c.textContent)).toEqual(['ⓘ tied with Embeddings', 'ⓘ similar rows, similar scores', 'ⓘ 49 152 scores, 3 shown'])
  expect(callouts.map((c) => c.dataset.section)).toEqual(['tied', 'similar-rows', 'top-k'])
  expect(callouts.every((c) => c.dataset.term === 'logit')).toBe(true)
})
