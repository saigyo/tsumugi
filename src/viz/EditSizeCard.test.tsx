import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { EditSizeCard } from './EditSizeCard'
import { texOf } from '../test/tex'

afterEach(cleanup)

const rows = {
  tokens: [{ id: 1, text: 'The' }, { id: 2, text: ' cat' }],
  attn: [[0.2, 0.4, 1.5], [0.1, 0.3, 0.6]],
  mlp: [[0.9, 0.2, 0.3], [0.5, 0.05, 1.2]],
}

test('two heatmaps of tokens × layers, MLP first, one cell per value', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  const maps = screen.getAllByTestId('edit-map')
  expect(maps.map((m) => m.getAttribute('data-block'))).toEqual(['mlp', 'attn'])
  expect(screen.getAllByTestId('edit-cell')).toHaveLength(2 * 2 * 3)
  const cell = screen.getAllByTestId('edit-cell').find((c) =>
    c.getAttribute('data-block') === 'attn' && c.getAttribute('data-row') === '0' && c.getAttribute('data-layer') === '2')!
  expect(cell.textContent).toBe('“The”, layer 2, attention: 1.50')
  expect(texOf(screen.getByTestId('edit-size-formula'))).toContain('\\lVert')
})

test('bars follow the last token by default and the clicked row after', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('“ cat”'.replace(' ', '␣'))
  fireEvent.click(screen.getAllByTestId('edit-row-label')[0])
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('“The”')
  const bars = screen.getAllByTestId('edit-bar')
  expect(bars).toHaveLength(2 * 3)
  expect(bars.find((b) => b.getAttribute('data-block') === 'mlp' && b.getAttribute('data-layer') === '0'))
    .toHaveAttribute('data-value', '0.90')
})

test('selection out of range falls back to the last visible token', () => {
  const { rerender } = render(<EditSizeCard rows={{ ...rows, tokens: [...rows.tokens, { id: 3, text: '!' }],
    attn: [...rows.attn, [1, 1, 1]], mlp: [...rows.mlp, [1, 1, 1]] }} note="unused" />)
  fireEvent.click(screen.getAllByTestId('edit-row-label')[2])
  rerender(<EditSizeCard rows={rows} note="unused" />)
  expect(screen.getAllByTestId('edit-row-label')[1]).toHaveAttribute('data-selected', 'true')
})

test('without rows the card keeps its heading and shows the note', () => {
  render(<EditSizeCard note="Real model only: simulated runs have no MLP to measure." />)
  expect(screen.getByTestId('edit-size-card')).toHaveTextContent('How much each layer changes the vector')
  expect(screen.getByTestId('edit-size-note')).toHaveTextContent(/Real model only/)
  expect(screen.queryAllByTestId('edit-cell')).toHaveLength(0)
})

test('three chips open the edit entry’s sections', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  const chips = screen.getAllByTestId('embed-callout')
  expect(chips.map((c) => c.getAttribute('data-section'))).toEqual(['layer-0', 'quiet-middle', 'attn-vs-mlp'])
  expect(chips[0]).toHaveTextContent('why layer 0 is so dark')
})
