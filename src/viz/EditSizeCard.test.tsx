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
const cells = () => screen.getAllByTestId('edit-cell')
const cellAt = (row: number, layer: number) =>
  cells().find((c) => c.getAttribute('data-row') === String(row) && c.getAttribute('data-layer') === String(layer))!

test('one heatmap of tokens × layers, MLP first; the toggle switches it to attention', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  const toggles = screen.getAllByTestId('edit-block-toggle')
  expect(toggles.map((t) => [t.textContent, t.getAttribute('aria-pressed')])).toEqual([['MLP', 'true'], ['attention', 'false']])
  expect(cells()).toHaveLength(2 * 3)
  expect(cells().every((c) => c.getAttribute('data-block') === 'mlp')).toBe(true)
  fireEvent.click(toggles[1])
  expect(screen.getAllByTestId('edit-block-toggle')[1]).toHaveAttribute('aria-pressed', 'true')
  expect(cells().every((c) => c.getAttribute('data-block') === 'attn')).toBe(true)
  expect(cellAt(0, 2)).toHaveAttribute('title', '“The”, layer 2, attention: 1.50')
  expect(texOf(screen.getByTestId('edit-size-formula'))).toContain('\\lVert')
})

test('the heatmap has a legend and layer labels', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  expect(screen.getByTestId('edit-legend')).toHaveTextContent('0')
  expect(screen.getByTestId('edit-legend')).toHaveTextContent('≥ 1.0')
  expect(screen.getByTestId('edit-axis')).toHaveTextContent('L0')
})

test('token chips select the token the bars show; the last token by default', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  const chips = screen.getAllByTestId('edit-row-label')
  expect(chips.map((c) => c.tagName)).toEqual(['BUTTON', 'BUTTON'])
  expect(chips.map((c) => c.getAttribute('aria-pressed'))).toEqual(['false', 'true'])
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('“␣cat”')
  fireEvent.click(chips[0])
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('“The”')
  const bar = screen.getAllByTestId('edit-bar').find((b) => b.getAttribute('data-block') === 'mlp' && b.getAttribute('data-layer') === '0')
  expect(bar).toHaveAttribute('data-value', '0.90')
})

test('the bar chart has a 0–2 axis, a legend and the 1.0 reference line', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  const chart = screen.getByTestId('edit-bars')
  for (const tick of ['0.0', '0.5', '1.0', '1.5', '2.0']) expect(chart).toHaveTextContent(tick)
  expect(chart).toHaveTextContent('attention')
  expect(chart).toHaveTextContent('MLP')
  expect(chart).toHaveTextContent('L0')
  expect(screen.getByTestId('edit-bars-ref')).toBeInTheDocument()
  expect(chart).toHaveTextContent('edit as large as the stream itself')
  expect(screen.getAllByTestId('edit-bar')).toHaveLength(2 * 3)
})

test('bars cap at 2.0 so one huge edit does not flatten the rest; the title keeps the true value', () => {
  render(<EditSizeCard rows={{ tokens: [{ id: 1, text: 'The' }], attn: [[0.5, 1, 0.2]], mlp: [[12, 1, 0.3]] }} note="unused" />)
  const bar = (block: string, layer: number) => screen.getAllByTestId('edit-bar')
    .find((b) => b.getAttribute('data-block') === block && b.getAttribute('data-layer') === String(layer))!
  const h = (el: Element) => Number(el.getAttribute('height'))
  expect(h(bar('mlp', 1)) / h(bar('mlp', 0))).toBeCloseTo(0.5, 2)
  expect(bar('mlp', 0)).toHaveAttribute('data-capped', 'true')
  expect(bar('mlp', 1)).toHaveAttribute('data-capped', 'false')
  expect(bar('mlp', 0).textContent).toBe('layer 0, MLP: 12.00')
})

test('selection out of range falls back to the last visible token', () => {
  const three = { tokens: [...rows.tokens, { id: 3, text: '!' }], attn: [...rows.attn, [1, 1, 1]], mlp: [...rows.mlp, [1, 1, 1]] }
  const { rerender } = render(<EditSizeCard rows={three} note="unused" />)
  fireEvent.click(screen.getAllByTestId('edit-row-label')[2])
  rerender(<EditSizeCard rows={rows} note="unused" />)
  expect(screen.getAllByTestId('edit-row-label')[1]).toHaveAttribute('aria-pressed', 'true')
})

test('a long prompt draws only the last 64 tokens and says so', () => {
  const n = 100
  const long = {
    tokens: Array.from({ length: n }, (_, i) => ({ id: i, text: ` t${i}` })),
    attn: Array.from({ length: n }, () => [0.1, 0.2, 0.3]),
    mlp: Array.from({ length: n }, () => [0.4, 0.5, 0.6]),
  }
  render(<EditSizeCard rows={long} note="unused" />)
  const labels = screen.getAllByTestId('edit-row-label')
  expect(labels).toHaveLength(64)
  expect(labels[0]).toHaveTextContent('␣t36')
  expect(screen.getByTestId('edit-rows-note')).toHaveTextContent('showing the last 64 of 100 tokens')
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('␣t99')
  fireEvent.click(labels[0])
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('␣t36')
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
