import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { AttentionHeatmap } from './AttentionHeatmap'

afterEach(() => cleanup())

const trace = makeFixtureTrace()
const attn = trace.find((e) => e.type === 'attention')
if (attn?.type !== 'attention') throw new Error('fixture lacks attention')
const tokens = [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }]

test('renders one chip per head and a triangular cell grid', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.getAllByTestId('head-chip')).toHaveLength(2)
  // 2 tokens → rows of length 1 and 2 → 3 cells
  expect(screen.getAllByTestId('attn-cell')).toHaveLength(3)
})

test('chip click switches the selected head and its hint', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  const chips = screen.getAllByTestId('head-chip')
  expect(chips[0].dataset.active).toBe('true')
  expect(screen.getByTestId('attn-hint')).toHaveTextContent(/first token/i)   // sink hint
  fireEvent.click(chips[1])
  expect(chips[1].dataset.active).toBe('true')
  expect(screen.getByTestId('attn-hint')).toHaveTextContent(/right before/i)  // previous-token hint
})

test('cells carry a from → to tooltip with the percentage', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  const cells = screen.getAllByTestId('attn-cell')
  const withPct = cells.find((c) => c.querySelector('title')?.textContent?.includes('%'))
  expect(withPct).toBeDefined()
  expect(withPct!.querySelector('title')!.textContent).toMatch(/→/)
})

test('column labels run along the diagonal', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  const cols = screen.getAllByTestId('col-label')
  expect(cols).toHaveLength(2)
  expect(cols.map((c) => c.textContent)).toEqual(['The', 'cat'])
})

test('hovering a cell highlights both labels and fills the readout', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  const cells = screen.getAllByTestId('attn-cell')
  fireEvent.mouseEnter(cells[1])  // row 1 ("cat"), col 0 ("The")
  expect(screen.getByTestId('attn-readout')).toHaveTextContent(/cat\s*→\s*The:\s*\d+%/)
  const rows = screen.getAllByTestId('row-label')
  const cols = screen.getAllByTestId('col-label')
  expect(rows[1].dataset.hl).toBe('true')
  expect(cols[0].dataset.hl).toBe('true')
  fireEvent.mouseLeave(cells[1])
  expect(screen.getByTestId('attn-readout')).not.toHaveTextContent('%')
  expect(rows[1].dataset.hl).toBe('false')
})

test('scored heads show the score and the measured note', () => {
  const scored = attn.heads.map((h) => ({ ...h, score: 0.87 }))
  render(<AttentionHeatmap heads={scored} tokens={tokens} />)
  expect(screen.getAllByTestId('head-chip')[0]).toHaveTextContent('0.87')
  expect(screen.getByTestId('attn-note')).toHaveTextContent(/measured on this prompt/i)
})

test('unscored heads keep the illustrative note', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.getByTestId('attn-note')).toHaveTextContent(/illustrative/i)
})

test('a pinned head with no score still shows the measured note, not the illustrative one', () => {
  const pinned = [{ layer: 1, head: 1, label: 'pinned' as const, matrix: attn.heads[0].matrix }]
  render(<AttentionHeatmap heads={pinned} tokens={tokens} />)
  expect(screen.getByTestId('attn-note')).toHaveTextContent(/measured on this prompt/i)
  expect(screen.getByTestId('attn-note')).toHaveTextContent(/accumulated over the whole run/i)
  expect(screen.getByTestId('attn-note')).not.toHaveTextContent(/illustrative/i)
})

test('focus prop selects the matching chip', () => {
  const target = attn.heads[1]
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens}
    focus={{ layer: target.layer, head: target.head, label: target.label }} />)
  const chips = screen.getAllByTestId('head-chip')
  expect(chips[1].dataset.active).toBe('true')
  expect(chips[0].dataset.active).toBe('false')
})

test('data-active tracks the clamped selection after heads shrinks', () => {
  const { rerender } = render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  const chips = screen.getAllByTestId('head-chip')
  fireEvent.click(chips[1])  // select index 1
  rerender(<AttentionHeatmap heads={[attn.heads[0]]} tokens={tokens} />)  // heads shrinks to 1
  const shrunkChips = screen.getAllByTestId('head-chip')
  expect(shrunkChips).toHaveLength(1)
  expect(shrunkChips[0].dataset.active).toBe('true')
})
