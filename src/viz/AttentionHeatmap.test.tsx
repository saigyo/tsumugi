import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace, CHAT_ANSWER_TOKENS, CHAT_PROMPT_TOKENS, makeFixtureTrace } from '../test/fixtures'
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

test('a newline token gets the ↵ glyph as its label, not an empty one', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={[{ id: 10, text: 'The' }, { id: 198, text: '\n' }]} />)
  expect(screen.getAllByTestId('attn-cell').map((c) => c.textContent)).toContain('↵ → The: ' + c0(attn.heads[0].matrix[1][0]))
})
const c0 = (w: number) => `${Math.round(w * 100)}%`

const chatTrace = buildFixtureTrace({ chat: true })
const chatAttn = chatTrace.filter((e) => e.type === 'attention').at(-1)
if (chatAttn?.type !== 'attention') throw new Error('chat fixture lacks attention')
const chatTokens = [...CHAT_PROMPT_TOKENS, ...CHAT_ANSWER_TOKENS]
const chat = { system: [1, 21] as [number, number], user: [24, 31] as [number, number] }

test('a chat run folds the system prompt into one column; the toggle opens the full matrix', () => {
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const n = chatAttn.heads[0].matrix.length
  expect(screen.getAllByTestId('row-label')).toHaveLength(n - 20)
  expect(screen.getAllByTestId('col-label').map((l) => l.textContent)).toContain('system prompt (20)')
  fireEvent.click(screen.getByTestId('attn-show-system'))
  expect(screen.getAllByTestId('row-label')).toHaveLength(n)
  expect(screen.getAllByTestId('col-label').map((l) => l.textContent)).not.toContain('system prompt (20)')
  expect(screen.queryAllByTestId('attn-cell').some((c) => c.classList.contains('attn-cell-fold'))).toBe(false)
})

test('the folded column’s label starts midway between its neighbours’ labels', () => {
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const ys = screen.getAllByTestId('col-label').map((l) => Number(l.getAttribute('y')))
  const fold = screen.getAllByTestId('col-label').findIndex((l) => l.textContent === 'system prompt (20)')
  expect(ys[fold] - ys[fold - 1]).toBeGreaterThan(0)
  expect(ys[fold] - ys[fold - 1]).toBe(ys[fold + 1] - ys[fold])
})

test('the folded column’s readout names the system prompt and its summed weight', () => {
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const cell = screen.getAllByTestId('attn-cell').find((c) => c.querySelector('title')?.textContent?.includes('→ system prompt (20)'))!
  fireEvent.mouseEnter(cell)
  expect(screen.getByTestId('attn-readout')).toHaveTextContent(/→ system prompt \(20\): \d+%/)
})

test('without spans there is no toggle and no fold; in compare there is no toggle either', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.queryByTestId('attn-show-system')).toBeNull()
  cleanup()
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} foldToggle={false} />)
  expect(screen.queryByTestId('attn-show-system')).toBeNull()
  expect(screen.getAllByTestId('col-label').map((l) => l.textContent)).toContain('system prompt (20)')
})

test('an induction head in a chat run explains that the template’s markers repeat', () => {
  const induction = [{ ...chatAttn.heads[0], label: 'induction' as const }]
  render(<AttentionHeatmap heads={induction} tokens={chatTokens} chat={chat} />)
  expect(screen.getByTestId('attn-hint')).toHaveTextContent('In Chat mode the template’s markers')
})

test('a narrow card shrinks the cells and thins the labels; the hovered row and column keep theirs', () => {
  const resize = vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(400)
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const cells = screen.getAllByTestId('attn-cell')
  expect(Number(cells[0].getAttribute('width'))).toBeLessThan(19)
  const shown = screen.getAllByTestId('row-label').filter((l) => l.textContent !== '')
  expect(shown.length).toBeLessThan(screen.getAllByTestId('row-label').length)
  const unlabelled = screen.getAllByTestId('row-label').findIndex((l) => l.textContent === '')
  const cell = cells.find((c) => c.getAttribute('data-row') === String(unlabelled))!
  fireEvent.mouseEnter(cell)
  expect(screen.getAllByTestId('row-label')[unlabelled].textContent).not.toBe('')
  resize.mockRestore()
})

test('hovering draws one outline on top and leaves the cells untouched', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.queryByTestId('attn-hover')).toBeNull()
  const cell = screen.getAllByTestId('attn-cell')[1]
  fireEvent.mouseEnter(cell)
  expect(screen.getByTestId('attn-hover')).toHaveAttribute('x', cell.getAttribute('x'))
  expect(screen.getAllByTestId('attn-cell').every((c) => !c.hasAttribute('stroke-width'))).toBe(true)
})

test('a hovered cell that vanishes when the matrix shrinks does not crash the readout', () => {
  const { rerender } = render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const cells = screen.getAllByTestId('attn-cell')
  fireEvent.mouseEnter(cells[cells.length - 1])
  rerender(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.getByTestId('attn-readout').textContent?.trim()).toBe('')
  expect(screen.queryByTestId('attn-hover')).toBeNull()
})

test('a hover cleared by a shrinking matrix does not come back when the matrix grows again', () => {
  const { rerender } = render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const cells = screen.getAllByTestId('attn-cell')
  fireEvent.mouseEnter(cells[cells.length - 1])
  rerender(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  rerender(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  expect(screen.queryByTestId('attn-hover')).toBeNull()
  expect(screen.getByTestId('attn-readout').textContent?.trim()).toBe('')
})

test('the simulated note says the real model would show computed weights, with a link to load it', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.getByTestId('attn-note')).toHaveTextContent('Illustrative pattern (simulated): the real model would show the weights it computed.')
  expect(screen.getByTestId('attn-note').querySelector('[data-testid="load-real"]')).not.toBeNull()
})

test('a new run starts with the system prompt folded again', () => {
  const { rerender } = render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  fireEvent.click(screen.getByTestId('attn-show-system'))
  expect(screen.getByTestId('attn-show-system')).toBeChecked()
  rerender(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={{ ...chat }} />)
  expect(screen.getByTestId('attn-show-system')).not.toBeChecked()
})

test('hovering a cell redraws the readout, not the memoised cells', () => {
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const cells = screen.getAllByTestId('attn-cell')
  // every cell computes its fill with two Math.round calls: a redraw of all cells shows up as hundreds
  const round = vi.spyOn(Math, 'round')
  try {
    fireEvent.mouseEnter(cells[cells.length - 1])
    fireEvent.mouseLeave(cells[cells.length - 1])
    expect(round.mock.calls.length).toBeLessThan(cells.length)
  } finally { round.mockRestore() }
})
