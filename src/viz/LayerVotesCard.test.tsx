import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { buildFixtureTrace } from '../test/fixtures'
import { texOf } from '../test/tex'
import { LogitsDetail } from './details/LogitsDetail'
import { LayerVotesCard } from './LayerVotesCard'
import { cycleVotes } from './selectors'

afterEach(cleanup)

const trace = buildFixtureTrace({ mlp: true, readout: true })
const voted = cycleVotes(trace, 4)
if (voted.status !== 'ok') throw new Error('fixture has votes')
const cands = voted.candidates

test('grid: one row per candidate, one cell per MLP layer, blue for adds and red for takes', () => {
  render(<LayerVotesCard candidates={cands} cycle={0} note="unused" />)
  expect(screen.getAllByTestId('vote-row-label')).toHaveLength(3)
  const cells = screen.getAllByTestId('vote-cell')
  expect(cells).toHaveLength(3 * 3)
  const v = cands[0].votes.mlp[1]
  const cell = cells.find((c) => c.getAttribute('data-row') === '0' && c.getAttribute('data-layer') === '1')!
  expect(cell.textContent).toMatch(new RegExp(`MLP 1 → “.*”: ${v >= 0 ? '\\+' : '−'}${Math.abs(v).toFixed(2)} to the score`))
  expect(cell.getAttribute('fill')).toMatch(v >= 0 ? /^hsl\(211/ : /^hsl\(13/)
  expect(texOf(screen.getByTestId('votes-formula'))).toContain('\\sum')
})

test('running total ends exactly on the logit; the breakdown total is the Logits card’s score', () => {
  render(<LayerVotesCard candidates={cands} cycle={0} note="unused" />)
  const line = screen.getAllByTestId('vote-line').find((l) => l.getAttribute('data-selected') === 'true')!
  expect(line.getAttribute('data-final')).toBe(cands[0].logit.toFixed(2))
  expect(screen.getByTestId('vote-final')).toHaveTextContent(cands[0].logit.toFixed(2))
  expect(screen.getByTestId('votes-total')).toHaveTextContent(cands[0].logit.toFixed(2))
  const labels = screen.getAllByTestId('votes-row').map((r) => r.textContent)
  expect(labels[0]).toMatch(/embedding/)
  expect(labels[1]).toMatch(/all 3 attention blocks/)
  expect(labels[2]).toMatch(/all 3 MLP blocks/)
  expect(labels.slice(3)).toHaveLength(3)          // the three largest MLP layers
  expect(labels.join()).not.toMatch(/remainder/)
})

test('clicking a candidate selects it here and on the Logits card', () => {
  render(<LayerVotesCard candidates={cands} cycle={0} note="unused" />)
  fireEvent.click(screen.getAllByTestId('vote-row-label')[2])
  expect(screen.getAllByTestId('vote-row-label')[2]).toHaveAttribute('data-selected', 'true')
  cleanup()
  render(<LogitsDetail events={trace} cursor={8} />)
  expect(screen.getAllByTestId('readout-candidate')[2]).toHaveAttribute('data-selected', 'true')
})

test('without candidates the card keeps its heading and shows the note', () => {
  render(<LayerVotesCard cycle={0} note="Votes appear once this cycle’s scores are computed." />)
  expect(screen.getByTestId('votes-card')).toHaveTextContent('What each layer voted for')
  expect(screen.getByTestId('votes-note')).toHaveTextContent(/Votes appear/)
  expect(screen.queryAllByTestId('vote-cell')).toHaveLength(0)
})

test('the caption names the Logits card’s score even when the rounded votes miss it by a hundredth', () => {
  const off = [{ ...cands[0], logit: 9.09 }, ...cands.slice(1)]   // votes still add up to 9.10
  render(<LayerVotesCard candidates={off} cycle={0} note="unused" />)
  expect(screen.getByTestId('vote-final')).toHaveTextContent('9.09')
  expect(screen.getByTestId('votes-total')).toHaveTextContent('9.09')
})
