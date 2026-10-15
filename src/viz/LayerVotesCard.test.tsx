import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
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
const cell = (row: number, layer: number) => screen.getAllByTestId('vote-cell')
  .find((c) => c.getAttribute('data-row') === String(row) && c.getAttribute('data-layer') === String(layer))!
const sum = (a: number[]) => a.reduce((s, v) => s + v, 0)
const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`

test('grid: one row per candidate, one cell per layer, MLP votes by default, blue adds and red takes', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  expect(screen.getAllByTestId('vote-row-label')).toHaveLength(3)
  expect(screen.getAllByTestId('vote-cell')).toHaveLength(3 * 3)
  const v = cands[0].votes.mlp[1]
  expect(cell(0, 1)).toHaveAttribute('data-value', v.toFixed(2))
  expect(cell(0, 1).style.background).toMatch(v >= 0 ? /hsl\(211|rgb/ : /hsl\(13|rgb/)
  expect(texOf(screen.getByTestId('votes-formula'))).toContain('\\sum')
})

test('the toggle switches between MLP, attention and both blocks; the row total follows', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  const toggles = screen.getAllByTestId('votes-block-toggle')
  expect(toggles.map((t) => t.textContent)).toEqual(['MLP', 'attention', 'both'])
  const total = () => screen.getAllByTestId('vote-row-total')[0].textContent
  expect(total()).toContain(signed(sum(cands[0].votes.mlp)))
  fireEvent.click(toggles[1])
  expect(cell(0, 1)).toHaveAttribute('data-value', cands[0].votes.attn[1].toFixed(2))
  expect(total()).toContain(signed(sum(cands[0].votes.attn)))
  fireEvent.click(toggles[2])
  expect(cell(0, 1)).toHaveAttribute('data-value', (cands[0].votes.attn[1] + cands[0].votes.mlp[1]).toFixed(2))
  expect(screen.getAllByTestId('votes-block-toggle')[2]).toHaveAttribute('aria-pressed', 'true')
})

test('hovering a cell writes its vote into the readout line', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  expect(screen.getByTestId('vote-readout')).toHaveTextContent('Hover a cell')
  fireEvent.mouseEnter(cell(0, 1))
  const v = cands[0].votes.mlp[1]
  expect(screen.getByTestId('vote-readout')).toHaveTextContent(
    `MLP 1 → “␣sat”: ${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)} to the score`)
  fireEvent.click(screen.getAllByTestId('votes-block-toggle')[2])
  fireEvent.mouseEnter(cell(1, 2))
  expect(screen.getByTestId('vote-readout')).toHaveTextContent('layer 2 (attention + MLP) → “␣ran”')
})

test('the legend names the scale’s ends and says it is a square-root scale', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  const peak = Math.max(...cands.flatMap((c) => c.votes.mlp.map(Math.abs)))
  const legend = screen.getByTestId('votes-legend')
  expect(legend).toHaveTextContent(`−${peak.toFixed(1)}`)
  expect(legend).toHaveTextContent(`+${peak.toFixed(1)}`)
  expect(legend).toHaveTextContent(/square-root/)
})

test('running score: one line per candidate with its name at the end, axes, and the selected line ending on the logit', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  const lines = screen.getAllByTestId('vote-line')
  expect(lines).toHaveLength(3)
  const sel = lines.find((l) => l.getAttribute('data-selected') === 'true')!
  expect(sel.getAttribute('data-final')).toBe(cands[0].logit.toFixed(2))
  expect(screen.getAllByTestId('vote-line-label').map((l) => l.textContent)).toEqual(expect.arrayContaining(['␣sat', '␣ran', '␣was']))
  const chart = screen.getByTestId('votes-lens')
  expect(chart).toHaveTextContent('L0')
  expect(chart).toHaveTextContent('L2')
  expect(screen.getByTestId('vote-final')).toHaveTextContent(cands[0].logit.toFixed(2))
})

test('the breakdown adds up to the Logits card’s score, without a remainder row', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  expect(screen.getByTestId('votes-total')).toHaveTextContent(cands[0].logit.toFixed(2))
  const labels = screen.getAllByTestId('votes-row').map((r) => r.textContent)
  expect(labels[0]).toMatch(/embedding/)
  expect(labels[1]).toMatch(/all 3 attention blocks/)
  expect(labels[2]).toMatch(/all 3 MLP blocks/)
  expect(labels.slice(3)).toHaveLength(3)
  expect(labels.join()).not.toMatch(/remainder/)
})

test('the caption names the Logits card’s score even when the rounded votes miss it by a hundredth', () => {
  const off = [{ ...cands[0], logit: 9.09 }, ...cands.slice(1)]   // votes still add up to 9.10
  render(<LayerVotesCard candidates={off} note="unused" />)
  expect(screen.getByTestId('vote-final')).toHaveTextContent('9.09')
  expect(screen.getByTestId('votes-total')).toHaveTextContent('9.09')
})

test('clicking a candidate chip selects it here and on the Logits card', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  const chips = screen.getAllByTestId('vote-row-label')
  expect(chips.map((c) => [c.tagName, c.getAttribute('aria-pressed')])).toEqual([['BUTTON', 'true'], ['BUTTON', 'false'], ['BUTTON', 'false']])
  fireEvent.click(chips[2])
  expect(screen.getAllByTestId('vote-row-label')[2]).toHaveAttribute('aria-pressed', 'true')
  cleanup()
  render(<LogitsDetail events={trace} cursor={8} />)
  expect(screen.getAllByTestId('readout-candidate')[2]).toHaveAttribute('data-selected', 'true')
})

test('without candidates the card keeps its heading and shows the note', () => {
  render(<LayerVotesCard note="Votes appear once this cycle’s scores are computed." />)
  expect(screen.getByTestId('votes-card')).toHaveTextContent('What each layer voted for')
  expect(screen.getByTestId('votes-note')).toHaveTextContent(/Votes appear/)
  expect(screen.queryAllByTestId('vote-cell')).toHaveLength(0)
})

test('with many candidates only the selected one and the four highest final scores get a name at the line end', () => {
  const many = Array.from({ length: 8 }, (_, i) => ({ ...cands[i % 3], id: 500 + i, text: ` c${i}`,
    logit: 9 - i, votes: { ...cands[0].votes, embed: cands[0].votes.embed - i } }))
  render(<LayerVotesCard candidates={many} note="unused" />)
  fireEvent.click(screen.getAllByTestId('vote-row-label')[7])
  expect(screen.getAllByTestId('vote-line-label').map((l) => l.textContent).sort())
    .toEqual(['␣c0', '␣c1', '␣c2', '␣c3', '␣c7'])
  expect(screen.getAllByTestId('vote-line')).toHaveLength(8)
})

test('the caption explains the vote as each piece’s share of the comparison with the candidate’s row', () => {
  render(<LayerVotesCard candidates={cands} note="unused" />)
  const card = screen.getByTestId('votes-card')
  expect(card).toHaveTextContent('The vector the layers hand to the Logits step is a sum')
  expect(card).toHaveTextContent('positive if the edit pushed the vector toward this candidate, negative if away from it')
  expect(texOf(screen.getByTestId('votes-formula'))).toContain('divided by the same number')
  expect(screen.getByTestId('votes-legend')).toHaveTextContent('blue: pushed toward the candidate · red: pushed away')
})

test('clicking a vote cell reports its layer', () => {
  const onLayer = vi.fn()
  const events = buildFixtureTrace({ mode: 'real', mlp: true })
  const v = cycleVotes(events, events.length - 1)
  render(<LayerVotesCard candidates={v.status === 'ok' ? v.candidates : undefined} note="" onLayer={onLayer} />)
  fireEvent.click(screen.getAllByTestId('vote-cell').find((c) => c.dataset.row === '1' && c.dataset.layer === '2')!)
  expect(onLayer).toHaveBeenCalledWith(2)
})
