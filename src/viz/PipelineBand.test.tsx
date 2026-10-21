import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../test/fixtures'
import { PipelineBand } from './PipelineBand'

const trace = makeFixtureTrace()

afterEach(() => cleanup())

test('renders six stage cards, softmax between logits and sampler', () => {
  render(<PipelineBand events={trace} cursor={-1} />)
  const stages = screen.getAllByTestId('stage-card').map((c) => c.dataset.stage)
  expect(stages).toEqual(['tokenizer', 'embeddings', 'layers', 'logits', 'softmax', 'sampler'])
})

test('highlights the stage of the cursor event', () => {
  render(<PipelineBand events={trace} cursor={1} />)  // tokenize event
  const active = screen.getAllByTestId('stage-card').filter((c) => c.dataset.active === 'true')
  expect(active).toHaveLength(1)
  expect(active[0].dataset.stage).toBe('tokenizer')
})

test('cards show micro-summaries once their data exists', () => {
  render(<PipelineBand events={trace} cursor={trace.length - 1} />)
  const summaryOf = (stage: string) =>
    screen.getAllByTestId('stage-card').find((c) => c.dataset.stage === stage)?.textContent
  expect(summaryOf('tokenizer')).toContain('2 tokens')
  expect(summaryOf('embeddings')).toContain('49 152 × 576')
  expect(summaryOf('layers')).toContain('3 layers')
  expect(summaryOf('logits')).toContain('on')      // top candidate of latest cycle
  expect(summaryOf('softmax')).toContain('70%')    // its probability after softmax
  expect(summaryOf('sampler')).toContain('on')     // chosen token of latest cycle
})

test('cards show no micro-summaries before any data', () => {
  render(<PipelineBand events={trace} cursor={-1} />)
  expect(screen.queryAllByTestId('stage-summary')).toHaveLength(0)
})

test('stage cards seek to the representative event when clickable', () => {
  const onStageClick = vi.fn()
  render(<PipelineBand events={trace} cursor={4} onStageClick={onStageClick} />)
  const card = screen.getAllByTestId('stage-card').find((c) => c.dataset.stage === 'layers')!
  fireEvent.click(card)
  expect(onStageClick).toHaveBeenCalledWith(6)  // cycle 0 attention
  expect(card.dataset.clickable).toBe('true')
})

test('connector labels show live tensor shapes', () => {
  render(<PipelineBand events={trace} cursor={7} />)
  const labels = screen.getAllByTestId('flow-shape').map((l) => l.textContent)
  expect(labels).toEqual(['[2]', '[2×576]', '[1×576]', '[49 152]', '[10]'])   // the funnel: all scores → top-k → 1 token
  expect(screen.getByTestId('loop-label')).toHaveTextContent('+1 token')
})

test('the logits card seeks to the logits event, the softmax card to the softmax event', () => {
  const onStageClick = vi.fn()
  render(<PipelineBand events={trace} cursor={4} onStageClick={onStageClick} />)
  const cards = screen.getAllByTestId('stage-card')
  fireEvent.click(cards.find((c) => c.dataset.stage === 'logits')!)
  expect(onStageClick).toHaveBeenLastCalledWith(7)
  fireEvent.click(cards.find((c) => c.dataset.stage === 'softmax')!)
  expect(onStageClick).toHaveBeenLastCalledWith(8)
})

test('connector labels are absent before a run', () => {
  render(<PipelineBand events={[]} cursor={-1} />)
  expect(screen.queryAllByTestId('flow-shape')).toHaveLength(0)
  expect(screen.queryByTestId('loop-label')).not.toBeInTheDocument()
})

test('cards without a target are not clickable', () => {
  const onStageClick = vi.fn()
  render(<PipelineBand events={trace.slice(0, 2)} cursor={1} onStageClick={onStageClick} />)
  const card = screen.getAllByTestId('stage-card').find((c) => c.dataset.stage === 'layers')!
  fireEvent.click(card)
  expect(onStageClick).not.toHaveBeenCalled()
  expect(card.dataset.clickable).toBe('false')
})

test('embeddings summary shows the embedding-table shape', () => {
  render(<PipelineBand events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getAllByTestId('stage-card')[1]).toHaveTextContent('49 152 × 576')
})

const tokenizerCard = () => screen.getAllByTestId('stage-card').find((c) => c.dataset.stage === 'tokenizer')!

test('during the first cycle the tokenizer card still seeks to the tokenize event', () => {
  const onStageClick = vi.fn()
  render(<PipelineBand events={trace} cursor={4} onStageClick={onStageClick} />)
  expect(tokenizerCard().dataset.outsideLoop).toBe('false')
  fireEvent.click(tokenizerCard())
  expect(onStageClick).toHaveBeenCalledWith(1)
})

test('after the first cycle the tokenizer card is dimmed and explains itself instead of rewinding', () => {
  const onStageClick = vi.fn()
  render(<PipelineBand events={trace} cursor={13} onStageClick={onStageClick} />)
  expect(tokenizerCard().dataset.outsideLoop).toBe('true')
  expect(screen.queryByTestId('tokenizer-note')).not.toBeInTheDocument()
  fireEvent.click(tokenizerCard())
  expect(onStageClick).not.toHaveBeenCalled()
  expect(screen.getByTestId('tokenizer-note')).toHaveTextContent('The tokenizer runs once, before the loop')
  expect(screen.getByTestId('tokenizer-note')).toHaveTextContent('never tokenized')
  fireEvent.click(tokenizerCard())
  expect(screen.queryByTestId('tokenizer-note')).not.toBeInTheDocument()
})

test('the note goes away when the cursor returns to the first cycle', () => {
  const { rerender } = render(<PipelineBand events={trace} cursor={13} onStageClick={() => {}} />)
  fireEvent.click(tokenizerCard())
  rerender(<PipelineBand events={trace} cursor={4} onStageClick={() => {}} />)
  expect(screen.queryByTestId('tokenizer-note')).not.toBeInTheDocument()
})

test('the loop arrow runs from the Sampler back to the Embeddings card', () => {
  render(<PipelineBand events={trace} cursor={7} />)
  const loop = screen.getByTestId('loop-arrow')
  expect(loop.dataset.from).toBe('sampler')
  expect(loop.dataset.to).toBe('embeddings')
})

test('back in a later cycle after visiting the first, the note stays closed', () => {
  const { rerender } = render(<PipelineBand events={trace} cursor={13} onStageClick={() => {}} />)
  fireEvent.click(tokenizerCard())
  rerender(<PipelineBand events={trace} cursor={4} onStageClick={() => {}} />)
  rerender(<PipelineBand events={trace} cursor={13} onStageClick={() => {}} />)
  expect(screen.queryByTestId('tokenizer-note')).not.toBeInTheDocument()
})

test('at a later cycle’s Embeddings step the loop arrow is the one lit, not the tokenizer’s', () => {
  const { rerender } = render(<PipelineBand events={trace} cursor={2} />)   // cycle 0 embed
  expect(screen.getByTestId('loop-arrow').dataset.active).toBe('false')
  rerender(<PipelineBand events={trace} cursor={11} />)                       // cycle 1 embed
  expect(screen.getByTestId('loop-arrow').dataset.active).toBe('true')
})

test('summaries show whitespace tokens as glyphs, like the token chips', () => {
  const nl = buildFixtureTrace({ cycles: 1, chosen: [{ id: 198, text: '\n' }] })
  render(<PipelineBand events={nl} cursor={nl.length - 1} />)
  const summaryOf = (stage: string) =>
    screen.getAllByTestId('stage-card').find((c) => c.dataset.stage === stage)!.querySelector('.stage-summary')!.textContent
  expect(summaryOf('logits')).toBe('top: ↵')
  expect(summaryOf('softmax')).toBe('↵: 70%')
  expect(summaryOf('sampler')).toBe('chose ↵')
})

test('the tokenizer card of a chat run says how many tokens are yours', () => {
  const events = buildFixtureTrace({ chat: true })
  render(<PipelineBand events={events} cursor={1} onStageClick={() => {}} />)
  expect(screen.getAllByTestId('stage-card')[0]).toHaveTextContent('37 tokens · 7 yours')
})
