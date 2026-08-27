import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { PipelineBand } from './PipelineBand'

const trace = makeFixtureTrace()

afterEach(() => cleanup())

test('renders five stage cards', () => {
  render(<PipelineBand events={trace} cursor={-1} />)
  expect(screen.getAllByTestId('stage-card')).toHaveLength(5)
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
  expect(summaryOf('embeddings')).toContain('576 dims')
  expect(summaryOf('layers')).toContain('3 layers')
  expect(summaryOf('logits')).toContain('on')      // top candidate of latest cycle
  expect(summaryOf('sampler')).toContain('on')     // chosen token of latest cycle
})

test('cards show no micro-summaries before any data', () => {
  render(<PipelineBand events={trace} cursor={-1} />)
  expect(screen.queryAllByTestId('stage-summary')).toHaveLength(0)
})
