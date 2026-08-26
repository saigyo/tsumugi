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
