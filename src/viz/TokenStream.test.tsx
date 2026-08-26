import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { TokenStream } from './TokenStream'

const trace = makeFixtureTrace()

afterEach(() => cleanup())

test('shows prompt tokens once tokenize is reached', () => {
  render(<TokenStream events={trace} cursor={1} />)
  expect(screen.getAllByTestId('prompt-token')).toHaveLength(2)
  expect(screen.queryAllByTestId('generated-token')).toHaveLength(0)
})

test('shows generated tokens up to cursor', () => {
  render(<TokenStream events={trace} cursor={trace.length - 1} />)
  expect(screen.getAllByTestId('generated-token')).toHaveLength(2)
  expect(screen.getAllByTestId('generated-token')[0]).toHaveTextContent('sat')
})
