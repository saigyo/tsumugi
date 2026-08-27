import { cleanup, fireEvent, render, screen } from '@testing-library/react'
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

test('hovering a generated token shows its sampling distribution', () => {
  render(<TokenStream events={trace} cursor={trace.length - 1} />)
  fireEvent.mouseEnter(screen.getAllByTestId('generated-token')[0])
  const pop = screen.getByTestId('token-popover')
  expect(pop).toHaveTextContent('sat')
  expect(pop).toHaveTextContent('70%')
  expect(pop).toHaveTextContent('ran')
  expect(pop).toHaveTextContent('20%')
  const chosen = screen.getAllByTestId('popover-row').filter((r) => r.dataset.chosen === 'true')
  expect(chosen).toHaveLength(1)
  expect(chosen[0]).toHaveTextContent('sat')
})

test('mouse leave hides the popover; prompt tokens have none', () => {
  render(<TokenStream events={trace} cursor={trace.length - 1} />)
  const tok = screen.getAllByTestId('generated-token')[1]
  fireEvent.mouseEnter(tok)
  expect(screen.getByTestId('token-popover')).toBeInTheDocument()
  fireEvent.mouseLeave(tok)
  expect(screen.queryByTestId('token-popover')).not.toBeInTheDocument()
  fireEvent.mouseEnter(screen.getAllByTestId('prompt-token')[0])
  expect(screen.queryByTestId('token-popover')).not.toBeInTheDocument()
})
