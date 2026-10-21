import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../test/fixtures'
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

test('whitespace tokens are shown as glyphs instead of blank chips', () => {
  const t = buildFixtureTrace({ promptTokens: [{ id: 504, text: 'The' }, { id: 11, text: ' cat' }],
    chosen: [{ id: 30, text: '.' }, { id: 198, text: '\n' }] })
  render(<TokenStream events={t} cursor={t.length - 1} />)
  expect(screen.getAllByTestId('prompt-token')[1]).toHaveTextContent('␣cat')
  expect(screen.getAllByTestId('generated-token')[1]).toHaveTextContent('↵')
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

test('a chat run folds the system turn after the first token; the chip opens and closes it', () => {
  const events = buildFixtureTrace({ chat: true })
  render(<TokenStream events={events} cursor={events.length - 1} />)
  const fold = screen.getByTestId('system-fold')
  expect(fold).toHaveTextContent('system prompt · 20 tokens ▸')
  expect(fold).toHaveAttribute('aria-expanded', 'false')
  expect(screen.getAllByTestId('prompt-token')).toHaveLength(37 - 20)
  expect(screen.getAllByTestId('prompt-token')[0]).toHaveTextContent('<|im_start|>')
  fireEvent.click(fold)
  expect(screen.getAllByTestId('prompt-token')).toHaveLength(37)
  expect(screen.getByTestId('system-fold')).toHaveTextContent('▾')
})

test('chat tokens carry their role; the generated <|im_end|> is marked as the end of the turn', () => {
  const events = buildFixtureTrace({ chat: true })
  render(<TokenStream events={events} cursor={events.length - 1} />)
  const prompt = screen.getAllByTestId('prompt-token')
  expect(prompt[0]).toHaveClass('token-special')
  expect(prompt.find((t) => t.textContent === 'user')).toHaveClass('token-template')
  expect(prompt.find((t) => t.textContent === 'What')).not.toHaveClass('token-template')
  const last = screen.getAllByTestId('generated-token').at(-1)!
  expect(last).toHaveClass('token-end')
  expect(last).toHaveAttribute('title', expect.stringContaining('end of turn'))
})

test('a chat run without a system turn has no fold chip', () => {
  const events = buildFixtureTrace({ chat: true })
  const t = events.find((e) => e.type === 'tokenize')!
  if (t.type === 'tokenize') t.chat = { user: [24, 31] }
  render(<TokenStream events={events} cursor={events.length - 1} />)
  expect(screen.queryByTestId('system-fold')).toBeNull()
})

test('a plain run looks as before: no fold chip, no template classes', () => {
  const events = buildFixtureTrace()
  render(<TokenStream events={events} cursor={events.length - 1} />)
  expect(screen.queryByTestId('system-fold')).toBeNull()
  expect(screen.getAllByTestId('prompt-token').every((t) => !t.className.includes('token-template'))).toBe(true)
})
