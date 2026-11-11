import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { CHAT_PROMPT_TOKENS } from '../../test/fixtures'
import { TokenizerDetail } from './TokenizerDetail'
import { washIndexes } from '../../ui/tokenWash'

afterEach(() => cleanup())

const chat = { system: [1, 21] as [number, number], user: [24, 31] as [number, number] }
const chip = (text: string) =>
  [...document.querySelectorAll('.tok')].find((c) => c.querySelector('.chip-text')?.textContent === text)!

test('special tokens are marked as special, like in the token strip; other template tokens stay dashed grey', () => {
  render(<TokenizerDetail tokens={CHAT_PROMPT_TOKENS} chat={chat} />)
  expect(chip('<|im_start|>')).toHaveClass('token-special')
  expect(chip('<|im_start|>')).not.toHaveClass('token-template')
  expect(chip('user')).toHaveClass('token-template')
  expect(chip('user')).not.toHaveClass('token-special')
})

test('new tokens start with the fold closed again', () => {
  const { rerender } = render(<TokenizerDetail tokens={CHAT_PROMPT_TOKENS} chat={chat} />)
  fireEvent.click(screen.getByTestId('system-fold'))
  expect(screen.getByTestId('system-fold')).toHaveAttribute('aria-expanded', 'true')
  rerender(<TokenizerDetail tokens={[...CHAT_PROMPT_TOKENS]} chat={chat} />)
  expect(screen.getByTestId('system-fold')).toHaveAttribute('aria-expanded', 'false')
})

const washOf = (el: Element) => Number(/tok-wash-(\d)/.exec(el.className)?.[1])

test('chips take their wash by token id over the whole prompt, also with the system prompt folded', () => {
  render(<TokenizerDetail tokens={CHAT_PROMPT_TOKENS} chat={chat} />)
  const expected = washIndexes(CHAT_PROMPT_TOKENS.map((t) => t.id))
  const shown = [...document.querySelectorAll('.tok:not(.token-fold)')]
  // the folded system turn takes tokens 1 to 20 out of the list, but not out of the count
  expect(shown.map(washOf)).toEqual([expected[0], ...expected.slice(21)])
})

test('two chips with the same token id carry the same wash class', () => {
  render(<TokenizerDetail tokens={CHAT_PROMPT_TOKENS} chat={chat} />)
  fireEvent.click(screen.getByTestId('system-fold'))
  const chips = [...document.querySelectorAll('.tok:not(.token-fold)')]
  const byId = new Map<string, Set<number>>()
  chips.forEach((c) => {
    const id = c.querySelector('.tok-id')!.textContent!
    byId.set(id, (byId.get(id) ?? new Set()).add(washOf(c)))
  })
  expect(chips.length).toBe(CHAT_PROMPT_TOKENS.length)
  expect([...byId.values()].every((s) => s.size === 1)).toBe(true)
  expect(byId.size).toBeLessThan(chips.length)   // the prompt does repeat ids
})
