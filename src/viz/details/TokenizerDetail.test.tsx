import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { CHAT_PROMPT_TOKENS } from '../../test/fixtures'
import { TokenizerDetail } from './TokenizerDetail'

afterEach(() => cleanup())

const chat = { system: [1, 21] as [number, number], user: [24, 31] as [number, number] }
const chip = (text: string) =>
  [...document.querySelectorAll('.token-chip')].find((c) => c.querySelector('.chip-text')?.textContent === text)!

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
