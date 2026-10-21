import { expect, test } from 'vitest'
import { foldItems, foldLabel, tokenRole } from './chatFold'

test('closed: the first token, then one fold item, then everything after the system turn', () => {
  expect(foldItems(6, [1, 4], false)).toEqual([
    { kind: 'token', index: 0 }, { kind: 'fold', count: 3 }, { kind: 'token', index: 4 }, { kind: 'token', index: 5 }])
})

test('open: the fold item stays in place and the folded tokens follow it', () => {
  expect(foldItems(4, [1, 3], true)).toEqual([
    { kind: 'token', index: 0 }, { kind: 'fold', count: 2 }, { kind: 'token', index: 1 },
    { kind: 'token', index: 2 }, { kind: 'token', index: 3 }])
})

test('without a system span every token is its own item', () => {
  expect(foldItems(3, undefined, false)).toEqual([0, 1, 2].map((index) => ({ kind: 'token', index })))
})

test('roles: special tokens by text, the user span as prompt, the rest as template; no spans → prompt', () => {
  const chat = { system: [1, 3] as [number, number], user: [5, 7] as [number, number] }
  expect(tokenRole(0, '<|im_start|>', chat)).toBe('special')
  expect(tokenRole(1, 'system', chat)).toBe('template')
  expect(tokenRole(5, 'What', chat)).toBe('prompt')
  expect(tokenRole(7, '<|im_end|>', chat)).toBe('special')
  expect(tokenRole(8, '\n', chat)).toBe('template')
  expect(tokenRole(0, '<|im_start|>', undefined)).toBe('prompt')
})

test('fold label', () => {
  expect(foldLabel(20, false)).toBe('system prompt · 20 tokens ▸')
  expect(foldLabel(20, true)).toBe('system prompt · 20 tokens ▾')
})

test('an empty or reversed system span is not folded', () => {
  expect(foldItems(5, [3, 3], false)).toHaveLength(5)
  expect(foldItems(5, [4, 2], false)).toHaveLength(5)
})
