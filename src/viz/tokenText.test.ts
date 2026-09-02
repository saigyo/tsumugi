import { expect, test } from 'vitest'
import { visibleToken } from './tokenText'

test('a leading space becomes ␣; the rest of the text is untouched', () => {
  expect(visibleToken(' cat')).toBe('␣cat')
  expect(visibleToken('cat')).toBe('cat')
  expect(visibleToken('')).toBe('')
})

test('a run of leading spaces gets one marker per space', () => {
  expect(visibleToken('   ')).toBe('␣␣␣')
  expect(visibleToken('  cat')).toBe('␣␣cat')
})

test('newlines and tabs become visible glyphs', () => {
  expect(visibleToken('\n')).toBe('↵')
  expect(visibleToken('\n\n')).toBe('↵↵')
  expect(visibleToken('\r\n')).toBe('↵')
  expect(visibleToken('\t')).toBe('⇥')
  expect(visibleToken(' \n')).toBe('␣↵')
})
