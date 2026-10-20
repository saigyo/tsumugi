import { afterEach, expect, test } from 'vitest'
import { setTokenPieces } from '../engine/tokenBytes'
import { visibleToken } from './tokenText'

afterEach(() => setTokenPieces(null))

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

test('a token passed whole shows the bytes of a partial character, with the usual glyphs', () => {
  setTokenPieces((id) => (id === 459 ? 'ĠâĢ' : undefined))
  expect(visibleToken({ id: 459, text: ' \uFFFD' })).toBe('␣<0xE2><0x80>')
  expect(visibleToken({ id: 7, text: ' cat' })).toBe('␣cat')
  expect(visibleToken(' \uFFFD')).toBe('␣\uFFFD')
})
