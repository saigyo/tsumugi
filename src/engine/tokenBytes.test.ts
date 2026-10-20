import { afterEach, expect, test } from 'vitest'
import { pieceBytes, readableBytes, readableTokenText, setTokenPieces } from './tokenBytes'

afterEach(() => setTokenPieces(null))

test('pieceBytes undoes the byte-level alphabet: Ġ is a space, printable ASCII maps to itself', () => {
  expect(pieceBytes('Ġcat')).toEqual([0x20, 0x63, 0x61, 0x74])
  expect(pieceBytes('Ċ')).toEqual([0x0a])
  expect(pieceBytes('ĠâĢ')).toEqual([0x20, 0xe2, 0x80])
  expect(pieceBytes('¡')).toEqual([0xa1])
})

test('pieceBytes refuses a character outside the byte alphabet', () => {
  expect(pieceBytes('<|endoftext|>x一')).toBeNull()
})

test('readableBytes keeps whole UTF-8 characters and spells out the stray bytes', () => {
  expect(readableBytes([0x20, 0xe2, 0x80])).toBe(' <0xE2><0x80>')
  expect(readableBytes([0xd0, 0xbe, 0xd0])).toBe('о<0xD0>')
  expect(readableBytes([0xa1])).toBe('<0xA1>')
  expect(readableBytes([0xe2, 0x82, 0xac, 0x21])).toBe('€!')
  expect(readableBytes([0xed, 0xa0, 0x80])).toBe('<0xED><0xA0><0x80>')   // a surrogate is not a character
})

test('readableTokenText rewrites only texts with a replacement character, and only when pieces are known', () => {
  expect(readableTokenText(459, ' �')).toBe(' �')
  setTokenPieces((id) => (id === 459 ? 'ĠâĢ' : undefined))
  expect(readableTokenText(459, ' �')).toBe(' <0xE2><0x80>')
  expect(readableTokenText(7, ' cat')).toBe(' cat')
  expect(readableTokenText(8, '�')).toBe('�')
})
