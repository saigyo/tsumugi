import { expect, test } from 'vitest'
import { mulberry32 } from '../prng'
import { candidateWords } from './candidates'

test('returns 10 distinct candidates', () => {
  const c = candidateWords('The cat', mulberry32(1))
  expect(c).toHaveLength(10)
  expect(new Set(c).size).toBe(10)
})

test('words have leading space, punctuation does not', () => {
  const c = candidateWords('The cat', mulberry32(1))
  for (const w of c) expect(w.startsWith(' ') || w === ',' || w === '.').toBe(true)
})

test('capitalizes after sentence end', () => {
  const c = candidateWords('It was late.', mulberry32(2))
  const words = c.filter((w) => w.startsWith(' '))
  expect(words.every((w) => /^[A-Z]/.test(w.trimStart()))).toBe(true)
})

test('offers a period late in a long sentence', () => {
  const long = 'the quick brown fox jumps over the lazy dog again and again'
  expect(candidateWords(long, mulberry32(3))).toContain('.')
  expect(candidateWords('Hi', mulberry32(3))).toContain(',')
})

test('deterministic for same rand seed', () => {
  expect(candidateWords('a b c', mulberry32(9))).toEqual(candidateWords('a b c', mulberry32(9)))
})
