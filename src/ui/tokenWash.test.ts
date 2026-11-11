import { expect, test } from 'vitest'
import { washIndexes } from './tokenWash'

test('a new token takes the next wash, a repeated token reuses its own', () => {
  expect(washIndexes([504, 2644, 2643, 335, 260, 1171, 975, 260, 2644, 436]))
    .toEqual([0, 1, 2, 3, 4, 5, 0, 4, 1, 2])   // 436 would take 1, the wash just before it, so it takes 2
})

test('no tokens, no washes; one token repeated keeps one wash', () => {
  expect(washIndexes([])).toEqual([])
  expect(washIndexes([7, 7, 7])).toEqual([0, 0, 0])
})

test('a longer list agrees with its prefix', () => {
  const ids = [5, 6, 5, 7, 8, 9, 10, 11, 6, 12]
  expect(washIndexes(ids).slice(0, 6)).toEqual(washIndexes(ids.slice(0, 6)))
})
