import { expect, test } from 'vitest'
import { forcedToken, mlpScale } from './knockout'

test('mlpScale with no knockout is all ones', () => {
  const s = mlpScale(30, null)
  expect(s).toBeInstanceOf(Float32Array)
  expect(s).toHaveLength(30)
  expect(Array.from(s).every((v) => v === 1)).toBe(true)
})

test('mlpScale zeroes only the knocked-out layer', () => {
  const s = mlpScale(30, 15)
  expect(s[15]).toBe(0)
  expect(Array.from(s).filter((v) => v === 0)).toHaveLength(1)
})

test('mlpScale ignores an out-of-range layer', () => {
  expect(Array.from(mlpScale(4, 9)).every((v) => v === 1)).toBe(true)
})

test('forcedToken returns the copied prefix then null', () => {
  expect(forcedToken([5, 9], 0)).toBe(5)
  expect(forcedToken([5, 9], 1)).toBe(9)
  expect(forcedToken([5, 9], 2)).toBeNull()
})
