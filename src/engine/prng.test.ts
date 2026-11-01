import { expect, test } from 'vitest'
import { cycleDraw, mulberry32, seedFromTokens } from './prng'

test('same seed → same sequence', () => {
  const a = mulberry32(42)
  const b = mulberry32(42)
  expect([a(), a(), a()]).toEqual([b(), b(), b()])
})

test('different seeds → different sequences', () => {
  expect(mulberry32(1)()).not.toBe(mulberry32(2)())
})

test('values in [0,1)', () => {
  const r = mulberry32(7)
  for (let i = 0; i < 100; i++) {
    const v = r()
    expect(v).toBeGreaterThanOrEqual(0)
    expect(v).toBeLessThan(1)
  }
})

test('seedFromTokens is deterministic and order-sensitive', () => {
  expect(seedFromTokens([1, 2, 3])).toBe(seedFromTokens([1, 2, 3]))
  expect(seedFromTokens([1, 2, 3])).not.toBe(seedFromTokens([3, 2, 1]))
})

test('cycleDraw is deterministic per seed and cycle and in [0, 1)', () => {
  expect(cycleDraw(42, 3)).toBe(cycleDraw(42, 3))
  expect(cycleDraw(42, 3)).not.toBe(cycleDraw(42, 4))
  expect(cycleDraw(42, 3)).not.toBe(cycleDraw(43, 3))
  for (let c = 0; c < 100; c++) {
    const d = cycleDraw(7, c)
    expect(d).toBeGreaterThanOrEqual(0)
    expect(d).toBeLessThan(1)
  }
})
