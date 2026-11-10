import { expect, test } from 'vitest'
import { divFill, seqFill, seqRgb } from './ramps'

test('seqFill mixes the ramp ends and clamps', () => {
  expect(seqFill(0)).toBe('color-mix(in hsl, var(--ramp-hi) 0%, var(--ramp-lo))')
  expect(seqFill(1)).toBe('color-mix(in hsl, var(--ramp-hi) 100%, var(--ramp-lo))')
  expect(seqFill(0.123)).toBe('color-mix(in hsl, var(--ramp-hi) 12.3%, var(--ramp-lo))')
  expect(seqFill(1.7)).toBe(seqFill(1)); expect(seqFill(-3)).toBe(seqFill(0)); expect(seqFill(NaN)).toBe(seqFill(0))
})

test('divFill uses each side its own ends', () => {
  expect(divFill(5, 10)).toBe('color-mix(in hsl, var(--div-pos) 50%, var(--div-pos-lo))')
  expect(divFill(-10, 10)).toBe('color-mix(in hsl, var(--div-neg) 100%, var(--div-neg-lo))')
  expect(divFill(0, 10)).toBe('color-mix(in hsl, var(--div-pos) 0%, var(--div-pos-lo))')
  expect(divFill(3, 0)).toBe(divFill(1, 1))
})

test('seqRgb mixes in JS', () => {
  expect(seqRgb(0, 'rgb(10, 20, 30)', 'rgb(110, 120, 130)')).toBe('rgb(10, 20, 30)')
  expect(seqRgb(1, 'rgb(10, 20, 30)', 'rgb(110, 120, 130)')).toBe('rgb(110, 120, 130)')
  expect(seqRgb(2, 'rgb(10, 20, 30)', 'rgb(110, 120, 130)')).toBe('rgb(110, 120, 130)')
})
