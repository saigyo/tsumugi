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

// independent hsl → rgb, for the expected values
function hslToRgb(h: number, s: number, l: number): number[] {
  s /= 100; l /= 100
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => { const k = (n + h / 30) % 12; return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))) }
  return [f(0), f(8), f(4)]
}
const channels = (c: string) => c.match(/\d+/g)!.map(Number)
const near = (got: string, want: number[]) => channels(got).forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThanOrEqual(1))

test('seqRgb at the middle is the exact HSL mix of the real light ends', () => {
  const want = hslToRgb(211, 42.5, 59)   // hsl(211 42.5% 59%)
  near(seqRgb(0.5, 'hsl(211 30% 94%)', 'hsl(211 55% 24%)'), want)
  near(seqRgb(0.25, 'hsl(211 30% 94%)', 'hsl(211 55% 24%)'), hslToRgb(211, 36.25, 76.5))
})

test('seqRgb reads hex, and falls back to the light ends for anything it cannot parse', () => {
  expect(seqRgb(0, '#0a141e', '#6e7882')).toBe('rgb(10, 20, 30)')
  expect(seqRgb(1, '#6e7882', '#0a141e')).toBe('rgb(10, 20, 30)')
  const light = seqRgb(0.5, 'hsl(211 30% 94%)', 'hsl(211 55% 24%)')
  expect(seqRgb(0.5, 'oklch(0.9 0.02 250)', 'color(srgb 0 0 1)')).toBe(light)
  expect(seqRgb(0.5, '', 'banana')).toBe(light)
})
