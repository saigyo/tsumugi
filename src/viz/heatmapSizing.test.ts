import { expect, test } from 'vitest'
import { cellSize, labelMargins, labelStep, textWidth } from './heatmapSizing'

test('short runs keep 20 px cells', () => {
  expect(cellSize(928, 20, 136)).toBe(20)
  expect(cellSize(928, 39, 136)).toBe(20)
})

test('longer runs shrink to the card width, never below 8 px', () => {
  expect(cellSize(928, 57, 136)).toBe(13)    // floor(792 / 57)
  expect(cellSize(928, 137, 136)).toBe(8)
})

test('before the first measurement the cell is 20 px', () => {
  expect(cellSize(null, 137, 136)).toBe(20)
})

test('labels thin out below 14 px', () => {
  expect(labelStep(20)).toBe(1)
  expect(labelStep(14)).toBe(1)
  expect(labelStep(13)).toBe(2)
  expect(labelStep(8)).toBe(2)
  expect(labelStep(6)).toBe(3)
})

const MINIMUM = { minLeft: 66, minTop: 58, minRight: 70 }

test('short labels keep the old margins', () => {
  expect(labelMargins(['the', 'cat'], ['the', 'cat'], MINIMUM)).toEqual({ left: 66, top: 58, right: 70 })
})

test('special tokens widen the row margin to their full text', () => {
  const m = labelMargins(['<|im_start|>', 'a'], ['a', 'b'], MINIMUM)
  expect(m.left).toBeGreaterThanOrEqual(textWidth('<|im_start|>') + 8)
  expect(m.left).toBeGreaterThan(66)
})

test('a long column label raises the top margin and the folded one gets half a row more', () => {
  const plain = labelMargins([], ['system prompt (20)'], MINIMUM)
  const folded = labelMargins([], ['system prompt (20)'], { ...MINIMUM, folded: true })
  expect(plain.top).toBeGreaterThan(textWidth('system prompt (20)') * 0.7)
  expect(folded.top).toBeGreaterThan(plain.top)
})

test('the right margin covers a long label near the last column only', () => {
  const near = labelMargins([], ['a', 'b', 'system prompt (20)'], MINIMUM)
  const far = labelMargins([], ['system prompt (20)', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], MINIMUM)
  expect(near.right).toBeGreaterThan(70)
  expect(far.right).toBe(70)
})
