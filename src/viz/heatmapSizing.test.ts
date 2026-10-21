import { expect, test } from 'vitest'
import { cellSize, labelStep } from './heatmapSizing'

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
