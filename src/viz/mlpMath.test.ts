import { expect, test } from 'vitest'
import { blockTotals, ratioFill, runningTotal, topMlpLayers, votePeak } from './mlpMath'

const v = { embed: 1, attn: [0.5, -0.25], mlp: [2, -1] }

test('runningTotal: embedding first, then per layer attention then MLP; ends on the sum', () => {
  expect(runningTotal(v)).toEqual([1, 1.5, 3.5, 3.25, 2.25])
})

test('blockTotals splits the score into embedding, all attention, all MLP', () => {
  expect(blockTotals(v)).toEqual({ embed: 1, attn: 0.25, mlp: 1 })
})

test('topMlpLayers ranks by absolute vote', () => {
  expect(topMlpLayers({ embed: 0, attn: [0, 0, 0], mlp: [0.1, -3, 2] }, 2)).toEqual([
    { layer: 1, value: -3 }, { layer: 2, value: 2 },
  ])
})

test('votePeak is the largest absolute MLP vote, at least a tiny positive number', () => {
  expect(votePeak([{ votes: v }, { votes: { embed: 9, attn: [9, 9], mlp: [-4, 0] } }])).toBe(4)
  expect(votePeak([{ votes: { embed: 0, attn: [0], mlp: [0] } }])).toBe(1)
})

test('ratioFill darkens with the ratio and clamps at 1', () => {
  expect(ratioFill(0)).toBe('hsl(211 30% 94%)')
  expect(ratioFill(1)).toBe('hsl(211 55% 24%)')
  expect(ratioFill(3)).toBe(ratioFill(1))
})
