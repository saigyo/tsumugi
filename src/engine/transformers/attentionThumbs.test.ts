import { expect, test } from 'vitest'
import { createAccumulator } from './attentionAccum'
import { headStats } from './attentionStats'
import { buildGridCells, poolThumb } from './attentionThumbs'

test('poolThumb on a 2-row matrix with 2 buckets is the identity with causal zeros', () => {
  expect(poolThumb([[1], [0.5, 0.5]], 2)).toEqual([[1, 0], [0.5, 0.5]])
})

test('poolThumb caps at the bucket count and stays within [0,1]', () => {
  const n = 30
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: i + 1 }, () => 1 / (i + 1)))
  const thumb = poolThumb(matrix)
  expect(thumb).toHaveLength(12)
  for (const row of thumb) {
    expect(row).toHaveLength(12)
    for (const v of row) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1) }
  }
  // upper-right block: entirely above the diagonal → 0
  expect(thumb[0][11]).toBe(0)
})

test('poolThumb uses fewer buckets than requested for short sequences', () => {
  const thumb = poolThumb([[1], [1, 0], [0, 1, 0]], 12)
  expect(thumb).toHaveLength(3)
  expect(thumb[2]).toEqual([0, 1, 0])
})

test('poolThumb of an empty matrix is empty', () => {
  expect(poolThumb([])).toEqual([])
})

test('poolThumb block means average only the defined causal entries', () => {
  // 4 rows, 2 buckets: block (1,0) covers rows 2-3 × cols 0-1 → mean of 4 entries
  const m = [[1], [0, 1], [0.2, 0.4, 0.4], [0.1, 0.3, 0.3, 0.3]]
  const thumb = poolThumb(m, 2)
  expect(thumb[1][0]).toBeCloseTo((0.2 + 0.4 + 0.1 + 0.3) / 4, 10)
})

test('buildGridCells emits one cell per head in layer-major order with the stats', () => {
  const acc = createAccumulator(2, 2)
  for (let l = 0; l < 2; l++) for (let h = 0; h < 2; h++)
    acc.rows[l][h] = [[1], [0.5, 0.5]]
  const stats = headStats(acc, [{ id: 0, text: 'a' }, { id: 1, text: ' b' }])
  const cells = buildGridCells(acc, stats)
  expect(cells).toHaveLength(4)
  expect(cells.map((c) => [c.layer, c.head])).toEqual([[0, 0], [0, 1], [1, 0], [1, 1]])
  expect(cells[0].thumb).toEqual([[1, 0], [0.5, 0.5]])
  expect(cells[0].prevTokenScore).toBe(stats[0].prevTokenScore)
  expect(cells[0].distinctiveScore).toBe(stats[0].distinctiveScore)
})
