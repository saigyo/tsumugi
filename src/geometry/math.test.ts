import { expect, test } from 'vitest'
import { makeGeometryAsset } from '../test/geometryFixture'
import { cosine, isRenderableToken, poolRow, renderableNeighbors, similarityMatrix } from './math'

test('cosine of parallel, orthogonal and zero vectors', () => {
  expect(cosine([1, 2], [2, 4])).toBeCloseTo(1, 6)
  expect(cosine([1, 0], [0, 1])).toBeCloseTo(0, 6)
  expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1, 6)
  expect(cosine([0, 0], [1, 1])).toBe(0)
})

test('similarityMatrix is symmetric with a unit diagonal', () => {
  const m = similarityMatrix([[1, 0], [0, 1], [1, 1]])
  expect(m).toEqual([[1, 0, 0.707], [0, 1, 0.707], [0.707, 0.707, 1]])
})

test('isRenderableToken filters byte fallbacks, empties, whitespace and control chars', () => {
  expect(isRenderableToken(' cat')).toBe(true)
  expect(isRenderableToken('<0x0A>')).toBe(false)
  expect(isRenderableToken('')).toBe(false)
  expect(isRenderableToken('   ')).toBe(false)
  expect(isRenderableToken('ab')).toBe(false)
})

test('renderableNeighbors skips unrenderable entries and returns n', () => {
  const a = makeGeometryAsset()
  const n = renderableNeighbors(a, 6, 8)   // ids 7 and 8 are unrenderable and among 6's closest
  expect(n).toHaveLength(8)
  expect(n.map((x) => x.id)).not.toContain(7)
  expect(n.map((x) => x.id)).not.toContain(8)
  expect(n[0].text).toBe('t5')
  expect(n[0].sim).toBeGreaterThan(0.9)
})

test('renderableNeighbors returns empty array when n <= 0', () => {
  const a = makeGeometryAsset()
  expect(renderableNeighbors(a, 6, 0)).toEqual([])
  expect(renderableNeighbors(a, 6, -1)).toEqual([])
})

test('poolRow mean-pools into buckets and is the identity for short rows', () => {
  expect(poolRow([1, 2, 3, 4, 5, 6], 3)).toEqual([1.5, 3.5, 5.5])
  expect(poolRow([1, 2, 3], 8)).toEqual([1, 2, 3])
  expect(poolRow([], 4)).toEqual([])
  expect(poolRow(Array.from({ length: 576 }, (_, i) => i), 96)).toHaveLength(96)
})
