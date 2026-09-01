import { expect, test } from 'vitest'
import { extractEmbedRows } from './embedRows'

test('absent output → absent', () => {
  expect(extractEmbedRows(undefined, 3, 4)).toEqual({ status: 'absent' })
})

test('wrong shape → bad-shape with the offending dims', () => {
  const t = { dims: [1, 2, 4], data: new Float32Array(8) }
  expect(extractEmbedRows(t, 3, 4)).toEqual({ status: 'bad-shape', dims: [1, 2, 4] })
  expect(extractEmbedRows({ dims: [2, 4], data: new Float32Array(8) }, 2, 4).status).toBe('bad-shape')
})

test('ok → one row per fed token, rounded to 3 decimals', () => {
  const data = new Float32Array([0.12345, -1.00049, 2, 3, 4.4444, 5, 6, 7])
  const r = extractEmbedRows({ dims: [1, 2, 4], data }, 2, 4)
  expect(r).toEqual({ status: 'ok', rows: [[0.123, -1, 2, 3], [4.444, 5, 6, 7]] })
})
