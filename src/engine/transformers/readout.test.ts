import { expect, test } from 'vitest'
import { extractReadout } from './readout'

const DIMS = 8
const CELLS = 4   // 2 dims per cell
const row = (seed: number) => Array.from({ length: DIMS }, (_, d) => Math.round(Math.sin(seed + d) * 1000) / 1000)
const last = row(1)
const E: Record<number, number[]> = { 5: row(2), 9: row(3), 2: row(4) }

function outputs(ids: Array<number | bigint> = [5, 9, 2], fed = 2) {
  const hidden = new Float32Array(fed * DIMS)
  hidden.set(row(99), 0)                 // an earlier fed token, must be ignored
  hidden.set(last, (fed - 1) * DIMS)     // the last fed token
  const embeds = new Float32Array(ids.length * DIMS)
  ids.forEach((id, i) => embeds.set(E[Number(id)], i * DIMS))
  const idData = typeof ids[0] === 'bigint' ? BigInt64Array.from(ids as bigint[]) : Int32Array.from(ids as number[])
  return {
    last_hidden_state: { dims: [1, fed, DIMS], data: hidden },
    topk_ids: { dims: [1, ids.length], data: idData },
    topk_embeds: { dims: [1, ids.length, DIMS], data: embeds },
  }
}

test('absent outputs → absent; a partial set → bad-shape', () => {
  expect(extractReadout({}, 2, DIMS, CELLS)).toEqual({ status: 'absent' })
  expect(extractReadout({ last_hidden_state: outputs().last_hidden_state }, 2, DIMS, CELLS).status).toBe('bad-shape')
})

test('wrong shapes → bad-shape naming the output', () => {
  const o = outputs()
  o.last_hidden_state.dims = [1, 3, DIMS]
  const r = extractReadout(o, 2, DIMS, CELLS)
  expect(r.status).toBe('bad-shape')
  if (r.status === 'bad-shape') expect(r.detail).toMatch(/last_hidden_state/)
  const p = outputs()
  p.topk_embeds.dims = [1, 3, DIMS + 1]
  const q = extractReadout(p, 2, DIMS, CELLS)
  if (q.status === 'bad-shape') expect(q.detail).toMatch(/topk_embeds/)
  else throw new Error('expected bad-shape')
})

test('ok → last fed row rounded to 3 decimals, rows pooled by mean, dots by sum, keyed by id', () => {
  const r = extractReadout(outputs(), 2, DIMS, CELLS)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.lastHidden).toEqual(last.map((v) => Math.round(v * 1000) / 1000))
  expect([...r.byId.keys()]).toEqual([5, 9, 2])
  const e5 = E[5]
  const mean = (a: number, b: number) => Math.round(((a + b) / 2) * 1000) / 1000
  expect(r.byId.get(5)!.row).toEqual([mean(e5[0], e5[1]), mean(e5[2], e5[3]), mean(e5[4], e5[5]), mean(e5[6], e5[7])])
  const dot = r.byId.get(5)!.dot
  expect(dot).toHaveLength(CELLS)
  const exact = e5.reduce((s, v, d) => s + v * last[d], 0)
  expect(dot.reduce((a, b) => a + b, 0)).toBeCloseTo(exact, 2)
})

test('accepts BigInt64Array ids', () => {
  const r = extractReadout(outputs([5n, 9n, 2n]), 2, DIMS, CELLS)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.byId.has(9)).toBe(true)
})

test('id missing from topk_ids is absent from the map', () => {
  const r = extractReadout(outputs([5, 9, 2]), 2, DIMS, CELLS)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.byId.get(7)).toBeUndefined()
})
