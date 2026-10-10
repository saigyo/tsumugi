import { expect, test } from 'vitest'
import { computeVotes, extractEdits } from './mlpData'

const L = 2, DIMS = 3

function ratios(fed: number) {
  // value = pos + layer/10 + block/100, so every cell is distinguishable
  const data = new Float32Array(fed * L * 2)
  for (let p = 0; p < fed; p++) for (let l = 0; l < L; l++) for (let b = 0; b < 2; b++)
    data[(p * L + l) * 2 + b] = p + l / 10 + b / 100
  return { dims: [1, fed, L, 2], data }
}

test('extractEdits: absent, bad shape, ok split into attention and MLP rows', () => {
  expect(extractEdits({}, 2, L)).toEqual({ status: 'absent' })
  const bad = extractEdits({ edit_ratios: ratios(3) }, 2, L)
  expect(bad.status).toBe('bad-shape')
  if (bad.status === 'bad-shape') expect(bad.detail).toMatch(/edit_ratios has shape \[1, 3, 2, 2\], expected \[1, 2, 2, 2\]/)
  const r = extractEdits({ edit_ratios: ratios(2) }, 2, L)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.attn).toEqual([[0, 0.1], [1, 1.1]])
  expect(r.mlp).toEqual([[0.01, 0.11], [1.01, 1.11]])
})

// parts: embedding, attn L0, attn L1, mlp L0, mlp L1 — rows of 3 dims
const PARTS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.5, 0, 0.5]]
const ROWS: Record<number, number[]> = { 7: [2, 0, 1], 4: [0, 3, 0] }
const exact = (id: number) => PARTS.map((p) => p.reduce((s, v, d) => s + v * ROWS[id][d], 0))

function voteOutputs(ids = [7, 4]) {
  return {
    stream_parts: { dims: [1, 1 + 2 * L, DIMS], data: Float32Array.from(PARTS.flat()) },
    topk_ids: { dims: [1, ids.length], data: BigInt64Array.from(ids.map(BigInt)) },
    topk_embeds: { dims: [1, ids.length, DIMS], data: Float32Array.from(ids.flatMap((id) => ROWS[id])) },
  }
}
const logitOf = (id: number) => exact(id).reduce((a, b) => a + b, 0)

test('computeVotes: per candidate, the parts dotted with its row, split by block', () => {
  const r = computeVotes(voteOutputs(), L, DIMS, logitOf)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.byId.get(7)).toEqual({ embed: 2, attn: [0, 1], mlp: [2, 1.5] })
  expect(r.byId.get(4)).toEqual({ embed: 0, attn: [3, 0], mlp: [3, 0] })
})

test('computeVotes: absent without stream_parts; bad shape names the output', () => {
  expect(computeVotes({}, L, DIMS, logitOf)).toEqual({ status: 'absent' })
  const o = voteOutputs()
  o.stream_parts.dims = [1, 4, DIMS]
  const r = computeVotes(o, L, DIMS, logitOf)
  if (r.status !== 'bad-shape') throw new Error(r.status)
  expect(r.detail).toMatch(/stream_parts/)
})

test('computeVotes: votes that do not add up to the logit are refused', () => {
  const r = computeVotes(voteOutputs(), L, DIMS, (id) => logitOf(id) + 1)
  if (r.status !== 'bad-shape') throw new Error(r.status)
  expect(r.detail).toMatch(/votes for id 7 add up to/)
})
