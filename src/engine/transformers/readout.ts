import { poolRow } from '../../geometry/math'
import { READOUT_CELLS } from '../../trace/types'

// Turns the v3 export's readout outputs into the logits event's fields. Pure,
// so the shape policy and the pooling are unit-testable outside the worker.
//   last_hidden_state [1, fed, dims] → the last fed row, exact (3 decimals)
//   topk_ids [1, k] + topk_embeds [1, k, dims] → per candidate id: its row of E
//   pooled by MEAN and the product x_last ⊙ E[id] pooled by SUM, so the dot
//   cells add up to the logit.
export type Tensorish = { dims: number[]; data: ArrayLike<number> | BigInt64Array }
export interface ReadoutOutputs { last_hidden_state?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish }
export type ReadoutResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; lastHidden: number[]; byId: Map<number, { row: number[]; dot: number[] }> }

export const r3 = (v: number) => Math.round(v * 1000) / 1000

function poolSum(values: number[], cells: number): number[] {
  const len = values.length
  const out: number[] = []
  for (let c = 0; c < cells; c++) {
    const start = Math.floor((c * len) / cells), end = Math.floor(((c + 1) * len) / cells)
    let s = 0
    for (let i = start; i < end; i++) s += values[i]
    out.push(s)
  }
  return out
}

export const sameDims = (t: Tensorish, want: number[]) =>
  t.dims.length === want.length && t.dims.every((d, i) => d === want[i])

export function extractReadout(out: ReadoutOutputs, fed: number, dims: number, cells = READOUT_CELLS): ReadoutResult {
  const { last_hidden_state: lh, topk_ids: ids, topk_embeds: emb } = out
  if (!lh && !ids && !emb) return { status: 'absent' }
  if (!lh || !ids || !emb)
    return { status: 'bad-shape', detail: `readout outputs incomplete: last_hidden_state ${!!lh}, topk_ids ${!!ids}, topk_embeds ${!!emb}` }
  if (!sameDims(lh, [1, fed, dims]) || lh.data.length !== fed * dims)
    return { status: 'bad-shape', detail: `last_hidden_state has shape [${lh.dims.join(', ')}], expected [1, ${fed}, ${dims}]` }
  const k = ids.dims[1]
  if (ids.dims.length !== 2 || ids.dims[0] !== 1 || ids.data.length !== k)
    return { status: 'bad-shape', detail: `topk_ids has shape [${ids.dims.join(', ')}], expected [1, k]` }
  if (!sameDims(emb, [1, k, dims]) || emb.data.length !== k * dims)
    return { status: 'bad-shape', detail: `topk_embeds has shape [${emb.dims.join(', ')}], expected [1, ${k}, ${dims}]` }

  const base = (fed - 1) * dims
  const x = Array.from({ length: dims }, (_, d) => Number(lh.data[base + d]))
  const byId = new Map<number, { row: number[]; dot: number[] }>()
  for (let i = 0; i < k; i++) {
    const id = Number(ids.data[i])
    const row = Array.from({ length: dims }, (_, d) => Number(emb.data[i * dims + d]))
    byId.set(id, {
      row: poolRow(row, cells).map(r3),
      dot: poolSum(row.map((v, d) => v * x[d]), cells).map(r3),
    })
  }
  return { status: 'ok', lastHidden: x.map(r3), byId }
}
