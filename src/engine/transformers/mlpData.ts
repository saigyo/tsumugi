import type { LayerVotes } from '../../trace/types'
import { r3, sameDims, type Tensorish } from './readout'

// The v4 export's MLP outputs turned into trace data. Pure, so the shape
// policy is unit-testable outside the worker (like readout.ts).
//   edit_ratios [1, fed, layers, 2] → attn and mlp rows per fed position
//   stream_parts [1, 1 + 2·layers, dims] · topk_embeds rows → per candidate
//   id its votes: embedding, attention per layer, MLP per layer
export interface MlpOutputs { edit_ratios?: Tensorish; stream_parts?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish }
export type EditsResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; attn: number[][]; mlp: number[][] }
export type VotesResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; byId: Map<number, LayerVotes> }

export function extractEdits(out: MlpOutputs, fed: number, layers: number): EditsResult {
  const t = out.edit_ratios
  if (!t) return { status: 'absent' }
  const want = [1, fed, layers, 2]
  if (!sameDims(t, want) || t.data.length !== fed * layers * 2)
    return { status: 'bad-shape', detail: `edit_ratios has shape [${t.dims.join(', ')}], expected [${want.join(', ')}]` }
  const at = (p: number, l: number, b: number) => r3(Number(t.data[(p * layers + l) * 2 + b]))
  const rows = (b: number) => Array.from({ length: fed }, (_, p) => Array.from({ length: layers }, (_, l) => at(p, l, b)))
  return { status: 'ok', attn: rows(0), mlp: rows(1) }
}

// Votes must reproduce the logit the card sits next to; a gap beyond
// rounding means the graph's parts are wrong, so the votes are dropped.
const tolerance = (logit: number) => 0.05 + 0.01 * Math.abs(logit)

export function computeVotes(out: MlpOutputs, layers: number, dims: number, logitOf: (id: number) => number): VotesResult {
  const { stream_parts: sp, topk_ids: ids, topk_embeds: emb } = out
  if (!sp) return { status: 'absent' }
  if (!ids || !emb) return { status: 'bad-shape', detail: 'stream_parts without topk_ids / topk_embeds' }
  const parts = 1 + 2 * layers
  if (!sameDims(sp, [1, parts, dims]) || sp.data.length !== parts * dims)
    return { status: 'bad-shape', detail: `stream_parts has shape [${sp.dims.join(', ')}], expected [1, ${parts}, ${dims}]` }
  const k = ids.dims[1]
  if (ids.dims.length !== 2 || !sameDims(emb, [1, k, dims]))
    return { status: 'bad-shape', detail: `topk_embeds has shape [${emb.dims.join(', ')}], expected [1, ${k}, ${dims}]` }
  const byId = new Map<number, LayerVotes>()
  for (let i = 0; i < k; i++) {
    const id = Number(ids.data[i])
    const v: number[] = []
    for (let p = 0; p < parts; p++) {
      let s = 0
      for (let d = 0; d < dims; d++) s += Number(sp.data[p * dims + d]) * Number(emb.data[i * dims + d])
      v.push(s)
    }
    const total = v.reduce((a, b) => a + b, 0)
    const logit = logitOf(id)
    if (Math.abs(total - logit) > tolerance(logit))
      return { status: 'bad-shape', detail: `votes for id ${id} add up to ${total.toFixed(3)}, logit is ${logit.toFixed(3)}` }
    byId.set(id, { embed: r3(v[0]), attn: v.slice(1, 1 + layers).map(r3), mlp: v.slice(1 + layers).map(r3) })
  }
  return { status: 'ok', byId }
}
