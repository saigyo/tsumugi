import { packFloat32, packHalf } from '../../trace/neuronPack'
import type { MlpSteps, NeuronData } from '../../trace/types'
import { sameDims, type Tensorish } from './readout'

// The v5 export's neuron outputs turned into the edits event's `neurons`
// field. Pure, so the shape policy is unit-testable outside the worker.
//   mlp_activations [1, layers, neurons] → float16 base64, layer-major
//   final_norm_inv_rms [1, 1, 1] → invRms
//   topk_ids + topk_embeds → the full rows of the candidates the card lists
export interface NeuronOutputs {
  mlp_activations?: Tensorish; final_norm_inv_rms?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish
}
export type NeuronResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; data: NeuronData }

const bad = (detail: string): NeuronResult => ({ status: 'bad-shape', detail })

export function extractNeurons(out: NeuronOutputs, layers: number, dims: number, candidateIds: number[]): NeuronResult {
  const { mlp_activations: act, final_norm_inv_rms: inv, topk_ids: ids, topk_embeds: emb } = out
  if (!act && !inv) return { status: 'absent' }
  if (!act || !inv || !ids || !emb)
    return bad(`neuron outputs incomplete: mlp_activations ${!!act}, final_norm_inv_rms ${!!inv}, topk_ids ${!!ids}, topk_embeds ${!!emb}`)
  const neurons = act.dims[2]
  if (act.dims.length !== 3 || act.dims[0] !== 1 || act.dims[1] !== layers || !(neurons > 0) || act.data.length !== layers * neurons)
    return bad(`mlp_activations has shape [${act.dims.join(', ')}], expected [1, ${layers}, neurons]`)
  if (inv.data.length !== 1) return bad(`final_norm_inv_rms has ${inv.data.length} values, expected 1`)
  const invRms = Number(inv.data[0])
  if (!Number.isFinite(invRms) || invRms <= 0) return bad(`final_norm_inv_rms is ${invRms}, expected a positive 1/rms`)
  const k = ids.dims[1]
  if (ids.dims.length !== 2 || !sameDims(emb, [1, k, dims]) || emb.data.length !== k * dims)
    return bad(`topk_embeds has shape [${emb.dims.join(', ')}], expected [1, ${k}, ${dims}]`)
  const index = new Map<number, number>()
  for (let i = 0; i < k; i++) index.set(Number(ids.data[i]), i)
  const candidates: NeuronData['candidates'] = []
  for (const id of candidateIds) {
    const i = index.get(id)
    if (i === undefined) return bad(`candidate ${id} is not among topk_ids`)
    const row = new Float32Array(dims)
    for (let d = 0; d < dims; d++) row[d] = Number(emb.data[i * dims + d])
    candidates.push({ id, row: packFloat32(row) })
  }
  return { status: 'ok', data: { neurons, invRms, activations: packHalf(act.data as ArrayLike<number>), candidates } }
}

// The v6 export's step outputs for the step-by-step card, all last position:
//   mlp_gate, mlp_up [1, layers, neurons]; mlp_in, mlp_out [1, layers, dims]
export interface StepOutputs { mlp_gate?: Tensorish; mlp_up?: Tensorish; mlp_in?: Tensorish; mlp_out?: Tensorish }
export type StepsResult = { status: 'absent' } | { status: 'bad-shape'; detail: string } | { status: 'ok'; steps: MlpSteps }

export function extractSteps(out: StepOutputs, layers: number, neurons: number, dims: number): StepsResult {
  const { mlp_gate: g, mlp_up: u, mlp_in: x, mlp_out: e } = out
  if (!g && !u && !x && !e) return { status: 'absent' }
  if (!g || !u || !x || !e)
    return { status: 'bad-shape', detail: `step outputs incomplete: mlp_gate ${!!g}, mlp_up ${!!u}, mlp_in ${!!x}, mlp_out ${!!e}` }
  const want: Array<[string, Tensorish, number]> = [['mlp_gate', g, neurons], ['mlp_up', u, neurons], ['mlp_in', x, dims], ['mlp_out', e, dims]]
  for (const [name, t, w] of want)
    if (!sameDims(t, [1, layers, w]) || t.data.length !== layers * w)
      return { status: 'bad-shape', detail: `${name} has shape [${t.dims.join(', ')}], expected [1, ${layers}, ${w}]` }
  const half = (t: Tensorish) => packHalf(t.data as ArrayLike<number>)
  return { status: 'ok', steps: { gate: half(g), up: half(u), mlpIn: half(x), mlpOut: half(e) } }
}
