import type { NeuronLayer, NeuronManifest } from './format'

// The numbers behind "Inside one MLP" (#16 spec, section 3). With W_i the
// dequantized column of neuron i, a_i its activation and u = g ⊙ E_c:
//   vote_i = a_i · (W_i · u) · 1/rms        (they add up to the MLP's vote)
//   size_i = |a_i| · ‖W_i‖                  (how far the neuron moved the vector)

export type Badge = 'top 0.1%' | 'top 1%' | 'top 10%' | 'not in top 10%'
const BADGES: Badge[] = ['top 0.1%', 'top 1%', 'top 10%']

export interface VoteRow { neuron: number; activation: number; vote: number; labels: number[]; away: boolean; badge: Badge }
export interface VoteTable { rows: VoteRow[]; rest: number; total: number }
export interface SizeRow { neuron: number; activation: number; size: number; labels: number[] }

export const layerActivations = (all: Float32Array, layer: number, neurons: number) =>
  all.subarray(layer * neurons, (layer + 1) * neurons)

function columnDot(layer: NeuronLayer, dims: number, i: number, u: Float32Array): number {
  let s = 0
  for (let d = 0; d < dims; d++) s += layer.columns[i * dims + d] * u[d]
  return s * layer.scales[i]
}

function columnNorm(layer: NeuronLayer, dims: number, i: number): number {
  let s = 0
  for (let d = 0; d < dims; d++) s += layer.columns[i * dims + d] ** 2
  return Math.sqrt(s) * layer.scales[i]
}

// sign 0 = the list for a positive activation (the column itself), 1 = for a negative one
function labels(m: NeuronManifest, layer: NeuronLayer, i: number, sign: number): number[] {
  const k = m.labelsPerSign, start = (i * 2 + sign) * k
  return Array.from(layer.labelIds.subarray(start, start + k))
}

function badge(m: NeuronManifest, layer: NeuronLayer, i: number, sign: number, cos: number): Badge {
  const f = m.fractions.length, start = (i * 2 + sign) * f
  for (let j = 0; j < f; j++) if (cos >= layer.thresholds[start + j]) return BADGES[j] ?? 'not in top 10%'
  return 'not in top 10%'
}

const topBy = <T,>(items: T[], key: (t: T) => number, n: number) =>
  [...items].sort((a, b) => key(b) - key(a)).slice(0, n)

export function behindTheVote(m: NeuronManifest, layer: NeuronLayer, acts: Float32Array, row: Float32Array,
  g: Float32Array, invRms: number, n = 8): VoteTable {
  const { neurons, dims } = m
  const u = new Float32Array(dims)
  let uNorm = 0
  for (let d = 0; d < dims; d++) { u[d] = g[d] * row[d]; uNorm += u[d] ** 2 }
  uNorm = Math.sqrt(uNorm) || 1
  const dots = Array.from({ length: neurons }, (_, i) => columnDot(layer, dims, i, u))
  const votes = dots.map((dot, i) => acts[i] * dot * invRms)
  const total = votes.reduce((a, b) => a + b, 0)
  const rows = topBy([...votes.keys()], (i) => Math.abs(votes[i]), n).map((i): VoteRow => {
    const fired = acts[i] >= 0 ? 0 : 1
    const away = votes[i] < 0
    const sign = away ? 1 - fired : fired
    const norm = columnNorm(layer, dims, i) || 1
    const cos = ((sign === 0 ? 1 : -1) * dots[i]) / (norm * uNorm)
    return { neuron: i, activation: acts[i], vote: votes[i], labels: labels(m, layer, i, sign), away, badge: badge(m, layer, i, sign, cos) }
  })
  const shown = rows.reduce((a, r) => a + r.vote, 0)
  return { rows, rest: total - shown, total }
}

export function firingMost(m: NeuronManifest, layer: NeuronLayer, acts: Float32Array, n = 8): SizeRow[] {
  const sizes = Array.from({ length: m.neurons }, (_, i) => Math.abs(acts[i]) * columnNorm(layer, m.dims, i))
  return topBy([...sizes.keys()], (i) => sizes[i], n).map((i) => ({
    neuron: i, activation: acts[i], size: sizes[i], labels: labels(m, layer, i, acts[i] >= 0 ? 0 : 1),
  }))
}

export const voteTolerance = (v: number) => 0.05 + 0.01 * Math.abs(v)

// Within tolerance, the table adopts the votes card's number, so its lines add
// up to exactly what that card shows; outside it, the computed total stays.
export function reconcile(table: VoteTable, cardValue: number): VoteTable & { exact: boolean } {
  if (Math.abs(table.total - cardValue) > voteTolerance(cardValue)) return { ...table, exact: false }
  const shown = table.rows.reduce((a, r) => a + r.vote, 0)
  return { ...table, total: cardValue, rest: cardValue - shown, exact: true }
}
