import { unpackHalf } from '../trace/neuronPack'
import type { MlpSteps } from '../trace/types'
import type { NeuronLayer } from './format'

// The numbers behind "The MLP, step by step" (#17), in the glossary's notation:
//   n = x / rms(x) ⊙ w,  g = W_gate · n,  u = W_up · n,  h = SiLU(g) ⊙ u,  edit = W_down · h
// g, u, x and the edit come from the model (v6 outputs); h is mlp_activations.

export interface DecodedSteps { gate: Float32Array; up: Float32Array; mlpIn: Float32Array; mlpOut: Float32Array }
export interface LayerSteps { x: Float32Array; edit: Float32Array; gate: Float32Array; up: Float32Array; h: Float32Array }

export const silu = (z: number) => z / (1 + Math.exp(-z))

const decoded = new WeakMap<MlpSteps, DecodedSteps>()
export function decodeSteps(s: MlpSteps): DecodedSteps {
  let d = decoded.get(s)
  if (!d) {
    d = { gate: unpackHalf(s.gate), up: unpackHalf(s.up), mlpIn: unpackHalf(s.mlpIn), mlpOut: unpackHalf(s.mlpOut) }
    decoded.set(s, d)
  }
  return d
}

const part = (a: Float32Array, l: number, w: number) => a.subarray(l * w, (l + 1) * w)

export function layerSteps(d: DecodedSteps, h: Float32Array, layer: number, neurons: number, dims: number): LayerSteps | null {
  const layers = h.length / neurons
  if (!Number.isInteger(layers) || layer >= layers || d.gate.length !== h.length || d.up.length !== h.length
    || d.mlpIn.length !== layers * dims || d.mlpOut.length !== layers * dims) return null
  return { x: part(d.mlpIn, layer, dims), edit: part(d.mlpOut, layer, dims), gate: part(d.gate, layer, neurons),
    up: part(d.up, layer, neurons), h: part(h, layer, neurons) }
}

export function rms(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s / (v.length || 1))
}

export function normalize(x: Float32Array, w: Float32Array, eps: number): { n: Float32Array; rms: number } {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] * x[i]
  const r = Math.sqrt(s / x.length + eps)
  return { n: Float32Array.from(x, (v, i) => (v / r) * w[i]), rms: r }
}

// Both sides of zero always show (early layers are almost all negative), plus 4 % padding.
export function gateDomain(gate: Float32Array): [number, number] {
  let lo = -1, hi = 1
  for (const v of gate) { if (v < lo) lo = v; if (v > hi) hi = v }
  const pad = (hi - lo) * 0.04
  return [lo - pad, hi + pad]
}

export function gateCounts(gate: Float32Array): { negative: number; positive: number } {
  let negative = 0
  for (const v of gate) if (v < 0) negative++
  return { negative, positive: gate.length - negative }
}

// Neurons by |h|, largest first: the first `shown`, how many carry half and 90 %
// of Σh², and the largest |h| left out (null when nothing is left out).
export function productSummary(h: Float32Array, shown = 120) {
  const all = Array.from(h.keys()).sort((a, b) => Math.abs(h[b]) - Math.abs(h[a]))
  const total = all.reduce((s, i) => s + h[i] * h[i], 0) || 1
  let cum = 0, half = 0, ninety = 0
  for (let k = 0; k < all.length; k++) {
    cum += h[all[k]] ** 2
    if (!half && cum / total >= 0.5) half = k + 1
    if (!ninety && cum / total >= 0.9) { ninety = k + 1; break }
  }
  return { order: all.slice(0, shown), half, ninety, restMax: all.length > shown ? Math.abs(h[all[shown]]) : null }
}

export function contribution(layer: NeuronLayer, dims: number, i: number, hi: number): Float32Array {
  const s = layer.scales[i] * hi
  return Float32Array.from({ length: dims }, (_, d) => layer.columns[i * dims + d] * s)
}

const norm = (v: Float32Array) => Math.sqrt(v.reduce((s, x) => s + x * x, 0))
export const editRatio = (edit: Float32Array, x: Float32Array) => norm(edit) / (norm(x) || 1)
