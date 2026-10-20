import { packFloat32, packHalf, unpackFloat32, unpackHalf } from '../trace/neuronPack'
import type { MlpSteps, NeuronData } from '../trace/types'
import { behindTheVote, layerActivations } from '../neurons/math'
import { layerFile, layerLayout, parseNeuronLayer, type NeuronLayer, type NeuronManifest } from '../neurons/format'

// A tiny neuron asset in the real layout: 3 layers × 10 neurons over the real
// 576 dims (the trace fixture's rows are 576 wide), a 256-token vocabulary.
export const NF = { layers: 3, neurons: 10, dims: 576, vocab: 256, labels: 5, fractions: [0.001, 0.01, 0.1] }

export const fixtureNormWeight = (l: number, d: number) => 1 + 0.25 * Math.sin(d * 0.2 + l)
export const fixtureG = (d: number) => 1 + 0.5 * Math.cos(d * 0.3)
const qValue = (l: number, i: number, d: number) => Math.round(100 * Math.sin(0.05 * d + l + 1.3 * i))
const labelId = (l: number, i: number, s: number, j: number) => (l * 40 + i * 10 + s * 5 + j) % NF.vocab
export const FIXTURE_THRESHOLDS = [0.5, 0.3, 0.1]

export function encodeNeuronFixture(): { manifest: NeuronManifest; layers: ArrayBuffer[]; tokens: string[] } {
  const { layers: L, neurons: n, dims, labels: k, fractions } = NF
  const layout = layerLayout(n, dims, k, fractions.length)
  const layers = Array.from({ length: L }, (_, l) => {
    const buf = new ArrayBuffer(layout.bytes)
    const view = new DataView(buf)
    for (let d = 0; d < dims; d++) view.setFloat32(layout.normWeight + d * 4, fixtureNormWeight(l, d), true)
    for (let i = 0; i < n; i++) {
      view.setFloat32(layout.scales + i * 4, 0.002 * (i + 1), true)
      for (let d = 0; d < dims; d++) view.setInt8(layout.columns + i * dims + d, qValue(l, i, d))
      for (let s = 0; s < 2; s++) {
        for (let j = 0; j < k; j++) view.setUint16(layout.labelIds + ((i * 2 + s) * k + j) * 2, labelId(l, i, s, j), true)
        FIXTURE_THRESHOLDS.forEach((t, j) => view.setFloat32(layout.thresholds + ((i * 2 + s) * fractions.length + j) * 4, t, true))
      }
    }
    return buf
  })
  const tokens = Array.from({ length: NF.vocab }, (_, id) => (id === 32 ? ' and' : `w${id}`))
  const files: NeuronManifest['files'] = Object.fromEntries(layers.map((b, l) => [layerFile(l), { bytes: b.byteLength, sha256: '0'.repeat(64) }]))
  files['tokens.json'] = { bytes: 0, sha256: '0'.repeat(64) }
  const manifest: NeuronManifest = {
    modelId: 'fixture', layers: L, neurons: n, dims, vocabSize: NF.vocab, labelsPerSign: k, fractions,
    labelMethod: 'fixture', layout, revision: 'v6', rmsNormEps: 1e-5, finalNormWeight: Array.from({ length: dims }, (_, d) => fixtureG(d)), files,
  }
  return { manifest, layers, tokens }
}

// The fixture's files by name, as served bytes (Playwright routes and fetch stubs).
export function neuronFixtureFile(name: string): Uint8Array<ArrayBuffer> | null {
  const f = encodeNeuronFixture()
  if (name === 'manifest.json') return new TextEncoder().encode(JSON.stringify(f.manifest))
  if (name === 'tokens.json') return new TextEncoder().encode(JSON.stringify(f.tokens))
  const l = f.layers.findIndex((_, i) => layerFile(i) === name)
  return l >= 0 ? new Uint8Array(f.layers[l]) : null
}

export function fixtureNeuronLayer(l: number): NeuronLayer {
  const f = encodeNeuronFixture()
  return parseNeuronLayer(f.manifest, f.layers[l])
}

export function fixtureNeuronAsset() {
  const f = encodeNeuronFixture()
  return { manifest: f.manifest, g: Float32Array.from(f.manifest.finalNormWeight), text: (id: number) => f.tokens[id] ?? `#${id}` }
}

export const FIXTURE_INV_RMS = 0.05

const silu = (z: number) => z / (1 + Math.exp(-z))
const fixtureGate = (cycle: number, k: number) => 4 * Math.cos(cycle + 0.7 * Math.floor(k / NF.neurons) + 1.1 * (k % NF.neurons)) + 0.5
const fixtureUp = (cycle: number, k: number) => 1.5 * Math.sin(cycle + 0.4 * k + 0.3)

export function fixtureSteps(cycle: number): MlpSteps {
  const n = NF.layers * NF.neurons, w = NF.layers * NF.dims
  return {
    gate: packHalf(Array.from({ length: n }, (_, k) => fixtureGate(cycle, k))),
    up: packHalf(Array.from({ length: n }, (_, k) => fixtureUp(cycle, k))),
    mlpIn: packHalf(Array.from({ length: w }, (_, k) => 20 * Math.cos(0.13 * k + cycle))),
    mlpOut: packHalf(Array.from({ length: w }, (_, k) => 6 * Math.sin(0.07 * k - cycle))),
  }
}

// Deterministic neuron data for a cycle, with each candidate's row from
// `rowOf` (the trace fixture passes fixtureEmbedding). Activations are
// SiLU(gate) · up of the step data.
export function fixtureNeuronData(cycle: number, ids: number[], rowOf: (id: number) => number[], withSteps = true): NeuronData {
  const acts = Array.from({ length: NF.layers * NF.neurons }, (_, k) => silu(fixtureGate(cycle, k)) * fixtureUp(cycle, k))
  return { neurons: NF.neurons, invRms: FIXTURE_INV_RMS, activations: packHalf(acts),
    candidates: ids.map((id) => ({ id, row: packFloat32(rowOf(id)) })), ...(withSteps ? { steps: fixtureSteps(cycle) } : {}) }
}

// The MLP votes per layer that the neuron data adds up to for candidate `id`
// (computed from the packed values, exactly as the card does).
export function fixtureMlpVotes(data: NeuronData, id: number): number[] {
  const asset = fixtureNeuronAsset()
  const acts = unpackHalf(data.activations)
  const row = unpackFloat32(data.candidates.find((c) => c.id === id)!.row)
  return Array.from({ length: NF.layers }, (_, l) => behindTheVote(asset.manifest, fixtureNeuronLayer(l),
    layerActivations(acts, l, NF.neurons), row, asset.g, data.invRms, NF.neurons).total)
}
