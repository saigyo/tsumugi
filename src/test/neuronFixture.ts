import { layerFile, layerLayout, parseNeuronLayer, type NeuronLayer, type NeuronManifest } from '../neurons/format'

// A tiny neuron asset in the real layout: 3 layers × 10 neurons over the real
// 576 dims (the trace fixture's rows are 576 wide), a 256-token vocabulary.
export const NF = { layers: 3, neurons: 10, dims: 576, vocab: 256, labels: 5, fractions: [0.001, 0.01, 0.1] }

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
    labelMethod: 'fixture', layout, finalNormWeight: Array.from({ length: dims }, (_, d) => fixtureG(d)), files,
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
