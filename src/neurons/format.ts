// The neurons/ asset's file format (#16 spec, section 2). No app imports, so
// the Playwright tests can build fixtures with it.

export interface LayerLayout { scales: number; columns: number; labelIds: number; thresholds: number; normWeight: number; bytes: number }

export interface NeuronManifest {
  modelId: string
  layers: number
  neurons: number
  dims: number
  vocabSize: number
  labelsPerSign: number
  fractions: number[]          // the badge levels: 0.001, 0.01, 0.1 of the vocabulary
  labelMethod: string
  layout: LayerLayout
  finalNormWeight: number[]    // g, [dims]
  revision: string             // the model tag the asset was built for (v6 on)
  rmsNormEps: number           // the RMSNorm epsilon, for n = x / rms(x) ⊙ w
  files: Record<string, { bytes: number; sha256: string }>
}

export interface NeuronLayer {
  scales: Float32Array         // [neurons]
  columns: Int8Array           // [neurons × dims]: column i at i·dims, times scales[i]
  labelIds: Uint16Array        // [neurons × 2 × labelsPerSign]: sign 0 = positive activation, 1 = negative
  thresholds: Float32Array     // [neurons × 2 × fractions]: the cosine at each badge level's rank
  normWeight: Float32Array     // [dims]: this layer's post-attention RMSNorm weight w
}

// Mirrors tools/export/src/tsumugi_export/neurons.py layer_layout.
export function layerLayout(neurons: number, dims: number, labels: number, fractions: number): LayerLayout {
  const columns = neurons * 4
  const labelIds = columns + neurons * dims
  const thresholds = labelIds + neurons * 2 * labels * 2
  const normWeight = thresholds + neurons * 2 * fractions * 4
  return { scales: 0, columns, labelIds, thresholds, normWeight, bytes: normWeight + dims * 4 }
}

export const layerFile = (l: number) => `layer-${String(l).padStart(2, '0')}.bin`

// Copies (slice) rather than views for the multi-byte blocks, so alignment
// never depends on the layout.
export function parseNeuronLayer(m: NeuronManifest, buf: ArrayBuffer): NeuronLayer {
  const { neurons: n, dims, labelsPerSign: k, fractions, layout } = m
  if (buf.byteLength !== layout.bytes) throw new Error(`layer file: expected ${layout.bytes} bytes, got ${buf.byteLength}`)
  return {
    scales: new Float32Array(buf.slice(layout.scales, layout.scales + n * 4)),
    columns: new Int8Array(buf, layout.columns, n * dims),
    labelIds: new Uint16Array(buf.slice(layout.labelIds, layout.labelIds + n * 2 * k * 2)),
    thresholds: new Float32Array(buf.slice(layout.thresholds, layout.thresholds + n * 2 * fractions.length * 4)),
    normWeight: new Float32Array(buf.slice(layout.normWeight, layout.normWeight + dims * 4)),
  }
}
