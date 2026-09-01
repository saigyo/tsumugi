import { vi } from 'vitest'
import { parseGeometry, type GeometryAsset, type GeometryManifest } from '../geometry/asset'

export const FIXTURE_VOCAB = 256
export const FIXTURE_K = 12
export const FIXTURE_PCA = 4
const SCALE = 1 / 127

export function fixtureVector(id: number): number[] {
  const t = (2 * Math.PI * id) / FIXTURE_VOCAB
  return [Math.cos(t), Math.sin(t), Math.cos(2 * t), Math.sin(2 * t)]
}

export function fixtureText(id: number): string {
  if (id === 7) return '<0x07>'
  if (id === 8) return ''
  return `t${id}`
}

const cos = (a: number[], b: number[]) => {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return dot / Math.sqrt(na * nb)
}

// Same layout the Python builder writes: ids block (uint16 LE) then sims block (uint8).
export function encodeGeometryFixture(): {
  manifest: GeometryManifest; neighbors: ArrayBuffer; vectors: ArrayBuffer; tokens: string[]
} {
  const v = FIXTURE_VOCAB, k = FIXTURE_K
  const vecs = Array.from({ length: v }, (_, id) => fixtureVector(id))
  const neighbors = new ArrayBuffer(v * k * 3)
  const view = new DataView(neighbors)
  for (let id = 0; id < v; id++) {
    const ranked = vecs.map((w, j) => ({ j, sim: cos(vecs[id], w) }))
      .filter((c) => c.j !== id)
      .sort((a, b) => b.sim - a.sim || a.j - b.j)
      .slice(0, k)
    ranked.forEach((c, i) => {
      view.setUint16((id * k + i) * 2, c.j, true)
      view.setUint8(v * k * 2 + id * k + i, Math.round(Math.max(c.sim, 0) * 255))
    })
  }
  const vectors = new ArrayBuffer(v * FIXTURE_PCA)
  const q = new Int8Array(vectors)
  vecs.forEach((vec, id) => vec.forEach((x, d) => { q[id * FIXTURE_PCA + d] = Math.round(x / SCALE) }))
  const tokens = Array.from({ length: v }, (_, id) => fixtureText(id))
  const manifest: GeometryManifest = {
    modelId: 'fixture', vocabSize: v, dims: 4, k, pcaDims: FIXTURE_PCA, scale: SCALE,
    explainedVariance: 1, sourceSha256: '0'.repeat(64),
    files: { 'neighbors.bin': neighbors.byteLength, 'vectors.bin': vectors.byteLength, 'tokens.json': 0 },
  }
  return { manifest, neighbors, vectors, tokens }
}

export function makeGeometryAsset(): GeometryAsset {
  const f = encodeGeometryFixture()
  return parseGeometry(f.manifest, f.neighbors, f.vectors, f.tokens)
}

type FakeResponse = { ok: boolean; status: number; json?: () => Promise<unknown>; arrayBuffer?: () => Promise<ArrayBuffer> }

// A fetch stub serving the fixture files by URL suffix; returns the mock so tests can count calls.
// `overrides` maps a file name to a function that returns a response or throws.
export function stubGeometryFetch(overrides: Partial<Record<string, () => FakeResponse>> = {}) {
  const f = encodeGeometryFixture()
  const body = (name: string): FakeResponse => {
    const o = overrides[name]
    if (o) return o()
    if (name === 'manifest.json') return { ok: true, status: 200, json: async () => f.manifest }
    if (name === 'neighbors.bin') return { ok: true, status: 200, arrayBuffer: async () => f.neighbors }
    if (name === 'vectors.bin') return { ok: true, status: 200, arrayBuffer: async () => f.vectors }
    if (name === 'tokens.json') return { ok: true, status: 200, json: async () => f.tokens }
    return { ok: false, status: 404 }
  }
  const mock = vi.fn(async (url: string) => body(url.slice(url.lastIndexOf('/') + 1)))
  vi.stubGlobal('fetch', mock as unknown as typeof fetch)
  return mock
}
