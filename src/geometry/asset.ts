// The Hub geometry asset (spec Component 1): exact top-k neighbours, PCA-reduced
// int8 vectors and decoded texts for every vocabulary id. Static, read-only,
// deterministic per model — fetched once per page load, so the UI stays a pure
// function of (trace, cursor, asset).
import { GEOMETRY_BASE_URL, GEOMETRY_MODEL_IDS } from '../engine/tokenizer'

export interface GeometryManifest {
  modelId: string
  vocabSize: number
  dims: number
  k: number
  pcaDims: number
  scale: number
  explainedVariance: number
  sourceSha256: string
  files: Record<string, number>
}

export interface GeometryAsset {
  manifest: GeometryManifest
  neighbors(id: number): Array<{ id: number; sim: number }>
  vector(id: number): Float32Array
  text(id: number): string
}

export function parseGeometry(
  manifest: GeometryManifest, neighbors: ArrayBuffer, vectors: ArrayBuffer, tokens: string[],
): GeometryAsset {
  const { vocabSize: v, k, pcaDims: p, scale } = manifest
  if (neighbors.byteLength !== v * k * 3)
    throw new Error(`neighbors.bin: expected ${v * k * 3} bytes, got ${neighbors.byteLength}`)
  if (vectors.byteLength !== v * p)
    throw new Error(`vectors.bin: expected ${v * p} bytes, got ${vectors.byteLength}`)
  if (tokens.length !== v)
    throw new Error(`tokens.json: expected ${v} entries, got ${tokens.length}`)
  // ids are little-endian on disk; every JS engine we run on is little-endian too
  const ids = new Uint16Array(neighbors, 0, v * k)
  const sims = new Uint8Array(neighbors, v * k * 2, v * k)
  const q = new Int8Array(vectors)
  const check = (id: number) => {
    if (!Number.isInteger(id) || id < 0 || id >= v) throw new RangeError(`token id ${id} outside vocabulary of ${v}`)
  }
  return {
    manifest,
    neighbors(id) {
      check(id)
      const out: Array<{ id: number; sim: number }> = []
      for (let i = 0; i < k; i++) out.push({ id: ids[id * k + i], sim: sims[id * k + i] / 255 })
      return out
    },
    vector(id) {
      check(id)
      const out = new Float32Array(p)
      for (let i = 0; i < p; i++) out[i] = q[id * p + i] * scale
      return out
    },
    text(id) { check(id); return tokens[id] },
  }
}

export function coversModel(manifest: GeometryManifest, modelId: string): boolean {
  return manifest.modelId === modelId
    || (GEOMETRY_MODEL_IDS.includes(manifest.modelId) && GEOMETRY_MODEL_IDS.includes(modelId))
}

let pending: Promise<GeometryAsset> | null = null

export function loadGeometry(baseUrl: string = GEOMETRY_BASE_URL): Promise<GeometryAsset> {
  if (!pending) {
    pending = fetchGeometry(baseUrl).catch((err) => { pending = null; throw err })
  }
  return pending
}

export function resetGeometryCache(): void { pending = null }

async function fetchGeometry(base: string): Promise<GeometryAsset> {
  const get = async (name: string) => {
    const r = await fetch(`${base}/${name}`)
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`)
    return r
  }
  const manifest = (await (await get('manifest.json')).json()) as GeometryManifest
  const [neighbors, vectors, tokens] = await Promise.all([
    get('neighbors.bin').then((r) => r.arrayBuffer()),
    get('vectors.bin').then((r) => r.arrayBuffer()),
    get('tokens.json').then((r) => r.json() as Promise<string[]>),
  ])
  return parseGeometry(manifest, neighbors, vectors, tokens)
}
