// The Hub neuron asset: manifest and texts once, layer files on demand. Files
// are immutable per tag and the URL carries the tag, so the Cache API copy
// never needs revalidating (transformers.js caches only its own files).
import { NEURONS_BASE_URL } from '../engine/tokenizer'
import { layerFile, parseNeuronLayer, type NeuronLayer, type NeuronManifest } from './format'

export interface NeuronAsset { manifest: NeuronManifest; g: Float32Array; text(id: number): string }

export const NEURON_CACHE = 'tsumugi-neurons'

async function openCache(): Promise<Cache | null> {
  try { return typeof caches === 'undefined' ? null : await caches.open(NEURON_CACHE) } catch { return null }
}

async function cachedFetch(url: string): Promise<Response> {
  const cache = await openCache()
  const hit = await cache?.match(url).catch(() => undefined)
  if (hit) return hit
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url.slice(url.lastIndexOf('/') + 1)}: HTTP ${r.status}`)
  if (cache) await cache.put(url, r.clone()).catch(() => {})
  return r
}

async function evict(url: string) {
  await (await openCache())?.delete(url).catch(() => false)
}

let assetPending: Promise<NeuronAsset> | null = null
const layerPending = new Map<number, Promise<NeuronLayer>>()

export function loadNeuronAsset(base: string = NEURONS_BASE_URL): Promise<NeuronAsset> {
  if (!assetPending) assetPending = fetchAsset(base).catch((err) => { assetPending = null; throw err })
  return assetPending
}

export function loadNeuronLayer(l: number, base: string = NEURONS_BASE_URL): Promise<NeuronLayer> {
  let p = layerPending.get(l)
  if (!p) {
    const url = `${base}/${layerFile(l)}`
    p = loadNeuronAsset(base)
      .then(async (asset) => {
        const buf = await (await cachedFetch(url)).arrayBuffer()
        try { return parseNeuronLayer(asset.manifest, buf) } catch (err) { await evict(url); throw err }
      })
      .catch((err) => { layerPending.delete(l); throw err })
    layerPending.set(l, p)
  }
  return p
}

export function resetNeuronCache(): void { assetPending = null; layerPending.clear() }

async function fetchAsset(base: string): Promise<NeuronAsset> {
  const manifest = (await (await cachedFetch(`${base}/manifest.json`)).json()) as NeuronManifest
  const tokens = (await (await cachedFetch(`${base}/tokens.json`)).json()) as string[]
  if (tokens.length !== manifest.vocabSize)
    throw new Error(`tokens.json: expected ${manifest.vocabSize} entries, got ${tokens.length}`)
  if (manifest.finalNormWeight.length !== manifest.dims)
    throw new Error(`manifest: finalNormWeight has ${manifest.finalNormWeight.length} values, expected ${manifest.dims}`)
  return { manifest, g: Float32Array.from(manifest.finalNormWeight), text: (id) => tokens[id] ?? `#${id}` }
}
