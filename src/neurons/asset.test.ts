import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { encodeNeuronFixture, NF, neuronFixtureFile } from '../test/neuronFixture'
import { layerLayout, parseNeuronLayer } from './format'
import { loadNeuronAsset, loadNeuronLayer, NEURON_CACHE, resetNeuronCache } from './asset'

const BASE = 'https://hub.test/neurons'

// A Cache API stand-in: a Map of URL → Response, shared by every open().
function stubCaches() {
  const store = new Map<string, Response>()
  const cache = {
    match: vi.fn(async (url: string) => store.get(url)?.clone()),
    put: vi.fn(async (url: string, r: Response) => { store.set(url, r) }),
    delete: vi.fn(async (url: string) => store.delete(url)),
  }
  const open = vi.fn(async (_name: string) => cache)
  vi.stubGlobal('caches', { open })
  return { store, cache, open }
}

function stubFetch(override: (name: string) => Response | null = () => null) {
  const mock = vi.fn(async (url: string) => {
    const name = url.slice(url.lastIndexOf('/') + 1)
    const o = override(name)
    if (o) return o
    const body = neuronFixtureFile(name)
    return body ? new Response(body) : new Response('missing', { status: 404 })
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

beforeEach(() => resetNeuronCache())
afterEach(() => vi.unstubAllGlobals())

test('layerLayout matches the Python builder for the real sizes', () => {
  expect(layerLayout(1536, 576, 5, 3)).toEqual({ scales: 0, columns: 6144, labelIds: 890880, thresholds: 921600, bytes: 958464 })
})

test('parseNeuronLayer reads every block and refuses a wrong size', () => {
  const f = encodeNeuronFixture()
  const layer = parseNeuronLayer(f.manifest, f.layers[1])
  expect(layer.scales[2]).toBeCloseTo(0.006, 6)
  expect(layer.columns).toHaveLength(NF.neurons * NF.dims)
  expect(layer.labelIds).toHaveLength(NF.neurons * 2 * 5)
  expect(layer.thresholds[0]).toBeCloseTo(0.5, 6)
  expect(() => parseNeuronLayer(f.manifest, new ArrayBuffer(10))).toThrow(/expected \d+ bytes, got 10/)
})

test('a miss fetches and stores in the cache; the next load is served from it', async () => {
  const { store, open } = stubCaches()
  const fetchMock = stubFetch()
  const asset = await loadNeuronAsset(BASE)
  await loadNeuronLayer(0, BASE)
  expect(open).toHaveBeenCalledWith(NEURON_CACHE)
  expect(asset.text(32)).toBe(' and')
  expect(asset.g[0]).toBeCloseTo(1.5, 6)
  expect([...store.keys()].sort()).toEqual([`${BASE}/layer-00.bin`, `${BASE}/manifest.json`, `${BASE}/tokens.json`])
  resetNeuronCache()
  fetchMock.mockClear()
  await loadNeuronLayer(0, BASE)
  expect(fetchMock).not.toHaveBeenCalled()
})

test('two requests for one layer fetch it once', async () => {
  stubCaches()
  const fetchMock = stubFetch()
  await Promise.all([loadNeuronLayer(2, BASE), loadNeuronLayer(2, BASE)])
  expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('layer-02.bin'))).toHaveLength(1)
})

test('without the Cache API it falls back to plain fetch', async () => {
  vi.stubGlobal('caches', undefined)
  stubFetch()
  const layer = await loadNeuronLayer(1, BASE)
  expect(layer.scales).toHaveLength(NF.neurons)
})

test('a failed fetch rejects, and the next load tries again', async () => {
  stubCaches()
  let fail = true
  stubFetch((name) => (fail && name === 'layer-01.bin' ? new Response('', { status: 503 }) : null))
  await expect(loadNeuronLayer(1, BASE)).rejects.toThrow('layer-01.bin: HTTP 503')
  fail = false
  await expect(loadNeuronLayer(1, BASE)).resolves.toBeDefined()
})

test('a truncated cached file is evicted, so a retry downloads it again', async () => {
  const { store, cache } = stubCaches()
  stubFetch()
  store.set(`${BASE}/layer-00.bin`, new Response(new Uint8Array(12)))
  await expect(loadNeuronLayer(0, BASE)).rejects.toThrow(/expected \d+ bytes, got 12/)
  expect(cache.delete).toHaveBeenCalledWith(`${BASE}/layer-00.bin`)
  await expect(loadNeuronLayer(0, BASE)).resolves.toBeDefined()
})
