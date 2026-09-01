import { afterEach, beforeEach, expect, test } from 'vitest'
import { encodeGeometryFixture, makeGeometryAsset, stubGeometryFetch } from '../test/geometryFixture'
import { coversModel, loadGeometry, parseGeometry, resetGeometryCache } from './asset'

beforeEach(() => resetGeometryCache())
afterEach(() => resetGeometryCache())

test('neighbors of a token are its angular neighbours, sorted by similarity', () => {
  const a = makeGeometryAsset()
  const n = a.neighbors(10)
  expect(n).toHaveLength(12)
  expect(new Set([n[0].id, n[1].id])).toEqual(new Set([9, 11]))
  expect(n[0].sim).toBeGreaterThan(n[2].sim)
  expect(n.every((x) => x.id !== 10)).toBe(true)
})

test('vectors dequantise to the encoded values', () => {
  const a = makeGeometryAsset()
  const v = a.vector(0)
  expect(v).toHaveLength(4)
  expect(v[0]).toBeCloseTo(1, 2)
  expect(v[1]).toBeCloseTo(0, 2)
})

test('text() returns the decoded token', () => {
  const a = makeGeometryAsset()
  expect(a.text(3)).toBe('t3')
  expect(a.text(7)).toBe('<0x07>')
})

test('ids outside the vocabulary throw', () => {
  const a = makeGeometryAsset()
  expect(() => a.neighbors(256)).toThrow(RangeError)
  expect(() => a.vector(-1)).toThrow(RangeError)
})

test('byte-length mismatches are rejected', () => {
  const f = encodeGeometryFixture()
  expect(() => parseGeometry(f.manifest, f.neighbors.slice(0, 10), f.vectors, f.tokens)).toThrow(/neighbors\.bin/)
  expect(() => parseGeometry(f.manifest, f.neighbors, f.vectors.slice(0, 10), f.tokens)).toThrow(/vectors\.bin/)
  expect(() => parseGeometry(f.manifest, f.neighbors, f.vectors, f.tokens.slice(1))).toThrow(/tokens\.json/)
})

test('coversModel accepts the stock id and the attention re-export interchangeably', () => {
  const m = { ...encodeGeometryFixture().manifest, modelId: 'HuggingFaceTB/SmolLM2-135M-Instruct' }
  expect(coversModel(m, 'HuggingFaceTB/SmolLM2-135M-Instruct')).toBe(true)
  expect(coversModel(m, 'saigyo-hoshi/smollm2-135m-attn-onnx')).toBe(true)
  expect(coversModel(m, 'someone/other-model')).toBe(false)
  expect(coversModel({ ...m, modelId: 'fixture' }, 'fixture')).toBe(true)
})

test('loadGeometry fetches the four files once and shares the promise', async () => {
  const fetchMock = stubGeometryFetch()
  const [a, b] = await Promise.all([loadGeometry('http://x/geo'), loadGeometry('http://x/geo')])
  expect(a).toBe(b)
  expect(fetchMock).toHaveBeenCalledTimes(4)
  expect(a.text(3)).toBe('t3')
})

test('a failed load is not cached: the next call re-fetches', async () => {
  stubGeometryFetch({ 'manifest.json': () => { throw new Error('offline') } })
  await expect(loadGeometry('http://x/geo')).rejects.toThrow('offline')
  const ok = stubGeometryFetch()
  await expect(loadGeometry('http://x/geo')).resolves.toBeTruthy()
  expect(ok).toHaveBeenCalledTimes(4)
})

test('a non-OK response rejects with the file name and status', async () => {
  stubGeometryFetch({ 'vectors.bin': () => ({ ok: false, status: 500 }) })
  await expect(loadGeometry('http://x/geo')).rejects.toThrow('vectors.bin: HTTP 500')
})
