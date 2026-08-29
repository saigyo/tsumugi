import { beforeEach, expect, test, vi } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { archiveImport, archiveRemove, archiveSeal, archiveTogglePin, initArchive, _resetArchiveForTests } from './runArchive'
import { createMemoryStorage } from './runStorage'
import { UNPINNED_CAP, useRunsStore, type SealMeta } from './runsStore'

const meta = (over: Partial<SealMeta> = {}): SealMeta => ({
  prompt: 'The cat', params: { temperature: 0.8, topK: 10, maxNewTokens: 2 },
  mode: 'sim', endedAt: 1, reason: 'max-tokens', ...over,
})

beforeEach(() => {
  useRunsStore.setState({ records: [], activeId: null, nextSeq: 1, persistFailed: false })
  _resetArchiveForTests()
})

test('seal writes through to storage; eviction deletes the evicted record', async () => {
  const storage = createMemoryStorage()
  await initArchive(storage)
  const first = archiveSeal(meta({ endedAt: 0 }), makeFixtureTrace())
  for (let i = 0; i < UNPINNED_CAP; i++) archiveSeal(meta({ endedAt: 10 + i }), makeFixtureTrace())
  await vi.waitFor(() => expect(storage.map.size).toBe(UNPINNED_CAP))
  expect(storage.map.has(first.id)).toBe(false)  // evicted → deleted from storage
})

test('pin toggle and remove mirror to storage', async () => {
  const storage = createMemoryStorage()
  await initArchive(storage)
  const record = archiveSeal(meta(), makeFixtureTrace())
  archiveTogglePin(record.id)
  await vi.waitFor(() => expect(storage.map.get(record.id)?.meta.pinned).toBe(true))
  archiveRemove(record.id)
  await vi.waitFor(() => expect(storage.map.size).toBe(0))
})

test('import mirrors to storage and arrives pinned', async () => {
  const storage = createMemoryStorage()
  await initArchive(storage)
  const record = archiveImport({ meta: meta(), events: makeFixtureTrace() })
  await vi.waitFor(() => expect(storage.map.get(record.id)?.meta.pinned).toBe(true))
})

test('initArchive hydrates the store from storage', async () => {
  const storage = createMemoryStorage()
  storage.map.set('x', { id: 'x', meta: { ...meta({ endedAt: 20 }), seq: 4, pinned: false }, events: makeFixtureTrace() })
  storage.map.set('y', { id: 'y', meta: { ...meta({ endedAt: 10 }), seq: 2, pinned: true }, events: makeFixtureTrace() })
  await initArchive(storage)
  const state = useRunsStore.getState()
  expect(state.records.map((r) => r.id)).toEqual(['y', 'x'])
  expect(state.nextSeq).toBe(5)
})

test('a run sealed during a pending hydrate is written back with its re-sequenced seq', async () => {
  const storage = createMemoryStorage()
  const old = { id: 'old', meta: { ...meta({ endedAt: 5 }), seq: 7, pinned: false }, events: makeFixtureTrace() }
  storage.map.set('old', old)
  // gate loadAll so a seal lands while hydration is pending; the snapshot
  // deliberately excludes the concurrently-sealed record (as a real
  // readonly transaction opened before the put would)
  let release!: () => void
  const gate = new Promise<void>((r) => { release = r })
  const delayed = { ...storage, loadAll: () => gate.then(() => [old]) }
  const init = initArchive(delayed)
  const sealed = archiveSeal(meta({ endedAt: 99 }), makeFixtureTrace())   // seq 1, mirrored as seq 1
  expect(storage.map.get(sealed.id)?.meta.seq).toBe(1)
  release()
  await init
  const kept = useRunsStore.getState().records.find((r) => r.id === sealed.id)
  expect(kept?.meta.seq).toBe(8)                                          // re-sequenced past loaded max
  await vi.waitFor(() => expect(storage.map.get(sealed.id)?.meta.seq).toBe(8))  // written back
})

test('a failing adapter degrades to session-only without throwing', async () => {
  await initArchive(createMemoryStorage({ failing: true }))
  expect(useRunsStore.getState().persistFailed).toBe(true)
  // store operations still work session-only
  const record = archiveSeal(meta(), makeFixtureTrace())
  expect(useRunsStore.getState().records[0].id).toBe(record.id)
})
