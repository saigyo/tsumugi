import { beforeEach, expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { UNPINNED_CAP, useRunsStore, type SealMeta } from './runsStore'

const meta = (over: Partial<SealMeta> = {}): SealMeta => ({
  prompt: 'The cat', params: { temperature: 0.8, topK: 10, maxNewTokens: 2 },
  mode: 'sim', endedAt: 1, reason: 'max-tokens', ...over,
})

beforeEach(() => {
  useRunsStore.setState({ records: [], activeId: null, nextSeq: 1, persistFailed: false })
})

test('seal appends, activates, and assigns a monotonic seq', () => {
  const s = useRunsStore.getState()
  const first = s.seal(meta(), makeFixtureTrace())
  const second = useRunsStore.getState().seal(meta({ endedAt: 2 }), makeFixtureTrace())
  const state = useRunsStore.getState()
  expect(state.records.map((r) => r.meta.seq)).toEqual([1, 2])
  expect(state.activeId).toBe(second.record.id)
  expect(first.record.meta.pinned).toBe(false)
  expect(first.evicted).toEqual([])
})

test('the ring evicts the oldest unpinned record beyond the cap', () => {
  for (let i = 0; i < UNPINNED_CAP; i++)
    useRunsStore.getState().seal(meta({ endedAt: i }), makeFixtureTrace())
  const { evicted } = useRunsStore.getState().seal(meta({ endedAt: 99 }), makeFixtureTrace())
  const state = useRunsStore.getState()
  expect(evicted.map((r) => r.meta.seq)).toEqual([1])
  expect(state.records).toHaveLength(UNPINNED_CAP)
  expect(state.records.map((r) => r.meta.seq)).toEqual([2, 3, 4, 5, 6, 7, 8, 9])
})

test('pinned records are exempt from eviction and do not count toward the cap', () => {
  const first = useRunsStore.getState().seal(meta({ endedAt: 0 }), makeFixtureTrace())
  useRunsStore.getState().togglePin(first.record.id)
  for (let i = 0; i < UNPINNED_CAP; i++)
    useRunsStore.getState().seal(meta({ endedAt: 10 + i }), makeFixtureTrace())
  // 1 pinned + 8 unpinned: no eviction yet
  expect(useRunsStore.getState().records).toHaveLength(9)
  const { evicted } = useRunsStore.getState().seal(meta({ endedAt: 99 }), makeFixtureTrace())
  // oldest UNPINNED (seq 2) goes; the pinned seq 1 survives
  expect(evicted.map((r) => r.meta.seq)).toEqual([2])
  expect(useRunsStore.getState().records[0].meta.seq).toBe(1)
})

test('togglePin flips, remove deletes and clears active, setActive ignores unknown ids', () => {
  const { record } = useRunsStore.getState().seal(meta(), makeFixtureTrace())
  useRunsStore.getState().togglePin(record.id)
  expect(useRunsStore.getState().records[0].meta.pinned).toBe(true)
  useRunsStore.getState().setActive('nope')
  expect(useRunsStore.getState().activeId).toBe(record.id)
  useRunsStore.getState().remove(record.id)
  expect(useRunsStore.getState().records).toHaveLength(0)
  expect(useRunsStore.getState().activeId).toBeNull()
})

test('importRecord arrives pinned with the next seq and leaves active unchanged', () => {
  const { record } = useRunsStore.getState().seal(meta(), makeFixtureTrace())
  const imported = useRunsStore.getState().importRecord({ meta: meta({ endedAt: 5 }), events: makeFixtureTrace() })
  expect(imported.meta.pinned).toBe(true)
  expect(imported.meta.seq).toBe(2)
  expect(useRunsStore.getState().activeId).toBe(record.id)
})

test('hydrate sorts by endedAt and resumes seq numbering after the max', () => {
  useRunsStore.getState().hydrate([
    { id: 'b', meta: { ...meta({ endedAt: 20 }), seq: 7, pinned: false }, events: makeFixtureTrace() },
    { id: 'a', meta: { ...meta({ endedAt: 10 }), seq: 3, pinned: true }, events: makeFixtureTrace() },
  ])
  const state = useRunsStore.getState()
  expect(state.records.map((r) => r.id)).toEqual(['a', 'b'])
  expect(state.nextSeq).toBe(8)
  expect(state.activeId).toBeNull()
})

test('hydrate merges records sealed while loading instead of dropping them', () => {
  const { record: sealed } = useRunsStore.getState().seal(meta({ endedAt: 100 }), makeFixtureTrace())
  useRunsStore.getState().hydrate([
    { id: 'a', meta: { ...meta({ endedAt: 10 }), seq: 3, pinned: false }, events: makeFixtureTrace() },
    { id: 'b', meta: { ...meta({ endedAt: 20 }), seq: 7, pinned: false }, events: makeFixtureTrace() },
  ])
  const state = useRunsStore.getState()
  expect(state.records.map((r) => r.id)).toEqual(['a', 'b', sealed.id])
  expect(state.records[2].meta.seq).toBe(8)
  expect(state.nextSeq).toBe(9)
  expect(state.activeId).toBe(sealed.id)
})

test('hydrate with empty storage keeps sealed records', () => {
  const { record: sealed } = useRunsStore.getState().seal(meta(), makeFixtureTrace())
  useRunsStore.getState().hydrate([])
  const state = useRunsStore.getState()
  expect(state.records).toHaveLength(1)
  expect(state.records[0].id).toBe(sealed.id)
  expect(state.records[0].meta.seq).toBe(sealed.meta.seq)
  expect(state.activeId).toBe(sealed.id)
})
