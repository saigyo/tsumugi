# Trace Archive with Run Comparison (M3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep completed runs in a persistent, exportable archive with a run shelf, and let the user compare any two runs cycle-by-cycle (aligned streams with fork detection, paired distributions, paired attention).

**Architecture:** Completed traces are sealed into `RunRecord`s inside a new pure Zustand `runsStore` (ring of 8 unpinned, pins exempt), mirrored write-through to IndexedDB behind a storage-adapter interface, and exported/imported as JSON files gated by `validateTrace`. The live trace store stays the single buffer the player replays; activating an archived run loads its events into it. A dedicated `CompareView` (pure props, cycle-indexed inspection — no playback) replaces everything below the shelf while two runs are selected, built on new pure `compareSelectors`.

**Tech Stack:** React 19 + TypeScript (strict) + Vite, Zustand, Vitest + @testing-library/react, Playwright (sim-only e2e), IndexedDB.

**Spec:** `docs/superpowers/specs/2026-08-28-trace-archive-compare-design.md` (conflicts in this plan resolve against the spec).

## Global Constraints

- TypeScript strict must stay green: verify with `npx tsc --noEmit -p tsconfig.app.json` (the root `tsconfig.json` is solution-style and checks nothing).
- Unit tests: `npx vitest run` must pass after every task.
- E2E stays sim-only: `npx playwright test` must stay green. The ONE new smoke this plan adds lands in Task 8; no other e2e edits.
- Persistence must never block or delay generation: every storage failure is caught and degrades to a session-only archive with one shelf note ("archive not persisted in this browser").
- Archived-run comparison is trace-only: no worker or `head-request` use anywhere in compare code — the side channel stays scoped to the live run (M2 spec).
- Viz components stay store-free: data and actions arrive via props/callbacks.
- The default fixture `makeFixtureTrace`'s event indices and contents must not change — index-based tests depend on them (tokenize=1, embed=2, layers=3–5, attention=6, …). Task 1 refactors it into a parameterized builder whose defaults reproduce it byte-for-byte; the existing test suite is the guard.
- The single-run replay path is unchanged: selectors, DetailPanel, player reducer, and viz components are untouched except where a task names them (trace store gains `load`; `Thumb` gains `export`).
- Linear history; small conventional commits. **Every commit ends with both trailers** (shown verbatim in each commit step):
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL`

## Conventions (environment)

- Run git from the repository root; never from a subdirectory (the rtk hook mangles pathed git).
- Write file content with the Edit/Write tools — never heredocs.
- `rtk` compresses vitest/playwright output; use `rtk proxy npx vitest run …` when you need raw output or warnings. Test output must be pristine.
- jsdom has no canvas: any test file that renders `Thumb` (directly or via `CompareView`) needs
  `vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never)` at module scope
  (pattern: `src/viz/AttentionGridExplorer.test.tsx:6-7`).
- Commit message form (single `-m`, literal newlines — proven in this repo):

```bash
git commit -m "type: subject

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/app/runsStore.ts` | create | RunMeta/RunRecord types, pure archive store: seal/ring/pin/remove/import/hydrate |
| `src/test/fixtures.ts` | modify | parameterized `buildFixtureTrace` + `makeRunRecord`; `makeFixtureTrace` output frozen |
| `src/app/runStorage.ts` | create | `RunStorage` interface, memory impl (tests), IndexedDB impl |
| `src/app/runArchive.ts` | create | store↔storage orchestration: init/hydrate + write-through, never-fail |
| `src/app/runFiles.ts` | create | export serialization + import parsing/validation (formatVersion 1) |
| `src/app/RunShelf.tsx` | create | run chips, pin/export/remove/import, compare arm/select/exit |
| `src/viz/compare/compareSelectors.ts` | create | `alignRuns`, `pairedDistributions`, `pairedHeads` (pure) |
| `src/viz/compare/CompareView.tsx` | create | metadata header, aligned streams + ruler, paired panels |
| `src/viz/AttentionGridExplorer.tsx` | modify | `Thumb` becomes a named export (no behavior change) |
| `src/trace/store.ts` | modify | + `load(events)` action |
| `src/App.tsx` | modify | seal hook, shelf wiring, activate/compare handlers, conditional CompareView |
| `src/index.css` | modify | shelf + compare styles (aizome tokens) |
| `e2e/archive.spec.ts` | create | archive/compare smoke (sim) |

---

### Task 1: `runsStore` and run-record fixtures

**Files:**
- Create: `src/app/runsStore.ts`
- Modify: `src/test/fixtures.ts` (refactor `makeFixtureTrace` body into `buildFixtureTrace`; append `makeRunRecord`)
- Test: `src/app/runsStore.test.ts`

**Interfaces:**
- Consumes: `GenParams`, `Mode`, `RunEndReason`, `TokenInfo`, `TraceEvent` from `src/trace/types` (unchanged).
- Produces (later tasks rely on these exact names):
  - `export interface RunMeta { seq: number; prompt: string; params: GenParams; mode: Mode; modelId?: string; endedAt: number; reason: RunEndReason; pinned: boolean }`
  - `export interface RunRecord { id: string; meta: RunMeta; events: TraceEvent[] }`
  - `export type SealMeta = Omit<RunMeta, 'seq' | 'pinned'>`
  - `export const UNPINNED_CAP = 8`
  - `useRunsStore` with state `{ records: RunRecord[]; activeId: string | null; nextSeq: number; persistFailed: boolean }` and actions `seal(meta: SealMeta, events: TraceEvent[]): { record: RunRecord; evicted: RunRecord[] }`, `setActive(id: string): void`, `togglePin(id: string): void`, `remove(id: string): void`, `importRecord(data: { meta: SealMeta; events: TraceEvent[] }): RunRecord`, `hydrate(records: RunRecord[]): void`, `setPersistFailed(): void`
  - `export function buildFixtureTrace(opts?: FixtureTraceOpts): TraceEvent[]` and `export function makeRunRecord(seq: number, opts?: FixtureTraceOpts & { pinned?: boolean; endedAt?: number; id?: string }): RunRecord` in `src/test/fixtures.ts`

- [ ] **Step 1: Refactor the fixture into a parameterized builder (output frozen)**

Replace the body of `src/test/fixtures.ts` lines 1–38 (imports + `makeFixtureTrace`) with the following; `makeGridEvent` below it stays untouched:

```ts
import type { AttentionGridCell, Mode, RunEndReason, TraceEvent, TokenInfo } from '../trace/types'
import type { RunRecord } from '../app/runsStore'

export interface FixtureTraceOpts {
  cycles?: number
  layers?: number
  prompt?: string
  promptTokens?: TokenInfo[]
  // one chosen token per cycle; defaults reproduce makeFixtureTrace exactly
  chosen?: TokenInfo[]
  temperature?: number
  mode?: Mode
  reason?: RunEndReason
}

const DEFAULT_WORDS = [' sat', ' on', ' the', ' mat']

export function buildFixtureTrace(opts: FixtureTraceOpts = {}): TraceEvent[] {
  const cycles = opts.cycles ?? 2
  const layers = opts.layers ?? 3
  const promptTokens = opts.promptTokens ?? [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }]
  const temperature = opts.temperature ?? 0.8
  const chosenFor = (c: number): TokenInfo =>
    opts.chosen?.[c] ?? { id: 100 + c, text: DEFAULT_WORDS[c % DEFAULT_WORDS.length] }
  const events: TraceEvent[] = [
    { type: 'run-start', prompt: opts.prompt ?? 'The cat', mode: opts.mode ?? 'sim', modelId: 'fixture',
      params: { temperature, topK: 10, maxNewTokens: cycles }, vocabSize: 49152 },
    { type: 'tokenize', tokens: promptTokens },
  ]
  for (let c = 0; c < cycles; c++) {
    const chosen = chosenFor(c)
    events.push({ type: 'embed', cycle: c, seqLen: promptTokens.length + c, dims: 576,
      preview: [[0.1, -0.2, 0.3], [0.0, 0.5, -0.1]] })
    for (let l = 0; l < layers; l++) events.push({ type: 'layer', cycle: c, index: l, total: layers })
    const seq = promptTokens.length + c
    const row = (i: number, weights: Array<[number, number]>): number[] => {
      const w = Array.from({ length: i + 1 }, () => 0)
      for (const [pos, mass] of weights) w[Math.min(pos, i)] += mass
      return w
    }
    events.push({ type: 'attention', cycle: c, heads: [
      { layer: 0, head: 3, label: 'attention-sink',
        matrix: Array.from({ length: seq }, (_, i) => i === 0 ? [1] : row(i, [[0, 0.7], [i - 1, 0.2], [i, 0.1]])) },
      { layer: 2, head: 1, label: 'previous-token',
        matrix: Array.from({ length: seq }, (_, i) => i === 0 ? [1] : row(i, [[i - 1, 0.8], [i, 0.2]])) },
    ] })
    events.push({ type: 'logits', cycle: c, topK: [
      { ...chosen, logit: 9.1 }, { id: 200, text: ' ran', logit: 7.2 }, { id: 201, text: ' was', logit: 5.0 },
    ] })
    events.push({ type: 'softmax', cycle: c, temperature, topK: [
      { ...chosen, prob: 0.7 }, { id: 200, text: ' ran', prob: 0.2 }, { id: 201, text: ' was', prob: 0.1 },
    ] })
    events.push({ type: 'sample', cycle: c, chosen, method: 'top-k' })
    events.push({ type: 'append', cycle: c, token: chosen })
  }
  events.push({ type: 'run-end', reason: opts.reason ?? 'max-tokens' })
  return events
}

export function makeFixtureTrace(cycles = 2, layers = 3): TraceEvent[] {
  return buildFixtureTrace({ cycles, layers })
}

export function makeRunRecord(
  seq: number, opts: FixtureTraceOpts & { pinned?: boolean; endedAt?: number; id?: string } = {},
): RunRecord {
  return {
    id: opts.id ?? `run-${seq}`,
    meta: {
      seq,
      prompt: opts.prompt ?? 'The cat',
      params: { temperature: opts.temperature ?? 0.8, topK: 10, maxNewTokens: opts.cycles ?? 2 },
      mode: opts.mode ?? 'sim',
      endedAt: opts.endedAt ?? 1000 + seq,
      reason: opts.reason ?? 'max-tokens',
      pinned: opts.pinned ?? false,
    },
    events: buildFixtureTrace(opts),
  }
}
```

(Every default matches the old literal values, including `temperature` flowing into the `softmax` event, so `makeFixtureTrace()` output is byte-identical. The existing index-based suites are the regression guard.)

- [ ] **Step 2: Run the full existing suite to prove the fixture is frozen**

Run: `npx vitest run`
Expected: PASS with the same test count as before this task (fixture output unchanged). Any failure here means the builder deviates — fix the builder, never the old tests.

- [ ] **Step 3: Write the failing store tests**

Create `src/app/runsStore.test.ts`:

```ts
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `npx vitest run src/app/runsStore.test.ts`
Expected: FAIL — module `./runsStore` does not exist.

- [ ] **Step 5: Implement the store**

Create `src/app/runsStore.ts`:

```ts
import { create } from 'zustand'
import type { GenParams, Mode, RunEndReason, TraceEvent } from '../trace/types'

export interface RunMeta {
  seq: number               // monotonic per-archive counter; never reused, so chip labels
                            // stay stable across evictions
  prompt: string
  params: GenParams
  mode: Mode
  modelId?: string          // real mode only
  endedAt: number           // epoch ms, stamped at seal time
  reason: RunEndReason
  pinned: boolean
}
export interface RunRecord { id: string; meta: RunMeta; events: TraceEvent[] }
export type SealMeta = Omit<RunMeta, 'seq' | 'pinned'>

export const UNPINNED_CAP = 8

interface RunsState {
  records: RunRecord[]      // oldest first
  activeId: string | null
  nextSeq: number
  persistFailed: boolean
  seal: (meta: SealMeta, events: TraceEvent[]) => { record: RunRecord; evicted: RunRecord[] }
  setActive: (id: string) => void
  togglePin: (id: string) => void
  remove: (id: string) => void
  importRecord: (data: { meta: SealMeta; events: TraceEvent[] }) => RunRecord
  hydrate: (records: RunRecord[]) => void
  setPersistFailed: () => void
}

export const useRunsStore = create<RunsState>()((set, get) => ({
  records: [], activeId: null, nextSeq: 1, persistFailed: false,
  seal: (meta, events) => {
    const record: RunRecord = {
      id: crypto.randomUUID(), meta: { ...meta, seq: get().nextSeq, pinned: false }, events,
    }
    const records = [...get().records, record]
    const evicted: RunRecord[] = []
    while (records.filter((r) => !r.meta.pinned).length > UNPINNED_CAP) {
      const oldest = records.find((r) => !r.meta.pinned)
      if (!oldest) break
      evicted.push(oldest)
      records.splice(records.indexOf(oldest), 1)
    }
    set((s) => ({ records, activeId: record.id, nextSeq: s.nextSeq + 1 }))
    return { record, evicted }
  },
  setActive: (id) => set((s) => (s.records.some((r) => r.id === id) ? { activeId: id } : {})),
  togglePin: (id) => set((s) => ({
    records: s.records.map((r) => r.id === id ? { ...r, meta: { ...r.meta, pinned: !r.meta.pinned } } : r),
  })),
  remove: (id) => set((s) => ({
    records: s.records.filter((r) => r.id !== id),
    activeId: s.activeId === id ? null : s.activeId,
  })),
  importRecord: (data) => {
    const record: RunRecord = {
      id: crypto.randomUUID(),
      meta: { ...data.meta, seq: get().nextSeq, pinned: true },   // pinned: imports must not fall off the ring
      events: data.events,
    }
    set((s) => ({ records: [...s.records, record], nextSeq: s.nextSeq + 1 }))
    return record
  },
  hydrate: (records) => {
    const sorted = [...records].sort((a, b) => a.meta.endedAt - b.meta.endedAt)
    set({ records: sorted, nextSeq: sorted.reduce((m, r) => Math.max(m, r.meta.seq), 0) + 1 })
  },
  setPersistFailed: () => set({ persistFailed: true }),
}))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npx vitest run src/app/runsStore.test.ts`
Expected: PASS. Then `npx vitest run` — full suite PASS.

- [ ] **Step 7: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/app/runsStore.ts src/app/runsStore.test.ts src/test/fixtures.ts
git commit -m "feat: runs store with sealed records, pin-aware ring, run fixtures

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

---

### Task 2: storage adapter and archive orchestration

**Files:**
- Create: `src/app/runStorage.ts`
- Create: `src/app/runArchive.ts`
- Test: `src/app/runArchive.test.ts`

**Interfaces:**
- Consumes: `useRunsStore`, `RunRecord`, `SealMeta` (Task 1).
- Produces:
  - `export interface RunStorage { loadAll(): Promise<RunRecord[]>; put(record: RunRecord): Promise<void>; delete(id: string): Promise<void> }`
  - `export function createMemoryStorage(opts?: { failing?: boolean }): RunStorage & { map: Map<string, RunRecord> }`
  - `export function createIndexedDbStorage(): RunStorage` (database `tsumugi`, object store `runs`, key `id`)
  - `runArchive.ts`: `initArchive(s: RunStorage): Promise<void>`, `archiveSeal(meta: SealMeta, events: TraceEvent[]): RunRecord`, `archiveTogglePin(id: string): void`, `archiveRemove(id: string): void`, `archiveImport(data: { meta: SealMeta; events: TraceEvent[] }): RunRecord`, `_resetArchiveForTests(): void`

- [ ] **Step 1: Write the failing tests**

Create `src/app/runArchive.test.ts`:

```ts
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

test('a failing adapter degrades to session-only without throwing', async () => {
  await initArchive(createMemoryStorage({ failing: true }))
  expect(useRunsStore.getState().persistFailed).toBe(true)
  // store operations still work session-only
  const record = archiveSeal(meta(), makeFixtureTrace())
  expect(useRunsStore.getState().records[0].id).toBe(record.id)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/app/runArchive.test.ts`
Expected: FAIL — modules `./runArchive` and `./runStorage` do not exist.

- [ ] **Step 3: Implement the storage adapter**

Create `src/app/runStorage.ts`:

```ts
import type { RunRecord } from './runsStore'

export interface RunStorage {
  loadAll(): Promise<RunRecord[]>
  put(record: RunRecord): Promise<void>
  delete(id: string): Promise<void>
}

export function createMemoryStorage(opts: { failing?: boolean } = {}): RunStorage & { map: Map<string, RunRecord> } {
  const map = new Map<string, RunRecord>()
  const fail = () => Promise.reject(new Error('storage unavailable'))
  return {
    map,
    loadAll: () => (opts.failing ? fail() : Promise.resolve([...map.values()])),
    put: (record) => { if (opts.failing) return fail(); map.set(record.id, record); return Promise.resolve() },
    delete: (id) => { if (opts.failing) return fail(); map.delete(id); return Promise.resolve() },
  }
}

// IndexedDB-backed storage. Every rejection is treated by callers as non-fatal
// (session-only archive) — including environments without indexedDB (jsdom, old browsers).
export function createIndexedDbStorage(): RunStorage {
  const open = (): Promise<IDBDatabase> => new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('indexedDB unavailable')); return }
    const req = indexedDB.open('tsumugi', 1)
    req.onupgradeneeded = () => { req.result.createObjectStore('runs', { keyPath: 'id' }) }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error ?? new Error('indexedDB open failed'))
  })
  const inTx = async <T,>(mode: IDBTransactionMode, op: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> => {
    const db = await open()
    try {
      return await new Promise<T>((resolve, reject) => {
        const request = op(db.transaction('runs', mode).objectStore('runs'))
        request.onsuccess = () => resolve(request.result)
        request.onerror = () => reject(request.error ?? new Error('indexedDB request failed'))
      })
    } finally { db.close() }
  }
  return {
    loadAll: () => inTx('readonly', (s) => s.getAll() as IDBRequest<RunRecord[]>),
    put: async (record) => { await inTx('readwrite', (s) => s.put(record)) },
    delete: async (id) => { await inTx('readwrite', (s) => s.delete(id)) },
  }
}
```

- [ ] **Step 4: Implement the orchestrator**

Create `src/app/runArchive.ts`:

```ts
import type { TraceEvent } from '../trace/types'
import type { RunStorage } from './runStorage'
import { useRunsStore, type RunRecord, type SealMeta } from './runsStore'

// Store ↔ storage glue. The store stays pure; this module mirrors every archive
// mutation to the adapter, fire-and-forget. Any storage failure flips the
// store's persistFailed flag — the archive continues session-only, and
// generation is never blocked or delayed (M1 never-fail policy).
let storage: RunStorage | null = null

const markFailed = () => useRunsStore.getState().setPersistFailed()
const mirror = (op: (s: RunStorage) => Promise<unknown>) => {
  if (storage) void op(storage).catch(markFailed)
}

export async function initArchive(s: RunStorage): Promise<void> {
  storage = s
  try {
    useRunsStore.getState().hydrate(await s.loadAll())
  } catch {
    markFailed()
  }
}

export function archiveSeal(meta: SealMeta, events: TraceEvent[]): RunRecord {
  const { record, evicted } = useRunsStore.getState().seal(meta, events)
  mirror((s) => s.put(record))
  for (const r of evicted) mirror((s) => s.delete(r.id))
  return record
}

export function archiveTogglePin(id: string): void {
  useRunsStore.getState().togglePin(id)
  const record = useRunsStore.getState().records.find((r) => r.id === id)
  if (record) mirror((s) => s.put(record))
}

export function archiveRemove(id: string): void {
  useRunsStore.getState().remove(id)
  mirror((s) => s.delete(id))
}

export function archiveImport(data: { meta: SealMeta; events: TraceEvent[] }): RunRecord {
  const record = useRunsStore.getState().importRecord(data)
  mirror((s) => s.put(record))
  return record
}

export function _resetArchiveForTests(): void { storage = null }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/app/runArchive.test.ts`
Expected: PASS.

- [ ] **Step 6: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/app/runStorage.ts src/app/runArchive.ts src/app/runArchive.test.ts
git commit -m "feat: run archive persistence — adapter interface, IndexedDB, never-fail write-through

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

---

### Task 3: run file export/import

**Files:**
- Create: `src/app/runFiles.ts`
- Test: `src/app/runFiles.test.ts`

**Interfaces:**
- Consumes: `RunRecord`, `RunMeta`, `SealMeta` (Task 1); `validateTrace(events: TraceEvent[]): string[]` from `src/trace/validate`; `makeRunRecord` (Task 1).
- Produces:
  - `export const RUN_FILE_VERSION = 1`
  - `export function serializeRun(record: RunRecord): { filename: string; json: string }`
  - `export type ParsedRunFile = { ok: true; meta: SealMeta; events: TraceEvent[] } | { ok: false; error: string }`
  - `export function parseRunFile(text: string): ParsedRunFile`

- [ ] **Step 1: Write the failing tests**

Create `src/app/runFiles.test.ts`:

```ts
import { expect, test } from 'vitest'
import { makeRunRecord } from '../test/fixtures'
import { parseRunFile, serializeRun } from './runFiles'

test('serialize → parse round-trips meta and events', () => {
  const record = makeRunRecord(3, { endedAt: Date.UTC(2026, 7, 28) })
  const { filename, json } = serializeRun(record)
  expect(filename).toBe('tsumugi-run-the-cat-20260828.json')
  const parsed = parseRunFile(json)
  expect(parsed.ok).toBe(true)
  if (parsed.ok) {
    expect(parsed.meta.prompt).toBe('The cat')
    expect(parsed.meta.endedAt).toBe(Date.UTC(2026, 7, 28))
    expect(parsed.events).toEqual(record.events)
  }
})

test('rejects invalid JSON, wrong version, and missing meta', () => {
  expect(parseRunFile('{nope')).toEqual({ ok: false, error: 'not valid JSON' })
  const record = makeRunRecord(1)
  const wrongVersion = JSON.stringify({ formatVersion: 2, meta: record.meta, events: record.events })
  expect(parseRunFile(wrongVersion)).toMatchObject({ ok: false, error: expect.stringContaining('format version') })
  const noMeta = JSON.stringify({ formatVersion: 1, events: record.events })
  expect(parseRunFile(noMeta)).toMatchObject({ ok: false, error: expect.stringContaining('metadata') })
})

test('rejects a file whose trace fails validation', () => {
  const record = makeRunRecord(1)
  const broken = JSON.stringify({ formatVersion: 1, meta: record.meta,
    events: [...record.events].reverse() })
  expect(parseRunFile(broken)).toMatchObject({ ok: false, error: expect.stringContaining('invalid trace') })
})

test('slug falls back for an unusable prompt', () => {
  const record = makeRunRecord(1, { prompt: '   ', endedAt: Date.UTC(2026, 7, 28) })
  expect(serializeRun(record).filename).toBe('tsumugi-run-run-20260828.json')
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/app/runFiles.test.ts`
Expected: FAIL — module `./runFiles` does not exist.

- [ ] **Step 3: Implement**

Create `src/app/runFiles.ts`:

```ts
import type { GenParams, RunEndReason, TraceEvent } from '../trace/types'
import { validateTrace } from '../trace/validate'
import type { RunMeta, RunRecord, SealMeta } from './runsStore'

export const RUN_FILE_VERSION = 1

export function serializeRun(record: RunRecord): { filename: string; json: string } {
  const slug = record.meta.prompt.trim().toLowerCase().split(/\s+/).slice(0, 3).join('-')
    .replace(/[^a-z0-9-]/g, '') || 'run'
  const date = new Date(record.meta.endedAt).toISOString().slice(0, 10).replace(/-/g, '')
  return {
    filename: `tsumugi-run-${slug}-${date}.json`,
    json: JSON.stringify({ formatVersion: RUN_FILE_VERSION, meta: record.meta, events: record.events }, null, 2),
  }
}

export type ParsedRunFile =
  | { ok: true; meta: SealMeta; events: TraceEvent[] }
  | { ok: false; error: string }

export function parseRunFile(text: string): ParsedRunFile {
  let data: unknown
  try { data = JSON.parse(text) } catch { return { ok: false, error: 'not valid JSON' } }
  if (typeof data !== 'object' || data === null) return { ok: false, error: 'not a run file' }
  const d = data as { formatVersion?: unknown; meta?: unknown; events?: unknown }
  if (d.formatVersion !== RUN_FILE_VERSION)
    return { ok: false, error: `unsupported format version (expected ${RUN_FILE_VERSION})` }
  const m = (typeof d.meta === 'object' && d.meta !== null ? d.meta : {}) as Partial<RunMeta>
  if (typeof m.prompt !== 'string' || typeof m.endedAt !== 'number'
    || (m.mode !== 'sim' && m.mode !== 'real') || typeof m.reason !== 'string'
    || typeof m.params !== 'object' || m.params === null)
    return { ok: false, error: 'missing or malformed run metadata' }
  if (!Array.isArray(d.events)) return { ok: false, error: 'missing events' }
  const events = d.events as TraceEvent[]
  const problems = validateTrace(events)
  if (problems.length > 0) return { ok: false, error: `invalid trace: ${problems[0]}` }
  return {
    ok: true,
    meta: {
      prompt: m.prompt, params: m.params as GenParams, mode: m.mode,
      ...(typeof m.modelId === 'string' ? { modelId: m.modelId } : {}),
      endedAt: m.endedAt, reason: m.reason as RunEndReason,
    },
    events,
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/app/runFiles.test.ts`
Expected: PASS. (If the reversed-trace test fails because `validateTrace` accepts it, pick a mutation it must flag — e.g. dropping the `run-start` with `record.events.slice(1)` — and keep the assertion; do NOT weaken `parseRunFile`.)

- [ ] **Step 5: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/app/runFiles.ts src/app/runFiles.test.ts
git commit -m "feat: run file export serialization and validated import parsing

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

---

### Task 4: `RunShelf` component

**Files:**
- Create: `src/app/RunShelf.tsx`
- Modify: `src/index.css` (append shelf styles)
- Test: `src/app/RunShelf.test.tsx`

**Interfaces:**
- Consumes: `RunRecord` (Task 1), `makeRunRecord` (Task 1).
- Produces:

```ts
export interface RunShelfProps {
  records: RunRecord[]
  activeId: string | null
  compare: { aId: string; bId: string } | null
  armed: boolean
  persistFailed: boolean
  importError: string | null
  onActivate(id: string): void
  onSelectCompareB(id: string): void
  onArmCompare(): void
  onExitCompare(): void
  onTogglePin(id: string): void
  onExport(id: string): void
  onRemove(id: string): void
  onImportFile(file: File): void
}
export function RunShelf(props: RunShelfProps): ReactNode
```

- Behavior: empty archive → renders `null`. One chip per record (oldest left): main button labeled `#<seq> · <first 3 words> · T=<t>` with mode glyph (○ sim / ● real) and 📌 when pinned; `data-active`, `data-pinned`, `data-role` (`a`/`b` while armed/comparing). Main click: comparing/armed → `onSelectCompareB` (ignored for run A); otherwise `onActivate`. Per-chip actions: pin, export, remove. Tools: compare arm (hidden with <2 records; disabled with no active run) ↔ exit-compare toggle, import via hidden file input. Notes: `importError`, `persistFailed`.

- [ ] **Step 1: Write the failing tests**

Create `src/app/RunShelf.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeRunRecord } from '../test/fixtures'
import { RunShelf, type RunShelfProps } from './RunShelf'

afterEach(() => cleanup())

const props = (over: Partial<RunShelfProps> = {}): RunShelfProps => ({
  records: [makeRunRecord(1), makeRunRecord(2, { temperature: 0.2 })],
  activeId: 'run-2', compare: null, armed: false, persistFailed: false, importError: null,
  onActivate: vi.fn(), onSelectCompareB: vi.fn(), onArmCompare: vi.fn(), onExitCompare: vi.fn(),
  onTogglePin: vi.fn(), onExport: vi.fn(), onRemove: vi.fn(), onImportFile: vi.fn(),
  ...over,
})

test('renders nothing for an empty archive', () => {
  const { container } = render(<RunShelf {...props({ records: [] })} />)
  expect(container.firstChild).toBeNull()
})

test('chips show seq labels and the active run is marked', () => {
  render(<RunShelf {...props()} />)
  const chips = screen.getAllByTestId('run-chip')
  expect(chips).toHaveLength(2)
  expect(chips[0]).toHaveTextContent('#1 · The cat · T=0.8')
  expect(chips[1]).toHaveTextContent('T=0.2')
  expect(chips[0].dataset.active).toBe('false')
  expect(chips[1].dataset.active).toBe('true')
})

test('main click activates when not armed', () => {
  const p = props()
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  expect(p.onActivate).toHaveBeenCalledWith('run-1')
})

test('armed: clicking a non-active chip selects run B; the active chip is ignored', () => {
  const p = props({ armed: true })
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])   // active run = A
  expect(p.onSelectCompareB).not.toHaveBeenCalled()
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  expect(p.onSelectCompareB).toHaveBeenCalledWith('run-1')
  expect(p.onActivate).not.toHaveBeenCalled()
})

test('compare roles are marked and the exit control shows', () => {
  render(<RunShelf {...props({ compare: { aId: 'run-2', bId: 'run-1' } })} />)
  const chips = screen.getAllByTestId('run-chip')
  expect(chips[1].dataset.role).toBe('a')
  expect(chips[0].dataset.role).toBe('b')
  expect(screen.getByTestId('btn-compare-exit')).toBeInTheDocument()
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
})

test('a single record hides the compare affordance', () => {
  render(<RunShelf {...props({ records: [makeRunRecord(1)], activeId: 'run-1' })} />)
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
})

test('pinned marker, pin/remove/export actions', () => {
  const p = props({ records: [makeRunRecord(1, { pinned: true })], activeId: 'run-1' })
  render(<RunShelf {...p} />)
  expect(screen.getByTestId('run-chip').dataset.pinned).toBe('true')
  fireEvent.click(screen.getByTestId('btn-chip-pin'))
  expect(p.onTogglePin).toHaveBeenCalledWith('run-1')
  fireEvent.click(screen.getByTestId('btn-chip-export'))
  expect(p.onExport).toHaveBeenCalledWith('run-1')
  fireEvent.click(screen.getByTestId('btn-chip-remove'))
  expect(p.onRemove).toHaveBeenCalledWith('run-1')
})

test('import file input forwards the file; notes render', () => {
  const p = props({ persistFailed: true, importError: 'bad file' })
  render(<RunShelf {...p} />)
  const file = new File(['{}'], 'run.json', { type: 'application/json' })
  fireEvent.change(screen.getByTestId('import-input'), { target: { files: [file] } })
  expect(p.onImportFile).toHaveBeenCalledWith(file)
  expect(screen.getByTestId('shelf-note')).toHaveTextContent('not persisted')
  expect(screen.getByTestId('import-error')).toHaveTextContent('bad file')
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/app/RunShelf.test.tsx`
Expected: FAIL — module `./RunShelf` does not exist.

- [ ] **Step 3: Implement the component**

Create `src/app/RunShelf.tsx`:

```tsx
import { useRef } from 'react'
import type { RunRecord } from './runsStore'

export interface RunShelfProps {
  records: RunRecord[]
  activeId: string | null
  compare: { aId: string; bId: string } | null
  armed: boolean
  persistFailed: boolean
  importError: string | null
  onActivate(id: string): void
  onSelectCompareB(id: string): void
  onArmCompare(): void
  onExitCompare(): void
  onTogglePin(id: string): void
  onExport(id: string): void
  onRemove(id: string): void
  onImportFile(file: File): void
}

const label = (r: RunRecord) =>
  `#${r.meta.seq} · ${r.meta.prompt.trim().split(/\s+/).slice(0, 3).join(' ')} · T=${r.meta.params.temperature}`

export function RunShelf(p: RunShelfProps) {
  const fileRef = useRef<HTMLInputElement>(null)
  if (p.records.length === 0) return null
  const selecting = p.armed || p.compare !== null
  const aId = p.compare?.aId ?? (p.armed ? p.activeId : null)
  const chipClick = (id: string) => {
    if (selecting) { if (id !== aId) p.onSelectCompareB(id); return }
    p.onActivate(id)
  }
  const role = (id: string) => (id === aId ? 'a' : id === p.compare?.bId ? 'b' : '')
  return (
    <div data-testid="run-shelf" className="run-shelf">
      {p.records.map((r) => (
        <div key={r.id} data-testid="run-chip" className="run-chip"
          data-active={String(r.id === p.activeId)} data-pinned={String(r.meta.pinned)} data-role={role(r.id)}>
          <button data-testid="run-chip-main" className="run-chip-main" onClick={() => chipClick(r.id)}
            title={`${r.meta.prompt} — ${r.meta.mode} · ended: ${r.meta.reason}`}>
            <span className="run-chip-glyph" aria-hidden="true">{r.meta.mode === 'real' ? '●' : '○'}</span>
            {label(r)}{r.meta.pinned && ' 📌'}
          </button>
          <button data-testid="btn-chip-pin" className="run-chip-action"
            title={r.meta.pinned ? 'unpin' : 'pin — never evicted'} onClick={() => p.onTogglePin(r.id)}>📌</button>
          <button data-testid="btn-chip-export" className="run-chip-action"
            title="download run as JSON" onClick={() => p.onExport(r.id)}>⇩</button>
          <button data-testid="btn-chip-remove" className="run-chip-action"
            title="remove run" onClick={() => p.onRemove(r.id)}>×</button>
        </div>
      ))}
      <span className="run-shelf-tools">
        {p.records.length > 1 && (selecting
          ? <button data-testid="btn-compare-exit" className="run-shelf-btn" onClick={p.onExitCompare}>× exit compare</button>
          : <button data-testid="btn-compare-arm" className="run-shelf-btn" disabled={p.activeId === null}
              title="pick a second run to compare against the active one" onClick={p.onArmCompare}>⇄ compare</button>)}
        <button data-testid="btn-import" className="run-shelf-btn" title="load an exported run"
          onClick={() => fileRef.current?.click()}>⇧ import</button>
        <input ref={fileRef} data-testid="import-input" type="file" accept="application/json" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) p.onImportFile(f); e.target.value = '' }} />
      </span>
      {p.importError && <span data-testid="import-error" className="run-shelf-note">{p.importError}</span>}
      {p.persistFailed && <span data-testid="shelf-note" className="run-shelf-note">archive not persisted in this browser</span>}
    </div>
  )
}
```

- [ ] **Step 4: Add the styles**

Append to `src/index.css`:

```css
.run-shelf { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem; margin: .5rem 0; }
.run-chip { display: inline-flex; align-items: center; border: 1px solid var(--line-strong);
  border-radius: 999px; background: var(--card); padding: 0 .3rem 0 0; }
.run-chip[data-active="true"] { border-color: var(--ai); background: var(--ai-wash); }
.run-chip[data-role="a"], .run-chip[data-role="b"] { border-color: var(--shu); background: var(--shu-wash); }
.run-chip-main { font: inherit; font-size: .76rem; background: none; border: none; cursor: pointer;
  padding: .18rem .3rem .18rem .55rem; color: var(--ink); }
.run-chip-glyph { color: var(--ai); margin-right: .3rem; }
.run-chip-action { font-size: .68rem; background: none; border: none; cursor: pointer;
  color: var(--ink-faint); padding: .1rem .15rem; }
.run-chip-action:hover { color: var(--shu); }
.run-shelf-btn { font: inherit; font-size: .76rem; color: var(--ai-deep); background: var(--ai-wash);
  border: 1px solid var(--line-strong); border-radius: 6px; padding: .15rem .5rem; cursor: pointer; }
.run-shelf-btn:disabled { opacity: .5; cursor: default; }
.run-shelf-note { font-size: .72rem; color: var(--ink-faint); }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/app/RunShelf.test.tsx`
Expected: PASS.

- [ ] **Step 6: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/app/RunShelf.tsx src/app/RunShelf.test.tsx src/index.css
git commit -m "feat: run shelf — chips, pin/export/remove/import, compare arm flow

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

---

### Task 5: compare selectors

**Files:**
- Create: `src/viz/compare/compareSelectors.ts`
- Test: `src/viz/compare/compareSelectors.test.ts`

**Interfaces:**
- Consumes: `distributionFor` from `../selectors`; trace types; `buildFixtureTrace` (Task 1).
- Produces:
  - `export interface AlignedRuns { promptA: TokenInfo[]; promptB: TokenInfo[]; chosenA: TokenInfo[]; chosenB: TokenInfo[]; maxCycles: number; samePrompt: boolean; forkCycle: number | null }`
  - `export function alignRuns(a: TraceEvent[], b: TraceEvent[]): AlignedRuns` — prompts compared by token ids; `forkCycle` = first cycle < min length whose chosen ids differ, only when `samePrompt`; else `null`.
  - `export interface PairedHead { layer: number; head: number; a?: AttentionHead; b?: AttentionHead }`
  - `export function pairedHeads(a: TraceEvent[], b: TraceEvent[], cycle: number): PairedHead[]` — union keyed `(layer, head)`, sorted by layer then head.
  - `export function pairedDistributions(a: TraceEvent[], b: TraceEvent[], cycle: number): { a: ReturnType<typeof distributionFor>; b: ReturnType<typeof distributionFor> }`

- [ ] **Step 1: Write the failing tests**

Create `src/viz/compare/compareSelectors.test.ts`:

```ts
import { expect, test } from 'vitest'
import { buildFixtureTrace } from '../../test/fixtures'
import type { TraceEvent } from '../../trace/types'
import { alignRuns, pairedDistributions, pairedHeads } from './compareSelectors'

test('identical runs: same prompt, no fork, cycle counts', () => {
  const r = alignRuns(buildFixtureTrace(), buildFixtureTrace())
  expect(r.samePrompt).toBe(true)
  expect(r.forkCycle).toBeNull()
  expect(r.maxCycles).toBe(2)
  expect(r.chosenA.map((t) => t.text)).toEqual([' sat', ' on'])
})

test('divergent chosen tokens: fork at the first differing cycle', () => {
  const a = buildFixtureTrace()
  const b = buildFixtureTrace({ chosen: [{ id: 100, text: ' sat' }, { id: 999, text: ' off' }] })
  const r = alignRuns(a, b)
  expect(r.samePrompt).toBe(true)
  expect(r.forkCycle).toBe(1)
})

test('different prompts: no fork marker even when outputs differ', () => {
  const b = buildFixtureTrace({ prompt: 'A dog', promptTokens: [{ id: 20, text: 'A' }, { id: 21, text: ' dog' }],
    chosen: [{ id: 500, text: ' ran' }, { id: 501, text: ' far' }] })
  const r = alignRuns(buildFixtureTrace(), b)
  expect(r.samePrompt).toBe(false)
  expect(r.forkCycle).toBeNull()
})

test('length mismatch: maxCycles covers the longer run, no fork on an equal prefix', () => {
  const r = alignRuns(buildFixtureTrace(), buildFixtureTrace({ cycles: 1 }))
  expect(r.maxCycles).toBe(2)
  expect(r.chosenB).toHaveLength(1)
  expect(r.forkCycle).toBeNull()
})

test('pairedHeads unions by (layer, head) in order', () => {
  const attn = (cycle: number, heads: Array<[number, number]>): TraceEvent => ({
    type: 'attention', cycle,
    heads: heads.map(([layer, head]) => ({ layer, head, label: 'previous-token' as const, matrix: [[1]] })),
  })
  const a: TraceEvent[] = [attn(0, [[0, 3], [2, 1]])]
  const b: TraceEvent[] = [attn(0, [[2, 1], [5, 0]])]
  const pairs = pairedHeads(a, b, 0)
  expect(pairs.map((p) => [p.layer, p.head])).toEqual([[0, 3], [2, 1], [5, 0]])
  expect(pairs[0].a).toBeDefined(); expect(pairs[0].b).toBeUndefined()
  expect(pairs[1].a).toBeDefined(); expect(pairs[1].b).toBeDefined()
  expect(pairs[2].a).toBeUndefined(); expect(pairs[2].b).toBeDefined()
})

test('pairedDistributions returns per-side data, undefined past a run end', () => {
  const short = buildFixtureTrace({ cycles: 1 })
  const d = pairedDistributions(buildFixtureTrace(), short, 1)
  expect(d.a?.sample.chosen.text).toBe(' on')
  expect(d.b).toBeUndefined()
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/viz/compare/compareSelectors.test.ts`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `src/viz/compare/compareSelectors.ts`:

```ts
import type { AttentionHead, TokenInfo, TraceEvent } from '../../trace/types'
import { distributionFor } from '../selectors'

export interface AlignedRuns {
  promptA: TokenInfo[]
  promptB: TokenInfo[]
  chosenA: TokenInfo[]
  chosenB: TokenInfo[]
  maxCycles: number
  samePrompt: boolean
  forkCycle: number | null
}

const promptOf = (events: TraceEvent[]): TokenInfo[] => {
  for (const e of events) if (e.type === 'tokenize') return e.tokens
  return []
}
const chosenOf = (events: TraceEvent[]): TokenInfo[] => {
  const out: TokenInfo[] = []
  for (const e of events) if (e.type === 'append') out[e.cycle] = e.token
  return out
}

export function alignRuns(a: TraceEvent[], b: TraceEvent[]): AlignedRuns {
  const promptA = promptOf(a), promptB = promptOf(b)
  const chosenA = chosenOf(a), chosenB = chosenOf(b)
  const samePrompt = promptA.length === promptB.length && promptA.every((t, i) => t.id === promptB[i].id)
  let forkCycle: number | null = null
  if (samePrompt) {
    for (let c = 0; c < Math.min(chosenA.length, chosenB.length); c++) {
      if (chosenA[c].id !== chosenB[c].id) { forkCycle = c; break }
    }
  }
  return { promptA, promptB, chosenA, chosenB,
    maxCycles: Math.max(chosenA.length, chosenB.length), samePrompt, forkCycle }
}

export interface PairedHead { layer: number; head: number; a?: AttentionHead; b?: AttentionHead }

const headsAt = (events: TraceEvent[], cycle: number): AttentionHead[] => {
  for (const e of events) if (e.type === 'attention' && e.cycle === cycle) return e.heads
  return []
}

export function pairedHeads(a: TraceEvent[], b: TraceEvent[], cycle: number): PairedHead[] {
  const byKey = new Map<string, PairedHead>()
  for (const h of headsAt(a, cycle)) byKey.set(`${h.layer}-${h.head}`, { layer: h.layer, head: h.head, a: h })
  for (const h of headsAt(b, cycle)) {
    const key = `${h.layer}-${h.head}`
    const existing = byKey.get(key)
    if (existing) existing.b = h
    else byKey.set(key, { layer: h.layer, head: h.head, b: h })
  }
  return [...byKey.values()].sort((x, y) => x.layer - y.layer || x.head - y.head)
}

export function pairedDistributions(a: TraceEvent[], b: TraceEvent[], cycle: number) {
  return { a: distributionFor(a, cycle), b: distributionFor(b, cycle) }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/viz/compare/compareSelectors.test.ts`
Expected: PASS.

- [ ] **Step 5: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/viz/compare/compareSelectors.ts src/viz/compare/compareSelectors.test.ts
git commit -m "feat: cycle-alignment, fork detection, and pairing selectors for run comparison

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

---

### Task 6: `CompareView` component

**Files:**
- Modify: `src/viz/AttentionGridExplorer.tsx:31` (`function Thumb` → `export function Thumb`; no other change)
- Create: `src/viz/compare/CompareView.tsx`
- Modify: `src/index.css` (append compare styles)
- Test: `src/viz/compare/CompareView.test.tsx`

**Interfaces:**
- Consumes: `RunRecord` (Task 1); `alignRuns`/`pairedHeads` (Task 5); `distributionFor`, `latestOfType`, `visibleTokens` from `../selectors`; `AttentionHeatmap`; `Thumb`; `makeRunRecord`, `makeGridEvent` (fixtures).
- Produces: `export function CompareView({ a, b }: { a: RunRecord; b: RunRecord }): ReactNode`. No exit control here — exit lives on the shelf (single control, spec's "one action exits").
- Behavior: metadata rows with `data-diff`; badge `different prompts — aligned by generation cycle` when prompts differ; note `identical outputs` when samePrompt ∧ forkCycle null ∧ equal lengths; two streams with `data-fork` on the fork token; clickable cycle ruler; on selection, paired distribution columns (`data-chosen` on the sampled token, "run ended at cycle N" for out-of-range sides) and paired attention (union chips; per side the full matrix via `AttentionHeatmap heads={[head]}` with that run's full token list, else grid-thumbnail fallback, else "not captured in this run").

- [ ] **Step 1: Write the failing tests**

Create `src/viz/compare/CompareView.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeGridEvent, makeRunRecord } from '../../test/fixtures'
import type { RunRecord } from '../../app/runsStore'
import { CompareView } from './CompareView'

vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never)

afterEach(() => cleanup())

const divergentB = () => makeRunRecord(2, { chosen: [{ id: 100, text: ' sat' }, { id: 999, text: ' off' }] })

test('identical runs: no badge, identical-outputs note, no fork marks', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2)} />)
  expect(screen.queryByTestId('cmp-badge')).toBeNull()
  expect(screen.getByTestId('cmp-note')).toHaveTextContent('identical outputs')
  expect(screen.getAllByTestId('cmp-token').every((t) => t.dataset.fork === 'false')).toBe(true)
})

test('metadata rows highlight only differing fields', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2, { temperature: 0.2 })} />)
  const rows = screen.getAllByTestId('cmp-meta-row')
  const tRow = rows.find((r) => r.textContent?.startsWith('T'))
  const promptRow = rows.find((r) => r.textContent?.startsWith('prompt'))
  expect(tRow?.dataset.diff).toBe('true')
  expect(promptRow?.dataset.diff).toBe('false')
})

test('fork cycle is marked on both streams and the ruler', () => {
  render(<CompareView a={makeRunRecord(1)} b={divergentB()} />)
  const forked = screen.getAllByTestId('cmp-token').filter((t) => t.dataset.fork === 'true')
  expect(forked).toHaveLength(2)              // one per stream, at cycle 1
  expect(forked.map((t) => t.textContent)).toEqual([' on', ' off'])
  const ticks = screen.getAllByTestId('cmp-tick')
  expect(ticks[1].dataset.fork).toBe('true')
  expect(screen.queryByTestId('cmp-note')).toBeNull()
})

test('different prompts show the badge instead of a fork', () => {
  const b = makeRunRecord(2, { prompt: 'A dog', promptTokens: [{ id: 20, text: 'A' }, { id: 21, text: ' dog' }] })
  render(<CompareView a={makeRunRecord(1)} b={b} />)
  expect(screen.getByTestId('cmp-badge')).toHaveTextContent('different prompts')
})

test('clicking a ruler tick opens paired distributions with the chosen token marked', () => {
  render(<CompareView a={makeRunRecord(1)} b={divergentB()} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[1])
  const sides = screen.getAllByTestId('cmp-dist-side')
  expect(sides).toHaveLength(2)
  const chosen = screen.getAllByTestId('cmp-bar-row').filter((r) => r.dataset.chosen === 'true')
  expect(chosen).toHaveLength(2)
})

test('a cycle past one run\'s end shows the run-ended side', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2, { cycles: 1 })} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[1])
  // B's side renders "run ended" in both the distributions and attention panels
  const ended = screen.getAllByTestId('cmp-ended')
  expect(ended.length).toBeGreaterThanOrEqual(1)
  expect(ended[0]).toHaveTextContent('run ended at cycle 0')
})

test('paired attention: shared heads render two heatmaps; a missing head falls back', () => {
  const a = makeRunRecord(1)
  const bBase = makeRunRecord(2)
  // strip L0·H3 from run B's attention events → that chip has no B-side matrix
  const b: RunRecord = { ...bBase, events: bBase.events.map((e) =>
    e.type === 'attention' ? { ...e, heads: e.heads.filter((h) => !(h.layer === 0 && h.head === 3)) } : e) }
  render(<CompareView a={a} b={b} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[0])
  const chips = screen.getAllByTestId('cmp-head-chip')
  expect(chips.map((c) => c.textContent)).toEqual(['L0·H3attention-sink', 'L2·H1previous-token'])
  // default selection = first chip (L0·H3): A side full heatmap, B side sim fallback note
  expect(screen.getAllByTestId('attention-heatmap')).toHaveLength(1)
  expect(screen.getByTestId('cmp-fallback')).toHaveTextContent('not captured in this run')
  // select the shared head: both sides render full heatmaps
  fireEvent.click(chips[1])
  expect(screen.getAllByTestId('attention-heatmap')).toHaveLength(2)
})

test('a run with a grid uses the run-level thumbnail as fallback', () => {
  const a = makeRunRecord(1)
  const bBase = makeRunRecord(2, { mode: 'real' })
  const grid = makeGridEvent(2, 4)   // covers layer 0, head 3
  const b: RunRecord = { ...bBase, events: [
    ...bBase.events.slice(0, -1).map((e) =>
      e.type === 'attention' ? { ...e, heads: e.heads.filter((h) => !(h.layer === 0 && h.head === 3)) } : e),
    grid, bBase.events.at(-1)!,
  ] }
  render(<CompareView a={a} b={b} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[0])
  expect(screen.getByTestId('cmp-fallback')).toHaveTextContent('run-level thumbnail')
})
```

Note on the chip-text assertion: chip content is `L{layer}·H{head}` plus the label span; the exact rendered strings are `L0·H3attention-sink` and `L2·H1previous-token` (the fixture's two heads). Write the expectation as those two literal strings.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/viz/compare/CompareView.test.tsx`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Export `Thumb`**

In `src/viz/AttentionGridExplorer.tsx` line 31: change `function Thumb(` to `export function Thumb(`. Nothing else.

- [ ] **Step 4: Implement the component**

Create `src/viz/compare/CompareView.tsx`:

```tsx
import { useState } from 'react'
import type { RunRecord } from '../../app/runsStore'
import type { AttentionHead, TokenInfo, TraceEvent } from '../../trace/types'
import { AttentionHeatmap } from '../AttentionHeatmap'
import { Thumb } from '../AttentionGridExplorer'
import { distributionFor, latestOfType, visibleTokens } from '../selectors'
import { alignRuns, pairedHeads, type PairedHead } from './compareSelectors'

const runTokens = (events: TraceEvent[]): TokenInfo[] => {
  const { prompt, generated } = visibleTokens(events, events.length - 1)
  return [...prompt, ...generated]
}

const gridThumbFor = (events: TraceEvent[], layer: number, head: number): number[][] | null => {
  const grid = latestOfType(events, events.length - 1, 'attention-grid')
  return grid?.cells.find((c) => c.layer === layer && c.head === head)?.thumb ?? null
}

function MetaRow({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div data-testid="cmp-meta-row" data-diff={String(a !== b)} className="cmp-meta-row">
      <span className="cmp-meta-label">{label}</span>
      <span className="cmp-meta-value">{a}</span>
      <span className="cmp-meta-value">{b}</span>
    </div>
  )
}

function DistSide({ events, cycle, name, lastCycle }: {
  events: TraceEvent[]; cycle: number; name: string; lastCycle: number
}) {
  const d = distributionFor(events, cycle)
  if (!d) return <div data-testid="cmp-ended" className="cmp-ended">{name}: run ended at cycle {lastCycle}</div>
  return (
    <div data-testid="cmp-dist-side" className="cmp-dist-side">
      <span className="cmp-side-name">{name} · T={d.softmax.temperature}</span>
      {d.softmax.topK.map((t) => (
        <div key={t.id} data-testid="cmp-bar-row" data-chosen={String(t.id === d.sample.chosen.id)}
          className="cmp-bar-row">
          <span className="cmp-bar-token">{t.text.trim() || '·'}</span>
          <span className="cmp-bar" style={{ width: `${Math.max(2, Math.round(t.prob * 100))}%` }} />
          <span className="cmp-bar-prob">{Math.round(t.prob * 100)}%</span>
        </div>
      ))}
    </div>
  )
}

function AttnSide({ events, head, pair, name, cycle, lastCycle }: {
  events: TraceEvent[]; head?: AttentionHead; pair: PairedHead; name: string
  cycle: number; lastCycle: number
}) {
  if (cycle > lastCycle)
    return <div data-testid="cmp-ended" className="cmp-ended">{name}: run ended at cycle {lastCycle}</div>
  if (head) {
    return (
      <div data-testid="cmp-attn-side" className="cmp-attn-side">
        <span className="cmp-side-name">{name}</span>
        <AttentionHeatmap heads={[head]} tokens={runTokens(events)} />
      </div>
    )
  }
  const thumb = gridThumbFor(events, pair.layer, pair.head)
  return (
    <div data-testid="cmp-attn-side" className="cmp-attn-side">
      <span className="cmp-side-name">{name}</span>
      {thumb ? (
        <div data-testid="cmp-fallback" className="cmp-fallback">
          <Thumb thumb={thumb} />
          <p className="attn-note">run-level thumbnail — full matrix not captured in this run</p>
        </div>
      ) : (
        <p data-testid="cmp-fallback" className="cmp-fallback attn-note">not captured in this run</p>
      )}
    </div>
  )
}

export function CompareView({ a, b }: { a: RunRecord; b: RunRecord }) {
  const aligned = alignRuns(a.events, b.events)
  const [cycle, setCycle] = useState<number | null>(null)
  const [headKey, setHeadKey] = useState<string | null>(null)
  const heads = cycle !== null ? pairedHeads(a.events, b.events, cycle) : []
  const selectedPair = heads.find((h) => `${h.layer}-${h.head}` === headKey) ?? heads[0]

  const stream = (which: 'a' | 'b') => {
    const record = which === 'a' ? a : b
    const prompt = which === 'a' ? aligned.promptA : aligned.promptB
    const chosen = which === 'a' ? aligned.chosenA : aligned.chosenB
    return (
      <div data-testid={`cmp-stream-${which}`} className="cmp-stream">
        <span className="cmp-side-name">{which.toUpperCase()} #{record.meta.seq}</span>
        {prompt.map((t, i) => <span key={`p${i}`} className="cmp-token cmp-token-prompt">{t.text}</span>)}
        <span className="cmp-stream-divider" aria-hidden="true" />
        {chosen.map((t, c) => (
          <span key={`c${c}`} data-testid="cmp-token" data-fork={String(aligned.forkCycle === c)}
            className="cmp-token">{t.text}</span>
        ))}
      </div>
    )
  }

  return (
    <div data-testid="compare-view" className="compare-view">
      <div className="cmp-header">
        <MetaRow label="prompt" a={a.meta.prompt} b={b.meta.prompt} />
        <MetaRow label="T" a={String(a.meta.params.temperature)} b={String(b.meta.params.temperature)} />
        <MetaRow label="top-k" a={String(a.meta.params.topK)} b={String(b.meta.params.topK)} />
        <MetaRow label="max" a={String(a.meta.params.maxNewTokens)} b={String(b.meta.params.maxNewTokens)} />
        <MetaRow label="mode" a={a.meta.mode} b={b.meta.mode} />
        <MetaRow label="model" a={a.meta.modelId ?? '—'} b={b.meta.modelId ?? '—'} />
        <MetaRow label="ended" a={a.meta.reason} b={b.meta.reason} />
      </div>
      {!aligned.samePrompt && (
        <p data-testid="cmp-badge" className="cmp-badge">different prompts — aligned by generation cycle</p>
      )}
      {aligned.samePrompt && aligned.forkCycle === null
        && aligned.chosenA.length === aligned.chosenB.length && (
        <p data-testid="cmp-note" className="cmp-badge">identical outputs</p>
      )}
      {stream('a')}
      {stream('b')}
      <div className="cmp-ruler">
        {Array.from({ length: aligned.maxCycles }, (_, c) => (
          <button key={c} data-testid="cmp-tick" data-selected={String(cycle === c)}
            data-fork={String(aligned.forkCycle === c)} className="cmp-tick"
            onClick={() => { setCycle(c); setHeadKey(null) }}>{c}</button>
        ))}
      </div>
      {cycle !== null && (
        <>
          <h3>cycle {cycle} · distributions</h3>
          <div className="cmp-pair">
            <DistSide events={a.events} cycle={cycle} name="A" lastCycle={aligned.chosenA.length - 1} />
            <DistSide events={b.events} cycle={cycle} name="B" lastCycle={aligned.chosenB.length - 1} />
          </div>
          <h3>cycle {cycle} · attention</h3>
          {heads.length === 0 ? (
            <p className="attn-note">no detected heads recorded at this cycle</p>
          ) : (
            <>
              <div className="head-chip-row">
                {heads.map((h) => (
                  <button key={`${h.layer}-${h.head}`} data-testid="cmp-head-chip"
                    data-active={String(selectedPair === h)} className="head-chip"
                    onClick={() => setHeadKey(`${h.layer}-${h.head}`)}>
                    L{h.layer}·H{h.head}
                    <span className="head-loc">{h.a?.label ?? h.b?.label}</span>
                  </button>
                ))}
              </div>
              {selectedPair && (
                <div className="cmp-pair">
                  <AttnSide events={a.events} head={selectedPair.a} pair={selectedPair} name="A"
                    cycle={cycle} lastCycle={aligned.chosenA.length - 1} />
                  <AttnSide events={b.events} head={selectedPair.b} pair={selectedPair} name="B"
                    cycle={cycle} lastCycle={aligned.chosenB.length - 1} />
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  )
}
```

- [ ] **Step 5: Add the styles**

Append to `src/index.css`:

```css
.compare-view { border: 1px solid var(--line); border-radius: 10px; background: var(--card);
  padding: .8rem 1rem; margin-top: .5rem; }
.cmp-header { display: grid; gap: .1rem; margin-bottom: .6rem; }
.cmp-meta-row { display: grid; grid-template-columns: 5rem 1fr 1fr; gap: .6rem; font-size: .78rem; }
.cmp-meta-row[data-diff="true"] .cmp-meta-value { color: var(--shu); font-weight: 600; }
.cmp-meta-label { color: var(--ink-faint); }
.cmp-badge { font-size: .78rem; color: var(--ink-soft); font-style: italic; margin: .2rem 0; }
.cmp-stream { display: flex; flex-wrap: wrap; align-items: center; gap: .15rem; margin: .25rem 0; }
.cmp-side-name { font-family: var(--mono); font-size: .7rem; color: var(--ai-deep); margin-right: .4rem; }
.cmp-token { border: 1px solid var(--line); border-radius: 5px; background: var(--washi);
  padding: .05rem .3rem; font-size: .8rem; white-space: pre; }
.cmp-token-prompt { color: var(--ink-soft); }
.cmp-token[data-fork="true"] { border-color: var(--shu); background: var(--shu-wash); font-weight: 600; }
.cmp-stream-divider { width: 1px; align-self: stretch; background: var(--line-strong); margin: 0 .3rem; }
.cmp-ruler { display: flex; gap: .2rem; margin: .4rem 0 .6rem; flex-wrap: wrap; }
.cmp-tick { font-family: var(--mono); font-size: .7rem; border: 1px solid var(--line-strong);
  border-radius: 5px; background: var(--card); padding: .1rem .4rem; cursor: pointer; }
.cmp-tick[data-selected="true"] { background: var(--ai); color: var(--washi); border-color: var(--ai); }
.cmp-tick[data-fork="true"] { border-color: var(--shu); }
.cmp-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: start; }
.cmp-dist-side { display: grid; gap: .15rem; }
.cmp-bar-row { display: grid; grid-template-columns: 4.5rem 1fr 2.5rem; align-items: center;
  gap: .4rem; font-size: .76rem; }
.cmp-bar-row[data-chosen="true"] .cmp-bar { background: var(--shu); }
.cmp-bar { display: inline-block; height: .55rem; background: var(--ai-soft); border-radius: 3px; }
.cmp-bar-prob { font-family: var(--mono); font-size: .68rem; color: var(--ink-faint); }
.cmp-ended, .cmp-fallback { font-size: .78rem; color: var(--ink-soft); }
.cmp-fallback .grid-thumb { width: 120px; height: 120px; }
@media (max-width: 720px) { .cmp-pair { grid-template-columns: 1fr; } }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npx vitest run src/viz/compare/CompareView.test.tsx src/viz/AttentionGridExplorer.test.tsx`
Expected: PASS (explorer tests confirm the `Thumb` export changed nothing).

- [ ] **Step 7: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/viz/compare/CompareView.tsx src/viz/compare/CompareView.test.tsx src/viz/AttentionGridExplorer.tsx src/index.css
git commit -m "feat: compare view — metadata diff, aligned streams, paired distributions and attention

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

---

### Task 7: App integration — sealing, shelf wiring, compare mount

**Files:**
- Modify: `src/trace/store.ts` (add `load`)
- Modify: `src/App.tsx`
- Test: `src/trace/store.test.ts` (extend), `src/App.test.tsx` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `useTraceStore` gains `load: (events: TraceEvent[]) => void` (replaces the whole buffer in one set).

- [ ] **Step 1: Write the failing trace-store test**

Append to `src/trace/store.test.ts`:

```ts
test('load replaces the buffer wholesale', () => {
  useTraceStore.getState().clear()
  useTraceStore.getState().append({ type: 'run-end', reason: 'eos' })
  const events = makeFixtureTrace()
  useTraceStore.getState().load(events)
  expect(useTraceStore.getState().events).toEqual(events)
})
```

(extend that file's imports with `import { makeFixtureTrace } from '../test/fixtures'` if absent).

Run: `npx vitest run src/trace/store.test.ts` — expected: FAIL, `load` is not a function.

- [ ] **Step 2: Implement `load`**

In `src/trace/store.ts`, add to the interface `load: (events: TraceEvent[]) => void` and to the store body:

```ts
  load: (events) => set({ events }),
```

Run: `npx vitest run src/trace/store.test.ts` — expected: PASS.

- [ ] **Step 3: Write the failing App tests**

Append to `src/App.test.tsx` (extend imports with `import { useRunsStore } from './app/runsStore'` and `import { _resetArchiveForTests } from './app/runArchive'`; add to the existing `beforeEach` body: `useRunsStore.setState({ records: [], activeId: null, nextSeq: 1, persistFailed: false })` and `_resetArchiveForTests()`):

```tsx
async function generateAndFinish(prompt: string) {
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: prompt } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => {
    expect(useTraceStore.getState().events.at(-1)?.type).toBe('run-end')
  })
}

test('completed runs are sealed onto the shelf', async () => {
  render(<App />)
  await generateAndFinish('one two three')
  await waitFor(() => expect(screen.getAllByTestId('run-chip')).toHaveLength(1))
  expect(screen.getByTestId('run-chip')).toHaveTextContent('#1 · one two three')
  expect(useRunsStore.getState().activeId).not.toBeNull()
})

test('activating an archived run reloads its trace, parked at run-end', async () => {
  render(<App />)
  await generateAndFinish('one two three')
  await generateAndFinish('red green blue')
  await waitFor(() => expect(screen.getAllByTestId('run-chip')).toHaveLength(2))
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  await waitFor(() => {
    const first = useTraceStore.getState().events[0]
    expect(first?.type === 'run-start' && first.prompt).toBe('one two three')
  })
  const { cursor, status } = usePlayerStore.getState()
  expect(cursor).toBe(useTraceStore.getState().events.length - 1)
  expect(status).toBe('paused')
})

test('compare arms from the shelf, opens the view, and exits', async () => {
  render(<App />)
  await generateAndFinish('one two three')
  await generateAndFinish('red green blue')
  await waitFor(() => expect(screen.getAllByTestId('run-chip')).toHaveLength(2))
  fireEvent.click(screen.getByTestId('btn-compare-arm'))
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  expect(screen.getByTestId('compare-view')).toBeInTheDocument()
  expect(screen.getByTestId('cmp-badge')).toHaveTextContent('different prompts')
  expect(screen.queryByTestId('stage-card')).toBeNull()          // player stack hidden
  fireEvent.click(screen.getByTestId('btn-compare-exit'))
  expect(screen.queryByTestId('compare-view')).toBeNull()
  expect(screen.getAllByTestId('stage-card').length).toBeGreaterThan(0)
})
```

Run: `npx vitest run src/App.test.tsx` — expected: FAIL (no shelf, no sealing).

- [ ] **Step 4: Wire `App.tsx`**

Apply these changes to `src/App.tsx`:

a) Extend imports:

```tsx
import { RunShelf } from './app/RunShelf'
import { archiveImport, archiveRemove, archiveSeal, archiveTogglePin, initArchive } from './app/runArchive'
import { parseRunFile, serializeRun } from './app/runFiles'
import { createIndexedDbStorage } from './app/runStorage'
import { useRunsStore } from './app/runsStore'
import { CompareView } from './viz/compare/CompareView'
```

b) Inside the component, after the `attn` state:

```tsx
  const records = useRunsStore((s) => s.records)
  const activeId = useRunsStore((s) => s.activeId)
  const persistFailed = useRunsStore((s) => s.persistFailed)
  const [compare, setCompare] = useState<{ aId: string; bId: string } | null>(null)
  const [compareArmed, setCompareArmed] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
```

c) After the tokenizer-loading effect, add archive init and the pin reset (spec: pins reset whenever the active run changes — covers shelf flips and sealing alike):

```tsx
  useEffect(() => { void initArchive(createIndexedDbStorage()) }, [])
  useEffect(() => { resetPins() }, [activeId, resetPins])
```

d) In `handleGenerate`, exit compare and seal on run-end. After `resetPins()` add:

```tsx
    setCompare(null)
    setCompareArmed(false)
```

and replace the emit callback `(e) => useTraceStore.getState().append(e)` with:

```tsx
      const handle = engine.run(prompt, params, (e) => {
        useTraceStore.getState().append(e)
        if (e.type === 'run-end') {
          // seal metadata comes from the trace itself — run-start carries
          // prompt/params/mode/modelId; the archive is trace-derived by design
          const events = useTraceStore.getState().events
          const start = events.find((x) => x.type === 'run-start')
          if (start && start.type === 'run-start') {
            archiveSeal({
              prompt: start.prompt, params: start.params, mode: start.mode,
              ...(start.mode === 'real' ? { modelId: start.modelId } : {}),
              endedAt: Date.now(), reason: e.reason,
            }, events)
          }
        }
      })
```

e) Add the shelf handlers before the return:

```tsx
  const handleActivate = async (id: string) => {
    runRef.current?.abort()
    await runRef.current?.done
    const record = useRunsStore.getState().records.find((r) => r.id === id)
    if (!record) return
    useRunsStore.getState().setActive(id)
    useTraceStore.getState().load(record.events)
    const dispatch = usePlayerStore.getState().dispatch
    dispatch({ type: 'traceGrew', length: record.events.length })
    dispatch({ type: 'seek', index: record.events.length - 1 })
    dispatch({ type: 'pause' })
  }

  const handleRemove = (id: string) => {
    archiveRemove(id)
    setCompare((c) => (c && (c.aId === id || c.bId === id) ? null : c))
  }

  const handleExport = (id: string) => {
    const record = useRunsStore.getState().records.find((r) => r.id === id)
    if (!record) return
    const { filename, json } = serializeRun(record)
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const handleImportFile = (file: File) => {
    void file.text().then((text) => {
      const parsed = parseRunFile(text)
      if (parsed.ok) { setImportError(null); archiveImport({ meta: parsed.meta, events: parsed.events }) }
      else setImportError(parsed.error)
    })
  }
```

f) Rework the returned JSX below `<PromptBar …/>`:

```tsx
      <RunShelf records={records} activeId={activeId} compare={compare} armed={compareArmed}
        persistFailed={persistFailed} importError={importError}
        onActivate={(id) => { void handleActivate(id) }}
        onSelectCompareB={(id) => {
          setCompareArmed(false)
          setCompare((c) => c ? { ...c, bId: id } : activeId ? { aId: activeId, bId: id } : null)
        }}
        onArmCompare={() => setCompareArmed(true)}
        onExitCompare={() => { setCompare(null); setCompareArmed(false) }}
        onTogglePin={archiveTogglePin} onRemove={handleRemove}
        onExport={handleExport} onImportFile={handleImportFile} />
      {(() => {
        const cmpA = compare && records.find((r) => r.id === compare.aId)
        const cmpB = compare && records.find((r) => r.id === compare.bId)
        if (cmpA && cmpB) return <CompareView a={cmpA} b={cmpB} />
        return (
          <>
            <TokenStream events={events} cursor={cursor} />
            <PipelineBand events={events} cursor={cursor} onStageClick={(index) => {
              usePlayerStore.getState().dispatch({ type: 'seek', index })
              usePlayerStore.getState().dispatch({ type: 'pause' })
            }} />
            <Controls />
            <DetailPanel events={events} cursor={cursor} mode={mode}
              pinnedHeads={pins} onPin={handlePin} pinNote={pinNote} />
          </>
        )
      })()}
```

(The existing `TokenStream`/`PipelineBand`/`Controls`/`DetailPanel` lines move inside this conditional unchanged.)

- [ ] **Step 5: Run the App tests, then the full suite**

Run: `npx vitest run src/App.test.tsx` — expected: PASS (including the three pre-existing App tests).
Run: `npx vitest run` — expected: PASS.
Note: jsdom has no `indexedDB`, so `initArchive` flips `persistFailed` and the shelf note renders in App tests — that is the designed degradation, not a failure. If any test output is not pristine (`rtk proxy npx vitest run src/App.test.tsx`), fix the noise (e.g. mock `URL.createObjectURL` if an export test is ever added; none is here).

- [ ] **Step 6: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/trace/store.ts src/trace/store.test.ts src/App.tsx src/App.test.tsx
git commit -m "feat: archive wiring — seal on run-end, shelf activation, compare mount

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

---

### Task 8: e2e smoke and verification sweep

**Files:**
- Create: `e2e/archive.spec.ts`

**Interfaces:** consumes the running app only (testids from Tasks 4/6/7).

> Recorded ruling (spec deviation, reasoned): the spec's e2e sketch says "fork
> visible", but the sim engine is seeded purely by prompt tokens
> (`SimulatedEngine.ts:38`), so two same-prompt runs are identical and a fork
> cannot be forced deterministically. The e2e therefore uses two different
> prompts and asserts the "different prompts" badge — same flow, deterministic.
> Fork rendering is pinned by CompareView component tests on crafted divergent
> fixtures (Task 6).

- [ ] **Step 1: Write the smoke**

Create `e2e/archive.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

test('runs archive to the shelf, compare opens and exits', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('one two three')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })

  await page.getByTestId('prompt-input').fill('red green blue')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(2, { timeout: 15000 })

  await page.getByTestId('btn-compare-arm').click()
  await page.getByTestId('run-chip-main').first().click()
  await expect(page.getByTestId('compare-view')).toBeVisible()
  await expect(page.getByTestId('cmp-badge')).toContainText('different prompts')

  await page.getByTestId('btn-compare-exit').click()
  await expect(page.getByTestId('compare-view')).not.toBeVisible()
  await expect(page.getByTestId('stage-card').first()).toBeVisible()
})
```

- [ ] **Step 2: Run the e2e suite**

Run: `npx playwright test`
Expected: PASS (both spec files). If the archive smoke fails, fix the regression in app code — never weaken the smoke or the pre-existing spec.

- [ ] **Step 3: Full verification sweep**

Run each; all must pass:

- `npx tsc --noEmit -p tsconfig.app.json` — no errors
- `npx vitest run` — all pass
- `npx playwright test` — all pass
- `npx vite build` — builds cleanly

- [ ] **Step 4: Commit**

```bash
git add e2e/archive.spec.ts
git commit -m "test: e2e smoke for run archive and comparison

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL"
```

If sweep steps required fixes, commit them separately as `fix: verification sweep fallout for the run archive` (with the same trailers).

**Manual browser verification (operator, on the branch or post-merge):** real-mode runs archive across a reload (IndexedDB); export a run, remove it, re-import it (arrives pinned); compare a T=0 vs T=1 pair on the same prompt in real mode and inspect the fork cycle's distributions; confirm generation latency is unchanged with the archive active.
