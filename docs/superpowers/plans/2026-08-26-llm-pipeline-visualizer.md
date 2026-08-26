# LLM Pipeline Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-page educational app that visualizes the LLM generation loop (tokenize → embed → layers → logits → softmax → sample → append) with a simulated engine and a real transformers.js engine, driven by debugger-style player controls.

**Architecture:** Trace-based. Engines emit a normalized append-only trace of typed events; the UI renders as a pure function of `(trace, cursor)`; a player reducer moves the cursor. The real engine runs in a Web Worker whose messages are trace events.

**Tech Stack:** Vite, React 18, TypeScript (strict), Zustand, @huggingface/transformers (v3), Vitest + Testing Library + jsdom, Playwright.

**Spec:** `docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md`

## Global Constraints

- TypeScript `strict: true`; no `any` except at the transformers.js boundary (worker + tokenizer adapter).
- Visualization is SVG + CSS transitions only — no canvas, no D3.
- Trace stores only top-k = 10 candidates at logits/softmax; embeddings stored as downsampled previews (≤ 4 tokens × 16 dims).
- Both modes use the same tokenizer model: `HuggingFaceTB/SmolLM2-135M-Instruct`.
- Real-model inference never runs in CI tests; worker logic is tested via pure functions + mocks, plus a manual verification step.
- The visualization reads only `(trace, cursor)`; engines only append; the player only moves the cursor.
- State: Zustand. Player logic is a pure reducer in a plain function.
- Simulated mode is the default; the app must remain usable with no network (tokenizer falls back to a byte-level fallback tokenizer).
- All test commands: `npx vitest run <file>` (unit), `npx playwright test` (e2e).

## File Structure

```
src/
  trace/types.ts            TraceEvent union, TokenInfo, GenParams
  trace/store.ts            Zustand trace store (append/clear)
  trace/validate.ts         validateTrace(events) → violations[]
  engine/types.ts           PipelineEngine, RunHandle, ProgressInfo
  engine/math.ts            softmax, topK, sampleIndex (shared by both engines)
  engine/prng.ts            mulberry32, seedFromTokens
  engine/tokenizer.ts       Tokenizer interface, HF loader + fallback, FakeTokenizer
  engine/simulated/candidates.ts    candidate-word heuristics
  engine/simulated/SimulatedEngine.ts
  engine/transformers/protocol.ts   WorkerRequest/WorkerResponse types
  engine/transformers/TransformersEngine.ts  main-thread client
  engine/transformers/worker.ts     Web Worker (transformers.js inference)
  player/reducer.ts         PlayerState, PlayerAction, playerReducer
  player/pacing.ts          pacing table + delayFor
  player/store.ts           Zustand player store + usePlaybackTicker
  viz/selectors.ts          visibleTokens, eventAt, latestOfType, cycleTickIndices, activeStage
  viz/TokenStream.tsx
  viz/PipelineBand.tsx
  viz/DetailPanel.tsx       dispatches to details/*
  viz/details/TokenizerDetail.tsx
  viz/details/EmbeddingsDetail.tsx
  viz/details/LayersDetail.tsx
  viz/details/LogitsDetail.tsx
  viz/details/SamplerDetail.tsx
  app/PromptBar.tsx         input, params, mode toggle, Generate
  app/Controls.tsx          player buttons + scrubber
  app/ModelStatus.tsx       download progress, backend chip
  App.tsx                   composition + engine wiring
  main.tsx
  test/fixtures.ts          makeFixtureTrace()
e2e/smoke.spec.ts
```

Each source file gets a colocated `*.test.ts(x)` next to it (except worker.ts and main.tsx).

---

### Task 1: Project scaffolding

**Files:**
- Create: `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/test/setup.ts`, `src/App.test.tsx`, `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: a running Vite app and a working `npx vitest run` pipeline every later task depends on.

- [ ] **Step 1: Scaffold and install**

```bash
npm create vite@latest . -- --template react-ts
npm install zustand @huggingface/transformers
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event @playwright/test
```

(If `npm create vite` refuses a non-empty dir because of `.git`/`docs`, scaffold into `tmp-scaffold/`, move its contents up, delete it.)

- [ ] **Step 2: Configure Vitest**

Replace `vite.config.ts`:

```ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['src/test/setup.ts'],
    exclude: ['e2e/**', 'node_modules/**'],
  },
})
```

Create `src/test/setup.ts`:

```ts
import '@testing-library/jest-dom/vitest'
```

Ensure `tsconfig.json` (or `tsconfig.app.json` in newer templates) has `"strict": true` and add `"types": ["vitest/globals"]` is NOT needed (we import from `vitest` explicitly).

- [ ] **Step 3: Minimal App + sanity test**

Replace `src/App.tsx`:

```tsx
export default function App() {
  return <h1>LLM Pipeline Visualizer</h1>
}
```

Create `src/App.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import App from './App'

test('renders title', () => {
  render(<App />)
  expect(screen.getByText('LLM Pipeline Visualizer')).toBeInTheDocument()
})
```

Delete template leftovers (`src/App.css`, `src/assets`, logo imports in `main.tsx`, contents of `src/index.css` except a minimal reset).

- [ ] **Step 4: Verify**

Run: `npx vitest run` → 1 test passes. Run: `npx tsc --noEmit` (or `npm run build`) → clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold Vite + React + TS + Vitest"
```

---

### Task 2: Trace types and trace store

**Files:**
- Create: `src/trace/types.ts`, `src/trace/store.ts`
- Test: `src/trace/store.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces (used by every later task):

```ts
// types.ts — exact exports
export interface TokenInfo { id: number; text: string }
export interface GenParams { temperature: number; topK: number; maxNewTokens: number }
export type Mode = 'sim' | 'real'
export type RunEndReason = 'eos' | 'max-tokens' | 'aborted' | 'error'
export type TraceEvent =
  | { type: 'run-start'; prompt: string; mode: Mode; modelId: string; params: GenParams }
  | { type: 'tokenize'; tokens: TokenInfo[]; truncated?: boolean }  // truncated: prompt cut to context window
  | { type: 'embed'; cycle: number; seqLen: number; dims: number; preview: number[][] }
  | { type: 'layer'; cycle: number; index: number; total: number; activationNorm?: number }
  | { type: 'logits'; cycle: number; topK: Array<TokenInfo & { logit: number }> }
  | { type: 'softmax'; cycle: number; temperature: number; topK: Array<TokenInfo & { prob: number }> }
  | { type: 'sample'; cycle: number; chosen: TokenInfo; method: 'greedy' | 'top-k' }
  | { type: 'append'; cycle: number; token: TokenInfo }
  | { type: 'run-end'; reason: RunEndReason; message?: string }
// store.ts
export const useTraceStore: UseBoundStore<StoreApi<{ events: TraceEvent[]; append: (e: TraceEvent) => void; clear: () => void }>>
```

- [ ] **Step 1: Write failing store test** (`src/trace/store.test.ts`)

```ts
import { beforeEach, expect, test } from 'vitest'
import { useTraceStore } from './store'

beforeEach(() => useTraceStore.getState().clear())

test('append adds events in order', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [{ id: 1, text: 'Hi' }] })
  useTraceStore.getState().append({ type: 'append', cycle: 0, token: { id: 2, text: ' there' } })
  const events = useTraceStore.getState().events
  expect(events).toHaveLength(2)
  expect(events[0].type).toBe('tokenize')
})

test('clear empties the trace', () => {
  useTraceStore.getState().append({ type: 'run-end', reason: 'max-tokens' })
  useTraceStore.getState().clear()
  expect(useTraceStore.getState().events).toEqual([])
})
```

- [ ] **Step 2: Run to verify failure** — `npx vitest run src/trace/store.test.ts` → FAIL (module not found).

- [ ] **Step 3: Implement** — `src/trace/types.ts` exactly as in Produces above; `src/trace/store.ts`:

```ts
import { create } from 'zustand'
import type { TraceEvent } from './types'

interface TraceState {
  events: TraceEvent[]
  append: (e: TraceEvent) => void
  clear: () => void
}

export const useTraceStore = create<TraceState>()((set) => ({
  events: [],
  append: (e) => set((s) => ({ events: [...s.events, e] })),
  clear: () => set({ events: [] }),
}))
```

- [ ] **Step 4: Verify pass** — `npx vitest run src/trace/store.test.ts` → PASS.

- [ ] **Step 5: Commit** — `git add src/trace && git commit -m "feat: trace event types and store"`

---

### Task 3: Shared engine math

**Files:**
- Create: `src/engine/math.ts`
- Test: `src/engine/math.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:

```ts
export function softmax(logits: number[], temperature: number): number[]  // temperature 0 → one-hot argmax
export function topK(data: ArrayLike<number>, k: number): Array<{ id: number; logit: number }>  // sorted desc by logit
export function sampleIndex(probs: number[], rand: () => number): number
```

- [ ] **Step 1: Failing tests** (`src/engine/math.test.ts`)

```ts
import { expect, test } from 'vitest'
import { sampleIndex, softmax, topK } from './math'

test('softmax sums to 1 and preserves order', () => {
  const p = softmax([3, 1, 0.5], 1)
  expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  expect(p[0]).toBeGreaterThan(p[1])
})

test('temperature 0 is one-hot argmax', () => {
  expect(softmax([1, 5, 2], 0)).toEqual([0, 1, 0])
})

test('low temperature sharpens the distribution', () => {
  const sharp = softmax([3, 1], 0.5)
  const soft = softmax([3, 1], 2)
  expect(sharp[0]).toBeGreaterThan(soft[0])
})

test('topK returns k best ids sorted desc', () => {
  expect(topK([0.1, 9, 3, 7], 2)).toEqual([
    { id: 1, logit: 9 },
    { id: 3, logit: 7 },
  ])
})

test('sampleIndex picks by cumulative probability', () => {
  expect(sampleIndex([0.2, 0.5, 0.3], () => 0.1)).toBe(0)
  expect(sampleIndex([0.2, 0.5, 0.3], () => 0.6)).toBe(1)
  expect(sampleIndex([0.2, 0.5, 0.3], () => 0.99)).toBe(2)
})
```

- [ ] **Step 2: Verify failure** — `npx vitest run src/engine/math.test.ts` → FAIL.

- [ ] **Step 3: Implement** (`src/engine/math.ts`)

```ts
export function softmax(logits: number[], temperature: number): number[] {
  if (temperature === 0) {
    const best = logits.indexOf(Math.max(...logits))
    return logits.map((_, i) => (i === best ? 1 : 0))
  }
  const scaled = logits.map((l) => l / temperature)
  const max = Math.max(...scaled)
  const exps = scaled.map((l) => Math.exp(l - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

export function topK(data: ArrayLike<number>, k: number): Array<{ id: number; logit: number }> {
  const entries: Array<{ id: number; logit: number }> = []
  for (let i = 0; i < data.length; i++) entries.push({ id: i, logit: data[i] })
  return entries.sort((a, b) => b.logit - a.logit).slice(0, k)
}

export function sampleIndex(probs: number[], rand: () => number): number {
  const r = rand()
  let acc = 0
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]
    if (r < acc) return i
  }
  return probs.length - 1
}
```

- [ ] **Step 4: Verify pass**, `npx tsc --noEmit` clean.

- [ ] **Step 5: Commit** — `git add src/engine && git commit -m "feat: shared softmax/topK/sampling math"`

---

### Task 4: Seeded PRNG

**Files:**
- Create: `src/engine/prng.ts`
- Test: `src/engine/prng.test.ts`

**Interfaces:**
- Produces:

```ts
export function mulberry32(seed: number): () => number   // deterministic, returns [0,1)
export function seedFromTokens(ids: number[]): number    // FNV-1a over ids
```

- [ ] **Step 1: Failing tests** (`src/engine/prng.test.ts`)

```ts
import { expect, test } from 'vitest'
import { mulberry32, seedFromTokens } from './prng'

test('same seed → same sequence', () => {
  const a = mulberry32(42)
  const b = mulberry32(42)
  expect([a(), a(), a()]).toEqual([b(), b(), b()])
})

test('different seeds → different sequences', () => {
  expect(mulberry32(1)()).not.toBe(mulberry32(2)())
})

test('values in [0,1)', () => {
  const r = mulberry32(7)
  for (let i = 0; i < 100; i++) {
    const v = r()
    expect(v).toBeGreaterThanOrEqual(0)
    expect(v).toBeLessThan(1)
  }
})

test('seedFromTokens is deterministic and order-sensitive', () => {
  expect(seedFromTokens([1, 2, 3])).toBe(seedFromTokens([1, 2, 3]))
  expect(seedFromTokens([1, 2, 3])).not.toBe(seedFromTokens([3, 2, 1]))
})
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement** (`src/engine/prng.ts`)

```ts
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function seedFromTokens(ids: number[]): number {
  let h = 0x811c9dc5
  for (const id of ids) {
    h ^= id
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}
```

- [ ] **Step 4: Verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat: seeded PRNG and token-derived seeds"`

---

### Task 5: Player reducer

**Files:**
- Create: `src/player/reducer.ts`
- Test: `src/player/reducer.test.ts`

**Interfaces:**
- Produces:

```ts
export interface PlayerState {
  cursor: number            // -1 = before start
  status: 'idle' | 'playing' | 'paused'
  speed: number             // 0.5 | 1 | 2 | 4
  followLive: boolean
  traceLength: number
}
export type PlayerAction =
  | { type: 'play' } | { type: 'pause' }
  | { type: 'stepForward'; auto?: boolean }   // auto = ticker-driven
  | { type: 'stepBack' }
  | { type: 'seek'; index: number }
  | { type: 'setSpeed'; speed: number }
  | { type: 'traceGrew'; length: number }
  | { type: 'goLive' }
  | { type: 'reset' }                          // new run: cursor -1, playing, followLive true
export const initialPlayerState: PlayerState
export function playerReducer(s: PlayerState, a: PlayerAction): PlayerState
```

Semantics (encode each as a test): manual `stepForward` pauses and drops followLive; `auto: true` keeps playing and keeps followLive; cursor clamps to `[0, traceLength-1]` (stepBack floor 0, forward ceiling frontier); `pause`/`stepBack`/`seek` drop followLive; `traceGrew` only updates `traceLength`; `goLive` jumps to frontier, sets followLive, plays; `reset` keeps speed, sets `cursor: -1, status: 'playing', followLive: true, traceLength: 0`.

- [ ] **Step 1: Failing tests** (`src/player/reducer.test.ts`)

```ts
import { expect, test } from 'vitest'
import { initialPlayerState as init, playerReducer as r } from './reducer'

const grown = r(init, { type: 'traceGrew', length: 5 })

test('initial state', () => {
  expect(init).toEqual({ cursor: -1, status: 'idle', speed: 1, followLive: true, traceLength: 0 })
})

test('play/pause toggle status; pause drops followLive', () => {
  const playing = r(grown, { type: 'play' })
  expect(playing.status).toBe('playing')
  const paused = r(playing, { type: 'pause' })
  expect(paused.status).toBe('paused')
  expect(paused.followLive).toBe(false)
})

test('manual step pauses and drops followLive; auto step does not', () => {
  const playing = r(grown, { type: 'play' })
  const manual = r(playing, { type: 'stepForward' })
  expect(manual).toMatchObject({ cursor: 0, status: 'paused', followLive: false })
  const auto = r(playing, { type: 'stepForward', auto: true })
  expect(auto).toMatchObject({ cursor: 0, status: 'playing', followLive: true })
})

test('cursor clamps at frontier and at 0', () => {
  let s = { ...grown, cursor: 4 }
  expect(r(s, { type: 'stepForward' }).cursor).toBe(4)
  s = { ...grown, cursor: 0 }
  expect(r(s, { type: 'stepBack' }).cursor).toBe(0)
})

test('seek clamps and drops followLive', () => {
  expect(r(grown, { type: 'seek', index: 99 })).toMatchObject({ cursor: 4, followLive: false })
  expect(r(grown, { type: 'seek', index: -3 }).cursor).toBe(0)
})

test('traceGrew only updates length', () => {
  const s = r({ ...grown, cursor: 2, followLive: false }, { type: 'traceGrew', length: 9 })
  expect(s).toMatchObject({ cursor: 2, traceLength: 9, followLive: false })
})

test('goLive jumps to frontier, follows, plays', () => {
  const s = r({ ...grown, cursor: 1, followLive: false, status: 'paused' as const }, { type: 'goLive' })
  expect(s).toMatchObject({ cursor: 4, followLive: true, status: 'playing' })
})

test('reset keeps speed, starts playing from -1', () => {
  const fast = r(grown, { type: 'setSpeed', speed: 4 })
  expect(r(fast, { type: 'reset' })).toEqual({ cursor: -1, status: 'playing', speed: 4, followLive: true, traceLength: 0 })
})
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement** (`src/player/reducer.ts`)

```ts
export interface PlayerState {
  cursor: number
  status: 'idle' | 'playing' | 'paused'
  speed: number
  followLive: boolean
  traceLength: number
}

export type PlayerAction =
  | { type: 'play' } | { type: 'pause' }
  | { type: 'stepForward'; auto?: boolean }
  | { type: 'stepBack' }
  | { type: 'seek'; index: number }
  | { type: 'setSpeed'; speed: number }
  | { type: 'traceGrew'; length: number }
  | { type: 'goLive' }
  | { type: 'reset' }

export const initialPlayerState: PlayerState = {
  cursor: -1, status: 'idle', speed: 1, followLive: true, traceLength: 0,
}

const clamp = (i: number, len: number) => Math.max(0, Math.min(i, len - 1))

export function playerReducer(s: PlayerState, a: PlayerAction): PlayerState {
  switch (a.type) {
    case 'play': return { ...s, status: 'playing' }
    case 'pause': return { ...s, status: 'paused', followLive: false }
    case 'stepForward': {
      const cursor = clamp(s.cursor + 1, s.traceLength)
      return a.auto ? { ...s, cursor } : { ...s, cursor, status: 'paused', followLive: false }
    }
    case 'stepBack': return { ...s, cursor: clamp(s.cursor - 1, s.traceLength), status: 'paused', followLive: false }
    case 'seek': return { ...s, cursor: clamp(a.index, s.traceLength), followLive: false }
    case 'setSpeed': return { ...s, speed: a.speed }
    case 'traceGrew': return { ...s, traceLength: a.length }
    case 'goLive': return { ...s, cursor: s.traceLength - 1, followLive: true, status: 'playing' }
    case 'reset': return { ...initialPlayerState, speed: s.speed, status: 'playing' }
  }
}
```

- [ ] **Step 4: Verify pass.**

- [ ] **Step 5: Commit** — `git add src/player && git commit -m "feat: player reducer"`

---

### Task 6: Pacing, player store, playback ticker

**Files:**
- Create: `src/player/pacing.ts`, `src/player/store.ts`
- Test: `src/player/pacing.test.ts`, `src/player/store.test.tsx`

**Interfaces:**
- Consumes: `playerReducer`, `useTraceStore`
- Produces:

```ts
// pacing.ts
export const BASE_MS = 600
export function delayFor(e: TraceEvent | undefined, speed: number): number
// store.ts
export const usePlayerStore   // PlayerState & { dispatch(a: PlayerAction): void }
export function usePlaybackTicker(): void   // hook mounted once in App
```

- [ ] **Step 1: Failing pacing tests** (`src/player/pacing.test.ts`)

```ts
import { expect, test } from 'vitest'
import { BASE_MS, delayFor } from './pacing'

test('layer events are much quicker than sample events', () => {
  const layer = delayFor({ type: 'layer', cycle: 0, index: 0, total: 12 }, 1)
  const sample = delayFor({ type: 'sample', cycle: 0, chosen: { id: 1, text: 'a' }, method: 'top-k' }, 1)
  expect(sample).toBeGreaterThan(layer * 4)
})

test('speed divides delay', () => {
  const e = { type: 'tokenize', tokens: [] } as const
  expect(delayFor(e, 2)).toBeCloseTo(delayFor(e, 1) / 2)
})

test('undefined event gets base delay', () => {
  expect(delayFor(undefined, 1)).toBe(BASE_MS)
})
```

- [ ] **Step 2: Verify failure; implement** (`src/player/pacing.ts`)

```ts
import type { TraceEvent } from '../trace/types'

export const BASE_MS = 600

const MULTIPLIER: Record<TraceEvent['type'], number> = {
  'run-start': 0.5, tokenize: 1.5, embed: 1.5, layer: 0.2,
  logits: 1.5, softmax: 1.5, sample: 2.5, append: 1.5, 'run-end': 0.5,
}

export function delayFor(e: TraceEvent | undefined, speed: number): number {
  const mult = e ? MULTIPLIER[e.type] : 1
  return (BASE_MS * mult) / speed
}
```

Run pacing tests → PASS.

- [ ] **Step 3: Failing ticker test** (`src/player/store.test.tsx`)

```tsx
import { renderHook } from '@testing-library/react'
import { act, beforeEach, expect, test, vi } from 'vitest'
import { useTraceStore } from '../trace/store'
import { usePlaybackTicker, usePlayerStore } from './store'
import { initialPlayerState } from './reducer'

beforeEach(() => {
  vi.useFakeTimers()
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
})

test('while playing, cursor advances one event per delay and parks at frontier', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [] })
  useTraceStore.getState().append({ type: 'run-end', reason: 'max-tokens' })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 2 })
  usePlayerStore.getState().dispatch({ type: 'play' })

  renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(5000))
  expect(usePlayerStore.getState().cursor).toBe(1)  // parked at frontier, still 'playing'
  expect(usePlayerStore.getState().status).toBe('playing')
})

test('paused: cursor does not move', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [] })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 1 })
  renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(5000))
  expect(usePlayerStore.getState().cursor).toBe(-1)
})
```

- [ ] **Step 4: Verify failure; implement** (`src/player/store.ts`)

```ts
import { useEffect } from 'react'
import { create } from 'zustand'
import { useTraceStore } from '../trace/store'
import { delayFor } from './pacing'
import { initialPlayerState, playerReducer, type PlayerAction, type PlayerState } from './reducer'

interface PlayerStore extends PlayerState {
  dispatch: (a: PlayerAction) => void
}

export const usePlayerStore = create<PlayerStore>()((set) => ({
  ...initialPlayerState,
  dispatch: (a) => set((s) => playerReducer(s, a)),
}))

export function usePlaybackTicker(): void {
  const status = usePlayerStore((s) => s.status)
  const speed = usePlayerStore((s) => s.speed)
  const cursor = usePlayerStore((s) => s.cursor)
  const length = useTraceStore((s) => s.events.length)

  useEffect(() => {
    if (status !== 'playing' || length === 0 || cursor >= length - 1) return
    const next = useTraceStore.getState().events[cursor + 1]
    const t = setTimeout(
      () => usePlayerStore.getState().dispatch({ type: 'stepForward', auto: true }),
      delayFor(next, speed),
    )
    return () => clearTimeout(t)
  }, [status, speed, cursor, length])
}
```

Note the parked-frontier behavior: when `cursor === length - 1` the effect schedules nothing, but a `length` change re-runs it — this is how live real-mode generation resumes playback automatically.

- [ ] **Step 5: Verify all player tests pass; commit** — `git add src/player && git commit -m "feat: pacing table and playback ticker"`

---

### Task 7: Engine interface, trace validation, fixtures

**Files:**
- Create: `src/engine/types.ts`, `src/trace/validate.ts`, `src/test/fixtures.ts`
- Test: `src/trace/validate.test.ts`

**Interfaces:**
- Produces:

```ts
// engine/types.ts
export interface ProgressInfo { file: string; loaded: number; total: number }
export interface RunHandle { abort(): void; done: Promise<void> }
export interface PipelineEngine {
  prepare(onProgress?: (p: ProgressInfo) => void): Promise<void>
  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void): RunHandle
}
// trace/validate.ts
export function validateTrace(events: TraceEvent[]): string[]  // [] = valid
// test/fixtures.ts
export function makeFixtureTrace(cycles?: number, layers?: number): TraceEvent[]  // valid 2-cycle default trace
```

`validateTrace` checks: first event `run-start`; second `tokenize`; per cycle the exact order `embed`, `layer × total` with ascending `index`, `logits`, `softmax`, `sample`, `append`; softmax `topK` probs sum to 1 ± 1e-4 and sorted descending; logits `topK` sorted descending; last event `run-end`.

- [ ] **Step 1: Failing tests** (`src/trace/validate.test.ts`)

```ts
import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { validateTrace } from './validate'

test('fixture trace is valid', () => {
  expect(validateTrace(makeFixtureTrace())).toEqual([])
})

test('missing run-start is flagged', () => {
  expect(validateTrace(makeFixtureTrace().slice(1)).length).toBeGreaterThan(0)
})

test('out-of-order layer indices are flagged', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((e) => e.type === 'layer')
  const j = t.findIndex((e, k) => e.type === 'layer' && k > i)
  ;[t[i], t[j]] = [t[j], t[i]]
  expect(validateTrace(t).some((v) => v.includes('layer'))).toBe(true)
})

test('softmax probs must sum to 1', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'softmax')
  if (e?.type === 'softmax') e.topK = e.topK.map((c) => ({ ...c, prob: c.prob * 2 }))
  expect(validateTrace(t).some((v) => v.includes('softmax'))).toBe(true)
})
```

- [ ] **Step 2: Verify failure; implement fixtures** (`src/test/fixtures.ts`)

```ts
import type { TraceEvent, TokenInfo } from '../trace/types'

export function makeFixtureTrace(cycles = 2, layers = 3): TraceEvent[] {
  const events: TraceEvent[] = [
    { type: 'run-start', prompt: 'The cat', mode: 'sim', modelId: 'fixture',
      params: { temperature: 0.8, topK: 10, maxNewTokens: cycles } },
    { type: 'tokenize', tokens: [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }] },
  ]
  const words = [' sat', ' on', ' the', ' mat']
  for (let c = 0; c < cycles; c++) {
    const chosen: TokenInfo = { id: 100 + c, text: words[c % words.length] }
    events.push({ type: 'embed', cycle: c, seqLen: 2 + c, dims: 576,
      preview: [[0.1, -0.2, 0.3], [0.0, 0.5, -0.1]] })
    for (let l = 0; l < layers; l++) events.push({ type: 'layer', cycle: c, index: l, total: layers })
    events.push({ type: 'logits', cycle: c, topK: [
      { ...chosen, logit: 9.1 }, { id: 200, text: ' ran', logit: 7.2 }, { id: 201, text: ' was', logit: 5.0 },
    ] })
    events.push({ type: 'softmax', cycle: c, temperature: 0.8, topK: [
      { ...chosen, prob: 0.7 }, { id: 200, text: ' ran', prob: 0.2 }, { id: 201, text: ' was', prob: 0.1 },
    ] })
    events.push({ type: 'sample', cycle: c, chosen, method: 'top-k' })
    events.push({ type: 'append', cycle: c, token: chosen })
  }
  events.push({ type: 'run-end', reason: 'max-tokens' })
  return events
}
```

- [ ] **Step 3: Implement validator** (`src/trace/validate.ts`)

```ts
import type { TraceEvent } from './types'

export function validateTrace(events: TraceEvent[]): string[] {
  const errs: string[] = []
  if (events[0]?.type !== 'run-start') errs.push('first event must be run-start')
  if (events[1]?.type !== 'tokenize') errs.push('second event must be tokenize')
  if (events[events.length - 1]?.type !== 'run-end') errs.push('last event must be run-end')

  const CYCLE = ['embed', 'layer', 'logits', 'softmax', 'sample', 'append'] as const
  let phase: (typeof CYCLE)[number] = 'embed'
  let layerIdx = 0
  for (const e of events.slice(2, -1)) {
    if (e.type === 'layer') {
      if (phase !== 'layer' && phase !== 'logits') { errs.push(`unexpected layer in phase ${phase}`); continue }
      if (e.index !== layerIdx) errs.push(`layer index ${e.index}, expected ${layerIdx}`)
      layerIdx++
      phase = layerIdx >= e.total ? 'logits' : 'layer'
      continue
    }
    if (e.type === 'embed' && phase === 'embed') { phase = 'layer'; layerIdx = 0; continue }
    if (e.type === phase) {
      phase = CYCLE[(CYCLE.indexOf(phase) + 1) % CYCLE.length]
      continue
    }
    errs.push(`unexpected ${e.type} in phase ${phase}`)
  }

  for (const e of events) {
    if (e.type === 'softmax') {
      const sum = e.topK.reduce((a, c) => a + c.prob, 0)
      if (Math.abs(sum - 1) > 1e-4) errs.push(`softmax probs sum to ${sum}`)
      for (let i = 1; i < e.topK.length; i++)
        if (e.topK[i].prob > e.topK[i - 1].prob) errs.push('softmax topK not sorted desc')
    }
    if (e.type === 'logits')
      for (let i = 1; i < e.topK.length; i++)
        if (e.topK[i].logit > e.topK[i - 1].logit) errs.push('logits topK not sorted desc')
  }
  return errs
}
```

Also create `src/engine/types.ts` exactly as in Produces (pure types, no test needed).

- [ ] **Step 4: Verify pass; `npx tsc --noEmit` clean.**

- [ ] **Step 5: Commit** — `git add src/engine/types.ts src/trace/validate.ts src/test && git commit -m "feat: engine interface, trace validator, fixtures"`

---

### Task 8: Tokenizer adapter with fallback

**Files:**
- Create: `src/engine/tokenizer.ts`
- Test: `src/engine/tokenizer.test.ts`

**Interfaces:**
- Produces:

```ts
export interface Tokenizer {
  encode(text: string): TokenInfo[]
  eosTokenId: number
}
export const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
export async function loadTokenizer(modelId?: string): Promise<Tokenizer>  // HF, falls back on failure
export function fallbackTokenizer(): Tokenizer   // offline word/punct splitter, exported for tests + fallback
export function fakeTokenizer(): Tokenizer       // alias of fallbackTokenizer for test readability
```

The fallback splits on `/\s*\S+/g` word chunks (each keeps its leading space, GPT-style) and derives stable ids by FNV-1a hash of the chunk text modulo 50000. It is used when the HF hub is unreachable **and** as the deterministic tokenizer in all unit tests (the HF path is covered by manual verification — no network in CI).

- [ ] **Step 1: Failing tests** (`src/engine/tokenizer.test.ts`)

```ts
import { expect, test, vi } from 'vitest'
import { fallbackTokenizer, loadTokenizer } from './tokenizer'

// hoisted by vitest — must be top-level, affects the whole file
vi.mock('@huggingface/transformers', () => { throw new Error('offline') })

test('fallback splits words keeping leading spaces', () => {
  const toks = fallbackTokenizer().encode('The cat sat.')
  expect(toks.map((t) => t.text)).toEqual(['The', ' cat', ' sat.'])
})

test('fallback ids are stable', () => {
  const t = fallbackTokenizer()
  expect(t.encode('cat')[0].id).toBe(t.encode('cat')[0].id)
})

test('loadTokenizer falls back when HF import fails', async () => {
  const tok = await loadTokenizer()
  expect(tok.encode('hi')[0].text).toBe('hi')
})
```

- [ ] **Step 2: Verify failure; implement** (`src/engine/tokenizer.ts`)

```ts
import type { TokenInfo } from '../trace/types'

export interface Tokenizer {
  encode(text: string): TokenInfo[]
  eosTokenId: number
}

export const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'

function hashId(text: string): number {
  let h = 0x811c9dc5
  for (const ch of text) {
    h ^= ch.codePointAt(0) ?? 0
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0) % 50000
}

export function fallbackTokenizer(): Tokenizer {
  return {
    eosTokenId: 0,
    encode: (text) => (text.match(/\s*\S+/g) ?? []).map((chunk) => ({ id: hashId(chunk), text: chunk })),
  }
}

export const fakeTokenizer = fallbackTokenizer

export async function loadTokenizer(modelId: string = MODEL_ID): Promise<Tokenizer> {
  try {
    const { AutoTokenizer } = await import('@huggingface/transformers')
    const tok = await AutoTokenizer.from_pretrained(modelId)
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const eos: number = (tok as any).model?.eos_token_id ?? (tok as any).eos_token_id ?? 0
    return {
      eosTokenId: eos,
      encode: (text) => {
        const ids: number[] = tok.encode(text, { add_special_tokens: false } as any)
        return ids.map((id) => ({ id, text: tok.decode([id]) }))
      },
    }
  } catch {
    return fallbackTokenizer()
  }
}
```

(The exact `eos_token_id` location must be confirmed against the installed `@huggingface/transformers` version during the manual verification in Task 17 — the fallback `?? 0` keeps it safe.)

- [ ] **Step 3: Verify pass.**

- [ ] **Step 4: Commit** — `git commit -am "feat: tokenizer adapter with offline fallback"`

---

### Task 9: Simulated candidate heuristics

**Files:**
- Create: `src/engine/simulated/candidates.ts`
- Test: `src/engine/simulated/candidates.test.ts`

**Interfaces:**
- Consumes: `mulberry32` style `rand: () => number`
- Produces:

```ts
export function candidateWords(prevText: string, rand: () => number): string[]
// exactly 10 distinct strings; words carry a leading space; punctuation does not;
// capitalized after sentence end; ',' early in a sentence, '.' late (>40 chars since last '.')
```

- [ ] **Step 1: Failing tests** (`src/engine/simulated/candidates.test.ts`)

```ts
import { expect, test } from 'vitest'
import { mulberry32 } from '../prng'
import { candidateWords } from './candidates'

test('returns 10 distinct candidates', () => {
  const c = candidateWords('The cat', mulberry32(1))
  expect(c).toHaveLength(10)
  expect(new Set(c).size).toBe(10)
})

test('words have leading space, punctuation does not', () => {
  const c = candidateWords('The cat', mulberry32(1))
  for (const w of c) expect(w.startsWith(' ') || w === ',' || w === '.').toBe(true)
})

test('capitalizes after sentence end', () => {
  const c = candidateWords('It was late.', mulberry32(2))
  const words = c.filter((w) => w.startsWith(' '))
  expect(words.every((w) => /^[A-Z]/.test(w.trimStart()))).toBe(true)
})

test('offers a period late in a long sentence', () => {
  const long = 'the quick brown fox jumps over the lazy dog again and again'
  expect(candidateWords(long, mulberry32(3))).toContain('.')
  expect(candidateWords('Hi', mulberry32(3))).toContain(',')
})

test('deterministic for same rand seed', () => {
  expect(candidateWords('a b c', mulberry32(9))).toEqual(candidateWords('a b c', mulberry32(9)))
})
```

- [ ] **Step 2: Verify failure; implement** (`src/engine/simulated/candidates.ts`)

```ts
const COMMON = [
  'the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'he', 'for', 'it',
  'with', 'as', 'his', 'on', 'be', 'at', 'by', 'had', 'not', 'are', 'but',
  'from', 'or', 'have', 'an', 'they', 'which', 'one', 'you', 'were', 'her',
  'all', 'she', 'there', 'would', 'their', 'we', 'him', 'been', 'has', 'when',
  'who', 'will', 'more', 'no', 'if', 'out', 'so',
]

export function candidateWords(prevText: string, rand: () => number): string[] {
  const sentenceStart = /[.!?]\s*$/.test(prevText) || prevText.trim() === ''
  const sinceEnd = prevText.length - Math.max(
    prevText.lastIndexOf('.'), prevText.lastIndexOf('!'), prevText.lastIndexOf('?'))

  const pool = [...COMMON]
  const picks: string[] = []
  while (picks.length < 9 && pool.length > 0) {
    const w = pool.splice(Math.floor(rand() * pool.length), 1)[0]
    picks.push(' ' + (sentenceStart ? w[0].toUpperCase() + w.slice(1) : w))
  }
  picks.push(sinceEnd > 40 ? '.' : ',')
  return picks
}
```

- [ ] **Step 3: Verify pass.**

- [ ] **Step 4: Commit** — `git add src/engine/simulated && git commit -m "feat: simulated candidate heuristics"`

---

### Task 10: SimulatedEngine

**Files:**
- Create: `src/engine/simulated/SimulatedEngine.ts`
- Test: `src/engine/simulated/SimulatedEngine.test.ts`

**Interfaces:**
- Consumes: `Tokenizer`, `candidateWords`, `mulberry32`, `seedFromTokens`, `softmax`, `sampleIndex`, `validateTrace`
- Produces:

```ts
export class SimulatedEngine implements PipelineEngine {
  constructor(tokenizer: Tokenizer, opts?: { layers?: number; dims?: number })  // defaults 12 / 576
  prepare(): Promise<void>            // no-op (tokenizer injected already loaded)
  run(prompt, params, emit): RunHandle
}
```

Behavior: seeded by `seedFromTokens(promptIds)` — same prompt + params → identical trace. Per cycle emits `embed` (preview: last ≤4 tokens × 16 dims in [-1, 1]), `layer × layers` with `activationNorm = 1 + 0.05 * index + 0.3 * rand()`, `logits` from `candidateWords` with descending values `10 - i * 1.1 + rand() * 0.6` **re-sorted descending after noise**, `softmax` via shared `softmax()` over the 10 candidate logits, `sample` (`greedy` iff `temperature === 0`, else `top-k`), `append`. Terminates with `run-end`: `'eos'` when `rand() < 0.03 * cycle` after a chosen `'.'`, `'max-tokens'` at the cap, `'aborted'` when aborted. The loop `await`s a `setTimeout(0)` between cycles so `abort()` takes effect at cycle boundaries.

- [ ] **Step 1: Failing tests** (`src/engine/simulated/SimulatedEngine.test.ts`)

```ts
import { expect, test } from 'vitest'
import type { TraceEvent } from '../../trace/types'
import { validateTrace } from '../../trace/validate'
import { fakeTokenizer } from '../tokenizer'
import { SimulatedEngine } from './SimulatedEngine'

const PARAMS = { temperature: 0.8, topK: 10, maxNewTokens: 4 }

async function collect(prompt: string, params = PARAMS): Promise<TraceEvent[]> {
  const engine = new SimulatedEngine(fakeTokenizer(), { layers: 3 })
  const events: TraceEvent[] = []
  await engine.run(prompt, params, (e) => events.push(e)).done
  return events
}

test('produces a valid trace', async () => {
  expect(validateTrace(await collect('The cat sat'))).toEqual([])
})

test('same prompt → identical trace (deterministic)', async () => {
  expect(await collect('Hello world')).toEqual(await collect('Hello world'))
})

test('different prompts → different traces', async () => {
  expect(JSON.stringify(await collect('aaa'))).not.toBe(JSON.stringify(await collect('bbb')))
})

test('temperature 0 uses greedy and picks the top candidate', async () => {
  const events = await collect('The cat', { ...PARAMS, temperature: 0 })
  const sample = events.find((e) => e.type === 'sample')
  const sm = events.find((e) => e.type === 'softmax')
  if (sample?.type !== 'sample' || sm?.type !== 'softmax') throw new Error('missing events')
  expect(sample.method).toBe('greedy')
  expect(sample.chosen.id).toBe(sm.topK[0].id)
})

test('respects maxNewTokens', async () => {
  const events = await collect('Hi', { ...PARAMS, maxNewTokens: 2 })
  expect(events.filter((e) => e.type === 'append').length).toBeLessThanOrEqual(2)
})

test('abort emits run-end aborted', async () => {
  const engine = new SimulatedEngine(fakeTokenizer(), { layers: 3 })
  const events: TraceEvent[] = []
  const handle = engine.run('Hi', { ...PARAMS, maxNewTokens: 50 }, (e) => events.push(e))
  handle.abort()
  await handle.done
  const last = events[events.length - 1]
  expect(last.type === 'run-end' && last.reason === 'aborted').toBe(true)
})

test('embed preview is capped at 4 tokens × 16 dims', async () => {
  const events = await collect('one two three four five six')
  const embed = events.find((e) => e.type === 'embed')
  if (embed?.type !== 'embed') throw new Error('missing embed')
  expect(embed.preview.length).toBeLessThanOrEqual(4)
  expect(embed.preview[0]).toHaveLength(16)
})
```

- [ ] **Step 2: Verify failure; implement** (`src/engine/simulated/SimulatedEngine.ts`)

```ts
import type { GenParams, TokenInfo, TraceEvent } from '../../trace/types'
import { sampleIndex, softmax } from '../math'
import { mulberry32, seedFromTokens } from '../prng'
import type { PipelineEngine, RunHandle } from '../types'
import type { Tokenizer } from '../tokenizer'
import { MODEL_ID } from '../tokenizer'
import { candidateWords } from './candidates'

export class SimulatedEngine implements PipelineEngine {
  private layers: number
  private dims: number

  constructor(private tokenizer: Tokenizer, opts?: { layers?: number; dims?: number }) {
    this.layers = opts?.layers ?? 12
    this.dims = opts?.dims ?? 576
  }

  async prepare(): Promise<void> {}

  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void): RunHandle {
    let aborted = false
    const done = this.loop(prompt, params, emit, () => aborted)
    return { abort: () => { aborted = true }, done }
  }

  private async loop(
    prompt: string, params: GenParams,
    emit: (e: TraceEvent) => void, isAborted: () => boolean,
  ): Promise<void> {
    emit({ type: 'run-start', prompt, mode: 'sim', modelId: MODEL_ID, params })
    const promptTokens = this.tokenizer.encode(prompt)
    emit({ type: 'tokenize', tokens: promptTokens })

    const rand = mulberry32(seedFromTokens(promptTokens.map((t) => t.id)))
    const seq: TokenInfo[] = [...promptTokens]
    let text = prompt

    for (let cycle = 0; cycle < params.maxNewTokens; cycle++) {
      await new Promise((r) => setTimeout(r, 0))
      if (isAborted()) { emit({ type: 'run-end', reason: 'aborted' }); return }

      const preview = seq.slice(-4).map(() =>
        Array.from({ length: 16 }, () => Math.round((rand() * 2 - 1) * 100) / 100))
      emit({ type: 'embed', cycle, seqLen: seq.length, dims: this.dims, preview })

      for (let i = 0; i < this.layers; i++)
        emit({ type: 'layer', cycle, index: i, total: this.layers,
          activationNorm: Math.round((1 + 0.05 * i + 0.3 * rand()) * 100) / 100 })

      const candidates = candidateWords(text, rand).map((word) => {
        const tok = this.tokenizer.encode(word)[0]
        return { id: tok.id, text: tok.text }
      })
      const scored = candidates
        .map((c, i) => ({ ...c, logit: 10 - i * 1.1 + rand() * 0.6 }))
        .sort((a, b) => b.logit - a.logit)
      emit({ type: 'logits', cycle, topK: scored })

      const probs = softmax(scored.map((c) => c.logit), params.temperature)
      emit({ type: 'softmax', cycle, temperature: params.temperature,
        topK: scored.map((c, i) => ({ id: c.id, text: c.text, prob: probs[i] })) })

      const method = params.temperature === 0 ? 'greedy' : 'top-k'
      const idx = method === 'greedy' ? 0 : sampleIndex(probs, rand)
      const chosen = { id: scored[idx].id, text: scored[idx].text }
      emit({ type: 'sample', cycle, chosen, method })
      emit({ type: 'append', cycle, token: chosen })
      seq.push(chosen)
      text += chosen.text

      if (chosen.text.trim() === '.' && rand() < 0.03 * cycle) {
        emit({ type: 'run-end', reason: 'eos' })
        return
      }
    }
    emit({ type: 'run-end', reason: 'max-tokens' })
  }
}
```

- [ ] **Step 3: Verify pass** — `npx vitest run src/engine` → all green.

- [ ] **Step 4: Commit** — `git add src/engine/simulated && git commit -m "feat: deterministic simulated engine"`

---

### Task 11: Viz selectors

**Files:**
- Create: `src/viz/selectors.ts`
- Test: `src/viz/selectors.test.ts`

**Interfaces:**
- Produces:

```ts
export type StageId = 'tokenizer' | 'embeddings' | 'layers' | 'logits' | 'sampler' | null
export function activeStage(e: TraceEvent | undefined): StageId
export function eventAt(events: TraceEvent[], cursor: number): TraceEvent | undefined
export function visibleTokens(events: TraceEvent[], cursor: number): { prompt: TokenInfo[]; generated: TokenInfo[] }
export function latestOfType<K extends TraceEvent['type']>(events: TraceEvent[], cursor: number, type: K): Extract<TraceEvent, { type: K }> | undefined
export function cycleTickIndices(events: TraceEvent[]): number[]  // indices of 'append' events
```

Stage mapping (from spec): `tokenize`→tokenizer, `embed`→embeddings, `layer`→layers, `logits`+`softmax`→logits, `sample`+`append`→sampler, `run-start`/`run-end`/undefined→null.

- [ ] **Step 1: Failing tests** (`src/viz/selectors.test.ts`)

```ts
import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { activeStage, cycleTickIndices, eventAt, latestOfType, visibleTokens } from './selectors'

const trace = makeFixtureTrace()  // 2 cycles, 3 layers

test('activeStage maps events to stage cards', () => {
  expect(activeStage({ type: 'tokenize', tokens: [] })).toBe('tokenizer')
  expect(activeStage(eventAt(trace, 2))).toBe('embeddings')
  expect(activeStage({ type: 'softmax', cycle: 0, temperature: 1, topK: [] })).toBe('logits')
  expect(activeStage({ type: 'append', cycle: 0, token: { id: 1, text: 'x' } })).toBe('sampler')
  expect(activeStage(undefined)).toBe(null)
  expect(activeStage({ type: 'run-end', reason: 'eos' })).toBe(null)
})

test('visibleTokens grows with cursor', () => {
  expect(visibleTokens(trace, 0).prompt).toHaveLength(0)          // before tokenize
  expect(visibleTokens(trace, 1).prompt).toHaveLength(2)          // after tokenize
  expect(visibleTokens(trace, 1).generated).toHaveLength(0)
  expect(visibleTokens(trace, trace.length - 1).generated).toHaveLength(2)
})

test('latestOfType finds most recent event at or before cursor', () => {
  const last = trace.length - 1
  expect(latestOfType(trace, last, 'softmax')?.cycle).toBe(1)
  expect(latestOfType(trace, 5, 'tokenize')?.type).toBe('tokenize')
  expect(latestOfType(trace, 0, 'softmax')).toBeUndefined()
})

test('cycleTickIndices marks append events', () => {
  const ticks = cycleTickIndices(trace)
  expect(ticks).toHaveLength(2)
  expect(trace[ticks[0]].type).toBe('append')
})
```

- [ ] **Step 2: Verify failure; implement** (`src/viz/selectors.ts`)

```ts
import type { TokenInfo, TraceEvent } from '../trace/types'

export type StageId = 'tokenizer' | 'embeddings' | 'layers' | 'logits' | 'sampler' | null

const STAGE_OF: Partial<Record<TraceEvent['type'], StageId>> = {
  tokenize: 'tokenizer', embed: 'embeddings', layer: 'layers',
  logits: 'logits', softmax: 'logits', sample: 'sampler', append: 'sampler',
}

export function activeStage(e: TraceEvent | undefined): StageId {
  return e ? STAGE_OF[e.type] ?? null : null
}

export function eventAt(events: TraceEvent[], cursor: number): TraceEvent | undefined {
  return cursor >= 0 ? events[cursor] : undefined
}

export function visibleTokens(events: TraceEvent[], cursor: number) {
  const prompt: TokenInfo[] = []
  const generated: TokenInfo[] = []
  for (const e of events.slice(0, cursor + 1)) {
    if (e.type === 'tokenize') prompt.push(...e.tokens)
    if (e.type === 'append') generated.push(e.token)
  }
  return { prompt, generated }
}

export function latestOfType<K extends TraceEvent['type']>(
  events: TraceEvent[], cursor: number, type: K,
): Extract<TraceEvent, { type: K }> | undefined {
  for (let i = Math.min(cursor, events.length - 1); i >= 0; i--)
    if (events[i].type === type) return events[i] as Extract<TraceEvent, { type: K }>
  return undefined
}

export function cycleTickIndices(events: TraceEvent[]): number[] {
  return events.flatMap((e, i) => (e.type === 'append' ? [i] : []))
}
```

- [ ] **Step 3: Verify pass.**

- [ ] **Step 4: Commit** — `git add src/viz && git commit -m "feat: viz selectors"`

---

### Task 12: TokenStream and PipelineBand components

**Files:**
- Create: `src/viz/TokenStream.tsx`, `src/viz/PipelineBand.tsx`
- Test: `src/viz/TokenStream.test.tsx`, `src/viz/PipelineBand.test.tsx`

**Interfaces:**
- Consumes: selectors from Task 11
- Produces:

```tsx
export function TokenStream(props: { events: TraceEvent[]; cursor: number }): JSX.Element
export function PipelineBand(props: { events: TraceEvent[]; cursor: number }): JSX.Element
```

Test hooks (exact `data-testid` values later tasks and e2e rely on): `prompt-token`, `generated-token`, `stage-card` (one per stage, with `data-stage` = StageId and `data-active` = 'true'/'false').

- [ ] **Step 1: Failing tests**

`src/viz/TokenStream.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { TokenStream } from './TokenStream'

const trace = makeFixtureTrace()

test('shows prompt tokens once tokenize is reached', () => {
  render(<TokenStream events={trace} cursor={1} />)
  expect(screen.getAllByTestId('prompt-token')).toHaveLength(2)
  expect(screen.queryAllByTestId('generated-token')).toHaveLength(0)
})

test('shows generated tokens up to cursor', () => {
  render(<TokenStream events={trace} cursor={trace.length - 1} />)
  expect(screen.getAllByTestId('generated-token')).toHaveLength(2)
  expect(screen.getAllByTestId('generated-token')[0]).toHaveTextContent('sat')
})
```

`src/viz/PipelineBand.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { PipelineBand } from './PipelineBand'

const trace = makeFixtureTrace()

test('renders five stage cards', () => {
  render(<PipelineBand events={trace} cursor={-1} />)
  expect(screen.getAllByTestId('stage-card')).toHaveLength(5)
})

test('highlights the stage of the cursor event', () => {
  render(<PipelineBand events={trace} cursor={1} />)  // tokenize event
  const active = screen.getAllByTestId('stage-card').filter((c) => c.dataset.active === 'true')
  expect(active).toHaveLength(1)
  expect(active[0].dataset.stage).toBe('tokenizer')
})
```

- [ ] **Step 2: Verify failure; implement**

`src/viz/TokenStream.tsx`:

```tsx
import type { TraceEvent } from '../trace/types'
import { visibleTokens } from './selectors'

export function TokenStream({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const { prompt, generated } = visibleTokens(events, cursor)
  return (
    <div className="token-stream" aria-label="Token stream">
      {prompt.map((t, i) => (
        <span key={`p${i}`} data-testid="prompt-token" className="token token-prompt" title={`id ${t.id}`}>
          {t.text}
        </span>
      ))}
      {generated.map((t, i) => (
        <span key={`g${i}`} data-testid="generated-token" className="token token-generated" title={`id ${t.id}`}>
          {t.text}
        </span>
      ))}
    </div>
  )
}
```

`src/viz/PipelineBand.tsx`:

```tsx
import type { TraceEvent } from '../trace/types'
import { activeStage, eventAt, type StageId } from './selectors'

const STAGES: Array<{ id: Exclude<StageId, null>; label: string }> = [
  { id: 'tokenizer', label: 'Tokenizer' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'layers', label: 'Layers' },
  { id: 'logits', label: 'Logits' },
  { id: 'sampler', label: 'Sampler' },
]

export function PipelineBand({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const active = activeStage(eventAt(events, cursor))
  return (
    <div className="pipeline-band">
      {STAGES.map((s, i) => (
        <div key={s.id} className="stage-wrap">
          {i > 0 && <span className="stage-arrow">→</span>}
          <div data-testid="stage-card" data-stage={s.id} data-active={String(active === s.id)}
            className="stage-card">
            {s.label}
          </div>
        </div>
      ))}
      <svg className="loop-arrow" viewBox="0 0 100 20" aria-label="loop back to token stream">
        <path d="M95 15 H10 M10 15 L16 10 M10 15 L16 20" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    </div>
  )
}
```

- [ ] **Step 3: Verify pass.**

- [ ] **Step 4: Commit** — `git add src/viz && git commit -m "feat: token stream and pipeline band"`

---

### Task 13: Detail panel — Tokenizer, Embeddings, Layers

**Files:**
- Create: `src/viz/DetailPanel.tsx`, `src/viz/details/TokenizerDetail.tsx`, `src/viz/details/EmbeddingsDetail.tsx`, `src/viz/details/LayersDetail.tsx`
- Test: `src/viz/DetailPanel.test.tsx`

**Interfaces:**
- Consumes: selectors
- Produces:

```tsx
export function DetailPanel(props: { events: TraceEvent[]; cursor: number; mode: Mode }): JSX.Element
// detail components each take precisely the events they render:
export function TokenizerDetail(props: { tokens: TokenInfo[] })
export function EmbeddingsDetail(props: { event: Extract<TraceEvent, { type: 'embed' }> })
export function LayersDetail(props: { event: Extract<TraceEvent, { type: 'layer' }>; mode: Mode })
```

`DetailPanel` switches on `activeStage(eventAt(events, cursor))` and pulls data via `latestOfType`. Test ids: `detail-tokenizer`, `detail-embeddings`, `detail-layers`, `detail-empty`. In real mode, `LayersDetail` renders the text `schematic` (spec: labeled illustrative).

- [ ] **Step 1: Failing tests** (`src/viz/DetailPanel.test.tsx`)

```tsx
import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { DetailPanel } from './DetailPanel'

const trace = makeFixtureTrace()  // cursor 1=tokenize, 2=embed, 3..5=layers

test('tokenizer detail shows token chips with ids', () => {
  render(<DetailPanel events={trace} cursor={1} mode="sim" />)
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('The')
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('10')  // token id
})

test('embeddings detail shows dims caption and heat cells', () => {
  render(<DetailPanel events={trace} cursor={2} mode="sim" />)
  expect(screen.getByTestId('detail-embeddings')).toHaveTextContent('576')
})

test('layers detail lights the active layer; sim shows norms', () => {
  render(<DetailPanel events={trace} cursor={4} mode="sim" />)  // layer index 1 of 3
  const blocks = screen.getAllByTestId('layer-block')
  expect(blocks).toHaveLength(3)
  expect(blocks[1].dataset.lit).toBe('true')
  expect(blocks[2].dataset.lit).toBe('false')
})

test('real mode labels layers as schematic', () => {
  render(<DetailPanel events={trace} cursor={4} mode="real" />)
  expect(screen.getByTestId('detail-layers')).toHaveTextContent(/schematic/i)
})

test('no relevant event renders empty state', () => {
  render(<DetailPanel events={trace} cursor={-1} mode="sim" />)
  expect(screen.getByTestId('detail-empty')).toBeInTheDocument()
})

test('truncated tokenize event shows a notice', () => {
  const t = makeFixtureTrace()
  const tok = t[1]
  if (tok.type === 'tokenize') tok.truncated = true
  render(<DetailPanel events={t} cursor={1} mode="sim" />)
  expect(screen.getByTestId('truncation-notice')).toBeInTheDocument()
})
```

- [ ] **Step 2: Verify failure; implement**

`src/viz/details/TokenizerDetail.tsx`:

```tsx
import type { TokenInfo } from '../../trace/types'

export function TokenizerDetail({ tokens, truncated }: { tokens: TokenInfo[]; truncated?: boolean }) {
  return (
    <div data-testid="detail-tokenizer" className="detail">
      <h3>Tokenizer</h3>
      <p>The prompt is split into {tokens.length} tokens, each mapped to a vocabulary ID.</p>
      {truncated && (
        <p data-testid="truncation-notice" className="notice">
          Prompt was longer than the model's context window — it was truncated to fit.
        </p>
      )}
      <div className="token-chip-row">
        {tokens.map((t, i) => (
          <span key={i} className={`token-chip hue-${i % 6}`}>
            <span className="chip-text">{t.text}</span>
            <span className="chip-id">{t.id}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
```

`src/viz/details/EmbeddingsDetail.tsx`:

```tsx
import type { TraceEvent } from '../../trace/types'

export function EmbeddingsDetail({ event }: { event: Extract<TraceEvent, { type: 'embed' }> }) {
  return (
    <div data-testid="detail-embeddings" className="detail">
      <h3>Embeddings</h3>
      <p>Each token becomes a vector of {event.dims} numbers (showing 16 dims of the last {event.preview.length} tokens).</p>
      <svg width={event.preview[0]?.length * 14} height={event.preview.length * 14}>
        {event.preview.map((row, r) =>
          row.map((v, c) => (
            <rect key={`${r}-${c}`} x={c * 14} y={r * 14} width={12} height={12}
              fill={`hsl(${v >= 0 ? 210 : 10} 70% ${50 + Math.abs(v) * 30}%)`} />
          )),
        )}
      </svg>
    </div>
  )
}
```

`src/viz/details/LayersDetail.tsx`:

```tsx
import type { Mode, TraceEvent } from '../../trace/types'

export function LayersDetail({ event, mode }: { event: Extract<TraceEvent, { type: 'layer' }>; mode: Mode }) {
  return (
    <div data-testid="detail-layers" className="detail">
      <h3>Transformer layers {mode === 'real' && <em>(schematic — real internals not exposed)</em>}</h3>
      <div className="layer-stack">
        {Array.from({ length: event.total }, (_, i) => (
          <div key={i} data-testid="layer-block" data-lit={String(i <= event.index)} className="layer-block">
            L{i}{mode === 'sim' && i === event.index && event.activationNorm != null && (
              <span className="layer-norm"> ‖h‖ {event.activationNorm}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
```

`src/viz/DetailPanel.tsx`:

```tsx
import type { Mode, TraceEvent } from '../trace/types'
import { activeStage, eventAt, latestOfType } from './selectors'
import { EmbeddingsDetail } from './details/EmbeddingsDetail'
import { LayersDetail } from './details/LayersDetail'
import { TokenizerDetail } from './details/TokenizerDetail'

export function DetailPanel({ events, cursor, mode }: { events: TraceEvent[]; cursor: number; mode: Mode }) {
  const stage = activeStage(eventAt(events, cursor))
  const empty = <div data-testid="detail-empty" className="detail">Press Generate, then step through the pipeline.</div>

  switch (stage) {
    case 'tokenizer': {
      const e = latestOfType(events, cursor, 'tokenize')
      return e ? <TokenizerDetail tokens={e.tokens} truncated={e.truncated} /> : empty
    }
    case 'embeddings': {
      const e = latestOfType(events, cursor, 'embed')
      return e ? <EmbeddingsDetail event={e} /> : empty
    }
    case 'layers': {
      const e = latestOfType(events, cursor, 'layer')
      return e ? <LayersDetail event={e} mode={mode} /> : empty
    }
    default:
      return empty  // 'logits' and 'sampler' branches added in Task 14
  }
}
```

- [ ] **Step 3: Verify pass.**

- [ ] **Step 4: Commit** — `git add src/viz && git commit -m "feat: detail panel with tokenizer/embeddings/layers views"`

---

### Task 14: Detail panel — Logits and Sampler

**Files:**
- Create: `src/viz/details/LogitsDetail.tsx`, `src/viz/details/SamplerDetail.tsx`
- Modify: `src/viz/DetailPanel.tsx` (replace the `default` branch)
- Test: extend `src/viz/DetailPanel.test.tsx`

**Interfaces:**
- Produces:

```tsx
export function LogitsDetail(props: {
  logits: Extract<TraceEvent, { type: 'logits' }>
  softmax?: Extract<TraceEvent, { type: 'softmax' }>   // present once cursor reaches softmax
})
export function SamplerDetail(props: {
  softmax: Extract<TraceEvent, { type: 'softmax' }>
  sample?: Extract<TraceEvent, { type: 'sample' }>
})
```

`LogitsDetail` renders one SVG bar per candidate; when `softmax` is present (cursor at/past the softmax event of the same cycle) bars are scaled by `prob`, otherwise by normalized logit — the switch is what visualizes "softmax happened". `SamplerDetail` shows the probability bars as a horizontal roulette strip; when `sample` is present, the chosen candidate is marked. Test ids: `detail-logits`, `logit-bar` (with `data-token`), `detail-sampler`, `chosen-marker`.

- [ ] **Step 1: Failing tests** (append to `src/viz/DetailPanel.test.tsx`)

```tsx
// makeFixtureTrace cycle 0: index 6=logits, 7=softmax, 8=sample, 9=append
test('logits detail shows one bar per candidate', () => {
  render(<DetailPanel events={trace} cursor={6} mode="sim" />)
  expect(screen.getAllByTestId('logit-bar')).toHaveLength(3)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('sat')
})

test('softmax cursor switches bars to probabilities', () => {
  render(<DetailPanel events={trace} cursor={7} mode="sim" />)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('70')  // 0.7 → 70%
})

test('sampler detail marks the chosen token', () => {
  render(<DetailPanel events={trace} cursor={8} mode="sim" />)
  expect(screen.getByTestId('chosen-marker')).toHaveTextContent('sat')
})
```

- [ ] **Step 2: Verify failure; implement**

`src/viz/details/LogitsDetail.tsx`:

```tsx
import type { TraceEvent } from '../../trace/types'

type LogitsEvent = Extract<TraceEvent, { type: 'logits' }>
type SoftmaxEvent = Extract<TraceEvent, { type: 'softmax' }>

export function LogitsDetail({ logits, softmax }: { logits: LogitsEvent; softmax?: SoftmaxEvent }) {
  const showProbs = softmax != null && softmax.cycle === logits.cycle
  const maxLogit = Math.max(...logits.topK.map((c) => c.logit))
  return (
    <div data-testid="detail-logits" className="detail">
      <h3>{showProbs ? `Softmax (T=${softmax!.temperature})` : 'Logits'}</h3>
      <p>{showProbs
        ? 'Softmax turns scores into probabilities that sum to 100%.'
        : 'The model scores every vocabulary token; these are the top candidates.'}</p>
      <div className="bar-chart">
        {logits.topK.map((c, i) => {
          const frac = showProbs ? softmax!.topK[i].prob : Math.max(0.02, c.logit / maxLogit)
          return (
            <div key={c.id} className="bar-row">
              <span className="bar-label">{c.text}</span>
              <svg data-testid="logit-bar" data-token={c.text} width="200" height="14">
                <rect width={200 * frac} height="14" className="bar-rect"
                  style={{ transition: 'width .4s' }} />
              </svg>
              <span className="bar-value">
                {showProbs ? `${Math.round(softmax!.topK[i].prob * 100)}%` : c.logit.toFixed(1)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
```

`src/viz/details/SamplerDetail.tsx`:

```tsx
import type { TraceEvent } from '../../trace/types'

type SoftmaxEvent = Extract<TraceEvent, { type: 'softmax' }>
type SampleEvent = Extract<TraceEvent, { type: 'sample' }>

export function SamplerDetail({ softmax, sample }: { softmax: SoftmaxEvent; sample?: SampleEvent }) {
  let offset = 0
  return (
    <div data-testid="detail-sampler" className="detail">
      <h3>Sampling {sample && `(${sample.method})`}</h3>
      <p>A weighted draw across the probability strip picks the next token.</p>
      <svg width="400" height="30" className="roulette">
        {softmax.topK.map((c) => {
          const x = offset
          const w = 400 * c.prob
          offset += w
          const chosen = sample?.chosen.id === c.id
          return <rect key={c.id} x={x} width={Math.max(w, 1)} height="30"
            data-chosen={String(chosen)} className={chosen ? 'slice slice-chosen' : 'slice'} />
        })}
      </svg>
      {sample && (
        <p data-testid="chosen-marker" className="chosen-marker">
          → chose “{sample.chosen.text}” (id {sample.chosen.id})
        </p>
      )}
    </div>
  )
}
```

Replace `DetailPanel`'s `default` branch:

```tsx
    case 'logits': {
      const logits = latestOfType(events, cursor, 'logits')
      const sm = latestOfType(events, cursor, 'softmax')
      return logits ? <LogitsDetail logits={logits} softmax={sm} /> : empty
    }
    case 'sampler': {
      const sm = latestOfType(events, cursor, 'softmax')
      const sample = latestOfType(events, cursor, 'sample')
      return sm ? <SamplerDetail softmax={sm} sample={sample} /> : empty
    }
    default:
      return empty
```

(add the imports for `LogitsDetail`, `SamplerDetail`).

- [ ] **Step 3: Verify pass** — `npx vitest run src/viz`.

- [ ] **Step 4: Commit** — `git add src/viz && git commit -m "feat: logits and sampler detail views"`

---

### Task 15: Player controls component

**Files:**
- Create: `src/app/Controls.tsx`
- Test: `src/app/Controls.test.tsx`

**Interfaces:**
- Consumes: `usePlayerStore`, `useTraceStore`, `cycleTickIndices`
- Produces: `export function Controls(): JSX.Element` — buttons (test ids `btn-play`, `btn-pause`, `btn-step-back`, `btn-step-fwd`, `btn-live`), cycle-step buttons (`btn-cycle-back`, `btn-cycle-fwd` — seek to the previous/next `append` event via `cycleTickIndices`), a scrubber `<input type="range" data-testid="scrubber">` with `<datalist>` ticks at cycle boundaries, and a speed `<select data-testid="speed">` with options 0.5/1/2/4.

- [ ] **Step 1: Failing tests** (`src/app/Controls.test.tsx`)

```tsx
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { useTraceStore } from '../trace/store'
import { initialPlayerState } from '../player/reducer'
import { usePlayerStore } from '../player/store'
import { Controls } from './Controls'

beforeEach(() => {
  useTraceStore.setState({ events: makeFixtureTrace() })
  const len = useTraceStore.getState().events.length
  usePlayerStore.setState({ ...initialPlayerState, traceLength: len, status: 'paused', cursor: 3 })
})

test('play and pause dispatch status changes', () => {
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-play'))
  expect(usePlayerStore.getState().status).toBe('playing')
  fireEvent.click(screen.getByTestId('btn-pause'))
  expect(usePlayerStore.getState().status).toBe('paused')
})

test('step buttons move the cursor', () => {
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-step-fwd'))
  expect(usePlayerStore.getState().cursor).toBe(4)
  fireEvent.click(screen.getByTestId('btn-step-back'))
  expect(usePlayerStore.getState().cursor).toBe(3)
})

test('scrubber seeks', () => {
  render(<Controls />)
  fireEvent.change(screen.getByTestId('scrubber'), { target: { value: '7' } })
  expect(usePlayerStore.getState().cursor).toBe(7)
  expect(usePlayerStore.getState().followLive).toBe(false)
})

test('live button jumps to frontier and plays', () => {
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-live'))
  const s = usePlayerStore.getState()
  expect(s.cursor).toBe(s.traceLength - 1)
  expect(s.status).toBe('playing')
})

test('speed select dispatches setSpeed', () => {
  render(<Controls />)
  fireEvent.change(screen.getByTestId('speed'), { target: { value: '2' } })
  expect(usePlayerStore.getState().speed).toBe(2)
})

test('cycle buttons jump between append events', () => {
  // fixture (2 cycles, 3 layers): append events at indices 9 and 17
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-cycle-fwd'))   // from cursor 3
  expect(usePlayerStore.getState().cursor).toBe(9)
  fireEvent.click(screen.getByTestId('btn-cycle-fwd'))
  expect(usePlayerStore.getState().cursor).toBe(17)
  fireEvent.click(screen.getByTestId('btn-cycle-back'))
  expect(usePlayerStore.getState().cursor).toBe(9)
})
```

- [ ] **Step 2: Verify failure; implement** (`src/app/Controls.tsx`)

```tsx
import { usePlayerStore } from '../player/store'
import { useTraceStore } from '../trace/store'
import { cycleTickIndices } from '../viz/selectors'

export function Controls() {
  const { cursor, status, speed, traceLength, dispatch } = usePlayerStore()
  const events = useTraceStore((s) => s.events)
  const ticks = cycleTickIndices(events)

  const cycleFwd = () => {
    const next = ticks.find((i) => i > cursor)
    if (next !== undefined) dispatch({ type: 'seek', index: next })
  }
  const cycleBack = () => {
    const prev = [...ticks].reverse().find((i) => i < cursor)
    if (prev !== undefined) dispatch({ type: 'seek', index: prev })
  }

  return (
    <div className="controls">
      <button data-testid="btn-cycle-back" onClick={cycleBack} title="Previous token cycle">|◀◀</button>
      <button data-testid="btn-step-back" onClick={() => dispatch({ type: 'stepBack' })} title="Step back">◀</button>
      {status === 'playing' ? (
        <button data-testid="btn-pause" onClick={() => dispatch({ type: 'pause' })} title="Pause">⏸</button>
      ) : (
        <button data-testid="btn-play" onClick={() => dispatch({ type: 'play' })} title="Play">▶</button>
      )}
      <button data-testid="btn-step-fwd" onClick={() => dispatch({ type: 'stepForward' })} title="Step">▶|</button>
      <button data-testid="btn-cycle-fwd" onClick={cycleFwd} title="Next token cycle">▶▶|</button>
      <input data-testid="scrubber" type="range" min={0} max={Math.max(traceLength - 1, 0)}
        value={Math.max(cursor, 0)} list="cycle-ticks"
        onChange={(e) => dispatch({ type: 'seek', index: Number(e.target.value) })} />
      <datalist id="cycle-ticks">
        {ticks.map((i) => <option key={i} value={i} />)}
      </datalist>
      <select data-testid="speed" value={speed} onChange={(e) => dispatch({ type: 'setSpeed', speed: Number(e.target.value) })}>
        {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
      </select>
      <button data-testid="btn-live" onClick={() => dispatch({ type: 'goLive' })} title="Jump to live">⏺ Live</button>
    </div>
  )
}
```

Note: the play/pause tests render both buttons across a click — Testing Library re-queries after state updates, so the conditional swap works. If a test flakes on the swap, render play and pause as always-present buttons and disable the inactive one instead.

- [ ] **Step 3: Verify pass.**

- [ ] **Step 4: Commit** — `git add src/app && git commit -m "feat: player controls with scrubber"`

---

### Task 16: App shell with simulated mode end-to-end

**Files:**
- Create: `src/app/PromptBar.tsx`
- Modify: `src/App.tsx`, `src/index.css`
- Test: `src/app/PromptBar.test.tsx`, replace `src/App.test.tsx`

**Interfaces:**
- Consumes: everything so far
- Produces:

```tsx
export interface PromptBarProps {
  mode: Mode
  onModeChange(mode: Mode): void
  onGenerate(prompt: string, params: GenParams): void
  busy: boolean
}
export function PromptBar(props: PromptBarProps): JSX.Element
// App wires: SimulatedEngine (fallback tokenizer upgraded to HF when loaded),
// Generate → traceStore.clear() + player reset + engine.run(..., append)
```

Test ids: `prompt-input`, `btn-generate`, `mode-toggle` (checkbox: unchecked = sim), `temp-input`, `topk-input`, `maxtok-input`.

- [ ] **Step 1: Failing PromptBar tests** (`src/app/PromptBar.test.tsx`)

```tsx
import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { PromptBar } from './PromptBar'

const noop = () => {}

test('generate disabled for empty prompt', () => {
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={noop} busy={false} />)
  expect(screen.getByTestId('btn-generate')).toBeDisabled()
})

test('generate passes prompt and params', () => {
  const onGenerate = vi.fn()
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat' } })
  fireEvent.change(screen.getByTestId('temp-input'), { target: { value: '0.5' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenCalledWith('The cat', { temperature: 0.5, topK: 10, maxNewTokens: 20 })
})

test('mode toggle reports changes', () => {
  const onModeChange = vi.fn()
  render(<PromptBar mode="sim" onModeChange={onModeChange} onGenerate={noop} busy={false} />)
  fireEvent.click(screen.getByTestId('mode-toggle'))
  expect(onModeChange).toHaveBeenCalledWith('real')
})
```

- [ ] **Step 2: Verify failure; implement** (`src/app/PromptBar.tsx`)

```tsx
import { useState } from 'react'
import type { GenParams, Mode } from '../trace/types'

export interface PromptBarProps {
  mode: Mode
  onModeChange(mode: Mode): void
  onGenerate(prompt: string, params: GenParams): void
  busy: boolean
}

export function PromptBar({ mode, onModeChange, onGenerate, busy }: PromptBarProps) {
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(10)
  const [maxNewTokens, setMaxNewTokens] = useState(20)

  return (
    <div className="prompt-bar">
      <input data-testid="prompt-input" value={prompt} placeholder="Type a prompt…"
        onChange={(e) => setPrompt(e.target.value)} />
      <label>
        <input data-testid="mode-toggle" type="checkbox" checked={mode === 'real'}
          onChange={(e) => onModeChange(e.target.checked ? 'real' : 'sim')} />
        Real model (~120 MB download on first use)
      </label>
      <label>T <input data-testid="temp-input" type="number" step="0.1" min="0" max="2" value={temperature}
        onChange={(e) => setTemperature(Number(e.target.value))} /></label>
      <label>top-k <input data-testid="topk-input" type="number" min="1" max="10" value={topK}
        onChange={(e) => setTopK(Number(e.target.value))} /></label>
      <label>max <input data-testid="maxtok-input" type="number" min="1" max="100" value={maxNewTokens}
        onChange={(e) => setMaxNewTokens(Number(e.target.value))} /></label>
      <button data-testid="btn-generate" disabled={prompt.trim() === '' || busy}
        onClick={() => onGenerate(prompt, { temperature, topK, maxNewTokens })}>
        Generate
      </button>
    </div>
  )
}
```

- [ ] **Step 3: Failing App integration test** (replace `src/App.test.tsx`)

```tsx
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, test } from 'vitest'
import App from './App'
import { useTraceStore } from './trace/store'
import { initialPlayerState } from './player/reducer'
import { usePlayerStore } from './player/store'

beforeEach(() => {
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
})

test('generate in sim mode fills the trace and plays', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => expect(useTraceStore.getState().events.length).toBeGreaterThan(10))
  const last = useTraceStore.getState().events.at(-1)
  expect(last?.type).toBe('run-end')
  expect(usePlayerStore.getState().status).toBe('playing')
})
```

- [ ] **Step 4: Implement App wiring** (replace `src/App.tsx`)

```tsx
import { useEffect, useRef, useState } from 'react'
import { Controls } from './app/Controls'
import { PromptBar } from './app/PromptBar'
import { SimulatedEngine } from './engine/simulated/SimulatedEngine'
import { fallbackTokenizer, loadTokenizer, type Tokenizer } from './engine/tokenizer'
import type { PipelineEngine, RunHandle } from './engine/types'
import { usePlaybackTicker, usePlayerStore } from './player/store'
import { useTraceStore } from './trace/store'
import type { GenParams, Mode } from './trace/types'
import { DetailPanel } from './viz/DetailPanel'
import { PipelineBand } from './viz/PipelineBand'
import { TokenStream } from './viz/TokenStream'

export default function App() {
  usePlaybackTicker()
  const events = useTraceStore((s) => s.events)
  const cursor = usePlayerStore((s) => s.cursor)
  const [mode, setMode] = useState<Mode>('sim')
  const [busy, setBusy] = useState(false)
  const tokenizerRef = useRef<Tokenizer>(fallbackTokenizer())
  const runRef = useRef<RunHandle | null>(null)

  useEffect(() => {
    let live = true
    loadTokenizer().then((t) => { if (live) tokenizerRef.current = t })
    return () => { live = false }
  }, [])

  // trace growth → player
  useEffect(() => {
    usePlayerStore.getState().dispatch({ type: 'traceGrew', length: events.length })
  }, [events.length])

  const handleGenerate = async (prompt: string, params: GenParams) => {
    runRef.current?.abort()
    await runRef.current?.done
    useTraceStore.getState().clear()
    usePlayerStore.getState().dispatch({ type: 'reset' })
    const engine: PipelineEngine = new SimulatedEngine(tokenizerRef.current)  // real engine wired in Task 18
    setBusy(true)
    const handle = engine.run(prompt, params, (e) => useTraceStore.getState().append(e))
    runRef.current = handle
    handle.done.finally(() => setBusy(false))
  }

  return (
    <div className="app">
      <h1>LLM Pipeline Visualizer</h1>
      <PromptBar mode={mode} onModeChange={setMode} onGenerate={handleGenerate} busy={busy} />
      <TokenStream events={events} cursor={cursor} />
      <PipelineBand events={events} cursor={cursor} />
      <DetailPanel events={events} cursor={cursor} mode={mode} />
      <Controls />
    </div>
  )
}
```

Append this layout CSS to `src/index.css` (functional baseline; visual polish is a later pass):

```css
.app { max-width: 960px; margin: 0 auto; padding: 1rem; font-family: system-ui, sans-serif; }
.prompt-bar { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap; }
.prompt-bar input[data-testid="prompt-input"] { flex: 1; min-width: 16rem; padding: .4rem; }
.token-stream { min-height: 2.2rem; padding: .5rem 0; }
.token { display: inline-block; border: 1px solid #bbb; border-radius: 4px; padding: 0 .3rem; margin: 0 1px; white-space: pre; }
.token-generated { border-color: #2a7; background: #e8f8f0; }
.pipeline-band { display: flex; align-items: center; gap: .25rem; position: relative; padding-bottom: 1.5rem; }
.stage-wrap { display: flex; align-items: center; gap: .25rem; }
.stage-card { border: 2px solid #ccc; border-radius: 6px; padding: .6rem .9rem; transition: border-color .3s, background .3s, box-shadow .3s; }
.stage-card[data-active="true"] { border-color: #d64; background: #fff3ee; box-shadow: 0 0 8px #d648; }
/* animated hand-off: the arrow feeding the active stage lights up */
.stage-arrow { color: #bbb; transition: color .3s; }
.stage-wrap:has(.stage-card[data-active="true"]) .stage-arrow { color: #d64; }
.loop-arrow { position: absolute; bottom: 0; left: 0; width: 100%; height: 1.2rem; color: #999; }
.layer-stack { display: flex; gap: .3rem; }
.layer-block { border: 1px solid #ccc; border-radius: 4px; padding: .3rem .5rem; transition: background .2s; }
.layer-block[data-lit="true"] { background: #dbeafe; border-color: #36c; }
.detail { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; min-height: 10rem; }
.bar-row { display: flex; align-items: center; gap: .5rem; }
.bar-label { width: 5rem; text-align: right; white-space: pre; }
.bar-rect { fill: #36c; }
.slice { fill: #cde; stroke: #fff; }
.slice-chosen { fill: #d64; }
.controls { display: flex; gap: .5rem; align-items: center; padding: .5rem 0; }
.controls input[type="range"] { flex: 1; }
.notice, .model-error { color: #b40; }
.device-chip { border: 1px solid #999; border-radius: 999px; padding: 0 .5rem; font-size: .8rem; }
```

(The spec's "animated packet crossing connectors" is realized minimally as the lit-arrow hand-off above; a literal moving dot is deferred polish.)

- [ ] **Step 5: Verify** — `npx vitest run` all green; `npm run dev` manually: type a prompt, Generate, watch playback advance through the stages, scrub back and forth.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: app shell — simulated mode fully playable"`

---

### Task 17: Transformers worker protocol and client engine

**Files:**
- Create: `src/engine/transformers/protocol.ts`, `src/engine/transformers/TransformersEngine.ts`
- Test: `src/engine/transformers/TransformersEngine.test.ts`

**Interfaces:**
- Produces:

```ts
// protocol.ts
export type WorkerRequest =
  | { type: 'prepare'; modelId: string }
  | { type: 'run'; runId: number; prompt: string; params: GenParams }
  | { type: 'abort' }
export type WorkerResponse =
  | { type: 'progress'; info: ProgressInfo }
  | { type: 'ready'; device: 'webgpu' | 'wasm' }
  | { type: 'trace'; runId: number; event: TraceEvent }
  | { type: 'done'; runId: number }
  | { type: 'fatal'; message: string }
// TransformersEngine.ts
export class TransformersEngine implements PipelineEngine {
  constructor(workerFactory?: () => Worker)   // default: new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  device: 'webgpu' | 'wasm' | null            // set after prepare()
}
```

Client behavior: `prepare()` posts `prepare`, resolves on `ready` (recording `device`), rejects on `fatal`, forwards `progress` to `onProgress`. `run()` assigns an incrementing `runId`, forwards only matching `trace` events to `emit`, resolves `done` on `done` message; on `fatal` mid-run it emits `{ type: 'run-end', reason: 'error', message }` itself and resolves. `abort()` posts `abort`.

- [ ] **Step 1: Failing tests with a FakeWorker** (`src/engine/transformers/TransformersEngine.test.ts`)

```ts
import { expect, test, vi } from 'vitest'
import type { TraceEvent } from '../../trace/types'
import type { WorkerRequest, WorkerResponse } from './protocol'
import { TransformersEngine } from './TransformersEngine'

class FakeWorker {
  sent: WorkerRequest[] = []
  onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null
  postMessage(msg: WorkerRequest) { this.sent.push(msg) }
  respond(msg: WorkerResponse) { this.onmessage?.({ data: msg } as MessageEvent<WorkerResponse>) }
  terminate() {}
}

function make() {
  const worker = new FakeWorker()
  const engine = new TransformersEngine(() => worker as unknown as Worker)
  return { worker, engine }
}

test('prepare resolves on ready and records device', async () => {
  const { worker, engine } = make()
  const onProgress = vi.fn()
  const p = engine.prepare(onProgress)
  worker.respond({ type: 'progress', info: { file: 'model.onnx', loaded: 1, total: 2 } })
  worker.respond({ type: 'ready', device: 'wasm' })
  await p
  expect(onProgress).toHaveBeenCalledOnce()
  expect(engine.device).toBe('wasm')
  expect(worker.sent[0]).toMatchObject({ type: 'prepare' })
})

test('prepare rejects on fatal', async () => {
  const { worker, engine } = make()
  const p = engine.prepare()
  worker.respond({ type: 'fatal', message: 'download failed' })
  await expect(p).rejects.toThrow('download failed')
})

test('run forwards matching trace events and resolves on done', async () => {
  const { worker, engine } = make()
  const events: TraceEvent[] = []
  const handle = engine.run('Hi', { temperature: 1, topK: 10, maxNewTokens: 2 }, (e) => events.push(e))
  const runId = (worker.sent.at(-1) as Extract<WorkerRequest, { type: 'run' }>).runId
  worker.respond({ type: 'trace', runId, event: { type: 'run-end', reason: 'max-tokens' } })
  worker.respond({ type: 'trace', runId: runId + 99, event: { type: 'run-end', reason: 'eos' } })  // ignored
  worker.respond({ type: 'done', runId })
  await handle.done
  expect(events).toHaveLength(1)
})

test('fatal mid-run emits synthetic error run-end', async () => {
  const { worker, engine } = make()
  const events: TraceEvent[] = []
  const handle = engine.run('Hi', { temperature: 1, topK: 10, maxNewTokens: 2 }, (e) => events.push(e))
  worker.respond({ type: 'fatal', message: 'worker crashed' })
  await handle.done
  expect(events.at(-1)).toMatchObject({ type: 'run-end', reason: 'error', message: 'worker crashed' })
})

test('abort posts abort message', () => {
  const { worker, engine } = make()
  const handle = engine.run('Hi', { temperature: 1, topK: 10, maxNewTokens: 2 }, () => {})
  handle.abort()
  expect(worker.sent.some((m) => m.type === 'abort')).toBe(true)
})
```

- [ ] **Step 2: Verify failure; implement**

`src/engine/transformers/protocol.ts` exactly as in Produces. `src/engine/transformers/TransformersEngine.ts`:

```ts
import type { GenParams, TraceEvent } from '../../trace/types'
import { MODEL_ID } from '../tokenizer'
import type { PipelineEngine, ProgressInfo, RunHandle } from '../types'
import type { WorkerRequest, WorkerResponse } from './protocol'

export class TransformersEngine implements PipelineEngine {
  private worker: Worker
  private nextRunId = 1
  device: 'webgpu' | 'wasm' | null = null
  private listeners = new Set<(msg: WorkerResponse) => void>()

  constructor(workerFactory: () => Worker = () =>
    new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })) {
    this.worker = workerFactory()
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      for (const l of [...this.listeners]) l(e.data)
    }
  }

  private post(msg: WorkerRequest) { this.worker.postMessage(msg) }

  prepare(onProgress?: (p: ProgressInfo) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'progress') onProgress?.(msg.info)
        if (msg.type === 'ready') { this.device = msg.device; this.listeners.delete(listener); resolve() }
        if (msg.type === 'fatal') { this.listeners.delete(listener); reject(new Error(msg.message)) }
      }
      this.listeners.add(listener)
      this.post({ type: 'prepare', modelId: MODEL_ID })
    })
  }

  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void): RunHandle {
    const runId = this.nextRunId++
    const done = new Promise<void>((resolve) => {
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'trace' && msg.runId === runId) emit(msg.event)
        if (msg.type === 'done' && msg.runId === runId) { this.listeners.delete(listener); resolve() }
        if (msg.type === 'fatal') {
          emit({ type: 'run-end', reason: 'error', message: msg.message })
          this.listeners.delete(listener); resolve()
        }
      }
      this.listeners.add(listener)
      this.post({ type: 'run', runId, prompt, params })
    })
    return { abort: () => this.post({ type: 'abort' }), done }
  }
}
```

- [ ] **Step 3: Verify pass.**

- [ ] **Step 4: Commit** — `git add src/engine/transformers && git commit -m "feat: transformers worker protocol and client engine"`

---

### Task 18: Worker implementation, real-mode wiring, ModelStatus

**Files:**
- Create: `src/engine/transformers/worker.ts`, `src/app/ModelStatus.tsx`
- Modify: `src/App.tsx`
- Test: `src/app/ModelStatus.test.tsx`

**Interfaces:**
- Consumes: protocol types, `topK`/`softmax`/`sampleIndex` from `engine/math.ts`
- Produces: the worker (untested in CI — pure math already covered; verified manually), `ModelStatus` (props `{ progress: ProgressInfo | null; device: 'webgpu' | 'wasm' | null; error: string | null; onFallback(): void }`, test ids `model-progress`, `device-chip`, `model-error`, `btn-fallback`), and App wiring that instantiates `TransformersEngine` lazily on first switch to real mode, shows progress, and falls back to sim on failure.

- [ ] **Step 1: Worker implementation** (`src/engine/transformers/worker.ts` — no unit test; the generation loop uses only already-tested math)

```ts
/// <reference lib="webworker" />
import type { GenParams, TokenInfo, TraceEvent } from '../../trace/types'
import { sampleIndex, softmax, topK } from '../math'
import type { WorkerRequest, WorkerResponse } from './protocol'

const post = (msg: WorkerResponse) => (self as unknown as Worker).postMessage(msg)

/* eslint-disable @typescript-eslint/no-explicit-any */
let tokenizer: any = null
let model: any = null
let aborted = false

async function prepare(modelId: string) {
  const { AutoTokenizer, AutoModelForCausalLM } = await import('@huggingface/transformers')
  const progress_callback = (p: any) => {
    if (p.status === 'progress') post({ type: 'progress', info: { file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 } })
  }
  tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback })
  const device = 'gpu' in navigator ? 'webgpu' : 'wasm'
  try {
    model = await AutoModelForCausalLM.from_pretrained(modelId, { dtype: 'q4', device, progress_callback })
    post({ type: 'ready', device })
  } catch {
    model = await AutoModelForCausalLM.from_pretrained(modelId, { dtype: 'q4', device: 'wasm', progress_callback })
    post({ type: 'ready', device: 'wasm' })
  }
}

const tokenInfo = (id: number): TokenInfo => ({ id, text: tokenizer.decode([id]) })

async function run(runId: number, prompt: string, params: GenParams) {
  aborted = false
  const emit = (event: TraceEvent) => post({ type: 'trace', runId, event })
  const { Tensor } = await import('@huggingface/transformers')

  emit({ type: 'run-start', prompt, mode: 'real', modelId: model.config._name_or_path ?? 'unknown', params })
  let promptIds: number[] = tokenizer.encode(prompt, { add_special_tokens: false })
  const maxCtx: number = model.config.max_position_embeddings ?? 2048
  const budget = Math.max(1, maxCtx - params.maxNewTokens)
  const truncated = promptIds.length > budget
  if (truncated) promptIds = promptIds.slice(-budget)   // keep the most recent tokens
  emit({ type: 'tokenize', tokens: promptIds.map(tokenInfo), ...(truncated ? { truncated } : {}) })

  const numLayers: number = model.config.num_hidden_layers ?? 12
  const dims: number = model.config.hidden_size ?? 576
  const eosId: number = model.config.eos_token_id ?? tokenizer.eos_token_id ?? -1
  const allIds = [...promptIds]
  let pastKeyValues: any = null
  let nextInputIds = promptIds

  for (let cycle = 0; cycle < params.maxNewTokens; cycle++) {
    if (aborted) { emit({ type: 'run-end', reason: 'aborted' }); break }

    // schematic embed preview (real hidden states not exposed; spec-accepted compromise)
    emit({ type: 'embed', cycle, seqLen: allIds.length, dims,
      preview: allIds.slice(-4).map((id) => Array.from({ length: 16 }, (_, d) => Math.sin(id * 0.7 + d))) })
    for (let l = 0; l < numLayers; l++) emit({ type: 'layer', cycle, index: l, total: numLayers })

    const input_ids = new Tensor('int64', BigInt64Array.from(nextInputIds.map(BigInt)), [1, nextInputIds.length])
    const attention_mask = new Tensor('int64', BigInt64Array.from(allIds.map(() => 1n)), [1, allIds.length])
    const out = await model({ input_ids, attention_mask, past_key_values: pastKeyValues })
    pastKeyValues = out.past_key_values

    const [, seq, vocab] = out.logits.dims as [number, number, number]
    const lastLogits: Float32Array = out.logits.data.slice((seq - 1) * vocab, seq * vocab)
    const top = topK(lastLogits, params.topK).map((c) => ({ ...tokenInfo(c.id), logit: Math.round(c.logit * 100) / 100 }))
    emit({ type: 'logits', cycle, topK: top })

    const probs = softmax(top.map((c) => c.logit), params.temperature)
    emit({ type: 'softmax', cycle, temperature: params.temperature,
      topK: top.map((c, i) => ({ id: c.id, text: c.text, prob: probs[i] })) })

    const method = params.temperature === 0 ? 'greedy' : 'top-k'
    const idx = method === 'greedy' ? 0 : sampleIndex(probs, Math.random)
    const chosen = { id: top[idx].id, text: top[idx].text }
    emit({ type: 'sample', cycle, chosen, method })
    emit({ type: 'append', cycle, token: chosen })

    allIds.push(chosen.id)
    nextInputIds = [chosen.id]
    if (chosen.id === eosId) { emit({ type: 'run-end', reason: 'eos' }); break }
    if (cycle === params.maxNewTokens - 1) emit({ type: 'run-end', reason: 'max-tokens' })
  }
  post({ type: 'done', runId })
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  try {
    const msg = e.data
    if (msg.type === 'prepare') await prepare(msg.modelId)
    if (msg.type === 'run') await run(msg.runId, msg.prompt, msg.params)
    if (msg.type === 'abort') aborted = true
  } catch (err) {
    post({ type: 'fatal', message: err instanceof Error ? err.message : String(err) })
  }
}
```

**Caveat for the implementer:** the exact call shapes (`Tensor` construction, `past_key_values` threading, `logits.dims`, config field names) must be checked against the installed `@huggingface/transformers` version's docs/types — fix mismatches here, not by changing the protocol. `Math.random` (not the seeded PRNG) is intentional: real mode is genuinely stochastic.

- [ ] **Step 2: Failing ModelStatus tests** (`src/app/ModelStatus.test.tsx`)

```tsx
import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { ModelStatus } from './ModelStatus'

test('shows download progress', () => {
  render(<ModelStatus progress={{ file: 'model.onnx', loaded: 50, total: 100 }} device={null} error={null} onFallback={() => {}} />)
  expect(screen.getByTestId('model-progress')).toHaveTextContent('50')
})

test('shows device chip when ready', () => {
  render(<ModelStatus progress={null} device="webgpu" error={null} onFallback={() => {}} />)
  expect(screen.getByTestId('device-chip')).toHaveTextContent('webgpu')
})

test('error offers fallback to simulated', () => {
  const onFallback = vi.fn()
  render(<ModelStatus progress={null} device={null} error="download failed" onFallback={onFallback} />)
  expect(screen.getByTestId('model-error')).toHaveTextContent('download failed')
  fireEvent.click(screen.getByTestId('btn-fallback'))
  expect(onFallback).toHaveBeenCalled()
})

test('renders nothing when idle', () => {
  const { container } = render(<ModelStatus progress={null} device={null} error={null} onFallback={() => {}} />)
  expect(container).toBeEmptyDOMElement()
})
```

- [ ] **Step 3: Implement ModelStatus** (`src/app/ModelStatus.tsx`)

```tsx
import type { ProgressInfo } from '../engine/types'

interface Props {
  progress: ProgressInfo | null
  device: 'webgpu' | 'wasm' | null
  error: string | null
  onFallback(): void
}

export function ModelStatus({ progress, device, error, onFallback }: Props) {
  if (error) return (
    <div data-testid="model-error" className="model-error">
      Real model unavailable: {error}
      <button data-testid="btn-fallback" onClick={onFallback}>Continue in Simulated mode</button>
    </div>
  )
  if (progress) return (
    <div data-testid="model-progress" className="model-progress">
      Downloading {progress.file}: {progress.total > 0 ? Math.round((progress.loaded / progress.total) * 100) : 0}%
    </div>
  )
  if (device) return <span data-testid="device-chip" className="device-chip">{device}</span>
  return null
}
```

- [ ] **Step 4: Wire real mode into App** (modify `src/App.tsx`)

Add state and swap the engine in `handleGenerate`:

```tsx
// new state next to `mode`:
const [progress, setProgress] = useState<ProgressInfo | null>(null)
const [modelError, setModelError] = useState<string | null>(null)
const realEngineRef = useRef<TransformersEngine | null>(null)
const [device, setDevice] = useState<'webgpu' | 'wasm' | null>(null)

const handleModeChange = async (m: Mode) => {
  setMode(m)
  setModelError(null)
  if (m === 'real' && !realEngineRef.current) {
    try {
      const engine = new TransformersEngine()
      await engine.prepare((p) => setProgress(p))
      realEngineRef.current = engine
      setDevice(engine.device)
    } catch (err) {
      setModelError(err instanceof Error ? err.message : String(err))
      setMode('sim')
    } finally {
      setProgress(null)
    }
  }
}

// in handleGenerate, replace the engine line:
const engine: PipelineEngine =
  mode === 'real' && realEngineRef.current
    ? realEngineRef.current
    : new SimulatedEngine(tokenizerRef.current)
```

Pass `onModeChange={handleModeChange}` to `PromptBar` and render
`<ModelStatus progress={progress} device={mode === 'real' ? device : null} error={modelError} onFallback={() => { setModelError(null); setMode('sim') }} />` in the header row. Import `TransformersEngine` and `ProgressInfo`.

- [ ] **Step 5: Verify** — `npx vitest run` all green; `npx tsc --noEmit` clean.

- [ ] **Step 6: MANUAL VERIFICATION (required, not CI):** `npm run dev`, toggle Real mode, confirm: download progress appears, device chip shows, generation streams token cycles live, logits/softmax bars show real values, temperature 0 vs 2 visibly changes outcomes, abort works mid-run, and killing the network before first toggle produces the error + fallback path. Fix worker API mismatches found here (see Step 1 caveat).

- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat: real-model mode via transformers.js worker"`

---

### Task 19: Playwright smoke test and README

**Files:**
- Create: `playwright.config.ts`, `e2e/smoke.spec.ts`, `README.md`
- Modify: `package.json` (scripts)

**Interfaces:**
- Consumes: test ids from Tasks 12/15/16 (`prompt-input`, `btn-generate`, `generated-token`, `stage-card`, `btn-pause`, `btn-step-fwd`)

- [ ] **Step 1: Config** (`playwright.config.ts`)

```ts
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: 'e2e',
  use: { baseURL: 'http://localhost:5173' },
  webServer: { command: 'npm run dev', url: 'http://localhost:5173', reuseExistingServer: true },
})
```

Add scripts to `package.json`: `"test": "vitest run"`, `"e2e": "playwright test"`. Run `npx playwright install chromium` once.

- [ ] **Step 2: Smoke test** (`e2e/smoke.spec.ts`)

```ts
import { expect, test } from '@playwright/test'

test('simulated run produces tokens and steps through stages', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()

  // playback auto-starts; a generated token eventually lands in the stream
  await expect(page.getByTestId('generated-token').first()).toBeVisible({ timeout: 15000 })

  // pause and manually step: exactly one stage card is active
  await page.getByTestId('btn-pause').click()
  await page.getByTestId('btn-step-fwd').click()
  const activeCards = page.locator('[data-testid="stage-card"][data-active="true"]')
  await expect(activeCards).toHaveCount(1)
})
```

- [ ] **Step 3: Verify** — `npx playwright test` → PASS (sim mode uses the fallback tokenizer if HF is unreachable, so this works offline).

- [ ] **Step 4: README** (`README.md`) — short: what the app teaches, screenshot placeholder is NOT allowed (omit images entirely), how to run (`npm install`, `npm run dev`), test commands (`npm test`, `npm run e2e`), the two modes (note the ~120 MB first-use download in real mode and the WebGPU/WASM fallback), and a pointer to the spec and this plan under `docs/superpowers/`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "test: e2e smoke test; docs: README"`

---

## Post-plan checks

- Full suite: `npx vitest run && npx tsc --noEmit && npx playwright test`.
- Manual: both modes end-to-end (Task 18 Step 6 checklist).
- Deferred (spec's out-of-scope list): attention deep-dive, chat templating, trace archive, per-token distribution hover, mobile layout.
