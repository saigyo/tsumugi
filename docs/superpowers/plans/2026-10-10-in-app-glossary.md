# In-app glossary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One glossary of 12 concept entries, opened from underlined terms, ⓘ chips and card headings, and from a global "?" button, shown in a non-modal side drawer with an optional "In this run" block computed from the active run.

**Architecture:** A new `src/glossary/` module: entry ids as a constant list (`types.ts`), one TSX file per entry, a registry, a zustand store for the open view and Back history, a `runSummary` builder over the trace, and three components (`Term`, `GlossaryDrawer`, `HelpButton`). `App` mounts the drawer once and passes it the run summary. Existing `title`-tooltip callouts become `Term` chips that open an entry at a section.

**Tech Stack:** React 19, TypeScript 6, zustand 5, vitest 4 + Testing Library (jsdom), Playwright, oxlint. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-10-in-app-glossary-design.md`

## Global Constraints

- No new npm dependencies.
- Entry ids (kebab-case, exactly these 12): `how-to-read`, `token`, `embedding`, `residual-stream`, `edit`, `layer`, `attention`, `mlp`, `logit`, `softmax`, `sampling`, `modes`.
- Stages and index order: `tokenize`, `embed`, `layers`, `logits`, `softmax`, `sample`, then `general`; `how-to-read` is listed first, above all groups.
- Stage labels: Tokenizer, Embeddings, Layers, Logits, Softmax, Sampler, General.
- Summaries are one plain sentence: end with a full stop and contain no other `. `.
- Drawer: right edge, about 380 px wide, full height, non-modal; full width below 600 px.
- Hint storage key: `tsumugi.glossaryHintDismissed` (value `'1'`); every storage access wrapped in try/catch.
- Hint copy: "New here? Start with *How to read Tsumugi*".
- Components and non-component exports never share a file (oxlint `react/only-export-components`).
- Styles go in `src/index.css`, using the existing tokens (`--ai`, `--ai-wash`, `--card`, `--line`, …).
- Copy style: plain, direct sentences; the user prefers no mannered phrasing. Unicode glyphs as in the app (×, ·, →, ␣).
- Every commit message ends with the trailer line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` after a blank line.
- Run lint with `npx oxlint` (the `npm run lint` output is mangled by the rtk shell hook).

## Review Focus

1. **Drawer open during live playback** — the run summary changes on every tick; the drawer must not steal focus or re-scroll on re-render, only when the view changes. Pinned in Task 4 (`re-render with a new run keeps focus where it is`).
2. **Run cleared while the drawer is open** (Generate starts a new run, trace empties) — the "In this run" block disappears, nothing crashes. Pinned in Task 4 (`run becomes null`).
3. **Section that does not exist** in the target entry — the drawer opens the entry at the top, no error. Pinned in Task 4 (`unknown section`).
4. **Clicking the same term twice** — no duplicate Back entry. Pinned in Task 1 (`opening the open view again is a no-op`).
5. **Opener gone from the DOM at close** (playback moved to another card, the chip that opened the drawer unmounted) — closing must not throw and focus lands on the "?" button. Pinned in Task 1 (`close with a detached opener focuses the help button`).

---

### Task 1: Types, store and run summary

**Files:**
- Create: `src/glossary/types.ts`
- Create: `src/glossary/store.ts`
- Create: `src/glossary/runSummary.ts`
- Test: `src/glossary/store.test.ts`
- Test: `src/glossary/runSummary.test.ts`

**Interfaces:**
- Produces:
  - `GLOSSARY_IDS` (readonly tuple), `type GlossaryId`, `type GlossaryStage`, `interface GlossaryEntry`, `interface RunSummary` from `types.ts`.
  - `type GlossaryView = { kind: 'index' } | { kind: 'entry'; id: GlossaryId; section?: string }`, `type FocusTarget = HTMLElement | SVGElement`, `useGlossaryStore` with state `{ view: GlossaryView | null; history: GlossaryView[]; opener: FocusTarget | null }` and actions `openEntry(id, section?, opener?)`, `openIndex(opener?)`, `back()`, `close()`, plus `resetGlossaryStore()` for tests, from `store.ts`.
  - `runSummary(events: TraceEvent[], cursor: number): RunSummary | null` from `runSummary.ts`.

- [ ] **Step 1: Write `types.ts`** (types only; no test of its own)

```ts
// src/glossary/types.ts
import type { ReactNode } from 'react'
import type { GenParams, Mode, TokenInfo } from '../trace/types'

// The ids are a plain list, not derived from the entry objects: entries refer
// to each other (related, <Term>), so deriving the type from them would be
// circular. The registry test checks one entry per id.
export const GLOSSARY_IDS = [
  'how-to-read', 'token', 'embedding', 'residual-stream', 'edit', 'layer',
  'attention', 'mlp', 'logit', 'softmax', 'sampling', 'modes',
] as const
export type GlossaryId = typeof GLOSSARY_IDS[number]

export type GlossaryStage = 'general' | 'tokenize' | 'embed' | 'layers' | 'logits' | 'softmax' | 'sample'

// What "In this run" blocks get to work with: plain values derived from the
// active trace at the cycle the player shows.
export interface RunSummary {
  mode: Mode
  modelId: string
  params: GenParams
  tokens: TokenInfo[]                    // the prompt's tokens
  cycle: number                          // viewed cycle, 0-based
  cycles: number                         // cycles in the trace so far
  stream?: { seqLen: number; dims: number; layers: number }
  vocabSize?: number
  softmax?: { temperature: number; top: Array<{ text: string; logit?: number; prob: number }> }
}

export interface GlossaryEntry {
  id: GlossaryId
  title: string
  summary: string
  stage: GlossaryStage
  related: GlossaryId[]
  body: () => ReactNode
  inRun?: (run: RunSummary) => ReactNode | null
}
```

- [ ] **Step 2: Write the failing store test**

```ts
// src/glossary/store.test.ts
import { afterEach, expect, test } from 'vitest'
import { resetGlossaryStore, useGlossaryStore } from './store'

afterEach(() => { resetGlossaryStore(); document.body.innerHTML = '' })
const s = () => useGlossaryStore.getState()

test('opening an entry from closed sets the view and remembers the opener', () => {
  const btn = document.createElement('button')
  s().openEntry('edit', undefined, btn)
  expect(s().view).toEqual({ kind: 'entry', id: 'edit' })
  expect(s().history).toEqual([])
  expect(s().opener).toBe(btn)
})

test('opening another entry pushes the previous view; back pops it', () => {
  s().openIndex()
  s().openEntry('softmax', 'temperature')
  expect(s().history).toEqual([{ kind: 'index' }])
  s().openEntry('logit')
  expect(s().history).toHaveLength(2)
  s().back()
  expect(s().view).toEqual({ kind: 'entry', id: 'softmax', section: 'temperature' })
  s().back()
  expect(s().view).toEqual({ kind: 'index' })
  s().back()                                   // empty history: stays put
  expect(s().view).toEqual({ kind: 'index' })
})

test('navigating inside the drawer keeps the original opener', () => {
  const first = document.createElement('button')
  s().openEntry('edit', undefined, first)
  s().openEntry('mlp', undefined, document.createElement('button'))
  expect(s().opener).toBe(first)
})

test('opening the open view again is a no-op', () => {
  s().openEntry('edit')
  s().openEntry('edit')
  expect(s().history).toEqual([])
  s().openIndex()
  s().openIndex()
  expect(s().history).toEqual([{ kind: 'entry', id: 'edit' }])
})

test('close clears view and history and returns focus to the opener', () => {
  const btn = document.createElement('button')
  document.body.append(btn)
  s().openEntry('edit', undefined, btn)
  s().openEntry('mlp')
  s().close()
  expect(s().view).toBeNull()
  expect(s().history).toEqual([])
  expect(s().opener).toBeNull()
  expect(document.activeElement).toBe(btn)
})

test('close with a detached opener focuses the help button', () => {
  const help = document.createElement('button')
  help.dataset.testid = 'btn-help'
  document.body.append(help)
  const gone = document.createElement('button')   // never attached: playback moved on
  s().openEntry('edit', undefined, gone)
  expect(() => s().close()).not.toThrow()
  expect(document.activeElement).toBe(help)
})
```

- [ ] **Step 3: Run it to verify it fails**

Run: `npx vitest run src/glossary/store.test.ts`
Expected: FAIL — cannot resolve `./store`.

- [ ] **Step 4: Write `store.ts`**

```ts
// src/glossary/store.ts
import { create } from 'zustand'
import type { GlossaryId } from './types'

export type GlossaryView = { kind: 'index' } | { kind: 'entry'; id: GlossaryId; section?: string }
export type FocusTarget = HTMLElement | SVGElement

interface GlossaryState {
  view: GlossaryView | null          // null = drawer closed
  history: GlossaryView[]            // views to go Back to, oldest first
  opener: FocusTarget | null         // gets focus back on close
  openEntry: (id: GlossaryId, section?: string, opener?: FocusTarget) => void
  openIndex: (opener?: FocusTarget) => void
  back: () => void
  close: () => void
}

const sameView = (a: GlossaryView, b: GlossaryView) =>
  a.kind === b.kind && (a.kind === 'index' || (b.kind === 'entry' && a.id === b.id && a.section === b.section))

const initial = { view: null, history: [], opener: null }

export const useGlossaryStore = create<GlossaryState>()((set, get) => {
  const go = (next: GlossaryView, opener?: FocusTarget) => {
    const { view, history } = get()
    if (view === null) set({ view: next, history: [], opener: opener ?? null })
    else if (!sameView(view, next)) set({ view: next, history: [...history, view] })
  }
  return {
    ...initial,
    openEntry: (id, section, opener) => go(section ? { kind: 'entry', id, section } : { kind: 'entry', id }, opener),
    openIndex: (opener) => go({ kind: 'index' }, opener),
    back: () => {
      const { history } = get()
      if (history.length === 0) return
      set({ view: history[history.length - 1], history: history.slice(0, -1) })
    },
    close: () => {
      const { opener } = get()
      set(initial)
      // the opener may have unmounted while the drawer was open (playback moved
      // to another card) — fall back to the "?" button
      const target = opener?.isConnected ? opener
        : document.querySelector<HTMLElement>('[data-testid="btn-help"]')
      target?.focus()
    },
  }
})

export function resetGlossaryStore(): void {
  useGlossaryStore.setState(initial)
}
```

- [ ] **Step 5: Run the store test to verify it passes**

Run: `npx vitest run src/glossary/store.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 6: Write the failing run-summary test**

```ts
// src/glossary/runSummary.test.ts
import { expect, test } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../test/fixtures'
import { runSummary } from './runSummary'

// makeFixtureTrace(): 0 run-start, 1 tokenize, cycle 0 = 2..10 (embed 2, layers 3-5,
// attention 6, logits 7, softmax 8, sample 9, append 10), cycle 1 = 11..19, 20 run-end

test('no trace, no summary', () => {
  expect(runSummary([], -1)).toBeNull()
})

test('summarises the run at the viewed cycle', () => {
  const r = runSummary(makeFixtureTrace(), 8)!
  expect(r.mode).toBe('sim')
  expect(r.modelId).toBe('fixture')
  expect(r.params).toEqual({ temperature: 0.8, topK: 10, maxNewTokens: 2 })
  expect(r.tokens.map((t) => t.text)).toEqual(['The', ' cat'])
  expect(r.cycle).toBe(0)
  expect(r.cycles).toBe(2)
  expect(r.stream).toEqual({ seqLen: 2, dims: 576, layers: 3 })
  expect(r.vocabSize).toBe(49152)
  expect(r.softmax?.temperature).toBe(0.8)
  expect(r.softmax?.top[0]).toEqual({ text: ' sat', logit: 9.1, prob: 0.7 })
})

test('follows the cursor into the next cycle', () => {
  const events = makeFixtureTrace()
  const softmax1 = events.findIndex((e) => e.type === 'softmax' && e.cycle === 1)
  const r = runSummary(events, softmax1)!
  expect(r.cycle).toBe(1)
  expect(r.stream?.seqLen).toBe(3)
  expect(r.softmax?.top[0].text).toBe(' on')
})

test('before the first softmax there is no softmax block', () => {
  const r = runSummary(buildFixtureTrace({ temperature: 1 }), 2)!
  expect(r.softmax).toBeUndefined()
  expect(r.stream).toEqual({ seqLen: 2, dims: 576, layers: 3 })
})
```

- [ ] **Step 7: Run it to verify it fails**

Run: `npx vitest run src/glossary/runSummary.test.ts`
Expected: FAIL — cannot resolve `./runSummary`.

- [ ] **Step 8: Write `runSummary.ts`**

```ts
// src/glossary/runSummary.ts
import type { TraceEvent } from '../trace/types'
import { latestOfType } from '../viz/selectors'
import type { RunSummary } from './types'

// Plain values for "In this run" blocks: run-level facts from the whole trace,
// per-cycle facts at the cursor. Null when no run is loaded.
export function runSummary(events: TraceEvent[], cursor: number): RunSummary | null {
  const last = events.length - 1
  const start = latestOfType(events, last, 'run-start')
  if (!start) return null
  const at = Math.max(0, Math.min(cursor, last))
  let cycle = 0
  let maxCycle = -1
  events.forEach((e, i) => {
    if (!('cycle' in e)) return
    if (i <= at) cycle = e.cycle
    maxCycle = Math.max(maxCycle, e.cycle)
  })
  const tokenize = latestOfType(events, last, 'tokenize')
  const embed = latestOfType(events, at, 'embed')
  const layer = latestOfType(events, last, 'layer')
  const sm = latestOfType(events, at, 'softmax')
  const lg = latestOfType(events, at, 'logits')
  const logitOf = new Map((lg && sm && lg.cycle === sm.cycle ? lg.topK : []).map((c) => [c.id, c.logit]))
  return {
    mode: start.mode,
    modelId: start.modelId,
    params: start.params,
    tokens: tokenize?.tokens ?? [],
    cycle,
    cycles: maxCycle + 1,
    ...(embed ? { stream: { seqLen: embed.seqLen, dims: embed.dims, layers: layer?.total ?? 0 } } : {}),
    ...(start.vocabSize ? { vocabSize: start.vocabSize } : {}),
    ...(sm ? { softmax: { temperature: sm.temperature,
      top: sm.topK.map((c) => ({ text: c.text, logit: logitOf.get(c.id), prob: c.prob })) } } : {}),
  }
}
```

Note: when a candidate has no logit, `logit` is `undefined`; `toEqual` treats an `undefined` property as absent, so the test above holds.

- [ ] **Step 9: Run both tests, type-check, lint**

Run: `npx vitest run src/glossary && npx tsc -b && npx oxlint src/glossary`
Expected: PASS (10 tests), no type errors, no lint warnings.

- [ ] **Step 10: Commit**

```bash
git add src/glossary
git commit -m "feat(glossary): entry types, glossary store and run summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Term, Section, registry and the first six entries

**Files:**
- Create: `src/glossary/Term.tsx`
- Create: `src/glossary/Section.tsx`
- Create: `src/glossary/registry.ts`
- Create: `src/glossary/entries/how-to-read.tsx`, `token.tsx`, `embedding.tsx`, `residual-stream.tsx`, `edit.tsx`, `layer.tsx`
- Modify: `src/index.css` (append the term and section styles)
- Test: `src/glossary/Term.test.tsx`
- Test: `src/glossary/registry.test.tsx`

**Interfaces:**
- Consumes: `GlossaryEntry`, `GlossaryId`, `RunSummary`, `useGlossaryStore`, `resetGlossaryStore` (Task 1).
- Produces:
  - `Term({ id: GlossaryId; section?: string; variant?: 'text' | 'chip' | 'icon'; label?: string; testId?: string; children: ReactNode })` — renders `<button data-testid={testId ?? 'gl-term'} data-term={id} data-section={section}>`; tooltip `[data-testid="gl-tip"]`.
  - `Section({ id: string; title: string; children: ReactNode })` — renders `<section data-gl-section={id}>` with an `h4`.
  - From `registry.ts`: `ENTRIES: GlossaryEntry[]`, `getEntry(id: string): GlossaryEntry | undefined`, `STAGE_ORDER: GlossaryStage[]`, `STAGE_LABEL: Record<GlossaryStage, string>`, `indexGroups(): Array<{ stage: GlossaryStage; label: string; entries: GlossaryEntry[] }>` (excludes `how-to-read`, which the drawer lists first on its own).
  - A sample run summary for tests: `sampleRun(over?: Partial<RunSummary>): RunSummary` exported from `src/test/glossaryFixture.ts`.

- [ ] **Step 1: Write the test fixture**

```ts
// src/test/glossaryFixture.ts
import type { RunSummary } from '../glossary/types'

export function sampleRun(over: Partial<RunSummary> = {}): RunSummary {
  return {
    mode: 'sim', modelId: 'fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 8 },
    tokens: [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }],
    cycle: 2, cycles: 8,
    stream: { seqLen: 4, dims: 576, layers: 30 },
    vocabSize: 49152,
    softmax: { temperature: 0.8, top: [
      { text: ' sat', logit: 9.1, prob: 0.7 }, { text: ' ran', logit: 7.2, prob: 0.2 }, { text: ' was', logit: 5.0, prob: 0.1 },
    ] },
    ...over,
  }
}
```

- [ ] **Step 2: Write `Section.tsx`**

```tsx
// src/glossary/Section.tsx
import type { ReactNode } from 'react'

// A part of an entry body that term links and callout chips can scroll to.
export function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section data-gl-section={id} className="gl-section">
      <h4>{title}</h4>
      {children}
    </section>
  )
}
```

- [ ] **Step 3: Write the six entries**

```tsx
// src/glossary/entries/how-to-read.tsx
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const howToRead: GlossaryEntry = {
  id: 'how-to-read',
  title: 'How to read Tsumugi',
  summary: 'A short tour of the screen, from the prompt to the run archive.',
  stage: 'general',
  related: ['token', 'residual-stream', 'modes'],
  body: () => (
    <>
      <p>Tsumugi shows what a language model does to produce each new token. The screen reads top to bottom:</p>
      <ol>
        <li><strong>Prompt and mode.</strong> Type a prompt or click an example (examples start a run at once). The
          mode switch chooses between a simulated model and the real SmolLM2-135M running in your browser
          (see <Term id="modes">modes</Term>).</li>
        <li><strong>Generate.</strong> The model produces one <Term id="token">token</Term> per cycle. Every cycle runs
          the whole pipeline once.</li>
        <li><strong>Pipeline band.</strong> Six stage cards: Tokenizer → Embeddings → Layers → Logits → Softmax →
          Sampler. The highlighted card is the step on screen; click a card to jump to that step of the current
          token. The labels on the arrows are the shapes of the data passed between stages.</li>
        <li><strong>Detail panel.</strong> Below the controls: the card for the current step, with its actual
          numbers.</li>
        <li><strong>Playback.</strong> The run is recorded and replayed at a readable pace. Pause, step forward or
          back, or jump to live.</li>
        <li><strong>Run shelf.</strong> Finished runs are kept as chips. Pin them, export them, or compare two runs
          side by side.</li>
      </ol>
      <p className="gl-aside">Underlined words open their entry in this panel. The ⓘ next to each card title opens
        the entry for that stage.</p>
    </>
  ),
  inRun: (run) => run.cycles > 0
    ? <p>You’re viewing token {run.cycle + 1} of {run.cycles} of a {run.mode === 'real' ? 'real-model' : 'simulated'} run.</p>
    : null,
}
```

```tsx
// src/glossary/entries/token.tsx
import { visibleToken } from '../../viz/tokenText'
import type { GlossaryEntry } from '../types'

const SHOWN = 12

export const token: GlossaryEntry = {
  id: 'token',
  title: 'Token',
  summary: 'A piece of text, often a word or part of a word, that the model reads and writes as one unit.',
  stage: 'tokenize',
  related: ['embedding', 'sampling'],
  body: () => (
    <>
      <p>The model never sees letters. The tokenizer splits text into pieces from a fixed vocabulary of 49 152
        entries and replaces each piece with its id. Common words are one token; rare words are split into
        several.</p>
      <pre className="gl-formula">{`"The cat sat"  →  The | ␣cat | ␣sat  →  three ids`}</pre>
      <p>A leading space belongs to the token, shown as ␣: “␣cat” and “cat” are different vocabulary entries.
        Some ids are control markers rather than text, such as <code>{'<|im_end|>'}</code>, which ends a turn.</p>
      <p>Generation works on ids too: each cycle produces one new id, which the tokenizer turns back into
        text.</p>
    </>
  ),
  inRun: (run) => run.tokens.length > 0 ? (
    <p>Your prompt became {run.tokens.length} tokens:{' '}
      {run.tokens.slice(0, SHOWN).map((t) => visibleToken(t.text)).join(' | ')}
      {run.tokens.length > SHOWN ? ' | …' : ''}.</p>
  ) : null,
}
```

```tsx
// src/glossary/entries/embedding.tsx
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const embedding: GlossaryEntry = {
  id: 'embedding',
  title: 'Embedding',
  summary: 'The vector a token id is turned into: one learned row of the embedding matrix E.',
  stage: 'embed',
  related: ['token', 'residual-stream', 'logit'],
  body: () => (
    <>
      <p>The embedding step is a lookup, not a computation. Each token id selects one row of a matrix E with one
        row per vocabulary entry. Stacked up, the rows form the <Term id="residual-stream">residual stream</Term> that
        enters the first layer.</p>
      <pre className="gl-formula">{`x[i] = E[id_i]        E: [49 152 × 576]
x    = [T × 576]      one row per token`}</pre>
      <Section id="learned" title="Learned, not designed">
        <p>Every row starts random and is adjusted during training so that the rest of the network predicts well.
          Nobody chose what dimension 17 means. Tokens used in similar contexts end up with rows that point in
          similar directions, which is why the Embeddings card can list meaningful neighbours.</p>
      </Section>
      <Section id="position" title="No position here">
        <p>SmolLM2 adds no position vector at this stage: “cat” gets the same row wherever it appears. Position
          enters inside <Term id="attention">attention</Term>, through rotary position embeddings (RoPE) applied to
          queries and keys.</p>
      </Section>
      <Section id="tied" title="Tied with the readout">
        <p>The same 49 152 × 576 matrix is used again at the end: the final vector is compared against every row
          to score each token (see <Term id="logit">logit</Term>). The model learns one vocabulary geometry, not
          two. In the model config this is <code>tie_word_embeddings</code>.</p>
      </Section>
    </>
  ),
  inRun: (run) => run.stream
    ? <p>Your {run.stream.seqLen} tokens became {run.stream.seqLen} rows: x [{run.stream.seqLen} × {run.stream.dims}].</p>
    : null,
}
```

```tsx
// src/glossary/entries/residual-stream.tsx
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const residualStream: GlossaryEntry = {
  id: 'residual-stream',
  title: 'Residual stream',
  summary: 'The matrix of token vectors that flows through every layer, one row per token.',
  stage: 'layers',
  related: ['edit', 'layer', 'embedding'],
  body: () => (
    <>
      <p>The residual stream starts as the stacked <Term id="embedding">embeddings</Term>, one 576-number row per
        token. Each <Term id="layer">layer</Term> reads it and adds its <Term id="edit">edits</Term> back. Nothing
        ever replaces it, so the original embedding stays underneath everything the layers add.</p>
      <pre className="gl-formula">{`x₀  = embeddings                 [T × 576]
x₃₀ = x₀ + all 60 edits          same shape`}</pre>
      <p>After the last layer, only the last row is read out to score the next token (see{' '}
        <Term id="logit">logit</Term>).</p>
      <p className="gl-aside">Why “stream”: each row carries information forward from layer to layer, and any layer
        can read what earlier layers wrote into it.</p>
    </>
  ),
  inRun: (run) => run.stream
    ? <p>In this run the stream is [{run.stream.seqLen} × {run.stream.dims}] and passes through {run.stream.layers} layers.</p>
    : null,
}
```

```tsx
// src/glossary/entries/edit.tsx
import { VectorStrip } from '../../viz/VectorStrip'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

const x = [0.8, -0.3, 0.1, 0.6, -0.9, 0.2, 0.4, -0.1]
const dx = [0.1, 0.4, -0.2, 0.0, 0.3, -0.1, 0.0, 0.2]
const sum = x.map((v, i) => v + dx[i])
const CELL = 14

export const edit: GlossaryEntry = {
  id: 'edit',
  title: 'Edit',
  summary: 'The vector a block adds to a token’s vector in the residual stream.',
  stage: 'layers',
  related: ['residual-stream', 'attention', 'mlp'],
  body: () => (
    <>
      <p>Each <Term id="layer">layer</Term> changes every token’s vector twice, once in{' '}
        <Term id="attention">attention</Term> and once in the <Term id="mlp">MLP</Term>. Neither block overwrites
        the vector. Each one computes a new vector of the same size and the model adds it on top. That added
        vector is the edit.</p>
      <div className="gl-figure" data-testid="gl-edit-figure">
        <VectorStrip cells={x} cellW={CELL} cellH={CELL} ariaLabel="x before the edit" />
        <span>+</span>
        <VectorStrip cells={dx} cellW={CELL} cellH={CELL} ariaLabel="the edit" />
        <span>=</span>
        <VectorStrip cells={sum} cellW={CELL} cellH={CELL} ariaLabel="x after the edit" />
      </div>
      <pre className="gl-formula">{`x ← x + Attention(norm(x))
x ← x + MLP(norm(x))`}</pre>
      <p>Because every block only adds, the final vector is the embedding plus all 60 edits. An edit can also
        point against what is already there and partly cancel it.</p>
      <p className="gl-aside">In papers: <em>block output</em>, <em>write to the residual stream</em>, or{' '}
        <em>residual update</em>.</p>
    </>
  ),
  inRun: (run) => run.stream ? (
    <p>Each layer adds two edits of [{run.stream.seqLen} × {run.stream.dims}] to your stream, one from attention
      and one from the MLP: {run.stream.layers * 2} in total.</p>
  ) : null,
}
```

```tsx
// src/glossary/entries/layer.tsx
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const layer: GlossaryEntry = {
  id: 'layer',
  title: 'Layer',
  summary: 'One transformer block: attention followed by an MLP, each adding an edit to the residual stream.',
  stage: 'layers',
  related: ['attention', 'mlp', 'edit'],
  body: () => (
    <>
      <p>SmolLM2-135M stacks 30 layers. They all have the same shape but their own weights. Each one does two
        things in order:</p>
      <pre className="gl-formula">{`x ← x + Attention(norm(x))    mixes information between tokens
x ← x + MLP(norm(x))          works on each token on its own`}</pre>
      <p><Term id="attention">Attention</Term> lets each token gather information from earlier tokens. The{' '}
        <Term id="mlp">MLP</Term> then processes each token’s vector separately. Both add their result to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <p>The <code>norm</code> is RMSNorm: it rescales each vector to a standard size before the block reads it, so
        no block has to cope with vectors that grow as edits pile up.</p>
    </>
  ),
  inRun: (run) => run.stream && run.stream.layers > 0
    ? <p>This run’s model has {run.stream.layers} layers.</p>
    : null,
}
```

- [ ] **Step 4: Write `registry.ts`** (first six entries; Task 3 adds the rest)

```ts
// src/glossary/registry.ts
import { edit } from './entries/edit'
import { embedding } from './entries/embedding'
import { howToRead } from './entries/how-to-read'
import { layer } from './entries/layer'
import { residualStream } from './entries/residual-stream'
import { token } from './entries/token'
import type { GlossaryEntry, GlossaryId, GlossaryStage } from './types'

export const ENTRIES: GlossaryEntry[] = [howToRead, token, embedding, residualStream, edit, layer]

const BY_ID = new Map<string, GlossaryEntry>(ENTRIES.map((e) => [e.id, e]))

// Looked up at render time (never at module load): entries import Term, and
// Term imports this module.
export const getEntry = (id: string): GlossaryEntry | undefined => BY_ID.get(id)

export const STAGE_ORDER: GlossaryStage[] = ['tokenize', 'embed', 'layers', 'logits', 'softmax', 'sample', 'general']

export const STAGE_LABEL: Record<GlossaryStage, string> = {
  tokenize: 'Tokenizer', embed: 'Embeddings', layers: 'Layers', logits: 'Logits',
  softmax: 'Softmax', sample: 'Sampler', general: 'General',
}

const FIRST: GlossaryId = 'how-to-read'

// The index below "How to read Tsumugi": entries grouped by stage in pipeline
// order, registry order within a group, empty groups left out.
export function indexGroups(): Array<{ stage: GlossaryStage; label: string; entries: GlossaryEntry[] }> {
  return STAGE_ORDER
    .map((stage) => ({ stage, label: STAGE_LABEL[stage], entries: ENTRIES.filter((e) => e.stage === stage && e.id !== FIRST) }))
    .filter((g) => g.entries.length > 0)
}
```

- [ ] **Step 5: Write the failing Term test**

```tsx
// src/glossary/Term.test.tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { resetGlossaryStore, useGlossaryStore } from './store'
import { Term } from './Term'
import type { GlossaryId } from './types'

afterEach(() => { cleanup(); resetGlossaryStore() })

test('hover shows the summary, leaving hides it', () => {
  render(<p>each block adds an <Term id="edit">edit</Term></p>)
  const term = screen.getByTestId('gl-term')
  expect(screen.queryByTestId('gl-tip')).toBeNull()
  fireEvent.mouseEnter(term)
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('The vector a block adds')
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('Click to read more.')
  expect(term).toHaveAttribute('aria-describedby', screen.getByTestId('gl-tip').id)
  fireEvent.mouseLeave(term)
  expect(screen.queryByTestId('gl-tip')).toBeNull()
})

test('keyboard focus shows the summary too', () => {
  render(<Term id="edit">edit</Term>)
  fireEvent.focus(screen.getByTestId('gl-term'))
  expect(screen.getByTestId('gl-tip')).toBeInTheDocument()
  fireEvent.blur(screen.getByTestId('gl-term'))
  expect(screen.queryByTestId('gl-tip')).toBeNull()
})

test('click opens the entry at the section and remembers the button as opener', () => {
  render(<Term id="embedding" section="learned" variant="chip" testId="embed-callout">ⓘ learned, not designed</Term>)
  const chip = screen.getByTestId('embed-callout')
  expect(chip).toHaveClass('embed-callout')
  expect(chip.dataset.term).toBe('embedding')
  expect(chip.dataset.section).toBe('learned')
  fireEvent.click(chip)
  const s = useGlossaryStore.getState()
  expect(s.view).toEqual({ kind: 'entry', id: 'embedding', section: 'learned' })
  expect(s.opener).toBe(chip)
  expect(screen.queryByTestId('gl-tip')).toBeNull()   // the tip closes when the drawer opens
})

test('icon variant carries an accessible name', () => {
  render(<Term id="token" variant="icon" label="About this stage">ⓘ</Term>)
  expect(screen.getByRole('button', { name: 'About this stage' })).toBeInTheDocument()
})

test('an id without an entry renders plain text, not a button', () => {
  render(<Term id={'nope' as GlossaryId}>plain words</Term>)
  expect(screen.queryByRole('button')).toBeNull()
  expect(screen.getByText('plain words')).toBeInTheDocument()
})
```

- [ ] **Step 6: Run it to verify it fails**

Run: `npx vitest run src/glossary/Term.test.tsx`
Expected: FAIL — cannot resolve `./Term`.

- [ ] **Step 7: Write `Term.tsx`**

```tsx
// src/glossary/Term.tsx
import { useId, useState, type ReactNode } from 'react'
import { getEntry } from './registry'
import { useGlossaryStore } from './store'
import type { GlossaryId } from './types'

const CLASS = { text: 'gl-term', chip: 'embed-callout gl-term gl-term-chip', icon: 'gl-term-icon' } as const

// A word in a caption (or a callout chip, or a heading's ⓘ) that shows the
// entry's summary on hover or focus and opens the entry on click.
export function Term({ id, section, variant = 'text', label, testId, children }: {
  id: GlossaryId
  section?: string
  variant?: 'text' | 'chip' | 'icon'
  label?: string
  testId?: string
  children: ReactNode
}) {
  const openEntry = useGlossaryStore((s) => s.openEntry)
  const [tip, setTip] = useState(false)
  const tipId = useId()
  const entry = getEntry(id)
  if (!entry) return <span>{children}</span>
  return (
    <span className="gl-term-wrap">
      <button type="button" className={CLASS[variant]} data-testid={testId ?? 'gl-term'}
        data-term={id} data-section={section} aria-label={label}
        aria-describedby={tip ? tipId : undefined}
        onMouseEnter={() => setTip(true)} onMouseLeave={() => setTip(false)}
        onFocus={() => setTip(true)} onBlur={() => setTip(false)}
        onClick={(e) => { setTip(false); openEntry(entry.id, section, e.currentTarget) }}>
        {children}
      </button>
      {tip && (
        <span role="tooltip" id={tipId} data-testid="gl-tip" className="gl-tip">
          {entry.summary} <em>Click to read more.</em>
        </span>
      )}
    </span>
  )
}
```

- [ ] **Step 8: Run the Term test to verify it passes**

Run: `npx vitest run src/glossary/Term.test.tsx`
Expected: PASS (5 tests).

- [ ] **Step 9: Write the registry test** (it covers the six entries now; Task 3 extends it)

```tsx
// src/glossary/registry.test.tsx
import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { sampleRun } from '../test/glossaryFixture'
import { ENTRIES, getEntry, indexGroups, STAGE_ORDER } from './registry'

afterEach(cleanup)

// raw sources of every entry file, for the inline <Term id section> check
const SOURCES = import.meta.glob('./entries/*.tsx', { query: '?raw', import: 'default', eager: true }) as Record<string, string>

const sectionsOf = (id: string): string[] => {
  const entry = getEntry(id)
  if (!entry) return []
  const { container } = render(<>{entry.body()}</>)
  const ids = [...container.querySelectorAll('[data-gl-section]')].map((el) => el.getAttribute('data-gl-section')!)
  cleanup()
  return ids
}

test('ids are unique and every entry has a known stage', () => {
  const ids = ENTRIES.map((e) => e.id)
  expect(new Set(ids).size).toBe(ids.length)
  for (const e of ENTRIES) expect(STAGE_ORDER).toContain(e.stage)
})

test('every summary is one sentence', () => {
  for (const e of ENTRIES) {
    expect(e.summary, e.id).toMatch(/\.$/)
    expect(e.summary.slice(0, -1), e.id).not.toMatch(/\. /)
  }
})

test('every body renders', () => {
  for (const e of ENTRIES) {
    const { container } = render(<>{e.body()}</>)
    expect(container.textContent?.length, e.id).toBeGreaterThan(80)
    cleanup()
  }
})

test('every inRun renders with a sample run and stays quiet without the data it needs', () => {
  const bare = sampleRun({ tokens: [], cycles: 0, stream: undefined, vocabSize: undefined, softmax: undefined })
  for (const e of ENTRIES) {
    if (!e.inRun) continue
    const { container } = render(<>{e.inRun(sampleRun())}</>)
    expect(container.textContent?.length, e.id).toBeGreaterThan(10)
    cleanup()
    const quiet = render(<>{e.inRun(bare)}</>)
    // entries whose block only needs mode/params may still speak; none may throw
    expect(quiet.container, e.id).toBeDefined()
    cleanup()
  }
})

test('inline <Term id section> links point at sections that exist', () => {
  const links = Object.values(SOURCES).flatMap((src) =>
    [...src.matchAll(/<Term id="([\w-]+)" section="([\w-]+)"/g)].map((m) => ({ id: m[1], section: m[2] })))
  for (const { id, section } of links) expect(sectionsOf(id), `${id} § ${section}`).toContain(section)
})

test('the index lists stages in pipeline order and leaves out how-to-read', () => {
  const groups = indexGroups()
  const order = groups.map((g) => STAGE_ORDER.indexOf(g.stage))
  expect(order).toEqual([...order].sort((a, b) => a - b))
  expect(groups.flatMap((g) => g.entries.map((e) => e.id))).not.toContain('how-to-read')
  expect(groups[0].label).toBe('Tokenizer')
})
```

Note: the `toContain` check on inline links guards future edits; with the Task 2 entries it may find no links, which is fine.

- [ ] **Step 10: Append term and section styles to `src/index.css`**

```css
/* ── Glossary: terms, tooltips, sections ─────────────────────────────── */
.gl-term-wrap { position: relative; display: inline; }
.gl-term {
  font: inherit; color: inherit; background: none; border: 0; border-radius: 0; padding: 0;
  border-bottom: 1px dotted var(--ai); cursor: help;
}
.gl-term:hover:not(:disabled) { background: var(--ai-wash); border-color: var(--ai); }
.gl-term-chip { font-size: inherit; color: var(--ai-deep); border-bottom-color: var(--ai-soft); }
.gl-term-icon {
  font: inherit; font-size: .8em; color: var(--ai); background: none; border: 0; padding: 0 .2rem;
  margin-left: .3rem; cursor: help; vertical-align: .1em;
}
.gl-term-icon:hover:not(:disabled) { background: var(--ai-wash); }
.gl-tip {
  position: absolute; left: 0; top: calc(100% + 4px); z-index: 30; width: max-content; max-width: 280px;
  background: var(--ink); color: var(--card); font-size: .78rem; font-weight: 400; line-height: 1.35;
  padding: .35rem .55rem; border-radius: 5px; box-shadow: 0 2px 8px rgba(43, 43, 49, .2); pointer-events: none;
}
.gl-tip em { color: var(--ai-wash-strong); font-style: normal; }
.gl-section { margin-top: .9rem; border-radius: 6px; transition: background-color .6s; }
.gl-section h4 { margin: 0 0 .3rem; font-size: .9rem; color: var(--ai-deep); }
.gl-section.gl-flash { background: var(--shu-wash); }
.gl-figure { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin: .6rem 0; font-family: var(--mono); }
.gl-formula {
  font-family: var(--mono); font-size: .78rem; color: var(--ai-deep); background: var(--ai-wash);
  padding: .4rem .6rem; border-radius: 4px; overflow-x: auto; white-space: pre;
}
.gl-aside { font-size: .8rem; color: var(--ink-soft); }
@media (prefers-reduced-motion: reduce) { .gl-section { transition: none; } }
```

- [ ] **Step 11: Run all glossary tests, type-check, lint**

Run: `npx vitest run src/glossary && npx tsc -b && npx oxlint src/glossary src/test`
Expected: PASS, no type errors, no lint warnings.

- [ ] **Step 12: Commit**

```bash
git add src/glossary src/test/glossaryFixture.ts src/index.css
git commit -m "feat(glossary): Term links, sections, registry and the first six entries

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The remaining six entries and the callout targets

**Files:**
- Create: `src/glossary/entries/attention.tsx`, `mlp.tsx`, `logit.tsx`, `softmax.tsx`, `sampling.tsx`, `modes.tsx`
- Create: `src/glossary/callouts.ts`
- Modify: `src/glossary/registry.ts` (import and list all 12)
- Modify: `src/glossary/registry.test.tsx` (completeness and callout-section tests)

**Interfaces:**
- Consumes: `Term`, `Section`, `GlossaryEntry`, `leadRatio` from `src/viz/softmaxText.ts`, `thousands` from `src/viz/selectors.ts`, `visibleToken` from `src/viz/tokenText.ts`.
- Produces: from `callouts.ts`: `interface CalloutTarget { label: string; id: GlossaryId; section: string }`, `EMBEDDING_CALLOUTS: CalloutTarget[]`, `logitsCallouts(shown: number, vocab: string): CalloutTarget[]`, `SOFTMAX_CALLOUTS: CalloutTarget[]`. Task 6 renders these.

- [ ] **Step 1: Write the failing completeness and callout tests** (append to `registry.test.tsx`)

```tsx
import { GLOSSARY_IDS } from './types'
import { EMBEDDING_CALLOUTS, logitsCallouts, SOFTMAX_CALLOUTS } from './callouts'

test('the registry holds exactly one entry per id, in the declared order', () => {
  expect(ENTRIES.map((e) => e.id)).toEqual([...GLOSSARY_IDS])
})

test('every related id has an entry', () => {
  for (const e of ENTRIES) for (const r of e.related) expect(getEntry(r), `${e.id} → ${r}`).toBeDefined()
})

test('every callout chip points at a section that exists', () => {
  for (const c of [...EMBEDDING_CALLOUTS, ...logitsCallouts(3, '49 152'), ...SOFTMAX_CALLOUTS])
    expect(sectionsOf(c.id), `${c.label} → ${c.id} § ${c.section}`).toContain(c.section)
})
```

Merge the new imports with the existing import lines at the top of the file.

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/glossary/registry.test.tsx`
Expected: FAIL — cannot resolve `./callouts`.

- [ ] **Step 3: Write the six entries**

```tsx
// src/glossary/entries/attention.tsx
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const attention: GlossaryEntry = {
  id: 'attention',
  title: 'Attention',
  summary: 'The step where each token gathers information from earlier tokens, weighted by how relevant they are.',
  stage: 'layers',
  related: ['layer', 'edit', 'mlp'],
  body: () => (
    <>
      <p>For each position, attention computes a weight for every position up to and including itself, never a
        later one. Its output is the weighted average of those positions’ value vectors, added to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <pre className="gl-formula">{`weights = softmax(q · kᵀ / √d)     one row per position, sums to 1
output  = weights · v`}</pre>
      <p>Each layer runs several heads in parallel, each with its own weights. SmolLM2-135M has 9 heads in each of
        its 30 layers: 270 in total.</p>
      <p>In a heatmap, a row is the token doing the looking and a column is the token looked at. Darker means more
        weight, and every row adds up to 100%.</p>
      <Section id="roles" title="Head roles Tsumugi detects">
        <ul>
          <li><strong>Attention sink</strong> puts most of its weight on the first token, a resting place when nothing
            else is relevant.</li>
          <li><strong>Previous token</strong> looks one position back.</li>
          <li><strong>Induction</strong> finds an earlier occurrence of the current token and looks at what followed
            it, which lets the model continue repeated patterns.</li>
          <li><strong>Coreference</strong> looks from a pronoun to the noun it refers to, such as “it” → “horse”.</li>
          <li><strong>Distinctive</strong> marks a head whose pattern matches none of the common templates.</li>
        </ul>
      </Section>
    </>
  ),
  inRun: (run) => run.mode === 'real'
    ? <p>This run uses the real model: the heatmaps show heads detected in its actual attention weights.</p>
    : <p>This run is simulated: the heatmaps show handcrafted patterns, not a real model’s weights.</p>,
}
```

```tsx
// src/glossary/entries/mlp.tsx
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const mlp: GlossaryEntry = {
  id: 'mlp',
  title: 'MLP',
  summary: 'The per-token step in each layer: widen the vector, apply a nonlinearity, narrow it back.',
  stage: 'layers',
  related: ['layer', 'edit', 'attention'],
  body: () => (
    <>
      <p>MLP stands for multi-layer perceptron, also called the feed-forward network. It works on each token’s
        vector on its own, with no mixing between positions. Its result is added to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <pre className="gl-formula">{`n    = norm(x)                                  [576]
h    = SiLU(W_gate · n) ⊙ (W_up · n)            [1536]
edit = W_down · h                               [576]`}</pre>
      <p>The gate and up projections widen the vector to 1536 numbers; SiLU and the element-wise product let some
        of them through and suppress others; the down projection brings the result back to 576. This variant is
        called SwiGLU.</p>
      <p>About 60% of the model’s weights sit in its 30 MLPs. A common reading: the gate and up projections
        detect patterns in the vector, and the down projection writes the information associated with them.</p>
      <p className="gl-aside">Tsumugi does not show the MLP’s internals yet.</p>
    </>
  ),
}
```

```tsx
// src/glossary/entries/logit.tsx
import { thousands } from '../../viz/selectors'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const logit: GlossaryEntry = {
  id: 'logit',
  title: 'Logit',
  summary: 'The raw score the model gives each vocabulary token as the next token, where higher means more likely.',
  stage: 'logits',
  related: ['softmax', 'embedding', 'residual-stream'],
  body: () => (
    <>
      <p>After the last layer, the last row of the <Term id="residual-stream">residual stream</Term> is normalised
        and compared with every row of the embedding matrix E. Each comparison is a dot product: 576 products added
        up. The result is one score per vocabulary entry.</p>
      <pre className="gl-formula">{`logits = norm(x_last) · Eᵀ
         [1 × 576] · [576 × 49 152] → [49 152]`}</pre>
      <p>Scores have no fixed range and can be negative. <Term id="softmax">Softmax</Term> turns them into
        probabilities.</p>
      <Section id="tied" title="Tied with Embeddings">
        <p>This is the Embeddings matrix read the other way: there a row was looked up by id, here every row is
          scored against the final vector. Tied weights mean the model learns one vocabulary geometry, not two.</p>
      </Section>
      <Section id="similar-rows" title="Similar rows, similar scores">
        <p>Tokens whose rows of E point the same way get nearly the same dot product, which is why near-synonyms sit
          together near the top.</p>
      </Section>
      <Section id="top-k" title="Only the top candidates are kept">
        <p>The multiplication produces a score for every vocabulary entry. The trace keeps only the top-k, the
          candidates the sampler can choose from.</p>
      </Section>
    </>
  ),
  inRun: (run) => run.vocabSize
    ? <p>{thousands(run.vocabSize)} scores were computed for each token of this run; the trace kept the top {run.params.topK}.</p>
    : null,
}
```

```tsx
// src/glossary/entries/softmax.tsx
import { leadRatio } from '../../viz/softmaxText'
import { visibleToken } from '../../viz/tokenText'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const softmax: GlossaryEntry = {
  id: 'softmax',
  title: 'Softmax',
  summary: 'The function that turns scores into probabilities that add up to 100%.',
  stage: 'softmax',
  related: ['logit', 'sampling'],
  body: () => (
    <>
      <p>Softmax takes the <Term id="logit">scores</Term> of the top-k candidates, divides each by the temperature
        T, exponentiates, and divides each result by the sum of all of them.</p>
      <pre className="gl-formula">{`p_i = exp(logit_i / T) ÷ Σ_j exp(logit_j / T)`}</pre>
      <Section id="exp" title="Why exponentiate">
        <p>Scores can be negative and differ by any amount. <code>exp</code> makes them all positive and turns a
          difference of d points into a ratio of e^(d / T). At T = 1, one point ahead means about 2.7 times the
          probability; two points about 7.4 times.</p>
      </Section>
      <Section id="temperature" title="Temperature">
        <p>Dividing every score by T before exponentiating scales every lead. T below 1 exaggerates leads (sharper,
          more predictable output); T above 1 shrinks them (flatter, more random). T = 0 skips the draw and takes
          the top score: greedy decoding.</p>
      </Section>
      <Section id="top-k" title="Top-k only">
        <p>The sum in the denominator runs over the k candidates kept by top-k, so the percentages are shares among
          them. All other vocabulary tokens get nothing and can never be drawn.</p>
      </Section>
    </>
  ),
  inRun: (run) => {
    const T = run.softmax?.temperature ?? run.params.temperature
    const top = run.softmax?.top.slice(0, 3) ?? []
    return (
      <>
        <p>{T === 0
          ? 'This run uses T = 0: no draw, the top score is taken (greedy).'
          : `At this run’s T = ${T}, 1 point ahead is ${leadRatio(1, T)}× the probability, 2 points ${leadRatio(2, T)}×.`}</p>
        {top.length > 0 && (
          <p>Current top: {top.map((c) => `${visibleToken(c.text)} ${Math.round(c.prob * 100)}%`).join(', ')}.</p>
        )}
      </>
    )
  },
}
```

```tsx
// src/glossary/entries/sampling.tsx
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const sampling: GlossaryEntry = {
  id: 'sampling',
  title: 'Sampling',
  summary: 'Picking the next token from the probabilities, by a weighted random draw or by taking the top one.',
  stage: 'sample',
  related: ['softmax', 'token'],
  body: () => (
    <>
      <p>With top-k sampling, the k most likely candidates keep their <Term id="softmax">softmax</Term> shares and
        one is drawn at random, weighted by those shares. A candidate at 70% wins about 7 times in 10. That is why
        the same prompt can continue differently between runs.</p>
      <p>With T = 0 (greedy), the top candidate is always taken and the output is the same every time.</p>
      <p>The chosen <Term id="token">token</Term> is appended to the text and the whole pipeline runs again for the
        next one: one cycle per token.</p>
      <p className="gl-aside">The curated examples in simulated mode follow a script, so their outcome does not
        change with temperature.</p>
    </>
  ),
  inRun: (run) => (
    <p>This run: {run.params.temperature === 0 ? 'greedy (T = 0)' : `top-k with k = ${run.params.topK}, T = ${run.params.temperature}`},
      up to {run.params.maxNewTokens} new tokens.</p>
  ),
}
```

```tsx
// src/glossary/entries/modes.tsx
import type { GlossaryEntry } from '../types'

export const modes: GlossaryEntry = {
  id: 'modes',
  title: 'Simulated and real mode',
  summary: 'Simulated mode shows scripted, illustrative data; real mode runs the SmolLM2-135M model in your browser.',
  stage: 'general',
  related: ['how-to-read', 'attention'],
  body: () => (
    <>
      <p><strong>Simulated</strong> needs no download and starts at once. A small program produces tokens, scores
        and attention patterns that illustrate each stage. The curated examples follow a script.</p>
      <p><strong>Real</strong> downloads SmolLM2-135M-Instruct once (about 250 MB, quantized ONNX) and runs it in
        the browser on WebGPU, or on WebAssembly where WebGPU is missing. Nothing leaves your machine. Every number
        shown comes from the model, including attention weights and embedding rows.</p>
      <p className="gl-aside">Where real mode cannot show a value, the card says so (for example “schematic”).</p>
    </>
  ),
  inRun: (run) => (
    <p>This run used {run.mode === 'real' ? `the real model (${run.modelId})` : 'the simulated model'}.</p>
  ),
}
```

- [ ] **Step 4: Write `callouts.ts`**

```ts
// src/glossary/callouts.ts
import type { GlossaryId } from './types'

// The ⓘ chips on the cards: label as shown, and the entry section it opens.
export interface CalloutTarget { label: string; id: GlossaryId; section: string }

export const EMBEDDING_CALLOUTS: CalloutTarget[] = [
  { label: 'learned, not designed', id: 'embedding', section: 'learned' },
  { label: 'no position here', id: 'embedding', section: 'position' },
  { label: 'tied with Logits', id: 'embedding', section: 'tied' },
]

export const logitsCallouts = (shown: number, vocab: string): CalloutTarget[] => [
  { label: 'tied with Embeddings', id: 'logit', section: 'tied' },
  { label: 'similar rows, similar scores', id: 'logit', section: 'similar-rows' },
  { label: `${vocab} scores, ${shown} shown`, id: 'logit', section: 'top-k' },
]

export const SOFTMAX_CALLOUTS: CalloutTarget[] = [
  { label: 'why exponentiate', id: 'softmax', section: 'exp' },
  { label: 'temperature', id: 'softmax', section: 'temperature' },
  { label: 'top-k only', id: 'softmax', section: 'top-k' },
]
```

- [ ] **Step 5: List all 12 entries in `registry.ts`**

Replace the import block and the `ENTRIES` line:

```ts
import { attention } from './entries/attention'
import { edit } from './entries/edit'
import { embedding } from './entries/embedding'
import { howToRead } from './entries/how-to-read'
import { layer } from './entries/layer'
import { logit } from './entries/logit'
import { mlp } from './entries/mlp'
import { modes } from './entries/modes'
import { residualStream } from './entries/residual-stream'
import { sampling } from './entries/sampling'
import { softmax } from './entries/softmax'
import { token } from './entries/token'
import type { GlossaryEntry, GlossaryId, GlossaryStage } from './types'

export const ENTRIES: GlossaryEntry[] = [
  howToRead, token, embedding, residualStream, edit, layer, attention, mlp, logit, softmax, sampling, modes,
]
```

- [ ] **Step 6: Run the glossary tests, type-check, lint**

Run: `npx vitest run src/glossary && npx tsc -b && npx oxlint src/glossary`
Expected: PASS (registry test now 9 tests), no type errors, no lint warnings.

- [ ] **Step 7: Commit**

```bash
git add src/glossary
git commit -m "feat(glossary): attention, MLP, logit, softmax, sampling and modes entries; callout targets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The drawer

**Files:**
- Create: `src/glossary/GlossaryDrawer.tsx`
- Modify: `src/index.css` (append drawer styles)
- Test: `src/glossary/GlossaryDrawer.test.tsx`

**Interfaces:**
- Consumes: `useGlossaryStore`, `resetGlossaryStore`, `GlossaryView` (Task 1); `getEntry`, `indexGroups`, `STAGE_LABEL` (Task 2/3); `RunSummary`; `sampleRun` (Task 2).
- Produces: `GlossaryDrawer({ run }: { run: RunSummary | null })`. Test ids: `glossary-drawer`, `gl-heading`, `gl-back`, `gl-close`, `gl-in-run`, `gl-related`, `gl-all-entries`, `gl-index-row` (with `data-entry`), `gl-index-group` (with `data-stage`).

- [ ] **Step 1: Write the failing drawer test**

```tsx
// src/glossary/GlossaryDrawer.test.tsx
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { sampleRun } from '../test/glossaryFixture'
import { GlossaryDrawer } from './GlossaryDrawer'
import { getEntry } from './registry'
import { resetGlossaryStore, useGlossaryStore } from './store'

const scrollIntoView = vi.fn()
beforeEach(() => { Element.prototype.scrollIntoView = scrollIntoView })
afterEach(() => { cleanup(); resetGlossaryStore(); scrollIntoView.mockReset(); vi.useRealTimers() })
const open = (fn: () => void) => act(fn)

test('closed: renders nothing', () => {
  render(<GlossaryDrawer run={null} />)
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
})

test('entry view: title, stage chip, body, related links; heading gets focus', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  const drawer = screen.getByTestId('glossary-drawer')
  expect(drawer).toHaveAttribute('aria-label', 'Glossary')
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Edit')
  expect(drawer).toHaveTextContent('Layers')                      // stage chip
  expect(screen.getByTestId('gl-edit-figure')).toBeInTheDocument()
  expect(document.activeElement).toBe(screen.getByTestId('gl-heading'))
  expect(screen.getAllByTestId('gl-related').map((b) => b.textContent)).toEqual(['Residual stream', 'Attention', 'MLP'])
  expect(screen.queryByTestId('gl-in-run')).toBeNull()             // no run loaded
  expect(screen.queryByTestId('gl-back')).toBeNull()               // nothing to go back to
})

test('related link and Back navigate', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.click(screen.getAllByTestId('gl-related')[2])
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('MLP')
  fireEvent.click(screen.getByTestId('gl-back'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Edit')
})

test('In this run block shows the run’s numbers', () => {
  render(<GlossaryDrawer run={sampleRun()} />)
  open(() => useGlossaryStore.getState().openEntry('softmax'))
  const block = screen.getByTestId('gl-in-run')
  expect(block).toHaveTextContent('In this run')
  expect(block).toHaveTextContent('T = 0.8, 1 point ahead is 3.5×')
  expect(block).toHaveTextContent('␣sat 70%')
})

test('run becomes null: the block disappears, the entry stays', () => {
  const { rerender } = render(<GlossaryDrawer run={sampleRun()} />)
  open(() => useGlossaryStore.getState().openEntry('softmax'))
  rerender(<GlossaryDrawer run={null} />)
  expect(screen.queryByTestId('gl-in-run')).toBeNull()
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Softmax')
})

test('re-render with a new run keeps focus where it is', () => {
  const { rerender } = render(<><input data-testid="elsewhere" /><GlossaryDrawer run={sampleRun()} /></>)
  open(() => useGlossaryStore.getState().openEntry('softmax'))
  screen.getByTestId('elsewhere').focus()
  scrollIntoView.mockReset()
  rerender(<><input data-testid="elsewhere" /><GlossaryDrawer run={sampleRun({ cycle: 3 })} /></>)
  expect(document.activeElement).toBe(screen.getByTestId('elsewhere'))
  expect(scrollIntoView).not.toHaveBeenCalled()
})

test('a throwing inRun is skipped; the entry still renders', () => {
  const entry = getEntry('token')!
  const original = entry.inRun
  entry.inRun = () => { throw new Error('boom') }
  const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
  try {
    render(<GlossaryDrawer run={sampleRun()} />)
    open(() => useGlossaryStore.getState().openEntry('token'))
    expect(screen.getByTestId('gl-heading')).toHaveTextContent('Token')
    expect(screen.queryByTestId('gl-in-run')).toBeNull()
  } finally {
    entry.inRun = original
    spy.mockRestore()
  }
})

test('section link scrolls to the section and highlights it briefly', () => {
  vi.useFakeTimers()
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('softmax', 'temperature'))
  const section = screen.getByTestId('glossary-drawer').querySelector('[data-gl-section="temperature"]')!
  expect(scrollIntoView).toHaveBeenCalledTimes(1)
  expect(scrollIntoView.mock.contexts[0]).toBe(section)
  expect(section).toHaveClass('gl-flash')
  act(() => { vi.advanceTimersByTime(1600) })
  expect(section).not.toHaveClass('gl-flash')
})

test('unknown section: opens at the top without error', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('softmax', 'no-such-section'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Softmax')
  expect(scrollIntoView).not.toHaveBeenCalled()
})

test('Escape and × close the drawer', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.click(screen.getByTestId('gl-close'))
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
})

test('index: how-to-read first, then groups in pipeline order; a row opens its entry', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openIndex())
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
  const rows = screen.getAllByTestId('gl-index-row')
  expect(rows[0].dataset.entry).toBe('how-to-read')
  expect(screen.getAllByTestId('gl-index-group').map((g) => g.dataset.stage))
    .toEqual(['tokenize', 'embed', 'layers', 'logits', 'softmax', 'sample', 'general'])
  expect(rows[0]).toHaveTextContent('A short tour of the screen')
  fireEvent.click(rows.find((r) => r.dataset.entry === 'logit')!)
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Logit')
  fireEvent.click(screen.getByTestId('gl-back'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
})

test('“All entries” link opens the index from an entry', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.click(screen.getByTestId('gl-all-entries'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/glossary/GlossaryDrawer.test.tsx`
Expected: FAIL — cannot resolve `./GlossaryDrawer`.

- [ ] **Step 3: Write `GlossaryDrawer.tsx`**

```tsx
// src/glossary/GlossaryDrawer.tsx
import { Component, useEffect, useRef, type ReactNode } from 'react'
import { getEntry, indexGroups, STAGE_LABEL } from './registry'
import { useGlossaryStore, type GlossaryView } from './store'
import type { GlossaryEntry, RunSummary } from './types'

const FLASH_MS = 1500

// An entry's "In this run" block must never take the entry down with it.
class InRunBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? null : this.props.children }
}

function InRunBlock({ entry, run }: { entry: GlossaryEntry; run: RunSummary }) {
  const content = entry.inRun?.(run)
  if (content == null || content === false) return null
  return (
    <div data-testid="gl-in-run" className="gl-in-run">
      <div className="gl-in-run-label">In this run</div>
      {content}
    </div>
  )
}

const viewKey = (v: GlossaryView | null) =>
  v === null ? '' : v.kind === 'index' ? 'index' : `${v.id}#${v.section ?? ''}`

// The glossary panel: fixed to the right edge, non-modal, one view at a time
// (an entry or the index). Mounted once in App.
export function GlossaryDrawer({ run }: { run: RunSummary | null }) {
  const view = useGlossaryStore((s) => s.view)
  const canGoBack = useGlossaryStore((s) => s.history.length > 0)
  const { openEntry, openIndex, back, close } = useGlossaryStore.getState()
  const headingRef = useRef<HTMLHeadingElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const key = viewKey(view)

  // focus and scroll only when the view changes — never on a run re-render
  useEffect(() => {
    if (!view) return
    headingRef.current?.focus()
    if (view.kind !== 'entry' || !view.section) { bodyRef.current?.scrollTo?.({ top: 0 }); return }
    const section = view.section
    const el = [...(bodyRef.current?.querySelectorAll('[data-gl-section]') ?? [])]
      .find((s) => s.getAttribute('data-gl-section') === section)
    if (!el) return
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    el.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' })
    el.classList.add('gl-flash')
    const t = setTimeout(() => el.classList.remove('gl-flash'), FLASH_MS)
    return () => { clearTimeout(t); el.classList.remove('gl-flash') }
    // keyed on the view identity on purpose: `view` is a new object only when it changes
  }, [key])

  useEffect(() => {
    if (!view) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [view, close])

  if (!view) return null
  const entry = view.kind === 'entry' ? getEntry(view.id) : undefined

  return (
    <aside data-testid="glossary-drawer" className="gl-drawer" aria-label="Glossary">
      <div className="gl-drawer-bar">
        {canGoBack
          ? <button type="button" data-testid="gl-back" className="gl-link" onClick={back}>← Back</button>
          : <span />}
        <button type="button" data-testid="gl-close" className="gl-close" aria-label="Close glossary" onClick={close}>×</button>
      </div>
      <div ref={bodyRef} className="gl-drawer-body">
        {entry ? (
          <>
            <h2 ref={headingRef} tabIndex={-1} data-testid="gl-heading" className="gl-heading">
              {entry.title} <span className="gl-stage-chip">{STAGE_LABEL[entry.stage]}</span>
            </h2>
            <p className="gl-summary">{entry.summary}</p>
            {entry.body()}
            {run && entry.inRun && (
              <InRunBoundary key={entry.id}><InRunBlock entry={entry} run={run} /></InRunBoundary>
            )}
            {entry.related.length > 0 && (
              <div className="gl-related">
                <span className="gl-related-label">Related:</span>
                {entry.related.map((id) => {
                  const r = getEntry(id)
                  return r && (
                    <button key={id} type="button" data-testid="gl-related" className="gl-link"
                      onClick={() => openEntry(id)}>{r.title}</button>
                  )
                })}
              </div>
            )}
            <button type="button" data-testid="gl-all-entries" className="gl-link gl-all" onClick={() => openIndex()}>
              All entries
            </button>
          </>
        ) : (
          <>
            <h2 ref={headingRef} tabIndex={-1} data-testid="gl-heading" className="gl-heading">Glossary</h2>
            <IndexRow id="how-to-read" />
            {indexGroups().map((g) => (
              <section key={g.stage} data-testid="gl-index-group" data-stage={g.stage} className="gl-index-group">
                <h3>{g.label}</h3>
                {g.entries.map((e) => <IndexRow key={e.id} id={e.id} />)}
              </section>
            ))}
          </>
        )}
      </div>
    </aside>
  )
}

function IndexRow({ id }: { id: GlossaryEntry['id'] }) {
  const entry = getEntry(id)
  const openEntry = useGlossaryStore((s) => s.openEntry)
  if (!entry) return null
  return (
    <button type="button" data-testid="gl-index-row" data-entry={id} className="gl-index-row" onClick={() => openEntry(id)}>
      <span className="gl-index-title">{entry.title}</span>
      <span className="gl-index-summary">{entry.summary}</span>
    </button>
  )
}
```

Notes for the implementer:
- The section is found by comparing attributes rather than with a `[data-gl-section="…"]` selector, so ids never need CSS escaping (jsdom's `CSS.escape` support is not guaranteed).
- `scrollTo` and `matchMedia` are called with optional chaining because jsdom does not implement them.
- The `IndexRow` component lives in the same file and is not exported, which the lint rule allows.

- [ ] **Step 4: Run the drawer test to verify it passes**

Run: `npx vitest run src/glossary/GlossaryDrawer.test.tsx`
Expected: PASS (13 tests).

- [ ] **Step 5: Append drawer styles to `src/index.css`**

```css
/* ── Glossary drawer ─────────────────────────────────────────────────── */
.gl-drawer {
  position: fixed; top: 0; right: 0; bottom: 0; z-index: 40; width: 380px; max-width: 100vw;
  display: flex; flex-direction: column; background: var(--card); border-left: 1px solid var(--line-strong);
  box-shadow: -4px 0 16px rgba(43, 43, 49, .08); font-size: .9rem; line-height: 1.5;
}
@media (max-width: 600px) { .gl-drawer { width: 100vw; border-left: 0; } }
.gl-drawer-bar { display: flex; justify-content: space-between; align-items: center; padding: .5rem .8rem; border-bottom: 1px solid var(--line); }
.gl-drawer-body { overflow-y: auto; padding: .8rem 1.1rem 1.5rem; flex: 1; }
.gl-close { font-size: 1.1rem; line-height: 1; padding: .15rem .5rem; }
.gl-link { font: inherit; color: var(--ai); background: none; border: 0; padding: 0; cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }
.gl-link:hover:not(:disabled) { background: none; color: var(--ai-deep); }
.gl-heading { margin: 0 0 .2rem; font-size: 1.2rem; color: var(--ai-deep); outline: none; }
.gl-stage-chip { font-size: .7rem; font-weight: 400; color: var(--ink-soft); background: var(--dye-2); border-radius: 999px; padding: .05rem .5rem; vertical-align: .2em; margin-left: .3rem; }
.gl-summary { margin: 0 0 .6rem; color: var(--ink-soft); }
.gl-drawer-body ol, .gl-drawer-body ul { padding-left: 1.2rem; }
.gl-drawer-body li { margin-bottom: .35rem; }
.gl-in-run { margin-top: 1rem; background: var(--ai-wash); border-radius: 6px; padding: .5rem .7rem; }
.gl-in-run p { margin: .2rem 0; }
.gl-in-run-label { font-size: .72rem; letter-spacing: .05em; text-transform: uppercase; color: var(--ai); }
.gl-related { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: .3rem .8rem; align-items: baseline; }
.gl-related-label { color: var(--ink-soft); font-size: .82rem; }
.gl-all { display: block; margin-top: .8rem; font-size: .85rem; }
.gl-index-group h3 { font-size: .75rem; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-faint); margin: 1rem 0 .3rem; font-weight: 600; }
.gl-index-row {
  display: grid; gap: .05rem; width: 100%; text-align: left; border: 0; border-radius: 6px;
  background: none; padding: .35rem .5rem; margin: 0 -.5rem;
}
.gl-index-row:hover:not(:disabled) { background: var(--ai-wash); }
.gl-index-title { color: var(--ai-deep); font-weight: 600; }
.gl-index-summary { color: var(--ink-soft); font-size: .82rem; }
```

- [ ] **Step 6: Run all glossary tests, type-check, lint**

Run: `npx vitest run src/glossary && npx tsc -b && npx oxlint src/glossary`
Expected: PASS, no type errors, no lint warnings.

- [ ] **Step 7: Commit**

```bash
git add src/glossary src/index.css
git commit -m "feat(glossary): side drawer with entry view, In this run block and index

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: "?" button, first-visit hint and App wiring

**Files:**
- Create: `src/glossary/HelpButton.tsx`
- Modify: `src/App.tsx` (header, drawer mount, run summary)
- Modify: `src/index.css` (header and hint styles)
- Test: `src/glossary/HelpButton.test.tsx`
- Test: `src/App.test.tsx` (one new test)

**Interfaces:**
- Consumes: `useGlossaryStore` (Task 1), `runSummary` (Task 1), `GlossaryDrawer` (Task 4).
- Produces: `HelpButton()` with test ids `btn-help`, `glossary-hint`, `btn-hint-open`, `btn-hint-dismiss`; `HINT_KEY = 'tsumugi.glossaryHintDismissed'` (module-internal).

- [ ] **Step 1: Write the failing HelpButton test**

```tsx
// src/glossary/HelpButton.test.tsx
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { HelpButton } from './HelpButton'
import { resetGlossaryStore, useGlossaryStore } from './store'

const KEY = 'tsumugi.glossaryHintDismissed'
beforeEach(() => localStorage.clear())
afterEach(() => { cleanup(); resetGlossaryStore(); vi.restoreAllMocks() })

test('the ? button opens the index', () => {
  render(<HelpButton />)
  fireEvent.click(screen.getByTestId('btn-help'))
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'index' })
  expect(useGlossaryStore.getState().opener).toBe(screen.getByTestId('btn-help'))
})

test('first visit shows the hint; following it opens How to read and dismisses it for good', () => {
  render(<HelpButton />)
  expect(screen.getByTestId('glossary-hint')).toHaveTextContent('New here? Start with How to read Tsumugi')
  fireEvent.click(screen.getByTestId('btn-hint-open'))
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'how-to-read' })
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
  expect(localStorage.getItem(KEY)).toBe('1')
})

test('× dismisses the hint and it stays gone on the next visit', () => {
  const first = render(<HelpButton />)
  fireEvent.click(screen.getByTestId('btn-hint-dismiss'))
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
  first.unmount()
  render(<HelpButton />)
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
})

test('opening any entry elsewhere dismisses the hint', () => {
  render(<HelpButton />)
  act(() => useGlossaryStore.getState().openEntry('edit'))
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
  expect(localStorage.getItem(KEY)).toBe('1')
})

test('storage that throws: the hint still shows and dismissing does not error', () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
  render(<HelpButton />)
  expect(screen.getByTestId('glossary-hint')).toBeInTheDocument()
  expect(() => fireEvent.click(screen.getByTestId('btn-hint-dismiss'))).not.toThrow()
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/glossary/HelpButton.test.tsx`
Expected: FAIL — cannot resolve `./HelpButton`.

- [ ] **Step 3: Write `HelpButton.tsx`**

```tsx
// src/glossary/HelpButton.tsx
import { useEffect, useState } from 'react'
import { useGlossaryStore } from './store'

const HINT_KEY = 'tsumugi.glossaryHintDismissed'

function readDismissed(): boolean {
  try { return localStorage.getItem(HINT_KEY) === '1' } catch { return false }
}
function writeDismissed(): void {
  try { localStorage.setItem(HINT_KEY, '1') } catch { /* private window: the hint returns next visit */ }
}

// The header's "?" (opens the glossary index) and, on a first visit, a
// one-time pointer to "How to read Tsumugi".
export function HelpButton() {
  const [hint, setHint] = useState(() => !readDismissed())
  const isOpen = useGlossaryStore((s) => s.view !== null)
  const { openIndex, openEntry } = useGlossaryStore.getState()
  const dismiss = () => { setHint(false); writeDismissed() }

  // opening anything in the glossary counts as having found it
  useEffect(() => { if (isOpen && hint) dismiss() }, [isOpen, hint])

  return (
    <div className="gl-help">
      {hint && (
        <span data-testid="glossary-hint" className="gl-hint">
          New here? Start with{' '}
          <button type="button" data-testid="btn-hint-open" className="gl-link"
            onClick={(e) => openEntry('how-to-read', undefined, e.currentTarget)}>How to read Tsumugi</button>
          <button type="button" data-testid="btn-hint-dismiss" className="gl-hint-dismiss"
            aria-label="Dismiss hint" onClick={dismiss}>×</button>
        </span>
      )}
      <button type="button" data-testid="btn-help" className="gl-help-btn" aria-label="Open the glossary"
        onClick={(e) => openIndex(e.currentTarget)}>?</button>
    </div>
  )
}
```

Note: the hint's link button unmounts once the hint is dismissed, so `close()` falls back to `btn-help` for focus (Task 1 handles this).

- [ ] **Step 4: Run the HelpButton test to verify it passes**

Run: `npx vitest run src/glossary/HelpButton.test.tsx`
Expected: PASS (5 tests).

- [ ] **Step 5: Write the failing App test** (append to `src/App.test.tsx`; reuse its existing imports and render helper — read the top of the file first and match how other tests render `<App />`)

```tsx
test('the ? button opens the glossary drawer, which closes again', async () => {
  render(<App />)
  fireEvent.click(screen.getByTestId('btn-help'))
  expect(screen.getByTestId('glossary-drawer')).toBeInTheDocument()
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
  fireEvent.click(screen.getByTestId('gl-close'))
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
})
```

Add `resetGlossaryStore()` from `./glossary/store` to the file's `afterEach` so the drawer state does not leak between tests.

- [ ] **Step 6: Run it to verify it fails**

Run: `npx vitest run src/App.test.tsx`
Expected: the new test FAILS — no element with test id `btn-help`.

- [ ] **Step 7: Wire `App.tsx`**

Add imports:

```tsx
import { useMemo } from 'react'          // merge into the existing 'react' import
import { GlossaryDrawer } from './glossary/GlossaryDrawer'
import { HelpButton } from './glossary/HelpButton'
import { runSummary } from './glossary/runSummary'
```

Inside `App`, after the `cursor` line:

```tsx
  const glossaryRun = useMemo(() => runSummary(events, cursor), [events, cursor])
```

Replace the `<h1>…</h1>` line with:

```tsx
      <header className="app-header">
        <h1><span className="app-mark">紬</span> Tsumugi <span className="app-subtitle">LLM Pipeline Visualizer</span></h1>
        <HelpButton />
      </header>
```

Add the drawer as the last child of `<div className="app">`, after the compare/pipeline block:

```tsx
      <GlossaryDrawer run={glossaryRun} />
```

- [ ] **Step 8: Append header and hint styles to `src/index.css`**

```css
/* ── Header with the glossary button ─────────────────────────────────── */
.app-header { display: flex; align-items: center; flex-wrap: wrap; gap: .5rem 1rem; margin: 0 0 1rem; }
.app-header h1 { margin: 0; }
.gl-help { margin-left: auto; display: flex; align-items: center; gap: .6rem; }
.gl-help-btn {
  width: 1.9rem; height: 1.9rem; padding: 0; border-radius: 999px; font-weight: 700;
  color: var(--ai-deep); border-color: var(--ai-soft);
}
.gl-hint {
  font-size: .82rem; color: var(--ink-soft); background: var(--shu-wash); border: 1px solid var(--shu-soft);
  border-radius: 6px; padding: .2rem .3rem .2rem .6rem; display: inline-flex; align-items: center; gap: .3rem;
}
.gl-hint-dismiss { border: 0; background: none; padding: 0 .3rem; color: var(--ink-soft); font-size: 1rem; line-height: 1; }
```

- [ ] **Step 9: Run the full unit suite, type-check, lint**

Run: `npx vitest run && npx tsc -b && npx oxlint`
Expected: all tests PASS (existing ones unchanged), no type errors, lint shows only the two known scaffold warnings, nothing new.

- [ ] **Step 10: Commit**

```bash
git add src/glossary src/App.tsx src/App.test.tsx src/index.css
git commit -m "feat(glossary): ? button with first-visit hint; drawer mounted in App with the run summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Replace the tooltips and link terms in the cards

**Files:**
- Modify: `src/viz/EmbeddingLookup.tsx` (callout chips)
- Modify: `src/viz/LogitsReadout.tsx` (callout chips)
- Modify: `src/viz/details/SoftmaxDetail.tsx` (callout chips, heading ⓘ, caption term)
- Modify: `src/viz/softmaxText.ts` (remove `softmaxCallouts`)
- Modify: `src/viz/ResidualStreamDiagram.tsx` (box links, caption terms)
- Modify: `src/viz/details/TokenizerDetail.tsx`, `EmbeddingsDetail.tsx`, `LayersDetail.tsx`, `LogitsDetail.tsx`, `SamplerDetail.tsx` (heading ⓘ, caption terms)
- Modify: `src/index.css` (diagram tooltip position)
- Test: `src/viz/details/EmbeddingsDetail.test.tsx`, `src/viz/details/SoftmaxDetail.test.tsx`, `src/viz/LogitsReadout.test.tsx` (callout tests rewritten)
- Test: `src/viz/ResidualStreamDiagram.test.tsx` (new)
- Test: `src/viz/DetailPanel.test.tsx` (heading ⓘ and caption terms)

**Interfaces:**
- Consumes: `Term` (Task 2), `EMBEDDING_CALLOUTS`, `logitsCallouts`, `SOFTMAX_CALLOUTS` (Task 3), `getEntry`, `useGlossaryStore`, `resetGlossaryStore`.
- Produces: no new exports. Chips keep `data-testid="embed-callout"` and their visible labels (`ⓘ learned, not designed`, …). Heading buttons use `data-testid="stage-info"`. Diagram boxes use `data-testid="rs-term-attention"` and `rs-term-mlp`.

- [ ] **Step 1: Rewrite the three callout tests to the new behaviour (failing)**

In `src/viz/details/EmbeddingsDetail.test.tsx`, replace the test `'three callouts carry their explanations'` with:

```tsx
test('three callouts open the Embedding entry at their sections', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  const callouts = screen.getAllByTestId('embed-callout')
  expect(callouts.map((c) => c.textContent)).toEqual(['ⓘ learned, not designed', 'ⓘ no position here', 'ⓘ tied with Logits'])
  expect(callouts.every((c) => c.getAttribute('title') === null)).toBe(true)
  fireEvent.click(callouts[1])
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'embedding', section: 'position' })
})
```

and add to the imports: `import { resetGlossaryStore, useGlossaryStore } from '../../glossary/store'`, and `resetGlossaryStore()` to the file's `afterEach`.

In `src/viz/details/SoftmaxDetail.test.tsx`, replace the test `'callouts explain exponentiation, temperature and the top-k cut, with the ratio computed from T'` with:

```tsx
test('callouts open the Softmax entry at exp, temperature and top-k', () => {
  render(<SoftmaxDetail events={makeFixtureTrace()} cursor={8} />)
  const callouts = screen.getAllByTestId('embed-callout')
  expect(callouts.map((c) => c.textContent)).toEqual(['ⓘ why exponentiate', 'ⓘ temperature', 'ⓘ top-k only'])
  expect(callouts.map((c) => c.dataset.section)).toEqual(['exp', 'temperature', 'top-k'])
  fireEvent.click(callouts[1])
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'softmax', section: 'temperature' })
})
```

and change the first import line to `import { cleanup, fireEvent, render, screen } from '@testing-library/react'`, add `import { resetGlossaryStore, useGlossaryStore } from '../../glossary/store'`, and make `afterEach` call both `cleanup()` and `resetGlossaryStore()`.

In `src/viz/LogitsReadout.test.tsx`, in the test `'matrix sketch marks every candidate and the callouts explain the tie'`, replace its last two lines with:

```tsx
  const callouts = screen.getAllByTestId('embed-callout')
  expect(callouts.map((c) => c.textContent)).toEqual(['ⓘ tied with Embeddings', 'ⓘ similar rows, similar scores', 'ⓘ 49 152 scores, 3 shown'])
  expect(callouts.map((c) => c.dataset.section)).toEqual(['tied', 'similar-rows', 'top-k'])
  expect(callouts.every((c) => c.dataset.term === 'logit')).toBe(true)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/viz/details/EmbeddingsDetail.test.tsx src/viz/details/SoftmaxDetail.test.tsx src/viz/LogitsReadout.test.tsx`
Expected: the three rewritten tests FAIL (chips still have `title`, no `data-section`, clicking opens nothing).

- [ ] **Step 3: Migrate the chips**

`src/viz/EmbeddingLookup.tsx`: delete the `CALLOUTS` constant and its comment block; add imports `import { EMBEDDING_CALLOUTS } from '../glossary/callouts'` and `import { Term } from '../glossary/Term'`; replace the chip map with:

```tsx
      <div className="embed-callouts">
        {EMBEDDING_CALLOUTS.map((c) => (
          <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
        ))}
      </div>
```

`src/viz/LogitsReadout.tsx`: delete the `CALLOUTS` function; add imports `import { logitsCallouts } from '../glossary/callouts'` and `import { Term } from '../glossary/Term'`; replace the chip map with:

```tsx
      <div className="embed-callouts">
        {logitsCallouts(candidates.length, vocab).map((c) => (
          <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
        ))}
      </div>
```

`src/viz/details/SoftmaxDetail.tsx`: change the softmaxText import to `import { shareDerivation, softmaxExplainer } from '../softmaxText'`; add `import { SOFTMAX_CALLOUTS } from '../../glossary/callouts'` and `import { Term } from '../../glossary/Term'`; replace the chip map with:

```tsx
      <div className="embed-callouts">
        {SOFTMAX_CALLOUTS.map((c) => (
          <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
        ))}
      </div>
```

`src/viz/softmaxText.ts`: delete the exported `softmaxCallouts` function (its run-specific wording now lives in the Softmax entry's "In this run" block). Keep `leadRatio`, `shareDerivation` and `softmaxExplainer`.

- [ ] **Step 4: Run the three tests to verify they pass**

Run: `npx vitest run src/viz/details/EmbeddingsDetail.test.tsx src/viz/details/SoftmaxDetail.test.tsx src/viz/LogitsReadout.test.tsx`
Expected: PASS.

- [ ] **Step 5: Write the failing diagram test**

```tsx
// src/viz/ResidualStreamDiagram.test.tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { resetGlossaryStore, useGlossaryStore } from '../glossary/store'
import { ResidualStreamDiagram } from './ResidualStreamDiagram'

afterEach(() => { cleanup(); resetGlossaryStore() })

test('the attention and MLP boxes open their entries by click and by keyboard', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  const attn = screen.getByTestId('rs-term-attention')
  expect(attn.querySelector('title')).toBeNull()
  fireEvent.click(attn)
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'attention' })
  resetGlossaryStore()
  fireEvent.keyDown(screen.getByTestId('rs-term-mlp'), { key: 'Enter' })
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'mlp' })
})

test('hover or focus on a box shows the entry summary', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  fireEvent.mouseEnter(screen.getByTestId('rs-term-mlp'))
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('widen the vector')
  fireEvent.mouseLeave(screen.getByTestId('rs-term-mlp'))
  expect(screen.queryByTestId('gl-tip')).toBeNull()
  fireEvent.focus(screen.getByTestId('rs-term-attention'))
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('gathers information')
})

test('the caption links edits and layers', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  const terms = screen.getAllByTestId('gl-term').map((t) => t.dataset.term)
  expect(terms).toEqual(['layer', 'edit'])
})
```

- [ ] **Step 6: Run it to verify it fails**

Run: `npx vitest run src/viz/ResidualStreamDiagram.test.tsx`
Expected: FAIL — no element with test id `rs-term-attention`.

- [ ] **Step 7: Rewrite `ResidualStreamDiagram.tsx`**

```tsx
import { useState, type KeyboardEvent, type SVGProps } from 'react'
import { getEntry } from '../glossary/registry'
import { useGlossaryStore } from '../glossary/store'
import { Term } from '../glossary/Term'

type BoxId = 'attention' | 'mlp'
// left edge of each box in SVG units; the diagram is drawn at 1:1 scale
const TIP_X: Record<BoxId, number> = { attention: 150, mlp: 356 }

// The anatomy of one transformer layer: a single matrix (the residual
// stream) flows left to right; attention and the MLP each ADD their edits
// back into it. Repeated for every layer. The two boxes open their glossary
// entries (SVG cannot hold the HTML Term button, so they wire it up by hand).
export function ResidualStreamDiagram({ seqLen, dims, layers }: {
  seqLen: number
  dims: number
  layers: number
}) {
  const openEntry = useGlossaryStore((s) => s.openEntry)
  const [tip, setTip] = useState<BoxId | null>(null)
  const boxProps = (id: BoxId): SVGProps<SVGGElement> => ({
    role: 'button', tabIndex: 0, className: 'rs-hover gl-svg-term',
    'aria-label': `${getEntry(id)?.title}: open the glossary entry`,
    onClick: (e) => { setTip(null); openEntry(id, undefined, e.currentTarget) },
    onKeyDown: (e: KeyboardEvent<SVGGElement>) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setTip(null); openEntry(id, undefined, e.currentTarget) }
    },
    onMouseEnter: () => setTip(id), onMouseLeave: () => setTip(null),
    onFocus: () => setTip(id), onBlur: () => setTip(null),
  })
  return (
    <div data-testid="residual-diagram" className="residual-diagram">
      <svg width="560" height="118" viewBox="0 0 560 118" role="img"
        aria-label="residual stream: attention and MLP each add their output back into the same matrix">
        {/* the stream */}
        <line x1="8" y1="88" x2="530" y2="88" className="rs-stream" />
        <path d="M530 88 l-8 -5 v10 z" className="rs-arrowhead" />
        <text x="8" y="106" className="rs-shape">x [{seqLen}×{dims}]</text>
        <text x="470" y="106" className="rs-shape">x′ [{seqLen}×{dims}]</text>
        {/* + attention branch */}
        <path d="M120 88 v-30 h30" fill="none" className="rs-branch" />
        <g data-testid="rs-term-attention" {...boxProps('attention')}>
          <rect x="150" y="42" width="112" height="32" rx="6" className="rs-box" />
          <text x="206" y="62" textAnchor="middle" className="rs-box-label">attention</text>
        </g>
        <path d="M262 58 h30 v30" fill="none" className="rs-branch" />
        <circle cx="292" cy="88" r="7" className="rs-adder" />
        <text x="292" y="92" textAnchor="middle" className="rs-plus">+</text>
        <text x="206" y="30" textAnchor="middle" className="rs-note">mixes across tokens — heatmap rows are the weights</text>
        {/* + MLP branch */}
        <path d="M330 88 v-30 h26" fill="none" className="rs-branch" />
        <g data-testid="rs-term-mlp" {...boxProps('mlp')}>
          <rect x="356" y="42" width="88" height="32" rx="6" className="rs-box" />
          <text x="400" y="62" textAnchor="middle" className="rs-box-label">MLP</text>
        </g>
        <path d="M444 58 h26 v30" fill="none" className="rs-branch" />
        <circle cx="470" cy="88" r="7" className="rs-adder" />
        <text x="470" y="92" textAnchor="middle" className="rs-plus">+</text>
        <text x="400" y="30" textAnchor="middle" className="rs-note">each position independently</text>
      </svg>
      {tip && (
        <span role="tooltip" data-testid="gl-tip" className="gl-tip gl-tip-svg" style={{ left: TIP_X[tip], top: 78 }}>
          {getEntry(tip)?.summary} <em>Click to read more.</em>
        </span>
      )}
      <p className="rs-caption">
        × {layers} <Term id="layer">layers</Term> — every layer <em>adds</em> its <Term id="edit">edits</Term> onto
        the same matrix; the original embedding stays underneath.
      </p>
    </div>
  )
}
```

Append to `src/index.css`:

```css
.residual-diagram { position: relative; }
.gl-tip-svg { position: absolute; }
.gl-svg-term:focus-visible { outline: 2px solid var(--ai); outline-offset: 2px; }
```

- [ ] **Step 8: Run the diagram test to verify it passes**

Run: `npx vitest run src/viz/ResidualStreamDiagram.test.tsx`
Expected: PASS (3 tests).

- [ ] **Step 9: Write the failing heading and caption test** (append to `src/viz/DetailPanel.test.tsx`; read the top of the file and reuse its imports and fixture setup)

```tsx
test('every stage card heading has an ⓘ that opens its entry, and captions link their first term', () => {
  const events = makeFixtureTrace()
  // one cursor per stage: tokenize 1, embed 2, layer 3, logits 7, softmax 8, sample 9
  const cases: Array<[number, string, string]> = [
    [1, 'token', 'token'], [2, 'embedding', 'residual-stream'], [3, 'layer', 'layer'],
    [7, 'logit', 'logit'], [8, 'softmax', 'logit'], [9, 'sampling', 'sampling'],
  ]
  for (const [cursor, headingEntry, captionEntry] of cases) {
    render(<DetailPanel events={events} cursor={cursor} mode="sim" />)
    const info = screen.getByTestId('stage-info')
    expect(info.closest('h3'), `cursor ${cursor}`).not.toBeNull()
    expect(info.dataset.term, `cursor ${cursor}`).toBe(headingEntry)
    expect(screen.getAllByTestId('gl-term').map((t) => t.dataset.term), `cursor ${cursor}`).toContain(captionEntry)
    cleanup()
  }
})
```

Make sure `cleanup` is imported from `@testing-library/react` in that file.

- [ ] **Step 10: Run it to verify it fails**

Run: `npx vitest run src/viz/DetailPanel.test.tsx`
Expected: the new test FAILS — no element with test id `stage-info`.

- [ ] **Step 11: Add heading ⓘ buttons and caption terms**

Every card imports `import { Term } from '../../glossary/Term'`. In each heading, the ⓘ goes after the existing heading text, inside the `h3`:

```tsx
<Term id="…" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term>
```

Exact edits:

`TokenizerDetail.tsx`:
```tsx
      <h3>Tokenizer <Term id="token" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>
        The prompt is split into {tokens.length} <Term id="token">tokens</Term>, each mapped to a vocabulary ID.
```

`EmbeddingsDetail.tsx`:
```tsx
      <h3>Embeddings <Term id="embedding" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>
        A lookup, not a computation. Each token id selects one row of a learned matrix{' '}
        <code>E [{vocab ? thousands(vocab) : '?'} × {dims}]</code>; the rows stacked up form the{' '}
        <Term id="residual-stream">residual stream</Term> <code>x [{tokens.length} × {dims}]</code>.
      </p>
```

`LayersDetail.tsx` (the residual-stream link already comes from the diagram caption; the heading links `layer`):
```tsx
      <h3>Transformer layers <Term id="layer" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term>{' '}
        {mode === 'real' && !attentionInRun && <em>(schematic — real internals not exposed)</em>}</h3>
```

`LogitsDetail.tsx`:
```tsx
      <h3>Logits <Term id="logit" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>Each <Term id="logit">score</Term> is a dot product: the last token’s final vector against one row of E — the same matrix the Embeddings stage looked rows up in.</p>
```

`SoftmaxDetail.tsx`:
```tsx
      <h3>Softmax (T={T}) <Term id="softmax" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>Softmax turns the <Term id="logit">scores</Term> into probabilities that sum to 100%: the second half of the readout.</p>
```

`SamplerDetail.tsx`:
```tsx
      <h3>Sampling {sample && `(${sample.method})`} <Term id="sampling" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>A weighted <Term id="sampling">draw</Term> across the probability strip picks the next token.</p>
```

For the Layers case in the test (cursor 3), the caption term `layer` comes from the diagram caption, which renders only when `streamShape` is known — the fixture has an embed event at index 2, so it is.

- [ ] **Step 12: Run the full unit suite, type-check, lint**

Run: `npx vitest run && npx tsc -b && npx oxlint`
Expected: all tests PASS. If an existing test compared a heading with exact text (for example `toHaveTextContent('Softmax (T=0.8)')`), it still passes because `toHaveTextContent` matches substrings; if any test uses an exact `getByText` on a heading, update it to a regex. No new lint warnings.

- [ ] **Step 13: Commit**

```bash
git add src/viz src/index.css
git commit -m "feat(glossary): callout chips, diagram boxes, card headings and captions open glossary entries

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: End-to-end test, screenshot scene, README, backlog

**Files:**
- Create: `e2e/glossary.spec.ts`
- Modify: `scripts/capture-screenshots.mjs` (new `glossary` scene)
- Modify: `README.md` (one sentence in the intro, one short section with the screenshot)
- Modify: `docs/BACKLOG.md` (tick #14, add to Done)
- Create: `docs/screenshots/glossary.png` (generated)

**Interfaces:**
- Consumes: test ids from Tasks 4–6: `btn-help`, `glossary-hint`, `btn-hint-open`, `glossary-drawer`, `gl-heading`, `gl-close`, `gl-in-run`, `embed-callout`, `stage-card`, `run-chip`, `btn-live`, `example-chip`, `prompt-input`, `btn-generate`.

- [ ] **Step 1: Write the e2e test**

```ts
// e2e/glossary.spec.ts
import { expect, test } from '@playwright/test'

test('first visit: hint opens How to read; ? opens the index; a card chip opens its entry with run numbers', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('glossary-hint')).toBeVisible()
  await page.getByTestId('btn-hint-open').click()
  await expect(page.getByTestId('gl-heading')).toContainText('How to read Tsumugi')
  await expect(page.getByTestId('glossary-hint')).toHaveCount(0)
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('glossary-drawer')).toHaveCount(0)

  await page.getByTestId('btn-help').click()
  await expect(page.getByTestId('gl-heading')).toHaveText('Glossary')
  await page.getByTestId('gl-close').click()

  // a simulated run, then the Softmax card's temperature chip
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()
  await page.getByTestId('run-chip').first().waitFor({ timeout: 30000 })   // the run has sealed
  await page.getByTestId('stage-card').nth(4).click()                      // Softmax
  await page.getByTestId('embed-callout').filter({ hasText: 'temperature' }).click()
  await expect(page.getByTestId('gl-heading')).toContainText('Softmax')
  await expect(page.getByTestId('gl-in-run')).toContainText('T =')
})
```

- [ ] **Step 2: Run the e2e suite**

Run: `npx playwright test`
Expected: PASS (smoke, archive, glossary). If a stray dev server from another checkout is on :5173, stop it first (`lsof -ti :5173 | xargs kill`), since Playwright reuses an existing server.

- [ ] **Step 3: Add the `glossary` screenshot scene**

In `scripts/capture-screenshots.mjs`, update the scene comment to list `glossary`, and insert this block directly after the `pipeline` scene (before the real-mode switch; it uses simulated mode):

```js
  // ---- glossary.png: simulated run, Softmax card with the drawer open -------
  if (want('glossary')) {
    await $('example-chip').first().click()                  // auto-generates
    await $('run-chip').first().waitFor({ timeout: 15000 })
    await $('btn-live').click()
    await $('stage-card').nth(4).click()                     // Softmax
    await $('embed-callout').filter({ hasText: 'why exponentiate' }).click()
    await $('gl-in-run').waitFor({ timeout: 5000 })
    await page.waitForTimeout(800)                           // let the section scroll and the highlight settle
    await page.screenshot({ path: `${DIR}/glossary.png` })
    await $('gl-close').click()
    console.log('glossary.png')
  }
```

The screenshot profile persists `localStorage`, so the first-visit hint may or may not show; either is fine for this shot.

- [ ] **Step 4: Capture the screenshot**

Run: `npm run screenshots -- glossary`
Expected: prints `glossary.png`; `docs/screenshots/glossary.png` exists. Open it and check that the drawer shows the Softmax entry with the "Why exponentiate" section and the "In this run" block next to the Softmax card. If the session cannot open a headed browser, skip this step, leave the README image reference in place and tell the user to run the command.

- [ ] **Step 5: Update the README**

After the intro paragraph (before `## Modes`), add one sentence:

```markdown
New to the concepts? The **?** button in the app's header opens a glossary, and underlined words and ⓘ chips in the cards open the matching entry.
```

Add a section before `## Comparing runs`:

```markdown
## The glossary

Every card links its terms to a glossary that opens in a side panel, so the
visualisation stays in view while you read. Entries start with a one-line
summary, then go into the formula and SmolLM2's real shapes. Where the active
run has something to add, an "In this run" block shows its numbers — here the
probability ratios at the run's temperature.

![Glossary drawer open next to the Softmax card](docs/screenshots/glossary.png)
```

- [ ] **Step 6: Tick the backlog item**

In `docs/BACKLOG.md`, change `- [ ] **#14 In-app glossary** — *next up; do before #15.*` to `- [x] **#14 In-app glossary** —` (drop the "next up" note), append ` Shipped 2026-10-10 (spec `docs/superpowers/specs/2026-10-10-in-app-glossary-design.md`).` to the end of the item, and add to the top of "Done (for the record)":

```markdown
- [x] In-app glossary: 12 entries, term links, drawer, "?" index, first-visit hint *(#14, 2026-10-10)*
```

Use the actual ship date if it differs.

- [ ] **Step 7: Full verification**

Run: `npx vitest run && npx tsc -b && npx oxlint && npm run build && npx playwright test`
Expected: everything passes; the build succeeds.

- [ ] **Step 8: Commit**

```bash
git add e2e/glossary.spec.ts scripts/capture-screenshots.mjs README.md docs/BACKLOG.md docs/screenshots/glossary.png
git commit -m "docs(glossary): e2e test, screenshot scene, README section; backlog #14 done

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
