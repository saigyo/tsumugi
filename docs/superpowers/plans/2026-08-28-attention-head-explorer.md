# Attention-Head Explorer (M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all 270 attention heads of the real model explorable: a small-multiples thumbnail grid with per-layer aggregates and sorting, click-to-pin into the existing heatmap viewer, a fourth detected role ("distinctive"), selection hysteresis across cycles, and sticky playback controls above the detail panel.

**Architecture:** A run-level `attention-grid` trace event (emitted once, directly before `run-end`, real mode only) carries thumbnails + stats for every head; pin exactness comes from a single `head-request`/`head-response` worker side channel — the one bounded exception to trace-only data. Pure modules (`attentionThumbs`, distinctive score, hysteresis) are built and tested first; the explorer UI is a store-free viz component wired through callback props.

**Tech Stack:** React 19 + TypeScript (strict) + Vite, Zustand stores, Vitest + @testing-library/react, Playwright (sim-only e2e), transformers.js worker.

**Spec:** `docs/superpowers/specs/2026-08-28-attention-head-explorer-design.md` (conflicts in this plan resolve against the spec).

## Global Constraints

- TypeScript strict must stay green: verify with `npx tsc --noEmit -p tsconfig.app.json` (the root `tsconfig.json` is solution-style and checks nothing).
- Unit tests: `npx vitest run` must pass after every task.
- E2E stays sim-only and unchanged: `npx playwright test` must stay green (checked in the final task; no e2e edits in this plan).
- Real-mode attention work must never fail generation: every new worker-side attention path is wrapped so failure flips `attnBroken` and generation continues (M1 policy).
- The default test fixture `makeFixtureTrace` must NOT gain the `attention-grid` event — index-based tests depend on its exact event positions (tokenize=1, embed=2, layers=3–5, attention=6, …). A separate `makeGridEvent()` helper serves grid tests.
- The simulated engine is untouched. `selectShowcaseHeads`'s new parameters are optional so existing call sites compile unchanged.
- Viz components stay store-free: data and actions arrive via props/callbacks (same pattern as `onStageClick`).
- Canvas rendering is allowed ONLY for `AttentionGridExplorer` thumbnails (recorded spec amendment to v1's "SVG + CSS only"); the main `AttentionHeatmap` stays SVG.
- Linear git history on the feature branch; small frequent commits with conventional-commit messages matching the repo log (`feat:`, `fix:`, `docs:`, scope optional).

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/engine/transformers/attentionStats.ts` | modify | + `distinctiveScore` in `HeadStats`, hysteresis + 4th chip in `selectShowcaseHeads`, new `resolveHeadLabel` |
| `src/trace/types.ts` | modify | + `'distinctive'`/`'pinned'` labels, `AttentionGridCell`, `attention-grid` event |
| `src/viz/AttentionHeatmap.tsx` | modify | + two `HINTS` entries (keeps `Record<AttentionLabel, string>` total) |
| `src/trace/validate.ts` | modify | + `attention-grid` placement/bounds rules |
| `src/test/fixtures.ts` | modify | + `makeGridEvent()` (fixture trace untouched) |
| `src/engine/transformers/attentionThumbs.ts` | create | `poolThumb`, `buildGridCells` — pure, model-free |
| `src/engine/transformers/protocol.ts` | modify | + `head-request` / `head-response` |
| `src/engine/transformers/TransformersEngine.ts` | modify | + `fetchHead`, `HeadData` |
| `src/engine/transformers/worker.ts` | modify | grid emission, run retention, head-request handler, hysteresis threading |
| `src/viz/AttentionGridExplorer.tsx` | create | thumbnail grid, aggregates, sort, pin clicks |
| `src/app/usePins.ts` | create | pin list state: fetch, dedupe, FIFO cap 5, stale note, reset |
| `src/viz/details/LayersDetail.tsx` | modify | explore toggle, explorer mount, pinned-head merge, pin note |
| `src/viz/DetailPanel.tsx` | modify | thread grid event + pin props to LayersDetail |
| `src/App.tsx` | modify | pin wiring; `Controls` moves above `DetailPanel` |
| `src/index.css` | modify | explorer styles; sticky `.controls` |
| `docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md` | modify | one-line layout amendment |

---

### Task 1: Distinctive score in `headStats`

**Files:**
- Modify: `src/engine/transformers/attentionStats.ts` (interface at lines 4–10, loop at lines 22–41)
- Test: `src/engine/transformers/attentionStats.test.ts`

**Interfaces:**
- Consumes: `AttnAccumulator` from `./attentionAccum` (unchanged).
- Produces: `HeadStats` gains `distinctiveScore: number`. Formula: `distinctive = (1 − templateMax) × (1 − uniformity)` with `templateMax = max(prevTokenScore, sinkScore, inductionScore ?? 0)` and `uniformity` = mean over rows i ≥ 1 of `entropy(row) / log(row.length)`. Heads with ≤ 1 accumulated row score 0.

- [ ] **Step 1: Write the failing tests**

Append to `src/engine/transformers/attentionStats.test.ts`:

```ts
test('distinctive score: template heads low, focused-untemplated head high, uniform head low', () => {
  const acc = createAccumulator(3, 1)
  // template: perfect previous-token head
  acc.rows[0][0] = fill(diagRow, 6)
  // focused but untemplated: rows 3.. lock onto position 1 (not prev, not sink)
  acc.rows[1][0] = [
    [1], [0.5, 0.5], [1 / 3, 1 / 3, 1 / 3],
    [0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0],
  ]
  // uniform: attention spread evenly
  acc.rows[2][0] = Array.from({ length: 6 }, (_, i) =>
    Array.from({ length: i + 1 }, () => 1 / (i + 1)))
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e', ' f'))
  expect(stats[0].distinctiveScore).toBe(0)                 // templateMax = 1
  expect(stats[1].distinctiveScore).toBeGreaterThan(0.4)    // ≈ 0.83 × 0.6
  expect(stats[2].distinctiveScore).toBeLessThan(0.01)      // uniformity ≈ 1
})

test('distinctive score is 0 for single-row heads', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = [[1]]
  expect(headStats(acc, toks('a'))[0].distinctiveScore).toBe(0)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engine/transformers/attentionStats.test.ts`
Expected: FAIL — `distinctiveScore` is `undefined` (property does not exist).

- [ ] **Step 3: Implement**

In `src/engine/transformers/attentionStats.ts`, add to the interface:

```ts
export interface HeadStats {
  layer: number
  head: number
  prevTokenScore: number
  sinkScore: number
  inductionScore: number | null
  // (1 − best template score) × (1 − mean normalized row entropy):
  // high = peaked attention that matches no known template
  distinctiveScore: number
}
```

In the `headStats` per-head loop, accumulate normalized row entropy alongside the existing sums (rows i ≥ 1 only; those rows have length ≥ 2, so `Math.log(m[i].length) > 0`):

```ts
      const m = acc.rows[l][h]
      let prev = 0, sink = 0, n = 0, ind = 0, indN = 0, ent = 0
      for (let i = 1; i < m.length; i++) {
        prev += m[i][i - 1]
        sink += m[i][0]
        n++
        let rowEnt = 0
        for (const w of m[i]) if (w > 0) rowEnt -= w * Math.log(w)
        ent += Math.min(1, rowEnt / Math.log(m[i].length))
        const t = targets[i]
        if (t !== null && t < m[i].length) { ind += m[i][t]; indN++ }
      }
      const prevTokenScore = n ? prev / n : 0
      const sinkScore = n ? sink / n : 0
      const inductionScore = indN ? ind / indN : null
      const templateMax = Math.max(prevTokenScore, sinkScore, inductionScore ?? 0)
      const uniformity = n ? ent / n : 1
      out.push({
        layer: l, head: h, prevTokenScore, sinkScore, inductionScore,
        distinctiveScore: n ? (1 - templateMax) * (1 - uniformity) : 0,
      })
```

(This replaces the whole per-head body — from `const m = acc.rows[l][h]` through the existing `out.push({...})`; the surrounding `for (let l …) for (let h …)` structure and the `targets` computation above are unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/engine/transformers/attentionStats.test.ts`
Expected: PASS (all tests, including the pre-existing ones).

- [ ] **Step 5: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/engine/transformers/attentionStats.ts src/engine/transformers/attentionStats.test.ts
git commit -m "feat: distinctive score in head stats"
```

---

### Task 2: `'distinctive'`/`'pinned'` labels, fourth showcase chip, selection hysteresis

**Files:**
- Modify: `src/trace/types.ts:5` (AttentionLabel)
- Modify: `src/viz/AttentionHeatmap.tsx:4-9` (HINTS)
- Modify: `src/engine/transformers/attentionStats.ts:46-65` (selectShowcaseHeads)
- Test: `src/engine/transformers/attentionStats.test.ts`

**Interfaces:**
- Consumes: `HeadStats.distinctiveScore` from Task 1.
- Produces:
  - `type AttentionLabel = 'previous-token' | 'attention-sink' | 'induction' | 'coreference' | 'distinctive' | 'pinned'`
  - `export type ShowcasePrev = Partial<Record<AttentionLabel, { layer: number; head: number }>>`
  - `selectShowcaseHeads(stats: HeadStats[], acc: AttnAccumulator, threshold = 0.3, prev?: ShowcasePrev): AttentionHead[]` — now returns ≤ 4 heads (distinctive uses its own threshold 0.25); with `prev`, an incumbent is kept unless the challenger's score exceeds the incumbent's current score by ≥ 0.05, or the incumbent's current score fell to ≤ its label's threshold.

- [ ] **Step 1: Extend the label union and HINTS (compile-safe pair)**

In `src/trace/types.ts` line 5:

```ts
export type AttentionLabel = 'previous-token' | 'attention-sink' | 'induction' | 'coreference' | 'distinctive' | 'pinned'
```

In `src/viz/AttentionHeatmap.tsx`, add to `HINTS` (it is `Record<AttentionLabel, string>`, so tsc forces this pairing):

```ts
  distinctive: 'Focused attention that fits no textbook pattern — look for what it tracks.',
  pinned: 'Hand-picked from the grid — compare against the patterns you know.',
```

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

- [ ] **Step 2: Write the failing tests**

Append to `src/engine/transformers/attentionStats.test.ts` (the file will not compile until Step 4 exports `ShowcasePrev` — that is the expected failure):

```ts
test('fourth chip: a distinctive head above 0.25 is selected', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = [
    [1], [0.5, 0.5], [1 / 3, 1 / 3, 1 / 3],
    [0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0],
  ]
  const heads = selectShowcaseHeads(headStats(acc, toks('a', ' b', ' c', ' d', ' e', ' f')), acc)
  const d = heads.find((h) => h.label === 'distinctive')
  expect(d).toBeDefined()
  expect(d!.score).toBeGreaterThan(0.25)
})

test('hysteresis: incumbent sticks under a <0.05 challenger lead', () => {
  const acc = createAccumulator(2, 1)
  // incumbent L0: prev-token score 0.96; challenger L1: 0.99 — lead 0.03 < 0.05
  const nearDiag = (p: number) => (i: number) =>
    i === 0 ? [1] : [...Array(Math.max(0, i - 1)).fill(0), p, 1 - p]
  acc.rows[0][0] = fill(nearDiag(0.96), 5)
  acc.rows[1][0] = fill(nearDiag(0.99), 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  const prev: ShowcasePrev = { 'previous-token': { layer: 0, head: 0 } }
  const kept = selectShowcaseHeads(stats, acc, 0.3, prev).find((h) => h.label === 'previous-token')
  expect(kept!.layer).toBe(0)
  // without prev, argmax wins
  const argmax = selectShowcaseHeads(stats, acc).find((h) => h.label === 'previous-token')
  expect(argmax!.layer).toBe(1)
})

test('hysteresis: challenger wins at a ≥0.05 lead', () => {
  const acc = createAccumulator(2, 1)
  const nearDiag = (p: number) => (i: number) =>
    i === 0 ? [1] : [...Array(Math.max(0, i - 1)).fill(0), p, 1 - p]
  acc.rows[0][0] = fill(nearDiag(0.9), 5)
  acc.rows[1][0] = fill(nearDiag(0.96), 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  const prev: ShowcasePrev = { 'previous-token': { layer: 0, head: 0 } }
  const winner = selectShowcaseHeads(stats, acc, 0.3, prev).find((h) => h.label === 'previous-token')
  expect(winner!.layer).toBe(1)
})

test('hysteresis: incumbent below threshold falls back to argmax', () => {
  const acc = createAccumulator(2, 1)
  // incumbent decayed to uniform: prevTokenScore = mean(1/2..1/6) ≈ 0.29 < 0.3
  acc.rows[0][0] = Array.from({ length: 6 }, (_, i) =>
    Array.from({ length: i + 1 }, () => 1 / (i + 1)))
  // challenger at 0.32: above threshold, but its lead over 0.29 is < 0.05 —
  // it must still win, because a sub-threshold incumbent loses its seat
  const nearDiag = (p: number) => (i: number) =>
    i === 0 ? [1] : [...Array(Math.max(0, i - 1)).fill(0), p, 1 - p]
  acc.rows[1][0] = fill(nearDiag(0.32), 6)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e', ' f'))
  const prev: ShowcasePrev = { 'previous-token': { layer: 0, head: 0 } }
  const winner = selectShowcaseHeads(stats, acc, 0.3, prev).find((h) => h.label === 'previous-token')
  expect(winner!.layer).toBe(1)
})
```

Also update the top import to `import { headStats, selectShowcaseHeads, type ShowcasePrev } from './attentionStats'`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `npx vitest run src/engine/transformers/attentionStats.test.ts`
Expected: FAIL — `ShowcasePrev` not exported; no `'distinctive'` head selected.

- [ ] **Step 4: Implement**

Replace `selectShowcaseHeads` in `src/engine/transformers/attentionStats.ts`:

```ts
export type ShowcasePrev = Partial<Record<AttentionLabel, { layer: number; head: number }>>

export function selectShowcaseHeads(
  stats: HeadStats[], acc: AttnAccumulator, threshold = 0.3, prev?: ShowcasePrev,
): AttentionHead[] {
  const pick = (
    label: AttentionLabel, score: (s: HeadStats) => number | null, thr = threshold,
  ): AttentionHead | null => {
    let best: HeadStats | null = null
    let bestScore = thr
    for (const s of stats) {
      const v = score(s)
      if (v !== null && v > bestScore) { best = s; bestScore = v }
    }
    // hysteresis: keep last cycle's head unless the challenger beats its
    // CURRENT score by ≥ 0.05; an incumbent fallen to ≤ thr loses its seat
    const p = prev?.[label]
    if (p) {
      const inc = stats.find((s) => s.layer === p.layer && s.head === p.head)
      const incScore = inc ? score(inc) : null
      if (inc && incScore !== null && incScore > thr && (!best || bestScore - incScore < 0.05)) {
        best = inc
        bestScore = incScore
      }
    }
    if (!best) return null
    return { layer: best.layer, head: best.head, label,
      score: round(bestScore), matrix: acc.rows[best.layer][best.head] }
  }
  return [
    pick('previous-token', (s) => s.prevTokenScore),
    pick('attention-sink', (s) => s.sinkScore),
    pick('induction', (s) => s.inductionScore),
    pick('distinctive', (s) => s.distinctiveScore, 0.25),
  ].filter((h): h is AttentionHead => h !== null)
}
```

- [ ] **Step 5: Run the full unit suite**

Run: `npx vitest run`
Expected: PASS. Note: the pre-existing tests `selectShowcaseHeads picks top head per label above threshold` (pure diag + pure sink heads: both have `distinctiveScore` 0, so still exactly 2 heads) and `heads below threshold are not selected` (scattered head: distinctive ≈ 0.09 < 0.25, so still 0 heads) must pass unchanged. If either fails, the distinctive formula is wrong — do not adjust the old tests.

- [ ] **Step 6: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/trace/types.ts src/viz/AttentionHeatmap.tsx src/engine/transformers/attentionStats.ts src/engine/transformers/attentionStats.test.ts
git commit -m "feat: distinctive/pinned labels, fourth showcase chip, selection hysteresis"
```

---

### Task 3: `attention-grid` trace event, validator rules, `makeGridEvent` helper

**Files:**
- Modify: `src/trace/types.ts` (event union, new cell interface)
- Modify: `src/trace/validate.ts`
- Modify: `src/test/fixtures.ts` (append helper only — `makeFixtureTrace` body untouched)
- Test: `src/trace/validate.test.ts`

**Interfaces:**
- Produces:
  - `export interface AttentionGridCell { layer: number; head: number; thumb: number[][]; prevTokenScore: number; sinkScore: number; inductionScore: number | null; distinctiveScore: number }`
  - TraceEvent member `| { type: 'attention-grid'; layers: number; heads: number; cells: AttentionGridCell[] }` (inserted directly before the `run-end` member)
  - `export function makeGridEvent(layers = 2, heads = 2): Extract<TraceEvent, { type: 'attention-grid' }>` in `src/test/fixtures.ts`
- Validator rules (only when the event is present): at most one; must sit directly before `run-end`; `cells.length === layers × heads`; thumb dims ≤ 12; thumb values within [0, 1]. No row-stochasticity check.

- [ ] **Step 1: Add the types (needed for the tests to compile)**

In `src/trace/types.ts`, after the `AttentionHead` interface:

```ts
export interface AttentionGridCell {
  layer: number
  head: number
  // ≤12×12 mean-pooled thumbnail of the accumulated causal matrix, values 0..1.
  // Mean pooling preserves relative mass; pooled rows do NOT sum to 1.
  thumb: number[][]
  prevTokenScore: number
  sinkScore: number
  inductionScore: number | null
  distinctiveScore: number
}
```

In the `TraceEvent` union, insert before the `run-end` line:

```ts
  // run-level snapshot: attention accumulated over the whole run (real mode only);
  // emitted once, directly before run-end
  | { type: 'attention-grid'; layers: number; heads: number; cells: AttentionGridCell[] }
```

- [ ] **Step 2: Write the failing tests**

Append to `src/trace/validate.test.ts` (extend the fixtures import to `import { makeFixtureTrace, makeGridEvent } from '../test/fixtures'`):

```ts
test('default fixture has no attention-grid event', () => {
  expect(makeFixtureTrace().some((e) => e.type === 'attention-grid')).toBe(false)
})

test('attention-grid directly before run-end is valid', () => {
  const t = makeFixtureTrace()
  t.splice(t.length - 1, 0, makeGridEvent())
  expect(validateTrace(t)).toEqual([])
})

test('attention-grid away from run-end is flagged', () => {
  const t = makeFixtureTrace()
  t.splice(2, 0, makeGridEvent())
  expect(validateTrace(t).some((v) => v.includes('attention-grid'))).toBe(true)
})

test('attention-grid cell count must be layers × heads', () => {
  const t = makeFixtureTrace()
  const g = makeGridEvent(2, 2)
  g.cells.pop()
  t.splice(t.length - 1, 0, g)
  expect(validateTrace(t).some((v) => v.includes('cells'))).toBe(true)
})

test('attention-grid thumbs must be ≤12×12 with values in [0,1]', () => {
  const t = makeFixtureTrace()
  const g = makeGridEvent(2, 2)
  g.cells[0].thumb[0][0] = 1.5
  g.cells[1].thumb = Array.from({ length: 13 }, () => Array.from({ length: 13 }, () => 0))
  t.splice(t.length - 1, 0, g)
  const errs = validateTrace(t)
  expect(errs.some((v) => v.includes('[0, 1]'))).toBe(true)
  expect(errs.some((v) => v.includes('12'))).toBe(true)
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `npx vitest run src/trace/validate.test.ts`
Expected: FAIL — `makeGridEvent` is not exported; the "valid" case fails with `unexpected attention-grid in phase …`.

- [ ] **Step 4: Implement the helper**

Extend the import at the top of `src/test/fixtures.ts` to
`import type { AttentionGridCell, TraceEvent, TokenInfo } from '../trace/types'`, then append:

```ts
// Grid tests use this instead of growing makeFixtureTrace — the default
// fixture's event indices are load-bearing for index-based tests.
export function makeGridEvent(layers = 2, heads = 2): Extract<TraceEvent, { type: 'attention-grid' }> {
  const thumb = (l: number, h: number): number[][] =>
    Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) =>
      c > r ? 0 : ((l * heads + h + 1) / (layers * heads + 1)) * (c === r ? 1 : 0.25)))
  const cells: AttentionGridCell[] = []
  for (let l = 0; l < layers; l++) {
    for (let h = 0; h < heads; h++) {
      const k = l * heads + h
      cells.push({
        layer: l, head: h, thumb: thumb(l, h),
        prevTokenScore: (k % 5) / 5,
        sinkScore: ((k + 1) % 5) / 5,
        inductionScore: k % 3 === 0 ? null : (k % 4) / 4,
        distinctiveScore: ((k * 3 + 1) % 5) / 5,
      })
    }
  }
  return { type: 'attention-grid', layers, heads, cells }
}
```

(With `makeGridEvent(2, 2)` the distinctive order is L0·H1 (0.8) > L1·H0 (0.4) > L0·H0 (0.2) > L1·H1 (0.0) — deliberately different from layer order so sort tests in Task 7 can tell them apart. Keep the explicit `AttentionGridCell[]` annotation on `cells`; an unannotated `[]` would be an implicitly-typed evolving array.)

- [ ] **Step 5: Implement the validator rules**

In `src/trace/validate.ts`, in the phase loop (inside `for (const e of events.slice(2, -1))`), add before the `if (e.type === 'embed' …)` line:

```ts
    if (e.type === 'attention-grid') continue  // placement checked below
```

After the phase loop (before the softmax/logits pass), add:

```ts
  const grids = events.flatMap((e, i) => (e.type === 'attention-grid' ? [{ e, i }] : []))
  if (grids.length > 1) errs.push('more than one attention-grid event')
  for (const { e, i } of grids) {
    if (i !== events.length - 2) errs.push('attention-grid must sit directly before run-end')
    if (e.cells.length !== e.layers * e.heads)
      errs.push(`attention-grid has ${e.cells.length} cells, expected layers × heads = ${e.layers * e.heads}`)
    for (const c of e.cells) {
      if (c.thumb.length > 12 || c.thumb.some((r) => r.length > 12))
        errs.push(`attention-grid thumb L${c.layer}·H${c.head} exceeds 12×12`)
      if (c.thumb.some((r) => r.some((v) => v < 0 || v > 1)))
        errs.push(`attention-grid thumb L${c.layer}·H${c.head} has values outside [0, 1]`)
    }
  }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `npx vitest run src/trace/validate.test.ts`
Expected: PASS (new and pre-existing tests).

- [ ] **Step 7: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/trace/types.ts src/trace/validate.ts src/test/fixtures.ts src/trace/validate.test.ts
git commit -m "feat: attention-grid trace event, validator rules, grid fixture helper"
```

---

### Task 4: `attentionThumbs` — mean-pooled thumbnails and grid cells

**Files:**
- Create: `src/engine/transformers/attentionThumbs.ts`
- Test: `src/engine/transformers/attentionThumbs.test.ts`

**Interfaces:**
- Consumes: `AttnAccumulator` (`acc.rows[layer][head]` is the ragged matrix), `HeadStats` (layer-major order from `headStats`), `AttentionGridCell` from `../../trace/types`.
- Produces:
  - `export function poolThumb(matrix: number[][], buckets = 12): number[][]` — square `min(buckets, rows)`-sized block-mean pooling; blocks entirely in the causal-empty region are 0; `[]` for an empty matrix.
  - `export function buildGridCells(acc: AttnAccumulator, stats: HeadStats[]): AttentionGridCell[]` — one cell per stats entry, in stats order (layer-major).

- [ ] **Step 1: Write the failing tests**

Create `src/engine/transformers/attentionThumbs.test.ts`:

```ts
import { expect, test } from 'vitest'
import { createAccumulator } from './attentionAccum'
import { headStats } from './attentionStats'
import { buildGridCells, poolThumb } from './attentionThumbs'

test('poolThumb on a 2-row matrix with 2 buckets is the identity with causal zeros', () => {
  expect(poolThumb([[1], [0.5, 0.5]], 2)).toEqual([[1, 0], [0.5, 0.5]])
})

test('poolThumb caps at the bucket count and stays within [0,1]', () => {
  const n = 30
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: i + 1 }, () => 1 / (i + 1)))
  const thumb = poolThumb(matrix)
  expect(thumb).toHaveLength(12)
  for (const row of thumb) {
    expect(row).toHaveLength(12)
    for (const v of row) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1) }
  }
  // upper-right block: entirely above the diagonal → 0
  expect(thumb[0][11]).toBe(0)
})

test('poolThumb uses fewer buckets than requested for short sequences', () => {
  const thumb = poolThumb([[1], [1, 0], [0, 1, 0]], 12)
  expect(thumb).toHaveLength(3)
  expect(thumb[2]).toEqual([0, 1, 0])
})

test('poolThumb of an empty matrix is empty', () => {
  expect(poolThumb([])).toEqual([])
})

test('poolThumb block means average only the defined causal entries', () => {
  // 4 rows, 2 buckets: block (1,0) covers rows 2-3 × cols 0-1 → mean of 4 entries
  const m = [[1], [0, 1], [0.2, 0.4, 0.4], [0.1, 0.3, 0.3, 0.3]]
  const thumb = poolThumb(m, 2)
  expect(thumb[1][0]).toBeCloseTo((0.2 + 0.4 + 0.1 + 0.3) / 4, 10)
})

test('buildGridCells emits one cell per head in layer-major order with the stats', () => {
  const acc = createAccumulator(2, 2)
  for (let l = 0; l < 2; l++) for (let h = 0; h < 2; h++)
    acc.rows[l][h] = [[1], [0.5, 0.5]]
  const stats = headStats(acc, [{ id: 0, text: 'a' }, { id: 1, text: ' b' }])
  const cells = buildGridCells(acc, stats)
  expect(cells).toHaveLength(4)
  expect(cells.map((c) => [c.layer, c.head])).toEqual([[0, 0], [0, 1], [1, 0], [1, 1]])
  expect(cells[0].thumb).toEqual([[1, 0], [0.5, 0.5]])
  expect(cells[0].prevTokenScore).toBe(stats[0].prevTokenScore)
  expect(cells[0].distinctiveScore).toBe(stats[0].distinctiveScore)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engine/transformers/attentionThumbs.test.ts`
Expected: FAIL — module `./attentionThumbs` does not exist.

- [ ] **Step 3: Implement**

Create `src/engine/transformers/attentionThumbs.ts`:

```ts
import type { AttentionGridCell } from '../../trace/types'
import type { AttnAccumulator } from './attentionAccum'
import type { HeadStats } from './attentionStats'

// Mean-pooled block average of a ragged causal matrix. Mean pooling
// preserves relative mass — max pooling would make every head look like
// its brightest cell. Pooled rows do NOT sum to 1; that is by design.
export function poolThumb(matrix: number[][], buckets = 12): number[][] {
  const n = matrix.length
  if (n === 0) return []
  const b = Math.min(buckets, n)
  const edge = (i: number) => Math.floor((i * n) / b)
  const out: number[][] = []
  for (let br = 0; br < b; br++) {
    const row: number[] = []
    for (let bc = 0; bc < b; bc++) {
      let sum = 0, count = 0
      for (let r = edge(br); r < edge(br + 1); r++) {
        for (let c = edge(bc); c < edge(bc + 1); c++) {
          if (c < matrix[r].length) { sum += matrix[r][c]; count++ }
        }
      }
      row.push(count ? sum / count : 0)
    }
    out.push(row)
  }
  return out
}

export function buildGridCells(acc: AttnAccumulator, stats: HeadStats[]): AttentionGridCell[] {
  return stats.map((s) => ({
    layer: s.layer,
    head: s.head,
    thumb: poolThumb(acc.rows[s.layer][s.head]),
    prevTokenScore: s.prevTokenScore,
    sinkScore: s.sinkScore,
    inductionScore: s.inductionScore,
    distinctiveScore: s.distinctiveScore,
  }))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/engine/transformers/attentionThumbs.test.ts`
Expected: PASS.

- [ ] **Step 5: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/engine/transformers/attentionThumbs.ts src/engine/transformers/attentionThumbs.test.ts
git commit -m "feat: mean-pooled attention thumbnails and grid cell builder"
```

---

### Task 5: `head-request`/`head-response` protocol and `TransformersEngine.fetchHead`

**Files:**
- Modify: `src/engine/transformers/protocol.ts`
- Modify: `src/engine/transformers/TransformersEngine.ts`
- Test: `src/engine/transformers/TransformersEngine.test.ts`

**Interfaces:**
- Produces:
  - `WorkerRequest` member `| { type: 'head-request'; layer: number; head: number }`
  - `WorkerResponse` member `| { type: 'head-response'; layer: number; head: number; matrix: number[][]; label: AttentionLabel | null; score: number | null }` (`matrix: []` = unavailable)
  - `export interface HeadData { layer: number; head: number; matrix: number[][]; label: AttentionLabel | null; score: number | null }` in `TransformersEngine.ts`
  - `TransformersEngine.fetchHead(layer: number, head: number): Promise<HeadData>` — resolves on the matching (layer, head) response; other responses are ignored (last-request-wins correlation is sufficient per spec).
- This side channel exists solely for pin exactness and must not grow other uses without a new design (put this sentence as a comment on the request type).

- [ ] **Step 1: Write the failing tests**

Append to `src/engine/transformers/TransformersEngine.test.ts`:

```ts
test('fetchHead posts head-request and resolves on the matching response', async () => {
  const { worker, engine } = make()
  const p = engine.fetchHead(2, 5)
  expect(worker.sent.at(-1)).toEqual({ type: 'head-request', layer: 2, head: 5 })
  worker.respond({ type: 'head-response', layer: 1, head: 5, matrix: [[1]], label: null, score: null })  // ignored
  worker.respond({ type: 'head-response', layer: 2, head: 5, matrix: [[1], [0.5, 0.5]], label: 'previous-token', score: 0.5 })
  const r = await p
  expect(r.matrix).toEqual([[1], [0.5, 0.5]])
  expect(r.label).toBe('previous-token')
  expect(r.score).toBe(0.5)
})

test('fetchHead resolves the unavailable case as an empty matrix', async () => {
  const { worker, engine } = make()
  const p = engine.fetchHead(0, 0)
  worker.respond({ type: 'head-response', layer: 0, head: 0, matrix: [], label: null, score: null })
  const r = await p
  expect(r.matrix).toEqual([])
  expect(r.label).toBeNull()
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engine/transformers/TransformersEngine.test.ts`
Expected: FAIL — `head-request`/`head-response` not in the protocol union; `fetchHead` does not exist.

- [ ] **Step 3: Implement**

In `src/engine/transformers/protocol.ts`, extend the first import to `import type { AttentionLabel, GenParams, TraceEvent } from '../../trace/types'` and the unions:

```ts
export type WorkerRequest =
  | { type: 'prepare'; modelId: string }
  | { type: 'run'; runId: number; prompt: string; params: GenParams }
  | { type: 'abort' }
  // pin-exactness side channel — the one data path beside the trace;
  // must not grow other uses without a new design
  | { type: 'head-request'; layer: number; head: number }
export type WorkerResponse =
  | { type: 'progress'; info: ProgressInfo }
  | { type: 'ready'; device: 'webgpu' | 'wasm'; attentions: boolean }
  | { type: 'trace'; runId: number; event: TraceEvent }
  | { type: 'done'; runId: number }
  | { type: 'fatal'; message: string }
  | { type: 'head-response'; layer: number; head: number
      matrix: number[][]                // exact accumulated ragged matrix; [] if unavailable
      label: AttentionLabel | null      // best template ≥ 0.3, else null
      score: number | null }
```

In `src/engine/transformers/TransformersEngine.ts`, extend the types import to include `AttentionLabel`, add above the class:

```ts
export interface HeadData {
  layer: number
  head: number
  matrix: number[][]
  label: AttentionLabel | null
  score: number | null
}
```

and add the method after `run`:

```ts
  fetchHead(layer: number, head: number): Promise<HeadData> {
    return new Promise((resolve) => {
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'head-response' && msg.layer === layer && msg.head === head) {
          this.listeners.delete(listener)
          resolve({ layer: msg.layer, head: msg.head, matrix: msg.matrix, label: msg.label, score: msg.score })
        }
      }
      this.listeners.add(listener)
      this.post({ type: 'head-request', layer, head })
    })
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/engine/transformers/TransformersEngine.test.ts`
Expected: PASS.

- [ ] **Step 5: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/engine/transformers/protocol.ts src/engine/transformers/TransformersEngine.ts src/engine/transformers/TransformersEngine.test.ts
git commit -m "feat: head-request side channel and fetchHead on the engine"
```

---

### Task 6: Worker — grid emission, run retention, head-request handler, hysteresis threading

**Files:**
- Modify: `src/engine/transformers/attentionStats.ts` (add `resolveHeadLabel`)
- Modify: `src/engine/transformers/worker.ts`
- Test: `src/engine/transformers/attentionStats.test.ts`

**Interfaces:**
- Consumes: `buildGridCells` (Task 4), `ShowcasePrev` + hysteresis (Task 2), protocol members (Task 5).
- Produces: `export function resolveHeadLabel(stats: HeadStats[], layer: number, head: number, threshold = 0.3): { label: AttentionLabel | null; score: number | null }` — best of the three template scores if ≥ threshold, else nulls (distinctive is not a template and is never returned here).
- Worker behavior (verified by tsc + the pure-module tests + manual browser check in the final task; `worker.ts` has no unit harness):
  - retains `{ acc, stats }` after a run finishes, until the next run starts;
  - emits `attention-grid` once, directly before every `run-end` it produces (eos / max-tokens / aborted), only when `acc` exists and `!attnBroken`; emission failures flip `attnBroken` and never kill the run;
  - answers `head-request` from the retained run (`matrix: []` when no retained run or indices out of range);
  - threads the previous cycle's selection into `selectShowcaseHeads`.

- [ ] **Step 1: Write the failing tests for `resolveHeadLabel`**

Append to `src/engine/transformers/attentionStats.test.ts` (extend the import line with `resolveHeadLabel`):

```ts
test('resolveHeadLabel returns the best template at or above 0.3', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = fill(diagRow, 5)
  acc.rows[1][0] = fill(sinkRow, 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  expect(resolveHeadLabel(stats, 0, 0)).toEqual({ label: 'previous-token', score: 1 })
  expect(resolveHeadLabel(stats, 1, 0)).toEqual({ label: 'attention-sink', score: 1 })
})

test('resolveHeadLabel returns nulls below threshold or for unknown heads', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = Array.from({ length: 5 }, (_, i) =>
    Array.from({ length: i + 1 }, () => 1 / (i + 1)))
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  expect(resolveHeadLabel(stats, 0, 0)).toEqual({ label: null, score: null })
  expect(resolveHeadLabel(stats, 7, 7)).toEqual({ label: null, score: null })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engine/transformers/attentionStats.test.ts`
Expected: FAIL — `resolveHeadLabel` not exported.

- [ ] **Step 3: Implement `resolveHeadLabel`**

Append to `src/engine/transformers/attentionStats.ts`:

```ts
// Label a single head for the pin flow: best TEMPLATE score at/above the
// threshold ('distinctive' is not a template and is never assigned here).
export function resolveHeadLabel(
  stats: HeadStats[], layer: number, head: number, threshold = 0.3,
): { label: AttentionLabel | null; score: number | null } {
  const s = stats.find((x) => x.layer === layer && x.head === head)
  if (!s) return { label: null, score: null }
  const candidates: Array<[AttentionLabel, number | null]> = [
    ['previous-token', s.prevTokenScore],
    ['attention-sink', s.sinkScore],
    ['induction', s.inductionScore],
  ]
  let label: AttentionLabel | null = null
  let best = threshold
  for (const [lab, v] of candidates) {
    if (v !== null && v >= best) { label = lab; best = v }
  }
  return label ? { label, score: round(best) } : { label: null, score: null }
}
```

Run: `npx vitest run src/engine/transformers/attentionStats.test.ts` — expected: PASS.

- [ ] **Step 4: Wire the worker**

In `src/engine/transformers/worker.ts`:

a) Extend imports:

```ts
import type { GenParams, RunEndReason, TokenInfo, TraceEvent } from '../../trace/types'
import { addAttentionOutput, createAccumulator, type AttnAccumulator } from './attentionAccum'
import { headStats, resolveHeadLabel, selectShowcaseHeads, type HeadStats, type ShowcasePrev } from './attentionStats'
import { buildGridCells } from './attentionThumbs'
```

b) Add module state next to `let aborted = false`:

```ts
// finished run kept for the head-request side channel until the next run
// starts (memory already bounded by ATTN_MAX_SEQ)
let lastRun: { acc: AttnAccumulator; stats: HeadStats[] } | null = null
```

c) In `run()`, immediately after `aborted = false`, add `lastRun = null`.

d) Replace the per-cycle showcase block (currently the second `if (acc && !attnBroken)` block) so stats are kept and the previous selection is threaded — declare, before the `try {` that opens the cycle loop:

```ts
  let stats: HeadStats[] | null = null
  let prevSel: ShowcasePrev = {}
```

and make the block:

```ts
      if (acc && !attnBroken) {
        try {
          stats = headStats(acc, allIds.map(tokenInfo))
          const heads = selectShowcaseHeads(stats, acc, 0.3, prevSel)
          prevSel = Object.fromEntries(heads.map((h) => [h.label, { layer: h.layer, head: h.head }]))
          if (heads.length > 0) emit({ type: 'attention', cycle, heads })
        } catch { attnBroken = true }
      }
```

(The existing aliasing comment above the block stays.)

e) Add an `endRun` helper inside `run()` — place it directly after the `let attnBroken = false` / `stats` / `prevSel` declarations, before the `try {` that opens the cycle loop, so every variable it closes over is already declared — and use it for all three run-end emissions (`aborted`, `eos`, `max-tokens`):

```ts
  // Grid emission shares the never-fail policy: a failure flips attnBroken
  // and the run still ends normally, just without a grid.
  const endRun = (reason: RunEndReason) => {
    if (acc && !attnBroken && stats) {
      try {
        emit({ type: 'attention-grid', layers: acc.layers, heads: acc.heads,
          cells: buildGridCells(acc, stats) })
        lastRun = { acc, stats }
      } catch { attnBroken = true }
    }
    emit({ type: 'run-end', reason })
  }
```

Then replace `emit({ type: 'run-end', reason: 'aborted' })` with `endRun('aborted')`, `emit({ type: 'run-end', reason: 'eos' })` with `endRun('eos')`, and `emit({ type: 'run-end', reason: 'max-tokens' })` with `endRun('max-tokens')`. (`acc` and `attnBroken` already sit before the cycle loop in the current code; only `stats`/`prevSel` are new declarations.)

f) Handle `head-request` in `self.onmessage`, after the `abort` line:

```ts
    if (msg.type === 'head-request') {
      const r = lastRun
      const ok = r !== null
        && msg.layer >= 0 && msg.layer < r.acc.layers
        && msg.head >= 0 && msg.head < r.acc.heads
      const resolved = ok ? resolveHeadLabel(r.stats, msg.layer, msg.head) : { label: null, score: null }
      post({ type: 'head-response', layer: msg.layer, head: msg.head,
        matrix: ok ? r.acc.rows[msg.layer][msg.head] : [],
        label: resolved.label, score: resolved.score })
    }
```

- [ ] **Step 5: Verify**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.
Run: `npx vitest run` — expected: PASS (worker itself has no unit harness; this confirms no module-level breakage).

- [ ] **Step 6: Commit**

```bash
git add src/engine/transformers/attentionStats.ts src/engine/transformers/attentionStats.test.ts src/engine/transformers/worker.ts
git commit -m "feat: worker emits attention grid, retains run for head requests, hysteresis across cycles"
```

---

### Task 7: `AttentionGridExplorer` component

**Files:**
- Create: `src/viz/AttentionGridExplorer.tsx`
- Modify: `src/index.css` (explorer styles)
- Test: `src/viz/AttentionGridExplorer.test.tsx`

**Interfaces:**
- Consumes: `Extract<TraceEvent, { type: 'attention-grid' }>`, `makeGridEvent` (tests).
- Produces: `export function AttentionGridExplorer({ grid, onPin }: { grid: Extract<TraceEvent, { type: 'attention-grid' }>; onPin: (layer: number, head: number) => void })`.
- Behavior: default sort `layer` renders one row per layer (top = L0) with a leading per-layer aggregate thumbnail (mean of the layer's head thumbs, computed client-side) followed by that layer's head cells; score sorts (`distinctive` / `previous-token` / `sink` / `induction`) render a flat list of all cells in descending score order without aggregates. Each cell is a 36 px canvas thumbnail in a button whose native `title` is ``L{l}·H{h} — {top statistic}``; clicking calls `onPin(layer, head)`. Header labels the semantics: "attention accumulated over the whole run".
- Canvas note: thumbnails draw one canvas pixel per bucket (`width = height = thumb.length`) upscaled by CSS with `image-rendering: pixelated`; in jsdom `getContext('2d')` returns null — the draw effect must guard on it so structure tests run without a canvas shim.

- [ ] **Step 1: Write the failing tests**

Create `src/viz/AttentionGridExplorer.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeGridEvent } from '../test/fixtures'
import { AttentionGridExplorer } from './AttentionGridExplorer'

afterEach(() => cleanup())

test('layer sort renders one row per layer with an aggregate and all cells', () => {
  render(<AttentionGridExplorer grid={makeGridEvent(2, 2)} onPin={() => {}} />)
  expect(screen.getAllByTestId('grid-row')).toHaveLength(2)
  expect(screen.getAllByTestId('grid-aggregate')).toHaveLength(2)
  expect(screen.getAllByTestId('grid-cell')).toHaveLength(4)
  expect(screen.getAllByTestId('grid-cell')[0]).toHaveTextContent('L0·H0')
  expect(screen.getByTestId('grid-explorer')).toHaveTextContent('attention accumulated over the whole run')
})

test('sorting by distinctive reorders cells and drops aggregates', () => {
  render(<AttentionGridExplorer grid={makeGridEvent(2, 2)} onPin={() => {}} />)
  fireEvent.change(screen.getByTestId('grid-sort'), { target: { value: 'distinctive' } })
  // makeGridEvent(2,2) distinctive scores: L0·H1=0.8 > L1·H0=0.4 > L0·H0=0.2 > L1·H1=0
  expect(screen.getAllByTestId('grid-cell')[0]).toHaveTextContent('L0·H1')
  expect(screen.queryAllByTestId('grid-aggregate')).toHaveLength(0)
})

test('clicking a cell pins it', () => {
  const onPin = vi.fn()
  render(<AttentionGridExplorer grid={makeGridEvent(2, 2)} onPin={onPin} />)
  fireEvent.click(screen.getAllByTestId('grid-cell')[3])
  expect(onPin).toHaveBeenCalledWith(1, 1)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/viz/AttentionGridExplorer.test.tsx`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the component**

Create `src/viz/AttentionGridExplorer.tsx`:

```tsx
import { useEffect, useRef, useState } from 'react'
import type { TraceEvent } from '../trace/types'

type GridEvent = Extract<TraceEvent, { type: 'attention-grid' }>
type GridCell = GridEvent['cells'][number]
type SortKey = 'layer' | 'distinctive' | 'previous-token' | 'sink' | 'induction'

const SORTS: SortKey[] = ['layer', 'distinctive', 'previous-token', 'sink', 'induction']

const SCORE_OF: Record<Exclude<SortKey, 'layer'>, (c: GridCell) => number> = {
  distinctive: (c) => c.distinctiveScore,
  'previous-token': (c) => c.prevTokenScore,
  sink: (c) => c.sinkScore,
  induction: (c) => c.inductionScore ?? 0,
}

function topStat(c: GridCell): string {
  const entries: Array<[string, number]> = [
    ['prev-token', c.prevTokenScore],
    ['sink', c.sinkScore],
    ['induction', c.inductionScore ?? 0],
    ['distinctive', c.distinctiveScore],
  ]
  entries.sort((a, b) => b[1] - a[1])
  return `${entries[0][0]} ${entries[0][1].toFixed(2)}`
}

// One canvas pixel per thumb bucket, upscaled by CSS (image-rendering:
// pixelated). Canvas here is a recorded amendment to the v1 "SVG + CSS only"
// constraint: 270 SVG thumbnails would jank; the main heatmap stays SVG.
function Thumb({ thumb }: { thumb: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return  // jsdom has no 2d context; structure still renders
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    thumb.forEach((row, r) => row.forEach((w, c) => {
      const v = Math.min(1, Math.max(0, w))
      ctx.fillStyle = `hsl(211 ${Math.round(30 + 25 * v)}% ${Math.round(94 - 70 * v)}%)`
      ctx.fillRect(c, r, 1, 1)
    }))
  }, [thumb])
  const dim = Math.max(thumb.length, 1)
  return <canvas ref={ref} width={dim} height={dim} className="grid-thumb" />
}

export function AttentionGridExplorer({ grid, onPin }: {
  grid: GridEvent
  onPin: (layer: number, head: number) => void
}) {
  const [sort, setSort] = useState<SortKey>('layer')

  const aggregate = (layer: number): number[][] => {
    const cells = grid.cells.filter((c) => c.layer === layer)
    const dim = cells[0]?.thumb.length ?? 0
    return Array.from({ length: dim }, (_, r) => Array.from({ length: dim }, (_, c) =>
      cells.reduce((sum, cell) => sum + (cell.thumb[r]?.[c] ?? 0), 0) / cells.length))
  }

  const cellButton = (c: GridCell) => (
    <button key={`${c.layer}-${c.head}`} data-testid="grid-cell" className="grid-cell"
      title={`L${c.layer}·H${c.head} — ${topStat(c)}`} onClick={() => onPin(c.layer, c.head)}>
      <Thumb thumb={c.thumb} />
      <span className="grid-loc">L{c.layer}·H{c.head}</span>
    </button>
  )

  return (
    <div data-testid="grid-explorer" className="grid-explorer">
      <div className="grid-explorer-header">
        <span className="grid-semantics">attention accumulated over the whole run</span>
        <label className="grid-sort-label">sort{' '}
          <select data-testid="grid-sort" value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}>
            {SORTS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
      </div>
      {sort === 'layer' ? (
        <div className="grid-scroll">
          {Array.from({ length: grid.layers }, (_, l) => (
            <div key={l} data-testid="grid-row" className="grid-row">
              <div data-testid="grid-aggregate" className="grid-aggregate" title={`L${l} — mean of heads`}>
                <Thumb thumb={aggregate(l)} />
                <span className="grid-loc">L{l} ∅</span>
              </div>
              {grid.cells.filter((c) => c.layer === l).map(cellButton)}
            </div>
          ))}
        </div>
      ) : (
        <div className="grid-scroll grid-flat">
          {[...grid.cells].sort((a, b) => SCORE_OF[sort](b) - SCORE_OF[sort](a)).map(cellButton)}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Add the styles**

Append to `src/index.css` (after the `.head-chip-row` block region):

```css
.grid-explorer { margin-top: .6rem; border-top: 1px dashed var(--line); padding-top: .5rem; }
.grid-explorer-header { display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; flex-wrap: wrap; }
.grid-semantics { font-size: .78rem; color: var(--ink-soft); font-style: italic; }
.grid-sort-label { font-size: .8rem; color: var(--ink-soft); }
.grid-sort-label select { font: inherit; background: var(--card); border: 1px solid var(--line-strong); border-radius: 6px; padding: .1rem .25rem; }
.grid-scroll { max-height: 24rem; overflow-y: auto; margin-top: .5rem; }
.grid-row { display: flex; gap: 3px; margin-bottom: 3px; align-items: flex-start; }
.grid-flat { display: flex; flex-wrap: wrap; gap: 3px; }
.grid-cell, .grid-aggregate { display: flex; flex-direction: column; align-items: center; gap: 1px;
  border: 1px solid transparent; border-radius: 4px; background: none; padding: 2px; font: inherit; }
.grid-cell { cursor: pointer; }
.grid-cell:hover { border-color: var(--shu); }
.grid-aggregate { border-color: var(--line); background: var(--ai-wash); margin-right: .4rem; cursor: help; }
.grid-thumb { width: 36px; height: 36px; image-rendering: pixelated; border: 1px solid var(--line); background: var(--card); }
.grid-loc { font-family: var(--mono); font-size: .58rem; color: var(--ink-faint); }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/viz/AttentionGridExplorer.test.tsx`
Expected: PASS.

- [ ] **Step 6: Typecheck and commit**

Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/viz/AttentionGridExplorer.tsx src/viz/AttentionGridExplorer.test.tsx src/index.css
git commit -m "feat: attention-head grid explorer with aggregates, sorting, canvas thumbnails"
```

---

### Task 8: Pin flow and wiring — `usePins`, `LayersDetail`, `DetailPanel`, `App`

**Files:**
- Create: `src/app/usePins.ts`
- Modify: `src/viz/details/LayersDetail.tsx`
- Modify: `src/viz/DetailPanel.tsx` (layers case, props)
- Modify: `src/App.tsx` (pin state wiring)
- Test: `src/app/usePins.test.ts`, `src/viz/DetailPanel.test.tsx`

**Interfaces:**
- Consumes: `HeadData` + `fetchHead` (Task 5), `AttentionGridExplorer` (Task 7), `makeGridEvent` (Task 3), `latestOfType` from `../viz/selectors`.
- Produces:
  - `export function usePins(fetchHead: (layer: number, head: number) => Promise<HeadData>): { pins: AttentionHead[]; note: string | null; pin: (layer: number, head: number) => Promise<void>; reset: () => void }` — duplicate (layer, head) ignored; empty-matrix response sets `note` to `'run data no longer available — regenerate to explore heads'` and adds nothing; successful pin clears the note and appends `{ layer, head, label: label ?? 'pinned', score?, matrix }`; max 5 pins FIFO (6th evicts the 1st); `reset` clears both.
  - `LayersDetail` new optional props: `grid?: Extract<TraceEvent, { type: 'attention-grid' }>`, `pinnedHeads?: AttentionHead[]`, `onPin?: (layer: number, head: number) => void`, `pinNote?: string | null`.
  - `DetailPanel` new optional props: `pinnedHeads?: AttentionHead[]`, `onPin?: (layer: number, head: number) => void`, `pinNote?: string | null` — threaded to `LayersDetail` in the layers case; the grid event is found whole-trace via `latestOfType(events, events.length - 1, 'attention-grid')` (same run-level idiom as `attentionInRun`).
- Note on pin-state location (recorded ruling): pins live in App state via `usePins`, not inside `LayersDetail` — `LayersDetail` unmounts whenever the cursor leaves the layers stage, and the spec requires pins to survive until "a new run replaces the trace". App resets them in `handleGenerate`, right where the trace store is cleared. Viz components stay store-free either way.

- [ ] **Step 1: Write the failing `usePins` tests**

Create `src/app/usePins.test.ts`:

```ts
import { act, renderHook } from '@testing-library/react'
import { expect, test } from 'vitest'
import type { HeadData } from '../engine/transformers/TransformersEngine'
import { usePins } from './usePins'

const data = (layer: number, head: number, extra?: Partial<HeadData>): HeadData => ({
  layer, head, matrix: [[1], [0.5, 0.5]], label: null, score: null, ...extra,
})

test('pin fetches and appends with the pinned fallback label', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  await act(() => result.current.pin(3, 1))
  expect(result.current.pins).toHaveLength(1)
  expect(result.current.pins[0]).toMatchObject({ layer: 3, head: 1, label: 'pinned' })
  expect(result.current.pins[0].matrix).toEqual([[1], [0.5, 0.5]])
  expect(result.current.note).toBeNull()
})

test('a resolved template label and score are kept', async () => {
  const { result } = renderHook(() =>
    usePins(async (l, h) => data(l, h, { label: 'previous-token', score: 0.8 })))
  await act(() => result.current.pin(0, 0))
  expect(result.current.pins[0]).toMatchObject({ label: 'previous-token', score: 0.8 })
})

test('duplicate pins are ignored', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  await act(() => result.current.pin(1, 1))
  await act(() => result.current.pin(1, 1))
  expect(result.current.pins).toHaveLength(1)
})

test('an empty matrix sets the stale note and adds nothing', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h, { matrix: [] })))
  await act(() => result.current.pin(0, 0))
  expect(result.current.pins).toHaveLength(0)
  expect(result.current.note).toBe('run data no longer available — regenerate to explore heads')
})

test('the sixth pin evicts the first (FIFO cap of 5)', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  for (let h = 0; h < 6; h++) await act(() => result.current.pin(0, h))
  expect(result.current.pins).toHaveLength(5)
  expect(result.current.pins.map((p) => p.head)).toEqual([1, 2, 3, 4, 5])
})

test('reset clears pins and note', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  await act(() => result.current.pin(0, 0))
  act(() => result.current.reset())
  expect(result.current.pins).toHaveLength(0)
  expect(result.current.note).toBeNull()
})
```

Run: `npx vitest run src/app/usePins.test.ts` — expected: FAIL, module does not exist.

- [ ] **Step 2: Implement `usePins`**

Create `src/app/usePins.ts`:

```ts
import { useCallback, useRef, useState } from 'react'
import type { HeadData } from '../engine/transformers/TransformersEngine'
import type { AttentionHead } from '../trace/types'

const STALE_NOTE = 'run data no longer available — regenerate to explore heads'
const MAX_PINS = 5

// Pinned heads are run-scoped UI state: App resets them when a new run
// replaces the trace (they must survive scrubbing away from the Layers stage,
// so they cannot live inside LayersDetail). A ref mirrors the pin list so the
// duplicate check reads current state synchronously — a setState-updater
// read is not guaranteed to run before the next line.
export function usePins(fetchHead: (layer: number, head: number) => Promise<HeadData>) {
  const [pins, setPins] = useState<AttentionHead[]>([])
  const [note, setNote] = useState<string | null>(null)
  const pinsRef = useRef<AttentionHead[]>([])
  const fetchRef = useRef(fetchHead)
  fetchRef.current = fetchHead

  const commit = useCallback((next: AttentionHead[]) => {
    pinsRef.current = next
    setPins(next)
  }, [])

  const pin = useCallback(async (layer: number, head: number) => {
    if (pinsRef.current.some((p) => p.layer === layer && p.head === head)) return
    const r = await fetchRef.current(layer, head)
    if (r.matrix.length === 0) { setNote(STALE_NOTE); return }
    if (pinsRef.current.some((p) => p.layer === layer && p.head === head)) return
    setNote(null)
    const next: AttentionHead = { layer, head, label: r.label ?? 'pinned',
      ...(r.score != null ? { score: r.score } : {}), matrix: r.matrix }
    commit([...pinsRef.current, next].slice(-MAX_PINS))
  }, [commit])

  const reset = useCallback(() => { commit([]); setNote(null) }, [commit])

  return { pins, note, pin, reset }
}
```

Run: `npx vitest run src/app/usePins.test.ts` — expected: PASS.

- [ ] **Step 3: Write the failing wiring tests**

Append to `src/viz/DetailPanel.test.tsx` (extend the fixtures import to include `makeGridEvent`, the vitest import to include `vi`, the testing-library import to include `fireEvent`, and add `import type { AttentionHead } from '../trace/types'`):

```tsx
function traceWithGrid() {
  const t = makeFixtureTrace()
  t.splice(t.length - 1, 0, makeGridEvent(2, 2))
  return t
}

test('layers detail offers the explorer toggle when the run has a grid', () => {
  render(<DetailPanel events={traceWithGrid()} cursor={3} mode="real" />)
  const toggle = screen.getByTestId('btn-explore-heads')
  expect(toggle).toHaveTextContent('Explore all heads (4)')
  fireEvent.click(toggle)
  expect(screen.getByTestId('grid-explorer')).toBeInTheDocument()
})

test('no explorer toggle without a grid event', () => {
  render(<DetailPanel events={trace} cursor={3} mode="sim" />)
  expect(screen.queryByTestId('btn-explore-heads')).toBeNull()
})

test('cell clicks reach onPin and pinned heads render as chips', () => {
  const onPin = vi.fn()
  const pinned: AttentionHead[] = [{ layer: 1, head: 1, label: 'pinned', matrix: [[1], [0.5, 0.5]] }]
  render(<DetailPanel events={traceWithGrid()} cursor={6} mode="real"
    pinnedHeads={pinned} onPin={onPin} pinNote={null} />)
  fireEvent.click(screen.getByTestId('btn-explore-heads'))
  fireEvent.click(screen.getAllByTestId('grid-cell')[0])
  expect(onPin).toHaveBeenCalledWith(0, 0)
  const chips = screen.getAllByTestId('head-chip')
  expect(chips.some((c) => c.textContent?.includes('pinned'))).toBe(true)
})

test('the stale-pin note renders', () => {
  render(<DetailPanel events={traceWithGrid()} cursor={3} mode="real"
    pinNote="run data no longer available — regenerate to explore heads" />)
  expect(screen.getByTestId('pin-note')).toHaveTextContent('regenerate')
})
```

Run: `npx vitest run src/viz/DetailPanel.test.tsx` — expected: FAIL (no toggle testid, unknown props).

- [ ] **Step 4: Implement `LayersDetail` and `DetailPanel` wiring**

Replace `src/viz/details/LayersDetail.tsx` with:

```tsx
import { useState } from 'react'
import type { AttentionHead, Mode, TokenInfo, TraceEvent } from '../../trace/types'
import { AttentionGridExplorer } from '../AttentionGridExplorer'
import { AttentionHeatmap } from '../AttentionHeatmap'
import { ResidualStreamDiagram } from '../ResidualStreamDiagram'

export function LayersDetail({ event, mode, attention, attentionInRun, tokens, streamShape, grid, pinnedHeads, onPin, pinNote }: {
  event: Extract<TraceEvent, { type: 'layer' }>
  mode: Mode
  attention?: Extract<TraceEvent, { type: 'attention' }>
  attentionInRun?: boolean
  tokens?: TokenInfo[]
  streamShape?: { seqLen: number; dims: number }
  grid?: Extract<TraceEvent, { type: 'attention-grid' }>
  pinnedHeads?: AttentionHead[]
  onPin?: (layer: number, head: number) => void
  pinNote?: string | null
}) {
  const [explore, setExplore] = useState(false)
  const cycleHeads = attention?.heads ?? []
  // a pinned head that is also this cycle's showcase head under the same
  // label would duplicate its chip (and its React key) — show it once
  const heads = [...cycleHeads, ...(pinnedHeads ?? []).filter((p) =>
    !cycleHeads.some((h) => h.layer === p.layer && h.head === p.head && h.label === p.label))]
  return (
    <div data-testid="detail-layers" className="detail">
      <h3>Transformer layers {mode === 'real' && !attentionInRun && <em>(schematic — real internals not exposed)</em>}</h3>
      {streamShape && (
        <ResidualStreamDiagram seqLen={streamShape.seqLen} dims={streamShape.dims} layers={event.total} />
      )}
      <div className="layer-stack">
        {Array.from({ length: event.total }, (_, i) => (
          <div key={i} data-testid="layer-block" data-lit={String(i <= event.index)} className="layer-block">
            L{i}{mode === 'sim' && i === event.index && event.activationNorm != null && (
              <span className="layer-norm"> ‖h‖ {event.activationNorm}</span>
            )}
          </div>
        ))}
      </div>
      {heads.length > 0 && tokens && <AttentionHeatmap heads={heads} tokens={tokens} />}
      {pinNote && <p data-testid="pin-note" className="attn-note">{pinNote}</p>}
      {grid && (
        <button data-testid="btn-explore-heads" className="explore-toggle"
          onClick={() => setExplore((v) => !v)}>
          {explore ? 'Hide head grid' : `Explore all heads (${grid.layers * grid.heads})`}
        </button>
      )}
      {grid && explore && <AttentionGridExplorer grid={grid} onPin={onPin ?? (() => {})} />}
    </div>
  )
}
```

(The toggle is gated on `grid` alone — not `grid && onPin` — because the explorer's presence is a run-level fact; `onPin` is an optional prop and defaults to a no-op at the mount site.)

In `src/viz/DetailPanel.tsx`:

```tsx
export function DetailPanel({ events, cursor, mode, pinnedHeads, onPin, pinNote }: {
  events: TraceEvent[]; cursor: number; mode: Mode
  pinnedHeads?: AttentionHead[]
  onPin?: (layer: number, head: number) => void
  pinNote?: string | null
}) {
```

(add `AttentionHead` to the type import from `../trace/types`), and in the `layers` case, before the return:

```tsx
      // run-level, like attentionInRun: the grid is a whole-run snapshot
      const grid = latestOfType(events, events.length - 1, 'attention-grid')
```

and extend the return:

```tsx
      return <LayersDetail event={e} mode={mode} attention={inCycle} attentionInRun={attentionInRun}
        tokens={tokens} streamShape={streamShape}
        grid={grid} pinnedHeads={pinnedHeads} onPin={onPin} pinNote={pinNote} />
```

Also add the explore-toggle style to `src/index.css` (next to the Task 7 block):

```css
.explore-toggle { margin-top: .6rem; font: inherit; font-size: .8rem; color: var(--ai-deep);
  background: var(--ai-wash); border: 1px solid var(--line-strong); border-radius: 6px;
  padding: .25rem .6rem; cursor: pointer; }
.explore-toggle:hover { background: var(--ai-wash-strong); }
```

- [ ] **Step 5: Run the wiring tests**

Run: `npx vitest run src/viz/DetailPanel.test.tsx src/viz/AttentionGridExplorer.test.tsx`
Expected: PASS. Note for the pinned-chip test: the AttentionHeatmap tokens list at cursor 6 is shorter than a pinned matrix would need — labels fall back to `#i`, which is acceptable; the chip itself must render.

- [ ] **Step 6: Wire `App.tsx`**

In `src/App.tsx`:

```tsx
import { usePins } from './app/usePins'
```

Inside the component, after the `attn` state:

```tsx
  const { pins, note: pinNote, pin: handlePin, reset: resetPins } = usePins((layer, head) => {
    const engine = realEngineRef.current
    return engine
      ? engine.fetchHead(layer, head)
      : Promise.resolve({ layer, head, matrix: [], label: null, score: null })
  })
```

In `handleGenerate`, directly after `usePlayerStore.getState().dispatch({ type: 'reset' })`:

```tsx
    resetPins()
```

And pass the props:

```tsx
      <DetailPanel events={events} cursor={cursor} mode={mode}
        pinnedHeads={pins} onPin={handlePin} pinNote={pinNote} />
```

- [ ] **Step 7: Full verification and commit**

Run: `npx vitest run` — expected: PASS.
Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/app/usePins.ts src/app/usePins.test.ts src/viz/details/LayersDetail.tsx src/viz/DetailPanel.tsx src/App.tsx src/viz/DetailPanel.test.tsx src/index.css
git commit -m "feat: head-explorer wiring — pin flow, explore toggle, run-scoped pin state"
```

---

### Task 9: Sticky playback controls above the detail panel; v1 spec amendment

**Files:**
- Modify: `src/App.tsx` (JSX order only)
- Modify: `src/index.css:188` (`.controls`)
- Modify: `docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md` (one-line amendment after the layout sketch, which ends at line 214)

No new tests: the Controls component's behavior and test-ids are unchanged; `src/app/Controls.test.tsx` must keep passing as-is.

- [ ] **Step 1: Reorder the JSX**

In `src/App.tsx`, move `<Controls />` from after `<DetailPanel …/>` to directly before it (order: `PipelineBand` → `Controls` → `DetailPanel`).

- [ ] **Step 2: Make the controls sticky**

In `src/index.css`, replace the `.controls` rule:

```css
.controls { display: flex; gap: .5rem; align-items: center; padding: .6rem 0; flex-wrap: wrap;
  position: sticky; top: 0; z-index: 3; background: var(--washi);
  border-bottom: 1px solid var(--line); }
```

- [ ] **Step 3: Amend the v1 spec's layout sketch**

In `docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md`, directly after the closing ``` of the layout sketch (line 214), insert:

```markdown
> **Amendment (2026-08-28, M2):** the player-controls row moved above the
> detail panel and became sticky — the panel's height varies too much
> (especially with the head-explorer grid) for a below-panel scrubber to be
> an easy target.
```

- [ ] **Step 4: Verify and commit**

Run: `npx vitest run` — expected: PASS (Controls tests unchanged).
Run: `npx tsc --noEmit -p tsconfig.app.json` — expected: no errors.

```bash
git add src/App.tsx src/index.css docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md
git commit -m "feat: sticky playback controls above the detail panel"
```

---

### Task 10: Full verification sweep

**Files:** none new — fixes only if a check fails.

- [ ] **Step 1: Typecheck**

Run: `npx tsc --noEmit -p tsconfig.app.json`
Expected: no errors.

- [ ] **Step 2: Full unit suite**

Run: `npx vitest run`
Expected: all tests pass.

- [ ] **Step 3: E2E (sim-only, unchanged)**

Run: `npx playwright test`
Expected: all pass. The Controls move is the only change visible to e2e; its selectors are test-ids and order-independent. If a test fails, fix the regression (never the e2e expectations) and re-run.

- [ ] **Step 4: Production build sanity**

Run: `npx vite build`
Expected: builds cleanly.

- [ ] **Step 5: Commit any fixes**

If Steps 1–4 required fixes:

```bash
git add -A
git commit -m "fix: verification sweep fallout for the head explorer"
```

Otherwise, nothing to commit.

**Manual browser verification (operator, post-merge or on the branch):** real mode on a fast network — grid opens with 270 thumbnails, visibly diverse patterns beyond the diagonal/sink chips, sort by distinctive surfaces non-template heads, pinning renders exact matrices as chips, stale pin after a re-run shows the regenerate note, controls stay put while scrolling the grid.
