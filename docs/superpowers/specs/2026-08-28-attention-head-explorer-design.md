# Attention-head explorer (M2) — design

**Date:** 2026-08-28
**Status:** Approved design, pre-implementation
**Builds on:** `docs/superpowers/specs/2026-08-27-real-attention-export-design.md`
(M1, shipped: real per-layer attention in real mode, statistical head
detection, ≤3 showcase heads per cycle in the `attention` trace event)

## Purpose

M1 surfaces the top-scoring head per known role — by construction the most
textbook-looking specimens. M2 makes the rest of the model explorable:

1. A **small-multiples grid** of all layers × heads (SmolLM2: 30 × 9 = 270)
   as thumbnail heatmaps, browsable and sortable, with per-layer aggregates.
2. **Pinning**: any grid head opens full-size in the existing heatmap viewer.
3. A fourth detected role, **distinctive** — the head least like any
   template — fixing the "detector optimized for boring" selection bias.
4. **Selection hysteresis** so showcase chips stop flickering across cycles.
5. A layout fix motivated by the explorer's height: the playback controls
   move above the detail panel and become sticky.

## Decisions already made

- Explorer lives **inside the Layers detail panel** (expand-in-place toggle),
  not an overlay or separate view.
- Clicking a thumbnail **pins the head into the existing viewer** (chip +
  full-size heatmap with all existing affordances).
- **Real mode only** — sim mode's 3–4 hand-shaped heads would make a grid
  misleading theater; the toggle appears only when measured data exists.
- **Data architecture: Approach C** — one `attention-grid` trace event at run
  end carries browsing data (thumbnails + stats for all heads, ~90 KB); pin
  exactness comes from a single on-demand worker request. The pipeline
  replay stays a pure function of `(trace, cursor)`; the explorer is
  explicitly a **run-level snapshot** ("attention over the whole run"),
  does not change while scrubbing, and is labeled as such in the UI.

## Component 1: the `attention-grid` trace event

Emitted **once per run**, real mode only, **immediately before `run-end`**,
on every exit path where the accumulator exists and attention is not broken
(mirrors M1's never-fail policy: grid emission failures are swallowed, never
fatal):

```ts
| { type: 'attention-grid'; layers: number; heads: number;
    cells: Array<{
      layer: number; head: number
      thumb: number[][]              // ≤12×12 mean-pooled causal thumbnail, values 0..1
      prevTokenScore: number; sinkScore: number
      inductionScore: number | null; distinctiveScore: number
    }> }
```

- Thumbnails are **mean-pooled** block averages of the accumulated ragged
  matrix (12 row-buckets × 12 column-buckets, fewer when the sequence is
  shorter than 12; approximately triangular). Mean pooling preserves
  relative mass; max pooling would make every head look like its brightest
  cell. Rows do NOT sum to 1 after pooling — by design.
- Size ≈ 90 KB for 270 heads; acceptable as a one-per-run trace event.
- `validateTrace` learns: when present, the event must sit directly before
  `run-end`; `cells.length === layers × heads`; thumb values within
  [0, 1] and thumb dims ≤ 12. No row-stochasticity check.
- **The default test fixture (`makeFixtureTrace`) does NOT gain this event**
  — that would shift the index-based tests again. A separate
  `makeGridEvent(layers, heads)` helper serves explorer/component tests.

## Component 2: protocol — the one bounded exception to trace-only data

```ts
WorkerRequest:  | { type: 'head-request'; layer: number; head: number }
WorkerResponse: | { type: 'head-response'; layer: number; head: number;
                    matrix: number[][]           // exact accumulated ragged matrix, [] if unavailable
                    label: AttentionLabel | null // best template ≥ 0.3, else null
                    score: number | null }
```

- The worker retains the finished run's accumulator + stats until the next
  run starts (memory already bounded by `ATTN_MAX_SEQ`).
- Unavailable data (new run started, indices out of range, no accumulator)
  → `matrix: []`; the UI renders "run data no longer available —
  regenerate", never an error state.
- `TransformersEngine` gains `fetchHead(layer, head): Promise<HeadResponse>`
  (correlate responses by layer+head; last-request-wins is sufficient).
- This is the only data path beside the trace; it exists solely for pin
  exactness and must not grow other uses without a new design.

## Component 3: worker — grid building, distinctive score, hysteresis

- New pure module `src/engine/transformers/attentionThumbs.ts`:
  `poolThumb(matrix, buckets=12): number[][]` and
  `buildGridCells(acc, stats): GridCell[]` — unit-testable without a model.
- **Distinctive score** (in `attentionStats.ts`):
  `distinctive = (1 − templateMax) × (1 − uniformity)` where
  `templateMax = max(prevTokenScore, sinkScore, inductionScore ?? 0)` and
  `uniformity` = mean row entropy normalized by `log(rowLength)` (rows
  i ≥ 1; single-row heads score 0). High = peaked attention matching no
  template. `AttentionLabel` gains `'distinctive'`; `selectShowcaseHeads`
  picks a fourth chip for it (threshold 0.25) — the M1 "≤3 heads per
  cycle" bound becomes **≤4**, still trace-size-safe. `AttentionHeatmap.HINTS`
  gains: "Focused attention that fits no textbook pattern — look for what
  it tracks." Sim mode never emits it (sim engine untouched). The
  hysteresis parameter is optional; existing call sites (and sim) omit it
  unchanged.
- **Selection hysteresis** in `selectShowcaseHeads`: the function takes the
  previous cycle's selection (label → {layer, head}) and keeps the
  incumbent unless a challenger's score exceeds the incumbent's **current**
  score by ≥ 0.05. The worker threads the previous selection through the
  cycle loop; sim mode passes nothing (unchanged behavior, deterministic
  patterns don't flicker anyway).

## Component 4: explorer UI

- `LayersDetail` shows an "**Explore all heads (N)**" toggle button under
  the chip row whenever the run's trace contains an `attention-grid` event
  (found whole-trace, like `attentionInRun`). Sim mode never has one.
- New component `src/viz/AttentionGridExplorer.tsx`:
  - Grid: **rows = layers** (top = L0), **columns = heads**, one 36 px
    thumbnail per cell; a leading **aggregate column** shows each layer's
    mean-of-heads thumbnail (computed client-side from the cells). Vertical
    scroll inside a max-height container.
  - **Canvas rendering for thumbnails** — a deliberate, recorded amendment
    to v1's "SVG + CSS only" constraint: 270 SVG thumbnails ≈ 38k DOM
    nodes would jank; the constraint's own justification ("data volumes
    are tiny") stops holding exactly here. The main heatmap stays SVG.
  - Header labels the semantics: "attention accumulated over the whole run".
  - Sort control: by layer (default) / distinctive / previous-token / sink /
    induction score.
  - Hover: highlight + `L{l}·H{h}` and the head's top statistic (native
    title).
  - **Click pins**: calls the `onFetchHead(layer, head)` callback prop
    (wired from App via `DetailPanel` — viz components stay store-free,
    same pattern as `onStageClick`); the exact matrix becomes a chip in the
    head-chip row labeled by the response's `label`, falling back to a new
    `'pinned'` member of `AttentionLabel` (hint: "hand-picked from the
    grid — compare against the patterns you know"), selected and fully
    usable in the existing viewer.
    Max **5 pins, FIFO**; pinned heads are component state and reset when a
    new run replaces the trace.

## Component 5: stable playback controls (layout change)

The `Controls` row moves from below the detail panel to **directly above
it** (order: pipeline band → controls → detail panel), so its position no
longer depends on the panel's highly variable height. The row is
**`position: sticky; top: 0`** with the washi background and a hairline
bottom border, staying reachable while scrolling the explorer grid —
scrubbing while watching thumbnails works. One JSX reorder in `App.tsx` +
CSS; no test-id or behavior changes. The v1 spec's layout sketch gains a
one-line amendment noting the move and its reason.

## Error handling

- No grid event (sim, attention broken, over `ATTN_MAX_SEQ`) → no toggle.
- Stale/failed pin fetch → inline note in the chip row area, nothing else;
  fetch failures never affect generation or the pipeline view (M1 policy).
- Grid emission is wrapped like all attention work: failure flips
  `attnBroken`, generation continues, no grid appears.

## Testing

- Pure modules (CI): `poolThumb` (block means, causal-ish shape, [0,1],
  short-sequence degenerate cases), `buildGridCells` (cell count, ordering),
  distinctive score (template heads low; crafted focused-but-untemplated
  head high; uniform head low), hysteresis (sticky under ±0.04 wobble,
  switches at ≥0.05, incumbent disappearing falls back to argmax).
- Validator: grid-event placement + bounds tests.
- Protocol: `head-request`/`head-response` round-trip and the
  stale-request (`matrix: []`) path in the FakeWorker suite.
- Components: explorer with a small synthetic `makeGridEvent(2, 2)` —
  toggle renders the grid, cell count and aggregate column correct, sort
  reorders, pin flow via a mocked `onFetchHead` adds a chip and renders
  the fetched matrix, 6th pin evicts the 1st, stale fetch shows the note.
  Controls-position change needs no new tests (no behavior change).
- E2E: unchanged, sim-only.
- Manual browser verification (real mode, fast network): grid opens with
  270 thumbnails, visibly diverse patterns beyond the diagonal/sink chips,
  sort by distinctive surfaces non-template heads, pinning works, controls
  stay put while scrolling.

## Out of scope (M3 and later)

- Per-cycle grid evolution (grid is a run-level snapshot by design)
- Cross-run comparison of grids (pairs with the trace-archive backlog item)
- Automated coreference-head discovery
- Any additional uses of the head-request side channel
