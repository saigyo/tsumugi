# Trace archive with run comparison (M3) — design

**Date:** 2026-08-28
**Status:** Approved design, pre-implementation
**Builds on:** `docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md` (v1
trace/player architecture), `docs/superpowers/specs/2026-08-28-attention-head-explorer-design.md`
(M2: attention-grid event, run-scoped pins)

## Purpose

A completed run is already a self-contained, serializable artifact: the trace
is a plain append-only event list and the UI replays it as a pure function of
`(trace, cursor)`. M3 stops throwing that artifact away. It delivers:

1. A **run archive**: completed runs are kept automatically (ring buffer),
   flippable via a run shelf, protectable with per-run pins, persistent
   across reloads, and exportable/importable as JSON files.
2. **Cycle-aligned comparison**: select two archived runs and study where and
   why they differ — token streams aligned by cycle with the fork marked,
   paired top-k distributions, paired attention heads.

The educational payoff is every question of the form "what changed between
these two runs": temperature/top-k effects ("same beliefs, different draw"
vs "different beliefs"), prompt surgery, sampling stochasticity, sim vs
real, head stability across runs.

## Decisions already made

- **Scope:** archive + comparison land together in M3.
- **Archiving:** automatic — every completed run enters a ring buffer;
  per-run **pin** exempts a run from eviction (unpinned cap: 8).
- **Persistence:** IndexedDB, plus per-run **export/import** as JSON files.
- **Interaction model:** comparison is **cycle-indexed inspection**, not
  playback — no second scrubber, no dual animation.
- **Panels:** metadata header + paired distributions + paired attention.
  Grid side-by-side and probability-delta views are out of scope.
- **Layout (mockup Option A):** a **dedicated compare view** replaces
  everything below the run shelf; the player, pipeline band, and scrubber
  are hidden while comparing. One action exits back to the player.
- **Trace-only comparison:** archived runs never talk to the worker. The M2
  head-request side channel remains scoped to the latest live run; compare
  draws exclusively on trace-resident data (per-cycle showcase matrices,
  run-level grid thumbnails). Live, reloaded, and imported runs therefore
  compare identically.

## Component 1: run records and the runs store

```ts
export interface RunMeta {
  seq: number               // monotonic per-archive counter assigned at seal; never reused,
                            // so chip labels stay stable across evictions
  prompt: string
  params: GenParams
  mode: Mode
  modelId?: string          // real mode only
  endedAt: number           // epoch ms, stamped at seal time
  reason: RunEndReason
  pinned: boolean
}
export interface RunRecord { id: string; meta: RunMeta; events: TraceEvent[] }
```

New Zustand store `src/app/runsStore.ts`:

- `records: RunRecord[]` (oldest first), `activeId: string | null`.
- `seal(meta, events)` — called once per run on `run-end`: wraps the live
  trace into a `RunRecord` (id: `crypto.randomUUID()`), appends it, makes it
  active, then evicts the **oldest unpinned** record while the unpinned
  count exceeds 8. Pinned records never count toward the cap and are never
  evicted.
- `setActive(id)`, `togglePin(id)`, `remove(id)` (explicit delete via the
  chip, pinned or not), `importRecord(record)` (appends **pinned**, active
  unchanged).
- The engine/trace-store flow is untouched upstream: the live trace buffer
  fills exactly as today; sealing copies the completed event list into the
  archive. The player replays `records.find(activeId).events` — selectors,
  DetailPanel, and all viz components are unchanged for single-run viewing.
- M2 pin state (`usePins`) resets when the USER switches runs — on Generate
  and on shelf activation, not on the seal-driven `activeId` change at
  run-end (a blanket activeId effect would wipe pins made mid-run; found and
  corrected in the final branch review).

## Component 2: persistence and files

- Storage adapter interface `src/app/runStorage.ts`:
  `{ loadAll(): Promise<RunRecord[]>; put(r: RunRecord): Promise<void>;
  delete(id: string): Promise<void> }` — IndexedDB implementation
  (database `tsumugi`, object store `runs`, key `id`); a fake in-memory
  implementation serves tests.
- Write-through: seal, pin toggle, eviction, and removal mirror to storage;
  startup hydrates the store (records sorted by `endedAt`, `activeId` left
  null until the user picks a run or generates).
- **All storage failures are non-fatal**: catch, surface one small shelf
  note ("archive not persisted in this browser"), continue session-only.
  Persistence must never block or delay generation (M1's never-fail
  spirit).
- **Export**: chip action downloads
  `{ formatVersion: 1, meta, events }` as
  `tsumugi-run-<first-words>-<yyyymmdd>.json`.
- **Import**: file picker → JSON parse → structural check
  (`formatVersion === 1`, `meta` fields present) → `validateTrace(events)`
  must return no errors → `importRecord` (arrives **pinned** so it cannot
  silently fall off the ring). Any failure: inline error near the shelf,
  nothing imported.

## Component 3: run shelf

A thin strip between the prompt-bar rows and the pipeline band:

- One chip per record, oldest left: `#<seq> · <first words of prompt> ·
  T=<t>` plus a mode glyph (sim/real) and 📌 when pinned. Auto-label only —
  no renaming. (`seq` comes from the archive counter, so labels never shift
  when older runs are evicted. Imported records get the next local seq.)
- Click: make active — the player loads that run with the cursor parked at
  `run-end` (scrub freely from there). Chip actions: pin toggle, export,
  remove. An import affordance sits at the strip's end.
- **Compare entry:** a "compare" toggle on the shelf arms selection with
  the active run pre-selected as A; the next chip click selects B and
  opens the compare view (while comparing, any further chip click simply
  re-selects B; A stays the run that was active at entry). While comparing, the shelf stays
  visible with A/B marked; clicking the exit control (or toggling compare
  off) returns to the player on the active run.
- Empty archive: strip renders nothing (first run creates it). One record:
  chips render without the compare affordance.

## Component 4: compare view (dedicated, below the shelf)

`src/viz/compare/CompareView.tsx` replaces the pipeline band, controls, and
detail panel while active. Pure props: the two `RunRecord`s; internal state
is only the selected cycle and selected head. Sections top to bottom:

1. **Metadata header** — both runs' prompt, T, top-k, max, mode, model,
   end reason; fields that differ are highlighted (vermillion accent).
2. **Aligned token streams** — run A above run B, aligned by cycle index
   (prompt tokens left of a divider, one generated token per cycle). A
   shared **cycle ruler** beneath (clickable ticks). When both prompts are
   token-identical, the **fork cycle** — the first cycle whose chosen token
   differs — is highlighted on both streams and the ruler; runs that never
   diverge get a "identical outputs" note. When prompts differ, no fork
   marker: a badge reads "different prompts — aligned by generation cycle".
   Streams of different lengths simply end at different ticks.
3. **Paired distributions** (on cycle selection) — both runs' `softmax`
   top-k for that cycle as two aligned bar columns, chosen token marked;
   temperature shown per side. Data read directly from each trace's events.
4. **Paired attention** (on cycle selection) — chips are the union of both
   runs' showcase heads for that cycle, grouped by `L·H` (a head present
   in both runs is one chip). Selecting a chip renders both runs' matrices
   side by side (existing `AttentionHeatmap`, one per side, with the
   measured caption). When one run has no full matrix for that head at
   that cycle, its side falls back to the run-level grid thumbnail
   (upscaled canvas, labeled "run-level thumbnail — full matrix not
   captured in this run") or, with no grid either (sim), a "not captured
   in this run" note. Sim runs show their existing illustrative caption.
5. Cycles beyond a run's end render that run's panel side as "run ended at
   cycle N".

New pure selectors in `src/viz/compare/compareSelectors.ts`:
`alignRuns(a, b)` → `{ cyclesA, cyclesB, maxCycles, samePrompt, forkCycle:
number | null }` (built on `cycleTickIndices`); `pairedDistributions(a, b,
cycle)`; `pairedHeads(a, b, cycle)`. All unit-testable on fixture traces.

## Error handling

- IndexedDB unavailable/failing → session-only archive + one shelf note.
- Import failures → inline error, archive untouched.
- Compare with a cycle out of range for one run → explicit "run ended"
  panel side, never an empty crash.
- Sealing happens on every `run-end` reason (eos, max-tokens, aborted) —
  aborted runs archive too (they may still be worth comparing).
- Quota exceeded on IndexedDB put → treat as storage failure
  (session-only note), never evict pinned records to make room.

## Testing

- `runsStore` (pure): seal appends + activates; ring evicts oldest
  unpinned at >8 unpinned; pinned exempt; togglePin/remove/importRecord
  (import arrives pinned).
- Storage adapter: fake-adapter round-trip; hydrate ordering; failure path
  sets the session-only flag without throwing.
- Export/import: round-trip equality; rejection of wrong `formatVersion`,
  missing meta, and traces failing `validateTrace`.
- `compareSelectors` (pure, fixture traces): alignment lengths; fork on
  crafted divergent fixtures; `forkCycle: null` for identical outputs and
  for different prompts; paired heads union/grouping.
- `CompareView` component tests with two fixture variants: header
  diff-highlight, fork highlight, ruler click → paired panels, missing-head
  fallback note, "run ended" side.
- Shelf component tests: chip render, active switch, pin toggle, compare
  arm/select/exit flow.
- E2E (sim-only) gains one smoke: generate twice → two chips → compare →
  fork visible → exit.
- Manual browser verification: real-mode runs archive across a reload;
  export → import round-trip; compare a T=0 vs T=1 pair on the same
  prompt.

## Out of scope (backlog)

- Grid side-by-side comparison and probability-delta view
- Per-cycle grid evolution; run renaming; >2-run comparison
- Cross-run head fetching (would require archiving full accumulators)
