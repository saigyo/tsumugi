# MLP Knockouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Knockouts card in the Layers detail that measures, on request, how the token chosen at the current cycle depends on each layer's MLP, and a one-click knockout run opened in Compare.

**Architecture:** Export v7 adds a `mlp_scale [30]` graph input that multiplies each layer's MLP edit before the residual Add. The worker passes it on every forward call, samples from a per-run seed, runs the 31-pass sweep on a side channel, and can generate a knockout run that copies the original's tokens up to a cycle. Sweep results live on the run record (not in the trace); the card reads them, and the App orchestrates sweeps and knockout runs through a small zustand request store, like `realModelRequest.ts`.

**Tech Stack:** Python (onnx, onnxruntime, torch, pytest via `uv`) in `tools/export`; TypeScript, React 19, zustand, transformers.js 4, vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-01-mlp-knockouts-design.md`

## Global Constraints

- Model input name `mlp_scale`, float32, shape `[num_layers]` (30); all ones = the v6 model.
- Hub tag `v7`; `ATTN_MODEL_REVISION = 'v7'` in `src/engine/tokenizer.ts`. Tags never move; publish refuses an existing tag.
- Quantized checks run with `ORT_DISABLE_ALL` (browser semantics), as all existing ones do.
- Eligible runs: `mode === 'real'`, `seed` and `revision` recorded, revision ≥ v7 and equal to the loaded model's, not itself a knockout run, sequence ≤ 512 tokens (`ATTN_MAX_SEQ`).
- Sweep = 31 forward passes without the KV cache (1 baseline + 1 per layer); probabilities under the run's own top-k and temperature, T = 0 shown at T = 1.
- Copy (exact): title "Which layers the chosen token depends on"; button "Measure all 30 layers" (number from the run); action "Run with MLP N off and compare"; shelf/Compare label "MLP N off".
- Note texts (exact):
  - sim: `Real model only: simulated runs have no MLP to measure.`
  - no model / model without `mlp_scale`: `Load the real model to measure knockouts.` + `LoadRealButton`
  - old run or other revision: `Recorded with an earlier version of the model — generate the run again to measure knockouts.`
  - knockout run: `This run already has MLP N removed — open the original run to measure.`
  - too long: `This run is longer than 512 tokens, too long to measure.`
  - removed layer (Inside one MLP, Step by step): `MLP N was removed in this run: its neurons fired as shown, but its edit was discarded.`
- Never-fail: a model without `mlp_scale` disables the feature quietly (card note), ordinary runs are unaffected.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Rebase + fast-forward only, no merge commits.
- Resolved before planning: transformers.js `decoder_forward` passes any input listed in `session.inputNames` (`pick(new_model_inputs, session.inputNames)` in `models/modeling_utils.js`), so `mlp_scale` reaches the session through the normal `model({...})` call; and a v7 session *requires* it, so the worker must always send it when the input exists.

## Review Focus

1. A sweep or knockout started while another run or sweep is in flight — Generate, Measure and the knockout action must be mutually exclusive, and Stop must cancel whichever is running (Task 9 test).
2. Leaving the run (activate another run, Generate, import) mid-sweep — the partial result must be discarded, never saved onto the wrong record (Task 9 test).
3. A chosen token outside the knockout's top-k — its probability is 0, the bar shows the full drop, `flips` is true (Task 5 test).
4. A knockout at cycle 0 (nothing to copy) and at the run's last cycle — both must work; Compare opens at that cycle (Task 4 and Task 9 tests).
5. An imported run file with malformed `sweeps` — import keeps the run and drops the sweeps with a note, like damaged neuron data (Task 6 test).

---

### Task 1: Export — `mlp_scale` input

**Files:**
- Modify: `tools/export/src/tsumugi_export/postprocess.py`
- Test: `tools/export/tests/test_postprocess_knockout.py` (reuse `layered_graph()` from `test_postprocess_mlp.py` by import)

**Interfaces:**
- Produces: `MLP_SCALE_INPUT = "mlp_scale"`; `add_mlp_scale(model: onnx.ModelProto) -> onnx.ModelProto`; `postprocess_graph` calls it after `expose_mlp_steps` and before `expose_inputs_embeds`.

- [ ] **Step 1: Write the failing tests**

```python
def test_mlp_scale_is_a_float_input_of_layer_length():
    m = postprocess_graph(layered_graph())   # postprocess_graph calls add_mlp_scale
    inp = next(i for i in m.graph.input if i.name == MLP_SCALE_INPUT)
    assert inp.type.tensor_type.elem_type == TensorProto.FLOAT
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [LAYERS]

def test_ones_reproduce_the_unscaled_graph():          # every output, atol 1e-6
def test_zero_removes_one_layers_edit():
    # scale [1, 0]: mlp_out[:, 1] == 0, stream_parts row 1 + LAYERS + 1 == 0,
    # edit_ratios[..., 1, 1] == 0, logits == a hand-computed forward without layer 1's MLP (atol 1e-5)
def test_mlp_activations_are_not_scaled():              # scale [0, 0]: mlp_activations equal the ones run
def test_only_the_scale_mul_reads_the_raw_edit():
    # for each layer, the down_proj MatMul output has exactly one consumer: the tsumugi Mul
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_postprocess_knockout.py -q`
Expected: FAIL (ImportError: `add_mlp_scale`).

- [ ] **Step 3: Implement `add_mlp_scale`**

Per layer `l` with MLP edit `m` (from `find_blocks`): `Gather(mlp_scale, const l)` → scalar `tsumugi_mlp_scale{l}`, `Mul(m, scalar)` → `tsumugi_mlp{l}_scaled`, then rewire every node input equal to `m` (except the new Mul) to the scaled name — this covers the residual Add and the `edit_ratios` / `stream_parts` / `mlp_out` taps added earlier. Add the graph input. Update the module docstring's list of outputs.

- [ ] **Step 4: Run the export test suite**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass (existing postprocess tests unchanged — ones behave like v6).

- [ ] **Step 5: Commit**

```bash
git add tools/export/src/tsumugi_export/postprocess.py tools/export/tests/test_postprocess_knockout.py
git commit -m "feat(export): mlp_scale input multiplies each layer's MLP edit (v7)"
```

### Task 2: Export — validation and the v7 release

**Files:**
- Modify: `tools/export/src/tsumugi_export/validate.py`, `tools/export/README.md` (release steps: `--revision v7`, `--tag v7`)
- Test: `tools/export/tests/test_validate_knockout.py`

**Interfaces:**
- Consumes: `MLP_SCALE_INPUT` (Task 1).
- Produces: `_feeds` adds `mlp_scale = ones(layers)` whenever the session has that input; `check_knockout_parity(ours: dict[int, np.ndarray], ref: dict[int, np.ndarray], k: int, logit_tol: float, tie_margin: float) -> dict` comparing last-position logits per knocked-out layer; report key `knockout-parity` (gating, like the others).

- [ ] **Step 1: Write the failing tests** — `check_knockout_parity` passes for identical logits; fails when the argmax differs beyond `TIE_MARGIN` (0.1); passes on a near-tie within the margin; fails when a top-k logit is off by more than `logit_tol`.
- [ ] **Step 2: Run** `uv run pytest tests/test_validate_knockout.py -q` — Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** the check, the `_feeds` change, and the reference: PyTorch fp32 stock model with a forward hook on `model.layers[l].mlp` returning zeros, for layers `[0, 13, 29]` on the validation probe prompt; ORT side feeds `mlp_scale` with that layer zeroed. Tolerances: fp32 export `logit_tol = 1e-3`; quantized model tie-aware argmax plus `logit_tol = 0.5` (gross-error gate, like the other q8 checks).
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: all pass.
- [ ] **Step 5: Build and validate v7 locally** (README steps 1–7 with `--revision v7`). Expected: `validation-report.json` shows every check PASS, including `knockout-parity`, and the existing greedy/readout/votes/steps checks unchanged (they now run with `mlp_scale` = ones).
- [ ] **Step 6: Commit**, then **hand the publish to the user**: `uv run python -m tsumugi_export publish --model-dir out/model --tag v7`. Confirm `geometry/` and `neurons/` (manifest revision `v7`) are on the tag before Task 3 bumps the revision.

```bash
git add tools/export
git commit -m "feat(export): knockout parity check against a zeroed-MLP PyTorch reference"
```

### Task 3: Trace types, seeded sampling, run-start seed and revision

**Files:**
- Modify: `src/trace/types.ts`, `src/engine/prng.ts`, `src/engine/transformers/worker.ts`, `src/trace/validate.ts`
- Test: `src/engine/prng.test.ts`, `src/trace/validate.test.ts`

**Interfaces:**
- Produces (types.ts):
  - `export interface KnockoutOrigin { layer: number; cycle: number; sourceId: string }`
  - run-start gains `seed?: number; revision?: string; knockout?: KnockoutOrigin`
  - sample `method: 'greedy' | 'top-k' | 'forced'`
  - `export interface KnockoutLayer { layer: number; prob: number; top: TokenInfo & { prob: number }; flips: boolean }`
  - `export interface KnockoutSweep { cycle: number; revision: string; chosen: TokenInfo; baseline: number; layers: KnockoutLayer[] }`
- Produces (prng.ts): `cycleDraw(seed: number, cycle: number): number` = `mulberry32(seedFromTokens([seed, cycle]))()`.

- [ ] **Step 1: Write the failing tests**

```ts
it('cycleDraw is deterministic per seed and cycle and in [0, 1)', () => {
  expect(cycleDraw(42, 3)).toBe(cycleDraw(42, 3))
  expect(cycleDraw(42, 3)).not.toBe(cycleDraw(42, 4))
  expect(cycleDraw(42, 3)).not.toBe(cycleDraw(43, 3))
  for (let c = 0; c < 100; c++) { const d = cycleDraw(7, c); expect(d).toBeGreaterThanOrEqual(0); expect(d).toBeLessThan(1) }
})
it('validateTrace accepts a forced sample and rejects an unknown method')
```

- [ ] **Step 2: Run** `npx vitest run src/engine/prng.test.ts src/trace/validate.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.** `cycleDraw`; validator check on `sample.method`. Worker: draw a seed per run with `crypto.getRandomValues(new Uint32Array(1))[0]`, put `seed` and (when the attn model loaded) `revision: ATTN_MODEL_REVISION` on run-start, sample with `sampleIndex(probs, () => cycleDraw(seed, cycle))`.
- [ ] **Step 4: Run** `npm test` — Expected: all pass.
- [ ] **Step 5: Commit** `feat(engine): seeded per-cycle sampling; runs record seed and model revision`.

### Task 4: Worker — `mlp_scale` on every call, knockout runs

**Files:**
- Create: `src/engine/knockout.ts` (pure helpers shared by worker and views)
- Modify: `src/engine/transformers/worker.ts`, `protocol.ts`, `TransformersEngine.ts`, `src/engine/types.ts`, `src/engine/tokenizer.ts` (`ATTN_MODEL_REVISION = 'v7'` — in this task, because a v7 session requires `mlp_scale`; only after the user confirms the v7 publish)
- Test: `src/engine/knockout.test.ts`, `src/engine/transformers/TransformersEngine.test.ts` (fake worker, as existing engine tests do; create if absent)

**Interfaces:**
- Produces (knockout.ts): `mlpScale(layers: number, off: number | null): Float32Array`; `forcedToken(forced: number[], cycle: number): number | null`.
- Produces (types.ts): `export interface RunOptions { knockout?: { origin: KnockoutOrigin; forced: number[]; seed: number } }`; `PipelineEngine.run(prompt, params, emit, format?, options?: RunOptions)`.
- Produces (protocol): `run` request gains `knockout?: RunOptions['knockout']`; `ready` gains `knockout: boolean`; `TransformersEngine.knockout: boolean | null`, `TransformersEngine.revision: string | null`.

- [ ] **Step 1: Failing tests** — `mlpScale(30, null)` all ones; `mlpScale(30, 15)` zero only at 15; `forcedToken([5, 9], 0) === 5`, `forcedToken([5, 9], 2) === null`; engine posts `knockout` in the run request and exposes `knockout`/`revision` from `ready`.
- [ ] **Step 2: Run** `npx vitest run src/engine` — Expected: FAIL.
- [ ] **Step 3: Implement.** Worker: `hasScale = model.sessions.model.inputNames.includes('mlp_scale')`, sent on `ready`; every forward call adds `mlp_scale: new Tensor('float32', mlpScale(numLayers, knockout?.origin.layer ?? null), [numLayers])` when `hasScale`. Knockout run: seed from the request, `knockout: origin` on run-start, cycles with a forced token emit `sample` with `method: 'forced'` and that token. Simulated engine ignores `options`.
- [ ] **Step 4: Run** `npm test` — Expected: pass.
- [ ] **Step 5: Commit** `feat(engine): knockout runs — mlp_scale per call, copied prefix, original seed`.

### Task 5: Worker — the sweep side channel

**Files:**
- Modify: `src/engine/knockout.ts`, `worker.ts`, `protocol.ts` (also update the `head-request` comment to name both side-channel uses), `TransformersEngine.ts`
- Test: `src/engine/knockout.test.ts`, `TransformersEngine.test.ts`

**Interfaces:**
- Produces (knockout.ts): `sweepEntry(lastLogits: Float32Array, params: GenParams, chosenId: number, draw: number): { prob: number; topId: number; topProb: number; flips: boolean }` — top-k at `params.topK`, softmax at `params.temperature || 1`; `prob` = chosen's share (0 if outside top-k); `flips` = T = 0 ? top id ≠ chosen : `sampleIndex(probs, () => draw)` picks ≠ chosen.
- Produces (protocol): request `{ type: 'sweep-request'; sweepId: number; ids: number[]; params: GenParams; chosenId: number; seed: number; cycle: number }`; responses `{ type: 'sweep-baseline'; sweepId; prob: number }`, `{ type: 'sweep-layer'; sweepId; entry: KnockoutLayer }`, `{ type: 'sweep-done'; sweepId; aborted: boolean }`. `abort` also stops a sweep.
- Produces (engine): `sweep(req: Omit<SweepRequest, 'type' | 'sweepId'>, onBaseline: (p: number) => void, onLayer: (e: KnockoutLayer) => void): { abort(): void; done: Promise<boolean /* completed */> }`.

- [ ] **Step 1: Failing tests** — `sweepEntry` on hand-made logits: chosen at top → `prob` matches softmax, `flips` false; chosen pushed out of top-k → `prob === 0`, `flips === true`; T = 0 uses T = 1 for `prob`; with T = 0.8 and a fixed draw, `flips` agrees with `sampleIndex`. Engine: forwards baseline/layer messages for its own `sweepId` only, resolves `done` with `false` on `aborted: true`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.** Worker loop: pass 0 all ones (baseline), then layers 0..29 with one zero; no `past_key_values`; each pass one `model({ input_ids, attention_mask, mlp_scale })` over `ids`; check `aborted` between passes; dispose outputs per pass. `ids` = prompt tokens + appended tokens before `cycle` (built by the caller).
- [ ] **Step 4: Run** `npm test` — Expected: pass.
- [ ] **Step 5: Commit** `feat(engine): knockout sweep side channel`.

### Task 6: Storage — sweeps on the record, knockout label

**Files:**
- Modify: `src/app/runsStore.ts`, `src/app/runArchive.ts`, `src/app/runFiles.ts`, `src/trace/validate.ts`, `src/app/RunShelf.tsx`, `src/App.tsx` (seal meta carries `knockout`)
- Test: `src/app/runsStore.test.ts`, `src/app/runFiles.test.ts`, `src/app/RunShelf.test.tsx`

**Interfaces:**
- Produces: `RunRecord.sweeps?: KnockoutSweep[]` (one per cycle); `RunMeta.knockout?: KnockoutOrigin`; store action `addSweep(id: string, sweep: KnockoutSweep): void` (replaces a sweep of the same cycle); `archiveAddSweep(id, sweep)` mirrors to storage; `sweepProblems(s: unknown, layers: number): string[]` in validate.ts; run files write `sweeps` beside `events` (format version stays 1).

- [ ] **Step 1: Failing tests** — `addSweep` stores, replaces same cycle, ignores unknown id; export → parse round-trips `sweeps` and `meta.knockout`; a file with a malformed sweep imports with the sweep dropped and note `Knockout data of 1 cycle was damaged and left out.`; shelf chip label of a knockout run ends with `· MLP 15 off`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test` — Expected: pass.
- [ ] **Step 5: Commit** `feat(archive): sweeps stored with the run; knockout runs labelled`.

### Task 7: Eligibility and roles (pure)

**Files:**
- Create: `src/viz/knockoutEligibility.ts`, `src/viz/knockoutRoles.ts`
- Test: matching `.test.ts` files

**Interfaces:**
- Produces: `type Ineligible = 'sim' | 'knockout-run' | 'old-run' | 'no-model' | 'too-long'`; `knockoutEligibility(run: Extract<TraceEvent, { type: 'run-start' }>, seqLen: number, model: { knockout: boolean; revision: string } | null): { ok: true } | { ok: false; reason: Ineligible }` — check order sim → knockout-run → old-run (no seed, no revision, or revision < v7) → no-model (null or `knockout` false) → old-run (revision ≠ model's) → too-long (> 512).
- Produces: `type KnockoutRole = 'decisive' | 'preparatory' | 'compensated' | 'dampening'`; `knockoutRoles(sweep: KnockoutSweep, votes: number[] /* MLP vote for the chosen token per layer */, t?: RoleThresholds): Array<{ layer: number; role: KnockoutRole }>` — at most `MAX_ROLE_CHIPS = 5`, decisive first then by |Δp|; `ROLE_THRESHOLDS` exported with provisional values (`drop: 0.15`, `rise: 0.05`, `smallVote: 0.25`, `largeVote: 1.0`, vote thresholds as fractions of the largest |vote|), replaced in Task 11.

- [ ] **Step 1: Failing tests** — one per eligibility reason plus ok; roles: `flips` → decisive; small vote + drop 0.4 → preparatory; large vote + |Δp| 0.01 → compensated; rise 0.1 → dampening; at most 5, decisive first.
- [ ] **Step 2–4:** run (FAIL), implement, run `npm test` (PASS).
- [ ] **Step 5: Commit** `feat(viz): knockout eligibility and layer roles`.

### Task 8: The Knockouts card

**Files:**
- Create: `src/viz/KnockoutCard.tsx`, `src/viz/knockoutRequest.ts` (zustand store, pattern of `realModelRequest.ts`)
- Modify: `src/viz/layersSections.ts` (`{ id: 'layers-knockouts', label: 'Knockouts' }` after Votes), `src/viz/details/LayersDetail.tsx`, `src/viz/DetailPanel.tsx`, `src/index.css`, `src/test/setup.ts` (reset the store)
- Test: `src/viz/KnockoutCard.test.tsx`, `src/viz/LayersNav.test.tsx`

**Interfaces:**
- Produces (knockoutRequest.ts): `useKnockoutRequest` with `{ model: { knockout: boolean; revision: string } | null; running: { recordId: string; cycle: number; baseline: number | null; layers: KnockoutLayer[] } | null; measure: ((cycle: number) => void) | null; runKnockout: ((layer: number, cycle: number) => void) | null }`; `resetKnockoutRequest()`.
- Consumes: `knockoutEligibility`, `knockoutRoles` (Task 7); the active record's `sweeps` from `useRunsStore`; `cycleVotes` for the direct-vote row; `useMlpLayer().select` for the shared layer.
- Props: `KnockoutCard({ run, cycle, seqLen, chosen, votes, sweep, running, model, noteAction })`.

- [ ] **Step 1: Failing tests** — each ineligible reason shows its exact note (Global Constraints); eligible without sweep shows `Measure all 30 layers` and clicking calls `measure(cycle)`; a running sweep shows bars as entries arrive and disables the button; a saved sweep shows 30 bars with `data-drop` = baseline − prob, the outlined bar for `flips`, the direct-vote row, role chips; clicking a bar shows the detail line `MLP 15 — “ Paris” 98% → 40%, top token unchanged` and the action `Run with MLP 15 off and compare`, and sets the shared MLP layer; the section bar lists Knockouts after Votes.
- [ ] **Step 2–4:** run (FAIL), implement per mockup A (bars fill the card width; red drop, blue rise; vote row on the votes card's colour scale), run `npm test` (PASS).
- [ ] **Step 5: Commit** `feat(viz): Knockouts card`.

### Task 9: App orchestration — measure, knockout run, Compare at the cycle

**Files:**
- Modify: `src/App.tsx`, `src/app/PromptBar.tsx` (prop `activity?: string`, default `'Generating…'`), `src/viz/compare/CompareView.tsx` (prop `initialCycle?: number`; App keys it by `aId-bId`)
- Test: `src/App.knockout.test.tsx` (fake engine, pattern of `App.modeGuard.test.tsx`), `src/viz/compare/CompareView.test.tsx`

**Interfaces:**
- Consumes: `TransformersEngine.sweep`, `.knockout`, `.revision` (Tasks 4–5); `archiveAddSweep` (Task 6); `useKnockoutRequest` (Task 8).
- Behaviour: App publishes `model` after `prepare`; `measure(cycle)` builds `ids` (prompt + appended before cycle), sets `generating` with activity `Measuring…`, fills `running`, saves via `archiveAddSweep` on completion, discards on abort or when the active record changes; Stop aborts the sweep. `runKnockout(layer, cycle)` starts a run with the original's prompt, format, params, seed, `forced` = appended tokens before `cycle`, `origin = { layer, cycle, sourceId }`; on seal opens Compare `{ aId: sourceId, bId: newId }` at `initialCycle = cycle`.

- [ ] **Step 1: Failing tests** — Measure disabled while generating and Generate disabled while measuring; Stop during a sweep saves nothing; activating another run mid-sweep saves nothing; a completed sweep lands on the right record; knockout at cycle 0 sends `forced: []`; after the knockout run seals, Compare shows A = original, B = knockout, cycle selected = the measured cycle.
- [ ] **Step 2–4:** run (FAIL), implement, run `npm test` (PASS).
- [ ] **Step 5: Commit** `feat(app): measure knockouts and open the knockout run in Compare`.

### Task 10: The removed layer in a knockout run

**Files:**
- Modify: `src/viz/NeuronCard.tsx`, `src/viz/MlpStepsCard.tsx`, `src/viz/details/LayersDetail.tsx` (pass `run.knockout?.layer`)
- Test: `src/viz/NeuronCard.test.tsx`, `src/viz/MlpStepsCard.test.tsx`

- [ ] **Step 1: Failing tests** — in a knockout run with layer 15 removed, both cards at layer 15 show `MLP 15 was removed in this run: its neurons fired as shown, but its edit was discarded.`; at layer 14 no note; in an ordinary run no note.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS). The neuron card's "≈" marker may appear on the removed layer (neuron votes from activations vs a zero vote); the note explains it, no further change.
- [ ] **Step 5: Commit** `feat(viz): note the removed layer in knockout runs`.

### Task 11: Calibration spike, glossary, docs

**Files:**
- Create: `src/glossary/entries/knockout.tsx`, `docs/research/<spike date>-knockout-roles-spike.md` (dated the day the spike runs)
- Modify: `src/glossary/types.ts` (id `knockout`), `src/glossary/registry.ts`, `src/viz/knockoutRoles.ts` (final thresholds), `README.md` ("Knocking out a layer" section), `scripts/capture-screenshots.mjs` (scene `knockouts`), `docs/BACKLOG.md` (#18 done)
- Test: `src/glossary/entries/knockout.test.tsx` (entry renders; links from the card's ⓘ; "In this run" shows the saved sweep's top layers)

- [ ] **Step 1: Spike (real model, v7, headless Playwright on `.screenshots-profile`).** Sweep five prompts (France/Paris chat, the farmer/horse prompt, the hungry-cat example, an induction example, one free prompt) at two cycles each; export the runs; tabulate Δp and votes; pick thresholds so each prompt gets 1–5 chips and none look arbitrary; record timings per sweep. Write the research note.
- [ ] **Step 2:** Update `ROLE_THRESHOLDS` and the Task 7 role tests to the calibrated values; run `npm test` (PASS).
- [ ] **Step 3:** Glossary entry: removing a layer; direct vs total effect with the four roles; caveats (zero ablation vs mean ablation; one prompt is a case study). Test first, then entry.
- [ ] **Step 4:** README section + screenshot scene; `npm run screenshots knockouts` (kill any stale :5199 server first).
- [ ] **Step 5: Commit** `docs: knockout roles calibrated, glossary entry, README`.

### Task 12: Whole-branch verification

- [ ] **Step 1:** `npm test`, `npx oxlint`, `npm run build`, `npm run e2e` — all pass (e2e is sim-only: the card shows the sim note).
- [ ] **Step 2:** Real-mode browser check (persistent profile, own dev-server port): generate, measure at two cycles, reload (sweeps persist), export/import (sweeps travel), run a knockout from a decisive layer (Compare forks at the measured cycle), Stop mid-sweep (nothing saved), an old v6 run shows the "generate again" note.
- [ ] **Step 3:** Update memory and backlog; branch integrated by rebase + fast-forward.
