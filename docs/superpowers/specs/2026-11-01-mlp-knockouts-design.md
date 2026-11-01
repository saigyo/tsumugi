# MLP knockouts — design (backlog #18)

Date: 2026-11-01. Status: draft for review.

## Intent

The MLP cards so far observe one run. The per-layer votes card shows each
layer's **direct effect** on a candidate's score: what its MLP edit writes
straight into the vector the Logits step reads. It cannot show **indirect
effects** — what a layer's edit changes in every later layer — and so it cannot
say whether the chosen token actually depends on that layer.

This feature answers one question, asked by the user about the cycle they are
looking at: **which layers really matter for the token chosen at this cycle?**
It does so by measurement, not heuristics: the app removes one layer's MLP at a
time and asks the model again (the *knockout sweep*). When a layer turns out to
matter, the user runs the full generation with that layer removed and sees both
runs side by side in the existing Compare view.

What a user should come away with: the gap between direct and total effect
(a layer can vote little and still be needed, or vote a lot and be replaced by
later layers), and that knockouts are a case study of one prompt, not a law
about a layer.

## Goals

1. A **Knockouts** card in the Layers detail that, on request, measures for the
   current cycle how the chosen token's probability changes when each of the 30
   MLPs is removed, and shows it next to the layer's direct vote.
2. **Role chips** naming the notable layers (decisive, preparatory, compensated,
   dampening), like the attention head chips, but derived from measurements.
3. A **"Run with MLP N off and compare"** action that generates the knockout run
   and opens the original (A) and the knockout (B) in Compare at the measured
   cycle.
4. **Fair comparisons**: every difference between the two runs comes from the
   knockout, never from sampling chance.

## Non-goals

- Knocking out attention blocks, single neurons, or several layers at once.
- Knocking out only at the last position (only "every position" is built).
- Mean ablation (replacing the output with an average); zero only.
- A sweep in simulated mode — simulated runs have no MLP.
- A vote-based *prediction* of important layers when no model is loaded.
- Running the sweep automatically.

## Anti-goals (failure even if it technically works)

- A comparison where part of the difference is sampling noise.
- Wording that presents a knockout result as an explanation of what a layer
  "means" in general, rather than what it did for this prompt and token.
- Any change to the outputs of ordinary runs (no knockout) compared with v6.
- A sweep that freezes the UI or can't be stopped.

## Decisions made with the user

- The card is its own section, **Knockouts**, between Votes and Inside one MLP,
  with an entry in the section bar (mockup option A,
  `.superpowers/brainstorm/…/sweep-placement.html`, numbers invented).
- The sweep targets **the token chosen at the current cycle** of the playback.
- The sweep runs **only when the user asks** ("Measure all 30 layers").
- A knockout removes the layer's MLP **at every position**.
- Fairness comes from a **seed per run**, not greedy decoding: the knockout run
  replays the original's random draws.
- The knockout run **copies the original's tokens up to the measured cycle** and
  generates freely from there with the same settings and draws.
- Results are **saved with the run** (survive reloads, travel with exports).
- No model loaded, or a simulated run: the card shows **a note with a "Load the
  real model" button**, nothing predicted.

## Design

### 1. Export v7: a per-layer MLP multiplier

The browser cannot edit the graph, so the export gains one input:

- `mlp_scale`, float32 `[num_layers]` (30). In every layer `l`, the down_proj
  output is multiplied by `mlp_scale[l]` **before** the residual Add, so
  everything downstream (the residual stream, `stream_parts`, edit ratios,
  readout) sees the scaled edit. All ones = the v6 model.
- Validation (in `tools/export`, with `ORT_DISABLE_ALL` as for every quantized
  check): (a) all ones reproduces v6's outputs (logits, readout, stream parts)
  within the existing parity tolerances; (b) for a few layers, `mlp_scale[l]=0`
  matches a PyTorch reference that zeroes that MLP's output (forward hook on the
  fp32 model) in greedy token and top-k logits within tolerance.
- Release as Hub tag `v7`, then bump `ATTN_MODEL_REVISION` in
  `src/engine/tokenizer.ts`. The `geometry/` and `neurons/` assets must be
  present under the v7 tag too, since the app pins everything to the tag.
- The worker passes `mlp_scale` on every forward call (all ones for ordinary
  runs). It detects support from the session's input names and reports it on
  `ready` (e.g. `knockout: boolean`); an older cached model leaves the feature
  off with the card's note — never-fail, like the other export features.
- **Risk for the builder to check first:** that transformers.js forwards a
  custom model input to the ONNX session. If it filters inputs, the worker
  needs a lower-level session call for this input; settle this in a spike before
  the export work.

### 2. Seeded sampling

- The worker picks a random 32-bit seed per run and records it as
  `seed?: number` on the `run-start` event (absent in older traces).
- The draw for cycle `n` is derived from the seed and `n` (e.g.
  `mulberry32(seedFromTokens([seed, n]))()` with the existing helpers in
  `src/engine/prng.ts`), not taken from one running sequence, so a copied
  prefix can't shift the draws out of step. `sampleIndex` is unchanged.
- Ordinary runs still differ from each other (fresh seed each), so Compare's
  existing fork view keeps working.
- `run-start` also records the model revision (`revision?: string`, e.g.
  `'v7'`; absent in older traces). Today runs record only `modelId`, which is
  the same for every export tag, so an archived run's model can't be told apart.

### 2a. Which runs are eligible

A sweep and a knockout run both re-run the model, so they are only fair on runs
recorded with the model that will re-run them. **Eligible runs are real runs
that record a seed and a revision of v7 or later** (in practice: every real run
recorded after this feature ships). For anything else the card shows a note
instead of the button:

- older real runs (no seed or no revision): "Recorded with an earlier version of
  the model — generate the run again to measure knockouts";
- a run that is itself a knockout run: "This run already has MLP N removed —
  open the original run to measure"; no sweep and no further knockout (two
  layers off is a non-goal);
- a run recorded on a different revision than the loaded model (future v8+):
  the same "generate again" note.

### 3. The sweep

Triggered by "Measure all 30 layers" for the current cycle `c` of the viewed
run. Works for live and archived real runs whenever a v7 model is loaded.

- Input: the run's prompt token ids (tokenize event) plus the tokens appended
  before cycle `c`, the run's top-k and temperature, the chosen token at `c`,
  and the run's seed if it has one.
- 31 forward passes over the whole sequence **without** the KV cache: one
  baseline (all ones) and one per layer with `mlp_scale[l] = 0`. The baseline is
  recomputed rather than taken from the trace, so cache-versus-no-cache
  numerical differences cancel.
- Per pass, from the last position's logits: the chosen token's probability
  under the run's own sampling distribution (same top-k, same temperature, so
  the baseline matches the Softmax card; greedy runs use T = 1 for display);
  the new top token and its probability; and whether the knockout run would
  pick a different token at this cycle (re-sample with the run's draw for
  cycle `c`; greedy runs: the top token changed).
- Results stream to the card as they arrive (progress fills the bars). Generate
  is disabled during a sweep and the Stop button cancels it. The existing
  512-token guard (`ATTN_MAX_SEQ`) applies; longer sequences get a note.
- Transport: a new worker request/response pair beside the trace. The protocol
  comment on `head-request` says the side channel must not grow without a new
  design; this document is that design, and the comment should be updated to
  name both uses.
- Expected time: about 5–10 s for a short prompt on WebGPU, extrapolated from
  generation speed. Each pass also computes the export's other outputs
  (attention, steps), so the builder should measure early; if it is much
  slower, consider whether those outputs can be skipped for sweep passes.

### 4. The Knockouts card

Layout per mockup A:

- Title "Which layers the chosen token depends on" with an ⓘ to a new glossary
  entry; a one-line caption on what the sweep does and how long it takes.
- Before measuring: the button and, for the current cycle, nothing else. After:
  "done for cycle N", and a "measure again" affordance is not needed (results
  are deterministic per model version).
- **Bars**, one per layer on the same 30 columns as the votes card: the change
  in the chosen token's probability in percentage points (red: drop, blue:
  rise). An outlined bar = the knockout run would pick a different token here.
- **Direct-vote row** under the bars: the layer's MLP vote for the chosen token
  from the votes card, same colour scale as there.
- **Role chips** for the notable layers, at most about five:
  - *decisive* — the knockout run would pick a different token at this cycle;
  - *preparatory* — small direct vote, large probability drop;
  - *compensated* — large direct vote, small probability change;
  - *dampening* — the probability rises without the layer.
  Thresholds are calibrated in a spike on several prompts (as the coreference
  head was), with a short note in `docs/research/`.
- Clicking a bar or chip selects the layer: a detail line ("MLP 15 — ' Paris'
  98% → 40%, top token unchanged") and the button **"Run with MLP 15 off and
  compare"**. Selecting a layer also sets the shared MLP layer (section bar
  stepper), so Inside one MLP and Step by step show the same layer.
- Results belong to their cycle: moving to another cycle shows that cycle's
  saved result, or the button.
- Notes instead of the card body: simulated run (no MLP); real run but no
  model loaded, or a model without `mlp_scale` → note + "Load the real model"
  button (`LoadRealButton`); sequence over the length guard.
- Ineligible runs (section 2a) get the matching note instead of the button.

### 5. The knockout run

- Same prompt, format (chat spans, system prompt), and parameters as the
  original; `mlp_scale[l] = 0` for the whole run; the original's seed.
- Cycles before `c` take the original's tokens instead of sampling (the
  knockout still applies, so their distributions may differ — Compare shows
  that). These cycles' `sample` events need a way to say the token was copied
  (e.g. `method: 'forced'`); the builder updates `validate.ts` and any view that
  switches on `method`.
- `run-start` records the knockout (e.g. `knockout?: { layer, cycle,
  sourceId }`), so the shelf chip and Compare labels can say "MLP 15 off".
- When it seals, the app opens Compare with the original as A and the knockout
  as B, at cycle `c`. Compare needs an initial-cycle input for this.
- Only eligible runs (section 2a) offer the action.
- **The other MLP cards in a knockout run.** The removed layer's MLP still
  computes; only its edit is discarded. In run B, for that layer: the votes and
  edit sizes show zero (they read the scaled edit); the step-by-step card's
  final "down" step shows the scaled (zero) edit, so the v7 export takes its
  `mlp_out` output **after** the multiplier; the neuron activations, gate, up
  and product stay as computed. Inside one MLP and Step by step show a note for
  that layer: "MLP N was removed in this run: its neurons fired as shown, but
  its edit was discarded." Other layers are unaffected.

### 6. Storage

- Sweep results are stored with the run record and written through to
  IndexedDB; JSON export includes them; imports without them are valid; import
  validation checks their shape when present.
- They are not playback events: the player and stage cards must ignore them.
  Whether they live in a new trace event or a separate field on `RunRecord` is
  the builder's choice under that constraint.

### 7. Glossary and docs

- New glossary entry (e.g. `knockout`): what removing a layer means, direct vs
  total effect with the four roles, and the caveats — zeroing an output the
  model never produces can cause damage of its own (mean ablation is the
  gentler variant used in research), and one prompt is a case study. "In this
  run" block when a sweep exists.
- README: a short "Knocking out a layer" section with a screenshot scene;
  `docs/BACKLOG.md` updated.

## Calls made by Claude (open to correction)

- Zero ablation, not mean ablation.
- Attention knockouts out of scope.
- The bar measure is the chosen token's probability under the run's own
  sampling distribution (top-k, temperature), so it matches the Softmax card.
- The outlined "decisive" bar uses the same seeded draw as the knockout run, so
  it predicts exactly where Compare will fork.
- Selecting a layer in the Knockouts card also sets the shared MLP layer.
- Stop cancels a running sweep.
- Only runs recorded with a seed on v7 or later are eligible. This replaces the
  earlier idea of allowing old greedy runs: their model can't be identified,
  and a v3–v5 original against a v7 knockout would mix a model change into the
  comparison.
- Runs record the model revision from now on.
- Knockout runs get no sweep and no further knockout.
- In a knockout run, the removed layer's neuron and step cards keep the
  computed activations, show the zeroed edit, and carry a note; the export
  takes `mlp_out` after the multiplier.

## Left to the builder

- Exact message names, store shape and component split.
- Role-chip thresholds (from the calibration spike).
- How the sweep results are stored (constraint in section 6).
- Whether sweep passes can skip the export's heavy outputs.
- Visual polish within the mockup's layout; charts fill the card width.
