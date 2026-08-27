# LLM Pipeline Visualizer — Design Spec

**Date:** 2026-08-26
**Status:** Approved design, pre-implementation

## Purpose

A single-page educational web app that visualizes how data flows through
an LLM's generation pipeline. The user types a prompt and watches — live
or step-by-step — how it becomes tokens, embeddings, layer activations,
logits, probabilities, and finally sampled tokens that loop back into the
input. Two modes: a fully **simulated** mode with realistic fake data
(instant, no downloads beyond a tokenizer) and a **real** mode that runs
an actual small model in the browser via transformers.js.

**Audience:** developers and CS students. Comfortable with code and
vectors; show tokenization details, logits, softmax, and sampling
parameters, but not full matrix math.

**Scope:** the core autoregressive generation loop —
tokenize → embed → transformer layers → logits → softmax → sample →
append → repeat. No chat templating, no attention-head deep-dive (both
are possible later extensions).

## Architecture

Static SPA — Vite + React + TypeScript, deployable to any static host,
no backend. The central design is **trace-based** (Approach A of the
brainstorm):

Both engines emit a **normalized trace**: an append-only array of typed
step events. The visualization renders as a pure function of
`(trace, cursor)`. The player only moves the cursor.

```
User input ──▶ Engine ──emits──▶ Trace Store ◀──cursor── Player
                                      │
                                      ▼
                              Visualization (pure render of trace + cursor)
```

Why trace-based won over alternatives:

- Scrubbing backward/forward is free — re-render at a different cursor,
  no state to unwind.
- Both modes are identical from the UI's perspective; the real engine
  appends asynchronously, the simulated one near-instantly.
- Trivially testable (feed a fixed trace, assert the render) and
  debuggable (dump trace as JSON).
- Rejected: live event streaming (step-back requires rebuilding trace
  machinery implicitly) and pull-based stepping (a real forward pass
  cannot be paused mid-stage; you'd buffer into a trace anyway).

### Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `src/trace/` | Trace event schema (discriminated union) + append-only store (append, subscribe, clear) | nothing |
| `src/engine/` | `PipelineEngine` interface; `SimulatedEngine` (main thread); `TransformersEngine` (Web Worker client) | `trace` types |
| `src/player/` | Playback state: cursor, play/pause, speed. Pure reducer (pacing + park model) | `trace` types |
| `src/viz/` | One component per stage + pipeline layout; renders `(trace, cursor)` | `trace`, `player` state |
| `src/app/` | Shell: prompt input, mode toggle, gen params, player controls, model-loading UI | all of the above |

**Invariant:** the visualization only reads `(trace, cursor)`; engines
only append to the trace; the player only moves the cursor. No
cross-boundary communication.

State lives in a small Zustand store (trace + player slices). The player
reducer core is a plain function, unit-testable without React.

## Trace event schema

One generation run produces one trace:

```ts
type TraceEvent =
  | { type: 'run-start'; prompt: string; mode: 'sim' | 'real'; modelId: string;
      params: { temperature: number; topK: number; maxNewTokens: number } }
  | { type: 'tokenize'; tokens: Array<{ id: number; text: string }> }
  // ── per generated token (one "cycle"): ──
  | { type: 'embed'; cycle: number; seqLen: number; dims: number;
      preview: number[][] }                 // last few tokens × ~16 downsampled dims
  | { type: 'layer'; cycle: number; index: number; total: number;
      activationNorm?: number }             // real value if obtainable, else synthetic
  | { type: 'logits'; cycle: number; topK: Array<{ id: number; text: string; logit: number }> }
  | { type: 'softmax'; cycle: number; temperature: number;
      topK: Array<{ id: number; text: string; prob: number }> }
  | { type: 'sample'; cycle: number; chosen: { id: number; text: string };
      method: 'greedy' | 'top-k' }
  | { type: 'append'; cycle: number; token: { id: number; text: string } }
  | { type: 'run-end'; reason: 'eos' | 'max-tokens' | 'aborted' | 'error'; message?: string }
```

Deliberate size choices:

- Only **top-k (≈10)** candidates stored at logits/softmax — never the
  full vocabulary distribution. A 50-token run stays at a few hundred KB.
- Embeddings stored as **downsampled previews** (the point is "text
  became numbers", not the numbers).
- Cursor = index into the array. The repeating
  `embed → layer×N → logits → softmax → sample → append` rhythm *is* the
  autoregressive loop being taught.

## Engines

Shared interface:

```ts
interface PipelineEngine {
  prepare(onProgress: (p: ProgressInfo) => void): Promise<void>;  // load tokenizer/model
  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void): RunHandle;
}
interface RunHandle { abort(): void; done: Promise<void> }
```

### SimulatedEngine (main thread)

- `prepare()` loads **only the tokenizer** via
  `AutoTokenizer.from_pretrained()` (`@huggingface/transformers`) — the
  same tokenizer as the real model, so token splits and IDs are
  identical across modes. ~1–2 MB, near-instant.
- Internals are procedural and **deterministic**: seeded PRNG (seed
  derived from token IDs; same prompt → identical trace). Generates
  embedding previews, plausibly drifting per-layer activation norms, and
  logit values with realistic shape (one or two dominant candidates,
  long tail).
- Top-k candidate *identity*: small built-in word-frequency list plus
  light heuristics (continuations fitting the last token, punctuation
  after long clauses, EOS probability rising with length). Output reads
  as plausible filler, not coherent prose — accepted compromise; it
  makes clear the mode fakes the *intelligence* while being honest about
  the *mechanics*.
- Emits the full event cycle through the same emit path as the real
  engine; traces are structurally indistinguishable.

### TransformersEngine (Web Worker)

- Worker owns transformers.js and the model; the main-thread class
  forwards `postMessage` traffic. Worker → main messages are literally
  `TraceEvent` objects (plain serializable data).
- Default model: a small decoder such as
  `HuggingFaceTB/SmolLM2-135M-Instruct` (ONNX, ~100–150 MB quantized).
  WebGPU when available, WASM fallback. Cached by the browser after
  first download.
- **No high-level `pipeline()` helper** — it hides everything we want to
  show. Manual loop instead: forward pass with KV-cache → last
  position's logits → extract top-k → apply temperature + softmax in
  JS → sample in JS → append, repeat. `logits`, `softmax`, and `sample`
  events carry *real* numbers; UI sampling params genuinely change the
  outcome.
- Accepted compromise: transformers.js does not reliably expose
  per-layer hidden states/attention for ONNX models, so in real mode
  `layer` events carry only index/total and render as a schematic
  animation labeled "illustrative". Logits onward are fully real.
  Hidden-state norms can be added later if a model config exposes them
  cheaply; the design does not depend on it.
- Abort: flag checked between cycles (sim) / message stopping the worker
  loop between forward passes (real).

## Player

Pure reducer, no React/DOM:

```ts
interface PlayerState {
  cursor: number;          // index into trace; -1 = before start
  status: 'idle' | 'playing' | 'paused';
  speed: number;           // 0.5×, 1×, 2×, 4×
}
```

Actions: `play`, `pause`, `stepForward`, `stepBack`, `seek(index)`,
`setSpeed`, `traceGrew(newLength)`, `goLive`, `reset`.

Behaviors (the **pacing + park** model):

- **Playing** advances the cursor on a timer derived from `speed`, with
  per-event pacing multipliers (a `sample` lingers; one of twelve
  `layer` events is quick). The pacing table lives in the player, not
  components.
- "Real time" and "step-by-step" are unified by paced playback plus
  **parking**: a playing cursor that reaches the frontier of a
  still-generating trace parks there and resumes automatically as
  events stream in — during live generation, playback naturally chases
  the frontier. Pausing, stepping back, or scrubbing lets the user
  inspect history while generation continues appending; the "Live ⏺"
  button (`goLive`) seeks to the frontier and plays. "Live" is a
  derived condition (cursor at frontier while playing), not stored
  state.
- Reaching the frontier of a *finished* trace (its `run-end` event)
  pauses playback; the detail panel then shows a run summary (finish
  reason, token counts, sampling params).
- **Scrubber**: range input over `trace.length`, tick marks at each
  `append` event to jump between token cycles. Step controls work by
  *event* or by *cycle* (modifier or dedicated buttons).

## Visualization

Layout (three bands + controls):

```
┌─────────────────────────────────────────────────────────────┐
│  Prompt input  [Simulated ⇄ Real]  temp/top-k  [Generate]   │
├─────────────────────────────────────────────────────────────┤
│  TOKEN STREAM: [The][ cat][ sat][ on] … [ mat]▊             │
├─────────────────────────────────────────────────────────────┤
│  [Tokenizer]→[Embeddings]→[Layers ×N]→[Logits]→[Sampler]──┐ │
│        (loop arrow from Sampler back to Token stream) ◀───┘ │
├─────────────────────────────────────────────────────────────┤
│  DETAIL PANEL: expanded view of the active stage            │
├─────────────────────────────────────────────────────────────┤
│  ◀◀  ◀  ▶/⏸  ▶▶   ────●──────────  1×  [⏺ Live]            │
└─────────────────────────────────────────────────────────────┘
```

- **Pipeline band**: five stage cards with arrows and a drawn **loop
  arrow** back to the token stream (the loop is the single most
  important idea — drawn, not implied). The cursor's event determines
  the "hot" card (`tokenize` → Tokenizer, `embed` → Embeddings,
  `layer` → Layers, `logits` and `softmax` → Logits, `sample` and
  `append` → Sampler); an animated packet crosses connectors on stage
  transitions. Cards show micro-summaries (token count, `N layers`, top
  candidate).
- **Detail panel** per stage:
  - *Tokenizer*: prompt with colored token-boundary underlays; chips
    show text + ID (subword splits made visceral).
  - *Embeddings*: chips morph into heat-strips of the downsampled
    preview, captioned with real dimensionality.
  - *Layers*: stack of N blocks lighting in sequence; sim mode annotates
    activation norms; real mode labeled "schematic".
  - *Logits → Softmax*: horizontal bar chart of top-k candidates,
    animated rescale when temperature/softmax applies — shows exactly
    what temperature does.
  - *Sampler*: bars become a roulette strip; marker drops on the chosen
    token; chip flies up to the token stream.
- **Cross-mode identity**: both modes render through identical
  components; visible differences are a mode badge and the "schematic"
  layer label. The prompt and sampling params persist across a mode
  switch, so comparing modes is one click of Generate away (an
  automatic re-run on toggle was considered and rejected: it would
  surprise-start generation and, on first real-mode use, a ~120 MB
  download).
- SVG + CSS transitions only (no canvas, no D3 — data volumes are
  tiny). Animations keyed to cursor changes, so scrubbing backward
  replays them in reverse for free.
- **Per-token distribution inspection**: hovering a generated token in
  the stream shows a popover with the probability distribution it was
  sampled from (the cycle's softmax top-k, chosen token highlighted) —
  every token's alternatives stay explorable after the fact.

## Error handling

- **Model download**: per-file progress (transformers.js callbacks);
  mode toggle warns "~120 MB download" before first use. On failure:
  retry + one-click fallback to simulated mode. Simulated is the default
  on first visit; the app never blocks on the real model.
- **Backend selection**: WebGPU → WASM automatic fallback; active
  backend shown in a status chip. If WASM init also fails, real mode is
  disabled with an explanatory tooltip.
- **Worker crashes / inference errors**: worker wraps its loop in
  try/catch and emits `{ type: 'run-end', reason: 'error', message }` —
  failure is just another trace event; the partial trace remains fully
  inspectable.
- **Abort/re-run**: a new run aborts the active one (abort latency ≤ one
  forward pass). One trace per run; no trace archive in v1.
- **Input edge cases**: empty prompt disables Generate; over-long
  prompts truncate to the context window with a visible notice at the
  tokenizer stage.

## Testing

- **Engines (Vitest, node)**: SimulatedEngine is seeded-deterministic —
  assert exact trace structure: event ordering per cycle, probability
  sums ≈ 1, top-k sorted, same prompt → same trace. A shared **trace
  contract suite** (event-ordering assertions) runs against any engine;
  the TransformersEngine worker protocol is tested against a mocked
  transformers.js module. Real inference stays out of CI.
- **Player (Vitest)**: exhaustive unit tests of the pure reducer —
  boundary stepping, park-at-frontier vs pause-on-run-end, seek
  clamping, pacing table.
- **Components (Vitest + Testing Library)**: fixed traces at fixed
  cursors → assert active stage and detail-panel contents. No animation
  testing.
- **E2E (Playwright, one smoke test)**: load app, type prompt, run
  simulated mode, step through a full cycle, assert a token lands in the
  output stream. Real-mode E2E is manual (no 120 MB download in CI).

## Simulated attention heatmaps (post-v1 addition)

Sim mode emits one `attention` trace event per cycle, after the cycle's
`layer` events and before `logits`:

```ts
| { type: 'attention'; cycle: number; heads: Array<{
    layer: number; head: number;
    label: 'previous-token' | 'attention-sink' | 'induction' | 'coreference';
    matrix: number[][]   // ragged causal: row i covers positions 0..i, sums to 1
  }> }
```

Patterns are deterministic and hand-shaped (no PRNG draw — the engine's
random stream is untouched): a previous-token head, an attention-sink
head, an induction head that recomputes over the full sequence each
cycle, and — for the curated coreference example only — a handcrafted
head linking the pronoun to its antecedent, anchored by finding the
tokens rather than hardcoded indices (omitted if the anchors are missing
under a different tokenizer). Curated example prompts are one-click chips
under the prompt input; clicking fills the prompt and starts generation.
Curated runs follow a hand-written continuation: the scripted token is
inserted as the clearly-dominant top candidate each cycle and chosen
deterministically (temperature still reshapes the displayed softmax but
not the outcome on scripted runs), ending with `eos` when the script is
exhausted. Stage cards are clickable and seek to the current cycle's
representative event (Layers → its `attention` event, Logits → its
`softmax`, Sampler → its `sample`), giving stage × cycle navigation
together with the cycle-step buttons.
The Layers detail panel renders the heatmap with head-selector chips,
per-label reading hints, and an "illustrative" disclaimer. Real mode
emits no attention events, so the panel stays schematic there; when a
custom model export with real attention outputs exists (see
docs/research/2026-08-27-attention-weights-in-browser.md), the same
event and UI carry the real data.

## Out of scope (v1)

- Real attention weights (simulated attention heatmaps shipped post-v1,
  see below; real weights need a custom ONNX export — docs/research/)
- Chat templating & system-prompt assembly stage
- Trace archive / comparison of multiple runs side by side
- Mobile-optimized layout (desktop-first; should degrade gracefully)
