# Real attention weights for Tsumugi — export tooling and integration (M1)

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation
**Prerequisite reading:** `docs/research/2026-08-27-attention-weights-in-browser.md`
(mechanism, constraints, and spike evidence this design builds on)

## Purpose

Real mode currently renders the Layers stage schematically because stock ONNX
exports expose no attention weights. This milestone (M1) delivers:

1. A Python export tool that produces a SmolLM2-135M-Instruct ONNX graph with
   per-layer, per-head attention outputs, validated and published to the
   Hugging Face Hub.
2. Worker integration: real mode consumes those outputs, accumulates per-head
   attention matrices, detects head roles statistically, and emits the
   existing `attention` trace event with real data.
3. Minimal UI adaptation: the existing heatmap viewer shows detected real
   heads with their evidence scores.

A second milestone (separate spec, out of scope here) adds the full-grid
small-multiples browser over all layers × heads and per-layer aggregates.

## Decisions already made

- **Hosting:** the user's Hugging Face account. Model repo:
  `saigyo-hoshi/smollm2-135m-attn-onnx`.
- **Scope:** two milestones; this spec is M1 only.
- **Real-mode behavior:** the attention-enabled export replaces the stock
  model in real mode, with automatic fallback to stock (schematic layers,
  today's behavior) if the attention model fails to load.
- **Extraction strategy:** Approach A — cache-enabled export with per-step
  attention rows accumulated client-side; Approach B (cache-less export,
  full matrices per step, the community-proven template) is the documented
  fallback, selectable by a tool flag, and requires no downstream redesign
  (prefill handling is identical; B's per-step matrices are consumed by
  taking their last row).

Why A: with `use_cache=True`, each decode step emits one attention row per
head (`[1, heads, 1, past+1]`) — the new token's attention over its past.
Row *t* computed at step *t* is the model's exact causal attention for that
token; later steps never revise earlier rows, so client-side accumulation
reconstructs the exact triangular matrices while generation stays
O(1)-per-token and per-step readback stays small. The prefill pass emits the
full prompt triangle at once.

## Component 1: the export tool (`tools/export/`)

Self-contained Python project, isolated from the npm package and never run in
CI. Managed with **uv**: `pyproject.toml` + `uv.lock`, all commands invoked
via `uv run`; the README's setup section is `uv sync` and nothing else.
Python ≥ 3.10. Dependencies: `optimum[exporters]`, `onnx`, `onnxruntime`,
`transformers`, `huggingface_hub`.

CLI: `uv run python -m tsumugi_export <command>` with four commands:

### `export`
- Custom `OnnxConfig` subclass extending Optimum's Llama-family text-decoder
  config; overrides `outputs` to add `attentions.{i}` (one per layer) with
  dynamic axes `{0: batch, 2: query_len, 3: kv_len}` alongside the standard
  `logits` + `present.*` outputs.
- Passed via `main_export(..., custom_onnx_configs=...)` with
  `model_kwargs={"output_attentions": True, "attn_implementation": "eager"}`
  — eager attention is required; SDPA/flash kernels never materialize the
  weight matrices.
- `--no-cache` flag switches to Approach B (`use_cache=False`, no
  `present.*` outputs, full attention matrices per call).
- Output: a local model directory (loadable by the app during development
  before anything is published).

### `quantize`
- Produces `onnx/model_q4.onnx` (the file transformers.js `dtype: 'q4'`
  loads) and `onnx/model_fp16.onnx` as insurance. Quantization is
  weights-only; attention outputs are activations and are not themselves
  quantized, but whether q4 preserves their correctness is the unproven part
  — decided empirically by `validate`. If q4 fails validation, publish q8 or
  fp16 instead and record that in the model card.

### `validate`
Runs before any publish; loads exports in Python `onnxruntime` and asserts,
on a fixed prompt set (including one with repeated tokens):

1. **Logits parity**: greedy continuations and last-position logits match the
   stock export within tolerance — the added outputs must not perturb
   generation.
2. **Attention well-formedness**: every attention row sums to 1 (±1e-4) and
   is causal (no mass on future positions).
3. **A ≡ B equivalence**: the cached path's incremental rows equal the
   corresponding rows of the no-cache full-matrix export — this check is the
   mechanical verdict on Approach A; if it fails, ship Approach B.
4. **Cache integrity**: `present.*` outputs exist (cached variant) and
   multi-step cached generation matches single-shot full-context logits.

Prints a verdict table and writes `validation-report.json` (checks, results,
artifact file hashes) into the model directory; `publish` refuses to upload
an artifact whose hash lacks a passing report, and uploads the report as part
of the repo.

### `publish`
Assembles the HF repo layout and uploads via `huggingface_hub` (token from
the environment / `hf auth login`). Model card records provenance (derived
from HuggingFaceTB/SmolLM2-135M-Instruct), what was added, which validation
checks passed, quantization level, and a link back to the tsumugi repo.

## Component 2: the HF model repo

`saigyo-hoshi/smollm2-135m-attn-onnx`, laid out like an `onnx-community`
transformers.js repo so `AutoModelForCausalLM.from_pretrained` works with
only a model-id change:

```
config.json               (copied from stock; num layers/heads unchanged)
generation_config.json    (copied from stock)
tokenizer.json / tokenizer_config.json / special_tokens_map.json  (copied)
onnx/model_q4.onnx        (primary; or q8/fp16 per validation verdict)
onnx/model_fp16.onnx      (insurance variant)
README.md                 (model card, provenance, validation results)
```

The app continues to load the **tokenizer from the stock repo** (unchanged,
already cached for existing users); only model weights come from the new
repo. Sim-mode users download nothing new.

## Component 3: worker integration and head-role detection

### Model selection and protocol
- `src/engine/tokenizer.ts` gains `ATTN_MODEL_ID`; stock `MODEL_ID` stays the
  tokenizer source.
- Worker `prepare()`: try `ATTN_MODEL_ID` (q4, then fp16 on load failure);
  on any failure fall back to stock `MODEL_ID` (today's behavior). The
  `ready` message gains `attentions: boolean`; `TransformersEngine` records
  it. Download progress is reported for whichever model actually downloads.

### Accumulation
- Each forward pass, the worker reads all `attentions.{i}` outputs: at
  prefill the full prompt triangle, per step one row per (layer, head).
- Rows accumulate into per-head ragged matrices for **all** layers × heads
  (SmolLM2: 30 × 9 = 270; worst case ≈ 15 MB at max sequence length),
  worker-local, reset per run. A pure helper (e.g. `accumulateAttention`)
  performs the tensor-to-rows conversion so it is unit-testable without a
  model.

### Detection — `src/engine/transformers/attentionStats.ts` (pure module)
Per-head statistics over accumulated rows:
- `prevTokenScore`: mean mass at column *i−1* (rows *i* ≥ 1)
- `sinkScore`: mean mass at column 0 (rows *i* ≥ 1)
- `inductionScore`: over rows whose token repeats an earlier token, mean mass
  at (previous occurrence + 1); undefined when the sequence has no repeats
- `entropy`: mean row entropy (diagnostic; reserved for M2 ranking)

`selectShowcaseHeads(stats)` returns the top-scoring head per label
(`previous-token`, `attention-sink`, `induction`) above a minimum threshold
(default 0.3 mean mass; induction omitted when undefined). Coreference has no
cheap statistical detector and is not emitted in real mode — in M1 it remains
a sim-mode teaching label; discovering such heads by browsing is M2 territory.

### Trace emission
- Per cycle, after the schematic `layer` events and before `logits` (the slot
  `validateTrace` already accepts), the worker emits the existing
  `attention` event carrying only the selected heads (≤3) with their full
  accumulated matrices — the trace stays the same order of size as sim mode.
- `AttentionHead` gains `score?: number` (the detector evidence). Sim mode
  emits no scores; the field's presence distinguishes measured from
  illustrative heads.
- Accepted quirk: head selection is recomputed per cycle and may change
  between cycles as statistics evolve; scrubbing shows each cycle's own
  selection. Selection hysteresis is an M2 concern.

## Component 4: UI (M1 — deliberately minimal)

- The existing `AttentionHeatmap` renders real matrices unchanged.
- Head chips append the score when present: `previous-token L14·H3 · 0.87`.
- The footer note is conditional: heads with scores get "Measured on this
  prompt — head roles detected from the attention weights, not labeled by
  the model." instead of the "Illustrative pattern (simulated)" note.
- `LayersDetail` keeps its "(schematic — real internals not exposed)" header
  tag only when no attention data exists for the run; the pearls/norms remain
  schematic in real mode either way.
- `ModelStatus` device chip shows `webgpu · attn` (or `wasm · attn`) when the
  attention model is active; the plain device name signals the stock
  fallback.
- Sim mode is untouched.

## Error handling

- **Load-time**: attention model q4 → fp16 → stock model → (existing) fatal
  with sim-mode fallback offer. Each step reports progress; `ready` says
  what was loaded.
- **Run-time**: malformed or missing attention outputs stop attention-event
  emission for the run but never fail generation — generation always
  outranks visualization.
- Accumulators reset per run; abort and KV-cache disposal behavior unchanged.

## Testing

- **Python**: the `validate` command is the test suite for the artifact
  (logits parity, well-formedness, A≡B, cache integrity). Run manually
  before publish; never in CI.
- **TypeScript (CI)**: unit tests for `attentionStats` (crafted matrices with
  known diagonal/sink/induction structure → known scores and selections) and
  for the row-accumulation helper; the FakeWorker suite covers the `ready`
  protocol extension. Real attention events flow through the existing
  `validateTrace` contract — no new trace machinery.
- **Manual protocol** (fast network required, documented in the tool README):
  real mode on one curated prompt and one free prompt with a repeat; confirm
  the chip shows `· attn`, detected labels are plausible (a diagonal-looking
  heatmap tops `previous-token`, first-column mass tops `attention-sink`),
  rows sum to 100%, and stock fallback still works when the attn repo is
  unreachable.
- **E2E**: unchanged, sim-only.

## Out of scope (M2 and later)

- Small-multiples browser over all 270 heads; per-layer aggregate heatmaps
- Head-selection hysteresis across cycles
- Coreference-head discovery in real mode
- Hidden-state norms via graph-side `ReduceL2` outputs (noted in the research
  doc as cheap once an export pipeline exists)
