# tsumugi-export

Exports `HuggingFaceTB/SmolLM2-135M-Instruct` to ONNX with per-layer
attention-probability outputs, the embedding lookup (`inputs_embeds`) and the
readout (`last_hidden_state`, `topk_ids`, `topk_embeds`) and the MLP internals
(`edit_ratios`, `stream_parts`), plus a
vocabulary-geometry asset, quantizes it, validates the result against
the stock model, and publishes it to `saigyo-hoshi/smollm2-135m-attn-onnx`
on the Hugging Face Hub — for use by Tsumugi's in-browser attention
visualization.

This tool is a standalone [uv](https://docs.astral.sh/uv/) project. It is
**never run in CI**: every command past the CLI skeleton downloads or
processes hundreds of megabytes of model weights and is operator-gated.

## Setup

```bash
cd tools/export
uv sync
```

## Commands

Run any command with `uv run python -m tsumugi_export <command> --help`.

- `export` — export SmolLM2 to ONNX with attention outputs (`--out`,
  `--no-cache` for the Approach B full-matrix-per-step variant).
- `quantize` — produce `model_q4.onnx` (primary) and `model_fp16.onnx`
  (insurance) beside the fp32 export (`--model-dir`).
- `geometry` — build `geometry/` (exact top-12 neighbours, PCA-64 int8
  vectors, token texts, manifest) from the stock embedding table
  (`--model-dir`).
- `validate` — run the exported graphs in onnxruntime, check logits parity
  with the stock export, attention-row well-formedness, cache integrity,
  and (when both variants are present) cached-vs-no-cache equivalence;
  writes `validation-report.json` (`--model-dir`).
- `publish` — assemble the repo layout (model card, tokenizer files, ONNX
  variants) and upload to the Hugging Face Hub; refuses to run without a
  passing `validation-report.json` whose artifact hashes match the files on
  disk (`--model-dir`, `--repo-id` to override the target repo). `--tag`
  (required) creates a Hub tag on the published commit; the app pins it via
  `ATTN_MODEL_REVISION` in `src/engine/tokenizer.ts`. A tag that already
  exists is refused — tags never move.

## Manual verification protocol

Because this tool never runs in CI, an operator must verify each real
export by hand before it ships. This is not a placeholder — treat it as
the actual acceptance checklist for a release.

1. **Fast network check first.** Before running the heavy commands,
   confirm you can reach the Hugging Face Hub (`hf auth whoami` or
   `curl -sI https://huggingface.co`) and that you're authenticated
   (`HF_TOKEN` env var or `hf auth login`) if you intend to `publish`.
2. **Export the primary (with-past) variant:**
   `uv run python -m tsumugi_export export --out out/model`
3. **Export the no-cache (Approach B) variant, for the A≡B equivalence
   check:**
   `uv run python -m tsumugi_export export --out out/model-nocache --no-cache`
   This step is not optional. `validate` looks for
   `out/model-nocache/onnx/model.onnx` specifically; without it, the
   `a-equiv-b` check is reported as **SKIPPED** rather than run, and the
   release has *not* been fully validated — cached incremental attention
   rows will not have been checked against the full-matrix ground truth.
4. **Quantize:** `uv run python -m tsumugi_export quantize --model-dir out/model`
5. **Build the geometry asset:** `uv run python -m tsumugi_export geometry --model-dir out/model`
6. **Build the neuron asset:** `uv run python -m tsumugi_export neurons --model-dir out/model --revision v7`
   `--revision` is the tag you will publish under; `publish` refuses a manifest built for another tag, and the app refuses one that does not match `ATTN_MODEL_REVISION`.
7. **Validate:** `uv run python -m tsumugi_export validate --model-dir out/model`.
   Do not skip this — `publish` refuses to run without a passing report,
   but you should also read the printed verdict table yourself (including
   confirming `a-equiv-b` actually ran, not SKIPPED, and that
   `*:inputs-embeds-parity`, `*:last-hidden-parity`, `*:topk-parity`,
   `*:edit-ratios-parity`, `*:stream-parts-sum`, `*:votes-sum`,
   `*:edit-cache-parity`, `*:activations-parity`, `*:inv-rms-parity`,
   `*:neuron-votes-sum`, `*:argmax-parity`, `*:steps-parity`,
   `*:silu-product`, `*:knockout-parity`, `geometry` and `neurons` must PASS) rather than relying solely on the exit code.
   Quantized variants are validated with graph optimizations disabled, because onnxruntime-web runs the unfused graph (#19).
   The quantized variant's edit-ratio checks compare values clipped at
   2.0 with atol 0.25: quantization moves the huge first-token MLP ratios
   noticeably, and the card draws every ratio ≥ 1 alike. The fp32
   checks (atol 1e-3, unclipped) are what prove the outputs are wired right.
7b. **Browser check.** `npm run dev` in the repository root, then open `http://localhost:5173/tools/export/browser-check.html` in Chrome. It loads `out/model` through transformers.js on WebGPU and WASM and must end with `VERDICT: PASS`. This is the pre-publish in-app check step 9 used to lack.
8. **Publish:** `uv run python -m tsumugi_export publish --model-dir out/model --tag v7`
   Then set `ATTN_MODEL_REVISION` in `src/engine/tokenizer.ts` to the same
   tag and deploy the app. A release is always these two steps: browsers
   cache model files by URL and never re-check, so publishing alone changes
   nothing for the app, and bumping the constant alone points at a tag that
   does not exist. Tags never move and `publish` refuses an existing one: if
   the in-app check in step 9 fails, fix, re-validate, publish under the
   **next** tag (`v8`) and bump the constant again.
9. **In the actual Tsumugi app**, load the exported model in real
   (non-fallback) attention mode and check the items below. Pre-publish
   in-app testing against a not-yet-uploaded artifact is not supported in
   M1 — this step necessarily runs against the model repo `publish` just
   uploaded, so verification happens post-publish; if it fails, fix and
   publish under a new tag (see step 8).
   - **One curated prompt** with a known, hand-checked attention pattern
     (e.g. an induction-style repeat like `"one two three one two three
     one"`) — confirm the highlighted source token(s) for the induction
     head(s) match what you expect.
   - **One free-form prompt**, run twice (a repeat), confirming the
     attention rows are stable and deterministic across the two runs.
   - The UI shows the **`· attn`** chip indicating real (not synthetic)
     attention data is being used.
   - The highlighted token labels are **plausible** — they land on
     semantically or positionally sensible source tokens, not noise.
   - Each attention row's displayed weights **sum to 100%** (row-stochastic,
     matching what `validate` already checked at the ONNX level).
   - **The Logits card** shows a dot-product bench for the top candidate;
     the sum under the product strip equals the logit on the bar to one
     decimal.
10. **Stock fallback check.** Simulate the attention repo being unreachable
   (e.g. block the `saigyo-hoshi/smollm2-135m-attn-onnx` host, or point the
   app at a bad repo id) and confirm the app falls back gracefully to the
   stock model without attention data, rather than erroring out.

Only after all nine steps pass — including a non-SKIPPED `a-equiv-b`
result — should the exported artifacts be considered release-ready.

**Operator note on `model_fp16.onnx:attn-wellformed`.** The row-sum
tolerance there is 1e-3. A marginal failure with row sums landing just
outside that band (e.g. 1.001–1.002) is expected fp16 rounding noise
accumulating across a softmax over many keys — a numerics question, not
evidence of a broken graph. Re-run validate to confirm the failure is
small and stable before treating it as a real defect; a large or wildly
varying deviation is a different story and should be investigated.

## Geometry asset layout

| File | Content |
|---|---|
| `manifest.json` | modelId, vocabSize, dims, k, pcaDims, scale, explainedVariance, sourceSha256, files |
| `neighbors.bin` | ids block (`vocabSize × k` little-endian uint16) then sims block (`vocabSize × k` uint8, `round(max(cos,0)×255)`), per token sorted by similarity descending, self excluded |
| `vectors.bin` | `vocabSize × pcaDims` int8, `float = int8 × scale`, PCA on the mean-centred table |
| `tokens.json` | JSON array of decoded token strings, index = id |
