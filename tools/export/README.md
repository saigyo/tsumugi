# tsumugi-export

Exports `HuggingFaceTB/SmolLM2-135M-Instruct` to ONNX with per-layer
attention-probability outputs, quantizes it, validates the result against
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
- `validate` — run the exported graphs in onnxruntime, check logits parity
  with the stock export, attention-row well-formedness, cache integrity,
  and (when both variants are present) cached-vs-no-cache equivalence;
  writes `validation-report.json` (`--model-dir`).
- `publish` — assemble the repo layout (model card, tokenizer files, ONNX
  variants) and upload to the Hugging Face Hub; refuses to run without a
  passing `validation-report.json` whose artifact hashes match the files on
  disk (`--model-dir`, `--repo-id` to override the target repo).

## Manual verification protocol

Because this tool never runs in CI, an operator must verify each real
export by hand before it ships. This is not a placeholder — treat it as
the actual acceptance checklist for a release.

1. **Fast network check first.** Before running the heavy commands,
   confirm you can reach the Hugging Face Hub (`huggingface-cli whoami` or
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
5. **Validate:** `uv run python -m tsumugi_export validate --model-dir out/model`.
   Do not skip this — `publish` refuses to run without a passing report,
   but you should also read the printed verdict table yourself (including
   confirming `a-equiv-b` actually ran, not SKIPPED) rather than relying
   solely on the exit code.
6. **Publish:** `uv run python -m tsumugi_export publish --model-dir out/model`
7. **In the actual Tsumugi app**, load the exported model in real
   (non-fallback) attention mode and check:
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
8. **Stock fallback check.** Simulate the attention repo being unreachable
   (e.g. block the `saigyo-hoshi/smollm2-135m-attn-onnx` host, or point the
   app at a bad repo id) and confirm the app falls back gracefully to the
   stock model without attention data, rather than erroring out.

Only after all eight steps pass — including a non-SKIPPED `a-equiv-b`
result — should the exported artifacts be considered release-ready.
