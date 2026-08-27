# 紬 Tsumugi — An LLM Pipeline Visualizer

**Tsumugi** (紬) is a traditional Japanese cloth woven from raw silk,
prized for its rough, hand-spun texture. The name plays on the related
verb 紡ぐ (*tsumugu*) — to spin, to weave: an LLM pipeline weaves its
output token by token into text, each new thread pulled from the
probability distribution the model spins over its vocabulary.

Tsumugi is a single-page app that visualizes how a prompt flows through
an LLM's autoregressive generation loop — tokenize → embed → transformer
layers → logits → softmax → sample → append → repeat. Built for
developers and CS students who are comfortable with code but want to
*see* what happens between typing a prompt and getting tokens back:
tokenization, embeddings, per-layer activations, attention patterns,
logits, probability distributions, and sampling.

You type a prompt, hit generate, and watch the pipeline stages light up as
each token is produced — live, or one step at a time via manual playback
controls.

## Modes

- **Simulated (default)** — instant, fully offline, no model download.
  Uses a small real tokenizer (falling back to a built-in tokenizer if the
  Hugging Face tokenizer files can't be fetched) plus synthetic-but-realistic
  activations, logits, and sampling, so the pipeline stages behave the way a
  real model would without needing to run one.
- **Real** — runs an actual small model (`HuggingFaceTB/SmolLM2-135M-Instruct`)
  in-browser via [transformers.js](https://github.com/huggingface/transformers.js),
  in a Web Worker. First use downloads roughly **120 MB** of model weights,
  cached by the browser afterward. It prefers WebGPU and automatically falls
  back to WASM on browsers/devices without WebGPU support.

## Reading the attention heatmaps

The Layers stage shows attention heatmaps (simulated mode): a triangular
grid per attention head, where row *i* shows how much the token at
position *i* attends to each earlier token — every row sums to 100%.
Real attention heads show strikingly legible patterns, and the simulated
heads reproduce the canonical ones:

- **Previous-token head** — a bright diagonal stripe: this head mostly
  copies from the token just before. Local syntax.
- **Attention sink** — a bright first column: many heads dump most of
  their attention on the first token as a learned "do nothing" default.
  A famous, counterintuitive phenomenon invisible in any other view of
  the model.
- **Induction head** — with a repeated pattern in the prompt
  ("one two three one …"), attention jumps from a repeated token to
  whatever followed its previous occurrence. This is the circuit that
  interpretability research credits for in-context learning.
- **Coreference** — in "The cat sat on the mat because it was tired",
  watch the row for "it" attend back to "cat": the mechanism by which
  the model resolves what a pronoun refers to.

The example chips under the prompt input load prompts crafted so these
patterns visibly connect to the input; each head's caption says what to
look for.

Two honest caveats. First, the heatmaps in this app are **illustrative**:
browser-run ONNX models don't expose their real attention weights (see
[`docs/research/`](docs/research/) for what it would take), so simulated
mode shows deterministic, hand-shaped patterns of the kinds real models
exhibit. Second, even real attention weights are **not explanations** —
they show what the mechanism computes, not *why* the model produced its
output (Jain & Wallace, "Attention is not Explanation", 2019). Read them
as "how information flows", never as "why the model answered X".

## Running it

```bash
npm install
npm run dev
```

Then open the printed local URL (defaults to `http://localhost:5173`).

## Testing

```bash
npm test      # unit tests (vitest)
npm run e2e   # end-to-end smoke test (playwright, chromium)
```

The end-to-end test drives the app in simulated mode only — it never
downloads or runs the real model, so it works fully offline and in CI. Run
`npx playwright install chromium` once before the first `npm run e2e`.

## Docs

The original design spec and implementation plan live under
[`docs/superpowers/`](docs/superpowers/):

- [`specs/2026-08-26-llm-pipeline-visualizer-design.md`](docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md)
- [`plans/2026-08-26-llm-pipeline-visualizer.md`](docs/superpowers/plans/2026-08-26-llm-pipeline-visualizer.md)

## License

[MIT](LICENSE)
