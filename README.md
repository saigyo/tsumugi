# LLM Pipeline Visualizer

A single-page app that visualizes how a prompt flows through an LLM's
autoregressive generation loop — tokenize → embed → transformer layers →
logits → softmax → sample → append → repeat. Built for developers and CS
students who are comfortable with code but want to *see* what happens
between typing a prompt and getting tokens back: tokenization, embeddings,
per-layer activations, logits, probability distributions, and sampling.

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
