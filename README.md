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
- **Real** — runs an actual small model in-browser via
  [transformers.js](https://github.com/huggingface/transformers.js), in a
  Web Worker. It first tries a custom export of SmolLM2-135M-Instruct with
  **real per-layer attention outputs**
  ([`saigyo-hoshi/smollm2-135m-attn-onnx`](https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx),
  ~240 MB, downloaded once and cached): the Layers stage then shows measured
  attention heatmaps, with head roles (previous-token, attention-sink,
  induction) detected statistically from the weights on your prompt and
  labeled with their evidence scores. If that download fails it falls back
  to the stock `HuggingFaceTB/SmolLM2-135M-Instruct` export (~120 MB,
  schematic layers). Prefers WebGPU, falls back to WASM automatically.

## One matrix, end to end — the residual stream

The pipeline stages look like separate machines, but there is only one
object flowing through the whole pipeline: a matrix **X of shape
`[seq_len × d_model]`** — one 576-dimensional vector per token position
(SmolLM2's numbers). This is the *residual stream*, and it answers every
"what happens between the stages" question:

- **Embeddings → Layers**: no hand-off, no conversion. The embedding
  stage's output — one vector per token, looked up from the embedding
  table — *is* layer 0's input, verbatim. The heat-strips in the
  Embeddings panel are (downsampled) rows of exactly the matrix that
  enters the first layer.
- **Layer → Layer**: each layer edits this matrix and passes it on —
  and it edits by **addition**: `X ← X + attention(X)`, then
  `X ← X + MLP(X)`. Each token's vector accumulates refinements layer by
  layer, like a document going through thirty rounds of margin notes;
  the original embedding is still in there, with the corrections added
  on top. Inside a layer the jobs are strictly divided: **attention is
  the only place information moves between token positions** (the
  heatmap row for token *i* is the mixing recipe — row "it" = 65% "cat"
  means the vector at "it" gets a large dose of "cat"'s vector added
  in), while the MLP transforms each position's vector in isolation.
- **Layers → Logits**: take only the **last token's row** of the final
  matrix — one vector — normalize it, and multiply by the unembedding
  matrix (`d_model × vocab`, essentially the embedding table
  transposed). Each logit is literally a **dot product**: how similar is
  the final state of the last position to vocabulary token *v*'s
  direction? That single matrix multiply *is* the Logits stage; softmax
  and sampling just turn the ~49k scores into a choice.

So the causal chain to the probabilities runs: attention weights decide
which vectors get blended into the last position's vector; the layers'
accumulated edits shape that vector; the logits read it out against
every vocabulary direction. The app shows this carrier explicitly — the
tensor-shape labels on the pipeline arrows (watch `[10×576]` narrow to
`[1×576]` before Logits), the layer-anatomy diagram in the Layers panel,
and the readout formula in the Logits panel.

The name of the app tells the same story: each token is a thread,
attention decides which existing threads get twisted into the new one at
each layer, and at the end the finished strand is held up against 49k
reference threads to see which it resembles most.

## Reading the attention heatmaps

The Layers stage shows attention heatmaps (simulated mode): a triangular
grid per attention head, where row *i* shows how much the token at
position *i* attends to each earlier token — every row sums to 100%.
Mechanically these weights are the mixing recipe of the residual
stream's attention step described above: they decide whose vectors get
blended into each position.
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

Two honest caveats. First, simulated mode's heatmaps are **illustrative**
— deterministic, hand-shaped patterns of the kinds real models exhibit —
while real mode shows **measured** weights from the custom model export
(the footer under each heatmap says which you're looking at; head roles
in real mode are detected from the weights, not labeled by the model).
Second, even real attention weights are **not explanations** —
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
