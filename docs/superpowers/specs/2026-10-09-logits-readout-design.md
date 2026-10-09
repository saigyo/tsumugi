# Logits readout: the dot product made visible — design

**Date:** 2026-10-09
**Status:** Approved design, pre-implementation
**Builds on:** `docs/superpowers/specs/2026-09-02-embeddings-explained-design.md`
(M4: the `inputs_embeds` export output, the Hub geometry asset, the
Embeddings card) and the export tool in `tools/export` from
`docs/superpowers/specs/2026-08-27-real-attention-export-design.md`.

## Purpose

The Embeddings card ends with a pointer: "the same matrix is reused at the
end: the final vector is compared against every row to score each token".
The Logits card the pointer lands on is a bar chart with a formula. This
work makes the pointer land on something: the Logits card shows the
readout as the Embeddings lookup read in reverse, and in real mode it
shows the actual dot product — the last token's final vector, the
candidate's row of E, and their element-wise product whose cells add up
to the logit on the bar.

Two ideas, in this order:

1. **Structure — one vector against many rows.** The last token's final
   vector `x_last [1 × 576]` is compared against every row of the same
   `E [49152 × 576]` seen at the Embeddings stage. One score per row; the
   bars are the ten highest.
2. **Mechanism — a score is a sum of 576 products.** For any candidate,
   `logit = Σ_d x_last[d] · E[id][d]`. Cells where the two vectors agree in
   sign push the score up, cells where they disagree pull it down.

## Decisions already made

- **The model change comes first** so the card is built against real
  data. The export gains three outputs; the Hub model is republished.
- **Candidate rows come from the graph.** Full 576-dimension rows of E for
  the candidates are gathered inside the ONNX graph for the 16 top-scoring
  ids at the last position. The alternatives — a second forward pass per
  cycle, or downloading the whole table (28 MB int8) — were rejected. The
  UI caps top-k at 10; 16 leaves headroom.
- **Exactness.** The new Gather node joins `/lm_head/MatMul` on the
  quantization exclusion list, so candidate rows are fp32 and the summed
  products reproduce the logit to display precision.
- **Pooled, not full, in the trace.** The trace stores the exact final
  vector but pools each candidate's row and product into 64 cells. A
  50-cycle run grows by roughly 400 KB rather than 2 MB.
- **Pinned revision.** Transformers.js caches model files by URL and never
  revalidates, so returning visitors would keep the old model until they
  cleared site data. The app loads the model and the geometry asset from a
  Hub tag named in one constant; `publish` creates the tag. Bumping the
  constant is the cache-bust.
- **Pure-function rule preserved:** the card is a function of
  `(trace, cursor, asset)`. Everything real-mode specific rides in the
  trace; the asset supplies reduced rows for sim mode and legacy traces,
  exactly as the Embeddings card does.
- **No mixing of exact and reduced vectors in one strip.** Real traces show
  worker-pooled exact data; sim mode and legacy traces show PCA-64 asset
  rows with the Embeddings card's provenance caption and no products.

## Component 1: export tool (`tools/export`)

### Graph outputs

The post-processing step (`postprocess.py`) today keeps `hidden_states.0`
as `inputs_embeds` and drops `hidden_states.1..N`. It now also:

| Output | Shape | Meaning |
|---|---|---|
| `last_hidden_state` | `[batch, seq, 576]` float32 | `hidden_states.N` (N = `num_hidden_layers`, 30): the residual stream after the final RMSNorm — the tensor `lm_head` multiplies. HF's own name for this tensor. With the KV cache, `seq` is the number of tokens fed this call. |
| `topk_ids` | `[batch, 16]` int64 | the 16 highest `logits` at the last position, descending |
| `topk_embeds` | `[batch, 16, 576]` float32 | `E[topk_ids]`, the candidates' rows of the embedding table |

Construction, all added nodes named with a `tsumugi_` prefix so they can be
addressed by name:

1. `Slice(logits, starts=[-1], ends=[INT64_MAX], axes=[1])` → `[batch, 1, vocab]`,
   `Squeeze(axes=[1])` → `[batch, vocab]`.
2. `TopK(k=16, axis=-1, largest=1, sorted=1)` → values and indices, both
   `[batch, 16]`. The indices become graph output `topk_ids`; the values
   output is named but consumed by nothing, which ONNX allows.
3. `Gather(Wᵀ, topk_ids, axis=1)` → `[576, batch, 16]`, `Transpose(perm=[1,2,0])`
   → `[batch, 16, 576]` → graph output `topk_embeds`. Gather node name
   `tsumugi_topk_embeds_gather`.

`Wᵀ` is the lm_head's own weight, located structurally, not by name: the
node producing `logits` is a `MatMul` (possibly behind `Identity`/`Cast`
nodes, which the locator walks through) whose second input is an
initializer of shape `[576, vocab]`. The export keeps this fp32 copy of
`Eᵀ` even though the model ties the weights, and it is what the quantizer
leaves alone when `/lm_head/MatMul` is excluded. The embedding table
itself (`model.embed_tokens.weight`) is **not** a usable source: the
quantizer turns it into int8 for the embedding `Gather`, and a second
consumer, even an excluded one, gets a dequantized copy (found during the
first operator run: rows off by 7.5e-3). Reading the lm_head weight also
makes the rows exactly those that produced the logits. Post-processing
raises if the locator finds no such `MatMul`. The constant `TOPK = 16`
lives in `tsumugi_export/__init__.py` next to the repo ids.

`quantize.py` excludes `tsumugi_topk_embeds_gather` alongside
`/lm_head/MatMul`. The other new nodes carry no weights.

Cost per step: one TopK over 49 152 values and a 16-row gather —
negligible next to a forward pass.

### Validation (`validate.py`), both blocking, run per variant

- `last-hidden-parity`: for `PROMPTS[0]`, `last_hidden_state[0, -1] @ E.T`
  equals `logits[0, -1]` within `atol 1e-3` for `model.onnx` and `2e-2` for
  the quantized file; shape must be `[1, seq, 576]`.
- `topk-parity`: `topk_ids[0]` equals `argsort(-logits[0, -1])[:16]` as a
  set (order may differ only among exact ties), and
  `topk_embeds[0] == E[topk_ids[0]]` within `atol 1e-3` for every variant
  (the gather is excluded from quantization, so no quantization slack).

Both checks appear in the printed verdict table and in
`validation-report.json`, and therefore in the publish gate without
further changes.

### Publish (`publish.py`)

- Model card: a paragraph after the `inputs_embeds` one describing the
  three outputs and their shapes, and naming the tag.
- `--tag` is required (no default), so an operator cannot publish untagged
  by accident. Before uploading anything, `publish` lists the repo's refs
  and **refuses if the tag already exists**: a Hub tag never moves, so
  re-publishing under the same name would silently leave the app on the
  first upload, and browsers that cached that URL would never re-check.
  After `upload_folder`, `HfApi.create_tag(repo_id, tag, revision=<the
  upload's commit>)`. A fix after an in-app check is therefore a **new
  tag** (`v3` → `v4`) plus a bump of `ATTN_MODEL_REVISION`. The runbook
  names the tag for this release: `v3` (v1: attention outputs, v2:
  inputs_embeds and geometry, retroactively).

### Docs and tests

- `tools/export/README.md`: the new outputs in the intro, `--tag` in the
  publish step, and a runbook step 8 item: in the app, the Logits card
  shows a dot-product bench whose sum matches the logit.
- `tests/test_postprocess.py`: on a hand-built tiny graph (a Gather on a
  `[7 × 4]` table feeding a fake `hidden_states.0`, a `[1, 3, 7]` `logits`
  output, `hidden_states.1`), post-processing yields exactly the expected
  output names, `onnx.checker` passes, and running it in onnxruntime
  returns `topk_ids` equal to numpy's top-k and `topk_embeds` equal to the
  gathered rows. `TOPK` is patched to 3 for the test.
- `tests/test_quantize_exclusions.py` (or an assertion in an existing
  test): the exclusion list contains both node names.

## Component 2: app constants and model loading

`src/engine/tokenizer.ts`:

```ts
export const ATTN_MODEL_REVISION = 'v3'
export const GEOMETRY_BASE_URL =
  import.meta.env.VITE_GEOMETRY_BASE_URL
  ?? `https://huggingface.co/${ATTN_MODEL_ID}/resolve/${ATTN_MODEL_REVISION}/geometry`
```

The worker passes `revision: ATTN_MODEL_REVISION` to
`AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`
for the attention model only (the stock fallback keeps `main`). The
`ready` message and the status chip are unchanged. Tests assert the
geometry URL contains the revision.

## Component 3: trace and engines

`logits` event:

```ts
| { type: 'logits'; cycle: number
    topK: Array<TokenInfo & { logit: number; row?: number[]; dot?: number[] }>
    lastHidden?: number[] }
```

- `lastHidden`: the exact final vector of the last fed token, 576 values
  rounded to 3 decimals. Present only when the graph returned
  `last_hidden_state` with shape `[1, fed, 576]`.
- `row`: the candidate's row of E pooled by **mean** into `READOUT_CELLS = 64`
  cells of 9 dimensions, 3 decimals. `dot`: the element-wise product
  `x_last ⊙ E[id]` pooled by **sum** into the same 64 cells, 3 decimals, so
  `Σ dot` matches the logit (itself rounded to 2 decimals) to the one
  decimal the card displays; 2 decimals would let 64 roundings drift by
  up to 0.3. Both present only when the
  candidate's id appears in `topk_ids` (always true for the top-10 when the
  graph output is present and well-formed) and `lastHidden` is present.
- `READOUT_CELLS` lives in `src/trace/types.ts` next to the event type, as
  the shared contract between worker pooling and the card's pooling of
  `lastHidden`.
- **Validator** (`src/trace/validate.ts`): when present, `lastHidden` is a
  numeric array; `row` and `dot` are numeric arrays of length
  `READOUT_CELLS`. Legacy events without the fields stay valid.
- **Worker** (`worker.ts`): after the forward call, a pure helper
  `extractReadout(out, fed, dims, topIds)` in
  `src/engine/transformers/readout.ts` returns
  `{ status: 'absent' } | { status: 'bad-shape'; detail: string } |
  { status: 'ok'; lastHidden: number[]; byId: Map<number, { row: number[]; dot: number[] }> }`.
  It reads `last_hidden_state` (last row), `topk_ids`, `topk_embeds`,
  checks shapes, pools, and matches candidates **by id**, never by
  position (the worker's own `topK` and the graph's TopK agree on the
  logits but may order exact ties differently). `absent` → event without
  the fields (old cached model). `bad-shape` → one `console.warn`, a
  per-run `readoutOff` flag, fields omitted for the rest of the run —
  the never-fail policy from attention and `inputs_embeds`.
- **SimulatedEngine**: unchanged; it emits none of the new fields.
- **Fixtures** (`src/test/fixtures.ts`): the real-mode fixture's `logits`
  events carry `lastHidden`, `row` and `dot` with small deterministic
  values whose `dot` cells sum to the fixture logit; the sim fixture does
  not.

Budget: per cycle, 576 + 10 × (64 + 64) ≈ 1 900 numbers, about 8 KB of
JSON; a 50-cycle run adds about 400 KB — comparable to M4's embedding
rows. The archive ring of 8 unpinned runs stays within a few MB.

## Component 4: the card

`LogitsDetail.tsx` becomes a container following `EmbeddingsDetail.tsx`:
it reads the cycle's `logits` and `softmax` events, `run-start.vocabSize`
and `embed.dims`, resolves the geometry asset through `useGeometry`, owns
one piece of state (the selected candidate index, default 0, snapping to
0 when the cursor moves to another cycle), and renders `LogitsReadout`.

```
┌ Logits ────────────────────────────────────────────────────────────────┐
│ Each score is a dot product: the last token’s final vector against one │
│ row of E — the same matrix the Embeddings stage looked rows up in.     │
│                                                                        │
│  x_last [1×576]              ·   Eᵀ [576×49152]      =   scores [49152]│
│  ▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒      ┌───────────┐          ␣sat  ▒▓░▒▓░ ████████ 18.3 │
│  the last position after         │ E         │ ←━━      ␣was  ░▒▓░▒▓ ██████   16.1 │
│  layer 30 and the final norm     │ 49152×576 │ ←─       ␣is   ▒░▒░▒░ ████     14.0 │
│                                  └───────────┘ ←─       …                     │
│                                                                        │
│ ── x_last ⊙ E[␣sat] ─────────────────────────────────────────────────── │
│  x_last    ▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒▓▓░░▒▒      │
│  E[␣sat]   ░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒░▓▒      │
│  product   ▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓      │
│  64 cells of 9 dimensions; blue cells add to the score, red subtract.  │
│  Σ = 18.31  — the logit on the bar.                                    │
│                                                                        │
│  ⓘ tied with Embeddings   ⓘ similar rows, similar scores   ⓘ 49 152 scores, 10 shown │
└────────────────────────────────────────────────────────────────────────┘
```

**`LogitsReadout.tsx`** (presentation, pure):

- *Equation row.* Left: `x_last` as a 64-cell strip (pooled by mean from
  `lastHidden` with `poolRow`) in the Embeddings diverging scale; when
  absent, a strip-shaped outline labelled `[1×576]` with the caption "not
  recorded in this run" (sim mode, legacy traces, old cached model).
  Middle: the E matrix box from `EmbeddingLookup` — same size, same
  classes — labelled `E` and `[49152 × 576]`, with one row mark per
  candidate at a schematic height derived from its id (same formula as the
  Embeddings row mark) in the candidate's hue, the selected one drawn
  thicker; a small `Eᵀ` note says it is read sideways here. Right: the
  candidate list.
- *Candidate rows.* Token chip (`token-chip hue-i`, click selects), a
  compact product strip (`dot`, 64 cells at 3 px) when present, the
  existing bar, the value. In the softmax state the bars and values switch
  to probabilities as today; the strips and the bench do not change.
- *Bench.* For the selected candidate: three aligned 64-cell strips at
  5 px — `x_last`, `E[id]` (`row`), and `product` (`dot`) — with row labels
  and the sum of the `dot` cells to 2 decimals next to the logit. Each
  strip's colour scale is relative to its own peak (as in Embeddings).
  Cells carry `<title>` tooltips: "dims 63–71: x̄ 0.42 · ē −0.18 → −0.77".
  Without `dot` (sim/legacy) the bench shows `x_last` placeholder and the
  candidate's row from the asset — 64 PCA cells, the Embeddings caption
  "PCA-reduced, offline" — and no product line; one line underneath:
  "Run a real-mode generation to see the actual products."
- *Formula line.* The existing `logits-formula` element stays, now above
  the equation row.
- *Callouts* (`embed-callout` pattern): **tied with Embeddings** — "This is
  the Embeddings matrix read sideways: there a row was looked up by id;
  here every row is scored against x_last. Tied weights mean the model
  learns one vocabulary geometry, not two." **similar rows, similar
  scores** — "Tokens whose rows of E point the same way get nearly the
  same dot product, which is why near-synonyms sit together at the top."
  **49 152 scores, 10 shown** — "The multiply produces one score per
  vocabulary entry; the trace keeps only the top-k the sampler can
  choose from."
- `data-testid` hooks: `logits-readout`, `readout-x`, `readout-matrix`,
  `readout-candidate`, `readout-bench`, `readout-sum`, `readout-product`.

**`VectorStrip.tsx`**: `cellFill`, `peakOf`, and the cell-rendering loop
move out of `EmbeddingLookup.tsx` into a shared component
`<VectorStrip cells cellW cellH testId ariaLabel titleFor? />`. The
Embeddings card's rendering and tests are unchanged in behaviour; the
`embed-strip-cell` and `embed-stack-cell` test ids are preserved through
the `testId` prop.

**Styling**: `src/index.css` gains a `Logits card` block reusing the
`embed-*` tokens (col labels, captions, callouts) and the bar classes; no
new colours. The readout grid collapses to a single column under 56 rem
like the Embeddings lookup.

**Pipeline band**: unchanged (`[1×576]` → Logits, `[49 152]` → Sampler
labels already exist).

## Error handling

- Graph outputs absent (old cached model, stock fallback model): the
  event carries no readout fields; the card shows the sim/legacy
  presentation. No message beyond the bench's one-liner.
- Wrong shapes: `console.warn` once, `readoutOff` for the run, same
  presentation.
- `topk_ids` lacks one of the worker's top-10 ids (should not happen; ties
  at the 16th place could in principle): that candidate gets no `row`/`dot`
  and its product strip is blank; the bench for it shows the asset row and
  the one-liner.
- Geometry asset unavailable in sim/legacy: candidate rows are blank with
  the Embeddings "vector values unavailable" note; the structural diagram
  still renders.
- A candidate id beyond the asset's vocabulary (fallback hash tokenizer):
  guarded as in `EmbeddingsDetail`.
- The sum of `dot` cells and the logit differ by more than 0.05 after
  rounding: the bench shows both numbers anyway; the Python
  `last-hidden-parity` check is where this is prevented.

## Testing

Python (`tools/export/tests`): as listed under Component 1.

TypeScript (vitest):

- `readout.test.ts`: absent → absent; wrong `last_hidden_state`,
  `topk_ids` or `topk_embeds` shape → bad-shape with a detail; ok →
  `lastHidden` is the last fed row rounded to 3 decimals, `row` pooled by
  mean, `dot` pooled by sum, candidates matched by id with an id missing
  from `topk_ids` absent from the map.
- `validate.test.ts`: fields valid; wrong `row` length invalid; legacy
  event valid.
- `tokenizer.test.ts` (or a new small test): `GEOMETRY_BASE_URL` contains
  `ATTN_MODEL_REVISION`.
- `LogitsReadout.test.tsx`: real fixture renders the x strip, one product
  strip per candidate, a bench whose `readout-sum` equals the logit; click
  selects another candidate and the bench follows; sim fixture renders the
  placeholder, asset rows and the one-liner; softmax state switches bars
  to percentages while strips stay.
- `DetailPanel.test.tsx`: existing logits tests keep passing (`logit-bar`,
  `logits-formula`, dot-product copy); one test that the selected
  candidate resets when the cursor moves to the next cycle.
- `EmbeddingLookup`/`EmbeddingsDetail` tests unchanged and green after the
  `VectorStrip` extraction.

Playwright: `scripts/capture-screenshots.mjs` gains a `logits` scene (real
mode, Logits stage at the first cycle of a curated prompt, a candidate
selected so the bench is visible) → `docs/screenshots/logits-readout.png`.

## Delivery order

1. **Export tool**: post-processing outputs, quantize exclusion, validate
   checks, publish tag, README, tests. The executor stops and hands the
   operator the runbook: export with-past and no-cache, quantize,
   geometry, validate, publish `--tag v3`. The app keeps working against
   the new model before any app change because unknown outputs are
   ignored.
2. **App constants**: `ATTN_MODEL_REVISION`, revision in `from_pretrained`,
   geometry URL. In-app check that the new model loads and real mode still
   shows attention and exact embedding rows.
3. **Trace and worker**: event fields, `readout.ts`, validator, fixtures.
4. **Card**: `VectorStrip` extraction, `LogitsReadout`, container, CSS.
5. **Docs**: README section with the screenshot, backlog entry #13 checked
   off, memory note.

## Out of scope

- Per-layer hidden states and residual-stream evolution (would need the
  dropped `hidden_states.1..29` back; a separate spec, see M4 out of scope).
- Showing all 49 152 scores (a histogram of the full distribution); the
  trace keeps the top-k only.
- Logit lens or tuned-lens style readouts at intermediate layers.
- Chat templating (#8), mobile layout (#9) — unchanged.
