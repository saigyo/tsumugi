# Embeddings explained (M4) — design

**Date:** 2026-09-02
**Status:** Approved design, pre-implementation
**Builds on:** `docs/superpowers/specs/2026-08-27-real-attention-export-design.md`
(the custom ONNX export in `tools/export` and the Hub repo
`saigyo-hoshi/smollm2-135m-attn-onnx`) and the v1 trace architecture in
`docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md`.

## Purpose

The Embeddings stage today teaches one sentence — "text became numbers" —
and illustrates it with numbers that are not real: real mode renders
`sin(id * 0.7 + d)`, sim mode renders seeded noise. M4 replaces the card
with one that teaches two ideas, in this order:

1. **Mechanism — it is a lookup, not a computation.** A token id selects
   one row of a learned matrix `E [49152 × 576]`; the rows of the current
   sequence stacked up are the residual stream `x [n × 576]` that the
   Layers stage then edits. Three facts ride along: rows are learned by
   training, not designed; no positional information is added here
   (SmolLM2 uses rotary position embeddings inside attention); the same
   matrix reads the answer back out at the Logits stage
   (`tie_word_embeddings: true`).
2. **Geometry — meaning is distance.** Similar tokens have similar rows.
   Shown two ways: the nearest vocabulary neighbours of a selected token,
   and a cosine-similarity matrix of the current sequence against itself —
   "this is what the model knows about these tokens before any context is
   applied", a bridge to the attention heatmaps in the next stage.

## Decisions already made

- **Content:** mechanism first, then geometry. Geometry shows nearest
  neighbours and the sequence self-similarity matrix. A 2D vocabulary map
  and analogy arithmetic are deliberately excluded: projections of
  576-dim space mislead, and analogies fail too often at the input layer
  of a 135M model to be a trustworthy demo.
- **Data: Approach C.** Vocabulary geometry comes from a **static asset**
  computed offline from the stock model's embedding matrix and hosted on
  the Hub next to the ONNX model; the app fetches it lazily and computes
  neighbours and similarities **in the view**. Real mode additionally
  emits the **exact embedding rows** of the sequence into the trace, from
  a new `inputs_embeds` graph output added to the custom export. Sim mode
  stays instant: it emits no vectors and never fetches the asset before a
  run.
- **Pure-function rule preserved:** the UI is a function of
  `(trace, cursor, asset)`. The asset is deterministic per model, static,
  read-only, and fetched once per page load — a second sanctioned
  non-trace data path next to `fetchHead`, but a strictly simpler one
  (no engine round trip, no run state).
- **Reduction:** `vectors.bin` stores a PCA-64 projection, int8. 128 dims
  would double the file for a modest accuracy gain the sim-mode matrix
  does not need.
- **Asset is self-contained:** it carries the decoded text of every
  vocabulary id, so the geometry module does not depend on the app's
  tokenizer (which only encodes) and nothing is threaded through the
  view tree.

## Component 1: the Hub geometry asset

Folder `geometry/` in `saigyo-hoshi/smollm2-135m-attn-onnx`, fetched from
the same `resolve/main/` base URL the model uses, so ordinary browser
caching applies.

| File | Content | Approx. size |
|---|---|---|
| `manifest.json` | `modelId`, `vocabSize` (49152), `dims` (576), `k` (12), `pcaDims` (64), `scale` (int8 → float), `explainedVariance` (fraction, 0..1), `sourceSha256` (of the stock `embed_tokens.weight` tensor), `files` (name → byte length) | < 1 KB |
| `neighbors.bin` | two blocks: first an ids block of `vocabSize × k` little-endian uint16 (token `t`'s neighbours at `[t*k, t*k+k)`), then a sims block of `vocabSize × k` uint8 (`round(max(cos, 0) × 255)`), same indexing. Per token sorted by similarity descending. Computed **exactly** in the full 576-dim space; the token itself excluded. | ~1.8 MB |
| `vectors.bin` | per token, in id order: `pcaDims` int8 values; `float = int8 × scale`. PCA fitted on the full matrix, mean-centred; the mean is not stored (cosine on centred vectors is what we want). | ~3.1 MB |
| `tokens.json` | JSON array of `vocabSize` strings: the decoded text of each id via the model's tokenizer (`decode([id])`), same convention the app's tokenizer uses for `TokenInfo.text`. | ~500 KB (compresses well) |

The app shows 8 neighbours; storing 12 leaves room to skip entries whose
text is unrenderable (byte-fallback tokens `<0xNN>`, empty strings,
control characters) without leaving gaps.

## Component 2: export tool changes (`tools/export`)

Same CLI, `uv run python -m tsumugi_export <command>`:

- **New `geometry` command** (`geometry.py`): loads the stock
  `HuggingFaceTB/SmolLM2-135M-Instruct` weights, takes
  `model.embed_tokens.weight` in fp32, and writes the four files above to
  `out/model/geometry/`. Neighbours are computed exactly by blocked
  matrix multiplication over L2-normalised rows (49152² cosines in fp32
  is ~10 GB of intermediate if done at once; block by 2048 rows). PCA via
  `numpy.linalg.svd` on the centred matrix (or `torch.pca_lowrank`
  with `q = 64`); record explained variance. int8 scale =
  `max(|projected|) / 127`. Records each file's sha256 into
  `validation-report.json` under `artifacts`, like the model files.
- **`export` adds one output.** `AttnLlamaOnnxConfig.outputs` gains
  `inputs_embeds` with dynamic axes `{0: "batch_size", 1: "sequence_length"}`
  and static last dim 576: the result of the embedding lookup for the
  input ids of that forward call (with KV cache, the new tokens only).
  The output name is chosen to match the HF convention for the tensor.
  Cost per step is `seq × 576` floats — negligible next to attention.
  The q8 quantized variant keeps this output in float (it is a gather of
  the quantized table, dequantised; parity check below defines the
  tolerance).
- **`validate` gains two checks**, both blocking:
  - `inputs_embeds_parity`: for a fixed prompt, `inputs_embeds` from the
    exported graph equals `embed_tokens.weight[input_ids]` from the stock
    model within `atol 2e-2` for the quantized file (fp16: `1e-3`).
  - `geometry_spotcheck`: for 32 fixed-seed random ids, recompute exact
    top-12 neighbours from the stock matrix and assert the ids match the
    table (order may differ only among ties, i.e. equal uint8 sims), and
    that `tokens.json[id] == tokenizer.decode([id])`.
- **`publish` uploads `geometry/`** with the rest of the folder. Its
  existing hash gate covers the new artifacts because `geometry` records
  them. Republishing changes the model file browsers have cached; they
  re-download once. Accepted.
- **Docs:** `tools/export/README` (or the module docstring that serves as
  one) gets the new command and the asset layout table.

## Component 3: trace and engines

`embed` event, replacing the fake preview:

```ts
| { type: 'embed'; cycle: number; seqLen: number; dims: number;
    source: 'model' | 'asset';
    rows?: number[][] }
```

- `source: 'model'` — real mode with the new export. `rows` holds the
  `inputs_embeds` rows for the tokens **fed this cycle**: the whole prompt
  at cycle 0, one row per cycle afterwards (KV-cached decoding feeds one
  token). Values rounded to 3 decimals. Budget: a 30-token prompt and 50
  cycles add ~80 × 576 values ≈ 250 KB of JSON — within the "few hundred
  KB per run" budget the v1 spec set; the archive ring of 8 unpinned runs
  stays under a few MB.
- `source: 'asset'` — sim mode, and real mode when the loaded model lacks
  `inputs_embeds` (old cached export) or returns it with an unexpected
  shape. No `rows`; the card resolves vectors from the asset by id.
- **Validator** (`src/trace/validate.ts`): the phase machine is unchanged;
  `rows`, when present, must be an array of `dims`-length numeric arrays.
  Legacy archived events carrying `preview` and no `source` remain valid
  and are treated as `source: 'asset'`. `preview` is removed from the
  type; the validator ignores it.
- **Worker** (`src/engine/transformers/worker.ts`): today `embed` and
  the `layer` events are emitted *before* the forward call (they carried
  no measured data). They move to *after* it, still first in the cycle
  and in the same order, so the trace's phase order is unchanged; only
  the live-mode timing shifts slightly. After the forward, if
  `out.inputs_embeds` exists with dims `[1, fed, 576]`, its rows are
  emitted with `source: 'model'`; if the output is absent, the event is
  `source: 'asset'`. A present-but-wrong-shape output flips a per-run
  flag, logs once, and the rest of the run uses `source: 'asset'` — the
  never-fail policy from attention.
- **SimulatedEngine**: emits `{ type: 'embed', cycle, seqLen, dims: 576,
  source: 'asset' }`. No fetch, no vectors: sim mode's "instant" promise
  holds.
- **Fixtures** (`src/test/fixtures.ts`) and existing tests that build
  `embed` events are updated to the new shape.

## Component 4: geometry modules (`src/geometry/`)

- `asset.ts` — `loadGeometry(): Promise<GeometryAsset>`. Fetches the four
  files (manifest first, to learn byte lengths and validate `modelId`),
  parses the binaries into typed arrays, and exposes:
  - `neighbors(id): Array<{ id: number; sim: number }>` (k entries, sim
    as 0..1 float)
  - `vector(id): Float32Array` (pcaDims, dequantised)
  - `text(id): string`
  - `manifest`
  One module-level promise so the fetch happens once per page load;
  rejection is cached too and cleared by an explicit `retryGeometry()`.
  Base URL is a constant next to `ATTN_MODEL_ID` in `engine/tokenizer.ts`
  (`GEOMETRY_BASE_URL`), overridable by `VITE_GEOMETRY_BASE_URL` for tests
  and local files.
- `math.ts` — pure functions: `cosine(a, b)`, `similarityMatrix(vectors)`
  (returns `number[][]`, symmetric, diagonal 1), `renderableNeighbors(asset,
  id, n)` (filters unrenderable texts, returns first `n` of the stored
  `k`), `isRenderableToken(text)`.
- `useGeometry.ts` — hook returning
  `{ status: 'idle' | 'loading' | 'ready' | 'error'; asset?: GeometryAsset; retry(): void }`.
  Starts loading on first call (i.e. first time the Embeddings card
  mounts).
- `src/viz/selectors.ts` gains
  `embeddingRows(events, cursor): { ids: number[]; rows?: number[][]; source: 'model' | 'asset' }`
  — the ids of the visible sequence (prompt plus generated up to the
  cursor's cycle, from `visibleTokens`) and, when every visible position
  has a `model` row available from `embed` events up to the cursor, those
  rows in position order; otherwise `rows` is undefined and `source` is
  `'asset'`. Mixed runs (early cycles `model`, later `asset` after a
  fallback) therefore degrade wholesale to asset for consistency.

## Component 5: the card

`src/viz/details/EmbeddingsDetail.tsx` becomes a thin container that
reads `embeddingRows`, `useGeometry`, and owns one piece of state: the
selected position (default: the newest visible token; snaps to the newest
if the selection is beyond the visible sequence after scrubbing back).
Two child components:

```
┌ Embeddings ───────────────────────────────────────────────────────────┐
│ A lookup, not a computation. Each token id selects one row of a       │
│ learned matrix E [49152 × 576]; the rows stacked up are x [n × 576].  │
│                                                                       │
│  ids            E [49152 × 576]                  x [n × 576]          │
│  [504]  ─┐    ┌───────────────┐                                       │
│  [ 3459] ─┼──▶ │ ░░░░░░░░░░░░░ │ row 3459  ─────▶ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒   │
│  [ 1234] ─┘    │ ...           │                  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒   │
│                └───────────────┘                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ◀ new │
│                                                                       │
│  ⓘ learned, not designed   ⓘ no position here (RoPE)   ⓘ tied with Logits │
│                                                                       │
│ ── Geometry: meaning is distance ──────────────────────────────────── │
│  Selected: [ cat]      (click any token above to change)              │
│  Nearest in E:   cats  ████████ 0.71    kitten ██████ 0.62  …  (8)    │
│                                                                       │
│  Prompt tokens vs. each other (cosine, before any context):           │
│       The  cat  sat  on  the  mat                                     │
│  The  ■    ░    ░    ░    ▓    ░                                      │
│  cat  ░    ■    ░    ░    ░    ▒                                      │
│  …                                                                    │
│  caption: provenance line, see below                                  │
└───────────────────────────────────────────────────────────────────────┘
```

**`EmbeddingLookup.tsx` (mechanism).** Left column: the visible sequence
as token chips with their ids; clicking selects. Middle: the matrix `E`
drawn as a tall rectangle labelled `E [49152 × 576]` with the selected
id's arrow into it and the selected row drawn as a strip of cells
(576 values in `model` source, downsampled by mean-pooling to 96 cells;
64 values in `asset` source, one cell each), diverging colour scale as
today (blue positive, red negative). Right: the stacked `x [n × 576]`
matrix as thin rows, newest row highlighted; each cycle appends one.
Three hover callouts using the `rs-hover`/`<title>` pattern from
`ResidualStreamDiagram`:

- *learned, not designed* — "Every row starts random and is adjusted
  during training so that the rest of the network predicts well. Nobody
  chose what dimension 17 means."
- *no position here* — "SmolLM2 adds no position vector at this stage.
  Position enters inside attention, via rotary position embeddings
  applied to queries and keys."
- *tied with Logits* — "The same 49152 × 576 matrix is reused at the end:
  the final vector is compared against every row to score each token
  (tie_word_embeddings)."

**`EmbeddingGeometry.tsx`.** Neighbour list: 8 entries for the selected
token, text plus horizontal similarity bar plus value to 2 decimals.
Similarity matrix: the visible sequence against itself, capped at the
**last 24 tokens** with a "showing the last 24 of n" note; SVG heatmap
using the attention heatmap's colour ramp and label style, cells have
`<title>` tooltips with the pair and value. Vectors: `model` rows when
`embeddingRows.source === 'model'`, else asset PCA-64 vectors. Provenance
caption:

- model: "Exact rows from the running model."
- asset: "Real SmolLM2 embedding rows, reduced to 64 dimensions offline;
  similarities are approximate."

Both components get `data-testid` hooks (`embed-lookup`, `embed-geometry`,
`embed-neighbors`, `embed-similarity`) for tests and screenshots.

**Pipeline band** summary for the stage becomes `49152 × 576` (from
`run-start.vocabSize` and `embed.dims`); falls back to `576 dims` when
`vocabSize` is absent.

**Styling** follows the existing card CSS in `src/index.css`; no new
design language.

## Error handling

- Geometry fetch fails, or `manifest.modelId` ≠ the run's `modelId`
  (from `run-start`): `EmbeddingGeometry` renders one line, "Vocabulary
  geometry couldn't be loaded", with a "retry" button calling `retry()`.
  The mechanism section still renders: fully in `model` source; in
  `asset` source the strip shows a shape-only placeholder with the note
  "vector values unavailable offline".
- Worker: `inputs_embeds` missing → `source: 'asset'` silently (it is the
  expected state for an old cached model). Present with wrong shape →
  flag, one `console.warn`, `source: 'asset'` for the rest of the run.
- Selection beyond the visible sequence after scrubbing back snaps to
  the newest visible token.
- Fallback hash tokenizer active (Hub unreachable at load): token ids are
  meaningless for geometry, but the same offline condition makes the
  geometry fetch fail, so the load-error line covers it. Accepted
  limitation, noted here rather than detected.
- Legacy archived runs (`preview`, no `source`) render as `asset` source.

## Testing

Python (`tools/export/tests`):
- `test_geometry.py`: on a synthetic 50 × 8 matrix — neighbours exact and
  self-excluded, sims sorted descending, uint8 quantisation monotone,
  int8 round trip within `scale`, PCA dims and explained variance in
  `(0, 1]`, `tokens.json` length equals `vocabSize`, manifest byte lengths
  match files.
- `test_onnx_config.py`: `inputs_embeds` present in `outputs` with the
  stated axes.
- `test_publish_gate.py`: geometry artifact hashes participate in the
  gate.

TypeScript (vitest):
- `src/geometry/asset.test.ts`: parser on a hand-built 5-token fixture in
  the exact binary layout (served via a mocked `fetch`); manifest model
  mismatch rejects; retry after failure re-fetches.
- `src/geometry/math.test.ts`: cosine, symmetric matrix with unit
  diagonal, neighbour filtering of `<0xNN>` and empty texts.
- `src/viz/selectors.test.ts`: `embeddingRows` across cycles, cursor
  before/after the cycle's `embed`, mixed-source degradation.
- `src/trace/validate.test.ts`: new shape valid; legacy `preview` events
  valid; `rows` with wrong dims invalid.
- `SimulatedEngine.test.ts`: `embed` has `source: 'asset'` and no `rows`
  (replaces the "capped at 4 × 16" test).
- Worker: a unit test around the row extraction (`inputs_embeds` → rows,
  bad shape → asset) factored into a small pure helper in
  `src/engine/transformers/embedRows.ts`.
- Components: `EmbeddingsDetail` in `model` and `asset` source, loading,
  and error states; selection click; 24-token cap note.

Playwright: `scripts/capture-screenshots.mjs` gains an `embeddings` scene
(real mode, after the run seals, Embeddings stage selected, a prompt with
a repeated word so the matrix has visible structure).

## Delivery order

1. **Export tool**: `geometry` command, `inputs_embeds` output, validate
   checks, publish. Python only. The app keeps working against the new
   model file because it ignores unknown outputs. Publishing needs the
   local HF token: the executor stops and hands the publish command to
   the human.
2. **Trace and engines**: new `embed` shape, worker rows, sim `asset`
   source, validator leniency, fixtures.
3. **Geometry modules and the card**: mechanism first, then geometry.
4. **Screenshots, README section, backlog**: a new backlog item for M4
   is added and checked off; a follow-up item is added for "position:
   show RoPE inside the attention detail" if the callout leaves learners
   wanting more.

## Out of scope

- 2D vocabulary map, analogy arithmetic (decided against, see above).
- Per-layer hidden states / residual-stream evolution (would need further
  graph outputs; a separate spec).
- Showing the unembedding side by side at the Logits stage (the callout
  points there; a Logits-card enhancement is its own item).
- Chat templating (#8), mobile layout (#9) — unchanged.
