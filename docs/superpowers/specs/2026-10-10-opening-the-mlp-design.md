# Opening the MLP — design

**Date:** 2026-10-10
**Status:** Approved design, pre-implementation
**Backlog:** #15 (builds on #14, the in-app glossary)
**Mockup (invented numbers):** https://claude.ai/artifact/GYLgxJPsnVC3Si8MKiVSAR

## Purpose

The Layers detail opened up attention (heatmaps, head explorer) but the MLP
is still a box with a label. This work adds two cards that show what the
MLP blocks do in a real run:

- **Edit size** — how much each block changes the residual stream, per
  token and layer, for MLP and attention side by side.
- **Per-layer votes** — how much each layer's MLP added to or took from
  each candidate's score, adding up exactly to the score on the Logits
  card.

## Decisions already made

- **Simulated mode shows a note**, not invented numbers: "Real model only:
  simulated runs have no MLP to measure." Invented edit sizes and votes
  would teach nothing or the wrong thing.
- **Exact sum, attention included.** The final RMSNorm is frozen at its
  real scale for the last position, which makes the score split exactly
  into embedding + attention edits + MLP edits. No remainder row.
- **Both cards always visible**, below the attention section of the
  Layers detail, edit size first.
- **Data per cycle.** The model runs with a KV cache: cycle 0 processes
  the prompt, every later cycle one new token. Edit sizes therefore arrive
  as new rows per cycle and the card stacks them.

## Terminology

- **Edit** — the vector a block (attention or MLP) adds to the residual
  stream; the app's term, already in the glossary (papers: block output,
  residual update).
- **Edit ratio** — ‖edit‖ / ‖stream the edit is added to‖, for one
  position, layer and block.
- **Stream part** — one of the 61 summands of the final vector at the last
  position (embedding, 30 attention edits, 30 MLP edits), each scaled by
  the final norm's gain and divided by its root mean square there.
- **Vote** — a stream part dotted with a candidate's row of E; the votes
  of one candidate add up to its logit.

## The maths

SmolLM2 ends with `logits = norm(x) · Eᵀ`, where for the last position
`norm(x) = g ⊙ x / rms(x)` with a fixed gain vector `g`. The stream is a
sum, `x = e + Σ_l a_l + Σ_l m_l` (embedding, attention edits, MLP edits).
Holding `rms(x)` at its real value for this position makes `norm` linear,
so

```
logit(c) = (g ⊙ e / rms(x)) · E[c] + Σ_l (g ⊙ a_l / rms(x)) · E[c] + Σ_l (g ⊙ m_l / rms(x)) · E[c]
```

exactly. This is not the classic "logit lens", which re-applies the norm
to each partial sum and so does not add up; the glossary entry says so.

## Export v4

New outputs, added by ONNX graph post-processing in `tools/export` like
the v3 readout (`expose_readout` in `postprocess.py`), by locating each
layer's two residual Add nodes and the final norm:

| Output | Shape | Content |
|---|---|---|
| `edit_ratios` | [1, fed, 30, 2] | per fed position and layer: ‖a_l‖/‖x before attention‖ (index 0), ‖m_l‖/‖x before MLP‖ (index 1) |
| `stream_parts` | [1, 61, 576] | last fed position: row 0 embedding, rows 1–30 attention edits, rows 31–60 MLP edits, each `g ⊙ part / rms(x_final)` |

`fed` is the number of tokens fed this cycle. The rows of `stream_parts`
add up to the last row of `last_hidden_state` (the norm's output, already
exported in v3).

**Risk to check first:** the v3 work found that the exported graph does
not always have the shape the PyTorch code suggests (the readout tensor
turned out to be the norm's input). The plan starts with a short probe of
the exported graph that locates the residual Adds and the final norm
before any post-processing is written. If optimum or quantization fuses
the residual Adds into `SkipSimplifiedLayerNormalization` nodes, the
outputs are taken before fusion (post-process the fp32 export, then
quantize) — the probe decides.

**Validator** (new checks, all must PASS before `publish`):

- `edit-ratios-parity` — matches ratios computed with PyTorch forward
  hooks on the stock model, for the cached and no-cache variants.
- `stream-parts-sum` — the parts add up to `last_hidden_state`'s last row.
- `votes-sum` — parts · E[c] add up to the logits for the top-k ids.

**Release:** Hub tag `v4`, then `ATTN_MODEL_REVISION = 'v4'` in
`src/engine/tokenizer.ts`, following the manual checklist in
`tools/export/README.md` (updated with the new checks and tag).

## Worker

Two pure functions beside `readout.ts`, never-fail like the readout:
absent output → no data, wrong shape → a console warning, the feature off
for the rest of the run, and the run continues.

- `extractEdits(out, fed, layers)` → `{ status: 'absent' } | { status:
  'bad-shape', detail } | { status: 'ok', attn: number[][], mlp: number[][] }`
  (rows = fed positions, columns = layers, 3 decimals).
- `computeVotes(out, topkEmbeds, layers)` → per top-k candidate id
  `{ embed: number; attn: number[]; mlp: number[] }` (3 decimals). Uses the
  `topk_embeds` rows already read by `extractReadout`.

The `edits` event is emitted after the cycle's `layer` events (and after
`attention`, if present), before `logits`; votes ride on the logits
event's candidates.

## Trace format

Both additions are optional, so older traces stay valid:

```ts
| { type: 'edits'; cycle: number; start: number; attn: number[][]; mlp: number[][] }

interface LogitCandidate extends TokenInfo {
  // … existing logit, row, dot
  votes?: { embed: number; attn: number[]; mlp: number[] }
}
```

- `start` is the position of the first row (0 at cycle 0, then the new
  token's position); `attn` and `mlp` are [rows × layers].
- `validateTrace`: `edits` sits in the same optional slot as `attention`;
  rows of equal length (= layers), finite numbers, `start` + rows ≤
  `seqLen`; `votes` arrays have length = layers, finite numbers.
- Size: a 30-token, 20-cycle run adds about 20 KB.

## UI

Both cards show the cycle the player is viewing. They live in
`src/viz/`, are rendered by `LayersDetail` below the attention section
and the head-grid button, and get their data from `DetailPanel` (edits
rows stacked from all `edits` events up to the cursor; votes from the
viewed cycle's `logits` event). The logits event comes after the layers
in the trace, so on the Layers stage the votes card looks ahead to it in
the full event list; while a live run has not emitted it yet, the card
shows "Votes appear once this cycle's scores are computed."

### Card A — "How much each layer changes the vector" (`EditSizeCard`)

- Heading with ⓘ (`Term` icon) → `edit`.
- Caption with term links to *edit* and *residual stream*; formula
  ‖Δx‖ / ‖x‖ as MathML via `<Formula>`.
- Two heatmaps side by side, MLP and attention: rows = tokens (labels
  from the visible tokens), columns = L0–L29, one shared colour scale
  0 → ≥ 1.0 (values above 1 clamp to the top colour). Cell tooltip:
  token, layer, block, value.
- Click a token row to select it; paired bars per layer (attention, MLP)
  for the selected token below. Default: the last token. Selection resets
  when it falls outside the visible rows.
- Three glossary chips: "why layer 0 is so dark", "quiet middle, loud
  end", "attention vs MLP" → `edit` § `layer-0`, `quiet-middle`,
  `attn-vs-mlp`. Wording describes typical patterns, not this run.

### Card B — "What each layer voted for" (`LayerVotesCard`)

- Heading with ⓘ → `votes`.
- Caption with the exact identity as MathML (score = embedding · E +
  Σ attention edits · E + Σ MLP edits · E, each part scaled by the final
  norm) and a term link to *logit*.
- Grid: candidates × 30 MLP layers, diverging scale (blue adds, red
  subtracts), symmetric around 0, scaled by the largest absolute MLP vote
  in the cycle. Hover: "MLP 22 → "barn": +1.84 to the score".
- Candidate selection shared with the Logits card: selecting on either
  card selects on the other. Default: the top-scoring candidate (index 0),
  as on the Logits card today — not the sampled token, which the player
  has not reached yet on the Layers stage. The selection moves from
  `LogitsDetail`'s local state into a small zustand store keyed by cycle
  (`src/viz/candidateSelection.ts`).
- Running total: line chart of the selected candidate's score building up
  across layers — embedding first, then per layer its attention vote then
  its MLP vote — other candidates as faint lines. Ends exactly on the
  logit.
- Breakdown: embedding, all attention blocks, all MLP blocks, then the
  three largest MLP layers (by absolute vote) indented; total = the
  Logits card's score.

### Notes instead of a card

- Simulated mode: "Real model only: simulated runs have no MLP to
  measure."
- Real run without the data (v3 model, a bad-shape output, or a trace
  recorded before this feature): "This run was recorded before MLP data
  was exported."

Each card shows its own note, so a run with edits but no votes (or the
reverse) still shows what it has.

## Glossary

- **New entry `votes` ("Per-layer votes")**, stage `layers`: the frozen
  norm and why the sum is exact, how to read the running total, the
  difference from the classic logit lens. "In this run": the MLP layer
  that added most to the top candidate's score at the viewed cycle.
  Related: `logit`, `mlp`, `edit`. Added to `GLOSSARY_IDS`; `RunSummary`
  gains what the block needs.
- **`edit`**: sections `layer-0`, `quiet-middle`, `attn-vs-mlp`; "In this
  run": the largest MLP edit ratio for the last token.
- **`mlp`**: a closing paragraph pointing to both cards.

## Testing

- **Export (Python, `tools/export/tests`, not run in CI):**
  post-processing adds both outputs with the right shapes on a small
  graph; validator checks pass and fail on constructed inputs.
- **Worker:** `extractEdits` and `computeVotes` for absent, bad-shape and
  ok inputs; votes add up to the logit within rounding.
- **Trace:** `validateTrace` accepts `edits` and `votes`, rejects bad
  shapes, and existing fixtures (older traces) still pass.
- **Cards:** edit heatmap stacks rows across cycles; token selection and
  default; vote colours and scale; shared candidate selection with the
  Logits card; running total ends on the logit; breakdown total; both
  notes.
- **Glossary:** registry and section tests cover `votes` and the new chip
  targets.
- **E2E (sim, CI):** the Layers detail shows the "real model only" notes.
- **Manual after publishing v4:** a real run in the browser; the
  breakdown total equals the Logits card's score.

## Docs

- README: the Layers detail section describes both cards; new real-mode
  screenshot scene `mlp.png` in `scripts/capture-screenshots.mjs`.
- `tools/export/README.md`: new checks, tag `v4`.
- `docs/BACKLOG.md`: tick #15; add the follow-ups below as new items.

## Out of scope

- Neurons labelled by the tokens their down-projection promotes.
- A SwiGLU anatomy detail view.
- MLP knockout runs opened in the Compare view.
- Clicking a layer column to jump to attention (no per-layer attention
  view exists).
- Votes for attention as grid rows (attention appears in the running
  total and the breakdown only).
