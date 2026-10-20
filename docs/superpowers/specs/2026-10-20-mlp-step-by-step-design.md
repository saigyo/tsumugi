# The MLP, step by step — design

**Date:** 2026-10-20
**Status:** Approved design, pre-implementation
**Backlog:** #17 (SwiGLU anatomy view), #19 (faithful browser model), and the
"model tag in the neurons manifest" item from #20. Builds on #15 and #16.
**Mockups (real fp32 values):**
- the card: `.superpowers/brainstorm/67669-1792530069/content/anatomy-v2.html`
- the navigation bar, option A: `.superpowers/brainstorm/67669-1792530069/content/navigation.html`

## Purpose

The glossary explains what an MLP computes; the "Inside one MLP" card splits
one layer's vote into neurons. This work shows the computation itself: a
fourth card walks through one MLP's five steps (norm, gate and up, SiLU,
product, down) with the real numbers for the last token. It ships with a
more faithful browser model, so those numbers are what the model computed,
and with a navigation bar, because the Layers detail becomes very long.

## Findings

**From the probe** (fp32 SmolLM2-135M-Instruct, prompt "The cat sat on the
mat because it was tired", last token "␣tired"):

- Layer 29: 885 of 1,536 gates are negative and land between −0.28 and 0
  after SiLU; 651 are positive and pass almost unchanged. The neurons with
  the largest products have gates of 9–12 (SiLU ≈ identity there) and up
  values of −2.7 to +2.3: the gate sets the size, the up value mostly the
  sign. 9 neurons carry half of the squared product, 44 carry 90%.
- Early layers are spread out: layer 5 has 1,292 negative gates and needs
  111 neurons for half; layer 11 needs 59.
- ‖edit‖ at layer 29 is 405, 0.57 of the stream's length.

**From the #19 spike** (throwaway scripts; four prompts, last-position logits
against the fp32 export):

| variant | size | same argmax, browser semantics | max \|Δlogit\| |
|---|---|---|---|
| published q8 (dynamic int8 activations) | 250 MB | 1–2 of 4 | up to 11 |
| weight-only 8-bit, block 32, symmetric | 347 MB | 4 of 4 | 0.6 |
| same, plus uint8 embedding Gather | **263 MB** | **4 of 4** | **1.0** |
| weight-only 8-bit, block 128 | 347 MB | 3 of 4 | 1.0 |
| weight-only 4-bit, block 32 | 294 MB | 1 of 4 | 8.4 |

"Browser semantics" means onnxruntime with graph optimizations disabled
(Python), onnxruntime-web 1.26 WASM in Node, and onnxruntime-web WebGPU in
Chromium (Apple Metal). All three agreed for the chosen variant. On WebGPU it
runs about 30 ms per full-prompt pass against q8's 100 ms; on WASM about
150 ms against 50 ms. Other GPU vendors are untested.

## Decisions already made

- **One model version, v6**, carrying the faithful model, the new outputs
  and the model tag. The feasibility of the model was checked first (above).
- **Model:** weight-only 8-bit (`MatMulNBits`, bits 8, block 32, symmetric)
  with the same exclusions as today (tied lm_head and the readout gather stay
  fp32), plus the embedding Gather quantized to uint8 as q8 does today:
  263 MB. Rejected: the 347 MB variant with an fp32 embedding table (more
  exact, 85 MB more download), block 128, 4-bit, fp16 (garbage on WebGPU,
  see `quantize.py`).
- **The card:** as in the approved mockup — five steps, top to bottom, each
  a label column with the formula and one sentence, and a chart.
- **Navigation:** option A, a sticky bar at the top of the Layers detail with
  six section links and the shared MLP layer stepper.
- **Charts use the available width.** Every chart sizes itself from its
  container's width (as measured, not a fixed pixel value); the built card
  is compared with the mockup side by side before it counts as done.

## 1. The card: "The MLP, step by step"

**Placement:** fourth card in the Layers detail, below "Inside one MLP".

**Header:** "The MLP, step by step: layer N", a ◀ ▶ stepper and an ⓘ link to
the glossary's MLP entry. One line under it: "What layer N's MLP computes for
the last token, in the order it computes it. Same layer as the cards above;
● marks the neurons listed in "Inside one MLP"."

**Which layer.** The same layer as "Inside one MLP": the shared pick, or by
default the layer with the largest MLP vote for the selected candidate. The
rule moves into one shared helper used by the neuron card, this card and
the navigation bar, so all three always agree.

**Marked neurons.** The 8 neurons "Inside one MLP" currently lists (its mode
and selected candidate), drawn in the accent orange in every chart. Others
are grey dots or signed bars.

**Each step** is a row: a label column (step name, formula, one sentence;
fixed width, about 170 px) and a chart that takes the rest of the card's
width. Texts as in the mockup.

1. **Norm.** Three heat strips, 576 cells each, full chart width: x (the
   vector into the MLP), the norm's learned weights g with "÷ rms(x) = …"
   in the label, and n. Diverging blue/red, clipped at 2× the vector's
   typical size (its root mean square), noted under the strips.
2. **Gate and up.** Scatter, one dot per neuron: gate across, up upward,
   axes with ticks, the gate < 0 half shaded with the note "gate < 0: SiLU
   will squash these". The three largest marked neurons carry their labels.
3. **SiLU.** The SiLU curve on the **same gate axis** as step 2 (identical
   x scale and width, aligned), every neuron a dot on the curve, with the
   counts "N negative gates → between −0.28 and 0" and "N positive gates
   pass almost unchanged".
4. **Product.** The 120 largest products by size, signed bars, largest
   first; dashed lines at the neuron counts carrying half and 90% of the
   squared product; footnote "the other 1 416 are all below X".
5. **Down.** Strips of the three largest marked neurons' contributions
   (product × that neuron's column of W_down), "+ 1 533 more", and the
   edit itself, all on one colour scale; under it "‖edit‖ = …, … of the
   stream's length (the edit size card's cell)".

**Hover.** Any dot or bar for a neuron shows a box: "N1233 · gate 11.49 →
SiLU 11.49 · × up −2.46 = product −28.25 · pushes toward birds, animals,
Laura…" (labels from the neuron asset). Keyboard: the marked neurons are
focusable and show the same box.

**Three callouts** to new glossary sections: "why SiLU and not a switch",
"why two projections", "early layers look different" (with the layer 5 /
11 / 29 comparison from the findings, phrased generally).

**States.**
- Simulated run: "Real model only: simulated runs don't compute an MLP."
- Run recorded before v6 (no step data): "This run was recorded before the
  step-by-step data existed. Generate it again to see it."
- Pending cycle: the same pending note as the other MLP cards.
- Layer file still loading or failed (step 1's g and step 5's columns come
  from it): the loading line and the retry button the neuron card uses;
  steps 2–4 render without waiting.

## 2. The navigation bar

**Where:** directly under the "Transformer layers" heading, inside the
Layers detail. `position: sticky; top: 0` while the detail is on screen;
it scrolls away with the detail.

**Contents:** "Jump to" and six links — Overview, Attention, Edit sizes,
Votes, Inside one MLP, Step by step — each pointing at its section's
heading. Right-aligned: "MLP layer N ◀ ▶", the shared layer stepper.

**Behaviour.**
- A click scrolls smoothly to the section, leaving the heading just below
  the bar (scroll margin = bar height).
- The link of the section on screen is highlighted (IntersectionObserver;
  the topmost section whose heading has passed under the bar wins).
- In simulated runs, and in runs without MLP data, the layer stepper is
  hidden; the links stay (those sections show their notes).
- Narrow windows: the links scroll sideways in one row; the stepper stays
  visible.
- Sections without a heading of their own (Overview, Attention) get an
  anchor on their first element.
- Accessibility: a `nav` landmark labelled "Layers sections", the current
  link marked `aria-current`.

## 3. Export v6

**Model.** `quantize.py` produces the variant chosen above as
`model_quantized.onnx` (the file the app loads with `dtype: 'q8'`; the name
stays so the worker's load path does not change). Its docstring records the
spike results next to the 2026-08-28 campaign.

**New outputs**, last position only, all layers:
- `mlp_gate` [batch, layers, neurons]: the gate projection before SiLU.
- `mlp_up` [batch, layers, neurons]: the up projection.

Everything else the card needs is already exported: the products are
`mlp_activations`; `stream_parts` holds the embedding, then the 30
attention edits, then the 30 MLP edits at the last position, so x for layer
l is the embedding plus every attention and MLP edit of layers before l plus
layer l's attention edit, and the MLP edit is layer l's MLP part.

**Neuron asset.**
- Each layer file gains that layer's post-attention norm weight (576
  float32) at a new offset in the layout (`normWeight`).
- The manifest gains `"revision": "v6"`, the model tag it was built for
  (the #20 item). The app refuses a manifest whose revision differs from
  `ATTN_MODEL_REVISION` (error state with retry, cached files evicted).
- `rms_norm_eps` goes into the manifest, so the browser computes n exactly.

**Validator.**
- The quantized variant's checks run **with graph optimizations disabled**,
  so the gate sees what browsers compute. On the two validation prompts plus
  "The capital of France is" and "Once upon a time, there was a little":
  same argmax as fp32 on all, max |Δlogit| ≤ 1.5, and the 8-step greedy
  check against the stock export as today.
- `mlp_gate` and `mlp_up` match PyTorch hooks; SiLU(gate) · up reproduces
  `mlp_activations`.
- Neuron votes add up to the MLP votes for every layer under browser
  semantics (today 7 of 30 layers show "≈" at the first generated token;
  v6 should show none).
- Layer files' `normWeight` matches the model; the manifest's revision
  matches the `--tag` given to `publish`.

**Before publishing:** load the new `model_quantized.onnx` through
transformers.js 4.2 in a local browser page (WebGPU and WASM), from a local
copy of the repo layout, to confirm transformers.js accepts the
`MatMulNBits` graph under the q8 file name.

**Publish:** tag v6, then bump `ATTN_MODEL_REVISION` to `'v6'` and deploy,
as the export README describes.

## 4. Run data and the browser

**Neuron data on the `edits` event** gains optional fields (float16 base64,
layer-major, like `activations`):
- `gate` [layers × neurons], `up` [layers × neurons];
- `mlpIn` [layers × dims] (x per layer) and `mlpOut` [layers × dims] (the
  MLP edit per layer), computed in the worker from `stream_parts`.

Cost: about 250 KB of float16 per generated token, about 340 KB as base64
in a stored run, on top of about 150 KB today. Runs without these fields show the "recorded
before" note; the trace validator accepts both.

**Worker.** Extends `extractNeurons` with the new outputs; a bad shape drops
only the step-by-step fields (logged once), the neuron card still works.

**Math module** (pure, main thread), per layer: n from x, g and eps; SiLU;
the sorted products with the half / 90% counts; the three contributions
from the int8 columns; the hover values. Charts get plain arrays.

**Chart sizing.** A shared `useWidth` hook (ResizeObserver on the chart's
container) feeds each chart its width; the norm strips' cell width, the
scatter and SiLU plots' x scale, the product bars and the down strips all
derive from it. Heights are fixed per chart, as in the mockup.

## 5. Docs

- Glossary: the MLP entry links to the card; the three callout sections.
- README: the card, the navigation bar, the model change.
- Backlog: #17 and #19 done; the model-tag item removed from #20; the
  q8 note in the worker and in #16's "≈" README section updated.

## Testing

**Export (pytest):** the quantize step produces `MatMulNBits` nodes with
the exclusions; the new outputs' shapes; the layer layout with
`normWeight`; the manifest's revision and eps; quantized sessions are created with
graph optimizations disabled. The real gate (the published v5 file fails,
the new variant passes) is the operator-run `validate` step, as today.

**App (vitest):**
- Math module on a hand-built layer: n, SiLU, sorted products and counts,
  contributions, the 120 cut-off and its footnote value.
- Worker extraction: shapes, `mlpIn`/`mlpOut` from a small `stream_parts`,
  bad shapes drop only the new fields.
- The shared layer helper: pick, default, run change.
- Card: every state; the marked neurons follow the neuron card's mode and
  candidate; the stepper; hover box contents; charts take the width the
  `useWidth` hook reports (mocked).
- Navigation bar: links, highlight from a mocked IntersectionObserver,
  stepper shown only with MLP data, `aria-current`.
- Manifest revision mismatch → error state, files evicted.

**Browser test (Playwright):** a recorded v6 run with a fake asset via route
interception: the navigation bar sticks while scrolling and jumps to each
section; the stepper changes the layer in the neuron card and the new card;
at 1 280 px the scatter and SiLU plots are at least 700 px wide and share
their x positions.

**By hand after deploy:** a live run on the probe prompt shows layer 29's
counts (885 / 651, 9 / 44) within quantization tolerance; the neuron card
shows no "≈" at the first generated token; the built card and bar are
screenshotted next to the mockups and compared.

## Out of scope

- Steps for positions other than the last token.
- Knocking out neurons or layers (#18).
- Chat templating (#8) and the other #20 minors.
