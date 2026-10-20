# Backlog

Ideas and follow-ups collected across milestones (v1, M1 real attention,
M2 head explorer, M3 trace archive + comparison). Check items off as they
ship; add provenance when adding new ones. Deliberately lighter-weight
than GitHub issues.

Items carry permanent numbers for easy reference ("#2 next?"). A new item
takes the next free number; numbers are never reused or renumbered, so
they are IDs, not positions.

## Help and explanations

- [x] **#14 In-app glossary** — One glossary of
  concept entries (id, title, one-line summary, formatted body with
  formulas, links to related entries), reached two ways: underlined terms
  in captions show the summary on hover and open the full entry on click,
  and a "?" button opens the same glossary as a browsable index with a
  short "how to read this app" entry walking the pipeline stages. Entries
  open in a side drawer, not a centred modal, so the visualisation they
  describe stays visible. Replaces the `title`-tooltip ⓘ callouts, which
  don't work on touch or with the keyboard. Starter entries from what the
  README already explains: token, embedding, residual stream, edit (the
  vector a block adds to the stream; papers say block output / residual
  update), attention head and the detected roles, MLP, logit, softmax and
  temperature, top-k sampling, simulated vs real mode. Motivation: the
  app has outgrown "read the README first". Shipped 2026-10-10 (spec
  `docs/superpowers/specs/2026-10-10-in-app-glossary-design.md`). *(2026-10-10 discussion)*

## Comparison extensions (builds on M3)

- [ ] **#1 Grid side-by-side compare** — both runs' 270-head thumbnail grids
  next to each other: "do heads keep their jobs across runs/prompts?" at a
  glance. Data is already in the traces (`attention-grid` events); the cost
  is screen real estate, and it is real-mode only. *(M3 spec, out of scope)*
- [x] **#2 Probability-delta view** — computed diff at the selected cycle:
  which tokens gained/lost probability between the runs, sorted by delta.
  Most quantitative comparison panel; needs a small derived-data layer the
  current panels don't have. Teaches best on same-prompt pairs at
  different temperatures. *(M3 spec, out of scope)*
- [ ] **#3 >2-run comparison** — generalize A/B to N columns. High UI cost;
  most pedagogy lives in pairwise contrasts — deliberately parked.
  *(M3 spec, out of scope)*

## Attention / interpretability depth

- [ ] **#4 Per-cycle grid evolution** — make the head-explorer grid scrubbable
  so attention accumulates as the run progresses (watch the induction head
  "switch on" when the pattern repeats). Needs per-cycle thumbnails, which
  multiplies grid data by the cycle count — a real data-architecture
  decision, spec-worthy. *(M2 spec, out of scope)*
- [x] **#5 Coreference-head discovery** — real mode detects previous-token /
  sink / induction / distinctive, but the coreference role (poster child of
  the sim examples) had no real-mode detector. The spike
  (`docs/research/2026-09-03-coreference-head-spike.md`) found a clean one in
  SmolLM2-135M (L13H8) and an antecedent-blind template that ranks it first;
  shipped as a fourth showcase role, a pin label and a grid-explorer sort.
  *(M1/M2 backlog; done 2026-09-03)*
- [x] **#15 Opening the MLP** — Builds on #14: the cards use glossary terms.
  Two Layers-detail cards from one export release (v4): (a) **edit size**
  — per token and layer, ‖Δx‖/‖x‖ for the MLP and for attention, as a
  token × layer heatmap plus per-layer bars for the selected token
  (computed in-graph, T × 30 numbers per block); (b) **per-layer votes**
  — each layer's MLP output at the last position dotted with the top-k
  candidates' rows of E (`topk_embeds` from v3), as a candidate × layer
  grid (blue adds, red subtracts), a running-total line per candidate
  ("logit lens") and a breakdown summing to the Logits card's score with
  a final-norm remainder. Needs the per-layer MLP output at the last
  position (30 × 576). Release = new Hub tag + bump
  `ATTN_MODEL_REVISION`; sim mode needs fallbacks. Mockup with invented
  numbers: https://claude.ai/artifact/GYLgxJPsnVC3Si8MKiVSAR. Later
  follow-ups discussed: top firing neurons labelled by the tokens their
  down-projection promotes (offline asset), a SwiGLU anatomy detail view,
  and MLP knockout runs opened in the Compare view (now #16–#18).
  Shipped 2026-10-11 (spec
  `docs/superpowers/specs/2026-10-10-opening-the-mlp-design.md`); export
  v4. *(2026-10-10 discussion)*
- [x] **#16 Labelled MLP neurons** — top firing neurons per layer, each
  labelled by the tokens its down-projection column promotes most (an
  offline asset from the export tool). *(#15 follow-up)* Shipped as the
  "Inside one MLP" card (spec
  `docs/superpowers/specs/2026-10-14-labelled-mlp-neurons-design.md`);
  export v5 with the `neurons/` asset. Parked minors: #20.
- [ ] **#17 SwiGLU anatomy view** — a detail view of one MLP's steps
  (norm, gate, up, SiLU, product, down) with real values for the selected
  token. *(#15 follow-up)*
- [ ] **#18 MLP knockout runs** — re-run a prompt with one layer's MLP
  zeroed and open the pair in the Compare view. *(#15 follow-up)*

## Pipeline pedagogy

- [x] **#11 Embeddings explained** — the Embeddings card teaches the lookup
  (ids → row of E → stacked stream, with learned/no-position/tied-weights
  notes) and the geometry (nearest neighbours, self-similarity matrix).
  Real rows via an `inputs_embeds` export output; vocabulary geometry from
  a Hub-hosted static asset. *(M4, 2026-09-02)*
- [ ] **#12 Position inside attention** — the Embeddings card says "no
  position here (RoPE)"; show where rotary position embeddings act in the
  Layers detail so the pointer lands somewhere. *(M4 spec follow-up)*
- [x] **#13 Logits readout** — the Logits card shows the readout as the
  Embeddings lookup in reverse and, in real mode, the dot product itself:
  final vector, candidate row of E, element-wise product summing to the
  logit. Needed three new export outputs (`last_hidden_state`, `topk_ids`,
  `topk_embeds`) and a pinned Hub tag (`v3`). Softmax became its own
  pipeline stage with its own card (formula, derivation in words, per-bar
  tooltips) — before, the Logits card click always landed on the softmax
  event and the scores view was only reachable by stepping back.
  *(2026-10-09)*

## Real-mode quality

- [ ] **#8 Chat templating** — run prompts through SmolLM2-Instruct's actual
  chat template so the model sees its expected format; should noticeably
  improve real-mode output quality and is honest about how instruct models
  are invoked. Touches the tokenize stage's story. *(v1 spec, out of scope)*

- [ ] **#19 Faithful browser model** — the published q8 model
  (dynamic int8, one activation scale per tensor) loses the small
  activations whenever the first token's huge values share a batch with
  them: layer 11's MLP outputs exactly 0 for every prompt token after the
  first, and under browser semantics the next-token argmax differs from
  fp32 on 4/4 validation prompts (max |Δlogit| 8–15). Python's validator
  missed it because ORT's full graph optimization fuses
  DynamicQuantizeLinear + MatMulInteger into a kernel that keeps them;
  onnxruntime-web (WASM and WebGPU) runs the unfused graph, as does Python
  with `ORT_DISABLE_ALL`. Keeping the down-projections in fp32 is not
  enough (still 2/4 argmax mismatches). Next: (a) the validator runs
  quantized variants with optimizations disabled, so the gate matches the
  browser; (b) try weight-only quantization (MatMulNBits, 8- or 4-bit, if
  onnxruntime-web supports it on WASM and WebGPU) or a re-try of fp16;
  (c) publish as v5. Affects real mode since M1. *(#15 final review,
  2026-10-11)*
  Also visible in #16's neuron card: at the first generated token about 7
  of 30 layers show "≈" (neuron votes from unrounded activations vs the
  rounded model's MLP vote, up to 1.2 at layer 29); later tokens match.
  A faithful model would make the card exact everywhere.

## Housekeeping / parked minors

- [x] **#6 Storage-consistency follow-up** — two parked M3 minors together:
  (a) after merge-hydrate re-sequences concurrently-sealed records, write
  the new seqs back to IndexedDB (stale seq can collide into duplicate
  chip labels on the next reload); (b) one shared IndexedDB connection
  instead of per-op opens (also removes a theoretical put/delete
  cross-connection ordering hazard that could resurrect a removed run).
  *(M3 final review)*
- [ ] **#7 Polish sweep over parked minors** — `aria-label`s for the
  emoji-only chip action buttons; memoize compare selectors and explorer
  aggregate thumbnails (currently recomputed per render); defer
  `URL.revokeObjectURL` after export click (`setTimeout(0)`); slug fallback
  for punctuation-only prompts (currently `tsumugi-run---….json`);
  `onblocked` handler comment for future IndexedDB schema bumps; the
  unused-but-tested `pairedDistributions` selector (use it in CompareView
  or mark it API-for-tests). *(M2/M3 reviews)*
- [ ] **#20 Neuron card parked minors** — from #16's reviews: pin
  float16 packing with exact-bit tests (ties, exponent carry, subnormal
  boundary); clamp `params.topK` to the graph's k = 16 in the worker;
  worker-level test that bad neuron outputs set `neuronsOff` and log once,
  plus tests for a multi-value 1/rms and a mismatched `topk_embeds`; move
  `resetNeuronCache` out of `asset.ts` so test setup doesn't import the
  tokenizer; a manifest `fractions` length check (the badge fallback hides
  a mismatch); a fixture test pinning `votes.mlp` to the packed neuron
  data; add the model tag to the neurons manifest at v6; array-valued
  events pass run-file import's object check; the editor flags `Buffer` in
  `e2e/` (no node types there). *(#16 reviews,
  2026-10-15; the visible items and the manifest/tokens eviction were fixed
  2026-10-20)*
- [ ] **#21 Import with an empty archive** — the run shelf, and with it
  the Import button, is hidden until the first run exists, so a first-time
  visitor cannot load a shared run file. *(#16 Playwright test,
  2026-10-15)*
- [ ] **#9 Mobile layout** — the app is desktop-first; a narrow-viewport pass
  is substantial design work for an audience that mostly explores on
  laptops. *(v1 spec, out of scope)*
- [ ] **#10 npm-audit recheck** — 4 high transitive CVEs via
  @huggingface/transformers (onnxruntime-node/sharp), node-side only, no
  upstream fix; re-check on dependency bumps. *(v1)*

## Done (for the record)

- [x] Opening the MLP: edit sizes, per-layer votes; export v4 *(#15, 2026-10-11)*
- [x] In-app glossary: 12 entries, term links, drawer, "?" index, first-visit hint *(#14, 2026-10-10)*
- [x] Logits readout: final vector, candidate rows, visible dot product; export v3 *(#13, 2026-10-09)*
- [x] Coreference-head detector in real mode (L13H8 spike, blind template) *(#5, 2026-09-03)*
- [x] Embeddings explained: real rows, neighbours, similarity *(M4, 2026-09-02)*
- [x] Trace archive + cycle-aligned run comparison *(M3, 2026-08-29)*
- [x] 270-head explorer, distinctive detection, selection hysteresis *(M2, 2026-08-28)*
- [x] Real per-layer attention via custom ONNX export *(M1, 2026-08-28)*
- [x] Per-token distribution hover *(2026-08-27)*
