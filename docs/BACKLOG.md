# Backlog

Ideas and follow-ups collected across milestones (v1, M1 real attention,
M2 head explorer, M3 trace archive + comparison). Check items off as they
ship; add provenance when adding new ones. Deliberately lighter-weight
than GitHub issues.

Items carry permanent numbers for easy reference ("#2 next?"). A new item
takes the next free number; numbers are never reused or renumbered, so
they are IDs, not positions.

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
- [ ] **#5 Coreference-head discovery** — real mode detects previous-token /
  sink / induction / distinctive, but the coreference role (poster child of
  the sim examples) has no real-mode detector. Research-flavored: needs a
  statistical template for "pronoun attends to its referent", which may not
  exist cleanly in a 135M model — run a spike first. *(M1/M2 backlog)*

## Real-mode quality

- [ ] **#8 Chat templating** — run prompts through SmolLM2-Instruct's actual
  chat template so the model sees its expected format; should noticeably
  improve real-mode output quality and is honest about how instruct models
  are invoked. Touches the tokenize stage's story. *(v1 spec, out of scope)*

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
- [ ] **#9 Mobile layout** — the app is desktop-first; a narrow-viewport pass
  is substantial design work for an audience that mostly explores on
  laptops. *(v1 spec, out of scope)*
- [ ] **#10 npm-audit recheck** — 4 high transitive CVEs via
  @huggingface/transformers (onnxruntime-node/sharp), node-side only, no
  upstream fix; re-check on dependency bumps. *(v1)*

## Done (for the record)

- [x] Trace archive + cycle-aligned run comparison *(M3, 2026-08-29)*
- [x] 270-head explorer, distinctive detection, selection hysteresis *(M2, 2026-08-28)*
- [x] Real per-layer attention via custom ONNX export *(M1, 2026-08-28)*
- [x] Per-token distribution hover *(2026-08-27)*
