# Labelled MLP neurons — design

**Date:** 2026-10-14
**Status:** Approved design, pre-implementation
**Backlog:** #16 (builds on #15, the MLP cards, and #13, the Logits readout)
**Brainstorm mockup (real probe numbers, illustrative badges):** `.superpowers/brainstorm/66126-1791926765/content/card.html`

## Purpose

The #15 votes card shows how much each layer's MLP added to or took from a
candidate's score. This work opens one MLP: an MLP's edit is the sum of
1,536 neuron edits, so its vote splits exactly into one vote per neuron.
A third card lists the neurons behind a layer's vote, what each one pushes
toward when it fires the way it did, and how the rest of the vote is spread.

## Findings from the offline probe

Probe: fp32 SmolLM2-135M-Instruct, prompt "The cat sat on the mat because it
was tired", top candidate "␣and".

- Labels from raw dot products are dominated by junk tokens ("strugg",
  "affili"). Cosine against the readout rows (final norm weight g applied to
  the embedding rows) gives readable lists.
- Activations are often negative, so a neuron's list depends on the sign it
  fired with. Labels exist for both signs.
- Votes are spread out: in layer 29 the top 5 neurons carry +3.76 of the
  MLP's +9.09. The top voters' lists rarely name the candidate.
- Early layers have tiny votes and noisy lists.

Real layer 29 rows used in the mockup:

| neuron | activation | vote for "␣and" | pushes toward |
|---|---|---|---|
| N1233 | −28.25 | +1.27 | birds, animals, Laura, Miss, Helen |
| N1512 | −24.95 | +1.27 | management, monitoring, payment |
| N288 | +24.36 | +1.19 | experts, myths, spiritual |
| N1066 | −13.28 | +0.90 | Holland, Brazil, behind |
| N600 | −3.59 | −0.88 | away from: and, ␣And, ␣and |

## Decisions already made

- **Both modes**, switched by a toggle on the card: "behind the vote"
  (default; neurons ranked by their vote for the selected candidate) and
  "firing most" (neurons ranked by edit size).
- **Placement:** a third card, "Inside one MLP", always visible below the
  two #15 cards.
- **Votes computed lazily in the browser.** The model exposes activations,
  the down-projection weights ship as an 8-bit asset, and per-neuron votes
  are computed only for the layer being viewed. Rejected: computing votes
  in the graph for the top 10 candidates (about twice the compute per
  token), and shipping activations only (no exact sum).
- **Weights are hosted with the model** on Hugging Face, same repo and tag,
  and cached by the app through the Cache API.
- **Publish v5 early**, as soon as the export passes its checks.

## 1. The card

**Header:** "Inside one MLP: layer N", a ◀ ▶ layer stepper, and an ⓘ link
to the glossary. One line under the header: "The MLP's edit is the sum of
1,536 neuron edits, so its vote splits into one vote per neuron. Click a
layer in either card above to open it here."

**Which layer is shown.**
- Clicking a layer in either #15 card (a heatmap column in the edit-size
  card, a cell or column in the votes card) sets it.
- The stepper moves one layer at a time and stops at layers 0 and 29.
- Until the user picks a layer, the card shows the layer with the largest
  MLP vote (by absolute value) for the selected candidate.

**Candidate.** The same candidate chips as the votes card, sharing its
selection: picking a candidate in either card changes both.

**Mode "behind the vote"** (default). The top 8 neurons by absolute vote.

| column | content |
|---|---|
| neuron | `N1233` |
| activation | signed, one decimal |
| vote for "<candidate>" | signed bar (blue positive, red negative) and number, two decimals |
| when it fires like this, it pushes toward | top 5 tokens for the sign it fired with; when its vote is negative, an "away from" chip followed by the top 5 tokens of the opposite sign (what it pushes away from, as in the mockup's N600 row) |
| "<candidate>" in that list | badge: top 0.1% / top 1% / top 10% / not in top 10%, with "(away)" when the vote is negative |

Below the 8 rows:
- "the other 1,528 neurons": the layer total minus the 8 shown votes;
- "layer N MLP, total": equal to the votes card's MLP number for that layer
  and candidate, followed by "= the votes card's cell".

**Mode "firing most".** The top 8 neurons by edit size
(|activation| × length of the neuron's column). The columns are neuron,
activation, edit size (bar and number) and the token list. There is no
candidate, badge, rest line or total. The line under the header becomes:
"The neurons that changed the vector most for the last token, whatever they
voted for. Size = |activation| × length of the neuron's column."

**Three callouts** below the table, each linking to a section of the
glossary entry (section 5):
- why the list rarely names the candidate;
- negative activations;
- early layers.

**States.**

| condition | card shows |
|---|---|
| simulated run | "Real model only: simulated runs have no neurons to show." |
| run recorded without neuron data | "This run was recorded before neuron data existed." |
| viewed cycle not finished yet | the votes card's pending note, same wording |
| layer file loading | "Loading the MLP weights (about 0.9 MB per layer, once per model version)…" |
| layer file failed to load | "Couldn't load the MLP weights." and a Retry button |
| total outside tolerance (section 3) | table shown; total line reads "≈ <computed total>"; note: "The 8-bit weights are slightly off for this layer, so the neuron votes don't add up exactly to the votes card." |

## 2. Export v5, the asset and delivery

**Model outputs (export v5).** Added to the existing graph, last token only:
- `mlp_activations` [30 × 1536]: the input of each
  `/model/layers.{l}/mlp/down_proj/MatMul`, tapped on the fp32 graph before
  q8 quantization, like the v3 and v4 outputs;
- `final_norm_inv_rms` [1]: the final norm's 1/rms for the last token.

The candidates' embedding rows come from the existing v3 readout output
(`topk_embeds`).

**Asset `neurons/`** in `saigyo-hoshi/smollm2-135m-attn-onnx`, tag v5:
- `neurons/layer-NN.bin`, one per layer (about 0.9 MB each, about 27 MB
  in total), containing:
  - the down-projection columns as int8 [1536 × 576], with one float32
    scale per column;
  - per neuron and sign: the top 5 token ids and three cosine thresholds,
    for the top 0.1%, 1% and 10% of the vocabulary.
- `neurons/tokens.json`: the decoded text of every id, for the labels (about
  450 KB; the geometry asset would mean an extra 5 MB uncached download).
- `neurons/manifest.json`: the byte layout of the layer files, the final
  norm weight vector g (576 float32 values), the label method ("cosine
  against g⊙E rows, fp32 weights, both signs"), the model tag, and a
  SHA-256 for each layer file.

Labels and thresholds are computed from the fp32 weights: for neuron i and
sign s, the cosine between s·W[:, i] and every row g⊙E[t].

**Delivery.** The app loads the manifest once per session and each layer
file the first time it is needed. Files go through the Cache API, keyed by
their tagged URL; where the Cache API is unavailable, plain fetch is used.
Geometry loading is unchanged and out of scope.

**Export checks** (in the export tool, before publishing):
- activation parity between the q8 model and fp32 PyTorch hooks, with the
  clipped tolerance used for q8 in #15;
- per-neuron votes computed from the int8 columns sum to the MLP vote from
  v4's votes outputs on the probe prompt, within the section 3 tolerance;
- every layer file matches the manifest's layout and checksum.

`publish.py` uploads `neurons/` with the model. The model card documents the
new outputs and the asset. `ATTN_MODEL_REVISION` moves to v5.

## 3. Run data and the computation in the browser

**Neuron data on the `edits` event**, as an optional field, one per cycle of
a real run. Every trace event is a player step, so a separate event would add
an empty step. As a field it has no playback time of its own and appears
with the edits step:

```ts
interface NeuronData {
  neurons: number;                                // per layer (1536)
  activations: string;                            // base64 of float16 [30 × 1536], layer-major
  invRms: number;                                 // final norm 1/rms for the last token
  candidates: Array<{ id: number; row: string }>; // each candidate's row of E, base64 float32 [576]
}
// edits event: { …; neurons?: NeuronData }
```

About 150 KB per generated token, so about 3 MB for a 20-token run, in the
archive and in exported files.

**Worker.** It adds the field to the `edits` event, which it now emits after computing the cycle's top candidates (still before `logits`). Following the
`editsOff`/`votesOff` pattern, a `neuronsOff` flag stops the event, with one
`console.warn`, if the outputs are missing or malformed. The run is never
failed because of this event.

**Math module** (pure, main thread). Input: one layer file, that layer's
activations, a candidate row r, g, and inv_rms. With W_i the dequantized
column of neuron i and u = g ⊙ r:
- vote_i = a_i · (W_i · u) · inv_rms;
- edit size_i = |a_i| · ‖W_i‖;
- badge: cosine between sign(a_i)·W_i and u, compared with neuron i's
  thresholds for that sign. When vote_i is negative, the opposite sign's
  list and thresholds are used and the badge says "(away)".
- Output: the top 8 by |vote| or by edit size, the sum of the rest, and the
  total.

One layer for one candidate is about 0.9 million multiply-adds. Results are
memoized per (run, cycle, layer, candidate).

**Asset loader** (module). It loads the manifest once and each layer file at
most once (a request for a layer already in flight waits for that request).
It checks each layer file's size against the manifest and exposes the
loading, error and retry states the card uses.

**Exactness check.** The computed total is compared with the votes card's MLP
number for the layer and candidate, using the #15 tolerance (0.05 + 1% of
the magnitude).
- Within tolerance, the total line shows the votes card's number, and the
  rest line is that number minus the 8 shown votes, so the table adds up
  exactly to the votes card.
- Outside tolerance, the card shows "≈" and the note from section 1.

**Selectors and state.**
- A selector returns the viewed cycle's neuron data or the reason for a
  state note: simulated run, no neuron data, or cycle not finished.
- The shown layer is new shared state in the viz store, set by clicks in
  either #15 card and by the stepper. It is cleared when a new run starts,
  so that the largest-vote default applies again.
- The candidate selection is the votes card's existing state.
- The mode toggle is local state of the card.
- The detail panel uses the viewed run's own mode (from its `run-start`),
  so an archived real run keeps its MLP cards while the app is in sim mode.
  Today such a run wrongly shows the simulated notes on the #15 cards too.

**Imports.** The trace validator checks the new field. A malformed or
truncated `neurons` field is dropped with a note in the import result, and
the rest of the file still loads. This fixes, for this event, the deferred
#15 minor where a broken import threw an error.

## 4. Out of scope

- Neurons of token positions other than the last.
- Attention heads' votes (no change to the head explorer).
- Geometry caching.
- Faithful browser activations (#19): the card shows what the q8 model
  computed.

## 5. Glossary and docs

- **New glossary entry "MLP neuron"**, explaining step by step:
  - what an activation is;
  - that each neuron adds its column to the vector, scaled by its activation;
  - how the token lists are made (cosine against the readout rows);
  - why the sign matters ("away from").

  It has three sections that the card's callouts link to:
  - **Why the list rarely names the candidate:** a neuron's vote is its
    push along the candidate's row. A neuron can push strongly toward its
    own tokens and still add to the candidate's score through a smaller
    share of that push.
  - **Negative activations:** SwiGLU lets activations go below zero; a
    negative activation pushes the vector along the opposite direction, so
    the list for that sign is shown.
  - **Early layers:** their votes are small and their lists look like
    noise, because the vector there is still far from the readout space.
- **Existing entries:** "MLP" and "votes" link to the new entry. The votes
  entry adds one sentence: an MLP's vote is the sum of its neurons' votes.
- **README:** a short paragraph in the MLP section describes the third card
  and its toggle, and states that the weights download once per model
  version.
- **Screenshot:** a new screenshot of the card, made with
  `npm run screenshots`.

## 6. Testing

**Export (Python, `tools/export`):**
- the v5 activations match fp32 PyTorch hooks within the #15 clipped
  tolerance;
- int8 columns survive a round trip within each column's scale;
- on the probe prompt, per-neuron votes sum to the MLP vote within the
  section 3 tolerance;
- the manifest's checksums match the files;
- labels on a tiny fixture model match hand-computed lists for both signs.

**Unit tests (vitest):**
- **Math module**, on a hand-built layer with 4 neurons: votes, the top 8
  with the rest and total, edit sizes, each badge level including "(away)",
  and both exactness outcomes.
- **Packing:** float16 and base64 round trip within float16 rounding.
- **Asset loader:** cache hit; cache miss fetches and stores; a failed fetch
  ends in the error state and retry works; plain fetch is used without the
  Cache API; concurrent requests for one layer fetch it once.
- **Worker:** missing or malformed outputs set `neuronsOff`, log one
  warning, and the run finishes.
- **Trace validator:** a valid event is accepted; a truncated one is dropped
  with a note and the rest of the import loads.
- **Selectors and card:** every state from section 1; the mode toggle; the
  stepper stops at 0 and 29; clicks in either #15 card move the third card;
  the candidate selection is shared with the votes card; the default is
  the largest-vote layer.

**Browser test (Playwright):** a recorded run plus a small fake asset served
through route interception. It opens the card, switches mode, steps
through layers, and checks that the total matches the votes card's cell.

**By hand after deploy:** a live run on the probe prompt reproduces the
layer 29 numbers within q8 tolerance, and the built card is compared with
the approved mockup.
