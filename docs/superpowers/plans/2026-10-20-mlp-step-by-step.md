# The MLP, step by step — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship model v6 (faithful weight-only 8-bit model, four new MLP outputs, manifest revision), a fourth Layers card that walks through one MLP's five steps with real values, and a sticky navigation bar for the Layers detail.

**Architecture:** The export tool taps four more tensors per layer, quantizes with `MatMulNBits` (8-bit, block 32) plus a uint8 embedding Gather, and validates under browser semantics (graph optimizations off). The worker packs the new outputs as an optional `steps` object on the neuron data. A pure math module derives what the card draws; the card's charts are full-width SVGs (viewBox width 720), like the #15 cards. One shared hook decides the MLP layer for the neuron card, the new card and the navigation bar.

**Tech Stack:** Python 3.13 + onnx + onnxruntime 1.29 + torch (tools/export, uv, pytest); React 19 + TypeScript + zustand + vitest + Playwright; transformers.js 4.2 / onnxruntime-web 1.26.

**Spec:** `docs/superpowers/specs/2026-10-20-mlp-step-by-step-design.md` (read it first). Mockups: `.superpowers/brainstorm/67669-1792530069/content/anatomy-v2.html` (card) and `navigation.html` (bar, option A).

## Global Constraints

- Model variant: `MatMulNBits`, bits 8, block 32, symmetric, `nodes_to_exclude = NODES_TO_EXCLUDE` (`/lm_head/MatMul`, `tsumugi_topk_embeds_gather`), then `quantize_dynamic(weight_type=QUInt8, op_types_to_quantize=["Gather"], nodes_to_exclude=NODES_TO_EXCLUDE)`. File name stays `onnx/model_quantized.onnx` (loaded with `dtype: 'q8'`). Expected size ≈ 263 MB.
- New graph outputs: `mlp_gate`, `mlp_up` `[batch, layers, neurons]`; `mlp_in`, `mlp_out` `[batch, layers, dims]`; last position only.
- Model tag: `v6`. `ATTN_MODEL_REVISION = 'v6'`. Manifest fields `"revision": "v6"`, `"rmsNormEps": 1e-05`; layer layout gains `normWeight` (dims float32, at the end).
- Validator gate for quantized variants: sessions with `GraphOptimizationLevel.ORT_DISABLE_ALL`; on `ARGMAX_PROMPTS` (four prompts below) same argmax as stock fp32 and max |Δlogit| ≤ 1.5.
- `ARGMAX_PROMPTS = ["The cat sat on the mat because it was tired", "one two three one two three one", "The capital of France is", "Once upon a time, there was a little"]`.
- Run data: `NeuronData.steps?: { gate: string; up: string; mlpIn: string; mlpOut: string }`, float16 base64, layer-major.
- Card notation follows the glossary: n = norm(x) with learned scale w, g = W_gate · n, u = W_up · n, h = SiLU(g) ⊙ u, edit = W_down · h.
- Card title "The MLP, step by step: layer N"; notes: sim "Real model only: simulated runs don’t compute an MLP."; old "This run was recorded before the step-by-step data existed. Generate it again to see it."
- Charts: SVG `viewBox` width 720, class `mlp-chart` (`width: 100%`); label column 150 px; rows stack below 700 px window width; charts never narrower than 520 px (`overflow-x: auto` around them).
- Marked-neuron accent: CSS variable `--mark: #d08a1c`.
- Navigation labels, in order: Overview, Attention, Edit sizes, Votes, Inside one MLP, Step by step; stepper text "MLP layer N".
- Checks before every commit that touches app code: `npx vitest run`, `npx tsc -b`, `npx oxlint` (two known warnings in `usePins.ts` and `SamplerDetail.tsx` only). Export code: `cd tools/export && uv run pytest -q`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on branch `feat/17-mlp-step-by-step`. Never push, publish to Hugging Face, or merge without the user's go-ahead (Task 15 stops for it).

## Review Focus

1. **A layer whose gates are all on one side or tightly bunched** (early layers: 1,292 of 1,536 negative, range −4.1…+1.7) — the scatter and SiLU charts must still show both sides of zero and the curve's bend. Pinned by `gateDomain` tests in Task 10.
2. **An imported v5 run** (neuron data without `steps`) — the new card shows the "recorded before" note; the neuron card is unaffected. Pinned in Task 12.
3. **Stepping layers while a layer file loads** — the card must never show one layer's columns or norm weight with another layer's values. Pinned in Task 12 (stale-load test).
4. **Narrow windows (390 px)** — the page must not scroll sideways; step rows stack. Pinned in Task 14's Playwright test.
5. **Damaged step data in an imported file** (wrong length, NaN) — only `steps` is dropped, the run and the neuron card survive. Pinned in Task 7.

---

## File Structure

Export (tools/export/src/tsumugi_export):
- `postprocess.py` — add `find_mlp_projections`, `expose_mlp_steps`; wire into `postprocess_graph`.
- `quantize.py` — weight-only 8-bit plus uint8 Gather.
- `stock.py` — add `stock_mlp_norms`.
- `neurons.py` — `normWeight` in the layout, manifest `revision` and `rmsNormEps`, CLI `--revision`.
- `validate.py` — browser-semantics sessions, argmax gate, step checks, gating neuron votes, neuron file checks, `browser-ref.json`.
- `publish.py` — model card text, refuse a manifest revision ≠ tag.
- `__main__.py` — `neurons --revision`.
- `tools/export/browser-check.html` (new) — loads the local model through transformers.js on WebGPU and WASM.

App (src):
- `trace/types.ts` — `MlpSteps`, `NeuronData.steps`.
- `trace/validate.ts` — `stepProblems`, drop only `steps`.
- `engine/transformers/neuronData.ts` — `extractSteps`.
- `engine/transformers/worker.ts` — wire `extractSteps`, `stepsOff`.
- `engine/tokenizer.ts` — `ATTN_MODEL_REVISION = 'v6'`.
- `neurons/format.ts` — layout `normWeight`, manifest `revision`, `rmsNormEps`.
- `neurons/asset.ts` — refuse a manifest whose revision differs.
- `neurons/math.ts` — export `columnNorm`, `neuronLabels`; add `listedNeurons`.
- `neurons/steps.ts` (new) — the card's math.
- `viz/mlpLayerSelection.ts` — `effectiveMlpLayer`, `useMlpLayerControl`.
- `viz/NeuronCard.tsx` — use `useMlpLayerControl`.
- `viz/MlpStepCharts.tsx` (new) — the five charts.
- `viz/MlpStepsCard.tsx` (new) — the card.
- `viz/LayersNav.tsx` (new) — the navigation bar.
- `viz/details/LayersDetail.tsx` — sections, nav, the card.
- `glossary/callouts.ts`, `glossary/entries/mlp.tsx` — three callouts and sections.
- `index.css` — card, chart and nav styles.
- `test/neuronFixture.ts`, `test/fixtures.ts` — fixture steps, normWeight, revision.
- `e2e/steps.spec.ts` (new).

---

### Task 1: Export the four step outputs

**Files:**
- Modify: `tools/export/src/tsumugi_export/postprocess.py` (after `expose_neurons`, and `postprocess_graph`)
- Test: `tools/export/tests/test_postprocess_mlp.py`

**Interfaces:**
- Produces: `MLP_GATE_OUTPUT = "mlp_gate"`, `MLP_UP_OUTPUT = "mlp_up"`, `MLP_IN_OUTPUT = "mlp_in"`, `MLP_OUT_OUTPUT = "mlp_out"`; `find_mlp_projections(graph) -> list[tuple[str, str]]` (gate, up output names per layer); `expose_mlp_steps(model) -> model`.

- [ ] **Step 1: Give the miniature graph gate and up projections**

In `tests/test_postprocess_mlp.py`, add module constants after `GAIN`:

```python
NEUR = 6
WG = [rng.normal(scale=0.5, size=(DIMS, NEUR)).astype(np.float32) for _ in range(LAYERS)]
WU = [rng.normal(scale=0.5, size=(DIMS, NEUR)).astype(np.float32) for _ in range(LAYERS)]
```

In `layered_graph()`, inside the layer loop, extend `inits` with `wg{l}` and `wu{l}` and add two nodes after the `post_attention_layernorm/Cast` node (their outputs are unused, like dangling taps; `find_blocks` and `find_mlp_inputs` are unaffected):

```python
        inits += [helper.make_tensor(f"wg{l}", f, [DIMS, NEUR], WG[l].flatten().tolist()),
                  helper.make_tensor(f"wu{l}", f, [DIMS, NEUR], WU[l].flatten().tolist())]
        nodes += [
            helper.make_node("MatMul", [f"{p}/post_attention_layernorm/Cast_output_0", f"wg{l}"],
                             [f"{p}/mlp/gate_proj/MatMul_output_0"], name=f"{p}/mlp/gate_proj/MatMul"),
            helper.make_node("MatMul", [f"{p}/post_attention_layernorm/Cast_output_0", f"wu{l}"],
                             [f"{p}/mlp/up_proj/MatMul_output_0"], name=f"{p}/mlp/up_proj/MatMul"),
        ]
```

(Place these `nodes +=` lines after the Cast node is appended and before the down_proj MatMul; topological order only requires the Cast first.)

- [ ] **Step 2: Write the failing tests**

Add to the imports: `MLP_GATE_OUTPUT, MLP_IN_OUTPUT, MLP_OUT_OUTPUT, MLP_UP_OUTPUT, expose_mlp_steps, find_mlp_projections`. Add:

```python
def test_find_mlp_projections_names_each_layers_gate_and_up_outputs():
    assert find_mlp_projections(layered_graph().graph) == [
        (f"/model/layers.{l}/mlp/gate_proj/MatMul_output_0", f"/model/layers.{l}/mlp/up_proj/MatMul_output_0")
        for l in range(LAYERS)]


def test_step_outputs_are_the_last_positions_gate_up_input_and_edit():
    model = expose_mlp_steps(layered_graph())
    onnx.checker.check_model(model)
    ids = np.array([[3, 1, 4, 1]], dtype=np.int64)
    out = _run(model, ids)
    _, _, mlp, _, before_m, _, _ = reference(ids)
    assert out[MLP_GATE_OUTPUT].shape == (1, LAYERS, NEUR)
    assert out[MLP_IN_OUTPUT].shape == (1, LAYERS, DIMS)
    for l in range(LAYERS):
        x = before_m[l][-1]
        np.testing.assert_allclose(out[MLP_GATE_OUTPUT][0, l], x @ WG[l], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[MLP_UP_OUTPUT][0, l], x @ WU[l], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[MLP_IN_OUTPUT][0, l], x, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[MLP_OUT_OUTPUT][0, l], mlp[l][-1], rtol=1e-5, atol=1e-6)
    declared = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim][1:] for o in model.graph.output}
    assert declared[MLP_UP_OUTPUT] == [LAYERS, NEUR] and declared[MLP_OUT_OUTPUT] == [LAYERS, DIMS]


def test_a_layer_without_up_proj_is_refused():
    model = layered_graph()
    for node in model.graph.node:
        if node.name == "/model/layers.1/mlp/up_proj/MatMul":
            node.name = "/model/layers.1/mlp/renamed/MatMul"
    try:
        expose_mlp_steps(model)
    except ValueError as err:
        assert "up_proj" in str(err)
    else:
        raise AssertionError("expected ValueError")
```

Change `test_postprocess_graph_adds_the_mlp_outputs_after_the_readout`'s assertion to:

```python
    assert names[-9:] == [EDIT_RATIOS_OUTPUT, STREAM_PARTS_OUTPUT, MLP_ACTIVATIONS_OUTPUT, INV_RMS_OUTPUT,
                          MLP_GATE_OUTPUT, MLP_UP_OUTPUT, MLP_IN_OUTPUT, MLP_OUT_OUTPUT, "inputs_embeds"]
```

Note: in the miniature, down_proj's weight is `[DIMS, DIMS]`, so `_neuron_count` reports DIMS for `mlp_activations`; `expose_mlp_steps` must take the neuron count from the gate weight instead (it is `NEUR` here).

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_postprocess_mlp.py -q`
Expected: FAIL with `ImportError: cannot import name 'MLP_GATE_OUTPUT'`.

- [ ] **Step 4: Implement**

In `postprocess.py`, after `expose_neurons`:

```python
MLP_GATE_OUTPUT = "mlp_gate"
MLP_UP_OUTPUT = "mlp_up"
MLP_IN_OUTPUT = "mlp_in"
MLP_OUT_OUTPUT = "mlp_out"
GATE_MATMUL = "/model/layers.{l}/mlp/gate_proj/MatMul"
UP_MATMUL = "/model/layers.{l}/mlp/up_proj/MatMul"


def find_mlp_projections(graph: onnx.GraphProto) -> list[tuple[str, str]]:
    """Per layer, the outputs of the gate_proj and up_proj MatMuls: the gate
    before SiLU and the up values it scales."""
    by_name = {n.name: n for n in graph.node}
    out: list[tuple[str, str]] = []
    while (gate := by_name.get(GATE_MATMUL.format(l=len(out)))) is not None:
        up = by_name.get(UP_MATMUL.format(l=len(out)))
        if up is None:
            raise ValueError(f"no {UP_MATMUL.format(l=len(out))} next to its gate_proj")
        out.append((gate.output[0], up.output[0]))
    if not out:
        raise ValueError(f"no {GATE_MATMUL.format(l=0)} — is this the SmolLM2 export?")
    return out


def _weight_cols(graph: onnx.GraphProto, matmul_name: str) -> int:
    mm = next(n for n in graph.node if n.name == matmul_name)
    init = next((t for t in graph.initializer if t.name == mm.input[1]), None)
    if init is None or len(init.dims) != 2:
        raise ValueError(f"{matmul_name} weight is not a 2-D initializer")
    return int(init.dims[1])


def expose_mlp_steps(model: onnx.ModelProto) -> onnx.ModelProto:
    """For the step-by-step card (#17), last position only: mlp_gate and mlp_up
    [batch, layers, neurons] (the gate before SiLU and the up values), mlp_in and
    mlp_out [batch, layers, dims] (the stream going into each MLP and its edit).
    Tapped on the fp32 graph before quantization; the quantizer keeps tensor names."""
    graph = model.graph
    proj = find_mlp_projections(graph)
    blocks = find_blocks(graph)
    if len(blocks) != len(proj):
        raise ValueError(f"{len(proj)} gate/up pairs for {len(blocks)} layers")
    _, _, dims = find_lm_head(graph)
    neurons = _weight_cols(graph, GATE_MATMUL.format(l=0))
    graph.initializer.extend([
        helper.make_tensor("tsumugi_step_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_step_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_step_slice_axes", TensorProto.INT64, [1], [1]),
    ])
    slice_args = ["tsumugi_step_slice_starts", "tsumugi_step_slice_ends", "tsumugi_step_slice_axes"]
    sources = {
        MLP_GATE_OUTPUT: [g for g, _ in proj], MLP_UP_OUTPUT: [u for _, u in proj],
        MLP_IN_OUTPUT: [b[3] for b in blocks], MLP_OUT_OUTPUT: [b[2] for b in blocks],
    }
    nodes: list[onnx.NodeProto] = []
    for out_name, names in sources.items():
        parts = [f"tsumugi_{out_name}{l}" for l in range(len(names))]
        nodes += [helper.make_node("Slice", [name, *slice_args], [part], name=part) for name, part in zip(names, parts)]
        nodes.append(helper.make_node("Concat", parts, [out_name], name=f"tsumugi_{out_name}", axis=1))
    graph.node.extend(nodes)
    width = {MLP_GATE_OUTPUT: neurons, MLP_UP_OUTPUT: neurons, MLP_IN_OUTPUT: dims, MLP_OUT_OUTPUT: dims}
    graph.output.extend([helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch_size", len(proj), width[name]])
                         for name in sources])
    return model
```

Change `postprocess_graph` to `return expose_inputs_embeds(expose_mlp_steps(expose_neurons(expose_mlp_internals(expose_readout(model, k)))))` and add "step outputs" to its docstring.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tools/export/src/tsumugi_export/postprocess.py tools/export/tests/test_postprocess_mlp.py
git commit -m "feat(export): mlp_gate, mlp_up, mlp_in and mlp_out outputs for the step-by-step card

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Weight-only 8-bit quantization

**Files:**
- Modify: `tools/export/src/tsumugi_export/quantize.py`
- Test: `tools/export/tests/test_quantize.py` (new)

**Interfaces:**
- Produces: `quantize_file(src: Path, dst: Path) -> None`; `run(args)` writes `onnx/model_quantized.onnx` via `quantize_file`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import onnx
from onnx import TensorProto, helper

from tsumugi_export.quantize import quantize_file


def tiny_model(path):
    """input ids → embedding Gather → a MatMul named like a layer → /lm_head/MatMul."""
    f = TensorProto.FLOAT
    rng = np.random.default_rng(0)
    table = rng.normal(size=(50, 64)).astype(np.float32)
    w = rng.normal(size=(64, 64)).astype(np.float32)
    head = rng.normal(size=(64, 50)).astype(np.float32)
    inits = [helper.make_tensor("model.embed_tokens.weight", f, [50, 64], table.flatten().tolist()),
             helper.make_tensor("w", f, [64, 64], w.flatten().tolist()),
             helper.make_tensor("head", f, [64, 50], head.flatten().tolist())]
    nodes = [helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["e"], name="/model/embed_tokens/Gather"),
             helper.make_node("MatMul", ["e", "w"], ["h"], name="/model/layers.0/mlp/up_proj/MatMul"),
             helper.make_node("MatMul", ["h", "head"], ["logits"], name="/lm_head/MatMul")]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "s"])],
                              [helper.make_tensor_value_info("logits", f, [1, "s", 50])], initializer=inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]), str(path))


def test_layers_become_8_bit_weight_only_the_head_stays_fp32_and_the_embedding_is_uint8(tmp_path):
    src, dst = tmp_path / "m.onnx", tmp_path / "q.onnx"
    tiny_model(src)
    quantize_file(src, dst)
    m = onnx.load(str(dst))
    by_op = {}
    for n in m.graph.node:
        by_op.setdefault(n.op_type, []).append(n)
    nbits = by_op["MatMulNBits"]
    assert len(nbits) == 1
    attrs = {a.name: helper.get_attribute_value(a) for a in nbits[0].attribute}
    assert attrs["bits"] == 8 and attrs["block_size"] == 32
    assert [n.name for n in by_op["MatMul"]] == ["/lm_head/MatMul"]
    assert "DynamicQuantizeLinear" not in by_op                  # activations are never rounded
    assert any(t.data_type == TensorProto.UINT8 and list(t.dims) == [50, 64] for t in m.graph.initializer)


def test_the_quantized_model_stays_close_to_fp32(tmp_path):
    import onnxruntime as ort
    src, dst = tmp_path / "m.onnx", tmp_path / "q.onnx"
    tiny_model(src)
    quantize_file(src, dst)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ids = {"input_ids": np.array([[1, 7, 42]], dtype=np.int64)}
    ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, ids)[0]
    got = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"]).run(None, ids)[0]
    assert np.max(np.abs(got - ref)) < 0.05 * np.max(np.abs(ref))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools/export && uv run pytest tests/test_quantize.py -q`
Expected: FAIL with `ImportError: cannot import name 'quantize_file'`.

- [ ] **Step 3: Implement**

Replace the body of `quantize.py` below the imports (keep `NODES_TO_EXCLUDE`); extend the module docstring with a paragraph:

```
- 2026-10-20 (#19): dynamic int8 rounds activations with one scale per
  tensor; when the first token's huge values share that scale, small
  activations round to zero (layer 11's MLP outputs exactly 0 after the first
  token). Python's validator missed it because full graph optimization fuses
  the quantized matmuls into a kernel that keeps them; onnxruntime-web runs
  the unfused graph. Weight-only 8-bit (MatMulNBits, block 32, symmetric)
  keeps activations in fp32: same argmax as fp32 on 4/4 prompts under browser
  semantics (q8: 1–2/4), max |Δlogit| ≈ 1.0, on WASM and WebGPU. Block 128:
  3/4; 4-bit: 1/4. The embedding Gather is stored as uint8 as before
  (263 MB in total; 347 MB with an fp32 embedding table).
```

```python
def quantize_file(src: Path, dst: Path) -> None:
    """Weight-only 8-bit matmuls (activations stay fp32), then the embedding
    Gather as uint8; the tied lm_head and the readout gather stay fp32."""
    import tempfile

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    q = MatMulNBitsQuantizer(onnx.load(str(src)), bits=8, block_size=32, is_symmetric=True,
                             nodes_to_exclude=NODES_TO_EXCLUDE)
    q.process()
    with tempfile.TemporaryDirectory() as tmp:
        mid = Path(tmp) / "w8.onnx"
        q.model.save_model_to_file(str(mid), use_external_data_format=False)
        quantize_dynamic(str(mid), str(dst), weight_type=QuantType.QUInt8,
                         op_types_to_quantize=["Gather"], nodes_to_exclude=NODES_TO_EXCLUDE)


def run(args) -> int:
    onnx_dir = Path(args.model_dir) / "onnx"
    src = onnx_dir / "model.onnx"
    if not src.exists():
        print(f"missing {src} — run export first")
        return 1
    quantize_file(src, onnx_dir / "model_quantized.onnx")
    print("wrote model_quantized.onnx (weight-only 8-bit, uint8 embedding; lm_head and readout gather fp32)")
    return 0
```

Also change the `quantize` subparser help in `__main__.py` to `"produce the weight-only 8-bit variant (model_quantized.onnx)"`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass. If `quantize_dynamic` refuses the `com.microsoft` domain or the uint8 initializer check finds a different shape, print the initializers of `q.onnx` and adapt the assertion to the quantizer's naming, not the behaviour.

- [ ] **Step 5: Commit**

```bash
git add tools/export/src/tsumugi_export/quantize.py tools/export/src/tsumugi_export/__main__.py tools/export/tests/test_quantize.py
git commit -m "feat(export): weight-only 8-bit model with uint8 embedding (#19)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Neuron asset — norm weights, revision, eps

**Files:**
- Modify: `tools/export/src/tsumugi_export/neurons.py`, `stock.py`, `__main__.py`
- Test: `tools/export/tests/test_neurons.py`

**Interfaces:**
- Produces: `layer_layout(neurons, dims, labels=5, fractions=FRACTIONS)` now also returns `"normWeight"` (offset) and `bytes` grows by `dims * 4`; `build_layer(W, g, E, norm=None, ...)`; `build_neurons(W_layers, g, E, texts, norms=None, revision="dev", eps=1e-5)`; manifest keys `revision`, `rmsNormEps`; `stock_mlp_norms() -> list[np.ndarray]` (each `[dims]`, post_attention_layernorm weight); CLI `neurons --revision v6` (required).

- [ ] **Step 1: Write the failing tests**

In `test_neurons.py`, update `test_layout_offsets_follow_each_other_and_stay_aligned`:

```python
def test_layout_offsets_follow_each_other_and_stay_aligned():
    lay = layer_layout(1536, 576)
    assert lay == {"scales": 0, "columns": 6144, "labelIds": 890880, "thresholds": 921600,
                   "normWeight": 958464, "bytes": 960768}
    assert lay["labelIds"] % 2 == 0 and lay["thresholds"] % 4 == 0 and lay["normWeight"] % 4 == 0
```

Update `decode` to slice thresholds up to `lay["normWeight"]` and return a fifth value `norm = np.frombuffer(raw[lay["normWeight"]:lay["bytes"]], dtype="<f4")`; update its callers to unpack five values (`scales, q, _, _, _ = ...`). Add:

```python
def test_each_layer_file_ends_with_that_layers_norm_weight():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(8, 6)).astype(np.float32)
    E = rng.normal(size=(40, 8)).astype(np.float32)
    norm = rng.uniform(0.5, 2, 8).astype(np.float32)
    *_, got = decode(build_layer(W, np.ones(8, np.float32), E, norm), 6, 8)
    np.testing.assert_array_equal(got, norm)


def test_manifest_names_the_revision_and_the_norm_eps(tmp_path):
    rng = np.random.default_rng(4)
    W = [rng.normal(size=(4, 3)).astype(np.float32) for _ in range(2)]
    E = rng.normal(size=(30, 4)).astype(np.float32)
    built = build_neurons(W, np.ones(4, np.float32), E, [f"t{i}" for i in range(30)],
                          norms=[np.full(4, 2.0, np.float32)] * 2, revision="v6", eps=1e-5)
    assert built["manifest"]["revision"] == "v6" and built["manifest"]["rmsNormEps"] == 1e-5
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_neurons.py -q`
Expected: FAIL (layout dict lacks `normWeight`).

- [ ] **Step 3: Implement**

```python
def layer_layout(neurons: int, dims: int, labels: int = LABELS, fractions: tuple = FRACTIONS) -> dict:
    columns = neurons * 4
    label_ids = columns + neurons * dims
    thresholds = label_ids + neurons * 2 * labels * 2
    norm_weight = thresholds + neurons * 2 * len(fractions) * 4
    return {"scales": 0, "columns": columns, "labelIds": label_ids, "thresholds": thresholds,
            "normWeight": norm_weight, "bytes": norm_weight + dims * 4}
```

`build_layer(W, g, E, norm=None, labels=LABELS, fractions=FRACTIONS, block=256)`: at the end `norm = np.ones(W.shape[0], np.float32) if norm is None else norm` and append `norm.astype("<f4").tobytes()` to the returned bytes. Docstring: "norm is the layer's post-attention RMSNorm weight [dims], for the step-by-step card."

`build_neurons(W_layers, g, E, texts, norms=None, revision="dev", eps=1e-5)`: pass `norms[l] if norms else None` to `build_layer`; add `"revision": revision, "rmsNormEps": float(eps)` to the manifest.

In `stock.py`:

```python
def stock_mlp_norms() -> list[np.ndarray]:
    """Every layer's post-attention RMSNorm weight, fp32 [dims], in layer order."""
    state = _stock_state()
    out, l = [], 0
    while (key := f"model.layers.{l}.post_attention_layernorm.weight") in state:
        out.append(state[key].float().numpy())
        l += 1
    return out
```

In `neurons.run`: import `stock_mlp_norms`; read eps with `AutoConfig.from_pretrained(STOCK_MODEL_ID).rms_norm_eps`; call `build_neurons(stock_mlp_down(), stock_final_norm(), E, texts, norms=stock_mlp_norms(), revision=args.revision, eps=eps)`; print the revision. In `__main__.py`: `p_neu.add_argument("--revision", required=True, help="the Hub tag this asset ships under (e.g. v6); the app refuses a manifest for another tag")`.

- [ ] **Step 4: Run to verify they pass**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass (update `test_validate_neurons.py` helpers that build layer files if they break, by passing nothing new — `norm` defaults to ones).

- [ ] **Step 5: Commit**

```bash
git add tools/export/src/tsumugi_export/neurons.py tools/export/src/tsumugi_export/stock.py tools/export/src/tsumugi_export/__main__.py tools/export/tests/
git commit -m "feat(export): neuron asset carries per-layer norm weights, its revision and the norm eps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Validator under browser semantics, step checks, publish guard

**Files:**
- Modify: `tools/export/src/tsumugi_export/validate.py`, `publish.py`
- Test: `tools/export/tests/test_validate_steps.py` (new), `tools/export/tests/test_publish_gate.py`

**Interfaces:**
- Consumes: Task 1 output names; Task 3 manifest keys and layout.
- Produces: `_session(path, browser=False)`; `ARGMAX_PROMPTS`; `check_argmax_parity(cases: list[tuple[np.ndarray, np.ndarray]], max_delta=1.5) -> dict` (pairs of ours/stock last-position logits); `check_steps_parity(outputs, ref: dict[str, np.ndarray], abs_tol, rel_tol) -> dict`; `check_silu_product(outputs, rtol=1e-3, atol=1e-4) -> dict`; `torch_step_reference(ids) -> dict` with keys `mlp_gate, mlp_up, mlp_in, mlp_out` (`[layers, n]`); `write_browser_reference(path, model_dir, stock_sess)`; `assert_manifest_matches_tag(manifest: dict, tag: str)` in publish.py.

- [ ] **Step 1: Write the failing tests**

`tests/test_validate_steps.py`:

```python
import numpy as np

from tsumugi_export.validate import check_argmax_parity, check_silu_product, check_steps_parity

rng = np.random.default_rng(5)


def test_argmax_parity_passes_when_every_prompt_agrees_within_the_delta():
    a = rng.normal(size=50).astype(np.float32)
    assert check_argmax_parity([(a + 0.1, a), (a, a)])["passed"]


def test_argmax_parity_fails_on_a_different_argmax_or_a_large_delta():
    a = np.zeros(10, np.float32); a[3] = 5
    b = a.copy(); b[7] = 6
    assert not check_argmax_parity([(b, a)])["passed"]
    c = a.copy(); c[0] = 2.0                         # same argmax, Δ = 2.0 > 1.5
    r = check_argmax_parity([(c, a)])
    assert not r["passed"] and "2.0" in r["detail"]


def test_steps_parity_names_the_output_and_layer_that_differ():
    ref = {k: rng.normal(size=(3, 8)).astype(np.float32) for k in ("mlp_gate", "mlp_up", "mlp_in", "mlp_out")}
    outputs = {k: v[None].copy() for k, v in ref.items()}
    assert check_steps_parity(outputs, ref, 1e-3, 1e-3)["passed"]
    outputs["mlp_up"][0, 2, 4] += 1
    r = check_steps_parity(outputs, ref, 1e-3, 1e-3)
    assert not r["passed"] and "mlp_up" in r["detail"] and "layer 2" in r["detail"]
    del outputs["mlp_in"]
    assert "no mlp_in" in check_steps_parity(outputs, ref, 1e-3, 1e-3)["detail"]


def test_silu_of_the_gate_times_up_reproduces_the_activations():
    gate = rng.normal(size=(1, 3, 8)).astype(np.float32)
    up = rng.normal(size=(1, 3, 8)).astype(np.float32)
    acts = gate / (1 + np.exp(-gate)) * up
    assert check_silu_product({"mlp_gate": gate, "mlp_up": up, "mlp_activations": acts})["passed"]
    assert not check_silu_product({"mlp_gate": gate, "mlp_up": up, "mlp_activations": acts + 0.1})["passed"]
```

In `test_publish_gate.py` add:

```python
from tsumugi_export.publish import assert_manifest_matches_tag


def test_a_neuron_manifest_for_another_tag_is_refused():
    assert_manifest_matches_tag({"revision": "v6"}, "v6")
    for manifest in ({"revision": "v5"}, {}):
        try:
            assert_manifest_matches_tag(manifest, "v6")
        except ValueError as err:
            assert "v6" in str(err)
        else:
            raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_validate_steps.py tests/test_publish_gate.py -q`
Expected: FAIL with ImportErrors.

- [ ] **Step 3: Implement the check functions**

In `validate.py`:

```python
ARGMAX_PROMPTS = [
    "The cat sat on the mat because it was tired",
    "one two three one two three one",
    "The capital of France is",
    "Once upon a time, there was a little",
]
STEP_OUTPUTS = ("mlp_gate", "mlp_up", "mlp_in", "mlp_out")


def _session(path: Path, browser: bool = False):
    """browser=True disables graph optimizations: onnxruntime-web runs the
    unfused graph, and full optimization fuses quantized matmuls into kernels
    that hide rounding the browser does (the #19 lesson)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    if browser:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def check_argmax_parity(cases: list[tuple[np.ndarray, np.ndarray]], max_delta: float = 1.5) -> dict:
    """(ours, stock) last-position logits per prompt: same argmax everywhere and max|Δ| ≤ max_delta."""
    same = sum(int(np.argmax(o) == np.argmax(s)) for o, s in cases)
    worst = max(float(np.max(np.abs(o - s))) for o, s in cases)
    return {"passed": same == len(cases) and worst <= max_delta,
            "detail": f"same argmax {same}/{len(cases)}, max|Δlogit| {worst:.2f} (limit {max_delta})"}


def check_steps_parity(outputs: dict, ref: dict, abs_tol: float, rel_tol: float) -> dict:
    """Each step output [1, layers, n] against the reference [layers, n]:
    per layer max|Δ| ≤ abs_tol + rel_tol · max|ref of that layer|."""
    worst = (0.0, "", 0)
    for name in STEP_OUTPUTS:
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
        got, want = outputs[name], ref[name]
        if got.shape != (1, *want.shape):
            return {"passed": False, "detail": f"{name} shape {got.shape}, expected (1, {', '.join(map(str, want.shape))})"}
        for l in range(want.shape[0]):
            diff = float(np.max(np.abs(got[0, l] - want[l])))
            allowed = abs_tol + rel_tol * float(np.max(np.abs(want[l])))
            if diff > allowed:
                return {"passed": False, "detail": f"{name} layer {l}: max|Δ|={diff:.3e} > {allowed:.3e}"}
            worst = max(worst, (diff, name, l))
    return {"passed": True, "detail": f"max|Δ|={worst[0]:.2e} ({worst[1]} layer {worst[2]})"}


def check_silu_product(outputs: dict, rtol: float = 1e-3, atol: float = 1e-4) -> dict:
    """SiLU(mlp_gate) · mlp_up reproduces mlp_activations: the card's steps 3 and 4 are the graph's own."""
    for name in ("mlp_gate", "mlp_up", "mlp_activations"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    g, u, a = outputs["mlp_gate"], outputs["mlp_up"], outputs["mlp_activations"]
    h = g / (1 + np.exp(-g)) * u
    ok = np.allclose(h, a, rtol=rtol, atol=atol)
    return {"passed": bool(ok), "detail": f"max|SiLU(g)·u − a| = {float(np.max(np.abs(h - a))):.2e}"}


def torch_step_reference(ids: np.ndarray) -> dict:
    """Last-position gate, up, MLP input and MLP output per layer, from hooks on the stock PyTorch model."""
    import torch
    model = _stock_torch_model()
    rec: dict = {}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        hooks += [
            layer.mlp.gate_proj.register_forward_hook(lambda m, a, o, l=l: rec.__setitem__(("mlp_gate", l), o[0, -1].detach())),
            layer.mlp.up_proj.register_forward_hook(lambda m, a, o, l=l: rec.__setitem__(("mlp_up", l), o[0, -1].detach())),
            layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a, l=l: rec.__setitem__(("mlp_in", l), a[0][0, -1].detach())),
            layer.mlp.register_forward_hook(lambda m, a, o, l=l: rec.__setitem__(("mlp_out", l), o[0, -1].detach())),
        ]
    with torch.no_grad():
        model(torch.tensor(ids))
    for h in hooks:
        h.remove()
    n = len(model.model.layers)
    return {k: np.stack([rec[(k, l)].numpy() for l in range(n)]).astype(np.float32) for k in STEP_OUTPUTS}


def write_browser_reference(path: Path, model_dir: Path, stock_sess) -> None:
    """Token ids and stock fp32 last-position logits for ARGMAX_PROMPTS, for browser-check.html."""
    cases = []
    for prompt in ARGMAX_PROMPTS:
        ids = _tokenize(model_dir, prompt)
        logits = stock_sess.run(["logits"], _feeds(stock_sess, ids))[0][0, -1]
        cases.append({"prompt": prompt, "ids": ids[0].tolist(), "logits": [float(v) for v in logits]})
    path.write_text(json.dumps({"cases": cases}))
```

In `publish.py`:

```python
def assert_manifest_matches_tag(manifest: dict, tag: str) -> None:
    """The app refuses a neuron manifest built for another tag, so publishing
    one would break the neuron and step-by-step cards."""
    if manifest.get("revision") != tag:
        raise ValueError(f"neurons/manifest.json was built for {manifest.get('revision')!r}, not {tag} — "
                         f"rebuild with: neurons --revision {tag}")
```

Call it in `run` right after `tag = tag_name(args)`: load `model_dir / "neurons" / "manifest.json"`, print the error and `return 1` on `ValueError`.

- [ ] **Step 4: Wire the checks into `validate.run`**

- `ref_steps = torch_step_reference(ref_ids)` next to `ref_acts`.
- In the variant loop: `quantized = variant != "model.onnx"`; `sess = _session(path, browser=quantized)`.
- After `greedy`: `checks[f"{variant}:argmax-parity"] = check_argmax_parity([(_last_logits(sess, model_dir, p), _last_logits(stock, model_dir, p)) for p in ARGMAX_PROMPTS])` with helper

```python
def _last_logits(sess, model_dir: Path, prompt: str) -> np.ndarray:
    ids = _tokenize(model_dir, prompt)
    return sess.run(["logits"], _feeds(sess, ids))[0][0, -1]
```

  For `model.onnx` keep the same check (it passes trivially and proves the helper).
- Step checks after `inv-rms-parity`: `step_abs, step_rel = (1e-3, 1e-3) if fp32 else (0.5, 0.25)`; `checks[f"{variant}:steps-parity"] = check_steps_parity(ro_out, ref_steps, step_abs, step_rel)`; `checks[f"{variant}:silu-product"] = check_silu_product(ro_out)`.
- Neuron votes: make it gating for every variant (`checks[f"{variant}:neuron-votes-sum"] = votes`) and replace the "informational" comment with: `# v6 computes on fp32 activations (weight-only 8-bit), so neuron votes from the int8 asset must add up to the MLP votes here too (#19)`.
- Update the comment above the quantized activation tolerances: weight-only 8-bit, measured worst layer ≈ 9 % of the layer's peak in the spike; keep `(0.5, 0.25, 0.1)`.
- `write_browser_reference(model_dir.parent / "browser-ref.json", model_dir, stock)` once, before the variant loop.
- `check_neuron_files(model_dir, texts, g, norms, eps)`: add parameters; after the `finalNormWeight` check, add

```python
    if "revision" not in m or m.get("rmsNormEps") != eps:
        return {"passed": False, "detail": f"manifest revision {m.get('revision')!r} / rmsNormEps {m.get('rmsNormEps')} (expected a revision and {eps})"}
    lay = m["layout"]
    for l in range(m["layers"]):
        raw = (ndir / f"layer-{l:02d}.bin").read_bytes()
        if not np.allclose(np.frombuffer(raw[lay["normWeight"]:lay["bytes"]], dtype="<f4"), norms[l], atol=1e-6):
            return {"passed": False, "detail": f"layer-{l:02d}.bin: normWeight differs from the stock post-attention norm"}
```

  and call it with `stock_mlp_norms()` and `AutoConfig.from_pretrained(STOCK_MODEL_ID).rms_norm_eps`. Update `test_validate_neurons.py` callers of `check_neuron_files` to pass `norms=[np.ones(dims)] * layers, eps=1e-5` and build manifests with `revision="v6"` (check failures there name what to change).
- Module docstring: add "12. argmax parity under browser semantics; mlp_gate/up/in/out match PyTorch hooks and SiLU(gate)·up = mlp_activations".

Update the model card in `publish.py`: add a paragraph after the neuron paragraph:

```
For the step-by-step card it also exposes `mlp_gate` and `mlp_up`
(`[batch, 30, 1536]`, every layer's gate before SiLU and up values at the
last position) and `mlp_in` and `mlp_out` (`[batch, 30, 576]`, the stream
going into each MLP and the MLP's edit). Each `neurons/` layer file ends with
that layer's post-attention norm weight; the manifest names the tag it was
built for (`revision`) and the norm's epsilon.
```

and replace the **Variants** paragraph with:

```
**Variants:** `onnx/model_quantized.onnx` (loaded with `dtype: 'q8'`) is
weight-only 8-bit: `MatMulNBits` with blocks of 32 weights, activations in
fp32, the embedding table as uint8, `/lm_head/MatMul` and the readout gather
in fp32. Under browser semantics (no graph fusion; onnxruntime-web WASM and
WebGPU) it picks the same next token as the fp32 export on all validation
prompts. The earlier dynamic-int8 variant (tags up to v5) rounded
activations with one scale per tensor and lost small values in browsers.
```

- [ ] **Step 5: Run to verify everything passes**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tools/export/src/tsumugi_export/validate.py tools/export/src/tsumugi_export/publish.py tools/export/tests/
git commit -m "feat(export): validate quantized models as browsers run them; step output checks; publish refuses a manifest for another tag

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Browser check page

**Files:**
- Create: `tools/export/browser-check.html`
- Modify: `tools/export/README.md`

**Interfaces:**
- Consumes: `tools/export/out/browser-ref.json` (Task 4), `tools/export/out/model/` (repo layout).
- Produces: `window.checkResult: Promise<string[]>` — one line per device and prompt, ending with `PASS` or `FAIL`.

- [ ] **Step 1: Write the page**

```html
<!doctype html>
<meta charset="utf-8">
<title>Tsumugi model browser check</title>
<pre id="log">running…</pre>
<!-- Served by the app's Vite dev server (npm run dev), opened at
     http://localhost:5173/tools/export/browser-check.html after `validate`
     wrote out/browser-ref.json. Loads out/model through transformers.js, as
     the app does, on WebGPU and WASM, and compares the last-position logits
     with the stock fp32 export's. -->
<script type="module">
import { AutoModelForCausalLM, Tensor, env } from '@huggingface/transformers'

env.allowRemoteModels = false
env.allowLocalModels = true
env.localModelPath = '/tools/export/out/'

const log = (s) => { document.getElementById('log').textContent += '\n' + s }

async function check() {
  const ref = await (await fetch('/tools/export/out/browser-ref.json')).json()
  const lines = []
  let ok = true
  for (const device of ['webgpu', 'wasm']) {
    if (device === 'webgpu' && !('gpu' in navigator)) { lines.push('webgpu: not available in this browser'); continue }
    const model = await AutoModelForCausalLM.from_pretrained('model', { dtype: 'q8', device })
    for (const c of ref.cases) {
      const n = c.ids.length
      const out = await model({
        input_ids: new Tensor('int64', BigInt64Array.from(c.ids.map(BigInt)), [1, n]),
        attention_mask: new Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
      })
      const V = c.logits.length
      const last = out.logits.data.subarray((n - 1) * V, n * V)
      let am = 0, bm = 0, worst = 0
      for (let k = 0; k < V; k++) {
        if (last[k] > last[am]) am = k
        if (c.logits[k] > c.logits[bm]) bm = k
        worst = Math.max(worst, Math.abs(last[k] - c.logits[k]))
      }
      const pass = am === bm && worst <= 1.5
      ok &&= pass
      lines.push(`${device}  ${c.prompt.padEnd(46)}  argmax ${am === bm ? 'same' : 'DIFF'}  max|Δlogit| ${worst.toFixed(2)}  ${pass ? 'PASS' : 'FAIL'}`)
    }
    await model.dispose()
  }
  lines.push(ok ? 'VERDICT: PASS' : 'VERDICT: FAIL')
  lines.forEach(log)
  return lines
}
window.checkResult = check().catch((e) => { log('ERROR ' + e); return ['ERROR ' + e] })
</script>
```

- [ ] **Step 2: Document it in the README**

In `tools/export/README.md`:
- Step 6 (neurons) becomes `uv run python -m tsumugi_export neurons --model-dir out/model --revision v6` with a sentence: "`--revision` is the tag you will publish under; `publish` refuses a manifest built for another tag, and the app refuses one that does not match `ATTN_MODEL_REVISION`."
- Step 7: add `*:argmax-parity`, `*:steps-parity`, `*:silu-product` to the must-PASS list; add: "Quantized variants are validated with graph optimizations disabled, because onnxruntime-web runs the unfused graph (#19)."
- New step 7b: "**Browser check.** `npm run dev` in the repository root, then open `http://localhost:5173/tools/export/browser-check.html` in Chrome. It loads `out/model` through transformers.js on WebGPU and WASM and must end with `VERDICT: PASS`. This is the pre-publish in-app check step 9 used to lack."
- Step 8: `--tag v6` in the example.

- [ ] **Step 3: Verify the page is served and its module resolves**

Run: `npx vite --port 5179 &` then `curl -s http://localhost:5179/tools/export/browser-check.html | grep -c "@vite/client\|checkResult"` and stop the server.
Expected: `2` or more (Vite injected its client and served the page). The full run happens in Task 6.

- [ ] **Step 4: Commit**

```bash
git add tools/export/browser-check.html tools/export/README.md
git commit -m "docs(export): browser check page and v6 release steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Build and validate v6 locally (operator run, no publishing)

**Files:** none committed (outputs under `tools/export/out/`, git-ignored).

- [ ] **Step 1: Export, quantize, build assets**

```bash
cd tools/export
uv run python -m tsumugi_export export --out out/model
uv run python -m tsumugi_export export --out out/model-nocache --no-cache
uv run python -m tsumugi_export quantize --model-dir out/model
uv run python -m tsumugi_export geometry --model-dir out/model
uv run python -m tsumugi_export neurons --model-dir out/model --revision v6
ls -la out/model/onnx
```

Expected: `model_quantized.onnx` ≈ 263 MB (between 250 and 280).

- [ ] **Step 2: Validate**

Run: `uv run python -m tsumugi_export validate --model-dir out/model 2>&1 | tee /tmp/validate-v6.txt | tail -60`
Expected: `VERDICT: PASS`; `model_quantized.onnx:argmax-parity` shows 4/4; `a-equiv-b` ran (not SKIPPED); `model_quantized.onnx:neuron-votes-sum` PASS.

If `neuron-votes-sum` fails for the quantized variant only: record the reported gap in the ledger, and rule — the spec expects no "≈"; if the gap is below 0.3 at its worst layer, make the quantized check informational again with a comment quoting the measured gap, and add the finding to the final report. Do not loosen fp32 tolerances.

- [ ] **Step 3: Browser check**

With `npm run dev` running at the repository root, open `http://localhost:5173/tools/export/browser-check.html` in the Playwright browser and evaluate `await window.checkResult`.
Expected: eight PASS lines (WebGPU and WASM × 4 prompts) and `VERDICT: PASS`. Save the output in the ledger.

- [ ] **Step 4: Record**

Append to the ledger: model size, the validate lines for argmax-parity, steps-parity, silu-product, neuron-votes-sum, and the browser-check verdict.

---

### Task 7: Run data — `steps` on the neuron data

**Files:**
- Modify: `src/trace/types.ts`, `src/trace/validate.ts`
- Test: `src/trace/validate.test.ts`

**Interfaces:**
- Produces:

```ts
export interface MlpSteps {
  gate: string     // float16 base64 [layers × neurons]: the gate before SiLU
  up: string       // float16 base64 [layers × neurons]
  mlpIn: string    // float16 base64 [layers × dims]: the stream going into each MLP
  mlpOut: string   // float16 base64 [layers × dims]: each MLP's edit
}
// NeuronData gains: steps?: MlpSteps   (v6 export)
export function stepProblems(s: unknown, layers: number, neurons: number, dims: number): string[]
```

- [ ] **Step 1: Write the failing tests**

In `validate.test.ts`, add (reusing the file's `neuronData` helper and `packHalf`):

```ts
const steps = (layers: number, neurons = 4, dims = 576): MlpSteps => ({
  gate: packHalf(Array.from({ length: layers * neurons }, (_, i) => i / 7 - 1)),
  up: packHalf(Array.from({ length: layers * neurons }, (_, i) => Math.sin(i))),
  mlpIn: packHalf(Array.from({ length: layers * dims }, (_, i) => Math.cos(i))),
  mlpOut: packHalf(Array.from({ length: layers * dims }, (_, i) => Math.sin(i) / 3)),
})

test('well-formed steps pass', () => {
  expect(stepProblems(steps(3), 3, 4, 576)).toEqual([])
})

test('steps with a wrong length or non-finite values are reported by field', () => {
  expect(stepProblems({ ...steps(3), up: packHalf([1, 2, 3]) }, 3, 4, 576)).toEqual(['steps.up holds 3 values, expected 3 × 4'])
  expect(stepProblems({ ...steps(3), mlpOut: packHalf(Array(3 * 576).fill(NaN)) }, 3, 4, 576)).toEqual(['steps.mlpOut values are not all finite'])
  expect(stepProblems(null, 3, 4, 576)).toEqual(['steps is not an object'])
})

test('an import drops only broken steps and keeps the neuron data', () => {
  const events = withNeurons(buildFixtureTrace({ mode: 'real', mlp: true, layers: 2 }),
    { ...neuronData(2), steps: { ...steps(2), gate: 'not base64!' } })
  const { events: out, dropped } = dropBrokenNeurons(events)
  const edits = out.filter((e) => e.type === 'edits') as Array<Extract<TraceEvent, { type: 'edits' }>>
  expect(dropped).toBe(edits.length)
  expect(edits.every((e) => e.neurons !== undefined && e.neurons.steps === undefined)).toBe(true)
  expect(validateTrace(out)).toEqual([])
})
```

`withNeurons(events, n)` is the existing helper at line ~177 that sets `neurons` on every edits event; if it has a different name, use that. Import `MlpSteps` and `stepProblems`.

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/trace/validate.test.ts`
Expected: FAIL (`stepProblems` is not exported).

- [ ] **Step 3: Implement**

`types.ts`: add `MlpSteps` (above) and `steps?: MlpSteps` on `NeuronData` with the comment `// v6 export: the step-by-step card's data; absent in older runs`; update the edits event comment to "v5 adds the last token's neuron data, v6 its MLP steps".

`validate.ts`:

```ts
export function stepProblems(s: unknown, layers: number, neurons: number, dims: number): string[] {
  if (typeof s !== 'object' || s === null) return ['steps is not an object']
  const d = s as Partial<Record<keyof MlpSteps, unknown>>
  const out: string[] = []
  const field = (k: keyof MlpSteps, width: number, label: string) => {
    try {
      const v = unpackHalf(String(d[k]))
      if (v.length !== layers * width) out.push(`steps.${k} holds ${v.length} values, expected ${layers} × ${label}`)
      else if (v.some((x) => !Number.isFinite(x))) out.push(`steps.${k} values are not all finite`)
    } catch { out.push(`steps.${k} is not base64 float16`) }
  }
  field('gate', neurons, String(neurons)); field('up', neurons, String(neurons))
  field('mlpIn', dims, String(dims)); field('mlpOut', dims, String(dims))
  return out
}
```

In `validateTrace`, after the `neuronProblems` loop: `if (e.neurons?.steps !== undefined) for (const p of stepProblems(e.neurons.steps, layerTotal, e.neurons.neurons, dims)) errs.push(\`edits cycle ${e.cycle} ${p}\`)`.

In `dropBrokenNeurons`, replace the map body's last part:

```ts
    if (e.type !== 'edits' || e.neurons === undefined) return e
    if (neuronProblems(e.neurons, layers, dims).length > 0) {
      dropped++
      const rest = { ...e }
      delete rest.neurons
      return rest
    }
    if (e.neurons.steps === undefined || stepProblems(e.neurons.steps, layers, e.neurons.neurons, dims).length === 0) return e
    dropped++
    const neurons = { ...e.neurons }
    delete neurons.steps
    return { ...e, neurons }
```

Update the comment above it: "…the broken fields are dropped (neuron data whole, or only its steps) and counted."

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/trace && npx tsc -b`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/trace
git commit -m "feat(trace): optional MLP step data on the neuron data, validated and dropped on its own

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Worker — extract and emit the steps

**Files:**
- Modify: `src/engine/transformers/neuronData.ts`, `src/engine/transformers/worker.ts`
- Test: `src/engine/transformers/neuronData.test.ts`

**Interfaces:**
- Consumes: `MlpSteps` (Task 7).
- Produces:

```ts
export interface StepOutputs { mlp_gate?: Tensorish; mlp_up?: Tensorish; mlp_in?: Tensorish; mlp_out?: Tensorish }
export type StepsResult = { status: 'absent' } | { status: 'bad-shape'; detail: string } | { status: 'ok'; steps: MlpSteps }
export function extractSteps(out: StepOutputs, layers: number, neurons: number, dims: number): StepsResult
```

- [ ] **Step 1: Write the failing tests**

```ts
const t = (dims: number[], fill: (i: number) => number = (i) => i / 10) =>
  ({ dims, data: Float32Array.from({ length: dims.reduce((a, b) => a * b, 1) }, (_, i) => fill(i)) })

test('step outputs become float16 fields, layer-major', () => {
  const r = extractSteps({ mlp_gate: t([1, 2, 3]), mlp_up: t([1, 2, 3], (i) => -i), mlp_in: t([1, 2, 4]), mlp_out: t([1, 2, 4], (i) => i * 2) }, 2, 3, 4)
  expect(r.status).toBe('ok')
  if (r.status !== 'ok') return
  expect(Array.from(unpackHalf(r.steps.up))).toEqual([0, -1, -2, -3, -4, -5])
  expect(unpackHalf(r.steps.mlpOut)).toHaveLength(8)
})

test('no step outputs is absent; some missing or a wrong shape is bad-shape', () => {
  expect(extractSteps({}, 2, 3, 4).status).toBe('absent')
  expect(extractSteps({ mlp_gate: t([1, 2, 3]) }, 2, 3, 4)).toEqual({ status: 'bad-shape', detail: expect.stringContaining('mlp_up false') })
  const r = extractSteps({ mlp_gate: t([1, 2, 3]), mlp_up: t([1, 2, 3]), mlp_in: t([1, 2, 5]), mlp_out: t([1, 2, 4]) }, 2, 3, 4)
  expect(r).toEqual({ status: 'bad-shape', detail: 'mlp_in has shape [1, 2, 5], expected [1, 2, 4]' })
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/engine/transformers/neuronData.test.ts`
Expected: FAIL (`extractSteps` not exported).

- [ ] **Step 3: Implement**

In `neuronData.ts` (import `MlpSteps`):

```ts
// The v6 export's step outputs for the step-by-step card, all last position:
//   mlp_gate, mlp_up [1, layers, neurons]; mlp_in, mlp_out [1, layers, dims]
export interface StepOutputs { mlp_gate?: Tensorish; mlp_up?: Tensorish; mlp_in?: Tensorish; mlp_out?: Tensorish }
export type StepsResult = { status: 'absent' } | { status: 'bad-shape'; detail: string } | { status: 'ok'; steps: MlpSteps }

export function extractSteps(out: StepOutputs, layers: number, neurons: number, dims: number): StepsResult {
  const { mlp_gate: g, mlp_up: u, mlp_in: x, mlp_out: e } = out
  if (!g && !u && !x && !e) return { status: 'absent' }
  if (!g || !u || !x || !e)
    return { status: 'bad-shape', detail: `step outputs incomplete: mlp_gate ${!!g}, mlp_up ${!!u}, mlp_in ${!!x}, mlp_out ${!!e}` }
  const want: Array<[string, Tensorish, number]> = [['mlp_gate', g, neurons], ['mlp_up', u, neurons], ['mlp_in', x, dims], ['mlp_out', e, dims]]
  for (const [name, t, w] of want)
    if (!sameDims(t, [1, layers, w]) || t.data.length !== layers * w)
      return { status: 'bad-shape', detail: `${name} has shape [${t.dims.join(', ')}], expected [1, ${layers}, ${w}]` }
  const half = (t: Tensorish) => packHalf(t.data as ArrayLike<number>)
  return { status: 'ok', steps: { gate: half(g), up: half(u), mlpIn: half(x), mlpOut: half(e) } }
}
```

In `worker.ts`: add `let stepsOff = false` next to `neuronsOff` with the comment `// step-by-step card (v6 export): flipped by a bad shape — never-fail, the neuron card keeps working`. Inside the `if (n.status === 'ok')` branch:

```ts
            if (n.status === 'ok') {
              neurons = n.data
              if (!stepsOff) {
                const s = extractSteps(out, numLayers, n.data.neurons, dims)
                if (s.status === 'ok') neurons = { ...n.data, steps: s.steps }
                else if (s.status === 'bad-shape') {
                  stepsOff = true
                  console.warn(`step outputs unusable (${s.detail}) — the step-by-step card shows its note`)
                }
              }
            }
```

Update the q8 comment at the model load (`// q8 is the attn repo's sole published variant…`) to: `// model_quantized.onnx (dtype 'q8') is the attn repo's sole published variant: from v6 weight-only 8-bit with fp32 activations, the same next token as fp32 in browsers (#19); see tools/export/src/tsumugi_export/quantize.py.`

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/engine && npx tsc -b`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/engine/transformers
git commit -m "feat(worker): pack the v6 step outputs onto the neuron data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Asset format v6 — norm weight, revision check, fixtures

**Files:**
- Modify: `src/neurons/format.ts`, `src/neurons/asset.ts`, `src/engine/tokenizer.ts`, `src/test/neuronFixture.ts`, `src/test/fixtures.ts`
- Test: `src/neurons/asset.test.ts`, `src/test/neuronFixture.test.ts` (new)

**Interfaces:**
- Produces: `LayerLayout.normWeight: number`; `NeuronManifest.revision: string`, `NeuronManifest.rmsNormEps: number`; `NeuronLayer.normWeight: Float32Array` ([dims]); `layerLayout(neurons, dims, labels, fractions)` returns `normWeight` and `bytes` grows by `dims * 4`; `ATTN_MODEL_REVISION = 'v6'`; fixture: `fixtureNormWeight(l, d)`, `fixtureSteps(cycle): MlpSteps`, and `fixtureNeuronData` returns data with `steps` whose `SiLU(gate) · up` equals the activations; `buildFixtureTrace({ neurons: true, steps: false })` omits steps (default true).

- [ ] **Step 1: Write the failing tests**

`asset.test.ts` additions (follow the file's existing fetch-stub pattern for serving the fixture files):

```ts
test('a manifest built for another revision is refused and evicted', async () => {
  const f = encodeNeuronFixture()
  stubNeuronFetch({ ...f.manifest, revision: 'v5' })          // the file's existing stub helper, with a manifest override
  await expect(loadNeuronAsset('https://x/neurons')).rejects.toThrow('manifest is for v5, the app needs v6')
})

test('a layer file carries its norm weight', async () => {
  const layer = fixtureNeuronLayer(1)
  expect(layer.normWeight).toHaveLength(NF.dims)
  expect(layer.normWeight[7]).toBeCloseTo(fixtureNormWeight(1, 7), 6)
})
```

If the file has no stub helper taking a manifest override, add one there that serves `neuronFixtureFile(name)` except `manifest.json`.

`src/test/neuronFixture.test.ts`:

```ts
import { expect, test } from 'vitest'
import { ATTN_MODEL_REVISION } from '../engine/tokenizer'
import { unpackHalf } from '../trace/neuronPack'
import { encodeNeuronFixture, fixtureNeuronData, NF } from './neuronFixture'

test('the fixture manifest is for the revision the app pins', () => {
  expect(encodeNeuronFixture().manifest.revision).toBe(ATTN_MODEL_REVISION)
})

test('fixture steps are consistent: SiLU(gate) · up is the activation', () => {
  const d = fixtureNeuronData(1, [100], () => Array(576).fill(0.1))
  const g = unpackHalf(d.steps!.gate), u = unpackHalf(d.steps!.up), a = unpackHalf(d.activations)
  for (let i = 0; i < NF.layers * NF.neurons; i++) expect(a[i]).toBeCloseTo((g[i] / (1 + Math.exp(-g[i]))) * u[i], 2)
  expect(unpackHalf(d.steps!.mlpIn)).toHaveLength(NF.layers * NF.dims)
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/neurons src/test/neuronFixture.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement**

`format.ts`:

```ts
export interface LayerLayout { scales: number; columns: number; labelIds: number; thresholds: number; normWeight: number; bytes: number }
// NeuronManifest gains:
  revision: string             // the model tag the asset was built for (v6 on)
  rmsNormEps: number           // the RMSNorm epsilon, for n = x / rms(x) ⊙ w
// NeuronLayer gains:
  normWeight: Float32Array     // [dims]: this layer's post-attention RMSNorm weight w

export function layerLayout(neurons: number, dims: number, labels: number, fractions: number): LayerLayout {
  const columns = neurons * 4
  const labelIds = columns + neurons * dims
  const thresholds = labelIds + neurons * 2 * labels * 2
  const normWeight = thresholds + neurons * 2 * fractions * 4
  return { scales: 0, columns, labelIds, thresholds, normWeight, bytes: normWeight + dims * 4 }
}
```

`parseNeuronLayer` adds `normWeight: new Float32Array(buf.slice(layout.normWeight, layout.normWeight + dims * 4))`.

`asset.ts` `fetchAsset`, after the `finalNormWeight` check:

```ts
    if (manifest.revision !== ATTN_MODEL_REVISION)
      throw new Error(`manifest is for ${manifest.revision ?? 'an unnamed revision'}, the app needs ${ATTN_MODEL_REVISION}`)
```

(import `ATTN_MODEL_REVISION` from `../engine/tokenizer`; the existing catch evicts both files).

`tokenizer.ts`: `export const ATTN_MODEL_REVISION = 'v6'` and extend the comment above it: "v6 = weight-only 8-bit model (#19) and the step outputs (mlp_gate, mlp_up, mlp_in, mlp_out); the neurons manifest names its revision."

`neuronFixture.ts`:
- `export const fixtureNormWeight = (l: number, d: number) => 1 + 0.25 * Math.sin(d * 0.2 + l)`; in `encodeNeuronFixture` write it at `layout.normWeight + d * 4` per layer; manifest gets `revision: 'v6', rmsNormEps: 1e-5`.
- Replace `fixtureNeuronData`'s activations with values derived from gate and up:

```ts
const silu = (z: number) => z / (1 + Math.exp(-z))
const fixtureGate = (cycle: number, k: number) => 4 * Math.cos(cycle + 0.7 * Math.floor(k / NF.neurons) + 1.1 * (k % NF.neurons)) + 0.5
const fixtureUp = (cycle: number, k: number) => 1.5 * Math.sin(cycle + 0.4 * k + 0.3)

export function fixtureSteps(cycle: number): MlpSteps {
  const n = NF.layers * NF.neurons, w = NF.layers * NF.dims
  return {
    gate: packHalf(Array.from({ length: n }, (_, k) => fixtureGate(cycle, k))),
    up: packHalf(Array.from({ length: n }, (_, k) => fixtureUp(cycle, k))),
    mlpIn: packHalf(Array.from({ length: w }, (_, k) => 20 * Math.cos(0.13 * k + cycle))),
    mlpOut: packHalf(Array.from({ length: w }, (_, k) => 6 * Math.sin(0.07 * k - cycle))),
  }
}

export function fixtureNeuronData(cycle: number, ids: number[], rowOf: (id: number) => number[], withSteps = true): NeuronData {
  const acts = Array.from({ length: NF.layers * NF.neurons }, (_, k) => silu(fixtureGate(cycle, k)) * fixtureUp(cycle, k))
  return { neurons: NF.neurons, invRms: FIXTURE_INV_RMS, activations: packHalf(acts),
    candidates: ids.map((id) => ({ id, row: packFloat32(rowOf(id)) })), ...(withSteps ? { steps: fixtureSteps(cycle) } : {}) }
}
```

`fixtures.ts`: add option `steps?: boolean` (comment: "with neurons: include the v6 step data (default true)") and pass `opts.steps ?? true` to `fixtureNeuronData`.

The fixture activations change, so tests that pinned exact neuron numbers may move. Run the suite; fix any test that hard-coded a fixture value by recomputing from the fixture helpers (do not weaken assertions).

- [ ] **Step 4: Run to verify everything passes**

Run: `npx vitest run && npx tsc -b`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/neurons src/engine/tokenizer.ts src/test
git commit -m "feat(neurons): v6 asset format — per-layer norm weight, manifest revision checked against the pinned tag

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: The steps math module and shared helpers

**Files:**
- Create: `src/neurons/steps.ts`, `src/neurons/steps.test.ts`
- Modify: `src/neurons/math.ts` (exports), `src/viz/mlpLayerSelection.ts`, `src/viz/NeuronCard.tsx`
- Test: `src/viz/mlpLayerSelection.test.ts`, `src/neurons/math.test.ts`

**Interfaces:**
- Consumes: `NeuronLayer.normWeight`, `MlpSteps` (Tasks 7, 9).
- Produces (`steps.ts`):

```ts
export interface DecodedSteps { gate: Float32Array; up: Float32Array; mlpIn: Float32Array; mlpOut: Float32Array }
export interface LayerSteps { x: Float32Array; edit: Float32Array; gate: Float32Array; up: Float32Array; h: Float32Array }
export const silu: (z: number) => number
export function decodeSteps(s: MlpSteps): DecodedSteps                        // cached per object
export function layerSteps(d: DecodedSteps, h: Float32Array, layer: number, neurons: number, dims: number): LayerSteps | null  // null when lengths don't fit
export function normalize(x: Float32Array, w: Float32Array, eps: number): { n: Float32Array; rms: number }
export const rms: (v: ArrayLike<number>) => number
export function gateDomain(gate: Float32Array): [number, number]
export function gateCounts(gate: Float32Array): { negative: number; positive: number }
export function productSummary(h: Float32Array, shown?: number): { order: number[]; half: number; ninety: number; restMax: number | null }
export function contribution(layer: NeuronLayer, dims: number, i: number, hi: number): Float32Array
export const editRatio: (edit: Float32Array, x: Float32Array) => number
```

- Produces (`math.ts`): `export function columnNorm(...)` (now exported), `export function neuronLabels(m, layer, i, sign): number[]` (the former private `labels`), `export function listedNeurons(m, layer, acts, mode: 'vote' | 'size', row: Float32Array | null, g, invRms): number[]`.
- Produces (`mlpLayerSelection.ts`):

```ts
export function effectiveMlpLayer(pick: MlpLayerSelection['pick'], run: TraceEvent | undefined,
  candidates: VotedCandidate[] | undefined, selected: number): number | null
export function useMlpLayerControl(candidates: VotedCandidate[] | undefined, run: TraceEvent | undefined):
  { layer: number | null; layers: number; selected: number; step(delta: number): void }
```

- [ ] **Step 1: Write the failing tests**

`src/neurons/steps.test.ts`:

```ts
import { expect, test } from 'vitest'
import { contribution, editRatio, gateCounts, gateDomain, normalize, productSummary, rms, silu } from './steps'
import type { NeuronLayer } from './format'

test('SiLU passes large values, squashes negatives to at least −0.28', () => {
  expect(silu(10)).toBeCloseTo(9.9995, 3)
  expect(silu(0)).toBe(0)
  expect(silu(-1.278)).toBeCloseTo(-0.2785, 3)
  expect(silu(-20)).toBeCloseTo(0, 6)
})

test('normalize divides by the root mean square (with eps) and applies the learned scale', () => {
  const { n, rms: r } = normalize(Float32Array.from([3, -4]), Float32Array.from([1, 2]), 0)
  expect(r).toBeCloseTo(Math.sqrt(12.5), 6)
  expect(Array.from(n).map((v) => +v.toFixed(4))).toEqual([+(3 / r).toFixed(4), +((-4 / r) * 2).toFixed(4)])
  expect(rms([3, -4])).toBeCloseTo(Math.sqrt(12.5), 6)
})

test('the gate axis always shows both sides of zero, padded', () => {
  const [lo, hi] = gateDomain(Float32Array.from([-4.1, -2, 1.7]))
  expect(lo).toBeLessThan(-4.1); expect(hi).toBeGreaterThan(1.7)
  const [lo2, hi2] = gateDomain(Float32Array.from([-3, -2.5]))     // all negative: zero and +1 still on the axis
  expect(lo2).toBeLessThan(-3); expect(hi2).toBeGreaterThanOrEqual(1)
  expect(gateCounts(Float32Array.from([-1, 0.5, -0.2, 3]))).toEqual({ negative: 2, positive: 2 })
})

test('the product summary orders by size and counts the neurons carrying half and 90 % of the square', () => {
  const h = Float32Array.from([0.1, -3, 0.2, 4, 1, -0.05])   // squares 0.01 9 0.04 16 1 0.0025, total 26.0525
  const s = productSummary(h, 3)
  expect(s.order).toEqual([3, 1, 4])
  expect(s.half).toBe(1)        // 16 / 26.05 = 0.61
  expect(s.ninety).toBe(2)      // 25 / 26.05 = 0.96
  expect(s.restMax).toBeCloseTo(0.2, 6)
  expect(productSummary(h, 10).restMax).toBeNull()
})

test('a contribution is the dequantized column times the activation', () => {
  const layer = { scales: Float32Array.from([0.5, 2]), columns: Int8Array.from([1, -2, 3, 4]),
    labelIds: new Uint16Array(), thresholds: new Float32Array(), normWeight: new Float32Array(2) } as NeuronLayer
  expect(Array.from(contribution(layer, 2, 1, -1))).toEqual([-6, -8])
  expect(editRatio(Float32Array.from([3, 4]), Float32Array.from([0, 10]))).toBeCloseTo(0.5, 6)
})
```

`mlpLayerSelection.test.ts` additions:

```ts
test('the effective layer is the pick for this run, else the largest MLP vote of the selected candidate', () => {
  const run = buildFixtureTrace()[0]
  const cands = [{ votes: { embed: 0, attn: [0, 0, 0], mlp: [1, -5, 2] } }, { votes: { embed: 0, attn: [0, 0, 0], mlp: [9, 0, 0] } }] as unknown as VotedCandidate[]
  expect(effectiveMlpLayer(null, run, cands, 0)).toBe(1)
  expect(effectiveMlpLayer(null, run, cands, 1)).toBe(0)
  expect(effectiveMlpLayer({ run, layer: 2 }, run, cands, 0)).toBe(2)
  expect(effectiveMlpLayer({ run: buildFixtureTrace()[0], layer: 2 }, run, cands, 0)).toBe(1)   // another run's pick
  expect(effectiveMlpLayer(null, run, undefined, 0)).toBeNull()
})
```

(Use the exact `LayerVotes` field names from `src/trace/types.ts`; adjust the literal if they differ.)

`math.test.ts` addition:

```ts
test('listedNeurons returns the neurons the card lists in each mode', () => {
  const asset = fixtureNeuronAsset(), layer = fixtureNeuronLayer(0)
  const acts = Float32Array.from({ length: NF.neurons }, (_, i) => (i - 4) / 2)
  const row = Float32Array.from({ length: NF.dims }, (_, d) => Math.sin(d))
  expect(listedNeurons(asset.manifest, layer, acts, 'vote', row, asset.g, 0.05))
    .toEqual(behindTheVote(asset.manifest, layer, acts, row, asset.g, 0.05).rows.map((r) => r.neuron))
  expect(listedNeurons(asset.manifest, layer, acts, 'size', null, asset.g, 0.05))
    .toEqual(firingMost(asset.manifest, layer, acts).map((r) => r.neuron))
  expect(listedNeurons(asset.manifest, layer, acts, 'vote', null, asset.g, 0.05)).toEqual([])
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/neurons src/viz/mlpLayerSelection.test.ts`
Expected: FAIL (missing modules/exports).

- [ ] **Step 3: Implement `steps.ts`**

```ts
import { unpackHalf } from '../trace/neuronPack'
import type { MlpSteps } from '../trace/types'
import type { NeuronLayer } from './format'

// The numbers behind "The MLP, step by step" (#17), in the glossary's notation:
//   n = x / rms(x) ⊙ w,  g = W_gate · n,  u = W_up · n,  h = SiLU(g) ⊙ u,  edit = W_down · h
// g, u, x and the edit come from the model (v6 outputs); h is mlp_activations.

export interface DecodedSteps { gate: Float32Array; up: Float32Array; mlpIn: Float32Array; mlpOut: Float32Array }
export interface LayerSteps { x: Float32Array; edit: Float32Array; gate: Float32Array; up: Float32Array; h: Float32Array }

export const silu = (z: number) => z / (1 + Math.exp(-z))

const decoded = new WeakMap<MlpSteps, DecodedSteps>()
export function decodeSteps(s: MlpSteps): DecodedSteps {
  let d = decoded.get(s)
  if (!d) {
    d = { gate: unpackHalf(s.gate), up: unpackHalf(s.up), mlpIn: unpackHalf(s.mlpIn), mlpOut: unpackHalf(s.mlpOut) }
    decoded.set(s, d)
  }
  return d
}

const part = (a: Float32Array, l: number, w: number) => a.subarray(l * w, (l + 1) * w)

export function layerSteps(d: DecodedSteps, h: Float32Array, layer: number, neurons: number, dims: number): LayerSteps | null {
  const layers = h.length / neurons
  if (!Number.isInteger(layers) || layer >= layers || d.gate.length !== h.length || d.up.length !== h.length
    || d.mlpIn.length !== layers * dims || d.mlpOut.length !== layers * dims) return null
  return { x: part(d.mlpIn, layer, dims), edit: part(d.mlpOut, layer, dims), gate: part(d.gate, layer, neurons),
    up: part(d.up, layer, neurons), h: part(h, layer, neurons) }
}

export function rms(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s / (v.length || 1))
}

export function normalize(x: Float32Array, w: Float32Array, eps: number): { n: Float32Array; rms: number } {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] * x[i]
  const r = Math.sqrt(s / x.length + eps)
  return { n: Float32Array.from(x, (v, i) => (v / r) * w[i]), rms: r }
}

// Both sides of zero always show (early layers are almost all negative), plus 4 % padding.
export function gateDomain(gate: Float32Array): [number, number] {
  let lo = -1, hi = 1
  for (const v of gate) { if (v < lo) lo = v; if (v > hi) hi = v }
  const pad = (hi - lo) * 0.04
  return [lo - pad, hi + pad]
}

export function gateCounts(gate: Float32Array): { negative: number; positive: number } {
  let negative = 0
  for (const v of gate) if (v < 0) negative++
  return { negative, positive: gate.length - negative }
}

// Neurons by |h|, largest first: the first `shown`, how many carry half and 90 %
// of Σh², and the largest |h| left out (null when nothing is left out).
export function productSummary(h: Float32Array, shown = 120) {
  const all = Array.from(h.keys()).sort((a, b) => Math.abs(h[b]) - Math.abs(h[a]))
  const total = all.reduce((s, i) => s + h[i] * h[i], 0) || 1
  let cum = 0, half = 0, ninety = 0
  for (let k = 0; k < all.length; k++) {
    cum += h[all[k]] ** 2
    if (!half && cum / total >= 0.5) half = k + 1
    if (!ninety && cum / total >= 0.9) { ninety = k + 1; break }
  }
  return { order: all.slice(0, shown), half, ninety, restMax: all.length > shown ? Math.abs(h[all[shown]]) : null }
}

export function contribution(layer: NeuronLayer, dims: number, i: number, hi: number): Float32Array {
  const s = layer.scales[i] * hi
  return Float32Array.from({ length: dims }, (_, d) => layer.columns[i * dims + d] * s)
}

const norm = (v: Float32Array) => Math.sqrt(v.reduce((s, x) => s + x * x, 0))
export const editRatio = (edit: Float32Array, x: Float32Array) => norm(edit) / (norm(x) || 1)
```

In `math.ts`: export `columnNorm`; rename `labels` to exported `neuronLabels` (update its two callers); add

```ts
// The neurons "Inside one MLP" lists for this mode — the step-by-step card marks the same ones.
export function listedNeurons(m: NeuronManifest, layer: NeuronLayer, acts: Float32Array, mode: 'vote' | 'size',
  row: Float32Array | null, g: Float32Array, invRms: number): number[] {
  if (mode === 'size') return firingMost(m, layer, acts).map((r) => r.neuron)
  return row ? behindTheVote(m, layer, acts, row, g, invRms).rows.map((r) => r.neuron) : []
}
```

- [ ] **Step 4: Implement the shared layer helpers and use them in the neuron card**

`mlpLayerSelection.ts` (keep existing exports):

```ts
import { selectedCandidate, useCandidateSelection } from './candidateSelection'
import { topMlpLayers } from './mlpMath'
import type { VotedCandidate } from './selectors'

// The MLP layer the neuron card, the step-by-step card and the navigation bar
// show: this run's pick, else the layer with the largest MLP vote for the
// selected candidate. One rule, so the three always agree.
export function effectiveMlpLayer(pick: MlpLayerSelection['pick'], run: TraceEvent | undefined,
  candidates: VotedCandidate[] | undefined, selected: number): number | null {
  if (!candidates || candidates.length === 0) return null
  return selectedLayer(pick, run) ?? topMlpLayers(candidates[Math.min(selected, candidates.length - 1)].votes, 1)[0].layer
}

export function useMlpLayerControl(candidates: VotedCandidate[] | undefined, run: TraceEvent | undefined) {
  const pick = useMlpLayer((s) => s.pick)
  const select = useMlpLayer((s) => s.select)
  const candPick = useCandidateSelection((s) => s.pick)
  const selected = candidates && candidates.length > 0 ? selectedCandidate(candPick, candidates) : 0
  const layer = effectiveMlpLayer(pick, run, candidates, selected)
  const layers = candidates?.[0]?.votes.mlp.length ?? 0
  const step = (d: number) => { if (run && layer !== null) select(run, Math.min(layers - 1, Math.max(0, layer + d))) }
  return { layer, layers, selected, step }
}
```

Export the `MlpLayerSelection` interface type if the signature needs it. If importing `selectors.ts` here creates an import cycle that breaks tests, move `VotedCandidate` usage to `import type` (types only) — it already is.

`NeuronCard.tsx`: replace the `pick/layerPick/selectLayer/sel/layers/layer/step` lines with

```ts
  const ready = data && candidates && candidates.length > 0
  const { layer, layers, selected: sel, step } = useMlpLayerControl(ready ? candidates : undefined, run)
```

and drop the now-unused imports. Behaviour is unchanged; the existing NeuronCard tests prove it.

- [ ] **Step 5: Run to verify everything passes**

Run: `npx vitest run && npx tsc -b && npx oxlint`
Expected: pass; oxlint shows only the two known warnings.

- [ ] **Step 6: Commit**

```bash
git add src/neurons src/viz/mlpLayerSelection.ts src/viz/mlpLayerSelection.test.ts src/viz/NeuronCard.tsx
git commit -m "feat(neurons): step-by-step math; one shared rule for the MLP layer on screen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: The five charts

**Files:**
- Create: `src/viz/MlpStepCharts.tsx`, `src/viz/MlpStepCharts.test.tsx`
- Modify: `src/index.css`

**Interfaces:**
- Consumes: `cellFill` from `./stripMath`; `gateDomain`, `silu`, `productSummary` results from Task 10.
- Produces (all exported from `MlpStepCharts.tsx`):

```ts
export const CHART_W = 720
// x, y: the hovered mark in viewBox units; height: the chart's viewBox height (to place the box next to the mark)
export interface Hover { neuron: number; chart: 'scatter' | 'silu' | 'product'; x: number; y: number; height: number }
export function StepStrip(p: { values: Float32Array; lim: number; label: string; testId: string }): JSX.Element
export function GateUpScatter(p: { gate: Float32Array; up: Float32Array; domain: [number, number]; marked: number[];
  named: number[]; hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode }): JSX.Element
export function SiluChart(p: { gate: Float32Array; domain: [number, number]; marked: number[]; counts: { negative: number; positive: number };
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode }): JSX.Element
export function ProductBars(p: { h: Float32Array; summary: ReturnType<typeof productSummary>; marked: number[];
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode }): JSX.Element
export const gateX: (domain: [number, number]) => (v: number) => number   // shared by scatter and SiLU
```

Layout constants (viewBox units): scatter `720 × 210`, SiLU `720 × 150`, product `720 × 150`; plot x range `X0 = 36` … `X1 = 710`; strips `viewBox="0 0 {n} 1"` with `preserveAspectRatio="none"`, drawn 14 px tall.

- [ ] **Step 1: Write the failing tests**

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { CHART_W, GateUpScatter, ProductBars, SiluChart, StepStrip, gateX } from './MlpStepCharts'
import { productSummary } from '../neurons/steps'

afterEach(cleanup)
const gate = Float32Array.from([-3, -0.5, 0.2, 6, 11])
const up = Float32Array.from([1, -2, 0.5, -1.5, 2])
const h = Float32Array.from(gate, (g, i) => (g / (1 + Math.exp(-g))) * up[i])

test('every chart is a full-width SVG on the shared 720-unit frame', () => {
  render(<>
    <GateUpScatter gate={gate} up={up} domain={[-4, 12]} marked={[4]} named={[4]} hover={null} onHover={() => {}} box={null} />
    <SiluChart gate={gate} domain={[-4, 12]} marked={[4]} counts={{ negative: 2, positive: 3 }} hover={null} onHover={() => {}} box={null} />
    <ProductBars h={h} summary={productSummary(h)} marked={[4]} hover={null} onHover={() => {}} box={null} />
  </>)
  for (const id of ['steps-scatter', 'steps-silu', 'steps-product']) {
    const svg = screen.getByTestId(id)
    expect(svg.getAttribute('viewBox')).toMatch(new RegExp(`^0 0 ${CHART_W} \\d+$`))
    expect(svg).toHaveClass('mlp-chart')
  }
})

test('scatter and SiLU place a neuron at the same x', () => {
  render(<>
    <GateUpScatter gate={gate} up={up} domain={[-4, 12]} marked={[]} named={[]} hover={null} onHover={() => {}} box={null} />
    <SiluChart gate={gate} domain={[-4, 12]} marked={[]} counts={{ negative: 2, positive: 3 }} hover={null} onHover={() => {}} box={null} />
  </>)
  const x = (id: string) => screen.getByTestId(id).querySelector('[data-neuron="3"]')!.getAttribute('cx')
  expect(x('steps-scatter')).toBe(x('steps-silu'))
  expect(Number(x('steps-scatter'))).toBeCloseTo(gateX([-4, 12])(6), 3)
})

test('marked neurons are drawn in the accent and are focusable; others are not', () => {
  render(<GateUpScatter gate={gate} up={up} domain={[-4, 12]} marked={[4, 1]} named={[4]} hover={null} onHover={() => {}} box={null} />)
  const dot = (i: number) => screen.getByTestId('steps-scatter').querySelector(`[data-neuron="${i}"]`)!
  expect(dot(4)).toHaveClass('step-dot-marked')
  expect(dot(4).getAttribute('tabindex')).toBe('0')
  expect(dot(0).getAttribute('tabindex')).toBeNull()
  expect(screen.getByTestId('steps-scatter')).toHaveTextContent('N4')
})

test('hovering a dot or bar reports the neuron and the chart', () => {
  const onHover = vi.fn()
  render(<ProductBars h={h} summary={productSummary(h)} marked={[]} hover={null} onHover={onHover} box={null} />)
  fireEvent.mouseEnter(screen.getByTestId('steps-product').querySelector('[data-neuron="4"]')!)
  expect(onHover).toHaveBeenCalledWith(expect.objectContaining({ neuron: 4, chart: 'product', height: 150 }))
})

test('the product chart marks the half and 90 % counts and names what it leaves out', () => {
  const many = Float32Array.from({ length: 200 }, (_, i) => 10 / (i + 1))
  render(<ProductBars h={many} summary={productSummary(many)} marked={[]} hover={null} onHover={() => {}} box={null} />)
  const svg = screen.getByTestId('steps-product')
  expect(svg.querySelectorAll('[data-neuron]')).toHaveLength(120)
  expect(svg).toHaveTextContent(/\d+ neurons carry half/)
  expect(svg).toHaveTextContent(/the other 80 are all below 0\.08/)
})

test('a strip stretches one cell per value across the full width', () => {
  render(<StepStrip values={Float32Array.from([1, -1, 0])} lim={1} label="x, into the MLP" testId="strip-x" />)
  const svg = screen.getByTestId('strip-x')
  expect(svg.getAttribute('viewBox')).toBe('0 0 3 1')
  expect(svg.getAttribute('preserveAspectRatio')).toBe('none')
  expect(svg.querySelectorAll('rect')).toHaveLength(3)
  expect(screen.getByText('x, into the MLP')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/MlpStepCharts.test.tsx`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```tsx
import type { ReactNode } from 'react'
import type { productSummary } from '../neurons/steps'
import { silu } from '../neurons/steps'
import { cellFill } from './stripMath'

// The step-by-step card's charts (#17). Every chart is an SVG on a 720-unit-wide
// frame drawn at width 100 %, so it fills the card's chart column (the #15
// sizing rule). The gate axis is shared: a neuron sits at the same x in the
// scatter and on the SiLU curve.

export const CHART_W = 720
const X0 = 36, X1 = 710
export interface Hover { neuron: number; chart: 'scatter' | 'silu' | 'product'; x: number; y: number; height: number }

export const gateX = ([lo, hi]: [number, number]) => (v: number) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0)
const clamp = (v: number, lim: number) => Math.max(-lim, Math.min(lim, v))
const ticks = ([lo, hi]: [number, number]) => {
  const step = hi - lo > 12 ? 5 : hi - lo > 5 ? 2 : 1
  const out: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push(t)
  return out
}

export function StepStrip({ values, lim, label, testId }: { values: Float32Array; lim: number; label: string; testId: string }) {
  return (
    <div className="step-strip-row">
      <span className="step-strip-label">{label}</span>
      <svg data-testid={testId} className="step-strip" viewBox={`0 0 ${values.length} 1`} preserveAspectRatio="none" role="img" aria-label={label}>
        {Array.from(values, (v, i) => <rect key={i} x={i} y={0} width={1.02} height={1} fill={cellFill(clamp(v, lim), lim)} />)}
      </svg>
    </div>
  )
}

function Dots({ xs, ys, marked, chart, height, onHover }: { xs: number[]; ys: number[]; marked: number[]; chart: Hover['chart']; height: number; onHover(h: Hover | null): void }) {
  const m = new Set(marked)
  const show = (i: number) => onHover({ neuron: i, chart, x: xs[i], y: ys[i], height })
  const dot = (i: number) => (
    <circle key={i} data-neuron={i} cx={xs[i]} cy={ys[i]} r={m.has(i) ? 4 : 1.6}
      className={m.has(i) ? 'step-dot-marked' : 'step-dot'} tabIndex={m.has(i) ? 0 : undefined}
      onMouseEnter={() => show(i)} onMouseLeave={() => onHover(null)}
      onFocus={() => show(i)} onBlur={() => onHover(null)} />
  )
  return <>{xs.map((_, i) => (m.has(i) ? null : dot(i)))}{marked.map(dot)}</>
}

function Frame({ testId, h, label, box, children }: { testId: string; h: number; label: string; box: ReactNode; children: ReactNode }) {
  return (
    <div className="step-chart">
      <svg data-testid={testId} className="mlp-chart" viewBox={`0 0 ${CHART_W} ${h}`} role="img" aria-label={label}>{children}</svg>
      {box}
    </div>
  )
}

export function GateUpScatter({ gate, up, domain, marked, named, onHover, box }: {
  gate: Float32Array; up: Float32Array; domain: [number, number]; marked: number[]; named: number[]
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode
}) {
  const H = 210, top = 10, bottom = H - 20
  const x = gateX(domain)
  let lo = -1, hi = 1
  for (const v of up) { if (v < lo) lo = v; if (v > hi) hi = v }
  const pad = (hi - lo) * 0.06
  const y = (v: number) => top + ((hi + pad - v) / (hi - lo + 2 * pad)) * (bottom - top)
  return (
    <Frame testId="steps-scatter" h={H} label="gate against up, one dot per neuron" box={box}>
      <rect x={x(domain[0])} y={top} width={x(0) - x(domain[0])} height={bottom - top} className="step-squash" />
      <line x1={x(0)} x2={x(0)} y1={top} y2={bottom} className="step-zero" />
      <line x1={X0} x2={X1} y1={y(0)} y2={y(0)} className="step-zero" />
      <text x={x(domain[0]) + 6} y={top + 14} className="step-note">g &lt; 0: SiLU will squash these</text>
      <text x={X1 - 4} y={top + 14} textAnchor="end" className="step-axis">u ↑ · g →</text>
      {ticks(domain).map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="step-axis">{t}</text>)}
      {ticks([lo, hi]).map((t) => <text key={t} x={X0 - 6} y={y(t) + 3} textAnchor="end" className="step-axis">{t}</text>)}
      <Dots xs={Array.from(gate, x)} ys={Array.from(up, y)} marked={marked} chart="scatter" height={H} onHover={onHover} />
      {named.map((i) => <text key={i} x={x(gate[i]) + 6} y={y(up[i]) + 4} className="step-name">N{i}</text>)}
    </Frame>
  )
}

export function SiluChart({ gate, domain, marked, counts, onHover, box }: {
  gate: Float32Array; domain: [number, number]; marked: number[]; counts: { negative: number; positive: number }
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode
}) {
  const H = 150, top = 8, bottom = H - 20
  const x = gateX(domain)
  const yMax = Math.max(1, silu(domain[1]))
  const y = (v: number) => top + ((yMax - v) / (yMax + 0.6)) * (bottom - top)
  const curve = Array.from({ length: 241 }, (_, k) => domain[0] + (k / 240) * (domain[1] - domain[0]))
    .map((v) => `${x(v).toFixed(1)},${y(silu(v)).toFixed(1)}`).join(' ')
  return (
    <Frame testId="steps-silu" h={H} label="SiLU of the gate, one dot per neuron" box={box}>
      <rect x={x(domain[0])} y={top} width={x(0) - x(domain[0])} height={bottom - top} className="step-squash" />
      <line x1={X0} x2={X1} y1={y(0)} y2={y(0)} className="step-zero" />
      <line x1={x(0)} x2={x(0)} y1={top} y2={bottom} className="step-zero" />
      <polyline points={curve} className="step-curve" />
      <text x={x(domain[0]) + 6} y={y(0) - 22} className="step-note">{counts.negative} negative gates</text>
      <text x={x(domain[0]) + 6} y={y(0) - 10} className="step-note">→ between −0.28 and 0</text>
      <text x={X1 - 4} y={top + 12} textAnchor="end" className="step-axis">{counts.positive} positive gates pass almost unchanged</text>
      {ticks(domain).map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="step-axis">{t}</text>)}
      <Dots xs={Array.from(gate, x)} ys={Array.from(gate, (g) => y(silu(g)))} marked={marked} chart="silu" height={H} onHover={onHover} />
    </Frame>
  )
}

export function ProductBars({ h, summary, marked, onHover, box }: {
  h: Float32Array; summary: ReturnType<typeof productSummary>; marked: number[]
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode
}) {
  const H = 150, mid = H / 2, reach = mid - 14
  const { order, half, ninety, restMax } = summary
  const peak = Math.abs(h[order[0]] ?? 0) || 1
  const bw = (X1 - X0) / Math.max(order.length, 1)
  const m = new Set(marked)
  const at = (k: number) => X0 + k * bw
  return (
    <Frame testId="steps-product" h={H} label="the largest activations, largest first" box={box}>
      <line x1={X0} x2={X1} y1={mid} y2={mid} className="step-zero" />
      {order.map((i, k) => {
        const len = (Math.abs(h[i]) / peak) * reach
        return <rect key={i} data-neuron={i} x={at(k)} width={Math.max(bw - 1, 0.6)} y={h[i] >= 0 ? mid - len : mid} height={len}
          className={m.has(i) ? 'step-bar-marked' : h[i] >= 0 ? 'step-bar-pos' : 'step-bar-neg'}
          onMouseEnter={() => onHover({ neuron: i, chart: 'product', x: at(k) + bw / 2, y: h[i] >= 0 ? mid - len : mid + len, height: H })}
          onMouseLeave={() => onHover(null)} />
      })}
      {[[half, 'half', 16], [ninety, '90%', 30]].map(([k, name, ty]) => Number(k) > 0 && Number(k) <= order.length && (
        <g key={String(name)}>
          <line x1={at(Number(k))} x2={at(Number(k))} y1={6} y2={H - 6} className="step-cut" />
          <text x={at(Number(k)) + 3} y={Number(ty)} className="step-note">{k} neurons carry {name}</text>
        </g>
      ))}
      <text x={X0 - 6} y={mid - reach + 4} textAnchor="end" className="step-axis">+{peak.toFixed(0)}</text>
      <text x={X0 - 6} y={mid + reach} textAnchor="end" className="step-axis">−{peak.toFixed(0)}</text>
      <text x={X1} y={H - 6} textAnchor="end" className="step-axis">
        the {order.length} largest, largest first →{restMax !== null && ` · the other ${h.length - order.length} are all below ${restMax < 0.1 ? restMax.toFixed(2) : restMax.toFixed(1)}`}</text>
    </Frame>
  )
}
```

CSS (append to `index.css`, near the `.mlp-card` rules):

```css
:root { --mark: #d08a1c; }
.step-row { display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 12px; padding: .7rem 0; border-top: 1px solid var(--line); }
.step-row h5 { margin: 0 0 .25rem; font-size: .85rem; color: var(--ai-deep); }
.step-row .step-text { margin: .3rem 0 0; font-size: .74rem; color: var(--ink-soft); line-height: 1.35; }
.step-charts { min-width: 0; overflow-x: auto; }
.step-charts > * { min-width: 520px; }
.step-chart { position: relative; }
.step-strip-row { display: grid; grid-template-columns: 150px minmax(0, 1fr); align-items: center; gap: 8px; margin: 3px 0; }
.step-strip-label { font-size: .72rem; color: var(--ink-soft); white-space: nowrap; }
.step-strip { display: block; width: 100%; height: 14px; border: 1px solid var(--line); }
.step-dot { fill: var(--ink-faint); fill-opacity: .45; }
.step-dot-marked, .step-bar-marked { fill: var(--mark); stroke: var(--card); stroke-width: 1; }
.step-dot-marked:focus { outline: none; stroke: var(--ai-deep); stroke-width: 2; }
.step-bar-pos { fill: var(--ai); } .step-bar-neg { fill: var(--shu); }
.step-squash { fill: var(--dye-2); fill-opacity: .35; }
.step-zero { stroke: var(--line-strong); stroke-dasharray: 3 3; }
.step-cut { stroke: var(--ink); stroke-dasharray: 2 2; }
.step-curve { fill: none; stroke: var(--ink); stroke-width: 1.5; }
.step-axis { font-size: 10px; fill: var(--ink-soft); }
.step-note { font-size: 10px; fill: #8a7a5a; }
.step-name { font-size: 10px; fill: var(--ink); }
@media (max-width: 700px) { .step-row { grid-template-columns: minmax(0, 1fr); } }
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/viz/MlpStepCharts.test.tsx && npx tsc -b && npx oxlint`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/viz/MlpStepCharts.tsx src/viz/MlpStepCharts.test.tsx src/index.css
git commit -m "feat(viz): the step-by-step card's charts on a shared full-width frame

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: The card, its glossary sections, and the Layers detail

**Files:**
- Create: `src/viz/MlpStepsCard.tsx`, `src/viz/MlpStepsCard.test.tsx`
- Modify: `src/viz/details/LayersDetail.tsx`, `src/glossary/callouts.ts`, `src/glossary/entries/mlp.tsx`, `src/glossary/registry.test.tsx`, `src/index.css`

**Interfaces:**
- Consumes: Tasks 9–11 (`useNeuronLayer`, `decodeSteps`, `layerSteps`, `normalize`, `rms`, `gateDomain`, `gateCounts`, `productSummary`, `contribution`, `editRatio`, `listedNeurons`, `neuronLabels`, `columnNorm`, `useMlpLayerControl`, `useNeuronMode`, charts).
- Produces: `MlpStepsCard({ data?: NeuronData; candidates?: VotedCandidate[]; note: string; run?: TraceEvent })`, test ids `steps-card`, `steps-title`, `steps-prev`, `steps-next`, `steps-note`, `steps-loading`, `steps-error`, `steps-retry`, `steps-box`, `steps-ratio`, `steps-callout`, strips `strip-x`, `strip-w`, `strip-n`, `strip-c{neuron}`, `strip-edit`; `MLP_STEP_CALLOUTS`; glossary `mlp` sections `silu-not-switch`, `two-projections`, `early-layers`; exported notes `STEPS_SIM_NOTE`, `STEPS_OLD_NOTE` from `LayersDetail.tsx`.

- [ ] **Step 1: Write the failing tests**

`MlpStepsCard.test.tsx` (mirror `NeuronCard.test.tsx`'s setup: same imports, `vi.spyOn(asset, 'loadNeuronAsset')` and `loadNeuronLayer` with the fixtures, `events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })`, `data`, `candidates`, `run`):

```tsx
const card = (props: Partial<Parameters<typeof MlpStepsCard>[0]> = {}) =>
  render(<MlpStepsCard data={data} candidates={candidates} note="the note" run={run} {...props} />)

test('without step data it shows the note it is given', () => {
  card({ data: { ...data, steps: undefined } })
  expect(screen.getByTestId('steps-note')).toHaveTextContent('the note')
})

test('opens on the same layer as Inside one MLP and follows the shared stepper', async () => {
  render(<><NeuronCard data={data} candidates={candidates} note="n" run={run} /><MlpStepsCard data={data} candidates={candidates} note="n" run={run} /></>)
  const layerOf = (id: string) => screen.getByTestId(id).textContent!.match(/layer (\d+)/)![1]
  expect(layerOf('steps-title')).toBe(layerOf('neuron-title'))
  fireEvent.click(screen.getByTestId(Number(layerOf('steps-title')) > 0 ? 'steps-prev' : 'steps-next'))
  expect(layerOf('steps-title')).toBe(layerOf('neuron-title'))
})

test('draws all five steps once the layer file is loaded', async () => {
  card()
  expect(screen.getByTestId('steps-scatter')).toBeInTheDocument()          // steps 2–4 need no layer file
  expect(screen.getByTestId('steps-loading')).toBeInTheDocument()
  await waitFor(() => expect(screen.getByTestId('strip-n')).toBeInTheDocument())
  for (const id of ['strip-x', 'strip-w', 'steps-silu', 'steps-product', 'strip-edit']) expect(screen.getByTestId(id)).toBeInTheDocument()
  expect(screen.getByTestId('steps-card')).toHaveTextContent(`+ ${NF.neurons - 3} more`)
})

test('the ratio line equals ‖edit‖ / ‖x‖ of this layer', async () => {
  card()
  await waitFor(() => screen.getByTestId('steps-ratio'))
  const layer = Number(screen.getByTestId('steps-title').textContent!.match(/layer (\d+)/)![1])
  const d = decodeSteps(data.steps!)
  const s = layerSteps(d, unpackHalf(data.activations), layer, NF.neurons, NF.dims)!
  expect(screen.getByTestId('steps-ratio')).toHaveTextContent(`${editRatio(s.edit, s.x).toFixed(2)} of the stream’s length`)
})

test('the marked neurons are the ones Inside one MLP lists, in its current mode', async () => {
  card()
  await waitFor(() => screen.getByTestId('strip-n'))
  const marked = () => [...screen.getByTestId('steps-scatter').querySelectorAll('.step-dot-marked')].map((e) => Number(e.getAttribute('data-neuron'))).sort((a, b) => a - b)
  const voteMarked = marked()
  expect(voteMarked).toHaveLength(8)
  act(() => useNeuronMode.getState().setMode('size'))
  await waitFor(() => expect(marked()).not.toEqual(voteMarked))
})

test('hovering a neuron shows its gate, SiLU, up, product and labels', async () => {
  card()
  await waitFor(() => screen.getByTestId('strip-n'))
  const dot = screen.getByTestId('steps-scatter').querySelector('.step-dot-marked')!
  fireEvent.mouseEnter(dot)
  const box = screen.getByTestId('steps-box')
  expect(box).toHaveTextContent(new RegExp(`^N${dot.getAttribute('data-neuron')}`))
  expect(box).toHaveTextContent(/g −?\d+\.\d{2} → SiLU −?\d+\.\d{2}/)
  expect(box).toHaveTextContent(/× u −?\d+\.\d{2} = h −?\d+\.\d{2}/)
  expect(box).toHaveTextContent(/pushes toward/)
})

test('stepping while a layer file loads never mixes layers', async () => {
  let release!: () => void
  vi.mocked(asset.loadNeuronLayer).mockImplementation((l) => l === 0
    ? new Promise((res) => { release = () => res(fixtureNeuronLayer(0)) }) : Promise.resolve(fixtureNeuronLayer(l)))
  useMlpLayer.getState().select(run, 0)
  card()
  fireEvent.click(screen.getByTestId('steps-next'))
  await waitFor(() => screen.getByTestId('strip-w'))
  const w1 = screen.getByTestId('strip-w').innerHTML
  act(() => release())
  await Promise.resolve()
  expect(screen.getByTestId('steps-title')).toHaveTextContent('layer 1')
  expect(screen.getByTestId('strip-w').innerHTML).toBe(w1)
})

test('a load error offers retry', async () => {
  vi.mocked(asset.loadNeuronLayer).mockRejectedValueOnce(new Error('HTTP 503'))
  card()
  await waitFor(() => screen.getByTestId('steps-error'))
  fireEvent.click(screen.getByTestId('steps-retry'))
  await waitFor(() => screen.getByTestId('strip-n'))
})
```

In `registry.test.tsx` add `...MLP_STEP_CALLOUTS` to the callouts list in "every callout chip points at a section that exists" and import it.

Add to the LayersDetail tests (or `DetailPanel.test.tsx`, wherever the Layers detail's MLP cards are tested): a simulated run shows `steps-note` with `STEPS_SIM_NOTE`; a real v6 run without steps (`buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, steps: false })`) shows `STEPS_OLD_NOTE` while `neuron-row`s still render.

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/MlpStepsCard.test.tsx src/glossary`
Expected: FAIL.

- [ ] **Step 3: Implement the card**

```tsx
import { useMemo, useState } from 'react'
import { MLP_STEP_CALLOUTS } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import type { NeuronAsset } from '../neurons/asset'
import type { NeuronLayer } from '../neurons/format'
import { columnNorm, layerActivations, listedNeurons, neuronLabels } from '../neurons/math'
import { contribution, decodeSteps, editRatio, gateCounts, gateDomain, layerSteps, normalize, productSummary, rms, silu, type LayerSteps } from '../neurons/steps'
import { useNeuronLayer } from '../neurons/useNeuronLayer'
import { unpackFloat32, unpackHalf } from '../trace/neuronPack'
import type { NeuronData, TraceEvent } from '../trace/types'
import { Formula } from './Formula'
import { CHART_W, GateUpScatter, ProductBars, SiluChart, StepStrip, type Hover } from './MlpStepCharts'
import { useMlpLayerControl } from './mlpLayerSelection'
import { useNeuronMode } from './neuronMode'
import type { VotedCandidate } from './selectors'
import { visibleToken } from './tokenText'

const LOADING = 'Loading the MLP weights (about 0.9 MB per layer, once per model version)…'
const MISMATCH = 'This run’s step data doesn’t match the MLP weights file.'
const f2 = (v: number) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}`

const acts = new WeakMap<NeuronData, Float32Array>()
const activationsOf = (d: NeuronData) => { let a = acts.get(d); if (!a) { a = unpackHalf(d.activations); acts.set(d, a) } return a }

// "The MLP, step by step" (#17): one layer's MLP for the last token in five
// steps, with the model's own numbers. Layer and marked neurons follow
// "Inside one MLP"; steps 1 and 5 need the layer file (norm weight, columns).
export function MlpStepsCard({ data, candidates, note, run }: {
  data?: NeuronData; candidates?: VotedCandidate[]; note: string; run?: TraceEvent
}) {
  const ready = data?.steps && candidates && candidates.length > 0
  const { layer, layers, selected, step } = useMlpLayerControl(ready ? candidates : undefined, run)
  return (
    <section data-testid="steps-card" className="mlp-card">
      <h4 data-testid="steps-title">The MLP, step by step{layer !== null ? `: layer ${layer}` : ''}{layer !== null && (
        <span className="neuron-stepper">
          <button type="button" data-testid="steps-prev" aria-label="Previous layer" disabled={layer <= 0} onClick={() => step(-1)}>◀</button>
          <button type="button" data-testid="steps-next" aria-label="Next layer" disabled={layer >= layers - 1} onClick={() => step(1)}>▶</button>
        </span>
      )}{' '}<Term id="mlp" variant="icon" label="About the MLP" testId="steps-info">ⓘ</Term></h4>
      {!ready || layer === null ? <p data-testid="steps-note" className="mlp-note">{note}</p>
        : <StepsBody data={data} candidates={candidates} selected={selected} layer={layer} />}
    </section>
  )
}

function StepsBody({ data, candidates, selected, layer }: { data: NeuronData; candidates: VotedCandidate[]; selected: number; layer: number }) {
  const state = useNeuronLayer(layer)
  const mode = useNeuronMode((s) => s.mode)
  const [hover, setHover] = useState<Hover | null>(null)
  const dims = decodeSteps(data.steps!).mlpIn.length / (activationsOf(data).length / data.neurons)
  const s = useMemo(() => layerSteps(decodeSteps(data.steps!), activationsOf(data), layer, data.neurons, dims), [data, layer, dims])
  const weights = state.status === 'ready' && state.asset.manifest.neurons === data.neurons && state.asset.manifest.dims === dims ? state : null
  const rowB64 = data.candidates.find((c) => c.id === candidates[selected].id)?.row
  const marked = useMemo(() => {
    if (!weights || !s) return []
    const row = mode === 'vote' && rowB64 ? unpackFloat32(rowB64) : null
    return listedNeurons(weights.asset.manifest, weights.layer, layerActivations(activationsOf(data), layer, data.neurons), mode, row, weights.asset.g, data.invRms)
  }, [weights, s, mode, rowB64, data, layer])
  if (!s) return <p data-testid="steps-note" className="mlp-note">{MISMATCH}</p>
  const domain = gateDomain(s.gate)
  const named = [...marked].sort((a, b) => Math.abs(s.h[b]) - Math.abs(s.h[a])).slice(0, 3)
  const box = (chart: Hover['chart']) => hover?.chart === chart
    ? <HoverBox s={s} hover={hover} weights={weights} /> : null
  const pending = state.status === 'error'
    ? <p data-testid="steps-error" className="mlp-note">Couldn’t load the MLP weights.{' '}
        <button type="button" data-testid="steps-retry" onClick={state.retry}>Retry</button></p>
    : <p data-testid="steps-loading" className="mlp-note">{LOADING}</p>
  return (
    <>
      <p className="mlp-caption">What layer {layer}’s MLP computes for the last token, in the order it computes it. Same layer as the
        cards above; <span className="step-mark" aria-hidden="true" /> marks the neurons listed in “Inside one MLP”.</p>
      <Step n={1} name="Norm" tex={String.raw`n = \operatorname{norm}(x)`}
        text="The vector coming into the MLP is scaled to a standard size, then each of the 576 numbers gets its own learned weight w.">
        {weights ? <NormStrips s={s} w={weights.layer.normWeight} eps={weights.asset.manifest.rmsNormEps} /> : pending}
      </Step>
      <Step n={2} name="Gate and up" tex={String.raw`g = W_{\text{gate}} \cdot n \quad u = W_{\text{up}} \cdot n`}
        text="Two different weight matrices turn n into two lists of 1,536 numbers, one pair per neuron. Each dot is one neuron.">
        <GateUpScatter gate={s.gate} up={s.up} domain={domain} marked={marked} named={named} hover={hover} onHover={setHover} box={box('scatter')} />
      </Step>
      <Step n={3} name="SiLU" tex={String.raw`\operatorname{SiLU}(g) = g \cdot \sigma(g)`}
        text="The gate goes through SiLU. Same gate axis as above: each dot slides straight down onto the curve.">
        <SiluChart gate={s.gate} domain={domain} marked={marked} counts={gateCounts(s.gate)} hover={hover} onHover={setHover} box={box('silu')} />
      </Step>
      <Step n={4} name="Product" tex={String.raw`h = \operatorname{SiLU}(g) \odot u`}
        text="Each neuron’s up value is scaled by its gate. This is the neuron’s activation, the number “Inside one MLP” shows.">
        <ProductBars h={s.h} summary={productSummary(s.h)} marked={marked} hover={hover} onHover={setHover} box={box('product')} />
      </Step>
      <Step n={5} name="Down" tex={String.raw`\text{edit} = W_{\text{down}} \cdot h`}
        text="Each neuron adds its column of W_down, scaled by its activation. The sum is what this layer’s MLP adds to the stream.">
        {weights ? <DownStrips s={s} layer={weights.layer} dims={dims} marked={marked} /> : pending}
      </Step>
      <div className="embed-callouts">
        {MLP_STEP_CALLOUTS.map((c) => <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="steps-callout">ⓘ {c.label}</Term>)}
      </div>
    </>
  )
}

function Step({ n, name, tex, text, children }: { n: number; name: string; tex: string; text: string; children: React.ReactNode }) {
  return (
    <div className="step-row">
      <div><h5>{n} · {name}</h5><Formula inline tex={tex} /><p className="step-text">{text}</p></div>
      <div className="step-charts">{children}</div>
    </div>
  )
}

function NormStrips({ s, w, eps }: { s: LayerSteps; w: Float32Array; eps: number }) {
  const { n, rms: r } = normalize(s.x, w, eps)
  const peakW = w.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1
  return (
    <div>
      <StepStrip values={s.x} lim={2 * rms(s.x)} label="x, into the MLP" testId="strip-x" />
      <StepStrip values={w} lim={peakW} label={`÷ rms(x) = ${r.toFixed(1)}, ⊙ w`} testId="strip-w" />
      <StepStrip values={n} lim={2 * rms(n)} label="n, normalized" testId="strip-n" />
      <p className="mlp-scale-note">{s.x.length} numbers each; blue positive, red negative, colours clipped at 2× the typical size.</p>
    </div>
  )
}

function DownStrips({ s, layer, dims, marked }: { s: LayerSteps; layer: NeuronLayer; dims: number; marked: number[] }) {
  const top = [...marked].sort((a, b) => Math.abs(s.h[b]) * columnNorm(layer, dims, b) - Math.abs(s.h[a]) * columnNorm(layer, dims, a)).slice(0, 3)
  const lim = 2 * rms(s.edit)
  return (
    <div>
      {top.map((i) => <StepStrip key={i} values={contribution(layer, dims, i, s.h[i])} lim={lim} label={`N${i}: ${f2(s.h[i])} × column`} testId={`strip-c${i}`} />)}
      <div className="step-strip-row"><span className="step-strip-label">+ {s.h.length - top.length} more</span><span className="mlp-note">…</span></div>
      <StepStrip values={s.edit} lim={lim} label="= edit" testId="strip-edit" />
      <p data-testid="steps-ratio" className="mlp-scale-note">‖edit‖ = {Math.hypot(...s.edit).toFixed(0)},{' '}
        {editRatio(s.edit, s.x).toFixed(2)} of the stream’s length (the edit size card’s cell). Same colour scale on every row.</p>
    </div>
  )
}

// Placed next to the hovered mark, as in the mockup: to its left in the right half of the chart, to its right otherwise.
function HoverBox({ s, hover, weights }: { s: LayerSteps; hover: Hover; weights: { asset: NeuronAsset; layer: NeuronLayer } | null }) {
  const i = hover.neuron
  const left = hover.x > CHART_W / 2
  const style = { left: `${(hover.x / CHART_W) * 100}%`, top: `${(hover.y / hover.height) * 100}%`,
    transform: left ? 'translate(calc(-100% - 10px), 8px)' : 'translate(10px, 8px)' }
  const labels = weights ? neuronLabels(weights.asset.manifest, weights.layer, i, s.h[i] >= 0 ? 0 : 1).slice(0, 3).map((id) => visibleToken(weights.asset.text(id))) : []
  return (
    <div data-testid="steps-box" className="step-box" role="status" style={style}>
      N{i} · g {f2(s.gate[i])} → SiLU {f2(silu(s.gate[i]))} · × u {f2(s.up[i])} = h {f2(s.h[i])}
      {labels.length > 0 && <> · pushes toward {labels.join(', ')}…</>}
    </div>
  )
}
```

Notes for the implementer: `Math.hypot(...s.edit)` on 576 values is fine. The `dims` derivation uses the steps length divided by the layer count from the activations; `layerSteps` returns null when anything mismatches, which shows `MISMATCH`. The `named` labels use the marked neurons (none until the layer file loads).

CSS additions:

```css
.step-mark { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--mark); vertical-align: middle; }
.step-box { position: absolute; width: max-content; max-width: 22rem; padding: .35rem .55rem; border: 1px solid var(--ai); border-radius: 5px;
  background: var(--card); font-size: .74rem; line-height: 1.35; box-shadow: 0 1px 4px rgba(0,0,0,.08); pointer-events: none; }
```

- [ ] **Step 4: Glossary callouts and sections**

`callouts.ts`:

```ts
export const MLP_STEP_CALLOUTS: CalloutTarget[] = [
  { label: 'why SiLU and not a switch', id: 'mlp', section: 'silu-not-switch' },
  { label: 'why two projections', id: 'mlp', section: 'two-projections' },
  { label: 'early layers look different', id: 'mlp', section: 'early-layers' },
]
```

`entries/mlp.tsx`: change the last paragraph to also name the new card ("…and <em>The MLP, step by step</em> shows one layer’s five steps with the model’s own numbers.") and add, after the "three matrices" paragraph:

```tsx
      <Section id="silu-not-switch" title="Why SiLU and not a switch">
        <p>A hard switch (ReLU) would pass every positive gate score and set every negative one to exactly zero. SiLU is
          smooth instead: negative scores give small negative values, down to about −0.28 at a score of −1.28, and large
          positive scores pass almost unchanged. Because the curve has no flat zero region, training can still adjust
          neurons whose gate is currently negative, and their small outputs still add a little to the edit.</p>
      </Section>
      <Section id="two-projections" title="Why two projections">
        <p>The gate and up matrices read the same normalized vector but learn different rows, so a neuron’s two values
          are mostly unrelated. The gate decides how much passes, the up value what passes and with which sign. In the
          last layers the largest activations come from gates around 10, where SiLU barely changes the value: there the
          gate sets the size and the up value mostly the sign.</p>
      </Section>
      <Section id="early-layers" title="Early layers look different">
        <p>In early layers most gate scores are negative and the activation is spread over many neurons: on a typical
          prompt, half of layer 5’s squared activation needs about a hundred neurons, while layer 29 needs about ten.
          The last layers concentrate on a few neurons with large gates, which is also where the per-layer votes grow.</p>
      </Section>
```

- [ ] **Step 5: Put the card into the Layers detail**

`LayersDetail.tsx`: export `STEPS_SIM_NOTE = 'Real model only: simulated runs don’t compute an MLP.'` and `STEPS_OLD_NOTE = 'This run was recorded before the step-by-step data existed. Generate it again to see it.'`; after `<NeuronCard … />` add

```tsx
        <MlpStepsCard
          data={mode === 'real' && neurons?.status === 'ok' ? neurons.data : undefined}
          candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined}
          note={mode === 'sim' ? STEPS_SIM_NOTE
            : neurons?.status === 'pending' || votes?.status === 'pending' ? PENDING_NOTE : STEPS_OLD_NOTE}
          run={run} />
```

- [ ] **Step 6: Run to verify everything passes**

Run: `npx vitest run && npx tsc -b && npx oxlint`
Expected: pass.

- [ ] **Step 7: Compare with the mockup**

Start `npm run dev`, import a fixture run as in `e2e/neurons.spec.ts` (or generate a simulated run to see the notes), open the Layers detail and screenshot the card at 1280 × 1800 next to `anatomy-v2.html` (the brainstorm server, or open the file). Check: step order, label column, full-width charts, aligned gate axis, orange marks, hover box, footnote texts. Fix differences before committing. Record "compared with mockup: …" in the ledger.

- [ ] **Step 8: Commit**

```bash
git add src/viz/MlpStepsCard.tsx src/viz/MlpStepsCard.test.tsx src/viz/details/LayersDetail.tsx src/glossary src/index.css src/viz/*.test.tsx
git commit -m "feat(viz): The MLP, step by step — the fourth Layers card (#17)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: The navigation bar

**Files:**
- Create: `src/viz/LayersNav.tsx`, `src/viz/LayersNav.test.tsx`
- Modify: `src/viz/details/LayersDetail.tsx`, `src/index.css`

**Interfaces:**
- Consumes: `useMlpLayerControl` (Task 10).
- Produces: `LAYER_SECTIONS: Array<{ id: string; label: string }>` (ids `layers-overview`, `layers-attention`, `layers-edits`, `layers-votes`, `layers-neurons`, `layers-steps`); `LayersNav({ candidates?: VotedCandidate[]; run?: TraceEvent; showStepper: boolean })`; test ids `layers-nav`, `nav-link` (with `data-section`), `nav-prev`, `nav-next`, `nav-layer`.

- [ ] **Step 1: Write the failing tests**

```tsx
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace } from '../test/fixtures'
import { cycleVotes, type VotedCandidate } from './selectors'
import { LAYER_SECTIONS, LayersNav } from './LayersNav'
import { useMlpLayer } from './mlpLayerSelection'

const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
const candidates = (cycleVotes(events, events.length - 1) as { candidates: VotedCandidate[] }).candidates
const run = events[0]

function page(tops: number[]) {
  return render(<>
    <LayersNav candidates={candidates} run={run} showStepper />
    {LAYER_SECTIONS.map((s, i) => {
      return <div key={s.id} id={s.id} ref={(el) => { if (el) el.getBoundingClientRect = () => ({ top: tops[i] } as DOMRect) }} />
    })}
  </>)
}

beforeEach(() => { Element.prototype.scrollIntoView = vi.fn() })
afterEach(cleanup)

test('six links in pipeline order, in a labelled nav landmark', () => {
  page([0, 500, 1000, 1500, 2000, 2500])
  expect(screen.getByRole('navigation', { name: 'Layers sections' })).toBeInTheDocument()
  expect(screen.getAllByTestId('nav-link').map((l) => l.textContent)).toEqual(['Overview', 'Attention', 'Edit sizes', 'Votes', 'Inside one MLP', 'Step by step'])
})

test('a click scrolls its section into view', () => {
  page([0, 500, 1000, 1500, 2000, 2500])
  fireEvent.click(screen.getAllByTestId('nav-link')[5])
  expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' })
})

test('the section whose top has passed under the bar is current', () => {
  page([-1200, -700, -200, 30, 600, 1100])
  act(() => { window.dispatchEvent(new Event('scroll')) })
  const current = screen.getAllByTestId('nav-link').filter((l) => l.getAttribute('aria-current') === 'true')
  expect(current.map((l) => l.textContent)).toEqual(['Votes'])
})

test('the stepper shows and moves the shared MLP layer', () => {
  useMlpLayer.getState().select(run, 1)
  page([0, 1, 2, 3, 4, 5])
  expect(screen.getByTestId('nav-layer')).toHaveTextContent('MLP layer 1')
  fireEvent.click(screen.getByTestId('nav-next'))
  expect(screen.getByTestId('nav-layer')).toHaveTextContent('MLP layer 2')
})

test('without MLP data the stepper is hidden and the links stay', () => {
  render(<LayersNav run={run} showStepper={false} />)
  expect(screen.queryByTestId('nav-layer')).toBeNull()
  expect(screen.getAllByTestId('nav-link')).toHaveLength(6)
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/LayersNav.test.tsx`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```tsx
import { useEffect, useRef, useState } from 'react'
import type { TraceEvent } from '../trace/types'
import { useMlpLayerControl } from './mlpLayerSelection'
import type { VotedCandidate } from './selectors'

export const LAYER_SECTIONS = [
  { id: 'layers-overview', label: 'Overview' },
  { id: 'layers-attention', label: 'Attention' },
  { id: 'layers-edits', label: 'Edit sizes' },
  { id: 'layers-votes', label: 'Votes' },
  { id: 'layers-neurons', label: 'Inside one MLP' },
  { id: 'layers-steps', label: 'Step by step' },
]

// The Layers detail's sticky section bar (#17 spec, section 2): jump links,
// the section on screen highlighted, and the shared MLP layer stepper.
export function LayersNav({ candidates, run, showStepper }: { candidates?: VotedCandidate[]; run?: TraceEvent; showStepper: boolean }) {
  const bar = useRef<HTMLElement>(null)
  const [current, setCurrent] = useState(LAYER_SECTIONS[0].id)
  const { layer, layers, step } = useMlpLayerControl(showStepper ? candidates : undefined, run)
  useEffect(() => {
    let frame = 0
    const update = () => {
      frame = 0
      const line = (bar.current?.getBoundingClientRect().bottom ?? 0) + 1
      let id = LAYER_SECTIONS[0].id
      for (const s of LAYER_SECTIONS) {
        const el = document.getElementById(s.id)
        if (el && el.getBoundingClientRect().top <= line) id = s.id
      }
      setCurrent(id)
    }
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => { window.removeEventListener('scroll', onScroll); if (frame) cancelAnimationFrame(frame) }
  }, [])
  return (
    <nav ref={bar} data-testid="layers-nav" className="layers-nav" aria-label="Layers sections">
      <span className="layers-nav-title">Jump to</span>
      <div className="layers-nav-links">
        {LAYER_SECTIONS.map((s) => (
          <a key={s.id} href={`#${s.id}`} data-testid="nav-link" data-section={s.id} className="layers-nav-link"
            aria-current={s.id === current ? 'true' : undefined}
            onClick={(e) => { e.preventDefault(); document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }}>
            {s.label}</a>
        ))}
      </div>
      {showStepper && layer !== null && (
        <span data-testid="nav-layer" className="layers-nav-layer">MLP layer {layer}
          <button type="button" data-testid="nav-prev" aria-label="Previous MLP layer" disabled={layer <= 0} onClick={() => step(-1)}>◀</button>
          <button type="button" data-testid="nav-next" aria-label="Next MLP layer" disabled={layer >= layers - 1} onClick={() => step(1)}>▶</button>
        </span>
      )}
    </nav>
  )
}
```

The scroll test dispatches `scroll` and then waits on `requestAnimationFrame`: jsdom's rAF runs on a timer, so wrap with `vi.useFakeTimers()` + `vi.advanceTimersByTime(20)` inside `act`, or call `update` directly when `requestAnimationFrame` is unavailable. Pick whichever makes the test deterministic; keep the production code above.

CSS:

```css
.layers-nav { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: .5rem; flex-wrap: nowrap;
  margin: .4rem -1.2rem .6rem; padding: .45rem 1.2rem; background: var(--card); border-bottom: 1px solid var(--line);
  box-shadow: 0 2px 4px rgba(43, 43, 49, .05); }
.layers-nav-title { font-weight: 600; font-size: .8rem; white-space: nowrap; }
.layers-nav-links { display: flex; gap: .35rem; overflow-x: auto; min-width: 0; flex: 1; scrollbar-width: thin; }
.layers-nav-link { padding: .1rem .6rem; border: 1px solid var(--line-strong); border-radius: 999px; font-size: .76rem;
  color: var(--ink); text-decoration: none; white-space: nowrap; background: var(--washi); }
.layers-nav-link[aria-current="true"] { background: var(--ai); border-color: var(--ai); color: #fff; }
.layers-nav-layer { margin-left: auto; font-size: .76rem; white-space: nowrap; display: flex; align-items: center; gap: .2rem; }
.layers-section { scroll-margin-top: 3.2rem; }
```

- [ ] **Step 4: Wire it into the Layers detail**

In `LayersDetail.tsx`:
- After the `<h3>`: `<LayersNav candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined} run={run} showStepper={mode === 'real' && votes?.status === 'ok'} />`.
- Wrap the residual diagram and the layer stack in `<div id="layers-overview" className="layers-section">`; the heatmap, pin note, explore button and grid in `<div id="layers-attention" className="layers-section">`; each MLP card in `<div id="layers-edits|layers-votes|layers-neurons|layers-steps" className="layers-section">` inside `mlp-cards`.

Run the existing LayersDetail/DetailPanel tests; fix only selectors that depended on direct children.

- [ ] **Step 5: Run to verify everything passes**

Run: `npx vitest run && npx tsc -b && npx oxlint`
Expected: pass.

- [ ] **Step 6: Compare with the mockup**

Screenshot the Layers detail scrolled to the votes card next to `navigation.html` option A; check placement, chip look, highlight, stepper at the right. Record in the ledger.

- [ ] **Step 7: Commit**

```bash
git add src/viz/LayersNav.tsx src/viz/LayersNav.test.tsx src/viz/details/LayersDetail.tsx src/index.css src/viz/*.test.tsx
git commit -m "feat(viz): sticky section bar with the shared MLP layer stepper in the Layers detail

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Browser test

**Files:**
- Create: `e2e/steps.spec.ts`

- [ ] **Step 1: Write the test**

```ts
import { expect, test, type Page } from '@playwright/test'
import { buildFixtureTrace } from '../src/test/fixtures'
import { neuronFixtureFile } from '../src/test/neuronFixture'

async function openFixtureRun(page: Page) {
  await page.route(/huggingface\.co\/.*\/neurons\//, async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').pop()!
    const body = neuronFixtureFile(name)
    await route.fulfill(body
      ? { status: 200, body: Buffer.from(body), headers: { 'access-control-allow-origin': '*' },
          contentType: name.endsWith('.json') ? 'application/json' : 'application/octet-stream' }
      : { status: 404, headers: { 'access-control-allow-origin': '*' } })
  })
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('seed run')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, prompt: 'steps fixture' })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'steps fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 2 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(), reason: 'max-tokens' } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  await page.getByTestId('run-chip').filter({ hasText: 'steps fixture' }).getByTestId('run-chip-main').click()
  await page.getByTestId('stage-card').nth(2).click()
}

test('the section bar sticks, jumps and steps the shared layer; the charts fill the card', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await openFixtureRun(page)
  const nav = page.getByTestId('layers-nav')
  await expect(nav).toBeVisible()

  await page.getByTestId('nav-link').filter({ hasText: 'Step by step' }).click()
  await expect(page.getByTestId('steps-title')).toBeInViewport()
  await expect(page.getByTestId('nav-link').filter({ hasText: 'Step by step' })).toHaveAttribute('aria-current', 'true')
  expect(Math.round((await nav.boundingBox())!.y)).toBe(0)                 // stuck to the top

  const layerOf = async (id: string) => Number((await page.getByTestId(id).textContent())!.match(/layer (\d+)/)![1])
  const before = await layerOf('steps-title')
  await page.getByTestId(before > 0 ? 'nav-prev' : 'nav-next').click()
  const after = await layerOf('steps-title')
  expect(after).toBe(before > 0 ? before - 1 : before + 1)
  expect(await layerOf('neuron-title')).toBe(after)

  await expect(page.getByTestId('strip-n')).toBeVisible()
  const scatter = (await page.getByTestId('steps-scatter').boundingBox())!
  const silu = (await page.getByTestId('steps-silu').boundingBox())!
  expect(scatter.width).toBeGreaterThanOrEqual(700)
  expect(Math.abs(scatter.x - silu.x)).toBeLessThan(1)
  expect(Math.abs(scatter.width - silu.width)).toBeLessThan(1)
})

test('at phone width the page does not scroll sideways', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await openFixtureRun(page)
  await page.getByTestId('nav-link').filter({ hasText: 'Step by step' }).click()
  await expect(page.getByTestId('steps-scatter')).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  expect(overflow).toBeLessThanOrEqual(0)
})
```

- [ ] **Step 2: Run it**

Run: `npx playwright test`
Expected: all specs pass (including the existing `neurons.spec.ts`, which now serves a v6 fixture manifest). If the phone-width test finds overflow, fix the CSS (the offending element is usually a chart without the `.step-charts` scroll container) rather than the test.

- [ ] **Step 3: Commit**

```bash
git add e2e/steps.spec.ts
git commit -m "test(e2e): section bar, shared layer stepper and chart widths

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Docs, final review, publish v6 (stops for the user)

**Files:**
- Modify: `README.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Docs**

- `README.md`: in the Layers section describe the fourth card (five steps, real numbers for the last token, marked neurons, shared layer) and the section bar; in the model section say v6 is weight-only 8-bit and picks the same next token as fp32 in browsers; update the "≈" paragraph from #16 to say it should no longer appear with v6 and remains as a safeguard.
- `docs/BACKLOG.md`: mark #17 and #19 done with a one-paragraph record each (date 2026-10-20 or the actual date, model size, argmax 4/4, the card); remove "add the model tag to the neurons manifest at v6" from #20.

Commit: `docs: #17 step-by-step card, #19 faithful model, section bar` (with the trailer).

- [ ] **Step 2: Whole-branch review**

Run the final review per the execution skill (fresh reviewer on the most capable model), passing this plan's Review Focus. Fix Critical/Important findings with a failing test first; run `npx vitest run`, `npx tsc -b`, `npx oxlint`, `npx playwright test`, `cd tools/export && uv run pytest -q`.

- [ ] **Step 3: STOP — ask the user before publishing**

Publishing to Hugging Face is outward-facing. Report the Task 6 results (size, validate verdict, browser check) and ask for the go-ahead to run:

```bash
cd tools/export && uv run python -m tsumugi_export publish --model-dir out/model --tag v6
```

- [ ] **Step 4: After the go-ahead — publish and check the published files**

Run the publish command. Then `curl -sI https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx/resolve/v6/neurons/manifest.json | head -1` (expect 200) and `curl -s https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx/resolve/v6/neurons/manifest.json | grep '"revision"'` (expect `"v6"`).

- [ ] **Step 5: Local check against the published model**

`npm run dev`, real mode, prompt "The cat sat on the mat because it was tired": generate, open the Layers detail, layer 29. Expect: gate counts close to 885 negative / 651 positive and "9 neurons carry half" / "44 … 90%" (within quantization tolerance); the neuron card's total has no "≈" at the first generated token on any layer (step through all 30 with the bar's stepper). Screenshot the card and the bar next to the mockups.

- [ ] **Step 6: Hand back for merge**

Report to the user; merging, pushing and the live test drive follow the usual flow once they say so.
