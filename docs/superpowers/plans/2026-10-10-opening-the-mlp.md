# Opening the MLP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two Layers-detail cards — edit size per token and layer, and per-layer votes that add up exactly to the Logits card's score — fed by two new outputs of a v4 model export.

**Architecture:** The export's post-processing adds `edit_ratios` and `stream_parts` to the ONNX graph, built from each layer's residual Add nodes and the final RMSNorm's own `1/rms` tensor. The worker turns them into a new `edits` trace event and a `votes` field on logits candidates. Pure selectors stack the data for the viewed cycle, two presentational cards draw it, and the candidate selection moves into a small store shared with the Logits card.

**Tech Stack:** Python (onnx, onnxruntime, torch, pytest) in `tools/export`; React 19, TypeScript, zustand 5, vitest 4 + Testing Library, Playwright, Temml MathML in the app.

**Spec:** `docs/superpowers/specs/2026-10-10-opening-the-mlp-design.md`

## Global Constraints

- Simulated mode note, verbatim: "Real model only: simulated runs have no MLP to measure."
- Missing-data note, verbatim: "This run was recorded before MLP data was exported."
- Pending-votes note, verbatim: "Votes appear once this cycle’s scores are computed."
- Card titles, verbatim: "How much each layer changes the vector" and "What each layer voted for".
- Edit chips, verbatim: "why layer 0 is so dark", "quiet middle, loud end", "attention vs MLP" → `edit` § `layer-0`, `quiet-middle`, `attn-vs-mlp`.
- New trace data is optional: traces recorded before this feature must still validate and load.
- Worker policy is never-fail: absent output → no data; bad shape → `console.warn`, feature off for the rest of the run, run continues.
- Values in the trace are rounded to 3 decimals.
- Blue adds, red subtracts: reuse `cellFill` from `src/viz/stripMath.ts` (indigo positive, vermillion negative).
- `ATTN_MODEL_REVISION` is bumped to `'v4'` only after tag `v4` exists on the Hub; publishing needs the user's explicit go-ahead.
- Lint: oxlint `react-hooks/exhaustive-deps`, `react/set-state-in-effect`, `react/only-export-components` (component files export only components).
- Node 26 hides jsdom's `localStorage`; don't touch it in new tests.

## Review Focus

1. **A run whose cycles mix data and no data** (votes turned off mid-run by a bad shape, or a candidate the graph's top-16 missed because of a tie): the votes card must show the missing-data note for that cycle, never a partial grid. Pinned by the `cycleVotes` "none when any candidate lacks votes" test (Task 5).
2. **Viewing an early cycle of a long run:** the heatmap must show only the tokens visible at that cycle, not rows from later cycles. Pinned by the `editRows` "never shows rows past the viewed cycle" test (Task 5).
3. **Selecting the last token, then stepping back a cycle:** the token row no longer exists; the bars must fall back to the last visible token, not crash or show an empty chart. Pinned by the EditSizeCard "selection out of range falls back" test (Task 8).
4. **Candidate selection after a new run:** the shared store keeps an index keyed by cycle; an index past the new cycle's candidate count must fall back to 0. Pinned by the `selectedCandidate` bounds test (Task 6).
5. **Votes that don't add up** (a graph whose final-norm tracing went wrong): the worker must drop the votes rather than ship a breakdown that disagrees with the Logits card. Pinned by the `computeVotes` mismatch test (Task 4).

---

## File Structure

**Export (`tools/export`)**
- Modify `src/tsumugi_export/postprocess.py` — add `find_blocks`, `find_final_norm`, `expose_mlp_internals`; `postprocess_graph` calls it.
- Modify `src/tsumugi_export/validate.py` — add `torch_edit_ratios` and four pure checks; wire them into `run`.
- Modify `src/tsumugi_export/publish.py` (model card text), `README.md` (checklist, tag v4).
- Create `tests/test_postprocess_mlp.py`, `tests/test_validate_mlp.py`.

**App**
- Modify `src/trace/types.ts`, `src/trace/validate.ts`, `src/player/pacing.ts`, `src/viz/selectors.ts`, `src/test/fixtures.ts`.
- Create `src/engine/transformers/mlpData.ts` (+ test); modify `readout.ts` (export two helpers) and `worker.ts`.
- Create `src/viz/mlpMath.ts` (+ test), `src/viz/candidateSelection.ts` (+ test), `src/viz/EditSizeCard.tsx` (+ test), `src/viz/LayerVotesCard.tsx` (+ test).
- Modify `src/viz/details/LogitsDetail.tsx`, `src/viz/details/LayersDetail.tsx`, `src/viz/DetailPanel.tsx`, `src/index.css`, `src/test/setup.ts`.
- Glossary: modify `types.ts`, `runSummary.ts`, `registry.ts`, `callouts.ts`, `entries/edit.tsx`, `entries/mlp.tsx`, `src/test/glossaryFixture.ts`; create `entries/votes.tsx`.
- Docs: `README.md`, `docs/BACKLOG.md`, `scripts/capture-screenshots.mjs`, `e2e/mlp.spec.ts`.

Probe already done (2026-10-10, on the local v3 export in `tools/export/out/model`): in the fp32 graph (opset 18) layer `l` has `/model/layers.{l}/self_attn/o_proj/MatMul` whose output feeds exactly one Add (`/model/layers.{l}/Add`, other input = the stream before attention, a no-op `Cast` of the previous stream), and `/model/layers.{l}/mlp/down_proj/MatMul` feeding `/model/layers.{l}/Add_1` (other input = the stream before the MLP). The final norm traced back from the lm_head input is `Mul(model.norm.weight [576], Cast(Mul(Cast(x), /model/norm/Div_output_0)))` where `Div = 1 / Sqrt(ReduceMean(x²) + eps)`. The quantized graph keeps the Add input tensor names, so post-processing the fp32 graph before quantization carries over.

---

### Task 1: Export post-processing — `edit_ratios` and `stream_parts`

**Files:**
- Modify: `tools/export/src/tsumugi_export/postprocess.py`
- Modify: `tools/export/tests/test_postprocess.py` (one existing test switches graph)
- Create: `tools/export/tests/test_postprocess_mlp.py`

**Interfaces:**
- Consumes: `find_lm_head(graph) -> (x, weight_t, dims)`, `_producer`, `INT64_MAX` (existing, same file).
- Produces: constants `EDIT_RATIOS_OUTPUT = "edit_ratios"`, `STREAM_PARTS_OUTPUT = "stream_parts"`; `find_blocks(graph) -> list[tuple[str, str, str, str]]` (per layer: attention edit, stream before attention, MLP edit, stream before MLP); `find_final_norm(graph) -> tuple[str, str]` (gain initializer name, inverse-rms tensor name); `expose_mlp_internals(model) -> ModelProto`. Output shapes: `edit_ratios` [batch, seq, layers, 2] (index 0 attention, 1 MLP); `stream_parts` [batch, 1 + 2·layers, dims] (row 0 embedding, rows 1..L attention, rows L+1..2L MLP), each row `gain ⊙ part · (1/rms)` at the last position.

- [ ] **Step 1: Write the failing tests**

Create `tools/export/tests/test_postprocess_mlp.py`:

```python
import numpy as np
import onnx
from onnx import TensorProto, helper

from tsumugi_export.postprocess import (
    EDIT_RATIOS_OUTPUT, STREAM_PARTS_OUTPUT, expose_mlp_internals, find_blocks, find_final_norm,
    postprocess_graph,
)

DIMS, VOCAB, LAYERS, EPS = 4, 7, 2, 1e-5
rng = np.random.default_rng(1)
TABLE = rng.normal(size=(VOCAB, DIMS)).astype(np.float32)
WA = [rng.normal(scale=0.5, size=(DIMS, DIMS)).astype(np.float32) for _ in range(LAYERS)]
WM = [rng.normal(scale=0.5, size=(DIMS, DIMS)).astype(np.float32) for _ in range(LAYERS)]
GAIN = rng.uniform(0.5, 1.5, size=DIMS).astype(np.float32)


def layered_graph() -> onnx.ModelProto:
    """A miniature of the real export's layout (names and op pattern from the
    2026-10-10 probe): per layer a no-op Cast of the stream, an o_proj MatMul
    feeding the residual Add, another Cast, a down_proj MatMul feeding Add_1;
    then the final RMSNorm (Pow, ReduceMean, Add eps, Sqrt, Div 1/·, Mul,
    Cast, Mul gain) and the lm_head MatMul."""
    f = TensorProto.FLOAT
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"])
    inits = [
        helper.make_tensor("model.embed_tokens.weight", f, [VOCAB, DIMS], TABLE.flatten().tolist()),
        helper.make_tensor("lm_head.weight_t", f, [DIMS, VOCAB], TABLE.T.flatten().tolist()),
        helper.make_tensor("model.norm.weight", f, [DIMS], GAIN.tolist()),
        helper.make_tensor("two", f, [], [2.0]),
        helper.make_tensor("one", f, [], [1.0]),
        helper.make_tensor("eps", f, [], [EPS]),
        helper.make_tensor("axes_last", TensorProto.INT64, [1], [-1]),
    ]
    nodes = [helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["hidden_states.0"],
                              name="/model/embed_tokens/Gather", axis=0)]
    stream = "hidden_states.0"
    for l in range(LAYERS):
        p = f"/model/layers.{l}"
        inits += [helper.make_tensor(f"wa{l}", f, [DIMS, DIMS], WA[l].flatten().tolist()),
                  helper.make_tensor(f"wm{l}", f, [DIMS, DIMS], WM[l].flatten().tolist())]
        nodes += [
            helper.make_node("Cast", [stream], [f"{p}/input_layernorm/Cast_output_0"], name=f"{p}/input_layernorm/Cast", to=1),
            helper.make_node("MatMul", [f"{p}/input_layernorm/Cast_output_0", f"wa{l}"],
                             [f"{p}/self_attn/o_proj/MatMul_output_0"], name=f"{p}/self_attn/o_proj/MatMul"),
            helper.make_node("Add", [f"{p}/input_layernorm/Cast_output_0", f"{p}/self_attn/o_proj/MatMul_output_0"],
                             [f"{p}/Add_output_0"], name=f"{p}/Add"),
            helper.make_node("Cast", [f"{p}/Add_output_0"], [f"{p}/post_attention_layernorm/Cast_output_0"],
                             name=f"{p}/post_attention_layernorm/Cast", to=1),
            helper.make_node("MatMul", [f"{p}/post_attention_layernorm/Cast_output_0", f"wm{l}"],
                             [f"{p}/mlp/down_proj/MatMul_output_0"], name=f"{p}/mlp/down_proj/MatMul"),
            helper.make_node("Add", [f"{p}/post_attention_layernorm/Cast_output_0", f"{p}/mlp/down_proj/MatMul_output_0"],
                             [f"{p}/Add_1_output_0"], name=f"{p}/Add_1"),
        ]
        stream = f"{p}/Add_1_output_0"
    n = "/model/norm"
    nodes += [
        helper.make_node("Cast", [stream], [f"{n}/Cast_output_0"], name=f"{n}/Cast", to=1),
        helper.make_node("Pow", [f"{n}/Cast_output_0", "two"], [f"{n}/Pow_output_0"], name=f"{n}/Pow"),
        helper.make_node("ReduceMean", [f"{n}/Pow_output_0", "axes_last"], [f"{n}/ReduceMean_output_0"],
                         name=f"{n}/ReduceMean", keepdims=1),
        helper.make_node("Add", [f"{n}/ReduceMean_output_0", "eps"], [f"{n}/Add_output_0"], name=f"{n}/Add"),
        helper.make_node("Sqrt", [f"{n}/Add_output_0"], [f"{n}/Sqrt_output_0"], name=f"{n}/Sqrt"),
        helper.make_node("Div", ["one", f"{n}/Sqrt_output_0"], [f"{n}/Div_output_0"], name=f"{n}/Div"),
        helper.make_node("Mul", [f"{n}/Cast_output_0", f"{n}/Div_output_0"], [f"{n}/Mul_output_0"], name=f"{n}/Mul"),
        helper.make_node("Cast", [f"{n}/Mul_output_0"], [f"{n}/Cast_1_output_0"], name=f"{n}/Cast_1", to=1),
        helper.make_node("Mul", ["model.norm.weight", f"{n}/Cast_1_output_0"], [f"{n}/Mul_1_output_0"], name=f"{n}/Mul_1"),
        helper.make_node("MatMul", [f"{n}/Mul_1_output_0", "lm_head.weight_t"], ["logits"], name="/lm_head/MatMul"),
    ]
    outs = [helper.make_tensor_value_info("logits", f, [1, "seq", VOCAB]),
            helper.make_tensor_value_info("hidden_states.0", f, [1, "seq", DIMS])]
    graph = helper.make_graph(nodes, "layered", [ids], outs, initializer=inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def reference(ids: np.ndarray):
    """numpy forward of layered_graph: per-layer edits, streams, the final x and 1/rms."""
    x = TABLE[ids[0]]
    e = x.copy()
    attn, mlp, before_a, before_m = [], [], [], []
    for l in range(LAYERS):
        a = x @ WA[l]; before_a.append(x); attn.append(a); x = x + a
        m = x @ WM[l]; before_m.append(x); mlp.append(m); x = x + m
    inv = 1.0 / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS)
    return e, attn, mlp, before_a, before_m, x, inv


def _run(model: onnx.ModelProto, ids: np.ndarray) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": ids})))


def test_find_blocks_returns_edit_and_stream_per_layer():
    blocks = find_blocks(layered_graph().graph)
    assert len(blocks) == LAYERS
    a, xa, m, xm = blocks[1]
    assert a == "/model/layers.1/self_attn/o_proj/MatMul_output_0"
    assert xa == "/model/layers.1/input_layernorm/Cast_output_0"
    assert m == "/model/layers.1/mlp/down_proj/MatMul_output_0"
    assert xm == "/model/layers.1/post_attention_layernorm/Cast_output_0"


def test_find_final_norm_returns_gain_and_inverse_rms():
    assert find_final_norm(layered_graph().graph) == ("model.norm.weight", "/model/norm/Div_output_0")


def test_edit_ratios_match_numpy():
    model = expose_mlp_internals(layered_graph())
    onnx.checker.check_model(model)
    ids = np.array([[2, 5, 1]], dtype=np.int64)
    out = _run(model, ids)
    _, attn, mlp, before_a, before_m, _, _ = reference(ids)
    got = out[EDIT_RATIOS_OUTPUT]
    assert got.shape == (1, 3, LAYERS, 2)
    for l in range(LAYERS):
        np.testing.assert_allclose(got[0, :, l, 0], np.linalg.norm(attn[l], axis=-1) / np.linalg.norm(before_a[l], axis=-1), rtol=1e-5)
        np.testing.assert_allclose(got[0, :, l, 1], np.linalg.norm(mlp[l], axis=-1) / np.linalg.norm(before_m[l], axis=-1), rtol=1e-5)


def test_stream_parts_are_scaled_last_position_parts_and_sum_to_the_norm_output():
    model = expose_mlp_internals(layered_graph())
    ids = np.array([[4, 0, 6, 3]], dtype=np.int64)
    out = _run(model, ids)
    e, attn, mlp, _, _, x, inv = reference(ids)
    parts = out[STREAM_PARTS_OUTPUT]
    assert parts.shape == (1, 1 + 2 * LAYERS, DIMS)
    scale = GAIN * inv[-1]
    np.testing.assert_allclose(parts[0, 0], e[-1] * scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0, 1 + 1], attn[1][-1] * scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0, 1 + LAYERS], mlp[0][-1] * scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0].sum(axis=0), GAIN * x[-1] * inv[-1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0].sum(axis=0) @ TABLE.T, out["logits"][0, -1], rtol=1e-4, atol=1e-5)


def test_postprocess_graph_adds_the_mlp_outputs_after_the_readout():
    model = postprocess_graph(layered_graph(), k=3)
    onnx.checker.check_model(model)
    names = [o.name for o in model.graph.output]
    assert names[-3:] == [EDIT_RATIOS_OUTPUT, STREAM_PARTS_OUTPUT, "inputs_embeds"]


def test_a_graph_without_layers_is_refused():
    model = layered_graph()
    for node in model.graph.node:
        if node.name.endswith("o_proj/MatMul"):
            node.name = node.name.replace("o_proj", "renamed")
    try:
        expose_mlp_internals(model)
    except ValueError as err:
        assert "o_proj" in str(err)
    else:
        raise AssertionError("expected ValueError")
```

In `tools/export/tests/test_postprocess.py`, `test_postprocess_graph_combines_readout_and_inputs_embeds` builds from `readout_graph()`, which has no layers; `postprocess_graph` will now refuse it. Replace that test with:

```python
def test_postprocess_graph_combines_readout_and_inputs_embeds():
    from test_postprocess_mlp import TABLE as LAYERED_TABLE, layered_graph
    model = postprocess_graph(layered_graph(), k=3)
    names = [o.name for o in model.graph.output]
    assert names == ["logits", LAST_HIDDEN_OUTPUT, TOPK_IDS_OUTPUT, TOPK_EMBEDS_OUTPUT,
                     "edit_ratios", "stream_parts", "inputs_embeds"]
    out = _run(model, np.array([[0, 6]], dtype=np.int64))
    np.testing.assert_allclose(out["inputs_embeds"][0], LAYERED_TABLE[[0, 6]], atol=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_postprocess_mlp.py tests/test_postprocess.py -q`
Expected: FAIL — `ImportError: cannot import name 'EDIT_RATIOS_OUTPUT'`.

If `from test_postprocess_mlp import …` fails with `ModuleNotFoundError` once the code exists, add `pythonpath = ["tests"]` under a new `[tool.pytest.ini_options]` table in `tools/export/pyproject.toml`.

- [ ] **Step 3: Implement**

In `postprocess.py`, update the module docstring's first paragraph to mention the new outputs (one sentence: "We also add `edit_ratios` and `stream_parts` for the MLP cards, built from each layer's residual Add nodes and the final norm's own 1/rms tensor (see expose_mlp_internals)."). Then add below `expose_readout`:

```python
EDIT_RATIOS_OUTPUT = "edit_ratios"
STREAM_PARTS_OUTPUT = "stream_parts"
ATTN_MATMUL = "/model/layers.{l}/self_attn/o_proj/MatMul"
MLP_MATMUL = "/model/layers.{l}/mlp/down_proj/MatMul"


def _residual_add(graph: onnx.GraphProto, matmul_name: str) -> tuple[str, str]:
    """(edit, stream before) of the residual Add fed by the named MatMul: the
    MatMul's output is the block's edit, the Add's other input is the stream
    the edit is added to."""
    mm = next((n for n in graph.node if n.name == matmul_name), None)
    if mm is None:
        raise ValueError(f"no node {matmul_name}")
    edit = mm.output[0]
    adds = [n for n in graph.node if n.op_type == "Add" and edit in n.input]
    if len(adds) != 1:
        raise ValueError(f"{matmul_name}: expected one residual Add, found {len(adds)}")
    before = [i for i in adds[0].input if i != edit]
    if len(before) != 1:
        raise ValueError(f"{matmul_name}: residual Add has inputs {list(adds[0].input)}")
    return edit, before[0]


def find_blocks(graph: onnx.GraphProto) -> list[tuple[str, str, str, str]]:
    """Per layer (attention edit, stream before attention, MLP edit, stream
    before MLP), located by the o_proj / down_proj MatMul names (the probe of
    2026-10-10 found the residual Adds right behind them; the quantizer keeps
    these tensor names, and quantize.py already relies on optimum's node
    names for /lm_head/MatMul)."""
    names = {n.name for n in graph.node}
    layers = 0
    while ATTN_MATMUL.format(l=layers) in names:
        layers += 1
    if layers == 0:
        raise ValueError(f"no {ATTN_MATMUL.format(l=0)} — is this the SmolLM2 export?")
    blocks = []
    for l in range(layers):
        a, xa = _residual_add(graph, ATTN_MATMUL.format(l=l))
        m, xm = _residual_add(graph, MLP_MATMUL.format(l=l))
        blocks.append((a, xa, m, xm))
    return blocks


def find_final_norm(graph: onnx.GraphProto) -> tuple[str, str]:
    """(gain, inverse rms) of the final RMSNorm, traced back from the lm_head
    input: Mul(gain [dims], Cast?(Mul(Cast?(x), inv))) with inv made by a Div
    (1 / sqrt(mean(x²) + eps)). Reusing the graph's own inv makes the stream
    parts add up to exactly what the lm_head multiplies."""
    x, _, dims = find_lm_head(graph)
    inits = {t.name: list(t.dims) for t in graph.initializer}

    def through_casts(name: str) -> onnx.NodeProto | None:
        node = _producer(graph, name)
        while node is not None and node.op_type in ("Cast", "Identity"):
            node = _producer(graph, node.input[0])
        return node

    gain_mul = through_casts(x)
    if gain_mul is None or gain_mul.op_type != "Mul":
        raise ValueError("the lm_head input is not produced by the final norm's gain Mul")
    gain = next((i for i in gain_mul.input if inits.get(i) == [dims]), None)
    if gain is None:
        raise ValueError(f"final norm Mul has no [{dims}] gain initializer")
    other = next(i for i in gain_mul.input if i != gain)
    scale_mul = through_casts(other)
    if scale_mul is None or scale_mul.op_type != "Mul":
        raise ValueError("the final norm has no x · (1/rms) Mul")
    inv = next((i for i in scale_mul.input
                if (p := _producer(graph, i)) is not None and p.op_type == "Div"), None)
    if inv is None:
        raise ValueError("the final norm's 1/rms is not produced by a Div")
    return gain, inv


def expose_mlp_internals(model: onnx.ModelProto) -> onnx.ModelProto:
    """edit_ratios [batch, seq, layers, 2]: ‖edit‖ / ‖stream before‖ per fed
    position, layer and block (0 attention, 1 MLP). stream_parts [batch,
    1 + 2·layers, dims]: the last position's embedding, attention edits and
    MLP edits, each multiplied by the final norm's gain and 1/rms — so they
    add up to what the lm_head multiplies (opset 18: axes are inputs)."""
    graph = model.graph
    blocks = find_blocks(graph)
    gain, inv = find_final_norm(graph)
    _, _, dims = find_lm_head(graph)
    layers = len(blocks)
    graph.initializer.extend([
        helper.make_tensor("tsumugi_mlp_axis_last", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_mlp_axis_3", TensorProto.INT64, [1], [3]),
        helper.make_tensor("tsumugi_mlp_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_mlp_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_mlp_slice_axes", TensorProto.INT64, [1], [1]),
    ])
    nodes: list[onnx.NodeProto] = []
    ratios: dict[str, list[str]] = {"attn": [], "mlp": []}
    for l, (a, xa, m, xm) in enumerate(blocks):
        for kind, edit, before in (("attn", a, xa), ("mlp", m, xm)):
            p = f"tsumugi_{kind}{l}"
            nodes += [
                helper.make_node("ReduceL2", [edit, "tsumugi_mlp_axis_last"], [f"{p}_edit_norm"],
                                 name=f"{p}_edit_norm", keepdims=1),
                helper.make_node("ReduceL2", [before, "tsumugi_mlp_axis_last"], [f"{p}_stream_norm"],
                                 name=f"{p}_stream_norm", keepdims=1),
                helper.make_node("Div", [f"{p}_edit_norm", f"{p}_stream_norm"], [f"{p}_ratio"], name=f"{p}_ratio"),
            ]
            ratios[kind].append(f"{p}_ratio")   # [batch, seq, 1]
    for kind in ("attn", "mlp"):
        nodes += [
            helper.make_node("Concat", ratios[kind], [f"tsumugi_{kind}_ratios"], name=f"tsumugi_{kind}_ratios", axis=2),
            helper.make_node("Unsqueeze", [f"tsumugi_{kind}_ratios", "tsumugi_mlp_axis_3"],
                             [f"tsumugi_{kind}_ratios_4d"], name=f"tsumugi_{kind}_ratios_4d"),
        ]
    nodes.append(helper.make_node("Concat", ["tsumugi_attn_ratios_4d", "tsumugi_mlp_ratios_4d"],
                                  [EDIT_RATIOS_OUTPUT], name="tsumugi_edit_ratios", axis=3))

    sources = [blocks[0][1]] + [b[0] for b in blocks] + [b[2] for b in blocks]
    parts = []
    for i, src in enumerate(sources):
        nodes.append(helper.make_node(
            "Slice", [src, "tsumugi_mlp_slice_starts", "tsumugi_mlp_slice_ends", "tsumugi_mlp_slice_axes"],
            [f"tsumugi_part{i}"], name=f"tsumugi_part{i}"))
        parts.append(f"tsumugi_part{i}")
    nodes += [
        helper.make_node("Concat", parts, ["tsumugi_parts_raw"], name="tsumugi_parts_raw", axis=1),
        helper.make_node("Slice", [inv, "tsumugi_mlp_slice_starts", "tsumugi_mlp_slice_ends", "tsumugi_mlp_slice_axes"],
                         ["tsumugi_inv_last"], name="tsumugi_inv_last"),
        helper.make_node("Mul", ["tsumugi_parts_raw", "tsumugi_inv_last"], ["tsumugi_parts_scaled"],
                         name="tsumugi_parts_scaled"),
        helper.make_node("Mul", ["tsumugi_parts_scaled", gain], [STREAM_PARTS_OUTPUT], name="tsumugi_stream_parts"),
    ]
    graph.node.extend(nodes)
    graph.output.extend([
        helper.make_tensor_value_info(EDIT_RATIOS_OUTPUT, TensorProto.FLOAT, ["batch_size", "sequence_length", layers, 2]),
        helper.make_tensor_value_info(STREAM_PARTS_OUTPUT, TensorProto.FLOAT, ["batch_size", 1 + 2 * layers, dims]),
    ])
    return model
```

Change `postprocess_graph`:

```python
def postprocess_graph(model: onnx.ModelProto, k: int = TOPK) -> onnx.ModelProto:
    """Readout and MLP internals (both derived from graph structure, independent
    of the output list), then the inputs_embeds rename, which drops every
    hidden_states.* output."""
    return expose_inputs_embeds(expose_mlp_internals(expose_readout(model, k)))
```

Note: the real graph's ops run in fp32 (the no-op Casts have `to=1`); the quantizer only rewrites MatMul/Gather nodes, so the new nodes need no entry in `quantize.NODES_TO_EXCLUDE`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools/export && uv run pytest -q`
Expected: all tests PASS (the six new ones plus the existing suite).

- [ ] **Step 5: Commit**

```bash
git add tools/export/src/tsumugi_export/postprocess.py tools/export/tests/test_postprocess_mlp.py tools/export/tests/test_postprocess.py tools/export/pyproject.toml
git commit -m "feat(export): edit_ratios and stream_parts outputs for the MLP cards"
```

---

### Task 2: Export validator checks, model card, export README

**Files:**
- Modify: `tools/export/src/tsumugi_export/validate.py`
- Modify: `tools/export/src/tsumugi_export/publish.py` (model card paragraph near line 30–44)
- Modify: `tools/export/README.md`
- Create: `tools/export/tests/test_validate_mlp.py`

**Interfaces:**
- Consumes: output names from Task 1 (`edit_ratios`, `stream_parts`), existing `last_hidden_state`, `topk_ids`, `topk_embeds`, `logits`; `_drive_cached_stepwise`, `_feeds`, `_tokenize`.
- Produces: `check_edit_ratios_parity(outputs, expected, tol) -> dict`, `check_stream_parts_sum(outputs, tol) -> dict`, `check_votes_sum(outputs, tol) -> dict`, `check_edit_cache_parity(single, steps, tol) -> dict`, `torch_edit_ratios(ids) -> np.ndarray [seq, layers, 2]`. Report keys `{variant}:edit-ratios-parity`, `{variant}:stream-parts-sum`, `{variant}:votes-sum`, `{variant}:edit-cache-parity`.

- [ ] **Step 1: Write the failing tests**

Create `tools/export/tests/test_validate_mlp.py`:

```python
import numpy as np

from tsumugi_export.validate import (
    check_edit_cache_parity, check_edit_ratios_parity, check_stream_parts_sum, check_votes_sum,
)

rng = np.random.default_rng(2)
L, D, V, K, SEQ = 3, 8, 40, 4, 5
E = rng.normal(size=(V, D)).astype(np.float32)


def good_outputs() -> dict:
    parts = rng.normal(size=(1, 1 + 2 * L, D)).astype(np.float32)
    last = rng.normal(size=(1, SEQ, D)).astype(np.float32)
    last[0, -1] = parts[0].sum(axis=0)
    logits = last @ E.T
    ids = np.argsort(-logits[0, -1], kind="stable")[:K]
    return {
        "stream_parts": parts, "last_hidden_state": last, "logits": logits,
        "topk_ids": ids[None, :].astype(np.int64), "topk_embeds": E[ids][None, :, :],
        "edit_ratios": rng.uniform(0, 2, size=(1, SEQ, L, 2)).astype(np.float32),
    }


def test_stream_parts_sum_passes_and_fails():
    out = good_outputs()
    assert check_stream_parts_sum(out, tol=1e-4)["passed"]
    out["stream_parts"][0, 3] += 0.1
    r = check_stream_parts_sum(out, tol=1e-4)
    assert not r["passed"] and "Σ parts" in r["detail"]


def test_stream_parts_sum_reports_missing_and_misshapen():
    out = good_outputs()
    del out["stream_parts"]
    assert "no stream_parts" in check_stream_parts_sum(out, tol=1e-4)["detail"]
    out = good_outputs()
    out["stream_parts"] = out["stream_parts"][:, :, :4]
    assert "shape" in check_stream_parts_sum(out, tol=1e-4)["detail"]


def test_votes_sum_passes_and_fails():
    out = good_outputs()
    assert check_votes_sum(out, tol=1e-3)["passed"]
    out["topk_embeds"] = out["topk_embeds"] + 0.05
    assert not check_votes_sum(out, tol=1e-3)["passed"]


def test_edit_ratios_parity():
    out = good_outputs()
    expected = out["edit_ratios"][0].copy()
    assert check_edit_ratios_parity(out, expected, tol=1e-3)["passed"]
    r = check_edit_ratios_parity(out, expected + 0.01, tol=1e-3)
    assert not r["passed"] and "Δratio" in r["detail"]
    r = check_edit_ratios_parity(out, expected[:, :2], tol=1e-3)
    assert not r["passed"] and "shape" in r["detail"]
    assert "no edit_ratios" in check_edit_ratios_parity({}, expected, tol=1e-3)["detail"]


def test_edit_cache_parity_compares_each_step_with_its_single_shot_row():
    single = good_outputs()
    steps = [{"edit_ratios": single["edit_ratios"][:, t:t + 1]} for t in range(SEQ)]
    assert check_edit_cache_parity(single, steps, tol=1e-4)["passed"]
    steps[2] = {"edit_ratios": steps[2]["edit_ratios"] + 0.2}
    assert not check_edit_cache_parity(single, steps, tol=1e-4)["passed"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_validate_mlp.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_edit_cache_parity'`.

- [ ] **Step 3: Implement the pure checks and the torch reference**

In `validate.py`, extend the module docstring list with:

```
9. edit_ratios match PyTorch forward hooks on the stock model, and the
   cached step-by-step rows match the single-shot rows
10. stream_parts add up to last_hidden_state's last row, and parts · E
    add up to the logits of the top-k ids
```

Add after `check_topk_parity`:

```python
def check_stream_parts_sum(outputs: dict, tol: float) -> dict:
    """The parts (embedding, attention edits, MLP edits, each scaled by the
    final norm) must add up to what the lm_head multiplies."""
    for name in ("stream_parts", "last_hidden_state"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    parts, last = outputs["stream_parts"], outputs["last_hidden_state"]
    if parts.ndim != 3 or parts.shape[0] != 1 or parts.shape[2] != last.shape[2]:
        return {"passed": False, "detail": f"stream_parts shape {parts.shape}, expected (1, 1 + 2·layers, {last.shape[2]})"}
    diff = float(np.max(np.abs(parts[0].sum(axis=0) - last[0, -1])))
    return {"passed": diff <= tol, "detail": f"max|Σ parts − last_hidden|={diff:.2e} (atol {tol})"}


def check_votes_sum(outputs: dict, tol: float) -> dict:
    """parts · E[c] summed over the parts must reproduce each top-k logit —
    the identity the votes card shows."""
    for name in ("stream_parts", "topk_ids", "topk_embeds", "logits"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    votes = outputs["stream_parts"][0] @ outputs["topk_embeds"][0].T     # [parts, k]
    want = outputs["logits"][0, -1, outputs["topk_ids"][0]]
    diff = float(np.max(np.abs(votes.sum(axis=0) - want)))
    return {"passed": diff <= tol, "detail": f"max|Σ votes − logit|={diff:.2e} (atol {tol})"}


def check_edit_ratios_parity(outputs: dict, expected: np.ndarray, tol: float) -> dict:
    """edit_ratios [1, seq, layers, 2] against a reference [seq, layers, 2]."""
    if "edit_ratios" not in outputs:
        return {"passed": False, "detail": "graph has no edit_ratios output"}
    got = outputs["edit_ratios"]
    if got.shape != (1, *expected.shape):
        return {"passed": False, "detail": f"edit_ratios shape {got.shape}, expected (1, {', '.join(map(str, expected.shape))})"}
    diff = float(np.max(np.abs(got[0] - expected)))
    return {"passed": diff <= tol, "detail": f"max|Δratio|={diff:.2e} (atol {tol})"}


def check_edit_cache_parity(single: dict, steps: list[dict], tol: float) -> dict:
    """Step t of a cached run feeds one token; its edit_ratios row must equal
    row t of the single-shot run over the whole sequence."""
    if "edit_ratios" not in single or any("edit_ratios" not in s for s in steps):
        return {"passed": False, "detail": "graph has no edit_ratios output"}
    diff = max(float(np.max(np.abs(s["edit_ratios"][0, 0] - single["edit_ratios"][0, t])))
               for t, s in enumerate(steps))
    return {"passed": diff <= tol, "detail": f"max|Δratio| step-by-step vs single-shot={diff:.2e} (atol {tol})"}


def torch_edit_ratios(ids: np.ndarray) -> np.ndarray:
    """Reference [seq, layers, 2] from the stock PyTorch model: forward hooks
    record each decoder layer's input x, its attention output a and its MLP
    output m; ratios are ‖a‖/‖x‖ and ‖m‖/‖x + a‖. Heavy (loads the model)."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        STOCK_MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager").eval()
    rec: dict = {}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, l=l):
            rec[("x", l)] = (args[0] if args else kwargs["hidden_states"]).detach()
        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(
            lambda mod, args, out, l=l: rec.__setitem__(("a", l), out[0].detach())))
        hooks.append(layer.mlp.register_forward_hook(
            lambda mod, args, out, l=l: rec.__setitem__(("m", l), out.detach())))
    with torch.no_grad():
        model(torch.tensor(ids))
    for h in hooks:
        h.remove()
    layers = len(model.model.layers)
    out = np.zeros((ids.shape[1], layers, 2), dtype=np.float32)
    for l in range(layers):
        x, a, m = rec[("x", l)][0], rec[("a", l)][0], rec[("m", l)][0]
        out[:, l, 0] = (a.norm(dim=-1) / x.norm(dim=-1)).numpy()
        out[:, l, 1] = (m.norm(dim=-1) / (x + a).norm(dim=-1)).numpy()
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools/export && uv run pytest tests/test_validate_mlp.py -q`
Expected: 5 PASS.

- [ ] **Step 5: Wire the checks into `run`**

In `run`, before the `for variant in [...]` loop, add:

```python
    ref_ids = _tokenize(model_dir, PROMPTS[0])
    ref_ratios = torch_edit_ratios(ref_ids)
```

Inside the loop, after the `topk-parity` line, add:

```python
        fp32 = variant == "model.onnx"
        checks[f"{variant}:edit-ratios-parity"] = check_edit_ratios_parity(ro_out, ref_ratios, 1e-3 if fp32 else 0.1)
        checks[f"{variant}:stream-parts-sum"] = check_stream_parts_sum(ro_out, 1e-3)
        checks[f"{variant}:votes-sum"] = check_votes_sum(ro_out, 1e-3 if fp32 else 1e-2)
        checks[f"{variant}:edit-cache-parity"] = check_edit_cache_parity(
            ro_out, _drive_cached_stepwise(sess, ro_ids), 1e-3 if fp32 else 0.1)
```

(`ro_ids` equals `ref_ids`: both tokenize `PROMPTS[0]`.) The quantized tolerances (0.1 on ratios of order 0.1–2) are a first guess; if the real export fails them with ratios that look right, widen with a ledgered ruling and say so in the export README.

- [ ] **Step 6: Model card and README**

In `publish.py`'s model card text, after the readout paragraph (the one ending "summing to the logit."), add:

```
For the MLP cards it also exposes `edit_ratios` (`[batch, seq, 30, 2]`,
‖edit‖ / ‖stream before‖ per position, layer and block: 0 attention, 1 MLP)
and `stream_parts` (`[batch, 61, 576]`, the last position's embedding,
30 attention edits and 30 MLP edits, each multiplied by the final norm's
gain and 1/rms, so they add up to `last_hidden_state`'s last row).
```

and extend the validation sentence (line ~43) with ", edit-ratio parity with PyTorch hooks, and stream-part / vote sums".

In `tools/export/README.md`:
- first paragraph: after "the readout (`last_hidden_state`, `topk_ids`, `topk_embeds`)" add "and the MLP internals (`edit_ratios`, `stream_parts`)".
- step 6: extend the must-PASS list with `*:edit-ratios-parity`, `*:stream-parts-sum`, `*:votes-sum`, `*:edit-cache-parity`.
- step 7: `--tag v3` → `--tag v4`, "(`v4`)" → "(`v5`)" in the retry sentence.

- [ ] **Step 7: Run the export suite and commit**

Run: `cd tools/export && uv run pytest -q`
Expected: all PASS.

```bash
git add tools/export/src/tsumugi_export/validate.py tools/export/src/tsumugi_export/publish.py tools/export/README.md tools/export/tests/test_validate_mlp.py
git commit -m "feat(export): validate edit ratios against PyTorch hooks and the stream-part sums"
```

---

### Task 3: Trace format — `edits` event, `votes` field, validation, pacing, fixture

**Files:**
- Modify: `src/trace/types.ts`, `src/trace/validate.ts`, `src/player/pacing.ts`, `src/viz/selectors.ts` (STAGE_OF only), `src/test/fixtures.ts`
- Test: `src/trace/validate.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface LayerVotes { embed: number; attn: number[]; mlp: number[] }
  // LogitCandidate gains: votes?: LayerVotes
  // TraceEvent gains: | { type: 'edits'; cycle: number; start: number; attn: number[][]; mlp: number[][] }
  ```
  `FixtureTraceOpts.mlp?: boolean` — adds an `edits` event after each cycle's `attention` and `votes` on every candidate (votes add up to the logit exactly); `fixtureVotes(id: number, logit: number, layers: number): LayerVotes`.

- [ ] **Step 1: Write the failing tests**

Append to `src/trace/validate.test.ts` (keep its existing imports; add `buildFixtureTrace` to the fixtures import if missing):

```ts
test('a trace with MLP data validates; old traces still do', () => {
  expect(validateTrace(buildFixtureTrace({ mlp: true, readout: true }))).toEqual([])
  expect(validateTrace(buildFixtureTrace())).toEqual([])
})

test('edits rows must be one value per layer and fit the sequence', () => {
  const t = buildFixtureTrace({ mlp: true })
  const i = t.findIndex((e) => e.type === 'edits')
  const e = t[i] as Extract<TraceEvent, { type: 'edits' }>
  t[i] = { ...e, attn: e.attn.map((r) => r.slice(1)) }
  expect(validateTrace(t).join()).toMatch(/edits cycle 0 attn row 0 is not 3 finite numbers/)
  t[i] = { ...e, start: 5 }
  expect(validateTrace(t).join()).toMatch(/edits cycle 0 rows 5\.\.7 exceed seqLen 2/)
  t[i] = { ...e, mlp: e.mlp.slice(1) }
  expect(validateTrace(t).join()).toMatch(/edits cycle 0 has 2 attn rows but 1 mlp rows/)
})

test('edits sit after the layers, before logits', () => {
  const t = buildFixtureTrace({ mlp: true })
  const i = t.findIndex((e) => e.type === 'edits')
  const [ev] = t.splice(i, 1)
  t.splice(3, 0, ev)   // right after embed, before the layers
  expect(validateTrace(t).join()).toMatch(/unexpected edits in phase layer/)
})

test('votes need one value per layer', () => {
  const t = buildFixtureTrace({ mlp: true })
  const lg = t.find((e) => e.type === 'logits') as Extract<TraceEvent, { type: 'logits' }>
  lg.topK[0] = { ...lg.topK[0], votes: { ...lg.topK[0].votes!, mlp: [1] } }
  expect(validateTrace(t).join()).toMatch(/candidate \d+ votes are not 3 finite numbers per block/)
})
```

Add `import type { TraceEvent } from './types'` if the file lacks it.

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/trace/validate.test.ts`
Expected: FAIL — TypeScript/vitest error on the unknown `mlp` option or `'edits'` type, or the assertions fail.

- [ ] **Step 3: Implement**

`src/trace/types.ts` — add above `LogitCandidate`:

```ts
// Real mode with the v4 export: one candidate's score split into the
// embedding's share and one share per attention and MLP block, each part
// scaled by the final norm, so embed + Σ attn + Σ mlp = the logit.
export interface LayerVotes { embed: number; attn: number[]; mlp: number[] }
```

In `LogitCandidate` add `votes?: LayerVotes` after `dot?`. In `TraceEvent` add after the `attention` line:

```ts
  // v4 export: ‖edit‖/‖stream before‖ per fed position (rows start at `start`) and layer
  | { type: 'edits'; cycle: number; start: number; attn: number[][]; mlp: number[][] }
```

`src/player/pacing.ts` — add `edits: 1.5,` after `attention: 2,`.

`src/viz/selectors.ts` — in `STAGE_OF` add `edits: 'layers',`.

`src/trace/validate.ts`:
- Before the loop, add `let layerTotal = 0` and `let seqLen = 0`.
- In the `layer` branch, before `layerIdx++`, add `layerTotal = e.total`.
- In the `embed` branch (the `e.type === 'embed' && phase === 'embed'` block), first line: `seqLen = e.seqLen`.
- After the `attention` branch add:

```ts
    if (e.type === 'edits') {
      // optional slot, like attention: after the cycle's layers, before its logits
      if (phase !== 'logits') { errs.push(`unexpected edits in phase ${phase}`); continue }
      const ok = (r: unknown) => Array.isArray(r) && r.length === layerTotal
        && r.every((v) => typeof v === 'number' && Number.isFinite(v))
      if (e.attn.length !== e.mlp.length)
        errs.push(`edits cycle ${e.cycle} has ${e.attn.length} attn rows but ${e.mlp.length} mlp rows`)
      e.attn.forEach((r, i) => { if (!ok(r)) errs.push(`edits cycle ${e.cycle} attn row ${i} is not ${layerTotal} finite numbers`) })
      e.mlp.forEach((r, i) => { if (!ok(r)) errs.push(`edits cycle ${e.cycle} mlp row ${i} is not ${layerTotal} finite numbers`) })
      if (e.start < 0 || e.start + e.attn.length > seqLen)
        errs.push(`edits cycle ${e.cycle} rows ${e.start}..${e.start + e.attn.length} exceed seqLen ${seqLen}`)
      continue
    }
```

- In the final `logits` check loop, after the `row`/`dot` loop, inside `for (const c of e.topK)`, add:

```ts
        const v = c.votes
        if (v !== undefined && (typeof v.embed !== 'number' || !Number.isFinite(v.embed)
          || !finite(v.attn) || !finite(v.mlp) || v.attn.length !== layerTotal || v.mlp.length !== layerTotal))
          errs.push(`logits cycle ${e.cycle} candidate ${c.id} votes are not ${layerTotal} finite numbers per block`)
```

(`layerTotal` holds the last layer total seen, which is the run's layer count by the time the final loop runs.)

`src/test/fixtures.ts`:
- Add `mlp?: boolean` to `FixtureTraceOpts` with comment `// edits events and per-candidate votes (v4 export data)`.
- Add `LayerVotes` to the type import and, after `fixtureReadout`:

```ts
// Deterministic votes whose parts add up to the logit exactly (the embedding
// share absorbs the rounding), because the votes card shows that identity.
export function fixtureVotes(id: number, logit: number, layers: number): LayerVotes {
  const attn = Array.from({ length: layers }, (_, l) => r3(0.2 * Math.sin(id * 0.3 + l)))
  const mlp = Array.from({ length: layers }, (_, l) => r3((logit / layers) * 0.8 + 0.4 * Math.cos(id * 0.5 + l * 0.7)))
  const rest = [...attn, ...mlp].reduce((a, b) => a + b, 0)
  return { embed: Math.round((logit - rest) * 1e6) / 1e6, attn, mlp }
}

// Deterministic edit ratio for a position, layer and block, in [0, 1.4].
export const fixtureRatio = (pos: number, layer: number, block: 0 | 1) =>
  r3(0.7 + 0.7 * Math.sin(pos * 0.9 + layer * 0.5 + block * 1.3))
```

- In `buildFixtureTrace`, after the `attention` push:

```ts
    if (opts.mlp) {
      const start = c === 0 ? 0 : promptTokens.length + c - 1
      const positions = Array.from({ length: fed.length }, (_, i) => start + i)
      events.push({ type: 'edits', cycle: c, start,
        attn: positions.map((p) => Array.from({ length: layers }, (_, l) => fixtureRatio(p, l, 0))),
        mlp: positions.map((p) => Array.from({ length: layers }, (_, l) => fixtureRatio(p, l, 1))) })
    }
```

- Replace the `scored` → logits push so votes are added when `opts.mlp`:

```ts
    const withVotes = (s: LogitCandidate): LogitCandidate =>
      opts.mlp ? { ...s, votes: fixtureVotes(s.id, s.logit, layers) } : s
    events.push(opts.readout
      ? { type: 'logits', cycle: c, lastHidden: fixtureLastHidden(c),
          topK: scored.map((s) => withVotes({ ...s, ...fixtureReadout(s.id, s.logit) })) }
      : { type: 'logits', cycle: c, topK: scored.map(withVotes) })
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/trace src/player src/viz/selectors.test.ts && npx tsc -b`
Expected: PASS; `tsc` clean.

- [ ] **Step 5: Commit**

```bash
git add src/trace src/player/pacing.ts src/viz/selectors.ts src/test/fixtures.ts
git commit -m "feat(trace): edits event and per-candidate votes, optional for old traces"
```

---

### Task 4: Worker — `extractEdits`, `computeVotes`, wiring

**Files:**
- Create: `src/engine/transformers/mlpData.ts`, `src/engine/transformers/mlpData.test.ts`
- Modify: `src/engine/transformers/readout.ts` (export `sameDims` and `r3`), `src/engine/transformers/worker.ts`

**Interfaces:**
- Consumes: `Tensorish` from `readout.ts`; `LayerVotes` from Task 3.
- Produces:
  ```ts
  export interface MlpOutputs { edit_ratios?: Tensorish; stream_parts?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish }
  export type EditsResult = { status: 'absent' } | { status: 'bad-shape'; detail: string } | { status: 'ok'; attn: number[][]; mlp: number[][] }
  export type VotesResult = { status: 'absent' } | { status: 'bad-shape'; detail: string } | { status: 'ok'; byId: Map<number, LayerVotes> }
  export function extractEdits(out: MlpOutputs, fed: number, layers: number): EditsResult
  export function computeVotes(out: MlpOutputs, layers: number, dims: number, logitOf: (id: number) => number): VotesResult
  ```

- [ ] **Step 1: Write the failing tests**

Create `src/engine/transformers/mlpData.test.ts`:

```ts
import { expect, test } from 'vitest'
import { computeVotes, extractEdits } from './mlpData'

const L = 2, DIMS = 3

function ratios(fed: number) {
  // value = pos + layer/10 + block/100, so every cell is distinguishable
  const data = new Float32Array(fed * L * 2)
  for (let p = 0; p < fed; p++) for (let l = 0; l < L; l++) for (let b = 0; b < 2; b++)
    data[(p * L + l) * 2 + b] = p + l / 10 + b / 100
  return { dims: [1, fed, L, 2], data }
}

test('extractEdits: absent, bad shape, ok split into attention and MLP rows', () => {
  expect(extractEdits({}, 2, L)).toEqual({ status: 'absent' })
  const bad = extractEdits({ edit_ratios: ratios(3) }, 2, L)
  expect(bad.status).toBe('bad-shape')
  if (bad.status === 'bad-shape') expect(bad.detail).toMatch(/edit_ratios has shape \[1, 3, 2, 2\], expected \[1, 2, 2, 2\]/)
  const r = extractEdits({ edit_ratios: ratios(2) }, 2, L)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.attn).toEqual([[0, 0.1], [1, 1.1]])
  expect(r.mlp).toEqual([[0.01, 0.11], [1.01, 1.11]])
})

// parts: embedding, attn L0, attn L1, mlp L0, mlp L1 — rows of 3 dims
const PARTS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.5, 0, 0.5]]
const ROWS: Record<number, number[]> = { 7: [2, 0, 1], 4: [0, 3, 0] }
const exact = (id: number) => PARTS.map((p) => p.reduce((s, v, d) => s + v * ROWS[id][d], 0))

function voteOutputs(ids = [7, 4]) {
  return {
    stream_parts: { dims: [1, 1 + 2 * L, DIMS], data: Float32Array.from(PARTS.flat()) },
    topk_ids: { dims: [1, ids.length], data: BigInt64Array.from(ids.map(BigInt)) },
    topk_embeds: { dims: [1, ids.length, DIMS], data: Float32Array.from(ids.flatMap((id) => ROWS[id])) },
  }
}
const logitOf = (id: number) => exact(id).reduce((a, b) => a + b, 0)

test('computeVotes: per candidate, the parts dotted with its row, split by block', () => {
  const r = computeVotes(voteOutputs(), L, DIMS, logitOf)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.byId.get(7)).toEqual({ embed: 2, attn: [0, 1], mlp: [2, 1.5] })
  expect(r.byId.get(4)).toEqual({ embed: 0, attn: [3, 0], mlp: [3, 0] })
})

test('computeVotes: absent without stream_parts; bad shape names the output', () => {
  expect(computeVotes({}, L, DIMS, logitOf)).toEqual({ status: 'absent' })
  const o = voteOutputs()
  o.stream_parts.dims = [1, 4, DIMS]
  const r = computeVotes(o, L, DIMS, logitOf)
  if (r.status !== 'bad-shape') throw new Error(r.status)
  expect(r.detail).toMatch(/stream_parts/)
})

test('computeVotes: votes that do not add up to the logit are refused', () => {
  const r = computeVotes(voteOutputs(), L, DIMS, (id) => logitOf(id) + 1)
  if (r.status !== 'bad-shape') throw new Error(r.status)
  expect(r.detail).toMatch(/votes for id 7 add up to/)
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/engine/transformers/mlpData.test.ts`
Expected: FAIL — cannot resolve `./mlpData`.

- [ ] **Step 3: Implement**

In `readout.ts`, change `const r3 = …` to `export const r3 = …` and `const sameDims = …` to `export const sameDims = …`.

Create `src/engine/transformers/mlpData.ts`:

```ts
import type { LayerVotes } from '../../trace/types'
import { r3, sameDims, type Tensorish } from './readout'

// The v4 export's MLP outputs turned into trace data. Pure, so the shape
// policy is unit-testable outside the worker (like readout.ts).
//   edit_ratios [1, fed, layers, 2] → attn and mlp rows per fed position
//   stream_parts [1, 1 + 2·layers, dims] · topk_embeds rows → per candidate
//   id its votes: embedding, attention per layer, MLP per layer
export interface MlpOutputs { edit_ratios?: Tensorish; stream_parts?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish }
export type EditsResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; attn: number[][]; mlp: number[][] }
export type VotesResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; byId: Map<number, LayerVotes> }

export function extractEdits(out: MlpOutputs, fed: number, layers: number): EditsResult {
  const t = out.edit_ratios
  if (!t) return { status: 'absent' }
  const want = [1, fed, layers, 2]
  if (!sameDims(t, want) || t.data.length !== fed * layers * 2)
    return { status: 'bad-shape', detail: `edit_ratios has shape [${t.dims.join(', ')}], expected [${want.join(', ')}]` }
  const at = (p: number, l: number, b: number) => r3(Number(t.data[(p * layers + l) * 2 + b]))
  const rows = (b: number) => Array.from({ length: fed }, (_, p) => Array.from({ length: layers }, (_, l) => at(p, l, b)))
  return { status: 'ok', attn: rows(0), mlp: rows(1) }
}

// Votes must reproduce the logit the card sits next to; a gap beyond
// rounding means the graph's parts are wrong, so the votes are dropped.
const tolerance = (logit: number) => 0.05 + 0.01 * Math.abs(logit)

export function computeVotes(out: MlpOutputs, layers: number, dims: number, logitOf: (id: number) => number): VotesResult {
  const { stream_parts: sp, topk_ids: ids, topk_embeds: emb } = out
  if (!sp) return { status: 'absent' }
  if (!ids || !emb) return { status: 'bad-shape', detail: 'stream_parts without topk_ids / topk_embeds' }
  const parts = 1 + 2 * layers
  if (!sameDims(sp, [1, parts, dims]) || sp.data.length !== parts * dims)
    return { status: 'bad-shape', detail: `stream_parts has shape [${sp.dims.join(', ')}], expected [1, ${parts}, ${dims}]` }
  const k = ids.dims[1]
  if (ids.dims.length !== 2 || !sameDims(emb, [1, k, dims]))
    return { status: 'bad-shape', detail: `topk_embeds has shape [${emb.dims.join(', ')}], expected [1, ${k}, ${dims}]` }
  const byId = new Map<number, LayerVotes>()
  for (let i = 0; i < k; i++) {
    const id = Number(ids.data[i])
    const v: number[] = []
    for (let p = 0; p < parts; p++) {
      let s = 0
      for (let d = 0; d < dims; d++) s += Number(sp.data[p * dims + d]) * Number(emb.data[i * dims + d])
      v.push(s)
    }
    const total = v.reduce((a, b) => a + b, 0)
    const logit = logitOf(id)
    if (Math.abs(total - logit) > tolerance(logit))
      return { status: 'bad-shape', detail: `votes for id ${id} add up to ${total.toFixed(3)}, logit is ${logit.toFixed(3)}` }
    byId.set(id, { embed: r3(v[0]), attn: v.slice(1, 1 + layers).map(r3), mlp: v.slice(1 + layers).map(r3) })
  }
  return { status: 'ok', byId }
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/engine/transformers`
Expected: PASS (new and existing readout tests).

- [ ] **Step 5: Wire into the worker**

In `worker.ts`:
- Import: `import { computeVotes, extractEdits } from './mlpData'` and add `LayerVotes` to the trace types import.
- Next to `let readoutOff = false` add:

```ts
  // MLP cards (v4 export): each flipped by a bad shape — never-fail.
  let editsOff = false
  let votesOff = false
```

- After the attention emit `try { … } catch { attnBroken = true }` block that ends with the showcase emit, add:

```ts
      if (!editsOff) {
        const r = extractEdits(out, nextInputIds.length, numLayers)
        if (r.status === 'ok')
          emit({ type: 'edits', cycle, start: allIds.length - nextInputIds.length, attn: r.attn, mlp: r.mlp })
        else if (r.status === 'bad-shape') {
          editsOff = true
          console.warn(`edit_ratios unusable (${r.detail}) — the edit-size card shows its note`)
        }
      }
```

- After the readout block (before `const candidates = …`), add:

```ts
      let votes = new Map<number, LayerVotes>()
      if (!votesOff) {
        const v = computeVotes(out, numLayers, dims, (id) => lastLogits[id])
        if (v.status === 'ok') votes = v.byId
        else if (v.status === 'bad-shape') {
          votesOff = true
          console.warn(`stream_parts unusable (${v.detail}) — the votes card shows its note`)
        }
      }
```

- Change the `candidates` map to also attach votes:

```ts
      const candidates = top.map((c) => {
        const r = readout.status === 'ok' ? readout.byId.get(c.id) : undefined
        const v = votes.get(c.id)
        return { ...c, ...(r ? { row: r.row, dot: r.dot } : {}), ...(v ? { votes: v } : {}) }
      })
```

- [ ] **Step 6: Type-check, run the suite, commit**

Run: `npx tsc -b && npx vitest run src/engine`
Expected: clean; PASS.

```bash
git add src/engine/transformers
git commit -m "feat(worker): edits events and per-candidate votes from the v4 outputs"
```

---

### Task 5: Selectors and maths for the cards

**Files:**
- Modify: `src/viz/selectors.ts`
- Create: `src/viz/mlpMath.ts`, `src/viz/mlpMath.test.ts`
- Test: `src/viz/selectors.test.ts`

**Interfaces:**
- Consumes: Task 3 types and `buildFixtureTrace({ mlp: true })`.
- Produces:
  ```ts
  // selectors.ts
  export interface EditRows { tokens: TokenInfo[]; attn: number[][]; mlp: number[][] }
  export function editRows(events: TraceEvent[], cursor: number): EditRows | undefined
  export type VotedCandidate = LogitCandidate & { votes: LayerVotes }
  export type CycleVotes = { status: 'ok'; candidates: VotedCandidate[] } | { status: 'pending' } | { status: 'none' }
  export function cycleVotes(events: TraceEvent[], cursor: number): CycleVotes
  export function viewedCycle(events: TraceEvent[], cursor: number): number   // the existing private cycleAt, exported
  // mlpMath.ts
  export function runningTotal(v: LayerVotes): number[]          // length 1 + 2·layers
  export function blockTotals(v: LayerVotes): { embed: number; attn: number; mlp: number }
  export function topMlpLayers(v: LayerVotes, n?: number): Array<{ layer: number; value: number }>
  export function votePeak(cands: Array<{ votes: LayerVotes }>): number
  export function ratioFill(v: number): string
  ```

- [ ] **Step 1: Write the failing tests**

Append to `src/viz/selectors.test.ts` (add imports `buildFixtureTrace` from `../test/fixtures` and `cycleVotes, editRows, viewedCycle` from `./selectors` as needed):

```ts
// mlp fixture (2 prompt tokens, 3 layers, 2 cycles), indices:
// 0 run-start, 1 tokenize, 2 embed c0, 3..5 layers, 6 attention, 7 edits, 8 logits, 9 softmax, 10 sample, 11 append,
// 12 embed c1, 13..15 layers, 16 attention, 17 edits, 18 logits, …
const mlpTrace = buildFixtureTrace({ mlp: true })

test('editRows: on cycle 0 the prompt rows, looked up even before the edits event is reached', () => {
  const r = editRows(mlpTrace, 3)!
  expect(r.tokens.map((t) => t.text)).toEqual(['The', ' cat'])
  expect(r.attn).toHaveLength(2)
  expect(r.mlp[1]).toHaveLength(3)
})

test('editRows: on cycle 1 the generated token gets its row', () => {
  const r = editRows(mlpTrace, 13)!
  expect(r.tokens.map((t) => t.text)).toEqual(['The', ' cat', ' sat'])
  expect(r.attn).toHaveLength(3)
})

test('editRows: never shows rows past the viewed cycle', () => {
  expect(editRows(mlpTrace, 8)!.attn).toHaveLength(2)
})

test('editRows: undefined without edits events', () => {
  expect(editRows(buildFixtureTrace(), 4)).toBeUndefined()
})

test('cycleVotes: ok on the layers stage (looks ahead to the cycle’s logits), pending before it exists, none without votes', () => {
  const ok = cycleVotes(mlpTrace, 4)
  expect(ok.status).toBe('ok')
  if (ok.status === 'ok') expect(ok.candidates[0].votes.mlp).toHaveLength(3)
  expect(cycleVotes(mlpTrace.slice(0, 7), 4)).toEqual({ status: 'pending' })
  expect(cycleVotes(buildFixtureTrace(), 4)).toEqual({ status: 'none' })
})

test('cycleVotes: none when any candidate lacks votes', () => {
  const t = buildFixtureTrace({ mlp: true })
  const lg = t[8] as Extract<TraceEvent, { type: 'logits' }>
  const { votes: _dropped, ...rest } = lg.topK[2]
  void _dropped
  lg.topK[2] = rest
  expect(cycleVotes(t, 4)).toEqual({ status: 'none' })
})

test('viewedCycle follows the cursor', () => {
  expect(viewedCycle(mlpTrace, 4)).toBe(0)
  expect(viewedCycle(mlpTrace, 14)).toBe(1)
})
```

Create `src/viz/mlpMath.test.ts`:

```ts
import { expect, test } from 'vitest'
import { blockTotals, ratioFill, runningTotal, topMlpLayers, votePeak } from './mlpMath'

const v = { embed: 1, attn: [0.5, -0.25], mlp: [2, -1] }

test('runningTotal: embedding first, then per layer attention then MLP; ends on the sum', () => {
  expect(runningTotal(v)).toEqual([1, 1.5, 3.5, 3.25, 2.25])
})

test('blockTotals splits the score into embedding, all attention, all MLP', () => {
  expect(blockTotals(v)).toEqual({ embed: 1, attn: 0.25, mlp: 1 })
})

test('topMlpLayers ranks by absolute vote', () => {
  expect(topMlpLayers({ embed: 0, attn: [0, 0, 0], mlp: [0.1, -3, 2] }, 2)).toEqual([
    { layer: 1, value: -3 }, { layer: 2, value: 2 },
  ])
})

test('votePeak is the largest absolute MLP vote, at least a tiny positive number', () => {
  expect(votePeak([{ votes: v }, { votes: { embed: 9, attn: [9, 9], mlp: [-4, 0] } }])).toBe(4)
  expect(votePeak([{ votes: { embed: 0, attn: [0], mlp: [0] } }])).toBe(1)
})

test('ratioFill darkens with the ratio and clamps at 1', () => {
  expect(ratioFill(0)).toBe('hsl(211 30% 94%)')
  expect(ratioFill(1)).toBe('hsl(211 55% 24%)')
  expect(ratioFill(3)).toBe(ratioFill(1))
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/selectors.test.ts src/viz/mlpMath.test.ts`
Expected: FAIL — missing exports / module.

- [ ] **Step 3: Implement**

In `selectors.ts`: rename `cycleAt` to `viewedCycle`, export it, and update its call in `stageEventIndex`. Extend the trace type import with `LayerVotes` and `LogitCandidate`. Append:

```ts
export interface EditRows { tokens: TokenInfo[]; attn: number[][]; mlp: number[][] }

// The edit-size heatmap for the viewed cycle: rows from every edits event up
// to and including that cycle (looked up in the whole trace, like the votes,
// so the Layers stage shows the cycle's rows before the player reaches the
// edits event), cut to the visible tokens. Undefined when the run has none.
export function editRows(events: TraceEvent[], cursor: number): EditRows | undefined {
  const cycle = viewedCycle(events, cursor)
  const attn: number[][] = [], mlp: number[][] = []
  for (const e of events) {
    if (e.type !== 'edits' || e.cycle > cycle) continue
    e.attn.forEach((row, i) => { attn[e.start + i] = row; mlp[e.start + i] = e.mlp[i] })
  }
  let n = 0
  while (n < attn.length && attn[n] && mlp[n]) n++   // contiguous from position 0
  if (n === 0) return undefined
  const { prompt, generated } = visibleTokens(events, cursor)
  const tokens = [...prompt, ...generated.map(({ id, text }) => ({ id, text }))]
  const rows = Math.min(n, tokens.length)
  return { tokens: tokens.slice(0, rows), attn: attn.slice(0, rows), mlp: mlp.slice(0, rows) }
}

export type VotedCandidate = LogitCandidate & { votes: LayerVotes }
export type CycleVotes =
  | { status: 'ok'; candidates: VotedCandidate[] }
  | { status: 'pending' }   // the viewed cycle's logits are not in the trace yet
  | { status: 'none' }      // they are, without votes for every candidate

// The votes card reads the viewed cycle's logits event, which comes after
// the layers in the trace: on the Layers stage that is a look ahead.
export function cycleVotes(events: TraceEvent[], cursor: number): CycleVotes {
  const cycle = viewedCycle(events, cursor)
  const logits = events.find((e): e is Extract<TraceEvent, { type: 'logits' }> => e.type === 'logits' && e.cycle === cycle)
  if (!logits) return { status: 'pending' }
  if (!logits.topK.every((c) => c.votes)) return { status: 'none' }
  return { status: 'ok', candidates: logits.topK as VotedCandidate[] }
}
```

Create `src/viz/mlpMath.ts`:

```ts
import type { LayerVotes } from '../trace/types'

// Numbers behind the MLP cards. Kept apart from the components so those
// files export only components.

// A candidate's score building up through the model: the embedding's share,
// then per layer its attention vote followed by its MLP vote.
export function runningTotal(v: LayerVotes): number[] {
  const out = [v.embed]
  let s = v.embed
  v.attn.forEach((a, l) => { s += a; out.push(s); s += v.mlp[l]; out.push(s) })
  return out
}

const sum = (a: number[]) => a.reduce((s, x) => s + x, 0)

export const blockTotals = (v: LayerVotes) => ({ embed: v.embed, attn: sum(v.attn), mlp: sum(v.mlp) })

export function topMlpLayers(v: LayerVotes, n = 3): Array<{ layer: number; value: number }> {
  return v.mlp.map((value, layer) => ({ layer, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, n)
}

// One symmetric colour scale for the whole grid, so cells compare across candidates.
export const votePeak = (cands: Array<{ votes: LayerVotes }>) =>
  cands.reduce((m, c) => Math.max(m, ...c.votes.mlp.map(Math.abs)), 0) || 1

// Edit size 0 → pale, ≥ 1 → deep indigo (the attention heatmap's ramp).
export function ratioFill(v: number): string {
  const t = Math.min(1, Math.max(0, v))
  return `hsl(211 ${Math.round(30 + 25 * t)}% ${Math.round(94 - 70 * t)}%)`
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/viz && npx tsc -b`
Expected: PASS; clean.

- [ ] **Step 5: Commit**

```bash
git add src/viz/selectors.ts src/viz/selectors.test.ts src/viz/mlpMath.ts src/viz/mlpMath.test.ts
git commit -m "feat(viz): selectors and maths for the edit-size and votes cards"
```

---

### Task 6: Shared candidate selection

**Files:**
- Create: `src/viz/candidateSelection.ts`, `src/viz/candidateSelection.test.ts`
- Modify: `src/viz/details/LogitsDetail.tsx`, `src/test/setup.ts`
- Test: `src/viz/details/LogitsDetail.test.tsx`

**Interfaces:**
- Produces:
  ```ts
  export const useCandidateSelection: UseBoundStore<…>   // state { pick: { cycle: number; index: number } | null; select(cycle: number, index: number): void }
  export function selectedCandidate(pick: { cycle: number; index: number } | null, cycle: number, count: number): number
  export function resetCandidateSelection(): void
  ```

- [ ] **Step 1: Write the failing tests**

Create `src/viz/candidateSelection.test.ts`:

```ts
import { expect, test } from 'vitest'
import { resetCandidateSelection, selectedCandidate, useCandidateSelection } from './candidateSelection'

test('selectedCandidate: the pick on its own cycle, else the top candidate', () => {
  expect(selectedCandidate(null, 0, 3)).toBe(0)
  expect(selectedCandidate({ cycle: 1, index: 2 }, 1, 3)).toBe(2)
  expect(selectedCandidate({ cycle: 1, index: 2 }, 0, 3)).toBe(0)
})

test('selectedCandidate: an index past this cycle’s candidates falls back to 0', () => {
  expect(selectedCandidate({ cycle: 0, index: 5 }, 0, 3)).toBe(0)
})

test('select stores the pick; reset clears it', () => {
  useCandidateSelection.getState().select(2, 1)
  expect(useCandidateSelection.getState().pick).toEqual({ cycle: 2, index: 1 })
  resetCandidateSelection()
  expect(useCandidateSelection.getState().pick).toBeNull()
})
```

Append to `src/viz/details/LogitsDetail.test.tsx`:

```ts
test('the selected candidate survives a remount (switching stages and back)', () => {
  const events = buildFixtureTrace({ readout: true })
  render(<LogitsDetail events={events} cursor={7} />)
  fireEvent.click(screen.getAllByTestId('readout-candidate')[1])
  cleanup()
  render(<LogitsDetail events={events} cursor={7} />)
  expect(screen.getAllByTestId('readout-candidate')[1]).toHaveAttribute('data-selected', 'true')
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/candidateSelection.test.ts src/viz/details/LogitsDetail.test.tsx`
Expected: FAIL — module missing; the remount test fails (local state resets to 0).

- [ ] **Step 3: Implement**

Create `src/viz/candidateSelection.ts`:

```ts
import { create } from 'zustand'

// The candidate the Logits card and the votes card both highlight. Keyed by
// cycle: moving to another cycle falls back to the top candidate. A store,
// not component state, so the pick survives switching stages.
interface CandidateSelection {
  pick: { cycle: number; index: number } | null
  select: (cycle: number, index: number) => void
}

export const useCandidateSelection = create<CandidateSelection>((set) => ({
  pick: null,
  select: (cycle, index) => set({ pick: { cycle, index } }),
}))

export const resetCandidateSelection = () => useCandidateSelection.setState({ pick: null })

export function selectedCandidate(pick: CandidateSelection['pick'], cycle: number, count: number): number {
  return pick && pick.cycle === cycle && pick.index < count ? pick.index : 0
}
```

In `LogitsDetail.tsx`: drop `useState` from the React import (remove the import line if nothing else uses it), import `{ selectedCandidate, useCandidateSelection } from '../candidateSelection'`, and replace the state lines:

```ts
  const pick = useCandidateSelection((s) => s.pick)
  const select = useCandidateSelection((s) => s.select)
```

```ts
  const selected = selectedCandidate(pick, logits.cycle, logits.topK.length)
```

and `onSelect={(index) => select(logits.cycle, index)}`. Update the header comment: "…and shows the selected candidate (shared with the votes card through candidateSelection; snaps to the top candidate on another cycle)."

In `src/test/setup.ts` add:

```ts
import { afterEach } from 'vitest'
import { resetCandidateSelection } from '../viz/candidateSelection'

// the candidate pick is a global store; tests start from the top candidate
afterEach(() => resetCandidateSelection())
```

(merge the `vitest` import with the existing `vi` import.)

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/viz`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/viz/candidateSelection.ts src/viz/candidateSelection.test.ts src/viz/details/LogitsDetail.tsx src/viz/details/LogitsDetail.test.tsx src/test/setup.ts
git commit -m "refactor(logits): candidate selection in a store shared across cards"
```

---

### Task 7: Glossary — `votes` entry, `edit` sections, run summary

**Files:**
- Create: `src/glossary/entries/votes.tsx`
- Modify: `src/glossary/types.ts`, `src/glossary/registry.ts`, `src/glossary/runSummary.ts`, `src/glossary/callouts.ts`, `src/glossary/entries/edit.tsx`, `src/glossary/entries/mlp.tsx`, `src/test/glossaryFixture.ts`
- Test: `src/glossary/registry.test.tsx`, `src/glossary/runSummary.test.ts`

**Interfaces:**
- Consumes: `editRows`, `cycleVotes` (Task 5).
- Produces: `EDIT_CALLOUTS: CalloutTarget[]` in `callouts.ts` (used by Task 8); `GLOSSARY_IDS` gains `'votes'` after `'mlp'`; `RunSummary` gains `edits?: { token: string; layer: number; ratio: number }` and `votes?: { candidate: string; layer: number; value: number }`.

- [ ] **Step 1: Write the failing tests**

In `registry.test.tsx`: import `EDIT_CALLOUTS` (defined in Step 3 below) and extend the chip test's list to `[...EMBEDDING_CALLOUTS, ...logitsCallouts(3, '49 152'), ...SOFTMAX_CALLOUTS, ...EDIT_CALLOUTS]`. Append:

```tsx
test('the votes entry explains the frozen norm and how it differs from the logit lens', () => {
  const { container } = render(<>{getEntry('votes')!.body()}</>)
  expect(container.textContent).toMatch(/logit lens/i)
  expect(container.textContent).toMatch(/root mean square/i)
  expect(container.querySelector('[data-formula-error]')).toBeNull()
})

test('the MLP entry no longer says the app hides its internals', () => {
  const { container } = render(<>{getEntry('mlp')!.body()}</>)
  expect(container.textContent).not.toMatch(/does not show the MLP’s internals/)
  expect(container.textContent).toMatch(/What each layer voted for/)
})
```

In `runSummary.test.ts` append (import `buildFixtureTrace`):

```ts
test('MLP facts: the last token’s largest MLP edit, and the MLP layer that added most to the top candidate', () => {
  const t = buildFixtureTrace({ mlp: true, mode: 'real' })
  const s = runSummary(t, 4)!
  expect(s.edits?.token).toBe(' cat')
  expect(s.edits?.ratio).toBeGreaterThan(0)
  expect(s.votes?.candidate).toBe(' sat')
  expect(s.votes?.layer).toBeGreaterThanOrEqual(0)
  expect(runSummary(buildFixtureTrace(), 4)!.edits).toBeUndefined()
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/glossary`
Expected: FAIL — unknown sections `layer-0` etc., no `votes` entry, no `edits` in the summary.

- [ ] **Step 3: Implement**

`callouts.ts`: append

```ts
export const EDIT_CALLOUTS: CalloutTarget[] = [
  { label: 'why layer 0 is so dark', id: 'edit', section: 'layer-0' },
  { label: 'quiet middle, loud end', id: 'edit', section: 'quiet-middle' },
  { label: 'attention vs MLP', id: 'edit', section: 'attn-vs-mlp' },
]
```

`types.ts`: add `'votes'` after `'mlp'` in `GLOSSARY_IDS`; in `RunSummary` add:

```ts
  edits?: { token: string; layer: number; ratio: number }      // largest MLP edit of the last visible token
  votes?: { candidate: string; layer: number; value: number }  // MLP layer adding most to the top candidate
```

`runSummary.ts`: import `cycleVotes, editRows` alongside `latestOfType`; before the `return`, add:

```ts
  const ed = editRows(events, at)
  const lastRow = ed?.mlp.at(-1)
  const edits = ed && lastRow
    ? (() => { const layer = lastRow.indexOf(Math.max(...lastRow))
        return { token: ed.tokens.at(-1)!.text, layer, ratio: lastRow[layer] } })()
    : undefined
  const cv = cycleVotes(events, at)
  const votes = cv.status === 'ok'
    ? (() => { const top = cv.candidates[0]; const layer = top.votes.mlp.indexOf(Math.max(...top.votes.mlp))
        return { candidate: top.text, layer, value: top.votes.mlp[layer] } })()
    : undefined
```

and spread into the result: `...(edits ? { edits } : {}), ...(votes ? { votes } : {})`.

`src/test/glossaryFixture.ts`: add to the defaults `edits: { token: ' cat', layer: 29, ratio: 1.42 }, votes: { candidate: ' sat', layer: 27, value: 2.31 },`. In `registry.test.tsx`'s `bare` run add `edits: undefined, votes: undefined`.

`entries/edit.tsx`: import `Section` from `'../Section'`. Before the `gl-aside` paragraph add:

```tsx
      <Section id="layer-0" title="Why layer 0 is so dark">
        <p>The stream starts as a token’s embedding, which is small. The first blocks write edits as large as the
          embedding or larger, so their ratio is high. It does not mean layer 0 does the most important work, only
          that there is little to compare against yet.</p>
      </Section>
      <Section id="quiet-middle" title="Quiet middle, loud end">
        <p>As the stream grows, each edit is compared with a larger vector, so middle layers often look quiet.
          The last layers usually write large edits again: they prepare the vector for the readout, pushing it
          towards the next token’s row of E.</p>
      </Section>
      <Section id="attn-vs-mlp" title="Attention vs MLP">
        <p>Attention edits move information between positions; MLP edits work on each position alone. Seeing which
          of the two writes more at a layer tells you whether that layer mostly gathers context or mostly
          transforms what is already there. These are typical patterns in small models, not rules.</p>
      </Section>
```

Change its `inRun` to:

```tsx
  inRun: (run) => run.stream ? (
    <>
      <p>Each layer adds two edits of [{run.stream.seqLen} × {run.stream.dims}] to your stream, one from attention
        and one from the MLP: {run.stream.layers * 2} in total.</p>
      {run.edits && <p>The largest MLP edit for “{run.edits.token}” is at layer {run.edits.layer}:{' '}
        {run.edits.ratio.toFixed(2)} times the size of the stream it is added to.</p>}
    </>
  ) : null,
```

`entries/mlp.tsx`: replace `<p className="gl-aside">Tsumugi does not show the MLP’s internals yet.</p>` with:

```tsx
      <p>In a real run the Layers detail shows what the MLPs do: <em>How much each layer changes the vector</em>{' '}
        compares each MLP edit with the stream, and <em>What each layer voted for</em> shows how much each layer’s
        MLP added to each candidate’s score. See <Term id="votes">per-layer votes</Term>.</p>
```

(`Term` is already imported there; check.) Add `'votes'` to its `related`.

Create `src/glossary/entries/votes.tsx`:

```tsx
import { Formula } from '../../viz/Formula'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const votes: GlossaryEntry = {
  id: 'votes',
  title: 'Per-layer votes',
  summary: 'How much each block added to or took from a candidate’s score, adding up exactly to that score.',
  stage: 'layers',
  related: ['logit', 'mlp', 'edit'],
  body: () => (
    <>
      <p>A <Term id="logit">score</Term> is the final vector, after the final norm, dotted with the candidate’s row
        of E. The final vector is a sum: the embedding plus every <Term id="edit">edit</Term>. If the norm were a
        plain linear step, the score would split into one share per part.</p>
      <p>At the last position the norm divides the whole vector by one number, its root mean square, and multiplies
        each dimension by a fixed gain. Holding that number at its real value for this position makes the norm
        linear, so the split is exact:</p>
      <Formula tex={String.raw`\begin{aligned} \text{score}(c) &= \tilde e \cdot E_c + \sum_{\ell} \tilde a_\ell \cdot E_c + \sum_{\ell} \tilde m_\ell \cdot E_c \\ \tilde v &= g \odot v \,/\, \operatorname{rms}(x) \end{aligned}`} />
      <p>Each term is one block’s vote. A positive vote raised the score; a negative one lowered it. The votes card
        shows the MLP votes per layer, and a running total that adds the embedding first, then each layer’s
        attention vote and MLP vote in order. It ends exactly on the Logits card’s score.</p>
      <p className="gl-aside">This is related to, but not the same as, the <em>logit lens</em>, which applies the
        final norm again to every partial sum. That shows what the model would predict if it stopped at a layer,
        but its steps do not add up to the final score.</p>
    </>
  ),
  inRun: (run) => run.votes ? (
    <p>At this cycle the MLP of layer {run.votes.layer} added the most to “{run.votes.candidate}”: {' '}
      {run.votes.value >= 0 ? '+' : '−'}{Math.abs(run.votes.value).toFixed(2)} to its score.</p>
  ) : null,
}
```

`registry.ts`: import `votes` and insert it after `mlp` in `ENTRIES`.

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/glossary && npx tsc -b && npx oxlint src`
Expected: PASS; clean. Then the whole unit suite: `npx vitest run` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/glossary src/test/glossaryFixture.ts
git commit -m "docs(glossary): per-layer votes entry, edit sections for the card chips"
```

---

### Task 8: EditSizeCard

**Files:**
- Create: `src/viz/EditSizeCard.tsx`, `src/viz/EditSizeCard.test.tsx`
- Modify: `src/index.css`

**Interfaces:**
- Consumes: `EditRows` (Task 5), `ratioFill` (Task 5), `EDIT_CALLOUTS` (Task 7), `Term`, `Formula`, `visibleToken`.
- Produces: `export function EditSizeCard({ rows, note }: { rows?: EditRows; note: string })`. Test ids: `edit-size-card`, `edit-size-info`, `edit-size-note`, `edit-size-formula`, `edit-map` (`data-block`), `edit-cell` (`data-block`, `data-row`, `data-layer`), `edit-row-label` (`data-row`, `data-selected`), `edit-bars-label`, `edit-bar` (`data-block`, `data-layer`), chips `embed-callout`.

- [ ] **Step 1: Write the failing tests**

Create `src/viz/EditSizeCard.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { EditSizeCard } from './EditSizeCard'
import { texOf } from '../test/tex'

afterEach(cleanup)

const rows = {
  tokens: [{ id: 1, text: 'The' }, { id: 2, text: ' cat' }],
  attn: [[0.2, 0.4, 1.5], [0.1, 0.3, 0.6]],
  mlp: [[0.9, 0.2, 0.3], [0.5, 0.05, 1.2]],
}

test('two heatmaps of tokens × layers, MLP first, one cell per value', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  const maps = screen.getAllByTestId('edit-map')
  expect(maps.map((m) => m.getAttribute('data-block'))).toEqual(['mlp', 'attn'])
  expect(screen.getAllByTestId('edit-cell')).toHaveLength(2 * 2 * 3)
  const cell = screen.getAllByTestId('edit-cell').find((c) =>
    c.getAttribute('data-block') === 'attn' && c.getAttribute('data-row') === '0' && c.getAttribute('data-layer') === '2')!
  expect(cell.textContent).toBe('“The”, layer 2, attention: 1.50')
  expect(texOf(screen.getByTestId('edit-size-formula'))).toContain('\\lVert')
})

test('bars follow the last token by default and the clicked row after', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('“ cat”'.replace(' ', '␣'))
  fireEvent.click(screen.getAllByTestId('edit-row-label')[0])
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('“The”')
  const bars = screen.getAllByTestId('edit-bar')
  expect(bars).toHaveLength(2 * 3)
  expect(bars.find((b) => b.getAttribute('data-block') === 'mlp' && b.getAttribute('data-layer') === '0'))
    .toHaveAttribute('data-value', '0.90')
})

test('selection out of range falls back to the last visible token', () => {
  const { rerender } = render(<EditSizeCard rows={{ ...rows, tokens: [...rows.tokens, { id: 3, text: '!' }],
    attn: [...rows.attn, [1, 1, 1]], mlp: [...rows.mlp, [1, 1, 1]] }} note="unused" />)
  fireEvent.click(screen.getAllByTestId('edit-row-label')[2])
  rerender(<EditSizeCard rows={rows} note="unused" />)
  expect(screen.getAllByTestId('edit-row-label')[1]).toHaveAttribute('data-selected', 'true')
})

test('without rows the card keeps its heading and shows the note', () => {
  render(<EditSizeCard note="Real model only: simulated runs have no MLP to measure." />)
  expect(screen.getByTestId('edit-size-card')).toHaveTextContent('How much each layer changes the vector')
  expect(screen.getByTestId('edit-size-note')).toHaveTextContent(/Real model only/)
  expect(screen.queryAllByTestId('edit-cell')).toHaveLength(0)
})

test('three chips open the edit entry’s sections', () => {
  render(<EditSizeCard rows={rows} note="unused" />)
  const chips = screen.getAllByTestId('embed-callout')
  expect(chips.map((c) => c.getAttribute('data-section'))).toEqual(['layer-0', 'quiet-middle', 'attn-vs-mlp'])
  expect(chips[0]).toHaveTextContent('why layer 0 is so dark')
})
```

Note on the bars label: `visibleToken` renders a leading space as `␣`; the expected text is `“␣cat”`. If `visibleToken` differs, assert with `visibleToken(' cat')` imported from `./tokenText` instead.

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/EditSizeCard.test.tsx`
Expected: FAIL — cannot resolve `./EditSizeCard`.

- [ ] **Step 3: Implement**

Create `src/viz/EditSizeCard.tsx`:

```tsx
import { useState } from 'react'
import { EDIT_CALLOUTS } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import { Formula } from './Formula'
import { ratioFill } from './mlpMath'
import type { EditRows } from './selectors'
import { visibleToken } from './tokenText'

const CELL = 7, ROW_H = 12, LABEL_W = 60, BAR_W = 3, BAR_H = 56
type Block = 'attn' | 'mlp'
const BLOCK_NAME: Record<Block, string> = { attn: 'attention', mlp: 'MLP' }

// "How much each layer changes the vector": per token and layer, the size of
// each block's edit relative to the stream it is added to. Pure
// presentation; the selected token is local (it only drives the bars).
export function EditSizeCard({ rows, note }: { rows?: EditRows; note: string }) {
  const [picked, setPicked] = useState<number | null>(null)
  const last = rows ? rows.tokens.length - 1 : -1
  const sel = picked !== null && picked <= last ? picked : last
  return (
    <section data-testid="edit-size-card" className="mlp-card">
      <h4>How much each layer changes the vector{' '}
        <Term id="edit" variant="icon" label="About edit sizes" testId="edit-size-info">ⓘ</Term></h4>
      {!rows ? <p data-testid="edit-size-note" className="mlp-note">{note}</p> : (
        <>
          <p className="mlp-caption">Every block adds an <Term id="edit">edit</Term> to the{' '}
            <Term id="residual-stream">residual stream</Term>. The colour shows how big that edit is compared with
            the vector it is added to: 1.0 means the block wrote something as large as everything before it.</p>
          <Formula testId="edit-size-formula" className="card-formula" inline
            tex={String.raw`\displaystyle \text{size} = \frac{\lVert \Delta x \rVert}{\lVert x \rVert}`} />
          <div className="edit-maps">
            {(['mlp', 'attn'] as const).map((block) => (
              <EditMap key={block} block={block} rows={rows} selected={sel} onSelect={setPicked} />
            ))}
          </div>
          <div className="mlp-scale">0 <span className="mlp-scale-ramp" aria-hidden="true" /> ≥ 1.0</div>
          <EditBars rows={rows} index={sel} />
          <div className="embed-callouts">
            {EDIT_CALLOUTS.map((c) => (
              <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function EditMap({ block, rows, selected, onSelect }: {
  block: Block; rows: EditRows; selected: number; onSelect: (row: number) => void
}) {
  const layers = rows[block][0]?.length ?? 0
  const w = LABEL_W + layers * CELL, h = rows.tokens.length * ROW_H + 14
  return (
    <figure className="edit-map-fig">
      <figcaption>{BLOCK_NAME[block]}</figcaption>
      <svg data-testid="edit-map" data-block={block} width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="group"
        aria-label={`${BLOCK_NAME[block]} edit sizes, tokens by layers`}>
        {rows.tokens.map((t, r) => (
          <g key={r}>
            <text data-testid="edit-row-label" data-row={r} data-selected={String(r === selected)}
              className={`edit-row-label${r === selected ? ' edit-row-selected' : ''}`}
              x={LABEL_W - 4} y={r * ROW_H + 9} textAnchor="end" role="button" tabIndex={0}
              onClick={() => onSelect(r)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(r) }}>
              {visibleToken(t.text)}
            </text>
            {rows[block][r].map((v, l) => (
              <rect key={l} data-testid="edit-cell" data-block={block} data-row={r} data-layer={l}
                x={LABEL_W + l * CELL} y={r * ROW_H} width={CELL - 1} height={ROW_H - 1} fill={ratioFill(v)}
                onClick={() => onSelect(r)}>
                <title>{`“${visibleToken(t.text)}”, layer ${l}, ${BLOCK_NAME[block]}: ${v.toFixed(2)}`}</title>
              </rect>
            ))}
          </g>
        ))}
        {[0, 10, 20].filter((l) => l < layers).map((l) => (
          <text key={l} className="edit-axis" x={LABEL_W + l * CELL} y={h - 2}>L{l}</text>
        ))}
      </svg>
    </figure>
  )
}

function EditBars({ rows, index }: { rows: EditRows; index: number }) {
  const a = rows.attn[index], m = rows.mlp[index]
  const peak = Math.max(...a, ...m) || 1
  const w = a.length * (2 * BAR_W + 2)
  return (
    <figure className="edit-bars-fig">
      <figcaption data-testid="edit-bars-label">edit size per layer for “{visibleToken(rows.tokens[index].text)}”,
        attention vs MLP</figcaption>
      <svg width={w} height={BAR_H} viewBox={`0 0 ${w} ${BAR_H}`} role="img"
        aria-label="edit size per layer, attention and MLP bars">
        {a.map((_, l) => (['attn', 'mlp'] as const).map((block, j) => {
          const v = block === 'attn' ? a[l] : m[l]
          const bh = (v / peak) * (BAR_H - 2)
          return (
            <rect key={`${l}-${block}`} data-testid="edit-bar" data-block={block} data-layer={l} data-value={v.toFixed(2)}
              className={`edit-bar edit-bar-${block}`} x={l * (2 * BAR_W + 2) + j * BAR_W} y={BAR_H - bh}
              width={BAR_W - 0.5} height={bh}>
              <title>{`layer ${l}, ${BLOCK_NAME[block]}: ${v.toFixed(2)}`}</title>
            </rect>
          )
        }))}
      </svg>
    </figure>
  )
}
```

In the test, the cell's accessible text comes from `<title>`; `cell.textContent` equals the title text.

Append to `src/index.css` (beside the other card styles):

```css
/* MLP cards (Layers detail) */
.mlp-card { margin-top: 1.25rem; padding-top: .75rem; border-top: 1px solid var(--ai-wash-strong); }
.mlp-card h4 { margin: 0 0 .4rem; color: var(--ai-deep); font-weight: 600; }
.mlp-note { color: var(--ink-soft, #666); font-style: italic; }
.edit-maps { display: flex; flex-wrap: wrap; gap: 1rem; }
.edit-map-fig, .edit-bars-fig { margin: .25rem 0; max-width: 100%; overflow-x: auto; }
.edit-map-fig figcaption, .edit-bars-fig figcaption { font-size: .75rem; color: var(--ai); }
.edit-row-label { font-size: 9px; font-family: var(--mono); cursor: pointer; fill: currentColor; }
.edit-row-selected { font-weight: 700; fill: var(--ai-deep); }
.edit-axis { font-size: 8px; fill: var(--ai-soft); }
.edit-bar-attn { fill: var(--ai-soft); }
.edit-bar-mlp { fill: var(--ai-deep); }
.mlp-scale { font-size: .7rem; display: flex; align-items: center; gap: .3rem; }
.mlp-scale-ramp { display: inline-block; width: 80px; height: 8px;
  background: linear-gradient(90deg, hsl(211 30% 94%), hsl(211 55% 24%)); }
```

(Check `index.css` for the muted text token's real name — `--ink-soft` is a guess; use whatever the file uses for muted text, e.g. on `.attn-note`.)

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/viz/EditSizeCard.test.tsx && npx tsc -b && npx oxlint src`
Expected: PASS; clean.

- [ ] **Step 5: Commit**

```bash
git add src/viz/EditSizeCard.tsx src/viz/EditSizeCard.test.tsx src/index.css
git commit -m "feat(viz): edit-size card — tokens × layers heatmaps and per-token bars"
```

---

### Task 9: LayerVotesCard

**Files:**
- Create: `src/viz/LayerVotesCard.tsx`, `src/viz/LayerVotesCard.test.tsx`
- Modify: `src/index.css`

**Interfaces:**
- Consumes: `VotedCandidate` (Task 5), `runningTotal`, `blockTotals`, `topMlpLayers`, `votePeak` (Task 5), `useCandidateSelection`, `selectedCandidate` (Task 6), `cellFill` from `./stripMath`.
- Produces: `export function LayerVotesCard({ candidates, cycle, note }: { candidates?: VotedCandidate[]; cycle: number; note: string })`. Test ids: `votes-card`, `votes-info`, `votes-note`, `votes-formula`, `vote-row-label` (`data-row`, `data-selected`), `vote-cell` (`data-row`, `data-layer`), `vote-line` (`data-row`, `data-selected`, `data-final`), `vote-final`, `votes-row`, `votes-total`.

- [ ] **Step 1: Write the failing tests**

Create `src/viz/LayerVotesCard.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { buildFixtureTrace } from '../test/fixtures'
import { texOf } from '../test/tex'
import { LogitsDetail } from './details/LogitsDetail'
import { LayerVotesCard } from './LayerVotesCard'
import { cycleVotes } from './selectors'

afterEach(cleanup)

const trace = buildFixtureTrace({ mlp: true, readout: true })
const voted = cycleVotes(trace, 4)
if (voted.status !== 'ok') throw new Error('fixture has votes')
const cands = voted.candidates

test('grid: one row per candidate, one cell per MLP layer, blue for adds and red for takes', () => {
  render(<LayerVotesCard candidates={cands} cycle={0} note="unused" />)
  expect(screen.getAllByTestId('vote-row-label')).toHaveLength(3)
  const cells = screen.getAllByTestId('vote-cell')
  expect(cells).toHaveLength(3 * 3)
  const v = cands[0].votes.mlp[1]
  const cell = cells.find((c) => c.getAttribute('data-row') === '0' && c.getAttribute('data-layer') === '1')!
  expect(cell.textContent).toMatch(new RegExp(`MLP 1 → “.*”: ${v >= 0 ? '\\+' : '−'}${Math.abs(v).toFixed(2)} to the score`))
  expect(cell.getAttribute('fill')).toMatch(v >= 0 ? /^hsl\(211/ : /^hsl\(13/)
  expect(texOf(screen.getByTestId('votes-formula'))).toContain('\\sum')
})

test('running total ends exactly on the logit; the breakdown total is the Logits card’s score', () => {
  render(<LayerVotesCard candidates={cands} cycle={0} note="unused" />)
  const line = screen.getAllByTestId('vote-line').find((l) => l.getAttribute('data-selected') === 'true')!
  expect(line.getAttribute('data-final')).toBe(cands[0].logit.toFixed(2))
  expect(screen.getByTestId('vote-final')).toHaveTextContent(cands[0].logit.toFixed(2))
  expect(screen.getByTestId('votes-total')).toHaveTextContent(cands[0].logit.toFixed(2))
  const labels = screen.getAllByTestId('votes-row').map((r) => r.textContent)
  expect(labels[0]).toMatch(/embedding/)
  expect(labels[1]).toMatch(/all 3 attention blocks/)
  expect(labels[2]).toMatch(/all 3 MLP blocks/)
  expect(labels.slice(3)).toHaveLength(3)          // the three largest MLP layers
  expect(labels.join()).not.toMatch(/remainder/)
})

test('clicking a candidate selects it here and on the Logits card', () => {
  render(<LayerVotesCard candidates={cands} cycle={0} note="unused" />)
  fireEvent.click(screen.getAllByTestId('vote-row-label')[2])
  expect(screen.getAllByTestId('vote-row-label')[2]).toHaveAttribute('data-selected', 'true')
  cleanup()
  render(<LogitsDetail events={trace} cursor={8} />)
  expect(screen.getAllByTestId('readout-candidate')[2]).toHaveAttribute('data-selected', 'true')
})

test('without candidates the card keeps its heading and shows the note', () => {
  render(<LayerVotesCard cycle={0} note="Votes appear once this cycle’s scores are computed." />)
  expect(screen.getByTestId('votes-card')).toHaveTextContent('What each layer voted for')
  expect(screen.getByTestId('votes-note')).toHaveTextContent(/Votes appear/)
  expect(screen.queryAllByTestId('vote-cell')).toHaveLength(0)
})
```

(`cursor={8}` is the mlp fixture's cycle-0 logits event; see the index comment in Task 5.)

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/LayerVotesCard.test.tsx`
Expected: FAIL — cannot resolve `./LayerVotesCard`.

- [ ] **Step 3: Implement**

Create `src/viz/LayerVotesCard.tsx`:

```tsx
import { Term } from '../glossary/Term'
import { selectedCandidate, useCandidateSelection } from './candidateSelection'
import { Formula } from './Formula'
import { blockTotals, runningTotal, topMlpLayers, votePeak } from './mlpMath'
import type { VotedCandidate } from './selectors'
import { cellFill } from './stripMath'
import { visibleToken } from './tokenText'

const CELL = 8, ROW_H = 14, LABEL_W = 64, LINE_W = 260, LINE_H = 90, BAR_MAX = 90

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`

// "What each layer voted for": the score split exactly into one share per
// block (final norm frozen at its real scale). The candidate selection is
// shared with the Logits card.
export function LayerVotesCard({ candidates, cycle, note }: { candidates?: VotedCandidate[]; cycle: number; note: string }) {
  const pick = useCandidateSelection((s) => s.pick)
  const select = useCandidateSelection((s) => s.select)
  return (
    <section data-testid="votes-card" className="mlp-card">
      <h4>What each layer voted for{' '}
        <Term id="votes" variant="icon" label="About per-layer votes" testId="votes-info">ⓘ</Term></h4>
      {!candidates || candidates.length === 0 ? <p data-testid="votes-note" className="mlp-note">{note}</p> : (
        <VotesBody candidates={candidates} selected={selectedCandidate(pick, cycle, candidates.length)}
          onSelect={(i) => select(cycle, i)} />
      )}
    </section>
  )
}

function VotesBody({ candidates, selected, onSelect }: {
  candidates: VotedCandidate[]; selected: number; onSelect: (i: number) => void
}) {
  const layers = candidates[0].votes.mlp.length
  const peak = votePeak(candidates)
  const sel = candidates[selected]
  const totals = blockTotals(sel.votes)
  const rows: Array<{ label: string; value: number; sub?: boolean }> = [
    { label: 'embedding of the last token', value: totals.embed },
    { label: `all ${layers} attention blocks`, value: totals.attn },
    { label: `all ${layers} MLP blocks`, value: totals.mlp },
    ...topMlpLayers(sel.votes).map(({ layer, value }) => ({ label: `of which MLP ${layer}`, value, sub: true })),
  ]
  const barMax = Math.max(...rows.map((r) => Math.abs(r.value))) || 1
  const w = LABEL_W + layers * CELL, h = candidates.length * ROW_H
  return (
    <>
      <p className="mlp-caption">The <Term id="logit">score</Term> on the Logits card splits exactly into one
        share per block. Each cell is how much one layer’s MLP added to (blue) or took from (red) a candidate’s
        score.</p>
      <Formula testId="votes-formula" className="card-formula" inline tex={String.raw`\displaystyle \begin{aligned} &\text{score}(c) = e \cdot E_c + \sum_{\ell} a_\ell \cdot E_c + \sum_{\ell} m_\ell \cdot E_c \\ &\text{each part scaled by the final norm} \end{aligned}`} />
      <figure className="edit-map-fig">
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="group" aria-label="MLP votes, candidates by layers">
          {candidates.map((c, r) => (
            <g key={c.id}>
              <text data-testid="vote-row-label" data-row={r} data-selected={String(r === selected)}
                className={`edit-row-label${r === selected ? ' edit-row-selected' : ''}`}
                x={LABEL_W - 4} y={r * ROW_H + 10} textAnchor="end" role="button" tabIndex={0}
                onClick={() => onSelect(r)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(r) }}>
                {visibleToken(c.text)}
              </text>
              {c.votes.mlp.map((v, l) => (
                <rect key={l} data-testid="vote-cell" data-row={r} data-layer={l} x={LABEL_W + l * CELL} y={r * ROW_H}
                  width={CELL - 1} height={ROW_H - 1} fill={cellFill(v, peak)} onClick={() => onSelect(r)}>
                  <title>{`MLP ${l} → “${visibleToken(c.text)}”: ${signed(v)} to the score`}</title>
                </rect>
              ))}
            </g>
          ))}
        </svg>
      </figure>
      <RunningTotal candidates={candidates} selected={selected} />
      <table className="votes-breakdown">
        <caption>where the score of “{visibleToken(sel.text)}” comes from</caption>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} data-testid="votes-row" className={r.sub ? 'votes-sub' : undefined}>
              <th scope="row">{r.label}</th>
              <td><span className={`votes-bar ${r.value >= 0 ? 'votes-bar-pos' : 'votes-bar-neg'}`}
                style={{ width: `${(Math.abs(r.value) / barMax) * BAR_MAX}px` }} /></td>
              <td>{signed(r.value)}</td>
            </tr>
          ))}
          <tr data-testid="votes-total" className="votes-total">
            <th scope="row">score on the Logits card</th><td /><td>{sel.logit.toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
    </>
  )
}

function RunningTotal({ candidates, selected }: { candidates: VotedCandidate[]; selected: number }) {
  const series = candidates.map((c) => runningTotal(c.votes))
  const all = series.flat()
  const lo = Math.min(0, ...all), hi = Math.max(...all)
  const steps = series[0].length - 1
  const x = (j: number) => (j / steps) * (LINE_W - 4) + 2
  const y = (v: number) => LINE_H - 4 - ((v - lo) / (hi - lo || 1)) * (LINE_H - 8)
  const sel = candidates[selected]
  const final = series[selected].at(-1)!
  return (
    <figure className="votes-lens">
      <figcaption>running score, “{visibleToken(sel.text)}” against the rest — ends at{' '}
        <span data-testid="vote-final">{final.toFixed(2)}</span></figcaption>
      <svg width={LINE_W} height={LINE_H} viewBox={`0 0 ${LINE_W} ${LINE_H}`} role="img"
        aria-label="each candidate's score building up across the layers">
        {series.map((s, r) => (
          <polyline key={candidates[r].id} data-testid="vote-line" data-row={r} data-selected={String(r === selected)}
            data-final={s.at(-1)!.toFixed(2)} className={r === selected ? 'vote-line-selected' : 'vote-line'}
            points={s.map((v, j) => `${x(j).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} />
        ))}
      </svg>
    </figure>
  )
}
```

The fixture's votes add up to the logit exactly, so `data-final` equals `logit.toFixed(2)`. In real runs the worker's votes agree with the logit within rounding (Task 4's tolerance), so the line may end a hundredth away from the total row; that is accepted.

Append to `src/index.css`:

```css
.votes-lens { margin: .5rem 0; }
.votes-lens figcaption { font-size: .75rem; color: var(--ai); }
.vote-line { fill: none; stroke: var(--ai-soft); stroke-width: 1; opacity: .5; }
.vote-line-selected { fill: none; stroke: var(--ai-deep); stroke-width: 2; }
.votes-breakdown { border-collapse: collapse; font-size: .8rem; }
.votes-breakdown caption { text-align: left; font-size: .75rem; color: var(--ai); }
.votes-breakdown th { text-align: left; font-weight: 400; padding-right: .5rem; }
.votes-breakdown td { padding: 1px .3rem; font-family: var(--mono); }
.votes-sub th { padding-left: 1rem; color: var(--ai); }
.votes-total th, .votes-total td { border-top: 1px solid var(--ai-wash-strong); font-weight: 600; }
.votes-bar { display: inline-block; height: 8px; }
.votes-bar-pos { background: hsl(211 45% 55%); }
.votes-bar-neg { background: hsl(13 55% 60%); }
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/viz/LayerVotesCard.test.tsx && npx tsc -b && npx oxlint src`
Expected: PASS; clean (the glossary id `votes` exists since Task 7).

- [ ] **Step 5: Commit**

```bash
git add src/viz/LayerVotesCard.tsx src/viz/LayerVotesCard.test.tsx src/index.css
git commit -m "feat(viz): votes card — MLP vote grid, running total, exact breakdown"
```

---

### Task 10: Layers detail wiring

**Files:**
- Modify: `src/viz/details/LayersDetail.tsx`, `src/viz/DetailPanel.tsx`
- Test: `src/viz/DetailPanel.test.tsx`

**Interfaces:**
- Consumes: `editRows`, `cycleVotes` (Task 5), `EditSizeCard` (Task 8), `LayerVotesCard` (Task 9).
- Produces: `LayersDetail` gains props `edits?: EditRows` and `votes?: CycleVotes`.

- [ ] **Step 1: Write the failing tests**

Append to `src/viz/DetailPanel.test.tsx` (import `buildFixtureTrace` if missing):

```ts
test('real run with MLP data: both cards below the attention section', () => {
  render(<DetailPanel events={buildFixtureTrace({ mlp: true, mode: 'real' })} cursor={4} mode="real" />)
  const layers = screen.getByTestId('detail-layers')
  expect(within(layers).getAllByTestId('edit-cell').length).toBeGreaterThan(0)
  expect(within(layers).getAllByTestId('vote-cell').length).toBeGreaterThan(0)
  const order = [...layers.querySelectorAll('[data-testid="edit-size-card"], [data-testid="votes-card"]')]
    .map((el) => el.getAttribute('data-testid'))
  expect(order).toEqual(['edit-size-card', 'votes-card'])
})

test('simulated run: both cards show the real-model-only note', () => {
  render(<DetailPanel events={buildFixtureTrace({ mlp: true })} cursor={4} mode="sim" />)
  expect(screen.getByTestId('edit-size-note')).toHaveTextContent('Real model only: simulated runs have no MLP to measure.')
  expect(screen.getByTestId('votes-note')).toHaveTextContent('Real model only: simulated runs have no MLP to measure.')
})

test('real run recorded before MLP data: the older-data note on both cards', () => {
  render(<DetailPanel events={buildFixtureTrace({ mode: 'real' })} cursor={4} mode="real" />)
  expect(screen.getByTestId('edit-size-note')).toHaveTextContent('This run was recorded before MLP data was exported.')
  expect(screen.getByTestId('votes-note')).toHaveTextContent('This run was recorded before MLP data was exported.')
})

test('live run before the cycle’s logits: the votes card says votes are coming', () => {
  const t = buildFixtureTrace({ mlp: true, mode: 'real' }).slice(0, 8)   // through cycle 0's edits
  render(<DetailPanel events={t} cursor={4} mode="real" />)
  expect(screen.getByTestId('votes-note')).toHaveTextContent('Votes appear once this cycle’s scores are computed.')
  expect(screen.getAllByTestId('edit-cell').length).toBeGreaterThan(0)
})
```

(Add `within` to the Testing Library import.)

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/viz/DetailPanel.test.tsx`
Expected: FAIL — no `edit-cell` / notes rendered.

- [ ] **Step 3: Implement**

In `LayersDetail.tsx` add imports:

```ts
import { EditSizeCard } from '../EditSizeCard'
import { LayerVotesCard } from '../LayerVotesCard'
import type { CycleVotes, EditRows } from '../selectors'
```

Above the component:

```ts
const SIM_NOTE = 'Real model only: simulated runs have no MLP to measure.'
const OLD_NOTE = 'This run was recorded before MLP data was exported.'
const PENDING_NOTE = 'Votes appear once this cycle’s scores are computed.'
```

Add `edits, votes` to the destructured props and their types (`edits?: EditRows`, `votes?: CycleVotes`). At the end of the returned `div`, after the head-grid explorer line:

```tsx
      <EditSizeCard rows={mode === 'real' ? edits : undefined} note={mode === 'sim' ? SIM_NOTE : OLD_NOTE} />
      <LayerVotesCard cycle={event.cycle}
        candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined}
        note={mode === 'sim' ? SIM_NOTE : votes?.status === 'pending' ? PENDING_NOTE : OLD_NOTE} />
```

In `DetailPanel.tsx`, import `cycleVotes, editRows` from `./selectors` and pass to `LayersDetail`:

```tsx
        edits={editRows(events, cursor)} votes={cycleVotes(events, cursor)}
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/viz && npx tsc -b && npx oxlint src`
Expected: PASS; clean.

- [ ] **Step 5: Commit**

```bash
git add src/viz/details/LayersDetail.tsx src/viz/DetailPanel.tsx src/viz/DetailPanel.test.tsx
git commit -m "feat(layers): edit-size and votes cards in the Layers detail, with notes"
```

---

### Task 11: E2E, screenshot scene, README, backlog

**Files:**
- Create: `e2e/mlp.spec.ts`
- Modify: `scripts/capture-screenshots.mjs`, `README.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Write the e2e test**

Create `e2e/mlp.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

test('simulated run: the Layers detail shows both MLP cards with the real-model-only note', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()
  await page.getByTestId('run-chip').first().waitFor({ timeout: 30000 })   // the run has sealed
  await page.getByTestId('stage-card').nth(2).click()                      // Layers
  await expect(page.getByTestId('edit-size-note')).toHaveText('Real model only: simulated runs have no MLP to measure.')
  await expect(page.getByTestId('votes-note')).toHaveText('Real model only: simulated runs have no MLP to measure.')
  await page.getByTestId('votes-info').click()
  await expect(page.getByTestId('gl-heading')).toContainText('Per-layer votes')
})
```

- [ ] **Step 2: Run it**

Run: `npx playwright test e2e/mlp.spec.ts`
Expected: PASS (the feature is already in place; this test guards the sim path in CI). If it fails, debug the cause — don't adjust the expectations to the output.

- [ ] **Step 3: Screenshot scene**

In `scripts/capture-screenshots.mjs`, after the `logits` scene block and before the `finally`, add:

```js
  // ---- mlp.png: real mode, Layers stage — edit sizes and per-layer votes
  if (want('mlp')) {
    await $('prompt-input').fill('The farmer bought a horse and rode it home.')
    await $('maxtok-input').fill('1')
    const before = await $('run-chip').count()
    await $('btn-generate').click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })
    await $('btn-live').click()
    await $('stage-card').nth(2).click()                            // Layers
    await $('vote-cell').first().waitFor({ timeout: 10000 })
    await page.setViewportSize({ width: 1380, height: 1600 })
    const cards = [await $('edit-size-card').boundingBox(), await $('votes-card').boundingBox()]
    const pad = 12
    const top = cards[0].y - pad, bottom = cards[1].y + cards[1].height + pad
    await page.screenshot({ path: `${DIR}/mlp.png`, clip: {
      x: cards[0].x - pad, y: top, width: Math.max(cards[0].width, cards[1].width) + 2 * pad, height: bottom - top,
    } })
    await page.setViewportSize({ width: 1380, height: 940 })
    await $('maxtok-input').fill('20')
    console.log('mlp.png')
  }
```

This scene needs the v4 model, so it runs after Task 12's release.

- [ ] **Step 4: README and backlog**

In `README.md`, after the "Reading the attention heatmaps" section (before "## The glossary"), add:

```markdown
## Opening the MLP — edit sizes and per-layer votes

In a real run, the Layers detail has two more cards below the attention
heatmaps. **How much each layer changes the vector** shows, for every token
and layer, how big each block's edit is compared with the stream it is
added to (‖Δx‖ / ‖x‖), as two heatmaps (MLP and attention) and per-layer
bars for a selected token. **What each layer voted for** splits the Logits
card's score into one share per block: holding the final norm at its real
scale for the last position makes it linear, so embedding + attention
votes + MLP votes add up exactly to the score. The grid shows each layer's
MLP vote per candidate (blue adds, red subtracts), a running total traces
the score building up through the layers, and a breakdown names the
largest contributors. Simulated runs show a note instead: there is no MLP
to measure.

![The edit-size and per-layer votes cards](docs/screenshots/mlp.png)
```

In `docs/BACKLOG.md`: tick #15 (`- [x]`), replace "*next up.*" with a shipped note ("Shipped 2026-10-10 (spec `docs/superpowers/specs/2026-10-10-opening-the-mlp-design.md`); export v4."), add `- [x] Opening the MLP: edit sizes, per-layer votes; export v4 *(#15, 2026-10-10)*` at the top of "Done (for the record)", and add three new items under "Attention / interpretability depth":

```markdown
- [ ] **#16 Labelled MLP neurons** — top firing neurons per layer, each
  labelled by the tokens its down-projection column promotes most (an
  offline asset from the export tool). *(#15 follow-up)*
- [ ] **#17 SwiGLU anatomy view** — a detail view of one MLP's steps
  (norm, gate, up, SiLU, product, down) with real values for the selected
  token. *(#15 follow-up)*
- [ ] **#18 MLP knockout runs** — re-run a prompt with one layer's MLP
  zeroed and open the pair in the Compare view. *(#15 follow-up)*
```

- [ ] **Step 5: Commit**

```bash
git add e2e/mlp.spec.ts scripts/capture-screenshots.mjs README.md docs/BACKLOG.md
git commit -m "docs(mlp): e2e guard, screenshot scene, README section, backlog"
```

---

### Task 12: Release v4 (operator steps — needs the user's go-ahead)

Publishing to the Hugging Face Hub is outward-facing and tags never move. **Stop and ask the user before Step 4.** Steps 1–3 only write under `tools/export/out/`.

- [ ] **Step 1: Export both variants**

```bash
cd tools/export
uv run python -m tsumugi_export export --out out/model
uv run python -m tsumugi_export export --out out/model-nocache --no-cache
```
Expected: both print `exported (…) → out/…`. (These overwrite the local v3 export.)

- [ ] **Step 2: Quantize and rebuild geometry**

```bash
uv run python -m tsumugi_export quantize --model-dir out/model
uv run python -m tsumugi_export geometry --model-dir out/model
```

- [ ] **Step 3: Validate**

```bash
uv run python -m tsumugi_export validate --model-dir out/model
```
Expected: `VERDICT: PASS`, with `a-equiv-b` run (not SKIP) and every `*:edit-ratios-parity`, `*:stream-parts-sum`, `*:votes-sum`, `*:edit-cache-parity` row PASS. A quantized tolerance failure with plausible values → widen per Task 2's note, with a ledgered ruling, and re-run.

- [ ] **Step 4: Ask the user, then publish**

Ask: "Validation passed; publish to `saigyo-hoshi/smollm2-135m-attn-onnx` as tag `v4`?" On yes:

```bash
uv run python -m tsumugi_export publish --model-dir out/model --tag v4
```

- [ ] **Step 5: Point the app at v4**

In `src/engine/tokenizer.ts`: `export const ATTN_MODEL_REVISION = 'v4'` and extend the comment with "v4 = MLP internals (edit_ratios, stream_parts)." Run `npx vitest run && npx tsc -b && npm run build` → all green.

- [ ] **Step 6: Manual check in the browser, screenshots, commit**

`npm run dev`, real mode, any prompt, Layers stage: both cards filled; select a candidate on the votes card — the breakdown's "score on the Logits card" equals the Logits card's score for that candidate, and the Logits card shows the same candidate selected. Then kill any stale server on :5199 and run `npm run screenshots -- mlp`.

```bash
git add src/engine/tokenizer.ts docs/screenshots/mlp.png
git commit -m "feat(mlp): point the app at export v4"
```
