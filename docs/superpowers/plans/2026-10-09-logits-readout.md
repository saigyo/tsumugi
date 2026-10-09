# Logits Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Logits card show the readout as a visible dot product: the last token's final vector against each candidate's row of E, with real values from a republished ONNX export.

**Architecture:** The export tool gains three graph outputs (`last_hidden_state`, `topk_ids`, `topk_embeds`), validation checks for them, and a tagged publish. The app pins that tag, the worker pools the new outputs into the `logits` trace event through a pure helper, and a new `LogitsReadout` component renders the equation row and a dot-product bench, falling back to geometry-asset rows for sim mode and legacy traces.

**Tech Stack:** Python 3.10+ / onnx / onnxruntime / huggingface_hub (in `tools/export`, run with `uv`); React 19 + TypeScript + Vite; vitest + @testing-library/react; Playwright for screenshots.

**Spec:** `docs/superpowers/specs/2026-10-09-logits-readout-design.md`

## Global Constraints

- Graph output names and shapes are the artifact contract: `last_hidden_state` `[batch, seq, 576]` float32, `topk_ids` `[batch, 16]` int64, `topk_embeds` `[batch, 16, 576]` float32. Added nodes carry the `tsumugi_` name prefix.
- `TOPK = 16` lives in `tsumugi_export/__init__.py`; the gather node `tsumugi_topk_embeds_gather` is excluded from quantization alongside `/lm_head/MatMul`.
- The trace event fields are optional; legacy traces must keep validating.
- `READOUT_CELLS = 64` in `src/trace/types.ts` is the single pooling constant for worker and card.
- `lastHidden` rounded to 3 decimals; `row` pooled by mean, 3 decimals; `dot` pooled by sum, 3 decimals.
- Worker degrades without failing: absent outputs → fields omitted; wrong shape → one `console.warn`, fields omitted for the rest of the run.
- The card never mixes exact and asset-reduced vectors in one strip. Asset rows carry the Embeddings caption "PCA-reduced, offline".
- Candidates are matched to graph rows **by id**, never by position.
- Existing test ids `logit-bar`, `logits-formula`, `detail-logits`, `embed-strip-cell`, `embed-stack-cell` keep working.
- Linear git history; every commit message ends with the two trailers:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL`.
- Inside the worktree use `/usr/bin/git`, no compound shell commands; Python tests run as `uv run --project tools/export pytest tools/export/tests -q`; TypeScript tests as `npx vitest run <path>`.

## Review Focus

1. A top-10 candidate id missing from `topk_ids` (an exact tie at rank 16): the candidate must still render, with a blank product strip and the asset-row bench — Task 6 test "id missing from topk_ids is absent from the map", Task 8 test "candidate without dot renders without a product strip".
2. `topk_ids` arriving as `BigInt64Array` (transformers.js int64 tensors): ids must be compared as numbers — Task 6 test "accepts BigInt64Array ids".
3. A trace whose `logits` events carry `dot` of the wrong length (hand-edited import): the validator must flag it, not the card crash — Task 5 test "readout fields with wrong length are flagged".
4. Scrubbing from cycle 1 back to cycle 0 after selecting candidate 2 in cycle 1, where cycle 0 has fewer candidates: the selection must snap to 0, not index past the array — Task 9 test "selection resets when the cursor moves to another cycle".
5. An id beyond the asset's vocabulary (fallback hash tokenizer) in the sim/legacy bench: the asset lookup must be guarded — Task 9 test "asset row lookup is guarded for ids outside the vocabulary".

---

### Task 1: Export — expose `last_hidden_state`, `topk_ids`, `topk_embeds`

**Files:**
- Modify: `tools/export/src/tsumugi_export/__init__.py`
- Modify: `tools/export/src/tsumugi_export/postprocess.py`
- Modify: `tools/export/src/tsumugi_export/export.py:10,48`
- Test: `tools/export/tests/test_postprocess.py`

**Interfaces:**
- Produces: `TOPK = 16` in `tsumugi_export/__init__.py`; `postprocess.expose_readout(model, k=TOPK) -> onnx.ModelProto`; `postprocess.postprocess_graph(model) -> onnx.ModelProto` (readout then inputs_embeds); `postprocess.postprocess_file(path)`; constants `LAST_HIDDEN_OUTPUT = "last_hidden_state"`, `TOPK_IDS_OUTPUT = "topk_ids"`, `TOPK_EMBEDS_OUTPUT = "topk_embeds"`, `TOPK_GATHER_NODE = "tsumugi_topk_embeds_gather"`. Task 2 imports `TOPK_GATHER_NODE`; Task 3 documents the names.

- [ ] **Step 1: Write the failing tests**

Append to `tools/export/tests/test_postprocess.py`:

```python
from tsumugi_export.postprocess import (
    LAST_HIDDEN_OUTPUT, TOPK_EMBEDS_OUTPUT, TOPK_GATHER_NODE, TOPK_IDS_OUTPUT,
    expose_readout, postprocess_graph,
)

TABLE = np.arange(28, dtype=np.float32).reshape(7, 4) / 10.0   # E: 7 tokens × 4 dims


def readout_graph() -> onnx.ModelProto:
    """A miniature causal LM: Gather(E) → hidden_states.0, Neg → hidden_states.1,
    MatMul(hidden_states.1, Eᵀ) → logits. Mirrors the real export's shape
    conventions: hidden_states.N is the tensor the lm_head multiplies."""
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"])
    table = helper.make_tensor("model.embed_tokens.weight", TensorProto.FLOAT, [7, 4], TABLE.flatten().tolist())
    table_t = helper.make_tensor("lm_head.weight_t", TensorProto.FLOAT, [4, 7], TABLE.T.flatten().tolist())
    nodes = [
        helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["hidden_states.0"], axis=0),
        helper.make_node("Neg", ["hidden_states.0"], ["hidden_states.1"]),
        helper.make_node("MatMul", ["hidden_states.1", "lm_head.weight_t"], ["logits"]),
    ]
    outs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "seq", 7]),
        helper.make_tensor_value_info("hidden_states.0", TensorProto.FLOAT, [1, "seq", 4]),
        helper.make_tensor_value_info("hidden_states.1", TensorProto.FLOAT, [1, "seq", 4]),
    ]
    graph = helper.make_graph(nodes, "readout", [ids], outs, initializer=[table, table_t])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _run(model: onnx.ModelProto, ids: np.ndarray) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": ids})))


def test_readout_adds_three_outputs_and_keeps_graph_valid():
    model = expose_readout(readout_graph(), k=3)
    onnx.checker.check_model(model)
    names = [o.name for o in model.graph.output]
    assert names[-3:] == [LAST_HIDDEN_OUTPUT, TOPK_IDS_OUTPUT, TOPK_EMBEDS_OUTPUT]
    assert any(n.name == TOPK_GATHER_NODE for n in model.graph.node)


def test_readout_values_match_numpy():
    model = expose_readout(readout_graph(), k=3)
    ids = np.array([[2, 5, 1]], dtype=np.int64)
    out = _run(model, ids)
    last = -TABLE[ids[0]]                       # Neg of the gathered rows
    logits = last @ TABLE.T
    np.testing.assert_allclose(out[LAST_HIDDEN_OUTPUT][0], last, atol=1e-6)
    expected_ids = np.argsort(-logits[-1], kind="stable")[:3]
    np.testing.assert_array_equal(out[TOPK_IDS_OUTPUT][0], expected_ids)
    np.testing.assert_allclose(out[TOPK_EMBEDS_OUTPUT][0], TABLE[expected_ids], atol=1e-6)
    assert out[TOPK_IDS_OUTPUT].shape == (1, 3)
    assert out[TOPK_EMBEDS_OUTPUT].shape == (1, 3, 4)


def test_postprocess_graph_combines_readout_and_inputs_embeds():
    model = postprocess_graph(readout_graph(), k=3)
    names = [o.name for o in model.graph.output]
    assert names == ["logits", LAST_HIDDEN_OUTPUT, TOPK_IDS_OUTPUT, TOPK_EMBEDS_OUTPUT, "inputs_embeds"]
    out = _run(model, np.array([[0, 6]], dtype=np.int64))
    np.testing.assert_allclose(out["inputs_embeds"][0], TABLE[[0, 6]], atol=1e-6)


def test_readout_without_final_hidden_state_raises():
    model = readout_graph()
    del model.graph.output[2]   # hidden_states.1 — leaves only the embedding lookup
    try:
        expose_readout(model, k=3)
    except ValueError as err:
        assert "hidden_states.1" in str(err) or "final" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_readout_requires_a_gather_behind_hidden_states_0():
    model = readout_graph()
    model.graph.node[0].op_type = "Identity"
    del model.graph.node[0].input[1]
    try:
        expose_readout(model, k=3)
    except ValueError as err:
        assert "Gather" in str(err)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --project tools/export pytest tools/export/tests/test_postprocess.py -q`
Expected: FAIL with `ImportError: cannot import name 'LAST_HIDDEN_OUTPUT'`.

- [ ] **Step 3: Add the constant and implement the post-processing**

`tools/export/src/tsumugi_export/__init__.py`:

```python
STOCK_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
ATTN_REPO_ID = "saigyo-hoshi/smollm2-135m-attn-onnx"
# Candidate rows of E gathered inside the graph at the last position. The app
# caps top-k at 10; 16 leaves headroom for ties.
TOPK = 16
```

Replace `tools/export/src/tsumugi_export/postprocess.py` with:

```python
"""Graph post-processing after the optimum export.

The export declares hidden_states.{0..N} as outputs only because torch.onnx
names forward outputs positionally (see onnx_config.py). Here we keep two of
them under artifact-contract names — hidden_states.0 as `inputs_embeds` (the
embedding lookup) and hidden_states.N as `last_hidden_state` (the residual
stream after the final RMSNorm, the tensor lm_head multiplies) — and drop the
rest. We also add a readout branch: TopK over the last position's logits and
a Gather of those candidates' rows of E, as `topk_ids` / `topk_embeds`. The
tensors already in the graph stay; only the output list and a handful of tiny
nodes change, so this costs nothing measurable at inference time."""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from tsumugi_export import TOPK

EMBEDS_OUTPUT = "inputs_embeds"
SOURCE_OUTPUT = "hidden_states.0"
LAST_HIDDEN_OUTPUT = "last_hidden_state"
TOPK_IDS_OUTPUT = "topk_ids"
TOPK_EMBEDS_OUTPUT = "topk_embeds"
TOPK_GATHER_NODE = "tsumugi_topk_embeds_gather"
INT64_MAX = 2**63 - 1


def _producer(graph: onnx.GraphProto, name: str) -> onnx.NodeProto | None:
    return next((n for n in graph.node if name in n.output), None)


def find_embedding_table(graph: onnx.GraphProto) -> str:
    """The tensor name of E: the data input of the Gather that produces
    hidden_states.0, looking through Identity/Cast wrappers. Structural, not
    by name — optimum's initializer names are not part of our contract."""
    name = SOURCE_OUTPUT
    for _ in range(8):
        node = _producer(graph, name)
        if node is None:
            raise ValueError(f"no node produces {name}")
        if node.op_type == "Gather":
            return node.input[0]
        if node.op_type in ("Identity", "Cast"):
            name = node.input[0]
            continue
        raise ValueError(f"{SOURCE_OUTPUT} is produced by {node.op_type}, expected Gather")
    raise ValueError(f"gave up tracing {SOURCE_OUTPUT} back to a Gather")


def expose_readout(model: onnx.ModelProto, k: int = TOPK) -> onnx.ModelProto:
    graph = model.graph
    outputs = list(graph.output)
    hidden = [o for o in outputs if o.name.startswith("hidden_states.")]
    n = max((int(o.name.split(".")[1]) for o in hidden), default=-1)
    if n < 1:
        raise ValueError("graph has no hidden_states.1 or later — no final hidden state to expose")
    last = next(o for o in hidden if o.name == f"hidden_states.{n}")
    if not any(o.name == "logits" for o in outputs):
        raise ValueError("graph has no logits output")
    dims = last.type.tensor_type.shape.dim[2].dim_value
    if dims <= 0:
        raise ValueError(f"hidden_states.{n} has no static last dimension")
    table = find_embedding_table(graph)

    graph.initializer.extend([
        helper.make_tensor("tsumugi_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_slice_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("tsumugi_squeeze_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("tsumugi_topk_k", TensorProto.INT64, [1], [k]),
    ])
    graph.node.extend([
        helper.make_node("Identity", [f"hidden_states.{n}"], [LAST_HIDDEN_OUTPUT],
                         name="tsumugi_last_hidden_identity"),
        helper.make_node("Slice", ["logits", "tsumugi_slice_starts", "tsumugi_slice_ends", "tsumugi_slice_axes"],
                         ["tsumugi_last_logits_3d"], name="tsumugi_last_logits_slice"),
        helper.make_node("Squeeze", ["tsumugi_last_logits_3d", "tsumugi_squeeze_axes"],
                         ["tsumugi_last_logits"], name="tsumugi_last_logits_squeeze"),
        # values output is named but consumed by nothing, which ONNX allows
        helper.make_node("TopK", ["tsumugi_last_logits", "tsumugi_topk_k"],
                         ["tsumugi_topk_values", TOPK_IDS_OUTPUT],
                         name="tsumugi_topk", axis=-1, largest=1, sorted=1),
        helper.make_node("Gather", [table, TOPK_IDS_OUTPUT], [TOPK_EMBEDS_OUTPUT],
                         name=TOPK_GATHER_NODE, axis=0),
    ])
    last_vi = onnx.ValueInfoProto()
    last_vi.CopyFrom(last)
    last_vi.name = LAST_HIDDEN_OUTPUT
    graph.output.extend([
        last_vi,
        helper.make_tensor_value_info(TOPK_IDS_OUTPUT, TensorProto.INT64, ["batch_size", k]),
        helper.make_tensor_value_info(TOPK_EMBEDS_OUTPUT, TensorProto.FLOAT, ["batch_size", k, dims]),
    ])
    return model


def expose_inputs_embeds(model: onnx.ModelProto) -> onnx.ModelProto:
    outputs = list(model.graph.output)
    source = next((o for o in outputs if o.name == SOURCE_OUTPUT), None)
    if source is None:
        raise ValueError(f"graph has no {SOURCE_OUTPUT} output — was output_hidden_states set?")
    keep = [o for o in outputs if not o.name.startswith("hidden_states.")]
    embeds = onnx.ValueInfoProto()
    embeds.CopyFrom(source)
    embeds.name = EMBEDS_OUTPUT
    model.graph.node.append(
        helper.make_node("Identity", [SOURCE_OUTPUT], [EMBEDS_OUTPUT], name="inputs_embeds_identity"))
    del model.graph.output[:]
    model.graph.output.extend(keep + [embeds])
    return model


def postprocess_graph(model: onnx.ModelProto, k: int = TOPK) -> onnx.ModelProto:
    """Readout first (it reads hidden_states.N from the output list), then the
    inputs_embeds rename, which drops every hidden_states.* output."""
    return expose_inputs_embeds(expose_readout(model, k))


def postprocess_file(path: Path) -> None:
    """Rewrite `path` in place. onnx.load pulls any external data into the
    proto, so the saved file is self-contained; the 135M fp32 graph (~540 MB)
    is well under protobuf's 2 GB limit. A stale external-data sidecar from
    the optimum export is removed."""
    model = onnx.load(str(path))
    onnx.save(postprocess_graph(model), str(path))
    sidecar = path.with_name(path.name + "_data")
    if sidecar.exists():
        sidecar.unlink()


# kept for callers and tests that predate the readout branch
expose_inputs_embeds_file = postprocess_file
```

In `tools/export/src/tsumugi_export/export.py` change the import and the call:

```python
from tsumugi_export.postprocess import postprocess_file
...
    postprocess_file(onnx_dir / "model.onnx")
```

and the docstring's first line to: `"""Export SmolLM2 with attention outputs, the embedding lookup (inputs_embeds) and the readout (last_hidden_state, topk_ids, topk_embeds). Heavy: ...`.

- [ ] **Step 4: Run the whole Python suite**

Run: `uv run --project tools/export pytest tools/export/tests -q`
Expected: all PASS, including the three pre-existing `test_postprocess.py` tests (the tiny graph there has `hidden_states.1`, so `expose_inputs_embeds` alone still behaves as before).

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add tools/export/src/tsumugi_export/__init__.py tools/export/src/tsumugi_export/postprocess.py tools/export/src/tsumugi_export/export.py tools/export/tests/test_postprocess.py
/usr/bin/git commit -m "feat(export): expose last_hidden_state and top-16 candidate rows of E"
```

---

### Task 2: Export — quantization exclusion and validation checks

**Files:**
- Modify: `tools/export/src/tsumugi_export/quantize.py`
- Modify: `tools/export/src/tsumugi_export/validate.py`
- Create: `tools/export/tests/test_validate_readout.py`

**Interfaces:**
- Consumes: `TOPK`, `TOPK_GATHER_NODE`, output-name constants from Task 1.
- Produces: `validate.check_last_hidden_parity(outputs: dict, E: np.ndarray, tol: float) -> dict`, `validate.check_topk_parity(outputs: dict, E: np.ndarray, k: int, tol: float) -> dict`, `quantize.NODES_TO_EXCLUDE: list[str]`. Each check returns `{"passed": bool, "detail": str}` like the existing ones.

- [ ] **Step 1: Write the failing tests**

`tools/export/tests/test_validate_readout.py`:

```python
import numpy as np

from tsumugi_export.postprocess import TOPK_GATHER_NODE
from tsumugi_export.quantize import NODES_TO_EXCLUDE
from tsumugi_export.validate import check_last_hidden_parity, check_topk_parity

rng = np.random.default_rng(0)
E = rng.normal(size=(50, 8)).astype(np.float32)


def good_outputs(seq: int = 3, k: int = 4) -> dict:
    last = rng.normal(size=(1, seq, 8)).astype(np.float32)
    logits = last @ E.T                                   # [1, seq, 50]
    ids = np.argsort(-logits[0, -1], kind="stable")[:k]
    return {
        "logits": logits,
        "last_hidden_state": last,
        "topk_ids": ids[None, :].astype(np.int64),
        "topk_embeds": E[ids][None, :, :],
    }


def test_last_hidden_parity_passes_on_consistent_outputs():
    r = check_last_hidden_parity(good_outputs(), E, tol=1e-4)
    assert r["passed"], r["detail"]


def test_last_hidden_parity_fails_on_drift():
    out = good_outputs()
    out["last_hidden_state"] = out["last_hidden_state"] + 0.1
    assert not check_last_hidden_parity(out, E, tol=1e-4)["passed"]


def test_last_hidden_parity_fails_on_missing_or_misshapen_output():
    out = good_outputs()
    del out["last_hidden_state"]
    r = check_last_hidden_parity(out, E, tol=1e-4)
    assert not r["passed"] and "no last_hidden_state" in r["detail"]
    out = good_outputs()
    out["last_hidden_state"] = out["last_hidden_state"][:, :, :4]
    r = check_last_hidden_parity(out, E, tol=1e-4)
    assert not r["passed"] and "shape" in r["detail"]


def test_topk_parity_passes_on_consistent_outputs():
    r = check_topk_parity(good_outputs(k=4), E, k=4, tol=1e-6)
    assert r["passed"], r["detail"]


def test_topk_parity_accepts_a_different_order_among_exact_ties():
    out = good_outputs(k=4)
    out["logits"][0, -1, :] = 0.0                          # every id ties
    ids = np.array([[7, 3, 11, 0]], dtype=np.int64)
    out["topk_ids"] = ids
    out["topk_embeds"] = E[ids[0]][None, :, :]
    assert check_topk_parity(out, E, k=4, tol=1e-6)["passed"]


def test_topk_parity_fails_when_ids_are_not_the_top_k():
    out = good_outputs(k=4)
    out["topk_ids"][0, 0] = (out["topk_ids"][0, 0] + 1) % 50 if out["topk_ids"][0, 0] not in np.argsort(-out["logits"][0, -1])[:4] else np.argsort(out["logits"][0, -1])[0]
    out["topk_embeds"] = E[out["topk_ids"][0]][None, :, :]
    assert not check_topk_parity(out, E, k=4, tol=1e-6)["passed"]


def test_topk_parity_fails_when_rows_do_not_match_ids():
    out = good_outputs(k=4)
    out["topk_embeds"] = out["topk_embeds"] + 0.01
    r = check_topk_parity(out, E, k=4, tol=1e-6)
    assert not r["passed"] and "rows" in r["detail"]


def test_quantize_excludes_lm_head_and_the_readout_gather():
    assert "/lm_head/MatMul" in NODES_TO_EXCLUDE
    assert TOPK_GATHER_NODE in NODES_TO_EXCLUDE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --project tools/export pytest tools/export/tests/test_validate_readout.py -q`
Expected: FAIL with `ImportError: cannot import name 'NODES_TO_EXCLUDE'`.

- [ ] **Step 3: Implement the exclusion list and the checks**

In `tools/export/src/tsumugi_export/quantize.py`, add after the imports and use it in `run`:

```python
from tsumugi_export.postprocess import TOPK_GATHER_NODE

# The tied lm_head must stay fp32 (see the module docstring); the readout
# gather must too, so the candidate rows the app shows are exactly the rows
# the lm_head multiplied — their products then sum to the logit.
NODES_TO_EXCLUDE = ["/lm_head/MatMul", TOPK_GATHER_NODE]
```

```python
    quantize_dynamic(str(src), str(onnx_dir / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8,
                     nodes_to_exclude=NODES_TO_EXCLUDE)
    print("wrote model_quantized.onnx (q8, lm_head and readout gather excluded)")
```

In `tools/export/src/tsumugi_export/validate.py`, extend the module docstring list with
`7. last_hidden_state · Eᵀ reproduces the last-position logits` and
`8. topk_ids are the exact top-k and topk_embeds their rows of E`, add the import
`from tsumugi_export import TOPK` next to the existing `STOCK_MODEL_ID` import inside `run`
(move it to module level: `from tsumugi_export import STOCK_MODEL_ID, TOPK`), and add the
two pure checks after `_check_inputs_embeds_parity`:

```python
def check_last_hidden_parity(outputs: dict, E: np.ndarray, tol: float) -> dict:
    """last_hidden_state is what the tied lm_head multiplies, so its last row
    against Eᵀ must reproduce the last-position logits."""
    if "last_hidden_state" not in outputs:
        return {"passed": False, "detail": "graph has no last_hidden_state output"}
    last, logits = outputs["last_hidden_state"], outputs["logits"]
    expected = (1, logits.shape[1], E.shape[1])
    if last.shape != expected:
        return {"passed": False, "detail": f"last_hidden_state shape {last.shape}, expected {expected}"}
    diff = float(np.max(np.abs(last[0, -1] @ E.T - logits[0, -1])))
    return {"passed": diff <= tol, "detail": f"max|x·Eᵀ − logits|={diff:.2e} (atol {tol})"}


def check_topk_parity(outputs: dict, E: np.ndarray, k: int, tol: float) -> dict:
    """topk_ids must be the exact top-k of the last-position logits (order may
    differ only among exact ties) and topk_embeds their rows of E. The gather
    is excluded from quantization, so no quantization slack applies."""
    for name in ("topk_ids", "topk_embeds"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    ids, rows, logits = outputs["topk_ids"], outputs["topk_embeds"], outputs["logits"]
    if ids.shape != (1, k):
        return {"passed": False, "detail": f"topk_ids shape {ids.shape}, expected (1, {k})"}
    if rows.shape != (1, k, E.shape[1]):
        return {"passed": False, "detail": f"topk_embeds shape {rows.shape}, expected (1, {k}, {E.shape[1]})"}
    last = logits[0, -1]
    kth = np.sort(last)[::-1][k - 1]
    listed = [int(i) for i in ids[0]]
    if len(set(listed)) != k or any(last[i] < kth for i in listed):
        return {"passed": False, "detail": f"topk_ids {listed} are not the top-{k} of the last-position logits"}
    diff = float(np.max(np.abs(rows[0] - E[listed])))
    if diff > tol:
        return {"passed": False, "detail": f"topk_embeds rows differ from E[topk_ids] by {diff:.2e} (atol {tol})"}
    return {"passed": True, "detail": f"top-{k} ids exact; rows match E within {diff:.2e}"}
```

In `run`, inside the per-variant loop directly after the `inputs-embeds-parity` line:

```python
        ro_out = dict(zip(out_names, sess.run(None, _feeds(sess, _tokenize(model_dir, PROMPTS[0])))))
        lh_tol = 1e-3 if variant == "model.onnx" else TOL
        checks[f"{variant}:last-hidden-parity"] = check_last_hidden_parity(ro_out, E, lh_tol)
        checks[f"{variant}:topk-parity"] = check_topk_parity(ro_out, E, TOPK, 1e-3)
```

- [ ] **Step 4: Run the Python suite**

Run: `uv run --project tools/export pytest tools/export/tests -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add tools/export/src/tsumugi_export/quantize.py tools/export/src/tsumugi_export/validate.py tools/export/tests/test_validate_readout.py
/usr/bin/git commit -m "feat(export): validate the readout outputs; keep the candidate-row gather fp32"
```

---

### Task 3: Export — tagged publish, model card, runbook

**Files:**
- Modify: `tools/export/src/tsumugi_export/publish.py`
- Modify: `tools/export/src/tsumugi_export/__main__.py`
- Modify: `tools/export/README.md`
- Test: `tools/export/tests/test_publish_gate.py`

**Interfaces:**
- Produces: `publish --tag <name>` (required argument); `publish.MODEL_CARD` mentions the three outputs; `publish.tag_name(args) -> str` raises `ValueError` for an empty tag; `publish.assert_tag_is_new(existing_tags: list[str], tag: str) -> None` raises `ValueError` when the tag exists. A Hub tag never moves, so re-publishing under an existing tag must be refused, not tolerated.

- [ ] **Step 1: Write the failing tests**

Append to `tools/export/tests/test_publish_gate.py`:

```python
import argparse

from tsumugi_export.publish import MODEL_CARD, tag_name


def test_model_card_documents_the_readout_outputs():
    for name in ("last_hidden_state", "topk_ids", "topk_embeds", "[batch, 16, 576]"):
        assert name in MODEL_CARD


def test_tag_name_rejects_empty():
    assert tag_name(argparse.Namespace(tag="v3")) == "v3"
    for bad in ("", "  "):
        try:
            tag_name(argparse.Namespace(tag=bad))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_existing_tag_is_refused():
    from tsumugi_export.publish import assert_tag_is_new
    assert_tag_is_new(["v1", "v2"], "v3")          # new tag: no error
    try:
        assert_tag_is_new(["v1", "v2", "v3"], "v3")
    except ValueError as err:
        assert "v3" in str(err) and "v4" in str(err)   # the message proposes the next name
    else:
        raise AssertionError("expected ValueError")


def test_cli_requires_tag_for_publish():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "tsumugi_export", "publish"], capture_output=True, text=True)
    assert r.returncode == 2 and "--tag" in r.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --project tools/export pytest tools/export/tests/test_publish_gate.py -q`
Expected: FAIL with `ImportError: cannot import name 'tag_name'`.

- [ ] **Step 3: Implement**

In `tools/export/src/tsumugi_export/__main__.py`, after the `--repo-id` argument:

```python
    p_pub.add_argument("--tag", required=True,
                        help="Hub tag to create on the published commit (the app pins it: "
                             "ATTN_MODEL_REVISION in src/engine/tokenizer.ts); e.g. v3")
```

In `tools/export/src/tsumugi_export/publish.py`, insert into `MODEL_CARD` after the `inputs_embeds` paragraph:

```
Exposes the readout as well: `last_hidden_state` (`[batch, seq, 576]`, the
residual stream after the final RMSNorm — the tensor the tied `lm_head`
multiplies), `topk_ids` (`[batch, 16]`, the sixteen highest-scoring ids at the
last position) and `topk_embeds` (`[batch, 16, 576]`, their rows of the
embedding table, kept fp32). Tsumugi's Logits card shows the dot product
`last_hidden_state · topk_embeds[i]` cell by cell, summing to the logit.

Releases are tagged (`v3` added the readout outputs); the app pins a tag so
browsers that cached an older file re-download once.
```

Add the helper and the tag step:

```python
def tag_name(args) -> str:
    tag = (getattr(args, "tag", None) or "").strip()
    if not tag:
        raise ValueError("publish needs --tag (e.g. v3): the app pins the model by tag")
    return tag


def _next_tag(tag: str) -> str:
    digits = tag.lstrip("v")
    return f"v{int(digits) + 1}" if digits.isdigit() else f"{tag}-next"


def assert_tag_is_new(existing_tags: list[str], tag: str) -> None:
    """A Hub tag never moves. Publishing under an existing tag would leave the
    app (which pins the tag) on the earlier upload, and browsers that cached
    that URL never re-check — so a fix is always a new tag plus a bump of
    ATTN_MODEL_REVISION in src/engine/tokenizer.ts."""
    if tag in existing_tags:
        raise ValueError(
            f"tag {tag} already exists on the Hub and tags never move — publish as "
            f"{_next_tag(tag)} and set ATTN_MODEL_REVISION to match")
```

In `run`, before `api = HfApi()`, add `tag = tag_name(args)`, and replace from `api.create_repo` through the print with:

```python
    api.create_repo(repo_id, exist_ok=True)
    refs = api.list_repo_refs(repo_id)
    try:
        assert_tag_is_new([t.name for t in refs.tags], tag)
    except ValueError as err:
        print(err)
        return 1
    info = api.upload_folder(folder_path=str(model_dir), repo_id=repo_id,
                              ignore_patterns=["onnx/model.onnx"],
                              delete_patterns=["onnx/model_fp16.onnx", "onnx/model_q4.onnx"])
    api.create_tag(repo_id, tag=tag, revision=info.oid)
    print(f"published → https://huggingface.co/{repo_id} (tag {tag})")
```

The refs check runs before the upload, so a refused publish changes nothing on the Hub.

In `tools/export/README.md`:
- Intro paragraph: after "the embedding lookup (`inputs_embeds`)," add "the readout (`last_hidden_state`, `topk_ids`, `topk_embeds`),".
- Commands list, `publish` bullet: append "`--tag` (required) creates a Hub tag on the published commit; the app pins it via `ATTN_MODEL_REVISION`."
- Runbook step 6: add `*:last-hidden-parity` and `*:topk-parity` to the list of checks that must PASS.
- Runbook step 7 becomes: `uv run python -m tsumugi_export publish --model-dir out/model --tag v3`, followed by: "Then set `ATTN_MODEL_REVISION` in `src/engine/tokenizer.ts` to the same tag. Tags never move and `publish` refuses an existing one: if the in-app check in step 8 fails, fix, re-validate, publish under the **next** tag (`v4`) and bump the constant again."
- Runbook step 8: replace "if it fails, fix and re-publish" with "if it fails, fix and publish under a new tag (see step 7)"; add a bullet "**The Logits card** shows a dot-product bench for the top candidate; the sum under the product strip equals the logit on the bar to one decimal."

- [ ] **Step 4: Run the Python suite**

Run: `uv run --project tools/export pytest tools/export/tests -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add tools/export/src/tsumugi_export/publish.py tools/export/src/tsumugi_export/__main__.py tools/export/README.md tools/export/tests/test_publish_gate.py
/usr/bin/git commit -m "feat(export): publish --tag; model card and runbook for the readout outputs"
```

---

### Task 4: Operator checkpoint — export, validate, publish `v3`

**Files:** none in the repo. This task is run by the human operator; the executor stops and hands over the commands, then waits.

- [ ] **Step 1: Hand the operator this runbook (from `tools/export`)**

```bash
hf auth whoami                                                   # must be authenticated
uv run python -m tsumugi_export export --out out/model
uv run python -m tsumugi_export export --out out/model-nocache --no-cache
uv run python -m tsumugi_export quantize --model-dir out/model
uv run python -m tsumugi_export geometry --model-dir out/model
uv run python -m tsumugi_export validate --model-dir out/model
```

Expected in the verdict table: `a-equiv-b` ran (not SKIPPED); every `*:inputs-embeds-parity`, `*:last-hidden-parity`, `*:topk-parity` and `geometry` row PASS; `VERDICT: PASS`.

```bash
uv run python -m tsumugi_export publish --model-dir out/model --tag v3
```

- [ ] **Step 2: Confirm the Hub state**

Run: `curl -s https://huggingface.co/api/models/saigyo-hoshi/smollm2-135m-attn-onnx/refs`
Expected: a tag named `v3` is listed. Record its target commit in the ledger.

If validation fails, the fix belongs in Tasks 1–3; do not proceed to Task 5 until the tag exists. If a problem only shows up in the app after publishing, the fixed model goes out under the next tag (`v4`), because `publish` refuses an existing tag and a tag never moves; Task 5 then pins whatever tag was actually published. Record the final tag name in the ledger.

---

### Task 5: App — pin the model revision

**Files:**
- Modify: `src/engine/tokenizer.ts:9-14`
- Modify: `src/engine/transformers/worker.ts:44-60`
- Test: `src/engine/tokenizer.test.ts`

**Interfaces:**
- Produces: `ATTN_MODEL_REVISION = 'v3'`, `geometryUrlFor(modelId: string, revision: string): string` in `src/engine/tokenizer.ts`.

- [ ] **Step 1: Write the failing test**

Append to `src/engine/tokenizer.test.ts` (extend the import line to `import { ATTN_MODEL_ID, ATTN_MODEL_REVISION, GEOMETRY_BASE_URL, fallbackTokenizer, geometryUrlFor, loadTokenizer } from './tokenizer'`):

```ts
test('geometry asset URL is pinned to the model revision', () => {
  expect(geometryUrlFor('a/b', 'v9')).toBe('https://huggingface.co/a/b/resolve/v9/geometry')
  expect(ATTN_MODEL_REVISION).toMatch(/^v\d+$/)
  if (!import.meta.env.VITE_GEOMETRY_BASE_URL)
    expect(GEOMETRY_BASE_URL).toBe(geometryUrlFor(ATTN_MODEL_ID, ATTN_MODEL_REVISION))
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/engine/tokenizer.test.ts`
Expected: FAIL — `geometryUrlFor` is not exported.

- [ ] **Step 3: Implement**

In `src/engine/tokenizer.ts` replace the `ATTN_MODEL_ID` … `GEOMETRY_BASE_URL` block with:

```ts
export const ATTN_MODEL_ID = 'saigyo-hoshi/smollm2-135m-attn-onnx'
// Hub tag the app loads the model and the geometry asset from. Transformers.js
// caches files by URL and never revalidates, so a new release needs a new
// tag here to reach browsers that cached the previous one. v3 = readout
// outputs (last_hidden_state, topk_ids, topk_embeds).
export const ATTN_MODEL_REVISION = 'v3'

// Both ids share the same weights, so one geometry asset covers both.
export const GEOMETRY_MODEL_IDS: readonly string[] = [MODEL_ID, ATTN_MODEL_ID]
export const geometryUrlFor = (modelId: string, revision: string) =>
  `https://huggingface.co/${modelId}/resolve/${revision}/geometry`
export const GEOMETRY_BASE_URL: string =
  import.meta.env.VITE_GEOMETRY_BASE_URL ?? geometryUrlFor(ATTN_MODEL_ID, ATTN_MODEL_REVISION)
```

In `src/engine/transformers/worker.ts`: import `ATTN_MODEL_REVISION` next to `ATTN_MODEL_ID`, and give `loadOn` a revision:

```ts
  const loadOn = async (id: string, dtype: 'q4' | 'q8' | 'fp16', revision = 'main'): Promise<{ m: any; device: 'webgpu' | 'wasm' }> => {
    const opts = { dtype, progress_callback, revision }
    try {
      return { m: await AutoModelForCausalLM.from_pretrained(id, { ...opts, device: preferred }), device: preferred }
    } catch {
      return { m: await AutoModelForCausalLM.from_pretrained(id, { ...opts, device: 'wasm' }), device: 'wasm' }
    }
  }
```

and pass the revision on the two attention-model loads:

```ts
    ;({ m: model, device } = await loadOn(ATTN_MODEL_ID, 'q8', ATTN_MODEL_REVISION))
    ...
      ;({ m: model, device } = await loadOn(ATTN_MODEL_ID, 'fp16', ATTN_MODEL_REVISION))
```

The stock fallback `loadOn(modelId, 'q4')` keeps `main`.

- [ ] **Step 4: Run the tests and type-check**

Run: `npx vitest run src/engine` then `npx tsc -b`
Expected: PASS, no type errors.

- [ ] **Step 5: In-app check**

Start `npx vite --port 5199 --strictPort`, open the app in a browser with the model cache cleared or in a fresh profile, switch to real mode, run a curated example. Expected: the status chip shows `attn`, the network panel shows `resolve/v3/onnx/model_quantized.onnx`, the Layers card shows detected heads and the Embeddings card shows "Exact rows from the running model". Record the result in the ledger.

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add src/engine/tokenizer.ts src/engine/tokenizer.test.ts src/engine/transformers/worker.ts
/usr/bin/git commit -m "feat(engine): pin the attention model and geometry asset to Hub tag v3"
```

---

### Task 6: Trace — readout fields, validator, fixtures

**Files:**
- Modify: `src/trace/types.ts:48`
- Modify: `src/trace/validate.ts:71-73`
- Modify: `src/test/fixtures.ts`
- Test: `src/trace/validate.test.ts`

**Interfaces:**
- Produces: `READOUT_CELLS = 64`, `interface LogitCandidate extends TokenInfo { logit: number; row?: number[]; dot?: number[] }`, `logits` event `{ type: 'logits'; cycle: number; topK: LogitCandidate[]; lastHidden?: number[] }` in `src/trace/types.ts`; `buildFixtureTrace({ readout: true })` and `fixtureReadout(id: number, logit: number): { row: number[]; dot: number[] }`, `fixtureLastHidden(cycle: number): number[]` in `src/test/fixtures.ts`. Fixture `dot` cells sum to the candidate's logit within 1e-6.

- [ ] **Step 1: Write the failing tests**

Append to `src/trace/validate.test.ts`:

```ts
import { READOUT_CELLS } from './types'
import { buildFixtureTrace, fixtureReadout } from '../test/fixtures'

test('readout fields on logits events validate, and dot cells sum to the logit', () => {
  const t = buildFixtureTrace({ readout: true })
  expect(validateTrace(t)).toEqual([])
  const logits = t.find((e) => e.type === 'logits')
  if (logits?.type !== 'logits') throw new Error('no logits event')
  expect(logits.lastHidden).toHaveLength(576)
  for (const c of logits.topK) {
    expect(c.row).toHaveLength(READOUT_CELLS)
    expect(c.dot).toHaveLength(READOUT_CELLS)
    expect(c.dot!.reduce((a, b) => a + b, 0)).toBeCloseTo(c.logit, 6)
  }
})

test('readout fields with wrong length are flagged', () => {
  const t = buildFixtureTrace({ readout: true })
  const logits = t.find((e) => e.type === 'logits')
  if (logits?.type !== 'logits') throw new Error('no logits event')
  logits.topK[0].dot = [1, 2, 3]
  logits.lastHidden = [1, Number.NaN]
  const errs = validateTrace(t)
  expect(errs.some((e) => /dot/.test(e) && /64/.test(e))).toBe(true)
  expect(errs.some((e) => /lastHidden/.test(e))).toBe(true)
})

test('logits events without readout fields stay valid', () => {
  expect(validateTrace(buildFixtureTrace())).toEqual([])
})

test('fixtureReadout is deterministic and sums to the logit', () => {
  expect(fixtureReadout(42, 9.1)).toEqual(fixtureReadout(42, 9.1))
  expect(fixtureReadout(42, 9.1).dot.reduce((a, b) => a + b, 0)).toBeCloseTo(9.1, 6)
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/trace/validate.test.ts`
Expected: FAIL — `READOUT_CELLS` and `fixtureReadout` do not exist.

- [ ] **Step 3: Implement**

`src/trace/types.ts` — replace the `logits` line and add above `TraceEvent`:

```ts
// The readout strips: the final vector and E rows pooled into this many cells
// (576 / 64 = 9 dimensions per cell). Shared by the worker's pooling and the
// Logits card's pooling of lastHidden, so the strips line up cell for cell.
export const READOUT_CELLS = 64

export interface LogitCandidate extends TokenInfo {
  logit: number
  // real mode with the v3 export: the candidate's row of E pooled by mean,
  // and the element-wise product x_last ⊙ E[id] pooled by sum (Σ dot ≈ logit)
  row?: number[]
  dot?: number[]
}
```

```ts
  | { type: 'logits'; cycle: number; topK: LogitCandidate[]; lastHidden?: number[] }
```

`src/trace/validate.ts` — replace the `logits` branch at the end:

```ts
    if (e.type === 'logits') {
      for (let i = 1; i < e.topK.length; i++)
        if (e.topK[i].logit > e.topK[i - 1].logit) errs.push('logits topK not sorted desc')
      const finite = (a: unknown): a is number[] =>
        Array.isArray(a) && a.every((v) => typeof v === 'number' && Number.isFinite(v))
      if (e.lastHidden !== undefined && !finite(e.lastHidden))
        errs.push(`logits cycle ${e.cycle} lastHidden is not an array of finite numbers`)
      for (const c of e.topK) {
        for (const key of ['row', 'dot'] as const) {
          const v = c[key]
          if (v !== undefined && (!finite(v) || v.length !== READOUT_CELLS))
            errs.push(`logits cycle ${e.cycle} candidate ${c.id} ${key} is not ${READOUT_CELLS} finite numbers`)
        }
      }
    }
```

and import `READOUT_CELLS` from `./types` (change the type-only import to `import { READOUT_CELLS, type TraceEvent } from './types'`).

`src/test/fixtures.ts` — add `readout?: boolean` to `FixtureTraceOpts`, import `READOUT_CELLS` and `LogitCandidate`, and add:

```ts
const r3 = (v: number) => Math.round(v * 1000) / 1000

// Deterministic final vector for a cycle, 576 values.
export function fixtureLastHidden(cycle: number): number[] {
  return Array.from({ length: 576 }, (_, d) => r3(Math.cos(cycle * 0.9 + d * 0.07)))
}

// Deterministic pooled strips for a candidate: `row` is any smooth signal,
// `dot` is built so its cells sum to the logit exactly (last cell absorbs
// the rounding), because the card displays that sum next to the logit.
export function fixtureReadout(id: number, logit: number): { row: number[]; dot: number[] } {
  const row = Array.from({ length: READOUT_CELLS }, (_, c) => r3(Math.cos(id * 0.5 + c * 0.3)))
  const dot = Array.from({ length: READOUT_CELLS }, (_, c) => r3(logit / READOUT_CELLS + 0.3 * Math.sin(id * 0.7 + c * 0.4)))
  const partial = dot.slice(0, -1).reduce((a, b) => a + b, 0)
  dot[READOUT_CELLS - 1] = Math.round((logit - partial) * 1e6) / 1e6
  return { row, dot }
}
```

and in `buildFixtureTrace`, replace the `logits` push:

```ts
    const scored: LogitCandidate[] = [
      { ...chosen, logit: 9.1 }, { id: 200, text: ' ran', logit: 7.2 }, { id: 201, text: ' was', logit: 5.0 },
    ]
    events.push(opts.readout
      ? { type: 'logits', cycle: c, lastHidden: fixtureLastHidden(c),
          topK: scored.map((s) => ({ ...s, ...fixtureReadout(s.id, s.logit) })) }
      : { type: 'logits', cycle: c, topK: scored })
```

- [ ] **Step 4: Run the suite and type-check**

Run: `npx vitest run` then `npx tsc -b`
Expected: all PASS (the fixture default is unchanged, so index-based tests hold).

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/trace/types.ts src/trace/validate.ts src/trace/validate.test.ts src/test/fixtures.ts
/usr/bin/git commit -m "feat(trace): optional readout fields on logits events (lastHidden, row, dot)"
```

---

### Task 7: Worker — `extractReadout` helper and emission

**Files:**
- Create: `src/engine/transformers/readout.ts`
- Create: `src/engine/transformers/readout.test.ts`
- Modify: `src/engine/transformers/worker.ts:200-206`

**Interfaces:**
- Consumes: `READOUT_CELLS` (Task 6), `poolRow` from `src/geometry/math.ts`.
- Produces: `extractReadout(out, fed, dims, cells = READOUT_CELLS): ReadoutResult` where
  `ReadoutResult = { status: 'absent' } | { status: 'bad-shape'; detail: string } | { status: 'ok'; lastHidden: number[]; byId: Map<number, { row: number[]; dot: number[] }> }`
  and `out` is `{ last_hidden_state?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish }`, `Tensorish = { dims: number[]; data: ArrayLike<number> | BigInt64Array }`.

- [ ] **Step 1: Write the failing tests**

`src/engine/transformers/readout.test.ts`:

```ts
import { expect, test } from 'vitest'
import { extractReadout } from './readout'

const DIMS = 8
const CELLS = 4   // 2 dims per cell
const row = (seed: number) => Array.from({ length: DIMS }, (_, d) => Math.round(Math.sin(seed + d) * 1000) / 1000)
const last = row(1)
const E: Record<number, number[]> = { 5: row(2), 9: row(3), 2: row(4) }

function outputs(ids: Array<number | bigint> = [5, 9, 2], fed = 2) {
  const hidden = new Float32Array(fed * DIMS)
  hidden.set(row(99), 0)                 // an earlier fed token, must be ignored
  hidden.set(last, (fed - 1) * DIMS)     // the last fed token
  const embeds = new Float32Array(ids.length * DIMS)
  ids.forEach((id, i) => embeds.set(E[Number(id)], i * DIMS))
  const idData = typeof ids[0] === 'bigint' ? BigInt64Array.from(ids as bigint[]) : Int32Array.from(ids as number[])
  return {
    last_hidden_state: { dims: [1, fed, DIMS], data: hidden },
    topk_ids: { dims: [1, ids.length], data: idData },
    topk_embeds: { dims: [1, ids.length, DIMS], data: embeds },
  }
}

test('absent outputs → absent', () => {
  expect(extractReadout({}, 2, DIMS, CELLS)).toEqual({ status: 'absent' })
  expect(extractReadout({ last_hidden_state: outputs().last_hidden_state }, 2, DIMS, CELLS).status).toBe('bad-shape')
})

test('wrong shapes → bad-shape naming the output', () => {
  const o = outputs()
  o.last_hidden_state.dims = [1, 3, DIMS]
  const r = extractReadout(o, 2, DIMS, CELLS)
  expect(r.status).toBe('bad-shape')
  if (r.status === 'bad-shape') expect(r.detail).toMatch(/last_hidden_state/)
  const p = outputs()
  p.topk_embeds.dims = [1, 3, DIMS + 1]
  const q = extractReadout(p, 2, DIMS, CELLS)
  if (q.status === 'bad-shape') expect(q.detail).toMatch(/topk_embeds/)
  else throw new Error('expected bad-shape')
})

test('ok → last fed row rounded to 3 decimals, rows pooled by mean, dots by sum, keyed by id', () => {
  const r = extractReadout(outputs(), 2, DIMS, CELLS)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.lastHidden).toEqual(last.map((v) => Math.round(v * 1000) / 1000))
  expect([...r.byId.keys()]).toEqual([5, 9, 2])
  const e5 = E[5]
  const mean = (a: number, b: number) => Math.round(((a + b) / 2) * 1000) / 1000
  expect(r.byId.get(5)!.row).toEqual([mean(e5[0], e5[1]), mean(e5[2], e5[3]), mean(e5[4], e5[5]), mean(e5[6], e5[7])])
  const dot = r.byId.get(5)!.dot
  expect(dot).toHaveLength(CELLS)
  const exact = e5.reduce((s, v, d) => s + v * last[d], 0)
  expect(dot.reduce((a, b) => a + b, 0)).toBeCloseTo(exact, 2)
})

test('accepts BigInt64Array ids', () => {
  const r = extractReadout(outputs([5n, 9n, 2n]), 2, DIMS, CELLS)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.byId.has(9)).toBe(true)
})

test('id missing from topk_ids is absent from the map', () => {
  const r = extractReadout(outputs([5, 9, 2]), 2, DIMS, CELLS)
  if (r.status !== 'ok') throw new Error(r.status)
  expect(r.byId.get(7)).toBeUndefined()
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engine/transformers/readout.test.ts`
Expected: FAIL — module `./readout` not found.

- [ ] **Step 3: Implement the helper**

`src/engine/transformers/readout.ts`:

```ts
import { poolRow } from '../../geometry/math'
import { READOUT_CELLS } from '../../trace/types'

// Turns the v3 export's readout outputs into the logits event's fields. Pure,
// so the shape policy and the pooling are unit-testable outside the worker.
//   last_hidden_state [1, fed, dims] → the last fed row, exact (3 decimals)
//   topk_ids [1, k] + topk_embeds [1, k, dims] → per candidate id: its row of E
//   pooled by MEAN and the product x_last ⊙ E[id] pooled by SUM, so the dot
//   cells add up to the logit.
export type Tensorish = { dims: number[]; data: ArrayLike<number> | BigInt64Array }
export interface ReadoutOutputs { last_hidden_state?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish }
export type ReadoutResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; lastHidden: number[]; byId: Map<number, { row: number[]; dot: number[] }> }

const r3 = (v: number) => Math.round(v * 1000) / 1000

function poolSum(values: number[], cells: number): number[] {
  const len = values.length
  const out: number[] = []
  for (let c = 0; c < cells; c++) {
    const start = Math.floor((c * len) / cells), end = Math.floor(((c + 1) * len) / cells)
    let s = 0
    for (let i = start; i < end; i++) s += values[i]
    out.push(s)
  }
  return out
}

const sameDims = (t: Tensorish, want: number[]) =>
  t.dims.length === want.length && t.dims.every((d, i) => d === want[i])

export function extractReadout(out: ReadoutOutputs, fed: number, dims: number, cells = READOUT_CELLS): ReadoutResult {
  const { last_hidden_state: lh, topk_ids: ids, topk_embeds: emb } = out
  if (!lh && !ids && !emb) return { status: 'absent' }
  if (!lh || !ids || !emb)
    return { status: 'bad-shape', detail: `readout outputs incomplete: last_hidden_state ${!!lh}, topk_ids ${!!ids}, topk_embeds ${!!emb}` }
  if (!sameDims(lh, [1, fed, dims]) || lh.data.length !== fed * dims)
    return { status: 'bad-shape', detail: `last_hidden_state has shape [${lh.dims.join(', ')}], expected [1, ${fed}, ${dims}]` }
  const k = ids.dims[1]
  if (ids.dims.length !== 2 || ids.dims[0] !== 1 || ids.data.length !== k)
    return { status: 'bad-shape', detail: `topk_ids has shape [${ids.dims.join(', ')}], expected [1, k]` }
  if (!sameDims(emb, [1, k, dims]) || emb.data.length !== k * dims)
    return { status: 'bad-shape', detail: `topk_embeds has shape [${emb.dims.join(', ')}], expected [1, ${k}, ${dims}]` }

  const base = (fed - 1) * dims
  const x = Array.from({ length: dims }, (_, d) => Number(lh.data[base + d]))
  const byId = new Map<number, { row: number[]; dot: number[] }>()
  for (let i = 0; i < k; i++) {
    const id = Number(ids.data[i])
    const row = Array.from({ length: dims }, (_, d) => Number(emb.data[i * dims + d]))
    byId.set(id, {
      row: poolRow(row, cells).map(r3),
      dot: poolSum(row.map((v, d) => v * x[d]), cells).map(r3),
    })
  }
  return { status: 'ok', lastHidden: x.map(r3), byId }
}
```

- [ ] **Step 4: Run the tests**

Run: `npx vitest run src/engine/transformers/readout.test.ts`
Expected: PASS.

- [ ] **Step 5: Wire the worker**

In `src/engine/transformers/worker.ts`: import `extractReadout, type ReadoutResult` from `./readout`; declare `let readoutOff = false` next to `let embedsOff …` inside `run`; replace the two lines that build `top` and emit `logits` with:

```ts
      const top = topK(lastLogits, params.topK).map((c) => ({ ...tokenInfo(c.id), logit: Math.round(c.logit * 100) / 100 }))
      // Readout for the Logits card (v3 export): never-fail, like inputs_embeds.
      let readout: ReadoutResult = { status: 'absent' }
      if (!readoutOff) {
        readout = extractReadout(out, nextInputIds.length, dims)
        if (readout.status === 'bad-shape') {
          readoutOff = true
          console.warn(`readout outputs unusable (${readout.detail}) — the Logits card shows the schematic readout`)
        }
      }
      const candidates = top.map((c) => {
        const r = readout.status === 'ok' ? readout.byId.get(c.id) : undefined
        return r ? { ...c, row: r.row, dot: r.dot } : c
      })
      emit({ type: 'logits', cycle, topK: candidates,
        ...(readout.status === 'ok' ? { lastHidden: readout.lastHidden } : {}) })
```

The later `softmax` emission reads `top`, which is unchanged.

- [ ] **Step 6: Type-check and run the suite**

Run: `npx tsc -b` then `npx vitest run`
Expected: no type errors (the `out` object is `any`, so it satisfies `ReadoutOutputs`), all PASS.

- [ ] **Step 7: In-app check**

With the dev server running and the v3 model loaded, run a real-mode generation, export the run as JSON from the shelf, and inspect one `logits` event: `lastHidden` has 576 entries; each candidate has 64-entry `row` and `dot`; `dot` sums to the candidate's logit within 0.05. Record the observed sum and logit in the ledger.

- [ ] **Step 8: Commit**

```bash
/usr/bin/git add src/engine/transformers/readout.ts src/engine/transformers/readout.test.ts src/engine/transformers/worker.ts
/usr/bin/git commit -m "feat(engine): emit the readout — final vector and pooled candidate rows/products"
```

---

### Task 8: Viz — extract `VectorStrip` from `EmbeddingLookup`

**Files:**
- Create: `src/viz/VectorStrip.tsx`
- Modify: `src/viz/EmbeddingLookup.tsx`
- Test: `src/viz/details/EmbeddingsDetail.test.tsx` (must stay green, no edits)

**Interfaces:**
- Produces: `cellFill(v: number, peak: number): string`, `peakOf(cells: number[]): number`, `rowMarkY(id: number): number` (= `40 + (id % 41)`), and
  `VectorStrip({ cells, cellW, cellH, gap = 1, testId, cellTestId, ariaLabel, titleFor, className })` rendering one `<svg>` of `cells.length` rects.

- [ ] **Step 1: Create the shared module**

`src/viz/VectorStrip.tsx`:

```tsx
// Shared drawing vocabulary for vectors: the Embeddings card's row strips and
// the Logits card's readout strips use the same cells and the same colours, so
// a reader recognises "this is a row of E" across the two stages.

// Diverging fill: indigo for positive, vermillion for negative, intensity
// relative to the strip's own peak magnitude.
export function cellFill(v: number, peak: number): string {
  const t = Math.abs(v) / peak
  return v >= 0
    ? `hsl(211 45% ${Math.round(88 - t * 50)}%)`
    : `hsl(13 55% ${Math.round(90 - t * 40)}%)`
}

export const peakOf = (cells: number[]) => cells.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1

// A schematic row position inside the drawn E box for a token id.
export const rowMarkY = (id: number) => 40 + (id % 41)

export function VectorStrip({ cells, cellW, cellH, gap = 1, testId, cellTestId, ariaLabel, titleFor, className }: {
  cells: number[]
  cellW: number
  cellH: number
  gap?: number
  testId?: string
  cellTestId?: string
  ariaLabel: string
  titleFor?: (index: number, value: number) => string
  className?: string
}) {
  const peak = peakOf(cells)
  const w = cells.length * cellW
  return (
    <svg data-testid={testId} className={className} width={w} height={cellH} viewBox={`0 0 ${w} ${cellH}`} role="img" aria-label={ariaLabel}>
      {cells.map((v, c) => (
        <rect key={c} data-testid={cellTestId} x={c * cellW} y={0} width={cellW - gap} height={cellH} fill={cellFill(v, peak)}>
          {titleFor && <title>{titleFor(c, v)}</title>}
        </rect>
      ))}
    </svg>
  )
}
```

- [ ] **Step 2: Use it in `EmbeddingLookup.tsx`**

Remove the local `cellFill` and `peakOf` definitions; import `{ VectorStrip, cellFill, peakOf, rowMarkY } from './VectorStrip'`. Replace `const rowY = token ? 40 + (token.id % 41) : 0` with `const rowY = token ? rowMarkY(token.id) : 0`. Replace the main strip `<svg data-testid="embed-strip" …>…</svg>` with:

```tsx
          <VectorStrip cells={cells} cellW={CELL_W} cellH={CELL_H} testId="embed-strip" cellTestId="embed-strip-cell"
            ariaLabel={`embedding row of token ${token?.text ?? ''}`} />
```

The stacked rows keep using `cellFill` directly (they live inside one shared `<svg>`).

- [ ] **Step 3: Run the viz tests and type-check**

Run: `npx vitest run src/viz` then `npx tsc -b`
Expected: all PASS unchanged, including `model-source rows render a 96-cell pooled strip`.

- [ ] **Step 4: Commit**

```bash
/usr/bin/git add src/viz/VectorStrip.tsx src/viz/EmbeddingLookup.tsx
/usr/bin/git commit -m "refactor(viz): share the vector strip between the Embeddings and Logits cards"
```

---

### Task 9: Viz — `LogitsReadout`, the container, CSS

**Files:**
- Create: `src/viz/LogitsReadout.tsx`
- Create: `src/viz/LogitsReadout.test.tsx`
- Modify: `src/viz/details/LogitsDetail.tsx` (rewrite as container)
- Create: `src/viz/details/LogitsDetail.test.tsx`
- Modify: `src/viz/DetailPanel.tsx:63-70`
- Modify: `src/index.css` (after the Embeddings card block)
- Test: `src/viz/DetailPanel.test.tsx` (existing logits tests stay green)

**Interfaces:**
- Consumes: `LogitCandidate`, `READOUT_CELLS` (Task 6); `VectorStrip`, `rowMarkY` (Task 8); `poolRow` (`src/geometry/math.ts`); `visibleToken` (`src/viz/tokenText.ts`); `thousands` (`src/viz/selectors.ts`); `useGeometry`, `coversModel` as used in `EmbeddingsDetail.tsx`.
- Produces: `LogitsReadout(props: LogitsReadoutProps)` and `LogitsDetail({ events, cursor })`.

```ts
export interface LogitsReadoutProps {
  candidates: LogitCandidate[]
  probs?: number[]                 // softmax cursor: one per candidate, same order
  lastHidden?: number[]
  dims: number
  vocabSize?: number
  selected: number
  onSelect: (index: number) => void
  assetRowFor: (id: number) => ArrayLike<number> | undefined   // PCA fallback rows
  assetNote: string                // shown when assetRowFor returns undefined
}
```

- [ ] **Step 1: Write the failing component tests**

`src/viz/LogitsReadout.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { READOUT_CELLS, type LogitCandidate } from '../trace/types'
import { fixtureLastHidden, fixtureReadout } from '../test/fixtures'
import { LogitsReadout, type LogitsReadoutProps } from './LogitsReadout'

afterEach(cleanup)

const real: LogitCandidate[] = [
  { id: 100, text: ' sat', logit: 9.1, ...fixtureReadout(100, 9.1) },
  { id: 200, text: ' ran', logit: 7.2, ...fixtureReadout(200, 7.2) },
  { id: 201, text: ' was', logit: 5.0 },                      // no graph row (tie at rank 16)
]
const sim: LogitCandidate[] = real.map(({ id, text, logit }) => ({ id, text, logit }))

function props(over: Partial<LogitsReadoutProps> = {}): LogitsReadoutProps {
  return {
    candidates: real, lastHidden: fixtureLastHidden(0), dims: 576, vocabSize: 49152,
    selected: 0, onSelect: vi.fn(), assetRowFor: (id) => (id < 256 ? [1, 2, 3, 4] : undefined),
    assetNote: 'vector values unavailable', ...over,
  }
}

test('real run: x strip, one product strip per candidate with a row, bench sum equals the logit', () => {
  render(<LogitsReadout {...props()} />)
  expect(screen.getByTestId('readout-x').querySelectorAll('rect')).toHaveLength(READOUT_CELLS)
  expect(screen.getAllByTestId('readout-product')).toHaveLength(2)
  expect(screen.getAllByTestId('readout-candidate')).toHaveLength(3)
  expect(screen.getAllByTestId('logit-bar')).toHaveLength(3)
  expect(screen.getByTestId('readout-sum')).toHaveTextContent('9.1')
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('sat')
})

test('candidate without dot renders without a product strip and gets the asset bench', () => {
  render(<LogitsReadout {...props({ selected: 2 })} />)
  const bench = screen.getByTestId('readout-bench')
  expect(bench).toHaveTextContent('PCA-reduced')
  expect(bench).toHaveTextContent(/real-mode generation/)
  expect(bench.querySelector('[data-testid="bench-product"]')).toBeNull()
})

test('clicking a candidate reports its index', () => {
  const onSelect = vi.fn()
  render(<LogitsReadout {...props({ onSelect })} />)
  fireEvent.click(screen.getAllByTestId('readout-candidate')[1])
  expect(onSelect).toHaveBeenCalledWith(1)
})

test('sim/legacy run: placeholder x, asset rows, no products, one-liner', () => {
  render(<LogitsReadout {...props({ candidates: sim, lastHidden: undefined })} />)
  expect(screen.getByTestId('readout-x-missing')).toHaveTextContent('[1×576]')
  expect(screen.queryAllByTestId('readout-product')).toHaveLength(0)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent(/real-mode generation/)
  expect(screen.getByTestId('bench-row').querySelectorAll('rect')).toHaveLength(4)   // fixture asset vectors have 4 dims
})

test('asset row unavailable shows the note instead of a strip', () => {
  render(<LogitsReadout {...props({ candidates: sim, lastHidden: undefined, assetRowFor: () => undefined })} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('vector values unavailable')
})

test('softmax state shows percentages while strips stay', () => {
  render(<LogitsReadout {...props({ probs: [0.7, 0.2, 0.1] })} />)
  expect(screen.getAllByTestId('readout-candidate')[0]).toHaveTextContent('70%')
  expect(screen.getAllByTestId('readout-product')).toHaveLength(2)
})

test('matrix sketch marks every candidate and the callouts explain the tie', () => {
  render(<LogitsReadout {...props()} />)
  expect(screen.getByTestId('readout-matrix').querySelectorAll('[data-testid="readout-row-mark"]')).toHaveLength(3)
  const callouts = screen.getAllByTestId('embed-callout').map((c) => c.textContent)
  expect(callouts).toEqual(['ⓘ tied with Embeddings', 'ⓘ similar rows, similar scores', 'ⓘ 49 152 scores, 3 shown'])
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/viz/LogitsReadout.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `LogitsReadout.tsx`**

```tsx
import { poolRow } from '../geometry/math'
import { READOUT_CELLS, type LogitCandidate } from '../trace/types'
import { thousands } from './selectors'
import { visibleToken } from './tokenText'
import { VectorStrip, rowMarkY } from './VectorStrip'

const X_CELL_W = 5, X_CELL_H = 14          // equation row and bench strips
const DOT_CELL_W = 3, DOT_CELL_H = 10      // compact product strip per candidate
const BAR_W = 160

const CALLOUTS = (shown: number, vocab: string): Array<{ label: string; text: string }> => [
  { label: 'tied with Embeddings',
    text: 'This is the Embeddings matrix read sideways: there a row was looked up by id; here every row is scored against x_last. Tied weights mean the model learns one vocabulary geometry, not two.' },
  { label: 'similar rows, similar scores',
    text: 'Tokens whose rows of E point the same way get nearly the same dot product, which is why near-synonyms sit together at the top.' },
  { label: `${vocab} scores, ${shown} shown`,
    text: 'The multiply produces one score per vocabulary entry; the trace keeps only the top-k the sampler can choose from.' },
]

export interface LogitsReadoutProps {
  candidates: LogitCandidate[]
  probs?: number[]
  lastHidden?: number[]
  dims: number
  vocabSize?: number
  selected: number
  onSelect: (index: number) => void
  assetRowFor: (id: number) => ArrayLike<number> | undefined
  assetNote: string
}

const sum = (a: number[]) => a.reduce((s, v) => s + v, 0)

// The Embeddings lookup read in reverse: x_last · Eᵀ = scores, then a bench
// that lays the selected candidate's dot product out cell by cell. Pure
// presentation; the container resolves vectors and owns the selection.
export function LogitsReadout({ candidates, probs, lastHidden, dims, vocabSize, selected, onSelect, assetRowFor, assetNote }: LogitsReadoutProps) {
  const vocab = vocabSize ? thousands(vocabSize) : '?'
  const xCells = lastHidden ? poolRow(lastHidden, READOUT_CELLS) : undefined
  const per = Math.round(dims / READOUT_CELLS)
  const maxLogit = Math.max(...candidates.map((c) => c.logit))
  const sel = candidates[Math.min(selected, candidates.length - 1)]
  const exact = sel?.dot !== undefined && sel.row !== undefined && xCells !== undefined
  const assetVec = sel && !exact ? assetRowFor(sel.id) : undefined
  const assetCells = assetVec ? poolRow(assetVec, READOUT_CELLS) : undefined
  const dotSum = exact ? sum(sel.dot!) : undefined
  return (
    <div data-testid="logits-readout" className="logits-readout">
      <div className="readout-equation">
        <div className="readout-x">
          <div className="embed-col-label">x_last [1×{dims}]</div>
          {xCells ? (
            <VectorStrip cells={xCells} cellW={X_CELL_W} cellH={X_CELL_H} testId="readout-x"
              ariaLabel="the last token’s final vector, pooled" titleFor={(c, v) => `dims ${c * per}–${c * per + per - 1}: mean ${v.toFixed(2)}`} />
          ) : (
            <div data-testid="readout-x-missing" className="readout-x-missing">[1×{dims}] — not recorded in this run</div>
          )}
          <div className="embed-caption">the last position after the final layer and norm</div>
        </div>
        <div className="readout-op">·</div>
        <svg data-testid="readout-matrix" className="embed-matrix" width="150" height="110" viewBox="0 0 150 110" role="img"
          aria-label={`the embedding matrix E read as Eᵀ, ${vocab} rows by ${dims} columns`}>
          <rect x="36" y="6" width="90" height="98" rx="4" className="rs-box" />
          <text x="81" y="24" textAnchor="middle" className="rs-box-label">Eᵀ</text>
          <text x="81" y="98" textAnchor="middle" className="rs-shape">[{vocab} × {dims}]</text>
          {candidates.map((c, i) => (
            <rect key={c.id} data-testid="readout-row-mark" x="36" y={rowMarkY(c.id)} width="90" height={i === selected ? 4 : 2}
              className={`readout-row-mark hue-fill-${i % 6}`} data-selected={String(i === selected)} />
          ))}
          <text x="4" y="60" className="rs-note">rows</text>
        </svg>
        <div className="readout-op">=</div>
        <div className="readout-scores">
          <div className="embed-col-label">{probs ? 'softmax →' : 'scores'} [{vocab}]</div>
          {candidates.map((c, i) => {
            const frac = probs ? probs[i] : Math.max(0.02, c.logit / maxLogit)
            return (
              <button key={c.id} type="button" data-testid="readout-candidate" data-selected={String(i === selected)}
                className="readout-candidate" onClick={() => onSelect(i)} title="Show this dot product on the bench">
                <span className={`token-chip hue-${i % 6}`}><span className="chip-text">{visibleToken(c.text)}</span></span>
                {c.dot ? (
                  <VectorStrip cells={c.dot} cellW={DOT_CELL_W} cellH={DOT_CELL_H} gap={0} testId="readout-product"
                    ariaLabel={`products for ${c.text}`} />
                ) : <span className="readout-nostrip" aria-hidden="true" />}
                <svg data-testid="logit-bar" data-token={c.text} width={BAR_W} height="14">
                  <rect width={BAR_W * frac} height="14" className="bar-rect" style={{ transition: 'width .4s' }} />
                </svg>
                <span className="bar-value">{probs ? `${Math.round(probs[i] * 100)}%` : c.logit.toFixed(1)}</span>
              </button>
            )
          })}
        </div>
      </div>

      {sel && (
        <div data-testid="readout-bench" className="readout-bench">
          <div className="embed-col-label">x_last ⊙ E[{visibleToken(sel.text)}]</div>
          <div className="readout-bench-rows">
            <span className="readout-bench-label">x_last</span>
            {xCells
              ? <VectorStrip cells={xCells} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-x" ariaLabel="x_last, pooled" />
              : <span className="readout-x-missing">[1×{dims}] — not recorded in this run</span>}
            <span className="readout-bench-label">E[{sel.id}]</span>
            {exact
              ? <VectorStrip cells={sel.row!} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-row" ariaLabel={`row ${sel.id} of E, pooled`} />
              : assetCells
                ? <VectorStrip cells={assetCells} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-row" ariaLabel={`row ${sel.id} of E, PCA-reduced`} />
                : <span className="embed-strip-missing">{assetNote}</span>}
            {exact && (
              <>
                <span className="readout-bench-label">product</span>
                <VectorStrip cells={sel.dot!} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-product" ariaLabel="element-wise products, summed per cell"
                  titleFor={(c, v) => `dims ${c * per}–${c * per + per - 1}: x̄ ${xCells![c].toFixed(2)} · ē ${sel.row![c].toFixed(2)} → ${v.toFixed(2)}`} />
              </>
            )}
          </div>
          {exact ? (
            <div data-testid="readout-sum" className="readout-sum">
              Σ of the {READOUT_CELLS} cells = {dotSum!.toFixed(1)} — the logit on the bar
              {Math.abs(dotSum! - sel.logit) > 0.05 ? ` (logit ${sel.logit.toFixed(1)})` : ''}
            </div>
          ) : (
            <div data-testid="readout-sum" className="readout-sum">
              {assetCells ? `${assetCells.length} of ${dims} dimensions (PCA-reduced, offline). ` : ''}
              Run a real-mode generation to see the actual products.
            </div>
          )}
          <div className="embed-caption">
            {READOUT_CELLS} cells of {per} dimensions; blue cells add to the score, red cells subtract.
          </div>
        </div>
      )}

      <div className="embed-callouts">
        {CALLOUTS(candidates.length, vocab).map((c) => (
          <span key={c.label} data-testid="embed-callout" className="embed-callout rs-hover" title={c.text}>ⓘ {c.label}</span>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Run the component tests**

Run: `npx vitest run src/viz/LogitsReadout.test.tsx`
Expected: PASS.

- [ ] **Step 5: Write the failing container tests**

`src/viz/details/LogitsDetail.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../../test/fixtures'
import { makeGeometryAsset } from '../../test/geometryFixture'
import type { GeometryState } from '../../geometry/useGeometry'
import { LogitsDetail } from './LogitsDetail'

const ready = (): GeometryState => ({ status: 'ready', asset: makeGeometryAsset(), retry: () => {} })
let geo: GeometryState = ready()
vi.mock('../../geometry/useGeometry', () => ({ useGeometry: () => geo }))
afterEach(() => { cleanup(); geo = ready() })

// fixture indices: cycle 0 logits = 7, softmax = 8; cycle 1 logits = 16

test('real fixture: bench sum equals the top candidate’s logit; formula and copy present', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true })} cursor={7} />)
  expect(screen.getByTestId('logits-formula')).toHaveTextContent('[1×576]')
  expect(screen.getByTestId('detail-logits')).toHaveTextContent(/dot product/i)
  expect(screen.getByTestId('readout-sum')).toHaveTextContent('9.1')
})

test('softmax cursor keeps the readout and shows percentages', () => {
  render(<LogitsDetail events={buildFixtureTrace({ readout: true })} cursor={8} />)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('Softmax')
  expect(screen.getAllByTestId('readout-candidate')[0]).toHaveTextContent('70%')
  expect(screen.getAllByTestId('readout-product').length).toBeGreaterThan(0)
})

test('selection resets when the cursor moves to another cycle', () => {
  const t = buildFixtureTrace({ readout: true })
  const { rerender } = render(<LogitsDetail events={t} cursor={16} />)
  fireEvent.click(screen.getAllByTestId('readout-candidate')[2])
  expect(screen.getAllByTestId('readout-candidate')[2].dataset.selected).toBe('true')
  rerender(<LogitsDetail events={t} cursor={7} />)
  expect(screen.getAllByTestId('readout-candidate')[0].dataset.selected).toBe('true')
})

test('sim fixture resolves candidate rows from the geometry asset', () => {
  render(<LogitsDetail events={makeFixtureTrace()} cursor={7} />)
  expect(screen.getByTestId('readout-x-missing')).toBeInTheDocument()
  expect(screen.getByTestId('bench-row').querySelectorAll('rect')).toHaveLength(4)
  expect(screen.getByTestId('readout-sum')).toHaveTextContent('PCA-reduced')
})

test('asset row lookup is guarded for ids outside the vocabulary', () => {
  const t = makeFixtureTrace()
  const logits = t[7]
  if (logits.type !== 'logits') throw new Error('fixture index moved')
  logits.topK[0].id = 99999   // fallback hash tokenizer ids can exceed the asset vocab (256 in the fixture)
  render(<LogitsDetail events={t} cursor={7} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('unavailable')
})

test('geometry loading and error states show their notes', () => {
  geo = { status: 'loading', retry: () => {} }
  render(<LogitsDetail events={makeFixtureTrace()} cursor={7} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent(/loading/i)
  cleanup()
  geo = { status: 'error', error: 'offline', retry: () => {} }
  render(<LogitsDetail events={makeFixtureTrace()} cursor={7} />)
  expect(screen.getByTestId('readout-bench')).toHaveTextContent('unavailable')
})
```

- [ ] **Step 6: Run the container tests to verify they fail**

Run: `npx vitest run src/viz/details/LogitsDetail.test.tsx`
Expected: FAIL — `LogitsDetail` does not accept `events`/`cursor`.

- [ ] **Step 7: Rewrite the container and wire `DetailPanel`**

`src/viz/details/LogitsDetail.tsx`:

```tsx
import { useState } from 'react'
import { coversModel } from '../../geometry/asset'
import { useGeometry } from '../../geometry/useGeometry'
import type { TraceEvent } from '../../trace/types'
import { LogitsReadout } from '../LogitsReadout'
import { latestOfType, thousands } from '../selectors'

// Container: picks the cycle's logits/softmax events, resolves asset rows for
// sim and legacy traces, and owns the selected candidate (snaps to the top
// candidate when the cursor moves to another cycle).
export function LogitsDetail({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const logits = latestOfType(events, cursor, 'logits')
  const softmax = latestOfType(events, cursor, 'softmax')
  const embed = latestOfType(events, cursor, 'embed')
  const runStart = latestOfType(events, cursor, 'run-start')
  const [pick, setPick] = useState<{ cycle: number; index: number } | null>(null)
  const geo = useGeometry()
  if (!logits) return null
  const showProbs = softmax != null && softmax.cycle === logits.cycle
  const dims = embed?.dims ?? logits.lastHidden?.length ?? 576
  const vocabSize = runStart?.vocabSize
  const selected = pick && pick.cycle === logits.cycle && pick.index < logits.topK.length ? pick.index : 0
  const covered = geo.status === 'ready' && geo.asset != null && runStart != null
    && coversModel(geo.asset.manifest, runStart.modelId)
  const asset = covered ? geo.asset : undefined
  const assetRowFor = (id: number) =>
    asset && id >= 0 && id < asset.manifest.vocabSize ? asset.vector(id) : undefined
  const assetNote = geo.status === 'loading' ? 'loading vocabulary geometry…' : 'vector values unavailable'
  return (
    <div data-testid="detail-logits" className="detail">
      <h3>{showProbs ? `Softmax (T=${softmax!.temperature})` : 'Logits'}</h3>
      <p>{showProbs
        ? 'Softmax turns scores into probabilities that sum to 100%.'
        : 'Each score is a dot product: the last token’s final vector against one row of E — the same matrix the Embeddings stage looked rows up in.'}</p>
      {vocabSize && (
        <p data-testid="logits-formula" className="logits-formula">
          logits = x_last · Eᵀ&emsp;[1×{dims}] · [{dims}×{thousands(vocabSize)}] → [{thousands(vocabSize)}]
        </p>
      )}
      <LogitsReadout candidates={logits.topK} probs={showProbs ? softmax!.topK.map((c) => c.prob) : undefined}
        lastHidden={logits.lastHidden} dims={dims} vocabSize={vocabSize}
        selected={selected} onSelect={(index) => setPick({ cycle: logits.cycle, index })}
        assetRowFor={assetRowFor} assetNote={assetNote} />
    </div>
  )
}
```

`src/viz/DetailPanel.tsx` — replace the `case 'logits'` block:

```tsx
    case 'logits':
      return latestOfType(events, cursor, 'logits') ? <LogitsDetail events={events} cursor={cursor} /> : empty
```

Note the existing test `logits detail frames scores as a dot-product readout` expects the formula text `[576×49 152]`; the formula above keeps that shape string. The formula now shows in the softmax state too (it was hidden before); `softmax cursor switches bars to probabilities` only checks for `70`, so it still passes.

- [ ] **Step 8: Add the CSS**

Append to `src/index.css` after the Embeddings card block:

```css
/* ── Logits card ─────────────────────────────────────────────────────── */
.logits-readout { margin: .6rem 0 .4rem; }
.readout-equation { display: grid; grid-template-columns: auto auto auto auto minmax(16rem, 1fr);
  gap: .6rem 1rem; align-items: start; }
.readout-op { font-size: 1.4rem; color: var(--ai-deep); align-self: center; padding-top: 1rem; }
.readout-x-missing { font-family: var(--mono); font-size: .78rem; color: var(--ink-faint);
  border: 1px dashed var(--ai-soft); border-radius: 3px; padding: .25rem .5rem; display: inline-block; }
.readout-row-mark { opacity: .9; }
.hue-fill-0 { fill: var(--dye-0); } .hue-fill-1 { fill: var(--dye-1); } .hue-fill-2 { fill: var(--dye-2); }
.hue-fill-3 { fill: var(--dye-3); } .hue-fill-4 { fill: var(--dye-4); } .hue-fill-5 { fill: var(--dye-5); }
.readout-scores { display: flex; flex-direction: column; row-gap: .3rem; }
.readout-candidate { display: grid; grid-template-columns: 5.5rem 192px 160px 3rem; align-items: center; gap: .5rem;
  font: inherit; background: none; border: 1px solid transparent; border-radius: 6px; padding: .1rem .3rem; cursor: pointer; text-align: left; }
.readout-candidate[data-selected="true"] { border-color: var(--shu); box-shadow: 0 0 0 2px var(--shu-wash); }
.readout-candidate .token-chip { justify-self: start; }
.readout-nostrip { display: inline-block; width: 192px; height: 10px; border-bottom: 1px dotted var(--ai-soft); }
.readout-bench { border-top: 1px solid var(--line); margin-top: .8rem; padding-top: .6rem; }
.readout-bench-rows { display: grid; grid-template-columns: 4.5rem auto; gap: .35rem .6rem; align-items: center; }
.readout-bench-label { font-family: var(--mono); font-size: .72rem; color: var(--ai-deep); }
.readout-sum { font-family: var(--mono); font-size: .8rem; color: var(--ai-deep); margin-top: .4rem; }
@media (max-width: 720px) { .readout-equation { grid-template-columns: 1fr; } .readout-op { display: none; } }
```

- [ ] **Step 9: Run the whole suite and type-check**

Run: `npx vitest run` then `npx tsc -b` then `npx oxlint`
Expected: all PASS; no type errors; oxlint reports only the 2 pre-existing warnings.

- [ ] **Step 10: In-app check**

Dev server, real mode, the farmer/horse or cat example; click the Logits stage card. Expected: x strip populated, product strips on every candidate, bench sum equal to the top logit to one decimal; clicking another candidate moves the bench; stepping to the softmax event shows percentages. Switch to sim mode and run an example: placeholder x, PCA rows, one-liner. Record both in the ledger.

- [ ] **Step 11: Commit**

```bash
/usr/bin/git add src/viz/LogitsReadout.tsx src/viz/LogitsReadout.test.tsx src/viz/details/LogitsDetail.tsx src/viz/details/LogitsDetail.test.tsx src/viz/DetailPanel.tsx src/index.css
/usr/bin/git commit -m "feat(viz): Logits readout — x_last · Eᵀ equation row and the dot-product bench"
```

---

### Task 10: Screenshot scene, README, backlog

**Files:**
- Modify: `scripts/capture-screenshots.mjs:38-42,64-66,118-144`
- Modify: `README.md` (new section after "What an embedding is", Docs list)
- Modify: `docs/BACKLOG.md`
- Create: `docs/screenshots/logits-readout.png` (captured by the script)

- [ ] **Step 1: Add the `logits` scene**

In `scripts/capture-screenshots.mjs`: extend the comment on line 38 with `logits`; change the real-mode gate to
`if (want('attention') || want('compare') || want('embeddings') || want('logits'))`; append after the embeddings scene:

```js
  // ---- logits-readout.png: real mode, Logits stage — equation row + bench
  if (want('logits')) {
    await $('prompt-input').fill('The farmer bought a horse and rode it home.')
    await $('maxtok-input').fill('6')
    const before = await $('run-chip').count()
    await $('btn-generate').click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })   // wait for the run to SEAL
    await $('btn-live').click()
    await $('stage-card').nth(3).click()                            // Logits
    await $('readout-bench').waitFor({ timeout: 10000 })
    await $('readout-candidate').nth(1).click()                     // second candidate on the bench
    const card = await $('detail-logits').boundingBox()
    await $('detail-logits').scrollIntoViewIfNeeded()
    await page.waitForTimeout(300)
    const box = await $('detail-logits').boundingBox()
    const pad = 12
    await page.screenshot({ path: `${DIR}/logits-readout.png`, clip: {
      x: box.x - pad, y: box.y - pad, width: card.width + 2 * pad, height: box.height + 2 * pad,
    } })
    await $('maxtok-input').fill('20')
    console.log('logits-readout.png')
  }
```

- [ ] **Step 2: Capture**

Run: `npm run screenshots -- logits`
Expected: `docs/screenshots/logits-readout.png` written; open it and confirm the bench sum matches the logit on the selected bar. If the headed run cannot reach the Hub, record that and capture later.

- [ ] **Step 3: README section**

Insert before `## Reading the attention heatmaps`:

```markdown
## The readout — a dot product you can see

![The Logits card: x_last against Eᵀ, the candidate bars, and the dot-product bench](docs/screenshots/logits-readout.png)

The Logits card is the Embeddings card read in reverse. At the Embeddings
stage a token id picked one row of `E`; here the last token's final vector
`x_last [1×576]` is scored against *every* row of the same matrix, one dot
product each, and the bars are the ten highest. Tied weights are not a
footnote: the model learns one vocabulary geometry and uses it at both ends.

Click a candidate to put its score on the bench: `x_last`, that token's row
of `E`, and their element-wise product, each pooled into 64 cells of 9
dimensions. Blue cells push the score up, red cells pull it down, and the
sum under the product strip is the logit on the bar. In real mode every
number is exact: the custom export exposes the final hidden state and the
candidates' rows (`last_hidden_state`, `topk_ids`, `topk_embeds`; see
`tools/export`). Simulated runs and archived runs from older versions show the
structure with PCA-reduced rows and no products.
```

In the Docs list add:

```markdown
- [`specs/2026-10-09-logits-readout-design.md`](docs/superpowers/specs/2026-10-09-logits-readout-design.md)
- [`plans/2026-10-09-logits-readout.md`](docs/superpowers/plans/2026-10-09-logits-readout.md)
```

- [ ] **Step 4: Backlog**

In `docs/BACKLOG.md`, under "Pipeline pedagogy" after #12:

```markdown
- [x] **#13 Logits readout** — the Logits card shows the readout as the
  Embeddings lookup in reverse and, in real mode, the dot product itself:
  final vector, candidate row of E, element-wise product summing to the
  logit. Needed three new export outputs (`last_hidden_state`, `topk_ids`,
  `topk_embeds`) and a pinned Hub tag (`v3`). *(2026-10-09)*
```

and in "Done (for the record)" at the top:

```markdown
- [x] Logits readout: final vector, candidate rows, visible dot product; export v3 *(#13, 2026-10-09)*
```

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add scripts/capture-screenshots.mjs README.md docs/BACKLOG.md docs/screenshots/logits-readout.png
/usr/bin/git commit -m "docs: the readout section, logits screenshot scene, backlog #13"
```

---

## Self-review notes

- Spec coverage: Component 1 → Tasks 1–3 (and the operator run, Task 4); Component 2 → Task 5; Component 3 → Tasks 6–7; Component 4 → Tasks 8–9; error handling → Tasks 7 and 9 tests; testing section → each task; screenshots/README/backlog → Task 10. The spec's `tests/test_quantize_exclusions.py` is folded into `test_validate_readout.py`.
- Type consistency: `LogitCandidate.row/dot`, `lastHidden`, `READOUT_CELLS` are defined in Task 6 and consumed by Tasks 7 and 9 under the same names; `extractReadout`'s `byId` map is consumed in Task 7's worker wiring; `VectorStrip` props from Task 8 match their use in Task 9.
- Review Focus items 1–5 are pinned to tests in Tasks 6, 7 and 9 as listed.
