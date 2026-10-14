# Labelled MLP Neurons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A third card under the two MLP cards, "Inside one MLP", that splits one layer's MLP vote for a candidate into one vote per neuron and labels each neuron with the tokens it pushes toward.

**Architecture:** Export v5 adds two outputs to the ONNX graph: the last token's neuron activations and the final norm's 1/rms. A new `neurons/` asset on the Hub holds the down-projection columns as int8 with per-neuron labels and thresholds. The worker packs the activations into the cycle's `edits` event as base64. The main thread fetches one layer file at a time through the Cache API and computes votes, edit sizes and badges on demand.

**Tech Stack:** Python export tool (onnx, onnxruntime, numpy, torch for reference values, pytest), React 19 + TypeScript, zustand, vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-14-labelled-mlp-neurons-design.md`

## Global Constraints

- Card title: "Inside one MLP: layer N". Modes: "behind the vote" (default) and "firing most". Top 8 neurons.
- Card intro: "The MLP’s edit is the sum of 1,536 neuron edits, so its vote splits into one vote per neuron. Click a layer in either card above to open it here."
- Firing-most intro: "The neurons that changed the vector most for the last token, whatever they voted for. Size = |activation| × length of the neuron’s column."
- Notes, verbatim:
  - simulated run: "Real model only: simulated runs have no neurons to show."
  - run without neuron data: "This run was recorded before neuron data existed."
  - viewed cycle not finished: "Votes appear once this cycle’s scores are computed." (the votes card's pending note)
  - layer file loading: "Loading the MLP weights (about 0.9 MB per layer, once per model version)…"
  - layer file failed: "Couldn’t load the MLP weights." plus a Retry button
  - total outside tolerance: "The 8-bit weights are slightly off for this layer, so the neuron votes don’t add up exactly to the votes card."
- Badges: `top 0.1%`, `top 1%`, `top 10%`, `not in top 10%`, with " (away)" appended when the vote is negative.
- Exactness tolerance (same as #15): `0.05 + 0.01 · |votes card value|`.
- Labels: cosine between ±column and the rows of g ⊙ E, computed from the stock fp32 weights, both signs, 5 tokens per sign.
- Asset path: `neurons/` in `saigyo-hoshi/smollm2-135m-attn-onnx`, tag **v5**. Cache name `tsumugi-neurons`.
- Never fail a run because of neuron data. On bad outputs, log one `console.warn` and stop including it.
- Token text in the UI always goes through `visibleToken` (whitespace shown as ␣ and ↵).

## Plan-level decisions (where the spec left room)

1. **The neuron data is an optional `neurons` field on the `edits` event, not a separate event.** The spec asks for data that "gets no playback time of its own and appears together with the edits step". Every event is a player step, so a separate event would add an empty step. A field on `edits` satisfies that requirement exactly. The worker emits `edits` after computing the cycle's top candidates, still before `logits`.
2. **Candidate rows are keyed by id and stored as base64 float32:** `candidates: [{ id, row }]`. Rows are about 31 KB per cycle instead of about 100 KB as JSON numbers, and a candidate's row can't be paired with the wrong id.
3. **The asset ships its own `tokens.json`** (the decoded text of every id, about 450 KB, covered by the manifest's checksums). The labels need texts, and the geometry asset would mean an extra 5 MB uncached download.
4. **"Away from" shows the opposite sign's list.** For a neuron whose vote is negative, the label column shows an "away from" chip followed by the top 5 tokens of the *opposite* sign. This is what the approved mockup shows (N600: "away from: and, ␣And, ␣and"), and the badge uses the same list.
5. **Activation parity tolerances.** The spec says "clipped tolerance as in #15", but #15's clip applied to ratios. For activations the fp32 gate is `max|Δ| ≤ 1e-3 + 1e-3·max|ref|` per layer and the q8 gate is `max|Δ| ≤ 0.05 + 0.1·max|ref|` per layer. For 1/rms: fp32 relative 1e-4, q8 relative 2e-2.
6. **Archived runs use the run's own mode.** `App` passes the active trace's `run-start` mode to `DetailPanel`. Today an archived real run viewed in sim mode shows "simulated runs have no MLP", which is wrong for the #15 cards too. The Playwright test depends on this fix.

## Review Focus

1. **A layer file cached in a broken state** (truncated download): the loader must delete the cached entry so that Retry fetches it again. Test in Task 7.
2. **An imported run whose neuron data doesn't match the asset** (different neuron count): the card shows a note and doesn't throw. Test in Task 10.
3. **A candidate with no row in the neuron data** (an edited or old import): the card shows a note for that candidate and doesn't throw. Test in Task 10.
4. **Fast stepping through layers:** a slow load for an earlier layer must not overwrite the current one. Test in Task 7 (the hook ignores stale results).
5. **Badge sign for negative activations:** a neuron with a negative activation and a positive vote uses the negative-sign list and thresholds. Test in Task 8.

---

### Task 1: Export — neuron outputs in the graph

**Files:**
- Modify: `tools/export/src/tsumugi_export/postprocess.py`
- Test: `tools/export/tests/test_postprocess_mlp.py`

**Interfaces:**
- Produces: graph outputs `mlp_activations` `[batch, layers, neurons]` and `final_norm_inv_rms` `[batch, 1, 1]`; Python names `MLP_ACTIVATIONS_OUTPUT`, `INV_RMS_OUTPUT`, `find_mlp_inputs(graph) -> list[str]`, `expose_neurons(model) -> model`.

- [ ] **Step 1: Write the failing tests** (append to `tools/export/tests/test_postprocess_mlp.py`; extend its import line with `INV_RMS_OUTPUT, MLP_ACTIVATIONS_OUTPUT, expose_neurons, find_mlp_inputs`)

```python
def test_find_mlp_inputs_names_each_layers_down_proj_input():
    assert find_mlp_inputs(layered_graph().graph) == [
        f"/model/layers.{l}/post_attention_layernorm/Cast_output_0" for l in range(LAYERS)]


def test_neuron_outputs_are_the_last_positions_down_proj_inputs_and_inverse_rms():
    model = expose_neurons(layered_graph())
    onnx.checker.check_model(model)
    ids = np.array([[3, 1, 4]], dtype=np.int64)
    out = _run(model, ids)
    _, _, _, _, before_m, _, inv = reference(ids)
    acts = out[MLP_ACTIVATIONS_OUTPUT]
    assert acts.shape == (1, LAYERS, DIMS)
    for l in range(LAYERS):
        np.testing.assert_allclose(acts[0, l], before_m[l][-1], rtol=1e-5, atol=1e-6)
    assert out[INV_RMS_OUTPUT].shape == (1, 1, 1)
    np.testing.assert_allclose(out[INV_RMS_OUTPUT].reshape(-1), inv[-1], rtol=1e-5)
    declared = next(o for o in model.graph.output if o.name == MLP_ACTIVATIONS_OUTPUT)
    assert [d.dim_value for d in declared.type.tensor_type.shape.dim][1:] == [LAYERS, DIMS]
```

Also change the last assertion of `test_postprocess_graph_adds_the_mlp_outputs_after_the_readout` to:

```python
    assert names[-5:] == [EDIT_RATIOS_OUTPUT, STREAM_PARTS_OUTPUT, MLP_ACTIVATIONS_OUTPUT, INV_RMS_OUTPUT, "inputs_embeds"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_postprocess_mlp.py -q`
Expected: collection error, `ImportError: cannot import name 'INV_RMS_OUTPUT'`.

- [ ] **Step 3: Implement** (in `postprocess.py`, after `expose_mlp_internals`; also add one sentence to the module docstring: "We also add `mlp_activations` and `final_norm_inv_rms` for the neuron card (see expose_neurons).")

```python
MLP_ACTIVATIONS_OUTPUT = "mlp_activations"
INV_RMS_OUTPUT = "final_norm_inv_rms"


def find_mlp_inputs(graph: onnx.GraphProto) -> list[str]:
    """Per layer, the down_proj MatMul's first input: the neuron activations
    SiLU(gate) ⊙ up that the down projection turns into the MLP's edit."""
    by_name = {n.name: n for n in graph.node}
    inputs = []
    while (node := by_name.get(MLP_MATMUL.format(l=len(inputs)))) is not None:
        inputs.append(node.input[0])
    if not inputs:
        raise ValueError(f"no {MLP_MATMUL.format(l=0)} — is this the SmolLM2 export?")
    return inputs


def _neuron_count(graph: onnx.GraphProto) -> int:
    """Rows of layer 0's down_proj weight ([neurons, dims] in the MatMul)."""
    mm = next(n for n in graph.node if n.name == MLP_MATMUL.format(l=0))
    init = next((t for t in graph.initializer if t.name == mm.input[1]), None)
    if init is None or len(init.dims) != 2:
        raise ValueError(f"{MLP_MATMUL.format(l=0)} weight is not a 2-D initializer")
    return int(init.dims[0])


def expose_neurons(model: onnx.ModelProto) -> onnx.ModelProto:
    """mlp_activations [batch, layers, neurons]: the last position's down_proj
    inputs per layer. final_norm_inv_rms [batch, 1, 1]: the final norm's 1/rms
    at the last position. Tapped on the fp32 graph, before quantization, like
    the other added outputs."""
    graph = model.graph
    inputs = find_mlp_inputs(graph)
    _, inv = find_final_norm(graph)
    neurons = _neuron_count(graph)
    graph.initializer.extend([
        helper.make_tensor("tsumugi_neuron_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_neuron_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_neuron_slice_axes", TensorProto.INT64, [1], [1]),
    ])
    slice_args = ["tsumugi_neuron_slice_starts", "tsumugi_neuron_slice_ends", "tsumugi_neuron_slice_axes"]
    nodes = [helper.make_node("Slice", [name, *slice_args], [f"tsumugi_neurons{l}"], name=f"tsumugi_neurons{l}")
             for l, name in enumerate(inputs)]
    nodes += [
        helper.make_node("Concat", [f"tsumugi_neurons{l}" for l in range(len(inputs))], [MLP_ACTIVATIONS_OUTPUT],
                         name="tsumugi_mlp_activations", axis=1),
        helper.make_node("Slice", [inv, *slice_args], [INV_RMS_OUTPUT], name="tsumugi_final_norm_inv_rms"),
    ]
    graph.node.extend(nodes)
    graph.output.extend([
        helper.make_tensor_value_info(MLP_ACTIVATIONS_OUTPUT, TensorProto.FLOAT, ["batch_size", len(inputs), neurons]),
        helper.make_tensor_value_info(INV_RMS_OUTPUT, TensorProto.FLOAT, ["batch_size", 1, 1]),
    ])
    return model
```

Change `postprocess_graph` to:

```python
def postprocess_graph(model: onnx.ModelProto, k: int = TOPK) -> onnx.ModelProto:
    """Readout, MLP internals and neuron outputs (all derived from graph
    structure, independent of the output list), then the inputs_embeds
    rename, which drops every hidden_states.* output."""
    return expose_inputs_embeds(expose_neurons(expose_mlp_internals(expose_readout(model, k))))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools/export && uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/export/src/tsumugi_export/postprocess.py tools/export/tests/test_postprocess_mlp.py
git commit -m "feat(export): mlp_activations and final_norm_inv_rms outputs for the neuron card"
```

---

### Task 2: Export — the `neurons/` asset

**Files:**
- Create: `tools/export/src/tsumugi_export/neurons.py`
- Modify: `tools/export/src/tsumugi_export/stock.py`, `tools/export/src/tsumugi_export/__main__.py`
- Test: `tools/export/tests/test_neurons.py`

**Interfaces:**
- Produces: `layer_layout(neurons, dims, labels=5, fractions=(0.001, 0.01, 0.1)) -> dict` with keys `scales, columns, labelIds, thresholds, bytes`; `build_layer(W, g, E) -> bytes`; `build_neurons(W_layers, g, E, texts) -> {"manifest", "layers": {name: bytes}, "tokens": bytes}`; `write_neurons(out_dir, built)`; CLI `neurons --model-dir`.
- Layer file layout, little-endian, one file per layer (`layer-00.bin` … `layer-29.bin`):
  - `scales`: float32 [neurons]
  - `columns`: int8 [neurons × dims], with column i at offset i·dims
  - `labelIds`: uint16 [neurons × 2 × 5] (sign 0 is a positive activation, 1 a negative one)
  - `thresholds`: float32 [neurons × 2 × 3], the cosine at ranks ⌈0.001·V⌉, ⌈0.01·V⌉ and ⌈0.1·V⌉
- Manifest keys: `modelId, layers, neurons, dims, vocabSize, labelsPerSign, fractions, labelMethod, layout, finalNormWeight, files` (`files[name] = {bytes, sha256}`, including `tokens.json`).

- [ ] **Step 1: Write the failing tests** (`tools/export/tests/test_neurons.py`)

```python
import hashlib
import json

import numpy as np

from tsumugi_export.neurons import build_layer, build_neurons, layer_layout, write_neurons


def decode(raw: bytes, neurons: int, dims: int, labels: int = 5, fractions: int = 3):
    lay = layer_layout(neurons, dims, labels, tuple([0.1] * fractions))
    scales = np.frombuffer(raw[lay["scales"]:lay["columns"]], dtype="<f4")
    q = np.frombuffer(raw[lay["columns"]:lay["labelIds"]], dtype=np.int8).reshape(neurons, dims)
    ids = np.frombuffer(raw[lay["labelIds"]:lay["thresholds"]], dtype="<u2").reshape(neurons, 2, labels)
    thr = np.frombuffer(raw[lay["thresholds"]:lay["bytes"]], dtype="<f4").reshape(neurons, 2, fractions)
    return scales, q, ids, thr


def test_layout_offsets_follow_each_other_and_stay_aligned():
    lay = layer_layout(1536, 576)
    assert lay == {"scales": 0, "columns": 6144, "labelIds": 890880, "thresholds": 921600, "bytes": 958464}
    assert lay["labelIds"] % 2 == 0 and lay["thresholds"] % 4 == 0


def test_columns_survive_the_int8_round_trip_within_half_a_step():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 6)).astype(np.float32)          # [dims, neurons]
    E = rng.normal(size=(40, 8)).astype(np.float32)
    scales, q, _, _ = decode(build_layer(W, np.ones(8, np.float32), E), 6, 8)
    np.testing.assert_allclose(q * scales[:, None], W.T, atol=float(scales.max()) / 2 + 1e-7)


def test_labels_are_the_closest_rows_for_each_sign_and_thresholds_rank_cosines():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(40, 4)).astype(np.float32)
    E[3] = [1, 0, 0, 0]
    E[5] = [-2, 0, 0, 0]                                     # the antipode of row 3
    W = np.zeros((4, 2), np.float32)
    W[:, 0] = [0.5, 0, 0, 0]
    W[:, 1] = rng.normal(size=4)
    _, _, ids, thr = decode(build_layer(W, np.ones(4, np.float32), E), 2, 4)
    assert ids[0, 0, 0] == 3 and ids[0, 1, 0] == 5
    assert thr[0, 0, 0] == np.float32(1.0)                   # rank ⌈0.001·40⌉ = 1: the best cosine
    R = E / np.linalg.norm(E, axis=1, keepdims=True)
    cos = R @ (W[:, 1] / np.linalg.norm(W[:, 1]))
    np.testing.assert_allclose(thr[1, 0], [np.sort(cos)[::-1][r - 1] for r in (1, 1, 4)], rtol=1e-5)
    np.testing.assert_allclose(thr[1, 1], [np.sort(-cos)[::-1][r - 1] for r in (1, 1, 4)], rtol=1e-5)


def test_labels_compare_with_the_rows_scaled_by_the_final_norm_weight():
    E = np.array([[1, 1, 0, 0], [1, -1, 0, 0]] + [[0, 0, 1, 0]] * 3, dtype=np.float32)
    W = np.array([[1], [0.2], [0], [0]], dtype=np.float32)
    _, _, ids, _ = decode(build_layer(W, np.ones(4, np.float32), E), 1, 4)
    assert ids[0, 0, 0] == 0
    _, _, ids, _ = decode(build_layer(W, np.array([1, -1, 1, 1], np.float32), E), 1, 4)
    assert ids[0, 0, 0] == 1


def test_manifest_lists_every_file_with_its_size_and_checksum(tmp_path):
    rng = np.random.default_rng(2)
    W = [rng.normal(size=(4, 3)).astype(np.float32) for _ in range(2)]
    E = rng.normal(size=(30, 4)).astype(np.float32)
    g = rng.uniform(0.5, 1.5, 4).astype(np.float32)
    built = build_neurons(W, g, E, [f"t{i}" for i in range(30)])
    write_neurons(tmp_path, built)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert (m["layers"], m["neurons"], m["dims"], m["vocabSize"]) == (2, 3, 4, 30)
    assert m["finalNormWeight"] == [float(v) for v in g]
    assert set(m["files"]) == {"layer-00.bin", "layer-01.bin", "tokens.json"}
    for name, f in m["files"].items():
        data = (tmp_path / name).read_bytes()
        assert len(data) == f["bytes"] and hashlib.sha256(data).hexdigest() == f["sha256"]
    assert m["files"]["layer-00.bin"]["bytes"] == m["layout"]["bytes"]
    assert json.loads((tmp_path / "tokens.json").read_text()) == [f"t{i}" for i in range(30)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_neurons.py -q`
Expected: `ModuleNotFoundError: No module named 'tsumugi_export.neurons'`.

- [ ] **Step 3: Implement `neurons.py`**

```python
"""Neuron asset for the "Inside one MLP" card (#16 spec, section 2): per layer
the down-projection columns as int8 with one scale per column, and per neuron
and sign the five tokens its column points at most plus three cosine
thresholds; the decoded text of every id; a manifest with the layout, the
final norm weight and checksums."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tsumugi_export import STOCK_MODEL_ID

LABELS = 5
FRACTIONS = (0.001, 0.01, 0.1)
LABEL_METHOD = "cosine between ±W_down[:, i] and g ⊙ E[t] over the whole vocabulary, fp32 weights"


def layer_layout(neurons: int, dims: int, labels: int = LABELS, fractions: tuple = FRACTIONS) -> dict:
    columns = neurons * 4
    label_ids = columns + neurons * dims
    thresholds = label_ids + neurons * 2 * labels * 2
    return {"scales": 0, "columns": columns, "labelIds": label_ids, "thresholds": thresholds,
            "bytes": thresholds + neurons * 2 * len(fractions) * 4}


def build_layer(W: np.ndarray, g: np.ndarray, E: np.ndarray, labels: int = LABELS,
                fractions: tuple = FRACTIONS, block: int = 256) -> bytes:
    """W is down_proj.weight [dims, neurons]: column i is what neuron i writes,
    scaled by its activation."""
    vocab = E.shape[0]
    if vocab > 65535:
        raise ValueError("label ids are uint16")
    cols = np.ascontiguousarray(W.T, dtype=np.float32)                     # [neurons, dims]
    neurons = cols.shape[0]
    scales = np.abs(cols).max(axis=1) / 127
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(cols / scales[:, None]), -127, 127).astype(np.int8)
    R = (E * g[None, :]).astype(np.float32)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    U = cols / np.maximum(np.linalg.norm(cols, axis=1, keepdims=True), 1e-12)
    ranks = [max(1, int(np.ceil(f * vocab))) for f in fractions]
    ids = np.empty((neurons, 2, labels), dtype="<u2")
    thr = np.empty((neurons, 2, len(fractions)), dtype="<f4")
    for start in range(0, neurons, block):                                 # [block, vocab] cosines at a time
        S = U[start:start + block] @ R.T
        for s, M in enumerate((S, -S)):
            top = np.argpartition(-M, labels - 1, axis=1)[:, :labels]
            order = np.argsort(-np.take_along_axis(M, top, axis=1), axis=1, kind="stable")
            ids[start:start + block, s] = np.take_along_axis(top, order, axis=1)
            part = -np.partition(-M, [r - 1 for r in ranks], axis=1)
            thr[start:start + block, s] = part[:, [r - 1 for r in ranks]]
    return scales.astype("<f4").tobytes() + q.tobytes() + ids.tobytes() + thr.tobytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_neurons(W_layers: list[np.ndarray], g: np.ndarray, E: np.ndarray, texts: list[str]) -> dict:
    dims, neurons = W_layers[0].shape
    vocab = E.shape[0]
    if len(texts) != vocab:
        raise ValueError(f"{len(texts)} texts for a vocabulary of {vocab}")
    layout = layer_layout(neurons, dims)
    layers = {f"layer-{l:02d}.bin": build_layer(W, g, E) for l, W in enumerate(W_layers)}
    for name, data in layers.items():
        if len(data) != layout["bytes"]:
            raise ValueError(f"{name}: {len(data)} bytes, layout says {layout['bytes']}")
    tokens = json.dumps(texts, ensure_ascii=False).encode("utf-8")
    files = {name: {"bytes": len(data), "sha256": _sha256(data)} for name, data in layers.items()}
    files["tokens.json"] = {"bytes": len(tokens), "sha256": _sha256(tokens)}
    manifest = {
        "modelId": STOCK_MODEL_ID, "layers": len(W_layers), "neurons": int(neurons), "dims": int(dims),
        "vocabSize": int(vocab), "labelsPerSign": LABELS, "fractions": list(FRACTIONS),
        "labelMethod": LABEL_METHOD, "layout": layout,
        "finalNormWeight": [float(v) for v in g], "files": files,
    }
    return {"manifest": manifest, "layers": layers, "tokens": tokens}


def write_neurons(out_dir: Path, built: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(built["manifest"], indent=2))
    for name, data in built["layers"].items():
        (out_dir / name).write_bytes(data)
    (out_dir / "tokens.json").write_bytes(built["tokens"])


def run(args) -> int:
    from tsumugi_export.stock import stock_embeddings, stock_final_norm, stock_mlp_down, stock_token_texts
    E = stock_embeddings()
    out = Path(args.model_dir) / "neurons"
    write_neurons(out, build_neurons(stock_mlp_down(), stock_final_norm(), E, stock_token_texts(E.shape[0])))
    m = json.loads((out / "manifest.json").read_text())
    print(f"wrote neurons → {out} ({m['layers']} layers × {m['neurons']} neurons, {m['layout']['bytes']} bytes each)")
    return 0
```

In `stock.py`, add:

```python
def _stock_state() -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    return load_file(hf_hub_download(STOCK_MODEL_ID, "model.safetensors"))


def stock_mlp_down() -> list[np.ndarray]:
    """Every layer's down_proj weight, fp32 [dims, neurons], in layer order."""
    state = _stock_state()
    out, l = [], 0
    while (key := f"model.layers.{l}.mlp.down_proj.weight") in state:
        out.append(state[key].float().numpy())
        l += 1
    return out


def stock_final_norm() -> np.ndarray:
    """The final RMSNorm's weight g, fp32 [dims]."""
    return _stock_state()["model.norm.weight"].float().numpy()
```

In `__main__.py`, after the `geometry` parser:

```python
    p_neu = sub.add_parser("neurons", help="build the neuron asset (neurons/) for the neuron card")
    p_neu.add_argument("--model-dir", default="out/model")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools/export && uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/export/src/tsumugi_export/neurons.py tools/export/src/tsumugi_export/stock.py tools/export/src/tsumugi_export/__main__.py tools/export/tests/test_neurons.py
git commit -m "feat(export): neurons/ asset — int8 down_proj columns, labels and thresholds per neuron and sign"
```

---

### Task 3: Export — validation checks, report and model card

**Files:**
- Modify: `tools/export/src/tsumugi_export/validate.py`, `tools/export/src/tsumugi_export/publish.py`, `tools/export/README.md`
- Test: `tools/export/tests/test_validate_neurons.py`

**Interfaces:**
- Consumes: Task 1 output names; Task 2 layout and manifest.
- Produces: `check_activations_parity(outputs, ref, abs_tol, rel_tol)`, `check_inv_rms_parity(outputs, ref_inv, rtol)`, `check_neuron_votes_sum(outputs, ndir, k=10)`, `check_neuron_files(model_dir, texts, g)`, `torch_neuron_reference(ids) -> (acts [layers, neurons], inv float)`.

- [ ] **Step 1: Write the failing tests** (`tools/export/tests/test_validate_neurons.py`)

```python
import numpy as np

from tsumugi_export.neurons import build_neurons, write_neurons
from tsumugi_export.validate import (
    check_activations_parity, check_inv_rms_parity, check_neuron_files, check_neuron_votes_sum,
)

rng = np.random.default_rng(3)
L, N, D, V, K = 2, 6, 8, 40, 4
E = rng.normal(size=(V, D)).astype(np.float32)
G = rng.uniform(0.5, 1.5, D).astype(np.float32)
W = [rng.normal(size=(D, N)).astype(np.float32) for _ in range(L)]
TEXTS = [f"t{i}" for i in range(V)]


def outputs() -> dict:
    """Model outputs consistent with W: MLP part l = W_l · a_l, scaled by g and 1/rms."""
    acts = rng.normal(size=(1, L, N)).astype(np.float32)
    inv = np.array([[[0.25]]], dtype=np.float32)
    parts = rng.normal(size=(1, 1 + 2 * L, D)).astype(np.float32)
    for l in range(L):
        parts[0, 1 + L + l] = (W[l] @ acts[0, l]) * G * inv[0, 0, 0]
    ids = np.arange(K, dtype=np.int64)
    return {"mlp_activations": acts, "final_norm_inv_rms": inv, "stream_parts": parts,
            "topk_ids": ids[None, :], "topk_embeds": E[ids][None, :, :]}


def test_activations_parity_passes_within_tolerance_and_names_the_layer_when_not():
    out = outputs()
    ref = out["mlp_activations"][0].copy()
    assert check_activations_parity(out, ref, 1e-3, 1e-3)["passed"]
    ref[1, 2] += 1.0
    r = check_activations_parity(out, ref, 1e-3, 1e-3)
    assert not r["passed"] and "layer 1" in r["detail"]


def test_activations_parity_reports_a_missing_output():
    out = outputs()
    del out["mlp_activations"]
    assert "no mlp_activations" in check_activations_parity(out, np.zeros((L, N)), 1e-3, 1e-3)["detail"]


def test_inv_rms_parity():
    out = outputs()
    assert check_inv_rms_parity(out, 0.25, 1e-4)["passed"]
    assert not check_inv_rms_parity(out, 0.3, 1e-4)["passed"]


def test_neuron_votes_from_the_int8_asset_add_up_to_the_mlp_votes(tmp_path):
    write_neurons(tmp_path, build_neurons(W, G, E, TEXTS))
    out = outputs()
    assert check_neuron_votes_sum(out, tmp_path, k=K)["passed"]
    out["stream_parts"][0, 1 + L] *= 3
    r = check_neuron_votes_sum(out, tmp_path, k=K)
    assert not r["passed"] and "layer 0" in r["detail"]


def test_neuron_files_match_the_manifest(tmp_path):
    ndir = tmp_path / "neurons"
    write_neurons(ndir, build_neurons(W, G, E, TEXTS))
    assert check_neuron_files(tmp_path, TEXTS, G)["passed"]
    (ndir / "layer-01.bin").write_bytes(b"\0" * 10)
    assert "layer-01.bin" in check_neuron_files(tmp_path, TEXTS, G)["detail"]


def test_neuron_files_check_the_norm_weight_and_texts(tmp_path):
    write_neurons(tmp_path / "neurons", build_neurons(W, G, E, TEXTS))
    assert "finalNormWeight" in check_neuron_files(tmp_path, TEXTS, G * 2)["detail"]
    assert "tokens.json" in check_neuron_files(tmp_path, ["x"] * V, G)["detail"]


def test_a_missing_asset_fails():
    from pathlib import Path
    assert "no neurons/" in check_neuron_files(Path("/nonexistent"), TEXTS, G)["detail"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_validate_neurons.py -q`
Expected: `ImportError: cannot import name 'check_activations_parity'`.

- [ ] **Step 3: Implement** (in `validate.py`, after `check_edit_cache_parity`; add `import functools` at the top; extend the module docstring list with "11. mlp_activations and final_norm_inv_rms match PyTorch hooks; per-neuron votes from neurons/ add up to the MLP votes; neurons/ matches its manifest")

```python
def check_activations_parity(outputs: dict, ref: np.ndarray, abs_tol: float, rel_tol: float) -> dict:
    """mlp_activations [1, layers, neurons] against a reference [layers, neurons]:
    per layer max|Δ| ≤ abs_tol + rel_tol · max|ref of that layer|."""
    if "mlp_activations" not in outputs:
        return {"passed": False, "detail": "graph has no mlp_activations output"}
    got = outputs["mlp_activations"]
    if got.shape != (1, *ref.shape):
        return {"passed": False, "detail": f"mlp_activations shape {got.shape}, expected (1, {', '.join(map(str, ref.shape))})"}
    worst, worst_layer = 0.0, 0
    for l in range(ref.shape[0]):
        diff = float(np.max(np.abs(got[0, l] - ref[l])))
        allowed = abs_tol + rel_tol * float(np.max(np.abs(ref[l])))
        if diff > allowed:
            return {"passed": False, "detail": f"layer {l}: max|Δ|={diff:.3e} > {allowed:.3e}"}
        if diff > worst:
            worst, worst_layer = diff, l
    return {"passed": True, "detail": f"max|Δ|={worst:.2e} (layer {worst_layer}; atol {abs_tol} + {rel_tol}·max|ref|)"}


def check_inv_rms_parity(outputs: dict, ref_inv: float, rtol: float) -> dict:
    if "final_norm_inv_rms" not in outputs:
        return {"passed": False, "detail": "graph has no final_norm_inv_rms output"}
    got = float(outputs["final_norm_inv_rms"].reshape(-1)[0])
    rel = abs(got - ref_inv) / abs(ref_inv)
    return {"passed": rel <= rtol, "detail": f"1/rms {got:.6f} vs {ref_inv:.6f} (relative {rel:.2e}, rtol {rtol})"}


def check_neuron_votes_sum(outputs: dict, ndir: Path, k: int = 10) -> dict:
    """The card's identity: per layer and top-k candidate c, Σ_i a_i · (W_i · (g ⊙ E_c)) · 1/rms,
    with W_i dequantized from neurons/, equals the MLP vote stream_parts[1 + L + l] · E_c
    within the #15 tolerance 0.05 + 1 %."""
    for name in ("mlp_activations", "final_norm_inv_rms", "stream_parts", "topk_embeds"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    m = json.loads((ndir / "manifest.json").read_text())
    n, d, lay = m["neurons"], m["dims"], m["layout"]
    g = np.array(m["finalNormWeight"], dtype=np.float32)
    acts = outputs["mlp_activations"][0]
    layers = acts.shape[0]
    inv = float(outputs["final_norm_inv_rms"].reshape(-1)[0])
    rows = outputs["topk_embeds"][0][:k]
    parts = outputs["stream_parts"][0]
    worst = 0.0
    for l in range(layers):
        raw = (ndir / f"layer-{l:02d}.bin").read_bytes()
        scales = np.frombuffer(raw[lay["scales"]:lay["columns"]], dtype="<f4")
        q = np.frombuffer(raw[lay["columns"]:lay["labelIds"]], dtype=np.int8).reshape(n, d).astype(np.float32)
        cols = q * scales[:, None]                                         # [neurons, dims]
        for c, row in enumerate(rows):
            got = float(np.sum(acts[l] * (cols @ (g * row))) * inv)
            want = float(parts[1 + layers + l] @ row)
            gap = abs(got - want)
            if gap > 0.05 + 0.01 * abs(want):
                return {"passed": False, "detail": f"layer {l} candidate {c}: neuron votes sum to {got:.3f}, MLP vote is {want:.3f}"}
            worst = max(worst, gap)
    return {"passed": True, "detail": f"{layers} layers × {len(rows)} candidates, max gap {worst:.3e} (tol 0.05 + 1 %)"}


def check_neuron_files(model_dir: Path, texts: list[str], g: np.ndarray) -> dict:
    """neurons/ against its manifest (sizes, SHA-256), the layout, the stock
    final norm weight and the tokenizer's texts."""
    ndir = model_dir / "neurons"
    if not (ndir / "manifest.json").exists():
        return {"passed": False, "detail": "no neurons/ — run neurons first"}
    m = json.loads((ndir / "manifest.json").read_text())
    for name, f in m["files"].items():
        path = ndir / name
        if not path.exists() or path.stat().st_size != f["bytes"]:
            return {"passed": False, "detail": f"{name}: size differs from the manifest"}
        if _sha256(path) != f["sha256"]:
            return {"passed": False, "detail": f"{name}: SHA-256 differs from the manifest"}
    expected = {f"layer-{l:02d}.bin" for l in range(m["layers"])} | {"tokens.json"}
    if set(m["files"]) != expected:
        return {"passed": False, "detail": f"manifest lists {sorted(m['files'])}, expected {sorted(expected)}"}
    if any(m["files"][f"layer-{l:02d}.bin"]["bytes"] != m["layout"]["bytes"] for l in range(m["layers"])):
        return {"passed": False, "detail": "a layer file's size differs from layout.bytes"}
    if not np.allclose(np.array(m["finalNormWeight"]), g, atol=1e-6):
        return {"passed": False, "detail": "finalNormWeight differs from the stock final norm weight"}
    if json.loads((ndir / "tokens.json").read_text()) != texts:
        return {"passed": False, "detail": "tokens.json differs from the tokenizer's texts"}
    return {"passed": True, "detail": f"{m['layers']} layer files and tokens.json match the manifest"}


@functools.lru_cache(maxsize=1)
def _stock_torch_model():
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        STOCK_MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager").eval()


def torch_neuron_reference(ids: np.ndarray) -> tuple[np.ndarray, float]:
    """[layers, neurons] down_proj inputs at the last position and the final
    norm's 1/rms there, from forward pre-hooks on the stock PyTorch model."""
    import torch
    model = _stock_torch_model()
    rec: dict = {}
    hooks = [layer.mlp.down_proj.register_forward_pre_hook(
        lambda mod, args, l=l: rec.__setitem__(l, args[0][0, -1].detach()))
        for l, layer in enumerate(model.model.layers)]
    hooks.append(model.model.norm.register_forward_pre_hook(
        lambda mod, args: rec.__setitem__("norm", args[0][0, -1].detach())))
    with torch.no_grad():
        model(torch.tensor(ids))
    for h in hooks:
        h.remove()
    acts = np.stack([rec[l].numpy() for l in range(len(model.model.layers))]).astype(np.float32)
    x = rec["norm"].double()
    inv = float(1.0 / torch.sqrt((x ** 2).mean() + model.config.rms_norm_eps))
    return acts, inv
```

Also make `torch_edit_ratios` use `model = _stock_torch_model()` instead of loading its own copy (delete its `from_pretrained` lines; keep `import torch`).

In `run()`:
- next to `ref_ratios = torch_edit_ratios(ref_ids)`, add `ref_acts, ref_inv = torch_neuron_reference(ref_ids)`;
- after the `edit-cache-parity` line inside the variant loop, add:

```python
        act_abs, act_rel, inv_rtol = (1e-3, 1e-3, 1e-4) if fp32 else (0.05, 0.1, 2e-2)
        checks[f"{variant}:activations-parity"] = check_activations_parity(ro_out, ref_acts, act_abs, act_rel)
        checks[f"{variant}:inv-rms-parity"] = check_inv_rms_parity(ro_out, ref_inv, inv_rtol)
        if (model_dir / "neurons" / "manifest.json").exists():
            checks[f"{variant}:neuron-votes-sum"] = check_neuron_votes_sum(ro_out, model_dir / "neurons")
```

- after `checks["geometry"] = ...`, add:

```python
    from tsumugi_export.stock import stock_final_norm
    checks["neurons"] = check_neuron_files(model_dir, texts, stock_final_norm())
```

- in the report's `artifacts`, add the neurons files after geometry:

```python
                      *(sorted((model_dir / "neurons").glob("*")) if (model_dir / "neurons").exists() else []),
```

In `publish.py`, insert into `MODEL_CARD` after the MLP-cards paragraph:

```
For the neuron card it also exposes `mlp_activations` (`[batch, 30, 1536]`,
the last position's input of every layer's `down_proj`: the neuron
activations SiLU(gate) ⊙ up) and `final_norm_inv_rms` (`[batch, 1, 1]`, the
final norm's 1/rms at the last position). `neurons/` holds one file per layer
with the `down_proj` columns as int8 (one float32 scale per column), the five
tokens each column points at most for each sign (cosine against the rows of
the embedding table scaled by the final norm weight) and three cosine
thresholds (top 0.1 %, 1 % and 10 % of the vocabulary), plus `tokens.json`
and a manifest with the byte layout, the final norm weight and SHA-256
checksums.
```

and extend the Validation sentence's list with "activation parity with PyTorch hooks, per-neuron votes adding up to the MLP votes".

In `tools/export/README.md`, insert a new step after step 5 and renumber the following steps: "6. **Build the neuron asset:** `uv run python -m tsumugi_export neurons --model-dir out/model`". In the publish step, change the example tag to `v5`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools/export && uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/export/src/tsumugi_export/validate.py tools/export/src/tsumugi_export/publish.py tools/export/README.md tools/export/tests/test_validate_neurons.py
git commit -m "feat(export): validate neuron outputs and asset; model card documents them"
```

---

### Task 4: Build, validate and publish v5; point the app at it

**Files:**
- Modify: `src/engine/tokenizer.ts`
- Test: `src/engine/tokenizer.test.ts`

**Interfaces:**
- Produces: `ATTN_MODEL_REVISION = 'v5'`, `neuronsUrlFor(modelId, revision)`, `NEURONS_BASE_URL` (overridable with `VITE_NEURONS_BASE_URL`).

- [ ] **Step 1: Run the pipeline** (from `tools/export`; the v4 no-cache export in `out/model-nocache` still serves the A≡B check, since v5 only adds outputs)

```bash
uv run python -m tsumugi_export export --out out/model
uv run python -m tsumugi_export quantize --model-dir out/model
uv run python -m tsumugi_export geometry --model-dir out/model
uv run python -m tsumugi_export neurons --model-dir out/model
uv run python -m tsumugi_export validate --model-dir out/model
```

Expected: every validate row reads PASS, including `model_quantized.onnx:activations-parity`, `model_quantized.onnx:inv-rms-parity`, `model_quantized.onnx:neuron-votes-sum` and `neurons`. Note the `neuron-votes-sum` max gap for the q8 variant in the ledger.

If `neuron-votes-sum` fails for q8 only, record the gap. The card shows "≈" when outside tolerance, so this does not block publishing, but the check must then be informational for q8 like `logits-parity`: change its gate for quantized variants to `{"passed": True, "detail": "informational: ..."}` and ledger the ruling. Any other FAIL stops the task. Debug it with superpowers:systematic-debugging.

- [ ] **Step 2: Publish**

```bash
uv run python -m tsumugi_export publish --model-dir out/model --tag v5
```

Expected: `published → https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx (tag v5)`. Then verify that `https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx/resolve/v5/neurons/manifest.json` returns the manifest: `curl -sL <url> | head -c 300`.

- [ ] **Step 3: Write the failing test** (append to `src/engine/tokenizer.test.ts`; add `ATTN_MODEL_REVISION, neuronsUrlFor` to its import from `./tokenizer`)

```ts
test('the neuron asset lives next to the model at the pinned tag', () => {
  expect(ATTN_MODEL_REVISION).toBe('v5')
  expect(neuronsUrlFor('a/b', 'v5')).toBe('https://huggingface.co/a/b/resolve/v5/neurons')
})
```

- [ ] **Step 4: Run it to verify it fails**

Run: `npx vitest run src/engine/tokenizer.test.ts`
Expected: FAIL, `neuronsUrlFor is not a function` (or revision `v4`).

- [ ] **Step 5: Implement** in `src/engine/tokenizer.ts`: set `ATTN_MODEL_REVISION = 'v5'`, append to its comment "v5 = neuron outputs (mlp_activations, final_norm_inv_rms) and the neurons/ asset.", and after `GEOMETRY_BASE_URL` add:

```ts
export const neuronsUrlFor = (modelId: string, revision: string) =>
  `https://huggingface.co/${modelId}/resolve/${revision}/neurons`
export const NEURONS_BASE_URL: string =
  import.meta.env.VITE_NEURONS_BASE_URL ?? neuronsUrlFor(ATTN_MODEL_ID, ATTN_MODEL_REVISION)
```

- [ ] **Step 6: Run the tests**

Run: `npm test`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/engine/tokenizer.ts src/engine/tokenizer.test.ts
git commit -m "feat(model): load export v5 (neuron outputs and the neurons/ asset)"
```

---

### Task 5: Trace — packing, the `neurons` field, validation and import

**Files:**
- Create: `src/trace/neuronPack.ts`, `src/trace/neuronPack.test.ts`
- Modify: `src/trace/types.ts`, `src/trace/validate.ts`, `src/trace/validate.test.ts`, `src/app/runFiles.ts`, `src/app/runFiles.test.ts`, `src/App.tsx`

**Interfaces:**
- Produces:
  - `packHalf(values: ArrayLike<number>): string`, `unpackHalf(b64: string): Float32Array`, `packFloat32(values: ArrayLike<number>): string`, `unpackFloat32(b64: string): Float32Array`;
  - `interface NeuronData { neurons: number; activations: string; invRms: number; candidates: Array<{ id: number; row: string }> }`, carried by the `edits` event as `neurons?: NeuronData`;
  - `neuronProblems(n: unknown, layers: number, dims: number): string[]`, `dropBrokenNeurons(events: TraceEvent[]): { events: TraceEvent[]; dropped: number }`;
  - `ParsedRunFile` ok branch gains `note?: string`.

- [ ] **Step 1: Write the failing tests**

`src/trace/neuronPack.test.ts`:

```ts
import { expect, test } from 'vitest'
import { packFloat32, packHalf, unpackFloat32, unpackHalf } from './neuronPack'

test('float16 round trip stays within half-precision rounding', () => {
  const values = [0, 1, -1, 0.1, -28.25, 24.36, 3.59e-3, 6.1e-5, 1e-7, 65504, -0.0004]
  const back = unpackHalf(packHalf(values))
  expect(back).toHaveLength(values.length)
  values.forEach((v, i) => {
    const tol = Math.abs(v) >= 6.1e-5 ? Math.abs(v) * 2 ** -11 : 2 ** -25
    expect(Math.abs(back[i] - v)).toBeLessThanOrEqual(tol)
  })
})

test('float16 saturates to infinity beyond its range and keeps the sign of zero', () => {
  const back = unpackHalf(packHalf([1e6, -1e6, -0]))
  expect(back[0]).toBe(Infinity)
  expect(back[1]).toBe(-Infinity)
  expect(Object.is(back[2], -0)).toBe(true)
})

test('a long array survives (chunked base64)', () => {
  const values = Array.from({ length: 46080 }, (_, i) => Math.sin(i) * 20)
  const back = unpackHalf(packHalf(values))
  expect(back).toHaveLength(46080)
  expect(Math.abs(back[12345] - values[12345])).toBeLessThan(0.02)
})

test('float32 round trip is exact', () => {
  const values = Float32Array.from([0.123456789, -3.5, 1e-30])
  expect([...unpackFloat32(packFloat32(values))]).toEqual([...values])
})

test('malformed data throws', () => {
  expect(() => unpackHalf('###')).toThrow()
  expect(() => unpackHalf(btoa('abc'))).toThrow(/odd/)
  expect(() => unpackFloat32(btoa('abcdef'))).toThrow(/multiple of 4/)
})
```

Append to `src/trace/validate.test.ts` (merge the imports into the existing import lines):

```ts
import { dropBrokenNeurons, neuronProblems } from './validate'
import { packFloat32, packHalf } from './neuronPack'
import type { NeuronData } from './types'

const neuronData = (layers: number, dims = 576, neurons = 4): NeuronData => ({
  neurons, invRms: 0.05,
  activations: packHalf(Array.from({ length: layers * neurons }, (_, i) => i / 10)),
  candidates: [{ id: 100, row: packFloat32(new Float32Array(dims)) }],
})

const withNeurons = (n: unknown) => buildFixtureTrace({ mode: 'real', mlp: true }).map((e) =>
  e.type === 'edits' ? { ...e, neurons: n as NeuronData } : e)

test('a well-formed neurons field passes', () => {
  expect(validateTrace(withNeurons(neuronData(3)))).toEqual([])
})

test('neuronProblems names each defect', () => {
  expect(neuronProblems(neuronData(3), 3, 576)).toEqual([])
  expect(neuronProblems(neuronData(2), 3, 576)[0]).toMatch(/activations hold 8 values, expected 3 × 4/)
  expect(neuronProblems({ ...neuronData(3), activations: '%%' }, 3, 576)[0]).toMatch(/not base64 float16/)
  expect(neuronProblems({ ...neuronData(3), invRms: -1 }, 3, 576)[0]).toMatch(/invRms/)
  expect(neuronProblems(neuronData(3, 10), 3, 576)[0]).toMatch(/candidate 100 row is not 576 float32 values/)
  expect(neuronProblems(null, 3, 576)).toEqual(['neurons is not an object'])
})

test('validateTrace reports a broken neurons field with its cycle', () => {
  expect(validateTrace(withNeurons(neuronData(2)))[0]).toMatch(/^edits cycle 0 activations hold/)
})

test('dropBrokenNeurons removes only the broken fields and counts them', () => {
  const events = withNeurons(neuronData(3))
  const firstEdits = events.findIndex((e) => e.type === 'edits')
  events[firstEdits] = { ...(events[firstEdits] as Extract<TraceEvent, { type: 'edits' }>), neurons: neuronData(2) }
  const { events: kept, dropped } = dropBrokenNeurons(events)
  expect(dropped).toBe(1)
  const edits = kept.filter((e): e is Extract<TraceEvent, { type: 'edits' }> => e.type === 'edits')
  expect(edits[0].neurons).toBeUndefined()
  expect(edits[1].neurons).toBeDefined()
  expect(validateTrace(kept)).toEqual([])
})
```

(`validate.test.ts` already imports `buildFixtureTrace`; check its import line and add `TraceEvent` to the type import if missing.)

Append to `src/app/runFiles.test.ts`:

```ts
test('an import with a broken neurons field loads without it and says so', () => {
  const events = buildFixtureTrace({ mode: 'real', mlp: true }).map((e) =>
    e.type === 'edits' ? { ...e, neurons: { neurons: 4, invRms: 0.05, activations: 'AAAA', candidates: [] } } : e)
  const text = JSON.stringify({ formatVersion: 1, meta: { prompt: 'p', params: { temperature: 0.8, topK: 10, maxNewTokens: 2 },
    mode: 'real', endedAt: 1, reason: 'max-tokens' }, events })
  const parsed = parseRunFile(text)
  expect(parsed.ok).toBe(true)
  if (!parsed.ok) return
  expect(parsed.note).toBe('Neuron data of 2 cycles was damaged and left out.')
  expect(parsed.events.some((e) => e.type === 'edits' && e.neurons)).toBe(false)
})
```

(Import `buildFixtureTrace` from `../test/fixtures` if the file doesn't already.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/trace src/app/runFiles.test.ts`
Expected: FAIL, the modules or exports `neuronPack`, `neuronProblems` and `dropBrokenNeurons` are missing.

- [ ] **Step 3: Implement**

`src/trace/neuronPack.ts`:

```ts
// Neuron data rides in the trace as base64: activations as float16 (half the
// bytes of float32, about 3 significant digits — the card shows one decimal)
// and candidate rows as float32 (the exact readout rows). Little-endian, like
// every engine we run on.

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

function toHalf(v: number): number {
  f32[0] = v
  const x = u32[0]
  const sign = (x >>> 16) & 0x8000
  const exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0)          // infinity, NaN
  const e = exp - 127 + 15
  if (e >= 0x1f) return sign | 0x7c00                                     // too large: infinity
  if (e <= 0) {                                                           // subnormal half, or zero
    if (e < -10) return sign
    mant |= 0x800000
    const shift = 14 - e
    let half = mant >> shift
    const rem = mant & ((1 << shift) - 1), mid = 1 << (shift - 1)
    if (rem > mid || (rem === mid && (half & 1))) half++
    return sign | half
  }
  let half = sign | (e << 10) | (mant >> 13)
  const rem = mant & 0x1fff
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++             // round to nearest even; a carry
  return half                                                             // into the exponent is correct
}

function fromHalf(h: number): number {
  const sign = h & 0x8000 ? -1 : 1
  const exp = (h >> 10) & 0x1f
  const mant = h & 0x3ff
  if (exp === 0) return sign * mant * 2 ** -24
  if (exp === 0x1f) return mant ? NaN : sign * Infinity
  return sign * (1 + mant / 1024) * 2 ** (exp - 15)
}

function toBase64(bytes: Uint8Array): string {
  let s = ''
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(s)
}

function fromBase64(b64: string): Uint8Array {
  const s = atob(b64)
  const out = new Uint8Array(s.length)
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i)
  return out
}

export function packHalf(values: ArrayLike<number>): string {
  const u = new Uint16Array(values.length)
  for (let i = 0; i < values.length; i++) u[i] = toHalf(values[i])
  return toBase64(new Uint8Array(u.buffer))
}

export function unpackHalf(b64: string): Float32Array {
  const bytes = fromBase64(b64)
  if (bytes.length % 2) throw new Error('odd byte count for float16 data')
  const u = new Uint16Array(bytes.buffer, 0, bytes.length / 2)
  const out = new Float32Array(u.length)
  for (let i = 0; i < u.length; i++) out[i] = fromHalf(u[i])
  return out
}

export function packFloat32(values: ArrayLike<number>): string {
  return toBase64(new Uint8Array(Float32Array.from(values).buffer))
}

export function unpackFloat32(b64: string): Float32Array {
  const bytes = fromBase64(b64)
  if (bytes.length % 4) throw new Error('float32 data is not a multiple of 4 bytes')
  return new Float32Array(bytes.buffer, 0, bytes.length / 4)
}
```

In `src/trace/types.ts`, before `TraceEvent`:

```ts
// v5 export, real mode: the last token's neuron activations for every layer
// and what the browser needs to split an MLP vote into neuron votes. Base64
// keeps it to about 150 KB per cycle (see trace/neuronPack.ts).
export interface NeuronData {
  neurons: number                                   // per layer
  activations: string                               // float16 [layers × neurons], layer-major
  invRms: number                                    // the final norm's 1/rms at the last position
  candidates: Array<{ id: number; row: string }>    // each candidate's row of E, float32 [dims]
}
```

and change the `edits` member to:

```ts
  // v4 export: ‖edit‖/‖stream before‖ per fed position (rows start at `start`) and layer;
  // v5 adds the last token's neuron data
  | { type: 'edits'; cycle: number; start: number; attn: number[][]; mlp: number[][]; neurons?: NeuronData }
```

In `src/trace/validate.ts`: import `unpackFloat32, unpackHalf` from `./neuronPack` and the `NeuronData` type; add `let dims = 0` beside `let seqLen = 0`, set `dims = e.dims` in the embed branch next to `seqLen = e.seqLen`, and at the end of the edits branch (before `continue`) add:

```ts
      if (e.neurons !== undefined)
        for (const p of neuronProblems(e.neurons, layerTotal, dims)) errs.push(`edits cycle ${e.cycle} ${p}`)
```

Then add the two exported functions:

```ts
export function neuronProblems(n: unknown, layers: number, dims: number): string[] {
  if (typeof n !== 'object' || n === null) return ['neurons is not an object']
  const d = n as Partial<NeuronData>
  const out: string[] = []
  const count = d.neurons
  if (typeof count !== 'number' || !Number.isInteger(count) || count <= 0) out.push('neurons count is not a positive integer')
  if (typeof d.invRms !== 'number' || !Number.isFinite(d.invRms) || d.invRms <= 0) out.push('invRms is not a positive number')
  try {
    const a = unpackHalf(String(d.activations))
    if (a.length !== layers * (count ?? 0)) out.push(`activations hold ${a.length} values, expected ${layers} × ${count}`)
    else if (a.some((v) => !Number.isFinite(v))) out.push('activations are not all finite')
  } catch { out.push('activations are not base64 float16') }
  if (!Array.isArray(d.candidates) || d.candidates.length === 0) out.push('candidates are missing')
  else for (const c of d.candidates) {
    const id = (c as { id?: unknown })?.id
    try {
      if (typeof id !== 'number' || unpackFloat32(String(c.row)).length !== dims)
        out.push(`candidate ${String(id)} row is not ${dims} float32 values`)
    } catch { out.push(`candidate ${String(id)} row is not base64 float32`) }
  }
  return out
}

// An import keeps its run when only neuron data is damaged: the broken fields
// are dropped (the neuron card then shows its no-data note) and counted.
export function dropBrokenNeurons(events: TraceEvent[]): { events: TraceEvent[]; dropped: number } {
  let layers = 0, dims = 0, dropped = 0
  const out = events.map((e) => {
    if (e.type === 'layer') layers = e.total
    if (e.type === 'embed') dims = e.dims
    if (e.type !== 'edits' || e.neurons === undefined || neuronProblems(e.neurons, layers, dims).length === 0) return e
    dropped++
    const { neurons: _, ...rest } = e
    return rest
  })
  return { events: out, dropped }
}
```

(The `layer` events of a cycle come before its `edits` event, so `layers` is known by then. If oxlint flags the unused `_`, use `delete` on a shallow copy instead.)

In `src/app/runFiles.ts`: import `dropBrokenNeurons`; add `note?: string` to the ok branch of `ParsedRunFile`; replace the two lines from `const events = d.events as TraceEvent[]` through the `problems` check with:

```ts
  const { events, dropped } = dropBrokenNeurons(d.events as TraceEvent[])
  const problems = validateTrace(events)
  if (problems.length > 0) return { ok: false, error: `invalid trace: ${problems[0]}` }
  const note = dropped > 0 ? `Neuron data of ${dropped} cycle${dropped === 1 ? '' : 's'} was damaged and left out.` : undefined
```

and add `...(note ? { note } : {}),` to the returned ok object.

In `src/App.tsx` `handleImportFile`: change `setImportError(null)` to `setImportError(parsed.note ?? null)`.

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trace src/app/runFiles.ts src/app/runFiles.test.ts src/App.tsx
git commit -m "feat(trace): neuron data on the edits event — float16/base64 packing, validation, damaged imports keep their run"
```

---

### Task 6: Worker — extract and emit the neuron data

**Files:**
- Create: `src/engine/transformers/neuronData.ts`, `src/engine/transformers/neuronData.test.ts`
- Modify: `src/engine/transformers/worker.ts`

**Interfaces:**
- Consumes: `packHalf` and `packFloat32` (Task 5); `NeuronData` (Task 5); `sameDims` and `Tensorish` from `./readout`.
- Produces: `extractNeurons(out: NeuronOutputs, layers: number, dims: number, candidateIds: number[]): NeuronResult`, where `NeuronResult = { status: 'absent' } | { status: 'bad-shape'; detail: string } | { status: 'ok'; data: NeuronData }`.

- [ ] **Step 1: Write the failing tests** (`src/engine/transformers/neuronData.test.ts`)

```ts
import { expect, test } from 'vitest'
import { unpackFloat32, unpackHalf } from '../../trace/neuronPack'
import { extractNeurons } from './neuronData'

const L = 2, N = 3, D = 4
const outputs = () => ({
  mlp_activations: { dims: [1, L, N], data: Float32Array.from([1, -2, 3, 0.5, 0, -28.25]) },
  final_norm_inv_rms: { dims: [1, 1, 1], data: Float32Array.from([0.04]) },
  topk_ids: { dims: [1, 3], data: BigInt64Array.from([7n, 9n, 11n]) },
  topk_embeds: { dims: [1, 3, D], data: Float32Array.from(Array.from({ length: 12 }, (_, i) => i / 4)) },
})

test('packs activations, 1/rms and the rows of the requested candidates by id', () => {
  const r = extractNeurons(outputs(), L, D, [9, 7])
  expect(r.status).toBe('ok')
  if (r.status !== 'ok') return
  expect(r.data.neurons).toBe(N)
  expect(r.data.invRms).toBeCloseTo(0.04, 6)
  expect([...unpackHalf(r.data.activations)]).toEqual([1, -2, 3, 0.5, 0, -28.25])
  expect(r.data.candidates.map((c) => c.id)).toEqual([9, 7])
  expect([...unpackFloat32(r.data.candidates[0].row)]).toEqual([1, 1.25, 1.5, 1.75])
})

test('absent when the export has no neuron outputs (v4 and older)', () => {
  const { mlp_activations: _a, final_norm_inv_rms: _b, ...rest } = outputs()
  expect(extractNeurons(rest, L, D, [7]).status).toBe('absent')
})

test('bad shape: wrong layer count, missing 1/rms, unknown candidate, bad 1/rms', () => {
  const o = outputs()
  expect(extractNeurons({ ...o, mlp_activations: { dims: [1, 3, 2], data: new Float32Array(6) } }, L, D, [7]))
    .toMatchObject({ status: 'bad-shape', detail: expect.stringMatching(/mlp_activations has shape \[1, 3, 2\]/) })
  expect(extractNeurons({ ...o, final_norm_inv_rms: undefined }, L, D, [7]))
    .toMatchObject({ status: 'bad-shape', detail: expect.stringMatching(/incomplete/) })
  expect(extractNeurons(o, L, D, [42])).toMatchObject({ status: 'bad-shape', detail: 'candidate 42 is not among topk_ids' })
  expect(extractNeurons({ ...o, final_norm_inv_rms: { dims: [1, 1, 1], data: Float32Array.from([0]) } }, L, D, [7]))
    .toMatchObject({ status: 'bad-shape', detail: expect.stringMatching(/1\/rms/) })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engine/transformers/neuronData.test.ts`
Expected: FAIL, `Cannot find module './neuronData'`.

- [ ] **Step 3: Implement `neuronData.ts`**

```ts
import { packFloat32, packHalf } from '../../trace/neuronPack'
import type { NeuronData } from '../../trace/types'
import { sameDims, type Tensorish } from './readout'

// The v5 export's neuron outputs turned into the edits event's `neurons`
// field. Pure, so the shape policy is unit-testable outside the worker.
//   mlp_activations [1, layers, neurons] → float16 base64, layer-major
//   final_norm_inv_rms [1, 1, 1] → invRms
//   topk_ids + topk_embeds → the full rows of the candidates the card lists
export interface NeuronOutputs {
  mlp_activations?: Tensorish; final_norm_inv_rms?: Tensorish; topk_ids?: Tensorish; topk_embeds?: Tensorish
}
export type NeuronResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; detail: string }
  | { status: 'ok'; data: NeuronData }

const bad = (detail: string): NeuronResult => ({ status: 'bad-shape', detail })

export function extractNeurons(out: NeuronOutputs, layers: number, dims: number, candidateIds: number[]): NeuronResult {
  const { mlp_activations: act, final_norm_inv_rms: inv, topk_ids: ids, topk_embeds: emb } = out
  if (!act && !inv) return { status: 'absent' }
  if (!act || !inv || !ids || !emb)
    return bad(`neuron outputs incomplete: mlp_activations ${!!act}, final_norm_inv_rms ${!!inv}, topk_ids ${!!ids}, topk_embeds ${!!emb}`)
  const neurons = act.dims[2]
  if (act.dims.length !== 3 || act.dims[0] !== 1 || act.dims[1] !== layers || !(neurons > 0) || act.data.length !== layers * neurons)
    return bad(`mlp_activations has shape [${act.dims.join(', ')}], expected [1, ${layers}, neurons]`)
  if (inv.data.length !== 1) return bad(`final_norm_inv_rms has ${inv.data.length} values, expected 1`)
  const invRms = Number(inv.data[0])
  if (!Number.isFinite(invRms) || invRms <= 0) return bad(`final_norm_inv_rms is ${invRms}, expected a positive 1/rms`)
  const k = ids.dims[1]
  if (ids.dims.length !== 2 || !sameDims(emb, [1, k, dims]) || emb.data.length !== k * dims)
    return bad(`topk_embeds has shape [${emb.dims.join(', ')}], expected [1, ${k}, ${dims}]`)
  const index = new Map<number, number>()
  for (let i = 0; i < k; i++) index.set(Number(ids.data[i]), i)
  const candidates: NeuronData['candidates'] = []
  for (const id of candidateIds) {
    const i = index.get(id)
    if (i === undefined) return bad(`candidate ${id} is not among topk_ids`)
    const row = new Float32Array(dims)
    for (let d = 0; d < dims; d++) row[d] = Number(emb.data[i * dims + d])
    candidates.push({ id, row: packFloat32(row) })
  }
  return { status: 'ok', data: { neurons, invRms, activations: packHalf(act.data as ArrayLike<number>), candidates } }
}
```

In `worker.ts`:
- import `extractNeurons` from `./neuronData` and add `NeuronData` to the trace type import;
- add `let neuronsOff = false` after `let votesOff = false`, with the comment `// neuron card (v5 export): flipped by a bad shape — never-fail`;
- move the whole `if (!editsOff) { … }` block to just after the `const top = …` line (still before the readout and the `logits` emit), and replace it with:

```ts
      // After `top`: the neuron data carries the rows of exactly these candidates.
      if (!editsOff) {
        const r = extractEdits(out, nextInputIds.length, numLayers)
        if (r.status === 'ok') {
          let neurons: NeuronData | undefined
          if (!neuronsOff) {
            const n = extractNeurons(out, numLayers, dims, top.map((c) => c.id))
            if (n.status === 'ok') neurons = n.data
            else if (n.status === 'bad-shape') {
              neuronsOff = true
              console.warn(`neuron outputs unusable (${n.detail}) — the neuron card shows its note`)
            }
          }
          emit({ type: 'edits', cycle, start: allIds.length - nextInputIds.length, attn: r.attn, mlp: r.mlp,
            ...(neurons ? { neurons } : {}) })
        } else if (r.status === 'bad-shape') {
          editsOff = true
          console.warn(`edit_ratios unusable (${r.detail}) — the edit-size card shows its note`)
        }
      }
```

- [ ] **Step 4: Run the tests and the type check**

Run: `npm test && npx tsc -b`
Expected: all pass, no type errors.

- [ ] **Step 5: Commit**

```bash
git add src/engine/transformers/neuronData.ts src/engine/transformers/neuronData.test.ts src/engine/transformers/worker.ts
git commit -m "feat(worker): neuron data on the edits event (v5 outputs), never-fail like edits and votes"
```

---

### Task 7: Neuron asset — format, cached loader, hook, test fixture

**Files:**
- Create: `src/neurons/format.ts`, `src/neurons/asset.ts`, `src/neurons/useNeuronLayer.ts`, `src/test/neuronFixture.ts`, `src/neurons/asset.test.ts`, `src/neurons/useNeuronLayer.test.tsx`
- Modify: `src/test/setup.ts`

**Interfaces:**
- Consumes: `NEURONS_BASE_URL` (Task 4).
- Produces:
  - in `format.ts`, no app imports, so Playwright can load it: `NeuronManifest`, `NeuronLayer { scales: Float32Array; columns: Int8Array; labelIds: Uint16Array; thresholds: Float32Array }`, `layerLayout(neurons, dims, labels, fractions)`, `layerFile(l): string`, `parseNeuronLayer(m, buf): NeuronLayer`;
  - in `asset.ts`: `NeuronAsset { manifest; g: Float32Array; text(id): string }`, `loadNeuronAsset(base?)`, `loadNeuronLayer(l, base?)`, `resetNeuronCache()`, `NEURON_CACHE = 'tsumugi-neurons'`;
  - in `useNeuronLayer.ts`: `useNeuronLayer(layer: number | null): NeuronLayerState & { retry(): void }`, where the state is `{ status: 'idle' } | { status: 'loading' } | { status: 'ready'; asset: NeuronAsset; layer: NeuronLayer } | { status: 'error'; error: string }`;
  - in `neuronFixture.ts`, no vitest imports: `NF`, `encodeNeuronFixture()`, `neuronFixtureFile(name): Uint8Array | null`, `fixtureNeuronAsset(): NeuronAsset`, `fixtureNeuronLayer(l): NeuronLayer`.

- [ ] **Step 1: Write the fixture** (`src/test/neuronFixture.ts`; no vitest imports, because Playwright imports it too)

```ts
import { layerFile, layerLayout, parseNeuronLayer, type NeuronLayer, type NeuronManifest } from '../neurons/format'

// A tiny neuron asset in the real layout: 3 layers × 10 neurons over the real
// 576 dims (the trace fixture's rows are 576 wide), a 256-token vocabulary.
export const NF = { layers: 3, neurons: 10, dims: 576, vocab: 256, labels: 5, fractions: [0.001, 0.01, 0.1] }

export const fixtureG = (d: number) => 1 + 0.5 * Math.cos(d * 0.3)
const qValue = (l: number, i: number, d: number) => Math.round(100 * Math.sin(0.05 * d + l + 1.3 * i))
const labelId = (l: number, i: number, s: number, j: number) => (l * 40 + i * 10 + s * 5 + j) % NF.vocab
export const FIXTURE_THRESHOLDS = [0.5, 0.3, 0.1]

export function encodeNeuronFixture(): { manifest: NeuronManifest; layers: ArrayBuffer[]; tokens: string[] } {
  const { layers: L, neurons: n, dims, labels: k, fractions } = NF
  const layout = layerLayout(n, dims, k, fractions.length)
  const layers = Array.from({ length: L }, (_, l) => {
    const buf = new ArrayBuffer(layout.bytes)
    const view = new DataView(buf)
    for (let i = 0; i < n; i++) {
      view.setFloat32(layout.scales + i * 4, 0.002 * (i + 1), true)
      for (let d = 0; d < dims; d++) view.setInt8(layout.columns + i * dims + d, qValue(l, i, d))
      for (let s = 0; s < 2; s++) {
        for (let j = 0; j < k; j++) view.setUint16(layout.labelIds + ((i * 2 + s) * k + j) * 2, labelId(l, i, s, j), true)
        FIXTURE_THRESHOLDS.forEach((t, j) => view.setFloat32(layout.thresholds + ((i * 2 + s) * fractions.length + j) * 4, t, true))
      }
    }
    return buf
  })
  const tokens = Array.from({ length: NF.vocab }, (_, id) => (id === 32 ? ' and' : `w${id}`))
  const files: NeuronManifest['files'] = Object.fromEntries(layers.map((b, l) => [layerFile(l), { bytes: b.byteLength, sha256: '0'.repeat(64) }]))
  files['tokens.json'] = { bytes: 0, sha256: '0'.repeat(64) }
  const manifest: NeuronManifest = {
    modelId: 'fixture', layers: L, neurons: n, dims, vocabSize: NF.vocab, labelsPerSign: k, fractions,
    labelMethod: 'fixture', layout, finalNormWeight: Array.from({ length: dims }, (_, d) => fixtureG(d)), files,
  }
  return { manifest, layers, tokens }
}

// The fixture's files by name, as served bytes (Playwright routes and fetch stubs).
export function neuronFixtureFile(name: string): Uint8Array | null {
  const f = encodeNeuronFixture()
  if (name === 'manifest.json') return new TextEncoder().encode(JSON.stringify(f.manifest))
  if (name === 'tokens.json') return new TextEncoder().encode(JSON.stringify(f.tokens))
  const l = f.layers.findIndex((_, i) => layerFile(i) === name)
  return l >= 0 ? new Uint8Array(f.layers[l]) : null
}

export function fixtureNeuronLayer(l: number): NeuronLayer {
  const f = encodeNeuronFixture()
  return parseNeuronLayer(f.manifest, f.layers[l])
}

export function fixtureNeuronAsset() {
  const f = encodeNeuronFixture()
  return { manifest: f.manifest, g: Float32Array.from(f.manifest.finalNormWeight), text: (id: number) => f.tokens[id] ?? `#${id}` }
}
```

- [ ] **Step 2: Write the failing tests**

`src/neurons/asset.test.ts`:

```ts
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { encodeNeuronFixture, NF, neuronFixtureFile } from '../test/neuronFixture'
import { layerLayout, parseNeuronLayer } from './format'
import { loadNeuronAsset, loadNeuronLayer, NEURON_CACHE, resetNeuronCache } from './asset'

const BASE = 'https://hub.test/neurons'

// A Cache API stand-in: a Map of URL → Response, shared by every open().
function stubCaches() {
  const store = new Map<string, Response>()
  const cache = {
    match: vi.fn(async (url: string) => store.get(url)?.clone()),
    put: vi.fn(async (url: string, r: Response) => { store.set(url, r) }),
    delete: vi.fn(async (url: string) => store.delete(url)),
  }
  const open = vi.fn(async (_name: string) => cache)
  vi.stubGlobal('caches', { open })
  return { store, cache, open }
}

function stubFetch(override: (name: string) => Response | null = () => null) {
  const mock = vi.fn(async (url: string) => {
    const name = url.slice(url.lastIndexOf('/') + 1)
    const o = override(name)
    if (o) return o
    const body = neuronFixtureFile(name)
    return body ? new Response(body) : new Response('missing', { status: 404 })
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

beforeEach(() => resetNeuronCache())
afterEach(() => vi.unstubAllGlobals())

test('layerLayout matches the Python builder for the real sizes', () => {
  expect(layerLayout(1536, 576, 5, 3)).toEqual({ scales: 0, columns: 6144, labelIds: 890880, thresholds: 921600, bytes: 958464 })
})

test('parseNeuronLayer reads every block and refuses a wrong size', () => {
  const f = encodeNeuronFixture()
  const layer = parseNeuronLayer(f.manifest, f.layers[1])
  expect(layer.scales[2]).toBeCloseTo(0.006, 6)
  expect(layer.columns).toHaveLength(NF.neurons * NF.dims)
  expect(layer.labelIds).toHaveLength(NF.neurons * 2 * 5)
  expect(layer.thresholds[0]).toBeCloseTo(0.5, 6)
  expect(() => parseNeuronLayer(f.manifest, new ArrayBuffer(10))).toThrow(/expected \d+ bytes, got 10/)
})

test('a miss fetches and stores in the cache; the next load is served from it', async () => {
  const { store, open } = stubCaches()
  const fetchMock = stubFetch()
  const asset = await loadNeuronAsset(BASE)
  await loadNeuronLayer(0, BASE)
  expect(open).toHaveBeenCalledWith(NEURON_CACHE)
  expect(asset.text(32)).toBe(' and')
  expect(asset.g[0]).toBeCloseTo(1.5, 6)
  expect([...store.keys()].sort()).toEqual([`${BASE}/layer-00.bin`, `${BASE}/manifest.json`, `${BASE}/tokens.json`])
  resetNeuronCache()
  fetchMock.mockClear()
  await loadNeuronLayer(0, BASE)
  expect(fetchMock).not.toHaveBeenCalled()
})

test('two requests for one layer fetch it once', async () => {
  stubCaches()
  const fetchMock = stubFetch()
  await Promise.all([loadNeuronLayer(2, BASE), loadNeuronLayer(2, BASE)])
  expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('layer-02.bin'))).toHaveLength(1)
})

test('without the Cache API it falls back to plain fetch', async () => {
  vi.stubGlobal('caches', undefined)
  stubFetch()
  const layer = await loadNeuronLayer(1, BASE)
  expect(layer.scales).toHaveLength(NF.neurons)
})

test('a failed fetch rejects, and the next load tries again', async () => {
  stubCaches()
  let fail = true
  stubFetch((name) => (fail && name === 'layer-01.bin' ? new Response('', { status: 503 }) : null))
  await expect(loadNeuronLayer(1, BASE)).rejects.toThrow('layer-01.bin: HTTP 503')
  fail = false
  await expect(loadNeuronLayer(1, BASE)).resolves.toBeDefined()
})

test('a truncated cached file is evicted, so a retry downloads it again', async () => {
  const { store, cache } = stubCaches()
  stubFetch()
  store.set(`${BASE}/layer-00.bin`, new Response(new Uint8Array(12)))
  await expect(loadNeuronLayer(0, BASE)).rejects.toThrow(/expected \d+ bytes, got 12/)
  expect(cache.delete).toHaveBeenCalledWith(`${BASE}/layer-00.bin`)
  await expect(loadNeuronLayer(0, BASE)).resolves.toBeDefined()
})
```

`src/neurons/useNeuronLayer.test.tsx`:

```tsx
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { fixtureNeuronAsset, fixtureNeuronLayer } from '../test/neuronFixture'
import * as asset from './asset'
import { useNeuronLayer } from './useNeuronLayer'

afterEach(() => vi.restoreAllMocks())

test('idle without a layer, then loading, then ready', async () => {
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  vi.spyOn(asset, 'loadNeuronLayer').mockImplementation(async (l) => fixtureNeuronLayer(l))
  const { result, rerender } = renderHook(({ l }) => useNeuronLayer(l), { initialProps: { l: null as number | null } })
  expect(result.current.status).toBe('idle')
  rerender({ l: 1 })
  expect(result.current.status).toBe('loading')
  await waitFor(() => expect(result.current.status).toBe('ready'))
})

test('a stale result for an earlier layer never replaces the current one', async () => {
  let release0: () => void = () => {}
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  vi.spyOn(asset, 'loadNeuronLayer').mockImplementation((l) => l === 0
    ? new Promise((resolve) => { release0 = () => resolve(fixtureNeuronLayer(0)) })
    : Promise.resolve(fixtureNeuronLayer(l)))
  const { result, rerender } = renderHook(({ l }) => useNeuronLayer(l), { initialProps: { l: 0 as number | null } })
  rerender({ l: 2 })
  await waitFor(() => expect(result.current.status).toBe('ready'))
  const shown = result.current.status === 'ready' ? result.current.layer : null
  await act(async () => { release0() })
  expect(result.current.status === 'ready' && result.current.layer).toBe(shown)
})

test('an error offers retry, which loads again', async () => {
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  const load = vi.spyOn(asset, 'loadNeuronLayer').mockRejectedValueOnce(new Error('HTTP 503'))
    .mockImplementation(async (l) => fixtureNeuronLayer(l))
  const { result } = renderHook(() => useNeuronLayer(0))
  await waitFor(() => expect(result.current.status).toBe('error'))
  act(() => result.current.retry())
  await waitFor(() => expect(result.current.status).toBe('ready'))
  expect(load).toHaveBeenCalledTimes(2)
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `npx vitest run src/neurons`
Expected: FAIL, `Cannot find module './format'` (or `./asset`).

- [ ] **Step 4: Implement**

`src/neurons/format.ts`:

```ts
// The neurons/ asset's file format (#16 spec, section 2). No app imports, so
// the Playwright tests can build fixtures with it.

export interface LayerLayout { scales: number; columns: number; labelIds: number; thresholds: number; bytes: number }

export interface NeuronManifest {
  modelId: string
  layers: number
  neurons: number
  dims: number
  vocabSize: number
  labelsPerSign: number
  fractions: number[]          // the badge levels: 0.001, 0.01, 0.1 of the vocabulary
  labelMethod: string
  layout: LayerLayout
  finalNormWeight: number[]    // g, [dims]
  files: Record<string, { bytes: number; sha256: string }>
}

export interface NeuronLayer {
  scales: Float32Array         // [neurons]
  columns: Int8Array           // [neurons × dims]: column i at i·dims, times scales[i]
  labelIds: Uint16Array        // [neurons × 2 × labelsPerSign]: sign 0 = positive activation, 1 = negative
  thresholds: Float32Array     // [neurons × 2 × fractions]: the cosine at each badge level's rank
}

// Mirrors tools/export/src/tsumugi_export/neurons.py layer_layout.
export function layerLayout(neurons: number, dims: number, labels: number, fractions: number): LayerLayout {
  const columns = neurons * 4
  const labelIds = columns + neurons * dims
  const thresholds = labelIds + neurons * 2 * labels * 2
  return { scales: 0, columns, labelIds, thresholds, bytes: thresholds + neurons * 2 * fractions * 4 }
}

export const layerFile = (l: number) => `layer-${String(l).padStart(2, '0')}.bin`

// Copies (slice) rather than views for the multi-byte blocks, so alignment
// never depends on the layout.
export function parseNeuronLayer(m: NeuronManifest, buf: ArrayBuffer): NeuronLayer {
  const { neurons: n, dims, labelsPerSign: k, fractions, layout } = m
  if (buf.byteLength !== layout.bytes) throw new Error(`layer file: expected ${layout.bytes} bytes, got ${buf.byteLength}`)
  return {
    scales: new Float32Array(buf.slice(layout.scales, layout.scales + n * 4)),
    columns: new Int8Array(buf, layout.columns, n * dims),
    labelIds: new Uint16Array(buf.slice(layout.labelIds, layout.labelIds + n * 2 * k * 2)),
    thresholds: new Float32Array(buf.slice(layout.thresholds, layout.thresholds + n * 2 * fractions.length * 4)),
  }
}
```

`src/neurons/asset.ts`:

```ts
// The Hub neuron asset: manifest and texts once, layer files on demand. Files
// are immutable per tag and the URL carries the tag, so the Cache API copy
// never needs revalidating (transformers.js caches only its own files).
import { NEURONS_BASE_URL } from '../engine/tokenizer'
import { layerFile, parseNeuronLayer, type NeuronLayer, type NeuronManifest } from './format'

export interface NeuronAsset { manifest: NeuronManifest; g: Float32Array; text(id: number): string }

export const NEURON_CACHE = 'tsumugi-neurons'

async function openCache(): Promise<Cache | null> {
  try { return typeof caches === 'undefined' ? null : await caches.open(NEURON_CACHE) } catch { return null }
}

async function cachedFetch(url: string): Promise<Response> {
  const cache = await openCache()
  const hit = await cache?.match(url).catch(() => undefined)
  if (hit) return hit
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url.slice(url.lastIndexOf('/') + 1)}: HTTP ${r.status}`)
  if (cache) await cache.put(url, r.clone()).catch(() => {})
  return r
}

async function evict(url: string) {
  await (await openCache())?.delete(url).catch(() => false)
}

let assetPending: Promise<NeuronAsset> | null = null
const layerPending = new Map<number, Promise<NeuronLayer>>()

export function loadNeuronAsset(base: string = NEURONS_BASE_URL): Promise<NeuronAsset> {
  if (!assetPending) assetPending = fetchAsset(base).catch((err) => { assetPending = null; throw err })
  return assetPending
}

export function loadNeuronLayer(l: number, base: string = NEURONS_BASE_URL): Promise<NeuronLayer> {
  let p = layerPending.get(l)
  if (!p) {
    const url = `${base}/${layerFile(l)}`
    p = loadNeuronAsset(base)
      .then(async (asset) => {
        const buf = await (await cachedFetch(url)).arrayBuffer()
        try { return parseNeuronLayer(asset.manifest, buf) } catch (err) { await evict(url); throw err }
      })
      .catch((err) => { layerPending.delete(l); throw err })
    layerPending.set(l, p)
  }
  return p
}

export function resetNeuronCache(): void { assetPending = null; layerPending.clear() }

async function fetchAsset(base: string): Promise<NeuronAsset> {
  const manifest = (await (await cachedFetch(`${base}/manifest.json`)).json()) as NeuronManifest
  const tokens = (await (await cachedFetch(`${base}/tokens.json`)).json()) as string[]
  if (tokens.length !== manifest.vocabSize)
    throw new Error(`tokens.json: expected ${manifest.vocabSize} entries, got ${tokens.length}`)
  if (manifest.finalNormWeight.length !== manifest.dims)
    throw new Error(`manifest: finalNormWeight has ${manifest.finalNormWeight.length} values, expected ${manifest.dims}`)
  return { manifest, g: Float32Array.from(manifest.finalNormWeight), text: (id) => tokens[id] ?? `#${id}` }
}
```

`src/neurons/useNeuronLayer.ts`:

```ts
import { useCallback, useEffect, useState } from 'react'
import { loadNeuronAsset, loadNeuronLayer, type NeuronAsset } from './asset'
import type { NeuronLayer } from './format'

export type NeuronLayerState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; asset: NeuronAsset; layer: NeuronLayer }
  | { status: 'error'; error: string }

// Loads the asset and one layer file. Results are keyed by (layer, attempt):
// a result for any other key is ignored, so a slow earlier layer never
// replaces the one on screen, and a new key reads as 'loading' at once
// (no state set inside the effect).
export function useNeuronLayer(layer: number | null): NeuronLayerState & { retry(): void } {
  const [attempt, setAttempt] = useState(0)
  const [result, setResult] = useState<{ key: string; state: NeuronLayerState } | null>(null)
  const key = layer === null ? null : `${layer}:${attempt}`
  useEffect(() => {
    if (layer === null) return
    let live = true
    const k = `${layer}:${attempt}`
    Promise.all([loadNeuronAsset(), loadNeuronLayer(layer)]).then(
      ([asset, data]) => { if (live) setResult({ key: k, state: { status: 'ready', asset, layer: data } }) },
      (err: unknown) => { if (live) setResult({ key: k, state: { status: 'error', error: err instanceof Error ? err.message : String(err) } }) },
    )
    return () => { live = false }
  }, [layer, attempt])
  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  const state: NeuronLayerState = key === null ? { status: 'idle' }
    : result?.key === key ? result.state : { status: 'loading' }
  return { ...state, retry }
}
```

In `src/test/setup.ts`, add `import { resetNeuronCache } from '../neurons/asset'` and call `resetNeuronCache()` in the existing `afterEach` callback.

- [ ] **Step 5: Run the tests**

Run: `npx vitest run src/neurons && npm test`
Expected: all pass. If `Response` is undefined in the jsdom environment, add `// @vitest-environment node` at the top of `asset.test.ts` and rerun.

- [ ] **Step 6: Commit**

```bash
git add src/neurons src/test/neuronFixture.ts src/test/setup.ts
git commit -m "feat(neurons): asset format, Cache API loader with per-layer dedupe and eviction, loading hook"
```

---

### Task 8: Neuron math

**Files:**
- Create: `src/neurons/math.ts`, `src/neurons/math.test.ts`

**Interfaces:**
- Consumes: `NeuronManifest` and `NeuronLayer` (Task 7).
- Produces:
  - `type Badge = 'top 0.1%' | 'top 1%' | 'top 10%' | 'not in top 10%'`;
  - `interface VoteRow { neuron; activation; vote; labels: number[]; away: boolean; badge: Badge }`;
  - `interface VoteTable { rows: VoteRow[]; rest: number; total: number }`;
  - `interface SizeRow { neuron; activation; size; labels: number[] }`;
  - `layerActivations(all: Float32Array, layer: number, neurons: number): Float32Array`;
  - `behindTheVote(m, layer, acts, row, g, invRms, n = 8): VoteTable`;
  - `firingMost(m, layer, acts, n = 8): SizeRow[]`;
  - `voteTolerance(v)`;
  - `reconcile(table, cardValue): VoteTable & { exact: boolean }`.

- [ ] **Step 1: Write the failing tests** (`src/neurons/math.test.ts`)

```ts
import { expect, test } from 'vitest'
import type { NeuronLayer, NeuronManifest } from './format'
import { behindTheVote, firingMost, layerActivations, reconcile } from './math'

// 4 neurons over 3 dims. Dequantized columns:
//   W0 = [2.54, 0, 0]   W1 = [0, 2.54, 0]   W2 = [0, 0, 3.81]   W3 = [1.27, 1.27, 0]
const m = { neurons: 4, dims: 3, labelsPerSign: 2, fractions: [0.001, 0.01, 0.1] } as NeuronManifest
const T = [0.9, 0.6, 0.3]
const layer: NeuronLayer = {
  scales: Float32Array.from([0.02, 0.02, 0.03, 0.01]),
  columns: Int8Array.from([127, 0, 0, 0, 127, 0, 0, 0, 127, 127, 127, 0]),
  labelIds: Uint16Array.from(Array.from({ length: 16 }, (_, k) => 10 * Math.floor(k / 4) + 5 * (Math.floor(k / 2) % 2) + (k % 2))),
  thresholds: Float32Array.from([0.5, 0.4, 0.2, ...T, ...T, ...T, ...T, ...T, ...T, ...T]),
}
const g = Float32Array.from([1, 1, 1])
const row = Float32Array.from([1, -1, 0.5])           // u = g ⊙ row
const acts = Float32Array.from([2, 1, 4, -3])
const INV = 0.5
// votes = a · (W·u) · inv: [2·2.54·.5, 1·(−2.54)·.5, 4·1.905·.5, −3·0·.5] = [2.54, −1.27, 3.81, 0]

test('votes rank by size; the rest line is the total minus the rows shown', () => {
  const t = behindTheVote(m, layer, acts, row, g, INV, 2)
  expect(t.rows.map((r) => r.neuron)).toEqual([2, 0])
  expect(t.rows[0].vote).toBeCloseTo(3.81, 4)
  expect(t.rows[1].vote).toBeCloseTo(2.54, 4)
  expect(t.total).toBeCloseTo(5.08, 4)
  expect(t.rest).toBeCloseTo(-1.27, 4)
})

test('labels and badges follow the sign the neuron fired with, or the opposite list when it votes away', () => {
  const byNeuron = Object.fromEntries(behindTheVote(m, layer, acts, row, g, INV, 4).rows.map((r) => [r.neuron, r]))
  // n0: fired +, cos(W0, u) = 1/1.5 ≥ 0.5 (its own thresholds) → top 0.1%
  expect(byNeuron[0]).toMatchObject({ away: false, badge: 'top 0.1%', labels: [0, 1] })
  // n1: fired +, votes away → negative list; cos(−W1, u) = 0.667 ≥ 0.6 → top 1%
  expect(byNeuron[1]).toMatchObject({ away: true, badge: 'top 1%', labels: [15, 16] })
  // n2: cos(W2, u) = 0.333 ≥ 0.3 → top 10%
  expect(byNeuron[2]).toMatchObject({ away: false, badge: 'top 10%', labels: [20, 21] })
  // n3: fired −, vote 0 → its negative list; cos(−W3, u) = 0 → not in top 10%
  expect(byNeuron[3]).toMatchObject({ away: false, badge: 'not in top 10%', labels: [35, 36] })
})

test('a negative activation with a positive vote uses the negative-sign list and thresholds', () => {
  const t = behindTheVote(m, layer, Float32Array.from([-2, 0, 0, 0]), Float32Array.from([-1, 0, 0]), g, INV, 1)
  expect(t.rows[0]).toMatchObject({ neuron: 0, away: false, labels: [5, 6], badge: 'top 0.1%' })
  expect(t.rows[0].vote).toBeCloseTo(2.54, 4)
})

test('the final norm weight scales the candidate row before the comparison', () => {
  const t = behindTheVote(m, layer, acts, row, Float32Array.from([2, 1, 1]), INV, 4)
  expect(t.rows.find((r) => r.neuron === 0)!.vote).toBeCloseTo(5.08, 4)
})

test('firing most ranks by |activation| × column length, labels from the fired sign', () => {
  const rows = firingMost(m, layer, acts, 2)
  expect(rows.map((r) => r.neuron)).toEqual([2, 3])
  expect(rows[0].size).toBeCloseTo(15.24, 3)
  expect(rows[1].size).toBeCloseTo(3 * 1.27 * Math.SQRT2, 3)
  expect(rows[1].labels).toEqual([35, 36])
})

test('reconcile adopts the votes card value within tolerance, keeps the computed total outside it', () => {
  const t = behindTheVote(m, layer, acts, row, g, INV, 2)
  const close = reconcile(t, 5.1)
  expect(close.exact).toBe(true)
  expect(close.total).toBe(5.1)
  expect(close.rest).toBeCloseTo(5.1 - 6.35, 4)
  const far = reconcile(t, 6)
  expect(far.exact).toBe(false)
  expect(far.total).toBeCloseTo(5.08, 4)
})

test('layerActivations slices one layer out of the layer-major array', () => {
  expect([...layerActivations(Float32Array.from([1, 2, 3, 4, 5, 6]), 1, 3)]).toEqual([4, 5, 6])
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/neurons/math.test.ts`
Expected: FAIL, `Cannot find module './math'`.

- [ ] **Step 3: Implement `src/neurons/math.ts`**

```ts
import type { NeuronLayer, NeuronManifest } from './format'

// The numbers behind "Inside one MLP" (#16 spec, section 3). With W_i the
// dequantized column of neuron i, a_i its activation and u = g ⊙ E_c:
//   vote_i = a_i · (W_i · u) · 1/rms        (they add up to the MLP's vote)
//   size_i = |a_i| · ‖W_i‖                  (how far the neuron moved the vector)

export type Badge = 'top 0.1%' | 'top 1%' | 'top 10%' | 'not in top 10%'
const BADGES: Badge[] = ['top 0.1%', 'top 1%', 'top 10%']

export interface VoteRow { neuron: number; activation: number; vote: number; labels: number[]; away: boolean; badge: Badge }
export interface VoteTable { rows: VoteRow[]; rest: number; total: number }
export interface SizeRow { neuron: number; activation: number; size: number; labels: number[] }

export const layerActivations = (all: Float32Array, layer: number, neurons: number) =>
  all.subarray(layer * neurons, (layer + 1) * neurons)

function columnDot(layer: NeuronLayer, dims: number, i: number, u: Float32Array): number {
  let s = 0
  for (let d = 0; d < dims; d++) s += layer.columns[i * dims + d] * u[d]
  return s * layer.scales[i]
}

function columnNorm(layer: NeuronLayer, dims: number, i: number): number {
  let s = 0
  for (let d = 0; d < dims; d++) s += layer.columns[i * dims + d] ** 2
  return Math.sqrt(s) * layer.scales[i]
}

// sign 0 = the list for a positive activation (the column itself), 1 = for a negative one
function labels(m: NeuronManifest, layer: NeuronLayer, i: number, sign: number): number[] {
  const k = m.labelsPerSign, start = (i * 2 + sign) * k
  return Array.from(layer.labelIds.subarray(start, start + k))
}

function badge(m: NeuronManifest, layer: NeuronLayer, i: number, sign: number, cos: number): Badge {
  const f = m.fractions.length, start = (i * 2 + sign) * f
  for (let j = 0; j < f; j++) if (cos >= layer.thresholds[start + j]) return BADGES[j] ?? 'not in top 10%'
  return 'not in top 10%'
}

const topBy = <T,>(items: T[], key: (t: T) => number, n: number) =>
  [...items].sort((a, b) => key(b) - key(a)).slice(0, n)

export function behindTheVote(m: NeuronManifest, layer: NeuronLayer, acts: Float32Array, row: Float32Array,
  g: Float32Array, invRms: number, n = 8): VoteTable {
  const { neurons, dims } = m
  const u = new Float32Array(dims)
  let uNorm = 0
  for (let d = 0; d < dims; d++) { u[d] = g[d] * row[d]; uNorm += u[d] ** 2 }
  uNorm = Math.sqrt(uNorm) || 1
  const dots = Array.from({ length: neurons }, (_, i) => columnDot(layer, dims, i, u))
  const votes = dots.map((dot, i) => acts[i] * dot * invRms)
  const total = votes.reduce((a, b) => a + b, 0)
  const rows = topBy([...votes.keys()], (i) => Math.abs(votes[i]), n).map((i): VoteRow => {
    const fired = acts[i] >= 0 ? 0 : 1
    const away = votes[i] < 0
    const sign = away ? 1 - fired : fired
    const norm = columnNorm(layer, dims, i) || 1
    const cos = ((sign === 0 ? 1 : -1) * dots[i]) / (norm * uNorm)
    return { neuron: i, activation: acts[i], vote: votes[i], labels: labels(m, layer, i, sign), away, badge: badge(m, layer, i, sign, cos) }
  })
  const shown = rows.reduce((a, r) => a + r.vote, 0)
  return { rows, rest: total - shown, total }
}

export function firingMost(m: NeuronManifest, layer: NeuronLayer, acts: Float32Array, n = 8): SizeRow[] {
  const sizes = Array.from({ length: m.neurons }, (_, i) => Math.abs(acts[i]) * columnNorm(layer, m.dims, i))
  return topBy([...sizes.keys()], (i) => sizes[i], n).map((i) => ({
    neuron: i, activation: acts[i], size: sizes[i], labels: labels(m, layer, i, acts[i] >= 0 ? 0 : 1),
  }))
}

export const voteTolerance = (v: number) => 0.05 + 0.01 * Math.abs(v)

// Within tolerance, the table adopts the votes card's number, so its lines add
// up to exactly what that card shows; outside it, the computed total stays.
export function reconcile(table: VoteTable, cardValue: number): VoteTable & { exact: boolean } {
  if (Math.abs(table.total - cardValue) > voteTolerance(cardValue)) return { ...table, exact: false }
  const shown = table.rows.reduce((a, r) => a + r.vote, 0)
  return { ...table, total: cardValue, rest: cardValue - shown, exact: true }
}
```

- [ ] **Step 4: Run the tests**

Run: `npx vitest run src/neurons/math.test.ts && npm test`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/neurons/math.ts src/neurons/math.test.ts
git commit -m "feat(neurons): per-neuron votes, edit sizes, sign-aware labels and badges, reconciliation with the votes card"
```

---

### Task 9: Selection, selector, clicks in the #15 cards, trace fixture

**Files:**
- Create: `src/viz/mlpLayerSelection.ts`, `src/viz/mlpLayerSelection.test.ts`
- Modify: `src/viz/selectors.ts`, `src/viz/selectors.test.ts`, `src/viz/EditSizeCard.tsx`, `src/viz/EditSizeCard.test.tsx`, `src/viz/LayerVotesCard.tsx`, `src/viz/LayerVotesCard.test.tsx`, `src/test/fixtures.ts`, `src/test/neuronFixture.ts`, `src/test/setup.ts`

**Interfaces:**
- Consumes: `NeuronData` (Task 5), `behindTheVote` (Task 8), `fixtureNeuronLayer` (Task 7), `packHalf` and `packFloat32`, `unpackHalf` and `unpackFloat32`.
- Produces:
  - `useMlpLayer` store `{ pick: { run: TraceEvent; layer: number } | null; select(run, layer) }`, `resetMlpLayer()`, `selectedLayer(pick, run): number | null`;
  - `CycleNeurons = { status: 'ok'; data: NeuronData } | { status: 'pending' } | { status: 'none' }`, `cycleNeurons(events, cursor)`;
  - `EditSizeCard` and `LayerVotesCard` take `onLayer?: (layer: number) => void`;
  - fixture `buildFixtureTrace({ neurons: true })` (requires `mlp: true` and 3 layers);
  - `fixtureNeuronData(cycle, ids): NeuronData`.

- [ ] **Step 1: Extend the fixture** (append to `src/test/neuronFixture.ts`; add the imports at its top)

```ts
import { packFloat32, packHalf, unpackFloat32, unpackHalf } from '../trace/neuronPack'
import type { NeuronData } from '../trace/types'
import { behindTheVote, layerActivations } from '../neurons/math'

export const FIXTURE_INV_RMS = 0.05

// Deterministic neuron data for a cycle, with each candidate's row from
// `rowOf` (the trace fixture passes fixtureEmbedding).
export function fixtureNeuronData(cycle: number, ids: number[], rowOf: (id: number) => number[]): NeuronData {
  const acts = Array.from({ length: NF.layers * NF.neurons }, (_, k) =>
    3 * Math.cos(cycle + 0.7 * Math.floor(k / NF.neurons) + 1.1 * (k % NF.neurons)))
  return { neurons: NF.neurons, invRms: FIXTURE_INV_RMS, activations: packHalf(acts),
    candidates: ids.map((id) => ({ id, row: packFloat32(rowOf(id)) })) }
}

// The MLP votes per layer that the neuron data adds up to for candidate `id`
// (computed from the packed values, exactly as the card does).
export function fixtureMlpVotes(data: NeuronData, id: number): number[] {
  const asset = fixtureNeuronAsset()
  const acts = unpackHalf(data.activations)
  const row = unpackFloat32(data.candidates.find((c) => c.id === id)!.row)
  return Array.from({ length: NF.layers }, (_, l) => behindTheVote(asset.manifest, fixtureNeuronLayer(l),
    layerActivations(acts, l, NF.neurons), row, asset.g, data.invRms, NF.neurons).total)
}
```

In `src/test/fixtures.ts`:
- import `fixtureMlpVotes, fixtureNeuronData` from `./neuronFixture`;
- add to `FixtureTraceOpts`: `// neuron data on the edits events (v5 export); needs mlp and 3 layers` and `neurons?: boolean`;
- in `buildFixtureTrace`, compute `scored` before the `if (opts.mlp)` edits block (move the `const scored: LogitCandidate[] = […]` line up). Then:

```ts
    const neurons = opts.mlp && opts.neurons
      ? fixtureNeuronData(c, scored.map((s) => s.id), fixtureEmbedding) : undefined
```

  add `...(neurons ? { neurons } : {})` to the pushed edits event; and replace `withVotes` with:

```ts
    const withVotes = (s: LogitCandidate): LogitCandidate => {
      if (!opts.mlp) return s
      const v = fixtureVotes(s.id, s.logit, layers)
      if (!neurons) return { ...s, votes: v }
      const mlp = fixtureMlpVotes(neurons, s.id).map(r3)
      const rest = [...v.attn, ...mlp].reduce((a, b) => a + b, 0)
      return { ...s, votes: { embed: Math.round((s.logit - rest) * 1e6) / 1e6, attn: v.attn, mlp } }
    }
```

  and at the top of `buildFixtureTrace` add `if (opts.neurons && layers !== 3) throw new Error('neuron fixture has 3 layers')`.

- [ ] **Step 2: Write the failing tests**

`src/viz/mlpLayerSelection.test.ts`:

```ts
import { expect, test } from 'vitest'
import { buildFixtureTrace } from '../test/fixtures'
import { selectedLayer, useMlpLayer } from './mlpLayerSelection'

test('a pick holds for its run and falls back for another run', () => {
  const a = buildFixtureTrace()[0], b = buildFixtureTrace()[0]
  useMlpLayer.getState().select(a, 2)
  const pick = useMlpLayer.getState().pick
  expect(selectedLayer(pick, a)).toBe(2)
  expect(selectedLayer(pick, b)).toBeNull()
  expect(selectedLayer(null, a)).toBeNull()
})
```

Append to `src/viz/selectors.test.ts` (import `cycleNeurons`):

```ts
test('cycleNeurons: ok with data, none for older runs, pending while the cycle runs', () => {
  const real = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
  const layerAt = real.findIndex((e) => e.type === 'layer')
  expect(cycleNeurons(real, layerAt).status).toBe('ok')
  const old = buildFixtureTrace({ mode: 'real', mlp: true })
  expect(cycleNeurons(old, old.findIndex((e) => e.type === 'layer')).status).toBe('none')
  const running = real.slice(0, layerAt + 1)
  expect(cycleNeurons(running, layerAt).status).toBe('pending')
})

test('the trace fixture with neurons validates and its votes still add up to the logits', () => {
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
  expect(validateTrace(events)).toEqual([])
  const logits = events.find((e): e is Extract<TraceEvent, { type: 'logits' }> => e.type === 'logits')!
  for (const c of logits.topK) {
    const v = c.votes!
    expect(v.embed + v.attn.reduce((a, b) => a + b, 0) + v.mlp.reduce((a, b) => a + b, 0)).toBeCloseTo(c.logit, 5)
  }
})
```

(Import `validateTrace` from `../trace/validate` and the `TraceEvent` type if missing.)

Append to `src/viz/EditSizeCard.test.tsx`:

```tsx
test('clicking a heatmap cell or a bar reports its layer', () => {
  const onLayer = vi.fn()
  const events = buildFixtureTrace({ mode: 'real', mlp: true })
  render(<EditSizeCard rows={editRows(events, events.length - 1)!} note="" onLayer={onLayer} />)
  fireEvent.click(screen.getAllByTestId('edit-cell').find((c) => c.dataset.layer === '2')!)
  fireEvent.click(screen.getAllByTestId('edit-bar').find((b) => b.dataset.layer === '1')!)
  expect(onLayer.mock.calls).toEqual([[2], [1]])
})
```

Append to `src/viz/LayerVotesCard.test.tsx`:

```tsx
test('clicking a vote cell reports its layer', () => {
  const onLayer = vi.fn()
  const events = buildFixtureTrace({ mode: 'real', mlp: true })
  const v = cycleVotes(events, events.length - 1)
  render(<LayerVotesCard candidates={v.status === 'ok' ? v.candidates : undefined} note="" onLayer={onLayer} />)
  fireEvent.click(screen.getAllByTestId('vote-cell').find((c) => c.dataset.row === '1' && c.dataset.layer === '2')!)
  expect(onLayer).toHaveBeenCalledWith(2)
})
```

(Merge any missing imports — `vi`, `fireEvent`, `editRows`, `cycleVotes`, `buildFixtureTrace` — into each file's existing import lines.)

- [ ] **Step 3: Run the tests to verify they fail**

Run: `npx vitest run src/viz`
Expected: FAIL, `mlpLayerSelection` and `cycleNeurons` are missing, and `onLayer` is never called.

- [ ] **Step 4: Implement**

`src/viz/mlpLayerSelection.ts`:

```ts
import { create } from 'zustand'
import type { TraceEvent } from '../trace/types'

// The layer "Inside one MLP" shows, set by clicks in either #15 card and by
// its own stepper. Tied to the run it was picked in (that trace's run-start
// event): a new run, another archived run or an import falls back to the
// default layer (the largest MLP vote for the selected candidate).
interface MlpLayerSelection {
  pick: { run: TraceEvent; layer: number } | null
  select: (run: TraceEvent, layer: number) => void
}

export const useMlpLayer = create<MlpLayerSelection>((set) => ({
  pick: null,
  select: (run, layer) => set({ pick: { run, layer } }),
}))

export const resetMlpLayer = () => useMlpLayer.setState({ pick: null })

export function selectedLayer(pick: MlpLayerSelection['pick'], run: TraceEvent | undefined): number | null {
  return pick && run && pick.run === run ? pick.layer : null
}
```

Add `resetMlpLayer()` to `src/test/setup.ts`'s `afterEach` (with its import).

In `src/viz/selectors.ts` (import `NeuronData`):

```ts
export type CycleNeurons =
  | { status: 'ok'; data: NeuronData }
  | { status: 'pending' }   // the viewed cycle's edits event is not in the trace yet, and the run goes on
  | { status: 'none' }      // the cycle went past without neuron data (older export, simulated run)

// Like cycleVotes, looked up in the whole trace: the edits event follows the
// layers, so on the Layers stage this is a look ahead.
export function cycleNeurons(events: TraceEvent[], cursor: number): CycleNeurons {
  const cycle = viewedCycle(events, cursor)
  const edits = events.find((e): e is Extract<TraceEvent, { type: 'edits' }> => e.type === 'edits' && e.cycle === cycle)
  if (edits?.neurons) return { status: 'ok', data: edits.neurons }
  if (edits || events.some((e) => (e.type === 'logits' && e.cycle === cycle) || e.type === 'run-end')) return { status: 'none' }
  return { status: 'pending' }
}
```

`EditSizeCard.tsx`: add `onLayer?: (layer: number) => void` to the props and pass it to `EditHeatmap` and `EditBars`. In `EditHeatmap`, change the cell's `onClick` to `() => { onSelect(r); onLayer?.(l) }`. In `EditBars`, add `onClick={() => onLayer?.(l)}` to each `rect` with `data-testid="edit-bar"`, and add `edit-bar-clickable` to its className when `onLayer` is set.

`LayerVotesCard.tsx`: add `onLayer?: (layer: number) => void` to `LayerVotesCard`'s props and pass it to `VotesBody`. Change the vote cell's `onClick` to `() => { onSelect(r); onLayer?.(l) }`.

CSS (`src/index.css`, after the `.mlp-cand` rules): `.edit-bar-clickable, .votes-cells > div, .mlp-cells > div[data-testid="edit-cell"] { cursor: pointer; }`

- [ ] **Step 5: Run the tests**

Run: `npm test`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/viz src/test src/index.css
git commit -m "feat(mlp): shared layer selection, cycleNeurons selector, layer clicks in both MLP cards, neuron trace fixture"
```

---

### Task 10: Glossary entry and callouts

**Files:**
- Create: `src/glossary/entries/mlp-neuron.tsx`
- Modify: `src/glossary/types.ts`, `src/glossary/registry.ts`, `src/glossary/callouts.ts`, `src/glossary/entries/mlp.tsx`, `src/glossary/entries/votes.tsx`, `src/glossary/registry.test.tsx`

**Interfaces:**
- Produces: glossary id `'mlp-neuron'` with sections `rarely-names`, `negative`, `early-layers`; `NEURON_CALLOUTS` in `callouts.ts`.

- [ ] **Step 1: Write the failing tests** (in `src/glossary/registry.test.tsx`: add `NEURON_CALLOUTS` to the callouts import and to the array in "every callout chip points at a section that exists"; then append)

```tsx
test('the MLP neuron entry explains the split step by step and links from MLP and votes', () => {
  const { container } = render(<>{getEntry('mlp-neuron')!.body()}</>)
  expect(container.querySelectorAll('ol.gl-steps > li')).toHaveLength(4)
  expect(sectionsOf('mlp-neuron')).toEqual(['rarely-names', 'negative', 'early-layers'])
  expect(getEntry('mlp')!.related).toContain('mlp-neuron')
  expect(getEntry('votes')!.related).toContain('mlp-neuron')
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/glossary`
Expected: FAIL, no entry for `mlp-neuron` (and the registry order test fails).

- [ ] **Step 3: Implement**

`src/glossary/entries/mlp-neuron.tsx`:

```tsx
import { Formula } from '../../viz/Formula'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const mlpNeuron: GlossaryEntry = {
  id: 'mlp-neuron',
  title: 'MLP neuron',
  summary: 'One of the 1,536 positions inside an MLP: how strongly it fires decides how much of its column is added to the vector.',
  stage: 'layers',
  related: ['mlp', 'votes', 'edit'],
  body: () => (
    <>
      <p>Inside each <Term id="mlp">MLP</Term>, the token’s vector is widened to 1,536 numbers and narrowed back to
        576. Each of the 1,536 positions is called a neuron. Here is how one neuron changes the vector, and how
        the card splits an MLP’s vote among them.</p>
      <ol className="gl-steps">
        <li><strong>The activation.</strong> The neuron’s number for this token, SiLU(gate) × up, is its
          activation. It can be large, near zero, or negative.</li>
        <li><strong>The column.</strong> Each neuron owns one column of the down matrix, 576 numbers. The MLP’s{' '}
          <Term id="edit">edit</Term> is the sum over all neurons of activation × column, so every neuron adds its
          column, scaled by how strongly it fired.</li>
        <li><strong>The vote.</strong> A sum of edits splits into a sum of <Term id="votes">votes</Term>, so the
          MLP’s vote for a candidate is the sum of 1,536 neuron votes: activation × (column · the candidate’s row),
          scaled by the final norm like every other piece.</li>
        <li><strong>The label.</strong> To name what a column writes, it is compared with every row of the embedding
          table (each scaled by the final norm’s weight), by cosine. The five closest rows are the tokens the
          neuron pushes toward when it fires positively; the five closest to the flipped column, when it fires
          negatively.</li>
      </ol>
      <Formula tex={String.raw`\text{vote}_i(c) = h_i \,\bigl(d_i \cdot (g \odot E_c)\bigr) \,/\, \operatorname{rms}(x)`} />
      <p>Here <Formula inline tex="h_i" /> is neuron <Formula inline tex="i" />’s activation,{' '}
        <Formula inline tex="d_i" /> its column of the down matrix, <Formula inline tex="g" /> the final norm’s
        weight and <Formula inline tex="E_c" /> the candidate’s row of E.</p>
      <p>The badge says where the candidate falls in the neuron’s full ranking of the 49,152 tokens: among the top
        0.1 %, 1 % or 10 %, or further down.</p>
      <Section id="rarely-names" title="Why the list rarely names the candidate">
        <p>A neuron’s vote is only the part of its push that points along the candidate’s row. A neuron can push
          hard toward its own tokens and still add to the candidate’s score through a small share of that push,
          and a large activation makes even a small share count. Votes are also spread out: the top few neurons
          usually carry well under half of a layer’s vote.</p>
      </Section>
      <Section id="negative" title="Negative activations">
        <p>SiLU lets gate values go slightly below zero, and the up value can be negative too, so an activation is
          often negative. Then the neuron adds its column flipped, and the tokens it pushes toward are the ones
          closest to the flipped column. The card shows the list for the sign the neuron actually fired with.
          When a neuron’s vote is negative, the card shows what it pushes away from instead.</p>
      </Section>
      <Section id="early-layers" title="Early layers">
        <p>In early layers the votes are small and the token lists look like noise. The vector is still far from
          the space the readout compares with, so a column’s closest rows of E say little about what the neuron
          does there.</p>
      </Section>
      <p className="gl-aside">The browser uses an 8-bit copy of the down matrix for these numbers. When rounding
        makes the neuron votes miss the votes card’s number by more than a small tolerance, the total shows “≈”.</p>
    </>
  ),
}
```

`src/glossary/registry.ts`: import `mlpNeuron` and insert it after `votes` in `ENTRIES`. `src/glossary/types.ts`: add `'mlp-neuron'` after `'votes'` in `GLOSSARY_IDS`.

`src/glossary/callouts.ts`:

```ts
export const NEURON_CALLOUTS: CalloutTarget[] = [
  { label: 'why the list rarely names the candidate', id: 'mlp-neuron', section: 'rarely-names' },
  { label: 'negative activations', id: 'mlp-neuron', section: 'negative' },
  { label: 'early layers', id: 'mlp-neuron', section: 'early-layers' },
]
```

`src/glossary/entries/mlp.tsx`: set `related: ['layer', 'edit', 'attention', 'votes', 'mlp-neuron']`. Replace the final paragraph with:

```tsx
      <p>In a real run the Layers detail shows what the MLPs do: <em>How much each layer changes the vector</em>{' '}
        compares each MLP edit with the stream, <em>What each layer voted for</em> shows how much each layer’s
        MLP added to each candidate’s score (see <Term id="votes">per-layer votes</Term>), and{' '}
        <em>Inside one MLP</em> splits one layer’s vote among its <Term id="mlp-neuron">neurons</Term>.</p>
```

`src/glossary/entries/votes.tsx`: set `related: ['logit', 'mlp', 'edit', 'mlp-neuron']`. After the paragraph starting "The votes card shows these votes per layer", add:

```tsx
      <p>An MLP’s vote is itself a sum: one vote per <Term id="mlp-neuron">neuron</Term>. The card below the
        votes card lists the neurons behind one layer’s vote.</p>
```

- [ ] **Step 4: Run the tests**

Run: `npm test && npx tsc -b && npm run lint`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/glossary
git commit -m "docs(neurons): MLP neuron glossary entry with callout sections, linked from MLP and votes"
```

---

### Task 11: The card, README and screenshot

**Files:**
- Create: `src/viz/NeuronCard.tsx`, `src/viz/NeuronCard.test.tsx`, `docs/screenshots/neurons.png`
- Modify: `src/viz/details/LayersDetail.tsx`, `src/viz/DetailPanel.tsx`, `src/viz/DetailPanel.test.tsx`, `src/index.css`, `README.md`, `scripts/capture-screenshots.mjs`, `docs/screenshots/mlp.png`

**Interfaces:**
- Consumes: everything from Tasks 7–9; `NEURON_CALLOUTS` and the `'mlp-neuron'` glossary id (Task 10); `useCandidateSelection` and `selectedCandidate`; `topMlpLayers` from `./mlpMath`; `visibleToken`.
- Produces: `NeuronCard({ data, candidates, note, run })` with test ids:
  - container and header: `neuron-card`, `neuron-title`, `neuron-prev`, `neuron-next`, `neuron-info`;
  - controls: `neuron-mode-toggle`, `neuron-cand`;
  - states: `neuron-note`, `neuron-loading`, `neuron-error`, `neuron-retry`;
  - table: `neuron-table`, `neuron-row`, `neuron-vote`, `neuron-label`, `neuron-away`, `neuron-badge`, `neuron-rest`, `neuron-total`, `neuron-approx-note`, `neuron-callout`.

- [ ] **Step 1: Write the failing tests** (`src/viz/NeuronCard.test.tsx`)

```tsx
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import * as asset from '../neurons/asset'
import { buildFixtureTrace } from '../test/fixtures'
import { fixtureNeuronAsset, fixtureNeuronLayer, NF } from '../test/neuronFixture'
import type { NeuronData, TraceEvent } from '../trace/types'
import { NeuronCard } from './NeuronCard'
import { cycleNeurons, cycleVotes, type VotedCandidate } from './selectors'
import { useMlpLayer } from './mlpLayerSelection'

const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
const last = events.length - 1
const data = (cycleNeurons(events, last) as { data: NeuronData }).data
const candidates = (cycleVotes(events, last) as { candidates: VotedCandidate[] }).candidates
const run = events[0]

const card = (props: Partial<Parameters<typeof NeuronCard>[0]> = {}) =>
  render(<NeuronCard data={data} candidates={candidates} note="the note" run={run} {...props} />)

beforeEach(() => {
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  vi.spyOn(asset, 'loadNeuronLayer').mockImplementation(async (l) => fixtureNeuronLayer(l))
})
afterEach(() => vi.restoreAllMocks())

const largestLayer = (c: VotedCandidate) => c.votes.mlp.reduce((best, v, l, a) => (Math.abs(v) > Math.abs(a[best]) ? l : best), 0)

test('without data it shows the note it is given', () => {
  card({ data: undefined })
  expect(screen.getByTestId('neuron-note')).toHaveTextContent('the note')
})

test('opens on the layer with the largest MLP vote, shows loading, then 8 rows that add up to the votes card', async () => {
  card()
  const layer = largestLayer(candidates[0])
  expect(screen.getByTestId('neuron-title')).toHaveTextContent(`Inside one MLP: layer ${layer}`)
  expect(screen.getByTestId('neuron-loading')).toHaveTextContent('Loading the MLP weights (about 0.9 MB per layer, once per model version)…')
  await waitFor(() => expect(screen.getAllByTestId('neuron-row')).toHaveLength(8))
  expect(screen.getByTestId('neuron-rest')).toHaveTextContent(`the other ${NF.neurons - 8} neurons`)
  const cell = candidates[0].votes.mlp[layer]
  expect(screen.getByTestId('neuron-total')).toHaveTextContent(`${cell >= 0 ? '+' : '−'}${Math.abs(cell).toFixed(2)}`)
  expect(screen.getByTestId('neuron-total')).toHaveTextContent('= the votes card’s cell')
  expect(screen.queryByTestId('neuron-approx-note')).toBeNull()
})

test('labels show as visible tokens, negative votes get an away chip and an (away) badge', async () => {
  card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  expect(screen.getAllByTestId('neuron-label')[0].textContent).toMatch(/^(w\d+|␣and)$/)
  for (const r of screen.getAllByTestId('neuron-row')) {
    const negative = r.querySelector('[data-testid="neuron-vote"]')!.textContent!.startsWith('−')
    expect(!!r.querySelector('[data-testid="neuron-away"]')).toBe(negative)
    expect(r.querySelector('[data-testid="neuron-badge"]')!.textContent!.endsWith('(away)')).toBe(negative)
  }
})

test('the stepper moves one layer and stops at the ends', async () => {
  useMlpLayer.getState().select(run, 0)
  card()
  expect(screen.getByTestId('neuron-prev')).toBeDisabled()
  fireEvent.click(screen.getByTestId('neuron-next'))
  expect(screen.getByTestId('neuron-title')).toHaveTextContent('layer 1')
  fireEvent.click(screen.getByTestId('neuron-next'))
  expect(screen.getByTestId('neuron-next')).toBeDisabled()
})

test('a layer picked in another card opens here', () => {
  useMlpLayer.getState().select(run, 1)
  card()
  expect(screen.getByTestId('neuron-title')).toHaveTextContent('layer 1')
})

test('firing most drops the candidate, the badge and the total', async () => {
  card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  fireEvent.click(screen.getAllByTestId('neuron-mode-toggle')[1])
  expect(screen.getByTestId('neuron-table')).toHaveTextContent('edit size')
  expect(screen.queryByTestId('neuron-badge')).toBeNull()
  expect(screen.queryByTestId('neuron-total')).toBeNull()
  expect(screen.queryByTestId('neuron-cand')).toBeNull()
})

test('picking another candidate changes the vote column', async () => {
  card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  fireEvent.click(screen.getAllByTestId('neuron-cand')[1])
  expect(screen.getByTestId('neuron-table')).toHaveTextContent(`vote for “${candidates[1].text.replace(/^ /, '␣')}”`)
})

test('a total off by more than the tolerance shows ≈ and the note', async () => {
  const off = candidates.map((c, i) => i === 0 ? { ...c, votes: { ...c.votes, mlp: c.votes.mlp.map((v) => v + 5) } } : c)
  card({ candidates: off })
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  expect(screen.getByTestId('neuron-total')).toHaveTextContent('≈')
  expect(screen.getByTestId('neuron-approx-note')).toHaveTextContent('The 8-bit weights are slightly off for this layer')
})

test('a load error offers retry', async () => {
  vi.mocked(asset.loadNeuronLayer).mockRejectedValueOnce(new Error('HTTP 503'))
  card()
  await waitFor(() => expect(screen.getByTestId('neuron-error')).toHaveTextContent('Couldn’t load the MLP weights.'))
  fireEvent.click(screen.getByTestId('neuron-retry'))
  await waitFor(() => expect(screen.getAllByTestId('neuron-row')).toHaveLength(8))
})

test('neuron data that does not match the weights file shows a note instead of throwing', async () => {
  card({ data: { ...data, neurons: 7 } })
  await waitFor(() => expect(screen.getByTestId('neuron-note')).toHaveTextContent('This run’s neuron data doesn’t match the MLP weights file.'))
})

test('a candidate without a row shows a note instead of throwing', async () => {
  card({ data: { ...data, candidates: data.candidates.slice(1) } })
  await waitFor(() => expect(screen.getByTestId('neuron-note')).toHaveTextContent('No neuron data for this candidate.'))
})

test('the run-start of another trace resets a pick', () => {
  useMlpLayer.getState().select(buildFixtureTrace()[0] as TraceEvent, 2)
  card()
  expect(screen.getByTestId('neuron-title')).toHaveTextContent(`layer ${largestLayer(candidates[0])}`)
})
```

Append to `src/viz/DetailPanel.test.tsx`:

```tsx
test('real mode: the Layers detail shows the neuron card under the two MLP cards; sim mode shows its note', () => {
  const real = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
  const at = real.findIndex((e) => e.type === 'layer')
  const { unmount } = render(<DetailPanel events={real} cursor={at} mode="real" />)
  const cards = screen.getByTestId('mlp-cards').querySelectorAll('section')
  expect([...cards].map((s) => s.dataset.testid)).toEqual(['edit-size-card', 'votes-card', 'neuron-card'])
  unmount()
  render(<DetailPanel events={real} cursor={at} mode="sim" />)
  expect(screen.getByTestId('neuron-note')).toHaveTextContent('Real model only: simulated runs have no neurons to show.')
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/viz/NeuronCard.test.tsx src/viz/DetailPanel.test.tsx`
Expected: FAIL, `Cannot find module './NeuronCard'`.

- [ ] **Step 3: Implement**

`src/viz/NeuronCard.tsx`:

```tsx
import { useMemo, useState } from 'react'
import { NEURON_CALLOUTS } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import type { NeuronAsset } from '../neurons/asset'
import type { NeuronLayer } from '../neurons/format'
import { behindTheVote, firingMost, layerActivations, reconcile, type VoteRow } from '../neurons/math'
import { useNeuronLayer } from '../neurons/useNeuronLayer'
import { unpackFloat32, unpackHalf } from '../trace/neuronPack'
import type { NeuronData, TraceEvent } from '../trace/types'
import { selectedCandidate, useCandidateSelection } from './candidateSelection'
import { selectedLayer, useMlpLayer } from './mlpLayerSelection'
import { topMlpLayers } from './mlpMath'
import type { VotedCandidate } from './selectors'
import { visibleToken } from './tokenText'

type Mode = 'vote' | 'size'
const MODE_NAME: Record<Mode, string> = { vote: 'behind the vote', size: 'firing most' }
const LOADING = 'Loading the MLP weights (about 0.9 MB per layer, once per model version)…'
const MISMATCH = 'This run’s neuron data doesn’t match the MLP weights file.'
const NO_ROW = 'No neuron data for this candidate.'
const APPROX = 'The 8-bit weights are slightly off for this layer, so the neuron votes don’t add up exactly to the votes card.'
const BAR_MAX = 70

const signed = (v: number, digits = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`
const BADGE_CLASS: Record<string, string> = { 'top 0.1%': 'strong', 'top 1%': 'mid', 'top 10%': 'low', 'not in top 10%': 'none' }

// Decoding 46 080 float16 values once per cycle's data, not per render.
const decoded = new WeakMap<NeuronData, Float32Array>()
const activationsOf = (d: NeuronData) => {
  let a = decoded.get(d)
  if (!a) { a = unpackHalf(d.activations); decoded.set(d, a) }
  return a
}

// "Inside one MLP": one layer's MLP vote for the selected candidate split into
// one vote per neuron (#16). Candidate selection is shared with the Logits and
// votes cards, the layer with both MLP cards; the mode is local view state.
export function NeuronCard({ data, candidates, note, run }: {
  data?: NeuronData; candidates?: VotedCandidate[]; note: string; run?: TraceEvent
}) {
  const pick = useCandidateSelection((s) => s.pick)
  const layerPick = useMlpLayer((s) => s.pick)
  const selectLayer = useMlpLayer((s) => s.select)
  const ready = data && candidates && candidates.length > 0
  const sel = ready ? selectedCandidate(pick, candidates) : 0
  const layers = ready ? candidates[0].votes.mlp.length : 0
  const layer = ready ? selectedLayer(layerPick, run) ?? topMlpLayers(candidates[sel].votes, 1)[0].layer : null
  const step = (d: number) => { if (run && layer !== null) selectLayer(run, Math.min(layers - 1, Math.max(0, layer + d))) }
  return (
    <section data-testid="neuron-card" className="mlp-card">
      <h4 data-testid="neuron-title">Inside one MLP{layer !== null ? `: layer ${layer}` : ''}{layer !== null && (
        <span className="neuron-stepper">
          <button type="button" data-testid="neuron-prev" aria-label="Previous layer" disabled={layer <= 0} onClick={() => step(-1)}>◀</button>
          <button type="button" data-testid="neuron-next" aria-label="Next layer" disabled={layer >= layers - 1} onClick={() => step(1)}>▶</button>
        </span>
      )}{' '}<Term id="mlp-neuron" variant="icon" label="About MLP neurons" testId="neuron-info">ⓘ</Term></h4>
      {!ready || layer === null ? <p data-testid="neuron-note" className="mlp-note">{note}</p>
        : <NeuronBody data={data} candidates={candidates} selected={sel} layer={layer} />}
    </section>
  )
}

function NeuronBody({ data, candidates, selected, layer }: {
  data: NeuronData; candidates: VotedCandidate[]; selected: number; layer: number
}) {
  const [mode, setMode] = useState<Mode>('vote')
  const select = useCandidateSelection((s) => s.select)
  const state = useNeuronLayer(layer)
  const cand = candidates[selected]
  return (
    <>
      <p className="mlp-caption">{mode === 'vote'
        ? 'The MLP’s edit is the sum of 1,536 neuron edits, so its vote splits into one vote per neuron. Click a layer in either card above to open it here.'
        : 'The neurons that changed the vector most for the last token, whatever they voted for. Size = |activation| × length of the neuron’s column.'}</p>
      <div className="mlp-controls">
        <div className="mlp-seg" role="group" aria-label="Which neurons">
          {(['vote', 'size'] as const).map((m) => (
            <button key={m} type="button" data-testid="neuron-mode-toggle" aria-pressed={m === mode}
              onClick={() => setMode(m)}>{MODE_NAME[m]}</button>
          ))}
        </div>
        {mode === 'vote' && (
          <div className="neuron-cands" role="group" aria-label="Candidate">candidate:{' '}
            {candidates.map((c, i) => (
              <button key={c.id} type="button" data-testid="neuron-cand" className="mlp-cand" aria-pressed={i === selected}
                onClick={() => select(candidates, i)}>{visibleToken(c.text)}</button>
            ))}
          </div>
        )}
      </div>
      {state.status === 'loading' || state.status === 'idle' ? <p data-testid="neuron-loading" className="mlp-note">{LOADING}</p>
        : state.status === 'error' ? (
          <p data-testid="neuron-error" className="mlp-note">Couldn’t load the MLP weights.{' '}
            <button type="button" data-testid="neuron-retry" onClick={state.retry}>Retry</button></p>
        ) : <NeuronTable data={data} asset={state.asset} weights={state.layer} layer={layer} mode={mode} cand={cand} />}
      <div className="embed-callouts">
        {NEURON_CALLOUTS.map((c) => (
          <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="neuron-callout">ⓘ {c.label}</Term>
        ))}
      </div>
    </>
  )
}

function NeuronTable({ data, asset, weights, layer, mode, cand }: {
  data: NeuronData; asset: NeuronAsset; weights: NeuronLayer; layer: number; mode: Mode; cand: VotedCandidate
}) {
  const m = asset.manifest
  const fits = data.neurons === m.neurons && activationsOf(data).length === m.layers * m.neurons
  const rowB64 = data.candidates.find((c) => c.id === cand.id)?.row
  const acts = fits ? layerActivations(activationsOf(data), layer, m.neurons) : null
  const table = useMemo(() => {
    if (!acts || mode !== 'vote' || !rowB64) return null
    const row = unpackFloat32(rowB64)
    if (row.length !== m.dims) return null
    return reconcile(behindTheVote(m, weights, acts, row, asset.g, data.invRms), cand.votes.mlp[layer])
  }, [acts, mode, rowB64, m, weights, asset.g, data.invRms, cand, layer])
  if (!acts) return <p data-testid="neuron-note" className="mlp-note">{MISMATCH}</p>
  const chips = (ids: number[]) => ids.map((id, j) => (
    <span key={j} data-testid="neuron-label" className="neuron-chip">{visibleToken(asset.text(id))}</span>
  ))
  if (mode === 'size') {
    const rows = firingMost(m, weights, acts)
    const peak = Math.max(...rows.map((r) => r.size)) || 1
    return (
      <table data-testid="neuron-table" className="neuron-table">
        <thead><tr><th>neuron</th><th>activation</th><th>edit size</th><th>when it fires like this, it pushes toward</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.neuron} data-testid="neuron-row">
            <td className="tk">N{r.neuron}</td><td className="tk">{signed(r.activation, 1)}</td>
            <td><span className="neuron-size-bar" style={{ width: `${(r.size / peak) * BAR_MAX}px` }} />{r.size.toFixed(2)}</td>
            <td>{chips(r.labels)}</td>
          </tr>
        ))}</tbody>
      </table>
    )
  }
  if (!table) return <p data-testid="neuron-note" className="mlp-note">{NO_ROW}</p>
  const name = visibleToken(cand.text)
  const peak = Math.max(...table.rows.map((r) => Math.abs(r.vote))) || 1
  return (
    <>
      <table data-testid="neuron-table" className="neuron-table">
        <thead><tr><th>neuron</th><th>activation</th><th>vote for “{name}”</th>
          <th>when it fires like this, it pushes toward</th><th>“{name}” in that list</th></tr></thead>
        <tbody>
          {table.rows.map((r: VoteRow) => (
            <tr key={r.neuron} data-testid="neuron-row">
              <td className="tk">N{r.neuron}</td><td className="tk">{signed(r.activation, 1)}</td>
              <td><span className={`votes-bar ${r.vote >= 0 ? 'votes-bar-pos' : 'votes-bar-neg'}`}
                style={{ width: `${(Math.abs(r.vote) / peak) * BAR_MAX}px` }} /><span data-testid="neuron-vote">{signed(r.vote)}</span></td>
              <td>{r.away && <span data-testid="neuron-away" className="neuron-chip neuron-away">away from</span>}{chips(r.labels)}</td>
              <td><span data-testid="neuron-badge" className={`neuron-badge neuron-badge-${BADGE_CLASS[r.badge]}`}>
                {r.badge}{r.away ? ' (away)' : ''}</span></td>
            </tr>
          ))}
          <tr data-testid="neuron-rest" className="neuron-rest">
            <td colSpan={2}>the other {(m.neurons - table.rows.length).toLocaleString('en-US')} neurons</td>
            <td colSpan={3}>{signed(table.rest)}</td></tr>
          <tr data-testid="neuron-total" className="neuron-total">
            <td colSpan={2}>layer {layer} MLP, total</td>
            <td colSpan={3}>{table.exact ? signed(table.total) : `≈ ${signed(table.total)}`}{table.exact && (
              <span className="mlp-legend-note"> = the votes card’s cell</span>)}</td></tr>
        </tbody>
      </table>
      {!table.exact && <p data-testid="neuron-approx-note" className="mlp-scale-note">{APPROX}</p>}
    </>
  )
}
```

(`useMemo` sits above the early returns, so the hook order is stable. `acts` is a fresh subarray each render, so the memo recomputes each render, which is acceptable at about 1 ms per layer. If profiling shows otherwise, memoize `acts` with `useMemo([data, layer, fits])`.)

In `src/viz/details/LayersDetail.tsx`:
- import `NeuronCard`, `useMlpLayer` and the `CycleNeurons` type;
- add the constant `const NEURON_SIM_NOTE = 'Real model only: simulated runs have no neurons to show.'` and `const NEURON_OLD_NOTE = 'This run was recorded before neuron data existed.'`;
- add props `neurons?: CycleNeurons` and `run?: TraceEvent`;
- inside the component: `const selectLayer = useMlpLayer((s) => s.select)` and `const onLayer = run ? (l: number) => selectLayer(run, l) : undefined`;
- pass `onLayer={onLayer}` to `EditSizeCard` and `LayerVotesCard`, and after `LayerVotesCard` add:

```tsx
        <NeuronCard
          data={mode === 'real' && neurons?.status === 'ok' ? neurons.data : undefined}
          candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined}
          note={mode === 'sim' ? NEURON_SIM_NOTE
            : neurons?.status === 'pending' || votes?.status === 'pending' ? PENDING_NOTE : NEURON_OLD_NOTE}
          run={run} />
```

In `src/viz/DetailPanel.tsx`: import `cycleNeurons`, and add `neurons={cycleNeurons(events, cursor)} run={events[0]}` to the `LayersDetail` element.

CSS (`src/index.css`, after the votes-card rules):

```css
.neuron-stepper { display: inline-flex; gap: .25rem; margin-left: .5rem; vertical-align: middle; }
.neuron-stepper button { font: inherit; font-size: .75rem; padding: 0 .4rem; border: 1px solid var(--line-strong); border-radius: 4px; background: var(--card); color: var(--ink); }
.neuron-stepper button:disabled { opacity: .4; }
.neuron-cands { display: flex; flex-wrap: wrap; gap: .25rem; align-items: center; font-size: .8rem; color: var(--ink-soft); }
.neuron-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
.neuron-table th { text-align: left; font-weight: 600; color: var(--ink-soft); font-size: .75rem; padding: .3rem .4rem; border-bottom: 1px solid var(--line); }
.neuron-table td { padding: .3rem .4rem; border-bottom: 1px solid var(--line); vertical-align: middle; }
.neuron-table .tk { font-family: var(--mono, ui-monospace, Menlo, monospace); font-size: .78rem; white-space: nowrap; }
.neuron-table .votes-bar { margin-right: .35rem; vertical-align: middle; }
.neuron-chip { display: inline-block; white-space: pre; padding: 0 .45rem; margin: .1rem .15rem; border-radius: 999px; background: var(--ai-wash); font-size: .78rem; }
.neuron-away { background: var(--shu-wash); }
.neuron-badge { font-size: .72rem; padding: .05rem .4rem; border-radius: 4px; white-space: nowrap; background: var(--line); }
.neuron-badge-strong { background: var(--ai); color: #fff; }
.neuron-badge-mid { background: var(--ai-wash-strong); }
.neuron-rest td { color: var(--ink-soft); font-style: italic; }
.neuron-total td { font-weight: 600; border-bottom: 0; }
.neuron-size-bar { display: inline-block; height: 10px; background: var(--ink-soft); margin-right: .35rem; vertical-align: middle; }
@media (max-width: 640px) { .neuron-table { display: block; overflow-x: auto; } }
```

(Check that each variable used — `--ai-wash`, `--ai-wash-strong`, `--shu-wash`, `--ink-soft`, `--ai`, `--card`, `--line`, `--line-strong` — is defined in `:root` by grepping `index.css`. Replace any missing one with the nearest defined token, and do the same for the dark-mode block.)

- [ ] **Step 4: Run the tests and the type check**

Run: `npm test && npx tsc -b && npm run lint`
Expected: all green.

- [ ] **Step 5: Compare with the approved mockup**

Run `npm run dev`. In real mode, generate "The cat sat on the mat because it was tired" with max tokens 1, open Layers and compare the card with `.superpowers/brainstorm/66126-1791926765/content/card.html` (open the file in the browser). Check:
- the column order and header texts;
- the chip style and the "away from" chip;
- the badge colours: strong for top 0.1%, mid for top 1%;
- the rest and total lines;
- the callout chips;
- the stepper placement.

Fix any differences before committing.

- [ ] **Step 6: README and screenshot script**

`README.md`: rename the section heading to `## Opening the MLP — edit sizes, per-layer votes and neurons`. Before the "The empty layer-11 column" paragraph, add:

```markdown
**Inside one MLP** opens one layer: its MLP edit is the sum of 1,536 neuron
edits, so its vote for a candidate splits into one vote per neuron. The card
lists the eight neurons behind the vote (or, toggled, the eight that changed
the vector most), each with its activation, its vote, the tokens its column
points at when it fires that way, and where the candidate falls in that
ranking. The rest of the neurons and the total close the table, and the
total equals the votes card's cell. Click a layer in either card above to
open it, or step with ◀ ▶. The 8-bit weights it needs download once per
model version (about 0.9 MB per layer) and stay in the browser's cache.

![The Inside one MLP card](docs/screenshots/neurons.png)
```

`scripts/capture-screenshots.mjs`: add `neurons` to the scene list in the comment and to the condition `if (want('attention') || … || want('mlp'))` (append `|| want('neurons')`). After the `mlp` scene, add:

```js
  // ---- neurons.png: real mode, Layers stage — Inside one MLP ----------------
  if (want('neurons')) {
    await $('prompt-input').fill('The cat sat on the mat because it was tired')
    await $('maxtok-input').fill('1')
    const before = await $('run-chip').count()
    await $('btn-generate').click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })
    await $('btn-live').click()
    await $('stage-card').nth(2).click()                            // Layers
    await $('neuron-row').first().waitFor({ timeout: 30000 })
    const unstick = await page.addStyleTag({ content: '.controls { position: static !important; }' })
    await page.waitForTimeout(300)
    await $('neuron-card').screenshot({ path: `${DIR}/neurons.png` })
    await unstick.evaluate((el) => el.remove())
    await $('maxtok-input').fill('20')
    console.log('neurons.png')
  }
```

Also change "~120 MB real model" in the script's header comment to "~250 MB real model".

- [ ] **Step 7: Capture the screenshots**

Run: `npm run screenshots -- neurons mlp`
Expected: `neurons.png` and `mlp.png` are written. Open `docs/screenshots/neurons.png` and check that it shows the table with labels and badges. The `mlp` scene is recaptured because the mlp-cards container now includes the third card.

- [ ] **Step 8: Commit**

```bash
git add src/viz src/index.css README.md scripts/capture-screenshots.mjs docs/screenshots/neurons.png docs/screenshots/mlp.png
git commit -m "feat(mlp): Inside one MLP card — neurons behind a layer's vote or firing most, with labels, badges and an exact total"
```

---

### Task 12: Archived runs use their own mode; browser test

**Files:**
- Modify: `src/App.tsx`, `src/App.test.tsx`
- Create: `e2e/neurons.spec.ts`

**Interfaces:**
- Consumes: `buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })` (Task 9), `neuronFixtureFile` (Task 7).

- [ ] **Step 1: Write the failing tests**

Append to `src/App.test.tsx` (follow the file's existing setup; it already renders `App` with a mocked tokenizer):

```tsx
test('an archived real run shows its MLP cards while the app is in sim mode', async () => {
  const record = archiveImport({ meta: { prompt: 'The cat', params: { temperature: 0.8, topK: 10, maxNewTokens: 2 },
    mode: 'real', modelId: 'fixture', endedAt: 1, reason: 'max-tokens' },
    events: buildFixtureTrace({ mode: 'real', mlp: true }) })
  render(<App />)
  fireEvent.click(within(screen.getAllByTestId('run-chip').find((c) => c.textContent?.includes('The cat'))!).getByTestId('run-chip-main'))
  fireEvent.click(await screen.findAllByTestId('stage-card').then((s) => s[2]))
  expect(await screen.findByTestId('vote-cell', {}, { timeout: 2000 })).toBeInTheDocument()
  expect(record.id).toBeTruthy()
})
```

(Import `archiveImport` from `./app/runArchive`, `buildFixtureTrace` from `./test/fixtures`, and `within` from `@testing-library/react`. If `App.test.tsx` resets the runs store in `beforeEach`, the import must happen after that reset, which it does inside the test.)

`e2e/neurons.spec.ts`:

```ts
import { expect, test } from '@playwright/test'
import { buildFixtureTrace } from '../src/test/fixtures'
import { neuronFixtureFile } from '../src/test/neuronFixture'

test('an imported real run opens Inside one MLP with a total equal to the votes card cell', async ({ page }) => {
  await page.route('**/neurons/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').pop()!
    const body = neuronFixtureFile(name)
    await route.fulfill(body
      ? { status: 200, body: Buffer.from(body), headers: { 'access-control-allow-origin': '*' },
          contentType: name.endsWith('.json') ? 'application/json' : 'application/octet-stream' }
      : { status: 404, headers: { 'access-control-allow-origin': '*' } })
  })
  await page.goto('/')
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, prompt: 'neuron fixture' })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'neuron fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 2 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(), reason: 'max-tokens' } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  await page.getByTestId('run-chip').filter({ hasText: 'neuron fixture' }).getByTestId('run-chip-main').click()
  await page.getByTestId('stage-card').nth(2).click()                      // Layers

  const card = page.getByTestId('neuron-card')
  await expect(card.getByTestId('neuron-row')).toHaveCount(8)
  const layer = Number((await card.getByTestId('neuron-title').textContent())!.match(/layer (\d+)/)![1])
  const cell = await page.locator(`[data-testid="vote-cell"][data-row="0"][data-layer="${layer}"]`).getAttribute('data-value')
  const total = (await card.getByTestId('neuron-total').textContent())!.replace('−', '-')
  expect(Number(total.match(/[+-]\d+\.\d{2}/)![0])).toBeCloseTo(Number(cell), 2)

  await card.getByTestId('neuron-mode-toggle').nth(1).click()
  await expect(card.getByTestId('neuron-table')).toContainText('edit size')
  const stepper = layer < 2 ? 'neuron-next' : 'neuron-prev'
  await card.getByTestId(stepper).click()
  await expect(card.getByTestId('neuron-title')).toContainText(`layer ${layer < 2 ? layer + 1 : layer - 1}`)
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/App.test.tsx && npx playwright test e2e/neurons.spec.ts`
Expected: the App test fails (the votes card shows the simulated note, so `vote-cell` is never found). The e2e fails at `neuron-row` for the same reason.

- [ ] **Step 3: Implement** in `src/App.tsx`, next to where `events` is read for the detail panel:

```tsx
  // An archived run is shown as recorded: a real run keeps its MLP cards even
  // while the app itself is in sim mode, and vice versa.
  const viewMode: Mode = events[0]?.type === 'run-start' ? events[0].mode : mode
```

and pass `mode={viewMode}` to `DetailPanel` (only there; `PromptBar` and `ModelStatus` keep `mode`).

- [ ] **Step 4: Run all tests**

Run: `npm test && npx playwright test && npx tsc -b && npm run lint`
Expected: every unit test, all 5 e2e tests and the type check pass, and lint is clean.

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.test.tsx e2e/neurons.spec.ts
git commit -m "fix(app): archived runs show their own mode's cards; e2e for Inside one MLP on an imported run"
```

---

### Task 13: Live check

**Files:** none (verification only).

- [ ] **Step 1:** Run `npm run dev`. In real mode, run "The cat sat on the mat because it was tired" with max tokens 1. On Layers, select candidate "␣and" and step to layer 29.
- [ ] **Step 2:** Compare with the probe's fp32 numbers:
  - N1233: act −28.25, vote +1.27;
  - N1512: −24.95, +1.27;
  - N288: +24.36, +1.19;
  - N1066: −13.28, +0.90;
  - N600: −3.59, −0.88.

  The q8 model may move them a little: votes within about 0.1, activations within about 10 %. The same neurons should lead. The total must show no "≈". Record the observed rows in the ledger.
- [ ] **Step 3:** Check the loading note on the first open and that stepping back to a viewed layer is instant (served from cache). Reload the page, open the card again, and check in DevTools → Network that layer files come from the service cache (no Hub requests for files already viewed).
- [ ] **Step 4:** Check a simulated run (sim note) and an older archived real run, recorded before v5 if one is on the shelf ("This run was recorded before neuron data existed.").
