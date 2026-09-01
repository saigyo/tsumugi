# Embeddings Explained (M4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fake-numbers Embeddings card with one that teaches the lookup mechanism and vocabulary geometry, backed by real embedding rows from a re-exported ONNX graph and a Hub-hosted geometry asset.

**Architecture:** The Python export tool gains an `inputs_embeds` graph output and a `geometry` command that computes a static asset (exact top-12 neighbours, PCA-64 int8 vectors, token texts, manifest) published next to the model. In the app, the `embed` trace event carries real rows in real mode (`source: 'model'`) or nothing (`source: 'asset'`); a lazily fetched `src/geometry/` module resolves vectors, neighbours and similarities in the view, so the UI stays a pure function of `(trace, cursor, asset)`.

**Tech Stack:** Python 3.10+ / uv / optimum / onnx / onnxruntime / numpy / safetensors (tool); React 19 + TypeScript + Vite + vitest + testing-library + Playwright (app).

**Spec:** `docs/superpowers/specs/2026-09-02-embeddings-explained-design.md`

## Global Constraints

- ONNX output name for the embedding lookup: exactly `inputs_embeds`, shape `[batch, sequence, 576]`. Existing outputs (`logits`, `present.*`, `attentions.*`) keep their names and axes — they are the artifact contract.
- Geometry asset files: `manifest.json`, `neighbors.bin`, `vectors.bin`, `tokens.json` in `geometry/` of `saigyo-hoshi/smollm2-135m-attn-onnx`. `k = 12`, `pcaDims = 64`, sims stored as `round(max(cos, 0) × 255)` uint8, neighbour ids little-endian uint16, ids block then sims block, the token itself excluded, sorted by similarity descending. Vectors int8 with one global `scale`, `float = int8 × scale`.
- Trace `embed` event: `{ type: 'embed'; cycle; seqLen; dims; source: 'model' | 'asset'; rows?: number[][] }`. `rows` only with `source: 'model'`, values rounded to 3 decimals, one row per token fed that cycle. `preview` is removed; legacy events carrying it must still validate.
- Sim mode never fetches the asset before or during a run; it emits `source: 'asset'` with no rows.
- The card shows 8 neighbours; the similarity matrix is capped at the last 24 tokens.
- UI copy (verbatim): callout titles "learned, not designed", "no position here", "tied with Logits"; captions "Exact rows from the running model." and "Real SmolLM2 embedding rows, reduced to 64 dimensions offline; similarities are approximate."; error line "Vocabulary geometry couldn't be loaded".
- Unit tests never touch the network: `src/test/setup.ts` stubs `fetch` to reject; tests that need fetch stub it themselves.
- Type-check with `npx tsc --noEmit -p tsconfig.app.json` (IDE diagnostics are unreliable), lint with `npm run lint`, test with `npm test`. Python tests: `cd tools/export && uv run pytest`.
- The export tool is never run in CI; heavy commands are operator-run (Task 4).
- Every commit message ends with these two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL`.

## File structure

Python (`tools/export/src/tsumugi_export/`):
- `onnx_config.py` — modify: add `hidden_states.{0..N}` outputs before `attentions.*`.
- `postprocess.py` — create: `expose_inputs_embeds(model)` keeps `hidden_states.0` as `inputs_embeds`, drops the rest.
- `export.py` — modify: `output_hidden_states=True`, run the post-process.
- `stock.py` — create: `stock_embeddings()`, `stock_token_texts(vocab)`.
- `geometry.py` — create: `build_geometry(E, texts, k, pca_dims)`, `write_geometry(dir, built)`, CLI `run`.
- `validate.py` — modify: `inputs-embeds-parity` and `geometry` checks; artifacts keyed by model-dir-relative path.
- `publish.py` — modify: artifact path resolution, model card text.
- `__main__.py`, `README.md`, `pyproject.toml` — modify.

TypeScript (`src/`):
- `trace/types.ts`, `trace/validate.ts`, `test/fixtures.ts` — modify.
- `engine/simulated/SimulatedEngine.ts`, `engine/transformers/worker.ts` — modify; `engine/transformers/embedRows.ts` — create.
- `engine/tokenizer.ts` — modify: `GEOMETRY_MODEL_IDS`, `GEOMETRY_BASE_URL`.
- `geometry/asset.ts`, `geometry/math.ts`, `geometry/useGeometry.ts` — create; `test/geometryFixture.ts` — create.
- `viz/selectors.ts` — modify: `embeddingRows`, export `thousands`.
- `viz/details/EmbeddingsDetail.tsx` — rewrite (container); `viz/EmbeddingLookup.tsx`, `viz/EmbeddingGeometry.tsx` — create.
- `viz/DetailPanel.tsx`, `viz/PipelineBand.tsx`, `index.css` — modify.
- `scripts/capture-screenshots.mjs`, `README.md`, `docs/BACKLOG.md` — modify.

---

### Task 1: `inputs_embeds` graph output (ONNX config + post-processing)

**Files:**
- Modify: `tools/export/src/tsumugi_export/onnx_config.py`
- Create: `tools/export/src/tsumugi_export/postprocess.py`
- Modify: `tools/export/src/tsumugi_export/export.py`
- Test: `tools/export/tests/test_onnx_config.py`, `tools/export/tests/test_postprocess.py`

**Interfaces:**
- Produces: exported graphs (`onnx/model.onnx`, and the quantized variant derived from it) with an extra output `inputs_embeds` `[batch, sequence, 576]`; `expose_inputs_embeds(model: onnx.ModelProto) -> onnx.ModelProto`; `expose_inputs_embeds_file(path: Path) -> None`.

Background: `torch.onnx.export` assigns the config's output names positionally to the model's flattened forward outputs, whose order for `CausalLMOutputWithPast` is `logits, past_key_values, hidden_states, attentions`. With `output_hidden_states=True` transformers returns `num_hidden_layers + 1` hidden states, index 0 being the embedding lookup (the input to layer 0). So the config declares `hidden_states.0..N` between `present.*` and `attentions.*`, and a post-processing pass keeps only index 0 under the name `inputs_embeds`. If the installed transformers ever returned a different count, the positional names would shift and the `attn-wellformed` and `inputs-embeds-parity` checks in `validate` would fail loudly — that is the gate.

- [ ] **Step 1: Write the failing config test**

Append to `tools/export/tests/test_onnx_config.py`:

```python
def test_hidden_states_declared_between_cache_and_attentions():
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=True)
    keys = list(cfg.outputs)
    hidden = [k for k in keys if k.startswith("hidden_states.")]
    assert hidden == [f"hidden_states.{i}" for i in range(31)]   # 30 layers + 1
    assert cfg.outputs["hidden_states.0"] == {0: "batch_size", 1: "sequence_length"}
    # positional contract: every hidden_states.* precedes every attentions.*
    assert keys.index("hidden_states.30") < keys.index("attentions.0")
    assert keys.index("present.0.key") < keys.index("hidden_states.0")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd tools/export && uv run pytest tests/test_onnx_config.py -q`
Expected: 1 failed (`hidden == []`).

- [ ] **Step 3: Update the config**

Replace the `outputs` property in `onnx_config.py`:

```python
class AttnLlamaOnnxConfig(LlamaOnnxConfig):
    @property
    def outputs(self):
        outs = dict(super().outputs)
        n = self._config.num_hidden_layers
        # torch.onnx.export assigns these names POSITIONALLY to the flattened
        # forward outputs (CausalLMOutputWithPast order: logits, past_key_values,
        # hidden_states, attentions). transformers returns n+1 hidden states with
        # index 0 = the embedding lookup; postprocess.expose_inputs_embeds keeps
        # that one as `inputs_embeds` and drops the rest.
        for i in range(n + 1):
            outs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}
        for i in range(n):
            outs[f"attentions.{i}"] = {0: "batch_size", 2: "query_length", 3: "kv_length"}
        return outs
```

Update the module docstring's first line to: `"""Custom ONNX export config: stock Llama-family causal-LM outputs plus per-layer attention weights (attentions.{i}) and the hidden-state chain (hidden_states.{i}, reduced to inputs_embeds after export)."""`

- [ ] **Step 4: Run the config tests**

Run: `cd tools/export && uv run pytest tests/test_onnx_config.py -q`
Expected: all pass.

- [ ] **Step 5: Write the failing post-process test**

Create `tools/export/tests/test_postprocess.py`:

```python
import numpy as np
import onnx
from onnx import TensorProto, helper

from tsumugi_export.postprocess import expose_inputs_embeds


def tiny_graph() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, "seq", 4])
    nodes = [
        helper.make_node("Identity", ["input_ids"], ["hidden_states.0"]),
        helper.make_node("Neg", ["hidden_states.0"], ["hidden_states.1"]),
        helper.make_node("Identity", ["hidden_states.1"], ["logits"]),
        helper.make_node("Abs", ["input_ids"], ["attentions.0"]),
    ]
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, "seq", 4])
            for n in ["logits", "hidden_states.0", "hidden_states.1", "attentions.0"]]
    graph = helper.make_graph(nodes, "tiny", [x], outs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_keeps_embeds_drops_other_hidden_states():
    model = expose_inputs_embeds(tiny_graph())
    onnx.checker.check_model(model)
    assert [o.name for o in model.graph.output] == ["logits", "attentions.0", "inputs_embeds"]


def test_inputs_embeds_equals_hidden_states_0():
    import onnxruntime as ort
    model = expose_inputs_embeds(tiny_graph())
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": x})))
    np.testing.assert_array_equal(out["inputs_embeds"], x)
    np.testing.assert_array_equal(out["logits"], -x)


def test_missing_hidden_states_0_raises():
    model = tiny_graph()
    del model.graph.output[1]   # hidden_states.0
    try:
        expose_inputs_embeds(model)
    except ValueError as err:
        assert "hidden_states.0" in str(err)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd tools/export && uv run pytest tests/test_postprocess.py -q`
Expected: ImportError / ModuleNotFoundError for `tsumugi_export.postprocess`.

- [ ] **Step 7: Implement the post-process**

Create `tools/export/src/tsumugi_export/postprocess.py`:

```python
"""Graph post-processing after the optimum export.

The export declares hidden_states.{0..N} as outputs only because torch.onnx
names forward outputs positionally (see onnx_config.py). Here we keep the one
we want — hidden_states.0, the embedding lookup — under the artifact-contract
name `inputs_embeds`, and drop the other hidden_states.* outputs. The tensors
themselves stay in the graph (the layers consume them); only the output list
changes, so this costs nothing at inference time."""
from pathlib import Path

import onnx
from onnx import helper

EMBEDS_OUTPUT = "inputs_embeds"
SOURCE_OUTPUT = "hidden_states.0"


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


def expose_inputs_embeds_file(path: Path) -> None:
    """Rewrite `path` in place. onnx.load pulls any external data into the
    proto, so the saved file is self-contained; the 135M fp32 graph (~540 MB)
    is well under protobuf's 2 GB limit. A stale external-data sidecar from
    the optimum export is removed."""
    model = onnx.load(str(path))
    onnx.save(expose_inputs_embeds(model), str(path))
    sidecar = path.with_name(path.name + "_data")
    if sidecar.exists():
        sidecar.unlink()
```

- [ ] **Step 8: Run the post-process tests**

Run: `cd tools/export && uv run pytest tests/test_postprocess.py -q`
Expected: 3 passed.

- [ ] **Step 9: Wire it into `export.py`**

In `export.py`, add the import `from tsumugi_export.postprocess import expose_inputs_embeds_file`, change `model_kwargs` to
`model_kwargs={"output_attentions": True, "output_hidden_states": True, "attn_implementation": "eager"},`
and, directly after the `model.onnx` rename block (before the tokenizer-file copy), add:

```python
    expose_inputs_embeds_file(onnx_dir / "model.onnx")
```

Update the module docstring to mention `inputs_embeds`: `"""Export SmolLM2 with attention outputs and the embedding lookup (inputs_embeds). Heavy: ..."""`.

- [ ] **Step 10: Full Python test run and commit**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass.

```bash
git add tools/export/src/tsumugi_export/onnx_config.py tools/export/src/tsumugi_export/postprocess.py tools/export/src/tsumugi_export/export.py tools/export/tests/test_onnx_config.py tools/export/tests/test_postprocess.py
git commit -F- <<'MSG'
feat(export): expose the embedding lookup as an inputs_embeds graph output

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 2: `geometry` command — build the vocabulary-geometry asset

**Files:**
- Create: `tools/export/src/tsumugi_export/stock.py`
- Create: `tools/export/src/tsumugi_export/geometry.py`
- Modify: `tools/export/src/tsumugi_export/__main__.py`
- Modify: `tools/export/pyproject.toml` (add `"safetensors>=0.4"` to `dependencies`, then `uv sync`)
- Test: `tools/export/tests/test_geometry.py`

**Interfaces:**
- Produces: `build_geometry(E: np.ndarray, texts: list[str], k: int = 12, pca_dims: int = 64, block: int = 2048) -> dict` with keys `manifest` (dict), `neighbors` (bytes), `vectors` (bytes), `tokens` (bytes, UTF-8 JSON); `write_geometry(out_dir: Path, built: dict) -> None`; `stock_embeddings() -> np.ndarray` (fp32 `[49152, 576]`); `stock_token_texts(vocab: int) -> list[str]`. Task 3's validate check reads the same files back.

- [ ] **Step 1: Write the failing test**

Create `tools/export/tests/test_geometry.py`:

```python
import json

import numpy as np

from tsumugi_export.geometry import build_geometry, write_geometry


def synthetic(vocab=50, dims=8, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(vocab, dims)).astype(np.float32)
    E[1] = E[0] * 2.0            # token 1 is an exact direction-twin of token 0
    E[2] = -E[0]                 # token 2 is the antipode of token 0
    return E, [f"t{i}" for i in range(vocab)]


def decode(built):
    m = built["manifest"]
    v, k = m["vocabSize"], m["k"]
    raw = built["neighbors"]
    ids = np.frombuffer(raw[: v * k * 2], dtype="<u2").reshape(v, k)
    sims = np.frombuffer(raw[v * k * 2:], dtype=np.uint8).reshape(v, k)
    q = np.frombuffer(built["vectors"], dtype=np.int8).reshape(v, m["pcaDims"])
    return m, ids, sims, q


def test_neighbors_exact_self_excluded_and_sorted():
    E, texts = synthetic()
    m, ids, sims, _ = decode(build_geometry(E, texts, k=5, pca_dims=4, block=16))
    assert ids.shape == (50, 5)
    for t in range(50):
        assert t not in ids[t]
        assert list(sims[t]) == sorted(sims[t], reverse=True)
    assert ids[0][0] == 1 and sims[0][0] == 255      # cos = 1 → 255
    assert 2 not in ids[0]                            # antipode never a neighbour (cos = -1)


def test_sims_match_exact_cosine_at_uint8_resolution():
    E, texts = synthetic()
    _, ids, sims, _ = decode(build_geometry(E, texts, k=5, pca_dims=4, block=16))
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    for t in (0, 7, 49):
        for j, s in zip(ids[t], sims[t]):
            assert abs(np.rint(max(U[t] @ U[j], 0) * 255) - s) <= 1


def test_vectors_round_trip_within_scale():
    E, texts = synthetic()
    m, _, _, q = decode(build_geometry(E, texts, k=5, pca_dims=4, block=16))
    assert q.shape == (50, 4)
    C = E - E.mean(axis=0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    P = C @ vt[:4].T
    # PCA axes are sign-ambiguous; compare magnitudes column by column
    assert np.allclose(np.abs(q * m["scale"]), np.abs(P), atol=m["scale"])
    assert 0 < m["explainedVariance"] <= 1


def test_manifest_and_files(tmp_path):
    E, texts = synthetic()
    built = build_geometry(E, texts, k=5, pca_dims=4, block=16)
    write_geometry(tmp_path, built)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["vocabSize"] == 50 and m["dims"] == 8 and m["k"] == 5 and m["pcaDims"] == 4
    for name, size in m["files"].items():
        assert (tmp_path / name).stat().st_size == size
    assert json.loads((tmp_path / "tokens.json").read_text()) == texts
    assert len(m["sourceSha256"]) == 64


def test_texts_length_must_match_vocab():
    E, texts = synthetic()
    try:
        build_geometry(E, texts[:-1], k=5, pca_dims=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd tools/export && uv run pytest tests/test_geometry.py -q`
Expected: ModuleNotFoundError for `tsumugi_export.geometry`.

- [ ] **Step 3: Implement `stock.py`**

```python
"""Stock SmolLM2 weights and tokenizer, for the geometry asset and validation."""
import numpy as np

from tsumugi_export import STOCK_MODEL_ID


def stock_embeddings() -> np.ndarray:
    """The input embedding matrix, fp32 [vocab, dims]. Read straight from the
    safetensors file (via torch: the checkpoint is bf16, which numpy lacks)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    path = hf_hub_download(STOCK_MODEL_ID, "model.safetensors")
    return load_file(path)["model.embed_tokens.weight"].float().numpy()


def stock_token_texts(vocab: int) -> list[str]:
    """decode([id]) for every id — the same convention the app uses for TokenInfo.text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STOCK_MODEL_ID)
    return [tok.decode([i]) for i in range(vocab)]
```

- [ ] **Step 4: Implement `geometry.py`**

```python
"""Vocabulary-geometry asset for the Embeddings card (M4 spec, Component 1):
exact top-k cosine neighbours over the full embedding space, a PCA-reduced
int8 copy of every row, the decoded text of every id, and a manifest."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tsumugi_export import STOCK_MODEL_ID

K = 12
PCA_DIMS = 64
FILES = ("neighbors.bin", "vectors.bin", "tokens.json")


def build_geometry(E: np.ndarray, texts: list[str], k: int = K, pca_dims: int = PCA_DIMS,
                   block: int = 2048) -> dict:
    vocab, dims = E.shape
    if len(texts) != vocab:
        raise ValueError(f"{len(texts)} texts for a vocabulary of {vocab}")
    if vocab > 65535:
        raise ValueError("neighbour ids are uint16")
    E = np.ascontiguousarray(E, dtype=np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    U = E / np.maximum(norms, 1e-12)

    ids = np.empty((vocab, k), dtype="<u2")
    sims = np.empty((vocab, k), dtype=np.uint8)
    for start in range(0, vocab, block):          # 49152² cosines at once is ~10 GB; block it
        S = U[start:start + block] @ U.T
        rows = np.arange(S.shape[0])
        S[rows, start + rows] = -np.inf           # a token is not its own neighbour
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-top_s, axis=1, kind="stable")
        ids[start:start + block] = np.take_along_axis(top, order, axis=1)
        sims[start:start + block] = np.rint(np.clip(np.take_along_axis(top_s, order, axis=1), 0, 1) * 255)

    mean = E.mean(axis=0)
    C = E - mean
    _, s, vt = np.linalg.svd(C, full_matrices=False)
    P = C @ vt[:pca_dims].T
    explained = float((s[:pca_dims] ** 2).sum() / (s ** 2).sum())
    scale = float(np.abs(P).max() / 127) or 1.0
    q = np.clip(np.rint(P / scale), -127, 127).astype(np.int8)

    neighbors = ids.tobytes() + sims.tobytes()
    vectors = q.tobytes()
    tokens = json.dumps(texts, ensure_ascii=False).encode("utf-8")
    manifest = {
        "modelId": STOCK_MODEL_ID,
        "vocabSize": int(vocab), "dims": int(dims), "k": int(k), "pcaDims": int(pca_dims),
        "scale": scale, "explainedVariance": explained,
        "sourceSha256": hashlib.sha256(E.tobytes()).hexdigest(),
        "files": {"neighbors.bin": len(neighbors), "vectors.bin": len(vectors), "tokens.json": len(tokens)},
    }
    return {"manifest": manifest, "neighbors": neighbors, "vectors": vectors, "tokens": tokens}


def write_geometry(out_dir: Path, built: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(built["manifest"], indent=2))
    (out_dir / "neighbors.bin").write_bytes(built["neighbors"])
    (out_dir / "vectors.bin").write_bytes(built["vectors"])
    (out_dir / "tokens.json").write_bytes(built["tokens"])


def run(args) -> int:
    from tsumugi_export.stock import stock_embeddings, stock_token_texts
    E = stock_embeddings()
    texts = stock_token_texts(E.shape[0])
    out = Path(args.model_dir) / "geometry"
    write_geometry(out, build_geometry(E, texts))
    m = json.loads((out / "manifest.json").read_text())
    print(f"wrote geometry → {out} (PCA-{m['pcaDims']} explains {m['explainedVariance']:.1%} of variance)")
    return 0
```

- [ ] **Step 5: Register the command and the dependency**

In `__main__.py`, after the `quantize` parser:

```python
    p_geo = sub.add_parser("geometry", help="build the vocabulary-geometry asset (geometry/)")
    p_geo.add_argument("--model-dir", default="out/model")
```

In `pyproject.toml` add `"safetensors>=0.4",` to `dependencies`, then run `cd tools/export && uv sync`.

- [ ] **Step 6: Run the tests**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tools/export/src/tsumugi_export/stock.py tools/export/src/tsumugi_export/geometry.py tools/export/src/tsumugi_export/__main__.py tools/export/pyproject.toml tools/export/uv.lock tools/export/tests/test_geometry.py
git commit -F- <<'MSG'
feat(export): geometry command builds the vocabulary-geometry asset

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 3: validation checks, artifact hashes, publish and tool docs

**Files:**
- Modify: `tools/export/src/tsumugi_export/validate.py`
- Modify: `tools/export/src/tsumugi_export/publish.py`
- Modify: `tools/export/README.md`
- Test: `tools/export/tests/test_validate_geometry.py`, `tools/export/tests/test_publish_gate.py`

**Interfaces:**
- Consumes: `stock_embeddings`, `stock_token_texts` (Task 2); the `inputs_embeds` output (Task 1).
- Produces: `validation-report.json` whose `artifacts` keys are model-dir-relative paths (`onnx/model.onnx`, `geometry/neighbors.bin`, …); checks `<variant>:inputs-embeds-parity` and `geometry`; `publish.artifact_path(model_dir, name) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tools/export/tests/test_validate_geometry.py`:

```python
import json

import numpy as np

from tsumugi_export.geometry import build_geometry, write_geometry
from tsumugi_export.validate import check_geometry_files


def make(tmp_path, vocab=40, dims=6):
    rng = np.random.default_rng(1)
    E = rng.normal(size=(vocab, dims)).astype(np.float32)
    texts = [f"t{i}" for i in range(vocab)]
    write_geometry(tmp_path / "geometry", build_geometry(E, texts, k=4, pca_dims=3, block=8))
    return E, texts


def test_geometry_check_passes_on_fresh_asset(tmp_path):
    E, texts = make(tmp_path)
    res = check_geometry_files(tmp_path, E, texts, sample=10)
    assert res["passed"], res["detail"]


def test_geometry_check_fails_on_corrupt_neighbors(tmp_path):
    E, texts = make(tmp_path)
    p = tmp_path / "geometry" / "neighbors.bin"
    raw = bytearray(p.read_bytes())
    raw[0] ^= 0xFF
    p.write_bytes(bytes(raw))
    assert not check_geometry_files(tmp_path, E, texts, sample=40)["passed"]


def test_geometry_check_fails_on_wrong_text(tmp_path):
    E, texts = make(tmp_path)
    p = tmp_path / "geometry" / "tokens.json"
    bad = json.loads(p.read_text())
    bad[3] = "wrong"
    p.write_text(json.dumps(bad))
    assert not check_geometry_files(tmp_path, E, texts, sample=40)["passed"]


def test_geometry_check_fails_when_missing(tmp_path):
    res = check_geometry_files(tmp_path, np.zeros((2, 2), np.float32), ["a", "b"])
    assert not res["passed"] and "geometry" in res["detail"]
```

Append to `tools/export/tests/test_publish_gate.py`:

```python
from pathlib import Path

from tsumugi_export.publish import artifact_path


def test_artifact_paths_are_model_dir_relative():
    assert artifact_path(Path("out/m"), "onnx/model.onnx") == Path("out/m/onnx/model.onnx")
    assert artifact_path(Path("out/m"), "geometry/neighbors.bin") == Path("out/m/geometry/neighbors.bin")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd tools/export && uv run pytest tests/test_validate_geometry.py tests/test_publish_gate.py -q`
Expected: ImportError on `check_geometry_files` and `artifact_path`.

- [ ] **Step 3: Add the checks to `validate.py`**

Add after `_check_cache_integrity`:

```python
def _check_inputs_embeds_parity(sess, model_dir: Path, E: np.ndarray, tol: float) -> dict:
    """inputs_embeds must be exactly the embedding-table rows of the fed ids
    (within quantization noise for the int8 variant)."""
    out_names = [o.name for o in sess.get_outputs()]
    if "inputs_embeds" not in out_names:
        return {"passed": False, "detail": "graph has no inputs_embeds output"}
    ids = _tokenize(model_dir, PROMPTS[0])
    outputs = dict(zip(out_names, sess.run(None, _feeds(sess, ids))))
    got = outputs["inputs_embeds"]
    expected = E[ids[0]][None, :, :]
    if got.shape != expected.shape:
        return {"passed": False, "detail": f"inputs_embeds shape {got.shape}, expected {expected.shape}"}
    diff = float(np.max(np.abs(got - expected)))
    return {"passed": diff <= tol, "detail": f"max|Δembed|={diff:.2e} (atol {tol})"}


def check_geometry_files(model_dir: Path, E: np.ndarray, texts: list[str], sample: int = 32) -> dict:
    """Spot-check geometry/ against the stock matrix: file sizes match the
    manifest, every listed neighbour of a sampled token is within uint8
    resolution of the exact k-th best cosine, the top similarity round-trips,
    and tokens.json matches the tokenizer."""
    gdir = model_dir / "geometry"
    if not (gdir / "manifest.json").exists():
        return {"passed": False, "detail": "no geometry/ — run geometry first"}
    m = json.loads((gdir / "manifest.json").read_text())
    for name, size in m["files"].items():
        actual = (gdir / name).stat().st_size if (gdir / name).exists() else -1
        if actual != size:
            return {"passed": False, "detail": f"{name}: {actual} bytes, manifest says {size}"}
    vocab, k = m["vocabSize"], m["k"]
    if vocab != E.shape[0] or len(texts) != vocab:
        return {"passed": False, "detail": f"vocab {vocab} vs stock {E.shape[0]} / {len(texts)} texts"}
    raw = (gdir / "neighbors.bin").read_bytes()
    ids = np.frombuffer(raw[: vocab * k * 2], dtype="<u2").reshape(vocab, k)
    sims = np.frombuffer(raw[vocab * k * 2:], dtype=np.uint8).reshape(vocab, k)
    tokens = json.loads((gdir / "tokens.json").read_text())
    U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    for t in rng.choice(vocab, size=min(sample, vocab), replace=False):
        if tokens[t] != texts[t]:
            return {"passed": False, "detail": f"tokens.json[{t}]={tokens[t]!r} != tokenizer {texts[t]!r}"}
        s = U[t] @ U.T
        s[t] = -np.inf
        kth = np.sort(s)[-k]
        if any(s[j] < kth - 1 / 255 for j in ids[t]):
            return {"passed": False, "detail": f"token {t}: listed neighbour below the exact k-th best"}
        if abs(np.rint(max(s[ids[t][0]], 0) * 255) - sims[t][0]) > 1:
            return {"passed": False, "detail": f"token {t}: top similarity does not round-trip"}
    return {"passed": True, "detail": f"{min(sample, vocab)} sampled tokens match exact neighbours and texts"}
```

In `run`, load the stock matrix once after `stock = _session(stock_path)`:

```python
    from tsumugi_export.stock import stock_embeddings, stock_token_texts
    E = stock_embeddings()
    texts = stock_token_texts(E.shape[0])
```

Inside the variant loop, after the `cache-integrity` line:

```python
        pe_tol = 1e-3 if variant in ("model.onnx", "model_fp16.onnx") else TOL
        checks[f"{variant}:inputs-embeds-parity"] = _check_inputs_embeds_parity(sess, model_dir, E, pe_tol)
```

After the A≡B block: `checks["geometry"] = check_geometry_files(model_dir, E, texts)`.

Replace the `artifacts` line with model-dir-relative keys covering both folders:

```python
        "artifacts": {
            str(p.relative_to(model_dir)): _sha256(p)
            for p in [*sorted((model_dir / "onnx").glob("*.onnx")),
                      *(sorted((model_dir / "geometry").glob("*")) if (model_dir / "geometry").exists() else [])]
        },
```

Extend the module docstring list with `5. inputs_embeds parity with the stock embedding table` and `6. geometry/ spot-check against exact neighbours and the tokenizer`.

- [ ] **Step 4: Update `publish.py`**

Add after `_sha256`:

```python
def artifact_path(model_dir: Path, name: str) -> Path:
    """validation-report.json keys are model-dir-relative (onnx/…, geometry/…)."""
    return model_dir / name
```

Change the hash loop to `actual = _sha256(artifact_path(model_dir, name))` and the ignore pattern stays `["onnx/model.onnx"]` (the geometry folder uploads with the rest). Extend `MODEL_CARD` after the attentions sentence:

```
Also exposes the embedding lookup as `inputs_embeds` (`[batch, seq, 576]`),
and ships `geometry/` — exact top-12 cosine neighbours, a PCA-64 int8 copy of
the embedding table, decoded token texts and a manifest — for the Embeddings
card in Tsumugi.
```

- [ ] **Step 5: Run the tests**

Run: `cd tools/export && uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Document the tool changes**

In `tools/export/README.md`: first paragraph gains "and the embedding lookup (`inputs_embeds`), plus a vocabulary-geometry asset"; add to Commands after `quantize`:

`- `geometry` — build `geometry/` (exact top-12 neighbours, PCA-64 int8 vectors, token texts, manifest) from the stock embedding table (`--model-dir`).`

In the protocol, insert a step between Quantize and Validate: `**Build the geometry asset:** `uv run python -m tsumugi_export geometry --model-dir out/model``, and add to the validate step's expectations: "`*:inputs-embeds-parity` and `geometry` must PASS". Add a subsection:

```
## Geometry asset layout

| File | Content |
|---|---|
| `manifest.json` | modelId, vocabSize, dims, k, pcaDims, scale, explainedVariance, sourceSha256, files |
| `neighbors.bin` | ids block (`vocabSize × k` little-endian uint16) then sims block (`vocabSize × k` uint8, `round(max(cos,0)×255)`), per token sorted by similarity descending, self excluded |
| `vectors.bin` | `vocabSize × pcaDims` int8, `float = int8 × scale`, PCA on the mean-centred table |
| `tokens.json` | JSON array of decoded token strings, index = id |
```

- [ ] **Step 7: Commit**

```bash
git add tools/export/src/tsumugi_export/validate.py tools/export/src/tsumugi_export/publish.py tools/export/README.md tools/export/tests/test_validate_geometry.py tools/export/tests/test_publish_gate.py
git commit -F- <<'MSG'
feat(export): validate inputs_embeds parity and the geometry asset; publish both

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 4: Operator step — re-export, validate, publish (human-run)

**Files:** none in the repo. This task is executed by the human operator; the executor stops here and hands over the commands, then continues with Task 5 while the operator runs them (Tasks 5–13 do not depend on the Hub state; only the final in-browser check does).

- [ ] **Step 1: Run the pipeline**

```bash
cd tools/export && uv sync
uv run python -m tsumugi_export export --out out/model
uv run python -m tsumugi_export export --out out/model-nocache --no-cache
uv run python -m tsumugi_export quantize --model-dir out/model
uv run python -m tsumugi_export geometry --model-dir out/model
uv run python -m tsumugi_export validate --model-dir out/model
```

Expected verdict table: every row PASS, including `model.onnx:inputs-embeds-parity`, `model_quantized.onnx:inputs-embeds-parity`, `geometry`, and a non-SKIPPED `a-equiv-b`. `VERDICT: PASS`.

- [ ] **Step 2: Publish**

```bash
uv run python -m tsumugi_export publish --model-dir out/model
```

Then confirm `https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx/resolve/main/geometry/manifest.json` returns the manifest and `.../geometry/neighbors.bin` is ~1.8 MB.

---

### Task 5: Trace shape — `embed` gains `source`/`rows`, loses `preview`

**Files:**
- Modify: `src/trace/types.ts:32`
- Modify: `src/trace/validate.ts:33`
- Modify: `src/test/fixtures.ts`
- Modify: `src/engine/simulated/SimulatedEngine.ts:47-49`
- Modify: `src/engine/transformers/worker.ts:146-148` (minimal; Task 6 does the real rows)
- Modify: `src/viz/details/EmbeddingsDetail.tsx` (minimal compile fix; Task 10 rebuilds it)
- Test: `src/trace/validate.test.ts`, `src/engine/simulated/SimulatedEngine.test.ts`

**Interfaces:**
- Produces: `export type EmbedSource = 'model' | 'asset'`; embed event `{ type: 'embed'; cycle: number; seqLen: number; dims: number; source: EmbedSource; rows?: number[][] }`; fixture option `embedRows?: boolean` on `FixtureTraceOpts` (when true, embed events are `source: 'model'` with deterministic 576-dim rows from `fixtureEmbedding(id)`); exported `fixtureEmbedding(id: number): number[]`.

- [ ] **Step 1: Write the failing validator tests**

Append to `src/trace/validate.test.ts`:

```ts
test('embed rows must be dims-long finite numbers', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'embed')
  if (e?.type === 'embed') { e.source = 'model'; e.rows = [[0.1, 0.2]] }   // dims is 576
  expect(validateTrace(t).some((v) => v.includes('embed'))).toBe(true)
})

test('model-source embed without rows is flagged', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'embed')
  if (e?.type === 'embed') e.source = 'model'
  expect(validateTrace(t).some((v) => v.includes('embed'))).toBe(true)
})

test('model-source embed with well-formed rows is valid', () => {
  expect(validateTrace(buildFixtureTrace({ embedRows: true }))).toEqual([])
})

test('legacy embed events carrying preview and no source still validate', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((x) => x.type === 'embed')
  t[i] = { type: 'embed', cycle: 0, seqLen: 2, dims: 576, preview: [[0.1, 0.2]] } as unknown as TraceEvent
  expect(validateTrace(t)).toEqual([])
})
```

Add `buildFixtureTrace` to the fixtures import and `import type { TraceEvent } from './types'`.

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/trace/validate.test.ts`
Expected: type errors / failures on `source`, `rows`, `embedRows`.

- [ ] **Step 3: Change the type**

In `src/trace/types.ts` replace the embed line with:

```ts
  | { type: 'embed'; cycle: number; seqLen: number; dims: number; source: EmbedSource; rows?: number[][] }
```

and add above `TraceEvent`:

```ts
// Where a run's embedding vectors come from: 'model' = exact rows emitted into
// the trace (real mode with the inputs_embeds export); 'asset' = the view looks
// vectors up by id in the Hub geometry asset (sim mode, or an old cached model).
export type EmbedSource = 'model' | 'asset'
```

- [ ] **Step 4: Validate rows**

In `src/trace/validate.ts` replace the `if (e.type === 'embed' && phase === 'embed') { ... }` line with:

```ts
    if (e.type === 'embed' && phase === 'embed') {
      if (e.rows) {
        e.rows.forEach((row, i) => {
          if (!Array.isArray(row) || row.length !== e.dims || row.some((v) => typeof v !== 'number' || !Number.isFinite(v)))
            errs.push(`embed cycle ${e.cycle} row ${i} is not ${e.dims} finite numbers`)
        })
      }
      if (e.source === 'model' && !e.rows) errs.push(`embed cycle ${e.cycle} claims model source without rows`)
      phase = 'layer'; layerIdx = 0; continue
    }
```

(Legacy events have `source === undefined`, which is neither checked branch, so they pass.)

- [ ] **Step 5: Update fixtures**

In `src/test/fixtures.ts` add `embedRows?: boolean` to `FixtureTraceOpts` and, above `buildFixtureTrace`:

```ts
// Deterministic 576-dim "embedding" for a token id, for model-source fixtures.
export function fixtureEmbedding(id: number): number[] {
  return Array.from({ length: 576 }, (_, d) => Math.round(Math.sin(id * 0.37 + d * 0.11) * 1000) / 1000)
}
```

Replace the embed push inside the cycle loop with:

```ts
    // the rows fed this cycle: the whole prompt at cycle 0, then the token chosen last cycle
    const fed = c === 0 ? promptTokens.map((t) => t.id) : [chosenFor(c - 1).id]
    events.push(opts.embedRows
      ? { type: 'embed', cycle: c, seqLen: promptTokens.length + c, dims: 576, source: 'model', rows: fed.map(fixtureEmbedding) }
      : { type: 'embed', cycle: c, seqLen: promptTokens.length + c, dims: 576, source: 'asset' })
```

- [ ] **Step 6: Engines and the card, minimally**

`src/engine/simulated/SimulatedEngine.ts`: delete the `const preview = …` statement and emit
`emit({ type: 'embed', cycle, seqLen: seq.length, dims: this.dims, source: 'asset' })`.

`src/engine/transformers/worker.ts`: replace the two-line schematic emit (and its comment) with
`emit({ type: 'embed', cycle, seqLen: allIds.length, dims, source: 'asset' })` — Task 6 replaces this again.

`src/viz/details/EmbeddingsDetail.tsx` becomes, for now:

```tsx
import type { TraceEvent } from '../../trace/types'

export function EmbeddingsDetail({ event }: { event: Extract<TraceEvent, { type: 'embed' }> }) {
  return (
    <div data-testid="detail-embeddings" className="detail">
      <h3>Embeddings</h3>
      <p>Each token becomes a vector of {event.dims} numbers.</p>
    </div>
  )
}
```

Replace the `embed preview is capped…` test in `SimulatedEngine.test.ts` with:

```ts
test('embed events are asset-sourced and carry no vectors (sim stays instant)', async () => {
  const events = await collect('one two three four five six')
  const embeds = events.filter((e) => e.type === 'embed')
  expect(embeds.length).toBeGreaterThan(0)
  for (const e of embeds) {
    if (e.type !== 'embed') continue
    expect(e.source).toBe('asset')
    expect(e.rows).toBeUndefined()
    expect(e.dims).toBe(576)
  }
})
```

- [ ] **Step 7: Verify**

Run: `npx tsc --noEmit -p tsconfig.app.json && npm run lint && npm test`
Expected: clean, all tests pass (the DetailPanel test `embeddings detail shows dims caption` still finds `576`).

- [ ] **Step 8: Commit**

```bash
git add src/trace/types.ts src/trace/validate.ts src/trace/validate.test.ts src/test/fixtures.ts src/engine/simulated/SimulatedEngine.ts src/engine/simulated/SimulatedEngine.test.ts src/engine/transformers/worker.ts src/viz/details/EmbeddingsDetail.tsx
git commit -F- <<'MSG'
feat(trace): embed events carry a vector source and optional real rows

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 6: Worker emits real embedding rows

**Files:**
- Create: `src/engine/transformers/embedRows.ts`
- Modify: `src/engine/transformers/worker.ts` (the cycle loop)
- Test: `src/engine/transformers/embedRows.test.ts`

**Interfaces:**
- Produces: `extractEmbedRows(tensor: { dims: number[]; data: ArrayLike<number> } | undefined, fed: number, dims: number, decimals?: number): EmbedRowsResult` where `EmbedRowsResult = { status: 'absent' } | { status: 'bad-shape'; dims: number[] } | { status: 'ok'; rows: number[][] }`.

- [ ] **Step 1: Write the failing test**

Create `src/engine/transformers/embedRows.test.ts`:

```ts
import { expect, test } from 'vitest'
import { extractEmbedRows } from './embedRows'

test('absent output → absent', () => {
  expect(extractEmbedRows(undefined, 3, 4)).toEqual({ status: 'absent' })
})

test('wrong shape → bad-shape with the offending dims', () => {
  const t = { dims: [1, 2, 4], data: new Float32Array(8) }
  expect(extractEmbedRows(t, 3, 4)).toEqual({ status: 'bad-shape', dims: [1, 2, 4] })
  expect(extractEmbedRows({ dims: [2, 4], data: new Float32Array(8) }, 2, 4).status).toBe('bad-shape')
})

test('ok → one row per fed token, rounded to 3 decimals', () => {
  const data = new Float32Array([0.12345, -1.00049, 2, 3, 4.4444, 5, 6, 7])
  const r = extractEmbedRows({ dims: [1, 2, 4], data }, 2, 4)
  expect(r).toEqual({ status: 'ok', rows: [[0.123, -1, 2, 3], [4.444, 5, 6, 7]] })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/engine/transformers/embedRows.test.ts`
Expected: cannot resolve `./embedRows`.

- [ ] **Step 3: Implement**

Create `src/engine/transformers/embedRows.ts`:

```ts
// Turns the model's `inputs_embeds` output ([1, fed, dims], the embedding
// lookup for the tokens fed this cycle) into trace rows. Pure, so the shape
// policy is unit-testable outside the worker.
export type EmbedRowsResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; dims: number[] }
  | { status: 'ok'; rows: number[][] }

export function extractEmbedRows(
  tensor: { dims: number[]; data: ArrayLike<number> } | undefined,
  fed: number, dims: number, decimals = 3,
): EmbedRowsResult {
  if (!tensor) return { status: 'absent' }
  const d = tensor.dims
  if (d.length !== 3 || d[0] !== 1 || d[1] !== fed || d[2] !== dims || tensor.data.length !== fed * dims)
    return { status: 'bad-shape', dims: [...d] }
  const f = 10 ** decimals
  const rows: number[][] = []
  for (let r = 0; r < fed; r++) {
    const row = new Array<number>(dims)
    for (let c = 0; c < dims; c++) row[c] = Math.round(tensor.data[r * dims + c] * f) / f
    rows.push(row)
  }
  return { status: 'ok', rows }
}
```

- [ ] **Step 4: Run the test**

Run: `npx vitest run src/engine/transformers/embedRows.test.ts`
Expected: 3 passed.

- [ ] **Step 5: Rewire the worker loop**

In `worker.ts` add `import { extractEmbedRows } from './embedRows'` and, next to `let attnBroken = false`, add `let embedsBroken = false`. Then restructure the top of the cycle loop so the `embed` and `layer` events are emitted **after** the forward call (they were emitted before it, carrying no measured data; trace order is unchanged):

```ts
    for (let cycle = 0; cycle < params.maxNewTokens; cycle++) {
      if (aborted) { endRun('aborted'); break }

      const input_ids = new Tensor('int64', BigInt64Array.from(nextInputIds.map(BigInt)), [1, nextInputIds.length])
      const attention_mask = new Tensor('int64', BigInt64Array.from(allIds.map(() => 1n)), [1, allIds.length])
      const out = await model({ input_ids, attention_mask, past_key_values: pastKeyValues })
      pastKeyValues = updateCache(DynamicCache, out, pastKeyValues)

      // Real embedding rows for the tokens fed this cycle (whole prompt at cycle 0,
      // one token afterwards). Absent output = old cached export → 'asset' quietly;
      // a wrong shape flips embedsBroken for the run — never-fail, like attention.
      let rows: number[][] | undefined
      if (!embedsBroken) {
        const r = extractEmbedRows(out.inputs_embeds, nextInputIds.length, dims)
        if (r.status === 'ok') rows = r.rows
        else if (r.status === 'bad-shape') {
          embedsBroken = true
          console.warn(`inputs_embeds has shape [${r.dims.join(', ')}], expected [1, ${nextInputIds.length}, ${dims}] — using asset vectors`)
        }
      }
      emit(rows
        ? { type: 'embed', cycle, seqLen: allIds.length, dims, source: 'model', rows }
        : { type: 'embed', cycle, seqLen: allIds.length, dims, source: 'asset' })
      for (let l = 0; l < numLayers; l++) emit({ type: 'layer', cycle, index: l, total: numLayers })

      if (acc && !attnBroken) {
        // … unchanged from here on …
```

Delete the old `embed`/`layer` emits and the old `input_ids`/`attention_mask`/`out`/`pastKeyValues` lines that followed them (they now live above).

- [ ] **Step 6: Verify**

Run: `npx tsc --noEmit -p tsconfig.app.json && npm run lint && npm test`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/engine/transformers/embedRows.ts src/engine/transformers/embedRows.test.ts src/engine/transformers/worker.ts
git commit -F- <<'MSG'
feat(worker): emit real embedding rows from the inputs_embeds output

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 7: Geometry asset loader

**Files:**
- Modify: `src/engine/tokenizer.ts` (constants)
- Create: `src/geometry/asset.ts`
- Create: `src/test/geometryFixture.ts`
- Modify: `src/test/setup.ts` (fetch stub)
- Test: `src/geometry/asset.test.ts`

**Interfaces:**
- Produces:
  - `GEOMETRY_MODEL_IDS: readonly string[]`, `GEOMETRY_BASE_URL: string` (in `engine/tokenizer.ts`).
  - `interface GeometryManifest { modelId; vocabSize; dims; k; pcaDims; scale; explainedVariance; sourceSha256; files: Record<string, number> }`
  - `interface GeometryAsset { manifest: GeometryManifest; neighbors(id): Array<{ id: number; sim: number }>; vector(id): Float32Array; text(id): string }`
  - `parseGeometry(manifest, neighbors: ArrayBuffer, vectors: ArrayBuffer, tokens: string[]): GeometryAsset`
  - `coversModel(manifest, modelId): boolean`
  - `loadGeometry(baseUrl?): Promise<GeometryAsset>` (one shared promise; a rejection clears it so the next call re-fetches), `resetGeometryCache()` (tests).
  - Test fixture: `encodeGeometryFixture()` → `{ manifest, neighbors, vectors, tokens }` in the exact binary layout; `makeGeometryAsset(): GeometryAsset`; `stubGeometryFetch(overrides?)` installs a fetch mock serving the fixture; constants `FIXTURE_VOCAB = 256`, `FIXTURE_K = 12`, `FIXTURE_PCA = 4`. Fixture vectors: `θ = 2π·id/256`, `[cos θ, sin θ, cos 2θ, sin 2θ]`, so a token's nearest neighbours are `id ± 1`. Texts `t{id}`, except id 7 → `<0x07>` and id 8 → `''` (unrenderable). Manifest `modelId: 'fixture'` (matches the fixture trace's `run-start.modelId`).

- [ ] **Step 1: Disable the network in unit tests**

`src/test/setup.ts`:

```ts
import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// Unit tests never touch the network: any un-stubbed fetch rejects immediately.
// Tests that need fetch install their own stub with vi.stubGlobal('fetch', …).
vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in unit tests'))))
```

Run `npm test` — everything still passes (nothing in the suite fetches today).

- [ ] **Step 2: Write the fixture**

Create `src/test/geometryFixture.ts`:

```ts
import { vi } from 'vitest'
import { parseGeometry, type GeometryAsset, type GeometryManifest } from '../geometry/asset'

export const FIXTURE_VOCAB = 256
export const FIXTURE_K = 12
export const FIXTURE_PCA = 4
const SCALE = 1 / 127

export function fixtureVector(id: number): number[] {
  const t = (2 * Math.PI * id) / FIXTURE_VOCAB
  return [Math.cos(t), Math.sin(t), Math.cos(2 * t), Math.sin(2 * t)]
}

export function fixtureText(id: number): string {
  if (id === 7) return '<0x07>'
  if (id === 8) return ''
  return `t${id}`
}

const cos = (a: number[], b: number[]) => {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return dot / Math.sqrt(na * nb)
}

// Same layout the Python builder writes: ids block (uint16 LE) then sims block (uint8).
export function encodeGeometryFixture(): {
  manifest: GeometryManifest; neighbors: ArrayBuffer; vectors: ArrayBuffer; tokens: string[]
} {
  const v = FIXTURE_VOCAB, k = FIXTURE_K
  const vecs = Array.from({ length: v }, (_, id) => fixtureVector(id))
  const neighbors = new ArrayBuffer(v * k * 3)
  const view = new DataView(neighbors)
  for (let id = 0; id < v; id++) {
    const ranked = vecs.map((w, j) => ({ j, sim: cos(vecs[id], w) }))
      .filter((c) => c.j !== id)
      .sort((a, b) => b.sim - a.sim || a.j - b.j)
      .slice(0, k)
    ranked.forEach((c, i) => {
      view.setUint16((id * k + i) * 2, c.j, true)
      view.setUint8(v * k * 2 + id * k + i, Math.round(Math.max(c.sim, 0) * 255))
    })
  }
  const vectors = new ArrayBuffer(v * FIXTURE_PCA)
  const q = new Int8Array(vectors)
  vecs.forEach((vec, id) => vec.forEach((x, d) => { q[id * FIXTURE_PCA + d] = Math.round(x / SCALE) }))
  const tokens = Array.from({ length: v }, (_, id) => fixtureText(id))
  const manifest: GeometryManifest = {
    modelId: 'fixture', vocabSize: v, dims: 4, k, pcaDims: FIXTURE_PCA, scale: SCALE,
    explainedVariance: 1, sourceSha256: '0'.repeat(64),
    files: { 'neighbors.bin': neighbors.byteLength, 'vectors.bin': vectors.byteLength, 'tokens.json': 0 },
  }
  return { manifest, neighbors, vectors, tokens }
}

export function makeGeometryAsset(): GeometryAsset {
  const f = encodeGeometryFixture()
  return parseGeometry(f.manifest, f.neighbors, f.vectors, f.tokens)
}

type FakeResponse = { ok: boolean; status: number; json?: () => Promise<unknown>; arrayBuffer?: () => Promise<ArrayBuffer> }

// A fetch stub serving the fixture files by URL suffix; returns the mock so tests can count calls.
// `overrides` maps a file name to a function that returns a response or throws.
export function stubGeometryFetch(overrides: Partial<Record<string, () => FakeResponse>> = {}) {
  const f = encodeGeometryFixture()
  const body = (name: string): FakeResponse => {
    const o = overrides[name]
    if (o) return o()
    if (name === 'manifest.json') return { ok: true, status: 200, json: async () => f.manifest }
    if (name === 'neighbors.bin') return { ok: true, status: 200, arrayBuffer: async () => f.neighbors }
    if (name === 'vectors.bin') return { ok: true, status: 200, arrayBuffer: async () => f.vectors }
    if (name === 'tokens.json') return { ok: true, status: 200, json: async () => f.tokens }
    return { ok: false, status: 404 }
  }
  const mock = vi.fn(async (url: string) => body(url.slice(url.lastIndexOf('/') + 1)))
  vi.stubGlobal('fetch', mock)
  return mock
}
```

- [ ] **Step 3: Write the failing tests**

Create `src/geometry/asset.test.ts`:

```ts
import { afterEach, beforeEach, expect, test } from 'vitest'
import { encodeGeometryFixture, makeGeometryAsset, stubGeometryFetch } from '../test/geometryFixture'
import { coversModel, loadGeometry, parseGeometry, resetGeometryCache } from './asset'

beforeEach(() => resetGeometryCache())
afterEach(() => resetGeometryCache())

test('neighbors of a token are its angular neighbours, sorted by similarity', () => {
  const a = makeGeometryAsset()
  const n = a.neighbors(10)
  expect(n).toHaveLength(12)
  expect(new Set([n[0].id, n[1].id])).toEqual(new Set([9, 11]))
  expect(n[0].sim).toBeGreaterThan(n[2].sim)
  expect(n.every((x) => x.id !== 10)).toBe(true)
})

test('vectors dequantise to the encoded values', () => {
  const a = makeGeometryAsset()
  const v = a.vector(0)
  expect(v).toHaveLength(4)
  expect(v[0]).toBeCloseTo(1, 2)
  expect(v[1]).toBeCloseTo(0, 2)
})

test('text() returns the decoded token', () => {
  const a = makeGeometryAsset()
  expect(a.text(3)).toBe('t3')
  expect(a.text(7)).toBe('<0x07>')
})

test('ids outside the vocabulary throw', () => {
  const a = makeGeometryAsset()
  expect(() => a.neighbors(256)).toThrow(RangeError)
  expect(() => a.vector(-1)).toThrow(RangeError)
})

test('byte-length mismatches are rejected', () => {
  const f = encodeGeometryFixture()
  expect(() => parseGeometry(f.manifest, f.neighbors.slice(0, 10), f.vectors, f.tokens)).toThrow(/neighbors\.bin/)
  expect(() => parseGeometry(f.manifest, f.neighbors, f.vectors.slice(0, 10), f.tokens)).toThrow(/vectors\.bin/)
  expect(() => parseGeometry(f.manifest, f.neighbors, f.vectors, f.tokens.slice(1))).toThrow(/tokens\.json/)
})

test('coversModel accepts the stock id and the attention re-export interchangeably', () => {
  const m = { ...encodeGeometryFixture().manifest, modelId: 'HuggingFaceTB/SmolLM2-135M-Instruct' }
  expect(coversModel(m, 'HuggingFaceTB/SmolLM2-135M-Instruct')).toBe(true)
  expect(coversModel(m, 'saigyo-hoshi/smollm2-135m-attn-onnx')).toBe(true)
  expect(coversModel(m, 'someone/other-model')).toBe(false)
  expect(coversModel({ ...m, modelId: 'fixture' }, 'fixture')).toBe(true)
})

test('loadGeometry fetches the four files once and shares the promise', async () => {
  const fetchMock = stubGeometryFetch()
  const [a, b] = await Promise.all([loadGeometry('http://x/geo'), loadGeometry('http://x/geo')])
  expect(a).toBe(b)
  expect(fetchMock).toHaveBeenCalledTimes(4)
  expect(a.text(3)).toBe('t3')
})

test('a failed load is not cached: the next call re-fetches', async () => {
  stubGeometryFetch({ 'manifest.json': () => { throw new Error('offline') } })
  await expect(loadGeometry('http://x/geo')).rejects.toThrow('offline')
  const ok = stubGeometryFetch()
  await expect(loadGeometry('http://x/geo')).resolves.toBeTruthy()
  expect(ok).toHaveBeenCalledTimes(4)
})

test('a non-OK response rejects with the file name and status', async () => {
  stubGeometryFetch({ 'vectors.bin': () => ({ ok: false, status: 500 }) })
  await expect(loadGeometry('http://x/geo')).rejects.toThrow('vectors.bin: HTTP 500')
})
```

- [ ] **Step 4: Run them to verify they fail**

Run: `npx vitest run src/geometry/asset.test.ts`
Expected: cannot resolve `./asset`.

- [ ] **Step 5: Constants**

Append to `src/engine/tokenizer.ts` after `ATTN_MODEL_ID`:

```ts
// Both ids share the same weights, so one geometry asset covers both.
export const GEOMETRY_MODEL_IDS: readonly string[] = [MODEL_ID, ATTN_MODEL_ID]
export const GEOMETRY_BASE_URL: string =
  import.meta.env.VITE_GEOMETRY_BASE_URL ?? `https://huggingface.co/${ATTN_MODEL_ID}/resolve/main/geometry`
```

- [ ] **Step 6: Implement `src/geometry/asset.ts`**

```ts
// The Hub geometry asset (spec Component 1): exact top-k neighbours, PCA-reduced
// int8 vectors and decoded texts for every vocabulary id. Static, read-only,
// deterministic per model — fetched once per page load, so the UI stays a pure
// function of (trace, cursor, asset).
import { GEOMETRY_BASE_URL, GEOMETRY_MODEL_IDS } from '../engine/tokenizer'

export interface GeometryManifest {
  modelId: string
  vocabSize: number
  dims: number
  k: number
  pcaDims: number
  scale: number
  explainedVariance: number
  sourceSha256: string
  files: Record<string, number>
}

export interface GeometryAsset {
  manifest: GeometryManifest
  neighbors(id: number): Array<{ id: number; sim: number }>
  vector(id: number): Float32Array
  text(id: number): string
}

export function parseGeometry(
  manifest: GeometryManifest, neighbors: ArrayBuffer, vectors: ArrayBuffer, tokens: string[],
): GeometryAsset {
  const { vocabSize: v, k, pcaDims: p, scale } = manifest
  if (neighbors.byteLength !== v * k * 3)
    throw new Error(`neighbors.bin: expected ${v * k * 3} bytes, got ${neighbors.byteLength}`)
  if (vectors.byteLength !== v * p)
    throw new Error(`vectors.bin: expected ${v * p} bytes, got ${vectors.byteLength}`)
  if (tokens.length !== v)
    throw new Error(`tokens.json: expected ${v} entries, got ${tokens.length}`)
  // ids are little-endian on disk; every JS engine we run on is little-endian too
  const ids = new Uint16Array(neighbors, 0, v * k)
  const sims = new Uint8Array(neighbors, v * k * 2, v * k)
  const q = new Int8Array(vectors)
  const check = (id: number) => {
    if (!Number.isInteger(id) || id < 0 || id >= v) throw new RangeError(`token id ${id} outside vocabulary of ${v}`)
  }
  return {
    manifest,
    neighbors(id) {
      check(id)
      const out: Array<{ id: number; sim: number }> = []
      for (let i = 0; i < k; i++) out.push({ id: ids[id * k + i], sim: sims[id * k + i] / 255 })
      return out
    },
    vector(id) {
      check(id)
      const out = new Float32Array(p)
      for (let i = 0; i < p; i++) out[i] = q[id * p + i] * scale
      return out
    },
    text(id) { check(id); return tokens[id] },
  }
}

export function coversModel(manifest: GeometryManifest, modelId: string): boolean {
  return manifest.modelId === modelId
    || (GEOMETRY_MODEL_IDS.includes(manifest.modelId) && GEOMETRY_MODEL_IDS.includes(modelId))
}

let pending: Promise<GeometryAsset> | null = null

export function loadGeometry(baseUrl: string = GEOMETRY_BASE_URL): Promise<GeometryAsset> {
  if (!pending) {
    pending = fetchGeometry(baseUrl).catch((err) => { pending = null; throw err })
  }
  return pending
}

export function resetGeometryCache(): void { pending = null }

async function fetchGeometry(base: string): Promise<GeometryAsset> {
  const get = async (name: string) => {
    const r = await fetch(`${base}/${name}`)
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`)
    return r
  }
  const manifest = (await (await get('manifest.json')).json()) as GeometryManifest
  const [neighbors, vectors, tokens] = await Promise.all([
    get('neighbors.bin').then((r) => r.arrayBuffer()),
    get('vectors.bin').then((r) => r.arrayBuffer()),
    get('tokens.json').then((r) => r.json() as Promise<string[]>),
  ])
  return parseGeometry(manifest, neighbors, vectors, tokens)
}
```

- [ ] **Step 7: Verify**

Run: `npx vitest run src/geometry/asset.test.ts && npx tsc --noEmit -p tsconfig.app.json && npm run lint`
Expected: 9 passed, clean.

- [ ] **Step 8: Commit**

```bash
git add src/engine/tokenizer.ts src/geometry/asset.ts src/geometry/asset.test.ts src/test/geometryFixture.ts src/test/setup.ts
git commit -F- <<'MSG'
feat(geometry): lazily loaded vocabulary-geometry asset with binary parser

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 8: Geometry math

**Files:**
- Create: `src/geometry/math.ts`
- Test: `src/geometry/math.test.ts`

**Interfaces:**
- Consumes: `GeometryAsset` (Task 7).
- Produces: `cosine(a: ArrayLike<number>, b: ArrayLike<number>): number` (0 when either vector is all zeros); `similarityMatrix(vectors: ArrayLike<number>[]): number[][]` (symmetric, diagonal 1 for non-zero vectors, rounded to 3 decimals); `isRenderableToken(text: string): boolean`; `renderableNeighbors(asset: GeometryAsset, id: number, n: number): Array<{ id: number; sim: number; text: string }>`; `poolRow(row: ArrayLike<number>, cells: number): number[]` (mean-pool into at most `cells` buckets; identity when `row.length <= cells`).

- [ ] **Step 1: Write the failing tests**

Create `src/geometry/math.test.ts`:

```ts
import { expect, test } from 'vitest'
import { makeGeometryAsset } from '../test/geometryFixture'
import { cosine, isRenderableToken, poolRow, renderableNeighbors, similarityMatrix } from './math'

test('cosine of parallel, orthogonal and zero vectors', () => {
  expect(cosine([1, 2], [2, 4])).toBeCloseTo(1, 6)
  expect(cosine([1, 0], [0, 1])).toBeCloseTo(0, 6)
  expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1, 6)
  expect(cosine([0, 0], [1, 1])).toBe(0)
})

test('similarityMatrix is symmetric with a unit diagonal', () => {
  const m = similarityMatrix([[1, 0], [0, 1], [1, 1]])
  expect(m).toEqual([[1, 0, 0.707], [0, 1, 0.707], [0.707, 0.707, 1]])
})

test('isRenderableToken filters byte fallbacks, empties, whitespace and control chars', () => {
  expect(isRenderableToken(' cat')).toBe(true)
  expect(isRenderableToken('<0x0A>')).toBe(false)
  expect(isRenderableToken('')).toBe(false)
  expect(isRenderableToken('   ')).toBe(false)
  expect(isRenderableToken('a\u0007b')).toBe(false)
})

test('renderableNeighbors skips unrenderable entries and returns n', () => {
  const a = makeGeometryAsset()
  const n = renderableNeighbors(a, 6, 8)   // ids 7 and 8 are unrenderable and among 6's closest
  expect(n).toHaveLength(8)
  expect(n.map((x) => x.id)).not.toContain(7)
  expect(n.map((x) => x.id)).not.toContain(8)
  expect(n[0].text).toBe('t5')
  expect(n[0].sim).toBeGreaterThan(0.9)
})

test('poolRow mean-pools into buckets and is the identity for short rows', () => {
  expect(poolRow([1, 2, 3, 4, 5, 6], 3)).toEqual([1.5, 3.5, 5.5])
  expect(poolRow([1, 2, 3], 8)).toEqual([1, 2, 3])
  expect(poolRow([], 4)).toEqual([])
  expect(poolRow(Array.from({ length: 576 }, (_, i) => i), 96)).toHaveLength(96)
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/geometry/math.test.ts`
Expected: cannot resolve `./math`.

- [ ] **Step 3: Implement `src/geometry/math.ts`**

```ts
import type { GeometryAsset } from './asset'

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb)
}

const r3 = (v: number) => Math.round(v * 1000) / 1000

export function similarityMatrix(vectors: ArrayLike<number>[]): number[][] {
  const n = vectors.length
  const m: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = r3(cosine(vectors[i], vectors[j]))
      m[i][j] = v
      m[j][i] = v
    }
  }
  return m
}

const BYTE_FALLBACK = /^<0x[0-9A-Fa-f]{2}>$/
const CONTROL = /\p{Cc}/u

export function isRenderableToken(text: string): boolean {
  return text.trim().length > 0 && !BYTE_FALLBACK.test(text) && !CONTROL.test(text)
}

export function renderableNeighbors(asset: GeometryAsset, id: number, n: number): Array<{ id: number; sim: number; text: string }> {
  const out: Array<{ id: number; sim: number; text: string }> = []
  for (const nb of asset.neighbors(id)) {
    const text = asset.text(nb.id)
    if (!isRenderableToken(text)) continue
    out.push({ ...nb, text })
    if (out.length === n) break
  }
  return out
}

export function poolRow(row: ArrayLike<number>, cells: number): number[] {
  const len = row.length
  if (len <= cells) return Array.from(row)
  const out: number[] = []
  for (let c = 0; c < cells; c++) {
    const start = Math.floor((c * len) / cells)
    const end = Math.floor(((c + 1) * len) / cells)
    let sum = 0
    for (let i = start; i < end; i++) sum += row[i]
    out.push(sum / (end - start))
  }
  return out
}
```

- [ ] **Step 4: Verify and commit**

Run: `npx vitest run src/geometry/math.test.ts && npx tsc --noEmit -p tsconfig.app.json && npm run lint`
Expected: 5 passed, clean.

```bash
git add src/geometry/math.ts src/geometry/math.test.ts
git commit -F- <<'MSG'
feat(geometry): cosine, similarity matrix, neighbour filtering, row pooling

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 9: `useGeometry` hook and the `embeddingRows` selector

**Files:**
- Create: `src/geometry/useGeometry.ts`
- Modify: `src/viz/selectors.ts`
- Test: `src/geometry/useGeometry.test.tsx`, `src/viz/selectors.test.ts`

**Interfaces:**
- Consumes: `loadGeometry`, `GeometryAsset` (Task 7); `visibleTokens`, fixtures with `embedRows` (Task 5).
- Produces: `useGeometry(): GeometryState` where `GeometryState = { status: 'idle' | 'loading' | 'ready' | 'error'; asset?: GeometryAsset; error?: string; retry(): void }`; `embeddingRows(events, cursor): EmbeddingRows` where `EmbeddingRows = { tokens: TokenInfo[]; rows?: number[][]; source: EmbedSource }`; `thousands(n: number): string` becomes an export of `selectors.ts`.

- [ ] **Step 1: Write the failing hook test**

Create `src/geometry/useGeometry.test.tsx`:

```tsx
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test } from 'vitest'
import { stubGeometryFetch } from '../test/geometryFixture'
import { resetGeometryCache } from './asset'
import { useGeometry } from './useGeometry'

beforeEach(() => resetGeometryCache())
afterEach(() => resetGeometryCache())

test('loads on mount and reports ready with the asset', async () => {
  stubGeometryFetch()
  const { result } = renderHook(() => useGeometry())
  expect(result.current.status).toBe('loading')
  await waitFor(() => expect(result.current.status).toBe('ready'))
  expect(result.current.asset?.text(3)).toBe('t3')
})

test('reports error with the message, and retry re-fetches', async () => {
  stubGeometryFetch({ 'manifest.json': () => { throw new Error('offline') } })
  const { result } = renderHook(() => useGeometry())
  await waitFor(() => expect(result.current.status).toBe('error'))
  expect(result.current.error).toBe('offline')
  const ok = stubGeometryFetch()
  act(() => result.current.retry())
  await waitFor(() => expect(result.current.status).toBe('ready'))
  expect(ok).toHaveBeenCalledTimes(4)
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/geometry/useGeometry.test.tsx`
Expected: cannot resolve `./useGeometry`.

- [ ] **Step 3: Implement the hook**

Create `src/geometry/useGeometry.ts`:

```ts
import { useCallback, useEffect, useState } from 'react'
import { loadGeometry, type GeometryAsset } from './asset'

export type GeometryStatus = 'idle' | 'loading' | 'ready' | 'error'
export interface GeometryState {
  status: GeometryStatus
  asset?: GeometryAsset
  error?: string
  retry(): void
}

// Kicks off the shared, once-per-page asset load on first mount; `retry`
// re-runs it after a failure (loadGeometry drops a rejected promise, so the
// retry really re-fetches).
export function useGeometry(): GeometryState {
  const [state, setState] = useState<Omit<GeometryState, 'retry'>>({ status: 'idle' })
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let live = true
    setState({ status: 'loading' })
    loadGeometry().then(
      (asset) => { if (live) setState({ status: 'ready', asset }) },
      (err: unknown) => { if (live) setState({ status: 'error', error: err instanceof Error ? err.message : String(err) }) },
    )
    return () => { live = false }
  }, [attempt])
  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  return { ...state, retry }
}
```

- [ ] **Step 4: Run the hook test**

Run: `npx vitest run src/geometry/useGeometry.test.tsx`
Expected: 2 passed.

- [ ] **Step 5: Write the failing selector tests**

Append to `src/viz/selectors.test.ts` (add `buildFixtureTrace, fixtureEmbedding` to the fixtures import and `embeddingRows` to the selectors import):

```ts
// fixture indices: 2 = cycle-0 embed, 10 = cycle-0 append, 11 = cycle-1 embed
test('embeddingRows: asset-source runs return tokens only', () => {
  const r = embeddingRows(makeFixtureTrace(), 2)
  expect(r.source).toBe('asset')
  expect(r.rows).toBeUndefined()
  expect(r.tokens.map((t) => t.id)).toEqual([10, 11])
})

test('embeddingRows: model-source runs return one row per visible token', () => {
  const t = buildFixtureTrace({ embedRows: true })
  const c0 = embeddingRows(t, 2)
  expect(c0.source).toBe('model')
  expect(c0.rows).toHaveLength(2)
  expect(c0.rows?.[1]).toEqual(fixtureEmbedding(11))
  const c1 = embeddingRows(t, 11)
  expect(c1.tokens).toHaveLength(3)
  expect(c1.rows).toHaveLength(3)
  expect(c1.rows?.[2]).toEqual(fixtureEmbedding(100))
})

test('embeddingRows: a later asset-source cycle degrades the whole run to asset', () => {
  const t = buildFixtureTrace({ embedRows: true })
  const e = t[11]
  if (e.type === 'embed') { e.source = 'asset'; delete e.rows }
  expect(embeddingRows(t, 2).source).toBe('model')
  expect(embeddingRows(t, 11).source).toBe('asset')
})

test('embeddingRows: a token without a known row yet (cursor on append) falls back to asset', () => {
  const t = buildFixtureTrace({ embedRows: true })
  const r = embeddingRows(t, 10)
  expect(r.tokens).toHaveLength(3)
  expect(r.source).toBe('asset')
})
```

- [ ] **Step 6: Run them to verify they fail**

Run: `npx vitest run src/viz/selectors.test.ts`
Expected: `embeddingRows` is not a function.

- [ ] **Step 7: Implement the selector**

In `src/viz/selectors.ts`: change `const thousands = …` to `export const thousands = …`; add `EmbedSource` to the type import; append:

```ts
export interface EmbeddingRows {
  tokens: TokenInfo[]
  rows?: number[][]     // one per token, only when every visible position has a model row
  source: EmbedSource
}

// The visible sequence and, when the run recorded exact rows for all of it,
// those rows in position order. Cycle c's embed event carries the rows fed
// that cycle (the prompt at cycle 0, the previous chosen token after), so at
// an embed cursor the rows cover exactly the visible tokens. Any asset-source
// cycle, or a position without a row yet, degrades the whole run to 'asset'
// so the card never mixes exact and reduced vectors.
export function embeddingRows(events: TraceEvent[], cursor: number): EmbeddingRows {
  const { prompt, generated } = visibleTokens(events, cursor)
  const tokens: TokenInfo[] = [...prompt, ...generated.map(({ id, text }) => ({ id, text }))]
  const rows: number[][] = []
  let model = true
  for (const e of events.slice(0, cursor + 1)) {
    if (e.type !== 'embed') continue
    if (e.source !== 'model' || !e.rows) { model = false; break }
    rows.push(...e.rows)
  }
  if (!model || rows.length < tokens.length) return { tokens, source: 'asset' }
  return { tokens, rows: rows.slice(0, tokens.length), source: 'model' }
}
```

- [ ] **Step 8: Verify and commit**

Run: `npx vitest run src/viz/selectors.test.ts src/geometry && npx tsc --noEmit -p tsconfig.app.json && npm run lint`
Expected: all pass, clean.

```bash
git add src/geometry/useGeometry.ts src/geometry/useGeometry.test.tsx src/viz/selectors.ts src/viz/selectors.test.ts
git commit -F- <<'MSG'
feat(geometry): useGeometry hook and embeddingRows selector

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 10: The mechanism section — `EmbeddingLookup` and the rebuilt card container

**Files:**
- Create: `src/viz/EmbeddingLookup.tsx`
- Rewrite: `src/viz/details/EmbeddingsDetail.tsx`
- Modify: `src/viz/DetailPanel.tsx:34-37`
- Modify: `src/index.css` (append a block)
- Test: `src/viz/details/EmbeddingsDetail.test.tsx` (new), `src/viz/DetailPanel.test.tsx`

**Interfaces:**
- Consumes: `embeddingRows`, `thousands` (Task 9); `useGeometry` (Task 9); `coversModel` (Task 7); `poolRow` (Task 8).
- Produces: `EmbeddingLookup` props `{ tokens: TokenInfo[]; dims: number; vocabSize?: number; selected: number; onSelect(pos: number): void; vectorFor(pos: number): ArrayLike<number> | undefined; source: EmbedSource; missingNote: string }`; `EmbeddingsDetail` props `{ events: TraceEvent[]; cursor: number }`. Task 11 adds `EmbeddingGeometry` under the lookup inside this container.

- [ ] **Step 1: Write the failing tests**

Create `src/viz/details/EmbeddingsDetail.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace } from '../../test/fixtures'
import { makeGeometryAsset } from '../../test/geometryFixture'
import type { GeometryState } from '../../geometry/useGeometry'
import { EmbeddingsDetail } from './EmbeddingsDetail'

const ready = (): GeometryState => ({ status: 'ready', asset: makeGeometryAsset(), retry: () => {} })
let geo: GeometryState = ready()
vi.mock('../../geometry/useGeometry', () => ({ useGeometry: () => geo }))
afterEach(() => { cleanup(); geo = ready() })

// fixture indices: 2 = cycle-0 embed (The, cat), 11 = cycle-1 embed (The, cat, sat)

test('lists the visible tokens as chips with the newest selected', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={11} />)
  const chips = screen.getAllByTestId('embed-token')
  expect(chips).toHaveLength(3)
  expect(chips[2].dataset.selected).toBe('true')
  expect(screen.getByTestId('detail-embeddings')).toHaveTextContent('49 152 × 576')
})

test('clicking a chip selects that row', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={11} />)
  fireEvent.click(screen.getAllByTestId('embed-token')[0])
  expect(screen.getAllByTestId('embed-token')[0].dataset.selected).toBe('true')
  expect(screen.getByTestId('embed-lookup')).toHaveTextContent('row 10')
})

test('model-source rows render a 96-cell pooled strip', () => {
  render(<EmbeddingsDetail events={buildFixtureTrace({ embedRows: true })} cursor={2} />)
  expect(screen.getAllByTestId('embed-strip-cell')).toHaveLength(96)
  expect(screen.getByTestId('embed-lookup')).toHaveTextContent('mean-pooled into 96 cells')
})

test('asset-source rows come from the geometry asset (fixture vectors have 4 dims)', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getAllByTestId('embed-strip-cell')).toHaveLength(4)
  expect(screen.getByTestId('embed-lookup')).toHaveTextContent('PCA-reduced')
})

test('asset source without geometry shows the offline placeholder', () => {
  geo = { status: 'error', error: 'offline', retry: () => {} }
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getByTestId('embed-strip-missing')).toHaveTextContent('unavailable offline')
})

test('asset source while geometry loads says so', () => {
  geo = { status: 'loading', retry: () => {} }
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getByTestId('embed-strip-missing')).toHaveTextContent(/loading/i)
})

test('three callouts carry their explanations', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  const callouts = screen.getAllByTestId('embed-callout')
  expect(callouts.map((c) => c.textContent)).toEqual(['ⓘ learned, not designed', 'ⓘ no position here', 'ⓘ tied with Logits'])
  expect(callouts[1].getAttribute('title')).toMatch(/rotary/)
})

test('stack has one row per token with the newest marked', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={11} />)
  const rows = screen.getAllByTestId('embed-stack-row')
  expect(rows).toHaveLength(3)
  expect(rows[2].dataset.newest).toBe('true')
})
```

In `src/viz/DetailPanel.test.tsx` add, after the canvas mock:

```ts
vi.mock('../geometry/useGeometry', () => ({ useGeometry: () => ({ status: 'loading', retry: () => {} }) }))
```

and change the embeddings test to:

```ts
test('embeddings detail shows the embedding-table shape', () => {
  render(<DetailPanel events={trace} cursor={2} mode="sim" />)
  expect(screen.getByTestId('detail-embeddings')).toHaveTextContent('49 152 × 576')
  expect(screen.getAllByTestId('embed-token')).toHaveLength(2)
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/viz/details/EmbeddingsDetail.test.tsx src/viz/DetailPanel.test.tsx`
Expected: failures (props mismatch, missing test ids).

- [ ] **Step 3: Create `src/viz/EmbeddingLookup.tsx`**

```tsx
import { poolRow } from '../geometry/math'
import type { EmbedSource, TokenInfo } from '../trace/types'
import { thousands } from './selectors'

const STRIP_CELLS = 96
const CELL_W = 5
const CELL_H = 14

const CALLOUTS: Array<{ label: string; text: string }> = [
  { label: 'learned, not designed',
    text: 'Every row starts random and is adjusted during training so that the rest of the network predicts well. Nobody chose what dimension 17 means.' },
  { label: 'no position here',
    text: 'SmolLM2 adds no position vector at this stage. Position enters inside attention, via rotary position embeddings applied to queries and keys.' },
  { label: 'tied with Logits',
    text: 'The same 49152 × 576 matrix is reused at the end: the final vector is compared against every row to score each token (tie_word_embeddings).' },
]

// The mechanism half of the Embeddings card: ids → one row of E → the stacked
// residual stream x. Pure presentation; the container decides where vectors
// come from (trace rows or the geometry asset).
export function EmbeddingLookup({ tokens, dims, vocabSize, selected, onSelect, vectorFor, source, missingNote }: {
  tokens: TokenInfo[]
  dims: number
  vocabSize?: number
  selected: number
  onSelect: (pos: number) => void
  vectorFor: (pos: number) => ArrayLike<number> | undefined
  source: EmbedSource
  missingNote: string
}) {
  const token = tokens[selected]
  const vec = token ? vectorFor(selected) : undefined
  const cells = vec ? poolRow(vec, STRIP_CELLS) : []
  const peak = cells.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1
  const vocab = vocabSize ? thousands(vocabSize) : '?'
  const rowY = token ? 40 + (token.id % 41) : 0   // a schematic position inside E
  return (
    <div data-testid="embed-lookup" className="embed-lookup">
      <div className="embed-ids">
        <div className="embed-col-label">ids</div>
        <div className="token-chip-row embed-chips">
          {tokens.map((t, i) => (
            <button key={i} type="button" data-testid="embed-token" data-selected={String(i === selected)}
              className={`token-chip embed-chip hue-${i % 6}`} onClick={() => onSelect(i)}
              title={`Show row ${t.id} of E`}>
              <span className="chip-text">{t.text}</span>
              <span className="chip-id">{t.id}</span>
            </button>
          ))}
        </div>
      </div>
      <svg className="embed-matrix" width="150" height="110" viewBox="0 0 150 110" role="img"
        aria-label={`embedding matrix E, ${vocab} rows by ${dims} columns`}>
        <path d="M4 55 h26" className="rs-branch" />
        <path d="M32 55 l-7 -4 v8 z" className="rs-arrowhead" />
        <rect x="36" y="6" width="90" height="98" rx="4" className="rs-box" />
        <text x="81" y="24" textAnchor="middle" className="rs-box-label">E</text>
        <text x="81" y="98" textAnchor="middle" className="rs-shape">[{vocab} × {dims}]</text>
        {token && <rect x="36" y={rowY} width="90" height="4" className="embed-row-mark" />}
      </svg>
      <div className="embed-strip-wrap">
        <div className="embed-col-label">row {token?.id ?? '—'}</div>
        {cells.length > 0 ? (
          <svg data-testid="embed-strip" width={cells.length * CELL_W} height={CELL_H}
            viewBox={`0 0 ${cells.length * CELL_W} ${CELL_H}`} role="img"
            aria-label={`embedding row of token ${token?.text ?? ''}`}>
            {cells.map((v, c) => (
              <rect key={c} data-testid="embed-strip-cell" x={c * CELL_W} y={0} width={CELL_W - 1} height={CELL_H}
                fill={v >= 0
                  ? `hsl(211 45% ${Math.round(88 - (Math.abs(v) / peak) * 50)}%)`
                  : `hsl(13 55% ${Math.round(90 - (Math.abs(v) / peak) * 40)}%)`} />
            ))}
          </svg>
        ) : (
          <div data-testid="embed-strip-missing" className="embed-strip-missing">{missingNote}</div>
        )}
        <div className="embed-caption">
          {source === 'model'
            ? `${dims} values, mean-pooled into ${STRIP_CELLS} cells`
            : `${cells.length || 64} of ${dims} dimensions (PCA-reduced, offline)`}
        </div>
      </div>
      <div className="embed-stack">
        <div className="embed-col-label">x [{tokens.length} × {dims}]</div>
        <div data-testid="embed-stack" className="embed-stack-rows">
          {tokens.map((_, i) => (
            <div key={i} data-testid="embed-stack-row" data-newest={String(i === tokens.length - 1)}
              data-selected={String(i === selected)} className="embed-stack-row" />
          ))}
        </div>
        <div className="embed-caption">one row per token; the newest was added this cycle</div>
      </div>
      <div className="embed-callouts">
        {CALLOUTS.map((c) => (
          <span key={c.label} data-testid="embed-callout" className="embed-callout rs-hover" title={c.text}>ⓘ {c.label}</span>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Rewrite `src/viz/details/EmbeddingsDetail.tsx`**

```tsx
import { useState } from 'react'
import { coversModel } from '../../geometry/asset'
import { useGeometry } from '../../geometry/useGeometry'
import type { TraceEvent } from '../../trace/types'
import { EmbeddingLookup } from '../EmbeddingLookup'
import { embeddingRows, latestOfType, thousands } from '../selectors'

// Container: resolves where vectors come from (exact trace rows, else the
// geometry asset by id) and owns the selected position. Pure function of
// (events, cursor, asset) apart from that one piece of UI state.
export function EmbeddingsDetail({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const embed = latestOfType(events, cursor, 'embed')
  const runStart = latestOfType(events, cursor, 'run-start')
  const { tokens, rows, source } = embeddingRows(events, cursor)
  // snaps to the newest token when unset, or when scrubbing back past the pick
  const [picked, setPicked] = useState<number | null>(null)
  const selected = picked == null || picked >= tokens.length ? tokens.length - 1 : picked
  const geo = useGeometry()
  const covered = geo.status === 'ready' && geo.asset != null && runStart != null
    && coversModel(geo.asset.manifest, runStart.modelId)
  const asset = covered ? geo.asset : undefined
  const pending = geo.status === 'idle' || geo.status === 'loading'
  if (!embed) return null
  const dims = embed.dims
  const vocab = runStart?.vocabSize
  const vectorFor = (pos: number): ArrayLike<number> | undefined =>
    rows ? rows[pos] : asset?.vector(tokens[pos].id)
  return (
    <div data-testid="detail-embeddings" className="detail">
      <h3>Embeddings</h3>
      <p>
        A lookup, not a computation. Each token id selects one row of a learned matrix{' '}
        <code>E [{vocab ? thousands(vocab) : '?'} × {dims}]</code>; the rows stacked up are{' '}
        <code>x [{tokens.length} × {dims}]</code>.
      </p>
      <EmbeddingLookup tokens={tokens} dims={dims} vocabSize={vocab} selected={selected} onSelect={setPicked}
        vectorFor={vectorFor} source={source}
        missingNote={pending ? 'loading vocabulary geometry…' : 'vector values unavailable offline'} />
    </div>
  )
}
```

(Task 11 adds the geometry section below the lookup.)

- [ ] **Step 5: Wire `DetailPanel.tsx`**

Replace the `case 'embeddings'` block with:

```tsx
    case 'embeddings': {
      const e = latestOfType(events, cursor, 'embed')
      return e ? <EmbeddingsDetail events={events} cursor={cursor} /> : empty
    }
```

- [ ] **Step 6: Styles**

Append to `src/index.css`:

```css
/* ── Embeddings card ─────────────────────────────────────────────────── */
.embed-lookup { display: grid; grid-template-columns: minmax(8rem, 16rem) auto minmax(12rem, 1fr) auto;
  gap: .8rem 1.2rem; align-items: start; margin: .4rem 0 .6rem; }
.embed-col-label { font-family: var(--mono); font-size: .72rem; color: var(--ai-deep); margin-bottom: .3rem; }
.embed-chips { margin-top: 0; }
.embed-chip { font: inherit; border: 1px solid transparent; cursor: pointer; }
.embed-chip[data-selected="true"] { border-color: var(--shu); box-shadow: 0 0 0 2px var(--shu-wash); }
.embed-row-mark { fill: var(--shu); }
.embed-strip-wrap { overflow-x: auto; }
.embed-strip-missing { font-size: .78rem; color: var(--ink-faint); padding: .2rem 0; }
.embed-caption { font-size: .72rem; color: var(--ink-soft); margin-top: .25rem; }
.embed-stack-rows { display: flex; flex-direction: column; gap: 2px; width: 6rem; }
.embed-stack-row { height: 4px; background: var(--ai-soft); border-radius: 1px; }
.embed-stack-row[data-selected="true"] { background: var(--shu); }
.embed-stack-row[data-newest="true"] { outline: 1px solid var(--ai-deep); }
.embed-callouts { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: .4rem 1.2rem;
  font-size: .78rem; color: var(--ai-deep); }
.embed-callout { border-bottom: 1px dotted var(--ai-soft); }
.embed-geometry { border-top: 1px solid var(--line); margin-top: .8rem; padding-top: .6rem; }
.embed-geometry h4 { margin: 0 0 .4rem; font-size: .9rem; color: var(--ai-deep); }
.embed-neighbors { margin-bottom: .8rem; }
.embed-sim { overflow-x: auto; }
```

- [ ] **Step 7: Verify**

Run: `npx vitest run src/viz && npx tsc --noEmit -p tsconfig.app.json && npm run lint && npm test`
Expected: all pass, clean. (App-level tests that reach the Embeddings card now hit the setup fetch stub, which rejects instantly — no network.)

- [ ] **Step 8: Commit**

```bash
git add src/viz/EmbeddingLookup.tsx src/viz/details/EmbeddingsDetail.tsx src/viz/details/EmbeddingsDetail.test.tsx src/viz/DetailPanel.tsx src/viz/DetailPanel.test.tsx src/index.css
git commit -F- <<'MSG'
feat(viz): Embeddings card teaches the lookup — ids, row of E, stacked stream

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 11: The geometry section — `EmbeddingGeometry`

**Files:**
- Create: `src/viz/EmbeddingGeometry.tsx`
- Modify: `src/viz/details/EmbeddingsDetail.tsx` (mount it)
- Test: `src/viz/EmbeddingGeometry.test.tsx`, `src/viz/details/EmbeddingsDetail.test.tsx`

**Interfaces:**
- Consumes: `renderableNeighbors`, `similarityMatrix` (Task 8); `GeometryAsset` (Task 7); the container's `vectorFor`/`selected` (Task 10).
- Produces: `EmbeddingGeometry` props `{ tokens: TokenInfo[]; selected: number; vectorFor(pos): ArrayLike<number> | undefined; asset?: GeometryAsset; loading: boolean; error?: string; retry(): void; source: EmbedSource }`.

- [ ] **Step 1: Write the failing tests**

Create `src/viz/EmbeddingGeometry.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { fixtureVector, makeGeometryAsset } from '../test/geometryFixture'
import type { TokenInfo } from '../trace/types'
import { EmbeddingGeometry } from './EmbeddingGeometry'

afterEach(() => cleanup())
const asset = makeGeometryAsset()
const toks = (ids: number[]): TokenInfo[] => ids.map((id) => ({ id, text: ` t${id}` }))
const vecFor = (tokens: TokenInfo[]) => (pos: number) => fixtureVector(tokens[pos].id)
const noop = () => {}

test('lists 8 renderable neighbours of the selected token with similarities', () => {
  const tokens = toks([6, 20])
  render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  const rows = screen.getAllByTestId('embed-neighbor')
  expect(rows).toHaveLength(8)
  expect(rows[0]).toHaveTextContent('t5')          // ids 7 and 8 are unrenderable and skipped
  expect(screen.getByTestId('embed-neighbors')).not.toHaveTextContent('<0x07>')
  expect(rows[0]).toHaveTextContent(/0\.9\d/)
})

test('similarity matrix has n² cells and no cap note for short sequences', () => {
  const tokens = toks([1, 2, 130])
  render(<EmbeddingGeometry tokens={tokens} selected={2} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  expect(screen.getAllByTestId('sim-cell')).toHaveLength(9)
  expect(screen.queryByTestId('embed-sim-cap')).toBeNull()
})

test('matrix is capped at the last 24 tokens with a note', () => {
  const tokens = toks(Array.from({ length: 30 }, (_, i) => i + 40))
  render(<EmbeddingGeometry tokens={tokens} selected={29} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  expect(screen.getAllByTestId('sim-cell')).toHaveLength(24 * 24)
  expect(screen.getByTestId('embed-sim-cap')).toHaveTextContent('last 24 of 30')
})

test('error state shows the message and retry calls back; no matrix without vectors', () => {
  const retry = vi.fn()
  render(<EmbeddingGeometry tokens={toks([1])} selected={0} vectorFor={() => undefined}
    loading={false} error="offline" retry={retry} source="asset" />)
  expect(screen.getByTestId('embed-geometry-error')).toHaveTextContent("Vocabulary geometry couldn't be loaded")
  fireEvent.click(screen.getByTestId('embed-geometry-retry'))
  expect(retry).toHaveBeenCalledTimes(1)
  expect(screen.queryByTestId('embed-similarity')).toBeNull()
  expect(screen.queryByTestId('embed-neighbors')).toBeNull()
})

test('loading state', () => {
  render(<EmbeddingGeometry tokens={toks([1])} selected={0} vectorFor={() => undefined}
    loading={true} retry={noop} source="asset" />)
  expect(screen.getByTestId('embed-geometry-loading')).toBeInTheDocument()
})

test('provenance caption follows the source', () => {
  const tokens = toks([1, 2])
  const { unmount } = render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="model" />)
  expect(screen.getByTestId('embed-provenance')).toHaveTextContent('Exact rows from the running model.')
  unmount()
  render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  expect(screen.getByTestId('embed-provenance')).toHaveTextContent('reduced to 64 dimensions offline; similarities are approximate.')
})
```

Append to `src/viz/details/EmbeddingsDetail.test.tsx`:

```tsx
test('geometry section shows neighbours of the selected token and the matrix', () => {
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={11} />)
  expect(screen.getAllByTestId('embed-neighbor')).toHaveLength(8)
  expect(screen.getAllByTestId('sim-cell')).toHaveLength(9)
  fireEvent.click(screen.getAllByTestId('embed-token')[0])            // id 10
  expect(screen.getAllByTestId('embed-neighbor')[0]).toHaveTextContent(/t(9|11)/)
})

test('geometry error is shown in the geometry section with a retry', () => {
  geo = { status: 'error', error: 'offline', retry: () => {} }
  render(<EmbeddingsDetail events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getByTestId('embed-geometry-error')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/viz/EmbeddingGeometry.test.tsx src/viz/details/EmbeddingsDetail.test.tsx`
Expected: cannot resolve `./EmbeddingGeometry`; missing test ids.

- [ ] **Step 3: Create `src/viz/EmbeddingGeometry.tsx`**

```tsx
import type { GeometryAsset } from '../geometry/asset'
import { renderableNeighbors, similarityMatrix } from '../geometry/math'
import type { EmbedSource, TokenInfo } from '../trace/types'

const NEIGHBORS = 8
const MATRIX_CAP = 24
const CELL = 18
const LABEL_W = 60
const DIAG_PAD = 52
const RIGHT_PAD = 60

// The geometry half of the Embeddings card: nearest vocabulary neighbours of
// the selected token, and the visible sequence's self-similarity — what the
// model "knows" about these tokens before any context is applied.
export function EmbeddingGeometry({ tokens, selected, vectorFor, asset, loading, error, retry, source }: {
  tokens: TokenInfo[]
  selected: number
  vectorFor: (pos: number) => ArrayLike<number> | undefined
  asset?: GeometryAsset
  loading: boolean
  error?: string
  retry: () => void
  source: EmbedSource
}) {
  const token = tokens[selected]
  const start = Math.max(0, tokens.length - MATRIX_CAP)
  const shown = tokens.slice(start)
  const vectors = shown.map((_, i) => vectorFor(start + i))
  const matrix = vectors.every((v) => v !== undefined) ? similarityMatrix(vectors as ArrayLike<number>[]) : null
  const neighbors = asset && token ? renderableNeighbors(asset, token.id, NEIGHBORS) : null
  const label = (i: number) => shown[i]?.text.trim() || `#${start + i}`
  const w = LABEL_W + shown.length * CELL + RIGHT_PAD
  const h = DIAG_PAD + shown.length * CELL + 4
  return (
    <div data-testid="embed-geometry" className="embed-geometry">
      <h4>Geometry: meaning is distance</h4>
      {loading && !asset && (
        <p data-testid="embed-geometry-loading" className="embed-caption">Loading vocabulary geometry…</p>
      )}
      {error && !asset && (
        <p data-testid="embed-geometry-error" className="notice">
          Vocabulary geometry couldn't be loaded{' '}
          <button type="button" data-testid="embed-geometry-retry" className="explore-toggle" onClick={retry}>retry</button>
        </p>
      )}
      {neighbors && token && (
        <div data-testid="embed-neighbors" className="embed-neighbors">
          <div className="embed-caption">
            Nearest to <span className="chip-text">{token.text}</span> in E — click any token above to change
          </div>
          <div className="bar-chart">
            {neighbors.map((n) => (
              <div key={n.id} data-testid="embed-neighbor" className="bar-row">
                <span className="bar-label">{n.text}</span>
                <svg width="120" height="12"><rect width={120 * n.sim} height="12" className="bar-rect" /></svg>
                <span className="bar-value">{n.sim.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {matrix && (
        <div className="embed-sim">
          <div className="embed-caption">
            Tokens vs. each other (cosine, before any context)
            {start > 0 && <span data-testid="embed-sim-cap"> — showing the last {MATRIX_CAP} of {tokens.length}</span>}
          </div>
          <svg data-testid="embed-similarity" width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img"
            aria-label="cosine similarity between the embedding rows of the visible tokens">
            {shown.map((_, i) => {
              const cx = LABEL_W + i * CELL + CELL / 2
              const cy = DIAG_PAD - 5
              return (
                <text key={`c${i}`} x={cx} y={cy} textAnchor="start" transform={`rotate(-45 ${cx} ${cy})`}
                  className="attn-label">{label(i)}</text>
              )
            })}
            {matrix.map((row, r) => (
              <g key={r}>
                <text x={LABEL_W - 8} y={DIAG_PAD + r * CELL + CELL / 2 + 4} textAnchor="end" className="attn-label">
                  {label(r)}
                </text>
                {row.map((v, c) => {
                  const m = Math.max(0, v)
                  return (
                    <rect key={c} data-testid="sim-cell" x={LABEL_W + c * CELL} y={DIAG_PAD + r * CELL}
                      width={CELL - 1} height={CELL - 1}
                      fill={`hsl(211 ${Math.round(30 + 25 * m)}% ${Math.round(94 - 70 * m)}%)`}>
                      <title>{`${label(r)} · ${label(c)}: ${v.toFixed(2)}`}</title>
                    </rect>
                  )
                })}
              </g>
            ))}
          </svg>
        </div>
      )}
      <p data-testid="embed-provenance" className="embed-caption">
        {source === 'model'
          ? 'Exact rows from the running model.'
          : 'Real SmolLM2 embedding rows, reduced to 64 dimensions offline; similarities are approximate.'}
      </p>
    </div>
  )
}
```

- [ ] **Step 4: Mount it in the container**

In `src/viz/details/EmbeddingsDetail.tsx` add `import { EmbeddingGeometry } from '../EmbeddingGeometry'`, compute the error text after `pending`:

```ts
  const geoError = geo.status === 'error' ? (geo.error ?? 'load failed')
    : geo.status === 'ready' && !covered ? 'asset does not cover this model' : undefined
```

and render, directly after `<EmbeddingLookup … />`:

```tsx
      <EmbeddingGeometry tokens={tokens} selected={selected} vectorFor={vectorFor} asset={asset}
        loading={pending} error={geoError} retry={geo.retry} source={source} />
```

- [ ] **Step 5: Verify and commit**

Run: `npx vitest run src/viz && npx tsc --noEmit -p tsconfig.app.json && npm run lint && npm test`
Expected: all pass, clean.

```bash
git add src/viz/EmbeddingGeometry.tsx src/viz/EmbeddingGeometry.test.tsx src/viz/details/EmbeddingsDetail.tsx src/viz/details/EmbeddingsDetail.test.tsx
git commit -F- <<'MSG'
feat(viz): Embeddings geometry — nearest neighbours and self-similarity matrix

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 12: Pipeline-band summary shows the table shape

**Files:**
- Modify: `src/viz/PipelineBand.tsx:21-24`
- Test: `src/viz/PipelineBand.test.tsx`

**Interfaces:**
- Consumes: `thousands` (Task 9).

- [ ] **Step 1: Write the failing test**

Append to `src/viz/PipelineBand.test.tsx`, following the file's existing render pattern (it renders `<PipelineBand events={…} cursor={…} />` with `makeFixtureTrace()`); if an existing assertion expects `576 dims`, change it to the new text:

```tsx
test('embeddings summary shows the embedding-table shape', () => {
  render(<PipelineBand events={makeFixtureTrace()} cursor={2} />)
  expect(screen.getAllByTestId('stage-card')[1]).toHaveTextContent('49 152 × 576')
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/viz/PipelineBand.test.tsx`
Expected: text `576 dims` instead of the shape.

- [ ] **Step 3: Implement**

In `src/viz/PipelineBand.tsx` add `thousands` to the `./selectors` import and replace the `embeddings` case:

```ts
    case 'embeddings': {
      const e = latestOfType(events, cursor, 'embed')
      if (!e) return null
      const vocab = latestOfType(events, cursor, 'run-start')?.vocabSize
      return vocab ? `${thousands(vocab)} × ${e.dims}` : `${e.dims} dims`
    }
```

- [ ] **Step 4: Verify and commit**

Run: `npx vitest run src/viz/PipelineBand.test.tsx && npx tsc --noEmit -p tsconfig.app.json && npm run lint`
Expected: pass, clean.

```bash
git add src/viz/PipelineBand.tsx src/viz/PipelineBand.test.tsx
git commit -F- <<'MSG'
feat(viz): Embeddings stage summary shows the embedding-table shape

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

---

### Task 13: Screenshot scene, README section, backlog

**Files:**
- Modify: `scripts/capture-screenshots.mjs`
- Modify: `README.md`
- Modify: `docs/BACKLOG.md`

This task needs the republished model and geometry on the Hub (Task 4). If Task 4 has not completed, do Steps 2–4 (docs) and leave Step 1's capture to the operator, noting it in the report.

- [ ] **Step 1: Add the `embeddings` scene**

In `scripts/capture-screenshots.mjs`, change the real-mode guard to
`if (want('attention') || want('compare') || want('embeddings')) {` and add, after the compare block (still inside the `try`):

```js
  // ---- embeddings.png: real mode, Embeddings stage — lookup + neighbours + matrix
  if (want('embeddings')) {
    await $('prompt-input').fill('The cat sat on the mat because the cat was tired')
    const before = await $('run-chip').count()
    await $('btn-generate').click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })   // wait for the run to SEAL
    await $('btn-live').click()
    await $('stage-card').nth(1).click()                            // Embeddings
    await $('embed-neighbors').waitFor({ timeout: 60000 })          // geometry asset fetched from the Hub
    await $('detail-embeddings').scrollIntoViewIfNeeded()
    await page.screenshot({ path: `${DIR}/embeddings.png` })
    console.log('embeddings.png')
  }
```

Update the header comment's scene list example to `pipeline attention compare embeddings`. Run `npm run screenshots -- embeddings` and check `docs/screenshots/embeddings.png` shows chips, the strip, 8 neighbours and the matrix.

- [ ] **Step 2: README section**

In `README.md`, in the "One matrix, end to end" section, replace the sentence
`The heat-strips in the Embeddings panel are (downsampled) rows of exactly the matrix that enters the first layer.`
with
`The row strip in the Embeddings panel is a row of exactly the matrix that enters the first layer.`

Insert a new section directly before `## Reading the attention heatmaps`:

```markdown
## What an embedding is — a lookup, then geometry

![The Embeddings card: the row lookup, nearest neighbours and the self-similarity matrix](docs/screenshots/embeddings.png)

The Embeddings stage is the least mysterious step in the pipeline and the
one most often hand-waved. The card makes two points, in order.

**It is a lookup, not a computation.** A token id selects one row of a
learned matrix `E [49152 × 576]`; the rows of the current sequence stacked
up are the residual stream `x [n × 576]` that the layers then edit. Click
any token chip to see its row. Three things worth knowing ride along as
hover notes: the rows are *learned*, not designed (nobody chose what
dimension 17 means); *no position* is added here — SmolLM2 applies rotary
position embeddings inside attention instead; and the same matrix is
*reused at the Logits stage* to read the answer back out (tied weights).

**Meaning is distance.** Similar tokens have similar rows. The card shows
the eight nearest vocabulary neighbours of the selected token by cosine
similarity, and a similarity matrix of the visible tokens against each
other — what the model "knows" about these tokens *before any context is
applied*, a useful contrast with the attention heatmaps one stage later.

In real mode the rows are the exact vectors from the running model (the
custom ONNX export exposes the embedding lookup as `inputs_embeds`). In
simulated mode, and for archived runs, they come from a small
vocabulary-geometry asset published next to the model on the Hugging Face
Hub: exact nearest neighbours computed offline over the full 576-dim table,
plus a PCA-64 int8 copy of every row for the similarity matrix — the caption
says which you are looking at.
```

In the `## Docs` section, add after the existing two bullet links:

```markdown
- [`specs/2026-09-02-embeddings-explained-design.md`](docs/superpowers/specs/2026-09-02-embeddings-explained-design.md)
- [`plans/2026-09-02-embeddings-explained.md`](docs/superpowers/plans/2026-09-02-embeddings-explained.md)
```

- [ ] **Step 3: Backlog**

In `docs/BACKLOG.md`, add a section before `## Real-mode quality`:

```markdown
## Pipeline pedagogy

- [x] **#11 Embeddings explained** — the Embeddings card teaches the lookup
  (ids → row of E → stacked stream, with learned/no-position/tied-weights
  notes) and the geometry (nearest neighbours, self-similarity matrix).
  Real rows via an `inputs_embeds` export output; vocabulary geometry from
  a Hub-hosted static asset. *(M4, 2026-09-02)*
- [ ] **#12 Position inside attention** — the Embeddings card says "no
  position here (RoPE)"; show where rotary position embeddings act in the
  Layers detail so the pointer lands somewhere. *(M4 spec follow-up)*
```

And in `## Done (for the record)` add at the top:
`- [x] Embeddings explained: real rows, neighbours, similarity *(M4, 2026-09-02)*`

- [ ] **Step 4: Verify and commit**

Run: `npm test && npm run lint`
Expected: pass.

```bash
git add scripts/capture-screenshots.mjs README.md docs/BACKLOG.md docs/screenshots/embeddings.png
git commit -F- <<'MSG'
docs: Embeddings section, screenshot scene, backlog #11/#12

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C5Vm2gjXesexa1FxxqruHL
MSG
```

(Omit `docs/screenshots/embeddings.png` from `git add` if Step 1 was deferred.)

---

## Final in-browser check (after Task 4 and Task 13)

1. `npm run dev`, real mode, generate a prompt with a repeated word. Open the Embeddings stage: the strip caption reads "576 values, mean-pooled into 96 cells", the provenance line reads "Exact rows from the running model.", eight neighbours appear (e.g. for ` cat`: ` cats`, ` Cat`, ` kitten`, ` dog` or similar), the matrix lights up the repeated word's pair.
2. Switch to simulated mode, generate: provenance reads "reduced to 64 dimensions offline", neighbours still appear (the asset is cached).
3. Reload an archived run from before this change (if any): the card renders with asset vectors and no errors.
4. DevTools → Network offline, reload, generate in sim mode: the geometry section shows "Vocabulary geometry couldn't be loaded" with a retry; go online, retry works.
