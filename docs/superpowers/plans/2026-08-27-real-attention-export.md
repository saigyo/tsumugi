# Real-Attention Export (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export SmolLM2-135M-Instruct to ONNX with per-layer attention outputs, publish it to the HF Hub, and make Tsumugi's real mode consume, detect, and display real attention heads.

**Architecture:** A uv-managed Python tool (`tools/export/`) produces and validates the artifact (Approach A: cache-enabled export, per-step attention rows; `--no-cache` = fallback Approach B). The worker accumulates rows into per-head matrices, scores head roles statistically, and emits the existing `attention` trace event with ≤3 detected heads. UI shows detection scores on the existing heatmap viewer.

**Tech Stack:** Python ≥3.10 + uv, optimum[exporters], onnx, onnxruntime, huggingface_hub, torch (CPU); TypeScript/Vitest for the app side.

**Spec:** `docs/superpowers/specs/2026-08-27-real-attention-export-design.md` (read it; also skim `docs/research/2026-08-27-attention-weights-in-browser.md` for the mechanism evidence)

## Global Constraints

- Python env management is **uv only**: `pyproject.toml` + `uv.lock`, run everything with `uv run`; README setup is `uv sync` and nothing else. Python ≥ 3.10.
- The Python tool is **never run in CI**; its `validate` command is its test suite. Small pure-unit pytest tests are allowed (run locally via `uv run pytest`).
- Model repo id: exactly `saigyo-hoshi/smollm2-135m-attn-onnx`. Stock model/tokenizer id stays `HuggingFaceTB/SmolLM2-135M-Instruct`; the app keeps loading the tokenizer from stock.
- Export must use `attn_implementation="eager"` and `output_attentions=True`; attention output names are exactly `attentions.0` … `attentions.{n-1}` with dynamic axes `{0: batch, 2: query_len, 3: kv_len}`.
- Worker emits ≤3 detected heads per cycle (labels `previous-token`, `attention-sink`, `induction`; min score threshold 0.3; no coreference in real mode). Never fail generation over attention data.
- TypeScript strict; `any` only in worker + HF boundary; verify with `npx tsc --noEmit -p tsconfig.app.json`; tests `npx vitest run <file>`; sim mode behavior untouched; e2e stays sim-only.
- Python API-shape caveat (mirrors the Task-18 pattern from the v1 plan): the exact optimum/onnxruntime call shapes in Tasks 2–4 must be checked against the *installed* library versions; fix mismatches in the tool, never by changing the artifact contract (output names/axes).
- Network-gated tasks (10, 11) download hundreds of MB and need the user's HF token for publish — they are operator runbooks, not CI steps.

## File Structure

```
tools/export/
  pyproject.toml                       uv project "tsumugi-export"
  README.md                            uv sync + command runbook + manual protocol
  src/tsumugi_export/__init__.py
  src/tsumugi_export/__main__.py       argparse CLI: export|quantize|validate|publish
  src/tsumugi_export/onnx_config.py    AttnLlamaOnnxConfig (adds attentions.{i})
  src/tsumugi_export/export.py
  src/tsumugi_export/quantize.py
  src/tsumugi_export/validate.py       checks + validation-report.json
  src/tsumugi_export/publish.py
  tests/test_onnx_config.py            pure-unit (no network)
src/engine/transformers/attentionAccum.ts    (+ .test.ts)  row accumulation
src/engine/transformers/attentionStats.ts    (+ .test.ts)  scores + selection
src/engine/transformers/protocol.ts          ready gains attentions
src/engine/transformers/TransformersEngine.ts  attentions flag
src/engine/transformers/worker.ts            load chain + accumulate + detect + emit
src/engine/tokenizer.ts                      ATTN_MODEL_ID
src/trace/types.ts                           AttentionHead.score?
src/viz/AttentionHeatmap.tsx                 chip scores + conditional note
src/viz/details/LayersDetail.tsx             conditional schematic tag
src/app/ModelStatus.tsx                      "· attn" chip suffix
src/App.tsx                                  attentions plumbing
```

---

### Task 1: Python tool scaffold (uv + CLI skeleton)

**Files:**
- Create: `tools/export/pyproject.toml`, `tools/export/README.md`, `tools/export/src/tsumugi_export/__init__.py`, `tools/export/src/tsumugi_export/__main__.py`, `tools/export/.gitignore`

**Interfaces:**
- Produces: `uv run python -m tsumugi_export {export|quantize|validate|publish}` CLI; each subcommand delegates to `tsumugi_export.<module>.run(args)` (modules arrive in Tasks 2–5; the skeleton stubs them with a clear "not implemented yet" exit).

- [ ] **Step 1: Create the uv project**

`tools/export/pyproject.toml`:

```toml
[project]
name = "tsumugi-export"
version = "0.1.0"
description = "Exports SmolLM2-135M-Instruct to ONNX with attention outputs for Tsumugi"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "transformers>=4.40",
    "optimum[exporters]>=1.20",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "huggingface_hub>=0.23",
    "onnxconverter-common>=1.14",
    "numpy>=1.26",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tsumugi_export"]
```

`tools/export/.gitignore`:

```
.venv/
out/
__pycache__/
*.onnx
```

- [ ] **Step 2: CLI skeleton**

`tools/export/src/tsumugi_export/__init__.py`:

```python
STOCK_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
ATTN_REPO_ID = "saigyo-hoshi/smollm2-135m-attn-onnx"
```

`tools/export/src/tsumugi_export/__main__.py`:

```python
import argparse
import importlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="tsumugi_export")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="export SmolLM2 to ONNX with attention outputs")
    p_export.add_argument("--out", default="out/model", help="output model directory")
    p_export.add_argument("--no-cache", action="store_true",
                          help="Approach B: use_cache=False, full matrices per step")

    p_quant = sub.add_parser("quantize", help="produce q4 and fp16 variants")
    p_quant.add_argument("--model-dir", default="out/model")

    p_val = sub.add_parser("validate", help="validate exported artifacts against stock")
    p_val.add_argument("--model-dir", default="out/model")

    p_pub = sub.add_parser("publish", help="assemble repo layout and upload to the HF Hub")
    p_pub.add_argument("--model-dir", default="out/model")
    p_pub.add_argument("--repo-id", default=None, help="override target repo id")

    args = parser.parse_args()
    module = importlib.import_module(f"tsumugi_export.{args.command}")
    return module.run(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: README skeleton** (`tools/export/README.md`) — setup (`uv sync`), the four commands with one-line descriptions, a note that this tool never runs in CI, and a placeholder-free "Manual verification protocol" section written out per the spec's Testing section (fast network; real mode on one curated prompt and one free prompt with a repeat; confirm `· attn` chip, plausible labels, rows sum to 100%, stock fallback works when the attn repo is unreachable).

- [ ] **Step 4: Verify**

Run (from `tools/export/`): `uv sync` then `uv run python -m tsumugi_export export --help`
Expected: help text; `uv run python -m tsumugi_export export` → clean "module has no attribute run"-style failure is acceptable at this stage ONLY if Step 5's stub note is added instead — preferred: add stub `run(args)` in four placeholder modules that `print("not implemented yet"); return 1` so every command exits cleanly. Create those four stubs now (`export.py`, `quantize.py`, `validate.py`, `publish.py`).

- [ ] **Step 5: Commit**

```bash
git add tools/export
git commit -m "feat(export): uv project scaffold with CLI skeleton"
```

---

### Task 2: Custom OnnxConfig + export command

**Files:**
- Create: `tools/export/src/tsumugi_export/onnx_config.py`, `tools/export/tests/test_onnx_config.py`
- Modify: `tools/export/src/tsumugi_export/export.py` (replace stub)

**Interfaces:**
- Consumes: `STOCK_MODEL_ID` from `tsumugi_export.__init__`
- Produces: `AttnLlamaOnnxConfig(config, task="text-generation", use_past=bool)` whose `outputs` includes `logits`, `present.*` (when `use_past`), and `attentions.{i}`; `export.run(args)` writing a loadable model dir at `args.out` containing `onnx/model.onnx` (fp32) + copied `config.json`/`generation_config.json`/tokenizer files.

- [ ] **Step 1: Write the failing pure-unit test** (`tools/export/tests/test_onnx_config.py` — no network: construct the config in code)

```python
from transformers import LlamaConfig

from tsumugi_export.onnx_config import AttnLlamaOnnxConfig


def make_config() -> LlamaConfig:
    return LlamaConfig(
        num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
        hidden_size=576, intermediate_size=1536, vocab_size=49152,
    )


def test_outputs_include_per_layer_attentions():
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=True)
    outs = cfg.outputs
    assert "logits" in outs
    for i in range(30):
        assert f"attentions.{i}" in outs
        assert outs[f"attentions.{i}"] == {0: "batch_size", 2: "query_length", 3: "kv_length"}


def test_cache_outputs_survive():
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=True)
    assert any(k.startswith("present") for k in cfg.outputs)


def test_no_cache_variant_has_no_present():
    cfg = AttnLlamaOnnxConfig(make_config(), task="text-generation", use_past=False)
    assert not any(k.startswith("present") for k in cfg.outputs)
    assert "attentions.0" in cfg.outputs
```

- [ ] **Step 2: Run to verify failure** — from `tools/export/`: `uv run pytest tests/ -v` → FAIL (module not found).

- [ ] **Step 3: Implement the config** (`tools/export/src/tsumugi_export/onnx_config.py`)

```python
"""Custom ONNX export config: stock Llama-family causal-LM outputs plus
per-layer attention weights (attentions.{i}).

API-shape caveat: verify the base-class import path and constructor signature
against the INSTALLED optimum version (optimum.exporters.onnx.model_configs
.LlamaOnnxConfig at the time of writing). Fix mismatches here; never change
the output names or axes — they are the artifact contract."""
from optimum.exporters.onnx.model_configs import LlamaOnnxConfig


class AttnLlamaOnnxConfig(LlamaOnnxConfig):
    @property
    def outputs(self):
        outs = dict(super().outputs)
        for i in range(self._config.num_hidden_layers):
            outs[f"attentions.{i}"] = {0: "batch_size", 2: "query_length", 3: "kv_length"}
        return outs
```

- [ ] **Step 4: Run tests to verify pass** — `uv run pytest tests/ -v` → PASS (3 tests).

- [ ] **Step 5: Implement the export command** (`tools/export/src/tsumugi_export/export.py`)

```python
"""Export SmolLM2 with attention outputs. Heavy: downloads torch weights
(~270 MB) on first run — network-gated, run by the operator (Task 10)."""
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from tsumugi_export import STOCK_MODEL_ID
from tsumugi_export.onnx_config import AttnLlamaOnnxConfig

TOKENIZER_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
]


def run(args) -> int:
    from optimum.exporters.onnx import main_export
    from transformers import AutoConfig

    out = Path(args.out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(STOCK_MODEL_ID)
    use_past = not args.no_cache
    custom = AttnLlamaOnnxConfig(config, task="text-generation", use_past=use_past)

    main_export(
        STOCK_MODEL_ID,
        output=str(onnx_dir),
        task="text-generation-with-past" if use_past else "text-generation",
        custom_onnx_configs={"model": custom},
        model_kwargs={"output_attentions": True, "attn_implementation": "eager"},
        do_validation=False,  # our validate command is the real gate
    )
    # optimum writes model.onnx into onnx_dir; normalize the name if needed
    exported = list(onnx_dir.glob("*.onnx"))
    if len(exported) == 1 and exported[0].name != "model.onnx":
        exported[0].rename(onnx_dir / "model.onnx")

    stock = Path(snapshot_download(STOCK_MODEL_ID, allow_patterns=TOKENIZER_FILES))
    for name in TOKENIZER_FILES:
        src = stock / name
        if src.exists():
            shutil.copy2(src, out / name)

    print(f"exported ({'with-past' if use_past else 'no-cache'}) → {out}")
    return 0
```

- [ ] **Step 6: Smoke-check plumbing without the heavy download** — `uv run python -m tsumugi_export export --help` shows the flags; `uv run pytest tests/ -v` still green. (The real export run is Task 10.)

- [ ] **Step 7: Commit**

```bash
git add tools/export
git commit -m "feat(export): custom OnnxConfig with attention outputs and export command"
```

---

### Task 3: quantize command

**Files:**
- Modify: `tools/export/src/tsumugi_export/quantize.py` (replace stub)

**Interfaces:**
- Consumes: `out/model/onnx/model.onnx` from Task 2
- Produces: `onnx/model_q4.onnx` and `onnx/model_fp16.onnx` beside it; `quantize.run(args) -> int`.

- [ ] **Step 1: Implement** (`tools/export/src/tsumugi_export/quantize.py`)

```python
"""Produce the q4 (primary) and fp16 (insurance) variants.

API-shape caveat: MatMul4BitsQuantizer import path moved across onnxruntime
versions (onnxruntime.quantization.matmul_4bits_quantizer at the time of
writing). q4 here matches what transformers.js dtype:'q4' loads
(model_q4.onnx, MatMulNBits weights). Whether q4 preserves attention-output
correctness is decided by `validate`, not assumed here."""
from pathlib import Path

import onnx


def run(args) -> int:
    onnx_dir = Path(args.model_dir) / "onnx"
    src = onnx_dir / "model.onnx"
    if not src.exists():
        print(f"missing {src} — run export first")
        return 1

    # fp16 variant
    from onnxconverter_common import float16
    model = onnx.load(str(src))
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16_model, str(onnx_dir / "model_fp16.onnx"))
    print("wrote model_fp16.onnx")

    # q4 variant (weights-only MatMulNBits)
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
    model = onnx.load(str(src))
    quant = MatMul4BitsQuantizer(model, block_size=32, is_symmetric=True)
    quant.process()
    onnx.save_model(quant.model.model, str(onnx_dir / "model_q4.onnx"))
    print("wrote model_q4.onnx")
    return 0
```

- [ ] **Step 2: Verify plumbing** — `uv run python -m tsumugi_export quantize` against a missing dir prints the "run export first" message and exits 1. `uv run pytest tests/ -v` green. (Real run: Task 10.)

- [ ] **Step 3: Commit**

```bash
git add tools/export
git commit -m "feat(export): q4 and fp16 quantize command"
```

---

### Task 4: validate command (+ validation-report.json)

**Files:**
- Modify: `tools/export/src/tsumugi_export/validate.py` (replace stub)

**Interfaces:**
- Consumes: exported dirs from Tasks 2–3; stock ONNX from the HF hub (`HuggingFaceTB/SmolLM2-135M-Instruct`, `onnx/model.onnx`)
- Produces: console verdict table; `<model-dir>/validation-report.json` of shape `{"checks": {name: {"passed": bool, "detail": str}}, "artifacts": {filename: sha256}}`. Task 5's publish reads exactly this shape.

- [ ] **Step 1: Implement** (`tools/export/src/tsumugi_export/validate.py`)

```python
"""Validation gate. Runs the exported graphs in onnxruntime and checks:
1. logits parity with the stock export (greedy continuation + last logits)
2. attention rows row-stochastic and causal
3. A≡B: cached incremental rows equal no-cache full-matrix rows (only when
   both variants are present in out/; otherwise reported as skipped)
4. cache integrity: present.* outputs exist and multi-step cached logits
   match single-shot full-context logits
Writes validation-report.json with artifact hashes; publish refuses to
upload without a passing report."""
import hashlib
import json
from pathlib import Path

import numpy as np

PROMPTS = [
    "The cat sat on the mat because it was tired",
    "one two three one two three one",     # repeats → induction rows exist
]
TOL = 2e-2  # fp32-vs-quantized logit tolerance; fp32-vs-fp32 uses 1e-4


def _session(path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _tokenize(model_dir: Path, prompt: str) -> np.ndarray:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return np.array([tok.encode(prompt)], dtype=np.int64)


def _feeds(sess, input_ids: np.ndarray) -> dict:
    names = {i.name for i in sess.get_inputs()}
    feeds = {"input_ids": input_ids,
             "attention_mask": np.ones_like(input_ids)}
    if "position_ids" in names:
        feeds["position_ids"] = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    # empty past for cached graphs
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values"):
            shape = [d if isinstance(d, int) else (1 if "batch" in str(d) else 0)
                     for d in i.shape]
            feeds[i.name] = np.zeros(shape, dtype=np.float32)
    return feeds


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args) -> int:
    from huggingface_hub import hf_hub_download
    from tsumugi_export import STOCK_MODEL_ID

    model_dir = Path(args.model_dir)
    checks: dict[str, dict] = {}

    stock_path = Path(hf_hub_download(STOCK_MODEL_ID, "onnx/model.onnx"))
    stock = _session(stock_path)

    for variant in ["model.onnx", "model_q4.onnx", "model_fp16.onnx"]:
        path = model_dir / "onnx" / variant
        if not path.exists():
            continue
        sess = _session(path)
        out_names = [o.name for o in sess.get_outputs()]
        attn_names = sorted(n for n in out_names if n.startswith("attentions."))

        for prompt in PROMPTS:
            ids = _tokenize(model_dir, prompt)
            ours = dict(zip(out_names, sess.run(None, _feeds(sess, ids))))
            theirs_names = [o.name for o in stock.get_outputs()]
            theirs = dict(zip(theirs_names, stock.run(None, _feeds(stock, ids))))

            tol = 1e-4 if variant == "model.onnx" else TOL
            diff = float(np.max(np.abs(ours["logits"][0, -1] - theirs["logits"][0, -1])))
            key = f"{variant}:logits-parity:{prompt[:12]}"
            checks[key] = {"passed": diff < tol, "detail": f"max|Δlogit|={diff:.2e} (tol {tol})"}

            ok, detail = True, "all rows row-stochastic and causal"
            for name in attn_names:
                a = ours[name][0]  # [heads, q, kv]
                sums = a.sum(axis=-1)
                if not np.allclose(sums, 1.0, atol=1e-3):
                    ok, detail = False, f"{name}: row sums off (max err {np.max(np.abs(sums-1)):.2e})"
                    break
                q = a.shape[1]
                upper = np.triu(np.ones((q, a.shape[2])), k=1)[None, ...]
                if float(np.max(a * upper[..., :a.shape[2]])) > 1e-5:
                    ok, detail = False, f"{name}: mass on future positions"
                    break
            checks[f"{variant}:attn-wellformed:{prompt[:12]}"] = {"passed": ok, "detail": detail}

        checks[f"{variant}:has-attentions"] = {
            "passed": len(attn_names) > 0, "detail": f"{len(attn_names)} attention outputs"}
        checks[f"{variant}:has-cache"] = {
            "passed": any(n.startswith("present") for n in out_names),
            "detail": "present.* outputs present"}

    # A≡B equivalence when the operator exported both variants
    nc = model_dir.parent / "model-nocache" / "onnx" / "model.onnx"
    if nc.exists():
        # run cached graph step by step and compare each new row with the
        # no-cache full matrix — implementation mirrors the checks above and
        # reports max row deviation per layer
        checks["a-equiv-b"] = _check_a_equiv_b(model_dir, nc)  # defined below
    else:
        checks["a-equiv-b"] = {"passed": True, "detail": "SKIPPED (no no-cache export present)"}

    passed = all(c["passed"] for c in checks.values())
    report = {
        "checks": checks,
        "artifacts": {p.name: _sha256(p) for p in (model_dir / "onnx").glob("*.onnx")},
    }
    (model_dir / "validation-report.json").write_text(json.dumps(report, indent=2))

    width = max(len(k) for k in checks)
    for k, c in sorted(checks.items()):
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {k.ljust(width)}  {c['detail']}")
    print("VERDICT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1
```

Also implement `_check_a_equiv_b(model_dir, nocache_model)` in the same file: tokenize `PROMPTS[0]`, run the no-cache graph once for the full sequence, then drive the cached graph token by token feeding `present.*` back as `past_key_values.*`, and compare each step's attention row (last query row per layer) against the corresponding row of the no-cache matrices with `atol=1e-3`; return `{"passed": bool, "detail": "max row deviation ..."}`.

- [ ] **Step 2: Verify plumbing** — `uv run python -m tsumugi_export validate` against a missing dir exits nonzero with a clear message (guard at the top of `run`: if `model_dir/"onnx"` has no `.onnx` files, print and return 1 — add this guard). `uv run pytest tests/ -v` green.

- [ ] **Step 3: Commit**

```bash
git add tools/export
git commit -m "feat(export): validation gate with report file"
```

---

### Task 5: publish command

**Files:**
- Modify: `tools/export/src/tsumugi_export/publish.py` (replace stub)

**Interfaces:**
- Consumes: model dir + `validation-report.json` (Task 4's exact shape); `ATTN_REPO_ID`
- Produces: HF repo upload; refuses without a passing report or on artifact-hash mismatch.

- [ ] **Step 1: Implement** (`tools/export/src/tsumugi_export/publish.py`)

```python
"""Assemble and upload the model repo. Requires HF auth (env HF_TOKEN or
`hf auth login`). Refuses to publish artifacts without a passing
validation-report.json whose hashes match the files on disk."""
import hashlib
import json
from pathlib import Path

from tsumugi_export import ATTN_REPO_ID, STOCK_MODEL_ID

MODEL_CARD = """---
license: apache-2.0
base_model: {stock}
library_name: transformers.js
pipeline_tag: text-generation
---

# SmolLM2-135M-Instruct with attention outputs (ONNX)

Derived from [{stock}](https://huggingface.co/{stock}): same weights, ONNX
graph re-exported with per-layer attention-probability outputs
(`attentions.0` … `attentions.{last}`) alongside `logits` and the KV cache,
for in-browser attention visualization in
[Tsumugi](https://github.com/saigyo/tsumugi).

Validation (see `validation-report.json`): logits parity with the stock
export, row-stochastic causal attention rows, cached-vs-full-matrix
equivalence.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args) -> int:
    from huggingface_hub import HfApi

    model_dir = Path(args.model_dir)
    report_path = model_dir / "validation-report.json"
    if not report_path.exists():
        print("no validation-report.json — run validate first")
        return 1
    report = json.loads(report_path.read_text())
    if not all(c["passed"] for c in report["checks"].values()):
        print("validation report contains failures — refusing to publish")
        return 1
    for name, expected in report["artifacts"].items():
        actual = _sha256(model_dir / "onnx" / name)
        if actual != expected:
            print(f"{name} changed since validation (hash mismatch) — re-run validate")
            return 1

    from transformers import AutoConfig
    n_layers = AutoConfig.from_pretrained(str(model_dir)).num_hidden_layers
    (model_dir / "README.md").write_text(
        MODEL_CARD.format(stock=STOCK_MODEL_ID, last=n_layers - 1))

    repo_id = args.repo_id or ATTN_REPO_ID
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id)
    print(f"published → https://huggingface.co/{repo_id}")
    return 0
```

- [ ] **Step 2: Verify plumbing** — `uv run python -m tsumugi_export publish` without a report prints the refusal and exits 1; `uv run pytest tests/ -v` green.

- [ ] **Step 3: Commit**

```bash
git add tools/export
git commit -m "feat(export): publish command gated on validation report"
```

---

### Task 6: attention row accumulation (TS, pure)

**Files:**
- Create: `src/engine/transformers/attentionAccum.ts`
- Test: `src/engine/transformers/attentionAccum.test.ts`

**Interfaces:**
- Produces:

```ts
export interface AttnAccumulator { layers: number; heads: number; rows: number[][][][] }
// rows[layer][head] = ragged causal matrix (array of rows, row i has length i+1)
export function createAccumulator(layers: number, heads: number): AttnAccumulator
// dims = [batch, heads, qLen, kvLen]; data = flat Float32Array from the tensor
export function addAttentionOutput(acc: AttnAccumulator, layer: number,
  dims: number[], data: Float32Array | number[]): void
// prefill (qLen === kvLen) appends qLen rows; a step (qLen === 1) appends one
// row of length kvLen. Rows are truncated to causal length (row index + 1).
```

- [ ] **Step 1: Failing tests** (`src/engine/transformers/attentionAccum.test.ts`)

```ts
import { expect, test } from 'vitest'
import { addAttentionOutput, createAccumulator } from './attentionAccum'

test('prefill appends the full causal triangle', () => {
  const acc = createAccumulator(1, 2)
  // batch 1, 2 heads, 3 query rows, 3 kv — rows may carry zeros above the diagonal
  const head0 = [1, 0, 0, /**/ 0.5, 0.5, 0, /**/ 0.2, 0.3, 0.5]
  const head1 = head0.map((v) => v)
  addAttentionOutput(acc, 0, [1, 2, 3, 3], Float32Array.from([...head0, ...head1]))
  expect(acc.rows[0][0]).toHaveLength(3)
  expect(acc.rows[0][0][0]).toEqual([1])            // causal truncation
  expect(acc.rows[0][0][2]).toEqual([0.2, 0.3, 0.5])
})

test('a decode step appends one row per head', () => {
  const acc = createAccumulator(1, 1)
  addAttentionOutput(acc, 0, [1, 1, 2, 2], Float32Array.from([1, 0, 0.4, 0.6]))
  addAttentionOutput(acc, 0, [1, 1, 1, 3], Float32Array.from([0.1, 0.2, 0.7]))
  expect(acc.rows[0][0]).toHaveLength(3)
  expect(acc.rows[0][0][2]).toEqual([0.1, 0.2, 0.7])
})

test('no-cache full matrices are consumed by keeping only new rows', () => {
  const acc = createAccumulator(1, 1)
  addAttentionOutput(acc, 0, [1, 1, 2, 2], Float32Array.from([1, 0, 0.4, 0.6]))
  // Approach B: next step re-sends the FULL 3×3 matrix; only row 2 is new
  addAttentionOutput(acc, 0, [1, 1, 3, 3], Float32Array.from([1, 0, 0, 0.4, 0.6, 0, 0.1, 0.2, 0.7]))
  expect(acc.rows[0][0]).toHaveLength(3)
  expect(acc.rows[0][0][2]).toEqual([0.1, 0.2, 0.7])
})
```

- [ ] **Step 2: Verify red** — `npx vitest run src/engine/transformers/attentionAccum.test.ts` → FAIL.

- [ ] **Step 3: Implement** (`src/engine/transformers/attentionAccum.ts`)

```ts
export interface AttnAccumulator {
  layers: number
  heads: number
  rows: number[][][][]
}

export function createAccumulator(layers: number, heads: number): AttnAccumulator {
  return {
    layers, heads,
    rows: Array.from({ length: layers }, () => Array.from({ length: heads }, () => [])),
  }
}

export function addAttentionOutput(
  acc: AttnAccumulator, layer: number, dims: number[], data: Float32Array | number[],
): void {
  const [, heads, qLen, kvLen] = dims
  for (let h = 0; h < heads; h++) {
    const existing = acc.rows[layer][h].length
    for (let q = 0; q < qLen; q++) {
      const rowIndex = qLen === 1 ? kvLen - 1 : q + (kvLen - qLen)
      if (rowIndex < existing) continue  // Approach B resends old rows; keep only new
      const offset = (h * qLen + q) * kvLen
      const row: number[] = []
      for (let c = 0; c <= rowIndex && c < kvLen; c++) row.push(data[offset + c])
      acc.rows[layer][h].push(row)
    }
  }
}
```

- [ ] **Step 4: Verify green**, `npx tsc --noEmit -p tsconfig.app.json` clean.

- [ ] **Step 5: Commit**

```bash
git add src/engine/transformers/attentionAccum.ts src/engine/transformers/attentionAccum.test.ts
git commit -m "feat: attention row accumulator for real-mode matrices"
```

---

### Task 7: head statistics + showcase selection (TS, pure)

**Files:**
- Create: `src/engine/transformers/attentionStats.ts`
- Modify: `src/trace/types.ts` (AttentionHead gains `score?: number`)
- Test: `src/engine/transformers/attentionStats.test.ts`

**Interfaces:**
- Consumes: `AttnAccumulator` (Task 6), `TokenInfo`, `AttentionHead`, `AttentionLabel` from `src/trace/types.ts`
- Produces:

```ts
export interface HeadStats { layer: number; head: number;
  prevTokenScore: number; sinkScore: number; inductionScore: number | null }
export function headStats(acc: AttnAccumulator, tokens: TokenInfo[]): HeadStats[]
export function selectShowcaseHeads(stats: HeadStats[], acc: AttnAccumulator,
  threshold?: number): AttentionHead[]   // default threshold 0.3; ≤3 heads with
                                         // labels previous-token/attention-sink/induction,
                                         // matrix copied from acc, score = the stat
```

- [ ] **Step 1: Add `score?: number` to `AttentionHead`** in `src/trace/types.ts` (one line under `label`). Run `npx vitest run` — all green (additive).

- [ ] **Step 2: Failing tests** (`src/engine/transformers/attentionStats.test.ts`)

```ts
import { expect, test } from 'vitest'
import type { TokenInfo } from '../../trace/types'
import { createAccumulator } from './attentionAccum'
import { headStats, selectShowcaseHeads } from './attentionStats'

const toks = (...t: string[]): TokenInfo[] => t.map((text, id) => ({ id, text }))

function fill(rowsFor: (i: number) => number[], n: number) {
  return Array.from({ length: n }, (_, i) => rowsFor(i))
}
const diagRow = (i: number) => i === 0 ? [1] : [...Array(i - 1).fill(0), 1, 0]
const sinkRow = (i: number) => i === 0 ? [1] : [1, ...Array(i).fill(0)]

test('scores identify diagonal and sink heads', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = fill(diagRow, 5)
  acc.rows[1][0] = fill(sinkRow, 5)
  const stats = headStats(acc, toks('a', ' b', ' c', ' d', ' e'))
  expect(stats[0].prevTokenScore).toBe(1)
  expect(stats[1].sinkScore).toBe(1)
  expect(stats[0].inductionScore).toBeNull()   // no repeated tokens
})

test('induction score measured only on repeat rows', () => {
  const acc = createAccumulator(1, 1)
  // tokens: a b a — row 2 repeats row 0's token; induction target = col 1
  acc.rows[0][0] = [[1], [0.5, 0.5], [0, 1, 0]]
  const stats = headStats(acc, toks('a', ' b', 'a'))
  expect(stats[0].inductionScore).toBe(1)
})

test('selectShowcaseHeads picks top head per label above threshold', () => {
  const acc = createAccumulator(2, 1)
  acc.rows[0][0] = fill(diagRow, 5)
  acc.rows[1][0] = fill(sinkRow, 5)
  const heads = selectShowcaseHeads(headStats(acc, toks('a', ' b', ' c', ' d', ' e')), acc)
  expect(heads.map((h) => h.label).sort()).toEqual(['attention-sink', 'previous-token'])
  const prev = heads.find((h) => h.label === 'previous-token')!
  expect(prev.layer).toBe(0)
  expect(prev.score).toBe(1)
  expect(prev.matrix).toHaveLength(5)
})

test('heads below threshold are not selected', () => {
  const acc = createAccumulator(1, 1)
  acc.rows[0][0] = fill((i) => i === 0 ? [1] : Array(i + 1).fill(1 / (i + 1)), 5)  // uniform
  const heads = selectShowcaseHeads(headStats(acc, toks('a', ' b', ' c', ' d', ' e')), acc)
  expect(heads).toHaveLength(0)
})
```

- [ ] **Step 3: Verify red**, then implement (`src/engine/transformers/attentionStats.ts`)

```ts
import type { AttentionHead, AttentionLabel, TokenInfo } from '../../trace/types'
import type { AttnAccumulator } from './attentionAccum'

export interface HeadStats {
  layer: number
  head: number
  prevTokenScore: number
  sinkScore: number
  inductionScore: number | null
}

export function headStats(acc: AttnAccumulator, tokens: TokenInfo[]): HeadStats[] {
  // induction targets: for row i whose token appeared at j < i, target j+1
  const targets: Array<number | null> = tokens.map((t, i) => {
    for (let j = i - 1; j >= 0; j--) {
      if (tokens[j].text.trim() === t.text.trim() && j + 1 <= i) return j + 1
    }
    return null
  })
  const out: HeadStats[] = []
  for (let l = 0; l < acc.layers; l++) {
    for (let h = 0; h < acc.heads; h++) {
      const m = acc.rows[l][h]
      let prev = 0, sink = 0, n = 0, ind = 0, indN = 0
      for (let i = 1; i < m.length; i++) {
        prev += m[i][i - 1]
        sink += m[i][0]
        n++
        const t = targets[i]
        if (t !== null && t < m[i].length) { ind += m[i][t]; indN++ }
      }
      out.push({
        layer: l, head: h,
        prevTokenScore: n ? prev / n : 0,
        sinkScore: n ? sink / n : 0,
        inductionScore: indN ? ind / indN : null,
      })
    }
  }
  return out
}

const round = (x: number) => Math.round(x * 100) / 100

export function selectShowcaseHeads(
  stats: HeadStats[], acc: AttnAccumulator, threshold = 0.3,
): AttentionHead[] {
  const pick = (label: AttentionLabel, score: (s: HeadStats) => number | null): AttentionHead | null => {
    let best: HeadStats | null = null
    let bestScore = threshold
    for (const s of stats) {
      const v = score(s)
      if (v !== null && v > bestScore) { best = s; bestScore = v }
    }
    if (!best) return null
    return { layer: best.layer, head: best.head, label,
      score: round(bestScore), matrix: acc.rows[best.layer][best.head] }
  }
  return [
    pick('previous-token', (s) => s.prevTokenScore),
    pick('attention-sink', (s) => s.sinkScore),
    pick('induction', (s) => s.inductionScore),
  ].filter((h): h is AttentionHead => h !== null)
}
```

- [ ] **Step 4: Verify green**, tsc clean.

- [ ] **Step 5: Commit**

```bash
git add src/engine/transformers/attentionStats.ts src/engine/transformers/attentionStats.test.ts src/trace/types.ts
git commit -m "feat: head-role detection from accumulated attention"
```

---

### Task 8: protocol + client `attentions` flag

**Files:**
- Modify: `src/engine/transformers/protocol.ts` (`ready` gains `attentions: boolean`), `src/engine/transformers/TransformersEngine.ts` (records it), `src/engine/tokenizer.ts` (add `ATTN_MODEL_ID`)
- Test: `src/engine/transformers/TransformersEngine.test.ts` (extend)

**Interfaces:**
- Produces: `export const ATTN_MODEL_ID = 'saigyo-hoshi/smollm2-135m-attn-onnx'` (tokenizer.ts); protocol `{ type: 'ready'; device: 'webgpu' | 'wasm'; attentions: boolean }`; `TransformersEngine.attentions: boolean | null` set on ready. Task 10's App wiring reads `engine.attentions`.

- [ ] **Step 1: Failing test** (extend the FakeWorker suite)

```ts
test('ready records the attentions capability', async () => {
  const { worker, engine } = make()
  const p = engine.prepare()
  worker.respond({ type: 'ready', device: 'webgpu', attentions: true })
  await p
  expect(engine.attentions).toBe(true)
})
```

Also update the existing 'prepare resolves on ready' test's `respond` call to include `attentions: false` (the field is required in the protocol type).

- [ ] **Step 2: Verify red** (type error + failing test), then implement: add the field to `WorkerResponse`'s ready variant; in `TransformersEngine` add `attentions: boolean | null = null` and set it in the prepare listener beside `device`; add the `ATTN_MODEL_ID` export to tokenizer.ts.

- [ ] **Step 3: Verify green** — `npx vitest run src/engine` and tsc clean.

- [ ] **Step 4: Commit**

```bash
git add src/engine
git commit -m "feat: attentions capability in worker protocol and client"
```

---

### Task 9: worker integration

**Files:**
- Modify: `src/engine/transformers/worker.ts`

**Interfaces:**
- Consumes: `createAccumulator`/`addAttentionOutput` (Task 6), `headStats`/`selectShowcaseHeads` (Task 7), `ATTN_MODEL_ID` + `MODEL_ID` (Task 8), existing math/emit helpers.
- Produces: real-mode `attention` trace events (≤3 detected heads, emitted between the last `layer` event and `logits` — the slot `validateTrace` accepts); `ready` message carries the real `attentions` flag.

No unit test (worker is CI-untestable by design; the logic with judgment lives in Tasks 6–7). Changes:

- [ ] **Step 1: Load chain.** In `prepare(modelId)`: try `AutoModelForCausalLM.from_pretrained(ATTN_MODEL_ID, { dtype: 'q4', device, progress_callback })`; on throw retry `{ dtype: 'fp16' }`; on throw fall back to the stock `modelId` exactly as today. Track `let hasAttentions = <loaded attn model>`; post `{ type: 'ready', device, attentions: hasAttentions }`. (Keep the existing WebGPU→WASM fallback wrapping each attempt.)

- [ ] **Step 2: Accumulate + emit.** In `run()`: after `run-start`, when `hasAttentions`, `const acc = createAccumulator(numLayers, numHeads)` (`numHeads = model.config.num_attention_heads ?? 9`). After each forward pass, guarded by try/catch that flips a `let attnBroken = false` flag (never fail generation):

```ts
if (hasAttentions && !attnBroken) {
  try {
    for (let l = 0; l < numLayers; l++) {
      const t = out[`attentions.${l}`]
      if (!t) throw new Error(`missing attentions.${l}`)
      addAttentionOutput(acc, l, t.dims as number[], t.data as Float32Array)
    }
  } catch { attnBroken = true }
}
```

- [ ] **Step 3: Emit the event** after the schematic `layer` events and before `logits` each cycle:

```ts
if (hasAttentions && !attnBroken) {
  const heads = selectShowcaseHeads(headStats(acc, allIds.map(tokenInfo)), acc)
  if (heads.length > 0) emit({ type: 'attention', cycle, heads })
}
```

(Compute `allIds.map(tokenInfo)` once per cycle and reuse. Note: decode `tokenInfo` per id is already memo-cheap at these lengths.)

- [ ] **Step 4: Verify** — `npx vitest run` all green (worker untested but imported types must compile), `npx tsc --noEmit -p tsconfig.app.json` clean, `npm run build` succeeds (worker chunk builds).

- [ ] **Step 5: Commit**

```bash
git add src/engine/transformers/worker.ts
git commit -m "feat: worker loads attention model, accumulates and emits detected heads"
```

---

### Task 10: M1 UI — scores, notes, status chip

**Files:**
- Modify: `src/viz/AttentionHeatmap.tsx`, `src/viz/details/LayersDetail.tsx`, `src/app/ModelStatus.tsx`, `src/App.tsx`
- Test: `src/viz/AttentionHeatmap.test.tsx`, `src/viz/DetailPanel.test.tsx`, `src/app/ModelStatus.test.tsx` (extend each)

**Interfaces:**
- Consumes: `AttentionHead.score?` (Task 7), `TransformersEngine.attentions` (Task 8)
- Produces: UI behavior below; `ModelStatus` gains prop `attentions?: boolean`.

- [ ] **Step 1: Failing tests**

`AttentionHeatmap.test.tsx` additions:

```tsx
test('scored heads show the score and the measured note', () => {
  const scored = attn.heads.map((h) => ({ ...h, score: 0.87 }))
  render(<AttentionHeatmap heads={scored} tokens={tokens} />)
  expect(screen.getAllByTestId('head-chip')[0]).toHaveTextContent('0.87')
  expect(screen.getByTestId('attn-note')).toHaveTextContent(/measured on this prompt/i)
})

test('unscored heads keep the illustrative note', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.getByTestId('attn-note')).toHaveTextContent(/illustrative/i)
})
```

(Give the note element `data-testid="attn-note"` — it currently has only a class.)

`ModelStatus.test.tsx` addition:

```tsx
test('device chip marks attention capability', () => {
  render(<ModelStatus progress={null} device="webgpu" error={null} attentions onFallback={() => {}} />)
  expect(screen.getByTestId('device-chip')).toHaveTextContent('webgpu · attn')
})
```

`DetailPanel.test.tsx` addition:

```tsx
test('real mode drops the schematic tag when attention data exists', () => {
  render(<DetailPanel events={trace} cursor={6} mode="real" />)  // cursor on attention event
  expect(screen.getByTestId('detail-layers')).not.toHaveTextContent(/schematic/i)
})
```

- [ ] **Step 2: Verify red, implement:**
  - `AttentionHeatmap`: chip renders `{h.score != null && <span className="head-score">· {h.score}</span>}`; note becomes `head.score != null ? 'Measured on this prompt — head roles detected from the attention weights, not labeled by the model.' : 'Illustrative pattern (simulated) — real attention weights are not exposed by the browser model.'`; add `data-testid="attn-note"`.
  - `LayersDetail`: header shows the `(schematic — real internals not exposed)` em-tag only when `mode === 'real' && !attention`.
  - `ModelStatus`: new optional prop `attentions?: boolean`; chip text `{device}{attentions ? ' · attn' : ''}`.
  - `App.tsx`: track `const [attn, setAttn] = useState(false)`; after `prepare()` resolves set from `engine.attentions ?? false`; pass `attentions={mode === 'real' && attn}` to ModelStatus.

- [ ] **Step 3: Verify** — full `npx vitest run` green, tsc clean.

- [ ] **Step 4: Commit**

```bash
git add src/viz src/app src/App.tsx
git commit -m "feat: detected-head scores and attention status in the UI"
```

---

### Task 11: OPERATOR RUNBOOK — run the export pipeline (network + HF token gated)

**Files:** none created by the plan; artifacts land in `tools/export/out/` (git-ignored) and on the HF Hub.

This task is executed by a human/controller with a fast network and the user's HF token — never by CI, never by a sandboxed subagent. Document completion in the PR/commit message.

- [ ] **Step 1:** `cd tools/export && uv sync`
- [ ] **Step 2:** `uv run python -m tsumugi_export export --out out/model` (downloads ~270 MB torch weights; several minutes CPU export)
- [ ] **Step 3:** `uv run python -m tsumugi_export export --out out/model-nocache --no-cache` (enables the A≡B check)
- [ ] **Step 4:** `uv run python -m tsumugi_export quantize --model-dir out/model`
- [ ] **Step 5:** `uv run python -m tsumugi_export validate --model-dir out/model`
  Expected: verdict table, `VERDICT: PASS`, `out/model/validation-report.json` written. If `a-equiv-b` FAILS: re-run everything with the `--no-cache` export as the primary (Approach B fallback per spec) and note it in the model card.
  If q4 checks FAIL but fp32/fp16 pass: remove `model_q4.onnx`, publish fp16 as primary, note it.
- [ ] **Step 6:** `HF_TOKEN=… uv run python -m tsumugi_export publish --model-dir out/model` (user's token; creates `saigyo-hoshi/smollm2-135m-attn-onnx`)
- [ ] **Step 7:** Record results (which variant published, validation summary) in the commit that flips the app to the new model (Task 12).

---

### Task 12: In-app verification + docs sync

**Files:**
- Modify: `README.md` (real-mode section mentions detected attention heads), `docs/superpowers/specs/2026-08-26-llm-pipeline-visualizer-design.md` (the "real mode emits no attention events" sentence in the simulated-attention section gets a pointer to the M1 spec)

- [ ] **Step 1: Manual browser verification** (per the protocol in `tools/export/README.md`): `npm run dev`, enable Real model on a fast network; confirm download progress for the attn repo, chip shows `webgpu · attn` (or `wasm · attn`), generation streams, the Layers panel shows real heatmaps with scored chips, rows sum to ~100% via hover readout, detected labels are plausible (diagonal-looking → previous-token, first-column → attention-sink; induction appears on the repeated-pattern example), and — by temporarily blocking the attn repo (devtools offline after cache-clear or an invalid `ATTN_MODEL_ID` build) — the stock fallback still produces today's schematic behavior with plain device chip.
- [ ] **Step 2: Update the two docs** as listed; run `npx vitest run` + e2e once more.
- [ ] **Step 3: Commit**

```bash
git add README.md docs
git commit -m "docs: real-mode attention detection notes; M1 verification recorded"
```

---

## Post-plan checks

- Full suite: `npx vitest run && npx tsc --noEmit -p tsconfig.app.json && npx playwright test` (all sim-mode, unaffected by network).
- Python: `uv run pytest tests/ -v` inside `tools/export/`.
- The heavy pipeline (Task 11) and browser verification (Task 12 Step 1) require operator presence; everything else is sandbox-safe.
