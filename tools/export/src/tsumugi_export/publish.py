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

**Variants:** `onnx/model_fp16.onnx` is the primary artifact. A q4
(MatMulNBits) variant was produced but failed greedy-parity validation
against the stock export (structurally divergent output, not ordinary
quantization noise) and is deliberately not published; consumers
requesting `dtype: 'q4'` should expect a 404 and fall back to fp16.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_blocking(check: dict, allow_skipped_equivalence: bool = False) -> bool:
    """Whether a single validation-report check should block publish.

    A passed check never blocks. A skipped check (currently only
    a-equiv-b when no no-cache export was present) blocks UNLESS the
    operator explicitly passed --allow-skipped-equivalence — an untested
    Approach-A hypothesis must not silently pass the gate. Any other
    failure always blocks, regardless of the flag."""
    if check.get("passed"):
        return False
    if check.get("skipped") and allow_skipped_equivalence:
        return False
    return True


def run(args) -> int:
    from huggingface_hub import HfApi

    model_dir = Path(args.model_dir)
    report_path = model_dir / "validation-report.json"
    if not report_path.exists():
        print("no validation-report.json — run validate first")
        return 1
    report = json.loads(report_path.read_text())
    allow_skipped = getattr(args, "allow_skipped_equivalence", False)
    blocking = {name: c for name, c in report["checks"].items()
                if is_blocking(c, allow_skipped)}
    if blocking:
        print("validation report contains failures — refusing to publish")
        for name, c in sorted(blocking.items()):
            reason = "SKIPPED" if c.get("skipped") else "FAILED"
            print(f"  {reason}  {name}: {c['detail']}")
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
    # The fp32 onnx/model.onnx (~½ GB) is a validation baseline, not part of
    # the spec's shipped repo layout (model_q4.onnx is primary, model_fp16.onnx
    # is insurance) — its hash still lives in validation-report.json.
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id,
                       ignore_patterns=["onnx/model.onnx"])
    print(f"published → https://huggingface.co/{repo_id}")
    return 0
