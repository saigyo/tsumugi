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
