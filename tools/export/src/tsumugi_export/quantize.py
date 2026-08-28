"""Produce the quantized variant for publication.

Outcome of the operator-run validation campaign (2026-08-28), preserved here
so the choices aren't re-litigated blind:

- q8 dynamic int8 (this file's output, `model_quantized.onnx`, loaded by
  transformers.js `dtype: 'q8'`) with `/lm_head/MatMul` EXCLUDED is
  token-identical to the stock fp32 export over the 8-step greedy check —
  better than the official stock q4, which diverges at step 4.
- Excluding the lm_head matmul is the load-bearing detail: SmolLM2 ties the
  lm_head to the input embeddings, and quantizing it degraded every variant
  (q4 MatMulNBits, plain q8) far beyond normal quantization noise.
- q4 MatMulNBits with the same exclusion reaches parity with the official
  stock q4 (3-token greedy prefix, then benign divergence) but is strictly
  worse than q8 at a similar size — not published.
- fp16 via onnxruntime.transformers.float16 loads and validates on Python
  CPU but computes garbage on the WebGPU EP (and is rejected outright by
  ORT-node): a marginal graph — disqualified, not published.
"""
from pathlib import Path


def run(args) -> int:
    onnx_dir = Path(args.model_dir) / "onnx"
    src = onnx_dir / "model.onnx"
    if not src.exists():
        print(f"missing {src} — run export first")
        return 1

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(onnx_dir / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8,
                     nodes_to_exclude=["/lm_head/MatMul"])
    print("wrote model_quantized.onnx (q8, lm_head excluded)")
    return 0
