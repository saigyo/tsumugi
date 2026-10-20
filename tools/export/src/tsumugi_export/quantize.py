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
"""
from pathlib import Path

from tsumugi_export.postprocess import TOPK_GATHER_NODE

# The tied lm_head must stay fp32 (see the module docstring); the readout
# gather must too, so the candidate rows the app shows are exactly the rows
# the lm_head multiplied — their products then sum to the logit.
NODES_TO_EXCLUDE = ["/lm_head/MatMul", TOPK_GATHER_NODE]


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
