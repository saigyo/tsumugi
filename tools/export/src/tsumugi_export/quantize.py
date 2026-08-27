"""Produce the q4 (primary) and fp16 (insurance) variants.

API-shape caveat: the installed onnxruntime (1.29.0) renamed the weights-only
quantizer class and moved its module: `MatMul4BitsQuantizer` in
`onnxruntime.quantization.matmul_4bits_quantizer` (as assumed by the original
brief) is now `MatMulNBitsQuantizer` in
`onnxruntime.quantization.matmul_nbits_quantizer`, taking an explicit `bits`
argument (we pass bits=4 to keep the q4 semantics explicit). That module also
now imports `onnx_ir` at import time, which is not pulled in transitively by
onnxruntime — added as an explicit direct dependency. q4 here matches what
transformers.js dtype:'q4' loads (model_q4.onnx, MatMulNBits weights).
Whether q4 preserves attention-output correctness is decided by `validate`,
not assumed here."""
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
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    model = onnx.load(str(src))
    quant = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=True)
    quant.process()
    onnx.save_model(quant.model.model, str(onnx_dir / "model_q4.onnx"))
    print("wrote model_q4.onnx")
    return 0
