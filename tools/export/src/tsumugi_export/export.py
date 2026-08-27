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
    custom = AttnLlamaOnnxConfig(
        config, task="text-generation", use_past=use_past, use_past_in_inputs=use_past)

    main_export(
        STOCK_MODEL_ID,
        output=str(onnx_dir),
        task="text-generation-with-past" if use_past else "text-generation",
        custom_onnx_configs={"model": custom},
        model_kwargs={"output_attentions": True, "attn_implementation": "eager"},
        do_validation=False,  # our validate command is the real gate
    )
    # optimum writes model.onnx into onnx_dir; normalize the name if needed.
    # Guard on model.onnx being absent (not on "exactly one .onnx file") so a
    # directory that already has model.onnx plus sibling files (e.g. a
    # re-run, or external/*.onnx_data) is left alone rather than misfiring.
    if not (onnx_dir / "model.onnx").exists():
        exported = list(onnx_dir.glob("*.onnx"))
        if len(exported) == 1:
            exported[0].rename(onnx_dir / "model.onnx")

    stock = Path(snapshot_download(STOCK_MODEL_ID, allow_patterns=TOKENIZER_FILES))
    for name in TOKENIZER_FILES:
        src = stock / name
        if src.exists():
            shutil.copy2(src, out / name)

    print(f"exported ({'with-past' if use_past else 'no-cache'}) → {out}")
    return 0
