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
