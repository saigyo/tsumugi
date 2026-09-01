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
