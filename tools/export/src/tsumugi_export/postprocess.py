"""Graph post-processing after the optimum export.

The export declares hidden_states.{0..N} as outputs only because torch.onnx
names forward outputs positionally (see onnx_config.py). Here we keep two of
them under artifact-contract names — hidden_states.0 as `inputs_embeds` (the
embedding lookup) and hidden_states.N as `last_hidden_state` (the residual
stream after the final RMSNorm, the tensor lm_head multiplies) — and drop the
rest. We also add a readout branch: TopK over the last position's logits and
a Gather of those candidates' rows of E, as `topk_ids` / `topk_embeds`. The
tensors already in the graph stay; only the output list and a handful of tiny
nodes change, so this costs nothing measurable at inference time."""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from tsumugi_export import TOPK

EMBEDS_OUTPUT = "inputs_embeds"
SOURCE_OUTPUT = "hidden_states.0"
LAST_HIDDEN_OUTPUT = "last_hidden_state"
TOPK_IDS_OUTPUT = "topk_ids"
TOPK_EMBEDS_OUTPUT = "topk_embeds"
TOPK_GATHER_NODE = "tsumugi_topk_embeds_gather"
INT64_MAX = 2**63 - 1


def _producer(graph: onnx.GraphProto, name: str) -> onnx.NodeProto | None:
    return next((n for n in graph.node if name in n.output), None)


def find_embedding_table(graph: onnx.GraphProto) -> str:
    """The tensor name of E: the data input of the Gather that produces
    hidden_states.0, looking through Identity/Cast wrappers. Structural, not
    by name — optimum's initializer names are not part of our contract."""
    name = SOURCE_OUTPUT
    for _ in range(8):
        node = _producer(graph, name)
        if node is None:
            raise ValueError(f"no node produces {name} — expected a Gather behind {SOURCE_OUTPUT}")
        if node.op_type == "Gather":
            return node.input[0]
        if node.op_type in ("Identity", "Cast"):
            name = node.input[0]
            continue
        raise ValueError(f"{SOURCE_OUTPUT} is produced by {node.op_type}, expected Gather")
    raise ValueError(f"gave up tracing {SOURCE_OUTPUT} back to a Gather")


def expose_readout(model: onnx.ModelProto, k: int = TOPK) -> onnx.ModelProto:
    graph = model.graph
    outputs = list(graph.output)
    hidden = [o for o in outputs if o.name.startswith("hidden_states.")]
    n = max((int(o.name.split(".")[1]) for o in hidden), default=-1)
    if n < 1:
        raise ValueError("graph has no hidden_states.1 or later — no final hidden state to expose")
    last = next(o for o in hidden if o.name == f"hidden_states.{n}")
    if not any(o.name == "logits" for o in outputs):
        raise ValueError("graph has no logits output")
    dims = last.type.tensor_type.shape.dim[2].dim_value
    if dims <= 0:
        raise ValueError(f"hidden_states.{n} has no static last dimension")
    table = find_embedding_table(graph)

    graph.initializer.extend([
        helper.make_tensor("tsumugi_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_slice_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("tsumugi_squeeze_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("tsumugi_topk_k", TensorProto.INT64, [1], [k]),
    ])
    graph.node.extend([
        helper.make_node("Identity", [f"hidden_states.{n}"], [LAST_HIDDEN_OUTPUT],
                         name="tsumugi_last_hidden_identity"),
        helper.make_node("Slice", ["logits", "tsumugi_slice_starts", "tsumugi_slice_ends", "tsumugi_slice_axes"],
                         ["tsumugi_last_logits_3d"], name="tsumugi_last_logits_slice"),
        helper.make_node("Squeeze", ["tsumugi_last_logits_3d", "tsumugi_squeeze_axes"],
                         ["tsumugi_last_logits"], name="tsumugi_last_logits_squeeze"),
        # values output is named but consumed by nothing, which ONNX allows
        helper.make_node("TopK", ["tsumugi_last_logits", "tsumugi_topk_k"],
                         ["tsumugi_topk_values", TOPK_IDS_OUTPUT],
                         name="tsumugi_topk", axis=-1, largest=1, sorted=1),
        helper.make_node("Gather", [table, TOPK_IDS_OUTPUT], [TOPK_EMBEDS_OUTPUT],
                         name=TOPK_GATHER_NODE, axis=0),
    ])
    last_vi = onnx.ValueInfoProto()
    last_vi.CopyFrom(last)
    last_vi.name = LAST_HIDDEN_OUTPUT
    graph.output.extend([
        last_vi,
        helper.make_tensor_value_info(TOPK_IDS_OUTPUT, TensorProto.INT64, ["batch_size", k]),
        helper.make_tensor_value_info(TOPK_EMBEDS_OUTPUT, TensorProto.FLOAT, ["batch_size", k, dims]),
    ])
    return model


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


def postprocess_graph(model: onnx.ModelProto, k: int = TOPK) -> onnx.ModelProto:
    """Readout first (it reads hidden_states.N from the output list), then the
    inputs_embeds rename, which drops every hidden_states.* output."""
    return expose_inputs_embeds(expose_readout(model, k))


def postprocess_file(path: Path) -> None:
    """Rewrite `path` in place. onnx.load pulls any external data into the
    proto, so the saved file is self-contained; the 135M fp32 graph (~540 MB)
    is well under protobuf's 2 GB limit. A stale external-data sidecar from
    the optimum export is removed."""
    model = onnx.load(str(path))
    onnx.save(postprocess_graph(model), str(path))
    sidecar = path.with_name(path.name + "_data")
    if sidecar.exists():
        sidecar.unlink()


# kept for callers and tests that predate the readout branch
expose_inputs_embeds_file = postprocess_file
