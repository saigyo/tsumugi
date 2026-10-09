"""Graph post-processing after the optimum export.

The export declares hidden_states.{0..N} as outputs only because torch.onnx
names forward outputs positionally (see onnx_config.py). Here we keep
hidden_states.0 under the artifact-contract name `inputs_embeds` (the
embedding lookup) and drop the rest. We also add the readout, derived from
the lm_head MatMul itself (see find_lm_head): its input as `last_hidden_state`
(the residual stream after the final RMSNorm), and a TopK over the last
position's logits plus a Gather of those candidates' rows from the lm_head's
own fp32 weight, as `topk_ids` / `topk_embeds`. The tensors already in the
graph stay; only the output list and a handful of tiny nodes change, so this
costs nothing measurable at inference time."""
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


def find_lm_head(graph: onnx.GraphProto) -> tuple[str, str, int]:
    """(x, Wᵀ, dims) of the lm_head: the MatMul that produces `logits` (through
    Identity/Cast wrappers). Both readout tensors derive from it:

    - x, its first input, is the residual stream after the final RMSNorm —
      what the lm_head actually multiplies. NOT hidden_states.N: in this export
      that is the norm's INPUT (found in the first operator run: x·Eᵀ off by
      hundreds), so the hidden_states.* outputs are not consulted at all.
    - Wᵀ, its second input, is an initializer of shape [dims, vocab]: the
      export's own fp32 copy of Eᵀ, which the quantizer leaves alone when the
      MatMul is excluded. The embedding table itself is NOT a usable source —
      the quantizer turns it into int8 for the embedding Gather and rewires
      every other consumer to a dequantized copy (rows off by 7.5e-3).

    Structural, not by name — optimum's tensor names are not our contract."""
    name = "logits"
    for _ in range(8):
        node = _producer(graph, name)
        if node is None:
            raise ValueError(f"no node produces {name} — expected a MatMul behind logits")
        if node.op_type in ("Identity", "Cast"):
            name = node.input[0]
            continue
        if node.op_type != "MatMul":
            raise ValueError(f"logits is produced by {node.op_type}, expected MatMul")
        x, weight = node.input[0], node.input[1]
        init = next((t for t in graph.initializer if t.name == weight), None)
        if init is None:
            raise ValueError(f"lm_head MatMul weight {weight} is not an initializer")
        if len(init.dims) != 2 or init.dims[0] >= init.dims[1]:
            raise ValueError(f"lm_head weight {weight} has shape {list(init.dims)}, expected [dims, vocab]")
        return x, weight, int(init.dims[0])
    raise ValueError("gave up tracing logits back to a MatMul")


def expose_readout(model: onnx.ModelProto, k: int = TOPK) -> onnx.ModelProto:
    graph = model.graph
    if not any(o.name == "logits" for o in graph.output):
        raise ValueError("graph has no logits output")
    x, weight_t, dims = find_lm_head(graph)   # Wᵀ is [dims, vocab]: rows of E are its columns

    graph.initializer.extend([
        helper.make_tensor("tsumugi_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_slice_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("tsumugi_squeeze_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("tsumugi_topk_k", TensorProto.INT64, [1], [k]),
    ])
    graph.node.extend([
        helper.make_node("Identity", [x], [LAST_HIDDEN_OUTPUT], name="tsumugi_last_hidden_identity"),
        helper.make_node("Slice", ["logits", "tsumugi_slice_starts", "tsumugi_slice_ends", "tsumugi_slice_axes"],
                         ["tsumugi_last_logits_3d"], name="tsumugi_last_logits_slice"),
        helper.make_node("Squeeze", ["tsumugi_last_logits_3d", "tsumugi_squeeze_axes"],
                         ["tsumugi_last_logits"], name="tsumugi_last_logits_squeeze"),
        # values output is named but consumed by nothing, which ONNX allows
        helper.make_node("TopK", ["tsumugi_last_logits", "tsumugi_topk_k"],
                         ["tsumugi_topk_values", TOPK_IDS_OUTPUT],
                         name="tsumugi_topk", axis=-1, largest=1, sorted=1),
        # Gather columns of Eᵀ: [dims, vocab] × ids [batch, k] → [dims, batch, k],
        # then bring the row axis last → [batch, k, dims]
        helper.make_node("Gather", [weight_t, TOPK_IDS_OUTPUT], ["tsumugi_topk_embeds_t"],
                         name=TOPK_GATHER_NODE, axis=1),
        helper.make_node("Transpose", ["tsumugi_topk_embeds_t"], [TOPK_EMBEDS_OUTPUT],
                         name="tsumugi_topk_embeds_transpose", perm=[1, 2, 0]),
    ])
    graph.output.extend([
        helper.make_tensor_value_info(LAST_HIDDEN_OUTPUT, TensorProto.FLOAT, ["batch_size", "sequence_length", dims]),
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
    """Readout (derived from the lm_head MatMul, independent of the output
    list), then the inputs_embeds rename, which drops every hidden_states.*
    output."""
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
