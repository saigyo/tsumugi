"""Graph post-processing after the optimum export.

The export declares hidden_states.{0..N} as outputs only because torch.onnx
names forward outputs positionally (see onnx_config.py). Here we keep
hidden_states.0 under the artifact-contract name `inputs_embeds` (the
embedding lookup) and drop the rest. We also add the readout, derived from
the lm_head MatMul itself (see find_lm_head): its input as `last_hidden_state`
(the residual stream after the final RMSNorm), and a TopK over the last
position's logits plus a Gather of those candidates' rows from the lm_head's
own fp32 weight, as `topk_ids` / `topk_embeds`. We also add `edit_ratios` and
`stream_parts` for the MLP cards, built from each layer's residual Add nodes and
the final norm's own 1/rms tensor (see expose_mlp_internals). We also add `mlp_activations` and `final_norm_inv_rms`
for the neuron card (see expose_neurons). The tensors already in the
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


EDIT_RATIOS_OUTPUT = "edit_ratios"
STREAM_PARTS_OUTPUT = "stream_parts"
ATTN_MATMUL = "/model/layers.{l}/self_attn/o_proj/MatMul"
MLP_MATMUL = "/model/layers.{l}/mlp/down_proj/MatMul"


def _residual_add(graph: onnx.GraphProto, matmul_name: str) -> tuple[str, str]:
    """(edit, stream before) of the residual Add fed by the named MatMul: the
    MatMul's output is the block's edit, the Add's other input is the stream
    the edit is added to."""
    mm = next((n for n in graph.node if n.name == matmul_name), None)
    if mm is None:
        raise ValueError(f"no node {matmul_name}")
    edit = mm.output[0]
    adds = [n for n in graph.node if n.op_type == "Add" and edit in n.input]
    if len(adds) != 1:
        raise ValueError(f"{matmul_name}: expected one residual Add, found {len(adds)}")
    before = [i for i in adds[0].input if i != edit]
    if len(before) != 1:
        raise ValueError(f"{matmul_name}: residual Add has inputs {list(adds[0].input)}")
    return edit, before[0]


def find_blocks(graph: onnx.GraphProto) -> list[tuple[str, str, str, str]]:
    """Per layer (attention edit, stream before attention, MLP edit, stream
    before MLP), located by the o_proj / down_proj MatMul names (the probe of
    2026-10-10 found the residual Adds right behind them; the quantizer keeps
    these tensor names, and quantize.py already relies on optimum's node
    names for /lm_head/MatMul)."""
    names = {n.name for n in graph.node}
    layers = 0
    while ATTN_MATMUL.format(l=layers) in names:
        layers += 1
    if layers == 0:
        raise ValueError(f"no {ATTN_MATMUL.format(l=0)} — is this the SmolLM2 export?")
    blocks = []
    for l in range(layers):
        a, xa = _residual_add(graph, ATTN_MATMUL.format(l=l))
        m, xm = _residual_add(graph, MLP_MATMUL.format(l=l))
        blocks.append((a, xa, m, xm))
    return blocks


def find_final_norm(graph: onnx.GraphProto) -> tuple[str, str]:
    """(gain, inverse rms) of the final RMSNorm, traced back from the lm_head
    input: Mul(gain [dims], Cast?(Mul(Cast?(x), inv))) with inv made by a Div
    (1 / sqrt(mean(x²) + eps)). Reusing the graph's own inv makes the stream
    parts add up to exactly what the lm_head multiplies."""
    x, _, dims = find_lm_head(graph)
    inits = {t.name: list(t.dims) for t in graph.initializer}

    def through_casts(name: str) -> onnx.NodeProto | None:
        node = _producer(graph, name)
        while node is not None and node.op_type in ("Cast", "Identity"):
            node = _producer(graph, node.input[0])
        return node

    gain_mul = through_casts(x)
    if gain_mul is None or gain_mul.op_type != "Mul":
        raise ValueError("the lm_head input is not produced by the final norm's gain Mul")
    gain = next((i for i in gain_mul.input if inits.get(i) == [dims]), None)
    if gain is None:
        raise ValueError(f"final norm Mul has no [{dims}] gain initializer")
    other = next(i for i in gain_mul.input if i != gain)
    scale_mul = through_casts(other)
    if scale_mul is None or scale_mul.op_type != "Mul":
        raise ValueError("the final norm has no x · (1/rms) Mul")
    inv = next((i for i in scale_mul.input
                if (p := _producer(graph, i)) is not None and p.op_type == "Div"), None)
    if inv is None:
        raise ValueError("the final norm's 1/rms is not produced by a Div")
    return gain, inv


def expose_mlp_internals(model: onnx.ModelProto) -> onnx.ModelProto:
    """edit_ratios [batch, seq, layers, 2]: ‖edit‖ / ‖stream before‖ per fed
    position, layer and block (0 attention, 1 MLP). stream_parts [batch,
    1 + 2·layers, dims]: the last position's embedding, attention edits and
    MLP edits, each multiplied by the final norm's gain and 1/rms — so they
    add up to what the lm_head multiplies (opset 18: axes are inputs)."""
    graph = model.graph
    blocks = find_blocks(graph)
    gain, inv = find_final_norm(graph)
    _, _, dims = find_lm_head(graph)
    layers = len(blocks)
    graph.initializer.extend([
        helper.make_tensor("tsumugi_mlp_axis_last", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_mlp_axis_3", TensorProto.INT64, [1], [3]),
        helper.make_tensor("tsumugi_mlp_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_mlp_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_mlp_slice_axes", TensorProto.INT64, [1], [1]),
    ])
    nodes: list[onnx.NodeProto] = []
    ratios: dict[str, list[str]] = {"attn": [], "mlp": []}
    for l, (a, xa, m, xm) in enumerate(blocks):
        for kind, edit, before in (("attn", a, xa), ("mlp", m, xm)):
            p = f"tsumugi_{kind}{l}"
            nodes += [
                helper.make_node("ReduceL2", [edit, "tsumugi_mlp_axis_last"], [f"{p}_edit_norm"],
                                 name=f"{p}_edit_norm", keepdims=1),
                helper.make_node("ReduceL2", [before, "tsumugi_mlp_axis_last"], [f"{p}_stream_norm"],
                                 name=f"{p}_stream_norm", keepdims=1),
                helper.make_node("Div", [f"{p}_edit_norm", f"{p}_stream_norm"], [f"{p}_ratio"], name=f"{p}_ratio"),
            ]
            ratios[kind].append(f"{p}_ratio")   # [batch, seq, 1]
    for kind in ("attn", "mlp"):
        nodes += [
            helper.make_node("Concat", ratios[kind], [f"tsumugi_{kind}_ratios"], name=f"tsumugi_{kind}_ratios", axis=2),
            helper.make_node("Unsqueeze", [f"tsumugi_{kind}_ratios", "tsumugi_mlp_axis_3"],
                             [f"tsumugi_{kind}_ratios_4d"], name=f"tsumugi_{kind}_ratios_4d"),
        ]
    nodes.append(helper.make_node("Concat", ["tsumugi_attn_ratios_4d", "tsumugi_mlp_ratios_4d"],
                                  [EDIT_RATIOS_OUTPUT], name="tsumugi_edit_ratios", axis=3))

    sources = [blocks[0][1]] + [b[0] for b in blocks] + [b[2] for b in blocks]
    parts = []
    for i, src in enumerate(sources):
        nodes.append(helper.make_node(
            "Slice", [src, "tsumugi_mlp_slice_starts", "tsumugi_mlp_slice_ends", "tsumugi_mlp_slice_axes"],
            [f"tsumugi_part{i}"], name=f"tsumugi_part{i}"))
        parts.append(f"tsumugi_part{i}")
    nodes += [
        helper.make_node("Concat", parts, ["tsumugi_parts_raw"], name="tsumugi_parts_raw", axis=1),
        helper.make_node("Slice", [inv, "tsumugi_mlp_slice_starts", "tsumugi_mlp_slice_ends", "tsumugi_mlp_slice_axes"],
                         ["tsumugi_inv_last"], name="tsumugi_inv_last"),
        helper.make_node("Mul", ["tsumugi_parts_raw", "tsumugi_inv_last"], ["tsumugi_parts_scaled"],
                         name="tsumugi_parts_scaled"),
        helper.make_node("Mul", ["tsumugi_parts_scaled", gain], [STREAM_PARTS_OUTPUT], name="tsumugi_stream_parts"),
    ]
    graph.node.extend(nodes)
    graph.output.extend([
        helper.make_tensor_value_info(EDIT_RATIOS_OUTPUT, TensorProto.FLOAT, ["batch_size", "sequence_length", layers, 2]),
        helper.make_tensor_value_info(STREAM_PARTS_OUTPUT, TensorProto.FLOAT, ["batch_size", 1 + 2 * layers, dims]),
    ])
    return model

MLP_ACTIVATIONS_OUTPUT = "mlp_activations"
INV_RMS_OUTPUT = "final_norm_inv_rms"


def find_mlp_inputs(graph: onnx.GraphProto) -> list[str]:
    """Per layer, the down_proj MatMul's first input: the neuron activations
    SiLU(gate) ⊙ up that the down projection turns into the MLP's edit."""
    by_name = {n.name: n for n in graph.node}
    inputs = []
    while (node := by_name.get(MLP_MATMUL.format(l=len(inputs)))) is not None:
        inputs.append(node.input[0])
    if not inputs:
        raise ValueError(f"no {MLP_MATMUL.format(l=0)} — is this the SmolLM2 export?")
    return inputs


def _neuron_count(graph: onnx.GraphProto) -> int:
    """Rows of layer 0's down_proj weight ([neurons, dims] in the MatMul)."""
    mm = next(n for n in graph.node if n.name == MLP_MATMUL.format(l=0))
    init = next((t for t in graph.initializer if t.name == mm.input[1]), None)
    if init is None or len(init.dims) != 2:
        raise ValueError(f"{MLP_MATMUL.format(l=0)} weight is not a 2-D initializer")
    return int(init.dims[0])


def expose_neurons(model: onnx.ModelProto) -> onnx.ModelProto:
    """mlp_activations [batch, layers, neurons]: the last position's down_proj
    inputs per layer. final_norm_inv_rms [batch, 1, 1]: the final norm's 1/rms
    at the last position. Tapped on the fp32 graph, before quantization, like
    the other added outputs."""
    graph = model.graph
    inputs = find_mlp_inputs(graph)
    _, inv = find_final_norm(graph)
    neurons = _neuron_count(graph)
    graph.initializer.extend([
        helper.make_tensor("tsumugi_neuron_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_neuron_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_neuron_slice_axes", TensorProto.INT64, [1], [1]),
    ])
    slice_args = ["tsumugi_neuron_slice_starts", "tsumugi_neuron_slice_ends", "tsumugi_neuron_slice_axes"]
    nodes = [helper.make_node("Slice", [name, *slice_args], [f"tsumugi_neurons{l}"], name=f"tsumugi_neurons{l}")
             for l, name in enumerate(inputs)]
    nodes += [
        helper.make_node("Concat", [f"tsumugi_neurons{l}" for l in range(len(inputs))], [MLP_ACTIVATIONS_OUTPUT],
                         name="tsumugi_mlp_activations", axis=1),
        helper.make_node("Slice", [inv, *slice_args], [INV_RMS_OUTPUT], name="tsumugi_final_norm_inv_rms"),
    ]
    graph.node.extend(nodes)
    graph.output.extend([
        helper.make_tensor_value_info(MLP_ACTIVATIONS_OUTPUT, TensorProto.FLOAT, ["batch_size", len(inputs), neurons]),
        helper.make_tensor_value_info(INV_RMS_OUTPUT, TensorProto.FLOAT, ["batch_size", 1, 1]),
    ])
    return model


MLP_GATE_OUTPUT = "mlp_gate"
MLP_UP_OUTPUT = "mlp_up"
MLP_IN_OUTPUT = "mlp_in"
MLP_OUT_OUTPUT = "mlp_out"
GATE_MATMUL = "/model/layers.{l}/mlp/gate_proj/MatMul"
UP_MATMUL = "/model/layers.{l}/mlp/up_proj/MatMul"


def find_mlp_projections(graph: onnx.GraphProto) -> list[tuple[str, str]]:
    """Per layer, the outputs of the gate_proj and up_proj MatMuls: the gate
    before SiLU and the up values it scales."""
    by_name = {n.name: n for n in graph.node}
    out: list[tuple[str, str]] = []
    while (gate := by_name.get(GATE_MATMUL.format(l=len(out)))) is not None:
        up = by_name.get(UP_MATMUL.format(l=len(out)))
        if up is None:
            raise ValueError(f"no {UP_MATMUL.format(l=len(out))} next to its gate_proj")
        out.append((gate.output[0], up.output[0]))
    if not out:
        raise ValueError(f"no {GATE_MATMUL.format(l=0)} — is this the SmolLM2 export?")
    return out


def _weight_cols(graph: onnx.GraphProto, matmul_name: str) -> int:
    mm = next(n for n in graph.node if n.name == matmul_name)
    init = next((t for t in graph.initializer if t.name == mm.input[1]), None)
    if init is None or len(init.dims) != 2:
        raise ValueError(f"{matmul_name} weight is not a 2-D initializer")
    return int(init.dims[1])


def expose_mlp_steps(model: onnx.ModelProto) -> onnx.ModelProto:
    """For the step-by-step card (#17), last position only: mlp_gate and mlp_up
    [batch, layers, neurons] (the gate before SiLU and the up values), mlp_in and
    mlp_out [batch, layers, dims] (the stream going into each MLP and its edit).
    Tapped on the fp32 graph before quantization; the quantizer keeps tensor names."""
    graph = model.graph
    proj = find_mlp_projections(graph)
    blocks = find_blocks(graph)
    if len(blocks) != len(proj):
        raise ValueError(f"{len(proj)} gate/up pairs for {len(blocks)} layers")
    _, _, dims = find_lm_head(graph)
    neurons = _weight_cols(graph, GATE_MATMUL.format(l=0))
    graph.initializer.extend([
        helper.make_tensor("tsumugi_step_slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("tsumugi_step_slice_ends", TensorProto.INT64, [1], [INT64_MAX]),
        helper.make_tensor("tsumugi_step_slice_axes", TensorProto.INT64, [1], [1]),
    ])
    slice_args = ["tsumugi_step_slice_starts", "tsumugi_step_slice_ends", "tsumugi_step_slice_axes"]
    sources = {
        MLP_GATE_OUTPUT: [g for g, _ in proj], MLP_UP_OUTPUT: [u for _, u in proj],
        MLP_IN_OUTPUT: [b[3] for b in blocks], MLP_OUT_OUTPUT: [b[2] for b in blocks],
    }
    nodes: list[onnx.NodeProto] = []
    for out_name, names in sources.items():
        parts = [f"tsumugi_{out_name}{l}" for l in range(len(names))]
        nodes += [helper.make_node("Slice", [name, *slice_args], [part], name=part) for name, part in zip(names, parts)]
        nodes.append(helper.make_node("Concat", parts, [out_name], name=f"tsumugi_{out_name}", axis=1))
    graph.node.extend(nodes)
    width = {MLP_GATE_OUTPUT: neurons, MLP_UP_OUTPUT: neurons, MLP_IN_OUTPUT: dims, MLP_OUT_OUTPUT: dims}
    graph.output.extend([helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch_size", len(proj), width[name]])
                         for name in sources])
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
    """Readout, MLP internals, neuron outputs and step outputs (all derived from
    graph structure, independent of the output list), then the inputs_embeds
    rename, which drops every hidden_states.* output."""
    return expose_inputs_embeds(expose_mlp_steps(expose_neurons(expose_mlp_internals(expose_readout(model, k)))))

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
