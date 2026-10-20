import numpy as np
import onnx
from onnx import TensorProto, helper

from tsumugi_export.postprocess import (
    EDIT_RATIOS_OUTPUT, INV_RMS_OUTPUT, MLP_ACTIVATIONS_OUTPUT, MLP_GATE_OUTPUT, MLP_IN_OUTPUT, MLP_OUT_OUTPUT,
    MLP_UP_OUTPUT, STREAM_PARTS_OUTPUT, expose_mlp_internals, expose_mlp_steps, expose_neurons, find_blocks,
    find_final_norm, find_mlp_inputs, find_mlp_projections, postprocess_graph,
)

DIMS, VOCAB, LAYERS, EPS = 4, 7, 2, 1e-5
rng = np.random.default_rng(1)
TABLE = rng.normal(size=(VOCAB, DIMS)).astype(np.float32)
WA = [rng.normal(scale=0.5, size=(DIMS, DIMS)).astype(np.float32) for _ in range(LAYERS)]
WM = [rng.normal(scale=0.5, size=(DIMS, DIMS)).astype(np.float32) for _ in range(LAYERS)]
GAIN = rng.uniform(0.5, 1.5, size=DIMS).astype(np.float32)
NEUR = 6
WG = [rng.normal(scale=0.5, size=(DIMS, NEUR)).astype(np.float32) for _ in range(LAYERS)]
WU = [rng.normal(scale=0.5, size=(DIMS, NEUR)).astype(np.float32) for _ in range(LAYERS)]


def layered_graph() -> onnx.ModelProto:
    """A miniature of the real export's layout (names and op pattern from the
    2026-10-10 probe): per layer a no-op Cast of the stream, an o_proj MatMul
    feeding the residual Add, another Cast, a down_proj MatMul feeding Add_1;
    then the final RMSNorm (Pow, ReduceMean, Add eps, Sqrt, Div 1/·, Mul,
    Cast, Mul gain) and the lm_head MatMul."""
    f = TensorProto.FLOAT
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"])
    inits = [
        helper.make_tensor("model.embed_tokens.weight", f, [VOCAB, DIMS], TABLE.flatten().tolist()),
        helper.make_tensor("lm_head.weight_t", f, [DIMS, VOCAB], TABLE.T.flatten().tolist()),
        helper.make_tensor("model.norm.weight", f, [DIMS], GAIN.tolist()),
        helper.make_tensor("two", f, [], [2.0]),
        helper.make_tensor("one", f, [], [1.0]),
        helper.make_tensor("eps", f, [], [EPS]),
        helper.make_tensor("axes_last", TensorProto.INT64, [1], [-1]),
    ]
    nodes = [helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["hidden_states.0"],
                              name="/model/embed_tokens/Gather", axis=0)]
    stream = "hidden_states.0"
    for l in range(LAYERS):
        p = f"/model/layers.{l}"
        inits += [helper.make_tensor(f"wa{l}", f, [DIMS, DIMS], WA[l].flatten().tolist()),
                  helper.make_tensor(f"wm{l}", f, [DIMS, DIMS], WM[l].flatten().tolist()),
                  helper.make_tensor(f"wg{l}", f, [DIMS, NEUR], WG[l].flatten().tolist()),
                  helper.make_tensor(f"wu{l}", f, [DIMS, NEUR], WU[l].flatten().tolist())]
        nodes += [
            helper.make_node("Cast", [stream], [f"{p}/input_layernorm/Cast_output_0"], name=f"{p}/input_layernorm/Cast", to=1),
            helper.make_node("MatMul", [f"{p}/input_layernorm/Cast_output_0", f"wa{l}"],
                             [f"{p}/self_attn/o_proj/MatMul_output_0"], name=f"{p}/self_attn/o_proj/MatMul"),
            helper.make_node("Add", [f"{p}/input_layernorm/Cast_output_0", f"{p}/self_attn/o_proj/MatMul_output_0"],
                             [f"{p}/Add_output_0"], name=f"{p}/Add"),
            helper.make_node("Cast", [f"{p}/Add_output_0"], [f"{p}/post_attention_layernorm/Cast_output_0"],
                             name=f"{p}/post_attention_layernorm/Cast", to=1),
            helper.make_node("MatMul", [f"{p}/post_attention_layernorm/Cast_output_0", f"wg{l}"],
                             [f"{p}/mlp/gate_proj/MatMul_output_0"], name=f"{p}/mlp/gate_proj/MatMul"),
            helper.make_node("MatMul", [f"{p}/post_attention_layernorm/Cast_output_0", f"wu{l}"],
                             [f"{p}/mlp/up_proj/MatMul_output_0"], name=f"{p}/mlp/up_proj/MatMul"),
            helper.make_node("MatMul", [f"{p}/post_attention_layernorm/Cast_output_0", f"wm{l}"],
                             [f"{p}/mlp/down_proj/MatMul_output_0"], name=f"{p}/mlp/down_proj/MatMul"),
            helper.make_node("Add", [f"{p}/post_attention_layernorm/Cast_output_0", f"{p}/mlp/down_proj/MatMul_output_0"],
                             [f"{p}/Add_1_output_0"], name=f"{p}/Add_1"),
        ]
        stream = f"{p}/Add_1_output_0"
    n = "/model/norm"
    nodes += [
        helper.make_node("Cast", [stream], [f"{n}/Cast_output_0"], name=f"{n}/Cast", to=1),
        helper.make_node("Pow", [f"{n}/Cast_output_0", "two"], [f"{n}/Pow_output_0"], name=f"{n}/Pow"),
        helper.make_node("ReduceMean", [f"{n}/Pow_output_0", "axes_last"], [f"{n}/ReduceMean_output_0"],
                         name=f"{n}/ReduceMean", keepdims=1),
        helper.make_node("Add", [f"{n}/ReduceMean_output_0", "eps"], [f"{n}/Add_output_0"], name=f"{n}/Add"),
        helper.make_node("Sqrt", [f"{n}/Add_output_0"], [f"{n}/Sqrt_output_0"], name=f"{n}/Sqrt"),
        helper.make_node("Div", ["one", f"{n}/Sqrt_output_0"], [f"{n}/Div_output_0"], name=f"{n}/Div"),
        helper.make_node("Mul", [f"{n}/Cast_output_0", f"{n}/Div_output_0"], [f"{n}/Mul_output_0"], name=f"{n}/Mul"),
        helper.make_node("Cast", [f"{n}/Mul_output_0"], [f"{n}/Cast_1_output_0"], name=f"{n}/Cast_1", to=1),
        helper.make_node("Mul", ["model.norm.weight", f"{n}/Cast_1_output_0"], [f"{n}/Mul_1_output_0"], name=f"{n}/Mul_1"),
        helper.make_node("MatMul", [f"{n}/Mul_1_output_0", "lm_head.weight_t"], ["logits"], name="/lm_head/MatMul"),
    ]
    outs = [helper.make_tensor_value_info("logits", f, [1, "seq", VOCAB]),
            helper.make_tensor_value_info("hidden_states.0", f, [1, "seq", DIMS])]
    graph = helper.make_graph(nodes, "layered", [ids], outs, initializer=inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def reference(ids: np.ndarray):
    """numpy forward of layered_graph: per-layer edits, streams, the final x and 1/rms."""
    x = TABLE[ids[0]]
    e = x.copy()
    attn, mlp, before_a, before_m = [], [], [], []
    for l in range(LAYERS):
        a = x @ WA[l]; before_a.append(x); attn.append(a); x = x + a
        m = x @ WM[l]; before_m.append(x); mlp.append(m); x = x + m
    inv = 1.0 / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS)
    return e, attn, mlp, before_a, before_m, x, inv


def _run(model: onnx.ModelProto, ids: np.ndarray) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": ids})))


def test_find_blocks_returns_edit_and_stream_per_layer():
    blocks = find_blocks(layered_graph().graph)
    assert len(blocks) == LAYERS
    a, xa, m, xm = blocks[1]
    assert a == "/model/layers.1/self_attn/o_proj/MatMul_output_0"
    assert xa == "/model/layers.1/input_layernorm/Cast_output_0"
    assert m == "/model/layers.1/mlp/down_proj/MatMul_output_0"
    assert xm == "/model/layers.1/post_attention_layernorm/Cast_output_0"


def test_find_final_norm_returns_gain_and_inverse_rms():
    assert find_final_norm(layered_graph().graph) == ("model.norm.weight", "/model/norm/Div_output_0")


def test_edit_ratios_match_numpy():
    model = expose_mlp_internals(layered_graph())
    onnx.checker.check_model(model)
    ids = np.array([[2, 5, 1]], dtype=np.int64)
    out = _run(model, ids)
    _, attn, mlp, before_a, before_m, _, _ = reference(ids)
    got = out[EDIT_RATIOS_OUTPUT]
    assert got.shape == (1, 3, LAYERS, 2)
    for l in range(LAYERS):
        np.testing.assert_allclose(got[0, :, l, 0], np.linalg.norm(attn[l], axis=-1) / np.linalg.norm(before_a[l], axis=-1), rtol=1e-5)
        np.testing.assert_allclose(got[0, :, l, 1], np.linalg.norm(mlp[l], axis=-1) / np.linalg.norm(before_m[l], axis=-1), rtol=1e-5)


def test_stream_parts_are_scaled_last_position_parts_and_sum_to_the_norm_output():
    model = expose_mlp_internals(layered_graph())
    ids = np.array([[4, 0, 6, 3]], dtype=np.int64)
    out = _run(model, ids)
    e, attn, mlp, _, _, x, inv = reference(ids)
    parts = out[STREAM_PARTS_OUTPUT]
    assert parts.shape == (1, 1 + 2 * LAYERS, DIMS)
    scale = GAIN * inv[-1]
    np.testing.assert_allclose(parts[0, 0], e[-1] * scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0, 1 + 1], attn[1][-1] * scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0, 1 + LAYERS], mlp[0][-1] * scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0].sum(axis=0), GAIN * x[-1] * inv[-1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts[0].sum(axis=0) @ TABLE.T, out["logits"][0, -1], rtol=1e-4, atol=1e-5)


def test_postprocess_graph_adds_the_mlp_outputs_after_the_readout():
    model = postprocess_graph(layered_graph(), k=3)
    onnx.checker.check_model(model)
    names = [o.name for o in model.graph.output]
    assert names[-9:] == [EDIT_RATIOS_OUTPUT, STREAM_PARTS_OUTPUT, MLP_ACTIVATIONS_OUTPUT, INV_RMS_OUTPUT,
                          MLP_GATE_OUTPUT, MLP_UP_OUTPUT, MLP_IN_OUTPUT, MLP_OUT_OUTPUT, "inputs_embeds"]


def test_a_graph_without_layers_is_refused():
    model = layered_graph()
    for node in model.graph.node:
        if node.name.endswith("o_proj/MatMul"):
            node.name = node.name.replace("o_proj", "renamed")
    try:
        expose_mlp_internals(model)
    except ValueError as err:
        assert "o_proj" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_find_mlp_inputs_names_each_layers_down_proj_input():
    assert find_mlp_inputs(layered_graph().graph) == [
        f"/model/layers.{l}/post_attention_layernorm/Cast_output_0" for l in range(LAYERS)]


def test_neuron_outputs_are_the_last_positions_down_proj_inputs_and_inverse_rms():
    model = expose_neurons(layered_graph())
    onnx.checker.check_model(model)
    ids = np.array([[3, 1, 4]], dtype=np.int64)
    out = _run(model, ids)
    _, _, _, _, before_m, _, inv = reference(ids)
    acts = out[MLP_ACTIVATIONS_OUTPUT]
    assert acts.shape == (1, LAYERS, DIMS)
    for l in range(LAYERS):
        np.testing.assert_allclose(acts[0, l], before_m[l][-1], rtol=1e-5, atol=1e-6)
    assert out[INV_RMS_OUTPUT].shape == (1, 1, 1)
    np.testing.assert_allclose(out[INV_RMS_OUTPUT].reshape(-1), inv[-1], rtol=1e-5)
    declared = next(o for o in model.graph.output if o.name == MLP_ACTIVATIONS_OUTPUT)
    assert [d.dim_value for d in declared.type.tensor_type.shape.dim][1:] == [LAYERS, DIMS]


def test_find_mlp_projections_names_each_layers_gate_and_up_outputs():
    assert find_mlp_projections(layered_graph().graph) == [
        (f"/model/layers.{l}/mlp/gate_proj/MatMul_output_0", f"/model/layers.{l}/mlp/up_proj/MatMul_output_0")
        for l in range(LAYERS)]


def test_step_outputs_are_the_last_positions_gate_up_input_and_edit():
    model = expose_mlp_steps(layered_graph())
    onnx.checker.check_model(model)
    ids = np.array([[3, 1, 4, 1]], dtype=np.int64)
    out = _run(model, ids)
    _, _, mlp, _, before_m, _, _ = reference(ids)
    assert out[MLP_GATE_OUTPUT].shape == (1, LAYERS, NEUR)
    assert out[MLP_IN_OUTPUT].shape == (1, LAYERS, DIMS)
    for l in range(LAYERS):
        x = before_m[l][-1]
        np.testing.assert_allclose(out[MLP_GATE_OUTPUT][0, l], x @ WG[l], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[MLP_UP_OUTPUT][0, l], x @ WU[l], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[MLP_IN_OUTPUT][0, l], x, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[MLP_OUT_OUTPUT][0, l], mlp[l][-1], rtol=1e-5, atol=1e-6)
    declared = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim][1:] for o in model.graph.output}
    assert declared[MLP_UP_OUTPUT] == [LAYERS, NEUR] and declared[MLP_OUT_OUTPUT] == [LAYERS, DIMS]


def test_a_layer_without_up_proj_is_refused():
    model = layered_graph()
    for node in model.graph.node:
        if node.name == "/model/layers.1/mlp/up_proj/MatMul":
            node.name = "/model/layers.1/mlp/renamed/MatMul"
    try:
        expose_mlp_steps(model)
    except ValueError as err:
        assert "up_proj" in str(err)
    else:
        raise AssertionError("expected ValueError")
