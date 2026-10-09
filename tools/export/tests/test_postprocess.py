import numpy as np
import onnx
from onnx import TensorProto, helper

from tsumugi_export.postprocess import expose_inputs_embeds


def tiny_graph() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, "seq", 4])
    nodes = [
        helper.make_node("Identity", ["input_ids"], ["hidden_states.0"]),
        helper.make_node("Neg", ["hidden_states.0"], ["hidden_states.1"]),
        helper.make_node("Identity", ["hidden_states.1"], ["logits"]),
        helper.make_node("Abs", ["input_ids"], ["attentions.0"]),
    ]
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, "seq", 4])
            for n in ["logits", "hidden_states.0", "hidden_states.1", "attentions.0"]]
    graph = helper.make_graph(nodes, "tiny", [x], outs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_keeps_embeds_drops_other_hidden_states():
    model = expose_inputs_embeds(tiny_graph())
    onnx.checker.check_model(model)
    assert [o.name for o in model.graph.output] == ["logits", "attentions.0", "inputs_embeds"]


def test_inputs_embeds_equals_hidden_states_0():
    import onnxruntime as ort
    model = expose_inputs_embeds(tiny_graph())
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": x})))
    np.testing.assert_array_equal(out["inputs_embeds"], x)
    np.testing.assert_array_equal(out["logits"], -x)


def test_missing_hidden_states_0_raises():
    model = tiny_graph()
    del model.graph.output[1]   # hidden_states.0
    try:
        expose_inputs_embeds(model)
    except ValueError as err:
        assert "hidden_states.0" in str(err)
    else:
        raise AssertionError("expected ValueError")


from tsumugi_export.postprocess import (  # noqa: E402 — readout tests below
    LAST_HIDDEN_OUTPUT, TOPK_EMBEDS_OUTPUT, TOPK_GATHER_NODE, TOPK_IDS_OUTPUT,
    expose_readout, postprocess_graph,
)

TABLE = np.arange(28, dtype=np.float32).reshape(7, 4) / 10.0   # E: 7 tokens × 4 dims


def readout_graph() -> onnx.ModelProto:
    """A miniature causal LM: Gather(E) → hidden_states.0, Neg → hidden_states.1,
    MatMul(hidden_states.1, Eᵀ) → logits. Mirrors the real export's shape
    conventions: hidden_states.N is the tensor the lm_head multiplies."""
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"])
    table = helper.make_tensor("model.embed_tokens.weight", TensorProto.FLOAT, [7, 4], TABLE.flatten().tolist())
    table_t = helper.make_tensor("lm_head.weight_t", TensorProto.FLOAT, [4, 7], TABLE.T.flatten().tolist())
    nodes = [
        helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["hidden_states.0"], axis=0),
        helper.make_node("Neg", ["hidden_states.0"], ["hidden_states.1"]),
        helper.make_node("MatMul", ["hidden_states.1", "lm_head.weight_t"], ["logits"]),
    ]
    outs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "seq", 7]),
        helper.make_tensor_value_info("hidden_states.0", TensorProto.FLOAT, [1, "seq", 4]),
        helper.make_tensor_value_info("hidden_states.1", TensorProto.FLOAT, [1, "seq", 4]),
    ]
    graph = helper.make_graph(nodes, "readout", [ids], outs, initializer=[table, table_t])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _run(model: onnx.ModelProto, ids: np.ndarray) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": ids})))


def test_readout_adds_three_outputs_and_keeps_graph_valid():
    model = expose_readout(readout_graph(), k=3)
    onnx.checker.check_model(model)
    names = [o.name for o in model.graph.output]
    assert names[-3:] == [LAST_HIDDEN_OUTPUT, TOPK_IDS_OUTPUT, TOPK_EMBEDS_OUTPUT]
    assert any(n.name == TOPK_GATHER_NODE for n in model.graph.node)


def test_readout_values_match_numpy():
    model = expose_readout(readout_graph(), k=3)
    ids = np.array([[2, 5, 1]], dtype=np.int64)
    out = _run(model, ids)
    last = -TABLE[ids[0]]                       # Neg of the gathered rows
    logits = last @ TABLE.T
    np.testing.assert_allclose(out[LAST_HIDDEN_OUTPUT][0], last, atol=1e-6)
    expected_ids = np.argsort(-logits[-1], kind="stable")[:3]
    np.testing.assert_array_equal(out[TOPK_IDS_OUTPUT][0], expected_ids)
    np.testing.assert_allclose(out[TOPK_EMBEDS_OUTPUT][0], TABLE[expected_ids], atol=1e-6)
    assert out[TOPK_IDS_OUTPUT].shape == (1, 3)
    assert out[TOPK_EMBEDS_OUTPUT].shape == (1, 3, 4)


def test_postprocess_graph_combines_readout_and_inputs_embeds():
    model = postprocess_graph(readout_graph(), k=3)
    names = [o.name for o in model.graph.output]
    assert names == ["logits", LAST_HIDDEN_OUTPUT, TOPK_IDS_OUTPUT, TOPK_EMBEDS_OUTPUT, "inputs_embeds"]
    out = _run(model, np.array([[0, 6]], dtype=np.int64))
    np.testing.assert_allclose(out["inputs_embeds"][0], TABLE[[0, 6]], atol=1e-6)


def test_readout_without_final_hidden_state_raises():
    model = readout_graph()
    del model.graph.output[2]   # hidden_states.1 — leaves only the embedding lookup
    try:
        expose_readout(model, k=3)
    except ValueError as err:
        assert "hidden_states.1" in str(err) or "final" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_readout_gathers_from_the_lm_head_weight_not_the_embed_table():
    """The quantizer turns the embed table into int8 for the embedding Gather;
    the lm_head keeps its own fp32 copy. The readout must read that copy so
    the rows are exactly what produced the logits."""
    model = expose_readout(readout_graph(), k=3)
    gather = next(n for n in model.graph.node if n.name == TOPK_GATHER_NODE)
    assert gather.input[0] == "lm_head.weight_t"


def test_readout_requires_an_lm_head_matmul_on_an_initializer():
    model = readout_graph()
    matmul = model.graph.node[2]
    matmul.op_type = "Identity"
    del matmul.input[1]
    try:
        expose_readout(model, k=3)
    except ValueError as err:
        assert "MatMul" in str(err)
    else:
        raise AssertionError("expected ValueError")
