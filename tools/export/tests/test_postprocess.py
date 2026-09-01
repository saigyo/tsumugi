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
