import numpy as np
import onnx
from onnx import TensorProto, helper

from tsumugi_export.quantize import quantize_file


def tiny_model(path):
    """input ids → embedding Gather → a MatMul named like a layer → /lm_head/MatMul."""
    f = TensorProto.FLOAT
    rng = np.random.default_rng(0)
    table = rng.normal(size=(50, 64)).astype(np.float32)
    w = rng.normal(size=(64, 64)).astype(np.float32)
    head = rng.normal(size=(64, 50)).astype(np.float32)
    inits = [helper.make_tensor("model.embed_tokens.weight", f, [50, 64], table.flatten().tolist()),
             helper.make_tensor("w", f, [64, 64], w.flatten().tolist()),
             helper.make_tensor("head", f, [64, 50], head.flatten().tolist())]
    nodes = [helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["e"], name="/model/embed_tokens/Gather"),
             helper.make_node("MatMul", ["e", "w"], ["h"], name="/model/layers.0/mlp/up_proj/MatMul"),
             helper.make_node("MatMul", ["h", "head"], ["logits"], name="/lm_head/MatMul")]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "s"])],
                              [helper.make_tensor_value_info("logits", f, [1, "s", 50])], initializer=inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]), str(path))


def test_layers_become_8_bit_weight_only_the_head_stays_fp32_and_the_embedding_is_uint8(tmp_path):
    src, dst = tmp_path / "m.onnx", tmp_path / "q.onnx"
    tiny_model(src)
    quantize_file(src, dst)
    m = onnx.load(str(dst))
    by_op = {}
    for n in m.graph.node:
        by_op.setdefault(n.op_type, []).append(n)
    nbits = by_op["MatMulNBits"]
    assert len(nbits) == 1
    attrs = {a.name: helper.get_attribute_value(a) for a in nbits[0].attribute}
    assert attrs["bits"] == 8 and attrs["block_size"] == 32
    assert [n.name for n in by_op["MatMul"]] == ["/lm_head/MatMul"]
    assert "DynamicQuantizeLinear" not in by_op                  # activations are never rounded
    assert any(t.data_type == TensorProto.UINT8 and list(t.dims) == [50, 64] for t in m.graph.initializer)


def test_the_quantized_model_stays_close_to_fp32(tmp_path):
    import onnxruntime as ort
    src, dst = tmp_path / "m.onnx", tmp_path / "q.onnx"
    tiny_model(src)
    quantize_file(src, dst)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ids = {"input_ids": np.array([[1, 7, 42]], dtype=np.int64)}
    ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, ids)[0]
    got = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"]).run(None, ids)[0]
    assert np.max(np.abs(got - ref)) < 0.05 * np.max(np.abs(ref))
