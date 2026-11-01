import numpy as np
import onnx
from onnx import TensorProto

from test_postprocess_mlp import DIMS, GAIN, LAYERS, TABLE, WA, WM, layered_graph
from tsumugi_export.postprocess import (
    EDIT_RATIOS_OUTPUT, MLP_ACTIVATIONS_OUTPUT, MLP_OUT_OUTPUT, MLP_SCALE_INPUT, MLP_MATMUL, STREAM_PARTS_OUTPUT,
    postprocess_graph,
)

IDS = np.array([[3, 1, 4, 1]], dtype=np.int64)


def _run(model: onnx.ModelProto, scale: list[float]) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {"input_ids": IDS, MLP_SCALE_INPUT: np.array(scale, dtype=np.float32)}
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))


def test_mlp_scale_is_a_float_input_of_layer_length():
    m = postprocess_graph(layered_graph(), k=3)
    onnx.checker.check_model(m)
    inp = next(i for i in m.graph.input if i.name == MLP_SCALE_INPUT)
    assert inp.type.tensor_type.elem_type == TensorProto.FLOAT
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [LAYERS]


def test_ones_reproduce_the_unscaled_graph():
    scaled = _run(postprocess_graph(layered_graph(), k=3), [1.0] * LAYERS)
    # the same passes without add_mlp_scale: drop its input and Muls by running the pre-scale graph
    from tsumugi_export import postprocess as pp
    plain = pp.expose_inputs_embeds(pp.expose_mlp_steps(pp.expose_neurons(
        pp.expose_mlp_internals(pp.expose_readout(layered_graph(), 3)))))
    import onnxruntime as ort
    sess = ort.InferenceSession(plain.SerializeToString(), providers=["CPUExecutionProvider"])
    ref = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": IDS})))
    assert scaled.keys() == ref.keys()
    for name in ref:
        np.testing.assert_allclose(scaled[name], ref[name], atol=1e-6, err_msg=name)


def test_zero_removes_one_layers_edit():
    out = _run(postprocess_graph(layered_graph(), k=3), [1.0, 0.0])
    np.testing.assert_array_equal(out[MLP_OUT_OUTPUT][0, 1], np.zeros(DIMS))
    assert np.any(out[MLP_OUT_OUTPUT][0, 0] != 0)
    np.testing.assert_array_equal(out[STREAM_PARTS_OUTPUT][0, 1 + LAYERS + 1], np.zeros(DIMS))
    np.testing.assert_array_equal(out[EDIT_RATIOS_OUTPUT][..., 1, 1], np.zeros((1, IDS.shape[1])))
    x = TABLE[IDS[0]]
    for l in range(LAYERS):
        x = x + x @ WA[l]
        if l == 0:
            x = x + x @ WM[l]
    inv = 1.0 / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(out["logits"][0], (GAIN * x * inv) @ TABLE.T, atol=1e-5)


def test_mlp_activations_are_not_scaled():
    model = postprocess_graph(layered_graph(), k=3)
    ones = _run(model, [1.0, 1.0])[MLP_ACTIVATIONS_OUTPUT]
    # layer 1's activations depend on layer 0's edit, so only layer 0 is comparable
    zeros = _run(model, [0.0, 0.0])[MLP_ACTIVATIONS_OUTPUT]
    np.testing.assert_allclose(zeros[0, 0], ones[0, 0], atol=1e-6)


def test_only_the_scale_mul_reads_the_raw_edit():
    model = postprocess_graph(layered_graph(), k=3)
    for l in range(LAYERS):
        edit = next(n for n in model.graph.node if n.name == MLP_MATMUL.format(l=l)).output[0]
        readers = [n for n in model.graph.node if edit in n.input]
        assert [(n.op_type, n.name) for n in readers] == [("Mul", f"tsumugi_mlp{l}_scale_mul")]
