import hashlib
import json

import numpy as np

from tsumugi_export.neurons import build_layer, build_neurons, layer_layout, write_neurons


def decode(raw: bytes, neurons: int, dims: int, labels: int = 5, fractions: int = 3):
    lay = layer_layout(neurons, dims, labels, tuple([0.1] * fractions))
    scales = np.frombuffer(raw[lay["scales"]:lay["columns"]], dtype="<f4")
    q = np.frombuffer(raw[lay["columns"]:lay["labelIds"]], dtype=np.int8).reshape(neurons, dims)
    ids = np.frombuffer(raw[lay["labelIds"]:lay["thresholds"]], dtype="<u2").reshape(neurons, 2, labels)
    thr = np.frombuffer(raw[lay["thresholds"]:lay["bytes"]], dtype="<f4").reshape(neurons, 2, fractions)
    return scales, q, ids, thr


def test_layout_offsets_follow_each_other_and_stay_aligned():
    lay = layer_layout(1536, 576)
    assert lay == {"scales": 0, "columns": 6144, "labelIds": 890880, "thresholds": 921600, "bytes": 958464}
    assert lay["labelIds"] % 2 == 0 and lay["thresholds"] % 4 == 0


def test_columns_survive_the_int8_round_trip_within_half_a_step():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 6)).astype(np.float32)          # [dims, neurons]
    E = rng.normal(size=(40, 8)).astype(np.float32)
    scales, q, _, _ = decode(build_layer(W, np.ones(8, np.float32), E), 6, 8)
    np.testing.assert_allclose(q * scales[:, None], W.T, atol=float(scales.max()) / 2 + 1e-7)


def test_labels_are_the_closest_rows_for_each_sign_and_thresholds_rank_cosines():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(40, 4)).astype(np.float32)
    E[3] = [1, 0, 0, 0]
    E[5] = [-2, 0, 0, 0]                                     # the antipode of row 3
    W = np.zeros((4, 2), np.float32)
    W[:, 0] = [0.5, 0, 0, 0]
    W[:, 1] = rng.normal(size=4)
    _, _, ids, thr = decode(build_layer(W, np.ones(4, np.float32), E), 2, 4)
    assert ids[0, 0, 0] == 3 and ids[0, 1, 0] == 5
    assert thr[0, 0, 0] == np.float32(1.0)                   # rank ⌈0.001·40⌉ = 1: the best cosine
    R = E / np.linalg.norm(E, axis=1, keepdims=True)
    cos = R @ (W[:, 1] / np.linalg.norm(W[:, 1]))
    np.testing.assert_allclose(thr[1, 0], [np.sort(cos)[::-1][r - 1] for r in (1, 1, 4)], rtol=1e-5)
    np.testing.assert_allclose(thr[1, 1], [np.sort(-cos)[::-1][r - 1] for r in (1, 1, 4)], rtol=1e-5)


def test_labels_compare_with_the_rows_scaled_by_the_final_norm_weight():
    E = np.array([[1, 1, 0, 0], [1, -1, 0, 0]] + [[0, 0, 1, 0]] * 3, dtype=np.float32)
    W = np.array([[1], [0.2], [0], [0]], dtype=np.float32)
    _, _, ids, _ = decode(build_layer(W, np.ones(4, np.float32), E), 1, 4)
    assert ids[0, 0, 0] == 0
    _, _, ids, _ = decode(build_layer(W, np.array([1, -1, 1, 1], np.float32), E), 1, 4)
    assert ids[0, 0, 0] == 1


def test_manifest_lists_every_file_with_its_size_and_checksum(tmp_path):
    rng = np.random.default_rng(2)
    W = [rng.normal(size=(4, 3)).astype(np.float32) for _ in range(2)]
    E = rng.normal(size=(30, 4)).astype(np.float32)
    g = rng.uniform(0.5, 1.5, 4).astype(np.float32)
    built = build_neurons(W, g, E, [f"t{i}" for i in range(30)])
    write_neurons(tmp_path, built)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert (m["layers"], m["neurons"], m["dims"], m["vocabSize"]) == (2, 3, 4, 30)
    assert m["finalNormWeight"] == [float(v) for v in g]
    assert set(m["files"]) == {"layer-00.bin", "layer-01.bin", "tokens.json"}
    for name, f in m["files"].items():
        data = (tmp_path / name).read_bytes()
        assert len(data) == f["bytes"] and hashlib.sha256(data).hexdigest() == f["sha256"]
    assert m["files"]["layer-00.bin"]["bytes"] == m["layout"]["bytes"]
    assert json.loads((tmp_path / "tokens.json").read_text()) == [f"t{i}" for i in range(30)]
