import json

import numpy as np

from tsumugi_export.geometry import build_geometry, write_geometry


def synthetic(vocab=50, dims=8, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(vocab, dims)).astype(np.float32)
    E[1] = E[0] * 2.0            # token 1 is an exact direction-twin of token 0
    E[2] = -E[0]                 # token 2 is the antipode of token 0
    return E, [f"t{i}" for i in range(vocab)]


def decode(built):
    m = built["manifest"]
    v, k = m["vocabSize"], m["k"]
    raw = built["neighbors"]
    ids = np.frombuffer(raw[: v * k * 2], dtype="<u2").reshape(v, k)
    sims = np.frombuffer(raw[v * k * 2:], dtype=np.uint8).reshape(v, k)
    q = np.frombuffer(built["vectors"], dtype=np.int8).reshape(v, m["pcaDims"])
    return m, ids, sims, q


def test_neighbors_exact_self_excluded_and_sorted():
    E, texts = synthetic()
    m, ids, sims, _ = decode(build_geometry(E, texts, k=5, pca_dims=4, block=16))
    assert ids.shape == (50, 5)
    for t in range(50):
        assert t not in ids[t]
        assert list(sims[t]) == sorted(sims[t], reverse=True)
    assert ids[0][0] == 1 and sims[0][0] == 255      # cos = 1 → 255
    assert 2 not in ids[0]                            # antipode never a neighbour (cos = -1)


def test_sims_match_exact_cosine_at_uint8_resolution():
    E, texts = synthetic()
    _, ids, sims, _ = decode(build_geometry(E, texts, k=5, pca_dims=4, block=16))
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    for t in (0, 7, 49):
        for j, s in zip(ids[t], sims[t]):
            assert abs(np.rint(max(U[t] @ U[j], 0) * 255) - s) <= 1


def test_vectors_round_trip_within_scale():
    E, texts = synthetic()
    m, _, _, q = decode(build_geometry(E, texts, k=5, pca_dims=4, block=16))
    assert q.shape == (50, 4)
    C = E - E.mean(axis=0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    P = C @ vt[:4].T
    # PCA axes are sign-ambiguous; compare magnitudes column by column
    assert np.allclose(np.abs(q * m["scale"]), np.abs(P), atol=m["scale"])
    assert 0 < m["explainedVariance"] <= 1


def test_manifest_and_files(tmp_path):
    E, texts = synthetic()
    built = build_geometry(E, texts, k=5, pca_dims=4, block=16)
    write_geometry(tmp_path, built)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["vocabSize"] == 50 and m["dims"] == 8 and m["k"] == 5 and m["pcaDims"] == 4
    for name, size in m["files"].items():
        assert (tmp_path / name).stat().st_size == size
    assert json.loads((tmp_path / "tokens.json").read_text()) == texts
    assert len(m["sourceSha256"]) == 64


def test_texts_length_must_match_vocab():
    E, texts = synthetic()
    try:
        build_geometry(E, texts[:-1], k=5, pca_dims=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
