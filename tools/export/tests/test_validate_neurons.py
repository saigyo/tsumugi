import numpy as np

from tsumugi_export.neurons import build_neurons, write_neurons
from tsumugi_export.validate import (
    check_activations_parity, check_inv_rms_parity, check_neuron_files, check_neuron_votes_sum,
)

rng = np.random.default_rng(3)
L, N, D, V, K = 2, 6, 8, 40, 4
E = rng.normal(size=(V, D)).astype(np.float32)
G = rng.uniform(0.5, 1.5, D).astype(np.float32)
W = [rng.normal(size=(D, N)).astype(np.float32) for _ in range(L)]
TEXTS = [f"t{i}" for i in range(V)]
NORMS = [np.ones(D, np.float32) * (l + 1) for l in range(L)]
EPS = 1e-5


def built():
    return build_neurons(W, G, E, TEXTS, norms=NORMS, revision="v6", eps=EPS)


def outputs() -> dict:
    """Model outputs consistent with W: MLP part l = W_l · a_l, scaled by g and 1/rms."""
    acts = rng.normal(size=(1, L, N)).astype(np.float32)
    inv = np.array([[[0.25]]], dtype=np.float32)
    parts = rng.normal(size=(1, 1 + 2 * L, D)).astype(np.float32)
    for l in range(L):
        parts[0, 1 + L + l] = (W[l] @ acts[0, l]) * G * inv[0, 0, 0]
    ids = np.arange(K, dtype=np.int64)
    return {"mlp_activations": acts, "final_norm_inv_rms": inv, "stream_parts": parts,
            "topk_ids": ids[None, :], "topk_embeds": E[ids][None, :, :]}


def test_activations_parity_passes_within_tolerance_and_names_the_layer_when_not():
    out = outputs()
    ref = out["mlp_activations"][0].copy()
    assert check_activations_parity(out, ref, 1e-3, 1e-3)["passed"]
    ref[1, 2] += 1.0
    r = check_activations_parity(out, ref, 1e-3, 1e-3)
    assert not r["passed"] and "layer 1" in r["detail"]


def test_activations_parity_reports_a_missing_output():
    out = outputs()
    del out["mlp_activations"]
    assert "no mlp_activations" in check_activations_parity(out, np.zeros((L, N)), 1e-3, 1e-3)["detail"]


def test_inv_rms_parity():
    out = outputs()
    assert check_inv_rms_parity(out, 0.25, 1e-4)["passed"]
    assert not check_inv_rms_parity(out, 0.3, 1e-4)["passed"]


def test_neuron_votes_from_the_int8_asset_add_up_to_the_mlp_votes(tmp_path):
    write_neurons(tmp_path, built())
    out = outputs()
    assert check_neuron_votes_sum(out, tmp_path, k=K)["passed"]
    out["stream_parts"][0, 1 + L] *= 3
    r = check_neuron_votes_sum(out, tmp_path, k=K)
    assert not r["passed"] and "layer 0" in r["detail"]


def test_neuron_files_match_the_manifest(tmp_path):
    ndir = tmp_path / "neurons"
    write_neurons(ndir, built())
    assert check_neuron_files(tmp_path, TEXTS, G, NORMS, EPS)["passed"]
    (ndir / "layer-01.bin").write_bytes(b"\0" * 10)
    assert "layer-01.bin" in check_neuron_files(tmp_path, TEXTS, G, NORMS, EPS)["detail"]


def test_neuron_files_check_the_norm_weight_and_texts(tmp_path):
    write_neurons(tmp_path / "neurons", built())
    assert "finalNormWeight" in check_neuron_files(tmp_path, TEXTS, G * 2, NORMS, EPS)["detail"]
    assert "tokens.json" in check_neuron_files(tmp_path, ["x"] * V, G, NORMS, EPS)["detail"]


def test_a_missing_asset_fails():
    from pathlib import Path
    assert "no neurons/" in check_neuron_files(Path("/nonexistent"), TEXTS, G, NORMS, EPS)["detail"]


def test_neuron_files_check_the_revision_eps_and_post_attention_norm(tmp_path):
    write_neurons(tmp_path / "neurons", built())
    assert check_neuron_files(tmp_path, TEXTS, G, NORMS, EPS)["passed"]
    assert "rmsNormEps" in check_neuron_files(tmp_path, TEXTS, G, NORMS, 1e-6)["detail"]
    assert "normWeight" in check_neuron_files(tmp_path, TEXTS, G, [NORMS[0], NORMS[1] * 2], EPS)["detail"]
