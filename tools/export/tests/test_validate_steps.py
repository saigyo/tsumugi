import numpy as np

from tsumugi_export.validate import check_argmax_parity, check_silu_product, check_steps_parity

rng = np.random.default_rng(5)


def test_argmax_parity_passes_when_every_prompt_agrees_within_the_delta():
    a = rng.normal(size=50).astype(np.float32)
    assert check_argmax_parity([(a + 0.1, a), (a, a)])["passed"]


def test_argmax_parity_fails_on_a_different_argmax_or_a_large_delta():
    a = np.zeros(10, np.float32); a[3] = 5
    b = a.copy(); b[7] = 6
    assert not check_argmax_parity([(b, a)])["passed"]
    c = a.copy(); c[0] = 2.0                         # same argmax, Δ = 2.0 > 1.5
    r = check_argmax_parity([(c, a)])
    assert not r["passed"] and "2.0" in r["detail"]


def test_steps_parity_names_the_output_and_layer_that_differ():
    ref = {k: rng.normal(size=(3, 8)).astype(np.float32) for k in ("mlp_gate", "mlp_up", "mlp_in", "mlp_out")}
    outputs = {k: v[None].copy() for k, v in ref.items()}
    assert check_steps_parity(outputs, ref, 1e-3, 1e-3)["passed"]
    outputs["mlp_up"][0, 2, 4] += 1
    r = check_steps_parity(outputs, ref, 1e-3, 1e-3)
    assert not r["passed"] and "mlp_up" in r["detail"] and "layer 2" in r["detail"]
    del outputs["mlp_in"]
    assert "no mlp_in" in check_steps_parity(outputs, ref, 1e-3, 1e-3)["detail"]


def test_silu_of_the_gate_times_up_reproduces_the_activations():
    gate = rng.normal(size=(1, 3, 8)).astype(np.float32)
    up = rng.normal(size=(1, 3, 8)).astype(np.float32)
    acts = gate / (1 + np.exp(-gate)) * up
    assert check_silu_product({"mlp_gate": gate, "mlp_up": up, "mlp_activations": acts})["passed"]
    assert not check_silu_product({"mlp_gate": gate, "mlp_up": up, "mlp_activations": acts + 0.1})["passed"]
