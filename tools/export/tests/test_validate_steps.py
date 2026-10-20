import numpy as np

from tsumugi_export.validate import (check_argmax_parity, check_silu_product, check_steps_parity,
                                     greedy_verdict)

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


def _split_logits(top, ours_tok, margin):
    v = np.zeros(20000, np.float32)
    v[top] = 17.75
    v[ours_tok] = 17.75 - margin
    return v


def test_greedy_verdict_passes_identical_sequences():
    v = greedy_verdict([1, 2, 3], [1, 2, 3], None)
    assert v["passed"] and "matches stock" in v["detail"]


def test_greedy_verdict_accepts_a_flip_on_a_near_tie():
    v = greedy_verdict([284, 15100, 30], [284, 357, 436], _split_logits(357, 15100, 0.01))
    assert v["passed"] and "near-tie" in v["detail"] and "step 2" in v["detail"]


def test_greedy_verdict_fails_a_flip_with_a_clear_margin_and_names_it():
    v = greedy_verdict([284, 15100, 30], [284, 357, 436], _split_logits(357, 15100, 2.0))
    assert not v["passed"] and "2.000" in v["detail"]
