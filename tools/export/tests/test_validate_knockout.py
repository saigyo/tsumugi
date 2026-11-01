import numpy as np

from tsumugi_export.validate import TIE_MARGIN, check_knockout_parity

rng = np.random.default_rng(11)


def _logits():
    return {l: rng.normal(size=500).astype(np.float32) for l in (0, 13, 29)}


def test_identical_logits_pass():
    ref = _logits()
    r = check_knockout_parity({l: v.copy() for l, v in ref.items()}, ref, 10, 1e-3, TIE_MARGIN)
    assert r["passed"]


def test_argmax_flip_beyond_the_tie_margin_fails_and_names_the_layer():
    ref = _logits()
    ours = {l: v.copy() for l, v in ref.items()}
    top = int(np.argmax(ref[13]))
    other = (top + 1) % 500
    ours[13][top] = ref[13][top] - 5
    ours[13][other] = ref[13][top] + 1
    r = check_knockout_parity(ours, ref, 10, 100.0, TIE_MARGIN)
    assert not r["passed"] and "13" in r["detail"]


def test_argmax_flip_within_the_tie_margin_passes():
    ref = {0: np.zeros(50, np.float32)}
    ref[0][3] = 10.0
    ref[0][4] = 10.0 - 0.05
    ours = {0: ref[0].copy()}
    ours[0][4] = 10.0 + 0.01
    ours[0][3] = 10.0
    assert check_knockout_parity(ours, ref, 5, 0.1, TIE_MARGIN)["passed"]


def test_a_top_k_logit_off_by_more_than_the_tolerance_fails():
    ref = _logits()
    ours = {l: v.copy() for l, v in ref.items()}
    second = int(np.argsort(ref[29])[-2])
    ours[29][second] += 0.01
    assert not check_knockout_parity(ours, ref, 10, 1e-3, TIE_MARGIN)["passed"]
    assert check_knockout_parity(ours, ref, 10, 0.1, TIE_MARGIN)["passed"]


def test_missing_layer_fails():
    ref = _logits()
    assert not check_knockout_parity({0: ref[0]}, ref, 10, 1e-3, TIE_MARGIN)["passed"]
