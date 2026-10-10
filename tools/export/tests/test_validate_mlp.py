import numpy as np

from tsumugi_export.validate import (
    check_edit_cache_parity, check_edit_ratios_parity, check_stream_parts_sum, check_votes_sum,
)

rng = np.random.default_rng(2)
L, D, V, K, SEQ = 3, 8, 40, 4, 5
E = rng.normal(size=(V, D)).astype(np.float32)


def good_outputs() -> dict:
    parts = rng.normal(size=(1, 1 + 2 * L, D)).astype(np.float32)
    last = rng.normal(size=(1, SEQ, D)).astype(np.float32)
    last[0, -1] = parts[0].sum(axis=0)
    logits = last @ E.T
    ids = np.argsort(-logits[0, -1], kind="stable")[:K]
    return {
        "stream_parts": parts, "last_hidden_state": last, "logits": logits,
        "topk_ids": ids[None, :].astype(np.int64), "topk_embeds": E[ids][None, :, :],
        "edit_ratios": rng.uniform(0, 2, size=(1, SEQ, L, 2)).astype(np.float32),
    }


def test_stream_parts_sum_passes_and_fails():
    out = good_outputs()
    assert check_stream_parts_sum(out, tol=1e-4)["passed"]
    out["stream_parts"][0, 3] += 0.1
    r = check_stream_parts_sum(out, tol=1e-4)
    assert not r["passed"] and "Σ parts" in r["detail"]


def test_stream_parts_sum_reports_missing_and_misshapen():
    out = good_outputs()
    del out["stream_parts"]
    assert "no stream_parts" in check_stream_parts_sum(out, tol=1e-4)["detail"]
    out = good_outputs()
    out["stream_parts"] = out["stream_parts"][:, :, :4]
    assert "shape" in check_stream_parts_sum(out, tol=1e-4)["detail"]


def test_votes_sum_passes_and_fails():
    out = good_outputs()
    assert check_votes_sum(out, tol=1e-3)["passed"]
    out["topk_embeds"] = out["topk_embeds"] + 0.05
    assert not check_votes_sum(out, tol=1e-3)["passed"]


def test_edit_ratios_parity():
    out = good_outputs()
    expected = out["edit_ratios"][0].copy()
    assert check_edit_ratios_parity(out, expected, tol=1e-3)["passed"]
    r = check_edit_ratios_parity(out, expected + 0.01, tol=1e-3)
    assert not r["passed"] and "Δratio" in r["detail"]
    r = check_edit_ratios_parity(out, expected[:, :2], tol=1e-3)
    assert not r["passed"] and "shape" in r["detail"]
    assert "no edit_ratios" in check_edit_ratios_parity({}, expected, tol=1e-3)["detail"]


def test_edit_cache_parity_compares_each_step_with_its_single_shot_row():
    single = good_outputs()
    steps = [{"edit_ratios": single["edit_ratios"][:, t:t + 1]} for t in range(SEQ)]
    assert check_edit_cache_parity(single, steps, tol=1e-4)["passed"]
    steps[2] = {"edit_ratios": steps[2]["edit_ratios"] + 0.2}
    assert not check_edit_cache_parity(single, steps, tol=1e-4)["passed"]


def test_clipped_comparison_ignores_differences_far_above_the_display_range():
    """q8 moves the huge first-token MLP ratios (≈100) by ~10%; the card draws
    everything ≥ 1 alike, so the quantized gate compares values clipped at 2."""
    out = good_outputs()
    expected = out["edit_ratios"][0].copy()
    out["edit_ratios"][0, 0, 1, 1] = 150.0
    expected[0, 1, 1] = 130.0
    assert not check_edit_ratios_parity(out, expected, tol=0.1)["passed"]
    assert check_edit_ratios_parity(out, expected, tol=0.1, clip=2.0)["passed"]
    out["edit_ratios"][0, 1, 0, 0] = expected[1, 0, 0] + 0.5      # inside the display range: still caught
    assert not check_edit_ratios_parity(out, expected, tol=0.1, clip=2.0)["passed"]
    single = good_outputs()
    steps = [{"edit_ratios": single["edit_ratios"][:, t:t + 1].copy()} for t in range(SEQ)]
    single["edit_ratios"][0, 2, 0, 1] = 90.0
    steps[2]["edit_ratios"][0, 0, 0, 1] = 110.0
    assert not check_edit_cache_parity(single, steps, tol=0.1)["passed"]
    assert check_edit_cache_parity(single, steps, tol=0.1, clip=2.0)["passed"]
