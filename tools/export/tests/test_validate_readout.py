import numpy as np

from tsumugi_export.postprocess import TOPK_GATHER_NODE
from tsumugi_export.quantize import NODES_TO_EXCLUDE
from tsumugi_export.validate import check_last_hidden_parity, check_topk_parity

rng = np.random.default_rng(0)
E = rng.normal(size=(50, 8)).astype(np.float32)


def good_outputs(seq: int = 3, k: int = 4) -> dict:
    last = rng.normal(size=(1, seq, 8)).astype(np.float32)
    logits = last @ E.T                                   # [1, seq, 50]
    ids = np.argsort(-logits[0, -1], kind="stable")[:k]
    return {
        "logits": logits,
        "last_hidden_state": last,
        "topk_ids": ids[None, :].astype(np.int64),
        "topk_embeds": E[ids][None, :, :],
    }


def test_last_hidden_parity_passes_on_consistent_outputs():
    r = check_last_hidden_parity(good_outputs(), E, tol=1e-4)
    assert r["passed"], r["detail"]


def test_last_hidden_parity_fails_on_drift():
    out = good_outputs()
    out["last_hidden_state"] = out["last_hidden_state"] + 0.1
    assert not check_last_hidden_parity(out, E, tol=1e-4)["passed"]


def test_last_hidden_parity_fails_on_missing_or_misshapen_output():
    out = good_outputs()
    del out["last_hidden_state"]
    r = check_last_hidden_parity(out, E, tol=1e-4)
    assert not r["passed"] and "no last_hidden_state" in r["detail"]
    out = good_outputs()
    out["last_hidden_state"] = out["last_hidden_state"][:, :, :4]
    r = check_last_hidden_parity(out, E, tol=1e-4)
    assert not r["passed"] and "shape" in r["detail"]


def test_topk_parity_passes_on_consistent_outputs():
    r = check_topk_parity(good_outputs(k=4), E, k=4, tol=1e-6)
    assert r["passed"], r["detail"]


def test_topk_parity_accepts_a_different_order_among_exact_ties():
    out = good_outputs(k=4)
    out["logits"][0, -1, :] = 0.0                          # every id ties
    ids = np.array([[7, 3, 11, 0]], dtype=np.int64)
    out["topk_ids"] = ids
    out["topk_embeds"] = E[ids[0]][None, :, :]
    assert check_topk_parity(out, E, k=4, tol=1e-6)["passed"]


def test_topk_parity_fails_when_ids_are_not_the_top_k():
    out = good_outputs(k=4)
    worst = int(np.argmin(out["logits"][0, -1]))           # certainly outside the top 4 of 50
    out["topk_ids"][0, 0] = worst
    out["topk_embeds"] = E[out["topk_ids"][0]][None, :, :]
    r = check_topk_parity(out, E, k=4, tol=1e-6)
    assert not r["passed"] and "top-4" in r["detail"]


def test_topk_parity_fails_when_rows_do_not_match_ids():
    out = good_outputs(k=4)
    out["topk_embeds"] = out["topk_embeds"] + 0.01
    r = check_topk_parity(out, E, k=4, tol=1e-6)
    assert not r["passed"] and "rows" in r["detail"]


def test_topk_parity_fails_on_missing_or_misshapen_output():
    out = good_outputs(k=4)
    del out["topk_embeds"]
    r = check_topk_parity(out, E, k=4, tol=1e-6)
    assert not r["passed"] and "no topk_embeds" in r["detail"]
    out = good_outputs(k=4)
    out["topk_ids"] = out["topk_ids"][:, :3]
    r = check_topk_parity(out, E, k=4, tol=1e-6)
    assert not r["passed"] and "shape" in r["detail"]


def test_quantize_excludes_lm_head_and_the_readout_gather():
    assert "/lm_head/MatMul" in NODES_TO_EXCLUDE
    assert TOPK_GATHER_NODE in NODES_TO_EXCLUDE
