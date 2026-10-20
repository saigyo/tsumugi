"""Validation gate. Runs the exported graphs in onnxruntime and checks:
1. logits parity with the stock export (greedy continuation + last logits)
2. attention rows row-stochastic and causal
3. A≡B: cached incremental rows equal no-cache full-matrix rows (only when
   both variants are present in out/; otherwise reported as skipped)
4. cache integrity: present.* outputs exist and multi-step cached logits
   match single-shot full-context logits
5. inputs_embeds parity with the stock embedding table
6. geometry/ spot-check against exact neighbours and the tokenizer
7. last_hidden_state · Eᵀ reproduces the last-position logits
8. topk_ids are the exact top-k and topk_embeds their rows of E
9. edit_ratios match PyTorch forward hooks on the stock model, and the
   cached step-by-step rows match the single-shot rows
10. stream_parts add up to last_hidden_state's last row, and parts · E
    add up to the logits of the top-k ids
11. mlp_activations and final_norm_inv_rms match PyTorch hooks; per-neuron
    votes from neurons/ add up to the MLP votes; neurons/ matches its manifest
12. argmax parity under browser semantics; mlp_gate/up/in/out match PyTorch
    hooks and SiLU(gate)·up = mlp_activations
Writes validation-report.json with artifact hashes; publish refuses to
upload without a passing report."""
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from tsumugi_export import STOCK_MODEL_ID, TOPK

PROMPTS = [
    "The cat sat on the mat because it was tired",
    "one two three one two three one",     # repeats → induction rows exist
]
TOL = 2e-2  # fp32-vs-quantized logit tolerance; fp32-vs-fp32 uses 1e-4


ARGMAX_PROMPTS = [
    "The cat sat on the mat because it was tired",
    "one two three one two three one",
    "The capital of France is",
    "Once upon a time, there was a little",
]
STEP_OUTPUTS = ("mlp_gate", "mlp_up", "mlp_in", "mlp_out")


def _session(path: Path, browser: bool = False):
    """browser=True disables graph optimizations: onnxruntime-web runs the
    unfused graph, and full optimization fuses quantized matmuls into kernels
    that hide rounding the browser does (the #19 lesson)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    if browser:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _tokenize(model_dir: Path, prompt: str) -> np.ndarray:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return np.array([tok.encode(prompt)], dtype=np.int64)


def _feeds(sess, input_ids: np.ndarray) -> dict:
    names = {i.name for i in sess.get_inputs()}
    feeds = {"input_ids": input_ids,
             "attention_mask": np.ones_like(input_ids)}
    if "position_ids" in names:
        feeds["position_ids"] = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    # empty past for cached graphs
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values"):
            shape = [d if isinstance(d, int) else (1 if "batch" in str(d) else 0)
                     for d in i.shape]
            feeds[i.name] = np.zeros(shape, dtype=np.float32)
    return feeds


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_past_inputs(sess) -> bool:
    return any(i.name.startswith("past_key_values") for i in sess.get_inputs())


def _drive_cached_stepwise(sess, ids: np.ndarray) -> list[dict]:
    """Run a with-past graph token by token over `ids`, feeding each
    step's present.* outputs back in as the next step's past_key_values.*
    inputs (step 0 uses an all-zero, past_sequence_length=0 cache).
    Returns one output dict per token position — shared by both the A≡B
    equivalence check and the cache-integrity check so the stepping logic
    isn't duplicated between them."""
    seq_len = ids.shape[1]
    out_names = [o.name for o in sess.get_outputs()]
    in_names = {i.name for i in sess.get_inputs()}
    past_specs = {i.name: i for i in sess.get_inputs()
                  if i.name.startswith("past_key_values")}

    def empty_past() -> dict:
        feeds = {}
        for name, spec in past_specs.items():
            shape = [d if isinstance(d, int) else (1 if "batch" in str(d) else 0)
                     for d in spec.shape]
            feeds[name] = np.zeros(shape, dtype=np.float32)
        return feeds

    past_feeds = empty_past()
    steps = []

    for t in range(seq_len):
        feeds = {
            "input_ids": ids[:, t:t + 1],
            "attention_mask": np.ones((1, t + 1), dtype=np.int64),
        }
        if "position_ids" in in_names:
            feeds["position_ids"] = np.array([[t]], dtype=np.int64)
        feeds.update(past_feeds)

        outputs = dict(zip(out_names, sess.run(None, feeds)))
        steps.append(outputs)

        # carry this step's present.* forward as next step's past_key_values.*
        next_past = {}
        for name in past_specs:
            layer_key = name[len("past_key_values."):]  # "{i}.key" / "{i}.value"
            next_past[name] = outputs[f"present.{layer_key}"]
        past_feeds = next_past

    return steps


def _check_a_equiv_b(model_dir: Path, nocache_model: Path) -> dict:
    """Drive the cached (with-past) graph token by token (via
    _drive_cached_stepwise) and compare each step's per-layer attention
    row (the row for the newly-generated query position) against the
    corresponding row of the no-cache graph's full attention matrix (run
    once, over the whole sequence). The two should agree up to numerical
    noise: causal attention over the same tokens should produce identical
    probabilities whether computed incrementally with a KV cache or all
    at once."""
    cached_sess = _session(model_dir / "onnx" / "model.onnx")
    nocache_sess = _session(nocache_model)

    ids = _tokenize(model_dir, PROMPTS[0])

    # single full-sequence pass through the no-cache graph
    nc_out_names = [o.name for o in nocache_sess.get_outputs()]
    nc_result = dict(zip(nc_out_names, nocache_sess.run(None, _feeds(nocache_sess, ids))))
    nc_attn = {n: v for n, v in nc_result.items() if n.startswith("attentions.")}

    steps = _drive_cached_stepwise(cached_sess, ids)
    cached_attn_names = sorted(
        (n for n in steps[0] if n.startswith("attentions.")),
        key=lambda n: int(n.split(".")[1]),
    )

    max_dev = 0.0
    for t, outputs in enumerate(steps):
        for name in cached_attn_names:
            row = outputs[name][0, :, -1, :]           # [heads, t+1] (this step's query)
            nc_row = nc_attn[name][0, :, t, :t + 1]     # [heads, t+1] (row t of full matrix)
            dev = float(np.max(np.abs(row - nc_row)))
            max_dev = max(max_dev, dev)

    return {
        "passed": max_dev <= 1e-3,
        "detail": f"max row deviation {max_dev:.2e} (atol 1e-3)",
    }


def _greedy_continue(sess, model_dir: Path, prompt: str, n_steps: int) -> list[int]:
    """Greedily generate `n_steps` continuation token ids for `prompt`.
    Used as the primary cross-precision parity check (spec check 4):
    quantization legitimately perturbs individual logit values, but a sound
    quantized graph should still make the same greedy choices as the fp32
    stock model almost all of the time. Graphs with a KV cache are driven
    via _drive_cached_stepwise, re-run from scratch on the growing sequence
    each step — simple, reuses the existing stepping logic, and N=8 steps
    over a short prompt is cheap; graphs without a cache do a full forward
    pass over the growing sequence each step."""
    ids = _tokenize(model_dir, prompt)
    out_names = [o.name for o in sess.get_outputs()]
    generated: list[int] = []

    for _ in range(n_steps):
        if _has_past_inputs(sess):
            steps = _drive_cached_stepwise(sess, ids)
            last_logits = steps[-1]["logits"][0, -1]
        else:
            outputs = dict(zip(out_names, sess.run(None, _feeds(sess, ids))))
            last_logits = outputs["logits"][0, -1]
        next_id = int(np.argmax(last_logits))
        generated.append(next_id)
        ids = np.concatenate([ids, np.array([[next_id]], dtype=np.int64)], axis=1)

    return generated


def _check_greedy_parity(sess, stock_sess, model_dir: Path, n_steps: int = 8) -> dict:
    """Greedy-decode N tokens from PROMPTS[0] on both `sess` and the stock
    fp32 session and compare the generated token-id sequences. This is the
    PRIMARY parity verdict for the quantized variants (model_q4/model_fp16):
    their per-logit diff against stock is expected to exceed a tight
    tolerance by design, so token identity — not the raw magnitude — is
    what should gate publish."""
    ours = _greedy_continue(sess, model_dir, PROMPTS[0], n_steps)
    theirs = _greedy_continue(stock_sess, model_dir, PROMPTS[0], n_steps)
    match = ours == theirs
    return {
        "passed": match,
        "detail": (f"{n_steps}-step greedy continuation matches stock: {ours}" if match
                   else f"{n_steps}-step greedy continuation diverges: ours={ours} stock={theirs}"),
    }


def _check_cache_integrity(sess, model_dir: Path, tol: float, strict: bool = True) -> dict:
    """Cache integrity: tokenize PROMPTS[0], run the full sequence in one
    single-shot (prefill) call to get the last-position logits, then
    drive the same graph token by token via _drive_cached_stepwise and
    take the final step's logits — the two must agree, since they are the
    same causal computation done two different ways. If the graph has no
    past_key_values inputs at all (e.g. export.py failed to pass
    use_past_in_inputs), there is no cache to validate — fail loudly
    rather than silently skipping."""
    if not _has_past_inputs(sess):
        return {"passed": False, "detail": "graph has no past_key_values inputs"}

    ids = _tokenize(model_dir, PROMPTS[0])
    out_names = [o.name for o in sess.get_outputs()]

    single_shot = dict(zip(out_names, sess.run(None, _feeds(sess, ids))))
    single_logits = single_shot["logits"][0, -1]

    steps = _drive_cached_stepwise(sess, ids)
    multi_logits = steps[-1]["logits"][0, -1]

    diff = float(np.max(np.abs(single_logits - multi_logits)))
    if strict:
        return {
            "passed": diff < tol,
            "detail": f"max|Δlogit| multi-step vs single-shot={diff:.2e} (tol {tol})",
        }
    # Dynamically-quantized variants compute activation scales per call, so
    # the two computation paths legitimately differ numerically; the
    # integrity bar there is argmax agreement (the same next token), with
    # the numeric drift kept as information.
    same_argmax = int(np.argmax(single_logits)) == int(np.argmax(multi_logits))
    return {
        "passed": same_argmax,
        "detail": f"argmax agrees={same_argmax}; max|Δlogit|={diff:.2e} (informational for quantized variants)",
    }


def check_inputs_embeds_parity(outputs: dict, ids: np.ndarray, E: np.ndarray, tol: float) -> dict:
    """inputs_embeds must be exactly the embedding-table rows of the fed ids
    (within quantization noise for the int8 variant). Pure over one run's
    outputs, like the readout checks, so the three share a single forward."""
    if "inputs_embeds" not in outputs:
        return {"passed": False, "detail": "graph has no inputs_embeds output"}
    got = outputs["inputs_embeds"]
    expected = E[ids[0]][None, :, :]
    if got.shape != expected.shape:
        return {"passed": False, "detail": f"inputs_embeds shape {got.shape}, expected {expected.shape}"}
    diff = float(np.max(np.abs(got - expected)))
    return {"passed": diff <= tol, "detail": f"max|Δembed|={diff:.2e} (atol {tol})"}


def check_last_hidden_parity(outputs: dict, E: np.ndarray, tol: float) -> dict:
    """last_hidden_state is what the tied lm_head multiplies, so its last row
    against Eᵀ must reproduce the last-position logits."""
    if "last_hidden_state" not in outputs:
        return {"passed": False, "detail": "graph has no last_hidden_state output"}
    last, logits = outputs["last_hidden_state"], outputs["logits"]
    expected = (1, logits.shape[1], E.shape[1])
    if last.shape != expected:
        return {"passed": False, "detail": f"last_hidden_state shape {last.shape}, expected {expected}"}
    diff = float(np.max(np.abs(last[0, -1] @ E.T - logits[0, -1])))
    return {"passed": diff <= tol, "detail": f"max|x·Eᵀ − logits|={diff:.2e} (atol {tol})"}


def check_topk_parity(outputs: dict, E: np.ndarray, k: int, tol: float) -> dict:
    """topk_ids must be the exact top-k of the last-position logits (order may
    differ only among exact ties) and topk_embeds their rows of E. The gather
    is excluded from quantization, so no quantization slack applies."""
    for name in ("topk_ids", "topk_embeds"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    ids, rows, logits = outputs["topk_ids"], outputs["topk_embeds"], outputs["logits"]
    if ids.shape != (1, k):
        return {"passed": False, "detail": f"topk_ids shape {ids.shape}, expected (1, {k})"}
    if rows.shape != (1, k, E.shape[1]):
        return {"passed": False, "detail": f"topk_embeds shape {rows.shape}, expected (1, {k}, {E.shape[1]})"}
    last = logits[0, -1]
    kth = np.sort(last)[::-1][k - 1]
    listed = [int(i) for i in ids[0]]
    if len(set(listed)) != k or any(last[i] < kth for i in listed):
        return {"passed": False, "detail": f"topk_ids {listed} are not the top-{k} of the last-position logits"}
    diff = float(np.max(np.abs(rows[0] - E[listed])))
    if diff > tol:
        return {"passed": False, "detail": f"topk_embeds rows differ from E[topk_ids] by {diff:.2e} (atol {tol})"}
    return {"passed": True, "detail": f"top-{k} ids exact; rows match E within {diff:.2e}"}


def check_stream_parts_sum(outputs: dict, tol: float) -> dict:
    """The parts (embedding, attention edits, MLP edits, each scaled by the
    final norm) must add up to what the lm_head multiplies."""
    for name in ("stream_parts", "last_hidden_state"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    parts, last = outputs["stream_parts"], outputs["last_hidden_state"]
    if parts.ndim != 3 or parts.shape[0] != 1 or parts.shape[2] != last.shape[2]:
        return {"passed": False, "detail": f"stream_parts shape {parts.shape}, expected (1, 1 + 2·layers, {last.shape[2]})"}
    diff = float(np.max(np.abs(parts[0].sum(axis=0) - last[0, -1])))
    return {"passed": diff <= tol, "detail": f"max|Σ parts − last_hidden|={diff:.2e} (atol {tol})"}


def check_votes_sum(outputs: dict, tol: float) -> dict:
    """parts · E[c] summed over the parts must reproduce each top-k logit —
    the identity the votes card shows."""
    for name in ("stream_parts", "topk_ids", "topk_embeds", "logits"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    votes = outputs["stream_parts"][0] @ outputs["topk_embeds"][0].T     # [parts, k]
    want = outputs["logits"][0, -1, outputs["topk_ids"][0]]
    diff = float(np.max(np.abs(votes.sum(axis=0) - want)))
    return {"passed": diff <= tol, "detail": f"max|Σ votes − logit|={diff:.2e} (atol {tol})"}


def _clipped(a: np.ndarray, clip: float | None) -> np.ndarray:
    return a if clip is None else np.minimum(a, clip)


def check_edit_ratios_parity(outputs: dict, expected: np.ndarray, tol: float, clip: float | None = None) -> dict:
    """edit_ratios [1, seq, layers, 2] against a reference [seq, layers, 2].
    With `clip`, both sides are capped first: the quantized gate, because q8
    moves the huge first-token MLP ratios (≈100) by ~10% and the card draws
    every ratio ≥ 1 alike."""
    if "edit_ratios" not in outputs:
        return {"passed": False, "detail": "graph has no edit_ratios output"}
    got = outputs["edit_ratios"]
    if got.shape != (1, *expected.shape):
        return {"passed": False, "detail": f"edit_ratios shape {got.shape}, expected (1, {', '.join(map(str, expected.shape))})"}
    diff = float(np.max(np.abs(_clipped(got[0], clip) - _clipped(expected, clip))))
    capped = f", clipped at {clip}" if clip is not None else ""
    return {"passed": diff <= tol, "detail": f"max|Δratio|={diff:.2e} (atol {tol}{capped})"}


def check_edit_cache_parity(single: dict, steps: list[dict], tol: float, clip: float | None = None) -> dict:
    """Step t of a cached run feeds one token; its edit_ratios row must equal
    row t of the single-shot run over the whole sequence."""
    if "edit_ratios" not in single or any("edit_ratios" not in s for s in steps):
        return {"passed": False, "detail": "graph has no edit_ratios output"}
    diff = max(float(np.max(np.abs(_clipped(s["edit_ratios"][0, 0], clip) - _clipped(single["edit_ratios"][0, t], clip))))
               for t, s in enumerate(steps))
    capped = f", clipped at {clip}" if clip is not None else ""
    return {"passed": diff <= tol, "detail": f"max|Δratio| step-by-step vs single-shot={diff:.2e} (atol {tol}{capped})"}


def torch_edit_ratios(ids: np.ndarray) -> np.ndarray:
    """Reference [seq, layers, 2] from the stock PyTorch model: forward hooks
    record each decoder layer's input x, its attention output a and its MLP
    output m; ratios are ‖a‖/‖x‖ and ‖m‖/‖x + a‖. Heavy (loads the model)."""
    import torch
    model = _stock_torch_model()
    rec: dict = {}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, l=l):
            rec[("x", l)] = (args[0] if args else kwargs["hidden_states"]).detach()
        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(
            lambda mod, args, out, l=l: rec.__setitem__(("a", l), out[0].detach())))
        hooks.append(layer.mlp.register_forward_hook(
            lambda mod, args, out, l=l: rec.__setitem__(("m", l), out.detach())))
    with torch.no_grad():
        model(torch.tensor(ids))
    for h in hooks:
        h.remove()
    layers = len(model.model.layers)
    out = np.zeros((ids.shape[1], layers, 2), dtype=np.float32)
    for l in range(layers):
        x, a, m = rec[("x", l)][0], rec[("a", l)][0], rec[("m", l)][0]
        out[:, l, 0] = (a.norm(dim=-1) / x.norm(dim=-1)).numpy()
        out[:, l, 1] = (m.norm(dim=-1) / (x + a).norm(dim=-1)).numpy()
    return out


def check_activations_parity(outputs: dict, ref: np.ndarray, abs_tol: float, rel_tol: float) -> dict:
    """mlp_activations [1, layers, neurons] against a reference [layers, neurons]:
    per layer max|Δ| ≤ abs_tol + rel_tol · max|ref of that layer|."""
    if "mlp_activations" not in outputs:
        return {"passed": False, "detail": "graph has no mlp_activations output"}
    got = outputs["mlp_activations"]
    if got.shape != (1, *ref.shape):
        return {"passed": False, "detail": f"mlp_activations shape {got.shape}, expected (1, {', '.join(map(str, ref.shape))})"}
    worst, worst_layer = 0.0, 0
    for l in range(ref.shape[0]):
        diff = float(np.max(np.abs(got[0, l] - ref[l])))
        allowed = abs_tol + rel_tol * float(np.max(np.abs(ref[l])))
        if diff > allowed:
            return {"passed": False, "detail": f"layer {l}: max|Δ|={diff:.3e} > {allowed:.3e}"}
        if diff > worst:
            worst, worst_layer = diff, l
    return {"passed": True, "detail": f"max|Δ|={worst:.2e} (layer {worst_layer}; atol {abs_tol} + {rel_tol}·max|ref|)"}


def check_inv_rms_parity(outputs: dict, ref_inv: float, rtol: float) -> dict:
    if "final_norm_inv_rms" not in outputs:
        return {"passed": False, "detail": "graph has no final_norm_inv_rms output"}
    got = float(outputs["final_norm_inv_rms"].reshape(-1)[0])
    rel = abs(got - ref_inv) / abs(ref_inv)
    return {"passed": rel <= rtol, "detail": f"1/rms {got:.6f} vs {ref_inv:.6f} (relative {rel:.2e}, rtol {rtol})"}


def check_neuron_votes_sum(outputs: dict, ndir: Path, k: int = 10) -> dict:
    """The card's identity: per layer and top-k candidate c, Σ_i a_i · (W_i · (g ⊙ E_c)) · 1/rms,
    with W_i dequantized from neurons/, equals the MLP vote stream_parts[1 + L + l] · E_c
    within the #15 tolerance 0.05 + 1 %."""
    for name in ("mlp_activations", "final_norm_inv_rms", "stream_parts", "topk_embeds"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    m = json.loads((ndir / "manifest.json").read_text())
    n, d, lay = m["neurons"], m["dims"], m["layout"]
    g = np.array(m["finalNormWeight"], dtype=np.float32)
    acts = outputs["mlp_activations"][0]
    layers = acts.shape[0]
    inv = float(outputs["final_norm_inv_rms"].reshape(-1)[0])
    rows = outputs["topk_embeds"][0][:k]
    parts = outputs["stream_parts"][0]
    worst = 0.0
    for l in range(layers):
        raw = (ndir / f"layer-{l:02d}.bin").read_bytes()
        scales = np.frombuffer(raw[lay["scales"]:lay["columns"]], dtype="<f4")
        q = np.frombuffer(raw[lay["columns"]:lay["labelIds"]], dtype=np.int8).reshape(n, d).astype(np.float32)
        cols = q * scales[:, None]                                         # [neurons, dims]
        for c, row in enumerate(rows):
            got = float(np.sum(acts[l] * (cols @ (g * row))) * inv)
            want = float(parts[1 + layers + l] @ row)
            gap = abs(got - want)
            if gap > 0.05 + 0.01 * abs(want):
                return {"passed": False, "detail": f"layer {l} candidate {c}: neuron votes sum to {got:.3f}, MLP vote is {want:.3f}"}
            worst = max(worst, gap)
    return {"passed": True, "detail": f"{layers} layers × {len(rows)} candidates, max gap {worst:.3e} (tol 0.05 + 1 %)"}


def check_argmax_parity(cases: list[tuple[np.ndarray, np.ndarray]], max_delta: float = 1.5) -> dict:
    """(ours, stock) last-position logits per prompt: same argmax everywhere and max|Δ| ≤ max_delta."""
    same = sum(int(np.argmax(o) == np.argmax(s)) for o, s in cases)
    worst = max(float(np.max(np.abs(o - s))) for o, s in cases)
    return {"passed": same == len(cases) and worst <= max_delta,
            "detail": f"same argmax {same}/{len(cases)}, max|Δlogit| {worst:.2f} (limit {max_delta})"}


def check_steps_parity(outputs: dict, ref: dict, abs_tol: float, rel_tol: float) -> dict:
    """Each step output [1, layers, n] against the reference [layers, n]:
    per layer max|Δ| ≤ abs_tol + rel_tol · max|ref of that layer|."""
    for name in STEP_OUTPUTS:
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    worst = (0.0, "", 0)
    for name in STEP_OUTPUTS:
        got, want = outputs[name], ref[name]
        if got.shape != (1, *want.shape):
            return {"passed": False, "detail": f"{name} shape {got.shape}, expected (1, {', '.join(map(str, want.shape))})"}
        for l in range(want.shape[0]):
            diff = float(np.max(np.abs(got[0, l] - want[l])))
            allowed = abs_tol + rel_tol * float(np.max(np.abs(want[l])))
            if diff > allowed:
                return {"passed": False, "detail": f"{name} layer {l}: max|Δ|={diff:.3e} > {allowed:.3e}"}
            worst = max(worst, (diff, name, l))
    return {"passed": True, "detail": f"max|Δ|={worst[0]:.2e} ({worst[1]} layer {worst[2]})"}


def check_silu_product(outputs: dict, rtol: float = 1e-3, atol: float = 1e-4) -> dict:
    """SiLU(mlp_gate) · mlp_up reproduces mlp_activations: the card's steps 3 and 4 are the graph's own."""
    for name in ("mlp_gate", "mlp_up", "mlp_activations"):
        if name not in outputs:
            return {"passed": False, "detail": f"graph has no {name} output"}
    g, u, a = outputs["mlp_gate"], outputs["mlp_up"], outputs["mlp_activations"]
    h = g / (1 + np.exp(-g)) * u
    ok = np.allclose(h, a, rtol=rtol, atol=atol)
    return {"passed": bool(ok), "detail": f"max|SiLU(g)·u − a| = {float(np.max(np.abs(h - a))):.2e}"}


def _last_logits(sess, model_dir: Path, prompt: str) -> np.ndarray:
    ids = _tokenize(model_dir, prompt)
    return sess.run(["logits"], _feeds(sess, ids))[0][0, -1]


def write_browser_reference(path: Path, model_dir: Path, stock_sess) -> None:
    """Token ids and stock fp32 last-position logits for ARGMAX_PROMPTS, for browser-check.html."""
    cases = []
    for prompt in ARGMAX_PROMPTS:
        ids = _tokenize(model_dir, prompt)
        logits = stock_sess.run(["logits"], _feeds(stock_sess, ids))[0][0, -1]
        cases.append({"prompt": prompt, "ids": ids[0].tolist(), "logits": [float(v) for v in logits]})
    path.write_text(json.dumps({"cases": cases}))


def check_neuron_files(model_dir: Path, texts: list[str], g: np.ndarray,
                       norms: list[np.ndarray], eps: float) -> dict:
    """neurons/ against its manifest (sizes, SHA-256), the layout, the stock
    final norm weight and the tokenizer's texts."""
    ndir = model_dir / "neurons"
    if not (ndir / "manifest.json").exists():
        return {"passed": False, "detail": "no neurons/ — run neurons first"}
    m = json.loads((ndir / "manifest.json").read_text())
    for name, f in m["files"].items():
        path = ndir / name
        if not path.exists() or path.stat().st_size != f["bytes"]:
            return {"passed": False, "detail": f"{name}: size differs from the manifest"}
        if _sha256(path) != f["sha256"]:
            return {"passed": False, "detail": f"{name}: SHA-256 differs from the manifest"}
    expected = {f"layer-{l:02d}.bin" for l in range(m["layers"])} | {"tokens.json"}
    if set(m["files"]) != expected:
        return {"passed": False, "detail": f"manifest lists {sorted(m['files'])}, expected {sorted(expected)}"}
    if any(m["files"][f"layer-{l:02d}.bin"]["bytes"] != m["layout"]["bytes"] for l in range(m["layers"])):
        return {"passed": False, "detail": "a layer file's size differs from layout.bytes"}
    if not np.allclose(np.array(m["finalNormWeight"]), g, atol=1e-6):
        return {"passed": False, "detail": "finalNormWeight differs from the stock final norm weight"}
    if "revision" not in m or m.get("rmsNormEps") != eps:
        return {"passed": False, "detail": f"manifest revision {m.get('revision')!r} / rmsNormEps {m.get('rmsNormEps')} (expected a revision and {eps})"}
    lay = m["layout"]
    for l in range(m["layers"]):
        raw = (ndir / f"layer-{l:02d}.bin").read_bytes()
        if not np.allclose(np.frombuffer(raw[lay["normWeight"]:lay["bytes"]], dtype="<f4"), norms[l], atol=1e-6):
            return {"passed": False, "detail": f"layer-{l:02d}.bin: normWeight differs from the stock post-attention norm"}
    if json.loads((ndir / "tokens.json").read_text()) != texts:
        return {"passed": False, "detail": "tokens.json differs from the tokenizer's texts"}
    return {"passed": True, "detail": f"{m['layers']} layer files and tokens.json match the manifest"}


@functools.lru_cache(maxsize=1)
def _stock_torch_model():
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        STOCK_MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager").eval()


def torch_neuron_reference(ids: np.ndarray) -> tuple[np.ndarray, float]:
    """[layers, neurons] down_proj inputs at the last position and the final
    norm's 1/rms there, from forward pre-hooks on the stock PyTorch model."""
    import torch
    model = _stock_torch_model()
    rec: dict = {}
    hooks = [layer.mlp.down_proj.register_forward_pre_hook(
        lambda mod, args, l=l: rec.__setitem__(l, args[0][0, -1].detach()))
        for l, layer in enumerate(model.model.layers)]
    hooks.append(model.model.norm.register_forward_pre_hook(
        lambda mod, args: rec.__setitem__("norm", args[0][0, -1].detach())))
    with torch.no_grad():
        model(torch.tensor(ids))
    for h in hooks:
        h.remove()
    acts = np.stack([rec[l].numpy() for l in range(len(model.model.layers))]).astype(np.float32)
    x = rec["norm"].double()
    inv = float(1.0 / torch.sqrt((x ** 2).mean() + model.config.rms_norm_eps))
    return acts, inv


def torch_step_reference(ids: np.ndarray) -> dict:
    """Last-position gate, up, MLP input and MLP output per layer, from hooks on the stock PyTorch model."""
    import torch
    model = _stock_torch_model()
    rec: dict = {}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        hooks += [
            layer.mlp.gate_proj.register_forward_hook(lambda m, a, o, l=l: rec.__setitem__(("mlp_gate", l), o[0, -1].detach())),
            layer.mlp.up_proj.register_forward_hook(lambda m, a, o, l=l: rec.__setitem__(("mlp_up", l), o[0, -1].detach())),
            layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a, l=l: rec.__setitem__(("mlp_in", l), a[0][0, -1].detach())),
            layer.mlp.register_forward_hook(lambda m, a, o, l=l: rec.__setitem__(("mlp_out", l), o[0, -1].detach())),
        ]
    with torch.no_grad():
        model(torch.tensor(ids))
    for h in hooks:
        h.remove()
    n = len(model.model.layers)
    return {k: np.stack([rec[(k, l)].numpy() for l in range(n)]).astype(np.float32) for k in STEP_OUTPUTS}


def check_geometry_files(model_dir: Path, E: np.ndarray, texts: list[str], sample: int = 32) -> dict:
    """Spot-check geometry/ against the stock matrix: file sizes match the
    manifest, every sampled token's listed neighbour ids are distinct, each
    equals the exact top-k set (or, for a uint8 tie, is within 1/255 of the
    k-th best exact cosine), the stored similarities are non-increasing, and
    tokens.json matches the tokenizer."""
    gdir = model_dir / "geometry"
    if not (gdir / "manifest.json").exists():
        return {"passed": False, "detail": "no geometry/ — run geometry first"}
    m = json.loads((gdir / "manifest.json").read_text())
    for name, size in m["files"].items():
        actual = (gdir / name).stat().st_size if (gdir / name).exists() else -1
        if actual != size:
            return {"passed": False, "detail": f"{name}: {actual} bytes, manifest says {size}"}
    vocab, k = m["vocabSize"], m["k"]
    if vocab != E.shape[0] or len(texts) != vocab:
        return {"passed": False, "detail": f"vocab {vocab} vs stock {E.shape[0]} / {len(texts)} texts"}
    raw = (gdir / "neighbors.bin").read_bytes()
    ids = np.frombuffer(raw[: vocab * k * 2], dtype="<u2").reshape(vocab, k)
    sims = np.frombuffer(raw[vocab * k * 2:], dtype=np.uint8).reshape(vocab, k)
    tokens = json.loads((gdir / "tokens.json").read_text())
    U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    for t in rng.choice(vocab, size=min(sample, vocab), replace=False):
        if tokens[t] != texts[t]:
            return {"passed": False, "detail": f"tokens.json[{t}]={tokens[t]!r} != tokenizer {texts[t]!r}"}
        for j in ids[t]:
            if j >= vocab:
                return {"passed": False, "detail": f"token {t}: neighbour id {j} outside vocabulary"}
        s = U[t] @ U.T
        s[t] = -np.inf
        listed = [int(j) for j in ids[t]]
        if len(set(listed)) != len(listed):
            return {"passed": False, "detail": f"token {t}: listed neighbour ids are not distinct"}
        exact_order = np.argsort(-s, kind="stable")
        exact_top = exact_order[:k]
        exact_set = {int(j) for j in exact_top}
        kth_sim = float(s[exact_top[-1]])
        for j in listed:
            if j not in exact_set and s[j] < kth_sim - 1 / 255:
                return {"passed": False,
                        "detail": f"token {t}: neighbour {j} (sim {s[j]:.4f}) is not among the exact "
                                  f"top-{k} (k-th best {kth_sim:.4f}) and not a uint8 tie"}
        row_sims = [int(x) for x in sims[t]]
        if any(row_sims[i] < row_sims[i + 1] for i in range(len(row_sims) - 1)):
            return {"passed": False, "detail": f"token {t}: stored similarities are not non-increasing"}
    return {"passed": True, "detail": f"{min(sample, vocab)} sampled tokens match exact neighbours and texts"}


def run(args) -> int:
    from huggingface_hub import hf_hub_download

    model_dir = Path(args.model_dir)
    onnx_dir = model_dir / "onnx"
    if not onnx_dir.exists() or not list(onnx_dir.glob("*.onnx")):
        print(f"no .onnx files found in {onnx_dir} — run export first")
        return 1

    checks: dict[str, dict] = {}

    stock_path = Path(hf_hub_download(STOCK_MODEL_ID, "onnx/model.onnx"))
    stock = _session(stock_path)

    from tsumugi_export.stock import stock_embeddings, stock_token_texts
    E = stock_embeddings()
    texts = stock_token_texts(E.shape[0])

    ref_ids = _tokenize(model_dir, PROMPTS[0])
    ref_ratios = torch_edit_ratios(ref_ids)
    ref_acts, ref_inv = torch_neuron_reference(ref_ids)
    ref_steps = torch_step_reference(ref_ids)
    write_browser_reference(model_dir.parent / "browser-ref.json", model_dir, stock)

    for variant in ["model.onnx", "model_quantized.onnx", "model_q4.onnx", "model_fp16.onnx"]:
        path = model_dir / "onnx" / variant
        if not path.exists():
            continue
        quantized = variant != "model.onnx"
        try:
            sess = _session(path, browser=quantized)
        except Exception as err:  # a broken artifact is a FAIL row, never a crash
            checks[f"{variant}:loadable"] = {"passed": False, "detail": f"session load failed: {err}"}
            continue
        out_names = [o.name for o in sess.get_outputs()]
        attn_names = sorted(n for n in out_names if n.startswith("attentions."))

        # Greedy-continuation token-identity check (spec check 1). For the
        # quantized variants this is the PRIMARY parity verdict — see
        # _check_greedy_parity — so it's computed once per variant, ahead of
        # the per-prompt logits-parity checks below that consult it.
        greedy = _check_greedy_parity(sess, stock, model_dir)
        checks[f"{variant}:greedy-parity"] = greedy
        checks[f"{variant}:argmax-parity"] = check_argmax_parity(
            [(_last_logits(sess, model_dir, p), _last_logits(stock, model_dir, p)) for p in ARGMAX_PROMPTS])

        for prompt in PROMPTS:
            ids = _tokenize(model_dir, prompt)
            ours = dict(zip(out_names, sess.run(None, _feeds(sess, ids))))
            theirs_names = [o.name for o in stock.get_outputs()]
            theirs = dict(zip(theirs_names, stock.run(None, _feeds(stock, ids))))

            tol = 1e-4 if variant == "model.onnx" else TOL
            diff = float(np.max(np.abs(ours["logits"][0, -1] - theirs["logits"][0, -1])))
            key = f"{variant}:logits-parity:{prompt[:12]}"
            if variant == "model.onnx":
                # fp32-vs-fp32: logits should match tightly, no excuse for drift.
                checks[key] = {"passed": diff < tol, "detail": f"max|Δlogit|={diff:.2e} (tol {tol})"}
            else:
                # Quantization moves individual logits by design — record the
                # magnitude for visibility, but gate on greedy token identity
                # (checked once per variant above) instead of this tolerance.
                checks[key] = {
                    "passed": greedy["passed"],
                    "detail": f"max|Δlogit|={diff:.2e} (informational for quantized variants; "
                              f"verdict via {variant}:greedy-parity)",
                }

            ok, detail = True, "all rows row-stochastic and causal"
            for name in attn_names:
                a = ours[name][0]  # [heads, q, kv]
                sums = a.sum(axis=-1)
                if not np.allclose(sums, 1.0, atol=1e-3):
                    ok, detail = False, f"{name}: row sums off (max err {np.max(np.abs(sums-1)):.2e})"
                    break
                q = a.shape[1]
                upper = np.triu(np.ones((q, a.shape[2])), k=1)[None, ...]
                if float(np.max(a * upper[..., :a.shape[2]])) > 1e-5:
                    ok, detail = False, f"{name}: mass on future positions"
                    break
            checks[f"{variant}:attn-wellformed:{prompt[:12]}"] = {"passed": ok, "detail": detail}

        checks[f"{variant}:has-attentions"] = {
            "passed": len(attn_names) > 0, "detail": f"{len(attn_names)} attention outputs"}
        checks[f"{variant}:has-cache"] = {
            "passed": any(n.startswith("present") for n in out_names),
            "detail": "present.* outputs present"}

        ci_tol = 1e-3 if variant == "model.onnx" else TOL
        checks[f"{variant}:cache-integrity"] = _check_cache_integrity(
            sess, model_dir, ci_tol, strict=(variant == "model.onnx"))

        # one forward on PROMPTS[0] serves the embedding-row and readout checks
        ro_ids = _tokenize(model_dir, PROMPTS[0])
        ro_out = dict(zip(out_names, sess.run(None, _feeds(sess, ro_ids))))
        pe_tol = 1e-3 if variant in ("model.onnx", "model_fp16.onnx") else TOL
        checks[f"{variant}:inputs-embeds-parity"] = check_inputs_embeds_parity(ro_out, ro_ids, E, pe_tol)
        lh_tol = 1e-3 if variant == "model.onnx" else TOL
        checks[f"{variant}:last-hidden-parity"] = check_last_hidden_parity(ro_out, E, lh_tol)
        checks[f"{variant}:topk-parity"] = check_topk_parity(ro_out, E, TOPK, 1e-3)
        fp32 = variant == "model.onnx"
        # fp32 proves the wiring; q8 runs the same added nodes on quantized
        # matmuls, so its gate only has to catch gross errors (see _clipped)
        ratio_tol, ratio_clip = (1e-3, None) if fp32 else (0.25, 2.0)
        checks[f"{variant}:edit-ratios-parity"] = check_edit_ratios_parity(ro_out, ref_ratios, ratio_tol, ratio_clip)
        checks[f"{variant}:stream-parts-sum"] = check_stream_parts_sum(ro_out, 1e-3)
        checks[f"{variant}:votes-sum"] = check_votes_sum(ro_out, 1e-3 if fp32 else 1e-2)
        checks[f"{variant}:edit-cache-parity"] = check_edit_cache_parity(
            ro_out, _drive_cached_stepwise(sess, ro_ids), ratio_tol, ratio_clip)
        # fp32 proves the wiring tightly; q8 is weight-only 8-bit (measured worst
        # layer ≈ 9 % of the layer's peak in the spike), so its gate only catches a wrong tensor
        act_abs, act_rel, inv_rtol = (1e-3, 1e-3, 1e-4) if fp32 else (0.5, 0.25, 0.1)
        checks[f"{variant}:activations-parity"] = check_activations_parity(ro_out, ref_acts, act_abs, act_rel)
        checks[f"{variant}:inv-rms-parity"] = check_inv_rms_parity(ro_out, ref_inv, inv_rtol)
        step_abs, step_rel = (1e-3, 1e-3) if fp32 else (0.5, 0.25)
        checks[f"{variant}:steps-parity"] = check_steps_parity(ro_out, ref_steps, step_abs, step_rel)
        checks[f"{variant}:silu-product"] = check_silu_product(ro_out)
        if (model_dir / "neurons" / "manifest.json").exists():
            # v6 computes on fp32 activations (weight-only 8-bit), so neuron votes from the
            # int8 asset must add up to the MLP votes here too (#19)
            checks[f"{variant}:neuron-votes-sum"] = check_neuron_votes_sum(ro_out, model_dir / "neurons")

    # A≡B equivalence when the operator exported both variants
    nc = model_dir.parent / "model-nocache" / "onnx" / "model.onnx"
    if nc.exists():
        # run cached graph step by step and compare each new row with the
        # no-cache full matrix — implementation mirrors the checks above and
        # reports max row deviation per layer
        checks["a-equiv-b"] = _check_a_equiv_b(model_dir, nc)
    else:
        # Not "passed": True — an untested Approach-A hypothesis must not
        # look like a pass to callers that gate on `passed` (publish.py's
        # upload gate in particular). See is_blocking() in publish.py.
        checks["a-equiv-b"] = {
            "passed": False, "skipped": True,
            "detail": "SKIPPED (no no-cache export present)",
        }

    checks["geometry"] = check_geometry_files(model_dir, E, texts)
    from transformers import AutoConfig
    from tsumugi_export.stock import stock_final_norm, stock_mlp_norms
    checks["neurons"] = check_neuron_files(
        model_dir, texts, stock_final_norm(), stock_mlp_norms(),
        AutoConfig.from_pretrained(STOCK_MODEL_ID).rms_norm_eps)

    passed = all(c["passed"] for c in checks.values())
    report = {
        "checks": checks,
        "artifacts": {
            str(p.relative_to(model_dir)): _sha256(p)
            for p in [*sorted((model_dir / "onnx").glob("*.onnx")),
                      *(sorted((model_dir / "geometry").glob("*")) if (model_dir / "geometry").exists() else []),
                      *(sorted((model_dir / "neurons").glob("*")) if (model_dir / "neurons").exists() else [])]
        },
    }
    (model_dir / "validation-report.json").write_text(json.dumps(report, indent=2))

    width = max(len(k) for k in checks)
    for k, c in sorted(checks.items()):
        label = "SKIP" if c.get("skipped") else ("PASS" if c["passed"] else "FAIL")
        print(f"{label}  {k.ljust(width)}  {c['detail']}")
    print("VERDICT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1
