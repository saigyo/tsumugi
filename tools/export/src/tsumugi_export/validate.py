"""Validation gate. Runs the exported graphs in onnxruntime and checks:
1. logits parity with the stock export (greedy continuation + last logits)
2. attention rows row-stochastic and causal
3. A≡B: cached incremental rows equal no-cache full-matrix rows (only when
   both variants are present in out/; otherwise reported as skipped)
4. cache integrity: present.* outputs exist and multi-step cached logits
   match single-shot full-context logits
5. inputs_embeds parity with the stock embedding table
6. geometry/ spot-check against exact neighbours and the tokenizer
Writes validation-report.json with artifact hashes; publish refuses to
upload without a passing report."""
import hashlib
import json
from pathlib import Path

import numpy as np

PROMPTS = [
    "The cat sat on the mat because it was tired",
    "one two three one two three one",     # repeats → induction rows exist
]
TOL = 2e-2  # fp32-vs-quantized logit tolerance; fp32-vs-fp32 uses 1e-4


def _session(path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


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


def _check_inputs_embeds_parity(sess, model_dir: Path, E: np.ndarray, tol: float) -> dict:
    """inputs_embeds must be exactly the embedding-table rows of the fed ids
    (within quantization noise for the int8 variant)."""
    out_names = [o.name for o in sess.get_outputs()]
    if "inputs_embeds" not in out_names:
        return {"passed": False, "detail": "graph has no inputs_embeds output"}
    ids = _tokenize(model_dir, PROMPTS[0])
    outputs = dict(zip(out_names, sess.run(None, _feeds(sess, ids))))
    got = outputs["inputs_embeds"]
    expected = E[ids[0]][None, :, :]
    if got.shape != expected.shape:
        return {"passed": False, "detail": f"inputs_embeds shape {got.shape}, expected {expected.shape}"}
    diff = float(np.max(np.abs(got - expected)))
    return {"passed": diff <= tol, "detail": f"max|Δembed|={diff:.2e} (atol {tol})"}


def check_geometry_files(model_dir: Path, E: np.ndarray, texts: list[str], sample: int = 32) -> dict:
    """Spot-check geometry/ against the stock matrix: file sizes match the
    manifest, every listed neighbour of a sampled token is within uint8
    resolution of the exact k-th best cosine, the top similarity round-trips,
    and tokens.json matches the tokenizer."""
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
        kth = np.sort(s)[-k]
        if any(s[j] < kth - 1 / 255 for j in ids[t]):
            return {"passed": False, "detail": f"token {t}: listed neighbour below the exact k-th best"}
        if abs(np.rint(max(s[ids[t][0]], 0) * 255) - sims[t][0]) > 1:
            return {"passed": False, "detail": f"token {t}: top similarity does not round-trip"}
    return {"passed": True, "detail": f"{min(sample, vocab)} sampled tokens match exact neighbours and texts"}


def run(args) -> int:
    from huggingface_hub import hf_hub_download
    from tsumugi_export import STOCK_MODEL_ID

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

    for variant in ["model.onnx", "model_quantized.onnx", "model_q4.onnx", "model_fp16.onnx"]:
        path = model_dir / "onnx" / variant
        if not path.exists():
            continue
        try:
            sess = _session(path)
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

        pe_tol = 1e-3 if variant in ("model.onnx", "model_fp16.onnx") else TOL
        checks[f"{variant}:inputs-embeds-parity"] = _check_inputs_embeds_parity(sess, model_dir, E, pe_tol)

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

    passed = all(c["passed"] for c in checks.values())
    report = {
        "checks": checks,
        "artifacts": {
            str(p.relative_to(model_dir)): _sha256(p)
            for p in [*sorted((model_dir / "onnx").glob("*.onnx")),
                      *(sorted((model_dir / "geometry").glob("*")) if (model_dir / "geometry").exists() else [])]
        },
    }
    (model_dir / "validation-report.json").write_text(json.dumps(report, indent=2))

    width = max(len(k) for k in checks)
    for k, c in sorted(checks.items()):
        label = "SKIP" if c.get("skipped") else ("PASS" if c["passed"] else "FAIL")
        print(f"{label}  {k.ljust(width)}  {c['detail']}")
    print("VERDICT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1
