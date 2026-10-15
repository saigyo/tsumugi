"""Neuron asset for the "Inside one MLP" card (#16 spec, section 2): per layer
the down-projection columns as int8 with one scale per column, and per neuron
and sign the five tokens its column points at most plus three cosine
thresholds; the decoded text of every id; a manifest with the layout, the
final norm weight and checksums."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tsumugi_export import STOCK_MODEL_ID

LABELS = 5
FRACTIONS = (0.001, 0.01, 0.1)
LABEL_METHOD = "cosine between ±W_down[:, i] and g ⊙ E[t] over the whole vocabulary, fp32 weights"


def layer_layout(neurons: int, dims: int, labels: int = LABELS, fractions: tuple = FRACTIONS) -> dict:
    columns = neurons * 4
    label_ids = columns + neurons * dims
    thresholds = label_ids + neurons * 2 * labels * 2
    return {"scales": 0, "columns": columns, "labelIds": label_ids, "thresholds": thresholds,
            "bytes": thresholds + neurons * 2 * len(fractions) * 4}


def build_layer(W: np.ndarray, g: np.ndarray, E: np.ndarray, labels: int = LABELS,
                fractions: tuple = FRACTIONS, block: int = 256) -> bytes:
    """W is down_proj.weight [dims, neurons]: column i is what neuron i writes,
    scaled by its activation."""
    vocab = E.shape[0]
    if vocab > 65535:
        raise ValueError("label ids are uint16")
    cols = np.ascontiguousarray(W.T, dtype=np.float32)                     # [neurons, dims]
    neurons = cols.shape[0]
    scales = np.abs(cols).max(axis=1) / 127
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(cols / scales[:, None]), -127, 127).astype(np.int8)
    R = (E * g[None, :]).astype(np.float32)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    U = cols / np.maximum(np.linalg.norm(cols, axis=1, keepdims=True), 1e-12)
    ranks = [max(1, int(np.ceil(f * vocab))) for f in fractions]
    unique = sorted(set(ranks))                                            # np.partition wants distinct kth
    ids = np.empty((neurons, 2, labels), dtype="<u2")
    thr = np.empty((neurons, 2, len(fractions)), dtype="<f4")
    for start in range(0, neurons, block):                                 # [block, vocab] cosines at a time
        S = U[start:start + block] @ R.T
        for s, M in enumerate((S, -S)):
            top = np.argpartition(-M, labels - 1, axis=1)[:, :labels]
            order = np.argsort(-np.take_along_axis(M, top, axis=1), axis=1, kind="stable")
            ids[start:start + block, s] = np.take_along_axis(top, order, axis=1)
            part = -np.partition(-M, [r - 1 for r in unique], axis=1)
            thr[start:start + block, s] = part[:, [r - 1 for r in ranks]]
    return scales.astype("<f4").tobytes() + q.tobytes() + ids.tobytes() + thr.tobytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_neurons(W_layers: list[np.ndarray], g: np.ndarray, E: np.ndarray, texts: list[str]) -> dict:
    dims, neurons = W_layers[0].shape
    vocab = E.shape[0]
    if len(texts) != vocab:
        raise ValueError(f"{len(texts)} texts for a vocabulary of {vocab}")
    layout = layer_layout(neurons, dims)
    layers = {f"layer-{l:02d}.bin": build_layer(W, g, E) for l, W in enumerate(W_layers)}
    for name, data in layers.items():
        if len(data) != layout["bytes"]:
            raise ValueError(f"{name}: {len(data)} bytes, layout says {layout['bytes']}")
    tokens = json.dumps(texts, ensure_ascii=False).encode("utf-8")
    files = {name: {"bytes": len(data), "sha256": _sha256(data)} for name, data in layers.items()}
    files["tokens.json"] = {"bytes": len(tokens), "sha256": _sha256(tokens)}
    manifest = {
        "modelId": STOCK_MODEL_ID, "layers": len(W_layers), "neurons": int(neurons), "dims": int(dims),
        "vocabSize": int(vocab), "labelsPerSign": LABELS, "fractions": list(FRACTIONS),
        "labelMethod": LABEL_METHOD, "layout": layout,
        "finalNormWeight": [float(v) for v in g], "files": files,
    }
    return {"manifest": manifest, "layers": layers, "tokens": tokens}


def write_neurons(out_dir: Path, built: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(built["manifest"], indent=2))
    for name, data in built["layers"].items():
        (out_dir / name).write_bytes(data)
    (out_dir / "tokens.json").write_bytes(built["tokens"])


def run(args) -> int:
    from tsumugi_export.stock import stock_embeddings, stock_final_norm, stock_mlp_down, stock_token_texts
    E = stock_embeddings()
    out = Path(args.model_dir) / "neurons"
    write_neurons(out, build_neurons(stock_mlp_down(), stock_final_norm(), E, stock_token_texts(E.shape[0])))
    m = json.loads((out / "manifest.json").read_text())
    print(f"wrote neurons → {out} ({m['layers']} layers × {m['neurons']} neurons, {m['layout']['bytes']} bytes each)")
    return 0
