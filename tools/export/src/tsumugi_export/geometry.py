"""Vocabulary-geometry asset for the Embeddings card (M4 spec, Component 1):
exact top-k cosine neighbours over the full embedding space, a PCA-reduced
int8 copy of every row, the decoded text of every id, and a manifest."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tsumugi_export import STOCK_MODEL_ID

K = 12
PCA_DIMS = 64
FILES = ("neighbors.bin", "vectors.bin", "tokens.json")


def build_geometry(E: np.ndarray, texts: list[str], k: int = K, pca_dims: int = PCA_DIMS,
                   block: int = 2048) -> dict:
    vocab, dims = E.shape
    if len(texts) != vocab:
        raise ValueError(f"{len(texts)} texts for a vocabulary of {vocab}")
    if vocab > 65535:
        raise ValueError("neighbour ids are uint16")
    E = np.ascontiguousarray(E, dtype=np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    U = E / np.maximum(norms, 1e-12)

    ids = np.empty((vocab, k), dtype="<u2")
    sims = np.empty((vocab, k), dtype=np.uint8)
    for start in range(0, vocab, block):          # 49152² cosines at once is ~10 GB; block it
        S = U[start:start + block] @ U.T
        rows = np.arange(S.shape[0])
        S[rows, start + rows] = -np.inf           # a token is not its own neighbour
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-top_s, axis=1, kind="stable")
        ids[start:start + block] = np.take_along_axis(top, order, axis=1)
        sims[start:start + block] = np.rint(np.clip(np.take_along_axis(top_s, order, axis=1), 0, 1) * 255)

    mean = E.mean(axis=0)
    C = E - mean
    _, s, vt = np.linalg.svd(C, full_matrices=False)
    P = C @ vt[:pca_dims].T
    explained = float((s[:pca_dims] ** 2).sum() / (s ** 2).sum())
    scale = float(np.abs(P).max() / 127) or 1.0
    q = np.clip(np.rint(P / scale), -127, 127).astype(np.int8)

    neighbors = ids.tobytes() + sims.tobytes()
    vectors = q.tobytes()
    tokens = json.dumps(texts, ensure_ascii=False).encode("utf-8")
    manifest = {
        "modelId": STOCK_MODEL_ID,
        "vocabSize": int(vocab), "dims": int(dims), "k": int(k), "pcaDims": int(pca_dims),
        "scale": scale, "explainedVariance": explained,
        "sourceSha256": hashlib.sha256(E.tobytes()).hexdigest(),
        "files": {"neighbors.bin": len(neighbors), "vectors.bin": len(vectors), "tokens.json": len(tokens)},
    }
    return {"manifest": manifest, "neighbors": neighbors, "vectors": vectors, "tokens": tokens}


def write_geometry(out_dir: Path, built: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(built["manifest"], indent=2))
    (out_dir / "neighbors.bin").write_bytes(built["neighbors"])
    (out_dir / "vectors.bin").write_bytes(built["vectors"])
    (out_dir / "tokens.json").write_bytes(built["tokens"])


def run(args) -> int:
    from tsumugi_export.stock import stock_embeddings, stock_token_texts
    E = stock_embeddings()
    texts = stock_token_texts(E.shape[0])
    out = Path(args.model_dir) / "geometry"
    write_geometry(out, build_geometry(E, texts))
    m = json.loads((out / "manifest.json").read_text())
    print(f"wrote geometry → {out} (PCA-{m['pcaDims']} explains {m['explainedVariance']:.1%} of variance)")
    return 0
