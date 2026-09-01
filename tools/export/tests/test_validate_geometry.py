import json

import numpy as np

from tsumugi_export.geometry import build_geometry, write_geometry
from tsumugi_export.validate import check_geometry_files


def make(tmp_path, vocab=40, dims=6):
    rng = np.random.default_rng(1)
    E = rng.normal(size=(vocab, dims)).astype(np.float32)
    texts = [f"t{i}" for i in range(vocab)]
    write_geometry(tmp_path / "geometry", build_geometry(E, texts, k=4, pca_dims=3, block=8))
    return E, texts


def test_geometry_check_passes_on_fresh_asset(tmp_path):
    E, texts = make(tmp_path)
    res = check_geometry_files(tmp_path, E, texts, sample=10)
    assert res["passed"], res["detail"]


def test_geometry_check_fails_on_corrupt_neighbors(tmp_path):
    E, texts = make(tmp_path)
    p = tmp_path / "geometry" / "neighbors.bin"
    raw = bytearray(p.read_bytes())
    raw[0] ^= 0xFF
    p.write_bytes(bytes(raw))
    assert not check_geometry_files(tmp_path, E, texts, sample=40)["passed"]


def test_geometry_check_fails_on_wrong_text(tmp_path):
    E, texts = make(tmp_path)
    p = tmp_path / "geometry" / "tokens.json"
    bad = json.loads(p.read_text())
    bad[3] = "wrong"
    p.write_text(json.dumps(bad))
    assert not check_geometry_files(tmp_path, E, texts, sample=40)["passed"]


def test_geometry_check_fails_when_missing(tmp_path):
    res = check_geometry_files(tmp_path, np.zeros((2, 2), np.float32), ["a", "b"])
    assert not res["passed"] and "geometry" in res["detail"]
