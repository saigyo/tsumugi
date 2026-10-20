import argparse
import subprocess
import sys
from pathlib import Path

from tsumugi_export.publish import (
    MODEL_CARD, artifact_path, assert_manifest_matches_tag, assert_tag_is_new, is_blocking, tag_name,
)


def test_passed_check_never_blocks():
    assert is_blocking({"passed": True, "detail": "ok"}) is False
    assert is_blocking({"passed": True, "detail": "ok"}, allow_skipped_equivalence=True) is False


def test_failed_non_skipped_check_always_blocks():
    check = {"passed": False, "detail": "boom"}
    assert is_blocking(check) is True
    assert is_blocking(check, allow_skipped_equivalence=True) is True


def test_skipped_check_blocks_by_default():
    check = {"passed": False, "skipped": True, "detail": "SKIPPED (no no-cache export present)"}
    assert is_blocking(check) is True


def test_skipped_check_allowed_only_with_explicit_flag():
    check = {"passed": False, "skipped": True, "detail": "SKIPPED (no no-cache export present)"}
    assert is_blocking(check, allow_skipped_equivalence=True) is False


def test_artifact_paths_are_model_dir_relative():
    assert artifact_path(Path("out/m"), "onnx/model.onnx") == Path("out/m/onnx/model.onnx")
    assert artifact_path(Path("out/m"), "geometry/neighbors.bin") == Path("out/m/geometry/neighbors.bin")


def test_model_card_documents_the_readout_outputs():
    for name in ("last_hidden_state", "topk_ids", "topk_embeds", "[batch, 16, 576]"):
        assert name in MODEL_CARD


def test_model_card_names_the_published_tag():
    card = MODEL_CARD.format(stock="s/m", last=29, tag="v9")
    assert "`v9`" in card
    assert "v3" not in card          # the tag is not hardcoded to this release


def test_tag_name_rejects_empty():
    assert tag_name(argparse.Namespace(tag="v3")) == "v3"
    for bad in ("", "  "):
        try:
            tag_name(argparse.Namespace(tag=bad))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_existing_tag_is_refused():
    assert_tag_is_new(["v1", "v2"], "v3")          # new tag: no error
    try:
        assert_tag_is_new(["v1", "v2", "v3"], "v3")
    except ValueError as err:
        assert "v3" in str(err) and "v4" in str(err)   # the message proposes the next name
    else:
        raise AssertionError("expected ValueError")


def test_cli_requires_tag_for_publish():
    r = subprocess.run([sys.executable, "-m", "tsumugi_export", "publish"], capture_output=True, text=True)
    assert r.returncode == 2 and "--tag" in r.stderr


def test_a_neuron_manifest_for_another_tag_is_refused():
    assert_manifest_matches_tag({"revision": "v6"}, "v6")
    for manifest in ({"revision": "v5"}, {}):
        try:
            assert_manifest_matches_tag(manifest, "v6")
        except ValueError as err:
            assert "v6" in str(err)
        else:
            raise AssertionError("expected ValueError")
