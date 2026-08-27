from tsumugi_export.publish import is_blocking


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
