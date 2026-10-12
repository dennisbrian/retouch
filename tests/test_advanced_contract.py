"""Deterministic Advanced Retouch base/session evidence tests."""

import copy

import numpy as np
import pytest

from retouch.advanced_contract import (
    BASE_KIND_PROCESSED,
    BASE_KIND_SOURCE,
    AdvancedContractError,
    build_base_contract,
    build_session_payload,
    compare_base_contracts,
    delivery_decision,
    parse_session_payload,
    replay_result_matches,
    source_file_matches,
)


def _image(value=80):
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _full_render_evidence():
    return {
        "source_path": None,
        "render_mode": "export_full_quality",
        "render_revision": 4,
        "settings_sha256": "a" * 64,
        "native": {"width": 32, "height": 24},
        "output": {"width": 32, "height": 24},
    }


def test_same_dimensions_with_different_pixels_refuse_base_match():
    expected = build_base_contract(_image(80), kind=BASE_KIND_SOURCE)
    actual = build_base_contract(_image(81), kind=BASE_KIND_SOURCE)

    comparison = compare_base_contracts(expected, actual)

    assert comparison["matches"] is False
    assert "pixel_sha256_mismatch" in comparison["reasons"]


def test_processed_preview_is_editable_evidence_but_not_delivery_eligible():
    preview = build_base_contract(
        _image(),
        kind=BASE_KIND_PROCESSED,
        render_evidence={
            **_full_render_evidence(),
            "render_mode": "render_preview",
        },
    )
    native = build_base_contract(
        _image(),
        kind=BASE_KIND_PROCESSED,
        render_evidence=_full_render_evidence(),
    )

    assert preview["render"]["effective_detail"] == "proxy"
    assert delivery_decision(preview)["allowed"] is False
    assert native["render"]["effective_detail"] == "native"
    assert delivery_decision(native)["allowed"] is True


def test_v2_session_hashes_base_edits_and_replay_result():
    base_image = _image(80)
    result_image = _image(90)
    base = build_base_contract(base_image, kind=BASE_KIND_SOURCE)
    edits = [{"mode": "Adjust", "operation": "exposure", "strength": 10}]
    payload = build_session_payload(
        edits,
        base,
        history_state={"base_cursor": 0, "cursor": 1},
        result_rgb=result_image,
    )

    parsed = parse_session_payload(payload)
    assert parsed["version"] == 2
    assert replay_result_matches(parsed, result_image)["matches"] is True
    assert replay_result_matches(parsed, _image(91))["matches"] is False

    tampered = copy.deepcopy(payload)
    tampered["edits"][0]["strength"] = 99
    with pytest.raises(AdvancedContractError, match="edit hash"):
        parse_session_payload(tampered)


def test_truncated_history_cannot_be_saved_as_replayable_session():
    base = build_base_contract(_image(), kind=BASE_KIND_SOURCE)
    with pytest.raises(AdvancedContractError, match="history was truncated"):
        build_session_payload(
            [{"mode": "Adjust"}],
            base,
            history_state={"base_cursor": 1, "cursor": 2},
            result_rgb=_image(),
        )


def test_legacy_edit_only_payload_is_pending_unverified():
    parsed = parse_session_payload({"version": 1, "edits": [{"mode": "Adjust"}]})
    assert parsed["legacy_unverified"] is True
    assert parsed["base"] is None


def test_source_metadata_identity_requires_exact_file_bytes(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"original source bytes")
    base = build_base_contract(_image(), kind=BASE_KIND_SOURCE, source_path=source)

    assert source_file_matches(base, source) is True
    source.write_bytes(b"changed source bytes")
    assert source_file_matches(base, source) is False
