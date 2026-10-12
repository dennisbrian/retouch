"""Pure tests for the GUI preview/full-export contract."""

from __future__ import annotations

import json

import pytest

from retouch.gui_render_modes import (
    MODE_EXPORT_ALL,
    MODE_EXPORT_FULL_QUALITY,
    MODE_RENDER_PREVIEW,
    RenderContractError,
    build_export_all,
    build_export_full_quality,
    build_render_preview,
    capture_settings_snapshot,
    compare_revisions,
    require_current_contract,
    validate_render_contract,
)


def _snapshot(revision=12):
    return capture_settings_snapshot(
        {
            "recipe": "natural",
            "smooth": 31,
            "fast": True,
            "nested": {"preserve_icc": True},
        },
        revision,
    )


def test_settings_snapshot_is_json_safe_copied_and_hashed():
    settings = {"smooth": [1, 2], "recipe": "natural"}
    snapshot = capture_settings_snapshot(settings, 4)
    settings["smooth"].append(99)

    assert snapshot["revision"] == 4
    assert snapshot["settings"]["smooth"] == [1, 2]
    assert len(snapshot["settings_sha256"]) == 64
    json.dumps(snapshot)


def test_preview_selects_first_image_fast_and_never_exports():
    contract = build_render_preview(
        ["first.jpg", "second.jpg"],
        _snapshot(12),
        draft_revision=12,
        preview_revision=11,
    )

    assert contract["mode"] == MODE_RENDER_PREVIEW
    assert contract["mode_label"] == "Render Preview"
    assert contract["source"]["selected_paths"] == ["first.jpg"]
    assert contract["source"]["first_image_only"] is True
    assert contract["render"]["fast"] is True
    assert contract["render"]["automatic_export"] is False
    assert contract["export_intent"]["requested"] is False
    assert contract["revision_labels"] == {
        "draft": "Draft revision 12",
        "preview": "Preview revision 11",
        "captured": "Captured settings revision 12",
        "render": "Rendering settings revision 12",
    }
    assert contract["validation"]["valid"] is True
    assert contract["safe_to_commit"] is True
    json.dumps(contract)


def test_full_quality_export_is_single_image_slow_and_metadata_preserving():
    contract = build_export_full_quality(
        "portrait.jpg",
        _snapshot(8),
        draft_revision=8,
    )

    assert contract["mode"] == MODE_EXPORT_FULL_QUALITY
    assert contract["render"]["fast"] is False
    assert contract["render"]["automatic_export"] is True
    assert contract["render"]["output"] == "full_quality_export"
    assert contract["export_intent"]["metadata_preserving"] is True
    assert contract["export_intent"]["writer_policy"] == "metadata_preserving"
    assert contract["export_intent"]["fast"] is False
    assert contract["validation"]["valid"] is True

    with pytest.raises(RenderContractError):
        build_export_full_quality(["one.jpg", "two.jpg"], _snapshot(8), 8)


def test_export_all_is_a_full_quality_batch_handoff():
    contract = build_export_all(
        ["one.jpg", "two.jpg", "three.jpg"],
        _snapshot(20),
        draft_revision=20,
        batch_job_id="batch-20",
    )

    assert contract["mode"] == MODE_EXPORT_ALL
    assert contract["render"]["execute"] is False
    assert contract["render"]["fast"] is False
    assert contract["batch_handoff"]["required"] is True
    assert contract["batch_handoff"]["workflow"] == "Batch"
    assert contract["batch_handoff"]["source_paths"] == ["one.jpg", "two.jpg", "three.jpg"]
    assert contract["batch_handoff"]["job_id"] == "batch-20"
    assert contract["export_intent"]["metadata_preserving"] is True
    assert contract["validation"]["valid"] is True
    json.dumps(contract)


def test_stale_render_is_explicit_and_cannot_be_committed():
    contract = build_render_preview(
        ["portrait.jpg"],
        _snapshot(12),
        draft_revision=13,
        preview_revision=11,
    )

    assert contract["stale"] is True
    assert contract["revisions"]["render_vs_draft"]["status"] == "stale"
    assert contract["validation"]["valid"] is True
    assert contract["validation"]["safe_to_commit"] is False
    assert contract["safe_to_commit"] is False
    with pytest.raises(RenderContractError):
        require_current_contract(contract)


def test_revision_comparison_fails_closed_for_missing_and_future_values():
    missing = compare_revisions(None, 3)
    assert missing["valid"] is False
    assert missing["stale"] is True
    assert missing["status"] == "unresolved"

    future = compare_revisions(4, 3)
    assert future["valid"] is False
    assert future["stale"] is True
    assert future["status"] == "invalid_order"


def test_validation_rejects_tampered_snapshot_and_contradictory_mode():
    contract = build_render_preview(["portrait.jpg"], _snapshot(2), 2)
    tampered = json.loads(json.dumps(contract))
    tampered["settings"]["snapshot"]["settings"]["smooth"] = 999
    report = validate_render_contract(tampered)
    assert report["valid"] is False
    assert "settings_snapshot_hash_mismatch" in report["errors"]

    contradictory = json.loads(json.dumps(contract))
    contradictory["render"]["fast"] = False
    report = validate_render_contract(contradictory)
    assert report["valid"] is False
    assert "render.fast" in report["errors"]


def test_builders_reject_invalid_inputs_before_claiming_a_render():
    with pytest.raises(RenderContractError):
        build_render_preview([], _snapshot(1), 1)
    with pytest.raises(RenderContractError):
        build_render_preview(["portrait.jpg"], {"revision": 1}, 1)
    with pytest.raises(RenderContractError):
        capture_settings_snapshot({"bad": float("nan")}, 1)
    with pytest.raises(RenderContractError):
        build_export_all(["portrait.jpg"], _snapshot(1), 1, batch_job_id=" ")


def test_require_current_contract_returns_a_json_safe_copy():
    contract = build_export_full_quality("portrait.jpg", _snapshot(5), 5)
    accepted = require_current_contract(contract)
    assert accepted == contract
    assert accepted is not contract
    json.dumps(accepted)
