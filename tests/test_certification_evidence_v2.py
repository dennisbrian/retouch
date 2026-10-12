"""Pure contract tests for :mod:`retouch.certification_evidence_v2`."""

from __future__ import annotations

import copy
import hashlib
import json

from retouch.certification_evidence_v2 import (
    build_automatic_quality_evidence,
    build_corpus_evidence,
    build_face_aware_evidence,
    build_final_manifest,
    build_human_review_evidence,
    build_render_artifact_evidence,
    build_runtime_doctor_evidence,
    canonical_json,
    canonical_sha256,
    certification_gate,
    seal_final_manifest,
    sha256_file,
    verify_sealed_manifest,
)


def _write_render(tmp_path, name="render.png"):
    output = tmp_path / name
    output.write_bytes(b"rendered bytes")
    metadata = {"icc_profile": "sRGB", "exif": {"orientation": 1}}

    def decoder(path):
        assert path == output
        return {
            "decodable": True,
            "decoder_backend": "mock",
            "dimensions": [640, 480],
            "dtype": "uint16",
            "metadata_present": True,
            "metadata": metadata,
        }

    return output, metadata, decoder


def _complete_evidence(tmp_path):
    output, metadata, decoder = _write_render(tmp_path)
    runtime = build_runtime_doctor_evidence({
        "status": "ok",
        "detector": {"status": "ok", "backend": "legacy"},
        "onnx": {"available_providers": ["CPUExecutionProvider"]},
        "path": "/private/user/retouch",
        "api_token": "must-not-be-stored",
    })
    render = build_render_artifact_evidence(
        output,
        expected_recipe="natural",
        actual_recipe="natural",
        expected_dimensions=[640, 480],
        expected_dtype="uint16",
        expected_metadata=metadata,
        metadata=metadata,
        decoder=decoder,
        expected_sha256=sha256_file(output),
    )
    face = build_face_aware_evidence(
        requested_mode="full",
        detector_probe={
            "probed": True,
            "probe_succeeded": True,
            "detector_initialized": True,
            "backend": "legacy",
        },
        expected_face_count=1,
        detected_face_count=1,
        landmarks=[[{"x": 0.5, "y": 0.5}]],
        parser_backend="BiSeNet",
        actual_provider="CPUExecutionProvider",
        provider_verified=True,
    )
    quality = build_automatic_quality_evidence(
        {
            "texture_retention": {"value": 0.94, "flagged": False},
            "halo_rate": {"value": 0.01, "flagged": False},
        },
        metric_version="metrics-2026-01",
        threshold_version="thresholds-pilot-1",
        thresholds={"texture_retention": {"minimum": 0.90}},
        roi_provenance={"skin": {"source": "bisenet", "face_index": 0}},
        detector_failures=[],
        outcome="approved",
    )
    corpus = build_corpus_evidence(
        [
            {"asset_id": "pilot-1", "sha256": "1" * 64, "tags": ["skin-tone-a"], "tags_validated": True, "split": "pilot"},
            {"asset_id": "holdout-1", "sha256": "2" * 64, "tags": ["lighting-high-key"], "tags_validated": True, "split": "holdout"},
        ],
        consent_reference="consent-batch-2026-01",
        required_tags=["skin-tone-a", "lighting-high-key"],
    )
    human = build_human_review_evidence([
        {"item_id": "holdout-1", "recipe": "natural", "reviewer_id": "r1", "blinded": True, "decision": "approved", "confidence": 0.9, "defect_labels": []},
        {"item_id": "holdout-1", "recipe": "natural", "reviewer_id": "r2", "blinded": True, "decision": "approved", "confidence": 0.8, "defect_labels": []},
    ])
    return runtime, render, face, quality, corpus, human


def test_canonical_json_and_file_hash_are_stable(tmp_path):
    assert canonical_json({"b": 2, "a": [True, "x"]}) == '{"a":[true,"x"],"b":2}'
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"retouch-evidence")
    assert sha256_file(payload) == hashlib.sha256(b"retouch-evidence").hexdigest()


def test_runtime_doctor_snapshot_is_sanitized_and_hashed():
    evidence = build_runtime_doctor_evidence({
        "status": "ok",
        "path": "/Users/example/private",
        "executable": "/usr/bin/python",
        "api_token": "secret",
        "detector": {"backend": "legacy", "status": "ok"},
    })

    assert "path" not in evidence["snapshot"]
    assert "executable" not in evidence["snapshot"]
    assert "api_token" not in evidence["snapshot"]
    assert evidence["snapshot_sha256"] == canonical_sha256(evidence["snapshot"])
    json.dumps(evidence)


def test_render_evidence_requires_real_observations_and_retains_metadata(tmp_path):
    output, metadata, decoder = _write_render(tmp_path)
    evidence = build_render_artifact_evidence(
        output,
        expected_recipe="natural",
        actual_recipe="natural",
        expected_dimensions=[640, 480],
        expected_dtype="uint16",
        expected_metadata=metadata,
        metadata=metadata,
        decoder=decoder,
    )

    assert evidence["render_completed"] is True
    assert evidence["decode"]["decodable"] is True
    assert evidence["recipe"]["matches"] is True
    assert evidence["dimensions"]["matches"] is True
    assert evidence["dtype"]["matches"] is True
    assert evidence["metadata"]["observed"] == metadata
    assert len(evidence["artifact"]["sha256"]) == 64
    json.dumps(evidence)


def test_render_evidence_rejects_recipe_mismatch_and_corrupt_decode(tmp_path):
    output, metadata, decoder = _write_render(tmp_path)
    mismatch = build_render_artifact_evidence(
        output,
        expected_recipe="natural",
        actual_recipe="dramatic",
        expected_dimensions=[640, 480],
        expected_dtype="uint16",
        expected_metadata=metadata,
        metadata=metadata,
        decoder=decoder,
    )
    assert mismatch["render_completed"] is False
    assert "recipe_identity" in mismatch["failed_checks"]

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    corrupt_evidence = build_render_artifact_evidence(
        corrupt,
        expected_recipe="natural",
        actual_recipe="natural",
        expected_dimensions=[640, 480],
        expected_dtype="uint16",
        expected_metadata={},
        metadata={},
        decoder=lambda path: {"decodable": False, "error": "decode failed"},
    )
    assert corrupt_evidence["render_completed"] is False
    assert corrupt_evidence["decode"]["decodable"] is False


def test_face_aware_evidence_does_not_infer_from_requested_full_mode():
    static_only = build_face_aware_evidence(
        requested_mode="full",
        detector_probe={"available": True, "backend": "legacy"},
        expected_face_count=1,
        detected_face_count=1,
        landmarks=[[{"x": 0.1, "y": 0.1}]],
        parser_backend="BiSeNet",
        actual_provider="CPUExecutionProvider",
    )
    assert static_only["requested_full_mode"] is True
    assert static_only["face_aware"] is False
    assert static_only["mode"] == "unresolved"
    assert "probe_attempted" in static_only["missing_observations"]


def test_face_aware_evidence_requires_probe_counts_landmarks_parser_and_provider():
    evidence = build_face_aware_evidence(
        requested_mode="full",
        detector_probe={
            "probed": True,
            "probe_succeeded": True,
            "detector_initialized": True,
            "backend": "legacy",
        },
        expected_face_count=2,
        detected_face_count=1,
        landmarks=[[{"x": 0.1, "y": 0.1}]],
        parser_backend="BiSeNet",
        actual_provider="CPUExecutionProvider",
    )
    assert evidence["face_aware"] is False
    assert evidence["face_count_matches"] is False
    assert evidence["status"] == "failed"

    passed = build_face_aware_evidence(
        requested_mode="full",
        detector_probe={"probed": True, "probe_succeeded": True, "detector_initialized": True, "backend": "legacy"},
        expected_face_count=1,
        detected_face_count=1,
        landmarks=[[{"x": 0.1, "y": 0.1}]],
        parser_backend="BiSeNet",
        actual_provider="CPUExecutionProvider",
        provider_verified=True,
    )
    assert passed["face_aware"] is True
    assert passed["mode"] == "face-aware"


def test_quality_vector_retains_non_flagged_metrics_roi_and_failures():
    metrics = {
        "texture_retention": {"value": 0.92, "flagged": False},
        "color_drift": {"value": 0.02, "flagged": False},
    }
    evidence = build_automatic_quality_evidence(
        metrics,
        metric_version="metrics-v2",
        threshold_version="thresholds-v1",
        thresholds={"texture_retention": {"minimum": 0.9}},
        roi_provenance={"skin": {"mask_hash": "abc", "source": "parser"}},
        detector_failures=[],
        outcome="approved",
    )
    assert evidence["metric_vector"] == metrics
    assert evidence["flagged_metrics"] == []
    assert evidence["roi_provenance"]["skin"]["source"] == "parser"
    assert evidence["detector_failures"] == []
    assert evidence["automatic_pass"] is True

    unresolved = build_automatic_quality_evidence(
        metrics,
        metric_version="metrics-v2",
        threshold_version="thresholds-v1",
        thresholds={},
        roi_provenance={"skin": {"source": "parser"}},
        detector_failures=[{"code": "detector_timeout", "face_index": 0}],
        outcome="uncertain",
    )
    assert unresolved["metric_vector"] == metrics
    assert unresolved["detector_failures"][0]["code"] == "detector_timeout"
    assert unresolved["outcome"] == "uncertain"
    assert unresolved["automatic_pass"] is False


def test_corpus_evidence_rejects_duplicate_hashes_and_requires_validated_split_tags():
    invalid = build_corpus_evidence([
        {"asset_id": "a", "sha256": "a" * 64, "tags": ["tone"], "tags_validated": True, "split": "pilot"},
        {"asset_id": "b", "sha256": "a" * 64, "tags": ["lighting"], "tags_validated": False, "split": "holdout"},
    ], consent_reference="consent", required_tags=["tone", "lighting"])
    assert invalid["complete"] is False
    assert invalid["unique_asset_hashes"] is False
    assert "duplicate_sha256" in invalid["invalid_assets"][0]["reasons"] or invalid["duplicate_sha256"]

    valid = build_corpus_evidence([
        {"asset_id": "a", "sha256": "a" * 64, "tags": ["tone"], "tags_validated": True, "split": "pilot"},
        {"asset_id": "b", "sha256": "b" * 64, "tags": ["lighting"], "tags_validated": True, "split": "holdout"},
    ], consent_reference="consent", required_tags=["tone", "lighting"])
    assert valid["complete"] is True
    assert valid["split_counts"] == {"holdout": 1, "pilot": 1}


def test_human_review_requires_two_blinded_independent_reviews_and_preserves_uncertainty():
    accepted = build_human_review_evidence([
        {"reviewer_id": "r1", "blinded": True, "decision": "approved", "confidence": 0.9, "defect_labels": []},
        {"reviewer_id": "r2", "blinded": True, "decision": "approved", "confidence": 0.8, "defect_labels": []},
    ])
    assert accepted["human_accepted"] is True
    assert accepted["independent_reviewer_count"] == 2

    uncertain = build_human_review_evidence([
        {"reviewer_id": "r1", "blinded": True, "decision": "approved", "confidence": 0.9, "defect_labels": []},
        {"reviewer_id": "r2", "blinded": True, "decision": "uncertain", "confidence": 0.5, "defect_labels": ["halo"]},
    ])
    assert uncertain["human_accepted"] is False
    assert uncertain["third_reviewer_required"] is True
    assert uncertain["third_reviewer_present"] is False
    assert uncertain["outcome"] == "uncertain"

    resolved = build_human_review_evidence([
        {"reviewer_id": "r1", "blinded": True, "decision": "approved", "confidence": 0.9, "defect_labels": []},
        {"reviewer_id": "r2", "blinded": True, "decision": "rejected", "confidence": 0.7, "defect_labels": ["halo"]},
        {"reviewer_id": "r3", "blinded": True, "decision": "approved", "confidence": 0.8, "defect_labels": []},
    ])
    assert resolved["third_reviewer_present"] is True
    assert resolved["human_accepted"] is True


def test_final_gate_requires_all_prerequisites_and_detects_mutation(tmp_path):
    runtime, render, face, quality, corpus, human = _complete_evidence(tmp_path)
    sealed = build_final_manifest(
        runtime_doctor=runtime,
        render_artifacts=[render],
        face_aware=face,
        automatic_quality=quality,
        corpus=corpus,
        human_review=human,
        policy={"metric_version": "metrics-2026-01", "threshold_version": "thresholds-pilot-1"},
    )

    decision = certification_gate(sealed)
    assert decision["final_certified"] is True
    assert all(decision["prerequisites"].values())
    assert verify_sealed_manifest(sealed)["valid"] is True
    json.dumps(sealed)

    tampered = copy.deepcopy(sealed)
    tampered["manifest"]["evidence"]["face_aware"]["face_aware"] = False
    assert verify_sealed_manifest(tampered)["valid"] is False
    assert certification_gate(tampered)["final_certified"] is False


def test_final_gate_ignores_asserted_certified_flag_when_evidence_is_missing():
    sealed = seal_final_manifest({
        "schema_version": 2,
        "evidence": {},
        "final_certified": True,
    })
    decision = certification_gate(sealed)
    assert decision["manifest_valid"] is True
    assert decision["final_certified"] is False
    assert "render_artifacts_missing_or_invalid" in decision["reason_codes"]
