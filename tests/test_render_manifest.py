"""Focused tests for the standalone pixel-free render manifest contract."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest

from retouch.render_manifest import (
    CONTRACT_NAME,
    SCHEMA_VERSION,
    RenderManifest,
    RenderManifestError,
    build_render_manifest,
    validate_render_manifest,
)


SOURCE_SHA = "1" * 64
OUTPUT_SHA = "2" * 64
SETTINGS_SHA = "3" * 64


def _manifest(**overrides):
    values = {
        "render_revision": 12,
        "settings_sha256": SETTINGS_SHA,
        "mode": "preview",
        "source": {
            "id": "shoot/portrait-001",
            "path": "/session/source/portrait.jpg",
            "sha256": SOURCE_SHA,
            "dimensions": {"width": 6240, "height": 4160, "channels": 3},
            "dtype": "uint8",
            "format": "JPEG",
        },
        "output": {
            "id": "render/portrait-001-preview",
            "uri": "/session/output/portrait-preview.webp",
            "sha256": OUTPUT_SHA,
            "size_bytes": 18234,
            "dimensions": {"width": 800, "height": 533, "channels": 3},
            "dtype": "uint8",
            "format": "WEBP",
        },
        "cache": {
            "status": "hit",
            "hit": True,
            "key": "cache-key-abc",
            "stages": {"decode": "hit", "face_detection": "miss"},
        },
        "color_context": {
            "input_profile": "Display P3",
            "input_profile_source": "embedded",
            "working_space": "sRGB",
            "output_profile": "sRGB",
            "transform_applied": True,
            "assumed_srgb": False,
        },
        "metadata_result": {
            "status": "preserved",
            "preserved_fields": ["EXIF", "ICC"],
            "dropped_fields": [],
            "warnings": [],
        },
        "face_count": 2,
        "timings_ms": {
            "total": 148.5,
            "decode": 12.0,
            "stages": {"face_detection": 20.25, "render": 95.0},
        },
        "qa": {
            "status": "observed",
            "metrics": {
                "texture_retention": 0.94,
                "clipping_fraction": 0.001,
            },
            "warnings": ["low_confidence_secondary_face"],
            "roi_provenance": {"skin": "face-parser-v1"},
            "detector_failures": [],
        },
        "safe_auto_decisions": [
            {
                "stage": "skin_smoothing",
                "action": "apply",
                "confidence": 0.92,
                "reason": "stable face mask",
            }
        ],
        "backend": {
            "engine": "retouch-engine",
            "detector": "mediapipe",
            "parser": "bisenet",
            "model_fingerprint": "model-set-v1",
        },
        "provider": "CPUExecutionProvider",
        "hashes": {
            "source_sha256": SOURCE_SHA,
            "output_sha256": OUTPUT_SHA,
        },
        "extensions": {"inspector_note": "pixel-free"},
    }
    values.update(overrides)
    return build_render_manifest(**values)


def test_completed_manifest_records_inspector_contract_without_pixels():
    manifest = _manifest()
    payload = manifest.to_dict()

    assert payload["contract"] == CONTRACT_NAME
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["render_revision"] == 12
    assert payload["settings_sha256"] == SETTINGS_SHA
    assert payload["mode"] == "preview"
    assert payload["source"]["sha256"] == SOURCE_SHA
    assert payload["output"]["dimensions"] == {"width": 800, "height": 533, "channels": 3}
    assert payload["output"]["dtype"] == "uint8"
    assert payload["cache"]["status"] == "hit"
    assert payload["color_context"]["working_space"] == "sRGB"
    assert payload["metadata_result"]["status"] == "preserved"
    assert payload["face_count"] == 2
    assert payload["timings_ms"]["stages"]["render"] == 95.0
    assert payload["qa"]["metrics"]["texture_retention"] == 0.94
    assert payload["safe_auto_decisions"][0]["action"] == "apply"
    assert payload["backend"]["detector"] == "mediapipe"
    assert payload["provider"] == "CPUExecutionProvider"
    assert payload["hashes"]["settings_sha256"] == SETTINGS_SHA
    assert payload["hashes"]["manifest_sha256"] == manifest.manifest_sha256
    assert json.loads(manifest.to_json())["contract"] == CONTRACT_NAME
    assert validate_render_manifest(payload)["valid"] is True


def test_manifest_is_detached_deepcopy_safe_and_round_trips():
    manifest = _manifest()
    payload = manifest.deepcopy_state()
    payload["qa"]["metrics"]["texture_retention"] = 0.1
    payload["source"]["dimensions"]["width"] = 1

    fresh = manifest.to_dict()
    assert fresh["qa"]["metrics"]["texture_retention"] == 0.94
    assert fresh["source"]["dimensions"]["width"] == 6240

    cloned = copy.deepcopy(manifest)
    assert cloned.to_dict() == manifest.to_dict()
    restored = RenderManifest.from_dict(manifest.to_dict())
    assert restored.to_dict() == manifest.to_dict()
    assert RenderManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()


def test_manifest_hash_is_stable_and_tampering_is_detected():
    first = _manifest().to_dict()
    reordered = dict(reversed(list(first.items())))
    assert RenderManifest.from_dict(reordered).manifest_sha256 == first["hashes"]["manifest_sha256"]

    tampered = copy.deepcopy(first)
    tampered["qa"]["metrics"]["texture_retention"] = 0.2
    report = validate_render_manifest(tampered)
    assert report["valid"] is False
    assert "manifest_sha256" in report["errors"][0]


def test_pixel_bearing_values_are_rejected():
    @dataclass
    class FakeArray:
        shape: tuple = (10, 10, 3)
        dtype: str = "uint8"

    with pytest.raises(RenderManifestError, match="array-like pixel data"):
        _manifest(qa={"preview": FakeArray()})
    with pytest.raises(RenderManifestError, match="byte buffer"):
        _manifest(metadata_result={"raw": b"pixels"})
    with pytest.raises(RenderManifestError, match="non-finite"):
        _manifest(timings_ms={"render": float("nan")})


def test_schema_rejects_invalid_identity_precision_mode_and_cache_values():
    with pytest.raises(RenderManifestError, match="completed output must include dimensions"):
        _manifest(output={"id": "output", "dtype": "uint8"})
    with pytest.raises(RenderManifestError, match="completed output must include dtype"):
        _manifest(output={"id": "output", "dimensions": {"width": 1, "height": 1}})
    with pytest.raises(RenderManifestError, match="mode must be one of"):
        _manifest(mode="certification")
    with pytest.raises(RenderManifestError, match="cache.status"):
        _manifest(cache={"status": "maybe"})
    with pytest.raises(RenderManifestError, match="settings_sha256"):
        _manifest(hashes={"settings_sha256": "4" * 64})


def test_full_alias_and_failed_lifecycle_are_supported_without_fake_output():
    full = _manifest(mode="export_full_quality")
    assert full.mode == "full"

    failed = _manifest(
        status="failed",
        output=None,
        error="detector unavailable",
        cache={"status": "bypassed", "hit": False},
    )
    payload = failed.to_dict()
    assert payload["status"] == "failed"
    assert payload["output"] is None
    assert payload["error"] == "detector unavailable"
    assert validate_render_manifest(payload)["valid"] is True

    with pytest.raises(RenderManifestError, match="failed manifests must include error"):
        _manifest(status="failed", output=None)


def test_validation_report_is_non_throwing_for_wrong_version_and_unknown_fields():
    invalid_version = _manifest().to_dict()
    invalid_version["schema_version"] = SCHEMA_VERSION + 1
    report = validate_render_manifest(invalid_version)
    assert report["valid"] is False
    assert "unsupported schema_version" in report["errors"][0]

    unknown = _manifest().to_dict()
    unknown["pixels"] = "must never be accepted"
    report = validate_render_manifest(unknown)
    assert report["valid"] is False
    assert "unknown manifest fields" in report["errors"][0]
