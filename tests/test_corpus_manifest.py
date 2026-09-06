from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from retouch.corpus_manifest import (
    CorpusManifestError,
    corpus_split_report,
    validate_corpus_coverage,
    validate_corpus_manifest,
    write_corpus_manifest,
)


def _asset_file(root: Path, name: str, content: bytes) -> tuple[Path, str]:
    path = root / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _probe(_path: Path):
    return {
        "probe_status": "ok",
        "decodable": True,
        "width": 1200,
        "height": 900,
        "format": "JPEG",
        "mode": "RGB",
    }


def _manifest(root: Path, *, names_and_contents=None):
    if names_and_contents is None:
        names_and_contents = [("portrait-a.jpg", b"portrait-a"), ("portrait-b.jpg", b"portrait-b")]
    paths = [_asset_file(root, name, content) for name, content in names_and_contents]
    assets = [
        {
            "asset_id": "asset-a",
            "path": paths[0][0].name,
            "sha256": paths[0][1],
            "split": "dev",
            "person_id": "person-01",
            "tags": ["glasses"],
            "strata": {
                "skin_tone": "medium",
                "lighting": "studio",
                "face_scale": "close",
                "pose": "frontal",
                "faces": "single",
            },
            "label_validation": {"status": "validated", "method": "human", "reference": "labels-a"},
            "metadata": {"decodable": True, "width": 1200, "height": 900},
        },
        {
            "asset_id": "asset-b",
            "path": paths[1][0].name,
            "sha256": paths[1][1],
            "split": "calibration",
            "person_id": "person-02",
            "tags": ["marks", "multi_image_shoot"],
            "strata": {
                "skin_tone": "deep",
                "lighting": "low_key",
                "face_scale": "medium",
                "pose": "three_quarter",
                "faces": "multiple",
            },
            "group_applicable": True,
            "group_id": "shoot-calibration-01",
            "label_validation": {"status": "validated", "method": "human", "reference": "labels-b"},
        },
    ]
    return {
        "schema_version": 3,
        "consent_reference": "consent/core-v3/participant-set-01",
        "assets": assets,
    }


def test_valid_dev_calibration_manifest_is_deterministic_and_json_safe(tmp_path):
    manifest = _manifest(tmp_path)

    first = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)
    second = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert first["valid"] is True
    assert first == second
    assert first["split_counts"] == {"dev": 1, "calibration": 1, "locked_test": 0}
    assert first["person_counts_by_split"] == {"dev": 1, "calibration": 1, "locked_test": 0}
    assert first["coverage"]["strata"]["skin_tone"] == {
        "deep": ["asset-b"],
        "medium": ["asset-a"],
    }
    assert first["assets"][0]["metadata"]["decodable"] is True
    assert first["assets"][0]["person_ids"] == ["person-01"]
    json.dumps(first, sort_keys=True)

    output = tmp_path / "canonical.json"
    written = write_corpus_manifest(output, manifest, root=tmp_path, image_probe=_probe)
    assert written["valid"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == written["canonical_manifest"]


def test_manifest_with_only_dev_split_still_validates(tmp_path):
    """v3 does not require calibration/locked_test to be nonempty (unlike
    v2's unconditional pilot_split_required/holdout_split_required) -- a
    manifest built incrementally, with every asset still in dev, must
    validate as-is."""
    manifest = _manifest(tmp_path)
    manifest["assets"][1]["split"] = "dev"
    manifest["assets"][1]["person_id"] = "person-01"

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is True
    assert report["split_counts"] == {"dev": 2, "calibration": 0, "locked_test": 0}


def test_required_split_minimums_is_a_policy_input_not_a_default(tmp_path):
    manifest = _manifest(tmp_path)

    report = validate_corpus_manifest(
        manifest,
        root=tmp_path,
        image_probe=_probe,
        required_split_minimums={"locked_test": 1},
    )

    assert report["valid"] is False
    assert "split_minimum_not_met" in {issue["code"] for issue in report["errors"]}

    passing = validate_corpus_manifest(
        manifest,
        root=tmp_path,
        image_probe=_probe,
        required_split_minimums={"dev": 1, "calibration": 1},
    )
    assert passing["valid"] is True


def test_filename_containing_every_stratum_never_supplies_labels(tmp_path):
    path, digest = _asset_file(
        tmp_path,
        "skin_tone_lighting_glasses_wig_hands_marks_high_key_low_key_group.jpg",
        b"not-an-image-but-a-real-file",
    )
    manifest = {
        "schema_version": 3,
        "consent_reference": "consent/test",
        "assets": [
            {
                "asset_id": "named-like-a-corpus",
                "path": path.name,
                "sha256": digest,
                "split": "dev",
                # Deliberately no person_id, tags, strata, or validation metadata.
            },
            {
                "asset_id": "calibration",
                "path": path.name,
                "sha256": digest,
                "split": "calibration",
                "person_id": "person-03",
                "tags": [],
                "strata": {},
                "label_validation": {"status": "validated", "method": "human"},
            },
        ],
    }

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is False
    codes = {issue["code"] for issue in report["errors"]}
    assert {"tags_required", "strata_required", "label_validation_required", "person_id_required"} <= codes
    assert report["coverage"]["tags"] == {}
    assert report["coverage"]["strata"] == {}


def test_duplicate_content_and_duplicate_asset_ids_are_rejected(tmp_path):
    first, digest = _asset_file(tmp_path, "first.jpg", b"same-content")
    second = tmp_path / "second.jpg"
    second.write_bytes(first.read_bytes())
    manifest = _manifest(tmp_path)
    manifest["assets"] = [
        {
            "asset_id": "duplicate",
            "path": first.name,
            "sha256": digest,
            "split": "dev",
            "person_id": "person-01",
            "tags": [],
            "strata": {"skin_tone": "medium", "lighting": "studio", "face_scale": "close", "pose": "frontal"},
            "label_validation": {"status": "validated", "method": "human"},
        },
        {
            "asset_id": "duplicate",
            "path": second.name,
            "sha256": digest,
            "split": "calibration",
            "person_id": "person-02",
            "tags": [],
            "strata": {"skin_tone": "deep", "lighting": "low_key", "face_scale": "medium", "pose": "profile"},
            "label_validation": {"status": "validated", "method": "human"},
        },
    ]

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is False
    codes = {issue["code"] for issue in report["errors"]}
    assert "duplicate_asset_id" in codes
    assert "duplicate_content_hash" in codes


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda m: m.pop("consent_reference"), "consent_reference_required"),
        (lambda m: m["assets"][0].update(split="pilot"), "split_invalid"),
        (lambda m: m["assets"][0]["tags"].append("invented_tag"), "unknown_tag"),
        (lambda m: m["assets"][0]["strata"].update(lighting="moonlight"), "unknown_stratum_value"),
        (lambda m: m["assets"][0].pop("person_id"), "person_id_required"),
    ],
)
def test_missing_consent_invalid_split_and_unknown_labels_are_rejected(tmp_path, change, expected):
    manifest = _manifest(tmp_path)
    change(manifest)

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is False
    assert expected in {issue["code"] for issue in report["errors"]}


def test_group_id_is_required_for_group_assets_and_cannot_cross_splits(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["assets"][0]["tags"] = ["group_portrait"]
    manifest["assets"][0]["group_id"] = "shoot-01"
    manifest["assets"][1]["group_id"] = "shoot-01"

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is False
    assert "group_crosses_split" in {issue["code"] for issue in report["errors"]}


def test_person_id_cannot_cross_splits(tmp_path):
    """The core subject-separation guarantee: the same person_id must not
    appear in two different splits, independent of group_id."""
    manifest = _manifest(tmp_path)
    manifest["assets"][1]["person_id"] = "person-01"  # same person as asset-a, different split

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is False
    codes = {issue["code"] for issue in report["errors"]}
    assert "person_crosses_split" in codes


def test_group_portrait_asset_can_declare_multiple_person_ids(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["assets"][1].pop("person_id", None)
    manifest["assets"][1]["person_ids"] = ["person-02", "person-03"]

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is True
    asset_b = next(asset for asset in report["assets"] if asset["asset_id"] == "asset-b")
    assert asset_b["person_ids"] == ["person-02", "person-03"]
    assert report["person_counts_by_split"]["calibration"] == 2


def test_identity_marks_and_provenance_fields_round_trip(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["assets"][0]["identity_marks"] = ["left_cheek_mole", "chin_scar"]
    manifest["assets"][0]["consent_reference"] = "consent/asset-a/2026-09-06"
    manifest["assets"][0]["source"] = "studio-shoot-2026-09-06"

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert report["valid"] is True
    asset_a = next(asset for asset in report["assets"] if asset["asset_id"] == "asset-a")
    assert asset_a["identity_marks"] == ["chin_scar", "left_cheek_mole"]
    assert asset_a["consent_reference"] == "consent/asset-a/2026-09-06"
    assert asset_a["source"] == "studio-shoot-2026-09-06"


def test_label_validator_callback_can_supply_validation_evidence(tmp_path):
    manifest = _manifest(tmp_path)
    for asset in manifest["assets"]:
        asset.pop("label_validation", None)
    seen = []

    def validator(asset):
        seen.append(asset["asset_id"])
        return {"valid": True, "method": "locked-label-service"}

    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe, asset_validator=validator)

    assert report["valid"] is True
    assert seen == ["asset-a", "asset-b"]
    assert all(asset["label_validation"]["method"] == "locked-label-service" for asset in report["assets"])


def test_explicit_coverage_validator_does_not_use_filenames(tmp_path):
    manifest = _manifest(tmp_path)
    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    coverage = validate_corpus_coverage(
        report,
        required_tags=["glasses", "wig"],
        required_strata={"lighting": ["studio", "low_key", "high_key"]},
    )

    assert coverage["valid"] is False
    missing = {issue["field"] for issue in coverage["errors"]}
    assert "tags.wig" in missing
    assert "strata.lighting.high_key" in missing


def test_corpus_split_report_from_validated_report(tmp_path):
    manifest = _manifest(tmp_path)
    report = validate_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    split_report = corpus_split_report(report)

    assert split_report["split_counts"] == {"dev": 1, "calibration": 1, "locked_test": 0}
    assert split_report["person_counts_by_split"] == {"dev": 1, "calibration": 1, "locked_test": 0}
    assert split_report["coverage_by_split"]["dev"]["tags"] == {"glasses": ["asset-a"]}
    assert split_report["coverage_by_split"]["locked_test"]["tags"] == {}


def test_corpus_split_report_from_raw_manifest_without_validating(tmp_path):
    manifest = _manifest(tmp_path)

    split_report = corpus_split_report(manifest)

    assert split_report["split_counts"] == {"dev": 1, "calibration": 1, "locked_test": 0}
    assert split_report["person_counts_by_split"] == {"dev": 1, "calibration": 1, "locked_test": 0}


def test_require_valid_manifest_exposes_machine_readable_report(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["assets"][0]["sha256"] = "0" * 64

    with pytest.raises(CorpusManifestError) as exc_info:
        from retouch.corpus_manifest import require_valid_corpus_manifest

        require_valid_corpus_manifest(manifest, root=tmp_path, image_probe=_probe)

    assert exc_info.value.report["valid"] is False
    assert "sha256_mismatch" in {issue["code"] for issue in exc_info.value.report["errors"]}
