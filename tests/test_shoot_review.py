"""Persistent, non-destructive Shoot Review Manifest tests."""

import csv
import json
import shutil
from pathlib import Path

from PIL import Image

from retouch.shoot_intelligence import BurstGroup, CullingCandidate, inspect_asset
from retouch.shoot_review import (
    FaceQualityEvidence,
    ShootReviewManifest,
    asset_id_for,
    build_review_manifest,
)


def _write_image(path: Path, value: int = 100) -> None:
    Image.new("RGB", (16, 12), (value, value, value)).save(path)


def test_asset_id_includes_relative_path_and_content_hash():
    assert asset_id_for("a.jpg", "hash") != asset_id_for("b.jpg", "hash")
    assert asset_id_for("a.jpg", "hash") != asset_id_for("a.jpg", "other")


def test_manifest_persists_evidence_uncertainty_and_human_override(tmp_path: Path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_image(first, 100)
    _write_image(second, 101)
    assets = [inspect_asset(first), inspect_asset(second)]
    group = BurstGroup(
        group_id="burst-0001",
        asset_paths=(str(first.resolve()), str(second.resolve())),
        confidence=0.8,
        reasons=("test grouping",),
        evidence={"count": 2},
    )
    candidates = [
        CullingCandidate(
            path=str(first.resolve()), rank=1, score=0.9,
            evidence={"scoring_policy": "test"},
        ),
        CullingCandidate(
            path=str(second.resolve()), rank=2, score=0.2,
            evidence={"scoring_policy": "test"},
        ),
    ]

    manifest = build_review_manifest(tmp_path, assets, [group], {group.group_id: candidates})
    assert manifest.assets
    first_record = next(item for item in manifest.assets.values() if item.relative_path == "first.jpg")
    assert first_record.decision == "hold"
    assert first_record.decision_origin == "automatic"
    assert first_record.culling_evidence["candidate"]["rank"] == 1
    assert "face_quality_unavailable" in first_record.uncertainty
    assert first_record.review_required is True

    manifest.set_human_review(
        first_record.asset_id,
        "select",
        rating=5,
        labels=["hero", "warm"],
        reviewer="qa",
        note="Eyes and expression checked by human.",
    )
    manifest.save()

    restored = ShootReviewManifest.load(ShootReviewManifest.default_path(tmp_path))
    restored_record = restored.assets[first_record.asset_id]
    assert restored_record.decision == "select"
    assert restored_record.decision_origin == "human"
    assert restored_record.rating == 5
    assert restored_record.labels == ["hero", "warm"]
    assert len(restored_record.override_history) == 1
    assert restored_record.review_required is False


def test_rescan_refreshes_automatic_evidence_without_overwriting_human_review(tmp_path: Path):
    source = tmp_path / "capture.jpg"
    _write_image(source)
    assets = [inspect_asset(source)]
    first = build_review_manifest(tmp_path, assets, [], {})
    record = next(iter(first.assets.values()))
    first.set_human_review(record.asset_id, "reject", rating=1, labels=["blink"])
    first.save()

    second = build_review_manifest(tmp_path, assets, [], {})
    refreshed = second.assets[record.asset_id]
    assert refreshed.decision == "reject"
    assert refreshed.decision_origin == "human"
    assert refreshed.rating == 1
    assert refreshed.override_history[0].decision == "reject"


def test_rename_preserves_asset_instance_and_human_review(tmp_path: Path):
    original = tmp_path / "original.jpg"
    renamed = tmp_path / "renamed.jpg"
    _write_image(original)
    first = build_review_manifest(tmp_path, [inspect_asset(original)], [], {})
    record = next(iter(first.assets.values()))
    first.set_human_review(record.asset_id, "select", rating=5, labels=["hero"])
    first.save()

    original.rename(renamed)
    second = build_review_manifest(tmp_path, [inspect_asset(renamed)], [], {})

    assert list(second.assets) == [record.asset_id]
    restored = second.assets[record.asset_id]
    assert restored.relative_path == "renamed.jpg"
    assert restored.decision == "select"
    assert restored.rating == 5
    assert restored.identity_history[-1]["kind"] == "rename"


def test_custom_manifest_path_keeps_selected_shoot_root(tmp_path: Path):
    source = tmp_path / "capture.jpg"
    _write_image(source)
    manifest_path = tmp_path / "shared-state" / "review.json"

    manifest = build_review_manifest(
        tmp_path, [inspect_asset(source)], [], {}, manifest_path=manifest_path,
    )

    assert manifest.project_root == str(tmp_path.resolve())
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["project_root"] == str(tmp_path.resolve())


def test_identical_copy_is_a_distinct_asset_instance(tmp_path: Path):
    original = tmp_path / "card-a" / "IMG_0001.jpg"
    copy_path = tmp_path / "card-b" / "IMG_0001.jpg"
    original.parent.mkdir()
    copy_path.parent.mkdir()
    _write_image(original)
    shutil.copyfile(original, copy_path)

    manifest = build_review_manifest(
        tmp_path,
        [inspect_asset(original), inspect_asset(copy_path)],
        [],
        {},
    )

    assert len(manifest.assets) == 2
    assert len({record.asset_id for record in manifest.assets.values()}) == 2


def test_content_change_invalidates_current_decision_but_keeps_history(tmp_path: Path):
    source = tmp_path / "capture.jpg"
    _write_image(source, 100)
    first = build_review_manifest(tmp_path, [inspect_asset(source)], [], {})
    record = next(iter(first.assets.values()))
    first.set_human_review(record.asset_id, "select", rating=4)
    first.save()

    _write_image(source, 220)
    second = build_review_manifest(tmp_path, [inspect_asset(source)], [], {})
    changed = second.assets[record.asset_id]

    assert changed.decision == "hold"
    assert changed.decision_origin == "automatic"
    assert changed.override_history[0].decision == "select"
    assert changed.identity_history[-1]["kind"] == "content_changed"


def test_manifest_json_and_csv_exports_are_non_destructive(tmp_path: Path):
    source = tmp_path / "capture.jpg"
    _write_image(source)
    manifest = build_review_manifest(tmp_path, [inspect_asset(source)], [], {})
    record = next(iter(manifest.assets.values()))
    manifest.set_human_review(record.asset_id, "select", rating=4, labels=["deliver"])

    json_path = tmp_path / "review.json"
    csv_path = tmp_path / "selection.csv"
    manifest.save(json_path)
    manifest.export_csv(csv_path, selected_only=True)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "retouch.shoot_review"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["decision"] == "select"
    assert rows[0]["rating"] == "4"
    assert rows[0]["manifest_schema_version"] == "2"
    assert rows[0]["content_id"] == rows[0]["source_sha256"]
    assert json.loads(rows[0]["culling_evidence_json"])["policy"]
    assert json.loads(rows[0]["face_evidence_json"]) == []
    assert json.loads(rows[0]["job_provenance_json"]) == {}
    assert json.loads(rows[0]["identity_history_json"]) == []
    assert source.is_file()


def test_csv_export_keeps_face_and_job_evidence_inspectable(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    evidence = FaceQualityEvidence(
        face_id="face-1",
        bbox=(0.1, 0.2, 0.4, 0.5),
        coverage=0.2,
        face_sharpness=3.5,
        left_eye_sharpness=1.2,
        right_eye_sharpness=1.1,
        eyes_open="uncertain",
        measurement_version="face-quality-v1",
        sharpness_method="contrast-normalized-tenengrad-v1",
        measurement_details={"face_bbox_pixels": [2, 3, 4, 5]},
        uncertainty=["blink_analysis_deferred"],
    )
    manifest = build_review_manifest(
        tmp_path,
        [inspect_asset(source)],
        [],
        {},
        face_evidence_by_path={str(source.resolve()): [evidence]},
    )
    record = next(iter(manifest.assets.values()))
    manifest.attach_job_provenance(record.asset_id, "preview", {
        "job_id": "job-1",
        "status": "done",
        "output_sha256": "abc",
    })

    row = next(csv.DictReader(manifest.export_csv_text().splitlines()))
    faces = json.loads(row["face_evidence_json"])
    jobs = json.loads(row["job_provenance_json"])
    assert faces[0]["face_id"] == "face-1"
    assert faces[0]["measurement_details"]["face_bbox_pixels"] == [2, 3, 4, 5]
    assert faces[0]["uncertainty"] == ["blink_analysis_deferred"]
    assert jobs["preview"]["output_sha256"] == "abc"
    assert record.faces[0].face_id == "face-1"
    assert source.is_file()


def test_face_quality_evidence_preserves_explicit_uncertainty():
    evidence = FaceQualityEvidence(
        face_id="face-0", coverage=0.42, detector_confidence=0.77,
        eyes_open="uncertain", uncertainty=["occluded_by_glasses"],
    )
    restored = FaceQualityEvidence.from_dict(evidence.to_dict())
    assert restored.eyes_open == "uncertain"
    assert restored.uncertainty == ["occluded_by_glasses"]
