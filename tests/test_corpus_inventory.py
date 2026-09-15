from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qa"))

import corpus_inventory  # noqa: E402


def _write_pilot_manifest(root: Path, name: str, *, person_id: str, source: str, skin_tone: str) -> Path:
    pilot_dir = root / f"pilot_{name}"
    pilot_dir.mkdir()
    image_path = pilot_dir / f"{name}.jpg"
    content = f"content-{name}".encode()
    image_path.write_bytes(content)
    manifest = {
        "schema_version": 3,
        "consent_reference": f"consent for {source}",
        "assets": [
            {
                "asset_id": f"{source}_{name}",
                "path": image_path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "split": "dev",
                "person_ids": [person_id],
                "source": source,
                "tags": [],
                "strata": {
                    "skin_tone": skin_tone,
                    "lighting": "daylight",
                    "face_scale": "close",
                    "pose": "frontal",
                },
                "label_validation": {"status": "validated", "method": "owner_manual_review"},
            }
        ],
    }
    manifest_path = pilot_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _probe(_path):
    return {"probe_status": "ok", "decodable": True, "width": 100, "height": 100, "format": "JPEG", "mode": "RGB"}


def test_merge_combines_assets_and_preserves_per_asset_consent(tmp_path):
    manifest_paths = [
        _write_pilot_manifest(tmp_path, "aaa1", person_id="subject_1", source="shoot_a", skin_tone="light"),
        _write_pilot_manifest(tmp_path, "bbb2", person_id="subject_2", source="shoot_b", skin_tone="very_light"),
    ]
    merged, sources = corpus_inventory.merge_manifests(manifest_paths, merged_root=tmp_path)

    assert len(sources) == 2
    assert len(merged["assets"]) == 2
    consents = {asset["consent_reference"] for asset in merged["assets"]}
    assert consents == {"consent for shoot_a", "consent for shoot_b"}


def test_missing_strata_table_flags_unset_dimensions():
    coverage = {
        "tags": {},
        "strata": {
            "skin_tone": {"light": ["a"], "very_light": ["b"]},
            "lighting": {"daylight": ["a", "b"]},
            "face_scale": {"close": ["a", "b"]},
            "pose": {"frontal": ["a", "b"]},
        },
    }
    gaps = corpus_inventory.missing_strata_table(coverage)

    assert "occlusion" in gaps["dimensions_with_zero_coverage"]
    assert "texture" in gaps["dimensions_with_zero_coverage"]
    assert "deep" in gaps["dimensions_missing_some_values"]["skin_tone"]
    assert "very_deep" in gaps["dimensions_missing_some_values"]["skin_tone"]


def test_candidate_groupings_reports_source_field_not_group_id():
    assets = [
        {"asset_id": "shoot_a_1", "source": "shoot_a"},
        {"asset_id": "shoot_a_2", "source": "shoot_a"},
        {"asset_id": "shoot_b_1", "source": "shoot_b"},
    ]
    groups = corpus_inventory.candidate_groupings(assets)

    assert groups["shoot_a"] == ["shoot_a_1", "shoot_a_2"]
    assert groups["shoot_b"] == ["shoot_b_1"]


def test_merge_and_validate_end_to_end_is_valid(tmp_path):
    manifest_paths = [
        _write_pilot_manifest(tmp_path, "ccc3", person_id="subject_3", source="shoot_c", skin_tone="light"),
        _write_pilot_manifest(tmp_path, "ddd4", person_id="subject_4", source="shoot_d", skin_tone="light"),
    ]
    merged, sources = corpus_inventory.merge_manifests(manifest_paths, merged_root=tmp_path)
    assert len(sources) == 2

    report = corpus_inventory.validate_corpus_manifest(merged, root=tmp_path, image_probe=_probe)

    assert report["valid"], report["errors"]
    assert report["asset_count"] == 2
