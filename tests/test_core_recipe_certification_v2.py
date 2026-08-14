"""Pure fixture tests for the Core Certification Evidence v2 runner path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import zlib
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "recipes" / "core_recipe_certification.py"
_SPEC = importlib.util.spec_from_file_location("core_recipe_certification_v2_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)


def _write_png(path: Path, variant: int = 0) -> None:
    first = bytes((0x20 + variant, 0x40, 0x60))
    second = bytes((0x80, 0xA0 + variant, 0xC0))
    pixels = b"\x00" + first * 2 + b"\x00" + second * 2
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _runtime_snapshot() -> Dict[str, Any]:
    snapshot = {
        "schema_version": 1,
        "status": "ok",
        "project_root": "/private/user/project",
        "detector": {"status": "ok", "backend": "mediapipe_legacy"},
    }
    return {
        "schema_version": 1,
        "probe_detector": True,
        "probe_parser": True,
        "snapshot": snapshot,
        "sha256": core._sha256_json(snapshot),
        "status": "ok",
    }


def _metric() -> Dict[str, Any]:
    return {
        "metric": "texture_retention",
        "version": "metric-v2-test",
        "threshold_version": "threshold-v2-test",
        "value": 0.98,
        "threshold": 0.80,
        "status": "pass",
        "flagged": False,
        "available": True,
        "roi": {
            "source": "face_landmarks",
            "face_ids": ["face-0"],
            "pixel_count": 4,
            "mask_hash": "0" * 64,
        },
        "details": {"raw_score": 0.98},
    }


def _face_evidence() -> Dict[str, Any]:
    return {
        "detector": {
            "probed": True,
            "initialized": True,
            "backend": "mediapipe_legacy",
        },
        "expected_face_count": 1,
        "detected_face_count": 1,
        "faces": [{"id": "face-0", "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0}]}],
        "parser": {"status": "ok", "backend": "bisenet_onnx"},
        "provider": {"name": "CPUExecutionProvider", "verified": True},
    }


def _write_sweep_fixture(
    case_output: Path,
    recipes: list[str],
    *,
    missing: Optional[str] = None,
) -> None:
    case_output.mkdir(parents=True, exist_ok=True)
    outputs = []
    face = _face_evidence()
    for index, recipe in enumerate(recipes):
        output = case_output / f"{index:02d}_{recipe}.png"
        _write_png(output)
        row: Dict[str, Any] = {
            "recipe": recipe,
            "output": output.name,
            "status": "done",
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "render_evidence": {"width": 2, "height": 2, "dtype": "uint8"},
        }
        if missing != "face":
            face = _face_evidence()
            row.update({
                "face_evidence": {
                    "face_count": face["detected_face_count"],
                    "faces": face["faces"],
                },
                "detector_evidence": face["detector"],
                "parser_evidence": face["parser"],
                "provider_evidence": {
                    "active": [face["provider"]["name"]],
                    "actual_provider": face["provider"]["name"],
                    "verified": True,
                },
            })
        if missing != "quality":
            quality = {
                "version": 2,
                "metric_version": "metric-v2-test",
                "threshold_version": "threshold-v2-test",
                "complete": True,
                "coverage_complete": True,
                "metrics": [_metric()],
                "observations": [_metric()],
                "unavailable": [],
            }
            row.update({
                "qa_evidence": quality,
                "quality_evidence": quality,
                "automatic_quality": quality,
            })
        outputs.append(row)
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "global_only": False,
        "outputs": outputs,
        "contact_sheet": "contact_sheet.jpg",
    }
    (case_output / "contact_sheet.jpg").write_bytes(b"contact-sheet-fixture")
    (case_output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_corpus_manifest(
    path: Path,
    source: Path,
    *,
    tags: Optional[list[str]] = None,
    case_id: str = "manifest-asset-1",
    incomplete: bool = False,
) -> None:
    pilot = source.parent / "pilot-fixture.png"
    _write_png(pilot, variant=1)
    all_tags = tags if tags is not None else ["group_portrait"]
    pilot_asset = {
        "asset_id": "pilot-fixture-1",
        "path": pilot.name,
        "sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(),
        "split": "pilot",
        "tags": ["glasses"],
        "strata": {
            "skin_tone": "medium",
            "lighting": "studio",
            "face_scale": "close",
            "pose": "frontal",
        },
        "label_validation": {"status": "validated", "method": "fixture-labels"},
        "expected_face_count": 1,
    }
    holdout_asset = {
        "asset_id": case_id,
        "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": "holdout",
        "tags": all_tags,
        "strata": {
            "skin_tone": "deep",
            "lighting": "low_key",
            "face_scale": "medium",
            "pose": "three_quarter",
            "faces": "multiple",
        },
        "label_validation": {"status": "validated", "method": "fixture-labels"},
        "group_id": "fixture-shoot-1",
        "expected_face_count": 1,
    }
    if incomplete:
        # A filename containing every old v1 alias must not supply any of
        # these explicit labels. Keep the second asset so the strict validator
        # still reports its pilot/holdout contract rather than a legacy alias.
        holdout_asset.pop("tags")
        holdout_asset.pop("strata")
        holdout_asset.pop("label_validation")
    payload = {
        "schema_version": 2,
        "consent_reference": "consent-fixture-1",
        "assets": [pilot_asset, holdout_asset],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path, source: Path, corpus: Path, case_name: str = "portrait") -> Namespace:
    return Namespace(
        case=[f"{case_name}={source}"],
        output=str(tmp_path / "certification"),
        corpus_manifest=str(corpus),
        max_dim=None,
        global_only=False,
        allow_incomplete_corpus=False,
        stop_on_fail=False,
    )


def _fake_subprocess(recipes: list[str], source: Path):
    def run(command, **kwargs):
        case_output = Path(command[command.index("-o") + 1])
        _write_sweep_fixture(case_output, recipes)

        class Completed:
            returncode = 0

        return Completed()

    return run


def test_v2_matrix_uses_manifest_tags_and_records_runtime_snapshot_hash(tmp_path, monkeypatch):
    source = tmp_path / "plain.png"
    _write_png(source)
    corpus = tmp_path / "corpus.json"
    _write_corpus_manifest(corpus, source)
    monkeypatch.setattr(core, "_runtime_doctor_snapshot", _runtime_snapshot)
    monkeypatch.setattr(core.subprocess, "run", _fake_subprocess(list(core.CORE_RECIPE_NAMES), source))

    result = core.run(_args(tmp_path, source, corpus))

    matrix = json.loads((tmp_path / "certification" / "matrix_manifest.json").read_text())
    assert result == 0
    assert matrix["version"] == 2
    assert matrix["schema_version"] == 2
    assert matrix["contract_version"] == "certification-evidence-v2"
    assert matrix["automatic_pass"] is True
    assert matrix["runtime_doctor"]["sha256"] == matrix["runtime_doctor_sha256"]
    assert matrix["runtime_doctor"]["snapshot"] == matrix["runtime_doctor_snapshot"]
    assert matrix["runtime_doctor_sha256"] == core._sha256_json(matrix["runtime_doctor_snapshot"])
    assert matrix["cases"][0]["per_image_evidence"]["outputs"]
    assert matrix["cases"][0]["face_evidence"]["detected_face_count"] == 1
    assert matrix["cases"][0]["quality_evidence"]["passed"] is True


def test_v2_filename_spoof_cannot_complete_manifest_coverage(tmp_path, monkeypatch):
    source = tmp_path / "ordinary.png"
    _write_png(source)
    corpus = tmp_path / "corpus.json"
    _write_corpus_manifest(corpus, source, incomplete=True)
    monkeypatch.setattr(core, "_runtime_doctor_snapshot", _runtime_snapshot)
    monkeypatch.setattr(core.subprocess, "run", _fake_subprocess(list(core.CORE_RECIPE_NAMES), source))
    args = _args(
        tmp_path,
        source,
        corpus,
        case_name="skin_tone_lighting_glasses_wig_hands_on_face_group_portrait",
    )
    args.allow_incomplete_corpus = True

    result = core.run(args)

    matrix = json.loads((tmp_path / "certification" / "matrix_manifest.json").read_text())
    assert result == 1
    assert matrix["corpus_complete"] is False
    assert matrix["automatic_pass"] is False
    assert matrix["corpus"]["valid"] is False
    assert matrix["corpus"]["coverage"]["strata"]
    assert matrix["cases"][0]["automatic_pass"] is True


def test_v2_missing_face_or_quality_evidence_fails_automatic_gate(tmp_path, monkeypatch):
    for missing in ("face", "quality"):
        case_root = tmp_path / missing
        source = case_root / "portrait.png"
        source.parent.mkdir(parents=True)
        _write_png(source)
        corpus = case_root / "corpus.json"
        _write_corpus_manifest(corpus, source)
        monkeypatch.setattr(core, "_runtime_doctor_snapshot", _runtime_snapshot)

        def fake_run(command, **kwargs):
            case_output = Path(command[command.index("-o") + 1])
            case_output.mkdir(parents=True, exist_ok=True)
            _write_sweep_fixture(case_output, list(core.CORE_RECIPE_NAMES), missing=missing)

            class Completed:
                returncode = 0

            return Completed()

        monkeypatch.setattr(core.subprocess, "run", fake_run)
        assert core.run(_args(case_root, source, corpus)) == 1
        matrix = json.loads((case_root / "certification" / "matrix_manifest.json").read_text())
        row = matrix["cases"][0]
        assert matrix["automatic_pass"] is False
        assert row["automatic_pass"] is False
        if missing == "face":
            assert "probed" in row["automatic_gate"]["reasons"]
        else:
            assert any("quality" in reason for reason in row["automatic_gate"]["reasons"])


def test_no_corpus_manifest_keeps_v1_diagnostic_matrix(tmp_path, monkeypatch):
    source = tmp_path / "portrait.png"
    _write_png(source)
    recipes = list(core.CORE_RECIPE_NAMES)

    def fake_run(command, **kwargs):
        case_output = Path(command[command.index("-o") + 1])
        case_output.mkdir(parents=True, exist_ok=True)
        outputs = []
        for index, recipe in enumerate(recipes):
            name = f"{index:02d}_{recipe}.jpg"
            (case_output / name).write_bytes(b"v1-diagnostic-output")
            outputs.append({"recipe": recipe, "output": name, "status": "done"})
        (case_output / "contact_sheet.jpg").write_bytes(b"sheet")
        (case_output / "manifest.json").write_text(json.dumps({
            "global_only": False,
            "outputs": outputs,
            "contact_sheet": "contact_sheet.jpg",
        }))

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(core.subprocess, "run", fake_run)
    args = Namespace(
        case=["skin_tone_lighting_glasses_wig_hands_on_face_group_portrait=" + str(source)],
        output=str(tmp_path / "v1"),
        max_dim=None,
        global_only=False,
        allow_incomplete_corpus=False,
        stop_on_fail=False,
    )

    assert core.run(args) == 0
    matrix = json.loads((tmp_path / "v1" / "matrix_manifest.json").read_text())
    assert matrix["version"] == 1
    assert "runtime_doctor" not in matrix
    assert matrix["automatic_pass"] is True
