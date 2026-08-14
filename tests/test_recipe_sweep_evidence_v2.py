"""Native-free evidence contract tests for the recipe sweep."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "recipes" / "recipe_sweep.py"
_SPEC = importlib.util.spec_from_file_location("recipe_sweep_evidence_v2", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
recipe_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recipe_sweep)


def _result_stub(*, detector_available: bool = True):
    landmarks = SimpleNamespace(
        landmark=[
            SimpleNamespace(x=0.25, y=0.30, z=-0.01),
            SimpleNamespace(x=0.75, y=0.30, z=0.02),
        ]
    )
    face_data = SimpleNamespace(
        landmarks=landmarks,
        bbox=(10, 12, 80, 90),
        ied=42.0,
        confidence=0.97,
        confidence_source="retinaface",
    )
    regions = SimpleNamespace(skin=np.ones((8, 8), dtype=np.float32))
    context = SimpleNamespace(face_data=face_data, regions=regions)
    detector = SimpleNamespace(
        available=detector_available,
        backend_name="stub_tasks" if detector_available else "unavailable",
        unavailable_reason=None if detector_available else "stub detector unavailable",
        runtime_status=lambda: {
            "mode": "face_aware" if detector_available else "global_only",
            "available": detector_available,
            "backend": "stub_tasks" if detector_available else "unavailable",
            "probe_state": "initialized" if detector_available else "blocked",
            "reason": None if detector_available else "stub detector unavailable",
        },
    )
    session = SimpleNamespace(get_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    parser = SimpleNamespace(
        _sess=session,
        _model_path="models/bisenet.onnx",
        backend_name="bisenet_onnx",
    )
    params = SimpleNamespace(
        _qa_results={"halo": {"raw_context_metric": 4.0}},
        _aa6_warp_field=None,
    )
    result = SimpleNamespace(
        qa=[SimpleNamespace(
            detector="halo",
            score=0.9,
            threshold=15.0,
            flagged=True,
            message="halo detected",
            details={"raw_warning_detail": "kept"},
        )],
        qa_observations={
            "legacy_complete": {
                "score": 0.2,
                "flagged": False,
                "threshold": 1.0,
                "successful_metric": 0.2,
            }
        },
        face_count=1,
        face_contexts=[context],
        skin_mask=np.ones((8, 8), dtype=np.float32),
        runtime_diagnostics={"denoise": {"backend": "stub"}},
        params=params,
        shape=(8, 8, 3),
    )
    engine = SimpleNamespace(_detector=detector, _parser=parser)
    return result, engine


def test_complete_qa_vector_keeps_clean_flagged_raw_and_roi_details():
    result, _engine = _result_stub()
    calls = {}

    def runner(image, **kwargs):
        calls["shape"] = image.shape
        calls["kwargs"] = kwargs
        return {
            "texture": {
                "score": 0.12,
                "threshold": 0.50,
                "flagged": False,
                "texture_energy": 0.88,
                "roi": {"source": "face_skin", "pixels": 32},
            },
            "halo": {
                "score": 0.80,
                "threshold": 0.50,
                "flagged": True,
                "available": True,
                "edge_count": 14,
            },
            "parser_metric": {
                "score": 0.0,
                "flagged": False,
                "available": False,
                "error": "stub parser metric unavailable",
            },
        }

    output = np.zeros((8, 8, 3), dtype=np.uint8)
    evidence = recipe_sweep._extract_qa_evidence(
        result,
        output,
        output.copy(),
        global_only=False,
        qa_runner=runner,
    )

    assert calls["shape"] == (8, 8, 3)
    assert calls["kwargs"]["face_skin_mask"] is result.skin_mask
    observations = {item["detector"]: item for item in evidence["observations"]}
    assert set(observations) == {"texture", "halo", "parser_metric", "legacy_complete"}
    assert observations["texture"]["flagged"] is False
    assert observations["texture"]["details"]["texture_energy"] == 0.88
    assert observations["texture"]["raw_details"]["roi"]["pixels"] == 32
    assert observations["texture"]["roi_provenance"]["roi"]["source"] == "face_skin"
    assert observations["halo"]["message"] == "halo detected"
    assert observations["halo"]["details"]["edge_count"] == 14
    assert observations["halo"]["result_qa"]["details"]["raw_warning_detail"] == "kept"
    assert observations["parser_metric"]["status"] == "unavailable"
    assert [item["detector"] for item in evidence["flagged"]] == ["halo"]
    assert [item["detector"] for item in evidence["unavailable"]] == ["parser_metric"]
    assert [item["detector"] for item in evidence["result_qa"]] == ["halo"]
    assert evidence["coverage_complete"] is True


def test_runtime_evidence_records_faces_landmarks_parser_provider_and_roi():
    result, engine = _result_stub()
    evidence = recipe_sweep._extract_runtime_evidence(result, engine, global_only=False)

    assert evidence["version"] == 2
    assert evidence["mode"] == "face-aware"
    assert evidence["face_aware_run"] is True
    assert evidence["face"]["face_count"] == 1
    assert evidence["face"]["faces"][0]["landmarks"] == [
        {"x": 0.25, "y": 0.3, "z": -0.01},
        {"x": 0.75, "y": 0.3, "z": 0.02},
    ]
    assert evidence["detector"]["backend"] == "stub_tasks"
    assert evidence["detector"]["probe_state"] == "initialized"
    assert evidence["parser"]["status"] == "initialized"
    assert evidence["provider_evidence"]["active"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert evidence["roi_provenance"]["masks"]["skin_mask"]["source"] == "processing_result.skin_mask"
    assert evidence["runtime_diagnostics"]["denoise"]["backend"] == "stub"


def test_unavailable_detector_is_explicitly_non_face_aware():
    result, engine = _result_stub(detector_available=False)
    evidence = recipe_sweep._extract_runtime_evidence(result, engine, global_only=False)

    assert evidence["face_aware_run"] is False
    assert evidence["certifying"] is False
    assert evidence["mode"] == "global-only-fallback"
    assert evidence["detector"]["status"] == "unavailable"
    assert evidence["detector"]["reason"] == "stub detector unavailable"


def test_global_only_evidence_is_diagnostic_and_has_no_face_count():
    output = np.zeros((8, 8, 3), dtype=np.uint8)

    def runner(_image, **_kwargs):
        return {"global_metric": {"score": 0.1, "flagged": False, "metric": 0.1}}

    evidence = recipe_sweep._extract_qa_evidence(
        None,
        output,
        output.copy(),
        global_only=True,
        qa_runner=runner,
    )
    runtime = recipe_sweep._extract_runtime_evidence(None, None, global_only=True)

    assert evidence["status"] == "diagnostic_only"
    assert evidence["certifying"] is False
    assert evidence["observations"][0]["details"]["metric"] == 0.1
    assert runtime["mode"] == "global-only"
    assert runtime["face"]["face_count"] is None
    assert runtime["detector"]["reason_code"] == "global_only_requested"
    assert runtime["certifying"] is False


def test_qa_runner_failure_is_retained_as_unavailable_observation():
    result, _engine = _result_stub()

    def failing_runner(_image, **_kwargs):
        raise RuntimeError("stub QA backend failed")

    evidence = recipe_sweep._extract_qa_evidence(
        result,
        np.zeros((8, 8, 3), dtype=np.uint8),
        None,
        global_only=False,
        qa_runner=failing_runner,
    )

    assert evidence["runner"]["status"] == "error"
    assert evidence["certifying"] is False
    failure = next(item for item in evidence["observations"] if item["detector"] == "qa_runner")
    assert failure["available"] is False
    assert failure["status"] == "unavailable"
    assert "stub QA backend failed" in failure["error"]


def test_manifest_keeps_full_qa_and_marks_global_only_non_certifying(tmp_path, monkeypatch):
    input_path = tmp_path / "portrait.jpg"
    input_path.write_bytes(b"input")
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(recipe_sweep, "imread_exif", lambda _path: image.copy())
    monkeypatch.setattr(recipe_sweep, "resize_for_processing", lambda img, _max_dim: (img, 1.0))
    monkeypatch.setattr(
        recipe_sweep,
        "_write_image",
        lambda path, _image, _fmt, _quality: path.write_bytes(b"output"),
    )
    monkeypatch.setattr(recipe_sweep, "generate_contact_sheet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(recipe_sweep, "_parse_recipe_list", lambda raw: ["natural"])

    def fake_process(**_kwargs):
        observation = {
            "detector": "global_metric",
            "score": 0.2,
            "threshold": 0.5,
            "flagged": False,
            "available": True,
            "status": "ok",
            "details": {"successful_metric": 0.2},
            "raw_details": {"successful_metric": 0.2},
            "roi_provenance": {},
            "message": "",
            "source": "stub",
        }
        qa = {
            "version": 2,
            "status": "diagnostic_only",
            "certifying": False,
            "coverage_complete": True,
            "observations": [observation],
            "flagged": [],
            "unavailable": [],
            "runner": {"status": "complete"},
            "result_qa": [],
            "raw_qa": [],
            "context_qa": [],
        }
        runtime = {
            "version": 2,
            "mode": "global-only",
            "certifying": False,
            "face_aware_run": False,
            "detector": {"status": "not_run", "certifying": False},
            "face": {"face_count": None},
            "parser": {"status": "not_run"},
            "provider_evidence": {},
            "runtime_diagnostics": {},
            "roi_provenance": {},
        }
        return image.copy(), None, {"qa": qa, "runtime": runtime}

    monkeypatch.setattr(recipe_sweep, "_process_recipe_with_evidence", fake_process)
    args = SimpleNamespace(
        input=str(input_path),
        output=str(tmp_path / "evidence"),
        recipes="natural",
        skip="",
        format="jpg",
        quality=95,
        max_dim=None,
        global_only=True,
        compare=False,
        contact_sheet=False,
        sheet_cols=4,
        cell_size=320,
        fail_on_qa=False,
        keep_going=True,
        resume=False,
        restart_engine_per_recipe=False,
    )

    assert recipe_sweep.run_sweep(args) == 0
    manifest = json.loads((Path(args.output) / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["outputs"][0]
    assert manifest["evidence_version"] == 2
    assert manifest["diagnostic_only"] is True
    assert manifest["certifying"] is False
    assert manifest["face_aware_run"] is False
    assert row["qa"][0]["flagged"] is False
    assert row["qa_warnings"] == []
    assert row["qa_evidence"]["status"] == "diagnostic_only"
    assert row["runtime_evidence"]["detector"]["status"] == "not_run"
