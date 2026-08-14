"""Focused tests for the dependency-light Runtime Doctor."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from retouch import runtime_doctor as doctor


def _manifest_entry(payload: bytes, *, url="", availability="unavailable"):
    return {
        "filename": "model.bin",
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "availability": availability,
    }


def test_declared_constraints_and_version_matching(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """[project]
requires-python = ">=3.9,<4"
dependencies = [
    "mediapipe==0.10.5",
    "opencv-contrib-python>=4.8,<4.12",
]
""",
        encoding="utf-8",
    )

    declarations = doctor.read_declared_project(pyproject)

    assert declarations["status"] == doctor.STATUS_OK
    assert declarations["requires_python"] == ">=3.9,<4"
    assert [item["canonical_name"] for item in declarations["dependencies"]] == [
        "mediapipe",
        "opencv-contrib-python",
    ]
    assert doctor.version_satisfies("4.11.0", ">=4.8,<4.12")
    assert not doctor.version_satisfies("4.13.0", ">=4.8,<4.12")


def test_package_inventory_reports_installed_version_and_mismatch(monkeypatch):
    versions = {"mediapipe": "0.10.35", "numpy": "1.26.4"}
    monkeypatch.setattr(
        doctor,
        "_installed_distribution_version",
        lambda name: (versions.get(name), None),
    )

    report = doctor.inspect_packages([
        {"name": "mediapipe", "canonical_name": "mediapipe", "specifier": "==0.10.5"},
        {"name": "numpy", "canonical_name": "numpy", "specifier": ">=1.24,<2"},
    ])

    assert report["status"] == doctor.STATUS_ERROR
    assert report["items"]["mediapipe"]["installed_version"] == "0.10.35"
    assert report["items"]["mediapipe"]["reason_code"] == "distribution_constraint_mismatch"
    assert report["items"]["numpy"]["compatible"] is True


def test_pip_check_reports_transitive_conflicts(monkeypatch):
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="opencv-python requires numpy>=2",
            stderr="grpcio is not supported on this platform",
        ),
    )

    report = doctor.check_pip_check()

    assert report["status"] == doctor.STATUS_ERROR
    assert report["reason_code"] == "pip_check_failed"
    assert "numpy>=2" in report["output"]
    assert "grpcio" in report["output"]


def test_duplicate_opencv_distributions_are_blocking(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_opencv_metadata",
        lambda: {
            "opencv-python": "4.13.0.92",
            "opencv-contrib-python": "4.13.0.92",
        },
    )
    monkeypatch.setattr(
        doctor,
        "_safe_import",
        lambda name: (SimpleNamespace(__version__="4.13.0", __file__="/tmp/cv2.so"), None),
    )

    report = doctor.check_opencv()

    assert report["status"] == doctor.STATUS_ERROR
    assert report["reason_code"] == "duplicate_opencv_distributions"
    assert report["distribution_count"] == 2


def test_model_hashes_and_controlled_distribution_metadata(tmp_path):
    good_payload = b"verified model"
    bad_payload = b"changed model"
    (tmp_path / "model.bin").write_bytes(good_payload)
    (tmp_path / "bad.bin").write_bytes(bad_payload)
    manifest = {
        "schema_version": 1,
        "models": {
            "good": _manifest_entry(good_payload),
            "downloadable_missing": {
                **_manifest_entry(
                    b"future",
                    url="https://models.example.test/future.bin",
                    availability="downloadable",
                ),
                "filename": "missing.bin",
            },
            "bad": {
                **_manifest_entry(b"expected"),
                "filename": "bad.bin",
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = doctor.inspect_models(manifest_path)

    assert report["items"]["good"]["status"] == doctor.STATUS_OK
    assert report["items"]["good"]["candidates"][0]["reason_code"] == "model_hash_verified"
    assert report["items"]["downloadable_missing"]["status"] == doctor.STATUS_WARNING
    assert report["items"]["downloadable_missing"]["distribution"]["downloadable"] is True
    assert report["items"]["bad"]["reason_code"] == "model_integrity_or_distribution_problem"
    json.dumps(report)


def test_optional_backend_failures_are_reported_without_raising(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_mediapipe_static_info",
        lambda: {
            "status": doctor.STATUS_UNAVAILABLE,
            "reason_code": "mediapipe_not_installed",
            "reason": "MediaPipe is not installed",
            "installed_version": None,
            "module_present": False,
            "metadata_error": None,
            "module_spec_error": None,
        },
    )
    detector = doctor.check_detector({}, probe=False)
    parser = doctor.check_parser({}, {"status": doctor.STATUS_UNAVAILABLE})

    monkeypatch.setattr(doctor, "_installed_distribution_version", lambda name: (None, None))
    monkeypatch.setattr(doctor, "_module_exists", lambda name: (False, None))
    onnx = doctor.check_onnx_providers()

    assert detector["status"] == doctor.STATUS_UNAVAILABLE
    assert detector["available"] is False
    assert parser["backend"] == "landmark_only"
    assert onnx["status"] == doctor.STATUS_UNAVAILABLE
    assert onnx["active_provider_verified"] is False
    assert json.dumps({"detector": detector, "parser": parser, "onnx": onnx})


def test_unsupported_macos_tasks_backend_is_explicitly_global_only(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_mediapipe_static_info",
        lambda: {
            "status": doctor.STATUS_OK,
            "installed_version": "0.10.35",
            "module_present": True,
            "metadata_error": None,
            "module_spec_error": None,
        },
    )

    detector = doctor.check_detector(
        {"face_landmarker": {"available": True}},
        platform_name="darwin",
        probe=False,
    )
    mode = doctor.check_mode(detector)

    assert detector["available"] is False
    assert detector["backend"] == "unavailable"
    assert mode["effective"] == "global-only"
    assert mode["global_only"] is True


def test_unsupported_macos_probe_skips_native_import(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_mediapipe_static_info",
        lambda: {
            "status": doctor.STATUS_OK,
            "installed_version": "0.10.35",
            "module_present": True,
            "metadata_error": None,
            "module_spec_error": None,
        },
    )
    monkeypatch.setattr(doctor, "_safe_import", lambda name: (_ for _ in ()).throw(AssertionError(name)))

    detector = doctor.check_detector(
        {"face_landmarker": {"available": True}},
        platform_name="darwin",
        probe=True,
        environ={"RETOUCH_MEDIAPIPE_BACKEND": "auto"},
    )

    assert detector["status"] == doctor.STATUS_UNAVAILABLE
    assert detector["probe_skipped"] is True


def test_detector_probe_uses_authoritative_runtime_status(monkeypatch):
    class FakeDetector:
        def __init__(self, allow_unavailable=False):
            assert allow_unavailable is True

        def runtime_status(self):
            return {
                "mode": "face_aware",
                "available": True,
                "backend": "mediapipe_legacy",
                "reason": None,
                "probe_state": "initialized",
            }

    monkeypatch.setattr(
        doctor,
        "_mediapipe_static_info",
        lambda: {
            "status": doctor.STATUS_OK,
            "installed_version": "0.10.5",
            "module_present": True,
            "metadata_error": None,
            "module_spec_error": None,
        },
    )

    def fake_import(name):
        if name == "mediapipe":
            return SimpleNamespace(solutions=object(), tasks=object()), None
        if name == "retouch.detection":
            return SimpleNamespace(FaceDetector=FakeDetector), None
        raise AssertionError(name)

    monkeypatch.setattr(doctor, "_safe_import", fake_import)
    detector = doctor.check_detector({}, probe=True, platform_name="linux")

    assert detector["status"] == doctor.STATUS_OK
    assert detector["available"] is True
    assert detector["runtime_status"]["mode"] == "face_aware"


def test_mode_helpers_and_gui_aliases(monkeypatch):
    detector = {"available": False}
    report = {"mode": doctor.check_mode(detector), "global_only": True}

    assert doctor.face_mode(report) == "global-only"
    assert doctor.check_mode({"available": True})["effective"] == "face-aware"
    assert doctor.check_mode({"available": False}, "face-aware")["status"] == doctor.STATUS_ERROR
    assert "Runtime Doctor:" in doctor.format_runtime_report(report)

    monkeypatch.setattr(doctor, "collect_runtime_report", lambda **kwargs: {"global_only": True})
    assert doctor.run_doctor() == {"global_only": True}
    assert doctor.runtime_doctor_report() == {"global_only": True}


def test_report_is_json_serializable_and_exposes_explicit_mode(tmp_path, monkeypatch):
    common = {
        "status": doctor.STATUS_OK,
        "reason_code": "ok",
        "reason": "ok",
    }
    monkeypatch.setattr(doctor, "check_opencv", lambda: dict(common))
    monkeypatch.setattr(doctor, "inspect_models", lambda path: {**common, "items": {}})
    monkeypatch.setattr(doctor, "check_onnx_providers", lambda: {**common, "available": []})
    monkeypatch.setattr(
        doctor,
        "check_detector",
        lambda models, probe=False: {**common, "available": True, "backend": "fake"},
    )
    monkeypatch.setattr(
        doctor,
        "check_parser",
        lambda models, onnx, probe=False: {**common, "backend": "fake"},
    )
    monkeypatch.setattr(doctor, "check_entrypoint_seam", lambda root: dict(common))

    report = doctor.collect_runtime_report(project_root=tmp_path)

    assert report["face_aware"] is True
    assert report["global_only"] is False
    assert doctor.face_mode(report) == "face-aware"
    json.dumps(report)


def test_module_cli_works_without_site_packages(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nrequires-python = \">=3.9\"\ndependencies = []\n",
        encoding="utf-8",
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text('{"schema_version": 1, "models": {}}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-m",
            "retouch.runtime_doctor",
            "--project-root",
            str(tmp_path),
            "--compact",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.stdout.strip()
    payload = json.loads(result.stdout)
    assert "detector" in payload
    assert "global_only" in payload


def test_entrypoint_seam_is_visible_without_editing_init():
    seam = doctor.check_entrypoint_seam(Path(__file__).resolve().parents[1])

    assert seam["reason_code"] in {
        "package_initializer_eager_engine_import",
        "package_initializer_lazy_enough",
    }
    if seam["reason_code"] == "package_initializer_eager_engine_import":
        assert "__init__.py" in seam["required_follow_up"]
    else:
        assert seam["status"] == doctor.STATUS_OK
