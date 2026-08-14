"""Tests for the detector's explicit face-aware/global-only contract."""

from __future__ import annotations

from retouch.detection import FaceDetector


def test_initialized_detector_is_face_aware_even_before_a_face_is_found() -> None:
    detector = FaceDetector.__new__(FaceDetector)
    detector.available = True
    detector.backend_name = "mediapipe_legacy"
    detector.unavailable_reason = None

    assert detector.runtime_status() == {
        "mode": "face_aware",
        "available": True,
        "backend": "mediapipe_legacy",
        "reason": None,
        "probe_state": "initialized",
    }


def test_unavailable_detector_is_explicitly_global_only() -> None:
    detector = FaceDetector.__new__(FaceDetector)
    detector.available = False
    detector.backend_name = "unavailable"
    detector.unavailable_reason = "RuntimeError: incompatible MediaPipe runtime"

    status = detector.runtime_status()

    assert status["mode"] == "global_only"
    assert status["available"] is False
    assert status["reason"] == "RuntimeError: incompatible MediaPipe runtime"
    assert status["probe_state"] == "blocked"
