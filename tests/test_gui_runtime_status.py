"""Focused checks for the GUI's explicit runtime-mode wording."""

from __future__ import annotations


def test_gui_formats_face_aware_and_global_only_modes() -> None:
    import gui

    class FaceAware:
        def runtime_status(self) -> dict[str, str]:
            return {
                "mode": "face_aware",
                "backend": "mediapipe_legacy",
            }

    class GlobalOnly:
        def runtime_status(self) -> dict[str, str]:
            return {
                "mode": "global_only",
                "reason": "incompatible MediaPipe runtime",
            }

    assert gui._face_runtime_label(FaceAware()) == "Face-aware: mediapipe_legacy"
    assert gui._face_runtime_label(GlobalOnly()) == "Global-only: incompatible MediaPipe runtime"
    assert gui._face_runtime_label(None) == "Face-aware: unprobed"
