"""Geometry-only reference handling for post-reshape photometric QA."""

import numpy as np

from retouch import qa_detectors


def test_photometric_detectors_use_geometry_reference(monkeypatch):
    image = np.full((48, 48, 3), 120, dtype=np.uint8)
    original = np.full_like(image, 90)
    geometry_only = np.full_like(image, 105)
    mask = np.ones(image.shape[:2], dtype=np.float32)
    seen = {}

    def capture(name, reference):
        seen[name] = reference
        return {"score": 0.0, "flagged": False}

    monkeypatch.setattr(
        qa_detectors,
        "detect_plastic_skin",
        lambda _img, _mask, reference: capture("plastic_skin", reference),
    )
    monkeypatch.setattr(
        qa_detectors,
        "detect_halo",
        lambda _img, _mask, img_before=None: capture("halo", img_before),
    )
    monkeypatch.setattr(
        qa_detectors,
        "detect_color_drift",
        lambda _img, _mask, reference: capture("color_drift", reference),
    )
    monkeypatch.setattr(
        qa_detectors,
        "detect_pore_spectrum_distance",
        lambda _img, _mask, reference: capture("pore_spectrum", reference),
    )
    monkeypatch.setattr(
        qa_detectors,
        "gui_skin_score",
        lambda _img, _mask, reference: capture("skin_score", reference),
    )
    monkeypatch.setattr(
        qa_detectors,
        "evaluate_harmony",
        lambda _img, **kwargs: capture("harmony", kwargs["reference_img_bgr"]),
    )
    monkeypatch.setattr(
        qa_detectors,
        "perceived_retouching_vector",
        lambda _img, reference, face_mask=None, **_kwargs: capture("perceived", reference),
    )

    qa_detectors.run_all(
        image,
        skin_mask=mask,
        reference_img_bgr=original,
        img_before=original,
        person_mask=mask,
        face_skin_mask=mask,
        body_skin_mask=mask,
        warp_field=np.zeros((*mask.shape, 2), dtype=np.float32),
        geometry_reference_img_bgr=geometry_only,
    )

    for name in (
        "plastic_skin", "halo", "color_drift", "pore_spectrum", "skin_score", "harmony"
    ):
        assert seen[name] is geometry_only
    # The forensic vector still needs the original reference and warp field
    # to quantify geometry change rather than erase it from the comparison.
    assert seen["perceived"] is original


def test_unavailable_geometry_reference_makes_photo_comparisons_not_run():
    image = np.full((48, 48, 3), 120, dtype=np.uint8)
    original = np.full_like(image, 90)
    mask = np.ones(image.shape[:2], dtype=np.float32)

    evidence = qa_detectors.run_all(
        image,
        skin_mask=mask,
        reference_img_bgr=original,
        img_before=original,
        person_mask=mask,
        face_skin_mask=mask,
        geometry_reference_reason="geometry-only reference shape mismatch after proxy scaling",
    )

    for name in ("color_drift", "pore_spectrum"):
        assert evidence[name]["status"] == qa_detectors.QA_STATUS_NOT_RUN
        assert evidence[name]["reason"] == (
            "geometry-only reference shape mismatch after proxy scaling"
        )
        assert evidence[name]["geometry_reference_available"] is False
