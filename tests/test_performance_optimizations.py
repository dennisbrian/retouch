"""Exactness tests for the native-resolution performance optimizations.

These tests exercise only private, geometry-heavy helpers.  They intentionally
avoid production recipe changes and keep the old full-canvas implementation
available in-process where the optimized wrapper can be compared directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from retouch.freckle import FreckleRemover
from retouch.hairwork import _flyaway_mask, _flyaway_mask_full
from retouch.marks import _relative_features
from retouch.undereye import build_undereye_support


def _scene(h: int = 192, w: int = 256):
    image = np.full((h, w, 3), (150, 175, 205), dtype=np.uint8)
    cv2.circle(image, (w // 3, h // 3), 3, (110, 120, 160), -1)
    cv2.circle(image, (w // 2, h // 2), 5, (85, 95, 120), -1)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1, -1)
    return image, mask


def test_freckle_bbox_slice_matches_full_canvas_component_evaluation():
    image, face = _scene()
    remover = FreckleRemover()
    lab = remover._to_lab(image)
    labels, stats, _centroids, n_labels = remover._detect_components(image, face)
    assert labels is not None and stats is not None
    skin_pixels = lab[face > 0.3]
    a_median = float(np.median(skin_pixels[:, 1]))
    a_std = float(np.std(skin_pixels[:, 1]))
    l_median = float(np.median(skin_pixels[:, 0]))
    l_std = float(np.std(skin_pixels[:, 0]))

    checked = 0
    for label in range(1, n_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) < remover._MIN_AREA:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        full = remover._classify_anomaly(
            lab, labels == label, a_median, a_std, l_median, l_std
        )
        local = remover._classify_anomaly(
            lab[y : y + height, x : x + width],
            labels[y : y + height, x : x + width] == label,
            a_median,
            a_std,
            l_median,
            l_std,
        )
        assert local == full
        checked += 1
    assert checked > 0


def test_relative_features_precomputed_context_matches_legacy_fallback():
    image, face = _scene()
    remover = FreckleRemover()
    classifications = remover.classify_anomalies(image, face_mask=face, confidence_threshold=0.0)
    assert classifications
    classification = classifications[0]
    mel = np.linspace(0.0, 1.0, image.shape[0] * image.shape[1], dtype=np.float32).reshape(image.shape[:2])
    hb = mel[::-1]

    legacy = _relative_features(image, classification, face, mel, hb)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference = face > 0.3
    reference_pixels = lab[reference] if np.any(reference) else lab.reshape(-1, 3)
    reference_stats = {}
    for channel in (0, 1):
        values = reference_pixels[:, channel]
        median = float(np.median(values))
        reference_stats[channel] = (median, float(np.median(np.abs(values - median))))
    map_context = {}
    for name, source in (("mel_rel", mel), ("hb_rel", hb)):
        values = source.astype(np.float32)
        baseline = values[reference] if np.any(reference) else values.reshape(-1)
        median = float(np.median(baseline))
        map_context[name] = (values, median, float(np.median(np.abs(baseline - median))))

    optimized = _relative_features(
        image,
        classification,
        face,
        mel,
        hb,
        lab=lab,
        reference_pixels=reference_pixels,
        reference_stats=reference_stats,
        map_context=map_context,
    )
    assert optimized == legacy


def test_under_eye_roi_support_preserves_full_frame_mid_and_edge_contracts():
    # Cover both normal and clamped ROI bounds while keeping the fixture small
    # and deterministic.  The public contract is full-frame geometry with
    # support/ring confined to the padded neighborhood and valid true outside
    # exclusions.
    h = w = 256
    for cx, cy in ((128, 128), (8, 8)):
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (cx, cy), (18, 8), 0, 0, 360, 1, -1)
        exclude = np.zeros_like(mask)
        cv2.ellipse(exclude, (cx, cy - 10), (22, 4), 0, 0, 360, 1, -1)
        support, ring, valid = build_undereye_support(
            mask, 90.0, exclude=exclude, skin=np.ones_like(mask)
        )
        assert support.shape == (h, w)
        assert ring.shape == (h, w)
        assert valid.shape == (h, w)
        assert support.dtype == np.float32
        assert ring.dtype == bool
        assert valid.dtype == bool
        assert np.all((support >= 0.0) & (support <= 1.0))
        # The crop pad is finite; no support/ring may appear in the opposite
        # far corner of the frame, and pixels outside the exclusion remain
        # valid for the low-pass estimate.
        assert support[max(0, cy - 80): min(h, cy + 80), max(0, cx - 80): min(w, cx + 80)].max() > 0
        assert support[h - 1, w - 1] == 0.0
        assert not bool(ring[h - 1, w - 1])
        assert bool(valid[h - 1, w - 1])


def test_flyaway_roi_wrapper_matches_full_canvas_reference():
    rng = np.random.default_rng(20260908)
    h, w = 192, 256
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    hair = np.zeros((h, w), np.float32)
    cv2.ellipse(hair, (w // 2, h // 2), (w // 4, h // 4), 15, 0, 360, 1, -1)
    orientation = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    coherence = rng.random((h, w), dtype=np.float32)
    exclude = np.zeros((h, w), np.float32)
    cv2.ellipse(exclude, (w // 2, h // 3), (w // 5, max(3, h // 20)), 0, 0, 360, 1, -1)

    full = _flyaway_mask_full(image, hair, orientation, coherence, 40, exclude, 100.0)
    roi = _flyaway_mask(image, hair, orientation, coherence, 40, exclude, 100.0)
    np.testing.assert_array_equal(roi, full)
