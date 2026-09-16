from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.input_rescue import analyze_input_quality
from retouch.perceptual_metrics import facial_feature_contrast, skin_homogeneity_state


def _regions(shape=(128, 128)):
    h, w = shape
    skin = np.ones(shape, np.float32)
    lips = np.zeros(shape, np.float32)
    lips[82:98, 44:84] = 1.0
    left_eye = np.zeros(shape, np.float32)
    left_eye[42:54, 32:50] = 1.0
    right_eye = np.zeros(shape, np.float32)
    right_eye[42:54, 78:96] = 1.0
    left_brow = np.zeros(shape, np.float32)
    left_brow[30:36, 28:52] = 1.0
    right_brow = np.zeros(shape, np.float32)
    right_brow[30:36, 76:100] = 1.0
    return SimpleNamespace(skin=skin, lips=lips, left_eye=left_eye, right_eye=right_eye,
                           left_eyebrow=left_brow, right_eyebrow=right_brow)


def _face(regions, dark_lips=False):
    image = np.full((128, 128, 3), (110, 145, 185), dtype=np.uint8)
    if dark_lips:
        image[regions.lips > 0.5] = (70, 55, 105)
    return image


def test_feature_contrast_is_relative_and_monotonic():
    regions = _regions()
    plain = facial_feature_contrast(_face(regions), regions)
    darker = facial_feature_contrast(_face(regions, dark_lips=True), regions)
    assert darker["lips"].confidence > 0.0
    assert abs(darker["lips"].luminance_contrast) > abs(plain["lips"].luminance_contrast)
    assert darker["lips"].chroma_contrast > plain["lips"].chroma_contrast


def test_missing_skin_is_low_confidence_not_a_fake_measurement():
    regions = _regions()
    regions.skin = None
    report = facial_feature_contrast(_face(_regions()), regions)
    assert all(item.confidence == 0.0 for item in report.values())


def test_feature_confidence_is_pure_support_size_not_an_evidence_ranking():
    # RESEARCH_RETOUCH_REMAINING_GAPS_2026_09_15.md #4 flags that this
    # confidence is "a support-size heuristic, not a correctness probability".
    # This test pins the exact relationship: confidence is a deterministic,
    # monotonic function of min(feature_pixels, surround_pixels) alone, with
    # no other signal (chroma, luminance contrast, etc.) contributing. Two
    # measurements with the same pixel counts but very different photographed
    # content must report identical confidence -- if this ever fails after an
    # intentional change, the "confidence == support size" framing used in
    # the gap research is stale and needs correcting alongside the code.
    regions = _regions()
    plain = facial_feature_contrast(_face(regions), regions)
    darker = facial_feature_contrast(_face(regions, dark_lips=True), regions)

    assert plain["lips"].feature_pixels == darker["lips"].feature_pixels
    assert plain["lips"].surround_pixels == darker["lips"].surround_pixels
    assert plain["lips"].confidence == darker["lips"].confidence
    assert abs(darker["lips"].luminance_contrast) > abs(plain["lips"].luminance_contrast)

    expected = min(plain["lips"].feature_pixels, plain["lips"].surround_pixels) / 500.0
    assert plain["lips"].confidence == pytest.approx(min(expected, 1.0))


def test_homogeneity_separates_chroma_blotch_and_pore_retention():
    rng = np.random.default_rng(3)
    skin = np.ones((128, 128), np.float32)
    clean = np.full((128, 128, 3), (112, 146, 186), dtype=np.uint8)
    blotchy = clean.copy().astype(np.int16)
    blotchy[..., 0] += cv2.GaussianBlur(rng.normal(0, 15, (128, 128)).astype(np.float32), (0, 0), 2).astype(np.int16)
    blotchy[..., 2] -= cv2.GaussianBlur(rng.normal(0, 15, (128, 128)).astype(np.float32), (0, 0), 2).astype(np.int16)
    blotchy = np.clip(blotchy, 0, 255).astype(np.uint8)
    clean_state = skin_homogeneity_state(clean, skin, reference_img_bgr=clean)
    blotch_state = skin_homogeneity_state(blotchy, skin, reference_img_bgr=clean)
    assert blotch_state.chroma_blotch_std > clean_state.chroma_blotch_std
    assert blotch_state.pore_retention is not None


def test_input_preflight_flags_synthetic_chroma_noise_and_grid():
    clean = np.full((128, 128, 3), (125, 140, 160), dtype=np.uint8)
    noisy = clean.astype(np.int16)
    rng = np.random.default_rng(8)
    noisy[..., 0] += rng.normal(0, 9, (128, 128)).astype(np.int16)
    noisy[..., 2] -= rng.normal(0, 9, (128, 128)).astype(np.int16)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    grid = clean.copy()
    grid[:, 7::8] = np.clip(grid[:, 7::8].astype(np.int16) + 18, 0, 255).astype(np.uint8)
    assert analyze_input_quality(noisy).chroma_noise > analyze_input_quality(clean).chroma_noise
    assert analyze_input_quality(grid).jpeg_grid_evidence > analyze_input_quality(clean).jpeg_grid_evidence
