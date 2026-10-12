"""Tests for full-resolution RAW calibration to a Fuji camera preview."""

import cv2
import numpy as np
import pytest

from retouch.fuji_match import calibrate_to_fuji_preview


def _lab(img):
    return cv2.cvtColor(
        np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)


def test_zero_strength_is_an_exact_float_copy():
    source = np.full((12, 16, 3), 90, dtype=np.float32)
    preview = np.full((8, 10, 3), 180, dtype=np.uint8)

    result = calibrate_to_fuji_preview(source, preview, strength=0.0)

    assert result.dtype == np.float32
    assert np.array_equal(result, source)
    assert result is not source


def test_calibration_moves_tone_and_chroma_toward_preview():
    rng = np.random.default_rng(7)
    source = rng.uniform(20, 180, size=(80, 120, 3)).astype(np.float32)
    # A brighter, warmer rendition with a materially different distribution.
    preview = source.copy()
    preview[:, :, 2] = np.clip(preview[:, :, 2] * 1.25 + 22, 0, 255)
    preview[:, :, 0] = np.clip(preview[:, :, 0] * 0.72, 0, 255)
    preview = np.clip(preview * 1.12, 0, 255).astype(np.uint8)

    result = calibrate_to_fuji_preview(source, preview, strength=1.0)

    source_lab, preview_lab, result_lab = _lab(source), _lab(preview), _lab(result)
    quantiles = [10, 50, 90]
    source_l_error = np.abs(
        np.percentile(source_lab[:, :, 0], quantiles)
        - np.percentile(preview_lab[:, :, 0], quantiles)
    ).mean()
    result_l_error = np.abs(
        np.percentile(result_lab[:, :, 0], quantiles)
        - np.percentile(preview_lab[:, :, 0], quantiles)
    ).mean()

    assert result.dtype == np.float32
    assert result_l_error < source_l_error
    for channel in (1, 2):
        assert abs(result_lab[:, :, channel].mean() - preview_lab[:, :, channel].mean()) < \
            abs(source_lab[:, :, channel].mean() - preview_lab[:, :, channel].mean())


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_calibration_rejects_invalid_strength(strength):
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="strength"):
        calibrate_to_fuji_preview(image, image, strength=strength)
