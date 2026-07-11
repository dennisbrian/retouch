"""Tests for retouch/chromophore.py — melanin/hemoglobin skin decomposition."""

import numpy as np
import cv2
import pytest

from retouch.chromophore import (
    decompose_chromophores,
    hemoglobin_breakdown_phase,
)


def _patch(r: int, g: int, b: int, h: int = 16, w: int = 16) -> np.ndarray:
    """Build a uniform uint8 BGR patch."""
    return np.full((h, w, 3), (b, g, r), dtype=np.uint8)


def test_red_patch_hemoglobin_dominant():
    """A red patch (high R, low G/B) yields hemoglobin >> melanin."""
    img = _patch(230, 20, 20)
    melanin, hemoglobin = decompose_chromophores(img)
    assert hemoglobin.mean() > melanin.mean()
    assert hemoglobin.mean() > 0.0


def test_brown_patch_melanin_dominant():
    """A dark/brown patch yields high melanin, low hemoglobin."""
    img = _patch(80, 45, 25)
    melanin, hemoglobin = decompose_chromophores(img)
    assert melanin.mean() > hemoglobin.mean()
    assert melanin.mean() > 0.0


def test_gray_patch_both_low():
    """A neutral gray patch yields both low / near-equal, non-negative."""
    img = _patch(128, 128, 128)
    melanin, hemoglobin = decompose_chromophores(img)
    assert melanin.min() >= 0.0
    assert hemoglobin.min() >= 0.0
    assert melanin.mean() < 1.0
    assert hemoglobin.mean() < 1.0


def test_non_negative_and_ordering():
    red = decompose_chromophores(_patch(230, 20, 20))
    brown = decompose_chromophores(_patch(80, 45, 25))
    for m, h in (red, brown):
        assert m.min() >= 0.0
        assert h.min() >= 0.0
    # red -> more hemoglobin than brown; brown -> more melanin than red
    assert red[1].mean() > brown[1].mean()
    assert brown[0].mean() > red[0].mean()


def test_dtype_and_shape():
    img = _patch(150, 100, 80, h=32, w=24)
    melanin, hemoglobin = decompose_chromophores(img)
    assert melanin.dtype == np.float32
    assert hemoglobin.dtype == np.float32
    assert melanin.shape == (32, 24)
    assert hemoglobin.shape == (32, 24)


def test_uint8_input_no_raise():
    img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    try:
        decompose_chromophores(img)
    except Exception as exc:  # noqa: BLE001 - test must not raise
        pytest.fail(f"decompose_chromophores raised on uint8 input: {exc}")


def test_phase_dtype_and_values():
    img = _patch(150, 100, 80)
    _, hemoglobin = decompose_chromophores(img)
    phase = hemoglobin_breakdown_phase(hemoglobin, img)
    assert phase.dtype == np.uint8
    assert set(np.unique(phase).tolist()).issubset({0, 1, 2})


def test_phase_stable_on_uniform():
    img = _patch(120, 60, 150)  # purple-ish uniform
    _, hemoglobin = decompose_chromophores(img)
    phase = hemoglobin_breakdown_phase(hemoglobin, img)
    assert np.unique(phase).size == 1


def test_phase_purple_vs_yellow():
    purple = _patch(140, 60, 200)  # R>B>G -> purple/magenta
    yellow = _patch(230, 220, 60)  # R,G high, B low -> yellow
    _, hb_p = decompose_chromophores(purple)
    _, hb_y = decompose_chromophores(yellow)
    phase_p = hemoglobin_breakdown_phase(hb_p, purple)
    phase_y = hemoglobin_breakdown_phase(hb_y, yellow)
    assert phase_p.mean() != phase_y.mean()


def test_phase_accepts_hue_map():
    img = _patch(140, 60, 200)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    _, hemoglobin = decompose_chromophores(img)
    phase = hemoglobin_breakdown_phase(hemoglobin, hue)
    assert phase.dtype == np.uint8
    assert np.unique(phase).size == 1
