"""``dark_circles`` (legacy, eyes.dark_circles) and ``undereye_darken_removal``
(undereye.darken_removal) drive one and the same UnderEyeProcessor darken pass.
2026-08-31 per-op audit finding 1: applying both sequentially double-brightened
the under-eye in 27/128 shipping recipes. They are now aliased at the
_process_face_core dispatch (single pass at max(a, b), not a + b)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests.test_engine import _build_synthetic_regions, _build_synthetic_face

ROI = 256


def _processors():
    from retouch.skin import SkinProcessor
    from retouch.blemish import BlemishRemover
    from retouch.eyes import EyeEnhancer
    from retouch.undereye import UnderEyeRepairer
    from retouch.lips import LipEnhancer
    from retouch.teeth import TeethWhitener
    from retouch.makeup import MakeupEngine
    from retouch.hair import HairEnhancer
    from retouch.relight import Relighter
    from retouch.frequency import FrequencySeparator

    return {
        "skin": SkinProcessor(), "relighter": Relighter(), "blemish": BlemishRemover(),
        "undereye": UnderEyeRepairer(), "eyes": EyeEnhancer(), "teeth": TeethWhitener(),
        "lips": LipEnhancer(), "makeup": MakeupEngine(), "hair": HairEnhancer(),
        "frequency": FrequencySeparator(),
    }


def _ctx(dark_circles: float, undereye_darken_removal: float):
    from retouch.engine import ProcessingContext

    return ProcessingContext(
        smooth=0.0, whiten=0.0, equalize=0.0, blemish=0.0, nose_smooth=None,
        pore_synthesis=0.0, specular_bloom=0.0, dodge_burn=0.0, relight=0.0,
        eye_enhance=0.0, catchlight=0.0, lip_enhance=0.0, teeth_whiten=0.0,
        blush=0.0, hair_enhance=0.0, slimming=0.0,
        dark_circles=dark_circles, undereye_darken_removal=undereye_darken_removal,
    )


@pytest.fixture(scope="module")
def scene():
    """Skin-toned canvas with genuinely dark under-eye patches beneath the masks."""
    regions = _build_synthetic_regions(ROI, ROI, skin_value=1.0)
    left = np.zeros((ROI, ROI), np.float32)
    right = np.zeros((ROI, ROI), np.float32)
    cv2.ellipse(left, (90, 120), (28, 12), 0, 0, 360, 1.0, -1)
    cv2.ellipse(right, (166, 120), (28, 12), 0, 0, 360, 1.0, -1)
    regions.left_under_eye = left
    regions.right_under_eye = right
    canvas = np.full((ROI, ROI, 3), (150, 175, 210), np.uint8)  # BGR skin tone
    shade = cv2.GaussianBlur(np.maximum(left, right), (0, 0), 2)[..., None]
    canvas = (canvas.astype(np.float32) * (1.0 - 0.35 * shade)).clip(0, 255).astype(np.uint8)
    return canvas, regions, _build_synthetic_face(ied=30.0, size=100), np.ones((ROI, ROI), np.float32), _processors()


def _render(scene, dark_circles, undereye_darken_removal):
    from retouch.perf_optimizations import _process_face_core

    canvas, regions, face, person, procs = scene
    fr = _process_face_core(
        canvas.copy(), regions, face, _ctx(dark_circles, undereye_darken_removal),
        0, 0, ROI, ROI, person, procs,
    )
    return np.asarray(fr.canvas)


def test_darken_pass_is_live_on_scene(scene):
    off = _render(scene, 0, 0)
    on = _render(scene, 0, 60)
    assert np.abs(on.astype(int) - off.astype(int)).max() >= 3, \
        "under-eye darken pass is inert on the scene; alias tests would be vacuous"


def test_both_keys_equal_single_pass_at_max(scene):
    both = _render(scene, 20, 60)
    single = _render(scene, 0, 60)
    assert np.array_equal(both, single)


def test_legacy_key_is_alias_of_new_key(scene):
    assert np.array_equal(_render(scene, 60, 0), _render(scene, 0, 60))


def test_weaker_legacy_key_does_not_stack(scene):
    """Regression for the double-apply: both keys set must not brighten more
    than the stronger key alone."""
    off = _render(scene, 0, 0).astype(int)
    lift_both = (_render(scene, 20, 60).astype(int) - off).sum()
    lift_single = (_render(scene, 0, 60).astype(int) - off).sum()
    assert lift_both == lift_single
