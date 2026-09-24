"""``flyaway_cleanup`` (hair.flyaway_cleanup) aliases to the same H1
``hairwork.remove_flyaways`` dispatch as ``hair_remove_flyaways``
(hair.remove_flyaways), max-combined (e73d3ba pattern), rather than calling
``hair.py::cleanup_flyaway_strands`` directly.

Restored 2026-09-24: flyaway_cleanup's original call site (32084c0,
2026-07-22) called cleanup_flyaway_strands, whose "allowed zone" excludes
only a ~41px ring around the hair silhouette — everywhere else (clothing
embroidery, background texture) is fair game to it. That call site was
deleted 2026-08-17 (aaaa1e5) as believed-dead A/B residue; restoring it
verbatim and testing on a real cosplay render produced visible fabric/
texture smearing outside the hair region. Aliasing to hair_remove_flyaways'
dispatch instead reuses the live, hair-mask-scoped, tested implementation
that 4 recipes already use.
"""
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


def _ctx(hair_remove_flyaways: float, flyaway_cleanup: float):
    from retouch.engine import ProcessingContext

    return ProcessingContext(
        smooth=0.0, whiten=0.0, equalize=0.0, blemish=0.0, nose_smooth=None,
        pore_synthesis=0.0, specular_bloom=0.0, dodge_burn=0.0, relight=0.0,
        eye_enhance=0.0, catchlight=0.0, lip_enhance=0.0, teeth_whiten=0.0,
        blush=0.0, hair_enhance=0.0, slimming=0.0,
        hair_remove_flyaways=hair_remove_flyaways, flyaway_cleanup=flyaway_cleanup,
    )


@pytest.fixture(scope="module")
def scene():
    """Skin-toned canvas with a hair region containing vertical strands
    (matches test_flyaway_removal.py's synthetic-strand pattern) plus a thin
    diagonal flyaway crossing them, so the H1 detector has something real to
    find."""
    regions = _build_synthetic_regions(ROI, ROI, skin_value=1.0)
    hair = np.zeros((ROI, ROI), np.float32)
    cv2.rectangle(hair, (40, 20), (ROI - 40, 140), 1.0, -1)
    regions.hair = hair

    canvas = np.full((ROI, ROI, 3), 80, dtype=np.uint8)
    stripe_w = 6
    for x in range(0, ROI, 2 * stripe_w):
        canvas[:, x: x + stripe_w] = 160
    # thin diagonal flyaway strand crossing the vertical strands
    cv2.line(canvas, (30, 10), (110, 150), (20, 20, 20), 2)
    return canvas, regions, _build_synthetic_face(ied=30.0, size=100), np.ones((ROI, ROI), np.float32), _processors()


def _render(scene, hair_remove_flyaways, flyaway_cleanup):
    from retouch.perf_optimizations import _process_face_core

    canvas, regions, face, person, procs = scene
    fr = _process_face_core(
        canvas.copy(), regions, face, _ctx(hair_remove_flyaways, flyaway_cleanup),
        0, 0, ROI, ROI, person, procs,
    )
    return np.asarray(fr.canvas)


def test_flyaway_pass_is_live_on_scene(scene):
    off = _render(scene, 0, 0)
    on = _render(scene, 35, 0)
    assert np.abs(on.astype(int) - off.astype(int)).max() >= 3, \
        "flyaway removal is inert on the scene; alias tests would be vacuous"


def test_flyaway_cleanup_is_alias_of_hair_remove_flyaways(scene):
    assert np.array_equal(_render(scene, 35, 0), _render(scene, 0, 35))


def test_both_keys_equal_single_pass_at_max(scene):
    both = _render(scene, 10, 35)
    single = _render(scene, 0, 35)
    assert np.array_equal(both, single)


def test_flyaway_cleanup_does_not_touch_out_of_hair_region(scene):
    """The whole point of the alias: flyaway_cleanup must stay scoped to the
    hair mask, unlike the broken cleanup_flyaway_strands call site it
    replaces (which fired on ~everything outside a ~41px hair ring)."""
    canvas, regions, face, person, procs = scene
    off = _render(scene, 0, 0)
    on = _render(scene, 0, 60)
    # Bottom strip of the canvas is well outside the synthetic hair
    # rectangle (hair covers rows 20-140 of a 256-row canvas).
    outside = slice(180, 256)
    assert np.array_equal(off[outside], on[outside])
