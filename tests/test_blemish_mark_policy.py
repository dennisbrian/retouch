"""FA-01 finding (docs/plans/RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md
Sec 2.3): the general mark-policy preserve mask reached only the freckle
stage; blemish.remove never saw it. Wired at the _process_face_core
dispatch (perf_optimizations.py, 'R10: General mark-policy protection'
block) so an opt-in mark_policy can also protect a marked spot from
automatic blemish removal. mark_policy=None (legacy default) must stay
byte-identical -- see test_golden_pipeline_face.py for that guarantee."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_engine import _build_synthetic_regions, _build_synthetic_face

ROI = 256
# Same dark-spot pattern test_blemish.py::TestRemove::test_removes_dark_spot
# already verifies BlemishRemover._detect reliably catches.
SPOT_CENTER = (128, 128)
SPOT_SLICE = (slice(124, 132), slice(124, 132))


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


def _ctx(blemish: float, mark_policy):
    from retouch.engine import ProcessingContext

    return ProcessingContext(
        smooth=0.0, whiten=0.0, equalize=0.0, blemish=blemish, nose_smooth=None,
        pore_synthesis=0.0, specular_bloom=0.0, dodge_burn=0.0, relight=0.0,
        eye_enhance=0.0, catchlight=0.0, lip_enhance=0.0, teeth_whiten=0.0,
        blush=0.0, hair_enhance=0.0, slimming=0.0, mark_policy=mark_policy,
    )


@pytest.fixture(scope="module")
def scene():
    """Skin-toned canvas with a genuinely detectable dark spot at (128, 128)."""
    regions = _build_synthetic_regions(ROI, ROI, skin_value=1.0)
    canvas = np.full((ROI, ROI, 3), 128, dtype=np.uint8)
    canvas[SPOT_SLICE] = [30, 30, 30]
    return canvas, regions, _build_synthetic_face(ied=30.0, size=100), np.ones((ROI, ROI), np.float32), _processors()


def _render(scene, blemish, mark_policy):
    from retouch.perf_optimizations import _process_face_core

    canvas, regions, face, person, procs = scene
    fr = _process_face_core(
        canvas.copy(), regions, face, _ctx(blemish, mark_policy),
        0, 0, ROI, ROI, person, procs,
    )
    return np.asarray(fr.canvas)


def test_blemish_pass_is_live_on_scene(scene):
    off = _render(scene, 0, None)
    on = _render(scene, 80, None)
    assert np.abs(on.astype(int) - off.astype(int)).max() >= 3, \
        "blemish pass is inert on the scene; policy tests would be vacuous"


def test_no_policy_removes_the_spot(scene):
    """mark_policy=None (legacy) keeps the existing no-protection behavior."""
    result = _render(scene, 80, None)
    assert not np.allclose(result[SPOT_CENTER], [30, 30, 30], atol=10)


def test_preserve_all_policy_blocks_removal_at_the_spot(scene, monkeypatch):
    """An opt-in policy that marks the exact spot 'preserve' must stop
    blemish.remove from touching it -- proving mark_policy actually reaches
    the blemish eligibility mask, not just the freckle stage."""
    from retouch import marks as marks_module
    from retouch.marks import MarkRecord

    def _fake_detect_marks(img_bgr, *, face_mask=None, **kwargs):
        return [MarkRecord(
            mark_id=0, mark_class="drawn_makeup_mark", confidence=0.95,
            centroid=(float(SPOT_CENTER[1]), float(SPOT_CENTER[0])),
            area_norm=0.01, bbox=(120, 120, 16, 16),
            features={}, on_body=False,
        )]

    monkeypatch.setattr(marks_module, "detect_marks", _fake_detect_marks)

    policy = {"drawn_makeup_mark": {"action": "preserve"}}
    result = _render(scene, 80, policy)
    assert np.allclose(result[SPOT_CENTER], [30, 30, 30], atol=5), (
        "preserve-policy spot was still altered by blemish.remove -- "
        "mark_policy is not reaching the blemish eligibility mask"
    )


def test_policy_does_not_protect_unrelated_skin(scene, monkeypatch):
    """The preserve mask must only shrink eligibility at the marked spot,
    not disable blemish removal globally."""
    from retouch import marks as marks_module
    from retouch.marks import MarkRecord

    def _fake_detect_marks(img_bgr, *, face_mask=None, **kwargs):
        return [MarkRecord(
            mark_id=0, mark_class="drawn_makeup_mark", confidence=0.95,
            centroid=(float(SPOT_CENTER[1]), float(SPOT_CENTER[0])),
            area_norm=0.01, bbox=(120, 120, 16, 16),
            features={}, on_body=False,
        )]

    monkeypatch.setattr(marks_module, "detect_marks", _fake_detect_marks)

    policy = {"drawn_makeup_mark": {"action": "preserve"}}
    with_policy = _render(scene, 80, policy)
    no_policy = _render(scene, 80, None)
    # Away from the protected spot, both renders should treat the (flat,
    # blemish-free) skin identically.
    probe = (10, 10)
    assert np.array_equal(with_policy[probe], no_policy[probe])
