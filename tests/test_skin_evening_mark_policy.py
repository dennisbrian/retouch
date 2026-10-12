"""FA-01 finding (docs/plans/RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md
Sec 2.3), second tranche: the general mark-policy preserve mask now also
protects the 9 variance-reducing/evening skin ops (flatten, micro_dodge_burn,
redness_even, hb_even, hemoglobin_smooth, vein_attenuate, equalize,
unify_hue_line, unify_tone) -- wired at the top of _process_face_core as a
single, feathered skin_n_marks_protected mask computed once on the pristine
canvas_original.

Deliberately NOT extended to the global shading/tone ops (whiten, relight,
sculpt, shine_removal, specular_finish/bloom, face_exposure, apply_sss,
quantize_tones, hb_shift): excluding a marked spot from a whole-face shading
change creates a hard/soft-edged island artifact rather than protecting
anything, since (unlike a heal) these ops re-render the surrounding pixels
too. See the module-level comment in perf_optimizations.py for the full
rationale. This file only tests the 9 evening ops; blemish/freckle coverage
is tested in test_blemish_mark_policy.py and test_marks.py.

mark_policy=None (legacy default) must stay byte-identical -- see
test_golden_pipeline_face.py for that guarantee."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_engine import _build_synthetic_regions, _build_synthetic_face

ROI = 256
# A patch of skin whose a/b chroma is deliberately far from the face median,
# so skin.equalize's median-pull has real work to do there.
SPOT_CENTER = (128, 128)
SPOT_SLICE = (slice(120, 136), slice(120, 136))


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


def _ctx(equalize: float, mark_policy):
    from retouch.engine import ProcessingContext

    return ProcessingContext(
        smooth=0.0, whiten=0.0, equalize=equalize, blemish=0.0, nose_smooth=None,
        pore_synthesis=0.0, specular_bloom=0.0, dodge_burn=0.0, relight=0.0,
        eye_enhance=0.0, catchlight=0.0, lip_enhance=0.0, teeth_whiten=0.0,
        blush=0.0, hair_enhance=0.0, slimming=0.0, mark_policy=mark_policy,
    )


@pytest.fixture(scope="module")
def scene():
    """Skin-toned canvas with a distinct-chroma patch at (128, 128) that
    skin.equalize's a/b median pull will visibly change when eligible."""
    regions = _build_synthetic_regions(ROI, ROI, skin_value=1.0)
    canvas = np.full((ROI, ROI, 3), (150, 175, 210), dtype=np.uint8)  # BGR skin tone
    canvas[SPOT_SLICE] = (110, 140, 170)  # shifted chroma, same skin mask
    return canvas, regions, _build_synthetic_face(ied=30.0, size=100), np.ones((ROI, ROI), np.float32), _processors()


def _render(scene, equalize, mark_policy):
    from retouch.perf_optimizations import _process_face_core

    canvas, regions, face, person, procs = scene
    fr = _process_face_core(
        canvas.copy(), regions, face, _ctx(equalize, mark_policy),
        0, 0, ROI, ROI, person, procs,
    )
    return np.asarray(fr.canvas)


def test_equalize_pass_is_live_on_scene(scene):
    off = _render(scene, 0, None)
    on = _render(scene, 90, None)
    assert np.abs(on.astype(int) - off.astype(int))[SPOT_SLICE].max() >= 3, \
        "equalize pass is inert on the scene; policy tests would be vacuous"


def test_no_policy_pulls_the_spot_toward_face_median(scene):
    """mark_policy=None (legacy) keeps the existing no-protection behavior."""
    off = _render(scene, 0, None)
    on = _render(scene, 90, None)
    assert not np.array_equal(on[SPOT_CENTER], off[SPOT_CENTER])


def test_preserve_policy_blocks_equalize_at_the_marked_spot(scene, monkeypatch):
    """An opt-in policy that marks the exact spot 'preserve' must stop
    equalize's a/b pull from touching it -- proving mark_policy reaches the
    evening-ops mask, not just blemish/freckle."""
    from retouch import marks as marks_module
    from retouch.marks import MarkRecord

    def _fake_detect_marks(img_bgr, *, face_mask=None, **kwargs):
        return [MarkRecord(
            mark_id=0, mark_class="drawn_makeup_mark", confidence=0.95,
            centroid=(float(SPOT_CENTER[1]), float(SPOT_CENTER[0])),
            area_norm=0.01, bbox=(118, 118, 20, 20),
            features={}, on_body=False,
        )]

    monkeypatch.setattr(marks_module, "detect_marks", _fake_detect_marks)

    policy = {"drawn_makeup_mark": {"action": "preserve"}}
    off = _render(scene, 0, None)
    with_policy = _render(scene, 90, policy)
    assert np.array_equal(with_policy[SPOT_CENTER], off[SPOT_CENTER]), (
        "preserve-policy spot was still altered by equalize -- mark_policy "
        "is not reaching the evening-ops eligibility mask"
    )


def test_policy_does_not_protect_unrelated_skin(scene, monkeypatch):
    """The preserve mask must only shrink eligibility near the marked spot,
    not disable equalize globally."""
    from retouch import marks as marks_module
    from retouch.marks import MarkRecord

    def _fake_detect_marks(img_bgr, *, face_mask=None, **kwargs):
        return [MarkRecord(
            mark_id=0, mark_class="drawn_makeup_mark", confidence=0.95,
            centroid=(float(SPOT_CENTER[1]), float(SPOT_CENTER[0])),
            area_norm=0.01, bbox=(118, 118, 20, 20),
            features={}, on_body=False,
        )]

    monkeypatch.setattr(marks_module, "detect_marks", _fake_detect_marks)

    policy = {"drawn_makeup_mark": {"action": "preserve"}}
    with_policy = _render(scene, 90, policy)
    no_policy = _render(scene, 90, None)
    probe = (20, 20)  # far from the marked spot, same flat skin tone
    assert np.array_equal(with_policy[probe], no_policy[probe])


def test_hb_even_is_protected_but_hb_shift_is_not(scene, monkeypatch):
    """hb_even (evening op) must receive the smaller, mark-protected mask;
    hb_shift (deliberate exclusion -- a global shading op, not an evening
    op) must keep receiving the full, unprotected skin mask. Captured at
    the actual call boundary rather than asserted via source text, so a
    rename can't make this pass vacuously; and captured rather than judged
    by render diff, since shift_hemoglobin is itself close to a no-op on a
    flat two-tone synthetic canvas (its magnitude derives from the
    subject's own hemoglobin IQR, which collapses on a near-uniform image)."""
    from retouch import chromophore_v2 as chromophore_module
    from retouch import marks as marks_module
    from retouch.marks import MarkRecord

    def _fake_detect_marks(img_bgr, *, face_mask=None, **kwargs):
        return [MarkRecord(
            mark_id=0, mark_class="drawn_makeup_mark", confidence=0.95,
            centroid=(float(SPOT_CENTER[1]), float(SPOT_CENTER[0])),
            area_norm=0.01, bbox=(118, 118, 20, 20),
            features={}, on_body=False,
        )]

    monkeypatch.setattr(marks_module, "detect_marks", _fake_detect_marks)

    seen = {}
    _orig_reduce = chromophore_module.reduce_hemoglobin_variance
    _orig_shift = chromophore_module.shift_hemoglobin

    def _cap_reduce(img, strength, *, skin_mask=None, decomposition=None):
        seen["even_mask"] = skin_mask
        return _orig_reduce(img, strength, skin_mask=skin_mask, decomposition=decomposition)

    def _cap_shift(img, shift, *, skin_mask=None, decomposition=None):
        seen["shift_mask"] = skin_mask
        return _orig_shift(img, shift, skin_mask=skin_mask, decomposition=decomposition)

    monkeypatch.setattr(chromophore_module, "reduce_hemoglobin_variance", _cap_reduce)
    monkeypatch.setattr(chromophore_module, "shift_hemoglobin", _cap_shift)

    from retouch.engine import ProcessingContext
    from retouch.perf_optimizations import _process_face_core

    canvas, regions, face, person, procs = scene
    ctx = ProcessingContext(
        smooth=0.0, whiten=0.0, equalize=0.0, blemish=0.0, nose_smooth=None,
        pore_synthesis=0.0, specular_bloom=0.0, dodge_burn=0.0, relight=0.0,
        eye_enhance=0.0, catchlight=0.0, lip_enhance=0.0, teeth_whiten=0.0,
        blush=0.0, hair_enhance=0.0, slimming=0.0,
        mark_policy={"drawn_makeup_mark": {"action": "preserve"}},
        hb_even=0.5, hb_shift=0.5,
    )
    _process_face_core(canvas.copy(), regions, face, ctx, 0, 0, ROI, ROI, person, procs)

    assert "even_mask" in seen and "shift_mask" in seen
    even_coverage = float((seen["even_mask"] > 0.3).sum())
    shift_coverage = float((seen["shift_mask"] > 0.3).sum())
    assert even_coverage < shift_coverage, (
        "hb_even's mask must be smaller than hb_shift's (mark-protected vs. "
        "unprotected) -- got even_coverage=%r, shift_coverage=%r"
        % (even_coverage, shift_coverage)
    )
    # And shift's mask must be the plain skin mask, untouched by policy.
    assert shift_coverage == float((regions.skin > 0.3).sum())
