"""Tests for retouch/freckle.py — FreckleRemover / FreckleClassification."""

import numpy as np
import pytest
import cv2

from retouch.freckle import FreckleRemover, FreckleClassification


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _skin(shape=(240, 240, 3), rgb=(200, 170, 140)):
    """Uniform skin-tone canvas (rgb) as uint8 BGR."""
    return np.full(shape, tuple(reversed(rgb)), dtype=np.uint8)


def _mask(shape=(240, 240)):
    return np.ones(shape, dtype=np.float32)


def _stamp(img, x, y, r, bgr):
    cv2.circle(img, (x, y), r, bgr, -1)


def _freckle_spot(rgb):
    """A reddish, slightly darker-than-skin freckle RGB color."""
    sr, sg, sb = rgb
    fr = min(255, int(sr * 0.75) + 30)
    return (min(255, int(sb * 0.7)), int(sg * 0.7), fr)  # -> BGR


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def test_classify_freckle():
    img = _skin()
    _stamp(img, 120, 120, 2, _freckle_spot((210, 180, 150)))
    cls = FreckleRemover().classify_anomalies(img, _mask(), confidence_threshold=0.0)
    assert cls
    assert cls[0].classification == "freckle"
    assert cls[0].confidence >= 0.6


def test_classify_beauty_mark():
    img = _skin()
    _stamp(img, 120, 120, 4, (20, 15, 12))  # very dark, flat
    cls = FreckleRemover().classify_anomalies(img, _mask(), confidence_threshold=0.0)
    assert cls
    assert cls[0].classification == "beauty_mark"
    assert cls[0].confidence >= 0.6


def test_classify_blemish():
    r = FreckleRemover()
    img = _skin(rgb=(150, 120, 95))
    _stamp(img, 120, 120, 6, (70, 40, 200))  # inflamed red base
    rs = np.random.RandomState(1)
    for _ in range(25):  # speckle -> internal variance (uneven/inflamed)
        x, y = 120 + rs.randint(-5, 6), 120 + rs.randint(-5, 6)
        b = rs.randint(40, 120)
        _stamp(img, x, y, 1, (b, b + 20, b + 120))
    cls = r.classify_anomalies(img, _mask(), confidence_threshold=0.0)
    types = {c.classification for c in cls}
    assert "blemish" in types


def test_classify_noise():
    img = _skin()
    _stamp(img, 120, 120, 1, (60, 80, 120))  # tiny
    cls = FreckleRemover().classify_anomalies(img, _mask(), confidence_threshold=0.0)
    assert cls
    assert cls[0].classification == "noise"


def test_classify_sorted_by_confidence():
    r = FreckleRemover()
    img = _skin()
    _stamp(img, 60, 120, 2, _freckle_spot((210, 180, 150)))
    _stamp(img, 180, 120, 4, (20, 15, 12))  # beauty mark, high confidence
    cls = r.classify_anomalies(img, _mask(), confidence_threshold=0.0)
    confs = [c.confidence for c in cls]
    assert confs == sorted(confs, reverse=True)


def test_fitzpatrick_i_to_vi_freckle_detection_and_classification_is_invariant():
    """The same relative freckle must not disappear on deep skin tones."""
    r = FreckleRemover()
    # Controlled sRGB fixtures spanning Fitzpatrick I-VI-like skin values.
    # The mark is generated from each base using the same relative recipe.
    for rgb in [
        (235, 205, 185), (210, 180, 150), (185, 150, 120),
        (150, 115, 90), (115, 85, 65), (85, 60, 45),
    ]:
        img = _skin(rgb=rgb)
        _stamp(img, 120, 120, 2, _freckle_spot(rgb))
        cls = r.classify_anomalies(img, _mask(), confidence_threshold=0.0)
        assert cls, f"no detection on skin {rgb}"
        assert cls[0].classification == "freckle", f"skin {rgb} -> {cls[0].classification}"


def test_fitzpatrick_i_to_vi_relative_beauty_marks_remain_preserved():
    """A dark identity mark must stay a beauty mark at every tested tone."""
    r = FreckleRemover()
    for rgb in [
        (235, 205, 185), (210, 180, 150), (185, 150, 120),
        (150, 115, 90), (115, 85, 65), (85, 60, 45),
    ]:
        img = _skin(rgb=rgb)
        # Same relative neutral-density mark, expressed in each base tone.
        bgr = tuple(max(1, int(value * 0.20)) for value in reversed(rgb))
        _stamp(img, 120, 120, 4, bgr)
        cls = r.classify_anomalies(img, _mask(), confidence_threshold=0.0)
        assert cls, f"no detection on skin {rgb}"
        assert cls[0].classification == "beauty_mark", (
            f"skin {rgb} -> {cls[0].classification}"
        )


@pytest.mark.parametrize(
    "rgb",
    [
        (235, 205, 185), (210, 180, 150), (185, 150, 120),
        (150, 115, 90), (115, 85, 65), (85, 60, 45),
    ],
)
def test_fitzpatrick_i_to_vi_flat_skin_has_no_false_candidates(rgb):
    assert FreckleRemover().classify_anomalies(
        _skin(rgb=rgb), _mask(), confidence_threshold=0.0
    ) == []


def test_classification_invalid_type_raises():
    with pytest.raises(ValueError):
        FreckleClassification(
            anomaly_id=0, classification="bogus", confidence=0.9, reason="x"
        )


def test_classify_plain_skin_empty():
    img = _skin()
    cls = FreckleRemover().classify_anomalies(img, _mask(), confidence_threshold=0.0)
    assert cls == []


# --------------------------------------------------------------------------- #
# Removal
# --------------------------------------------------------------------------- #
def _scene_freckle_and_mark():
    img = _skin()
    _stamp(img, 80, 120, 2, _freckle_spot((210, 180, 150)))  # freckle
    _stamp(img, 160, 120, 4, (20, 15, 12))  # beauty mark
    return img, _mask()


def test_remove_zero_strength_byte_identical():
    img, m = _scene_freckle_and_mark()
    out = FreckleRemover().remove(img, m, freckle_removal=0)
    assert np.array_equal(out, img)


def test_remove_dtype_uint8_preserved():
    img, m = _scene_freckle_and_mark()
    out = FreckleRemover().remove(img, m, freckle_removal=100, confidence_threshold=0.7)
    assert out.dtype == np.uint8


def test_remove_dtype_float32_preserved():
    img, m = _scene_freckle_and_mark()
    img_f = img.astype(np.float32)
    out = FreckleRemover().remove(img_f, m, freckle_removal=100, confidence_threshold=0.7)
    assert out.dtype == np.float32


def test_remove_freckles_removed_beauty_preserved():
    img, m = _scene_freckle_and_mark()
    before = img.copy()
    out = FreckleRemover().remove(img, m, freckle_removal=100, confidence_threshold=0.7)
    freckle_change = int(
        np.abs(out[120, 80].astype(int) - before[120, 80].astype(int)).sum()
    )
    mark_change = int(
        np.abs(out[120, 160].astype(int) - before[120, 160].astype(int)).sum()
    )
    assert freckle_change > 0, "freckle should be healed"
    assert mark_change == 0, "beauty mark should be preserved"


def test_light_skin_removal_preserves_fixture_behavior():
    """S1 tone adaptation must preserve the established light-skin behavior."""
    img, m = _scene_freckle_and_mark()
    out = FreckleRemover().remove(
        img, m, freckle_removal=100, confidence_threshold=0.7
    )
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert not np.array_equal(out[120, 80], img[120, 80])
    assert np.array_equal(out[120, 160], img[120, 160])


def test_remove_user_preserve_mask_override():
    img, m = _scene_freckle_and_mark()
    before = img.copy()
    pm = np.zeros((240, 240), dtype=np.float32)
    cv2.circle(pm, (80, 120), 6, 1.0, -1)  # protect the freckle
    out = FreckleRemover().remove(
        img, m, freckle_removal=100, freckle_preserve_mask=pm, confidence_threshold=0.7
    )
    change = int(np.abs(out[120, 80].astype(int) - before[120, 80].astype(int)).sum())
    assert change == 0, "user preserve_mask must protect the freckle"


def test_remove_no_anomalies_noop():
    img = _skin()  # plain skin, nothing to remove
    out = FreckleRemover().remove(img, _mask(), freckle_removal=100)
    assert np.array_equal(out, img)


def test_remove_low_strength_preserves_freckle():
    img, m = _scene_freckle_and_mark()
    before = img.copy()
    out = FreckleRemover().remove(img, m, freckle_removal=10, confidence_threshold=0.7)
    change = int(np.abs(out[120, 80].astype(int) - before[120, 80].astype(int)).sum())
    assert change == 0, "very low strength should not heal freckles"


# --------------------------------------------------------------------------- #
# Tie/near-tie resolution (dictionary-order independence fix)
#
# docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md: two drawn
# eyeliner components on a real face (DSCF2310) score beauty_mark=0.70 and
# blemish=0.70 exactly -- an unresolved tie that previously resolved to
# beauty_mark (-> mole downstream, via marks.py's _FRECKLE_CLASS_MAP) purely
# because "beauty_mark" appears before "blemish" in the scores dict literal.
# The fix narrows to exactly this case: a "beauty_mark" win within
# _TIE_MARGIN of a competing score routes to "ambiguous" instead. Freckle/
# blemish ties (which also occur legitimately -- their rule weights both
# sum to exactly 1.00) are untouched, since neither produces a false "mole".
# --------------------------------------------------------------------------- #

def _dscf2310_style_scene():
    """End-to-end scene (real classify_anomalies pipeline, not just the
    scoring function) reproducing the audit's exact score pattern
    (beauty_mark=0.70, blemish=0.70) for a thin, elongated, internally
    two-toned dark stroke -- not a round, uniform mole. Built by
    round-tripping the audit's actual LAB skin baseline (L=185, a=142,
    b=128) and two neutral-chroma dark shades through LAB2BGR, so the
    component genuinely has near-zero chroma (like a matte drawn line)
    and internal L variance (two shades) without touching a/b -- the
    exact combination that ties beauty_mark's dark-flat rule against
    blemish's uneven-variance rule."""
    h, w = 200, 200
    skin_lab = np.zeros((h, w, 3), dtype=np.uint8)
    skin_lab[:] = (185, 142, 128)
    img = cv2.cvtColor(skin_lab, cv2.COLOR_LAB2BGR)

    stroke_lab_dark1 = np.array([60, 142, 128], dtype=np.uint8).reshape(1, 1, 3)
    stroke_lab_dark2 = np.array([100, 142, 128], dtype=np.uint8).reshape(1, 1, 3)
    c1 = cv2.cvtColor(stroke_lab_dark1, cv2.COLOR_LAB2BGR)[0, 0]
    c2 = cv2.cvtColor(stroke_lab_dark2, cv2.COLOR_LAB2BGR)[0, 0]
    img[95:97, 80:130] = c1
    img[97:100, 80:130] = c2

    # Mild background texture so global a_std/l_std aren't degenerate zero
    # (a uniform canvas makes a_norm blow up unrealistically -- the same
    # scene-construction pitfall this session's FA-01/FA-02 experiments
    # caught with insufficient synthetic noise).
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 3.0, (h, w)).astype(np.float32)
    stroke_area = np.zeros((h, w), dtype=bool)
    stroke_area[95:100, 80:130] = True
    img_f = img.astype(np.float32)
    img_f[~stroke_area] += noise[~stroke_area][:, None]
    return np.clip(img_f, 0, 255).astype(np.uint8)


class TestTieBreakDoesNotSilentlyResolveToMole:
    def test_dscf2310_style_elongated_stroke_is_ambiguous_not_mole(self):
        """Reproduces the exact confirmed DSCF2310 failure end-to-end
        through classify_anomalies (not just the scoring function)."""
        img = _dscf2310_style_scene()
        cls = FreckleRemover().classify_anomalies(img, np.ones((200, 200), dtype=np.float32), confidence_threshold=0.0)
        stroke = [c for c in cls if c.bbox[2] * c.bbox[3] > 20]
        assert stroke, "expected the elongated stroke to be detected as a component"
        assert stroke[0].classification == "ambiguous", (
            f"an elongated, internally two-toned component tied between "
            f"beauty_mark and blemish must not silently become a confident "
            f"mole; got {stroke[0].classification} ({stroke[0].reason})"
        )
        assert stroke[0].confidence > 0.0

    @staticmethod
    def _tied_component_lab():
        """Builds a component whose score pattern is exactly
        beauty_mark=0.70 / blemish=0.70 (the audit's real DSCF2310
        pattern): area in both bands, L_norm<-2 fires (beauty's dark
        bonus) and a_norm/chroma stay inside neutral range (so beauty's
        a_norm bonus and blemish's chroma bonus do NOT fire), and internal
        L variance is high (blemish's uneven bonus fires). Two interleaved
        L shades give internal std without touching a/b."""
        side = 10
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        comp_mask = np.zeros((side, side), dtype=bool)
        comp_mask[:5, :] = True  # area 50: inside beauty (10-70) and blemish (6-80)
        r = FreckleRemover()
        lab = r._to_lab(canvas).astype(np.float32)
        lab[:2, :, 0] = 60.0
        lab[2:5, :, 0] = 100.0
        lab[:5, :, 1] = 142.0  # a at median -> a_norm ~ 0 (neither >0.3 nor <-0.3)
        lab[:5, :, 2] = 128.0  # b neutral -> chroma stays low
        return r, lab, comp_mask

    def test_direct_tie_beauty_mark_and_blemish_resolves_to_ambiguous(self):
        """Isolates the exact score pattern from the research audit
        (beauty_mark=.70, blemish=.70) via a component built to hit those
        specific rule thresholds, bypassing image-level detection so the
        tie-break itself is what's under test."""
        r, lab, comp_mask = self._tied_component_lab()
        cls, conf, reason, area, _, _ = r._classify_anomaly(
            lab, comp_mask, a_median=142.0, a_std=3.74, l_median=185.0, l_std=34.34,
        )
        assert cls == "ambiguous", f"expected ambiguous tie, got {cls} ({reason})"
        assert conf > 0.0, "ambiguous must not be silently dropped via zero confidence"

    def test_ambiguous_confidence_clears_the_default_threshold(self):
        """An ambiguous result must carry the contested score as its
        confidence, not 0.0 -- otherwise classify_anomalies's default
        0.6 threshold would silently drop it, defeating the point of an
        explicit outcome (must be visible, not disappeared)."""
        r, lab, comp_mask = self._tied_component_lab()
        cls, conf, reason, area, _, _ = r._classify_anomaly(
            lab, comp_mask, a_median=142.0, a_std=3.74, l_median=185.0, l_std=34.34,
        )
        assert cls == "ambiguous"
        assert conf >= 0.6, f"ambiguous confidence {conf} would be dropped by the default threshold"

    def test_ambiguous_is_a_valid_classification_type(self):
        FreckleClassification(anomaly_id=0, classification="ambiguous", confidence=0.7, reason="tie")

    def test_ambiguous_maps_to_unknown_mark_class_without_crashing(self):
        """marks.py's _FRECKLE_CLASS_MAP must have an entry for every
        FreckleRemover classification value; a missing entry would raise
        KeyError the first time an ambiguous result reaches detect_marks."""
        from retouch.marks import _FRECKLE_CLASS_MAP
        from retouch.freckle import _CLASS_TYPES

        missing = _CLASS_TYPES - set(_FRECKLE_CLASS_MAP)
        assert not missing, f"_FRECKLE_CLASS_MAP is missing entries for: {missing}"
        assert _FRECKLE_CLASS_MAP["ambiguous"] == "unknown"


class TestOrdinaryNonTiedClassificationsAreUnchanged:
    """The legacy tie-break order (freckle, beauty_mark, blemish, noise)
    is preserved exactly for every score pattern except a beauty_mark win
    within _TIE_MARGIN of a competitor -- these tests use fixtures already
    exercised in this file to confirm classification is unaffected."""

    def test_clear_freckle_is_still_freckle(self):
        img = _skin()
        _stamp(img, 120, 120, 2, _freckle_spot((210, 180, 150)))
        cls = FreckleRemover().classify_anomalies(img, _mask(), confidence_threshold=0.0)
        assert cls[0].classification == "freckle"

    def test_clear_beauty_mark_is_still_beauty_mark(self):
        img = _skin()
        _stamp(img, 120, 120, 4, (20, 15, 12))
        cls = FreckleRemover().classify_anomalies(img, _mask(), confidence_threshold=0.0)
        assert cls[0].classification == "beauty_mark"

    def test_clear_blemish_is_still_blemish(self):
        r = FreckleRemover()
        img = _skin(rgb=(150, 120, 95))
        _stamp(img, 120, 120, 6, (70, 40, 200))
        rs = np.random.RandomState(1)
        for _ in range(25):
            x, y = 120 + rs.randint(-5, 6), 120 + rs.randint(-5, 6)
            b = rs.randint(40, 120)
            _stamp(img, x, y, 1, (b, b + 20, b + 120))
        cls = r.classify_anomalies(img, _mask(), confidence_threshold=0.0)
        types = {c.classification for c in cls}
        assert "blemish" in types

    def test_freckle_blemish_exact_tie_still_resolves_to_freckle_not_ambiguous(self):
        """A freckle/blemish exact tie is legacy, legitimate behavior (both
        rule-weight sums total exactly 1.00 by construction: 0.5+0.45+0.05
        and 0.2+0.3+0.5) and produces no false mole either way -- only
        beauty_mark ties are narrowed to ambiguous. This is the exact
        pattern that exposed the over-broad first draft of this fix
        (routing every tie to ambiguous broke freckle healing); verified
        here as a genuine 1.0/1.0 score tie via _classify_anomaly directly,
        not merely a component that happens to end up classified freckle
        for an unrelated reason."""
        r = FreckleRemover()
        side = 8
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        comp_mask = np.zeros((side, side), dtype=bool)
        comp_mask.flat[:12] = True  # area 12: inside freckle(4-25) and blemish(6-80)
        lab = r._to_lab(canvas).astype(np.float32)
        # Reddish (a_norm>0.3, fires freckle's bonus), L_norm>=-2 (freckle's
        # small bonus), AND high chroma + high internal L variance (fires
        # both of blemish's remaining bonuses) -- two interleaved L shades
        # for the variance, without touching a/b enough to break either
        # rule's own threshold.
        lab[:6, :, 0] = 170.0
        lab[6:, :, 0] = 140.0
        lab[:, :, 1] = 150.0  # a above median+0.3*std -> fires freckle's a_norm>0.3
        lab[:, :, 2] = 160.0  # pushes chroma above 25 for blemish
        cls, conf, reason, area, _, _ = r._classify_anomaly(
            lab, comp_mask, a_median=142.0, a_std=3.74, l_median=185.0, l_std=34.34,
        )
        assert cls == "freckle", (
            f"a genuine freckle/blemish tie must keep resolving to freckle "
            f"(legacy order), not be narrowed to ambiguous; got {cls} ({reason})"
        )

    def test_beauty_mark_noise_tie_on_tiny_component_also_routes_to_ambiguous(self):
        """A second real tie path found via the before/after production
        diff on DSCF2310 (bbox (398,592,2,2) and two more on DSCF2306):
        a component below the noise-area cutoff can still score
        beauty_mark=0.9 (from a_norm+L_norm bonuses alone, no area bonus
        needed) exactly tying noise=0.9. This also must not silently
        become a confident mole."""
        r = FreckleRemover()
        side = 6
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        comp_mask = np.zeros((side, side), dtype=bool)
        comp_mask.flat[:3] = True  # area 3 < _NOISE_MAX_AREA (4)
        lab = r._to_lab(canvas).astype(np.float32)
        lab[:, :, 0] = 60.0   # dark -> L_norm<-2 fires beauty's +0.5
        lab[:, :, 1] = 118.0  # a_norm<-0.3 fires beauty's +0.4 (0.4+0.5=0.9)
        lab[:, :, 2] = 128.0
        cls, conf, reason, area, _, _ = r._classify_anomaly(
            lab, comp_mask, a_median=142.0, a_std=3.74, l_median=185.0, l_std=34.34,
        )
        assert cls == "ambiguous", f"expected ambiguous beauty_mark/noise tie, got {cls} ({reason})"
        assert conf > 0.0

    def test_fitzpatrick_freckle_invariance_still_holds(self):
        r = FreckleRemover()
        for rgb in [
            (235, 205, 185), (210, 180, 150), (185, 150, 120),
            (150, 115, 90), (115, 85, 65), (85, 60, 45),
        ]:
            img = _skin(rgb=rgb)
            _stamp(img, 120, 120, 2, _freckle_spot(rgb))
            cls = r.classify_anomalies(img, _mask(), confidence_threshold=0.0)
            assert cls[0].classification == "freckle", f"skin {rgb} -> {cls[0].classification}"

    def test_freckle_healing_and_beauty_mark_preservation_still_work(self):
        """The exact scenario the over-broad first draft of this fix broke
        (freckle_change == 0 because the freckle/blemish tie was routed to
        ambiguous, which remove() never heals): must still heal."""
        img, m = _scene_freckle_and_mark()
        before = img.copy()
        out = FreckleRemover().remove(img, m, freckle_removal=100, confidence_threshold=0.7)
        freckle_change = int(np.abs(out[120, 80].astype(int) - before[120, 80].astype(int)).sum())
        mark_change = int(np.abs(out[120, 160].astype(int) - before[120, 160].astype(int)).sum())
        assert freckle_change > 0, "freckle should still be healed"
        assert mark_change == 0, "beauty mark should still be preserved"

    def test_light_skin_removal_is_deterministic_within_runtime(self):
        """The same light-skin fixture produces stable bytes in one runtime."""
        img, m = _scene_freckle_and_mark()
        remover = FreckleRemover()
        out = remover.remove(
            img, m, freckle_removal=100, confidence_threshold=0.7
        )
        repeat = remover.remove(
            img, m, freckle_removal=100, confidence_threshold=0.7
        )
        np.testing.assert_array_equal(out, repeat)
