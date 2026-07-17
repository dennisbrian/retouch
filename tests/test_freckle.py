"""Tests for retouch/freckle.py — FreckleRemover / FreckleClassification."""

import hashlib
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


def test_light_skin_removal_output_is_byte_identical():
    """S1 tone adaptation must not change the established light-skin fixture."""
    img, m = _scene_freckle_and_mark()
    out = FreckleRemover().remove(
        img, m, freckle_removal=100, confidence_threshold=0.7
    )
    assert hashlib.sha256(out.tobytes()).hexdigest() == (
        "681455108f8be4caf0644ff23544b44316adb409cbc9fa62d6f53cabf6fb0e45"
    )


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
