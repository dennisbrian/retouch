"""Experimental texture must use smoothing loss, never undo later healing.

Eligibility is deliberately relaxed in the pipeline probe so it exercises the
candidate branch. This tests operation ordering/support, not texture quality.
"""

import cv2
import numpy as np
import pytest

from retouch import fa02_texture_eligibility as elig


@pytest.fixture(scope="module")
def boundary_engine():
    from retouch.engine import RetouchEngine
    engine = RetouchEngine()
    yield engine
    engine.close()


@pytest.mark.parametrize("mode", ["dog", "multiscale"])
def test_later_healing_is_not_texture_and_its_footprint_is_excluded(boundary_engine, monkeypatch, mode):
    from retouch.frequency import FrequencySeparator
    from retouch.freckle import FreckleRemover
    from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

    captured = {}
    combine = FrequencySeparator.combine
    evaluate = elig.evaluate_face_eligibility
    extract = elig.extract_detail

    def smoothing(self, *args, **kwargs):
        out = combine(self, *args, **kwargs)
        captured["smoothed"] = out.copy()
        return out

    def heal(self, img_bgr, face_mask=None, **kwargs):
        # Controlled correction inside eligible skin; no detector assumptions.
        out = img_bgr.copy()
        ys, xs = np.where(np.asarray(face_mask) > 0.9)
        y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
        edited = np.zeros(out.shape[:2], bool)
        edited[y-2:y+3, x-2:x+3] = True
        out[edited] = np.clip(out[edited] + 25, 0, 255)
        captured["edited"] = edited
        return out

    def gate(before, support, face_width, protected_mask=None):
        captured["before"] = before.copy()
        captured["support"] = support.copy()
        return evaluate(before, support, face_width, protected_mask,
                        min_face_width_px=1, min_highpass_std=0,
                        max_noise_sigma=1e6, min_support_px=1)

    def detail(residual, selected_mode, face_width):
        captured["residual"] = residual.copy()
        return extract(residual, selected_mode, face_width)

    monkeypatch.setattr(FrequencySeparator, "combine", smoothing)
    monkeypatch.setattr(FreckleRemover, "remove", heal)
    monkeypatch.setattr(elig, "evaluate_face_eligibility", gate)
    monkeypatch.setattr(elig, "extract_detail", detail)
    img = make_synthetic_face_image()
    h, w = img.shape[:2]
    result = boundary_engine.process(
        img, recipe="natural", face_contexts=[make_face_context(w, h, img)],
        fa02_texture_mode=mode, freckle_removal=100,
    )
    assert result.fa02_diagnostics[0]["eligible"]
    assert "residual" in captured, "candidate must actually execute"
    assert np.allclose(captured["residual"], captured["before"] - captured["smoothed"], atol=1e-6)
    assert np.all(captured["support"][captured["edited"]] == 0)


@pytest.mark.parametrize("mode", ["dog", "multiscale"])
@pytest.mark.parametrize("face_width", [250.0, 500.0, 1000.0])
def test_corrected_detail_cannot_return_in_healed_pixel_or_filter_ring(mode, face_width):
    # One photographed-like dark detail that smoothing partly removed, then a
    # later repair removed completely. It must not return from the residual.
    source = np.full((96, 96, 3), 140, np.float32)
    source[46:50, 46:50] = 70
    smooth = cv2.GaussianBlur(source, (0, 0), 1.0)
    residual = source - smooth
    detail = elig.extract_detail(residual, mode, face_width)
    corrected = np.zeros((96, 96), bool)
    corrected[46:50, 46:50] = True
    allowed = np.full((96, 96), 0.7, np.float32)
    guarded, radius = elig.exclude_corrected_support(allowed, corrected, mode, face_width)
    expected = cv2.dilate(corrected.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) != 0
    assert np.all(guarded[expected] == 0)
    assert np.array_equal(guarded[~expected], allowed[~expected])
    # At the sampling floor DoG is a declared no-op, not a signal-recovery win.
    if 0 not in elig.collapsed_bands(face_width) or mode == "multiscale":
        assert np.max(np.abs(detail[corrected])) > 0
    repaired = smooth.copy()
    repaired[corrected] = 140
    output = repaired + detail * guarded[..., None] * 0.1
    assert np.array_equal(output[expected], repaired[expected])


def test_correction_support_refuses_a_different_coordinate_grid():
    with pytest.raises(ValueError, match="match the 2D"):
        elig.exclude_corrected_support(np.ones((10, 20)), np.ones((20, 10)), "dog", 500)


def test_no_correction_keeps_support_unchanged():
    allowed = np.full((12, 14), 0.6, np.float32)
    got, radius = elig.exclude_corrected_support(allowed, np.zeros_like(allowed), "multiscale", 500)
    assert got is allowed and radius == 8


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_unmeasurable_filter_footprint_abstains(width):
    support, radius = elig.exclude_corrected_support(np.ones((10, 10)), None, "dog", width)
    assert support is None and radius == 0
