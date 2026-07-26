"""Z3 phase 0 — synthetic ground-truth harness for background layer bleed.

These tests pin the defect described in ``docs/plans/PLAN_Z3_ALPHA_MATTING.md`` §1.1:
``blur_background`` / ``lens_blur`` historically blurred the *whole image* (subject
included) and composited that subject-contaminated result into the background, so
background pixels near the subject edge picked up subject colour even when the mask
was mathematically perfect.

The probes here deliberately use a **hard rectangular mask** so mask error is zero by
construction — any measured contamination is layer bleed, not matte error.
"""

from __future__ import annotations

import numpy as np
import pytest

from retouch.background import BackgroundReplacer


def _synthetic_scene(h: int = 400, w: int = 420) -> tuple:
    """Return (img, person_mask, x_edge) with a high-contrast subject on a dark bg.

    The subject is a bright vertical band; the background is dark. Any bleed of
    subject colour into the background is therefore a large, unambiguous signal.
    """
    img = np.full((h, w, 3), 40.0, dtype=np.float32)  # dark background
    # Structured background so a "blur did nothing" bug cannot pass silently.
    img[:, ::7] = 70.0
    x0, x1 = int(w * 0.35), int(w * 0.55)
    img[int(h * 0.10):int(h * 0.90), x0:x1] = 230.0  # bright subject

    mask = np.zeros((h, w), dtype=np.float32)
    mask[int(h * 0.10):int(h * 0.90), x0:x1] = 1.0
    return img.astype(np.uint8), mask, x1


def _bleed_profile(before: np.ndarray, after: np.ndarray, row: int, x_edge: int) -> dict:
    """Mean per-bin shift of background pixels toward subject brightness.

    Positive values mean the background got brighter, i.e. subject luminance leaked
    outward. Bins are distances (px) to the right of the subject edge.
    """
    b = before.astype(np.float32).mean(axis=2)
    a = after.astype(np.float32).mean(axis=2)
    out = {}
    for lo, hi in ((3, 15), (15, 35), (35, 70), (70, 120)):
        x_lo, x_hi = x_edge + lo, x_edge + hi
        if x_hi >= before.shape[1]:
            continue
        out[(lo, hi)] = float(a[row, x_lo:x_hi].mean() - b[row, x_lo:x_hi].mean())
    return out


class TestBlurBackgroundLayerBleed:
    """blur_background must not composite subject pixels into the background."""

    def test_perfect_mask_produces_no_subject_bleed(self):
        img, mask, x_edge = _synthetic_scene()
        out = BackgroundReplacer().blur_background(img, mask, 100.0)

        profile = _bleed_profile(img, out, row=img.shape[0] // 2, x_edge=x_edge)
        assert profile, "probe bins fell outside the image — fix the test geometry"

        # With a perfect mask the background must not brighten toward the subject.
        # Pre-fix this peaked near +90 levels; allow a small tolerance for the
        # structured-background blur redistributing its own energy.
        worst = max(profile.values())
        assert worst < 12.0, (
            f"subject colour bled into the background with a perfect mask: {profile}"
        )

    def test_bleed_does_not_grow_toward_the_subject_edge(self):
        """The bleed signature is monotone-increasing toward the edge; assert it is gone.

        A whole-image blur contaminates most strongly just outside the subject, so
        near-edge shift >> far-field shift. After the fix the profile must be flat.
        """
        img, mask, x_edge = _synthetic_scene()
        out = BackgroundReplacer().blur_background(img, mask, 100.0)

        profile = _bleed_profile(img, out, row=img.shape[0] // 2, x_edge=x_edge)
        near = profile[(15, 35)]
        far = profile[(70, 120)]
        assert near - far < 10.0, (
            f"near-edge bleed exceeds far-field by {near - far:.1f} levels: {profile}"
        )

    def test_subject_pixels_are_untouched(self):
        img, mask, _ = _synthetic_scene()
        out = BackgroundReplacer().blur_background(img, mask, 100.0)
        core = mask > 0.99
        # Erode the comparison region so the feathered seam is excluded.
        h, w = mask.shape
        ys, xs = np.where(core)
        y0, y1 = ys.min() + 12, ys.max() - 12
        x0, x1 = xs.min() + 12, xs.max() - 12
        delta = np.abs(
            out[y0:y1, x0:x1].astype(np.float32) - img[y0:y1, x0:x1].astype(np.float32)
        )
        assert delta.max() < 2.0, "subject interior must stay sharp and unmodified"

    def test_background_is_actually_blurred(self):
        """Guard against 'fixing' the bleed by simply not blurring."""
        img, mask, x_edge = _synthetic_scene()
        out = BackgroundReplacer().blur_background(img, mask, 100.0)
        # The structured 7px stripes in the far background must be smoothed away.
        row = img.shape[0] // 2
        far_before = img[row, x_edge + 70 : x_edge + 120].astype(np.float32).mean(axis=1)
        far_after = out[row, x_edge + 70 : x_edge + 120].astype(np.float32).mean(axis=1)
        assert far_after.std() < far_before.std() * 0.5, "background was not blurred"


class TestLensBlurLayerBleed:
    """lens_blur shares the same whole-image blur source and the same defect."""

    def test_perfect_mask_produces_no_subject_bleed(self):
        img, mask, x_edge = _synthetic_scene()
        out = BackgroundReplacer().lens_blur(img, mask, 100.0)

        profile = _bleed_profile(img, out, row=img.shape[0] // 2, x_edge=x_edge)
        worst = max(profile.values())
        assert worst < 12.0, (
            f"subject colour bled into the background with a perfect mask: {profile}"
        )


class TestCleanInputIdentity:
    """Strength 0 must be byte-identical, and dtypes must round-trip."""

    @pytest.mark.parametrize("op", ["blur_background", "lens_blur"])
    def test_zero_strength_is_identity(self, op):
        img, mask, _ = _synthetic_scene()
        out = getattr(BackgroundReplacer(), op)(img, mask, 0.0)
        assert np.array_equal(out, img)

    @pytest.mark.parametrize("op", ["blur_background", "lens_blur"])
    def test_dtype_round_trip(self, op):
        img, mask, _ = _synthetic_scene()
        u8 = getattr(BackgroundReplacer(), op)(img, mask, 60.0)
        assert u8.dtype == np.uint8

        f32 = getattr(BackgroundReplacer(), op)(img.astype(np.float32), mask, 60.0)
        assert f32.dtype == np.float32
        # The two paths must agree to within rounding.
        assert np.abs(f32 - u8.astype(np.float32)).mean() < 1.5
