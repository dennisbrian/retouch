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


class TestPhase2ClosedFormMatting:
    """Z3 phase 2 — closed-form alpha + foreground estimation."""

    def _soft_alpha_scene(self):
        """Known soft alpha (anti-aliased strands) over a known background."""
        rng = np.random.default_rng(0)
        h, w = 300, 420
        alpha = np.zeros((h, w), np.float32)
        import cv2

        cv2.rectangle(alpha, (120, 40), (260, h), 1.0, -1)
        for _ in range(70):
            x0 = int(rng.integers(115, 265))
            y0 = int(rng.integers(40, 150))
            ang = float(rng.uniform(-1.2, 1.2))
            ln = int(rng.integers(30, 90))
            cv2.line(
                alpha, (x0, y0),
                (int(x0 + ln * np.sin(ang)), int(y0 - ln * np.cos(ang))),
                float(rng.uniform(0.5, 1.0)),
                thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA,
            )
        alpha = np.clip(cv2.GaussianBlur(alpha, (3, 3), 0.6), 0, 1)
        F = np.zeros((h, w, 3), np.float32)
        F[:, :, 0], F[:, :, 1], F[:, :, 2] = 45, 40, 38
        B = np.zeros((h, w, 3), np.float32)
        B[:, :, 1] = np.linspace(70, 190, w)[None, :]
        B[:, :, 0], B[:, :, 2] = 50, 35
        comp = (alpha[:, :, None] * F + (1 - alpha[:, :, None]) * B).astype(np.float32)
        return comp, alpha, F, B

    def test_closed_form_alpha_beats_feathered_mask(self):
        from retouch.matting import build_auto_trimap, solve_closed_form_alpha, _HAVE_SCIPY

        if not _HAVE_SCIPY:
            pytest.skip("scipy not installed; solver falls back by design")

        comp, alpha, _F, _B = self._soft_alpha_scene()
        h, w = alpha.shape
        binary = (alpha > 0.5).astype(np.float32)
        feathered = BackgroundReplacer()._feathered_person_mask(binary, (h, w))
        trimap = build_auto_trimap(binary, band_radius=4)
        solved = solve_closed_form_alpha(comp.astype(np.uint8), trimap)
        assert solved is not None

        band = (alpha > 0.02) & (alpha < 0.98)
        sad_feather = np.abs(feathered[band] - alpha[band]).mean()
        sad_solved = np.abs(solved[band] - alpha[band]).mean()
        assert sad_solved < sad_feather, (
            f"closed-form alpha ({sad_solved:.4f}) must beat the feathered "
            f"mask ({sad_feather:.4f}) on ground truth"
        )

    def test_foreground_estimation_removes_the_colour_fringe(self):
        """The naive composite double-counts the old background; F removes it."""
        from retouch.matting import estimate_foreground

        comp, alpha, F, _B = self._soft_alpha_scene()
        band = (alpha > 0.02) & (alpha < 0.98)
        est = estimate_foreground(comp, alpha)
        # The estimate must be closer to true F than the observed pixel is.
        err_naive = np.abs(comp - F)[band].mean()
        err_est = np.abs(est - F)[band].mean()
        assert err_est < err_naive, (
            f"F estimate ({err_est:.2f}) must beat using src directly ({err_naive:.2f})"
        )

    def test_solver_returns_none_without_scipy(self, monkeypatch):
        """SciPy is an optional dependency; callers must get a clean fallback."""
        import retouch.matting as m

        monkeypatch.setattr(m, "_HAVE_SCIPY", False)
        comp, alpha, _F, _B = self._soft_alpha_scene()
        trimap = m.build_auto_trimap((alpha > 0.5).astype(np.float32), band_radius=4)
        assert m.solve_closed_form_alpha(comp.astype(np.uint8), trimap) is None


class TestPhase3Wiring:
    """Z3 phase 3 — matte compositing wired into the background ops."""

    def test_matte_cache_reuses_and_invalidates(self):
        img, mask, _ = _synthetic_scene()
        r = BackgroundReplacer()
        src = img.astype(np.float32)
        a1 = r._composite_alpha(src, mask)
        a2 = r._composite_alpha(src, mask)
        assert a1 is a2, "identical inputs must hit the matte cache"

        # A different image with the same mask must NOT reuse the matte:
        # the matte is solved from image colour.
        other = np.clip(src * 0.4 + 30.0, 0, 255).astype(np.float32)
        a3 = r._composite_alpha(other, mask)
        assert a3 is not a1, "cache must key on image content, not the mask alone"

    def test_wired_ops_still_protect_the_subject(self):
        """Matte compositing must not eat into the opaque subject interior."""
        img, mask, _ = _synthetic_scene()
        for op, arg in (("blur_background", 100.0), ("grade_background", None)):
            r = BackgroundReplacer()
            if arg is None:
                out = r.grade_background(img, mask, {"desaturation": 80.0})
            else:
                out = getattr(r, op)(img, mask, arg)
            ys, xs = np.where(mask > 0.99)
            y0, y1 = ys.min() + 12, ys.max() - 12
            x0, x1 = xs.min() + 12, xs.max() - 12
            delta = np.abs(
                out[y0:y1, x0:x1].astype(np.float32)
                - img[y0:y1, x0:x1].astype(np.float32)
            )
            assert delta.max() < 3.0, f"{op} modified the subject interior"

    def test_no_scipy_path_still_produces_a_valid_render(self, monkeypatch):
        import retouch.matting as m

        monkeypatch.setattr(m, "_HAVE_SCIPY", False)
        img, mask, _ = _synthetic_scene()
        out = BackgroundReplacer().blur_background(img, mask, 100.0)
        assert out.shape == img.shape and out.dtype == img.dtype
        assert np.isfinite(out.astype(np.float32)).all()
