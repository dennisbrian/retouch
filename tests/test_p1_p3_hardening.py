"""Focused production-hardening checks for P1-P3.

These tests deliberately avoid model inference.  They cover the contracts that
must remain true when the full runtime is unavailable in CI: bounded blur
behavior and the registry's real wrapper dispatch.
"""

from types import SimpleNamespace

import numpy as np
import pytest


def test_large_sigma_blur_handles_oversized_non_square_frame():
    from retouch.grading import _large_sigma_blur

    # A portrait-like frame whose short edge is above the compute threshold.
    # This exercises the reduced-resolution path without allocating a 4K image.
    image = np.random.default_rng(12).random((1500, 2200), dtype=np.float32)
    result = _large_sigma_blur(image, 101, max_compute_dim=700)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()


def test_large_sigma_blur_wide_panorama_does_not_scale_from_long_edge():
    from retouch.grading import _large_sigma_blur

    image = np.random.default_rng(14).random((320, 2400, 3), dtype=np.float32)
    result = _large_sigma_blur(image, 81, max_compute_dim=160)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()


def test_large_sigma_blur_preserves_exact_small_frame_contract():
    import cv2
    from retouch.grading import _large_sigma_blur

    image = np.random.default_rng(13).random((100, 140, 3), dtype=np.float32)
    expected = cv2.GaussianBlur(image, (9, 9), 0)
    result = _large_sigma_blur(image, 9, max_compute_dim=140)

    assert np.array_equal(result, expected)


def test_peak_rss_probe_rejects_unsafe_dimensions_before_allocation():
    from scripts.bench.benchmark import assert_grading_peak_rss

    with pytest.raises(ValueError, match="allocation safety ceiling"):
        assert_grading_peak_rss([(10_000, 10_000)], max_pixels=1_000)


def test_global_registry_routes_all_stages_in_order_and_applies_gates():
    from retouch.stage_wrappers import build_global_registry
    from retouch.stages import PipelineState

    calls = []

    class FakeEngine:
        def _record(self, name, image, *args, **kwargs):
            calls.append((name, args, kwargs))
            # Make behavior observable: every dispatched stage increments one
            # channel, proving the fold uses each returned image.
            return image + 1.0

        def __getattr__(self, name):
            if name.startswith("_stage_"):
                return lambda image, *args, _name=name, **kwargs: self._record(
                    _name, image, *args, **kwargs
                )
            raise AttributeError(name)

    ctx = SimpleNamespace(
        subject_separation=0.0, background_harmonize=0.0,
        _local_adjustments=[{"kind": "brush"}],
    )
    state = PipelineState(
        img=np.zeros((4, 4, 3), dtype=np.float32), ctx=ctx,
        person_mask=np.ones((4, 4), dtype=np.float32),
        acc_skin=np.ones((4, 4), dtype=np.float32),
        acc_skin_hair=np.ones((4, 4), dtype=np.float32),
        acc_lips=np.ones((4, 4), dtype=np.float32),
        acc_sharpen=np.ones((4, 4), dtype=np.float32),
        faces=[], h_img=4, w_img=4,
    )

    result = build_global_registry(FakeEngine()).run(state)
    names = [name.rsplit("_stage_", 1)[-1] for name, _, _ in calls]
    assert names == [
        "background", "body_skin", "cosplay_moat", "global", "grade",
        "local_adjustments", "finish", "body_reshape",
    ]
    assert result.img[0, 0, 0] == 8.0
    assert result.timings["subject_separation"] == 0.0
    assert result.timings["background"] >= 0.0

    # Parameter routing is part of the production contract, not just ordering.
    local = next(entry for entry in calls if entry[0] == "_stage_local_adjustments")
    assert local[2]["local_adjustments"] == ctx._local_adjustments
    assert local[2]["semantic_masks"]["skin"] is state.acc_skin
