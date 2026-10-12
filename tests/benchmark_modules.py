"""Micro-benchmarks for individual retouch modules.

Each test:
    * Uses synthetic numpy images and masks (no model loading, no I/O).
    * Runs a warm-up call so caches/JIT/cached lookups don't taint timings.
    * Times N iterations with ``time.perf_counter`` and reports average.
    * Asserts a reasonable upper bound (loose, so CI doesn't flake).
    * Prints the median time so it's visible in pytest -v output for tracking.

Run with:
    pytest tests/benchmark_modules.py -v
or pick a single module:
    pytest tests/benchmark_modules.py -v -k frequency
"""

from __future__ import annotations

import time
from typing import Callable, List, Tuple

import cv2
import numpy as np
import pytest

from tests.benchmark_utils import benchmark_iterations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bench(
    fn: Callable[[], object],
    iterations: int = 20,
    warmup: int = 2,
) -> Tuple[float, List[float]]:
    """Run *fn* ``warmup`` times, then ``iterations`` timed runs.

    Returns ``(median_ms, per_run_ms_list)``.
    """
    iterations = benchmark_iterations(iterations)
    for _ in range(max(0, warmup)):
        fn()
    samples: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    return median, samples


def _print(label: str, median_ms: float, samples: List[float]) -> None:
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print(
        f"\n[benchmark] {label:40s} median={median_ms:8.2f} ms "
        f"p95={p95:8.2f} ms (n={len(samples)})"
    )


def _make_skin_mask(h: int, w: int) -> np.ndarray:
    """Return a soft circular skin mask (float32, 0-1) covering ~30% of the image."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy, r = w // 2, h // 2, int(min(h, w) * 0.30)
    y, x = np.ogrid[:h, :w]
    inside = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    mask[inside] = 1.0
    # Soft edge for feathering
    mask = cv2.GaussianBlur(mask, (0, 0), 8)
    return np.clip(mask, 0, 1)


def _make_eye_masks(h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (left_eye_mask, right_eye_mask, iris_mask) — small soft ellipses."""
    def soft_ellipse(mask, cx, cy, rx, ry):
        y, x = np.ogrid[:h, :w]
        inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
        mask[inside] = 1.0
        return cv2.GaussianBlur(mask, (0, 0), 3)

    left = np.zeros((h, w), dtype=np.float32)
    right = np.zeros((h, w), dtype=np.float32)
    iris = np.zeros((h, w), dtype=np.float32)
    eye_y = int(h * 0.42)
    eye_offset = int(w * 0.10)
    left = soft_ellipse(left, w // 2 - eye_offset, eye_y, 22, 12)
    right = soft_ellipse(right, w // 2 + eye_offset, eye_y, 22, 12)
    iris = soft_ellipse(iris, w // 2 - eye_offset, eye_y, 9, 9)
    iris = iris + soft_ellipse(np.zeros((h, w), dtype=np.float32),
                               w // 2 + eye_offset, eye_y, 9, 9)
    return np.clip(left, 0, 1), np.clip(right, 0, 1), np.clip(iris, 0, 1)


def _make_lip_mask(h: int, w: int) -> np.ndarray:
    """Return a soft elliptical lip mask centred in the lower face."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, int(h * 0.72)
    rx, ry = int(w * 0.10), int(h * 0.04)
    y, x = np.ogrid[:h, :w]
    inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    mask[inside] = 1.0
    return np.clip(cv2.GaussianBlur(mask, (0, 0), 5), 0, 1)


def _make_regions(h: int, w: int):
    """Build a minimal FaceRegions-like object for the eye enhancer."""
    from retouch.parsing import FaceRegions
    left_eye, right_eye, iris = _make_eye_masks(h, w)
    regions = FaceRegions()
    regions.left_eye = left_eye
    regions.right_eye = right_eye
    regions.left_iris = iris * (left_eye > 0.5)
    regions.right_iris = iris * (right_eye > 0.5)
    regions.skin = _make_skin_mask(h, w)
    return regions


# ---------------------------------------------------------------------------
# 1. Frequency separation
# ---------------------------------------------------------------------------


class TestBenchmarkFrequencySeparation:
    """Measure FrequencySeparator.separate() and combine() latency."""

    def test_separate_400x400(self):
        from retouch.frequency import FrequencySeparator
        sep = FrequencySeparator()
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

        def run():
            sep.separate(img, face_width=120)

        median, samples = _bench(run, iterations=30, warmup=3)
        _print("frequency.separate 400x400", median, samples)
        assert median < 60.0, f"separate() too slow: {median:.1f}ms (expected < 60ms)"

    def test_separate_800x600(self):
        from retouch.frequency import FrequencySeparator
        sep = FrequencySeparator()
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        def run():
            sep.separate(img, face_width=200)

        median, samples = _bench(run, iterations=20, warmup=2)
        _print("frequency.separate 800x600", median, samples)
        assert median < 150.0, f"separate() too slow: {median:.1f}ms (expected < 150ms)"

    def test_combine_200x200(self):
        from retouch.frequency import FrequencySeparator
        sep = FrequencySeparator()
        # combine() runs cv2.bilateralFilter which is O(W*H) with a high
        # constant factor — keep the test image small so CI stays snappy.
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        skin_mask = _make_skin_mask(200, 200)
        layers = sep.separate(img, face_width=80)

        def run():
            sep.combine(
                layers,
                skin_mask=skin_mask,
                smooth_strength=0.5,
                mid_reduction=0.4,
                texture_opacity=1.0,
                face_width=80,
            )

        median, samples = _bench(run, iterations=10, warmup=2)
        _print("frequency.combine 200x200", median, samples)
        # combine() does a bilateral filter — bound is sized to the image
        assert median < 1500.0, f"combine() too slow: {median:.1f}ms (expected < 1500ms)"

    def test_reconstruct_round_trip(self):
        """A separate→combine round-trip stays within budget."""
        from retouch.frequency import FrequencySeparator
        sep = FrequencySeparator()
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        skin_mask = _make_skin_mask(200, 200)

        def run():
            layers = sep.separate(img, face_width=80)
            return sep.combine(
                layers,
                skin_mask=skin_mask,
                smooth_strength=0.4,
                face_width=80,
            )

        median, samples = _bench(run, iterations=10, warmup=2)
        _print("frequency.separate+combine 200x200", median, samples)
        assert median < 1500.0, f"round-trip too slow: {median:.1f}ms (expected < 1500ms)"


# ---------------------------------------------------------------------------
# 2. Skin processing
# ---------------------------------------------------------------------------


class TestBenchmarkSkinProcessing:
    """Micro-benchmarks for SkinProcessor (whiten, equalize, dodge_burn)."""

    def _skin(self):
        from retouch.skin import SkinProcessor
        return SkinProcessor()

    def test_skin_smooth_via_freq_combine(self):
        """End-to-end skin smoothing via the frequency combine path."""
        from retouch.frequency import FrequencySeparator
        sep = FrequencySeparator()
        proc = self._skin()
        # 200x200 keeps the bilateral filter fast enough for CI.
        img = np.random.randint(80, 200, (200, 200, 3), dtype=np.uint8)
        skin_mask = _make_skin_mask(200, 200)
        layers = sep.separate(img, face_width=80)

        def run():
            return sep.combine(
                layers,
                skin_mask=skin_mask,
                smooth_strength=0.6,
                mid_reduction=0.5,
                face_width=80,
            )

        median, samples = _bench(run, iterations=10, warmup=1)
        _print("skin.smooth 200x200", median, samples)
        assert median < 1500.0, f"skin smooth too slow: {median:.1f}ms (expected < 1500ms)"

    def test_skin_whiten(self):
        proc = self._skin()
        img = np.random.randint(60, 200, (400, 400, 3), dtype=np.uint8)
        skin_mask = _make_skin_mask(400, 400)

        def run():
            return proc.whiten(img, skin_mask, strength=30, tone="rosy")

        median, samples = _bench(run, iterations=30, warmup=3)
        _print("skin.whiten 400x400", median, samples)
        assert median < 30.0, f"whiten too slow: {median:.1f}ms (expected < 30ms)"

    def test_skin_equalize(self):
        proc = self._skin()
        img = np.random.randint(60, 200, (400, 400, 3), dtype=np.uint8)
        skin_mask = _make_skin_mask(400, 400)

        def run():
            return proc.equalize(img, skin_mask, strength=40)

        median, samples = _bench(run, iterations=20, warmup=2)
        _print("skin.equalize 400x400", median, samples)
        assert median < 40.0, f"equalize too slow: {median:.1f}ms (expected < 40ms)"


# ---------------------------------------------------------------------------
# 3. Eye enhancement
# ---------------------------------------------------------------------------


class TestBenchmarkEyeEnhancement:
    def test_eye_enhance_full(self):
        from retouch.eyes import EyeEnhancer
        enh = EyeEnhancer()
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        regions = _make_regions(400, 600)

        def run():
            return enh.enhance(img, regions, strength=40, catchlight_strength=40)

        median, samples = _bench(run, iterations=20, warmup=2)
        _print("eyes.enhance 400x600", median, samples)
        assert median < 60.0, f"eye enhance too slow: {median:.1f}ms (expected < 60ms)"

    def test_eye_enhance_zero_strength_fast_path(self):
        """Verify strength=0 short-circuits quickly."""
        from retouch.eyes import EyeEnhancer
        enh = EyeEnhancer()
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        regions = _make_regions(400, 600)

        def run():
            return enh.enhance(img, regions, strength=0)

        median, samples = _bench(run, iterations=100, warmup=10)
        _print("eyes.enhance(strength=0) 400x600", median, samples)
        assert median < 5.0, f"eye fast-path too slow: {median:.1f}ms (expected < 5ms)"


# ---------------------------------------------------------------------------
# 4. Lip enhancement
# ---------------------------------------------------------------------------


class TestBenchmarkLipEnhancement:
    def test_lip_enhance_full(self):
        from retouch.lips import LipEnhancer
        enh = LipEnhancer()
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        lip_mask = _make_lip_mask(400, 600)

        def run():
            return enh.enhance(img, lip_mask, strength=30, tint="pink", finish="gloss")

        median, samples = _bench(run, iterations=15, warmup=2)
        _print("lips.enhance 400x600", median, samples)
        assert median < 80.0, f"lip enhance too slow: {median:.1f}ms (expected < 80ms)"

    def test_lip_enhance_no_op(self):
        from retouch.lips import LipEnhancer
        enh = LipEnhancer()
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        lip_mask = _make_lip_mask(400, 600)

        def run():
            return enh.enhance(img, lip_mask, strength=0)

        median, samples = _bench(run, iterations=100, warmup=10)
        _print("lips.enhance(strength=0) 400x600", median, samples)
        assert median < 3.0, f"lip fast-path too slow: {median:.1f}ms (expected < 3ms)"


# ---------------------------------------------------------------------------
# 5. Color grading
# ---------------------------------------------------------------------------


class TestBenchmarkColorGrading:
    def test_grade_natural_preset(self):
        from retouch.grading import ColorGrader
        grader = ColorGrader()
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        def run():
            return grader.grade(img, preset="natural", intensity=1.0)

        median, samples = _bench(run, iterations=15, warmup=2)
        _print("grading.grade(natural) 800x600", median, samples)
        assert median < 100.0, f"grade(natural) too slow: {median:.1f}ms (expected < 100ms)"

    def test_grade_skip_post_effects(self):
        """Disabling an enabled post-effect should take the fast path."""
        from retouch.grading import ColorGrader
        grader = ColorGrader()
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        # natural has no enabled post-effects, so benchmarking it here would
        # measure the same work in both modes. Use a post-effect-only preset to
        # isolate the skip flag from the core color-operation budget.
        settings = {
            "halation": {"threshold": 210, "radius": 21, "intensity": 0.4},
        }

        def run():
            return grader.grade(
                img, preset=settings, intensity=1.0, skip_post_effects=True
            )

        median, samples = _bench(run, iterations=20, warmup=2)
        _print("grading.grade(halation,no-post) 800x600", median, samples)
        assert median < 60.0, (
            f"grade(halation,no-post) too slow: {median:.1f}ms (expected < 60ms)"
        )

    def test_grade_unknown_falls_back(self):
        """Unknown preset name should silently fall back to natural."""
        from retouch.grading import ColorGrader
        grader = ColorGrader()
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        def run():
            return grader.grade(img, preset="this_does_not_exist")

        median, samples = _bench(run, iterations=20, warmup=2)
        _print("grading.grade(unknown) 300x300", median, samples)
        assert median < 50.0, f"grade(fallback) too slow: {median:.1f}ms (expected < 50ms)"


# ---------------------------------------------------------------------------
# 6. Face reshaping
# ---------------------------------------------------------------------------


class TestBenchmarkFaceReshaping:
    def test_reshape_zero_strength(self):
        """strength=0 is the fast path and should be near-instant."""
        from retouch.geometry import FaceReshaper
        from retouch.detection import _Landmark, _LandmarkCompat, FaceData
        reshaper = FaceReshaper()
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        landmarks = [
            _Landmark(x=0.5 + 0.001 * (i % 17), y=0.5 + 0.001 * (i // 17))
            for i in range(478)
        ]
        compat = _LandmarkCompat(landmarks)
        face = FaceData(
            landmarks=compat, bbox=(200, 200, 400, 400), ied=80.0, confidence=1.0
        )

        def run():
            return reshaper.reshape(img, [face], strength=0)

        median, samples = _bench(run, iterations=200, warmup=20)
        _print("geometry.reshape(strength=0) 800x600", median, samples)
        assert median < 1.0, (
            f"reshape(0) too slow: {median:.2f}ms (expected < 1ms)"
        )

    def test_reshape_with_strength(self):
        from retouch.geometry import FaceReshaper
        from retouch.detection import _Landmark, _LandmarkCompat, FaceData
        reshaper = FaceReshaper()
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        # Spread landmarks across the expected face area so the jaw/cheek/chin
        # indices actually land inside the image bounds.
        landmarks = []
        for i in range(478):
            landmarks.append(
                _Landmark(x=0.30 + 0.40 * (i % 23) / 22.0,
                          y=0.25 + 0.50 * (i // 23) / 21.0)
            )
        compat = _LandmarkCompat(landmarks)
        face = FaceData(
            landmarks=compat, bbox=(200, 150, 400, 500), ied=80.0, confidence=1.0
        )

        def run():
            return reshaper.reshape(img, [face], strength=30)

        median, samples = _bench(run, iterations=20, warmup=2)
        _print("geometry.reshape(strength=30) 800x600", median, samples)
        # remap is O(H*W) — should still be well under 100ms for 800x600
        assert median < 100.0, f"reshape(30) too slow: {median:.1f}ms (expected < 100ms)"


if __name__ == "__main__":
    # Allow running as a plain script for ad-hoc profiling.
    pytest.main([__file__, "-v", "-s"])
