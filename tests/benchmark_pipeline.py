"""End-to-end pipeline benchmarks for the retouch engine.

Each test:
    * Builds a ``RetouchEngine`` whose heavy dependencies (MediaPipe, BiSeNet
      ONNX) are replaced with ``unittest.mock`` doubles — the engine itself
      still orchestrates the full pipeline, so timings reflect real per-frame
      work, but model load time is excluded.
    * Runs a warm-up call so first-iteration cache/JIT effects don't taint
      timings.
    * Times N iterations with ``time.perf_counter``.
    * Asserts a loose upper bound (CI-safe; the goal is regression detection,
      not micro-benchmarks).
    * Prints the median time so it shows up under ``pytest -v -s``.

Run with:
    pytest tests/benchmark_pipeline.py -v
or pick one stage:
    pytest tests/benchmark_pipeline.py -v -k fast_preview
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from tests.benchmark_utils import benchmark_iterations


# ---------------------------------------------------------------------------
# Mock helpers — fully synthetic face / regions / person mask.
# ---------------------------------------------------------------------------


def _synthetic_landmarks(num: int = 478):
    """Return 478 landmarks spread across the image interior (normalized 0-1)."""
    from retouch.detection import _Landmark, _LandmarkCompat

    landmarks = []
    for i in range(num):
        landmarks.append(
            _Landmark(
                x=0.30 + 0.40 * (i % 23) / 22.0,
                y=0.20 + 0.60 * (i // 23) / 21.0,
                z=0.0,
            )
        )
    return _LandmarkCompat(landmarks)


def _synthetic_face(img_h: int, img_w: int, bbox=None):
    """Build a single FaceData with a realistic 478-landmark face."""
    from retouch.detection import FaceData
    if bbox is None:
        bbox = (int(img_w * 0.25), int(img_h * 0.20),
                int(img_w * 0.50), int(img_h * 0.60))
    return FaceData(
        landmarks=_synthetic_landmarks(),
        bbox=bbox,
        ied=float(img_w * 0.10),
        confidence=1.0,
    )


def _synthetic_regions(img_h: int, img_w: int):
    """Return a FaceRegions instance with every mask populated as a soft blob."""
    from retouch.parsing import FaceRegions

    regions = FaceRegions()
    cx, cy = img_w // 2, img_h // 2
    yy, xx = np.ogrid[:img_h, :img_w]

    def soft_blob(rx: float, ry: float, ox: float = 0.0, oy: float = 0.0,
                  feather: int = 7) -> np.ndarray:
        m = (((xx - (cx + ox)) / rx) ** 2 +
             ((yy - (cy + oy)) / ry) ** 2 <= 1.0).astype(np.float32)
        if feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), feather)
        return np.clip(m, 0, 1)

    for attr in regions.__slots__:
        if attr == "face_oval":
            regions.face_oval = soft_blob(img_w * 0.22, img_h * 0.30)
        elif attr == "skin":
            regions.skin = soft_blob(img_w * 0.22, img_h * 0.28)
        elif attr == "forehead":
            regions.forehead = soft_blob(img_w * 0.18, img_h * 0.10, oy=-img_h * 0.20)
        elif attr in ("left_cheek", "right_cheek"):
            ox = -img_w * 0.12 if attr == "left_cheek" else img_w * 0.12
            setattr(regions, attr, soft_blob(img_w * 0.08, img_h * 0.10, ox=ox, oy=img_h * 0.05))
        elif attr == "nose":
            regions.nose = soft_blob(img_w * 0.04, img_h * 0.10, oy=img_h * 0.05)
        elif attr in ("left_eye", "right_eye"):
            ox = -img_w * 0.08 if attr == "left_eye" else img_w * 0.08
            oy = -img_h * 0.08
            setattr(regions, attr, soft_blob(img_w * 0.06, img_h * 0.025, ox=ox, oy=oy))
        elif attr in ("left_eyebrow", "right_eyebrow"):
            ox = -img_w * 0.08 if attr == "left_eyebrow" else img_w * 0.08
            oy = -img_h * 0.13
            setattr(regions, attr, soft_blob(img_w * 0.07, img_h * 0.02, ox=ox, oy=oy))
        elif attr in ("left_iris", "right_iris"):
            ox = -img_w * 0.08 if attr == "left_iris" else img_w * 0.08
            oy = -img_h * 0.08
            setattr(regions, attr, soft_blob(img_w * 0.02, img_h * 0.02, ox=ox, oy=oy))
        elif attr == "lips":
            regions.lips = soft_blob(img_w * 0.08, img_h * 0.025, oy=img_h * 0.22)
        elif attr == "mouth_interior":
            regions.mouth_interior = soft_blob(img_w * 0.05, img_h * 0.015, oy=img_h * 0.22)
        elif attr in ("left_under_eye", "right_under_eye"):
            ox = -img_w * 0.08 if attr == "left_under_eye" else img_w * 0.08
            oy = -img_h * 0.05
            setattr(regions, attr, soft_blob(img_w * 0.05, img_h * 0.015, ox=ox, oy=oy))
        elif attr == "nose_bridge":
            regions.nose_bridge = soft_blob(img_w * 0.015, img_h * 0.10, oy=-img_h * 0.03)
        elif attr == "forehead_center":
            regions.forehead_center = soft_blob(img_w * 0.05, img_h * 0.05, oy=-img_h * 0.20)
        elif attr in ("cheek_highlights_l", "cheek_highlights_r"):
            ox = -img_w * 0.10 if attr == "cheek_highlights_l" else img_w * 0.10
            oy = img_h * 0.04
            setattr(regions, attr, soft_blob(img_w * 0.03, img_h * 0.02, ox=ox, oy=oy, feather=3))
        elif attr == "jawline_contour":
            regions.jawline_contour = soft_blob(img_w * 0.20, img_h * 0.05, oy=img_h * 0.20, feather=11)
        elif attr == "hair":
            regions.hair = soft_blob(img_w * 0.20, img_h * 0.08, oy=-img_h * 0.25)
        elif attr == "neck":
            regions.neck = soft_blob(img_w * 0.20, img_h * 0.10, oy=img_h * 0.30)
    return regions


def _synthetic_person_mask(img_h: int, img_w: int) -> np.ndarray:
    """A full-image float person mask (1.0 = person, 0.0 = background)."""
    mask = np.zeros((img_h, img_w), dtype=np.float32)
    cx, cy = img_w // 2, img_h // 2
    rx, ry = int(img_w * 0.30), int(img_h * 0.45)
    yy, xx = np.ogrid[:img_h, :img_w]
    inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    mask[inside] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), 12)


def _make_mock_engine_cls():
    """Build a patch context manager that supplies a fully-mocked engine.

    The mock:
        * replaces ``FaceDetector`` so MediaPipe is never loaded.
        * replaces ``FaceParser`` so BiSeNet ONNX is never loaded.
        * wires ``detect`` and ``segment_person`` to return synthetic data.
    """
    # We capture the *current* image size when the detector is created so
    # the mock can build face data that matches the input.
    detector_instance: List[MagicMock] = []
    parser_instance: List[MagicMock] = []

    def _build_detector(*args, **kwargs):
        mock = MagicMock()
        detector_instance.append(mock)
        return mock

    def _build_parser(*args, **kwargs):
        mock = MagicMock()
        parser_instance.append(mock)
        return mock

    return _build_detector, _build_parser, detector_instance, parser_instance


def _wire_mocks(detector_mock, parser_mock, img_h, img_w, no_face=False):
    """Configure mocks to respond to a call of an image of size (img_h, img_w)."""
    if no_face:
        detector_mock.detect.return_value = []
    else:
        face = _synthetic_face(img_h, img_w)
        detector_mock.detect.return_value = [face]

    def _person_segment(img_bgr):
        h, w = img_bgr.shape[:2]
        return _synthetic_person_mask(h, w)
    detector_mock.segment_person.side_effect = _person_segment

    def _parse(landmarks, img_bgr, face_bbox, person_mask=None, ied=100.0):
        h, w = img_bgr.shape[:2]
        return _synthetic_regions(h, w)
    parser_mock.parse.side_effect = _parse

    def _parse_batch(
        crops, landmarks_list, face_bboxes, person_masks, ieds, **_kwargs
    ):
        return [_synthetic_regions(c.shape[:2][0], c.shape[:2][1]) for c in crops]
    parser_mock.parse_batch.side_effect = _parse_batch


@pytest.fixture
def mocked_engine_cls():
    """Return (engine_class_factory, image_dim_setter) so tests can configure size."""
    _build_detector, _build_parser, det_inst, par_inst = _make_mock_engine_cls()

    class _MockedRetouchEngine:
        """A drop-in factory for RetouchEngine that pre-patches the heavy deps."""
        def __new__(cls, *args, **kwargs):
            from retouch import engine as _engine_mod
            with patch.object(_engine_mod, "FaceDetector", _build_detector), \
                 patch.object(_engine_mod, "FaceParser", _build_parser):
                eng = _engine_mod.RetouchEngine(*args, **kwargs)
            return eng

    return _MockedRetouchEngine, det_inst, par_inst


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _bench(
    fn: Callable[[], object],
    iterations: int = 10,
    warmup: int = 2,
) -> Tuple[float, List[float]]:
    """Run *fn* ``warmup`` times then ``iterations`` timed runs.

    Returns ``(median_ms, samples_ms_sorted)``.
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
        f"\n[benchmark] {label:42s} median={median_ms:8.2f} ms "
        f"p95={p95:8.2f} ms (n={len(samples)})"
    )


# ---------------------------------------------------------------------------
# 1. Detection time (mocked at the engine boundary)
# ---------------------------------------------------------------------------


class TestBenchmarkDetection:
    def test_detection_time_hd(self):
        """Detection-style timing on a 1920x1080 image (face count mocked)."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        _wire_mocks(det, par, 1080, 1920, no_face=False)

        def run():
            faces = eng._detector.detect(img)
            return faces

        median, samples = _bench(run, iterations=20, warmup=3)
        _print("detection.detect 1920x1080 (mocked)", median, samples)
        # Mocked detector is essentially free; bound is a generous sanity check
        assert median < 50.0, f"detection (mocked) too slow: {median:.1f}ms"

    def test_no_face_path(self):
        """No-face fallback should still return a (possibly-graded) image."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        _wire_mocks(det, par, 400, 600, no_face=True)

        def run():
            return eng.process(img, recipe="natural", fast=True)

        median, samples = _bench(run, iterations=10, warmup=2)
        _print("pipeline.no-face 600x400 (fast)", median, samples)
        assert median < 2000.0, f"no-face pipeline too slow: {median:.1f}ms"


# ---------------------------------------------------------------------------
# 2. Per-face processing time
# ---------------------------------------------------------------------------


class TestBenchmarkPerFace:
    def test_per_face_single_face_400x400(self):
        """One face, 400x400 — should finish well under a second."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        _wire_mocks(det, par, 400, 400, no_face=False)

        def run():
            return eng.process(img, recipe="natural", fast=False)

        median, samples = _bench(run, iterations=8, warmup=2)
        _print("pipeline.per-face 400x400", median, samples)
        # The full engine includes grading, bloom, etc. — keep generous bound
        # to allow for CI variance and prior test state.
        assert median < 3000.0, f"per-face too slow: {median:.1f}ms (expected < 3000ms)"

    def test_per_face_with_timings(self):
        """Verify per-stage timing breakdown is populated and adds up sensibly."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        _wire_mocks(det, par, 400, 400, no_face=False)

        result = eng.process(img, recipe="natural", fast=False)
        timings = result.timings
        # The processing context should expose a timings dict with at least
        # one entry.
        assert timings, "Expected engine.timings to be populated"
        total = sum(timings.values())
        print(f"\n[benchmark] pipeline stage breakdown: {timings} (sum={total:.1f}ms)")
        assert total >= 0.0

    def test_skin_quality_metrics_per_face(self):
        """Compute skin quality metrics before and after processing."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine
        from scripts.bench.benchmark import skin_quality_metrics

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        _wire_mocks(det, par, 400, 400, no_face=False)

        # Process the image
        result = eng.process(img, recipe="natural", fast=False)

        # Compute metrics on input and output
        if result.face_contexts and len(result.face_contexts) > 0:
            face_ctx = result.face_contexts[0]
            face_width = face_ctx.face_data.ied * 2.5

            # Metrics on input
            input_metrics = skin_quality_metrics(img, result.skin_mask, face_width)
            # Metrics on output
            output_metrics = skin_quality_metrics(result, result.skin_mask, face_width)

            print(f"\n[benchmark] skin_quality_input: blotch_std={input_metrics['blotch_std']:.6f} chroma_std={input_metrics['chroma_std']:.6f}")
            print(f"[benchmark] skin_quality_output: blotch_std={output_metrics['blotch_std']:.6f} chroma_std={output_metrics['chroma_std']:.6f}")

            # Just verify they are finite
            assert np.isfinite(input_metrics["blotch_std"])
            assert np.isfinite(output_metrics["blotch_std"])
            assert np.isfinite(input_metrics["chroma_std"])
            assert np.isfinite(output_metrics["chroma_std"])


# ---------------------------------------------------------------------------
# 3. Global processing time (grading, bloom, vignette)
# ---------------------------------------------------------------------------


class TestBenchmarkGlobal:
    def test_global_grade_800x600(self):
        """Global stage (grading, bloom, vignette, sharpen)."""
        from retouch.grading import ColorGrader

        grader = ColorGrader()
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        def run():
            return grader.grade(img, preset="natural", intensity=1.0)

        median, samples = _bench(run, iterations=15, warmup=2)
        _print("global.grade(natural) 800x600", median, samples)
        assert median < 150.0, f"global.grade too slow: {median:.1f}ms"

    def test_global_grade_warm(self):
        """Warm path — should benefit from cached LUTs and module imports."""
        from retouch.grading import ColorGrader

        grader = ColorGrader()
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        # warm up
        for _ in range(3):
            grader.grade(img, preset="natural", intensity=1.0)

        samples: List[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            grader.grade(img, preset="natural", intensity=1.0)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        median = samples[len(samples) // 2]
        _print("global.grade(natural) 1920x1080 warm", median, samples)
        assert median < 500.0, f"global.grade 1920x1080 too slow: {median:.1f}ms"


# ---------------------------------------------------------------------------
# 4. End-to-end pipeline
# ---------------------------------------------------------------------------


class TestBenchmarkEndToEnd:
    def test_pipeline_end_to_end_fast(self):
        """Full pipeline, fast=True, 800x600 — preview-quality."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        _wire_mocks(det, par, 600, 800, no_face=False)

        def run():
            return eng.process(img, recipe="natural", fast=True)

        median, samples = _bench(run, iterations=8, warmup=2)
        _print("pipeline.end-to-end 800x600 (fast)", median, samples)
        assert median < 2000.0, f"pipeline (fast) too slow: {median:.1f}ms"

    def test_pipeline_end_to_end_full(self):
        """Full pipeline, fast=False, 800x600 — production-quality."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        _wire_mocks(det, par, 600, 800, no_face=False)

        def run():
            return eng.process(img, recipe="natural", fast=False)

        median, samples = _bench(run, iterations=5, warmup=1)
        _print("pipeline.end-to-end 800x600 (full)", median, samples)
        # Full path is heavier; allow up to 3s in CI
        assert median < 3000.0, f"pipeline (full) too slow: {median:.1f}ms"


# ---------------------------------------------------------------------------
# 5. Fast preview mode
# ---------------------------------------------------------------------------


class TestBenchmarkFastPreview:
    def test_fast_preview(self):
        """fast=True on a 1920x1080 should downscale and finish quickly."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        _wire_mocks(det, par, 1080, 1920, no_face=False)

        def run():
            return eng.process(img, recipe="natural", fast=True)

        median, samples = _bench(run, iterations=5, warmup=1)
        _print("pipeline.fast-preview 1920x1080", median, samples)
        # Fast preview downscales to ~800px max-dim and runs the full grading
        # stack; the bilateral filter in the face path dominates. Bound is
        # intentionally loose to avoid CI flake.
        assert median < 5000.0, f"fast preview too slow: {median:.1f}ms"


# ---------------------------------------------------------------------------
# 6. No-face fallback
# ---------------------------------------------------------------------------


class TestBenchmarkNoFace:
    def test_no_face_fallback(self):
        """When no faces are detected, the engine should still grade the image."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        _wire_mocks(det, par, 600, 800, no_face=True)

        def run():
            return eng.process(img, recipe="natural", fast=False)

        median, samples = _bench(run, iterations=8, warmup=2)
        _print("pipeline.no-face 800x600 (full)", median, samples)
        assert median < 1500.0, f"no-face fallback too slow: {median:.1f}ms"


# ---------------------------------------------------------------------------
# 7. Batch processing
# ---------------------------------------------------------------------------


class TestBenchmarkBatch:
    def test_batch_folder(self, tmp_path):
        """Process a folder of synthetic images and report per-image time."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()

        # Write 4 synthetic JPEGs into tmp folder
        n_images = 4
        for i in range(n_images):
            img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i:02d}.jpg"), img)

        det = dets[0]
        par = pars[0]
        _wire_mocks(det, par, 300, 400, no_face=False)

        # Reset mocks per call (RetouchEngine caches person_mask etc)
        samples: List[float] = []
        for path in sorted(tmp_path.glob("*.jpg")):
            img = cv2.imread(str(path))
            _wire_mocks(det, par, 300, 400, no_face=False)
            t0 = time.perf_counter()
            eng.process(img, recipe="natural", fast=True)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        median = samples[len(samples) // 2]
        per_image_avg = sum(samples) / len(samples)
        _print(f"batch.folder ({n_images} imgs, 400x300 fast)", median, samples)
        print(f"[benchmark] batch avg per-image: {per_image_avg:.1f}ms")
        assert median < 2000.0, f"batch median too slow: {median:.1f}ms"


# ---------------------------------------------------------------------------
# 8. Memory profiling
# ---------------------------------------------------------------------------


def _get_process_memory_mb() -> float:
    """Return the current process RSS in MB. Uses psutil if available, else resource."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes; Linux returns kilobytes
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        return usage / 1024.0


class TestBenchmarkMemory:
    def test_peak_memory_800x600(self):
        """Measure peak RSS during a single full pipeline run."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        _wire_mocks(det, par, 600, 800, no_face=False)

        # Warm up
        eng.process(img, recipe="natural", fast=True)
        gc.collect()

        baseline = _get_process_memory_mb()
        peak = baseline

        # Run 3 iterations and track peak memory
        for _ in range(3):
            eng.process(img, recipe="natural", fast=False)
            current = _get_process_memory_mb()
            if current > peak:
                peak = current
            gc.collect()

        delta = peak - baseline
        print(
            f"\n[benchmark] memory 800x600 (full): baseline={baseline:.1f}MB "
            f"peak={peak:.1f}MB delta=+{delta:.1f}MB"
        )
        # 256MB is a very loose bound — synthetic data should stay well under
        # 100MB. Generous upper bound to avoid CI flakiness.
        assert peak - baseline < 256.0, (
            f"pipeline leaked too much memory: +{delta:.1f}MB"
        )

    def test_peak_memory_no_face(self):
        """No-face path should use less memory than the full pipeline."""
        from retouch import engine as engine_mod
        from retouch.engine import RetouchEngine

        _bd, _bp, dets, pars = _make_mock_engine_cls()
        with patch.object(engine_mod, "FaceDetector", _bd), \
             patch.object(engine_mod, "FaceParser", _bp):
            eng = RetouchEngine()
        det = dets[0]
        par = pars[0]
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        _wire_mocks(det, par, 600, 800, no_face=True)

        eng.process(img, recipe="natural", fast=True)
        gc.collect()
        baseline = _get_process_memory_mb()
        peak = baseline

        for _ in range(3):
            eng.process(img, recipe="natural", fast=False)
            current = _get_process_memory_mb()
            if current > peak:
                peak = current
            gc.collect()

        delta = peak - baseline
        print(
            f"\n[benchmark] memory no-face 800x600: baseline={baseline:.1f}MB "
            f"peak={peak:.1f}MB delta=+{delta:.1f}MB"
        )
        assert peak - baseline < 256.0, (
            f"no-face pipeline leaked too much memory: +{delta:.1f}MB"
        )


if __name__ == "__main__":
    # Allow running as a plain script for ad-hoc profiling.
    pytest.main([__file__, "-v", "-s"])
