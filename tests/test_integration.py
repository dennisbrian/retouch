"""Integration tests — full RetouchEngine pipeline and end-to-end processing.

These tests validate that:
- The pipeline runs without crashing on both real and synthetic images.
- The output image has the correct shape, dtype, and channel order.
- The ProcessingResult carries all expected metadata (timings, face_count, etc.).
- Various recipes can be applied without error.
"""

import numpy as np
import cv2
import pytest

from retouch.engine import RetouchEngine, retouch, ProcessingResult
from retouch.recipes import RECIPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_output(result, input_img, expected_faces=None):
    """Assert basic invariants of a ProcessingResult."""
    assert result.shape == input_img.shape, f"Shape mismatch: {result.shape} vs {input_img.shape}"
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert isinstance(result.face_count, int)
    assert isinstance(result.timings, dict)
    if expected_faces is not None:
        assert result.face_count == expected_faces
    # Timings should be populated when a face is detected
    if result.face_count > 0:
        assert "detection" in result.timings
        assert "per_face" in result.timings


# ---------------------------------------------------------------------------
# Pipeline smoke tests  (synthetic face — may or may not be detected)
# ---------------------------------------------------------------------------

class TestPipelineOnSyntheticFace:
    """Run the full pipeline on a synthetic cartoon face.

    MediaPipe may not detect the drawn face, so these tests validate that
    the pipeline handles both the face-found and no-face paths gracefully.
    """

    def test_pipeline_runs_all_recipes(self, engine, synthetic_face):
        for recipe_name in RECIPES:
            result = engine.process(synthetic_face, recipe=recipe_name)
            _assert_valid_output(result, synthetic_face)
            assert result.params.active_recipe == recipe_name

    def test_pipeline_with_overrides(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="cosplay",
            smooth=80.0,
            whiten=50.0,
            contrast=15.0,
            eye_enhance=60.0,
            lip_enhance=40.0,
            impact=30.0,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_fast_preview(self, engine, synthetic_face):
        result = engine.process(synthetic_face, recipe="natural", fast=True)
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_auto_exposure(self, engine, synthetic_face):
        result = engine.process(synthetic_face, recipe="natural", auto_exposure=True)
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_tonal_adjustments(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            brightness=15.0,
            highlights=10.0,
            shadows=-5.0,
            whites=5.0,
            blacks=-3.0,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_color_grade_only(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            color_grade="film",
            grade_intensity=0.8,
            grain=0.05,
            chromatic_aberration=2.0,
            halation=0.2,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_nose_smooth(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            nose_smooth=30.0,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_specular_bloom(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            specular_bloom=50.0,
            specular_bloom_tone="neutral",
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_whiten_tone(self, engine, synthetic_face):
        for tone in ("rosy", "porcelain", "neutral"):
            result = engine.process(
                synthetic_face,
                recipe="natural",
                whiten=40.0,
                whiten_tone=tone,
            )
            _assert_valid_output(result, synthetic_face)

    def test_pipeline_dodge_burn(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            dodge_burn=50.0,
        )
        _assert_valid_output(result, synthetic_face)


# ---------------------------------------------------------------------------
# No-face fallback
# ---------------------------------------------------------------------------

class TestNoFaceFallback:
    """When no face is detected, the pipeline should still produce output."""

    def test_blank_image_no_crash(self, engine, blank_image):
        result = engine.process(blank_image, recipe="natural")
        _assert_valid_output(result, blank_image, expected_faces=0)

    def test_blank_image_with_color_grade(self, engine, blank_image):
        result = engine.process(
            blank_image,
            recipe="natural",
            color_grade="film",
            grade_intensity=0.7,
        )
        _assert_valid_output(result, blank_image, expected_faces=0)

    def test_blank_image_with_impact(self, engine, blank_image):
        result = engine.process(
            blank_image, recipe="natural", impact=50.0,
        )
        _assert_valid_output(result, blank_image, expected_faces=0)

    def test_blank_image_all_recipes(self, engine, blank_image):
        for recipe_name in RECIPES:
            result = engine.process(blank_image, recipe=recipe_name)
            _assert_valid_output(result, blank_image)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

class TestRetouchFunction:
    """Test the backward-compatible ``retouch()`` function."""

    def test_retouch_basic(self, synthetic_face):
        result = retouch(synthetic_face)
        _assert_valid_output(result, synthetic_face)

    def test_retouch_with_kwargs(self, synthetic_face):
        result = retouch(
            synthetic_face,
            smooth=70,
            whiten=40,
            eye_enhance=50,
            contrast=10,
            preset="cosplay",
        )
        _assert_valid_output(result, synthetic_face)

    def test_retouch_context_manager(self, synthetic_face):
        with RetouchEngine() as engine:
            result = engine.process(synthetic_face, preset="natural")
        _assert_valid_output(result, synthetic_face)

    def test_retouch_returns_processing_result(self, synthetic_face):
        result = retouch(synthetic_face)
        assert isinstance(result, ProcessingResult)


# ---------------------------------------------------------------------------
# Real-image integration  (only if test_output/ images exist)
# ---------------------------------------------------------------------------

class TestWithRealImage:
    """Full pipeline tests with a real human face image.

    These tests are skipped if no image is available (e.g. in CI).
    """

    @pytest.fixture
    def real_face(self, natural_image_path):
        if natural_image_path is None:
            pytest.skip("No real face image found (test_output/ is empty)")
        img = cv2.imread(natural_image_path)
        if img is None:
            pytest.skip(f"Could not read image: {natural_image_path}")
        return img

    def test_detects_face(self, engine, real_face):
        """Verify that at least one face is detected in the real image."""
        faces = engine._detector.detect(real_face)
        assert len(faces) >= 1, "No face detected in real test image"

    def test_full_pipeline_real_face(self, engine, real_face):
        result = engine.process(real_face, recipe="natural")
        _assert_valid_output(result, real_face)
        assert result.face_count >= 1

    def test_multiple_recipes_real_face(self, engine, real_face):
        for recipe in ("natural", "cosplay", "pink_dream", "beauty", "film"):
            result = engine.process(real_face, recipe=recipe)
            _assert_valid_output(result, real_face)

    def test_real_face_timings_populated(self, engine, real_face):
        result = engine.process(real_face, recipe="natural")
        expected_stages = {
            "detection", "reshape", "per_face",
            "global", "grading", "finish", "total",
        }
        assert expected_stages.issubset(result.timings.keys()), (
            f"Missing timing stages: {expected_stages - set(result.timings.keys())}"
        )


# ---------------------------------------------------------------------------
# Proxy pipeline — verifies the auto down/up-scaling path for high-res inputs
# (engine.py line 853: ``_process_with_proxy`` / line 883:
# ``_upscale_core_result``)
# ---------------------------------------------------------------------------

# Skip these if model files aren't present (the ``engine`` fixture in
# conftest.py also skips, but pytest still needs to enumerate the class).
_NEEDS_ENGINE = pytest.mark.skipif(
    "not __import__('os').path.exists("
    "__import__('os').path.join("
    "__import__('os').path.dirname(__import__('os').path.dirname(__file__)),"
    "'models', 'face_landmarker.task'))",
    reason="Face landmarker model not present",
)


def _patch_face_pool_to_threaded(engine):
    """Force the multi-face path to use ThreadPoolExecutor fallback.

    The real ``FaceProcessorPool`` spawns child processes which costs
    ~1-2 s and complicates test isolation. Returning ``None`` from
    ``process_faces`` triggers the in-process threaded fallback in
    ``_stage_per_face``.
    """
    engine._face_pool.process_faces = lambda payloads: None


def _mock_detector_for_faces(engine, faces, h, w):
    """Replace ``_detector.detect`` / ``segment_person`` with stubs."""
    engine._detector.detect = lambda img: list(faces)
    engine._detector.segment_person = lambda img: np.ones(
        (h, w), dtype=np.float32
    )


def _fake_parse_batch(regions_factory):
    """Build a ``parse_batch`` stub that returns one regions per crop."""
    def _stub(*args, **kwargs):
        crops = args[0] if args else kwargs.get("crop_list", [])
        return [regions_factory(c.shape[0], c.shape[1]) for c in crops]
    return _stub


def _zero_intensity_overrides():
    """Disable every retouch effect for predictable, fast tests."""
    return dict(
        smooth=0.0, whiten=0.0, equalize=0.0, blemish=0.0,
        eye_enhance=0.0, dark_circles=0.0, catchlight=0.0,
        lip_enhance=0.0, teeth_whiten=0.0, blush=0.0,
        hair_enhance=0.0, dodge_burn=0.0, relight=0.0,
        specular_bloom=0.0, slimming=0.0, contrast=0.0,
        impact=0.0, sharpen=0.0,
    )


def _make_synthetic_face(cx, cy, w_img, h_img, size=80):
    """Return a FaceData with 478 landmarks scattered near (cx, cy)."""
    from retouch.detection import _Landmark, _LandmarkCompat, FaceData

    lm_list = []
    for i in range(478):
        lx = (cx + ((i % 17) - 8) * 2.0) / w_img
        ly = (cy + ((i % 13) - 6) * 2.0) / h_img
        lx = float(max(0.01, min(0.99, lx)))
        ly = float(max(0.01, min(0.99, ly)))
        lm_list.append(_Landmark(lx, ly, 0.0))
    compat = _LandmarkCompat(lm_list)
    x = max(0, cx - size // 2)
    y = max(0, cy - size // 2)
    return FaceData(landmarks=compat, bbox=(x, y, size, size), ied=20.0)


@_NEEDS_ENGINE
class TestProxyPipeline:
    """High-resolution images (>PROXY_MAX_DIM) must round-trip through the
    proxy down/up-scaling helper without resolution loss or crash."""

    def test_proxy_pipeline_triggers_for_large_image(self, engine, monkeypatch):
        # Track whether _upscale_core_result was called — it is only called
        # when the proxy was actually triggered (proxy_scale < 1.0).
        upscale_calls = {"n": 0}
        original = RetouchEngine._upscale_core_result

        def _wrapped(core, target_h, target_w):
            upscale_calls["n"] += 1
            return original(core, target_h, target_w)

        monkeypatch.setattr(RetouchEngine, "_upscale_core_result", staticmethod(_wrapped))

        # 3000x2000 is well above the 2048 PROXY_MAX_DIM threshold
        img = np.full((2000, 3000, 3), 128, dtype=np.uint8)
        result = engine.process(img, recipe="natural")

        # Output must round-trip to the input resolution
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3
        # No faces in a blank image → face_count == 0
        assert result.face_count == 0
        # Detection stage should have run and produced timings
        assert "detection" in result.timings
        # The proxy path must have triggered at least once
        assert upscale_calls["n"] >= 1, (
            "Proxy pipeline did not invoke _upscale_core_result — "
            "_process_with_proxy may not be downscaling large images."
        )

    def test_proxy_pipeline_skipped_for_small_image(self, engine, monkeypatch):
        # Track whether _upscale_core_result was called — for a 1024-px
        # input it must NOT be called (proxy_scale stays at 1.0).
        upscale_calls = {"n": 0}
        original = RetouchEngine._upscale_core_result

        def _wrapped(core, target_h, target_w):
            upscale_calls["n"] += 1
            return original(core, target_h, target_w)

        monkeypatch.setattr(RetouchEngine, "_upscale_core_result", staticmethod(_wrapped))

        # 1024x1024 is well below the 2048 threshold
        img = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        result = engine.process(img, recipe="natural")

        assert result.shape == (1024, 1024, 3)
        assert result.dtype == np.uint8
        assert result.face_count == 0
        # Proxy must NOT have triggered for a small image
        assert upscale_calls["n"] == 0, (
            f"_upscale_core_result called {upscale_calls['n']} times for a "
            f"1024-px input — proxy should be skipped when max(h,w) <= 2048."
        )


# ---------------------------------------------------------------------------
# Multi-face processing — the parallel / threaded per-face path
# ---------------------------------------------------------------------------


@_NEEDS_ENGINE
class TestMultiFaceProcessing:
    """End-to-end test that the engine handles 2 synthetic faces through
    the same per-face code-path used for real multi-face photos."""

    def test_multi_face_two_faces(self, engine, monkeypatch):
        from retouch.parsing import FaceRegions

        h_img, w_img = 600, 600
        img = np.full((h_img, w_img, 3), 180, dtype=np.uint8)

        # Two synthetic faces in the left and right halves of the image
        face1 = _make_synthetic_face(150, 300, w_img, h_img, size=100)
        face2 = _make_synthetic_face(450, 300, w_img, h_img, size=100)
        _mock_detector_for_faces(engine, [face1, face2], h_img, w_img)
        _patch_face_pool_to_threaded(engine)

        def _make_regions(crop_h, crop_w):
            r = FaceRegions()
            r.skin = np.full((crop_h, crop_w), 0.3, dtype=np.float32)
            zeros = np.zeros((crop_h, crop_w), dtype=np.float32)
            r.hair = zeros.copy()
            r.lips = zeros.copy()
            r.neck = zeros.copy()
            r.left_eye = zeros.copy()
            r.right_eye = zeros.copy()
            r.left_under_eye = zeros.copy()
            r.right_under_eye = zeros.copy()
            r.left_eyebrow = zeros.copy()
            r.right_eyebrow = zeros.copy()
            r.nose = zeros.copy()
            r.face_oval = zeros.copy()
            r.mouth_interior = zeros.copy()
            r.left_iris = zeros.copy()
            r.right_iris = zeros.copy()
            return r

        monkeypatch.setattr(engine._parser, "parse_batch", _fake_parse_batch(_make_regions))

        result = engine.process(img, recipe="natural", **_zero_intensity_overrides())

        assert result.shape == (h_img, w_img, 3)
        assert result.dtype == np.uint8
        # Both mocked faces must be reflected in face_count
        assert result.face_count == 2
        # The per-face timing should be populated for a 2-face run
        assert "per_face" in result.timings
        assert "detection" in result.timings

    def test_multi_face_composite_returns_full_canvas(self, engine, monkeypatch):
        """Verify the composite step merges the per-face ROI canvases back
        into a result whose shape equals the input image."""
        from retouch.parsing import FaceRegions

        h_img, w_img = 500, 500
        img = np.full((h_img, w_img, 3), 200, dtype=np.uint8)
        faces = [
            _make_synthetic_face(120, 250, w_img, h_img, size=100),
            _make_synthetic_face(380, 250, w_img, h_img, size=100),
        ]
        _mock_detector_for_faces(engine, faces, h_img, w_img)
        _patch_face_pool_to_threaded(engine)

        def _regions(crop_h, crop_w):
            r = FaceRegions()
            r.skin = np.zeros((crop_h, crop_w), dtype=np.float32)
            zeros = np.zeros((crop_h, crop_w), dtype=np.float32)
            for attr in (
                "hair", "lips", "neck", "left_eye", "right_eye",
                "left_under_eye", "right_under_eye", "left_eyebrow",
                "right_eyebrow", "nose", "face_oval", "mouth_interior",
                "left_iris", "right_iris",
            ):
                setattr(r, attr, zeros.copy())
            return r

        monkeypatch.setattr(engine._parser, "parse_batch", _fake_parse_batch(_regions))

        result = engine.process(img, recipe="natural", **_zero_intensity_overrides())
        # Composite must not change the image dimensions
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# face_contexts caching — the second call must skip detection + parsing
# ---------------------------------------------------------------------------


@_NEEDS_ENGINE
class TestFaceContextsCaching:
    """When ``face_contexts`` is provided to ``process()``, the engine must
    reuse the cached ``FaceData``/``FaceRegions`` and avoid running
    detection or parsing again."""

    def test_caching_path_with_face_contexts(self, engine, monkeypatch):
        from retouch.parsing import FaceRegions

        h_img, w_img = 400, 400
        img = np.full((h_img, w_img, 3), 180, dtype=np.uint8)

        face = _make_synthetic_face(200, 200, w_img, h_img, size=120)
        detect_calls = {"n": 0}
        parse_calls = {"n": 0}

        def _detect_with_counter(*args, **kwargs):
            detect_calls["n"] += 1
            return [face]

        def _segment(img):
            return np.ones((h_img, w_img), dtype=np.float32)

        def _regions(crop_h, crop_w):
            parse_calls["n"] += 1
            r = FaceRegions()
            r.skin = np.full((crop_h, crop_w), 0.3, dtype=np.float32)
            zeros = np.zeros((crop_h, crop_w), dtype=np.float32)
            for attr in (
                "hair", "lips", "neck", "left_eye", "right_eye",
                "left_under_eye", "right_under_eye", "left_eyebrow",
                "right_eyebrow", "nose", "face_oval", "mouth_interior",
                "left_iris", "right_iris",
            ):
                setattr(r, attr, zeros.copy())
            return r

        monkeypatch.setattr(engine._detector, "detect", _detect_with_counter)
        monkeypatch.setattr(engine._detector, "segment_person", _segment)
        _patch_face_pool_to_threaded(engine)
        monkeypatch.setattr(engine._parser, "parse_batch", _fake_parse_batch(_regions))

        kwargs = _zero_intensity_overrides()

        # ---- First call: full detection + parsing pipeline ----
        result1 = engine.process(img, recipe="natural", **kwargs)
        assert result1.face_count == 1
        first_detect = detect_calls["n"]
        first_parse = parse_calls["n"]
        assert first_detect == 1
        assert first_parse == 1
        assert result1.face_contexts is not None
        assert len(result1.face_contexts) == 1

        # ---- Second call: pass cached face_contexts ----
        cached = result1.face_contexts
        result2 = engine.process(
            img, recipe="natural", face_contexts=cached, **kwargs,
        )
        # Detection must NOT have been called a second time
        assert detect_calls["n"] == first_detect, (
            f"Detector was re-invoked despite cached face_contexts: "
            f"{detect_calls['n']} calls vs expected {first_detect}."
        )
        # Parsing must NOT have been called a second time
        assert parse_calls["n"] == first_parse, (
            f"Parser was re-invoked despite cached face_contexts: "
            f"{parse_calls['n']} calls vs expected {first_parse}."
        )
        # Output should still be valid and report the same face count
        assert result2.face_count == 1
        assert result2.shape == (h_img, w_img, 3)
        assert result2.dtype == np.uint8
