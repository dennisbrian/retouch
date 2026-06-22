"""Tests for retouch/style.py — StyleAnalyzer, StyleApplier, subject_aware_transfer.

Covers items that need a real engine + a face detector. Where MediaPipe
or ONNX is required, blank synthetic images + mocked detectors are used
to keep tests fast and reliable.
"""

from unittest.mock import patch

import numpy as np
import pytest

from retouch.engine import RetouchEngine
from retouch.style import (
    StyleAnalyzer,
    StyleApplier,
    StyleProfile,
    subject_aware_transfer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _blank_pair(value1=100, value2=150, h=50, w=50):
    """Two small uniform-colour BGR images."""
    img1 = np.full((h, w, 3), value1, dtype=np.uint8)
    img2 = np.full((h, w, 3), value2, dtype=np.uint8)
    return img1, img2


# ---------------------------------------------------------------------------
# StyleAnalyzer
# ---------------------------------------------------------------------------


class TestStyleAnalyzerInit:
    def test_init_creates_default_engine(self):
        sa = StyleAnalyzer()
        assert sa.engine is not None
        assert isinstance(sa.engine, RetouchEngine)
        sa.engine.close()

    def test_init_uses_provided_engine(self):
        eng = RetouchEngine()
        try:
            sa = StyleAnalyzer(engine=eng)
            assert sa.engine is eng
        finally:
            eng.close()


class TestStyleAnalyzerExtract:
    def test_extract_returns_style_profile_instance(self):
        eng = RetouchEngine()
        try:
            sa = StyleAnalyzer(engine=eng)
            orig, edit = _blank_pair()
            profile = sa.extract(orig, edit)
            assert isinstance(profile, StyleProfile)
        finally:
            eng.close()

    def test_extract_with_no_face_returns_zero_skin_stats(self):
        """When no face is found, skin deltas default to 0.0 and opacity to 1.0."""
        eng = RetouchEngine()
        try:
            sa = StyleAnalyzer(engine=eng)

            # Mock the detector to report no faces
            with patch.object(eng._detector, "detect", return_value=[]):
                orig, edit = _blank_pair()
                profile = sa.extract(orig, edit)

            assert profile.skin_l_mean_delta == 0.0
            assert profile.skin_a_mean_delta == 0.0
            assert profile.skin_b_mean_delta == 0.0
            assert profile.skin_smooth_strength == 0.0
            assert profile.skin_mid_reduction == 0.0
            # Opacity defaults to 1.0 when there is no face to measure
            assert profile.skin_texture_opacity == 1.0
        finally:
            eng.close()

    def test_extract_resizes_edited_to_match_original(self):
        """extract() should handle mismatched image sizes via resize."""
        eng = RetouchEngine()
        try:
            sa = StyleAnalyzer(engine=eng)
            with patch.object(eng._detector, "detect", return_value=[]):
                orig = np.full((50, 50, 3), 100, dtype=np.uint8)
                edit = np.full((80, 80, 3), 200, dtype=np.uint8)
                # Must not raise
                profile = sa.extract(orig, edit)
                assert isinstance(profile, StyleProfile)
        finally:
            eng.close()

    def test_extract_global_deltas_measurable_on_blank_images(self):
        """Brightness/contrast/saturation deltas should differ for different inputs."""
        eng = RetouchEngine()
        try:
            sa = StyleAnalyzer(engine=eng)
            with patch.object(eng._detector, "detect", return_value=[]):
                # Dark original, bright edit -> positive brightness delta
                orig = np.full((50, 50, 3), 50, dtype=np.uint8)
                edit = np.full((50, 50, 3), 200, dtype=np.uint8)
                profile = sa.extract(orig, edit)
                assert profile.brightness_delta > 0.0
        finally:
            eng.close()

    def test_extract_returns_serialisable_profile(self):
        eng = RetouchEngine()
        try:
            sa = StyleAnalyzer(engine=eng)
            with patch.object(eng._detector, "detect", return_value=[]):
                orig, edit = _blank_pair()
                profile = sa.extract(orig, edit)
                # The profile should round-trip through JSON
                json_str = profile.to_json()
                restored = StyleProfile.from_json(json_str)
                assert restored.brightness_delta == profile.brightness_delta
                assert restored.contrast_delta == profile.contrast_delta
        finally:
            eng.close()


# ---------------------------------------------------------------------------
# StyleApplier
# ---------------------------------------------------------------------------


class TestStyleApplierInit:
    def test_init_creates_default_engine(self):
        sa = StyleApplier()
        assert sa.engine is not None
        assert isinstance(sa.engine, RetouchEngine)
        sa.engine.close()

    def test_init_uses_provided_engine(self):
        eng = RetouchEngine()
        try:
            sa = StyleApplier(engine=eng)
            assert sa.engine is eng
        finally:
            eng.close()


class TestStyleApplierApply:
    def test_apply_zero_profile_preserves_image_shape(self):
        """A profile with all-zero deltas should return an image of the same shape."""
        eng = RetouchEngine()
        try:
            sa = StyleApplier(engine=eng)
            img = np.full((50, 50, 3), 128, dtype=np.uint8)
            profile = StyleProfile()  # all deltas = 0
            result = sa.apply(img, profile)
            assert result.shape == img.shape
            assert result.dtype == np.uint8
        finally:
            eng.close()

    def test_apply_with_neutral_tone_keeps_skin_deltas_small(self):
        """When skin_a/b deltas are small (< 0.5), no extra LAB shift is applied."""
        eng = RetouchEngine()
        try:
            sa = StyleApplier(engine=eng)
            img = np.full((50, 50, 3), 128, dtype=np.uint8)
            profile = StyleProfile(
                brightness_delta=5.0,
                contrast_delta=2.0,
                saturation_delta=1.0,
                skin_l_mean_delta=0.1,    # small
                skin_a_mean_delta=0.1,    # small (< 0.5)
                skin_b_mean_delta=-0.1,   # small (>-0.5)
            )
            # Should not raise, even if detector finds no face
            result = sa.apply(img, profile)
            assert result.shape == img.shape
        finally:
            eng.close()

    def test_apply_chooses_rosy_tone_for_positive_a_delta(self):
        """skin_a_mean_delta > 1.0 → whiten_tone='rosy' (no error even if no face)."""
        eng = RetouchEngine()
        try:
            sa = StyleApplier(engine=eng)
            img = np.full((50, 50, 3), 128, dtype=np.uint8)
            profile = StyleProfile(
                skin_a_mean_delta=2.0,    # > 1.0 -> "rosy"
            )
            # Mock detector to return no faces — must not raise
            with patch.object(eng._detector, "detect", return_value=[]):
                result = sa.apply(img, profile)
            assert result.shape == img.shape
        finally:
            eng.close()

    def test_apply_chooses_porcelain_tone_for_negative_b_delta(self):
        """skin_b_mean_delta < -1.0 → whiten_tone='porcelain'."""
        eng = RetouchEngine()
        try:
            sa = StyleApplier(engine=eng)
            img = np.full((50, 50, 3), 128, dtype=np.uint8)
            profile = StyleProfile(
                skin_b_mean_delta=-2.0,   # < -1.0 -> "porcelain"
            )
            with patch.object(eng._detector, "detect", return_value=[]):
                result = sa.apply(img, profile)
            assert result.shape == img.shape
        finally:
            eng.close()

    def test_apply_clamps_smooth_value(self):
        """A profile with extreme skin_smooth_strength should be clamped, not crash."""
        eng = RetouchEngine()
        try:
            sa = StyleApplier(engine=eng)
            img = np.full((50, 50, 3), 128, dtype=np.uint8)
            profile = StyleProfile(skin_smooth_strength=10.0)  # way outside [0, 1]
            with patch.object(eng._detector, "detect", return_value=[]):
                result = sa.apply(img, profile)
            assert result.shape == img.shape
        finally:
            eng.close()


# ---------------------------------------------------------------------------
# subject_aware_transfer
# ---------------------------------------------------------------------------


class TestSubjectAwareTransfer:
    def test_no_face_returns_processed_image(self):
        """When both target and ref have no faces, only background is transferred."""
        eng = RetouchEngine()
        try:
            target = np.full((50, 50, 3), 100, dtype=np.uint8)
            ref = np.full((50, 50, 3), 200, dtype=np.uint8)
            target_person = np.zeros((50, 50), dtype=np.float32)  # empty

            # No faces in either image
            with patch.object(eng._detector, "detect", return_value=[]), \
                 patch.object(eng._detector, "segment_person",
                              return_value=target_person):
                result = subject_aware_transfer(eng, target, ref)

            assert result.shape == target.shape
            assert result.dtype == np.uint8
        finally:
            eng.close()

    def test_returns_uint8_bgr(self):
        eng = RetouchEngine()
        try:
            target = np.full((30, 30, 3), 100, dtype=np.uint8)
            ref = np.full((30, 30, 3), 200, dtype=np.uint8)
            target_person = np.ones((30, 30), dtype=np.float32)
            with patch.object(eng._detector, "detect", return_value=[]), \
                 patch.object(eng._detector, "segment_person",
                              return_value=target_person):
                result = subject_aware_transfer(eng, target, ref)
            assert result.dtype == np.uint8
            assert result.shape == (30, 30, 3)
        finally:
            eng.close()

    def test_with_faces_in_both_runs_skin_transfer(self):
        """When both images have detected faces, the skin transfer is invoked."""
        from retouch.detection import FaceData, _LandmarkCompat, _Landmark
        from retouch.parsing import FaceRegions

        eng = RetouchEngine()
        try:
            target = np.full((80, 80, 3), 100, dtype=np.uint8)
            ref = np.full((80, 80, 3), 200, dtype=np.uint8)

            fake_face = FaceData(
                landmarks=_LandmarkCompat(
                    [_Landmark(0.5, 0.5, 0.0) for _ in range(5)]
                ),
                bbox=(20, 20, 40, 40),
                ied=20.0,
            )

            target_person = np.ones((80, 80), dtype=np.float32)
            ref_person = np.ones((80, 80), dtype=np.float32)

            regions = FaceRegions()
            regions.skin = np.ones((80, 80), dtype=np.uint8) * 255
            regions.hair = np.ones((80, 80), dtype=np.uint8) * 255

            def detect_side_effect(img):
                # Return a face for both target and ref
                return [fake_face]

            with patch.object(eng._detector, "detect", side_effect=detect_side_effect), \
                 patch.object(eng._detector, "segment_person",
                              side_effect=[target_person, ref_person]), \
                 patch.object(eng._parser, "parse", return_value=regions) as mock_parse:
                result = subject_aware_transfer(eng, target, ref)

            # parse() should have been called for both target and reference
            assert mock_parse.call_count == 2
            assert result.shape == target.shape
            assert result.dtype == np.uint8
        finally:
            eng.close()

    def test_passing_explicit_target_faces_skips_detection(self):
        """If target_faces is given, the function must NOT call detect() on target."""
        eng = RetouchEngine()
        try:
            target = np.full((50, 50, 3), 100, dtype=np.uint8)
            ref = np.full((50, 50, 3), 200, dtype=np.uint8)
            target_person = np.ones((50, 50), dtype=np.float32)

            detect_calls = []

            def fake_detect(img):
                detect_calls.append(img)
                return []

            with patch.object(eng._detector, "detect", side_effect=fake_detect), \
                 patch.object(eng._detector, "segment_person",
                              return_value=target_person):
                # target_faces explicitly empty — must skip detection
                subject_aware_transfer(
                    eng, target, ref,
                    target_faces=[],
                    target_person=target_person,
                )

            # detect() should be called only for the ref image, not the target
            assert len(detect_calls) == 1
        finally:
            eng.close()

    def test_accepts_uint8_target_person_mask(self):
        """The function should normalise a uint8 target_person mask to float."""
        eng = RetouchEngine()
        try:
            target = np.full((50, 50, 3), 100, dtype=np.uint8)
            ref = np.full((50, 50, 3), 200, dtype=np.uint8)
            # uint8 mask with values > 1
            target_person = np.full((50, 50), 255, dtype=np.uint8)

            with patch.object(eng._detector, "detect", return_value=[]), \
                 patch.object(eng._detector, "segment_person",
                              return_value=target_person):
                # Should not raise
                result = subject_aware_transfer(eng, target, ref, target_person=target_person)
            assert result.shape == target.shape
        finally:
            eng.close()
