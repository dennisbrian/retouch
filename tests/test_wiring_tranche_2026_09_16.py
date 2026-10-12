"""Regression tests for the 2026-09-16 wiring tranche.

Covers:
- BB3: EXR float32 read/write round-trip (io.py)
- K7:  PQ-encoded EXR export (io.py + color_science.linear_to_pq)
- K6:  multi-illuminant skin adaptation ParamSpec + engine no-op default
- K1:  CAM16-UCS delta-E QA detector (qa_detectors.detect_cam16_delta_e)
- BB6: CLI --preflight-check argument registration
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from retouch.io import read_image_16bit, write_image_with_icc  # noqa: E402
from retouch.params import PROCESSING_PARAMS  # noqa: E402
from retouch.qa_detectors import (  # noqa: E402
    ALL_DETECTOR_NAMES,
    detect_cam16_delta_e,
    run_all,
)


class TestExrRoundTrip:
    """BB3: EXR float32 scene-linear reader/writer."""

    def test_sdr_roundtrip_lossless(self, tmp_path):
        img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        path = tmp_path / "sdr.exr"
        write_image_with_icc(str(path), img)
        assert path.exists() and path.stat().st_size > 0
        back = read_image_16bit(path)
        assert back.shape == img.shape
        assert back.dtype == np.float32
        err = np.abs(back - img.astype(np.float32))
        assert err.max() < 0.01  # float noise only, no quantization

    def test_hdr_shoulder_not_clipped_to_flat(self, tmp_path):
        # Linear values above 1.0 must compress smoothly, not hard-clip.
        hdr = np.float32([[[0.5, 1.0, 4.0]]])  # BGR
        path = tmp_path / "hdr.exr"
        assert cv2.imwrite(str(path), hdr)
        out = read_image_16bit(path)
        # B=0.5 linear -> sRGB(0.5)*255 ~ 187.5; G=1.0 -> 255; R=4.0 -> 255
        assert out[0, 0, 0] == pytest.approx(187.5, abs=1.0)
        assert out[0, 0, 1] == pytest.approx(255.0, abs=0.5)
        assert out[0, 0, 2] == pytest.approx(255.0, abs=0.5)

    def test_decode_failure_raises(self, tmp_path):
        bad = tmp_path / "bad.exr"
        bad.write_bytes(b"not an exr file")
        with pytest.raises(ValueError):
            read_image_16bit(bad)


class TestExrPqExport:
    """K7: ST 2084 PQ-encoded EXR export."""

    def test_pq_values_in_unit_range(self, tmp_path):
        from retouch.color_science import pq_to_linear

        img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        path = tmp_path / "pq.exr"
        write_image_with_icc(str(path), img, pq_encode=True)
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert raw is not None
        assert raw.min() >= 0.0 and raw.max() <= 1.0
        # Decoding returns scene-linear values inside [0, 1].
        lin = pq_to_linear(raw)
        assert lin.min() >= 0.0 and lin.max() <= 1.0001

    def test_plain_exr_is_linear_not_pq(self, tmp_path):
        img = np.full((8, 8, 3), 255, dtype=np.uint8)  # white
        path = tmp_path / "lin.exr"
        write_image_with_icc(str(path), img)
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        # Linear white = 1.0; PQ(1.0) would be ~0.99 (close but the sRGB
        # decode of 255 is 1.0 exactly, so linear storage must be exactly 1).
        assert raw.max() == pytest.approx(1.0, abs=1e-4)


class TestMultiIlluminantParams:
    """K6: ParamSpec registration + neutral-default engine no-op contract."""

    def test_params_registered(self):
        names = {s.name for s in PROCESSING_PARAMS}
        assert "multi_illuminant_key_kelvin" in names
        assert "multi_illuminant_fill_kelvin" in names
        assert "multi_illuminant_mix" in names

    def test_neutral_defaults(self):
        defaults = {s.name: s.default for s in PROCESSING_PARAMS}
        assert defaults["multi_illuminant_key_kelvin"] == 6500
        assert defaults["multi_illuminant_fill_kelvin"] == 6500
        assert defaults["multi_illuminant_mix"] == 0.0

    def test_adaptation_changes_skin_when_active(self):
        from retouch.color_science import adapt_multi_illuminant_skin

        img = np.full((16, 16, 3), 150, dtype=np.uint8)
        skin = np.ones((16, 16), dtype=np.float32)
        out = adapt_multi_illuminant_skin(
            img, skin_mask=skin,
            key_wp=(0.95047, 1.0, 1.08883),
            fill_wp=(0.6, 1.0, 1.8),  # cool fill
            mix_factor=0.5,
        )
        assert out.shape == img.shape
        assert not np.array_equal(out, img)

    def test_equal_white_points_is_identity(self):
        from retouch.color_science import adapt_multi_illuminant_skin

        img = np.full((8, 8, 3), 150, dtype=np.uint8)
        out = adapt_multi_illuminant_skin(
            img, skin_mask=np.ones((8, 8), np.float32),
            key_wp=(0.95047, 1.0, 1.08883),
            fill_wp=(0.95047, 1.0, 1.08883),
            mix_factor=0.5,
        )
        assert np.array_equal(out, img)

    def test_zero_skin_mask_is_identity(self):
        """Regression for the 2026-09-24 fix: pixels outside skin_mask must
        be left untouched. Before the fix, `adapted = img_key*(1-m) +
        img_fill*m` had no third "leave alone" state, so m==0 fell back to
        img_key (the FULL key-illuminant CAT16 adaptation), not the
        original pixel — a frame-wide color cast over hair/background/
        clothing on every render with a distinct key/fill kelvin, confirmed
        visually (hard blue cast, DSCF8481 render, 2026-09-24)."""
        from retouch.color_science import adapt_multi_illuminant_skin

        img = np.full((8, 8, 3), 150, dtype=np.uint8)
        out = adapt_multi_illuminant_skin(
            img, skin_mask=np.zeros((8, 8), np.float32),
            key_wp=(0.6, 1.0, 1.8),  # distinct key/fill, so the bug would fire
            fill_wp=(0.95047, 1.0, 1.08883),
            mix_factor=0.6,
        )
        assert np.array_equal(out, img)

    def test_partial_skin_mask_leaves_outside_unchanged(self):
        """Same regression, on a spatially partial mask (the realistic
        case — acc_skin covers only part of the frame)."""
        from retouch.color_science import adapt_multi_illuminant_skin

        img = np.full((8, 8, 3), 150, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[:4, :4] = 1.0  # skin in the top-left quadrant only
        out = adapt_multi_illuminant_skin(
            img, skin_mask=mask,
            key_wp=(0.6, 1.0, 1.8),
            fill_wp=(0.95047, 1.0, 1.08883),
            mix_factor=0.6,
        )
        assert np.array_equal(out[4:, 4:], img[4:, 4:])
        assert not np.array_equal(out[:4, :4], img[:4, :4])


class TestCam16DeltaEDetector:
    """K1: CAM16-UCS delta-E QA detector."""

    def test_in_all_detector_names(self):
        assert "cam16_delta_e" in ALL_DETECTOR_NAMES

    def test_identity_is_zero(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        r = detect_cam16_delta_e(img, img.copy(), np.ones((32, 32), np.float32))
        assert r["mean_delta_e"] == 0.0
        assert r["flagged"] is False

    def test_big_shift_flags(self):
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        ref = np.full((32, 32, 3), 200, dtype=np.uint8)
        r = detect_cam16_delta_e(img, ref, np.ones((32, 32), np.float32))
        assert r["flagged"] is True
        assert r["status"] == "checked-flagged"

    def test_moderate_shift_passes(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        shifted = img.copy()
        shifted[..., 2] = 180
        r = detect_cam16_delta_e(shifted, img, np.ones((32, 32), np.float32))
        assert r["flagged"] is False
        assert 0.0 < r["mean_delta_e"] < 10.0

    def test_run_all_includes_detector_with_reference(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        r = run_all(img, reference_img_bgr=img.copy())
        assert "cam16_delta_e" in r
        assert r["cam16_delta_e"]["mean_delta_e"] == 0.0

    def test_shape_mismatch_is_not_run(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        ref = np.full((16, 16, 3), 128, dtype=np.uint8)
        r = detect_cam16_delta_e(img, ref)
        assert r["status"] == "not-run"


class TestGamutTarget:
    """K5: selectable output gamut for the K3 chroma-compression knee."""

    def test_param_registered_with_choices(self):
        spec = next(s for s in PROCESSING_PARAMS if s.name == "gamut_target")
        assert spec.default == "srgb"
        assert set(spec.choices) == {"srgb", "p3", "rec2020"}

    def test_wider_target_retains_more_chroma(self):
        from retouch.color_science import gamut_compress

        # Out-of-gamut over-saturated chroma; radians hue like oklch_to_oklab.
        oklch = np.array([[[0.7, 0.35, np.radians(29.0)]]], dtype=np.float32)
        c_srgb = float(gamut_compress(oklch, target="srgb")[0, 0, 1])
        c_p3 = float(gamut_compress(oklch, target="p3")[0, 0, 1])
        c_r2020 = float(gamut_compress(oklch, target="rec2020")[0, 0, 1])
        assert c_srgb < c_p3 < c_r2020

    def test_invalid_target_raises(self):
        from retouch.color_science import gamut_compress

        oklch = np.array([[[0.7, 0.2, np.radians(29.0)]]], dtype=np.float32)
        with pytest.raises(ValueError):
            gamut_compress(oklch, target="prophoto")

    def test_grader_accepts_target(self):
        from retouch.grading import ColorGrader

        grader = ColorGrader()
        # uint8 input short-circuits (always in-gamut) but must accept the kwarg.
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        for target in ("srgb", "p3", "rec2020"):
            out = grader._apply_gamut_compress(img, target=target)
            assert np.array_equal(out, img)


class TestCliPreflightFlag:
    """BB6: --preflight-check registers as an opt-in exit-early mode."""

    def test_flag_registered(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "cli.py", "--preflight-check"],
            capture_output=True, text=True, timeout=300,
            env={**__import__("os").environ,
                 "RETOUCH_GPU": "0", "RETOUCH_MEDIAPIPE_BACKEND": "legacy"},
        )
        assert result.returncode == 0, result.stderr[-500:]
        assert "color_science_oklab" in result.stdout
        assert "qa_detectors_run_all" in result.stdout
        assert "Pre-flight checks passed" in result.stdout
