"""Tests for retouch/look_extractor.py — LookExtractor (F6)."""

from __future__ import annotations

import numpy as np
import pytest

from retouch.look_extractor import LookExtractor
from retouch.grading import ColorGrader


def _gradient(h: int = 256, w: int = 256) -> np.ndarray:
    yy = np.linspace(0, 255, w, dtype=np.float32)
    xx = np.linspace(0, 255, h, dtype=np.float32)
    grid = np.minimum(xx[:, None], yy[None, :])
    img = np.stack([grid, grid, grid], axis=-1).astype(np.uint8)
    return np.ascontiguousarray(img)


def _color_gradient(h: int = 256, w: int = 256) -> np.ndarray:
    yy = np.linspace(0, h - 1, h, dtype=np.float32)
    xx = np.linspace(0, w - 1, w, dtype=np.float32)
    r = np.tile(xx, (h, 1)) * 0.75 + 40
    g = np.tile(yy[:, None], (1, w)) * 0.55 + 30
    b = 260 - np.tile(xx, (h, 1)) * 0.75
    img = np.stack([b, g, r], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_extract_board_weighted_references_returns_board_mode():
    le = LookExtractor()
    warm = np.full((64, 64, 3), (80, 100, 180), dtype=np.uint8)
    cool = np.full((64, 64, 3), (180, 100, 80), dtype=np.uint8)

    result = le.extract_board([(warm, 0.75), (cool, 0.25)])

    assert result["mode"] == "board"
    assert result["reference_count"] == 2
    assert result["weights"] == [0.75, 0.25]
    assert "saturation" in result["engine_params"]


class TestExtractReturnsValidParams:
    def test_returns_dict_with_expected_keys(self):
        le = LookExtractor()
        ref = _gradient()
        result = le.extract(ref)
        assert isinstance(result, dict)
        assert set(result.keys()) >= {"engine_params", "preset", "mode"}
        assert result["mode"] == "unpaired"

    def test_engine_params_are_scalars(self):
        le = LookExtractor()
        ref = _gradient()
        params = le.extract(ref)["engine_params"]
        assert isinstance(params, dict)
        for k in ("shadows", "highlights", "whites", "blacks", "contrast", "brightness"):
            assert k in params, f"missing {k}"
            assert isinstance(params[k], (int, float))

    def test_preset_has_curves(self):
        le = LookExtractor()
        ref = _color_gradient()
        preset = le.extract(ref)["preset"]
        assert "curves" in preset
        assert "L" in preset["curves"]
        pts = preset["curves"]["L"]
        assert isinstance(pts, list)
        assert len(pts) >= 5
        for p in pts:
            assert len(p) == 2

    def test_curve_is_monotonic(self):
        le = LookExtractor()
        ref = _color_gradient()
        pts = le.extract(ref)["preset"]["curves"]["L"]
        ys = [p[1] for p in pts]
        for i in range(1, len(ys)):
            assert ys[i] >= ys[i - 1]

    def test_paired_mode(self):
        le = LookExtractor()
        base = _gradient()
        ref = _color_gradient()
        result = le.extract(ref, base_img=base)
        assert result["mode"] == "paired"
        assert "engine_params" in result

    def test_handles_float32_input(self):
        le = LookExtractor()
        ref = _gradient().astype(np.float32)
        result = le.extract(ref)
        assert "engine_params" in result

    def test_raises_on_bad_shape(self):
        le = LookExtractor()
        with pytest.raises(ValueError):
            le.extract(np.zeros((10, 10), dtype=np.uint8))


class TestApplyingParamsMovesTowardReference:
    def test_brightness_param_brightens_when_reference_is_brighter(self):
        le = LookExtractor()
        base = np.full((64, 64, 3), 80, dtype=np.uint8)
        ref = np.full((64, 64, 3), 200, dtype=np.uint8)
        params = le.extract(ref, base_img=base)["engine_params"]
        assert params["brightness"] > 0

    def test_saturation_param_increases_when_reference_is_more_saturated(self):
        le = LookExtractor()
        base = np.full((64, 64, 3), 128, dtype=np.uint8)
        ref = np.zeros((64, 64, 3), dtype=np.uint8)
        ref[:, :, 2] = 220  # saturated red
        ref[:, :, 1] = 60
        params = le.extract(ref, base_img=base)["engine_params"]
        assert params["saturation"] > 0

    def test_warmth_indicated_for_warm_reference(self):
        le = LookExtractor()
        base = np.full((64, 64, 3), 128, dtype=np.uint8)
        ref = np.full((64, 64, 3), 128, dtype=np.uint8)
        ref[:, :, 2] = 200  # warm red channel
        ref[:, :, 0] = 60   # low blue
        params = le.extract(ref, base_img=base)["engine_params"]
        assert "white_balance_kelvin" in params
        assert params["white_balance_kelvin"] < 6500

    def test_applying_extracted_grade_reduces_distance(self):
        cg = ColorGrader()
        base = _color_gradient(128, 128)
        target_preset = {
            "curves": {"L": [[0, 0], [64, 50], [128, 120], [192, 210], [255, 255]]},
            "warmth": 0.3,
            "saturation_boost": 0.2,
            "vignette": 0,
            "grain": 0,
        }
        reference = cg.grade(base, preset=target_preset, intensity=1.0)

        le = LookExtractor()
        extracted = le.extract(reference, base_img=base)
        extracted_preset = extracted["preset"]

        def _distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
            a = img_a.astype(np.float32)
            b = img_b.astype(np.float32)
            return float(np.mean(np.abs(a - b)))

        baseline_dist = _distance(base, reference)
        graded = cg.grade(base, preset=extracted_preset, intensity=1.0)
        graded_dist = _distance(graded, reference)

        assert graded_dist < baseline_dist, (
            f"graded {graded_dist:.2f} should be < baseline {baseline_dist:.2f}"
        )


class TestRoundTrip:
    def test_extract_from_known_preset_recovers_warmth_sign(self):
        cg = ColorGrader()
        base = _color_gradient(128, 128)
        warm_preset = {"warmth": 0.5, "saturation_boost": 0.1}
        ref = cg.grade(base, preset=warm_preset, intensity=1.0)

        le = LookExtractor()
        result = le.extract(ref, base_img=base)
        ep = result["engine_params"]

        if "white_balance_kelvin" in ep:
            assert ep["white_balance_kelvin"] < 6500, "warm ref should lower kelvin"

    def test_extract_from_grayscale_reference_no_film(self):
        le = LookExtractor()
        ref = _gradient(128, 128)
        params = le.extract(ref)["engine_params"]
        assert params.get("film_enable", False) is False

    def test_extract_is_deterministic(self):
        le = LookExtractor()
        ref = _color_gradient(128, 128)
        r1 = le.extract(ref)
        r2 = le.extract(ref)
        assert r1["engine_params"] == r2["engine_params"]
        assert r1["preset"] == r2["preset"]


class TestDtypeAndEdgeCases:
    def test_uint8_and_float32_produce_same_params(self):
        le = LookExtractor()
        ref_u8 = _gradient(64, 64)
        ref_f32 = ref_u8.astype(np.float32)
        p_u8 = le.extract(ref_u8)["engine_params"]
        p_f32 = le.extract(ref_f32)["engine_params"]
        for k in ("shadows", "highlights", "contrast", "brightness"):
            assert abs(p_u8[k] - p_f32[k]) < 0.5, f"{k}: {p_u8[k]} vs {p_f32[k]}"

    def test_small_image_does_not_crash(self):
        le = LookExtractor()
        ref = np.full((8, 8, 3), 128, dtype=np.uint8)
        result = le.extract(ref)
        assert "engine_params" in result

    def test_downsample_is_applied_for_large_image(self):
        le = LookExtractor(downsample_dim=64)
        ref = _gradient(512, 512)
        result = le.extract(ref)
        assert "engine_params" in result

    def test_saved_preset_uses_user_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RETOUCH_CACHE_DIR", str(tmp_path / "cache"))
        path = LookExtractor()._save_preset({"description": "test"}, "learned look")
        assert path == tmp_path / "cache" / "presets" / "learned_look.json"
        assert path.exists()
