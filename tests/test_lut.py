"""Tests for retouch/lut.py — CubeLUT, load_cube, trilinear_sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import retouch.lut as lut_mod
from retouch.lut import (
    CubeLUT,
    list_available_luts,
    load_3dl,
    load_cube,
    luts_dir,
    trilinear_sample,
)


def _identity_lut(n: int) -> np.ndarray:
    lut = np.empty((n, n, n, 3), dtype=np.float32)
    axis = np.arange(n, dtype=np.float32) / float(n - 1)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    lut[..., 0] = b
    lut[..., 1] = g
    lut[..., 2] = r
    return lut


class TestCubeLUT:
    def test_identity_construct_typical(self):
        lut = CubeLUT(33)
        assert lut.size == 33
        assert lut.array.shape == (33, 33, 33, 3)
        assert lut.array.dtype == np.float32

    def test_identity_construct_min(self):
        lut = CubeLUT(2)
        assert lut.array.shape == (2, 2, 2, 3)

    def test_identity_construct_max(self):
        lut = CubeLUT(64)
        assert lut.array.shape == (64, 64, 64, 3)

    def test_identity_diagonal_values(self):
        lut = CubeLUT(5)
        expected = np.arange(5, dtype=np.float32) / 4.0
        np.testing.assert_allclose(lut.array[0, 0, 0], (0.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(lut.array[4, 4, 4], (1.0, 1.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(lut.array[2, 0, 0], (0.5, 0.0, 0.0), atol=1e-6)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            CubeLUT(1)

    def test_invalid_array_shape_raises(self):
        with pytest.raises(ValueError):
            CubeLUT(np.zeros((2, 2, 2, 4), dtype=np.float32))

    def test_non_cubic_array_raises(self):
        with pytest.raises(ValueError):
            CubeLUT(np.zeros((2, 3, 2, 3), dtype=np.float32))

    def test_invalid_source_type_raises(self):
        with pytest.raises(TypeError):
            CubeLUT(2.5)

    def test_array_constructor_makes_copy(self):
        src = _identity_lut(4)
        lut = CubeLUT(src)
        src[0, 0, 0, 0] = 0.42
        assert lut.array[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_identity_apply_uint8(self):
        lut = CubeLUT(33)
        img = np.array(
            [
                [[0, 0, 0], [255, 255, 255], [128, 64, 200]],
                [[10, 20, 30], [200, 100, 50], [64, 128, 192]],
            ],
            dtype=np.uint8,
        )
        out = lut.apply(img)
        assert out.dtype == np.uint8
        assert out.shape == img.shape
        np.testing.assert_allclose(out, img, atol=1)

    def test_identity_apply_random_uint8(self):
        lut = CubeLUT(17)
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
        out = lut.apply(img)
        np.testing.assert_allclose(out, img, atol=1)

    def test_apply_rejects_bad_shape(self):
        lut = CubeLUT(4)
        with pytest.raises(ValueError):
            lut.apply(np.zeros((10, 10), dtype=np.uint8))


class TestTrilinearSample:
    def test_2x2x2_corners_and_center(self):
        lut = np.zeros((2, 2, 2, 3), dtype=np.float32)
        lut[0, 0, 0, :] = (0.0, 0.0, 0.0)
        lut[0, 0, 1, :] = (0.0, 0.0, 0.2)
        lut[0, 1, 0, :] = (0.0, 0.3, 0.0)
        lut[0, 1, 1, :] = (0.0, 0.3, 0.2)
        lut[1, 0, 0, :] = (0.5, 0.0, 0.0)
        lut[1, 0, 1, :] = (0.5, 0.0, 0.2)
        lut[1, 1, 0, :] = (0.5, 0.3, 0.0)
        lut[1, 1, 1, :] = (0.5, 0.3, 0.2)
        pts = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]],
            ],
            dtype=np.float32,
        )
        out = trilinear_sample(lut, pts)
        assert out.shape == (3, 2, 3)
        np.testing.assert_allclose(out[0, 0], (0.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(out[0, 1], (0.5, 0.3, 0.2), atol=1e-6)
        np.testing.assert_allclose(out[1, 0], (0.25, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(out[1, 1], (0.0, 0.15, 0.0), atol=1e-6)
        np.testing.assert_allclose(out[2, 0], (0.0, 0.0, 0.1), atol=1e-6)
        np.testing.assert_allclose(out[2, 1], (0.25, 0.15, 0.1), atol=1e-6)

    def test_clip_out_of_range_input(self):
        lut = np.full((2, 2, 2, 3), 0.5, dtype=np.float32)
        oob = np.array([[[-0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]], dtype=np.float32)
        boundary = np.array([[[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]]], dtype=np.float32)
        out_oob = trilinear_sample(lut, oob)
        out_b = trilinear_sample(lut, boundary)
        np.testing.assert_allclose(out_oob, out_b, atol=1e-6)

    def test_identity_image_passthrough(self):
        lut = _identity_lut(5)
        bgr = np.full((10, 12, 3), 0.5, dtype=np.float32)
        out = trilinear_sample(lut, bgr)
        assert out.shape == (10, 12, 3)
        np.testing.assert_allclose(out, bgr, atol=1e-5)

    def test_invalid_lut_shape(self):
        with pytest.raises(ValueError):
            trilinear_sample(np.zeros((2, 2, 2, 4), dtype=np.float32),
                             np.zeros((4, 4, 3), dtype=np.float32))

    def test_invalid_bgr_shape(self):
        with pytest.raises(ValueError):
            trilinear_sample(_identity_lut(3),
                             np.zeros((4, 4), dtype=np.float32))


class TestLoadCube:
    CUBE_2 = (
        'TITLE "Test LUT"\n'
        "LUT_3D_SIZE 2\n"
        "DOMAIN_MIN 0.0 0.0 0.0\n"
        "DOMAIN_MAX 1.0 1.0 1.0\n"
        "0.0 0.0 0.0\n"
        "0.2 0.0 0.0\n"
        "0.0 0.3 0.0\n"
        "0.2 0.3 0.0\n"
        "0.0 0.0 0.5\n"
        "0.2 0.0 0.5\n"
        "0.0 0.3 0.5\n"
        "0.2 0.3 0.5\n"
    )

    def test_parses_small_cube(self, tmp_path):
        p = tmp_path / "test.cube"
        p.write_text(self.CUBE_2)
        lut = load_cube(p)
        assert isinstance(lut, CubeLUT)
        assert lut.size == 2
        np.testing.assert_allclose(lut.array[0, 0, 0], (0.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(lut.array[1, 1, 1], (0.5, 0.3, 0.2), atol=1e-6)
        np.testing.assert_allclose(lut.array[0, 0, 1], (0.0, 0.0, 0.2), atol=1e-6)
        np.testing.assert_allclose(lut.array[1, 0, 0], (0.5, 0.0, 0.0), atol=1e-6)

    def test_handles_comments_and_blank_lines(self, tmp_path):
        text = (
            "# A header comment\n"
            "\n"
            "LUT_3D_SIZE 2\n"
            "0.0 0.0 0.0\n"
            "0.0 0.0 0.0\n"
            "0.0 0.0 0.0\n"
            "0.0 0.0 0.0\n"
            "0.0 0.0 0.0\n"
            "0.0 0.0 0.0\n"
            "0.0 0.0 0.0\n"
            "1.0 1.0 1.0\n"
        )
        p = tmp_path / "test.cube"
        p.write_text(text)
        lut = load_cube(p)
        assert lut.size == 2
        np.testing.assert_allclose(lut.array[1, 1, 1], (1.0, 1.0, 1.0), atol=1e-6)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cube(tmp_path / "missing.cube")

    def test_missing_size_raises(self, tmp_path):
        p = tmp_path / "bad.cube"
        p.write_text("0.0 0.0 0.0\n")
        with pytest.raises(ValueError):
            load_cube(p)

    def test_wrong_count_raises(self, tmp_path):
        p = tmp_path / "bad.cube"
        p.write_text("LUT_3D_SIZE 2\n0.0 0.0 0.0\n")
        with pytest.raises(ValueError):
            load_cube(p)

    def test_1d_lut_rejected(self, tmp_path):
        p = tmp_path / "bad.cube"
        p.write_text("LUT_1D_SIZE 2\n")
        with pytest.raises(ValueError):
            load_cube(p)

    def test_malformed_data_line(self, tmp_path):
        p = tmp_path / "bad.cube"
        p.write_text("LUT_3D_SIZE 2\nnot a triplet\n")
        with pytest.raises(ValueError):
            load_cube(p)

    def test_string_path_accepted(self, tmp_path):
        p = tmp_path / "test.cube"
        p.write_text(self.CUBE_2)
        lut = load_cube(str(p))
        assert lut.size == 2


class TestLutsDir:
    def test_canonical_path(self):
        p = luts_dir()
        assert p == Path(__file__).resolve().parent.parent / "luts"
        assert p.exists()
        assert p.is_dir()

    def test_idempotent_create(self, tmp_path, monkeypatch):
        target = tmp_path / "fresh_luts"
        assert not target.exists()
        target.mkdir(parents=True, exist_ok=True)
        assert target.exists()
        target.mkdir(parents=True, exist_ok=True)
        assert target.exists()


class TestListAvailableLuts:
    def test_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lut_mod, "luts_dir", lambda: tmp_path)
        assert list_available_luts() == []

    def test_returns_sorted_stems(self, tmp_path, monkeypatch):
        (tmp_path / "zeta.cube").write_text("LUT_3D_SIZE 2\n")
        (tmp_path / "alpha.cube").write_text("LUT_3D_SIZE 2\n")
        (tmp_path / "ignore.txt").write_text("not a lut")
        (tmp_path / "beta.cube").write_text("LUT_3D_SIZE 2\n")
        monkeypatch.setattr(lut_mod, "luts_dir", lambda: tmp_path)
        result = list_available_luts()
        assert result == ["alpha", "beta", "zeta"]


class TestLoad3dl:
    def test_raises_not_implemented(self, tmp_path):
        p = tmp_path / "x.3dl"
        p.write_text("dummy")
        with pytest.raises(NotImplementedError):
            load_3dl(p)
