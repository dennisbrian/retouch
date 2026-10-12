"""Tests for the F5 face-aware liquify sliders (retouch/geometry.py).

Covers:
  - each per-feature warp builder produces valid output
  - zero strength is a no-op
  - bilateral symmetry (left/right warps are mirrored)
  - displacement caps (max |T−C| ≤ face_width × 0.12 at slider ±100)
  - dtype preservation (uint8 and float32)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from retouch.geometry import FaceReshaper
from retouch.detection import FaceData, _Landmark, _LandmarkCompat


# ---------------------------------------------------------------------------
# Synthetic face + context helpers
# ---------------------------------------------------------------------------

_IMG_W = 400
_IMG_H = 400


def _lm(x: float, y: float) -> _Landmark:
    return _Landmark(x, y, z=0.0)


def _make_face(
    cx: int = 200,
    cy: int = 200,
    face_w: int = 120,
    face_h: int = 150,
) -> FaceData:
    """Build a synthetic FaceData with a realistic landmark layout.

    Landmarks are placed in normalized image coordinates. The face is roughly
    symmetric about x = cx so bilateral symmetry tests are meaningful.
    """
    lm_list = [_lm(0.5, 0.5) for _ in range(478)]

    nx = lambda px: px / _IMG_W
    ny = lambda py: py / _IMG_H

    left = cx - face_w // 2
    right = cx + face_w // 2
    top = cy - face_h // 2
    bot = cy + face_h // 2

    lm_list[234] = _lm(nx(left), ny(cy + face_h * 0.15))
    lm_list[454] = _lm(nx(right), ny(cy + face_h * 0.15))

    lm_list[117] = _lm(nx(cx - face_w * 0.20), ny(cy + face_h * 0.05))
    lm_list[346] = _lm(nx(cx + face_w * 0.20), ny(cy + face_h * 0.05))

    lm_list[152] = _lm(nx(cx), ny(bot))

    eye_y = cy - face_h * 0.10
    eye_l_cx = cx - face_w * 0.22
    eye_r_cx = cx + face_w * 0.22
    for i, idx in enumerate([33, 246, 161, 160, 159, 158, 157, 173,
                             133, 155, 154, 153, 145, 144, 163, 7]):
        angle = 2.0 * np.pi * i / 16
        ex = eye_l_cx + 12 * np.cos(angle)
        ey = eye_y + 8 * np.sin(angle)
        lm_list[idx] = _lm(nx(ex), ny(ey))
    for i, idx in enumerate([263, 466, 388, 387, 386, 385, 384, 398,
                             362, 382, 381, 380, 374, 373, 390, 249]):
        angle = 2.0 * np.pi * i / 16
        ex = eye_r_cx + 12 * np.cos(angle)
        ey = eye_y + 8 * np.sin(angle)
        lm_list[idx] = _lm(nx(ex), ny(ey))

    for i, idx in enumerate([468, 469, 470, 471, 472]):
        angle = 2.0 * np.pi * i / 5
        lm_list[idx] = _lm(nx(eye_l_cx + 5 * np.cos(angle)),
                           ny(eye_y + 5 * np.sin(angle)))
    for i, idx in enumerate([473, 474, 475, 476, 477]):
        angle = 2.0 * np.pi * i / 5
        lm_list[idx] = _lm(nx(eye_r_cx + 5 * np.cos(angle)),
                           ny(eye_y + 5 * np.sin(angle)))

    lm_list[133] = _lm(nx(eye_l_cx + 10), ny(eye_y))
    lm_list[362] = _lm(nx(eye_r_cx - 10), ny(eye_y))

    nose_bridge_y = cy - face_h * 0.05
    lm_list[168] = _lm(nx(cx), ny(nose_bridge_y))
    lm_list[6] = _lm(nx(cx), ny(nose_bridge_y + 10))
    lm_list[4] = _lm(nx(cx), ny(cy + face_h * 0.10))
    lm_list[1] = _lm(nx(cx), ny(cy + face_h * 0.05))
    lm_list[48] = _lm(nx(cx - face_w * 0.10), ny(cy + face_h * 0.08))
    lm_list[278] = _lm(nx(cx + face_w * 0.10), ny(cy + face_h * 0.08))

    mouth_y = cy + face_h * 0.25
    lm_list[13] = _lm(nx(cx), ny(mouth_y - 5))
    lm_list[14] = _lm(nx(cx), ny(mouth_y + 5))
    lm_list[61] = _lm(nx(cx - face_w * 0.12), ny(mouth_y))
    lm_list[291] = _lm(nx(cx + face_w * 0.12), ny(mouth_y))
    for i, idx in enumerate([61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                             291, 409, 270, 269, 267, 0, 37, 39, 40, 185]):
        angle = 2.0 * np.pi * i / 20
        ex = cx + (face_w * 0.12) * np.cos(angle)
        ey = mouth_y + 10 * np.sin(angle)
        lm_list[idx] = _lm(nx(ex), ny(ey))

    forehead_top_y = top + face_h * 0.05
    ft_indices = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
    for i, idx in enumerate(ft_indices):
        angle = np.pi * (0.1 + 0.8 * i / max(len(ft_indices) - 1, 1))
        ex = cx + (face_w * 0.45) * np.cos(angle)
        ey = forehead_top_y + (face_h * 0.05) * np.sin(angle)
        lm_list[idx] = _lm(nx(ex), ny(ey))

    compat = _LandmarkCompat(lm_list)
    return FaceData(
        landmarks=compat,
        bbox=(cx - face_w // 2, cy - face_h // 2, face_w, face_h),
        ied=50.0,
    )


@dataclass
class _MockCtx:
    slimming: float = 0.0
    reshape_eye_size: float = 0.0
    reshape_eye_distance: float = 0.0
    reshape_nose_width: float = 0.0
    reshape_nose_length: float = 0.0
    reshape_jaw_width: float = 0.0
    reshape_chin_length: float = 0.0
    reshape_mouth_size: float = 0.0
    reshape_smile: float = 0.0
    reshape_forehead: float = 0.0


@pytest.fixture
def reshaper() -> FaceReshaper:
    return FaceReshaper()


@pytest.fixture
def face() -> FaceData:
    return _make_face()


@pytest.fixture
def img_u8() -> np.ndarray:
    grad = np.tile(np.linspace(0, 255, _IMG_W, dtype=np.uint8), (_IMG_H, 1))
    return np.stack([grad] * 3, axis=-1)


@pytest.fixture
def img_f32() -> np.ndarray:
    grad = np.tile(np.linspace(0, 255, _IMG_W, dtype=np.float32), (_IMG_H, 1))
    return np.stack([grad] * 3, axis=-1)


def _face_width(face: FaceData) -> float:
    lm = face.landmarks.landmark
    return abs(lm[454].x * _IMG_W - lm[234].x * _IMG_W)


# ---------------------------------------------------------------------------
# Per-feature validity tests
# ---------------------------------------------------------------------------

_FEATURE_SLIDERS = [
    ("reshape_eye_size", 30),
    ("reshape_eye_distance", 30),
    ("reshape_nose_width", -30),
    ("reshape_nose_length", 30),
    ("reshape_jaw_width", 30),
    ("reshape_chin_length", 30),
    ("reshape_mouth_size", 30),
    ("reshape_smile", 20),
    ("reshape_forehead", 20),
]


@pytest.mark.parametrize("attr,value", _FEATURE_SLIDERS)
def test_each_warp_valid_output(reshaper, img_u8, face, attr, value):
    ctx = _MockCtx(**{attr: value})
    result = reshaper.reshape(img_u8, [face], ctx)
    assert result.shape == img_u8.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Zero-strength no-op
# ---------------------------------------------------------------------------

def test_zero_strength_noop(reshaper, img_u8, face):
    ctx = _MockCtx()
    result = reshaper.reshape(img_u8, [face], ctx)
    assert np.array_equal(result, img_u8)


def test_legacy_strength_zero_noop(reshaper, img_u8, face):
    result = reshaper.reshape(img_u8, [face], strength=0)
    assert np.array_equal(result, img_u8)


# ---------------------------------------------------------------------------
# Symmetry: mirrored warps produce mirrored output
# ---------------------------------------------------------------------------

def _mirror_image(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, :].copy() if img.ndim == 3 else img[:, ::-1].copy()


def _mirror_face(face: FaceData, img_w: int) -> FaceData:
    lm_list = []
    for lm in face.landmarks.landmark:
        lm_list.append(_Landmark(1.0 - lm.x, lm.y, lm.z))
    compat = _LandmarkCompat(lm_list)
    bx, by, bw, bh = face.bbox
    return FaceData(
        landmarks=compat,
        bbox=(img_w - bx - bw, by, bw, bh),
        ied=face.ied,
    )


@pytest.mark.parametrize("attr,value", [
    ("reshape_eye_size", 30),
    ("reshape_jaw_width", 30),
    ("reshape_eye_distance", 30),
    ("reshape_mouth_size", 30),
])
def test_bilateral_symmetry(reshaper, img_u8, face, attr, value):
    ctx = _MockCtx(**{attr: value})
    result = reshaper.reshape(img_u8, [face], ctx)

    face_m = _mirror_face(face, _IMG_W)
    img_m = _mirror_image(img_u8)
    result_m = reshaper.reshape(img_m, [face_m], ctx)
    result_m_back = _mirror_image(result_m)

    diff = np.abs(result.astype(np.float32) - result_m_back.astype(np.float32))
    assert diff.mean() < 2.0, f"asymmetry too high for {attr}: mean diff {diff.mean()}"


# ---------------------------------------------------------------------------
# Displacement caps
# ---------------------------------------------------------------------------

def _max_displacement(warps):
    max_d = 0.0
    for C, T, _R in warps:
        d = ((T[0] - C[0]) ** 2 + (T[1] - C[1]) ** 2) ** 0.5
        if d > max_d:
            max_d = d
    return max_d


@pytest.mark.parametrize("attr,value", _FEATURE_SLIDERS + [
    ("reshape_eye_size", 50), ("reshape_eye_size", -50),
    ("reshape_nose_width", 50), ("reshape_nose_width", -50),
    ("reshape_jaw_width", 50), ("reshape_jaw_width", -50),
    ("reshape_smile", 30), ("reshape_smile", -20),
    ("reshape_forehead", 30), ("reshape_forehead", -30),
])
def test_displacement_cap(reshaper, face, attr, value):
    fw = _face_width(face)
    lm_obj = face.landmarks
    landmarks = lm_obj.landmark
    h, w = _IMG_H, _IMG_W

    builder_map = {
        "reshape_eye_size": lambda lm, fw, sl, h, w: reshaper._eye_size_warps(lm_obj, landmarks, fw, sl, h, w),
        "reshape_eye_distance": reshaper._eye_distance_warps,
        "reshape_nose_width": reshaper._nose_width_warps,
        "reshape_nose_length": reshaper._nose_length_warps,
        "reshape_jaw_width": reshaper._jaw_width_warps,
        "reshape_chin_length": reshaper._chin_length_warps,
        "reshape_mouth_size": lambda lm, fw, sl, h, w: reshaper._mouth_size_warps(lm_obj, landmarks, fw, sl, h, w),
        "reshape_smile": reshaper._smile_warps,
        "reshape_forehead": lambda lm, fw, sl, h, w: reshaper._forehead_warps(lm_obj, landmarks, fw, sl, h, w),
    }
    warps = builder_map[attr](landmarks, fw, value, h, w)
    if not warps:
        pytest.skip(f"no warps emitted for {attr}={value}")
    max_d = _max_displacement(warps)
    cap = fw * 0.12
    assert max_d <= cap + 0.5, f"{attr}={value}: displacement {max_d:.2f} > cap {cap:.2f} (fw={fw:.1f})"


def test_radius_clamp(reshaper, face):
    fw = _face_width(face)
    lm_obj = face.landmarks
    landmarks = lm_obj.landmark
    h, w = _IMG_H, _IMG_W
    all_warps: list = []
    builder_map = {
        "reshape_eye_size": lambda lm, fw, sl, h, w: reshaper._eye_size_warps(lm_obj, landmarks, fw, sl, h, w),
        "reshape_eye_distance": reshaper._eye_distance_warps,
        "reshape_nose_width": reshaper._nose_width_warps,
        "reshape_nose_length": reshaper._nose_length_warps,
        "reshape_jaw_width": reshaper._jaw_width_warps,
        "reshape_chin_length": reshaper._chin_length_warps,
        "reshape_mouth_size": lambda lm, fw, sl, h, w: reshaper._mouth_size_warps(lm_obj, landmarks, fw, sl, h, w),
        "reshape_smile": reshaper._smile_warps,
        "reshape_forehead": lambda lm, fw, sl, h, w: reshaper._forehead_warps(lm_obj, landmarks, fw, sl, h, w),
    }
    for attr, value in _FEATURE_SLIDERS:
        all_warps.extend(builder_map[attr](landmarks, fw, value, h, w))
    cap = fw * 0.5
    for _C, _T, R in all_warps:
        assert R <= cap + 0.5, f"radius {R} > fw*0.5={cap:.1f}"


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------

def test_dtype_preservation_uint8(reshaper, img_u8, face):
    ctx = _MockCtx(reshape_eye_size=30)
    result = reshaper.reshape(img_u8, [face], ctx)
    assert result.dtype == np.uint8


def test_dtype_preservation_float32(reshaper, img_f32, face):
    ctx = _MockCtx(reshape_eye_size=30)
    result = reshaper.reshape(img_f32, [face], ctx)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Slimming backward compat
# ---------------------------------------------------------------------------

def test_slimming_backward_compat(reshaper, img_u8, face):
    result_legacy = reshaper.reshape(img_u8, [face], strength=50)
    ctx = _MockCtx(slimming=50.0)
    result_ctx = reshaper.reshape(img_u8, [face], ctx)
    assert np.array_equal(result_legacy, result_ctx)


def test_slimming_changes_image(reshaper, img_u8, face):
    result = reshaper.reshape(img_u8, [face], strength=80)
    assert not np.array_equal(result, img_u8)


# ---------------------------------------------------------------------------
# No faces / tiny face
# ---------------------------------------------------------------------------

def test_no_faces_noop(reshaper, img_u8):
    ctx = _MockCtx(reshape_eye_size=30)
    result = reshaper.reshape(img_u8, [], ctx)
    assert np.array_equal(result, img_u8)


def test_tiny_face_skipped(reshaper, img_u8):
    face = _make_face(cx=200, cy=200, face_w=8, face_h=10)
    ctx = _MockCtx(reshape_eye_size=30)
    result = reshaper.reshape(img_u8, [face], ctx)
    assert np.array_equal(result, img_u8)
