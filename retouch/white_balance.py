"""Linear-light CAT16 white-balance correction.

The public grading control expresses the estimated scene illuminant as a
Planckian colour temperature and a green/magenta correction.  This module
adapts that source white to D65 with a CAT16 von-Kries transform.  Unlike a
Lab/LCh hue rotation, neutral pixels participate in the correction.

Images use the project's established BGR contract: uint8 ``[0, 255]`` or
float32 ``[0, 255]``.  Float inputs remain float32 and never cross a uint8
boundary.
"""

from __future__ import annotations

import numpy as np


_D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

# CAT16 RGB cone-response matrix (XYZ D65, row-vector images use ``.T``).
_CAT16 = np.array(
    [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ],
    dtype=np.float64,
)
_CAT16_INV = np.linalg.inv(_CAT16)

# Linear sRGB <-> CIE XYZ (D65).
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

_MIN_KELVIN = 2000.0
_MAX_KELVIN = 50000.0
_MAX_TINT = 100.0
_TINT_UV_PER_UNIT = 0.00035


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert display-encoded sRGB ``[0, 1]`` to linear light."""
    values = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light RGB ``[0, 1]`` to display-encoded sRGB."""
    values = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def kelvin_to_xy(kelvin: float) -> tuple[float, float]:
    """Return a Planckian/daylight-locus chromaticity for ``kelvin``.

    The low-temperature branch is the standard Planckian closed-form fit.
    Above 4000K the CIE daylight locus is preferable for photographic white
    balance and remains stable through the historical 50000K API limit.
    """
    temperature = float(np.clip(kelvin, _MIN_KELVIN, _MAX_KELVIN))
    inv_t = 1000.0 / temperature
    if temperature <= 4000.0:
        x = (
            -0.2661239 * inv_t**3
            - 0.2343580 * inv_t**2
            + 0.8776956 * inv_t
            + 0.179910
        )
        if temperature <= 2222.0:
            y = (
                -1.1063814 * x**3
                - 1.34811020 * x**2
                + 2.18555832 * x
                - 0.20219683
            )
        else:
            y = (
                -0.9549476 * x**3
                - 1.37418593 * x**2
                + 2.09137015 * x
                - 0.16748867
            )
    elif temperature <= 7000.0:
        x = (
            0.244063
            + 0.09911 * inv_t
            + 2.9678 * inv_t**2
            - 4.6070 * inv_t**3
        )
        y = -3.0 * x**2 + 2.870 * x - 0.275
    else:
        x = (
            0.237040
            + 0.24748 * inv_t
            + 1.9018 * inv_t**2
            - 2.0064 * inv_t**3
        )
        y = -3.0 * x**2 + 2.870 * x - 0.275
    return float(x), float(y)


def _xy_to_uv1960(x: float, y: float) -> np.ndarray:
    denominator = -2.0 * x + 12.0 * y + 3.0
    return np.array([4.0 * x / denominator, 6.0 * y / denominator], dtype=np.float64)


def _uv1960_to_xy(uv: np.ndarray) -> tuple[float, float]:
    u, v = float(uv[0]), float(uv[1])
    denominator = 2.0 * u - 8.0 * v + 4.0
    return 3.0 * u / denominator, 2.0 * v / denominator


def _green_normal_uv(kelvin: float) -> np.ndarray:
    """Return the local Planckian normal oriented toward green in uv1960."""
    temperature = float(np.clip(kelvin, _MIN_KELVIN, _MAX_KELVIN))
    delta = min(100.0, max(10.0, temperature * 0.01))
    low = _xy_to_uv1960(*kelvin_to_xy(temperature - delta))
    high = _xy_to_uv1960(*kelvin_to_xy(temperature + delta))
    tangent = high - low
    length = float(np.linalg.norm(tangent))
    if length <= 1e-12:
        return np.array([-0.3, 0.95], dtype=np.float64)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64) / length
    # Green is the high-v side of the Planckian locus around photographic CCTs.
    if normal[1] < 0.0:
        normal = -normal
    return normal


def source_white_xyz(kelvin: float, tint: float = 0.0) -> np.ndarray:
    """Return the estimated source-white XYZ (Y = 1).

    ``tint`` follows the existing UI convention: positive requests a magenta
    correction and negative requests a green correction.  Since CAT operates
    source-to-D65, the source white is offset in the opposite (green for
    positive tint) direction before adaptation.
    """
    temperature = float(np.clip(kelvin, _MIN_KELVIN, _MAX_KELVIN))
    tint_value = float(np.clip(tint, -_MAX_TINT, _MAX_TINT))
    uv = _xy_to_uv1960(*kelvin_to_xy(temperature))
    # A magenta output correction compensates a green source illuminant.
    uv += _green_normal_uv(temperature) * tint_value * _TINT_UV_PER_UNIT
    x, y = _uv1960_to_xy(uv)
    y = max(y, 1e-6)
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def cat16_matrix(source_white: np.ndarray, target_white: np.ndarray = _D65_XYZ) -> np.ndarray:
    """Build a CAT16 von-Kries source-to-target XYZ adaptation matrix."""
    source_lms = _CAT16 @ np.asarray(source_white, dtype=np.float64)
    target_lms = _CAT16 @ np.asarray(target_white, dtype=np.float64)
    gains = target_lms / np.maximum(source_lms, 1e-9)
    return _CAT16_INV @ np.diag(gains) @ _CAT16


def white_balance_cat16(
    img_bgr: np.ndarray,
    temperature: float = 6500.0,
    tint: float = 0.0,
) -> np.ndarray:
    """Adapt a BGR image from its Kelvin/tint source white to D65.

    Inputs and outputs keep the legacy grading contract: uint8 or float32 BGR
    in ``[0, 255]``.  The exact neutral control returns the input unchanged,
    preserving default-render byte identity.
    """
    if abs(float(temperature) - 6500.0) < 1e-3 and abs(float(tint)) < 1e-4:
        return img_bgr
    if img_bgr.ndim != 3 or img_bgr.shape[-1] != 3:
        raise ValueError("white_balance_cat16 expects an HxWx3 BGR image")

    is_uint8 = img_bgr.dtype == np.uint8
    bgr = np.clip(img_bgr.astype(np.float32, copy=False), 0.0, 255.0)
    rgb_linear = srgb_to_linear(bgr[..., ::-1] / 255.0)
    xyz = rgb_linear @ _RGB_TO_XYZ.T
    adaptation = cat16_matrix(source_white_xyz(temperature, tint))
    adapted_xyz = xyz @ adaptation.T
    adapted_rgb = adapted_xyz @ _XYZ_TO_RGB.T
    out_bgr = linear_to_srgb(adapted_rgb)[..., ::-1] * 255.0
    out_bgr = np.clip(out_bgr, 0.0, 255.0)
    if is_uint8:
        return np.round(out_bgr).astype(np.uint8)
    return out_bgr.astype(np.float32)
