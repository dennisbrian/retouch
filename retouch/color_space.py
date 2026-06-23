"""Color space utilities: LCH (perceptually uniform edits) and wide-gamut
RGB (ProPhoto, Adobe RGB) conversions for Fuji X-Trans and other wide-gamut
camera sources.

All matrix-based conversions use the simple linear-matrix path
(BGR -> sRGB float -> XYZ D65 -> wide-gamut RGB) without gamma
linearization. This is fast and round-trip stable, but the resulting
wide-gamut values may exceed [0, 1] for sRGB inputs because the white
point shift between sRGB (D65) and ProPhoto (D50) is not compensated.
"""

from __future__ import annotations

import cv2
import numpy as np


def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert a uint8 BGR image to float32 LAB.

    Args:
        img: (H, W, 3) uint8 BGR image.

    Returns:
        (H, W, 3) float32 LAB with L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
    """
    lab_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_u8[:, :, 0] = lab_u8[:, :, 0] * (100.0 / 255.0)
    lab_u8[:, :, 1] -= 128.0
    lab_u8[:, :, 2] -= 128.0
    return lab_u8


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """Convert float32 LAB to uint8 BGR.

    Args:
        lab: (H, W, 3) float32 LAB with L* in [0, 100], a* in [-128, 127], b* in [-128, 127].

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    l = lab[:, :, 0] * (255.0 / 100.0)
    a = lab[:, :, 1] + 128.0
    b = lab[:, :, 2] + 128.0
    lab_u8 = np.clip(np.stack([l, a, b], axis=-1), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Convert LAB to LCH (cylindrical LAB).

    L is preserved, C = sqrt(a^2 + b^2), H = atan2(b, a) in [0, 360).

    Args:
        lab: (H, W, 3) float32 LAB.

    Returns:
        (H, W, 3) float32 LCH: L* in [0, 100], C* in [0, ~180], H° in [0, 360).
    """
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    c = np.sqrt(a * a + b * b)
    h = np.mod(np.degrees(np.arctan2(b, a)) + 360.0, 360.0)
    return np.stack([l, c, h], axis=-1).astype(np.float32)


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    """Convert LCH to LAB.

    a = C * cos(H), b = C * sin(H) (with H in radians).

    Args:
        lch: (H, W, 3) float32 LCH.

    Returns:
        (H, W, 3) float32 LAB.
    """
    l = lch[:, :, 0]
    c = lch[:, :, 1]
    h_rad = np.radians(lch[:, :, 2])
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    return np.stack([l, a, b], axis=-1).astype(np.float32)


def bgr_to_lch(img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR to float32 LCH (one-shot convenience).

    Args:
        img: (H, W, 3) uint8 BGR image.

    Returns:
        (H, W, 3) float32 LCH.
    """
    return lab_to_lch(bgr_to_lab(img))


def lch_to_bgr(lch: np.ndarray) -> np.ndarray:
    """Convert float32 LCH to uint8 BGR (one-shot convenience).

    Args:
        lch: (H, W, 3) float32 LCH.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    return lab_to_bgr(lch_to_lab(lch))


def adjust_luminance(lch: np.ndarray, delta: float) -> np.ndarray:
    """Shift the L* channel by ``delta`` and clamp to [0, 100].

    Args:
        lch: (H, W, 3) float32 LCH.
        delta: L* shift, typically in [-50, 50]. Positive brightens.

    Returns:
        New (H, W, 3) float32 LCH with adjusted L*.
    """
    out = lch.copy()
    out[:, :, 0] = np.clip(out[:, :, 0] + delta, 0.0, 100.0)
    return out


def adjust_chroma(lch: np.ndarray, factor: float) -> np.ndarray:
    """Multiply the C* channel by ``factor``.

    Args:
        lch: (H, W, 3) float32 LCH.
        factor: Multiplicative chroma scale (e.g. 1.2 = +20% saturation).

    Returns:
        New (H, W, 3) float32 LCH with adjusted C*.
    """
    out = lch.copy()
    out[:, :, 1] = out[:, :, 1] * factor
    return out


def adjust_hue(lch: np.ndarray, delta_deg: float) -> np.ndarray:
    """Shift the H° channel by ``delta_deg`` (mod 360).

    Args:
        lch: (H, W, 3) float32 LCH.
        delta_deg: Hue rotation in degrees.

    Returns:
        New (H, W, 3) float32 LCH with adjusted H°.
    """
    out = lch.copy()
    out[:, :, 2] = np.mod(out[:, :, 2] + delta_deg, 360.0)
    return out


def skin_mask_lch(
    lch: np.ndarray,
    hue_center: float = 25.0,
    hue_tolerance: float = 25.0,
    chroma_min: float = 8.0,
) -> np.ndarray:
    """Detect skin tones in LCH space.

    Output is ``exp(-((H - hue_center) / hue_tolerance) ** 2) * (C > chroma_min)``.

    Args:
        lch: (H, W, 3) float32 LCH.
        hue_center: Hue center in degrees. Default 25° covers red-orange-yellow.
        hue_tolerance: Hue falloff width in degrees.
        chroma_min: Minimum chroma required to be classified as skin.

    Returns:
        (H, W) float32 mask in [0, 1].
    """
    h = lch[:, :, 2]
    c = lch[:, :, 1]
    hue_term = np.exp(-((h - hue_center) / hue_tolerance) ** 2)
    chroma_gate = (c > chroma_min).astype(np.float32)
    return (hue_term * chroma_gate).astype(np.float32)


_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

_XYZ_TO_PROPHOTO = np.array(
    [
        [1.3459, -0.2556, -0.0511],
        [-0.5446, 1.5082, 0.0205],
        [0.0000, 0.0000, 1.2118],
    ],
    dtype=np.float32,
)

_XYZ_TO_ADOBE = np.array(
    [
        [2.0414, -0.5649, -0.3447],
        [-0.9692, 1.8760, 0.0416],
        [0.0134, -0.1184, 1.0154],
    ],
    dtype=np.float32,
)

_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ).astype(np.float32)
_PROPHOTO_TO_XYZ = np.linalg.inv(_XYZ_TO_PROPHOTO).astype(np.float32)
_ADOBE_TO_XYZ = np.linalg.inv(_XYZ_TO_ADOBE).astype(np.float32)

_SRGB_TO_PROPHOTO = (_XYZ_TO_PROPHOTO @ _SRGB_TO_XYZ).astype(np.float32)
_PROPHOTO_TO_SRGB = np.linalg.inv(_SRGB_TO_PROPHOTO).astype(np.float32)
_SRGB_TO_ADOBE = (_XYZ_TO_ADOBE @ _SRGB_TO_XYZ).astype(np.float32)
_ADOBE_TO_SRGB = np.linalg.inv(_SRGB_TO_ADOBE).astype(np.float32)


def srgb_to_xyz_matrix() -> np.ndarray:
    """Return the 3x3 sRGB -> XYZ (D65) matrix as float32.

    Returns:
        (3, 3) float32 ndarray. Standard Bruce Lindbloom sRGB D65 matrix,
        assumes linear sRGB input.
    """
    return _SRGB_TO_XYZ.copy()


def xyz_to_prophoto_matrix() -> np.ndarray:
    """Return the 3x3 XYZ -> ProPhoto RGB matrix as float32.

    Returns:
        (3, 3) float32 ndarray. ProPhoto RGB uses a D50 white point.
    """
    return _XYZ_TO_PROPHOTO.copy()


def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) * (1.0 / 255.0)
    if img.dtype == np.uint16:
        return img.astype(np.float32) * (1.0 / 65535.0)
    return img.astype(np.float32)


def _swap_rb(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]


def bgr_to_prophoto(img: np.ndarray) -> np.ndarray:
    """Convert BGR to ProPhoto RGB float32 via the XYZ (D65) matrix path.

    Args:
        img: (H, W, 3) BGR. Accepted dtypes: uint8, uint16, float32.
            Integer dtypes are normalized to [0, 1]; floats are used as-is.

    Returns:
        (H, W, 3) float32 ProPhoto RGB. Values may exceed [0, 1] for
        sRGB sources because the D65->D50 white-point shift is not
        compensated in the simple matrix path.
    """
    rgb = _swap_rb(_to_float01(img))
    h, w = rgb.shape[:2]
    out = (rgb.reshape(-1, 3) @ _SRGB_TO_PROPHOTO.T).reshape(h, w, 3)
    return out.astype(np.float32)


def prophoto_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert ProPhoto RGB float32 to uint8 BGR.

    Args:
        img: (H, W, 3) float32 ProPhoto RGB.

    Returns:
        (H, W, 3) uint8 BGR image. Values are clipped to [0, 255].
    """
    h, w = img.shape[:2]
    rgb = (img.reshape(-1, 3) @ _PROPHOTO_TO_SRGB.T).reshape(h, w, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return _swap_rb(rgb_u8)


def bgr_to_adobe_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to Adobe RGB float32 via the XYZ (D65) matrix path.

    Args:
        img: (H, W, 3) BGR. Accepted dtypes: uint8, uint16, float32.

    Returns:
        (H, W, 3) float32 Adobe RGB. Values may exceed [0, 1] for
        sources whose chromaticities are outside the sRGB gamut.
    """
    rgb = _swap_rb(_to_float01(img))
    h, w = rgb.shape[:2]
    out = (rgb.reshape(-1, 3) @ _SRGB_TO_ADOBE.T).reshape(h, w, 3)
    return out.astype(np.float32)


def adobe_rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert Adobe RGB float32 to uint8 BGR.

    Args:
        img: (H, W, 3) float32 Adobe RGB.

    Returns:
        (H, W, 3) uint8 BGR image. Values are clipped to [0, 255].
    """
    h, w = img.shape[:2]
    rgb = (img.reshape(-1, 3) @ _ADOBE_TO_SRGB.T).reshape(h, w, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return _swap_rb(rgb_u8)


def estimate_gamut(img: np.ndarray) -> str:
    """Heuristically estimate the source color gamut of an image.

    Heuristic:
        - 16-bit images are treated as ProPhoto (common for wide-gamut RAW).
        - float images with any channel value > 1.0 are treated as ProPhoto.
        - all other cases default to sRGB.

    Args:
        img: (H, W, 3) numpy array of any dtype.

    Returns:
        One of "srgb", "adobe_rgb", "prophoto". The "adobe_rgb" branch is
        reserved and is not currently returned by the simple heuristic.
    """
    if img.dtype == np.uint16:
        return "prophoto"

    fimg = _to_float01(img)
    if float(fimg.max()) > 1.0:
        return "prophoto"

    return "srgb"
