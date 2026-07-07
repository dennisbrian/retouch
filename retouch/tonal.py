"""Film H&D (Hurter-Driffield) tonal response curve.

Implements a soft-toe, soft-shoulder sigmoid in place of the digital sCurve
that most camera pipelines ship. This is the single biggest "Fuji-look"
unlock: a smooth onset in shadows, gentle roll-off in highlights, and an
S-shape that varies in depth per simulation (Provia = mild, Astia = soft,
Classic Chrome = hard).

Reference: ``docs/FUJI_COLOR_RESEARCH.md`` section 3.2.

Public API:
    hd_curve_lut(strength, toe, shoulder)  -> 256-entry uint8 LUT
    apply_hd_curve(img_bgr, strength, ...)  -> tone-mapped BGR (dtype-preserving)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import apply_curve, bgr_f32_to_lab_f32, lab_f32_to_bgr_f32


def hd_curve_lut(
    strength: float = 0.7,
    toe: float = 0.10,
    shoulder: float = 0.10,
    midpoint: float = 0.50,
    gamma: float = 1.0,
) -> np.ndarray:
    """Build a 256-entry H&D-style tone curve as a uint8 LUT.

    The curve is a piecewise cubic spline: linear in the toe/shoulder regions
    (controlled by ``toe`` and ``shoulder``), with a smooth sigmoid between
    them, modulated by ``gamma``. ``strength`` blends the curve against the
    identity (strength=0 => identity, strength=1 => full curve).

    Continuity: the toe ends at value ``toe``, the mid starts at ``toe`` and
    ends at ``1 - shoulder``, the shoulder starts at ``1 - shoulder`` and ends
    at ``1.0``. This guarantees a C^0-continuous, monotonically increasing
    curve for any ``toe, shoulder in [0, 0.5]``.

    Args:
        strength: Blend amount in [0, 1]. 0 = no-op, 1 = full effect.
        toe: Fraction of input range that is a soft toe (shadow onset).
            0.0 = no toe (hard linear start), 0.10 = Fuji-Astia-like toe.
        shoulder: Fraction of input range that is a soft shoulder (highlight
            roll-off). 0.0 = hard clip, 0.10 = soft film-like roll-off.
        midpoint: Input point that maps to 0.5 output (default 0.5 = centered).
        gamma: Additional gamma applied between toe and shoulder.

    Returns:
        256-element uint8 LUT suitable for ``cv2.LUT``. Always uint8:
        ``cv2.LUT`` requires a uint8 LUT regardless of input dtype, so this
        builder is intentionally dtype-fixed.
    """
    if strength <= 0.0:
        return np.arange(256, dtype=np.uint8)
    strength = float(np.clip(strength, 0.0, 1.0))
    toe = float(np.clip(toe, 0.0, 0.5))
    shoulder = float(np.clip(shoulder, 0.0, 0.5))
    midpoint = float(np.clip(midpoint, 0.0, 1.0))
    gamma = max(gamma, 0.01)

    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    toe_end = toe
    shoulder_start = 1.0 - shoulder
    raw = np.empty_like(x)

    in_toe = x < toe_end
    if in_toe.any():
        raw[in_toe] = _toe_segment(x[in_toe], toe_end, toe)

    in_mid = (x >= toe_end) & (x <= shoulder_start)
    if in_mid.any():
        t = (x[in_mid] - toe_end) / max(shoulder_start - toe_end, 1e-6)
        sig = _sigmoid(t, midpoint=_safe_midpoint(midpoint, toe, shoulder))
        if gamma != 1.0:
            sig = np.power(np.clip(sig, 0.0, 1.0), 1.0 / gamma)
        raw[in_mid] = toe + sig * (1.0 - toe - shoulder)

    in_shoulder = x > shoulder_start
    if in_shoulder.any():
        raw[in_shoulder] = _shoulder_segment(x[in_shoulder], shoulder_start, shoulder)

    raw = np.clip(raw, 0.0, 1.0)
    if strength < 1.0:
        raw = raw * strength + x * (1.0 - strength)

    return np.clip(raw * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _toe_segment(x: np.ndarray, toe_end: float, toe_depth: float) -> np.ndarray:
    """Cosine-eased toe: y(0)=0, y(toe_end)=toe_depth, dy/dx >= 0 throughout."""
    if toe_end <= 1e-6 or toe_depth <= 1e-6:
        return x.copy()
    t = x / toe_end
    return (toe_depth * 0.5) * (1.0 - np.cos(np.pi * t))


def _shoulder_segment(x: np.ndarray, shoulder_start: float, shoulder_depth: float) -> np.ndarray:
    """Mirror of the toe: y(shoulder_start)=1-shoulder_depth, y(1)=1."""
    if shoulder_start >= 1.0 - 1e-6 or shoulder_depth <= 1e-6:
        return x.copy()
    t = (x - shoulder_start) / max(1.0 - shoulder_start, 1e-6)
    return 1.0 - shoulder_depth + (shoulder_depth * 0.5) * (1.0 - np.cos(np.pi * t))


def _sigmoid(t: np.ndarray, midpoint: float = 0.5) -> np.ndarray:
    """Smoothstep with adjustable midpoint."""
    m = float(np.clip(midpoint, 0.05, 0.95))
    if m == 0.5:
        return t * t * (3.0 - 2.0 * t)
    k = 1.0 / max(m * (1.0 - m), 1e-3)
    a = k * m
    b = k * m * m - 2.0 * k * m + k
    s = a * t + b
    return np.where(
        s <= 0,
        np.zeros_like(t),
        np.where(s >= 1, np.ones_like(t), s * s * (3.0 - 2.0 * s)),
    )


def _safe_midpoint(midpoint: float, toe: float, shoulder: float) -> float:
    lo = toe + 0.05
    hi = 1.0 - shoulder - 0.05
    if hi <= lo:
        return 0.5
    return float(np.clip(midpoint, lo, hi))


def apply_hd_curve(
    img_bgr: np.ndarray,
    strength: float = 0.7,
    toe: float = 0.10,
    shoulder: float = 0.10,
    midpoint: float = 0.50,
    gamma: float = 1.0,
    luma_only: bool = True,
) -> np.ndarray:
    """Apply an H&D-style film response curve to a BGR image.

    By default the curve is applied to the L* channel of LAB so colour
    shifts are minimised (luminance / chroma decoupled -- the Astia trick).
    Set ``luma_only=False`` to curve each BGR channel independently (a
    stronger, more vintage look).

    Dtype-aware: accepts either uint8 BGR (legacy path, byte-identical to
    the original implementation) or float32 BGR in [0, 255] (E1 float path).
    The returned image matches the input dtype. ``cv2.LUT`` inherently
    requires uint8 input, so on the float path the LUT is applied to a
    uint8 snapshot and its *delta* (lut output minus snapshot) is added back
    to the float canvas -- pixels the curve does not move keep full float
    precision, and quantization is confined to the pixels actually remapped.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image in [0, 255].
        strength: Blend amount in [0, 1]. 0 = identity, 1 = full curve.
        toe: Shadow toe depth in [0, 0.5].
        shoulder: Highlight shoulder depth in [0, 0.5].
        midpoint: Curve midpoint in [0, 1] (default 0.5).
        gamma: Mid-range gamma adjustment.
        luma_only: If True (default), curve L* channel of LAB only.

    Returns:
        (H, W, 3) BGR image with the H&D curve applied, same dtype as input.
    """
    if strength <= 0.0:
        return img_bgr
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"apply_hd_curve: expected HxWx3 BGR, got shape {img_bgr.shape}")

    is_float = img_bgr.dtype == np.float32
    lut = hd_curve_lut(strength=strength, toe=toe, shoulder=shoulder,
                       midpoint=midpoint, gamma=gamma)

    if not is_float:
        if luma_only:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        b, g, r = cv2.split(img_bgr)
        b = cv2.LUT(b, lut)
        g = cv2.LUT(g, lut)
        r = cv2.LUT(r, lut)
        return cv2.merge([b, g, r])

    if luma_only:
        lab_f32 = bgr_f32_to_lab_f32(img_bgr)
        l_f32 = lab_f32[:, :, 0]
        l_u8 = np.clip(l_f32, 0, 255).astype(np.uint8)
        l_mapped_u8 = cv2.LUT(l_u8, lut)
        delta = l_mapped_u8.astype(np.float32) - l_u8.astype(np.float32)
        lab_f32[:, :, 0] = np.clip(l_f32 + delta, 0.0, 255.0)
        return lab_f32_to_bgr_f32(lab_f32)

    out = img_bgr.copy()
    for c in range(3):
        ch_f32 = out[:, :, c]
        ch_u8 = np.clip(ch_f32, 0, 255).astype(np.uint8)
        ch_mapped_u8 = cv2.LUT(ch_u8, lut)
        delta = ch_mapped_u8.astype(np.float32) - ch_u8.astype(np.float32)
        out[:, :, c] = np.clip(ch_f32 + delta, 0.0, 255.0)
    return out


def apply_lift_gamma_gain(
    img_bgr: np.ndarray,
    lift: float = 0.0,
    gamma: float = 1.0,
    gain: float = 1.0,
    luma_only: bool = True,
) -> np.ndarray:
    """Classic lift / gamma / gain on the L* channel.

    The lift raises the toe (lifted blacks in Classic Chrome), gamma shapes
    the midtones, gain stretches the highlights. All in [0, 1] in L* space.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        lift: Additive lift on the L* channel in [0, 1] (approx 0-255 units).
        gamma: Multiplicative gamma in L* space.
        gain: Multiplicative gain on the L* channel.
        luma_only: If True (default), operate on the L* channel only.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    if lift == 0.0 and gamma == 1.0 and gain == 1.0:
        return img_bgr
    if luma_only:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0] / 255.0
        l = np.clip(l * gain + lift, 0.0, 1.0)
        if gamma != 1.0:
            l = np.power(l, 1.0 / gamma)
        lab[:, :, 0] = np.clip(l * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img_f = img_bgr.astype(np.float32) / 255.0
    img_f = np.clip(img_f * gain + lift, 0.0, 1.0)
    if gamma != 1.0:
        img_f = np.power(img_f, 1.0 / gamma)
    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
