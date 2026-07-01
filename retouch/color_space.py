"""Color space utilities: LCH (perceptually uniform edits) and wide-gamut
RGB (ProPhoto, Adobe RGB) conversions for Fuji X-Trans and other wide-gamut
camera sources.

Public API:
    Conversion:  bgr_to_lch / lch_to_bgr / lab_to_lch / lch_to_lab
    Adjustment:  adjust_luminance / adjust_chroma / adjust_hue
    Range edits: hue_range_mask / adjust_hue_range / adjust_chroma_range / adjust_luminance_range
    Tone-based:  split_tone_lch / color_balance_lch
    Skin:        skin_mask_lch
    Wide-gamut:  bgr_to_prophoto / prophoto_to_bgr / bgr_to_adobe_rgb / adobe_rgb_to_bgr
    Utility:     estimate_gamut / srgb_to_xyz_matrix / xyz_to_prophoto_matrix

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


# ---------------------------------------------------------------------------
# Perceptual HSL range-editing primitives (LCH-based)
#
# These are the building blocks for a Lightroom-style HSL panel that
# operates in perceptually-uniform LCH space instead of device-centric
# HSV/HSL.  All functions accept and return float32 LCH arrays.
# ---------------------------------------------------------------------------


def hue_range_mask(
    lch: np.ndarray,
    hue_center: float = 0.0,
    hue_width: float = 30.0,
    falloff: float = 15.0,
) -> np.ndarray:
    """Soft symmetric mask around a hue center in [0, 360).

    The mask is cos²-falloff for natural blending — sharp at the edges of
    everyday colour sliders, but never creating a visible seam.

    Args:
        lch: (H, W, 3) float32 LCH.
        hue_center: Hue centre in degrees. 0=red, 120=green, 240=blue.
        hue_width: Half-width of the hue range in degrees. Default 30
            covers ±30° around hue_center.
        falloff: Additional falloff width in degrees (cos² ramp). Default
            15° gives a smooth blend at the range edges.

    Returns:
        (H, W) float32 mask in [0, 1].
    """
    d = np.abs(lch[:, :, 2] - hue_center)
    d = np.minimum(d, 360.0 - d)
    inner = hue_width
    outer = hue_width + falloff
    if outer <= inner:
        return (d <= inner).astype(np.float32)
    t = np.clip((d - inner) / (outer - inner), 0.0, 1.0)
    return (np.cos(t * np.pi * 0.5) ** 2).astype(np.float32)


def adjust_hue_range(
    lch: np.ndarray,
    hue_center: float,
    hue_width: float = 30.0,
    delta_deg: float = 0.0,
    falloff: float = 15.0,
) -> np.ndarray:
    """Shift hue *within* a hue range using a soft mask.

    This is the Lightroom "Hue" slider for a colour band: reds shift
    toward orange or magenta, blues toward cyan or purple, etc.

    Args:
        lch: (H, W, 3) float32 LCH.
        hue_center: Hue range centre in degrees.
        hue_width: Half-width in degrees.
        delta_deg: Hue shift in degrees within the range.
        falloff: Falloff width in degrees.

    Returns:
        New (H, W, 3) float32 LCH.
    """
    mask = hue_range_mask(lch, hue_center, hue_width, falloff)
    out = lch.copy()
    shift = delta_deg * mask
    out[:, :, 2] = np.mod(out[:, :, 2] + shift, 360.0)
    return out


def adjust_chroma_range(
    lch: np.ndarray,
    hue_center: float,
    hue_width: float = 30.0,
    factor: float = 1.0,
    falloff: float = 15.0,
) -> np.ndarray:
    """Scale chroma (perceptual saturation) *within* a hue range.

    This is the Lightroom "Saturation" slider for a colour band.
    ``factor=1.0`` is no-op, ``2.0`` doubles saturation, ``0.5`` halves it.

    Args:
        lch: (H, W, 3) float32 LCH.
        hue_center: Hue range centre in degrees.
        hue_width: Half-width in degrees.
        factor: Chroma scale factor (1.0 = no change).
        falloff: Falloff width in degrees.

    Returns:
        New (H, W, 3) float32 LCH.
    """
    mask = hue_range_mask(lch, hue_center, hue_width, falloff)
    out = lch.copy()
    blend = 1.0 + (factor - 1.0) * mask
    out[:, :, 1] = out[:, :, 1] * blend
    return out


def adjust_luminance_range(
    lch: np.ndarray,
    hue_center: float,
    hue_width: float = 30.0,
    delta: float = 0.0,
    falloff: float = 15.0,
) -> np.ndarray:
    """Shift lightness *within* a hue range.

    This is the Lightroom "Luminance" slider for a colour band.

    Args:
        lch: (H, W, 3) float32 LCH.
        hue_center: Hue range centre in degrees.
        hue_width: Half-width in degrees.
        delta: L* shift in [-100, 100].
        falloff: Falloff width in degrees.

    Returns:
        New (H, W, 3) float32 LCH.
    """
    mask = hue_range_mask(lch, hue_center, hue_width, falloff)
    out = lch.copy()
    out[:, :, 0] = np.clip(out[:, :, 0] + delta * mask, 0.0, 100.0)
    return out


def split_tone_lch(
    lch: np.ndarray,
    shadow_hue: float = 0.0,
    shadow_sat: float = 0.0,
    highlight_hue: float = 0.0,
    highlight_sat: float = 0.0,
    balance: float = 0.0,
) -> np.ndarray:
    """Perceptual split toning in LCH space.

    Uses the L* channel as the blend key (low L* → shadows, high L* →
    highlights) rather than an HSV value channel.  This gives a
    perceptually uniform transition: the split point at L*=50 separates
    regions the eye perceives as dark vs light.

    ``balance`` shifts the crossover point: negative favours shadows,
    positive favours highlights.  Range: [-100, 100].

    Args:
        lch: (H, W, 3) float32 LCH.
        shadow_hue: Hue to push shadow tones toward (degrees, 0–360).
        shadow_sat: Saturation strength for shadows (0=no tint, 1=full).
        highlight_hue: Hue for highlight tones.
        highlight_sat: Saturation strength for highlights.
        balance: Crossover shift in [-100, 100].

    Returns:
        New (H, W, 3) float32 LCH.
    """
    if shadow_sat <= 1e-6 and highlight_sat <= 1e-6:
        return lch.copy()

    l_norm = lch[:, :, 0] / 100.0
    midpoint = 0.5 + balance / 200.0
    midpoint = np.clip(midpoint, 0.05, 0.95)

    shadow_weight = np.clip((midpoint - l_norm) / max(midpoint, 0.05), 0.0, 1.0)
    highlight_weight = np.clip((l_norm - midpoint) / max(1.0 - midpoint, 0.05), 0.0, 1.0)

    out = lch.copy()
    if shadow_sat > 1e-6:
        shadow_h = np.full_like(out[:, :, 2], shadow_hue)
        blend_s = shadow_weight[:, :, np.newaxis] * shadow_sat
        out[:, :, 1] = out[:, :, 1] * (1.0 - blend_s[:, :, 0]) + shadow_sat * 60.0 * blend_s[:, :, 0]
        out[:, :, 2] = out[:, :, 2] * (1.0 - blend_s[:, :, 0]) + shadow_h * blend_s[:, :, 0]

    if highlight_sat > 1e-6:
        highlight_h = np.full_like(out[:, :, 2], highlight_hue)
        blend_h = highlight_weight[:, :, np.newaxis] * highlight_sat
        out[:, :, 1] = out[:, :, 1] * (1.0 - blend_h[:, :, 0]) + highlight_sat * 60.0 * blend_h[:, :, 0]
        out[:, :, 2] = out[:, :, 2] * (1.0 - blend_h[:, :, 0]) + highlight_h * blend_h[:, :, 0]

    return out


def color_balance_lch(
    lch: np.ndarray,
    cyan_red: float = 0.0,
    magenta_green: float = 0.0,
    yellow_blue: float = 0.0,
    preserve_luminosity: bool = True,
) -> np.ndarray:
    """Classic 3-way colour balance in LCH space.

    Maps the Photoshop / DaVinci colour-balance axes onto LCH primitives:
        - cyan_red:   shifts hue toward cyan (−) or red (+)  (hue ~180° vs ~0°)
        - magenta_green: shifts toward magenta (−) or green (+)  (hue ~300° vs ~120°)
        - yellow_blue:   shifts toward yellow (−) or blue (+)  (hue ~60° vs ~240°)

    Each axis range is [-1, 1] mapped to a nominal 30° hue-shift.

    Args:
        lch: (H, W, 3) float32 LCH.
        cyan_red: Cyan-red balance in [-1, 1].
        magenta_green: Magenta-green balance in [-1, 1].
        yellow_blue: Yellow-blue balance in [-1, 1].
        preserve_luminosity: If True (default), leave L* unchanged.

    Returns:
        New (H, W, 3) float32 LCH.
    """
    if abs(cyan_red) < 1e-6 and abs(magenta_green) < 1e-6 and abs(yellow_blue) < 1e-6:
        return lch.copy()

    MAX_SHIFT = 30.0
    c_red = np.clip(cyan_red, -1.0, 1.0) * MAX_SHIFT
    m_green = np.clip(magenta_green, -1.0, 1.0) * MAX_SHIFT
    y_blue = np.clip(yellow_blue, -1.0, 1.0) * MAX_SHIFT

    out = lch.copy()

    h = out[:, :, 2]
    c = out[:, :, 1]

    # Reduce shift for near-zero chroma (grey pixels stay grey)
    chroma_weight = np.clip(c / 10.0, 0.0, 1.0)

    # Red axis: push hue toward 0° (positive) or 180° (negative)
    red_weight = np.where(c_red >= 0, 1.0 - np.abs(h - 0.0) / 180.0, 0.0)
    cyan_weight = np.where(c_red < 0, 1.0 - np.abs(h - 180.0) / 180.0, 0.0)
    h += (c_red * red_weight - c_red * cyan_weight) * chroma_weight

    # Green axis: push hue toward 120° (positive) or 300°=magenta (negative)
    green_weight = np.where(m_green >= 0, 1.0 - np.abs(h - 120.0) / 180.0, 0.0)
    magenta_weight = np.where(m_green < 0, 1.0 - np.abs(h - 300.0) / 180.0, 0.0)
    h += (m_green * green_weight - m_green * magenta_weight) * chroma_weight

    # Blue axis: push hue toward 240° (positive) or 60°=yellow (negative)
    blue_weight = np.where(y_blue >= 0, 1.0 - np.abs(h - 240.0) / 180.0, 0.0)
    yellow_weight = np.where(y_blue < 0, 1.0 - np.abs(h - 60.0) / 180.0, 0.0)
    h += (y_blue * blue_weight - y_blue * yellow_weight) * chroma_weight

    out[:, :, 2] = np.mod(h, 360.0)

    if not preserve_luminosity:
        # Lightness shift follows the balance direction (named after print convention)
        l_shift = (cyan_red + magenta_green + yellow_blue) / 3.0 * 5.0
        out[:, :, 0] = np.clip(out[:, :, 0] + l_shift * chroma_weight, 0.0, 100.0)

    return out


def negative_split_tone_lch(
    lch: np.ndarray,
    shadow_desat: float = 0.0,
    highlight_desat: float = 0.0,
    shadow_threshold: float = 50.0,
    highlight_threshold: float = 60.0,
) -> np.ndarray:
    """Desaturate shadows and/or highlights independently.

    Unlike regular split toning (which *adds* a colour cast), this
    *removes* chroma from the targeted luminance zones — producing
    the "faded film", "bleach bypass", or "vintage" look.

    Shadows: pixels with L* ≤ ``shadow_threshold`` are desaturated by
    ``shadow_desat`` (0 = none, 1 = full grayscale in shadows).
    Highlights: pixels with L* ≥ ``highlight_threshold`` are desaturated
    by ``highlight_desat``.

    Both parameters use a smooth sigmoid transition for natural blending.

    Args:
        lch: (H, W, 3) float32 LCH.
        shadow_desat: Desaturation strength in shadows (0–1).
        highlight_desat: Desaturation strength in highlights (0–1).
        shadow_threshold: L* value below which shadows begin to desaturate.
        highlight_threshold: L* value above which highlights begin.

    Returns:
        New (H, W, 3) float32 LCH.
    """
    if shadow_desat <= 0.0 and highlight_desat <= 0.0:
        return lch.copy()

    out = lch.copy()
    l_star = out[:, :, 0]
    c = out[:, :, 1]

    if shadow_desat > 0.0:
        sd = float(np.clip(shadow_desat, 0.0, 1.0))
        st = float(np.clip(shadow_threshold, 5.0, 95.0))
        shadow_weight = np.clip(1.0 - l_star / st, 0.0, 1.0) ** 2
        ratio = 1.0 - shadow_weight * sd
        out[:, :, 1] = c * ratio

    if highlight_desat > 0.0:
        hd = float(np.clip(highlight_desat, 0.0, 1.0))
        ht = float(np.clip(highlight_threshold, 5.0, 95.0))
        highlight_weight = np.clip((l_star - ht) / (100.0 - ht), 0.0, 1.0) ** 2
        ratio = 1.0 - highlight_weight * hd
        out[:, :, 1] = out[:, :, 1] * ratio

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
