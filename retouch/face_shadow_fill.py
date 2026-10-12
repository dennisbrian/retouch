"""Face Shadow Fill (opt-in ``face_shadow_fill``): a soft fill light for the face.

Wig fringes, hat brims, ears, headpieces and deep eye sockets throw broad
shadows across the face. The existing ``shadow_lift`` op cannot fill them
evenly: it lifts each pixel by how far it sits below a blur of the whole
crop (hair and background included), in fixed Lab units (a 40 L dip gets
the full 25 L), so it brightens pores, lash lines and nostrils as much as a
cast shadow, flattens the nose and cheek shading it was never meant to
touch, and on darker skin a real shadow is fewer Lab units deep, so it gets
less lift. It is left as is (6 recipes and Face Polish use it).

This op works like a reflector held under the face:

1. **Illumination, not detail.** Linear luminance is taken to log2 (stops),
   and the face's broad lighting is a masked Gaussian of that over clean
   skin only (sigma 0.08 IED). Hair, eyes, brows and lips are outside the
   skin support; small dark features (nostrils, lash line, mouth corners:
   more than 0.8 stop below their surroundings) and makeup (chromaticity far
   from the face's own skin, in units of the skin's own spread) are dropped
   from the estimate in a second pass, so pores and pigment are never read as
   shadow.
2. **Lift only the deep part.** Each pixel's shadow depth is how many stops
   its illumination sits below the face's lit level (the 75th percentile of
   that illumination over clean skin). The first ``_KNEE_STOPS`` (0.35 stop:
   the normal modelling of cheeks, nose sides and jaw) is kept; only depth
   beyond it is reduced, by up to ``_FILL`` at 100, capped at
   ``_MAX_LIFT_STOPS``. The face keeps its shape; harsh shadows open up.
3. **Light, not paint.** The lift is one linear-light gain per pixel on all
   three channels, from the smooth illumination map, so skin texture, makeup
   colour, hue and saturation stay those of the photo, and nothing outside
   the skin mask (minus eyes, brows, lips and dark features) changes.

Everything is a ratio (stops) measured on the face itself, so a face shot
half as bright, or darker skin under the same light, gets the same lift in
stops. There is no absolute intensity threshold.
"""

from __future__ import annotations

import cv2
import numpy as np

from .utils import normalize_mask

# Illumination scale: Gaussian sigma as a fraction of the inter-eye distance.
_SIGMA_IED_FRAC = 0.08
# Depth below the lit level (stops) that is normal facial modelling, kept as is.
_KNEE_STOPS = 0.35
# Share of the depth beyond the knee removed at strength 100.
_FILL = 0.70
_MAX_LIFT_STOPS = 1.5
# The face's lit level: this percentile of the clean-skin illumination.
_LIT_PERCENTILE = 75.0
# Pixels this far (stops) below their surroundings are features, not shadow.
_FEATURE_STOPS = 0.8
# Chromaticity distance (in units of the skin's own robust spread) where
# a pixel starts to read as makeup rather than shaded skin.
_PIGMENT_SPREAD = 3.0
_MIN_SUPPORT_PX = 400
_SHOULDER = 0.8
_MIN_CHROMA_SPREAD = 0.006


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _masked_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(values * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-6), den


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] * q / 100.0)])


def illumination_map(
    canvas: np.ndarray, skin_mask: np.ndarray, ied: float,
    exclude_mask: np.ndarray | None = None,
):
    """Broad log2 illumination of the face's clean skin.

    Returns ``(log_y, illum, apply_mask, clean, lit)``: the photo's log2
    linear luminance, its smooth illumination over clean skin, the mask the
    fill may change, the clean-skin weight and the face's lit level (stops),
    or ``None`` when there is too little skin to measure.
    """
    skin = normalize_mask(skin_mask)
    if skin is None:
        return None
    skin = skin.astype(np.float32)
    if exclude_mask is not None:
        skin = skin * (1.0 - np.clip(normalize_mask(exclude_mask), 0.0, 1.0))
    if float((skin > 0.5).sum()) < _MIN_SUPPORT_PX:
        return None

    lin = _srgb_to_linear(np.clip(canvas.astype(np.float32) / 255.0, 0.0, 1.0))  # BGR
    y = 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]
    core = skin > 0.5
    # Floor relative to the face's own skin, so it scales with exposure.
    floor = max(float(np.median(y[core])) * 1e-3, 1e-7)
    log_y = np.log2(y + floor).astype(np.float32)
    sigma = max(1.5, _SIGMA_IED_FRAC * float(ied))

    # Pass 1: rough illumination over all skin, to find small dark features.
    rough, _ = _masked_blur(log_y, skin, sigma)
    feature = np.clip((rough - log_y - _FEATURE_STOPS * 0.5) / (_FEATURE_STOPS * 0.5),
                      0.0, 1.0)

    # Makeup: chromaticity far from the face's own skin colour.
    total = lin.sum(axis=2) + 1e-6
    chroma = np.stack([lin[..., 2] / total, lin[..., 1] / total], axis=2)
    centre = np.median(chroma[core], axis=0)
    dev = np.linalg.norm(chroma - centre, axis=2)
    # The floor keeps an evenly painted face (tiny spread) from reading the
    # slight colour shift of real shade as makeup. Chromaticity is a ratio,
    # so the floor does not depend on how light or dark the skin is.
    spread = max(1.4826 * float(np.median(dev[core])), _MIN_CHROMA_SPREAD)
    pigment = np.clip((dev / spread - _PIGMENT_SPREAD) / _PIGMENT_SPREAD, 0.0, 1.0)

    clean = skin * (1.0 - feature) * (1.0 - pigment)
    if float((clean > 0.5).sum()) < _MIN_SUPPORT_PX:
        return None
    illum, _ = _masked_blur(log_y, clean, sigma)
    lit = _weighted_percentile(illum[clean > 0.5], clean[clean > 0.5], _LIT_PERCENTILE)
    # Features and makeup sit in the same light as the skin around them,
    # so they are lifted with it; only features dark enough to be openings
    # (nostrils, lash line) are faded.
    apply_mask = skin * (1.0 - feature)
    return log_y, illum, apply_mask, clean, lit


def face_shadow_fill(
    canvas: np.ndarray,
    strength: float,
    ied: float,
    skin_mask: np.ndarray | None,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Lift harsh shadows on face skin by a soft, shape-keeping fill light.

    Args:
        canvas: (H, W, 3) BGR face crop, uint8 or float32 on the 0-255 scale.
        strength: 0-100; 0 returns ``canvas`` unchanged.
        ied: inter-eye distance in pixels (sets the illumination scale).
        skin_mask: face skin mask, 0-1 or 0-255.
        exclude_mask: optional mask of pixels never changed and never
            measured (eyes, brows, lips, mouth).

    Returns:
        Same shape and dtype as ``canvas``.
    """
    if strength <= 0 or skin_mask is None or ied <= 0:
        return canvas
    measured = illumination_map(canvas, skin_mask, ied, exclude_mask)
    if measured is None:
        return canvas
    _, illum, apply_mask, _, lit = measured

    depth = np.maximum(lit - illum, 0.0)
    s = min(float(strength), 100.0) / 100.0
    lift = np.minimum(np.maximum(depth - _KNEE_STOPS, 0.0) * _FILL * s, _MAX_LIFT_STOPS)
    gain = np.exp2(lift).astype(np.float32) - 1.0
    gain *= apply_mask
    if float(gain.max()) <= 1e-4:
        return canvas

    src = canvas.astype(np.float32)
    lin = _srgb_to_linear(np.clip(src / 255.0, 0.0, 1.0))
    lifted = lin * (1.0 + gain[..., None])
    # Shoulder above 0.8 linear: a lifted pixel rolls off instead of clipping.
    over = lifted > _SHOULDER
    lifted = np.where(over, _SHOULDER + (1.0 - _SHOULDER)
                      * np.tanh((lifted - _SHOULDER) / (1.0 - _SHOULDER)), lifted)
    out = _linear_to_srgb(lifted) * 255.0
    touched = gain[..., None] > 0
    out = np.where(touched, out, src)
    if canvas.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out.astype(canvas.dtype)
