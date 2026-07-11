"""R12 — Specular/diffuse re-render finish (dichromatic reflection model).

Skin reflectance = diffuse (matte, chroma-bearing) + specular (highlight,
desaturated, view/light-dependent). S4 (shine_removal) *removes* the specular
layer; R12 makes it *editable* and re-emits a single physically-consistent
matte<->powder<->dewy<->glass-skin slider.

Algorithm (no neural net) -- dichromatic reflection model:

    img = m_d * rho_d (diffuse, chromatic) + m_s * rho_s (specular, ~white)

A specular pixel is bright AND near-gray (low chroma). ``extract_specular``
isolates that layer; ``render_finish`` re-emits the requested finish by
attenating or re-adding it; ``specular_recolor`` neutralises a coloured
(venue-cast) highlight before it is re-added.

Colorspace: BGR input (engine convention) -> explicit normalization at the
boundary. All pixel math is float32; clip only at the end.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["extract_specular", "render_finish", "specular_recolor"]

_VALID_MODES = ("matte", "powder", "dewy", "glass_skin")


def _to_f32(img_bgr: np.ndarray) -> np.ndarray:
    """Normalize a BGR input to float32 [0, 255] without uint8 arithmetic."""
    if img_bgr.dtype == np.uint8:
        return img_bgr.astype(np.float32)
    if img_bgr.dtype == np.float32:
        return np.ascontiguousarray(img_bgr, dtype=np.float32)
    return img_bgr.astype(np.float32)


def _as_spec_layer(specular_layer: np.ndarray) -> np.ndarray:
    """Coerce a specular map to a single-channel float32 [0, 255] array."""
    s = np.ascontiguousarray(specular_layer, dtype=np.float32)
    if s.ndim == 3:
        s = np.max(s, axis=2)
    return s


def extract_specular(img_bgr: np.ndarray, rho: float = 0.95) -> np.ndarray:
    """Per-pixel specular (highlight) intensity map, float32 [0, 255].

    In normalized RGB a specular pixel has near-equal channels (low chroma)
    AND high intensity. We isolate it with the dichromatic specular-free
    residual gated by high intensity and low chroma:

        spec = I * intensity_gate * chroma_gate
        chroma_gate = 1 - clip(chroma / chroma_thresh, 0, 1)
        chroma_thresh = max(0.06, 0.15 * (2 - rho))   # rho = max diffuse chroma

    ``rho`` (max diffuse chromaticity, 0.9-1.0) widens the accepted chroma band:
    a higher ``rho`` treats more-saturated pixels as diffuse (less specular).
    The map is feathered (small Gaussian) so re-emission edges don't tear.

    Args:
        img_bgr: BGR image, uint8 or float32 [0, 255].
        rho: Max diffuse chromaticity in [0.9, 1.0].

    Returns:
        (H, W) float32 specular intensity in [0, 255].
    """
    img = _to_f32(img_bgr)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    s_sum = b + g + r
    eps = 1e-4
    nr = r / (s_sum + eps)
    ng = g / (s_sum + eps)
    nb = b / (s_sum + eps)

    # Max channel intensity (bright highlight body).
    I = np.max(img, axis=2)

    # Normalized chroma: 0 = perfectly gray (specular candidate),
    # up to ~1 = fully saturated (pure diffuse pigment).
    nmax = np.maximum(np.maximum(nr, ng), nb)
    nmin = np.minimum(np.minimum(nr, ng), nb)
    chroma = nmax - nmin

    rho = float(min(1.0, max(0.5, rho)))
    chroma_thresh = max(0.06, 0.15 * (2.0 - rho))
    chroma_gate = 1.0 - np.clip(chroma / chroma_thresh, 0.0, 1.0)

    # Only genuine highlights carry specular (not mid-gray skin).
    intensity_gate = np.clip((I - 170.0) / 70.0, 0.0, 1.0)

    spec = I * intensity_gate * chroma_gate

    # Feather so re-emission doesn't tear at highlight boundaries.
    min_dim = min(img.shape[:2])
    sigma = max(1.0, min_dim * 0.004)
    spec = cv2.GaussianBlur(spec, (0, 0), sigma)

    return np.clip(spec, 0.0, 255.0).astype(np.float32)


def specular_recolor(
    img_bgr: np.ndarray,
    specular_layer: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Neutralise the colour cast of specular highlights only.

    A venue-cast highlight (e.g. blue/green light) carries chroma in its
    specular contribution. We keep the neutral (min-channel) level and
    desaturate the coloured *excess* by ``strength`` inside specular regions.

    Args:
        img_bgr: BGR image, uint8 or float32 [0, 255].
        specular_layer: Specular map (single or 3 channel), float32 [0, 255].
            Used as the mask marking highlight regions.
        strength: 0 (no change) .. 1 (fully neutralised cast).

    Returns:
        (H, W, 3) float32 BGR image with recolor applied in highlights.
    """
    img = _to_f32(img_bgr)
    s = _as_spec_layer(specular_layer)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return img

    m = np.clip(s / 255.0, 0.0, 1.0)[..., np.newaxis]

    mn = np.min(img, axis=2, keepdims=True)            # neutral (white) level
    excess = img - mn                                  # coloured specular excess
    neutral = mn + excess * (1.0 - strength)          # desaturate the excess

    out = img * (1.0 - m) + neutral * m
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def render_finish(
    img_bgr: np.ndarray,
    specular_layer: np.ndarray,
    mode: str,
    strength: float,
    recolor_strength: float = 0.0,
) -> np.ndarray:
    """Re-emit the skin finish from the specular layer.

    Modes (``mode`` from {"matte","powder","dewy","glass_skin"}; unknown -> matte):
        * "matte":       attenuate specular:  out = img - strength * spec
        * "powder":      matte + a touch of broad diffuse softening
        * "dewy":        keep specular, add a wide specular *bloom* (soft sheen)
        * "glass_skin":  re-add specular with tight, high-contrast highlight
                         (K-beauty shuiguan ji)

    When ``recolor_strength > 0`` the specular contribution is neutralised
    toward gray (``specular_recolor``) before re-adding in dewy/glass modes.

    Args:
        img_bgr: BGR image, uint8 or float32 [0, 255].
        specular_layer: Specular map from ``extract_specular``, float32 [0, 255].
        mode: Finish mode string.
        strength: 0..1 re-emit strength.
        recolor_strength: 0..1 specular colour-cast neutralisation.

    Returns:
        (H, W, 3) float32 BGR image in [0, 255] (clipped only here).
    """
    img = _to_f32(img_bgr)
    S = _as_spec_layer(specular_layer)
    if mode not in _VALID_MODES:
        mode = "matte"
    strength = float(np.clip(strength, 0.0, 1.0))
    recolor_strength = float(np.clip(recolor_strength, 0.0, 1.0))

    h, w = img.shape[:2]
    min_dim = min(h, w)

    if mode == "matte":
        out = img - strength * S[..., np.newaxis]

    elif mode == "powder":
        out = img - strength * S[..., np.newaxis]
        # A touch of broad diffuse softening (keeps the matte powder look).
        k = max(3, int(min_dim * 0.02)) | 1
        soft = cv2.GaussianBlur(img, (k, k), 0)
        out = out + 0.12 * strength * (soft - img)

    elif mode == "dewy":
        base = img
        if recolor_strength > 0.0:
            base = specular_recolor(base, S, recolor_strength)
        # Wide, soft specular bloom -> controlled sheen.
        sigma = max(3.0, min_dim * 0.015)
        bloom = cv2.GaussianBlur(S, (0, 0), sigma)[..., np.newaxis]
        out = base + strength * bloom

    else:  # glass_skin
        base = img
        if recolor_strength > 0.0:
            base = specular_recolor(base, S, recolor_strength)
        # Tight, high-contrast highlight: sharpen the specular band and
        # re-add it on top of the base white sheen for shuiguan ji pop.
        sigma_tight = max(1.0, min_dim * 0.004)
        blurred = cv2.GaussianBlur(S, (0, 0), sigma_tight)
        tight = S - blurred                       # high-frequency highlight detail
        sheen = S[..., np.newaxis] + 1.5 * tight[..., np.newaxis]
        out = base + strength * sheen

    return np.clip(out, 0.0, 255.0).astype(np.float32)
