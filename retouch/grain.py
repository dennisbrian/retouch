"""Organic, clumped, luminance-correlated film grain.

Real silver-halide film grain is NOT uniform Gaussian noise. It is:

* Clumped -- silver halide crystals form aggregates; the noise has spatial
  autocorrelation. We model this by Gaussian-blurring a noise field.
* Luminance-correlated -- shadows have *more* visible grain than highlights
  because the d-min/d-max range is wider in the toe. We use
  ``amplitude * (1 - luma) ** luma_power``.
* Frequency-shaped -- a flat noise distribution would look "TV static."
  Real grain has a soft 1/f^alpha roll-off; we approximate via blur.

Reference: ``docs/FUJI_COLOR_RESEARCH.md`` section 3.4.

Public API:
    apply_film_grain(img_bgr, strength, clump_sigma, luma_power, chroma, seed) -> BGR
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .color_space import bgr_to_lab, lab_to_bgr, bgr_f32_to_lab_f32, lab_f32_to_bgr_f32


def apply_film_grain(
    img_bgr: np.ndarray,
    strength: float = 0.3,
    clump_sigma: float = 1.2,
    luma_power: float = 1.2,
    chroma: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply organic film grain to a BGR image.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        strength: Grain intensity in [0, 1]. 0 = no grain, 1 = very strong.
            A value of 0.3 is a "fine Fuji grain" baseline.
        clump_sigma: Gaussian sigma for the clumping blur, in pixels.
            1.2 approx 2-3 pixel autocorrelation length.
        luma_power: Exponent for luminance modulation. 1.2 = shadows
            get ~2x the grain of highlights.
        chroma: Fraction of grain that is chroma (vs luminance).
            0.0 = pure luminance grain (recommended for Fuji look);
            0.3 = subtle colour noise.
        seed: Optional integer seed for reproducible grain fields.

    Returns:
        (H, W, 3) uint8 BGR image with grain applied.
    """
    if strength <= 0.0:
        return img_bgr
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"apply_film_grain: expected HxWx3 BGR, got shape {img_bgr.shape}")

    is_float = img_bgr.dtype == np.float32
    strength = float(np.clip(strength, 0.0, 1.0))
    chroma = float(np.clip(chroma, 0.0, 1.0))

    h, w = img_bgr.shape[:2]
    rng = np.random.default_rng(seed)

    g_low = _generate_clumped_noise(h, w, clump_sigma, rng)
    g_mid = _generate_clumped_noise(h, w, clump_sigma, rng) if chroma > 0 else None
    g_high = _generate_clumped_noise(h, w, clump_sigma, rng) if chroma > 0 else None

    if is_float:
        lab = bgr_f32_to_lab_f32(img_bgr)
    else:
        lab = bgr_to_lab(img_bgr)
    l = lab[:, :, 0].astype(np.float32) / 100.0
    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    lum_mod = np.power(1.0 - l, luma_power)
    amplitude = strength * 7.0

    l_out = np.clip(lab[:, :, 0].astype(np.float32) + g_low * amplitude * lum_mod, 0, 100)
    if g_mid is not None and g_high is not None:
        a_out = np.clip(a_chan + g_mid * amplitude * lum_mod * chroma, -128, 127)
        b_out = np.clip(b_chan + g_high * amplitude * lum_mod * chroma, -128, 127)
        out_lab = np.stack([l_out, a_out, b_out], axis=-1).astype(np.float32)
    else:
        out_lab = np.stack([l_out, a_chan, b_chan], axis=-1).astype(np.float32)

    if is_float:
        return lab_f32_to_bgr_f32(out_lab)
    return lab_to_bgr(out_lab)


def _generate_clumped_noise(h: int, w: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a clumped noise field with the given autocorrelation length.

    Implementation: generate Gaussian noise at half resolution, Gaussian-blur
    it to introduce spatial autocorrelation, then upsample to full
    resolution. The half-resolution generation saves 4x the noise samples.

    Returns:
        (h, w) float32 noise field with mean approx 0 and unit std.
    """
    if sigma <= 0.0:
        return rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    gh = max(h // 2, 32)
    gw = max(w // 2, 32)
    noise = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
    ksize = max(int(sigma * 4) | 1, 3)
    noise = cv2.GaussianBlur(noise, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    if (gh, gw) != (h, w):
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    std = noise.std()
    if std > 1e-6:
        noise = noise / std
    return noise


def _generate_luminance_mask(shape: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
    """Generate a smooth low-frequency luminance mask in [0, 1].

    Used by the shadow_boost calculation in ``apply_film_grain``: returns a
    2D array where each pixel is a smooth, low-frequency random value in
    [0, 1] that modulates grain amplitude (so shadows get more grain than
    highlights at the *image* level, not just per-pixel).

    Args:
        shape: (H, W) output shape.
        seed: Optional seed for reproducibility.

    Returns:
        (H, W) float32 array in [0, 1].
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    gh = max(h // 16, 4)
    gw = max(w // 16, 4)
    low = rng.uniform(0.0, 1.0, size=(gh, gw)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(low, 0.0, 1.0).astype(np.float32)


def generate_film_grain(
    shape: Tuple[int, int],
    grain_strength: float = 0.5,
    clumping_sigma: float = 1.2,
    shadow_boost: float = 1.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a 2D organic film grain field in [-1, 1].

    Returns just the noise field (not applied to an image). Use
    ``apply_film_grain`` to apply to a BGR image. Combine via::

        grain = generate_film_grain(img.shape[:2], strength=0.5)
        # multiply by image luminance, add to L channel, etc.

    Args:
        shape: (H, W) output shape.
        grain_strength: 0 to 1, scales overall amplitude.
        clumping_sigma: Gaussian blur sigma for autocorrelation (1.2 ≈ 2-3 px).
        shadow_boost: Multiplier for grain in shadow regions (1.5 = 50% more).
        seed: Optional seed for reproducibility.

    Returns:
        (H, W) float32 in [-1, 1].
    """
    if grain_strength <= 0.0:
        return np.zeros(shape, dtype=np.float32)

    h, w = shape
    rng = np.random.default_rng(seed)
    g = _generate_clumped_noise(h, w, clumping_sigma, rng)

    if clumping_sigma > 0.0 and shadow_boost != 1.0:
        lum_mask = _generate_luminance_mask(shape, seed=seed)
        modulation = 1.0 + (1.0 - lum_mask) * (shadow_boost - 1.0)
        g = g * modulation * grain_strength
    else:
        g = g * grain_strength

    peak = float(np.max(np.abs(g))) if g.size else 0.0
    if peak > 1e-6:
        g = g / peak
    return g.astype(np.float32)


def grain_autocorrelation_check(grain: np.ndarray) -> float:
    """Measure how clumpy a grain field is via adjacent-pixel Pearson r.

    Returns a value in [-1, 1]:
    - 0 = white noise (uncorrelated)
    - close to 1 = highly clumped (autocorrelated)
    - close to -1 = checkerboard pattern (anti-correlated)

    Args:
        grain: (H, W) float32 noise field.

    Returns:
        float, mean of horizontal and vertical adjacent-pixel correlation.
    """
    if grain.ndim != 2 or grain.size < 2:
        raise ValueError(f"expected 2D grain field, got shape {grain.shape}")

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a_mean = a.mean()
        b_mean = b.mean()
        a_dev = a - a_mean
        b_dev = b - b_mean
        denom = np.sqrt(np.sum(a_dev * a_dev) * np.sum(b_dev * b_dev))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(a_dev * b_dev) / denom)

    h_corr = _pearson(grain[:, :-1].ravel(), grain[:, 1:].ravel())
    v_corr = _pearson(grain[:-1, :].ravel(), grain[1:, :].ravel())
    return 0.5 * (h_corr + v_corr)
