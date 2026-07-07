"""C3 — Parametric Film-Density Engine.

Replaces LUT-based film emulation with a physically-motivated subtractive
density model + hue-preserving master tone-map. Exposes the parameters LUTs
bake in: per-dye H&D curves, 3x3 crosstalk matrix on densities, and a skew
slider spanning "digital-clean" (hue-locked) to "filmic-drift" (per-channel).

Pipeline:
    1. Inverse sRGB EOTF -> scene-linear r in [0,1]
    2. Per-channel log-density H&D curve (toe/shoulder/midpoint/gamma per RGB)
    3. 3x3 crosstalk matrix on densities (dye impurity)
    4. Subtractive recombination: r' = r * 10^(-(M @ D))
    5. Hue-preserving master tone-map (luma compression + chroma re-attach)
       with skew slider blending toward per-channel result
    6. sRGB companding -> output

All math in float32 (uint8 is broken for shadow-density — see
docs/PLAN_C3_FILM_DENSITY.md §2.3). Accepts float32 [0,255] BGR or uint8 BGR,
returns input dtype.

Reference: docs/PLAN_C3_FILM_DENSITY.md
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .tonal import _sigmoid, _safe_midpoint


_EPS = 1e-6


def _srgb_to_linear(rgb01: np.ndarray) -> np.ndarray:
    """sRGB inverse companding (matches color_science.py:59-63)."""
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return np.where(
        rgb01 <= 0.04045,
        rgb01 / 12.92,
        np.power((rgb01 + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * np.power(lin, 1.0 / 2.4) - 0.055,
    )


def _hd_curve_logdensity(
    logE: np.ndarray,
    toe: float,
    shoulder: float,
    midpoint: float,
    gamma: float,
) -> np.ndarray:
    """Apply the H&D sigmoid in log-density space.

    Maps logE (which can be negative) to a density D in [0, ~2.5].
    Uses the same piecewise-cubic sigmoid as tonal._sigmoid, but remapped
    so that very low logE -> ~0 density and high logE -> D_max.

    Returns density in [0, D_MAX].
    """
    D_MAX = 2.2
    toe = float(np.clip(toe, 0.0, 0.5))
    shoulder = float(np.clip(shoulder, 0.0, 0.5))
    midpoint = float(np.clip(midpoint, 0.0, 1.0))
    gamma = max(gamma, 0.01)

    logE_min = -6.0
    logE_max = 0.0
    span = logE_max - logE_min
    t = np.clip((logE - logE_min) / max(span, _EPS), 0.0, 1.0)

    toe_end = toe
    shoulder_start = 1.0 - shoulder
    raw = np.empty_like(t)

    in_toe = t < toe_end
    if toe > _EPS:
        tt = np.clip(t[in_toe] / toe_end, 0.0, 1.0)
        raw[in_toe] = (toe * 0.5) * (1.0 - np.cos(np.pi * tt))
    else:
        raw[in_toe] = t[in_toe] * 0.0

    in_mid = (~in_toe) & (t <= shoulder_start)
    mid_span = max(shoulder_start - toe_end, _EPS)
    tm = np.clip((t[in_mid] - toe_end) / mid_span, 0.0, 1.0)
    sig = _sigmoid(tm, midpoint=_safe_midpoint(midpoint, toe, shoulder))
    if gamma != 1.0:
        sig = np.power(np.clip(sig, 0.0, 1.0), 1.0 / gamma)
    raw[in_mid] = toe + sig * (1.0 - toe - shoulder)

    in_shoulder = t > shoulder_start
    if shoulder > _EPS:
        sh_span = max(1.0 - shoulder_start, _EPS)
        ts = np.clip((t[in_shoulder] - shoulder_start) / sh_span, 0.0, 1.0)
        raw[in_shoulder] = 1.0 - shoulder + (shoulder * 0.5) * (1.0 - np.cos(np.pi * ts))
    else:
        raw[in_shoulder] = 1.0

    raw = np.clip(raw, 0.0, 1.0)
    return raw * D_MAX


def _build_crosstalk_matrix(
    cy_mg: float,
    cy_ye: float,
    mg_ye: float,
) -> np.ndarray:
    """Build the 3x3 dye-crosstalk matrix from 3 off-diagonal params.

    Ordering: rows/cols are (R, G, B) which map to (Cyan, Magenta, Yellow)
    dye layers respectively. Diagonal = 1.0 (the dye absorbs its own band).

    Sign convention (per design doc §2.2): positive off-diagonal = the
    row-dye absorbs extra column-band light. cy_mg = cyan leaks into
    magenta (shadow warmth).

    The matrix is asymmetric in general, but for the 3-param reduction we
    assume cyan->magenta, cyan->yellow, magenta->yellow are the dominant
    leaks and mirror them for symmetry (a Portra-class approximation).

    Returns a (3, 3) float32 matrix.
    """
    cy_mg = float(np.clip(cy_mg, -0.15, 0.15))
    cy_ye = float(np.clip(cy_ye, -0.15, 0.15))
    mg_ye = float(np.clip(mg_ye, -0.15, 0.15))

    M = np.array(
        [
            [1.0,   cy_mg, cy_ye],
            [cy_mg, 1.0,   mg_ye],
            [cy_ye, mg_ye, 1.0  ],
        ],
        dtype=np.float32,
    )
    return M


def _master_tonemap(
    rgb_lin: np.ndarray,
    toe: float,
    shoulder: float,
    midpoint: float,
    gamma: float,
    strength: float,
    skew: float,
) -> np.ndarray:
    """Hue-preserving master tone-map with skew control.

    At skew=0: luma is compressed by the H&D sigmoid, chroma is re-attached
    by the luma ratio (hue-locked, ACES/AgX-class).
    At skew=1: each channel is independently curve-mapped (classic drift).

    ``rgb_lin`` is scene-linear [0,1]. Returns scene-linear [0,1].
    """
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0).astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    skew = float(np.clip(skew, 0.0, 1.0))
    if strength <= 0.0:
        return rgb_lin

    Y = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    Y = np.clip(Y, 0.0, 1.0)

    logY = np.log10(np.maximum(Y, _EPS))
    D = _hd_curve_logdensity(logY, toe, shoulder, midpoint, gamma)
    Y_mapped = np.power(10.0, -D)
    Y_mapped = np.clip(Y_mapped, 0.0, 1.0)

    if skew <= _EPS:
        hue_locked = rgb_lin * (Y_mapped / np.maximum(Y, _EPS))[..., np.newaxis]
        result = hue_locked
    elif skew >= 1.0 - _EPS:
        per_channel = np.empty_like(rgb_lin)
        for c in range(3):
            logC = np.log10(np.maximum(rgb_lin[..., c], _EPS))
            Dc = _hd_curve_logdensity(logC, toe, shoulder, midpoint, gamma)
            per_channel[..., c] = np.clip(np.power(10.0, -Dc), 0.0, 1.0)
        result = per_channel
    else:
        hue_locked = rgb_lin * (Y_mapped / np.maximum(Y, _EPS))[..., np.newaxis]
        per_channel = np.empty_like(rgb_lin)
        for c in range(3):
            logC = np.log10(np.maximum(rgb_lin[..., c], _EPS))
            Dc = _hd_curve_logdensity(logC, toe, shoulder, midpoint, gamma)
            per_channel[..., c] = np.clip(np.power(10.0, -Dc), 0.0, 1.0)
        result = hue_locked * (1.0 - skew) + per_channel * skew

    result = rgb_lin * (1.0 - strength) + result * strength
    return np.clip(result, 0.0, 1.0).astype(np.float32)


class FilmDensityEngine:
    """Parametric film-density engine.

    All ops in float32. Accepts float32 [0,255] BGR or uint8 BGR, returns
    the input dtype.
    """

    def apply(
        self,
        img_bgr: np.ndarray,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Apply the film-density model to a BGR image.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 [0,255] BGR image.
            params: Dict of film params (see docs §4). Must contain at
                least ``enable``. Missing keys use defaults.

        Returns:
            (H, W, 3) same dtype as input.
        """
        if not params.get("enable", False):
            return img_bgr

        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(
                f"FilmDensityEngine.apply: expected HxWx3 BGR, got shape {img_bgr.shape}"
            )

        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_f = np.clip(img_bgr, 0.0, 255.0).astype(np.float32)
        else:
            img_f = img_bgr.astype(np.float32)
        original = img_f.copy()

        rgb01 = img_f[..., ::-1] * (1.0 / 255.0)
        rgb_lin = _srgb_to_linear(rgb01)
        rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

        p_toe = [
            params.get("toe_r", 0.10),
            params.get("toe_g", 0.10),
            params.get("toe_b", 0.10),
        ]
        p_shoulder = [
            params.get("shoulder_r", 0.10),
            params.get("shoulder_g", 0.10),
            params.get("shoulder_b", 0.10),
        ]
        p_midpoint = params.get("midpoint", 0.50)
        p_gamma = params.get("gamma", 1.0)

        ct_cy_mg = params.get("crosstalk_cy_mg", 0.06)
        ct_cy_ye = params.get("crosstalk_cy_ye", 0.03)
        ct_mg_ye = params.get("crosstalk_mg_ye", 0.02)

        density_active = (
            any(abs(t) > _EPS for t in p_toe)
            or any(abs(s) > _EPS for s in p_shoulder)
            or abs(ct_cy_mg) > _EPS
            or abs(ct_cy_ye) > _EPS
            or abs(ct_mg_ye) > _EPS
            or abs(p_gamma - 1.0) > _EPS
        )

        M = _build_crosstalk_matrix(ct_cy_mg, ct_cy_ye, ct_mg_ye)

        if density_active:
            density = np.empty_like(rgb_lin)
            for c in range(3):
                logE = np.log10(np.maximum(rgb_lin[..., c], _EPS))
                density[..., c] = _hd_curve_logdensity(
                    logE, p_toe[c], p_shoulder[c], p_midpoint, p_gamma
                )

            h, w = density.shape[:2]
            D_flat = density.reshape(-1, 3)
            D_mix = (D_flat @ M.T).reshape(h, w, 3)
            D_mix = np.clip(D_mix, 0.0, 3.0)

            T = np.power(10.0, -D_mix)
            rgb_post = rgb_lin * T
            rgb_post = np.clip(rgb_post, 0.0, 1.0).astype(np.float32)
        else:
            rgb_post = rgb_lin

        tm_strength = params.get("tonemap_strength", 0.7)
        tm_toe = params.get("tonemap_toe", 0.10)
        tm_shoulder = params.get("tonemap_shoulder", 0.15)
        tm_skew = params.get("skew", 0.3)

        if tm_strength > 0.0:
            rgb_post = _master_tonemap(
                rgb_post,
                toe=tm_toe,
                shoulder=tm_shoulder,
                midpoint=p_midpoint,
                gamma=p_gamma,
                strength=tm_strength,
                skew=tm_skew,
            )

        rgb_out_lin = np.clip(rgb_post, 0.0, 1.0)
        rgb_out_01 = _linear_to_srgb(rgb_out_lin)
        rgb_out_01 = np.clip(rgb_out_01, 0.0, 1.0)

        bgr_out_f = rgb_out_01[..., ::-1] * 255.0
        bgr_out_f = np.clip(bgr_out_f, 0.0, 255.0).astype(np.float32)

        strength = float(np.clip(params.get("strength", 1.0), 0.0, 1.0))
        if strength < 1.0:
            bgr_out_f = original * (1.0 - strength) + bgr_out_f * strength
            bgr_out_f = np.clip(bgr_out_f, 0.0, 255.0).astype(np.float32)

        if is_float:
            return bgr_out_f
        return np.clip(np.round(bgr_out_f), 0, 255).astype(np.uint8)

    def apply_from_context(
        self,
        img_bgr: np.ndarray,
        ctx: Any,
    ) -> np.ndarray:
        """Apply using a ProcessingContext (engine wiring helper).

        Reads the film.* params off the context object.
        """
        params = _context_to_params(ctx)
        return self.apply(img_bgr, params)


def _context_to_params(ctx: Any) -> Dict[str, Any]:
    """Extract the film.* param dict from a ProcessingContext."""
    return {
        "enable": getattr(ctx, "film_enable", False),
        "strength": getattr(ctx, "film_strength", 1.0),
        "toe_r": getattr(ctx, "film_toe_r", 0.10),
        "toe_g": getattr(ctx, "film_toe_g", 0.10),
        "toe_b": getattr(ctx, "film_toe_b", 0.10),
        "shoulder_r": getattr(ctx, "film_shoulder_r", 0.10),
        "shoulder_g": getattr(ctx, "film_shoulder_g", 0.10),
        "shoulder_b": getattr(ctx, "film_shoulder_b", 0.10),
        "midpoint": getattr(ctx, "film_midpoint", 0.50),
        "gamma": getattr(ctx, "film_gamma", 1.0),
        "crosstalk_cy_mg": getattr(ctx, "film_crosstalk_cy_mg", 0.06),
        "crosstalk_cy_ye": getattr(ctx, "film_crosstalk_cy_ye", 0.03),
        "crosstalk_mg_ye": getattr(ctx, "film_crosstalk_mg_ye", 0.02),
        "tonemap_strength": getattr(ctx, "film_tonemap_strength", 0.7),
        "tonemap_toe": getattr(ctx, "film_tonemap_toe", 0.10),
        "tonemap_shoulder": getattr(ctx, "film_tonemap_shoulder", 0.15),
        "skew": getattr(ctx, "film_skew", 0.3),
    }
