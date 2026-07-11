"""R9 — Intrinsic decomposition (albedo x shading).

Splits a face image into an albedo (pigment/reflectance) map and a shading
(form) map *before* any smoothing, so even-out-albedo smoothing never touches
form. This is the structural cure for plastic skin and the basis for the R12
reuse of ``decompose_intrinsic``.

Algorithm
---------
Weighted Least Squares intrinsic images (Fattal et al. 2008, "Edge-Preserving
Decompositions for Single Image Tone Mapping", adapted):

    L = luminance(img)            # float32 [0,1]
    logL = log(L + eps)
    S = argmin  sum w(x)(S-logL)^2 + lam * ||grad S||^2
    A = exp(logL - S)             # albedo (reflectance)

The edge-aware weight

    w = (|grad logL| / mean(|grad logL|) + eps) ** (-alpha)

is small at high-frequency albedo edges (texture) so the smooth shading S does
not chase pigment, and large over flat regions (and gentle form ramps) so S
tracks real form. The WLS problem is solved with a vectorized Jacobi fixed-point
over the discrete Euler-Lagrange equation:

    S_{new} = (w*logL + lam*(neighbors)) / (w + 4*lam)

which is O(N) per iteration and converges in ~12 passes for the smooth prior.

Colorspace: BGR in (engine convention) -> explicit cvtColor -> RGB -> luma.
All pixel math is float32.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["decompose_intrinsic", "even_albedo"]

_EPS_LOG = 1e-4
_WLS_ITERS = 20
_W_CAP = 50.0  # cap the edge-aware data weight so flat regions can be smoothed
_LUMA_W = (0.2126, 0.7152, 0.0722)  # Rec.709 luma weights (RGB order)
_REF_FACE_WIDTH = 200.0


def _to_f32_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary BGR input to float32 [0,1] without a uint8
    intermediate in any arithmetic path."""
    if img_bgr.dtype == np.uint8:
        return img_bgr.astype(np.float32) / 255.0
    if img_bgr.dtype != np.float32:
        return img_bgr.astype(np.float32)
    return img_bgr


def _shading_sigma(
    h: int, w: int, face_width: Optional[float]
) -> float:
    """Radius (px) of the broad low-frequency shading pass.

    Shading is the smooth form layer, so it must drop mid/low-frequency
    pigment too (pure WLS keeps a piecewise-constant albedo's DC offset
    because it has zero gradient). The broad pass removes that, and scales
    with resolution / face size per the kernel-scaling discipline.
    """
    base = max(6.0, min(h, w) * 0.05)
    if face_width is not None:
        base *= max(0.5, min(3.0, face_width / _REF_FACE_WIDTH))
    return float(base)


def _wls_smooth(
    logL: np.ndarray,
    lam: float,
    alpha: float,
    face_width: Optional[float] = None,
    iters: int = _WLS_ITERS,
) -> np.ndarray:
    """Edge-aware WLS smooth of ``logL`` via Jacobi fixed-point, then a broad
    low-frequency shading pass.

    The Jacobi step solves the discrete Euler-Lagrange equation for the
    edge-aware WLS prior (preserves form edges, drops fine texture). A final
    resolution-scaled Gaussian on the log-shading extracts the low-frequency
    form layer and removes mid-frequency pigment the local WLS cannot (the
    DC-offset problem for piecewise-constant albedo). Returns ``S`` (float32,
    same shape as ``logL``).
    """
    h, w = logL.shape[:2]
    # Forward-difference gradient magnitude of log-luminance.
    gx = np.roll(logL, -1, axis=1) - logL
    gy = np.roll(logL, -1, axis=0) - logL
    grad = np.sqrt(gx * gx + gy * gy)
    mean_grad = float(grad.mean()) + 1e-8
    wt = np.minimum(
        _W_CAP, np.power(grad / mean_grad + 1e-4, -alpha)
    ).astype(np.float32)

    denom = wt + 4.0 * lam + 1e-8
    S = logL.copy()
    for _ in range(iters):
        sp = np.pad(S, 1, mode="edge")
        neighbors = (
            sp[2:, 1:-1] + sp[:-2, 1:-1] + sp[1:-1, 2:] + sp[1:-1, :-2]
        )
        S = (wt * logL + lam * neighbors) / denom

    sigma = _shading_sigma(h, w, face_width)
    if sigma > 0.0:
        S = cv2.GaussianBlur(S, (0, 0), sigma)
    return S.astype(np.float32)


def _effective_lam(lam: float, face_width: Optional[float]) -> float:
    """Broaden smoothing for larger faces (broader shading span)."""
    if face_width is None:
        return float(lam)
    scale = max(0.5, min(3.0, face_width / _REF_FACE_WIDTH))
    return float(lam) * scale


def decompose_intrinsic(
    img_bgr: np.ndarray,
    face_width: Optional[float] = None,
    lam: float = 0.02,
    alpha: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose ``img_bgr`` into (albedo_bgr, shading_gray).

    Args:
        img_bgr: Input image, BGR, uint8 or float32 [0,1].
        face_width: Optional face width (px); broadens the smoothing kernel for
            larger faces. Default behavior needs no geometry.
        lam: Smoothness weight (higher = smoother shading). Sane 0.01-0.05.
        alpha: Edge-awareness exponent for the WLS weight. Sane ~1.2.

    Returns:
        albedo_bgr: float32 [0,1], BGR reflectance map (pigment only).
        shading_gray: float32, single-channel relative shading, mean ~= 1.
            Multiply ``albedo_bgr * shading_gray[..., None]`` to reconstruct.

    Reconstruction closes: luminance(albedo_bgr) * shading_gray == L (luminance).
    """
    img = _to_f32_bgr(img_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("decompose_intrinsic expects a 3-channel BGR image")

    # BGR -> RGB (explicit boundary per repo colorspace rule).
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    L = (
        _LUMA_W[0] * r + _LUMA_W[1] * g + _LUMA_W[2] * b
    ).astype(np.float32)

    logL = np.log(L + _EPS_LOG)

    lam_eff = _effective_lam(lam, face_width)
    S = _wls_smooth(logL, lam_eff, float(alpha), face_width=face_width)

    expS = np.exp(S)
    shading = expS / float(expS.mean())  # relative shading, mean ~= 1

    # Per-channel reflectance: dividing by shading removes form, keeps pigment.
    albedo_bgr = np.clip(img / shading[..., None], 0.0, 1.0).astype(np.float32)
    shading = shading.astype(np.float32)
    return albedo_bgr, shading


def even_albedo(
    img_bgr: np.ndarray,
    albedo_blur_strength: float,
    face_width: Optional[float] = None,
) -> np.ndarray:
    """Even-out-albedo primitive: blur the pigment map, keep the form.

    Decomposes into albedo x shading, Gaussian-blurs the albedo (pigment only,
    not skin form), then re-multiplies by the untouched shading layer. This is
    the "even out blotches, keep form" operator for S2/C2 reuse.

    Args:
        img_bgr: Input BGR image (uint8 or float32).
        albedo_blur_strength: 0 (no-op) .. 1 (fully evened albedo).
        face_width: Optional; broadens the decomposition smoothing.

    Returns:
        float32 [0,1] BGR image with evened pigment and preserved form.
    """
    if not 0.0 <= float(albedo_blur_strength) <= 1.0:
        raise ValueError("albedo_blur_strength must be in [0, 1]")

    albedo, shading = decompose_intrinsic(img_bgr, face_width=face_width)
    s = float(albedo_blur_strength)
    if s <= 0.0:
        return _to_f32_bgr(img_bgr)

    h, w = albedo.shape[:2]
    # Blur radius scales with resolution (KERNEL_SCALE discipline): broader
    # for larger faces and with blur strength, so a smooth albedo blotch is
    # actually evened rather than left untouched.
    if face_width is not None:
        sigma = max(6.0, face_width * 0.12 * (0.5 + s))
    else:
        sigma = max(6.0, min(h, w) * 0.12 * (0.5 + s))

    albedo_blurred = cv2.GaussianBlur(albedo, (0, 0), sigma)
    blended = albedo * (1.0 - s) + albedo_blurred * s
    out = np.clip(blended * shading[..., None], 0.0, 1.0)
    return out.astype(np.float32)
