"""P4 skin↔makeup unmix — classical alpha composite + multi-cue prior.

Full product path: unmix → (edit α / S) → recompose.
strength=0 / all gates off → identity (same array when possible).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .chromophore import decompose_chromophores, reconstruct_from_chromophores
from .color_science import bgr_to_oklab, oklab_to_oklch


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _prep_mask(mask: Optional[np.ndarray], h: int, w: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def estimate_makeup_alpha(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    mole_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-cue soft α prior (chroma ∨ white-paint ∨ reconstruct residual)."""
    h, w = img_bgr.shape[:2]
    m = _prep_mask(skin_mask, h, w)
    if m is None:
        return np.zeros((h, w), dtype=np.float32)

    work = _to_u8(img_bgr)
    oklab = bgr_to_oklab(work.astype(np.float32))
    oklch = oklab_to_oklch(oklab)
    L, C = oklch[..., 0], oklch[..., 1]

    mel, hb = decompose_chromophores(work)
    recon = reconstruct_from_chromophores(mel, hb, img_bgr=work, skin_mask=m)
    resid = np.abs(work.astype(np.float32) - recon).mean(axis=-1) / 255.0

    cue_chroma = _smoothstep(0.10, 0.20, C)
    cue_white = _smoothstep(0.85, 0.95, L) * (1.0 - _smoothstep(0.02, 0.08, C))
    cue_resid = _smoothstep(0.04, 0.12, resid)

    hint = np.maximum(np.maximum(cue_chroma, cue_white), cue_resid) * m
    mm = _prep_mask(mole_mask, h, w)
    if mm is not None:
        hint = hint * (1.0 - mm)
    ex = _prep_mask(exclude_mask, h, w)
    if ex is not None:
        hint = hint * (1.0 - ex)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8 = np.clip(hint * 255, 0, 255).astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
    return (u8.astype(np.float32) / 255.0)


def closed_form_alpha(
    I: np.ndarray, S: np.ndarray, M: np.ndarray,
) -> np.ndarray:
    If = I.astype(np.float32)
    Sf = S.astype(np.float32)
    Mf = M.astype(np.float32)
    d = Mf - Sf
    num = ((If - Sf) * d).sum(axis=-1)
    den = (d * d).sum(axis=-1) + 1e-6
    return np.clip(num / den, 0.0, 1.0).astype(np.float32)


def recompose(
    skin: np.ndarray, makeup: np.ndarray, alpha: np.ndarray,
) -> np.ndarray:
    a = alpha.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    out = (1.0 - a) * skin.astype(np.float32) + a * makeup.astype(np.float32)
    return np.clip(out, 0, 255)


def unmix_makeup(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    *,
    mole_mask: Optional[np.ndarray] = None,
    user_alpha: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (skin_bgr, makeup_bgr, alpha) float32 [0,255] / α HxW.

    Seeds M from high-α_init pixels, S from low-α; IRLS closed-form α.
    """
    h, w = img_bgr.shape[:2]
    I = img_bgr.astype(np.float32)
    base = _prep_mask(skin_mask, h, w)
    if base is None:
        z = np.zeros((h, w), dtype=np.float32)
        return I.copy(), I.copy(), z

    if user_alpha is not None:
        alpha = _prep_mask(user_alpha, h, w)
        assert alpha is not None
    else:
        alpha = estimate_makeup_alpha(
            img_bgr, skin_mask, mole_mask=mole_mask, exclude_mask=exclude_mask,
        )

    # Seed makeup color M from high-α pixels; skin S from low-α
    hi = alpha > 0.35
    lo = (alpha < 0.15) & (base > 0.5)
    if hi.any():
        Mcol = I[hi].mean(axis=0)
    else:
        Mcol = np.array([220.0, 200.0, 210.0], dtype=np.float32)  # pale default
    if lo.any():
        S_fill = I[lo].mean(axis=0)
    else:
        S_fill = I.reshape(-1, 3).mean(axis=0)

    M = np.broadcast_to(Mcol, I.shape).copy()
    # S: observed under low α, guided fill under high α
    S = I.copy()
    a3 = alpha[..., None]
    S = I * (1.0 - a3) + S_fill * a3  # soft blend toward bare-skin mean under paint
    sigma = max(4.0, min(h, w) * 0.03)
    S = cv2.GaussianBlur(S, (0, 0), sigmaX=sigma)
    S = I * (1.0 - a3) + S * a3  # keep bare pixels close to I

    for _ in range(max(1, iterations)):
        alpha = closed_form_alpha(I, S, M)
        alpha = alpha * base
        if mole_mask is not None:
            mm = _prep_mask(mole_mask, h, w)
            if mm is not None:
                alpha = alpha * (1.0 - mm)
        if exclude_mask is not None:
            ex = _prep_mask(exclude_mask, h, w)
            if ex is not None:
                alpha = alpha * (1.0 - ex)
        hi = alpha > 0.3
        if hi.any():
            Mcol = I[hi].mean(axis=0)
            M = np.broadcast_to(Mcol, I.shape).copy()
        # Refine S: S = (I - α M) / (1-α)
        a = np.clip(alpha, 0.0, 0.95)
        S = (I - a[..., None] * M) / (1.0 - a[..., None] + 1e-6)
        S = np.clip(S, 0, 255)

    return S.astype(np.float32), M.astype(np.float32), alpha.astype(np.float32)


def even_coverage_alpha(alpha: np.ndarray, strength: float, guide: np.ndarray) -> np.ndarray:
    """Low-pass α toward regional mean (coverage evening on the layer)."""
    if strength <= 0:
        return alpha
    s = float(np.clip(strength, 0.0, 1.0))
    a = alpha.astype(np.float32)
    sigma = max(3.0, min(a.shape[:2]) * 0.04)
    blur = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma)
    return np.clip(a * (1.0 - s) + blur * s, 0.0, 1.0)


def cake_reduce(alpha: np.ndarray, strength: float) -> np.ndarray:
    """Suppress high-freq α (pore caking) while keeping low-freq coverage."""
    if strength <= 0:
        return alpha
    s = float(np.clip(strength, 0.0, 1.0))
    a = alpha.astype(np.float32)
    low = cv2.GaussianBlur(a, (0, 0), sigmaX=max(2.0, min(a.shape[:2]) * 0.02))
    high = a - low
    return np.clip(low + high * (1.0 - s), 0.0, 1.0)


def even_coverage(
    img_bgr: np.ndarray,
    alpha: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Legacy: blur I under α (kept for tests). Prefer unmix path in engine."""
    if strength <= 0:
        return img_bgr
    s = float(np.clip(strength, 0.0, 1.0))
    h, w = img_bgr.shape[:2]
    a = _prep_mask(alpha, h, w)
    if a is None or a.max() < 1e-6:
        return img_bgr
    src = img_bgr.astype(np.float32)
    sigma = max(3.0, min(h, w) * 0.04)
    blur = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma)
    a3 = (a * s)[..., None]
    out = src * (1.0 - a3) + blur * a3
    if img_bgr.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(out, 0, 255).astype(np.float32)


def apply_makeup_unmix(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    *,
    coverage_even: float = 0.0,
    cake_reduce_strength: float = 0.0,
    mole_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full product entry: unmix → edit α → recompose. All strengths 0 = identity."""
    ce = float(coverage_even or 0.0)
    cr = float(cake_reduce_strength or 0.0)
    if ce <= 0 and cr <= 0:
        return img_bgr

    S, M, alpha = unmix_makeup(
        img_bgr, skin_mask, mole_mask=mole_mask, exclude_mask=exclude_mask,
    )
    if ce > 0:
        alpha = even_coverage_alpha(alpha, ce, img_bgr)
    if cr > 0:
        alpha = cake_reduce(alpha, cr)
    out = recompose(S, M, alpha)
    if img_bgr.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(out, 0, 255).astype(np.float32)


def apply_makeup_coverage_even(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
    mole_mask: Optional[np.ndarray] = None,
    cake_reduce_strength: float = 0.0,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Engine-facing: full unmix path when strength>0."""
    return apply_makeup_unmix(
        img_bgr,
        skin_mask,
        coverage_even=float(strength or 0.0),
        cake_reduce_strength=float(cake_reduce_strength or 0.0),
        mole_mask=mole_mask,
        exclude_mask=exclude_mask,
    )
