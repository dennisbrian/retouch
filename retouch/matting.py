"""Automatic-trimap alpha matting (Z3).

Two layers live here:

* ``build_auto_trimap`` / ``refine_alpha_matte`` — the original conservative
  spike.  Guided-filter band refinement with a confidence gate; always safe,
  never solves a linear system.
* ``solve_closed_form_alpha`` / ``estimate_foreground`` — Z3 phase 2.  Levin
  et al. (PAMI 2008) closed-form matting solved on the unknown band only, plus
  foreground-colour estimation.

**Why foreground estimation is not optional.** An observed edge pixel already
is ``src = a*F + (1-a)*B_original``.  Compositing ``a*src + (1-a)*new_bg``
therefore double-counts the *old* background, leaving a residual of
``a*(1-a)*(B_original - new_bg)`` — the classic colour fringe.  Measured on
synthetic ground truth: with a perfect matte but no F-estimate the composite
error is 4.295; with true F it is 0.000.  A better alpha alone buys nothing.

SciPy is an optional dependency: the sparse CG solve is soft-imported, and
callers fall back to the guided-filter refinement when it is unavailable.
See ``docs/plans/PLAN_Z3_ALPHA_MATTING.md`` §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import guided_filter

try:  # pragma: no cover - exercised via _HAVE_SCIPY in tests
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _sp = None
    _spla = None
    _HAVE_SCIPY = False


@dataclass(frozen=True)
class MatteResult:
    alpha: np.ndarray
    trimap: np.ndarray
    confidence: float
    used_fallback: bool


def build_auto_trimap(
    foreground_mask: np.ndarray,
    *,
    hair_mask: Optional[np.ndarray] = None,
    band_radius: int = 4,
) -> np.ndarray:
    """Create a 0/128/255 trimap from a soft foreground and optional hair mask."""
    fg = _mask(foreground_mask)
    if band_radius < 1:
        raise ValueError("band_radius must be >= 1")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_radius * 2 + 1, band_radius * 2 + 1))
    core = cv2.erode((fg > 0.80).astype(np.uint8), kernel) > 0
    background = cv2.erode((fg < 0.08).astype(np.uint8), kernel) > 0
    trimap = np.full(fg.shape, 128, dtype=np.uint8)
    trimap[background] = 0
    trimap[core] = 255
    if hair_mask is not None:
        hair = _mask(hair_mask)
        if hair.shape != fg.shape:
            raise ValueError("hair_mask must match foreground_mask")
        hair_band = cv2.dilate((hair > 0.05).astype(np.uint8), kernel) > 0
        trimap[hair_band & ~core & ~background] = 128
    return trimap


def refine_alpha_matte(
    img_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    hair_mask: Optional[np.ndarray] = None,
    band_radius: int = 4,
) -> MatteResult:
    """Refine only an automatic trimap's unknown band with guided alpha filtering.

    The closed-form sparse solver belongs behind this spike's synthetic halo
    gate.  This conservative implementation establishes trimap contracts,
    keeps known foreground/background exact, and fails back to the input mask
    on an unusable unknown band.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 BGR image")
    fg = _mask(foreground_mask)
    if fg.shape != img_bgr.shape[:2]:
        raise ValueError("foreground_mask must match img_bgr")
    trimap = build_auto_trimap(fg, hair_mask=hair_mask, band_radius=band_radius)
    unknown = trimap == 128
    unknown_fraction = float(np.mean(unknown))
    if unknown_fraction < 0.002 or unknown_fraction > 0.45:
        return MatteResult(fg.copy(), trimap, 0.0, True)

    gray = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    try:
        alpha = guided_filter(fg.astype(np.float32), radius=max(2, band_radius * 2), eps=9.0, guide=gray)
    except (cv2.error, ValueError):
        return MatteResult(fg.copy(), trimap, 0.0, True)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    alpha[trimap == 0] = 0.0
    alpha[trimap == 255] = 1.0
    confidence = float(np.clip(1.0 - abs(unknown_fraction - 0.08) / 0.25, 0.0, 1.0))
    return MatteResult(alpha, trimap, confidence, False)


def _mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if not np.isfinite(arr).all():
        raise ValueError("mask must contain finite values")
    arr = arr.astype(np.float32)
    if arr.size and float(arr.max()) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


# ----------------------------------------------------------------------
# Z3 phase 2 — closed-form matting + foreground estimation
# ----------------------------------------------------------------------


def _matting_laplacian(img: np.ndarray, eps: float = 1e-7, win_rad: int = 1):
    """Build Levin et al.'s matting Laplacian for a small image.

    Uses the standard closed-form derivation: for every 3x3 window, the local
    affine model of alpha on colour contributes an outer-product block.  The
    result is a sparse (N, N) matrix with N = H*W.

    Args:
        img: (H, W, 3) float32 BGR in [0, 1].
        eps: Regularization on the per-window colour covariance.
        win_rad: Window radius (1 => 3x3 windows).

    Returns:
        scipy.sparse.csr_matrix of shape (N, N).
    """
    h, w = img.shape[:2]
    win_size = (win_rad * 2 + 1) ** 2
    n = h * w

    # Index image so each window can gather its pixel ids.
    idx = np.arange(n, dtype=np.int64).reshape(h, w)
    win_h, win_w = h - 2 * win_rad, w - 2 * win_rad

    # (win_h, win_w, win_size) neighbour indices.
    win_idx = np.zeros((win_h, win_w, win_size), dtype=np.int64)
    k = 0
    for dy in range(win_rad * 2 + 1):
        for dx in range(win_rad * 2 + 1):
            win_idx[:, :, k] = idx[dy:dy + win_h, dx:dx + win_w]
            k += 1

    # Gather window colours: (win_h, win_w, win_size, 3)
    flat = img.reshape(n, 3)
    win_c = flat[win_idx]

    mean = win_c.mean(axis=2, keepdims=True)
    centered = win_c - mean
    # Per-window covariance (win_h, win_w, 3, 3)
    cov = np.einsum("ijkl,ijkm->ijlm", centered, centered) / win_size
    cov += (eps / win_size) * np.eye(3, dtype=np.float64)

    inv = np.linalg.inv(cov)
    # Levin's affine kernel: (1 + centered^T inv centered) / win_size
    tmp = np.einsum("ijkl,ijlm->ijkm", centered, inv)
    vals = np.einsum("ijkl,ijml->ijkm", tmp, centered)
    vals = (1.0 + vals) / win_size

    # Off-diagonal contributions: L = D - W, assembled as -vals then diag fix.
    row = np.repeat(win_idx[:, :, :, None], win_size, axis=3).ravel()
    col = np.repeat(win_idx[:, :, None, :], win_size, axis=2).ravel()
    data = -vals.ravel()

    lap = _sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    # Row sums must be zero: put the compensating mass on the diagonal.
    diag = np.asarray(lap.sum(axis=1)).ravel()
    lap = lap - _sp.diags(diag)
    return lap.tocsr()


def solve_closed_form_alpha(
    img_bgr: np.ndarray,
    trimap: np.ndarray,
    *,
    lambda_known: float = 100.0,
    max_dim: int = 320,
    eps: float = 1e-7,
) -> Optional[np.ndarray]:
    """Solve alpha in the trimap's unknown band via closed-form matting.

    The Laplacian is assembled and solved at a proxy resolution (``max_dim``),
    then upsampled with the guided filter using the full-res image as guide —
    the standard band-limited-solve / edge-aware-upsample split.  Background
    ops run at *native* resolution under ``quality="full"``, so solving at
    native size is not affordable; see PLAN_Z3_ALPHA_MATTING.md §4.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR [0, 255].
        trimap: (H, W) uint8 with 0 = background, 255 = foreground, 128 = unknown.
        lambda_known: Weight pinning known trimap pixels to their value.
        max_dim: Longest proxy edge for the solve.
        eps: Laplacian colour-covariance regularization.

    Returns:
        (H, W) float32 alpha in [0, 1], or ``None`` when SciPy is unavailable
        or the solve is not applicable (caller should fall back).
    """
    if not _HAVE_SCIPY:
        return None
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 BGR image")
    if trimap.shape != img_bgr.shape[:2]:
        raise ValueError("trimap must match img_bgr")

    h, w = img_bgr.shape[:2]
    unknown_full = trimap == 128
    if not unknown_full.any():
        return (trimap == 255).astype(np.float32)

    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        ph, pw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
        small = cv2.resize(img_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        # Nearest keeps the three trimap levels crisp.
        small_tri = cv2.resize(trimap, (pw, ph), interpolation=cv2.INTER_NEAREST)
    else:
        ph, pw = h, w
        small, small_tri = img_bgr, trimap

    small_f = np.clip(small.astype(np.float32), 0, 255) / 255.0
    unknown = small_tri == 128
    if not unknown.any():
        alpha_small = (small_tri == 255).astype(np.float32)
    else:
        n = ph * pw
        lap = _matting_laplacian(small_f, eps=eps)
        known = (~unknown).ravel().astype(np.float64)
        vals = (small_tri == 255).ravel().astype(np.float64)
        d = _sp.diags(lambda_known * known)
        a_mat = (lap + d).tocsr()
        b_vec = lambda_known * known * vals
        # CG is enough: the system is SPD and well conditioned by lambda.
        sol, _info = _spla.cg(a_mat, b_vec, rtol=1e-6, maxiter=2000)
        alpha_small = np.clip(sol.reshape(ph, pw), 0.0, 1.0).astype(np.float32)

    if scale < 1.0:
        alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(
            np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        try:
            alpha = guided_filter(alpha.astype(np.float32), radius=4, eps=1e-4, guide=gray)
        except (cv2.error, ValueError):
            pass
    else:
        alpha = alpha_small

    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    # Known regions stay exact regardless of upsampling blur.
    alpha[trimap == 0] = 0.0
    alpha[trimap == 255] = 1.0
    return alpha


def estimate_foreground(
    img_bgr: np.ndarray,
    alpha: np.ndarray,
    *,
    iterations: int = 12,
) -> np.ndarray:
    """Estimate the unmixed foreground colour F at partially transparent pixels.

    An observed pixel is ``src = a*F + (1-a)*B``.  Compositing ``a*src`` over a
    new background therefore drags the *old* background along, producing the
    colour fringe that makes wisps read as "cut out".  This recovers F so the
    composite is ``a*F + (1-a)*new_bg``.

    Uses alpha-weighted push-pull diffusion: confident foreground
    (``a`` near 1) is trusted as-is and propagated inward to the translucent
    band, which is the cheap, stable half of Levin's F/B estimation and needs
    no linear solve.

    Args:
        img_bgr: (H, W, 3) float32 BGR [0, 255].
        alpha: (H, W) float32 alpha in [0, 1].
        iterations: Diffusion passes.

    Returns:
        (H, W, 3) float32 estimated foreground colour, [0, 255].
    """
    src = np.clip(img_bgr.astype(np.float32), 0.0, 255.0)
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    # Trust only near-opaque pixels; they are essentially pure F.
    conf = np.clip((a - 0.85) / 0.15, 0.0, 1.0).astype(np.float32)
    if float(conf.max()) <= 1e-6:
        return src  # nothing opaque to learn from

    fg = src * conf[:, :, np.newaxis]
    wgt = conf.copy()
    for _ in range(max(1, iterations)):
        fg = cv2.GaussianBlur(fg, (5, 5), 1.2)
        wgt = cv2.GaussianBlur(wgt, (5, 5), 1.2)
        safe = wgt > 1e-5
        est = np.where(
            safe[:, :, np.newaxis],
            fg / np.maximum(wgt, 1e-5)[:, :, np.newaxis],
            0.0,
        ).astype(np.float32)
        keep = conf[:, :, np.newaxis]
        fg = src * keep + est * (1.0 - keep)
        wgt = np.maximum(conf, safe.astype(np.float32))

    return np.clip(fg, 0.0, 255.0).astype(np.float32)
