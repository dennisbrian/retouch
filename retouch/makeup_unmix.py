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
from .specular import extract_specular


# The classical cues can only identify local deviations from this face's skin
# baseline.  They are deliberately not a full-face foundation detector.
# At 0.2% a handful of false-positive pixels still seeded a visibly wrong M
# colour on real cosplay photos.  Track A is intentionally conservative: a
# user can supply a brush mask for smaller corrections.
_MIN_COMPONENT_FRACTION = 0.008
_MAX_COMPONENT_FRACTION = 0.12
_MAX_COMPONENT_BBOX_FRACTION = 0.20
_MAX_AUTOMATIC_EDIT_DELTA = 5.0


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
    if m.max() > 1.5:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def _skin_median_mad(values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Return robust per-face statistics, with a noise floor for flat patches."""
    sample = values[mask > 0.5]
    if not sample.size:
        sample = values.reshape(-1)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    return median, max(1.4826 * mad, 1e-4)


def _high_outlier_cue(
    values: np.ndarray, mask: np.ndarray, *, start_sigma: float = 3.0,
) -> np.ndarray:
    """Softly detect values that exceed this subject's normal skin variation."""
    median, sigma = _skin_median_mad(values, mask)
    return _smoothstep(
        median + start_sigma * sigma,
        median + (start_sigma + 3.0) * sigma,
        values,
    )


def _low_outlier_cue(
    values: np.ndarray, mask: np.ndarray, *, start_sigma: float = 2.0,
) -> np.ndarray:
    """Softly detect values below this subject's normal skin variation."""
    median, sigma = _skin_median_mad(values, mask)
    return 1.0 - _smoothstep(
        median - (start_sigma + 3.0) * sigma,
        median - start_sigma * sigma,
        values,
    )


def _merge_masks(*masks: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Combine optional exclusion masks in normalized float form."""
    present = [m for m in masks if m is not None]
    if not present:
        return None
    return np.maximum.reduce(present).astype(np.float32)


def _local_residual(values: np.ndarray, face_width: int) -> np.ndarray:
    """Remove face-scale lighting before measuring compact paint deviations."""
    sigma = max(2.0, 0.15 * face_width)
    baseline = cv2.GaussianBlur(values.astype(np.float32), (0, 0), sigmaX=sigma)
    return values.astype(np.float32) - baseline


def _dilated_exclusion(mask: Optional[np.ndarray], face_width: int) -> Optional[np.ndarray]:
    """Keep eye rims and other protected regions out of automatic alpha fitting."""
    if mask is None:
        return None
    size = max(5, int(round(face_width * 0.06)))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate((mask > 0.01).astype(np.uint8), kernel).astype(np.float32)


def _compact_components(hint: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """Retain only local regions that are safe to seed as makeup colour.

    A broad face-sized region is fundamentally ambiguous for this model: it
    cannot distinguish foundation from the face's baseline.  Bounding-box and
    area gates reject it even when its high-pass edge is locally conspicuous.
    """
    skin_area = max(1, int(np.count_nonzero(skin_mask > 0.5)))
    min_area = max(1, int(np.ceil(skin_area * _MIN_COMPONENT_FRACTION)))
    max_area = max(min_area, int(np.floor(skin_area * _MAX_COMPONENT_FRACTION)))
    max_bbox_area = max(min_area, int(np.floor(skin_area * _MAX_COMPONENT_BBOX_FRACTION)))
    binary = (hint > 0.35).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    keep = np.zeros_like(binary, dtype=np.float32)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox_area = int(stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT])
        if min_area <= area <= max_area and bbox_area <= max_bbox_area:
            keep[labels == label] = 1.0
    # The high-pass response has a one-pixel halo around sharp edges.  Retain
    # the component interior so coverage smoothing cannot edit bare skin next
    # to the local artifact.
    keep = cv2.erode(keep, np.ones((3, 3), dtype=np.uint8)).astype(np.float32)
    return hint * keep


def _compact_specular_exclusion(
    img_bgr: np.ndarray, skin_mask: np.ndarray,
) -> np.ndarray:
    """Exclude compact highlights without classifying broad white paint as shine."""
    spec = extract_specular(img_bgr, skin_mask=skin_mask)
    binary = (spec >= 20.0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    max_area = max(1, int(np.count_nonzero(skin_mask > 0.5) * 0.10))
    compact = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= max_area:
            compact[labels == label] = 1
    # Include the feathered edge surrounding a compact highlight so alpha
    # smoothing cannot pull its near-white color into adjacent bare skin.
    return cv2.dilate(compact, np.ones((21, 21), dtype=np.uint8)).astype(np.float32)


def estimate_makeup_alpha(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    mole_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    specular_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a tone-relative soft alpha prior for makeup coverage.

    Every cue is measured against robust statistics from the current face's
    skin pixels. This avoids treating a fixed reflectance or chroma value as
    makeup merely because the subject has darker or lighter skin.
    """
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
    # Makeup density is multiplicative in reflectance. Measuring its residual
    # in optical-density space keeps an equal physical change comparable on
    # light and dark skin instead of shrinking with the base RGB intensity.
    density = -np.log(np.clip(work.astype(np.float32) / 255.0, 1e-3, 1.0))
    recon_density = -np.log(np.clip(recon.astype(np.float32) / 255.0, 1e-3, 1.0))
    resid = np.abs(density - recon_density).mean(axis=-1)
    # Foundation coverage is broad while sensor noise is not. Smooth only the
    # detection signal so a low-reflectance face does not need an unrealistically
    # large density jump to clear a per-pixel noise threshold.
    density_l = cv2.GaussianBlur(density.mean(axis=-1), (0, 0), sigmaX=2.0)

    # Use local deviations instead of face-wide values.  This intentionally
    # discards full-face foundation signals while restoring compact artifacts
    # under normal facial shading and low-frequency skin variation.
    face_width = max(1, int(np.sqrt(np.count_nonzero(m > 0.5))))
    cue_chroma = _high_outlier_cue(np.abs(_local_residual(C, face_width)), m)
    cue_light = _high_outlier_cue(np.abs(_local_residual(L, face_width)), m)
    cue_resid = _high_outlier_cue(np.abs(_local_residual(resid, face_width)), m)
    cue_density = _high_outlier_cue(
        np.abs(_local_residual(density_l, face_width)), m,
    )

    hint = np.maximum.reduce(
        (cue_chroma, cue_light, cue_resid, cue_density)
    ) * m
    mm = _prep_mask(mole_mask, h, w)
    if mm is not None:
        hint = hint * (1.0 - mm)
    ex = _dilated_exclusion(_merge_masks(
        _prep_mask(exclude_mask, h, w),
        _prep_mask(specular_mask, h, w),
    ), face_width)
    if ex is not None:
        hint = hint * (1.0 - ex)

    # Opening removes isolated noise hits. The former closing operation grew
    # those hits into paint regions, which made compact shine bleed outward
    # when coverage-even blurred the resulting alpha map.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u8 = np.clip(hint * 255, 0, 255).astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k)
    return _compact_components(u8.astype(np.float32) / 255.0, m).astype(np.float32)


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
    specular_mask: Optional[np.ndarray] = None,
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

    if specular_mask is None and user_alpha is None:
        # A highlight can seed a near-white makeup color and then bleed that
        # color across bare skin during coverage evening. The specular module
        # already uses a face-relative brightness gate, so reuse it here.
        specular_mask = _compact_specular_exclusion(I, base)

    combined_exclude = _merge_masks(
        _prep_mask(exclude_mask, h, w),
        _prep_mask(specular_mask, h, w),
    )

    if user_alpha is not None:
        alpha = _prep_mask(user_alpha, h, w)
        assert alpha is not None
    else:
        alpha = estimate_makeup_alpha(
            img_bgr,
            skin_mask,
            mole_mask=mole_mask,
            exclude_mask=combined_exclude,
        )

    # Automatic alpha may only affect the compact component that cleared the
    # safety gate.  User-supplied alpha remains authoritative.
    auto_support = None if user_alpha is not None else (alpha > 0.0).astype(np.float32)

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

    # With no separation between estimated skin and makeup colors, projection
    # is underdetermined. Returning the masked prior avoids IRLS inventing a
    # direction that later coverage smoothing can spread across bare skin.
    if float(np.linalg.norm(Mcol - S_fill)) < 6.0:
        alpha = alpha * base
        if mole_mask is not None:
            mm = _prep_mask(mole_mask, h, w)
            if mm is not None:
                alpha *= 1.0 - mm
        if combined_exclude is not None:
            alpha *= 1.0 - combined_exclude
        return I.copy(), I.copy(), alpha.astype(np.float32)

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
        if auto_support is not None:
            alpha = alpha * auto_support
        if mole_mask is not None:
            mm = _prep_mask(mole_mask, h, w)
            if mm is not None:
                alpha = alpha * (1.0 - mm)
        if combined_exclude is not None:
            alpha = alpha * (1.0 - combined_exclude)
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
    # Never let smoothing spread an automatic local edit into adjacent bare
    # skin.  The non-zero support has already cleared the compactness gate.
    support = a > 1e-6
    return np.where(support, np.clip(a * (1.0 - s) + blur * s, 0.0, 1.0), 0.0)


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
    specular_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full product entry: unmix → edit α → recompose. All strengths 0 = identity."""
    ce = float(coverage_even or 0.0)
    cr = float(cake_reduce_strength or 0.0)
    if ce <= 0 and cr <= 0:
        return img_bgr

    S, M, alpha = unmix_makeup(
        img_bgr,
        skin_mask,
        mole_mask=mole_mask,
        exclude_mask=exclude_mask,
        specular_mask=specular_mask,
    )
    if ce > 0:
        alpha = even_coverage_alpha(alpha, ce, img_bgr)
    if cr > 0:
        alpha = cake_reduce(alpha, cr)
    out = recompose(S, M, alpha)
    # P4's automatic alpha is evidence, not a physical coverage measurement.
    # A mistaken compact component must not create a destructive visible edit.
    out = img_bgr.astype(np.float32) + np.clip(
        out - img_bgr.astype(np.float32),
        -_MAX_AUTOMATIC_EDIT_DELTA,
        _MAX_AUTOMATIC_EDIT_DELTA,
    )
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
    specular_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Engine-facing: full unmix path when strength>0."""
    return apply_makeup_unmix(
        img_bgr,
        skin_mask,
        coverage_even=float(strength or 0.0),
        cake_reduce_strength=float(cake_reduce_strength or 0.0),
        mole_mask=mole_mask,
        exclude_mask=exclude_mask,
        specular_mask=specular_mask,
    )
