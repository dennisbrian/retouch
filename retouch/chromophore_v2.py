"""Constrained chromophore decomposition in linear-light optical density.

This module is deliberately a *core*, not a promise of literal pigment
concentrations from arbitrary camera RGB.  Its two maps are relative optical
density coordinates: a melanin-like brown/blue absorber and a hemoglobin-like
redness absorber.  They are useful for controlled edits when a skin mask is
available, while ``confidence`` and ``used_fallback`` tell a caller when the
image did not contain enough colour variation for a per-image refinement.

Input contract
--------------
Functions accept BGR ``uint8`` in ``[0, 255]`` or BGR floating point arrays
in either ``[0, 1]`` or ``[0, 255]``.  Reconstruction and edits return BGR
``float32`` in ``[0, 255]``.  ``strength == 0`` in
``reduce_hemoglobin_variance`` returns the original input object unchanged.

Method
------
The encoded sRGB input is linearised before computing optical density.  The
neutral RGB direction is explicitly projected out, making the chromatic
coordinates invariant to multiplicative neutral shading.  A small constrained
ICA-style kurtosis search may rotate the two broad pigment priors by at most
12 degrees, so a global illuminant offset does not choose the axes.  This does
not solve arbitrary mixed illumination, specular reflection, makeup, or camera
metamerism; CAT16 white balance should run before this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


_EPS = np.float32(1e-5)
_NEUTRAL = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)
# Orthonormal basis of the plane perpendicular to neutral optical density.
_CHROMA_BASIS = np.array(
    [[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
     [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
     [0.0, -2.0 / np.sqrt(6.0)]],
    dtype=np.float32,
)

# Broad RGB optical-density anchors.  Melanin rises towards blue; hemoglobin
# has its strongest visible absorption in green/blue relative to red.  They
# are chromatic anchors, not a calibrated spectral database.
MELANIN_PRIOR_RGB = np.array([0.14, 0.28, 0.52], dtype=np.float32)
HEMOGLOBIN_PRIOR_RGB = np.array([0.08, 0.90, 0.60], dtype=np.float32)
_MAX_AXIS_ROTATION_RAD = np.deg2rad(12.0)
_MIN_AXIS_SEPARATION_RAD = np.deg2rad(20.0)
_MIN_SAMPLES = 128

# AA5: a full physiological oxygenation swing is ΔE 2.4.  Keep each
# hemoglobin edit inside that perceptual budget rather than limiting it with
# an RGB or luminance threshold.
MAX_HEMOGLOBIN_DELTA_E = 2.4


@dataclass(frozen=True)
class ChromophoreV2Decomposition:
    """A reversible relative-pigment decomposition.

    ``melanin`` and ``hemoglobin`` are signed coordinates around the image
    baseline, rather than absolute concentrations.  ``shading`` is the neutral
    optical-density component.  ``axes_rgb`` has RGB optical-density axes as
    columns.  A caller can retain it to decompose an edited image in the same
    coordinate system for QA.
    """

    melanin: np.ndarray
    hemoglobin: np.ndarray
    shading: np.ndarray
    axes_rgb: np.ndarray
    confidence: float
    used_fallback: bool


def _validate_bgr(img_bgr: np.ndarray) -> None:
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 BGR numpy array")
    if not np.issubdtype(img_bgr.dtype, np.number):
        raise ValueError("img_bgr must have a numeric dtype")
    if not np.isfinite(img_bgr).all():
        raise ValueError("img_bgr must contain only finite values")


def _bgr_to_255(img_bgr: np.ndarray) -> np.ndarray:
    """Normalise supported BGR inputs to finite float32 [0, 255]."""
    _validate_bgr(img_bgr)
    arr = img_bgr.astype(np.float32, copy=False)
    if np.issubdtype(img_bgr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.float32, copy=False)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _normalise_mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Return a float32 alpha mask; ``None`` means the full image."""
    h, w = shape
    if mask is None:
        return np.ones((h, w), dtype=np.float32)
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.shape != (h, w):
        raise ValueError(f"skin_mask must have shape {(h, w)}, got {m.shape}")
    if not np.issubdtype(m.dtype, np.number) or not np.isfinite(m).all():
        raise ValueError("skin_mask must be finite and numeric")
    m = m.astype(np.float32)
    if m.size and float(m.max()) > 1.0:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def _bgr255_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert float BGR [0, 255] to CIELab float32 (L* 0..100)."""
    rgb = np.clip(img_bgr[..., ::-1] / 255.0, 0.0, 1.0).astype(np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def _lab_to_bgr255(lab: np.ndarray) -> np.ndarray:
    """Convert CIELab float32 (L* 0..100) to float BGR [0, 255]."""
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(rgb[..., ::-1] * 255.0, 0.0, 255.0).astype(np.float32)


def delta_e_76(
    reference_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    *,
    skin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return per-pixel CIE76 ΔE, optionally zeroed outside the skin mask."""
    reference = _bgr_to_255(reference_bgr)
    edited = _bgr_to_255(edited_bgr)
    if reference.shape != edited.shape:
        raise ValueError("reference_bgr and edited_bgr must have the same shape")
    delta = _bgr255_to_lab(edited) - _bgr255_to_lab(reference)
    out = np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32)
    if skin_mask is not None:
        out *= _normalise_mask(skin_mask, out.shape)
    return out


def cap_delta_e_76(
    reference_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    *,
    skin_mask: Optional[np.ndarray] = None,
    max_delta_e: float = MAX_HEMOGLOBIN_DELTA_E,
) -> np.ndarray:
    """Clamp an edit to a per-pixel CIE76 ΔE budget in Lab space.

    Interpolation happens in Lab, so the cap is exact in the metric being
    promised. Pixels outside ``skin_mask`` remain byte-identical to the
    reference float255 representation.
    """
    if not np.isfinite(max_delta_e) or max_delta_e <= 0.0:
        raise ValueError("max_delta_e must be finite and positive")
    reference = _bgr_to_255(reference_bgr)
    edited = _bgr_to_255(edited_bgr)
    if reference.shape != edited.shape:
        raise ValueError("reference_bgr and edited_bgr must have the same shape")
    alpha = _normalise_mask(skin_mask, reference.shape[:2])
    ref_lab = _bgr255_to_lab(reference)
    edited_lab = _bgr255_to_lab(edited)
    delta = edited_lab - ref_lab
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    scale = np.minimum(1.0, float(max_delta_e) / np.maximum(distance, _EPS)) * alpha
    capped = ref_lab + delta * scale[:, :, None]
    result = _lab_to_bgr255(capped)
    return (reference * (1.0 - alpha[:, :, None]) + result * alpha[:, :, None]).astype(np.float32)


def _bounded_hemoglobin_edit(
    bgr255: np.ndarray,
    decomposition: ChromophoreV2Decomposition,
    proposed_hemoglobin: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Apply a hemoglobin-only edit while enforcing AA5's ΔE cap.

    The search scales the hemoglobin-coordinate delta before recomposition,
    rather than blending in Lab. This preserves the decomposition's melanin
    coordinate exactly (up to the existing float reconstruction tolerance).
    """
    change = proposed_hemoglobin.astype(np.float32) - decomposition.hemoglobin

    def compose(scale: float) -> np.ndarray:
        edited = recompose_chromophores_v2(
            decomposition,
            hemoglobin=decomposition.hemoglobin + np.float32(scale) * change,
        )
        return (bgr255 * (1.0 - alpha[..., None]) + edited * alpha[..., None]).astype(np.float32)

    full = compose(1.0)
    if float(delta_e_76(bgr255, full, skin_mask=alpha).max()) <= MAX_HEMOGLOBIN_DELTA_E:
        return full

    lo, hi = 0.0, 1.0
    for _ in range(16):
        mid = (lo + hi) * 0.5
        candidate = compose(mid)
        if float(delta_e_76(bgr255, candidate, skin_mask=alpha).max()) <= MAX_HEMOGLOBIN_DELTA_E:
            lo = mid
        else:
            hi = mid
    return compose(lo)


def _project_chromatic(od_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split optical density into neutral shading and its 2D chromatic plane."""
    shading = np.tensordot(od_rgb, _NEUTRAL, axes=([-1], [0])).astype(np.float32)
    chroma = od_rgb - shading[..., None] * _NEUTRAL
    return shading, chroma.astype(np.float32)


def _prior_axis(prior: np.ndarray) -> np.ndarray:
    projected = prior - np.dot(prior, _NEUTRAL) * _NEUTRAL
    norm = float(np.linalg.norm(projected))
    if norm < 1e-8:
        raise RuntimeError("pigment prior unexpectedly lies on neutral axis")
    return (projected / norm).astype(np.float32)


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _constrained_ica_axis(samples_2d: np.ndarray, prior_2d: np.ndarray) -> np.ndarray:
    """Choose the most non-Gaussian axis in a narrow, prior-centred arc.

    This is intentionally constrained: it only refines the colour direction
    supported by a particular image; it never relabels makeup/background as a
    new pigment axis.
    """
    centered = samples_2d - np.mean(samples_2d, axis=0, keepdims=True)
    theta0 = float(np.arctan2(prior_2d[1], prior_2d[0]))
    offsets = np.linspace(-_MAX_AXIS_ROTATION_RAD, _MAX_AXIS_ROTATION_RAD, 25)
    angles = theta0 + offsets
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    projections = centered @ dirs.T
    variance = np.mean(projections * projections, axis=0)
    fourth = np.mean(projections**4, axis=0)
    excess_kurtosis = fourth / np.maximum(variance * variance, 1e-12) - 3.0
    # Prefer a close prior on ties: the small term prevents noise chasing.
    score = np.abs(excess_kurtosis) - 0.02 * np.abs(offsets / _MAX_AXIS_ROTATION_RAD)
    return dirs[int(np.argmax(score))]


def _estimate_axes(chroma: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, float, bool]:
    mel_prior = _prior_axis(MELANIN_PRIOR_RGB)
    hb_prior = _prior_axis(HEMOGLOBIN_PRIOR_RGB)
    fallback = np.stack([mel_prior, hb_prior], axis=1).astype(np.float32)
    samples = chroma[active > 0.5]
    if samples.shape[0] < _MIN_SAMPLES:
        return fallback, 0.0, True

    samples_2d = samples @ _CHROMA_BASIS
    covariance = np.cov(samples_2d, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return fallback, 0.0, True
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(eigenvalues[0]) < 1e-8:
        return fallback, 0.0, True

    mel_2d = mel_prior @ _CHROMA_BASIS
    hb_2d = hb_prior @ _CHROMA_BASIS
    mel_dir = _constrained_ica_axis(samples_2d, mel_2d)
    hb_dir = _constrained_ica_axis(samples_2d, hb_2d)
    separation = abs(float(_wrap_angle(
        np.arctan2(mel_dir[1], mel_dir[0]) - np.arctan2(hb_dir[1], hb_dir[0])
    )))
    if separation < _MIN_AXIS_SEPARATION_RAD:
        return fallback, 0.0, True

    axes = (_CHROMA_BASIS @ np.stack([mel_dir, hb_dir], axis=1)).astype(np.float32)
    # The smaller variance eigenvalue is the amount of independently usable
    # chromatic evidence.  Saturate it into an interpretable 0..1 confidence.
    confidence = float(np.clip(np.sqrt(float(eigenvalues[0])) / 0.03, 0.0, 1.0))
    return axes, confidence, False


def _validate_axes(axes_rgb: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes_rgb, dtype=np.float32)
    if axes.shape != (3, 2) or not np.isfinite(axes).all():
        raise ValueError("axes_rgb must be a finite (3, 2) RGB optical-density matrix")
    # Remove neutral leakage caused by serialized/rounded caller arrays.
    axes = axes - _NEUTRAL[:, None] * (_NEUTRAL @ axes)[None, :]
    if np.linalg.matrix_rank(axes) != 2:
        raise ValueError("axes_rgb must contain two independent chromatic axes")
    return axes.astype(np.float32)


def decompose_chromophores_v2(
    img_bgr: np.ndarray,
    *,
    skin_mask: Optional[np.ndarray] = None,
    axes_rgb: Optional[np.ndarray] = None,
) -> ChromophoreV2Decomposition:
    """Decompose BGR image data into relative melanin and hemoglobin maps.

    ``skin_mask`` is used to estimate axes; it does not erase pixels from the
    returned maps.  Supplying ``axes_rgb`` skips per-image estimation, useful
    for before/after QA in a fixed coordinate system.
    """
    bgr255 = _bgr_to_255(img_bgr)
    active = _normalise_mask(skin_mask, bgr255.shape[:2])
    rgb_linear = _srgb_to_linear(bgr255[..., ::-1] / 255.0)
    od_rgb = -np.log(np.clip(rgb_linear, _EPS, 1.0)).astype(np.float32)
    shading, chroma = _project_chromatic(od_rgb)

    if axes_rgb is None:
        axes, confidence, fallback = _estimate_axes(chroma, active)
    else:
        axes = _validate_axes(axes_rgb)
        confidence, fallback = 1.0, False

    coordinates = chroma @ np.linalg.pinv(axes).T
    return ChromophoreV2Decomposition(
        melanin=coordinates[..., 0].astype(np.float32),
        hemoglobin=coordinates[..., 1].astype(np.float32),
        shading=shading.astype(np.float32),
        axes_rgb=axes.astype(np.float32),
        confidence=confidence,
        used_fallback=fallback,
    )


def recompose_chromophores_v2(
    decomposition: ChromophoreV2Decomposition,
    *,
    melanin: Optional[np.ndarray] = None,
    hemoglobin: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Recompose a decomposition to BGR float32 in ``[0, 255]``.

    The supplied maps must match the decomposition shape.  Omitting both gives
    a near-exact float round trip; any residual is solely sRGB float precision.
    """
    base_mel = decomposition.melanin
    base_hb = decomposition.hemoglobin
    mel = base_mel if melanin is None else np.asarray(melanin, dtype=np.float32)
    hb = base_hb if hemoglobin is None else np.asarray(hemoglobin, dtype=np.float32)
    if mel.shape != base_mel.shape or hb.shape != base_hb.shape:
        raise ValueError("melanin and hemoglobin maps must match decomposition shape")
    if not np.isfinite(mel).all() or not np.isfinite(hb).all():
        raise ValueError("melanin and hemoglobin maps must be finite")
    axes = _validate_axes(decomposition.axes_rgb)
    od = (
        decomposition.shading[..., None] * _NEUTRAL
        + np.stack([mel, hb], axis=-1) @ axes.T
    )
    rgb_linear = np.exp(-np.clip(od, 0.0, 20.0))
    bgr = _linear_to_srgb(rgb_linear)[..., ::-1] * 255.0
    return np.clip(bgr, 0.0, 255.0).astype(np.float32)


def reduce_hemoglobin_variance(
    img_bgr: np.ndarray,
    strength: float,
    *,
    skin_mask: Optional[np.ndarray] = None,
    decomposition: Optional[ChromophoreV2Decomposition] = None,
) -> np.ndarray:
    """Reduce hemoglobin-coordinate variance without changing melanin.

    The edit pulls hemoglobin toward its masked median in the fixed axes of
    ``decomposition``.  It intentionally does not attempt to infer whether a
    red feature is medical, makeup, or a desired blush.  Callers must therefore
    provide a semantically appropriate skin mask and conservative strength.
    """
    if not np.isfinite(strength) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    if float(strength) == 0.0:
        return img_bgr

    bgr255 = _bgr_to_255(img_bgr)
    alpha = _normalise_mask(skin_mask, bgr255.shape[:2])
    if not np.any(alpha > 0.0):
        return img_bgr
    dec = decomposition or decompose_chromophores_v2(img_bgr, skin_mask=skin_mask)
    if dec.hemoglobin.shape != alpha.shape:
        raise ValueError("decomposition shape must match img_bgr")

    selected = dec.hemoglobin[alpha > 0.5]
    if selected.size == 0:
        return img_bgr
    target = np.float32(np.median(selected))
    hb_new = dec.hemoglobin + np.float32(strength) * alpha * (target - dec.hemoglobin)
    edited = recompose_chromophores_v2(dec, hemoglobin=hb_new)
    # Preserve unselected pixels exactly in the documented float255 contract.
    return _bounded_hemoglobin_edit(bgr255, dec, hb_new, alpha)


def shift_hemoglobin(
    img_bgr: np.ndarray,
    shift: float,
    *,
    skin_mask: Optional[np.ndarray] = None,
    decomposition: Optional[ChromophoreV2Decomposition] = None,
) -> np.ndarray:
    """Shift the masked hemoglobin coordinate by a face-relative amount.

    ``shift`` is in ``[-1, 1]``.  Positive values add flush and negative
    values reduce it.  The magnitude is derived from the subject's own
    interquartile hemoglobin span, rather than an absolute RGB or skin-tone
    threshold.  Melanin is deliberately left untouched.
    """
    if not np.isfinite(shift) or not -1.0 <= float(shift) <= 1.0:
        raise ValueError("shift must be finite and in [-1, 1]")
    if float(shift) == 0.0:
        return img_bgr

    bgr255 = _bgr_to_255(img_bgr)
    alpha = _normalise_mask(skin_mask, bgr255.shape[:2])
    if not np.any(alpha > 0.0):
        return img_bgr
    dec = decomposition or decompose_chromophores_v2(img_bgr, skin_mask=skin_mask)
    if dec.hemoglobin.shape != alpha.shape:
        raise ValueError("decomposition shape must match img_bgr")

    selected = dec.hemoglobin[alpha > 0.5]
    if selected.size == 0:
        return img_bgr
    q25, q75 = np.percentile(selected, (25.0, 75.0))
    # A full-scale slider move is intentionally conservative: half the
    # subject's robust spread, then feathered at the skin boundary.
    delta = np.float32(float(shift) * 0.5 * float(q75 - q25))
    hb_new = dec.hemoglobin + alpha * delta
    edited = recompose_chromophores_v2(dec, hemoglobin=hb_new)
    return _bounded_hemoglobin_edit(bgr255, dec, hb_new, alpha)


__all__ = [
    "ChromophoreV2Decomposition",
    "HEMOGLOBIN_PRIOR_RGB",
    "MELANIN_PRIOR_RGB",
    "MAX_HEMOGLOBIN_DELTA_E",
    "cap_delta_e_76",
    "decompose_chromophores_v2",
    "delta_e_76",
    "recompose_chromophores_v2",
    "reduce_hemoglobin_variance",
    "shift_hemoglobin",
]
