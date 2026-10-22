"""User protection mask ("keep this"): one mask every retouch op honours.

The user paints over a mole, a drawn tear-mole, a freckle cluster or a
makeup detail, and the retouch leaves it alone. ``freckle_preserve_mask``
(the old, Python-only, freckle-healing-only input) is an alias: both are
merged into one mask at the start of a render.

Coordinates. The mask is stored in the pixel grid of the image handed to
``RetouchEngine.process`` (source-native). Every render path works on a
different grid (the 800 px fast preview, the 2048 px draft proxy, native
face crops, a reshaped frame), so consumers never use the stored array
directly: :func:`frame_mask` resamples it to the frame a stage runs on
and, when reshape moved the face, warps it with the same displacement
field so the mask follows the mole.

How it is honoured. Detection-style ops (spot heal, legacy blemish,
freckle healing, stray hair, highlight repair, guided smoothing) get it as
an exclusion, so they never pick a protected spot. On top of that every
protected stage window (the per-face core, the body-skin stages, lens
glare and red-eye) ends with :func:`keep_protected`, which is the hard
guarantee: inside the mask the pixels keep their own pre-window detail
and only take the *broad* tone change the window made to the unprotected
skin around them. A protected mole on a whitened face is therefore
whitened with the face (no dark island), but its own shape, contrast and
colour against the skin survive. Global stages (tone, grade, film,
finish) run after these windows and apply inside the mask as everywhere
else.

All sizes are relative (face width, frame long side), so the preview, the
draft proxy and the native export protect the same area.
"""

from __future__ import annotations

import base64
from typing import Any, Optional, Tuple

import cv2
import numpy as np

# Broad tone carried into the mask: Gaussian sigma of the fill, as a fraction
# of face width (per-face core) or of the frame's long side (frame stages).
FACE_TONE_SIGMA = 0.05
FRAME_TONE_SIGMA = 0.012
# Soft edge added outside the painted stroke, same units.
FACE_FEATHER = 0.008
FRAME_FEATHER = 0.0015
# Aspect-ratio tolerance when a mask of another size is resized to the
# source (a mask saved from a downsized preview, a --max-dim run).
ASPECT_TOLERANCE = 0.01


_DATA_URI = "data:image/png;base64,"


def encode_mask(mask: Any) -> Optional[str]:
    """A mask as a PNG data URI (8-bit), for JSON sessions and GUI state."""
    m = coerce_mask(mask)
    if m is None:
        return None
    ok, buf = cv2.imencode(".png", np.rint(m * 255.0).astype(np.uint8))
    if not ok:
        raise ValueError("could not encode the protect mask")
    return _DATA_URI + base64.b64encode(buf.tobytes()).decode("ascii")


def decode_mask(value: str) -> Optional[np.ndarray]:
    """Inverse of :func:`encode_mask` (any PNG/JPEG data URI)."""
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise ValueError("protect mask string must be an image data URI")
    raw = base64.b64decode(value.split(",", 1)[1])
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("protect mask data URI is not a readable image")
    return arr


def coerce_mask(mask: Any) -> Optional[np.ndarray]:
    """Return a float32 (H, W) mask in [0, 1], or None when empty/absent.

    Accepts bool, uint8 (0-255), uint16 and float arrays, RGB/RGBA images
    (the brightest channel, or alpha when it is not flat), a PNG data URI
    (:func:`encode_mask`) and the GUI's ``{"mask": <data URI>}`` state.
    """
    if mask is None:
        return None
    if isinstance(mask, dict):
        mask = mask.get("mask")
        if mask is None:
            return None
    if isinstance(mask, str):
        if not mask:
            return None
        mask = decode_mask(mask)
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[2] == 4 and float(np.ptp(m[..., 3])) > 0:
            m = m[..., 3]
        else:
            m = m[..., :3].max(axis=2)
    if m.ndim != 2 or m.size == 0:
        raise ValueError(f"protect mask must be (H, W), got shape {np.shape(mask)}")
    if m.dtype == bool:
        out = m.astype(np.float32)
    elif m.dtype == np.uint16:
        out = m.astype(np.float32) / 65535.0
    else:
        out = m.astype(np.float32)
        # Same epsilon-safe 0-255 test as perf_optimizations._norm_mask.
        if out.max() > 1.5:
            out /= 255.0
    out = np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)
    if not np.any(out > 1e-3):
        return None
    return out


def merge_masks(*masks: Any) -> Optional[np.ndarray]:
    """Union (max) of several masks; all non-empty ones must share a shape."""
    out = None
    for raw in masks:
        m = coerce_mask(raw)
        if m is None:
            continue
        if out is None:
            out = m
            continue
        if m.shape != out.shape:
            m = resize_mask(m, out.shape)
        out = np.maximum(out, m)
    return out


def resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resample ``mask`` to ``shape`` (H, W): area when shrinking, linear up."""
    h, w = int(shape[0]), int(shape[1])
    if mask.shape[:2] == (h, w):
        return mask
    shrink = h * w < mask.shape[0] * mask.shape[1]
    out = cv2.resize(
        mask.astype(np.float32), (w, h),
        interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR,
    )
    return np.clip(out, 0.0, 1.0)


def fit_to_source(mask: Any, source_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Coerce and resize a user mask to the source image grid.

    A mask of another size is accepted when its aspect ratio matches the
    source within :data:`ASPECT_TOLERANCE` (it was painted on a downsized
    copy); anything else is a different framing and is refused.
    """
    m = coerce_mask(mask)
    if m is None:
        return None
    sh, sw = int(source_shape[0]), int(source_shape[1])
    if m.shape != (sh, sw):
        mh, mw = m.shape
        if abs(mw / float(mh) - sw / float(sh)) > ASPECT_TOLERANCE * (sw / float(sh)):
            raise ValueError(
                f"protect mask {mw}x{mh} does not match the photo's framing "
                f"{sw}x{sh}; paint it on this photo (any size, same aspect)."
            )
        m = resize_mask(m, (sh, sw))
    return m


def warp_mask(mask: np.ndarray, displacement: Optional[np.ndarray]) -> np.ndarray:
    """Move ``mask`` with a reshape displacement field (output = grid - map)."""
    if displacement is None:
        return mask
    field = np.asarray(displacement, dtype=np.float32)
    if field.shape[:2] != mask.shape or field.ndim != 3 or not np.any(field):
        return mask
    h, w = mask.shape
    gx = np.arange(w, dtype=np.float32)[None, :]
    gy = np.arange(h, dtype=np.float32)[:, None]
    map_x = (gx - field[:, :, 0]).astype(np.float32)
    map_y = (gy - field[:, :, 1]).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def frame_mask(ctx: Any, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """The render's protection mask on a ``shape`` (H, W) frame, or None.

    Resampled from the source-grid mask and warped by the render's reshape
    field when that field lives on the same frame. Cached per frame size on
    ``ctx`` (one render owns one ctx).
    """
    source = getattr(ctx, "_protect_source", None)
    if source is None:
        return None
    h, w = int(shape[0]), int(shape[1])
    field = getattr(ctx, "_aa6_warp_field", None)
    warped = field is not None and np.shape(field)[:2] == (h, w) and bool(np.any(field))
    cache = getattr(ctx, "_protect_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            ctx._protect_cache = cache
        except AttributeError:
            pass
    key = (h, w, warped)
    if key not in cache:
        m = resize_mask(source, (h, w))
        if warped:
            m = warp_mask(m, field)
        cache[key] = m if np.any(m > 1e-3) else None
    return cache[key]


def soft_mask(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """Painted strokes keep weight 1 inside; a soft edge grows outside."""
    sigma = max(float(feather_px), 0.5)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.clip(np.maximum(mask, 2.0 * blurred), 0.0, 1.0)


def _blur_down(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, done on a reduced copy for large sigmas."""
    f = max(1.0, sigma / 6.0)
    if f <= 1.0:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape[:2]
    sw, sh = max(1, int(round(w / f))), max(1, int(round(h / f)))
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def keep_protected(
    before: np.ndarray,
    after: np.ndarray,
    protect: Optional[np.ndarray],
    tone_sigma_px: float,
    feather_px: float = 0.0,
) -> np.ndarray:
    """Undo a stage window's edits inside ``protect``, keeping its broad tone.

    Inside the (feathered) mask the output is ``before`` plus the window's
    change measured on the unprotected pixels around it and spread with a
    Gaussian of ``tone_sigma_px`` (normalised convolution). Local edits such
    as heals, smoothing and sharpening average out at that scale, so the
    protected detail comes back; broad tone such as whitening, relighting or
    face exposure carries over, so the protected spot doesn't become an
    island. Outside the soft mask ``after`` is returned untouched.

    ``before``/``after`` share shape and dtype (uint8, or float in any range).
    """
    if protect is None:
        return after
    p = protect if feather_px <= 0 else soft_mask(protect, feather_px)
    if not np.any(p > 1e-3):
        return after
    ys, xs = np.nonzero(p > 1e-3)
    pad = int(np.ceil(3.0 * max(tone_sigma_px, 1.0))) + 2
    h, w = p.shape
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)

    b = before[y1:y2, x1:x2].astype(np.float32)
    a = after[y1:y2, x1:x2].astype(np.float32)
    pw = p[y1:y2, x1:x2].astype(np.float32)
    delta = a - b
    if delta.ndim == 2:
        delta = delta[:, :, None]
    free = (1.0 - pw)
    num = _blur_down(delta * free[:, :, None], tone_sigma_px)
    if num.ndim == 2:
        num = num[:, :, None]
    den = _blur_down(free, tone_sigma_px)[:, :, None]
    if np.any(free > 1e-3):
        mean_delta = (delta * free[:, :, None]).sum(axis=(0, 1)) / max(float(free.sum()), 1e-6)
    else:
        mean_delta = np.zeros(delta.shape[2], np.float32)
    # Where nothing unprotected lies within reach, carry the window's mean.
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-6), mean_delta[None, None, :])
    out = b + delta * free[:, :, None] + fill * pw[:, :, None] if b.ndim == 3 else (
        b + (delta * free[:, :, None] + fill * pw[:, :, None])[:, :, 0]
    )

    result = np.array(after, copy=True)
    if np.issubdtype(after.dtype, np.integer):
        info = np.iinfo(after.dtype)
        result[y1:y2, x1:x2] = np.clip(np.rint(out), info.min, info.max).astype(after.dtype)
    else:
        # Float frames are [0, 1] (global phase) or [0, 255] (face crops).
        hi = 1.0 if max(float(before.max()), float(after.max())) <= 1.0 + 1e-6 else 255.0
        result[y1:y2, x1:x2] = np.clip(out, 0.0, hi).astype(after.dtype)
    return result


def summary(mask: Optional[np.ndarray]) -> dict:
    """Small JSON-safe description of a source-grid mask (edit report)."""
    if mask is None:
        return {"painted": False}
    return {
        "painted": True,
        "coverage_pct": round(float((mask > 0.5).mean() * 100.0), 3),
        "frame": [int(mask.shape[1]), int(mask.shape[0])],
    }
