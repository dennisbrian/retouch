"""Deterministic, source-constrained exemplar filling.

This is intentionally a small classical PatchMatch-style primitive rather
than a semantic inpainting model.  It grows a hole from known pixels, keeps a
nearest-neighbour field (NNF) of source patch centres, and proposes both
neighbour-propagated and seeded random source patches.  A caller may restrict
the source material with ``source_mask``; a source patch is never sampled from
outside that mask or from the hole being repaired.

It is suitable for repeated skin, fabric, and background texture.  It cannot
reconstruct a unique object that is completely occluded by the mask.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np


SourceMap = np.ndarray
FillResult = Union[np.ndarray, Tuple[np.ndarray, SourceMap]]


def patchmatch_fill(
    image: np.ndarray,
    hole_mask: np.ndarray,
    *,
    source_mask: Optional[np.ndarray] = None,
    patch_size: int = 7,
    iterations: int = 5,
    seed: int = 0,
    max_candidates: int = 512,
    return_source_map: bool = False,
) -> FillResult:
    """Fill ``hole_mask`` using patches from an allowed source region.

    Args:
        image: ``(H, W, 3)`` uint8 or float32 BGR image. Float pixels are
            interpreted in the same value scale as the input; no colour-space
            conversion is performed.
        hole_mask: Non-zero pixels are repaired.
        source_mask: Optional non-zero source-pixel mask. When omitted, every
            pixel outside the hole is eligible. Source patches must fit fully
            within this mask and never overlap the hole.
        patch_size: Odd patch width. Smaller patches preserve fine texture;
            larger patches favour broad repeated material.
        iterations: Number of deterministic propagation proposal passes.
        seed: Fixed random proposal seed. The same inputs and seed always
            produce the same output.
        max_candidates: Cap on random source candidates per destination. This
            bounds the cost on high-resolution brush masks.
        return_source_map: Also return ``(H, W, 2)`` integer source centres
            in ``(y, x)`` order; unfilled/non-hole pixels are ``(-1, -1)``.

    Returns:
        Filled image preserving dtype. On a degenerate source region it falls
        back to OpenCV Telea and marks source-map entries ``(-1, -1)``.

    The routine is classical texture synthesis, not object reconstruction:
    it needs suitable same-material source pixels.  Callers should provide a
    semantic skin/fabric/background source mask where available.
    """
    _validate_image(image)
    hole = _coerce_mask(hole_mask, image.shape[:2], "hole_mask")
    source_map = np.full((*image.shape[:2], 2), -1, dtype=np.int32)
    if not np.any(hole):
        result = image.copy()
        return (result, source_map) if return_source_map else result

    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    allowed = ~hole
    if source_mask is not None:
        allowed &= _coerce_mask(source_mask, image.shape[:2], "source_mask")

    radius = patch_size // 2
    valid_centres = _valid_source_centres(allowed, radius)
    if len(valid_centres) == 0:
        fallback = _telea_fallback(image, hole)
        return (fallback, source_map) if return_source_map else fallback

    image_float = image.astype(np.float32, copy=False)
    result = image_float.copy()
    # A one-pixel ring gives every boundary candidate an initial comparison
    # context. Filled pixels are then promoted as the front advances inward.
    known = ~hole
    ys, xs = np.nonzero(hole)
    order = _inward_order(hole)
    rng = np.random.default_rng(seed)
    sample_count = min(len(valid_centres), max_candidates)

    for _ in range(iterations):
        changed = False
        for y, x in order:
            candidates = _candidate_centres(
                y,
                x,
                source_map,
                valid_centres,
                sample_count,
                rng,
            )
            best_y, best_x = _best_patch_candidate(
                result,
                known,
                y,
                x,
                candidates,
                radius,
            )
            if best_y < 0:
                continue
            previous = source_map[y, x]
            source_map[y, x] = (best_y, best_x)
            result[y, x] = image_float[best_y, best_x]
            known[y, x] = True
            changed |= bool(previous[0] != best_y or previous[1] != best_x)
        # Further passes refine the NNF; do not pay for them if all choices
        # have stabilized after the first front propagation.
        if not changed:
            break

    # In practice every point has candidates after the first pass. This keeps
    # a deterministic contract for unusual disconnected/tiny masks.
    unresolved = hole & (source_map[..., 0] < 0)
    if np.any(unresolved):
        fallback = _telea_fallback(image, unresolved)
        result[unresolved] = fallback[unresolved].astype(np.float32)

    converted = _restore_dtype(result, image.dtype)
    return (converted, source_map) if return_source_map else converted


def seamless_blend_roi(
    base: np.ndarray,
    filled: np.ndarray,
    mask: np.ndarray,
    *,
    feather_radius: int = 3,
) -> np.ndarray:
    """Blend a filled image into ``base`` inside the mask's bounding ROI.

    ``cv2.seamlessClone`` is attempted only on the local ROI. Its result is
    copied back only where ``mask`` is non-zero, so pixels outside the repair
    remain byte-identical. If OpenCV rejects the input (or the ROI is too
    small), a deterministic inward feather is used instead.
    """
    _validate_image(base)
    _validate_image(filled)
    if base.shape != filled.shape or base.dtype != filled.dtype:
        raise ValueError("base and filled must have matching shape and dtype")
    repair = _coerce_mask(mask, base.shape[:2], "mask")
    if not np.any(repair):
        return base.copy()

    y0, y1, x0, x1 = _roi_bounds(repair, padding=max(2, feather_radius))
    base_roi = base[y0:y1, x0:x1]
    filled_roi = filled[y0:y1, x0:x1]
    mask_roi = (repair[y0:y1, x0:x1].astype(np.uint8) * 255)

    try:
        base_u8 = _as_uint8(base_roi)
        filled_u8 = _as_uint8(filled_roi)
        clone = cv2.seamlessClone(
            filled_u8,
            base_u8,
            mask_roi,
            (base_u8.shape[1] // 2, base_u8.shape[0] // 2),
            cv2.NORMAL_CLONE,
        )
        blended_roi = _from_uint8(clone, base.dtype)
    except cv2.error:
        blended_roi = _feather_fallback(base_roi, filled_roi, mask_roi, feather_radius)

    output = base.copy()
    inside = repair[y0:y1, x0:x1]
    output_roi = output[y0:y1, x0:x1]
    output_roi[inside] = blended_roi[inside]
    return output


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if image.dtype not in (np.uint8, np.float32):
        raise ValueError(f"image must be uint8 or float32, got {image.dtype}")


def _coerce_mask(mask: np.ndarray, shape: Tuple[int, int], name: str) -> np.ndarray:
    if mask.shape[:2] != shape or mask.ndim != 2:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    return mask > 0


def _valid_source_centres(allowed: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    valid = cv2.erode(allowed.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT)
    # OpenCV's morphology border default treats the outside as the extremal
    # value for erosion. Explicitly reject edges so every candidate can be
    # sliced as a complete source patch.
    valid[:radius, :] = 0
    valid[-radius:, :] = 0
    valid[:, :radius] = 0
    valid[:, -radius:] = 0
    return np.argwhere(valid > 0).astype(np.int32)


def _inward_order(hole: np.ndarray) -> np.ndarray:
    # Distance from the known boundary gives an exemplar-filling order. Stable
    # lexicographic sorting keeps equal-distance decisions deterministic.
    distance = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(hole)
    order = np.lexsort((xs, ys, distance[ys, xs]))
    return np.column_stack((ys[order], xs[order])).astype(np.int32)


def _candidate_centres(
    y: int,
    x: int,
    source_map: SourceMap,
    valid_centres: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    proposals = []
    # Neighbour-propagated offsets preserve coherent texture translations.
    for ny, nx in ((y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)):
        if 0 <= ny < source_map.shape[0] and 0 <= nx < source_map.shape[1]:
            sy, sx = source_map[ny, nx]
            if sy >= 0:
                proposals.append((int(sy + y - ny), int(sx + x - nx)))

    indices = rng.choice(len(valid_centres), size=sample_count, replace=False)
    proposals.extend((int(yx[0]), int(yx[1])) for yx in valid_centres[indices])
    if not proposals:
        return valid_centres[:0]

    candidates = np.asarray(proposals, dtype=np.int32)
    # Propagated coordinates may fall outside the valid source set. A compact
    # structured membership test avoids building a large set for every pixel.
    in_bounds = (
        (candidates[:, 0] >= 0)
        & (candidates[:, 0] < source_map.shape[0])
        & (candidates[:, 1] >= 0)
        & (candidates[:, 1] < source_map.shape[1])
    )
    candidates = candidates[in_bounds]
    if len(candidates) == 0:
        return valid_centres[indices]
    valid_key = valid_centres[:, 0].astype(np.int64) * source_map.shape[1] + valid_centres[:, 1]
    cand_key = candidates[:, 0].astype(np.int64) * source_map.shape[1] + candidates[:, 1]
    keep = np.isin(cand_key, valid_key)
    candidates = candidates[keep]
    if len(candidates) == 0:
        return valid_centres[indices]
    return np.unique(candidates, axis=0)


def _best_patch_candidate(
    image: np.ndarray,
    known: np.ndarray,
    y: int,
    x: int,
    candidates: np.ndarray,
    radius: int,
) -> Tuple[int, int]:
    if len(candidates) == 0:
        return -1, -1
    patch = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    known_patch = known[y - radius : y + radius + 1, x - radius : x + radius + 1]
    if patch.shape[0] != radius * 2 + 1 or not np.any(known_patch):
        # This can only occur for a hole touching the image edge. The source
        # centre still gives a deterministic and valid texture proposal.
        return int(candidates[0, 0]), int(candidates[0, 1])

    source_patches = np.stack(
        [
            image[sy - radius : sy + radius + 1, sx - radius : sx + radius + 1]
            for sy, sx in candidates
        ],
        axis=0,
    )
    weights = known_patch[..., None].astype(np.float32)
    squared_error = ((source_patches - patch[None, ...]) ** 2 * weights).sum(axis=(1, 2, 3))
    best = int(np.argmin(squared_error))
    return int(candidates[best, 0]), int(candidates[best, 1])


def _telea_fallback(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    mask_u8 = (hole.astype(np.uint8) * 255)
    if image.dtype == np.uint8:
        return cv2.inpaint(image, mask_u8, 3, cv2.INPAINT_TELEA)
    channels = [cv2.inpaint(image[..., channel], mask_u8, 3, cv2.INPAINT_TELEA) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


def _restore_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return image.astype(np.float32, copy=False)


def _roi_bounds(mask: np.ndarray, padding: int) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return (
        max(0, int(ys.min()) - padding),
        min(mask.shape[0], int(ys.max()) + padding + 1),
        max(0, int(xs.min()) - padding),
        min(mask.shape[1], int(xs.max()) + padding + 1),
    )


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _from_uint8(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return image if dtype == np.uint8 else image.astype(np.float32)


def _feather_fallback(
    base: np.ndarray,
    filled: np.ndarray,
    mask_u8: np.ndarray,
    feather_radius: int,
) -> np.ndarray:
    inside = mask_u8 > 0
    if feather_radius <= 0:
        result = base.copy()
        result[inside] = filled[inside]
        return result
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    alpha = np.clip(distance / float(feather_radius), 0.0, 1.0)[..., None]
    mixed = base.astype(np.float32) * (1.0 - alpha) + filled.astype(np.float32) * alpha
    result = base.copy()
    result[inside] = _restore_dtype(mixed, base.dtype)[inside]
    return result
