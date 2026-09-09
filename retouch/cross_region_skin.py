"""P7 — bounded cross-region skin appearance propagation.

This module deliberately implements an appearance edit, not recovery of hidden
skin reflectance or illumination.  A face edit is summarized as a robust LAB
delta and that delta is applied to an independently owned same-person support.
The support remains soft and the source image is restored exactly outside it.

All image arithmetic is float32. LAB arithmetic is reported in the
repository's explicit OpenCV-LAB code convention: L255 is [0, 255] and a/b
are centred at 128. Conversion uses the float-native CIELAB helpers underneath
so no uint8 pixel arithmetic is needed. Public helpers accept uint8 BGR or
float32 BGR in [0, 1] and return the edited image's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .color_space import bgr_f32_to_lch_f32, skin_mask_lch
from .utils import (
    bgr_f32_to_lab_f32,
    lab_f32_to_bgr_f32,
    normalize_mask,
    restore_outside_support,
    squeeze_mask,
)

__all__ = [
    "CrossRegionSkinResult",
    "infer_same_person_skin_support",
    "propagate_face_edit_delta",
]


# These are intentionally conservative candidate limits in OpenCV-LAB code
# units. They are not calibrated skin truth or a claim of physical albedo.
MAX_L_DELTA = 6.0
MAX_AB_DELTA = 4.0
MIN_FACE_PIXELS = 32
MIN_TARGET_PIXELS = 32
MIN_DELTA = 0.05


def _lab_float_to_lab255(lab: np.ndarray) -> np.ndarray:
    """Convert float-native CIELAB (L* / centred a,b) to explicit LAB codes."""
    out = lab.astype(np.float32, copy=True)
    out[..., 0] *= np.float32(255.0 / 100.0)
    out[..., 1:] += np.float32(128.0)
    return out


def _lab255_to_lab_float(lab255: np.ndarray) -> np.ndarray:
    """Convert explicit LAB codes back to float-native CIELAB."""
    out = lab255.astype(np.float32, copy=True)
    out[..., 0] *= np.float32(100.0 / 255.0)
    out[..., 1:] -= np.float32(128.0)
    return out


@dataclass(frozen=True)
class CrossRegionSkinResult:
    """Result and auditable disposition of one P7 propagation attempt."""

    image: np.ndarray
    applied: bool
    abstained: bool
    reason: str
    face_pixels: int = 0
    target_pixels: int = 0
    changed_pixels: int = 0
    face_delta_lab255: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    applied_delta_lab255: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe diagnostics for manifests and API callers."""
        return {
            "applied": bool(self.applied),
            "abstained": bool(self.abstained),
            "reason": self.reason,
            "face_pixels": int(self.face_pixels),
            "target_pixels": int(self.target_pixels),
            "changed_pixels": int(self.changed_pixels),
            "face_delta_lab255": [float(v) for v in self.face_delta_lab255],
            "applied_delta_lab255": [float(v) for v in self.applied_delta_lab255],
        }


def _as_float01(image: np.ndarray, name: str) -> Tuple[np.ndarray, bool]:
    """Validate BGR input and return float32 [0, 1] plus its uint8 flag."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f"{name} must be a non-empty HxWx3 BGR array")

    if image.dtype == np.uint8:
        return image.astype(np.float32) * (1.0 / 255.0), True
    if image.dtype != np.float32:
        raise TypeError(f"{name} must be uint8 or float32, got {image.dtype}")
    if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError(f"{name} float32 values must be finite and in [0, 1]")
    return image.astype(np.float32, copy=False), False


def _mask(mask: Optional[np.ndarray], shape: Tuple[int, int], name: str) -> Optional[np.ndarray]:
    """Validate and normalize one soft mask to float32 [0, 1]."""
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        raise ValueError(f"{name} must be a non-empty NumPy mask")
    normalized = normalize_mask(mask)
    if normalized is None:
        return None
    normalized = squeeze_mask(normalized).astype(np.float32, copy=False)
    if normalized.ndim != 2 or normalized.shape != shape:
        raise ValueError(
            f"{name} shape {normalized.shape} does not match image shape {shape}"
        )
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.clip(normalized, 0.0, 1.0)


def infer_same_person_skin_support(
    source_bgr: np.ndarray,
    person_mask: np.ndarray,
    face_mask: np.ndarray,
    *,
    face_exclusion: Optional[np.ndarray] = None,
    lip_exclusion: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Infer a conservative body-skin support for one detected person.

    The support is the connected person-mask component containing the face,
    intersected with a source-image LCH skin-color likelihood and excluding
    already-owned face/hair/lip regions.  This is a segmentation convenience,
    not a semantic guarantee; callers should prefer an explicit reviewed mask
    when available.  A zero mask means that no trustworthy support was found.

    Args:
        source_bgr: uint8 BGR or float32 BGR [0, 1] reference image.
        person_mask: Float/uint8 person segmentation mask.
        face_mask: Float/uint8 face-skin mask.
        face_exclusion: Optional face+hair mask to keep out of the body support.
        lip_exclusion: Optional lip mask to keep out of the body support.

    Returns:
        Float32 mask in [0, 1], same spatial shape as ``source_bgr``.
    """
    source, _ = _as_float01(source_bgr, "source_bgr")
    shape = source.shape[:2]
    person = _mask(person_mask, shape, "person_mask")
    face = _mask(face_mask, shape, "face_mask")
    if person is None or face is None:
        return np.zeros(shape, dtype=np.float32)

    person_binary = (person > 0.5).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(person_binary)
    face_seed = (face > 0.5) & (person > 0.5)
    owner_ids = np.unique(labels[face_seed])
    owner_ids = owner_ids[owner_ids != 0]
    if owner_ids.size != 1:
        # Multiple disconnected components are ambiguous ownership. The engine
        # also rejects multi-face calls before reaching this helper.
        return np.zeros(shape, dtype=np.float32)
    owner = (labels == int(owner_ids[0])).astype(np.float32)

    lch = bgr_f32_to_lch_f32(source * 255.0)
    skin_likelihood = skin_mask_lch(
        lch,
        hue_center=25.0,
        hue_tolerance=25.0,
        chroma_min=8.0,
    )

    exclusion = face.copy()
    for extra, extra_name in ((face_exclusion, "face_exclusion"), (lip_exclusion, "lip_exclusion")):
        if extra is not None:
            extra_mask = _mask(extra, shape, extra_name)
            if extra_mask is not None:
                exclusion = np.maximum(exclusion, extra_mask)

    # Keep the segmentation probability soft. Multiplying by (1-exclusion)
    # prevents the inferred support from re-entering face-owned pixels.
    support = owner * person * skin_likelihood * np.clip(1.0 - exclusion, 0.0, 1.0)
    return np.clip(support, 0.0, 1.0).astype(np.float32)


def _abstain(
    edited: np.ndarray,
    reason: str,
    *,
    face_pixels: int = 0,
    target_pixels: int = 0,
    face_delta: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CrossRegionSkinResult:
    """Build an exact-identity, auditable abstention result."""
    return CrossRegionSkinResult(
        image=edited.copy(),
        applied=False,
        abstained=True,
        reason=reason,
        face_pixels=face_pixels,
        target_pixels=target_pixels,
        face_delta_lab255=face_delta,
    )


def propagate_face_edit_delta(
    source_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    face_mask: np.ndarray,
    target_mask: Optional[np.ndarray],
    *,
    strength: float = 1.0,
    protect_mask: Optional[np.ndarray] = None,
    max_l_delta: float = MAX_L_DELTA,
    max_ab_delta: float = MAX_AB_DELTA,
    min_face_pixels: int = MIN_FACE_PIXELS,
    min_target_pixels: int = MIN_TARGET_PIXELS,
) -> CrossRegionSkinResult:
    """Propagate a bounded face appearance delta to an owned target support.

        ``source_bgr`` is the image immediately before face edits and
    ``edited_bgr`` is the current image after those edits. The robust median
    LAB delta over ``face_mask`` is the approved appearance change. It is
    applied uniformly in LAB to ``target_mask``; local luminance/chroma
    variation and high-frequency texture in the target are retained. A
    uniform L offset preserves existing shading gradients but does not claim to
    recover physical illumination.

    Args:
        source_bgr: uint8 BGR or float32 BGR [0, 1] before face edits.
        edited_bgr: Same-shape current BGR image; output keeps its dtype.
        face_mask: Float/uint8 face-skin support used to estimate the delta.
        target_mask: Reviewed same-person support. ``None`` always abstains.
        strength: Opacity in [0, 1] for the bounded delta.
        protect_mask: Optional float/uint8 mask removed from target support.
        max_l_delta: Maximum absolute L255 delta before strength.
        max_ab_delta: Maximum absolute a/b delta before strength.

    Returns:
        :class:`CrossRegionSkinResult` with an image and auditable disposition.
    """
    source, _ = _as_float01(source_bgr, "source_bgr")
    edited, edited_uint8 = _as_float01(edited_bgr, "edited_bgr")
    if source.shape != edited.shape:
        raise ValueError(
            f"source_bgr shape {source.shape} does not match edited_bgr {edited.shape}"
        )
    if not np.isfinite(strength) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"strength must be finite and in [0, 1], got {strength!r}")
    if not np.isfinite(max_l_delta) or not np.isfinite(max_ab_delta):
        raise ValueError("delta bounds must be finite")
    if max_l_delta < 0.0 or max_ab_delta < 0.0:
        raise ValueError("delta bounds must be non-negative")

    shape = source.shape[:2]
    face = _mask(face_mask, shape, "face_mask")
    target = _mask(target_mask, shape, "target_mask")
    protect = _mask(protect_mask, shape, "protect_mask")
    # Abstentions are still part of the public image contract: return an
    # exact copy in the caller's original dtype rather than leaking the
    # float32 working representation used for LAB arithmetic.
    identity = edited_bgr.copy() if edited_uint8 else edited.copy()
    if face is None:
        return _abstain(identity, "face_reference_required")
    if target is None:
        return _abstain(identity, "target_support_required")

    face_indices = face > 0.5
    target_support = target.copy()
    target_support[face_indices] = 0.0
    if protect is not None:
        target_support *= np.clip(1.0 - protect, 0.0, 1.0)

    face_pixels = int(np.count_nonzero(face_indices))
    target_pixels = int(np.count_nonzero(target_support > 0.1))
    if face_pixels < max(1, int(min_face_pixels)):
        return _abstain(
            identity,
            "face_reference_insufficient",
            face_pixels=face_pixels,
            target_pixels=target_pixels,
        )
    if target_pixels < max(1, int(min_target_pixels)):
        return _abstain(
            identity,
            "target_support_insufficient",
            face_pixels=face_pixels,
            target_pixels=target_pixels,
        )

    source_lab = _lab_float_to_lab255(bgr_f32_to_lab_f32(source * 255.0))
    edited_lab = _lab_float_to_lab255(bgr_f32_to_lab_f32(edited * 255.0))
    delta_pixels = edited_lab[face_indices] - source_lab[face_indices]
    if not np.isfinite(delta_pixels).all():
        return _abstain(
            identity,
            "face_reference_nonfinite",
            face_pixels=face_pixels,
            target_pixels=target_pixels,
        )

    face_delta = np.median(delta_pixels, axis=0).astype(np.float32)
    face_delta_tuple = tuple(float(v) for v in face_delta)
    if float(np.max(np.abs(face_delta))) < float(MIN_DELTA):
        return _abstain(
            identity,
            "no_approved_face_delta",
            face_pixels=face_pixels,
            target_pixels=target_pixels,
            face_delta=face_delta_tuple,
        )

    bounded = np.array(
        [
            np.clip(face_delta[0], -float(max_l_delta), float(max_l_delta)),
            np.clip(face_delta[1], -float(max_ab_delta), float(max_ab_delta)),
            np.clip(face_delta[2], -float(max_ab_delta), float(max_ab_delta)),
        ],
        dtype=np.float32,
    )
    applied_delta = bounded * np.float32(strength)
    applied_delta_tuple = tuple(float(v) for v in applied_delta)
    if float(np.max(np.abs(applied_delta))) < float(MIN_DELTA):
        return _abstain(
            identity,
            "bounded_delta_below_threshold",
            face_pixels=face_pixels,
            target_pixels=target_pixels,
            face_delta=face_delta_tuple,
        )

    result_lab255 = edited_lab.copy()
    result_lab255 += target_support[:, :, None] * applied_delta[None, None, :]
    result_lab255 = np.clip(result_lab255, 0.0, 255.0)
    result_lab = _lab255_to_lab_float(result_lab255)
    processed = np.clip(lab_f32_to_bgr_f32(result_lab) * (1.0 / 255.0), 0.0, 1.0).astype(np.float32)
    # The operation has no authority outside the final support. This also
    # prevents a float LAB round trip from changing unrelated background pixels.
    processed = restore_outside_support(edited, processed, target_support)

    if edited_uint8:
        output = np.clip(processed * 255.0, 0.0, 255.0).astype(np.uint8)
        output[target_support == 0.0] = edited_bgr[target_support == 0.0]
    else:
        output = processed

    changed = np.any(
        np.abs(output.astype(np.float32) - edited_bgr.astype(np.float32)) > (
            0.0 if edited_uint8 else 1e-7
        ),
        axis=2,
    )
    return CrossRegionSkinResult(
        image=output,
        applied=True,
        abstained=False,
        reason="applied",
        face_pixels=face_pixels,
        target_pixels=target_pixels,
        changed_pixels=int(np.count_nonzero(changed)),
        face_delta_lab255=face_delta_tuple,
        applied_delta_lab255=applied_delta_tuple,
    )
