"""Read-only face-to-body harmony metrics for portrait retouch QA.

The metrics are deliberately relative to the same subject's source image.
They diagnose a mismatch; they do not tune retouch parameters automatically.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from .freckle import FreckleRemover
from .specular import extract_specular


_MIN_REGION_PIXELS = 512

# A preserved mark must re-detect within this many pixels of its original
# centroid (spike-calibrated at a 1600px long edge).
_MARK_MATCH_RADIUS = 6.0
# H4 is intentionally a review gate. It warns when a preserve-class identity
# mark disappears, but never guesses which retouch control caused the loss.
MARK_RETENTION_FLAG_FLOOR = 0.85


def _mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    out = mask[..., 0] if mask.ndim == 3 else mask
    if out.shape != shape:
        out = cv2.resize(out.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    out = out.astype(np.float32)
    if out.max() > 1.5:
        out /= 255.0
    return np.clip(out, 0.0, 1.0)


def build_face_anchored_body_mask(
    img_bgr: np.ndarray,
    face_skin_mask: np.ndarray,
    person_mask: np.ndarray,
    hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate exposed body skin from the subject's own face chroma locus.

    This is a QA-only conservative mask. It uses robust face-relative Lab a/b
    distances rather than fixed hue/chroma thresholds, then requires person
    connectivity to avoid treating the scene background as body skin.

    ``hair_mask`` (optional, full-image) guards against wigs whose chroma sits
    inside the face locus (pink/blond) being measured as body skin. It can
    only remove pixels, so a missing or under-covering hair parse degrades
    gracefully to the chroma gate alone.
    """
    h, w = img_bgr.shape[:2]
    face = _mask(face_skin_mask, (h, w))
    person = _mask(person_mask, (h, w))
    if face is None or person is None or np.count_nonzero(face > 0.5) < _MIN_REGION_PIXELS:
        return np.zeros((h, w), dtype=np.float32)

    lab = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    selected = face > 0.5
    ab = lab[..., 1:3]
    center = np.median(ab[selected], axis=0)
    deviations = np.abs(ab[selected] - center)
    # Scale comes entirely from the subject's face. The epsilon only avoids a
    # singular variance on synthetic uniform patches.
    scale = np.maximum(np.median(deviations, axis=0) * 1.4826, 1e-3)
    distance = np.sqrt(np.sum(((ab - center) / scale) ** 2, axis=-1))
    candidate = ((distance <= 3.5) & (person > 0.5)).astype(np.float32)

    # Keep the face out, including segmentation gaps around brows and jaw.
    face_exclusion = cv2.dilate((face > 0.2).astype(np.uint8), np.ones((15, 15), np.uint8))
    candidate *= 1.0 - face_exclusion.astype(np.float32)

    hair = _mask(hair_mask, (h, w))
    if hair is not None:
        candidate *= (hair < 0.5).astype(np.float32)

    # Contiguity: geodesically walk outward from the face along the person
    # silhouette, then keep every candidate *component* the walk touches.
    # The walk gates component connectivity, not individual pixels — a chest
    # or arm patch whose top edge is reachable is kept in full. (Using the
    # walked region as a hard per-pixel gate was measured to truncate the
    # mask to a ~50px near-face ring on real portraits, so the metric never
    # saw the body skin the engine's body stage actually edits.)
    reachable = (face > 0.2).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for _ in range(12):
        reachable = cv2.dilate(reachable, kernel) * (person > 0.5)
    n_labels, labels = cv2.connectedComponents((candidate > 0.3).astype(np.uint8))
    if n_labels > 1:
        reach_sel = reachable > 0.1
        overlap = np.bincount(
            labels[reach_sel & (labels > 0)], minlength=n_labels)
        kept_ids = np.nonzero(overlap > 10)[0]
        candidate = (candidate * np.isin(labels, kept_ids)).astype(np.float32)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.GaussianBlur(candidate, (0, 0), 2.0)
    return (candidate * (1.0 - face_exclusion.astype(np.float32))).astype(np.float32)


def _highband_energy(img_bgr: np.ndarray, mask: np.ndarray, sigma: float) -> float:
    gray = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    high = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    values = high[mask > 0.5]
    if values.size < _MIN_REGION_PIXELS:
        return float("nan")
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _specular_energy(img_bgr: np.ndarray, mask: np.ndarray) -> float:
    values = extract_specular(img_bgr, skin_mask=mask)[mask > 0.5]
    if values.size < _MIN_REGION_PIXELS:
        return float("nan")
    return float(np.mean(values))


def _to_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.dtype == np.uint8:
        return img_bgr
    return np.clip(img_bgr, 0, 255).astype(np.uint8)


def _mark_retention(
    reference_img_bgr: np.ndarray,
    processed_img_bgr: np.ndarray,
    face_mask: np.ndarray,
    mark_policy: Optional[Mapping[str, Any]] = None,
) -> tuple[int, int]:
    """H4: preserve-class (beauty-mark) survival, reference vs processed.

    Detection runs on the face-mask bounding box only — identity marks are a
    face-signature concern here, and the crop keeps the QA budget small. Body
    marks become relevant once the body parity op exists (blemish plan S6).
    """
    ys, xs = np.nonzero(face_mask > 0.5)
    if ys.size < _MIN_REGION_PIXELS:
        return 0, 0
    pad = 8
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, face_mask.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, face_mask.shape[1])
    crop_mask = face_mask[y0:y1, x0:x1]

    if mark_policy is None:
        # Compatibility boundary: existing H4 treats beauty marks as the
        # preserve class until an explicit policy is supplied.
        remover = FreckleRemover()
        before = [
            c for c in remover.classify_anomalies(
                _to_u8(reference_img_bgr[y0:y1, x0:x1]),
                face_mask=crop_mask,
                confidence_threshold=0.6,
            )
            if c.classification == "beauty_mark"
        ]
        after = remover.classify_anomalies(
            _to_u8(processed_img_bgr[y0:y1, x0:x1]),
            face_mask=crop_mask,
            confidence_threshold=0.4,
        )
        same_class = lambda _before, _after: True
    else:
        # An explicit policy changes the denominator to preserve-class marks
        # only. Acne configured for removal must never count as an H4 loss.
        from .marks import detect_marks, resolve_mark_action

        before = [
            record for record in detect_marks(
                _to_u8(reference_img_bgr[y0:y1, x0:x1]),
                face_mask=crop_mask,
                confidence_threshold=0.6,
            )
            if (action := resolve_mark_action(record, mark_policy)) is not None
            and action[0] == "preserve"
        ]
        after = detect_marks(
            _to_u8(processed_img_bgr[y0:y1, x0:x1]),
            face_mask=crop_mask,
            confidence_threshold=0.4,
        )
        same_class = lambda before_record, after_record: before_record.mark_class == after_record.mark_class
    if not before:
        return 0, 0
    kept = 0
    for b in before:
        for a in after:
            if (same_class(b, a)
                    and np.hypot(a.centroid[0] - b.centroid[0],
                                 a.centroid[1] - b.centroid[1]) <= _MARK_MATCH_RADIUS):
                kept += 1
                break
    return len(before), kept


def evaluate_harmony(
    img_bgr: np.ndarray,
    *,
    face_skin_mask: Optional[np.ndarray],
    body_skin_mask: Optional[np.ndarray],
    reference_img_bgr: Optional[np.ndarray] = None,
    mark_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float | bool]:
    """Measure face/body parity with a review-only H4 retention flag.

    Other harmony metrics remain observational because their real-engine
    thresholds are not calibrated. A harmony flag never triggers auto-backoff.
    """
    h, w = img_bgr.shape[:2]
    face = _mask(face_skin_mask, (h, w))
    body = _mask(body_skin_mask, (h, w))
    unavailable: Dict[str, float | bool] = {
        "available": False,
        "flagged": False,
        "score": 0.0,
        "texture_parity_ratio": float("nan"),
        "texture_parity_drift": float("nan"),
        "specular_parity_ratio": float("nan"),
        "specular_parity_drift": float("nan"),
        "face_pixels": 0,
        "body_pixels": 0,
        "face_banding": float("nan"),
        "body_banding": float("nan"),
        "face_banding_delta": float("nan"),
        "body_banding_delta": float("nan"),
        "marks_before": 0,
        "marks_kept": 0,
        "mark_retention": float("nan"),
    }
    if face is None or body is None:
        return unavailable
    face_px = int(np.count_nonzero(face > 0.5))
    body_px = int(np.count_nonzero(body > 0.5))
    if face_px < _MIN_REGION_PIXELS or body_px < _MIN_REGION_PIXELS:
        unavailable.update(face_pixels=face_px, body_pixels=body_px)
        return unavailable

    sigma = max(1.5, np.sqrt(face_px) / 120.0)
    face_energy = _highband_energy(img_bgr, face, sigma)
    body_energy = _highband_energy(img_bgr, body, sigma)
    face_specular = _specular_energy(img_bgr, face)
    body_specular = _specular_energy(img_bgr, body)
    texture_ratio = body_energy / max(face_energy, 1e-6)
    specular_ratio = body_specular / max(face_specular, 1e-6)

    texture_drift = float("nan")
    specular_drift = float("nan")
    if reference_img_bgr is not None and reference_img_bgr.shape[:2] == (h, w):
        ref_face_energy = _highband_energy(reference_img_bgr, face, sigma)
        ref_body_energy = _highband_energy(reference_img_bgr, body, sigma)
        ref_face_specular = _specular_energy(reference_img_bgr, face)
        ref_body_specular = _specular_energy(reference_img_bgr, body)
        ref_texture_ratio = ref_body_energy / max(ref_face_energy, 1e-6)
        ref_specular_ratio = ref_body_specular / max(ref_face_specular, 1e-6)
        texture_drift = float(abs(np.log((texture_ratio + 1e-6) / (ref_texture_ratio + 1e-6))))
        specular_drift = float(abs(np.log((specular_ratio + 1e-6) / (ref_specular_ratio + 1e-6))))

    # Lazy import avoids a qa_detectors -> harmony import cycle.
    from .qa_detectors import detect_banding

    face_banding = float(detect_banding(img_bgr, face).get("score", 0.0))
    body_banding = float(detect_banding(img_bgr, body).get("score", 0.0))

    # H5 must be differential: absolute banding scores flag busy-but-untouched
    # originals (spike §3.5), so only the change vs the reference is a defect.
    face_banding_delta = float("nan")
    body_banding_delta = float("nan")
    marks_before = 0
    marks_kept = 0
    mark_retention = float("nan")
    if reference_img_bgr is not None and reference_img_bgr.shape[:2] == (h, w):
        face_banding_delta = face_banding - float(
            detect_banding(reference_img_bgr, face).get("score", 0.0))
        body_banding_delta = body_banding - float(
            detect_banding(reference_img_bgr, body).get("score", 0.0))
        marks_before, marks_kept = _mark_retention(
            reference_img_bgr, img_bgr, face, mark_policy=mark_policy)
        if marks_before:
            mark_retention = marks_kept / marks_before

    h4_flagged = bool(
        np.isfinite(mark_retention) and mark_retention < MARK_RETENTION_FLAG_FLOOR
    )

    return {
        "available": True,
        "flagged": h4_flagged,
        "score": float(1.0 - mark_retention) if np.isfinite(mark_retention) else 0.0,
        "texture_parity_ratio": float(texture_ratio),
        "texture_parity_drift": texture_drift,
        "specular_parity_ratio": float(specular_ratio),
        "specular_parity_drift": specular_drift,
        "face_pixels": face_px,
        "body_pixels": body_px,
        "face_banding": face_banding,
        "body_banding": body_banding,
        "face_banding_delta": face_banding_delta,
        "body_banding_delta": body_banding_delta,
        "marks_before": marks_before,
        "marks_kept": marks_kept,
        "mark_retention": mark_retention,
        "mark_retention_floor": MARK_RETENTION_FLAG_FLOOR,
    }
