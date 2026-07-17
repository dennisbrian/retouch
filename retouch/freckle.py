"""Freckle and beauty-mark selective removal.

Classifies facial anomalies (freckle / beauty_mark / blemish / noise) with
skin-tone-normalized LAB scoring, then removes freckles via inpainting while
preserving beauty marks. Detection and healing are implemented locally; the
shared ``inpaint_and_blend`` helper from :mod:`retouch.blemish` does the heal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import logging
import numpy as np

from .utils import apply_u8_op_float, normalize_mask
from .blemish import inpaint_and_blend

logger = logging.getLogger(__name__)

_CLASS_TYPES = frozenset({"freckle", "beauty_mark", "blemish", "noise"})


@dataclass(frozen=True)
class FreckleClassification:
    """Classification result for a single detected anomaly."""

    anomaly_id: int
    classification: str
    confidence: float
    reason: str
    centroid: Tuple[float, float] = (0.0, 0.0)
    area: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        if self.classification not in _CLASS_TYPES:
            raise ValueError(
                f"Unknown classification {self.classification!r}; "
                f"expected one of {sorted(_CLASS_TYPES)}"
            )
        object.__setattr__(
            self, "confidence", min(1.0, max(0.0, float(self.confidence)))
        )


class FreckleRemover:
    """Detect and selectively remove freckles while preserving beauty marks."""

    # Connected-component area tuning (px^2).
    _MIN_AREA = 2
    _NOISE_MAX_AREA = 4
    _FRECKLE_SIZE = (4, 25)
    _BEAUTY_SIZE = (10, 70)
    _BLEMISH_SIZE = (6, 80)
    _MAX_COMPONENTS = 2000
    # Local dark-spot contrast is multiplicative with skin reflectance. A
    # ratio avoids the former fixed seven-code-value gate disappearing on
    # darker skin while retaining a comparable light-skin sensitivity.
    _DARK_SPOT_RELATIVE_DEVIATION = 0.05

    def _to_lab(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR -> LAB in the uint8-scale convention (L 0-255, a/b centered 128).

        Uses the float LAB path for float32 input to avoid uint8 quantization.
        """
        if img_bgr.dtype == np.float32:
            from .utils import bgr_f32_to_lab_f32

            return bgr_f32_to_lab_f32(img_bgr)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    def _detect_components(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        """Find dark-spot / reddish anomaly components within the skin region.

        Returns ``(labels, stats, centroids, n_labels)`` from
        :func:`cv2.connectedComponentsWithStats`, or ``(None, None, None, 0)``
        when nothing is found.
        """
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        except cv2.error as exc:
            logger.warning("Freckle detection: grayscale conversion failed: %s", exc)
            return None, None, None, 0

        ksize = 11
        local_mean = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        deviation = local_mean - gray  # positive = darker than surroundings
        relative_deviation = deviation / np.maximum(local_mean, 1.0)
        candidates = (
            relative_deviation > self._DARK_SPOT_RELATIVE_DEVIATION
        ).astype(np.uint8) * 255

        if skin_mask is not None:
            skin_binary = (skin_mask > 0.3).astype(np.uint8) * 255
            candidates = cv2.bitwise_and(candidates, skin_binary)

        # Redness channel: catch inflamed spots that are not necessarily dark.
        try:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        except cv2.error as exc:
            logger.warning("Freckle detection: HSV conversion failed: %s", exc)
            hsv = None
        if hsv is not None:
            red1 = cv2.inRange(hsv, (0, 60, 50), (15, 255, 255))
            red2 = cv2.inRange(hsv, (165, 60, 50), (180, 255, 255))
            red_combined = cv2.bitwise_or(red1, red2)
            red_blemish = cv2.bitwise_and(red_combined, candidates)
            candidates = cv2.bitwise_or(candidates, red_blemish)

        k_open = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidates, connectivity=8
        )
        return labels, stats, centroids, n_labels

    def _classify_anomaly(
        self,
        lab: np.ndarray,
        comp_mask: np.ndarray,
        a_median: float,
        a_std: float,
        l_median: float,
        l_std: float,
    ) -> Tuple[str, float, str, float, Tuple[float, float], Tuple[int, int, int, int]]:
        """Score one component into freckle/beauty_mark/blemish/noise.

        Returns (classification, confidence, reason, area, centroid, bbox).
        """
        pixels = lab[comp_mask]
        area = float(pixels.shape[0])
        if area == 0:
            return "noise", 0.0, "empty", 0.0, (0.0, 0.0), (0, 0, 0, 0)

        l_mean = float(np.mean(pixels[:, 0]))
        a_mean = float(np.mean(pixels[:, 1]))
        b_mean = float(np.mean(pixels[:, 2]))
        l_component_std = float(np.std(pixels[:, 0]))
        a_norm = (a_mean - a_median) / (a_std + 1e-6)
        # A nearly uniform face crop has a vanishing L standard deviation;
        # retain a scale tied to the face's own luminance in that case so a
        # small component cannot inflate itself into a many-sigma outlier.
        l_scale = max(l_std, l_median * 0.10, 1.0)
        l_norm = (l_mean - l_median) / l_scale
        chroma = float(np.sqrt((a_mean - 128.0) ** 2 + (b_mean - 128.0) ** 2))

        scores = {"freckle": 0.0, "beauty_mark": 0.0, "blemish": 0.0, "noise": 0.0}
        reasons: dict = {k: [] for k in scores}

        fl, fh = self._FRECKLE_SIZE
        bl, bh = self._BEAUTY_SIZE
        cl, ch = self._BLEMISH_SIZE

        # Freckle: small, reddish relative to skin, flat (low internal variance).
        if fl <= area <= fh:
            scores["freckle"] += 0.5
            reasons["freckle"].append(f"size={area:.0f}")
        if a_norm > 0.3:
            scores["freckle"] += 0.45
            reasons["freckle"].append(f"a_norm={a_norm:.2f}")
        # This was ``l_mean >= 70``. Absolute LAB-L changes with the subject's
        # skin tone, so freckles on dark skin were preserved as beauty marks.
        # Only use the component's margin from this face's skin distribution.
        if l_norm >= -2.0:
            scores["freckle"] += 0.05
            reasons["freckle"].append(f"L_norm={l_norm:.2f}")

        # Beauty mark: dark, larger, cool relative to skin, flat.
        if bl <= area <= bh:
            scores["beauty_mark"] += 0.2
            reasons["beauty_mark"].append(f"size={area:.0f}")
        if a_norm < -0.3:
            scores["beauty_mark"] += 0.4
            reasons["beauty_mark"].append(f"a_norm={a_norm:.2f}")
        if l_norm < -2.0:
            scores["beauty_mark"] += 0.5
            reasons["beauty_mark"].append(f"L_norm={l_norm:.2f}")

        # Blemish: inflamed -> high chroma AND high internal variance (uneven).
        if cl <= area <= ch:
            scores["blemish"] += 0.2
            reasons["blemish"].append(f"size={area:.0f}")
        if chroma > 25.0:
            scores["blemish"] += 0.3
            reasons["blemish"].append(f"chroma={chroma:.1f}")
        if l_component_std > 8.0:
            scores["blemish"] += 0.5
            reasons["blemish"].append(f"L_std={l_component_std:.1f}")

        # Noise: tiny.
        if area < self._NOISE_MAX_AREA:
            scores["noise"] = 0.9
            reasons["noise"].append(f"size={area:.0f}<{self._NOISE_MAX_AREA}")

        best = max(scores, key=scores.get)
        confidence = float(min(1.0, scores[best]))
        return best, confidence, " ".join(reasons[best]), area, (0.0, 0.0), (0, 0, 0, 0)

    def classify_anomalies(
        self,
        img_bgr: np.ndarray,
        face_mask: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.6,
    ) -> List[FreckleClassification]:
        """Detect and classify all anomalies, sorted by confidence descending.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            face_mask: (H, W) float/binary mask restricting detection to skin.
            confidence_threshold: Drop classifications below this confidence.

        Returns:
            List of :class:`FreckleClassification`, highest confidence first.
        """
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"img_bgr must be (H,W,3), got shape {img_bgr.shape}")

        skin_mask = normalize_mask(face_mask)
        if skin_mask is None:
            skin_mask = np.ones(img_bgr.shape[:2], dtype=np.float32)

        lab = self._to_lab(img_bgr)

        skin_pixels = lab[skin_mask > 0.3] if skin_mask.max() > 0 else lab.reshape(-1, 3)
        if skin_pixels.size == 0:
            a_median = 128.0
            a_std = 1.0
            l_median = 128.0
            l_std = 1.0
        else:
            a_median = float(np.median(skin_pixels[:, 1]))
            a_std = float(np.std(skin_pixels[:, 1]))
            l_median = float(np.median(skin_pixels[:, 0]))
            l_std = float(np.std(skin_pixels[:, 0]))

        labels, stats, centroids, n_labels = self._detect_components(img_bgr, skin_mask)
        if labels is None or n_labels <= 1:
            return []

        classifications: List[FreckleClassification] = []
        count = 0
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self._MIN_AREA:
                continue
            count += 1
            if count > self._MAX_COMPONENTS:
                logger.warning(
                    "Freckle: %d+ components detected; capping at %d",
                    self._MAX_COMPONENTS,
                    self._MAX_COMPONENTS,
                )
                break
            comp_mask = labels == i
            cls, conf, reason, _area, _, _ = self._classify_anomaly(
                lab, comp_mask, a_median, a_std, l_median, l_std
            )
            cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            classifications.append(
                FreckleClassification(
                    anomaly_id=len(classifications),
                    classification=cls,
                    confidence=conf,
                    reason=reason,
                    centroid=(cx, cy),
                    area=float(area),
                    bbox=(x, y, w, h),
                )
            )
            del comp_mask

        kept = [c for c in classifications if c.confidence >= confidence_threshold]
        kept.sort(key=lambda c: c.confidence, reverse=True)
        return kept

    def remove(
        self,
        img_bgr: np.ndarray,
        face_mask: Optional[np.ndarray] = None,
        freckle_removal: float = 0.0,
        freckle_preserve_mask: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.7,
        mole_mask: Optional[np.ndarray] = None,
        heal_engine: str = "telea",
    ) -> np.ndarray:
        """Remove freckles while preserving beauty marks via inpainting.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 [0,255] BGR image.
            face_mask: (H, W) float/binary skin mask.
            freckle_removal: 0-100 strength. <=0 is a byte-identical no-op.
            freckle_preserve_mask: Optional (H, W) mask of pixels to never heal.
            confidence_threshold: Min classification confidence to act on.
            mole_mask: Optional R10 mole mask (uint8/float) — never heal these.

        Returns:
            Image matching input dtype with freckles inpainted.
        """
        if freckle_removal <= 0:
            return img_bgr.copy()

        if img_bgr.dtype == np.float32:
            return apply_u8_op_float(
                img_bgr,
                self.remove,
                face_mask,
                freckle_removal,
                freckle_preserve_mask,
                confidence_threshold,
                mole_mask,
                heal_engine,
            )

        classifications = self.classify_anomalies(
            img_bgr, face_mask=face_mask, confidence_threshold=confidence_threshold
        )
        if not classifications:
            return img_bgr.copy()

        h, w = img_bgr.shape[:2]
        preserve_mask = np.zeros((h, w), dtype=np.uint8)
        if freckle_preserve_mask is not None:
            user = normalize_mask(freckle_preserve_mask)
            preserve_mask = (user > 0.1).astype(np.uint8) * 255
        if mole_mask is not None:
            mm = normalize_mask(mole_mask)
            preserve_mask = np.maximum(
                preserve_mask, (mm > 0.1).astype(np.uint8) * 255
            )

        for c in classifications:
            if c.classification == "beauty_mark":
                cx, cy = int(round(c.centroid[0])), int(round(c.centroid[1]))
                radius = int(np.sqrt(c.area / np.pi)) + 3
                cv2.circle(preserve_mask, (cx, cy), radius, 255, -1)

        strength01 = min(1.0, max(0.0, freckle_removal / 100.0))
        removal_float = np.zeros((h, w), dtype=np.float32)
        for c in classifications:
            if c.classification != "freckle":
                continue
            cx, cy = int(round(c.centroid[0])), int(round(c.centroid[1]))
            radius = int(np.sqrt(c.area / np.pi)) + 2
            intensity = strength01 * c.confidence
            cv2.circle(removal_float, (cx, cy), radius, intensity, -1)

        # cv2.inpaint is binary, so the continuous intensity is thresholded into a
        # heal mask. The gate is intentionally low (not 0.5) so the slider is
        # responsive across most of its 0-100 range: a freckle is healed once
        # ``strength01 * confidence`` clears the gate, which happens at lower
        # strengths for high-confidence freckles and higher strengths for weaker
        # ones — giving a natural progressive gradient instead of a dead lower half.
        _REMOVAL_GATE = 0.12
        removal_mask = (removal_float >= _REMOVAL_GATE).astype(np.uint8) * 255
        removal_mask = cv2.bitwise_and(removal_mask, cv2.bitwise_not(preserve_mask))

        if removal_mask.sum() == 0:
            return img_bgr.copy()

        face_width = float(w)
        scale = face_width / 500.0
        inpaint_r = max(int(3 * scale), 2)
        blend_k = max(int(7 * scale), 3) | 1
        if heal_engine == "telea":
            return inpaint_and_blend(
                img_bgr, removal_mask, inpaint_r, cv2.INPAINT_TELEA, blend_k
            )
        if heal_engine == "patchmatch":
            from .heal import heal_region

            return heal_region(
                img_bgr,
                removal_mask,
                method="patchmatch",
                source_mask=face_mask,
                patch_size=7,
                iterations=5,
                seamless=False,
            )
        raise ValueError("heal_engine must be 'telea' or 'patchmatch'")
