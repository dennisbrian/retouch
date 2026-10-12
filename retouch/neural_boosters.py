"""Neural boosters for advanced stray hair and defect segmentation.

PARKED: This module provides infrastructure for stray hair and defect segmentation
neural networks, but is currently disabled pending A1 (competitive benchmark) evidence.

Owner decision 2026-07-03: stay classical-only for now. Generative capabilities
(AI pore-detail synthesis, identity-preserving diffusion refinement) are explicitly
parked here, not scope-crept into S6/C5. Revisit once A1 evidence exists.

See MASTER_PLAN.md line 112 (Phase 6, A4) for status and context.

Current implementation: placeholders that return empty masks, no processing cost.
They are not exposed as production GUI/CLI controls and all built-in recipe
defaults are zero. Old sessions may still carry the fields; the engine emits a
warning when such a requested value is encountered and leaves the image
unchanged.
When real implementations are added, keep them classical (segmentation/masking),
not generative (synthesis/diffusion).

Future expected interface (when enabled):
- NeuralBooster protocol with detect(img) -> mask
- StrayHairSegmenter: thin line detection on LAB L, flow-alignment, silhouette
- DefectSegmenter: blemish/pore/texture anomaly detection, confidence scores
Both return uint8 [0,255] masks, uint8/float32 [0,255] input accepted.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class NeuralBooster(ABC):
    """Abstract protocol for neural segmentation stages.

    Implementations should:
    - Accept uint8/float32 [0,255] BGR images
    - Return uint8 [0,255] masks (0=background, 255=detected region)
    - Be deterministic on same input
    - Handle variable image sizes gracefully
    - Raise ValueError for invalid inputs (unsupported dtype/shape)
    """

    @abstractmethod
    def detect(self, img: np.ndarray) -> np.ndarray:
        """Detect and segment target features.

        Parameters
        ----------
        img : np.ndarray
            Input BGR image, uint8 [0,255] or float32 [0,255].
            Shape (h, w, 3), contiguous.

        Returns
        -------
        np.ndarray
            Segmentation mask, uint8 [0,255].
            Shape (h, w), same size as input.
            0 = background/not target; 255 = target; intermediate values OK.
        """
        pass


class StrayHairSegmenter(NeuralBooster):
    """Detect and segment stray hairs (flyaway fibers outside main hair mass).

    PARKED: Placeholder that returns empty mask. When real implementation is added:
    - Multi-scale thin-line detection on LAB L (thickness <= 4px)
    - Flow-alignment to hair strand directions
    - Silhouette extension from hairline
    - Eye/brow exclusion
    - Optional skin-mask face-crossing extension

    Rationale for classical approach: flyaway detection is geometric (thin lines
    at specific locations), not semantic — segmentation networks overkill and
    costly at 6K resolution. A hand-tuned morphological pipeline (matching S5's
    ridge detection / H0's flow field) is more interpretable and parallelizable.
    """

    def __init__(self):
        """Initialize placeholder segmenter."""
        self.enabled = False

    def detect(self, img: np.ndarray) -> np.ndarray:
        """Return empty mask (parked).

        When enabled:
        - Validates input dtype/shape
        - Detects thin lines on LAB L via multi-scale black-hat
        - Filters by flow alignment and silhouette proximity
        - Returns uint8 [0,255] mask

        Parameters
        ----------
        img : np.ndarray
            BGR image, uint8 or float32, shape (h, w, 3).

        Returns
        -------
        np.ndarray
            Empty uint8 [0,255] mask, shape (h, w).
        """
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError(
                f"StrayHairSegmenter.detect() expects uint8 or float32, got {img.dtype}"
            )
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(
                f"StrayHairSegmenter.detect() expects shape (h, w, 3), got {img.shape}"
            )

        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)


class DefectSegmenter(NeuralBooster):
    """Detect and segment skin defects (blemishes, pores, texture anomalies).

    PARKED: Placeholder that returns empty mask. When real implementation is added:
    - Multi-class detection: blemishes, enlarged pores, fine texture anomalies
    - Confidence scoring per class
    - Skin-tone normalization (handle dark/light/medium skin equally)
    - False-positive rejection via local context (avoid normal texture)
    - Export per-class masks with confidence

    Rationale for classical approach: defect detection is primarily based on
    local color/chroma anomalies (existing S3/S4 logic handles most cases).
    A neural classifier could improve false-positive rates on ambiguous pixels,
    but only if A1 evidence shows > 2-3% improvement over classical thresholds.
    Keep interpretable (parametric) by default, add neural refinement as opt-in.
    """

    def __init__(self):
        """Initialize placeholder segmenter."""
        self.enabled = False

    def detect(self, img: np.ndarray) -> np.ndarray:
        """Return empty mask (parked).

        When enabled:
        - Validates input dtype/shape
        - Detects chroma/saturation anomalies via learned attention
        - Filters false positives via local texture baseline
        - Returns uint8 [0,255] confidence mask

        Parameters
        ----------
        img : np.ndarray
            BGR image, uint8 or float32, shape (h, w, 3).

        Returns
        -------
        np.ndarray
            Empty uint8 [0,255] mask, shape (h, w).
            0 = normal skin; 255 = high-confidence defect.
        """
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError(
                f"DefectSegmenter.detect() expects uint8 or float32, got {img.dtype}"
            )
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(
                f"DefectSegmenter.detect() expects shape (h, w, 3), got {img.shape}"
            )

        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
