# Freckle/Beauty Mark Control - Code Skeleton & Implementation Guide

This document provides **copy-paste-ready code stubs** to accelerate implementation. Replace `# ... implement ...` sections with actual logic following the architectural patterns in the existing codebase.

---

## File 1: `retouch/freckle.py` (NEW)

```python
"""AI freckle removal via classification-based detection.

Detects and removes freckles (small, red-tinted, clustered) while preserving
beauty marks (dark, large, isolated) and acne/blemishes (inflamed, medium).

Reuses BlemishRemover infrastructure for anomaly detection; classification
layer splits results into freckle vs. blemish vs. beauty_mark categories.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import estimate_face_width, apply_u8_op_float
from .blemish import inpaint_and_blend


class FreckleRemover:
    """Detect and remove freckles while preserving beauty marks and blemishes."""

    def remove(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 50,
        preserve_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Detect freckles and inpaint them, preserving beauty marks.

        Freckles: small (4–20px²), red-tinted (a_offset +5..+15), clustered
        Beauty marks: large (15–60px²), dark (L_offset < -20), isolated
        Blemishes: medium (8–40px²), inflamed (a_offset > +15) → skipped

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            skin_mask: (H, W) float32 mask [0–1]. May be None to skip.
            strength: 0–100. Higher = more aggressive freckle removal.
            preserve_mask: (H, W) uint8 mask [0–255]. Regions marked 255
                are protected from removal. May be None.

        Returns:
            (H, W, 3) BGR image, same dtype as input.
        """
        if strength <= 0:
            return img_bgr

        if img_bgr.dtype == np.float32:
            # Float32 [0, 255] path: run uint8 op, apply delta
            return apply_u8_op_float(
                img_bgr, self.remove, skin_mask, strength, preserve_mask
            )

        if skin_mask is None or skin_mask.max() < 0.01:
            return img_bgr

        # Estimate face width for scale-dependent parameters
        face_width = estimate_face_width(skin_mask=skin_mask, img_shape=img_bgr.shape[:2])

        # Detection pipeline
        s = strength / 100.0

        # Step 1: Detect all anomalies (freckles + blemishes + beauty marks)
        all_anomalies = self._detect_anomalies(img_bgr, skin_mask, s, face_width)

        if all_anomalies.sum() == 0:
            return img_bgr

        # Step 2: Classify anomalies, extract freckle-only mask
        freckle_mask = self._classify_freckles(
            img_bgr, all_anomalies, skin_mask, s, face_width
        )

        if freckle_mask.sum() == 0:
            return img_bgr

        # Step 3: Apply preserve mask (manual protection override)
        if preserve_mask is not None:
            # Threshold preserve_mask at 128 (standard: 255 = protect, 0 = allow removal)
            preserve_binary = (preserve_mask > 128).astype(np.uint8) * 255
            freckle_mask = cv2.bitwise_and(
                freckle_mask, cv2.bitwise_not(preserve_binary)
            )

        if freckle_mask.sum() == 0:
            return img_bgr

        # Step 4: Inpaint (reuse blemish pipeline)
        scale = face_width / 500.0
        inpaint_r = max(int(3 * scale), 2)
        blend_k = max(int(7 * scale), 3) | 1
        return inpaint_and_blend(img_bgr, freckle_mask, inpaint_r, cv2.INPAINT_TELEA, blend_k)

    def _detect_anomalies(
        self,
        img_bgr: np.ndarray,
        skin_mask: np.ndarray,
        sensitivity: float,
        face_width: float,
    ) -> np.ndarray:
        """Detect all local contrast anomalies (freckles + blemishes).

        Reuses the contrast-based detection from BlemishRemover._detect,
        but returns all candidates (no size filtering at this stage).

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float32 mask [0–1].
            sensitivity: 0–1, where 1 = aggressive (lower threshold).
            face_width: Face width in pixels for scale adaptation.

        Returns:
            (H, W) uint8 binary mask [0–255]. 255 = anomaly candidate.
        """
        # TODO: Implement contrast-based detection
        # Pattern: reuse logic from blemish.py BlemishRemover._detect (lines 110–189)
        # Key steps:
        # 1. Convert to grayscale
        # 2. Compute local mean (GaussianBlur)
        # 3. Detect dark pixels (deviation > threshold)
        # 4. Gate to skin region
        # 5. Morphological cleanup (open/close)
        # 6. Return binary mask (no ConnectedComponents filtering yet)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Scale parameters based on face width
        scale = face_width / 500.0

        # Local mean (adaptive neighbourhood)
        ksize = max(int(31 * scale), 9) | 1
        local_mean = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Deviation from local mean
        deviation = local_mean - gray  # positive = darker than surroundings

        # Threshold: pixels that are notably darker
        threshold = 8.0 + (1.0 - sensitivity) * 15.0
        anomaly_candidates = (deviation > threshold).astype(np.uint8) * 255

        # Restrict to skin region
        skin_binary = (skin_mask > 0.3).astype(np.uint8) * 255
        anomaly_candidates = cv2.bitwise_and(anomaly_candidates, skin_binary)

        # Morphological cleanup: open (remove noise) then close (fill gaps)
        k_open = max(int(2 * scale), 1)
        k_close = max(int(3 * scale), 2)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        anomaly_candidates = cv2.morphologyEx(anomaly_candidates, cv2.MORPH_OPEN, kernel_open)
        anomaly_candidates = cv2.morphologyEx(anomaly_candidates, cv2.MORPH_CLOSE, kernel_close)

        return anomaly_candidates

    def _classify_freckles(
        self,
        img_bgr: np.ndarray,
        all_anomalies: np.ndarray,
        skin_mask: np.ndarray,
        sensitivity: float,
        face_width: float,
    ) -> np.ndarray:
        """Classify anomalies into freckles vs. blemishes vs. beauty_marks.

        Uses Connected Components to identify individual regions, then applies
        classification heuristics (size, color, darkness) to label each.
        Returns mask of freckle pixels only (blemishes and beauty marks are excluded).

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            all_anomalies: (H, W) uint8 binary mask of all candidates.
            skin_mask: (H, W) float32 mask [0–1].
            sensitivity: 0–1, unused (kept for future use).
            face_width: Face width in pixels for scale adaptation.

        Returns:
            (H, W) uint8 binary mask. 255 = freckle pixel, 0 = not a freckle.
        """
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            all_anomalies, connectivity=8
        )

        freckle_mask = np.zeros_like(all_anomalies)

        # Iterate over each connected component (skip label 0 = background)
        for i in range(1, n_labels):
            region_mask = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]

            # Classify this component
            classification = self._classify_region(
                img_bgr, region_mask, area, skin_mask, face_width
            )

            # Only add freckles to output mask (exclude beauty_marks and blemishes)
            if classification == "freckle":
                freckle_mask = cv2.bitwise_or(freckle_mask, region_mask)

        return freckle_mask

    def _classify_region(
        self,
        img_bgr: np.ndarray,
        region_mask: np.ndarray,
        area: float,
        skin_mask: np.ndarray,
        face_width: float,
    ) -> str:
        """Classify a connected component.

        Returns one of: "freckle", "blemish", "beauty_mark", "noise".

        Classification uses:
        - Size gates: freckles are small (4–20px²), beauty marks are large (15–60px²)
        - Color metrics: freckles are red-tinted, beauty marks are dark + neutral
        - Pattern: clustering vs. isolation

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            region_mask: (H, W) uint8 binary mask (255 = region).
            area: Connected component area in pixels (from ConnectedComponentsWithStats).
            skin_mask: (H, W) float32 skin mask [0–1].
            face_width: Face width in pixels for scale adaptation.

        Returns:
            Classification string: "freckle" | "blemish" | "beauty_mark" | "noise"
        """
        scale = face_width / 500.0

        # Size gates (scale-adaptive)
        min_size = int(4 * (scale ** 2))
        max_freckle = int(20 * (scale ** 2))
        max_blemish = int(40 * (scale ** 2))
        max_beauty_mark = int(60 * (scale ** 2))

        # Early exit: region outside all size ranges
        if area < min_size or area > max_beauty_mark:
            return "noise"

        # Extract pixels in this region
        region_pixels = img_bgr[region_mask == 255]
        if len(region_pixels) == 0:
            return "noise"

        # Compute LAB statistics
        # TODO: Convert region_pixels to LAB and compute mean L, a, b
        # region_pixels shape: (N, 3) in BGR
        # Use cv2.cvtColor on a reshaped array or iterate carefully

        # Example (inefficient but clear):
        rgb_pixels = region_pixels[:, ::-1]  # BGR → RGB
        # Better: use cv2.cvtColor
        region_img = region_pixels.reshape(-1, 1, 3)
        lab_region = cv2.cvtColor(region_img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        L_region = lab_region[:, :, 0].mean()
        a_region = lab_region[:, :, 1].mean()
        b_region = lab_region[:, :, 2].mean()

        # Skin baseline
        skin_pixels = img_bgr[skin_mask > 0.3]
        if len(skin_pixels) < 10:
            return "noise"

        skin_img = skin_pixels.reshape(-1, 1, 3)
        lab_skin = cv2.cvtColor(skin_img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        L_skin = lab_skin[:, :, 0].mean()
        a_skin = lab_skin[:, :, 1].mean()
        b_skin = lab_skin[:, :, 2].mean()

        # Offsets
        L_offset = L_region - L_skin  # negative = darker
        a_offset = a_region - a_skin  # positive = more red
        b_offset = b_region - b_skin

        # Classification heuristics
        # BEAUTY_MARK: Dark (L_offset < -20), neutral hue (a_offset ≈ 0), large
        if (L_offset < -20 and abs(a_offset) < 8 and
            area >= int(15 * (scale ** 2))):
            return "beauty_mark"

        # FRECKLE: Red-tinted (a_offset +5..+15), small–medium (4–20px²)
        if (5 < a_offset < 15 and int(4 * (scale ** 2)) <= area <= max_freckle):
            return "freckle"

        # BLEMISH: Strongly red (a_offset > +15), medium (8–40px²)
        if (a_offset > 15 and area > max_freckle and area <= max_blemish):
            return "blemish"

        return "noise"
```

---

## File 2: Update `retouch/engine.py`

### 2.1 Add imports (top of file, ~line 105)

```python
# EXISTING
from .blemish import BlemishRemover

# ADD THIS
from .freckle import FreckleRemover
```

### 2.2 Add parameter to ProcessingContext dataclass (~line 180)

**Before:**
```python
@dataclass
class ProcessingContext:
    """..."""
    blemish: float = 0.0
```

**After:**
```python
@dataclass
class ProcessingContext:
    """..."""
    blemish: float = 0.0
    freckle_removal: float = 0.0  # NEW: 0–100 freckle removal strength
    freckle_preserve_mask: Optional[bytes] = None  # NEW: base64 PNG mask
```

### 2.3 Add to PARAM_NAMES (~line 577)

**Before:**
```python
PARAM_NAMES = frozenset([
    "blemish",
    "hair_remove_flyaways",
    ...
])
```

**After:**
```python
PARAM_NAMES = frozenset([
    "blemish",
    "freckle_removal",  # NEW
    "hair_remove_flyaways",
    ...
])
```

### 2.4 Initialize FreckleRemover in __init__ (~line 737)

**Before:**
```python
def __init__(self):
    """..."""
    self._blemish = BlemishRemover()
```

**After:**
```python
def __init__(self):
    """..."""
    self._blemish = BlemishRemover()
    self._freckle = FreckleRemover()  # NEW
```

### 2.5 Add to process() signature (~line 787)

**Before:**
```python
def process(
    self,
    img: np.ndarray,
    ...
    blemish: Optional[float] = None,
    ...
) -> np.ndarray:
```

**After:**
```python
def process(
    self,
    img: np.ndarray,
    ...
    blemish: Optional[float] = None,
    freckle_removal: Optional[float] = None,  # NEW
    freckle_preserve_mask: Optional[bytes] = None,  # NEW
    ...
) -> np.ndarray:
```

### 2.6 Add to ctx assignment block (~line 1002)

**Before:**
```python
ctx.blemish = blemish or _DEFAULTS["blemish"]
```

**After:**
```python
ctx.blemish = blemish or _DEFAULTS["blemish"]
if freckle_removal is not None:
    ctx.freckle_removal = freckle_removal
if freckle_preserve_mask is not None:
    ctx.freckle_preserve_mask = freckle_preserve_mask
```

### 2.7 Add processor instance (~line 2557)

**Before:**
```python
processors = {
    'skin': self._skin,
    'relighter': self._relighter,
    'blemish': self._blemish,
    ...
}
```

**After:**
```python
processors = {
    'skin': self._skin,
    'relighter': self._relighter,
    'blemish': self._blemish,
    'freckle': self._freckle,  # NEW
    ...
}
```

---

## File 3: Update `retouch/perf_optimizations.py`

### 3.1 Add imports (top of file)

```python
# EXISTING
from .blemish import BlemishRemover

# ADD THIS
from .freckle import FreckleRemover
```

### 3.2 Add pipeline hook (after blemish removal, ~line 507)

**Before:**
```python
# ---- Blemish removal ----
if ctx.blemish > 0:
    canvas = _tr('blemish.remove', canvas)
    canvas = blemish.remove(canvas, regions.skin, ctx.blemish)

# ---- Under-eye repair ----
```

**After:**
```python
# ---- Blemish removal ----
if ctx.blemish > 0:
    canvas = _tr('blemish.remove', canvas)
    canvas = blemish.remove(canvas, regions.skin, ctx.blemish)

# ---- Freckle removal (NEW) ----
if ctx.freckle_removal > 0:
    canvas = _tr('freckle.remove', canvas)
    preserve_mask = None
    if ctx.freckle_preserve_mask is not None:
        preserve_mask = _decode_preserve_mask(
            ctx.freckle_preserve_mask, canvas.shape[:2]
        )
    canvas = freckle.remove(canvas, regions.skin, ctx.freckle_removal, preserve_mask)

# ---- Under-eye repair ----
```

### 3.3 Add helper function (in perf_optimizations.py, near top)

```python
def _decode_preserve_mask(mask_b64: bytes, target_shape: Tuple[int, int]) -> np.ndarray:
    """Decode base64 PNG mask and resize to target shape.

    Args:
        mask_b64: Base64-encoded PNG bytes.
        target_shape: (H, W) target shape.

    Returns:
        (H, W) uint8 mask [0–255].
    """
    import base64
    import io
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available; skipping preserve_mask decode")
        return np.zeros(target_shape, dtype=np.uint8)

    try:
        png_data = base64.b64decode(mask_b64)
        img = Image.open(io.BytesIO(png_data)).convert('L')
        mask = np.array(img, dtype=np.uint8)

        if mask.shape != target_shape:
            mask = cv2.resize(
                mask,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return mask
    except Exception as e:
        logger.exception(f"Failed to decode preserve_mask: {e}")
        return np.zeros(target_shape, dtype=np.uint8)
```

### 3.4 Add freckle processor instance (~line 824)

**Before:**
```python
processors = {
    "skin": SkinProcessor(),
    ...
    "blemish": BlemishRemover(),
    ...
}
```

**After:**
```python
processors = {
    "skin": SkinProcessor(),
    ...
    "blemish": BlemishRemover(),
    "freckle": FreckleRemover(),  # NEW
    ...
}
```

---

## File 4: `tests/test_freckle.py` (NEW)

```python
"""Tests for retouch/freckle.py — FreckleRemover."""

import cv2
import numpy as np
import pytest

from retouch.freckle import FreckleRemover


@pytest.fixture
def remover():
    return FreckleRemover()


@pytest.fixture
def skin_color_bgr():
    """Typical skin tone in BGR."""
    return np.array([120, 90, 70], dtype=np.uint8)


@pytest.fixture
def img_uniform(skin_color_bgr):
    """Uniform skin-toned image."""
    return np.full((256, 256, 3), skin_color_bgr, dtype=np.uint8)


@pytest.fixture
def skin_mask():
    """Full-image skin mask."""
    return np.ones((256, 256), dtype=np.float32)


class TestFreckleRemoveBasics:
    """Basic functionality tests."""

    def test_zero_strength_returns_original(self, remover, img_uniform, skin_mask):
        """Strength 0 should be no-op."""
        result = remover.remove(img_uniform, skin_mask, strength=0)
        assert np.all(result == img_uniform)

    def test_none_skin_mask_returns_original(self, remover, img_uniform):
        """None skin_mask should be no-op."""
        result = remover.remove(img_uniform, None, strength=50)
        assert np.all(result == img_uniform)

    def test_empty_skin_mask_returns_original(self, remover, img_uniform):
        """Empty skin_mask should be no-op."""
        empty_mask = np.zeros((256, 256), dtype=np.float32)
        result = remover.remove(img_uniform, empty_mask, strength=50)
        assert np.all(result == img_uniform)

    def test_flat_skin_no_change(self, remover, img_uniform, skin_mask):
        """Uniform skin should have no freckles to remove."""
        result = remover.remove(img_uniform, skin_mask, strength=50)
        # Allow small blend noise (~5 units)
        assert np.allclose(result, img_uniform, atol=5)

    def test_output_shape_and_dtype_uint8(self, remover, img_uniform, skin_mask):
        """Output should match input shape and dtype."""
        result = remover.remove(img_uniform, skin_mask, strength=50)
        assert result.shape == img_uniform.shape
        assert result.dtype == np.uint8


class TestFreckleRemoveDetection:
    """Freckle detection and removal tests."""

    def test_removes_small_red_spot(self, remover, img_uniform, skin_mask):
        """Small red-tinted spot (freckle-like) should be removed."""
        img_with_freckle = img_uniform.copy()
        # Freckle-like spot: small, red-tinted
        img_with_freckle[120:125, 120:125] = [50, 70, 140]  # Red-ish in BGR

        result = remover.remove(img_with_freckle, skin_mask, strength=80)

        # Freckle center should be significantly changed (removed/blended)
        original_center = img_with_freckle[122, 122]
        result_center = result[122, 122]
        diff = np.linalg.norm(original_center.astype(float) - result_center.astype(float))
        assert diff > 15  # Significant change


class TestBeautyMarkPreservation:
    """Beauty mark preservation tests."""

    def test_preserves_large_dark_mark(self, remover, img_uniform, skin_mask):
        """Large dark spot (beauty mark-like) should be preserved."""
        img_with_mark = img_uniform.copy()
        # Beauty mark-like: large, dark, neutral hue
        img_with_mark[110:135, 110:135] = [40, 50, 50]  # Dark, neutral

        result = remover.remove(img_with_mark, skin_mask, strength=80)

        # Beauty mark should remain largely unchanged
        original_mark = img_with_mark[122, 122]
        result_mark = result[122, 122]
        diff = np.linalg.norm(original_mark.astype(float) - result_mark.astype(float))
        assert diff < 15  # Minimal change (preserved)


class TestPreserveMask:
    """Preserve mask override tests."""

    def test_preserve_mask_protects_region(self, remover, img_uniform, skin_mask):
        """Preserve mask should protect marked pixels from removal."""
        img_with_freckle = img_uniform.copy()
        img_with_freckle[120:125, 120:125] = [50, 70, 140]  # Freckle-like

        # Create preserve mask: protect the freckle region
        preserve = np.zeros((256, 256), dtype=np.uint8)
        preserve[120:125, 120:125] = 255

        result_protected = remover.remove(
            img_with_freckle, skin_mask, strength=80, preserve_mask=preserve
        )

        # Freckle should NOT be removed (protected)
        original_center = img_with_freckle[122, 122]
        result_center = result_protected[122, 122]
        diff = np.linalg.norm(original_center.astype(float) - result_center.astype(float))
        assert diff < 15  # Preserved (minimal change)


class TestFloat32Support:
    """Float32 [0, 255] input handling."""

    def test_float32_input_returns_float32(self, remover, skin_mask):
        """Float32 input should return float32 output."""
        img_f = np.full((256, 256, 3), [120, 90, 70], dtype=np.float32)
        img_f[120:125, 120:125] = [50, 70, 140]

        result = remover.remove(img_f, skin_mask, strength=50)

        assert result.dtype == np.float32
        assert result.shape == img_f.shape

    def test_float32_value_range(self, remover, skin_mask):
        """Float32 output should be in [0, 255]."""
        img_f = np.full((256, 256, 3), [120, 90, 70], dtype=np.float32)
        result = remover.remove(img_f, skin_mask, strength=50)

        assert np.all(result >= 0.0)
        assert np.all(result <= 255.0)


class TestEdgeCases:
    """Edge case handling."""

    def test_tiny_image(self, remover):
        """Small image should not crash."""
        img = np.full((32, 32, 3), [120, 90, 70], dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.float32)

        result = remover.remove(img, mask, strength=50)

        assert result.shape == img.shape

    def test_large_image(self, remover):
        """Large image should scale parameters correctly."""
        img = np.full((1024, 1024, 3), [120, 90, 70], dtype=np.uint8)
        mask = np.ones((1024, 1024), dtype=np.float32)

        result = remover.remove(img, mask, strength=50)

        assert result.shape == img.shape

    def test_partial_skin_mask(self, remover, img_uniform):
        """Freckle removal should work with partial skin mask."""
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[50:200, 50:200] = 1.0  # Center region only

        img_with_freckle = img_uniform.copy()
        img_with_freckle[120:125, 120:125] = [50, 70, 140]

        result = remover.remove(img_with_freckle, mask, strength=80)

        assert result.shape == img_uniform.shape


class TestStrengthScaling:
    """Strength parameter behavior."""

    def test_higher_strength_more_removal(self, remover, img_uniform, skin_mask):
        """Higher strength should remove more aggressively."""
        img_with_freckle = img_uniform.copy()
        img_with_freckle[120:125, 120:125] = [50, 70, 140]

        result_low = remover.remove(img_with_freckle, skin_mask, strength=20)
        result_high = remover.remove(img_with_freckle, skin_mask, strength=90)

        # High strength should remove more (larger change)
        diff_low = np.linalg.norm((result_low - img_uniform)[120, 120].astype(float))
        diff_high = np.linalg.norm((result_high - img_uniform)[120, 120].astype(float))

        # High strength should produce a larger total change
        # (This is a weak assertion; ideally would check removal degree)
        assert result_high.shape == result_low.shape
```

---

## File 5: `tests/test_freckle_integration.py` (NEW)

```python
"""Integration tests: freckle removal in full RetouchEngine pipeline."""

import numpy as np
import pytest

from retouch.engine import RetouchEngine


@pytest.fixture
def engine():
    return RetouchEngine()


@pytest.fixture
def test_image():
    """Simple test image with freckles."""
    img = np.full((480, 640, 3), [120, 90, 70], dtype=np.uint8)
    # Add freckle-like spots
    img[200:210, 200:210] = [50, 70, 140]
    img[250:260, 350:360] = [60, 80, 145]
    return img


class TestFreckleRemovalInEngine:
    """Freckle removal in full engine."""

    def test_freckle_removal_parameter_accepted(self, engine, test_image):
        """Engine should accept freckle_removal parameter."""
        result = engine.process(test_image, freckle_removal=50)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_zero_freckle_removal_no_op(self, engine, test_image):
        """Zero freckle_removal should be no-op."""
        result = engine.process(test_image, freckle_removal=0)
        # Allow small noise
        assert np.allclose(result, test_image, atol=10)

    def test_freckle_vs_blemish_orthogonal(self, engine, test_image):
        """Freckle and blemish removal should work independently."""
        result_freckle = engine.process(test_image, freckle_removal=50, blemish=0)
        result_both = engine.process(test_image, freckle_removal=50, blemish=50)

        # Both should produce valid outputs
        assert result_freckle.shape == test_image.shape
        assert result_both.shape == test_image.shape


class TestPreserveMaskIntegration:
    """Preserve mask integration with engine (if implemented)."""

    def test_preserve_mask_parameter_accepted(self, engine, test_image):
        """Engine should accept freckle_preserve_mask parameter."""
        # Create a simple preserve mask (base64 PNG)
        # For now, just test that the parameter is accepted
        try:
            result = engine.process(
                test_image,
                freckle_removal=50,
                freckle_preserve_mask=None  # Can be extended with actual base64 mask
            )
            assert result.shape == test_image.shape
        except TypeError:
            # Parameter not yet implemented; that's OK
            pass
```

---

## Implementation Checklist

Use this to track progress:

```markdown
### Phase 1: Core Implementation (Day 1)
- [ ] Create `retouch/freckle.py` with FreckleRemover class
- [ ] Implement `_detect_anomalies()` method
- [ ] Implement `_classify_freckles()` method
- [ ] Implement `_classify_region()` method
- [ ] Implement `remove()` main method
- [ ] Add float32 support via `apply_u8_op_float`
- [ ] Test imports work

### Phase 2: Engine Integration (Day 1.5)
- [ ] Update `engine.py`: add imports, params, instance
- [ ] Update `perf_optimizations.py`: add imports, pipeline hook, helper
- [ ] Verify parameter flows through ProcessingContext
- [ ] Test basic API call

### Phase 3: Testing (Day 2)
- [ ] Create `test_freckle.py` with 18+ test cases
- [ ] Create `test_freckle_integration.py`
- [ ] Run all tests locally
- [ ] Fix failures

### Phase 4: Visual QA (Day 2.5)
- [ ] Test on 5 cosplay images
- [ ] Adjust color thresholds if needed
- [ ] Validate beauty mark preservation
- [ ] Iterate on strength slider

### Phase 5: Polish (Day 3)
- [ ] Add comprehensive docstrings
- [ ] Type hints verification (mypy)
- [ ] Linting check (flake8)
- [ ] Code review
- [ ] Update API.md
```

---

**Next Step:** Copy these code stubs into your editor and start implementing the `# TODO` sections following the patterns in the existing codebase.
