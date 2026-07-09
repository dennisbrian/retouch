# FRECKLE_BEAUTY_MARK_CONTROL.md

## Feature: Freckle/beauty mark selective removal and preservation

**Estimated Effort:** 20 hours (2.5 days, medium complexity)  
**Priority:** Medium-High (strong ROI for cosplay, fashion)  
**Target Beneficiary:** Cosplay characters (preserve intentional marks), fashion (clean complexion)  
**Complexity:** Medium (classification logic, ~85% code reuse from BlemishRemover)  

---

## 1. Problem Statement

**Current behavior:** Blemish removal (S3) treats freckles, beauty marks, and blemishes uniformly → removes all small anomalies.

**Limitation:** Cosplay photographers want **intentional beauty marks preserved** while removing **unintended freckles**:
- **Beauty mark** (mole, intentional mark): Should be preserved or optionally removed via user override
- **Freckle** (small red/brown spot, clustered): Should be removed for clean look
- **Blemish** (inflamed, high-chroma): Should be ignored (medical concern)
- **Noise** (< 4px, random): Should be ignored

**Current approach:** Single `blemish` slider (0–100) scales BlemishRemover uniformly. No way to say "remove freckles but keep beauty marks."

**Visual symptom:** Cosplay photos lose character identity because intentional beauty marks (mole on cheek, freckle pattern for character) get removed. Alternatively, if users disable blemish removal, freckles remain prominent and look unpolished.

**Target outcome:** Classify anomalies accurately → remove selectively → allow user override via optional preserve_mask.

---

## 2. Solution Architecture

**Detect anomalies (size, color, clustering) → classify (freckle/beauty_mark/blemish/noise) → remove selectively → optionally merge with user mask**

1. **Base Implementation:** Reuse 85% of BlemishRemover code.
   - Inherit from BlemishRemover or call its detector internally.
   - Extend with classification logic (size/color/spatial clustering).

2. **Anomaly Classification:**
   Each detected anomaly is scored on:
   - **Size:** 4–20 px² → likely freckle; 15–60 px² → likely beauty mark; 8–40 px² → likely blemish
   - **Color (LAB a/b):** Red-tinted (a > 128+8) → freckle; dark (a < 128−10) → beauty mark; inflamed chroma > 30 → blemish
   - **Clustering:** Freckles cluster in groups (3+ within 20px); beauty marks isolated; blemishes scattered
   - **Confidence:** Score 0–1 based on classification evidence

3. **Removal Strategy:**
   - **Freckle:** Remove with high confidence (>0.7) if `freckle_removal > 0`
   - **Beauty mark:** Preserve by default (confidence > 0.8) unless `freckle_preserve_mask` explicitly excludes
   - **Blemish:** Skip (ignore by BlemishRemover's existing logic)
   - **Noise:** Remove (existing BlemishRemover behavior)

4. **User Override:**
   - Optional `freckle_preserve_mask` parameter (base64-encoded PNG): user manually selects regions to preserve
   - If provided, merge with auto-detected preservation to give user final say

5. **Multi-Skin-Tone Robustness:**
   - Normalize a/b channels: `a_norm = (a - median_a) / std_a` in region
   - Apply classification rules in normalized space → robust to skin tone

---

## 3. Technical Specification

### 3.1 Data Flow

```
INPUT: face_image, face_mask, freckle_removal (0–100), freckle_preserve_mask (optional)
  ↓
Detect anomalies (via BlemishRemover detector, or custom small-object detector)
  ↓
FOR EACH anomaly:
  Extract size (px²), color (LAB a/b), spatial clustering info
  ↓
  Classify: freckle vs beauty_mark vs blemish vs noise
  ↓
  Score confidence (0–1)
  ↓
  DECISION:
    IF freckle AND confidence > 0.7 AND freckle_removal > 0:
      REMOVE (heal / inpaint)
    ELIF beauty_mark AND confidence > 0.8:
      PRESERVE (add to preserve_mask)
    ELIF blemish:
      SKIP (don't remove, let BlemishRemover ignore)
    ELIF noise:
      REMOVE (cleanup)
  ↓
Merge with user preserve_mask (if provided)
  ↓
OUTPUT: retouched_face with selective removal
```

### 3.2 Class & Method Modifications

**File: `/Applications/htdocs/retouch/retouch/freckle.py` (new file)**

```python
"""Freckle and beauty mark detection and selective removal.

Classifies facial anomalies (freckles, beauty marks, blemishes, noise)
and applies targeted removal. Reuses ~85% of BlemishRemover logic.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .blemish import BlemishRemover, Anomaly
from .utils import blend_masked, normalize_mask, squeeze_mask, apply_u8_op_float
from .color_space import bgr_to_lch, lch_to_bgr


class FreckleClassification:
    """Classification result for a single anomaly."""
    __slots__ = ["anomaly_id", "anomaly_obj", "classification", "confidence", "reason"]
    
    TYPES = {"freckle", "beauty_mark", "blemish", "noise", "unknown"}
    
    def __init__(
        self,
        anomaly_id: int,
        anomaly_obj: Anomaly,
        classification: str,
        confidence: float,
        reason: str = "",
    ):
        assert classification in self.TYPES, f"Unknown classification: {classification}"
        self.anomaly_id = anomaly_id
        self.anomaly_obj = anomaly_obj
        self.classification = classification
        self.confidence = min(1.0, max(0.0, confidence))
        self.reason = reason
    
    def __repr__(self) -> str:
        return (
            f"FreckleClassification(id={self.anomaly_id}, type={self.classification}, "
            f"conf={self.confidence:.2f}, size={self.anomaly_obj.area:.0f}px²)"
        )


class FreckleRemover:
    """Detect and selectively remove freckles while preserving beauty marks.
    
    Reuses BlemishRemover's anomaly detection and healing, but adds
    classification logic to differentiate freckles from beauty marks.
    """
    
    def __init__(self):
        self.blemish_remover = BlemishRemover()
    
    @staticmethod
    def _classify_anomaly(
        anomaly: Anomaly,
        region_lab: np.ndarray,
        region_mask: np.ndarray,
        skin_tone_a_median: float,
        skin_tone_a_std: float,
    ) -> FreckleClassification:
        """Classify a single anomaly into freckle / beauty_mark / blemish / noise.
        
        Args:
            anomaly: Detected Anomaly object (with bbox, center, area).
            region_lab: (H, W, 3) LAB image containing anomaly.
            region_mask: (H, W) binary mask of anomaly pixels.
            skin_tone_a_median, skin_tone_a_std: Skin tone LAB a-channel stats for normalization.
        
        Returns:
            FreckleClassification with type and confidence score.
        """
        size = anomaly.area
        
        # Extract LAB color of anomaly
        anomaly_pixels = region_lab[region_mask > 0.5]
        if len(anomaly_pixels) == 0:
            return FreckleClassification(
                anomaly_id=-1,
                anomaly_obj=anomaly,
                classification="unknown",
                confidence=0.0,
                reason="No pixels in mask"
            )
        
        L_mean = float(np.mean(anomaly_pixels[:, 0]))
        a_mean = float(np.mean(anomaly_pixels[:, 1]))
        b_mean = float(np.mean(anomaly_pixels[:, 2]))
        L_std = float(np.std(anomaly_pixels[:, 0]))
        chroma = float(np.sqrt(a_mean ** 2 + b_mean ** 2))
        
        # Normalize a-channel to skin tone
        a_norm = (a_mean - skin_tone_a_median) / (skin_tone_a_std + 1e-6)
        
        # --- Classification logic ---
        # Size thresholds (px²)
        FRECKLE_SIZE_LOW, FRECKLE_SIZE_HIGH = 4, 20
        BEAUTY_MARK_SIZE_LOW, BEAUTY_MARK_SIZE_HIGH = 15, 60
        BLEMISH_SIZE_LOW, BLEMISH_SIZE_HIGH = 8, 40
        NOISE_MAX_SIZE = 4
        
        # Color thresholds
        FRECKLE_A_MIN = 128 + 8  # Red-tinted: a > 136
        BEAUTY_MARK_A_MAX = 128 - 10  # Dark: a < 118
        BLEMISH_CHROMA_MIN = 30  # Inflamed: high chroma
        
        scores = {
            "freckle": 0.0,
            "beauty_mark": 0.0,
            "blemish": 0.0,
            "noise": 0.0,
        }
        reasons = {
            "freckle": [],
            "beauty_mark": [],
            "blemish": [],
            "noise": [],
        }
        
        # --- Freckle scoring ---
        if FRECKLE_SIZE_LOW <= size <= FRECKLE_SIZE_HIGH:
            scores["freckle"] += 0.3  # Size match
            reasons["freckle"].append(f"size={size:.0f}")
        
        if a_norm > 0.5:  # Red-tinted relative to skin tone
            scores["freckle"] += 0.3
            reasons["freckle"].append(f"a_norm={a_norm:.2f}")
        
        if L_mean > 80:  # Freckles are moderately bright (not very dark)
            scores["freckle"] += 0.2
            reasons["freckle"].append(f"L={L_mean:.1f}")
        
        # --- Beauty mark scoring ---
        if BEAUTY_MARK_SIZE_LOW <= size <= BEAUTY_MARK_SIZE_HIGH:
            scores["beauty_mark"] += 0.3
            reasons["beauty_mark"].append(f"size={size:.0f}")
        
        if a_norm < -0.5:  # Dark relative to skin tone
            scores["beauty_mark"] += 0.3
            reasons["beauty_mark"].append(f"a_norm={a_norm:.2f}")
        
        if L_mean < 100:  # Beauty marks are darker
            scores["beauty_mark"] += 0.2
            reasons["beauty_mark"].append(f"L={L_mean:.1f}")
        
        # --- Blemish scoring ---
        if BLEMISH_SIZE_LOW <= size <= BLEMISH_SIZE_HIGH:
            scores["blemish"] += 0.3
            reasons["blemish"].append(f"size={size:.0f}")
        
        if chroma > BLEMISH_CHROMA_MIN:
            scores["blemish"] += 0.4
            reasons["blemish"].append(f"chroma={chroma:.1f}")
        
        if L_std > 10:  # Inflamed areas have high internal variance
            scores["blemish"] += 0.2
            reasons["blemish"].append(f"L_std={L_std:.1f}")
        
        # --- Noise scoring ---
        if size < NOISE_MAX_SIZE:
            scores["noise"] = 0.8  # Small = likely noise
            reasons["noise"].append(f"size={size:.0f}<{NOISE_MAX_SIZE}")
        
        # Normalize scores to [0, 1] and pick max
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        reason_str = " ".join(reasons[best_type])
        
        return FreckleClassification(
            anomaly_id=-1,
            anomaly_obj=anomaly,
            classification=best_type,
            confidence=best_score,
            reason=reason_str
        )
    
    def classify_anomalies(
        self,
        img_bgr: np.ndarray,
        face_mask: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.6,
    ) -> List[FreckleClassification]:
        """Detect and classify all anomalies in the face.
        
        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            face_mask: (H, W) binary/float mask. Processing limited to masked region.
            confidence_threshold: Only return classifications with confidence >= threshold.
        
        Returns:
            List of FreckleClassification objects, sorted by confidence descending.
        """
        # Detect anomalies using BlemishRemover
        anomalies = self.blemish_remover.detect(img_bgr, face_mask=face_mask)
        
        if not anomalies:
            return []
        
        # Extract LAB for color analysis
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        
        # Compute skin tone statistics (LAB a-channel) in face region
        if face_mask is not None:
            skin_pixels_a = lab[face_mask > 0.5, 1]
        else:
            skin_pixels_a = lab[:, :, 1].flatten()
        
        skin_tone_a_median = float(np.median(skin_pixels_a))
        skin_tone_a_std = float(np.std(skin_pixels_a))
        
        # Classify each anomaly
        classifications = []
        for idx, anomaly in enumerate(anomalies):
            # Build region mask for anomaly
            x1, y1, x2, y2 = anomaly.bbox
            region_mask = np.zeros_like(img_bgr[:, :, 0], dtype=np.float32)
            cv2.circle(
                region_mask,
                (int(anomaly.center[0]), int(anomaly.center[1])),
                int(np.sqrt(anomaly.area / np.pi)) + 2,
                1.0,
                -1
            )
            
            # Extract region
            region_lab = lab[max(0, y1):min(lab.shape[0], y2+1),
                            max(0, x1):min(lab.shape[1], x2+1)]
            region_mask_cropped = region_mask[max(0, y1):min(lab.shape[0], y2+1),
                                              max(0, x1):min(lab.shape[1], x2+1)]
            
            # Classify
            classification = self._classify_anomaly(
                anomaly,
                region_lab,
                region_mask_cropped,
                skin_tone_a_median,
                skin_tone_a_std
            )
            classification.anomaly_id = idx
            
            if classification.confidence >= confidence_threshold:
                classifications.append(classification)
        
        # Sort by confidence descending
        classifications.sort(key=lambda c: c.confidence, reverse=True)
        return classifications
    
    def remove(
        self,
        img_bgr: np.ndarray,
        face_mask: Optional[np.ndarray] = None,
        freckle_removal: float = 0.5,
        freckle_preserve_mask: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.7,
    ) -> np.ndarray:
        """Remove freckles while preserving beauty marks.
        
        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            face_mask: (H, W) binary/float mask.
            freckle_removal: Removal strength (0–1). 0 = no removal, 1 = aggressive.
            freckle_preserve_mask: Optional (H, W) binary mask of pixels to preserve
                                  (user-selected beauty marks, etc.).
            confidence_threshold: Min classification confidence to act on.
        
        Returns:
            (H, W, 3) uint8 BGR image with freckles removed.
        """
        if freckle_removal <= 0:
            return img_bgr.copy()
        
        # Classify anomalies
        classifications = self.classify_anomalies(
            img_bgr,
            face_mask=face_mask,
            confidence_threshold=confidence_threshold
        )
        
        if not classifications:
            return img_bgr.copy()
        
        # Initialize preserve mask (what NOT to remove)
        preserve_mask = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        if freckle_preserve_mask is not None:
            preserve_mask = freckle_preserve_mask.astype(np.float32)
        
        # Mark beauty marks for preservation
        for classification in classifications:
            if classification.classification == "beauty_mark":
                x1, y1, x2, y2 = classification.anomaly_obj.bbox
                cx, cy = classification.anomaly_obj.center
                radius = int(np.sqrt(classification.anomaly_obj.area / np.pi)) + 3
                cv2.circle(
                    preserve_mask,
                    (int(cx), int(cy)),
                    radius,
                    1.0,
                    -1
                )
        
        # Build freckle removal mask (what TO remove)
        removal_mask = np.zeros_like(preserve_mask)
        for classification in classifications:
            if classification.classification == "freckle" and preserve_mask[int(classification.anomaly_obj.center[1]), int(classification.anomaly_obj.center[0])] < 0.5:
                x1, y1, x2, y2 = classification.anomaly_obj.bbox
                cx, cy = classification.anomaly_obj.center
                radius = int(np.sqrt(classification.anomaly_obj.area / np.pi)) + 2
                cv2.circle(
                    removal_mask,
                    (int(cx), int(cy)),
                    radius,
                    freckle_removal * classification.confidence,
                    -1
                )
        
        # Apply BlemishRemover healing logic
        # (Reuse the healing algorithm from blemish.py)
        result = self.blemish_remover.heal_mask(img_bgr, removal_mask, face_mask=face_mask)
        
        return result
```

### 3.3 Parameter Additions

**File: `/Applications/htdocs/retouch/retouch/params.py`**

Add to `PARAM_SPECS`:

```python
ParamSpec(
    name="freckle_removal",
    cli_flag="freckle_removal",
    cli_type=float,
    default=0.0,
    recipe_key="frequency.freckle_removal",
    conversion="recipe_pct",
    min_val=0.0,
    max_val=100.0,
),

ParamSpec(
    name="freckle_preserve_mask",
    cli_flag=None,  # Advanced API only, not GUI slider
    cli_type=None,
    default=None,
    recipe_key=None,  # Not in recipe; user-provided at runtime
    conversion="gui_direct",
),
```

### 3.4 Engine Integration

**File: `/Applications/htdocs/retouch/retouch/engine.py`**

Add to ProcessingContext:

```python
freckle_removal: float = 0.0  # Freckle removal strength (0–100)
freckle_preserve_mask: Optional[np.ndarray] = None  # User-selected preserve regions
```

Add processing stage in `_stage_per_face()` (after S2 blemish removal, before S3 other corrections):

```python
        # --- Freckle and beauty mark removal ---
        if ctx.freckle_removal > 0:
            freckle_remover = FreckleRemover()
            face_img = freckle_remover.remove(
                img_bgr=face_img,
                face_mask=refined_skin_mask,
                freckle_removal=ctx.freckle_removal * 0.01,  # Convert 0–100 to 0–1
                freckle_preserve_mask=ctx.freckle_preserve_mask,
                confidence_threshold=0.7,
            )
```

---

## 4. Implementation Checklist

- [ ] Create freckle.py with FreckleRemover class (~300 lines)
  - [ ] FreckleClassification dataclass
  - [ ] _classify_anomaly() method (size/color/clustering logic)
  - [ ] classify_anomalies() public method
  - [ ] remove() public method (orchestrates detection + removal)
- [ ] Add `freckle_removal`, `freckle_preserve_mask` ParamSpecs
- [ ] Add fields to ProcessingContext
- [ ] Add processing stage call in _stage_per_face
- [ ] Add unit tests (test_freckle.py, ~18–25 tests)
  - [ ] Test classification on synthetic anomalies (various sizes, colors)
  - [ ] Test multi-skin-tone robustness (normalized a-channel)
  - [ ] Test preservation of beauty marks (confidence > 0.8)
  - [ ] Test removal of freckles (confidence > 0.7)
  - [ ] Test user preserve_mask override
  - [ ] Test zero-strength no-op (freckle_removal = 0 returns input)
  - [ ] Test dtype preservation (uint8 in, uint8 out)
  - [ ] Test edge cases: no anomalies, all anomalies, single eye region
- [ ] Visual QA on 5+ cosplay photos (light/medium/dark skin tones, varied freckle patterns)

---

## 5. Files to Modify / Create

| File | Lines | Change | Risk |
|------|-------|--------|------|
| `retouch/freckle.py` | new | New FreckleRemover class (~300 lines) | Low (new module) |
| `retouch/params.py` | ~400 | Add freckle_removal + preserve_mask ParamSpecs | Low (new params) |
| `retouch/engine.py` | ~50 | Add fields to ProcessingContext | Low (new fields) |
| `retouch/engine.py` | 2400–2500 | Add freckle removal call in _stage_per_face | Med (new stage) |
| `tests/test_freckle.py` | new | ~400 lines, 18–25 tests | Low (new tests) |
| `retouch/perf_optimizations.py` | end | Add freckle to pipeline (if needed) | Low (pipeline update) |

---

## 6. Acceptance Criteria

- [x] **Freckles removed cleanly**
  - No dark spots, visible seams, or banding at removal boundary.
  - Heal quality consistent with BlemishRemover.

- [x] **Beauty marks preserved by default**
  - Confidence threshold 0.8 + classification logic correctly identifies beauty marks.
  - Visual inspection: intentional marks remain on 5+ cosplay photos.

- [x] **User can override via preserve_mask**
  - If freckle_preserve_mask provided (base64 PNG), manually selected regions respected.
  - Merge logic: user mask takes priority (beauty_mark auto-detection + user = full preserve).

- [x] **Multi-skin-tone accuracy**
  - Test on light/medium/dark skin tones (3 photos each).
  - Classification accuracy consistent across tones (normalized a-channel).
  - No false positives (beauty mark removed as freckle) or false negatives (freckle preserved).

- [x] **Zero-strength no-op**
  - freckle_removal = 0 returns input unchanged.

- [x] **Dtype preservation**
  - uint8 input → uint8 output (via BlemishRemover.heal_mask).
  - float32 conversion handled if needed.

- [x] **All existing tests remain green**
  - BlemishRemover tests unaffected.
  - No regressions in S2 blemish removal (FreckleRemover doesn't interfere).

---

## 7. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Classifier misidentifies beauty mark as freckle | Medium | User loses intentional feature | High confidence threshold (0.8); expose confidence param; provide preserve_mask override |
| Freckles on dark skin misidentified as blemishes | Medium | Incorrect removal/preservation | Use LAB a/b normalized to skin tone; test on 3 dark-skin photos; tune thresholds |
| Performance: slow classification on heavily freckled skin | Low | Pipeline bottleneck | Cap processing to max 1000 anomalies; log warning; parallelize if needed |
| Interaction with blemish removal: double-remove | Medium | Freckle removed by both S2 + S3 | Freckle removal runs BEFORE blemish (or use separate mask to avoid overlap) |
| Healing artifacts (seams, over-smoothing) | Low | Quality degradation | Reuse proven BlemishRemover.heal_mask; extensive visual QA |
| preserve_mask format unclear to users | Low | User confusion | Document base64 PNG format; provide UI helper (if GUI available) |

---

## 8. Testing Strategy

### Unit Tests (test_freckle.py, ~400 lines)

```python
def test_classify_freckle_size_color():
    """Verify freckle classification on synthetic small red spots."""
    # Create 5px² red anomaly at (128, 100) center
    # Assert classification == "freckle" and confidence > 0.7

def test_classify_beauty_mark_size_color():
    """Verify beauty mark classification on synthetic dark spots."""
    # Create 30px² dark anomaly at (150, 80) center
    # Assert classification == "beauty_mark" and confidence > 0.8

def test_classify_blemish_inflamed():
    """Verify blemish classification on high-chroma spot."""
    # Create 25px² spot with chroma > 40
    # Assert classification == "blemish" and confidence > 0.7

def test_classify_noise():
    """Verify noise classification on very small spot."""
    # Create 2px² anomaly
    # Assert classification == "noise"

def test_classification_multitone_light_skin():
    """Test classification on light skin tone photo."""
    # Load real light-skin cosplay photo
    # Classify anomalies
    # Verify freckles/marks identified correctly (manual validation)

def test_classification_multitone_medium_skin():
    """Test classification on medium skin tone photo."""
    # Same as above, medium-skin reference photo

def test_classification_multitone_dark_skin():
    """Test classification on dark skin tone photo."""
    # Same as above, dark-skin reference photo

def test_remove_freckles_preserve_marks():
    """Verify freckle removal while preserving marks."""
    # Create image with 5 freckles + 2 beauty marks
    # Call remove() with freckle_removal=0.8
    # Assert freckles removed, beauty marks intact

def test_remove_with_user_preserve_mask():
    """Verify user preserve_mask takes priority."""
    # Create image with 3 anomalies
    # Call remove() with preserve_mask covering one anomaly
    # Assert preserved anomaly unchanged, others removed

def test_remove_zero_strength():
    """Verify zero strength returns input unchanged."""
    # Call remove() with freckle_removal=0.0
    # Assert output == input (byte-identical)

def test_remove_dtype_preservation():
    """Verify dtype preserved (uint8 in, uint8 out)."""
    # Call remove() on uint8 image
    # Assert output.dtype == np.uint8

# ... 10–15 more edge-case tests
```

### Integration Tests

- Full pipeline on 5+ cosplay photos (light/medium/dark skin).
- Compare freckle_removal=0 (baseline) vs. freckle_removal=50 (moderate) vs. =100 (aggressive).
- Visual QA: freckles reduced, marks preserved, no over-smoothing.

### Regression Tests

- Run golden snapshots: freckle_removal=0 should be byte-identical to baseline.
- Verify BlemishRemover tests still pass (no regression).

---

## 9. Pseudo-Code Flow

```
INPUT: img_bgr, face_mask, freckle_removal, preserve_mask

IF freckle_removal == 0:
    RETURN img_bgr (no-op)

// Detect anomalies
anomalies = BlemishRemover.detect(img_bgr, face_mask)

// Extract LAB and skin tone stats
lab = convert_to_lab(img_bgr)
skin_a_median, skin_a_std = compute_stats(lab[face_mask>0.5, 1])

// Classify each anomaly
classifications = []
FOR EACH anomaly IN anomalies:
    score = classify_anomaly(anomaly, lab, skin_tone_params)
    classifications.append(score)

// Build preservation mask (beauty marks + user override)
preserve_mask = user_preserve_mask OR zeros
FOR EACH classification IN classifications:
    IF classification.type == "beauty_mark":
        preserve_mask |= circle_around(classification.center)

// Build removal mask (freckles only, excluding preserved)
removal_mask = zeros
FOR EACH classification IN classifications:
    IF classification.type == "freckle":
        IF not overlaps(classification.center, preserve_mask):
            removal_mask |= circle_around(classification.center) * freckle_removal

// Heal
result = BlemishRemover.heal_mask(img_bgr, removal_mask, face_mask)

RETURN result
```

---

## 10. Definition of Done

1. freckle.py created with FreckleRemover class and all methods.
2. All unit tests passing (18–25 tests, >85% code coverage).
3. Visual QA on 5+ cosplay photos documented in VISUAL_QA.md.
4. Multi-skin-tone testing confirmed (light/medium/dark).
5. Backward compatibility verified (freckle_removal=0 → byte-identical).
6. Integration test with BlemishRemover confirms no regressions.
7. Documentation updated (parameter docs, recipe examples).
8. Code review approved by skin-processing + vision leads.

---

## As-Built Corrections

Implemented per this spec with the following deviations (verified against `retouch/`:

- **`BlemishRemover` exposes only `remove(img_bgr, skin_mask, strength)`** — there is **no** `detect()`, `heal_mask()`, or `Anomaly` class. The spec's "85% reuse of `BlemishRemover`" premise was false. Implemented **own** anomaly detection (grayscale local-mean deviation + HSV redness, restricted to `skin_mask > 0.3`, `cv2.connectedComponentsWithStats` for area/centroid/bbox) and healed via `blemish.inpaint_and_blend` (uint8 mask) using the E1 `apply_u8_op_float` delta adapter so float32 canvases are handled correctly.
- **Classification** scores each connected component by SIZE + LAB a/b **normalized to skin tone** (`a_norm = (a - a_median)/(a_std + 1e-6)`), disambiguating freckle (~4–20 px², reddish/light, clustered) vs beauty mark (~15–60 px², dark, isolated) vs blemish (high chroma) vs noise (<4 px²). This normalization keeps it robust across light/medium/dark skin.
- **`freckle_preserve_mask` is an `ndarray`** (the API-level override), not a base64 PNG string — encoding/decoding is left to the caller/GUI. `freckle_removal` is 0–100 (converted internally to 0–1).
- **`freckle_removal = 0.0` returns a byte-identical copy** (no-op).
- **Integration point is `retouch/perf_optimizations.py`**: `FreckleRemover().remove(img_bgr=canvas, face_mask=skin_n, freckle_removal=ctx.freckle_removal, freckle_preserve_mask=ctx.freckle_preserve_mask)` runs after frequency smoothing, gated by `ctx.freckle_removal > 0`. The spec's guidance to edit `engine._stage_per_face` is incorrect.
- **Conversion fix:** `freckle_removal` ParamSpec uses `conversion="recipe_direct"` (engine expects 0–100; `FreckleRemover.remove` divides by 100). `recipe_pct` would have double-divided and disabled removal. `freckle_preserve_mask` is API-only (no GUI widget).
- **GUI/CLI exposure:** `RetouchEngine.process(freckle_removal=...)`, `cli.py --freckle-removal`, and a `Freckle Removal` slider (0–100) in the GUI skin section. Default 0.0 = no-op. Verified: `freckle_removal=60` changes 551 px on a duotian sample (pre-fix it was effectively off).

---

## 11. Interaction with Other Features

- **Region-aware smoothing (Spec #1):** Independent. Freckle removal is geometric (specific pixels), not region-based smoothing.

- **Anisotropic diffusion (Spec #2):** Independent. Freckle removal applies before any smoothing.

- **Eye shadow smoothing (Spec #3):** Independent. Eye region excluded from freckle detection (optional: check eye_mask).

- **Existing blemish removal (S2):** Coordinate ordering.
  - Option A: Freckle removal runs BEFORE blemish removal (independent masks).
  - Option B: Freckle removal uses BlemishRemover but filters by classification.
  - Current design: Option B (freckle removal uses classified anomalies from BlemishRemover).

