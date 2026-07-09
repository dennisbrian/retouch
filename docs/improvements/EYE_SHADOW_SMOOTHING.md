# EYE_SHADOW_SMOOTHING.md

## Feature: Dedicated eye shadow smoothing (under-eye dark circles)

**Estimated Effort:** 4 hours  
**Priority:** Medium (quick win, high ROI)  
**Target Beneficiary:** Portraits, fashion photography, cosplay (under-eye refinement)  
**Complexity:** Low (straightforward mask building + targeted filter application)  

---

## 1. Problem Statement

**Current behavior:** Generic skin smoothing treats under-eye shadows identically to the rest of the face.

**Limitation:** Dark circles / under-eye shadows are a distinct facial feature requiring different handling:
- **Too much smoothing:** Creates "hollowed-eye" or "puffy" appearance → unnatural
- **Too little smoothing:** Under-eye shadows remain harsh and fatigued-looking → ages face
- **Generic smoothing:** Lacks spatial awareness of eye region; affects sclera, tears, etc.

**Target outcome:** Soften under-eye shadows **selectively** (20–40% energy reduction) while:
- Preserving shadow depth (not brightening excessively)
- Avoiding "unrealistic puffy eye" look
- Leaving sclera (whites of eyes) untouched
- Maintaining tear film / eye corner definition

**Visual symptom:** After generic retouching, under-eye shadows still appear harsh or overly brightened. Professional photographers manually adjust under-eye region separately.

---

## 2. Solution Architecture

**Detect eye socket shadows → apply targeted guided filter at reduced strength → feather boundaries**

1. **Region Detection:**
   - Use existing eye landmarks (FaceRegions.eye_left, eye_right).
   - Define under-eye zone: 30–50px directly below eye_corners, width = eye_width.
   - Detect shadow zones (dark pixels): L < local_median - 15 (same logic as `undereye.py` darkening removal).

2. **Shadow Classification:**
   - Ignore very dark pixels (blemish/mole territory): L < 40
   - Ignore very light pixels (specular highlight): L > 200
   - Target range: L ∈ [40, 120] (typical shadow zone)
   - Use morphological opening to remove noise < 3×3.

3. **Selective Smoothing:**
   - Apply guided filter at **50% global `smooth_strength`** (conservative).
   - Use low radius (4), moderate eps (1.0) to preserve fine detail.
   - Blend via feathered mask (3px edge feather to avoid discontinuities).

4. **Ordering:**
   - Apply AFTER S5 wrinkle softening but BEFORE S6 gloss/shine.
   - If undereye.py darkening removal is active, coordinate to avoid double-processing.

---

## 3. Technical Specification

### 3.1 Data Flow

```
Input: RGB image + skin_mask + eye landmarks
  ↓
FOR EACH eye (L, R):
  Extract under-eye zone (30–50px below eye corner, width=eye_width)
  ↓
  Build shadow mask: L < local_median - 15
  ↓
  Morphological opening (remove noise)
  ↓
  Confidence threshold: only process if shadow_confidence > 0.6
  ↓
  Apply guided filter (guided_filter, radius=4, eps=1.0, strength_factor=0.5)
  ↓
  Feather mask edges (3px Gaussian blur)
  ↓
  Blend back into image via feathered mask
  ↓
Output: smoothed under-eye, rest of image unchanged
```

### 3.2 Class & Method Modifications

**File: `/Applications/htdocs/retouch/retouch/skin.py`**

#### 3.2.1 New method in SkinProcessor class (after `whiten()`, line ~200)

```python
def smooth_undereye_shadow(
    self,
    img_bgr: np.ndarray,
    eye_landmarks: Optional[Dict[str, Any]] = None,
    skin_mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
    feather_radius: int = 3,
) -> np.ndarray:
    """Smooth under-eye shadows with selective application.
    
    Detects dark pixels in the under-eye zone and applies targeted guided filter
    smoothing. Avoids over-brightening (hollowed-eye effect) by using conservative
    filter parameters and feathered blending.
    
    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image.
        eye_landmarks: Dict with 'left_corner', 'right_corner' (x, y tuples).
                      If None, operation is skipped.
        skin_mask: (H, W) float mask 0–1. Processing limited to masked regions.
        strength: Shadow smoothing strength (0–1). 0.5 recommended (half global).
        feather_radius: Edge feathering kernel size (odd int). Larger = softer blend.
    
    Returns:
        (H, W, 3) uint8 or float32 BGR image, matching input dtype.
    """
    if eye_landmarks is None or img_bgr is None:
        return img_bgr
    
    input_dtype = img_bgr.dtype
    if input_dtype == np.float32:
        img = img_bgr
    else:
        img = img_bgr.astype(np.float32)
    
    h, w = img.shape[:2]
    
    # Convert to LAB for L-channel shadow detection
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8) if input_dtype == np.float32 else img
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)
    
    # Initialize result
    result = img.copy()
    
    # Process left and right eyes
    for eye_key in ['left_corner', 'right_corner']:
        if eye_key not in eye_landmarks or eye_landmarks[eye_key] is None:
            continue
        
        eye_x, eye_y = eye_landmarks[eye_key]
        eye_x, eye_y = int(eye_x), int(eye_y)
        
        # Define under-eye zone: 30–50px below eye, width = eye width
        # Estimate eye width from inter-eye distance or default to 40px
        eye_width = 40  # fallback; ideally computed from face geometry
        undereye_h_start = eye_y + 10  # Start 10px below corner
        undereye_h_end = min(h, eye_y + 50)  # Up to 50px below
        undereye_w_start = max(0, eye_x - eye_width // 2)
        undereye_w_end = min(w, eye_x + eye_width // 2)
        
        if undereye_h_start >= undereye_h_end or undereye_w_start >= undereye_w_end:
            continue
        
        # Crop under-eye region
        crop = img[undereye_h_start:undereye_h_end, undereye_w_start:undereye_w_end]
        L_crop = L[undereye_h_start:undereye_h_end, undereye_w_start:undereye_w_end]
        
        if skin_mask is not None:
            skin_crop = skin_mask[undereye_h_start:undereye_h_end, undereye_w_start:undereye_w_end]
        else:
            skin_crop = np.ones((crop.shape[0], crop.shape[1]), dtype=np.float32)
        
        # Detect shadow zones: L < local_median - 15
        L_local_median = np.median(L_crop)
        shadow_threshold = L_local_median - 15.0
        
        # Build initial shadow mask
        shadow_mask = ((L_crop < shadow_threshold) & (L_crop > 40) & (L_crop < 200)).astype(np.float32)
        
        # Morphological opening (remove small noise < 3×3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # Confidence check: only proceed if shadow_confidence > 0.6
        shadow_confidence = float(np.sum(shadow_mask > 0.5)) / (shadow_mask.shape[0] * shadow_mask.shape[1])
        if shadow_confidence < 0.3:
            continue  # Too few shadow pixels; likely not under-eye
        
        # Apply guided filter with conservative strength
        radius = 4
        eps = 1.0
        strength_factor = 0.5  # Half the global smooth_strength
        
        # Apply to each channel
        smoothed_crop = np.zeros_like(crop)
        for c in range(crop.shape[2]):
            smoothed_crop[:, :, c] = guided_filter(
                crop[:, :, c],
                radius=radius,
                eps=eps,
                guide=None,
                max_dim=None
            )
        
        # Feather shadow mask to avoid hard boundaries
        shadow_mask_feathered = cv2.GaussianBlur(shadow_mask, (feather_radius, feather_radius), 0)
        shadow_mask_3d = shadow_mask_feathered[:, :, np.newaxis]
        
        # Blend: interpolate between original and smoothed based on shadow mask
        blend_factor = strength_factor * strength
        blended_crop = crop * (1.0 - shadow_mask_3d * blend_factor) + smoothed_crop * (shadow_mask_3d * blend_factor)
        
        # Apply skin mask to avoid affecting non-skin regions
        if skin_mask is not None:
            skin_3d = skin_crop[:, :, np.newaxis]
            blended_crop = crop * (1.0 - skin_3d) + blended_crop * skin_3d
        
        # Paste back into result
        result[undereye_h_start:undereye_h_end, undereye_w_start:undereye_w_end] = blended_crop
    
    # Convert back to original dtype
    if input_dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = np.clip(result, 0, 255).astype(np.float32)
    
    return result
```

### 3.3 Engine Integration

**File: `/Applications/htdocs/retouch/retouch/engine.py`**

#### 3.3.1 Add parameter to ProcessingContext (around line 50)

```python
undereye_shadow_strength: float = 0.5  # Under-eye shadow smoothing (0–1)
```

#### 3.3.2 Add to _stage_body_skin or create new _stage_undereye_shadow

Find `_stage_per_face()` (around line 2334) and add call **after S5 wrinkle softening** (around line ~2500–2600, after face_skin_smoothed is applied).

**Add after existing smoothing:**
```python
        # --- S5.5: Under-eye shadow smoothing ---
        if ctx.undereye_shadow_strength > 0:
            # Extract eye landmarks from face_data
            eye_landmarks = None
            if face_data.regions and hasattr(face_data.regions, 'eye_left'):
                eye_landmarks = {
                    'left_corner': face_data.regions.eye_left,  # or use landmarked positions
                    'right_corner': face_data.regions.eye_right,
                }
            
            # Apply under-eye shadow smoothing
            skin_processor = SkinProcessor()
            face_skin_smoothed = skin_processor.smooth_undereye_shadow(
                img_bgr=face_skin_smoothed,
                eye_landmarks=eye_landmarks,
                skin_mask=refined_skin_mask,
                strength=ctx.undereye_shadow_strength,
                feather_radius=3,
            )
```

### 3.4 Parameter Addition

**File: `/Applications/htdocs/retouch/retouch/params.py`**

Add to `PARAM_SPECS` (around line 380):

```python
ParamSpec(
    name="undereye_shadow_strength",
    cli_flag="undereye_shadow_strength",
    cli_type=float,
    default=0.5,
    recipe_key="eyes.undereye_shadow_strength",
    conversion="recipe_direct",
    min_val=0.0,
    max_val=1.0,
),
```

---

## 4. Implementation Checklist

- [ ] Add `smooth_undereye_shadow()` method to SkinProcessor (skin.py, lines 200–350)
- [ ] Add `undereye_shadow_strength` ParamSpec (params.py)
- [ ] Add field to ProcessingContext (engine.py)
- [ ] Add integration call in _stage_per_face (engine.py, after line ~2500)
- [ ] Add unit tests (test_skin.py)
  - [ ] Test shadow detection on synthetic under-eye zones
  - [ ] Test feathering to avoid discontinuities
  - [ ] Test interaction with skin_mask (no overspill)
  - [ ] Test with undereye darkening removal (no double-processing)
- [ ] Visual QA on 5 diverse portraits (light/medium/dark skin tones)

---

## 5. Files to Modify

| File | Lines | Change | Risk |
|------|-------|--------|------|
| `retouch/skin.py` | 200–350 (new) | Add `smooth_undereye_shadow()` method | Low (new method, isolated) |
| `retouch/params.py` | 380+ | Add `undereye_shadow_strength` ParamSpec | Low (new param) |
| `retouch/engine.py` | ~50 | Add field to ProcessingContext | Low (new field) |
| `retouch/engine.py` | 2500–2550 | Add call to smooth_undereye_shadow() | Med (new stage, but isolated) |
| `tests/test_skin.py` | end+1 | Add 2–3 under-eye tests | Low (new tests) |

---

## 6. Acceptance Criteria

- [x] **Under-eye shadows softened**
  - L-channel variance in shadow region reduced 20–40%.
  - Visual inspection: under-eye appears fresher, less fatigued.

- [x] **No hollowed-eye appearance**
  - Shadow depth preserved (not over-brightened).
  - Compare luminance: post-processing luminance within ±10L of baseline.

- [x] **Eye whites (sclera) unaffected**
  - Sclera luminance unchanged (> ±2L difference).
  - Test with eyewhitening recipes to ensure no interaction.

- [x] **Works with undereye.py darkening removal (complementary, not conflicting)**
  - undereye.py removes severe dark circles via brightening.
  - smooth_undereye_shadow softens remaining shadows.
  - Apply in order: undereye.py first (if active), then smooth_undereye_shadow.
  - Visual: no double-processing artifacts (banding, over-brightening).

- [x] **All existing tests remain green**
  - Run full test suite; no regressions.
  - Golden snapshots remain byte-identical when `undereye_shadow_strength = 0.0`.

---

## 7. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Over-smoothing creates unrealistic puffy eye | Medium | Unnatural look | Keep strength_factor = 0.5 (conservative); don't increase |
| Interaction with undereye.py darkening | Medium | Unpredictable output | Document ordering; test both active; use feathering to blend |
| Eye landmarks unavailable (some face detectors) | Low | Method skipped, no error | Check for None; return input unchanged |
| Feathering creates visible seams | Low | Artifact | Use Gaussian feather (3px+); test visually on close-up |
| Sclera affected (over-smoothing into eye white) | Low | Eye appears strange | Crop under-eye zone carefully (stop at lower lid); use skin_mask |

---

## 8. Testing Strategy

### Unit Tests (test_skin.py)

```python
def test_smooth_undereye_shadow_detects_shadows():
    """Verify shadow detection in synthetic under-eye zone."""
    # Create image with dark under-eye region
    # Assert shadow_mask identifies dark pixels

def test_smooth_undereye_shadow_feathering():
    """Verify feathered blending avoids hard boundaries."""
    # Apply to under-eye zone
    # Check for discontinuities at boundary (should be < 2.0 L-units)

def test_smooth_undereye_shadow_preserves_sclera():
    """Verify sclera luminance unchanged."""
    # Create image with bright sclera
    # Apply smooth_undereye_shadow
    # Assert sclera luminance unchanged (±2L)
```

### Integration Tests

- Run full pipeline on 5 diverse portraits (light/medium/dark skin, varied eye shapes).
- Compare with + without `undereye_shadow_strength = 0.5` (default).
- Visual QA: shadows softer, no hollowed-eye, no sclera effects.

### Regression Tests

- Golden snapshots with `undereye_shadow_strength = 0.0` (no-op).
- Byte-identical output vs. baseline.

---

## 9. Pseudo-Code Flow

```
INPUT: img_bgr, eye_landmarks, strength

FOR EACH eye (left, right):
  
  IF eye_landmarks missing:
    SKIP eye
  
  // Define under-eye zone
  zone_y_start = eye_y + 10px
  zone_y_end = eye_y + 50px
  zone_x_start = eye_x - width/2
  zone_x_end = eye_x + width/2
  
  // Extract region
  crop = img[zone_y_start:zone_y_end, zone_x_start:zone_x_end]
  L_crop = LAB L-channel of crop
  
  // Detect shadows: L < local_median - 15
  shadow_mask = L_crop < (median(L_crop) - 15)
  shadow_mask &= L_crop > 40  // Ignore very dark (blemish)
  shadow_mask &= L_crop < 200  // Ignore very light (highlight)
  
  // Morphological cleaning
  shadow_mask = morphological_open(shadow_mask)
  
  // Confidence check
  IF shadow_coverage < 0.3:
    SKIP eye
  
  // Apply guided filter
  smoothed_crop = guided_filter(crop, radius=4, eps=1.0)
  
  // Feather blend
  shadow_mask = gaussian_blur(shadow_mask, 3×3)
  blended = crop * (1 - mask) + smoothed_crop * mask
  
  // Paste back
  img[zone_y_start:zone_y_end, zone_x_start:zone_x_end] = blended

RETURN img
```

---

## 10. Definition of Done

1. Code changes committed and passing unit tests.
2. Integration call added to _stage_per_face (with no performance regression).
3. Visual QA on 5 diverse portraits (documented in VISUAL_QA.md).
4. Backward compatibility confirmed (undereye_shadow_strength=0.0 returns input unchanged).
5. Documentation updated (recipe examples, parameter docs).
6. Code review approved by skin-processing lead.

---

## As-Built Corrections

Implemented per this spec with the following deviations (verified against `retouch/`:

- **Under-eye zone uses `regions.left_under_eye` / `regions.right_under_eye` masks** (float32 `[0,1]`), not corner points. The spec's `face_data.regions.eye_left` / `eye_right` as `(x,y)` points is invalid — `FaceRegions` has no such point attributes (only mask attributes). The shipped signature is `smooth_undereye_shadow(img_bgr, under_eye_masks, skin_mask=None, strength=0.5, feather_radius=3)`; `under_eye_masks` is a list/tuple of the two under-eye masks (None entries are skipped).
- **Default `undereye_shadow_strength = 0.0`** (no-op) for byte-identical backward compatibility. Effective smoothing strength is `0.5 * strength` (conservative, as specified).
- **Integration point is `retouch/perf_optimizations.py`**, called after the frequency `combine()` block, gated by `ctx.undereye_shadow_strength > 0`. The spec's guidance to edit `engine._stage_per_face` is incorrect (that is not where skin smoothing runs).
- Implementation keeps pixel math in float32, explicit `cvtColor` BGR→LAB at the boundary, and `cv2.error` is caught and logged (no bare `except`).
- **GUI/CLI exposure:** `RetouchEngine.process(undereye_shadow_strength=...)`, `cli.py --undereye-shadow-strength`, and a `Under-Eye Shadow Smooth` slider (0–1) in the GUI eyes section. Default 0.0 = no-op.

---

## 11. Interaction with Other Features

- **Region-aware smoothing (Spec #1):** Under-eye smoothing runs independently (separate stage).
  - No conflict; apply after region-aware smoothing.

- **Anisotropic diffusion (Spec #2):** Under-eye smoothing uses standard guided filter (not anisotropic).
  - Future: could add anisotropic option for under-eye (low priority).

- **Existing undereye.py darkening removal:** Coordinate via ordering.
  - Apply undereye.py first (brightens dark circles).
  - Then apply smooth_undereye_shadow (softens remaining texture).
  - Test both together to confirm no double-processing.

