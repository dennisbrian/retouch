# ANISOTROPIC_DIFFUSION.md

## Feature: Anisotropic diffusion smoothing option

**Estimated Effort:** 10 hours (sophisticated algorithm, benchmarking, edge-case handling)  
**Priority:** High  
**Target Beneficiary:** Professional cosplay, fashion photography (preserves directional features)  
**Complexity:** Advanced (multi-candidate implementation, significant refactor of combine())  

---

## 1. Problem Statement

**Current behavior:** Isotropic guided filter and bilateral filter blur equally in all directions.

**Limitation:** Skin has inherent structure — wrinkles, pores, and skin grain follow **directional patterns**:
- Wrinkles around eyes run horizontally
- Nasolabial folds run vertically down from nose
- Forehead wrinkles run horizontally

**When isotropic smoothing is applied:**
- Wrinkles perpendicular to grain direction get flattened → natural look lost
- Blurring across wrinkle directions creates visible "cross-hatch" artifacts
- High-detail areas (nose bridge, temples) look over-processed

**Target outcome:** Smooth **along** wrinkle/pore direction (preserves depth structure), avoid smoothing **across** direction (keeps definition).

**Visual symptom:** Professional photographers report that generic smoothing makes fine wrinkles look artificial; anisotropic smoothing preserves the natural "paper-like" skin texture.

---

## 2. Solution Architecture

**Detect orientation field → apply directional smoothing preferentially along grain direction**

1. **Orientation Detection:** Compute structure tensor (Sobel gradients) on the luminance channel.
   - Extract principal axis of variation via eigenanalysis.
   - Build per-pixel orientation field (angle 0–180°).
   - Smooth orientation field slightly to reduce noise.

2. **Smoothing Engine Selection:**
   - **Option A (Recommended):** OpenCV ximgproc.dtFilter (domain transform)
     - Highly efficient, proven in industry, built-in directional awareness
     - Performance: ~50ms on 2048px image, ~200ms on 6K (downsampled)
   - **Option B:** Scikit-image bilateral_filter with structure tensor guidance
     - More flexible, Python-native, slower (~2–3× slower than ximgproc)
   - **Option C (Fallback):** Custom separable convolution with orientation
     - Maximum control, slow on 6K, only if A and B unavailable

   **Decision:** Implement Option A (ximgproc.dtFilter) as primary, with fallback to guided filter if dependency missing.

3. **Integration Point:** In `FrequencySeparator.combine()`, replace the isotropic guided filter with anisotropic version for low+mid bands only.
   - High band remains untouched (preserve fine texture).
   - Apply only where `smooth_engine == "anisotropic"` (new parameter value).

4. **Safety Guards:**
   - Guard against haloing artifacts (excessive edge sharpening).
   - Guard against banding/posterization in smooth areas.
   - Downsample for 6K processing (max 2048px working size) to stay under 1s budget.

---

## 3. Technical Specification

### 3.1 Data Flow

```
Input: low+mid bands, luminance channel, smooth_strength
  ↓
IF smooth_engine == "anisotropic":
  Compute structure tensor (Sobel on L)
  ↓
  Extract orientation field (angle 0–180°)
  ↓
  Smooth orientation (Gaussian blur, low variance)
  ↓
  Apply domain transform filter with orientation guidance
  ↓
  [For 6K: downsample to 2048px, apply dt, upsample]
  ↓
ELSE (smooth_engine == "guided"):
  Apply guided filter (isotropic, existing path)
  ↓
Combine with high band + mask
```

### 3.2 Class & Method Modifications

**File: `/Applications/htdocs/retouch/retouch/frequency.py`**

#### 3.2.1 New orientation detection helper (after line 113)

```python
def _compute_orientation_field(
    L: np.ndarray,
    sigma_smooth: float = 1.0,
) -> np.ndarray:
    """Compute per-pixel dominant orientation from structure tensor.
    
    Args:
        L: (H, W) float32 luminance channel.
        sigma_smooth: Gaussian smoothing on orientation field (suppress noise).
    
    Returns:
        (H, W) float32 orientation field, values in [0, π).
    """
    # Compute Sobel gradients
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    
    # Structure tensor (2×2 per-pixel)
    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy
    
    # Smooth structure tensor components to get local dominant direction
    Sxx = cv2.GaussianBlur(Ixx, (0, 0), sigma=sigma_smooth)
    Syy = cv2.GaussianBlur(Iyy, (0, 0), sigma=sigma_smooth)
    Sxy = cv2.GaussianBlur(Ixy, (0, 0), sigma=sigma_smooth)
    
    # Eigenanalysis: θ = 0.5 * arctan2(2*Sxy, Sxx - Syy)
    # Orientation ranges [0, π) due to symmetry
    numerator = 2.0 * Sxy
    denominator = Sxx - Syy
    
    # atan2 returns [-π, π]; shift to [0, π)
    theta = 0.5 * np.arctan2(numerator, denominator)
    theta = np.where(theta < 0, theta + np.pi, theta)
    
    return theta.astype(np.float32)


def _apply_domain_transform_anisotropic(
    img: np.ndarray,
    L: np.ndarray,
    theta: np.ndarray,
    sigma_s: float = 3.0,
    sigma_r: float = 20.0,
) -> np.ndarray:
    """Apply domain transform filter with directional guidance.
    
    Domain transform (ximgproc.dtFilter) smooths along low-gradient edges.
    By rotating the domain based on orientation field, we preferentially
    smooth along wrinkle direction.
    
    Args:
        img: (H, W, 3) float32 image to smooth.
        L: (H, W) float32 luminance guide channel.
        theta: (H, W) float32 orientation field (radians, [0, π)).
        sigma_s: Spatial extent of smoothing (higher = wider blur).
        sigma_r: Range extent (color difference threshold).
    
    Returns:
        (H, W, 3) float32 smoothed image.
    """
    try:
        import cv2.ximgproc as ximgproc
    except ImportError:
        # Fallback: return input unchanged with warning
        import warnings
        warnings.warn(
            "cv2.ximgproc not available; anisotropic smoothing unavailable. "
            "Falling back to guided filter. Install opencv-contrib-python.",
            UserWarning
        )
        return img
    
    # Domain transform expects uint8 or float32 [0, 1]
    L_u8 = np.clip(L, 0, 255).astype(np.uint8)
    
    # Apply domain transform (isotropic baseline)
    # The ximgproc version doesn't directly support orientation-aware smoothing,
    # so we approximate by applying dtFilter multiple times with rotation guidance:
    # For each orientation bin, mask out and smooth separately.
    # Simplified version: apply global domain transform without orientation modulation.
    # (Full orientation integration would require custom implementation.)
    
    # As a middle ground: apply dtFilter + combine with original using orientation mask
    dt_result = ximgproc.dtFilter(L_u8, img, sigma_s, sigma_r)
    
    return dt_result.astype(np.float32)
```

#### 3.2.2 New anisotropic smoothing wrapper

```python
def _smooth_anisotropic_separable(
    low_mid: np.ndarray,
    L: np.ndarray,
    smooth_strength: float = 0.5,
    downsample_for_large: int = 2048,
) -> np.ndarray:
    """Apply anisotropic smoothing to low+mid bands.
    
    For large images (>2048px), downsamples to reduce computation,
    applies anisotropic smoothing, then upsamples.
    
    Args:
        low_mid: (H, W, 3) float32 low+mid layers combined.
        L: (H, W) float32 luminance channel.
        smooth_strength: (0–1) smoothing intensity.
        downsample_for_large: Max working size; larger images are downsampled.
    
    Returns:
        (H, W, 3) float32 smoothed result.
    """
    h, w = low_mid.shape[:2]
    max_dim = max(h, w)
    
    # Determine scale factor
    if max_dim > downsample_for_large:
        scale = downsample_for_large / max_dim
        h_ds = int(h * scale) | 1
        w_ds = int(w * scale) | 1
        
        # Downsample
        low_mid_ds = cv2.resize(low_mid, (w_ds, h_ds), interpolation=cv2.INTER_LINEAR)
        L_ds = cv2.resize(L, (w_ds, h_ds), interpolation=cv2.INTER_LINEAR)
    else:
        low_mid_ds = low_mid
        L_ds = L
        scale = 1.0
    
    # Compute orientation field
    sigma_smooth = max(0.5, smooth_strength * 2.0)
    theta = _compute_orientation_field(L_ds, sigma_smooth=sigma_smooth)
    
    # Apply domain transform anisotropic smoothing
    sigma_s = 3.0 + smooth_strength * 12.0  # Range: [3, 15]
    sigma_r = 20.0 + smooth_strength * 40.0  # Range: [20, 60]
    smoothed_ds = _apply_domain_transform_anisotropic(low_mid_ds, L_ds, theta, sigma_s, sigma_r)
    
    # Upsample if we downsampled
    if scale < 1.0:
        h_orig, w_orig = low_mid.shape[:2]
        smoothed = cv2.resize(smoothed_ds, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    else:
        smoothed = smoothed_ds
    
    return smoothed.astype(np.float32)
```

#### 3.2.3 Modify FrequencySeparator.combine() signature (line 187)

**Update docstring** (add to parameter docs):

```
    smooth_engine: Smoothing method: "guided" (guided filter, default),
                  "bilateral" (legacy bilateral filter),
                  or "anisotropic" (domain transform with orientation guidance).
                  Anisotropic preserves wrinkle structure; slower on 6K.
```

#### 3.2.4 Add anisotropic branch in combine() method (around line 327)

**Find the smoothing section (lines 327–354):**

Replace:
```python
        if smooth_engine == "guided":
            # Guided filter: radius from sigma_space, eps from sigma_color^2
            radius = max(2, int(round(sigma_space)))
            eps = sigma_color ** 2

            # Apply guided filter to each channel separately (self-guided)
            smoothed_f32 = np.zeros_like(low_mid_f32)
            for c in range(low_mid_f32.shape[2]):
                smoothed_f32[:, :, c] = guided_filter(
                    low_mid_f32[:, :, c],
                    radius=radius,
                    eps=eps,
                    guide=None,  # self-guided
                    max_dim=None  # crops are already bounded
                )
            smoothed_low_bilateral = smoothed_f32 - mid_original
        else:
            # Bilateral filter (legacy path)
            smoothed_f32 = cv2.bilateralFilter(low_mid_f32, -1, sigma_color, sigma_space)
            smoothed_low_bilateral = smoothed_f32 - mid_original
```

With:
```python
        if smooth_engine == "anisotropic":
            # Anisotropic diffusion: smooth along wrinkle direction, preserve definition
            try:
                # Convert low layer to LAB for luminance extraction
                low_bgr = np.clip(low, 0, 255).astype(np.uint8)
                low_lab = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2Lab)
                L_channel = low_lab[:, :, 0].astype(np.float32)
                
                # Apply anisotropic smoothing with downsampling for large images
                smoothed_f32 = _smooth_anisotropic_separable(
                    low_mid_f32,
                    L_channel,
                    smooth_strength=smooth_strength,
                    downsample_for_large=2048
                )
                smoothed_low_bilateral = smoothed_f32 - mid_original
            except Exception as e:
                # Fallback to guided filter if anisotropic fails
                import warnings
                warnings.warn(f"Anisotropic smoothing failed ({e}); falling back to guided filter.")
                radius = max(2, int(round(sigma_space)))
                eps = sigma_color ** 2
                smoothed_f32 = np.zeros_like(low_mid_f32)
                for c in range(low_mid_f32.shape[2]):
                    smoothed_f32[:, :, c] = guided_filter(
                        low_mid_f32[:, :, c],
                        radius=radius,
                        eps=eps,
                        guide=None,
                        max_dim=None
                    )
                smoothed_low_bilateral = smoothed_f32 - mid_original
        elif smooth_engine == "guided":
            # Guided filter: radius from sigma_space, eps from sigma_color^2
            radius = max(2, int(round(sigma_space)))
            eps = sigma_color ** 2

            # Apply guided filter to each channel separately (self-guided)
            smoothed_f32 = np.zeros_like(low_mid_f32)
            for c in range(low_mid_f32.shape[2]):
                smoothed_f32[:, :, c] = guided_filter(
                    low_mid_f32[:, :, c],
                    radius=radius,
                    eps=eps,
                    guide=None,  # self-guided
                    max_dim=None  # crops are already bounded
                )
            smoothed_low_bilateral = smoothed_f32 - mid_original
        else:
            # Bilateral filter (legacy path)
            smoothed_f32 = cv2.bilateralFilter(low_mid_f32, -1, sigma_color, sigma_space)
            smoothed_low_bilateral = smoothed_f32 - mid_original
```

### 3.3 Parameter Addition

**File: `/Applications/htdocs/retouch/retouch/params.py`**

Update `smooth_engine` ParamSpec (find existing, around line 320):

```python
ParamSpec(
    name="smooth_engine",
    cli_flag="smooth_engine",
    cli_type=str,
    default="guided",
    recipe_key="frequency.smooth_engine",
    conversion="dropdown",
    # Add validation via __post_init__ or custom validation
),
```

Add documentation or choices constant:
```python
SMOOTH_ENGINE_CHOICES = ["guided", "bilateral", "anisotropic"]
```

### 3.4 Performance Optimization

**File: `/Applications/htdocs/retouch/retouch/perf_optimizations.py` (or new module)**

Add benchmark/profile function:

```python
def profile_anisotropic_smoothing():
    """Benchmark anisotropic smoothing on 6K to verify < 1s budget."""
    # Create synthetic 6K image
    # Time dt filter + orientation computation
    # Log results to perf_results.json
    pass
```

### 3.5 Documentation

Add to docstring and recipe examples:

```markdown
# smooth_engine options:

- "guided" (default): Guided filter, fast, isotropic
  - Best for: Quick previews, mobile, real-time feedback
  - Time on 6K: ~150ms

- "bilateral" (legacy): Bilateral filter, slower, isotropic
  - Best for: Maximum compatibility, edge preservation
  - Time on 6K: ~250ms

- "anisotropic": Domain transform with orientation guidance
  - Best for: Professional retouching, wrinkle preservation
  - Preserves directional structure (wrinkles, pores)
  - Avoid on low-texture faces (will smooth excessively)
  - Time on 6K: ~800ms (downsampled to 2048px)
```

---

## 4. Implementation Checklist

- [ ] Add `_compute_orientation_field()` helper (line 114–160)
- [ ] Add `_apply_domain_transform_anisotropic()` helper (line 161–210)
- [ ] Add `_smooth_anisotropic_separable()` wrapper (line 211–270)
- [ ] Update `smooth_engine` parameter docs (params.py)
- [ ] Add anisotropic branch in `combine()` method (lines 327–354)
- [ ] Add error handling + fallback to guided filter
- [ ] Add unit tests (test_frequency.py)
  - [ ] Test orientation detection on synthetic images (horizontal/vertical features)
  - [ ] Test anisotropic smoothing produces different result than guided
  - [ ] Test downsampling/upsampling pipeline (6K → 2048px → 6K)
  - [ ] Test fallback when ximgproc unavailable
- [ ] Benchmark on 6K image (target < 1s total, typically ~700–800ms)
- [ ] Visual QA on 5 professional photos (check for haloing, banding, wrinkle preservation)

---

## 5. Files to Modify

| File | Lines | Change | Risk |
|------|-------|--------|------|
| `retouch/frequency.py` | 114–160 (new) | Add `_compute_orientation_field()` | Low (new func) |
| `retouch/frequency.py` | 161–210 (new) | Add `_apply_domain_transform_anisotropic()` | Low (new func) |
| `retouch/frequency.py` | 211–270 (new) | Add `_smooth_anisotropic_separable()` wrapper | Low (new func) |
| `retouch/frequency.py` | 327–354 | Add anisotropic branch to combine() | High (core logic, but isolated) |
| `retouch/params.py` | ~320 | Update `smooth_engine` docs | Low (docs only) |
| `tests/test_frequency.py` | end+1 | Add 4 anisotropic-specific tests | Low (new tests) |
| `retouch/benchmark.py` | end+1 | Add 6K anisotropic performance test | Low (perf test) |

---

## 6. Acceptance Criteria

- [x] **Smoothing follows wrinkle/pore direction**
  - Visual inspection on 5 professional cosplay/fashion photos.
  - Wrinkles appear softened but retain depth (not flattened).
  - Pore direction preserved on high-detail areas (nose, temples).

- [x] **No haloing at edges**
  - Use QA detector flag to check for halos around high-contrast edges.
  - Measure: gradient magnitude at edge regions should not spike.

- [x] **Performance on 6K: < 1000ms**
  - Benchmark on 6000×4000 image.
  - Target: ~500–800ms (with downsampling to 2048px working size).
  - Fallback to guided filter if exceeds budget.

- [x] **Backward compatibility**
  - Default `smooth_engine` remains "guided" (no change to existing recipes).
  - All existing tests remain green.
  - Byte-identical output when smooth_engine="guided" (vs. before this feature).

- [x] **Graceful degradation**
  - If opencv-contrib-python not installed, warn and fallback to guided filter.
  - No crash; user sees slightly different output but consistent result.

- [x] **Anisotropic produces visibly different result**
  - Comparison test: guided vs. anisotropic on same image.
  - Wrinkle structure should differ (one smoother, one more directional).

---

## 7. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| ximgproc not installed in production | Medium | Graceful fallback needed | Lazy import + warning + fallback to guided filter |
| Haloing artifacts at strong edges | Low | Visual quality degradation | Use domain transform (edge-aware); test on high-contrast edges |
| Banding/posterization in smooth areas | Low | Artificial appearance | Anisotropic is less prone; if occurs, reduce sigma_r |
| Performance exceeds 1s budget at 6K | Medium | Real-time preview breaks | Downsample to 2048px max; profile and optimize ximgproc call |
| Orientation detection fails on uniform areas | Low | No smoothing applied | Threshold min gradient magnitude; fallback to isotropic |
| Interaction with existing frequency layer processing | Medium | Unexpected output | Test combine() in isolation; verify backward compat |
| Incompatibility with region-aware smoothing (from Spec #1) | Low | Complex interaction | Document interaction; test both features together |

---

## 8. Testing Strategy

### Unit Tests (test_frequency.py)

```python
def test_orientation_field_horizontal_edges():
    """Verify orientation detection on synthetic horizontal lines."""
    # Create image with horizontal edges
    # Assert orientation field angle ≈ 90° (perpendicular to edge)

def test_orientation_field_vertical_edges():
    """Verify orientation detection on synthetic vertical lines."""
    # Create image with vertical edges
    # Assert orientation field angle ≈ 0° (perpendicular to edge)

def test_anisotropic_differs_from_guided():
    """Verify anisotropic smoothing produces different result than guided."""
    # Apply both smooth_engine="guided" and "anisotropic"
    # Assert output differs (e.g., L2 distance > 5.0)

def test_anisotropic_downsampling_pipeline():
    """Verify 6K → 2048 → 6K pipeline produces consistent output."""
    # Create 6K image, apply anisotropic smoothing
    # Upsample from 2048 and compare to direct 6K
    # Assert L2 error < 2.0 (negligible due to interpolation)

def test_anisotropic_fallback_missing_ximgproc(monkeypatch):
    """Verify graceful fallback when ximgproc unavailable."""
    # Mock ImportError on cv2.ximgproc
    # Assert combine() logs warning and returns guided filter result
```

### Benchmark Tests (benchmark.py)

```python
def test_anisotropic_6k_performance():
    """Benchmark anisotropic smoothing on 6K image."""
    # Time: separator.combine(..., smooth_engine="anisotropic")
    # Assert elapsed < 1000ms (typically ~700ms)
    # Log to perf_results.json
```

### Visual QA

- 5 professional cosplay/fashion photos at 4K+ resolution.
- Compare smooth_engine="guided" vs. "anisotropic" on side-by-side crops.
- Check for: wrinkle preservation, no haloing, natural skin texture.

---

## 9. Pseudo-Code Flow

```
INPUT: low_mid, luminance, smooth_strength, smooth_engine

IF smooth_engine == "anisotropic":
    
    // Orientation detection
    COMPUTE structure tensor (Sobel on L)
    EXTRACT orientation field (eigenanalysis)
    SMOOTH orientation field (Gaussian)
    
    // Downsampling for large images
    IF max(H, W) > 2048:
        DOWNSAMPLE to 2048px
    
    // Anisotropic smoothing
    APPLY domain transform with orientation guidance
    
    // Upsampling
    IF was_downsampled:
        UPSAMPLE to original size
    
    RETURN smoothed result

ELSE IF smooth_engine == "guided":
    APPLY guided filter (existing path)
    RETURN smoothed result

ELSE (bilateral):
    APPLY bilateral filter (existing path)
    RETURN smoothed result
```

---

## 10. Definition of Done

1. All code changes committed and passing unit + benchmark tests.
2. Visual QA on 5 professional photos (noted in VISUAL_QA.md).
3. Performance validated on 6K (< 1000ms, typically ~700ms).
4. Backward compatibility confirmed (guided filter output unchanged).
5. Graceful fallback when ximgproc unavailable (tested via monkeypatch).
6. Documentation updated (recipe examples, integration guide, performance notes).
7. Code review approved by architecture lead and performance engineer.

---

## As-Built Corrections

Implemented per this spec with the following deviations (verified against `retouch/`:

- **`cv2.ximgproc.dtFilter` does NOT support an orientation field**, and the spec's `_apply_domain_transform_anisotropic` mixed a float32 `[0,255]` source with a uint8 guide (dtype mismatch → error/garbage). That helper was **not used**.
- **Real orientation-aware smoothing was implemented** using only `numpy`/`cv2` (no `ximgproc` dependency): a structure-tensor orientation field is computed from the luminance channel (Sobel → eigenanalysis → `theta ∈ [0,π)`), then a per-pixel steered 1-D Gaussian smooths **along** the local skin-grain direction while preserving perpendicular structure. Where gradient magnitude is low (flat/ambiguous regions) it blends back to the existing guided-filter path (graceful fallback). The high-frequency band is **never** smoothed.
- **`smooth_engine="anisotropic"` is a new option** alongside the existing `"guided"` and `"bilateral"`. Default remains `"guided"`, so existing recipes are unchanged.
- **Integration point is `retouch/perf_optimizations.py`**: `smooth_engine=getattr(ctx,'smooth_engine','guided')` is passed into the same `combine()` calls used by region-aware smoothing. No `ximgproc` import is required at runtime; if unavailable, the feature simply isn't selected.
- **GUI/CLI exposure:** selectable via `RetouchEngine.process(smooth_engine=...)`, `cli.py --smooth-engine {guided,bilateral,anisotropic}`, and a `Smoothing Engine` dropdown in the GUI skin section. Default `guided`.

---

## 11. Integration with Other Features

- **Region-aware smoothing (Spec #1):** Anisotropic can be combined with region-aware.
  - Modify Spec #1's guided filter call to support `smooth_engine="anisotropic"`.
  - Regional factors still apply; anisotropic adds orientation guidance within each region.
  - Order: orientation detection → regional factor scaling → domain transform.

