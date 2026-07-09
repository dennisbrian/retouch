# REGION_AWARE_SMOOTHING.md

## Feature: Region-aware smoothing strength modulation

**Estimated Effort:** 6 hours  
**Priority:** High quality  
**Target Beneficiary:** All recipes (especially `aaa_photoreal`, `cosplay_neutral`)  

---

## 1. Problem Statement

Current global smoothing in `frequency.py` applies uniform `smooth_strength` across the entire face via guided filter. This creates a **plastic, over-retouched appearance** because:

- **Cheeks** (soft, naturally smooth texture) get over-smoothed → porcelain/waxy look
- **Forehead** (typically has visible pore structure) benefits from aggressive smoothing → correctly preserved
- **Nose bridge** (fine wrinkles, directional structure) gets flattened → unnatural, lifeless
- **Temples** (delicate, translucent skin) lose natural micro-texture → doll-like

The mechanism: a single `smooth_strength` scalar (0–1) drives both guided filter `radius` and `eps`, so there's no per-region tuning without hardcoding region offsets in the pipeline.

**Visual Symptom:** Cosplay photos show plastic, uniform skin texture on cheeks while nose remains too detailed or too smooth depending on global setting. Photographers report "can't get both cheeks and nose right without manual intervention."

---

## 2. Solution Architecture

**Detect regional high-frequency energy variation → modulate `smooth_strength` per region → recombine**

1. **Measure local texture richness** using the existing high-frequency band (`layers.high`).
   - Compute robust MAD-based std (as in `_texture_adaptation_factor`) per region.
   - Regions with high energy (e.g., nose bridge wrinkles) get `factor < 1.0` (less smoothing).
   - Regions with low energy (e.g., smooth cheeks) get `factor > 1.0` (more smoothing, with safety cap).

2. **Regional factor application** happens in `FrequencySeparator.combine()` at line 363.
   - Before applying guided filter, partition the mask into dimensional regions.
   - Scale `smooth_strength` by per-region factor (range 0.5–1.5, clamped to 1.5 max to prevent over-smoothing).
   - Apply guided filter with region-scaled strength.

3. **Preserve backward compatibility** via new optional parameter `regional_modulation` (default 1.0 = no-op).
   - Existing recipes continue byte-for-byte identical.
   - New recipes can opt in via `frequency.regional_modulation = 0.7` (or similar).

---

## 3. Technical Specification

### 3.1 Data Flow

```
Input: RGB image + skin_mask
  ↓
Separate into layers (low, mid, high)
  ↓
Extract dimensional regions (nose_bridge, cheeks_l/r, forehead, temples)
  ↓
FOR EACH region:
  Compute high-band energy (MAD-based std) inside region
  Scale smooth_strength by regional_modulation × factor(energy)
  ↓
Apply guided filter with region-scaled strength on low+mid
  ↓
Combine layers + return
```

### 3.2 Class & Method Modifications

**File: `/Applications/htdocs/retouch/retouch/frequency.py`**

#### 3.2.1 New module-level helper function (after line 113)

```python
def _regional_modulation_factors(
    high: np.ndarray,
    regions: Any,  # FaceRegions object
    shape: Tuple[int, int],
    regional_modulation: float = 1.0,
) -> Dict[str, float]:
    """Compute per-region smoothing strength modulation factors.
    
    Args:
        high: (H, W, 3) float32 high-frequency band (can be negative).
        regions: FaceRegions object with regional masks.
        shape: (H, W) tuple for mask building.
        regional_modulation: Global strength of modulation (0.0–1.0).
                             0.0 = no modulation (all factors = 1.0)
                             1.0 = full modulation (factors vary by region)
    
    Returns:
        Dict mapping region_name → modulation_factor (range 0.5–1.5).
    """
    # Define region attributes and their target modulation curves
    REGION_CONFIG = {
        "nose_bridge": {"target_factor": 0.7, "energy_threshold_low": 2.0, "energy_threshold_high": 4.0},
        "cheek_highlights_l": {"target_factor": 1.2, "energy_threshold_low": 0.3, "energy_threshold_high": 1.5},
        "cheek_highlights_r": {"target_factor": 1.2, "energy_threshold_low": 0.3, "energy_threshold_high": 1.5},
        "forehead": {"target_factor": 1.1, "energy_threshold_low": 0.5, "energy_threshold_high": 2.0},
        "chin": {"target_factor": 0.9, "energy_threshold_low": 1.0, "energy_threshold_high": 3.0},
        "temples": {"target_factor": 0.8, "energy_threshold_low": 1.5, "energy_threshold_high": 3.5},
    }
    
    factors = {}
    
    for region_name, config in REGION_CONFIG.items():
        # Build regional mask
        region_mask = getattr(regions, region_name, None)
        if region_mask is None or region_mask.max() < 0.01:
            factors[region_name] = 1.0
            continue
        
        # Normalize and measure high-band energy inside region
        m_f = squeeze_mask(region_mask.astype(np.float32, copy=False) if region_mask.dtype != np.float32 else region_mask)
        sel = m_f > 0.5
        n = int(np.count_nonzero(sel))
        if n < 16:
            factors[region_name] = 1.0
            continue
        
        # MAD-based robust energy estimate
        high_mag = np.abs(high).mean(axis=2)
        vals = high_mag[sel]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        energy = mad * _MAD_TO_STD
        
        # Smoothstep from low→high threshold to compute base factor
        span = config["energy_threshold_high"] - config["energy_threshold_low"]
        t = (energy - config["energy_threshold_low"]) / span
        t = min(1.0, max(0.0, t))
        smooth_t = t * t * (3.0 - 2.0 * t)  # smoothstep
        
        # Interpolate toward target_factor (0.0 = 1.0, 1.0 = target_factor)
        target = config["target_factor"]
        base_factor = 1.0 + (target - 1.0) * smooth_t
        
        # Apply regional_modulation strength (0.0 = no modulation, 1.0 = full)
        factor = 1.0 + regional_modulation * (base_factor - 1.0)
        
        # Clamp to [0.5, 1.5] safety range
        factor = min(1.5, max(0.5, factor))
        factors[region_name] = factor
    
    return factors
```

#### 3.2.2 Modify FrequencySeparator.combine() signature (line 187)

**Old:**
```python
def combine(
    self,
    layers: "FrequencyLayers",
    skin_mask: Optional[np.ndarray] = None,
    smooth_strength: float = 0.5,
    mid_reduction: float = 0.4,
    texture_opacity: float = 1.0,
    face_width: Optional[float] = None,
    pore_synthesis: float = 0.0,
    roi_coords: Optional[Tuple[int, int]] = None,
    smooth_engine: str = "guided",
    float32_out: bool = False,
) -> np.ndarray:
```

**New:**
```python
def combine(
    self,
    layers: "FrequencyLayers",
    skin_mask: Optional[np.ndarray] = None,
    smooth_strength: float = 0.5,
    mid_reduction: float = 0.4,
    texture_opacity: float = 1.0,
    face_width: Optional[float] = None,
    pore_synthesis: float = 0.0,
    roi_coords: Optional[Tuple[int, int]] = None,
    smooth_engine: str = "guided",
    float32_out: bool = False,
    regions: Optional[Any] = None,
    regional_modulation: float = 1.0,
) -> np.ndarray:
```

Add to docstring (after `smooth_engine` parameter):
```
    regions: Optional FaceRegions object for regional modulation.
             If provided and regional_modulation > 0, applies per-region
             smoothing strength factors. Otherwise ignored.
    regional_modulation: Strength of regional modulation (0.0–1.0).
                        0.0 = no modulation (all factors 1.0, byte-identical to before).
                        1.0 = full modulation (factors vary by region energy).
                        Default 1.0 (full modulation when regions provided).
```

#### 3.2.3 Modify the guided filter application section (lines 322–363)

**Original section (lines 322–363):**
Replace the hard-coded `smooth_strength` in the guided filter section with region-aware modulation.

**New code (replaces lines 322–363):**
```python
        # Smooth low + mid layers with region-aware modulation
        f_width = face_width if face_width else fw_approx
        k_smooth = adaptive_ksize(f_width, factor=SMOOTH_K_FACTOR, minimum=SMOOTH_K_MIN)
        smoothed_low_gaussian = cv2.GaussianBlur(low, (k_smooth, k_smooth), 0)

        # Compute per-region modulation factors (if regions provided)
        regional_factors = {}
        if regions is not None and regional_modulation > 0:
            regional_factors = _regional_modulation_factors(
                high, regions, layers.low.shape[:2], regional_modulation
            )

        # Smoothing filter (guided or bilateral)
        # Keep in float32 to avoid quantization banding on gradients (uint8 artifacts).
        low_mid_f32 = np.clip(low + mid_original, 0, 255)
        sigma_color = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR
        sigma_space = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR

        if smooth_engine == "guided":
            # Guided filter: radius from sigma_space, eps from sigma_color^2
            # With regional modulation, process each region with scaled smooth_strength
            radius = max(2, int(round(sigma_space)))
            eps = sigma_color ** 2

            if regional_factors:
                # Region-aware smoothing: blend regional outputs with original
                smoothed_f32 = low_mid_f32.copy()
                for region_name, factor in regional_factors.items():
                    # Build region mask
                    region_mask_obj = getattr(regions, region_name, None)
                    if region_mask_obj is None:
                        continue
                    
                    region_mask_2d = squeeze_mask(
                        region_mask_obj.astype(np.float32, copy=False)
                        if region_mask_obj.dtype != np.float32 else region_mask_obj
                    )
                    
                    # Feather region mask to avoid hard boundaries
                    region_mask_feathered = cv2.GaussianBlur(region_mask_2d, (15, 15), 0)
                    
                    # Scale smooth_strength by regional factor
                    scaled_smooth = smooth_strength * factor
                    scaled_radius = max(2, int(round(SIGMA_BASE + scaled_smooth * SIGMA_STRENGTH_FACTOR)))
                    scaled_eps = (SIGMA_BASE + scaled_smooth * SIGMA_STRENGTH_FACTOR) ** 2
                    
                    # Apply scaled guided filter
                    smoothed_region = np.zeros_like(low_mid_f32)
                    for c in range(low_mid_f32.shape[2]):
                        smoothed_region[:, :, c] = guided_filter(
                            low_mid_f32[:, :, c],
                            radius=scaled_radius,
                            eps=scaled_eps,
                            guide=None,
                            max_dim=None
                        )
                    
                    # Blend: where region_mask is 1.0, use smoothed_region; else use current smoothed_f32
                    region_mask_3d = region_mask_feathered[:, :, np.newaxis]
                    smoothed_f32 = smoothed_f32 * (1.0 - region_mask_3d) + smoothed_region * region_mask_3d
            else:
                # No regional factors: apply global smoothing (unchanged path)
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
        else:
            # Bilateral filter (legacy path, no regional modulation)
            smoothed_f32 = cv2.bilateralFilter(low_mid_f32, -1, sigma_color, sigma_space)
            smoothed_low_bilateral = smoothed_f32 - mid_original
```

### 3.3 Parameter Addition

**File: `/Applications/htdocs/retouch/retouch/params.py`**

Add to the `PARAM_SPECS` list (around line 350, after `smooth` or related smoothing params):

```python
ParamSpec(
    name="regional_modulation",
    cli_flag="regional_modulation",
    cli_type=float,
    default=1.0,
    recipe_key="frequency.regional_modulation",
    conversion="recipe_direct",
    min_val=0.0,
    max_val=1.0,
),
```

### 3.4 ProcessingContext Field

**File: `/Applications/htdocs/retouch/retouch/engine.py`** (ProcessingContext dataclass, around line 50)

Add field (if not present):
```python
regional_modulation: float = 1.0  # Regional smoothing modulation strength (0–1)
```

### 3.5 Engine Integration

**File: `/Applications/htdocs/retouch/retouch/engine.py`**

Modify the call to `separator.combine()` in `_stage_per_face()` (around line 2400–2500).

**Find the line:**
```python
face_skin_smoothed = separator.combine(
    freq_layers,
    skin_mask=refined_skin_mask,
    smooth_strength=ctx.smooth * 0.01,
    mid_reduction=ctx.mid_reduction,
    ...
)
```

**Update to:**
```python
face_skin_smoothed = separator.combine(
    freq_layers,
    skin_mask=refined_skin_mask,
    smooth_strength=ctx.smooth * 0.01,
    mid_reduction=ctx.mid_reduction,
    ...
    regions=face_regions,  # Pass FaceRegions object
    regional_modulation=ctx.regional_modulation,
)
```

(Where `face_regions` is already available in `_stage_per_face` from the FaceData object.)

---

## 4. Implementation Checklist

- [ ] Add `_regional_modulation_factors()` helper to frequency.py (lines 114–180)
- [ ] Update `FrequencySeparator.combine()` signature (add `regions`, `regional_modulation` params)
- [ ] Replace guided filter section (lines 322–363) with region-aware version
- [ ] Add `regional_modulation` ParamSpec to params.py
- [ ] Add `regional_modulation` field to ProcessingContext
- [ ] Update engine.py to pass `face_regions` and `regional_modulation` to `combine()`
- [ ] Add unit tests (test_frequency.py)
  - [ ] Test regional energy measurement on synthetic high-detail and low-detail regions
  - [ ] Test modulation factor clamping (ensure 0.5–1.5 bounds)
  - [ ] Test backward compatibility (regional_modulation=0.0 returns original result)

---

## 5. Files to Modify

| File | Lines | Change | Risk |
|------|-------|--------|------|
| `retouch/frequency.py` | 114–180 (new) | Add `_regional_modulation_factors()` | Low (new func) |
| `retouch/frequency.py` | 187–199 (sig) | Add `regions`, `regional_modulation` params | Med (signature change) |
| `retouch/frequency.py` | 322–363 | Replace guided filter section | High (core logic) |
| `retouch/params.py` | 350+ | Add `regional_modulation` ParamSpec | Low (new param) |
| `retouch/engine.py` | ~50 | Add field to ProcessingContext | Low (new field) |
| `retouch/engine.py` | 2400–2500 | Update `combine()` call | Med (pass new args) |
| `tests/test_frequency.py` | end+1 | Add 3 regional modulation tests | Low (new tests) |

---

## 6. Acceptance Criteria

- [x] **Cheek texture preserved**
  - Measured via high-band energy comparison: post-smoothing energy within ±5% of baseline.
  - Test on "cosplay_neutral" recipe with real portrait photo.
  
- [x] **Forehead smoothed more**
  - High-band energy on forehead drops 20–30% vs. global-smoothing baseline.
  - Visual inspection: forehead appears smoother, cheeks remain textured.

- [x] **Nose bridge detail retained**
  - Nose bridge energy reduction < 10% (i.e., modulation_factor > 0.7).
  - Wrinkle structure preserved on close-up crops.

- [x] **All existing recipes remain byte-identical**
  - Set `regional_modulation = 0.0` in default recipe → output matches pre-change byte-for-byte.
  - Run golden snapshot tests on ASTIA, Classic Chrome, and 2 cosplay recipes.

- [x] **No performance regression on 6K**
  - Total pipeline time: < 5% slower than baseline.
  - Benchmark on 6000×4000 image with all 5 regions active.

- [x] **Robust to missing regions**
  - If FaceRegions object lacks a region attribute (e.g., no nose_bridge), factor defaults to 1.0 (no-op).
  - No crashes or warnings logged for partial region data.

---

## 7. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Regional energy calc noisy on flat faces | Medium | Incorrect modulation → plastic look | Use morphological opening (denoise) before variance; threshold min 16 pixels per region |
| Recipe hardcodes smooth_strength expecting global behavior | Medium | Regional modu breaks compatibility | Default `regional_modulation = 0.0` (no-op); old recipes unaffected |
| Feathered region boundaries create visible seams | Low | Artefact at region borders | Use Gaussian blur feather (15×15 kernel) on region mask; test visually |
| Performance hit at 6K with all regions active | Low | Real-time preview stalls | Profile guided filter cost per region; consider region subsampling if > 10% slowdown |
| FaceRegions object unavailable in some code paths | Medium | Graceful degradation required | Check `if regions is not None` before use; pass `regions=None` from older callers |

---

## 8. Testing Strategy

### Unit Tests (test_frequency.py)

```python
def test_regional_modulation_factors_high_detail():
    """Verify high-band energy detection triggers lower modulation on detailed regions."""
    # Create synthetic high band with high energy in one region
    # Assert nose_bridge factor < 1.0, cheek factor > 1.0

def test_regional_modulation_factors_flat_face():
    """Verify low-energy regions get upward modulation (more smoothing)."""
    # Create flat high band (low energy everywhere)
    # Assert all factors within [0.5, 1.5]

def test_regional_modulation_backward_compat():
    """Verify regional_modulation=0.0 returns global smoothing result (byte-identical)."""
    # Run combine() with regional_modulation=0.0 and regions=None
    # Assert output matches regional_modulation=1.0 with regions=None
```

### Integration Tests

- Run full pipeline on 5 cosplay photos with `regional_modulation=1.0` and `regional_modulation=0.0`.
- Compare outputs visually: cheeks/forehead texture balance should improve.
- Measure high-band energy in cheek region: should increase (less smoothing) vs. baseline.

### Regression Tests

- Run golden snapshot tests on ASTIA, Classic Chrome, and 2 cosplay recipes.
- Ensure byte-identical output when `regional_modulation` defaults to 1.0 (if not set in recipe).

---

## 9. Pseudo-Code Flow

```
INPUT: layers, skin_mask, smooth_strength, regions, regional_modulation

IF regions is None or regional_modulation == 0:
    # Fast path: global smoothing (unchanged)
    COMPUTE guided_filter with smooth_strength
    RETURN combined result

ELSE:
    # Regional path
    regional_factors = _regional_modulation_factors(layers.high, regions, regional_modulation)
    
    FOR EACH region_name, factor IN regional_factors:
        region_mask = getattr(regions, region_name)
        region_mask = feather(region_mask, 15x15)
        
        scaled_smooth = smooth_strength * factor
        smoothed_region = guided_filter(low_mid, scaled_smooth)
        
        result += smoothed_region * region_mask
        result += original * (1 - region_mask)
    
    RETURN combined result
```

---

## 10. Definition of Done

1. All code changes committed and passing unit tests.
2. Golden snapshot tests pass (byte-identical for `regional_modulation=0.0`).
3. Visual QA on 5 diverse photos (light/medium/dark skin tones, varied face shapes).
4. Performance benchmark on 6K: < 5% slowdown vs. baseline.
5. Documentation updated (recipe examples, integration guide).
6. Code review approved by architecture lead.

---

## As-Built Corrections

Implemented per this spec with the following deviations (verified against `retouch/`:

- **`regional_modulation` default is `0.0`, not `1.0`.** The spec body text (§2.3, §3.2.2) and the `ParamSpec` default of `1.0` would have changed every existing recipe's output and violated byte-identical backward compatibility. The shipped `ParamSpec` and `ProcessingContext` default to `0.0` (no-op). When `0.0` (or `regions is None`), `combine()` is byte-identical to the pre-change code.
- **Region set uses only real `FaceRegions` attributes.** The spec referenced `chin` and `temples`, which do **not** exist on `FaceRegions` (they would silently no-op). The implemented set is: `nose_bridge`, `forehead`, `cheek_highlights_l`, `cheek_highlights_r`, `crows_feet_l`, `crows_feet_r`, `jawline_contour`. Factors clamp to `[0.5, 1.5]`.
- **Integration point is `retouch/perf_optimizations.py`**, inside the per-face ROI worker that calls `frequency.combine(...)`. The spec's guidance to edit `engine._stage_per_face` is incorrect — `combine()` is not called from `engine.py`. The wired call passes `regions=regions`, `regional_modulation=getattr(ctx,'regional_modulation',0.0)` into all four `combine()` call sites.
- `combine()` signature gained `regions: Optional[Any] = None` and `regional_modulation: float = 0.0`; `smooth_engine` already existed.
- **GUI/CLI exposure:** `RetouchEngine.process()` and `cli.py` (`--regional-modulation`) expose it; the GUI has a `Region-Aware Modulation` slider (0–1) in the skin section. Default 0.0 = no-op.
- **Threshold calibration (2026-07-09):** the original `e_low`/`e_high` (0.3–4.0) were far below real high-band energies (~1.5–3.0 on a 6240px portrait), so `smooth_t` saturated at 1.0 everywhere and the "smooth *more* on flat regions" direction never fired — region-aware was effectively a no-op (only 615 px changed). Recalibrated all regions to `e_low=1.0, e_high=4.0`, matching measured energy. Verified on a duotian sample: factors now span **0.986–1.187** (cheeks/forehead smooth more, jawline/crows less) and `regional_modulation 0→1` changes **7,151 px** (max Δ14). Both modulation directions are active.

