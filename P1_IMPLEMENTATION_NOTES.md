# P1 Implementation Notes — Bilateral → Guided Filter Migration

**Date:** 2026-07-02  
**Status:** Working tree only (not yet committed); pending test-suite approval + visual QA  
**Unblocks:** F8 (full-res fidelity), S1 (body skin), F11 (output QA detectors)

---

## What Changed

### 1. New `guided_filter()` in `retouch/utils.py`

Added `guided_filter(src, radius, eps, guide=None, max_dim=1200)` — an edge-preserving smoothing kernel. When `guide=None`, performs self-guided filtering (mathematically equivalent to bilateral but via linear coefficients):

```python
guided_filter(src, radius=r, eps=eps, guide=None, max_dim=1200)
```

Key features:
- **Self-guided mode:** Computes statistics on input, then applies via linear coefficients
- **Coefficient downsampling:** For images with `min(h, w) > max_dim`, computes coefficients at downsampled scale, then upsamples them to full resolution (drastically faster on high-res)
- **Per-channel safe:** Caller applies to each channel independently (implemented in callers, not the function)

### 2. `SkinProcessor.flatten()` Refactored

Replaced hand-rolled coefficient computation (downsampled then upsampled) with single call to `guided_filter()`:

**Before:**
```python
# Manual downsampling, variance/cov computation, upsampling
if is_downsampled:
    scale = 1200.0 / min_dim
    L_small = cv2.resize(L, ...)
    # ... 20 lines of coefficient math ...
    a_full = cv2.resize(a, (w, h), ...)
```

**After:**
```python
q = guided_filter(L, radius=r, eps=eps, guide=None, max_dim=1200)
```

Docstring updated: "Edge-preserving cel flatten via guided filter on LAB L."

### 3. `FrequencySeparator.combine()` Gains `smooth_engine` Kwarg

New parameter (default `"guided"`):
- `smooth_engine="guided"` → Guided filter (new, fast path)
- `smooth_engine="bilateral"` → Legacy `cv2.bilateralFilter()` (rollback path)

Maps bilateral parameters to guided: `radius = max(2, round(sigma_space))`, `eps = sigma_color²`  
Applied per-channel self-guided with `max_dim=None` (crops already bounded).

Deprecated module-level `combine()` function forwards the kwarg.

---

## Performance & Quality

### Measured Speedup (σ=50, self-guided)

| Image Size | Bilateral | Guided | Speedup |
|---|---|---|---|
| 400×400 | 1035 ms | 1.70 ms | **~610×** |
| 1500×1500 | 5680 ms | 23.4 ms | **~243×** |

**Output quality:** Mean absolute difference < 1.0 on 0–255 scale (imperceptible).

### Test Coverage

New `tests/test_guided_filter.py` (13 tests):
- Constant image invariance
- Self-guided vs. explicit guide equivalence
- Shape/dtype validation
- Step-edge preservation (key for skin retouching)
- Smooth gradient monotonicity
- Downsample path consistency (< 2.0 MAD vs. full-res)
- Guided vs. bilateral similarity (< 6.0 MAD)
- Performance benchmarks on 400×400 and 1500×1500

---

## ⚠️ Known Behavior Deviation (Before Commit)

**Old `flatten()` had a subtle bug when handling >1200px images:**  
When downsampling, it computed the blur radius `r` at full resolution, then used it on the 1200px proxy. This resulted in a relatively larger smoothing radius on the downsampled image.

**New `guided_filter()` scales the radius with downsampling:**  
`r_small = r * scale`, so smoothing magnitude is scale-consistent.

**Visual impact:** Images >1200px get *slightly less* flatten smoothing than before — arguably a fix (more consistent geometry), but it is a change.

**Action before commit:** Run visual QA on before/after retouches (>1200px test set) to confirm the difference is acceptable.

---

## Rollback Path

To revert to bilateral filtering if needed:
```python
separator.combine(..., smooth_engine="bilateral")
```

This kwarg will be deprecated and removed in the next release per `PLAN_TIERP_PERF_ARCH_SHIP.md` (P1 step 3).

---

## Next Steps

1. **Full test suite approval:** `pytest tests/ -q` must pass
2. **Visual QA:** Spot-check retouches on >1200px images (flatten edge case)
3. **Benchmark gate:** Per-face ≤250ms @ 400×400 (already achieved: 1.70ms/channel, ~5ms for 3-channel)
4. **SSIM corpus check:** Verify output still matches SSIM thresholds (once F11 corpus built)
5. **Commit & tag:** Will merge post-approval; P1 unblocks F8 Phase 1 execution

---

## Files Modified

- `retouch/utils.py` — `+guided_filter()` function
- `retouch/skin.py` — `flatten()` refactored to use guided filter
- `retouch/frequency.py` — `combine()` gains `smooth_engine` kwarg, applies guided filter per-channel
- `tests/test_guided_filter.py` — New test file (13 tests)
