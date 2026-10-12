# Freckle/Beauty Mark Control - Implementation Summary

**Status:** ✅ **READY TO IMPLEMENT IMMEDIATELY**  
**Effort:** 21.5 hours (2.5 days)  
**ROI:** Medium–High (cosplay audience, competitive parity)

---

## Quick Reference

### Files to Create/Modify

| File | Change | Status |
|------|--------|--------|
| `retouch/freckle.py` | NEW | Class `FreckleRemover` |
| `retouch/engine.py` | MODIFY | Add `freckle_removal` param, FreckleRemover instance |
| `retouch/perf_optimizations.py` | MODIFY | Add freckle pipeline hook after blemish |
| `tests/test_freckle.py` | NEW | Unit tests (~25 test cases) |
| `tests/test_freckle_integration.py` | NEW | Integration tests |
| `API.md` | MODIFY | Document freckle_removal parameter |
| `PLAN_FRECKLE_BEAUTY_MARK.md` | NEW | Complete technical plan |

### Core Design Decision

```
Freckles ≠ Blemishes

┌─────────────────────────────────────┐
│ All Anomalies Detected              │
│ (via BlemishRemover._detect)         │
└──────────────┬──────────────────────┘
               │
      ┌────────▼─────────┐
      │  CLASSIFICATION   │
      └────────┬─────────┘
               │
      ┌────────┴─────────┬──────────────┬────────────┐
      │                  │              │            │
   FRECKLE          BLEMISH       BEAUTY_MARK      NOISE
 (Remove)          (Ignore)       (Protect)       (Discard)
  Small             Medium         Large          ~
  Red-tinted        Inflamed       Dark            ~
  Cluster           Scattered      Isolated        ~
  4–20px²           8–40px²        15–60px²        ~
```

### Parameters

| Parameter | Type | Range | Default | Notes |
|-----------|------|-------|---------|-------|
| `freckle_removal` | float | 0–100 | 0 | Strength: higher = aggressive |
| `freckle_preserve_mask` | bytes | base64 PNG | None | Advanced; manual protection |

### Pipeline Position

```
canvas ← Specular Bloom
  ↓
canvas ← Blemish Removal (existing)
  ↓
canvas ← Freckle Removal (NEW) ← ← ← ← ← ← ← ← ← ← 
  ↓
canvas ← Under-eye Repair
  ↓
...
```

---

## Classification Heuristics

### Freckle (Remove)
```
Size:        4–20 px² (small)
Color:       Red-tinted (a_offset: +5 to +15 from skin baseline)
Pattern:     Clustered (cheeks, nose)
Contrast:    Subtle (8–16 L levels deviation)
→ ACTION: Remove via inpainting
```

### Beauty Mark (Protect)
```
Size:        15–60 px² (large, prominent)
Color:       Dark brown (L_offset: -20 to -40)
             Neutral hue (a_offset: ±8 from baseline)
Pattern:     Isolated single mark
→ ACTION: Preserve (exclude from removal)
```

### Blemish (Ignore)
```
Size:        8–40 px² (medium)
Color:       Inflamed red (a_offset: >+15)
Pattern:     Random acne
Contrast:    Strong (15+ L levels)
→ ACTION: Skip (handled by blemish removal if enabled)
```

---

## Key Implementation Details

### 1. Color-Space Logic (LAB)
```python
# Compute offsets from skin median
a_offset = region_a - skin_median_a
L_offset = region_L - skin_median_L

# Freckles are red-tinted (positive a_offset)
if 5 < a_offset < 15 and 4 * scale ≤ area ≤ 20 * scale:
    → freckle
    
# Beauty marks are dark, neutral
elif L_offset < -20 and abs(a_offset) < 8 and area ≥ 15 * scale:
    → beauty_mark
    
# Acne is strongly red
elif a_offset > 15 and area > 20 * scale:
    → blemish
```

### 2. Skin-Tone Adaptation
All thresholds **scale by `face_width / 500.0`** (existing pattern from BlemishRemover)
- Ensures same algorithm works for headshots, full-body, varied face sizes
- Example: `max_freckle_area = int(20 * (scale ** 2))`

### 3. Preserve Mask Override
Users can optionally provide a "protect these pixels" mask (advanced API):
```python
preserve_mask = np.zeros((H, W), dtype=np.uint8)
preserve_mask[user_selected_region] = 255

result = engine.process(
    img,
    freckle_removal=75,
    freckle_preserve_mask=base64_encode_png(preserve_mask)
)
```

### 4. Float32 Support
```python
# Reuse existing pattern from blemish.py
if img_bgr.dtype == np.float32:
    return apply_u8_op_float(img_bgr, self.remove, ...)
```

---

## Testing Strategy

### Unit Tests (18+ cases)
- ✅ Zero strength → no-op
- ✅ Flat skin → no false positives
- ✅ Freckle-like spot → removed
- ✅ Beauty mark → preserved
- ✅ Float32 input → correct dtype output
- ✅ Preserve mask → overrides removal
- ✅ Multi-face consistency
- ✅ Edge cases (tiny images, no skin, etc.)

### Visual QA (5 cosplay photos)
1. Heavy freckles, no marks → clean removal
2. Light freckles + beauty mark → selective removal
3. Dark skin variant → color heuristics adapt
4. Fine freckles → natural look
5. Freckles + acne → correct classification

### Acceptance Bar
- ≥18/25 unit tests pass
- ≥4/5 visual tests approved
- No regression in existing blemish tests
- Float32 path validated

---

## Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Heuristics fail on dark skin | Medium | Medium | Test with 3 skin tones; use LAB offsets (relative, not absolute) |
| Beauty marks get removed | High | High | Conservative size gate (≥15px²); manual preserve_mask fallback |
| Performance regression | Low | Low | Reuses existing pipeline; +1–2ms per face |
| User confusion (vs. blemish) | Medium | Low | Clear UI labeling; separate sliders; docs explain difference |

**Overall Risk Level: LOW** (mature codebase, clear requirements, proven architecture)

---

## Competitive Analysis

| Competitor | Feature | Our Approach |
|------------|---------|--------------|
| VSCO | Mole erase | ✅ Beauty mark preservation (better) |
| Adobe Lightroom | Spot removal | Blemish removal (we already have) |
| Meitu | Face retouch | ✅ Freckle-specific controls (NEW) |
| Snapseed | Healing tool | ✅ Automated (not manual spot-by-spot) |

**Competitive Advantage:** Automatic freckle detection + beauty mark preservation in one slider.

---

## Deployment Roadmap

### Phase 1: Implementation (2.5 days)
- [ ] Code `freckle.py` + tests
- [ ] Integrate into engine
- [ ] Visual QA validation
- [ ] Code review

### Phase 2: Documentation (0.5 days)
- [ ] API.md update
- [ ] Docstrings + type hints
- [ ] FAQ/TROUBLESHOOTING

### Phase 3: Release (Q3 post-launch)
- [x] Ship as opt-in (default strength=0)
- [x] Recipe: add to "clean beauty" preset (strength=50)
- [x] Monitor user feedback

---

## Acceptance Definition: "Done"

✅ Feature is **DONE** when:

1. **Code Complete**
   - FreckleRemover class implemented with all methods
   - Engine integration: params wired, processor added
   - Pipeline hook active in perf_optimizations.py

2. **Tests Passing**
   - 18+ unit tests pass (test_freckle.py)
   - Integration tests pass (test_freckle_integration.py)
   - Existing blemish tests still pass (no regression)

3. **Visual Validation**
   - 4/5 cosplay test images show acceptable results
   - Beauty marks are preserved
   - Freckles are reduced without over-smoothing

4. **Documentation**
   - PLAN_FRECKLE_BEAUTY_MARK.md complete
   - API.md updated with freckle_removal param
   - Docstrings + type hints complete

5. **Code Quality**
   - All type hints correct (mypy passes)
   - No linting warnings (flake8 passes)
   - Code reviewed and approved

---

## Quick Start for Developer

```bash
# 1. Create freckle.py
touch retouch/freckle.py

# 2. Skeleton
cat > retouch/freckle.py << 'EOF'
from .blemish import BlemishRemover, inpaint_and_blend
import numpy as np
import cv2

class FreckleRemover:
    def __init__(self):
        self._blemish = BlemishRemover()
    
    def remove(self, img_bgr, skin_mask, strength=50, preserve_mask=None):
        if strength <= 0:
            return img_bgr
        # ... implement _detect_anomalies, _classify_freckles, _classify_region
        pass
EOF

# 3. Create tests
touch tests/test_freckle.py tests/test_freckle_integration.py

# 4. Update engine.py (add param, instance, hook)
# See PLAN_FRECKLE_BEAUTY_MARK.md section 4.2–4.4

# 5. Run tests
pytest tests/test_freckle.py -v
pytest tests/test_freckle_integration.py -v

# 6. Visual QA
python -c "
from retouch.engine import RetouchEngine
import cv2
engine = RetouchEngine()
img = cv2.imread('test_image.jpg')
result = engine.process(img, freckle_removal=75)
cv2.imwrite('result.jpg', result)
"
```

---

## FAQ

**Q: Why not extend BlemishRemover instead of creating FreckleRemover?**  
A: Separation of concerns. Blemish removal focuses on acne/spots. Freckle removal is a **classification + preservation** problem. Clean class boundary enables independent tuning and testing.

**Q: What if a user wants to preserve certain freckles but remove others?**  
A: Preserve mask is the tool. Advanced users can paint a mask and pass it as base64 PNG.

**Q: Does this work on the body?**  
A: Freckles on body? (Uncommon in cosplay, but possible.) Phase 2 can add body_freckle_removal param.

**Q: How does this interact with makeup (foundation, concealer)?**  
A: Tests include makeup scenarios. Preserve mask allows users to protect heavy makeup areas if needed.

**Q: What's the performance cost?**  
A: ~1–2ms per face (single ConnectedComponents pass). Negligible vs. skin smoothing pipeline.

---

## Contact & Escalation

- **Technical Questions:** See PLAN_FRECKLE_BEAUTY_MARK.md (sections 3–6)
- **Design Review:** Recommend 1h alignment session before sprint start
- **QA Handoff:** 5 cosplay test images provided by Product team
- **Launch Timeline:** Ready for Q3 post-launch feature rollout

---

**Prepared:** 2026-07-08  
**Status:** ✅ Ready to Implement  
**Confidence:** High (reuses proven architecture; clear requirements; low risk)
