# IMPLEMENTATION_ROADMAP.md

## Master Roadmap: Coordinated Implementation of Beauty Improvements

**Created:** July 2026  
**Target Completion:** August 2026 (phased delivery)  
**Coordinating Architect:** Fable-level reasoning  
**Handoff Recipients:** Opus/Haiku implementation agents  

---

## 1. Executive Summary

Four feature specifications have been created to improve face retouching quality:

1. **REGION_AWARE_SMOOTHING.md** — Region-aware smoothing strength (6 hours)
2. **ANISOTROPIC_DIFFUSION.md** — Anisotropic diffusion option (10 hours)
3. **EYE_SHADOW_SMOOTHING.md** — Dedicated eye shadow smoothing (4 hours)
4. **FRECKLE_BEAUTY_MARK_CONTROL.md** — Freckle/beauty mark selective control (20 hours)

**Total Effort:** ~40 hours (~5 days, distributed across 2 weeks)

**Architecture Principle:** Features are **modular and independent**. Each can be implemented, tested, and deployed separately. However, ordering and coordination points exist.

---

## 2. Execution Plan: Phased Delivery

### Phase A: Foundation (Parallel, Days 1–2, 10 hours total)

**Goal:** High-quality fundamentals that improve most recipes immediately.

**Features:**
- ✅ **REGION_AWARE_SMOOTHING** (6 hours)
- ✅ **EYE_SHADOW_SMOOTHING** (4 hours)

**Why parallel?**
- No dependencies between them.
- Both modify skin processing independently (different regions/methods).
- Early delivery provides immediate ROI.
- Enables testing and visual QA in parallel.

**Acceptance Gate:**
- All unit tests passing (8 tests total)
- Golden snapshots match baseline (backward compat verified)
- Visual QA on 10 diverse photos (light/medium/dark skin, varied face shapes)
- Performance: < 2% slowdown on 6K pipeline

**Deliverables:**
- Updated `frequency.py` (region-aware smoothing)
- Updated `skin.py` (eye shadow smoothing)
- Updated `params.py` and `engine.py` (both features)
- Test coverage: `test_frequency.py` (+3 tests), `test_skin.py` (+2 tests)

---

### Phase B: Advanced Smoothing (Sequential, Days 3–4, 10 hours)

**Goal:** Professional-grade wrinkle/pore preservation via orientation guidance.

**Features:**
- ✅ **ANISOTROPIC_DIFFUSION** (10 hours)

**Why sequential?**
- Depends on validation of Phase A (no surprises in combine() method signature).
- Requires benchmark on 6K to ensure < 1s budget (performance-critical).
- Can reuse Phase A infrastructure (smooth_engine parameter, region-aware integration).

**Acceptance Gate:**
- Unit tests passing (4 tests: orientation, domain transform, downsampling, fallback)
- Benchmark on 6K: < 1000ms, target ~700ms
- Visual QA on 5 professional cosplay/fashion photos
  - Check: wrinkle preservation, no haloing, no banding
  - Compare guided vs. anisotropic side-by-side
- Graceful fallback when ximgproc unavailable (tested via monkeypatch)
- Backward compatibility: default smooth_engine="guided" → unchanged output

**Deliverables:**
- Updated `frequency.py` (anisotropic branch in combine())
- New helper functions (`_compute_orientation_field`, `_apply_domain_transform_anisotropic`, `_smooth_anisotropic_separable`)
- Benchmark test in `benchmark.py`
- Test coverage: `test_frequency.py` (+4 tests)

**Integration Notes:**
- Anisotropic can be combined with region-aware smoothing (Spec #1).
  - Regional factors still apply; anisotropic adds orientation guidance within each region.
  - Update Spec #1's guided filter call to support `smooth_engine="anisotropic"`.

---

### Phase C: Selective Cosmetics (Sequential, Days 5–7, 20 hours)

**Goal:** Preserve intentional beauty marks; remove freckles selectively.

**Features:**
- ✅ **FRECKLE_BEAUTY_MARK_CONTROL** (20 hours)

**Why sequential?**
- Most complex feature (new classifier, ~85% BlemishRemover reuse).
- Extensive unit testing required (18–25 tests).
- Visual QA on multi-skin-tone samples (light/medium/dark) critical.
- Can be deployed independently; not blocking other features.

**Acceptance Gate:**
- Unit tests passing (18–25 tests, >85% code coverage)
- Multi-skin-tone testing verified:
  - Light skin: freckles/marks classified correctly on 3+ photos
  - Medium skin: same
  - Dark skin: same (a-channel normalization robust)
- Visual QA on 5+ cosplay photos
  - No false positives (beauty mark removed as freckle)
  - No false negatives (freckle preserved as mark)
  - User preserve_mask override works as expected
- Backward compatibility: freckle_removal=0 → byte-identical output
- No BlemishRemover regression (all S2 tests pass)

**Deliverables:**
- New `freckle.py` module (~300 lines)
  - FreckleClassification, FreckleRemover classes
  - Classification logic (size/color/clustering)
  - remove() orchestration
- Updated `params.py` (freckle_removal, freckle_preserve_mask ParamSpecs)
- Updated `engine.py` (ProcessingContext fields, stage call in _stage_per_face)
- Test coverage: new `test_freckle.py` (~400 lines, 18–25 tests)

---

## 3. Detailed Timeline

### Week 1 (Mon–Fri)

| Day | Phase | Feature | Hours | Milestone |
|-----|-------|---------|-------|-----------|
| Mon | A | Region-aware smoothing | 6 | Code + unit tests (3) + visual QA (5 photos) |
| Mon–Tue | A | Eye shadow smoothing | 4 | Code + unit tests (2) + visual QA (5 photos) |
| **Tue EOD** | **Gate** | **Phase A Review** | 2 | All tests pass, backward compat verified |
| Wed–Thu | B | Anisotropic diffusion | 10 | Code + helpers + benchmark + unit tests (4) + visual QA (5 photos) |
| **Thu EOD** | **Gate** | **Phase B Review** | 1 | Performance budget met, fallback tested |
| Fri–Sat | C | Freckle/beauty mark | 20 | Code (~300 lines) + unit tests (18–25) + visual QA (5+ photos) |

### Week 2 (Mon–Wed)

| Day | Phase | Feature | Hours | Milestone |
|-----|-------|---------|-------|-----------|
| Mon–Wed | C | Freckle/beauty mark (continued) | 15 | Multi-skin-tone testing, user override validation |
| **Wed EOD** | **Gate** | **Phase C Review** | 1 | All classifications accurate, no regression |
| Thu | Integration | All phases | 3 | Verify no conflicts between features |
| **Fri EOD** | **Final** | **Release** | — | All features merged, comprehensive visual QA |

---

## 4. Dependency Map & Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE A (Independent)                     │
├──────────────────────┬──────────────────────────────────────┤
│ Region-Aware         │ Eye Shadow                           │
│ Smoothing            │ Smoothing                            │
│ (frequency.py)       │ (skin.py)                            │
│ 6 hours              │ 4 hours                              │
└──────────────────┬───┴─────────────────┬────────────────────┘
                   │                     │
          Updates combine()        Adds smooth_undereye_shadow()
          method signature         integration point
                   │                     │
                   └─────────┬───────────┘
                             │
                      GATE REVIEW
                             │
              ┌──────────────┴──────────────┐
              │                             │
        ┌─────▼──────────────────────────┐ │
        │     PHASE B (Sequential)       │ │
        │ Anisotropic Diffusion          │ │
        │ (frequency.py continued)       │ │
        │ 10 hours                       │ │
        │                                │ │
        │ Adds anisotropic branch to     │ │
        │ combine() method (if not       │ │
        │ already updated in Phase A)    │ │
        └──────────┬─────────────────────┘ │
                   │                       │
            Can integrate with Phase A:    │
            - Regional factors + anisotropic guidance
            - Update Spec #1 to support smooth_engine="anisotropic"
                   │                       │
            GATE REVIEW (Performance)      │
                   │                       │
        ┌──────────▼──────────────────────▼─┐
        │       PHASE C (Sequential)         │
        │ Freckle/Beauty Mark Control        │
        │ (freckle.py new module)            │
        │ 20 hours                           │
        │                                    │
        │ Independent of A/B                 │
        │ Runs before blemish (S2) or after  │
        │ as separate stage                  │
        └──────────┬───────────────────────┘
                   │
            GATE REVIEW (Classification accuracy)
                   │
         ┌─────────▼──────────────┐
         │ FINAL INTEGRATION      │
         │ All phases together    │
         │ 3 hours                │
         │                        │
         │ Verify no conflicts    │
         │ Performance testing    │
         │ Final visual QA        │
         └────────────────────────┘
```

---

## 5. Coordination Points & Conflict Avoidance

### 5.1 Signature Changes in frequency.py::combine()

**Risk:** Both Phase A and Phase B modify `combine()` signature.

**Mitigation:**
- **Phase A** adds: `regions`, `regional_modulation` parameters
- **Phase B** uses existing: `smooth_engine` parameter (update its docstring)
- **Order:** Implement Phase A first; Phase B extends without changing Phase A's additions
- **Backward Compat:** Both maintain default values (regional_modulation=1.0, smooth_engine="guided")

**Updated combine() signature (final):**
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
    smooth_engine: str = "guided",  # Phase B: extended to support "anisotropic"
    float32_out: bool = False,
    regions: Optional[Any] = None,  # Phase A: regional modulation
    regional_modulation: float = 1.0,  # Phase A: regional modulation strength
) -> np.ndarray:
```

### 5.2 Engine Integration Points

**Phase A & Phase B → engine.py::_stage_per_face()**

One call to `separator.combine()` with all parameters:
```python
face_skin_smoothed = separator.combine(
    freq_layers,
    skin_mask=refined_skin_mask,
    smooth_strength=ctx.smooth * 0.01,
    mid_reduction=ctx.mid_reduction,
    texture_opacity=ctx.texture_opacity,
    face_width=face_width,
    pore_synthesis=ctx.pore_synthesis,
    roi_coords=roi_coords,
    smooth_engine=ctx.smooth_engine,  # Phase B param (default "guided")
    float32_out=False,
    regions=face_regions,  # Phase A param
    regional_modulation=ctx.regional_modulation,  # Phase A param
)
```

**Phase A & Phase C → engine.py::_stage_per_face()**

Two separate calls (no conflict):
- `separator.combine()` for frequency smoothing (Phase A/B)
- `skin_processor.smooth_undereye_shadow()` for eye shadows (Phase A, lines ~2500)
- `freckle_remover.remove()` for freckle removal (Phase C, lines ~2400 or separate stage)

**Ordering in _stage_per_face():**
1. Frequency separation
2. Region-aware smoothing (via combine()) — Phase A
3. Eye shadow smoothing — Phase A
4. Freckle removal — Phase C (before or after, depends on preference)
5. Existing stages (blemish, wrinkle, gloss, etc.)

### 5.3 Parameter Additions (No Conflicts)

Each phase adds new parameters to params.py and ProcessingContext:

| Phase | Parameters | Conflicts? |
|-------|-----------|-----------|
| A | `regional_modulation` | No (new) |
| A | `undereye_shadow_strength` | No (new) |
| B | (extends `smooth_engine` docstring only) | No (existing param) |
| C | `freckle_removal`, `freckle_preserve_mask` | No (new) |

**Total new parameters:** 4 (no conflicts, all additive)

### 5.4 Test Isolation

Each phase's tests are independent:
- Phase A: `test_frequency.py` (region-aware tests) + `test_skin.py` (eye shadow tests)
- Phase B: `test_frequency.py` (anisotropic tests) — can coexist with Phase A tests
- Phase C: `test_freckle.py` (new module, independent)

**No test conflicts.** Cross-feature integration tests run in final gate.

---

## 6. Testing Strategy Across Phases

### 6.1 Unit Test Pyramid

```
            ▲
           /│\
          / │ \
         /  │  \  Integration Tests
        /   │   \ (Full pipeline, all features)
       /    │    \
      ├─────┼─────┤
      │     │     │ Component Tests
      │ A   │ B   │ (Phase-specific, e.g., anisotropic
      │ 5t  │ 4t  │  without region-aware)
      ├─┬───┴──┬──┤
      │A│  B   │C │ Unit Tests
      │8│ +C   │18│ (Per-method, no I/O)
      │t│ 22t  │+ │
      └─┴──────┴──┘

A = Phase A (8 unit tests)
B = Phase B (4 unit tests)
C = Phase C (18–25 unit tests)

Integration: All features in full pipeline (3 test scenarios)
```

### 6.2 Golden Snapshots

- **Phase A+B**: Update golden snapshots.
  - Old recipes (no new params) should match baseline exactly (backward compat).
  - New recipes with regional_modulation=1.0, smooth_engine="anisotropic" capture new behavior.

- **Phase C**: Freckle removal snapshots.
  - New, since freckle_removal=0 is default (no-op).
  - Byte-identical when freckle_removal=0.

**Snapshot Files:**
- `tests/golden_pipeline_snapshots.json` (existing, updated with Phase A/B)
- `tests/golden_freckle_snapshots.json` (new, for Phase C)

### 6.3 Visual QA Tracking

**Document in `/Applications/htdocs/retouch/docs/VISUAL_QA.md`:**

| Phase | Feature | Photos Tested | Result | Notes |
|-------|---------|---------------|--------|-------|
| A | Region-aware | 10 (light/med/dark) | ✓ | Cheek/forehead balance improved |
| A | Eye shadow | 10 (light/med/dark) | ✓ | Under-eye softened, no hollowing |
| B | Anisotropic | 5 (professional) | ✓ | Wrinkles preserved, no halos |
| C | Freckle removal | 5+ (light/med/dark) | ✓ | Marks preserved, freckles removed |

---

## 7. Performance Budgets

### 7.1 Per-Feature Overhead

| Feature | Expected Overhead | Budget | Risk |
|---------|------------------|--------|------|
| Region-aware smoothing | ~40ms (6K) | < 50ms | Low (lightweight mask operations) |
| Eye shadow smoothing | ~20ms (6K) | < 30ms | Low (small region, guided filter) |
| Anisotropic diffusion | ~500ms (6K, downsampled) | < 1000ms | Med (orientation detection + domain transform) |
| Freckle removal | ~100ms (6K) | < 150ms | Low (reuses BlemishRemover detector) |
| **Total** | **~660ms** | **< 1200ms** | **Med** |

**Baseline pipeline (before features):** ~2–3 seconds (6K image, full recipe)

**Expected total after all phases:** ~3–4 seconds (6K, full recipe with all features active)

### 7.2 Mitigation Strategies

- **Anisotropic:** Downsample to 2048px for processing (already budgeted in Spec #2).
- **Freckle removal:** Cap anomaly detection to 1000 per face (99th percentile).
- **Region-aware:** Profile on real recipes; if > 50ms, optimize mask operations.

---

## 8. Handoff Checklist for Implementation Agents

### For Phase A Agent (Opus/Haiku)

- [ ] Read REGION_AWARE_SMOOTHING.md completely
- [ ] Read EYE_SHADOW_SMOOTHING.md completely
- [ ] Understand frequency.py architecture (layers, combine method)
- [ ] Understand skin.py architecture (SkinProcessor, mask building)
- [ ] Verify guided_filter utility function exists and works
- [ ] Implement Phase A code (both features in parallel)
- [ ] Run all unit tests (8 total)
- [ ] Run golden snapshot tests (backward compat)
- [ ] Perform visual QA on 10 photos (document in VISUAL_QA.md)
- [ ] Gate review: performance < 2% overhead, all tests green, visual QA approved
- [ ] Open PR with Phase A changes

### For Phase B Agent (Opus/Haiku)

- [ ] Read ANISOTROPIC_DIFFUSION.md completely
- [ ] Review Phase A PR (understand new combine() signature)
- [ ] Understand cv2.ximgproc availability and fallback strategy
- [ ] Implement Phase B code (orientation detection, domain transform, branch in combine())
- [ ] Run all unit tests (4 total)
- [ ] Run benchmark on 6K image (must be < 1000ms, target ~700ms)
- [ ] Perform visual QA on 5 professional photos
- [ ] Gate review: performance budget met, fallback tested, visual QA approved
- [ ] Open PR with Phase B changes

### For Phase C Agent (Opus/Haiku)

- [ ] Read FRECKLE_BEAUTY_MARK_CONTROL.md completely
- [ ] Study BlemishRemover class (reuse ~85%)
- [ ] Understand LAB color space and skin tone normalization
- [ ] Implement FreckleRemover class (~300 lines)
- [ ] Run all unit tests (18–25 total)
- [ ] Perform multi-skin-tone visual QA (light/medium/dark, 5+ photos)
- [ ] Validate user preserve_mask override
- [ ] Gate review: classification accuracy on all skin tones, visual QA approved, no BlemishRemover regression
- [ ] Open PR with Phase C changes

### Final Integration Agent (Opus/Haiku)

- [ ] Merge Phase A/B/C PRs into develop
- [ ] Run full test suite (no regressions)
- [ ] Run performance benchmark (full pipeline on 6K with all features active)
- [ ] Final visual QA across all features (5 comprehensive photos)
- [ ] Update documentation (parameter reference, recipe examples, troubleshooting guide)
- [ ] Merge to main for release

---

## 9. Communication Plan

### 9.1 Pre-Implementation

- [ ] Share all 4 specs + roadmap with implementation team
- [ ] Pair programming session (30min) to review architecture
- [ ] Q&A: clarify any ambiguities in specs

### 9.2 During Implementation

- [ ] Daily standups (15min each phase) to track progress
- [ ] Slack updates on blockers, discoveries, integration points
- [ ] Code review on PRs (before merging to develop)

### 9.3 Post-Implementation

- [ ] Release notes documenting new features, parameters, recipes
- [ ] User guide / tutorial on freckle_preserve_mask, regional_modulation, smooth_engine
- [ ] Internal documentation: architecture decisions, trade-offs, future improvements

---

## 10. Risk Management

### 10.1 Critical Risks

| Risk | Phase | Mitigation |
|------|-------|-----------|
| Backward compatibility break | A/B | Default params = no-op (byte-identical); golden snapshot tests |
| Performance budget exceeded (esp. anisotropic 6K) | B | Downsample to 2048px; profile early; optimize if needed |
| Freckle classifier fails on dark skin | C | Normalize LAB a/b to skin tone; test on 3 dark photos; high confidence threshold |
| Integration conflict (params, signature, stages) | All | Coordination points documented; dependency map clear; staged rollout |

### 10.2 Contingency Plans

- **Phase A delayed:** Phase B/C can start independently (no strict dependency).
- **Phase B performance issue:** Fallback to guided filter (documented in Spec #2).
- **Phase C classifier accuracy low:** Increase confidence threshold; reduce freckle_removal range.
- **Post-release regression:** Hotfix PR with feature flag to disable problematic feature (e.g., regional_modulation=0).

---

## 11. Success Metrics

### 11.1 Quality Metrics

- ✅ All unit tests passing (50+total)
- ✅ Golden snapshots match baseline (backward compat)
- ✅ Visual QA approved on 25+ diverse photos
- ✅ Multi-skin-tone robustness validated
- ✅ No performance regression > 5%

### 11.2 User Satisfaction

- ✅ Photographers report natural-looking skin texture (vs. plastic before)
- ✅ Cosplay characters retain beauty marks as intended
- ✅ Under-eye freshness improved without hollowing
- ✅ Wrinkle/pore structure preserved with anisotropic smoothing

### 11.3 Technical Metrics

- ✅ Code review approved (architecture, readability, test coverage)
- ✅ Documentation complete (parameter reference, recipe examples, troubleshooting)
- ✅ CI/CD pipeline green (all tests, linting, type checking)
- ✅ Deployment smooth (no rollback needed)

---

## 12. Appendix: Feature Summary Table

| Feature | Spec File | Phase | Hours | Complexity | ROI | Dependencies |
|---------|-----------|-------|-------|-----------|-----|---|
| Region-aware smoothing | REGION_AWARE_SMOOTHING.md | A | 6 | Low | High | None |
| Eye shadow smoothing | EYE_SHADOW_SMOOTHING.md | A | 4 | Low | Medium | None |
| Anisotropic diffusion | ANISOTROPIC_DIFFUSION.md | B | 10 | Med | High | Phase A (coordinate) |
| Freckle/beauty mark control | FRECKLE_BEAUTY_MARK_CONTROL.md | C | 20 | Med | High | None |

**Total:** 40 hours, distributed over 2 weeks, phased delivery.

---

## 13. Next Steps

1. **Architect Review (This Document)**
   - Verify phasing strategy and coordination points
   - Confirm resource allocation (agents, timeline)
   - Approve risk mitigations

2. **Assign Implementation Agents**
   - Phase A: Opus or Haiku (parallel on 2 features)
   - Phase B: Opus or Haiku (sequential after Phase A)
   - Phase C: Opus or Haiku (sequential after Phase B)
   - Integration: Same team or rotation

3. **Kick-off Meeting**
   - Review all 4 specs with implementation team
   - Clarify ambiguities
   - Establish communication cadence (daily standups, PR reviews)

4. **Start Phase A**
   - Agents begin implementing Region-aware smoothing and Eye shadow smoothing in parallel
   - Target completion: 2 days (Mon–Tue)

---

## 14. Conclusion

This roadmap provides a structured, phased approach to implementing four complementary beauty improvement features. The modular design allows parallel work early (Phase A) and independent testing throughout. Clear coordination points, performance budgets, and risk mitigations minimize integration risk.

**Expected Outcome:** By end of Week 2, all four features deployed, tested, and approved for production release. Photographers gain powerful new tools for preserving intentional beauty while enhancing polished appearance.

---

## As-Built Corrections (integration points)

The spec's integration guidance to edit `engine._stage_per_face()` and call
`separator.combine(...)` there is **incorrect** — `combine()` is not invoked from
`engine.py`. The real skin-smoothing pipeline runs in `retouch/perf_optimizations.py`
(the per-face ROI worker). Correct wiring, as built:

- **Region-aware + Anisotropic** (both in `frequency.combine`): all four
  `frequency.combine(...)` call sites now pass
  `regions=regions`, `regional_modulation=getattr(ctx,'regional_modulation',0.0)`,
  `smooth_engine=getattr(ctx,'smooth_engine','guided')`.
- **Eye-shadow**: `skin.smooth_undereye_shadow(...)` called after the combine block,
  gated by `ctx.undereye_shadow_strength > 0`.
- **Freckle/beauty-mark**: `FreckleRemover().remove(...)` called after smoothing,
  gated by `ctx.freckle_removal > 0`.

All new parameters default to a **no-op** (`regional_modulation=0.0`,
`undereye_shadow_strength=0.0`, `freckle_removal=0.0`, `smooth_engine="guided"`),
so existing recipes remain byte-identical.

