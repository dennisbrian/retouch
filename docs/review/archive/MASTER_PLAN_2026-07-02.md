# MASTER PLAN — Priority Roadmap (2026-07-02)

**Current Status:** v1 (Fuji-quality color) ~95% complete. Shipping soon.  
**Next Focus:** v1.5 (polish + stabilization), then v2 (face editing).

---

## 🚨 CRITICAL PATH (Do Now — Next 1 week)

### 1. Moonlight Porcelain Testing & Merge
- **Status:** Ready for Visual QA
- **Action:** Test on 2-3 sample portraits (dark BG, white costume, warm props)
- **Artifact checklist:** See PLAN_MOONLIGHT_PORCELAIN.md §Step 4
- **Blocker:** None (zero engine changes, auto-discovered)
- **Timeline:** 3-5 hours visual testing + merge

### 2. Merge WIP skin.py Fixes
- **Status:** Fixes latent `_build_dimensional_mask` bug (returns 200×200 zero mask)
- **Action:** Run Visual QA, verify no regressions
- **Blocker:** Visual testing not yet done
- **Timeline:** 2-3 hours testing + merge

### 3. Document Dead anime_crystal_void Keys ✅
- **Status:** DONE (commit `af18eb0`)
- **Action:** None (documented inline, links to plan)

---

## 🟡 HIGH PRIORITY (Next 2 weeks)

### 4. Recipe Validator Test
- **What:** Automated test catching dead recipe keys (like `anime_crystal_void`'s 7 dead keys)
- **Why:** Would prevent shipping broken presets
- **Effort:** 1-2 hours
- **Impact:** HIGH (future-proofs recipe system)
- **Recommendation:** Do this before adding more recipes

### 5. CI Workflow Consolidation ✅
- **Status:** DONE (clarified in commit `0774941`)
- **What:** Marked ci.yml as "fast-feedback", test.yml as "canonical"
- **Action:** None (documented roles)

### 6. LUT Hot-Reload Wiring
- **What:** Integrate `watch_luts_dir()` daemon into GUI
- **Why:** Users want to load new `.cube` files without restart
- **Effort:** 3-4 hours
- **Impact:** MEDIUM (convenience, not critical)
- **Timeline:** Defer to v1.5 if v1 shipping is urgent

---

## 🟠 MEDIUM PRIORITY (Next 4 weeks)

### 7. Add `.3dl` Bounds Check ✅
- **Status:** DONE (commit `6ffa1e3`)
- **What:** Max 256³ LUTs (prevents theoretical DoS)
- **Impact:** LOW (not reachable from GUI/CLI)

### 8. Module Refactoring (Optional)
- **What:** Split large modules (`grading.py`, `engine.py`, `gui.py` are 1.4k-1.8k LOC)
- **Why:** Easier to navigate and test
- **Effort:** 4-8 hours
- **Timeline:** Defer to v2 or next refactor cycle (not blocking)

### 9. Performance Profiling on Real Workflows
- **What:** Benchmark on actual user portrait workflows (batch, group, fashion)
- **Why:** Find actual bottlenecks vs. synthetic benchmarks
- **Effort:** 4-6 hours
- **Timeline:** v1.5

### 10. Community Documentation
- [ ] CONTRIBUTING.md (setup, workflow, testing)
- [ ] TROUBLESHOOTING.md (common issues + fixes)
- [ ] PERFORMANCE_TUNING.md (preview vs. quality trade-offs)
- **Effort:** 2-3 hours total
- **Timeline:** v1.5

---

## 🟢 NICE-TO-HAVE (v1.5+)

### 11. Undo/Redo History
- Status: Gradio doesn't support natively
- Workaround: Save/load recipe snapshots
- Timeline: v3+

### 12. Subject Separation Revival
- Status: Dead params documented in PLAN_MOONLIGHT_PORCELAIN.md (§Optional Stage D)
- What: Wire `background_blur`, `background_desaturation`, `light_wrap` as real params
- Why: Strengthen smoky-background/rim-glow match for anime looks
- Timeline: v1.5+ or v2

### 13. More Fuji Sims
- Status: 3 shipped (Classic Chrome, Astia, Provia)
- What: Add Velvia, Provia-X, Acros (B&W)
- Timeline: v2+

---

## 📊 v2+ ROADMAP (Deferred)

| Phase | Focus | Timeline | Status |
|-------|-------|----------|--------|
| **v2** | Face editing (lipstick, body reshape, expression) | 3-4 months | PLANNED |
| **v3** | Polish + workflow (undo/redo, local adjustments, plugin API v0) | 3-4 months | PLANNED |
| **v4** | Strategic expansion (mobile, plugin ecosystem, AI agents) | 2027+ | ASPIRATIONAL |

---

## Budget & Sustainability

**Current:** $20/month, 3-6 months runway  
**Cost Control:** Haiku + thinking OFF + agents disabled  
**Sustainable:** YES — no runaway costs

---

## Success Criteria (v1 → v1.5 → v2)

### v1 (SHIPPING NOW)
- ✅ 3 official Fuji sims (90-95% match to real Fuji JPEGs)
- ✅ 1,626 passing tests (89% coverage)
- ✅ Zero CRITICAL/MAJOR defects
- ✅ Comprehensive documentation (ARCHITECTURE.md, CLAUDE.md, PROJECT_REVIEW.md)
- ✅ Recipe save/load + import/export
- ✅ Film grain, highlight rolloff, wide-gamut ICC support

### v1.5 (NEXT 4-8 weeks)
- [ ] Moonlight Porcelain + visual QA ✅ ready
- [ ] Recipe validator test
- [ ] LUT hot-reload integrated
- [ ] Module size stabilized (refactor if needed)
- [ ] Community docs (CONTRIBUTING, TROUBLESHOOTING, PERF_TUNING)
- [ ] Performance profiling on real workflows

### v2 (3-4 months after v1 ships)
- [ ] Face editing MVP (lipstick, basic body reshape)
- [ ] Expression editing (smile, eye-open, brow)
- [ ] Per-region face editing
- [ ] New 1,626+ tests for face features

---

## Execution Rhythm

**Weekly:**
- Code review on PRs
- Visual QA on new recipes/presets
- Bug triage + high-priority fixes

**Bi-weekly:**
- Sync on outstanding issues (dead keys, module size, perf)
- Plan v2 architecture (face editing API design)

**Monthly:**
- Full project review (like `docs/review/session/PROJECT_REVIEW_2026-07-02.md`)
- Roadmap update if priorities shift

---

## Decision Matrix (What to Do Next?)

```
Does it block v1 shipping?
├─ YES → Do it now (CRITICAL PATH)
├─ NO → Does it improve stability/maintainability?
│    ├─ YES → Schedule for v1.5 (HIGH/MEDIUM)
│    └─ NO → Defer to v2+ (NICE-TO-HAVE)
```

---

## Accountability

**In-Flight Items:**
- Moonlight Porcelain → User (testing) + Code (visual QA)
- WIP skin.py → User (testing) + Code (merge)
- Recipe validator → Code (implement)

**Owner:** Dennis (user) + Claude (code support)

---

**Last Updated:** 2026-07-02  
**Next Sync:** 2026-07-09 (or after Moonlight testing completes)
