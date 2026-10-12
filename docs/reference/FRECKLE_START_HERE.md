# Freckle/Beauty Mark Control Feature - START HERE

**Status:** ✅ **READY TO IMPLEMENT IMMEDIATELY**

This directory contains a complete, production-ready implementation plan for the freckle/beauty mark control feature. Read these documents in order:

---

## 📋 Documentation Index

### 1. **Executive Summary** (5 min read)
**File:** `FRECKLE_EXECUTIVE_SUMMARY.txt`

Quick overview of the feature, problem statement, and key decisions. Start here if you need to understand the business case or brief someone else.

**Contains:**
- Problem & solution
- Time estimate (2.5 days)
- Risk assessment
- Competitive analysis
- Next steps

---

### 2. **Implementation Summary** (10 min read)
**File:** `FRECKLE_IMPLEMENTATION_SUMMARY.md`

High-level technical design with architecture diagrams, classification heuristics, and acceptance criteria.

**Contains:**
- Quick reference tables
- Pipeline position diagram
- Classification logic (freckle vs. beauty mark vs. blemish)
- Competitive analysis
- FAQ

---

### 3. **Complete Technical Plan** (30 min read)
**File:** `PLAN_FRECKLE_BEAUTY_MARK.md`

12-section detailed plan covering architecture, design, implementation, testing, and deployment.

**Contains:**
- Current architecture analysis
- Design specification (scope, detection algorithm, strategy)
- Exact class/method signatures
- Testing strategy (unit, integration, visual QA)
- Time breakdown (hourly)
- Risk assessment & mitigation
- Acceptance criteria
- Deployment roadmap

**Recommended for:** Technical leads, architects, code reviewers

---

### 4. **Code Skeleton & Implementation Guide** (60 min work)
**File:** `FRECKLE_CODE_SKELETON.md`

Copy-paste-ready code stubs for all new and modified files, with `# TODO` sections marked for implementation.

**Contains:**
- Complete `retouch/freckle.py` skeleton
- Engine integration changes (exact line numbers)
- Test file templates (18+ test cases)
- Implementation checklist
- Quick start guide

**Recommended for:** Developers implementing the feature

---

## 🚀 Quick Start (for Developers)

1. **Read:** `FRECKLE_IMPLEMENTATION_SUMMARY.md` (classification logic)
2. **Review:** `PLAN_FRECKLE_BEAUTY_MARK.md` sections 3–4 (architecture, design)
3. **Code:** Use `FRECKLE_CODE_SKELETON.md` as your starting point
4. **Follow:** Implementation checklist at end of skeleton

---

## 🎯 Key Numbers

| Metric | Value |
|--------|-------|
| **Effort** | 21.5 hours (2.5 days) |
| **Files to create** | 3 (freckle.py, 2 test files) |
| **Files to modify** | 2 (engine.py, perf_optimizations.py) |
| **Unit tests** | 18–25 test cases |
| **Visual QA** | 5 cosplay test images |
| **Risk level** | LOW |
| **Code reuse** | 85% (from BlemishRemover) |

---

## 📌 Core Design Decision

```
FRECKLES ≠ BLEMISHES ≠ BEAUTY MARKS

Input Image
    ↓
[Detect all anomalies]
    ↓
┌───────┴───────┬───────────────┬──────────┐
│               │               │          │
FRECKLE      BLEMISH    BEAUTY_MARK      NOISE
Small         Medium         Large        ~
Red-tinted    Inflamed       Dark         ~
Clustered     Scattered      Isolated     ~
→ Remove      → Ignore       → Preserve   → Discard
```

---

## 🔗 File Locations

```
/Applications/htdocs/retouch/
├── FRECKLE_START_HERE.md               ← You are here
├── FRECKLE_EXECUTIVE_SUMMARY.txt       ← Business case
├── FRECKLE_IMPLEMENTATION_SUMMARY.md   ← Technical overview
├── PLAN_FRECKLE_BEAUTY_MARK.md         ← Full plan
├── FRECKLE_CODE_SKELETON.md            ← Implementation guide
│
├── retouch/
│   ├── freckle.py                      ← NEW (create)
│   ├── engine.py                       ← MODIFY (add params)
│   └── perf_optimizations.py           ← MODIFY (add hook)
│
└── tests/
    ├── test_freckle.py                 ← NEW (unit tests)
    └── test_freckle_integration.py     ← NEW (integration tests)
```

---

## ✅ Acceptance Criteria

Feature is **DONE** when:

- [ ] Code: FreckleRemover class + engine integration complete
- [ ] Tests: 18+ unit tests passing; no regression in blemish tests
- [ ] Visual QA: 4/5 cosplay images approved (freckles removed, beauty marks preserved)
- [ ] Docs: PLAN file complete, API.md updated, docstrings added
- [ ] Code quality: Type hints, linting, code review passed

---

## 🛠️ For Different Roles

### **Product Manager / Decision Maker**
1. Read: `FRECKLE_EXECUTIVE_SUMMARY.txt` (8 min)
2. Decision: Approve implementation?
3. Action: Assign engineer + set timeline

### **Architect / Tech Lead**
1. Read: `FRECKLE_IMPLEMENTATION_SUMMARY.md` (10 min)
2. Review: `PLAN_FRECKLE_BEAUTY_MARK.md` sections 1–6 (30 min)
3. Decision: Architecture sound? Any concerns?
4. Action: Approve design, brief engineer

### **Developer / Implementer**
1. Read: `FRECKLE_IMPLEMENTATION_SUMMARY.md` section "Classification Heuristics" (5 min)
2. Read: `PLAN_FRECKLE_BEAUTY_MARK.md` sections 3–4 (15 min)
3. Use: `FRECKLE_CODE_SKELETON.md` to start coding (60 min+)
4. Test: Run unit tests as you code
5. Iterate: Visual QA feedback → threshold adjustments

### **QA / Tester**
1. Read: `PLAN_FRECKLE_BEAUTY_MARK.md` section 5 (testing strategy)
2. Prepare: 5 cosplay test images (varied skin tones, freckle patterns, beauty marks)
3. Test: Run test_freckle.py and test_freckle_integration.py
4. Visual QA: Compare before/after on cosplay photos
5. Feedback: Report threshold tuning needs

---

## 📊 Timeline

**Total: 2.5 days (21.5 hours)**

| Day | Phase | Hours |
|-----|-------|-------|
| 1   | Design + Core Implementation | 8 |
| 1.5 | Engine Integration + Float32 | 3 |
| 2   | Testing (unit + integration) | 4 |
| 2.5 | Visual QA + Polish | 3.5 |
| 3   | Final review, bug fixes | 3 |

---

## 🎓 Learning Resources (If Needed)

If you're unfamiliar with the codebase:

1. **Read existing code:**
   - `retouch/blemish.py` — Template for FreckleRemover
   - `tests/test_blemish.py` — Template for test structure

2. **Understand architecture:**
   - `retouch/engine.py` lines 100–110 (imports)
   - `retouch/engine.py` lines 180 (ProcessingContext)
   - `retouch/perf_optimizations.py` lines 240–270 (pipeline)

3. **Color space (LAB):**
   - `retouch/color_space.py` — Color conversion utilities
   - Key: LAB separates luminance (L) from color (a, b)
   - Freckles are red-tinted → positive a offset
   - Beauty marks are dark → negative L offset

---

## ❓ FAQ

**Q: Do I need to read all documents?**  
A: No. Start with your role's summary above. Deep-dive on demand.

**Q: Where's the code to implement?**  
A: `FRECKLE_CODE_SKELETON.md` — 90% of the scaffolding is there.

**Q: How much is copied from BlemishRemover?**  
A: ~85%. The detection and inpainting logic are reused. New code is the classification layer (~200 lines).

**Q: What if color heuristics don't work on my test images?**  
A: See `PLAN_FRECKLE_BEAUTY_MARK.md` section 7 (risk mitigation). Tests will surface this; adjust thresholds iteratively.

**Q: Can users customize the preserve behavior?**  
A: Yes. Advanced API: `freckle_preserve_mask` parameter (base64 PNG).

---

## 🚨 Critical Success Factors

1. ✅ **Use LAB color offsets (relative)**, not absolute thresholds
   - Ensures generalization across skin tones
   
2. ✅ **Conservative beauty mark detection** (size ≥15px²)
   - Avoid false positives (removing intended marks)
   
3. ✅ **Visual QA on 5 diverse cosplay photos**
   - Validate before shipping

4. ✅ **Separate parameters from blemish removal**
   - Users tune each independently

---

## 📞 Questions or Blockers?

- **Technical:** Refer to `PLAN_FRECKLE_BEAUTY_MARK.md` (it's comprehensive)
- **Architecture:** Ping architect; design review recommended
- **Threshold tuning:** Visual QA + iteration (5 cosplay photos)
- **Code patterns:** See `retouch/blemish.py` and existing tests

---

## 📝 Next Action

1. **Stakeholder Decision:**
   - [ ] Read `FRECKLE_EXECUTIVE_SUMMARY.txt`
   - [ ] Approve for Phase 8 roadmap?
   - [ ] Assign engineer?

2. **Engineer Kickoff:**
   - [ ] Read `FRECKLE_IMPLEMENTATION_SUMMARY.md`
   - [ ] Review design with tech lead (1h optional)
   - [ ] Start coding using `FRECKLE_CODE_SKELETON.md`

3. **QA Prep:**
   - [ ] Gather 5 cosplay test images
   - [ ] Set up test environment
   - [ ] Prepare visual comparison tools

---

**Status:** ✅ Ready to implement  
**Confidence:** High  
**Date:** 2026-07-08

---

**Last Updated:** 2026-07-08  
**Maintainer:** Claude Code (Planning Agent)
