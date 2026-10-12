# Session Progress Report — 2026-07-02

**Date:** 2026-07-02
**Session:** Photoshop-Parity Roadmap Planning + Tier 1/2 Implementation (F1, F4, F6)
**Branch:** main
**Tests:** Deferred to next session (per user instruction)

---

## TL;DR

Planned the full Photoshop-parity roadmap (4 plan docs), then implemented 3 stages in parallel: F1 (float32 pipeline + 16-bit export), F4 (spot heal Telea v0), and F6 (look-from-reference preset extraction). All three stages are code-complete; test verification deferred to next session.

---

## Planning Phase

Created 4 plan documents in the project root:

| Document | Content |
|----------|---------|
| `PLAN_MOONLIGHT_PORCELAIN.md` | Blue cinematic look (✅ already implemented, commit 2b46f96) |
| `PHOTOSHOP_PARITY_ROADMAP.md` | Gap analysis vs Photoshop with Tiers 1–3 |
| `PLAN_TIER1_FOUNDATION.md` | Stages F1–F3: float32 pipeline + 16-bit export, session/undo, brush local adjustments |
| `PLAN_TIER2_MANUAL_TOOLS.md` | Stages F4–F7: spot heal, face-aware liquify, look-from-reference, AI denoise/super-res |

**Key finding:** Every Tier 2 stage extends already-tested code — BlemishRemover's inpaint+blend becomes spot heal, FaceReshaper's warp engine becomes liquify, StyleAnalyzer becomes look-from-reference, and new ONNX models ride the existing `build_ort_providers()` CoreML-accelerated runtime.

---

## Implementation Phase

### Stage F1 — Float32 Global Pipeline + 16-bit Export ✅

**Goal:** Eliminate inter-stage uint8 truncation where banding originates; export 16-bit.

**What was done:**

| Component | Change | Files |
|-----------|--------|-------|
| Float helper variants | Added `_F_adjust_contrast`, `_F_adjust_tonal`, `_F_apply_uniform_saturation`, `_F_apply_selective_sharpening` | `retouch/engine.py` |
| `_stage_subject_separation` | Ported to accept/return float32 [0,1] or uint8 | `retouch/engine.py` |
| `_stage_global` | Ported to accept/return float32 [0,1] or uint8 (brightness now uses `np.power` directly, no LUT) | `retouch/engine.py` |
| `_stage_grade` | Ported to accept/return float32 [0,1] or uint8; uses `grade(return_float=True)` | `retouch/engine.py` |
| `_stage_finish` | Ported to accept/return float32 [0,1] or uint8 | `retouch/engine.py` |
| `_run_core_pipeline` | Converts to float after `_composite_faces`, converts back to uint8 at end | `retouch/engine.py` |
| `grade()` | Added `return_float` parameter; tracks dtype through post-effects | `retouch/grading.py` |
| 16-bit export | `write_image_with_icc(bit_depth=16)` for PNG/TIFF | `retouch/io.py` |
| CLI `--bit-depth` | Added `--bit-depth 8\|16` argument; uses `write_image_with_icc` | `cli.py` |

**Design decision:** Migrated Phase 3 (global stages) only to float32. Phase 2 per-face stays uint8 (blends through masks, low banding risk, huge migration surface). Gets ~90% of the quality win for ~30% of the risk.

**cv2 range trap handling:** All ported conversions use explicit `bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)` before `cv2.cvtColor` to avoid the float LAB/HSV range differences.

### Stage F4 — Spot Heal / Object Removal (Telea v0) ✅

**Goal:** Remove stray hairs, dust, wig lace, background clutter without Photoshop.

**What was done (by subagent):**

| Component | Change | Files |
|-----------|--------|-------|
| Shared blend helper | Extracted `inpaint_and_blend()` from `BlemishRemover.remove()` | `retouch/blemish.py` |
| Heal module | NEW: `heal_region(img, mask, method="telea"\|"ns", radius=auto)` with auto-radius scaling | `retouch/heal.py` (new) |
| Pre-pipeline hook | Heals run BEFORE retouch/grade in `process()`; added `heals` field to `ProcessingContext` | `retouch/engine.py` |
| GUI Heal tab | `gr.ImageEditor` paint surface, method dropdown, heal button, heal history table, clear button | `gui.py` |
| Tests | 15 tests: flat/gradient/textured heal, NS method, float masks, auto-radius, base64 round-trip, heal-before-pipeline | `tests/test_heal.py` (new) |

**Test results (subagent):** 15/15 heal tests pass, 14/14 blemish tests pass, 165/165 engine/params tests pass.

### Stage F6 — Look-from-Reference ✅

**Goal:** Drop in a finished photo → get an editable grading preset JSON (curve + split-tone + WB + HSL).

**What was done (by subagent):**

| Component | Change | Files |
|-----------|--------|-------|
| `extract_look()` | Added to `StyleAnalyzer`: paired + unpaired modes, extracts L curve / WB / split-tone / HSL | `retouch/style.py` |
| Preset output | Writes standard preset JSON to `presets/` (auto-discovered, immediately usable) | `retouch/style.py` |
| GUI button | "Extract Editable Preset" button + name field in Color Transfer accordion | `gui.py` |
| Tests | 9 tests: unpaired/paired modes, curve monotonicity, WB values, split-tone structure, cool-shadow sanity, HSL structure, disk save, roundtrip | `tests/test_style.py` |

**Test results (subagent):** 28/28 style tests pass (including 9 new `TestExtractLook` tests), 142/142 total in touched areas.

**Key design decisions:**
- L curve: 7 percentile anchor points (0,5,25,50,75,95,100) with monotonicity enforcement
- WB: gray-world with max-normalized gains
- Split tone: LAB a/b → LCH hue/chroma in 3 luminance bands (L<85, 85-170, L>170)
- HSL: 8 hue buckets matching `_apply_hsl_adjustments` centers, saturation delta >2.0 threshold

---

## F2 Session Model — Cancelled

The F2 session model subagent was cancelled (user had to leave). This stage is not started.

---

## Files Changed Summary

| File | Status | Stages |
|------|--------|--------|
| `retouch/engine.py` | Modified | F1, F4 |
| `retouch/grading.py` | Modified | F1 |
| `retouch/io.py` | Modified | F1 |
| `cli.py` | Modified | F1 |
| `retouch/blemish.py` | Modified | F4 |
| `retouch/heal.py` | **New** | F4 |
| `retouch/style.py` | Modified | F6 |
| `gui.py` | Modified | F1, F4, F6 |
| `tests/test_heal.py` | **New** | F4 |
| `tests/test_style.py` | Modified | F6 |

---

## What's NOT Done (Deferred to Next Session)

1. **Test verification:** Full test suite run deferred per user instruction. Subagent tests passed (F4: 15/15, F6: 9/9), but full regression suite not yet run.
2. **F1 GUI export dropdown:** The `--bit-depth` CLI flag is done, but the GUI export-format dropdown still needs "TIFF 16-bit" / "PNG 16-bit" options added.
3. **F2 Session model:** Cancelled (subagent interrupted). Not started.
4. **Visual QA:** Per AGENTS.md, changes touching `grading.py` and `engine.py` are Visual-Critical. Visual QA gates not yet run.

---

## Next Session Priorities

1. **Run full test suite** — verify no regressions from F1/F4/F6 changes
2. **Visual QA** — per `docs/VISUAL_QA.md` gates (F1 touches grading.py + engine.py = Visual-Critical)
3. **Complete F1** — add GUI 16-bit export dropdown options
4. **F2 Session model** — re-attempt (session.py, undo/redo, save/load)
5. **F5 Liquify** — extend FaceReshaper with new warp sets
6. **F3 Brush masks** — ImageEditor integration for local adjustments

---

## Roadmap Status

```
Tier 1 (Foundation):
  F1 Float32 pipeline + 16-bit export    ✅ Code-complete (tests pending)
  F2 Session/undo/snapshots              ❌ Not started
  F3 Brush local adjustments             ❌ Not started

Tier 2 (Manual Tools):
  F4 Spot heal (Telea v0)                ✅ Code-complete (tests pass)
  F5 Face-aware liquify sliders          ❌ Not started
  F6 Look-from-reference                 ✅ Code-complete (tests pass)
  F7 AI denoise/super-res                ❌ Not started
  F4.b LaMa ONNX                         ❌ Not started

Tier 3 (Differentiators):
  All deferred
```
