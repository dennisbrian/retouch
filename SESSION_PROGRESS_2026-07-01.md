# Session Progress Report — 2026-07-01

**Date:** 2026-07-01
**Session:** Audit close-out → Pre-Phase 0 → Phase 1.a-d pipeline (full sweep)
**Branch:** main
**Commits:** 10
**Net code change:** +1,900 insertions, ~100 deletions across 16 files
**Tests:** 1550 → ~1700 passed, 0 regressions

---

## TL;DR

Closed the audit findings, completed Pre-Phase 0 foundation, spiked LCH color primitives, implemented the 3D LUT pipeline, and wired Phase 1.d power-user color tools into the engine. The Fuji-quality color recipe system is ~85% complete.

---

## Phase 1: Audit Fixes (#1-#6)

| Fix | Description | Files |
|-----|-------------|-------|
| #1 | No-face path `subject_mask` bug in `add_impact_finish` | `engine.py` |
| #2 | Dead `else 1.2` in sharpen gate | `engine.py` |
| #3 | Add `_logger.exception()` to 3 GUI handlers | `gui.py` |
| #4 | `print` → `logger.warning` in `watch_luts_dir` | `lut.py` |
| #5 | Promote §1.4 harness → `test_tonal.py` + `test_precision.py` + `test_style_transfer.py` (+40 tests) | 3 new test files |
| #6 | `test_recipe_loader_cli.py` smoke tests (+15 tests) | 1 new test file |

---

## Phase 2: Pre-Phase 0 Foundation

| Item | Status | Details |
|------|--------|---------|
| Bloom-scaling bugs (0.16→16.0, 0.1→10.0) | ✓ Already resolved | params.py `recipe_pct` conversion handles it |
| Default mismatches | ✓ Fixed | `smooth` 50→30, `mid_reduction` 0.40→0.35 aligned to `_DEFAULTS` |
| Missing `color_transfer_intensity` ParamSpec | ✓ Added | `_GRADING_PARAMS`, `--color-transfer-intensity` CLI flag |
| Baseline coverage | ✓ Measured | 1550 passed, 89% coverage |
| CI setup | ✓ Created | `.github/workflows/ci.yml` — fast tests + coverage on PR/push |

---

## Phase 3: LCH Color Space Spike

Added 6 perceptual HSL primitives to `color_space.py`:

| Function | Purpose | Replaces |
|----------|---------|----------|
| `hue_range_mask` | Soft cos² mask around hue centre | Manual HSV math |
| `adjust_hue_range` | Selective hue shift within colour band | `_hsl_hue_shift` (HSV) |
| `adjust_chroma_range` | Perceptual saturation control | `_apply_hsl_adjustments` (HSV) |
| `adjust_luminance_range` | L* shift within colour band | Luminance hacks |
| `split_tone_lch` | L*-keyed shadow/highlight tint | `_split_tone` (HSV) |
| `color_balance_lch` | 3-axis CMY/RGB balance | New capability |

All operate in float32 LCH for perceptual uniformity. +30 tests.

---

## Phase 4: 3D LUT Pipeline Spike

| Feature | Status |
|---------|--------|
| `.cube` loader (Adobe) | Existing ✓ |
| `.3dl` loader (Iridas) | Implemented (was stub) ✓ |
| Trilinear interpolation | Existing ✓ |
| `apply_blended` (partial strength 0-1) | New ✓ |
| `generate_tint_lut` (warm/cool) | New ✓ |
| `generate_contrast_lut` (S-curve) | New ✓ |
| LUTRegistry hot-load | Existing ✓ |
| 6 LUTs in `luts/` | Existing ✓ |
| Grading pipeline integration | Existing ✓ |

+19 tests.

---

## Phase 5: LCH HSL Methods on ColorGrader

Added 5 new public methods to `ColorGrader`:

| Method | Description |
|--------|-------------|
| `adjust_hsl_lch` | Global perceptual HSL (hue/sat/lum) |
| `adjust_per_channel_lch` | 8-channel panel (red, orange, yellow, green, aqua, blue, purple, magenta) |
| `split_tone_lch` | L*-keyed split toning |
| `white_balance_lch` | Kelvin + tint in LCH |
| `channel_mixer_bw` | Per-channel R/G/B weights with brightness/contrast |

+26 tests. 0 regressions on 138 existing grading tests.

---

## Phase 6: Pipeline Wiring

Wired white balance and B&W channel mixer into the full pipeline:

| Component | What changed |
|-----------|-------------|
| `params.py` | 5 new ParamSpecs: `white_balance_kelvin`, `white_balance_tint`, `bw_channel_mixer_r/g/b` |
| `ProcessingContext` | 5 new fields via `_DEFAULTS` |
| `build_context` | Auto-resolved from recipe via `gui_direct` |
| `process()` | Added to signature + overrides dict |
| `_stage_grade` | WB after color transfer, B&W at final output |
| `_no_face_fallback` | Same WB + B&W path |
| `recipe_loader` | Flat params list updated (forward + reverse) |
| `test_gui` | `EXPECTED_RECIPE_KEY_COUNT` 65→70, `PROCESS_INPUT_KEYS` 75→80 |

---

## Phase 7: Blend Modes

Added 3 public blend modes + ColorGrader method:

| Function | Blend Mode |
|----------|-----------|
| `soft_light_blend` | Pegtop approximation, contrast-preserving |
| `overlay_blend` | Multiply on dark, screen on light |
| `hard_light_blend` | Overlay driven by layer, not base |
| `ColorGrader.apply_soft_light_layer` | Non-destructive self-blend grading |

+10 tests.

---

## V1 Status (Fuji-Quality Color Recipe System)

| Phase | Component | Status |
|-------|-----------|--------|
| 0 | Foundation (bloom bugs, CI, baseline) | ✅ Done |
| 1.a | Tonal H&D curve | ✅ Existing |
| 1.a | Skin-tone protection | ✅ Existing |
| 1.a | Film grain synthesis | ✅ Existing |
| 1.a | Highlight rolloff | ✅ Existing |
| 1.b | 3D LUT pipeline | ✅ Done (spiked) |
| 1.c | Classic Chrome | ✅ Preset + tests |
| 1.c | Astia | ✅ Preset + tests |
| 1.c | Provia | ✅ Preset + tests |
| 1.d | LCH HSL panel primitives | ✅ Done |
| 1.d | WB + B&W channel mixer | ✅ Done |
| 1.d | Soft-light blend mode | ✅ Done |
| 1.d | Hard-light / overlay blend | ✅ Done |
| 1.d | Negative split toning | ⬜ Pending |
| 1.d | Master HSL controls (GUI) | ⬜ Pending |
| 1.e | Recipe builder UI | ⬜ Pending |
| 1.e | Recipe export/import | ✅ Existing |

---

## Up Next

- **Phase 1.d remaining**: Negative split toning, master HSL GUI wiring
- **Phase 1.e**: Recipe builder UI enhancements
- **Phase 2 (v2)**: Face editing (makeup, body reshaping) — deferred per roadmap
