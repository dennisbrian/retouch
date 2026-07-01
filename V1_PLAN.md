# V1 Plan — Fuji-Quality Color Recipe System

**Date locked:** 2026-06-23
**Duration:** 10-12 weeks (2.5-3 months)
**Goal:** Best color recipe system in any retouching app. 90-95% match to Fujifilm JPEG.
**Status:** PLANNING ONLY — no execution yet

---

## What V1 Is (and Isn't)

### ✅ IN V1
- Fuji-quality color foundation
- 3 official Fuji film simulations: **Classic Chrome**, **Astia**, **Provia**
- 3D LUT pipeline (real `.cube` support, not fake)
- Recipe system: load / save / apply / share / builder UI
- Power user color tools: HSL, curves, channel mixer, WB, blend modes

### ❌ OUT OF V1 (deferred to v2+)
- Makeup engine (lipstick, eyeshadow, etc.)
- Body reshaping, expression editing
- Smart Default, adjustment brush, undo/redo
- Plugin API, recipe cookbook
- Mobile, cloud, AI agents, video

---

## Pre-Phase 0 — Foundation (2 weeks)

- [x] Fix 2 pre-existing bloom-scaling bugs in `build_context` (already resolved by params.py)
- [x] Resolve 33 default mismatches — color-relevant ones first (smooth, mid_reduction aligned)
- [x] Add missing `color_transfer_intensity` ParamSpec
- [x] Measure baseline: `pytest --cov` + benchmarks (1550 passed, 89% coverage)
- [x] Set up CI with `pytest` + benchmark tracking (`.github/workflows/ci.yml`)
- [x] LCH color space utility spike (6 new primitives + 30 tests)
- [x] 3D LUT pipeline spike (.3dl loader, apply_blended, 2 generators + 19 tests)
- [ ] Mask system v0 (1w) — unified `apply_to_region(img, mask, op)` API (exists in regions.py)
- [ ] Fuji color research (1w) — what makes Fuji look like Fuji

**Outcome:** Foundation rails solid. Decisions validated. Ready for Phase 1.

---

## Phase 1.a — Fuji Foundation Layer (3 weeks)

- [ ] **Tonal response curve** — film H&D model, not digital sCurve. Single biggest "Fuji-look" unlock.
- [ ] **Skin-tone protection** — preserve skin hues during all color operations. Required for Astia-quality portraits.
- [ ] **Film grain synthesis** — organic, clumped, luminance-correlated. NOT Gaussian noise.
- [ ] **Highlight rolloff** — soft film-like clip, not digital hard clip.

**Outcome:** Foundation primitives in place. Color ops have film-like response.

---

## Phase 1.b — 3D LUT Pipeline (2 weeks)

- [ ] Real `.cube` / `.3dl` loader with trilinear interpolation
- [ ] LUT directory registration (`luts/` like `presets/`)
- [ ] 10+ film stocks as data (Kodak Portra 400, Fuji Pro 400H, Cinestill 800T, etc.)
- [ ] Hot-load support
- [ ] ICC profile input — read embedded ICC, convert to working space
- [ ] ICC profile embedding on export
- [ ] Wide-gamut working space — ProPhoto RGB internally
- [ ] 16-bit float internal pipeline

**Outcome:** Real 3D LUTs working. Wide-gamut color fidelity preserved.

---

## Phase 1.c — 3 Official Fuji Film Simulations (3 weeks)

- [ ] **Classic Chrome** — flagship, the "Fuji look" most people mean
- [ ] **Astia** — portrait specialist, beautiful skin tones
- [ ] **Provia** — neutral/standard, accurate reproduction
- Each sim is a carefully crafted recipe, not a hardcoded LUT.
- Combines tonal curve + skin protection + LUT + grain.

**Outcome:** 3 official Fuji sims that look 90-95% like real Fuji JPEGs.

---

## Phase 1.d — Power User Color Tools (2 weeks)

- [x] LCH-based HSL panel (replace HSV math — "luminance" actually controls L*) — primitives done
- [x] Channel mixer for B&W (per-channel R/G/B weights) — engine done, GUI pending
- [x] White balance GUI (Kelvin + tint sliders) — engine done, GUI pending
- [x] Soft-light blend mode for grade layer — public blend modes + ColorGrader method
- [x] Master HSL controls (Lightroom "All →" sliders) — params + pipeline done
- [x] Negative split toning (de-saturation in shadows/highlights) — params + pipeline done

**Outcome:** Power users can craft any color grade. Pro-tool parity for color.

---

## Phase 1.e — Recipe System v1 (1-2 weeks, parallel to 1.d)

- [ ] Recipe builder UI — power users craft custom recipes
- [ ] Recipe version diff — compare two recipes side by side
- [ ] Recipe export/import workflow + format docs
- [ ] Recipe validation + schema enforcement

**Outcome:** Recipes are first-class shareable artifacts. Format is documented and stable.

---

## Success Criteria

### Must-have
- ✅ 3 official Fuji sims (Classic Chrome, Astia, Provia) that look 90-95% like real Fuji JPEGs
- ✅ Recipe save/load as JSON
- ✅ Recipe share (export/import)
- ✅ 10+ film stocks in 3D LUT library
- ✅ Skin tones survive grading
- ✅ Film grain (organic, not digital)
- ✅ Wide-gamut color fidelity (ICC input/output)
- ✅ Recipe builder UI
- ✅ All 213 current tests pass + new tests for the 3 sims

### Nice-to-have
- ⭐ 95%+ subjective match to a Fuji JPEG in casual viewing
- ⭐ Community-contributed recipes work out of the box
- ⭐ AI recipe generator (from prompt) integrates with new color science

---

## Dependencies (in execution order)

```
Pre-Phase 0 (2w)
    ├── Foundation work
    └── Research spikes

Phase 1.a (3w) ───── Foundation primitives
    │
Phase 1.b (2w) ───── 3D LUT pipeline
    │
Phase 1.c (3w) ───── 3 Fuji sims
    │
Phase 1.d (2w) ───── Power user tools  ║
    │                                  ║ parallel
Phase 1.e (1-2w) ── Recipe system v1  ║

Total: ~10-12 weeks
```

---

## Risks + Mitigations

| Risk | Mitigation |
|---|---|
| LUT strategy delayed | Default to commercial ($200) in 1-2d. Re-evaluate after 1 month. |
| 3D LUT perf on 4K | Benchmark in week 1. Fall back to 1D if needed. |
| Mask quality bottleneck | BiSeNet spike in Pre-Phase 0. Try 2-3 models. |
| Film grain too "fake" | Reference real Portra 400 grain samples. Test against Fuji JPEG output. |
| Fuji quality takes longer | 3-month buffer in Phase 1. v1 can slip to 14 weeks without deferring. |
| 4.6 mismatches block color work | Resolve color-relevant ones in Pre-Phase 0. Defer non-color ones. |

---

## What "Recipe" Means in v1

A recipe is a JSON file describing:
- A color grade preset (e.g. "Classic Chrome")
- Parameter values for all 59 processing parameters
- Optional references to LUT files in `luts/`
- Metadata: name, author, version, target photo type

Recipes are:
- **Loadable**: from JSON file or built-in preset
- **Saveable**: as JSON
- **Applicable**: to any photo via `engine.process(img, recipe=...)`
- **Shareable**: export/import via file or URL
- **Diffable**: compare two recipes side by side
- **Validatable**: schema-enforced for safety

---

## Tracking

Update this file as work progresses. Check off completed items.

When all items checked, **v1 is DONE**. Move to v2 (face editing).
