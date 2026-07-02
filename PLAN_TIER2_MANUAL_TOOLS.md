# Tier 2 Implementation Plan — Manual Tools (Stages F4–F7)

## Context

Continuation of the roadmap planning: `PHOTOSHOP_PARITY_ROADMAP.md` (gap analysis) → `PLAN_TIER1_FOUNDATION.md` (F1 float pipeline, F2 sessions/undo, F3 brush masks). This plan turns **Tier 2** — the manual tools that close the "I still need Photoshop for this" loop — into executable stages, based on fresh exploration of what already exists.

**Exploration verdict: Tier 2 is also mostly leverage, not greenfield.** Every stage below extends a mechanism already shipped and tested:

| Existing asset (verified) | Powers |
|---|---|
| `BlemishRemover.remove()` — `cv2.inpaint` Telea + face-width-scaled feathered blend (`retouch/blemish.py:20-60`) | F4 spot heal: same inpaint+blend, user mask instead of auto-detected mask |
| `FaceReshaper.reshape()` — landmark-anchored local translation warps, `(1-d²/R²)²` falloff, accumulated into one `cv2.remap` (`retouch/geometry.py:15-120`) | F5 liquify: new warp definitions per feature, same engine |
| `StyleAnalyzer.extract()` + `StyleProfile` JSON save/load (`retouch/style.py:24-150`) | F6 look-from-reference: extend delta extraction to emit an editable grading preset |
| `build_ort_providers()` — ANE/CoreML→CUDA→DirectML→CPU discovery with fallback (`retouch/perf_optimizations.py:613`), session pattern in `parsing.py:129-138` | F7 denoise/super-res: new models ride the same accelerated runtime |
| Grading preset JSON format (curves/split-tone/hsl, auto-discovered from `presets/`) | F6 output format — reverse-engineered looks become normal presets |

---

## Stage F4 — Spot Heal / Object Removal (click-to-fix)

**Goal:** remove stray hairs, dust, wig lace, background clutter without Photoshop. The #1 round-trip reason.

### Steps
1. **`retouch/heal.py` (new):** `heal_region(img, mask, method="telea"|"ns", radius=auto)` — lift the inpaint + feathered-blend logic from `BlemishRemover.remove` (keep that class unchanged; extract the shared blend into a helper both use). Auto radius scales with mask bounding-box size, not face width (heals can be anywhere).
2. **GUI:** "🩹 Heal" tab using a `gr.ImageEditor` paint surface (same widget family as Tier 1 F3; if F3 isn't built yet, this ships standalone with its own editor instance). Paint → Heal button → result replaces working image; a small heal-history list allows removing an individual heal (re-applies remaining ones — cheap, heals are local).
3. **Heals persist as data:** list of `{mask_png_b64, method}` — stored in the session (F2 field) and applied as a **pre-pipeline step** in `process()` (heal first, then retouch/grade, so recipes operate on cleaned pixels).
4. **F4.b (optional, separate commit): LaMa ONNX** for large-region removal — `models/lama.onnx` (~200MB, gated behind a download-on-first-use prompt like other model acquisition), loaded via the `parsing.py` session pattern + `build_ort_providers()`. `method="lama"` in `heal_region`. Fixed 512px tile inference with mask-aware crop + seamless paste-back.

### Tests / artifact QA
- Heal on flat/gradient/textured synthetic regions: no visible seam (gradient continuity at feather edge — same assertion style as F3 QA).
- Heal + recipe ordering: healed pixels must not reappear (heal runs before frequency separation).
- LaMa: tile-seam test, CoreML fallback test (mirrors existing BiSeNet batch fallback at `parsing.py:360`).

**Files:** `retouch/heal.py` (new), `retouch/blemish.py` (extract shared blend), `retouch/engine.py` (pre-pipeline hook + kwarg), `gui.py`, `retouch/session.py` (field), tests.
**Effort:** ~1 week (v0 Telea); +1 week for F4.b LaMa.

---

## Stage F5 — Face-Aware Liquify Sliders

**Goal:** PS Face-Aware Liquify parity, automatic per face.

### Steps
1. **Refactor `FaceReshaper`:** extract the warp-accumulation loop into `_apply_warps(img, warps)`; the current jaw/cheek/chin definitions become the `slimming` warp set.
2. **New warp sets** (each a function of landmarks, face_width, strength; MediaPipe 468-landmark indices):
   - `eye_size` (scale warps radially around each eye center — landmarks 33/133/263/362 rings)
   - `eye_distance`, `nose_width` (alae 48/278), `nose_length` (tip 4 vertical), `jaw_width` (extend existing), `chin_length` (152 vertical, both directions), `mouth_size` (61/291 radial), `smile` (corners 61/291 up + slight out), `forehead` (hairline band vertical).
   - Scale warps = multiple translation warps arranged radially (reuses the same falloff kernel — no new math).
3. **Params:** 9 new ParamSpecs (`reshape_eye_size`, … range −50..+50, default 0) + recipe keys under `"reshape": {...}` — **run the dead-key guard pattern from the start.** GUI: extend the existing "Face Reshaping" accordion (`reset_face_reshaping_btn` already exists at `gui.py:1588`).
4. **Multi-face:** same loop as slimming (already per-face). Negative values allowed (enlarge eyes is the #1 cosplay ask).

### Tests / artifact QA
- **Background-line preservation (the classic liquify tell):** synthetic image with straight lines behind the face — assert lines stay straight outside face_width×0.6 of any control point.
- Identity guard: warp displacement magnitude capped (≤ face_width × 0.06 at strength 100).
- Landmark-symmetry test (left/right warps mirror), no-face no-op, all-sliders-zero = passthrough (byte-identical).

**Files:** `retouch/geometry.py`, `retouch/params.py`, `retouch/recipes.py` (optional recipe use), `gui.py`, tests.
**Effort:** ~2 weeks.

---

## Stage F6 — Look-from-Reference (reverse-engineer any edit into an editable preset)

**Goal:** drop in a finished photo (like the 爻一爻 references) → get an **editable grading preset JSON** (curve + split-tone + WB + HSL), not an opaque transfer. Stronger than the roadmap's "curve editor first" plan — and it makes the preset ecosystem self-feeding. The interactive curve-editor UI is deferred (Gradio has no native curve widget; a custom JS component is a timeboxed follow-up, not a dependency).

### Steps
1. **`StyleAnalyzer.extract_look(reference_img, base_img=None)`** in `retouch/style.py`: works in two modes — paired (original+edited, existing path) and **unpaired** (reference only, compare statistics against neutral expectations):
   - **L curve:** map percentiles (0,5,25,50,75,95,100) of base→reference L → curve points in the preset `curves.L` format.
   - **WB:** per-channel gain from gray-world / brightest-region estimate → `white_balance` dict.
   - **Split tone:** mean LAB a/b (→LCH hue/chroma) in shadow (L<85), midtone, highlight (L>170) bands → `split_tone_three_way`.
   - **Per-hue HSL:** compare saturation by hue bucket (reuse the 8 hue centers from `_apply_hsl_adjustments`, `grading.py:947`) → `hsl_adjustments`.
2. **Output = standard preset JSON** written to `presets/` (auto-discovered, immediately usable in the Color Grade dropdown and in recipes). Include `"description": "extracted from <filename>"`.
3. **GUI:** in the existing Color Transfer accordion, add "Extract Editable Preset" button + name field → saves preset, refreshes dropdown, applies at `grade_intensity` 65 for review.
4. **Validation loop:** apply extracted preset to base image, report ΔE against reference (rough fit score shown to the user; sets expectations that extraction is a starting point).

### Tests
Round-trip: grade an image with a known preset → extract from the pair → extracted parameters within tolerance of the known preset (curve monotonic, shadow hue ±15°). Unpaired mode: extraction from the moonlight reference class produces cool shadows (sanity assertions, not exact values).

**Files:** `retouch/style.py`, `gui.py`, `tests/test_style.py`.
**Effort:** ~1–1.5 weeks.

---

## Stage F7 — AI Denoise + Super-Resolution

**Goal:** high-ISO smoke-set photos (exactly the reference-image shooting conditions) come out clean; heavy crops export sharp.

### Steps
1. **`retouch/enhance.py` (new):** `denoise(img, strength)` (SCUNet or NAFNet ONNX) and `upscale(img, scale=2|4)` (Real-ESRGAN ONNX). Session loading copies the `parsing.py` lazy-init + provider-fallback pattern; models download-on-first-use into `models/` (add to `models/ACQUISITION`-style doc; luts/ has the precedent `ACQUISITION.md`).
2. **Tiling:** fixed 512px tiles with 32px overlap + feathered merge (both models are memory-hungry at 4K+). One shared `_tiled_inference(model, img, tile, overlap)` helper.
3. **Pipeline placement:** denoise = optional **pre-pipeline** step (before face detection — cleaner input helps detection too). Upscale = **export-time** option (GUI export panel: "Upscale 2×/4×"), never inside the pipeline.
4. **Params:** `denoise` ParamSpec (0–100, blends denoised vs original — strength = opacity, model runs once); upscale is an export flag, not a ParamSpec.

### Tests / artifact QA
- Tile-seam assertion (uniform-noise image → no periodic 512px structure in output FFT).
- Face-texture preservation: denoise at 50 must keep pore contrast above a floor (guard against plastic skin — measure high-band energy from `FrequencySeparator`).
- Provider fallback (CPU-only run), latency budget logged in `benchmark.py`.

**Files:** `retouch/enhance.py` (new), `retouch/engine.py` (pre-pipeline hook), `gui.py`, `io.py`/export path, `benchmark.py`, tests.
**Effort:** ~1.5–2 weeks.

---

## Recommended order & dependencies

> ℹ️ Within-tier order below is current. For where Tier 2 sits relative to the other tiers, see master sequencing in `PLAN_TIERP_PERF_ARCH_SHIP.md` (Tier 2 runs after P1/F8/F1/F11/P3/F2/F3).

```
F4 spot heal (v0 Telea)  ──►  F5 liquify  ──►  F6 look-from-reference  ──►  F7 denoise/SR  ──►  F4.b LaMa
```

- F4 v0 first: smallest, highest daily-use value, and establishes the paint-surface + pre-pipeline-step patterns.
- F5 next: pure extension of tested warp code, zero new dependencies.
- F6 before F7: no model downloads, feeds the preset ecosystem you actively use.
- F7 + F4.b last: both introduce large ONNX models — batch the "model acquisition UX" work once.
- Tier 1 interplay: none of F4–F7 *requires* F1–F3, but sessions (F2) make heals/reshapes persistent — if Tier 1 isn't done yet, F4 heal persistence degrades gracefully to per-run only.

## Verification (end-to-end)

1. Full pytest suite green after each stage (currently 442 in the touched areas; 1600+ overall).
2. Real-photo workflow test per stage: heal a stray hair at 100% zoom (F4); enlarge eyes +20 / slim nose −15 on a portrait, check background lines (F5); extract preset from the 爻一爻 reference, apply to an unedited shot, compare against the shipped `moonlight_porcelain` (F6 — great self-test: extraction should land near the hand-tuned preset); denoise an ISO-3200 shot, verify pores survive (F7).
3. `benchmark.py` run before/after F7 to record model latency on Apple Silicon (CoreML) vs CPU.
4. GUI smoke test via `dev.sh` after each stage.

## Open items carried forward (report-only)
- `anime_crystal_void` 7 dead keys (`retouch/recipes.py:378-384`) — still unwired; F5's ParamSpec work is a natural moment to also add the repo-wide recipe-key validation test if you want it.
- Interactive drag-curve editor UI — deferred from F6 (needs a custom JS component; revisit after Tier 2).
