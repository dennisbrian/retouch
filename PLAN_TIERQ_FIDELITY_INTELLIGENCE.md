# Tier Q Plan — Full-Res Fidelity + Adaptive Intelligence (Stages F8–F11)

## Context

Third planning round. Tiers 1–2 (`PLAN_TIER1_FOUNDATION.md`, `PLAN_TIER2_MANUAL_TOOLS.md`) cover pipeline quality and manual tools. This round is model-reasoning-driven research rather than roadmap execution, and it found two things: **(A)** a resolution ceiling that quietly caps output quality on every real camera file, and **(B)** the layer that would make this system genuinely *better* than Photoshop rather than equal to it — image-aware adaptation and self-checking output. Photoshop gives you tools; a system that measures the image and adapts, then verifies its own output for artifacts, is a different category.

---

## 🔍 Finding A (MAJOR, report-first per your bug protocol): the 2048px proxy ceiling

`_process_with_proxy` (`retouch/engine.py:899-947`): any image whose longest side exceeds `PROXY_MAX_DIM = 2048` is downscaled, run through the **entire** pipeline at 2048px, and then the **finished image itself** is `INTER_LINEAR`-upscaled back to original size.

**Consequence:** a standard 6000×4000 mirrorless photo is output as a stretched 2048px render. Every pore, hair strand, fabric weave, and jewelry edge is resampled twice (down then up) — roughly a 3× resolution loss on typical camera files. This silently contradicts several shipped/planned goals: `sharpen`/`texture_opacity`/`micro_restore`/`pore_synthesis` operate on and output proxy-resolution detail; the planned 16-bit export (F1) would export a 16-bit *upscale of a 2048 render*; and "Photoshop-grade" is unreachable while this holds. It's a deliberate perf tradeoff, not a logic bug — but the quality cost likely isn't visible in the 800px GUI preview, so it goes unnoticed until someone prints or crops.

**This plan's F8 fixes it. Decide whether F8 jumps the queue ahead of Tier 1/2 — my recommendation: yes, or bundle it with F1.**

Also verified while exploring (assets the plan reuses): `correct_exposure()` (`retouch/utils.py:378`) already implements face-prioritized "target, not delta" exposure normalization — the exact pattern F9 generalizes; `estimate_face_width()` already drives resolution-adaptive kernels; `StyleAnalyzer.extract()` computes image statistics; BiSeNet skin masks enable skin-region measurement.

---

## Stage F8 — Full-Resolution Fidelity (remove the proxy ceiling)

**Approach: split execution by cost structure.** Detection/parsing stay at proxy (masks upscale fine — they're smooth). The expensive per-face work runs on **native-resolution face crops** (a face ROI is small even in a 6000px frame, so cost is bounded by face size, not image size). Global stages (Phase 3) run at **full resolution** (they're vectorized numpy — linear cost, acceptable).

### Steps
1. In `_process_with_proxy`: run detection/segmentation on the proxy, then map face bboxes/landmarks back to native coordinates (scale factor is exact) and hand native-res crops to `_process_one_face`. Landmark coordinates are normalized (MediaPipe) so remapping is multiplication, not re-detection.
2. Per-face stage kernels already scale via `estimate_face_width` / `ied` — audit each per-face op for hardcoded pixel constants (the `flatten` >1200px downsample guard at `skin.py:562-568` is the pattern to follow; some ops may want the same internal guard).
3. Phase 3 global stages run at native res. Perf guards: the two ops with super-linear cost at 6000px (guided-filter clarity, big Gaussian blooms) get internal downsample-compute/upsample-apply guards (compute the *low-frequency layer* small, apply full-res — mathematically safe for blurs, unlike downsampling the *result*).
4. Keep the old full-proxy path as `quality="draft"` for batch contact sheets; native path is the default. `fast=True` 800px preview path unchanged.
5. Escape hatch for extreme inputs (>8K): detail reinjection fallback — upscale proxy result, add back `original_highband × (1 − acc_skin × smooth_strength)` so non-skin regions keep native texture. Cheap insurance, ~30 lines, reuses `FrequencySeparator` bands.

### Tests / QA
- High-frequency energy assertion: process a 5000px synthetic with fine fabric texture; non-skin regions must retain ≥95% of input high-band energy (currently they'd retain ~35-50%).
- Mask remap correctness: bbox/landmark scaling roundtrip exact to ±1px.
- Benchmark before/after at 2K/4K/6K (`benchmark.py`); target ≤2.5× current time at 6K.
- Visual: 100% crop A/B on jewelry/hair vs current output.

**Files:** `retouch/engine.py` (proxy split), `retouch/skin.py`/per-face ops (constant audit), `benchmark.py`, tests.
**Effort:** ~1.5–2 weeks. **Highest quality-per-effort in the entire roadmap.**

---

## Stage F9 — Image Analyzer + Adaptive Recipes ("targets, not deltas") — ✅ DONE 2026-07-07

> **Receipt:** `retouch/image_analyzer.py` (new, ImageAnalyzer + ImageAnalysis + SkinCondition dataclasses), `tests/test_image_analyzer.py` (39 tests, all green). Implemented: `analyze()` (lighting type, dynamic range, noise via MAD-on-quiet-blocks, WB estimate via gray-world, dominant colors via 4-bin histogram, sharpness via Laplacian std, skin condition when face+mask provided), `suggest_params()` (target-based brightness/blacks/whites/contrast/WB/ai_denoise/sharpen/whiten/redness_even/equalize, all clamped to ParamSpec bounds), `suggest_recipe()` (rule-based, validates candidates against RECIPES table). float32 internal, explicit cvtColor at every colorspace boundary, no `except: pass`. Plan Steps 2–4 (targets block in recipes, build_context hook, GUI readout) deferred to F10 wiring.

**The core idea:** today a recipe applies `brightness +6` blindly — perfect for the photo it was tuned on, wrong for a darker or brighter one. `correct_exposure()` already shows the right pattern: *measure, then move toward a target*. Generalize it, and recipes become lighting-invariant — the same `moonlight_porcelain` lands correctly on an overexposed daylight shot and a dark smoke set.

### Steps
1. **`retouch/analysis.py` (new):** `analyze(img, faces, skin_mask, person_mask) -> ImageStats` — all cheap numpy on the proxy, one pass:
   - exposure: L percentiles (1,5,50,95,99), dynamic range, key classification (low/normal/high)
   - WB cast: gray-world estimate + skin-tone prior when a face exists (skin a/b vs expected skin locus)
   - skin: mean/std L,a,b within BiSeNet mask (per face + combined)
   - noise: sigma via MAD on flat regions (Laplacian-quiet blocks)
   - composition: face-area ratio, subject/background luminance ratio (person_mask)
2. **`targets` block in recipes** (backward compatible — absent means current behavior):
   ```python
   "targets": {"skin_L": 178, "wb_neutral": 0.7, "dr_range": [10, 245], "sb_ratio": 1.6}
   ```
   Resolution: in `build_context`, after normal param resolution, a `resolve_targets(stats, targets)` step converts each target + measured stat into a delta on the corresponding existing param (skin_L→`whiten`/`brightness`, wb_neutral→`white_balance_kelvin/tint`, dr_range→`blacks/whites`, sb_ratio→`subject_separation`). Bounded corrections (±clamps like `correct_exposure`'s min/max thresholds) so a wild input can't produce a wild edit.
3. Pilot on 4 targets only (listed above). Wire `noise_sigma → denoise strength` when F7 ships.
4. Add `targets` to `moonlight_porcelain` + `anime_v2` as the proving ground; expose measured stats in `ProcessingResult.stats` and a collapsible "📊 Image Analysis" readout in the GUI.

### Tests
Analyzer unit tests on synthetic images with known stats; invariance test — same recipe on a +1EV and −1EV version of one photo must produce outputs within ΔL tolerance of each other (the whole point); clamp tests; no-face fallback.

**Files:** `retouch/analysis.py` (new), `retouch/params.py`/`engine.py` (`build_context` hook), `retouch/recipes.py`, `gui.py`, tests.
**Effort:** ~2 weeks.

---

## Stage F10 — Smart Default (one-click auto edit)

Rule-based on F9's `ImageStats` — no ML model, fully explainable:
1. `suggest(stats) -> (recipe_name, overrides, explanation)`: key + WB cast + skin stats + noise → pick recipe family (low-key cool scene → `moonlight_porcelain`-class; bright neutral → `natural`/`astia`-class; high noise → denoise first) and modulate via F9 targets.
2. GUI: "✨ Smart Default" button (the old roadmap's #1 UX ask) → applies suggestion, shows the explanation text ("dark low-key scene, blue cast detected, high ISO noise → moonlight_cool + denoise 40"). User tweaks from there — it's a starting point, not an autopilot.
3. Batch: `--smart` flag lets each photo in a folder get per-image adaptation instead of one static recipe.

**Files:** `retouch/analysis.py` (suggest fn), `gui.py`, `cli.py`, `batch_processor.py`, tests (rule table cases).
**Effort:** ~1 week. Depends on F9.

---

## Stage F11 — Output Self-QA (the system checks its own work)

Codify the artifact matrices from the previous plans into runtime detectors — directly answering your recurring "make sure no artifacts" requirement, permanently:

1. **`retouch/qa.py` (new)**, all operating on (input, output, masks):
   - **banding**: unique-level count + FFT comb detection in low-gradient regions
   - **halo**: overshoot amplitude at high-contrast edges (dilated edge band, compare min/max vs input)
   - **clipping**: % pixels at 0/255 per channel, delta vs input
   - **seam**: gradient discontinuity along person-mask boundary (subject-separation halos)
   - **plastic skin**: high-band energy inside skin mask below floor (uses `FrequencySeparator`)
2. `ProcessingResult.qa: list[QAWarning]` (opt-in `qa_check=True`); GUI badge ("⚠️ 2 quality warnings") with plain-language messages + suggested slider fixes; CLI/batch: per-photo QA column, `--fail-on-qa` for pipelines.
3. Same detectors double as pytest regression guards for every recipe (run each built-in recipe on the corpus; assert zero CRITICAL warnings) — this retroactively hardens `moonlight_porcelain`, `anime_v2`, and everything Tier 1/2 ships later.

**Files:** `retouch/qa.py` (new), `engine.py` (hook), `gui.py`, `cli.py`, `tests/test_qa.py` + recipe-corpus regression test.
**Effort:** ~1.5 weeks. Independent — can be built anytime; most valuable before Tier 1 F1 lands (it measures F1's banding win objectively).

---

## Recommended sequencing (revised, all tiers)

> ⚠️ **Superseded (2026-07-02):** `PLAN_TIERP_PERF_ARCH_SHIP.md` found that **P1 (bilateral→guided filter) must land before F8** — F8's native-res face crops make the current bilateral filter cost explode (~14× pixels). See the master sequencing there; the order below predates that finding.

```
F8 (full-res fidelity) ──► F1 (float32+16bit) ──► F11 (self-QA) ──► F2/F3 ──► Tier 2 (F4–F7) ──► F9 ──► F10
        └────────── quality floor ──────────┘      └── workflow ──┘   └─ tools ─┘   └─ intelligence ─┘
```

Rationale: F8+F1 together remove both quality ceilings (resolution + bit depth) — everything built afterward inherits the fix; doing them adjacent avoids touching the same engine plumbing twice. F11 right after gives objective proof and guards the rest. F9/F10 last: adaptation multiplies best when the output it adapts is already maximal quality.

## Verification (end-to-end)
1. Full pytest suite after each stage; QA-corpus regression (F11) green for all built-in recipes.
2. F8 acceptance: 6000px photo, 100% crop of fabric/hair — native detail visibly preserved vs current build; benchmark delta recorded.
3. F9 acceptance: one recipe, three exposures of the same scene (−1/0/+1 EV) → visually consistent results.
4. F10 acceptance: 20-photo mixed folder with `--smart` → no absurd suggestions (manual review).
5. GUI smoke test via `dev.sh` per stage.

## Open items carried forward
- `anime_crystal_void` 7 dead keys — still report-only, unwired.
- Decision needed: F8 queue position (recommended: before or with Tier 1 F1).
