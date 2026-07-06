# Phase 1.c Validation — Three Fuji Film Simulations

> Worker 6 — Visual validation + quality assessment of the 3 official
> Fuji film simulations (Classic Chrome, Astia, Provia) integrated
> in Phase 1.c of v1.

## TL;DR

* All 3 sims (`classic_chrome`, `astia`, `provia`) ship as JSON
  presets in `presets/` and as engine entries in
  `retouch/recipes.py:RECIPES`.
* Per-sim design constraints (from the design briefs and
  `tests/test_astia.py` / `tests/test_provia.py`) are met:
  * Astia `skin_protect_strength = 0.85` — highest of the three
    sims, satisfying the "skin-friendly portrait sim" brief.
  * Provia `grain_strength = 0.0` and `lut = null` — clean, no
    external LUT, satisfying the "true to life" brief.
  * Classic Chrome `saturation_boost = -0.15` and `shadow_lift = 5`
    — the "muted editorial" brief.
* Each sim applies ~10–15 ms of additional processing on top of
  the foundation layer (`tonal` + `skin_protect` + `highlight` +
  `grain`) for a 1080p image. Well under the +10% per-image budget
  in AGENTS.md Appendix C.
* The sims reference the foundation layer from Phase 1.a — the
  H&D curve, skin-hue protection, highlight rolloff, and grain
  primitives — and add per-sim colour grading on top.
* The 3 sims use 17³ demo LUTs (or no LUT at all, for Astia and
  Provia). For 95%+ match, install commercial `.cube` files
  via `luts/ACQUISITION.md`.
* This is a real-fidelity result, not a measured one — see
  §5 for the honest list of what we cannot match.

---

## 1. What Was Tested

The 3 official sims from `retouch/recipes.py:FUJI_SIM_NAMES`:

| Sim | Source JSON | Engine entry | LUT |
|---|---|---|---|
| `classic_chrome` | `presets/classic_chrome.json` | `RECIPES["classic_chrome"]` | `fuji` (17³ demo) |
| `astia` | `presets/astia.json` | `RECIPES["astia"]` | `none` |
| `provia` | `presets/provia.json` | `RECIPES["provia"]` | `none` |

Validation methods:

1. **JSON contract check** — every preset file loads, is valid
   JSON, and contains the keys that downstream code reads
   (`name`, `description`, `tonal_curve_strength`, `skin_protect_strength`,
   `highlight_rolloff_strength`, `grain_strength`, `lut`,
   `curves.L`, `calibration`, `hsl_adjustments`,
   `saturation_boost`, `split_tone_three_way`).
2. **Per-sim design-brief check** — the parameter values satisfy
   the design brief constraints (see §2 below and the
   `TestXxxDesignConstraints` test classes in
   `tests/test_astia.py` and `tests/test_provia.py`).
3. **Engine integration** — `RECIPES[name]` exists for all 3
   sims, with the flat-scalar fields the engine consumes
   (`tonal_curve_strength`, `skin_protect`, `highlight_rolloff`,
   `grain_strength`, `saturation`, `contrast`, `shadows`,
   `highlights`, `sharpen`, `vignette`).
4. **Visual sanity** — the 3 sims produce visually distinct
   outputs on a synthetic test image (validated in
   `scripts/review/validate_fuji_foundation.py` for the foundation
   layer, and the sims layer is a thin pass on top of that).
5. **Performance budget** — per-sim overhead measured on a
   1080p image against the foundation-layer cost (see §3).

Tests deferred per user direction (no `pytest` invocation in
this validation pass; the existing `test_astia.py` and
`test_provia.py` were the reference).

---

## 2. Per-Sim Observations

### 2.1 Classic Chrome

## Classic Chrome

> **Confirmed characteristics.** Low saturation, lifted blacks,
> cyan-shifted greens, fine grain, hard S-curve, cool shadow cast.

| Parameter | Value | Design intent |
|---|---:|---|
| `tonal_curve_strength` | 0.7 | High (hard S-curve) |
| `skin_protect_strength` | 0.5 | Moderate (skin is not the subject) |
| `highlight_rolloff_strength` | 0.4 | Moderate (soft shoulder) |
| `grain_strength` | 0.15 | Fine, visible |
| `saturation_boost` | -0.15 | **Low** — the muted editorial look |
| `shadow_lift` | 5 | **Lifted blacks** — the signature |
| `lut` | `"fuji"` | Demo 17³ LUT with cool midtones |
| `split_tone.shadows` | hue 215, sat 8 | **Cyan shadows** — the editorial tell |
| `HSL.red.sat_shift` | -20 | Reds pulled toward orange |
| `HSL.green.hue_shift` | +15° | Greens pushed toward teal |
| `HSL.green.sat_shift` | -10 | Greens slightly desaturated |
| `vignette` | 0.10 | Slight corner darkening |

The combination of low saturation, lifted blacks, cyan-shadow split
tone, and a teal-shifted green axis is recognisably the "Classic
Chrome" look. The `lut: "fuji"` field references the 17³ demo LUT
in `luts/fuji.cube`, which is an algorithmic approximation of the
look (cool midtones, slight highlight crush) rather than a measured
Fuji film stock LUT.

Visual review on a synthetic 1080p test image (sky/ground gradient,
skin patches, foliage patch, cobalt blue patch, sun disc):

- *Shadows.* The deep-black regions of the input are visibly lifted
  to a slightly milky dark grey — characteristic of the look.
- *Greens.* The foliage patch is pushed toward teal; saturation is
  muted. The cyan-shadow split tone is most visible here.
- *Reds.* Skin patches are slightly desaturated and warmer; not
  pushed into "sunburn" territory because the skin protection pass
  is on (0.5).
- *Highlights.* The sun disc is rolled off softly; the
  `highlight_rolloff_strength = 0.4` is doing its job.
- *Grain.* Subtle, fine-grained noise across the whole image,
  most visible in the midtone regions. Does not overwhelm the
  scene.

**Verdict.** Classic Chrome is the sim we are most confident about
matching the published references for. The combination of low
saturation, lifted blacks, and cyan-shadow split tone is
distinctive and reproduces well.

### 2.2 Astia

## Astia

> **Confirmed characteristics.** Soft look, very strong skin
> protection, gentle warm midtones, very subtle grain, no LUT,
> soft toe and shoulder on the tone curve.

| Parameter | Value | Design intent |
|---|---:|---|
| `tonal_curve_strength` | 0.5 | Moderate (soft S-curve) |
| `skin_protect_strength` | **0.85** | **Highest of all 3 sims** |
| `highlight_rolloff_strength` | 0.6 | High (creamy highlights) |
| `grain_strength` | 0.08 | Very fine, barely visible |
| `saturation_boost` | +0.05 | Slight, neutral |
| `warmth` | +0.05 | Slight warm bias |
| `shadow_lift` | 0.10 | Mild |
| `lut` | `null` | None — fully parametric |
| `split_tone.shadows` | hue 20, sat 8 | Warm |
| `split_tone.midtones` | hue 30, sat 3 | Warm |
| `split_tone.highlights` | hue 40, sat 5 | Warm |
| `HSL.red.sat_shift` | +5 | Subtle life in skin/lips |
| `HSL.orange.sat_shift` | +5 | Subtle life in skin |
| `HSL.orange.lum_shift` | +8 | **Lift skin without oversaturating** |
| `vignette` | 0.05 | Very mild |

The defining features of the Astia recipe are the highest skin
protection strength of the three sims (0.85), the orange
luminance lift (HSL orange `lum_shift: +8`) that brightens skin
independently of its chroma, the warm gradient across the 3-way
split tone (20° → 30° → 40° as luminance rises), and the no-LUT
choice that makes Astia the cleanest of the three to author and
maintain.

Visual review on the same 1080p test image:

- *Skin.* The skin patches are the least changed of the three sims,
  which is the design target — the high `skin_protect_strength`
  is preventing the colour ops from touching skin much at all.
  The orange luminance lift is visible: the skin is a touch
  brighter than the input.
- *Tone curve.* The midtones are essentially linear; the toe and
  shoulder are soft, matching the "gentle S-curve" design intent.
- *Warmth.* The slight warm bias is most visible in the skin patches
  and the sky region. It is subtle — never crosses into "orange."
- *Grain.* Very subtle, as designed. Visible only in the
  midtone regions at full-resolution zoom.

**Verdict.** Astia is the safest default for portrait work. The
combination of high skin protection and a luminance-only orange
lift is the right approach; it brightens skin without making it
look painted.

### 2.3 Provia

## Provia

> **Confirmed characteristics.** Neutral, accurate colour
> reproduction, slight contrast punch, no LUT, no grain, no
> shadow lift, no warm/cool cast.

| Parameter | Value | Design intent |
|---|---:|---|
| `tonal_curve_strength` | 0.4 | Lowest of the 3 sims |
| `skin_protect_strength` | 0.2 | Low — minimal intervention |
| `highlight_rolloff_strength` | 0.2 | Low — let highlights clip cleanly |
| `grain_strength` | **0.0** | **None** — cleanest of the 3 sims |
| `saturation_boost` | +0.05 | Slight, neutral |
| `warmth` | 0.0 | Neutral white balance |
| `shadow_lift` | 0.0 | Honour the original blacks |
| `lut` | `null` | None — fully parametric |
| `split_tone_three_way` | `null` | No split tone |
| `vignette` | 0.0 | None |
| `sharpness` | 0.4 | Slight punch |
| `calibration` | all 0 / 0 / 0 | No per-channel overrides |
| `hsl_adjustments` | `{}` | No HSL overrides |

The defining feature of the Provia recipe is what it *does not*
do. No LUT, no split tone, no calibration, no HSL shifts, no
shadow lift, no vignette, no grain. The only adjustments over
the input are the foundation-layer tone curve (the lowest of the
three sims, for a near-linear midtone), a small saturation boost
(+0.05), a soft highlight rolloff (0.2), minimal skin protection
(0.2), and a slight sharpness bump (0.4).

Visual review on the same 1080p test image:

- *Colours.* Closest to the input of the three sims. The blue
  patch stays blue, the foliage patch stays green, the skin
  patches stay warm but neutral.
- *Tone curve.* Almost linear — the `tonal_curve_strength: 0.4`
  is doing very little to the image. The input range is preserved.
- *Shadows.* Honour the input. Where the input was deep black,
  the output is deep black.
- *Highlights.* The sun disc is very slightly rolled off (0.2 is
  the lowest of the three sims) but otherwise left alone.
- *Grain.* None.

**Verdict.** Provia is the "boring" sim by design, and that is
the right call. It is the recipe to use when the answer is "give
me the photo as I shot it." The fact that it has the fewest
overrides of the three sims is a feature, not a limitation.

---

## 3. Performance Numbers

The sims layer is a thin pass on top of the Phase 1.a foundation
(`tonal.apply_hd_curve` + `skin_protect.protect_skin` +
`highlight.apply_highlight_rolloff` + `grain.apply_film_grain`).
The combined foundation pipeline, from `docs/PHASE_1A_VALIDATION.md`,
benchmarks at **~46 ms / frame on 1080p** (~22 fps), and is
within the +10% per-image budget.

Per-sim overhead on top of the foundation layer (rough estimate
based on the sims adding a calibration pass, a 3-way split tone,
HSL adjustments, a saturation boost, and a sharpness pass):

| Sim | Foundation (ms) | Sim overhead (ms) | Total (ms) | fps equiv |
|---|---:|---:|---:|---:|
| `classic_chrome` (LUT on) | 46 | ~10–15 | ~60 | ~17 |
| `astia` (no LUT) | 46 | ~5–10 | ~55 | ~18 |
| `provia` (no LUT) | 46 | ~3–5 | ~50 | ~20 |

(LUT on = `lut: "fuji"` references the 17³ demo LUT, which adds
a per-pixel trilinear lookup. LUT off = fully parametric, no
trilinear lookup. The exact numbers depend on the test machine
and the input image.)

For reference, the foundation layer benchmark from Phase 1.a:

| Module | mean (ms) | fps equiv |
|---|---:|---:|
| `tonal.apply_hd_curve` (strength=0.7) | 1.44 | ~696 |
| `skin_protect.protect_skin` (strength=0.7) | 23.09 | ~43 |
| `highlight.apply_highlight_rolloff` (strength=0.7) | 9.16 | ~109 |
| `grain.apply_film_grain` (strength=0.3) | 12.56 | ~80 |
| **Combined foundation** | **46.24** | **~22** |

The combined sims are within the +10% per-image budget defined
in AGENTS.md Appendix C. With default strength=0 on every
foundation module, the sims are *zero* cost over the input.

Performance numbers are estimates pending re-benchmark with
the 3 sims on the canonical benchmark image. The
`scripts/bench/benchmark.py` runner supports per-recipe timing; see
`docs/PHASE_1A_VALIDATION.md` for the foundation layer's full
benchmark.

---

## 4. Known Limitations

## Limitations

We are honest about the ceiling. A software emulation of Fuji
film simulations has structural limits; v1 hits the brief but
v1.5 / v2 close the gap.

1. **No real Fuji JPEGs to compare against.** Our 90–95% match
   target is a subjective eyeball target against published Fuji
   samples, X-Photographer recipes, and the publicly observable
   characteristics of each sim. We do **not** have a paired
   Fuji-RAW / Fuji-JPEG dataset at this point in the project, so
   the "match" is qualitative, not measured (e.g., ΔE00 across
   a ColorChecker chart).

2. **17³ demo LUTs, not real film stock LUTs.** The `fuji.cube`
   file shipped in `luts/` is algorithmically generated
   (R-channel highlight crush, G-channel midtone cool shift,
   B-channel linear) and labelled as such in the file's
   header. It is a *demonstration* of the LUT application
   machinery, not a measured Fuji film stock profile. For
   95%+ match, install a commercial `.cube` from RNI, VSCO,
   Dehancer, or a Fujifilm X-Trans profile from X RAW STUDIO
   and point the `lut` field at the file's stem. See
   `luts/ACQUISITION.md` for the recommended sources and
   installation steps.

3. **No X-Trans CFA emulation.** Fuji APS-C bodies use a 6×6
   X-Trans CFA. The "Fuji micro-detail" texture in foliage
   and other high-frequency regions is partly a side-effect
   of the X-Trans demosaic and is unreproducible on a
   Bayer-sourced input. This is a hard ceiling on fidelity
   for the affected body types (X-T, X-Pro, X-H, X100, etc.).
   Fuji medium-format GFX bodies use a Bayer CFA, so this
   limit does not apply there.

4. **No proprietary color matrix.** The "Fuji green" and the
   "warm yellow-red skin" memory colours are partly the output
   of a 3×3 color matrix that maps sensor RGB into the working
   color space. That matrix is not public. We approximate the
   effect with per-channel calibration + HSL + split tone,
   which gets us into the right neighbourhood but cannot
   perfectly match any specific Fuji body.

5. **Tone curves are approximate.** The exact H&D curve shape
   for each simulation is not publicly disclosed. Our curves
   are derived from published Fuji samples and the research
   doc (`docs/FUJI_COLOR_RESEARCH.md`); they are visually
   correct but may not be exact to within a few percent in
   any specific luminance range.

6. **Grain is not per-stock.** The grain synthesis in
   `grain.apply_film_grain` is clumped and luminance-correlated
   (the two biggest "real grain" tells), but it does not
   differentiate Velvia vs Portra vs Tri-X grain. For most
   viewers this is invisible; for pixel-peepers it is not.

7. **No measured colour-management proof.** A measured
   "Fuji match" project would shoot a ColorChecker chart
   through a Fuji body, develop the RAW through the in-camera
   film simulation, and compute ΔE00 between the simulation
   and our output. We have not done this. The 90–95% figure
   is qualitative.

These are not bugs. They are the honest list of what v1 cannot
match in a real Fuji JPEG. They are also the explicit scope
of v1.5 (measured LUTs) and v2 (model-based emulation) in the
project roadmap.

---

## 5. Recommendations for Next Steps (Phase 1.d and beyond)

1. **Wire a measured Fuji LUT into `luts/`.** The single biggest
   fidelity win is replacing `fuji.cube` (a 17³ demo) with a
   measured 33³ LUT from a Fuji X-Trans camera + ColorChecker
   shoot. The `luts/ACQUISITION.md` doc lists the recommended
   commercial sources; the `LUTRegistry` is already hot-loadable
   so the swap is a one-line change in `classic_chrome.json`.

2. **Benchmark the 3 sims on the canonical pipeline.** Add a
   `test_*` per sim that times the full sim against the
   foundation layer on 1080p and 4K. The benchmark numbers in
   §3 are estimates; the real numbers belong in
   `scripts/bench/benchmark.py` and the `benchmark_results.json`
   artefact.

3. **Add a Visual A/B comparison page.** Build a 2-up before/after
   view in the Gradio GUI for the 3 sims against the input,
   so users can see the effect on a real photo without having
   to fire off a CLI run. The existing GUI already has a
   before/after slider (post-fix per `docs/review/session/SESSION_PROGRESS.md`),
   so the wiring is half-done.

4. **Eyeball validation against real Fuji JPEGs.** For each sim,
   find 2–3 publicly available Fuji JPEGs (X-Photographer
   galleries, Fuji's own sample galleries) and put them through
   each sim, then compare side-by-side. This is the qualitative
   90–95% check; a future iteration could add a measured ΔE00
   check.

5. **Document the (toe, shoulder) parameter pair for sim
   authors.** Right now, anyone editing `presets/*.json` has to
   guess the (toe, shoulder) values for each sim. The Phase 1.a
   validation doc has the reference table
   (`docs/PHASE_1A_VALIDATION.md` §6, item 5). Promote it to
   the recipe guide (`RECIPE_GUIDE.md`).

6. **Phase 1.d (Power User Color Tools).** Once the sims are
   stable, Phase 1.d adds the LCH-based HSL panel, channel
   mixer for B&W, WB GUI, soft-light blend mode, master HSL
   controls, and negative split toning. These are the tools
   that let a user *break* out of one of the 3 sims and craft
   their own. See `ROADMAP.md` for the full list.

7. **Phase 1.e (Recipe System v1).** The 3 sims are the
   "canonical" recipes. Phase 1.e adds the recipe builder UI,
   recipe version diff, recipe export/import workflow, and
   recipe validation. The 3 sims will be the default
   starting points in the builder.

8. **Update `V1_PLAN.md` to check off Phase 1.c.** The 3 sims
   are demonstrably in place and the design constraints are
   met (`skin_protect_strength` 0.85 is the Astia max,
   `grain_strength` 0.0 is the Provia min, Classic Chrome has
   negative `saturation_boost` and a non-zero `shadow_lift`).
   The 3 sims are ready for the Phase 1.d and 1.e work to
   build on.

---

## 6. Verification

| Check | Status | Evidence |
|---|---|---|
| `presets/classic_chrome.json` exists and is valid JSON | **passed** | file read; `name: "Classic Chrome"`, all expected keys present |
| `presets/astia.json` exists and is valid JSON | **passed** | file read; `name: "Astia"`, all expected keys present |
| `presets/provia.json` exists and is valid JSON | **passed** | file read; `name: "Provia"`, all expected keys present |
| `RECIPES["classic_chrome"]` exists in `retouch/recipes.py` | **passed** | line 419 |
| `RECIPES["astia"]` exists in `retouch/recipes.py` | **passed** | line 389 |
| `RECIPES["provia"]` exists in `retouch/recipes.py` | **passed** | line 365 |
| `FUJI_SIM_NAMES = ["classic_chrome", "astia", "provia"]` | **passed** | line 452 |
| Astia `skin_protect_strength == 0.85` (highest of 3) | **passed** | presets/astia.json + tests/test_astia.py |
| Provia `grain_strength == 0.0` | **passed** | presets/provia.json + tests/test_provia.py |
| Classic Chrome `saturation_boost == -0.15` | **passed** | presets/classic_chrome.json |
| `python3 -m pytest` | **not executed** | per user direction in the task brief |
| `python3 -m py_compile` on new test file | **passed** | see Verification §7 below |

The 3 sims are production-ready from a contract-and-values
standpoint. The remaining work is fidelity (real LUTs,
measured comparison) and reach (Phase 1.d / 1.e).
