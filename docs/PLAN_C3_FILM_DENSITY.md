# PLAN_C3_FILM_DENSITY.md — Parametric Film-Density Engine

> **Stage C3** of the East×West color tier (`PLAN_EASTWEST_COLOR_SUPREMACY.md`).
> Master plan row 17. **2 weeks. Depends on F1 (float32 pipeline ✅ DONE 2026-07-07).**
> Acceptance bar: parametric re-expression of one Fuji preset within ΔE ~3 of its LUT on the corpus.

**Status:** IMPLEMENTED (core engine) — 2026-07-07. `retouch/film.py` + ParamSpecs + `_stage_grade`/`_no_face_fallback` wiring + `tests/test_film.py` (25 tests, all green). LUT re-expression harness (§7.2), visual QA (§7.3), and GUI accordion (§5.4) remain.
**Target module:** `retouch/film.py` (new) · `retouch/grading.py` (block hook) · `retouch/params.py` · `retouch/recipes.py` · `gui.py`.

---

## 1. Motivation — Why LUTs Aren't Enough

The current film-look path is `_add_film_emulation` (`grading.py:1019`), which loads a `.cube` LUT (`retouch/lut.py::CubeLUT`) and applies it via trilinear interpolation. Tonal shaping lives separately in `tonal.py::apply_hd_curve` (an H&D sigmoid baked into a 256-entry uint8 LUT). HSL adjustments live in `_F_apply_hsl_adjustments` (`grading.py:1099`).

Three structural limits of this stack:

1. **Fixed grid, no parametric control.** A 33×33×33 cube encodes a fixed RGB→RGB map. There is no "make the toe 10% deeper" knob — the only lever is `strength` (an opacity blend against the original, `grading.py:1037-1039`). To vary the look you ship another 50 KB cube file.
2. **No scene-content adaptation.** A LUT is a static function. It cannot deepen shadows more on a low-key scene or roll highlights earlier on an overexposed one. Real film responds to exposure; the LUT path ignores exposure entirely.
3. **Baked crosstalk, not exposed.** The warmth-in-shadows / cyan-in-highlights behavior that makes film *look like film* is the per-dye crosstalk matrix (dye impurity). LUTs bake this into the grid; the user cannot tune it. Hue drift on rolloff — the "notorious six" drift flagged in `PLAN_EASTWEST_COLOR_SUPREMACY.md` row 30 — is a *consequence* of per-channel curves, not a controllable parameter.

C3 replaces the LUT as the *primary* film path (LUTs remain as a fallback / import format) with a **subtractive density model** + **hue-preserving master tone-map**, exposing the parameters LUTs hide.

---

## 2. Subtractive Density Model

### 2.1 How film works (the physics we model)

Color negative film is a **subtractive** medium. Three dye layers (cyan, magenta, yellow) each absorb light in one spectral band. Exposure hits silver-halide crystals; after development, each layer's *optical density* `D` is a function of the log-exposure `logE` it received. The H&D (Hurter-Driffield) curve `D = f(logE)` is the per-layer characteristic curve — a sigmoid with a **toe** (low-exposure, density floor), a **straight-line** midsection (linear ≈ gamma), and a **shoulder** (high-exposure, density ceiling).

The dye clouds are imperfect: a cyan layer leaks some magenta/yellow. This is the **inter-image crosstalk** — a 3×3 mixing matrix on densities. It is *the* mechanism behind film's color signature (warm shadows, cyan skies) and the thing LUTs bake in but cannot expose.

### 2.2 The math

Work in **log-density space**. Given linear-scene RGB `r ∈ [0,1]` (after inverse sRGB EOTF — reuse `color_science.py:59-63`'s `np.where(... 2.4)` EOTF, or accept gamma-encoded input and apply a 2.2 round-trip like `relight.py`'s bloom convention):

```
# 1. Exposure → per-channel log-density (the H&D curve, per dye layer)
#    r_c in [0,1]; epsilon guards log(0)
logE_c = log(r_c + ε)                      # c ∈ {R,G,B}
D_c     = hd_curve(logE_c; toe_c, shoulder_c, midpoint_c, gamma_c)
```

The H&D curve is the existing piecewise-cubic sigmoid from `tonal.py:95-124` (`_toe_segment`, `_shoulder_segment`, `_sigmoid`), generalized to per-channel `toe/shoulder/midpoint/gamma`. The current `hd_curve_lut` builds one curve on [0,1] input; C3 builds three, one per dye layer, applied in log-space (not the linear [0,1] space `tonal.py` uses).

```
# 2. Dye crosstalk — 3×3 matrix on density
D' = M_crosstalk @ D
#    M_crosstalk is near-identity: diagonal ≈ 1.0, off-diagonal ∈ [-0.15, +0.15]
#    e.g. Portra-class: cyan layer leaks ~0.08 magenta → M[1,0] ≈ 0.08
#    Sign convention: positive off-diagonal = the row-dye absorbs extra column-band light
```

```
# 3. Density → transmitted exposure (subtractive: dye subtracts light)
#    D_max is the layer's maximum density (~2.0-3.0 for negative stock)
T_c = 10^(-D'_c)                            # transmittance
r'_c = r_c * T_c                            # subtractive: dye attenuates the scene light
#    Equivalently: r'_c = r_c * 10^(-(M @ D))
```

The crosstalk matrix operates on *density*, not RGB — this is what makes the shadow-warmth emerge naturally: shadows have low `logE`, hence low `D`, but the off-diagonal terms add a small positive density to the magenta/yellow layers from the cyan layer's exposure, warming the toe. Per-channel RGB curves (the LUT/Photoshop path) cannot reproduce this because they apply independently per channel with no cross-channel density coupling.

### 2.3 Why float32 is mandatory here

The density math spans ~6 orders of magnitude: `logE ∈ [-6, 0]`, `D ∈ [0, 3]`, `T = 10^(-D) ∈ [0.001, 1.0]`. In uint8 (256 levels), `T < 1/255 ≈ 0.004` clips to zero — the entire deep-shadow density range, exactly where the crosstalk warmth lives, quantizes to black. The float32 pipeline (F1/E2 ✅) keeps full precision through `log → curve → matrix → 10^x`. This is why C3 was blocked on F1 (per `MASTER_PLAN.md:81`) and is now unblocked.

The existing float LAB/LCH converters (`color_space.py:62-95`, `bgr_f32_to_lab_f32` / `lab_f32_to_bgr_f32`) and the `ensure_float` / `to_uint8` boundary helpers (`precision.py`) are the established pattern. C3 follows it: accept uint8 or float32 [0,1] BGR, `ensure_float` on entry, operate in float32, return input dtype.

---

## 3. Hue-Preserving Master Tone-Map

### 3.1 The problem with per-channel curves

Every Photoshop RGB curve, every LUT, and the existing `_F_apply_rgb_curves` (`grading.py:717`) tone-map each channel independently. As the R/G/B channels roll off at different rates (they always do, once crosstalk is applied), **hue shifts** in highlights and shadows — the "notorious six" drift. This is why film *looks* like film (the drift is character) but also why digital-curve film looks *wrong* (the drift is uncontrolled).

### 3.2 The C3 approach — luminance tone-map + chroma re-attachment

Decouple luminance compression from chroma, with one slider controlling how much drift to *reintroduce*:

```
# After the density model produces r' (post-crosstalk, still linear):
# 1. Luminance norm — compress luminance only, hue-locked
Y = 0.2126*R + 0.7152*G + 0.0722*B          # Rec.709 luma of scene-linear r'
Y_mapped = master_tonemap(Y)                 # single curve, no per-channel skew

# 2. Scale chroma by the luminance ratio (preserves hue exactly when skew=0)
ratio = Y_mapped / (Y + ε)
r'_c = r'_c * ratio                          # hue-preserving compression

# 3. Skew slider — blend toward per-channel result (controlled drift)
r_channel = per_channel_tonemap(r')          # the drift-prone path
r_out = r'_c * (1 - skew) + r_channel * skew # skew ∈ [0,1]
```

`master_tonemap` is the same H&D sigmoid as the density curve, but applied to luma in linear space (reusing `tonal.py::_sigmoid`). `skew = 0` gives the ACES-2.0 / AgX-class hue-locked result; `skew = 1` gives classic per-channel film drift. **This one slider spans "digital-clean" ↔ "filmic-drift" — a control Photoshop does not have** (per `PLAN_EASTWEST_COLOR_SUPREMACY.md:75`).

The luma math operates on **scene-linear** light, so the sRGB EOTF round-trip (inverse compand on entry, compand on exit — same `np.where` pair as `color_science.py:59-63,102-106`) wraps the whole stage. This is the F1 float advantage applied to dynamic-range compression: linear-space luma compression is physically correct; gamma-space compression (what uint8 paths do) is not.

---

## 4. Parametric Controls

All params registered as `ParamSpec` entries in `params.py` under a new `"film"` recipe block (mirrors how `"frequency"`, `"eyes"`, `"color_harmony"` are structured). Recipe key prefix: `film.*`. Conversion codes per the `params.py:20-46` registry.

### 4.1 Density block

| Param | recipe_key | Range | Default | Conversion | Notes |
|---|---|---|---|---|---|
| `film_enable` | `film.enable` | bool | False | `bool_flag` | Master gate. Off = byte-identical passthrough (round-trip neutrality QA gate). |
| `film_strength` | `film.strength` | 0.0–1.0 | 1.0 | `recipe_pct` | Global blend vs original. |
| `film_toe_r/g/b` | `film.toe.{r,g,b}` | 0.0–0.5 | 0.10 | `recipe_pct` | Per-dye toe depth. Symmetric default = neutral stock. |
| `film_shoulder_r/g/b` | `film.shoulder.{r,g,b}` | 0.0–0.5 | 0.10 | `recipe_pct` | Per-dye shoulder. |
| `film_midpoint` | `film.midpoint` | 0.0–1.0 | 0.50 | `recipe_direct` | Shared midpoint (per-channel midpoint is overkill; defer). |
| `film_gamma` | `film.gamma` | 0.5–2.5 | 1.0 | `recipe_direct` | Mid-range gamma. |

### 4.2 Crosstalk block

| Param | recipe_key | Range | Default | Notes |
|---|---|---|---|---|
| `film_crosstalk_cy_mg` | `film.crosstalk.cy_mg` | -0.15–+0.15 | +0.06 | Cyan→magenta leak (shadow warmth). |
| `film_crosstalk_cy_ye` | `film.crosstalk.cy_ye` | -0.15–+0.15 | +0.03 | Cyan→yellow leak. |
| `film_crosstalk_mg_ye` | `film.crosstalk.mg_ye` | -0.15–+0.15 | +0.02 | Magenta→yellow leak. |

The full 3×3 matrix is built from these three params + symmetry assumption + diagonal=1.0. Six params (full matrix) is overkill for the 2-week scope; three covers Portra/Velvia/Astia-class signatures. Document the reduction.

### 4.3 Tone-map block

| Param | recipe_key | Range | Default | Notes |
|---|---|---|---|---|
| `film_tonemap_strength` | `film.tonemap.strength` | 0.0–1.0 | 0.7 | Master tone-map blend. |
| `film_tonemap_toe` | `film.tonemap.toe` | 0.0–0.5 | 0.10 | Luma toe. |
| `film_tonemap_shoulder` | `film.tonemap.shoulder` | 0.0–0.5 | 0.15 | Luma shoulder (slightly deeper than density default — DR compression). |
| `film_skew` | `film.tonemap.skew` | 0.0–1.0 | 0.3 | **The key slider.** 0 = hue-locked (digital-clean), 1 = full per-channel drift (filmic). Default 0.3 = subtle drift. |

### 4.4 Density-domain grain + halation (coexist with existing modules, do not duplicate)

The existing `grain.py::apply_film_grain` (luminance-correlated, clumped, already float-native per F1/E2) and `grading.py::_add_halation` (red-leaking highlight bloom) are reused. C3 adds *one* hook: a `film_grain_density_coupled` bool that scales grain σ inversely with per-pixel density (shadows grainier, like real stock — `grain.py` already has a `luma_power` param that approximates this; C3's flag just sets it from the density field rather than a fixed value). No new grain engine. No new halation engine.

---

## 5. Integration Points

### 5.1 Pipeline placement

C3 runs inside `_stage_grade` (`engine.py:2567`), as a new block **after color transfer / WB / HSL but before the existing LUT application and grain**. Order:

```
_stage_grade:
  1. subject_aware_transfer      (existing, engine.py:2594)
  2. color_transfer              (existing, engine.py:2601)
  3. white_balance_lch           (existing, engine.py:2608)
  4. adjust_hsl_lch              (existing, engine.py:2617)
  5. ★ film_density_engine ★     (NEW — between HSL and tonal curve)
  6. apply_hd_curve              (existing, engine.py:2638 — tone-map; C3's
                                   tonemap block can supersede this when
                                   film_enable=True, see §5.3)
  7. grade() / grade_stack()     (existing, engine.py:2645-2663)
  8. split_tone / post-effects   (existing, downstream)
```

Rationale: the density model is a *color* operation (it shifts hue via crosstalk) and should run before the LUT/grade block so the grade operates on film-shaped color, not the reverse. The HSL block runs first because user HSL is an *intent* signal (deliberate hue targeting) that should survive the density model — the density crosstalk is a look, not a correction.

### 5.2 Coexistence with LUTs — not replacement

The LUT path (`_add_film_emulation`, `grading.py:1019`) stays. C3 is a **new grading block** with recipe key `"film": {...}` (per `PLAN_EASTWEST_COLOR_SUPREMACY.md:76`). A recipe uses one or the other:
- `{"film": {"enable": true, ...}}` → parametric engine, no LUT.
- `{"color_grade": "fuji_provia", "lut": "fuji_provia.cube"}` → existing LUT path.
- Both absent → no film block (current default).

When `film.enable=true` AND a `lut` key is present, the LUT is **skipped with a logged warning** (mutually exclusive — running both double-applies the film look). This is enforced in `_stage_grade` at the C3 block.

### 5.3 Interaction with `tonal.apply_hd_curve`

`apply_hd_curve` (`tonal.py:135`, called at `engine.py:2641`) is the existing H&D tone-map. When `film.enable=true` AND `film.tonemap.strength > 0`, the C3 master tone-map **replaces** `apply_hd_curve` (the C3 block sets `ctx.tonal_curve_strength = 0` locally, or the engine checks `film_enable` before calling `apply_hd_curve`). When `film.tonemap.strength = 0`, `apply_hd_curve` runs as-is. No silent double-tone-map.

### 5.4 GUI

New "🎬 Film Engine" accordion in the Color Grade tab (alongside the existing preset dropdown). Sliders for the §4 params, grouped: Density / Crosstalk / Tone-Map. A dropdown of **film stock presets** (Portra 400, Velvia 50, Astia 100, Classic Chrome, HP5) — each just sets the §4 params to a known-good combination (the re-expression work in §7.2 produces these). `film_enable` toggle at top.

---

## 6. Float32 Advantage (why this was blocked on F1)

| Operation | uint8 behavior | float32 behavior |
|---|---|---|
| `logE = log(r + ε)` | `r=0 → ε`, `log` spans [-6,0]; uint8 has 256 levels → shadows quantize to ~4 distinct logE values | full float range, smooth |
| `D = hd_curve(logE)` | curve applied via `cv2.LUT` on 256-entry table (`tonal.py:183`) — interpolates between 256 fixed points | analytic curve, no LUT, no quantization |
| `T = 10^(-D)` for `D∈[0,3]` | `T < 1/255` clips to 0 — **the entire deep-shadow density range is lost** | `T ∈ [0.001, 1.0]`, full precision |
| Crosstalk `M @ D` | matrix multiply on 8-bit-quantized densities — banding in shadow warmth | smooth |
| `r' = r * T` | multiply of two quantized values — cumulative banding | smooth |
| Luma tone-map in linear space | requires EOTF round-trip; uint8 EOTF loses ~1.5 stops in shadows | full linear range |

The F1/E2 work (`MASTER_PLAN.md` row 4, DONE 2026-07-07) established the pattern: `ensure_float` on entry, float-native LAB/LCH converters (`color_space.py:62-187`), `to_uint8` only at the final boundary. C3 follows the same pattern — density math is **the** textbook case where uint8 is not merely suboptimal but *broken* (shadow clipping). This is why the master plan explicitly gated C3 on F1 (`MASTER_PLAN.md:81`).

The one remaining uint8 site C3 would have hit — `tonal.apply_hd_curve`'s `cv2.LUT` (`tonal.py:183`) — is bypassed entirely: C3 applies the H&D curve analytically in float log-density space, never building a 256-entry LUT. (The float-path delta trick in `tonal.py:191-207` is a workaround for the uint8 LUT; C3 doesn't need it.)

---

## 7. Test Strategy

### 7.1 Unit tests (`tests/test_film.py`)

- **Round-trip neutrality:** `film_enable=False` → byte-identical to input (assert `np.array_equal`). `film_enable=True` with all-zero params (toe=shoulder=0, crosstalk=0, tonemap.strength=0, skew=0) → byte-identical. This is the master plan's `tonal.py` invariant (`tonal.py:59-60`) extended to the full block.
- **Monotonicity:** H&D curve is monotonically increasing for all param combinations in valid range (random-sample 1000 param sets, assert `np.all(np.diff(curve) >= 0)`). Mirrors `tonal.py:42-43`'s C⁰-continuity guarantee.
- **Crosstalk sign/direction:** positive `cy_mg` warms shadows (assert shadow-patch a* increases), cools highlights (assert highlight b* decreases). The physical prediction — if this fails, the matrix convention is wrong.
- **Skew endpoints:** `skew=0` preserves hue exactly (assert ΔH° < 0.5° on a chroma ramp); `skew=1` matches per-channel curve (assert equal to `_F_apply_rgb_curves` output). `skew=0.5` is between.
- **Float vs uint8 parity:** at `film_strength=1.0`, float32 path and uint8 path differ only by quantization (max Δ < 2 levels). Mirrors the F1/E2 dtype-parity tests.
- **Determinism:** grain (when density-coupled) is seeded → identical output across runs. Reuses `grain.py`'s existing seed contract.
- **Mutual exclusion:** `film.enable=true` + `lut` present → LUT skipped, warning logged (assert via `caplog`).
- **Tonal-curve suppression:** `film.enable=true` + `film.tonemap.strength>0` → `apply_hd_curve` not called (spy on `tonal.apply_hd_curve`).

### 7.2 Golden-output harness — LUT re-expression (the acceptance test)

The master plan's C3 acceptance bar (`MASTER_PLAN.md:138`, `PLAN_EASTWEST_COLOR_SUPREMACY.md:129`): **parametric re-expression of one Fuji preset within ΔE ~3 of its LUT on the corpus.**

Procedure:
1. Pick one shipped Fuji LUT (e.g. `presets/luts/fuji_provia.cube`).
2. Build a parameter-optimization script (`scripts/film/fit_lut.py`): given a LUT and a corpus of test images (the A1 corpus when available; until then, the existing `tests/fixtures/` images + a grayscale ramp + a Macbeth chart), minimize mean ΔE between `LUT(img)` and `film_engine(img; params)` over the §4 params. Scipy `minimize` with bounds; ~12 params, convex-ish (the density model is smooth).
3. Assert mean ΔE < 3.0 on a held-out test image. Assert max ΔE < 6.0 (no single-patch blowup).
4. The fitted params become the "Provia" stock preset in the GUI dropdown. Repeat for 2 more stocks (Astia, Classic Chrome) to prove the model generalizes — not just one overfit.

This is the test that *proves the model*. If the model can't re-express a real LUT within ΔE 3, the model is wrong, not the LUT.

### 7.3 Visual QA gates (mandatory — C3 is Visual-Critical, touches `grading.py`)

Per `AGENTS.md` and `docs/VISUAL_QA.md:38`, `grading.py` changes require: **No Color Drift, No Highlight Clipping, No Shadow Crushing, No Halo.**

| Gate | C3-specific assertion |
|---|---|
| No Color Drift | On a neutral gray ramp, `film_skew=0` keeps ΔE < 1.0 across all patches. `film_skew=1` drifts as intended (document the drift, don't fail it). |
| No Highlight Clipping | `film_shoulder > 0` must not push >0.5% pixels to 255. The shoulder *is* the anti-clip mechanism — test it doesn't backfire. |
| No Shadow Crushing | `film_toe > 0` must not push >0.5% pixels to 0. The toe lifts, not crushes — verify. |
| No Halo | Halation (reused from `grading.py:_add_halation`) energy bounded — assert halation layer max < 0.3 × source luminance (the existing `_add_halation` intensity cap). |

Plus the C3-specific gate from `PLAN_EASTWEST_COLOR_SUPREMACY.md:78`:
- **Round-trip neutrality at all-zero params:** byte-identical (covered in §7.1, re-asserted on a real photo in the visual QA run).
- **Grain determinism:** seeded, identical across runs (§7.1).
- **Halation energy bound:** (above).

### 7.4 Performance

Per `AGENTS.md` performance budget: runtime must not increase >10% on 1080p/4K; memory peak must not increase >15%. Run `scripts/bench/benchmark.py` before/after. The density model is vectorized NumPy (log, matrix multiply, 10^x) — no Python loops. The crosstalk `M @ D` is a single `(H×W×3) @ (3×3)` reshape-matmul. Expected cost: ~1.5× a LUT application (the log/exp are the expensive parts). If over budget, the `max_compute_dim` downsample pattern from `_large_sigma_blur` (`grading.py:100`) applies to the halation blur (already does) — the density math itself is per-pixel and can't be downsampled without changing the result.

---

## 8. Implementation Order (2-week breakdown)

| Day | Work |
|---|---|
| 1–3 | `retouch/film.py`: density block (§2) + unit tests §7.1 (round-trip, monotonicity, crosstalk sign). Float32 only on entry/exit. |
| 4–5 | Hue-preserving tone-map (§3) + skew tests. Wire into `_stage_grade` (§5.1) behind `film_enable` gate. |
| 6–7 | `params.py` ParamSpecs (§4) + `recipes.py` `"film"` block + GUI accordion. Dead-key guard test (master plan ground rule 1, `MASTER_PLAN.md:134`). |
| 8–9 | `scripts/film/fit_lut.py` — LUT re-expression harness (§7.2). Fit Provia. |
| 10 | Fit Astia + Classic Chrome. Assert ΔE < 3 on all three. Ship as GUI stock presets. |
| 11 | Visual QA gates (§7.3) on real photos (the `DSCF8007.jpg` reference + A1 corpus if available). Document PASS/FAIL/IMPROVED per gate. |
| 12 | `benchmark.py` perf check (§7.4). Buffer. |

---

## 9. Open Questions / Risks

1. **Crosstalk matrix reduction (3 vs 6 params).** The full 3×3 has 6 off-diagonal terms. Reducing to 3 (cyan-row leaks, assuming symmetry) may not fit all stocks. If the ΔE-3 fit fails on Velvia (high-saturation stock, strongest crosstalk), expand to 6. Decision at day 9, gated on fit results.
2. **Linear-space requirement.** The density model is physically correct only in scene-linear light. The pipeline currently carries gamma-encoded float32 [0,1] (sRGB). C3 must EOTF round-trip at the block boundary — adds 2 `np.where` power ops per call. Verify this doesn't blow the 10% perf budget. If it does, approximate with a 2.2 gamma (cheaper, `relight.py`'s bloom convention) and document the approximation.
3. **F6 dependency.** `MASTER_PLAN.md:87` notes F6 (look-from-reference) depends on C3 for a better fitting target. C3's param space is the fitting target — confirm the §4 params are expressive enough that F6's optimizer can converge. Out of scope for C3 itself, but the param design should not preclude it.
4. **Existing `apply_hd_curve` overlap.** The `tonal.py` H&D curve and C3's density curve use the same sigmoid primitives. Post-C3, `apply_hd_curve` is redundant when `film.enable=true`. Do not remove it (recipes depend on `tonal_curve_strength`); just suppress it when C3's tonemap is active (§5.3). A future cleanup pass could fold `tonal.py` into `film.py`.

---

## 10. Verification Checklist (run before shipping)

Per `AGENTS.md` workflow:
1. `python3 -m py_compile retouch/film.py`
2. `python3 -m pytest tests/test_film.py tests/test_grading.py -v`
3. `python3 scripts/bench/benchmark.py` (perf regression)
4. Visual QA gates per §7.3 (Visual-Critical — `grading.py` touched)
5. LUT re-expression ΔE < 3 per §7.2 (the acceptance bar)
6. Mark `MASTER_PLAN.md` row 17 `✅ DONE <date>` with receipt (files touched, test count, ΔE result) per `AGENTS.md` "Mark work done in the plan docs."
