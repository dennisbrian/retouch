# Retouch Project — Code Quality Audit Report

Generated: 2026-06-22

## Table of Contents

1. [Dead Code](#1-dead-code)
2. [Unused Parameters](#2-unused-parameters)
3. [Duplicate Logic](#3-duplicate-logic)
4. [Architectural Inconsistencies](#4-architectural-inconsistencies)
5. [Documentation Audit](#5-documentation-audit)
6. [Test Coverage Gaps](#6-test-coverage-gaps)

---

## 1. Dead Code

### 1.1 Unused Imports

| File | Line | Import | Notes |
|------|------|--------|-------|
| `retouch/engine.py` | 80 | `field` from `dataclasses` | `field()` never called; only `@dataclass` decorator used |
| `retouch/detection.py` | 13 | `import sys` | Not referenced anywhere in file |
| `retouch/gui.py` | 9 | `BytesIO` from `io` | Not referenced anywhere in file |
| `retouch/grading.py` | 8 | `Union` from `typing` | File uses `X | Y` syntax via `from __future__ import annotations` |
| `retouch/perf_optimizations.py` | 21 | `field` from `dataclasses` | Never called |

### 1.2 Dead Constants

| File | Line | Constant | Notes |
|------|------|----------|-------|
| `retouch/frequency.py` | 34 | `BILATERAL_D_FACTOR = 0.006` | Never referenced anywhere |
| `retouch/frequency.py` | 35 | `BILATERAL_D_MIN = 9` | Never referenced anywhere |

### 1.3 Dead Functions (Production — Never Called)

| File | Line | Function | Call Chain |
|------|------|----------|------------|
| `retouch/skin.py` | 18 | `SkinProcessor.smooth()` | Deprecated no-op, emits `DeprecationWarning`, returns input unchanged |
| `retouch/skin.py` | 295 | `adaptive_smooth()` | Module-level, never imported or called |
| `retouch/frequency.py` | 212 | `combine_adaptive()` | Only called from `skin.adaptive_smooth()` — chain is entirely dead |
| `retouch/relight.py` | 166 | `_retinex_normalize()` | Only called from `retinex_msr()` |
| `retouch/relight.py` | 174 | `retinex_msr()` | Never imported or called from any production code |
| `retouch/relight.py` | 206 | `retinex_ssr()` | Never imported or called from any production code |
| `retouch/perf_optimizations.py` | 321 | `detect_faces_downscaled()` | Never imported or called |
| `retouch/perf_optimizations.py` | 442 | `init_onnx_session()` | Never imported or called |
| `retouch/perf_optimizations.py` | 459 | `init_mediapipe_with_gpu()` | Never imported or called |
| `retouch/perf_optimizations.py` | 540 | `apply_tonal_lut()` | Never imported or called |
| `retouch/grading.py` | 641 | `detect_color_patches()` | Only called from tests |
| `retouch/grading.py` | 680 | `correct_color_patches()` | Only called from tests |

### 1.4 Dead Methods

| File | Line | Method | Notes |
|------|------|--------|-------|
| `retouch/skin.py` | 18 | `SkinProcessor.smooth()` | Deprecated no-op |
| `retouch/grading.py` | 545 | `ColorGrader.color_transfer_hist()` | Only called from tests |

### 1.5 Dead Classes

| File | Line | Class | Notes |
|------|------|-------|-------|
| `retouch/perf_optimizations.py` | 56 | `RegionMasks` | Never imported or instantiated |
| `retouch/perf_optimizations.py` | 86 | `FaceData` | Shadows `detection.FaceData`; used only by dead function `detect_faces_downscaled()` |

### 1.6 Dead Computation

| File | Line | Variable | Notes |
|------|------|----------|-------|
| `retouch/batch_processor.py` | 131-133 | `face_width` | Computed and stored in cache dict, but never read back when cached info is retrieved |

### 1.7 Buggy (Broken, Not Dead)

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `retouch/cli.py` | 130 | `RetouchEngine._adjust_contrast(...)` called as classmethod/static method | `_adjust_contrast` is a module-level function at `engine.py:1884`, not a class method. Will raise `AttributeError` at runtime. Replace with `from retouch.engine import _adjust_contrast` |

---

## 2. Unused Parameters

### 2.1 Stage Method — Unused Parameter

| File | Function | Line | Parameter | Notes |
|------|----------|------|-----------|-------|
| `retouch/engine.py` | `_stage_grade()` | 1694 | `acc_skin_hair: np.ndarray` | Passed in at call site (line 1260) but **never referenced** in method body. Only `acc_skin`, `acc_lips`, and `person_mask` are used. |

### 2.2 ProcessingContext — Set but Never Read by Stages

| File | Field | Line | Set By | Consumed By |
|------|-------|------|--------|-------------|
| `retouch/engine.py` | `glow: float = 0.0` | 194 | `build_context()` line 449, `process()` line 848 | **No stage method** — `ctx.glow` is never read. The `glow_mask` variable in `_stage_grade` is a computed mask, not driven by this parameter. |
| `retouch/engine.py` | `vignette: float = 0.0` | 197 | `build_context()` line 450, `process()` line 849 | **No stage method** — `ctx.vignette` is never read. Vignette is only available via color-grading presets (`ColorGrader._add_vignette`). |
| `retouch/engine.py` | `active_recipe: str = "natural"` | 208 | `build_context()` line 469 | Metadata only — stored for `ProcessingResult.params` inspection, never read by stage logic. |

**Impact**: The GUI glow and vignette sliders, as well as `--glow`/`--vignette` if exposed via CLI, have zero effect on output.

---

## 3. Duplicate Logic

### 3.1 CRITICAL — Recipe Dict → Params Mapping (3 Copies)

`recipe_defaults()` in `gui.py:45-115` and `build_context()` in `engine.py:303-473` traverse the identical `RECIPES` dict structure with nearly identical logic. Every recipe key is parsed independently in both places:

| Concept | gui.py | engine.py |
|---------|--------|-----------|
| Smooth → `frequency.smooth` | Line 48 | Line 321 |
| Mid reduction | Line 49 | Line 326 |
| Texture opacity | Line 50 | Line 327 |
| Pore synthesis | Line 51 | Line 328 |
| Whitening → `skin.rosy`/`skin.porcelain` | Lines 53, 87 | Lines 323-324, 331-334 |
| Equalize | Line 54 | Line 322 |
| Relight | Lines 55-58 | Lines 340-344 |
| Relight azimuth/elevation | Lines 59-60 | Lines 345-346 |
| Eye enhance | Line 61 | Line 350 |
| Lip enhance | Line 62 | Line 357 |
| Lip tint | Line 63 | Line 358 |
| Blush | Line 64 | Line 403 |
| Teeth whiten | Line 65 | Line 353 |
| Hair shine | Line 66 | Line 361 |
| Dodge burn | Lines 67-70 | Lines 335-339 |
| Specular bloom | Line 71 | Lines 329-330 |
| Bloom opacity | Line 72 | Line 397 |
| Bloom threshold/softness | Lines 73-74 | Lines 398-399 |
| Contrast | Line 75 | Line 376 |
| Brightness | Line 76 | Line 377 |
| Highlights, Shadows, Whites, Blacks | Lines 77-80 | Lines 378-381 |
| Nose/Under-eye blush, Costume lift | Lines 81-83 | Lines 405-407 |
| Blemish | Line 85 | Line 325 |
| Dark circles | Line 86 | Line 352 |
| Whiten tone | Line 87 | Lines 331-334 |
| Lip finish | Line 89 | Line 404 |
| Slimming | Line 90 | Line 402 |
| Impact | Line 91 | Line 396 |
| Clarity, Vibrance, Saturation | Lines 93-95 | Lines 382-384 |
| Glow, Vignette, Sharpen, Radius | Lines 96-99 | Lines 391-394 |
| Subject separation | Line 100 | Line 395 |
| Specular bloom tone | Line 101 | Line 330 |
| Color grade | Line 102 | Line 365 |
| Grade intensity | Line 103 | Line 366 |
| Chromatic aberration, Grain, Halation, LUT | Lines 104-107 | (via overrides) |
| Split toning (6 params) | Lines 109-114 | Lines 385-390 |

**Additionally**, `cli.py` has a third miniature version at lines 114-120 handling only `color_grade`, `grade_intensity`, and `impact`.

**Recommendation**: Create a single recipe-to-params translation function in `recipes.py` or `utils.py` that both `gui.py` and `engine.py` call.

### 3.2 HIGH — Export Format/Resolution Maps (4 Definitions)

| Location | Lines | Content |
|----------|-------|---------|
| `gui.py` | 33-35 | `EXPORT_RES_MAP` dict + `EXT_MAP` dict |
| `gui.py` | 1357-1358 | Dropdown choices (UI strings) |
| `gui.py` | 1511-1513 | Duplicate dropdown choices for batch tab |
| `batch_processor.py` | 340-347, 356 | Identical inline dicts |
| `io.py` | 70-76 | `encode_write_params()` — format-to-OpenCV-param mapping |
| `gui.py` | 561-564 | Inline OpenCV params (bypasses `encode_write_params`) |

### 3.3 MEDIUM — Screen Blend Formula (5 Copies)

`255.0 - ((255.0 - a) * (255.0 - b) / 255.0)` hand-coded in:

| File | Line | Context |
|------|------|---------|
| `retouch/utils.py` | 323 | `apply_global_bloom()` |
| `retouch/grading.py` | 246 | `_add_glow()` |
| `retouch/grading.py` | 390 | `_add_halation()` |
| `retouch/grading.py` | 580 | `_add_haze()` |
| `retouch/grading.py` | 626 | `_add_sparkles()` |

### 3.4 MEDIUM — Mask Normalization Pattern (~24 Inline Copies)

The pattern `if x.max() > 1.0: x /= 255.0` appears inline across 10 files, despite a dedicated `_norm_mask()` helper existing in `engine.py:480-487` (which is private with `_` prefix, so other modules cannot use it).

Affected files: `skin.py`, `relight.py`, `makeup.py`, `hair.py`, `utils.py`, `engine.py`, `grading.py`, `parsing.py`, `batch_processor.py`, `style.py`.

### 3.5 MEDIUM — ROI Padding Calculation (Duplicated Within engine.py)

Identical face-crop padding computation at:
- `engine.py._stage_per_face()` lines 1366-1375
- `engine.py._process_one_face()` lines 1515-1526

### 3.6 MEDIUM — Post-Effects Assembly (Duplicated Within engine.py)

The block assembling `post_effects` from `chromatic_aberration`, `halation`, `grain`, `lut` appears in both:
- `_stage_grade()` lines 1730-1738
- `_no_face_fallback()` lines 1297-1305

### 3.7 MEDIUM — Person Mask Squeeze Pattern (5 Copies)

`if pm.ndim == 3: pm = pm.squeeze(-1)` appears at:
- `engine.py:1083, 1634, 1719`
- `hair.py:64`
- `grading.py:594`

### 3.8 MEDIUM — Vibrance (2 Different Implementations)

- `utils.py:137-156` — `vibrance(img_bgr, mask, strength)` — multi-channel HSV with mask, raw float strength
- `engine.py:1859-1871` — `_adjust_vibrance(img, vibrance)` — same formula + skin-hue protection mask, `vibrance/100` scaling

### 3.9 LOW — `_adjust_saturation` Name Collision (2 Different Algorithms)

- `engine.py:1874`: Uniform HSV multiplier: `factor = 1.0 + v / 100`
- `grading.py:278`: Vibrance-style: `factor = 1.0 + boost * (1.0 - s / 255.0)`

### 3.10 LOW — Face Width Estimation (3 Approaches)

| File | Line | Method |
|------|------|--------|
| `blemish.py` | 29-33 | From `skin_mask` x-extent |
| `lips.py` | 47-52 | From `lip_mask` x-extent × 3.3 |
| `frequency.py` | 119 | `APPROX_FACE_WIDTH_RATIO = 0.4` as fallback |

### 3.11 LOW — Other Duplications

- **Lip tint names** defined in `lips.py:15-23` (BGR dict) and `gui.py:683` (string list)
- **"none" → None conversion** repeated 3× in `gui.py:403-405`
- **`WHITEN_TONE_CHOICES` and `SPECULAR_BLOOM_TONE_CHOICES`** identical (`gui.py:29, 31`)
- **`log_crash` call pattern** duplicated at 5 call sites across `gui.py`, `cli.py`, `batch_processor.py`

---

## 4. Architectural Inconsistencies

### 4.1 CRITICAL — ProcessingContext Dead Ends

| Field | Set By | Never Consumed By |
|-------|--------|-------------------|
| `glow` | `build_context()`, `process()` overrides | **No stage method** — `ctx.glow` grep returns 0 hits |
| `vignette` | `build_context()`, `process()` overrides | **No stage method** — `ctx.vignette` grep returns 0 hits |

Both are fully wired: `process()` → `overrides dict` → `build_context()` → `ProcessingContext` → **nowhere**. They fall off the end. GUI sliders and any CLI flags for these are non-functional.

### 4.2 CRITICAL — `eyes.catchlight` Defined in Every Recipe, Never Read

Every recipe defines `"catchlight": <value>` inside `"eyes": { ... }`. But:
- `build_context()` at `engine.py:349-353` only reads `iris`, `whites`, and `dark_circles`
- `ProcessingContext` has no `catchlight` field
- `EyeEnhancer._enhance_catchlights()` is always called with the same `eye_enhance` strength — the recipe `catchlight` value is **completely ignored**

### 4.3 CRITICAL — Grain/Halation GUI Range Mismatch

| Parameter | GUI Slider | Engine Expects | Issue |
|-----------|-----------|----------------|-------|
| `grain` | 0-100 | ~0.0-0.2 | At mid-range (50), `_add_grain()` generates noise with stddev `255 × 50 = 12,750` — completely destructive |
| `halation` | 0-100 | ~0.0-1.0 | At mid-range (50), `_add_halation()` uses `intensity=50` — ~166× expected value |

**Root cause**: `process_image()` passes raw slider values without scaling:
```python
grain=grain if grain > 0 else None       # should be grain / 500.0
halation=halation if halation > 0 else None  # should be halation / 100.0
```

### 4.4 HIGH — Circular Dependencies

1. **`engine.py:100` → `parsing.py:16` → `perf_optimizations.py:192-193` → `engine.py`**
   - `perf_optimizations.py` uses deferred imports inside a worker function to avoid import-time crash
   - The circular chain exists: `engine → parsing → perf_optimizations → engine`

2. **`engine.py:100` → `style.py:66,225` → `engine.py`**
   - `StyleAnalyzer.__init__()` and `StyleApplier.__init__()` import `RetouchEngine` lazily
   - Again a deferred-import workaround for circular design

### 4.5 HIGH — 7-Way Parameter Sync

Adding one new parameter requires edits in **7 locations**:

1. `engine.py:process()` signature
2. `engine.py:overrides` dict
3. `gui.py:PROCESS_INPUT_KEYS` list
4. `gui.py:process_image()` → `engine.process()` kwargs
5. `gui.py:recipe_defaults()` (for slider defaults from recipe)
6. `cli.py:build_params()` (for CLI exposure)
7. `gui.py:_process_inputs` event binding list

### 4.6 HIGH — Triple Default Maintenance

Default values live in three places that must agree:

| Source | File | Lines |
|--------|------|-------|
| `ProcessingContext` dataclass | `engine.py` | 119-217 |
| `RECIPES["natural"]` | `recipes.py` | 7-25 |
| `recipe_defaults()` | `gui.py` | 45-115 |

### 4.7 HIGH — Silent Error Swallowing in `parsing.py`

```python
# parsing.py:218, 308
except Exception:
    pass
```

BiSeNet parsing errors are silently discarded. No logging, no warning, no fallback indicator. Errors in the ONNX inference path go completely undetected.

### 4.8 MEDIUM — Module Pattern Inconsistency

| Module | Pattern | All Others |
|--------|---------|------------|
| `frequency.py` | Module-level functions (`separate()`, `combine()`) | Class-based (`SkinProcessor`, `EyeEnhancer`, etc.) |
| `grading.py` | Class + module-level mixed | Class methods only |

### 4.9 MEDIUM — `perf_optimizations.FaceData` Name Collision

- `detection.py:38` — `FaceData(bbox, landmarks, ied, confidence)` — used by the main pipeline
- `perf_optimizations.py:86` — `FaceData(pixel_landmarks, original_width, original_height)` — completely different fields, used only by dead function `detect_faces_downscaled()`

Same name, incompatible shapes. Potential for confusing import errors.

### 4.10 MEDIUM — Recipe Key → ProcessingContext Field Mismatches

| Recipe Key | ProcessingContext Field | Location |
|------------|------------------------|----------|
| `relight_strength` | `relight` | recipes.py:264 → engine.py:136 |
| `light_azimuth` | `relight_azimuth` | recipes.py:265 → engine.py:137 |
| `light_elevation` | `relight_elevation` | recipes.py:266 → engine.py:138 |
| `color_harmony.preset` | `color_grade` | recipes.py → engine.py:171 |
| `color_harmony.amount` | `grade_intensity` | recipes.py → engine.py:172 |
| `finish.impact` | `impact` | recipes.py:199 → engine.py:203 |
| `dodge_burn` (float or dict) | `dodge_burn` | recipes.py:262 → engine.py:132 |

### 4.11 MEDIUM — `eyes.whites` Drives Two Independent Effects

```python
# engine.py build_context() lines 350-353
r_eye = _pct(eyes.get("iris", eyes.get("whites", 0.0)))    # eye_enhance
r_teeth = _pct(eyes.get("whites", 0.0))                      # teeth_whiten
```

The `"whites"` sub-key serves double duty: fallback for eye enhancement AND sole driver for teeth whitening. Changing teeth whitening intensity unavoidably changes eye enhancement when no explicit `iris` key exists.

### 4.12 MEDIUM — CLI `--halation` Type Mismatch

- CLI declares `--halation` as `type=float` with help "Film halation bleed intensity (0.0 - 1.0)"
- `build_params()` at line 228 converts it to a dict: `{"threshold": 210, "radius": 21, "intensity": args.halation}`
- Engine type hint is `halation: Optional[float] = None` — but receives a dict

### 4.13 LOW — CLI `--color-ref-strength` → Engine `color_transfer_intensity`

CLI flag name and engine parameter name have **no textual relationship**. User reading `--color-ref-strength` cannot guess the engine parameter is `color_transfer_intensity`.

### 4.14 LOW — `import sys; sys.path.insert(0, ...)` in `cli.py:14` and `gui.py:16`

Both entry points use filesystem path injection to import the `retouch` package instead of relying on proper installation. Works during development but is fragile.

---

## 5. Documentation Audit

### 5.1 `API.md`

| Issue | Details |
|-------|---------|
| `min_confidence` default wrong | Documents `0.5`, actual code uses `0.4` |
| Missing `face_contexts` | Parameter exists at `engine.py:895` but not documented |
| Missing `nose_blush` | Boolean flag at `engine.py:870` not documented |
| Missing `under_eye_blush` | Boolean flag at `engine.py:871` not documented |
| Missing `white_costume_lift` | Boolean flag at `engine.py:872` not documented |
| Missing `dark_circles` | Not individually listed in parameter table |
| `ProcessingResult.face_contexts` | Return attribute not documented |

### 5.2 `ARCHITECTURE.md`

- Generally accurate — pipeline stages, module responsibilities, data flow all match code
- Missing: `ProcessingContext` split toning fields (`shadow_hue/sat`, etc.)
- Module size estimates approximate and may drift

### 5.3 `BATCH_GUIDE.md`

- Accurate for documented features
- Missing 8 newer recipe names: `anime_cinematic_v1`, `anime_cinematic_soft`, `anime_cinematic_action`, `anime_crystal_void`, `anime_cinematic_fantasy`, `fuji_porcelain`, `blue_dream`, `xhs_ultrasoft`

### 5.4 `RECIPE_GUIDE.md`

- Core field reference accurate
- **Missing 20+ recipe fields**: `specular_bloom_tone`, `relight_strength`, `light_azimuth`, `light_elevation`, `background_blur`, `background_desaturation`, `light_wrap`, `blue_shadow_grade`, `cyan_midtone_grade`, `subject_sharpen`, `matte_black`, `shadow_hue`, `shadow_sat`, `midtone_hue`, `midtone_sat`, `highlight_hue`, `highlight_sat`, `pore_synthesis`, `eye_enhance`
- `color_harmony.preset` valid values not listed
- Conversion table incomplete (missing 12+ field conversions)

### 5.5 `README.md`

- Installation, quick start, test instructions correct
- Model URLs correct
- Missing link to GUI documentation in the docs section

---

## 6. Test Coverage Gaps

### 6.1 Modules with ZERO Test Coverage

| Module | Lines | Missing |
|--------|-------|---------|
| `retouch/perf_optimizations.py` | 609 | Entire file — all classes, functions, JIT kernels |
| `gui.py` | 1736 | Entire file — no test exists |
| `desktop.py` | 32 | Entire file |

### 6.2 Engine — Critical Untested Methods

| Method | Line | Why Matters |
|--------|------|-------------|
| `_process_with_proxy()` | 1114 | Proxy resolution for high-res images — core performance optimization |
| `_upscale_core_result()` | 1144 | Upscaling masks/results — critical for output quality |
| `_run_core_pipeline()` | 1164 | Main orchestrator, only tested indirectly through full `process()` |
| `_process_face_core()` | 532 | Standalone picklable function for parallel processing — has no direct test |
| `_stage_subject_separation()` | 1615 | Subject/background separation has no dedicated test |
| `_apply_white_costume_lift()` | 1813 | Costume lift feature — only build_context tested |

### 6.3 Engine — Untested Integration Paths

| Feature | Where | Missing Test |
|---------|-------|-------------|
| Split toning params | `process()` → `build_context()` → `_stage_grade()` | No integration test passes `shadow_hue/sat` etc. to `process()` |
| Post-effects pass-through | `chromatic_aberration`, `halation`, `grain`, `lut` to `process()` | Only internal grading methods tested, not the engine parameter pipeline |
| `debug_dir` | `process()` line 1069-1096 | No test checks that mask files are written to disk |
| `color_ref` ndarray | `process()` with `color_ref=` | No integration test passes a reference image array |
| `style_profile` | `process()` with `style_profile=` | No integration test verifies style profile application |
| `style_ref` | `process()` with `style_ref=` | No integration test passes a style reference image |
| `face_contexts` caching | `process()` with `face_contexts=` | No test exercises the caching code path |

### 6.4 Other Modules — Test Gaps

| Module | Untested |
|--------|----------|
| `style.py` | `StyleAnalyzer.extract()` (requires face images), `StyleApplier.apply()`, `subject_aware_transfer()` |
| `detection.py` | `FaceDetector.detect()`, `segment_person()`, `close()`, `_remap_landmarks()` — only dataclasses tested |
| `parsing.py` | ONNX path (`FaceParser.parse()` with BiSeNet) — only landmark fallback path tested |
| `grading.py` | `register_presets_dir()`, `load_preset()`, `load_all_presets()` — not tested at module level |
| `io.py` | `copy_exif()` — zero coverage |

### 6.5 Integration Test Gaps

- **No multi-face processing test** — no test exercises engine with 2+ synthetic faces
- **No proxy pipeline test** — no test creates a >2048px image to trigger `_process_with_proxy()`
- **No real-face test** — `test_integration.py:TestWithRealImage` requires external images and is skipped when absent
- **No GUI tests at all** — `gui.py` has zero test coverage
