# Plan: "Moonlight Porcelain" look — match the 爻一爻 blue cinematic cosplay edit

## Context

Stages 0–B are done (`anime_v2` recipe + skin flatten/quantize/unify/glow primitives, commit `00449e5`). The user shared two reference images (blue/white HSR-style cosplay, moody indoor set) and wants the engine to reproduce that finished editing style on unedited photos — face retouch + full color grade — with no artifacts.

**Style analysis of the references:**
- **Grade:** low-key cinematic. Shadows pushed blue-violet (~hue 240–250, strong sat), midtones cool cyan-blue (~205–215), highlights near-neutral with a slight cool cast. Deep (not matte) blacks with a blue tint, crushed curve toe, high contrast. Cool white balance overall.
- **Selective color:** warm accents survive the blue grade — cream/pink/red flowers stay saturated. This is per-hue treatment, not a global tint.
- **Skin:** porcelain-pale, flat, unified anime skin (exactly what the Stage A/B primitives do), with rosy lips/under-eye blush retained. Image 2 lets skin go cooler/bluer than image 1.
- **Atmosphere:** haze/bloom around light sources, subtle orton glow, halation; background darker + hazier than subject; subject stays sharp (hair strands, jewelry, costume detail).
- **Finish:** moderate vignette, subject/background exposure separation, sharp subject, fine grain.

**Key finding from exploration:** nearly everything needed already exists. The grading preset JSON system (`retouch/grading.py:grade()`) supports `curves`, `white_balance`, `calibration`, `split_tone_three_way`, **`hsl_adjustments` (per-hue sat/lum — this is how we keep the flowers warm)**, `haze`, `glow`, `orton_glow`, `vignette`, `grain` — so the whole grade is expressible as **a new preset JSON + a new recipe dict, zero engine code**. New presets auto-appear via `list_available_presets()`; new recipes auto-appear in the GUI dropdown via `RECIPES`.

---

## 🐛 Bugs found — REPORT ONLY, not fixed in this plan (per user instruction)

1. **`anime_crystal_void` has 7 dead recipe keys.** `background_blur`, `background_desaturation`, `light_wrap`, `blue_shadow_grade`, `cyan_midtone_grade`, `subject_sharpen`, `matte_black` (`retouch/recipes.py:378-384`) appear **nowhere else in the codebase**. Recipe→context translation is data-driven from `PROCESSING_PARAMS` ParamSpecs (`retouch/params.py`), and unknown keys are **silently dropped** — so the `anime_crystal_void` recipe does not do most of what it advertises. This is the same "dead key" failure mode as the `relight` bug fixed in Stage A. *(Ironically these dead params — background darken/desat, light wrap, blue shadow grade — are exactly the features this look would benefit from. Wiring them is listed as an optional stage below; your call whether to include it.)*
2. **Systemic risk, suggestion only:** there is no guard that every recipe key resolves to a real ParamSpec / known nested key. A small validation test would have caught both this and the Stage-A relight bug. Not planned; flagging it.

---

## Step 1 — New grading preset: `presets/moonlight_porcelain.json`

Modeled on `presets/blue_dream.json` but darker/moodier and with warm-accent preservation. Proposed starting values (to be tuned visually in Step 4):

```json
{
  "description": "Moody blue cinematic cosplay — indigo shadows, cyan mids, porcelain subject, warm flower accents preserved",
  "curves": { "L": [[0, 4], [42, 30], [128, 122], [210, 214], [255, 250]] },
  "white_balance": { "R": 0.93, "G": 1.00, "B": 1.08 },
  "calibration": {
    "red":   {"hue": 0.0,  "sat": 6.0},
    "green": {"hue": 15.0, "sat": -8.0},
    "blue":  {"hue": -8.0, "sat": 14.0}
  },
  "split_tone_three_way": {
    "shadows":    {"hue": 242.0, "sat": 28.0},
    "midtones":   {"hue": 207.0, "sat": 13.0},
    "highlights": {"hue": 210.0, "sat": 5.0},
    "balance": -15.0
  },
  "hsl_adjustments": {
    "saturation": {"red": 10, "magenta": 8, "orange": -4, "yellow": -12, "green": -10},
    "luminance":  {"blue": -6, "yellow": -5}
  },
  "haze": 0.06,
  "glow": 0.10,
  "orton_glow": 0.05,
  "vignette": 0.10,
  "grain": 0.03
}
```

Rationale: `split_tone_three_way` gives the indigo-shadow / cyan-mid / near-neutral-highlight structure; `hsl_adjustments` boosts red/magenta so flowers punch through the cool grade while desaturating yellow/green (nothing in the refs is yellow/green); `haze` + `orton_glow` + `glow` build the atmospheric smoke-glow; the L-curve toe crushes the background.

## Step 2 — New recipe: `moonlight_porcelain` in `retouch/recipes.py`

Extends `anime_v2` (keeps the whole anime skin stack: flatten 0.55, quantize 0.40, unify 0.50, relight 0.42, glow 0.20). Overrides — concrete starting values:

```python
"moonlight_porcelain": {
    "extends": "anime_v2",
    "color_harmony": {"preset": "moonlight_porcelain", "amount": 0.65},
    # skin: porcelain direction, keep anime stack from parent
    "whiten_tone": "porcelain",          # (verify exact recipe key: GUI uses whiten_tone dropdown)
    "skin": {"glow": 0.25},
    "skin_protect": 0.45,                # skin stays porcelain, not fully blue (ref image 1)
    # tone
    "brightness": 4.0,
    "contrast": 16.0,
    "highlights": -12.0,
    "whites": 6.0,
    "blacks": 8.0,
    "saturation": -6.0,
    "vibrance": 8.0,
    # split toning — OVERRIDE PARENT: anime_cinematic_v1 sets highlight_hue 320
    # (magenta) which fights the cool look
    "shadow_hue": 242.0, "shadow_sat": 22.0,
    "midtone_hue": 207.0, "midtone_sat": 8.0,
    "highlight_hue": 210.0, "highlight_sat": 4.0,
    # atmosphere / finish
    "subject_separation": 35.0,
    "bloom": {"opacity": 0.20, "threshold": 185.0},
    "glow": 12.0,
    "vignette": 12.0,
    "sharpen": 22.0,
    "chromatic_aberration": 2.0,         # parent's 4.0 fringes on silver jewelry
    "halation": 0.08,
    "grain": 0.0,                        # grain comes from the preset, avoid doubling
    # face accents (rosy lips/under-eye survive the cool grade)
    "blush": 25.0,
    "lips": {"tint": "rose", "gloss": 0.30},
    "hair": {"shine": 0.85},
    "eyes": {"iris": 0.30, "catchlight": 0.35},
},
```

Plus an optional cooler variant for the image-2 look (3 lines):

```python
"moonlight_cool": {
    "extends": "moonlight_porcelain",
    "skin_protect": 0.15,                # let skin take the blue ambient
    "midtone_sat": 14.0,
    "color_harmony": {"preset": "moonlight_porcelain", "amount": 0.80},
},
```

**Implementation notes:**
- Verify each override key against `PROCESSING_PARAMS` specs before committing (the dead-key lesson from `anime_crystal_void`) — especially `whiten_tone`, `skin_protect`, `halation` recipe-key spellings (`retouch/params.py:811-851`).
- Double-toning guard: the recipe's split-tone params AND the preset's `split_tone_three_way` both run. Keep recipe-side sat values low (as above) or zero them and rely solely on the preset — decide during visual tuning, whichever avoids over-toned shadows.

## Step 3 — Tests

In `tests/test_recipe_integration.py`, mirror the existing `anime_v2` tests:
- `resolve_recipe("moonlight_porcelain")` inherits the anime skin stack and applies overrides (highlight_hue is 210, not the parent's 320).
- End-to-end `engine.process()` run on the synthetic test face: no exceptions, valid uint8 output, no NaN, output differs from input.
- **New guard for this recipe only:** every key in the two new recipe dicts resolves to a known ParamSpec name / recipe_key (prevents shipping another `crystal_void`).
- Preset JSON loads via `load_preset("moonlight_porcelain")` and passes whatever schema checks the preset loader has.

## Step 4 — Visual tuning + artifact QA (the "no artifacts" requirement)

Run the CLI/GUI on 2–3 real portraits (ideally one on a dark background, one with white costume, one with warm props) and check these **specific interaction risks**:

| Risk | Where to look | Mitigation lever |
|---|---|---|
| Banding in dark blue background (8-bit + strong shadow split-tone + curve toe) | smooth dark gradients at 100% zoom | lower `shadow_sat`; preset grain 0.03 dithers residual banding |
| `quantize` (0.40) bands amplified by bloom/glow | cheek/forehead gradients | reduce `skin.quantize` to ~0.30 in this recipe |
| `flatten` + `sharpen 22` halos | jawline, hairline vs background | lower sharpen or flatten; flatten has downsample guard ≥1200px already |
| Milky/washed face: `skin.glow 0.25` + `bloom 0.20` + preset `haze/orton_glow` all stack | face highlights | this is 4 glow sources — trim to 2–3 if face loses contrast |
| Blue channel clipping on white costume (WB B=1.08 + whites +6) | B-channel histogram, costume texture | reduce `whites` or WB blue |
| Person-mask halo from `subject_separation 35` | hair wisps against darkened background | existing 2%-feather in `_stage_subject_separation` (engine.py:1492); reduce strength if MediaPipe mask is coarse |
| CA fringing on silver jewelry/heels | high-contrast metal edges | already reduced to 2.0; drop to 0 if visible |
| `unify_tone` fighting rose blush/lips | lip edges, cheek blush | unify runs before makeup per pipeline order — confirm blush is applied after; keep `unify` ≤ 0.50 |

Also run: `pytest tests/test_recipe_integration.py tests/test_skin.py tests/test_grading.py tests/test_gui.py` and confirm the new recipe appears in the GUI dropdown + `reset` functions return its defaults.

## Optional Stage D (deferred — ask before doing)

Wire the useful subset of the dead `anime_crystal_void` params as real ParamSpecs + engine stages: `background_desaturation`, `background_blur`, `light_wrap`. These would strengthen the smoky-background/rim-glow match AND revive `anime_crystal_void`. Skipped for now because (a) the user asked to report bugs rather than plan fixes, and (b) `subject_separation` + preset `haze`/`vignette` likely get close enough. Revisit after Step 4 A/B review.

## Files touched

| File | Change |
|---|---|
| `presets/moonlight_porcelain.json` | **new** — the grade preset (Step 1) |
| `retouch/recipes.py` | add `moonlight_porcelain` + `moonlight_cool` recipes (Step 2) |
| `tests/test_recipe_integration.py` | recipe resolution + e2e + dead-key guard tests (Step 3) |
| `RECIPE_GUIDE.md` | document the new recipes/preset (follow existing anime_v2 section pattern) |

No engine/GUI code changes needed — recipes and presets are auto-discovered.

## Verification

1. `pytest` suites above pass (plus full suite before commit).
2. CLI render of 2–3 sample photos with `--recipe moonlight_porcelain`; A/B against original and against `anime_v2` at 100% crop on face, hairline, background gradient, white costume, warm props.
3. Walk the artifact QA matrix (Step 4) on each render; adjust the flagged levers until clean.
4. GUI smoke test via `dev.sh`: select recipe in dropdown, confirm sliders populate with recipe defaults, process one image.
