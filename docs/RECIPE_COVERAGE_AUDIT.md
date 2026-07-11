# Recipe Coverage Audit — 54/196 params with no recipe

**Date:** 2026-07-11
**Owner:** Dennis
**Status:** Audit complete; 4 reachability recipes added; 3 params still unreachable by design.

## Summary

Of **196** engine params (`retouch.params.PROCESSING_PARAMS`), **54** had no recipe
supplying their `recipe_key` path at audit time. Method: flatten every merged recipe
dict (incl. `extends`) into dotted paths, then check each spec's `recipe_key` /
`gui_recipe_key` / `engine_recipe_key` against that set.

## Grouped gaps (uncovered families)

- **Face reshape (F5 liquify) — 17 params** (group `reshape.<k>`, sliders −100..+100):
  `reshape_eye_size`, `reshape_eye_distance`, `reshape_nose_width`, `reshape_nose_length`,
  `reshape_jaw_width`, `reshape_chin_length`, `reshape_mouth_size`, `reshape_smile`,
  `reshape_forehead`, `reshape_jaw_width_l`, `reshape_jaw_width_r`, `reshape_nose_width_l`,
  `reshape_nose_width_r`, `reshape_eye_size_l`, `reshape_eye_size_r`, `reshape_neck_width`,
  `reshape_neck_length`.
  → **FIXED** via `face_reshape_demo_v1` (verified: ctx=12, active, warps applied).

- **Monochrome / B&W — 5 params**: `bw_channel_mixer_r`, `bw_channel_mixer_g`,
  `bw_channel_mixer_b`, `negative_split_tone_shadow`, `negative_split_tone_highlight`.
  → **FIXED** via `mono_noir_v1`.

- **Creative grade — 9 params**: `hsl_hue_global`, `hsl_sat_global`, `hsl_lum_global`,
  `saturation_mode`, `halation`, `gamut_compress`, `color_transfer_intensity`, `grain`, `lut`.
  → **PARTIAL**: `mono_noir_v1`/`creative_grade_v1` cover HSL + saturation_mode + halation.
  `gamut_compress` (default True → no-op when set True), `color_transfer_intensity`
  (needs `--color-ref` to act), `grain`, `lut` (GUI sentinel `none`) still effectively
  no-op from a pure recipe without extra inputs.

- **Phase-7 "everything auto" batch — 11 params**: `fabric_wrinkle_smooth`,
  `backdrop_cleanup`, `eye_sclera_vessel_remove`, `hair_deglare`, `hair_ring_position`,
  `hair_ring_tint`, `hair_remove_flyaways`, `neural_stray_hair_boost`, `neural_defect_boost`,
  `auto_body_reshape`, `auto_exposure`.
  → **PARTIAL**: `auto_clean_v1` covers all except `auto_exposure` (see unreachable below).

- **Skin refinements — 8 params**: `pore_synthesis`, `redness_even`,
  `wrinkle_soften_forehead`, `wrinkle_soften_nasolabial`, `wrinkle_soften_neck`,
  `blotch_reduction`, `bloom_softness`, `freckle_preserve_mask`.
  → **PARTIAL**: `auto_clean_v1` covers all except `freckle_preserve_mask`.

- **makeup_v2 ombre — 3 params**: `mv2_ombre`, `mv2_ombre_color1`, `mv2_ombre_color2`
  (group `makeup_v2.ombre*`). Eyes/eyeliner/contour/brows already covered by existing
  recipes (`studio_dream_v2`, `editorial_elegance_v1`, etc.). Left as minor gap.

## Unreachable by design (need `spec.recipe_key` + engine wiring, not just a recipe)

- `ai_sr_scale` — `rkey=None`
- `auto_exposure` — `rkey=None`
- `freckle_preserve_mask` — `rkey=None`

These cannot be set via a recipe until a `recipe_key` is added to the `ParamSpec` and
`build_context` is taught to read it (same class of fix as the 2026-07-10 wiring-debt
audit for `body_reshape.*` / `cosplay.*` / `face_exposure`).

## Recipes added (2026-07-11, `retouch/recipes.py`)

`face_reshape_demo_v1`, `mono_noir_v1`, `creative_grade_v1`, `auto_clean_v1` — mirror the
wiring-debt fix: recipe-reachable via engine/CLI; GUI sliders still deferred.

Rendered on `~/Desktop/duotian nikke/DSCF8043.jpg` → `test_output/recipe_gap_fix/`
(contact sheet + per-recipe + `_compare`). Delta vs `natural`: mono_noir 734,
creative_grade 94, auto_clean 942 (all active); face_reshape MSE 0.6 but max pixel diff
89 (localized geo warp, working).

**Visual QA:** PENDING (skin/grading/geometry are Visual-Critical).

## Update 2026-07-12 — caller-only params wired + matsuri_glow_v1

`build_context()` (engine.py) previously took `halation`, `grain`, `lut`, `auto_exposure`,
and `color_transfer_intensity` ONLY from caller overrides (`_CALLER_ONLY`) — recipe values
for these were silently dead (creative_grade_v1's `halation: 0.25` never fired). Fixed via
`_recipe_or_caller()`: caller override wins → recipe value (spec recipe_key + conversion) →
historical fallback (None/None/None/False/1.0). `auto_exposure` gained
`recipe_key="auto_exposure"`; `process()` defaults for `auto_exposure` /
`color_transfer_intensity` changed to None so they no longer clobber recipe values.

- creative_grade_v1 now also sets `grain: 0.05` + `lut: "kodak"` → creative-grade family fully covered except `color_transfer_intensity` (needs a `--color-ref` image to act) and `gamut_compress` (default-True no-op).
- auto_clean_v1 now sets `auto_exposure: true` → Phase-7 family fully covered.
- NEW `matsuri_glow_v1` (Bon Odori festival evening portrait) covers the last family: makeup_v2 ombre (`mv2_ombre`, `mv2_ombre_color1`, `mv2_ombre_color2`), plus live halation/grain.
- Still unreachable by design: `ai_sr_scale` (export-time, caller opt-in per params.py comment), `freckle_preserve_mask` (ndarray mask — cannot be expressed in recipe JSON).
- Regression tests: `tests/test_recipe_posteffects_wiring.py` (20 tests).
- **BUG FOUND & FIXED during render QA:** the first renders came out solid black for any
  recipe with `halation` — `ColorGrader._add_halation` (grading.py) was not float-aware:
  on the per-face E1 float32 [0,1] canvas the uint8-scale threshold (200) zeroed the bleed
  and the final `astype(np.uint8)` floored the whole [0,1] image to black. Never triggered
  before because halation was caller-only and the caller paths hit uint8 inputs. Fixed with
  the standard E1 dtype adapter (same pattern as `_add_grain`); regression tests
  `tests/test_grading_internal.py::TestAddHalation::test_float_input_parity` / `test_float_input_not_black`.
- Rendered on `~/Desktop/bonodori/DSCF8083.jpg` + `DSCF8114.jpg` → `test_output/bonodori_recipe_wiring/`.
  Mean-abs delta vs `natural` (post-fix): DSCF8083 — matsuri_glow 7.34, creative_grade 13.52,
  auto_clean 17.65, mono_noir 15.64; DSCF8114 — matsuri_glow 6.71, creative_grade 12.48,
  auto_clean 12.02, mono_noir 19.56. fx_off wiring proof: creative_grade_v1 vs the same recipe
  with caller overrides `lut="none", grain=1e-6, halation=1e-6` → delta 8.25 (clearly non-zero
  → recipe-supplied lut/grain/halation now fire). auto_exposure proof: no-op on the
  well-exposed originals (delta 0.00 — `correct_exposure` is bounded, by design), but on a
  0.35×-darkened 1200px render auto_clean_v1 with recipe `auto_exposure` vs override False
  gives delta 23.03 (mean luma 67.4 → 90.3) → recipe-supplied auto_exposure fires.
- **auto_clean_v1 artifacts bisected & fixed (2026-07-12, DSCF8114 knockout bisect):**
  the solarized/posterized look was FOUR broken ops, each visually destructive at any
  strength, now REMOVED from the recipe (coverage for these params moves back to
  **gap (implementation broken)**):
  - `fabric.wrinkle_smooth: 40` — painted posterized white outlines over costume + hair
    (fires on fabric print edges, not wrinkles). Knockout collapsed the global delta:
    vs-full 13.16, vs-natural 0.39 (every other family knockout ≤0.21 global).
  - `skin.redness_even: 30` — mottled cyan/pink chroma noise across the whole face
    (face-crop bisect: removing it dropped face delta 3.34 → 2.07; visually clean).
  - `eyes.sclera_vessel_remove: 40` — opaque white/red ellipses painted over both eyes
    ("demon eyes"; face-crop knockout delta 0.85, visually unmistakable).
  - `hair.ring_position: 45` + `hair.ring_tint: 55` — solid blue streak painted into the
    bangs. Resolution-dependent: exact no-op at ≤1600px input (round-1 bisect missed it),
    fires at full 6240px (bangs-crop on/off delta 11.57).
  Everything else stays, incl. `auto_exposure`, `hair.deglare`, `hair.remove_flyaways`,
  `background.backdrop_cleanup`, `neural.*`, `body_reshape.auto` — note the bisect showed
  neural.*, body_reshape.auto, hair.deglare/remove_flyaways and auto_exposure as exact
  no-ops on these photos (deltas 0.00), consistent with the A4 "parked stub" comments in
  params.py: covered but inert. Post-fix full-res deltas vs natural: DSCF8083 **0.10**,
  DSCF8114 **0.33** (remaining ops are face-local + subtle bloom/sharpen). Bisect renders:
  `test_output/bonodori_recipe_wiring/bisect/`.

**Visual QA:** PENDING (grading/makeup are Visual-Critical). matsuri_glow_v1 /
creative_grade_v1 spot-checked on the two Bon Odori photos post-fix — natural skin, warm
lantern grade, no artifacts. auto_clean_v1 re-inspected post-bisect-fix on both photos:
face, eyes, costume and hair all render normally.
