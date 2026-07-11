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
