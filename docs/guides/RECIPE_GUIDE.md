# Recipe Creation Guide

## File

Edit `retouch/recipes.py` — add a new entry to the `RECIPES` dict.

## Quick Template

```python
"my_recipe": {
    "extends": "natural",           # inherit from any existing recipe
    "frequency": {"smooth": 0.55},  # override only what differs
    "skin": {"equalize": 0.30, "rosy": 0.40},
    "eyes": {"whites": 0.20, "iris": 0.25, "catchlight": 0.20},
    "lips": {"tint": "rose", "gloss": 0.15},
    "hair": {"shine": 0.20},
    "dodge_burn": {"amount": 0.15},
    "color_harmony": {"preset": "natural", "amount": 0.30},
    "bloom": {"opacity": 0.05},
    "texture": {"opacity": 0.90},
    "contrast": 0,
    "brightness": 0,
    "blush": 25.0,
},
```

## Field Reference — All Groups (Auto-Generated)

### Nested Groups

These keys are defined inside nested dictionary blocks inside a recipe.

| Group | Key | Range / Value | Engine Default | Parameter Name |
|---|---|---|---|---|
| `ai` | `denoise` | 0 to 100 | `0` | `ai_denoise` |
| `background` | `backdrop_cleanup` | 0 to 100 | `0` | `backdrop_cleanup` |
| `background` | `background_blur` | 0 to 100 | `0.0` | `background_blur` |
| `background` | `background_desaturation` | 0 to 100 | `0.0` | `background_desaturation` |
| `background` | `blue_shadow_grade` | 0 to 100 | `0.0` | `blue_shadow_grade` |
| `background` | `cyan_midtone_grade` | 0 to 100 | `0.0` | `cyan_midtone_grade` |
| `background` | `lens_blur` | 0 to 100 | `0.0` | `lens_blur` |
| `background` | `light_wrap` | 0 to 100 | `0.0` | `light_wrap` |
| `background` | `matte_black` | 0 to 100 | `0.0` | `matte_black` |
| `background` | `subject_sharpen` | 0 to 100 | `0.0` | `subject_sharpen` |
| `bloom` | `opacity` | 0.0 to 100.0 | `0.0` | `bloom` |
| `bloom` | `softness` | 0.0 to 1.0 | `30.0` | `bloom_softness` |
| `bloom` | `threshold` | 0.0 to 1.0 | `210.0` | `bloom_threshold` |
| `body_reshape` | `arm_length` | 0 to 100 | `50.0` | `body_reshape_arm_length` |
| `body_reshape` | `auto` | 0 to 100 | `0.0` | `auto_body_reshape` |
| `body_reshape` | `hip_width` | 0 to 100 | `50.0` | `body_reshape_hip_width` |
| `body_reshape` | `leg_length` | 0 to 100 | `50.0` | `body_reshape_leg_length` |
| `body_reshape` | `shoulder_width` | 0 to 100 | `50.0` | `body_reshape_shoulder_width` |
| `body_reshape` | `torso_width` | 0 to 100 | `50.0` | `body_reshape_torso_width` |
| `body_skin` | `dodge_burn` | 0 to 100 | `0` | `body_dodge_burn` |
| `body_skin` | `equalize` | 0 to 100 | `0` | `body_equalize` |
| `body_skin` | `match_face` | 0 to 100 | `0` | `body_match_face` |
| `body_skin` | `relight` | 0 to 100 | `0` | `body_relight` |
| `body_skin` | `shadow_lift` | 0 to 100 | `0` | `body_shadow_lift` |
| `body_skin` | `smooth` | 0 to 100 | `0` | `body_smooth` |
| `body_skin` | `whiten` | 0 to 100 | `0` | `body_whiten` |
| `calibration` | `blue.hue` | -180 to 180 | `0.0` | `calibration_blue_hue` |
| `calibration` | `blue.lum` | -100 to 100 | `0.0` | `calibration_blue_lum` |
| `calibration` | `blue.sat` | -100 to 100 | `0.0` | `calibration_blue_sat` |
| `calibration` | `green.hue` | -180 to 180 | `0.0` | `calibration_green_hue` |
| `calibration` | `green.lum` | -100 to 100 | `0.0` | `calibration_green_lum` |
| `calibration` | `green.sat` | -100 to 100 | `0.0` | `calibration_green_sat` |
| `calibration` | `red.hue` | -180 to 180 | `0.0` | `calibration_red_hue` |
| `calibration` | `red.lum` | -100 to 100 | `0.0` | `calibration_red_lum` |
| `calibration` | `red.sat` | -100 to 100 | `0.0` | `calibration_red_sat` |
| `color_harmony` | `amount` | 0.0 to 1.0 | `0.0` | `grade_intensity` |
| `color_harmony` | `preset` | 0.0 to 1.0 | `none` | `color_grade` |
| `cosplay` | `consistency_strength` | 0 to 100 | `0` | `cosplay_consistency_strength` |
| `cosplay` | `stockings_smooth` | 0 to 100 | `0` | `cosplay_stockings_smooth` |
| `cosplay` | `wig_lace_blend` | 0 to 100 | `0` | `cosplay_wig_lace_blend` |
| `eye` | `iris_brightness` | 0 to 100 | `0` | `eye_iris_brightness` |
| `eye` | `iris_hue_shift` | -30 to 30 | `0` | `eye_iris_hue_shift` |
| `eye` | `iris_saturate` | 0 to 100 | `0` | `eye_iris_saturate` |
| `eye` | `sclera_brighten` | 0 to 100 | `0` | `eye_sclera_brighten` |
| `eyes` | `catchlight` | 0 to 100 | `5` | `catchlight` |
| `eyes` | `dark_circles` | 0 to 100 | `0` | `dark_circles` |
| `eyes` | `sclera_vessel_remove` | 0 to 100 | `0` | `eye_sclera_vessel_remove` |
| `eyes` | `undereye_shadow_strength` | 0.0 to 1.0 | `0.0` | `undereye_shadow_strength` |
| `eyes` | `whites` | 0 to 100 | `5` | `eye_enhance` |
| `eyes` | `whites` | 0 to 100 | `5` | `teeth_whiten` |
| `fabric` | `wrinkle_smooth` | 0 to 100 | `0.0` | `fabric_wrinkle_smooth` |
| `film` | `crosstalk.cy_mg` | -0.15 to 0.15 | `0.06` | `film_crosstalk_cy_mg` |
| `film` | `crosstalk.cy_ye` | -0.15 to 0.15 | `0.03` | `film_crosstalk_cy_ye` |
| `film` | `crosstalk.mg_ye` | -0.15 to 0.15 | `0.02` | `film_crosstalk_mg_ye` |
| `film` | `enable` | 0.0 to 1.0 | `False` | `film_enable` |
| `film` | `gamma` | 0.5 to 2.5 | `1.0` | `film_gamma` |
| `film` | `midpoint` | 0.0 to 1.0 | `0.5` | `film_midpoint` |
| `film` | `shoulder.b` | 0.0 to 0.5 | `0.1` | `film_shoulder_b` |
| `film` | `shoulder.g` | 0.0 to 0.5 | `0.1` | `film_shoulder_g` |
| `film` | `shoulder.r` | 0.0 to 0.5 | `0.1` | `film_shoulder_r` |
| `film` | `strength` | 0.0 to 1.0 | `1.0` | `film_strength` |
| `film` | `toe.b` | 0.0 to 0.5 | `0.1` | `film_toe_b` |
| `film` | `toe.g` | 0.0 to 0.5 | `0.1` | `film_toe_g` |
| `film` | `toe.r` | 0.0 to 0.5 | `0.1` | `film_toe_r` |
| `film` | `tonemap.shoulder` | 0.0 to 0.5 | `0.15` | `film_tonemap_shoulder` |
| `film` | `tonemap.skew` | 0.0 to 1.0 | `0.3` | `film_skew` |
| `film` | `tonemap.strength` | 0.0 to 1.0 | `0.7` | `film_tonemap_strength` |
| `film` | `tonemap.toe` | 0.0 to 0.5 | `0.1` | `film_tonemap_toe` |
| `finish` | `airy_haze` | 0 to 100 | `0.0` | `airy_haze` |
| `finish` | `clarity_split_neg` | 0 to 100 | `0.0` | `clarity_split_neg` |
| `finish` | `clarity_split_pos` | 0 to 100 | `0.0` | `clarity_split_pos` |
| `finish` | `fade_toe` | 0 to 100 | `0.0` | `fade_toe` |
| `finish` | `highlight_drift` | 0 to 100 | `0.0` | `highlight_drift` |
| `finish` | `impact` | 0 to 100 | `0.0` | `impact` |
| `frequency` | `blotch_reduction` | 0.0 to 1.0 | `0.0` | `blotch_reduction` |
| `frequency` | `freckle_removal` | 0.0 to 100.0 | `0.0` | `freckle_removal` |
| `frequency` | `mid_reduction` | 0.0 to 1.0 | `0.35` | `mid_reduction` |
| `frequency` | `nose_smooth` | 0 to 100 | `0` | `nose_smooth` |
| `frequency` | `regional_modulation` | 0.0 to 1.0 | `0.0` | `regional_modulation` |
| `frequency` | `smooth_engine` | 0.0 to 1.0 | `guided` | `smooth_engine` |
| `frequency` | `smooth` | 0 to 100 | `30` | `smooth` |
| `hair` | `deglare` | 0 to 100 | `0` | `hair_deglare` |
| `hair` | `remove_flyaways` | 0 to 100 | `0` | `hair_remove_flyaways` |
| `hair` | `ring_position` | 0 to 100 | `30` | `hair_ring_position` |
| `hair` | `ring_tint` | 0 to 100 | `40` | `hair_ring_tint` |
| `hair` | `shine` | 0 to 100 | `5` | `hair_enhance` |
| `harmony` | `background_harmonize_mode` | 0.0 to 1.0 | `split` | `background_harmonize_mode` |
| `harmony` | `background_harmonize` | 0 to 100 | `0.0` | `background_harmonize` |
| `hsl_adjustments` | `hue.blue` | -180 to 180 | `0.0` | `hsl_hue_blue` |
| `hsl_adjustments` | `hue.cyan` | -180 to 180 | `0.0` | `hsl_hue_cyan` |
| `hsl_adjustments` | `hue.green` | -180 to 180 | `0.0` | `hsl_hue_green` |
| `hsl_adjustments` | `hue.magenta` | -180 to 180 | `0.0` | `hsl_hue_magenta` |
| `hsl_adjustments` | `hue.orange` | -180 to 180 | `0.0` | `hsl_hue_orange` |
| `hsl_adjustments` | `hue.purple` | -180 to 180 | `0.0` | `hsl_hue_purple` |
| `hsl_adjustments` | `hue.red` | -180 to 180 | `0.0` | `hsl_hue_red` |
| `hsl_adjustments` | `hue.yellow` | -180 to 180 | `0.0` | `hsl_hue_yellow` |
| `hsl_adjustments` | `luminance.blue` | -100 to 100 | `0.0` | `hsl_lum_blue` |
| `hsl_adjustments` | `luminance.cyan` | -100 to 100 | `0.0` | `hsl_lum_cyan` |
| `hsl_adjustments` | `luminance.green` | -100 to 100 | `0.0` | `hsl_lum_green` |
| `hsl_adjustments` | `luminance.magenta` | -100 to 100 | `0.0` | `hsl_lum_magenta` |
| `hsl_adjustments` | `luminance.orange` | -100 to 100 | `0.0` | `hsl_lum_orange` |
| `hsl_adjustments` | `luminance.purple` | -100 to 100 | `0.0` | `hsl_lum_purple` |
| `hsl_adjustments` | `luminance.red` | -100 to 100 | `0.0` | `hsl_lum_red` |
| `hsl_adjustments` | `luminance.yellow` | -100 to 100 | `0.0` | `hsl_lum_yellow` |
| `hsl_adjustments` | `saturation.blue` | -100 to 100 | `0.0` | `hsl_sat_blue` |
| `hsl_adjustments` | `saturation.cyan` | -100 to 100 | `0.0` | `hsl_sat_cyan` |
| `hsl_adjustments` | `saturation.green` | -100 to 100 | `0.0` | `hsl_sat_green` |
| `hsl_adjustments` | `saturation.magenta` | -100 to 100 | `0.0` | `hsl_sat_magenta` |
| `hsl_adjustments` | `saturation.orange` | -100 to 100 | `0.0` | `hsl_sat_orange` |
| `hsl_adjustments` | `saturation.purple` | -100 to 100 | `0.0` | `hsl_sat_purple` |
| `hsl_adjustments` | `saturation.red` | -100 to 100 | `0.0` | `hsl_sat_red` |
| `hsl_adjustments` | `saturation.yellow` | -100 to 100 | `0.0` | `hsl_sat_yellow` |
| `lips` | `gloss` | 0 to 100 | `5` | `lip_enhance` |
| `lips` | `tint` | 0.0 to 1.0 | `None` | `lip_tint` |
| `makeup_v2` | `brows_color` | 0.0 to 1.0 | `brown` | `mv2_brows_color` |
| `makeup_v2` | `brows` | 0 to 10 | `0` | `mv2_brows` |
| `makeup_v2` | `contour` | 0 to 100 | `0` | `mv2_contour` |
| `makeup_v2` | `eyeliner_color` | 0.0 to 1.0 | `black` | `mv2_eyeliner_color` |
| `makeup_v2` | `eyeliner_style` | 0.0 to 1.0 | `classic` | `mv2_eyeliner_style` |
| `makeup_v2` | `eyeliner` | 0 to 10 | `0` | `mv2_eyeliner` |
| `makeup_v2` | `eyeshadow_color` | 0.0 to 1.0 | `rose` | `mv2_eyeshadow_color` |
| `makeup_v2` | `eyeshadow_style` | 0.0 to 1.0 | `natural` | `mv2_eyeshadow_style` |
| `makeup_v2` | `eyeshadow` | 0 to 100 | `0` | `mv2_eyeshadow` |
| `makeup_v2` | `ombre_color1` | 0.0 to 1.0 | `red` | `mv2_ombre_color1` |
| `makeup_v2` | `ombre_color2` | 0.0 to 1.0 | `pink` | `mv2_ombre_color2` |
| `makeup_v2` | `ombre` | 0.0 to 1.0 | `False` | `mv2_ombre` |
| `neural` | `defect_boost` | 0 to 100 | `0` | `neural_defect_boost` |
| `neural` | `stray_hair_boost` | 0 to 100 | `0` | `neural_stray_hair_boost` |
| `reshape` | `chin_length` | -50 to 50 | `0` | `reshape_chin_length` |
| `reshape` | `eye_distance` | -50 to 50 | `0` | `reshape_eye_distance` |
| `reshape` | `eye_size_l` | -50 to 50 | `0` | `reshape_eye_size_l` |
| `reshape` | `eye_size_r` | -50 to 50 | `0` | `reshape_eye_size_r` |
| `reshape` | `eye_size` | -50 to 50 | `0` | `reshape_eye_size` |
| `reshape` | `forehead` | -30 to 30 | `0` | `reshape_forehead` |
| `reshape` | `jaw_width_l` | -50 to 50 | `0` | `reshape_jaw_width_l` |
| `reshape` | `jaw_width_r` | -50 to 50 | `0` | `reshape_jaw_width_r` |
| `reshape` | `jaw_width` | -50 to 50 | `0` | `reshape_jaw_width` |
| `reshape` | `mouth_size` | -50 to 50 | `0` | `reshape_mouth_size` |
| `reshape` | `neck_length` | -50 to 50 | `0` | `reshape_neck_length` |
| `reshape` | `neck_width` | -50 to 50 | `0` | `reshape_neck_width` |
| `reshape` | `nose_length` | -50 to 50 | `0` | `reshape_nose_length` |
| `reshape` | `nose_width_l` | -50 to 50 | `0` | `reshape_nose_width_l` |
| `reshape` | `nose_width_r` | -50 to 50 | `0` | `reshape_nose_width_r` |
| `reshape` | `nose_width` | -50 to 50 | `0` | `reshape_nose_width` |
| `reshape` | `smile` | -20 to 30 | `0` | `reshape_smile` |
| `skin` | `albedo_even` | 0.0 to 1.0 | `0.0` | `albedo_even` |
| `skin` | `chroma_even` | 0 to 100 | `0` | `skin_chroma_even` |
| `skin` | `equalize` | 0 to 100 | `0` | `equalize` |
| `skin` | `face_exposure` | 0 to 100 | `0` | `face_exposure` |
| `skin` | `flatten` | 0 to 100 | `0` | `skin_flatten` |
| `skin` | `glow` | 0 to 100 | `0` | `skin_glow` |
| `skin` | `hemoglobin_smooth` | 0.0 to 1.0 | `0.0` | `hemoglobin_smooth` |
| `skin` | `hue_unify` | 0 to 100 | `0` | `skin_hue_unify` |
| `skin` | `makeup_cake_reduce` | 0.0 to 1.0 | `0.0` | `makeup_cake_reduce` |
| `skin` | `makeup_coverage_even` | 0.0 to 1.0 | `0.0` | `makeup_coverage_even` |
| `skin` | `micro_db` | 0 to 100 | `0` | `micro_dodge_burn` |
| `skin` | `mole_protect` | 0.0 to 1.0 | `0.0` | `mole_protect` |
| `skin` | `nose_restore` | 0 to 100 | `0` | `nose_restore` |
| `skin` | `porcelain` | 0.0 to 1.0 | `rosy` | `whiten_tone` |
| `skin` | `quantize` | 0 to 100 | `0` | `skin_quantize` |
| `skin` | `redness_even` | 0 to 100 | `0` | `redness_even` |
| `skin` | `relight_azimuth` | 0.0 to 1.0 | `0.0` | `relight_azimuth` |
| `skin` | `relight_elevation` | 0.0 to 1.0 | `30.0` | `relight_elevation` |
| `skin` | `relight` | 0 to 100 | `0` | `relight` |
| `skin` | `rosy` | 0 to 100 | `10` | `whiten` |
| `skin` | `sculpt` | 0 to 100 | `0` | `sculpt` |
| `skin` | `sss` | 0 to 100 | `0` | `skin_sss` |
| `skin` | `shadow_lift` | 0 to 100 | `0` | `shadow_lift` |
| `skin` | `shine_removal` | 0 to 100 | `0` | `shine_removal` |
| `skin` | `specular_finish_strength` | 0.0 to 1.0 | `0.5` | `specular_finish_strength` |
| `skin` | `specular_finish` | 0.0 to 1.0 | `matte` | `specular_finish` |
| `skin` | `specular_recolor` | 0.0 to 1.0 | `0.0` | `specular_recolor` |
| `skin` | `texture_transplant` | 0 to 100 | `0` | `texture_transplant` |
| `skin` | `unify_hue` | -1.0 to 360.0 | `-1.0` | `skin_unify_hue` |
| `skin` | `unify` | 0 to 100 | `0` | `skin_unify` |
| `skin` | `vein_attenuate` | 0.0 to 1.0 | `0.0` | `vein_attenuate` |
| `skin` | `whiten_hue_stable` | 0.0 to 1.0 | `False` | `whiten_hue_stable` |
| `skin` | `wrinkle_soften_forehead` | 0 to 100 | `0.0` | `wrinkle_soften_forehead` |
| `skin` | `wrinkle_soften_nasolabial` | 0 to 100 | `0.0` | `wrinkle_soften_nasolabial` |
| `skin` | `wrinkle_soften_neck` | 0 to 100 | `0.0` | `wrinkle_soften_neck` |
| `skin` | `wrinkle_soften` | 0 to 100 | `0` | `wrinkle_soften` |
| `texture` | `opacity` | 0.0 to 1.0 | `1.0` | `texture_opacity` |
| `texture` | `pore_synthesis` | 0 to 100 | `0` | `pore_synthesis` |
| `undereye` | `darken_removal` | 0 to 100 | `0` | `undereye_darken_removal` |
| `undereye` | `puffiness_reduction` | 0 to 100 | `0` | `undereye_puffiness_reduction` |
| `mask` | `feather_mode` | `gaussian` or `guided` | `gaussian` | `mask_feather_mode` |

### Flat Fields (top-level, direct values)

These keys are defined directly at the top level of a recipe dictionary.

| Key | Range / Value | Engine Default | CLI Equivalent |
|---|---|---|---|
| `auto_exposure` | 0.0 to 1.0 | `False` | `--auto-exposure` if applicable |
| `blacks` | -100 to 100 | `None` | `--blacks` if applicable |
| `blush` | 0 to 100 | `0` | `--blush` if applicable |
| `brightness` | -50 to 50 | `None` | `--brightness` if applicable |
| `bw_channel_mixer_b` | -100 to 200 | `11` | `--bw-b` if applicable |
| `bw_channel_mixer_g` | -100 to 200 | `59` | `--bw-g` if applicable |
| `bw_channel_mixer_r` | -100 to 200 | `30` | `--bw-r` if applicable |
| `chromatic_aberration` | 0.0 to 1.0 | `0.0` | `--chromatic-aberration` if applicable |
| `clarity` | -50 to 50 | `0.0` | None |
| `color_transfer_intensity` | 0.0 to 1.0 | `1.0` | `--color-transfer-intensity` if applicable |
| `contrast` | -50 to 50 | `0.0` | `--contrast` if applicable |
| `dodge_burn` | 0 to 100 | `0` | `--dodge-burn` if applicable |
| `gamut_compress` | 0.0 to 1.0 | `True` | `--gamut-compress` if applicable |
| `glow` | 0 to 100 | `0.0` | None |
| `grain_strength` | 0.0 to 1.0 | `0.0` | `--film-grain` if applicable |
| `grain` | 0.0 to 1.0 | `0.0` | `--grain` if applicable |
| `halation` | 0.0 to 1.0 | `0.0` | None |
| `highlight_hue` | 0.0 to 1.0 | `0.0` | None |
| `highlight_rolloff` | 0.0 to 1.0 | `0.0` | `--highlight-rolloff` if applicable |
| `highlight_rolloff_strength` | 0.0 to 1.0 | `0.0` | `--highlight-rolloff` if applicable |
| `highlight_sat` | 0.0 to 1.0 | `0.0` | None |
| `highlights` | -100 to 100 | `None` | `--highlights` if applicable |
| `hsl_hue_global` | -100 to 100 | `0` | `--hsl-hue` if applicable |
| `hsl_lum_global` | -100 to 100 | `0` | `--hsl-lum` if applicable |
| `hsl_sat_global` | -100 to 100 | `0` | `--hsl-sat` if applicable |
| `lip_finish` | 0.0 to 1.0 | `gloss` | `--lip-finish` if applicable |
| `lut` | 0.0 to 1.0 | `none` | `--lut` if applicable |
| `micro_restore` | 0 to 50 | `20` | `--micro-restore` if applicable |
| `midtone_hue` | 0.0 to 1.0 | `0.0` | None |
| `midtone_sat` | 0.0 to 1.0 | `0.0` | None |
| `negative_split_tone_highlight` | 0.0 to 100.0 | `0.0` | `--neg-split-highlight` if applicable |
| `negative_split_tone_shadow` | 0.0 to 100.0 | `0.0` | `--neg-split-shadow` if applicable |
| `nose_blush` | 0.0 to 1.0 | `False` | `--nose-blush` if applicable |
| `saturation_mode` | 0.0 to 1.0 | `additive` | `--saturation-mode` if applicable |
| `saturation` | -100 to 100 | `0.0` | None |
| `shadow_hue` | 0.0 to 1.0 | `0.0` | None |
| `shadow_sat` | 0.0 to 1.0 | `0.0` | None |
| `shadows` | -100 to 100 | `None` | `--shadows` if applicable |
| `sharpen_radius` | 0.0 to 1.0 | `1.0` | None |
| `sharpen` | 0 to 100 | `0.0` | None |
| `skin_protect` | 0.0 to 1.0 | `0.0` | `--skin-protect` if applicable |
| `skin_protect_strength` | 0.0 to 1.0 | `0.0` | `--skin-protect` if applicable |
| `slimming` | 0 to 100 | `0` | `--slimming` if applicable |
| `specular_bloom_tone` | 0.0 to 1.0 | `rosy` | `--specular-bloom-tone` if applicable |
| `specular_bloom` | 0 to 100 | `0` | `--specular-bloom` if applicable |
| `subject_separation` | 0 to 100 | `0.0` | None |
| `tonal_curve_strength` | 0.0 to 1.0 | `0.0` | `--tonal-curve-strength` if applicable |
| `under_eye_blush` | 0.0 to 1.0 | `False` | `--under-eye-blush` if applicable |
| `vibrance` | -100 to 100 | `0.0` | None |
| `vignette` | -100 to 100 | `0.0` | None |
| `white_balance_kelvin` | 2000 to 12000 | `6500` | `--wb-kelvin` if applicable |
| `white_balance_tint` | -100.0 to 100.0 | `0.0` | `--wb-tint` if applicable |
| `white_costume_lift` | 0.0 to 1.0 | `False` | `--white-costume-lift` if applicable |
| `whites` | -100 to 100 | `None` | `--whites` if applicable |

### Runtime-only controls

These registered controls are accepted by the Python API or GUI state but do
not serialize as ordinary recipe values:

- `ai_sr_scale` — output upscale factor (`1`, `2`, or `4`).
- `blemish` — legacy API control; use the recipe's smoothing and blemish fields for new presets.
- `freckle_preserve_mask` — an in-memory mask supplied by API callers to protect selected marks.

#### `color_harmony.preset` valid values

Any preset JSON file in the `presets/` directory can be referenced. The bundled
presets are:

- `natural`
- `portrait`
- `cosplay`
- `beauty`
- `fantasy`
- `cyberpunk`
- `pink_dream`
- `blue_dream`
- `xhs_ultrasoft`
- `xiaohongshu`
- `fuji_porcelain`
- `film`
- `bw_noir`
- `golden_hour`

Custom presets dropped into `presets/` are picked up automatically — use the
filename (without the `.json` extension) as the preset name.

## GUI Slider ↔ Recipe Value Conversion

GUI slider values are converted in `gui.py:recipe_defaults()`. Most sliders
live on a 0–100 GUI scale; recipe values follow the engine's native scale
(typically 0.0–1.0 for fraction-style fields, signed 0–100 for tonal fields,
and 0–360 for split-toning hue).

| GUI Slider / Control | Recipe field | Conversion |
|---|---|---|
| Ai Denoise | `ai.denoise` | ÷ 100 |
| Airy Haze | `finish.airy_haze` | direct |
| Albedo Even | `skin.albedo_even` | direct |
| Auto Body Reshape | `body_reshape.auto` | direct |
| Auto Exposure | `auto_exposure` | direct |
| Backdrop Cleanup | `background.backdrop_cleanup` | ÷ 100 |
| Background Blur | `background.background_blur` | direct |
| Background Desaturation | `background.background_desaturation` | direct |
| Background Harmonize Mode | `harmony.background_harmonize_mode` | direct |
| Background Harmonize | `harmony.background_harmonize` | direct |
| Blacks | `blacks` | direct |
| Bloom Softness | `bloom.softness` | direct |
| Bloom Threshold | `bloom.threshold` | direct |
| Bloom | `bloom.opacity` | ÷ 100 |
| Blotch Reduction | `frequency.blotch_reduction` | direct |
| Blue Shadow Grade | `background.blue_shadow_grade` | direct |
| Blush | `blush` | direct |
| Body Dodge Burn | `body_skin.dodge_burn` | ÷ 100 |
| Body Equalize | `body_skin.equalize` | ÷ 100 |
| Body Match Face | `body_skin.match_face` | ÷ 100 |
| Body Relight | `body_skin.relight` | ÷ 100 |
| Body Reshape Arm Length | `body_reshape.arm_length` | direct |
| Body Reshape Hip Width | `body_reshape.hip_width` | direct |
| Body Reshape Leg Length | `body_reshape.leg_length` | direct |
| Body Reshape Shoulder Width | `body_reshape.shoulder_width` | direct |
| Body Reshape Torso Width | `body_reshape.torso_width` | direct |
| Body Shadow Lift | `body_skin.shadow_lift` | ÷ 100 |
| Body Smooth | `body_skin.smooth` | ÷ 100 |
| Body Whiten | `body_skin.whiten` | ÷ 100 |
| Brightness | `brightness` | direct |
| Bw Channel Mixer B | `bw_channel_mixer_b` | direct |
| Bw Channel Mixer G | `bw_channel_mixer_g` | direct |
| Bw Channel Mixer R | `bw_channel_mixer_r` | direct |
| Calibration Blue Hue | `calibration.blue.hue` | direct |
| Calibration Blue Lum | `calibration.blue.lum` | direct |
| Calibration Blue Sat | `calibration.blue.sat` | direct |
| Calibration Green Hue | `calibration.green.hue` | direct |
| Calibration Green Lum | `calibration.green.lum` | direct |
| Calibration Green Sat | `calibration.green.sat` | direct |
| Calibration Red Hue | `calibration.red.hue` | direct |
| Calibration Red Lum | `calibration.red.lum` | direct |
| Calibration Red Sat | `calibration.red.sat` | direct |
| Catchlight | `eyes.catchlight` | ÷ 100 |
| Chromatic Aberration | `chromatic_aberration` | direct |
| Clarity Split Neg | `finish.clarity_split_neg` | direct |
| Clarity Split Pos | `finish.clarity_split_pos` | direct |
| Clarity | `clarity` | direct |
| Color Grade | `color_harmony.preset` | direct |
| Color Transfer Intensity | `color_transfer_intensity` | direct |
| Contrast | `contrast` | direct |
| Cosplay Consistency Strength | `cosplay.consistency_strength` | ÷ 100 |
| Cosplay Stockings Smooth | `cosplay.stockings_smooth` | ÷ 100 |
| Cosplay Wig Lace Blend | `cosplay.wig_lace_blend` | ÷ 100 |
| Cyan Midtone Grade | `background.cyan_midtone_grade` | direct |
| Dark Circles | `eyes.dark_circles` | ÷ 100 |
| Dodge Burn | `dodge_burn` | direct |
| Equalize | `skin.equalize` | ÷ 100 |
| Eye Enhance | `eyes.whites` | ÷ 100 |
| Eye Iris Brightness | `eye.iris_brightness` | ÷ 100 |
| Eye Iris Hue Shift | `eye.iris_hue_shift` | direct |
| Eye Iris Saturate | `eye.iris_saturate` | ÷ 100 |
| Eye Sclera Brighten | `eye.sclera_brighten` | ÷ 100 |
| Eye Sclera Vessel Remove | `eyes.sclera_vessel_remove` | ÷ 100 |
| Fabric Wrinkle Smooth | `fabric.wrinkle_smooth` | ÷ 100 |
| Face Exposure | `skin.face_exposure` | ÷ 100 |
| Fade Toe | `finish.fade_toe` | direct |
| Film Crosstalk Cy Mg | `film.crosstalk.cy_mg` | direct |
| Film Crosstalk Cy Ye | `film.crosstalk.cy_ye` | direct |
| Film Crosstalk Mg Ye | `film.crosstalk.mg_ye` | direct |
| Film Enable | `film.enable` | direct |
| Film Gamma | `film.gamma` | direct |
| Film Midpoint | `film.midpoint` | direct |
| Film Shoulder B | `film.shoulder.b` | direct |
| Film Shoulder G | `film.shoulder.g` | direct |
| Film Shoulder R | `film.shoulder.r` | direct |
| Film Skew | `film.tonemap.skew` | direct |
| Film Strength | `film.strength` | direct |
| Film Toe B | `film.toe.b` | direct |
| Film Toe G | `film.toe.g` | direct |
| Film Toe R | `film.toe.r` | direct |
| Film Tonemap Shoulder | `film.tonemap.shoulder` | direct |
| Film Tonemap Strength | `film.tonemap.strength` | direct |
| Film Tonemap Toe | `film.tonemap.toe` | direct |
| Freckle Removal | `frequency.freckle_removal` | direct |
| Gamut Compress | `gamut_compress` | direct |
| Glow | `glow` | direct |
| Grade Intensity | `color_harmony.amount` | ÷ 100 |
| Grain Strength | `grain_strength` | direct |
| Grain | `grain` | ÷ 500 |
| Hair Deglare | `hair.deglare` | ÷ 100 |
| Hair Enhance | `hair.shine` | ÷ 100 |
| Hair Remove Flyaways | `hair.remove_flyaways` | ÷ 100 |
| Hair Ring Position | `hair.ring_position` | ÷ 100 |
| Hair Ring Tint | `hair.ring_tint` | ÷ 100 |
| Halation | `halation` | ÷ 100 |
| Hemoglobin Smooth | `skin.hemoglobin_smooth` | direct |
| Highlight Drift | `finish.highlight_drift` | direct |
| Highlight Hue | `highlight_hue` | direct |
| Highlight Rolloff Strength | `highlight_rolloff` | direct |
| Highlight Sat | `highlight_sat` | direct |
| Highlights | `highlights` | direct |
| Hsl Hue Blue | `hsl_adjustments.hue.blue` | direct |
| Hsl Hue Cyan | `hsl_adjustments.hue.cyan` | direct |
| Hsl Hue Global | `hsl_hue_global` | direct |
| Hsl Hue Green | `hsl_adjustments.hue.green` | direct |
| Hsl Hue Magenta | `hsl_adjustments.hue.magenta` | direct |
| Hsl Hue Orange | `hsl_adjustments.hue.orange` | direct |
| Hsl Hue Purple | `hsl_adjustments.hue.purple` | direct |
| Hsl Hue Red | `hsl_adjustments.hue.red` | direct |
| Hsl Hue Yellow | `hsl_adjustments.hue.yellow` | direct |
| Hsl Lum Blue | `hsl_adjustments.luminance.blue` | direct |
| Hsl Lum Cyan | `hsl_adjustments.luminance.cyan` | direct |
| Hsl Lum Global | `hsl_lum_global` | direct |
| Hsl Lum Green | `hsl_adjustments.luminance.green` | direct |
| Hsl Lum Magenta | `hsl_adjustments.luminance.magenta` | direct |
| Hsl Lum Orange | `hsl_adjustments.luminance.orange` | direct |
| Hsl Lum Purple | `hsl_adjustments.luminance.purple` | direct |
| Hsl Lum Red | `hsl_adjustments.luminance.red` | direct |
| Hsl Lum Yellow | `hsl_adjustments.luminance.yellow` | direct |
| Hsl Sat Blue | `hsl_adjustments.saturation.blue` | direct |
| Hsl Sat Cyan | `hsl_adjustments.saturation.cyan` | direct |
| Hsl Sat Global | `hsl_sat_global` | direct |
| Hsl Sat Green | `hsl_adjustments.saturation.green` | direct |
| Hsl Sat Magenta | `hsl_adjustments.saturation.magenta` | direct |
| Hsl Sat Orange | `hsl_adjustments.saturation.orange` | direct |
| Hsl Sat Purple | `hsl_adjustments.saturation.purple` | direct |
| Hsl Sat Red | `hsl_adjustments.saturation.red` | direct |
| Hsl Sat Yellow | `hsl_adjustments.saturation.yellow` | direct |
| Impact | `finish.impact` | ÷ 100 |
| Lens Blur | `background.lens_blur` | direct |
| Light Wrap | `background.light_wrap` | direct |
| Lip Enhance | `lips.gloss` | ÷ 100 |
| Lip Finish | `lip_finish` | direct |
| Lip Tint | `lips.tint` | direct |
| Lut | `lut` | direct |
| Makeup Cake Reduce | `skin.makeup_cake_reduce` | direct |
| Makeup Coverage Even | `skin.makeup_coverage_even` | direct |
| Mask Feather Mode | `mask.feather_mode` | direct |
| Matte Black | `background.matte_black` | direct |
| Micro Dodge Burn | `skin.micro_db` | ÷ 100 |
| Micro Restore | `micro_restore` | direct |
| Mid Reduction | `frequency.mid_reduction` | direct |
| Midtone Hue | `midtone_hue` | direct |
| Midtone Sat | `midtone_sat` | direct |
| Mole Protect | `skin.mole_protect` | direct |
| Mv2 Brows Color | `makeup_v2.brows_color` | direct |
| Mv2 Brows | `makeup_v2.brows` | direct |
| Mv2 Contour | `makeup_v2.contour` | direct |
| Mv2 Eyeliner Color | `makeup_v2.eyeliner_color` | direct |
| Mv2 Eyeliner Style | `makeup_v2.eyeliner_style` | direct |
| Mv2 Eyeliner | `makeup_v2.eyeliner` | direct |
| Mv2 Eyeshadow Color | `makeup_v2.eyeshadow_color` | direct |
| Mv2 Eyeshadow Style | `makeup_v2.eyeshadow_style` | direct |
| Mv2 Eyeshadow | `makeup_v2.eyeshadow` | direct |
| Mv2 Ombre Color1 | `makeup_v2.ombre_color1` | direct |
| Mv2 Ombre Color2 | `makeup_v2.ombre_color2` | direct |
| Mv2 Ombre | `makeup_v2.ombre` | direct |
| Negative Split Tone Highlight | `negative_split_tone_highlight` | direct |
| Negative Split Tone Shadow | `negative_split_tone_shadow` | direct |
| Neural Defect Boost | `neural.defect_boost` | ÷ 100 |
| Neural Stray Hair Boost | `neural.stray_hair_boost` | ÷ 100 |
| Nose Blush | `nose_blush` | direct |
| Nose Restore | `skin.nose_restore` | ÷ 100 |
| Nose Smooth | `frequency.nose_smooth` | direct |
| Pore Synthesis | `texture.pore_synthesis` | ÷ 100 |
| Redness Even | `skin.redness_even` | ÷ 100 |
| Regional Modulation | `frequency.regional_modulation` | direct |
| Relight Azimuth | `skin.relight_azimuth` | direct |
| Relight Elevation | `skin.relight_elevation` | direct |
| Relight | `skin.relight` | ÷ 100 |
| Reshape Chin Length | `reshape.chin_length` | direct |
| Reshape Eye Distance | `reshape.eye_distance` | direct |
| Reshape Eye Size L | `reshape.eye_size_l` | direct |
| Reshape Eye Size R | `reshape.eye_size_r` | direct |
| Reshape Eye Size | `reshape.eye_size` | direct |
| Reshape Forehead | `reshape.forehead` | direct |
| Reshape Jaw Width L | `reshape.jaw_width_l` | direct |
| Reshape Jaw Width R | `reshape.jaw_width_r` | direct |
| Reshape Jaw Width | `reshape.jaw_width` | direct |
| Reshape Mouth Size | `reshape.mouth_size` | direct |
| Reshape Neck Length | `reshape.neck_length` | direct |
| Reshape Neck Width | `reshape.neck_width` | direct |
| Reshape Nose Length | `reshape.nose_length` | direct |
| Reshape Nose Width L | `reshape.nose_width_l` | direct |
| Reshape Nose Width R | `reshape.nose_width_r` | direct |
| Reshape Nose Width | `reshape.nose_width` | direct |
| Reshape Smile | `reshape.smile` | direct |
| Saturation Mode | `saturation_mode` | direct |
| Saturation | `saturation` | direct |
| Sculpt | `skin.sculpt` | ÷ 100 |
| Shadow Hue | `shadow_hue` | direct |
| Shadow Lift | `skin.shadow_lift` | ÷ 100 |
| Shadow Sat | `shadow_sat` | direct |
| Shadows | `shadows` | direct |
| Sharpen Radius | `sharpen_radius` | direct |
| Sharpen | `sharpen` | direct |
| Shine Removal | `skin.shine_removal` | ÷ 100 |
| Skin Chroma Even | `skin.chroma_even` | ÷ 100 |
| Skin Flatten | `skin.flatten` | ÷ 100 |
| Skin Glow | `skin.glow` | ÷ 100 |
| Skin SSS | `skin.sss` | ÷ 100 |
| Skin Hue Unify | `skin.hue_unify` | ÷ 100 |
| Skin Protect Strength | `skin_protect` | direct |
| Skin Quantize | `skin.quantize` | ÷ 100 |
| Skin Unify Hue | `skin.unify_hue` | direct |
| Skin Unify | `skin.unify` | ÷ 100 |
| Slimming | `slimming` | direct |
| Smooth Engine | `frequency.smooth_engine` | direct |
| Smooth | `frequency.smooth` | ÷ 100 |
| Specular Bloom Tone | `specular_bloom_tone` | direct |
| Specular Bloom | `specular_bloom` | direct |
| Specular Finish Strength | `skin.specular_finish_strength` | direct |
| Specular Finish | `skin.specular_finish` | direct |
| Specular Recolor | `skin.specular_recolor` | direct |
| Subject Separation | `subject_separation` | direct |
| Subject Sharpen | `background.subject_sharpen` | direct |
| Teeth Whiten | `eyes.whites` | ÷ 100 |
| Texture Opacity | `texture.opacity` | direct |
| Texture Transplant | `skin.texture_transplant` | ÷ 100 |
| Tonal Curve Strength | `tonal_curve_strength` | direct |
| Under Eye Blush | `under_eye_blush` | direct |
| Undereye Darken Removal | `undereye.darken_removal` | ÷ 100 |
| Undereye Puffiness Reduction | `undereye.puffiness_reduction` | ÷ 100 |
| Undereye Shadow Strength | `eyes.undereye_shadow_strength` | direct |
| Vein Attenuate | `skin.vein_attenuate` | direct |
| Vibrance | `vibrance` | direct |
| Vignette | `vignette` | direct |
| White Balance Kelvin | `white_balance_kelvin` | direct |
| White Balance Tint | `white_balance_tint` | direct |
| White Costume Lift | `white_costume_lift` | direct |
| Whiten Hue Stable | `skin.whiten_hue_stable` | direct |
| Whiten Tone | `skin.porcelain` | direct |
| Whiten | `skin.rosy` | ÷ 100 |
| Whites | `whites` | direct |
| Wrinkle Soften Forehead | `skin.wrinkle_soften_forehead` | ÷ 100 |
| Wrinkle Soften Nasolabial | `skin.wrinkle_soften_nasolabial` | ÷ 100 |
| Wrinkle Soften Neck | `skin.wrinkle_soften_neck` | ÷ 100 |
| Wrinkle Soften | `skin.wrinkle_soften` | ÷ 100 |

> **Note:** Fields that use the engine's native scale (typically 0–100 or signed) pass through directly. Fields that use 0.0–1.0 ratios on the engine/recipe side are mapped to/from 0–100 GUI values by dividing or multiplying by 100 or 500 respectively.

## Inheriting with `"extends"`

A recipe without `"extends"` gets no defaults — specify everything.
A recipe with `"extends": "natural"` inherits all fields from `natural` and you only override what differs.

```python
"my_variant": {
    "extends": "cosplay",
    "skin": {"equalize": 0.00},           # override just equalize
    "bloom": {"opacity": 0.15},           # override bloom
},
```

## Xiaohongshu (XHS) Recipe Family

Three presets target the Xiaohongshu light-and-air look. They share the same
`extends: "natural"` parent and a light-sculpting stack (`relight`,
`specular_bloom`, `tonal_curve_strength`, `highlight_rolloff`, `skin_protect`,
cool/warm split-toning, `micro_restore`, `grain_strength`) but tune it with
different intensity:

| Recipe | Strength | `relight` | `specular_bloom` | `micro_restore` | `tonal_curve_strength` |
|---|---|---:|---:|---:|---:|
| `xiaohongshu` | Baseline | 40 | 30 | 25 | 0.25 |
| `xhs_ultrasoft` | Stronger | 55 | 40 | 30 | 0.35 |
| `xhs_soft_glow` | Strongest — full light-sculpting stack with bloom + rolloff + grain | 65 | 50 | 35 | 0.45 |

`xiaohongshu` and `xhs_ultrasoft` were updated to activate the dormant
light-sculpting stack; `xhs_soft_glow` is new and the strongest preset of the
three.

## Adding a Recipe

1. Add dict entry to `RECIPES` in `retouch/recipes.py`
2. Recipe name auto-appears in GUI dropdown (see `gui.py:22`, `gui.py:397`)
3. Restart the GUI to see it

## Testing

```python
from retouch import RetouchEngine
import cv2

engine = RetouchEngine()
img = cv2.imread("test.jpg")
result = engine.process(img, recipe="my_recipe")
cv2.imwrite("output.jpg", result)
```
