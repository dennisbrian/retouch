# Pro Max Face Retouch Engine — API Reference

This document provides a comprehensive API reference for the `retouch` package. It covers the core orchestrator `RetouchEngine`, the convenience wrapper function `retouch`, structured return types, parameter contexts, and style library helpers.

---

## 1. Package Entry Point

You can import the main engine orchestrator and the one-shot convenience function directly from the root of the package:

```python
from retouch import RetouchEngine, retouch, __version__
```

---

## 2. Core Orchestrator: `RetouchEngine`

The `RetouchEngine` class is the main orchestrator of the retouching pipeline. It initializes the face detector, facial landmarker, semantic parser, and specialized processors (skin smoothing, relighting, makeup, etc.).

### Lifecycle & Context Manager

The engine manages underlying MediaPipe resources and supports explicit resource cleanup. It can be used as a context manager to ensure release of native resources:

```python
# Recommended: Context manager usage
with RetouchEngine(max_faces=5) as engine:
    result = engine.process(img_bgr, recipe="cosplay")

# Manual lifecycle management
engine = RetouchEngine(max_faces=5)
try:
    result = engine.process(img_bgr, recipe="cosplay")
finally:
    engine.close()
```

### `__init__` Parameters

*   **`max_faces`** (`int`, default: `10`): Maximum number of faces to detect and process concurrently.
*   **`min_confidence`** (`float`, default: `0.4`): Minimum detection confidence score for face landmarker.

---

### `process()` Method

Run the full face retouching pipeline on a single image.

```python
def process(
    self,
    img_bgr: np.ndarray,
    recipe: Optional[str] = None,
    preset: Optional[str] = None,
    smooth: Optional[int] = None,
    mid_reduction: Optional[float] = None,
    blotch_reduction: Optional[float] = None,
    texture_opacity: Optional[float] = None,
    pore_synthesis: Optional[float] = None,
    nose_smooth: Optional[int] = None,
    regional_modulation: Optional[float] = None,
    smooth_engine: Optional[str] = None,
    undereye_shadow_strength: Optional[float] = None,
    freckle_removal: Optional[float] = None,
    freckle_preserve_mask: Optional[np.ndarray] = None,
    micro_restore: Optional[int] = None,
    micro_dodge_burn: Optional[int] = None,
    redness_even: Optional[int] = None,
    whiten_hue_stable: Optional[bool] = None,
    whiten: Optional[int] = None,
    equalize: Optional[int] = None,
    blemish: Optional[int] = None,
    whiten_tone: Optional[str] = None,
    nose_blush: Optional[bool] = None,
    under_eye_blush: Optional[bool] = None,
    white_costume_lift: Optional[bool] = None,
    dodge_burn: Optional[int] = None,
    relight: Optional[float] = None,
    relight_azimuth: Optional[float] = None,
    relight_elevation: Optional[float] = None,
    sculpt: Optional[float] = None,
    shine_removal: Optional[float] = None,
    wrinkle_soften: Optional[float] = None,
    wrinkle_soften_forehead: Optional[float] = None,
    wrinkle_soften_nasolabial: Optional[float] = None,
    wrinkle_soften_neck: Optional[float] = None,
    texture_transplant: Optional[float] = None,
    body_smooth: Optional[float] = None,
    body_equalize: Optional[float] = None,
    body_whiten: Optional[float] = None,
    body_match_face: Optional[float] = None,
    body_relight: Optional[float] = None,
    body_dodge_burn: Optional[float] = None,
    shadow_lift: Optional[float] = None,
    face_exposure: Optional[float] = None,
    body_shadow_lift: Optional[float] = None,
    nose_restore: Optional[float] = None,
    specular_bloom: Optional[int] = None,
    specular_bloom_tone: Optional[str] = None,
    specular_finish: Optional[str] = None,
    specular_finish_strength: Optional[float] = None,
    specular_recolor: Optional[float] = None,
    albedo_even: Optional[float] = None,
    makeup_coverage_even: Optional[float] = None,
    makeup_cake_reduce: Optional[float] = None,
    hemoglobin_smooth: Optional[float] = None,
    mole_protect: Optional[float] = None,
    vein_attenuate: Optional[float] = None,
    skin_flatten: Optional[int] = None,
    skin_quantize: Optional[int] = None,
    skin_unify: Optional[int] = None,
    skin_unify_hue: Optional[float] = None,
    skin_hue_unify: Optional[int] = None,
    skin_chroma_even: Optional[int] = None,
    skin_glow: Optional[int] = None,
    eye_enhance: Optional[int] = None,
    catchlight: Optional[int] = None,
    dark_circles: Optional[int] = None,
    undereye_darken_removal: Optional[int] = None,
    undereye_puffiness_reduction: Optional[int] = None,
    eye_sclera_brighten: Optional[int] = None,
    eye_sclera_vessel_remove: Optional[int] = None,
    backdrop_cleanup: Optional[int] = None,
    fabric_wrinkle_smooth: Optional[float] = None,
    eye_iris_saturate: Optional[int] = None,
    eye_iris_hue_shift: Optional[int] = None,
    eye_iris_brightness: Optional[int] = None,
    teeth_whiten: Optional[int] = None,
    lip_enhance: Optional[int] = None,
    lip_tint: Optional[str] = None,
    lip_finish: Optional[str] = None,
    blush: Optional[int] = None,
    slimming: Optional[int] = None,
    reshape_eye_size: Optional[float] = None,
    reshape_eye_distance: Optional[float] = None,
    reshape_nose_width: Optional[float] = None,
    reshape_nose_length: Optional[float] = None,
    reshape_jaw_width: Optional[float] = None,
    reshape_chin_length: Optional[float] = None,
    reshape_mouth_size: Optional[float] = None,
    reshape_smile: Optional[float] = None,
    reshape_forehead: Optional[float] = None,
    reshape_jaw_width_l: Optional[float] = None,
    reshape_jaw_width_r: Optional[float] = None,
    reshape_nose_width_l: Optional[float] = None,
    reshape_nose_width_r: Optional[float] = None,
    reshape_eye_size_l: Optional[float] = None,
    reshape_eye_size_r: Optional[float] = None,
    reshape_neck_width: Optional[float] = None,
    reshape_neck_length: Optional[float] = None,
    hair_enhance: Optional[int] = None,
    hair_deglare: Optional[int] = None,
    hair_ring_position: Optional[int] = None,
    hair_ring_tint: Optional[int] = None,
    hair_remove_flyaways: Optional[int] = None,
    contrast: Optional[int] = None,
    brightness: Optional[int] = None,
    highlights: Optional[int] = None,
    shadows: Optional[int] = None,
    whites: Optional[int] = None,
    blacks: Optional[int] = None,
    clarity: Optional[float] = None,
    clarity_noise_aware: Optional[bool] = None,
    self_blend_mode: Optional[str] = None,
    self_blend_amount: Optional[float] = None,
    self_blend_domain: Optional[str] = None,
    vibrance: Optional[float] = None,
    saturation: Optional[float] = None,
    auto_exposure: Optional[bool] = None,
    bloom: Optional[float] = None,
    bloom_threshold: Optional[float] = None,
    bloom_softness: Optional[float] = None,
    glow: Optional[float] = None,
    vignette: Optional[float] = None,
    sharpen: Optional[float] = None,
    sharpen_radius: Optional[float] = None,
    subject_separation: Optional[float] = None,
    impact: Optional[int] = None,
    fade_toe: Optional[float] = None,
    highlight_drift: Optional[float] = None,
    airy_haze: Optional[float] = None,
    clarity_split_neg: Optional[float] = None,
    clarity_split_pos: Optional[float] = None,
    color_grade: Optional[str] = None,
    grade_intensity: Optional[float] = None,
    color_transfer_intensity: Optional[float] = None,
    chromatic_aberration: Optional[float] = None,
    grain: Optional[float] = None,
    halation: Optional[float] = None,
    lut: Optional[str] = None,
    tonal_curve_strength: Optional[float] = None,
    skin_protect_strength: Optional[float] = None,
    grain_strength: Optional[float] = None,
    highlight_rolloff_strength: Optional[float] = None,
    gamut_compress: Optional[bool] = None,
    saturation_mode: Optional[str] = None,
    hsl_hue_red: Optional[float] = None,
    hsl_sat_red: Optional[float] = None,
    hsl_lum_red: Optional[float] = None,
    hsl_hue_orange: Optional[float] = None,
    hsl_sat_orange: Optional[float] = None,
    hsl_lum_orange: Optional[float] = None,
    hsl_hue_yellow: Optional[float] = None,
    hsl_sat_yellow: Optional[float] = None,
    hsl_lum_yellow: Optional[float] = None,
    hsl_hue_green: Optional[float] = None,
    hsl_sat_green: Optional[float] = None,
    hsl_lum_green: Optional[float] = None,
    hsl_hue_cyan: Optional[float] = None,
    hsl_sat_cyan: Optional[float] = None,
    hsl_lum_cyan: Optional[float] = None,
    hsl_hue_blue: Optional[float] = None,
    hsl_sat_blue: Optional[float] = None,
    hsl_lum_blue: Optional[float] = None,
    hsl_hue_purple: Optional[float] = None,
    hsl_sat_purple: Optional[float] = None,
    hsl_lum_purple: Optional[float] = None,
    hsl_hue_magenta: Optional[float] = None,
    hsl_sat_magenta: Optional[float] = None,
    hsl_lum_magenta: Optional[float] = None,
    calibration_red_hue: Optional[float] = None,
    calibration_red_sat: Optional[float] = None,
    calibration_red_lum: Optional[float] = None,
    calibration_green_hue: Optional[float] = None,
    calibration_green_sat: Optional[float] = None,
    calibration_green_lum: Optional[float] = None,
    calibration_blue_hue: Optional[float] = None,
    calibration_blue_sat: Optional[float] = None,
    calibration_blue_lum: Optional[float] = None,
    film_enable: Optional[bool] = None,
    film_strength: Optional[float] = None,
    film_toe_r: Optional[float] = None,
    film_toe_g: Optional[float] = None,
    film_toe_b: Optional[float] = None,
    film_shoulder_r: Optional[float] = None,
    film_shoulder_g: Optional[float] = None,
    film_shoulder_b: Optional[float] = None,
    film_midpoint: Optional[float] = None,
    film_gamma: Optional[float] = None,
    film_crosstalk_cy_mg: Optional[float] = None,
    film_crosstalk_cy_ye: Optional[float] = None,
    film_crosstalk_mg_ye: Optional[float] = None,
    film_tonemap_strength: Optional[float] = None,
    film_tonemap_toe: Optional[float] = None,
    film_tonemap_shoulder: Optional[float] = None,
    film_skew: Optional[float] = None,
    background_harmonize: Optional[float] = None,
    background_harmonize_mode: Optional[str] = None,
    background_blur: Optional[float] = None,
    lens_blur: Optional[float] = None,
    background_desaturation: Optional[float] = None,
    light_wrap: Optional[float] = None,
    blue_shadow_grade: Optional[float] = None,
    cyan_midtone_grade: Optional[float] = None,
    subject_sharpen: Optional[float] = None,
    matte_black: Optional[float] = None,
    shadow_hue: Optional[float] = None,
    shadow_sat: Optional[float] = None,
    midtone_hue: Optional[float] = None,
    midtone_sat: Optional[float] = None,
    highlight_hue: Optional[float] = None,
    highlight_sat: Optional[float] = None,
    white_balance_kelvin: Optional[int] = None,
    white_balance_tint: Optional[float] = None,
    bw_channel_mixer_r: Optional[int] = None,
    bw_channel_mixer_g: Optional[int] = None,
    bw_channel_mixer_b: Optional[int] = None,
    negative_split_tone_shadow: Optional[float] = None,
    negative_split_tone_highlight: Optional[float] = None,
    hsl_hue_global: Optional[int] = None,
    hsl_sat_global: Optional[int] = None,
    hsl_lum_global: Optional[int] = None,
    ai_denoise: Optional[int] = None,
    ai_sr_scale: Optional[int] = None,
    mv2_eyeshadow: Optional[float] = None,
    mv2_eyeshadow_color: Optional[str] = None,
    mv2_eyeshadow_style: Optional[str] = None,
    mv2_eyeliner: Optional[float] = None,
    mv2_eyeliner_color: Optional[str] = None,
    mv2_eyeliner_style: Optional[str] = None,
    mv2_contour: Optional[float] = None,
    mv2_brows: Optional[float] = None,
    mv2_brows_color: Optional[str] = None,
    mv2_ombre: Optional[bool] = None,
    mv2_ombre_color1: Optional[str] = None,
    mv2_ombre_color2: Optional[str] = None,
    neural_stray_hair_boost: Optional[int] = None,
    neural_defect_boost: Optional[int] = None,
    cosplay_wig_lace_blend: Optional[int] = None,
    cosplay_stockings_smooth: Optional[int] = None,
    cosplay_consistency_strength: Optional[int] = None,
    body_reshape_arm_length: Optional[float] = None,
    body_reshape_leg_length: Optional[float] = None,
    body_reshape_torso_width: Optional[float] = None,
    body_reshape_shoulder_width: Optional[float] = None,
    body_reshape_hip_width: Optional[float] = None,
    auto_body_reshape: Optional[float] = None,
    fast: bool = False,
    style_profile: Optional[StyleProfile] = None,
    debug_dir: Optional[str] = None,
    face_contexts: Optional[List["FaceContext"]] = None,
) -> ProcessingResult:
```

#### Inputs
*   **`img_bgr`** (`np.ndarray`): Input portrait image, expected in standard BGR format (e.g. from `cv2.imread`).
*   **`recipe` / `preset`** (`str`, optional): Built-in preset recipe key (e.g. `"natural"`, `"cosplay"`, `"cyber_doll"`, `"portrait"`, `"pink_dream"`, `"meitu_clone"`, `"xiaohongshu"`, `"xhs_ultrasoft"`, `"xhs_soft_glow"`). Preset is a backward-compatible alias.
*   **`fast`** (`bool`, default: `False`): Fast-preview path. Downsamples the input image to a maximum dimension of 800px *before* landmarker and segmenter runs, saving ~40% overhead on high-res photos.
*   **`style_profile`** (`StyleProfile`, optional): Active custom style overrides loaded from json/library.
*   **`style_ref`** (`np.ndarray`, optional): Reference image used to extract style dynamics on-the-fly.
*   **`debug_dir`** (`str`, optional): Directory path where intermediate masks and stages will be written for debugging.
*   **`face_contexts`** (`Optional[List[FaceContext]]`, optional): Reuse cached per-face detection/parsing data from a previous `process()` call. When provided, the detection and parsing stages are skipped and the cached face data is used directly. Pass back `result.face_contexts` from a prior call to skip expensive re-detection on follow-up passes.

#### Parameter Overrides (grouped logically)
Passing an explicit value override to these parameters takes precedence over the active recipe's defaults. Pass `None` to fallback.

##### **Skin Smoothing & Detail**

*   **`smooth`** (Type: `int`, Default: `30`, Range: `0` to `100`, Recipe key: `frequency.smooth`): Adjusts the smooth parameter.
*   **`mid_reduction`** (Type: `float`, Default: `0.35`, Range: `0.0` to `1.0`, Recipe key: `frequency.mid_reduction`): Adjusts the mid reduction parameter.
*   **`blotch_reduction`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `frequency.blotch_reduction`): Adjusts the blotch reduction parameter.
*   **`texture_opacity`** (Type: `float`, Default: `1.0`, Range: `0.0` to `1.0`, Recipe key: `texture.opacity`): Adjusts the texture opacity parameter.
*   **`pore_synthesis`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `texture.pore_synthesis`): Adjusts the pore synthesis parameter.
*   **`nose_smooth`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `frequency.nose_smooth`): Adjusts the nose smooth parameter.
*   **`regional_modulation`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `frequency.regional_modulation`): Adjusts the regional modulation parameter.
*   **`smooth_engine`** (Type: `str`, Default: `guided`, Range: None, Recipe key: `frequency.smooth_engine`): Adjusts the smooth engine parameter.
*   **`freckle_removal`** (Type: `float`, Default: `0.0`, Range: `0.0` to `100.0`, Recipe key: `frequency.freckle_removal`): Adjusts the freckle removal parameter.
*   **`freckle_preserve_mask`** (Type: `float`, Default: `None`, Range: None, Recipe key: None): Adjusts the freckle preserve mask parameter.
*   **`micro_restore`** (Type: `int`, Default: `20`, Range: `0` to `50`, Recipe key: `micro_restore`): Adjusts the micro restore parameter.
*   **`micro_dodge_burn`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.micro_db`): Adjusts the micro dodge burn parameter.
*   **`redness_even`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.redness_even`): Adjusts the redness even parameter.
*   **`whiten_hue_stable`** (Type: `bool`, Default: `False`, Range: None, Recipe key: `skin.whiten_hue_stable`): Boolean flag to toggle whiten hue stable.
*   **`whiten`** (Type: `int`, Default: `10`, Range: `0` to `100`, Recipe key: `skin.rosy`): Adjusts the whiten parameter.
*   **`equalize`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.equalize`): Adjusts the equalize parameter.
*   **`blemish`** (Type: `int`, Default: `30`, Range: None, Recipe key: None): Alias/mirror of the `smooth` slider.
*   **`whiten_tone`** (Type: `str`, Default: `rosy`, Range: None, Recipe key: `skin.porcelain`): Adjusts the whiten tone parameter.
*   **`white_costume_lift`** (Type: `bool`, Default: `False`, Range: None, Recipe key: `white_costume_lift`): Boolean flag to toggle white costume lift.
*   **`dodge_burn`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `dodge_burn`): Adjusts the dodge burn parameter.
*   **`relight`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.relight`): Adjusts the relight parameter.
*   **`relight_azimuth`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `skin.relight_azimuth`): Adjusts the relight azimuth parameter.
*   **`relight_elevation`** (Type: `float`, Default: `30.0`, Range: None, Recipe key: `skin.relight_elevation`): Adjusts the relight elevation parameter.
*   **`shine_removal`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.shine_removal`): Adjusts the shine removal parameter.
*   **`wrinkle_soften`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.wrinkle_soften`): Adjusts the wrinkle soften parameter.
*   **`wrinkle_soften_forehead`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `skin.wrinkle_soften_forehead`): Adjusts the wrinkle soften forehead parameter.
*   **`wrinkle_soften_nasolabial`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `skin.wrinkle_soften_nasolabial`): Adjusts the wrinkle soften nasolabial parameter.
*   **`wrinkle_soften_neck`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `skin.wrinkle_soften_neck`): Adjusts the wrinkle soften neck parameter.
*   **`texture_transplant`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.texture_transplant`): Adjusts the texture transplant parameter.
*   **`albedo_even`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin.albedo_even`): Adjusts the albedo even parameter.
*   **`hemoglobin_smooth`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin.hemoglobin_smooth`): Adjusts the hemoglobin smooth parameter.
*   **`mole_protect`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin.mole_protect`): Adjusts the mole protect parameter.
*   **`vein_attenuate`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin.vein_attenuate`): Adjusts the vein attenuate parameter.
*   **`skin_flatten`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.flatten`): Adjusts the skin flatten parameter.
*   **`skin_quantize`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.quantize`): Adjusts the skin quantize parameter.
*   **`skin_unify`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.unify`): Adjusts the skin unify parameter.
*   **`skin_hue_unify`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.hue_unify`): Adjusts the skin hue unify parameter.
*   **`skin_chroma_even`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.chroma_even`): Adjusts the skin chroma even parameter.
*   **`skin_glow`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.glow`): Adjusts the skin glow parameter.
*   **`skin_sss`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.sss`): Adds screen-space subsurface scattering for a translucent game-character skin finish.
*   **`mask_feather_mode`** (Type: `str`, Default: `gaussian`, Recipe key: `mask.feather_mode`): Selects `gaussian` mask feathering or `guided` edge-aware refinement for fine hair, wig, and lash boundaries.
*   **`fabric_wrinkle_smooth`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `fabric.wrinkle_smooth`): Adjusts the fabric wrinkle smooth parameter.
*   **`blush`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `blush`): Adjusts the blush parameter.
*   **`hair_enhance`** (Type: `int`, Default: `5`, Range: `0` to `100`, Recipe key: `hair.shine`): Adjusts the hair enhance parameter.
*   **`hair_deglare`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `hair.deglare`): Adjusts the hair deglare parameter.
*   **`hair_ring_position`** (Type: `int`, Default: `30`, Range: `0` to `100`, Recipe key: `hair.ring_position`): Adjusts the hair ring position parameter.
*   **`hair_ring_tint`** (Type: `int`, Default: `40`, Range: `0` to `100`, Recipe key: `hair.ring_tint`): Adjusts the hair ring tint parameter.
*   **`hair_remove_flyaways`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `hair.remove_flyaways`): Adjusts the hair remove flyaways parameter.
*   **`skin_protect_strength`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin_protect`): Adjusts the skin protect strength parameter.
*   **`hsl_hue_red`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.red`): Adjusts the hsl hue red parameter.
*   **`hsl_sat_red`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.red`): Adjusts the hsl sat red parameter.
*   **`hsl_lum_red`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.red`): Adjusts the hsl lum red parameter.
*   **`hsl_hue_orange`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.orange`): Adjusts the hsl hue orange parameter.
*   **`hsl_sat_orange`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.orange`): Adjusts the hsl sat orange parameter.
*   **`hsl_lum_orange`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.orange`): Adjusts the hsl lum orange parameter.
*   **`hsl_hue_yellow`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.yellow`): Adjusts the hsl hue yellow parameter.
*   **`hsl_sat_yellow`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.yellow`): Adjusts the hsl sat yellow parameter.
*   **`hsl_lum_yellow`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.yellow`): Adjusts the hsl lum yellow parameter.
*   **`hsl_hue_green`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.green`): Adjusts the hsl hue green parameter.
*   **`hsl_sat_green`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.green`): Adjusts the hsl sat green parameter.
*   **`hsl_lum_green`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.green`): Adjusts the hsl lum green parameter.
*   **`hsl_hue_cyan`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.cyan`): Adjusts the hsl hue cyan parameter.
*   **`hsl_sat_cyan`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.cyan`): Adjusts the hsl sat cyan parameter.
*   **`hsl_lum_cyan`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.cyan`): Adjusts the hsl lum cyan parameter.
*   **`hsl_hue_blue`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.blue`): Adjusts the hsl hue blue parameter.
*   **`hsl_sat_blue`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.blue`): Adjusts the hsl sat blue parameter.
*   **`hsl_lum_blue`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.blue`): Adjusts the hsl lum blue parameter.
*   **`hsl_hue_purple`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.purple`): Adjusts the hsl hue purple parameter.
*   **`hsl_sat_purple`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.purple`): Adjusts the hsl sat purple parameter.
*   **`hsl_lum_purple`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.purple`): Adjusts the hsl lum purple parameter.
*   **`hsl_hue_magenta`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `hsl_adjustments.hue.magenta`): Adjusts the hsl hue magenta parameter.
*   **`hsl_sat_magenta`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.saturation.magenta`): Adjusts the hsl sat magenta parameter.
*   **`hsl_lum_magenta`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `hsl_adjustments.luminance.magenta`): Adjusts the hsl lum magenta parameter.
*   **`calibration_red_lum`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `calibration.red.lum`): Adjusts the calibration red lum parameter.
*   **`calibration_green_lum`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `calibration.green.lum`): Adjusts the calibration green lum parameter.
*   **`calibration_blue_lum`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `calibration.blue.lum`): Adjusts the calibration blue lum parameter.
*   **`blue_shadow_grade`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.blue_shadow_grade`): Adjusts the blue shadow grade parameter.
*   **`cyan_midtone_grade`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.cyan_midtone_grade`): Adjusts the cyan midtone grade parameter.
*   **`subject_sharpen`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.subject_sharpen`): Adjusts the subject sharpen parameter.
*   **`matte_black`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.matte_black`): Adjusts the matte black parameter.
*   **`auto_body_reshape`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `body_reshape.auto`): Adjusts the auto body reshape parameter.

##### **Face Features (Eyes, Lips, Teeth, Nose)**

*   **`undereye_shadow_strength`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `eyes.undereye_shadow_strength`): Adjusts the undereye shadow strength parameter.
*   **`nose_blush`** (Type: `bool`, Default: `False`, Range: None, Recipe key: `nose_blush`): Boolean flag to toggle nose blush.
*   **`under_eye_blush`** (Type: `bool`, Default: `False`, Range: None, Recipe key: `under_eye_blush`): Boolean flag to toggle under eye blush.
*   **`nose_restore`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.nose_restore`): Adjusts the nose restore parameter.
*   **`eye_enhance`** (Type: `int`, Default: `5`, Range: `0` to `100`, Recipe key: `eyes.whites`): Adjusts the eye enhance parameter.
*   **`catchlight`** (Type: `int`, Default: `5`, Range: `0` to `100`, Recipe key: `eyes.catchlight`): Adjusts the catchlight parameter.
*   **`dark_circles`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `eyes.dark_circles`): Adjusts the dark circles parameter.
*   **`undereye_darken_removal`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `undereye.darken_removal`): Adjusts the undereye darken removal parameter.
*   **`undereye_puffiness_reduction`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `undereye.puffiness_reduction`): Adjusts the undereye puffiness reduction parameter.
*   **`eye_sclera_brighten`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `eye.sclera_brighten`): Adjusts the eye sclera brighten parameter.
*   **`eye_sclera_vessel_remove`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `eyes.sclera_vessel_remove`): Adjusts the eye sclera vessel remove parameter.
*   **`eye_gate`** (Type: `bool`, Default: `True`, Recipe key: `eyes.gate`): Per-eye occlusion gate — skip enhancing eyes detected as closed/occluded (prevents painting an iris onto hair or a closed lid).
*   **`eye_iris_saturate`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `eye.iris_saturate`): Adjusts the eye iris saturate parameter.
*   **`eye_iris_hue_shift`** (Type: `int`, Default: `0`, Range: `-30` to `30`, Recipe key: `eye.iris_hue_shift`): Adjusts the eye iris hue shift parameter.
*   **`eye_iris_brightness`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `eye.iris_brightness`): Adjusts the eye iris brightness parameter.
*   **`teeth_whiten`** (Type: `int`, Default: `5`, Range: `0` to `100`, Recipe key: `eyes.whites`): Adjusts the teeth whiten parameter.
*   **`lip_enhance`** (Type: `int`, Default: `5`, Range: `0` to `100`, Recipe key: `lips.gloss`): Adjusts the lip enhance parameter.
*   **`lip_tint`** (Type: `str`, Default: `None`, Range: None, Recipe key: `lips.tint`): Adjusts the lip tint parameter.
*   **`lip_finish`** (Type: `str`, Default: `gloss`, Range: None, Recipe key: `lip_finish`): Adjusts the lip finish parameter.

##### **Global Tonal Adjustments**

*   **`shadow_lift`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.shadow_lift`): Adjusts the shadow lift parameter.
*   **`face_exposure`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.face_exposure`): Adjusts the face exposure parameter.
*   **`contrast`** (Type: `int`, Default: `0.0`, Range: `-50` to `50`, Recipe key: `contrast`): Adjusts the contrast parameter.
*   **`brightness`** (Type: `int`, Default: `None`, Range: `-50` to `50`, Recipe key: `brightness`): Adjusts the brightness parameter.
*   **`highlights`** (Type: `int`, Default: `None`, Range: `-100` to `100`, Recipe key: `highlights`): Adjusts the highlights parameter.
*   **`shadows`** (Type: `int`, Default: `None`, Range: `-100` to `100`, Recipe key: `shadows`): Adjusts the shadows parameter.
*   **`whites`** (Type: `int`, Default: `None`, Range: `-100` to `100`, Recipe key: `whites`): Adjusts the whites parameter.
*   **`blacks`** (Type: `int`, Default: `None`, Range: `-100` to `100`, Recipe key: `blacks`): Adjusts the blacks parameter.
*   **`clarity`** (Type: `float`, Default: `0.0`, Range: `-50` to `50`, Recipe key: `clarity`): Adjusts the clarity parameter.
*   **`clarity_noise_aware`** (Type: `bool`, Default: `False`, Recipe key: caller-only): Explicitly opts into the P6 confidence-qualified noise-floor gate. Legacy clarity remains the default; this does not denoise, classify semantic regions, or claim a universal noise model.
*   **`self_blend_mode`** (Type: `str`, Default: `None`, Range: analytical P5 modes, Recipe key: caller-only): Applies one explicitly selected analytical duplicate-layer tone operator; omitted means no-op.
*   **`self_blend_amount`** (Type: `float`, Default: `None`/`1.0` when a mode is selected, Range: `0.0` to `1.0`, Recipe key: caller-only): Sets opaque-layer opacity for the selected self-blend operator.
*   **`self_blend_domain`** (Type: `str`, Default: `encoded`, Range: `encoded` or `linear`, Recipe key: caller-only): Selects the analytical processing domain; this is not evidence of Photoshop's working domain.
*   **`vibrance`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `vibrance`): Adjusts the vibrance parameter.
*   **`saturation`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `saturation`): Adjusts the saturation parameter.
*   **`auto_exposure`** (Type: `bool`, Default: `False`, Range: None, Recipe key: `auto_exposure`): Boolean flag to toggle auto exposure.
*   **`fade_toe`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `finish.fade_toe`): Adjusts the fade toe parameter.
*   **`highlight_drift`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `finish.highlight_drift`): Adjusts the highlight drift parameter.
*   **`airy_haze`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `finish.airy_haze`): Adjusts the airy haze parameter.
*   **`clarity_split_neg`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `finish.clarity_split_neg`): Adjusts the clarity split neg parameter.
*   **`clarity_split_pos`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `finish.clarity_split_pos`): Adjusts the clarity split pos parameter.

##### **Screen Effects & Lens Finish**

*   **`specular_bloom`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `specular_bloom`): Adjusts the specular bloom parameter.
*   **`specular_bloom_tone`** (Type: `str`, Default: `rosy`, Range: None, Recipe key: `specular_bloom_tone`): Adjusts the specular bloom tone parameter.
*   **`specular_finish`** (Type: `str`, Default: `matte`, Range: None, Recipe key: `skin.specular_finish`): Adjusts the specular finish parameter.
*   **`specular_finish_strength`** (Type: `float`, Default: `0.5`, Range: `0.0` to `1.0`, Recipe key: `skin.specular_finish_strength`): Adjusts the specular finish strength parameter.
*   **`specular_recolor`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin.specular_recolor`): Adjusts the specular recolor parameter.
*   **`bloom`** (Type: `float`, Default: `0.0`, Range: `0.0` to `100.0`, Recipe key: `bloom.opacity`): Adjusts the bloom parameter.
*   **`bloom_threshold`** (Type: `float`, Default: `210.0`, Range: None, Recipe key: `bloom.threshold`): Adjusts the bloom threshold parameter.
*   **`bloom_softness`** (Type: `float`, Default: `30.0`, Range: None, Recipe key: `bloom.softness`): Adjusts the bloom softness parameter.
*   **`glow`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `glow`): Adjusts the glow parameter.
*   **`vignette`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `vignette`): Adjusts the vignette parameter.
*   **`sharpen`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `sharpen`): Adjusts the sharpen parameter.
*   **`sharpen_radius`** (Type: `float`, Default: `1.0`, Range: None, Recipe key: `sharpen_radius`): Adjusts the sharpen radius parameter.
*   **`impact`** (Type: `int`, Default: `0.0`, Range: `0` to `100`, Recipe key: `finish.impact`): Adjusts the impact parameter.
*   **`chromatic_aberration`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `chromatic_aberration`): Adjusts the chromatic aberration parameter.
*   **`grain`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `grain`): Adjusts the grain parameter.
*   **`halation`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `halation`): Adjusts the halation parameter.

##### **Color Grading & LUTs**

*   **`color_grade`** (Type: `str`, Default: `none`, Range: None, Recipe key: `color_harmony.preset`): Adjusts the color grade parameter.
*   **`grade_intensity`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `color_harmony.amount`): Adjusts the grade intensity parameter.
*   **`lut`** (Type: `str`, Default: `none`, Range: None, Recipe key: `lut`): Adjusts the lut parameter.
*   **`gamut_compress`** (Type: `bool`, Default: `True`, Range: None, Recipe key: `gamut_compress`): Boolean flag to toggle gamut compress.
*   **`saturation_mode`** (Type: `str`, Default: `additive`, Range: None, Recipe key: `saturation_mode`): Adjusts the saturation mode parameter.

##### **Film Density & Simulation**

*   **`tonal_curve_strength`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `tonal_curve_strength`): Adjusts the tonal curve strength parameter.
*   **`grain_strength`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `grain_strength`): Adjusts the grain strength parameter.
*   **`highlight_rolloff_strength`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `highlight_rolloff`): Adjusts the highlight rolloff strength parameter.
*   **`film_enable`** (Type: `bool`, Default: `False`, Range: None, Recipe key: `film.enable`): Boolean flag to toggle film enable.
*   **`film_strength`** (Type: `float`, Default: `1.0`, Range: `0.0` to `1.0`, Recipe key: `film.strength`): Adjusts the film strength parameter.
*   **`film_toe_r`** (Type: `float`, Default: `0.1`, Range: `0.0` to `0.5`, Recipe key: `film.toe.r`): Adjusts the film toe r parameter.
*   **`film_toe_g`** (Type: `float`, Default: `0.1`, Range: `0.0` to `0.5`, Recipe key: `film.toe.g`): Adjusts the film toe g parameter.
*   **`film_toe_b`** (Type: `float`, Default: `0.1`, Range: `0.0` to `0.5`, Recipe key: `film.toe.b`): Adjusts the film toe b parameter.
*   **`film_shoulder_r`** (Type: `float`, Default: `0.1`, Range: `0.0` to `0.5`, Recipe key: `film.shoulder.r`): Adjusts the film shoulder r parameter.
*   **`film_shoulder_g`** (Type: `float`, Default: `0.1`, Range: `0.0` to `0.5`, Recipe key: `film.shoulder.g`): Adjusts the film shoulder g parameter.
*   **`film_shoulder_b`** (Type: `float`, Default: `0.1`, Range: `0.0` to `0.5`, Recipe key: `film.shoulder.b`): Adjusts the film shoulder b parameter.
*   **`film_midpoint`** (Type: `float`, Default: `0.5`, Range: `0.0` to `1.0`, Recipe key: `film.midpoint`): Adjusts the film midpoint parameter.
*   **`film_gamma`** (Type: `float`, Default: `1.0`, Range: `0.5` to `2.5`, Recipe key: `film.gamma`): Adjusts the film gamma parameter.
*   **`film_crosstalk_cy_mg`** (Type: `float`, Default: `0.06`, Range: `-0.15` to `0.15`, Recipe key: `film.crosstalk.cy_mg`): Adjusts the film crosstalk cy mg parameter.
*   **`film_crosstalk_cy_ye`** (Type: `float`, Default: `0.03`, Range: `-0.15` to `0.15`, Recipe key: `film.crosstalk.cy_ye`): Adjusts the film crosstalk cy ye parameter.
*   **`film_crosstalk_mg_ye`** (Type: `float`, Default: `0.02`, Range: `-0.15` to `0.15`, Recipe key: `film.crosstalk.mg_ye`): Adjusts the film crosstalk mg ye parameter.
*   **`film_tonemap_strength`** (Type: `float`, Default: `0.7`, Range: `0.0` to `1.0`, Recipe key: `film.tonemap.strength`): Adjusts the film tonemap strength parameter.
*   **`film_tonemap_toe`** (Type: `float`, Default: `0.1`, Range: `0.0` to `0.5`, Recipe key: `film.tonemap.toe`): Adjusts the film tonemap toe parameter.
*   **`film_tonemap_shoulder`** (Type: `float`, Default: `0.15`, Range: `0.0` to `0.5`, Recipe key: `film.tonemap.shoulder`): Adjusts the film tonemap shoulder parameter.
*   **`film_skew`** (Type: `float`, Default: `0.3`, Range: `0.0` to `1.0`, Recipe key: `film.tonemap.skew`): Adjusts the film skew parameter.

##### **Background Harmonization & Placement**

*   **`background_harmonize`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `harmony.background_harmonize`): Adjusts the background harmonize parameter.
*   **`background_harmonize_mode`** (Type: `float`, Default: `split`, Range: None, Recipe key: `harmony.background_harmonize_mode`): Adjusts the background harmonize mode parameter.
*   **`background_blur`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.background_blur`): Adjusts the background blur parameter.
*   **`background_desaturation`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.background_desaturation`): Adjusts the background desaturation parameter.
*   **`light_wrap`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.light_wrap`): Adjusts the light wrap parameter.

##### **Background Replacement & Separation**

*   **`backdrop_cleanup`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `background.backdrop_cleanup`): Adjusts the backdrop cleanup parameter.
*   **`subject_separation`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `subject_separation`): Adjusts the subject separation parameter.
*   **`lens_blur`** (Type: `float`, Default: `0.0`, Range: `0` to `100`, Recipe key: `background.lens_blur`): Adjusts the lens blur parameter.

##### **Split Toning (LAB 3-Way)**

*   **`skin_unify_hue`** (Type: `float`, Default: `-1.0`, Range: `-1.0` to `360.0`, Recipe key: `skin.unify_hue`): Adjusts the skin unify hue parameter.
*   **`color_transfer_intensity`** (Type: `float`, Default: `1.0`, Range: `0.0` to `1.0`, Recipe key: `color_transfer_intensity`): Adjusts the color transfer intensity parameter.
*   **`calibration_red_hue`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `calibration.red.hue`): Adjusts the calibration red hue parameter.
*   **`calibration_red_sat`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `calibration.red.sat`): Adjusts the calibration red sat parameter.
*   **`calibration_green_hue`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `calibration.green.hue`): Adjusts the calibration green hue parameter.
*   **`calibration_green_sat`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `calibration.green.sat`): Adjusts the calibration green sat parameter.
*   **`calibration_blue_hue`** (Type: `float`, Default: `0.0`, Range: `-180` to `180`, Recipe key: `calibration.blue.hue`): Adjusts the calibration blue hue parameter.
*   **`calibration_blue_sat`** (Type: `float`, Default: `0.0`, Range: `-100` to `100`, Recipe key: `calibration.blue.sat`): Adjusts the calibration blue sat parameter.
*   **`shadow_hue`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `shadow_hue`): Adjusts the shadow hue parameter.
*   **`shadow_sat`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `shadow_sat`): Adjusts the shadow sat parameter.
*   **`midtone_hue`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `midtone_hue`): Adjusts the midtone hue parameter.
*   **`midtone_sat`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `midtone_sat`): Adjusts the midtone sat parameter.
*   **`highlight_hue`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `highlight_hue`): Adjusts the highlight hue parameter.
*   **`highlight_sat`** (Type: `float`, Default: `0.0`, Range: None, Recipe key: `highlight_sat`): Adjusts the highlight sat parameter.
*   **`negative_split_tone_shadow`** (Type: `float`, Default: `0.0`, Range: `0.0` to `100.0`, Recipe key: `negative_split_tone_shadow`): Adjusts the negative split tone shadow parameter.
*   **`negative_split_tone_highlight`** (Type: `float`, Default: `0.0`, Range: `0.0` to `100.0`, Recipe key: `negative_split_tone_highlight`): Adjusts the negative split tone highlight parameter.

##### **White Balance & B&W Mixer**

*   **`white_balance_kelvin`** (Type: `int`, Default: `6500`, Range: `2000` to `12000`, Recipe key: `white_balance_kelvin`): Adjusts the white balance kelvin parameter.
*   **`white_balance_tint`** (Type: `float`, Default: `0.0`, Range: `-100.0` to `100.0`, Recipe key: `white_balance_tint`): Adjusts the white balance tint parameter.
*   **`bw_channel_mixer_r`** (Type: `int`, Default: `30`, Range: `-100` to `200`, Recipe key: `bw_channel_mixer_r`): Adjusts the bw channel mixer r parameter.
*   **`bw_channel_mixer_g`** (Type: `int`, Default: `59`, Range: `-100` to `200`, Recipe key: `bw_channel_mixer_g`): Adjusts the bw channel mixer g parameter.
*   **`bw_channel_mixer_b`** (Type: `int`, Default: `11`, Range: `-100` to `200`, Recipe key: `bw_channel_mixer_b`): Adjusts the bw channel mixer b parameter.
*   **`hsl_hue_global`** (Type: `int`, Default: `0`, Range: `-100` to `100`, Recipe key: `hsl_hue_global`): Adjusts the hsl hue global parameter.
*   **`hsl_sat_global`** (Type: `int`, Default: `0`, Range: `-100` to `100`, Recipe key: `hsl_sat_global`): Adjusts the hsl sat global parameter.
*   **`hsl_lum_global`** (Type: `int`, Default: `0`, Range: `-100` to `100`, Recipe key: `hsl_lum_global`): Adjusts the hsl lum global parameter.

##### **AI Enhance & Denoise**

*   **`ai_denoise`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `ai.denoise`): Adjusts the ai denoise parameter.
*   **`ai_sr_scale`** (Type: `int`, Default: `1`, Range: `1` to `4`, Recipe key: None): Adjusts the ai sr scale parameter.

##### **Makeup V2 Synthesis**

*   **`makeup_coverage_even`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin.makeup_coverage_even`): Adjusts the makeup coverage even parameter.
*   **`makeup_cake_reduce`** (Type: `float`, Default: `0.0`, Range: `0.0` to `1.0`, Recipe key: `skin.makeup_cake_reduce`): Adjusts the makeup cake reduce parameter.
*   **`mv2_eyeshadow`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `makeup_v2.eyeshadow`): Adjusts the mv2 eyeshadow parameter.
*   **`mv2_eyeshadow_color`** (Type: `float`, Default: `rose`, Range: None, Recipe key: `makeup_v2.eyeshadow_color`): Adjusts the mv2 eyeshadow color parameter.
*   **`mv2_eyeshadow_style`** (Type: `float`, Default: `natural`, Range: None, Recipe key: `makeup_v2.eyeshadow_style`): Adjusts the mv2 eyeshadow style parameter.
*   **`mv2_eyeliner`** (Type: `float`, Default: `0`, Range: `0` to `10`, Recipe key: `makeup_v2.eyeliner`): Adjusts the mv2 eyeliner parameter.
*   **`mv2_eyeliner_color`** (Type: `float`, Default: `black`, Range: None, Recipe key: `makeup_v2.eyeliner_color`): Adjusts the mv2 eyeliner color parameter.
*   **`mv2_eyeliner_style`** (Type: `float`, Default: `classic`, Range: None, Recipe key: `makeup_v2.eyeliner_style`): Adjusts the mv2 eyeliner style parameter.
*   **`mv2_contour`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `makeup_v2.contour`): Adjusts the mv2 contour parameter.
*   **`mv2_brows`** (Type: `float`, Default: `0`, Range: `0` to `10`, Recipe key: `makeup_v2.brows`): Adjusts the mv2 brows parameter.
*   **`mv2_brows_color`** (Type: `float`, Default: `brown`, Range: None, Recipe key: `makeup_v2.brows_color`): Adjusts the mv2 brows color parameter.
*   **`mv2_ombre`** (Type: `bool`, Default: `False`, Range: None, Recipe key: `makeup_v2.ombre`): Boolean flag to toggle mv2 ombre.
*   **`mv2_ombre_color1`** (Type: `float`, Default: `red`, Range: None, Recipe key: `makeup_v2.ombre_color1`): Adjusts the mv2 ombre color1 parameter.
*   **`mv2_ombre_color2`** (Type: `float`, Default: `pink`, Range: None, Recipe key: `makeup_v2.ombre_color2`): Adjusts the mv2 ombre color2 parameter.

##### **Neural Boosters**

*   **`neural_stray_hair_boost`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `neural.stray_hair_boost`): Adjusts the neural stray hair boost parameter.
*   **`neural_defect_boost`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `neural.defect_boost`): Adjusts the neural defect boost parameter.

##### **Cosplay Moat & Stockings**

*   **`cosplay_wig_lace_blend`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `cosplay.wig_lace_blend`): Adjusts the cosplay wig lace blend parameter.
*   **`cosplay_stockings_smooth`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `cosplay.stockings_smooth`): Adjusts the cosplay stockings smooth parameter.
*   **`cosplay_consistency_strength`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `cosplay.consistency_strength`): Adjusts the cosplay consistency strength parameter.

##### **Body & Face Reshaping**

*   **`sculpt`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `skin.sculpt`): Adjusts the sculpt parameter.
*   **`body_smooth`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `body_skin.smooth`): Adjusts the body smooth parameter.
*   **`body_equalize`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `body_skin.equalize`): Adjusts the body equalize parameter.
*   **`body_whiten`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `body_skin.whiten`): Adjusts the body whiten parameter.
*   **`body_match_face`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `body_skin.match_face`): Adjusts the body match face parameter.
*   **`body_relight`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `body_skin.relight`): Adjusts the body relight parameter.
*   **`body_dodge_burn`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `body_skin.dodge_burn`): Adjusts the body dodge burn parameter.
*   **`body_shadow_lift`** (Type: `float`, Default: `0`, Range: `0` to `100`, Recipe key: `body_skin.shadow_lift`): Adjusts the body shadow lift parameter.
*   **`slimming`** (Type: `int`, Default: `0`, Range: `0` to `100`, Recipe key: `slimming`): Adjusts the slimming parameter.
*   **`reshape_eye_size`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.eye_size`): Adjusts the reshape eye size parameter.
*   **`reshape_eye_distance`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.eye_distance`): Adjusts the reshape eye distance parameter.
*   **`reshape_nose_width`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.nose_width`): Adjusts the reshape nose width parameter.
*   **`reshape_nose_length`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.nose_length`): Adjusts the reshape nose length parameter.
*   **`reshape_jaw_width`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.jaw_width`): Adjusts the reshape jaw width parameter.
*   **`reshape_chin_length`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.chin_length`): Adjusts the reshape chin length parameter.
*   **`reshape_mouth_size`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.mouth_size`): Adjusts the reshape mouth size parameter.
*   **`reshape_smile`** (Type: `float`, Default: `0`, Range: `-20` to `30`, Recipe key: `reshape.smile`): Adjusts the reshape smile parameter.
*   **`reshape_forehead`** (Type: `float`, Default: `0`, Range: `-30` to `30`, Recipe key: `reshape.forehead`): Adjusts the reshape forehead parameter.
*   **`reshape_jaw_width_l`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.jaw_width_l`): Adjusts the reshape jaw width l parameter.
*   **`reshape_jaw_width_r`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.jaw_width_r`): Adjusts the reshape jaw width r parameter.
*   **`reshape_nose_width_l`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.nose_width_l`): Adjusts the reshape nose width l parameter.
*   **`reshape_nose_width_r`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.nose_width_r`): Adjusts the reshape nose width r parameter.
*   **`reshape_eye_size_l`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.eye_size_l`): Adjusts the reshape eye size l parameter.
*   **`reshape_eye_size_r`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.eye_size_r`): Adjusts the reshape eye size r parameter.
*   **`reshape_neck_width`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.neck_width`): Adjusts the reshape neck width parameter.
*   **`reshape_neck_length`** (Type: `float`, Default: `0`, Range: `-50` to `50`, Recipe key: `reshape.neck_length`): Adjusts the reshape neck length parameter.
*   **`body_reshape_arm_length`** (Type: `float`, Default: `50.0`, Range: `0` to `100`, Recipe key: `body_reshape.arm_length`): Adjusts the body reshape arm length parameter.
*   **`body_reshape_leg_length`** (Type: `float`, Default: `50.0`, Range: `0` to `100`, Recipe key: `body_reshape.leg_length`): Adjusts the body reshape leg length parameter.
*   **`body_reshape_torso_width`** (Type: `float`, Default: `50.0`, Range: `0` to `100`, Recipe key: `body_reshape.torso_width`): Adjusts the body reshape torso width parameter.
*   **`body_reshape_shoulder_width`** (Type: `float`, Default: `50.0`, Range: `0` to `100`, Recipe key: `body_reshape.shoulder_width`): Adjusts the body reshape shoulder width parameter.
*   **`body_reshape_hip_width`** (Type: `float`, Default: `50.0`, Range: `0` to `100`, Recipe key: `body_reshape.hip_width`): Adjusts the body reshape hip width parameter.

---

## 3. Convenience Function: `retouch()`

### Recently added advanced parameters

The registry also accepts these keyword overrides through `process(..., **kwargs)`:

* `corneal_shading`
* `film_highlight_purity`
* `flyaway_cleanup`
* `hb_even`
* `hb_shift`
* `heal_engine`
* `mark_policy`
* `micro_grain`
* `purple_fringing`
* `split_toning`

For simple scripts or backward-compatibility, you can use the `retouch` module-level wrapper. It handles the instantiation and teardown of the engine automatically.

```python
def retouch(
    img_bgr: np.ndarray,
    smooth: float = 50,
    whiten: float = 30,
    eye_enhance: float = 30,
    contrast: float = 0,
    preset: Optional[str] = None,
    **kwargs,
) -> ProcessingResult:
```

### Usage
```python
import cv2
from retouch import retouch

img = cv2.imread("face.jpg")
# Automatically spins up and shuts down a RetouchEngine instance
out = retouch(img, preset="cosplay", smooth=65, whiten=15)
cv2.imwrite("output.jpg", out)
```

---

## 4. Return Object: `ProcessingResult`

The return value of both `RetouchEngine.process()` and `retouch()` is a rich `ProcessingResult` object. 

### Legacy-Compatible Array Representation
`ProcessingResult` inherits from `numpy.ndarray` and points to the processed BGR image array. This guarantees that **any legacy code, OpenCV methods, or visualization libraries that expect a standard numpy image array will accept it immediately** without modification.

```python
result = engine.process(img)

# Directly access shape, slices, and dtype like a normal numpy array
h, w, c = result.shape
crop = result[100:200, 100:200]
print(result.dtype)  # dtype('uint8')
```

### Extended Debug & Performance Metadata

*   **`result.image`** (`np.ndarray`): The raw output BGR image array.
*   **`result.face_count`** (`int`): Count of unique faces detected and processed.
*   **`result.face_contexts`** (`Optional[List[FaceContext]]`): Cached per-face detection/parsing results (`None` when no faces were processed or when caching was disabled). Pass this list back into `process(..., face_contexts=...)` to skip detection/parsing on follow-up passes.
*   **`result.skin_mask`** (`np.ndarray`): A normalized single-channel float32 mask `[0.0, 1.0]` of the processed skin regions.
*   **`result.skin_hair_mask`** (`np.ndarray`): A combined float32 mask of the skin and hair regions.
*   **`result.lips_mask`** (`np.ndarray`): A mask of the lips region.
*   **`result.sharpen_mask`** (`np.ndarray`): The mask applied during the final sharpening stage.
*   **`result.params`** (`ProcessingContext`): The fully-resolved `ProcessingContext` containing all parameters (recipe values + manual overrides) applied.
*   **`result.timings`** (`Dict[str, float]`): Timing metrics (in milliseconds) for each stage of the pipeline. Example:
    ```python
    print(result.timings)
    # Output:
    # {
    #   'detection': 14.5,
    #   'reshape': 8.2,
    #   'per_face': 120.3,
    #   'global': 5.4,
    #   'grading': 24.1
    # }
    ```

---

## 5. Research-derived opt-in APIs

These APIs are explicit leaves and are not enabled by existing recipes.
They preserve the research evidence boundaries: neither operation claims to
recover hidden ground truth, and callers must supply any reviewed supports or
references required by the operation.

### Bounded makeup attenuation (P4)

```python
from retouch.makeup_unmix import (
    BoundedAttenuationResult,
    apply_bounded_makeup_attenuation,
)

result = apply_bounded_makeup_attenuation(
    image_bgr,
    support_mask,
    reference_bgr,
    reference_mask,
    category="blush",
    strength=0.25,
    protected_mask=protected_mask,
)
```

The same leaf is available through `RetouchEngine.apply_bounded_makeup_attenuation(...)`;
it remains explicit and does not run as part of `process()` unless a future
caller supplies a reviewed operation request.

This is reference-conditioned appearance attenuation, not bare-skin
reconstruction. The current supported category is confirmed translucent
blush/color; unsupported or protected cosmetics abstain. The result exposes
`applied`, `abstained`, `reason`, support coverage and proposed/applied delta
diagnostics. Image/reference arrays use the engine's uint8 or float32 BGR
`[0,255]` convention. The legacy `apply_makeup_unmix` path remains available for
backward compatibility but is not the basis for a physical unmix claim.

### Analytical self-blend tone operators (P5)

```python
from retouch.self_blend import apply_self_blend, self_blend_transfer

curve_value = self_blend_transfer(0.5, "soft_light")
edited = apply_self_blend(image, "soft_light", amount=0.25, domain="encoded")
```

For full-engine routing, pass `self_blend_mode`, `self_blend_amount`, and
`self_blend_domain` to `RetouchEngine.process()`. These caller-only arguments
apply the selected operator in the global grading stage; leaving the mode
`None` preserves the existing path byte-for-byte where its existing contract
applies.

The module provides the ten mathematically derived per-channel self-blend
operators. `amount` is opaque-layer opacity in the selected domain; `domain`
is explicitly `encoded` or `linear`; uint8/uint16 and float image contracts
are supported. These are analytical operators, not a
Photoshop pixel-parity or Fill implementation. Existing Retouch recipes do
not call them automatically.

## 6. Style Library & Machine Learning APIs

The `retouch` package provides tools to extract editing patterns from Lightroom/Photoshop-edited image pairs and save them as reusable custom styles.

### `StyleProfile` Dataclass

Represents the mathematical delta between an original unedited photo and an edited photo.

```python
@dataclass
class StyleProfile:
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    saturation_delta: float = 0.0
    skin_l_mean_delta: float = 0.0
    skin_a_mean_delta: float = 0.0
    skin_b_mean_delta: float = 0.0
    skin_smooth_strength: float = 0.0
    skin_mid_reduction: float = 0.0
    skin_texture_opacity: float = 1.0
```

#### Serialization Methods
*   **`to_dict()`** -> `Dict[str, Any]`
*   **`to_json()`** -> `str`
*   **`save(filepath: str)`**: Save the profile to a JSON file.
*   **`from_dict(d: Dict[str, Any])`** -> `StyleProfile`
*   **`from_json(s: str)`** -> `StyleProfile`
*   **`load(filepath: str)`** -> `StyleProfile`

---

### Style Library Functions

`from retouch.style_library import list_styles, save_style_profile, learn_dataset_style`

#### `list_styles`
```python
def list_styles(directory: Path | str = DEFAULT_STYLE_DIR) -> List[Dict[str, Any]]
```
Lists all custom style profiles in the JSON library with their metadata fields (e.g. `name`, `author`, `tags`, `profile`).

#### `save_style_profile`
```python
def save_style_profile(
    name: str,
    profile: StyleProfile | Dict[str, float],
    author: Optional[str] = None,
    version: str = "1.0",
    tags: Optional[List[str]] = None,
    directory: Path | str = DEFAULT_STYLE_DIR,
) -> Path:
```
Wraps and saves a profile with creation dates and version numbering. Protects history by auto-incrementing suffixes (e.g., `_v2`) if the name already exists.

#### `learn_dataset_style`
```python
def learn_dataset_style(
    original_dir: str | Path,
    edited_dir: str | Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[StyleProfile, int]:
```
Performs machine learning over a folder pair. It matches image filenames, filters invalid outliers using a trimmed mean, ignores failed face detections, and averages color and skin texture parameters. Returns the final averaged `StyleProfile` and the count of successfully processed image pairs.

---

## 7. Changelog

### 2026-06-23 — Micro-Texture Restore & Xiaohongshu Light-Sculpting

*   **New parameter `micro_restore`** (`0`–`50`, default `20`): re-injects dimensional micro-contrast in cheek/nose/under-eye zones after frequency-based smoothing. Backed by `SkinProcessor.restore_micro_texture()`. Exposed in the GUI as the **Micro-Texture Restore** slider and on the CLI as `--micro-restore`. See `SkinProcessor.restore_micro_texture()` for the underlying algorithm.
*   **New recipe `xhs_soft_glow`**: the strongest Xiaohongshu preset — full light-sculpting stack (face relight, specular bloom, tonal curve, highlight rolloff, cool/warm split-toning) plus restored micro-texture and organic grain.
*   **Updated recipes `xiaohongshu` and `xhs_ultrasoft`**: now wire the full light-sculpting stack (`relight`, `specular_bloom`, `tonal_curve_strength`, `highlight_rolloff`, `skin_protect`, split-toning, `micro_restore`, `grain_strength`).
*   **`relight` resolver fix** (`retouch/engine.py`): the engine now respects `spec.recipe_key` for the `relight` parameter instead of hard-coding the top-level key. Previously, every recipe that set `skin.relight` had its value silently dropped (e.g. `xiaohongshu` and `xhs_ultrasoft` resolved to `relight=0`). The fix is data-driven: any future spec that adds a nested `recipe_key` will Just Work without engine-side changes. As a side benefit, every existing recipe that already set `skin.relight` (`cosplay`, `portrait`, `idol`, `korean_beauty`, `wedding`, `fuji_porcelain`, the Fuji sims) now receives its intended light-sculpting treatment.
