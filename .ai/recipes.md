# Recipes: Preset Configurations & Customization

Recipes map high-level visual styles to specific numeric parameter ratios in `retouch/recipes.py`.

## Recipe Structure & Fields

A recipe entry consists of nested group dictionaries and flat top-level parameters:

- **`frequency`**: `smooth` (coarse blur), `mid_reduction` (blemish smoothing).
- **`skin`**: `equalize` (CLAHE tone flattening), `rosy` or `porcelain` (whitening tint amount).
- **`eyes`**: `whites` (sclera boost), `iris` (iris boost), `catchlight` (specular lift), `dark_circles` (bag repair).
- **`lips`**: `gloss` (specular highlight), `tint` (`cosplay`, `rose`, `pink`, `coral`, `berry`), `finish` (`gloss`, `velvet`, `matte`).
- **`hair`**: `shine` (specular reflection lift).
- **`dodge_burn`**: `amount` (contour sculpting).
- **`color_harmony`**: `preset` (grading preset name), `amount` (grade opacity).
- **`bloom`**: `opacity` (Orton glow screen blend), `threshold` (luminance cutoff).
- **`texture`**: `opacity` (fine grain skin pore blend).
- **`finish`**: `impact` (luminance/saturation boost amount).
- **Flat Fields**: Direct sliders overrides (`contrast`, `brightness`, `highlights`, `shadows`, `whites`, `blacks`, `clarity`, `saturation`, `vibrance`, `vignette`, `sharpen`, `grain`, `glow`, `slimming`, `blush`, `nose_blush` (bool), `under_eye_blush` (bool), `white_costume_lift` (bool)).

## The Inheritance (`extends`) Mechanism
Recipes can inherit from other recipes via the `"extends"` key. The system recursively merges configurations:
1. `resolve_recipe(name)` loads the target preset.
2. If `"extends": "base_name"` is present, it loads `base_name` first and merges the child parameters on top.
3. Nested dictionaries (e.g., `frequency`, `skin`) are deep-merged, overriding only specified sub-keys.

## Key Built-In Recipes Reference

| Recipe Name | Extends | Core Settings | Use Case |
|---|---|---|---|
| **`natural`** | None | `smooth: 0.30`, `equalize: 0.20`, `rosy: 0.10` | Daily portraits; subtle texture-preserving cleanups. |
| **`portrait`** | `natural` | `smooth: 0.45`, `equalize: 0.35`, `rosy: 0.20`, `dodge_burn: 0.10` | Standard professional portraiture. |
| **`cosplay`** | `natural` | `smooth: 0.55`, `equalize: 0.40`, `rosy: 0.35`, `slimming: 30`, `blush: 25`, `nose_blush: True` | Cosplay, anime event photos, heavy makeup washes. |
| **`scifi_cosplay`** | `natural` | `smooth: 0.85`, `texture: 0.25`, `color_harmony: cyberpunk`, `bloom: 0.15` | Cyberpunk look with smooth skin and colored grading. |
| **`fuji_porcelain`** | `natural` | `smooth: 0.72`, `equalize: 0.50`, `rosy: 0.70`, `specular_bloom: 50` | Desaturated, high-contrast fine art portraiture. |
| **`anime_cinematic_v1`** | `natural` | `smooth: 0.32`, `relight_strength: 35`, `bloom: 0.16`, `glow: 8`, `vignette: 6` | Dreamy, movie-like aesthetic with directional relighting. |

## Adding a New Recipe
1. Open [recipes.py](file:///Applications/htdocs/retouch/retouch/recipes.py).
2. Append a new dictionary config to `RECIPES`.
3. The preset name will automatically propagate to the CLI options and the Gradio GUI dropdown.
