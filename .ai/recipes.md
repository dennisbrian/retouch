# Recipes

Recipes map style names to numeric params in `recipes.py`.

## Structure

| Field | Type | Description |
|---|---|---|
| `frequency` | dict | `smooth`, `mid_reduction` |
| `skin` | dict | `equalize`, `rosy` / `porcelain` (whitening tint) |
| `eyes` | dict | `whites` (sclera), `iris`, `catchlight`, `dark_circles` |
| `lips` | dict | `gloss`, `tint` (cosplay/rose/pink/coral/berry), `finish` (gloss/velvet/matte) |
| `hair` | dict | `shine` |
| `dodge_burn` | dict | `amount` |
| `color_harmony` | dict | `preset` (grade name), `amount` (opacity) |
| `bloom` | dict | `opacity`, `threshold` |
| `texture` | dict | `opacity` |
| `finish` | dict | `impact` |
| Flat fields | | `contrast`, `brightness`, `highlights`, `shadows`, `whites`, `blacks`, `clarity`, `saturation`, `vibrance`, `vignette`, `sharpen`, `grain`, `glow`, `slimming`, `blush`, `nose_blush`, `under_eye_blush`, `white_costume_lift` |

## Inheritance

`extends` key deep-merges parent first, child overrides specific sub-keys only.

## Built-in Recipes

| Name | Extends | Key Settings | Use Case |
|---|---|---|---|
| `natural` | — | `smooth:0.30`, `equalize:0.20`, `rosy:0.10` | Subtle daily cleanup |
| `portrait` | `natural` | `smooth:0.45`, `equalize:0.35`, `rosy:0.20`, `dodge_burn:0.10` | Pro portraiture |
| `cosplay` | `natural` | `smooth:0.55`, `equalize:0.40`, `rosy:0.35`, `slimming:30`, `blush:25` | Anime/cosplay |
| `scifi_cosplay` | `natural` | `smooth:0.85`, `texture:0.25`, `color_harmony:cyberpunk`, `bloom:0.15` | Cyberpunk |
| `fuji_porcelain` | `natural` | `smooth:0.72`, `equalize:0.50`, `rosy:0.70`, `specular_bloom:50` | Fine art |
| `anime_cinematic_v1` | `natural` | `smooth:0.32`, `relight_strength:35`, `bloom:0.16`, `glow:8`, `vignette:6` | Dreamy cinematic |

## Adding

Append dict to `RECIPES` in `recipes.py` — name auto-propagates to CLI/Gradio.
