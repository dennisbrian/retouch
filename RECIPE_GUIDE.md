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

## Field Reference — All Groups

### Core Groups (nested dicts)

| Group | Keys | Range | GUI Slider |
|---|---|---|---|
| `frequency` | `smooth` | 0.0–1.0 | Smooth (×100) |
| `frequency` | `mid_reduction` | 0.0–1.0 | Mid Reduction |
| `skin` | `equalize` | 0.0–1.0 | Equalize (×100) |
| `skin` | `rosy` / `porcelain` | 0.0–1.0 | Whitening (×100) |
| `skin` | `nose_smooth` | 0.0–1.0 | Nose Smooth (×100) |
| `skin` | `flatten` | 0.0–1.0 | Skin Flatten — cel flatting (×100) |
| `skin` | `quantize` | 0.0–1.0 | Tone Quantize — cel bands (×100) |
| `skin` | `unify` | 0.0–1.0 | Hue Unify — anime skin tone (×100) |
| `skin` | `unify_hue` | −1–360 | Target hue for unify (−1 = auto) |
| `skin` | `glow` | 0.0–1.0 | Skin Light-Wrap glow (×100) |
| `skin` | `relight` | 0.0–1.0 | Virtual studio relighting (×100) |
| `eyes` | `whites` / `iris` | 0.0–1.0 | Eye Enhance (×100) |
| `eyes` | `catchlight` | 0.0–1.0 | Catchlight (×100) |
| `eyes` | `dark_circles` | 0.0–1.0 | Dark Circles (×100) |
| `lips` | `gloss` | 0.0–1.0 | Lip Enhance (×100) |
| `lips` | `tint` | `None`, `"rose"`, `"pink"`, `"coral"`, `"natural"`, `"berry"`, `"cosplay"` | Lip Tint |
| `hair` | `shine` | 0.0–1.0 | Hair Shine (×100) |
| `dodge_burn` | `amount` | 0.0–1.0 | Dodge & Burn (×100) |
| `color_harmony` | `preset` | see presets below | Color Preset (dropdown) |
| `color_harmony` | `amount` | 0.0–1.0 | Color Amount (×100) |
| `bloom` | `opacity` | 0.0–1.0 | Bloom (×100) |
| `bloom` | `threshold` | 150–250 | Bloom Threshold |
| `bloom` | `softness` | 1–100 | Bloom Softness |
| `texture` | `opacity` | 0.0–1.0 | Texture Opacity |
| `finish` | `impact` | 0.0–1.0 | Impact (×100) |
| `blemish` | (top-level) | 0.0–1.0 | Blemish (×100) |
| `pore_synthesis` | (top-level) | 0.0–1.0 | Pore Synthesis (×100) |

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

### Flat Fields (top-level, direct values)

```python
"contrast": -25,             # -50–50
"brightness": 6.0,           # -100–100
"highlights": -10.0,         # -100–100
"shadows": -4.0,             # -100–100
"whites": 8.0,               # -100–100
"blacks": 6.0,               # -100–100
"clarity": 14.0,             # -100–100
"saturation": 6.0,           # -100–100
"vibrance": 12.0,            # -100–100
"relight_strength": 35.0,    # 0–100
"light_azimuth": 45.0,       # -180–180  (also accepts `relight_azimuth`)
"light_elevation": 35.0,     # -90–90    (also accepts `relight_elevation`)
"bloom": {"opacity": 0.16, "threshold": 190.0, "softness": 30.0},  # or flat `bloom` (opacity)
"specular_bloom": 50,        # 0–100
"specular_bloom_tone": "rosy",  # "rosy", "porcelain", "neutral"
"shadow_hue": 225.0,         # 0–360
"shadow_sat": 30.0,          # 0–100
"midtone_hue": 0.0,          # 0–360
"midtone_sat": 15.0,         # 0–100
"highlight_hue": 320.0,      # 0–360
"highlight_sat": 25.0,       # 0–100
"vignette": 6.0,             # 0–100
"sharpen": 20.0,             # 0–100
"sharpen_radius": 1.2,       # 0.1–5.0
"chromatic_aberration": 4.0, # 0–20
"halation": 12.0,            # 0–100
"grain": 0.0,                # 0–100
"glow": 8.0,                 # 0–100
"subject_separation": 0.85,  # 0–100  (recipe-scale, see conversion table)
"catchlight": 18.0,          # 0–100
"dark_circles": 22.0,        # 0–100
"blemish": 30.0,             # 0–100
"pore_synthesis": 0.0,       # 0–100
"micro_restore": 20,         # 0–50  (default 20; re-injects dimensional micro-contrast after smoothing)
"eye_enhance": 30.0,         # 0–100  (overrides `eyes.iris` / `eyes.whites`)
"background_blur": 0.55,     # 0.0–1.0
"background_desaturation": 0.40, # 0.0–1.0
"light_wrap": 0.25,          # 0.0–1.0
"blue_shadow_grade": 0.75,   # 0.0–1.0
"cyan_midtone_grade": 0.45,  # 0.0–1.0
"subject_sharpen": 0.30,     # 0.0–1.0
"matte_black": 0.10,         # 0.0–1.0
"slimming": 30.0,            # 0–100
"blush": 25.0,               # 0–100
"lip_finish": "gloss",       # "gloss", "velvet", "matte"
"nose_blush": True,          # bool
"under_eye_blush": True,     # bool
"white_costume_lift": True,  # bool
"auto_exposure": False,      # bool
```

## GUI Slider ↔ Recipe Value Conversion

GUI slider values are converted in `gui.py:recipe_defaults()`. Most sliders
live on a 0–100 GUI scale; recipe values follow the engine's native scale
(typically 0.0–1.0 for fraction-style fields, signed 0–100 for tonal fields,
and 0–360 for split-toning hue). The two columns below make that mapping
explicit:

| GUI Slider | Recipe field | Conversion |
|---|---|---|
| Smooth (0–100) | `frequency.smooth` | ÷ 100 |
| Mid Reduction (0.0–1.0) | `frequency.mid_reduction` | direct |
| Texture Opacity (0.0–1.0) | `texture.opacity` | direct |
| Pore Synthesis (0–100) | `pore_synthesis` | ÷ 100 |
| Micro-Texture Restore (0–50) | `micro_restore` | direct |
| Blemish Removal (0–100) | `blemish` | ÷ 100 |
| Whitening (0–100) | `skin.rosy` or `skin.porcelain` | ÷ 100 |
| Equalize (0–100) | `skin.equalize` | ÷ 100 |
| Nose Smooth (0–100) | `skin.nose_smooth` | ÷ 100 |
| Whiten Tone (dropdown) | `skin.porcelain` (chooses) / `whiten_tone` (override) | direct (string) |
| Eye Enhance (0–100) | `eyes.whites` / `eyes.iris` | ÷ 100 |
| Catchlight (0–100) | `eyes.catchlight` | ÷ 100 |
| Dark Circles (0–100) | `eyes.dark_circles` | ÷ 100 |
| Teeth Whiten (0–100) | `eyes.teeth_whiten` or `eyes.whites` | ÷ 100 |
| Lip Enhance (0–100) | `lips.gloss` | ÷ 100 |
| Lip Tint (dropdown) | `lips.tint` | direct (string) |
| Hair Shine (0–100) | `hair.shine` | ÷ 100 |
| Dodge & Burn (0–100) | `dodge_burn.amount` | ÷ 100 |
| Specular Bloom (0–100) | `specular_bloom` | ÷ 100 |
| Specular Bloom Tone (dropdown) | `specular_bloom_tone` | direct (string) |
| Bloom Opacity (0–100) | `bloom.opacity` | ÷ 100 |
| Bloom Threshold (150–250) | `bloom.threshold` | direct |
| Bloom Softness (1–100) | `bloom.softness` | direct |
| Impact (0–100) | `impact` | ÷ 100 |
| Relight Strength (0–100) | `relight_strength` | ÷ 100 |
| Light Azimuth (-180–180) | `light_azimuth` (alias: `relight_azimuth`) | direct |
| Light Elevation (-90–90) | `light_elevation` (alias: `relight_elevation`) | direct |
| Contrast (-50–50) | `contrast` | direct |
| Brightness (-100–100) | `brightness` | direct |
| Clarity (-100–100) | `clarity` | direct |
| Vibrance (-100–100) | `vibrance` | direct |
| Saturation (-100–100) | `saturation` | direct |
| Highlights (-100–100) | `highlights` | direct |
| Shadows (-100–100) | `shadows` | direct |
| Whites (-100–100) | `whites` | direct |
| Blacks (-100–100) | `blacks` | direct |
| Glow (0–100) | `glow` | ÷ 100 |
| Vignette (0–100) | `vignette` | ÷ 100 |
| Sharpen (0–100) | `sharpen` | ÷ 100 |
| Sharpen Radius (0.1–5.0) | `sharpen_radius` | direct |
| Subject Separation (0–100) | `subject_separation` | ÷ 100 |
| Grain (0–100) | `grain` | ÷ 100 |
| Halation (0–100) | `halation` | ÷ 100 |
| Chromatic Aberration (0–20) | `chromatic_aberration` | direct |
| Shadow Hue (0–360) | `shadow_hue` | direct |
| Shadow Saturation (0–100) | `shadow_sat` | ÷ 100 |
| Midtone Hue (0–360) | `midtone_hue` | direct |
| Midtone Saturation (0–100) | `midtone_sat` | ÷ 100 |
| Highlight Hue (0–360) | `highlight_hue` | direct |
| Highlight Saturation (0–100) | `highlight_sat` | ÷ 100 |
| Color Harmony Preset (dropdown) | `color_harmony.preset` | direct (string) |
| Color Harmony Amount (0–100) | `color_harmony.amount` | ÷ 100 |
| Background Blur (0.0–1.0) | `background_blur` | direct |
| Background Desaturation (0.0–1.0) | `background_desaturation` | direct |
| Light Wrap (0.0–1.0) | `light_wrap` | direct |
| Blue Shadow Grade (0.0–1.0) | `blue_shadow_grade` | direct |
| Cyan Midtone Grade (0.0–1.0) | `cyan_midtone_grade` | direct |
| Subject Sharpen (0.0–1.0) | `subject_sharpen` | direct |
| Matte Black (0.0–1.0) | `matte_black` | direct |
| Slimming (0–100) | `slimming` | direct |
| Blush (0–100) | `blush` | direct |
| Lip Finish (dropdown) | `lip_finish` | direct (string) |
| Nose Blush (Checkbox) | `nose_blush` | direct (bool) |
| Under-Eye Blush (Checkbox) | `under_eye_blush` | direct (bool) |
| White Costume Lift (Checkbox) | `white_costume_lift` | direct (bool) |
| Auto Exposure (Checkbox) | `auto_exposure` | direct (bool) |

> **Note:** fields in the *Flat Fields* section above that are documented with
> a 0–100 range (e.g. `glow: 8.0`) use the engine's native 0–100 scale directly
> and are stored as-is. Fields in the *Core Groups* table that are documented
> with a 0.0–1.0 range are stored as the slider value divided by 100. The
> signed tonal fields (`clarity`, `vibrance`, `saturation`, `highlights`,
> `shadows`, `whites`, `blacks`) pass through unchanged because the GUI scale
> already matches the engine scale.

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
