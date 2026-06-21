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
| `eyes` | `whites` / `iris` | 0.0–1.0 | Eye Enhance (×100) |
| `eyes` | `catchlight` | 0.0–1.0 | — |
| `eyes` | `dark_circles` | 0.0–1.0 | — |
| `lips` | `gloss` | 0.0–1.0 | Lip Enhance (×100) |
| `lips` | `tint` | `None`, `"rose"`, `"pink"`, `"coral"`, `"natural"`, `"berry"`, `"cosplay"` | Lip Tint |
| `hair` | `shine` | 0.0–1.0 | Hair Shine (×100) |
| `dodge_burn` | `amount` | 0.0–1.0 | Dodge & Burn (×100) |
| `color_harmony` | `preset` | string preset name | — |
| `color_harmony` | `amount` | 0.0–1.0 | — |
| `bloom` | `opacity` | 0.0–1.0 | Bloom (×100) |
| `bloom` | `threshold` | 150–250 | Bloom Threshold |
| `texture` | `opacity` | 0.0–1.0 | Texture Opacity |
| `finish` | `impact` | 0.0–1.0 | — |

### Flat Fields (top-level, direct values)

```python
"contrast": -25,           # -50–50
"brightness": 6.0,         # -100–100
"highlights": -10.0,       # -100–100
"shadows": -4.0,           # -100–100
"whites": 8.0,             # -100–100
"blacks": 6.0,             # -100–100
"clarity": 14.0,           # 0–100
"saturation": 6.0,         # -100–100
"vibrance": 12.0,          # 0–100
"relight_strength": 35.0,  # 0–100
"light_azimuth": 45.0,     # -180–180
"light_elevation": 35.0,   # -90–90
"bloom": {"opacity": 0.16, "threshold": 190.0},  # or flat opacity
"specular_bloom": 50,      # 0–100
"vignette": 6.0,           # 0–100
"sharpen": 20.0,           # 0–100
"sharpen_radius": 1.2,     # float
"chromatic_aberration": 4.0, # float
"grain": 0.0,              # 0–100
"glow": 8.0,               # float
"subject_separation": 0.85, # 0.0–1.0
"background_blur": 0.55,   # 0.0–1.0
"background_desaturation": 0.40, # 0.0–1.0
"light_wrap": 0.25,        # 0.0–1.0
"blue_shadow_grade": 0.75, # 0.0–1.0
"cyan_midtone_grade": 0.45, # 0.0–1.0
"subject_sharpen": 0.30,   # 0.0–1.0
"matte_black": 0.10,       # 0.0–1.0
"slimming": 30.0,          # 0–100
"blush": 25.0,             # 0–100
"lip_finish": "gloss",     # "gloss", "velvet", "matte"
"nose_blush": True,        # bool
"under_eye_blush": True,   # bool
"white_costume_lift": True, # bool
```

## GUI Slider ↔ Recipe Value Conversion

GUI slider values are converted in `gui.py:recipe_defaults()`:

| GUI Slider | Recipe field | Conversion |
|---|---|---|
| Smooth (0–100) | `frequency.smooth` | ÷ 100 |
| Mid Reduction (0.0–1.0) | `frequency.mid_reduction` | direct |
| Texture Opacity (0.0–1.0) | `texture.opacity` | direct |
| Whitening (0–100) | `skin.rosy` or `skin.porcelain` | ÷ 100 |
| Equalize (0–100) | `skin.equalize` | ÷ 100 |
| Eye Enhance (0–100) | `eyes.whites` / `eyes.iris` | ÷ 100 |
| Lip Enhance (0–100) | `lips.gloss` | ÷ 100 |
| Hair Shine (0–100) | `hair.shine` | ÷ 100 |
| Dodge & Burn (0–100) | `dodge_burn.amount` | ÷ 100 |
| Bloom (0–100) | `bloom.opacity` | ÷ 100 |
| Contrast (-50–50) | `contrast` | direct |
| Brightness (-100–100) | `brightness` | direct |
| Highlights (-100–100) | `highlights` | direct |
| Shadows (-100–100) | `shadows` | direct |
| Whites (-100–100) | `whites` | direct |
| Blacks (-100–100) | `blacks` | direct |
| Nose Blush (Checkbox) | `nose_blush` | direct (bool) |
| Under-Eye Blush (Checkbox) | `under_eye_blush` | direct (bool) |
| White Costume Lift (Checkbox) | `white_costume_lift` | direct (bool) |

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
