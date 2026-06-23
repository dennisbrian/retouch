# Fuji Film Simulations — User Guide

> Phase 1.c of the v1 Fuji-Quality Color Recipe System.
> Three official simulations: **Classic Chrome**, **Astia**, **Provia**.
> All three ship as ready-to-use recipes in `presets/` and as engine entries
> in `retouch/recipes.py`.

---

## Overview

Fujifilm's color science is best known through its **film simulations** —
preset looks that turn the underlying RAW pipeline into a finished JPEG.
Three of those simulations are the most-requested in user recipes, and
v1 ships software-only emulations of all three.

Each sim is implemented as a **parametric recipe**, not a hard-coded LUT.
That means every parameter (saturation, contrast, highlight rolloff, skin
protection, grain, calibration, split tone, etc.) is exposed in the GUI
and JSON, and you can tune any one of them without touching the others.
This is the right architecture because the *only* way to get within
90–95% of a real Fuji JPEG in software is to combine a tone curve, a skin
protect pass, a finishing LUT, and grain — none of those alone is enough.

### Which sim should I use?

| Sim | Best for | Character |
|-----|----------|-----------|
| **Classic Chrome** | Editorial, urban, street, documentary, anything moody | Muted, magazine-grade, lifted blacks, cool greens |
| **Astia** | Portraits, weddings, lifestyle, anything with skin | Soft, warm midtones, skin-friendly |
| **Provia** | Landscapes, products, true-to-life work | Neutral, accurate, no LUT, no grain |

If you only remember one rule: **Astia for skin, Provia for accuracy,
Classic Chrome for mood.**

---

## Classic Chrome

> The "Fuji look" most people mean.

**When to use it.** Editorial shoots, street photography, urban scenes,
documentary work, moody portraits, anything where you want a magazine
finish. Classic Chrome is the most distinctive of the three sims — once
you've seen it, you recognise it instantly. Use it when the story is the
look, not the colour.

**Key characteristics.**

- **Low saturation.** This is the biggest surprise. Classic Chrome is
  *not* a vivid look — it's muted. Global `saturation_boost: -0.15`,
  HSL: red -20, orange -15, yellow -10, green -10, blue -5. Reds are
  pulled toward orange, greens are pulled toward teal.
- **Lifted blacks.** The L curve maps input 0 to output 8 (shadow
  lift: 5) — the deep blacks of a normal JPEG are replaced with a
  slightly milky grey. This is the single most "Fuji" tell.
- **Hard S-curve.** `tonal_curve_strength: 0.7` — the highest of the
  three sims — gives a long midtone ramp with steep ends.
- **Soft highlight shoulder.** The L curve maps input 255 to output
  245; `highlight_rolloff_strength: 0.4` adds a further film-like
  rolloff on top.
- **Cool shadows, warm highlights.** A 3-way split tone with
  shadows at hue 215 (cyan/blue, sat 8), midtones neutral, and
  highlights at hue 30 (warm, sat 5).
- **Cyan-shifted greens.** HSL green hue +15° (toward teal), green
  calibration sat -10. This is the brand-distinctive part of the look.
- **Fine grain.** `grain_strength: 0.15` — visible, but not Velvia-strong.
- **Vignette.** 0.10 (a touch darkening at the corners).

**Best photo types.** Urban scenes, street photography, editorial
fashion, documentary, moody environmental portraits, travel photography
in overcast cities, café/night/restaurant scenes, anything with strong
leading lines you want the viewer to follow.

**Avoid for:** bright sunny landscapes (the look is too moody), product
photography where colour accuracy matters, any image where you want the
saturated Velvia "wow" factor.

### Recipe (key values from `presets/classic_chrome.json`)

```json
{
  "tonal_curve_strength": 0.7,
  "skin_protect_strength": 0.5,
  "highlight_rolloff_strength": 0.4,
  "grain_strength": 0.15,
  "lut": "fuji",
  "saturation_boost": -0.15,
  "warmth": -0.02,
  "shadow_lift": 5,
  "vignette": 0.10,
  "split_tone_three_way": {
    "shadows":   {"hue": 215, "sat": 8},
    "midtones":  {"hue": 0,   "sat": 0},
    "highlights": {"hue": 30,  "sat": 5}
  }
}
```

---

## Astia

> The portrait specialist. Beautiful skin, always.

**When to use it.** Any time there is a face in the frame and you want
the skin to look like skin — not a wax figure, not a sunburn. Astia is
the recipe we recommend as the default for weddings, lifestyle shoots,
environmental portraits, family photos, and any other scene where
flattering skin tones are a hard requirement.

**Key characteristics.**

- **The strongest skin protection in any of our sims.**
  `skin_protect_strength: 0.85` — the design brief targeted
  "best possible skin rendering," and the Astia recipe is the
  one tuned hardest for that. The skin-tone protection pass wraps
  the colour ops in an LCH skin-mask blend, so global saturation
  boosts do *not* push skin into "sunburned" territory.
- **Gentle warm bias.** `warmth: 0.05` — a small positive shift
  that warms skin without making it look orange. The 3-way split
  tone runs warm across the whole range (shadows 20°, midtones
  30°, highlights 40°), so warm-midtones is a continuous gradient
  rather than a single shift.
- **Soft, gradual contrast.** `tonal_curve_strength: 0.5` and a
  gentle L curve ([[0,5], [32,35], [128,130], [224,222], [255,248]]).
  Almost linear midtones, soft toe, soft shoulder. The opposite of
  Classic Chrome's hard S-curve.
- **Slight, gentle saturation.** `saturation_boost: 0.05`, with
  a small bump in red (HSL red sat +5) and orange (HSL orange sat
  +5) — just enough to add life to skin and lips without overdoing it.
- **Lift skin without oversaturating.** HSL orange `lum_shift: 8`
  brightens skin luminance independently of its chroma. This is
  the "luminance-chroma decoupling" trick from the research doc.
- **Very subtle grain.** `grain_strength: 0.08` — barely visible,
  the design target is "fine and forgettable" for portraits.
- **No external LUT.** Astia is fully parametric. The colour
  rendering is built from the calibration, HSL, and split-tone
  adjustments on top of the foundation tone curve.

**Best photo types.** Portraits (any framing), weddings, engagement
shoots, family photos, lifestyle imagery, headshots, beauty work,
fashion editorial, newborn and children's portraits. Anywhere skin
is the subject (or at least part of the subject), Astia is the safe
default.

**Avoid for:** deep-shadow editorial work (use Classic Chrome), very
saturated landscape work (use Provia or stay with a Velvia LUT).

### Recipe (key values from `presets/astia.json`)

```json
{
  "tonal_curve_strength": 0.5,
  "skin_protect_strength": 0.85,
  "highlight_rolloff_strength": 0.6,
  "grain_strength": 0.08,
  "lut": null,
  "saturation_boost": 0.05,
  "warmth": 0.05,
  "shadow_lift": 0.10,
  "vignette": 0.05,
  "split_tone_three_way": {
    "shadows":    {"hue": 20, "sat": 8},
    "midtones":   {"hue": 30, "sat": 3},
    "highlights": {"hue": 40, "sat": 5}
  }
}
```

---

## Provia

> The standard. True to life.

**When to use it.** Anywhere you want the colours you captured to be
the colours the viewer sees. Provia is the neutral, accurate,
"this is what was in front of the lens" simulation — the one a
commercial photographer would pick for product shots, and the one a
landscape photographer would pick for a faithful sunset.

**Key characteristics.**

- **Slight, neutral saturation.** `saturation_boost: 0.05` — a
  hair above zero. No per-channel overrides, no saturation push
  in the HSL, no HSL skin or orange shifts.
- **Linear midtone tone curve.** `tonal_curve_strength: 0.4` —
  the lowest of the three sims. The L curve is essentially a
  straight line with a very gentle toe and shoulder.
  `[[0,0], [32,28], [128,132], [224,232], [255,255]]`.
- **No shadow lift.** `shadow_lift: 0.0` — Provia honours the deep
  blacks in the original scene. This is the opposite of Classic
  Chrome's milky-blacks look.
- **No warm or cool cast.** `warmth: 0.0`. No 3-way split tone.
  No calibration overrides on any of the colour channels. Whatever
  white balance the original capture had, Provia preserves.
- **No grain.** `grain_strength: 0.0`. The cleanest of the three
  sims. Suitable for product photography and any other scene
  where noise/grain is unwanted.
- **No LUT.** Provia is fully parametric, like Astia. No 3D LUT
  is applied; the foundation layer (tonal curve, skin protect,
  highlight rolloff) is what gives the look.
- **Sharpness is a touch higher.** `sharpness: 0.4` — the highest
  of the three sims. Provia is the "default accurate" look, and
  accurate means crisp.
- **Skin protection is on, but at low strength.** `skin_protect_strength: 0.2` —
  just enough to keep skin from going pink/magenta under the mild
  saturation boost, but not so much that it changes the rendering
  on non-skin pixels.

**Best photo types.** Landscapes (especially sunsets and
foliage), product photography, food photography, real-estate,
architecture, e-commerce, scientific / technical documentation,
anywhere accurate colour reproduction is more important than
artistic interpretation. Provia is also the right default for
*unknown* image content — when you don't know what the photo
will be, "what the camera saw" is the safest answer.

**Avoid for:** moody/editorial work (use Classic Chrome), portraits
where you want the skin-flattering warm treatment (use Astia).

### Recipe (key values from `presets/provia.json`)

```json
{
  "tonal_curve_strength": 0.4,
  "skin_protect_strength": 0.2,
  "highlight_rolloff_strength": 0.2,
  "grain_strength": 0.0,
  "lut": null,
  "saturation_boost": 0.05,
  "warmth": 0.0,
  "shadow_lift": 0.0,
  "vignette": 0.0,
  "sharpness": 0.4
}
```

---

## How to Use

### GUI (Gradio)

1. Launch with `python3 gui.py` (or open the desktop build).
2. In the **Recipe** dropdown, select one of:
   - `Classic Chrome`
   - `Astia`
   - `Provia`
3. Tweak any slider you want. The sim is a recipe, not a sealed
   preset — every parameter is exposed.
4. Process. Save the recipe as JSON if you want to keep your tweaks.

### CLI (single image)

```bash
python3 cli.py photo.jpg -o photo_out.jpg --recipe classic_chrome
python3 cli.py photo.jpg -o photo_out.jpg --recipe astia
python3 cli.py photo.jpg -o photo_out.jpg --recipe provia
```

`--recipe` accepts any of the three sim names, plus the older
presets (`natural`, `cosplay`, `fuji_porcelain`, etc.). See
`RECIPE_GUIDE.md` for the full list.

### CLI (batch)

```bash
python3 cli.py /path/to/input_dir -o /path/to/output_dir \
  --recipe astia --workers 6
```

The batch processor honours `--recipe` exactly the same way — pick
a sim, point it at a directory, and process everything in parallel.
See `BATCH_GUIDE.md` for full batch options.

### Python API

```python
from retouch.engine import RetouchEngine

engine = RetouchEngine()
result = engine.process(image_bgr, recipe="astia")
```

`recipe` can be the string name (`"classic_chrome"`, `"astia"`,
`"provia"`) or a dict with custom parameter overrides.

### Editing the sim

Each sim is a JSON file in `presets/`. Edit the file, reload the
engine, and your changes are picked up. For a full schema reference
see `RECIPE_GUIDE.md`.

```bash
# Edit Astia
$EDITOR presets/astia.json

# Re-run
python3 cli.py photo.jpg -o out.jpg --recipe astia
```

---

## Comparison Table

| Aspect | Classic Chrome | Astia | Provia |
|---|---|---|---|
| **Personality** | Editorial, moody | Soft, flattering | Neutral, accurate |
| **Saturation** | -0.15 (low) | +0.05 (slight) | +0.05 (slight) |
| **Contrast** | hard S-curve | gentle, soft toe + shoulder | linear midtone |
| **Tone curve strength** | 0.7 | 0.5 | 0.4 |
| **Skin protection** | 0.5 | **0.85** (highest) | 0.2 |
| **Highlight rolloff** | 0.4 | 0.6 | 0.2 |
| **Grain** | 0.15 (fine) | 0.08 (very fine) | 0.0 (none) |
| **Shadow lift** | 5 (lifted) | 0.10 (mild) | 0.0 (none) |
| **LUT** | `fuji` (demo) | none | none |
| **Split tone shadows** | hue 215 (cyan) | hue 20 (warm) | none |
| **Split tone highlights** | hue 30 (warm) | hue 40 (warm) | none |
| **HSL red sat** | -20 | +5 | 0 |
| **HSL green hue** | +15° (toward teal) | 0 | 0 |
| **HSL orange lum** | 0 | +8 (lift) | 0 |
| **Vignette** | 0.10 | 0.05 | 0.0 |
| **Sharpness** | 0.3 | 0.3 | 0.4 |
| **Use for** | editorial, street | portraits, weddings | landscapes, products |
| **Avoid for** | bright landscapes | moody work | artistic looks |

---

## Limitations

There are things a software emulation **cannot** match in a real
Fuji JPEG. We list them honestly so you know what you're getting.

1. **No X-Trans sensor.** Fuji's APS-C bodies (X-T series, X-Pro,
   X100, etc.) use a 6×6 X-Trans CFA, not the standard 2×2 Bayer
   pattern. The "Fuji micro-detail" look in foliage and other
   high-frequency textures is partly a side-effect of the X-Trans
   demosaic. Our pipeline accepts any image — Bayer, X-Trans, even
   phone photos — but we cannot recreate the X-Trans texture on
   a Bayer-sourced input. This is a hard ceiling on fidelity.

2. **Approximated tone curves.** The exact H&D curve shape for
   each simulation is not publicly disclosed by Fujifilm. The
   curves in our sims are derived from published Fuji samples
   and the publicly observable X-Photographer recipes, not
   reverse-engineered internal specifications. They look right;
   they may not be exact to within a few percent in any
   specific luminance range.

3. **Approximated color matrix.** The "Fuji green" and the
   "warm yellow-red skin" are partly the output of a 3×3 color
   matrix that maps sensor RGB into the working color space.
   That matrix is not public. We approximate the effect with
   per-channel calibration + HSL + split tone, which gets us
   into the right neighbourhood but cannot perfectly match
   any specific Fuji body.

4. **Demo LUTs, not real film stock.** Classic Chrome references
   `luts/fuji.cube`, which is a 17³ algorithmically-generated LUT
   inspired by the look (cool midtones, slight highlight crush).
   It is **not** a measured Fuji film stock LUT. For 95%+
   match, install a commercial `.cube` (RNI, VSCO, Dehancer) or
   a Fujifilm X-Trans profile from X RAW STUDIO and point the
   `lut` field at the file's stem. See `luts/ACQUISITION.md`.

5. **No grain engine parity with the camera.** The grain
   synthesis in `grain.apply_film_grain` is clumped and
   luminance-correlated (the two biggest "real grain" tells),
   but it is not a per-stock grain model. Real Velvia grain
   is finer than real Portra grain; our grain does not
   distinguish. For most viewers this is fine; pixel-peepers
   will see it.

6. **A perceptual match, not a measured one.** A real "Fuji
   match" project would train a model on paired Fuji-RAW /
   Fuji-JPEG images and learn a regression. We have not done
   that — the 90–95% figure is a subjective eyeball target
   against publicly visible Fuji samples, not a measured
   ΔE00.

These limits are honest. v1 hits the brief; v1.5 (the next
post-v1 phase) is where we close the gap with measured
LUTs and possibly a learned component. See `ROADMAP.md`
for the full plan.

---

## Where the Recipes Live

| Sim | JSON file | Engine entry |
|---|---|---|
| `classic_chrome` | `presets/classic_chrome.json` | `retouch.recipes.RECIPES["classic_chrome"]` |
| `astia` | `presets/astia.json` | `retouch.recipes.RECIPES["astia"]` |
| `provia` | `presets/provia.json` | `retouch.recipes.RECIPES["provia"]` |

The JSON file is the source of truth for descriptive metadata
(curves, calibration, HSL, split tone). The engine entry in
`retouch/recipes.py` is the runtime form (flat scalars consumed
by `build_context`). Both must stay in sync; the tests in
`tests/test_astia.py` and `tests/test_provia.py` enforce the
design constraints.

The canonical name list is `retouch.recipes.FUJI_SIM_NAMES`
(`["classic_chrome", "astia", "provia"]`) and is what the GUI
dropdown and CLI help text are generated from.
