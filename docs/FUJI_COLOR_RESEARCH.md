# Fujifilm Color Science Research

> Research document for the Retouch Engine v1 (Fuji-quality color recipe system).
> Goal: 90–95% match to a Fujifilm JPEG in software.
> Author: Worker H / P0.9 research. Compiled from public sources; no proprietary
> information was used. Where exact numbers are unknown, ranges are given.

---

## 1. Executive Summary

Fujifilm's reputation for "color science" is the combination of (a) a 90-year
film heritage whose H&D tonal response, color matrix, and grain structure
have been iteratively refined for decades, and (b) a modern X-Trans CMOS
pipeline that re-encodes those principles for digital. The "Fuji look" is
neither a single LUT nor a single film stock — it is a family of related
aesthetic choices: a soft shoulder on the highlight roll-off, a gentle warm
bias in the shadows, saturated but never neon greens and blues, and skin
tones that are protected from both oversaturation and hue rotation. To get
within 5–10% of a Fuji JPEG in software, we need a parametric pipeline
(matrix → tone curve → chroma protection → optional grain), not a single
3D LUT.

The most important non-obvious finding: Fuji's color rendering is dominated
by the *color matrix and skin-tone protection*, not by the film simulation
curve. A "Classic Chrome" look is reproducible from a Provia (standard) base
mostly by desaturating reds, dropping contrast in midtones, and pushing the
green/cyan axis. Conversely, putting a Fuji tone curve on a Sony RAW without
the color matrix will look "wrong" — washed out and slightly off-hue in the
foliage.

---

## 2. What Is "The Fuji Look"?

### 2.1 Fuji's Reputation in the Photography Community

Fujifilm is one of the few major camera manufacturers with a continuous
film-emulsion R&D history going back to 1934 ([Wikipedia: Fujifilm][1]).
That heritage is widely credited for the "film-like" character of their
JPEGs. Photographers describe the look with words like:

- *warm but not orange* — skin tones lean toward yellow-red rather than
  pink-magenta;
- *creamy highlights* — the roll-off into white is gentle, not clipped;
- *shadow detail retention* — blacks are not crushed, even in high-contrast
  scenes;
- *deep but not neon greens* — foliage looks rich, almost wet;
- *gentle saturation* — colors are saturated relative to a flat profile but
  never cross into cartoon territory (with the deliberate exception of
  Velvia/vivid modes).

This is consistent with Velvia's documented characteristics: "very high
saturation" with "very fine grain" and exposure latitude of "±½ stop"
([Wikipedia: Velvia][2]). Provia, by contrast, is described as having "less
saturated colors and contrast compared to Velvia" with finer granularity
([Wikipedia: Provia][3]).

### 2.2 Iconic Film Simulations (in order of introduction)

| Simulation | Film/emulation origin | What it's known for |
|------------|----------------------|---------------------|
| **Provia** | Fujichrome Provia 100F (RDP III) — neutral, accurate reversal film | Standard / faithful; high micro-contrast; the "default" Fuji look |
| **Velvia** | Fujichrome Velvia 50/100 (RVP) — saturated reversal film | Punchy landscapes; deep greens, electric blues; famously "high saturation" (Wikipedia notes Velvia's saturation is "very high" [2]) |
| **Astia** | Fujichrome Astia 100F — soft-portrait reversal film | Smooth skin, slightly desaturated reds, gentle contrast (portrait staple) |
| **ACROS** | Neopan ACROS II 100 — black-and-white film | Fine grain B&W with a slight warm/neutral bias and smooth tonal transition |
| **Classic Chrome** | Fujifilm's editorial/journalism simulation (introduced on X100S, 2013) | Desaturated, hard S-curve, slightly green shadows — the "magazine" look |
| **Classic Negative** | Fujicolor Superia 400 (consumer negative) emulation | Saturated greens, warm yellows, the "X-Photographer" travel look |
| **Eterna** | Motion picture film stock emulation | Low-contrast, log-style; built for video grading |
| **Bleach Bypass** | Darkroom bleach-bypass process | High contrast, low saturation, gritty; stylized |

These are preset *curves + chroma adjustments* layered on top of the same
underlying color science pipeline. They differ in tone curve, saturation
amounts per channel, and color matrix rotation — not in sensor or
demosaicing (which are shared).

### 2.3 Color Science vs Film Simulations: What's the Difference?

This distinction is important and often conflated:

- **"Fuji color science"** = the underlying processing pipeline: sensor
  CFA (X-Trans), demosaicing, color matrix (RGB→YCC), tone curve, gamma,
  white balance, sharpening, noise reduction, lens correction. This is
  fixed per camera body and is what makes a Fuji RAW *look* like a Fuji
  RAW in a third-party converter.
- **"Fuji film simulations"** = artist presets (Provia/Velvia/Astia/etc.)
  applied on top of that pipeline. They are switchable and conceptually
  live in the same layer as "vivid" or "monochrome" presets in any
  consumer camera.

Concretely: if you shoot RAW on an X-T5 and process in Capture One with
"Film Simulation: Classic Chrome" turned off, the image still has Fuji
color science — its color matrix, white balance multipliers, and tone
curve are still from Fuji's raw pipeline. Turning the simulation on
adds a parametric "look" on top. Conversely, if you apply a "Velvia"
LUT to a Sony RAW, you get the *colors* of Velvia but lose the
sensor/demosaic/tonal character of the original Fuji capture.

For our v1 goal of 90–95% match, we are matching the *combined output*:
the simulation + the underlying pipeline. We do not have access to Fuji's
matrix or demosaic, so we approximate the simulation as best we can and
treat the underlying pipeline as a fixed input from any modern camera.

---

## 3. The Four Key Technical Ingredients

These are the four areas where the Fuji look is engineered. Each can be
addressed independently in software, and together they account for the
majority of the "Fuji" character.

### 3.1 Color Matrix (RGB → Lab/YCC Conversion)

Every digital camera has a 3×3 color matrix that maps sensor RGB to a
standard color space (typically XYZ or Lab). This matrix is *the* most
brand-distinguishing part of color science. It is what determines whether
greens are "Fuji green" or "Canon green" or "Sony green."

What is publicly known / believed about Fuji's matrix:

- **Warm bias in skin hues.** Fuji's matrix is widely reported to push
  skin hues toward warm yellow-red (~25°–40° hue rotation in CIE LCh)
  rather than the cooler pink-magenta that is more typical of Canon/Sony
  default renderings. This is a deliberate "memory color" choice: warm
  skin is more flattering and reads as "natural" to most viewers.
- **Saturated but desaturated-from-neon greens.** The green primary is
  pushed outward in chroma but with a slight cyan-ward rotation, so
  foliage looks rich without looking radioactive.
- **Compressed blue gamut.** Blues are slightly pulled in from the
  edge of the gamut to avoid the "neon blue" look that plagues some
  Sony output in deep sky. The exact direction and amount is not
  publicly disclosed.

We do not have the exact Fuji matrix, but ICC profiles shipped with
X-Trans cameras (downloadable from Fuji's support site for some models)
are an excellent approximation. They can be parsed with `LittleCMS` or
similar to extract a 3×3 matrix from the Bradford-adapted XYZ output.

### 3.2 Tonal Response Curve (H&D Film Curve vs Digital sCurve)

A film emulsion's response to light follows the **Hurter–Driffield
(H&D) curve** — a sigmoid with a soft toe (gradual onset in shadows) and
a soft shoulder (gradual roll-off into highlights). Digital cameras
typically apply a sharper sCurve that was designed to maximize
signal-to-noise ratio and look "punchy" on a calibrated display.

Fuji's film simulations specifically emulate the H&D curve shape, with
the shoulder and toe depths varying per simulation:

- **Provia** — close to a straight gamma ~2.2, mild toe/shoulder
  compression. The "neutral" digital look.
- **Velvia** — pronounced S-curve. Shadows get a deep toe; highlights
  have a soft shoulder. This is the "punchy landscape" curve.
- **Astia** — soft toe, soft shoulder, gentle midtone. This is the
  "skin-friendly" curve: the soft toe keeps shadow detail in dark hair
  and dark skin; the soft shoulder protects highlight roll-off in
  foreheads and shoulders.
- **Classic Chrome** — strong S-curve, lifted blacks (lower toe point),
  and a hard shoulder. The "editorial" curve.
- **ACROS** — long linear midtone with very gentle toe/shoulder. Designed
  to maximize tonal separation in B&W.

**Why this matters for the "Fuji look"**: the soft shoulder is the single
biggest reason highlights in Fuji JPEGs look "creamy" rather than
clipped. The soft toe is the biggest reason shadows look "deep but
detailed." Replicating this in software is straightforward — apply a
parametric curve where `output = (input^k) / ((1-k)*input + k)` for
`k ∈ [0.7, 1.3]` per channel is a common approximation, but a
piecewise-cubic LUT (8 anchor points) gives more control.

### 3.3 Skin Tone Rendering

This is the most distinctive and least-documented part of Fuji's pipeline.
Three things are happening:

1. **Hue protection.** Skin hues (typically a narrow ~30° wedge in
   the orange-red region of the CIE hue wheel) are detected via
   chroma/luminance thresholds, and their saturation is *clamped*
   before any global saturation boost is applied. Without this,
   increasing saturation makes skin look sunburned or plastic.
2. **Luminance–chroma decoupling.** The Y (luminance) and C (chroma)
   channels are processed on different curves. Most digital pipelines
   apply a single tone curve to RGB, which couples luminance and
   saturation. Fuji (and good portrait pipelines like Capture One's
   "Skin Tone" tool) separate them so skin can be brightened without
   increasing its chroma.
3. **Warm-shadow cast.** Shadows across the whole image are biased
   ~100–300K warmer than midtones. This is subtle but is one of the
   things photographers mean when they say a Fuji image "feels warm
   without looking orange." Implementation: a low-amplitude blue
   channel multiplier (~0.96–0.98) in the lower luminance range, then
   ramped back to 1.0 above midtone.

Implementation note: chroma protection in our pipeline can be done with
a `chroma_mask = (hue_in_skin_range) & (chroma < threshold)` followed
by a soft `min(chroma, max_chroma)` clamp on the skin-mask region.

### 3.4 Film Grain

Grain in a real Fuji film is **not** uniform white noise. It is:

- **Clumped** — silver halide crystals form aggregates that vary in
  size and density across the emulsion, producing a "clumpy" texture
  with characteristic autocorrelation length (often quoted as 2–4
  pixels at 100% on a 24 MP sensor).
- **Luminance-correlated** — shadows have *more* visible grain than
  highlights, because the d-min/d-max range is wider in the toe.
  A simple model: grain amplitude `A(luma) = A_max * (1 - luma)^1.2`
  (so shadows get ~1.5–2× the amplitude of highlights).
- **Slightly chroma-modulated** — color negative film has interlayer
  effects that produce subtle color noise, but Fuji's digital grain
  is largely monochrome. Velvia 100F datasheet cites RMS granularity
  of 8 (Velvia 100/100F [2]), which is *very* fine — Fuji's grain is
  among the finest in the industry.
- **Frequency-shaped** — a flat noise distribution would look "TV
  static." Real grain has a soft roll-off in spatial frequency, often
  modeled as `1/f^α` with `α ≈ 1.0–1.5`.

A reasonable digital implementation:

```python
# Pseudocode — not for direct execution
grain_field = gaussian_blur(white_noise, sigma=1.2)   # clumping
grain_field *= (1.0 - luminance) ** 1.2               # luminance correlation
output = input + grain_field * amplitude              # additive
```

This is substantially more natural than a Gaussian noise overlay, which
is what most "grain effect" filters in consumer software do.

---

## 4. Implementing 90–95% Match

### 4.1 The Three Sims at a Glance

The three simulations that appear most in user-facing recipes (and that
are most important for v1):

| Parameter | Classic Chrome | Astia | Provia |
|-----------|---------------|-------|--------|
| Saturation | -1 to -2 (low) | 0 (neutral) | 0 to +1 (slight) |
| Contrast | +1 to +2 (high) | 0 to +1 (gentle) | 0 (neutral) |
| Highlight | -1 (rolled off) | 0 | 0 |
| Shadow | -1 to -2 (lifted) | 0 | 0 |
| Color chrome (warm/cool) | cool greens | warm | neutral |
| Red shift | toward orange | toward yellow | toward orange (mild) |
| Green shift | toward teal | toward yellow-green | toward green (natural) |
| Blue shift | toward cyan | toward blue (natural) | toward blue (natural) |
| Sharpness | low | medium | high (default) |
| Grain | optional, fine | none | none |

These ranges are derived from publicly observable Fuji defaults plus
typical X-Photographer published recipes. They are not from any
documented Fuji specification.

### 4.2 Classic Chrome Parameters in Detail

Classic Chrome is the most distinctive and most-imitated Fuji look. Key
characteristics:

- **Saturation is *low*.** This surprises people — the look is "muted
  editorial" rather than "vivid."
- **Contrast is high.** Combined with lifted shadows and rolled-off
  highlights, this gives a flat-but-punchy look: a long midtone ramp
  with steep ends.
- **Greens are pushed toward teal/cyan.** This is the most brand-
  distinctive part. Implementation: in CIE LCh, rotate green hues
  (90°–150°) by ~10–15° toward blue, *then* reduce saturation by
  ~20%.
- **Reds are pushed toward orange.** Similar technique, ~5–10°
  rotation, ~15% saturation reduction.
- **Slight overall cool cast.** A WB shift of -150 to -300K is
  typical for the look.

A parametric recipe in our engine might look like:

```
saturation: -1.5
contrast:   +1.5
highlights: -1.0
shadows:    -1.5
color_chrome_effect: "warm"
grain:      { size: "fine", strength: 0.3 }
wb_shift:   -200K
```

### 4.3 Astia Parameters in Detail

Astia is the portrait default. Key characteristics:

- **Smooth, gradual contrast.** Almost a "linear" midtone with
  gentle toe and shoulder. This is what makes skin look creamy.
- **Red saturation is reduced.** Reds are *desaturated* by ~10–20%
  relative to global, so lips and skin do not go bright. This is the
  "skin-hue protection" at work.
- **Slight warm bias overall.** ~+100 to +200K WB shift.
- **Highlights are protected** with a soft shoulder; shadows are
  *not* lifted (unlike Classic Chrome).

### 4.4 Provia Parameters in Detail

Provia is the "true to life" simulation. Key characteristics:

- **Saturated but not oversaturated.** A global saturation of 0 to
  +1 with no per-channel overrides.
- **High default sharpness** (Fuji applies a moderate unsharp mask
  by default on Provia).
- **Linear midtone, gentle ends.** A textbook tone curve.
- **Accurate white balance.** No intentional cast.
- **Foliage and skies are rendered "as captured"** but with Fuji's
  characteristic slightly-warm shadow bias.

### 4.5 3D LUT vs Custom Recipe — When to Use Which

- **3D LUT (e.g. `.cube` 17³ or 33³)**: A one-shot color transform.
  Good for matching a *specific* look on a *specific* input. Bad for
  parametric recipes because there is no user-tweakable axis. Best used
  for "Fuji JPEG → sRGB" output transforms or for batch-applying a
  finished look.
- **Custom recipe (parametric)**: A list of named parameters
  (saturation, contrast, WB shift, etc.) that compose a pipeline.
  More work to author but more flexible — the user can tune any
  parameter. Best for our v1 because the whole point of the recipe
  system is to let users express "make it more like Classic Chrome"
  with sliders.

For 90–95% match, **we need a parametric pipeline** and we should use
LUTs only for the per-simulation *finishing* step (a small 3D LUT to
catch non-linear interactions between matrix and curve).

### 4.6 Typical Parameter Values Across Simulations

Rough ranges observed in published Fuji recipes (treat as approximate;
exact values vary by generation of camera body and personal taste):

| Parameter | Typical range | Default |
|-----------|--------------|---------|
| Saturation | -2 to +2 | 0 |
| Contrast | -2 to +2 | 0 |
| Highlight tone | -2 to +1 | 0 |
| Shadow tone | -2 to +1 | 0 |
| Sharpness | -2 to +2 | 0 |
| Noise reduction | -2 to +2 | 0 |
| White balance shift | ±2500K, ±9 (R/B axis) | 0 |
| Color chrome effect | off / weak / strong | off |
| Clarity / midtone contrast | -5 to +5 | 0 |
| Grain strength | off / weak / strong | off |

Important caveat: every published recipe is opinion. The "right" Classic
Chrome for a portrait of a child is not the same as the "right" Classic
Chrome for a street photo in Tokyo. Our v1 should expose all of these
parameters and provide a small number of "canonical" recipes as starting
points, not lock users into a single look.

---

## 5. What We CAN'T Match

Three categories of "Fuji-ness" are out of reach in software:

1. **X-Trans Color Filter Array.** The 6×6 pattern produces
   different demosaicing artifacts and a different effective spatial
   resolution than the standard Bayer (2×2) pattern used in most
   other cameras. Fuji uses this in all X-series APS-C bodies (per
   the X-Trans sensor Wikipedia entry [4]). Some of the
   "Fuji micro-detail" look is a side-effect of X-Trans demosaic
   and cannot be replicated on a Bayer-sourced image. (Note: Fuji's
   medium-format GFX line uses a Bayer CFA, so this is specific to
   APS-C.)

2. **Twenty-plus years of JPEG engine refinement.** The current
   X-Trans processors are the result of continuous iteration since
   the X-Pro1 launch in 2012, building on top of a film-scanning
   pipeline that goes back to the early 1990s. Their default JPEG
   output is the result of a tightly-tuned chain of (matrix →
   noise reduction → sharpening → tone curve → chroma → demosaic →
   write). We can match the *output* of this chain, but we cannot
   match the per-step tuning without access to Fuji's internal
   specifications.

3. **Proprietary ICC profiles.** While the ICC profiles shipped
   with X-Trans cameras are downloadable, they do not include the
   full color matrix used internally for RAW conversion. They are
   "rendering" profiles, not "scanning" profiles. Some reverse-
   engineered profiles exist in the Darktable and RawTherapee
   communities but are not officially sanctioned.

4. **"Film X" feel beyond curve and grain.** Photographers who have
   shot Velvia for decades will sometimes say a digital Velvia sim
   "feels different" from the film. Part of this is the digital
   capture itself (linear sensor response, different dynamic range,
   no reciprocity failure). We cannot recreate reciprocity
   behavior, dye-coupling, or interlayer effects in post. We can
   only approximate the *output* look.

A reasonable v1 target is therefore 90–95% match on a side-by-side
eyeball test for *most* images. Pushing past 95% likely requires
training a model on a Fuji-JPEG/Fuji-RAW pair dataset, which is
post-v1 scope.

---

## 6. References

The following sources informed this document. URLs are given where
publicly available; some are subject to link rot and may require
archive.org retrieval.

1. **Fujifilm corporate history and product lines.**
   Wikipedia, "Fujifilm." <https://en.wikipedia.org/wiki/Fujifilm>
   *(Used for corporate timeline and product catalog.)*

2. **Velvia (Fujichrome Velvia 50, 100, 100F).**
   Wikipedia, "Velvia."
   <https://en.wikipedia.org/wiki/Velvia>
   *(Used for Velvia's documented "very high saturation," fine grain
   (RMS 8–9), ±½ stop exposure latitude, and "highest resolving power
   of any slide film" — up to 160 lines/mm on 35mm.)*

3. **Provia (Fujichrome Provia 100F [RDP III]).**
   Wikipedia, "Provia."
   <https://en.wikipedia.org/wiki/Provia>
   *(Used for Provia's documented "less saturated colors and
   contrast compared to Velvia" and its RMS 8 granularity.)*

4. **Fujifilm X-Trans sensor.**
   Wikipedia, "Fujifilm X-Trans sensor."
   <https://en.wikipedia.org/wiki/Fujifilm_X-Trans_sensor>
   *(Used for the 6×6 CFA pattern, claim of reduced moiré without
   an optical low-pass filter, and confirmation that medium-format
   GFX uses Bayer.)*

5. **RNI (Really Nice Images) — commercial Fuji/film emulations.**
   RNI Films. <https://reallyniceimages.com/>
   *(Used for context on the commercial film-emulation market. RNI's
   "All Films" products are profile-based (gen 4) and adjustment-based
   (gen 5), developed "after real film emulsions." Note: RNI is a
   commercial product and is *not* a free or open reference; the
   techniques used are proprietary.)*

6. **Cambridge in Colour — Color Management / Color Spaces tutorial.**
   <https://www.cambridgeincolour.com/tutorials/color-spaces.htm>
   *(Used as a general reference for CIE color spaces, gamut
   boundaries, and the distinction between device-dependent and
   device-independent spaces. Highly recommended background reading
   for the math of color science.)*

7. **DPReview interview with Fujifilm, CP+ 2017 (Yokohama).**
   <https://www.dpreview.com/interviews/6648162116/cp-2017-fujifilm-interview-we-hope-that-the-gfx-will-change-how-people-view-medium-format>
   *(Used for confirmation that medium-format GFX will continue to
   use Bayer while APS-C stays on X-Trans.)*

8. **Imaging Resource — Dave Etchells, "What's the story with
   Fujifilm's X-Trans sensor tech? Is it really all that different?"**
   <https://www.imaging-resource.com/news/2020/04/20/fujifilm-x-trans-is-it-really-all-that-different>
   *(Authoritative technical discussion of X-Trans; cited in the
   X-Trans Wikipedia article.)*

9. **PetaPixel — "Why and How Fuji Cameras Produce a Strange Purple
   Flare/Grid Artifact" (2017).**
   <https://petapixel.com/2017/02/23/fuji-cameras-produce-strange-purple-flaregrid-artifact/>
   *(A useful reminder that X-Trans II/III have *known* sensor
   artifacts that no software pipeline can fully correct.)*

10. **Fujifilm X-Trans CMOS official product page.**
    <https://fujifilm-x.com/global/products/x-trans-cmos/>
    *(Manufacturer's own description of the X-Trans technology.)*

### Reading list (not directly cited, recommended)

- *The Reproduction of Colour* by R.W.G. Hunt (6th edition) — the
  classic textbook on color science.
- Henry Wilhelm, *The Permanence and Care of Color Photographs* —
  cited in the Velvia Wikipedia article for dye-stability claims.
- Darktable's "Fuji X-Trans" profile source code (open source) —
  contains a public-domain reverse-engineered color matrix for some
  X-Trans sensors.
- RawPedia, "DCP and ICC profiles for Fujifilm cameras" — practical
  guide to the profiles that *are* publicly available.

---

## 7. Open Questions / Blockers for v1

- **Exact Fuji color matrix** for current X-Trans 5 HR sensors: not
  publicly disclosed. We can approximate using downloadable ICC
  profiles plus a small calibration pass against known Fuji JPEGs.
- **Tone-curve shape for each simulation**: not disclosed. We will
  need to sample from published Fuji JPEGs to fit parametric curves.
- **Skin-hue protection thresholds**: not disclosed. The general
  approach is known (chroma clamp on the orange-red hue wedge) but
  the exact hue range and chroma cap will need empirical tuning.
- **X-Trans-specific micro-texture**: cannot be reproduced on a
  Bayer-sourced input. v1 should accept this as a hard ceiling on
  fidelity.

---

[1]: https://en.wikipedia.org/wiki/Fujifilm
[2]: https://en.wikipedia.org/wiki/Velvia
[3]: https://en.wikipedia.org/wiki/Provia
[4]: https://en.wikipedia.org/wiki/Fujifilm_X-Trans_sensor
