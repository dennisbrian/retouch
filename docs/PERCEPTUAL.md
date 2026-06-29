# PERCEPTUAL.md — Perceptual Contract & Pipeline Invariants
> The reference standard for this engine is 像素蛋糕 (PixCake): skin texture and tone are solved
> *simultaneously*, not traded against each other. Every pixel touched must be justified by a
> perceptual goal, not a technical convenience.
>
> Load when: reviewing skin/grading/frequency/geometry changes, or when "tests pass but the image
> looks wrong". Translates the hierarchy in `AGENTS.md` §0 into enforceable rules.

---

## 1. Perceptual Contract

These rules define what "looks right at 100% zoom" means for this engine. They are enforceable.
A violation is an automatic MAJOR finding in the Review Loop, or CRITICAL if it produces an artifact visible at thumbnail size.

### 1A. Skin: Texture and Tone are Simultaneous, Not Negotiable

```
PERCEPTUAL CONTRACT — SKIN {
  rule:      skin tone correction and skin texture preservation must both hold
  forbidden:  smoothing skin to remove tone unevenness (destroys texture)
  forbidden:  preserving texture by leaving tone uneven (fails uniformity gate)
  method:    frequency separation — operate on low-freq band for tone,
             recombine with untouched high-freq band for texture
  test:      Texture Preservation SSIM ≥ 0.92 AND Skin Tone Uniformity PASS — both required
}
```

The PixCake principle: never trade texture for tone, or tone for texture.
Frequency separation is the *only* sanctioned method. Any other approach (Gaussian blur on skin,
bilateral filter on raw pixels, edge-aware smoothing that crosses pore boundaries) is a CRITICAL
finding regardless of how cleanly it tests.

### 1B. Color Identity: Neutral Stays Neutral

```
PERCEPTUAL CONTRACT — COLOR IDENTITY {
  rule:      global tone transfer (style/grading) must not shift neutral gray regions
  forbidden:  hue rotations that drift ΔE > 2.0 on neutral patches
  method:    operate on perceptual uniformity — split-tone via LAB; protect neutral axis
  test:      No Color Drift gate — ΔE ≤ 2.0 on 3 sampled gray patches
}
```

Skin tone is part of color identity. Skin equalization that turns a warm face cool (or vice versa)
is a color identity violation, not just a skin bug.

### 1C. Luminance Hierarchy: Preserve the Tonal Scale

```
PERCEPTUAL CONTRACT — LUMINANCE {
  rule:      highlight and shadow detail that exists in the input must exist in the output
  forbidden:  clipping specular highlights to flat 255 (loses catchlight/garment weave)
  forbidden:  crushing deep shadows to flat 0 (loses hair/fabric texture)
  forbidden:  compressing dynamic range so the image "muddies" — tonal scale must remain perceptible
  test:      No Highlight Clipping (<0.5% pixels at 255) AND No Shadow Crushing (<0.5% at 0)
  exception: grading MAY intentionally clip for stylistic blown-out look — documented in recipe
}
```

The grading stage is the *only* stage permitted to intentionally clip. Any other stage producing
highlight/shadow clipping is a CRITICAL finding — it is a lost-detail bug, not a stylistic choice.

### 1D. Edge Reality: No Halos, No Tearing, No Seams

```
PERCEPTUAL CONTRACT — EDGES {
  rule:      masking and warp transitions must be spatially invisible at 100% zoom
  forbidden:  luminance halos around face/feature/grading boundaries
  forbidden:  warp displacement discontinuities > 2px
  forbidden:  hard mask borders (mask feathered after thresholding)
  method:    feather probability masks before thresholding; use float32 soft blends, never uint8
  test:      No Halo + No Edge Tearing gates
}
```

Halos are the signature artifact of careless compositing. They are never acceptable.
If a mask correction produces an edge artifact at 100% zoom, the fix must come from the mask
construction or the blend weight — not from erasing the halo afterwards.

### 1E. Naturalness: The Uncanny Valley Test

```
PERCEPTUAL CONTRACT — NATURALNESS {
  rule:      output must look like a person, not a plastica rendering of a person
  forbidden:  "plastic skin" — over-smoothed texture with no pore structure
  forbidden:  artificial catchlight (implanted light source that conflicts with scene lighting)
  forbidden:  unnaturally precise mask boundaries (real faces have gradient transitions)
  test:      Natural Output gate — subjective reviewer judgment at 100% zoom
}
```

"Tests pass and looks right at thumbnail" is failure. "Looks right at 100% zoom" is the only success
criterion for naturalness. A single reviewer identifying "plastic skin" or uncanny-valley artifact
fails this contract regardless of objective metrics.

---

## 2. Pipeline Invariants

Hard constraints on what each stage may and may not do.
Violating an invariant produces one stage's side effects appearing in another stage's output.
Invariant violation = automatic CRITICAL finding in Review Loop.

```
PIPELINE INVARIANTS {
  detection.py  → reads pixels, never writes pixels
  parsing.py    → reads pixels, produces masks, never writes pixels
  geometry.py   → moves pixels, never changes color values
  frequency.py  → decomposes bands, never changes mean color
  skin.py       → changes luminance/chroma in skin region, never moves pixels
  eyes.py       → changes local feature pixels, never affects skin region
  lips.py       → changes local feature pixels, never affects skin region
  teeth.py      → changes local feature pixels, never affects skin region
  blemish.py    → inpaints over existing corrected skin, never changes masks
  style.py      → applies global tone transfer, never changes masks or landmarks
  grading.py    → applies final look; only stage permitted to intentionally clip highlights
}
```

### How to use this table

1. **In Planning**: An implementation that crosses an invariant is auto-rejected. Rework the plan.
   - Example: a `skin.py` change that shifts pixel coordinates violates `skin.py → never moves pixels`.
2. **In Review**: Any code that crosses an invariant is a CRITICAL finding, regardless of whether tests pass.
3. **In Blast-Radius reasoning**: If a stage's invariant changes (e.g., `geometry.py` now touches color),
   every downstream stage needs a full visual QA pass — the invariant boundary no longer holds.

### Common invariant violations

| Violation | Why it's wrong |
|----------|----------------|
| `skin.py` calls `cv2.warpAffine` to "reshape" features | Skin stage moves pixels. Use `geometry.py` for warps. |
| `geometry.py` adjusts brightness on warped region | Geometry stage changes color. Tone work belongs to `skin.py` or `grading.py`. |
| `eyes.py` reduces redness outside eye mask | Eyes stage affects skin region. Mask leakage. |
| `parsing.py` post-processes masks via inpainting | Parsing stage writes pixels to source. Mask refinement belongs to consumer. |
| `style.py` alters face landmark positions to "fix" expression | Style stage moves landmarks. Geometry only. |
| `frequency.py` tints the low-freq band | Frequency stage changes mean color. Tint belongs to `skin.py` or `grading.py`. |

---

## 3. The Hierarchy in Practice

From `AGENTS.md` §0:

```
Visual Fidelity > Correctness > Performance > Elegance
```

This means in conflict:

| Conflict | Resolution |
|----------|------------|
| Correct but introduces halo | Revert. Halo is a perceptual failure (§1D). |
| Fast but blurs pore structure | Revert. Texture destruction is a perceptual failure (§1A). |
| Elegant abstraction but output drifts | Revert. Color drift is a perceptual failure (§1B). |
| Passes tests, looks wrong at 100% | Revert. Tests do not override visual fidelity. |

The hierarchy is not advisory. Every contract in this document exists because a specific class of
"tests-pass-but-image-wrong" bug has been observed historically. Abiding by them is not optional.