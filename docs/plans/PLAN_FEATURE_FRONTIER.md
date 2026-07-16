# Feature Frontier — Eyes, Teeth, Lips optical & anatomical models (beyond the basic enhancers)

**Status:** 📋 RESEARCH (2026-07-11, Opus) · **Parent:** MASTER_PLAN, "different region" frontier flagged
in `PLAN_SKIN_FRONTIER.md` and the R-series exhaustion note.
**Purpose:** The R1–R15 / M1–M6 / I1–I5 backlog exhausted *skin*. Both prior docs name the next
research frontier as **a different region** (eyes/teeth/lips optical models) or **a different axis**
(temporal/perceptual). This doc opens the first. The shipped eye/teeth/lip modules are the *pre-R-series*
equivalent for these regions — channel adjusters, not optical models. The same leap skin made (adjust a
channel → model the physics: R7 chromophores, R9 albedo×shading, R12 dichromatic finish) is available
here and largely untouched by competitors, who also ship only channel adjusters for these regions.

**Grounded in the current code (verified 2026-07-11):**
- `teeth.py` — LAB whitening + threshold detection. No shade grading, no gum/tooth separation, no
  per-tooth structure.
- `eyes.py` / `eye_enhancement.py` — sclera whiten, iris saturation/hue, **catchlight amplify but
  explicitly "never create fakes"**, sclera-vessel inpaint (shipped 2026-07-10). No catchlight
  *synthesis*, no limbal-ring/iris-radial optical model beyond a darken/brighten sculpt, no scleral
  shading.
- `lips.py` — vibrance, tint, gloss (bright-spot boost), texture preserve. No volume/plumping model,
  no dichromatic lip finish, no chapping repair.
- `undereye.py` — LAB L-lift + chroma reduce. No hemoglobin-aware model (the R7/R10 chromophore idea
  never reached the under-eye, where dark circles are *literally* the hemoglobin problem).

This is the honest edge for these regions: after the items below, the eye/teeth/lip space is at the
same "recombination, not research" state the skin space reached.

---

## Part 1 — Eyes (the highest perceived-quality lever per pixel)

The eye is where a viewer looks first; small correct changes read as large quality gains, and small
wrong ones (dead stare, fake glass eyes) read as *worse* than untouched. The shipped enhancer is
conservative by design ("never create fakes") — correct, but it leaves the biggest levers on the table.

### E-EYE-1 — Catchlight synthesis (physically-plausible, gated) ⭐ novel
The shipped path only *amplifies existing* catchlights. The single largest eye-quality lever is
**adding** one when the light source left none (flat/overcast/backlit shots). Risk is the "pasted white
dot" tell. The optical fix: synthesize a catchlight as a *reflection of a plausible key light* on the
corneal sphere — shaped (softbox rectangle / window mullion / ring), positioned consistently across
both eyes from a single inferred light direction, sized to iris radius, with a soft falloff matching a
wet convex cornea (not a flat blur). Gate: only fire when no catchlight is detected AND both irises are
found (consistency requires two). This is R6 in the skin backlog, but the *hard* half (synthesis vs
enhancement) belongs here with the corneal-sphere model. Pairs with the existing iris-circle detector.

### E-EYE-2 — Corneal specular / scleral shading model (the "alive eye")
A real eyeball is a wet sphere: the sclera is *not* uniformly bright — it has a subtle gradient
(brighter toward the catchlight side, shadowed under the upper lid) and a faint limbal shadow where
the cornea bulges. Uniform sclera-whitening (what ships now) flattens this into a "boiled egg" look —
the eye equivalent of plastic skin. The model: after whitening, re-impose a gentle low-frequency
scleral shading gradient (lid shadow on top, ambient occlusion at the inner canthus) driven by the same
inferred light direction as E-EYE-1. This is the I2/I3 "add life back" move, applied to the eye. Small,
high-leverage, nobody does it.

### E-EYE-3 — Iris radial optical model (depth, not just saturation)
`_sculpt_iris` darkens pupil + limbal ring and brightens the body — a crude two-zone sculpt. A real iris
has **radial fiber structure** (crypts, Fuchs' collarette, radiating trabeculae) and a depth gradient
(darker at the pupil, luminous at mid-radius). Model it as a radial function anchored on the detected
iris circle: subtle radial contrast enhancement along the fiber direction (not isotropic), collarette
ring emphasis, and a controlled limbal-ring darkening that respects the actual limbus edge rather than a
fixed fraction. Turns "saturated iris" into "detailed iris." Overlaps R13's oriented-texture machinery
(radial instead of Langer-line).

### E-EYE-4 — Under-eye as a hemoglobin problem (chromophore-aware) ⭐
Dark circles are *predominantly* a hemoglobin/vascular signal (thin periorbital skin over a venous
plexus), not a luminance dip — that's why LAB-L brightening ships them gray/ashy. This is the exact R7/
R10 chromophore idea, and the under-eye is its **best single application**: decompose the periorbital
patch, attenuate the hemoglobin (blue-purple venous) component specifically, and only then lift L. Fixes
the "brightened but still corpse-toned" failure of the current remover. Directly reuses R7 maps if/when
R7 lands; can be prototyped standalone on the periorbital ROI (small, bounded) before R7.

**2026-07-16 spike:** `UndereyeProcessor.attenuate_hemoglobin()` now exists as an
experimental leaf operator. It measures excess hemoglobin against the local cheek ring,
attenuates only the excess, restores source luminance after reconstruction, and composites
only inside the landmark under-eye mask. Synthetic vascular tests cover hue reduction,
luminance preservation, float32 output, and zero-strength identity. It is deliberately not
engine-, recipe-, CLI-, or GUI-wired until real-image visual QA proves it improves dark
circles without erasing anatomical shadows.

### E-EYE-5 — Gaze / eye-symmetry micro-correction (geometry, gated, conservative)
Strabismus-lite and minor gaze asymmetry (one iris slightly off-center vs the palpebral opening) read as
"something's off" without the viewer knowing why. A *tiny*, capped iris-recenter warp (reuse F5 liquify
machinery, iris-circle anchored, sub-2px cap, symmetric-only) can correct it. High wow, high risk —
must be strongly gated and off by default. Flag as creative/advanced, not corrective default.

---

## Part 2 — Teeth (shade science, not "make white")

The shipped whitener is a LAB lift + threshold. "Make teeth white" is exactly the amateur tell —
fluorescent piano-key teeth. Dentistry has a shade science that consumer software ignores entirely.

### E-TEETH-1 — VITA-shade-aware whitening (natural target, not max white) ⭐ novel
Real whitening targets a *natural* shade band, not pure white. Map detected teeth onto an approximate
VITA-shade axis (the standard dental A1–D4 lightness/chroma/hue space), and whiten *toward a
natural-white target within the band* — reducing yellow (b*) and lifting L to a realistic ceiling, never
to paper-white, never removing the natural incisal-vs-cervical gradient. This is the teeth analogue of
C1's memory-color targeting: a locus, not a slider-to-max. The single biggest anti-amateur move for
teeth.

### E-TEETH-2 — Gum / tooth / lip-interior separation
The current threshold detector will catch pink gums and inner-lip as "not white and brighten around
them" or worse, whiten the gum-line edge. Separate the mouth interior into **gum (pink, hemoglobin),
teeth (neutral-yellow), and shadow (inter-tooth gaps + oral cavity)** before any correction, so
whitening never touches gums (grayed gums = uncanny) and gap-shadows stay dark (whitened gaps = "picket
fence"). A small classifier on the existing mouth mask; the enabling structure for everything else here.

### E-TEETH-3 — Incisal translucency & specular preservation
Natural incisal edges are slightly *translucent* (blue-gray) and enamel has soft specular highlights.
Flat whitening kills both → opaque tile look. Preserve (or re-impose) the incisal translucency gradient
and enamel speculars — the teeth version of the I-series "add life back." Needs E-TEETH-2's tooth mask.

### E-TEETH-4 — Per-tooth evening (stain/filling outlier repair)
Individual discolored teeth (single dead tooth, old filling, coffee stain on one surface) draw the eye.
Detect per-tooth lightness outliers within the tooth mask and even them toward the mouth's own tooth
median — the S3 "even the blotch" move at tooth scale. Corrective, high-value for portraits, and
impossible without E-TEETH-2.

---

## Part 3 — Lips (volume & finish, not just color)

Lips ship with vibrance/tint/gloss — a color layer. The frontier is the same optical-finish and form
work skin got in R12/R9.

### E-LIP-1 — Dichromatic lip finish slider (matte ↔ satin ↔ gloss ↔ wet) ⭐
Exactly R12's dichromatic specular/diffuse separation, applied to lips: pull the specular layer off,
then re-emit it along a single **matte → satin → gloss → wet/glass** slider. The current `_add_lip_gloss`
only *boosts existing* bright spots (same limitation as the eye catchlight). Real control means
synthesizing a physically-shaped lip-surface highlight (the characteristic lower-lip crescent) and
being able to *remove* unwanted glare (dry-flash shine) too. Reuses R12 machinery directly — lips are
the second-best home for it after skin.

### E-LIP-2 — Lip volume / plumping via shading (not warp) ⭐
"Plumper lips" done as a liquify warp (the naive approach) distorts the mouth. The optical approach:
lips look fuller when the **shading gradient** implies more convexity — darken the vermilion border
subtly, brighten the lower-lip pillow, add a soft occlusion in the philtrum/corner. This is I3 (AO/
dimensionality synthesis) applied to lips: add *form* via shading, zero geometry change. Natural,
reversible, no warp artifacts. Consumes the R9 shading concept.

### E-LIP-3 — Chapping / flaking / vertical-line repair (lip texture v2)
Dry lips have flaking scale and deep vertical lines. `_smooth` is a gentle bilateral — it either leaves
them or plastics the lip. Targeted repair: detect the high-frequency dry-flake band and the oriented
vertical fissures (Gabor, vertical prior) and attenuate *selectively* with a retention floor — the R13
directional-wrinkle move at lip scale. Keeps natural lip texture, removes only the dryness tell.

### E-LIP-4 — Vermilion-border definition & lip-liner effect (edge optics)
A crisp vermilion border reads as youthful/defined; it blurs with age and low light. Detect the
lip-skin boundary and apply a controlled local-contrast / subtle darkening along it (the makeup
"overline" done optically, capped to avoid the drawn-on look). Pairs with T2 makeup engine but is a
corrective-optical version, not a paint layer.

---

## Cross-region enabler

### E-COMMON-1 — Single inferred light-direction model
E-EYE-1 (catchlight placement), E-EYE-2 (scleral shading), E-LIP-1/2 (lip specular + form), and the
skin I3/I4/R9 work **all want the same thing: one estimate of the scene's key-light direction.** Infer
it once (from existing face speculars / the R9 shading map / catchlight position if present) and share
it. This is the connective tissue that makes the added highlights and shadings *mutually consistent* —
the difference between "each feature retouched" and "one coherently-lit face." Build this and half the
Part 1–3 items get their gating input for free. Strong candidate to do *first*.

---

## Ranking & recommendation

| Rank | Item | Why | Effort | Blocked by |
|---|---|---|---|---|
| 1 | **E-COMMON-1** light-direction model | Unlocks catchlight/scleral/lip-specular consistency; shared input | ~3–5 d | R9 shading (soft) |
| 2 | **E-TEETH-1 + E-TEETH-2** (shade-aware whiten + gum/tooth sep) | Kills the #1 amateur tell (piano-key teeth); small, classical | ~1 wk | — |
| 3 | **E-EYE-4** under-eye hemoglobin model | Fixes the ashy-dark-circle failure; best single R7/R10 application | ~1 wk | R7 (soft; ROI-prototypable) |
| 4 | **E-EYE-1 catchlight synthesis** | Biggest eye lever; genuinely novel vs "amplify only" | ~1 wk | E-COMMON-1 |
| 5 | **E-EYE-2 scleral shading** | The "alive eye"; cheap add-life move | ~3–5 d | E-COMMON-1 |
| 6 | **E-LIP-1 dichromatic finish** | R12 second home; real matte↔wet control | ~1 wk | R12 |
| 7 | **E-LIP-2 lip volume via shading** | Plumping without warp artifacts | ~3–5 d | R9 (soft) |
| 8 | **E-TEETH-3/4, E-EYE-3, E-LIP-3/4** | Completeness (add-life + outlier evening per region) | ~2–3 wk total | region masks |
| 9 | **E-EYE-5 gaze correction** | High wow, high risk; creative/advanced, off by default | ~1 wk | F5 warp |

**Standout recommendation:** build **E-COMMON-1** (shared light-direction) first — it's the enabler for
the whole optical half — then **E-TEETH-1+2**, because piano-key teeth is the most common and most
visible amateur tell and needs no dependency, and **E-EYE-4** because the under-eye *is* the chromophore
problem and the current LAB-lift is measurably wrong (ashy output), giving it a clear before/after story
that M2's erythema/melanin indices can quantify.

**Honest note on scope:** these three regions are where the shipped code is still channel-adjuster-era,
so the optical-model leap has real headroom. Once Parts 1–3 land, eyes/teeth/lips reach the same
"recombination, not research" state skin is in now. The remaining *un*opened frontiers after this are the
**axis** ones the skin doc flagged: temporal/burst coherence (retouch a burst consistently), and
perceptual-preference learning (train the auto-strength selector on human ratings, closing the M6 loop) —
neither is region-specific and both are larger, model-shaped efforts. Say the word and I'll open one.
