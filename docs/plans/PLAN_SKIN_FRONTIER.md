# Skin Frontier — Research beyond R1–R13 (measurement + "inverse" retouching)

**Status:** 📋 RESEARCH (2026-07-11, Fable) · **Parent:** MASTER_PLAN skin work (Phase 2, R-series)
**Purpose:** The R1–R13 backlog covers *processing* (how to alter skin). Two domains were still
unexplored and are documented here: (1) **measurement science** — how you *prove* skin is better,
not merely different, which is what A1/A2 actually need; and (2) **"inverse" retouching** — adding
life back that smoothing destroys, the opposite move from every flaw-removal idea so far.

This is the honest edge of the well. After this, remaining skin ideas are recombinations of
existing primitives, not new research.

---

## Part 1 — Measurement science (dermatology-grade objective metrics)

The product thesis ("indistinguishable from Retouch4me in blind A/B, beats PixCake on chroma
uniformity") is currently unmeasured — A1 is an owner-blocked corpus task and A2 is blind A/B.
Blind A/B tells you *which* is preferred; it cannot tell you *why* or *by how much*. Dermatology
and cosmetic-imaging research have standardized, citable metrics that turn "looks better" into a
number. These are cheap (all classical) and would make A1/A2 quantitative.

### M1 — Individual Typology Angle (ITA°) and Fitzpatrick auto-classification
`ITA = arctan((L* − 50) / b*) · 180/π`, measured in CIELAB on masked skin. Maps to the six
standard skin-type bands (very light → dark). Two uses:
- **Adaptive parameters:** scale smoothing/highlight/locus strength per ITA band → the C1
  dark-skin acceptance criterion becomes automatic behavior, not a one-off test.
- **Reporting:** state results per ITA band so "works on dark skin" is provable, not asserted.
Trivial to compute; unlocks a lot. (Overlaps buckets F/G from the last sweep — this is the citation.)

### M2 — Melanin Index & Erythema Index (narrow-band reflectance proxies)
Dermatology quantifies pigment and redness from R/G reflectance ratios:
`MI ∝ log(1/R_red)`, `EI ∝ log(R_green/R_red)` (Diffey/Dawson-style). These are the *scalar*
summaries of the R7 chromophore maps. Use as **before/after deltas** to prove S3/R7/R10 reduce
redness without graying (EI down, MI/L* stable). This is the objective version of S3's flagged
"~1.02 mean-a drift."

### M3 — Skin tone uniformity / evenness index
Std-dev (or IQR) of L*, a*, b* over the skin mask, optionally band-limited to the blotch
frequency. This is the literal metric behind "beats PixCake on chroma uniformity" — currently
claimed, never computed. Should become an F11 output and an A2 comparison column.

### M4 — Pore/texture quantification (GLCM / Haralick + power-spectrum)
The plastic-skin problem needs a *texture-preservation* number, not just the F11 detector's
verdict. Gray-level co-occurrence matrix features (contrast, homogeneity, energy) and the
radial power-spectrum slope over the skin mask quantify "is there still pore texture." Compute
in/out ratio → the direct signal for A5 back-off (this is the "plastic-skin metric v2" from the
last sweep's bucket G, now with the specific method named).

### M5 — Gloss / shine quantification
Specular-pixel fraction and specular-area L* percentile over skin. The scalar behind S4 ("oil
removed") and R12 ("finish slider set correctly"). Lets the finish slider report a target gloss
level instead of a blind strength.

### M6 — A perceptual skin-quality composite (the GUI number)
Combine M3 (uniformity) + M4 (texture retained) + M5 (gloss) + EI (M2) into a single 0–100
"skin score," calibrated once against a small human-rated set. Two payoffs: drives auto-strength
selection, and gives users a trustable number. This is genuinely novel in consumer software —
nobody surfaces a dermatology-grounded skin score.

**Why Part 1 matters most right now:** it directly de-risks the A1/A2 bottleneck. Even before the
owner runs the competitor corpus, M1–M5 can be computed on *our own* outputs to catch regressions
and tune S2/S3/C1/C2 objectively. Recommend M1+M3+M4 as an F11 extension — small, high-leverage,
unblocks measurement without waiting on A1.

---

## Part 2 — "Inverse" retouching (add life, don't just remove flaws)

Every skin idea so far *removes* something (blotch, shine, redness, wrinkles). But the deepest
plastic-skin tell is **absence** — smoothing removes the natural micro-variation that signals a
living surface, and no amount of texture *transplant* (S6) fully replaces it because S6 restores
high-frequency luminance, not the other cues below. These are the opposite move.

### I1 — Vellus hair (peach fuzz) preservation ⭐ novel
Real skin is covered in ultra-fine vellus hair that catches rim/side light as a faint bright
haze at the face silhouette and on cheeks. Smoothing annihilates it — this is a subtle but
strong "it's been retouched" tell, and *nobody* in consumer software preserves it. Detection:
oriented high-frequency energy at grazing-light regions (cheek edge, jaw, upper lip), distinct
from pores (which are isotropic dots) by its directionality. Policy: protect this band from
smoothing, or re-inject it from the pre-smooth image at the silhouette. Pairs naturally with the
R13 Gabor-orientation machinery and the R9 shading map (vellus glow lives where grazing light is).

### I2 — Vitality / chroma micro-variation re-introduction ⭐ novel
Living skin has gentle low-frequency chroma variation — slightly warmer cheeks and nose tip,
cooler forehead and jaw (the "three-zone" face colorimetry makeup artists exploit). Aggressive
evening (S3/equalize) flattens this into a mask-like uniform tone. After evening, *re-introduce*
a controlled, anatomically-placed warm-cheek/cool-perimeter gradient (landmarks already give the
zones). The inverse of S3, applied as a finishing touch: even out *random* blotch, then add back
*structured* color. This is what separates "corrected" from "healthy," and it's a natural
consumer of the R9 albedo layer (edit structured color in albedo, form untouched).

### I3 — Ambient-occlusion / dimensionality synthesis
Flat frontal-flash faces lack the soft contact-shadow darkening in creases (nose sides, under
lip, inner eye) that reads as three-dimensional. With the R9 shading map (or the landmark mesh's
concavity), synthesize subtle AO to *add* form to under-lit faces — the inverse of C2's
shadow-lifting, for the opposite failure mode. Gated by an "is this face flat-lit" analyzer
(shading-map dynamic range).

### I4 — Skin radiance / inner-glow synthesis (physically scoped bloom)
K-beauty "glow" is not global bloom — it's a soft light bleed from the *skin's own diffuse
highlights* (forehead, cheekbones, nose bridge), warm-tinted, SSS-colored. Scope the existing
`apply_global_bloom` to the skin diffuse-highlight layer only (from R9/R12 separation),
warm-tint it. Distinct from R12's dewy specular slider: I4 is diffuse-glow, R12 is
specular-finish. Together they are the full "lit-from-within" look.

### I5 — Aesthetic freckle redistribution (not removal)
The freckle-*preservation* work protects existing freckles. The inverse creative tool:
redistribute/rebalance or lightly synthesize freckles following the natural face density
gradient (dense on nose/cheekbones, sparse toward jaw) for the "sun-kissed" editorial look —
increasingly requested, and the opposite of the removal default. Lower priority (creative, not
corrective), but it completes the "we control freckles in both directions" story.

---

## Ranking & recommendation

| Rank | Item | Why | Effort | Blocked by |
|---|---|---|---|---|
| 1 | **M1 + M3 + M4** (ITA + uniformity + texture metrics into F11) | De-risks A1/A2; measures the product thesis; tunes everything objectively | ~3–5 d | — (do now) |
| 2 | **I1 vellus preservation** | The last big plastic-skin tell nobody addresses; genuinely novel | ~1 wk | R13 orientation, R9 shading (soft dep) |
| 3 | **I2 vitality re-introduction** | Turns "corrected" into "healthy"; natural R9-albedo consumer | ~1 wk | R9 (soft) |
| 4 | **M5 + M6** (gloss metric + composite skin score) | The trustable GUI number; drives auto-strength | ~1 wk | M1–M4 |
| 5 | **I4 radiance / I3 AO** | Complete the "lit-from-within" look with R12 | ~1–1.5 wk | R9, R12 |
| 6 | **M2 chromophore indices** | Objective S3/R7 redness proof | ~2 d | R7 (soft) |
| 7 | **I5 freckle redistribution** | Creative completeness, both directions | ~1 wk | freckle work |

**Standout recommendation:** M1+M3+M4 first (measurement is the current bottleneck and needs no
new algorithms), then **I1 vellus preservation** as the next genuinely-novel processing feature —
it attacks the plastic-skin problem from an angle (preserve the fuzz) that neither R9
(decomposition) nor R13 (texture) fully covers, and no competitor touches it.

**Honest note on exhaustion:** with Part 1 (measurement) and Part 2 (inverse) documented, the
skin research space is genuinely mapped end to end — physics (R9/R7/R12), signal processing
(R13), audience (R11), measurement (M1–M6), and the inverse/additive frontier (I1–I5). Further
skin ideas from here are recombinations of these primitives into recipes, not new research
directions. The next new *research* frontier would be a different region (eyes/teeth/lips optical
models) or a different axis (temporal/burst, perceptual-preference learning) — flag if wanted.
