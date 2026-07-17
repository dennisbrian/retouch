# RESEARCH — Perceptual Calibration, Boundaries & Reference Intelligence (Z-series)

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only)
**Owner ask:** fourth research axis, beyond color rendering
(`RESEARCH_COLOR_RETOUCH_MASSIVE_WINS`), skin optics (`X-series`), and
structure/healing/workflow (`Y-series`) — all three now assigned to
implementation agents.
**This doc's thesis:** the engine now has more *capability* than *calibration*.
Every plan asks "how strong is too strong?" and answers by eyeball (harmony
gates, porcelain looks, feature-focal strengths). There is a published
psychophysics literature that answers several of these questions with data —
nobody in the consumer tool space uses it. Pillar 1 turns that literature into
engine targets. Pillar 2 attacks the two quality boundaries every op shares
(hair-edge alpha, degraded inputs). Pillar 3 upgrades reference-driven color.
Verified absent at HEAD: no matting/trimap, look transfer is Reinhard mean/std
only, no chroma-wavelet/deblock pre-pass, no procedural pore synthesis, no
inter-face consistency op, no facial-contrast metric.

---

## 0. TL;DR ranking

| # | Item | What it unlocks | Effort | Deps |
|---|---|---|---|---|
| **Z1** | **Facial-contrast calibration (Russell)** | Evidence-based eye/lip/brow enhancement targets — S4 strengths chosen by science, not taste | ~3–5 d | none |
| **Z2** | **Skin-homogeneity targets (Fink/Matts)** | The "healthy vs waxy" line, from data — calibrates every evening/smoothing op | ~3–5 d | none |
| **Z3** | **Alpha matting for hair boundaries** | Kills the halo tell in every background blur/replace/grade | ~1–1.5 wk | none |
| **Z4** | **Distribution-transfer look matching (Monge–Kantorovich)** | Reference looks that match *distributions*, not just mean/std — skin-anchored | ~1 wk | none |
| **Z5** | **Group-shot inter-face consistency** | Every face in a group photo lands at the same exposure/WB/locus before retouch | ~1 wk | K4 |
| **Z6** | **Donor-free pore-field synthesis** | Plastic-skin *recovery* when no donor texture exists | ~1 wk | none |
| **Z7** | **High-ISO / JPEG input rescue pre-pass** | Con-floor ISO-6400 and re-compressed inputs stop poisoning skin ops | ~3–5 d | none |

Z1+Z2 are one slice ("perceptual targets"): tiny effort, and their output —
numeric target bands — feeds the auto-colorist (Y3), the harmony gates, the
feature-focal pass (S4), and the ITA scorecard (X9) simultaneously. Z3 is the
highest-leverage engineering item. Z5 wants K4 merged.

---

## Pillar 1 — Evidence-based targets (the calibration the plans keep wishing for)

### Z1 — Facial-contrast calibration ⭐ science nobody in the space uses

**The science:** Richard Russell's facial-contrast line of work (Perception
2009; Porcheron/Mauger/Russell, PLoS ONE 2013; follow-ups across ethnicities)
establishes that **luminance and color contrast between the eyes/lips/brows
and the surrounding skin** is a primary carrier of perceived youth,
femininity, and health — it is *the* mechanism by which cosmetics work, it
declines measurably with age, and the effect replicates across Caucasian,
Chinese, Latin American, and Black African faces (with per-population
magnitudes). This is exactly what the S4 "feature focal pass" and the
`blush`/lip/brow ops manipulate — currently with hand-tuned strengths.

**Codeable plan:** (1) implement the published **facial-contrast metric**
(mean L*/a*/b* difference between feature region and its surrounding annulus,
per feature — masks all exist in `FaceRegions`); (2) tabulate the published
age-band distributions as target bands (data tables in the papers); (3) make
the feature-focal ops **target-seeking**: measure the face's current feature
contrast, move it *toward the healthy-band centile chosen by the recipe*,
never past it — an over-made-up face gets *reduced*, a washed-out flash face
gets restored, and the parameter finally means something ("55th percentile
contrast" vs "lip_enhance 0.4"). (4) Expose the metric in QA output.

**Fairness note:** all contrast measures are *within-face relative*
(feature vs own surrounding skin) — tone-fair by construction, and the
cross-population studies give per-population validation data.

**Tests:** metric reproduces published example values on synthetic faces;
target-seeking is idempotent (applying twice moves <ε); monotone in strength;
I–VI sweep with constant relative contrast → constant metric.

### Z2 — Skin-homogeneity targets: the "healthy vs waxy" line ⭐

**The science:** Fink/Grammer/Matts (Evolution & Human Behavior 2006) and the
Matts skin-color-distribution work show **skin color homogeneity** drives
perceived age/health/attractiveness — but the effect has structure: what ages
a face is specifically the *chromatic* inhomogeneity at the blotch scale
(hemoglobin/melanin mottle), while *luminance micro-texture* (pores) carries
"real skin." Perfectly homogeneous skin does not read younger — it reads
artificial (the uncanny direction every porcelain recipe risks).

**Codeable plan:** define the engine's canonical **homogeneity state vector**
— σ_C at blotch band (X-series Y7/X2 give the band tools), hemoglobin
variance (X2), pore-band energy (shipped H2/pore-spectrum) — and tabulate
target *bands* per look family: "natural" = move σ_C toward the published
young-skin distribution but keep pore energy ≥ 85% of source; "porcelain" =
lower σ_C bound, pore floor still enforced. These become (a) auto-tune
targets for Y3's solver, (b) QA gates replacing today's ad-hoc thresholds,
(c) the definition of "credible photographic equivalent" the cosplay plan
currently leaves to eyeball. Deliverable is mostly a **data table + metric
wiring**, not a new op.

**Tests:** metric stability across resolution/proxy; corpus renders annotated
with state vectors; gate = pore floor violated ⇒ flagged (plastic guard
becomes principled).

---

## Pillar 2 — Boundary & input quality (shared ceilings)

### Z3 — Alpha matting for hair boundaries ⭐ the halo killer

**Gap:** person/hair masks are BiSeNet argmax (+ C3's guided feathering,
default-off). Every background op — `blur_background`, `lens_blur`,
`anime_crystal_void`, background grade/harmonize — composites through a soft
*mask*, not a true **alpha matte**, so hair wisps get haloed, eaten, or
ghosted. This is the #1 "shopped" tell in background-blurred cosplay wig
shots, and it caps the quality of four shipped features at once.

**Method (classical matting, mature literature):** build a trimap
automatically from BiSeNet (erode = FG, dilate-complement = BG, band =
unknown; band width from hair-region scale); solve alpha in the band via
**closed-form matting** (Levin et al., PAMI 2008 — sparse linear system, the
reference method) at proxy resolution on the unknown band only (its cost is
in the band, not the image), refine to full res with the guided filter
(He et al. — fast, edge-aware upsampler, already in `utils.py`). KNN matting
(Chen/Li/Tang 2013) as the cheap alternative if the Laplacian solve is too
heavy. Foreground-color estimation (Levin's or the fast multi-level variant)
so composites re-blend wisps over the new background without color fringes.

**Wire-in:** matte replaces the person mask *only* inside background ops'
compositing step (masks used for statistics stay as-is). Cache per
(image, parse) like face contexts.

**Tests:** synthetic hair-strand renders over checkerboard → matte MSE vs
ground truth; halo width on wig-over-bokeh renders drops measurably;
`detect_halo` QA scores improve on the cosplay corpus; runtime budget
(band-limited solve ≤ ~400ms at proxy).

### Z7 — High-ISO / JPEG input rescue pre-pass

**Gap:** the pipeline assumes clean input. Con-floor reality: ISO 3200–6400
(chroma noise mottles skin exactly at the blotch band → evening ops chase
noise) and re-compressed JPEGs (8×8 block edges on smooth skin → sharpening
amplifies grids, banding detector fires on source). NAFNet denoise exists but
is luminance-oriented and model-gated; nothing deblocks.

**Method:** conservative, classical, detect-then-treat: (1) noise estimate
(MAD of high band in flat regions — machinery exists); (2) **chroma-selective
wavelet shrinkage** (a/b planes only, luma untouched — kills color mottle
with zero texture cost; pairs with Y7); (3) JPEG block-grid detection (8-px
periodic energy in the gradient histogram) → SA-DCT-style or overlapped
smoothing *only on detected block edges in smooth regions*. All gated: clean
input ⇒ byte-identical.

**Tests:** synthetic noise/compression sweeps → downstream evening ops become
stable (same params, noisy vs clean input converge); clean-input identity;
block-grid energy drops without pore-band loss.

### Z6 — Donor-free pore-field synthesis (plastic recovery)

**Gap:** `texture_transplant` needs same-face donor texture; over-smoothed
regions (or heavily-foundationed skin with no pore signal anywhere) have no
donor — those faces are unrecoverable today.

**Method:** procedural, spectrum-matched: measure the target pore spectrum
(radial profile from H2's pore band; or, when absent, the published/corpus
prior scaled by face width), synthesize a **Gabor-noise field** (Lagae et
al.) or steerable-pyramid noise (Portilla–Simoncelli-lite: match radial +
orientation statistics only) with that spectrum, modulate amplitude by the
local smoothing map (only re-texture where texture was removed), add in the
high band. Explicitly *statistical* texture — no cloning, no repetition
artifacts, seed-deterministic.

**Guard:** synthesis amplitude capped relative to the face's remaining
texture (never dominant); H2 pore-spectrum QA must land inside the natural
band, not just "more energy" (the TPR-gaming caveat from the harmony plan).

---

## Pillar 3 — Reference & group intelligence

### Z4 — Distribution-transfer look matching (Monge–Kantorovich upgrade)

**Gap:** `style_transfer.py` is Reinhard **mean/std** matching per LAB
channel — it cannot transfer a bimodal grade (teal-orange), a specific
highlight tint, or any distribution *shape*; results drift when source and
reference content differ.

**Method:** Pitié/Kokaram's two classical tools: (a) the **linear
Monge–Kantorovich closed form** (covariance-based optimal linear map — one
3×3 solve, strictly better than mean/std at equal cost), and (b) **iterated
distribution transfer** (random 1-D projections + histogram matching,
~10–20 iterations) for full distribution shape, followed by their
grain-suppression regularization (gradient-field constraint) to prevent
posterization. Skin anchoring: solve the map on non-skin pixels (or
skin-downweighted), then apply to skin only through the C1 locus-hold gate —
the look moves the scene, skin stays on its corrected locus (the exact
failure S5's plan warns about).

**Tests:** transfer a synthetic bimodal grade — mean/std fails, IDT
reproduces both modes; grain regularizer holds banding delta ≈ 0 (H5);
skin ΔE bounded while background ΔE tracks reference.

### Z5 — Group-shot inter-face consistency

**Gap:** multi-face pipeline retouches faces *independently*; nothing makes
them consistent with each other. Group reality: on-camera flash falls off
with depth (near face +1 stop vs far face), mixed venue light hits faces
differently — after per-face retouch each face is individually "correct" and
the group photo still looks wrong.

**Method:** before per-face processing, measure each face's exposure
(median skin L vs the group median), WB (per-face adapted white via K4's
estimator scoped to the face), and locus offset; solve a per-face low-degree
correction (gain + CAT) that brings faces into a bounded consistency band —
anchored to the *most confident* face (largest/sharpest), never to an
absolute target (a group of different skin tones must stay different — only
*illumination* differences are equalized, measured against each face's own
locus class). Feathered per-face application via existing face hulls.

**Fairness note:** this is illumination equalization, not skin-tone
equalization — the correction is bounded by what the per-face white/exposure
estimates attribute to *lighting*, and the I–VI mixed-group synthetic test is
the acceptance gate (tones preserved, casts equalized).

**Tests:** synthetic group with injected per-face casts/falloff → corrected
spread within band, true tone differences preserved; single-face no-op;
B7's multi-face `style_ref` composes on top.

---

## Bounds / NO-GO

- **No learned aesthetic scorer** (standing NO-GO) — Z1/Z2 are its classical
  replacement: published human data as *fixed tables*, not a trained model.
- **Z1/Z2 targets are recipe defaults and QA bands, never silent auto-edits**
  of identity features; target-seeking ops remain user-strength-scaled.
- **No trimap UI in v1** (Z3 trimap is automatic; brush refinement later).
- **Z6 never synthesizes texture where real texture survives** (amplitude
  modulated by the smoothing map; it is recovery, not decoration).
- **P5 burst / video stays parked.**

## Sequencing vs in-flight work

- Zero overlap with in-flight scopes: K4/K12 agents, X-series
  (`spectral.py`/`skin_chromophore.py`/`specular.py`/`film.py`), Y-series
  (`patchmatch.py`/`perspective.py`/`autograde.py`/`palette_lock.py`/
  `gainmap.py`), body/marks agent. Z-work lands in `matting.py`,
  `face_contrast.py`, `input_rescue.py`, `texture_synth.py` (new) +
  `style_transfer.py` (Z4 extends, owned by no agent).
- Z1+Z2 first (days, unblock better targets for Y3/X9 while those are built).
- Z5 after K4 merges. Z4 anytime; coordinate with S5 palette-grade plan.
- Z3 independent; highest payoff once background ops are exercised by the
  cosplay recipes.

## Sources (canonical; verify exact citations when implementing)

- Russell, R., *A sex difference in facial contrast…*, Perception 2009;
  Porcheron, Mauger, Russell, *Aspects of facial contrast decrease with age…*,
  PLoS ONE 2013 (+ cross-cultural follow-ups) — Z1 metric + data.
- Fink, Grammer, Matts, *Visible skin color distribution…*, Evolution &
  Human Behavior 2006; Matts et al. on skin homogeneity — Z2 data.
- Levin, Lischinski, Weiss, *A Closed-Form Solution to Natural Image
  Matting*, PAMI 2008; Chen, Li, Tang, *KNN Matting*, PAMI 2013; He et al.,
  *Guided Image Filtering*, PAMI 2013 — Z3.
- Pitié, Kokaram, Dahyot, *Automated colour grading using colour distribution
  transfer*, CVIU 2007; Pitié & Kokaram, linear Monge–Kantorovich, 2007 — Z4.
- Lagae et al., *Procedural Noise using Sparse Gabor Convolution*, SIGGRAPH
  2009; Portilla & Simoncelli, IJCV 2000 — Z6.
- Foi, Katkovnik, Egiazarian, SA-DCT deblocking, TIP 2007 — Z7.
