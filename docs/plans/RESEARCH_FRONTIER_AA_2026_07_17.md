# RESEARCH — Frontier AA-series: post-f8e1880 status + internet-sourced next axes

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only on production code)
**Owner ask:** reconcile the research backlog against `f8e1880` ("calibrated
retouch quality cores"), then continue researching — internet sources where the
internal K/X/Y/Z docs are exhausted — toward further algorithm enhancement.
**Method:** grep/read verification of f8e1880 wiring at HEAD, plus targeted web
research (citations in §Sources; verify exact citations when implementing).
**Rules inherited:** classical/deterministic, unit-testable, no learned models,
tone-fair, no absolute intensity thresholds, no identity-changing defaults.

---

## 1. f8e1880 status reconciliation — shipped vs wired

**Implementation update (2026-07-17, post-research):** the first wire-up
slice below is now delivered and covered by focused regression tests. K12 is
intentionally still open: the engine has no single float-to-8-bit delivery
boundary, so wiring dither earlier would violate its own contract.

| Item | Module | Status |
|---|---|---|
| K4 CAT16 white balance | `white_balance.py` | **LIVE** — `grading.white_balance_lch` delegates to `white_balance_cat16`; both engine call sites (face + no-face paths) flow through it |
| K12 blue-noise dither | `precision.py` | Core + float plumbing live (`to_float`/`to_uint8` in engine/grading); **`to_uint8_dithered` has zero non-test callers** — delivery-boundary refactor still open, exactly as `TODO_COLOR_FIDELITY` §3 warns |
| Y1/Y8-lite PatchMatch heal | `patchmatch.py`, `heal.py`, `blemish.py`, `freckle.py` | **LIVE, opt-in** through `heal_engine="patchmatch"` in CLI/GUI/recipes; Telea remains the default. Automatic compact repairs use direct exemplar replacement because `seamlessClone` can reintroduce a dark defect on uniform patches. |
| X4 highlight purity | `film.py`, `params.py`, `gui.py` | **LIVE** through `film_highlight_purity`; exposed in the Film & Analog Effects UI and adopted conservatively by Provia, Astia, and Classic Chrome. |
| X2 chromophore v2 | `chromophore_v2.py`, `perf_optimizations.py` | **LIVE, opt-in** as `hb_even` and `hb_shift` after ordinary redness work. Both use the face skin mask; neither changes melanin. |
| Z0–Z2 perceptual metrics | `perceptual_metrics.py` | Observational by design (per TODO_Z); spike script + `calibrated_natural_v1` prototype recipe exist |
| Z7 input rescue | `input_rescue.py` | Detection-only by design (correct per plan) |
| Z3 matting spike | `matting.py` | Spike only (98 LOC); not wired into background compositing |

**Takeaway:** f8e1880 was a *capability* commit. This follow-up converts X2,
X4, and automatic PatchMatch into delivered controls; K12 remains the one
unresolved delivery-boundary item before moving to new algorithms.

### 1a. Wire-up backlog (highest value ÷ effort in the whole portfolio)

1. **Completed: X4 exposure.** `film_highlight_purity` is a ParamSpec and GUI
   control, with conservative Fuji-film recipe adoption.
2. **Completed: X2 consumers.** `hb_even` / `hb_shift` are mask-bound,
   default-off operations on `chromophore_v2`.
3. **K12 delivery boundary:** float-to-exporter refactor; call
   `to_uint8_dithered` only at the 8-bit write; 16-bit stays un-dithered.
4. **Completed: PatchMatch into auto paths.** `blemish.py` and
   `FreckleRemover` share a `heal_engine` selector; Telea stays the default.
5. **Z3 wire-in** behind background ops once halo QA passes (gate per TODO_Z).

### 1b. Still open from prior series (not re-planned here)

K2 Abney hue-linearity; `lighting.py` → sculpt/relight wiring; K1 (superseded
by AA4 below); K7 Kubelka–Munk; X1/X5 spectral film; X3 dichromatic specular;
X6/X7 K4 riders; X8 local Laplacian; X9 ITA scorecard; Y2–Y7; Z4–Z6.

---

## 2. New research — AA-series (internet-sourced, beyond K/X/Y/Z)

### 0. TL;DR ranking

| # | Item | What it unlocks | Effort | Deps |
|---|---|---|---|---|
| **AA1** | **Luma-guided chroma restoration (JPEG 4:2:0)** | Kills lip/blush/costume edge fringing before skin ops amplify it | ~2–3 d | Z7 detector |
| **AA2** | **Bilateral guided upsampling for the proxy path** | Full-res fidelity for *every* op at once; proxy pipeline stops blurring detail on upscale | ~1–1.5 wk | none |
| **AA3** | **Physically-based film grain (Boolean model)** | Grain that behaves like film (density-dependent, resolution-independent), not overlay noise | ~3–5 d | film.py |
| **AA4** | **Hellwig–Fairchild 2022 CAM16 mod** | One build covers K1 substrate **and** K10 H–K (J_HK/Q_HK correlates) | ~1 wk | replaces K1+K10 |
| **AA5** | **Perception-calibration pack #2 (gloss/redness/teeth/sclera)** | Z1/Z2-style published targets for four more shipped op families | ~3–5 d | Z0 contract |
| **AA6** | **Kee–Farid perceived-retouching score** | One validated "how retouched does this look" number per render — the global plastic/identity guard | ~3–5 d | none |
| **AA7** | **Mixed-illuminant spatially-varying WB** | The venue endgame: LED wash + flash corrected per-region, not globally | ~1.5–2 wk | K4 ✅, Z3 solver |
| **AA8** | **Y6 de-risk note (ISO 21496-1 published)** | Gain-map HDR export is now "just engineering" | n/a | K12 boundary |

### AA1 — Luma-guided chroma restoration for JPEG inputs (Z7 rider)

**Gap:** virtually all corpus JPEGs are 4:2:0 — chroma at half resolution,
upsampled bilinearly by every decoder. Result: chroma bleeding at saturated
edges (lips, blush boundary, wig/costume against skin), which `hue_unify`,
`chroma_even`, and sharpening then amplify. None of Z7's noise/deblock scope
covers it.

**Method:** re-upsample the chroma planes with the **guided filter** (already
in `utils.py`) using full-res luma as the guide — the classical luma-guided
chroma reconstruction result; strictly better than bilinear wherever luma and
chroma edges coincide (they do, on the edges we care about). Gate on detected
subsampling provenance (extend `input_rescue.py` with a half-res chroma-energy
check). Clean/4:4:4 input ⇒ byte-identical.

**Tests:** synthetic 4:2:0 round-trip → edge chroma MSE beats bilinear;
identity on 4:4:4; lip-edge fringe width drops on corpus fixtures.

### AA2 — Bilateral guided upsampling for the proxy path ⭐ architecture-level

**Gap:** the proxy pipeline (>2048px → process at proxy → upscale result)
upscales the *output image*, discarding whatever full-res detail the proxy
never saw. Every op pays this fidelity tax at high resolution.

**Method (Chen, Adams, Hasinoff — BGU, TOG 2016):** instead of upscaling the
result, fit a **low-res bilateral grid of local affine color transforms**
mapping proxy-input → proxy-output, then *apply the grid to the full-res
input*. Detail is preserved by construction (the transform is smooth; the
detail comes from the full-res source); bleeding/halo suppressed by the
bilateral-space locality. *Guided Linear Upsampling* (Liu et al. 2023) is the
simpler/faster modern variant to benchmark against.

**Scope guard:** BGU models per-pixel *color* transforms — geometric ops
(reshape, heal, liquify) and synthesized content cannot ride it. Split the
pipeline: geometric/synthesis ops keep the current path; the color-like global
stages (3, 5, 6) and per-face color ops go through BGU. Start with the global
stages only (lowest risk, biggest area).

**Tests:** proxy render + BGU vs native full-res render → ΔE and high-band
energy gap shrinks vs current upscale path; no bleeding across strong edges
(bilateral-grid octave sweep); runtime neutral or better.

### AA3 — Physically-based film grain (film engine rider)

**Gap:** `_add_grain` is overlay noise — uniform, resolution-tied, exposure
independent. Real film grain is a **Boolean model** of silver-halide grains:
variance depends on local density (strongest in mid-tones, vanishing near
D-min/D-max), scales with grain radius, and is resolution-independent.

**Method (Newson, Delon, Galerne — IPOL 2017 + follow-ups):** the published
Gaussian approximation of the Boolean model gives real-time synthesis: per
pixel, grain variance = f(local density, grain radius, pixel area) — closed
form — driven by the film engine's *density* planes (we already compute them
in `film.py`). Ship as `grain_engine="physical"` beside the current overlay;
per-stock grain radius in the stock tables (rides X1's data plumbing later).
The TOG 2023 "Film Grain Rendering and Parameter Estimation" paper adds
grain-parameter fitting from scans if we ever want stock-matched grain.

**Tests:** grain variance vs density curve matches the analytic model;
resolution sweep → constant apparent grain size in output-referred units;
zero-strength byte-identity.

### AA4 — Hellwig–Fairchild 2022 CAM16 modification (supersedes K1 + K10)

**Finding:** the K-series plans K1 (CAM16 substrate) and K10
(Helmholtz–Kohlrausch saturation) as two builds. The literature has since
merged them: **Hellwig & Fairchild 2022** (Color Res. & Application) extend
CAM16 with H–K-corrected lightness/brightness correlates (J_HK, Q_HK) — one
model covers both backlog items, and `colour-science` (BSD-3) ships a
reference implementation usable as a **test oracle** (not a runtime dep).
Follow-up work (Seong 2025, CIECAM16-based H–K lightness via heterochromatic
brightness matching) confirms the direction is current.

**Action:** update `PLAN_COLOR_SCIENCE.md` K1 to target the Hellwig 2022
variant and mark K10 as absorbed. No new effort estimate change (~1 wk).

### AA5 — Perception-calibration pack #2 (the Z1/Z2 pattern, four more ops)

Extend `perceptual_metrics.py` + the Z0 measurement contract with published
data for op families that today run on taste. Same discipline as Z1/Z2:
within-face-relative or index-based measures, observational first, bounded
deltas, never demographic inference.

- **Gloss/radiance (calibrates X3, `skin_sss`, water-glow):** Ikeda et al.
  2021 (Int. J. Cosmetic Science) distinguishes **radiant vs oily-shiny vs
  matte** reflection and shows radiance (not gloss quantity) drives
  younger/healthier impressions, with facial position mattering. Deliverable:
  a specular-band state (coverage × tightness × intensity margin over the
  face's own diffuse baseline, `extract_specular` machinery) with a target
  band between matte and oily. Shine ops become target-seeking instead of
  strength-scaled.
- **Redness bounds (calibrates X2 `hb_shift`/`hb_even`):** Re et al. 2011
  (oxygenated-blood colour-change thresholds for perceived facial redness/
  health) gives measured perception thresholds for redness change — the
  natural bound for how far a hemoglobin edit may move before it reads as a
  different face state.
- **Teeth (calibrates `teeth_whiten`):** the dental **WIO whiteness index**
  (optimized CIE whiteness for teeth) plus published perceptibility (~2.8
  ΔWIO) and acceptability (~6.5 ΔWIO) thresholds; whiteness perception was
  stable across gender/age/culture in a 500-observer five-country study.
  Deliverable: `teeth_whiten` becomes WIO-target-seeking with a natural-range
  cap — no more chiclet teeth at high strength.
- **Sclera (calibrates `eyes.py` whitening):** Provine et al. 2013: reduced
  *redness/yellowness* drives healthier/younger/more-attractive ratings, but
  **super-white sclera reads younger only — not healthier or more
  attractive** → hard cap on brightening. Russell et al. 2014: aging = rising
  sclera saturation (yellowing), falling brightness → the correct op is
  de-yellow-first (b* toward the face's own young-sclera axis), brighten
  second, both bounded. `_remove_sclera_vessels` already handles redness.

**Tests:** each metric follows the Z0 gates (no mutation, resolution
stability, confidence on thin masks); op sweeps move the paired metric
monotonically; targets expressed as bounded deltas from the input.

### AA6 — Kee–Farid perceived-retouching score (the global plastic guard)

**The science:** Kee & Farid, PNAS 2011 — a perceptually validated 1–5 scale
of "how retouched does this photo look," built from **8 summary statistics**:
4 geometric (mean/σ of the warp-field magnitude over face and body) and 4
photometric (mean/σ of local smoothing/sharpening filter extent + SSIM
statistics), validated against 350 observers on 450 before/after pairs.

**Our structural advantage:** Kee–Farid must *estimate* those quantities from
before/after pairs — the engine **knows them exactly**: the reshaping
displacement field, the per-region smoothing maps, and before/after SSIM are
all internal state. Deliverable: compute the 8-statistic vector per render,
expose it in QA output next to Z1/Z2, and calibrate a fixed monotone 1–5
mapping on our own corpus once (deterministic fit, stored coefficients — a
data table, not a runtime model). Recipes gain a "perceived-retouching
budget"; the harmony gates gain a single top-level number that catches
combined geometric+photometric overtreatment that no per-op gate sees.

**Tests:** zero-op render scores floor; score monotone in smoothing strength
and warp magnitude independently; stable across proxy resolutions.

### AA7 — Mixed-illuminant spatially-varying white balance (venue endgame)

**Gap:** K4 corrects **one global illuminant**. The convention-hall reality
that motivated it — LED wash + on-camera flash + colored stage light — is a
*mixture*, and a global CAT leaves the residual on whichever region lost the
vote (typically half the face).

**Method (Hsu, Kavusi, Durand et al., SIGGRAPH 2008):** model the scene as a
per-pixel **blend of two illuminants**; recover the blend field by solving a
**matting-Laplacian** system over dominant material colors — literally the
same Laplacian machinery Z3's closed-form matting builds (shared solver, two
consumers). Faces-first scoping: use the face's own skin locus as the
known-reflectance probe (X6's cue) to anchor the two illuminant estimates;
confidence-gated with global-K4 fallback.

**Sequencing:** after Z3's solver exists and X6 lands. Two-light assumption
only; three+ lights ⇒ fall back to global. Effort ~1.5–2 wk, medium-high risk
(needs the mixed-light corpus for QA).

### AA8 — Y6 de-risk note (no new work, update the plan)

ISO 21496-1 is now **published (ISO 21496-1:2025)** and supported in Android
15, iOS 18 / macOS 15, and Chromium on Windows; `libultrahdr` encodes the ISO
variant. Y6's spec risk is gone — it is now pure engineering behind the K12
delivery-boundary refactor (same exit point, as the Y-doc already noted).

---

## 3. Recommended attack order

1. **K12 delivery boundary (+AA8 rides it).** Keep output float-resident to
   the exporter, dither only 8-bit delivery, and retain 16-bit precision.
2. **AA5 + AA6 calibration/QA pack** — cheap, and every later flagship then
   ships with measured (not asserted) wins. Pairs naturally with X9.
3. **K2 + `lighting.py` wiring** — unchanged from the K-queue; K2 grows more
   urgent the more `hue_unify` ships.
4. **AA2 BGU proxy upgrade** — architecture-level fidelity for everything.
5. **Flagships by owner preference:** X1 spectral film (+X5, AA3 riders) or
   X3 dichromatic + Y3 auto-colorist (now with Z0–Z2 targets to aim at).
   AA7 after Z3 matures. AA1 anytime as a filler.

## Bounds / NO-GO (inherited + new)

- AA6's 1–5 mapping is fitted **once, offline, on our corpus** and shipped as
  fixed coefficients — no runtime learning, no aesthetic scoring of *people*
  (it scores the *edit*, not the face).
- AA5 targets are QA bands and recipe caps, never silent auto-edits; sclera/
  teeth targets are index-based (WIO, b*-axis), never "white = healthy" claims
  on the person.
- AA7 never exceeds two illuminants and never runs unconfident; AA2 never
  carries geometric/synthesis ops.

## Sources

- Hsu, Mertens, Paris, Avidan, Durand — *Light Mixture Estimation for
  Spatially Varying White Balance*, SIGGRAPH 2008 (AA7).
- Chen, Adams, Wadhwa, Hasinoff — *Bilateral Guided Upsampling*, TOG 2016;
  Liu et al., *Guided Linear Upsampling*, 2023 (AA2).
- Newson, Delon, Galerne (+ Faraj) — *Realistic Film Grain Rendering*, IPOL
  2017; *A Stochastic Film Grain Model for Resolution-Independent Rendering*;
  *Film Grain Rendering and Parameter Estimation*, TOG 2023 (AA3).
- Hellwig & Fairchild — *Brightness, lightness, colorfulness, and chroma in
  CIECAM02 and CAM16*, Color Res. Appl. 2022; + *Extending CIECAM02 and CAM16
  for the Helmholtz–Kohlrausch effect*, 2022; Seong et al. 2025 follow-up;
  `colour-science` `hellwig2022` reference implementation (AA4).
- Ikeda et al. — *Facial radiance influences facial attractiveness…*, Int. J.
  Cosmetic Science 2021 (AA5 gloss).
- Re, Whitehead et al. — *Oxygenated-blood colour change thresholds for
  perceived facial redness, health, and attractiveness*, 2011 (AA5 redness).
- Luo et al. — *Development of a whiteness index for dentistry (WIO)*, J.
  Dentistry 2009; *Investigation of the perceptual thresholds of tooth
  whiteness*, J. Dentistry 2017 (AA5 teeth).
- Provine et al. — *Red, yellow, and super-white sclera*, 2013; Russell et
  al. — *Sclera color changes with age…*, 2014 (AA5 sclera).
- Kee & Farid — *A perceptual metric for photo retouching*, PNAS 108(50),
  2011 (AA6).
- ISO 21496-1:2025; Android Ultra HDR v1.1 docs; google/libultrahdr (AA8).
