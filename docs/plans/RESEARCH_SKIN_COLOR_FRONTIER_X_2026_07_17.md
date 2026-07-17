# RESEARCH — Skin & Color Frontier (X-series): beyond the shipped Fuji/K/C stack

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only)
**Owner ask:** "more, even more powerful" skin/face/color science than the current
Fuji-style system — the ceiling of what classical, codeable algorithms can reach.
**Positioning:** everything here sits ABOVE what is already shipped and verified at
HEAD today: parametric film engine (`film.py`: linearization, log-density H&D
curves, crosstalk matrix, hue-locked↔filmic-drift master tonemap), K3/K8/K9,
C1 skin locus + hue_unify/chroma_even, C2 sculpt, C4 finish pack, C5 harmonizer,
E1 float core, skin_sss, Hable/Reinhard tone maps (`highlight.py`), and the
in-flight K4 (CAT16 WB) + K12 (export dither) — see
`docs/plans/TODO_COLOR_FIDELITY_2026_07_17.md`. Items K1/K2/K5/K6/K7/K10 from
`PLAN_COLOR_SCIENCE.md` remain valid; this doc goes past them, and X1/X2 are
designed to *absorb* K6/K7's spectral groundwork rather than duplicate it.
**Rule inherited:** classical, deterministic, unit-testable, no learned models,
no absolute tone thresholds.

---

## 0. TL;DR ranking (power × feasibility)

| # | Item | What it unlocks | Effort | Hard dep |
|---|---|---|---|---|
| **X1** | **Spectral film-stock emulation** | The *real* Fuji/Portra ceiling: film that responds to light like film, not a LUT | ~2–3 wk | K4 linear plumbing |
| **X2** | **Shading-free melanin/hemoglobin decomposition v2 (Tsumura-class)** | Independent "redness" vs "pigment" skin sliders — the killer skin-color op, tone-fair by physics | ~1.5–2 wk | K4 (WB leakage kills v1) |
| **X3** | **Dichromatic specular/diffuse separation** | True shine layer: remove/reshape gloss and *recover the skin color underneath* | ~1–1.5 wk | none |
| **X4** | **Highlight purity compression ("path to white")** | Skin highlights roll to cream, never neon yellow/orange — the cinema DRT trait | ~3–5 d | none |
| **X5** | **DIR-coupler + print-through stage for film.py** | The last 5% of film authenticity (edge effects, two-stage color) | ~1 wk | X1 or film.py as-is |
| **X6** | **Auto-WB: skin-locus + near-neutral voting** | One-click venue-cast removal for batch — K4 made *usable* unattended | ~3–5 d | K4 |
| **X7** | **Illuminant-adaptive preferred skin locus** | C1 stops fighting warm/cool ambient grades | ~2–3 d | K4 |
| **X8** | **Local Laplacian detail engine** | Halo-free clarity/texture at any strength (the known ceiling of unsharp-style ops) | ~1 wk | none |
| **X9** | **ITA-binned skin-reproduction scorecard** | Objective "our color vs Fuji-cam vs PixCake" number, per skin-tone bin — fairness made measurable | ~3–5 d | corpus assets |

Recommended attack order after K4/K12 land: **X2 → X4 → X3** (face-first, each
independent, ~1 month total), then the **X1 spectral film block** as the flagship
"nobody else has this" build, X5 riding it. X6/X7 are cheap K4 riders. X9 should
land alongside whichever ships first so wins are measured, not asserted.

---

## X1 — Spectral film-stock emulation (the real Fuji ceiling) ⭐⭐ flagship

**Why current film sims plateau:** every RGB-domain sim (LUTs, and even our
parametric density engine) bakes in ONE lighting condition. Real film's skin
magic is *spectral*: the three emulsion layers have overlapping spectral
sensitivities, so tungsten light, LED spikes, and deep-red skin reflectance land
on the layers differently than they land on a digital sensor + WB. That is why
film skin under mixed light has the specific creamy quality no LUT reproduces —
the LUT was measured under one illuminant and is wrong under every other.

**Pipeline (all published data, all closed-form):**
1. **RGB → reflectance spectrum** via Jakob–Hanika 2019 spectral upsampling
   (precomputed coefficient table → smooth 3-coefficient spectrum per pixel;
   evaluate at ~16–32 wavelengths). This is the same table K6 needs — build once.
2. **Scene light:** multiply by illuminant SPD (D65/tungsten/typical LED — CIE
   publishes all; the estimated CCT from K4's adaptation step selects it).
3. **Exposure per layer:** integrate against the film stock's published
   **spectral sensitivity curves** (Fuji/Kodak datasheets print these for every
   stock — digitize once, ship as small JSON tables).
4. **Characteristic curve:** per-layer H&D curve applied to **log exposure**
   (we already have `_hd_curve_logdensity`; feed it per-layer log-E instead of
   RGB density) → three dye densities.
5. **Density → color:** dye absorption spectra (also in datasheets) ×
   densities → transmission spectrum → integrate against CIE observer → XYZ →
   working RGB. (The crosstalk matrix we ship today is the 3×3 shadow of this
   full spectral step — X1 computes it exactly, per stock, per illuminant.)

**Cost control:** run at 16 wavelengths on a proxy-resolution image, fit the
result as a per-image 3D LUT (33³), apply the LUT at full res — spectral
accuracy at LUT speed. Deterministic, cacheable per (stock, illuminant).

**Tests:** color-checker under D65 vs tungsten produces *different* film
renderings matching qualitative datasheet behavior (tungsten warms shadows more
than highlights); all-zero params → byte-identical; existing Fuji preset A/B —
the spectral path should land near the LUT under D65 (validates the model) and
*diverge exactly where LUTs are known wrong* (non-D65 scenes).

**Why this is the "more powerful than Fuji-style" answer:** it upgrades the film
system from "a look" to "a physical process," and every stock becomes 3 small
data tables instead of a hand-tuned parameter set.

---

## X2 — Shading-free melanin/hemoglobin decomposition v2 ⭐⭐ the killer skin op

**Gap:** `skin_chromophore.py` uses fixed literature recolor directions on
gamma-encoded sRGB; the S1 audit measured WB leaking 2–5× into melanin and
mel/hb entangled. So today no op can answer "reduce the *redness* but keep the
*freckles*" — which is THE most-requested skin-color edit (rosacea, flush,
acne redness) and the one PixCake-class tools do with a network.

**The classical method that actually works (Tsumura et al., SIGGRAPH 2003):**
1. Work in **log-RGB of linear values** (post-K4, casts are already adapted out
   — the leakage the audit measured dies here).
2. Remove **shading**: in log space, shading is a translation along the
   illuminant direction (1,1,1)·k. Project it out → a 2D pigment plane.
3. In that plane, extract the two pigment axes by **per-image ICA constrained
   near the published melanin/hemoglobin extinction directions** (start from
   the literature axes, let ICA refine within a ±15° cone — per-image
   adaptation without losing physical identity).
4. Per-pixel (mel, hb) densities → edit → recompose exactly.

**Deliverable ops:** `hb_even` (compress hemoglobin variance → redness/blotch
evening that *provably* leaves melanin marks untouched), `hb_shift`
(flush/pallor control), `mel_even` (guarded, off by default — melanin evening
is identity-adjacent). Upgrades the existing under-eye hemoglobin op and R10
mole logic onto a sound substrate.

**Fairness note (why v2 is safer than v1):** shading removal + per-image axes
make the decomposition relative to *this face's* pigment distribution — no
absolute thresholds anywhere. Validate on Fitzpatrick I–VI synthetics (K7's
forward Kubelka–Munk generator, once built, is the perfect test oracle: 
generate known (mel, hb) → verify recovery). X2 and K7 are natural siblings.

**Tests:** synthetic KM skin with known chromophores → recovered axes within
tolerance at every ITA bin; hb edit leaves mel map byte-stable; WB sweep after
K4 moves mel estimate <10% (vs 2–5× today).

---

## X3 — Dichromatic specular/diffuse separation ⭐ true shine control

**Gap:** `specular.extract_specular` is a luminance margin-above-baseline — it
finds *where* shine is but cannot separate the specular layer from the diffuse
skin under it. So shine ops darken highlights instead of *removing the gloss
and revealing skin*.

**Method (dichromatic reflection model):** skin reflectance = diffuse (skin
chroma) + specular (illuminant color). Per pixel in a skin region: the diffuse
chromaticity is well-approximated by the local skin-locus chromaticity; the
specular component is the residual along the illuminant chromaticity (known
after K4). Solve the 2-term mixture per pixel (closed form, 2 unknowns / 3
channels — *over*determined, unlike P4's makeup problem) → specular intensity
map + reconstructed diffuse image.

**Unlocks:** (a) shine removal that reconstructs plausible skin color in the
hotspot (vs today's gray-ish darkening); (b) K-beauty water-glow: remove the
broad shine, re-add a *shaped tight* gloss along `lighting.py`'s key direction;
(c) partial recovery of skin color in near-clipped highlights (the diffuse
estimate extends under the specular cap). All tone-fair: the separation is
anchored to the face's own diffuse locus.

**Tests:** synthetic Lambertian+Phong skin sphere → recovered specular map
matches ground truth; diffuse reconstruction hue-stable across I–VI; shine
removal at 100% leaves pore texture intact (specular is low-band + sparse).

---

## X4 — Highlight purity compression / "path to white" ⭐ cheap cinema trait

**Gap:** the film engine's master tonemap controls hue *skew*, and Hable/
Reinhard shape luminance — but nothing compresses **purity** as luminance
approaches clip. Digital's tell: bright saturated skin/lips/neon rolls to
*yellow/orange at full chroma*; film and cinema DRTs (AgX, ACES 2.0 class)
desaturate toward white along the last stop — highlights "bloom to cream."

**Method:** in scene-linear (post-K4 plumbing), above a luminance knee Y_k,
scale chroma (in OKLab, about the adapted white) by a smooth function of
log(Y/Y_k) reaching 0 at clip; couple the rate to the existing filmic-drift
slider so "digital-clean ↔ filmic" now controls skew *and* purity together.
~40 LOC inside `film.py::_master_tonemap` + a `highlight_purity` param.

**Tests:** saturated ramp to clip shows monotonically falling chroma above the
knee, hue constant (uses K3's gamut math for the boundary); skin highlight
patch A/B renders cream vs today's yellow; identity below the knee.

---

## X5 — DIR couplers + print-through (film.py stage 2)

Two authenticity effects the density engine still lacks: **(a) DIR/interlayer
edge effects** — development inhibitors couple layers spatially: implement as a
density-domain unsharp where each channel's high-band subtracts a weighted mix
of the *other* channels' low bands (3×3 coupling matrix × spatial kernel) —
this is film's characteristic edge "snap" and saturation-dependent line color;
**(b) print-through** — negative densities pushed through a *second* H&D pair
(print paper curves, also in datasheets) — two-stage color is why optical
prints have richer blacks and a specific skin warmth single-stage sims miss.
Both are small once X1's data plumbing exists; both default OFF per stock.

---

## X6 — Auto white balance (skin-locus + near-neutral voting) — K4 rider

K4 gives *manual* correct WB; batch work needs the estimate too. Combine two
robust cues: (a) the P4 §20.3 neck-anchored bare-skin locus vs the K7/X2
plausible-skin manifold (skin is the one known-reflectance object in every
portrait); (b) bright near-neutral voting (top-decile low-chroma pixels).
Weighted fusion → estimated illuminant → CAT16 correction, confidence-gated
(skip below confidence, exactly the `lighting.py` discipline). Tests: known
cast injected on corpus → recovered within threshold; I–VI sweep (the skin cue
must be locus-relative, never "skin is this color").

## X7 — Illuminant-adaptive preferred locus — K4 rider

C1's preferred-locus targets are fixed; perceptually, preferred skin moves
with the adapted white (warm ambient → preferred skin sits warmer). Condition
the locus table on K4's adapted CCT (small published shift model, ~2 data
columns) so hue_unify stops fighting deliberate warm/cool grades. Cheap, kills
the "corrected skin looks wrong inside the grade" class of complaints.

## X8 — Local Laplacian detail engine (halo-free clarity ceiling)

`_add_clarity`/`clarity_split`/`local_clarity` are unsharp-class — they halo at
strength, which caps how hard any recipe can push "texture" or "soft-but-
detailed." The fast local Laplacian filter (Paris et al. 2011; Aubry et al.
fast approximation, ~O(N log N)) is the reference halo-free detail manipulator
— one engine, negative-to-positive detail at any strength, no ringing.
Implement as a third `smooth_engine`/clarity backend and A/B against guided.
Face-visible on every "clarity" and mid-band op at once.

## X9 — ITA-binned skin-reproduction scorecard (make wins measurable)

One benchmark script + JSON baseline: per ITA bin (the dermatology-standard
Individual Typology Angle, computed from L*/b*), measure (a) ΔE00 to the
per-bin preferred locus, (b) skin hue-angle spread, (c) σ_C uniformity, (d)
X2's hb/mel variance, before/after each recipe, across the corpus (needs the
§6a asset acquisition). Every X-item above then ships with a scorecard delta
instead of adjectives, and regressions on any tone bin block by construction —
fairness as a gate, not a promise.

---

## What NOT to build (bounds)

- **Learned tone/color transfer or neural film emulation** — out of scope per
  standing rules; X1 achieves the film ceiling classically with public data.
- **Full inverse Kubelka–Munk per pixel (color → 4 skin params)** — ill-posed
  at 3 channels (same trap as P4); X2's 2-chromophore log-linear model is the
  correct classical resolution. K7 *forward* remains a test oracle only.
- **Per-pixel spectral processing at full resolution** — always proxy+LUT
  (X1's cost-control pattern); native spectral per-pixel is 10–30× cost for
  no visible gain.
- **Skin "beautification" priors that move everyone toward one tone** — every
  X-item is relative to the subject's own locus/manifold; X9 enforces it.

## Sequencing vs in-flight work

- X2/X3/X4 are independent of each other; all want **K4 merged first** (X2
  hard-requires it; X3/X4 want the adapted white).
- X1 wants K4's linear plumbing and shares the Jakob–Hanika table with K6 and
  the datasheet tables with X5. Build X1 → X5 as one block.
- X6/X7 are K4 riders (days each). X8 anytime. X9 alongside the first ship.
- No file overlap with the running fix agent (body stage/marks/harmony) or the
  K4/K12 agents (`white_balance.py`, export boundary) — X-work lands in
  `spectral.py` (new), `skin_chromophore.py` (v2 rewrite), `specular.py`,
  `film.py`, `grading.py` clarity backend.

## Sources (canonical methods; verify exact citations when implementing)

- Jakob & Hanika, *A Low-Dimensional Function Space for Efficient Spectral
  Upsampling*, Eurographics 2019 (RGB→spectrum tables).
- Tsumura et al., *Image-based skin color and texture analysis/synthesis by
  extracting hemoglobin and melanin information*, SIGGRAPH 2003 (X2's core).
- Shafer, *Using color to separate reflection components*, 1985 (dichromatic
  model); Mallick et al., *Beyond Lambert* / SUV space, CVPR 2005 (X3).
- Paris, Hasinoff, Kautz, *Local Laplacian Filters*, SIGGRAPH 2011; Aubry et
  al., fast LLF (X8).
- AgX (T. Sobotka) / ACES 2.0 DRT development notes — purity compression /
  path-to-white behavior (X4).
- Kodak/Fujifilm film datasheets — spectral sensitivities, dye spectra, H&D
  curves (X1/X5 data).
- Chardon et al., ITA skin-tone classification (X9 binning).
