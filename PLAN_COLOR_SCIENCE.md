# Color Science — Implementation-grade research (skin + color, all codeable now)

**Status:** 🟢 PARTIAL BUILD (2026-07-11, Haiku) · **Parent:** MASTER_PLAN Tier C (color) + Phase 2 skin. · **Shipped this session:** K3 gamut compression, K8 ΔE color-QA gate, K9 subtractive saturation (already live). Open: K1/K4/K7/K10 substrate + moat items.
**Purpose:** The user asked for research that is *"able to code."* This doc is deliberately
implementation-grade: every item names the **algorithm, the math, the target module, and the codeable
plan**, not a direction. It opens **color-appearance science** — the layer above the OKLab/LAB
converters already in `color_science.py` / `color_space.py` — plus skin-color math the R-series never
made concrete.

---

## ⭐ HIGHEST-WIN PRIORITY (sorted across all 3 research docs — build in this order)

Sorted by **(win ÷ effort)**: real defect fixed + visible result + no dependency ranks highest.

| # | Item | Doc | Win | Effort | Dep | Why it's ranked here |
|---|---|---|---|---|---|---|
| 1 | **K3 gamut compression** | color | HIGH | 3–5 d | none | Fixes a *real current defect* (saturated skin/lips/costume hard-clips → hue shift + posterize). Visible on any saturated photo. Uses OKLab already shipped. | ✅ **DONE 2026-07-11** — `find_gamut_intersection` (OKLab bisection) + `gamut_compress` (hue/lightness-preserving rolloff, identity in-gamut → byte-identical golden path) in `color_science.py`; wired in `grading.py:_apply_gamut_compress` via `settings["gamut_compress"]` (default ON, no-op in-gamut). CLI `--gamut-compress`. Test `tests/test_color_science_k3k9.py`. Render `test_output/color_science_2026-07-11/K3_gamut_compress.png`. |
| — | ~~K9 subtractive saturation~~ | color | ✅ **ALREADY SHIPPED** | — | — | **Verified 2026-07-11:** `apply_subtractive_saturation` in `color_science.py`, wired in `grading.py:25/312` via `saturation_mode="subtractive"`. Confirmed: darkens saturated patch (luma 0.479→0.465), identity at boost 0. R1's density-saturation half is done. |
| 3 | **K8 ΔE color-QA gate** | color | MED-HIGH | 2–3 d | none | Adds CIEDE2000 skin-hue safety to F11/A5. Testable vs 34 Sharma pairs. Closes the color QA loop. | ✅ **DONE 2026-07-11** — `delta_e_2000` (CIEDE2000, verified vs Sharma 34 pairs) + `detect_color_drift` (in→out skin ΔΕ/Δh budget gate) in `qa_detectors.py`; `run_all` returns `color_drift`; `engine.py` consumes it with `COLOR_DRIFT_THRESHOLD` + QAWarning. Test `tests/test_color_qa_deltae.py`. Render `test_output/color_science_2026-07-11/K8_color_drift_demo.png`. |
| 4 | **K4 CAT16 white balance** | color | HIGH | 3–5 d | K1 soft | WB as chromatic adaptation → fixes venue-cast at the root (H3/R11c symptom). |
| 5 | **K1 CAM16 appearance model** | color | FOUNDATIONAL | ~1 wk | none | Reconciles a status bug (C6 claims CAM16, ships OKLab). Unlocks K2/K4/K5/K10. Build when you want the substrate. |
| 6 | **E-TEETH-1+2 shade-aware whiten** | feature | HIGH | ~1 wk | none | Kills the #1 amateur tell (piano-key teeth). Classical, no dep. |
| 7 | **E-EYE-4 under-eye hemoglobin** | feature | HIGH | ~1 wk | R7 soft | Current LAB-lift ships circles ashy — measurably wrong; clear before/after. |
| 8 | **K7 forward Kubelka–Munk skin** | color | WOW | ~1.5 wk | none | Codeable half of the P1 moonshot; physically-valid skin-color generator no consumer tool has. |
| 9 | **P4 skin↔makeup unmixing** | promax | HIGHEST MOAT | 3–4 wk | P1/K7 | Deepest cosplay moat (separate paint from person), but heaviest — do after K7 lands as its prior. |

**Do #1 (K3) first**, then K8 → K4. **K9 turned out already shipped** (verified this session) — one less thing to build; R1's saturation half is live via `saturation_mode="subtractive"`. **#5 (K1)** is the foundation if you'd rather build substrate-first. **#8 (K7)** is the one "wow" item that's still fully codeable.

---

**Verified starting point (2026-07-11):** `color_science.py` ships OKLab/OKLCh (Ottosson matrices),
`measure_skin_state`, preferred skin loci, `skin_chroma_std`, `resolve_locus_override`. `color_space.py`
ships LAB/LCh f32. **CAM16-UCS is claimed in C6 ("shipped inside C1") but there is no CAM16 code in
either file — only OKLab.** That gap is item K1 and should be reconciled against the C6 status.

---

## K1 — Real CAM16(-UCS) appearance model (reconcile the C6 claim) ⭐ foundational
**Gap:** C6 is marked ✅ "appearance-space substrate (OKLab/CAM16-UCS)" but the code is OKLab-only. OKLab
is a *perceptual* space; it is **not** an appearance model — it has no surround, no adaptation, no
lightness/chroma/hue *attributes* that respond to viewing conditions. Real memory-color and cross-media
work (K5, P3-output) need CAM16.

**Codeable plan:** implement CAM16 forward/inverse in a new `appearance.py` (pure NumPy, ~200 LOC —
it's a fixed sequence of matrix mults + nonlinearities):
1. sRGB→XYZ (have it), XYZ→CAT02 LMS via the CAT02 matrix.
2. Chromatic adaptation to the adopted white (D-factor from surround).
3. Post-adaptation cone response (Hunt-Pointer-Estevez), nonlinear compression
   `(F_L·R/100)^0.42`.
4. Opponent signals → correlates **J** (lightness), **C** (chroma), **h** (hue), **M** (colorfulness),
   **Q** (brightness). CAM16-UCS: `J' = 1.7·J/(1+0.007·J)`, `M' = ...`, then `a'=M'·cos h`, `b'=M'·sin h`.
Test: round-trip J'a'b'→sRGB within ΔE<1 on a color-checker; validate against published CAM16 worked
examples (the standard has numeric test vectors). **Fully deterministic, fully unit-testable, no models.**
Everything below that says "in appearance space" depends on this.

---

## K2 — Hue-linearity–corrected skin hue-line (the Abney/blue-shift fix) ⭐ codeable, high-value
**Science:** LAB and even OKLab are *not* hue-linear — constant-hue lines curve (the Abney effect, worst
in blues; measurable in skin's orange-red too). `skin.locus`/`unify_hue_line` currently operates in a
space where "same hue angle" ≠ "same perceived hue," so unifying to a hue *angle* drifts perceived hue
across lightness.

**Codeable plan:** replace the hue coordinate in the skin-locus math with a **hue-linear** one. Two
options, both codeable now: (a) do the hue-line work in CAM16 hue `h` (K1) which is hue-linear by
construction; or (b) apply a small **hue-correction LUT** (the published "hue angle → corrected angle"
tables from the OKLab/IPT hue-uniformity literature) to OKLCh before unifying. Target: `color_science.py`
+ `skin.py` `unify_hue_line`. Test: a synthetic skin ramp (constant physiological hue, varying L) should
map to a *constant* corrected-hue angle; assert max hue drift drops vs the current OKLCh path.

---

## K3 — Gamut-aware chroma compression (perceptual, not clipping) ⭐ codeable now
**Science:** After any saturation/locus move, out-of-gamut colors are currently hard-clipped per channel
→ hue shifts and flat "posterized" saturated skin/lips/costume. The pro move is **gamut mapping**:
compress chroma toward the gamut boundary *along a constant-hue, constant-lightness line*, preserving hue
and lightness, rolling off only chroma.

**Codeable plan:** in OKLCh or CAM16 (K1): for each pixel, find the max in-gamut chroma `C_max(L,h)` by a
few Newton/bisection steps on the sRGB-boundary test (Ottosson publishes the exact
`find_gamut_intersection` for OKLab — ~30 LOC, already the reference implementation), then apply a soft
knee `C_out = C_max·f(C_in/C_max)` (smooth compression above ~80% of boundary). Target: new helper in
`color_science.py`, called at the end of `grading.py` saturation and `skin.locus`. Test: neon-pushed
skin/red costume stays hue-stable (Δh≈0) instead of clipping; assert no channel-clip hue shift on a
saturation sweep. This is R1's spiritual partner but for the *output* end.

---

## K4 — Chromatic-adaptation white balance (CAT16, not Kelvin/tint) ⭐ codeable
**Science:** WB currently ships as Kelvin+tint offsets in a simple space. Real WB is a **chromatic
adaptation transform** — the same math the eye uses to discount the illuminant. Doing it as LAB/RGB
offsets shifts hues non-uniformly (skin goes green/magenta off-axis).

**Codeable plan:** implement CAT16 (or reuse CAT02 from K1): estimate source white (gray-world or the
existing WB picker), define target white, build the diagonal adaptation matrix in cone space
`M⁻¹·diag(ρ_t/ρ_s)·M`, apply as a 3×3 in linear RGB. ~40 LOC in a `white_balance.py` helper; wire behind
the existing `white_balance_kelvin`/`tint` params (map Kelvin→white point via the Planckian locus, which
is a small closed-form/LUT). Test: neutral gray card → neutral under adaptation; skin hue stable across a
Kelvin sweep (the current path's failure). Also fixes the venue-cast problem H3/R11(c) attack, at the
root.

---

## K5 — Memory-color rendering in appearance space (skin/sky/foliage anchors) ⭐ codeable
**Science:** Preferred reproduction shifts *memory colors* toward remembered ideals (skin slightly
warmer/lighter, sky more saturated-blue, foliage greener) — the core of every "pleasing" film sim, done
principled. `color_science.py` already has skin loci; this generalizes it and does it in CAM16 (K1) so
the shift is a defined J/C/h nudge, not an ad-hoc LAB push.

**Codeable plan:** define anchor regions (skin via mask, sky via blue+top-of-frame+low-texture, foliage
via green+texture) → for each, a soft membership weight → pull each toward its published memory-color
target in CAM16-UCS with a falloff that vanishes away from the anchor (so non-memory colors are
untouched). Target: `grading.py` new pass, or `harmonizer.py`. Test: skin/sky/foliage move toward
targets, a color-checker's non-memory patches stay within ΔE<1. This is a codeable, measurable "why our
color is pleasing" story that beats a static LUT because it *adapts to the actual colors present*.

---

## K6 — Spectral-upsampling skin relighting (metamerically correct) ⭐⭐ moonshot-but-codeable
**Science:** All current relighting/WB happens in 3-channel RGB, which is metamerically wrong under
colored light (two RGB-equal skins diverge under tungsten). The pro-grade fix is **spectral**: upsample
each RGB pixel to a plausible reflectance spectrum, multiply by an illuminant SPD, integrate back to RGB.

**Codeable plan (yes, codeable):** use the **Jakob-Hanika 2019 spectral upsampling** method — a
precomputed coefficient table maps RGB→3 polynomial coefficients defining a smooth reflectance curve.
Ship the table (small), evaluate the curve at ~16–32 wavelengths, multiply by a target illuminant SPD
(D65/tungsten/LED SPDs are published), integrate with CIE color-matching functions → new RGB. ~150 LOC +
data in a `spectral.py`. Scope to skin ROI for cost. Test: relighting a synthetic flat skin patch under a
tungsten SPD matches a reference spectral render; RGB-only relight visibly diverges. This is the
metamerically-correct version of relight/WB — genuinely beyond any consumer tool and still 100% classical.

---

## K7 — Physiological skin color synthesis (forward Kubelka–Munk, small & codeable) ⭐
**Science:** The P1 moonshot in `PLAN_SKIN_PROMAX.md`, reduced to its *codeable core*. Two-flux
Kubelka–Munk over two layers (epidermis melanin, dermis hemoglobin) has **closed-form reflectance** —
no iterative solver needed for the forward pass.

**Codeable plan:** implement `skin_reflectance(melanin_frac, blood_frac, oxygenation, thickness)` →
spectrum → RGB, using published absorption coefficients for melanin, oxy- and deoxy-hemoglobin
(tabulated, ship as data) and the KM two-layer closed form. *Forward* only first (given params → color):
this alone gives a **physically-valid skin-color generator** for (a) building better preferred-loci per
ITA band, (b) validating that an edit stays on the "real skin" manifold, (c) the P4 makeup-unmixing skin
prior. The inverse fit (color → params) is a later, harder pass. ~200 LOC + coefficient tables in
`spectral.py`/`skin_optics.py`. Test: sweeping melanin/blood produces a locus matching real
skin-tone distributions (ITA range); oxygenation at constant blood moves sallow↔flushed. The **codeable
half** of P1.

---

## K8 — Delta-E gated, hue-preserving auto-grade QA (measurable color safety) ⭐ codeable, cheap
**Science:** F11 QA has no *color* fidelity gate — nothing asserts a grade didn't wreck skin hue or push
memory colors implausibly. CIEDE2000 (or CAM16 ΔE') gives a per-region color-error budget.

**Codeable plan:** add a `qa_detectors.py` color check: compute ΔE2000 on skin, and Δh specifically, in→
out; flag if skin Δh exceeds a band (skin should shift ≤ a few degrees) or if any memory region leaves
its plausible box. CIEDE2000 is a fixed ~60-LOC formula (reference implementations abound), fully
testable against the Sharma test-data table (34 published pairs). Feeds A5-style back-off for *color*,
not just texture. Cheap, immediately useful, closes the M2/M3 measurement loop on the QA side.

---

## K9 — OKLab-native film density & subtractive saturation (R1, made concrete) 
**Science:** R1 ("subtractive/density saturation") backlogged the *idea*; here's the *codeable* method.
Film saturates by dye-density multiplication in a **subtractive (CMY) log space**, so saturated colors
darken instead of going neon.

**Codeable plan:** convert linear RGB → density `D = -log10(clamp(rgb))` → to CMY density → scale CMY
densities by the saturation factor (optionally per-dye for film-specific curves) → back to RGB via
`10^-D`. ~30 LOC in `grading.py`/`film.py`. The nonlinearity *is* the film look — no LUT needed. Test: a
saturation sweep on a bright patch shows luminance *dropping* as chroma rises (vs HSV/LAB scaling which
holds L flat). This unblocks R1 with an actual implementation.

---

## K10 — Constant-luminance saturation & the Helmholtz-Kohlrausch correction ⭐ codeable, subtle-but-pro
**Science:** Two paired effects the current saturation ignores: (1) HSV/LAB chroma scaling changes
*perceived* lightness even at constant L (H-K effect: saturated colors look lighter); (2) "constant-
luminance" saturation should hold *actual* luminance Y, not L*. Together they cause saturated skin/lips to
read too light or too dark.

**Codeable plan:** apply an H-K luminance correction (published Fairchild/Nayatani formula — a function
of chroma and hue that adjusts L to hold *perceived* brightness) after any chroma change; and offer a
true constant-Y saturation mode (scale chroma in a space, then renormalize to preserve linear Y). ~40 LOC
in `color_science.py` + hook in `grading.py`. Test: a saturation boost holds a perceived-brightness metric
flat (H-K corrected) vs drifting (uncorrected). Small, invisible-until-you-see-it, unmistakably pro.

---

## K11 — Structured chroma re-introduction, colorimetrically (I2, made concrete)
**Science:** The I2 "vitality" idea (add warm-cheek/cool-perimeter variation back after evening), given a
*colorimetric* recipe so it's codeable and safe.

**Codeable plan:** using landmark zones (have them), build a smooth per-zone target-offset field in
CAM16 or OKLab: cheeks/nose +warm (small +a, +C), forehead/perimeter/jaw −warm; magnitude = a fraction of
the *pre-evening* measured variation (so you restore what evening removed, not invent). Apply as a
low-frequency additive chroma field, luminance untouched. ~50 LOC in `skin.py`/`makeup.py`. Test: after
S3 evening + this pass, chroma variance returns toward source *structure* (zone-correlated) while random
blotch variance stays low (M3 distinguishes them). Turns I2 from prose into a diff.

---

## K12 — Perceptual banding elimination (R5 dither, done in the right space) 
**Science:** R5 ("blue-noise dither at export") backlogged the idea; the codeable subtlety is that dither
must be applied in the **display-encoded** space at the quantization step, amplitude = ±0.5 LSB, blue-noise
mask (not white noise, not ordered) to push quantization error into high spatial frequencies the eye
discounts.

**Codeable plan:** precompute/ship a blue-noise tile (void-and-cluster, or use a published 64×64 mask),
tile it over the image, add `(mask−0.5)` LSB in the 8-bit-encoded domain just before `astype(uint8)`. ~20
LOC in `io.py` export. Test: an 8-bit gradient after a heavy grade shows no visible contour; measured
first-difference histogram has no quantization spikes. Pairs with the E1 float pipeline. Concrete R5.

---

## Ranking & recommendation

| Rank | Item | What it is | Effort | Blocked by |
|---|---|---|---|---|
| 1 | **K1 CAM16(-UCS)** | Real appearance model; reconciles the C6 claim | ~1 wk | — (foundational) |
| 2 | **K3 gamut compression** | Hue-stable saturated skin/costume, no clip | ~3–5 d | K1 (or OKLab now) |
| 3 | **K4 CAT16 white balance** | WB as adaptation, fixes venue cast at root | ~3–5 d | K1 |
| 4 | **K8 ΔE color-QA gate** | CIEDE2000 skin-hue safety in F11/A5 | ~2–3 d | — |
| 5 | **K9 subtractive saturation** | Concrete R1 (film density in log CMY) | ~2 d | — |
| 6 | **K2 hue-linearity fix** | Abney-corrected skin hue-line | ~3–5 d | K1 (soft) |
| 7 | **K7 forward KM skin optics** | Codeable half of P1; physical skin-color generator | ~1.5 wk | — |
| 8 | **K5 memory-color rendering** | Principled "pleasing color" > static LUT | ~1 wk | K1 |
| 9 | **K10 H-K / constant-Y sat** | Perceived-brightness-stable saturation | ~2–3 d | K1 (soft) |
| 10 | **K11 structured chroma (I2)** | Codeable vitality re-introduction | ~2–3 d | I2 zones |
| 11 | **K12 blue-noise dither (R5)** | Concrete banding kill at export | ~1 d | E1 |
| 12 | **K6 spectral relighting** | Metamerically-correct relight (moonshot, codeable) | ~2 wk | K7 SPDs |

**Standout recommendation:** **K1 (real CAM16)** first — it's foundational, it *reconciles a status
discrepancy* (C6 claims CAM16 but ships OKLab), and it's a fixed, fully-testable ~200-LOC block with
published test vectors, so it's low-risk to implement. Then the cheap high-value trio **K3 + K4 + K8**
(gamut safety, adaptation WB, color QA gate) — each is a few days, each fixes a *real current defect*
(clip hue-shift, venue cast, no color QA), and each is unit-testable against published reference data. If
you want one "wow science" item: **K7** (forward Kubelka–Munk skin) is the codeable half of the P1
moonshot and gives you a physically-valid skin-color generator that nothing in the consumer space has.

**Every item here is classical, deterministic, and unit-testable against published reference data** — no
models, no training, no downloads (except small coefficient/SPD/blue-noise tables). That's the "able to
code" bar: each row could be a PR next week.
