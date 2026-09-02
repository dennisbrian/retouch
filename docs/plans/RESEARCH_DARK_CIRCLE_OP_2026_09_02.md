# Under-eye dark-circle op: root cause, v2 redesign, corpus calibration (2026-09-02)

**Status:** v2 shipped (`retouch/undereye.py`, dispatch in `perf_optimizations.py`);
calibration provisional on the cosplay-heavy DSCF corpus — see §7.
**Predecessor:** `RESEARCH_POST_EPSILON_FACEOP_REAUDIT_2026_09_02.md` §6 (finding only).
**Harnesses:** `scripts/qa/dark_circle_op_study.py` (per-eye v1-vs-v2 metrics, sheets),
`scripts/qa/dark_circle_op_summarize.py` (tables below),
`scripts/qa/dark_circle_engine_check.py` (real dispatch, recipe vs keys-off).
**Artifacts:** `test_output/dark_circle_op_study/` (`study.json`, `sheets/<stem>.jpg`,
`sheets/<stem>_grid.jpg`, `engine/<stem>_<recipe>.jpg`, logs).

---

## 0. Summary

- The shipped `dark_circles` / `undereye.darken_removal` op (v1) was inert on every
  face: its detector kept the darkest ≥100 px component of the landmark under-eye
  polygon that sat ≥15 L below the surround — which on real portraits is the **lower
  lash line and eye corners** (the polygon's top edge *is* the lower-lid contour). On
  146 corpus eyes, **80% (median) of v1's |ΔL| mass landed inside the dilated eye
  contour**, and the mean lift on under-eye skin was **0.03 L at strength 100** (max
  0.05). The lift was also double-capped (clip to 30 then ×0.3 → ≤9 L). 55 of 128
  recipes set the key (0.10–0.60 effective); none of them did anything.
- v2 rebuilds the op: support = polygon extended 0.28·IED down into the tear trough
  (0.06·IED sideways), lash margin 0.08·IED and the eye contour excluded with a
  feathered edge, gated by the skin mask, tapered with distance below the lid;
  darkness = masked low-pass L (σ 0.04·IED) relative to a cheek-ring median
  (smoothstep 4%→12% *of the face's own reference*, tone-invariant); lift = strength ×
  low-frequency gap (texture preserved), single 30 L cap, plus 75% of the a/b gap
  (concealer-style warmth); ROI-confined LAB roundtrip, `restore_outside_support`.
- Corpus (73 scored faces, 146 eyes): v2 puts **4% (median)** of its mass in the lash
  band, **0 px** inside the eye contour, **0 px** outside support; mean skin lift at
  strength 1 is **9.3 L** (p10 1.5, p90 17.7). Tone probe (image ×0.55): v2's relative
  lift is retained ×1.15 (invariant); v1 only "retained" because lashes are always
  15 L dark. 9/146 eyes correctly get ~0 weight (under-eye brighter than the cheek).
- Engine path verified on real renders (`clear_skin_v1`, effective 0.30 + puffiness
  0.20): under-eye-hull p99 3–12 levels, eye hull 0, crescent-shaped heat, no edge.
- **Known limitation:** the corpus is cosplay-heavy; on aegyo-sal / under-eye contour
  makeup (DSCF7204, 6963, 4463) the brown contour line reads as shadow and is partly
  lifted at high strength. A chroma guard would kill pigmented (warm) dark circles on
  Fitzpatrick IV–VI, so it was not added; recipe strengths (0.10–0.30 typical) keep
  it subtle. Owner call on recipes that set ≥0.45 (§7).

## 1. Method

- Corpus: the 83-image DSCF set (`test_output/detection_recall_study/corpus.txt`),
  2048 px proxy, `RetouchEngine` `natural` with all face ops off for contexts, then the
  real parser (`eng._parser.parse`) for masks; largest face per image. The 10
  hygiene exclusions from the re-audit §7 (hand/background "largest face", degenerate
  landmarks: 4554/4555/4589/4590/4592/4596/4597/4599/4600/4618) are recorded, not
  scored → 73 faces, 146 eyes.
- v1 is replicated verbatim in the study script (`v1_process`); v2 is the shipped
  `UndereyeProcessor.process` called exactly as the dispatch calls it (IED, landmark
  eye contour, `regions.skin`). Both at strength 1.0 (upper bound); sheets at 25/60/100.
- Zones per eye: *lash* = eye contour dilated 0.08·IED; *skin* = v2 support > 0.5.
  Reported: fraction of Σ|ΔL| inside the lash zone, mean/p95 |ΔL| on skin, max, px
  changed inside the eye contour and outside support.
- Tone probe: the same image ×0.55 (L scale 0.56), both ops re-run.
- Engine check: recipe render vs the same recipe with `dark_circles`,
  `undereye_darken_removal`, `undereye_puffiness_reduction` forced 0, same cached
  contexts, through `_process_face_core` (multi-face images go through the
  ProcessPool fallback, so the probe scripts must be files, not stdin).

## 2. v1: why it never worked

`UndereyeAnalyzer.detect_dark_circles` + `UndereyeRemover.brighten_dark_circles`:

| defect | evidence |
|---|---|
| polygon top edge = lower-lid landmarks 7/163/144/145/153/154/155 → lashes are inside the mask, and after the parser's 0.08·IED feather so is part of the eye | polygon height/IED med 0.14 (p10 0.11, p90 0.23): it covers the lid, not the tear trough |
| absolute `median − 15 L` threshold, then keep components ≥100 px | fires on 146/146 eyes — always the lash line (40+ L dark) |
| `darkness = clip(ref − L, 0, 30)` then `× strength × 30/100` | max lift 9 L at strength 1; measured max 4 (med), 8 (max) |
| whole-image LAB roundtrip + `blend_masked` | ~1-level drift in the mask halo |

Per-eye at strength 1.0 (146 eyes):

| measure | v1 | v2 |
|---|---|---|
| Σ\|ΔL\| fraction in lash zone | **0.80** (p10 0.48, p90 0.91, max 0.96) | **0.04** (p90 0.06, max 0.15 — feathered edge) |
| mean \|ΔL\| on under-eye skin | **0.03** (max 0.05) | **9.29** (p10 1.53, p90 17.71, max 27.2) |
| p95 \|ΔL\| on skin | 0.00 | 30 (cap) |
| px changed inside eye contour | — | **0** |
| px changed outside support | — | **0** |

## 3. v2 design (`retouch/undereye.py`)

All radii are fractions of IED (constants `_UE_*` at the top of the module):

1. **Support** (`build_undereye_support`): polygon > 0.5 → `_dilate_down` by
   0.28·IED with 0.06·IED sideways reach (upper-half-ellipse kernel anchored at its
   bottom row) → remove the eye contour dilated by the lash margin 0.08·IED → feather
   0.06·IED → multiply by the *feathered* inverse of the lash exclusion (a hard cut
   here left a 0.76 step against the lash band, visible as a grey wedge on DSCF4576)
   → hard-zero inside the raw eye contour and beyond a 3× feather halo → taper
   1.0→0.35 with distance below the polygon (`distanceTransform`) → × skin mask.
   Without an eye contour (legacy `repair` path), the top `lash_r` rows of the polygon
   are shaved instead (no extra dilation — that erased 0.12-IED-tall polygons).
2. **Reference ring**: support dilated 0.02→0.12·IED, minus the eye contour dilated
   0.12·IED, ∩ skin. Median LAB of the *masked low-pass*. Found on 146/146 eyes
   (median 1035 px); fallback to the support's own median if < 50 px.
3. **Signal** (`UndereyeAnalyzer.analyze`): normalised-convolution Gaussian low-pass
   (σ 0.04·IED) over pixels outside the lash margin (the first draft used the wider
   ring exclusion here and starved the estimate right under the lid);
   `rel = (ref_L − L_lp) / ref_L`; `weight = smoothstep(rel, 0.04, 0.12)`.
4. **Lift**: `ΔL = strength · (ref_L − L_lp) · weight · support`, clip [0, 30];
   `Δa, Δb = 0.75 · strength · (ref − lp) · weight · support`, clip ±8. Only the
   low-frequency gap is added, so pores/texture survive (synthetic test: high-pass
   std 3.15 → 3.65 under a 30 L lift). `strength` = fraction of the shadow removed.
5. Puffiness reuses `reduce_puffiness_chroma` on `weight · support`.
6. ROI-confined float LAB; uint8 inputs are rounded once; `restore_outside_support`
   makes everything outside the support bit-exact (checked: 0 px on 146 eyes).

Dispatch (`perf_optimizations._process_face_core`) now passes `ied`, the **landmark**
eye contour (BiSeNet's eye class collapses on ~65% of portraits — the visibility-gate
study) and `regions.skin`. The `dark_circles` / `undereye_darken_removal` alias
(`e73d3ba`, max not sum) is unchanged.

## 4. Corpus calibration

Signal distribution (146 eyes, strength-independent):

| measure | med | p10 | p90 | max |
|---|---|---|---|---|
| ring reference L | 171 | 154 | 199 | 219 |
| rel. darkness, median over support | 0.08 | −0.01 | 0.17 | 0.28 |
| rel. darkness, p90 over support | 0.21 | 0.08 | 0.31 | 0.46 |
| shadow depth, L (ref − median lp L) | 13.6 | −2.1 | 29.0 | 53.7 |
| shadow b (ref_b − lp b; + = bluer) | 0.96 | −1.1 | 2.8 | 4.4 |
| weight mean over support | 0.35 | 0.08 | 0.54 | 0.67 |
| fraction of support with weight > 0.5 | 0.49 | 0.11 | 0.75 | 1.00 |

Weight buckets: 9 eyes < 0.05 (op ~off, e.g. DSCF4551 whose under-eye is 12–19 L
*brighter* than its ring), 54 eyes 0.05–0.3, 83 eyes ≥ 0.3.

The 4%→12% band: below 4% is normal orbital shading on a lit face (the p10 of the
median rel. darkness is −1%, i.e. many eyes sit at 0–4% and are left alone); by 12%
the region reads as a visible circle on every sheet reviewed (6693, 4570, 4576).
`shadow b` is small but positive (under-eye slightly bluer than cheek) on most eyes,
which is what the 0.75 a/b pull corrects; the ±8 clamp was never reached.

Strong asymmetries (DSCF4559 L 17.7 / R 2.3; 4598, 4613, 4570–4573 mirrored) are
**yaw**: the far eye of a ¾ face is foreshortened, its support is small and its weight
low (0.08–0.10). The near eye's lift is correct on the sheets. Not a lighting artifact.

Tone probe (×0.55, realised L scale 0.56): v2 absolute lift ratio 0.65 → relative
retention **1.15** (a purely relative op would give 1.0; the taper/feather geometry
is unchanged so the small excess comes from the smoothstep sitting slightly higher on
the darker copy). v1 "retained" 0.88 only because lash pixels clear an absolute 15 L
threshold at any exposure — its skin lift was ~0 in both. The corpus has no
Fitzpatrick IV–VI subject; the probe is a stand-in, not a substitute (§7).

IED-from-mask fallback: polygon area/IED² med 0.02 (p10 0.01, p90 0.03) →
`_UE_AREA_PER_IED2 = 0.02` (the first draft's 0.06 was 40% off). Only the legacy
`UnderEyeRepairer.repair` path uses it; the dispatch passes the real IED.

## 5. Renders reviewed

`sheets/<stem>_grid.jpg` = base | v1@100 / v2@60 | v2@100 (eye band, ×1.7):

- **DSCF6693** (deep genuine circles): v1 lights the lash line only; v2@60 lifts the
  whole trough evenly; the first draft (0.12·IED extension) cut the shadow mid-way and
  left a lighter band above darker skin — fixed by the 0.28 extension + taper.
- **DSCF4576** (hair over the temple): the first draft's hard lash cut + sideways
  spread produced a grey wedge at the outer corner; fixed by the feathered exclusion,
  narrow sideways reach and the skin gate. Final render clean at 60 and 100.
- **DSCF4570 / 4559**: one real grey circle lifted and warmed naturally; the other
  eye (no shadow / foreshortened) untouched.
- **DSCF4463, 7204, 6963** (under-eye contour makeup): the brown line below the
  aegyo-sal band is read as shadow; subtle at 60, visibly softened at 100. No edges.
- Engine (`engine/<stem>_<recipe>.jpg`), recipe vs the same recipe with the three
  under-eye keys forced 0, |Δ| in the under-eye hull (polygon ⊕ 0.35·IED):

  | image | recipe (effective darken / puffiness) | mae | p99 | max | eye hull max | beyond hull max |
  |---|---|---|---|---|---|---|
  | DSCF4454 | clear_skin_v1 (0.30 / 0.20) | 0.18 | 3 | 7 | 0 | 3 |
  | DSCF4463 | clear_skin_v1 | 0.52 | 10 | 15 | 0 | 4 |
  | DSCF4503 | clear_skin_v1 | 0.74 | 12 | 17 | 0 | 4 |
  | DSCF4576 | clear_skin_v1 | 0.59 | 10 | 20 | 0 | 4 |
  | DSCF6693 | clear_skin_v1 | 1.33 | 18 | 26 | 0 | 5 |
  | DSCF6961 | clear_skin_v1 | 1.60 | 24 | 32 | 0 | 5 |
  | DSCF4463 | tired_eye_rescue_v1 (0.60 / 0.50) | 1.03 | 19 | 28 | 2 | 5 |
  | DSCF6693 | tired_eye_rescue_v1 | 2.58 | 33 | 37 | 0 | 6 |

  The ≤6 levels beyond the hull and the 2-level eye-hull touch on 4463 are downstream
  ops (sharpen / catchlight) re-reading the lifted band, not the op itself (0 px in
  isolation). Heat is a crescent under each eye, nothing on lashes.

## 6. Tests

`tests/test_undereye.py::TestUndereyeV2` (9 tests): `_dilate_down` direction and
sideways reach; support excludes the eye, never reaches full weight on lashes, and
has no step (max gradient < 0.3); lift lands on the shadow with texture preserved and
the eye bit-exact; outside-support bit-exact for uint8 and float32; strength
monotonic; relative-threshold tone invariance (×0.45 luminance → relative lift within
35%); no-shadow → no change; no-exclude fallback keeps a 0.12-IED polygon; legacy
`brighten_dark_circles` single cap. Existing 32 tests still pass; golden face
snapshots unchanged (`natural` sets no under-eye key; the fixture never exercised it).

## 7. Limitations and open items

- **Makeup vs shadow** (owner decision): 55 recipes set the op; `tired_eye_rescue_v1`
  and `convention_clear_v1` at 0.6 and `studio_porcelain_clear_v1` at 0.45 will
  visibly soften under-eye contour makeup on cosplay subjects. Options: leave (the op
  finally does what the recipe names promise), or cap those three at ~0.35. Not
  changed here.
- **Darker skin**: no Fitzpatrick IV–VI subject in the corpus. The relative band is
  designed for it (and the ×0.55 probe holds), but pigmented brown circles vs warm
  makeup is exactly the case a chroma guard would get wrong — re-verify on real
  samples before adding any colour-based gate.
- **Roll**: `_dilate_down` is image-axis "down"; heads rolled > ~20° get a skewed
  extension (feather absorbs small roll). A landmark-frame rotation is the fix if it
  shows up in QA.
- **Puffiness** still desaturates chroma inside the same weight — untouched by this
  study beyond mask reuse; no corpus evidence either way.
- The predecessor's §6 proposal ("drop the redundant factor, exclude the eye,
  relative threshold") is superseded by this design; its 15-L detector is kept only as
  the legacy `detect_dark_circles` API (no callers).
