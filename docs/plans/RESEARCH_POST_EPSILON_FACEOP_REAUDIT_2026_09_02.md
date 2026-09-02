# Research Results — Post-Epsilon Face-Op Re-Audit + Neck Gate (2026-09-02)

**Status:** COMPLETE (single-labeler visual pass on the flagged sheets; two fixes shipped,
one tuning finding left as a proposal).
**Branch:** `feat/color-science-k9-fix-and-frontier`
**Runtime:** `.venv/bin/python` (mediapipe 0.10.5, cv2 4.11, `RETOUCH_GPU=0`,
`RETOUCH_MEDIAPIPE_BACKEND=legacy`). Bare `python3` never used.
**Scripts:** `scripts/qa/post_epsilon_faceop_sweep.py` (pass 1),
`scripts/qa/post_epsilon_pass2_undereye_shine.py` (pass 2).
**Artifacts:** `test_output/post_epsilon_faceop_sweep/` — `sweep.json`, `pass2.json`,
`run.log`, `pass2.log`, `sheets/<stem>_natural.jpg` (source | shipping | emulated pre-fix |
heat), `sheets/<stem>_ops.jpg` (per-op crop | heat rows, incl. an isolated neck-op row),
`neck_fix_after.jpg`, `neck_fix_after_v2.jpg`.

---

## 0. Summary

`f69ab1e` fixed a mask normaliser that divided float region masks by 255 whenever
feathering overshot 1.0 by one ulp. On the affected portraits the face-core composite
alpha over skin collapsed to 1/255 and every skin-composite op was silently discarded.
This study asks two questions across the 83-face DSCF corpus: (1) now that those faces
are visible again, are the default strengths (tuned while ~24% of faces were invisible)
too strong, and which past "op X does nothing on image Y" conclusions were the bug?
(2) The neck yaw gate the 2026-09-02 yaw study left "not recalibrated" — what does it do?

Answers:

- **19/83 faces** were pre-fix victims (skin max 1.0000001–1.0000002 with clipping
  disabled). Post-fix their `natural` recipe skin delta **matches the 64 controls**
  (median 2.61 vs 2.50 levels, p99 12 vs 13). Default strengths are not over-strong;
  **no re-tune is warranted.** Pre-fix the same faces got 1.17 (global stages only).
- **Every per-op probe now fires on every affected face** (smooth/equalize/relight/
  sculpt/skin_sss/whiten: 0/19 near-zero). The remaining near-zero cases in the control
  group are all explained (§3.3): yaw-gated at ratio > 4, or multi-face images whose
  largest detection is a hand / blurred background person.
- **The 2026-08-31 audit's DSCF8007 attribution was wrong.** It blamed the plastic-skin
  QA backoff for 8007's zeroed skin column. 8007 is an epsilon victim (skin max
  1.0000002); post-fix its smooth/skin_sss/sculpt probes read 1.45/0.52/1.02 (were
  0.00/0.00/0.03). The backoff exists but does not nullify.
- **New P1 — the shared yaw ramp was inverted** (`utils.yaw_gate_factor` returned the
  raw smoothstep 0→1). Relight/sculpt got ~0 strength just past ratio 2.5 and ~full just
  under 4.0 (§4). Fixed in `ce56ba2` with a monotonicity test.
- **Neck gate — not a calibration problem, broken geometry.** The "depth gate" measured
  ~0.45× the vertical distance below the eye line, so 15/15 frontal faces lost 100% of the
  neck mask (neck harmonisation was a guaranteed no-op) while turned faces skipped the
  gate. When the op did run it painted hard 20–30-level patches on a collar and a hand.
  Gate removed, chroma gate tightened and applied to both mask paths, LAB roundtrip
  contained — `fb2474e` (§5).
- **Under-eye dark-circle removal is near-inert** at strength 60 on every corpus face
  (0/83 faces with p99 > 0 in the under-eye hull). Pass 2 quantifies why (§6). Left as a
  proposal — it is a tuning/design change, not a wiring bug.

## 1. Method

- Corpus: `test_output/detection_recall_study/corpus.txt` (83 images, 83 detected
  faces, 17 multi-face). Pre-shrunk to max-dim 2048. Largest face per image is the
  measured face (same convention as the yaw study; see §3.3 for where that fails).
- **Affected flag:** the engine is run once with `parsing.feather_mask`/`utils.feather_mask`
  replaced by the pre-`f69ab1e` unclipped body and `perf_optimizations._norm_mask` by its
  pre-fix `> 1.0` body; `affected = regions.skin.max() > 1.0` on the largest face. The
  same run yields the **emulated pre-fix `natural` render** (the dominant collapse
  mechanism — `_FaceResult.skin_mask` through `_norm_mask` — is reproduced; the two
  inline `p.max() > 1` sites in `_process_face_core` are not, so the emulation is a lower
  bound on what users lost).
- **Shipping path:** fresh detection, `natural` as shipped, then a baseline with all
  probed ops zeroed (`ALL_OFF`) plus one render per op with cached `face_contexts`:
  smooth 70, equalize 60, blemish 60, dark_circles 60, relight 60, sculpt 50, skin_sss 60,
  whiten 50. Metrics per render: mean/p99/frac>6 of max-channel |diff| inside the dilated
  FACE_OVAL hull (`mae_in`), outside face+neck (`mae_out`), inside a landmark neck band
  (`mae_neck`), inside the under-eye hulls (`mae_ue`), plus whole-frame p99/max.
- **Neck isolation:** equalize 60 rendered with and without `SkinProcessor.harmonize_neck`
  (stubbed to identity); the diff is the neck op's own contribution. The shipped 1.3 yaw
  switch and the plane-distance gate are replicated in full-image coordinates to record
  the fraction of the landmark neck rectangle that survives the gate.
- Visual pass on every flagged sheet before any claim (delta-blindness rule).

## 2. Natural recipe: what changed for users

| group | n | ship vs pre-fix mae (skin hull) | ship vs input mae | pre-fix vs input mae | ship vs input p99 |
|---|---|---|---|---|---|
| affected | 19 | med 1.88 (1.51–2.13 p10–p90) | **2.61** (2.21–2.74) | 1.17 (1.01–1.29) | 12 |
| unaffected | 64 | 0.00004 max | **2.50** (0.91–3.06) | = ship | 13 |

Affected faces: DSCF4553, 4556, 4557, 4558, 4562, 4563, 4567, 4568, 4574, 4575, 4582–4588,
6693, 8007. The unaffected group's floor (0.91) is the hand/background detections of §3.3,
not weaker skin ops. Sheets viewed: DSCF4568 (smoothing + tone evening confined to skin,
no seams — `sheets/DSCF4568_natural.jpg`), 4557, 4558, 8007 — no over-smoothing, no leak
into hair or costume at the default strengths.

The yaw study's commit message said 20/83; the exact-1.0 face (DSCF4463, max 1.0000000)
sits on the boundary and was never divided, so 19 is the operative count.

## 3. Per-op probes (shipping path, post-fix)

### 3.1 Affected vs unaffected, `mae_in` over the skin hull

| op (probe) | affected med [min, max] | ~zero | unaffected med [min, max] | ~zero |
|---|---|---|---|---|
| smooth 70 | 1.34 [1.05, 1.58] | 0/19 | 1.34 [0.00, 1.85] | 6/64 |
| equalize 60 | 0.89 [0.68, 1.78] | 0/19 | 1.26 [0.00, 3.55] | 2/64 |
| relight 60 | 0.97 [0.09, 1.49] | 0/19 | 0.91 [0.00, 1.66] | 11/64 |
| sculpt 50 | 0.62 [0.08, 1.13] | 0/19 | 0.66 [0.00, 1.64] | 8/64 |
| skin_sss 60 | 0.47 [0.33, 0.52] | 0/19 | 0.44 [0.00, 0.77] | 7/64 |
| whiten 50 | 1.60 [1.27, 2.48] | 0/19 | 1.93 [0.01, 2.74] | 1/64 |
| blemish 60 | 0.01 [0.00, 0.02] | 19/19 | 0.00 [0.00, 0.02] | 64/64 |
| dark_circles 60 | 0.00 [0.00, 0.01] | 19/19 | 0.00 [0.00, 0.01] | 64/64 |

"~zero" = mae_in < 0.05. Blemish is sparse by design (spot heals: whole-frame max 47–79
on 4463/4503/4563/8007 with p99 0; sheets show single-spot heals, no artefacts). Dark
circles is genuinely near-inert — §6.

Leak (`mae_out`, outside face+neck): ≤ 0.24 for every op on every face (worst: equalize
0.24 / whiten 0.22 on DSCF4577, a two-face image). Skin-class masks compose correctly —
the 08-31 "≤0.0013" figure used a stricter region and a 9-image set, but the conclusion
stands.

### 3.2 The 08-31 audit's nine images, then vs now

| img | affected | smooth old/new | equalize old/new | sculpt old/new | skin_sss old/new | whiten old/new |
|---|---|---|---|---|---|---|
| 4463 | 0 | 0.78/0.79 | 1.50/1.46 | 0.00/**0.69** | 0.32/0.29 | 1.45/1.44 |
| 4503 | 0 | 0.91/0.90 | 1.96/1.91 | 0.87/0.79 | 0.39/0.38 | 1.41/1.40 |
| 4550 | 0 | 1.04/1.04 | 1.40/1.34 | 0.00/**0.79** | 0.44/0.42 | 1.99/1.98 |
| 4552 | 0 | 1.33/1.32 | 1.11/1.08 | 0.00/0.11 | 0.45/0.42 | 2.14/2.13 |
| 4560 | 0 | 1.14/1.13 | 1.20/1.14 | 0.00/0.19 | 0.35/0.32 | 1.73/1.69 |
| 4576 | 0 | 0.94/1.70 | 1.05/1.90 | 0.27/1.27 | 0.29/0.48 | 1.30/2.37 |
| 6961 | 0 | 1.07/1.07 | 1.23/1.17 | 0.00/**0.79** | 0.35/0.34 | 1.07/1.04 |
| 7204 | 0 | 0.72/0.72 | 1.28/1.26 | 0.00/**0.58** | 0.25/0.22 | 0.93/0.92 |
| 8007 | **1** | 0.00/**1.45** | 0.23/**1.78** | 0.03/**1.02** | 0.00/**0.52** | 0.32/**1.97** |

Unaffected single-face images reproduce to ±0.06 across the two studies (different hull
dilation, same engine path) — a useful determinism check. Sculpt's 0.00 → 0.6–0.8 on
4463/4550/6961/7204 is the `9c26493` band change (those faces sit at ratio 2.0–3.4; the
inverted ramp of §4 still damped 4552/4560/7204). 4576 is two-face and its 08-31 rows
mixed pre-/post-`fafed66` code, so its shift is not interpretable. 8007 is the epsilon
bug, full stop.

### 3.3 Remaining near-zero cases (all in the control group) — explained

| img | faces | ratio | skin max | why |
|---|---|---|---|---|
| 4454, 4596, 4597, 4599 | 1–2 | 4.6–7.6 | — | yaw ≥ 4.0 → relight/sculpt correctly 0 (skin ops fire) |
| 4554, 4555 | 2–3 | 20.6, 293.7 | — | degenerate landmarks on a profile/false detection; yaw-gated |
| 4561, 7121 | 1 | 2.506, 2.510 | — | **inverted ramp** (§4): relight 0.00 one hundredth past START |
| 4589 | 2 | 1.28 | 1.0 | largest face is a **blurred background person**; ops fire at full strength on it (equalize p99 40, relight max 109) — `sheets/DSCF4589_ops.jpg` |
| 4618 | 2 | 1.26 | 1.0 | largest "face" is a **hand** — `sheets/DSCF4618_ops.jpg` |
| 4590, 4592, 4600 | 2 | 1.4–2.4 | 0.32–0.75 | largest detection not the subject; BiSeNet skin peak < 1 → smoothing alpha-limited |

None is an op defect. 4589/4618 join the yaw study's 4596/4597/4599 on the corpus-hygiene
list (largest face ≠ subject). Retouching an out-of-focus background face at full
strength (4589) is a product question already parked under "background faces out of
scope".

## 4. Inverted yaw ramp (P1, fixed `ce56ba2`)

While chasing 4561/7121 the relight code was instrumented: `yaw_gate_factor(2.506)`
returned **0.0001**. The helper returned `t*t*(3-2t)` — the smoothstep itself, 0 → 1 —
with hard-coded 1.0 below START and 0.0 above END. Corpus, relight 60 `mae_in`:

| ratio | 2.478 | 2.506 | 2.715 | 2.843 | 2.997 | 3.374 | 3.644 | 3.991 | ≥ 4.0 |
|---|---|---|---|---|---|---|---|---|---|
| shipped factor | 1.00 | 0.00 | 0.06 | 0.13 | 0.26 | 0.62 | 0.86 | 1.00 | 0 |
| relight mae_in | 0.77 | 0.00 | 0.09 | 0.15 | 0.27 | 0.58 | 0.91 | 1.51 | 0.00 |

Relight and sculpt track the inverted curve exactly. Provenance: `geometry.py`'s private
`_yaw_dampen_factor` (band 1.3→1.6) had the same body before `9c26493`, so slimming has
been inverted for as long as it has had a smoothstep; relight/sculpt used a correct linear
`1 - clip((r-1.5)/0.2)` until `9c26493` moved them onto the shared helper. The yaw study's
`smoothstep_damp` copy in `scripts/qa/yaw_gate_sweep.py` inherited the inversion, so the
`damp_geometry`/`damp_sculpt` mid-band columns in `test_output/yaw_gate_study/sweep.json`
are wrong (endpoints fine); its render validation used a gate *bypass* for slimming and
the sculpt sheet's quoted "factor ≈0.42" at ratio 3.37 was actually 0.62.

Fix: `1.0 - smoothstep`. `tests/test_utils.py::TestYawGateFactor` asserts monotone
non-increasing and near-start ≈1 / near-end ≈0. The existing relight/sculpt tests probe
only the band midpoint, where smoothstep(0.5) = 0.5 either way — they could never see
this. 265 targeted + 194 engine/perf tests pass; golden snapshots unchanged (`natural`
has slimming/relight/sculpt at 0).

## 5. Neck harmonisation (fixed `fb2474e`)

### 5.1 What the "depth gate" did

`skin.harmonize_neck`, frontal faces only (ratio ≤ 1.3): plane through the outer eye
corners and the nose bridge with z scaled by face width; keep pixels within
`0.15·face_w` of that plane, distance taken in XY only. Probe on two frontal faces:

| img | ratio | face_w | eye-line Δy to bridge | bridge Δz (scaled) | plane normal | thr | dist in neck rect min/med | survive |
|---|---|---|---|---|---|---|---|---|
| 4568 | 1.01 | 256 px | 23–60 px | −7.8 px | (−0.07, **0.47**, 0.88) | 38 px | 91 / 214 px | **0.00** |
| 4503 | 1.46 | 359 px | 20–80 px | −19.3 px | (−0.01, **0.43**, 0.90) | 54 px | 118 / 250 px | **0.00** |

The bridge sits far below the eye line in y but barely behind it in the scaled z, so the
normal has a ~0.45 y-component and the "distance" is ~0.45× the vertical distance below
the eyes. Neck pixels are 200+ px down; threshold is 38–54 px. Corpus: **15/15 faces with
ratio ≤ 1.3 kept 0.0% of the neck rectangle; isolated neck-op delta 0.000 on all 15.**
The 68 faces above 1.3 never entered the gate. The test that covered the gate
(`test_depth_gating_applied_when_low_yaw`) used a uniform grey image whose face and neck
medians were equal, so `l_diff = 0` and it passed regardless.

### 5.2 What the op did when it ran (68 turned faces)

Isolated neck-op delta: `mae_neck` med 0.08, p90 0.33; `p99_neck` med 1, p90 14, max 28;
face-hull drift med 0.38 (the whole-canvas LAB roundtrip, same class as the lips/teeth
`46be031` fix). Top cases viewed: DSCF4560 — a hard-edged **bright patch on the white
collar** (BiSeNet neck label includes it; a/b distance to face skin 14.4 passed the
`max(12, 2·(σa+σb)) ≈ 18` band); DSCF4463 — two **patches on the hand** under the chin
(distance 12.2, also admitted); DSCF7142 — fallback (person-mask rectangle) path, p99 22.

### 5.3 Fix and verification

Gate removed (the neck source mask already bounds the op). The face-chroma match that
only the fallback path applied now gates the BiSeNet path too, band tightened to
`max(10, 1.25·(σa+σb))` — real neck skin on the anchors sat at 2.2 (4568) / 4.5 (4575),
collar 14.4, hand 12.2. Roundtrip contained with `restore_outside_support`.
Equalize 60 with vs without the neck op, after the fix (`neck_fix_after_v2.jpg`):

| img | before: p99_neck | after: mae_neck / p99_neck / max | visual |
|---|---|---|---|
| 4568 (frontal, was no-op) | 0 | 0.065 / 3 / 9 | soft crescent lift under the chin, confined to skin |
| 4560 (collar) | 28 | 0.19 / 6 / 30 | patch reduced to a thin line along the collar edge |
| 4463 (hand) | 0 (hand is in face hull) | 0.00 / 0 / 34 | hand patches gone bar an edge sliver at the sleeve |
| 4575 (shadowed neck) | — | 0.35 / 21 / 26 | genuine neck lifted ~20 levels at strength 60; stops at the fabric edge |

The chroma band is **provisional**: four pale-skin anchors, no Fitzpatrick IV–VI sample
(`σa+σb` scales with skin chroma spread, the floor of 10 does not). New tests:
frontal/turned faces both harmonise, containment outside support is byte-exact, a
saturated-blue collar inside the neck mask is untouched. Golden face snapshots unchanged
(fixture supplies no person mask, so the op returns early there — the golden harness does
not cover this op).

## 6. Under-eye dark-circle removal — pass 2 (finding only, not applied)

Pass 1 read `dark_circles=60` as ~zero on all 83 faces (skin-hull mae ≤ 0.01, under-eye
hull p99 0). Pass 2 replicates `UndereyeAnalyzer.detect_dark_circles` on the landmark
under-eye hulls and probes `dark_circles=100`, `shine_removal=60`, `wrinkle_soften=60`
(166 eyes, 83 faces, cached contexts):

| measure | result |
|---|---|
| surround-median L − under-eye median L | med **40**, p10 6.5, p90 53 (L in 0–255) |
| fraction of under-eye px below `median − 15` | med 0.74 |
| detector fires (component ≥ 100 px) | **152/166 eyes (92%)** |
| `dark_circles=100` mae in under-eye hull | med 0.03, max 0.13; p99 med 1, max 5 levels |
| `dark_circles=100` when detector fires vs not | 0.015–0.085 vs ≤ 0.0001 |
| `shine_removal=60` | fires on 7/83 (4589, 4564, 4454 ≥ 0.08; rest ≤ 0.015) — adaptive detection working, consistent with the 08-31 verdict |
| `wrinkle_soften=60` | fires on 71/83, mae med 0.19, p99 7–16 — healthy |

So the detector is not the limiter — it fires almost everywhere, because the under-eye
hull's darkest connected component is the **lower lash line / eye corner**, which sits
40+ L below the cheek-and-sclera surround on nearly every face. Renders at strength 100
(`dark_circles_100.jpg`, DSCF4576/6693/4463, heat ×12) confirm it: the lift lands on the
lash line and inner/outer corners, and on DSCF6693 — the one face with genuine deep
under-eye shadow — the shadowed skin itself is essentially untouched.

Two compounding causes in `undereye.py`:

1. `brighten_dark_circles`: `l_lift = darkness · mask · strength · (max_lift / 100)` with
   `darkness` already clipped to `max_lift = 30`. The cap is applied twice — as a clip and
   as a ×0.3 multiplier — so the maximum possible lift is **9 L at strength 100**, ~5 at
   the recipe-typical 60, before `blend_masked` softens it further. Measured max 10.
2. `detect_dark_circles` thresholds on an **absolute** 15-L drop below the surround median
   and keeps the darkest components. That selects lashes/liner over diffuse shadow, and
   as an absolute L offset it will under-fire on darker skin (CLAUDE.md tone-invariance
   rule: use margin relative to the face's own baseline).

Proposal (owner decision — this changes the look of 27+ recipes that set the key):
drop the redundant `max_lift/100` factor (or make `max_lift` the sole cap), exclude the
eye/lash region from the detection mask (subtract the dilated eye hull), and express the
threshold as a fraction of the surround median. Any change must be re-verified on 6693
(genuine shadow), 4463 (liner, no shadow) and a darker-skin sample the corpus lacks.

## 7. Open items

- **Corpus hygiene:** exclude or relabel 4589, 4618 (largest detection is background
  person / hand) alongside the yaw study's 4596/4597/4599; 4554/4555 have degenerate
  landmarks on the largest face. Every "largest face" study on this corpus inherits these.
- **Neck chroma band** needs a darker-skin sample before it is more than provisional;
  the residual collar-edge line on 4560 suggests feathering the neck support wider than
  `0.03·face_w` once the band is settled.
- **Dark-circle op** — proposal in §6, not applied.
- Re-verify the 2026-08-31 finding-6 text (DSCF8007 / plastic-skin backoff) is not cited
  elsewhere as evidence that the backoff nullifies skin ops.
- `yaw_gate_study/sweep.json` mid-band `damp_*` columns are inverted (§4); regenerate if
  reused.
