# Research — Meitu Competitor QA (izunako cosplay corpus)

**Date:** 2026-08-29
**Status:** Observational, 5-pair pilot (2 kimono + 3 bunny-costume, §5). Not a calibration study
— no thresholds proposed here, findings are directional and scoped to what the metrics + visual
diff actually show.
**Inputs:** `/tmp/izunako_competitor_qa.py` (unregistered, §0) and
`/tmp/izunako_competitor_qa_registered.py` (landmark-registered, §2a + §5, latter added 3 more
pairs in-place), outputs at `/tmp/izunako-competitor-qa-20260829/`,
`/tmp/izunako-competitor-qa-registered-20260829/` (2-pair), and
`/tmp/izunako-competitor-qa-registered-20260829-batch2/` (5-pair rerun) — none of these paths are
inside this repo, not committed, referenced here for traceability only. Registered variant run via
`.venv/bin/python` (mediapipe 0.10.5, pinned runtime) reusing `retouch.detection.FaceDetector` —
per `RESEARCH_EYE_OCCLUSION_2026_08_26.md`'s trap warning, bare `python3` resolves to a drifted
system interpreter where `FaceDetector.available == False`.
**Source corpus:** `~/Library/CloudStorage/GoogleDrive-alexanderlooi88@gmail.com/My Drive/Photography/izunako`
— Fuji raw-derived JPEGs (`DSCF2362.jpg`, `DSCF2365.jpg`) vs. the same shots run through the
Meitu app (`IMG_20260829_213523.jpg`, `IMG_20260829_213448.jpg`).
**Branch:** `feat/color-science-k9-fix-and-frontier`

---

## 0. What this is

Two DSCF portraits (cosplay, kimono-style costume) were each run through Meitu, then compared
source-vs-edited on four hand-picked crop regions per pair (face + three body/costume regions),
using `metric_row()`: Lab-space MAE per channel, mean luma delta, mean saturation delta, a
high-frequency std-ratio (`hf_ratio` = edited HF std / source HF std, via Gaussian-residual), and
PSNR. Crops are naive `INTER_AREA`-resized rectangles, not landmark-registered — some of the MAE
signal below is geometry drift (Meitu reshapes), not just color/texture change, and the two are
not separated by this script. Full metrics: see the script's stdout, reproduced below for the
record.

```
PAIR 2362 DSCF2362.jpg -> IMG_20260829_213523.jpg
 full          mae_Lab=(10.19, 3.98, 2.37) luma=-6.14  sat=-29.53 hf_ratio=0.932 psnr=23.58
 face          mae_Lab=(14.94, 3.99, 3.06) luma=-14.8  sat=-16.69 hf_ratio=0.957 psnr=20.99
 left_shoulder mae_Lab=(19.99, 7.29, 3.85) luma=-18.15 sat=-15.34 hf_ratio=0.948 psnr=19.29
 right_arm     mae_Lab=(13.25, 4.79, 3.11) luma=-11.7  sat=-26.86 hf_ratio=0.862 psnr=22.01
 costume       mae_Lab=(17.68, 8.26, 3.4)  luma=-13.47 sat=-18.01 hf_ratio=0.861 psnr=19.98

PAIR 2365 DSCF2365.jpg -> IMG_20260829_213448.jpg
 full          mae_Lab=(10.09, 5.26, 4.05) luma=1.38  sat=-12.81 hf_ratio=1.189 psnr=20.26
 face          mae_Lab=(8.88, 2.77, 3.15)  luma=0.93  sat=-14.43 hf_ratio=1.125 psnr=24.66
 shoulder      mae_Lab=(24.23, 9.57, 7.02) luma=-9.24 sat=-17.5  hf_ratio=1.062 psnr=15.35
 forearm_hand  mae_Lab=(10.67, 4.73, 3.83) luma=-0.69 sat=-15.69 hf_ratio=1.136 psnr=22.32
 costume_hair  mae_Lab=(25.5, 13.54, 8.91) luma=-2.63 sat=-24.05 hf_ratio=1.071 psnr=14.88
```

Visual confirmation: `2362_detail_sheet.jpg` and `2365_detail_sheet.jpg` inspected directly
(4-panel face/shoulder/arm/costume, source | Meitu | absdiff×4 rows) — findings below are
grounded in the images, not metrics alone, per CLAUDE.md's verification rule against
metric-only, unrendered claims.

---

## 1. Face region — consistent across both pairs

- **Global brighten + desaturate.** 2362: luma −14.8, sat −16.7. 2365 face luma is flat (+0.9)
  but sat still drops −14.4 — the desaturation is the more load-bearing move than the brighten;
  2362's face was likely underexposed relative to 2365 and got compensated harder.
- **Skin micro-texture removed.** Visible in the 2362 face diff panel: pore-level and fine hair
  detail (baby hairs at the hairline, skin grain) collapses in the diff to a flat warm cast, while
  the two facial moles are preserved sharply — Meitu's skin-smoothing appears mole/landmark-aware
  (keeps recognizable marks, removes generic texture), similar in spirit to this engine's
  frequency-separation approach but visibly more aggressive on texture removal.
- **Eyes brightened/color-shifted.** Iris in the 2362 diff shows a cool-toned residual (sclera +
  iris both lightened, iris hue pulled warmer in the edited frame) — consistent with a generic
  "eye pop" filter, not a per-eye visibility gate; no evidence Meitu does anything
  occlusion-aware (no visible difference in aggressiveness between the two irises in either pair,
  and neither subject has a notably closed eye in these two shots, so this is not a strong test of
  that specific question either way).

## 2. Body/costume regions — initial (unregistered) read, SUPERSEDED by §2a

The unregistered `hf_ratio` split between the two pairs looked like a strong signal: 2362 body
crops 0.86–0.95 (detail removed), 2365 body crops 1.06–1.19 (detail ADDED, esp.
`costume`/`costume_hair`), read as evidence of a costume/fabric clarity-boost pass distinct from
skin smoothing. **§2a's landmark-registered rerun does not support the "detail added" half of
this — see below.** Left here struck through rather than deleted, since the registration result
is only meaningful in contrast to what it corrected.

## 2a. Landmark-registered rerun — corrects the clarity-boost claim

`izunako_competitor_qa_registered.py` fits a similarity transform (rotation + uniform scale +
translation; `cv2.estimateAffinePartial2D`, LMEDS) from 7 stable MediaPipe landmarks (eye
corners, nose tip, mouth corners) on source → edited, warps the *source* onto the edited frame's
geometry (Lanczos4), then re-runs the same `metric_row()` per crop. This separates "the crop
window points at a different physical location because the face moved/scaled slightly between
frames" from "Meitu changed color/texture at a fixed physical location."

Fit quality was good on both pairs — 7/7 landmark inliers, reprojection error 5.5px (2362) / 7.4px
(2365) on 3300–3500px-wide images, i.e. sub-pixel-scale relative to the crop regions measured:

```
PAIR 2362 (reproj err 5.48px, 7/7 inliers)
 full          mae_Lab=(12.45, 4.39, 2.81) luma=-6.4   sat=-29.8  hf_ratio=0.752 psnr=21.63
 face          mae_Lab=(19.69, 4.26, 3.29) luma=-15.09 sat=-16.75 hf_ratio=0.807 psnr=19.45
 left_shoulder mae_Lab=(21.08, 7.53, 4.24) luma=-18.11 sat=-15.51 hf_ratio=0.691 psnr=18.71
 right_arm     mae_Lab=(18.65, 5.82, 4.05) luma=-12.38 sat=-27.24 hf_ratio=0.704 psnr=18.81
 costume       mae_Lab=(21.88, 9.22, 4.5)  luma=-14.12 sat=-17.69 hf_ratio=0.695 psnr=17.97

PAIR 2365 (reproj err 7.37px, 7/7 inliers)
 full          mae_Lab=(11.6, 5.41, 4.19)  luma=1.57  sat=-12.68 hf_ratio=0.966 psnr=19.96
 face          mae_Lab=(11.37, 2.89, 3.2)  luma=1.1   sat=-14.63 hf_ratio=0.922 psnr=23.43
 shoulder      mae_Lab=(24.76, 9.61, 7.06) luma=-8.94 sat=-17.75 hf_ratio=0.827 psnr=15.36
 forearm_hand  mae_Lab=(11.75, 4.8, 3.92)  luma=-0.74 sat=-15.84 hf_ratio=0.915 psnr=21.97
 costume_hair  mae_Lab=(25.11, 13.31, 8.71) luma=-2.23 sat=-24.22 hf_ratio=0.892 psnr=15.0
```

**`hf_ratio` is now < 1.0 in every region of both pairs.** The 2365 "detail added" reading in §2
does not survive registration — it was very likely spurious HF energy from unaligned
skin/costume edge boundaries in the naive crop-diff (a 5-10px geometry offset creates a strong
edge-aligned residual that inflates the Gaussian-residual HF metric without reflecting a real
sharpening operation). **Retracted: there is no evidence of a costume/fabric clarity-boost pass
distinct from smoothing.** Meitu appears to smooth/soften detail in every measured region in both
pairs, full stop — no region-specific "add detail" behavior found.

MAE went *up* almost everywhere after registration rather than down, which at first reads as the
transform failing — but the fit quality numbers (7/7 inliers, sub-pixel reprojection error) say
otherwise. The likely explanation: a 4-DOF similarity transform (rotation/scale/translation only)
cannot express *non-rigid* deformation, and the residual MAE increase is concentrated in
exactly the regions with the strongest visual reshape evidence (shoulder, costume) rather than
spread evenly — consistent with real liquify-style reshape there, not registration failure. This
is inferred from the pattern, not proven independently; see caveats below.

**Revised conclusion:** the liquify-reshape finding (visible shoulder-silhouette narrowing,
confirmed by direct inspection of both `..._detail_sheet.jpg` outputs, registered and
unregistered) stands. The three-operation model from the original write-up is now two: (a) face +
body skin smoothing/brighten/desaturate (uniform in direction, present everywhere, `hf_ratio` < 1
across the board after registration), and (b) a body/shoulder liquify reshape. No evidence
remains for a separate costume-clarity-boost operation — that was a registration artifact.

## 3. Caveats / what this pilot does NOT establish

- **N=2 pairs, single subject, single lighting setup.** Do not generalize magnitudes; the
  face/body smoothing finding is qualitative and visually corroborated, but exact luma/sat deltas
  will vary by exposure of the source frame (2362 was clearly darker pre-edit than 2365, which
  likely explains the luma-delta gap more than a policy difference).
- **The registration transform is rigid (similarity, 4 DOF)** — it cannot separate "liquify
  reshape happened here" from "residual misalignment" with certainty; §2a's argument that the
  post-registration MAE increase reflects real reshape (not fit failure) rests on fit-quality
  numbers (7/7 inliers, sub-pixel reprojection error) plus visual corroboration in the diff
  images, not on an independent geometric measurement of the reshape itself. A non-rigid
  (thin-plate-spline / optical-flow) registration would be needed to quantify reshape magnitude
  directly rather than infer it from a residual.
- **No ground truth on Meitu's actual pipeline** — the two-operation model in §2a is inferred from
  crop-level metrics + visual inspection, not from Meitu's source or documentation.
- **This says nothing about output quality/preference** — only about what operations were applied
  and where. No claim here that Meitu's result is better or worse than this engine's; that would
  need a preference study, not a diff study.
- **Correction on record:** the original (unregistered) version of this doc claimed a
  costume/fabric clarity-boost pass in pair 2365 based on `hf_ratio` > 1. §2a's registered rerun
  retracts that claim — see §2a for why. Kept as a worked example of why unregistered crop-diff
  HF ratios are not reliable evidence of an "add detail" operation on their own.

## 5. Batch 2 — 3 more pairs (bunny costume, different shoot/lighting)

Added `DSCF2306`/`DSCF2308`/`DSCF2310` vs. their Meitu exports (`MEITU_20260829_221824277`,
`MEITU_20260829_221615396`, `MEITU_20260829_221436837` respectively) to
`izunako_competitor_qa_registered.py`, same crop-region + registration method as §2a, output at
`/tmp/izunako-competitor-qa-registered-20260829-batch2/`.

**Trap hit and corrected, worth recording:** none of the three new source/Meitu filenames
correspond by naming convention (unlike the `DSCF*`↔`IMG_*` timestamp-adjacency in §0's pair),
and two of the three sources share identical pixel dimensions (4160×6240), so an initial filename-
order pairing was silently wrong for two of three pairs (2306↔436837 and 2310↔824277 were
swapped) — first-run numbers were nonsense (MAE up to 71.9, PSNR down to 7.9dB, a magnitude
inconsistent with anything seen in §0/§2a) but *looked* like a plausible "much heavier edit"
finding until the actual detail-sheet images were inspected and showed mismatched poses between
the "source" and "Meitu" columns of the same row. Re-verified pairing via full-frame thumbnail
comparison (a 2×2/1×2 visual grid, not metrics) before rerunning — the general lesson: **when a
diff study's filenames don't self-encode the pairing, verify the pairing visually before trusting
any metric it produces**, especially outlier-looking ones.

Corrected metrics (registered, `.venv`, pinned mediapipe, same fit-quality bar as §2a — all three
got 7/7 inliers, 5.2–7.5px reprojection error on 3300–5500px-wide images):

```
PAIR 2306 DSCF2306.jpg -> MEITU_20260829_221824277.jpg (reproj err 5.19px, 7/7 inliers)
 full     mae_Lab=(18.77, 1.56, 1.71) luma=4.86  sat=-2.31 hf_ratio=1.195 psnr=16.97
 face     mae_Lab=(20.9, 1.89, 2.05)  luma=5.77  sat=-6.32 hf_ratio=1.118 psnr=17.23
 bodice   mae_Lab=(18.7, 2.31, 1.99)  luma=-1.7  sat=-2.96 hf_ratio=1.227 psnr=17.43
 thigh_leg mae_Lab=(16.94, 1.99, 1.89) luma=3.01 sat=-2.61 hf_ratio=1.012 psnr=17.51

PAIR 2308 DSCF2308.jpg -> MEITU_20260829_221615396.jpg (reproj err 7.51px, 7/7 inliers)
 full     mae_Lab=(21.1, 1.92, 2.36)  luma=0.87  sat=-1.09 hf_ratio=1.187 psnr=15.96
 face     mae_Lab=(23.77, 1.91, 2.23) luma=5.26  sat=-3.93 hf_ratio=0.938 psnr=15.24
 bodice   mae_Lab=(27.11, 3.13, 3.11) luma=-2.83 sat=-1.93 hf_ratio=1.094 psnr=15.02
 thigh_leg mae_Lab=(20.17, 2.54, 2.63) luma=-4.6 sat=1.42  hf_ratio=1.204 psnr=16.33

PAIR 2310 DSCF2310.jpg -> MEITU_20260829_221436837.jpg (reproj err 5.43px, 7/7 inliers)
 full     mae_Lab=(15.83, 1.62, 2.06) luma=4.52  sat=1.61  hf_ratio=1.189 psnr=18.37
 face     mae_Lab=(15.31, 1.37, 1.72) luma=3.94  sat=4.8   hf_ratio=1.179 psnr=18.92
 bodice   mae_Lab=(16.94, 1.66, 1.96) luma=3.84  sat=3.64  hf_ratio=1.181 psnr=18.04
 thigh_leg mae_Lab=(20.45, 2.66, 3.12) luma=4.47 sat=-2.2  hf_ratio=1.136 psnr=16.94
```

Visual confirmation: all three `..._detail_sheet.jpg` outputs inspected directly; pose match
between source and Meitu columns confirmed correct in every panel (the fix in the trap above).

**Findings, and how they differ from §1/§2a:**

- **Much lighter edit overall.** MAE (16–27) is in the same numeric range as the kimono pairs, but
  visually the face/bodice/thigh panels are near-identical between source and Meitu — fine fabric
  texture (bow-tie plaid, lace trim edges) and skin micro-texture are both largely intact, unlike
  the kimono pairs' visible texture flattening.
- **`hf_ratio` > 1.0 in nearly every region (1.01–1.23)** — the opposite sign from the registered
  kimono pairs (§2a: <1.0 everywhere). Given §2a's own finding that unregistered `hf_ratio` > 1
  was a registration artifact, this deserves the same skepticism — but here it holds up *after*
  registration with good fit quality, so on this batch it's not the same artifact. Most likely
  explanation: minor global contrast/clarity increase (visible as slightly crisper edges
  throughout, not a targeted "add detail to costume only" pass) rather than texture removal —
  consistent with these being a lighter edit overall, not evidence of the retracted
  clarity-boost claim reappearing.
- **No visible reshape/liquify** in any of the three — shoulder/bodice/thigh silhouettes overlay
  cleanly in the diff panels (diff energy is edge/lighting noise, not the coherent silhouette-shift
  blob seen in 2362's shoulder in §2a).
- **Makeup/eye intensification is the dominant visible change**: eyeliner and eye makeup are
  visibly stronger in the Meitu column across all three faces, with light local skin evening
  around eyes/cheeks — smaller in magnitude and more localized than the kimono pairs' global
  brighten+desaturate.

**Revised overall conclusion (supersedes the two-operation model at the end of §2a):** Meitu's
edit intensity is **not constant per pipeline run** — it varies substantially pair-to-pair, in a
way that isn't explained by exposure/lighting differences alone (this batch and the kimono batch
were shot in different sessions but by the same subject/photographer relationship, and the
resulting edits differ far more in character than the source exposure differences would predict).
The earlier "Meitu = smoothing + liquify" model (§1/§2a) describes the kimono pairs; this batch
shows a materially lighter touch with no reshape at all. The simplest explanation consistent with
the corroborating screenshot evidence (message context: a Meitu UI screenshot showing "Smart" vs.
"Classic" mode toggle and a manual "Contour" tool tab under "Manual / Face / Plump / Facelift") is
that **these are manually-tuned, per-shot edits**, not a fixed automatic filter — so "what Meitu
does" is not a single answer and any future competitive-parity target should be scoped as a range
of edit intensities, not one reference behavior.

## 6. Possible follow-ups (not started, no owner assigned)

- If competitive parity on body/shoulder reshape becomes a goal: this is new pipeline scope
  (torso/shoulder liquify keyed off segmentation class) beyond the current face-only reshape stage
  — likely belongs in `docs/plans/RESEARCH_SMART_WORKSPACE_PRODUCTIZATION_2026_08_29.md`'s
  backlog rather than as a standalone item, given that doc already ranks P0–P3 productization
  work. [Owner call, not made here.] Cross-check that doc's priority list before scheduling
  anything from this pilot — per its own rule to cut P2/P3 before touching safety/lifecycle/
  compatibility gates, a new "match competitor body reshape" item would need to be graded against
  that rubric, not treated as automatically in-scope.
- A non-rigid registration (TPS or dense optical flow fit on skin/costume, not just the 7 face
  landmarks) would let reshape magnitude be measured directly instead of inferred from a residual
  MAE increase — the natural next step if body-reshape parity becomes a real workstream.
- §5's finding that edit intensity varies per-shot (likely manual tuning, not a fixed pipeline)
  means any future N-expansion should track edit intensity as its own variable (e.g. via the
  Meitu "Smart"/"Classic" mode toggle visible in-app, if recoverable) rather than pooling all
  pairs as one population.
- Expanding N beyond 5 pairs, and ideally across more than 2 shoots/lighting setups, before
  drawing any quantitative (not just qualitative) conclusion.
