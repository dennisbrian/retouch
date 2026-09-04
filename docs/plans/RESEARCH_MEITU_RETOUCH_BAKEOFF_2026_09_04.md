# Research — Meitu vs Retouch same-source bake-off

**Date:** 2026-09-04
**Status:** Five-pair pilot rerun completed; visual and metric evidence produced. This is
not a powered preference study, not a full-resolution delivery test, and not a
MeituYunxiu professional-product comparison.
**Build under test:** Retouch `fb6ba5ee185a` on
`feat/color-science-k9-fix-and-frontier`; Meitu export metadata `Software=Meitu 121708`.
**Predecessor:** `RESEARCH_MEITU_COMPETITOR_QA_2026_08_29.md`.

## 1. Outcome

The earlier competitor test was found and recovered. Its five source/Meitu pairs still
exist in the private `izunako` corpus, but the two analysis programs and rendered sheets
under `/tmp` no longer exist. The new reusable runner
`scripts/qa/competitor_pair_bakeoff.py` replaces that ephemeral analysis path and was
used to compare each source against:

1. the preserved Meitu export;
2. current Retouch `natural`;
3. current Retouch `convention_clear_v1`.

The main result is not “Meitu is stronger” or “Retouch wins.” The products make
materially different choices:

- Meitu changed a median **68.8%** of registered face-crop pixels by more than eight
  8-bit levels in at least one BGR channel;
- Retouch `natural` changed **22.0%** and `convention_clear_v1` changed **36.7%** by the
  same descriptive measure;
- Meitu reduced face-crop HSV saturation in all five cases, median **−9.0** levels;
- `natural` was nearly neutral to mildly positive, median **+0.7**;
- `convention_clear_v1` increased face saturation in all five, median **+4.0**.

So the current gap is primarily **aesthetic direction and edit intensity**, not proof of
a missing smoothing algorithm. Meitu's tested direction is more porcelain/desaturated
and sometimes more structurally edited. Retouch is more source-authoritative and more
color-preserving, while the convention recipe deliberately strengthens eyes/skin color.
Only a blinded preference review can say which result the owner actually prefers.

## 2. What was run

The two Retouch recipes were genuinely face-aware:

- `.venv` Python;
- MediaPipe `0.10.5`, legacy CPU backend;
- BiSeNet on `CPUExecutionProvider`;
- `global_only=false`;
- input resized to a 2048 px long edge for this comparison render;
- pair analysis standardized to a 1600 px long edge;
- largest detected face selected as the subject for registration;
- seven stable landmarks fitted with a similarity transform;
- registered full-frame/face residuals plus symmetrically resampled appearance summaries;
- full, face-detail, and randomized blinded sheets written for every pair.

The 2048 px boundary makes this a practical visual bake-off, not native-resolution
certification. The recipe-sweep JPEG path also does not exercise the production
ICC/Exif-preserving delivery path.

Observed Retouch render time at 2048 px was 6.9–20.1 s for `natural` and 12.5–34.6 s
for `convention_clear_v1`. Meitu operator time and cloud/local state were not recorded,
so no throughput comparison is valid.

## 3. Measured comparison

`Face change` is the fraction of registered face-crop pixels whose largest channel
change exceeds eight levels. `Landmark p95` is the 95th-percentile landmark residual
after removing global rotation/scale/translation, normalized by source inter-eye
distance (IED). The latter can indicate non-rigid change, but it also contains landmark
detector drift and must not be called an identity or reshape score.

| Case | Shoot | Meitu face sat Δ | Meitu face change | Natural face change | Convention face change | Meitu landmark p95 | Convention landmark p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| DSCF2362 | kimono | −25.4 | 79.3% | 24.3% | 36.7% | 19.6% IED | 1.8% IED |
| DSCF2365 | kimono | −16.2 | 74.7% | 37.3% | 42.8% | 4.2% IED | 3.4% IED |
| DSCF2306 | bunny | −9.0 | 67.2% | 16.2% | 24.1% | 9.1% IED | 2.2% IED |
| DSCF2308 | bunny | −6.6 | 68.8% | 22.0% | 39.7% | 8.5% IED | 2.8% IED |
| DSCF2310 | bunny | −0.3 | 58.8% | 21.0% | 18.5% | 7.4% IED | 5.5% IED |
| **Median** | — | **−9.0** | **68.8%** | **22.0%** | **36.7%** | **8.5% IED** | **2.8% IED** |

Meitu's face-luma change ranged from **−11.7 to +6.7** Lab8 L levels. This confirms the
old study's conclusion that there is no single fixed “Meitu brighten” behavior. The
two shoots likely used different tools, strengths, modes, or manual adjustments.

High-frequency ratios were retained in the JSON evidence but are not used for the
headline conclusion. Even after symmetric resampling, an expanded face crop contains
hair, eye makeup, lashes, and boundaries; their sharpening can conceal skin-texture
loss. A skin-only, same-export-resolution study is required before ranking texture
preservation.

## 4. Visual read

### Kimono pair

- `DSCF2362`: Meitu applies the largest global color and face change in the set. The
  result is darker and much less saturated, with a visibly stronger porcelain treatment.
  It also has the largest non-rigid landmark residual. The earlier visually confirmed
  shoulder narrowing remains credible, but this face-landmark runner does not quantify
  body reshape.
- `DSCF2365`: Meitu again suppresses saturation strongly but without the same large
  face-geometry residual. The edit is still much broader than either Retouch recipe.
- Both source frames contain a large printed face in the background. Retouch detects two
  faces on the sources and usually on its outputs. That is an adversarial poster false
  positive, not successful multi-subject handling; it should be a hard-case row in the
  next benchmark.

### Bunny trio

- Meitu is much lighter than on the kimono pair, but still changes more of the face than
  either Retouch recipe and lowers face saturation in all three.
- The dominant visible changes are skin evening, eye/makeup emphasis, and subtle facial
  structure differences. No convincing body reshape is visible in this trio.
- `natural` is consistently the least invasive candidate. `convention_clear_v1` adds a
  clearer eye/under-eye and color treatment without approaching the kimono Meitu look.

Across all five 1600 px review sheets, no critical eye, mouth, or face-boundary artifact
was found in the two Retouch candidates. This is a visual pilot boundary, not native-zoom
human certification.

All ten Retouch outputs raised the same automatic warnings: banding, plastic skin, seam,
color drift, and asymmetry; one convention output also raised harmony. Because the
warnings do not discriminate between visually mild and stronger candidates on this set,
they are calibration evidence, not a valid winner signal. The competitor images were not
run through an equivalent mask-aware warning path, so comparing warning counts would
also be unfair.

## 5. Export and provenance confounds

All five competitor files identify `Meitu 121708`, which is valuable version evidence.
However, the export settings were not held constant:

- `DSCF2362` and `DSCF2365` sources are 4160×6240, while their Meitu outputs are
  2731×4096 (about 43% of the source pixel count);
- the three bunny outputs retain their source dimensions;
- the Meitu outputs record new edit timestamps and use a Google sRGB v4.3 profile,
  while the sources use sRGB IEC61966-2.1 v2.1.

This makes cross-shoot texture and file-size comparisons invalid. A locked rerun must use
one export resolution, JPEG quality, profile, app mode, and recorded slider/preset state.

The filename and metadata identify the tested product as the **consumer Meitu app**.
They do not establish a test of MeituYunxiu. This distinction matters because the current
[MeituYunxiu product page](https://yunxiu.meitu.com/home?language=zh-TW) separately claims
RAW conversion, batch processing, skin detail, face/body shaping, background cleanup,
fabric cleanup, presets, and high-throughput export. Those remain vendor claims until a
same-source professional-path run is captured.

Meitu's own consumer-product history documents face editing, body reshape, body-keypoint
handling, and makeup/skin tools. That supports the *capability* interpretation of the
observed edits, but not a claim about which exact tools were used on these five files:
[Meitu portrait/body feature history](https://www.meitu.com/en/media/305) and
[Meitu quick slimming/background repair](https://www.meitu.com/en/media/353).

## 6. New research since the earlier pilot

### 6.1 Personal preference is the important target

[PALATE](https://arxiv.org/abs/2608.18622), submitted by researchers including MT Lab,
Meitu, treats portrait retouching as a candidate-ranking problem: keep the editor fixed,
generate several valid candidates, then learn a lightweight user-specific preference
adapter from rankings. On its held-out PPR10K protocol it reports 72.83% pairwise
preference-prediction accuracy versus 58.06% for its strongest listed baseline.

This is a paper result, not evidence that current Meitu ships PALATE. Its important lesson
for this project is architectural: Retouch already has many safe recipe candidates, so a
small owner-ranked candidate set is a more defensible next step than changing the default
recipe to imitate one inconsistent Meitu output.

### 6.2 Review must remain multidimensional

[BeautyGRPO (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_BeautyGRPO_Aesthetic_Alignment_for_Face_Retouching_via_Dynamic_Path_Guidance_CVPR_2026_paper.pdf)
uses separate review dimensions for skin smoothing, blemish removal, texture quality,
clarity, and identity preservation. [F-Bench (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_F-Bench_Rethinking_Human_Preference_Evaluation_Metrics_for_Benchmarking_Face_Generation_ICCV_2025_paper.html)
likewise argues for fine-grained human-aligned face evaluation rather than trusting a
single generic image-quality score.

The blind review here should therefore not ask only “which is prettier?” It should record:

1. intended correction;
2. personal-feature and mark preservation;
3. skin texture/material;
4. eyes, lips, teeth, and makeup realism;
5. geometry/expression fidelity;
6. overall preference.

### 6.3 The user's pairs are exemplars, not yet training data

[MirrorPPR](https://arxiv.org/abs/2606.29308) studies transferring subtle structural
retouch operations from an exemplar before/after pair. That makes the five user-created
Meitu pairs conceptually useful as operation examples. They are not sufficient training
data: they cover one subject, two shoots, inconsistent export settings, and apparently
different edit strengths. A safe near-term use is to estimate bounded operation targets
and generate candidates, not to train or ship a generative structural editor.

## 7. Recommended next experiment

### P0 — owner blind ranking now

Review each source plus A/B/C without opening `blinded_key.json`. Rank A/B/C separately
at normal view and face view, then note any defect by the six dimensions above.

| Case | Full review | Face review |
|---|---|---|
| DSCF2362 | `test_output/competitive/meitu_reaudit_20260904/DSCF2362/blinded_full_review.jpg` | `test_output/competitive/meitu_reaudit_20260904/DSCF2362/blinded_face_review.jpg` |
| DSCF2365 | `test_output/competitive/meitu_reaudit_20260904/DSCF2365/blinded_full_review.jpg` | `test_output/competitive/meitu_reaudit_20260904/DSCF2365/blinded_face_review.jpg` |
| DSCF2306 | `test_output/competitive/meitu_reaudit_20260904/DSCF2306/blinded_full_review.jpg` | `test_output/competitive/meitu_reaudit_20260904/DSCF2306/blinded_face_review.jpg` |
| DSCF2308 | `test_output/competitive/meitu_reaudit_20260904/DSCF2308/blinded_full_review.jpg` | `test_output/competitive/meitu_reaudit_20260904/DSCF2308/blinded_face_review.jpg` |
| DSCF2310 | `test_output/competitive/meitu_reaudit_20260904/DSCF2310/blinded_full_review.jpg` | `test_output/competitive/meitu_reaudit_20260904/DSCF2310/blinded_face_review.jpg` |

A useful reply format is: `2362: B > C > A; B best overall, C best texture; A changes
identity`. The key should be opened only after all five rankings are frozen.

### P1 — controlled consumer rerun

Re-export the same five sources from one recorded Meitu mode/preset with:

- exact app version/build;
- Smart/Classic/manual mode;
- all non-zero controls or a preset identifier;
- no per-image manual correction for the automatic arm;
- a separate timed manual-cleanup arm if desired;
- original dimensions, sRGB target, fixed JPEG quality, and metadata policy.

Then rerun the same Retouch recipes at native delivery resolution through the production
writer. Keep the 2048 px set as a screening arm only.

### P1 — hard-case expansion

Add at least one consented case for each missing condition: darker skin appearance,
glasses, closed/occluded eye, strong yaw/profile, facial hair, exposed neck/body skin,
group scene, small face, printed/poster face, and minimal/no makeup. The current five
cannot support demographic, general quality, or release claims.

### P2 — professional workflow arm

Run MeituYunxiu separately from the consumer app, and record RAW handling, batch sync,
exception correction, operator time, export fidelity, and cloud/privacy state. Do not pool
those results with this consumer-app pilot.

## 8. Product decision boundary

- Do not retune `natural` or `convention_clear_v1` from these metrics alone.
- If the owner consistently prefers the Meitu-like candidate, add a separate named
  porcelain/desaturated candidate; do not silently change the source-authoritative
  defaults.
- Prioritize poster-face rejection/manual subject correction before claiming unattended
  event reliability on scenes like DSCF2362/2365.
- Keep automatic body reshape parked until a controlled preference result and background-
  distortion/identity safety protocol justify it.
- Do not claim Retouch beats Meitu, or vice versa, from five unblinded single-subject
  examples.
