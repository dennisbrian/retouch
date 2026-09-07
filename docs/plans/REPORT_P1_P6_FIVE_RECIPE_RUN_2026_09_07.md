# P1–P6 five-recipe native-resolution run

Date: 2026-09-07 (Asia/Kuala_Lumpur)

This report records the first real-image run after commit `86cb0f2`
(`Productionize accepted P1-P6 retouch safeguards`). It is a delivery/QA
record, not a claim that the five recipes are universally suitable or that
P4/P5/P6 experimental operations were silently enabled.

## Corpus and command

Source folder (unchanged):

`/Users/dennis/Desktop/Priority for Printing/`

The corpus contains five native JPEGs, all 4160×6240 pixels:

`DSCF2650.jpg`, `DSCF2709.jpg`, `DSCF2810.jpg`, `DSCF2843.jpg`, and
`DSCF2912.jpg`.

The five recently added recipes were rendered with the face-aware batch path:

* `heirloom_archive_v1`
* `documentary_preserve_v1`
* `tired_eye_rescue_v2`
* `cinema_grade_v1`
* `studio_headshot_protected_v1`

Command:

```text
RETOUCH_CACHE_DIR=/private/tmp/retouch RETOUCH_GPU=0 \
RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python scripts/batch/recipe_batch.py \
  "/Users/dennis/Desktop/Priority for Printing" \
  --output "/Users/dennis/Desktop/Priority for Printing_retouched_test_20260907" \
  --recipes "heirloom_archive_v1,documentary_preserve_v1,tired_eye_rescue_v2,cinema_grade_v1,studio_headshot_protected_v1" \
  --compare --format jpg --quality 94 --sheet-cols 5 --cell-size 420
```

Outputs are in the new sibling folder:

`/Users/dennis/Desktop/Priority for Printing_retouched_test_20260907/`

The prior `...20260906` output was not overwritten. The new folder contains
25 retouched images, 25 source/result comparisons, five contact sheets, and a
batch `manifest.json` (57 files including the macOS `.DS_Store`). Every row in
the manifest is `done`; failures: **0**.

## Runtime

The manifest's per-arm timings are native-resolution wall time:

| Recipe | Images | Total seconds | Mean seconds/image |
|---|---:|---:|---:|
| `heirloom_archive_v1` | 5 | 443.574 | 88.715 |
| `documentary_preserve_v1` | 5 | 655.297 | 131.059 |
| `tired_eye_rescue_v2` | 5 | 407.821 | 81.564 |
| `cinema_grade_v1` | 5 | 235.116 | 47.023 |
| `studio_headshot_protected_v1` | 5 | 854.804 | 170.961 |
| **all arms** | **25** | **2596.612** | **103.864** |

The run used the existing sequential engine. No new default memory or runtime
behavior was introduced by this batch. One non-fatal `Freckle: 2000+ components
detected; capping at 2000` warning was emitted; the affected recipe rows still
completed successfully.

## Objective output deltas

These are source-to-output JPEG channel statistics, useful for spotting a
broken or unexpectedly aggressive arm. They are not a perceptual quality
score, and JPEG encoding plus the absence of a ground-truth retouched target
means they cannot prove improvement.

| Recipe | Mean absolute DN | Mean RMSE DN | Worst absolute DN | Mean changed pixels | Mean clipped pixels |
|---|---:|---:|---:|---:|---:|
| `heirloom_archive_v1` | 0.770 | 1.211 | 165 | 61.08% | 1.56% |
| `documentary_preserve_v1` | 0.863 | 1.512 | 157 | 62.74% | 1.34% |
| `tired_eye_rescue_v2` | 0.845 | 1.498 | 194 | 62.87% | 1.38% |
| `cinema_grade_v1` | 8.005 | 10.361 | 184 | 99.82% | 15.10% |
| `studio_headshot_protected_v1` | 1.541 | 2.397 | 195 | 92.05% | 1.16% |

The cinematic arm is intentionally the strongest creative grade and has the
highest clipping rate; it needs owner review before print delivery. The other
four arms remain comparatively restrained in global DN movement, but these
figures do not certify mark preservation, skin quality, or aesthetic preference.

## What P1–P6 contributed

* **P1:** existing guided-filter smoothing and its mark-aware routing remained
  active; no algorithm redesign was made.
* **P2:** the existing large-sigma blur and allocation safeguards remained in
  force. The batch completed without an allocation failure.
* **P3:** the registered stage order and enable/disable routing were used by
  the face-aware engine path.
* **P4:** no automatic makeup inversion or bare-skin reconstruction ran. The
  bounded attenuation leaf remains caller-supplied-support/reference only.
* **P5:** no self-blend tone mode was injected into these recipes; the
  analytical operators remain explicit caller-only controls and Photoshop
  parity remains unverified.
* **P6:** legacy clarity remained the default. `clarity_noise_aware` was not
  enabled by any recipe or batch flag, so this run does not silently promote
  the experimental noise-gated candidate.

The concrete post-production improvement exercised here is safer integration:
the corrected float clarity dispatch avoids the historical LAB conversion
artifact, stage registry behavior is covered, and P4/P5/P6 opt-ins cannot be
activated accidentally by ordinary recipe selection. The run itself found no
render failures or NaN/range errors.

## Review shortlist

Start owner review with the five contact sheets:

* [`heirloom_archive_v1/contact_sheet.jpg`](</Users/dennis/Desktop/Priority for Printing_retouched_test_20260907/heirloom_archive_v1/contact_sheet.jpg>)
* [`documentary_preserve_v1/contact_sheet.jpg`](</Users/dennis/Desktop/Priority for Printing_retouched_test_20260907/documentary_preserve_v1/contact_sheet.jpg>)
* [`tired_eye_rescue_v2/contact_sheet.jpg`](</Users/dennis/Desktop/Priority for Printing_retouched_test_20260907/tired_eye_rescue_v2/contact_sheet.jpg>)
* [`cinema_grade_v1/contact_sheet.jpg`](</Users/dennis/Desktop/Priority for Printing_retouched_test_20260907/cinema_grade_v1/contact_sheet.jpg>)
* [`studio_headshot_protected_v1/contact_sheet.jpg`](</Users/dennis/Desktop/Priority for Printing_retouched_test_20260907/studio_headshot_protected_v1/contact_sheet.jpg>)

Native source/result pairs and exact timings are listed in
[`manifest.json`](</Users/dennis/Desktop/Priority for Printing_retouched_test_20260907/manifest.json>).

Human review is still required for eyeliner/lash boundaries, identity marks,
skin texture, hair, under-eye treatment, background halos, and print clipping.
The five-image set is a useful smoke/owner-review batch, not a held-out quality
certification corpus.

## Follow-up decision

The run is **operationally successful** (25/25 renders, native resolution,
zero failures) and the P1–P6 production checkpoint is committed. It is not
evidence to change recipe defaults or to claim P4 bare-skin recovery, P5
Photoshop parity, or P6 noise-aware superiority. Any recipe promotion should
follow owner review of the native pairs and the clipping noted above.
