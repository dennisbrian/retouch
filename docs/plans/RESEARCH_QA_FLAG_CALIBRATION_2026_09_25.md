# QA flag calibration (2026-09-25)

**Question.** The batch review page's "Flagged" filter was noise: every real
render carried 4–6 flags. Which QA detectors fire on clean retouches, why,
and what measurement separates a clean retouch from a real defect?

**Answer.** Four detectors (banding, plastic_skin, seam, asymmetry) used
absolute measurements over the person mask and flagged 100% (asymmetry 75%)
of real renders, edited or not. They now measure what the retouch *changed*
against the input photo. color_drift's p99 tail gate flagged most styled
cosplay renders because the person mask holds intentional lip/eye/makeup
edits; the tail gate moved to p95 > 20°. Every flag name and the meaning of
`flagged` are unchanged.

## Corpus

7 real photos, mixed skin tones and framing: `obama.jpg`, `biden.jpg`,
`two_people.jpg` and three `knn_examples/train` portraits from
ageitgey/face_recognition's examples, plus the cosplay tile
`docs/reference_targets/chang_e_cosplay_tamed_shine.jpg[0:480, 720:1080]`.
5 recipes: `natural`, `cosplay_clear_v1`, `cosplay_feed_pop_v1`,
`cosplay_neon_night_v1`, `cinema_grade_v1`. 35 renders through
`RetouchEngine.process`, with `qa_detectors.run_all`'s exact inputs (output,
person mask, face-skin mask, input and geometry references) captured to
disk so detectors could be iterated offline.

## Before (main at df7bc30)

| Detector | Flagged | Why |
|---|---|---|
| banding | 28/28 | Sobel > 0.8 L on low-variance pixels: any 1-level 8-bit step qualifies, so every smooth gradient in every photo counts. |
| plastic_skin | 28/28 | hf / mid-frequency ratio < 0.6 over the person mask; clothes and hair dominate the mid band, so the ratio is ~0.05–0.09 on any portrait (already recorded in TODO_WEEK_2026_09_21 Q1). |
| seam | 28/28 | Mean gradient on the person outline minus context; the subject/background edge is naturally 22–29 L-levels. |
| asymmetry | 21/28 | 3×3 whole-frame grid over the person mask compares raw texture energy of hair, suit and face zones. |
| color_drift | 0/7 natural, 15/21 styled | p99 Δh > 10° over the person mask; styled recipes moved the mean only 1.4–3.7° but lip/eye/makeup edits fill the tail. |

The first four were counted on the 28 renders of the first 4 recipes via
`result.qa_evidence`.

## Changes

| Detector | New measurement | Threshold |
|---|---|---|
| banding | Staircase steps: a 2–16 code jump in 8-bit luma with a perfectly flat 3-px run on **both** sides (either axis), dilated to the plateau. Differential: plateaus already present in the input (counting its 1-level staircases, i.e. JPEG blocks a brightening curve stretches) are removed. Score = max(new fraction of person, new fraction of face skin). | > 0.06 (unchanged name `BANDING_THRESHOLD`) |
| plastic_skin | Face-skin texture retention: std(L − G_σ2(L)) of output / input over the face-skin mask. No reference → `not-run`, never flagged. | < 0.60 (unchanged) |
| seam | Output outline excess minus input outline excess, the latter scaled by how much the context gradient grew (contrast/clarity looks sharpen every edge). | > 5 L-levels (unchanged) |
| asymmetry | Per-zone texture retention (output / input) on a 3×3 grid over the face-skin bounding box; zones < 64 px skipped. Without a reference the legacy absolute mode remains. | > 0.40 (unchanged) |
| color_drift | Tail gate p99 > 10° → p95 > 20°; mean > 6° unchanged. `deltaH_p99_deg` still reported. | mean > 6° or p95 > 20° |

## After (same 35 renders)

| Detector | Worst clean render | Flagged |
|---|---|---|
| banding | 0.026 (bright JPEG-blocky cheek) | 0/35 |
| plastic_skin | retention 0.89 | 0/35 |
| seam | 1.4 L | 0/35 |
| asymmetry | 0.19 | 0/35 |
| color_drift | styled cosplay p95 ≤ 11.8°, mean ≤ 3.7° | 7/35: all 7 `cinema_grade_v1` renders (mean 21–74°), the intended skin cast |

Remaining flags: clipping 3/35 (strong grades), harmony 6/35 (mark
retention, unchanged). A real CLI batch (`./run batch` with
`cosplay_clear_v1`, the same 7 photos) went from 7/7 images flagged to 1/7
(harmony, freckles on `rose_leslie`).

## Planted defects (sensitivity)

| Defect | Result |
|---|---|
| Face skin blurred σ=6 and posterized to 3-level steps | banding 0.095–0.57, flags 7/7 |
| Face skin posterized to 2-level steps | flags 6/7 (not the 3,870-px cosplay tile face) |
| Whole frame posterized to 8-level steps | banding ≥ 0.099, flags 7/7; 6-level: 6/7 |
| Face skin blurred σ=1.5 | plastic_skin retention 0.29–0.52, flags 7/7 |
| One cheek zone blurred σ=1.5 / σ=3 | asymmetry flags 4/7 (0.41–0.80); smooth or small faces read 0.15–0.30 |
| 2-px line +10 L traced round the whole outline | seam +6.0–9.2, flags 7/7; a +5 L line (1.8–3.3) does not |
| +8° hue rotation of the whole person | mean 5.3–12.5°, flags 6/7 |
| 25° cast on the lower half of the face only | flags 5/7: casts on under ~5% of the person barely move p95 |

## Limits

- Differential checks need the input photo. The pipeline always supplies
  it; when a reshape's geometry-only frame is unavailable, plastic_skin is
  `not-run` and seam/asymmetry/banding fall back to their absolute forms.
- The seam gain correction is approximate: a synthetic ×1.3 contrast
  stretch on a 95-L-level outline still reads +13; real looks topped out at
  1.4.
- Corpus is small (7 photos) and has one cosplay subject; no night-neon or
  heavy-makeup photo, and only one darker-skinned subject.

Code: `retouch/qa_detectors.py`; tests: `tests/test_qa_flag_calibration.py`.
