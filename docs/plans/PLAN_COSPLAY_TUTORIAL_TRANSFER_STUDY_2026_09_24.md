# Proposed cosplay tutorial transfer study — 2026-09-24

**Status:** Documentation-only proposal. No source/output portraits were
selected or processed, and no implementation or evaluation was run. This plan
does not authorize a photo study or a production change.

This study plan follows the
[batch review](RESEARCH_COSPLAY_TUTORIAL_BATCH_REVIEW_2026_09_24.md) and
[face evidence ledger](RESEARCH_COSPLAY_TUTORIAL_FACE_EVIDENCE_LEDGER_2026_09_24.md).
Before selecting a recipe for a future run, consult the dated
[current-code crosswalk](RESEARCH_COSPLAY_TUTORIAL_CURRENT_CODE_CROSSWALK_2026_09_25.md)
for the inspected branch's mark-policy, catchlight, slimming, and body-mask
boundaries; recheck the active checkout because that audit is branch-specific.
It asks which ideas from photographed Photoshop tutorials can guide a
preservation-aware Retouch review. The screenshots document one artist's manual
workflow; they do not establish image quality or a reusable algorithm.

## 1. Decision question

Which parts of the tutorial workflow produce a requested, visible improvement
on real cosplay photographs while preserving the person's appearance, makeup,
marks, hair, and costume boundaries?

The proposed review separates three intents:

| Intent | Tutorial cues | Review question |
|---|---|---|
| Photo-preserving finish | Local skin color at 143; subject light/color at 219 and 223–225 | Can a bounded tone or lighting change improve the image while keeping skin detail, makeup, and existing light direction? |
| Hair and subject boundary | Hair direction and strand work at 168 and 220 | Are existing hair edges and strands retained without spill into costume or background? |
| Explicit stylized beauty edit | Added eye highlights at 144 and 225; face slimming at 167 and 225 | Does a clearly requested stylized edit achieve its intent while making its visual and likeness changes reviewable? |

The first two intents are candidates for preservation review. Eye-highlight
painting and geometry reshaping belong in a separate, user-directed creative
study. They are excluded from any natural-face baseline or default recipe
qualification. This matches the existing
[cosplay portrait polish plan](PLAN_COSPLAY_PORTRAIT_POLISH.md), which defers
synthetic catchlights and geometry changes from its v1 visual contract. Recheck
that plan and the current branch before a future study is run.

## 2. Evidence available and missing

The review bundle contains tutorial-page photographs, including visible
intermediate composites. It does not contain the original photographs used by
the tutorials, editable layer files, or an independent review of the result.
The locally synced Google Drive `My Drive/Photography` folder does contain
candidate source/output collections by filename; the read-only
[Drive inventory](RESEARCH_COSPLAY_TUTORIAL_DRIVE_PHOTO_INVENTORY_2026_09_25.md)
records their counts and limits. No pair has yet been visually or technically
verified for this study, and no source/output photo has been selected or
processed. Before selection, confirm pair identity, stage, provenance, and
requested intent, then recheck the current branch, recipe, runtime, and output
workflow.

Any later study would need owner-authorized photographs and their exact
pre-edit originals and delivered outputs. Before selection, recheck the current
branch, recipe, runtime, and output workflow. The plans dated July and
September describe their own snapshots and do not certify today's checkout.

## 3. Proposed pilot cases

Select cases by visible photographic conditions and edit opportunities. Do not
infer sensitive traits from appearance. Keep the pilot small enough for careful
review, and choose its size after agreeing on intended decisions and harm
severity; this document sets no sample count or acceptance threshold.

Seek examples that cover:

- frontal and turned faces, including hair crossing the face;
- soft, hard, mixed, and low light with different skin exposure levels;
- near-white or silver wigs against bright backgrounds, including backlit
  windows. A separate [person-gate false-negative report](RESEARCH_PERSON_GATE_WIG_FALSENEG_2026_09_25.md)
  records a single-shoot cluster of real faces rejected by the person gate in
  this condition; treat it as one coverage warning, not a general rate;
- visible makeup, drawn marks, beauty marks, and facial hair where available;
- dark and light wigs, loose flyaways, and detailed costume/skin boundaries;
- single-person and multi-person frames, including small faces in the full image;
- a range of face sizes, focus levels, and delivery resolutions.

Record how many candidates were screened, selected, excluded, and why. Keep
images of the same shoot or known person together when dividing data for any
later calibration or qualification. Use an owner-provided grouping; do not add
face recognition to create it. Record both initial face detections and the
person-gate outcome. A face detected and then rejected by the gate is a
coverage miss, not a successful preservation result. Recheck the gate and
branch revision before a future run.

## 4. Freeze each comparison

For every selected photo, retain an unchanged source and make each derived
output a separate file. Pair records by stable, non-identifying asset ID and
content hash. Record:

- permission/provenance and source hash;
- code revision plus dirty patch, runtime and model versions, if applicable;
- recipe, resolved settings, requested edit intent, and per-face target;
- input/output dimensions, orientation, color profile/domain, and export format;
- the exact processing stage represented by every saved image;
- whether each intended operation executed, skipped, abstained, or failed;
- per-face detection, person-gate coverage, and whether the intended operation
  reached that face.

Use the same source, delivery dimensions, and viewing conditions across
comparisons. A proxy render can help locate gross failures; it cannot certify
native-detail preservation or final export appearance.

Use the existing [QA validity study](PLAN_RETOUCH_QA_VALIDATION_2026_09_21.md)
for stage-pair, provenance, missing-measurement, and delivery requirements.
This plan adds the cosplay-specific preservation labels and decision questions
below.

## 5. Candidate comparison arms

Do not combine several effects into one candidate during the first review.
Each optional effect should have its own clearly named arm.

| Arm | Output | What it isolates |
|---|---|---|
| Source | Unchanged input, decoded using the recorded input path | Starting appearance and source limitations |
| Current workflow | The current agreed natural or cosplay workflow, rechecked before the run | Existing behavior on the same source |
| Skin tone | Only the user-requested, bounded skin tone adjustment | Skin color change and mask boundary behavior |
| Local light | Only the user-requested local lighting change | Light change, clipping, and transition halos |
| Hair boundary | Only a defined hair-edge correction, when present in the current workflow | Strand retention, edge leakage, and costume/background ownership |
| Stylized eye treatment | Separate opt-in output only if the user explicitly requests eye-light painting | Requested eye emphasis, source catchlight, and artificial appearance |
| Stylized reshape | Separate opt-in output only if the user explicitly requests geometry change | Requested shape change, unintended change elsewhere, and perceived likeness |

The source, current workflow, skin tone, local light, and hair boundary arms
form the preservation pilot. The final two arms require a separate decision and
must not be merged into a natural-photo result.

## 6. Review sheet and endpoints

Hide method names and randomize candidate order. Reviewers will still see the
visual differences between outputs. Compare the same crop scale and full-frame
context; keep aesthetic preference separate from preservation judgment.
Preserve reviewer disagreement and an explicit “uncertain / cannot judge”
response.

For each case, capture:

| Endpoint | Reviewer prompt |
|---|---|
| Requested benefit | Did the requested color, light, or boundary change happen, and is it visible at intended viewing size? |
| Skin detail | Are pores, natural texture, fine lines, and form shading retained where they were present? |
| Marks and makeup | Were requested-to-preserve beauty marks, blemishes, freckles, tattoos, or drawn makeup retained? If a mark change was requested, did only that mark change? |
| Eyes | Were the iris, sclera, lashes, brows, and existing catchlights preserved? Was any new catchlight explicitly requested? |
| Hair and boundaries | Are strands and flyaways plausible? Did the edit leak into hair, costume, jewelry, or background? Are halos or cutout edges visible? |
| Person/face ownership | Was the intended person and face edited? Were other people or nearby skin/costume left alone? |
| Coverage | Was the face detected and retained by the person gate? Were any intended edits skipped because face/person support was absent? Count these outcomes in the study denominator. |
| Color and light | Did skin hue or exposure drift beyond the request? Did local light contradict visible scene lighting or clip detail? |
| Geometry and likeness | For the opt-in reshape arm only, is the requested change limited to its declared support, and do reviewers judge the person still recognizable as the source subject? |
| Delivery | Does the decoded final export preserve the reviewed appearance at native and intended display size? |
| Disposition | Accept for this requested edit, reject for harm, refer for manual review, or uncertain—with reason. |

Record case counts and dispositions by image and independent shoot/person group.
Show both harm among accepted edits and the rate of abstention/referral. A
high abstention rate is a coverage limitation; a low harm count on a handful of
cases is not a safety guarantee.

## 7. Review sequence

1. Confirm the requested edit intent and what must be preserved before looking
   at candidate outputs.
2. Inspect the full frame and the same native-resolution face/hair crops for
   each arm. Include delivered-size review for export appearance.
3. Complete preservation and benefit judgments independently, then record
   reviewer disagreement before any consensus discussion.
4. Inspect masks and stage comparisons only after the blinded visual ratings;
   use them to explain errors, not to overrule the visible result.
5. Group errors by edit, region, pose/light condition, and independent source
   group. Retain failures and uncertain cases in the report.
6. Revisit any case where the measured support differs from the reviewed
   intended region. Do not treat a correct detector count as proof of correct
   edit ownership.

## 8. Decision gates

The pilot may only make a bounded decision about its reviewed cases.

| Gate | Evidence required |
|---|---|
| Ready to review | Authorized source/output pairs; frozen comparison conditions; documented edit intent; reviewer sheet and exclusions prepared |
| Candidate worth further study | Visible requested benefit on the reviewed cases, no unresolved severe preservation error, and failures/uncertainty reported by stratum |
| Eligible for calibration planning | More than one independent shoot/person group, documented repeatability, owner agreement on acceptable benefit and harm, and an explicit coverage target |
| Eligible for product proposal | Separate product intent and interaction design; user controls for stylized changes; current-code audit; focused regression and delivery plan |

Numerical thresholds, sample sizes, calibration, recipe changes, and automatic
acceptance rules remain undecided. The pilot cannot support demographic
generalization, algorithm certification, or a release claim by itself.

## 9. Current result

This document defines a proposed review protocol only. The current evidence
supports manual compositing observations and a shortlist of questions. It does
not show that the tutorial procedures improve Retouch output. Source/output
photos, annotations, ratings, and measurements remain to be collected in a
separately authorized research pass.
