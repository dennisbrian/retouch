# Cosplay tutorial ideas and current Retouch code — 2026-09-25

**Scope:** Read-only source audit connecting the tutorial observations in the
[batch review](RESEARCH_COSPLAY_TUTORIAL_BATCH_REVIEW_2026_09_24.md) and
[face evidence ledger](RESEARCH_COSPLAY_TUTORIAL_FACE_EVIDENCE_LEDGER_2026_09_24.md)
to the current Retouch checkout. This records code paths and remaining evidence
gaps; it does not select a feature or certify visual quality.

**Checkout inspected:** `feat/wire-under-reached-ops` at `6ef691d`. The worktree
was clean before this documentation update. Findings below come from source
inspection only: no tests, renders, photo processing, branch comparison, or
release check were performed.

## Summary

Several tutorial ideas have related Retouch controls already: face-local color
and light operations, catchlight enhancement, face slimming, hair masks, a
cosplay portrait recipe, and mark-policy plumbing. They do not reproduce the
manual Photoshop steps one-for-one. The relevant boundaries are that the
portrait recipe does not enable a mark policy, its catchlight setting does not
create a catchlight when none is present, its face-slimming value is zero, and
the measured face-anchored body mask remains QA-only. No matched source/output
photo review has tested the tutorial ideas in Retouch.

The next useful evidence is a small authorized source/output study using a
frozen recipe and explicit preservation notes. Another general algorithm survey
would not resolve the current uncertainty.

## Tutorial-to-code crosswalk

| Tutorial observation | Related Retouch code in this checkout | What the mapping does and does not establish |
|---|---|---|
| Isolate skin for color adjustment and protect nearby features (143); add local subject light (219, 225) | `FaceRegions` supplies face-area masks; `retouch/lighting.py` estimates a face-local key-light direction. The engine passes that estimate to face processing, and `retouch/perf_optimizations.py` uses it to orient `sculpt` when explicit virtual relight is off. The engine also has explicit relight and body light controls. | This is selective retouch support, not the tutorial's painted Photoshop layer/mask workflow. The estimator does not change pixels by itself. Source inspection does not show that a Retouch light operation improves the same cases. |
| Paint a new eye highlight (144, 225) | `retouch/eyes.py` can amplify detected iris catchlights. If no catchlight is detected, `_enhance_catchlights` returns unchanged unless `synthetic_fallback` is enabled. The `cosplay_portrait_polish_v1` recipe sets `eyes.catchlight` to `0.12` and does not set synthetic fallback. | That recipe requests subtle enhancement of photographed catchlights; it does not request painting a missing highlight. Synthetic catchlight creation is a separate behavior and is not validated by the tutorial screenshots. |
| Refine hair by following existing strand direction (168, 220) | Face parsing exposes hair regions; Retouch has hair-related operations and excludes parsed hair from parts of the body-skin mask. | These provide masks and retouch controls, not the tutorial's manual strand drawing. There is no evidence here that the source bundle validates automatic strand synthesis or flyaway repair quality. |
| Slim the face as an explicit beauty edit (167, 225) | Face slimming is a separate parameter. `cosplay_portrait_polish_v1` sets `slimming` to `0.0`; the older general `cosplay` recipe sets it to `50.0`. | Recipe choice changes the intent. A preservation study must record the exact recipe and resolved settings; the generic `cosplay` recipe is not a no-reshape control. |
| Preserve drawn makeup, moles, scars, and other intentional details | `retouch/marks.py` defines `MarkRecord` and named policies. `mark_policy` is available as a parameter and some recipes opt into `protect_identity` or `preserve_all`. | The `cosplay_portrait_polish_v1` recipe does not set `mark_policy`, so it inherits the `legacy` default. In addition, policy masks do not protect every smoothing path: the source documents guided-engine-specific protection, while some cosplay recipes use anisotropic smoothing. Class definitions and a policy option are not proof that each cosplay mark is detected or preserved. |
| Keep face and exposed-body finish coherent | `cosplay_portrait_polish_v1` includes body smoothing, face-tone matching, body relight, and dodge/burn. `retouch/harmony.py` measures face/body texture and specular parity, and its face-anchored body mask is documented as QA-only. | The production body mask still starts with fixed LCH hue/chroma bounds in `engine._stage_body_skin`; smoothing is a body-scale guided filter and face matching uses median LAB tone. The source's connected-component logic addresses the earlier fixed 12-step contiguity failure, but no render was produced in this audit. Texture-parity thresholds remain uncalibrated for automatic correction. |

## Status corrections to older plan text

The July cosplay plan is a useful design record, but several status statements
are now historical. This checkout contains `retouch/marks.py` and a registered
`cosplay_portrait_polish_v1` recipe; face-light estimation is passed into the
sculpting path; and the body-mask contiguity check uses person-mask connected
components rather than the earlier fixed 12-step walk. The following gaps still
apply in the inspected source:

- There is no shared production `PortraitRegions` object assembling confident
  skin, costume, hair, makeup, tattoo, and preserve masks for the whole portrait.
- `cosplay_portrait_polish_v1` does not opt into `mark_policy`; mark-policy
  coverage varies by operator and smoothing engine.
- The face-anchored body mask in `harmony.py` is for QA. Production body-skin
  detection still uses fixed hue/chroma bounds.
- The body stage has not gained the proposed face/body texture-parity treatment;
  its existing smoothing and tone matching are separate from that proposal.
- The tutorial-specific questions remain unmeasured: manual-style local light,
  hair-boundary fidelity, mark/makeup preservation, perceived benefit, and
  export appearance on matched source/output photos.

These are source-presence findings, not proof that the operations execute
successfully or look acceptable on every input. The current branch is also not
evidence of a packaged release.

## Study setup implications

Before any authorized photo comparison, pin:

1. The exact recipe (`cosplay_portrait_polish_v1`, a protected variant, or a
   different workflow) and all resolved settings.
2. Whether mark policy is `legacy`, `protect_identity`, or `preserve_all`, and
   which operations actually consume each compiled mask on that path.
3. Whether the intent is source-preserving light/color work, synthetic eye
   highlight, hair-strand addition, or face reshaping. Keep these as separate
   comparison arms.
4. Source and final-export pairs, stage provenance, native-resolution crops,
   and the intended display size, as defined in the
   [transfer study plan](PLAN_COSPLAY_TUTORIAL_TRANSFER_STUDY_2026_09_24.md).

For a natural or preservation-oriented arm, use a recipe with face slimming at
zero and do not treat the `cosplay` recipe's nonzero slimming as a neutral
baseline. Do not infer that selecting a mark-policy preset protects every
operation; inspect its consumer path and review the actual output.

## Recommended next step

The face-focused documentation set is in place: the tutorial review, face
evidence ledger, transfer-study protocol, and current-code crosswalk. The
broader [stylized-compositing review](RESEARCH_COSPLAY_TUTORIAL_COMPOSITING_2026_09_25.md)
is documented separately. The next face-quality decision depends on authorized
matched source/output portraits. Until those are identified, these documents
support study setup only; they do not support a feature recommendation or
quality claim.
