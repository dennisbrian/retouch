# Proposed P9 — Calibrating when retouch evidence is usable

Date: 2026-09-11. Inspected source: `cabf3bc7c246c1662779c3f517a7cb89fb33b4a5`.
Status: documentation-only research proposal; P9 topic has not been confirmed.
Evidence: current source inspection and six primary research sources.
No implementation, models, corpus assignments, image processing, or experiments.

## 1. Scope and recommendation

The existing [Skin ProMax roadmap](PLAN_SKIN_PROMAX.md) defines P1–P8, with no
P9. The [first P8 report](RESEARCH_P8_AGING_VECTOR_2026_09_09.md) also records
an earlier P9 request being clarified as P8. This document therefore proposes
a next research milestone; it does not claim to recover an existing P9 spec
or silently add an approved feature to the roadmap.

**Recommendation: investigate operation-specific evidence calibration, starting
with whether P8 measurement support is usable.** A later P7 study can ask
whether inferred target skin belongs to the intended person. The eventual
product benefit would be fewer incorrect automatic edits and clearer reasons
for review. That benefit has not been demonstrated on Retouch images.

The question is concrete: when the system accepts a region, how often is that
region actually suitable for the requested measurement or operation, and how
much useful work is lost when it abstains?

This extends the existing FA-01 evidence program and Safe Auto contract. It is
an evaluation milestone, not a claimed ninth independent skin-physics axis.
If a different P9 topic was intended, retain this as an optional proposal.
The [companion protocol](PLAN_P9_SELECTIVE_RETOUCH_RESEARCH_2026_09_11.md)
defines the smallest useful pilot and its later decision gates.

## 2. Current implementation evidence

These findings describe functions at the inspected commit, not observed error
rates or an audit of every caller in the application.

| Surface | What the code establishes | What remains unestablished |
|---|---|---|
| [Parser confidence](../../retouch/parsing.py), `_bisenet_confidence_by_region` and `_bisenet_boundary_confidence_by_region` | Region and boundary summaries of the softmax top-two margin. Documentation explicitly limits them to observations. | Probability of correct material ownership, or probability that a retouch will preserve the portrait. |
| [Existing parser guard](../../tests/test_batch_onnx.py), `test_parse_confidence_not_read_by_any_op` | A regression guard exists against consuming parser confidence as an operation control. It was inspected, not executed in this research. | Permission to wire a new threshold into rendering. |
| [P8 readout](../../retouch/aging_cues.py), `measure_p8_cues` | Explicit missing/invalid/insufficient support, descriptor units, support counts, and raw measurements. | Semantic correctness of a numerically valid mask, calibrated uncertainty, or apparent-age validity. |
| [P8 underlying metrics](../../retouch/perceptual_metrics.py) | Skin confidence is `clip(pixel_count / 800, 0, 1)`; feature confidence is `clip(min(feature_pixels, surround_pixels) / 500, 0, 1)`. | A value of 1 does not mean 100% correctness. Pixel quantity alone does not establish material identity or independent information. |
| [P7 stage](../../retouch/engine.py), `_stage_cross_region_skin`, and [support helper](../../retouch/cross_region_skin.py) | Single-detected-face restriction, conflict abstention, target support/exclusions, bounded propagation, and structured reasons. | One detected face does not prove only one person is present. Connected person support and color compatibility are hypotheses about ownership, not human annotations. |
| [Safe Auto](../../retouch/safe_auto.py) | `SafeAutoDecision` already represents apply/dampen/skip/review, evidence, reasons, and scale. `decide_mask_stage` combines coverage/stability/model scores by minimum. | That minimum is not a fitted probability. Default action bands are policy constants, not calibration results established here. |
| [Corpus manifest v3](../../retouch/corpus_manifest.py) | Person/group split checks and `dev`, `calibration`, `locked_test` vocabulary already exist. | Schema support does not demonstrate a sufficiently populated, independently labeled calibration or test cohort. |

An analytic consequence of the P8 formulas: adding pixels through resampling
can increase the reported confidence without supplying new semantic evidence.
This is a deduction from the formula, not a resampling experiment run here.
Keep this quantity interpretable as a support heuristic in the study; a schema
rename or compatibility migration would be a separate implementation decision.

Likewise, `confidence_evidence` distinguishes detector-reported confidence
from MediaPipe compatibility values. Its `confidence_measured` provenance flag
does not establish calibration against Retouch correctness labels.

## 3. Relationship to existing research

| Existing track | Keep its responsibility | Increment proposed here |
|---|---|---|
| [FA-01 ontology/evidence](RESEARCH_FA01_ONTOLOGY_AND_EVIDENCE_2026_09_05.md) | Materials, protection policy, boundary evidence, provenance | Test the relationship between an observed score and independently reviewed support errors. |
| [P8 cue validity](RESEARCH_P8_CUE_VALIDITY_2026_09_10.md) | Descriptor repeatability, apparent-age association, then perceptual intervention | Qualify acceptance/review decisions for descriptor support; do not relabel the entire P8 program as P9. |
| [P7 research](RESEARCH_P7_CROSS_REGION_SKIN_CONSISTENCY_2026_09_09.md) | Bounded face-edit propagation with explicit ownership limits | Compare accepted target support with reviewed ownership and exclusions before claiming reliable automation. |
| Skin ProMax P6 | Learn the owner's preferred tuning | Correctness and preservation criteria; preference is a separate endpoint. |
| Safe Auto | Existing decision representation and application | Evaluate candidate policies offline before considering changes to that representation or its consumers. |

Older FA-01 statements that no split schema exists are superseded by current
`corpus_manifest.py`. Historical sample counts, missing strata, and identity
assignments were not re-inventoried here and must not be treated as current
population estimates. A parser margin also is not measured boundary leakage:
leakage requires independent reference support or a directly reviewed edit.

## 4. What the literature supports

**Calibration and accuracy are different.** Guo et al. evaluate confidence
calibration and show that temperature scaling can improve it on their tested
classifiers. This motivates testing score reliability, not assuming that
softmax magnitude is a probability of a successful retouch. Their results do
not validate calibration of Retouch's pixel counts or hand-built score minimum.
[S1: Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).

**Abstention must be evaluated with coverage.** SelectiveNet studies the
tradeoff between errors on accepted predictions and the fraction accepted.
Retouch can borrow that evaluation question without adopting its trained
network. A policy that rejects every face cannot win merely because it makes
no incorrect edits. [S2: Geifman and El-Yaniv, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html).

**Boundary quality needs a dedicated reference metric.** Boundary IoU is
designed to reveal segmentation-boundary errors that ordinary Mask IoU can
underweight. Use it alongside protected-region overlap and reviewed failures;
it is not an aesthetic score or a guarantee of mark preservation.
[S3: Cheng et al., CVPR 2021](https://arxiv.org/abs/2103.16562).

**Calibration can fail after capture conditions change.** Ovadia et al.
benchmark uncertainty under dataset shift and find that familiar post-hoc
methods can fall short. For this proposal, makeup, occlusion, exposure, face
scale, and compression are candidate stress conditions to evaluate explicitly.
The paper does not identify a calibrated model for this Retouch corpus.
[S4: Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).

**Statistical risk control is conditional on a valid study design.** Learn
then Test uses calibration data and multiple-testing procedures to qualify
candidate policies; its selective-classification example evaluates error
conditional on acceptance. Its i.i.d. calibration assumption means neighboring
pixels and repeated burst frames cannot simply be counted as independent
trials. This is a possible later analysis method, not a current guarantee.
[S5: Angelopoulos et al., Learn then Test, v5](https://arxiv.org/html/2110.01052v5).

**A smaller mask does not automatically make every loss monotone.** Conformal
Risk Control establishes expected-loss control for specified bounded,
monotone loss families under its sampling assumptions. Its Section 2.4 shows
why general non-monotone losses need further treatment. A retouch can develop
patchiness after mask erosion; error among accepted regions can also rise
when the denominator changes. Do not claim that its basic theorem directly
controls naturalness or conditional edit harm.
[S6: Angelopoulos et al., Conformal Risk Control, v4](https://arxiv.org/pdf/2208.02814).

Sources were checked on 2026-09-11. The CVF Boundary IoU page was inaccessible;
the authors' arXiv record was used. The OpenReview CRC page was also
inaccessible; the arXiv paper was used. No source assets or model weights were
installed. These papers supply methods and cautions, not Retouch results.

## 5. Proposed experiment, with a narrow first question

**Pilot question:** do existing evidence fields predict human-reviewed P8
support usability better than support size alone?

Start with skin, each eye, each brow, lips, and their relevant surrounds.
Keep per-side labels even where the current descriptor pools both eyes or
brows. A pooled value must not conceal one missing or occluded side.

Collect separate labels for region identity, boundary correctness, visible
occlusion, deliberate makeup/marks, and measurement usability. A visible
made-up lip may be valid for appearance measurement while unsuitable for a
bare-skin measurement. “Skin” does not mean “may be healed.” Make the intended
quantity and operation part of each label; preserve disagreement as unknown.

Compare these predeclared analysis arms:

1. Existing P8 availability decisions and support counts.
2. Support counts plus existing region/boundary margin summaries, offline.
3. The preceding evidence plus repeatability under declared input changes.
4. A simple fitted usability model only if there are enough independent labels
   and a separate fitting split; otherwise stop at descriptive comparisons.

For geometric transformations, map masks back before comparing. Resizing,
compression, exposure, makeup, and lighting do not all have the same expected
effect: a change in measured appearance can be real. Stability is secondary
evidence and can be high for a consistently wrong mask. It cannot replace
reference labels.

Report accepted-region count and rate, unusable-support rate among accepted
regions, usable regions wrongly rejected, unknown-label rate, and results by
predeclared stress condition. Report denominators per subject and shoot as
well as per region. With zero accepted regions, conditional error is undefined,
not zero. Boundary IoU and protected-material overlap remain separate metrics.

P7 should be a second pilot with a different outcome: correctness of target
ownership and protected-material exclusion. It cannot inherit P8 calibration.
Actual edit harm requires paired renders and human review in a later phase;
support accuracy alone cannot certify the rendered result.

## 6. Candidate decisions

| Candidate | Recommendation | Reason |
|---|---|---|
| Offline support qualification using current P8 and parser fields | First research pilot | Builds on present observations and directly addresses a measurement limitation. |
| P7 target-ownership qualification | Second pilot | Useful consumer, but requires body-region ownership labels and its own endpoint. |
| Learned operation-specific usability score | Conditional follow-up | Depends on enough independent labels and demonstrable improvement over simple baselines. |
| Temperature scaling the parser | Challenger only | Requires logits and semantic labels; it cannot calibrate arbitrary P8/Safe Auto scores. |
| Conformal or Learn-then-Test qualification | Later methodological option | Requires a specified loss, sampling assumptions, and sufficient independent calibration units. |
| General confidence slider, parser replacement, new age slider | Not justified by this study | No measured benefit or validated mapping to those product behaviors. |
| Generative skin reconstruction | Separate research topic | Changes source-pixel authority and is not required to answer support usability. |

## 7. Decision gates and completion boundary

Proceed from descriptive evidence to a policy only if the study establishes
an improvement over the baseline at comparable useful coverage, with enough
independent evidence to interpret uncertainty. Set risk tolerance, minimum
useful coverage, and annotation definitions before calibration, not after
viewing favorable results. No numerical production threshold or release date
is selected in this proposal.

Stop policy advancement if confidence merely tracks face size, if stress
conditions remove the apparent gain, if labels cannot distinguish material
from permission, or if the sample is too small. In those cases retain the
observations and improve support/labels within FA-01 or P8.

This research pass completes a proposed scope, current-source rationale,
primary-source comparison, and executable study design. It does not complete
P9 implementation, P8 validation, or a corpus certification. The canonical
P1–P8 roadmap remains unchanged pending topic selection.
