# Proposed Retouch QA validity study — 2026-09-21

Status: documentation-only proposal; **not executed and not authorization to
implement or process photos**. It follows the
[QA validity research](RESEARCH_RETOUCH_QA_VALIDITY_2026_09_21.md) and supports the
existing [face-retouch research roadmap](PLAN_FACE_RETOUCH_ALGORITHM_RESEARCH_EXECUTION_2026_09_05.md).
It does not allocate a new P/FA item.

## 1. Decision the study should support

Can a measurement distinguish harmful face-detail changes from intended tone or
geometry changes, pre-existing source limitations, and delivery transformations?
If not, retain it as an explicitly limited diagnostic. Do not use a cleaner-looking
score as justification for stronger smoothing or an automatic acceptance gate.

All cases and outcomes below are **proposed expectations**, not observed results.
Numerical tolerances, acceptable harm, sample sizes, and default changes require
agreement before a qualification study; this plan does not invent them.

## 2. Freeze the objects being compared

Each future run needs the exact code revision plus dirty patch, runtime/model
versions, recipe and resolved per-face parameters, source content IDs, processing
mode, dimensions, numeric/color domain, stage execution, and output content IDs.
Keep originals unchanged; derived evidence belongs in a separately scoped run.

| Pair | Reference and comparison | Question | Required qualification |
| --- | --- | --- | --- |
| Input preprocessing | Decoded input → post-preprocess/pre-face | What happened before the current QA reference? | Denoise/heal/resize/precision recorded separately |
| Geometry | Pre-reshape → actual geometry-only output | How much did shape/likeness change? | Map/alpha/support provenance and independent human review |
| Face photometry | Geometry-only counterfactual → post-face/pre-global | What did face correction change after controlling geometry? | Same coordinates, frozen reviewed per-face supports |
| Global finish | Post-face/pre-global → post-global/pre-neural | What did global finish add? | Separate intended style change from unrequested harm |
| Later engine stages | Core-QA output → final engine output | Did enabled later stages add damage? | Enabled/executed/skipped stages and scale transforms |
| Encoding fidelity | Intended delivery buffer → decoded saved file | Did resize, profile conversion, or encoding change delivery? | Identical output dimensions/domain or explicit conversion |
| End-to-end preservation | Registered source → decoded saved file | Is the delivered photo acceptable? | Preserve a separate unaligned geometry endpoint |

Do not add/subtract nonlinear scores across pairs to claim stage causality.
Use explicit stage ablations/counterfactuals for attribution. Align only known
sampling/geometry for photometry; do not normalize away a color change being
evaluated or use unconstrained registration that hides an unwanted shape edit.

## 3. Observation and availability contract

Proposed fields, to be designed before any future implementation:

- Run/iteration, detector implementation/version, and threshold/policy version.
- Reference/comparison image IDs and stage IDs; requested-reference versus actual
  fallback-reference identity and reason.
- Support ID/hash, person/face ownership, exclusions, support size, face scale,
  coordinate frame, transform identity, interpolation, numeric/color domain.
- Execution state and measurability state; score/units only when measurable.
- Separate diagnostic values, disposition, reason, and evidence contributing to
  that disposition. Diagnostic-only vectors are not qualified acceptance gates.
- Per-stage executed/skipped/failed coverage and final-delivery coverage.

Use explicit not-run/unavailable/checked outcomes. Missing reference, insufficient
support, low reference signal, unsupported color domain, and alignment failure
are not zero-damage scores. A legitimate measurable zero remains a real value.
Only identical observation identities may merge; preserve observations from
different stage pairs/supports even when the detector names match.

Coverage needs a predefined required-detector set for each mode, with named
exemptions. A nonempty evidence list does not prove complete measurement.

## 4. Discriminating control matrix

Synthetic controls can establish metric mechanics, not real-photo quality.
Later authorized photo cases must use accepted regions and unchanged originals.

| Proposed control | Expected distinction | Shortcut that must not pass |
| --- | --- | --- |
| Identity on textured support | Near-identity change within a justified numeric tolerance | A low absolute texture score interpreted as newly caused harm |
| Identical flat/low-signal inputs | Texture ratio unavailable; identity may still be checked independently | Maximum pore loss or a fictitious valid texture pass |
| Tone-only, no clipping | Record tone change separately; characterize metric sensitivity | Contrast-induced energy change called pore destruction |
| Geometry-only warp/blend | Geometry changes; photometry assessed against actual geometry-only output | Same-coordinate SSIM loss called smoothing |
| Same warp plus localized blur | Additional loss isolated on matched face support | Alignment erases the injected blur |
| Face-local blur; detailed background unchanged | Local loss visible on the affected face | Whole-image energy hides face damage |
| Background-only change, face unchanged | No false face-local damage outside filter-support margins | Person/global content represented as facial skin |
| Protected mark or makeup removal | Independent protected-feature harm is visible | A good average texture/perceptual score excuses deletion |
| Noise added after blur | Energy gain is separated from photographed-detail recovery | Noise restores a “pores intact” result |
| Boundary correction on/off | Leakage and edge halo assessed in/out of intended support | Mask-edge frequencies inflate preservation scores |
| Missing/mismatched reference; empty/tiny support | Explicit unmeasurable reason and no fictitious numeric pass | False confidence from default zeros |
| Multiple people, small harmed face | Per-owner/per-face harm retained, including worst face | Largest-face or whole-frame average hides it |
| Proxy/fast/full modes | Coordinate, support, and metric-scale changes are explicit | Equal dimensions after upscale assumed to mean native detail |
| Resize/JPEG delivery only | Delivery loss identified separately from retouch loss | Pre-encode QA relabelled final-file certification |
| Same detector, different pairs/supports | Two separately identified observations | One runner's score combined with another runner's flag |
| Plastic-skin reference changed, output fixed | Confirm which diagnostics depend on the reference | Claim back-off improved merely because reference loss is present |

For each control, record expected sensitivity/invariance, where it is undefined,
the actual outcome, and a reviewer disposition. Failure should narrow the claim
or keep the metric diagnostic; do not tune the case away and reuse it as locked
qualification evidence.

## 5. Real-photo preservation pilot

Before selecting an algorithm, review the intended task and annotation policy:

1. Inventory photo permissions, existing source hashes/person-group labels, and
   train/development/calibration/qualification membership. Repeated shots of a
   person or event must not silently cross splits. Use metadata/owner review;
   do not introduce biometric identity matching.
2. Label permitted correction support, protected marks/makeup/accessories,
   exclusion boundaries, intended subject ownership, and ambiguous regions.
   Preserve ambiguity and disagreements; an annotation is not certainty.
3. Compare methods on identical accepted supports first. Then vary detection or
   support selection separately. Keep detector quality separate from repair.
4. Review unchanged input, current recipe, and each isolated candidate at native
   crops and matched delivery size. Blind method names and randomize ordering.
   Keep harm review distinct from aesthetic preference.
5. Cover recorded appearance/lighting, makeup, marks, hair/accessory boundaries,
   small/profile/occluded faces, groups, noise/blur, and preview/export modes.
   Missing strata remain unqualified; do not generalize from light-tone-only
   development labels or assign clinical skin categories from images.

Predeclare separate endpoints: support correctness, protected-feature harm,
boundary/owner leakage, texture loss/noise gain, tone/material change,
geometry/likeness, requested benefit, preference, coverage, and runtime/memory.
No single blended quality score should erase a preservation failure.

## 6. Abstention, denominators, and uncertainty

Define the evaluation unit before counting: edit opportunity, face, image, or
independent subject/event group. Publish counts for each; do not treat pixels or
many shots of one person as independent qualification examples.

Report separately:

- Measurable support / eligible opportunities.
- Automatically accepted edits / eligible opportunities, including abstentions
  and manual-review referrals in the denominator.
- Reviewed harmful accepted edits / reviewed accepted edits, plus severity and
  independent-group counts. Explain any review subsampling.
- Beneficial accepted edits / reviewed accepted edits; also report useful edits
  per eligible opportunity so an edit-nothing policy is not labelled successful.
- Skip/review reasons, unavailable measurements, and outcomes by predeclared
  stratum. No zero-denominator percentage should be reported as zero risk.

Selective prediction provides a risk/coverage framing, not a drop-in Retouch
guarantee. A calibration proposal must define the loss and sampling assumptions,
freeze selection thresholds before qualification, and handle clustered data.
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html),
[risk-controlling prediction sets](https://arxiv.org/abs/2101.02703)

An illustrative calculation shows why a small clean pilot is weak evidence.
With zero harms in `n` independent Bernoulli trials and a fixed policy, the exact
one-sided 95% upper bound is `1 - 0.05^(1/n)`: about 25.9% for 10 trials, 4.87%
for 60, and 1.00% for 299. These are mathematical examples, not Retouch results,
recommended sample sizes, or approved risk limits. Correlated faces/photos,
threshold selection on the same data, multiple comparisons, and unrepresented
strata invalidate a naive application. [NIST exact binomial limits](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm)

## 7. Candidate correction-field study, after measurement qualification

Compare unchanged input, the current operation, and a bounded deterministic
correction field on identical accepted supports. Specify working color domain,
amplitude and spatial-frequency constraints, protected-feature exclusions,
boundary feathering, clipping handling, and no-edit fallback before evaluation.
If a learned field is later authorized, add model/license/privacy/runtime checks
and keep the same supports and endpoints. Do not compare a low-resolution learned
preview to a native-resolution baseline and attribute the difference to method.

The additive-map and DISTS literature rationale and access limits are in the
[research report](RESEARCH_RETOUCH_QA_VALIDITY_2026_09_21.md#8-literature-follow-up-bounded-corrections-and-preservation-metrics).
Neither additive composition nor perceptual similarity guarantees preservation
of the photographed pores, marks, makeup, or identity.

## 8. Promotion gates and handoff

| Gate | Evidence needed before proceeding |
| --- | --- |
| Measurement specification accepted | Stage pairs, observation identity, masks/coordinates, availability, units, and permissible claims |
| Diagnostic mechanics credible | Control matrix discriminates intended changes; failures/exemptions recorded |
| Photo pilot interpretable | Reviewed supports and independent harm/benefit assessment; no missing evidence relabelled passed |
| Policy calibrated | Separate grouped calibration data, frozen policy, agreed risk/benefit targets and uncertainty method |
| Qualification accepted | Locked grouped set and delivery review; supported strata stated; owner acceptance |
| Runtime/default change authorized | Explicit implementation scope, focused regression plan, deployment/fallback decision |

Future artifacts should include a frozen manifest, per-observation records,
reviewed support/annotation versions, control-case outcomes, native/delivery
comparisons, blinded review records, and a decision log with limitations. None
were generated in this documentation pass. Current tests, recipes, thresholds,
models, photos, and defaults remain unchanged.
