# Proposed P9 research protocol — Support usability and selective decisions

Date: 2026-09-11.
Status: proposed protocol; topic not confirmed; experiments not run.
Baseline: `cabf3bc7c246c1662779c3f517a7cb89fb33b4a5`.
Research basis: [P9 proposal and sources](RESEARCH_P9_SELECTIVE_RETOUCH_PROPOSAL_2026_09_11.md).

## 1. First deliverable

An offline P8 support-usability report comparing existing availability rules
with candidate evidence rankings. It must show where those signals agree with
human-reviewed regions and where they do not. The output is an evidence report,
not an edit-strength controller.

Reuse [P8 R0/R1](PLAN_P8_CUE_VALIDITY_RESEARCH_2026_09_10.md), the
[FA-01 evidence contract](RESEARCH_FA01_ONTOLOGY_AND_EVIDENCE_2026_09_05.md),
and corpus v3. Do not duplicate already usable annotations or invent a second
material taxonomy. Study-specific records can initially be sidecar metadata;
a production schema migration is not a prerequisite.

## 2. Stages and exit evidence

| Stage | Proposed work | Exit evidence | Current state |
|---|---|---|---|
| P9-R0 | Establish scope, annotation rubric, provenance, and independent sampling units | Versioned protocol; usable development inventory; gaps and planned analysis | Protocol drafted; inventory and topic confirmation pending |
| P9-R1 | Qualify P8 support measurements on reviewed development regions | Raw rows, failure examples, score-versus-error and coverage tables | Not run |
| P9-R2 | Fit/qualify a small number of candidate acceptance policies, if R1 supports it | Frozen candidates; independent calibration analysis with uncertainty | Not run |
| P9-R3 | Evaluate a frozen candidate on reserved subjects/shoots | Final support-usability and coverage report; stress-condition results | Not run |
| P9-R4 | Separate P7 ownership study; later, operation-specific render study | Ownership labels first; paired preservation/benefit evidence before any render integration | Not run |

Any implementation or photo experiment is future work beyond this research
pass. Completion of R1 must not be reported as calibration, age validation,
or evidence that a rendered edit is safe.

## 3. Corpus and annotation protocol

Use the current manifest's hashes, explicit person IDs, group IDs, validation
provenance, and split checks. Audit actual metadata before citing coverage.
Previously inspected/tuned images remain development material. Keep subjects
and linked shoots/groups together; document connected grouping when multiple
people appear in one shoot. Do not retroactively label familiar images as an
untouched test set.

For the first P8 pilot, record face location and verified target, individual
eye/brow/lip regions, surrounding skin, and occlusion/intentional-mark support.
For each measurement region, review:

- Is this the intended subject and anatomical/material region?
- Does the support or surrounding ring include excluded material?
- Is the visible region usable for this explicitly named descriptor?
- Is the answer clear, disputed, or unreviewable?

Use independent review on a subset and retain disagreement. For P7 later,
annotate exposed skin belonging to that subject separately from skin belonging
to another person, garments, props, and uncertain boundaries. Human-reviewed
ownership does not by itself validate the downstream tone correction.

Predeclare coverage strata: observed image tone/exposure, lighting, face scale,
pose, makeup, facial hair, glasses, hand/prop occlusion, multi-person scenes,
and compression/noise. Report absent strata. Do not infer physiological skin
types, ethnicity, age, or a person's identity from appearance for these labels.

## 4. Proposed evidence rows

Each offline row should preserve these fields or an equivalent lossless form:

| Group | Fields |
|---|---|
| Provenance | source hash, subject/group/split, face location, code/descriptor/parser versions, input dimensions and transformation |
| Observation | descriptor/operation, per-side region, raw score and its source, feature/surround counts, boundary summary, missing-data reason |
| Annotation | reference version, reviewer provenance, region correctness, excluded-material overlap, descriptor usability, disagreement/unknown |
| Candidate decision | candidate ID/version, accept/review/reject, explicit reason, declared fitted or unfitted status |
| Evaluation | accepted status, labeled error, usable coverage, boundary metric, stress stratum, timing if new computation was performed |

Keep P8 support counts, parser margins, and Safe Auto confidence as separately
named evidence. Do not multiply or average them into a probability without a
specified fitted model and held-out evaluation. Missing model evidence is
missing, not proof of either correctness or failure.

## 5. Analysis and acceptance

Freeze the candidate score definitions on development data. If a usability
model is fitted, fit it there; reserve calibration subjects for qualifying
policies. Predeclare a small candidate family, sampling unit, error tolerance,
minimum useful coverage, and uncertainty procedure before qualification.

Plot or tabulate conditional support error against coverage. Compare with
the existing P8 rule and a support-count-only ranking. Include unknown-label
rates and a sensitivity analysis treating accepted unknowns as failures, so
excluding difficult annotations cannot manufacture a good result. Report
numerators and denominators; an empty accepted set has undefined conditional
error and fails the useful-coverage gate.

Treat subject/shoot dependence explicitly. Region- or pixel-level counts may
describe errors but are not independent sample sizes. Clustered descriptive
intervals do not automatically satisfy the independent-unit assumptions of
formal risk-control methods. Any later guarantee needs an analysis matched
to the actual sampling design. Candidate selection on calibration data must
also account for testing several thresholds; see the research report's
Learn then Test discussion and source.

Stress transformations test declared sensitivity, not all-purpose invariance.
Distinguish fixed reviewed-support measurements from rerunning the parser:
the first isolates descriptor response; the second includes support drift.
Geometric comparisons require inverse alignment. Human review remains the
reference because a stable wrong region can survive every transformation.

R1 passes as a diagnostic study when its rows, references, missingness, and
failure examples are reproducible. R2/R3 require a useful improvement over
baseline within predeclared limits; otherwise report no qualified policy.
An apparent gain from rejecting almost all inputs or only serving easy strata
does not qualify for broad deployment.

## 6. Implementation boundary if the proposal proceeds

The initial harness should reuse `measure_p8_cues`, parser observations, and
corpus validation. Keep it in the QA/research path. Preserve the existing
parser-confidence render guard. Avoid fitting policy parameters inside
`RetouchEngine.process` or using measurement confidence to enable P7.

P7 ownership needs its own labels and qualification; do not reuse a P8
threshold. A future render pilot should compare source, current behavior,
and a candidate policy at fixed requested strength. Record wrong-material
edits, protected-feature damage, seams/halos, useful correction, no-op/review
cases, runtime, and native-resolution human judgments. A mask-only success
cannot close this render gate.

If production integration is later justified, use the existing Safe Auto
representation where applicable, preserve opt-in/default behavior contracts,
and define policy versioning and rollback. No second decision engine is
proposed by this protocol.

## 7. Work completed in this pass

Completed: source inventory, six-source method comparison, provisional P9
scope, staged protocol, and documentation links. Validation is limited to
documentation consistency and local link/whitespace checks. No Python tests,
new images, image annotations, rendered outputs, models, thresholds, or
runtime measurements were produced.
