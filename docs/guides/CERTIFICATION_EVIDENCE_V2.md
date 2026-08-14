# Certification Evidence v2

**Status:** proposed release-evidence contract; documentation only

**Scope:** Core-recipe certification, including render integrity, face-aware
execution, automatic quality, corpus coverage, human review, and final release
state.

This document defines what Retouch must prove before it makes a certification
claim. It does not claim that the v2 writer, validator, review UI, or immutable
finalizer is implemented. Existing v1 manifests remain diagnostic until every
v2 gate below is present.

## Principle

A successful process exit, a non-empty output directory, or
`global_only: false` is not proof of automatic image quality. Certification is
the conjunction of independently recorded gates:

```text
render completed
  AND face-aware evidence valid
  AND automatic quality vector complete and acceptable
  AND corpus manifest complete and validated
  AND human review accepted
  = certified
```

Every gate must be represented by evidence, not inferred from a convenience
boolean. A missing, stale, contradictory, or unverifiable record is a failed
gate. `uncertain` is never silently converted to `approved`.

## Versioning and immutability

The normative v2 package should use a canonical UTF-8 JSON manifest with at
least these top-level fields:

```json
{
  "schema_version": 2,
  "evidence_id": "content-addressed-id",
  "contract_version": "certification-evidence-v2",
  "policy_version": "frozen-policy-id",
  "metric_version": "frozen-metric-id",
  "source_revision": "repository-revision",
  "runtime_doctor": {},
  "corpus": {},
  "renders": [],
  "human_review": {},
  "gates": {},
  "integrity": {}
}
```

The exact filename and storage layout are implementation decisions. The
semantics are not:

- Hash every source asset, rendered output, review payload, model/reference
  artifact, and manifest with SHA-256.
- Hash canonical serialized JSON, with a documented key/order and encoding
  rule. Store the hash beside the referenced artifact and in the parent
  manifest.
- Store the sanitized Runtime Doctor snapshot and its SHA-256 in every
  certification matrix. Do not include secrets, user paths, access tokens, or
  other machine-specific identifiers in the sanitized snapshot.
- Once sealed, an evidence package is read-only. A finalization operation must
  create a new manifest that references the prior hashes; it must not silently
  rewrite historical evidence.
- `certified` must equal the conjunction of all required gate results. Derived
  compatibility fields must be recomputed from the same gate state, so
  `final_certified: true`, pending human review, and a false compatibility
  field cannot coexist.

## Required evidence by claim

| Claim | Required v2 evidence | Gate failure |
| --- | --- | --- |
| Render completed | Decodable output; expected recipe identity and resolved parameter fingerprint; dimensions, channels, dtype, color/profile metadata; output and source hashes; metadata/hash checks | No certification for that render |
| Face-aware | Probed detector backend; successful initialization; expected and detected face counts; per-face landmarks and confidence/bounds; parser backend and ROI provenance; actual model execution provider | The render is global-only or diagnostic, even if full mode was requested |
| Automatic quality | Versioned metric vector containing flagged and non-flagged results, raw details, thresholds, detector availability/failures, and ROI provenance | Automatic quality is unproven and routes to review or fails by policy |
| Corpus complete | Private manifest with unique asset hashes, consent references, validated tags, group identity, and locked pilot/holdout split | Corpus gate fails; filename labels are insufficient |
| Human accepted | Two independent blinded reviews, decisions, confidence, and structured defects; third review for disagreement or critical defects | Pending, rejected, or unresolved cells cannot pass |
| Certified | Immutable final manifest whose hashes and gate results satisfy every preceding row | `certified` must be false |

## 1. Render evidence

Each image/recipe cell must record, at minimum:

- source asset SHA-256 and the private corpus asset identifier;
- exact recipe name, recipe catalog revision, resolved parameters, and a
  parameter fingerprint;
- requested mode and effective mode;
- return status, output relative path, output SHA-256, and decode result;
- output width, height, channel layout, dtype, bit depth, and color profile;
- metadata policy and observed ICC/EXIF/XMP preservation or transformation;
- input/output dimensions before and after any processing resize;
- contact-sheet and comparison-artifact hashes when those artifacts are part
  of the review package.

The output must be decoded and checked, not accepted because a file exists or
has non-zero length. A recipe identity mismatch, stale output, missing
metadata observation, or hash mismatch fails the cell.

Current paths: `scripts/recipes/recipe_sweep.py` writes the v1
`manifest.json` and output rows; `scripts/recipes/core_recipe_certification.py`
consumes that manifest. Those files currently record useful paths, recipes,
shapes, statuses, and QA rows, but do not constitute this complete render
contract.

## 2. Face-aware evidence

Face-aware status must describe what happened to the specific source image. It
must not mean only that full mode was requested.

The per-image record must include:

- the detector probe result, backend name/version, initialization result, and
  failure reason if unavailable;
- declared expected face count and the actual detected face count;
- stable per-run face identifiers with bounding boxes, confidence, landmark
  count, and landmark/mask hashes where applicable;
- parser backend, parser status, and the provenance of each face/skin/body ROI;
- the actual model-session execution provider for the image, not merely a
  static list of providers installed on the machine;
- no-face, partial-detection, occlusion, and multi-face outcomes as explicit
  states.

`face_aware: true` is valid only when the detector initialized and the
per-image observations satisfy the expected-face policy. A detector that was
unavailable and then allowed the engine to continue is global-only evidence.
An explicitly expected no-face case must be labeled as such and cannot be
used to prove face-aware processing.

`retouch/runtime_doctor.py` is a useful runtime snapshot foundation, and
`scripts/qa/face_aware_runtime_probe.py` is a necessary environment/portrait
diagnostic. Neither static runtime information nor a separate probe replaces
the actual face counts and backend observations in every certification render.
`retouch/detection.py` and `retouch/engine.py` remain the implementation
surfaces that must expose and persist those observations.

## 3. Automatic quality vector

The automatic record is a complete, versioned vector. It must retain every
required detector result, including values that did not cross a warning
threshold. A warning-only list cannot be used to calibrate release quality.

Each metric entry should contain:

```json
{
  "metric": "texture_retention",
  "version": "metric-version",
  "value": 0.0,
  "threshold": 0.0,
  "status": "pass",
  "flagged": false,
  "available": true,
  "roi": {
    "source": "face_landmarks|parser|body_mask|global",
    "face_ids": ["face-0"],
    "pixel_count": 1234,
    "mask_hash": "sha256"
  },
  "details": {}
}
```

The vector should cover, where applicable, texture retention, clipping,
halos, seams/boundary bleed, color drift, exposure, naturalness, geometry or
likeness safeguards, and uncertainty. It must also record detector failures,
empty ROIs, unavailable models, and insufficient-resolution conditions rather
than dropping them.

The existing `retouch/qa_detectors.py` exposes scores, thresholds, flags, and
details through `QAWarning`; `retouch/stages.py` carries QA and face state in
`PipelineState`; and `retouch/engine.py` runs the QA stage. The current
`scripts/recipes/recipe_sweep.py` serializes the warning-oriented results it
receives from `result.qa`, but it does not yet guarantee a full non-flagged
vector, ROI provenance, or detector-failure record. The v2 automatic gate must
close that gap.

Thresholds are evidence policy, not universal truth. Fixed heuristic values
must be calibrated on the pilot, versioned, and frozen before the holdout.
When a metric cannot distinguish human-approved from rejected output, it is
diagnostic and the affected cell routes to human review.

## 4. Corpus, consent, and split

The certification corpus remains private unless every asset has separate
distribution permission. Public research datasets such as PIQ23 and PPR10K
may inform methodology, but their images and derived material are not
commercial Retouch certification assets without permission.

The private corpus manifest must record:

- unique source SHA-256, stable asset ID, and shoot/group ID;
- consent reference and permitted-use scope;
- human-validated tags with tag vocabulary/version, validator, and validation
  time;
- camera/source characteristics and relevant quality conditions;
- pilot or holdout split, with no source or shoot-group leakage between splits;
- duplicate checks, including exact hash rejection and review of near-duplicate
  frames where they could bias a result;
- the required strata and the number of evaluable cases in each stratum.

Coverage must be established from this manifest, not inferred from filenames.
The protocol should cover varied skin tones and ages; five lighting
conditions; camera/source variation; face scale and pose; glasses, facial
hair, wigs, makeup, hands/props, body skin, marks, and texture; multiple faces;
compression/noise; high-key whites; and low-key dark detail. At least eight
holdout cases must belong to multi-image shoot groups so consistency can be
measured.

The existing `retouch.certification.core_corpus_coverage()` and
`scripts/recipes/core_recipe_certification.py` use semantic case-name aliases
for a v1 diagnostic. A specially named image can satisfy several aliases, so
that behavior is not v2 corpus validation.

## 5. Human review

Every holdout render/recipe cell requires two independent reviews. Reviewers
must not see the other review, prior scores, recipe identity, or the outcome
being sought. Present source and output at 100% in randomized left/right order
on a calibrated sRGB display; retain the role-assignment key separately from
the review response.

Each review record must include:

- reviewer pseudonymous ID and review timestamp;
- randomized presentation/order and evidence-package hash;
- `approved`, `rejected`, or `uncertain` decision;
- confidence level;
- structured defect labels and optional notes across naturalness,
  texture/mark preservation, exposure/color, geometry/likeness, facial
  boundaries, and group consistency;
- explicit skipped/unavailable state where the cell cannot be evaluated.

A third independent reviewer is required for any disagreement between the
first two reviewers or any critical defect. The adjudication record must point
to all three review hashes. Uncertainty remains unresolved until adjudicated;
it never becomes approval through omission or a default value.

`scripts/recipes/core_recipe_certification.py` currently creates one visible
review row per recipe/case, and
`scripts/recipes/finalize_core_recipe_certification.py` consumes that v1
worksheet. These are not two-review blinded adjudication evidence.

## 6. Pilot and locked holdout protocol

### Pilot: 24 consented images

Use 24 consented images to test detector behavior, corpus tags, ROI
provenance, metric completeness, and review instructions. Use pilot results to
revise metrics and calibrate thresholds. Pilot images and judgments do not
count as locked holdout evidence.

Before opening the holdout, freeze:

- the v2 schema and canonicalization rules;
- recipe catalog and resolved recipe fingerprints;
- detector/parser/model versions and runtime policy;
- metric implementation/version and threshold policy;
- corpus tag vocabulary and split manifest;
- reviewer instructions, defect labels, and adjudication rules.

### Holdout: 48 unseen images

The holdout must contain 48 unseen, consented images, including at least eight
multi-image shoot groups. Run all 12 locked Core recipes: 48 × 12 = 576
holdout renders and 1,152 independent first/second reviewer judgments before
any conditional third reviews.

For every cell:

1. Capture the sanitized Runtime Doctor snapshot/hash and the actual per-image
   detector/parser/provider evidence.
2. Render the locked recipe and validate decode, dimensions, dtype, metadata,
   hashes, and the complete automatic vector.
3. Generate the blinded randomized review presentation.
4. Collect two independent decisions, confidence values, and defect labels.
5. Route disagreement or critical defects to a third reviewer.
6. Aggregate only after the holdout is locked; do not retune thresholds on
   holdout outcomes.

Historical warning saturation (for example, all outputs for the previously
observed DSCF8473, DSCF7585, and DSCF8007 sweeps) is a calibration signal, not
fresh v2 evidence. Those results must not be treated as a release statistic
until the current vector is captured and the run is reproducible.

## 7. Release policy

The following is the initial product policy to freeze before holdout review:

- zero confirmed critical defects;
- at least 90% majority approval per recipe;
- no required stratum below 80% approval when it has five or more evaluable
  cases;
- at most 10% unresolved or uncertain cells;
- no missing, failed, contradictory, stale, or unverifiable technical gate;
- every output and review artifact referenced by the final manifest has the
  expected SHA-256.

Approval percentages must identify their denominator, excluded/unavailable
cells, and all uncertainty. A cell with an unresolved critical defect cannot
be hidden by an aggregate score. These thresholds are release policy and must
not be presented as calibrated results until the pilot and locked holdout are
complete.

## Current implementation map and remaining gates

| Path | Current role | v2 status / remaining work |
| --- | --- | --- |
| `retouch/certification.py` | v1 corpus aliases, artifact checks, and matrix booleans | Add schema validation, unique-hash/tag/split validation, complete-vector checks, review aggregation, and consistent derived gate state |
| `scripts/recipes/core_recipe_certification.py` | Runs the Core matrix and writes `matrix_manifest.json` plus one-review worksheets | Capture per-render evidence and use validated private corpus manifests; do not derive face-aware state from requested mode or `global_only` alone |
| `scripts/recipes/recipe_sweep.py` | Renders recipes, checkpoints v1 rows, and writes warning-oriented QA data | Persist full detector observations, non-flagged metrics, ROI provenance, face evidence, decode/metadata checks, and hashes |
| `retouch/qa_detectors.py`, `retouch/stages.py`, `retouch/engine.py` | Produce QA and pipeline face/ROI state during processing | Define a stable serializable metric vector and explicit failure/empty-ROI semantics |
| `retouch/runtime_doctor.py` | Collects a dependency, model, detector, parser, provider, and effective-mode report | Sanitize and hash a snapshot per matrix; connect it to actual per-image evidence and active provider observations |
| `scripts/qa/face_aware_runtime_probe.py` | Necessary real-portrait detector diagnostic | Keep it as a prerequisite, but do not use it as a substitute for per-render counts and landmarks |
| `scripts/recipes/finalize_core_recipe_certification.py` | Finalizes the v1 worksheet by rewriting v1 matrix fields | Replace with an immutable, hash-linked v2 finalizer that cannot emit contradictory states |
| `tests/test_core_recipe_certification.py`, `tests/test_recipe_sweep.py` | Test the current v1 helper behavior | Add pure tests for canonical hashes, schema completeness, corpus rejection, vector preservation, blinded-review aggregation, uncertainty, and final-manifest consistency |

## Explicit remaining release gates

Certification v2 is not ready to claim until all of these are complete:

1. Recover and verify the pinned face-aware runtime: clean `pip check`, no
   duplicate OpenCV distributions, successful detector initialization, and a
   real portrait render with `global_only: false`.
2. Implement and test the immutable v2 schema, canonical hashing, sanitized
   Runtime Doctor snapshot, and consistent final gate derivation.
3. Build and validate the private consented corpus manifest, including unique
   hashes, human tags, group-aware splits, pilot/holdout separation, and
   required-stratum counts.
4. Persist complete per-render detector, parser, provider, ROI, output, and
   automatic metric evidence, including successful and failed observations.
5. Provide blinded randomized review capture with two independent reviewers,
   confidence and defect labels, and third-review adjudication.
6. Run the 24-image pilot, freeze policy/metric versions and thresholds, then
   run the 48-image holdout without tuning on holdout results.
7. Verify the release thresholds and publish one immutable final manifest whose
   hashes, gate results, and derived compatibility fields agree.

Methodology references are informative only: [NIST FATE face-quality vector
assessment](https://www.nist.gov/publications/face-analysis-technology-evaluation-fate-part-11-face-image-quality-vector-assessment),
[PIQ23](https://openaccess.thecvf.com/content/CVPR2023/html/Chahine_An_Image_Quality_Assessment_Dataset_for_Portraits_CVPR_2023_paper.html),
[AutoRetouch](https://openaccess.thecvf.com/content/WACV2021/papers/Shafaei_AutoRetouch_Automatic_Professional_Face_Retouching_WACV_2021_paper.pdf),
and [PPR10K](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html).
Their images remain subject to their respective licenses and are not bundled
by this guide.
