# AMG Day 3 2026 — Full v2 Engineering Implementation Specification

**Date:** 2026-09-01
**Status:** PROPOSED — documentation approved for planning; implementation has not
started.

**Authority boundary:** documentation only. This plan does not authorize changes to
engine code, tests, models, source photographs, existing outputs, or delivery folders.

**Research basis:**
[`REVIEW_AMGDAY32026_RETOUCH_BATCH_2026_09_01.md`](../review/REVIEW_AMGDAY32026_RETOUCH_BATCH_2026_09_01.md)

**Certification contract:**
[`CERTIFICATION_EVIDENCE_V2.md`](../guides/CERTIFICATION_EVIDENCE_V2.md)

**Target recipe:** `cosplay_portrait_polish_v1`

**Target source set:** 22 files, `DSCF2385.jpg` through `DSCF2406.jpg`

**Current accepted comparison batch:** `/Users/dennis/Desktop/amgday32026_retouched`

**Proposed sibling output:** `/Users/dennis/Desktop/amgday32026_retouched_full_v2`

---

## 0. Decision record

The implementation will be planned before code and split into independently reviewable
changes. A full-quality rerender is the final consumer of the work, not the first step.

The required order is:

```text
freeze baseline and contracts
  -> exact-support lip/teeth containment
  -> deterministic colour-delivery primitives
  -> complete QA evidence and atomic/resumable runner
  -> labelled mouth corpus and frozen calibration
  -> tongue-safe lip-support gate
  -> native 24 MP three-anchor pilot
  -> non-destructive 22-image full_v2 batch
  -> independent visual acceptance and evidence sealing
```

No phase may claim a later phase's completion. In particular:

- a passing unit test is not a real-render approval;
- a successful render is not complete QA evidence;
- 22 decodable files are not a certified batch;
- bounded 6.24 MP crop results are not a native 24 MP performance result;
- an anatomical ring is not tongue detection; and
- provisional thresholds are not allowed into the locked holdout.

## 1. Goals

### 1.1 Product outcome

Produce a new, non-destructive, native-resolution version of the AMG Day 3 batch that:

- runs the requested recipe through the engine's real `quality="full"` path;
- does not apply lip enhancement to tongue, teeth, or mouth-interior pixels;
- keeps pixels outside lip/teeth operation support exactly unchanged by those operations;
- uses explicit colour-managed ingest and export;
- produces 8-bit JPEG Q95 with a reviewed ICC profile and 4:4:4 chroma;
- normalizes orientation and applies a declared metadata policy;
- writes files atomically and resumes safely after interruption;
- records complete per-file render, face, parser, provider, QA, metadata, and integrity
  evidence; and
- preserves sources and the accepted draft output without modification.

### 1.2 Engineering outcome

Create reusable implementation surfaces rather than another event-specific OpenCV loop:

- one exact-support utility contract used by lip and teeth operations;
- one mouth-safety decision contract that is independent of recipe strength;
- one canonical Retouch sRGB ICC asset and hash;
- one reusable delivery-manifest writer/validator;
- one production batch entry point with unambiguous quality controls; and
- focused tests that prove safety through mutation and failure injection.

## 2. Non-goals

This implementation does not include:

- a neural tongue segmenter or new face-parsing model;
- colour-based “pink/red equals tongue” classification;
- automatic threshold selection from the 22-image event sequence;
- changes to eye retouching, under-eye behavior, yaw/reshape calibration, or unrelated
  recipes;
- automatic overwrite of an existing output;
- blanket preservation of every XMP, IPTC, or Photoshop metadata block;
- a true 16-bit processed output claim; the public engine result is currently 8-bit;
- multiple concurrent 24 MP renders before one-worker memory evidence exists;
- print-lab certification or browser-wide colour consistency claims; or
- certification from a warning-only QA list.

## 3. Established implementation facts

These are inputs to the design, not items to rediscover during coding:

1. `ProcessingContext.quality` defaults to `full`; `draft` is the proxy face-processing
   path. CLI `-q/--quality` currently controls output compression, not engine quality.
2. `--max-dim 2048` pre-shrinks the whole image and upscales later. It is proofing, not
   native full quality.
3. The current AMG runner uses `cv2.imread()`, `quality="draft"`, and `cv2.imwrite()`.
4. `write_image_with_color_context()` already supplies explicit working-profile output,
   EXIF handling, JPEG 4:4:4, and optional C2PA/JUMBF passthrough.
5. `BatchProcessor._process_single_file()` already demonstrates temporary-file output
   and `os.replace()`.
6. `recipe_sweep.py` already demonstrates resumable manifest checkpointing and rich
   detector/parser/provider diagnostics, but it is not a production folder renderer.
7. The current LittleCMS-generated working profile changes bytes between processes; a
   frozen asset is required for deterministic hashes.
8. BiSeNet supplies semantic lips and mouth interior but no tongue class. MediaPipe
   supplies outer/inner lip landmarks but no tongue landmark.
9. The 105-image study found severe semantic/landmark geometry conflicts. A universal
   landmark intersection is unsafe.
10. `_reapply_texture()`, `_add_lip_gloss()`, and `TeethWhitener.whiten()` perform
    full-canvas LAB roundtrips. Exact restoration outside effective support eliminates
    the observed drift without changing inside-support results.
11. Draft QA currently passes a proxy-sized reshape displacement field to native-sized
    AA6 evaluation. The detector fails explicitly, so draft automatic QA is incomplete.
12. Full-quality rendering improves the two inspected crop anchors but does not repair
    tongue masking.

## 4. Proposed architecture

### 4.1 Delivery flow

```text
source discovery
  -> immutable-source preflight and SHA-256
  -> EXIF-aware, ICC-aware decode into Retouch sRGB working space
  -> full-quality face-aware engine processing
  -> complete render/face/component/QA evidence collection
  -> managed JPEG encode into a same-directory temporary path
  -> decode, dimensions, profile, EXIF, subsampling, and hash validation
  -> atomic replacement into the new sibling output directory
  -> atomic manifest checkpoint
  -> A/B artifact generation after all primary outputs validate
```

Comparison sheets are derived review artifacts. They must never be accepted as proof
that the primary output itself passed decode, metadata, or integrity checks.

### 4.2 Component ownership

| Concern | Proposed implementation surface | Ownership rule |
| --- | --- | --- |
| Exact support | `retouch/lips.py`, `retouch/teeth.py`, optional small helper in `retouch/utils.py` | Mechanical containment only; no morphology policy |
| Mouth decision | New `retouch/mouth_safety.py` | Pure ROI-local mask/geometry decision; no image writing or recipe resolution |
| Pipeline wiring | `retouch/perf_optimizations.py` and evidence aggregation in `retouch/engine.py` | Compute once per face; all lip consumers use the same decision |
| Parser inputs | Existing `FaceRegions.lips`, `FaceRegions.mouth_interior`, `LIPS_OUTER`, `LIPS_INNER` | Do not reinterpret BiSeNet labels as tongue evidence |
| Colour profile | New reviewed asset, for example `retouch/assets/icc/retouch_srgb_v1.icc`; `retouch/io.py` loader | Exact bytes and SHA-256 are versioned |
| Delivery manifest | New `retouch/delivery_manifest.py` | Canonical serialization, validation, hashes, state transitions |
| Batch entry point | New `scripts/batch/full_v2_batch.py` | Thin orchestration over library functions; no duplicated colour science |
| QA completeness | `retouch/engine.py`, `retouch/qa_detectors.py`, manifest adapter | Every required metric is recorded, including passes and failures |
| Tests | Existing focused files plus new runner/mouth/manifest suites | Synthetic first, then real anchors and mutation |
| Documentation | Review, this plan, CLI/API/recovery guides after implementation | Documentation status must track real implementation state |

### 4.3 Public versus internal API

The first implementation should keep the mouth gate internal to per-face processing.
Do not add user-facing morphology sliders before calibration.

The evidence object should be public enough for the delivery runner to serialize without
parsing log text. Proposed `ProcessingResult` addition:

```text
component_evidence: list[dict]
```

Each per-face mouth decision contributes one JSON-safe record. This must remain distinct
from `safe_auto_decisions`; tongue containment is a component safety contract, not an
inferred automatic beauty adjustment.

## 5. Slice A — Exact-support lip and teeth containment

### 5.1 Required behavior

For each operation:

```text
original = input canvas
support  = final effective operation support
edited   = existing color-space operation
output   = edited where support > 0, otherwise original exactly
```

“Exactly” means:

- uint8: byte equality outside support;
- float32: array equality outside support, not only tolerance-based closeness;
- shape and dtype unchanged; and
- no newly non-finite values.

### 5.2 Operation-specific support

| Operation | Effective support definition | Additional rule |
| --- | --- | --- |
| Lip texture reapply | normalized lip mask after all upstream lip-safety decisions | Texture amplitude may be zero inside support; that does not expand support |
| Lip gloss | blurred detected specular mask multiplied by permitted lip support | Re-clip after Gaussian blur so feathering cannot grow beyond the permitted lip region |
| Teeth whitening | detected teeth mask multiplied by mouth mask and non-zero strength | Mouth interior alone is not permission to change non-teeth pixels |

Support uses exact zeros as the containment boundary. Do not invent a global epsilon that
silently erodes legitimate feather tails. If an epsilon is needed for numerical cleanup,
it becomes a calibrated/versioned constant with explicit tests.

### 5.3 Preferred implementation form

Use one small helper only if it makes all three call sites clearer:

```text
restore_outside_support(original, processed, support) -> same dtype/shape array
```

The helper validates array/mask dimensions and copies original pixels where support is
zero. It must not blur, normalize, threshold, or otherwise own mask policy.

### 5.4 Tests

Add or extend:

- `tests/test_lips.py`;
- `tests/test_teeth.py`;
- `tests/test_lips_hair_body_float32.py`; and
- the golden face and real-operation isolation suites.

Required tests:

1. zero strength and zero mask remain exact no-ops;
2. nonzero uint8 support changes the intended region only;
3. nonzero float32 support changes the intended region only;
4. gloss blur cannot cross permitted lip support;
5. teeth detection cannot change mouth void, tongue-like saturated pixels, or canvas
   outside detected teeth;
6. legacy inside-support output is preserved for a frozen synthetic case;
7. operation-order integration keeps unrelated eye/hair/background pixels unchanged; and
8. a mutation removing the final restore fails the containment tests.

### 5.5 Completion boundary

Slice A is complete only when outside-support delta is exactly zero for all three
operations and both dtypes, inside-support behavior has explicit snapshot review, and a
real lip/teeth isolation render passes at 100%. Reduced drift is not completion.

## 6. Slice B — Deterministic colour and metadata primitives

### 6.1 Canonical working profile

Replace per-process profile generation as the delivery identity with one reviewed,
versioned ICC asset.

Required asset record:

```json
{
  "profile_id": "retouch_srgb_v1",
  "relative_path": "retouch/assets/icc/retouch_srgb_v1.icc",
  "sha256": "frozen after asset approval",
  "color_space": "RGB",
  "profile_connection_space": "XYZ",
  "rendering_intent": "declared after inspection",
  "license_provenance": "recorded before commit"
}
```

Asset selection remains an owner decision. Do not commit arbitrary bytes generated by a
developer machine without inspecting profile tags and license/provenance.

### 6.2 Metadata policy

The runner takes a named policy rather than a vague `preserve_metadata=True` boolean.

Proposed first policy:

```text
delivery_metadata_v1
  keep: normalized orientation, resolution, color-space EXIF, approved camera fields
  keep when supported: source C2PA/JUMBF bytes as passthrough evidence
  do not promise: arbitrary XMP, IPTC, Photoshop blocks
  strip or redact: owner-configured location, serial, and private fields
```

The exact keep/redact list must be approved before implementation. The manifest records
requested policy, observed source groups, retained groups, removed groups, and warnings.

### 6.3 Tests

- cross-process ICC SHA-256 is identical;
- two writes of identical pixels/settings produce the same profile bytes;
- tagged non-sRGB input converts into working space and exports with the working profile;
- untagged input is explicitly recorded as assumed sRGB;
- JPEG is 8-bit 4:4:4 at the selected quality;
- orientation is applied once and stored as 1;
- metadata policy records every intentional retention/removal;
- corrupt or unsupported ICC input fails explicitly rather than silently relabelling
  pixels; and
- C2PA passthrough is described as passthrough, never as a newly signed transformation.

### 6.4 Completion boundary

Slice B is complete when the canonical asset is reviewed and hashed, cross-process tests
pass, DSCF2385 reopens within the existing colour tolerance, and ExifTool confirms the
declared output profile, EXIF policy, dimensions, bit depth, and 4:4:4 subsampling.

## 7. Slice C — Complete QA evidence, including AA6

### 7.1 Draft displacement-field mismatch

The draft path evaluates QA at native resolution after proxy face processing. Its reshape
field is still proxy-sized.

Before implementing a fix, confirm the reshaper field's unit contract. If displacement
components are proxy pixel units, the correct native conversion is conceptually:

```text
native_field = spatially resize(proxy_field, native_width, native_height)
native_field.x *= native_width  / proxy_width
native_field.y *= native_height / proxy_height
```

Do not resize vectors spatially without scaling their magnitudes. Test anisotropic image
dimensions; a single scalar is incorrect when x/y scales differ.

If trustworthy conversion evidence is unavailable, record AA6 as explicitly unavailable
and fail the certification gate. Never replace an unknown field with zeros and report a
pass.

### 7.2 Full QA vector contract

The engine/adapter must expose every required detector observation:

```json
{
  "metric": "perceived_retouching",
  "version": "aa6-v1",
  "available": true,
  "flagged": false,
  "value": {},
  "threshold": {},
  "roi": {
    "source": "face_parser_and_landmarks",
    "face_ids": ["face-0"],
    "pixel_count": 0,
    "mask_sha256": "..."
  },
  "details": {}
}
```

Passes, failures, empty ROIs, missing models, and exceptions are all serialized. The
warning convenience list may remain for UI compatibility but cannot be the evidence
source of truth.

### 7.3 Tests

- draft/no-reshape and draft/active-reshape;
- full/no-reshape and full/active-reshape;
- anisotropic proxy-to-native vector scaling;
- no face, one face, multiple faces, and partial detection;
- missing parser/provider and detector initialization failure;
- a mutation that removes or mis-scales the field fails a named AA6 test; and
- `fail_on_incomplete_qa` rejects any unavailable required detector.

### 7.4 Completion boundary

Slice C is complete only when no required QA result disappears, proxy/full inputs satisfy
the same shape/unit contract, and certification fails on intentionally removed evidence.

## 8. Slice D — Atomic, resumable full_v2 runner

### 8.1 Proposed command

The runner name is specific enough to communicate policy but reusable for another event:

```text
.venv/bin/python scripts/batch/full_v2_batch.py \
  /Users/dennis/Desktop/amgday32026 \
  --output /Users/dennis/Desktop/amgday32026_retouched_full_v2 \
  --recipe cosplay_portrait_polish_v1 \
  --engine-quality full \
  --jpeg-quality 95 \
  --metadata-policy delivery_metadata_v1 \
  --workers 1 \
  --expected-faces manifest \
  --resume \
  --fail-on-incomplete-qa
```

This is an interface proposal, not an implemented command.

### 8.2 Argument contract

| Argument | Required behavior |
| --- | --- |
| `input` | Resolve once; source root must exist and must not equal or contain output |
| `--output` | Must be a distinct sibling/new directory; existing finals require valid resume evidence |
| `--recipe` | Resolve before processing and store resolved-parameter fingerprint |
| `--engine-quality` | Explicit `full` or `draft`; never overloaded with JPEG quality |
| `--jpeg-quality` | Encoder quality only |
| `--metadata-policy` | Named, versioned retention/redaction contract |
| `--workers` | Default 1 for full; values above 1 require explicit resource approval/evidence |
| `--expected-faces` | Per-source expectation manifest, not a universal inferred count |
| `--resume` | Skip only a fully validated row whose source/output hashes still match |
| `--fail-on-incomplete-qa` | Fail a row if any required metric is unavailable or absent |

There should be no convenient `--force` in the first release. An invalid or mismatched
existing output is a preflight error requiring an operator decision, not silent
replacement.

### 8.3 Per-file state machine

```text
pending
  -> processing
  -> rendered
  -> temp_written
  -> temp_validated
  -> final_replaced
  -> manifest_validated
  -> done

any state -> failed_retryable | failed_terminal
```

Only `done` is resumable as complete. A prior `processing`, `rendered`, or
`temp_written` row is incomplete even if a file happens to exist.

### 8.4 Atomic write protocol

1. Validate source/output containment and collision rules.
2. Hash and decode source before rendering.
3. Create a uniquely named temporary file in the final output directory.
4. Render once with the resolved recipe and explicit engine quality.
5. Encode through the managed colour writer.
6. Close and sync the temporary file.
7. Reopen it and validate decode, dimensions, channels, bit depth, ICC, EXIF policy,
   chroma subsampling, and non-empty content.
8. Compute the temporary output SHA-256.
9. Atomically `os.replace()` the temporary file into its final new path.
10. Sync the containing directory where supported.
11. Atomically checkpoint the manifest row as `done`.

If validation fails, remove only the exact temporary file created by this job. Never
delete an existing final, source, output directory, or unmatched stale file.

### 8.5 Resume protocol

A row may be skipped only if all are true:

- schema/evidence version matches;
- source relative path and SHA-256 match;
- recipe name and resolved fingerprint match;
- source revision, model hashes, runtime policy, and canonical ICC hash match;
- requested/effective engine quality match;
- final relative path exists and decodes;
- output SHA-256, dimensions, metadata observations, and face/QA gates recompute; and
- row state is `done`.

Otherwise stop with a mismatch report. Do not overwrite or silently rerender.

### 8.6 Process exit contract

Proposed stable exit classes:

| Exit | Meaning |
| ---: | --- |
| 0 | All requested rows validated and completed |
| 2 | Preflight/configuration/collision error; no render should start |
| 3 | One or more render failures |
| 4 | Output decode/metadata/integrity failure |
| 5 | Face-aware or QA evidence incomplete/failed |
| 6 | Resume manifest contradiction or corruption |

Exact numeric values may align with existing CLI conventions during implementation, but
the classes and operator meanings must remain distinct.

## 9. Manifest specification

### 9.1 Top-level record

```json
{
  "schema_version": 2,
  "evidence_id": "content-addressed after sealing",
  "job_id": "operator-visible unique id",
  "created_at": "UTC timestamp",
  "source_revision": "git revision plus dirty-worktree observation",
  "recipe": {
    "name": "cosplay_portrait_polish_v1",
    "catalog_revision": "...",
    "resolved_params": {},
    "fingerprint": "sha256"
  },
  "runtime_doctor": {},
  "color_contract": {},
  "metadata_policy": {},
  "corpus": {},
  "renders": [],
  "review": {},
  "gates": {},
  "integrity": {}
}
```

### 9.2 Per-render record

Each row must include:

- stable asset ID, source relative path, source SHA-256, source dimensions/dtype/profile;
- expected face count and reason/source of expectation;
- requested and effective engine quality;
- exact recipe fingerprint and explicit overrides;
- start/end timestamps, wall time, peak RSS, process/worker identity;
- detector backend/version/status, parser backend/version/status, actual execution
  provider, face count, stable face IDs, bounds, confidence, landmark count, and ROI/mask
  hashes;
- mouth-safety decision and metrics for every detected face;
- full automatic QA vector with pass/fail/unavailable state;
- temporary/final write states, output relative path, output SHA-256, decode result,
  dimensions, channels, dtype, bit depth, ICC ID/hash, EXIF observations, chroma
  subsampling, and metadata-policy result;
- warnings and structured failure reason; and
- source-unchanged verification.

### 9.3 Canonicalization and integrity

- Serialize UTF-8 JSON with a documented key ordering and number representation.
- Hash artifacts as bytes and masks as canonical contiguous arrays with dtype/shape in
  the hash preimage.
- Checkpoint through a temporary manifest and atomic replacement.
- Final sealing creates a new immutable record referencing prior hashes; it does not
  rewrite prior evidence.
- `certified` is computed from gates. It is never a manually set convenience boolean.

## 10. Slice E — Mouth calibration corpus and tooling

### 10.1 Engineering deliverables before thresholds

Corpus tooling may be implemented before the tongue gate because it does not choose
production behavior.

Proposed artifacts:

- private corpus manifest schema;
- annotation instructions and versioned label vocabulary;
- source/group duplicate validator;
- ROI export tool that never modifies sources;
- geometry metric extractor;
- pilot/holdout group splitter; and
- calibration report generator with no production threshold side effect.

### 10.2 Label contract

Required annotations:

```text
visible_vermilion
overdrawn_lipstick
mouth_interior
visible_teeth
visible_tongue
occluded_or_uncertain
geometry_landmarks_valid
semantic_lip_mask_valid
```

Each annotation records asset ID, face ID, labeler ID, vocabulary version, timestamp,
mask/polygon hash, confidence, and adjudication link.

### 10.3 Split contract

- split by subject and shoot group;
- keep near-duplicate bursts in one split;
- pilot may tune thresholds/morphology;
- holdout remains unopened until implementation and thresholds are frozen; and
- no filename-derived label can satisfy a coverage requirement.

### 10.4 Completion boundary

Tooling is complete when manifests validate, duplicates/group leakage are rejected, two
labeler records and adjudication are supported, and a frozen holdout can be proven
unopened. The tongue implementation remains blocked until adequate labelled strata
exist.

## 11. Slice F — Tongue-safe lip-support decision

### 11.1 Proposed pure decision object

```text
LipSupportDecision
  mask: float32 HxW
  action: semantic | anatomical_ring | bypass
  available: bool
  reason: stable machine-readable enum
  calibration_version: string
  metrics:
    mouth_open_ratio
    semantic_outer_iou
    semantic_inside_inner_ratio
    semantic_outside_outer_ratio
    support_retention
    polygon_validity
  evidence:
    semantic_mask_hash
    outer_mask_hash
    inner_mask_hash
    output_mask_hash
```

The function is pure: same masks, landmarks, face scale, and calibration produce the
same decision and output mask.

### 11.2 Decision table

| Condition | Action | Pixel permission |
| --- | --- | --- |
| Lip enhancement disabled or semantic mask empty | `bypass` | No lip-operation pixels |
| Closed/stable mouth with valid semantic support | `semantic` | Existing semantic lips |
| Open mouth with valid landmarks and calibrated semantic/geometry agreement | `anatomical_ring` | Semantic lips intersected with outer contour and excluding calibrated inner contour |
| Malformed/degenerate landmarks | `bypass` | Preserve original pixels |
| Severe semantic/landmark conflict, profile failure, or uncertain evidence | `bypass` | Preserve original pixels |
| Parser unavailable with only landmark fallback | Policy decided by calibrated fallback arm; otherwise `bypass` | Never silently assume equivalence to BiSeNet arm |

“Fail open” is not used in code or tests because it is ambiguous. The defined behavior is
`bypass_and_preserve_original`.

### 11.3 Pipeline placement

Compute the decision once near the beginning of `_process_face_core()`, after regions and
shifted ROI-local landmarks are available and before `acc_lips` is finalized.

All downstream consumers must use the same safe support:

- `LipEnhancer.enhance()`;
- accumulated lip mask used by native/draft compositing;
- selective sharpening/exclusion behavior that consumes `acc_lips`;
- manifest/component evidence; and
- isolation/debug output.

Do not alter `regions.lips` in place if other diagnostics need the raw parser evidence.
Keep `semantic_lips` and `effective_lips` separately named.

### 11.4 Calibration registry

Thresholds and morphology live in one versioned configuration object, not scattered
literals:

```text
MouthSafetyCalibrationV1
  open_enter
  open_exit                 # optional hysteresis for video/future reuse
  min_semantic_outer_iou
  max_semantic_inside_inner
  min_support_retention
  max_support_retention
  outer_feather_ied_fraction
  inner_erode_or_dilate_ied_fraction
  numerical_cleanup_epsilon
```

This document deliberately supplies no numeric values. They must come from the labelled
pilot and be frozen before holdout.

### 11.5 Tests

#### Pure unit tests

- polygon validity and inner-inside-outer consistency;
- mask normalization and shape mismatch;
- closed semantic decision;
- open reliable ring decision;
- malformed/conflicting bypass;
- deterministic hashes/decision serialization; and
- thresholds loaded only from the versioned calibration.

#### Real-image isolation tests

- DSCF2397–DSCF2401 tongue anchors;
- ordinary closed-mouth controls;
- visible-teeth open smile;
- overdrawn lipstick;
- DSCF4560 profile geometry conflict;
- blur/small-face/coloured-light cases;
- parser fallback arm; and
- multi-face images with independent decisions.

#### Mutation tests

Each must fail at least one named acceptance test:

- remove inner-mouth exclusion;
- force semantic action on all open mouths;
- invert the geometry-agreement comparison;
- skip polygon-validity checks;
- force profile conflict to ring;
- replace bypass with semantic fallback;
- use raw semantic mask for `acc_lips` while enhancer uses safe mask; and
- drop component evidence serialization.

### 11.6 Completion boundary

Slice F is complete only after pilot calibration is frozen and the locked holdout proves:

- tongue pixels unchanged by lip-only isolation;
- teeth and mouth interior unchanged;
- normal lips still receive the intended enhancement;
- lipstick boundaries are not clipped unnaturally;
- profile/uncertain cases bypass safely;
- both parser arms meet their own declared policy; and
- required mutations fail.

## 12. Implementation and review sequence

No implementation change should mix unrelated risk classes.

### Change 0 — Documentation approval

Files:

- this plan;
- research review link; and
- documentation indexes.

Exit: owner agrees that code may begin and identifies which slice is authorized first.

### Change 1 — Exact-support containment

Files expected:

- `retouch/lips.py`;
- `retouch/teeth.py`;
- optional helper location;
- focused tests; and
- golden snapshot evidence only if behavior intentionally changes.

Rollback: revert this isolated change. No runner, ICC, parser, or threshold dependency.

### Change 2 — Canonical ICC and metadata policy primitives

Files expected:

- reviewed ICC asset and provenance record;
- `retouch/io.py`;
- color-context/ICC tests; and
- metadata-policy documentation.

Rollback: return to generated working profile without touching render algorithms. No
full_v2 delivery should proceed while determinism is regressed.

### Change 3 — QA vector and AA6 repair

Files expected:

- engine/QA evidence plumbing;
- proxy displacement conversion after unit confirmation;
- QA tests; and
- certification-evidence adapter tests.

Rollback: certification remains unavailable; rendering itself may still be diagnostic.

### Change 4 — Manifest library and batch runner

Files expected:

- new delivery-manifest module;
- new full_v2 batch entry point;
- atomic/resume/failure tests;
- CLI/API/recovery documentation; and
- no event-source paths hardcoded in library code.

Rollback: no production folder run; retain test artifacts only.

### Change 5 — Corpus tooling

Files expected:

- private-manifest validator/tooling;
- label vocabulary/instructions;
- split/duplicate tests; and
- no private images committed.

Rollback: thresholds remain blocked; no engine behavior changes.

### Change 6 — Calibrated mouth-safety gate

Files expected:

- new mouth-safety module;
- per-face wiring/evidence aggregation;
- frozen calibration record;
- unit, mutation, isolation, parser-arm, and golden tests; and
- human-reviewed pilot report.

Rollback: revert gate as one behavior change. Do not compensate by adjusting recipe
strength.

### Operational step 7 — Three-anchor 24 MP pilot

This is a controlled run, not a code commit. Run serially on:

- one stable closed/wink anchor;
- one ordinary open-mouth/teeth control; and
- one tongue anchor.

Stop if any output, resource, face, component, QA, metadata, or review gate fails.

### Operational step 8 — 22-image full_v2 batch

Begin only after step 7 approval. Render serially, validate every row, generate review
artifacts after primary outputs pass, and seal evidence only after independent review.

## 13. Test and acceptance matrix

| Layer | Automated proof | Mutation/failure proof | Human proof |
| --- | --- | --- | --- |
| Exact support | Outside-support equality for uint8/float32 | Remove restore/re-clip | 100% lip/teeth crops, no seams |
| Colour delivery | ICC hash, EXIF policy, 4:4:4, roundtrip | Change profile bytes, corrupt ICC/EXIF | Colour-aware source/output A/B |
| QA | Complete pass/fail/unavailable vector | Drop/mis-scale AA6 field | Inspect flagged anchors and false alarms |
| Runner | Atomic state, resume validation, hashes | Kill process, corrupt temp/final/manifest | Operator can understand recovery report |
| Corpus | Schema, duplicates, group split | Leak group/duplicate into holdout | Label agreement/adjudication review |
| Mouth gate | Closed/ring/bypass decisions | Eight named mutations | Tongue, teeth, lipstick, profile review |
| Native pilot | Three complete 24 MP rows | Resource/evidence stop policy | Randomized source/draft/full 100% review |
| Full batch | 22 complete rows and sealed hashes | Tamper one output/review hash | Two independent reviews plus adjudication |

Focused tests do not authorize a full-suite claim. Before each code slice is called
complete, run its focused tests, the affected integration/golden tests, and at least one
real render. Run the repository full suite only at the agreed release boundary.

## 14. Operational safety and rollback

### 14.1 Filesystem safety

- Source root is read-only by policy.
- Current draft output is read-only comparison evidence.
- Full_v2 output is a new sibling directory.
- Resolve and compare real paths before creating directories.
- Refuse source/output containment in either direction.
- Refuse final overwrite without a separately approved recovery procedure.
- Temporary cleanup targets exact recorded paths only.

### 14.2 Resource safety

- First native pilot uses one render worker.
- Record peak RSS and wall time per file.
- Do not infer 24 MP concurrency from 6.24 MP crops.
- Increase concurrency only if measured capacity leaves explicit operating headroom.
- On memory pressure or abnormal duration, finish or terminate the current controlled
  file according to operator policy; do not launch another worker.

### 14.3 Behavioral rollback

Each code slice has a separate commit/review boundary. If the mouth gate fails human
review, revert that behavior rather than weakening exact-support, colour, QA, or runner
integrity. If runner evidence fails, keep all outputs diagnostic and unsealed.

## 15. Open owner decisions before code

These decisions are intentionally not guessed:

1. Which reviewed ICC profile bytes and license/provenance become
   `retouch_srgb_v1`?
2. Which EXIF fields are retained or redacted under `delivery_metadata_v1`?
3. Is C2PA/JUMBF passthrough wanted when the transformed pixels cannot inherit a new
   valid signature assertion?
4. What labelled corpus size per mouth/pose/lighting stratum is achievable?
5. Who are the two labelers and adjudicator?
6. Does this personal 22-image delivery require two blinded reviewers, or is one owner
   approval sufficient while certification remains false?
7. What machine-specific memory headroom and duration limits stop a native pilot?
8. Which implementation slice is authorized first after this documentation review?

Tongue thresholds, morphology radii, and geometry-agreement boundaries are not owner
preference questions. They remain calibration outputs from pilot data.

## 16. Definition of ready for implementation

Engineering may begin only when:

- this plan is reviewed;
- the first authorized slice is named;
- overlapping dirty-worktree files are identified and protected;
- the test and real-render evidence required for that slice is accepted;
- any new asset/license decision required by the slice is resolved; and
- no one interprets planning approval as permission to run or overwrite the batch.

## 17. Current completion boundary

Completed now:

- research audit;
- bounded full/draft A/B evidence;
- colour-delivery and deterministic-profile diagnosis;
- 105-image mouth geometry study;
- exact-support containment proof;
- QA gap diagnosis;
- prioritized implementation architecture;
- proposed interfaces, state machine, manifest, tests, mutations, rollback, and gates;
  and
- this documentation-only implementation specification.

Not completed now:

- no engine, runner, parser, QA, ICC asset, or test implementation;
- no corpus annotation/calibration;
- no production threshold or morphology value;
- no native 24 MP pilot;
- no full_v2 output folder;
- no 22-image rerender; and
- no certification or sealed final evidence.

This document is ready for owner review. Code remains intentionally untouched.
