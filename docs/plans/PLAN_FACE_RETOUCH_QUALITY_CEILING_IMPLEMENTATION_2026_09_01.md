# Plan — Face Retouch Quality-Ceiling Implementation

- **Date:** 2026-09-01
- **Status:** IMPLEMENTATION STARTED — `FR-QC-0A` is implemented and automatically
  verified in the working tree; owner visual acceptance, commit, and release remain
  pending
- **Last implementation update:** 2026-09-02
- **Repository baseline used for planning:** `fafed66`
- **Program:** next Retouch face engine and shoot workflow
- **Primary target:** Retouch's best measured source-faithful face retouch path
- **Default product track:** Photographic Truth Core
- **Optional tracks:** Bounded Learned Assist, Explainable Intent Planner, and
  Generative Studio

Related documents:

- [Face Retouch Quality Ceiling research](RESEARCH_FACE_RETOUCH_QUALITY_CEILING_2026_09_01.md)
- [AMG Day 3 full-v2 Phase 0 specification](PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md)
- [AMG Day 3 batch review](../review/REVIEW_AMGDAY32026_RETOUCH_BATCH_2026_09_01.md)
- [Face-operation visual audit](RESEARCH_FACEOP_VISUAL_AUDIT_2026_08_31.md)
- [Certification Evidence v2](../guides/CERTIFICATION_EVIDENCE_V2.md)

---

## Implementation update — 2026-09-02

The owner subsequently authorized “simple engineering first.” That authorization was
interpreted narrowly as `FR-QC-0A` only; it did not authorize `FR-QC-0B`, corpus work,
downloads, batch rerendering, or any later phase.

Current `FR-QC-0A` working-tree state:

- a policy-free `restore_outside_support()` helper restores source pixels wherever
  final support is exactly zero and validates shape, dtype, support, and newly
  non-finite output;
- lip texture reapplication and teeth whitening restore outside their final effective
  support;
- gloss support is re-clipped to permitted lip support after Gaussian feathering;
- zero-opacity texture reapplication is an exact no-op;
- focused containment tests cover `uint8`, `float32`, feather tails, gloss blur,
  detected-teeth support, and frame-edge support;
- the pre-fix implementation failed all eight new operation-containment cases;
- the current focused utility/component/golden/integration set passes 170 tests;
- direct comparison with the committed baseline confirms identical intended results
  inside texture, gloss, and teeth support for both dtypes; and
- live detection/parsing on `DSCF2397` at a 2048-pixel proxy measured exact zero
  outside-support delta for lips and teeth, with non-zero intended inside-support edits.

Retained local visual evidence:

- [DSCF2397 source/lips/teeth contact sheet](../../test_output/fr_qc_0a_exact_support_2026_09_02/DSCF2397_source_lips_teeth_contact.jpg)
- [DSCF2397 100-percent mouth crop](../../test_output/fr_qc_0a_exact_support_2026_09_02/DSCF2397_source_lips_teeth_mouth_100pct.jpg)

This is not an owner visual acceptance, commit, release, Phase 0 seal, or tongue-safety
claim. The existing semantic lip mask still includes tongue-side mouth pixels in the
known `DSCF2397` anchor; that remains `FR-QC-0E/0F`, not `FR-QC-0A`.

---

## 0. Decision

Implement the quality ceiling as a sequence of independently authorizable,
independently testable, and independently reversible tranches.

The implementation order is fixed by dependency and risk:

```text
Phase 0: mechanical and delivery correctness
  -> Phase 1: evidence, ontology, and calibrated abstention
  -> Phase 2: identity and facial-material foundation
  -> Phase 3: optical eyes/lips/teeth and region certification
  -> Phase 4: job-local shoot and subject intelligence
  -> Phase 5: bounded learned-assist benchmark and admission decisions
  -> Phase 6: optional explainable intent planner
  -> Phase 7: optional separate Generative Studio
```

No later phase may compensate for an earlier failure. In particular:

- a better parser does not excuse outside-support pixel drift;
- a neural restorer does not excuse incorrect color or metadata;
- an identity embedding does not excuse changed marks, pores, expression, or likeness;
- a preference score does not excuse unsafe anatomy or demographic normalization;
- a successful batch does not excuse missing evidence;
- human preference does not excuse a hard mechanical failure;
- optional generative quality does not delay completion of the source-faithful core.

The recommended first future authorization is **FR-QC-0A: exact-support containment**.
Planning approval alone does not authorize it.

---

## 1. Program outcome and completion milestones

### 1.1 Product outcome

Build a face-retouch system that:

- changes only the smallest support justified by user intent and evidence;
- protects identity, likeness, personal marks, expression, facial hair, makeup,
  accessories, and occluders;
- treats skin, eyes, lips, teeth, gums, tongue, hair, and makeup as different
  materials;
- preserves native source texture wherever source evidence exists;
- automatically attenuates or abstains when evidence is weak;
- is consistent across a shoot without applying identical edits blindly;
- keeps learned restoration bounded and optional;
- keeps generative pixels in a separate, clearly labeled product mode;
- exports color, metadata, and provenance truthfully; and
- produces reproducible automatic and human acceptance evidence.

### 1.2 Milestones

| Milestone | Required phases | Product meaning | Optional work not required |
|---|---|---|---|
| `M0_DELIVERY_SAFE` | Phase 0 | Current face path can be rendered and delivered without known mechanical mouth/color/batch evidence defects | New parser, material engine, learned models |
| `M1_EVIDENCE_SAFE` | Phases 0–1 | Operations expose support, representation, semantics, uncertainty, and truthful abstention | Visible quality improvement |
| `M2_TRUTH_CORE_SINGLE` | Phases 0–3 | Best certified deterministic single-image face engine, with independently accepted skin/eye/lip/teeth behavior | Shoot intelligence, learned models |
| `M3_TRUTH_CORE_SHOOT` | Phases 0–4 | Source-faithful job-local subject consistency and outlier workflow | Any learned assist, planner, Generative Studio |
| `M4_BEST_MEASURED_RETOUCH` | Phases 0–5 plus locked comparison and human acceptance | Retouch's best measured face engine on named corpus/version; Phase 5 may ship zero neural models if none wins | Planner and Generative Studio |
| `M5_PLANNER` | Phase 6 | Optional explainable planning over certified operations | Direct learned pixel generation |
| `M6_GENERATIVE_STUDIO` | Phase 7 | Optional, separate creative/synthetic product | Inclusion in default Truth Core |

### 1.3 Program completion boundary

The core quality-ceiling implementation is complete at `M4_BEST_MEASURED_RETOUCH`.

This boundary is intentionally finite:

- Phase 5 is complete when every candidate has an evidence-backed admit/reject/defer
  decision; it does not require a neural candidate to win.
- Phase 6 is an optional workflow product and cannot block the core claim.
- Phase 7 is a separate creative product and cannot block or silently enter the core.
- A future paper or model does not reopen a completed milestone unless an approved
  refresh shows it materially changes the risk/quality decision.

---

## 2. Authority and non-goals

### 2.1 Current authority

This document authorizes only:

- implementation planning;
- proposed file/module boundaries;
- proposed tests and evidence;
- proposed corpus and review work;
- proposed ticket order;
- future authorization slices.

It does not authorize:

- edits to engine, test, model, GUI, or packaging code;
- downloading models or datasets;
- collecting or labeling private face images;
- rerendering user photographs;
- creating output folders;
- changing thresholds, morphology, recipes, or defaults;
- accepting model/data licenses;
- sending images or biometric evidence to external services;
- staging, committing, or pushing files; or
- claiming any future milestone complete.

### 2.2 Program non-goals

- medical skin diagnosis or phototype inference;
- attractiveness scoring;
- demographic beauty defaults;
- automatic unrequested face/body reshaping;
- reconstructing hidden anatomy in the Truth Core;
- generating pores and describing them as recovered source detail;
- persistent cross-project face recognition;
- a big-bang engine rewrite;
- replacing human acceptance with one automatic score;
- adopting a model because its paper or vendor says state of the art;
- making HDR, print, or signed provenance claims before those paths are certified.

---

## 3. Program rules

### 3.1 One risk class per change

Each code review or implementation tranche should primarily own one risk class:

- mechanical support;
- color/representation;
- evidence/manifest;
- semantic observation;
- calibration/abstention;
- visible material behavior;
- shoot association;
- learned-model behavior;
- planner behavior;
- generative behavior.

Do not combine a parser replacement, material algorithm, UI redesign, and batch
rerender into one change.

### 3.2 Evidence before behavior

For Phases 1–5:

1. define the observation/evidence schema;
2. produce diagnostic output without changing pixels;
3. validate and calibrate the evidence;
4. make one operation consume it behind a disabled/default-compatible control;
5. run isolation, mutation, corpus, and human review;
6. enable only the accepted operation;
7. retain rollback to the previous certified path.

### 3.3 Source-safe failure

Every optional or uncertain component must fail to one of:

- unchanged source pixels;
- the previous certified deterministic path;
- a labeled preview-only artifact;
- explicit manual review.

It must never fail to a broader or more generative operation.

### 3.4 Dirty-worktree discipline

Before every tranche:

- record branch, HEAD, status, and relevant diffs;
- identify user-owned overlapping changes;
- create an exact file allowlist;
- avoid staging unrelated files;
- do not reset, checkout, clean, or overwrite user work;
- inspect `git diff --check` for the tranche;
- stage or commit only if separately requested.

### 3.5 Completion vocabulary

Use these states consistently:

- `planned` — documented, not authorized;
- `authorized` — exact tranche and files approved;
- `implemented` — code exists locally;
- `focused_verified` — named focused checks passed;
- `integration_verified` — affected integration checks passed;
- `corpus_calibrated` — calibration report frozen;
- `automatic_accepted` — all required automatic/mutation gates passed;
- `human_accepted` — required blinded review passed;
- `release_validated` — package/runtime/delivery checks passed;
- `certified` — all phase-specific evidence is sealed and no required work remains.

Never compress these into “done.”

---

## 4. Logical ownership

Actual people are owner decisions. Until assigned, use these logical roles.

| Role | Responsibility | Cannot approve alone |
|---|---|---|
| Program owner | Scope, priority, product intent, final authorization | Technical proof or own blinded review |
| Engine owner | Render interfaces, operation isolation, compatibility | Human visual acceptance |
| Color/delivery owner | ICC, transfer, metadata, export, provenance state | Face material/likeness quality |
| Face-evidence owner | Parser, landmarks, ontology, uncertainty, fallback truth | Visible operation release alone |
| Material owner | Skin/eye/lip/teeth diagnosis and render behavior | Corpus independence or license approval |
| Data steward | Consent, annotation, split integrity, retention, deletion | Threshold selection alone |
| Model/license owner | Code/weight/data/dependency provenance and distribution review | Visual or engineering acceptance alone |
| QA owner | Automatic, mutation, failure, and artifact validation | Final human preference alone |
| Review coordinator | Blinding, reviewer assignment, adjudication, sealed results | Own candidate implementation approval |
| Release owner | Runtime, package, rollback, manifest, final artifacts | Waive earlier failed gates |

Minimum separation for critical releases:

- implementer and final reviewer are different people;
- calibration labels and sealed-test labels are controlled separately;
- a learned editor and learned evaluator do not certify each other;
- one person cannot waive identity, mask, color, or provenance critical defects.

---

## 5. Program dependency graph

```text
FR-QC-0A exact support ─┬─> FR-QC-0D runner ─> FR-QC-0G pilot ─> FR-QC-0H batch
FR-QC-0B color/export ─┤
FR-QC-0C QA/evidence ──┘
FR-QC-0E corpus tooling ─> FR-QC-0F mouth gate ────────────────┘

Phase 0 accepted
  └─> 1A operation result + 1B representation ledger
        ├─> 1C soft posteriors ─> 1D ontology ─> 1E calibration/abstention
        └─> 1F observation graph + residual evidence
              └─> 1G parser challenger decision

Phase 1 accepted
  └─> 2A mark ledger + 2B evidence levels + 2C material observations
        └─> 2D linear/material reconstruction + 2E texture metrics
              └─> 2F calibrated backoff ─> 2G one-operation-at-a-time enablement

Phase 2 accepted
  └─> 3A mouth semantics ─> 3B teeth ─> 3C lips
      3D eyes
      3E accessory/occluder exclusions
      3F face/body harmony
        └─> 3G independent region certification

Phase 3 accepted
  └─> 4A subject graph ─> 4B manual correction ─> 4C stable evidence
        └─> 4D per-frame diagnosis ─> 4E consistency plan/review
              └─> 4F privacy lifecycle ─> 4G shoot release

Phase 4 accepted
  └─> 5A model governance + 5B benchmark harness
        ├─> 5C current NAFNet audit
        ├─> 5D learned observation/parameter candidates
        └─> 5E learned residual candidates
              └─> 5F high-resolution/failure lab
                    └─> 5G blinded acceptance ─> 5H admit/reject decisions

M4 core complete
  ├─> optional Phase 6 planner
  └─> optional, separate Phase 7 Generative Studio
```

---

## 6. Definition of ready for any implementation tranche

A tranche is ready only when all applicable items are resolved.

### 6.1 Scope readiness

- tranche ID and objective are named;
- allowed files/directories are listed;
- non-goals are explicit;
- expected visible behavior is stated;
- compatibility expectation is stated;
- rollback is possible without reverting unrelated work.

### 6.2 Evidence readiness

- baseline behavior is reproducible;
- known failing examples are identified;
- required synthetic, real, and mutation cases are available;
- private images remain outside Git;
- calibration and sealed-test data are separated where thresholds are involved;
- automatic versus human completion is explicit.

### 6.3 Runtime readiness

- supported Python/runtime is identified;
- actual detector/parser/model capability is probed, not inferred from imports;
- required caches are writable and isolated;
- expected provider/device is recorded;
- resource/timeout stop rules are set for native images;
- no dependency or model download occurs without approval.

### 6.4 Product decisions

- user intent/default behavior is approved;
- any ICC/metadata/provenance policy needed by the tranche is approved;
- any model/data/license decision is approved;
- reviewer/labeler roles are assigned when required;
- output destinations and overwrite policy are approved for operational runs.

---

## 7. Phase 0 — mechanical and delivery correctness

Phase 0 is governed by the AMG full-v2 specification. This roadmap names its work as
program tranches and adds the minimum ontology/provenance requirements discovered in
the second research pass.

### FR-QC-0A — exact-support lip and teeth containment

**Goal:** eliminate full-canvas representation drift from local mouth operations.

Expected files:

- `retouch/lips.py`;
- `retouch/teeth.py`;
- optional small exact-support helper;
- `tests/test_lips.py`;
- `tests/test_teeth.py`;
- focused operation-isolation evidence.

Required work:

1. retain the original operation input;
2. define the final effective support after all masks and strength decisions;
3. perform the existing operation;
4. re-clip any blurred/gloss support to permitted support;
5. restore source exactly where final support is zero;
6. validate dtype, shape, and finite values;
7. expose support/residual evidence without changing public behavior unnecessarily.

Required tests:

- uint8 and float32 exact outside-support equality;
- zero and empty masks;
- full and partial support;
- feather tails;
- gloss blur containment;
- teeth whitening containment;
- multi-face and edge-of-frame isolation;
- mutation removing final source restore;
- real 100% mouth crops.

Exit:

- no unauthorized residual exists outside final support;
- inside-support intended output remains accepted;
- mutations fail;
- affected golden changes are explained;
- no parser, threshold, runner, or recipe behavior is mixed into the tranche.

Rollback: revert 0A only; no schema or model dependency.

### FR-QC-0B — canonical SDR color and metadata delivery

**Goal:** make ingest/export deterministic and truthful for the current 8-bit engine.

Expected files:

- reviewed ICC asset and provenance/license note;
- `retouch/io.py`;
- color-context/export tests;
- metadata policy documentation;
- manifest fields for delivery and provenance state.

Required decisions before code:

- exact ICC asset bytes and hash;
- `delivery_metadata_v1` retain/redact/reset fields;
- orientation behavior;
- JPEG quality/subsampling policy;
- C2PA byte-passthrough versus omission policy;
- user-visible provenance terminology.

Required tests:

- profile hash stable across process/machine fixtures where applicable;
- embedded-profile conversion before sRGB operations;
- orientation applied once and reset;
- JPEG 4:4:4 validation;
- EXIF policy allow/deny cases;
- decoded output dimensions and profile;
- current 8-bit precision stated truthfully;
- invalid/corrupt ICC and metadata fail safely;
- C2PA passthrough never reports a new valid derived assertion.

Exit:

- identical inputs/policy produce deterministic delivery metadata/profile behavior;
- decoded output and manifest agree;
- no PNG-16/end-to-end-16-bit overclaim;
- color review on representative sources passes.

Rollback: disable the new deterministic asset/policy path; full-v2 delivery remains
blocked until color determinism is restored.

### FR-QC-0C — complete QA vector and evidence truth

**Goal:** make every required detector/quality result present as pass, fail, or
unavailable with a reason.

Expected files:

- `retouch/engine.py`;
- `retouch/qa_detectors.py` or current QA owner;
- `retouch/render_manifest.py` and certification adapters;
- QA/evidence tests.

Required work:

- repair the draft AA6 displacement-field scale mismatch;
- define the required QA vector and schema version;
- distinguish not-run, unavailable, failed, warning, and passed;
- record detector/parser/provider/fallback truth;
- ensure proxy/native coordinate transforms are explicit;
- prevent warning-only records from certifying.

Exit:

- complete evidence for full and draft fixtures;
- coordinate mutation fails;
- omitted-field mutation fails;
- certification adapter cannot upgrade unavailable evidence;
- real anchor evidence is interpretable.

Rollback: render may remain diagnostic, but certification stays unavailable.

### FR-QC-0D — atomic resumable runner and manifest

**Goal:** create a reusable production folder renderer without event logic in the
library.

Expected files:

- proposed `retouch/delivery_manifest.py` or an approved extension of
  `retouch/render_manifest.py`;
- proposed `scripts/batch/full_v2_batch.py`;
- atomic/resume/failure tests;
- CLI/API/recovery documentation.

Required behavior:

- immutable source discovery and hashes;
- source/output containment refusal;
- explicit engine quality separate from JPEG quality;
- same-directory temporary output;
- decode/profile/metadata/dimensions/hash validation before atomic replace;
- atomic manifest checkpoint per file;
- resume only after revalidating completed rows;
- deterministic exit status;
- no overwrite without a separately approved policy;
- contact/review artifacts generated only after primary outputs validate.

Exit:

- kill/restart, corrupt temp, corrupt final, changed source, and changed-policy tests
  behave as documented;
- no partial output is reported complete;
- operator recovery record is understandable;
- no source image changes.

Rollback: retain only diagnostic test artifacts; do not run a production folder.

### FR-QC-0E — mouth corpus tooling

**Goal:** create private, reproducible calibration/holdout infrastructure without
inventing thresholds.

Expected files:

- label schema and validator;
- split/duplicate/leakage tooling;
- label instructions;
- corpus manifest format;
- tests with synthetic/non-private fixtures;
- no private images in Git.

Required labels:

- mouth openness and pose;
- lip vermilion and transition;
- oral aperture;
- teeth, gums, tongue, saliva/wet highlights;
- braces/retainers/grills/jewelry/food/unknown;
- lipstick/face paint;
- blur, scale, occlusion, parser arm, lighting;
- intended safe behavior.

Exit:

- schema and group split pass;
- near-duplicate and identity leakage tests pass;
- two-labeler/adjudication process is usable;
- corpus version/hash can be frozen;
- engine behavior remains unchanged.

Rollback: thresholds and 0F stay blocked.

### FR-QC-0F — calibrated tongue-safe mouth gate

**Goal:** consume calibrated mouth evidence once per face and fail open safely.

Expected files:

- proposed `retouch/mouth_safety.py`;
- `retouch/perf_optimizations.py`;
- evidence aggregation in `retouch/engine.py`;
- calibration record;
- unit, integration, mutation, real-isolation, and parser-arm tests.

Required behavior:

- closed-mouth, safe-core, uncertain, and bypass states;
- no broad color-only tongue detector;
- parser arm recorded;
- unknown/tongue/gum/teeth/mouth-interior exclusions;
- same safe support consumed by every lip sub-operation;
- malformed/degenerate evidence returns source-safe behavior;
- no user-facing morphology sliders before calibration.

Exit:

- locked holdout passes the AMG mouth boundary;
- normal lips still receive intended correction;
- tongue/gums/teeth/interior remain unchanged by lip-only isolation;
- profile/uncertain cases abstain safely;
- named mutations fail;
- human mouth review passes.

Rollback: disable/revert 0F only; retain 0A–0E.

### FR-QC-0G — three-anchor native pilot

**Goal:** prove the full Phase 0 chain at native resolution before a full batch.

Inputs:

- one stable closed/wink anchor;
- one ordinary open-mouth/teeth control;
- one tongue anchor.

Run rules:

- serial worker initially;
- new sibling diagnostic destination;
- source and accepted draft are read-only;
- record peak RSS, wall time, provider, detector/parser, output policy, and hashes;
- stop on any mechanical, resource, evidence, metadata, or review failure.

Exit:

- all three primary outputs validate;
- support/color/QA evidence is complete;
- randomized source/draft/full review passes;
- operating limits are recorded.

### FR-QC-0H — full-v2 AMG batch and Phase 0 seal

**Goal:** render, validate, review, and seal the 22-image batch non-destructively.

Exit:

- 22/22 completed and validated rows;
- zero unexplained failures;
- complete face/component/QA/color/metadata/integrity evidence;
- non-empty review artifacts;
- independent source/draft/full review and adjudication as approved;
- sealed manifest and review commitments;
- originals and prior draft unchanged;
- Phase 0 closeout documents exact known limitations.

Phase 0 is not complete if only files exist or only unit tests pass.

---

## 8. Phase 1 — evidence, ontology, and calibrated abstention

Phase 1 may preserve visible output. Its product is trustworthy evidence infrastructure.

### FR-QC-1A — common operation-result protocol

Proposed responsibility:

- operation proposal pixels;
- exact final support;
- protected and unknown support;
- residual summaries;
- evidence references;
- decision/apply state;
- fallback and failure reason.

Candidate files:

- new `retouch/operation_result.py` or approved equivalent;
- incremental adapters in `retouch/lips.py`, `retouch/teeth.py`, then other operations;
- serialization/schema tests.

Exit:

- at least lip and teeth use the common result without visible regression;
- final support cannot expand during serialization/composition;
- a mutation dropping support/evidence fails.

### FR-QC-1B — representation ledger

Proposed responsibility:

- current color space/profile;
- encoded versus linear transfer state;
- dtype and range;
- alpha/premultiplication state;
- native versus proxy coordinate scale;
- orientation and decode revision;
- conversion provenance.

Candidate files:

- new `retouch/representation.py` or an extension of `ColorContext`;
- `retouch/io.py`;
- shared conversion tests;
- cache/evidence schema updates.

Exit:

- every material/operation boundary states accepted and returned representation;
- illegal double decode, linear/encoded confusion, and hidden uint8 conversion
  mutations fail;
- preview and final use the correct render revision.

### FR-QC-1C — soft posterior extraction from current parser

Goal:

- retain class probabilities/logits and entropy before hard `argmax`;
- expose per-face parser arm and preprocessing;
- preserve current hard-mask behavior until calibrated consumers exist.

Candidate files:

- `retouch/parsing.py`;
- new posterior/evidence dataclasses;
- `tests/test_parsing_real_model.py`;
- `tests/test_parsing_fallback.py`;
- `tests/test_parsing_feather.py`;
- new posterior/calibration fixtures.

Exit:

- logits/posteriors are mapped correctly to native ROI coordinates;
- hard-mask compatibility passes where intended;
- fallback never reports neural probability;
- malformed model output fails open and truthfully.

### FR-QC-1D — editing ontology and protection graph

Goal:

- version editable/protected/unknown ownership for skin, mouth, eyes, hair, makeup,
  accessories, hands, costume elements, marks, and unknown pixels.

Proposed files:

- new `retouch/face_ontology.py`;
- new `retouch/protection_graph.py`;
- adapters from parser/landmarks/marks;
- schema, conflict, and mutation tests.

Exit:

- every migrated operation declares owned/protected classes;
- unknown is protected by construction;
- conflicting ownership is resolved deterministically and recorded;
- removal of hand/hair/glasses/makeup/mouth protection fails tests.

### FR-QC-1E — calibrated abstention controller

Goal:

- replace raw-confidence heuristics at critical paths with calibrated operation states:
  `APPLY`, `ATTENUATE`, `PREVIEW_ONLY`, `SKIP_SAFE`, `MANUAL_REQUIRED`.

Proposed files:

- new `retouch/abstention.py`;
- calibration registry/schema;
- reliability/coverage tooling;
- mouth and eye consumers first;
- mutation/OOD/fallback tests.

Exit:

- operation-specific calibration and sealed-test reports exist;
- coverage, support size, and abstention are reported by critical condition;
- no one global face-confidence scalar controls all operations;
- uncalibrated/model-fallback evidence cannot produce `APPLY` silently.

### FR-QC-1F — face observation and diagnosis graph

Goal:

- separate observations from edit decisions and pixel mutation.

Proposed boundaries:

- `face_observation` — detector/parser/landmark/visibility;
- `face_diagnosis` — material/degradation/mark/lighting evidence;
- `operation_plan` — user intent and policy;
- `operation_result` — pixels/support/evidence.

Implementation rule:

- adapt current modules incrementally;
- do not move all engine logic at once;
- add one read-only observation adapter at a time;
- preserve existing public API until a migration is approved.

Exit:

- at least one end-to-end critical operation flows through observation, plan, result,
  residual evidence, and abstention;
- observation modules cannot mutate pixels;
- fallback/provider/model truth is serialized.

### FR-QC-1G — parser challenger bake-off

Goal:

- benchmark current BiSeNet versus licensed candidates such as SegFace/FaRL only after
  the ontology, corpus, metrics, and runtime harness exist.

Required evidence:

- class and boundary metrics;
- calibration/reliability;
- critical-cost errors;
- face scale/pose/makeup/accessory/occlusion slices;
- runtime/memory/provider;
- code/weights/data/dependency license record;
- fallback and packaging feasibility.

Exit outcomes:

- `retain current`;
- `adopt challenger behind migration`;
- `use challenger only as secondary evidence`;
- `reject`;
- `defer for missing evidence`.

No model swap is required for Phase 1 completion.

### Phase 1 seal

Phase 1 is complete only when:

- operation and representation contracts are active;
- soft evidence and fallback truth are available;
- ontology/protection and abstention are mutation-sensitive;
- at least one critical operation consumes the new evidence safely;
- output compatibility or intended changes are documented;
- no unreviewed visible behavior ships;
- evidence schema and rollback are versioned.

---

## 9. Phase 2 — identity and facial-material foundation

Phase 2 begins diagnostic-only. Visible corrections are enabled individually.

### FR-QC-2A — subject mark ledger

Goal:

- distinguish stable identity marks, temporary blemishes, makeup/paint, hair, and
  unknown spots using policy plus cross-frame evidence where available.

Candidate files:

- `retouch/marks.py`;
- new job-local mark ledger module;
- schema and persistence tests;
- `tests/test_marks.py` plus cross-frame fixtures.

Required behavior:

- preserve policy always wins;
- stable identity evidence is never removed by generic blemish logic;
- uncertainty is recorded;
- no cross-project biometric storage;
- manual correction is versioned.

### FR-QC-2B — evidence-level classifier

Goal:

- assign `E0_RENDERED_SINGLE`, `E1_DECLARED_SINGLE`, `E2_SHOOT_MULTI`, or
  `E3_CALIBRATED_CAPTURE` per face property.

Inputs:

- source type/color context;
- scene/context availability;
- multi-frame registration/coverage;
- capture/calibration records;
- property-specific visibility.

Exit:

- evidence cannot be promoted by model confidence;
- claims use observed/inferred/generated vocabulary;
- manifests record downshifts and missing evidence.

### FR-QC-2C — unified material observations

Goal:

- coordinate current intrinsic, chromophore, specular, makeup, texture, lighting, and
  harmony signals into one read-only material evidence record.

Candidate files:

- `retouch/intrinsic.py`;
- `retouch/chromophore_v2.py`;
- `retouch/specular.py`;
- `retouch/makeup_unmix.py`;
- `retouch/frequency.py`;
- `retouch/lighting.py`;
- `retouch/harmony.py`;
- proposed `retouch/material_evidence.py`.

Required record:

- diffuse-like variation confidence;
- form/shadow confidence;
- specular intensity/roughness proxy;
- makeup/coating likelihood;
- texture observability and scale;
- facial/vellus hair protection;
- scene-light evidence;
- reconstruction residual;
- conflicts and unknowns.

No pixel changes in the first tranche.

### FR-QC-2D — linear-light and component reconstruction spike

Goal:

- prove current intrinsic/material operations use declared representations and can
  reconstruct the observed image within defined residuals.

Required comparisons:

- current encoded-space behavior;
- corrected linear-light candidate;
- skin/body/background boundaries;
- varied profiles and exposure;
- makeup and facial hair;
- component reconstruction error.

Exit outcome may be retain, migrate, narrow, or reject. Do not force a rewrite.

### FR-QC-2E — microtexture evidence and duplication alarms

Goal:

- measure texture retention without rewarding false or copied pore detail.

Required metrics:

- region/scale-stratified band energy;
- local phase/correlation;
- repeated-patch/nearest-neighbor duplication;
- edge overshoot/undershoot;
- vellus/facial-hair survival;
- pore observability status;
- native crops and human labels.

Rule:

- small faces below measurable scale receive `not_observable`, never “pores preserved.”

### FR-QC-2F — calibrated material backoff

Goal:

- attenuate or skip material operations when component conflict, evidence level,
  texture observability, marks, makeup, lighting, or occlusion is unsafe.

Exit:

- decision is operation-specific;
- source-safe fallback is proven;
- calibration and mutation gates pass;
- no demographic beauty default exists.

### FR-QC-2G — one-operation-at-a-time visible enablement

Recommended order:

1. observational evidence only;
2. low-frequency blotch/form separation;
3. bounded specular finish correction;
4. makeup-aware protection;
5. gradual blemish/temporary-mark correction;
6. texture transfer only if same-subject/region/scale evidence passes.

Each operation needs its own:

- support and component contract;
- strength/residual budget;
- exact isolation tests;
- mark/identity gates;
- color and texture review;
- automatic backoff;
- feature flag and rollback;
- blinded human acceptance.

### Phase 2 seal

Phase 2 is complete when:

- mark, evidence-level, and material records are versioned and truthful;
- diagnostic evidence passes the locked corpus;
- every visible operation enabled in this phase passes independently;
- disabled or rejected operations are documented without blocking the phase;
- no prior-generated map is presented as measured skin truth;
- previous certified behavior remains available.

---

## 10. Phase 3 — optical facial-feature quality

### FR-QC-3A — mouth semantic refinement

Own:

- lip vermilion/transition;
- oral aperture;
- teeth;
- gums;
- tongue;
- saliva/wet highlights;
- braces, retainers, grills, jewelry, food, and unknown objects;
- openness, blur, visibility, and occlusion.

Exit:

- critical classes have dedicated metrics and corpus slices;
- unknown is protected;
- mouth evidence is shared consistently across lip/teeth operations.

### FR-QC-3B — natural teeth material correction

Required behavior:

- gradual brightness/chroma adjustment;
- no gum/tongue/interior whitening;
- preserve tooth individuality, translucency, gaps, shadows, restorations, braces;
- no geometry/alignment change in Truth Core;
- high-confidence core or safe skip when evidence is weak.

Required review:

- warm/cool lighting;
- lipstick transfer;
- partial teeth;
- open smile and gummy smile;
- blur/compression;
- braces/restorations;
- diverse natural tooth color.

### FR-QC-3C — lip material correction

Required behavior:

- separate diffuse color, lines/texture, and wet/specular evidence;
- protect oral interior, teeth, gums, tongue, and lipstick boundaries;
- preserve deliberate makeup asymmetry and character colors;
- no generic gloss placement unrelated to photographed lighting.

### FR-QC-3D — eye material correction

Required regions:

- sclera;
- iris/pupil;
- corneal/catchlight layer;
- lids/lashes/brows;
- tear line;
- glasses lens/frame/glare;
- eye-specific visibility/occlusion.

Required behavior:

- preserve photographed catchlight direction/count/shape unless creative intent is
  explicit;
- bound scleral brightening/redness changes;
- prevent sharpening/paint on tiny or saturated irises;
- preserve eyelashes, liner, contacts, makeup, and glasses;
- gate each eye independently.

### FR-QC-3E — long-tail accessory and occluder protection

Minimum corpus:

- glasses and glare;
- earrings/piercings;
- facial/scalp hair;
- fingers/hands;
- masks/veils/lace;
- microphones;
- cosplay prosthetics, face paint, glitter, wigs, and costume edges;
- profile and edge-of-frame cases.

Exit:

- protected objects retain shape, count, color, and visibility;
- hidden face pixels remain source-derived;
- object-removal intent remains a separate operation.

### FR-QC-3F — face/body/shoot harmony calibration

Goal:

- prevent a polished face from separating unnaturally from neck, ears, body, noise,
  focus, light, or scene.

Rules:

- body skin is a separate region/material;
- face settings do not transfer numerically by default;
- harmony evidence is observational before automatic response;
- lighting and depth-of-field logic are preserved;
- body/scene edits require their own authorization.

### FR-QC-3G — independent region certification

Certification units:

- skin;
- eyes;
- lips;
- teeth/mouth;
- accessories/occlusion;
- full-face integration;
- face/body/photographic coherence.

No overall-face pass can hide a failed region.

### Phase 3 seal

Phase 3 is complete when each enabled region has:

- locked corpus results;
- exact support and representation proof;
- mutation proof;
- texture/material/identity evidence;
- two-reviewer acceptance and adjudication where required;
- versioned rollback;
- no critical defect.

---

## 11. Phase 4 — job-local shoot and subject intelligence

### FR-QC-4A — job-local subject graph

Goal:

- associate repeated faces inside one project without creating a persistent people
  database.

Inputs:

- embeddings as one signal;
- time/order;
- clothing/scene evidence;
- face visibility/quality;
- manual confirmation.

Rules:

- anonymous project-local IDs;
- uncertain clusters stay separate;
- no cloud service by default;
- association does not authorize pixel edits.

### FR-QC-4B — manual subject correction

Required workflow:

- add missed face;
- remove false face;
- merge/split subject clusters;
- select/deselect faces;
- correct policy per subject;
- show association confidence and affected frames;
- undo/reset with audit history.

### FR-QC-4C — stable mark and appearance evidence

Use repeated frames to improve:

- stable-mark classification;
- facial-hair and makeup continuity;
- recurrent texture evidence;
- view-dependent specular diagnosis;
- exposure/white-balance outlier detection;
- pose/visibility confidence.

Do not transplant pixels or construct a canonical rendered face.

### FR-QC-4D — frame-specific diagnosis

For every frame, compare subject baseline against:

- exposure and white balance;
- pose/expression;
- face scale and focus;
- occlusion/accessories;
- makeup/hair changes;
- temporary marks;
- noise/compression;
- requested style.

The same subject may legitimately receive different numeric strengths.

### FR-QC-4E — consistency plan and review

Deliverables:

- subject-level intent/policy;
- frame-level operation plans;
- exception/outlier queue;
- contact sheet grouped by subject and shot;
- stable-mark and consistency overlays;
- before/after/outlier review;
- rerender-only selected frames.

### FR-QC-4F — privacy and lifecycle

Required decisions:

- project-local storage location;
- encryption/access requirements if applicable;
- retention default;
- explicit export;
- destruction/reset procedure;
- logs/manifest fields that avoid unnecessary biometric data;
- no reuse across projects without separate consent.

### FR-QC-4G — shoot release and comparison

Required proof:

- measurable consistency improvement over independent-frame baseline;
- no subject association error in sealed test;
- no mark/identity/privacy regression;
- operator can correct clusters and exceptions;
- interrupted job resumes without cross-subject contamination;
- human review prefers or accepts the shoot result.

### Phase 4 seal

Phase 4 completion produces `M3_TRUTH_CORE_SHOOT`.

---

## 12. Phase 5 — bounded learned-assist benchmark and decisions

### FR-QC-5A — model governance registry

For every candidate record:

- operation and risk class `L1`–`L4`;
- repository/architecture;
- exact weight hash/source;
- code, weight, base-model, data, and dependency license review;
- training provenance and synthetic/real composition where known;
- input color/range/precision/resolution;
- provider/device/determinism;
- runtime/memory/tiling behavior;
- corpus limitations;
- fallback and rollback;
- packaging/distribution status.

No model download or acceptance occurs merely by adding a registry row.

### FR-QC-5B — common benchmark harness

Tasks are separate:

- sensor/noise denoise;
- blur/deblur;
- compression cleanup;
- blemish localization;
- low-frequency tone/color operator;
- severe face restoration.

Required outputs:

- source and candidate;
- exact support/residual;
- frequency and color analysis;
- mark/identity/geometry alarms;
- runtime/memory/provider;
- determinism/tiling artifacts;
- per-condition metrics;
- blinded review package.

### FR-QC-5C — current NAFNet baseline audit

Goal:

- determine whether current broad pre-pipeline NAFNet behavior is beneficial for face
  jobs and whether it should remain, narrow, move, or disable.

Compare:

- disabled/classical baseline;
- current NAFNet placement;
- face-local or degradation-gated candidate;
- small/large faces;
- clean/noisy/compressed sources;
- makeup, marks, hair, eyes, lips, teeth;
- native resolution and provider arms.

### FR-QC-5D — `L1`/`L2` candidates

Preferred candidates:

- learned blemish proposal masks;
- degradation estimates;
- bilateral/low-frequency gain or color coefficients;
- planner-only evidence.

Admission preference:

- source pixels remain render authority;
- output parameters are bounded;
- support is locked outside the model;
- fallback is deterministic.

### FR-QC-5E — `L3` residual candidates

Candidate tasks:

- denoise;
- deblur;
- limited restoration when source evidence remains.

Required extra gates:

- per-band residual budgets;
- no geometry/support expansion;
- mark and local-correlation survival;
- no copied/generated pore detail;
- source blend/backoff;
- no generative fallback.

### FR-QC-5F — high-resolution and failure laboratory

Test:

- native face-size strata;
- crop-origin shifts;
- tiled and untiled equivalence where claimed;
- padding and edge-of-frame;
- multi-face context;
- OOM/timeout/provider absence;
- model hash/provider substitution;
- repeated runs;
- full image versus isolated face crop;
- silent downscale/upscale detection.

### FR-QC-5G — blinded acceptance

Compare each candidate against the current best certified deterministic baseline.

Required result:

- correction success target met;
- non-inferior on every hard fidelity/safety pillar;
- no critical defect;
- per-condition limitations disclosed;
- automatic metric and human results reported separately;
- independent evaluator does not replace human review.

### FR-QC-5H — admit/reject/defer decisions

For each task/model choose:

- `ADMIT_OPT_IN` — wins and passes all gates;
- `ADMIT_DIAGNOSTIC_ONLY` — useful evidence, no pixel authority;
- `REJECT_QUALITY`;
- `REJECT_FIDELITY`;
- `REJECT_LICENSE`;
- `REJECT_RUNTIME`;
- `DEFER_EVIDENCE`.

Shipping rules:

- independent feature flag;
- exact model record;
- deterministic/classical fallback;
- no blanket “neural mode” approval;
- no `L4` default/fallback path.

### Phase 5 seal

Phase 5 is complete when all scoped candidates have sealed decisions. Zero admitted
models is a valid successful outcome.

`M4_BEST_MEASURED_RETOUCH` additionally requires:

- pairwise blinded comparison against the previous best certified Retouch path;
- pre-registered preference/non-inferiority rules;
- no critical defect;
- current competitor bake-off on the named corpus;
- reproducible delivery evidence;
- claim wording limited to the measured corpus/version.

---

## 13. Phase 6 — optional explainable intent planner

Phase 6 is optional and cannot delay the core engine.

### FR-QC-6A — operation vocabulary and intent schema

Define:

- target region/material;
- requested correction;
- preserve/remove policy;
- strength and maximum budgets;
- geometry permission;
- source-faithful versus creative intent;
- required evidence;
- manual-confirmation requirement.

### FR-QC-6B — deterministic plan resolver

Before any language model:

- resolve compatible operation order;
- detect conflicts;
- enforce protected/unknown support;
- consume evidence/abstention;
- show planned effects and rollback;
- never raise authority above evidence.

### FR-QC-6C — optional learned/language planning

Allowed:

- convert natural-language intent into the operation schema;
- explain suggestions;
- rank certified alternatives;
- request clarification.

Forbidden:

- direct pixel generation in Truth Core;
- inventing unsupported controls;
- bypassing policy or abstention;
- changing identity/geometry without explicit intent;
- self-certification.

### FR-QC-6D — adversarial and workflow evaluation

Test:

- vague beauty requests;
- conflicting instructions;
- demographic/age normalization requests;
- hidden geometry implications;
- unsupported region/model;
- prompt injection through filenames/metadata;
- creative request crossing Truth Core boundary;
- rollback and explanation accuracy.

### Phase 6 seal

- planner reduces workflow effort or errors in controlled study;
- renderer remains the sole core pixel authority;
- every plan is inspectable and reproducible;
- destructive/creative operations require human confirmation;
- planner can be disabled without changing engine results.

---

## 14. Phase 7 — optional separate Generative Studio

Phase 7 is not an engine fallback. It is a separate product boundary.

### FR-QC-7A — product and artifact separation

Require:

- separate mode/workspace;
- separate output naming;
- visible synthetic/creative label;
- separate manifest schema fields;
- source and generated-detail provenance;
- no automatic substitution into Truth Core output.

### FR-QC-7B — licensed model/data review

Review independently:

- code;
- weights;
- base model;
- training/fine-tuning data;
- generated training data;
- commercial/distribution terms;
- privacy and external-processing behavior.

### FR-QC-7C — creative operation boundaries

Candidate operations may include:

- severe restoration with invented detail;
- expression/age/gaze/geometry changes;
- makeup/style synthesis;
- object removal or hidden-face reconstruction;
- reference-conditioned creative editing;
- group generative editing.

Every operation discloses that source pixels/details may be synthesized.

### FR-QC-7D — Generative Studio acceptance

Require:

- identity/likeness review;
- object/accessory consistency;
- prompt/intent compliance;
- generated artifact detection;
- reference-photo privacy;
- human creative acceptance;
- provenance labeling;
- no confusion with archival/source-faithful output.

### Phase 7 seal

Users and reviewers can always identify the creative boundary, generated artifacts are
labeled and reproducible where claimed, and no error/fallback path can place a
Generative Studio result into the default photographic delivery.

---

## 15. Cross-cutting workstreams

### 15.1 Corpus program

Maintain separate, versioned layers:

- `C1` synthetic mechanical fixtures;
- `C2` licensed/consented real single-image corpus;
- `C3` professional paired-retouch corpus;
- `C4` shoot and multi-view corpus;
- private event/customer acceptance sets;
- sealed competitor bake-off set.

Rules:

- identity/group-disjoint splits;
- near-duplicate detection;
- immutable test commitments;
- no tuning on sealed outcomes;
- consent/license/retention record;
- private images outside source control;
- labels, schemas, and adjudication versioned;
- corpus deletion does not leave orphaned biometric caches.

### 15.2 Professional competitor bake-off

Products:

- Adobe Photoshop;
- Capture One;
- Retouch4me;
- Evoto;
- MeituYunxiu/Meitu professional path;
- any later product approved for the locked comparison.

Protocol:

- record exact version, model/cloud status, settings, and operator time;
- use the same source or RAW and explicit export target;
- separate automatic result from manual cleanup;
- register comparable crops without invalid geometric assumptions;
- inspect full image and native face crops;
- measure color/profile/metadata and residuals;
- blind product identity during review;
- disclose vendor/cloud/privacy limitations;
- never substitute vendor marketing examples for test output.

### 15.3 Standards and delivery

Version targets:

- ICC working/delivery policy;
- Exif/Exif-to-XMP mapping policy;
- JPEG/PNG/TIFF encoding policy;
- SDR default and any future HDR mode;
- C2PA validation-state vocabulary;
- output/manifest schema;
- package/runtime/model identity.

### 15.4 Documentation

Each tranche updates only applicable documents:

- architecture/API;
- recipe/parameter guide;
- runtime recovery;
- batch/CLI guide;
- certification evidence;
- model/license inventory;
- corpus/label instructions;
- release/rollback notes;
- this plan's status table.

Docs must state actual implementation and verification, not planned behavior.

---

## 16. Proposed module and file map

Names are provisional. Existing modules should be extended before creating duplicate
abstractions.

| Boundary | Candidate file(s) | First phase | Responsibility | Must not do |
|---|---|---:|---|---|
| Exact support | existing `lips.py`, `teeth.py`, optional helper | 0 | Restore source outside final operation support | Own mask morphology or intent |
| Delivery manifest | new module or `render_manifest.py` extension | 0 | Atomic schema/state/hash validation | Duplicate color/render algorithms |
| Mouth safety | proposed `mouth_safety.py` | 0 | Pure mouth support decision and evidence | Write files or resolve recipe |
| Operation result | proposed `operation_result.py` | 1 | Pixels/support/residual/evidence contract | Expand support silently |
| Representation | `io.py` plus proposed ledger | 1 | Color/transfer/dtype/scale/orientation truth | Mutate aesthetic intent |
| Region posteriors | `parsing.py` plus evidence type | 1 | Soft semantic probabilities/calibration inputs | Claim fallback probability |
| Face ontology | proposed ontology/protection modules | 1 | Editable/protected/unknown ownership | Decide beauty strength |
| Abstention | proposed `abstention.py` | 1 | Calibrated operation state | Override explicit protected intent |
| Face observation | incremental adapters | 1 | Detector/parser/landmark/visibility evidence | Mutate pixels |
| Mark ledger | `marks.py` plus job-local ledger | 2 | Stable/temporary/unknown mark evidence | Persist identity across projects |
| Material evidence | proposed coordinator over current modules | 2 | Diffuse/form/specular/makeup/texture/light evidence | Claim medical truth or generate pixels |
| Face fidelity QA | current QA/certification plus new metrics | 2 | Isolation/material/identity/shoot alarms | Certify alone |
| Subject calibration | proposed job-local module | 4 | Association and multi-frame evidence | Become a people database |
| Bounded restore | proposed adapters | 5 | Guarded `L1`–`L3` model execution | Use `L4` fallback |
| Intent planner | proposed planner/schema | 6 | Certified operation plans | Generate core pixels or self-certify |
| Generative Studio | separate package/workspace | 7 | Explicit synthetic/creative output | Enter default engine |

---

## 17. Test strategy

### 17.1 Test pyramid

1. **Pure unit tests**
   - mask ownership;
   - support restoration;
   - representations;
   - calibration/decision tables;
   - schema/state machines;
   - policy conflict resolution.
2. **Synthetic image tests**
   - exact masks and color patches;
   - boundary/feather behavior;
   - known frequency/texture patterns;
   - corruption and OOD fixtures;
   - deterministic mutations.
3. **Focused integration tests**
   - current real module combinations;
   - parser/fallback/provider arms;
   - engine result/evidence plumbing;
   - atomic runner/recovery;
   - package/model availability truth.
4. **Real isolated operation tests**
   - source/output/residual/support crops;
   - face scale, pose, makeup, hair, accessories, occlusion;
   - zero/nominal/maximum strength;
   - repeated application.
5. **Locked corpus tests**
   - calibration versus sealed test;
   - per-condition metrics;
   - fairness and abstention;
   - no test leakage.
6. **Native operational pilots**
   - full-resolution time/memory;
   - delivery/profile/metadata;
   - resumability and evidence.
7. **Blinded human review**
   - normal view and native 100%;
   - two reviewers plus adjudication where required;
   - critical defect veto.
8. **Release/package validation**
   - clean environment/package;
   - actual face detector/parser/model;
   - manifest and rollback;
   - supported platform.

### 17.2 Focused test command convention

Use the repository-managed runner:

```text
./scripts/dev/test -q <focused test files>
```

Before a tranche is described as complete, report:

- exact command;
- pass/fail/skip/warning counts;
- runtime/provider environment;
- mutation command/result where applicable;
- integration/golden result;
- real-render evidence;
- full-suite status, including if not run or blocked.

Do not infer full-suite or visual certification from focused checks.

### 17.3 Mutation minimum

Every high-risk tranche has at least one mutation that would reintroduce the targeted
defect and must fail.

Minimum families:

- remove exact source restore;
- swap semantic classes or eyes;
- replace soft evidence with hard `argmax`;
- mark fallback/model-unavailable as valid neural evidence;
- remove protected/unknown support;
- promote `E0` evidence;
- disable mark precedence;
- expand learned support;
- substitute model hash/provider;
- inject copied or synthetic texture;
- bypass critical human review;
- accept incomplete manifest;
- report C2PA passthrough as valid derived signature;
- cross Truth Core/Generative Studio fallback boundary.

### 17.4 Hard versus calibrated gates

Hard gates have no threshold tuning:

- source immutability;
- exact outside-support equality where required;
- schema/hash/destination safety;
- valid shape/dtype/finite values;
- no critical defect;
- no unapproved generative fallback;
- truthful evidence/provenance state.

Calibrated gates require pilot distributions and sealed tests:

- semantic confidence/coverage;
- morphology and feathering;
- identity/likeness alarms;
- texture/residual budgets;
- material/component confidence;
- shoot consistency;
- learned-model acceptance;
- human preference margins.

---

## 18. Evidence artifact contract

### 18.1 Per render

- source path/ID and hash;
- output path and hash;
- source/output dimensions and orientation;
- color/profile/transfer/precision policy;
- recipe and resolved parameters;
- engine revision and dirty state;
- runtime/provider/model hashes;
- detector/parser/fallback evidence;
- face instances and subject-local IDs if enabled;
- operation results and supports;
- protected/unknown overlap;
- representation ledger;
- automatic QA vector;
- warnings/failures/abstentions;
- output validation;
- provenance state.

### 18.2 Per operation

- operation/version;
- user intent and policy;
- face/region ID;
- evidence level and relevant property confidences;
- decision state;
- exact support artifact/hash;
- proposed/final residual summaries;
- per-band/color/material metrics;
- fallback/backoff reason;
- model record when learned;
- source restoration status.

### 18.3 Per calibration

- corpus and label version/hash;
- identity/group split commitments;
- train/development/calibration/test boundaries;
- metric/loss definition;
- candidate thresholds and selection process;
- distributions and condition slices;
- coverage/abstention/error trade-offs;
- chosen frozen values;
- sealed-test result;
- known missing/underpowered conditions;
- reviewer/approver record.

### 18.4 Per human review

- immutable pair commitment;
- randomized/blinded side mapping;
- reviewer IDs kept separate from render authorship;
- region/severity/confidence labels;
- normal-view and 100% answers;
- critical defect flags;
- disagreement/adjudication;
- final accept/reject;
- review artifact hashes.

### 18.5 Per release

- milestone/phase versions;
- included tranche IDs;
- source revision and diff scope;
- test and mutation summary;
- corpus and human-review summary;
- model/data/license records;
- runtime/package validation;
- output/manifest schema versions;
- known limitations;
- rollback instructions;
- claim wording.

---

## 19. Rollout and rollback

### 19.1 Rollout ladder

For visible behavior:

1. diagnostic-only evidence;
2. disabled feature flag;
3. synthetic/focused tests;
4. private development corpus;
5. sealed automatic test;
6. three-anchor native pilot;
7. small opt-in batch;
8. blinded acceptance;
9. default candidate;
10. release/package validation;
11. certified default.

Skip a rung only with an explicit written reason and equivalent evidence.

### 19.2 Rollback unit

Rollback must be no broader than the introduced behavior:

- support helper;
- color policy version;
- evidence schema adapter;
- parser challenger;
- operation material version;
- subject calibration;
- individual learned model;
- planner;
- Generative Studio mode.

Do not weaken exact support, identity, color, or evidence gates to keep a visually
preferred candidate.

### 19.3 Stop conditions

Stop rollout on:

- any critical identity/likeness/geometry/mask/color defect;
- unexplained outside-support residual;
- source/output overwrite ambiguity;
- invalid/incomplete evidence;
- model/provider/hash drift;
- private-data or cross-subject leakage;
- unreviewed license dependency;
- non-reproducible output where determinism is claimed;
- abnormal native memory/time beyond operator policy;
- human review blinding/integrity failure;
- provenance displayed more strongly than validated.

---

## 20. Owner decisions and decision gates

### Gate D0 — before FR-QC-0A

- authorize exact-support tranche and files;
- accept intended golden movement;
- identify overlapping dirty changes.

### Gate D1 — before FR-QC-0B

- choose/review ICC bytes and license;
- approve metadata/orientation/subsampling policy;
- approve C2PA passthrough/omission wording.

### Gate D2 — before private corpus work

- approve source/consent/license scope;
- choose labelers/adjudicator;
- set storage/access/retention/deletion;
- approve corpus size strategy without inventing thresholds.

### Gate D3 — before native pilot/batch

- approve exact source and sibling destination;
- approve overwrite refusal/recovery;
- set machine-specific RSS/time stop rules;
- approve reviewer count and certification wording.

### Gate D4 — before parser/model acquisition

- approve candidate and download;
- review code/weight/data/dependency licenses;
- approve network/external-service behavior;
- approve packaging/distribution intent.

### Gate D5 — before visible material enablement

- approve operation intent/default;
- accept calibrated gate and known limitations;
- approve human review plan;
- confirm rollback.

### Gate D6 — before subject intelligence

- approve job-local association semantics;
- approve retention/destruction policy;
- approve manual correction workflow;
- prohibit persistent identity reuse by default.

### Gate D7 — before external “best” claim

- approve locked baseline/corpus/competitor versions;
- pre-register preference/non-inferiority criteria;
- approve claim wording and limitations;
- require sealed automatic, human, delivery, and license evidence.

---

## 21. Prioritized ticket register

| Priority | Ticket | Depends on | Review unit | Release effect |
|---:|---|---|---|---|
| 1 | FR-QC-0A exact support | D0 | Mechanical code + tests | Removes known local-operation drift |
| 2 | FR-QC-0B color/metadata | D1 | Delivery code + asset + tests | Deterministic truthful SDR export |
| 3 | FR-QC-0C QA vector | 0A/0B interfaces | Evidence code + tests | Complete automatic evidence |
| 4 | FR-QC-0D runner/manifest | 0A–0C | Library + CLI + recovery tests | Safe resumable batches |
| 5 | FR-QC-0E mouth corpus tooling | D2 | Schema/tooling only | Enables calibration, no pixels |
| 6 | FR-QC-0F mouth gate | 0E + calibration | One behavior change | Tongue-safe lip support |
| 7 | FR-QC-0G pilot | 0A–0F + D3 | Operational evidence | Native proof |
| 8 | FR-QC-0H batch | 0G accepted | Operational delivery | Phase 0 seal |
| 9 | FR-QC-1A operation result | Phase 0 | Interface + adapters | Common evidence surface |
| 10 | FR-QC-1B representation ledger | 0B + 1A | Representation infrastructure | Prevents hidden conversion errors |
| 11 | FR-QC-1C soft posteriors | 1A/1B | Parser evidence only | Retains uncertainty |
| 12 | FR-QC-1D ontology/protection | 1C | Policy/evidence infrastructure | Protects long-tail/unknown regions |
| 13 | FR-QC-1E abstention | 1C/1D + corpus | Decision infrastructure | Calibrated apply/skip states |
| 14 | FR-QC-1F observation graph | 1A–1E | Incremental adapters | Separates evidence/intent/render |
| 15 | FR-QC-1G parser bake-off | 1C–1F + D4 | Benchmark decision | May retain or replace parser |
| 16 | FR-QC-2A mark ledger | Phase 1 | Evidence infrastructure | Stable identity features |
| 17 | FR-QC-2B evidence levels | Phase 1 | Claims/evidence | Authority proportional to input |
| 18 | FR-QC-2C material observations | 2A/2B | Diagnostic only | Unified material evidence |
| 19 | FR-QC-2D linear/reconstruction | 1B/2C | Spike/decision | Correct representation/material math |
| 20 | FR-QC-2E texture metrics | 2C | QA infrastructure | Detects wax/copy/false detail |
| 21 | FR-QC-2F material backoff | 2A–2E | Decision infrastructure | Safe attenuation/skip |
| 22 | FR-QC-2G visible operations | 2F + D5 | One operation each | Certified material improvements |
| 23 | FR-QC-3A mouth semantics | Phase 2 | Semantic infrastructure | Full optical mouth ownership |
| 24 | FR-QC-3B teeth | 3A | One operation | Natural bounded teeth correction |
| 25 | FR-QC-3C lips | 3A | One operation | Material-aware lip correction |
| 26 | FR-QC-3D eyes | Phase 2 | One region family | Optical eye correction |
| 27 | FR-QC-3E occluders/accessories | 1D + Phase 2 | Protection integration | Long-tail safety |
| 28 | FR-QC-3F harmony | 2C/2E | Diagnostic then behavior | Face/body/photo coherence |
| 29 | FR-QC-3G region certification | 3A–3F | Evidence/review | M2 seal |
| 30 | FR-QC-4A subject graph | M2 + D6 | Job-local evidence | Subject association |
| 31 | FR-QC-4B manual correction | 4A | Workflow/UI | Correctable clusters |
| 32 | FR-QC-4C stable evidence | 4A/4B | Diagnostic | Multi-frame mark/material stability |
| 33 | FR-QC-4D frame diagnosis | 4C | Diagnostic | Per-frame deviations |
| 34 | FR-QC-4E consistency review | 4D | Workflow/evidence | Batch outlier control |
| 35 | FR-QC-4F privacy lifecycle | 4A–4E | Storage/policy | Safe retention/destruction |
| 36 | FR-QC-4G shoot release | 4A–4F | Full integration/review | M3 seal |
| 37 | FR-QC-5A model registry | M3 + D4 | Governance | Candidate truth |
| 38 | FR-QC-5B benchmark harness | 5A | Test/evidence | Comparable task results |
| 39 | FR-QC-5C NAFNet audit | 5B | Candidate decision | Current learned path decision |
| 40 | FR-QC-5D L1/L2 candidates | 5A/5B | Candidate-by-candidate | Bounded learned evidence/operators |
| 41 | FR-QC-5E L3 candidates | 5A/5B | Candidate-by-candidate | Residual restoration decisions |
| 42 | FR-QC-5F high-res lab | 5C–5E | Failure/scale suite | Production feasibility |
| 43 | FR-QC-5G blinded acceptance | 5F | Human review | Quality/fidelity decision |
| 44 | FR-QC-5H final decisions | 5G + competitor bake-off | Release decision | M4 core seal |
| 45 | FR-QC-6A–6D planner | M4 | Optional separate phase | M5 |
| 46 | FR-QC-7A–7D studio | M4, separate D4/D7 | Optional separate product | M6 |

---

## 22. Definition of done

### 22.1 Ticket done

A ticket is complete only when:

- exact scope was authorized;
- implementation matches the approved contract;
- required focused/integration/real/mutation checks passed;
- documentation reflects actual behavior;
- known limitations are recorded;
- rollback is verified;
- no unrelated file was staged or changed intentionally;
- evidence artifacts are named and retained according to policy.

### 22.2 Phase done

A phase is complete only when:

- every required ticket is accepted or explicitly rejected/deferred without violating
  the phase outcome;
- phase-level automatic and human gates pass;
- required corpus/calibration records are frozen;
- package/runtime/delivery status is stated precisely;
- rollback to the prior phase remains available;
- the closeout lists implemented, tested, human-accepted, release-validated, and
  still-unimplemented items separately.

### 22.3 Release done

A release is complete only when:

- the intended milestone is named;
- source revision/diff scope is reproducible;
- runtime and actual face-aware capability pass;
- model and asset records are valid;
- output/manifest schemas validate;
- automatic and human evidence is sealed;
- delivery files decode with correct profiles/metadata;
- known limitations and rollback are published;
- no critical defect remains.

### 22.4 “Highest face retouch we ever built” done

The claim is permitted only for `M4_BEST_MEASURED_RETOUCH` after:

- comparison against the current best certified Retouch path;
- locked corpus and current competitor bake-off;
- pre-registered paired blinded preference target;
- non-inferiority on every hard fidelity/safety pillar;
- no critical identity, likeness, geometry, mask, mouth, eye, color, object, privacy,
  or delivery defect;
- calibration without sealed-test leakage;
- model/data/license approval for shipped candidates;
- reproducible source, recipe, runtime, provider, model, and export records;
- claim wording names corpus/version and limitations.

It must not be marketed as globally state of the art without an independent,
comparable current benchmark.

---

## 23. Explicit current completion boundary

### Complete now

- completed research translated into a finite implementation program;
- Phase 0–7 dependencies and optional boundaries;
- milestones and the core completion definition;
- 46 prioritized ticket/review units;
- proposed module/file ownership;
- ready, done, rollout, rollback, and stop rules;
- test/mutation/human/release evidence strategy;
- corpus, competitor, standards, privacy, and model-governance workstreams;
- owner decision gates;
- the first authorization decision; and
- `FR-QC-0A` implementation plus focused automatic and agent visual review evidence.

### Not complete now

- no owner visual acceptance or `FR-QC-0A` release seal;
- no commit, staging, push, package, or release;
- no `FR-QC-0B` or later implementation;
- no GUI, runner, model, asset, manifest schema, or packaging implementation;
- no private corpus collection/annotation;
- no thresholds or morphology values;
- no model or dataset acquisition;
- no native pilot or batch rerender;
- no competitor bake-off;
- no complete full-suite result or independent human acceptance result;
- no release or milestone certification.

### Recommended next-session start

Before authorizing another engineering tranche, review and accept or reject the retained
`FR-QC-0A` mouth evidence. If accepted, the next separately authorizable tranche is:

```text
FR-QC-0B — canonical SDR color and metadata delivery
```

Do not start `FR-QC-0B` merely because `FR-QC-0A` tests pass. Recheck the active
worktree, obtain the `D1` profile/metadata decisions, confirm the allowlist, and restate
the color/delivery completion boundary first.

---

## 24. Final handoff

Research is complete for the current baseline, and implementation planning is now
complete. The program has a finite core endpoint, explicit optional products, narrow
authorization units, and evidence-based stop/rollback rules.

The project can pause cleanly here. The next action is owner review of the retained
`FR-QC-0A` visual evidence. Only after acceptance and separate authorization should work
move to `FR-QC-0B`; no broader face-engine rewrite is implied.
