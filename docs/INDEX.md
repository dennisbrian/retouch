# Documentation Organization

## How to Use Retouch

Start with the [Getting Started guide](guides/GETTING_STARTED.md) for the
desktop workflow: launch the local GUI, retouch one photo, export a full-quality
result, or switch to the CLI and recipe-sweep workflow. Then use the focused
guides below for batch processing, recipe authoring, runtime recovery, and
output verification.

## Root Level (Keep Short)
- `README.md` — Installation & quick start
- `CLAUDE.md` — Development guidelines & architecture
- `docs/REPO_STATS.md` — Generated repo size truth (modules, tests, LOC); refreshed by `scripts/dev/governance --write-stats`
- `docs/GOVERNANCE.md` — Repository governance policy + tooling
- `docs/ENGINEERING_CONTROL_PLANE.md` — Multi-agent task queue, leases, planner & circuit breaker (`scripts/dev/control-plane`)
- `docs/QUALITY_LAB.md` — Autonomous visual regression & performance lab (`scripts/dev/quality-lab`)

## Guides (`docs/guides/`)
- `GETTING_STARTED.md` — Desktop one-photo workflow, CLI basics, and result checks
- `RECIPE_GUIDE.md` — Creating custom presets
- `GUI.md` — Gradio UI layout & components
- `BATCH_GUIDE.md` — Batch processing via CLI, including social crops export
- `SPLIT_SHOOT.md` — Split a shoot into folders by capture time before batching
- `GROUP_BY_COSPLAYER.md` — Group a shoot by cosplayer (costume colours + capture time, no face recognition)
- `DUPLICATES.md` — Find repeated shots of the same pose anywhere in a shoot and suggest keepers
- `PERFORMANCE_TUNING.md` — Optimization tips
- `COLOR_DELIVERY_TRUTH.md` — Working-space, precision, preview, and export contracts
- `CONTENT_CREDENTIALS.md` — Per-photo edit reports and signing outputs with Content Credentials (C2PA)
- `TROUBLESHOOTING.md` — Common issues & fixes
- `RECIPE_GENERATOR.md` — Recipe generation
- `RUNTIME_RECOVERY.md` — Isolated legacy recovery and Python 3.12 migration gates
- `CERTIFICATION_EVIDENCE_V2.md` — Versioned render, detector, quality, corpus, and review evidence contract

## Architecture (`docs/architecture/`)
- `ARCHITECTURE.md` — Pipeline design & modules
- `API.md` — Python API reference
- `PIPELINE_FLOW.md` — Data flow diagram (2026-06-23 Mermaid) + code-derived behavioural map of stage/op order (2026-09-02)

## Film / color (`docs/`)
- `FUJI_SIMS_GUIDE.md` — Fuji film simulation recipes
- `RECIPE_SWEEP.md` — Single-photo recipe sweeps and folder visual QA

## Plans (`docs/plans/`)
- `MASTER_PLAN.md` — Overall roadmap
- [Cosplay tutorial batch review — 2026-09-24](plans/RESEARCH_COSPLAY_TUTORIAL_BATCH_REVIEW_2026_09_24.md)
  — Documentation-only synthesis of 249 photographed Photoshop tutorial pages,
  corrected cross-batch continuity, face-relevant evidence, and proposed
  source/output preservation study
- [Cosplay tutorial face evidence ledger — 2026-09-24](plans/RESEARCH_COSPLAY_TUTORIAL_FACE_EVIDENCE_LEDGER_2026_09_24.md)
  — Working translations and confidence notes for face-related screenshots,
  repeated-page grouping, and a proposed evidence gate; no photo experiment run
- [Proposed cosplay tutorial transfer study — 2026-09-24](plans/PLAN_COSPLAY_TUTORIAL_TRANSFER_STUDY_2026_09_24.md)
  — Documentation-only photo-pair protocol, separated edit intents, preservation
  review sheet, and decision gates; no photos processed
- [Cosplay tutorial current-code crosswalk — 2026-09-25](plans/RESEARCH_COSPLAY_TUTORIAL_CURRENT_CODE_CROSSWALK_2026_09_25.md)
  — Read-only map of tutorial observations to the inspected Retouch branch,
  recipe and mark-policy boundaries, and remaining study evidence; no renders
  or photo comparisons performed
- [Cosplay tutorial compositing review — 2026-09-25](plans/RESEARCH_COSPLAY_TUTORIAL_COMPOSITING_2026_09_25.md)
  — Masks, depth, occlusion, reflected light, and the boundary between existing
  background polish and full asset compositing; no photo experiment run
- [Cosplay tutorial Google Drive Photography inventory — 2026-09-25](plans/RESEARCH_COSPLAY_TUTORIAL_DRIVE_PHOTO_INVENTORY_2026_09_25.md)
  — Candidate matched folders and filename overlap; no image-level review
- [QA flag calibration — 2026-09-25](plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md)
  — Differential (vs input photo) recalibration of the batch-review QA
  detectors: banding / plastic_skin / seam / asymmetry no longer flag 100% of
  clean renders; color_drift tail gate p99>10° → p95>20°
- [Person-gate false-negative on bright wigs — 2026-09-25](plans/RESEARCH_PERSON_GATE_WIG_FALSENEG_2026_09_25.md)
  — Status report (not fixed here) on the person-mask coverage gate dropping
  real faces under near-white wigs; led to the face-skin rescue second opinion
- [Over-smoothing check, patch-scale QA — 2026-09-27](plans/RESEARCH_OVER_SMOOTHING_CHECK_2026_09_27.md)
  — plastic_skin and color_drift now also scan cheek-sized windows per face so
  one waxy cheek or a partial-face cast is caught; 0 new flags on clean renders
- [Closed-eye flag on glasses, darker skin, bursts — 2026-09-27](plans/RESEARCH_CLOSED_EYE_CHECK_2026_09_27.md)
  — Misfire-free calibration of the closed-eye flag and burst blink check,
  plus the two burst misfires fixed in `shoot_intelligence.rank_burst_candidates`
- [Retouch target identity and preview/export agreement — 2026-09-23](plans/RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23.md)
  — Cache coordinate audit and follow-up fixes for preview inspection, per-face
  target association, and explicit manual rebase; full photo qualification pending
- [Proposed target and preview/export validation — 2026-09-23](plans/PLAN_RETOUCH_TARGET_AND_PREVIEW_VALIDATION_2026_09_23.md)
  — Focused cold/warm cache, target, and rebase contract checks completed;
  public GUI-to-engine controls and independent photo review remain pending
- [CLI photo input research — 2026-09-22](plans/RESEARCH_CLI_PHOTO_INPUT_2026_09_22.md)
  — Input-selection and dry-run audit plus bounded implementation: shared input
  plan, multiple-photo/filtering workflow, decoder checks, RAW+JPEG policy,
  hash-verified reruns, and opt-in resource estimation; no visual-quality claim
- [Retouch protection across operation boundaries — 2026-09-22](plans/RESEARCH_RETOUCH_PROTECTION_LIFECYCLE_2026_09_22.md)
  — Documentation-only repair-mask containment, donor/fallback constraints,
  cross-stage mark decisions, restoration reversing intentional corrections,
  and combined under-eye preservation; source-backed risks and proposed controls
- [Retouch QA validity research — 2026-09-21](plans/RESEARCH_RETOUCH_QA_VALIDITY_2026_09_21.md)
  — Post-reference-implementation source review: geometry/skin-support validity,
  reference loss versus back-off, mixed-scope observations, missing-measurement
  semantics, delivery coverage, and bounded correction-map literature
- [Proposed Retouch QA validation study — 2026-09-21](plans/PLAN_RETOUCH_QA_VALIDATION_2026_09_21.md)
  — Documentation-only stage-pair contract, discriminating controls, preservation
  pilot, grouped calibration, harm/benefit/abstention endpoints, and promotion gates;
  no implementation or experiments authorized or performed
- [Retouch improvement research — 2026-09-21](plans/RESEARCH_RETOUCH_IMPROVEMENTS_2026_09_21.md)
  — Research-only follow-up against newer local main: QA reference boundaries,
  EXR/color delivery findings, preservation evidence, additive correction-map
  literature, and proposed study order; no code changes or photo experiments
- [Feature scan: what to build next — 2026-09-24](plans/RESEARCH_FEATURE_SCAN_2026_09_24.md)
  — Outward research (competitors, cosplay needs, social output, culling, local
  models and licenses, labeling rules, video) ranked into 18 candidate features
- [Proposed P9: support usability and selective retouch research — 2026-09-11](plans/RESEARCH_P9_SELECTIVE_RETOUCH_PROPOSAL_2026_09_11.md)
  — Provisional topic after P8; current-code and six-primary-source review of
  evidence calibration, accepted-region errors, coverage, and abstention
- [Proposed P9 research protocol — 2026-09-11](plans/PLAN_P9_SELECTIVE_RETOUCH_RESEARCH_2026_09_11.md)
  — P8 support-usability pilot followed by independent policy qualification
  and a separate P7 ownership/render study; no experiments run or roadmap adoption
- [P8 apparent-age cue validity research — 2026-09-10](plans/RESEARCH_P8_CUE_VALIDITY_2026_09_10.md)
  — Primary-source and current-code review correcting the earlier P8 drafts;
  separates image descriptors, apparent-age associations and perceptual editing
- [P8 revised research protocol — 2026-09-10](plans/PLAN_P8_CUE_VALIDITY_RESEARCH_2026_09_10.md)
  — Proposed support, repeatability, labeling and intervention gates; research
  only, with no code or photo experiments executed
- P8 R1 implementation — `retouch/aging_cues.py` and
  `scripts/qa/p8_cue_readout.py` provide measurement-only observable cue
  readouts; no age slider or render-path coupling
- [Face retouch algorithm research — 2026-09-05](plans/RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md)
  — Documentation-only current-code review, thirteen classical/learned algorithm
  comparisons, and seven proposed experiments for masks, blemish repair, texture,
  under-eyes, tone, and profile/occlusion handling
- [Face retouch algorithm research execution plan — 2026-09-05](plans/PLAN_FACE_RETOUCH_ALGORITHM_RESEARCH_EXECUTION_2026_09_05.md)
  — Active phased engineering roadmap: FA-01 support & smoothing mark protection
  landed/shipped, followed by structured specifications for selective texture
  restoration (FA-02), 2-factor blemish matrix (FA-03/04), reference/material
  decomposition (FA-05/06), and bounded learned assistance (FA-07)
- [FA-01 protection-consistency and abstention-observability audit — 2026-09-05](plans/RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md)
  — Historical smoothing-protection gap, followed by implemented guided-engine
  protection (§1.1b) and abstention logging; non-guided-engine limits and
  accessory/facial-hair evidence gaps remain explicit
- `RESEARCH_COLOR_SCIENCE_2026_09_04.md` — Current-tree color-science audit and
  engineering contract: SDR sRGB baseline, ICC/RAW/high-bit/alpha findings,
  operation-domain rules, conformance matrix, and staged P3/HDR gates
- `RESEARCH_DARK_CIRCLE_OP_2026_09_02.md` — The `dark_circles` / `undereye.darken_removal`
  op was inert on every face (detector locked onto the lash line, lift double-capped);
  v2 redesign (tear-trough support, eye/lash exclusion, relative low-pass darkness,
  texture-preserving lift) calibrated on 146 corpus eyes and verified through the real
  dispatch; harnesses `scripts/qa/dark_circle_op_study.py` / `_summarize.py` /
  `dark_circle_engine_check.py`. Makeup-vs-shadow and darker-skin caveats.
- `RESEARCH_POST_EPSILON_FACEOP_REAUDIT_2026_09_02.md` — 83-face re-audit of the
  skin-composite ops after the mask-epsilon fix (`f69ab1e`): affected faces now match
  controls at the default recipe (no over-strength), the shared yaw ramp was found
  inverted (relight/sculpt ~0 at ratio 2.51, ~full at 3.99) and the neck "depth gate"
  was zeroing neck harmonisation on every frontal face; both fixed. Also under-eye
  detector fire-rate and neck-op fabric/hand patches.
- `RESEARCH_YAW_GATE_CALIBRATION_2026_09_02.md` — Corpus calibration of the
  nose-bridge/temple yaw gate (band 1.3→1.6 / 1.5→1.7 → 2.5→4.0, shipped `9c26493`)
  plus the composite-mask epsilon bug it uncovered (skin ops discarded on 24% of
  portraits, shipped `f69ab1e`); harnesses `scripts/qa/yaw_gate_sweep.py` / `_render.py`
- `PLAN_FACE_RETOUCH_QUALITY_CEILING_IMPLEMENTATION_2026_09_01.md` — Final
  documentation-only Phase 0–7 implementation roadmap with finite core milestones,
  46 review units, dependencies, proposed file ownership, owner gates, corpus/model/
  standards workstreams, tests, evidence, rollout, rollback, and exact completion
  boundaries
- `RESEARCH_FACE_RETOUCH_QUALITY_CEILING_2026_09_01.md` — Two-pass current-tree,
  primary-source face-retouch quality-ceiling research; defines the Photographic
  Truth Core, Bounded Learned Assist, separate Generative Studio, facial-appearance
  evidence levels, occlusion/material ownership, calibrated abstention, competitor
  and standards research, corpus/mutation gates, phased engineering, and explicit
  authorization/completion boundaries
- `PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md` — Documentation-only full_v2
  engineering specification covering exact-support mouth operations, deterministic
  colour delivery, complete QA evidence, atomic/resumable batching, calibrated
  tongue-safe masks, tests, rollback, and explicit authorization/completion gates
- `PLAN_NEXT_FEATURE_IMPROVEMENTS_2026_08_12.md` — Productization-first next-feature plan
- `RESEARCH_SMART_WORKSPACE_PRODUCTIZATION_2026_08_29.md` — Priority-ranked Smart/Classic workspace plan covering intent contracts, contextual editing, render lifecycle, safety, delivery, and staged implementation gates
- `RESEARCH_NEXT_FEATURE_FRONTIER_2026_08_12.md` — Post-Advanced-Retouch workflow and quality frontier research
- `RESEARCH_PROJECT_SHOOT_WORKFLOW_DELTA_2026_08_14.md` — Current implementation re-baseline and next Project/Shoot workflow gates
- `RESEARCH_SHOOT_REVIEW_NEXT_TRANCHE_2026_08_14.md` — Watch-job truth contract, face evidence, and XMP interoperability research
- `RESEARCH_FACE_QUALITY_EVIDENCE_V1_2026_08_14.md` — Transparent face/eye evidence contract and corpus-certification gates
- `RESEARCH_DETECTION_RECALL_2026_08_19.md` — Detection recall/precision on the 83-image DSCF corpus; subject 98.8% (→100% via `8811320`); 18 poster-FP exposure; RetinaFace inert in pinned env
- `RESEARCH_POSTERFP_VETO_2026_08_19.md` — Three-discriminator veto validation (joint person+texture rule kills 14/18 FPs, 0 collateral); texture-only veto proven unsafe; RetinaFace dependency dead-end proof
- `RESEARCH_EYE_OCCLUSION_2026_08_26.md` — Per-eye occlusion gate calibration protocol (EAR primary, tone-adaptive contrast secondary, hair tertiary); stratified BiSeNet-available/unavailable arms; Fitzpatrick I-III/IV-VI split; cost asymmetry (false gate mild / missed gate catastrophic)
- `RESEARCH_MEITU_RETOUCH_BAKEOFF_2026_09_04.md` — Reproducible same-source
  five-pair Meitu-app vs current `natural` / `convention_clear_v1` bake-off;
  face-aware manifests, registered metrics, export confounds, current competitor and
  preference research, and randomized owner-review sheets
- `RESEARCH_MEITU_COMPETITOR_QA_2026_08_29.md` — 5-pair diff study of Meitu-app output vs. DSCF source (izunako cosplay shoots, kimono + bunny costume); landmark-registered reruns show edit intensity varies substantially per-shot (heavy smoothing+reshape on one shoot, light touch/no-reshape on another) — likely manual per-shot tuning, not a fixed filter; retracts an earlier unregistered-crop "costume clarity boost" claim; observational pilot, no thresholds/owner set
- `PLAN_*.md` — Tier-based initiatives
- `PHOTOSHOP_PARITY_ROADMAP.md` — Feature parity goals
- `V1_PLAN.md`, `TASK_P2_PERF_GUARDS.md` — Historical plans

## Reference (`docs/reference/`)
- `FRECKLE_*.md` — Freckle removal implementation
- `P1_IMPLEMENTATION_NOTES.md` — Dev notes
- `PROGRESS.md`, `ROADMAP.md` — Status tracking
- `AGENTS.md` — Agent references

## Review & QA (`docs/review/`)
- `AUDIT_REPORT.md` — Security & coverage audit
- `REVIEW_AMGDAY32026_RETOUCH_BATCH_2026_09_01.md` — Documentation-only visual,
  delivery, and deep-research closeout for the 22-photo Desktop batch; records measured
  full/draft behaviour, colour/export and QA gaps, tongue-mask corpus evidence, and the
  prioritized implementation/completion contract
- `CORE_RECIPE_CERTIFICATION_2026_08_12.md` — 12-recipe automatic and human review gate
- `METAMORPHIC_ROBUSTNESS_LAB.md` — Cross-cutting input-variant robustness gate and Advanced Retouch evidence runner
- `REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md` — 15-agent review of the eye-occlusion gate + BiSeNet CPU pin; found P0 sclera no-op / P1 handedness swap (both now fixed), 78% corpus over-gating by mask-area signals (gate rewritten to EAR+contrast), vacuous tests (replaced with mutation-tested suite)
- `TEST_REPORT_*.md` — Test results
- `session/` — Session progress logs
- `archive/` — Historical snapshots (e.g. `MASTER_PLAN_2026-07-02.md`)
- `HOUSEKEEPING_AUDIT_2026_09_17.md` — Repo housekeeping audit
- `MONTHLY_CHANGELOG_2026_09.md` — September feature/fix changelog
- `DOCUMENTATION_AUDIT_2026-07-15.md` — Documentation link/staleness audit

## Update (`docs/update/`)
- `IMPROVEMENTS.md` — Running improvement notes

## Improvements (`docs/improvements/`)
- Feature enhancement proposals
- Technical spike notes

## Contributing
See `docs/CONTRIBUTING.md` for dev workflow
- `docs/EXPERIMENT_LAB.md` — Frozen experiment designs, isolated candidates, Quality Lab evidence, and promotion (`scripts/dev/experiment`)
