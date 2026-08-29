# Documentation Organization

## Root Level (Keep Short)
- `README.md` — Installation & quick start
- `CLAUDE.md` — Development guidelines & architecture

## Guides (`docs/guides/`)
- `RECIPE_GUIDE.md` — Creating custom presets
- `GUI.md` — Gradio UI layout & components
- `BATCH_GUIDE.md` — Batch processing via CLI
- `PERFORMANCE_TUNING.md` — Optimization tips
- `COLOR_DELIVERY_TRUTH.md` — Working-space, precision, preview, and export contracts
- `TROUBLESHOOTING.md` — Common issues & fixes
- `RECIPE_GENERATOR.md` — Recipe generation
- `RUNTIME_RECOVERY.md` — Isolated legacy recovery and Python 3.12 migration gates
- `CERTIFICATION_EVIDENCE_V2.md` — Versioned render, detector, quality, corpus, and review evidence contract

## Architecture (`docs/architecture/`)
- `ARCHITECTURE.md` — Pipeline design & modules
- `API.md` — Python API reference
- `PIPELINE_FLOW.md` — Data flow diagram

## Film / color (`docs/`)
- `FUJI_SIMS_GUIDE.md` — Fuji film simulation recipes
- `RECIPE_SWEEP.md` — Single-photo recipe sweeps and folder visual QA

## Plans (`docs/plans/`)
- `MASTER_PLAN.md` — Overall roadmap
- `PLAN_NEXT_FEATURE_IMPROVEMENTS_2026_08_12.md` — Productization-first next-feature plan
- `RESEARCH_SMART_WORKSPACE_PRODUCTIZATION_2026_08_29.md` — Priority-ranked Smart/Classic workspace plan covering intent contracts, contextual editing, render lifecycle, safety, delivery, and staged implementation gates
- `RESEARCH_NEXT_FEATURE_FRONTIER_2026_08_12.md` — Post-Advanced-Retouch workflow and quality frontier research
- `RESEARCH_PROJECT_SHOOT_WORKFLOW_DELTA_2026_08_14.md` — Current implementation re-baseline and next Project/Shoot workflow gates
- `RESEARCH_SHOOT_REVIEW_NEXT_TRANCHE_2026_08_14.md` — Watch-job truth contract, face evidence, and XMP interoperability research
- `RESEARCH_FACE_QUALITY_EVIDENCE_V1_2026_08_14.md` — Transparent face/eye evidence contract and corpus-certification gates
- `RESEARCH_DETECTION_RECALL_2026_08_19.md` — Detection recall/precision on the 83-image DSCF corpus; subject 98.8% (→100% via `8811320`); 18 poster-FP exposure; RetinaFace inert in pinned env
- `RESEARCH_POSTERFP_VETO_2026_08_19.md` — Three-discriminator veto validation (joint person+texture rule kills 14/18 FPs, 0 collateral); texture-only veto proven unsafe; RetinaFace dependency dead-end proof
- `RESEARCH_EYE_OCCLUSION_2026_08_26.md` — Per-eye occlusion gate calibration protocol (EAR primary, tone-adaptive contrast secondary, hair tertiary); stratified BiSeNet-available/unavailable arms; Fitzpatrick I-III/IV-VI split; cost asymmetry (false gate mild / missed gate catastrophic)
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
- `CORE_RECIPE_CERTIFICATION_2026_08_12.md` — 12-recipe automatic and human review gate
- `METAMORPHIC_ROBUSTNESS_LAB.md` — Cross-cutting input-variant robustness gate and Advanced Retouch evidence runner
- `REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md` — 15-agent review of the eye-occlusion gate + BiSeNet CPU pin; found P0 sclera no-op / P1 handedness swap (both now fixed), 78% corpus over-gating by mask-area signals (gate rewritten to EAR+contrast), vacuous tests (replaced with mutation-tested suite)
- `TEST_REPORT_*.md` — Test results
- `session/` — Session progress logs

## Improvements (`docs/improvements/`)
- Feature enhancement proposals
- Technical spike notes

## Contributing
See `docs/CONTRIBUTING.md` for dev workflow
