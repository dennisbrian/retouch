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
- `RESEARCH_NEXT_FEATURE_FRONTIER_2026_08_12.md` — Post-Advanced-Retouch workflow and quality frontier research
- `RESEARCH_PROJECT_SHOOT_WORKFLOW_DELTA_2026_08_14.md` — Current implementation re-baseline and next Project/Shoot workflow gates
- `RESEARCH_SHOOT_REVIEW_NEXT_TRANCHE_2026_08_14.md` — Watch-job truth contract, face evidence, and XMP interoperability research
- `RESEARCH_FACE_QUALITY_EVIDENCE_V1_2026_08_14.md` — Transparent face/eye evidence contract and corpus-certification gates
- `RESEARCH_DETECTION_RECALL_2026_08_19.md` — Detection recall/precision on the 83-image DSCF corpus; subject 98.8% (→100% via `8811320`); 18 poster-FP exposure; RetinaFace inert in pinned env
- `RESEARCH_POSTERFP_VETO_2026_08_19.md` — Three-discriminator veto validation (joint person+texture rule kills 14/18 FPs, 0 collateral); texture-only veto proven unsafe; RetinaFace dependency dead-end proof
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
- `TEST_REPORT_*.md` — Test results
- `session/` — Session progress logs

## Improvements (`docs/improvements/`)
- Feature enhancement proposals
- Technical spike notes

## Contributing
See `docs/CONTRIBUTING.md` for dev workflow
