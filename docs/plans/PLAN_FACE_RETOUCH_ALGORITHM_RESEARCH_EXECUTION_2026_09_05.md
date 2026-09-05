# Plan — Face Retouch Algorithm Research Execution

**Date:** 2026-09-05

**Status:** `planned` — documentation-first; no implementation, model acquisition,
or photo testing is authorized by this document.

**Research basis:** [Face retouch algorithm research](RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md)

**Program basis:** [Face Retouch Quality-Ceiling Implementation Plan](PLAN_FACE_RETOUCH_QUALITY_CEILING_IMPLEMENTATION_2026_09_01.md)

## Decision for the first work item

Start with **FA-01: spatial mask evidence, boundary ownership, and mark
protection**. Treat **FA-02: selective texture restoration** as the second
package, designed in parallel but evaluated only after the support and protection
contract is clear.

This order follows the existing roadmap's Phase 1 → Phase 2 dependency. A mask
or ownership error can contaminate smoothing, texture restoration, blemish repair,
and under-eye correction at once. Measuring texture or comparing repair algorithms
before that boundary is understood would make the result difficult to interpret.

Do not begin with a parser replacement, a learned repair model, or a new general
smoothing algorithm. The current research has not shown that the existing parser
or smoothing baseline is the limiting factor on the owner's photographs.

## Scope of this planning step

This step changes documentation only. It does not authorize changes under
`retouch/`, `tests/`, `scripts/`, `models/`, recipes, generated outputs, or private
photo folders. It also does not authorize downloading weights or dependencies,
running photo renders, selecting production thresholds, or changing default
behavior.

The unrelated working-tree edits already present in the repository remain outside
this plan. A future tranche must record `HEAD`, status, and an exact file allowlist
before it begins.

## Documentation to complete before any future code

The following decisions must be written down and reviewed first:

| Documented item | Required decision |
|---|---|
| **Face and material labels** | Skin, hair, lashes/eyes, lips/makeup, accessories, facial hair, stable marks, temporary blemishes, background/poster faces, and unknown regions. Unknown remains protected. |
| **Support ownership** | Which operations may read and change each class; where feathering ends; how protected marks and makeup are carried through smoothing, tone, blemish, under-eye, and texture stages. |
| **Evidence fields** | Hard label, spatial posterior or margin, boundary uncertainty, edit eligibility, blend alpha, parser/provider truth, and explicit skip/fallback state. These fields must not be collapsed into one confidence scalar. |
| **Corpus protocol** | Reviewed examples covering frontal, three-quarter, profile, small-face, lighting, makeup, glasses, wigs, facial hair, marks, occlusion, and multiple-face cases. Keep near-duplicate frames together and preserve separate development, calibration, and held-out review sets. |
| **Evaluation contract** | Boundary leakage, useful edit coverage, protected-mark preservation, abstention, native-resolution crop review, texture observability, and runtime records. Do not set numerical thresholds before labels exist. |
| **Decision record** | The possible outcomes are retain current, use a challenger as secondary evidence, adopt behind migration, reject, or defer for missing evidence. |
| **Provenance register** | Candidate model/code/weight/data/dependency sources, licenses, runtime assumptions, and availability. A public repository is not implementation readiness. |

The first future diagnostic tranche may begin only when these definitions are
reviewed and the exact corpus, output artifacts, and stop conditions are named.

## Ordered research queue

### 1. FA-01 — mask and ownership evidence

First document and later measure:

- whether current spatial parser margins/posteriors identify uncertain boundaries;
- whether hair, makeup, accessories, lashes, facial hair, and stable marks are
  protected consistently across operations;
- whether unknown or insufficiently supported regions produce a safe abstention;
- whether a candidate parser improves critical boundary errors at comparable useful
  edit coverage, rather than only improving an aggregate segmentation score.

The first future implementation, if separately authorized, should be diagnostic
only: retain the current hard-mask behavior and expose evidence without changing
rendered pixels. A parser challenger belongs after the ontology, corpus, metrics,
runtime harness, and provenance record exist.

### 2. FA-02 — texture preservation evidence

After the support contract is stable, compare the current restoration signal with
band-selective restoration that excludes approved defect supports. Keep smoothing,
masks, strength, and all other face operations fixed.

The review must distinguish photographed pores and fine hair from noise,
compression, makeup, facial form, and restored defects. Include restoration
disabled as an ablation. A high-frequency energy increase is not evidence of
authentic texture by itself.

### 3. FA-03 and FA-04 — blemish detection and repair as separate factors

Use manually accepted spot masks first. Compare repair methods on the same masks
before reintroducing automatic detection. Then run the detector × repair matrix so
an apparent improvement can be attributed to localization, filling, or both.

Use protected marks, makeup, shadows, and broad redness as separate labels. Keep
Telea as the small-defect baseline, restrict PatchMatch donors by same-face region,
scale, and lighting, and treat learned or generative repair as a later comparison.

### 4. FA-05 and FA-06 — under-eye reference and tone/material interpretation

Evaluate reference-ring validity, lash/makeup exclusions, natural residual shadow,
and the distinction between uneven tone, shine, form, and cast shadow. Preserve an
explicit skip outcome when a valid reference is unavailable.

### 5. FA-07 — bounded learned assistance

Only after the classical evidence and provenance work is complete, compare bounded
learned tone proposals or repair residuals. Source pixels remain render authority;
support is locked outside the model; transforms and residuals are bounded; and the
deterministic path remains available. Full learned or generative retouch is a
separate, later decision.

## Exit criteria for the documentation phase

This phase is complete when:

- FA-01 is the single named first research package;
- labels, ownership, evidence fields, corpus splits, metrics, and decision outcomes
  are written down;
- FA-02's texture exclusions and observability rules are defined;
- the future file allowlist, runtime assumptions, and rollback boundary are stated;
- no unresolved policy question silently becomes an algorithm threshold; and
- the plan still states that no code, model download, recipe change, render, or
  photo test has occurred.

The next decision after feedback should authorize at most one diagnostic tranche
for FA-01. It should not authorize a parser swap or visible retouch behavior.

