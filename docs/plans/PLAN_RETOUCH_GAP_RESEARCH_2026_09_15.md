# Next Retouch research tranche

Date: 2026-09-15. Proposed study sequence, not an implementation authorization.
Basis: [remaining-gap report](RESEARCH_RETOUCH_REMAINING_GAPS_2026_09_15.md).

## First deliverable

A reviewed development inventory and a preservation/support baseline using
existing FA-01, FA-03, and P8 definitions. Answer: where does today's Retouch
choose the wrong material, and which mistakes actually matter to the requested edit?

Reuse [FA-03 research](RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md), its
[existing experiment](RESEARCH_FA03_CLASSIFICATION_EXPERIMENT_2026_09_05.md), and
the [proposed support-usability protocol](PLAN_P9_SELECTIVE_RETOUCH_RESEARCH_2026_09_11.md).
Do not treat the earlier classifier's pre-fix baseline as today's production code.
The P9 label remains provisional; this plan does not change the P1–P8 roadmap.

## Proposed sequence and exit decisions

| Step | Work | Required output and decision |
|---|---|---|
| 1. Inventory and labels | Reuse saved development assets; locate missing real marks, occlusion, groups, low-light/noise and broader observed tone conditions | Source hashes, reviewed subject/shoot grouping, per-region labels, missing-strata table, disagreements; distinguish available photos from usable evidence |
| 2. Freeze current baseline | Record production code and experimental revisions, supports, class outcomes, P8 availability, score provenance and explicit unknowns | A reproducible table of baseline mistakes with denominators; old tuned examples remain development data |
| 3. Preservation/support pilot | Compare existing FA-03 arms with current production; independently compare P8 evidence rankings with support size alone | Class errors, support usability, useful coverage, false rejection and unknown-label rates; no winner from abstention frequency alone |
| 4. Texture pilot | With reviewed supports fixed, compare restoration off, legacy and existing FA-02 arms; isolate resize/noise/compression effects | Useful detail retained, corrected defects reintroduced, protected-region change, noise gain, seams and native visual review |
| 5. Ownership and delivery | Separately qualify P7 target support; compare preview and native export decisions and evidence records | Wrong-person/material overlap, target/disposition agreement, visible reasons, worker-path evidence; do not force identical pixel output across resolutions |
| 6. Independent qualification | Freeze any promising candidate, then evaluate on separately reserved subjects/shoots | Error-versus-coverage results with uncertainty and missing strata; report no qualified policy if evidence is inadequate |
| 7. Preference | Blind source/current/candidate presentation under matched export conditions | Owner preferences by intended look, preservation vetoes, ties and runtime/correction effort if actually measured |

Steps 1–3 are the first bounded tranche. Later steps are conditional and can
return a negative result without implying a failed study. P7 qualification is
independent of P8 support calibration; neither certifies downstream edit harm.

## Minimum useful development pilot

Start with 12–20 reviewed portraits across at least six recorded subjects and
three shoots as a practical planning target, not a statistical sufficiency claim.
Expand beyond that target when needed to cover the following cases:

- Real natural marks, including near-eye marks; deliberate liner/face paint.
- Fine hair or beard, skin texture, and genuinely smooth or soft source regions.
- Wig/glasses/hand/prop occlusion and difficult facial boundaries.
- Group portraits and a background person or printed-face adversary.
- Small faces, mixed light, darker image tones, noisy or compressed captures.

Do not label absent material by filename or infer persistent identity from face
appearance. Existing reviewed mappings can be reused. New subject/shoot mappings
and intended preservation decisions must have actual provenance. Preserve disputed
labels as unknown. Independently review a subset and report disagreement.

This pilot can identify failures and compare hypotheses. It cannot certify a low
harm rate or populate an untouched test set from images already used in research.
Choose qualification sample size later from predeclared error tolerance, useful
coverage, independent sampling units and desired uncertainty. Region/pixel counts
and burst frames are not independent people.

## Fair comparison rules

1. Freeze candidate generation for classifier comparisons; include a separate
   inventory of missed annotated marks so candidate-only precision cannot hide recall.
2. Compare repair methods using the same manually accepted defect supports before
   comparing detector plus repair end to end.
3. Keep mask identity, permission to edit, and blend strength separate.
4. Distinguish fixed-support descriptor changes from rerun-parser support drift.
5. For texture, a face-scale change can change resolvable information; seek a
   justified eligibility response, not numerical invariance at any cost.
6. Keep source, input color contract, requested strength, working resolution and
   export settings fixed within each comparison. Use native views for fine detail.
7. Report wrong decisions among accepted cases together with accepted coverage,
   usable cases rejected, unknowns, and stress-condition breakdowns.
8. Preserve requested tone/texture correction as a benefit endpoint. An unchanged
   image can be safe but unhelpful; it must not automatically win.

## Suggested result bundle

- Inventory and versioned annotation rubric, including missing strata.
- Machine-readable evidence rows with source hash, code/model provenance,
  person/shoot/split, operation, masks, raw scores, outcome and reason.
- Reviewed failure examples and source/current/candidate views where a render
  study is authorized and actually run.
- Separate tables for localization, classification, repair, preservation,
  support decisions, preference and runtime.
- A decision memo: retain baseline, expand evidence, or propose one narrowly
  justified change with its compatibility and verification boundary.

No automatic production thresholds or implementation dates are selected here.
This pass produced the research report and plan only; the proposed pilot has not run.
