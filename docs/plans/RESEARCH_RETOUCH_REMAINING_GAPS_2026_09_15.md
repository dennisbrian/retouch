# Retouch: remaining quality and research gaps

Date: 2026-09-15. Source inspected: `2dba95b9622a1a93cd48baac87fdee94b7a749d6`.
Scope: current source, saved research and pilot metadata, and primary web sources.
No production changes, models, photo processing, new annotations, or experiments.

## Recommendation

Prioritize reliable selective correction: decide which photographed detail to
preserve, which region belongs to the intended person, and when evidence is too
weak to edit. The largest unresolved opportunity is confidence in these choices.
Retouch already has extensive smoothing, texture, color, mask, and repair code.

For the next study, combine a reviewed preservation/support development set with
the existing FA-03 and P8 support protocols. Then evaluate texture restoration
under fixed, reviewed supports. Keep implementation and release decisions separate
from this research recommendation. See the [next-study plan](PLAN_RETOUCH_GAP_RESEARCH_2026_09_15.md).

## 1. Corrections to older backlog claims

| Older impression | Current evidence | Remaining limitation |
|---|---|---|
| P7 still lacks a product control | `retouch/params.py` registers `cross_region_skin`; `gui.py` has its slider and settings wiring | Ownership, preview/export decisions, and diagnostic visibility still need qualification |
| FA-03 always resolves a tied mole score by dictionary order | `retouch/freckle.py` now explicitly ranks classes and abstains when a winning beauty-mark score has a near competitor; `marks.py` maps `ambiguous` to `unknown` | This is a narrow tie fix, not a general makeup/mark classifier |
| FA-02 is only a representation proposal | `perf_optimizations.py` contains opt-in `raw_residual`, `dog`, and `multiscale` paths plus an eligibility gate | The gate explicitly declares its thresholds uncalibrated; the default remains `legacy` |
| P8 is just a paper plan | `aging_cues.py` and `scripts/qa/p8_cue_readout.py` implement observable-cue measurement | Measurement code does not establish apparent-age validity or reliable region ownership |
| There is no subject-separated corpus schema | `corpus_manifest.py` has v3 person/group separation and dev/calibration/locked_test support | A schema is not a populated, reviewed evaluation set |
| P9 is the approved next feature | The September 11 documents explicitly label it an unconfirmed proposal | Reuse its support-usability protocol without silently adopting a new roadmap item |

The July master plan and early September reports are historical evidence. Their
unimplemented/complete labels cannot substitute for inspecting today's code.

## 2. Ranked gaps

### 1. Real mark, makeup, and boundary preservation

**Existing:** class-aware mark policy, protection masks, narrow ambiguity handling,
and a five-arm FA-03 research experiment. `perf_optimizations.py` supplies mark
protection to selected skin/frequency operations. This is not absent infrastructure.

**Missing evidence:** dependable separation of intentional eyeliner, natural marks,
blemishes, fine hair, and uncertain material on independent real portraits; also
preservation across a complete recipe. A taxonomy containing `drawn_makeup_mark`
does not prove the detector can recognize it.

The [FA-03 experiment](RESEARCH_FA03_CLASSIFICATION_EXPERIMENT_2026_09_05.md)
reports three real faces, two known liner components, synthetic mark controls,
and no real natural-mark preservation proof in that study. Those historical
counts are not an inventory of every photo now available. The narrow ambiguity
fix has since landed; the experimental shape/context classifier is not equivalent
to the production scorer.

**Next question:** at matched useful correction coverage, do shape plus anatomical
context reduce harmful mark decisions relative to today's fixed baseline?
Keep candidate detection, classification, and repair as separate evaluations.

### 2. Distinguishing photographed texture from noise and corrected defects

**Existing:** legacy residual restoration, experimental frequency-band restoration,
and opt-in noise-aware clarity. `SkinProcessor.restore_micro_texture` restores part
of the source-minus-smoothed residual through dimensional masks. That residual can
contain useful detail, noise, pigment edges, and detail intentionally suppressed.

**Missing:** an eligibility measure and restoration policy that demonstrate useful
detail retention without reintroducing noise or removed defects. Current
`fa02_texture_eligibility.py` still uses sigma 2 px and a 3.5 high-pass threshold;
its source labels the thresholds provisional or placeholders.

The [metric-validity study](RESEARCH_FA02_METRIC_VALIDITY_2026_09_06.md) reported
resize-driven eligibility flips on 4/9 faces and noise-triggered crossings below
the sibling noise ceiling. These are saved study findings, not a fresh rerun.
Neither tested metric replacement earned promotion. The
[P6 clarity study](RESEARCH_P6_NOISE_AWARE_CLARITY_2026_09_07.md) likewise leaves
arbitrary camera/JPEG noise calibration unresolved.

**Next question:** with reviewed supports held fixed, which method restores accepted
detail while preserving corrected regions, makeup, and smooth/noisy regions?
Evaluate restoration off, current legacy, and the existing experimental arms.

### 3. Reliable support and ownership in difficult scenes

**Existing:** face parsing, landmark clipping, occlusion gates, P7 support inference,
and missing-support handling. P7 rejects multiple detected faces and conflicting
`body_match_face`; it propagates bounded face-edit LAB deltas.

**Missing:** proof that accepted body skin belongs to the target person under
occlusion, skin-colored garments, missed background people, and close group poses.
One detected face is not proof that only one person is present.

P8's runner computes one full-image person mask and passes it to every face parse.
The parser also uses individual face crops and landmark clipping, so this fact
alone does **not** prove every P8 region is globally pooled or contaminated.
It does show the runner supplies no explicit person-instance ownership map.
Validate overlapping faces and fallback parsing before multi-person claims.

**Next question:** compare automatic support with reviewed same-person regions and
exclusions. For face parsing, inspect rare accessories and boundaries separately
from average skin-region accuracy. P7 ownership and P8 descriptor usability require
different labels and cannot share a calibrated threshold by assumption.

### 4. Useful automatic decisions and representative evaluation

**Existing:** Safe Auto apply/dampen/skip/review decisions, parser margin summaries,
cue availability, corpus v3, and review tooling.

**Missing:** observed relationships between those evidence scores and actual errors.
P8 metric confidence includes `count / 800` and
`min(feature_count, surround_count) / 500`, clipped to one. These are support-size
heuristics, not correctness probabilities. Retouch already documents similar
limits on parser margins and Safe Auto scores.

Current metadata check: the ten `test_output/fa02_pilot_*/corpus_manifest.json`
files contain ten unique source hashes, eight recorded person IDs, all `dev`,
with only `light` and `very_light` tone labels. This is a count of those manifests,
not a full private-photo census, independent identity verification, fresh label
review, or image-decoding check. It does not establish representative calibration.

**Next question:** how much useful work is retained as accepted-support errors fall?
Always report accepted errors, valid cases rejected, unknown labels, and coverage.
A policy rejecting every image has undefined accepted-case error, not perfect safety.
The [P9 proposal](RESEARCH_P9_SELECTIVE_RETOUCH_PROPOSAL_2026_09_11.md) already
provides a suitable protocol; a second confidence framework is unnecessary.

### 5. Preview/export agreement and visible reasons for skipped edits

**Existing:** native export, render/cache provenance, runtime diagnostics, and P7
diagnostics attached to engine results. Recent CLI commits address worker teardown
hangs; those historical fixes should not be reported as still missing.

**Current source gap:** GUI preview and full quality select different `fast`
settings. The inspected `preview_cache.set_latest_render` payload in `gui.py`
includes Safe Auto decisions and backend metadata, but not `p7_diagnostics`.
The engine's P7 apply/abstain decision depends on detected face count.

This is a source-level reason to investigate decision drift; it is not a newly
reproduced GUI failure. Different resolutions need not have identical pixels,
but users need to know when the target, enabled operation, or skip reason changes.

**Next question:** for the same requested edit, do preview and export agree on
target and disposition, and does the interface expose any meaningful difference?
Include native multi-face worker execution and exported manifests in that audit.

### 6. Owner preference and a fair competitor comparison

The [Meitu bake-off](RESEARCH_MEITU_RETOUCH_BAKEOFF_2026_09_04.md) already has
five same-source pairs and three Retouch recipes, including `meitu_porcelain_v1`.
It reports 2048 px processing/1600 px comparison, varying competitor exports,
and no powered preference result. Its observed aesthetic differences do not
establish that Retouch needs stronger smoothing or that one system wins.

**Missing:** a blinded, matched-delivery owner preference study and evidence that
automatic QA warnings distinguish accepted results from actual defects. Existing
warning counts should not be used as a beauty ranking without that relationship.

**Next question:** which recipe is preferred for each intended look, while still
passing preservation requirements? Record preference separately for texture,
tone, likeness, makeup fidelity, and overall result, including ties.

## 3. External research and product comparison

Primary sources checked on 2026-09-15. These establish published methods or vendor
features, not performance on Retouch images.

| Source | Relevant lesson | Suggested use |
|---|---|---|
| [SegFace, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32661) | Uses class-specific transformer tokens to improve rare face-parsing classes; evaluates CelebAMask-HQ and LaPa | Offline parser challenger if accessory/boundary errors persist. Published averages and speed do not establish native cosplay performance or mark-preservation quality |
| [SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) | Evaluates selective prediction through error versus coverage | Reuse that evaluation question for support acceptance; adopting its neural architecture is unnecessary for a first study |
| [BeautyGRPO, CVPR 2026](https://arxiv.org/abs/2603.01163) | Introduces FRPref-10K and a preference-oriented reward model plus guided generation | Use as a methodological comparison for multidimensional preference evidence. A learned reward or attractive output does not certify preservation of photographed detail |
| [Capture One, August 2025 release](https://www.captureone.com/en/explore-features/whats-new/16-6-5) | Provides explicit facial protection regions for automatic blemish removal, including multiple faces | Benchmark the ease of preserving a chosen mark. Retouch already has mark policies and profile records; audit the complete user workflow before claiming it lacks manual protection |

The inference from this comparison is to invest first in local control, preservation
evidence, and useful abstention. SegFace and generative methods remain conditional
challengers. No model, dataset, runtime compatibility, or license was qualified
for integration in this pass. An additional ICCV 2025 retouching paper was found
in search, but direct CVF access returned 403; it is not used to rank candidates.

## 4. Priority and limits

1. Review preservation/support labels and extend difficult real cases (FA-01/FA-03).
2. Qualify texture/noise measurements and existing FA-02 restoration arms.
3. Resolve P7/P8 ownership evidence and preview/export disposition visibility.
4. Run blinded owner preference and matched-delivery competitor evaluation.
5. Consider parser replacement or learned assistance only for measured residual gaps.

P8 apparent-age associations remain a separate later question. Additional sliders,
age controls, and generative reconstruction do not close the above evidence gaps.
This pass is focused on still-portrait quality and its user workflow; it is not a
distribution/signing, full color-pipeline, or video-readiness audit.

Validation boundary: source and saved-document inspection, metadata aggregation,
primary-source review, and documentation checks only. No new quality scores,
visual acceptance, runtime measurements, or production-readiness certification.
