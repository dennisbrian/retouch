# P8 — Revised cue-validity research protocol

Date: 2026-09-10. Baseline: `cea7b2e215cd1d4adc85eefe75a68a4f1635484f`.
Status: proposed protocol; no experiments executed in this task.
Companion: [source review and decision](RESEARCH_P8_CUE_VALIDITY_2026_09_10.md).

This protocol replaces the recommendation to execute the older
[P8.M draft](PLAN_P8_AGING_CUE_MEASUREMENT.md) unchanged. It describes future
work for review. The current request authorizes research and documentation
only; it does not start a harness, modify the corpus schema, collect ratings,
process photographs, fit an age model or implement a control.

## 1. Objective

Determine which existing image descriptors are observable and interpretable
enough to justify a later apparent-age study. Begin with feature contrast and
chroma variation; consider dark-line and highlight descriptors next. Exclude
physiological translucency and volume/AO estimation from the first pilot.

The first decision is per descriptor and capture condition. Permitted outcomes
are: proceed, restrict the intended use, redesign the descriptor, or insufficient
evidence. A failed descriptor does not establish that all of P8 is impossible.

## 2. R0 — Define evidence before writing a harness

Deliverable: a reviewed measurement specification and eligible-development
inventory. Existing subject assignments may be reused after checking their
scope; do not infer new identity matches from image similarity. No production
schema changes are a prerequisite.

| Item | Required decision or record |
|---|---|
| Intended quantity | Describe image appearance, predict human apparent age, or estimate an intrinsic property. The pilot covers the first. |
| Image provenance | Asset identifier/hash, capture/session when available, processing/ICC history, crop, resolution and whether already retouched. Unknown stays explicit. |
| Subject and split | Existing approved person ID, face instance and dev assignment; keep all variants of a person together. Distinct files are not independent subjects. |
| Pairing | State which conditions are held fixed. Different poses or cosmetics make an observational pair, not a lighting-only experiment. |
| Support | Approved face/feature/skin regions, region visibility, protected makeup/marks, occlusions, valid pixel counts and mask provenance. |
| Conditions | Lighting, exposure, expression, pose, makeup, facial hair, face scale, compression and observed color coverage. Record confounding and gaps. |
| Output state | Measured value or explicit unavailable/unsupported result, plus reason. Distinguish absent region, failed detection, too few pixels, clipping and processing error. |
| Measurement provenance | Descriptor revision, units, scale definition, color convention, raw components and any denominator or baseline. |

Define manual region acceptance before evaluating automatic masks. Preserve
the distinction between photographed skin and permission to edit it. Existing
manifest rights/usage records must cover the eventual study; public dataset
terms and identities require review before acquisition or use.

Chronological age is unnecessary for R1. If collected later, keep age at capture
with its source and precision separately from apparent-age ratings. A study
sidecar is a possible future design; no schema or field layout is implemented
by this document.

**Gate:** do the records support the question? If controlled light pairs are
absent, R1 can study observed capture sensitivity but cannot attribute it solely
to illumination. If usable regions are absent, fix the specification or source
better evidence before a sweep.

## 3. R1 — Qualify the descriptors, without age labels

This phase starts only after the user requests implementation/testing. Its
first task is measurement observability, before measuring a large corpus.

| Descriptor | What to retain | Principal validity checks |
|---|---|---|
| Eyes/lips/brows versus skin contrast | Per-feature signed LAB differences, luminance/chroma ratios, feature/surround areas | Check actual feature/skin overlap, support exclusions, annulus contamination and channel direction. Empty support cannot become ordinary zero contrast. |
| Skin chroma variation | Bandpass statistic, filter scales, support size, raw band summaries | Hold color convention and face scale explicit. Separate color cast, cosmetics, boundary bleed and noise from bare-skin variation. |
| Dark-line response, optional second tranche | Negative DoG response, scale, zone, numerator and any normalizer | Correct the older black-hat specification. Compare against manually reviewed lines and confounders; never label the output physical depth. Derive scale from face evidence, not image height alone. |
| Highlight appearance, optional second tranche | Coverage under an explicitly defined support rule, response magnitude, diffuse baseline and feathering | Compare highlight changes separately from exposure/base color. Nonzero blurred support is not a stable physical area. Do not relabel the response sebum or recovered gloss. |

First resolve the static risks identified in the report: mask epsilon
normalization, feature support intersections, absent neck zones, color units
and pixel-count confidence. Any future corrective code remains a separately
scoped implementation task; source inspection is not a reproduction.

Use three kinds of evidence in a future pilot:

1. **Analytic and synthetic controls:** check the declared response to dark-line
   amplitude, spatial scale, base color and highlights; include blank, clipped,
   occluded and missing-support cases. Vary one model parameter at a time.
   Keep raw response and normalization separate so a denominator cannot hide
   sensitivity. Synthetic exposure scaling is not physical relighting.
2. **Repeated capture:** same person, stable expression/cosmetics and documented
   lighting/camera conditions where available. Separate changes within fixed
   conditions from changes across conditions. Existing unrelated photographs
   are an exploratory arm, not a substitute for controlled capture.
3. **Support comparison:** approved manual masks first, then automatic masks on
   the same images. This separates descriptor behavior from region-detection
   error. Assess native-resolution support and intended viewing-size behavior
   separately.

Report raw paired changes, available-region coverage, missing/error rates and
uncertainty by subject and condition. If there are enough independent subjects
and crossed conditions, a variance-components analysis may help; otherwise
show the paired evidence and declare the limitation. Re-running a deterministic
metric on identical pixels establishes software repeatability only.

Define useful tolerances from the intended measurement and pilot uncertainty
before a confirmatory study. No universal percentage, sample size or variance
ratio is justified here. Count independent subjects and sessions alongside
images, regions and variants. Do not treat all patches as independent samples.

**Gate:** a descriptor may proceed for a restricted appearance task despite
lighting sensitivity. It cannot proceed as an intrinsic age/tissue measurement
on that evidence. Unusable support, unexplained zeros or a normalization that
cancels the target response require redesign. Missing coverage yields an
inconclusive result, not a universal pass or failure.

## 4. R2 — Test association with apparent age

Prerequisites: R1 provides interpretable outputs; eligible portraits have
independent apparent-age ratings with a documented presentation protocol.
Choose subject and rater counts from a pilot and a precision/power objective,
not the older draft's informal corpus-size estimate.

Apparent age is attached to a displayed image, including crop and presentation
conditions. Preserve individual ratings, number of raters, dispersion and
exclusions. Separate it from chronological age at capture and from beauty,
health or owner preference. The
[APPA-REAL documentation](https://chalearnlap.cvc.uab.cat/dataset/26/description/)
illustrates separate actual/apparent labels and per-image rating summaries;
it does not establish that its images fit Retouch's domain or usage terms.

Use development data to select descriptors and specify analyses. Freeze those
choices before any held-out assessment; keep the project's locked-test corpus
untouched during exploration. Prevent person/session leakage and account for
repeated raters and photographs in uncertainty estimates.

Compare descriptor associations with simple nuisance baselines for expression,
makeup, lighting, pose and capture processing. Preserve per-feature directions
and interactions instead of forcing every cue into a universal scalar. Show
condition-specific results, missingness and uncertainty; a global correlation
can conceal poor support or reliance on one shoot.

**Gate:** an association can justify proposing an intervention. It cannot
demonstrate a natural edit, physical aging mechanism, or that translucency is
irrelevant. Lack of association may reflect weak descriptors, narrow age range,
labels or capture confounding; state which conclusion the evidence supports.

## 5. R3 — Later perceptual intervention proposal

This phase is a future proposal only. It needs an explicit request to create
and evaluate image edits. If reached, start with bounded adult portrait edits
on approved support; keep geometry and physiological effects outside the first
comparison.

Compare source/disabled, conversion-only, each qualified single-cue edit,
and the proposed combination. Include reduced combinations so a redundant or
harmful cue can be identified. Match the source, masks, encoding and viewing
conditions. Changing multiple controls without these arms cannot establish
which change caused an effect.

Use randomized blinded judgments of apparent-age direction alongside separate
naturalness, likeness, expression, makeup/mark preservation and texture review.
Review the whole portrait and native crops. A lower age-regressor output or a
lower chroma-variation score is not the primary success measure. Keep age
judgments separate from the owner's aesthetic preference.

Include unchanged pairs, repeated trials and balanced presentation. Estimate
uncertainty across subjects and raters. Keep protection, outside-support
identity and disabled/abstention behavior explicit for pre-export pixels;
evaluate export-induced changes separately.

**Gate:** propose further development only if human-visible benefit survives
preservation review and is supported in the claimed capture range. Otherwise
drop a cue, narrow the use or stop the candidate. Neither success nor failure
of one combination proves a universal aging vector or a need for P1.

## 6. Deliverables and current boundary

| Phase | Reviewable artifact | Status on 2026-09-10 |
|---|---|---|
| Literature/source review | Companion report with primary links and corrected current-code claims | Completed |
| R0 | Approved descriptor definitions, pairing inventory and evidence fields | Specified here; inventory/approval remains future work |
| R1 | Measurement-only raw rows, support diagnostics and per-descriptor validity report | Not run |
| R2 | Rating protocol, image-level labels and subject-separated association report | Not run |
| R3 | Controlled edit proposal, comparison evidence and human preservation verdict | Not run or authorized by this request |

The next useful work is R0, followed by a small R1 pilot if implementation and
testing are requested. Budget the later phases only after usable support,
pairing and label availability are known. No production feature, schema
migration, test suite, render, dataset download or model change was made for
this research task.

## Implementation boundary update — 2026-09-10

The later explicit implementation request authorized the bounded R1 readout:
`retouch/aging_cues.py` and `scripts/qa/p8_cue_readout.py` now qualify the
feature-contrast and skin-chroma measurements without rendering or editing.
This does not authorize or implement the R2/R3 study, an age model, or a
user-facing aging-vector control.
