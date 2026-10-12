# Research — Face retouch algorithms: detection, texture, and local correction

**Date:** 2026-09-05

**Status:** Research and documentation complete; proposed experiments remain unrun.

**Code snapshot:** `e3c33ac9c458`, with an existing local modification to
`retouch/engine.py`. That modification belongs to the incoming worktree and was
not changed by this research.

**Scope:** Still-image facial retouching, especially natural skin, blemishes,
under-eye correction, makeup, and difficult face angles in portrait/cosplay photos.

The recommended next research priority is **more selective correction**: improve
where an operation acts, which skin detail it preserves, and when it should leave
the photograph alone. Retouch already has several capable smoothing algorithms.
The source review gives stronger reasons to investigate localization, overlapping
operations, and texture selection before adding another general smoothing model.
This is an engineering recommendation, not a measured ranking of algorithms.

For the next comparison, prioritize three questions:

1. Can face masks protect hair, makeup, accessories, and stable marks more reliably?
2. Can blemish detection distinguish an editable spot from a mole, freckle, shadow,
   or drawn cosmetic mark before deciding how to repair it?
3. Can texture restoration preserve useful photographed detail without restoring
   noise or undoing an earlier correction?

This report extends the [September 1 quality-ceiling research](RESEARCH_FACE_RETOUCH_QUALITY_CEILING_2026_09_01.md)
and its [implementation roadmap](PLAN_FACE_RETOUCH_QUALITY_CEILING_IMPLEMENTATION_2026_09_01.md).
It adds a current-code rebaseline, specific algorithm limitations, a comparison of
classical and learned methods, and experiments that separate detection quality
from repair quality. It does not reopen completed fixes or authorize implementation.

## 1. Evidence and interpretation

The report uses four evidence levels:

| Label | Meaning |
|---|---|
| **Current code** | Read directly in the working tree during this research; not a new runtime or visual test. |
| **Prior project evidence** | Reported by an existing Retouch study, with its original sample and resolution limits. Its image outputs were not visually re-audited in this session. |
| **Published result** | A paper's or author's reported method/result; not independently reproduced here. |
| **Proposal** | An inference or experimental design for Retouch; expected benefits remain unproven. |

Primary literature and author resources were checked on 2026-09-05. The search
covered established filtering/healing methods and directly relevant 2024–2026
face-retouch research. It is a targeted review, not a systematic survey of every
paper. Source access limits are recorded in section 9.

“Face side” is treated as the facial-processing part of Retouch, including side
profiles. The working aesthetic is natural correction with photographed likeness,
makeup, and intentional marks preserved. A porcelain or stronger beauty look is a
separate preference to compare, not a universal definition of quality.

## 2. What Retouch already does

| Area | Current implementation inspected | Research implication |
|---|---|---|
| Facial parsing | [FaceParser](../../retouch/parsing.py#L336): 512×512, 19-class BiSeNet face crops; hard labels, landmark regions, and feathering. Per-region confidence summaries now exist. | Start with the existing parser's uncertainty and boundaries before assuming a model replacement is necessary. |
| Smoothing | [FrequencySeparator](../../retouch/frequency.py#L542): three-band separation, guided/bilateral/anisotropic choices, adaptive texture protection, optional regional modulation and blotch reduction. | Guided filtering and region-aware smoothing are already available. The missing evidence is which combinations best preserve useful skin detail. |
| Texture recovery | [restore_micro_texture](../../retouch/skin.py#L985) restores part of the pre/post difference in selected facial zones; other code supports local clarity and texture transplant. | A difference image is broader than a pore-only signal. Study which removed components are safe to restore. |
| Spot repair | [BlemishRemover](../../retouch/blemish.py#L68): dark local anomalies, morphology/area filtering, Telea or skin-restricted PatchMatch repair. | Detection and filling are separate decisions and need separate evaluation. |
| Mark policy | [marks.py](../../retouch/marks.py) defines mark classes and policies; the inspected [face-core consumer](../../retouch/perf_optimizations.py#L517) uses a preserve mask in the freckle stage. | An available taxonomy is not yet a policy enforced consistently by every facial operation. |
| Tone/material | [skin.py](../../retouch/skin.py), [intrinsic.py](../../retouch/intrinsic.py), and [chromophore_v2.py](../../retouch/chromophore_v2.py) already provide local tone, albedo/shading approximations, shine handling, and relative pigment-like coordinates. | Compare these existing approaches on distinct causes of uneven appearance; avoid stacking several corrections on the same symptom. |
| Under-eyes | [UndereyeProcessor.process](../../retouch/undereye.py#L387): eye/lash exclusions, a local reference ring, masked low-pass estimates, bounded tone/chroma deltas, and support restoration. | Reference quality and makeup/lighting ambiguity are the next questions. The older inert detector is not the baseline. |
| Dispatch | [Face core](../../retouch/perf_optimizations.py#L787) combines legacy and newer dark-circle controls with `max` and applies one darken pass per eye. | The previously reported duplicate dark-circle application is already addressed. |

The [September 2 dark-circle study](RESEARCH_DARK_CIRCLE_OP_2026_09_02.md)
and [post-epsilon re-audit](RESEARCH_POST_EPSILON_FACEOP_REAUDIT_2026_09_02.md)
document important repairs after the older August audits. Their measured results
remain prior evidence, not fresh validation of this working tree.

### 2.1 Confidence exists, but its spatial information is still compressed

**Current code:** `_bisenet_confidence_by_region` computes softmax top-one minus
top-two margins at model resolution, then averages them over each predicted
region. Both single and batch face parsing still build operation masks from
`argmax` labels. The confidence field explicitly remains observational and must
not drive operation strength under its current contract.

**Implication:** A high average can coexist with an uncertain eyelid or hairline.
The next research step is to retain and evaluate spatial margins/posteriors and
boundary errors. It is not to introduce “confidence” from scratch or immediately
multiply every effect by the existing average.

### 2.2 The redness branch does not expand automatic blemish candidates

**Current code:** In [BlemishRemover._detect](../../retouch/blemish.py#L131), let
`D` be the current dark-anomaly mask and `R` the red-HSV mask. The redness branch
computes `D OR (R AND D)`, which equals `D`. Therefore redness does not independently
add or rank a candidate in that branch. The following morphology and area filters
still operate, but cannot recover a spot absent from their input mask.

**Implication:** A red spot with little dark-luminance contrast needs a different
candidate signal. This is a logical limitation established by reading the source;
no new measurement of missed spots or visual harm was performed. It does not
justify removing red pixels indiscriminately: blush and cosplay makeup are common
counterexamples.

### 2.3 General mark policy is not an all-operation protection mask

**Current code:** The inspected face core compiles `mark_policy` protection when
the freckle stage is active. The later automatic blemish call receives its skin
eligibility mask and can subtract the separate `mole_protect` result, but does not
receive the same general preserve mask. The policy module also states that some
compiled removal/attenuation masks await dedicated consumers.

**Implication:** Evaluate protection across smoothing, tone, freckle healing,
automatic blemish repair, and texture restoration together. Do not describe
`preserve_all` as an established end-to-end guarantee for every operation.

### 2.4 “Micro-texture” restoration currently restores a broader difference

**Current code:** The restoration signal is `original_bgr - smoothed_bgr`, weighted
by strength, smoothing strength, and a dimensional-zone mask. It is not explicitly
filtered into pore frequencies. The pre-smoothing snapshot precedes frequency
processing and optional freckle repair/flattening before restoration is reached.

**Implication:** That difference can include tone changes or an earlier correction,
as well as lost texture. Where supports overlap, restoring it can partially oppose
the earlier operation. This is a proposed interaction study, not a claim that all
current recipes visibly reintroduce freckles or blemishes. Test the actual active
recipe order and record each operation's support.

### 2.5 The Meitu pilot does not establish a smoothing deficit

**Prior project evidence:** The [September 4 five-pair bake-off](RESEARCH_MEITU_RETOUCH_BAKEOFF_2026_09_04.md)
reports substantial differences in saturation, edit extent, and aesthetic intent.
Retouch was rendered at a 2048-pixel long edge and analysis used 1600 pixels. The
report identifies export-resolution/profile confounds and explains why whole-face
high-frequency ratios cannot rank skin texture reliably.

**Implication:** Retain those five pairs as preference examples. They cannot tell
us which algorithm preserves pores best at native resolution, or reveal the
competitor's proprietary algorithms. Compare texture at matched correction intent
and use separate tests for source-like and porcelain appearances.

## 3. Algorithm families worth comparing

The fit and limitations columns below are Retouch-specific assessments. A reported
paper result is evidence for a candidate, not a local quality or speed guarantee.

| Method and primary source | Published mechanism | Best research use in Retouch | Main limitation to test |
|---|---|---|---|
| [Guided Image Filtering, ECCV 2010 / TPAMI 2013][S1] | Local linear filtering guided by an image, with edge-aware smoothing and a linear-time algorithm. | Existing baseline for local smoothing and boundary refinement. | Guidance edges can include makeup, noise, and real marks; edge preservation does not identify editable skin. |
| [Local Laplacian Filters, SIGGRAPH 2011][S2] | Edge-aware manipulation through local remapping and Laplacian pyramids. | A challenger for local tone/detail separation. | Scale/contrast decisions need facial context; compare halos, runtime, and expression preservation. |
| [FabSoften, CVPR Workshops 2020][S3] | Attribute-dependent smoothing, guided feathering, hair preservation, and wavelet texture restoration. | A targeted texture-restoration challenger. | Restored frequency bands can contain defects/noise; the reported smartphone results do not establish native-print performance. |
| [Telea inpainting, documented by OpenCV][S4] | Fill inward from a masked boundary using weighted neighboring information. | Current simple baseline for small isolated spots. | Larger regions may lack believable pore texture; a good fill cannot correct a wrong removal mask. |
| [PatchMatch, SIGGRAPH 2009][S5] | Approximate patch correspondence using randomized search and propagation. | Existing exemplar-healing option, with stricter same-face donor selection. | Repeated texture, lighting mismatch, and unsuitable donors. A copied patch is replacement content. |
| [CGFR, ICME 2024][S6] | Separate texture/base appearance; fit local sums of Gaussians in relative chromophore coordinates for gradual spot edits. | Controlled attenuation of localized color irregularity. | Requires a selected spot and suitable surrounding reference; broad shadows and makeup can violate the model. |
| [RetouchFormer, AAAI 2024][S7] | A learned clean-face dictionary, imperfection localization, selective attention, and multiscale tokens synthesize repaired content. | Learned localization and repair challenger, evaluated separately. | Learned clean-skin priors may remove intended marks or replace authentic detail. |
| [Diffusion data generation and spectral restorement, ICCV 2025][S8] | Frequency selection/restoration and multiresolution fusion; a benchmark with 25,000 before/after pairs. | Compare spatial/frequency modeling for different defect sizes. | Dataset policy includes moles among blemish types; match labels to our preservation policy before using the benchmark. |
| [ABPN, CVPR 2022][S9] | Low-resolution local retouching followed by adaptive blend-pyramid expansion/refinement. | Study efficient application of a coarse correction to a detailed source. | Learned high-resolution expansion is not an exact texture guarantee. Reported 4K speed used a Tesla P100. |
| [SegFace, AAAI 2025][S10] | Class-specific transformer tokens improve parsing, particularly infrequent accessory classes. | First named parser challenger after the current BiSeNet baseline. | Aggregate parsing scores can hide errors at the exact pixels an edit would damage. |
| [COMPOSE, ECCV 2024][S11] | Separate ambient and dominant illumination, then estimate/edit light and synthesize shadows. | A dedicated portrait-shadow research branch. | Cast-shadow editing is broader than under-eye repair and needs separate likeness/material review. |
| [InstantRetouch, Wu et al., CVPR 2026][S12] | Distill a diffusion teacher into a predictor of bilateral affine transforms applied to the full-resolution input. | Optional local tone proposals with source-coordinate rendering. | A tone transform does not solve missing-texture repair; clipping and excessive contrast suppression remain possible. |
| [BeautyGRPO, CVPR 2026][S13] | Preference-driven training with Dynamic Path Guidance to stabilize a generative editing trajectory. | A later learned retouching comparator and reference for preference evaluation. | Preference alignment does not prove that every pore, mark, or identity cue is preserved. |

Here, **InstantRetouch means the Wu et al. bilateral-space paper** linked above.
There is also an unrelated paper using that name for personalized retouching;
model acquisition must use the full title and author repository.

Three distinctions matter when interpreting these methods:

- **Tone transformation:** Changes the values of existing pixels. It can keep
  coordinates fixed while still damaging appearance through clipping or flattening.
- **Exemplar repair:** Replaces a selected region using photographed donor material.
  The donor may be real, but it is not recovered evidence of the original spot.
- **Generative repair:** Predicts plausible content from learned priors. Restricting
  the output to a small mask limits its extent, not the possibility of altered detail.

## 4. Proposed improvements to the face algorithms

### 4.1 Improve semantic support before increasing correction strength

**Proposal:** Evaluate per-pixel class probabilities, class margins, and boundary
uncertainty from the current parser before replacing its backbone. Compare this
against SegFace using the same intended subjects, face crops, and downstream
operation settings. Accessory classes are useful because glasses, earrings, and
hairline errors can become visible editing artifacts even when average skin masks
look good. The SegFace paper motivates this comparison; it does not prove a win on
our cosplay corpus. [S10]

Keep three concepts distinct:

| Quantity | Question it answers |
|---|---|
| Semantic estimate | Does this pixel appear to belong to skin, hair, lip, or another class? |
| Edit eligibility | Does the requested operation have permission to change this material or mark? |
| Blend alpha | How strongly should an accepted correction transition into the source? |

Blurring a label mask produces a soft transition, not calibrated semantic
confidence. A correct “skin” label also does not imply that a mole or blush should
be removed.

Use calibration only after collecting labeled held-out examples. Temperature
scaling is a useful baseline from classification research, but its success there
does not establish calibrated facial-boundary decisions. For Retouch, evaluate
error versus retained edit coverage by region, pose, and difficult lighting; keep
the current `parse_confidence` observation-only contract until that work exists.
[On Calibration of Modern Neural Networks, ICML 2017][S14]

For a future operation, a conceptual eligibility map could combine approved skin,
visible support, and explicit exclusions. Confidence can help decide acceptance
after calibration. Do not interpret a product of these signals as a calibrated
probability: the signals are correlated.

### 4.2 Separate texture, defects, and facial form more carefully

Retouch's frequency bands are spatial scales. Pores, noise, fine freckles, and
makeup can overlap in frequency; expression lines and shadow edges also cross
bands. Therefore “keep high frequencies” and “remove the middle band” are useful
approximations, not semantic definitions.

**Proposal:** Compare the current three-band implementation with a small wavelet
or Laplacian representation. Keep the same masks and requested correction, then
inspect the following separately:

| Component | Intended behavior |
|---|---|
| Broad facial form | Retain cheek/nose/jaw shading unless lighting correction is explicitly requested. |
| Uneven local tone | Reduce identified blotches without flattening a whole cheek. |
| Photographed texture | Preserve pores and fine hairs supported by the source. |
| Defect support | Exclude approved removals from texture reinjection. |
| Noise/compression | Avoid increasing it merely to match a high-frequency energy target. |

FabSoften supplies a relevant wavelet-restoration precedent; local Laplacian
filtering supplies another scale-aware challenger. Neither replaces the need for
defect and makeup masks. [S2] [S3]

A useful conceptual update is `output = source + allowed_delta`. Compute the
delta in a declared representation and limit its support. For texture restoration,
use only eligible detail components from the source, with corrected-defect and
protected-feature regions handled explicitly. This is a design description, not
an implementation or an assertion that current detail can be perfectly classified.

Also compare **which pixels a filter reads**, not just where its result is pasted.
A cheek filter can sample dark eyeliner or a colored wig and contaminate nearby
skin even when final edits remain inside the skin mask. Mask-normalized local
statistics, with sufficient valid support, are a candidate control. The under-eye
analyzer already uses normalized convolution; that is a useful local precedent.
A normalized Gaussian by itself is not edge-aware, so compare guidance and valid
sample selection separately. [Current implementation](../../retouch/undereye.py#L193)

### 4.3 Detect a blemish first; select a repair second

**Proposal:** Structure the research around candidate generation, mark classification,
policy, and repair. Give each stage an independent outcome measure.

Candidate evidence worth comparing with the current dark-spot detector:

- Luminance anomaly relative to nearby valid skin, normalized by robust local
  variation with a noise floor.
- Independent local chromatic anomaly, especially redness relative to nearby skin,
  rather than a fixed global red-HSV range.
- Scale, compactness, edge structure, and surrounding texture.
- Agreement with a learned imperfection-localization model.
- Explicit user-protected marks and, later, confirmed observations across shots.

The last two signals need separate evaluation. A stable mark is not automatically
unwanted, and a learned blemish score is not user intent. When a spot is ambiguous,
preservation or user selection is a valid result.

Choose a repair according to the observed issue:

| Situation | Candidate treatment | Reason to decline or reduce |
|---|---|---|
| Small local color difference, useful texture still present | Gradual local tone/chroma or CGFR-like attenuation | The spot might be an intentional mark or makeup. |
| Tiny isolated defect with consistent neighbors | Telea baseline | Boundary colors/textures are unsuitable. |
| Texture must be replaced and clean same-face donors exist | Restricted PatchMatch | Donors mismatch pore scale, direction, illumination, or makeup. |
| Broad redness/blotch | Local tone/material correction | A broad lighting gradient is being mistaken for pigment. |
| Large defect or absent source detail | Manual decision or separately evaluated learned repair | An automatic repair would invent consequential facial detail. |

These are proposed routing rules, not validated thresholds. Telea's neighborhood
filling and PatchMatch's correspondence search motivate their different roles.
[S4] [S5]

For exemplar healing, exclude all known defect supports from the donor pool, not
just the target hole. Match donors by facial region, scale, and lighting; record
where patches came from and inspect repeated patterns. Restricting donors to the
skin class alone does not ensure suitable donor texture.

For the first repair comparison, use manually accepted spot masks. Only after
repair quality is understood should automatic detection be reintroduced. This
prevents a mask improvement from being mistaken for a better filling algorithm.

### 4.4 Treat under-eye correction as reference estimation

The current v2 processor is already substantially closer to this approach than
the old dark-pixel detector. Its next limitation to investigate is whether nearby
skin is a valid target for the particular under-eye region.

**Proposal:** Separate reference-ring selection, broad luminance correction, and
color correction in the evaluation. A valid reference should avoid eyeliner,
blush, highlights, glasses, hair, and dissimilar lighting. If no convincing
reference remains, preserve the region or offer a smaller correction.

Preserve the lower-lid contour and a natural degree of shadow; forcing both eyes
to the same brightness can erase real lighting asymmetry. Do not mirror the
visible eye onto a hidden eye or assume two eyes have equal usable support.

CGFR's local parametric approach is relevant to gradually attenuating a color
irregularity while retaining texture, but it is not an automatic diagnosis of
under-eye darkness. Its paper uses selected regions and relative chromophore
modeling. In a general photograph, color, makeup, and illumination remain
ambiguous. [S6]

### 4.5 Distinguish uneven tone, shine, and cast shadow

Retouch already has operations addressing these individually. The proposal is
to compare their explanations and interactions before chaining more of them.

| Observed appearance | Hypothesis to test | Common mistaken correction |
|---|---|---|
| Broad dark side of a face | Illumination/form | Whitening or pigment reduction that flattens facial structure. |
| Local red/brown irregularity | Relative color variation | Whole-face desaturation that also removes intended makeup. |
| Bright oily-looking patch | Specular component | Smoothing away both glare and the underlying texture. |
| Deliberate glossy makeup | Material/creative finish | Automatic shine removal that changes the intended appearance. |
| Hard shadow across face | Cast illumination | Blemish inpainting across a large region. |

This table proposes distinctions for evaluation; it does not claim they can all
be reliably inferred from one RGB image. Retouch's chromophore module correctly
documents its outputs as relative image coordinates, with limitations under mixed
illumination, makeup, specular reflection, and camera response.

COMPOSE makes controllable portrait shadows a worthwhile separate research topic:
its decomposition and learned shadow synthesis target lighting structure. A
COMPOSE-style shadow editor should be assessed independently from natural skin
cleanup, including how well it preserves the original scene's illumination.
[S11]

### 4.6 Handle profile and occluded faces by visible evidence

**Proposal:** Validate each visible region independently, while retaining existing
pose gates as the baseline. An extreme face angle can leave one cheek suitable
for gentle tone correction and make the far eye unsuitable for any enhancement.

Face width and inter-eye distance are useful scale references, but become less
stable when a landmark is occluded or strongly foreshortened. Compare their
stability against visible-region measurements; record the chosen scale source and
decline operations whose geometry is unreliable. No new yaw thresholds are proposed.

Specific hard cases for this project are wigs over one eye, opaque or colored
contacts, glasses glare, a hand over the cheek, open-mouth/tongue boundaries,
printed faces in the background, and multiple people at different distances.
The intended subject must be confirmed; “largest detected face” is not a reliable
substitute for that choice, as prior project studies already show.

### 4.7 Introduce learned methods according to what they output

**Proposal:** Evaluate a learned mask before permitting the same model to supply
replacement pixels. Then compare a small accepted repair residual, and only
later a complete learned retouch. These experiments answer different questions.

For tone-only assistance, the bilateral-space InstantRetouch is relevant because
the model predicts transforms applied at source resolution. The bounded-transform
idea is useful even without a language interface. It still needs identity-transform
fallback, support containment, color/gamut limits, and local texture review. This
is an inference about a possible Retouch integration, not an existing capability
or a claim of compatibility with the pinned runtime. [S12]

BeautyGRPO is useful for studying preference alignment. Its project describes
five-dimensional preference annotation by vision-language models followed by
human verification, and a FluxKontext-LoRA editing backbone. These preferences
should be compared with our owner's preferences rather than adopted as a universal
beauty target. Keep a source-preserving baseline and examine failed examples as
carefully as showcase results. [S13]

## 5. Proposed experiments, in order

All rows below are **future experiments**. No scripts, models, recipe changes, or
photo renders were created for them in this task. Begin with isolated operations
on a small reviewed subset, then expand the promising candidates.

| ID | Question and comparison | What stays fixed | Evidence needed to proceed |
|---|---|---|---|
| FA-01 | Current masks versus spatial posterior/boundary handling; then BiSeNet versus SegFace. | Intended subject, input/crop geometry, operation, and requested strength. | Fewer edits crossing manually labeled protected boundaries at comparable useful edit coverage. Report failure/skip rates. |
| FA-02 | Current texture restoration versus band-selective restoration with repaired-defect exclusions. | Masks, smoothing target, and all other face operations. | Better retained pore/hair appearance, no restored defect, no extra noise or repeated texture; include restoration disabled as an ablation. |
| FA-03 | Current blemish candidates versus independent local luminance/chroma candidates and a learned locator. | User mark policy and one repair algorithm. | Better editable-spot recall at a matched rate of unwanted mark removals; report each mark type separately. |
| FA-04 | Telea versus restricted PatchMatch versus a bounded learned repair. | Manually accepted spot masks and requested removal amount. | Better repaired texture/continuity without added seams or donor repetition. Repeat afterward with each detector to expose interactions. |
| FA-05 | Current under-eye reference versus reference selection with explicit makeup/lighting exclusions. | Under-eye support and requested correction. | Less lash/makeup change and more natural residual shadow; report eyes with insufficient reference support. |
| FA-06 | Existing local tone/chromophore/shine operations versus a CGFR-like local fit. | Problem labels, support, intended correction, and source color preparation. | Improvement on the targeted irregularity without flattening form or erasing makeup; inspect combined-operation conflicts. |
| FA-07 | Classical tone baseline versus bilateral learned tone; selected learned repair versus RetouchFormer/spectral/BeautyGRPO candidates when obtainable. | Input, export, accepted supports, and correction intent. | Blinded owner preference with no increase in critical likeness/mark failures; record full local runtime and memory. |

FA-03 and FA-04 form a useful two-factor study: **detector × repair method**.
The same final image can look better because a detector selected a better region,
because the fill improved, or both. A single end-to-end score cannot distinguish
those explanations. Manual masks provide a controlled repair baseline, not proof
that automatic mask selection is solved.

Profile/occlusion cases belong in every relevant experiment. They should not be
held until an otherwise complete frontal-face study.

### 5.1 Corpus and comparison rules

Reuse the existing project corpus and five Meitu pairs after verifying their
current manifests and pairings. The documented 83-image collection is a useful
starting pool, not a claim of 83 independent people or balanced coverage.

Curate examples of clear frontal faces, three-quarter/profile faces, small faces,
soft focus, compression/noise, hard light, mixed/colorful light, different visible
skin tones, freckles/moles, facial hair, strong makeup, glasses, wigs, occlusions,
and multiple faces/posters. Confirm the intended person and editable regions.
Do not infer ethnicity or a clinical skin type from the photograph.

Keep subjects and near-duplicate frames in the same data split. Choose parameters
on development images, calibrate confidence on a separate labeled set, and retain
an untouched test set. Report actual image, subject, eye, and spot counts. A small
pilot can reject poor candidates; it cannot establish universal superiority.

For each input, preserve a source reference and review:

1. The isolated operation, with other stages held constant.
2. The combined face recipe, to reveal interference among operations.
3. Native-resolution crops and the intended delivery view.

Use the same color preparation, resolution, export encoding, and subject crop for
appearance comparisons. For algorithm diagnostics, retain lossless working-buffer
comparisons before export. If a model only operates at a smaller size, label that
as a separate arm and evaluate its upsampling/compositing step.

For the competitor arm, retain the [existing bake-off limitations](RESEARCH_MEITU_RETOUCH_BAKEOFF_2026_09_04.md):
verify pairings visually, record product/version and settings, separate automatic
from manually assisted results, and measure operator intervention as well as
processing time. Register exported pairs symmetrically for appearance metrics;
review unwarped pairs too so registration cannot conceal geometry changes.

### 5.2 Match the correction, not just slider numbers

An amount of `50` in two algorithms need not produce the same effect. First
compare a small strength sweep at an agreed target, such as “attenuate this spot
while keeping the mole” or “soften this shadow while retaining the lower lid.”
Then compare quality and effort among candidates meeting that target.

Include unchanged input, the current preferred recipe, and milder/stronger
variants. Randomize review labels and permit ties or “neither acceptable.” Judge
natural and porcelain styles in separate preference arms. Otherwise the study can
reward a style preference as if it were a technical improvement.

## 6. What to measure

The following is a proposed evaluation contract. Numerical visual thresholds
must come from reviewed data; none are calibrated by this document.

| Dimension | Useful evidence | Why a single score is insufficient |
|---|---|---|
| Requested correction | Spot visibility or under-eye/blotch improvement on labeled targets; owner judgment. | Returning the source unchanged can score excellent fidelity while failing the request. |
| Mark preservation | Preserved-mark count, unwanted removals, false detection examples, and local source/output crops. | A general face-recognition score can stay high after a distinctive mark disappears. |
| Texture | Skin-only, band-specific residuals; pore/hair review; noise and patch-repetition inspection. | Sharp lashes, added noise, or repeated pores can inflate high-frequency energy. |
| Boundaries | Leakage into protected features, edge halos, and transition continuity at native zoom. | Mean region overlap can hide a small but conspicuous eyeliner error. |
| Tone/material | Local luminance/chroma changes, retained form, face/neck consistency where relevant, and makeup preservation. | Lower saturation or brighter skin is not automatically better. |
| Geometry/likeness | Unwarped source/output inspection and landmark movement as supporting evidence. | Detector drift is not true shape movement; fixed geometry alone does not prove unchanged likeness. |
| Uncertainty | Harm versus accepted edit coverage, region-specific confidence reliability, and abstentions. | Rejecting every difficult image can appear safe while providing little useful editing. |
| Performance | Cold/warm timing, detection/parsing/edit/export components, peak memory, face count and native pixel dimensions. | Published GPU timing does not predict this Mac's pinned Retouch runtime. |

Compute texture measurements in manually checked skin regions that exclude eyes,
hair, makeup edges, and intentionally repaired spots. Record face scale and
effective bandwidth. If the source has almost no energy in a band, report that
the ratio is unreliable instead of dividing by a tiny number and ranking it.
PSNR, SSIM, perceptual-distance models, and generic aesthetic scores can supplement
review; none alone establishes correct retouching or authentic pores.

There are some useful exact requirements for a future isolated operation:

- Zero strength returns its input unchanged.
- Pixels outside its declared support equal the pre-operation working buffer.
- Hard protected regions remain unchanged when that operation's contract excludes
  them, including after feathering/compositing.
- Missing or invalid evidence produces an explicit skip/fallback outcome.

These checks apply at the operation boundary before lossy encoding. A JPEG
re-encode can change pixels beyond a mask. Whole-recipe global color stages also
need a matched baseline when assessing a strict face-only operation. Do not use
their deltas as evidence of face-mask leakage.

For learned or exemplar repairs, additionally review the accepted repaired region
itself: perfect outside-support containment says nothing about whether the new
inside-support detail is correct or desirable.

## 7. Practical shortlist and connection to the existing roadmap

| Priority | Recommended research package | Existing roadmap connection |
|---|---|---|
| First | Spatial parser evidence, boundary ownership, and mark protection across operations. | FR-QC-1C/1D/1E/1G, with the current confidence summaries acknowledged. |
| First | Texture restoration limited by frequency and actual corrected regions. | FR-QC-2E and the shared operation/evidence work in FR-QC-1A. |
| Next | Separate blemish localization from repair; compare local chromatic detection, Telea, and restricted donors. | FR-QC-2A/2C/2G; a mark ledger supports this but is not required for the first manual-mask comparison. |
| Next | Under-eye reference quality and tone/material interactions. | FR-QC-2C/2F/2G, extending the shipped v2 under-eye work. |
| Later | Bounded learned tone/repair and owner preference comparison. | The roadmap's optional learned-assistance work; keep generative content distinguishable. |

The first two packages provide the best starting point because they address
concrete limitations visible in the current source and can improve several
operations at once. Learned methods are promising comparators, but source review
and publications alone do not show that replacing the current face engine would
improve this user's photos.

### 7.1 Availability is separate from implementation readiness

| Candidate | Public resource verified on 2026-09-05 | What remains unknown here |
|---|---|---|
| SegFace | [Author repository](https://github.com/Kartik-3004/SegFace) and [author model collection](https://huggingface.co/kartiknarayan/SegFace). | Local runtime/export behavior, selected checkpoint, corpus performance, and complete model/data reuse terms. |
| Bilateral InstantRetouch | [Author repository](https://github.com/OpenImagingLab/InstantRetouch) with training/inference instructions and an Apache-2.0 code-license statement. | Usable checkpoint completeness, dependencies, hardware needs, and base-model/data terms for the intended use. |
| BeautyGRPO | [Author repository](https://github.com/vivoCameraResearch/BeautyGRPO) and [linked LoRA model](https://huggingface.co/Sehnsucht24/BeautyGRPO). The README describes a default 1024-size inference path and separate base-model requirements. | Native-resolution integration, local resource cost, and applicability of model/base-model terms. The README explicitly separates those from its MIT code-license statement. |
| PatchMatch | Already present as a Retouch healing option. The [original author's linked reference code][S5] is described as noncommercial research code. | The original reference code's terms do not establish the provenance of Retouch's existing implementation. Do not import that reference code as an assumed production dependency. |
| CGFR, RetouchFormer, spectral method, ABPN, COMPOSE | Primary publication evidence located. | Runnable official checkpoints and deployment suitability were not established in this review. |

No repositories were cloned, dependencies installed, weights downloaded, or
photographs sent to external models. Existing implementation choices can be
researched from papers without adopting a particular reference-code package.

## 8. Research completion boundary

Completed in this task:

- Read the relevant current face-processing source and recent project studies.
- Documented four specific source findings and corrected older baseline assumptions.
- Compared thirteen algorithm families using primary publications/documentation.
- Specified seven future experiments, their controls, and evaluation criteria.
- Linked this report from the documentation index.

No engine, test, recipe, model, or image changes were made by this task. No new
runtime benchmark, test-suite run, native-image visual comparison, model
reproduction, or owner preference study was performed. Unrelated worktree edits
were preserved. The implementation and experiment results remain future work;
this document completes the requested research/documentation scope.

Documentation verification covered local link targets, all fourteen source
reference definitions, the seven experiment entries, and whitespace checks. The
existing `engine.py` content fingerprint was unchanged across this task's review.

## 9. Source register and reading limits

These are primary sources. Bibliographic publication years are used; search-engine
crawl dates are not publication dates. Method summaries elsewhere in this report
stay within what was accessible. Reported results were not independently reproduced.

| Ref | Publication/resource | Reading basis in this session |
|---|---|---|
| S1 | He, Sun, Tang — Guided Image Filtering, ECCV 2010 / TPAMI 2013. | Author project page and indexed paper abstract. |
| S2 | Paris, Hasinoff, Kautz — Local Laplacian Filters, SIGGRAPH 2011. | Author-hosted paper's indexed extract; direct PDF fetch exceeded the browser's size limit, so no claim of a full-paper review. |
| S3 | Velusamy et al. — FabSoften, CVPR Workshops 2020. | Indexed CVF paper abstract/introduction; author publication record. |
| S4 | OpenCV — Image Inpainting; documents Telea 2004 and related algorithms. | Official tutorial; the live 4.x URL resolved to 4.13.0 documentation. This is not the project's installed-version claim. |
| S5 | Barnes et al. — PatchMatch, SIGGRAPH 2009. | Author/university publication page, including the reference-code notice. |
| S6 | Shuai et al. — Controllable and Gradual Facial Blemishes Retouching via Physics-Based Modelling, ICME 2024. | Author-submitted arXiv record and HTML methods, including selected ROI, layer separation, and local Gaussian fitting. |
| S7 | Wen et al. — RetouchFormer, AAAI 2024. | Official proceedings abstract and bibliographic record. |
| S8 | Xu, Zhang, Lu — Face Retouching with Diffusion Data Generation and Spectral Restorement, ICCV 2025. | Indexed CVF abstract and official conference session abstract. Direct CVF HTML/PDF opening returned 403; training/reproduction details were not fully reviewed. |
| S9 | Lei et al. — ABPN, CVPR 2022. | Indexed CVF proceedings abstract and paper method extract. |
| S10 | Narayan, VS, Patel — SegFace, AAAI 2025. | Official proceedings abstract, author project/repository, and author checkpoint listing. |
| S11 | Hou et al. — COMPOSE, ECCV 2024. | Author-submitted arXiv abstract describing illumination representation and pipeline. |
| S12 | Wu et al. — InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space, CVPR 2026. | Author project/repository and indexed CVF paper/supplement method extracts. |
| S13 | Yang et al. — BeautyGRPO, CVPR 2026. | Author project, arXiv abstract, inference README, and linked model page. |
| S14 | Guo et al. — On Calibration of Modern Neural Networks, ICML 2017. | Official PMLR abstract; used as a calibration baseline, not as proof for segmentation. |

[S1]: https://people.csail.mit.edu/kaiming/eccv10/index.html
[S2]: https://people.csail.mit.edu/hasinoff/pubs/ParisEtAl11-lapfilters.pdf
[S3]: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Velusamy_FabSoften_Face_Beautification_via_Dynamic_Skin_Smoothing_Guided_Feathering_and_CVPRW_2020_paper.pdf
[S4]: https://docs.opencv.org/4.13.0/df/d3d/tutorial_py_inpainting.html
[S5]: https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/
[S6]: https://arxiv.org/html/2406.13227v1
[S7]: https://ojs.aaai.org/index.php/AAAI/article/view/28404
[S8]: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Face_Retouching_with_Diffusion_Data_Generation_and_Spectral_Restorement_ICCV_2025_paper.html
[S9]: https://openaccess.thecvf.com/content/CVPR2022/html/Lei_ABPN_Adaptive_Blend_Pyramid_Network_for_Real-Time_Local_Retouching_of_CVPR_2022_paper.html
[S10]: https://ojs.aaai.org/index.php/AAAI/article/view/32661
[S11]: https://arxiv.org/abs/2408.13922
[S12]: https://openimaginglab.github.io/InstantRetouch/
[S13]: https://beautygrpo.github.io/
[S14]: https://proceedings.mlr.press/v70/guo17a.html
