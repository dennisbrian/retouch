# P7 — Cross-region skin consistency

Date: 2026-09-09. Code baseline inspected: `2e88a89`.
Scope: literature review, code inspection and proposed experiments only.
No production edits, new models, image processing or test execution.

Scope assumption: P7 means the cross-region proposal in
[Skin ProMax](PLAN_SKIN_PROMAX.md#p7--cross-region-skin-consistency-neckearshandschest-as-one-tone-system--novel),
following the recent P4–P6 research. The repository also uses P7 for
Output-Conditioned Master; that separate export feature is outside this report.

## 1. Recommendation

**GO for an isolated study of subject-specific, reference-conditioned appearance
consistency. NO-GO for promising recovery of one true skin albedo and lighting
model from an ordinary single photograph.** These are research recommendations,
not measured candidate results or authorization to implement.

The useful product question is: can an approved face edit be extended to the
same person's visible skin without erasing neck shadows, backlit ears, natural
regional differences, makeup, tattoos or texture? “Make every region match the
face median” is an inadequate objective: it rewards removal of real lighting
and can transfer foundation color to unpainted hands.

Start with one person, manual region ownership and approved reference patches.
Study bounded low-frequency color-edit propagation first. Keep ears, palms,
deep shadows, ambiguous cosmetics and mixed illumination protected initially.
Texture parity should remain a separate arm aligned with the existing
[cosplay S2 plan](PLAN_COSPLAY_PORTRAIT_POLISH.md#s2---face-body-texture-and-tonal-parity).
A tone result must not be presented as texture or subsurface-scattering parity.

## 2. What the current implementation supplies

These are code observations, not newly reproduced photographic failures.

| Component | Inspected behavior | Consequence for P7 |
|---|---|---|
| [`SkinProcessor.harmonize_neck`](../../retouch/skin.py#L774) | Uses face and neck LAB medians; adjusts L by 0.75 and a/b by 0.7 times strength and median difference. Neck support uses a supplied mask or person-mask rectangle, a face-relative chroma gate, and Gaussian feathering. | Local statistical correction; no recovered light, anatomical region model or skin-reflectance identity. A shadow contributes to the measured L difference. |
| [Neck dispatch](../../retouch/perf_optimizations.py#L1116) | Runs when selected face whitening/equalization/color operations are active; strength is their maximum. | P7 must account for existing neck edits and avoid applying the same correction twice. |
| [`_stage_body_skin`](../../retouch/engine.py#L3751) | Body candidate uses LCH skin-color selection intersected with the person mask, face/hair/lip exclusions, connectivity and an OKLCh chroma gate intended to exclude paint/tattoos. | A color gate is not a semantic guarantee: muted paint or skin-colored fabric may qualify; valid skin under colored light may fail. |
| [Body face matching](../../retouch/engine.py#L3959) | Uses medians from accumulated face skin and body skin; applies a global bounded LAB offset (±8 L-channel codes and ±6 a/b codes). | No separate hand/chest/ear targets or lighting-conditioned comparison. OpenCV uint8 LAB channel units must not be mislabeled as CIELAB L* units. |
| [Face-mask accumulation](../../retouch/engine.py#L3475) | Unions face masks into one `acc_skin` canvas. Body matching measures that union. | With multiple faces, the reference can pool different people. Person-mask connectivity does not assign each hand to a specific face; touching people can share a component. |
| [`decompose_intrinsic`](../../retouch/intrinsic.py#L127) | Smooths log luminance, normalizes a scalar shading map, divides the image by it and clips the resulting color layer. The helper does not decode the transfer function. | An appearance decomposition, not independently measured pigment or geometry. Scalar shading cannot explain colored light. Clipping can invalidate exact reconstruction despite the docstring. |
| [`BodyRelighter.relight`](../../retouch/body_relight.py#L28) | Derives pseudo-normals from blurred image luminance gradients, then applies a requested direction. | Brightness gradients can contain pigmentation or cast shadows; these are not measured 3-D normals or a joint face/body light estimate. |

The body matcher reconstructs through a whole-image uint8 LAB round trip and
does not restore all pixels outside its correction support in that block.
P7 experiments must separate conversion-only changes from the intended edit.
The neck helper already restores pixels outside its blurred support, but that
blurred support is broader than the original semantic support. A future strict
protection contract must be applied after feathering as well as before it.

Prior neck mask/depth failures and the mask-normalization epsilon fix mean old
“no visible effect” findings cannot certify today's behavior. This report does
not rerun or upgrade those historical validation claims.

## 3. Why a joint physical solve is not yet a defensible promise

Even the simplified linear diffuse model `I(x) = R(x) * S(x)` admits
`R'(x) = R(x) * k(x)` and `S'(x) = S(x) / k(x)` for positive k.
Bounds and regularizers can choose a solution without proving it was observed.
Assuming equal reflectance across regions adds a prior; it does not establish
that the photographed regions actually have equal reflectance.

For portraits, a more useful conceptual model includes colored illumination,
visibility, specularity and transmission, plus camera rendering. Different
surface orientations can see different portions of the same environment light.
A shared illumination environment therefore does not imply equal pixel color
or one shading scalar for face, neck and hands. Ear transmission is particularly
poorly represented by simple diffuse median matching. These are modeling
limitations, not evidence that every image requires a full physical renderer.

The reviewed intrinsic literature explicitly describes single-image separation
as under-constrained. Older intrinsic tooling assumes sRGB inputs; newer work
separates colored diffuse shading and a residual rather than assuming grayscale
illumination. Neither establishes a validated whole-body cosmetic correction
for this application's portraits. [Bell et al. author implementation](https://github.com/seanbell/intrinsic),
[Careaga and Aksoy project and publications](https://yaksoy.github.io/intrinsic/).

## 4. Primary sources and their practical implications

| Source | Evidence relevant to the decision | Limit for Retouch |
|---|---|---|
| [Weyrich et al., Analysis of Human Faces using a Measurement-Based Skin Reflectance Model, 2006](https://www.merl.com/publications/TR2006-071) | Uses custom measurements of geometry, reflectance and subsurface scattering across 149 subjects. Shows what measured skin modeling actually requires. | Facial capture research; not a method for identifying all body-region reflectances from one edited JPEG. |
| [Bell et al., Intrinsic Images in the Wild, 2014 — author code](https://github.com/seanbell/intrinsic) | Established intrinsic-decomposition baseline with an explicit sRGB input convention and reflectance/shading outputs. | A decomposition baseline, not skin truth or a ready cross-person ownership solution. |
| [Careaga and Aksoy, Ordinal Shading, 2023 / Colorful Diffuse Intrinsics, 2024 — author implementation](https://github.com/compphoto/Intrinsic) | Global/local shading cues and colored diffuse shading address limitations of a scalar smooth-luminance decomposition. Separates residual components. | Candidate comparison only. Repository states academic use only and intellectual-property restrictions; do not treat it as an available production dependency. No weights were acquired. |
| [HumanOLAT, ICCV 2025](https://vcai.mpi-inf.mpg.de/projects/HumanOLAT/) and [paper](https://people.mpi-inf.mpg.de/~prao/papers/human_olat/paper.pdf) | Full-body multi-view, multi-illumination capture offers a route to controlled lighting evaluation: 21 subjects, three poses and 40 views. | Frames/views are not independent people. Exposed-skin coverage, terms and suitability for makeup/skin preservation need inspection before use; dataset acquisition was not performed. |
| [MediaPipe Image Segmenter documentation](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) | SelfieMulticlass labels face skin, body skin, hair, clothes and other categories; confidence outputs are available. | Its six categories do not label individual subjects or distinguish hands/chest/ears. The documented 256×256 input does not certify native-resolution skin boundaries or confidence calibration. |

Search was bounded to intrinsic ambiguity, skin measurement, controlled
full-body lighting evidence and segmentation capabilities. No general beauty,
aging-vector or generative face-replacement survey was conducted. Source
publication years above come from the publications, not crawler timestamps.

## 5. Proposed first experiment

### Separate the two edit objectives

**Edit propagation:** source region differences were acceptable, but the face
edit introduced a mismatch. Compare the face before/after edit and transfer
only an approved broad color change to eligible same-person regions. Do not
copy the face's final absolute color or its makeup-specific correction.

**Pre-existing mismatch correction:** the source already has an unwanted
regional cast or boundary. Require an explicitly approved reference/target and
comparable lighting evidence. Face edit deltas alone cannot fix this case.

These objectives need different labels and evaluation targets. Reducing raw
face/body DeltaE is not proof that either objective was achieved.

### Arms, holding masks and output encoding fixed

| Arm | Purpose |
|---|---|
| Source / disabled | Exact identity reference and existing regional differences. |
| Conversion only | Isolate color-space and quantization effects. |
| Current neck/body operators, separately and combined | Measure the actual reachable baseline and interaction/double-treatment risk. |
| Manual-mask bounded chroma correction with luminance held fixed | Simple appearance baseline; preserving L does not by itself preserve colored lighting. |
| Approved face-edit delta propagated by region | Test whether relative editing preserves source lighting relationships better than absolute median matching. |
| Region-conditioned regularized correction | Optional second stage: connect only same-person regions with compatible reference/lighting evidence; allow regional offsets and disconnected regions. |
| Intrinsic-conditioned diagnostic | Later optional comparator with independently qualified inputs and permitted dependencies; never use estimated albedo as ground truth. |

For the first candidate, record each region's owner, support, protected pixels,
reference, source statistics, approved change and skip reason. Begin with
manual masks to evaluate the correction independently of segmentation.
Reject mixed ownership, absent references, material ambiguity and clipped or
very low-SNR reference patches. When uncertain, return the source region.

The proposed region graph is an organization of evidence, not a claimed
physical solver. Nodes are regions, and edges express accepted same-subject
comparisons. A robust penalty can discourage disagreement in *approved edit
deltas* across compatible edges; regularization toward zero correction and
explicit protection preserve unsupported nodes. Do not penalize all original
region-color differences toward zero. No graph weights, thresholds, pixel
budgets or default strengths are calibrated by this report.

### Integration boundaries to resolve before a production proposal

Capture source reference statistics before face changes, retain them alongside
per-person face results, and apply a region correction once. Inventory overlap
with existing neck/body support before choosing a stage location. Any global
grade applied between source/reference measurements must be accounted for;
statistics in different processing states are not directly comparable.

Keep float processing and color-space conventions explicit. Apply the complete
protection mask after interpolation/feathering, restore outside-support pixels,
and require exact disabled/abstention output before export. Compare separately
after export, since JPEG compression can change untouched neighbors. Geometry,
pores, marks and clothing are not editable merely because they fall inside a
person mask.

## 6. Evidence and acceptance plan

Use controlled synthetic fixtures to distinguish same reflectance under
different lighting from different reflectance under the same lighting; include
colored lights, shadows, clipping, specular/transmission surrogates, neutral
fabric, muted cosmetics and multi-person ownership collisions. Synthetic
truth tests the declared model, not authentic skin appearance.

Build a manually reviewed portrait manifest covering face/neck/chest/hands,
backlit ears, palm/back-of-hand differences, makeup, tattoos, colored event
light, soft/deep shadows, touching people and partial occlusion. Include a
broad range of observed skin colors and exposure conditions. Do not infer
demographic categories from pixels or count variants of one photo as new
subjects. Split development and held-out cases by person/shoot.

The existing cosplay plan and weekly handoff record exposed-body/darker-skin
coverage gaps. No new corpus audit was done here; a current asset inventory
is required before declaring any subgroup or region adequately represented.

| Measure | Interpretation / acceptance boundary |
|---|---|
| Protected and outside-support pixel deltas | Exact zero before export; any nonzero delta fails the isolation contract. |
| Wrong-person edit support | Any confirmed cross-person propagation fails, irrespective of average visual improvement. |
| Boundary change profile | Measure newly introduced seam gradients and halos; keep real source boundaries separate. |
| Shadow and ear-highlight relationships | Compare against approved preservation targets, with signed regional luminance/chroma changes; do not reward equal medians. |
| Texture/mark survival | Inspect native aligned regions and frequency-band changes with noise controls; energy alone does not establish pore authenticity. |
| Target color error | Evaluate only accepted comparable patches or controlled truth; raw face/body DeltaE is descriptive elsewhere. |
| Risk versus coverage | Report apply/dampen/skip/review fractions and harmful-edit rate per scenario. An all-skip candidate is safe but offers no demonstrated utility. |
| Human review | Randomized source/baseline/candidate pairs at native size and delivery size; record preference and preservation harm separately. |
| Runtime and memory | Measure native-resolution costs when prototypes exist; no performance estimate is established here. |

Only the exact support/ownership contracts are hard criteria now. Perceptual
and numerical improvement margins must be set on development evidence, frozen,
then evaluated on held-out cases. Candidate promotion requires benefit over
both disabled and simple bounded baselines without preservation regressions;
an aggregate average cannot excuse damage on a region or lighting condition.

## 7. Next work and current boundary

1. Inventory and label existing candidate photographs; record missing regions
   and lighting cases. Produce the fixed evaluation manifest.
2. Run an isolated manual-mask baseline audit with conversion controls and
   native visual plates. Keep segmentation quality out of the first comparison.
3. Compare approved edit-delta propagation against absolute median matching.
   Stop or narrow scope if the candidate cannot preserve genuine differences.
4. Only after a correction benefit exists, study automatic skin support,
   per-person ownership and calibrated abstention. Extend to mixed lighting,
   ears and texture parity as separately evaluated capabilities.

Completed in this task: live source review, current-code inspection and this
research/experiment brief. No photos were changed, models downloaded, or
production/recipe/GUI path enabled. P7 is ready for a bounded experiment
proposal, not a production-quality claim.

**2026-09-09 addendum:** the isolated primitive described as a candidate in
§5 (edit-delta propagation, bounded LAB deltas, exact abstention) was
implemented the same day as `retouch/cross_region_skin.py` and wired as an
opt-in engine stage (`ProcessingContext.cross_region_skin`, default `0.0`,
no `ParamSpec`/CLI flag/recipe/GUI control sets it — verified by grep against
`params.py`, `cli.py`, `recipes*.py`). This is a leaf primitive matching the
"Edit propagation" arm's mechanics, not completion of the plan: none of §7's
next-work items (photograph inventory, manual-mask baseline audit,
edit-delta-vs-median-matching comparison, synthetic fixtures) have been run.
Tests: `tests/test_p7_cross_region.py` (contract/wiring: registry order,
`body_match_face` mutual-exclusion abstention, opt-in defaults) and
`tests/test_cross_region_skin.py` (input validation, per-abstention-reason
coverage, `infer_same_person_skin_support` ownership/exclusion). On a real
single-face portrait (DSCF8007) with no reviewed target mask supplied, the
stage does not abstain — it applies the auto-inferred candidate support; this
is exercised, not just asserted possible.
