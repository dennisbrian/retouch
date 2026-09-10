# P8 — Apparent-age cue validity and research decision

Date: 2026-09-10. Source baseline: `cea7b2e215cd1d4adc85eefe75a68a4f1635484f`.
Status: research complete for this bounded review; experiments remain proposed.
Scope: primary-source reading, static code inspection and existing metadata only.
No production code, tests, model acquisition, photo processing or new age labels.

## 1. Decision

**Continue P8 as a study of observable appearance cues. Defer the age slider.**
First establish whether regional feature contrast and skin-color variation can
be measured on valid support with interpretable uncertainty. Then investigate
dark-line and highlight descriptors. Keep physical translucency and facial
volume estimation outside this first study.

P7 already has a bounded, opt-in implementation and an existing
[research report](RESEARCH_P7_CROSS_REGION_SKIN_CONSISTENCY_2026_09_09.md).
That leaves useful new research work in P8. This report addresses the
[Skin ProMax P8 proposal](PLAN_SKIN_PROMAX.md), not a new feature series.

The earlier [P8 research draft](RESEARCH_P8_AGING_VECTOR_2026_09_09.md) and
[measurement implementation draft](PLAN_P8_AGING_CUE_MEASUREMENT.md) were
untracked at the start of this task and have been preserved. Use this report
and its [revised research protocol](PLAN_P8_CUE_VALIDITY_RESEARCH_2026_09_10.md)
for the current recommendation where the drafts disagree. The earlier drafts'
one-to-two-week estimates are not established delivery estimates.

## 2. Three different questions

| Question | Suitable evidence | What it cannot establish |
|---|---|---|
| What is visible in this photograph? | Valid masks and repeatable image descriptors under specified capture conditions | Actual wrinkle depth, sebum quantity, pigment concentration or chronological age |
| Which descriptors relate to apparent age? | Independent human ratings, subject-separated analysis, nuisance controls and uncertainty | A causal edit direction or a natural-looking result |
| Does a particular edit change apparent age while preserving the portrait? | Controlled before/after intervention, randomized human assessment, preservation review | A universal number of years or biological rejuvenation |

This distinction changes the lighting question. Lighting can change the
appearance being measured; that is not automatically estimator error. A
descriptor intended to estimate intrinsic tissue properties has a different
invariance requirement from one intended to describe the displayed photograph.
Define the intended quantity before demanding lighting invariance.

The proposal does not need to measure every physiological cue to investigate
a perceptual edit. Conversely, having six numerical outputs would not prove
that they form a coherent aging direction. Both are design inferences, not
results obtained on Retouch portraits.

## 3. What the current source actually supplies

These are static findings at the baseline above, not measured failure rates.

| Proposed cue | Source and actual behavior | Research consequence |
|---|---|---|
| Chroma uniformity | [`skin_homogeneity_state`](../../retouch/perceptual_metrics.py#L87) measures the standard deviation of the magnitude of a face-scaled LAB chroma bandpass. Its optional hemoglobin field summarizes a supplied map. | An image-color descriptor; it neither separates lighting from pigment nor estimates hemoglobin. The `pore_energy` field is also a frequency-band descriptor, not verified pore anatomy. |
| Edge definition | [`facial_feature_contrast`](../../retouch/perceptual_metrics.py#L52) measures mean LAB differences between eyes/lips/brows and surrounding skin. | Relevant to published feature-contrast research; it does not measure jaw definition or boundary sharpness. Individual signed channels matter; a single unsigned magnitude can hide the direction of change. |
| Gloss/sebum | [`extract_specular`](../../retouch/specular.py#L48) gates intensity relative to a skin median, multiplies by observed intensity and a chroma gate, then blurs. | A candidate highlight-appearance map. Tone-adaptive gating does not make its magnitude a physical or exposure-invariant gloss measurement; it is not a sebum readout. |
| Wrinkle depth | [`_wrinkle_soften_masked`](../../retouch/skin.py#L656) computes negative Difference-of-Gaussians response on L at lines 724–740. Scale derives from image height at lines 682–686. | Both earlier P8 drafts incorrectly describe multi-orientation black-hat morphology. The existing signal is dark mid-band contrast, not physical depth or a validated wrinkle detector. Reusing it needs a measurement contract, not merely exposing an intermediate array. |
| Translucency | [`apply_sss`](../../retouch/skin.py#L1771) applies an appearance effect; [`decompose_intrinsic`](../../retouch/intrinsic.py#L127) estimates smooth scalar shading. | Neither measures translucency. A future forward model alone would not guarantee that its parameters are identifiable from one photograph. |
| Volume/AO | [`relight.py`](../../retouch/relight.py#L53) contains landmark-derived geometry; [`sculpt`](../../retouch/relight.py#L426) applies shading. | Geometry machinery exists, but no validated fat-pad-volume or ambient-occlusion estimator was established. Neither “easy without P1” nor “requires P1/I3” follows from these functions. |

Additional measurement risks must precede a large sweep:

- `confidence` in the perceptual metrics is a clipped function of pixel count
  ([skin](../../retouch/perceptual_metrics.py#L134),
  [features](../../retouch/perceptual_metrics.py#L162)); it is not a calibrated
  probability of correctness or a confidence interval.
- Mask normalization still uses `max() > 1.0` in
  [perceptual metrics](../../retouch/perceptual_metrics.py#L197) and
  [specular extraction](../../retouch/specular.py#L126). A small float overshoot
  could collapse support or trigger a baseline fallback. This risk was not
  reproduced here. Feature/skin overlap also requires checking: the feature
  metric intersects features with skin, while landmark fallback excludes
  feature regions from skin.
- Constructed zones are not guaranteed observable tissue. The landmark path
  explicitly supplies an empty [neck mask](../../retouch/parsing.py#L876).
  Missing support must remain missing, not become a score of zero wrinkles.
- Existing [specular tests](../../tests/test_specular_finish.py#L62) check
  synthetic detection floors. They do not demonstrate equal descriptor values
  across tones, exposures or real subjects. Coverage based on strictly nonzero
  values in a blurred map also depends on feathering and support conventions.
- Retouch's [float LAB helper](../../retouch/utils.py#L393) uses L on a 0–255
  scale and a/b centered at 128. Record these units explicitly.

## 4. Primary evidence and its limits

Sources were checked live on 2026-09-10. This is a targeted review, not a
systematic literature search. Publication years refer to the papers.

| Primary source | Supported finding | Limit for P8 |
|---|---|---|
| [Porcheron, Mauger & Russell, 2013, Aspects of Facial Contrast Decrease with Age](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0057985) — full article | Examines feature contrast and includes photographic manipulations that changed perceived age. | Controlled female-face study with makeup exclusions; no validation of Retouch masks, metric equivalence, cosplay conditions or an exact-years control. |
| [Porcheron et al., 2017, Facial Contrast Is a Cross-Cultural Cue](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.01208/full) — full article | Extends contrast evidence across four female-face samples and two observer groups; increased-contrast variants were judged younger. | Broader evidence still has a defined population and task. It does not imply every feature/channel has one universal aging coefficient. |
| [Matts et al., 2007, Color homogeneity and visual perception](https://pubmed.ncbi.nlm.nih.gov/17719127/) — primary abstract | Cheek-image homogeneity was associated with perceived age; the study also examined chromophore maps. | Observational association in female skin; the published measurement is not Retouch's LAB bandpass. No conclusion about removing an individual's freckles or makeup follows. |
| [Fink & Matts, 2008, Skin colour distribution and topography cues](https://pubmed.ncbi.nlm.nih.gov/18081752/?dopt=Abstract) — primary abstract | Both cue families influenced judgments; surface topography accounted for more variation in age perception, while color distribution was a stronger health cue in that study. | Distinct perceptual outcomes must remain distinct. It does not prove a six-cue vector or that single-cue editing must look unnatural. |
| [Ganel & Goodale, 2021, Smiling and perceived age across the lifespan](https://www.nature.com/articles/s41598-021-02380-2) — full article | The apparent-aging effect of smiling differed across age groups and faces in the experiment. | Expression is a material confound for dark-line descriptors; temporary expression lines cannot simply be labeled structural aging. |
| [Hernandez et al., 2021, Different light angles and facial assessment](https://link.springer.com/article/10.1007/s00266-021-02314-3) — primary article/results | Controlled lighting changes affected facial assessments, but the reported age differences at 30° and 60° were not statistically significant. | Lighting deserves controlled study; this experiment does not show lighting always dominates apparent age. Nonsignificance is not proof of equivalence. |
| [Weyrich et al., 2006, Measurement-Based Skin Reflectance Model](https://www.merl.com/publications/TR2006-071) — author abstract | Uses custom measurements of geometry, reflectance and subsurface scattering; model parameters vary with subjects and external conditions. | Supports separating measured physical properties from appearance effects. It supplies no unique inverse solution for Retouch's ordinary single photographs. |
| [Kemelmacher-Shlizerman et al., 2014, Illumination-Aware Age Progression](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kemelmacher-Shlizerman_Illumination-Aware_Age_Progression_2014_CVPR_paper.pdf) — paper | Uses age-conditioned image subspaces and flow to synthesize changes in 2D without reconstructing explicit 3D models. | A counterexample to the claim that age synthesis requires either physical capture or modern generative latents. It learns from a photo collection and changes shape/texture; it is not evidence for the present classical Retouch proposal or permission to transfer donor features. |

The strongest practical literature lead is **regional feature contrast**, with
skin-color variation a second candidate. That ordering combines published
perceptual evidence and current source availability; it is not a measured
ranking of Retouch implementations. The reviewed evidence does not establish
a universal gloss, translucency or volume direction suitable for this control.

## 5. Corrections to the earlier study design

**A variance ratio cannot close P8.** High within-person variation can reject
a candidate descriptor for a specified use and capture range. It cannot prove
that no estimator can recover a cue. Between-person spread is not age signal:
it can reflect identity, cosmetics, tone, camera settings or sample composition.
Uncontrolled same-person photos do not isolate lighting alone. Measurement
repeatability and accuracy are separate concepts; changed conditions must be
specified. [NIST TN 1297, terminology](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-d1-terminology).

**Local normalization does not automatically preserve wrinkle sensitivity.**
As an analytic counterexample, let an unclipped zone be a constant baseline
plus a fixed dark-line pattern scaled by positive amplitude a. The DoG response
and the zone's standard deviation both scale with a. Dividing one by the other
can cancel the amplitude change. The earlier plan therefore cannot promise
both increasing amplitude response and invariance from that ratio alone.
Define whether the quantity is relative contrast or absolute image response;
retain the numerator and denominator. This is mathematical reasoning, not a
newly executed synthetic test.

**Correlation cannot answer whether translucency is necessary for natural
editing.** An observational cue study can identify associations and redundancy.
Testing an edit with unmodified translucency requires an intervention and
perceptual assessment. A failed edit also would not prove that translucency was
the missing cause: masks, strength, feature direction and support may be wrong.

**Landmarks are not volume ground truth.** Official MediaPipe documentation
describes relative z scaled under weak perspective and a separate face-transform
space based on camera assumptions and a canonical model. Neither is direct
measurement of tissue loss. The historical failed neck plane is a specific
implementation warning, not a theorem against all landmark geometry.
[MediaPipe Face Mesh documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md).

## 6. Corpus and label readiness

The earlier claim of zero assigned subjects is stale. Existing local pilot
metadata contains owner-confirmed development assignments and person IDs. For
example, [DSCF2650](../../test_output/fa02_pilot_DSCF2650/corpus_manifest.json)
and [DSCF2709](../../test_output/fa02_pilot_DSCF2709/corpus_manifest.json)
explicitly share one subject identifier. These local ignored artifacts are
metadata evidence, not independently reverified identities, controlled lighting
pairs or a complete corpus census. No age fields were present in these inspected
records; the [default manifest vocabulary](../../retouch/corpus_manifest.py#L60)
also has no age dimension. No locked-test images were opened or processed.

Keep **chronological age at capture** separate from **apparent age of a displayed
image**. Store the latter as multiple independent ratings with their dispersion,
presentation conditions and rater counts; a subject-level age band plus
`age_source` loses that distinction. APPA-REAL is a useful annotation precedent:
its official description provides separate actual/apparent labels, individual
ratings and per-image summary uncertainty. It is a candidate reference, not an
acquired or approved Retouch dataset.
[APPA-REAL official dataset description](https://chalearnlap.cvc.uab.cat/dataset/26/description/).

A reliability pilot does not need age labels, although it does need valid
support and reliable pairing. An age-association study does need image-level
apparent-age supervision. There is no reason to make a production manifest
schema change the mandatory first step. Use observed color/exposure coverage;
do not assign physiological Fitzpatrick types or demographic categories from
pixels. Synthetic base-color sweeps test declared image models, not population
coverage or real-skin fairness.

## 7. Recommended next research sequence

1. Specify the quantities, region support and missing-data rules; audit usable
   development pairs and label provenance.
2. After implementation/testing is separately requested, qualify feature
   contrast and chroma descriptors for observable support and capture
   sensitivity. Add dark-line/highlight descriptors only with explicit units
   and normalization hypotheses.
3. Collect or select suitable apparent-age ratings, then study associations
   with subject-separated analysis and expression/makeup/lighting controls.
4. Only if those results justify it, propose a bounded perceptual edit study
   with source, single-cue and combined arms. Human judgments and preservation
   evidence must decide whether there is a useful edit.

No slider, coefficient, exact-years promise, default strength or release date
is established by this review. The companion protocol specifies the future
deliverables and stop conditions. Current completion means a sourced research
decision and documentation, not experimentally validated aging behavior.

## 8. Implementation addendum — 2026-09-10

Following an explicit implementation request, the bounded R1 readout described
above is now available without changing the render path:

- `retouch/aging_cues.py` exposes `measure_p8_cues()` with signed feature LAB
  components, skin chroma variation, units, support counts, and explicit
  unavailable states.
- `scripts/qa/p8_cue_readout.py` performs face detection and parsing, then
  emits provenance/readout JSON without calling `RetouchEngine.process()` or
  writing edited images.
- `tests/test_aging_cues.py` covers support abstention, parser-style feature
  exclusion, mask overshoot, signed components, strict JSON, and API guards.

The implementation does not add an age score, coefficient, slider, recipe key,
chronological/apparent-age labels, translucency/volume estimator, or edit. R0
inventory, R2 apparent-age association, and R3 perceptual intervention remain
future research gates as specified in this report.
