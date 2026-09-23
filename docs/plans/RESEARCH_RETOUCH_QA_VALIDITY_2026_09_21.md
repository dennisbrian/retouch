# Retouch QA validity and preservation research — 2026-09-21

Documentation only. This pass reviewed local source and primary literature; it
did not change code or tests, run the pipeline/tests, process photos, download
models, annotate people, or make a visual-quality acceptance decision.

This continues the [earlier September 21 research](RESEARCH_RETOUCH_IMPROVEMENTS_2026_09_21.md)
after the working-tree QA-reference implementation. The proposed next study is
specified in the [validation protocol](PLAN_RETOUCH_QA_VALIDATION_2026_09_21.md).
Neither document authorizes implementation or changes the P1–P8/FA roadmap.

Next research tranche: [protection across operation boundaries](RESEARCH_RETOUCH_PROTECTION_LIFECYCLE_2026_09_22.md)
traces final repair masks, permitted donor material/fallback, persistent mark
decisions, restoration-stage ownership, and combined under-eye preservation.
It is also documentation only; the findings here remain the QA prerequisite.

## 1. Recommendation and evidence boundary

**Validate what each QA measurement actually measures before tuning retouch
strengths or adding a learned correction method.** The new pre-face snapshot is
useful: reference-based diagnostics can now observe earlier face edits. It does
not by itself establish reliable preservation measurement, correctly attributed
warnings, or final-file quality.

The most important new findings are:

1. Plastic-skin reference loss is reported but does not determine its flag or
   automatic back-off. Existing source texture and edit-induced loss remain
   conflated in that decision.
2. Same-shaped images are not necessarily geometrically corresponding images.
   Reshape can contaminate same-coordinate photometric comparisons.
3. Several skin-labelled checks receive a person mask; the pore-spectrum check
   ignores its mask entirely. Face-local damage can be diluted by other content.
4. The sweep can merge a flag from one comparison with a score from another.
   Stage provenance currently travels alongside, not as the identity of, each
   individual measurement.
5. Missing measurements can resemble passes, and core QA does not cover every
   later processing/export stage.

These are source-level findings and analytical implications, **not measured
failure rates or newly observed damaged photos**.

### Snapshot reviewed

- Checkout: `feat/color-science-k9-fix-and-frontier`.
- HEAD: `cdfd6fe5b65794e963480117d696ff0b8aff828a`.
- Local main: `6795b7bf2e9f5ecf9b0ea60cdde3d14cf0781fb7`; not checked out or fetched.
- Source references below mean the **dirty working tree**, including the new
  pre-face-reference/provenance edits, not either commit alone.
- Existing production/test edits and the staged `CLAUDE.md` were preserved.
- The previous implementation's reported test passes are historical wiring
  evidence, not newly rerun tests or proof of perceptual validity.

Selected file SHA-256 values identify the reviewed uncommitted source:

| File | SHA-256 |
| --- | --- |
| `retouch/engine.py` | `d7a9883ce2dfdfdd78e70d24f4da71be8fc601a4f47bec180d8ed896b153645d` |
| `retouch/qa_detectors.py` | `5e9b45372206b9208259a9348a7a8905dffde828a07c1c25741ed4efcd841ee3` |
| `retouch/geometry.py` | `5665cec6e3ae775372bcaea86c7ec548924880b2e3a8e9c57cdcc4a0e806338b` |
| `scripts/recipes/recipe_sweep.py` | `5a650d6bf74cef63ca8dc7d3396470f4532830b1281310058954e8bfcbd69dea` |

## 2. What the reference implementation establishes

In [engine.py](../../retouch/engine.py), `process()` captures a copied image at
line 1882, after input preprocessing and before reshape/per-face work. The
resolver at lines 1015–1077 labels a genuine capture separately from its
post-face/pre-global fallback. Results, GUI evidence, and sweep records expose
the new provenance. This closes the earlier reference-boundary visibility gap.

Important limits remain:

- This is not the untouched original: fast-preview reduction, input conversion,
  enabled denoise, and requested healing can precede capture.
- The validity check tests array dimensions/shape, not coordinate correspondence,
  semantic support, working color domain, or the actual image content hash.
- `reference_available=False` means the requested pre-face capture was not
  available; a fallback reference array still exists. Consumers must not confuse
  these two meanings.
- The main wiring test at [test_qa_pipeline_wiring.py](../../tests/test_qa_pipeline_wiring.py),
  lines 113–144, replaces the detector runner and verifies the copied reference
  and stage labels. That is useful capture/wiring evidence, not detector
  sensitivity, alignment, or preservation validation.

## 3. Reference loss is not yet the plastic-skin decision

In [qa_detectors.py](../../retouch/qa_detectors.py), lines 479–521,
`detect_plastic_skin()` computes `energy_loss_vs_reference`, but `flagged` is
derived only from the processed image's high-/mid-frequency energy ratio.
`_process_pipeline()` in [engine.py](../../retouch/engine.py), lines 2884–2940,
uses the `plastic_skin` flag to trigger bounded parameter back-off.

Therefore, changing only the reference can change the reported loss without
changing this flag. This is a direct dependency observation, not an experiment.
It also means the new capture alone does not make back-off distinguish an
already low-texture source from newly erased detail. The opposite error is
possible too: added noise/sharpening can increase energy without restoring the
photographed detail. Neither direction has been quantified in this pass.

Proposed study: retain separate source-state, change-from-reference, and
human-labelled harm endpoints. Test them on fixed skin supports before proposing
any new trigger. Do not simply replace one uncalibrated threshold with another
or interpret the code's historical “guarantee” wording as a verified guarantee.

## 4. Separate geometry from photometry

### Current measurement is not a reproduction of the published metric

`perceived_retouching_vector()` in [qa_detectors.py](../../retouch/qa_detectors.py),
lines 173–253, uses the warp field for geometric statistics. Its SSIM-like and
high-pass comparisons still operate on the unaligned image arrays. A fixed 9×9
high-pass response ratio is not a locally estimated blur/sharpening filter.

Kee and Farid's paper compares photometry after geometric alignment and locally
estimates a regularized 9×9 filter. Its purpose is modelling perceived alteration,
not certifying that edits preserve all wanted detail. The paper also describes
failures involving semantically important changes such as makeup. Thus the
current vector should be treated as an inspired diagnostic, not a reproduced,
calibrated perceptual rating. [Kee and Farid, PNAS 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3250123/)

### Do not replay the saved displacement as though it were an exact warp

[geometry.py](../../retouch/geometry.py), lines 337–359, samples the source with
`map_x/map_y`, then blends the warped and original images with the oval mask.
It saves `(grid - map) * alpha` where that blend applies. OpenCV's `remap` uses
destination-to-source sampling coordinates with specified interpolation and
border handling. [OpenCV 4.11 transformation documentation](https://docs.opencv.org/4.11.0/da/d54/group__imgproc__transform.html)

For a general image, the two expressions below are not equal:

```text
actual blend:        (1 - a(x)) I(x) + a(x) I(M(x))
scaled-field replay: I(x + a(x) (M(x) - x))
```

This is an algebraic limitation: blending sampled values is not generally the
same as sampling at a blended coordinate. Quantization and interpolation add
further differences. The stored displacement is useful as a descriptor but is
not sufficient to reconstruct this exact blended output.

Proposed contract: preserve an actual post-reshape snapshot, or replay the
original map **and** alpha, interpolation, borders, ordering, and numeric domain.
Use that geometry-only counterfactual to assess later photometric edits. Keep a
separate geometry/likeness endpoint; aligning photometry must not hide an
unwanted shape change. Record uncertain/unmatched support as unavailable rather
than optimizing a free optical-flow fit until real edits disappear.

## 5. Make measurement support explicit

`run_qa_with_evidence()` passes `skin_mask=person_mask` at
[qa_detectors.py](../../retouch/qa_detectors.py):1399–1408. `run_all()` passes that
general mask into plastic-skin and color-drift checks, among others. Its separate
face mask is used by harmony/perceived-retouching paths; it does not retroactively
make every detector face-local.

`detect_pore_spectrum_distance()` explicitly leaves `skin_mask` unused
(lines 927–989), calculates whole-image spectral energy, and clips the ratio at
one. This is not a measurement of individual photographed pores. Background
texture can dominate; a small face's loss can be diluted; increases above the
reference are collapsed. If reference band energy is below its epsilon, the
code sets ratio zero and score one—even if both inputs have no measurable
energy. This is an analytical edge case, not a test performed here.

Proposed requirements:

- Distinguish person, face skin, intended edit support, protected marks/makeup,
  and operator exclusion bands. Semantic skin is not permission to erase detail.
- Freeze reference-derived, reviewed supports for paired comparisons. Record
  face/person ownership and coordinate transforms; avoid output-selected masks
  that can remove damaged pixels from the measurement.
- Measure per face and report worst-face outcomes as well as aggregates.
- Use explicit band scales and native pixel/face scale. Erode measurement support
  to account for filter footprint; hard mask edges must not become fake texture.
- Treat low signal, insufficient support, noise, and clipping as separate
  eligibility/confounding conditions, not automatically “pores erased.”

The earlier [FA-02 validity study](RESEARCH_FA02_METRIC_VALIDITY_2026_09_06.md)
already documents resize/noise sensitivity. More complex metrics do not remove
the need to establish support and measurement validity first.

## 6. Bind provenance to observations, not just the enclosing result

In [recipe_sweep.py](../../scripts/recipes/recipe_sweep.py):767–792, a separate
runner compares the returned engine image with the supplied original input.
That differs from the engine's post-preprocess-to-pre-neural pair. Its mask
selection at lines 276–294 can also differ: if no result-owned person mask is
available it falls back to `skin_mask`.

`_merge_qa_observations()` at lines 304–334 joins by detector name alone. It can
retain the new runner's numeric score while overlaying the engine warning's
flag, threshold, and message. Nested originals are retained, which is useful,
but the merged observation can describe no single actual measurement. Passing
through engine `qa_provenance` does not identify the sweep runner's own pair.
Also, `_extract_raw_qa()` does not include the `qa_evidence` attribute in its
lookup list; the context path provides a separate legacy overlay.

Proposed observation identity:

```text
detector + implementation/version + reference/comparison content IDs
+ stage pair + support/owner ID + alignment/scale/color contract + run/iteration
```

Only merge metadata for identical identities. Otherwise retain separate
engine-stage and sweep/delivery observations. An overall disposition may combine
them by a documented policy, but must not fabricate a score/flag combination.
Version any future schema change and keep old observations visibly legacy.

## 7. Execution, availability, and delivery are different claims

| Source observation | Research consequence |
| --- | --- |
| Engine calls the reference resolver with `qa_ran=True` before invoking QA (`engine.py`:2728–2739); missing person support returns all detectors not-run (`qa_detectors.py`:1389–1390) | “Runner invoked” must be separate from measurable/checked detector counts |
| Color-drift and pore-spectrum missing/mismatched-reference branches return zero scores and `flagged=False` without explicit unavailable status (`qa_detectors.py`:795–814, 952–970) | `_qa_status()` can classify an unmeasured comparison as passed; null measurement plus a reason is more truthful |
| Sweep `coverage_complete` follows nonempty runner/raw observations, not a required-detector/measurability audit (`recipe_sweep.py`:385–415) | Structural evidence presence and the `certifying` Boolean do not establish visual qualification |
| Engine provenance names neural boosters as the post-QA stage (`engine.py`:1068–1073) | Other possible later stages also need an explicit coverage boundary |

Later stages include enabled neural boosters, fast-preview upscaling, optional
style-profile LAB shift, export super-resolution, and GUI resizing/encoding
([engine.py](../../retouch/engine.py):1945–1955, 1990–1996, 2746–2751;
[gui.py](../../gui.py):1653–1694). These are conditional paths, not a claim that
they executed in a particular run.

Future evidence should distinguish requested, executed, skipped, failed, and
checked stages. Use separate pairs for face work, global finishing, and final
decoded delivery. Do not sum their nonlinear QA scores as an attribution method.
For the decoded export, compare both against the intended pre-encode delivery
buffer and against an appropriately registered source; these answer different
questions. Preview appearance is not native export proof.

## 8. Literature follow-up: bounded corrections and preservation metrics

### Additive correction maps: useful design, not automatic texture safety

The earlier report could access the abstract only. This pass retrieved indexed
publisher-PDF sections describing 512×512 inputs, a three-channel map in [0,1]
with 0.5 as neutral, and `R = I + 2(B - 0.5)`. These details support studying
correction prediction rather than full-image reconstruction. Direct PDF fetch
still timed out; training/evaluation completeness, available weights, and
deployment rights were not verified. No model was downloaded. [Pegu, AAAI 2026,
indexed publisher PDF](https://ojs.aaai.org/index.php/AAAI/article/download/41481/45442)

Our mathematical inference, not a paper result: for a linear high-pass operator
`H` and an unclipped additive edit `R = I + delta`,
`H(R) - H(I) = H(delta)`. Bounded amplitude alone does not bound high-frequency
change. Further, masking the correction gives
`gradient(a * delta) = a * gradient(delta) + delta * gradient(a)`; support edges
can add structure. Clipping and nonlinear transfer functions further invalidate
an unconditional texture-preservation promise.

The useful experiment is consequently a **bounded, support-confined correction
field with measured spectral/edge effects**, first using a deterministic field
and identical accepted supports. Compare against the current operation and
unchanged input. Only consider a learned predictor after establishing that this
representation helps, with explicit color domain, clipping budget, preserved
marks/makeup, uncertainty, and no-edit fallback. This is a candidate under the
existing bounded-assistance research, not a new production feature commitment.

### Perceptual similarity is auxiliary, not source-detail correspondence

DISTS explicitly tolerates texture resampling and is relatively insensitive to
some geometric transformations. That is useful for perceived texture similarity,
but implies it cannot alone prove that the same photographed fine detail was
preserved. Use, if later justified, alongside registered native-detail and
protected-feature measurements, never in place of them. No DISTS dependency or
experiment was introduced. [Ding et al., DISTS](https://arxiv.org/abs/2004.07728)

## 9. Abstention must be evaluated with benefit and harm

SelectiveNet studies the risk/coverage trade-off for predictions with rejection.
Risk-controlling prediction sets use held-out calibration to control an expected
loss under their statistical setup. These motivate measuring acceptance and harm
together; they do not establish a safety guarantee for Retouch masks or edits.
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html),
[Bates et al.](https://arxiv.org/abs/2101.02703)

For the proposed study, distinguish measurable-support coverage, automatic edit
coverage, harm among accepted edits, and benefit among accepted edits. An engine
that edits nothing can avoid edit harm without meeting the user's purpose.
An engine that edits everything can hide poor selectivity behind an average
quality score. Skip/review reasons and denominators must be visible.

The earlier inventory's 10 assets/8 recorded people, all development, remains
historical context from the earlier report; it was not re-counted here. It is
not a locked, representative calibration/test population. Independent grouped
qualification and broader reviewed appearances are still necessary. Do not infer
ethnicity, clinical skin type, or identity from photos to fill missing strata.

## 10. Proposed order and stop conditions

1. Specify per-observation identity, measurable/unavailable states, and stage
   coverage. Resolve merged-evidence ambiguity before comparing metric values.
2. Validate fixed-support measurements using geometry-only, tone-only, texture
   loss, noise, protected-feature, and low-signal controls. Keep metrics diagnostic
   until they discriminate the intended failure.
3. Conduct a reviewed native-resolution preservation pilot with separate support,
   detector, repair, harm, benefit, and abstention endpoints.
4. Freeze thresholds/policies on separate grouped calibration data; perform locked
   qualification and delivery review before any gate/default promotion.
5. Study correction-field alternatives and blinded preferences only after those
   measurement foundations are credible.

The earlier main-only EXR/range findings remain separate delivery research; this
pass did not port, fix, or retest them. No threshold, default, recipe, algorithm,
model, or release decision is promoted by this document.
