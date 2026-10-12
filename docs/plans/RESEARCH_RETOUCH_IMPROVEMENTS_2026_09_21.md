# Retouch improvement research — 2026-09-21

Research and documentation only. No production edits, test runs, model downloads,
photo processing, new annotations, or visual acceptance were performed.

Later September 21 follow-up: the working tree now contains a pre-face QA
reference/provenance implementation made after this research pass. The
[QA validity follow-up](RESEARCH_RETOUCH_QA_VALIDITY_2026_09_21.md) reviews its
remaining measurement limitations and links a documentation-only validation
protocol. Findings below retain their original frozen-main baseline; they are
not a claim that the old reference-selection behavior remains unchanged in the
current dirty checkout.

## Recommendation

Retouch's best next face-quality study is **preservation-aware correction with
evidence covering the whole edit**. The existing engine already has extensive
skin, texture, mark-policy, parsing, and color machinery. The next decision is
which operations improve the requested result while preserving the photographed
marks, makeup, fine detail, and person boundaries.

This follow-up adds two concrete findings to the September 15 research:

1. The built-in reference-based QA compares against an image that has already
   undergone face processing. Its scores cannot establish preservation across
   those earlier edits.
2. The September 16 EXR additions contain identifiable highlight/range problems,
   and the new gamut control is not an end-to-end wide-gamut delivery contract.

A newly reviewed 2026 additive-blend-map paper also suggests a useful bounded
algorithm study. It does not establish a better result on Retouch photos.

## 1. Exact baseline and completed work

The checkout is `feat/color-science-k9-fix-and-frontier` at
`cdfd6fe5b65794e963480117d696ff0b8aff828a` (September 8). Newer work is available
on local `main` at `6795b7bf2e9f5ecf9b0ea60cdde3d14cf0781fb7` (September 16).
The review inspected both without switching branches. Unless explicitly stated
otherwise, source line references below mean **that frozen local main revision**,
not the older file currently visible in the checkout. Remote state was not fetched.

For example, a finding can be inspected read-only with:

```bash
git show 6795b7bf2e9f5ecf9b0ea60cdde3d14cf0781fb7:retouch/io.py
```

The following are existing foundations, not new feature recommendations:

- P7 bounded face-to-body edit propagation, its opt-in control, and P8 observable
  appearance readout are present on local main.
- FA-02 has legacy and experimental restoration paths. Its eligibility thresholds
  still explicitly declare themselves uncalibrated.
- Mark protection is wired into frequency processing and selected skin operations.
  A narrow ambiguity fix already exists; general real-photo classification remains
  an evidence question.
- Inventory merge, baseline-decision freeze, and source/output QA comparison
  scripts have landed. Saved outputs from those tools are present.
- Recent eye-v0 precision and multi-face worker-shutdown fixes are already in main.

The September 15 report and plan are available in main as
`docs/plans/RESEARCH_RETOUCH_REMAINING_GAPS_2026_09_15.md` and
`docs/plans/PLAN_RETOUCH_GAP_RESEARCH_2026_09_15.md`. The September 11 P9 documents
remain an unconfirmed proposal; this report does not create or approve a new
roadmap item.

### What the saved evidence now establishes

| Saved artifact | Observation from this read | Boundary |
|---|---|---|
| [Merged inventory](../../test_output/merged_inventory_2026_09_15.json) | 10 unique source hashes, 8 recorded person IDs, all `dev`; tone labels only `light` and `very_light` | These are saved metadata, not a new identity, consent, image, or annotation review |
| [Baseline freeze](../../test_output/baseline_freeze_2026_09_15.json) | 10 processed rows, `natural`/`full`, recorded commit `ea8d584165be70f85b6f3f7644ffce841abe91e9`; one multi-face asset | Not the September 16 color revision; no automatic transfer of its evidence to that revision |
| Same freeze | All rows declare `ground_truth_available=false`; all P7 diagnostics are empty | The script hardcodes the ground-truth field; it is not a fresh census proving that no legacy patch annotations exist. P7 is disabled under this recipe, not experimentally rejected |
| [Source/output QA check](../../test_output/qa_signal_source_check_2026_09_15.json) | Banding, plastic-skin and asymmetry flag states are unchanged source-to-output on 10/10 assets for each detector | Scores do change; an unchanged Boolean cannot establish unchanged appearance, no damage, or a false alarm |

The source/output checker uses the same returned **skin mask** for each pair.
Built-in QA supplies the person mask to its general detector argument and skin
separately. Therefore its numbers are not a direct replacement for every
built-in baseline score. The saved checker also lacks a recorded code revision.

## 2. Ranked opportunities

| Priority | Improvement to investigate | Why it comes next | Evidence today |
|---|---|---|---|
| First, before selecting algorithms | QA reference scope, provenance and meaningful change measurements | A candidate cannot be selected reliably using evidence that excludes its stage | Confirmed source wiring; saved scores support the concern |
| Before depending on new HDR/wide-gamut delivery | EXR highlight/range handling and explicit output color contract | The newest feature work contains deterministic contract problems | Source/algebra findings; no runtime reproduction this turn |
| Highest face-quality research value | Intentional mark/makeup preservation and edit eligibility | Existing masks and policies need independent evidence and an end-to-end preservation check | Prior studies and current wiring; no new measured error rate |
| Next algorithm comparison | Bounded correction maps versus smoothing plus residual restoration | Could reduce the need to remove detail and then add it back | New literature lead; Retouch benefit untested |
| Product qualification | Preview/export target and disposition agreement; P7 ownership | Users need the same intended person and operation at delivery | Current source concern; no new GUI reproduction |
| Conditional challenger | Better face parsing and learned localization | Useful only if reviewed boundary/localization failures justify the cost | Published methods, not validated replacements |

## 3. QA should identify exactly which edit it measures

### Confirmed reference boundary

In main, `retouch/engine.py:2394–2425` composites the per-face results and passes
that image into `_run_global_phases`. The ordinary path does the same at
`2869–2880`. Inside the function, QA receives `result` and its `img_bgr` argument
as the reference (`2745–2749`). That reference is **post-face, pre-global**.

Consequently, reference-based texture, mark-retention, color-drift and perceptual
change measurements there do not measure the already-completed face changes.
Single-image detectors can still flag a processed face; this finding does not
mean all QA is inactive. Neural boosters also run after this QA (`2757–2761`),
so these records do not cover their optional final changes.

The saved freeze has zero color drift and mean face SSIM contrast/structure of
1.0 in all 10 rows. Those are saved observations consistent with the narrower
reference scope, not proof that source and final photos are identical.

**Proposed improvement:** retain distinct measurement boundaries:

- An aligned pre-face reference versus post-face result for skin/mark preservation.
- Pre-global versus post-global for grading effects.
- Original versus final delivery for the complete requested result, accounting
  explicitly for any intended reshape and checking the decoded export separately.

Give every score its reference stage, image/support hashes, units, availability
reason, and code/model provenance. Keep source-condition warnings separate from
new or increased edit damage. Existing `qa_signal_source_check.py` already starts
the paired-source work; extend that foundation instead of inventing a second QA
framework. Follow continuous score changes as well as threshold crossings.

### Small but definite provenance issue

`scripts/qa/baseline_freeze.py:154` copies `result.timings` into a field called
`timings_seconds`. Engine stage timings are recorded in milliseconds, for example
`engine.py:2371,2385`. GUI metadata correctly calls the same values `timings_ms`
(`gui.py:1578`). The saved first row's `total=6096.801041` is therefore about
6.097 seconds of recorded stage time, not 6,097 seconds. Its separately measured
`wall_seconds=17.163` is a different measurement with additional overhead.

Correct units and provenance would make later optimization research interpretable.
This report does not alter the script or saved JSON.

## 4. Color/export findings from the newest main commit

### A. EXR highlights collapse rather than roll off

`retouch/io.py:347–353` uses, independently on each nonnegative channel:

```text
mapped(x) = x / (1 + max(x - 1, 0))
```

For `x > 1`, the denominator is `x`, so `mapped(x) = 1`. Linear values 1, 2,
4 and 8 all become the same encoded white. This is an algebraic result from
the source, not a rendered experiment. It contradicts the comment describing
a highlight-preserving shoulder.

The existing `test_hdr_shoulder_not_clipped_to_flat` in
`tests/test_wiring_tranche_2026_09_16.py:39–48` actually expects both 1 and 4
to become 255. That assertion cannot verify the behavior named by the test.

A further design constraint matters: a continuous monotone mapping cannot keep
every value in `[0,1]` unchanged, map into `[0,1]`, and still give values above
1 distinct brighter outputs. A real SDR shoulder must reserve headroom below 1,
change exposure, or use an HDR working/display path. Choose that contract before
choosing a replacement formula.

### B. EXR writing ignores its resolved float range

`write_image_with_icc` resolves `float_range` at `io.py:1384`, but the EXR branch
unconditionally divides samples by 255 at `1392`. A caller supplying normalized
float input with `float_range="unit"` will have its values divided again. This
does not apply to the tested uint8 byte-range case; it is an uncovered accepted
input path, including the unit-range output of a source-profile conversion.

### C. PQ EXR and wide-gamut controls need a complete delivery contract

The EXR writer optionally PQ-encodes samples; the EXR reader always interprets
them as linear and applies sRGB encoding. These entry points do not implement a
paired PQ-aware round trip, and the inspected writer does not explicitly supply
transfer-function/color-space identification. Its current PQ test only checks
numeric bounds after manually calling the inverse function.

OpenEXR's primary documentation describes scene-linear RGB values, values above
1, and explicit color-space identification; it also notes that the library can
store non-linear data despite the linear convention. Thus the issue is the
reader/writer and metadata contract, not an assertion that the container cannot
hold PQ values. [OpenEXR technical introduction][S7]

Similarly, `gamut_target` chooses an sRGB/P3/Rec.2020 compression boundary, but
`grading.py:600–614` still converts through BGR/OKLab helpers and clips the input
used to calculate the boundary decision. The engine's public delivery boundary
still calls `to_uint8` (`engine.py:2734–2740`). Selecting the wider boundary does
not establish wider output primaries or recover clipped/quantized information.
Explicit source-profile export conversion is a separate existing feature.

**Proposed improvement:** specify input primaries, transfer function, float range,
working precision, display mapping, destination encoding and metadata together.
Qualify the SDR contract independently of any future HDR route. The current
ITU index identifies BT.2100-3 as the in-force HDR recommendation; its full PDF
could not be retrieved in this pass, so no clause-level compliance claim is made.
[ITU recommendation record][S8]

## 5. Face preservation: complete the next evidence step

Keep the existing mark policy and FA-03 work. Main already compiles preservation
masks and clips feathered support to skin (`perf_optimizations.py:416–441`). The
remaining question is whether the proposed material and intended action are
correct, and whether the complete recipe honors that intent.

The next annotations should distinguish **what is visible**, **whether it should
change**, and **where that decision is supported**. A mole is not automatically a
removal target; an eyeliner fragment is not automatically a natural mark. Retain
uncertain material and uncertain intent explicitly. Capture One's documented
blemish-protection tool is a useful workflow comparison for preserving a chosen
area, not evidence that its detector solves these distinctions. [Vendor source][S6]

Compare current FA-03 rules, existing context/shape research arms, and manually
accepted supports. Measure missed blemishes separately from incorrectly edited
marks or cosmetics. Include real near-eye positives, occluded boundaries, fine
hair, mixed light and broader observed skin-color conditions. Existing development
images must remain development data; review new subject/shoot mappings rather
than inferring identity or demographic labels from pixels.

Report harmful decisions among accepted cases together with useful coverage,
valid cases rejected and unknowns. This follows the selective-prediction evaluation
question; it does not require installing SelectiveNet. [Primary paper][S5]

The current inventory and baseline scripts are ready foundations. The immediate
missing deliverable is a reviewed preservation/support mistakes table with honest
denominators, not another inventory implementation.

## 6. Texture: a new algorithm lead, with a narrow comparison

Legacy `SkinProcessor.restore_micro_texture` computes
`original - smoothed` and restores a weighted portion in dimensional zones
(`skin.py:1047–1064`). This is not intrinsically a pore-only signal: it can contain
texture, capture noise, pigment edges and intentionally corrected variation.

The existing [FA-02 metric-validity study](RESEARCH_FA02_METRIC_VALIDITY_2026_09_06.md)
reported eligibility flips under resizing on 4/9 development faces and threshold
crossings from added noise. Neither examined replacement metric earned promotion.
Main's gate still labels all thresholds uncalibrated
(`fa02_texture_eligibility.py:22–26,64–79`). Resizing changes resolvable detail;
the goal is a justified response to scale/noise, not forced numerical invariance.

**New literature lead:** Pegu's March 2026 IAAI paper describes a small U-Net
predicting additive correction maps, inspired by dodge-and-burn, with a reported
6 MB model. The transferable idea is to apply a bounded correction to the source.
The publisher abstract is available; full PDF access failed, and this pass did
not establish official deployable weights, model terms, or Retouch performance.
[Publisher record][S1]

The proposed Retouch experiment is an inference from this architecture, not a
reproduction claim: compare a conservative correction-field baseline with current
smoothing/restoration under identical reviewed supports. Start with existing
local tone/dodge-and-burn machinery before deciding whether a learned field adds
value. Require an equal, nonzero requested correction benefit; leaving the source
untouched must not automatically win the comparison.

An additive map does **not** guarantee texture preservation. Its spatial variation,
mask boundaries, clipping and color domain can change detail. Compare preserved
fine hair/pores, corrected-defect reappearance, noise gain, makeup/mark contrast,
boundary seams, naturalness and correction strength separately at native views.
Hold accepted supports fixed before comparing automatic localization.

For existing FA-02 arms, keep `legacy`, `restoration off`, `experimental abstained`
and `experimental executed` distinct. The abstained experimental path skips
restoration; it is not identical to the default `natural` recipe's legacy path.

### Other literature: what is actually worth borrowing

| Method | Useful lesson for Retouch | Recommendation and limit |
|---|---|---|
| [SegFace, AAAI 2025][S2] | Class-specific tokens address rare parsing classes such as accessories | Conditional offline parser challenger after labeling boundary failures; published benchmark results do not establish native cosplay suitability or local speed |
| [Spectral restoration/HGFR, ICCV 2025][S3] | Frequency selection and multiresolution restoration; 25,000 paired examples | Inspect supervision policy before training: the primary indexed paper includes moles/freckles as blemish categories and 23,760 synthetic examples. Its target is not automatically Retouch's preservation intent |
| [BeautyGRPO, CVPR 2026][S4] | Multidimensional preference evaluation | Borrow the review rubric; defer generative replacement. The official inference README uses FLUX.1-Kontext-dev with default size 1024 and separates code and model licenses; neither native texture fidelity nor local deployment is established here |

The older ABPN coarse-to-fine idea is already discussed in Retouch's research.
Do not present correction maps as an entirely new architectural invention, or
claim that an additive/coarse-to-fine design makes detail loss impossible.

## 7. Ownership and preview/export behavior

On main, `gui.py:1319–1325` selects fast preview and non-fast export. P7 requires
exactly one detected face (`engine.py:3940–3945`) and otherwise abstains. Automatic
target support is inferred from the person mask plus face/region exclusions
(`3956–3964`). One detected face does not establish exclusive person ownership.

The inspected GUI latest-render payload records Safe Auto, backend and precision
but not `p7_diagnostics` (`gui.py:1550–1591`). This supports a qualification task;
no new mismatch was reproduced in this session.

Compare the same requested edit at preview and export resolutions, recording
target identity/regions, apply-or-skip decisions, reasons and support overlap.
Include a missed background person, overlapping people, skin-colored fabric,
hands/props and fallback parsing. Preserve diagnostic visibility when a legitimate
resolution-dependent decision changes. Identical preview/export pixels are not
the requirement. Qualify P7 ownership separately from P8 descriptor usability.

## 8. Proposed next study order

These are future tasks, not work run or implementation authorized by this report.

1. **Make the evidence interpretable.** Pin the intended branch/revision, document
   each QA reference boundary, account for timing units and disabled operations,
   and define separate preservation, benefit and source-condition endpoints.
2. **Complete reviewed support/intent labels.** Reuse the existing inventory and
   FA-03 protocol. Produce a per-region mistakes table and missing-strata list.
   Reserve new independent subjects/shoots before tuning candidates.
3. **Run one bounded face comparison when authorized.** With accepted supports
   fixed, compare current processing, restoration off, existing qualified FA-02
   arms and a conservative correction-field baseline. Keep detector evaluation
   separate. No threshold is promoted just to increase the number of edited faces.
4. **Qualify ownership and delivery.** Independently check P7 regions and
   preview/export dispositions. Audit the EXR/range/output-color contract before
   relying on its HDR or wide-gamut claims; it can proceed independently of labels.
5. **Then judge preference and challengers.** Blind the source/current/candidate
   comparison under matched delivery, permit ties and preservation vetoes, and
   evaluate new parsers or learned models only for measured remaining failures.

For future correctness checks, the discriminating examples are: a face-only edit
that moves the appropriate QA score; a protected-mark change that cannot disappear
behind a post-face reference; several distinct HDR highlight values; unit/byte
float representations of the same color; and reader/writer agreement on transfer
encoding and profile. None were executed here.

## 9. Evidence ledger and limitations

Saved artifact SHA-256 values at inspection:

```text
merged_inventory_2026_09_15.json
88738bf98f4b4945b2a895a44213d476d0180d0110cfee9d364a77c109ca1ae1
baseline_freeze_2026_09_15.json
d8eafda4341b30d23d5c9fbe4ca7375bd8c1488bf2d82fe054fe1c25d7cd5c81
qa_signal_source_check_2026_09_15.json
168dbe6711ce6e527bdf7525c57f2a0fd9a354870694f3f795483c7aeafd650d
```

The existing FA-02 metric-validity, FA-03 classification-experiment and P6
noise-aware-clarity reports were unchanged between the inspected checkout and
main. Their measurements remain historical; they were not rerun. The ignored
artifact links above work in this workspace but may not exist in a fresh clone.

This is a targeted source/literature/metadata review, not a full audit, current
benchmark, visual comparison, model qualification, or release certification.
Research conclusions explicitly distinguish source facts, saved observations,
published claims and proposed experiments. Production defects found by inspection
were documented, not fixed.

### Primary external references checked on 2026-09-21

- [S1 — Pegu, additive blend maps, IAAI 2026][S1]: publisher abstract and date;
  full PDF retrieval failed. Reported model size is the author's claim.
- [S2 — SegFace author repository][S2]: method, benchmark context and project.
- [S3 — Xu et al., ICCV 2025][S3]: indexed primary PDF abstract and dataset
  extract; direct opening was unavailable. No full training/reproduction audit.
- [S4 — BeautyGRPO author repository][S4]: inference configuration and license
  boundary; no model downloaded or evaluated.
- [S5 — SelectiveNet, ICML 2019][S5]: risk-versus-coverage evaluation precedent.
- [S6 — Capture One blemish-protection release][S6]: vendor workflow description,
  not an independent performance comparison.
- [S7 — OpenEXR technical introduction][S7]: linear samples, highlight range and
  color identification. Installed OpenEXR binding/version compatibility untested.
- [S8 — ITU-R BT.2100-3 record][S8]: current version/status only; PDF retrieval
  timed out, so detailed standard conformance remains future work.

[S1]: https://ojs.aaai.org/index.php/AAAI/article/view/41481
[S2]: https://github.com/Kartik-3004/SegFace
[S3]: https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Face_Retouching_with_Diffusion_Data_Generation_and_Spectral_Restorement_ICCV_2025_paper.pdf
[S4]: https://github.com/vivoCameraResearch/BeautyGRPO
[S5]: https://proceedings.mlr.press/v97/geifman19a.html
[S6]: https://www.captureone.com/en/explore-features/whats-new/16-6-5
[S7]: https://openexr.com/en/latest/TechnicalIntroduction.html
[S8]: https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en
