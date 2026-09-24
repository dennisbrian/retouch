# Retouch protection across operation boundaries — 2026-09-22

Research and documentation only. No production/test edits, test execution,
renders, photo modifications, new annotations, model downloads, or Git mutations
were performed. Findings below are source observations and analytical risks,
not newly measured damage rates or visual-quality results.

This continues the [QA-validity review](RESEARCH_RETOUCH_QA_VALIDITY_2026_09_21.md).
The review began September 21 and documentation was completed September 22
(Asia/Kuala_Lumpur).
That review asks whether measurements are trustworthy; this one asks whether a
preservation decision survives the sequence of actual retouch operations.
The [validation protocol](PLAN_RETOUCH_QA_VALIDATION_2026_09_21.md) remains the
measurement prerequisite. No new P/FA roadmap item or implementation is approved.

## 1. Recommendation

The next algorithm-side research priority is **consistent protection through the
whole edit**, before replacing the mark classifier or strengthening smoothing.
Five specific interfaces deserve investigation:

| Priority | Source-supported risk | Proposed first investigation |
| --- | --- | --- |
| First | Blemish-mask closing/dilation occurs after protection is subtracted, without final eligibility clipping | Adjacent approved-repair/protected-feature boundary controls |
| First | Source-constrained PatchMatch falls back to Telea without that donor constraint | Empty/tiny donor-support contract and explicit fallback evidence |
| Next | Different stages re-detect protected marks on differently edited pixels | Source-anchored protection versus fresh observations, with stable IDs |
| Next | Texture restoration includes differences from intervening freckle healing and other edits | Smoothing-only residual versus whole-interval residual |
| Next | Under-eye smoothing, restoration, and later lightness/chroma repair have separate supports/references | Isolated and combined operation review, including makeup and poor reference rings |

These priorities indicate evidence dependency and potential preservation impact,
not measured prevalence. The first two can be specified using known masks before
the difficult question of mark classification is solved.

## 2. Baseline and already-completed work

Reviewed checkout: `feat/color-science-k9-fix-and-frontier`, HEAD
`cdfd6fe5b65794e963480117d696ff0b8aff828a`, with the existing uncommitted QA-reference
work preserved. Source line references mean this working tree. Local main was not
merged or switched to; this is not a current-main or remote-release audit.

Do not reopen these as if nothing had shipped:

- Guided frequency smoothing already accepts mark protection and attenuates its
  output blend. Its guided-only scope is deliberate; bilateral/anisotropic
  protection was not established by the earlier experiment. See the implemented
  section of the [FA-01 report](RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md#11b-implementation-guided-smoothing-mark-protection-shipped).
- The narrow beauty-mark tie/near-tie handling is present in
  [freckle.py](../../retouch/freckle.py):237–263. `ambiguous` maps to `unknown` in
  [marks.py](../../retouch/marks.py):28–39. The older FA-03 report's suggestion to
  add that path is historical, not an outstanding implementation recommendation.
  This is not a general makeup classifier or calibration of additive scores.
- The two dark-circle aliases already use `max`, not sequential application,
  in [perf_optimizations.py](../../retouch/perf_optimizations.py):1087–1096.
  The [under-eye report](RESEARCH_DARK_CIRCLE_OP_2026_09_02.md#7-limitations-and-open-items)
  records the owner's 0.35 darken-removal cap for three recipes. It is not a
  classifier for makeup versus shadow and is not a cap on every under-eye op.
- Policy `remove`/`attenuate` masks are compiled, but the inspected live face
  consumers use `.preserve`. Existing freckle/blemish removal still has its own
  detection and strength semantics. A preset entry is not proof that its named
  action/strength is executed. `recipes.py`:3775–3811 already documents this.

Selected source SHA-256 identifiers:

| File | SHA-256 |
| --- | --- |
| `retouch/perf_optimizations.py` | `504647e772ca6d15a89308398008453aae549c6b159fa8989fe9b2b64a1666ad` |
| `retouch/skin.py` | `e2ac1d20b318bb7a550221ff41bad7c06fbf3518cab2339c785fcf869396552f` |
| `retouch/marks.py` | `0b8192a0f36c1c78d6930e9a91dc033e56801fa571bbf180b185cd6c8a0e438a` |
| `retouch/blemish.py` | `cda1b7f1bd7fdf157e122356127ce8c9d037df2b0e415880eaf490fef7fdcb4e` |
| `retouch/patchmatch.py` | `83a9264e722052d84d4f9781125145e2c5dae990427a9b5ac878bd0a03691b3d` |
| `retouch/undereye.py` | `95b1c5998579cef4066ef9e7a7f8f7746529bec431f6bf40654311b973ae7e6a` |

## 3. Protection before morphology is not final repair containment

### Exact path

1. `perf_optimizations.py`:1048–1074 subtracts mole/policy preservation from
   `blemish_skin_mask`.
2. [blemish.py](../../retouch/blemish.py):166–168 intersects initial dark-spot
   candidates with `skin_mask > 0.3`.
3. Lines 183–188 apply opening and closing. After component filtering, lines
   205–208 dilate the result.
4. The detector returns this expanded mask without intersecting it again with
   eligible skin or the original protected support. `remove()` passes it directly
   to Telea or PatchMatch at lines 100–127.

Closing can fill excluded gaps and dilation can grow an eligible component into
an adjacent excluded pixel. That follows from the morphology definitions, not
from a newly executed repro. The missing final intersection is visible in the
source. Whether a particular face triggers it depends on the mask geometry,
scale, contrast, and component-size filters. [OpenCV 4.11 morphology definitions](https://docs.opencv.org/4.11.0/d4/d86/group__imgproc__filter.html)

### Proposed contract

For destructive repair, define a final hard write domain independently of
proposal generation: it must remain within approved repair eligibility and
outside protected/uncertain regions after **all** morphological expansion.
Keep the actual component support, intended repair margin, and blend alpha
separate. Do not solve a boundary breach by making every protected ellipse
arbitrarily larger; that may hide the breach while disabling useful repairs.

The inspected [blemish-policy tests](../../tests/test_blemish_mark_policy.py)
verify protection at a mocked mark's center and a distant unchanged probe.
Those checks are useful, but do not establish safety for an approved defect
touching a protected mark, a narrow protected gap, or a skin/eye boundary.
They were read, not rerun; this is not a complete test-coverage census.

### Important counterexample to an easy but incorrect diagnosis

`inpaint_and_blend()` feathers its alpha outside the binary mask. This alone
does not prove meaningful outside-mask image changes: where the backend leaves
the image unchanged, the repair delta is zero and feathering blends identical
values. Floating-point/rounding effects are a separate numerical question.
The concrete issue above is expansion of the **repair mask itself**, not merely
the presence of a Gaussian alpha. OpenCV defines nonzero mask pixels as the
region to reconstruct. [OpenCV inpainting contract](https://docs.opencv.org/4.11.0/d7/d8b/group__photo__inpaint.html)

## 4. Protected destinations and permitted donors are different

The local [PatchMatch-style filler](../../retouch/patchmatch.py) already has
useful controls: source patches must fit completely inside `source_mask` and
outside the repair hole (`_valid_source_centres`, lines 202–212), with seeded
proposals and an optional source-coordinate map. These are existing capabilities,
not recommendations to add another synthesis model.

Two boundaries remain:

**Donor material.** The freckle path subtracts its preserve mask from the repair
hole, but passes the general face mask as donor support
([freckle.py](../../retouch/freckle.py):402–457). A preserved mark/makeup pixel
can therefore still be donor material if included in that face mask. This can
permit copying protected material into a repair elsewhere; actual duplication
has not been measured. The blemish path passes its supplied skin eligibility as
donor support, so its policy exclusions differ. Neither a face label nor a
destination-preserve decision is a complete same-material donor contract.

**Fallback.** If no full source patch fits, `patchmatch_fill()` invokes
`_telea_fallback(image, hole)` (lines 79–87); unresolved pixels have another
fallback at lines 130–135. `_telea_fallback()` at lines 296–301 receives no donor
mask and calls OpenCV inpainting. Thus the source constraint is a property of
the patch-search path, not of every possible returned repair.

The optional source map marks fallback pixels with `(-1, -1)`, but
[heal.py](../../retouch/heal.py):55–65 requests only the image. Backend-request
metadata alone cannot establish which repair was actually used. The existing
[tiny-source test](../../tests/test_patchmatch_heal.py) checks dtype/shape and
the map sentinel; it does not establish donor-policy preservation in fallback.

Proposed study: independently specify destination permissions and donor
permissions, excluding other people, disallowed materials, protected marks,
other repair targets, and uncertain regions as appropriate. Record source-map
availability, valid-donor count, requested/executed backend, and fallback reason.
For a strict donor-policy mode, compare skip/review against a separately approved
constrained alternative when donors are insufficient. Do not silently treat
unconstrained fallback as satisfying the original promise. No fallback behavior
is changed by this report.

The original PatchMatch paper provides approximate patch correspondence and
editing mechanisms, not a guarantee that copied skin detail is the original
photographed detail or belongs to the correct material. Its project-page source
release has a noncommercial-research restriction; that is not a license finding
about Retouch's own implementation. No external code was copied. [Barnes et al.,
author project](https://pixl.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php)

Telea's method propagates nearby information into a masked region. It is a
different reconstruction mechanism, not a semantic donor-policy enforcer.
That difference is why backend fallback must be part of the evidence contract.
[Telea, author description](https://webspace.science.uu.nl/~telea001/Shapes/Inpainting)

## 5. Preserve intent should outlive a fresh detector observation

In [perf_optimizations.py](../../retouch/perf_optimizations.py):

| Stage | Pixels used for mark detection | Consumer |
| --- | --- | --- |
| Entry to per-face work, lines 360–441 | `canvas_original`, before these face operations | Guided smoothing and protected skin-evening masks |
| Freckle healing, lines 618–645 | Current canvas after smoothing/under-eye shadow smoothing | Freckle preservation mask |
| Blemish healing, lines 1058–1074 | Current canvas after additional restoration/color/skin work | Blemish eligibility subtraction |

The entry snapshot is a per-face processing reference, not necessarily the raw
camera original or the engine's pre-reshape reference. Its coordinate/stage
identity must be explicit.

The code even documents opposite rationales: early protection avoids changing
the protected set after edits, while blemish protection re-detects because the
canvas has changed. Both observations can be useful, but they answer different
questions. A mark weakened by an earlier operation may vanish from a later
detector result; disappearance is not evidence that the owner revoked its
preservation. Conversely, freezing a false-positive mark forever can suppress
legitimate correction. No disappearance rate was measured here.

Proposed distinction:

- A source-anchored, versioned **preservation decision** with subject/region ID,
  original support, uncertainty, policy, and provenance.
- A stage-local **appearance observation** that may update visibility, conflict,
  or repair candidates without silently deleting the preservation decision.
- Explicit reconciliation: ambiguity/conflict goes to abstain/review or an
  approved policy, not to automatic removal of the earlier protection.

`MarkRecord.mark_id` currently restarts from the detection result list, and
`_record_footprint()` reconstructs an axis-aligned ellipse from centroid/bounds
([marks.py](../../retouch/marks.py):350–356, 412–419). That is not a persistent
cross-stage ID or exact component support. The earlier
[FA-03 localization report](RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md)
already identifies support inflation. This follow-up's new requirement is to
retain support/decision lineage across consumers, not to rediscover that finding
or automatically ship the experimental classifier.

## 6. Restoration can reverse more than smoothing

The source sequence is:

```text
pre_smooth_canvas snapshot
  → frequency smoothing
  → optional under-eye shadow smoothing
  → optional freckle healing
  → optional exposure lock / flatten
  → micro-texture restoration against pre_smooth_canvas
  → later skin/color operations → blemish healing → under-eye repair
```

Relevant dispatch locations: `perf_optimizations.py`:482, 594–668, 699–707.
[skin.py](../../retouch/skin.py):1047–1060 computes the whole signed difference
between that old reference and the current canvas. The DoG/multiscale experiment
also extracts from this whole-interval residual (dispatch lines 805–810).

Analytically, ignoring clipping, let `I` be the snapshot, `S` the smoothed image,
and `J = S + delta_heal + delta_other` the image just before restoration. With
local restoration amount `a`, the existing raw-residual formula gives:

```text
J + a(I - J) = S + a(I - S) + (1 - a)(delta_heal + delta_other)
```

It therefore attenuates intervening edits as well as recovering the smoothing
residual wherever `a` is positive. This identity does not require a mark
classifier failure. It is a code-derived mechanism, not a newly rendered
freckle-regrowth result. Spectral filtering changes which components return but
does not attach semantic ownership to them.

The older [FA-02 study](RESEARCH_FA02_MICRO_TEXTURE_RESTORE_2026_09_05.md) established
why raw residual magnitude cannot distinguish wanted detail from defects. This
pass adds a narrower intervention to compare: **make the residual's stage
interval correspond to the operation whose detail is being restored**.

Future arms, on fixed reviewed support:

1. Current whole-interval residual, unchanged baseline.
2. Smoothing-only residual captured immediately after smoothing, without moving
   other operations. This isolates residual ownership from operation ordering.
3. Current residual excluding the recorded *executed* repair support, including
   its effective margin, not all detector candidates.
4. Restoration before healing, treated as a separate order-change experiment:
   later detection/repair inputs also change, so this is not equivalent to arm 2.

No arm is promoted. In particular, a smoothing-only residual can still restore
a defect that smoothing was intended to remove, and excluding a repair support
may create a texture discontinuity. Assess retained correction, retained source
detail, boundaries, and makeup separately.

## 7. Under-eye preservation needs a combined-operation contract

### The capped operation is not the only operation

The early `smooth_undereye_shadow()` uses a region-local luminance median and
guided smoothing ([skin.py](../../retouch/skin.py):281–411). It receives ordinary
skin support, not `skin_n_marks_protected`. Later, the v2 under-eye processor
uses its own support and cheek-reference analysis, with lightness lift and
chroma pull; puffiness adds another chroma operation
([undereye.py](../../retouch/undereye.py):523–539). Neither dispatch receives the
general mark-policy preserve mask.

For example, the current `convention_clear_v1` recipe explicitly has shadow
smoothing 0.7, dark-circle alias 0.25, darken removal 0.35, and puffiness 0.50
([recipes.py](../../retouch/recipes.py):3171–3195). Those values use different
operation semantics and must not be summed into a purported total strength.
The 0.35 owner cap remains in place; no recipe was changed. The source only
establishes that several paths can act, not that every path changes every face.

Similarly, `convention_clear_protected_v1` inherits an anisotropic smoothing
engine. The existing guided-only mark-protection gate does not cover that base
smoothing. A “protected” recipe name is not end-to-end material-preservation
evidence, and changing its engine would be a behavior change requiring review.

### Reference quality is a separate uncertainty

`build_undereye_support()` limits the write support and reference ring using
skin/eye masks, but its low-pass sampling mask `valid` excludes the lash margin
only (lines 234–250). Nearby non-skin or cosmetic color can influence the
low-pass estimate even when it is not itself edited. That is a potential
statistics-contamination path, not proof of visible leakage outside support.

`DarkCircleAnalyzer.analyze()` switches from ring samples to the under-eye
support when fewer than 50 ring samples are available (lines 294–299). If both
are too small it returns no reference; otherwise the fallback is an under-eye
median, not a cheek reference. The return value does not identify which baseline
was used. This changes the meaning of the comparison and deserves explicit
`reference_kind`, support counts, reference dispersion, and skip/fallback reasons.
The fixed count also warrants scale qualification; this pass does not replace it.

The known image-axis-down extension remains a roll sensitivity, already recorded
in the under-eye report. A local landmark-frame support is a candidate study,
not a newly discovered shipped fix or an excuse to retune strengths now.

### A dark region is not automatically unwanted shadow

Single-image intrinsic decomposition is ambiguous even before asking whether
an appearance is intentional makeup. The intrinsic-decomposition literature
supports retaining uncertainty; it does not validate a classifier for Retouch's
under-eye photos. [Janner et al., primary abstract](https://arxiv.org/abs/1711.03678)

Research implication: compare source-anchored cosmetic protection, reliable
reference-sector selection, and conservative no-edit/review on ambiguous support.
Do not revive the previously rejected simple chroma-based makeup veto, infer
fatigue/medical cause, or claim a high-frequency-preserving lift necessarily
preserves intentional low-frequency contour makeup.

## 8. Four different mask responsibilities

One mask cannot encode every policy without unwanted side effects:

| Responsibility | Meaning | Why it cannot substitute for the others |
| --- | --- | --- |
| Detection support | Where a candidate may be proposed | Closing/dilation and later selection can change the final region |
| Write support | Where this operation may change pixels, with hard exclusions and soft alpha distinguished | Does not restrict samples used by filters, reference statistics, or donors |
| Reference/donor support | Which pixels may influence a baseline or supply replacement material | A mark can be protected from editing but unsuitable as a donor |
| Evaluation support | Fixed reviewed regions used to judge preservation and harm | Output-selected supports can hide damage or change the denominator |

Operator-specific behavior remains important. Excluding a mark from a whole-face
tone change can create an unchanged-color island; preserving its structure while
allowing coherent tone movement differs from exact pixel preservation during
destructive healing. The proposed contract must name the invariant by operator,
not blanket-apply a single subtractive mask everywhere. Morphology/feathering,
coordinate transforms, and fallback must all respect that declared invariant.

## 9. Proposed discriminating study; nothing executed

Start with manually accepted masks and fixed code/parameters. Reuse the grouped
splits, stage identities, missing-measurement semantics, and native/delivery
review requirements from the [QA study plan](PLAN_RETOUCH_QA_VALIDATION_2026_09_21.md).
These additions test mechanisms that a classifier-accuracy benchmark cannot.

| Case | Controlled comparison | Required observation |
| --- | --- | --- |
| Approved spot adjacent to protected mark | Vary separation versus actual morphology kernel footprint | Final repair mask/support overlap and changed protected pixels |
| Narrow protected gap between repair candidates | Before/after closing and dilation | Whether protection survives every support-transform stage |
| Repair touching skin/eye boundary | Exact eligibility, mask expansion, final compositor | Hard containment plus boundary visual quality |
| Full versus insufficient donor patches | Same target; source support shrunk without changing its policy | Executed backend, fallback, donor-policy compliance; no false “constrained” label |
| Distinctive protected feature in donor area | Excluded from destination only versus excluded from both roles | Source-map provenance and duplicate-feature review |
| Protected source mark weakened by prior tone/smoothing | Source-anchored decision versus fresh stage detection | Decision retention, conflicts, and false protection, not just detector count |
| Freckle heal inside/outside restoration zone | Restoration off/current/smoothing-only/repair-excluded | Correction persistence and detail recovery measured independently |
| Under-eye contour makeup and genuine wanted shadow correction | Early smoothing, restoration, v2 repair, and combinations | Benefit versus contour/chroma harm; cap does not stand in for joint review |
| Poor cheek ring or contamination beside support | Good ring, sparse ring, support fallback, no baseline | Baseline identity, change in result, and abstention |
| Rolled/profile/small/occluded face | Geometry/mask validity before color/texture scoring | Correct coordinate support or explicit unqualified result |

For repair-only hard exclusions, count changed pixels and maximum delta with a
declared dtype/tolerance, plus visually inspect boundaries. For tone-compatible
preservation, compare mark shape/contrast/chroma against the surrounding intended
tone transformation rather than demanding raw-source pixel identity. Report
successful repair as well as harm so “skip everything” cannot win by definition.

Inspect isolated and combined operations, with saved intermediate stages and
policy decisions. Use actual source/target masks and fallback logs, not only
before/final images. Synthetic shapes establish mechanism; natural marks,
makeup, varied appearances, and independent human labels establish relevance.
No minimum corpus count or risk tolerance is newly claimed here.

## 10. Handoff and limits

Recommended next decision, when implementation/experiments are separately
authorized: first qualify final repair containment and donor fallback semantics;
then study source-anchored decisions and residual-stage ownership; then review
the combined under-eye chain. Keep all current recipes and defaults frozen
through diagnostic comparison. Classification replacement remains conditional
on its own reviewed, grouped evidence—not on these source findings alone.

Primary-source access was sufficient for the linked OpenCV contracts, Telea and
PatchMatch author descriptions, and the intrinsic-decomposition abstract.
The attempted full *Intrinsic Images in the Wild* PDF exceeded the web tool's
size limit and was not used as inspected full-text evidence. No external model,
dataset, source code, or license was adopted.

Documentation link/whitespace checks are not algorithm validation. All previous
production/test edits and source photos remain unchanged by this research pass.
