# FA-01 Ontology, Evidence Fields, and Decision Record (backfill)

**Date:** 2026-09-05

**Status:** Documentation backfill. `docs/plans/PLAN_FACE_RETOUCH_ALGORITHM_RESEARCH_EXECUTION_2026_09_05.md`
names a documentation-first contract (labels, evidence fields, corpus
protocol, evaluation contract, decision record, provenance register) as a
prerequisite for FA-02. This tranche wrote diagnostics and shipped two
production changes (`154d854` mark-policy wiring, `c665da1` guided-smoothing
mark protection) before that contract existed on paper. This document
closes that gap by writing down what the code and the FA-01 audits
(`docs/plans/RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md`)
already established empirically, rather than inventing a new scheme. No
code, model, recipe, or corpus change is made by this document. Corpus
coverage (§3) is left honestly incomplete where no measurement exists —
this document does not manufacture numbers to look complete.

---

## 1. Face and material labels

Two label spaces already exist in the codebase and this document keeps
them separate rather than merging them into one taxonomy, because they
answer different questions (`region ownership` vs. `mark identity`).

### 1a. Region/material labels (parser-level, `retouch/parsing.py`)

Source: BiSeNet trained on CelebAMask-HQ's 19-class label map, re-keyed by
`_masks_from_label_map`. This is the ownership label a caller checks
before an op reads or writes a pixel.

| Label | CelebAMask-HQ id(s) | Extracted as own mask? | Notes |
|---|---|---|---|
| skin | 1 | Yes (`regions.skin`) | Excludes nose (label 10) — see `skin-mask-nose-hole` memory; nose lives only inside the `face_oval` union. |
| right/left eyebrow | 2, 3 | Yes | Tier 1 protection (subtracted from skin). |
| right/left eye | 4, 5 | Yes | Tier 1; subject-anatomical convention (swapped from BiSeNet's camera-viewer output, `8811320`/eye-handedness fix). |
| nose | 10 | No — union only | Inside `face_oval`, never its own mask; a skin-masked op strands the nose. |
| mouth interior | 11 | Yes | Tier 1. |
| lips | 12, 13 | Yes (merged) | Tier 1. |
| neck | 14 | Yes | Tier 2 (separate class, excluded from skin by argmax exclusivity). |
| cloth | 16 | Yes | Tier 2. |
| hair | 17 | Yes | Tier 2. |
| unlabeled / background / ears / earrings / necklace / hat / eyeglasses (subset) | 0, 6, 7, 8, 9, 15, 18 | No dedicated mask | Tier 3 — protected only by argmax exclusivity, no test, no fallback if BiSeNet mislabels one of these as skin. Zero corpus cases (glasses/wig/hat) to measure against per CLAUDE.md's known-limitations note. |
| facial hair (beard/mustache) | *(no CelebAMask-HQ class)* | No | Tier 4 — no model class exists at all. `beard_shadow_neutralize` treats beard as skin (color-cast fix), the opposite of protection. |

Lashes are **not** a separate label — covered by the eye-region mask
(Tier 1) plus `undereye.py` v2's dedicated landmark lash-margin exclusion
at the one boundary that needed it. Documented here as "covered," not a
gap, per the existing audit.

### 1b. Mark/defect labels (detector-level, `retouch/marks.py`)

Source: `MARK_CLASSES` in `retouch/marks.py`, already shipped and in
production use via `mark_policy`.

```
mole, freckle, acne_blemish, scar, drawn_makeup_mark,
stray_hair, sensor_dust, vellus_sheen, unknown
```

Each detection is a `MarkRecord` (`mark_id, mark_class, confidence,
centroid, area_norm, bbox, features, on_body`) — a genuine per-instance
evidence record, not a collapsed scalar. Two named policies exist today
(`protect_identity`, `preserve_all`); `legacy`/`None` is the compatibility
floor (empty preserve/heal/attenuate masks, byte-identical output).

**Known confound, not yet fixed:** `detect_marks` misclassifies eyeliner
as `mole` on the cosplay/heavy-makeup corpus (documented FA-03 in
`fa01-boundary-confidence-and-mark-policy-trace` memory and in the
Meitu bake-off protocol). This document records it as an open detector
gap; FA-01/FA-02 work should not be blocked on fixing it, but any
"marks preserved" evaluation on that corpus must discount eyeliner false
positives rather than count them as protected marks.

### 1c. Reconciling the two label spaces

A mark (1b) always sits inside exactly one region (1a) — typically
`skin`, occasionally straddling a `skin`/`eye` or `skin`/`lips` boundary
at low mark confidence. `tests/test_frequency_mark_protection.py`'s
`TestMarkProtectNearParserBoundary` exercises exactly this case for the
smoothing path: a mark's protection footprint must never extend past
what the region-level `skin_mask` already permits, regardless of the
mark-level policy. This is the ownership rule this document is asked to
name: **region label is the outer bound; mark policy can only shrink
inside it, never grow past it.**

---

## 2. Support ownership

Per-stage read/write rights, generalizing the four-tier protection-
mechanism table already built in the audit doc
(`RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md` §1.2) plus
this tranche's smoothing addition:

| Stage class | Reads region labels? | Reads mark policy? | Where feathering ends |
|---|---|---|---|
| Healers (`blemish.remove`) | Yes (`regions.skin`) | Yes, since `f3b1008` | Heal-skip: excluded pixels are never touched, no surrounding re-render. |
| Evening ops (flatten, micro_dodge_burn, redness_even, hb_even, hemoglobin_smooth, vein_attenuate, equalize, unify_hue_line, unify_tone) | Yes | Yes, since `154d854` (9 ops) | Feather-then-clip to `skin_n_marks_protected`; feather order matters (feathering before clipping to skin re-grows the mask across region boundaries — the historical bug this fixed, 1318 vs 592 leaked px). |
| Shading/tone ops (7, deliberately excluded) | Yes | **No, by design** | Excluding marks here creates a new island artifact (whole-face re-render; measured on `skin.equalize`, a hard-edged step). This is a documented non-gap, not an oversight. |
| Base smoothing (guided engine, `frequency.combine`) | Yes (`skin_mask` param) | Yes, since `c665da1`, **guided engine only** | Per-blob distance-transform feather, capped at `0.6 × blob radius` (interpolated below the experimentally measured degradation threshold, not itself swept); shrinks final blend alpha only, guided filter's own input statistics untouched. |
| Base smoothing (bilateral/anisotropic engines) | Yes | **No — explicitly gated off, untested** | `mark_protect` argument silently ignored + `logger.debug` fired; `_smooth_anisotropic`'s internal `cv2.error` fallback to guided does *not* retroactively gain protection — the gate keys on the requested engine, not on what ran. |
| Under-eye v2 | Yes, plus a dedicated lash-margin exclusion | Not yet wired to `mark_policy` | Cheek-ring relative smoothstep, ROI-confined, `restore_outside_support`. |
| Accessories/facial hair (Tier 3/4) | No dedicated mask exists | N/A | Open gap, unmeasured (§1a). |

**Where feathering ends, as a general rule:** every mechanism above
feathers *inward* from a hard region/mark boundary (shrinking the
editable footprint), never outward past it. The one exception under
active use is the base-smoothing guided path, which can appear to affect
pixels near a mark that a naive reading of `skin_mask` alone wouldn't
predict — because the *unprotected* baseline behavior recruits neighbor
statistics via the guided filter's unmasked `cv2.blur`. This is why
`filter_input` (excluding marks from the filter's own reference
statistics) was evaluated and rejected in favor of blend-alpha-only
exclusion: touching the filter's input is the one place "ownership"
could silently expand rather than shrink, and it was measured to cost
more (halo) than it returns (marginal contrast).

---

## 3. Corpus protocol

**What already exists:** `test_output/detection_recall_study/corpus.txt`
— 83 real DSCF-prefixed images, already used as the reference corpus
across `RESEARCH_DETECTION_RECALL_2026_08_19`, `RESEARCH_POSTERFP_VETO`,
`RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31`, `RESEARCH_YAW_GATE_CALIBRATION`,
`RESEARCH_POST_EPSILON_FACEOP_REAUDIT`, and `RESEARCH_DARK_CIRCLE_OP`
(2026-09-02). This is a real, load-bearing corpus, not a new one — this
document formalizes it as FA-01/FA-02's shared reference corpus rather
than inventing a separate one.

**What is honestly missing:** no document anywhere assigns per-image
attributes (frontal / three-quarter / profile / small-face / lighting /
makeup / glasses / wig / facial hair / marks / occlusion / multi-face) to
the 83 images. Prior studies filtered ad hoc (e.g. "15/15 frontal corpus
faces" in the neck-gate fix, "4 pale anchors" in the chroma-gate fix,
"146 corpus eyes" in the dark-circle study) without a standing manifest.
This is a real gap against the plan's corpus-protocol requirement, and
this document does not manufacture the missing labels — attribute-tagging
the 83 images is named as the first concrete FA-01 backlog item (§5)
rather than skipped silently.

**Known corpus limitations** (already documented in CLAUDE.md and
repeated here because the evaluation contract in §4 depends on them):
zero glasses, wig, or hat cases; zero genuine natural moles (all
mark-protection evidence to date is semi-synthetic — a real face crop
plus a composited synthetic mark, per
`scripts/qa/smoothing_mark_protection_experiment.py`); zero Fitzpatrick
V-VI stratum representation; heavy skew toward cosplay/makeup portraits,
which confounds mark detection (§1b) and is the reason the Meitu
bake-off protocol memory warns against retuning natural/`convention_clear_v1`
from that specific corpus.

**Splits:** no formal development/calibration/held-out split exists for
this corpus today. Every study to date has both tuned and measured
against the same 83 images (e.g. the eye-gate threshold recalibration
grid-searched and validated on the same corpus). This is a real
methodological gap for any future *threshold-setting* work (as opposed
to one-off diagnostic renders, which don't overfit the same way) — flagged
here, not resolved, since resolving it means either growing the corpus or
formally partitioning the existing 83 images, and both are corpus changes
this documentation-only tranche does not authorize.

---

## 4. Evaluation contract

Reusing the plan's named dimensions, mapped to what is and isn't
measurable today:

| Dimension | Measurable today? | Instrument |
|---|---|---|
| Boundary leakage | Yes | `_bisenet_boundary_confidence_by_region` (observation-only, `6b1e72d`); the neck depth-gate and eye-handedness fixes were both found via boundary-leakage-style direct pixel inspection, not this field — the field is new and not yet used to catch a live bug. |
| Useful edit coverage | Partial | No single metric; approximated per-study (e.g. "skin lift 9.3 L median" in the dark-circle study, "1318 vs 592 leaked px" in the mark-feather fix). |
| Protected-mark preservation | Yes, for guided smoothing | `_mark_contrast` (ring-minus-mark, ported from the experiment script into `tests/test_frequency_mark_protection.py`) and the `_outer_halo` instrument for boundary artifacts. Not yet extended to bilateral/anisotropic or to real (non-composited) marks. |
| Abstention | Yes, partially | `SafeAutoDecision` (`retouch/safe_auto.py`) is a genuine structured evidence record (`stage, action, confidence, reason, evidence, strength_scale`) — this already satisfies the plan's "evidence fields must not be collapsed into one scalar" requirement for the safe-auto path. `eye_visibility_gate`, yaw-gate, and eye-artifact-scale abstention are logged (`8a2869e`) but not structured into `SafeAutoDecision` records — they're `logger.debug` lines, inspectable but not machine-collectible the way `safe_auto_decisions` is. |
| Native-resolution crop review | No | Every FA-01 measurement to date has been either full-frame delta or a hand-picked pixel-coordinate probe; no standing "crop and view at native res" step exists as a named procedure. CLAUDE.md's verification rule ("view the actual output") is followed ad hoc per session, not as a checklist item. |
| Texture observability | No | Not yet needed — no texture-restoration work has started (that's FA-02's job). |
| Runtime records | Partial | Bench numbers exist in CLAUDE.md (702ms/face, 4.7ms no-face) but are not re-measured per FA-01 change. The `c665da1` mark-protection addition (`distanceTransform` + `connectedComponents` on a small mark blob, called once per `combine()` invocation when a mark is policy-protected) was measured directly for this backfill on a 512×512 synthetic canvas, 20 iterations: 30.98ms/call without `mark_protect` vs. 30.64ms/call with a 10px-radius mark supplied — within noise, no measurable cost at this scale. Not re-measured on a face with many simultaneous marks or at native (multi-megapixel) resolution. |

**Do not set numerical thresholds before labels exist** (plan's own
rule): the `MARK_PROTECT_FEATHER_FACTOR = 0.6` constant is the one
number this backfill flags as already having been set ahead of a full
corpus sweep — it was interpolated from a 3-point feather-width sweep on
2 scenes, documented as such in `frequency.py`'s comment and in the audit
doc §1.1b, not presented as corpus-calibrated the way the eye-gate
thresholds were.

---

## 5. Decision record

Per-topic disposition, using the plan's five allowed outcomes (retain
current / use as secondary evidence / adopt behind migration / reject /
defer for missing evidence):

| Topic | Decision | Basis |
|---|---|---|
| BiSeNet boundary-confidence exposure | **Adopt, observation-only** | `6b1e72d`; guarded by `test_parse_confidence_not_read_by_any_op` so it cannot silently become a threshold. |
| mark_policy in `blemish.remove` | **Adopt** | `f3b1008`; opt-in, `legacy`/`None` byte-identical. |
| mark_policy in evening ops | **Adopt, scoped to 9 of ~16 candidate ops** | `154d854`; shading/tone ops explicitly rejected for mark exclusion (would create island artifacts) — this is a **reject** decision for those 7 ops, not a deferral. |
| Abstention logging (yaw gate, eye-artifact scale) | **Adopt, log-only** | `8a2869e`; golden hashes unchanged. |
| mark_protect in guided smoothing (blend-alpha mechanism) | **Adopt** | `c665da1`, following the `c492291` experiment. |
| mark_protect via `filter_input` (guided-filter statistics exclusion) | **Reject** | `c492291`: marginal contrast gain (+0.1 to +3.9 pts) at a consistently worse halo cost (0.26-4.17 vs blend-alpha's 0.04-0.5) — explicit reject, not a deferral, because both arms were measured. |
| mark_protect in bilateral/anisotropic smoothing | **Defer for missing evidence** | Never evaluated on either engine; the plan's own rule ("do not generalize... until validated") applies directly. |
| Accessory (Tier 3) dedicated masks | **Defer for missing evidence** | Zero corpus cases; cannot be measured, not a capability gap that's been tested and found wanting. |
| Facial-hair (Tier 4) protection | **Defer for missing evidence** | No model class exists; needs either a new parser class or a dedicated detector, out of scope for diagnostic work. |
| Corpus attribute tagging / dev-calibration-holdout split | **Defer, named as FA-01 backlog** | Not started; see §3. |
| SegFace-vs-BiSeNet boundary comparison | **Defer for missing evidence** | Depends on the corpus/metrics work above per the plan's own ordering; untouched. |

---

## 6. Provenance register

| Candidate | Source | License | Runtime assumption | Availability |
|---|---|---|---|---|
| BiSeNet (current parser) | Already vendored, ONNX | *(not re-audited here — pre-existing dependency)* | CPU/GPU ONNX runtime, already in the pinned environment | In production use; no provenance question raised by this tranche. |
| RetinaFace | Evaluated and rejected for the current environment | N/A | `pip` resolution forces protobuf 6/TF 2.20/numpy 2/cv2 5 — structurally conflicts with the pinned runtime | Not available; MediaPipe-only remains the supported detection path (CLAUDE.md Known Limitations). The `cc61e67` F1/F2 fixes are documented-not-live for this reason. |
| SegFace (named in the parent research report as a future comparison) | Not yet investigated | Unknown | Unknown | Not started — this is exactly the item the execution plan orders *after* the ontology/corpus/metrics work, which is what this document is completing. No model has been downloaded; no code references it. |
| `detect_marks` (existing) | Already vendored, classical CV (no external model) | N/A | Runs on CPU, already in the hot path when `mark_policy is not None` | In production use; the eyeliner-misclassification gap (§1b) is a known limitation of this existing detector, not a new candidate. |

A public repository existing for SegFace (or any other candidate) is
explicitly not implementation readiness, per the plan's own instruction —
this register records that no such readiness assessment has been done
yet, rather than skipping the row.

---

## 7. What this document does and does not authorize

This document changes documentation only. It does not authorize:
- downloading any model weights (SegFace or otherwise),
- corpus changes (new images, splits, or attribute-tag files),
- production behavior changes,
- numerical threshold changes to `MARK_PROTECT_FEATHER_FACTOR` or any
  other constant named above.

It **does** close the plan's documentation-phase exit criteria for FA-01
(labels, ownership, evidence fields, corpus protocol, evaluation
contract, decision record, provenance register are now written down,
with gaps stated honestly rather than hidden) so that FA-02 can begin
under the plan's own ordering rule ("evaluated only after the support and
protection contract is clear").
