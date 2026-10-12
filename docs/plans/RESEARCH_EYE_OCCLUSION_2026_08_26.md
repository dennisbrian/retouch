# Research — Per-Eye Occlusion Gate Calibration & Validation

**Date:** 2026-08-26 (follows `REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md`, same branch)
**Status:** DSCF-corpus arm EXECUTED 2026-08-31 — results in
`RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31.md`. Headline: shipped thresholds miss 16/19
occluded eyes (84% vs ≤5% target) — `_MIN_EAR=0.12` was calibrated against a mislabeled
anchor (DSCF4454-R is closed, not open); recommended `EAR<0.28 OR contrast<0.55` reaches
100% occluded recall at ~5% visible false-gate. Hard-case sourcing, Fitzpatrick IV-VI,
second labeler, and the threshold commit remain open.
**Runtime:** `.venv` — the pinned supported runtime (mediapipe 0.10.5, protobuf 3.20.3,
`RETOUCH_GPU=0`, `RETOUCH_MEDIAPIPE_BACKEND=legacy`). All measurements below, when run, MUST
use this interpreter. **Trap to document and avoid:** bare `python3` resolves to the drifted
system Python 3.9.6 (mediapipe 0.10.35, protobuf 6.33.6), in which `FaceDetector.available ==
False` and `person_mask` silently degrades to all-ones — see review §2 / §B2 retraction. CLAUDE.md's
own Key Commands block documents `python3 -m pytest tests/ -q` with bare `python3`; that command
silently degrades the whole face path and MUST NOT be used for this study. Use
`.venv/bin/python` explicitly for every measurement.
**Branch:** `feat/color-science-k9-fix-and-frontier`

---

## 0. Premises (resolved before this study runs)

These are stated as resolved; the study assumes them, it does not re-prove them:

1. **P0 — sclera no-op subtraction is fixed.** `parsing.py:753-761`'s
   `left_sclera = clip(left_eye - left_iris)` previously subtracted two disjoint masks (a
   handedness-mismatch symptom), so `left_sclera == left_eye` and the brightener painted the
   *other* eye's iris as sclera. Fixed as a prerequisite; this study measures on the fixed tree.
2. **P1 — BiSeNet 4/5 handedness swap is fixed** in `_masks_from_label_map` in the working tree
   on this branch. BiSeNet labels 4/5 and MediaPipe left/right previously had opposite
   handedness (7 of 8 corpus faces mismatched; on 7/10 a second failure mode collapses class 4
   over both sockets and class 5 to empty — review §B0a). The fix aligns BiSeNet to MediaPipe so
   `regions.left_iris` and `regions.left_eye` describe the same physical eye. The naming
   contradiction at `face_quality.py:27` (`LEFT_EYE_INDICES` = the 263-cluster) vs
   `parsing.py:161` (`LEFT_EYE` = the 33-cluster) is documented in the parsing.py header comment
   and is camera-viewer convention; it is NOT reconciled by renaming and does not need to be.
3. **The gate `retouch/eye_visibility.py` is being rewritten** away from the mask-area/overlap
   heuristics that review §B0/§B0a/§B4 proved blind to closed lids, blind to small faces
   (IED < 30), and structurally dead on the landmark-fallback path (review §B1). The rewritten
   gate uses:
   - **PRIMARY — EAR (eyelid aspect ratio).** Standard 6-point formula over MediaPipe
     eye-contour landmarks. Ear is independent of the iris circle, of BiSeNet, and of which
     parsing branch ran — the only candidate immune to all three root causes above. Indices
     already present: `parsing.py:176-183` (`LEFT_EYE`/`RIGHT_EYE` 16-point contours, from which
     the 6 EAR points are a subset) and `face_quality.py:27-28` (`LEFT_EYE_INDICES`/
     `RIGHT_EYE_INDICES`).
   - **SECONDARY — tone-adaptive iris-vs-sclera luminance contrast ratio.** Expressed against
     the eye's own high percentile (e.g. `p95` of luminance inside the eye contour), NEVER an
     absolute luminance threshold, per CLAUDE.md "Tone-Invariance & Fairness": absolute
     intensity gates systematically fail on darker skin tones (Fitzpatrick V-VI) and are
     forbidden. Reference implementation pattern: `retouch/specular.py::extract_specular`
     (margin-above-baseline relative to each face's own diffuse baseline).
   - **TERTIARY — hair-mask overlap** (retained from the current gate). Structurally dead on
     the landmark-fallback path: there `regions.hair = clip(person_mask - face_oval)` is exactly
     zero everywhere inside `face_oval`, which is where the eye/iris landmarks live — so
     `_weighted_overlap(hair, iris)` and `_weighted_overlap(hair, eye)` are always 0.0 and the
     hair branch can never fire (review §B1, pixel-level proof). This is a documented limitation,
     not a fixable threshold; the hair signal only lives on the BiSeNet-available arm.
4. **The gate is computed once at parse time** and stored on `FaceRegions` as new `__slots__`
   entries (`parsing.py:232-247` has no spare slot today — one must be added per review §4).
   This kills the 35.6 ms / 148.8 ms per-call cost (review §4) and the redundant double-call
   when both enhancer sites fire.
5. **Thresholds are PROVISIONAL.** Every numeric threshold in the rewritten gate is a
   placeholder pending this study's ROC curves. No threshold ships to production until the
   acceptance criteria in §6 are met.

---

## 1. Purpose

Calibrate and validate the per-eye occlusion gate against labeled data before trusting it. The
review (§0 verdict) established that the previous gate "should not ship in its current form"
— it suppressed 78% of real corpus eyes (§B0), 15/26 small visible faces (§B4), and missed its
target case entirely (a collapsed-lid synthetic at EAR 0.057 read as a fully open eye). This
study replaces that heuristic's thresholds with measured operating points and decides, per
signal, whether to keep, tune, or reject it.

---

## 2. Cost asymmetry (governs threshold placement)

| Error | Cost | Why |
|---|---|---|
| **False gate** (visible eye suppressed) | Mild | One eye goes un-enhanced. The subject still looks like themselves; the untreated eye is merely less retouched. Asymmetric in-frame if only one eye is gated (review §B6 catchlight coupling), but the per-eye fix in `eyes.py:119-121` (review §8a.0) closes the inter-eye leak. |
| **Missed gate** (occluded eye enhanced) | **Catastrophic** | The enhancer synthesizes a plausible iris + sclera disc onto the occluder — hair, a wig, a hand, a closed lid. This reads as an obvious retouch artifact on the highest-scrutiny region of a portrait. Review §B1 reproduced this end-to-end: 101/101 hair-covered iris pixels changed, max delta 89, strength 60 → delta 224/765, rendered as a complete painted eye on solid hair. |

This asymmetry justifies **biasing the threshold toward gating** (toward higher false-gate
rate, lower missed-gate rate). A symmetric-accuracy operating point is the wrong objective
function. The study reports the full ROC but recommends operating points selected at a fixed,
low missed-gate rate (see §5).

---

## 3. Labeling protocol

### 3.1 Corpus

- **83-image DSCF corpus** — the convention cosplay shoot corpus already established by
  `RESEARCH_DETECTION_RECALL_2026_08_19.md`. Images live in `test_output/`; the verified list is
  at `test_output/detection_recall_study/corpus.txt` (83 lines, one image per line). Same
  corpus, same pinned `.venv`, so results are directly comparable to that study's recall
  numbers.
- **Sourced hard cases** (NOT in the DSCF corpus — review §9 / §B0a gaps): blink, bangs, wig,
  hand-over-eye, profile, sunglasses, small/distant faces. These must be sourced externally
  before the study is closed; the DSCF corpus alone has none of them (review §B0a: "no
  sunglasses, no profile, no small/distant-face cases, no true full-occlusion case in the
  sample; absence of false negatives is weak evidence, not a clean pass").

### 3.2 Per-eye labels

Three classes, assigned per eye (not per face — a face can have one visible + one occluded):

| Label | Definition |
|---|---|
| `visible` | Eye open enough to enhance: iris and sclera both present in the crop, no occluder over the iris disc. |
| `partial` | Squint / bangs-over-top-half / half-blink: iris still substantially visible but lid or hair covers part of the eye contour. This is the hardest class — review §3a's "partial coverage" point applies. |
| `occluded` | Eye fully closed (lid down over iris), OR iris disc fully covered by hair/wig/hand/sunglasses. This is the class whose miss is catastrophic. |

Labels are recorded as a JSON manifest keyed by `(image, side)` with a `label` field and a
free-text `evidence` field (e.g. "EAR 0.108, lids meet", "wig lace over iris, 100% hair cover",
"subject-facing profile, no eye contour visible"). The `evidence` field is mandatory — review
§B0a noted prior studies had no human label recorded and could only verify by texture + geometry
+ detector silence.

### 3.3 Sourcing the hard cases

The hard-case set is scoped to close the review's named gaps:

| Gap from review | Sourced as | n (target) |
|---|---|---|
| Blink | Frame-strided capture from video (eyes cycle through blink) | 10 |
| Bangs | Sourced portraits, heavy-bangs subjects | 10 |
| Wig | Sourced cosplayer portraits with wig lace line over eye | 8 |
| Hand-over-eye | Sourced portrait poses | 6 |
| Profile (3/4 and full) | Sourced portraits | 8 |
| Sunglasses | Sourced portraits (opaque + reflective) | 8 |
| Small/distant faces | Group-shot crops at IED 20-35 (review §B4 crossover) | 8 |

Target: 83 (DSCF) + ~58 (hard cases) = ~141 images, ~282 eyes. Small/distant faces are included
specifically because review §B4 pinned the false-gate crossover at IED ≈ 30 (15/26 visible eyes
gated); the rewritten EAR-based gate should be size-independent by construction (EAR is a ratio),
but this must be measured, not assumed.

---

## 4. Stratification — two arms

The fallback path is structurally different from the BiSeNet path (review §B1). They are
measured as **separate arms**, never pooled:

| Arm | When it runs | Hair signal | Eye/iris mask provenance |
|---|---|---|---|
| **A — BiSeNet available** | `.venv`, `models/resnet18.onnx` present, `model_status("resnet18_bisenet")` available | Live (full-image hair mask, cropped to ROI) | BiSeNet eye-class masks, handedness-fixed |
| **B — BiSeNet unavailable** | Model absent, OR MediaPipe/protobuf env drift (the recurring condition documented in review §2) | **Dead** (hair = `clip(person_mask - face_oval)` is exactly zero inside `face_oval` by construction) | Landmark polygons only (`parsing.py:670-671`) |

On arm B the hair-overlap tertiary signal is **structurally inert** — this is a documented
limitation, not a tunable threshold. EAR (primary) and the iris-vs-sclera contrast (secondary)
are the only live signals on arm B. The study MUST report per-arm ROC; a pooled curve would
hide the arm-B degradation that review §B1 proved end-to-end (101/101 hair-covered iris pixels
enhanced on the fallback path with the old gate).

Forcing arm B: set `parser._sess = None` before `parse()` (the pattern the golden fixture
already uses, review §2). Forcing arm A: confirm `model_status("resnet18_bisenet")` reports
`available: True` (review §6.5 notes it tracks something other than on-disk file presence).

---

## 5. Metrics

### 5.1 ROC per signal, per arm

For each of the three signals — EAR, iris-vs-sclera contrast, hair overlap — and for each arm
(A, B), report a full ROC curve over the labeled `occluded` vs `visible` classes (with
`partial` held out from the ROC fit but plotted as a third population on the curve). The
contrast ratio is the only signal whose form is constrained by the fairness rule: it MUST be
expressed against the eye's own high percentile (e.g. `(lum_iris - lum_eye_p95) / lum_eye_p95`
or similar), never as an absolute luminance.

### 5.2 Operating points and confusion matrices

Operating points are chosen post-hoc from the ROC curves at a **fixed missed-gate rate** (the
cost-asymmetry rule from §2). Concrete target: missed-gate rate ≤ 5% on the `occluded` class.
At that operating point, report the full 3×3 confusion matrix (`visible` / `partial` /
`occluded` → predicted `visible` / `partial` / `occluded`) per arm. The `partial` row is the
decision boundary's hardest case and is reported for transparency, not optimized against.

### 5.3 Fairness split

Per CLAUDE.md "Tone-Invariance & Fairness", split the labeled set by Fitzpatrick type:

| Stratum | Status in local corpus |
|---|---|
| Fitzpatrick I-III | Present (the DSCF corpus is uniformly light-skinned East Asian cosplayers) |
| Fitzpatrick IV-VI | **ABSENT locally** — review §B0a: "no Fitzpatrick V-VI subject exists in the local corpus." These MUST be sourced externally before the fairness claim is closed. The sourced hard-case set (§3.3) is the vehicle for this, but it must be augmented with explicit Fitzpatrick IV-VI subjects (not just hard cases that happen to be dark-skinned). |

Report per-stratum missed-gate rate at the chosen operating point. A gate that meets the ≤5%
target on I-III but degrades on IV-VI is NOT a passing gate; it is a gate with an open fairness
finding. The fairness claim stays open until IV-VI data exists and passes.

### 5.4 Reference data points (prior evidence, cited from review §8a.3)

These are NOT new measurements — they are anchors from the review to sanity-check the study's
EAR scale:

| Image / condition | EAR | Class (expected) |
|---|---|---|
| DSCF4576 | 0.108 | `occluded` (genuinely closed) |
| DSCF4612 | 0.122 | `occluded` (genuinely closed) |
| DSCF6961 | 0.513 | `visible` (wide open) |
| Collapsed-lid synthetic | 0.057 | `occluded` (the case the old gate missed — review §B0) |
| Squint ~30% aperture | ~0.30 | `partial` boundary (review §B4: defensible to gate at ≤30%) |

If the study's measured EAR distribution does not place these anchors in the expected classes,
something is wrong with the EAR implementation, not the anchors.

---

## 6. Acceptance criteria (keep / tune / reject per signal and per threshold)

Format mirrors `RESEARCH_DETECTION_RECALL_2026_08_19.md`'s follow-up sections. Each signal and
each threshold gets a verdict, not a blanket pass.

### 6.1 EAR (primary)

| Criterion | Keep | Tune | Reject |
|---|---|---|---|
| `occluded` recall @ chosen op | ≥ 0.95 | 0.80-0.95 | < 0.80 |
| `visible` precision @ chosen op | ≥ 0.90 | 0.75-0.90 | < 0.75 |
| Arm A vs arm B gap (AUC) | < 0.05 | 0.05-0.15 | > 0.15 |
| Fitzpatrick I-III vs IV-VI gap (missed-gate) | < 0.05 | 0.05-0.10 | > 0.10 |

If EAR is `Reject`, the gate has no primary signal and does not ship — the whole rewrite is
revisited, not patched.

### 6.2 Iris-vs-sclera contrast (secondary)

| Criterion | Keep | Tune | Reject |
|---|---|---|---|
| ΔAUC vs EAR-alone | > +0.03 | 0 to +0.03 | ≤ 0 or negative |
| Fairness: I-III vs IV-VI AUC gap | < 0.10 | 0.10-0.20 | > 0.20 |
| Form | ratio vs eye's own high percentile | — | absolute luminance (forbidden) |

If the contrast signal does not add ≥3 AUC points over EAR-alone, it is `Tune`-able to dead
weight or `Reject`. The fairness gap is the stricter gate: a contrast signal that helps
overall but degrades on IV-VI is `Reject` regardless of aggregate AUC.

### 6.3 Hair overlap (tertiary)

| Criterion | Keep | Tune | Reject |
|---|---|---|---|
| Arm A ΔAUC vs EAR+contrast | > +0.02 | 0 to +0.02 | ≤ 0 |
| Arm B live? | — | — | structurally dead (documented) |

Hair overlap is kept only if it adds measurable signal on arm A beyond EAR+contrast. On arm B
it is `Reject` by construction (§4) — it stays in the code as a documented no-op on that path,
not a removed branch (removing it would silently change arm-A behavior).

### 6.4 Threshold placement

Each threshold in the rewritten `eye_visibility.py` gets a per-threshold verdict after the
ROC curves are fit. The cost-asymmetry rule (§2) governs: a threshold that achieves symmetric
accuracy but missed-gate > 5% is `Reject`, even if its aggregate accuracy is higher. The
threshold table in `eye_visibility.py:27-32` (the old `_MIN_*` / `_HAIR_*` constants) is
replaced by EAR / contrast / hair constants whose values come from this study's operating
points — not from the old gate's numbers, which review §B0/§B4 proved wrong.

---

## 7. Environment & reproducibility

### 7.1 Interpreter

Every measurement uses `.venv/bin/python`. Bare `python3` is drifted system Python 3.9.6 and
MUST NOT be used (review §2, §B2 retraction root cause). This is a documented, recurring trap;
CLAUDE.md's own Key Commands block (`python3 -m pytest tests/ -q`) silently invokes the wrong
interpreter. Document this in the study's reproducibility section verbatim.

### 7.2 Env vars

```
RETOUCH_GPU=0
RETOUCH_MEDIAPIPE_BACKEND=legacy
```

mediapipe 0.10.5 (Tasks API unsupported on this macOS runtime — review §2), protobuf 3.20.3
(pre-4 API, the pin that mediapipe 0.10.5 requires). System Python's protobuf 6.33.6 /
mediapipe 0.10.35 combination is a different runtime and produces different masks.

### 7.3 Scripts

New scripts live in `scripts/qa/` (untracked until commit, per the detection study's
convention). Candidate set:

| Script | Purpose |
|---|---|
| `eye_occlusion_label_template.py` | Emits the empty JSON manifest for the 83-image DSCF corpus + hard-case slots |
| `eye_occlusion_ear_sweep.py` | Per-eye EAR over the labeled set, both arms |
| `eye_occlusion_contrast_sweep.py` | Tone-adaptive iris-vs-sclera contrast, per-arm |
| `eye_occlusion_hair_sweep.py` | Hair overlap (arm A live, arm B dead-documented) |
| `eye_occlusion_roc.py` | Per-signal per-arm ROC + AUC, Fitzpatrick split |
| `eye_occlusion_confusion.py` | 3×3 confusion matrices at chosen operating points |

Artifacts: `test_output/eye_occlusion_study/` (`labels.json`, `ear_sweep.json`,
`contrast_sweep.json`, `hair_sweep.json`, `roc_by_arm.json`, `confusion_by_arm.json`).

---

## 8. Limits of this study

- The 83-image DSCF corpus is one photographer, convention-cosplay subject matter, uniformly
  light-skinned (review §B0a). The fairness split is open, not closed, until Fitzpatrick IV-VI
  data is sourced and measured.
- No true full-occlusion case exists in the DSCF sample (review §B0a); the hard-case set is the
  only source of `occluded` labels for the catastrophic-cost class. A study that sources too few
  `occluded` cases will have wide confidence intervals on missed-gate rate.
- `partial` is the hardest label and the one review §3a flagged as the under-eye-ops failure
  mode ("a binary 'mostly occluded' gate would miss exactly this"). The per-eye gate inherits
  the same boundary; the `partial` row of the confusion matrix is reported for transparency but
  the operating point is chosen on `occluded` recall, not `partial` accuracy.
- Arm B (BiSeNet unavailable) is the path that goes live under MediaPipe/protobuf env drift —
  a documented recurring condition. The study forces it via `parser._sess = None`, which is the
  golden-fixture pattern (review §2) but not the production env-drift pattern; the two may
  differ in `person_mask` behavior (review §2: env drift degrades `person_mask` to all-ones,
  which the forced-`None` path does not reproduce). This is a residual gap.
- Human labels are the ground truth here, unlike the detection study's cross-detector
  consensus. Labeler disagreement is recorded (Cohen's kappa if ≥2 labelers); single-labeler
  runs are flagged as provisional.
- The contrast signal's exact form (which percentile, which luminance space) is itself a
  design choice this study measures, not a fixed quantity. A negative result on one form does
  not rule out the signal class — but the fairness-rule constraint (ratio vs absolute) is
  non-negotiable and is enforced as a `Reject` criterion in §6.2.

---

## 9. Deliverables checklist

Executed items below reference `RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31.md` (the
2026-08-31 DSCF-corpus run; scripts `scripts/qa/eye_occlusion_{sweep,roc}.py`).

- [x] Label manifest emitted (folded into `eye_occlusion_sweep.py` + `labels.json`; no
      separate template script)
- [x] Labels collected for the 83-image DSCF corpus (166 eyes, `evidence` mandatory,
      single-labeler → PROVISIONAL; a 4th class `no_face` was added for 10 images whose
      largest detection is off-subject) — hard-case set still ☐ unsourced
- [ ] Fitzpatrick IV-VI subjects sourced as a separate stratum (fairness claim open until done)
- [x] EAR sweep run on both arms (A: BiSeNet live, B: `parser._sess = None`) — AUC 0.9922
- [x] Contrast sweep run, form verified as ratio-vs-eye's-own-percentile (not absolute)
- [x] Hair sweep run, arm-B dead-documented (structural zero confirmed empirically on all
      166 eyes, not tuned) — arm-A value inconclusive: zero hair-occlusion positives in corpus
- [x] Per-signal per-arm ROC + AUC committed to `test_output/eye_occlusion_study/`
- [ ] Fitzpatrick I-III vs IV-VI missed-gate gap reported at chosen operating point
- [x] Confusion matrices per arm at the §2 cost-asymmetry operating point
      (`confusion_by_arm.json`, shipped + proposed thresholds)
- [x] Per-signal verdict table filled (results doc §2: EAR keep, contrast tune-secondary,
      hair inconclusive/dead-on-B)
- [x] Per-threshold verdict table filled (results doc §4: shipped `_MIN_EAR=0.12` REJECT —
      misses 16/19 occluded; recommended `EAR<0.28 OR contrast<0.55`)
- [ ] Rewritten `eye_visibility.py` thresholds updated from this study's numbers (separate
      commit — deliberately NOT made; blocked on hard cases + second labeler + fairness)
- [ ] Arm-B coverage extended in `tests/test_eye_visibility.py` (review §8a.6: it never
      exercises the BiSeNet-unavailable arm today — it paints hair explicitly)
- [x] Sanity check: §5.4 anchors — DSCF4576 (0.101) and DSCF6961 (0.504) land as expected;
      DSCF4612 measures 0.234 not 0.122 (anchor value stale; eye still closed); the
      threshold comment's "DSCF4454-R plainly open at 0.136" is WRONG — that eye is closed
      (root cause of the bad `_MIN_EAR`)

## 10. Out of scope

- Under-eye op gating (review §3a) — separate problem, separate doc; the per-eye gate's
  verdict does not block it and is not blocked by it.
- The `_enhance_catchlights` per-eye fix (review §8a.0) — prerequisite, already scheduled.
- Sharpen-mask leak (review §B5) — the gate is hoisted to `_process_face_core` per review
  §B5's recommended fix; that is a wiring change, not a research question.
- Poster-FP veto (review §8a.5 is the eye-occlusion doc; the poster work is
  `RESEARCH_POSTERFP_VETO_2026_08_19.md`, already measured).
