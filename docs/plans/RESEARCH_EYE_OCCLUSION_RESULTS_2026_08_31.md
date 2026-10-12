# Research Results — Per-Eye Occlusion Gate Calibration (DSCF-corpus arm)

**Date:** 2026-08-31
**Protocol:** `RESEARCH_EYE_OCCLUSION_2026_08_26.md` (this doc executes its locally-runnable arm)
**Branch:** `feat/color-science-k9-fix-and-frontier`
**Runtime:** `.venv/bin/python` (mediapipe 0.10.5, protobuf 3.20.3, `RETOUCH_GPU=0`,
`RETOUCH_MEDIAPIPE_BACKEND=legacy`) — verified at run start. Bare `python3` never used.
**Scripts:** `scripts/qa/eye_occlusion_sweep.py`, `scripts/qa/eye_occlusion_roc.py`
**Artifacts:** `test_output/eye_occlusion_study/` — `sweep_full.json`, `ear_sweep.json`,
`contrast_sweep.json`, `hair_sweep.json`, `labels.json`, `roc_by_arm.json`,
`confusion_by_arm.json`, `crops/` (166 per-eye crops), `label_sheet_*.png`, `ambig_sheet.png`.

---

## 0. Headline verdict

**The shipped gate thresholds fail the protocol's acceptance criteria and must be
recalibrated before the gate can be trusted.** At the current
`_MIN_EAR = 0.12` / `_MIN_CONTRAST = 0.45`, the gate misses **16 of 19 genuinely
closed/occluded eyes (84% missed-gate rate)** on both arms — against the §2 target of
≤ 5%. The catastrophic-cost class (enhancer paints an iris onto a closed lid) is
essentially unprotected. Meanwhile false-gates on visible eyes are 0/122 (arm A): the
gate was implicitly tuned for perfect precision at the cost of near-total recall — the
exact inversion of the protocol's §2 cost asymmetry.

**Root cause of the miscalibration, found and verified:** the threshold rationale in
`eye_visibility.py` ("the lowest open-eye EAR measured on the DSCF corpus is 0.136
(DSCF4454-R, plainly open)") is built on a **mislabeled eye**. At 300 px crop
resolution DSCF4454-R is unambiguously **closed** (lid down, lashes fanned —
see `crops/DSCF4454_right.png` / `ambig_sheet.png`; both 4454 eyes are closed).
The true open-eye floor on this corpus is **0.194** (DSCF7204-R, open, heavy lashes).
`_MIN_EAR = 0.12` sits *below almost the entire closed-eye distribution*
(0.101–0.283), so EAR catches only 1/19 closed eyes (DSCF4576-R, 0.101).

EAR itself is an excellent signal (AUC 0.992) — the *threshold* is wrong, not the
signal. This is a keep-signal / reject-threshold outcome.

---

## 1. Corpus, labels, method

- 83-image DSCF corpus (`test_output/detection_recall_study/corpus.txt`), 83/83 largest
  detections landmarked, 166 eyes measured on both arms in 42.6 s. Images pre-shrunk to
  max-dim 2048 (detection-study proxy scale; EAR is a ratio, contrast is relative — both
  scale-stable; recorded as a study condition in `sweep_full.json` meta).
- **Labels:** per-eye, from 170 px contact sheets with a second 300 px pass over 20
  ambiguous crops. Counts: **122 visible / 19 occluded / 5 partial / 20 no_face**.
  Single labeler (Claude) → **PROVISIONAL** per protocol §8; `evidence` field recorded
  for every non-visible eye.
- **Protocol deviation — 4th class `no_face`:** on 10/83 images (4554, 4555, 4589,
  4590, 4592, 4596, 4597, 4599, 4600, 4618) the largest detection is **off-subject**
  (back of head, defocused background, blurred poster) and MediaPipe still fits a full
  landmark set to it. These are operationally must-gate (enhancement paints an eye onto
  non-face pixels) but are not lid/hair occlusion, so they are reported as their own
  stratum, not pooled into `occluded`.
- Arm A = BiSeNet live (CPU EP); arm B = `parser._sess = None` (golden-fixture
  pattern). `person_mask` not supplied.

## 2. Per-signal results (occluded vs visible; `partial` held out)

| Signal | Arm | AUC | Coverage (occl / vis) | Occluded range | Visible range | Verdict (§6) |
|---|---|---|---|---|---|---|
| **EAR** | A = B | **0.9922** | 19/19 · 122/122 | 0.101–0.283 (med 0.207) | 0.194–0.542 (med 0.402) | **Keep** — recalibrate threshold |
| Contrast | A | 0.9419 | 17/19 · **79/122** | 0.099–0.851 | 0.500–0.948 | **Tune** — secondary only; arm-A coverage broken (35% of visible eyes have no usable BiSeNet eye mask — the documented class-5 collapse), cannot reach 95% recall alone (max 0.89) |
| Contrast | B | 0.9556 | 19/19 · 122/122 | 0.012–0.863 | 0.064–0.949 | **Tune** — full coverage on landmark masks; adds real value in the joint rule (below) |
| Hair overlap | A | 0.539 | full | 0.000–0.001 | 0.000–0.096 | **Inconclusive** — the corpus contains **zero** hair-over-eye positives (all 19 occluded are lids), so the signal is untested, not rejected. Retain at conservative thresholds pending hard-case sourcing. |
| Hair overlap | B | 0.500 | — | all 0.0 | all 0.0 | **Structurally dead, confirmed** — every arm-B hair value is exactly 0.0 across all 166 eyes (protocol §4 prediction verified empirically). Documented no-op. |

EAR is identical across arms by construction (landmark-only) — measured identical. The
protocol's §6.1 arm-gap criterion (AUC gap < 0.05) passes trivially.

## 3. Anchor sanity checks (§5.4)

| Anchor | Expected | Measured | Status |
|---|---|---|---|
| DSCF4576 closed, EAR 0.108 | occluded | R = 0.101, labeled occluded | ✓ |
| DSCF6961 open, EAR 0.513 | visible | R = 0.504 (L 0.468), visible | ✓ |
| DSCF4612 closed, EAR 0.122 | occluded | R = **0.234**, labeled occluded | **✗ number does not reproduce.** The eye is closed (label agrees) but its EAR is 0.234, matching `eye_visibility.py`'s *own comment* ("closed eyes with EAR 0.20–0.28 (DSCF4612)") — the protocol's 0.122 anchor value appears stale/erroneous. |
| DSCF4454-R "plainly open" 0.136 (threshold comment, not §5.4) | open | R = 0.149 and **the eye is closed** | **✗ mislabeled — this is the root cause of the bad `_MIN_EAR`** (§0). |

## 4. Operating points (cost-asymmetry rule, §2/§5.2)

Chosen by grid search at **occluded recall = 1.00** (protocol target: missed-gate ≤ 5%;
with n=19 a single miss is 5.3%, so 100% recall is the only conforming point):

| Rule | Arm | Occl. recall | Visible false-gate | Partial gated |
|---|---|---|---|---|
| Shipped (`EAR<0.12` OR `contrast<0.45` OR hair) | A | 3/19 = 0.16 | 0/122 | 0/5 |
| Shipped | B | 3/19 = 0.16 | 1/122 | 0/5 |
| EAR < 0.285 alone | A = B | **1.00** | 6/122 = 4.9% | 3/5 |
| **Proposed: EAR < 0.285 OR contrast < 0.55** | A | **1.00** | 6/122 = 4.9% | 3/5 |
| Proposed | B | **1.00** | 7/122 = 5.7% | 3/5 |
| (Arm-B-optimal joint: EAR < 0.265 OR contrast < 0.575) | B | 1.00 | 4/122 = 3.3% | 3/5 |

False-gated visible eyes at the proposed point: 4552-R, 4553-R, 4569-R, 4607-R, 4611-R,
7204-R (+4551-L on arm B) — all narrow-but-open eyes of the same blue-wig subject plus
one heavy-lash case. Cost is the mild class (§2): one eye un-enhanced. The 3 gated
`partial` eyes are the heavy-squint 4559–4561 series, which the protocol explicitly
accepts gating (review §B4: "defensible to gate at ≤30% aperture").

**Shipped production thresholds (2026-08-31, this commit):**
`_MIN_EAR = 0.285`, `_MIN_CONTRAST = 0.55`, hair constants unchanged. Note: an earlier
draft of this section recommended `0.28`, which is *below* the measured occluded
ceiling (0.2833, DSCF4454-L) and would have missed that eye — non-conforming by the
protocol's own 100%-recall bar. Re-verified directly against `sweep_full.json` +
`labels.json` (exhaustive grid search, both arms) before shipping: `0.285` clears the
occluded ceiling and stays below the golden-face fixture's frozen open-eye landmark
reading (0.288), so it doesn't silently re-gate that regression test's eye ops. Both
arms combined: 0/38 occluded eyes missed, 13/244 visible eyes false-gated (5.3%).
A softer alternative worth considering at implementation time: hard gate below EAR 0.22
+ strength ramp 0.22→0.30 (the `eye_artifact_safety.py` pattern), which would convert
most of the false-gates into partial-strength enhancement instead of on/off. Not done
here — separate change, needs new plumbing.

## 5. The `no_face` stratum — out of the eye gate's reach

At the proposed thresholds the gate still passes 8/20 (arm A) off-subject eyes: EAR on
landmarks-fit-to-background is meaningless (measured range 0.094–0.526, spanning both
classes). **The eye gate cannot and should not carry this case.** These are detection
false positives on images where the subject faces away — precisely the territory of the
poster-FP joint veto (`RESEARCH_POSTERFP_VETO_2026_08_19.md`: person-coverage/texture
veto, validated, awaiting owner sign-off). This study adds 10 more motivating cases for
shipping that veto: today the engine will run the full face pipeline (not just eye ops)
on these detections.

## 6. What remains open (deliverables not closed by this run)

- **Hard-case sourcing** (§3.3): blink series, bangs, wig-lace, hand-over-eye, profile,
  sunglasses, small/distant faces — corpus has zero hair/object-occlusion positives, so
  the hair signal and the wig case are still unvalidated.
- **Fitzpatrick IV–VI stratum** (§5.3): absent locally; the fairness claim stays open.
  (Signal *forms* comply: EAR is geometric, contrast is ratio-vs-own-p95.)
- **Label provenance:** single-labeler (model), provisional. A second labeler pass +
  kappa is required before thresholds ship.
- **Threshold update commit:** made 2026-08-31, same session as this doc's initial
  version (`eye_visibility.py` `_MIN_EAR=0.285`/`_MIN_CONTRAST=0.55`, see §4 amendment
  above). Shipped ahead of hard-case sourcing / Fitzpatrick stratum / second labeler at
  the owner's explicit direction; those remain open per the list below. Golden face
  snapshot hashes are unaffected — checked directly: `tests/test_golden_pipeline_face.py`
  passes unchanged, because the fixture's frozen open-eye landmarks (EAR 0.288) clear
  the new threshold and gate identically before and after. Verified on real renders
  instead, comparing the gate forced on vs. off: DSCF4576 (documented closed-eye anchor,
  already caught pre-recalibration — confirms no regression, not the fix itself),
  DSCF6961 (open, unaffected), and two genuinely newly-gated cases from the 0.21-0.28
  EAR band — DSCF4612-R (0.2338, partial squint: subtle sharpen-mask-boundary diff only,
  max delta 6/255) and DSCF4454-L (0.2833, fully closed: zero pixel diff, since the shut
  lid leaves negligible iris/sclera mask area for either code path to act on).
- Arm B forced via `parser._sess = None` does not reproduce env-drift's
  `person_mask`→all-ones degradation (protocol §8 residual gap, unchanged).

## 7. Protocol §9 checklist status after this run

Done: label manifest (as `labels.json`, template folded into the sweep), 83-corpus
labels with evidence, EAR sweep both arms, contrast sweep both arms (form verified:
ratio vs own p95), hair sweep with arm-B structural zero confirmed empirically,
per-signal per-arm ROC/AUC committed, 3×3(+`no_face`) confusion matrices per arm at
shipped and proposed operating points, §5.4 anchors checked (2 pass, 2 expose stale/
wrong prior numbers), per-signal and per-threshold verdicts (§2, §4).
Not done: hard cases, Fitzpatrick IV–VI, second labeler, threshold commit, arm-B test
coverage in `tests/test_eye_visibility.py`.
