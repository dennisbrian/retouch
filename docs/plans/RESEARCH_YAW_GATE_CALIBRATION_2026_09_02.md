# Research Results — Yaw-Gate Calibration + Composite-Mask Epsilon Bug (2026-09-02)

**Runtime:** `.venv/bin/python` (mediapipe 0.10.5, cv2 4.11, `RETOUCH_GPU=0`,
`RETOUCH_MEDIAPIPE_BACKEND=legacy`). Bare `python3` (cv2 4.13) produces different
golden hashes and no detection — never use it for studies.
**Corpus:** 83-image DSCF corpus (`test_output/detection_recall_study/corpus.txt`),
largest detected face per image, images pre-shrunk to max-dim 2048.
**Harness:** `scripts/qa/yaw_gate_sweep.py` (measurement, writes
`test_output/yaw_gate_study/sweep.json`), `scripts/qa/yaw_gate_render.py`
(slimming/sculpt contact sheets for visual judgment).
**Shipped:** `9c26493` (yaw band), `f69ab1e` (mask epsilon — see §5). Same session also shipped `e73d3ba` (undereye alias) and `46be031` (lips/teeth roundtrip containment) from the 2026-08-31 audit.

## 0. Summary

1. The nose-bridge/temple x-ratio (`lm6` vs `lm234/454`) is a *good* yaw proxy —
   Spearman 0.92 against crop-remap-corrected MediaPipe-depth yaw — but its shipped
   bands (geometry 1.3→1.6, relight/sculpt 1.5→1.7) sat at ~2–10° of head turn.
   Slimming was damped on 68/83 corpus faces (zero on 58/83); relight+sculpt on 61/83.
   Same failure shape as the EAR gate (`RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31.md`):
   right signal, wrong operating band.
2. Slimming at strength 70 with the gate bypassed renders artifact-free through ratio
   ≈4.0 (~25° corrected depth-yaw). New shared band **2.5 → 4.0** (smoothstep), owned
   by `utils.YAW_GATE_START/END`, used by geometry, relight and sculpt.
3. While validating sculpt, a far larger defect surfaced: on **20/83 (24%)** corpus
   images every skin-region op was silently discarded at composite because float32
   masks overshoot 1.0 by one ulp and every normaliser treated `max() > 1.0` as
   "this is a 0–255 mask". Fixed at 21 sites. This likely explains earlier
   "ops no-op on some images" observations (see `bonodori-qa-delta-blindness` memory).

## 1. Yaw signals measured per face

| signal | definition | Spearman vs \|z-yaw corr\| |
|---|---|---|
| `ratio_bridge6` (shipped) | max/min of \|x6−x234\|, \|x454−x6\| | **0.92** |
| `ratio_nasion168` | same with lm168 | 0.81 (uncorrected z) |
| `ratio_tip1` | same with nose tip | saturates early (protrusion) |
| `eye_width_ratio` | \|x33−x133\| vs \|x362−x263\| | 0.69 |
| `zyaw_corr_deg` | atan2(Δz·zscale, Δx) on temples 234→454 | reference |
| `pnp_yaw_deg` | solvePnP, generic 6-point head model | **rejected** (ρ 0.52; 8–14° on frontal faces; sign flips) |

**Crop-remap correction is required.** `FaceDetector._remap_landmarks` rescales x/y to
full-image units but leaves z in crop units, so faces found by the tiled/legacy crop
pass (22/83 here, zscale 0.33 or 0.5) had depth-yaw inflated up to 2–3×. Uncorrected,
ρ was 0.87 and the 15–20° bin contained a ratio-293 face. `zscale` is tagged in the
sweep via a monkeypatch; the production code is unaffected (no production path reads
z for yaw).

**Absolute calibration is NOT established.** MediaPipe z is "roughly x-scaled"; the
degrees quoted are a consistent relative unit, not measured head pose. The band was
chosen by visual judgment on renders, with the angle only used to rank faces.

## 2. Ratio ↔ corrected depth-yaw on the corpus

| ratio bin | \|yaw\| min / median / max | n |
|---|---|---|
| 1.0–1.3 | 0.2 / 1.0 / 9.7 | 15 |
| 1.3–1.6 | 2.2 / 5.1 / 10.3 | 10 |
| 1.6–2.0 | 5.5 / 8.9 / 14.2 | 26 |
| 2.0–2.5 | 7.6 / 12.6 / 16.5 | 15 |
| 2.5–3.5 | 13.0 / 15.3 / 17.3 | 9 |
| ≥3.5 | 16.4 / 22.1 / 35.6 | 8 |

## 3. Candidate bands (faces at full / partial / zero strength, of 83)

| band | full | partial | zero |
|---|---|---|---|
| shipped ratio 1.3→1.6 | 15 | 10 | **58** |
| ratio 2.0→3.0 | 51 | 21 | 11 |
| ratio 2.2→3.5 | 54 | 21 | 8 |
| **ratio 2.5→4.0 (chosen)** | 68 | 8 | 7 |
| angle 15→30 | 69 | 11 | 3 |

The 7 zeroed faces under the chosen band are ratio >4 strong turns (DSCF4454, 4554,
4555) or crowd shots where the *largest* detected face is a blurred background person
(DSCF4596/4597/4599) — those three are a corpus-labelling issue, not a gate issue.

## 4. Visual validation

`yaw_gate_render.py`, slimming=70, gate bypassed, jaw crops + ×3 diff heat:
DSCF4568 (r 1.01), 4563 (1.97), 4574 (1.91), 4551 (3.99), 7204 (3.37): warp is a clean
jawline contraction in every case; diff energy sits on the jaw edge and displaced hair
boundary, no tearing or asymmetric collapse at 3.99. Sculpt could not be validated in
the same pass — it was inert on 3/4 test faces for the reason in §5; re-rendered after
that fix (see §5.3).

## 5. Composite-mask epsilon bug (found during sculpt validation)

### 5.1 Symptom
`relighter.sculpt` changed the face-core crop by up to 20 levels on DSCF4568, but the
final image differed by ≤5. Relight: 72 in-core → 4 final. Slimming (stage 1, whole
canvas) survived. So the loss was between `_process_face_core` and the output.

### 5.2 Cause
`_FaceResult.skin_mask.max()` was **0.004 = 1/255**. `regions.skin.max()` entering the
core was `1.000000238` — one float32 ulp above 1.0 after feathering — and
`_norm_mask` did `if m.max() > 1.0: m /= 255`. `_composite_faces` alpha =
max(skin_hair, lips, sharpen), so over skin alpha ≈ 0.004 and the edited crop was
blended at 0.4%. Lips (max exactly 1.0) and sharpen kept alpha, which is why renders
still visibly changed and the defect hid for months. Prevalence: **20/83** corpus
images (skin max > 1.0); on the other 63 the max landed at 0.9999997.

### 5.3 Fix + verification
Threshold `> 1.5` + clip in `perf_optimizations._norm_mask` and `utils.normalize_mask`
(feeds `feather_mask`), clip at the feather source, same threshold at 19 further
sites. After the fix on DSCF4568: sculpt 20 / relight 74 / smooth 32 / equalize 48
levels reach the output. `natural` before/after viewed
(`test_output/yaw_gate_study/mask_epsilon_before_after.jpg`): skin smoothing and tone
evening now present, no seams. Regression: `tests/test_mask_norm_epsilon.py`.

## 6. Open items

- **Neck gate** (`skin.py`, hard 1.3 on the same ratio) not recalibrated — different
  op; needs its own render check. At 1.3 it is closed on 68/83 corpus faces.
- **Absolute yaw calibration**: needs a pose-labelled set or a proper canonical-model
  PnP (the mediapipe wheel in `.venv` ships no `canonical_face_model.obj`).
- **Corpus hygiene**: DSCF4596/4597/4599 largest face ≠ subject; exclude or relabel.
- **Sculpt at yaw** — done after the §5 fix (`test_output/yaw_gate_study/sculpt_sheet.jpg`,
  sculpt=70): DSCF4563 (r 1.97, factor 1.0) clean cheek/jaw shading; DSCF7204 (r 3.37,
  factor ≈0.42) subtle, confined to skin, no seams; DSCF4551 (r 3.99, factor ≈0) only a
  low-amplitude blotchy residual visible at ×3 diff gain — worth a look at whether a
  sculpt>0 code path still round-trips the canvas when effective strength is ~0.
- **Re-run past QA that concluded an op "does nothing"** on real images — 24% of those
  conclusions may have been this bug.
