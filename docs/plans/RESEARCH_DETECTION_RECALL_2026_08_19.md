# Research — Detection Recall & Precision on the Real DSCF Corpus

**Date:** 2026-08-19 (afternoon session, follows `60166c6`)
**Status:** Measured study complete; no engine code changed (research session per repo convention)
**Runtime:** `.venv` — the pinned supported runtime (mediapipe 0.10.5 legacy backend, `RETOUCH_GPU=0`, `RETOUCH_MEDIAPIPE_BACKEND=legacy`). All numbers below are from actual runs; scripts are committed as evidence.

---

## Executive summary

The 08-19 session report's "next session" item was to re-measure the "~4%
undetected faces" CLAUDE.md figure after the three detection fixes (`cc61e67`
×2, `58d5fbb` ×1). This study answers it on the 83-image DSCF convention
corpus (the real photos in `test_output/`), and finds a different, more
important problem than recall:

1. **Subject recall is 98.8%** (82/83 RetinaFace-confirmed subjects found by
   the engine's actual detection path: legacy FaceMesh @ 2048 proxy). The
   "~4%" claim is stale but the right order of magnitude for subjects. A
   dual-scale pass (2048 ∪ 1024, IoU-dedup) reaches **100%** on this corpus.
2. **A precision problem the figure never covered:** MediaPipe FaceMesh fires
   on anime posters/banners at convention shoots. 18 false-positive
   detections across 83 images (concentrated in 4 shoots). Engine-verified
   harm at `cosplay` recipe strength: poster scenery receives visible
   per-face work (p99 |Δ|=29, max=91 — vs frame mean 2.2).
3. **The single subject miss is caused by FP suppression:** on DSCF4598 the
   poster FP counts as a detection, so the zero-face-gated tiled fallback
   (`1f3318e`) never fires, and the real subject receives no face work
   (mean |Δ|=1.60 vs 4.31 for a detected subject in the same shoot —
   inconsistent treatment across the set).
4. **RetinaFace is not in `requirements/base.txt`** — the `cc61e67` F1/F2
   fixes (threshold pass-through, BGR feed) are inert in the pinned
   supported runtime; `detection.py` imports it opportunistically and
   silently falls back to MediaPipe-only. Production detection IS
   MediaPipe-only today.

---

## Method

### Corpus
83 bare DSCF photos (convention cosplay shoots, 4160×6240 / 6240×4160),
listed in `test_output/detection_recall_study/corpus.txt`.

### Detection strategies measured (all at the engine's settings)
- **S1** — engine-equivalent: legacy FaceMesh @ 2048 proxy (what
  `RetouchEngine._process_with_proxy` actually runs)
- **S2** — legacy FaceMesh @ native 6240px
- **S3** — legacy FaceMesh @ 1024 downscale
- **S4** — 3×3 tiled @ 75% scale (matches `1f3318e`'s tiled fallback shape)
- **S5** — 4×4 tiled @ 60% scale
- **RF** — RetinaFace 0.0.18 @ native + @ 2048 (system python; independent
  CNN detector used as the cross-check anchor, NOT the engine path)

### Ground truth strategy
No labeled corpus exists, so truth is established by **cross-detector
consensus**, not by the union of MediaPipe strategies (which turned out to
be FP-contaminated — see "What the naive numbers get wrong"):

- RetinaFace-native fires on 83/83 images, ~1 face each, at high confidence —
  the subject anchor.
- A MediaPipe detection counts as a *real face* when RF confirms it
  (IoU ≥ 0.4) or when a non-tiled MediaPipe pass at a different scale
  agrees.
- Tiled-only detections unconfirmed by RF were forensically separated
  (Laplacian-variance at fixed analysis scale; confirmed faces p10=499 vs
  unconfirmed p90=256 — total separation on this corpus) and geometry-locked
  across shoot series (normalized-geometry std 0.007–0.008 across 10 frames
  = the same poster re-detected, not a person).

### Scripts (all in `scripts/qa/`, untracked until commit)
| Script | What it does |
|---|---|
| `detection_recall_gt.py` | Runs S1–S5, builds union pseudo-GT (`.venv`) |
| `detection_recall_retinaface.py` | RF native+proxy pass (system python) |
| `detection_recall_analyze.py` | Recall vs union-GT, size buckets, miss crops |
| `detection_recall_merge.py` | Subject/background split, RF confirmation, strip filter |
| `detection_recall_consensus.py` | Cross-detector subject recall, FP-suppression triage |
| `detection_recall_forensics.py` | Texture + geometry FP discrimination (T1–T4) |
| `detection_recall_dualscale.py` | Dual-scale union upside, FP-suppressed fallback count |
| `detection_recall_engine_check.py` | Real `RetouchEngine` runs on FP images (natural) |
| `detection_recall_recipe_harm.py` | Same at `cosplay` strength (flagship use-case) |

Artifacts: `test_output/detection_recall_study/` (`gt_union.json`,
`retinaface.json`, `consensus_summary.json`, `forensics_rows.json`,
`dualscale_summary.json`, verification sheets).

---

## Results

### Recall (subject = RetinaFace's most confident face per image)

| Detection set | Subject recall |
|---|---|
| S1 (2048 proxy, engine path) | **82/83 = 98.8%** |
| S3 (1024) | 82/83 = 98.8% |
| S1 ∪ S3 (dual-scale) | **83/83 = 100%** |
| S1 ∪ S3 ∪ S2 | 83/83 = 100% |

The one S1 miss (DSCF4598): S1 finds only the poster FP; S3 alone finds the
subject. The `1f3318e` tiled fallback would also have recovered it — but it
is gated on zero faces, and the FP is a face.

### What the naive numbers get wrong (methodology finding)

The raw union pseudo-GT (196 faces) gives S1 recall 51%, and "largest box =
subject" gives 83.1% subject recall. Both are corrupted by tiled-pass
false positives on anime scenery: poster FP boxes are *larger* than real
subject boxes in several frames, and 25 tiled-only unconfirmed detections
survive even a conservative strip filter. Any future recall study on
convention corpora must not use MediaPipe-only unions as ground truth —
RF cross-check + texture forensics is the minimum bar.

### Precision (the real finding)

- 18/100 S1 detections (18 images of 83) are RF-unconfirmed MediaPipe
  detections with poster/anime texture signatures (Laplacian var 87–349 vs
  real-face p10 499; zero distribution overlap on this corpus).
- Concentration: DSCF4588–4592, 4596–4601 (one shoot, same poster every
  frame — 10 images), DSCF4554/4555 (2), DSCF4576/4577 (pair), DSCF4612/
  4617 (pair), DSCF4618 (1).
- Engine-verified harm (real `RetouchEngine.process` runs, 2048-capped):

| Image | Recipe | Poster-FP box | Subject box |
|---|---|---|---|
| DSCF4588 | natural | mean 0.39 / p99 1.0 | mean 0.80 / p99 3.0 |
| DSCF4588 | cosplay | mean 3.74 / p99 **29** / max **91** | mean 4.31 / p99 42 |
| DSCF4598 | natural | mean 0.54 / p99 2.0 | mean 1.32 (global-only) |
| DSCF4598 | cosplay | mean 2.25 / p99 7.0 | mean 1.60 — **no face work** |

### Runtime availability finding

`retinaface` appears in no requirements file. In the pinned `.venv`
(mediapipe 0.10.5), `detection.py`'s `from retinaface import RetinaFace`
raises `ImportError` and silently degrades to MediaPipe-only. Therefore:
- The `cc61e67` F1 (threshold) / F2 (BGR) fixes have **no effect in the
  supported runtime** — they only help environments where the user
  independently installed retinaface.
- The Tasks-path architecture (RetinaFace boxes → MediaPipe landmarking)
  never executes on the supported runtime; the legacy FaceMesh path is the
  production path.

---

## Recommended follow-ups (design candidates — NOT implemented)

> **UPDATE, same day (post-study):** follow-up #1 was implemented and landed
> after this section was written — see "Follow-up #1 outcome" at the bottom.
> Items below retain their original research framing.

Ranked by expected value / risk:

1. **Dual-scale detect union (S1 ∪ S3)** — +1 subject recall here (the only
   miss), ~+10–30 ms per image at 2048-proxy sizes, no new deps. Design
   note: dedup at IoU 0.5, keep the higher-quality landmark set (S1's) on
   overlap.
2. **Widen the tiled-fallback gate from "zero faces" to "no *confirmed*
   face"** — the DSCF4598 class of failure. Needs a confirmation signal:
   RF when available; otherwise the texture discriminator below.
3. **Poster/anime FP veto (candidate signal, needs broader validation)** —
   Laplacian-variance at fixed analysis scale separated real faces from
   poster FPs with zero overlap *on this corpus* (p10 499 vs p90 256).
   Before shipping as a rule it must be validated on non-convention
   portraits (soft-focus, heavy-makeup, beauty-filtered faces could sit
   lower). A veto threshold ≈ 350 with a wide margin of safety is the
   obvious candidate; treat as research, not a spec.
4. **Decide RetinaFace's production status** — either add to requirements
   (heavy TF dependency; last verified against 0.0.18's internal contract)
   or document MediaPipe-only as the supported detection path and stop
   describing the F1/F2 fixes as live recall gains.
5. **Re-word the CLAUDE.md "~4%" limitation** — it conflates subject and
   background faces. Measured: subject miss rate 1.2% on this corpus;
   background-face recall 27–35% (crowd faces, largely out of scope for
   portrait retouching but relevant for future group-shot features).

## Limits of this study

- 83 images, one photographer, convention-cosplay subject matter; the FP
  rate is likely corpus-specific (posters are rare in studio portraits).
- Ground truth rests on RetinaFace agreement + forensics, not human labels;
  RF itself has blind spots (it confirmed 83/83 subjects here but found only
  84 faces total — it misses almost all background faces MediaPipe finds,
  so background-face recall numbers are lower bounds with unknown bias).
- S2 (native 6240px FaceMesh) is not the engine path and is included only
  as evidence that scale, not model, drives the misses.
- The engine-harm measurements are at 2048-capped input; at full native
  resolution the same detections apply (detection runs at the proxy either
  way) but deltas would scale with the per-face stages' native-res work.

---

## Follow-up #1 outcome — dual-scale detect union (implemented same day)

Design refined from the research candidate after two pre-implementation
measurements changed it materially:

1. **Naive S1∪S3 union rejected.** The 1024 pass adds 6 boxes to the union;
   only 2 are real (DSCF4598's subject + one blurry background face) — the
   other 5 are MORE poster FPs, including DSCF4454's "recovered second
   face" from the `1f3318e` commit message, which this study's texture
   forensics show was a poster all along (tex=65.5 vs real-face p10=499).
2. **Person-gate added instead.** The engine's own selfie-segmenter mask
   separates the poles perfectly on this corpus: poster additions
   person-coverage 0.000 vs RF-confirmed subjects 1.000 (n=8 controls all
   1.000). Gate: central-40% coverage >= 0.5.

Implementation (`retouch/detection.py::_dual_scale_augment_legacy`):
- Runs only when the main 2048-scale legacy pass found >= 1 face AND
  max(w, h) > 1024 (scale is only a plausible failure axis for large
  inputs; zero-face images keep the existing tiled fallback).
- Second FaceMesh pass at 1024px; additions must pass IoU>0.5 dedup vs the
  main pass (main pass's higher-res landmarks win overlaps) AND the person
  gate. Segmenter failure => add nothing (recall nicety must not trade
  precision).
- `confidence_source` stays `mediapipe_presence_unavailable` (same model,
  same provenance class; `face_quality.py`'s measured-confidence set is
  untouched).

Verified end-to-end (scripts committed: `detection_recall_s3_additions.py`,
`detection_recall_persongate.py`, `detection_recall_verify_change.py`,
`detection_recall_engine_after.py`):

| Check | Result |
|---|---|
| Subject recall, engine path @2048 | **83/83 = 100%** (was 82/83) |
| New detections vs pre-change | **1** — DSCF4598 subject, RF-confirmed |
| RF-unconfirmed detections | 18 before = 18 after (zero new FPs) |
| Median detect() at proxy | 22 ms (p90 30 ms) |
| DSCF4598 subject face work (cosplay) | mean |Δ| 1.60 → **3.36** (detected-class ~4.3) |
| tests/test_detection.py + new dual-scale tests | 88 passed |
| Full suite | **4,532 passed, 11 skipped, 0 failed** |

The 18 pre-existing poster FPs are unchanged by design — that is follow-up
#3 (poster-FP veto), still gated on non-convention validation (soft-focus /
heavy-makeup faces could sit below the texture threshold; the person gate
cannot remove them because MediaPipe fires on posters the segmenter
sometimes includes in-frame).
