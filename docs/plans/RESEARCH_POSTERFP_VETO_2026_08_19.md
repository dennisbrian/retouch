# Research — Poster-FP Veto Validation & RetinaFace Dependency Dead-End

**Date:** 2026-08-19 (evening session, follows `8342dbb` + `8811320`)
**Status:** Measured study complete; a partial-veto implementation candidate is validated but NOT implemented (needs owner sign-off on the residual-risk trade)
**Runtime:** `.venv` — pinned supported runtime (mediapipe 0.10.5 legacy, pb 3.20.3). All numbers from actual runs; scripts committed as evidence.

---

## Executive summary

Two open threads from the detection study (`RESEARCH_DETECTION_RECALL_2026_08_19.md`)
were closed with measurements:

1. **Poster-FP veto (follow-up #3): a SAFE PARTIAL VETO exists.** The joint
   person-coverage + texture rule kills **14/18 poster FPs with 0/82 real-face
   collateral**, on a threshold plateau (T=50–350, all zero-collateral — the
   rule is robust to the exact threshold, not knife-edge). The remaining 4
   FPs are unvetoable by any measured single feature without false
   positives: they sit ON the person (coverage ≥ 0.978), and every
   additional discriminator (chromophore skinlike-fraction) overlaps real
   faces. A texture-only veto is proven UNSAFE (real faces degrade to
   Laplacian-var 1.0–170 under blur/makeup abuse, below poster baselines).
2. **RetinaFace production status (follow-up #4): structurally dead in the
   supported runtime.** `pip install --dry-run retina-face==0.0.18` resolves
   to `protobuf-6.33.6 + tensorflow-2.20.0 + keras-3.10 + numpy-2.0.2 +
   opencv-python-5.0` — violating every core pin (pb<4, mp==0.10.5,
   numpy<2, cv2<4.12). The `cc61e67` F1/F2 fixes can never ship in the
   pinned env; the documented decision should be MediaPipe-only.

---

## Part 1 — Poster-FP veto: three discriminators, measured

### A1. Person-coverage (engine's own selfie-segmenter)

Central-40% coverage at 2048 proxy scale, all 100 S1 detections:

| Class | n | min | p10 | median | max |
|---|---|---|---|---|---|
| RF-confirmed real faces | 82 | **1.000** | 1.000 | 1.000 | 1.000 |
| Poster FPs | 18 | 0.000 | 0.000 | 0.000 | 1.000 |

13/18 posters at ≤ 0.01; 5 at ≥ 0.879 (posters physically on/near the
person). **Zero real faces below 1.000** on this corpus — the 0.5 gate has
maximum headroom against real-face collateral *for the coverage term
alone*.

### A2. Real-face texture floor under degradation (veto safety)

83 RF-confirmed subjects × 17 degradation chains (blur σ=0.5–4, median
k5/k9, bilateral×2 "heavy makeup", JPEG q85–30, downsample 2×/4×,
combinations), fixed 256px analysis geometry:

| Degradation | min over 83 faces |
|---|---|
| baseline | 312.8 |
| blur σ=1 | 38.9 |
| blur σ=4 | 1.0 |
| bilateral×2 (makeup) | 9.1 |
| makeup + JPEG 70 | 31.6 |
| **GLOBAL floor** | **1.0** |

**A texture-only veto at the poster ceiling (≈350) would eat real faces** —
blurred/soft-focus/beauty-filtered portraits legitimately sit at 30–170.
This kills the research doc's original "tex<350 veto" candidate as a
standalone rule.

### A3. Poster texture ceiling under adversarial push

18 poster crops × unsharp/noise pushes: unsharp 1.0 already lifts poster
medians to 450 (above real-face p10=499 territory); noise explodes the
metric (2k–29k — Laplacian-var is meaningless on noisy input). Separation
is one-directional: pristine posters sit low, but nothing *keeps* them
there.

### A4. Joint rule + leave-one-out

```
veto = (person_coverage < 0.5)                       # kills 13
    OR (person_coverage < 0.9 AND texture < T)       # kills 1 more (DSCF4589)
```

Leave-one-out over T (50–350): **14/18 killed at 0 collateral for every
candidate T** — a plateau, not a knife-edge. Real-face corpus-wide texture
floor is 312.8 pristine / 31.6 under makeup+JPEG, so T=80 sits below any
pristine real face but above the makeup-degraded floor; the coverage<0.9
conjunction is what makes it safe (no real face is below 0.9 coverage on
this corpus).

Residual: 4 FPs survive (DSCF4588 cov=0.978, DSCF4612/4617/4618 cov=1.000).

### Chromophore probe on the 4 survivors — measured, insufficient

`decompose_chromophores` skinlike-fraction (Hb∈[0.05,0.6] ∧ melanin∈[0.02,0.5])
at 320px central-60%:

| Class | min | p05 | median | max |
|---|---|---|---|---|
| Real faces (n=82) | 0.019 | 0.093 | 0.240 | — |
| Poster FPs (n=18) | — | — | 0.013 | 0.241 |

Overlaps both directions: posters reach 0.241 (inside real median); real
faces reach 0.019 (below poster p95). Best-case threshold T=0.05 kills 4/4
survivors but takes 3/82 real faces with it. **Rejected as a veto term.**
(The 3 low-skinlike real faces — DSCF4551/4463/8007 — are exactly the
heavy-makeup/wig-coverage population the veto must never eat. Interesting
side-observation: hb_std separates better (real min 0.087 vs poster max
0.433 overlap only slightly), but not cleanly enough.)

### Veto impact if implemented

14 FPs removed across 13/83 images (16% of corpus). Engine effect: those
images stop receiving per-face work on poster scenery (previously p99 |Δ|
up to 29 at cosplay strength). 4 images keep their on-person poster FP.

### Implementation candidate (NOT shipped this session)

Same shape as `_dual_scale_augment_legacy`'s gate: after detection, compute
person coverage + texture per face; drop vetoed faces before parsing. Cost:
texture is ~0.2 ms/face at 256px; person mask already computed at proxy
scale. **Open question for owner:** is 14/18 with a documented residual
(hard 4) worth shipping, or should the convention-corpus FP problem wait
for a BiSeNet-parsing-level signal (skin-mask plausibility per detected
face, available downstream where the poster would fail face-region
parsing anyway)?

---

## Part 2 — RetinaFace: structurally unsupportable in the pinned runtime

Chain of evidence (all commands run this session):

1. `retinaface` is absent from `requirements/*` and from `.venv`; the
   `detection.py` opportunistic import silently degrades to MediaPipe-only.
2. The installed copy (system python) is `retina-face 0.0.18` (dist name
   `retina-face`, import `retinaface`), `Requires-Dist: tensorflow>=1.9.0`.
3. PyPI state: `retina-face` latest is 1.1.1; **0.0.18 is still resolvable**
   (unlike the bare name `retinaface==0.0.18`, which resolves to a different
   abandoned 0.0.x series max 1.1.1).
4. `.venv pip install --dry-run retina-face==0.0.18` →
   `ResolutionImpossible` / would install `protobuf-6.33.6`,
   `tensorflow-2.20.0`, `keras-3.10.0`, `numpy-2.0.2`, `opencv-python-5.0.0.93`
   — breaking **every** core pin: `protobuf<4` (mp 0.10.5 requires pre-4
   protobuf API per `requirements/base.txt` comment), `mediapipe==0.10.5`,
   `numpy<2`, `opencv-contrib-python<4.12`.
5. System python proves the conflict is environmental, not hypothetical:
   it runs retina-face 0.0.18 *because* it has protobuf 6.33.6 — and that
   env's mediapipe is 0.10.35 (no Solutions API, unsupported by this repo).

**Conclusion:** the Tasks-path (RetinaFace boxes → MediaPipe landmarking)
cannot ship alongside the pinned supported runtime. Recommended action:
document MediaPipe-only as the supported detection path (CLAUDE.md +
TROUBLESHOOTING), stop describing F1/F2 as live recall gains, and treat any
future RetinaFace revival as a full dependency-strategy decision (e.g.
ONNX-ported RetinaFace without TF, or scrapping for a MediaPipe-only
multi-scale stack — which `8811320` already started).

---

## Scripts (evidence, `scripts/qa/`)

| Script | Purpose |
|---|---|
| `posterfp_persongate_full.py` | A1: coverage for all 100 detections |
| `posterfp_texture_sweep.py` | A2+A3: degradation floor + adversarial ceiling |
| `posterfp_joint_veto.py` | A4: joint rule + leave-one-out |
| `posterfp_survivors.py` | chromophore probe on the 4 survivors |
| `posterfp_chromophore_full.py` | corpus-wide chromophore validation |

Artifacts: `test_output/detection_recall_study/posterfp_*.json`.

## Limits

- Same 83-image convention corpus; the "real-face coverage = 1.000" pole is
  a property of single-subject portrait framing. Group shots / arms-over-
  shoulder occlusion will produce real faces below 1.0 coverage — the veto
  threshold against *real-face* collateral must be re-validated on a
  group-shot corpus before shipping.
- Chromophore plausibility ranges ([0.05,0.6] Hb etc.) are first-principles
  placeholders, not calibrated ranges.
- The 4 survivors were verified as posters by texture + geometry + RF
  silence only; no human label was recorded this session (image-viewing
  unavailable in this environment).
