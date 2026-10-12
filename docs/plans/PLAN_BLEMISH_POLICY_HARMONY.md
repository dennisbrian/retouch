# PLAN — Per-Blemish Policy Layer + Facial Harmony Objective

**Date:** 2026-07-17 · **Author:** Fable research pass
**Origin:** last unstarted Tier-1 item, `FABLE_TASK_LIST_2026_07_15.md` via
`VISION_SOTA_FACE_ENGINE_2026_07_15.md` reconciliation items 2–3 (per-blemish
preserve/attenuate/enhance policy; facial harmony / cross-region consistency
objective).
**North-star reference:** owner-supplied professionally retouched cosplay
portrait (`~/Downloads/Screenshot_20260717_001633_...jpg`, viewed 2026-07-17).
Defining traits, confirmed by inspection: (1) face AND body skin (arms, chest,
torso, thighs) at the same near-poreless porcelain finish — no texture seam at
neck/chest; (2) dimensionality fully preserved (nose/cheek modeling, collarbone
and limb shading, controlled speculars) — smooth, not flat; (3) zero visible
defects yet no banding/posterization — long clean tonal ramps; (4) deliberate
identity styling KEPT (faint drawn under-eye cheek lines are makeup, not
defects, and the retoucher preserved them).

**Spike evidence:** `scripts/spike_harmony_metrics.py` (this pass; panels in
`test_output/spike_harmony_DSCF{4550,7204,7011}.png`). All numbers below are
measured, seed-free deterministic, re-runnable.

---

## 1. Codebase findings (evidence)

### 1.1 What already exists per mark type — fragmented, no unified policy

| Mark type | Detector | Action today | File |
|---|---|---|---|
| freckle | `FreckleRemover.classify_anomalies` (LAB + size heuristics) | inpaint at `freckle_removal` strength | `retouch/freckle.py` |
| beauty mark / mole | same classifier (`beauty_mark` class) + R10 `blemish_vs_mole` (mel/hb compact spots) | always preserve (hard circle in preserve mask) | `retouch/freckle.py:322-326`, `retouch/skin.py:1854 apply_mole_protect`, `retouch/skin_chromophore.py:148` |
| acne/blemish | `FreckleRemover` `blemish` class (detected, **never acted on** — only `freckle` class is healed, `freckle.py:330-336`); separately `BlemishRemover` (`retouch/blemish.py:68`) | mid-band reduction / inpaint, own slider | two disjoint paths |
| stray/flyaway hair | `hairwork.py:862-866` via `SpotHealer` | remove | `retouch/hairwork.py` |
| drawn/tattooed makeup mark | **no class exists** | destroyed by any smoothing/heal strong enough | — |
| sensor dust | **no class exists** (noise class ≈ tiny components only) | — | — |
| scar | **no class exists** | — | — |
| vellus-hair sheen | no discrete class; high band is globally protected by `TEXTURE_ADAPT` floor | kept via frequency policy | `retouch/frequency.py:44-64` |

Key structural fact: each detector feeds its own op with its own slider. There
is no shared mark record, no per-class action parameter, and the class list is
closed (`freckle.py:23 _CLASS_TYPES` is a frozenset of 4). "Policy" today is
hardcoded: freckles→remove, beauty_marks→preserve, blemish→ignore(!), noise→ignore.

### 1.2 Tone-invariance bugs found in `freckle.py` (5th instance of the known bug class)

The project has fixed 4 instances of "absolute intensity threshold on a
tone-scaled signal" (`PLAN_P4_MAKEUP_UNMIX.md` §14-17). `freckle.py` contains
two more, unrecorded:

- **`_classify_anomaly` L gates (`freckle.py:157-170`):** `l_mean >= 70` scores
  freckle, `l_mean < 70` scores beauty_mark (weight 0.5 — the largest single
  beauty-mark cue). `l_mean` is the anomaly's absolute LAB L, which scales with
  subject skin tone. On Fitzpatrick V–VI skin ordinary freckles/blemishes sit
  below L=70 → misclassified as beauty marks → wrongly preserved (removal
  slider silently inert); on very light skin, genuinely dark moles near L≈70
  lose protection. Fix: margin below the face's own skin-L median (the
  `a_norm = (a_mean − a_median)/a_std` pattern two lines up is already
  correct — apply the same normalization to L).
- **`_detect_components` deviation gate (`freckle.py:91-92`):** `deviation >
  7.0` on gray intensity. The deviation is local-baseline-relative (good), but
  reflectance contrast is multiplicative: the same physical spot yields a
  smaller *absolute* gray deviation on darker skin → systematic
  under-detection. Fix: `deviation / (local_mean + eps) > ratio` or normalize
  by per-face skin median luminance.
- Also `red1/red2 = cv2.inRange(hsv, (0,60,50)…)` (`freckle.py:105-107`):
  fixed absolute S/V floors on the redness channel; same class, milder impact
  (it only augments candidates).

These must be fixed **before** the policy layer builds on this classifier —
otherwise the policy inherits tone-dependent class assignments.

### 1.3 Chromophore features: usable only as within-face-relative densities

`PLAN_S1_CHROMOPHORE_AUDIT.md`: `decompose_chromophores` is two fixed scalar
log-ratios on gamma-encoded sRGB (melanin ≈ `10·log(G/B)`), WB leaks into both
maps (2–5× melanin swing warm↔cool), mel/hb entangled (reddening drags melanin
−2.6), and `blemish_vs_mole` is ranked the **highest-risk consumer** because
its product promise needs attribution the model lacks. Consequences for the
classifier here:
- Features may use mel/hb **only** as margin-above-this-face's-own-median (the
  `undereye.py:294-298` median+MAD pattern, which S1 rates robust), never raw
  magnitude, never cross-photo comparisons.
- The mole-vs-blemish decision should lean on the compact-spot geometry
  heuristic (`_compact_spots`) plus L/a\* relative cues, with mel-vs-hb
  attribution as a weak vote only — exactly S1 §5's interim guidance.

### 1.4 Deliberate-makeup-mark detection: what P4 §19-20 licenses and forbids

- **Forbidden basis:** broad-coverage detection via per-face median/MAD outlier
  stats is structurally impossible — cues self-mask once makeup dominates the
  baseline (§19.1 F1 coverage cliff at 44–60%; §20.1). Do NOT build a
  "foundation present?" feature this way.
- **Licensed basis:** compact artifacts survive outlier statistics *if* the cue
  signals are high-passed first (§20.2 Track A: subtract a ≥0.15·face-width
  Gaussian baseline before median/MAD — reverses the measured 11.7σ→1.7σ MAD
  inflation collapse). The reference image's drawn cheek lines are exactly this
  regime: compact, high-chroma-or-high-contrast, minority-coverage strokes.
- **Discriminative cues defect-vs-drawn** (all within-face-relative):
  geometry — strokes are elongated (high eccentricity), smooth-curved, with
  coherent orientation (the orientation-field machinery already exists:
  `frequency.py:_compute_orientation_field`); symmetry — deliberate marks
  frequently mirror across the face midline (478 landmarks give a
  midline/normalized coordinate frame; `detect_over_retouch_asymmetry` already
  builds mirrored comparisons); color — pigment sits off the wearer's own
  skin locus in density space (P4 §20.3's neck-anchored bare-skin locus;
  `parsing.py` provides `neck` = BiSeNet label 14), while acne sits along the
  hb axis and moles along mel. Blemishes are compact blobs, not strokes; stray
  hairs are thin curves but darker/thinner and also occur off-face.

### 1.5 Body skin path: reachable but texture-blind

`engine.py:3224 _stage_body_skin` (reachable since `b8bec2d` via
`BodySkinStage`, `stage_wrappers.py:99`):
- Mask: LCH heuristic (`skin_mask_lch`, hue 25±25, chroma≥8) ∩ person mask,
  minus face-hull/hair/lips, geodesic face-contiguity filter, OKLCh C<0.18
  tattoo gate. Solid engineering, but the hue/chroma constants are absolute —
  coverage on very deep or strongly-lit skin tones is unaudited (see §6).
- Smoothing: **single guided filter** (`engine.py:3404-3424`) — no frequency
  separation, no texture-adaptive floor (`TEXTURE_ADAPT` protects only the
  face path), no mark classification, no mole protect, no freckle logic on
  body. `body_match_face` matches median LAB **tone** only (`engine.py:3427+`);
  nothing matches **texture**. This is the exact mechanism that produces the
  face/body texture seam the reference image avoids.
- `blend_masked` round-trips through uint8 inside the float stage
  (`engine.py:3421-3424`) — an F1-float32-completion site; banding-relevant for
  porcelain looks (long smooth ramps on chest/thighs are where 8-bit
  quantization shows first).
- `skin.py::texture_transplant` (same-face high-band donor cloning) already
  exists — the natural "re-texture over-smoothed zones" tool for both
  attenuate-not-erase policies and body texture parity.

### 1.6 QA scaffolding that the harmony objective can extend

`qa_detectors.py` already has: pore-band radial FFT energy + reference-ratio
plastic metric (`detect_pore_spectrum_distance`, band = period 2–8 px),
banding, halo, seam, ΔE2000 drift, mirrored-asymmetry. All are face-global or
image-global; **none compares face vs body**. `run_all` is wired into
`engine._run_qa` with `QABackoff` recovery (verified live per VISION doc item
1) — so a new harmony detector added to `run_all` is automatically both a QA
gate and a backoff-driven auto-tuning signal. This is the cheapest credible
integration point.

---

## 2. Taxonomy + policy schema

### 2.1 Mark taxonomy (v1 — 9 classes)

| class | detector basis (all within-face-relative) | v1 confidence | default action |
|---|---|---|---|
| `mole` | compact, L margin below skin median, low internal variance, mel-axis vote; merges freckle.py `beauty_mark` + R10 mole mask | HIGH | preserve |
| `freckle` | small, a\*-margin above median, flat, clustered (freckles come in fields — cluster density is a strong new cue) | HIGH | preserve (natural) / attenuate (glam) |
| `acne_blemish` | compact, hb-axis + chroma margin, high internal L variance (inflamed) | HIGH | remove |
| `scar` | elongated or irregular, low chroma, L margin either sign, sharp-edged | LOW (schema slot + user brush; auto-detect deferred — see §7 NO-GO) | preserve |
| `drawn_makeup_mark` | stroke geometry (eccentricity>~3, smooth curvature), off-skin-locus color in density space (neck-anchored), midline-symmetry bonus, high-passed cues per P4 §20.2 | MEDIUM | preserve |
| `stray_hair` | thin curve, high length/width, extends beyond skin mask; existing hairwork detector | HIGH | remove |
| `sensor_dust` | small, soft-edged, appears at fixed high-band scale, luminance-only (no chroma signature), also present off-skin/background | MEDIUM | remove |
| `vellus_sheen` | not a discrete mark — a regional high-band + specular property; policy routes to frequency/specular strength, not heal | n/a (regional) | preserve |
| `unknown` | anything below class-confidence gate | — | **preserve** (safe default: never heal what we can't name) |

### 2.2 Data model

```python
@dataclass(frozen=True)
class MarkRecord:                     # retouch/marks.py (new module)
    mark_id: int
    mark_class: str                   # taxonomy above
    confidence: float                 # [0,1]
    centroid: tuple[float, float]
    area_norm: float                  # area / face_width² — scale-free
    bbox: tuple[int, int, int, int]
    features: dict[str, float]        # l_margin, a_margin, mel_rel, hb_rel,
                                      # eccentricity, edge_sharpness,
                                      # symmetry_score, cluster_density
    on_body: bool                     # detected on body mask vs face mask
```
`FreckleClassification` remains (freckle.py stays API-stable); `MarkRecord` is
the superset the unified detector emits, with adapters from the three existing
detectors (freckle, blemish_vs_mole, hairwork flyaway).

### 2.3 Policy schema (recipe JSON, single-source-of-truth via params.py)

```jsonc
"mark_policy": {
  "mole":             {"action": "preserve"},
  "freckle":          {"action": "attenuate", "strength": 60},
  "acne_blemish":     {"action": "remove"},
  "scar":             {"action": "preserve"},
  "drawn_makeup_mark":{"action": "preserve"},
  "stray_hair":       {"action": "remove"},
  "sensor_dust":      {"action": "remove"},
  "min_confidence": 0.6,       // below this → class treated as "unknown"
  "unknown":          {"action": "preserve"},
  "body": "inherit"            // or a full per-class override block
}
```
- Actions: `preserve` (hard guard mask, as `mole_mask` today) ·
  `attenuate(strength 0-100)` (mid-band local reduction / partial inpaint
  blend — reduce contrast, keep the mark; freckle "softening" not erasure) ·
  `remove` (inpaint_and_blend, existing) · `enhance(strength)` (increase local
  contrast of the mark, e.g. freckle-forward editorial looks or darkening a
  signature mole; implement as inverse of attenuate + optional
  `texture_transplant` reinforcement).
- **Backwards compatibility invariant:** absent `mark_policy`, behavior is
  byte-identical to today (freckle_removal/mole_protect sliders keep working;
  policy layer compiles them into an implicit policy). This is the same no-op
  guarantee discipline used for skin.rosy (`d0cad0f`).
- Example presets: **cosplay_porcelain** — remove everything except
  `drawn_makeup_mark`+`mole:attenuate(30)` (porcelain looks often keep a
  signature mole faintly), min_confidence 0.5, body inherit;
  **natural_editorial** — preserve mole/freckle/scar/drawn, remove
  acne_blemish/stray_hair/sensor_dust only.
- NOTE: `retouch/recipes.py` is owned by a concurrent session — v1 presets
  ship as recipe JSON files via `recipe_loader.py`, not edits to recipes.py.

---

## 3. Facial harmony objective

Design principle: every term is a **ratio against the same photo's own
baseline** (before-image or within-subject region pair) — tone-invariant by
construction, and usable both as a QA gate (`qa_detectors.run_all` +
`QABackoff`) and as an auto-tuning target (pick the body/face strength that
minimizes drift subject to guards).

### 3.1 H1 — Texture Parity Ratio (primary; face/body seam guard)

For region R with mask M, luminance high band `h = gray − G_σ(gray)` with
σ = face_width/300 (physical detail scale tied to face size):

```
E(R)  = 1.4826 · median(|h − median(h)|)   over M          (robust std, MAD)
TPR   = E(body_skin) / E(face_skin)
D_TPR = | ln(TPR_after) − ln(TPR_before) |                  (harmony drift)
```

**Measured baselines (spike, 3 real portraits @1600px):**

| asset | TPR orig | TPR face-only smooth | TPR face+body smooth | D face-only | D consistent |
|---|---|---|---|---|---|
| DSCF4550 | 1.46 | 2.72 | 1.62 | 0.62 | 0.10 |
| DSCF7204 | 0.69 | 1.60 | 0.78 | 0.84 | 0.12 |
| DSCF7011 | 0.98 | 2.59 | 1.48 | 0.97 | 0.41 |

Face-only smoothing (the seam failure) drifts D ∈ [0.62, 0.97]; consistent
smoothing D ∈ [0.10, 0.41]. **Proposed gate: D_TPR ≤ 0.45** (flag), with the
auto-tune loop minimizing D_TPR by adjusting `body_smooth`. The 0.41 case
(DSCF7011) is honest evidence that "same radius" ≠ "same perceived smoothing"
— which is precisely why a closed-loop metric beats matched constants.
Note TPR-orig is not 1.0 (0.69–1.46): faces and bodies naturally differ, so
the objective preserves the *original* ratio (drift), never forces TPR=1.

### 3.2 H2 — Pore-band spectral parity (secondary, corroborating)

Patch-FFT (32px patches fully inside mask, Hann-windowed) pore-band fraction
(period 2–8 px, same band as `detect_pore_spectrum_distance`); parity ratio
PF_body/PF_face. Measured: orig 1.1–2.0; face-only smooth 6.4–15.3; consistent
2.0–6.5. Discriminates in the same direction but noisier than H1 (patch
availability varies with mask shape) — report, don't gate, in v1.

### 3.3 H3 — Specular-shape preservation (dimensionality guard)

Using `extract_specular` (already tone-safe): specular mean energy S_E and
relative-area S_A (fraction above half the region's own 95th percentile)
within face skin, after/before ratios. Measured under guided-filter smoothing:
energy ratio 0.91–1.30, area drift <10% — smoothing at these strengths keeps
speculars (matching the reference image's kept highlights). **Gate: energy
ratio ≥ 0.7 AND area drift ≤ 30%** — catches the "flat vinyl" failure where
highlight structure is blurred away.

### 3.4 H4 — Identity-mark retention score

`retention = |preserve-class marks re-detected within r=6px after processing| /
|preserve-class marks before|`. Measured: unprotected guided smoothing on
DSCF4550 erased the detected beauty mark (retention 0/1 = 0.0) while DSCF7204's
survived (1/1) — the metric catches over-retouch of identity marks exactly as
intended. **Gate: retention = 1.0** for `preserve`-policy classes (any loss is
a defect); for `attenuate`, require re-detection at the *lower* confidence
threshold (mark still present, softer).

### 3.5 H5 — Banding/posterization guard (must be differential)

Spike finding: absolute `detect_banding` flags ALL three untouched originals
(scores 0.85–0.92 — busy backgrounds read as banding). The harmony guard must
therefore be **Δscore = score_after − score_before**, ideally restricted to the
person mask. Measured Δ under smoothing: −0.02…0.0 (guided filter does not
posterize at these strengths). **Gate: Δ ≤ +0.03.** Porcelain recipes must
additionally run this on the float32 path (§1.5 uint8 round-trip is a live
banding source on long ramps — F1 dependency).

### 3.6 Composite

`harmony_ok = (D_TPR ≤ 0.45) ∧ (H3 gates) ∧ (H4 retention) ∧ (H5 Δ ≤ 0.03)`,
reported per-face with the raw numbers (never a single opaque scalar — per the
delta-blindness lesson, numbers accompany, never replace, visual QA renders).
As auto-tune target: minimize D_TPR over `body_smooth` (1-D search, ~4 engine
evals at proxy res) subject to the other gates — the cheapest useful version
of the VISION doc's "joint optimization."

### 3.7 Spike caveats (honesty)

- Spike body mask is a simplified builder (no person-mask intersect / geodesic
  contiguity): DSCF4550's includes some background skin-toned pixels.
  Production must reuse the engine's `body_skin_mask`. Contamination inflates
  E_body toward "unretouched," i.e. biases TPR *toward flagging* — safe
  direction, but fix anyway.
- Smoothing proxy was a guided filter, not the full face pipeline; Stage 5
  below re-baselines the gates against real `engine.process()` renders before
  they gate anything.
- 3 assets, all light-skinned cosplay subjects (the repo's known asset gap,
  P4 §19.3). Gates are provisional until Stage 5's broader sweep.

---

## 4. Body-skin parity plan (no separate body pipeline)

1. **Same policy layer, same detector, body mask:** run the unified mark
   detector on `body_skin_mask` with `on_body=True`, area thresholds in
   `area_norm` units (freckle.py's px² constants are face-at-500px-width
   tuned; normalize by face width so a chest mole and cheek mole classify
   identically). Mole/tattoo/drawn-mark preserve guards then apply to
   `body_smooth` exactly as `mole_mask` applies to face ops today. The OKLCh
   tattoo gate (`engine.py:3391`) already implements "preserve deliberate body
   ink" — fold it into the policy layer as the body analog of
   `drawn_makeup_mark` rather than a hardcoded gate.
2. **Texture parity via the harmony loop, not a body frequency-separation
   port:** keep the guided-filter body op, but drive its effective strength by
   minimizing D_TPR (§3.6). Where body ends up *over*-smoothed relative to
   face (TPR < before-ratio), `texture_transplant` re-seeds high band — it
   already exists and needs only a body-mask entry point.
3. **Face-anchored body mask (fairness fix):** replace `skin_mask_lch`'s fixed
   hue 25±25 / chroma≥8 with a locus sampled from *this subject's* parsed face
   skin (median hue/chroma ± scaled MAD in LCH), falling back to current
   constants when no face. Tone-adaptive by construction; also directly reuses
   P4 §20.3's neck-anchor idea (neck = label 14 is available).
4. **Seam-zone continuity check:** H1 evaluated specifically on the neck mask
   vs adjacent face-oval band (the reference image's giveaway zone) as a
   sub-metric — both masks already exist in `FaceRegions`.

---

## 5. Staged implementation plan (with QA gates and model routing)

| Stage | What | Model | QA gate |
|---|---|---|---|
| **S1** | ✅ **DONE 2026-07-17.** Fix freckle.py tone-invariance bugs (§1.2): L gate → margin-vs-skin-median; deviation → ratio; document as bug-class instance #5 in P4 plan §16 style | **Opus** (bug fix w/ root cause, per policy) | PASS: synthetic Fitzpatrick I–VI sweep classifies the same relative freckle consistently; relative beauty marks preserve; flat-skin controls remain empty; light-skin removal fixture is byte-identical |
| **S2** | `retouch/marks.py`: `MarkRecord`, unified detector (adapters over freckle/blemish_vs_mole/hairwork), policy engine compiling `mark_policy` → preserve/heal/attenuate masks; feature extraction incl. relative mel/hb, eccentricity, cluster density | **Opus** | No-policy path byte-identical (golden test); unit tests per class on synthetic marks; chromophore features asserted median-relative (no absolute constants — grep gate) |
| **S3** | params.py `ParamSpec` for `mark_policy` (+min_confidence), CLI flag, GUI `_process_input_components` entry, recipe key | **Haiku** (mechanical, follows documented pattern) | Import-time drift guard passes; `tests/test_gui.py::TestProcessInputKeys` |
| **S4** | `drawn_makeup_mark` detector: P4 Track-A high-passed cues + stroke geometry + midline symmetry; neck-anchored off-locus color evidence | **Opus** (judgment-heavy; escalate to Fable if the symmetry frame gets hairy) | Real-photo panels (DSCF set): drawn marks preserved under cosplay_porcelain; zero false-positive strokes on bare-skin renders; mottled synthetic gate per P4 §20.2(4) |
| **S5** | `retouch/harmony.py`: H1/H3/H4/H5 as `qa_detectors`-style functions; wire into `run_all` (reference-aware) + `QABackoff` mapping (D_TPR high → raise body_smooth, not lower face smooth — face is the signature); re-baseline gates on real `engine.process()` renders across ≥6 assets | **Opus** build, **Fable** if gate calibration disputes arise | Spike numbers reproduced through engine path; face-only-smooth render flagged, consistent render passes, on all assets; runtime budget ≤ ~60ms added (FFT patches at proxy res) |
| **S6** | Body parity: face-anchored body mask locus, policy layer on body mask, TPR-driven body_smooth auto-tune, texture_transplant body entry | **Opus** | H1 seam sub-metric (§4.4) within gate on porcelain renders; **mandatory visual montage review** (bonodori lesson — deltas hide face damage); tattoo/drawn-ink preserved |
| **S7** | Recipe presets `cosplay_porcelain.json`, `natural_editorial.json` via recipe_loader (NOT recipes.py — owned by concurrent session) | **Haiku** | Recipe round-trip tests; policy visible in `--recipe` CLI |

Order rationale: S1 before S2 (don't build policy on tone-biased classes); S5
before S6 (the metric must exist before it can drive body auto-tune); S3/S7
parallelizable Haiku work.

### 5.1 S1 execution record (2026-07-17)

Implemented in `retouch/freckle.py`:

- Replaced the fixed dark-spot `deviation > 7.0` candidate gate with
  `(local_mean - gray) / local_mean > 0.05`.
- Replaced the absolute `L=70` freckle/beauty-mark cue with a margin from the
  current face's median LAB-L. The normalization scale is the greater of the
  observed skin-L standard deviation and 10% of that median, preventing a
  nearly uniform crop from creating an artificial many-sigma outlier.
- Existing light-skin removal output is SHA-256 pinned. New I-VI fixtures
  cover freckle detection/classification, beauty-mark preservation, and plain
  skin with no candidate components.

Focused verification: `23 passed` in `tests/test_freckle.py`; the diff is
whitespace-clean. This is the prerequisite gate for S2. No policy-layer code
has been started.

### 5.2 S5 execution record (2026-07-17, continuation pass)

**Built.** `retouch/harmony.py` now implements the full metric set: H1 TPR +
drift, H3 specular parity + drift, **H4 mark retention** (preserve-class
beauty marks re-detected within r=6px vs the reference, face-bbox-cropped for
QA budget), and **H5 as differential banding** (`Δ = banding(processed) −
banding(reference)` per region — the synthetic test reproduced §3.5's finding
that absolute banding flags untouched content). Wired read-only into
`qa_detectors.run_all` (`flagged` stays False pending calibration).
Fitzpatrick I–VI synthetic tests assert D_TPR flags face-only and passes
matched smoothing at every tone, mask coverage does not collapse on V–VI, and
H4/H5 behave as designed. `tests/test_harmony.py`: 11 passed.

**QA body-mask bug found & fixed during re-baseline.**
`build_face_anchored_body_mask` used its 12-step geodesic walk as a *hard
per-pixel gate*, truncating the mask to a ~50px near-face ring — the metric
never saw the chest/arm skin the engine's body stage actually edits. Now the
walk gates *component connectivity* (mirroring the engine's own filter) and
an optional `hair_mask` guards against skin-chroma wigs (pink wig measured
contaminating the mask on DSCF7204).

**Engine-path re-baseline (S5 QA gate) — the provisional gates do NOT
transfer.** `scripts/spike_harmony_engine_baseline.py`, 6 assets (DSCF4550/
4463/4503/7011/7142/7204, 1600px), two regimes: bare `smooth=70` and
`cosplay_portrait_polish_v1` recipe strength, each face-only vs consistent:

| measurement | spike (guided-filter proxy) | real engine |
|---|---|---|
| D_TPR face-only | 0.62–0.97 | **0.03–0.19** |
| D_TPR consistent | 0.10–0.41 | 0.02–0.28 |
| discrimination at D_TPR≤0.45 | 3/3 | **0/6 (both regimes)** |
| body banding Δ | −0.02…0.0 | −0.009…+0.017 |
| H4 retention (recipe) | n/a | 1/1–15/18; **12/18 on DSCF4550** |

Root causes, verified by direct probes (not inferred):
1. The real face path preserves high band by design (`TEXTURE_ADAPT` floor,
   freq-sep `mid_reduction`) — visible smoothing lives in low/mid band. A
   mid-band TPR variant was probed and does **not** discriminate either
   (D_mid f/o ≈ con on all 4 probed assets).
2. The body ops barely move the measured statistic: their dominant effect is
   low-band tone (`match_face`), and on several assets they touch few or no
   pixels at all (next item).
3. **NEW ENGINE BUG (B-class, "runs but silently inert"):**
   `_stage_body_skin`'s exclusion stack (face convex hull + face-crop
   `acc_skin_hair` + full-image hair parse + morph open) can consume the
   entire 12-step contiguity corridor around the face; every candidate
   component then has 0 overlap with the geodesic walk and the **whole body
   stage no-ops with valid params** (proven live on DSCF4503 via frame
   introspection: walk=96k px, 53 components, all overlaps 0.0, final mask
   empty; `co−fo` max delta 0.0 anywhere). Fix direction: run the walk to
   convergence within the person silhouette (the 12-step cap is arbitrary)
   and/or compute component overlap against the pre-exclusion candidate.
   Related coverage failures: full-image hair parse eats 90% of body-skin
   candidates under a pink wig (DSCF7204) and misses white/silver wigs
   entirely (DSCF4463).

**Gate verdicts (calibrated):**
- **D_TPR ≤ 0.45 stays, re-scoped as a destructive-smoothing tripwire** —
  real renders sit ≤0.28, destructive guided-filter smoothing sits ≥0.62, so
  it can flag safely with zero false positives on this set — but it cannot
  detect recipe-level seams and must not be sold as a parity auto-tune
  objective until the body stage actually moves texture (see bug above).
- **H4 retention is the one gate with real signal at recipe strength today**
  (the recipe destroys 6/18 identity marks on DSCF4550) — candidate for the
  first `flagged=True` promotion, after mole-vs-freckle policy is settled
  (S2), since "marks" currently conflates preserve/remove classes.
- **H5 Δ ≤ +0.03 confirmed** — engine smoothing does not posterize
  (|Δ| ≤ 0.017 across 12 runs); gate kept.
- The **parity auto-tune loop (§3.6) is premature**: its objective gradient
  is ≈flat through the real engine. Blocked on the body-stage contiguity fix
  + stronger mid/high-band body ops (cosplay plan S2), and on assets with
  meaningful exposed body skin.

**Asset gap widened:** the known missing Fitzpatrick V–VI portraits, *plus*
the current cosplay QA set is mostly clothed (hands-only body skin on
DSCF4550; costume/sheer coverage on 7204) — parity work also needs 2–3
light-skin portraits with real exposed chest/arms/shoulders. Contact sheets:
`test_output/harmony_engine_baseline/`.

---

## 6. Risks & fairness (adversarial review of own proposal)

- **Classifier inherits freckle.py's biases if S1 slips.** Named the concrete
  bugs (§1.2); S1 is a hard prerequisite, not a nice-to-have.
- **Chromophore features are pseudo-physical.** S1 audit: WB swings mel 2–5×;
  a warm-lit photo can flip mel/hb votes. Mitigation: features are
  median-relative AND weighted as weak votes below geometry/size cues;
  `blemish_vs_mole` disagreement with the LAB classifier lowers confidence
  → `unknown` → preserve (fails safe).
- **Drawn-mark false negatives destroy styling silently.** A missed cheek line
  is removed with no error. Mitigation: preserve-by-default for `unknown`;
  H4 retention runs on drawn-class marks too; cosplay recipes surface the
  drawn-mark count in QA output so a 0 on an obviously-styled face is
  reviewable.
- **Drawn-mark false positives protect real defects.** Elongated inflamed
  scratch ≈ stroke. Acceptable failure direction (under-retouch beats
  destroyed styling); attenuate-not-remove policies soften the cost.
- **TPR can be gamed by adding noise, not texture.** Matching MAD energy with
  grain isn't matching pore structure. H2 (spectral shape) corroborates; the
  auto-tune loop only moves existing smoothing strengths, never injects grain.
- **Body-mask coverage disparity is a fairness issue upstream of everything.**
  If `skin_mask_lch` under-covers dark skin, both the policy layer and H1
  silently under-apply to exactly the subjects the tone rules protect. §4.3's
  face-anchored locus is the fix; S6's QA must include a body-mask coverage
  check across a tone sweep (synthetic if real dark-skin assets remain
  unavailable — same asset gap P4 §19.3 documents; flag to owner: acquiring
  2–3 Fitzpatrick V–VI portraits with visible body skin unblocks honest
  validation for BOTH this plan and P4).
- **Harmony metrics are within-subject ratios — tone-fair by construction,
  but the *detectors feeding them* need masks that are tone-fair** (previous
  point). No H-term uses an absolute intensity threshold; H3 rides
  `extract_specular`'s already-verified margin-above-baseline gate.
- **Banding guard depends on F1 float32 completion** for porcelain looks
  (§1.5 body uint8 round-trip). Not blocked — the guard works today — but
  gate failures on long ramps may root-cause to F1, not this feature.

## 7. NO-GO decisions

- **NO-GO: broad foundation/coverage detection as a classifier feature** —
  structurally impossible with per-face outlier stats (P4 §19.1/§20.1);
  anything needing it waits for P4 Track B's neck-anchored locus to ship.
- **NO-GO: automatic scar detection in v1** — no reliable classical signature
  distinguishing scars from folds/highlights without data we don't have; the
  schema keeps the `scar` slot (user brush + policy applies), auto-detection
  deferred.
- **NO-GO: learned aesthetic/harmony scoring** — data-acquisition-gated, same
  reasoning as parked A1/A2 track; classical ratio metrics above are
  sufficient for the QA/auto-tune loop.
- **NO-GO: forcing TPR → 1.0** — faces and bodies legitimately differ in
  texture; the objective preserves the original ratio (drift minimization),
  matching how the reference retoucher kept relative, not absolute, texture.

## Appendix — reproduce

```
python3 scripts/spike_harmony_metrics.py          # defaults: DSCF4550/7204/7011
# prints H1/H2/H3/H5 tables + mark-retention; writes
# test_output/spike_harmony_<name>.png (image | face mask | body mask | high band ×8)
```
