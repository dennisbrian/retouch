# RESEARCH — Cosplay Portrait Polish: capability audit, reference target analysis, ROI ranking

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only; no production code changed)
**Primary plan:** `PLAN_COSPLAY_PORTRAIT_POLISH.md`
**Supporting:** `PLAN_BLEMISH_POLICY_HARMONY.md`, `PLAN_P4_MAKEUP_UNMIX.md`,
`PLAN_S1_CHROMOPHORE_AUDIT.md`, `PLAN_FEATURE_FRONTIER.md`
**Reference (visual target only):**
`~/Downloads/Screenshot_20260717_001633_com_twitter_android_MainActivity.jpg`
(viewed 2026-07-17 at full resolution).

**Constraint restated:** the reference is a generative/heavily-idealized image.
This report does **not** treat its idealized skin, anatomy, or lens behaviour as
a classically reproducible bar. It is read only as a direction of intent.

---

## 0. TL;DR / verdict

**Recommended next implementation:** build the **facial-harmony measurement layer
first** — `retouch/harmony.py` with H1 Texture Parity Ratio (+ H3 specular, H4
mark-retention, H5 differential banding) as `qa_detectors`-style functions wired
into `run_all` + `QABackoff`, **paired with the face-anchored body-skin mask
locus** (`PLAN_BLEMISH_POLICY_HARMONY.md` §4.3). This is *not* the plan's stated
"highest ROI" line item (face-body parity) — it is that item's hard prerequisite,
and it is the highest-ROI thing that can be built **safely and measurably right
now** without depending on unbuilt modules.

Rationale in one sentence: the plans themselves say the parity feature must be
*tuned against a metric*, that metric does not exist yet, and building the parity
op first means tuning against deltas — which this repo's own hard-won lesson
(delta-blindness / bonodori) says fails.

**Go/no-go summary (detail in §7):**

| Proposed algorithm | Verdict | Gating condition |
|---|---|---|
| S0 baseline suite + no-regression harness | **GO** | none — pure scaffolding |
| S1 `PortraitRegions` masks | **GO (partial)** | depends on marks S2 for preserve masks; body-mask locus fairness fix required |
| **Harmony metric layer (H1/H3/H4/H5) + body-mask locus** | **GO — recommended next** | none; spiked and measured |
| S2 face-body texture parity (three-band body op) | **CONDITIONAL GO** | needs harmony metric first; **dark-skin validation blocked on assets** |
| S3 light-aware porcelain finish | **GO (v1.1)** | `lighting.py` confidence gate; consumer to be built |
| S4 feature focal pass (eyes/lips/under-eye) | **GO (low urgency)** | existing enhancers; coordinate, don't rebuild |
| S5 palette-aware editorial grade | **GO (v1.1)** | skin-ΔE isolation gate |
| S6 recipe/controls | **GO (last)** | after S0–S5 visual pass |
| P4 full-face foundation unmix | **NO-GO for default recipes** | parked: light-skin-only, specular-unsafe (§P4 audit) |
| Generative pore/anatomy/catchlight synthesis | **NO-GO (classical)** | requires a learned model — see §6 |

---

## 1. Reference image — measured visual traits (target only)

Read directly. A pink-wig editorial cosplay portrait, three-quarter reclining
pose, cool patterned-gray backdrop and black leather furniture.

| Trait | What the reference shows | Classical reproducibility |
|---|---|---|
| **Face/body texture parity** | Face, chest, shoulders, arms, hands, and thighs all sit at the *same* near-poreless finish. No texture seam at the neck or décolletage. This is the single defining trait. | **Partial.** Texture *statistics* can be matched (H1/H2 objective); the reference's absolute poreless-ness is beyond a safe classical floor without going plastic. |
| **Highlight direction / specular** | Soft key from upper-frontal camera-left; controlled speculars kept on nose bridge, cheekbones, collarbone, shoulder, and lower lip — not erased, not blown. | **Yes, conservatively.** `extract_specular` (tone-safe) + `lighting.py` estimate support "attenuate excess, protect ridges." |
| **Skin tone / shadow palette** | Warm-neutral skin held near a natural locus; cool blue-gray in the backdrop and cast shadows; restrained pink support pulled from the wig — *not* a global magenta wash. | **Yes.** Constrained split-tone with skin held near its corrected locus (S5). |
| **Eye/lip focal hierarchy** | Eyes and lips read slightly crisper than surrounding skin: defined lashes, catchlights *present in source*, satin (not wet-glass) lip with a real specular crescent. | **Yes, but only enhancement.** Catchlights are present in the source, so no synthesis needed here; lip satin rides existing specular evidence. |
| **Wig/costume preservation** | Individual pink wig fibres, sheer white lace with visible weave, ribbon, rings, and the horn prop all crisp — none smoothed or recoloured. | **Yes — this is a masking discipline, not an algorithm.** The hard requirement is that no skin-smooth/bloom/grade mask ever touches wig/lace/jewelry pixels. |
| **Background separation** | Subject clearly lifted from backdrop by depth-of-field and a soft vignette; no hard cutout halo. | **Yes.** Existing `blur_background` / `lens_blur` / vignette, subject-mask gated. |

**Honest caveat:** the reference's skin has generative-image evenness (no visible
pore structure at all in places) and idealized anatomy/lens bokeh. A classical
engine must aim for the *credible photographic equivalent* — luminous, even,
dimensional, seamed-free — not this exact finish. Aiming at the literal reference
produces plastic skin.

---

## 2. Capability audit — what exists, verified in code

Verification performed by direct read + grep on 2026-07-17.

### 2.1 Skin, frequency, texture, specular

- **Face path is rich and frequency-based.** `frequency.FrequencySeparator` +
  `_texture_adaptation_factor` (a high-band protection floor), `skin.py`
  chromophore/albedo/blotch ops, `specular.py` (tone-safe after the §15 fix).
- **`skin.texture_transplant`** (`skin.py:1926`) — same-face high-band donor
  cloning — **exists and is face-wired only.** `perf_optimizations.py:879`
  calls it with `regions.skin` + `face_width`; **no body-mask entry point.**
  This is the natural "re-texture over-smoothed body zones" tool for parity.
- **`specular.extract_specular`** is tone-safe (margin-above-baseline gate,
  §15 of P4 plan) and already consumed by the harmony spike's H3.

### 2.2 Face lighting estimate — built, **zero image consumers**

- `retouch/lighting.py::estimate_light_direction` returns a confidence-rated
  `LightDirection` (paired-catchlight preferred; low-freq shading plane
  fallback; `unknown` when weak). Verified.
- Cached on every `FaceContext` (`engine.py:2836`,
  `detection.py:59`). Verified.
- **No consumer edits the image.** `grep` for `light_direction` outside
  `lighting.py` returns only the dataclass field + the cache-write site. This
  is a fully-built, unwired capability. (`PLAN_FEATURE_FRONTIER.md`
  E-COMMON-1 confirms: "no image-editing consumer yet.")

### 2.3 Masks / parser

- `FaceRegions` (`parsing.py:205`, `__slots__`) exposes: `face_oval, skin,
  forehead, cheeks, nose, eyes, eyebrows, iris, sclera, lips, mouth_interior,
  under_eye, nose_bridge, cheek_highlights, jawline_contour, hair, neck,
  nasolabial, crows_feet, cloth`. **`neck` is present** (BiSeNet label 14,
  `parsing.py:65`).
- **No body / chest / arm / hand masks in `FaceRegions`.** Body skin is built
  ad-hoc inside the engine's body stage (below).
- **No `PortraitRegions` value object** (`grep` empty). S1 of the cosplay plan
  is unbuilt.

### 2.4 Body-skin stage — reachable, single guided filter, texture-blind

`engine.py:3224 _stage_body_skin`, verified in full:

- **Mask:** `skin_mask_lch(hue_center=25, hue_tolerance=25, chroma_min=8)` ∩
  person mask, minus face-hull / hair (`parse_hair_full_image`) / lips,
  geodesic face-contiguity walk (12 steps), OKLCh `C<0.18` tattoo gate,
  morph open/close, feather. **Solid engineering, but the hue/chroma
  constants are absolute** → coverage on deep or strongly-lit skin tones is
  unaudited. This is the same fairness bug class fixed five times elsewhere
  (P4 §14–16).
- **Smoothing = one guided filter** (`engine.py:~3404`), blended via
  `blend_masked` which **round-trips through uint8 inside the float stage**
  (banding-relevant on long chest/thigh ramps). No frequency separation, no
  `_texture_adaptation_factor` floor, no mark classification, no
  `texture_transplant`.
- **`body_match_face`** matches **median LAB tone only** (clamped ±8 L, ±6
  a/b). **Nothing matches texture.** This is precisely the mechanism that
  produces the face/body texture seam the reference avoids.

### 2.5 Per-mark classification — fragmented, no unified policy

- `freckle.FreckleRemover` (LAB + size heuristics; **S1 tone-fixes DONE
  2026-07-17**, P4 §16.4), `blemish.BlemishRemover`, `hairwork` flyaway,
  `skin_chromophore.blemish_vs_mole`. Each feeds its own op/slider.
- **No `MarkRecord`, no `marks.py`, no unified policy** (`grep` empty). The
  cosplay plan's S1 preserve masks and the blemish plan's S2 are unbuilt.
- **No `drawn_makeup_mark` class** — the reference's drawn under-eye cheek
  lines would be destroyed by any strong smoothing. This is the styling-
  preservation gap.

### 2.6 Chromophore features — within-face-relative only

`PLAN_S1_CHROMOPHORE_AUDIT.md`: `decompose_chromophores` is two fixed log-ratios
on gamma-encoded sRGB; WB leaks 2–5× into melanin; mel/hb entangled. **Usable
only as margin-above-this-face's-own-median**, never absolute, never cross-photo.
Constrains any mark/makeup feature to relative cues.

### 2.7 QA scaffolding — extendable, no face-vs-body term

`qa_detectors.py` has `detect_banding, clipping, plastic_skin, halo, seam,
color_drift, pore_spectrum_distance, over_retouch_asymmetry`, all wired into
`run_all` + `QABackoff`. **All are face-global or image-global; none compares
face vs body.** A new harmony detector added to `run_all` becomes both a QA gate
and a backoff auto-tune signal for free — the cheapest credible integration point.

### 2.8 Harmony metric — **spiked, measured, not yet a module**

`scripts/spike_harmony_metrics.py` exists and is re-runnable. It measures H1 TPR,
H2 pore-spectrum, H3 specular-shape, H4 mark-retention, H5 banding on real
portraits. Measured discrimination (3 light-skin assets, plan §3.1): face-only
smoothing drifts D_TPR ∈ [0.62, 0.97]; consistent smoothing D ∈ [0.10, 0.41].
**`retouch/harmony.py` does not exist** (`grep` empty) — the spike has not been
promoted to a module or wired into `run_all`.

### 2.9 Recipes

- `recipes.py` has `cosplay`, `cosplay_3d`, `cosplay_no_eq`,
  `porcelain_unified_v1`, `cosplay_sculpt_v1`, `game_character_v2` (SSS).
- **No `cosplay_porcelain` / `cosplay_editorial` / `natural_editorial`** JSON
  recipe (`find` empty). S6 unbuilt.
- `recipe_loader.py` is the correct home for new presets (`recipes.py` is owned
  by a concurrent session per both plans).

---

## 3. Stage-by-stage bucket classification (task item 2)

Every proposed stage sorted into: **(A) already implemented, merely unwired**;
**(B) implementable with classical image processing**; **(C) requires a stronger
prior/model**; **(D) unsafe or unlikely to deliver value.**

| Stage | Bucket | Evidence |
|---|---|---|
| **S0** baseline suite + harness | **B** | Pure scaffolding; `qa_detectors` + `test_output/` conventions exist. |
| **S1** `PortraitRegions` masks | **B**, with an **A** core | `neck`/`face_oval`/eye/lip masks exist (A); the *body* mask + preserve-mask assembly is new classical work (B). Depends on marks S2 for preserve masks. |
| **Harmony metric layer** | **A/B** | H1/H3/H4/H5 spiked (A-ish: code exists in a spike); promotion to `harmony.py` + `run_all` wiring is small classical work (B). |
| **`lighting.py` estimate** | **A (unwired)** | Fully built, confidence-rated, cached, zero consumers. |
| **`texture_transplant` on body** | **A (unwired)** | Exists face-only; needs a body-mask entry point. |
| **S2** face-body texture parity | **B** | Three-band body op + seam feather + `texture_transplant` re-seed; all classical. Blocked on harmony metric (to tune) + fairness mask fix. |
| **S3** light-aware porcelain finish | **B** | Specular attenuation, dodge/burn, gated bloom — all classical; `lighting.py` supplies the gate (A). |
| **S4** feature focal pass | **A** | `eyes.enhance`, `lips.enhance`, `undereye.process`/`attenuate_hemoglobin` all exist; coordinate under one capped strength. |
| **S5** palette-aware editorial grade | **B** | Constrained split-tone with skin-locus hold; `grading.py`/`background` vignette+blur exist. |
| **S6** recipe/controls | **B** | `recipe_loader.py` + `params.py` single-source pattern. |
| P4 full-face foundation unmix | **C, parked → D for default** | §P4 audit: solver is light-skin-only, specular-unsafe; no recipe enables it. Full foundation eveness needs Track B's neck-anchored locus (unbuilt). |
| Generative pore/anatomy/catchlight | **C** | Not classical — see §6. |

**Observation:** none of S0–S6 themselves require a learned model. The entire
"needs a stronger model" answer lives in the non-goals (§6). The two cheapest
wins hiding in plain sight are the two **A (unwired)** items: `lighting.py` and
`texture_transplant`-on-body.

---

## 4. Top-3 highest-ROI gaps — concrete algorithms

Ranked by the discriminating test: *which build makes measurable, safe progress
toward the reference's defining trait (face-body parity) without depending on
something unbuilt?*

### GAP 1 — Facial-harmony measurement layer + face-anchored body-mask locus ⭐ recommended

**Why #1:** parity is the reference's giveaway trait, but the parity *operation*
cannot be tuned safely until the seam is measurable. The metric is already
spiked; promoting it is the lowest-cost, lowest-render-risk, highest-leverage
build. It converts the eventual body rewrite from "tune against deltas" (known to
fail here) into a closed-loop objective.

- **Existing modules / integration point:** new `retouch/harmony.py` (mirror
  `qa_detectors.py` function style); wire H1/H3/H4/H5 into
  `qa_detectors.run_all` so `QABackoff` sees them. Body-mask locus fix goes in
  the `skin_mask_lch` call inside `engine.py:_stage_body_skin` (add a
  face-anchored variant; fall back to current constants when no face).
- **Signal / mask inputs:** face-skin mask (`regions.skin`), body-skin mask
  (engine body stage), `neck` + adjacent face-oval band (seam sub-metric),
  `extract_specular` (H3), `FreckleRemover.classify_anomalies` (H4). All
  within-subject ratios — tone-fair by construction.
- **Failure modes / fairness risks:** (1) TPR can be gamed by adding grain not
  structure → H2 pore-spectrum corroborates; the auto-tune loop only moves
  existing smoothing strengths, never injects grain. (2) **The body mask
  feeding H1 must be tone-fair** — the fixed hue 25±25 / chroma≥8 under-covers
  dark skin, silently under-applying the metric to exactly the subjects the
  tone rules protect. The face-anchored locus (median hue/chroma ± scaled MAD
  from parsed face skin) is the fix and is part of this gap, not a follow-up.
- **QA design:** *synthetic* — Fitzpatrick I–VI swatch pairs (face vs body)
  with matched vs mismatched smoothing; assert D_TPR flags mismatch and passes
  matched at every tone; assert body-mask coverage does not collapse on
  V–VI. *Real* — reproduce the spike's face-only-flag / consistent-pass result
  through the real `engine.process()` path on ≥6 assets; mandatory visual
  montage (deltas hide face damage).
- **Expected visible benefit:** none directly (it is measurement) — but it is
  the enabler that lets every subsequent skin/body strength be chosen instead
  of guessed. Indirect but foundational.
- **Difficulty:** **Low.** Spike code exists; ~1 module + `run_all` wiring +
  the mask-locus change. The blemish plan routes this to Opus (build) / Fable
  (gate-calibration disputes).

### GAP 2 — Face-body texture & tonal parity (three-band body op)

**Why #2:** highest *outcome* value (the seam is the reference's signature),
but strictly downstream of GAP 1 — it needs the metric to tune against.

- **Existing modules / integration point:** replace the single guided filter in
  `engine.py:_stage_body_skin` with a three-band policy mirroring the face
  finish; re-seed high band via a **new body entry point to
  `skin.texture_transplant`** (pass the body mask + a body-scale `face_width`
  proxy). Fix the `blend_masked` uint8 round-trip (float32 path) to kill
  banding on long ramps.
- **Signal / mask inputs:** face-anchored body-skin mask (GAP 1), face-skin
  texture statistics as the target, `neck` seam band for feathered blending.
- **Failure modes / fairness risks:** over-smoothing the body flat (H3/H1
  guard); copying facial pores literally onto the body (donor is *body's own*
  clean skin, not face); **dark-skin parity cannot be visually validated —
  every asset including the reference is light-skinned.** Any GO is conditional
  on acquiring 2–3 Fitzpatrick V–VI portraits with visible body skin (owner
  action).
- **QA design:** *synthetic* — banding on smooth chest/arm/thigh ramps at
  float32; texture-parity on matched swatch pairs. *Real* — H1 seam sub-metric
  within gate on porcelain renders + mandatory visual montage; tattoo/drawn-ink
  preserved.
- **Expected visible benefit:** **High and directly visible** — eliminates the
  smooth-face / untreated-neck-chest-arm seam, the top failure in polished
  cosplay portraits.
- **Difficulty:** **Medium-high.** New three-band body op + `texture_transplant`
  body wiring + float32 completion + the auto-tune loop.

### GAP 3 — Light-aware porcelain finish (consumer for `lighting.py`)

**Why #3:** cheapest way to turn an already-built, zero-consumer capability
(`lighting.py`) into visible dimensional quality, and it directly serves the
reference's "kept, directional speculars" trait.

- **Existing modules / integration point:** new conservative `portrait_finish`
  pass (in the proposed `portrait_polish.py` orchestrator, after parse / before
  global grade). Consumes the cached `FaceContext.light_direction`;
  `extract_specular` for the specular map; `skin.py` dodge/burn for form.
- **Signal / mask inputs:** `light_direction` (must honor `confidence`; skip on
  `unknown`), `extract_specular` map, face-skin mask, highlight-ridge masks
  (nose_bridge, cheek_highlights) from `FaceRegions`.
- **Failure modes / fairness risks:** flattening real dimensionality (protect
  ridges matching the inferred key); low-confidence light estimate → **must
  skip the stage, output byte-identical** (the estimate reports `unknown`
  honestly — honor it). No synthetic catchlights, no global whitening.
- **QA design:** *synthetic* — flat-lit swatch (estimate = unknown) must return
  byte-identical; a directionally-lit synthetic must keep its highlight
  direction unchanged. *Real* — highlight clipping does not increase; skin ΔE
  bounded; visual montage.
- **Expected visible benefit:** **Medium.** "Smooth but dimensional" instead of
  "smooth and flat" — attenuate excess hotspots, restore form after albedo
  evening. Lower urgency than parity because a flat-but-seamless face still
  reads as retouched; a seamed one reads as amateur.
- **Difficulty:** **Medium.** New consumer + conservative gating; the hard part
  (the estimate) is already built and tested.

---

## 5. ROI ranking of the next implementation task (task item 5)

**Do not** follow the plan's literal build order (which puts marks-S2 first) as
gospel, nor its "highest immediate ROI = parity" headline as the *next task*.
The evidence reorders as follows:

1. **Harmony metric layer + body-mask locus (GAP 1)** — recommended next.
   Prerequisite for tuning parity; spiked; near-zero render risk; fixes a live
   fairness bug in the body mask as a side effect.
2. **Face-body texture parity (GAP 2)** — the actual outcome the owner wants,
   but blind without #1. Build immediately after.
3. **Light-aware finish (GAP 3)** — turns a free, already-built capability into
   dimensionality; independent of #1/#2.
4. Marks S2 / `MarkRecord` (preserve-mask backbone) — needed before S2 parity
   can honor drawn-makeup/mole preservation on the body, but its S1 tone-fix
   prerequisite is already DONE, so it can proceed in parallel with GAP 1.
5. S4 feature focal, S5 grade, S6 recipes — after the base skin finish passes
   visual QA.

**Why this beats "parity first":** the blemish/harmony plan states it outright
("the metric must exist before it can drive body auto-tune", §5 order rationale),
and this repo's memory (`bonodori-qa-delta-blindness`) records that global deltas
hide face damage — so tuning a body op without a face-vs-body metric repeats a
known failure. Building the metric first is a small, safe, high-leverage move
that de-risks the expensive one.

---

## 6. What must NOT be built with classical retouching (task item 6)

These require generative editing or a learned model; classical processing will
either fail or produce tells:

- **Generative pore/skin synthesis to the reference's absolute evenness.** The
  reference has generative-image poreless-ness in places. Classically reaching
  it means over-smoothing → plastic. Classical target is *statistical texture
  parity*, not literal poreless-ness. (Non-goal §7 of the plan.)
- **Full-face foundation coverage evening / makeup unmixing (P4).** §P4 audit:
  the shipped solver is **light-skin-only and specular-unsafe**; foundation-
  present detection via per-face outlier stats is *structurally impossible*
  (coverage cliff at 44–60%). Full-face evening needs Track B's neck-anchored
  locus (unbuilt) or a learned paint segmenter. **P4 stays parked; no default
  recipe may enable it.** Compact drawn-mark *preservation* (Track A, shipped)
  is fine; full-face *unmixing* is not.
- **Catchlight / eye-geometry synthesis.** The reference's catchlights are
  present in-source, so not needed here — but *synthesizing* them where absent
  (E-EYE-1) or recentering irises (E-EYE-5) is a learned/geometry-model risk,
  off-scope for this pipeline (the plan's non-goal: "no fake catchlights, no
  geometry changes").
- **Anatomy / body-shape / clothing-shape idealization.** The reference's
  proportions are idealized. Reshaping is an explicit non-goal; geometry tools
  stay separate creative controls.
- **A learned aesthetic/harmony scorer.** Data-acquisition-gated; the classical
  within-subject ratio metrics (GAP 1) are sufficient for QA/auto-tune. (Blemish
  plan §7 NO-GO.)

---

## 7. Go/no-go verdict per proposed algorithm

- **S0 baseline harness — GO.** Pure scaffolding, no render risk. Build first
  alongside GAP 1.
- **S1 `PortraitRegions` — GO (partial).** Face/neck/eye/lip masks are an A-core;
  the body mask must use the face-anchored locus (fairness), and preserve masks
  depend on marks S2. Ship the geometry/eye/lip/neck masks now; gate the
  preserve masks on marks S2.
- **Harmony metric layer + body-mask locus (GAP 1) — GO. Recommended next.** No
  blocker; spiked and measured. Fairness note: re-baseline gates on real engine
  renders across ≥6 assets before they gate anything.
- **S2 face-body parity (GAP 2) — CONDITIONAL GO.** GO on the algorithm; the
  **dark-skin fairness verdict is conditional on acquiring Fitzpatrick V–VI
  assets with visible body skin** — a hard owner action, not a footnote. Until
  then, validated for light skin only and must say so. Blocked on GAP 1 for
  tuning.
- **S3 light-aware finish (GAP 3) — GO (v1.1).** GO conditioned on honoring the
  `lighting.py` confidence gate (skip on `unknown` → byte-identical).
- **S4 feature focal — GO (low urgency).** Coordinate existing enhancers; do not
  rebuild. No catchlight synthesis in v1.
- **S5 palette grade — GO (v1.1).** Conditioned on an independent skin-ΔE
  isolation gate so the scene grade never contaminates skin or neutral whites.
- **S6 recipe/controls — GO (last).** Via `recipe_loader.py`, after S0–S5 visual
  pass; no default recipe enables P4.
- **P4 full-face unmix — NO-GO for default recipes.** Parked (§P4 audit).
- **Generative pore/anatomy/catchlight synthesis — NO-GO (classical).** Requires
  a learned model (§6).

---

## 8. Required visual-QA assets

- **Have (light-skin cosplay, repo):** `DSCF4550`, `DSCF4454`, `DSCF7204`,
  `DSCF7011` (harmony spike defaults); the owner's reference (visual target,
  not a sole gate).
- **Have (stress):** `DSCF6102` (equalize pale-face regression asset),
  `DSCF7142` (nose-bridge shading regression).
- **MISSING — blocking for a fair parity verdict:** 2–3 **Fitzpatrick V–VI
  portraits with visible body skin** (neck/chest/arms). This gap blocks honest
  dark-skin validation for BOTH this plan and P4 (§P4 §19.3). **Flag to owner:
  acquiring these is the single highest-value QA unblock.**
- **Synthetic (buildable now, no assets needed):** Fitzpatrick I–VI face/body
  swatch pairs for H1 tone-fairness; long smooth ramps (chest/arm/thigh) for
  H5 banding at float32; matched vs mismatched smoothing pairs for D_TPR
  discrimination.
- **Contact-sheet requirement (all stages):** original, polished, 4× delta,
  face/body masks, preserve masks, and a manifest of active stages. A small
  delta is never evidence of safety — visual montage is mandatory
  (delta-blindness lesson).

---

## 9. Recommended next implementation (restated for the record)

**Build `retouch/harmony.py` (H1 Texture Parity Ratio + H3 specular-shape + H4
mark-retention + H5 differential banding) as `qa_detectors`-style functions,
wire them into `qa_detectors.run_all` + `QABackoff`, and add the face-anchored
body-skin mask locus** (`skin_mask_lch` → face-median hue/chroma ± scaled MAD,
with the current constants as the no-face fallback).

Deliver with: (a) synthetic Fitzpatrick I–VI face/body swatch tests asserting
D_TPR flags mismatch and passes matched at every tone, and body-mask coverage
does not collapse on V–VI; (b) reproduction of the spike's face-only-flag /
consistent-pass result through the real `engine.process()` path on ≥6 assets;
(c) a visual montage. This unblocks GAP 2 (parity) by making it tunable instead
of blind, and fixes a live fairness bug in the body mask as a side effect.

### Alternatives rejected and why

- **Parity op first (plan's literal "highest ROI").** Rejected as the *next*
  task: it would be tuned against deltas with no face-vs-body metric — a known
  failure mode here (delta-blindness). It remains the #2 build, immediately
  after the metric. This is a sequencing correction, not a disagreement about
  its value.
- **Marks S2 / `MarkRecord` first (plan's §5 order).** Its blocking prerequisite
  (freckle tone-fixes, S1) is already DONE, so it can run in parallel — but it
  does not itself move the seam, the reference's defining trait, so it is not
  the highest-ROI *single* next task. Parallelizable, not first.
- **Light-aware finish (GAP 3) first.** Rejected as #1: a flat-but-seamless face
  still reads as professionally retouched; a seamed one reads as amateur. Depth
  is a smaller lever than the seam, so it ranks #3 despite being cheap (its
  dependency, `lighting.py`, is already built).
- **P4 makeup unmix.** Rejected: parked, light-skin-only, specular-unsafe; not a
  default-recipe candidate.

### Go/no-go verdict — final

**GO** on the harmony-metric + body-mask-locus build as the next implementation
(§7, §9). **CONDITIONAL GO** on the parity op that follows it, with the dark-skin
verdict explicitly gated on owner-supplied Fitzpatrick V–VI body-skin assets.
**NO-GO** on P4 full-face unmixing for any default recipe and on all generative/
learned-model items (§6), which are out of scope for classical retouching.

---

## 10. Addendum — engine-path re-baseline outcome (same day, continuation pass)

GAP 1 was built and then re-baselined through real `engine.process()` renders
(6 assets × 2 regimes; full record in `PLAN_BLEMISH_POLICY_HARMONY.md` §5.2).
Three corrections to this report's claims:

1. **§2.8's spike numbers do not transfer to the engine.** Face-only
   smoothing through the real pipeline drifts D_TPR only 0.03–0.19 (spike
   proxy: 0.62–0.97) because the face path deliberately preserves the high
   band and the body ops mostly move low-band tone. D_TPR≤0.45 is re-scoped
   as a destructive-smoothing tripwire, not a seam detector; H4 mark
   retention is the metric with real recipe-strength signal today.
2. **GAP 2's auto-tune premise is weaker than stated:** the objective's
   gradient is ≈flat through the current engine, and a new B-class bug was
   root-caused — `_stage_body_skin` can silently no-op entirely when its
   exclusion stack severs the geodesic contiguity corridor (proven on
   DSCF4503). The parity op is now *also* blocked on that engine fix, not
   just on the metric.
3. **§8's asset gap is wider:** besides Fitzpatrick V–VI, the repo's cosplay
   QA set has almost no exposed body skin (hands only on DSCF4550; costume/
   sheer coverage elsewhere). Parity validation needs portraits with real
   chest/arm/shoulder skin at any tone.

## Appendix — reproduce the evidence

```bash
python3 scripts/spike_harmony_metrics.py          # H1/H3/H4/H5 tables + panels
python3 scripts/spike_harmony_engine_baseline.py  # real-engine re-baseline (§10)
python3 scripts/spike_s1_chromophore_audit.py     # chromophore fairness audit
grep -rn "light_direction" retouch/                # confirm lighting.py unwired
grep -rln "MarkRecord\|PortraitRegions\|harmony.py" retouch/   # confirm unbuilt
```
