# PLAN C3 — Soft BiSeNet Logits / Guided-Feather Masks

**Date:** 2026-07-16 · **Status:** Research complete, decision-ready
**Spike:** `scripts/spike_c3_soft_masks.py` (re-runnable; ~30s, 3 images at 1600px)
**Evidence renders:** `test_output/spike_c3_{DSCF8007,DSCF7204,DSCF4550}_{hairline,lips}.png`
(panels: image | current binary+Gaussian | raw softmax | guided filter | softmax+guided;
top row = mask grayscale, bottom row = image×mask halo test)

## Verdict

**Raw softmax probabilities as masks: NO-GO.**
**Guided-filter feathering of the existing argmax masks: GO** (staged, behind a param,
default byte-identical). It needs no logits, no cache change, and visibly resolves
hair strands / lashes that the current Gaussian feather smears.

---

## 1. What the logits actually look like (Q1)

Measured on the real `models/resnet18.onnx` via the exact `parsing.py` preprocessing
(30%-padded bbox → 512×512 → ImageNet norm), largest face per image, images at 1600px:

| Image | median maxprob (whole crop) | frac maxprob<0.9 | hairline band | jawline band | lip border band | eye contour band |
|---|---|---|---|---|---|---|
| DSCF8007 | 0.732 | **98.5%** | 18.8 img px | 18.1 | 13.1 | 12.9 |
| DSCF7204 | 0.716 | **99.4%** | 24.5 img px | 24.4 | 21.1 | 20.1 |
| DSCF4550 | 0.739 | **97.6%** | 20.8 img px | 20.1 | 15.3 | 14.0 |

("band" = uncertainty-band width, pixels with maxprob<0.9 per unit boundary length,
in image pixels at 1600px. Current Gaussian feather: r=7–8 for skin/hair, r=3–4 for
lips/eyes — i.e. a ±7px transition.)

Key findings:

- **The model is globally under-confident, not edge-localized.** Median maxprob is
  ~0.72–0.74 *everywhere*, including flat face interiors; 97–99% of all pixels are
  below 0.9. maxprob<0.9 does not mean "near an edge" for this export.
- **Probability transitions are wide and blurry.** A scanline through the hairline
  shows skin-prob ramping 0.29→0.65 gradually over ~17px at 512-res. This is the
  signature of BiSeNet's 1/8-resolution feature maps bilinearly upsampled — the
  softmax encodes *feature-map blur*, not sub-pixel edge position.
- The uncertainty band (13–25 img px) is **2–3× wider than the current Gaussian
  feather** (±7px). Soft masks would make edges *softer/worse-localized* than today,
  not tighter.
- At-boundary median maxprob is ~0.43–0.56 on every boundary type — plausible as a
  50/50 split point, but the interior never saturates, which is fatal for direct use
  (see below).

**Answer: the model is NOT calibrated/localized enough at edges for softmax masks to
beat blur-of-binary.** The image itself carries the edge information the 512-res
softmax lacks — which is exactly what a guided filter exploits.

## 2. Visual comparison (Q2) — look at the renders

Four variants per region, all built at crop resolution:
(a) **current** = argmax binary → NEAREST upsample → `feather_mask` Gaussian,
(b) **soft** = per-class softmax prob → bilinear upsample,
(c) **guided** = binary → bilinear → `cv2.ximgproc.guidedFilter(guide=crop, r=feather, eps=1e-3)`,
(d) **soft+guided** = (b) refined by guided filter.

Judgments from the rendered panels (all three images agree):

- **soft (b) is unusable as a drop-in mask.** Interior skin/lip probability plateaus
  at ~0.6–0.8 instead of 1.0 → the image×mask composite is visibly dimmed mid-face,
  which would *attenuate every op* (smoothing, tint) in the mask interior, not just
  soften edges. Gray probability haze leaks over hair and background, and DSCF8007
  shows a low-confidence vertical band across the nose. Blobby 1/8-feature structure
  is obvious.
- **guided (c) is the clear winner at hair/skin boundaries.** In DSCF8007 it resolves
  individual white fringe strands; in DSCF7204 it snaps to eyelash/brow structure and
  pink hair wisps; in DSCF4550 it tracks the blue-wig strand edges. The current
  Gaussian feather smears all of these into a uniform ~7px ramp — this is precisely
  the halo/edge-artifact mechanism at the hairline in skin smoothing. Interior stays
  fully saturated (1.0), so downstream thresholds (`>0.3`, `>0.4`, `>0.5` in skin.py /
  lips.py / hairwork.py / eyes.py) are unaffected.
- **Lips:** guided ≈ current, marginally crisper vermilion border. Small win only.
- **soft+guided (d)** inherits soft's muddiness — no.
- **One genuine soft-mask virtue observed:** in DSCF7204 the softmax recovers
  upper-lip coverage that hard argmax misclassified as skin (present at ~0.4 prob).
  That is a *recall* fix, not an edge fix — noted as a possible future targeted use
  (e.g. `lips = union(argmax, prob>0.35)` before feathering), but out of scope here.

## 3. Cost (Q3)

- **Raw logits/softmax carried per face:** 19×512×512 float32 = **19.9 MB/face**
  (vs 0.26 MB uint8 argmax; the FaceContext cache today stores ~29 crop-res float32
  region masks, so +19.9 MB/face would be a major cache-size regression, plus a
  dtype/shape change that invalidates cached contexts and complicates the F8.2
  proxy-res cache-discard path). Trimmed variants (4 needed classes f32 = 4.2 MB,
  uint8-quantized = 1.0 MB) still change cache schema. **Moot under the
  recommendation: guided feathering needs no logits at all — zero cache change.**
- **Runtime measured (M3 Pro, CPU/CoreML ORT):** inference 74–91 ms/face; softmax
  over (19,512,512) 11–19 ms; guidedFilter of one mask at crop res (317×406 to
  400×594) **0.8–9.1 ms**. Feathering ~4 region masks with guided filter ≈ +5–25
  ms/face worst case — vs the current Gaussian feather which runs on *full-image-size*
  masks (the paste-back happens before feathering in parsing.py), so guided-at-crop-res
  is roughly cost-neutral and possibly cheaper if feathering moves before paste-back.

## 4. Integration design (Q4)

### Current wiring facts (verified)

- Three argmax sites: `retouch/parsing.py:213` (`parse`), `:313`
  (`parse_hair_full_image`), `:433` (`parse_batch`). `parse` and `parse_batch`
  duplicate the same post-processing block (class→mask, feather, skin-exclusion
  subtraction) — lines 222–249 and 439–467.
- All region masks are already **float32 [0,1] soft masks** downstream
  (`feather_mask` output). No consumer structurally requires binary: morphology
  (`cv2.dilate` in `perf_optimizations.py::_process_face_core`) works on float32;
  threshold consumers (`skin.py >0.3`, `lips.py >0.4`, `hairwork.py >0.05/0.3/0.5`,
  `eyes.py >0.1/0.25/0.5`) rely on saturated interiors, which guided feathering
  preserves. `hairwork.py:763-764` connectedComponents runs on its *own* derived
  map, not the region mask.
- Masks are consumed via `FaceContext.regions` (`detection.py:48`), built once in
  `engine.py:_stage.. → parse_batch`, cached, and combined in
  `perf_optimizations.py::_process_face_core` (`acc_skin`, `acc_skin_hair`,
  `smooth_mask` exclusion list — hair, eyes, brows, lips, under-eye).

### Ranked beneficiaries (face-quality impact)

1. **`regions.skin` + `regions.hair` at the hairline** — the smooth-mask hair
   exclusion is where Gaussian feather causes smoothing halo over strands.
   Largest visible win (see DSCF8007/DSCF7204 renders).
2. **`regions.hair`** for hairwork ops (shine/flyaway) — strand-accurate edge.
3. **`regions.left_eye`/`right_eye` (+brows)** — lash/brow protection in the
   smoothing exclusion (DSCF7204 render shows lashes resolved).
4. **`regions.lips`** — marginal; keep same treatment for uniformity but expect
   little visible change.
   Keep as-is (landmark-derived, no BiSeNet edge): iris, sclera, cheeks, forehead,
   nose, under-eye, contour/highlight circles — these are geometric, not parsed.

### Minimal-risk staged plan (if GO is accepted)

- **Stage 0 — refactor, byte-identical:** extract the duplicated post-processing in
  `parse`/`parse_batch` into one helper
  `_masks_from_label_map(full_label_map, feather, img_bgr, mode)`. Test: existing
  parsing tests + a new golden test asserting `parse` output unchanged
  (`np.array_equal`) on a fixture face.
- **Stage 1 — param + guided path, default off:** add ParamSpec
  `mask_feather_mode` (choices `"gaussian"` (default) / `"guided"`) in
  `retouch/params.py`; per CLAUDE.md, add the matching
  `_process_input_components` entry in `gui.py` (import-time drift guard enforces
  this). In the helper, `"guided"` = bilinear-upsampled binary → guidedFilter
  against the *image* (crop-res guide before paste-back if feasible, else full-res),
  radius = current feather radius per class group, eps=1e-3 on [0,1] guide; clip to
  [0,1]; keep the existing skin-minus-exclusions subtraction order unchanged.
  Fallback: if `cv2.ximgproc` is missing, use `retouch/utils.py::guided_filter`
  (already exists, gray guide) — never crash to Gaussian silently without a log line.
- **Stage 2 — visual QA gate:** re-render the spike panels through the real
  pipeline (one flagship recipe, DSCF8007 + DSCF7204 + one dark-skin/dark-hair
  asset) at face-crop zoom; a human (or session with image-reading) must LOOK at
  hairline/lash crops before/after — per project norm, deltas alone are banned.
  Automated guard: assert interior saturation (median mask value over eroded
  interior ≥ 0.99) and monotone edge profile.
- **Stage 3 — recipe adoption / default flip:** enable in 1–2 flagship recipes
  first; consider default flip only after batch visual QA across test_output set.

### Risks

- **Low-contrast edges (fairness/tone-invariance):** guidedFilter's `eps` is an
  absolute local-variance scale. On dark hair against Fitzpatrick V–VI skin (or any
  low-contrast boundary) the filter degrades toward plain box-blur behavior —
  *graceful* (never worse than today's Gaussian), but the strand-resolution win
  shrinks. No absolute intensity thresholds are introduced (CLAUDE.md rule
  satisfied); if calibration is later desired, scale eps by local guide variance,
  not by fixed luminance cutoffs. Include at least one dark-skin/dark-hair image in
  the Stage 2 QA set.
- **Cache:** guided path keeps mask dtype/shape identical (float32 H×W) — cached
  FaceContexts remain valid. (The rejected soft-logits path would have added
  ~20 MB/face and a schema change; avoided.)
- **Proxy pipeline:** masks built at ≤2048 proxy are upscaled with the image as
  today; guided feathering happens at parse resolution, so proxy behavior is
  unchanged. Optional future enhancement (out of scope): re-run guided filter at
  native res during F8.2 upscale for even crisper full-res edges.
- **Crop-boundary clipping:** hair extending beyond the 30%-padded parse crop is
  clipped today and will remain clipped; guided feathering does not fix or worsen it.
- **Over-crisp edges where ops want soft falloff:** guided edges are near-binary at
  strong boundaries; a smoothing op blending at such an edge could show a visible
  transition line on *skin-to-skin* boundaries. Mitigation: apply a small residual
  Gaussian (r = feather//3) after the guided filter, or blend
  `0.7*guided + 0.3*gaussian`; decide in Stage 2 by looking.
- **`parse_hair_full_image` (site :313):** whole-frame 512 parse is intentionally
  coarse ("soft exclusion signal"); leave on argmax+LINEAR, out of scope.

## 5. Explicit answers

1. **Calibrated enough for soft masks?** No. Probabilities are globally soft
   (median maxprob 0.72–0.74, 98% of pixels <0.9) and transitions span 13–25 image
   px — 2–3× the current feather — because they encode upsampled 1/8-res feature
   blur, not edge position.
2. **Best feathering?** Guided filter of the argmax binary, guided by the image.
   Visually superior at hairline/lashes, equal at lips, saturated interior, 0.8–9 ms
   per mask. Raw softmax and softmax+guided both visually rejected.
3. **Cost of soft masks:** +19.9 MB/face f32 logits (or 4.2 MB trimmed) + cache
   schema change — avoided entirely by the guided approach (zero cache delta,
   ~+5–25 ms/face worst case, likely neutral if feathering moves to crop-res).
4. **Wiring:** single helper extracted from the duplicated parse/parse_batch
   blocks; `mask_feather_mode` ParamSpec defaulting to `"gaussian"` (byte-identical
   off state); GUI component registered per the drift-guard pattern; staged rollout
   with a mandatory visual QA gate.
