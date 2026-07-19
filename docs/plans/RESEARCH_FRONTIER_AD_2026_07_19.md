# RESEARCH — Frontier AD-series: BiSeNet skin mask bleeds onto occluding hands

**Date:** 2026-07-19 · **Author:** Fable, triggered by owner-reported visual
defect on two real cosplay portraits (not a research-backlog item — filed
under the same AA/AB/BB/AC naming convention since it's a verified,
implementation-grade finding of the same class).
**Method:** direct reproduction against real photos (not simulated), skin-mask
visualization overlays, and code reading at HEAD to confirm mechanism —
every claim below is either an executed measurement or a cited line anchor.
**Rules inherited:** classical/deterministic, unit-testable, no learned
models beyond what's already shipped, tone-fair, no absolute intensity
thresholds, no identity-changing defaults.

**Fixed (2026-07-20):** shipped in the working tree — a new
`_clip_bisenet_skin_to_landmark_oval` helper on `FaceParser`, wired into
both `parse()` and `parse_batch()` right after `regions.skin`/
`regions.face_oval` are populated from BiSeNet, before
`_add_landmark_subregions` derives every sub-region from them. One
implementation caveat found only by measuring, not by reasoning: an initial
verification pass using un-shifted (full-image-normalized) landmarks
against a crop produced a false "near-total mask collapse" result — the
bug was in the test script, not the fix, and only surfaced by comparing
mask images directly rather than trusting the numbers. Re-verified with
landmarks correctly shifted to crop space (matching `engine.py`'s own
`shifted_landmarks` step): near-lossless on an unoccluded face (~3% of
pixels change, all at the true boundary), and confirmed on both original
reproduction images with mask-overlay visualizations and one real
full-engine render — the hand is now visibly untouched by face-retouch
ops.

A second, distinct pitfall surfaced while writing the regression test
(`tests/test_parsing_fallback.py::TestBiSenetSkinClippedToLandmarkOval`):
the first synthetic-fixture construction sampled a "hand" point that was
being forced to a *different* BiSeNet class by the test's own logits
override, rather than genuinely testing whether a point BiSeNet calls
"skin" gets excluded when it's outside the landmark oval — so the
assertion passed regardless of whether the fix existed (verified by
temporarily reverting `parsing.py` and re-running: the flawed test still
passed). Caught by explicitly checking the test fails pre-fix and passes
post-fix — not just that it passes post-fix — the same discipline applied
to every fix this session. Corrected construction: no class override,
uniform "skin" logits everywhere in BiSeNet's crop (matching the real bug
exactly — BiSeNet has no competing class to assign, it just always says
skin), sampling a point that's genuinely inside the crop and outside the
landmark circle. Re-verified: fails without the fix, passes with it.

183 tests pass across the parsing/engine suites (2 new regression tests
included). See `TODO_WEEK_2026_07_20.md`-style verification record inline
below rather than a separate doc, since this fix landed same-session as
the diagnosis.

---

## 0. TL;DR

Owner observation: "the retouch engine has some issue to detect if one part
of face covered." Reproduced and root-caused on two real photos where a hand
partially occludes/rests near the face (chin-on-fist, hand-near-hairline).
**Face detection itself is not the problem** — both faces detect at
confidence ≥ 0.98. The actual defect is downstream: the primary skin-mask
path (`parsing.py::parse()`, line 331) uses BiSeNet's raw per-pixel label
output with **zero geometric constraint**, so any skin-toned pixels
BiSeNet's classifier associates with "face skin" — including a nearby
hand/knuckle — get labeled `skin` and receive full face-retouch treatment
(smoothing, tone-evening, etc.), producing the smudged/blurred patch the
owner saw at the chin in the rendered output.

A geometrically-bounded alternative **already exists in the same file**
(`_landmark_fallback_only`, line 576) but is gated behind "BiSeNet failed or
produced an empty mask" (line 344) — it never engages when BiSeNet succeeds,
which is exactly when the bleed happens (BiSeNet found *plenty* of skin, it
just included the hand).

---

## 1. Reproduction (both images, verified by execution)

Test images (not committed; owner-provided, real cosplay portraits with a
hand near/on the face):

| File | Pose | Detection | BiSeNet skin mask mean |
|---|---|---|---|
| `DSCF8470.jpg` | Chin resting on closed fist | 1 face, confidence 0.986, bbox `(1662,1045,1445,1766)` | 0.215 |
| `DSCF8777.jpg` | Hand/wrist near upper-right hairline, holding an object | 1 face, confidence 0.999, bbox `(1183,1205,1362,1417)` | 0.253 |

Both detected via `FaceDetector().detect()` directly, no engine wrapper, no
mocking — a fresh, unmodified detector correctly finds both faces at high
confidence. **Detection is not the failure mode**, contrary to the initial
hypothesis.

Skin-mask overlay visualization (`regions.skin > 0.5` painted red on the
crop) on both images shows the mask **extending past the jawline directly
onto the occluding hand**:
- `DSCF8470.jpg`: mask bleeds onto both visible knuckles of the fist
  propping up the chin — a distinct red patch below/beside the jaw that is
  unambiguously hand, not face.
- `DSCF8777.jpg`: mask extends up and over onto the hand/wrist visible near
  the upper-right hairline — same pattern, different pose.

The full-engine render of `DSCF8470.jpg` (`recipe="natural"`) shows the
visible consequence: a soft, blurred, oddly-toned patch on the knuckles
where face-only smoothing/tone-evening ops ran on hand skin as if it were
cheek/jaw skin — this is what the owner saw and correctly identified as a
defect (their word "detect" was slightly imprecise — it's not a *detection*
miss, it's a *segmentation-boundary* miss, but the underlying instinct that
"something about occlusion is wrong" was exactly right).

---

## 2. Root cause (verified by reading, line-anchored)

`parsing.py::parse()` (the primary path, used whenever BiSeNet succeeds):

```python
# line 331
regions.skin = bisenet_masks.get('skin')
```

`bisenet_masks['skin']` comes from `_masks_from_label_map()` (line 58):

```python
bisenet_masks['skin'] = (full_label_map == 1).astype(np.float32)
```

This is a **direct per-pixel BiSeNet class lookup** — BiSeNet is the
standard CelebAMask-HQ 19-class model; there is no "hand"/"limb" class in
that taxonomy. Any skin-toned pixel the network associates with "face skin"
by local texture/context — including a hand resting near the jaw — gets
label 1. Nothing after this point constrains the mask to a plausible face
boundary before `regions.skin` is used downstream by every face-only op.

The only fallback trigger is line 344:

```python
if regions.skin is None or regions.skin.max() < 0.01:
    regions = self._landmark_fallback_only(...)
```

This only fires when BiSeNet's skin output is **empty or near-empty** — it
has no relationship to whether the mask is *spatially plausible*. In both
reproduction images BiSeNet found abundant skin (mean 0.215–0.253,
correctly covering most of the actual face), so the fallback never engages
— it just also, incorrectly, includes the hand.

**The fix ingredient already exists, unused on this path.**
`_landmark_fallback_only()` (line 576) builds `regions.face_oval` from
actual MediaPipe landmark geometry, not BiSeNet labels:

```python
# line 588
regions.face_oval = self._mask(landmarks, FACE_OVAL, w_img, h_img, feather)
```

`_mask()` (line 686) is a pure geometric polygon fill
(`create_polygon_mask` over the `FACE_OVAL` landmark contour, line 127 — the
standard MediaPipe jawline/hairline/temple outline), independent of pixel
color or texture. A hand near the chin is spatially outside this polygon
regardless of skin tone, so this construction is **naturally immune** to
the bleed the primary path suffers. It is currently only reachable when
BiSeNet fails outright — never as a *bound* on a successful BiSeNet result.

---

## 3. Fix spec (implementation-grade, not yet applied — this is a research/
diagnosis pass, matching repo convention of read-only research before
owner-approved implementation)

**Approach: intersect, don't replace.** BiSeNet's per-pixel skin mask is
finer-grained and more accurate *within* the actual face (it correctly
excludes stray hair strands crossing the cheek, follows the jaw's true
edge better than a fixed landmark contour, etc. — throwing it away in favor
of the landmark-only fallback would be a regression in the common,
unoccluded case). The fix is to **clip** the BiSeNet skin mask to the
landmark-derived face-oval boundary, not to switch paths:

1. In `parse()`, after line 331. Note `regions.face_oval` (populated at
   line 338 from `bisenet_masks.get('face_oval')`) is **confirmed** (this
   pass, re-reading `_masks_from_label_map` line 70) to be *also*
   unconstrained per-pixel — it's a union of BiSeNet classes
   `1|2|3|4|5|10|11|12|13`, skin (class 1) among them, with zero landmark
   geometry involved. It inherits the identical bleed risk as `skin` for
   the identical reason. The landmark-geometric face-oval from
   `_mask(landmarks, FACE_OVAL, ...)` (line 588, currently only reachable
   via `_landmark_fallback_only`) is confirmed the correct, occlusion-
   robust boundary to intersect *both* masks against.
2. Compute that landmark-geometric face-oval mask unconditionally (cheap —
   one polygon fill, reusing the exact call from `_landmark_fallback_only`
   line 588) and intersect: `regions.skin = bisenet_masks['skin'] *
   landmark_face_oval_mask`. Same feather radius as the existing call so
   the boundary softness matches current behavior in the unoccluded case.
3. Apply the same intersection to `regions.face_oval` (line 338) — every
   downstream consumer of `face_oval` (skin exclusion subtraction,
   hair-mask subtraction at line 611, etc.) inherits the identical risk
   otherwise, and would still bleed onto the hand even after `skin` alone
   is fixed.
4. **Byte-identical when no occlusion is present**, by construction: for a
   normal unoccluded face, BiSeNet's skin mask is already spatially inside
   the landmark face-oval (that's what makes it a face mask), so
   intersecting with a correctly-sized, correctly-feathered face-oval
   polygon should be near-lossless — the acceptance test is exactly this:
   confirm no measurable mask change on the existing occlusion-free test
   corpus before landing.

**Tests:**
- Regression on the two reproduction images (or synthetic equivalents —
  composite a skin-toned rectangle near a real face crop's jawline,
  confirm the intersected mask excludes it while the unclipped mask
  includes it) — this is the actual bug, pin it directly.
- Byte-identity / near-identity sweep on existing occlusion-free fixtures
  (per repo convention: golden outputs stay stable) — confirms the fix is
  additive-safe, not just correct on the two new cases.
- Confirm `_add_landmark_subregions` (called right after `regions.skin` is
  set on the primary path, line 347) doesn't depend on the pre-clip skin
  mask's exact shape in a way the clip would break.

**Effort:** small — one polygon fill (already-existing helper) + one
multiply, reusing code that's proven correct in the fallback path.
**Risk:** low, gated by the byte-identity check in item 4.

---

## Bounds / NO-GO

- This is a **masking/segmentation-boundary fix**, not a new detection
  capability — no new model, no learned occlusion-reasoning, purely
  geometric clipping of an existing signal against an existing landmark
  contour that's already computed elsewhere in the same file.
- Does not attempt to *reconstruct* face pixels hidden behind a hand
  (inpainting occluded face regions is a different, much larger problem,
  out of scope here) — it only stops mis-classified hand pixels from
  receiving face-retouch treatment. The hand itself is simply left alone
  (no longer smoothed as if it were cheek), which is the correct behavior.
- No absolute intensity/color threshold introduced — the fix is purely
  geometric (landmark polygon), inherently tone-fair by construction (skin
  tone plays no role in which pixels get excluded).

## Sources

- Direct reproduction this pass: `DSCF8470.jpg`, `DSCF8777.jpg` (owner-
  provided, not committed to the repo).
- BiSeNet / CelebAMask-HQ 19-class taxonomy — standard face-parsing model
  already in use (`parsing.py` `_masks_from_label_map`), no new citation
  needed; the "no hand class" limitation is a property of the taxonomy
  itself (skin/eyebrows/eyes/nose/lips/ears/hair/hat/neck/cloth — 19 total,
  no limb/hand/arm class exists in this or comparable face-parsing sets).
