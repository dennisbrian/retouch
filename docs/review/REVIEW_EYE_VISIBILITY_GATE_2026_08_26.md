# Review — eye-visibility gate + BiSeNet CPU pin (uncommitted)

**Date:** 2026-08-26
**Branch:** `feat/color-science-k9-fix-and-frontier`
**Scope reviewed:** uncommitted working-tree changes only — new `retouch/eye_visibility.py`,
new `tests/test_eye_visibility.py`, diffs to `retouch/eyes.py`, `retouch/eye_enhancement.py`,
`retouch/parsing.py`, `tests/test_parsing_fallback.py`.
**Method:** inline verification + 15 parallel review agents (gate logic, un-gated paths,
golden regression, CPU pin, test quality, perf/concurrency, corpus validation, conventions,
broad suite, adversarial, engine wiring, visual QA, git history, design alternatives, verifier).
**Status:** all 15 agents reported; every deliverable produced. §8 records handover.
**No code was changed.** This document is findings only.

---

## 0. Verdict

The guard addresses a real failure mode (a landmark-derived iris mask survives occlusion, so
the enhancer can paint a convincing iris onto hair/a wig/a hand). It is correctly placed at
the two call sites it touches, its module structure is clean, and it does not violate the
tone-invariance rule.

It should **not** ship in its current form. Four blocking problems:

| # | Blocker | Severity |
|---|---|---|
| P0 | **Pre-existing, committed:** `left_sclera` subtraction is a no-op — sclera brightening paints the *other* eye's iris | Critical |
| P1 | **Pre-existing, committed:** BiSeNet 4/5 and MediaPipe left/right have opposite handedness | Critical |
| B1 | Structurally inert on the landmark-fallback path — the exact path it exists to protect | Critical |
| B2 | **RETRACTED** — golden failures are pre-existing env drift, not this diff (see §2) | — |
| B3 | Its two integration tests are vacuous — mutation-verified, they do not detect the gate | Blocking |
| B4 | **28/36 real eyes gated (78%)** on the DSCF corpus, most wide open; small faces (IED < 30) also gated | Critical |
| B5 | Gate result never reaches the sharpen mask — occluders get sharpened at weight 1.0 | High |

Plus a measured perf regression (§4) and a coverage gap of 7 un-gated eye-editing paths (§3).

B2 was asserted in an earlier draft of this document and is now retracted; §2 records the
correction and its cause.

**P0 and P1 are in committed code and outrank the entire uncommitted diff.** Fix them first;
see §2a. Real-corpus measurement (18 DSCF portraits, BiSeNet live, pinned `.venv`) also raises
B4 to Critical and shows the gate misses its target case entirely — see §2b.

---

## 1. What the change does

`retouch/eye_visibility.py::gate_occluded_eye_regions(regions)` inspects the parser's
eye/iris/hair masks and, when one eye lacks visibility evidence, returns a **shallow copy**
of `regions` with that side's `*_eye`, `*_iris`, `*_sclera` zeroed. Non-gated eyes and the
caller's original object are untouched. Wired unconditionally at the top of both
`EyeEnhancer.enhance` implementations (`retouch/eyes.py:81-84`, `retouch/eye_enhancement.py:249-251`).

Thresholds (`retouch/eye_visibility.py:27-32`), all uncalibrated:

```
_MIN_IRIS_PIXELS       = 5
_MIN_EYE_TO_IRIS_AREA  = 1.25
_MIN_IRIS_INSIDE_EYE   = 0.35
_HAIR_OVER_IRIS        = 0.50
_HAIR_OVER_EYE         = 0.75
```

The `parsing.py` diff is unrelated: it pins the BiSeNet ONNX session to
`["CPUExecutionProvider"]` instead of `build_ort_providers()`. Covered in §6.

---

## 2. Blocking findings

### P0/P1 — Pre-existing bugs in committed code (CONFIRMED on real corpus)

**P1 — handedness mismatch.** BiSeNet labels 4/5 (`parsing.py:369-370`) and MediaPipe
left/right (`parsing.py:709`) have **opposite handedness**. On DSCF7011, `regions.left_eye`
centroid sits at x~=300 while `regions.left_iris` sits at x~=247. Across the corpus, **7 of 8**
faces with both eye masks present are mismatched. The codebase contradicts itself on the
convention: `face_quality.py:27` calls the 263-cluster `LEFT_EYE_INDICES`; `parsing.py:161`
calls the same cluster `RIGHT_EYE` — the likely origin.

**P0 — `left_sclera` subtraction is a total no-op.** The same mismatch poisons
`parsing.py:753-761`: `left_sclera = clip(left_eye - left_iris)` subtracts two **disjoint**
masks.

```
np.array_equal(r.left_sclera, r.left_eye)  -> True      (both eyes)
own iris inside sclera                     -> 0.0
OPPOSITE iris inside sclera                -> 0.75 / 0.92
```

So `left_sclera` is the entire *other* eye including its iris, and `eyes.py` brightens that as
sclera. **This is a live rendering defect in committed code, independent of the uncommitted
gate, and likely explains eye artifacts already present in output.** Fix before anything else.

### B0 — Real-corpus measurement: the gate suppresses most real eyes and misses its target case

Measured on 18 real DSCF portraits in the pinned `.venv` (MediaPipe 0.10.5, BiSeNet available):

- **28 of 36 eyes gated (78%)**, almost all wide open (EAR 0.3-0.5), mostly **asymmetrically** —
  one eye enhanced, one not.
- Driven by P1, not by occlusion: `iris_inside_eye` is **0.000** on most real faces because the
  masks are handedness-swapped. Swapping them drops gating 28 -> 17.
- **10 of 18 faces have no eye mask at all** (`eyemx 0.00`), so no threshold on that mask is
  tunable.

And it does not catch what it was written for. Collapsing the upper lid onto the lower on real
landmarks (EAR 0.302 -> 0.057, a fully shut eye):

```
eye_to_iris_area = 3.62      iris_inside_eye = 1.0      _should_gate_eye = False
```

The mask-area signal is **blind to closed lids by construction**.

### B0a — Second BiSeNet failure mode: eye-class collapse (CONFIRMED, independent corpus run)

A separate corpus pass (10 DSCF images, pinned `.venv`, real BiSeNet inference on
`models/resnet18.onnx`) reproduced P1 independently and found a **second**, more common failure:

| mode | frequency | signature |
|---|---|---|
| **Class collapse** | 7/10 | `left_eye` (class 4) covers **both** eyes; `right_eye` (class 5) returns empty (`sum=0.0`) |
| **Genuine L/R swap** | 2/10 | both classes present and localized, cross-overlap ~0.98-1.0, same-side overlap 0.0 |
| Masks mostly absent | 1/10 | downcast pose, both overlaps ~0 |

Verified visually: green (`left_eye`) painted over both eye sockets, red (`right_eye`) absent.

**8 of 10 images had at least one eye gated**, including confirmed false positives on fully open,
unoccluded eyes — DSCF8007 (both eyes wide open, direct gaze, no occluder) gated on **both**
sides; DSCF6961 gated one of two visibly-equal eyes.

In every gated case the firing metric was `iris_inside_eye` at exactly **0.000** — a hard fail,
not a near-miss threshold question. `hair_over_iris`/`hair_over_eye` never exceeded **0.013**
across all 20 eyes, *including heavy-bangs faces*, so the hair branch is **structurally
unexercised on real input** — unvalidated, not validated-passing. (`parse()` uses the face-crop
hair mask, not `parse_hair_full_image`.)

**`_should_gate_eye` is doing exactly what it was designed to do. Its input is broken.**

**Gaps, stated explicitly:** no Fitzpatrick V-VI subject exists in the local corpus (uniformly
light-skinned East Asian cosplayers); no sunglasses, profile, or small/distant-face cases; and no
true full-occlusion case among the sample — so the absence of a false negative here is weak
evidence, not a clean pass.

### B6 — Gating one eye changes the other, via a latent catchlight bug (CONFIRMED by render)

Real renders of `EyeEnhancer.enhance` on the golden fixture (`/tmp/eyegate_*.png`):

- **Scenario (a), both eyes visible:** gated vs. ungated is **bitwise identical** over the full
  ROI (max=0, nnz=0), while enhancement genuinely ran (gated vs. input: max=65, 4898 px). A true
  no-op, not a vacuous one.
- **Scenario (b), occluded:** ungated, the pre-change code **paints a full iris + sclera onto an
  eye whose mask says it isn't there** (max=65). Gated, only that eye reverts to raw input; the
  left eye stays enhanced (max=53). **No seam or halo** — the inter-eye gap band is max=0.
- **But the good eye shifts by 30 px, max per-channel delta 19.**

Bisected to `_enhance_catchlights` (`eyes.py:119-121`), which pools centroid and radius over the
**combined** iris mask. A second, independent test-writing pass corroborated the coupling and
narrowed it further: re-running with `vessel_strength=0, catchlight_strength=0` **kept** a
3/255 leak, so `_enhance_whites` also contributes via the combined `whites_mask_l + whites_mask_r`
(`eyes.py:92-98`) — the catchlight pass is not the only source. Strict `array_equal` on the
visible eye is therefore unachievable in the legacy path; assert a ratio instead. With both eyes present the synthetic spot lands at (117,128) — *between*
the irises — and `*iris_mask` annihilates it (0 surviving px). With the right eye gated, the
centroid moves to (121,100) and **44 px survive inside the left iris**.

So the synthetic catchlight fallback is **effectively dead code for two-eye faces, and the gate
resurrects it**. On a real image where natural catchlight detection fails, gating one eye would
newly synthesize a catchlight on the good eye that the pre-change code never drew. This is a
**pre-existing latent bug the gate exposes rather than causes.** Verified fix: calling
`_enhance_catchlights` per-eye drops coupling from 30 px to 0. `eye_enhancement.py` is strictly
per-eye and structurally free of this.

### B1 — The gate is a structural no-op on the landmark-fallback path (CONFIRMED)

In `_landmark_fallback_only` (`retouch/parsing.py:660-693`):

- `regions.hair = clip(person_mask - regions.face_oval)` (`parsing.py:692`) is **exactly zero
  everywhere inside `face_oval`** by construction — which is precisely where the eye and iris
  landmarks live. `_weighted_overlap(hair, iris)` and `_weighted_overlap(hair, eye)` are
  therefore always `0.0`, so `_HAIR_OVER_IRIS` and `_HAIR_OVER_EYE` can never fire.
- On the same path `left_eye`/`right_eye` are landmark polygons (`parsing.py:670-671`), not
  independent BiSeNet segmentation. The area-ratio and inside-eye checks then compare two
  geometrically-linked landmark quantities that both survive occlusion together, so neither
  degrades when the eye is actually covered.

All four gates are inert on this path. This is the ONNX-free path that goes live under
MediaPipe/protobuf env drift (a documented, recurring condition in this project), and it is
the path the golden-face fixture uses. `tests/test_eye_visibility.py` exercises only synthetic
stubs, never this path — so the failure is untested.

**Pixel-level proof (red-team, real parser output).** Masks built by the actual
`FaceParser._landmark_fallback_only` from the frozen landmarks in `tests/golden_face_fixture.py`,
with a physical hair lock placed over one eye covering 101/101 iris pixels:

```
hair mask total sum = 276010.0
hair-over-iris      = 0.0        <- structurally, by construction
hair-over-eye       = 0.0
_should_gate_eye    = False
gate returns SAME object (no gating)? True
```

Through an engine-shaped call (`perf_optimizations.py:789`, `ctx.eye_enhance=40`):

```
hair-covered iris px changed: 101 / 101   max delta 89
```

At strength 60 the delta reaches 224/765 (hair pixel `[28,38,62]` -> `[2,12,36]`). The amplified
delta image was rendered and visually inspected: it shows a **complete painted eye — sclera plus
iris disc — on solid hair**. This is the exact failure the module was written to prevent,
reproduced end-to-end.

### B2 — RETRACTED: golden face-path failures are pre-existing, not caused by this diff

An earlier draft of this review claimed three `tests/test_golden_pipeline_face.py` tests
(`natural`, `porcelain_unified_v1`, `outdoor_harsh_sun_v1`) regressed. **That was wrong.**

Two agents independently confirmed via a clean `git worktree add --detach /tmp/head-wt HEAD`
that the three tests fail **identically on HEAD and on the working tree**, with byte-identical
hashes on both sides:

| recipe | expected | got (HEAD *and* dirty tree) |
|---|---|---|
| `natural` | `55a90872668eb27d` | `e74ddbbbeb11dc68` |
| `porcelain_unified_v1` | `3b4f8cd5ec50409d` | `e762929eac5afae1` |
| `outdoor_harsh_sun_v1` | `144e220a3d123849` | `a404051c837e6e57` |

**Real cause — interpreter drift.** The failure tracks which Python runs pytest, not the tree:

- `python3 -m pytest` → system Python 3.9.6 (mediapipe 0.10.35, protobuf 6.33.6) → **3 failed**
- `.venv/bin/python -m pytest` → (mediapipe 0.10.5, protobuf 3.20.3) → **8 passed, with the
  full uncommitted diff applied**

Under system Python, `FaceDetector.available == False` ("MediaPipe 0.10.35 Tasks FaceLandmarker
is not supported by this macOS runtime"). `engine.py:2302` calls
`self._detector.segment_person(img_bgr)` **unconditionally** — the `face_contexts=` cache
short-circuits detection but *not* segmentation — so `person_mask` silently degrades to all-ones
(mean 1.0 vs. the real 9.879e-06). Proven bidirectionally: monkeypatching `segment_person` to
return ones inside a working `.venv` reproduces `e74ddbbbeb11dc68` exactly; restoring it recovers
the expected `55a90872668eb27d`. `person_mask` is the sole differing input.

Hypotheses eliminated with evidence: gate firing at engine/proxy resolution (patching
`gate_occluded_eye_regions` to identity across all three importing namespaces left hashes
bit-identical); regions object identity/mutation; the `CPUExecutionProvider` pin (the fixture
sets `parser._sess = None`, so BiSeNet never runs); import-time side effects of the removed
`build_ort_providers` import.

**Process lesson:** CLAUDE.md's Key Commands block documents `python3 -m pytest tests/ -q` with
bare `python3`, which resolves to the drifted system interpreter and silently degrades the whole
face path. This trap will recur. Worth an owner decision on pinning the documented command to
`.venv/bin/python`.

**Caveat that survives the retraction:** the golden harness gives **zero coverage of the gate**.
On this fixture `regions.hair` is entirely zero, area ratios are 2.6–3.0 against a 1.25
threshold, and `iris_inside_eye` is ~0.95–1.0 — so `_should_gate_eye` is `False` on both sides
and the gate returns the original object untouched. That is *why* the diff is hash-neutral here.
A green golden run means "this did not regress what the harness exercises," not "the gate works."

### B3 — The two integration tests are vacuous (CONFIRMED by mutation)

`tests/test_eye_visibility.py:63-78` and `:81-96` manually zero the right eye's masks in the
input, then compare against an enhancer run with those same masks zeroed. This proves the
enhancer respects zero masks — not that `gate_occluded_eye_regions` fires. Replacing the gate
with an identity function leaves **both tests passing**. They do not protect the feature they
exist for.

The correct assertion shape: compare *gate active on occluded input* against *gate patched to
identity on the same occluded input*, asserting the outputs differ in the occluded eye and are
identical in the visible eye.

### B4 — Small faces are falsely gated; enhancement drops to zero (CONFIRMED, reproduced)

Two distinct false-positive regimes, the second far more serious.

**Squint (moderate).** With production feather radii, an eye at ~30% openness yields an area
ratio of 1.07–1.09, below `_MIN_EYE_TO_IRIS_AREA = 1.25`; at 35% it is 1.36 and passes. The
red-team pass judged gating at <=30% aperture (`iris_inside 0.33`) arguably *correct* behavior
for a nearly-closed eye, so this alone is defensible.

**Small faces (high).** `_iris_mask` clamps iris radius to `max(int(ied*0.07), 4)`
(`parsing.py:805`). Below IED ~= 57 the iris mask stops shrinking (frozen at 61 px) while the
eye mask keeps shrinking, so `eye_to_iris_area` collapses through the threshold. Crossover
pinned at **IED ~= 30**:

```
  29.5   2.06->4  CLAMPED     61     72  1.18  True    <- gated
  30.5   2.13->4  CLAMPED     61     79  1.30  False
```

On a plainly visible small face, enhancement is suppressed entirely:

```
ied=25.1  gate produced a copy (i.e. gated): True
  total px changed by enhance(): 0   -> visible eyes get ZERO enhancement
```

**15 of 26** fully visible, unoccluded eyes were gated across the swept range. This is not an
exotic input: this project's own notes record 16 px faces detecting fine, and group shots at the
2048 px proxy routinely land under IED 30.

**Root cause is in `parsing.py`, not `eye_visibility.py`.** The clamped iris radius (and, for
B1, `hair = person_mask - face_oval`) means **tuning the thresholds in isolation fixes neither
B1 nor B4**. What would be needed: a hair signal that survives inside the face oval, and an area
ratio normalized against the *clamped* iris radius rather than raw pixel counts.

### B5 — The gate's result never reaches the sharpen mask (CONFIRMED)

`retouch/eyes.py:84` does `regions = gate_occluded_eye_regions(regions)` — a **local variable
rebinding inside `EyeEnhancer.enhance`**. Python does not propagate that back to the caller.
`_process_face_core` (`perf_optimizations.py:236`) holds its own `regions` reference, passes it
to `eyes.enhance(...)` at `:789`, and then at `:1010-1011` builds `acc_sharpen`'s eye
contribution from `regions.left_eye`/`regions.right_eye` — the **original, ungated masks**.

Consequence: every eye the gate suppresses for pixel-editing still contributes at **weight 1.0**
(the highest of any region — eyebrows and hair-edges are 0.53, `perf_optimizations.py:1022-1024`)
to the selective-sharpening mask. The occluder itself — a hair strand, a hand, a closed eyelid —
gets targeted unsharp masking exactly where the pipeline intended zero enhancement. On the
highest-scrutiny region of a portrait, that reads as an obvious retouch artifact.

This fires unconditionally on every gated eye, independent of the `ones_like` fallback below.

**Recommended fix (not applied):** hoist the gate one level up, into `_process_face_core`, so
every downstream consumer sees the gated copy:

```python
from .eye_visibility import gate_occluded_eye_regions
regions = gate_occluded_eye_regions(regions)   # once, before all per-region use
```

The existing call inside `EyeEnhancer.enhance` then becomes an idempotent no-op (re-gating
already-zeroed masks finds nothing to gate). One line closes the leak at the source.

---

## 3. Coverage gap — the gate protects 2 of 9 eye-editing paths (CONFIRMED)

`_EYE_FIELDS` (`retouch/eye_visibility.py:34-37`) covers only `*_eye`, `*_iris`, `*_sclera`.

**Gated (correct):** `eyes.py::EyeEnhancer.enhance` (whites, iris sculpt, catchlights, and
`apply_corneal_curvature_shading`, whose only call site is inside the gated method);
`eye_enhancement.py::EyeEnhancer.enhance` (sclera brighten, iris sat/hue/brightness).

**Not gated, reachable, and painting eye pixels:**

| Consumer | Site | Live when |
|---|---|---|
| `skin.py::restore_micro_texture` | `skin.py:1020-1028` | `ctx.micro_restore > 0` |
| `skin.py::local_clarity` | `skin.py:1094-1097` | `ctx.clarity > 0` |
| Global selective sharpening | `perf_optimizations.py:1010-1024` → `engine.py:4432-4452` | `ctx.sharpen > 0` |
| GUI Advanced Retouch "Eyes" mask | `advanced_retouch.py:258` → `:322-350`, wired `gui.py:3249` | user action |
| — same, on session replay | `advanced_retouch.py:356-386` | session reload |
| `skin.smooth_undereye_shadow` | `perf_optimizations.py:479-484` | `ctx.undereye_shadow_strength > 0` |
| `undereye.py::UnderEyeRepairer.repair` | `perf_optimizations.py:757-761` | `ctx.dark_circles > 0` |
| `undereye._processor.process` | `perf_optimizations.py:763-770` | darken/puffiness params |

Two aggravating details:

- Eye masks carry weight **1.0** in the sharpen accumulator — the highest of any region. And
  `engine.py:4433-4434` falls back to `np.ones_like` when the mask is empty, so a fully-gated
  eye could receive *frame-wide* sharpening rather than none. Reachability being quantified (§8).
- The three under-eye ops share a root cause: `left_under_eye`/`right_under_eye` are landmark
  polygons only (`parsing.py:716-717`) with no BiSeNet cross-check, and the gate's field list
  omits them — so routing those call sites through the gate today would be a **no-op**. The
  field list must be extended first.

### 3a. Under-eye scoping — severity corrected downward

An earlier draft grouped the three under-eye ops with the rest as equally exposed. That
overstated the risk; the corrected picture:

**The masks are not raw polygons.** `_add_landmark_subregions` (`parsing.py:740-750`) multiplies
`left_under_eye`/`right_under_eye` by `regions.skin` before any op sees them. On the BiSeNet path
`skin` is `label==1` and `hair` is the disjoint `label==17`, so a hair-covered strip goes to ~0 —
a real, working occlusion signal, though implicit, soft (Gaussian-feathered), and undocumented as
a safety property. On the **fallback path** `skin` is pure landmark geometry (`face_oval` minus
eye/eyebrow/lip polygons, times the coarse `person_mask`) with no appearance evidence, so the
cross-check is a no-op — the same structural failure as B1.

**The three ops are not uniform risk:**

| Op | Baseline | Self-limiting under occlusion? |
|---|---|---|
| `skin.smooth_undereye_shadow` (`skin.py:280-410`) | median *inside* the zone (`skin.py:350`) | Largely yes — under full hair the baseline *is* hair luminance, and `min_coverage` usually rejects a flat patch |
| `undereye.detect_dark_circles` (`undereye.py:37-77`) | **dilated surround** minus mask (`:59-61`) | **No** — worst case is *partial* coverage (bangs over strip, cheek bright); a binary "mostly occluded" gate would miss exactly this |

**Shipped exposure is small.** `undereye_darken_removal`/`undereye_puffiness_reduction` — the
non-self-limiting, surround-baseline op — appear in **zero recipes** (grep-confirmed); they are
manual/GUI-only. Shipped `dark_circles` values are 0.10-0.25, giving
`l_lift = clip(median-L,0,30) * mask * strength * 0.3` a max of **~1.5-2.25 L units**.

**The most-exposed consumer was not in the original list:** `under_eye_blush`
(`makeup.py:89-97`) is `True` in 8 shipped recipes (`recipes.py:92, 917, 936, 955, 1033, 1086,
1407, 1441`), applying `lab[a] += blush_mask*16*s` (max ~5.6 a-units) — a pink *chroma* wash into
hair, which reads dirtier than a luminance lift. Also unlisted: `skin.py:1027-1028`
micro-contrast reinjection.

**Verdict: nice-to-have, not must-fix.** Unlike the iris case — which *synthesizes* plausible
structure (a fake iris) over an occluder — under-eye ops only nudge existing pixels within small
bounded deltas.

**Design trap for whoever implements this.** `perf_optimizations.py:394` uses the under-eye masks
as **exclusions** from frequency-separation smoothing. Zeroing them there means *less* exclusion,
i.e. *more* smoothing at the eye socket — the gate would fail open and make that op more
aggressive exactly where evidence is weakest. So the fix is **per-call-site strength attenuation
with a soft ramp** (full strength above ~0.85 skin coverage, fully gated below ~0.30), not
zeroing a shared mask field. A soft ramp also handles the partial-occlusion case that a binary
cutoff misses. Note the coverage denominator requires capturing pre-clip polygon area as a new
field, since the raw polygon is gone after the skin multiply.

**Coupling: one-way only.** A gated eye should attenuate that side's under-eye (hair over the
socket almost always covers the strip below). The reverse must **not** be added — it would
regress visible-eye cases where only the tear-trough reads as low-coverage (heavy makeup, motion
blur, small-face feather floor).

**Correctly excluded, not findings:** smoothing/shine/wrinkle exclusion masks (they
dilate-and-subtract to *protect* eyes); `makeup_v2` eyeliner/eyeshadow and `geometry.py` eye
warps (driven by landmark indices, not parser masks); `lighting.py` catchlight estimation
(analysis-only, no pixel writes).

---

## 4. Performance — measured regression (CONFIRMED)

Masks are crop-local ROI masks, not full-source-image masks (`engine.py:3008-3013` crops before
parsing) — but `quality="full"` (the default) runs per-face work at native resolution, so ROIs
are large.

| ROI | Cost per `gate_occluded_eye_regions` call |
|---|---|
| 400×400 (CLAUDE.md benchmark scale) | 1.3 ms |
| 2600×1600 (typical padded ROI) | **35.6 ms** |
| 4160×4400 (clamped worst case, 24MP source) | **148.8 ms** |

Both enhancer sites can fire in the same `process()` (`perf_optimizations.py:787` and
`:795-796`), doubling this to ~70 ms / ~300 ms per face — redundantly, since `regions` is
identical between the two calls. Reference point: 702 ms/face documented benchmark. The GUI
re-runs it on every slider drag (`gui.py:1511-1512` reuses cached contexts, which skips
parsing but not the render pipeline).

**Root cause is an ordering bug.** `eye_visibility.py:78-84` normalizes `eye`, `iris`, *and*
`hair` through `_as_mask` unconditionally, *before* the cheap `iris is None or iris.max() < 0.01`
early-out at line 84. cProfile over 20 calls at 2600×1600: `_as_mask`'s `nan_to_num`/`clip`
chain is **87% of total gate time** (0.66 s of 0.759 s), firing 6× per call even when nothing
gates. The docstring implies cheap-guard-first; the code does the opposite.

**Memory:** ~50 MB per gated eye at mid-crop (~440 MB if both gate at worst case), plus ~12
transient full-ROI float32 arrays churned per call on the common *no-op* path.

**Caching:** the gate is a pure function of five masks fixed at parse time. It should be
computed once and memoized on `FaceRegions`/`FaceContext`. Note `FaceRegions.__slots__`
(`parsing.py:232-247`) has no spare slot — one must be added.

---

## 5. Concurrency and aliasing — clean (CONFIRMED, no hazard)

- Each face gets a distinct `FaceRegions` (`engine.py:3046` cached / `:3052` fresh), so the
  `ThreadPoolExecutor` fan-out (`engine.py:3148-3161`) never shares an instance across threads.
- `copy.copy` on the `__slots__` class verified correct: untouched fields stay shared by
  reference, gated fields decouple after `setattr`, the original array is never mutated.
- A grep across the full per-face consumer chain (`skin`, `blemish`, `undereye`, `makeup`,
  `makeup_v2`, `hair`, `relight`, `teeth`, `lips`, `perf_optimizations`) found **no in-place
  writes** into `regions.*` fields. The shallow copy therefore leaks nothing back into the
  GUI's long-lived cached `FaceContext`.

**Caution for any future change:** `eyes.py:94` does
`np.clip(regions.left_eye - regions.left_iris, 0, 1)` with no `None` guard — a "return `None`
instead of zeros" optimization would break there.

---

## 6. BiSeNet CPU pin (`retouch/parsing.py`)

1. **The justifying comment is undocumented and contradicted.** `parsing.py:275` claims BiSeNet
   "fails CoreML MLProgram compilation on the supported macOS runtime."
   `docs/guides/PERFORMANCE_TUNING.md:131` documents CPU pinning for **NAFNet**, not BiSeNet.
   `docs/plans/FABLE_TASK_LIST_2026_07_15.md` states BiSeNet-on-CoreML has "no evidence this is
   broken, just unaudited." "MLProgram" appears nowhere else in docs or git log in connection
   with a BiSeNet failure. Either attach evidence or reword to "unaudited, pinned conservatively."
2. **The pin is platform-blind.** `build_ort_providers()` (`perf_optimizations.py:1307`) uses
   `if CoreML / elif CUDA / elif DirectML / else CPU` — mutually exclusive, so exactly one
   accelerator is selected per environment (not a stacked fallback chain; an earlier draft of
   this document said "CoreML > CUDA > DirectML > CPU", which overstated it). The substantive
   point stands: pinning CPU forecloses whichever accelerator that environment would have
   selected, so **CUDA and DirectML** hosts lose BiSeNet acceleration too, not just CoreML. `PLAN_CODEBASE_AUDIT_2026_07_14.md` (ARCH-2) asks for a
   per-model *denylist* (drop CoreML, keep CUDA); no such mechanism exists yet.
3. **Unparametrized is fine.** `enhance.py:165-180` pins NAFNet to CPU unconditionally with no
   `ParamSpec`. Precedent supports a correctness guard without a user-facing switch.
4. **Import removal is clean** — no dangling `build_ort_providers` reference in `parsing.py`.
5. **The new test is fragile on a clean checkout.**
   `tests/test_parsing_fallback.py::test_bisenet_session_is_pinned_to_cpu` patches
   `os.path.exists` but not `model_status`. Without the 53 MB BiSeNet model file, `_model_path`
   stays `None`, the ONNX block never runs, and `mock_sess_cls.call_args.kwargs` raises
   `AttributeError: 'NoneType' object has no attribute 'kwargs'` — reproduced verbatim by
   patching `model_status` to report unavailable. Note the precondition is hypothetical for
   *this* checkout: `model_status("resnet18_bisenet")` currently returns `available: True`
   (it tracks something other than on-disk file presence), so the test passes as run today.

---

## 7. Conventions, fairness, history

**No hard violations.**

- **Tone-invariance:** satisfied. The gate uses mask-area and mask-overlap ratios, never pixel
  intensity. Upstream BiSeNet masks are argmax-class decisions, not luminance thresholds, and
  feathering is geometric. One flagged risk: dark hair against low hair/skin contrast may
  inflate `hair_over_eye`, giving a differential false-positive rate. It fails toward *not*
  enhancing — the safe direction — but warrants corpus validation, which this change lacks.
- **`params.py` single-source-of-truth:** no `ParamSpec` needed; consistent with the NAFNet
  precedent for always-on correctness guards.
- **Module structure:** `eye_visibility.py` is a correct leaf-module factoring (both enhancers
  depend on it, it depends on neither). No circular import. Naming and docstrings compliant.
- **Minor gap:** `eye_visibility.py:142` has `except Exception: return regions` with **no log
  line**, unlike every comparable fallback in `enhance.py`. Note this fallback fails *open* —
  it defeats the gate entirely rather than degrading. Same for the silent shape-mismatch skip
  on `hair` (line 111) and the `except (TypeError, ValueError)` on `setattr` (line 157), which
  would miss `AttributeError`.
- **History:** genuinely new ground — no prior eye-occlusion attempt in git history, no design
  doc it diverges from. The poster-FP "owner sign-off" convention is scoped to that veto
  specifically and is not violated. But the guard lands as an always-on heuristic with no
  real-corpus precision/recall measurement, unlike the detection work's rigor.
- **Branch fit:** the eye guard is defensible on this branch (which already drifted into
  detection work). The BiSeNet CPU pin is orthogonal to every theme here.
- **Indexing:** if this ships, CLAUDE.md's Outstanding Fixes needs a one-line entry per repo
  convention.

---

## 8. Still outstanding

All agents reported. Nothing outstanding. Handover artifacts:

**Replacement test suite — `/tmp/test_eye_visibility_proposed.py` (25 tests) + `/tmp/conftest.py`.**
Both files are required: the mutation harness lives in the conftest, so copying only the test
file into `tests/` silently loses the non-vacuity proof.

```bash
python3 -m pytest /tmp/test_eye_visibility_proposed.py -q -p no:warnings
#   -> 24 passed, 1 xfailed
GATE_MUTATE=1 python3 -m pytest /tmp/test_eye_visibility_proposed.py -q -p no:warnings
#   -> 10 failed, 14 passed, 1 xfailed
```

Ten tests fail under mutation; the current committed suite fails **zero** (§B3). Coverage added:
each gate branch via the public API, real `FaceRegions` `__slots__`, feathered end-to-end,
shape mismatch, `None` iris, NaN/inf scrubbing, and the `copy.copy` fail-open path (paired with
a copyable control proving the `except` branch was actually reached).

Rows that do *not* catch the mutation are intentional — they assert **non**-gating (guarding
over-gating, which identity trivially satisfies) or test `_should_gate_eye` directly for
"which predicate fired" diagnostics, deliberately paired with the public-API tests that prove
the plumbing.

**Note:** these renders and scripts live in `/tmp` and will not survive a reboot. Move them into
the repo before relying on them.

### Broad-suite regression check — clean (CONFIRMED)

Full suite in 6 batches (hang-risk mitigation per CLAUDE.md), excluding
`test_golden_pipeline_face.py` (covered in §2):

```
4553 passed, 8 failed, 10 skipped   (4571 collected)
```

**All 8 failures are PRE-EXISTING**, reproduced byte-identically on a clean
`git worktree add /tmp/retouch-head HEAD` (`6504109`) with the same gitignored local artifacts
(`test_output/*.jpg`, `models/*`) populated. **Zero new failures** attributable to
`eye_visibility.py`, `eyes.py`, `eye_enhancement.py`, or the `parsing.py` provider pin.

| Failures | Cause |
|---|---|
| `test_body_skin.py` x3 (smooth/whiten/blemish gate independence) | `BlemishRemover.remove()` never invoked for the body-skin stage — pre-existing `engine.py` gating bug, untouched by this diff |
| `test_integration.py::TestWithRealImage` x3, `test_style_extraction.py` x1, `test_batch_workflow.py` x1 | MediaPipe 0.10.35 / macOS runtime incompatibility (`detection.py:192`) — same env drift as §2 |

Syntax check clean across `retouch/*.py`, `gui.py`, `cli.py`. Import-removal safety confirmed:
`build_ort_providers` remains defined in `perf_optimizations.py` and in use by `enhance.py` and
`diagnostics.py`, both passing; `tests/test_parsing_fallback.py` including the new
`test_bisenet_session_is_pinned_to_cpu` passes clean.

**Reading this correctly:** a clean broad-suite run means the diff introduces no *detectable*
regression. It does **not** validate the gate — §B0/§B0a show the existing suite has no coverage
of the real-corpus behavior where the gate suppresses 78% of eyes.

### B7 — Feathering eats the hair-threshold margin (found while writing the tests)

Hair covering the **top half** of a feathered eye — the bangs/lid-down geometry the gate exists
to catch — produces:

```
hair_over_iris = 0.499   (threshold 0.50)
hair_over_eye  = 0.402   (threshold 0.75)
```

Neither fires, so a half-occluded eye is **fully enhanced**. `_weighted_overlap` is a soft ratio,
so feathering costs real margin. Pinned as `xfail(strict=True)` rather than tuned green — any
threshold change turns the suite red and forces a deliberate un-xfail **in the same commit**.


## 8a. Recommended order of work

0. **Fix `_enhance_catchlights` to run per-eye** (`eyes.py:119-121`) — latent, and any eye gate
   will expose it (§B6).
1. **Fix P0** (`parsing.py:753-761`, sclera subtraction) — committed, live, highest severity.
2. **Fix P1** — align BiSeNet 4/5 to MediaPipe left/right at `parsing.py:369-370`, and reconcile
   the naming contradiction (`face_quality.py:27` vs `parsing.py:161`).
3. **Do not ship `eye_visibility.py` as-is.** Keep `gate_occluded_eye_regions`'s shallow-copy
   per-eye plumbing (`:120-162`) verbatim — it is sound. Replace `_should_gate_eye`'s *evidence*:
   **EAR** (eyelid aspect ratio) as primary, indices already present at `face_quality.py:27-28`
   and `parsing.py:161`; a tone-adaptive iris-vs-sclera luminance ratio as secondary (expressed
   against the eye's own high percentile, never an absolute threshold, per the fairness rule).
   EAR is independent of the iris circle, of BiSeNet, and of which parsing branch ran — the only
   candidate immune to all three root causes. Validated on real pixels: DSCF4576 (0.108) and
   DSCF4612 (0.122) are genuinely closed; DSCF6961 (0.513) is wide open.
   Signals ruled out, each with a citation: landmark visibility (`_detach_landmarks`,
   `detection.py:106-123`, copies only x/y/z); MediaPipe z (canonical-model-relative, not scene
   depth); detector confidence (per-face, `confidence_source="mediapipe_presence_unavailable"`);
   person mask (a hand or bangs are *inside* it); `parse_hair_full_image` (sub-pixel at the iris,
   plus a full-frame inference now pinned to CPU).
4. Add a `ParamSpec` + `_process_input_components` entry + a log line — the gate is currently
   unconditional and silent at two render-path call sites. **Owner call:** gating *one* eye may
   be worse product behavior than gating neither.
5. Write `docs/plans/RESEARCH_EYE_OCCLUSION_2026_08_26.md`: per-eye visible/partial/occluded
   labels over the 83-image DSCF corpus plus sourced hard cases (blink, bangs, wig, hand,
   profile, sunglasses, small faces). **Stratify BiSeNet-available vs unavailable as separate
   arms.** State the cost asymmetry (false gate = mild regression; missed gate = catastrophic
   painted iris), report ROC per signal, split Fitzpatrick I-III / IV-VI per the fairness rule.
6. Extend tests to cover the BiSeNet-unavailable arm, which `tests/test_eye_visibility.py`
   never exercises (it paints hair explicitly).


Closed since the first draft: golden-regression root cause (§2, retracted); sharpen-fallback
reachability (§B5); engine wiring / cache interaction (§5); independent verification of the §2
and §6 claims; adversarial red-team (§B1 pixel proof, §B4, §9); under-eye scoping (§3a); design/signal review and real-corpus validation (§P0/P1, §B0, §B0a, §8a); visual QA renders (§B6).

---

## 9. Minor findings

- `_MIN_IRIS_PIXELS = 5` is unreachable in practice — `parsing.py:805` floors iris radius at
  `max(int(ied*0.07), 4)`, yielding ≥69 support pixels after feathering. Dead margin.
- A weak, uniformly sub-0.25 iris mask passes the `iris.max() < 0.01` liveness check but yields
  `iris_pixels == 0`, returning "not gated" rather than "unreliable". Downstream, sclera
  brightening may still run across the iris region (`eye_enhancement.py:256`).
- `_MIN_IRIS_INSIDE_EYE = 0.35` has genuine margin — swept to 50% eye aperture, overlap stayed
  ≥0.72. Not a standalone false-positive risk.
- Soft hair confidence slips under the threshold: uniform 0.49 over a fully covered eye passes
  (0.50 gates, 0.49 does not). Relevant because `parse_hair_full_image` returns softmax hair
  *confidence*, not argmax, per its own docstring.
- Shape mismatch fails **open** (`eye_visibility.py:111`): fully opaque hair covering everything
  is ignored when shapes differ. Realistic, since `parse_hair_full_image` yields a full-image
  mask while iris masks are ROI-crop sized.
- `_as_mask` clips but never normalizes, so a uint8 0-255 mask becomes effectively binary and can
  **invert the verdict**: float hair 0.4 -> not gated; the same mask as uint8 102 -> gated.
- `setattr` at `eye_visibility.py:157` catches only `(TypeError, ValueError)`; a read-only
  property or frozen dataclass raises `AttributeError`/`FrozenInstanceError` out of the eye
  stage. Latent — no in-repo regions type is non-settable today.
- Aliasing is real but not currently exploitable: the shallow copy shares non-eye masks
  (`g.skin is r.skin`), so in-place math would leak to the caller's cached regions. A repo-wide
  grep for `+=`/`*=`/`-=`/`/=`/slice-assign on region attributes returned **zero hits**. Latent
  footgun if anyone adds in-place mask math later.
- Type fuzzing held: `dtype=object`, 3-D, `np.matrix`, `(0,0)`, read-only, memmap, all-NaN,
  ragged lists, and scalar/string `hair` all fail open cleanly with no exception.
- `corneal_strength > 0` alone technically skips the gate, but the early return at `eyes.py:78`
  fires first (0/101 px changed). Cosmetic ordering wart, not exploitable.
- Test coverage gaps beyond B3: no test uses a real `__slots__` `FaceRegions`; no isolated test
  for `_HAIR_OVER_IRIS`, `_HAIR_OVER_EYE`, `eye_to_iris_area`, or `iris_inside_eye`; no coverage
  of shape mismatch, `None` iris, NaN masks, feathered (non-binary) masks, or the `copy.copy`
  fail-open path.
