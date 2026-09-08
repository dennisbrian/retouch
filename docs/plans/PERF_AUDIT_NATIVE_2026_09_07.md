# Native-resolution performance audit (P1–P6 five-recipe baseline)

Date: 2026-09-07 (Asia/Kuala_Lumpur)
Scope: profiling and recommendations only. **No production behavior was changed.**
Baseline: `docs/plans/REPORT_P1_P6_FIVE_RECIPE_RUN_2026_09_07.md`

---

## 1. Method

Profiling used the baseline's exact environment and inputs — no new corpus, no
recipe edits:

```
RETOUCH_CACHE_DIR=/private/tmp/retouch RETOUCH_GPU=0 \
RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python
```

Instrumentation lived entirely in throwaway scripts under `/tmp/perf_audit/`;
nothing was added to `retouch/`. Two sources were used:

1. The engine's existing `result.timings` dict (stage-level, already public).
2. `cProfile` on one native render (`DSCF2650.jpg`, 6240×4160,
   `studio_headshot_protected_v1`).

**Profile fidelity check.** The profiled render took 323.6 s vs the baseline
manifest's 325.0 s for the same image/recipe (0.4% apart), and reported
`face_count=1`. Face detection was verified live (1 face at native, 1 at proxy),
so the per-face pipeline genuinely executed — this is not a `_no_face_fallback`
profile.

**Path confirmation.** The manifest records `max_dim: null` and `scale: 1.0`,
and the profile enters `_process_native_faces` (engine.py:2123). This is the
F8.2 `quality="full"` native path. CLAUDE.md's ">25 min, never observed
completing" note for 24 MP `quality="full"` is **stale** — 26 MP renders
complete in 27–325 s depending on recipe. Recommend correcting that note.

---

## 2. Measured stage timings

Whole-batch shape (25 renders, from the baseline manifest):

| | mean s/img | min | max |
|---|---:|---:|---:|
| all arms | 103.9 | 27.1 | 325.0 |

Per-image totals across the five arms vary **4.4×** (DSCF2843 208 s vs DSCF2650
916 s) at *identical* 26 MP dimensions. Cost tracks face size/count and how many
per-face ops a recipe enables — not pixel count.

Stage timings, `studio_headshot_protected_v1` on DSCF2650 (323.6 s wall):

| Stage | Seconds | % wall |
|---|---:|---:|
| `per_face` | 297.0 | 88.1% |
| `grading` | 4.4 | 1.3% |
| `reshape` | 2.7 | 0.8% |
| `finish` | 0.7 | 0.2% |
| `detection` | 0.02 | 0.0% |

For contrast, `cinema_grade_v1` on the same image (62.4 s): `per_face` 29.6 s
(47.5%), `grading` 14.2 s (22.8%). **`per_face` dominates in both arms.**

### Ranked hotspots inside `per_face` (cumulative seconds, one render)

| Cum s | Site | Nature |
|---:|---|---|
| 95.3 | `marks.py:259 detect_marks` (4 calls) | Python per-blob loop |
| 63.0 | `hairwork.py:777 remove_flyaways` | full-res elliptical morphology |
| 62.7 | `marks.py:207 _relative_features` (1437 calls) | loop-invariant work per blob |
| 38.0 | `undereye.py:104 build_undereye_support` (2 calls) | full-canvas dilate ×5 |
| 32.2 | `freckle.py:271 classify_anomalies` (4 calls) | Python per-blob loop |
| 30.7 | `freckle.py:146 _classify_anomaly` (1535 calls) | full-canvas boolean per blob |
| 30.2 | `qa_detectors.py run_qa_with_evidence` | post-render QA analysis |
| 23.4 | `chromophore_v2.py:378 reduce_hemoglobin_variance` | full-res colour work |

Primitive totals for the same render: `dilate` **82.2 s**, `GaussianBlur`
**53.5 s**, `erode` 10.3 s.

`dilate` attribution by caller: `hairwork._flyaway_mask` 42.6 s,
`undereye.build_undereye_support` 32.1 s, `undereye._dilate_down` 4.0 s,
`perf_optimizations._process_face_core` 2.8 s.

---

## 3. Root causes (measured, not inferred)

### R1 — Elliptical structuring elements are O(kernel area) and unseparable

Measured on a 6240×4160 float32 mask:

| face_width | band_w | kernel | ELLIPSE | RECT | ratio |
|---:|---:|---:|---:|---:|---:|
| 700 | 42 | 85 px | 5.23 s | 0.141 s | 37× |
| 1200 | 72 | 145 px | 16.92 s | 0.243 s | 70× |
| 1800 | 108 | 217 px | 36.01 s | 0.430 s | 84× |

OpenCV optimises rectangular morphology to O(1)/pixel but runs ellipses
naively. Kernel radii are derived from IED / `face_width` (e.g.
`hairwork.py:655 band_w = face_width * 0.06`), so at native resolution the
kernels become enormous. This is the single largest primitive cost.

### R2 — Full-canvas morphology for spatially tiny masks

`undereye.build_undereye_support` issues five full-frame `cv2.dilate` calls plus
a full-frame `distanceTransform` for a mask confined to a small eye ROI. Cost is
paid over all 26 MP regardless of support extent.

### R3 — Loop-invariant work inside per-blob loops

`marks.py:207 _relative_features` runs a `cvtColor` and recomputes the
reference median/MAD over the face mask on *every one of 1437 calls*. Both are
identical across iterations.

Measured directly by wrapping the function during a real render (1437 calls,
crop shapes 1269×1070 and 5888×3385, reference mask ~668 790 px):

| Invariant work | Total over 1437 calls |
|---|---:|
| `cvtColor` | 8.2 s |
| reference median/MAD | **57.6 s** |
| **total retired by hoisting** | **65.8 s** |

Note the cost is dominated by the **median/MAD**, not the colour conversion —
`np.median` over ~669 k reference pixels, twice per call. `detect_marks`
receives a per-face crop (not the full 26 MP frame), which is why the
colour-conversion share is modest.

### R4 — Full-canvas boolean indexing per component

`freckle.py:327` does `comp_mask = labels == i` — a full 26 MP comparison per
component, up to the 2000-component cap. `stats` from
`connectedComponentsWithStats` already carries each component's bbox.

**On the 2000-component cap:** the cap bounds *count*, not cost. Each capped
iteration still scans the full canvas, so the cap is what makes the worst case
finite rather than what makes it cheap. The baseline's
`Freckle: 2000+ components detected` warning marks a render paying ~2000 ×
26 MP scans.

---

## 4. Recommendations

Ordered by (value ÷ risk). Equivalence classes:
**[BIT-EXACT]** = output byte-identical; **[TOLERANCE]** = numerically close,
requires a stated tolerance and a viewed render.

---

### C1 — bbox-slice the component loops **[BIT-EXACT]**

`freckle.py:327`, and the same pattern in the marks path. Replace
`labels == i` over the full canvas with a slice using the existing
`stats[i]` bbox, then compare within the slice.

* **Expected speedup:** **~10% of a heavy render.** The bound is the profile's
  30.7 s (`_classify_anomaly`) + 32.2 s (`classify_anomalies`) — *not* the
  headline ratio. The isolated micro-benchmark gave bit-identical output at
  ~6500× (77.0 s → 0.012 s for 1500 components at 26 MP), but that is the
  primitive ratio in isolation and **overstates the render-level win**; the real
  loop does other work per component. Luna should size expectations from the
  ~63 s profile attribution.
* **Memory:** strictly lower (slices, not full-canvas temporaries).
* **Equivalence risk:** very low. Pixel set per component is unchanged by
  construction; verified identical in benchmark.
* **Luna validates:** `tests/test_golden_pipeline_face.py` hashes must be
  **unchanged**. Add a unit test asserting old/new component statistics match
  exactly on a synthetic multi-blob mask, including a blob touching the image
  border.

---

### C2 — hoist loop-invariants out of `_relative_features` **[BIT-EXACT]**

`marks.py:207`. Compute the LAB image and the reference median/MAD **once** in
`detect_marks` (marks.py:259) and pass them in; keep the per-blob math untouched.

* **Expected speedup:** measured in-pipeline, not extrapolated — the invariant
  work totals **65.8 s** over the 1437 real calls (median/MAD 57.6 s +
  `cvtColor` 8.2 s), against `_relative_features`' 62.7 s cumulative profile
  attribution. → **~15–19% of a heavy render.** The two figures bracket each
  other because the wrapper adds its own measurement overhead; treat ~60 s as
  the realistic retirement.
* **Priority within C2:** hoist the **median/MAD first** (57.6 s, 88% of the
  win). The `cvtColor` hoist is worth only 8.2 s.
* **Memory:** one float32 LAB buffer for the crop (not the full frame — see R3),
  replacing 1437 sequential allocations. Peak is not expected to rise.
* **Equivalence risk:** low, but **not zero**: this is a genuine refactor of a
  signature. The arithmetic is unchanged; the risk is a plumbing mistake, which
  tests catch.
* **Luna validates:** golden face hashes unchanged. Assert the hoisted LAB and
  median/MAD equal the per-call values for several blobs (`np.array_equal`).

---

### C3 — ROI-confine under-eye morphology **[BIT-EXACT]**

`undereye.py:104`. Compute the mask bounding box, pad by the largest kernel
radius used, run all dilations/distance transform inside that ROI, write back
into a zeroed full-frame canvas.

* **Expected speedup:** verified **bit-exact** and 5.7× on the isolated
  operation (16.93 s → 2.96 s). Against `build_undereye_support`'s 38.0 s →
  **~8–10% of a heavy render.**
* **Memory:** strictly lower.
* **Equivalence risk:** low *provided the pad is ≥ the largest kernel radius and
  the ROI is clamped to image bounds*. An under-sized pad silently truncates the
  halo — this is the one detail to review carefully.
* **Luna validates:** golden hashes unchanged. Add a test comparing ROI vs
  full-frame output with `np.array_equal` for an eye mask **near an image edge**
  (the clamping case) and one mid-frame.

---

### C4 — decompose large elliptical kernels **[TOLERANCE — needs care]**

`hairwork.py:_flyaway_mask` (42.6 s of dilate) and other IED-scaled ellipses.
Two options, and they are **not** equally safe:

* **C4a, ROI-confine first [BIT-EXACT]** — same trick as C3, applied to the
  silhouette band. Do this before considering C4b.
* **C4b, substitute kernel shape [TOLERANCE]** — a rectangle is 37–84× faster
  but is a *different morphological operator*; it changes the mask, hence the
  render. Only viable if a shape change is acceptable here, which this audit
  does **not** establish.

* **Expected speedup:** C4a alone should retire a large share of 63.0 s
  `remove_flyaways` → **~10–15%**. C4b is additional but gated on quality review.
* **Memory:** C4a lower; C4b unchanged.
* **Equivalence risk:** C4a low; **C4b high — visible mask change on hair
  silhouettes.**
* **Luna validates:** C4a — golden hashes unchanged, `np.array_equal` vs
  full-frame. C4b — must not be validated by hashes alone; requires viewed
  native renders on hair-heavy frames (DSCF2650, DSCF2810) reviewed by the owner
  against the current output.

---

### C5 — approximate large-sigma Gaussian blurs **[TOLERANCE]**

Downsample → blur at reduced sigma → upsample. Measured at 26 MP:

| sigma | exact | approx | speedup | max err |
|---:|---:|---:|---:|---:|
| 8 | 0.18 s | 0.01 s | 14.6× | **4.90 DN** |
| 20 | 0.38 s | 0.01 s | 56.3× | 1.27 DN |
| 60 | 1.42 s | 0.02 s | 76.0× | 0.23 DN |

* **Expected speedup:** GaussianBlur totals 53.5 s but is spread across ~20
  call sites with no dominant one; realistic gain **~5–8%**, and only if applied
  to the large-sigma sites (relight `sculpt` 7.2 s, `lighting._shading_estimate`
  5.8 s, `skin._blotch_bandpass` 4.2 s).
* **Memory:** lower (blur runs on a 1/16-area buffer).
* **Equivalence risk:** **moderate and sigma-dependent.** Error is negligible at
  σ≥20 but reaches ~4.9 DN at σ=8 — visible in smooth gradients. Apply only
  above a sigma floor; never blanket-replace.
* **Luna validates:** golden hashes **will change** — this needs an explicit
  tolerance (suggest max ≤1 DN) plus viewed renders. Gate strictly on sigma.

---

### C6 — image-level parallelism **[BIT-EXACT per image]**

The batch loop (`scripts/batch/recipe_batch.py:96`) is fully sequential.

* **Measured constraint:** peak RSS **5.64 GB** for one 26 MP render; system RAM
  **18 GB**. That permits **2 workers** (~11.3 GB), not 4 (~22.6 GB, would swap
  or OOM).
* **Expected speedup:** ~1.8× wall clock on batches at 2 workers.
* **Memory:** roughly doubles peak. This is the main tradeoff.
* **Equivalence risk:** low per image (separate processes, no shared mutable
  state), but MediaPipe/ONNX teardown across processes is a known hazard in this
  repo (see CLAUDE.md teardown-deadlock entries). Use `ProcessPoolExecutor` with
  one engine per worker, never threads.
* **Luna validates:** byte-compare parallel vs sequential outputs for all 25
  renders. Confirm clean teardown (no hung workers) and record peak RSS.

---

### C7 — make QA optional for batch renders **[BIT-EXACT for pixels]**

`run_qa_with_evidence` costs **30.2 s (9.3%)** and produces diagnostics, not
pixels.

* **Expected speedup:** ~9% when disabled.
* **Equivalence risk:** none for image output; **it removes QA evidence**, which
  this repo treats as a first-class artifact. Propose as an opt-out flag, not a
  new default.
* **Luna validates:** confirm output images are byte-identical with QA on vs
  off, and that QA remains on by default.

---

### Not recommended: cross-recipe caching of detection

The intuitive win — `recipe_batch.py:121` calls `engine.process(source,
recipe=recipe)` in the recipe loop with no `face_contexts=`, so detection
appears to run 25× where 5 would do — **does not pay off**, and the audit
should say so plainly.

* **Measured:** `detect` 0.05 s + `segment_person` 0.01 s = **0.06 s**, i.e.
  0.02% of a 324 s render. Eliminating 20 redundant detections saves ~1.2 s of
  2596 s.
* **Structural blocker:** `_process_native_faces` (engine.py:2242–2249)
  *deliberately discards* supplied `face_contexts` because they were built at
  proxy resolution and are invalid for native face crops. Native contexts are
  produced (`built_contexts`, engine.py:2293) but the cache contract does not
  distinguish proxy-built from native-built.
* **The expensive part is not shareable anyway:** the 297 s of `per_face` is
  per-recipe op work (marks, flyaways, chromophore), which differs by recipe by
  construction. Only the BiSeNet region parse is recipe-independent, and it did
  not surface as a hotspot.

Rewiring the cache contract would be an invasive change to a deliberate
correctness guard for ~0.05% gain. **Recommend against.**

---

## 5. Recommended implementation order for Luna

Bit-exact work first — it is validated by unchanged golden hashes, so it can
land with high confidence and reduces the noise floor for anything riskier.

| # | Item | Class | Est. gain | Confidence |
|---|---|---|---:|---|
| 1 | C1 bbox-slice component loops | BIT-EXACT | ~10% | high |
| 2 | C2 hoist `_relative_features` invariants | BIT-EXACT | ~15–19% | high |
| 3 | C3 ROI-confine under-eye morphology | BIT-EXACT | ~8–10% | high |
| 4 | C4a ROI-confine flyaway morphology | BIT-EXACT | ~10–15% | medium |
| 5 | C6 two-worker batch parallelism | BIT-EXACT | ~1.8× batch | medium |
| 6 | C7 optional QA flag | opt-in | ~9% | high |
| 7 | C5 / C4b tolerance-class changes | TOLERANCE | ~5–8% | **owner review first** |

Items 1–4 are independent and individually revertible. Cumulatively they target
roughly **40–50% of single-render wall time** with byte-identical output; with
C6 on top, the 43-minute baseline batch would fall to roughly 12–15 minutes.

**Standing validation rule for items 1–4:** if
`tests/test_golden_pipeline_face.py` hashes move, the change is not bit-exact —
stop and investigate rather than re-baselining the snapshots.

---

## 6. Honest limitations

* One image × one recipe was cProfiled end-to-end. Stage-level `timings` were
  captured for a second recipe; the other 23 baseline renders were not
  re-profiled (the full batch is ~43 min, and re-running it under cProfile was
  judged not worth the cost).
* Speedup percentages are derived from measured primitive costs and profile
  attribution, not from implemented patches. They are estimates.
* **Systematic bias in the micro-benchmarks — read this before quoting any
  headline ratio.** The synthetic benchmarks (kernel scaling, bbox slicing, blur
  approximation) ran on full 26 MP arrays and therefore **overstate per-call cost
  relative to the real pipeline**, which mostly operates on per-face crops
  (measured: 1269×1070 and 5888×3385, not 6240×4160). This inflated both C1's
  6500× and an earlier C2 estimate of ~397 s. Both have been re-anchored to
  profile/in-pipeline measurements. **Where a benchmark ratio and a profile
  number disagree, the profile number governs.**
* C2 is the one figure measured directly in-pipeline (65.8 s, by wrapping
  `_relative_features` during a real render) rather than extrapolated. That
  instrumentation was throwaway and added its own overhead, so ~60 s is the
  realistic figure.
* No image-quality claims are made. No recipe, default, or production code was
  modified by this audit.
