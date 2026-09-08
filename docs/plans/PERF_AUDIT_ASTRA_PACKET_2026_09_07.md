# Astra review packet — native performance audit

Companion to `PERF_AUDIT_NATIVE_2026_09_07.md`. Date: 2026-09-07.

**Scope: do NOT redo the performance audit.** Five claims below carry the
recommendation. Each names the specific thing to falsify. If a claim survives,
say so briefly; if it breaks, that changes the implementation order.

Reproduce anything with the baseline environment — bare `python3` gives a
different cv2 and no detection, which invalidates comparisons:

```
RETOUCH_CACHE_DIR=/private/tmp/retouch RETOUCH_GPU=0 \
RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python
```

---

## Claim 1 — bbox-slicing the component loop is bit-exact, not just faster

* **Claim:** replacing `comp_mask = labels == i` with a slice from the existing
  `stats[i]` bbox yields an identical pixel set per component, at ~6500× lower
  cost (measured 77.0 s → 0.012 s for 1500 components at 26 MP).
* **Code:** `retouch/freckle.py:327` (`labels == i`), consumed by
  `_classify_anomaly` at `retouch/freckle.py:146`. Bbox source:
  `connectedComponentsWithStats` at `retouch/freckle.py:141`.
* **Evidence:** `_classify_anomaly` 30.7 s tottime / 1535 calls;
  `classify_anomalies` 32.2 s cumulative in the DSCF2650 profile.
* **Proposed:** slice `labels` and `lab` by bbox, compare within the slice.
* **Falsify this:** find a case where the bbox slice does **not** reproduce the
  full-canvas pixel set. Specifically — is `CC_STAT_WIDTH/HEIGHT` guaranteed to
  bound the component tightly for 8-connectivity, and does any downstream
  consumer depend on `comp_mask` being full-canvas shaped (centroid coordinates,
  bbox offsets) rather than slice-local? An off-by-one or a coordinate frame
  left in slice-local space is the realistic failure.

---

## Claim 2 — `_relative_features` recomputes loop-invariant work 1437×

* **Claim:** the `cvtColor` and the reference median/MAD are identical on every
  call and can be hoisted into `detect_marks`, retiring **~60 s**. The win is
  overwhelmingly the **median/MAD (57.6 s), not the cvtColor (8.2 s)**.
* **Code:** `retouch/marks.py:207` (`_relative_features`, does
  `cv2.cvtColor(img_bgr, ...)` then medians over `reference_mask`), called at
  `retouch/marks.py:299` inside the per-classification loop.
* **Evidence:** measured in-pipeline by wrapping the function during a real
  render — 1437 calls, crop shapes 1269×1070 and 5888×3385, reference mask
  ~668 790 px: cvtColor 8.2 s + median/MAD 57.6 s = **65.8 s** invariant.
  Profile independently attributes 62.7 s cumulative / 39.6 s tottime.
* **Proposed:** compute the reference median/MAD (and LAB) once in
  `detect_marks`, pass them in. Do the median/MAD first.
* **Falsify this:** confirm `img_bgr` and `normalized_face` are genuinely
  invariant across the loop — is `detect_marks` ever called with a *mutated*
  image between iterations, or `_relative_features` called from any other site
  with a different reference mask? Note there are **three** `detect_marks` call
  sites (`perf_optimizations.py:420`, `:627`, `:1068`) passing different canvases
  (`canvas_original` vs `canvas`), so the hoist must be per-call-site, not
  global/module-level caching. Also check the `mel_map`/`hb_map` branch, which I
  did not read in full — if those vary per classification the hoist is partial.

---

## Claim 3 — cross-recipe detection caching is NOT worth doing

* **Claim:** despite `recipe_batch.py` re-running detection per recipe arm
  (25× instead of 5×), the saving is ~1.2 s out of 2596 s, and the F8.2 cache
  guard makes reuse structurally unsafe. Recommend against.
* **Code:** `scripts/batch/recipe_batch.py:121` (`engine.process(source,
  recipe=recipe)`, no `face_contexts=`); guard at `retouch/engine.py:2242–2249`
  (discards proxy-built contexts as "invalid for native face crops"); native
  contexts built at `retouch/engine.py:2293`.
* **Evidence:** measured `detect` 0.05 s + `segment_person` 0.01 s on a 26 MP
  frame = 0.02% of a 324 s render. `timings["detection"]` = 0.02 s in-pipeline.
* **Proposed:** no change. Spend the effort on C1/C2/C3 instead.
* **Falsify this:** the load-bearing assumption is that **BiSeNet region parsing
  is not a hotspot**. I could not time `parse()` standalone (signature mismatch)
  and it did not appear in the top-25 cumulative list — but if parsing is
  recipe-independent and materially expensive on multi-face images, cross-recipe
  reuse of *native* contexts becomes attractive and this claim flips. Check a
  multi-face frame (DSCF2810 or DSCF2650's larger arms) before accepting.

---

## Claim 4 — elliptical morphology, not blur, is the top primitive cost

* **Claim:** `dilate` costs 82.2 s vs `GaussianBlur` 53.5 s, and the dilate cost
  is concentrated in two callers with IED-scaled elliptical kernels. ROI-confining
  them is bit-exact (verified: `np.array_equal` True, 5.7× faster).
* **Code:** `retouch/hairwork.py:623 _flyaway_mask` (42.6 s of dilate;
  `band_w = face_width * 0.06` at hairwork.py:655),
  `retouch/undereye.py:104 build_undereye_support` (32.1 s; five full-canvas
  dilates + a full-canvas `distanceTransform`).
* **Evidence:** kernel scaling measured at 26 MP — ellipse 5.23/16.92/36.01 s
  for face_width 700/1200/1800 vs rect 0.141/0.243/0.430 s (37–84×).
* **Proposed:** C3/C4a ROI-confine (bit-exact). C4b rect substitution is a
  **different operator** and is explicitly *not* recommended without owner review.
* **Falsify this:** the ROI pad must be ≥ the largest kernel radius used
  downstream in the same function. I flagged the `distanceTransform` at
  undereye.py:167 as the likely place "bit-exact" breaks — distances are global,
  so a crop could change them — then **tested it and it holds**: with a pad of
  `3 × ext_px` the ROI distances matched the full-canvas distances exactly
  (max abs diff 0.0), because the taper saturates via
  `clip(dist / ext_px, 0, 1)` and distances beyond `ext_px` are discarded.
  Taper max diff 3e-08 (float assoc. only).
  **What remains to falsify:** that saturation argument is the entire basis for
  the label. Confirm no *other* consumer of the raw `dist` exists, and that the
  pad is derived from **the max of every kernel radius used in the function** —
  not just `ext_px`. These are separately scaled constants:
  `ext_px = ied * _UE_EXTEND_DOWN` (0.28, undereye.py:56/133) vs the largest
  kernel `_ellipse(feather_r * 3)` (undereye.py:158) where
  `feather_r = ied * _UE_FEATHER` (0.06, undereye.py:60/152) — a halo of
  0.18·ied. **0.18 < 0.28, so an `ext_px`-sized pad covers the halo today, but
  only by coincidence of two unrelated constants.** Luna should size the pad
  from an explicit `max(ext_px, feather_r * 3, ring radii)` so a future tweak to
  `_UE_FEATHER` cannot silently break exactness; also confirm the ring dilations
  (undereye.py:176–177, `_UE_RING_OUTER + _UE_RING_GAP`) fit inside it. Finally,
  confirm ROI clamping at image edges — an eye near the frame border is
  untested geometry.

---

## Claim 5 — memory caps batch parallelism at 2 workers, not 4

* **Claim:** peak RSS is 5.64 GB for a single 26 MP render on an 18 GB machine,
  so 2 processes (~11.3 GB) is the safe ceiling; 4 (~22.6 GB) would swap or OOM.
* **Code:** sequential loop at `scripts/batch/recipe_batch.py:96–135`; one shared
  `RetouchEngine` created at line 93.
* **Evidence:** measured peak RSS 5.64 GB (cProfile run) and 4.87–5.14 GB
  (timings run); `hw.memsize` = 18 GB. One float32 BGR buffer at this size is
  297 MB; one mask is 99 MB.
* **Proposed:** `ProcessPoolExecutor`, 2 workers, one engine per worker
  (threads are unsafe here — see CLAUDE.md MediaPipe teardown-deadlock entries).
* **Falsify this:** is 5.64 GB the true peak, or did cProfile inflate it? RSS is
  also a high-water mark that does not fall back, so two workers may not actually
  peak *simultaneously* — the real ceiling could be higher (3 workers viable) or
  lower if a heavier image than DSCF2650 exists in a production batch. Worth one
  check before Luna sizes the pool.

---

## What I deliberately did not do

* Did not re-run the 43-minute baseline batch under cProfile.
* Did not profile the other 23 renders (one full cProfile + one stage-level
  two-recipe run only).
* Did not implement or prototype any fix in `retouch/`.
* Made no image-quality claims and changed no recipes or defaults.
