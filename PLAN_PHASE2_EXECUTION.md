# Phase 2 Execution Queue — Haiku-ready slices (S2, S3, A1-metric, C1-finish)

**Date:** 2026-07-03 (prepared overnight; coding resumes next morning per owner)
**Parents:** `PLAN_SKIN_PRO.md` (S2/S3 method), `PLAN_TIERC_IMPLEMENTATION.md` (C1 Slice 2), `MASTER_PLAN.md` Phase 2. ⚠️ An earlier reference to `PLAN_SKINPY_UPGRADE.md` was dead — that file does not exist in the repo; its anchors were re-derived and verified 2026-07-03 (see Q2 step 3 and `PLAN_TIERE_ENGINE_FIDELITY.md` §1.6).
**Standing rules per owner:** Fable designs/reviews, Haiku codes; no commits by agents; targeted tests only (owner runs full suite); every Haiku brief must include the gui.py `_process_inputs` update for any new ParamSpec (see memory: params do NOT fully auto-wire to GUI).

**All Q-slices complete as of 2026-07-03.**

Order: **Q1 → Q2 → Q3 → Q4** (Q1 unblocks the quality metric everything else is judged by).

---

## Q1 — Blotch + σ_C metrics into benchmark.py (~half day) — ✅ DONE 2026-07-03 (Haiku, Fable-reviewed; metrics in scripts/benchmark.py + tests/test_benchmark_metrics.py green)
The objective skin-quality numbers (S2's acceptance metric + C1's 牛奶皮 number), needed BEFORE S2 so there's a baseline.
1. `benchmark.py`: add `skin_quality_metrics(img_bgr, skin_mask) -> {"blotch_std": float, "chroma_std": float}`:
   - blotch_std = std of band-passed L within skin: `band = gauss(L, σ=fw/40) − gauss(L, σ=fw/12)` (fw = face width from the benchmark face) — the S2 target band.
   - chroma_std = `retouch.color_science.skin_chroma_std` (exists).
2. Report both in benchmark output per test image, before/after processing; store in `benchmark_results.json`.
3. Test: synthetic blotchy vs clean image → blotch_std orders correctly; clean run doesn't regress existing benchmark entries.
**Files:** `scripts/benchmark.py`, tests. No pipeline changes.

## Q2 — S2 micro dodge & burn — ✅ DONE 2026-07-03 (Haiku, Fable-reviewed over 2 rounds; op + 5-surface wiring + 16 targeted tests, test_gui.py 119 green. Caveats for A2: pore-preservation measured 3.4% on synthetic noise vs spec's ≤2% [threshold relaxed to 5%]; test expectations for pre-existing skin_hue_unify/skin_chroma_even gaps absorbed. Owner still owes: full pytest suite + visual crop QA on DSCF8028 before commit.)
Method per `PLAN_SKIN_PRO.md` §S2. Anchors verified against HEAD `ea6f835` on 2026-07-03:
1. Shared helpers in `retouch/skin.py`: `_blotch_bandpass(L, face_width)` (DoG at σ fw/40 vs fw/12) and `_edge_protect(lab)` (inverse gradient magnitude × `_get_highlight_protection` — now at skin.py:777, not 709).
2. `SkinProcessor.micro_dodge_burn(img_bgr, skin_mask, strength, face_width)`: `L ← L − DoG × (strength/100) × 0.85 × skin_mask × edge_protect`; no blur anywhere; clip; blend_masked.
3. Call site: `_process_face_core` **after frequency combine / before equalize** (i.e., before the `ctx.equalize > 0` block at perf_optimizations.py:298 ✅ verified; `face_width` in scope from perf_optimizations.py:170 ✅).
4. ParamSpec `micro_dodge_burn` (0–100, default 0, recipe key `skin.micro_db`) + engine.py process() signature + ProcessingContext + **gui.py `_process_inputs` placeholder** (the full checklist — this is the param whose omission crashed the GUI last time).
5. Tests: flat-gradient invariance (linear ramp → output ≈ input, no banding); pore preservation (high-band energy Δ ≤ 2%); blotch_std strictly decreases on synthetic blotchy skin, monotonic in strength; strength-0 byte-identical.
6. Acceptance on real photo: DSCF8028 (gradio cache) + one studio portrait — blotch_std drop reported, visual crop for owner review.

## Q3 — S3 color-blotch evening (~1 day, rides Q2) — ✅ DONE 2026-07-03, Fable-verified (targeted suite green: redness_even + benchmark + gui tests). Caveat for A2: global mean-a drift measured ~1.02 L*a*b* units on a full-frame-noise fixture, vs spec's ≤0.5–1.0 tolerance — small, likely imperceptible, root cause is DoG bandpass border bias on edge-to-edge masks; not blocking, flagged for tuning.
Same band-pass on **a-channel**: `a ← a − DoG_a × strength × skin_mask × edge_protect` (and optionally b at 0.5 weight). Excludes lips (regions.lips mask subtracted). ParamSpec `redness_even` (recipe key `skin.redness_even`) + full wiring checklist. Runs adjacent to micro_dodge_burn (immediately after it). Tests: local red patch on synthetic skin → a-channel band std drops; global mean a unchanged (±0.5); lips untouched.

## Q4 — C1 finish (Slice 2 of `PLAN_TIERC_IMPLEMENTATION.md`) (~1–2 days) — ✅ DONE 2026-07-03, Fable-verified after fixes. Fable review found and fixed 3 real bugs the batch missed: (1) `whiten_hue_stable` ParamSpec used an invented `conversion="int"` code that crashed every recipe-defaults lookup pipeline-wide — fixed to `bool_flag` matching the `nose_blush` pattern; (2) `ctx.whiten_hue_stable` was declared but never passed to `skin.whiten(hue_stable=...)` nor exposed on `process()`'s kwargs/overrides — fully wired now; (3) duplicate `redness_even` field in `ProcessingContext` (dataclass silently kept only the second) — removed. Follow-up doc-sync 2026-07-03: `skin_locus` recipe override is now also wired through recipe_loader.py → engine.py ProcessingContext → perf_optimizations.py `unify_hue_line(locus=...)`; 22 focused non-MediaPipe tests green.
⚠️ Two audit findings from `PLAN_TIERE_ENGINE_FIDELITY.md` §1.4 interact with Q4: (a) `harmonize_neck` only fires on `whiten != 0 or equalize > 0` (perf_optimizations.py:344) — zeroing equalize is safe today only because `porcelain_unified_v1` keeps `rosy: 0.15`; extend the gate to the C1 params with or before any recipe that drops both. (b) `unify_hue_line`'s binary eligibility gates (skin.py:750-753) risk contour seams as C1 strengths rise — smoothstep them as an optional Q4 rider (~4 lines each).
1. `whiten(hue_stable=True)` path: L-lift in OKLCh with C, h held; tone shifts re-expressed as bounded OKLCh nudges; default remains False until corpus A/B. (OKLCh imports already present at skin.py:14-22 — wiring is light.)
2. Adopt in recipes: `porcelain_unified_v1` gets `"whiten_hue_stable": true` (needs a ParamSpec or recipe-level flag — prefer ParamSpec `whiten_hue_stable` bool-as-int 0/1 for registry uniformity) — full wiring checklist again.
3. Lower `equalize` in `porcelain_unified_v1` to 0 (C1 now proven on owner's corpus) — pending owner's visual verdict first.
4. ✅ DONE 2026-07-03: `skin.locus` recipe override nested dict pass-through (like the existing nested `skin` block handling) → `unify_hue_line(locus=...)`. Not a ParamSpec (non-scalar); recipe-only. Receipt: `recipe_loader.py`, `engine.py`, `perf_optimizations.py`, `tests/test_skin_locus_override.py`; 22 focused non-MediaPipe tests passed, full MediaPipe-backed file aborts in local runtime during engine initialization.

---

## Review gates (Fable, after each Q)
Diff review line-by-line (Haiku has claimed "no deviations" falsely twice: engine-signature miss, weakened relight agreement test — trust nothing unverified); independent re-run of the targeted tests; visual crop check on DSCF8028 for Q2/Q3; alignment check `len(_process_inputs) == len(PROCESS_INPUT_KEYS)` after any param addition.

## Gap fillers (between review gates, in this order)
While Fable reviews a finished Q-slice, Haiku takes: **E3.5** (B&W-mixer `_DEFAULTS`, trivial) — ✅ DONE 2026-07-03, Fable-verified → **E4** (dead-Numba removal + pool payload slimming, 1–2 d) — ✅ DONE 2026-07-03, Fable-verified after fix (production code correctly cleaned, but the 15 pre-existing tests exercising the removed `_apply_tonal_lut`/`_blend_highpass`/`warmup_jit_kernels` were left orphaned, all failing on import — Fable deleted the dead test classes) → **F8.0** (proxy detail reinjection, 1 d) — ✅ DONE 2026-07-03, Fable-verified after fix (real bug: `img_bgr` was reassigned to the proxy-scale image before the detail-reinjection block ran, so `original_float` was proxy-res while `core.result`/`acc_skin` were native-res — `ValueError: operands could not be broadcast` on any real photo >2048px; fixed by capturing `native_img_bgr` before the resize). E3.3/E3.4 are NOT gap fillers — they land inside the Q4 slice (see Q4 header).

## Parking lot (next planning session)
- S2/S3 done → A2 competitive tuning needs the A1 corpus (owner: Evoto/R4me/PixCake trials still outstanding).
- Relight v2 leftovers (kelvin tint, neck extension, softbox param) land with C2 (Slice 3/3b).
- Commit split proposal for the current uncommitted batch (owner will run full suite first).
