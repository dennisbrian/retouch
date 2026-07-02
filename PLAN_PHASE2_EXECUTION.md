# Phase 2 Execution Queue — Haiku-ready slices (S2, S3, A1-metric, C1-finish)

**Date:** 2026-07-03 (prepared overnight; coding resumes next morning per owner)
**Parents:** `PLAN_SKIN_PRO.md` (S2/S3 method), `PLAN_TIERC_IMPLEMENTATION.md` (C1 Slice 2), `MASTER_PLAN.md` Phase 2. ⚠️ An earlier reference to `PLAN_SKINPY_UPGRADE.md` was dead — that file does not exist in the repo; its anchors were re-derived and verified 2026-07-03 (see Q2 step 3 and `PLAN_TIERE_ENGINE_FIDELITY.md` §1.6).
**Standing rules per owner:** Fable designs/reviews, Haiku codes; no commits by agents; targeted tests only (owner runs full suite); every Haiku brief must include the gui.py `_process_inputs` update for any new ParamSpec (see memory: params do NOT fully auto-wire to GUI).

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

## Q2 — S2 micro dodge & burn (~2–3 days of Haiku slices)
Method per `PLAN_SKIN_PRO.md` §S2. Anchors verified against HEAD `ea6f835` on 2026-07-03:
1. Shared helpers in `retouch/skin.py`: `_blotch_bandpass(L, face_width)` (DoG at σ fw/40 vs fw/12) and `_edge_protect(lab)` (inverse gradient magnitude × `_get_highlight_protection` — now at skin.py:777, not 709).
2. `SkinProcessor.micro_dodge_burn(img_bgr, skin_mask, strength, face_width)`: `L ← L − DoG × (strength/100) × 0.85 × skin_mask × edge_protect`; no blur anywhere; clip; blend_masked.
3. Call site: `_process_face_core` **after frequency combine / before equalize** (i.e., before the `ctx.equalize > 0` block at perf_optimizations.py:298 ✅ verified; `face_width` in scope from perf_optimizations.py:170 ✅).
4. ParamSpec `micro_dodge_burn` (0–100, default 0, recipe key `skin.micro_db`) + engine.py process() signature + ProcessingContext + **gui.py `_process_inputs` placeholder** (the full checklist — this is the param whose omission crashed the GUI last time).
5. Tests: flat-gradient invariance (linear ramp → output ≈ input, no banding); pore preservation (high-band energy Δ ≤ 2%); blotch_std strictly decreases on synthetic blotchy skin, monotonic in strength; strength-0 byte-identical.
6. Acceptance on real photo: DSCF8028 (gradio cache) + one studio portrait — blotch_std drop reported, visual crop for owner review.

## Q3 — S3 color-blotch evening (~1 day, rides Q2)
Same band-pass on **a-channel**: `a ← a − DoG_a × strength × skin_mask × edge_protect` (and optionally b at 0.5 weight). Excludes lips (regions.lips mask subtracted). ParamSpec `redness_even` (recipe key `skin.redness_even`) + full wiring checklist. Runs adjacent to micro_dodge_burn (immediately after it). Tests: local red patch on synthetic skin → a-channel band std drops; global mean a unchanged (±0.5); lips untouched.

## Q4 — C1 finish (Slice 2 of `PLAN_TIERC_IMPLEMENTATION.md`) (~1–2 days)
⚠️ Two audit findings from `PLAN_TIERE_ENGINE_FIDELITY.md` §1.4 interact with Q4: (a) `harmonize_neck` only fires on `whiten != 0 or equalize > 0` (perf_optimizations.py:344) — zeroing equalize is safe today only because `porcelain_unified_v1` keeps `rosy: 0.15`; extend the gate to the C1 params with or before any recipe that drops both. (b) `unify_hue_line`'s binary eligibility gates (skin.py:750-753) risk contour seams as C1 strengths rise — smoothstep them as an optional Q4 rider (~4 lines each).
1. `whiten(hue_stable=True)` path: L-lift in OKLCh with C, h held; tone shifts re-expressed as bounded OKLCh nudges; default remains False until corpus A/B. (OKLCh imports already present at skin.py:14-22 — wiring is light.)
2. Adopt in recipes: `porcelain_unified_v1` gets `"whiten_hue_stable": true` (needs a ParamSpec or recipe-level flag — prefer ParamSpec `whiten_hue_stable` bool-as-int 0/1 for registry uniformity) — full wiring checklist again.
3. Lower `equalize` in `porcelain_unified_v1` to 0 (C1 now proven on owner's corpus) — pending owner's visual verdict first.
4. `skin.locus` recipe override: nested dict pass-through (like the existing nested `skin` block handling) → `unify_hue_line(locus=...)`. Not a ParamSpec (non-scalar); recipe-only.

---

## Review gates (Fable, after each Q)
Diff review line-by-line (Haiku has claimed "no deviations" falsely twice: engine-signature miss, weakened relight agreement test — trust nothing unverified); independent re-run of the targeted tests; visual crop check on DSCF8028 for Q2/Q3; alignment check `len(_process_inputs) == len(PROCESS_INPUT_KEYS)` after any param addition.

## Gap fillers (between review gates, in this order)
While Fable reviews a finished Q-slice, Haiku takes: **E3.5** (B&W-mixer `_DEFAULTS`, trivial) → **E4** (dead-Numba removal + pool payload slimming, 1–2 d) → **F8.0** (proxy detail reinjection, 1 d — needs owner visual QA, so it gets its own review gate). E3.3/E3.4 are NOT gap fillers — they land inside the Q4 slice (see Q4 header).

## Parking lot (next planning session)
- ~~Fold Tier H / Tier E into MASTER_PLAN~~ **Adopted 2026-07-03** — slots recorded in `MASTER_PLAN.md` (E riders + F8.0 in Phase 2 gaps, E1/E2 with F8/F1, H3 Phase 3, H0–H2/H4 Phase 4).
- S2/S3 done → A2 competitive tuning needs the A1 corpus (owner: Evoto/R4me/PixCake trials still outstanding).
- Relight v2 leftovers (kelvin tint, neck extension, softbox param) land with C2 (Slice 3/3b).
- Commit split proposal for the current uncommitted batch (owner will run full suite first).
