# Tier E Plan — Engine & Skin Fidelity Audit (Stages E1–E4)

**⚠️ 2026-07-16 re-verification: most of Part 1's headline findings are now
STALE (fixed since this doc was written) — see §18 below before treating
any citation in this file as current. `_stage_grade` is at `engine.py:3683`
today, not `:1655`; the file has grown from whatever HEAD `ea6f835` was to
4486 lines. Two genuine residual gaps were found and fixed in §18
(`apply_global_bloom`'s internal uint8 quantization; `_no_face_fallback`'s
white-balance/HSL round-trip). Treat this doc as a historical snapshot for
Parts 1.2–1.6 and Part 2 below; §18 is the current-truth appendix.**

**Date:** 2026-07-03
**Type:** Code-audit research + stage definitions (no code). Companion to `PLAN_TIERH_HAIR.md` (same session's planning frontier).
**Question:** Tier H covered the hair gap; what does a deep read of the *existing* core (`engine.py`, `skin.py`, `perf_optimizations.py`, `frequency.py`) reveal that no current plan covers?
**Method:** full read of all four modules (verified 2026-07-03, HEAD = `ea6f835`); every claim below carries a file:line anchor.

---

## Part 1 — Audit findings

### 1.1 🔴 The quantization ledger (flagship finding — the per-face chain is 8-bit Swiss cheese)

**Per-face chain (Stage 2):** every skin op takes a uint8 BGR canvas, converts to its working space (LAB float / LCh / OKLab), operates, and re-quantizes to uint8 via `blend_masked` (`utils.py:200-223` — always returns uint8). With a `porcelain_unified_v1`-class recipe a skin pixel passes through **~7–10 uint8 quantizations before Phase 3 even starts**: `frequency.combine` (uint8 out, `frequency.py:263-266`) → `equalize` (LAB, `skin.py:133`) → `unify_hue_line` (OKLab, `skin.py:696`) → `whiten` (LAB, `skin.py:63`) → `harmonize_neck` (LAB, `skin.py:239`) → plus flatten/quantize/bloom/dodge_burn/local_clarity when active (`perf_optimizations.py:280-396`). Each round-trip adds ±0.5-level truncation noise and one full-crop `cvtColor` + two `astype` passes. On the smooth porcelain gradients this project optimizes for (牛奶皮), accumulated 8-bit steps are exactly the banding class F11 is being built to detect — we manufacture upstream what we plan to detect downstream.

**Phase 3 "float32" stage is largely nominal.** `_run_core_pipeline` converts to float at `engine.py:1064` per F1, but `_stage_grade` then punches back down to uint8 at **16 sites** via `_to_uint8_if_float` (`engine.py:1655-1815`): style transfer, color_ref transfer, white balance, master HSL, tonal curve, grade_stack, 3-way split tone, highlight rolloff, skin_glow diffusion, bloom, glow, white-costume lift, vignette, film grain, negative split tone, B&W mixer. Only contrast/brightness/tonal/sharpen and `grade()` itself are float-native. Worse, three "float" helpers are fake: `_F_adjust_vibrance` (`engine.py:1904-1914`) and `_F_apply_uniform_saturation` (`engine.py:1927-1936`) internally round-trip through uint8, and `_stage_subject_separation` does uint8-LAB even on float input (`engine.py:1548-1561`). And the **no-face path gets zero float benefit at all**: `_no_face_fallback` is called before `to_float` (`engine.py:1024` vs `engine.py:1064`), so `--global-only` batch runs are 100 % uint8-chained.

**Consequence:** MASTER_PLAN's Phase-1 milestone claim ("float32 global pipeline") is currently ~30 % real. The 16-site list above *is* the F1 work inventory — F1's estimate should be checked against it.

### 1.2 🔴 Registry-bypass bug class has a live successor (report, not fixed — standing rule)

- The WB-literal bug in CLAUDE.md's "Outstanding Fixes" **is already fixed** — both sites use `_DEFAULTS` (`engine.py:1144`, `engine.py:1668`). CLAUDE.md is stale.
- **The same class lives on in the B&W channel mixer:** `ctx.bw_channel_mixer_r != 30 / != 59 / != 11` hardcoded at `engine.py:1212-1216` *and* `engine.py:1802-1806`, while `params.py:951/961/971` define defaults 30/59/11. Identical latent desync: currently matching, silently breaks if params.py changes. Fix is the same one-liner pattern as the WB fix, in two places.

### 1.3 🟠 Response-curve defects (sliders that lie)

- **Sharpen slider is inert from 1–60:** `amount = max(1.2, ctx.sharpen / 100.0 * 2.0)` (`engine.py:1839`) — any value 1–60 yields amount 1.2 (and 0→1 is a cliff from off to strong). The slider only responds above 60.
- **Equalize responds quadratically:** the a/b pull is scaled by `s` internally (`skin.py:167`) and the final blend is scaled by `s` again (`skin.py:184`) — effective chroma evening ≈ 0.20·s². Mid-slider is much weaker than it reads. Matters for A2 tuning sweeps and for interpreting Q1's σ_C metric vs slider position.

### 1.4 🟠 C1-era interaction traps (directly relevant to tomorrow's Q4)

- **`harmonize_neck` gate omits the C1 params:** it only runs when `ctx.whiten != 0 or ctx.equalize > 0` (`perf_optimizations.py:344`), with strength `max(abs(whiten), equalize)`. `skin_hue_unify`/`skin_chroma_even` don't trigger or scale it. Today `porcelain_unified_v1` survives via `rosy: 0.15` (`recipes.py:125`), but Q4's plan (equalize→0, whiten→hue-stable) walks straight toward a recipe where the face is C1-unified and the neck silently isn't. Gate should include the C1 params when C1 becomes the primary path.
- **`unify_hue_line` eligibility gates are hard binary steps:** `(C <= 0.18)`, `(L >= 0.25)`, `(L <= 0.95)` (`skin.py:750-753`). On smooth skin gradients these thresholds are iso-contours — one side gets the full hue rotation, the other none → potential visible seams at 100 % zoom as C1 strengths rise. Every comparable mask in skin.py uses soft ramps (`_get_highlight_protection`, shadow_protection). Fix is ~4 lines of smoothstep per gate.

### 1.5 🟡 Smaller findings

- **Smooth mask is eroded 3 px twice:** `_build_smooth_mask` erodes on every call (`perf_optimizations.py:66-67`) and is called twice per face (`perf_optimizations.py:201`, `:216`). At proxy-res small faces (ied ≈ 40 px) the doubled shrink leaves a visible unsmoothed rim at the skin boundary. Intent unclear — one erosion may be intended.
- **Dead Numba kernels, live warm-up cost:** `_apply_tonal_lut` and `_blend_highpass` (`perf_optimizations.py:675-741`) have **zero production callers** (grep-verified); only `warmup_jit_kernels` exercises them, and it runs at every `RetouchEngine()` init (`engine.py:567`). Pure startup cost for unused code — remove or wire.
- **FaceProcessorPool payload bloat:** the full `ctx` is pickled per face (`engine.py:1337-1351`), including `color_ref` when set — a full-size reference image × N faces over IPC. Also `max_workers = os.cpu_count()` (`perf_optimizations.py:540`) vs the documented 4-worker policy.
- **Face and no-face paths render the same recipe in different op order:** no-face applies tonal_curve *before* WB and impact *before* grain/negative-split/B&W (`engine.py:1140-1223`); the face path applies tonal_curve after HSL and impact dead-last after sharpen (`engine.py:1700-1849`). Duplicated ~80-line chain, already diverged — this is the concrete evidence P3 (stage registry) has been waiting for.
- **`blemish.remove` runs *after* smoothing** (`perf_optimizations.py:336`) — pro workflow (and Retouch4me Heal) heals first so detection sees unsmoothed anomalies and inpainting isn't fed pre-blurred context. Not asserted as wrong — flag for an A1-corpus A/B (order swap is trivial to test once the harness exists).

### 1.6 Stale documentation found (fixed in this session's doc updates)

- CLAUDE.md "Outstanding Fixes": WB literals → fixed; `_build_dimensional_mask` WIP → merged (shape param present at `skin.py:31-61`, working tree clean).
- `PLAN_PHASE2_EXECUTION.md` cites **`PLAN_SKINPY_UPGRADE.md` which does not exist in the repo** (grep-verified) as the Q2/Q3 blueprint with "exact anchors". Anchors re-derived and verified in this audit: S2 call site = before `if ctx.equalize > 0:` at `perf_optimizations.py:298`; `face_width` in scope from `perf_optimizations.py:170`; `_get_highlight_protection` now at `skin.py:777` (not 709 — S1 shifted it); OKLCh imports Q4 needs already present at `skin.py:14-22`.

## Part 2 — Stages

### Stage E1 — Float-resident per-face core (~1 week) 🔴 the fidelity unlock
Convert the ROI canvas to float32 LAB **once** at the top of `_process_face_core`, keep every skin op float-LAB in/out (ops that need OKLab/LCh convert from float LAB, not from uint8 BGR), single uint8 conversion at the end. Kills ~7–10 quantizations *and* ~10 full-crop `cvtColor`+`astype` passes per face (perf win rides free). Migration: adapter pattern — ops accept either dtype, migrate one op per commit, golden-image gated (same discipline as P3). Natural window: alongside F8.1/F8.2, which restructure the same code.

### Stage E2 — F1 hole-closing inventory (fold into F1, not a new stage)
The 16 `_to_uint8_if_float` sites + 3 fake-float helpers + the uint8 no-face path (§1.1) become F1's checklist. Each site migrates to a float-native implementation or gets an explicit "inherently uint8 (cv2 LUT path)" annotation. Acceptance: a synthetic smooth-gradient image through a max-stack recipe shows zero new banding steps vs single-op baseline (F11's banding detector, once built, automates this).

### Stage E3 — Response-curve & gate repairs (~2–3 days, can ride the Phase 2 queue)
1. Sharpen mapping: replace `max(1.2, …)` with a monotone map over the full 0–100 range (`engine.py:1839`).
2. Equalize linearization: decide intended response (document s² or make it s), then **re-tune affected recipes** — every recipe with `equalize` shifts if the curve changes; needs Q1's metrics first.
3. `unify_hue_line` smoothstep gates (§1.4) — natural Q4 rider.
4. `harmonize_neck` gate extended to C1 params (§1.4) — should land with or before Q4's recipe change.
5. B&W mixer literals → `_DEFAULTS` (§1.2) — the same commit pattern as the WB fix.

### Stage E4 — Pool & dead-code hygiene (~1–2 days, anytime)
Remove or wire the dead Numba kernels (+ drop warm-up if removed); slim FaceProcessorPool payloads (strip `color_ref`/`face_contexts` from the pickled ctx); align `max_workers` with policy; decide single vs double erosion in `_build_smooth_mask` (visual A/B on a small-face crop).

## Sequencing & placement

```
E3 + E4 (small, ride Phase 2 gaps) ──► E2 (≡ F1, Phase 1 slot) ──► E1 (with F8.1/F8.2 window)
```
Nothing here blocks the Phase 2 queue; §1.4 items are the only ones with a *deadline flavor* (they interact with Q4). **Adopted into MASTER_PLAN.md 2026-07-03:** E3.3/E3.4 land inside the Q4 slice; E3.5 + E4 are Phase 2 gap fillers (row 9b); E3.1/E3.2 attach to A2 (row 10); E1 joins the F8 block (Phase 1 row 3); E2 is F1's work inventory (row 4).

## Verification
- E1/E2: synthetic 16-bit-origin gradient corpus — banding step count and high-band energy before/after; golden-image byte-identity for migrated-but-unchanged ops; benchmark.py per-face time (expect improvement).
- E3: slider monotonicity tests (output delta strictly increasing in slider value — extends the existing monotonicity test pattern); recipe re-tune signed off on owner's corpus.
- E4: engine init time before/after; multi-face IPC payload size logged; full pytest throughout.

---

## 18. F1 re-verification against current HEAD (2026-07-16)

Task: verify this doc's §1.1 "16 `_to_uint8_if_float` sites + 3 fake-float
helpers + 100% uint8 no-face path" inventory against current code before
doing any migration work, per F1's own acceptance criterion ("the 16-site
list above *is* the F1 work inventory"). HEAD at time of this audit:
`d6b7d8d` (engine.py: 4486 lines, vs whatever `ea6f835` was on 2026-07-03).

### 18.1 Verdict: the doc's headline inventory is mostly STALE — most of it has already shipped

| §1.1 claim (2026-07-03) | Current status (2026-07-16) |
|---|---|
| 16 `_to_uint8_if_float` sites in `_stage_grade` | **3 remain**, at `engine.py:3712/3801/3808`, each with an inline comment justifying it as inherently-uint8: `subject_aware_transfer` (internal `cvtColor`), `tonal.apply_hd_curve` (`cv2.LUT` requires uint8), `grade_stack` (no `return_float` param). The other 13 (color_transfer, white balance, master HSL, split-tone, highlight rolloff, skin_glow, bloom, glow, fade_toe/highlight_drift/airy_haze/clarity_split finish pack, white-costume lift, vignette, grain, negative split tone, B&W mixer) are now float-native or explicitly scale-wrapped float calls — each carries an `F1/E2: ... now dtype-aware` comment. |
| `_F_adjust_vibrance` / `_F_apply_uniform_saturation` fake-float | **Fixed.** Both docstrings now explicitly say "genuinely float-native" and document the prior bug ("the old version round-tripped through uint8 twice (fake float)"). `_F_apply_uniform_saturation` correctly handles cv2's float-HSV `S` range being `[0,1]` (not `[0,255]` as in the uint8 path) — the exact pitfall this kind of migration is prone to. |
| `_stage_subject_separation` uint8-LAB even on float input | **Fixed.** Genuinely float-native via `bgr_f32_to_lab_f32`/`lab_f32_to_bgr_f32`, dtype-branched throughout. |
| No-face path 100% uint8 (`_no_face_fallback` before `to_float`) | **Partially fixed, partially still true** — see §18.2/§18.3. The docstring claimed "fixed" but was itself misleading (see below). |
| §1.2 B&W mixer literals vs `_DEFAULTS` | **Fixed** at both sites (`engine.py:2634-2636`, `:3956-3958`). |
| §1.3 sharpen inert 1-60 | **Fixed** — monotone map at `engine.py:4033-4036` (`0→0, 50→1.0, 100→3.0`, no dead zone). |
| §1.3 equalize s² non-linearity | Not independently re-checked this pass (out of scope for F1; flagged in CLAUDE.md's "Outstanding Fixes" as previously addressed — `1640eeb`/later commits). |

**Why this matters for anyone picking up F1 next:** don't treat this doc's
Part 1 as a live work-order. Grep current `engine.py` for
`_to_uint8_if_float(` and cross-check against §18.1's table above before
assuming any specific site is still broken — most of the doc's own
citations no longer resolve to the described state.

### 18.2 Genuine residual gap #1, FOUND + FIXED: `apply_global_bloom`'s internal uint8 quantization (`retouch/utils.py`)

Despite `_stage_grade`'s call site carrying the comment "Apply Global
Cinematic Bloom (float-native: handles float32 [0,1] I/O)" — **that comment
was misleading.** `apply_global_bloom` accepted and returned float32, but
internally converted to `img_u8 = np.clip(img_bgr*255,0,255).astype(np.uint8)`
at the very top and did all of its actual work (LAB highlight isolation,
linearization, three-scale cascaded Gaussian blur, screen blend) on that
quantized 8-bit data — a "fake float" round-trip, same bug class as the 3
helpers §1.1 already named, just not caught by the original audit (or
introduced/left uncorrected after).

**Measured** (smooth linear gradient, 120→250 across 512px, spanning into
the `threshold=210` highlight range so bloom actually triggers — a flat/
sub-threshold gradient would false-pass any banding check):

| metric | before fix | after fix |
|---|---|---|
| distinct output levels (of 512 columns) | 258 | **512** (full resolution) |
| vs true uint8-path distinct levels | 127 | 127 (unchanged, as expected) |

**Fix** (`retouch/utils.py::apply_global_bloom`): float32 input now stays in
float32 throughout via `bgr_f32_to_lab_f32` (same helper `apply_skin_diffusion`
already used correctly) instead of quantizing to uint8 for the LAB
conversion and the linearized base image. uint8 input path is untouched.

**Verified:**
- uint8-path output byte-identical before/after (checksummed, not just
  eyeballed): two different random uint8 images at different
  strength/threshold/softness combinations, exact integer-sum match.
- Float path: 512/512 distinct levels on the gradient test (no banding),
  vs 258 pre-fix.
- Float vs uint8-path mean absolute delta < 2.0/255 (same algorithm,
  different precision — small disagreement expected, not a scale bug).
- `tests/test_bloom_linear.py`: 22/22 pass (18 pre-existing + 4 new,
  `TestBloomFloatNativeNoBanding`). `tests/test_utils.py`: bundled, all pass.

### 18.3 Genuine residual gap #2, FOUND + FIXED: `_no_face_fallback`'s white-balance/HSL round-trip (`retouch/engine.py`)

`_no_face_fallback`'s own docstring claimed "F1/E2: uint8 no-face path
fixed — now converts to float32 [0,1] at the top... get the same
float-pipeline benefit as face-detected runs." **This was only true for the
front half of the function.** The conversion to float does happen at the
top, and tonal curve / white balance / master HSL / film density / the main
`grade()` call all ran on float32 — but white balance and master HSL still
round-tripped through `to_uint8()`/`to_float()` at each call
(`engine.py:2494/2500` and `:2504/2511` pre-fix), even though
`white_balance_lch` and `adjust_hsl_lch` are dtype-aware and the *face*
path (`_stage_grade`) already calls them directly on float32 (scaled to
`[0,255]`) without any uint8 boundary. Additionally, **the function always
returns uint8** (`result_u8` at the final `return`) — from the post-`grade()`
post-effects boundary onward (highlight rolloff, bloom, glow, vignette,
impact, grain, negative split tone, B&W mixer) it converts to uint8 and
never converts back, so the docstring's "same float-pipeline benefit as
face-detected runs" claim was not fully accurate even before this fix, and
still isn't for that back half.

**Fix** (`retouch/engine.py::_no_face_fallback`): white balance and master
HSL now use the same scale-around-the-call float32 pattern as
`_stage_grade` (scale `[0,1]` → `[0,255]` float, call directly, scale back)
instead of `to_uint8`/`to_float`. Docstring corrected to accurately state
what is and isn't float-native in this function today (the back-half gap
is now documented, not silently missed).

**Verified (and a real trap caught along the way):** the naive check
"does the no-face WB/HSL output change vs before the fix" showed deltas up
to ~33/255 — larger than a single quantization step, which is exactly the
kind of scale-mismatch red flag this class of fix can hide. Isolated the
cause with a pure identity round-trip (`bgr_to_lch`→`lch_to_bgr` vs
`bgr_f32_to_lch_f32`→`lch_f32_to_bgr_f32`, zero adjustment applied): the
**uint8 path's own round-trip has a max delta of 35 vs the original image**
(cv2's uint8 LAB conversion quantizes before any adjustment math even
runs), while the float32 path's round-trip max delta is 1 (correct,
near-identity). The ~33-level disagreement is the uint8 path's own
pre-existing imprecision, not a bug introduced by the fix — confirmed via
high correlation (0.87) between the two paths' delta-from-original maps,
i.e. both apply the same white-balance direction/magnitude, just with
different quantization noise. This is the intended F1 outcome (removing
quantization error), not a regression.

- `tests/test_engine.py`: 65/65 pass (60 pre-existing + 5 new,
  `TestNoFaceFallbackFloatNative`, including a direct precision-comparison
  test against the uint8 round-trip).
- `tests/test_float_pipeline.py`, `tests/test_grading_lch.py`,
  `tests/test_grading_internal.py`, `tests/test_grading.py`: 273/273 pass
  (bundled run, unaffected files included for safety).

### 18.4 What remains (explicitly not done in this pass)

- **The 3 remaining `_to_uint8_if_float` sites in `_stage_grade`** — legitimately
  boundary-bound today (cv2.LUT, subject_aware_transfer's internal cvtColor,
  grade_stack's uint8-only return). Converting them needs upstream work
  (a float-native curve-interpolation path, a `return_float` param on
  `grade_stack`) that is separately scopeable, not a same-pattern fix like
  the two above. Annotate-not-fix, per the original plan's own E2
  acceptance criterion ("each site migrates to a float-native
  implementation OR gets an explicit inherently-uint8 annotation") — the
  annotations already exist in the code comments; this section is that
  annotation's audit trail.
- **`_no_face_fallback`'s back half** (highlight rolloff through B&W mixer,
  §18.3) — lower priority per the standing rule (this path only affects
  images with zero detected faces), and it's a larger mechanical lift
  (mirroring ~8 more of `_stage_grade`'s float-native call patterns) than
  the two fixes landed here. Docstring now accurately flags this as open
  rather than silently claiming it's done.
- **`apply_skin_diffusion`'s absolute L gate** (`l_chan - 110.0) / 90.0`,
  `retouch/utils.py`) — noticed while confirming this function was
  *already* float-native (it is). Not a float-precision issue, but the
  same skin-tone-relative-vs-absolute-threshold pattern documented at
  length in `PLAN_P4_MAKEUP_UNMIX.md` §§14-17. Out of scope for F1;
  flagged here as a pointer for whoever next works that thread.
