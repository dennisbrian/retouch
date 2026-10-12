# Research — Per-Op Face-Retouch Visual Quality Audit (isolation renders, DSCF corpus)

**Date:** 2026-08-31
**Status:** COMPLETE (single-labeler visual pass; fixes deliberately NOT applied — findings only).
**Branch:** `feat/color-science-k9-fix-and-frontier`
**Runtime:** `.venv/bin/python` (mediapipe 0.10.5, `RETOUCH_GPU=0`,
`RETOUCH_MEDIAPIPE_BACKEND=legacy`). Bare `python3` never used.
**Script:** `scripts/qa/faceop_visual_audit.py`
**Artifacts:** `test_output/faceop_visual_audit/` — `metrics.json`, `run.log`,
`sheets/<image>_<op>.jpg` (face crop + top-2 diff hotspots, source | op | diff-heat).

---

## 0. Motivation & scope

The delta-blindness lesson (Bonodori QA; catchlight/handedness P0s of 2026-08-26) is that
global metrics hide face damage and "wired" ops can be inert or paint the wrong region.
No prior study rendered each face op **in isolation** across a real corpus and looked at
the pixels. This study does exactly that for 24 per-face ops (skin / eyes / under-eye /
mouth / cheeks / geometry classes); it does not cover global color/tone stages, makeup_v2,
hair, or body ops.

## 1. Method

- **Isolation:** baseline = `natural` recipe with **all 24 audited ops explicitly zeroed**;
  each op render flips exactly one kwarg to a moderate-strong test value (values in the
  script's `OPS` table). absdiff(baseline, op render) is attributable to that op alone.
- **Corpus subset (9):** DSCF4463/4503/4550 (verified real-face anchors), 4552 (narrow-eye
  wig subject), 4560 (heavy squint), 4576 (closed eyes — the eye gate must suppress eye
  ops), 6961 (open-eye anchor), 7204 (heavy lashes), 8007 (different shoot). Pre-shrunk to
  max-dim 2048.
- **Cache condition:** baseline rendered twice — without and with cached `face_contexts`;
  op renders reuse the cached contexts and diff against the cached-context baseline
  (apples-to-apples). The no-cache vs cache baseline MAE is recorded as a
  determinism/cache-equivalence check.
- **Automated detectors (ranking heuristics; the sheets are the evidence):**
  - `leak_ratio` = mean |diff| outside the op's expected-region mask / inside — flags
    "op paints where it shouldn't" (the 2026-08-26 handedness-bug class). Expected
    regions are landmark-cluster hulls (`parsing.py` index sets) dilated by fractions of
    IED; `FaceRegions` masks are ROI-local and unusable in full-image coords. The skin
    class uses the dilated FACE_OVAL hull, which does **not** cover the neck — BiSeNet
    skin ops legitimately edit the neck, so skin-op leak over-reports by construction.
  - per-eye diff energy (`eye_l` / `eye_r`) — handedness / cross-eye-coupling detector
    for eye-class ops.
  - inertness — `mae_in ≈ 0` where the op should visibly act.
- **Visual pass:** metrics rank; contact sheets are then inspected by eye before any
  claim is made (CLAUDE.md verification rule).

## 2. Code findings (established during harness construction, pre-sweep)

- **`catchlight` is silently inert unless another eye-stack op is active.** The per-face
  dispatch guard (`perf_optimizations.py:810`) is
  `eye_enhance > 0 or eye_sclera_vessel_remove > 0 or corneal_shading > 0` — `catchlight`
  is consumed *inside* `eyes.enhance` but is **not in the guard**, so
  `catchlight=60, eye_enhance=0` renders zero changed pixels (verified empirically,
  DSCF4463). Checked all shipping recipes via `resolve_recipe()`: **0 recipes** hit the
  dead combination (every catchlight>0 recipe also sets whites>0), so this is a **latent
  GUI trap** (user slides catchlight alone → silent no-op), not a live recipe bug.
  Candidate one-line fix: add `ctx.catchlight > 0` to the guard.
- **`teeth_whiten` ParamSpec false alarm, on record:** the spec's
  `recipe_key="eyes.whites"` looks like a copy-paste bug (same key as `eye_enhance`), but
  `engine.py:789` has a hand-written branch that resolves `eyes.teeth_whiten` first and
  falls back to whites — recipes' teeth values are live. Verify-don't-assume pattern;
  noted so nobody "fixes" the spec into a behavior change.
- **`shine_removal` zero-diff is adaptive, possibly legit:** detection requires
  `L > skin-median + 20` AND chroma < 40% of skin median (`skin.py::shine_removal`,
  tone-adaptive per CLAUDE.md fairness rules). Zero change on a matte face is correct
  behavior; verdict deferred to the corpus-wide result (zero on **all** images incl.
  flash-lit would instead suggest the threshold never fires at 2048px).

## 3. Sweep results (225 records, 714 s, 9 images × 24 ops + baselines)

**Version caveat:** another session (`retouch-18`) edited `perf_optimizations.py` /
`eyes.py` at 16:29 *while the sweep ran* (adding `ctx.catchlight > 0` to the eye-stack
guard + `eye_artifact_scales` wiring). The sweep's in-process module predates the edit,
but multi-face images spawn pool workers that imported the NEW code — so single-face
rows are pre-edit code and DSCF4576 (2 faces) is post-edit code. Findings below were
each re-verified against a single code version where it matters.

Determinism/cache checks passed on all 9 images: `cache_mae = 0.0000` (supplying cached
`face_contexts` is render-identical), baseline-vs-input MAE 0.49–0.60 (global `natural`
stages only, sane).

### Confirmed findings, by severity

1. **LIVE — 27 recipes double-apply undereye darken removal.** `dark_circles` (legacy)
   and `undereye_darken_removal` both drive the *same*
   `UnderEyeProcessor.process(darken_removal_strength=…)` (`undereye.py:354` vs
   `perf_optimizations.py:786`), sequentially in one render. Sweep: the two ops produce
   byte-identical isolated diffs on all 9 images. 27/128 shipping recipes set both keys
   (incl. `apex_cosplay_v1` 0.10+0.15, `tired_eye_rescue_v1` 0.20+0.60,
   `convention_clear_v1` 0.25+0.60) → systematic undereye over-brightening ("concealer
   look") in the flagship clear-family recipes. Fix decision needed: alias the params
   (max, not sum) or de-dupe the recipes.

2. **SYSTEMIC — pose-gated ops are dead on typical portraits (yaw formula saturates).**
   `slimming` rendered zero change on 7/9 images; `sculpt` on 6/9. Both trace to the
   shared nose-tip/temple x-ratio yaw formula (`geometry.py::_yaw_dampen_factor` ramp
   1.3→1.6; `relight.py::sculpt` its own copy at 1.5→1.7; `skin.py` neck gate hard 1.3).
   Measured: DSCF4463 yaw_ratio 2.03 → damp 0.00 while its per-eye-width ratio is 1.29
   (mild turn); DSCF4503 eye-width ratio 1.01 (frontal) still gets damp 0.54. Root
   cause: the nose tip protrudes, so its x-offset saturates the ratio at mild yaw; a
   roll-corrected variant changes nothing (verified — ratios are real, the *threshold
   band* maps to ordinary portrait poses). Isolated repro: `relighter.sculpt(strength=50)`
   returns byte-identical input on 4463, works on 4503. Same shape as the EAR
   miscalibration: good signal, wrong operating band. Needs its own calibration study
   before any threshold change.

3. **SYSTEMIC — unmasked full-canvas colorspace/dtype roundtrips contaminate the whole
   face ROI** (E1-contract violation class):
   - `lips.py::_reapply_texture` (line 221–223) and `_add_lip_gloss` convert the entire
     image BGR→LAB→BGR and return it wholesale — the edit is masked, the roundtrip is
     not. Unit probe: pixels strictly outside a zero mask move up to 1.0 level (mean
     0.40) per enhance() call on a float32 canvas. `natural` ships `lips.gloss=0.05`,
     so **every default render takes this face-wide drift**.
   - `teeth.py::whiten` — same pattern (`return _from_lab(lab)` full-image).
     Sweep: teeth op shows eye-region diff ≈0.5 (4552) — a teeth op touching eyes.
   - eye-v0 dispatch (`perf_optimizations.py`) converts the whole ROI canvas
     float→uint8→float whenever sclera/iris ops are active; on DSCF4552 the intended
     iris-saturate edit is invisible (backed off by `eye_artifact_safety` on saturated
     contact lenses — correct) while the render still carries ROI-wide dither over wig,
     hand, background (sheet `DSCF4552_eye_iris_saturate.jpg`). 27/128 recipes activate
     eye-v0. Magnitudes are ≤1 level each but compound across ops and re-renders.

4. **LATENT (fixed mid-session by `retouch-18`) — catchlight-only was silently inert.**
   Pre-edit guard omitted `catchlight`; sweep confirms 0.000 diff on all 8 single-face
   images (old code) and activity only on 4576 (new-code workers). 0/128 recipes hit the
   dead combination; GUI slider alone did nothing. The 16:29 working-tree edit adds
   `ctx.catchlight > 0` to the guard — finding retained as corpus evidence for that fix.

5. **CONFIRMED END-TO-END — closed-eye painting on gate-missed eyes.** On DSCF4576
   (right eye closed, left eye open per the calibration study's `labels.json` — this
   finding originally recorded "both eyes closed", corrected 2026-08-31; the closed
   right eye was the calibration study's *already-caught* anchor, gated even by the
   pre-`fafed66` thresholds), `eye_enhance=60` and `catchlight=60` still alter the face
   ROI — [2026-08-31 reinterpretation: with the side correction, this image contains no
   gate-missed closed eye — the left is genuinely open and enhancing it is correct, the
   right was gated. The residual therefore does not demonstrate closed-eye painting; it
   is fully explained by the byte-identical shared `eyes.enhance` entry side effect
   below (likely finding 3's ROI uint8 roundtrip). A true gate-missed case (e.g.
   DSCF4612-R or DSCF4454-L pre-recalibration) would be needed to re-establish the
   end-to-end claim — moot now that `fafed66` shipped.]
   and produce **byte-identical** outputs (mae 0.010, max 4 levels, 19,771 px, all
   inside face-1's ROI; the off-subject bokeh detection untouched). The magnitude is
   small only because `eye_artifact_safety` backs off — defense-in-depth catching what
   the miscalibrated EAR threshold misses. Direct evidence for shipping the EAR<0.28
   recalibration. The byte-identity (two different ops, same output) means the residual
   is a shared side effect of entering `eyes.enhance`, not op-specific edits — exact
   micro-mechanism unresolved (see §4).

6. **DESIGNED-BUT-NOTABLE — A5 plastic-skin backoff can fully nullify requested
   smoothing.** DSCF8007's entire skin-op column collapsed (smooth 0.00, skin_sss 0.00,
   face_exposure 0.33 vs 4.3–7.1 elsewhere). Probe: the source flags `plastic_skin`
   (hf_energy_ratio 0.123 — it is an already-ultra-soft asset, probably a previous
   engine output living in `test_output/`), so QA backoff slashes skin params to ~zero.
   Working as designed; but (a) the corpus asset is contaminated for skin-op studies,
   and (b) users get silent total nullification of an explicit slider with only a log
   line.

### Behaviors verified correct (worth recording)

- `shine_removal` zero-diff on 6/9 images is adaptive detection working (fires 0.08–0.13
  on the three images with actual shine); tone-adaptive form confirmed at call site.
- `teeth_whiten` fires only where teeth are visible (4552/4560 open smiles) — correct.
- `eye_sclera_vessel_remove` exits byte-clean when sclera masks are empty/gated (float
  path, no roundtrip) — the one eye op with clean isolation.
- `teeth_whiten` ParamSpec `recipe_key="eyes.whites"` is NOT a bug — `engine.py:789`
  resolves `eyes.teeth_whiten` first (verify-don't-assume; do not "fix" the spec).
- Skin-class ops (smooth/equalize/whiten/redness/sss/wrinkle) leak ≤0.0013 outside the
  face+neck — masks compose correctly. Per-eye energies for skin ops are eyelid skin
  (legitimate; sheets checked).
- `reshape_eye_size` warps both eyes with the expected large-support field; bangs
  overlapping the eye region are dragged with the warp (liquify artifact class — an
  A/B flicker call, not a defect; sheet `DSCF6961_reshape_eye_size.jpg`).
- Supplying cached `face_contexts` is render-identical (cache_mae 0.0 on 9/9).

## 4. Follow-ups (none started)

- **Yaw-gate calibration study** (finding 2) — EAR-study-shaped: measure yaw_ratio vs a
  pose ground truth (eye-width ratio or solvePnP) over the 83-image corpus; pick a band
  that distinguishes "warps will mis-scale" from "typical portrait turn". Owner
  decision on where slimming should engage.
- **Roundtrip hygiene pass** (finding 3): mask-blend or bbox-localize the LAB roundtrips
  in `lips.py`/`teeth.py`; float32 path for eye-v0 (`eye_enhancement.py`). Mechanical
  once decided; golden face snapshots will move.
- **Undereye de-dupe** (finding 1): decide alias-vs-recipe-fix, then one commit each.
- Root-cause the shared `eyes.enhance` entry side effect (finding 5's byte-identity).
- Replace/annotate DSCF8007 in the corpus for skin-op studies (finding 6).
- ~~Coordinate with `retouch-18` before touching `eyes.py`/`perf_optimizations.py`~~ —
  resolved 2026-08-31: `retouch-18` replied and committed `fafed66` (eye-gate threshold
  recalibration to `_MIN_EAR=0.285`/`_MIN_CONTRAST=0.55` plus the whole in-flight eye
  stack: `eye_visibility.py`, `eye_artifact_safety.py`, wiring in `engine.py`/`eyes.py`/
  `eye_enhancement.py`/`params.py`/`parsing.py`/`perf_optimizations.py`/`conftest.py`).
  It is not planning further edits, and did not touch `lips.py`/`teeth.py`/`undereye.py`/
  `geometry.py` — findings 1–3 remain unaddressed and unclaimed. Consequences for this
  study: the DSCF4576 rows (finding 4) mixed pre-/post-commit code versions; pool
  workers now import a consistent eye stack. Eye-side correction (verified against
  `labels.json`): DSCF4576's closed-eye anchor is the RIGHT eye
  (`occluded, "smile squint, lid shut"`); DSCF4576:left is labeled `visible` — earlier
  session notes saying "4576-L" had the side flipped. `retouch-18` already visually
  re-confirmed finding 5's case under the shipped thresholds during its `fafed66`
  verification pass (closed right eye: no painted iris/catchlight; open left eye
  enhances normally), so a 4576 re-render is only needed for independent confirmation,
  not correctness.
  `retouch-18` also confirmed the process-pool observation (in-process monkeypatches/log
  capture see zero gate calls for multi-face images unless `_face_pool.process_faces` is
  forced to raise into the ThreadPoolExecutor fallback).

