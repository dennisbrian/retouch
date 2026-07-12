# Tier C Implementation Plan — execution slices (C1–C6)

**Date:** 2026-07-02
**Parent:** `PLAN_EASTWEST_COLOR_SUPREMACY.md` (research + stage definitions), `MASTER_PLAN.md` (adopted phasing)
**Ground truth verified this session:** per-face op order lives in `retouch/perf_optimizations.py:_process_face_core` — flatten (`:281`) → equalize (`:298`) → skin_unify (`:302`) → whiten (`:306`) → quantize (`:322`); `SkinProcessor.equalize` median a/b pull at `skin.py:157-161`; `whiten` LAB math at `skin.py:77-123`; ParamSpec pattern at `params.py:398-447` (`skin.flatten` etc. nested recipe keys, `recipe_pct` conversion).

Coding is delegated to Haiku agents per owner instruction; each slice below is one Haiku brief. Rules for every slice: **do not commit; do not run the full test suite** (targeted test files only); syntax-check touched files; never touch `gui.py` (auto-wires from params) beyond what a slice states.

---

## Slice 1 — C6 substrate + C1 core (FIRST) ✅ ready to code

### 1a. New module `retouch/color_science.py`
- `bgr_to_oklab(img_bgr: uint8) -> float32 (H,W,3)` and `oklab_to_bgr(...) -> uint8`: sRGB EOTF → linear → LMS (Oklab M1) → cbrt → M2. Fully vectorized; no pixel loops. `oklab_to_oklch` / `oklch_to_oklab` (h in degrees).
- `measure_skin_state(img_bgr, skin_mask, thresh=0.3) -> SkinState` dataclass: `L_mean, C_mean, C_std, h_mean` (circular mean for hue) over mask.
- `SKIN_LOCI` dict (module data, JSON-shaped): class chosen by `L_mean` — `fair` (L̄ ≥ 0.72): h_target 45°, C_target 0.085; `tan` (0.55–0.72): h 52°, C 0.105; `deep` (< 0.55): h 58°, C 0.125. All OKLCh units. These are literature-informed starting points; owner perceptual validation pass required before defaults ship (open item in parent doc).
- `skin_chroma_std(img_bgr, skin_mask) -> float` — the σ_C quality metric (benchmark wiring in Slice 2).

### 1b. `SkinProcessor.unify_hue_line()` in `retouch/skin.py`
Signature: `unify_hue_line(img_bgr, skin_mask, hue_strength: int = 0, chroma_strength: int = 0, locus: dict | None = None) -> uint8 BGR`.
1. No-op guards like siblings (strength ≤ 0 or mask None → return input).
2. OKLCh convert; `measure_skin_state`; pick locus class by L_mean unless `locus` given.
3. **Hue-line pull:** per-pixel Δh = `clip(shortest_arc(h_target − h), −8°, +8°) × (hue_strength/100) × skin_mask × eligible`; rotate hue only. `eligible` excludes C > 0.18 (body paint/makeup/tattoo), L < 0.25 (dark features/hair shadow), and applies `_get_highlight_protection`-style specular guard (reuse existing helper on a LAB view, or L > 0.95 cut).
4. **Chroma evening:** `C ← C + clip(C_target − C, −0.04, +0.04) × (chroma_strength/100) × skin_mask × eligible`. L untouched (that's S2's job).
5. Convert back, `blend_masked(img_bgr, out, skin_mask)` like siblings.

### 1c. Wiring
- Two ParamSpecs in `params.py` after `skin_unify_hue`: `skin_hue_unify` (int 0–100, default 0, recipe key `skin.hue_unify`, `recipe_pct`) and `skin_chroma_even` (int 0–100, default 0, recipe key `skin.chroma_even`, `recipe_pct`).
- Call site in `_process_face_core` between the equalize block and the skin_unify block (i.e., after `perf_optimizations.py:299`):
  ```python
  if ctx.skin_hue_unify > 0 or ctx.skin_chroma_even > 0:
      canvas = skin.unify_hue_line(canvas, regions.skin, int(ctx.skin_hue_unify), int(ctx.skin_chroma_even))
  ```
- No recipe changes in this slice (recipes adopt after visual QA). `skin_locus` recipe override deferred here — ParamSpec registry is scalar-only; nested-dict recipe handling lands in Slice 2.

### 1d. Tests `tests/test_color_science.py` (+ additions to `tests/test_skin.py`)
Round-trip BGR→Oklab→BGR max error ≤ 2/255; known anchors (pure white → L≈1.0, C≈0; sRGB red hue ≈ 29°); hue rotation never exceeds 8° at strength 100; synthetic patchy-skin image (two hue clusters 15° apart): post-op circular hue std strictly decreases and is monotonic in strength; σ_C decreases under chroma_even; strength-0 byte-identical; None-mask no-op; lips-free mask assumption documented (call passes `regions.skin`, which BiSeNet already separates from lips).

**Deferred from C1 (Slice 2):** luminance-priority whitening refactor of `whiten` (behavior change to a shipped op — wants its own visual A/B), benchmark σ_C wiring, recipe adoption, locus override.

---

## Slice 2 — C1 finish (whiten refactor + metric + recipes) — ✅ DONE 2026-07-03
`whiten` gains `hue_stable: bool = False`: when set, the L-lift (skin.py:98-104) runs in OKLCh with C and h held (kills high-strength chalkiness); tone shifts (rosy/porcelain) still apply after, expressed as bounded OKLCh hue/chroma nudges. Old path stays default until corpus A/B. σ_C metric into `benchmark.py`. `skin.locus` nested recipe key (dict pass-through like existing nested `skin` blocks). Add modest `hue_unify`/`chroma_even` to one portrait recipe as proving ground.

Receipt 2026-07-03: `whiten_hue_stable`, σ_C benchmark metric, C1 recipe updates, and `skin.locus` override wiring are present. Follow-up doc-sync verified `skin_locus` flows recipe_loader.py → engine.py ProcessingContext → perf_optimizations.py `unify_hue_line(locus=...)`; 22 focused non-MediaPipe tests passed. Full `tests/test_skin_locus_override.py` is still blocked locally by a MediaPipe abort during `RetouchEngine()` initialization, so real face-detected visual QA remains owner/full-runtime work.

## Slice 3 — C2 structural light-shadow (`retouch/sculpt.py`)
Landmark-mesh shading model + form-band correction per parent doc §C2; shares light-direction estimate with relight. Blocked on: Slice 1 shipped + S2 helpers existing (S2 is the adjacent Phase 2 item — build S2 first per MASTER_PLAN order).

## Slice 3b — Relight v2 (review outcome, 2026-07-03) — companion to C2
Fable review of `retouch/relight.py` (triggered by the facet-mottling bug, fixed 2026-07-03 via coarse-scale shading). Structural weaknesses remaining even after the fix:
1. Adds a second light instead of moving the light — diffuse term multiplies the photo's existing shading (`relight.py:153`); no delight step.
2. Specular is a colorless LAB-L push (`:158`) → chalky highlights; real speculars carry light color and reduce skin chroma.
3. Gamma-space math (lighting should be ~linear; wants F1).
4. Face-only mask — neck/ears/hair keep old lighting, mismatch at strength.
5. No estimate of the photo's existing light direction; user azimuth can fight reality.
Keep: yaw guard, highlight protection, landmark mesh geometry at coarse scale.

**v2 design (rides C2's shared shading engine):** estimate existing light (least-squares fit of low-band L vs mesh normals) → bounded delight (divide out estimated low-band shading) → re-shade with target direction in ≈linear space → apply as tinted RGB gain (`relight_kelvin` param) with chroma-aware specular → extend through neck/ear masks, feathered → `relight_softness` (shading-field blur = softbox size). Build as part of Slice 3, one engine for sculpt + relight. QA: existing-vs-new light agreement test (relight toward the estimated direction ≈ no-op at low strength), chalkiness check (skin chroma under specular must not rise), neck-face lighting continuity.

## Slice 4 — C4 finish pack (`grading.py` ops: fade_toe, highlight_drift, airy_haze, clarity_split + `jp_transparent_v1` recipe).
## Slice 5 — C3 film engine (`retouch/film.py`) — blocked on F1 float pipeline.
## Slice 6 — C5 harmonization — pairs with T1.

---

## Verification per slice
Targeted pytest files only (owner runs full suite + commits). Slice 1 gates: new tests green, `test_skin.py` untouched-and-green, `py_compile` on touched files, no behavior change with new params at defaults (spot-check: one image hash unchanged through `_process_face_core` with defaults).
