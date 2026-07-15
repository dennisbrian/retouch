# RESEARCH_FUJI_SIM_QUALITY_CEILING.md — what blocks "best-in-class" Fuji sims

**Status:** research handoff, not started. Written 2026-07-14 after a user
ask to make the 14 Fuji film simulations (see `retouch/recipes.py`
`FUJI_SIM_NAMES`) "among the best" / "exhaust all the ability."

**STATUS UPDATE (2026-07-15):** The two headline blockers listed below—FilmDensityEngine brightness-crush bug (§2) and absent selective-color/HSL calibration (§3)—were fixed the same day this doc was written. See commits `9699157` (fix: FilmDensityEngine characteristic curve) and `f3290c3` (feat: selective HSL/calibration with hue/sat/luminance per-band adjustments now wired in `engine.py`). Only the reference-corpus-acquisition item (§4) remains open. Treat the rest of this doc as historical context, not an active task list.

**Context:** `docs/FUJI_COLOR_RESEARCH.md` is the existing research doc.
This file picks up where that one's theory meets the actual engine code —
i.e. which of its four "key technical ingredients" (§3) are real, wired,
and usable from a recipe dict today, and which are aspirational/broken.

---

## 1. What's real and already used

- **3-way luminance split-tone** (`shadow_hue/sat`, `midtone_hue/sat`,
  `highlight_hue/sat` recipe keys → `engine.py` `_split_tone_three_way` /
  `_F_split_tone_three_way`). Real, wired, used by all 14 sims.
- **LCh skin-hue protection** (`skin_protect` recipe key →
  `skin_protect.protect_skin`, masks on `skin_mask_lch`). Real, wired,
  matches research §3.3 "hue protection" concept.
- **Physically-modeled grain** (`grain_strength` → `grain.apply_film_grain`).
  Real, wired. Already clumped + luminance-correlated per research §3.4
  (`clump_sigma=1.2, luma_power=1.2` defaults) — engine only ever passes
  `strength`, the other params stay at their (already-good) defaults.
- **True grayscale** (`bw_channel_mixer_r/g/b` → `grading.channel_mixer_bw`,
  gated last in `_stage_color_grading`). Real, wired. **Gotcha**: the
  activation gate in `engine.py` (~line 2540/3840) compares the three
  weights against the literal engine defaults (30/59/11, which happen to
  be BT.601) — a recipe using BT.601 weights on purpose will silently
  no-op. `monochrome` deliberately uses Rec.709 (21/72/7) to trip the gate.

## 2. What's broken

- **`FilmDensityEngine`** (`retouch/film.py`, design doc
  `docs/PLAN_C3_FILM_DENSITY.md`) — the H&D toe/shoulder curve + 3×3 dye
  crosstalk matrix. This is conceptually exactly what research §3.2
  describes and would be the biggest lever for authentic tonal response.
  **It crushes brightness even at its own defaults**: tested
  `FilmDensityEngine().apply(img, {"enable": True})` (all other params at
  documented defaults) on a real photo and mean luminance dropped 144→62
  (~57%). Root cause: `Y_mapped = 10**(-D)` only preserves identity when
  `D == -log10(Y)`; `_hd_curve_logdensity` produces `D ≈ raw * D_MAX` from
  a sigmoid over normalized `logE`, which never equals `-log10(Y)` — so
  even "neutral" density crushes midtones hard.
  - Zero recipes use it (`grep film_ retouch/recipes.py` → only
    `film_noir_cinema_v1` / `minimal_film_v1`, unrelated names).
  - `tests/test_film.py` (25 tests) all pass — none check absolute
    brightness/rendering fidelity on a real image, only shape/dtype/
    monotonicity-class properties. This is why the bug was never caught.
  - **Do not use this module for film sims until it's fixed.** Fixing it
    is a standalone engine task with its own Visual-Critical QA gate per
    `PLAN_C3_FILM_DENSITY.md` §7.3 — not a recipe-tuning task.

## 3. What doesn't exist at all

- **Per-hue-band selective color** (the "greens→teal, reds→orange"
  rotation that research §4.2 calls the single most Fuji-distinctive
  move for Classic Chrome). `presets/astia.json` on disk carries
  `hsl_adjustments` / `calibration` blocks that *look* like this — but
  they are **decorative**. Confirmed via
  `grep -n "hsl_adjustments\|calibration\|hue_rotation" retouch/engine.py`
  — zero matches. The only recipe-level hue controls that actually reach
  the engine are: one global `hsl_hue_global`/`hsl_sat_global` scalar,
  and the 3-way luminance split-tone above (tonal-range-based, not
  hue-band-based). There is no mechanism today to say "rotate greens by
  X° and desaturate them by Y%" independent of other hues.
  - Building this would be a real engine feature (a hue-band mask in LCh
    space, similar in spirit to `skin_mask_lch` but for arbitrary hue
    wedges, then per-band hue-rotate + desaturate). Not present anywhere
    in the codebase today under any name.

## 4. No reference corpus

Even if both gaps above were closed, there is currently no way to
*verify* a "best-in-class" claim — no real Fuji-JPEG or Fuji-RAW
reference set to diff against. `docs/FUJI_COLOR_RESEARCH.md` itself
frames 90-95% as an eyeball-only target with no ground truth ("every
published recipe is opinion," §4.6). Any future "make it better" pass
should include acquiring or building a small reference set (a few
real Fuji X-series JPEGs per sim, shot or sourced) so improvement is
measurable rather than vibes-based.

## 5. Suggested order if picked back up

1. Fix `FilmDensityEngine` brightness-preservation bug (§2), verify with
   a brightness-delta test on a real image (not just the existing
   shape/dtype unit tests) — add that regression test.
2. Once verified, calibrate `provia`/`astia`/`classic_chrome` through it
   (`film_enable: true` + tuned toe/shoulder/crosstalk), since those
   have the deepest research already in `FUJI_COLOR_RESEARCH.md` §4.2-4.4.
3. If deeper authenticity is still wanted: design a per-hue-band
   selective-color primitive (§3) as its own engine feature, wire it to
   a new recipe key, apply to Classic Chrome's green→teal / red→orange
   move specifically (the one thing current recipes cannot do).
4. Acquire a small reference corpus (§4) before claiming any
   "best-in-class" result — otherwise there's no way to know if a change
   is an improvement.

## 6. What NOT to redo

- Don't re-derive that `FilmDensityEngine` defaults crush brightness —
  confirmed empirically, see §2, reproducible with:
  ```python
  from retouch.film import FilmDensityEngine
  FilmDensityEngine().apply(img_bgr, {"enable": True})  # mean drops ~57%
  ```
- Don't re-grep for `hsl_adjustments`/`calibration` engine wiring — confirmed
  absent, see §3.
- The 14 existing Fuji recipes (`retouch/recipes.py`, commit `fd842b3`,
  polish commit follows this doc) are real and render correctly — this
  doc is about the *ceiling* above them, not a claim they're broken.
