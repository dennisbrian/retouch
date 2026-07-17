# RESEARCH — Face/Body Retouch & Color: massive-improvement candidates

**Date:** 2026-07-17 · **Author:** Fable research pass (read-only; no production code changed)
**Question (owner):** which face/body retouch and color algorithms can deliver a *massive* improvement?
**Method:** status reconciliation of the three prior color/fidelity research docs
(`PLAN_COLOR_SCIENCE.md` K1–K12, `PLAN_EASTWEST_COLOR_SUPREMACY.md` C1–C6,
`PLAN_TIERE_ENGINE_FIDELITY.md` E1–E4) against HEAD by direct grep/read, plus
live probes where a claim needed evidence.
**Scope guard:** items already owned by the concurrently running fix agent
(body-stage no-op fix, hair-parse audit, `marks.py`, H4 gate promotion, body
parity op) are deliberately excluded — see `FABLE_TASK_LIST_2026_07_15.md` §6a.

---

## 1. Status reconciliation — the backlog is smaller than the docs say

Verified at HEAD (grep/read, 2026-07-17). **Shipped since those docs were
written** (do not re-plan): K3 gamut compression, K8 ΔE QA gate, K9 subtractive
saturation (+ OKLCh hue-lock fix), C1 preferred-skin-locus core
(`hue_unify`/`chroma_even`/`skin.locus`, live in shipping recipes), C2
structural sculpt (`relight.py::sculpt`, Delaunay normals + Lambertian target
shading on the form band), C3 parametric film engine (`film.py` + density
dtype fix + HSL calibration), C4 透明感 finish pack (`fade_toe`,
`highlight_drift`, `airy_haze`, `clarity_split`), C5 harmonizer
(`harmonizer.py`), **E1 float-resident per-face core** (canvas converted to
float32 once, single uint8 exit — `perf_optimizations.py`), `blend_masked`
now dtype-preserving, `_stage_grade` uint8 punch-downs reduced 16 → 4 sites,
no-face fallback WB now float. The E-audit's flagship "8-bit Swiss cheese"
finding is essentially fixed.

**Verified still unbuilt:** K1 CAM16(-UCS), K2 hue-linearity (Abney)
correction, K4 chromatic-adaptation white balance, K5 memory-color rendering,
K6 spectral relighting, K7 forward Kubelka–Munk skin optics, K10
Helmholtz-Kohlrausch/constant-Y saturation, K11 structured chroma
re-introduction, K12 blue-noise export dither, F1 residual (4
`_to_uint8_if_float` sites).

---

## 2. Headline new finding — the white-balance control is not white balance

`grading.py::white_balance_lch` (the only WB implementation; used by both the
face-path grade stage and the no-face fallback) maps Kelvin to a **hue-angle
rotation in LCh, weighted toward low-chroma pixels**. Measured live:

```
gray (128,128,128) -> white_balance_lch(T) -> (128,128,128)  for T = 3000K, 4500K, 8000K
```

**A neutral pixel is a fixed point of the control at every temperature.**
Consequences, all inherent to the algorithm (not a bug in it):
- It cannot warm or cool a neutral image — the primary thing a WB slider does.
- It cannot remove a color cast from neutral surfaces (white costumes, gray
  backdrops — half the cosplay corpus) because correction requires *moving*
  the neutral axis, and this only rotates hue of chroma that already exists.
- Rotating the hue of near-neutral pixels re-aims residual color noise
  instead of neutralizing it.
- `tint` has the same structure (rotation toward 150°/330°), same limits.

The K4 plan called the current WB "Kelvin+tint offsets in a simple space"
that "shifts hues non-uniformly." The reality is categorically worse: the
control is **non-functional as white balance**. This also means replacing it
is *low-risk*: defaults are a no-op (6500K/0), and any recipe currently using
it is getting an effect users would not describe as WB anyway.

---

## 3. Ranked massive-improvement candidates

Ranked by (visible quality ceiling raised) ÷ (effort + risk), face-first per
the standing rule.

### #1 — K4: real white balance (CAT16 von-Kries in linear RGB + Planckian Kelvin mapping) ⭐ do first
- **Why massive:** replaces a broken control (§2); venue casts are endemic to
  the convention-hall/cosplay corpus (mixed LED lighting), and every
  downstream skin op (locus unification, chromophore cues, freckle
  classification) inherits whatever cast survives. Fixing WB at the root
  de-biases all of them at once.
- **Shape:** `white_balance.py` — sRGB→linear, Kelvin→white point via
  Planckian locus (closed form), CAT16 diagonal adaptation in cone space
  (~40 LOC), linear→sRGB. Wire behind the existing
  `white_balance_kelvin`/`tint` params; old path retired (defaults no-op, so
  golden renders unchanged).
- **Tests:** gray card neutralizes under any simulated cast (currently
  impossible); skin hue stable across a Kelvin sweep; identity at 6500K/0.
  This also happens to BE Tier C item "C1 Planckian-locus WB" from the Fable
  hardest-tier list — one item, two backlog entries closed.
- **Effort:** ~3–5 days. **Risk:** low (no-op defaults).

### #2 — K12: blue-noise dither at export ⭐ 1-day rider
- **Why now:** E1/F1 made the *internals* float; the final
  `astype(uint8)` at export is the last remaining banding source, and
  porcelain looks (long smooth chest/cheek ramps) are exactly where 8-bit
  contours show. The harmony H5 banding-delta gate built today gives a free
  acceptance metric.
- **Shape:** ±0.5 LSB blue-noise mask added in display-encoded space at the
  single uint8 exit. ~20 LOC + a shipped 64×64 void-and-cluster tile.
- **Effort:** ~1 day. **Risk:** near-zero (amplitude below visibility).

### #3 — K2: hue-linearity (Abney) correction for the skin hue-line ops
- **Why massive now (it wasn't in 2026-07-11):** C1 shipped and is live —
  `cosplay_portrait_polish_v1` runs `hue_unify: 0.55`, `chroma_even: 0.42`
  in production. Those ops treat "same OKLCh hue angle" as "same perceived
  hue," which is false along the skin orange-red line: unifying to an angle
  drifts perceived hue across the L range — i.e., the flagship skin-color op
  has a built-in tone-dependent hue error, which is also a fairness surface
  (error grows with L distance from the anchor).
- **Shape:** hue-correction LUT (published OKLab/IPT hue-uniformity tables)
  applied before/after the hue-line math in `skin.py::unify_hue_line` /
  locus code. No K1 dependency in this form.
- **Tests:** synthetic constant-physiological-hue skin ramp maps to constant
  corrected hue; drift vs current path measurably drops; Fitzpatrick I–VI
  sweep.
- **Effort:** ~3–5 days.

### #4 — Wire `lighting.py` into `relight`/`sculpt` (retouch-side coherence win)
- **Finding:** `relight.py::sculpt` (C2, shipped) auto-estimates light from a
  blurred L-gradient — a cruder, unvalidated estimate — while the
  confidence-rated, catchlight-preferring `lighting.py` estimate (built
  today, E-COMMON-1) is cached on every `FaceContext` with **zero
  consumers**. Sculpt, relight, and the pending GAP-3 portrait finish should
  share one light estimate or they will sculpt/relight in *different
  directions* on the same face.
- **Shape:** pass `FaceContext.light_direction` into `sculpt()`/`relight()`
  when confidence is high; keep the L-gradient as fallback. Byte-identical
  when the estimate is `unknown`.
- **Effort:** ~1–2 days. **Risk:** low; A/B against current renders.

### #5 — K1: CAM16(-UCS) appearance substrate
- Foundational, unlocks K5 (memory color), K10 (H-K), and the stricter form
  of K2 — but its user-visible win is indirect. Build when starting any of
  those three, not before. Also reconciles the stale C6 "CAM16 shipped"
  claim (still OKLab-only, re-verified today). ~1 week, published test
  vectors.

### #6 — K7: forward Kubelka–Munk skin optics (the moat item)
- Physically-valid skin-color generator (melanin/hemoglobin/oxygenation →
  spectrum → RGB, closed form). Three consumers waiting: better preferred
  loci per ITA band (tone-fairness by construction), a "does this edit stay
  on the real-skin manifold" validator, and the P4 Track-B prior the makeup
  plan is parked on. ~1.5 weeks. The one "nobody else has this" item that is
  still fully classical.

### #7 — K10 H-K/constant-Y saturation, then K5 memory color (post-K1 polish tier)
- Both real, both smaller; sequence after K1 exists.

### Deliberately excluded (owned elsewhere or already covered)
- Body three-band parity port, body-stage no-op fix, hair-parse audit,
  `marks.py`, H4 promotion → running agent (`FABLE_TASK_LIST` §6a).
- F1 residual 4 uint8 sites → mechanical, Haiku-tier, fold into F1 checklist.
- Teeth/eye optical models → already Fable list #3 (frontier plan).
- K6 spectral relighting → moonshot; revisit after K7 ships its SPD tables.

---

## 4. Recommended next implementation

**K4 white-balance replacement + K12 export dither as one slice** (~1 week
total): both standalone, both cheap, both visibly testable, and together they
close the two remaining root-cause color-fidelity holes (cast correction that
works; banding-free export). Then K2 (hue-linear skin line), then the
K1 → K5/K10 substrate chain, with K7 as the strategic moat build when a
full week is available.

## Appendix — reproduce the evidence

```bash
# WB gray fixed-point proof
python3 -c "import sys; sys.path.insert(0,'.'); import numpy as np; \
from retouch.grading import ColorGrader; g=ColorGrader(); \
gray=np.full((8,8,3),128,dtype=np.float32); \
print([tuple(np.round(g.white_balance_lch(gray.copy(),temperature=T)[4,4],2)) for T in (3000,4500,8000)])"

grep -rln "CAM16\|CAT16\|kubelka\|blue_noise\|hue_linear" retouch/   # all empty = still unbuilt
grep -rn "light_direction" retouch/relight.py                        # empty = sculpt not yet a consumer
```
