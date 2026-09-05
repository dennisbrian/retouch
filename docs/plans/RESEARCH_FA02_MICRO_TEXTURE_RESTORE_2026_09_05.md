# FA-02: does `restore_micro_texture` bring back a suppressed defect?

**Date:** 2026-09-05

**Status:** Evidence tranche only. No production code changed. Follows
`docs/plans/RESEARCH_FA01_ONTOLOGY_AND_EVIDENCE_2026_09_05.md`'s backfilled
documentation contract, per the execution plan's ordering rule (FA-02
begins once FA-01's support/protection contract is written down).

**Plan basis:** `docs/plans/PLAN_FACE_RETOUCH_ALGORITHM_RESEARCH_EXECUTION_2026_09_05.md`
"FA-02 — texture preservation evidence": compare current texture
restoration against band-selective restoration that excludes approved
defect supports, holding masks/smoothing/all other ops fixed, with
restoration disabled as an ablation. Named trap: "a high-frequency energy
increase is not evidence of authentic texture by itself."

---

## 1. Locating the actual FA-02 subject

The pipeline has two texture-restoration mechanisms, not one, and only
one of them is a plausible fit for "restored defect":

1. **`frequency.combine()`'s `texture_opacity` / `pore_synthesis`**
   (`retouch/frequency.py`). Runs inside `_process_face_core`
   (`retouch/perf_optimizations.py`) at source-line ~479–572, **before**
   `blemish.remove` at line ~885, both within the same function body in
   linear source order (confirmed by reading the function directly, not
   inferred from call-graph structure). `texture_opacity` reprojects a
   scalar fraction of the pre-existing high-frequency band; `pore_synthesis`
   adds independently-seeded synthetic noise. Neither can be "restoring an
   already-repaired defect," because at the point this mechanism runs,
   nothing has been repaired yet — blemish removal hasn't executed. Ruled
   out as the FA-02 subject on ordering grounds alone.

2. **`skin.SkinProcessor.restore_micro_texture`** (`retouch/skin.py:985`).
   Runs at `perf_optimizations.py:670`, **after** the base smoothing stage
   (`frequency.combine`, ~572) and **before** `blemish.remove` (~885).
   Computes `detail = pre_smooth_canvas - smoothed_canvas` (i.e., exactly
   what smoothing removed) and adds a fraction of it back, confined to a
   fixed "dimensional" zone union (`nose_bridge`, `cheek_highlights_l/r`,
   `left/right_under_eye`, `left/right_eye`, `crows_feet_l/r`). This
   mechanism has **zero mark-policy or defect awareness** — if a mark's
   footprint overlaps a dimensional zone, whatever smoothing suppressed
   there (including a mark's own contrast, whether or not `mark_policy`
   is active) gets partially restored. This is the live FA-02 subject.
   `micro_restore` is a registered, shipped `ParamSpec`; 4 recipes set it
   (strengths 8–35), so this is not a dormant code path.

Correction to an earlier draft of this reasoning: the defect this
mechanism can restore is **whatever base smoothing suppressed**, not "an
already-repaired defect" — `restore_micro_texture` runs before
`blemish.remove`, so a healed blemish is never the input here. The
correct framing is: smoothing legitimately reduces a mark/mole/blemish's
contrast (more so if `mark_policy` doesn't protect it, per FA-01 §1.1),
and this mechanism can then partially undo that reduction inside a
dimensional zone, independent of whether the reduction was itself
intentional or protected.

---

## 2. Method

Three arms, matching the plan's required ablation, all calling the
**real, unmodified `SkinProcessor.restore_micro_texture`** directly
(`scripts/qa/micro_texture_restore_experiment.py`) — not a
reimplementation, since it is a small, pure function of plain arrays plus
a minimal `regions`-like object:

- `disabled` — `strength=0` (ablation).
- `current` — called exactly as production does today (unmodified
  dimensional mask).
- `defect_excluded` — the mark's footprint subtracted from the
  dimensional mask *before* the same function call (band-selective
  restoration with a defect exclusion), the most direct reading of the
  plan's "band-selective restoration with repaired-defect exclusions."

Base smoothing runs through the real, unmodified `frequency.separate`/
`combine()` path, including the FA-01 `mark_protect` mechanism already
shipped (`c665da1`), so the experiment measures interaction with the
*current* pipeline, not a stripped-down one. Two smoothing conditions are
crossed with the three restoration arms: `mark_policy=None` (the shipped
default in every recipe that sets `micro_restore`) and a `mark_protect`
mask supplied (as if a policy were active).

**Discriminator**, addressing the plan's own warning directly: whole-face
or whole-mask high-frequency energy cannot tell an authentic pore-texture
increase apart from a restored defect, because both are correlated with
the source and both raise high-band energy. This script instead measures
energy **inside the mark's own footprint specifically**
(`in_mark_mean_abs_delta_vs_smoothed`), plus the same ring-minus-mark
contrast instrument FA-01 used, which directly asks "is the defect
visually back," not "did some measure of texture go up."

**Scenes:**
- Synthetic textured canvas (same construction as the FA-01 smoothing
  experiment — noise std=30 avoids silently flooring
  `_texture_adaptation_factor`), 3 mark radii (4, 8, 15px), dimensional
  zone centered on the mark (worst case — full overlap).
- A geometry check with the dimensional zone **offset 22px from the
  mark's center** (partial overlap, the realistic case — a mole near,
  not on top of, a cheek highlight), to check for a boundary artifact at
  the exclusion edge.
- A real face: the same DSCF2306 + composited-mark scene from the FA-01
  smoothing experiment (`/tmp/real_face_composite.png`, real skin
  texture/lighting, mark at a hand-picked coordinate, radius 6).

---

## 3. Results

### Full-overlap synthetic scene (dimensional zone centered on the mark)

| Radius | Policy | smoothed contrast | `current` restored | `current` recovery | `defect_excluded` restored | `defect_excluded` recovery |
|---|---|---|---|---|---|---|
| 8 | none | 6.82 | 25.96 | **+19.14** | 6.80 | −0.02 |
| 15 | none | 8.20 | 27.73 | **+19.53** | 9.45 | +1.25 |
| 8 | mark_protect | 76.80 | 76.78 | −0.02 | 76.78 | −0.02 |
| 15 | mark_protect | 72.56 | 73.81 | +1.25 | 73.81 | +1.25 |

(radius 4 shows no effect either way — the mark is smaller than the
guided filter's own smoothing kernel at this scene's `face_width`, so
smoothing barely touches it in the first place; not evidence against the
mechanism, just outside its operating range at this scale.)

**With no mark policy** (the shipped default), `current` recovers 19+
points of contrast smoothing had suppressed — a real, visually confirmed
defect restoration (crop comparison: `/tmp/fa02_comparison.png`,
generated during this session; the mark is visibly darker in `current`
than in `disabled`, and `defect_excluded` matches `disabled`).
`in_mark_mean_abs_delta_vs_smoothed` for `current` is ~15 vs. ~0.3–0.8 for
`defect_excluded` — an order of magnitude difference, not a marginal one.

**With `mark_protect` active**, the recovery gap between `current` and
`defect_excluded` closes to exactly zero at both radii (identical to
machine precision). This means **FA-01's shipped smoothing protection
already closes most of this hole when a mark policy is active** — because
`pre_smooth − smoothed` is already near-zero inside a protected mark, so
there is little detail left for `restore_micro_texture` to restore or
exclude either way. The residual +1.25 at radius 15 is real dimensional
detail near the mark, not the mark itself (both arms recover the same
amount, so exclusion isn't suppressing anything there).

**The live exposure is squarely `mark_policy=None`** — the default every
one of the 4 recipes that set `micro_restore` actually runs, unless the
caller separately opts into a mark policy.

### Real face (DSCF2306 + composited mark, radius 6, no mark policy)

| Arm | restored contrast | recovery vs. smoothed (27.54) |
|---|---|---|
| disabled | 27.54 | +0.00 |
| current | 46.06 | **+18.53** |
| defect_excluded | 32.71 | +5.17 |

Confirms the synthetic result on real skin texture and lighting: `current`
recovers +18.5 points; `defect_excluded` recovers only +5.2 (residual is
real dimensional detail near the mark, not the mark's own contrast — the
crop comparison, `/tmp/fa02_real_comparison.png`, shows `current`
visibly darker at the mark than `disabled`, and `defect_excluded` close
to `disabled`). With `mark_protect` active on this scene, `current` and
`defect_excluded` are identical (80.22 both) — same pattern as synthetic.

### Boundary-artifact check (offset geometry, partial overlap)

Dimensional zone offset 22px from the mark center (radius 8, no policy).
Sampling the row through the mark center, `defect_excluded − current`:
zero from x=110–118 and x=138–150, rising smoothly to a peak of ~21 at
x=127–129 (the mark's own center) and tapering symmetrically back to
zero — a **bell-shaped profile matching the mark's own blurred footprint,
not a step at the mask-subtraction boundary**. Visual crop comparison
(`/tmp/fa02_offset_comparison.png`) confirms: no ring, halo, or
discontinuity where the mask was clipped; `defect_excluded` closely
matches `disabled` and both differ smoothly from `current` only over the
mark's own footprint. This is a single geometry (radius 8, one offset,
one mark shape) — not a sweep — but it directly answers the concrete
worry raised before writing this up (that hard-subtracting a mask from
`dim_mask` could create the same island-artifact class rejected for the
7 shading ops in `154d854`): at this geometry it does not, plausibly
because the subtraction happens on `dim_mask` itself, which
`restore_micro_texture` already blends via its own
`restore_amount * dim_mask_3d` multiplication — no new hard edge is
introduced beyond what the function's existing blend already smooths.

---

## 4. The policy-gated fix does not reach the exposure it was meant to close

The band-selective exclusion described above (subtract the mark's
protected footprint from `dim_mask` before `restore_micro_texture`'s
existing `restore_amount * dim_mask_3d` blend) is mechanically sound and
was validated with no boundary artifact — but checking
`perf_optimizations.py:361-424` before implementing it surfaced a
disqualifying fact: **`mark_protect_mask` is `None` unless
`_mark_policy is not None`.** The +18–19.5 point defect-restoration effect
measured in §3 occurs specifically under `mark_policy=None` — the shipped
default for every recipe that sets `micro_restore` — and that is exactly
the path with no mark mask to subtract. A caller who opts into a mark
policy already sees the `current`/`defect_excluded` gap at ~0 (§3), so
wiring the exclusion under the existing opt-in-only contract would close
a residual that is already negligible, while leaving the real, measured
+18.5pt exposure completely untouched. This was caught before writing any
implementation code, not after.

## 5. Investigated and rejected: a mark-independent restoration cap

Since the real exposure lives on the no-policy default path, a fix that
needs no mark detection at all was investigated: cap
`restore_micro_texture`'s restoration wherever `detail = pre_smooth -
smoothed` is unusually large and spatially concentrated, reasoning that a
mark produces a strong, tightly-localized removal while authentic
pore/skin texture produces a weaker, more diffuse one.

**On the synthetic (noise-only) canvas this looked strong**: mark-region
`detail` magnitude was ~30x the plain-texture region's (54.59 vs. 1.78 mean
absolute delta), and thresholding at the 90th percentile isolated a single
~386px connected component closely matching the mark's own area (a
radius-8 circle ≈ 201px before feathering).

**On a real face this discriminator fails.** The same measurement on a
clean DSCF2306 crop (no mark composited) already produces connected
high-detail components of 495, 455, 453, 366px — as large as, or larger
than, the mark's own ~490px component when a radius-6 mark *is* present,
with near-identical component magnitudes in both cases (mark: top
components 490/455/453/366px at 49.8/49.1/54.1/61.0 mean magnitude;
clean: 495/455/453/366px at 49.9/49.1/54.1/61.0 — the mark is not even
distinguishable in the ranked list). Real hair strands and skin-fold
shadows already produce detail blobs at the same scale and amplitude as
a small mark. **Rejected**, not just "not yet tuned": this is the same
class of scene-construction confound as the `_texture_adaptation_factor`
noise-floor bug caught earlier in this session (a synthetic canvas with
no coherent structure gave a clean, misleadingly strong separation that
real skin's own structure does not support) — no threshold on `detail`'s
magnitude or connected-component size can be expected to reliably
separate "mark" from "authentic strong texture" on real portraits, because
`detail` itself carries no information about *why* smoothing removed a
given signal. Distinguishing "this was a mark" from "this was a hair
strand or a fold shadow" requires knowing what the pixel *is* — which
means detection, not a property derivable from the smoothing residual
alone. This is a structural conclusion about the mechanism, not a
threshold-tuning failure that a different percentile or feature could fix.

**One geometry only** (one crop, one mark radius/position, one smoothing
strength) — a larger mark or a flatter-skin crop might separate more
cleanly. Named as a limitation, not chased further: the mechanism-level
argument (the residual carries no provenance) holds regardless of whether
one particular radius happens to separate on one particular crop.

## 6. Recommendation

No mark-independent fix survives measurement (§5). The only two
mechanisms that can close the `mark_policy=None` default-path exposure
measured in §3 are:

- **Run `detect_marks` unconditionally** for this purpose, closing the
  real gap but breaking the `mark_policy=None` byte-identical contract
  every FA-01 change to date has preserved, and adding per-face detection
  cost to every default render (not benchmarked here).
- **Accept the gap on the default path** and only wire the exclusion for
  `mark_policy` opt-in users (§4) — small, in-scope, honest about the
  fact that it does not address the measured +18.5pt exposure, since that
  exposure is specifically where no mark mask exists to exclude.

**This document does not choose between them.** Both are real behavior-
policy decisions belonging to the owner, not a default this research
tranche can select. Per the plan and this session's established pattern,
evidence is produced here; the choice and its implementation are a
separate, explicitly authorized step.

## 7. Limitations, stated plainly

- One mark shape (circular), 3 radii, one dimensional-zone size (30px
  cheek-highlight circle), one offset distance (22px) — not a sweep over
  zone size, offset distance, or mark shape. The §5 rejection is one
  crop/radius/strength for the mark-independent discriminator specifically
  (see its own caveat) — larger marks or flatter-skin crops were not
  checked and might separate more cleanly, though the structural argument
  (no provenance in the residual) does not depend on that.
- No genuine natural mole in the corpus (CLAUDE.md known limitation,
  same caveat as every FA-01 mark experiment); real-face evidence is
  semi-synthetic (real skin + composited mark).
- `pore_synthesis` (mechanism 1) was not evaluated here — it was ruled
  out as the FA-02 subject on ordering grounds (§1), not tested and found
  safe. If `pore_synthesis`'s independently-seeded noise happens to land
  on a protected mark, it has never been measured.
- Only the `guided` smoothing engine's interaction was tested (inherited
  from FA-01's own scope boundary — `mark_protect` is guided-only).
- Runtime cost of computing `mark_protect_mask` a second time for this
  purpose (if implemented via a fresh `detect_marks` call rather than
  reusing the one already computed in `_process_face_core`) has not been
  measured; the FA-01 doc's runtime benchmark covered `frequency.combine`
  only, not this call site.
