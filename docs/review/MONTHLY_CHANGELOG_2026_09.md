# Monthly Changelog — September 2026

Rollup of `feat`/`fix` commits on `main` for 2026-09-01 through 2026-09-17
(session cutoff; month not yet complete at time of writing). Generated from
`git log --grep "^feat"` / `--grep "^fix"` — not a hand-maintained log, so
re-derive from git history rather than editing this file incrementally.

---

## New Features

### Engine / color science
- **BB3 — real EXR float32 I/O** (`6795b7b`): proper HDR round-trip in
  `io.py`. `OPENCV_IO_ENABLE_OPENEXR=1` set before `cv2` import; reader
  tone-maps HDR with an above-1.0-only Reinhard shoulder (SDR pixels
  round-trip exactly, max err < 0.01); writer exports scene-linear float32.
- **K7 — ST 2084 PQ encoding** (`6795b7b`) on EXR export via
  `write_image_with_icc(..., pq_encode=True)`.
- **K6 — multi-illuminant skin adaptation** (`6795b7b`): new key-light /
  fill-light Kelvin + mix ParamSpecs (`mi-key-kelvin` / `mi-fill-kelvin` /
  `mi-mix`, neutral defaults = byte-identical no-op), wired into
  `_stage_grade` skin-masked via `acc_skin`. GUI sliders added under "LCH
  Color Tools." Verified on the golden face fixture: active
  3200K/7000K/mix50 = mean 32.2 delta, neutral = byte-identical.
- **K5 — gamut target control** (`6795b7b`): `gamut_target` ParamSpec
  (srgb/p3/rec2020) threaded through `grade()` into `_apply_gamut_compress`;
  P3/Rec.2020 boundaries gate the in-gamut short-circuit. Verified monotonic
  chroma retention 0.209/0.261/0.292; default srgb byte-identical.
- **K1 — new QA detector** (`6795b7b`): `detect_cam16_delta_e`
  (CAM16-UCS skin delta-E vs reference, thresholds mean>10 / p99.5>30),
  registered in `ALL_DETECTOR_NAMES`, wired into `--fail-on-qa`.
  Stride-subsampled above 256px (4MP: 2718ms → 71ms) to keep the QA hot
  path affordable.
- **P7 — cross-region skin appearance propagation** (`c0e0210`, hardened
  `ee88d3d`/`cea7b2e`, exposed `cabf3bc`): opt-in primitive for bounded
  skin-tone consistency across face regions. Exposed via ParamSpec, CLI,
  and an experimental GUI control; every recipe default keeps it disabled.
- **P8 R1 — observable appearance readout** (`cabf3bc`): signed
  feature-contrast and skin-chroma-variation metrics via a non-rendering QA
  command, with explicit support states. Unvalidated age slider and vector
  model deliberately deferred.
- **Eye-visibility gate hardening** (session work, `salvage-k9` →
  `9c2750b`, rebased onto `main`): `eye_gate` user control (GUI checkbox,
  default on) to skip enhancing closed/occluded eyes; per-eye-support
  artifact-scale guard (`eye_artifact_safety.py`) that smoothly backs off
  enhancement strength for small/low-confidence irises; per-eye (not
  pooled) catchlight compositing to avoid coupling the two eyes.

### CLI
- **`--preflight-check`** (BB6, `6795b7b`): opt-in integrity checks + a
  one-image 720p throughput benchmark, exit 1 on failure. Never runs
  implicitly (no per-init overhead added).
- **Disk-space preflight** (`bc2ac4b`): estimates output size as
  `input_bytes × 2.0` and warns/exits before a batch run would drop free
  space below 5 GiB; `--skip-disk-check` bypasses it.

### Recipes
- `meitu_porcelain_v1` (`6ce4b2f`, exposed experimental `911572c`) —
  Meitu-style porcelain-skin candidate recipe.
- 5 new recipes exposing recently-shipped, under-reached capabilities
  (`0b6dad5`).
- `fa02_texture_experimental_v1` (`0947d76`) — research-only
  texture-restoration recipe.

### QA / research infrastructure (internal, not user-facing)
- FA-02 texture-restoration scaffold, scoring contract, and
  `corpus_manifest` v3 governance bridge (`cce14e5`, `51cbabc`, `067cc22`,
  `652ee53`, `9de77af`, `d8e2171`).
- FA-01 BiSeNet boundary-ring parse confidence signal, observation-only
  (`e3c33ac`, `6b1e72d`).
- Yaw-gate / eye-artifact-scale backoff decision logging (`8a2869e`).
- Gap-research corpus inventory merge tool and baseline-decision freeze
  scripts (`7eaf892`, `849e487`).

---

## Bug Fixes

### Correctness bugs that silently disabled retouching (highest impact)
- **`f69ab1e`** — mask-epsilon bug: every mask normaliser used
  `max() > 1.0` to detect 0–255 masks; float32 region masks overshoot 1.0
  by one ulp (1.0000002) after feathering on ~24% of real portraits (20/83
  DSCF corpus), so skin/hair/neck masks were divided by 255 and composite
  alpha over skin collapsed to 1/255 — smoothing/equalize/blemish/
  under-eye/relight/sculpt were silently discarded. Threshold raised to
  `> 1.5` + clip, applied at 21 sites. **Any pre-2026-09-02 "this op does
  nothing on image Y" conclusion may have been this bug, not the op.**
- **`fb2474e`** — `skin.harmonize_neck`'s depth gate kept pixels within
  0.15·face_w of an eye-corner/nose-bridge plane measured in XY only; with
  z scaled by face width the plane tilts into the image, so the gate was a
  guaranteed no-op on every frontal face (15/15 corpus faces lost 100% of
  the neck mask). Gate removed; replaced with a chroma gate.
- **`c89e65f`** — dark-circle op v1 kept the darkest ≥100px component of
  the landmark under-eye polygon, which on real faces is the lower lash
  line, not the shadow — 80% of the lift landed on lashes, 0.03 L on skin
  (146 corpus eyes). Inert on every face despite being set in 55/128
  recipes. Replaced with v2 (`retouch/undereye.py`): support extended into
  the tear trough, masked low-pass darkness relative to a cheek-ring
  median, tone-invariant.
- **`ce56ba2`** + **`9c26493`** — yaw-gate ramp returned the raw smoothstep
  (0→1) instead of 1−smoothstep, so relight/sculpt/slimming got ~0 strength
  just past the gate start and ~full strength near the end (inverted from
  intent). Band also recalibrated from 1.3–1.6 / 1.5–1.7 to a shared
  2.5–4.0, mapped to ~2–10° of head turn (previously zeroed slimming on
  58/83 corpus faces at minimal turn).
- **`e73d3ba`** — `dark_circles` and `undereye_darken_removal` were
  double-applying (summed) instead of aliasing to max, across 27 recipes.
- **`1b3e341`** — clarity op never ran its float path, injecting a
  LAB-roundtrip noise artifact instead of the intended clarity effect.
- **`4aa87e5`** — eye-v0 dispatch wrapped the whole canvas in a
  float→uint8→float roundtrip around `EyeEnhancerV0.enhance()`, dithering
  every non-eye pixel in the ROI (wig, hand, background) by ≤1 level even
  though the edit is masked to sclera/iris. Roundtrip was unneeded — the
  enhancer already dispatches on dtype internally.

### Containment / safety fixes (an op leaking outside its intended region)
- **`46be031`** — lips/teeth whole-image LAB roundtrips contained to their
  actual support via `utils.restore_outside_support`.
- **`c665da1`** — guided-smoothing blend was overriding policy-approved
  mark protection.
- **`154d854`**, **`f3b1008`** — `mark_policy` preserve-mask wasn't wired
  into `blemish.remove` or 9 "evening" skin ops.
- **`b39bd94`** — under-eye darken removal wasn't capped, risking erasure
  of intentional cosplay contour makeup.
- **`0071447`** — `meitu_porcelain_v1`'s luma lift routed through a hole in
  the skin mask at the nose (skin mask excludes the nose region, label 10
  vs skin label 1).
- **`fb6ba5e`** — `mid_reduction` in `FrequencySeparator.combine` wasn't
  clamped to `[0, 1]`.
- **`9f2380b`** — texture-adaptation floor recalibrated; guard extended to
  `mid_reduction`.

### Color / IO correctness
- **`bcf07db`** — gamut mapping had two different mappers instead of one
  correct pre-quantization mapper; unified.
- **`f3bcff9`** — RAW/ICC/high-bit/alpha color ingest and linear TIFF
  export were broken.
- **`e604c87`** — path traversal components weren't rejected on file IO
  (security-adjacent).
- **`c7a8650`** — neural booster didn't handle float32 input.
- **`ee88d3d`**, **`cea7b2e`** — P7 LAB delta wasn't clamped before
  color-space conversion; hardened LAB/compatibility contracts.

### Process reliability
- **`cef4eff`** — the actual fix for the multi-face `cli.py --workers>1`
  batch hang: `RetouchEngine`'s inner `FaceProcessorPool` spawns non-daemon
  grandchild processes that `multiprocessing`'s own exit handler joins
  before the outer worker can reach interpreter shutdown, making the
  atexit-based cleanup structurally unreachable. Fixed by shutting the
  inner pool down inline, synchronously, at the end of each task.
- **`c8ddc6f`** — earlier attempt at the same hang (close per-worker
  `RetouchEngine` on exit) — insufficient for the multi-face case,
  superseded by `cef4eff`.
- **`ee5ef6d`** — CI matrix/benchmark gates repaired.
- **`6a9de3b`** — QA evidence was being discarded except for flagged
  results; now preserved complete.
- **`cf7998a`**, **`b9292e6`** — denominator / determinacy bugs in the
  gap-research QA scripts (`baseline_freeze`, P7 diagnostics reporting).

### GUI polish
- **`b2335e1`**, **`2e88a89`**, **`b01a9fb`** — removed a dead checkbox,
  reorganized Develop accordions into sections, polished the Smart Color
  card, fixed the light-mode recipe list, unburied primary action buttons.

### Recipe tuning
- **`0537bbc`** — `cinema_grade_v1` grain reduced from 0.07 to 0.03 (was
  visibly coarse noise, not intended film grain texture).

---

## Notable pattern this month

Several of the highest-impact fixes (`f69ab1e`, `fb2474e`, `c89e65f`,
`ce56ba2`) share a shape: an op looked wired and tested, but a boundary
condition (epsilon overshoot, XY-only depth math, wrong polygon edge,
inverted ramp direction) made it a silent no-op or near-no-op on a large
fraction of real portraits while unit tests on synthetic/uniform inputs
stayed green. The common fix pattern was testing against a real corpus
(the 83-image DSCF set) rather than trusting midpoint-only or
uniform-input unit tests — see `f69ab1e`'s note that "any past 'op X does
nothing' conclusion may have been this."
