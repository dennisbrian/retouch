# RESEARCH — Frontier AE-series: reshape warps ignore head roll, punch a
# lopsided bulge into rolled/tilted portraits

**Date:** 2026-07-20 · **Author:** Fable, triggered by owner-reported visual
defect ("getting punch") on a real cosplay portrait (`DSCF8777.jpg`, the same
photo used to reproduce the AD occlusion-masking bug earlier this session).
**Method:** direct reproduction against the real photo (not simulated),
component isolation (`slimming=0` vs as-is), and code reading at HEAD to
confirm mechanism — every claim below is either an executed measurement or a
cited line anchor.
**Rules inherited:** classical/deterministic, unit-testable, no learned
models beyond what's already shipped, tone-fair, no absolute intensity
thresholds, no identity-changing defaults.

**Fixed (2026-07-20):** shipped in the working tree — a new
`_face_local_axes()` helper on `FaceReshaper` (`geometry.py`), returning
the face's own horizontal (jaw-to-jaw, landmarks 234→454) and vertical
(forehead-to-chin, landmarks 10→152) unit vectors, rotated with however the
head is actually posed in frame. `_slimming_warps`, `_jaw_width_warps`, and
`_smile_warps` — the three functions found to hardcode raw screen `±x`/`±y`
displacement — now scale their intended displacement magnitude along these
face-local axes instead. 8 new regression tests in
`tests/test_geometry.py::TestRolledFaceWarpDirection`; all pass, and were
verified to fail without the fix via `git stash` (5 of 7 direction-sensitive
tests fail pre-fix; 2 environment/no-crash tests correctly pass either way).
30/30 `test_geometry.py` pass overall (22 pre-existing + 8 new), 79/79 across
`test_geometry.py` + `test_liquify.py` + `test_body_reshape.py`. Re-rendered
`DSCF8777.jpg` with the `cosplay` recipe post-fix: the lopsided cheek/jaw
bulge is gone, matching the `slimming=0` isolation reference exactly in that
region.

One measurement pitfall found while isolating the cause, flagged by
`advisor()` before any code was written: the first candidate render used for
diagnosis (`/tmp/occlusion_test_DSCF8777.jpg`, held over from the AD session)
had no known recipe/params — its generating script was gone. Rather than
assume it was `cosplay` and build a 3-function fix on that assumption, first
re-rendered with explicit, known params (`recipe="cosplay"` vs
`recipe="cosplay", slimming=0`) and confirmed by direct visual comparison
that disabling slimming alone removes the bulge — ruling out color grading,
bloom, relight, or the underlying pose/lighting as the cause before writing
any geometry code.

---

## 0. TL;DR

Owner observation, verbatim: "want to ask something i do see this pic
DSCF8777.jpg but the issue the retouch seem like weird like it is getting
punch?" Reproduced on the same real photo used for the AD occlusion bug — a
close-up cosplay portrait with the head rolled **~52° from upright** (a
tilted, near-sideways framing, chin toward lower-left, crown toward
upper-right). Rendering with the `cosplay` recipe's default `slimming=50.0`
produces a visibly lopsided, puffy bulge along one side of the
cheek/jaw/mouth contour — not a clean inward "slimming" compression, closer
to a swollen/punched look, exactly as the owner described.

**Root cause:** `geometry.py::_slimming_warps` (and two siblings,
`_jaw_width_warps` and `_smile_warps`) compute their target displacement as
a **raw screen-space offset** — `(c[0] + fw*k, c[1])`, always horizontal in
image pixels, or `(c[0], c[1] - fw*k)`, always vertical in image pixels —
regardless of how the face is actually rotated in the frame. This is correct
*only* when the face is upright (jaw-to-jaw axis ≈ screen-horizontal). On a
rolled face, the intended "pull the jaw toward the face center" displacement
gets misdirected by the roll angle, pushing tissue sideways across the face
(along a direction closer to the face's own *vertical* than its horizontal)
instead of compressing it toward center — visually reading as one side of
the face swelling relative to the other.

---

## 1. Reproduction (verified by execution)

`DSCF8777.jpg` (owner-provided, same file used for the AD bug this session):
1 face detected, confidence 0.999, bbox `(1183,1205,1362,1417)`.

Measured jaw-to-jaw axis (landmarks 234→454, in image-pixel space):
`52.0°` from screen-horizontal. Measured forehead-to-chin axis (landmarks
10→152): `37.3°` from screen-horizontal — consistent with a single coherent
head roll of roughly that magnitude, not measurement noise.

Component isolation, both rendered with `RetouchEngine().process(img,
recipe="cosplay", ...)`, known params, no caching path involved:

| Render | `slimming` | Cheek/jaw contour (viewed directly, full-res crop) |
|---|---|---|
| `cosplay` as-is | 50.0 (recipe default) | Visible lopsided puffy bulge, lower-right cheek down through the jaw |
| `cosplay`, override | 0.0 | Smooth, natural contour — bulge absent |
| `cosplay` (post-fix) | 50.0 (recipe default, fix applied) | Matches the `slimming=0` reference — bulge absent |

This isolates the defect to the slimming warp specifically (not color
grading, bloom, dodge/burn, or relight, all of which are identical across
the first two rows) before any code was changed.

---

## 2. Root cause (verified by reading + direct measurement, line-anchored)

`geometry.py::_slimming_warps` (pre-fix, lines 333-339 in the version at
session start):

```python
c_rjaw = (int(landmarks[234].x * w), int(landmarks[234].y * h))
t_rjaw = (int(c_rjaw[0] + fw * 0.04 * s), c_rjaw[1])   # pure +x, zero y
warps.append((c_rjaw, t_rjaw, int(fw * 0.5)))

c_ljaw = (int(landmarks[454].x * w), int(landmarks[454].y * h))
t_ljaw = (int(c_ljaw[0] - fw * 0.04 * s), c_ljaw[1])   # pure -x, zero y
warps.append((c_ljaw, t_ljaw, int(fw * 0.5)))
```

Landmarks 234/454 are MediaPipe's fixed *anatomical* right/left jaw points —
they track the same physical spot on the face regardless of camera roll.
The warp displacement, however, is always `(±fw*0.04*s, 0)`: a fixed
screen-space vector. On an upright face this is fine, because the true
jaw-to-jaw axis is close to screen-horizontal, so `(±x, 0)` approximates
"pull toward center." On a rolled face it doesn't — measured directly on
this photo:

```
jaw-to-jaw axis (normalized): [ 0.616  -0.788 ]  → 52.0° from screen-horizontal
warp direction (hardcoded):   [ 1.0     0.0   ]  → 0.0° from screen-horizontal
angle between them: 52.0°
```

At 52° off-axis, `cos(52°) ≈ 0.62` of the intended displacement magnitude
still pulls "inward-ish," but `sin(52°) ≈ 0.79` of it is misdirected
**sideways across the face**, in a direction close to the face's own
up-down axis — exactly the failure mode that reads as one cheek/jaw region
bulging toward another.

**Two more functions share the identical pattern**, found by grepping every
warp-target construction in the file for raw `±x`/`±y` landmark offsets
(`grep -n '\[0\] +\|\[0\] -\|\[1\] +\|\[1\] -' geometry.py`):

- `_jaw_width_warps` (pre-fix lines 529-537) — the standalone "jaw width"
  slider (independent of `slimming`, its own docstring says so at line
  514), same `c[0] ± disp` construction on the same 234/454 landmarks.
- `_smile_warps` (pre-fix lines 596-604) — "lift mouth corners up +
  slightly out," using raw `disp_y = -fw*k` (screen `-y`) and
  `disp_x = ±fw*k*0.5` (screen `±x`) on landmarks 61/291.

**Contrast: the already-correct pattern in the same file.**
`_eye_distance_warps` (line 419, unmodified) computes its axis from actual
landmark geometry instead of hardcoding it:

```python
axis = np.array([p_inner_r[0] - p_inner_l[0], p_inner_r[1] - p_inner_l[1]])
dir_vec = axis / np.linalg.norm(axis)
```

This is inherently rotation-invariant — exactly the template the fix
generalizes. Radial-scale warps (`_eye_size_warps`, `_nose_width_warps`,
`_mouth_size_warps`) are unaffected by roll for a different reason: they
scale symmetrically around a center point rather than translating along a
fixed axis, so they have no "wrong direction" to have.

---

## 3. Fix (implemented)

New helper, `FaceReshaper._face_local_axes(landmarks, w, h)`:

```python
p_r = landmarks[234]; p_l = landmarks[454]   # jaw-to-jaw
horiz = normalize(p_l - p_r)

p_top = landmarks[10]; p_chin = landmarks[152]   # forehead-to-chin
vert = normalize(p_chin - p_top)
```

Both fall back to the legacy screen axes (`[1,0]` / `[0,1]`) if the two
landmarks coincide (near-zero-norm guard) — a degenerate-input safeguard,
not expected in practice with real detections.

`_slimming_warps`, `_jaw_width_warps`, `_smile_warps` rewritten to scale
their existing displacement *magnitude* (unchanged — same `fw * k`
coefficients as before, calibration untouched) along `horiz`/`vert` instead
of raw `(±1, 0)`/`(0, ±1)`. At zero roll this is mathematically identical to
the old behavior (confirmed: `test_slimming_upright_byte_identical_to_legacy_axis`
passes both pre- and post-fix), so the common upright-portrait case is
unaffected.

**Deliberately not fixed in this pass:** `_face_width()` (line ~691-695)
computes face width as `abs(x_454 - x_234)` — an x-only delta that also
implicitly assumes an upright face, underestimating the true jaw-to-jaw
distance on a rolled face. This feeds the magnitude (not direction) of
nearly every warp in the file via the shared `fw` parameter. Only 2 call
sites, but changing it rescales every warp's magnitude across the whole
reshape pipeline — a materially bigger, higher-risk change than the
direction fix, and not needed to resolve the confirmed visible symptom
(the isolation test in §1 shows direction alone was sufficient). Flagged
here as a candidate follow-up, not fixed speculatively.

**Tests:** `tests/test_geometry.py::TestRolledFaceWarpDirection` (8 tests) —
axis-helper correctness at 0° and 52° roll, per-function direction checks
for all three fixed functions (jaw pull, jaw-width pull, smile lift —
isolating the "up" component from the smile warp's combined up+out
displacement by summing both mouth corners, since the "out" components are
equal-and-opposite and cancel), a byte-identity check at zero roll, and an
end-to-end no-crash/no-silent-no-op pipeline check. Verified via
`git stash push -- retouch/geometry.py` that 5 of 7 direction-sensitive
tests fail without the fix (the other 2 are correctly roll-agnostic: the
upright-identity check and the crash/no-op check).

---

## Bounds / NO-GO

- Pure geometric fix — no new detection capability, no learned model, no
  absolute intensity/color threshold. Tone-fair by construction (roll angle
  is a property of head pose, not skin tone).
- Does not touch warp *magnitude*/calibration (`_EYE_SIZE_K`,
  `_JAW_WIDTH_K`, etc.) or any other reshape function beyond the three
  confirmed to use raw screen axes.
- `_face_width`'s x-only magnitude approximation is a known, related, but
  separate gap — explicitly deferred, not silently left unmentioned. It is
  provably safe to defer: `_face_width` returns `abs(x_454 - x_234)`, which
  on a rolled face equals `true_jaw_distance * cos(roll)` — always an
  *under*-estimate, never an over-estimate, so every warp's magnitude is
  attenuated (weaker), not amplified. On `DSCF8777.jpg` specifically
  (measured from §1's coordinates: x-delta 753.7px vs. true jaw distance
  1223.5px → ratio 0.616 ≈ `cos(52°)`), post-fix slimming is running at
  ~62% of its nominal strength — correctly *aimed* now, but under-powered
  until `_face_width` is also fixed. This is why the post-fix render in §1
  reads as near-identical to the `slimming=0` isolation reference in that
  crop: the two effects (correctly-aimed-but-attenuated vs. fully-disabled)
  happen to look similar at this specific roll angle and crop distance —
  it is not evidence that slimming stopped doing anything post-fix (the
  unit tests confirm nonzero, correctly-directed displacement magnitude
  directly). An under-measured `fw` can never *reintroduce* the bulge,
  which is what makes deferring it safe.

## Sources

- Direct reproduction this pass: `DSCF8777.jpg` (owner-provided, not
  committed to the repo; same file as the AD-series occlusion bug).
- MediaPipe 478-point landmark indices (234/454 jaw, 10/152
  forehead/chin, 61/291 mouth corners) — standard indices already in use
  elsewhere in `geometry.py` and `parsing.py`, no new citation needed.
