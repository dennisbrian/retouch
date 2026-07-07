# PLAN_F5_LIQUIFY — Face-Aware Liquify Sliders

> **Master-plan row 19** (`MASTER_PLAN.md:85`): *Face-aware liquify sliders — 2 wk, no deps.*
> **Tier-2 plan source:** `PLAN_TIER2_MANUAL_TOOLS.md:39-58` (F5 section).
> **Type:** MEDIUM (Visual-Critical — touches `geometry.py`, a pipeline stage).
> **Status:** DESIGN — not yet implemented.

---

## 1. Motivation

Photoshop's *Face-Aware Liquify* is the single feature cosplay/portrait editors
still leave the app for. Today the engine ships **one** reshape knob —
`slimming` (0–100) — which applies a fixed jaw + cheek + chin compression
(`retouch/geometry.py:15-120`, `FaceReshaper.reshape()`). It works, but it is
all-or-nothing: there is no way to enlarge eyes, narrow a nose, lift cheeks,
or reshape a chin independently, and the negative direction (enlarge) is not
even allowed (`slimming` min is 0, `retouch/params.py:716`).

F5 turns that one knob into a **slider bank**, each slider driving a named
facial warp. The design constraint is parity with the manual tool's *feel* —
slider-based, automatic per face, no brush — not free-form mesh editing.

### Design principles (in priority order, per `AGENTS.md`)

1. **Visual Fidelity > Correctness > Performance > Elegance.** A warp that
   reads as "plastic surgery" is reverted, even if the math is correct.
2. **Reuse the tested warp engine.** The existing `(1 − d²/R²)²` translation
   kernel + single `cv2.remap` pass (`geometry.py:113-133`) is already
   artifact-free on the test corpus. F5 adds warp *definitions*, not a new
   deformation math. A scale warp = N radial translation warps arranged around
   a center — same kernel, no new code path.
3. **Symmetric by construction.** Every bilateral slider (eyes, cheeks, jaw)
   emits a mirrored pair of warps in one call; there is no "left only" path.
4. **Bounded by face geometry.** All displacements scale with `face_width`
   (inter-temple distance, `|454 − 234|`), never absolute pixels — so the
   same slider value reads the same on a 4K crop and an 800px preview.

---

## 2. Warp Math

### 2.1 Why not RBF / MLS

The plan brief asks about RBF (radial basis function) and Moving-Least-Squares
(MLS) deformations. Both are the textbook answer for free-form mesh liquify,
but **neither is needed here**, and adopting either would violate principle
#2 (reuse tested engine) and the `AGENTS.md` anti-artifact rules
("no new math for a feature that the existing math covers").

| Approach | Pros | Cons for F5 |
|---|---|---|
| **RBF (Wendland / Gaussian)** | Smooth global field from sparse control points | Solves a dense linear system per face; over-smooths far-field (background lines bow — fails the F5 background-line gate, `PLAN_TIER2_MANUAL_TOOLS.md:53`); new code path = new artifact surface |
| **MLS (affine / similarity)** | Rigidity-preserving, the academic gold standard | O(N·P) per pixel for P control points; the rigid variant still drags background near the warp radius; new code path |
| **Existing radial falloff** (`geometry.py:113-120`) | Already tested, already artifact-free, single `cv2.remap` pass, bounded support (zero outside `R`) | Cannot do true global rigidity — but F5 doesn't need it, because every warp radius is capped at `face_width × 0.5` and the background-line gate is enforced by *radius discipline*, not by the kernel shape |

**Decision: keep the existing kernel.** F5's quality bar is met by composing
many small radial translation warps (the same primitive `geometry.py` already
loops over) into scale and directional deformations. This is exactly how the
tier-2 plan frames it: *"Scale warps = multiple translation warps arranged
radially (reuses the same falloff kernel — no new math)"*
(`PLAN_TIER2_MANUAL_TOOLS.md:48`).

If a future stage (T3 body reshape, row 35 — depends on "F5 warps") needs
true rigidity across a larger area, MLS can be revisited then as a
*separate* kernel behind the same `_apply_warps` interface. F5 does not
block on it.

### 2.2 The kernel (existing, unchanged)

For each control point `C` with target `T` and radius `R`, the displacement
field at pixel `p` is (`geometry.py:104-120`):

```
d(p) = (T − C) · w(p),   where   w(p) = (1 − |p−C|² / R²)²   for |p−C| < R, else 0
```

`map_x` / `map_y` are decremented by `w · (T−C)` (source-lookup convention),
then a single `cv2.remap(..., INTER_LINEAR, BORDER_REFLECT_101)` applies all
accumulated warps at once (`geometry.py:127-133`). Multiple warps whose radii
overlap simply sum in the displacement maps — this is already how jaw + cheek
+ chin coexist today.

### 2.3 Composing a *scale* warp from translations

To enlarge or shrink a feature (eye, mouth) by factor `(1 + k)` around center
`C`, emit `N=8` radial translation warps on a circle of radius `r₀` around
`C`, each moving *outward* (enlarge, `k>0`) or *inward* (shrink, `k<0`) by
`|k| · r₀`. This is the standard "radial pinch/bloat" decomposition and it
falls out of the existing kernel for free:

```
for i in range(N):
    θ = 2π·i / N
    P_i = C + r₀·(cos θ, sin θ)          # control point on the ring
    T_i = C + (1+k)·r₀·(cos θ, sin θ)    # target pushed outward (or inward)
    warps.append((P_i, T_i, R_ring))     # R_ring ≈ 2·r₀ to overlap neighbors
```

The overlap of the 8 falloffs approximates a smooth radial scale. Because each
individual warp is bounded and C¹-smooth at its boundary, the composed field
is too — no seam, no new discontinuity to test.

### 2.4 Displacement caps (anti-"plastic surgery")

Per `PLAN_TIER2_MANUAL_TOOLS.md:54`: *"warp displacement magnitude capped
(≤ face_width × 0.06 at strength 100)"*. Each slider's `k` is mapped through
a per-slider `max_displace(fw)` so that **at slider=±50** (the GUI default
range) no single control point moves more than `fw × 0.06`, and at the
extended ±100 cap no point moves more than `fw × 0.12`. These caps are
enforced in `_apply_warps` as a final clamp on `(T − C)`, not as a slider
range limit — so recipe authors can't bypass them by writing raw recipe JSON.

---

## 3. Sliders (public API)

Nine new `ParamSpec` entries under a `"reshape"` recipe namespace, plus the
existing `slimming` (which becomes `reshape.slimming` for recipe-key
consistency, kept at top-level `slimming` for backward compat — see §5.2).

| Slider | ParamSpec name | Range | Default | Recipe key | What it does |
|---|---|---|---|---|---|
| Face Slimming | `slimming` (existing) | 0–100 | 0 | `slimming` | Jaw + cheek + chin inward compression (current behavior, unchanged) |
| Eye Size | `reshape_eye_size` | −50..+50 | 0 | `reshape.eye_size` | Radial scale of both eye outlines; negative = shrink |
| Eye Distance | `reshape_eye_distance` | −50..+50 | 0 | `reshape.eye_distance` | Translate both eyes outward (positive) or inward (negative) along the inter-eye axis |
| Nose Width | `reshape_nose_width` | −50..+50 | 0 | `reshape.nose_width` | Radial scale of the alae; negative = narrow |
| Nose Length | `reshape_nose_length` | −50..+50 | 0 | `reshape.nose_length` | Translate nose tip up (positive) / down (negative) |
| Jaw Width | `reshape_jaw_width` | −50..+50 | 0 | `reshape.jaw_width` | Inward (positive) / outward (negative) compression at the jaw angles, extends `slimming`'s jaw component independently |
| Chin Length | `reshape_chin_length` | −50..+50 | 0 | `reshape.chin_length` | Translate chin point 152 up (positive) / down (negative) |
| Mouth Size | `reshape_mouth_size` | −50..+50 | 0 | `reshape.mouth_size` | Radial scale of the outer lip contour |
| Smile | `reshape_smile` | −20..+30 | 0 | `reshape.smile` | Lift mouth corners (61/291) up + slightly out; negative = subtle frown |
| Forehead | `reshape_forehead` | −30..+30 | 0 | `reshape.forehead` | Translate the hairline band up (positive, more forehead) / down (negative) |

**Why these nine:** they match the Photoshop Face-Aware Liquify panel
one-for-one (Eye, Nose, Mouth, Face Shape sections). `cheek_lift` from the
task brief is folded into `slimming` + `jaw_width` (a true isolated cheek
*lift* requires vertical translation of the cheek fat pad, which reads as
uncanny on still photos — deferred until evidence it's wanted).

**Conversion code:** `gui_direct` (same as `slimming`, `params.py:714`) —
the GUI slider integer is the engine value directly. No 0–1 ↔ 0–100
conversion, no recipe-pct path. Range is enforced by `min_val` / `max_val`
on the spec, mirroring how `slimming` declares `min_val=0, max_val=100`
(`params.py:715-716`).

---

## 4. Landmark Regions

All indices are MediaPipe Face Mesh 468-point indices, already used
throughout `retouch/parsing.py` (lines 31-106) and `retouch/geometry.py`
(lines 60-85). The indices below are **verified** against
`parsing.py`'s constant tables — no new topology is invented.

### 4.1 Face-width anchor (every slider scales off this)

```
face_width = |landmarks[454].x − landmarks[234].x| · W     # inter-temple
```
Same definition as `geometry.py:50-53`. `face_width < 10` ⇒ skip the face
(legacy guard, `geometry.py:53-54`).

### 4.2 Per-feature anchor indices

| Feature | Anchor landmarks | Source in codebase |
|---|---|---|
| **Eye centers** (L / R) | `33` (left outer corner), `133` (left inner corner); `263` (right outer), `362` (right inner). Eye center = midpoint of the four `LEFT_EYE` / `RIGHT_EYE` rings. | `parsing.py:37-44` (`LEFT_EYE`, `RIGHT_EYE`) |
| **Eye ring** (for scale warp) | Full `LEFT_EYE` (16 pts) + `LEFT_IRIS` (5 pts, 468-472); mirrored right. Iris center used as the scale origin. | `parsing.py:37-44, 58-59` |
| **Inter-eye axis** (eye_distance) | `133` ↔ `362` (inner canthi). Displacement direction = normalized `(362 − 133)`. | `parsing.py:39, 43` |
| **Nose alae** (nose_width) | `48` (left ala), `278` (right ala). Scale origin = nose bridge `168` (or `6`). | `parsing.py:65` (`NOSE`), `parsing.py:578` (`nose_bridge = [168,6,197,195]`) |
| **Nose tip** (nose_length) | `4` (tip) + `1` (subnasale). Direction = vertical. | `parsing.py:63` (`NOSE`) |
| **Jaw angles** (jaw_width, slimming) | `234` (right jaw angle), `454` (left jaw angle) — the existing slimming anchors. Cheek anchors `117` / `346` reused for the cheek component of `slimming`. | `geometry.py:60-79` |
| **Chin tip** (chin_length) | `152` (bottom chin) — the existing chin anchor. | `geometry.py:82-85`, `parsing.py:34` (`FACE_OVAL`) |
| **Mouth corners** (mouth_size, smile) | `61` (left corner), `291` (right corner). Mouth center = midpoint `13`/`14` (upper/lower lip center). | `parsing.py:49-52` (`LIPS_OUTER`), `parsing.py:54-56` (`LIPS_INNER`) |
| **Hairline band** (forehead) | `FOREHEAD_TOP` (`parsing.py:78`) = `[10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]`. Displace vertically; radius scales with `face_width × 0.4`. | `parsing.py:78-83` |
| **Jawline contour** (already parsed) | `regions.jawline_contour` mask exists (`parsing.py:586-590`) from indices `[234,127,93,132,58,172,136,150]` + `[454,323,361,288,397,365,379,378]`. Used for *feathering the warp boundary*, not as control points. | `parsing.py:586-590` |

### 4.3 Region masks for boundary feathering

Each warp's `cv2.remap` is global, but its *visible effect* must be confined
to the face (no background bowing). Two mechanisms, both already in the
codebase:

1. **Radius discipline:** `R ≤ face_width × 0.5` for every warp ⇒ the
   `(1 − d²/R²)²` falloff is exactly 0 at `R`, so background beyond `R` from
   every control point is untouched. This is the primary defense and is
   enforced in `_apply_warps` as a hard clamp on `R`.
2. **Mask-gated composite:** after `cv2.remap`, blend
   `result = original·(1−M) + warped·M` where `M = regions.face_oval`
   (BiSeNet face oval, `parsing.py:231-235`) feathered by
   `feather_mask(M, radius=ied×0.08)` (the existing feather constant,
   `parsing.py:178`). This catches any sub-pixel bleed at the oval edge.

Mechanism #2 is **new to `geometry.py`** (the current `slimming` path does
not mask-gate — it relies purely on radius discipline, which works because
`slimming`'s radii are small). For F5's larger-magnitude sliders (eye
enlarge, jaw widen-negative), the mask gate is added as defense-in-depth.
It costs one `cv2.addWeighted`-style blend per face, negligible vs. the
`cv2.remap` itself.

---

## 5. Integration

### 5.1 Refactor `FaceReshaper` (extract `_apply_warps`)

The current `reshape()` method (`geometry.py:18-133`) does three things in
one body: builds the slimming warp list, accumulates them into `map_x/map_y`,
and calls `cv2.remap`. F5 splits the first concern out:

```python
class FaceReshaper:
    def reshape(self, img, faces, ctx) -> np.ndarray:
        """New entry point — takes the full ProcessingContext (or a thin
        reshape-only view of it) so it can read all 9 sliders."""
        warps = []
        for face in faces:
            fw = self._face_width(face, img.shape)
            if fw < 10:
                continue
            warps += self._slimming_warps(face, fw, ctx.slimming)         # existing set
            warps += self._eye_size_warps(face, fw, ctx.reshape_eye_size)
            warps += self._eye_distance_warps(face, fw, ctx.reshape_eye_distance)
            warps += self._nose_width_warps(face, fw, ctx.reshape_nose_width)
            warps += self._nose_length_warps(face, fw, ctx.reshape_nose_length)
            warps += self._jaw_width_warps(face, fw, ctx.reshape_jaw_width)
            warps += self._chin_length_warps(face, fw, ctx.reshape_chin_length)
            warps += self._mouth_size_warps(face, fw, ctx.reshape_mouth_size)
            warps += self._smile_warps(face, fw, ctx.reshape_smile)
            warps += self._forehead_warps(face, fw, ctx.reshape_forehead)
        return self._apply_warps(img, warps, face_oval_mask=...)

    def _apply_warps(self, img, warps, face_oval_mask=None) -> np.ndarray:
        """The extracted accumulation + remap + optional mask gate.
        This is the current lines 40-133 of geometry.py, verbatim in spirit."""
        ...
```

Each `_<feature>_warps()` returns a `List[(C, T, R)]` tuple list in the
existing format. The accumulation loop (`geometry.py:88-121`) is unchanged —
it already handles an arbitrary-length `warps` list.

### 5.2 Backward compatibility

- `slimming` stays a top-level ParamSpec (`params.py:709`) with `recipe_key="slimming"`.
  Its warp set is the existing jaw/cheek/chin definitions, factored into
  `_slimming_warps()`. **Behavior at `slimming=N, all_reshape_*=0` is byte-identical
  to today** — this is the golden-output assertion (§7).
- The 9 new `reshape_*` ParamSpecs use `recipe_key="reshape.<feature>"`. Recipes
  that don't include a `"reshape"` block get all zeros ⇒ no-op.
- `ProcessingContext` gains 9 new `float` fields (default 0.0), mirroring how
  `slimming` is declared (`engine.py:228`). `build_context()` needs no special-case
  — the data-driven `_resolve_recipe_value` path handles `gui_direct` specs
  generically (`params.py:486-491`).
- The `process()` signature gains 9 `Optional[float] = None` kwargs, threaded
  into the `overrides` dict (`engine.py:804-911` pattern). Each gets added to
  the GUI defaults dict (`gui.py:104`, `gui.py:492`) and the reset-button list
  (`gui.py:1637-1640`).

### 5.3 `_stage_reshape` (the engine call site)

`engine.py:1820-1823` currently:

```python
def _stage_reshape(self, img, faces, ctx):
    if ctx.slimming > 0:
        return self._reshaper.reshape(img, faces, ctx.slimming)
    return img.copy()
```

Becomes:

```python
def _stage_reshape(self, img, faces, ctx):
    if self._any_reshape_active(ctx):
        return self._reshaper.reshape(img, faces, ctx)
    return img.copy()
```

where `_any_reshape_active` checks `slimming > 0` **or** any
`reshape_* != 0`. The `face_oval` mask needed by `_apply_warps`'s mask gate
(§4.3) is **not** available at this pipeline stage — `_stage_reshape` runs
*before* `_stage_per_face` (which is where BiSeNet parsing happens,
`engine.py:1905`). Two options:

- **(A) Move reshape after parsing.** Reorder so reshape runs on the
  per-face canvas with `regions.face_oval` available. Risk: changes the
  pipeline order documented in `engine.py:1-19` and the `docs/PIPELINE.md`
  contract. Also means reshaping happens at native-face-crop resolution
  (good for quality) but the warp must be applied per-crop and the
  displacement maps stitched back — non-trivial.
- **(B) Lightweight landmark-only oval inside `_apply_warps`.** Build a
  coarse face oval from `FACE_OVAL` indices (`parsing.py:31-35`) using
  `create_polygon_mask` + `feather_mask` (the same helpers `_landmark_fallback_only`
  uses, `parsing.py:525`). No BiSeNet dependency, no pipeline reorder. The
  mask is coarser than BiSeNet but sufficient for boundary bleed control
  (the radius discipline already does the heavy lifting).

**Decision: (B).** Reshape stays where it is (stage 0, pre-per-face),
landmark-only oval is computed on-demand inside `_apply_warps` when any
warp's radius exceeds `face_width × 0.4`. This keeps the pipeline contract
intact and avoids per-crop warp stitching. Option (A) is revisited if the
mask gate proves insufficient in visual QA — but radius discipline + a
landmark oval is what every open-source liquify implementation uses, so
it's the right default.

### 5.4 Multi-face

The existing `reshape()` already loops over `faces` (`geometry.py:47`).
The refactor preserves this — each face contributes its own warp set to the
*global* `map_x/map_y`, then one `cv2.remap` applies all. This is correct
because warps from different faces have disjoint support (radii don't
overlap) so summing is safe. No change needed.

### 5.5 Thread-safety / multiprocess

`_stage_reshape` runs in the main thread, before the `ThreadPoolExecutor`
per-face loop (`engine.py:1836`). The refactor introduces no shared mutable
state — `_apply_warps` builds fresh `map_x/map_y` per call. No
`FaceContext`-across-process-boundary concern (the `AGENTS.md` multiprocess
pickling rule, `retouch/perf_optimizations.py:246`) because reshape is
pre-pool. `[VERIFIED]` by reading `engine.py:1320` and `engine.py:1488` —
both call sites are in `_process_native_faces` / `_run_detection_and_faces`,
both serial.

---

## 6. Anti-Artifact Strategy

The "plastic surgery" look comes from four failure modes. Each has an
explicit guard.

### 6.1 Over-displacement (the "alien" look)

- **Cause:** slider at 100 moves a feature too far.
- **Guard:** per-slider `max_displace(fw)` cap (§2.4), enforced as a
  clamp on `|T − C|` inside `_apply_warps`, not as a slider limit. Even
  recipe JSON with `reshape.eye_size: 999` resolves to the capped
  displacement.
- **Test:** identity-guard test — at slider=100 on a synthetic face,
  assert `max(|T − C|) ≤ fw × 0.12` for every emitted warp
  (`PLAN_TIER2_MANUAL_TOOLS.md:54`).

### 6.2 Background bowing (the classic liquify tell)

- **Cause:** warp radius extends into the background, dragging straight
  lines into curves.
- **Guard:** `R ≤ fw × 0.5` hard clamp (§4.3 mechanism #1) + landmark-oval
  mask gate (§4.3 mechanism #2, option B).
- **Test:** background-line preservation gate
  (`PLAN_TIER2_MANUAL_TOOLS.md:53`) — synthetic image with horizontal
  lines behind the face, assert lines stay straight outside
  `fw × 0.6` of any control point. This becomes a permanent pytest
  assertion in `tests/test_geometry.py`.

### 6.3 Asymmetry (the "stroke face" look)

- **Cause:** left and right warps drift independently.
- **Guard:** every bilateral warp is emitted as a *mirrored pair* in the
  same `_<feature>_warps()` call. The mirror axis is the vertical line
  through `landmarks[168]` (nose bridge center, `parsing.py:578`), not
  the image center — so a slightly turned head still gets a symmetric
  *facial* warp.
- **Test:** landmark-symmetry test
  (`PLAN_TIER2_MANUAL_TOOLS.md:55`) — for every bilateral slider, assert
  the right-side warp list is the left-side list reflected across the
  face's vertical axis.

### 6.4 Hard edges (the "cutout" look)

- **Cause:** warp boundary visible as a ring.
- **Guard:** the existing `(1 − d²/R²)²` falloff is C¹-continuous at `R`
  (derivative is 0 at the boundary), so there is no edge by construction.
  The mask gate (§4.3) uses `feather_mask` (`parsing.py:178` constant
  `ied × 0.08`), the same feather used everywhere else in the parser —
  no new feather value to tune.
- **Test:** No-Edge-Tearing gate from `docs/VISUAL_QA.md:13` — diff the
  warp displacement field for discontinuities > 2px. Already a required
  gate for `geometry.py` (`docs/VISUAL_QA.md:30`).

### 6.5 Interaction with skin smoothing

A subtle one: reshaping moves pores. If `_stage_reshape` runs *before*
frequency separation (it does — stage 0 vs stage 4, `engine.py:3-7`), the
frequency bands are extracted from the *warped* image, so pores are
naturally carried along. This is correct. If reshape ran *after* smoothing,
the smooth layer would be warped but the re-injected texture band would
not, producing a misaligned-pore artifact. **Current order is right; do not
change it.** `[VERIFIED]` by reading the pipeline order comment in
`engine.py:1-19`.

---

## 7. Test Strategy

### 7.1 Unit tests (`tests/test_geometry.py`, extended)

The existing 7 tests (`tests/test_geometry.py:33-70`) cover the slimming
path. F5 adds, in the same file:

| Test | Asserts |
|---|---|
| `test_all_sliders_zero_passthrough` | `reshape(img, face, ctx_all_zero)` is byte-identical to `img` (`PLAN_TIER2_MANUAL_TOOLS.md:55`) |
| `test_no_face_no_op` | (already exists for slimming; extend to the new entry point) |
| `test_eye_size_positive_enlarges` | On a synthetic face with a marked eye ring, `reshape_eye_size=+30` increases the ring's enclosing-circle radius |
| `test_eye_size_negative_shrinks` | Mirror of above |
| `test_bilateral_symmetry_eye_size` | Left and right warp lists are mirror images across the face axis (`§6.3`) |
| `test_bilateral_symmetry_jaw_width` | Same for jaw |
| `test_displacement_cap_eye_size` | At `reshape_eye_size=100`, `max(|T−C|) ≤ fw × 0.12` for every eye warp (`§6.1`) |
| `test_displacement_cap_all_sliders` | Parameterized over all 9 sliders at ±100, assert cap holds |
| `test_radius_clamp` | Every emitted `R ≤ fw × 0.5` (`§6.2`) |
| `test_background_line_preservation` | Synthetic image: horizontal lines + a face. After warp at slider=50, lines outside `fw × 0.6` of any control point deviate < 0.5px from straight (`PLAN_TIER2_MANUAL_TOOLS.md:53`) |
| `test_multi_face_independent` | Two faces with different slider values via per-face ctx (future-proofing; today ctx is global) |
| `test_slimming_backward_compat` | `slimming=50, all_reshape_*=0` produces byte-identical output to the pre-F5 `FaceReshaper.reshape(img, faces, 50)` — golden-output regression |

The last test requires a frozen baseline image committed under
`tests/golden/reshape_slimming50.png`, generated from the current code
*before* the refactor and re-asserted after. This is the standard
golden-output pattern referenced in `MASTER_PLAN.md:136`.

### 7.2 ParamSpec / recipe validation

Per the standing ground rule (`MASTER_PLAN.md:134`): every new recipe key
gets the dead-key guard. Add the 9 `reshape.*` keys to
`tests/test_recipe_validation.py`'s `_VALID_KEYS` set. Any recipe that
references a `reshape.<feature>` not in the ParamSpec list fails validation.

### 7.3 Visual QA gates (mandatory — Visual-Critical)

`geometry.py` is in the Visual-Critical list (`AGENTS.md`). Per
`docs/VISUAL_QA.md:30`, the required gates are **No Edge Tearing** and
**Natural Output**. F5 adds a third from the table: **No Halo Artifacts**
(the mask gate could introduce a luminance ring at the oval edge if the
feather is too tight).

Run on the reference test corpus (`docs/reference_targets/README.md`):

| Gate | How | PASS criterion |
|---|---|---|
| No Edge Tearing | Diff warp displacement field; inspect hairline + jaw at 100% | No pixel displacement discontinuity > 2px (`docs/VISUAL_QA.md:13`) |
| Natural Output | Full-image 100% review on 3 portraits (front, ¾, profile) | No reviewer identifies "plastic surgery" / artificial look (`docs/VISUAL_QA.md:18`) |
| No Halo | Inspect face oval boundary at 100% after mask-gated blend | No luminance ring (`docs/VISUAL_QA.md:12`) |
| Background-line preservation (F5-specific) | Synthetic line grid behind face | Lines straight outside `fw × 0.6` (`PLAN_TIER2_MANUAL_TOOLS.md:53`) |

Per `AGENTS.md`: after 2 consecutive failures of the same gate → STUCK,
escalate. "Tests pass" without visual confirmation is a protocol violation.

### 7.4 Performance budget

Per `AGENTS.md`: runtime must not increase >10% on 1080p/4K, memory peak
>15%. The refactor adds 9 warp-set builders (each O(1) landmark lookups)
and potentially 9× more `(C,T,R)` tuples in the accumulation loop. The
loop is already O(warps × R²) per bounding box (`geometry.py:88-121`); the
remap cost dominates. Benchmark with `scripts/bench/benchmark.py` before
and after; log in the F5 receipt row of `MASTER_PLAN.md`.

---

## 8. Files Touched

| File | Change | Size |
|---|---|---|
| `retouch/geometry.py` | Refactor `reshape()` → `_apply_warps` + 9 `_<feature>_warps` builders + mask gate | MEDIUM (~250 lines added) |
| `retouch/params.py` | 9 new `ParamSpec` entries under `reshape.*` recipe namespace | SMALL |
| `retouch/engine.py` | `_stage_reshape` signature change (pass `ctx` not `ctx.slimming`); 9 new `process()` kwargs + override dict entries; 9 new `ProcessingContext` fields | SMALL-MEDIUM |
| `retouch/recipes.py` | (Optional) add `"reshape": {...}` blocks to 2-3 flagship recipes that want subtle eye enlarge | SMALL |
| `gui.py` | 9 new sliders in the "🧬 Face Reshaping" accordion (`gui.py:1413`); extend `reset_face_reshaping` (`gui.py:529`) and the defaults dict (`gui.py:104, 492`) | MEDIUM |
| `tests/test_geometry.py` | ~12 new tests (§7.1) | MEDIUM |
| `tests/test_recipe_validation.py` | Add 9 `reshape.*` keys to `_VALID_KEYS` | SMALL |
| `tests/golden/reshape_slimming50.png` | New baseline image for backward-compat regression | SMALL (binary) |
| `MASTER_PLAN.md` | Annotate row 19 with `✅ DONE <date>` + receipt (per `AGENTS.md` workflow rule) | SMALL |

**Effort:** ~2 weeks (`MASTER_PLAN.md:85`), matching the tier-2 plan estimate
(`PLAN_TIER2_MANUAL_TOOLS.md:58`).

---

## 9. Out of Scope / Deferred

- **Brush-based liquify** (free-form mesh drag). F5 is slider-only by design
  (`PLAN_TIER2_MANUAL_TOOLS.md:41`). Brush liquify would need MLS + a mesh UI;
  not on the roadmap.
- **Per-face slider values.** Today `ProcessingContext` is global per
  `process()` call. Multi-face images get the same slider values on every
  face. Per-face override is a GUI/ctx-shape change deferred to F2 (sessions)
  or later.
- **Cheek lift as an isolated slider.** Folded into `slimming` + `jaw_width`
  (§3 footnote). Revisit if real-photo QA shows a need.
- **T3 body reshape** (`MASTER_PLAN.md:111`). Depends on "F5 warps" — F5's
  `_apply_warps` extraction (§5.1) is the dependency. T3 will add MediaPipe
  Pose control points and reuse `_apply_warps` directly.

---

## 10. Open Questions

1. **Smile slider ethical range.** `reshape_smile` at +30 is a noticeable
   grin. Should the GUI cap at +20 by default with an "advanced" toggle for
   +30? Defer to UX review during implementation.
2. **Forehead slider direction sign convention.** "More forehead" = hairline
   up = positive, but users may expect "lift forehead" = hairline down.
   Needs a user-facing label decision ("Forehead Height" vs "Hairline
   Position"). Resolved in GUI review, not in the math.
3. **Should `slimming`'s recipe key migrate to `reshape.slimming`?** Pro:
   namespace consistency. Con: breaks every existing recipe that has
   top-level `"slimming": N`. Decision: keep top-level `slimming` for
   backward compat; the `reshape.*` namespace is for the 9 new keys only.
