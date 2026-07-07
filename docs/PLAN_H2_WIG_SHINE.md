# PLAN_H2_WIG_SHINE — Wig Shine Shaping (Deglare + Anisotropic Angel Ring)

> **Master-plan row 18c** (`MASTER_PLAN.md:84`): *Wig shine shaping (deglare + anisotropic angel ring); optional depth D&B along flow — ~1.5 wk, depends on H0 ✅.*
> **Tier-H plan source:** `PLAN_TIERH_HAIR.md:32-37` (H2 section).
> **Type:** MEDIUM / **Visual-Critical** — touches `hairwork.py` (a pipeline stage) and reuses `skin.shine_removal` math at hair scale.
> **Status:** ✅ DONE 2026-07-07 — `retouch/hairwork.py` `deglare_wig()` + `add_angel_ring()` (flow-steered, float32/uint8); `retouch/params.py` `hair_deglare`/`hair_ring_position`/`hair_ring_tint` ParamSpecs; `retouch/engine.py` ProcessingContext fields + build_context kwargs; `retouch/perf_optimizations.py` hair stage wired (deglare→ring, legacy fallback). Tests: `tests/test_wig_shine.py` (16 cases) + existing hair suite (48) green.
> **Depends on:** H0 (strand flow field) ✅ shipped — `retouch/hairwork.py:42` `hair_flow()`.

---

## 1. Motivation

Cosplay wigs are made of synthetic high-temperature fiber (高温丝 / 卡丝). The
fiber is smoother and more reflective than real hair, so under venue or strobe
light it produces a **broad, plasticky specular** that reads as "fake wig" even
when the cut and color are good. CN pro retouch tutorials call this out as the
single biggest wig tell, and the standard manual fix is two-sided:

1. **Tame** the harsh specular (deglare) — but *without* killing all highlights,
   because a flat-matte wig looks dead.
2. **Re-add** a coherent highlight band called the **天使环 ("angel ring" /
   halo)** — a smooth anisotropic band running *perpendicular to strand flow*
   at the crown, tinted toward the hair's own chroma (never pure white). This
   is the highlight pattern real hair naturally produces (Kajiya-Kay), and its
   absence on a wig is the second-biggest tell after the glare itself.

The engine today ships exactly one hair-shine knob: `hair_enhance`
(`retouch/params.py:719`), implemented by `HairEnhancer.enhance()`
(`retouch/hair.py:40`). That op is **isotropic** — it lifts exposure/midtones/
highlights across the whole hair mask and boosts any L > mean+0.8σ region as a
specular blob (`hair.py:111-160`). On a wig this makes the problem *worse*: it
amplifies the plasticky glare rather than reshaping it into a flow-coherent
band. H2 replaces that behaviour with the two-sided pro workflow, gated to the
hair mask and steered by the H0 strand flow field.

### Design principles (in priority order, per `AGENTS.md`)

1. **Visual Fidelity > Correctness > Performance > Elegance.** A deglare that
   flattens the wig's dimensionality is reverted; an angel ring that reads as
   an isotropic blob is reverted.
2. **Anisotropic by construction.** Both sub-ops are modulated by the H0 flow
   field. No Gaussian-blob highlights anywhere — that is the existing
   `hair_enhance` failure mode H2 is replacing.
3. **Reuse tested math.** Deglare adapts `SkinProcessor.shine_removal()`
   (`retouch/skin.py:1200`); the angel ring reuses the Blinn-Phong half-vector
   geometry from `Relighter._shading_geometry()` (`retouch/relight.py:24`). No
   new shading kernel.
4. **Mask-disciplined.** Everything is gated by `hair_mask`; the angel ring is
   further gated by a coherence floor so it breaks at partings/occlusions and
   never fires on fuzzy/short wigs.

---

## 2. Sub-op A — Deglare (tame synthetic specular)

### 2.1 Why adapt S4's `shine_removal`, not write new

`SkinProcessor.shine_removal()` (`skin.py:1200-1364`) already solves the exact
problem at skin scale: detect bright low-chroma regions, reconstruct their
chroma from neighbours via guided filter, and compress L toward a local
non-shine median with an exponential soft rolloff capped at 70% reduction. The
detection, the chroma inpaint, and the soft-clip rolloff are all
resolution-independent and colorspace-agnostic — only the *scale* of the
operators (kernel radii, threshold offsets) are skin-tuned.

H2 reuses the algorithm and retunes the scale for hair. **No new math**, per
principle #3.

### 2.2 Detection — what counts as "wig glare"

Run in LAB (consistent with `shine_removal` and the `AGENTS.md` rule that skin/
hair tone ops use LAB). Within `hair_mask`:

```
chroma   = sqrt((a-128)² + (b-128)²)            # skin.py:1258 convention
L_gate   = smoothstep(median_L_hair + ΔL,  +15,  L)
C_gate   = 1 - smoothstep(0, chroma_thresh, chroma)
glare_m  = L_gate * C_gate * hair_mask
```

Tuning differences from S4 (skin):

| Parameter | S4 skin value | H2 hair value | Rationale |
|---|---|---|---|
| `ΔL` (shine threshold above region median L) | +20 | **+12** | Wig glare is broader but not as far above local L as oily skin shine; synthetic specular sits closer to the strand base luminance. |
| `chroma_thresh` | `0.4 × median_chroma_skin` | **`0.25 × median_chroma_hair`** | Wig glare is *more* desaturated relative to hair color than skin shine is to skin color — synthetic fiber specular is near-neutral. Tighter gate. |
| Feather kernel | `(11,11)` Gaussian | `(face_width/30 \| 1)` Gaussian | Scale with resolution per `AGENTS.md` (KERNEL_SCALE), not hardcoded. |

`eyes_mask` exclusion (S4 step 2, `skin.py:1293`) is **not** ported — eyes are
outside `hair_mask` already. Instead an **eyebrow/eyelash guard** is added
(see §6.3).

### 2.3 Flow-aware directional coherence

The key upgrade over S4: glare detection is *steered by the flow field* so it
follows strand structure rather than treating glare as isotropic blobs.

H0 returns `(orientation, coherence)` (`hairwork.py:42`), where `orientation`
is the **gradient** principal axis (strand direction = orientation + π/2). The
strand tangent vector field is:

```
t = (cos(orientation + π/2), sin(orientation + π/2))
```

Two uses:

1. **Anisotropic feathering of `glare_m`.** Instead of the isotropic
   `(11,11)` Gaussian feather S4 uses, blur `glare_m` with a kernel elongated
   along `t` (e.g. `cv2.getStructuringElement` along the tangent, or a
   separable two-pass Gaussian with `σ_along = 4σ_across`). This makes the
   deglare region a *band along the strand*, not a blob — matching how real
   specular highlights lie on hair.
2. **Coherence gate.** Multiply `glare_m` by `coherence` (from `hair_flow`).
   Low-coherence regions (fuzzy wisps, flyaways, parting gaps) are excluded
   from deglare — they're already diffuse, and compressing them would flatten
   legitimate matte texture. Threshold: `coherence > 0.25` (the same floor H1
   flyaway detection uses, per `PLAN_TIERH_HAIR.md:27`).

### 2.4 Reconstruction & L compression

Identical to S4 steps 4–5 (`skin.py:1304-1353`), retuned:

- **Chroma inpaint:** `guided_filter(a, radius=face_width/8, eps=100)` and same
  for `b` — radius scales with face width, not the hardcoded `30` S4 uses.
- **Local non-glare L target:** mask glare regions, Gaussian-blur the
  L-masked image at `σ = face_width/8` (S4 uses fixed `(31,31)`).
- **Soft compression:** `L_new = L - L_excess · compression_factor · glare_m`,
  with `compression_factor = 1 - exp(-1.5 · normalized_excess)`, capped at
  `0.70 · s` (same 70% over-removal guard as S4, `skin.py:1349`). This
  guarantees a **residual highlight survives** — the wig still reads as
  reflective, just not plasticky.

### 2.5 Mid-band dimensionality floor (QA gate, §7)

To prevent the deglare from flattening the wig's shading dimensionality, after
compression measure the mid-band energy (`_blotch_bandpass`, `skin.py:35`) of
the hair region before and after. If post-deglare mid-band energy drops below
**85%** of pre-deglare, scale back `compression_factor` proportionally. This is
the "deglare preserves dimensionality (mid-band energy floor)" gate promised in
`PLAN_TIERH_HAIR.md:37`.

---

## 3. Sub-op B — Anisotropic Angel Ring

### 3.1 What the ring is

A smooth, coherent highlight **band** laid along the crown, perpendicular to
strand flow, tinted toward the hair's own chroma. In Kajiya-Kay terms it is the
primary specular lobe: `I_spec ∝ (sin(θ_t, L) )^α` where `θ_t` is the angle
between the strand tangent and the light direction. The ring is *added* on top
of the deglared hair, not painted over it.

### 3.2 Band geometry — crown arc from head landmarks

The ring sits on the top-of-head arc. Two sources:

1. **Primary (landmark arc):** MediaPipe crown landmarks (e.g. `lm[10]` forehead
   top, `lm[151]` brow top, `lm[18]` chin) define a head ellipse; the upper
   third of that ellipse is the crown arc. Project the arc to image space and
   dilate by `face_width × 0.15` to form a soft crown band `crown_band_m`.
2. **Fallback (vertical slider):** if landmarks are unavailable (no face, or
   face below frame), the band is a horizontal stripe at vertical position
   `ring_position · H` (the `hair_ring_position` param, §5), feathered by
   `face_width × 0.10`. This matches the manual-tutorial "paint a band at the
   crown" workflow.

The band is **modulated by coherence** (H0): `crown_band_m *= coherence >
coherence_floor`. This makes the ring **break at partings and occlusions**
(`PLAN_TIERH_HAIR.md:37`) and **never fire on fuzzy/short wigs** (the coherence
floor).

### 3.3 Anisotropic profile — Kajiya-Kay modulation

For each pixel in `crown_band_m`, compute the strand tangent `t` (§2.3) and the
local light direction `L` (fixed: `azimuth=0°, elevation=70°` — a high frontal
key, the classic angel-ring light). The ring intensity is:

```
cos_θ = | t · L_perp |              # L_perp = L projected onto image plane
ring_profile = cos_θ ** α           # α ≈ 8 (Blinn-Phong-ish, reuse relight.py:22 α=32 baseline, softened)
ring_m = crown_band_m * ring_profile * coherence
```

`L_perp` is the in-image-plane component of the light vector — this is the
half-vector / tangent dot product from `Relighter._relight_v2`
(`relight.py:316-323`), repurposed from surface normals to strand tangents. The
`α` exponent is the same specular-roughness parameter `Relighter.__init__`
takes (`relight.py:16`); H2 instantiates with a softer `α=8` for a broader band
than the skin specular uses.

The result is a band that is **brightest where strands run perpendicular to the
light** and **dims to nothing where strands run parallel** — exactly the
anisotropic behaviour real hair exhibits and the manual "天使环" produces.

### 3.4 Tint — never pure white

The ring is added in **LAB L**, but the L lift is accompanied by a small chroma
pull toward the hair's median `(a, b)`:

```
L_ring  = L + ring_strength * ring_m * (255 - L) * 0.12     # additive, highlight-protected
a_ring  = a + (median_a_hair - a) * ring_m * 0.15
b_ring  = b + (median_b_hair - b) * ring_m * 0.15
```

This is the "tinted slightly toward hair chroma (not pure white)" requirement
from `PLAN_TIERH_HAIR.md:36`. `median_a_hair / median_b_hair` are computed
once over `hair_mask > 0.3` (same pattern as `shine_removal` at
`skin.py:1264-1265`).

Highlight protection follows the S4/relight convention (`relight.py:197`,
`skin.py:187`): `prot = clip(1 - (L - 220)/30, 0, 1)`, so the ring never
pushes already-bright pixels into clipping (passes the *No Highlight Clipping*
gate, `VISUAL_QA.md:15`).

### 3.5 What the ring is *not*

- Not an isotropic Gaussian blob at the crown (that is what `hair_enhance`
  already does wrong, `hair.py:137-160`).
- Not a full halo around the head — only the upper arc (crown), where the
  angel-ring light physically produces the highlight.
- Not applied below the `coherence_floor` — fuzzy wigs get no ring, by design.

---

## 4. Integration

### 4.1 Pipeline placement

```
H0 (flow field) ──► H1 (flyaways) ──► H2 (deglare + ring) ──► H3 (color unify) ──► H4 (depth D&B)
```
*(from `PLAN_TIERH_HAIR.md:47`)*

H2 runs **after H1** (so flyaway inpaint doesn't fight the ring) and **before
H3** (so H3's chroma pull sees the reshaped specular, not the original glare).
H4 depth D&B is the optional finisher on top.

### 4.2 Engine call site

The current hair stage is a single call (`perf_optimizations.py:555-560`):

```python
if ctx.hair_enhance > 0:
    canvas = _tr('hair.enhance', canvas)
    canvas = hair.enhance(canvas, roi_person_mask, regions.face_oval,
                          shifted_face.bbox, ctx.hair_enhance, regions.hair)
```

H2 **replaces** this call site with the new two-sided op. The `hair_enhance`
param (`params.py:719`, recipe key `hair.shine`) is reinterpreted as the
**angel-ring strength** (an upgrade, not a duplicate — `PLAN_TIERH_HAIR.md:36`
explicitly calls this out). Two new params (`hair_deglare`,
`hair_ring_position`) are added alongside it (§5).

### 4.3 Flow field caching

`hair_flow()` is computed once per pipeline run (it is also needed by H1 and
H4). H2 receives `(orientation, coherence)` from the engine context, it does
not recompute. Per `perf_optimizations.py:674-685`, the `HairEnhancer` instance
is built once per worker; the flow field is cached on the engine context
alongside `regions.hair`, `face_width`, etc.

### 4.4 float32 path

Both sub-ops must support the float32 path used by the E1 frequency pipeline
(`skin.py:1238-1241`, `hair.py:65`). The `is_float` branching pattern used
throughout `skin.py` (`bgr_f32_to_lab_f32` / `lab_f32_to_bgr_f32`) is followed
verbatim.

---

## 5. Params

Registered in `retouch/params.py` in the `_FACIAL_PARAMS` / hair block
alongside `hair_enhance` (`params.py:718-727`). All are `conversion="gui_direct"`
or `"recipe_pct"` and dead-key guarded (0 = no-op) per `params.py:41` convention.

| Param | CLI flag | Range | Default | Recipe key | Purpose |
|---|---|---|---|---|---|
| `hair_enhance` *(reinterpreted)* | `--hair-enhance` | 0–100 | 5 | `hair.shine` | **Angel-ring strength.** 0 = no ring. Existing slider, new behaviour. |
| `hair_deglare` *(new)* | `--hair-deglare` | 0–100 | 0 | `hair.deglare` | Synthetic-glare compression strength. 0 = no deglare. |
| `hair_ring_position` *(new)* | `--hair-ring-position` | 0–100 | 30 | `hair.ring_position` | Vertical position of the ring band (0=top of frame, 1=bottom of hair). Default 30 = upper crown. Ignored when landmarks supply the crown arc. |
| `hair_ring_tint` *(new, optional)* | `--hair-ring-tint` | 0–100 | 40 | `hair.ring_tint` | How strongly the ring is pulled toward hair chroma vs neutral. 0 = neutral white, 100 = full hair-color match. |

`hair_deglare` and `hair_enhance` (ring) are **independent** — a user can
deglare without adding a ring, or add a ring without deglaring. This matches
the two-sided pro workflow where the two steps are tuned separately.

---

## 6. File plan & guards

### 6.1 New code location

All H2 logic lives in **`retouch/hairwork.py`** (the H0 module — it already
holds `hair_flow` and `unify_hair_color`, so H2 is the natural next function
family there). Two new functions:

- `deglare_wig(img_bgr, hair_mask, orientation, coherence, strength, face_width) -> img_bgr`
- `add_angel_ring(img_bgr, hair_mask, orientation, coherence, landmarks_or_position, strength, tint, face_width) -> img_bgr`

The orchestrator (`perf_optimizations.py:555`) calls them in sequence when
their params are > 0. The legacy `HairEnhancer.enhance()` (`hair.py:40`) is
kept as a fallback for the non-flow-field path (e.g. if H0 failed), gated by a
feature flag in the recipe.

### 6.2 Precision invariants (per `AGENTS.md` Auto-CRITICAL)

- **float32 internal:** all LAB arithmetic in float32; the `is_float` branch
  preserves float32 in/out (matches `skin.py` / `hair.py`).
- **Colorspace:** explicit `bgr_f32_to_lab_f32` / `cv2.cvtColor(BGR2LAB)` at
  every boundary; never assume LAB.
- **Masks:** `hair_mask` normalized to float32 [0,1] before compositing
  (`hairwork.py:122-124` already does this; H2 reuses the pattern).
- **Kernel sizing:** all kernel radii scale with `face_width` (or `KERNEL_SCALE`),
  never hardcoded — the S4 hardcoded `(31,31)` and `radius=30` are **not**
  copied verbatim, they are retuned to `face_width/N` (§2.4).
- **No `GaussianBlur` on hair pixels for smoothing:** the deglare compresses L
  toward a *target*, it does not blur the hair itself. The only Gaussian
  applied to hair pixels is the `L_target_smooth` (a target field, not the
  output) — same distinction S4 maintains.

### 6.3 Region guards

- `hair_mask` gates everything (never touch skin/background).
- **Eyebrow/eyelash exclusion:** deglare subtracts a dilated
  `regions.left_eyebrow ∪ regions.right_eyebrow` from `glare_m` (mirrors the
  `dodge_burn` eyebrow exclusion at `skin.py:378-386`).
- **Coherence floor** (`> 0.25`) gates the angel ring and the anisotropic
  feather — fuzzy/short wigs get neither.
- **Highlight protection** (`prot = clip(1-(L-220)/30, 0, 1)`) on the ring's
  additive L lift, so no clipping (passes *No Highlight Clipping* gate).

---

## 7. Test Strategy — Visual QA Gates

Per `AGENTS.md`, H2 touches `hairwork.py` (a pipeline stage) → **Visual-Critical**.
Visual QA gates from `docs/VISUAL_QA.md` cannot be bypassed.

### 7.1 Gate scope

Applying the `VISUAL_QA.md:26` module→gates table for `hairwork.py` (treated as
a new pipeline stage; closest existing analog is `skin.py`):

| Gate | Required? | H2-specific criterion |
|---|---|---|
| **Texture Preservation** | YES | Pore/strand high-freq band SSIM on a hair patch ≥ 0.92 (deglare must not blur strands). |
| **No Halo Artifacts** | YES | No luminance ring at the hair/skin or hair/background boundary. The angel ring must be *inside* `hair_mask` and feathered to zero at the mask edge — no spillover. |
| **No Highlight Clipping** | YES | Ring's additive L lift does not push > 0.5% of hair pixels to 255. Verified by `prot` gate (§3.4). |
| **No Shadow Crushing** | YES | Deglare's L compression does not push > 0.5% of hair pixels to 0. |
| **No Color Drift** | YES | Neutral background/garment ΔE < 2.0 (everything is `hair_mask`-gated, so this should be trivially PASS — confirms mask discipline). |
| **Natural Output** | YES | Full-image 100% review: ring reads as a coherent band along flow, not an isotropic blob; wig still looks reflective, not matte. |

### 7.2 H2-specific gates (from `PLAN_TIERH_HAIR.md:37`)

These are the gates the tier-H plan explicitly mandates; they are checked in
addition to the eight standard gates:

1. **Ring coherence break:** On a test image with a visible parting, the ring
   must visibly break/gap at the parting (driven by the coherence gate). A
   continuous ring across a parting = FAIL.
2. **Coherence floor (fuzzy wig):** On a test image of a short/fuzzy wig
   (coherence median < 0.25), the ring must not appear at all. Any ring = FAIL.
3. **Deglare dimensionality floor:** Mid-band energy (`_blotch_bandpass`,
   `skin.py:35`) of the hair region post-deglare ≥ 85% of pre-deglare (the
   §2.5 floor). Below 85% = FAIL.
4. **Anisotropy (ring is a band, not a blob):** Measure the ring's spatial
   extent along vs across the dominant strand direction. The along-strand
   extent must be ≥ 2× the across-strand extent. Equal extents = isotropic
   blob = FAIL.

### 7.3 Test corpus

- **Reference set:** the DSCF cosplay corpus (silver/white wigs with warm
  venue-light casts — the `PLAN_TIERH_HAIR.md:40` H3 case), plus a
  dark-haired wig under strobe, plus a fuzzy/short wig for the coherence-floor
  gate.
- **A/B:** before/after at 100% on the crown band and on a glare hotspot.
- **Comparison:** side-by-side with a manual angel-ring edit on the same image
  (the `PLAN_TIERH_HAIR.md:54` "ring coherence visual A/B vs manual
  angel-ring edit" gate).

### 7.4 Unit tests (`tests/`)

- `deglare_wig` on a synthetic stripe image (vertical strands + a bright
  horizontal glare band): glare L compressed ≥ 40%, strand high-freq SSIM ≥
  0.95.
- `add_angel_ring` on a synthetic flow field (uniform vertical strands): ring
  intensity peaks where `cos(t · L_perp)` peaks, dims to zero where strands
  align with light — verify the profile shape, not just magnitude.
- Coherence-floor test: zero-coherence input → no ring, no deglare.
- Mask-discipline test: deglare/ring with `hair_mask` as a single isolated blob
  → output outside the blob is byte-identical to input.

---

## 8. Out of scope / parked

- **H4 depth D&B along flow** (`MASTER_PLAN.md:84`, `PLAN_TIERH_HAIR.md:42`)
  is the "optional" rider on row 18c. It is a separate stage that runs *after*
  H2 and H3; this doc does not design it. It will reuse the same `(orientation,
  coherence)` field and the `_blotch_bandpass` low-band decomposition.
- **Neural ring fitting** (learn the ring position/profile from a corpus of
  pro-retouched wigs) — parked under A4 (`MASTER_PLAN.md:112`), evidence-gated.
- **Multi-lobe Marschner ring** (primary + secondary specular lobes) — the
  single-lobe Kajiya-Kay ring is the manual-tutorial standard; Marschner is
  over-engineering for a retouch op, revisit only if the single-lobe ring
  fails the anisotropy gate on the corpus.

---

## 9. Verification checklist (pre-ship, per `AGENTS.md`)

1. `python3 -m py_compile retouch/hairwork.py`
2. `python3 -m pytest tests/ -v` (incl. new H2 unit tests)
3. `python3 scripts/bench/benchmark.py` — runtime must not increase >10% on
   1080p/4K; the flow field is already computed by H0 so H2's marginal cost is
   the deglare guided-filter + ring compositing, both O(N).
4. Visual QA gates (§7) on the reference corpus — all standard gates PASS, all
   four H2-specific gates PASS, A/B vs manual angel-ring edit.
5. Annotate `MASTER_PLAN.md:84` row 18c with `✅ DONE <date>` + receipt per
   `AGENTS.md` "Mark work done in the plan docs".
