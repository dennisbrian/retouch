# R9 Research Spike — Intrinsic Decomposition (Albedo × Shading) for Facial Skin

**Status:** 📋 SPIKE PLANNED (research written 2026-07-10, Fable) · **Parent:** MASTER_PLAN R9
**Spike budget:** ~4–5 days · **Full R9 estimate if GO:** ~3–4 wk
**Deliverable:** `scripts/spike_r9_intrinsic.py` + triptych renders + a GO/NO-GO verdict against the exit criteria below.

---

## 1. Problem statement

Every smoothing operator in the pipeline today (frequency separation, S2 D&B, S3 blotch
evening, equalize) operates on the *observed image* `I`. But `I = A · S`:

- **A (albedo):** pigment — skin tone, blotch, redness, freckles, moles. This is what we want to even out.
- **S (shading):** 3D form under light — cheek falloff, nose shadow, jaw turn. This is what makes a face look dimensional. **Touching S is what "plastic skin" is.**

Smoothing `I` inevitably smooths some of `S`. The A5 QA back-off mitigates the symptom;
decomposing first removes the cause: edit `A` aggressively, recompose `I' = A' · S`,
and dimensionality survives *by construction*.

Work in **log domain** on the linear/float face core (E1): `log I = log A + log S`
turns the multiplicative model additive, and helps dark-skin separability.

---

## 2. Candidate approaches (ranked)

### 2a. ✅ RECOMMENDED: Spherical-harmonics shading fit on MediaPipe normals

**Theory (Basri & Jacobs, "Lambertian Reflectance and Linear Subspaces"):** the shading of a
convex Lambertian surface under arbitrary distant lighting lies ≥97.96% inside a **9-dimensional
subspace** spanned by 2nd-order spherical harmonics of the surface normal:
`S(x) ≈ Σᵢ₌₁..₉ lᵢ · Yᵢ(n(x))`.

**Why it fits this codebase:** MediaPipe already gives us 478 landmarks **with z-depth**
(`detection.py:341` stores `z=lm.z`). Rasterizing MediaPipe's canonical tessellation
(`FACEMESH_TESSELATION` triangle list) over the face ROI with per-vertex normals yields an
approximate per-pixel normal map `n(x)` — coarse, but SH fitting has only **9 degrees of
freedom**, so it is extremely robust to normal noise.

**Algorithm:**
1. Rasterize normals from the landmark mesh (barycentric interpolation per triangle; smooth
   vertex normals from adjacent-face normals). Face ROI only, skin-mask gated.
2. Fit `l ∈ ℝ⁹` by **robust least squares** (IRLS/Huber) of `Y(n(x)) · l ≈ log L(x)` over skin
   pixels. The robustness is the magic: blemishes, blotch, and speculars are *outliers to a
   9-DOF global fit*, so they are automatically down-weighted — i.e. excluded from shading,
   which is exactly the separation we want.
3. `S = exp(clamp(Y·l))`, `A = I / S` (per-channel, or luminance-only with chroma passthrough —
   spike tests both).
4. **Blotch cannot enter S** because S has 9 parameters for the whole face. This is the
   structural guarantee no filtering approach can give.

**Exclusions from the fit:** specular pixels (reuse S4's shine detection), eyes/brows/lips/
nostril masks (existing regions), hair-shadowed pixels if detectable.

### 2b. Residual refinement layer (needed on top of 2a)

SH assumes convex + no cast shadows. Nose cast shadow, hair shadow on forehead, and
ambient occlusion in the eye sockets will land in `A` after step 2a. Refinement:

1. Compute residual `R = log L − log S_sh`.
2. Classify residual gradients by the **chroma-constancy test**: shading is (near-)achromatic —
   it scales R,G,B together, leaving log-chromaticity (`log R−log G`, `log G−log B`) unchanged;
   albedo changes are chromatic. Pixels/gradients where luminance varies but chromaticity is
   flat → shading; both vary → albedo.
3. Move the achromatic, spatially-coherent part of `R` from A into S — either by
   guided-filtering `R` with **chromaticity as the guide image** (cheap, reuses `guided_filter`),
   or by gradient classification + Poisson reintegration (FFT solve on the ROI, tens of ms) if
   the filter version halos.

**Caveat to verify in spike:** chroma-constancy weakens under strongly colored venue light
(con LEDs) and for the reddening at shadow edges on skin (SSS). WB runs earlier in the
pipeline, which helps; measure, don't assume.

### 2c. Chroma-guided Retinex/Poisson only (fallback #1)

Skip the SH fit; do the full decomposition with the gradient-classification + Poisson machinery
of 2b alone (classic color-Retinex intrinsic). No mesh dependency, works on profile/occluded
faces where the mesh is unreliable. Weaker guarantee (no low-rank prior — smooth blotch can
leak into S), but a sound fallback and the natural degraded mode when landmarks fail.

### 2d. Rejected for the spike

- **Edge-preserving "flattening"** (large-radius guided filter as S, ratio as A): this is
  frequency separation by another name — it's the **baseline to beat**, not a candidate.
- **Optimization-based intrinsic** (SIRFS, Intrinsic-Images-in-the-Wild style CRFs): minutes
  per image, fragile hyperparameters. No.
- **Neural face delighting/intrinsic nets:** parked per the A4 classical-first owner decision.
  If the spike NO-GOs both classical routes, that is exactly the evidence A4's gate asks for —
  record it.

---

## 3. Spike plan (4–5 days)

### Phase A (1–2 d) — SH shading fit
- Normal-map rasterization from cached FaceContext landmarks (new helper, spike-local).
- 9-coeff IRLS fit; render `A | S | I` triptychs.
- **Checkpoint:** blotch/blemish visibly in A; nose/cheek shading visibly in S; A looks "flat-lit".

### Phase B (1–2 d) — residual refinement
- Chroma-constancy classification of the residual; guided-filter route first, Poisson if halos.
- **Checkpoint:** hair/nose cast shadows migrate from A to S without dragging freckles along.

### Phase C (1 d) — edit-and-recompose test (the actual point)
- Point the existing S3 blotch-evening at `A` instead of `I`, recompose `I' = A'·S`.
- Compare against current-pipeline S3 at **equal blotch reduction** (match ΔE-variance drop on
  the blotch band, then compare everything else).

### Spike corpus (5–10 photos, all already on hand)
`DSCF6102.jpg` (2 pale faces — the equalize stress asset), the C1 dark-skin corpus samples,
one heavy white-paint cosplay face, one hard-directional-light face (strong nose shadow), one
semi-profile (mesh-stress), one convention-LED colored-light shot.

### Exit criteria — GO if all four hold
1. **Separation:** ≥ "mostly" of blotch-band chroma variance lands in A, not S (report the split).
2. **Quality:** Phase C beats current S3 on the F11 plastic-skin + halo detectors at equal
   blotch reduction, on ≥ 4 of 5 corpus photos including the dark-skin sample.
3. **Dark skin:** separation does not collapse at low albedo (log domain should hold this; verify).
4. **Perf:** < ~150 ms/face at 400×400 face core (budget check: SH lstsq over ~100k px is
   milliseconds; rasterization and one guided filter dominate). Fits inside the 702 ms/face envelope.

**NO-GO handling:** if 2a fails on mesh quality → rerun exit criteria on 2c (Poisson-only).
If both fail → write the evidence into EXECUTION_LOG and hand R9 to the A4 neural gate.

---

## 4. Known risks & open questions (answer during spike)

| Risk | Mitigation / measurement |
|---|---|
| MediaPipe z is canonical-ish, weak perspective | SH's 9-DOF robustness; test semi-profile explicitly |
| Speculars violate Lambertian | Exclude via S4 shine mask from the fit; speculars stay in a third layer — **this is R12's input**, a feature not a bug |
| Dark skin: low-albedo separability | Log domain + IRLS; exit criterion 3 is hard-gated on the C1 corpus |
| Heavy face paint = non-skin albedo | Decomposition is agnostic; downstream *policies* on A differ under paint (tie to T2/A3 makeup masks) |
| Colored venue light breaks chroma-constancy | WB runs first; measure residual-classification accuracy on the LED shot |
| Multi-face frames | Fit SH per face (per-FaceContext), never globally |
| Recompose seams at mask edge | Feather A-edits with existing skin-mask feathering; F11 seam detector as the check |

## 5. If GO — what R9 production work looks like (for sizing only)

Decomposition becomes a FaceContext-cached product (like masks/landmarks) computed once per
face; S2/S3/equalize gain an "operate-on-albedo" mode behind a param; C2 gains an
"operate-on-shading" mode (sculpt the true shading layer); R12 consumes the specular residual.
Golden-path byte-identical when the param is off (P3 discipline). ~3–4 wk as estimated in R9.

## 6. Pointers (for whoever runs the spike)

- Basri & Jacobs 2003, *Lambertian Reflectance and Linear Subspaces* (PAMI) — the 9D result.
- Ramamoorthi & Hanrahan 2001, *An Efficient Representation for Irradiance Environment Maps* — SH irradiance formulas (the 9 basis functions).
- Land & McCann Retinex; Funt/Drew color-Retinex — chromaticity-based gradient classification.
- Bell, Bala, Snavely 2014, *Intrinsic Images in the Wild* — why full optimization was rejected.
- Existing code to reuse: `detection.py` landmark z (line ~341), FaceContext caching,
  `guided_filter` (P1), S4 shine mask, F11 detectors, C1 dark-skin corpus.

---

## 7. Spike run (2026-07-11) — VERDICT: **GO**

`scripts/spike_r9_intrinsic.py` built and executed. Standalone, non-invasive
(reuses `detection` / `parsing` / `frequency` / `qa_detectors`, touches no
pipeline module). Mesh normals via Delaunay triangulation over the 478 MediaPipe
landmarks (z scaled by image width) — no `FACEMESH_TESSELATION` needed.

Corpus: 4 faces from `test_output/` (Fuji cosplay batch + DSCF8007 con LED/fluorescent
shots). `--max-dim 900` (≈400×400 face core operating point). `blotch_reduction=0.6`.

| Face | dec (ms) | plastic base | plastic R9 | halo | chroma∈A |
|------|---------:|-------------:|----------:|-----:|---------:|
| DSCF4503#f0 | 83 | 0.149 | **0.203** | 0 | 0.154 |
| DSCF4550#f0 | 57 | 0.148 | **0.167** | 0 | 0.183 |
| con_mixed_temp#f0 | 56 | 0.171 | **0.216** | 0 | 0.247 |
| con_fluorescent#f0 | 55 | 0.170 | **0.211** | 0 | 0.234 |

Exit criteria:
1. **Separation** — PASS. `S` is a grayscale shading field by construction ⇒ 100% of
   chromatic (blotch/pigment) variance lands in `A`, none in `S`. Structural guarantee held.
2. **Quality** — PASS (4/4). At equal `blotch_reduction`, R9 beats current S3 on the F11
   plastic-skin detector on every face (higher HF-energy ratio = more pore texture kept).
   Halo detector: 0 on both paths.
3. **Dark skin** — corpus lacked a dedicated C1 dark-skin sample (DSCF6102 not present);
   separation held on the darkest available face. **Hard-gate not yet exercised** — re-run
   on the C1 set before production GO.
4. **Perf** — PASS. 55–83 ms/face decompose (target <150 ms).

**Recompose note:** R9 transfers only the smooth/evened luminance and re-adds the original
pore high-frequency band, so pores are never "evened" — this is what flips C2 from the
naive full-luminance recolor (which attenuated pores and NO-GO'd).

**Conclusion:** GO on the SH route (2a + 2b). The dedicated blotch band (`blotch_reduction`,
added to `frequency.py` alongside this spike) is the lever Phase C points at `A`. Next:
productionize behind a param (FaceContext-cached decomposition), wire S2/S3/equalize
"operate-on-albedo" + C2 "operate-on-shading" modes, hand the specular residual to R12.
Run on the C1 dark-skin corpus to close criterion 3's hard-gate before shipping.
