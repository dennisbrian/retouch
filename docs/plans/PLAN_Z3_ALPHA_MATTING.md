# Z3 — Alpha Matting & Layer-Separated Background Compositing (Scoping Plan)

**Status:** ✅ **ALL PHASES SHIPPED (2026-07-26)** — 0+1 `52026f9`, 2 `6dd0089`, 3 `5ea3dbf` · validated on synthetic GT only (see §5 corpus gap) · supersedes the Z3 framing in
`RESEARCH_PERCEPTUAL_CALIBRATION_Z_2026_07_17.md` §Pillar 2 · **Parent:** MASTER_PLAN research backlog
**Effort:** phase 0 ~1 d · phase 1 ~2–3 d · phase 2 ~4–6 d · phase 3 ~2–3 d (~2–3 wk total, phased)
**Standout:** kills the #1 "shopped" tell in background-blurred cosplay/wig shots, and caps
the quality of four shipped features at once (`blur_background`, `lens_blur`, `grade_background`,
background harmonize).

---

## 0. Executive summary — the plan doc's premise was half wrong

The Z-series research doc frames Z3 as "build a trimap, solve closed-form alpha, halo goes away."
**Measured at HEAD, that is not the primary defect.** The dominant background-blur artifact
survives a *mathematically perfect* mask, because it originates in the blur **source**, not
the mask.

Consequences for scope:

- The load-bearing fix is **foreground/background layer separation**, not the Laplacian solve.
- The alpha solver is real, but it is **phase 2** — it buys hair-wisp fidelity *after* the
  layer bug is fixed. Shipping the solver alone would leave the visible ring untouched.
- The plan doc's proposed acceptance gate (`detect_halo` scores improve) is **invalid**
  and is replaced in §5.

---

## 1. Problem statement — measured, not assumed

### 1.1 The layer-bleed defect (primary, phase 1)

`background.py::blur_background` composites:

```python
blurred = cv2.GaussianBlur(src_f, ksize, sigma)   # sigma up to 4% of min dim
bg_mask = self._background_mask(person_mask, (h, w))
out     = blend_masked(src_f, blurred, bg_mask)
```

`blurred` is the Gaussian of the **whole image, subject included**. At a background pixel
within ~1 sigma of the subject edge, that average is dominated by subject pixels. The
composite then writes that subject-contaminated value into the background.

**Experiment (2026-07-26).** `chang_e_cosplay_tamed_shine.jpg` at 378×504, blur strength
100 (sigma ≈ 15px), person mask = a **hard rectangle** — zero mask error by construction.
Measuring the background scanline right of the subject edge, binned by distance, as a
projection onto the subject-color direction:

| distance from edge | \|shift\| | shift toward subject color |
|---|---|---|
| 3–15 px | 14.1 | 2.2 |
| 15–35 px | 93.8 | **93.3** |
| 35–70 px | 64.7 | **64.6** |
| 70–120 px | 35.6 | **35.4** |

Background pixels move up to **93 levels toward subject color**, decaying with distance in a
Gaussian-bleed signature that extends well past sigma. Rendered and viewed directly
(`/tmp/ring_proof.jpg`): a bright halo hugs the outside of the mask boundary with pale
subject color smeared along it.

**No alpha matte can fix this.** The mask was perfect.

> Note the 3–15px bin is *low* (2.2): the subject-side feather still holds sharp pixels there.
> The ring peaks just outside the feather — which is exactly where the eye reads "cut out."

### 1.2 The matte-fidelity defect (secondary, phase 2)

Separately and genuinely: person/hair masks are BiSeNet argmax plus C3 guided feathering
(default-off). A soft *mask* is not an *alpha matte* — semi-transparent hair wisps are binary-ish,
so they get eaten or ghosted. This is real, and it is what closed-form matting addresses. It is
simply not the cause of §1.1.

### 1.3 What exists today

- `retouch/matting.py` (98 LOC) — a spike. Its own docstring defers the closed-form solver;
  what's implemented is `guided_filter` on the mask behind a confidence gate. **Zero production
  importers**; only `tests/test_matting.py` (synthetic rectangles, bounds/dtype/fallback only).
- `background.py::_feathered_person_mask` already does guided/feathered masking at
  4 call sites (lines 111, 253, 503, 565).

So wiring `matting.py` as-is swaps one guided mask for another and changes nothing visible.

---

## 2. Method

### Phase 1 — layer-separated compositing (the actual halo fix)

Estimate a background-only layer with the foreground excluded, blur *that*, then composite:

```
out = α·F + (1−α)·blur(B)
```

- **B estimate:** push-pull / pyramid fill (or `cv2.inpaint` on the eroded subject region) so
  no subject pixel contributes to the background blur. Cost is one pyramid pass.
- Applies identically to `blur_background` and `lens_blur`. `grade_background` is a color
  op, not a spatial one — it does not bleed and needs no B-layer (confirm during phase 1).
- **Ships standalone.** Visible improvement with today's masks, no solver, no new dependency.

### Phase 2 — closed-form alpha in the unknown band

Trimap from BiSeNet (erode = FG, dilate-complement = BG, band from hair-region scale;
body-wide hair incl. wig hair is available at `parsing.py:359`). Solve alpha in the band via
**closed-form matting** (Levin et al., PAMI 2008): assemble the matting Laplacian, CG-solve
band-only, refine to full res with `utils.guided_filter` (He et al.).

Plus **foreground-color estimation** so wisps re-blend over the new background without
color fringes — this pairs with the phase-1 B-layer to complete `α·F + (1−α)·B`.

KNN matting (Chen/Li/Tang 2013) is the fallback if the Laplacian solve is too heavy.

### Phase 3 — wire-in

Matte replaces the person mask **only inside background ops' compositing step**. Masks used
for *statistics* (harmonize's skin anchor, exposure sampling) stay as-is — swapping those
would shift color decisions, which is out of scope and a separate risk.

---

## 3. Dependencies — decision required

`scipy.sparse` is the natural way to assemble and CG-solve Levin's Laplacian.

- **Installed locally:** scipy **1.13.1** ✅
- **Declared:** ❌ **not in any requirements file / pyproject / setup.py**
- `cv2.alphamat` is **not available** in this OpenCV build (checked: `hasattr → False`).

**This is a decision for the owner, not an implementation detail.** Options:

1. Declare scipy as a real dependency (it's already imported transitively by the ML stack in
   practice, but undeclared).
2. Keep it optional — gate phase 2 behind a soft import, fall back to phase-1-only behavior.
3. Hand-roll a band-limited CG solver on dense numpy blocks (avoids the dep; slower, more code).

**Recommendation: (2).** Phase 1 needs no solver at all, so the engine keeps working with zero
new hard dependencies, and phase 2 degrades gracefully where scipy is absent.

---

## 4. Integration constraints

- **Resolution.** Background ops are global stages. Under the documented default
  `quality="full"` path these run at **native** resolution. So "solve at proxy, guided-filter
  upsample" is a **requirement**, not an optimization. Note `utils.guided_filter` already
  downsamples its coefficients (`max_dim=1200` default).
- **Caching.** The plan doc says "cache per (image, parse) like face contexts," but
  `FaceContext` is **per-face** and the matte is **whole-image** — it does not belong there.
  It needs an image-level cache slot, because GUI slider drags re-enter `process()` with cached
  contexts; an uncached band solve would make every drag pay for it.
- **Budget.** Band-limited solve ≤ ~400 ms at proxy (from the research doc; carry it forward).
- **Dtype.** Both uint8 and float32 [0,255] paths exist via `_to_f32_255` / `_restore_dtype`;
  keep round-trip parity.

---

## 5. Acceptance criteria

> **`detect_halo` is NOT a valid gate.** `qa_detectors.py:469` measures *sharpening overshoot*
> via Canny edges + L-channel plateau. It scores sharpening artifacts, not matte fidelity or
> layer bleed. The research doc's "detect_halo scores improve" criterion is dropped.

**Phase 0 (build first, ~1 d).** Synthetic ground-truth harness: composite a known alpha
(hair-strand render) over two known backgrounds. Gives true GT for both defects:
- **Layer bleed:** with GT alpha supplied, background-region error vs GT must drop to ~0.
  The §1.1 distance-binned probe becomes the regression test.
- **Matte fidelity:** SAD / gradient error in the unknown band vs GT alpha.

**Phase 1.** Distance-binned contamination (the §1.1 table) collapses toward flat.
Clean-input identity: strength 0 ⇒ byte-identical. uint8/float32 parity.

**Phase 2.** Band SAD beats the current feathered mask on synthetic GT. Runtime within budget.

**Phase 3.** Real-image renders viewed directly (repo rule — no prose-only claims), across all
four call sites, both dtype paths, `quality="draft"` and `"full"`.

**Corpus gap — needs owner input.** `docs/reference_targets/` holds exactly **one** JPEG. The
research doc assumes a "cosplay corpus" that is not in the repo. Validation is scoped as
*synthetic GT + N real images, corpus path TBD*. The newly shipped
`./executable/random_image_test <folder>` takes an image folder directly, so pointing it at a
wig/hair-over-busy-background set is the intended path once the owner supplies one.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Laplacian solve too slow at native res | Band-only solve at proxy + guided upsample (§4); KNN fallback |
| scipy dependency unwanted | Phase 1 needs none; gate phase 2 behind soft import (§3) |
| B-layer inpaint invents texture behind subject | It is blurred immediately; only low-frequency content survives. Verify on busy backgrounds |
| Matte regresses statistics-driven ops | Phase 3 restricts the swap to compositing only (§2) |
| Over-fitting to one image | Phase 0 synthetic GT is the primary gate; real corpus secondary |

---

## 7. Phase ordering

Each phase is independently shippable and verifiable:

| Phase | Deliverable | Ships alone? | Status |
|---|---|---|---|
| 0 | Synthetic GT harness + §1.1 regression probe | ✅ tests only | ✅ **DONE** `52026f9` |
| 1 | Layer-separated composite, existing masks | ✅ **visible halo fix** | ✅ **DONE** `52026f9` |
| 2 | Closed-form alpha + F-color estimation in band | ✅ wisp fidelity | ✅ **DONE** `6dd0089` |
| 3 | Wire to the background-compositing sites | ✅ full feature | ✅ **DONE** `5ea3dbf` |

### Phases 2+3 as shipped (2026-07-26)

**The headline finding: F-estimation is load-bearing, alpha alone is not.** An observed edge
pixel already is `src = a·F + (1−a)·B_original`, so compositing `a·src + (1−a)·new_bg`
double-counts the old background, leaving `a·(1−a)·(B_original − new_bg)` — the colour
fringe. Measured with a **perfect** matte: composite error **4.295** using `src` as
foreground vs **0.000** using true F. A better matte with no F-estimate buys nothing.

**A phase-1 bug the soft-alpha GT exposed.** `_background_layer` seeded its push-pull fill
from the soft weight `1−a`. A translucent pixel carries subject colour, so admitting it
reintroduced the very bleed phase 1 removed — precisely at the wisp band. The estimated
background layer was **9.68** levels off GT; seeding from confident background only (≥ 0.98)
brings that to **1.06**, and *improved phase 1's own* hard-mask numbers (near-edge
3.40 → 0.65, whole-bg MAE 0.86 → 0.70).

Composite error vs GT in the unknown band (70 anti-aliased strands over a known background):

| configuration | error |
|---|---|
| feathered mask, no F | 2.199 ← previous behaviour |
| closed-form, no F | 2.000 |
| **closed-form + F** | **1.765** ← shipped |
| GT alpha + F | 1.502 (ceiling) |

Alpha SAD in band: 0.1508 (feathered) → 0.1351 (closed-form).

**Wire-in decisions.** Matte applies to `blur_background` and `grade_background` — the ops
that composite a modified *background* against the subject. Deliberately unchanged:
`lens_blur` (composites through a graded 3-level depth map, not a single mask — needs the
depth blend reworked, not a substitution), `light_wrap` (reads the silhouette to spill glow
inward; alpha changes wrap geometry), `sharpen_subject` (targets the subject),
`replace` / `relight_scene` (full-frame). Statistics masks stay as-is per §2.

**Performance.** `estimate_foreground` ran 12 full-res Gaussian blurs — 1282 ms at 4K alone.
Now solved on a 512px proxy with full-res opaque pixels preserved exactly. The matte is
cached per (mask ⊕ image) decimated checksum, since GUI drags re-enter `process()`.

| resolution | cold | warm |
|---|---|---|
| 1440×1080 | 566 ms | 150 ms |
| 2880×2160 | 1183 ms | 712 ms |

**SciPy stays optional and undeclared** (§3 option 2): `solve_closed_form_alpha` returns
`None` without it and callers fall back to the guided-filter refinement. A regression test
pins that the no-SciPy path still renders.

> ⚠️ **Validated on synthetic ground truth only.** No real wig/hair corpus exists in-repo, so
> the hair-wisp fidelity claim rests on synthetic anti-aliased strands, not photographs. Real
> renders were viewed for absence of halos/artifacts, but not for wisp fidelity against a
> reference. Point `./executable/random_image_test <folder>` at a wig-over-busy-background
> set to close this.

### Phases 0+1 as shipped (2026-07-26)

`background.py::_background_layer` estimates a background-only layer by normalized
("push-pull") convolution — blur the masked image and the mask with the same kernel, then
divide, so every output pixel averages **background pixels only**. `blur_background` and
`lens_blur` blur that layer; the subject still composites from the sharp original.

Measured against true ground truth (real subject over a known synthetic background, hard
mask, so the correct bokeh is computable):

| distance from edge | OLD err | NEW err |
|---|---|---|
| 3–15 px | 23.69 | **3.40** |
| 15–35 px | 0.44 | 0.67 |
| 35–70 px | 0.00 | 0.67 |
| **whole-background MAE** | **1.69** | **0.86** |

The small far-field increase (~0.67) is the proxy fill's smooth extrapolation — visually
inert, confirmed by direct render comparison against ground truth.

**Performance note.** The fill is solved on a 256px proxy: it only has to be smooth and
subject-free (the caller blurs it immediately), and image-spanning kernels at native
resolution cost ~14.6 s at 4K. After the proxy, 4K `blur_background` is **589 ms**
(was 14,638 ms); 1440×1080 is 99 ms.

**Phase 0 harness** (`tests/test_background_layer_bleed.py`) uses a hard rectangular mask so
mask error is zero by construction, and was confirmed to **fail pre-fix** (+47.6 levels
`blur_background`, +14.5 `lens_blur`) before being made to pass. It also guards the obvious
false fix — one test asserts the background is still actually blurred.

> **Measurement caveat, recorded so it is not repeated.** The §1.1 probe on the raw
> `chang_e_cosplay_tamed_shine.jpg` is **not** a valid A/B: that file is a 3×3 *collage*, so an
> arbitrary rectangle does not bound a subject and the "background" beside it contains bright
> collage panels lying near the subject-colour direction. Post-fix it still reads ~93 because it
> is measuring the blur redistributing *background* content, not subject bleed. Use the
> ground-truth composite above (or the synthetic harness) for any future A/B.

**Phase 1 carries most of the user-visible value at ~2–3 d.** If budget is tight, phases 0+1
are the high-return slice; 2+3 are the quality ceiling.

---

## 8. Files touched

`retouch/background.py` (composite paths), `retouch/matting.py` (solver — currently a spike),
`retouch/utils.py` (reuse `guided_filter`), `retouch/params.py` (+ `gui.py` component entry
per the single-source-of-truth rule), `tests/test_matting.py`, new synthetic-GT test module.
