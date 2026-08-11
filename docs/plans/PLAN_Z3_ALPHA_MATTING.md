# Z3 — Alpha Matting & Layer-Separated Background Compositing (Scoping Plan)

**Status:** ✅ phases 0–3 SHIPPED (2026-07-26) — 0+1 `52026f9`, 2 `6dd0089`, 3 `5ea3dbf` · halo fix
confirmed on real photos 2026-07-30 · ⚠️ **known gap found on real photos (2026-07-30):**
flyaway wisps past the main silhouette are erased, not haloed — two stacked defects
(`build_auto_trimap`'s hair branch is vacuous as written, and `hair_mask` is separately never
passed at the call sites) — see "Corpus gap closed" below · supersedes the Z3 framing in
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

### Corpus gap closed (2026-07-30) — real photos confirm phase 1, find a phase-2 wiring gap

Owner supplied two real folders: `~/Desktop/duotian nikke` (Nikke cosplay, white wig,
flyaway strands, busy indoor set) and `~/Desktop/bonodori2026` (matsuri crowd, real hair,
string-light bokeh background). Swept both through `studio_dream_v2`, `film_noir_cinema_v1`,
`editorial_elegance_v1`, `high_energy_glow_v1`, `minimal_film_v1` (all set `background_blur`,
the Z3-affected path), `anime_crystal_void`, and `apex_cinema_v1` (`lens_blur` only — negative
control, deliberately unwired per §"Wire-in decisions") at `--max-dim 2048`.

**Phase-1 halo fix holds on real photos.** Across every blur strength tested (6–35) and both
subjects, no bright ring or subject-colour bleed at the hair/bokeh boundary — the defect
measured in §1.1 does not reproduce post-fix on real images either.

**New finding: flyaway wisps are still erased, and the cause is two stacked defects upstream
of the solver — not just a missing wire.** Every fine flyaway strand extending past the main
wig/hair silhouette (the Nikke wig's loose strands past the antler prop; a strand crossing the
kimono/flower background in the bonodori shot) is cut cleanly at the silhouette edge — not
haloed, just gone — regardless of blur strength. Root-caused by reading the call sites and
verified with a synthetic unit check, not assumed:

- **Defect A — `build_auto_trimap`'s hair branch is vacuous as written.** Read the set algebra
  at `matting.py:60–75`: `trimap` starts all-128, then `background = erode(fg < 0.08)` is set to
  0 and `core = erode(fg > 0.80)` is set to 255. The remaining 128 pixels are *exactly*
  `~core & ~background`. The hair line then does
  `trimap[hair_band & ~core & ~background] = 128` — assigning 128 to a subset of pixels that
  are already 128. It cannot mark a single additional pixel unknown, no matter what
  `hair_mask` contains. Verified directly:
  ```python
  fg = np.zeros((200,200), np.float32); fg[60:140, 60:140] = 1.0
  hair = np.zeros((200,200), np.float32); hair[95:105, 140:180] = 1.0  # strand 40px past the edge
  build_auto_trimap(fg) == build_auto_trimap(fg, hair_mask=hair)  # True, 0-pixel delta
  ```
  `tests/test_matting.py` has zero references to `hair_mask` — this branch has never been
  under test, vacuous or otherwise.
- **Defect B — `hair_mask` is additionally never passed at either call site.**
  `background.py::blur_background` / `grade_background` both call `_composite_alpha(src_f,
  person_mask)` with no `hair_mask` argument (lines 493, 676) — it defaults to `None`. Even
  if Defect A were fixed, wisp recovery would still be inert without this.
- **Fixing B alone does nothing.** A future session must fix Defect A first (let hair evidence
  carve *into* the background set — e.g. subtract `hair_band` from `background` before the 0
  assignment, or add `trimap[hair_band & ~core] = 128` so it can flip pixels currently in
  `background`) — then wire `parse_hair_full_image` through `_stage_background`
  (`engine.py:3240`, no hair-mask parameter today) to `_composite_alpha`. The plan named the
  right mask source (§Method phase 2, "available at `parsing.py:359`") but neither the carving
  logic nor the plumbing exists.

**Where the next wall is, once A+B are fixed:** `solve_closed_form_alpha`'s proxy resolution
(default `max_dim=320`) is currently masked by Defect A — the wisp pixels never enter the solve
at all, so the proxy can't be blamed yet. Once the trimap actually admits wisps as unknown, a
1px strand at a 2048px input becomes ~0.16px in the 320px solve grid, which will be the next
binding constraint on wisp fidelity. Not measured this session; flagged so the next session
doesn't rediscover it from scratch.

**Not closed by this session:** both defects above need fixing, plus a decision on whether
`_stage_background` recomputes `parse_hair_full_image` (extra BiSeNet pass cost) or reuses the
`body_hair_mask` already computed in the body-skin-exclusion path (`engine.py` ~line 3430) —
real plumbing, not a one-line fix, and left for a follow-up session.

**Unrelated defect spotted, logged separately, not chased here:** `high_energy_glow_v1` on
the Nikke image produces a blown-out pink/red periocular region — a colour-grading issue, not
a background/matting one. Needs its own investigation.

Source images: `~/Desktop/duotian nikke/DSCF7585.jpg`, `~/Desktop/bonodori2026/DSCF8056.jpg`
(owner's machine, not copied into the repo — 24 MP RAF-derived JPEGs, too large to check in).
Cropped comparison evidence (source vs. rendered, wisp region) saved at
`docs/reference_targets/z3_corpus_evidence_2026-07-30/`.

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

### Defects A+B fixed (2026-08-12) — wisps now reach the solver, but still do not render

`57ac259`. Defect A (vacuous hair branch) and Defect B (`hair_mask` never passed) are both
fixed, with the engine resolving a hair mask in `_stage_background` via `parse_hair_full_image`
(the body-skin path computes the same mask but runs in a *later* stage, so it cannot be
reused — the plan's open "recompute or reuse?" question resolves to *recompute*, gated on a
matte op being active).

**Both fixes verified working at their own layer.** On `DSCF7585.jpg` at `--max-dim 2048` the
trimap gains 81,557 unknown pixels overall and 6,835 in the wig region — the wisps now enter
the solve, which is exactly what A+B were for.

**But the rendered output barely moves: only 57 px change in the head/wig third (max delta 6,
visually nothing), against 7,188 px on the legs/costume.** The wisps reach the solver and are
then discarded downstream. Root cause measured on controlled synthetics:

- Closed-form matting is under-confident on thin structures. An *opaque* 6px strand solves to
  alpha 0.544, and alpha is non-monotonic in strand width (4px 0.195 · 6px 0.544 · 10px 0.294 ·
  20px 1.000 · 30px 0.656 · 60px 0.000). A strand disconnected from the main silhouette solves
  to exactly 0.000 — closed-form propagates alpha from known-foreground through colour affinity,
  so an unknown island surrounded by known-background has no foreground constraint to draw from.
- `estimate_foreground` seeds trusted foreground only from `alpha > 0.85`. At 0.544 the strand
  contributes **0 of 390** seed pixels, so its colour is diffused in from the body across
  intervening zero-weight background and diluted toward the background colour.
- Sweeping `lambda_known` (1→1000) and `eps` (1e-7→1e-4) does not recover it; neither is a
  principled fix.

**Do not lower the 0.85 seed gate.** At alpha 0.5 the observed pixel is half old background by
construction (`src = a*F + (1-a)*B`), so seeding from it propagates old background into F —
reintroducing the exact fringe that phase 1 fixed and F-estimation exists to remove. That would
trade erased wisps for tinted wisps across every image, a regression on corpus-verified behaviour.

**Also worth noting:** BiSeNet's full-image hair confidence maxes at 0.190 on this frame (it
returns confidence, not argmax, by design) and is scattered across dark costume fabric — which
is why most of the render delta landed on the legs rather than the wig. Any real wisp fix needs
hair evidence that is better localized than this mask, or a connectivity constraint that only
admits hair contiguous with the silhouette.

**Next session — the real fix is in the solve/estimate layer, not the trimap:** either (a) seed
F from trimap-known-255 topology rather than alpha magnitude, and diffuse *along* the hair band
so F propagates down the strand instead of across background; or (b) raise the solve resolution
for thin structures (`solve_closed_form_alpha`'s `max_dim=320` proxy is a second, still-unmeasured
wall at 2048px input) . A+B are necessary but not sufficient; they are committed because they are
correct, tested, and a strict precondition for any of the above.
