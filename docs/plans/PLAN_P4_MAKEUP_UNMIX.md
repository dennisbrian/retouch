# P4 — Skin ↔ Makeup Unmixing (Research + Implementation Plan)

**Status:** 📋 RESEARCH LOCKED (design 2026-07-14) · **⚠️ 2026-07-15 dark-skin audit: shipped solver fails on Fitzpatrick V–VI — see §14 before enabling in recipes** · **✅ 2026-07-15 sibling bug found + FIXED in `specular.py` (same absolute-threshold pattern) — see §15** · **✅ 2026-07-15 third instance found + FIXED in `lips.py`, fourth found + documented (not fixed, higher blast radius) in `skin.py::whiten()` — see §16** · **Parent:** `PLAN_SKIN_PROMAX.md` §P4 · MASTER_PLAN research backlog  
**Effort:** spike 4–5 d · full feature ~3–4 wk if GO  
**Standout:** highest-moat cosplay axis — separate **paint layer** from **person**, not merely mask around paint (A3/R11).

---

## 1. Problem statement

Cosplay retouch is not “skin with lipstick.” Observed face is often:

```
I ≈ f( skin_reflectance , makeup_stack , lighting )
```

Today’s stack:

| Capability | What it does | Limit |
|---|---|---|
| A3 makeup-aware smooth | Avoids smoothing **across** paint edges | Never models paint as a layer |
| R11 `facepaint_crack_repair` / `paint_coverage_even` | Needs a **given** `paint_mask` | No automatic unmix; mask is external |
| R7 `decompose_chromophores` | Melanin + hemoglobin in **one plane** | Assumes skin optics; makeup violates prior |
| R9 `decompose_intrinsic` | Albedo × shading | Makeup bleeds into albedo; no α coverage |
| T2 `makeup_v2` | **Synthesizes** makeup onto skin | Forward only — cannot remove/edit real paint |

User wants:

1. Even **foundation coverage** without shifting undertone.  
2. Fix **skin** (redness, oil, blotch) *through* light makeup.  
3. Kill **caking** (α spikes in pores/creases) without wiping contour.  
4. Edit contour/paint **symmetry** independent of face shading.  
5. Leave bare skin alone when α≈0.

Masking ≠ unmixing. P4 is the unmixing sibling of P1 (physiological prior) and the product sibling of A3.

---

## 2. Physical / computational model (v1 classical)

### 2.1 Minimal image formation (shippable)

**Overpaint (alpha composite) in linear RGB:**

\[
I = (1 - \alpha)\, S + \alpha\, M
\]

- \(S\): skin color (diffuse, post-shade or pre-shade — see §2.3)  
- \(M\): makeup color (foundation / paint / contour)  
- \(\alpha \in [0,1]\): coverage map  

**Foundation density (optional second mode):** multiplicative in optical density:

\[
L = -\log I \approx L_S + \alpha\, D_M
\]

Density mode matches heavy white paint / stage makeup better; alpha mode matches sheer foundation. Spike both; pick one default.

### 2.2 Skin prior (bootstrap — do not wait for full P1)

**Use R7 as soft prior (available today):**

- On pixels that reconstruct well from melanin+Hb (`I_hat` from concentrations), residual \(\|I - I_hat\|\) is low → likely bare skin.  
- High residual + high chroma / atypical hue for Fitzpatrick band → makeup candidate.  
- Moles/freckles: compact melanin (R10 `blemish_vs_mole`) → **not** makeup; protect.

**Optional upgrade path:** P1 Kubelka–Munk / dipole SSS as stronger prior later. P4 v1 must GO without P1.

### 2.3 Shading interaction

Makeup sits mostly on **albedo**. Prefer:

1. R9: \(I = A \cdot Sh\) (or log-additive)  
2. Unmix on \(A\): \(A = (1-\alpha) A_S + \alpha M\)  
3. Recompose \(I' = A' \cdot Sh\)

If R9 unstable on a crop, fall back to unmix on \(I\) with soft shading freeze (guided filter on luminance only).

### 2.4 Contour vs foundation

| Layer | Signature | Edit policy |
|---|---|---|
| Foundation / base | Broad α, low-freq, near-skin hue | Even α; smooth \(A_S\) under |
| Contour / blush / eyeshadow | Spatial priors (landmark zones) + chroma | Edit M or α in zone; don’t global-even |
| Lipstick / liner | BiSeNet lips + high chroma | Separate; usually leave or use lip tools |
| White stage paint | Very high L, low chroma, high residual from R7 | Density mode; crack repair already R11 |

v1 ships **foundation/base unmix** + generic α map; contour-aware split is v1.1.

---

## 3. Algorithm sketch (spike → product)

### Stage U0 — Inputs

- Face ROI BGR (float32 preferred, E1 canvas)  
- Skin mask, landmarks, optional BiSeNet makeup-adjacent labels  
- Optional user paint mask (GUI brush) — always wins over auto α

### Stage U1 — Candidate makeup residual

```text
mel, hb = decompose_chromophores(I)
I_hat   = reconstruct_from_chromophores(mel, hb)   # NEW thin helper if not exported
R       = ||I - I_hat||  (perceptual: ΔE or chroma-weighted)
skin_ok = R < t_skin  AND  in_skin_mask  AND  not mole_mask
makeup_hint = softmask(R > t_mu OR chroma_outlier OR L_outlier)
```

### Stage U2 — Solve α, M, S (constrained)

Per-pixel underdetermined (3 eqs, many unknowns). Use **region constraints**:

**Recommended spike solver (classical, robust):**

1. Estimate global or low-order \(M\) in paint-heavy zones (median color of high-`makeup_hint` pixels, or per superpixel).  
2. Estimate \(S\) in `skin_ok` zones; propagate under makeup via low-rank / guided fill (use shading from R9 or bilateral on neighbor bare skin).  
3. Closed form α:

\[
\alpha = \mathrm{clamp}\left(\frac{(I - S)\cdot(M - S)}{\|M - S\|^2 + \epsilon}, 0, 1\right)
\]

4. IRLS: re-estimate \(M\) as coverage-weighted mean; iterate 2–4 times.  
5. Regularize α: guided filter with \(I\) as guide; morphological open to kill pore noise; α=0 outside face/skin dilate.

**Alternative (if closed form fails):** NMF / sparse coding on log-RGB with 2 endmembers (skin atlas + makeup) — heavier, keep as spike Plan B.

### Stage U3 — Operators on layers (product API)

| Op | Math intent | User param |
|---|---|---|
| `makeup_coverage_even` | Low-pass α toward regional median | 0–1 |
| `makeup_cake_reduce` | Suppress high-freq α (pores/creases) | 0–1 |
| `skin_through_makeup` | Run existing skin ops on \(S\), recompose | reuses smooth/redness |
| `makeup_recolor` | Shift \(M\) hue/value, keep α | optional |
| `makeup_remove` | α → 0 blend (dangerous; cap) | 0–1 soft |

Recompose:

\[
I' = (1-\alpha') S' + \alpha' M'
\]

then convert back to pipeline dtype.

### Stage U4 — Safety

- α floor/ceiling; max remove strength  
- Never edit eyes/lips/brows by default (mask out)  
- Mole/freckle protect (R10)  
- Byte-identical when all P4 strengths = 0  
- F11 / pore_spectrum / plastic detectors still run post-recompose

---

## 4. Code placement (follow project patterns)

| Piece | Module | Notes |
|---|---|---|
| Core unmix | `retouch/makeup_unmix.py` **new** | Leaf; imports chromophore, intrinsic, numpy/cv2 only |
| Reconstruct helper | `chromophore.py` or unmix module | If adding reconstruct, keep R7 API stable |
| Params | `params.py` | `makeup_unmix`, `makeup_coverage_even`, `makeup_cake_reduce`, `skin_through_makeup` (names TBD single-source) |
| Wire | `skin.py` / `_process_face_core` | After parse, before or after frequency smooth — **spike decides order** |
| Recipe keys | nested `makeup_unmix.*` | dead-key guard must recurse (already does) |
| Tests | `tests/test_makeup_unmix.py` | synthetic composites first |

**Do not** put solver in `makeup_v2.py` (that is synthesis) or `cosplay_moat.py` (A3 edge blend).

---

## 5. Spike plan (4–5 days) — GO / NO-GO

Mirror `PLAN_R9_INTRINSIC_SPIKE.md` discipline.

### Deliverables

1. `scripts/spike_p4_makeup_unmix.py`  
2. Synthetic suite: known α/M/S → recover α within error budget  
3. Real cosplay triptychs: input | α heatmap | recomposed even-coverage  
4. Written GO/NO-GO in EXECUTION_LOG

### Synthetic tests (must pass for GO)

| ID | Setup | Pass |
|---|---|---|
| S1 | Constant S, constant M, disk α=0.6 | recovered mean α error < 0.08 |
| S2 | S with smooth gradient, M white paint, α soft edge | edge IoU > 0.7 vs GT α>0.3 |
| S3 | Bare skin only α=0 | mean recovered α < 0.05 |
| S4 | Mole (dark melanin spot) under no paint | not classified as makeup (α low on mole) |
| S5 | strength=0 path | byte-identical to input |

### Real-image exit (GO needs ≥3/4 subjective)

Corpus: white-makeup cosplay, sheer foundation portrait, bare-skin control, heavy contour.

| Gate | Criterion |
|---|---|
| Coverage even | Patchy foundation looks more even; undertone not grayed |
| Skin through | Redness_even / blotch on S reduces blotch without nuking paint edge |
| Cake | Pore caking reduced; no plastic slab |
| Bare control | No visible change on bare-skin portrait at default auto α |

### NO-GO criteria

- Cannot beat R11 `paint_coverage_even` given an oracle mask on the same images  
- α always paints hairline/eyes  
- Dark skin (Fitzpatrick V–VI) systematically over-estimated as makeup  
- Runtime > 200 ms extra per 400px face crop on CPU (ballpark; soft)

If NO-GO: document failure mode; options = (a) user-brush α only, (b) wait P1 prior, (c) A4 neural segmenter for paint (parked).

---

## 6. Full implementation slices (only if GO)

### Slice P4.1 — Library + synthetic tests (~1 wk)

- `makeup_unmix.py`: `unmix_makeup`, `recompose`, `even_coverage`, `cake_reduce`  
- Synthetic GO tests promoted to pytest  
- No engine wire yet  

### Slice P4.2 — Engine wire + params (~1 wk)

- ParamSpecs + recipe keys  
- Call site in face core; 0 = identity  
- Golden path byte-identical  
- GUI sliders only if wiring agent free (else CLI/recipe first — ground rule #7 reachability via recipe)

### Slice P4.3 — Cosplay recipes + visual QA (~1 wk)

- Flagship recipe or `cosplay_*` bundles turn on light unmix  
- Visual QA CRITICAL gates (texture / natural / edge)  
- Compare vs A3-only baseline  

### Slice P4.4 — Contour layer split (optional ~1 wk)

- Landmark-zoned M estimation (cheek contour vs base)  
- Symmetry helper  

---

## 7. Interaction with existing systems

| System | Interaction |
|---|---|
| A3 wig lace / stockings | Orthogonal (body/hair); no conflict |
| R11 paint ops | Prefer P4 α as automatic `paint_mask`; keep R11 for crack/inpaint |
| R10 chromophore suite | Share mel/Hb maps; run unmix before or after mole protect — **protect moles in α** |
| R12 specular finish | Specular is surface; run specular after recompose or exclude specular pixels from α fit |
| A5 QA back-off | If plastic after even_coverage, reduce coverage_even first |
| Per-face recipes | P4 strengths are face-local → work with `face_params` for free |

**Recommended face-core order (draft):**

```
parse → (optional) unmix to S,α,M
     → skin ops on S (frequency, blotch, chromophore)
     → recompose with edited α/M
     → makeup_v2 synthesis (adds virtual makeup on top)
     → lips/eyes
```

Synthesis after unmix avoids fighting real paint with virtual paint.

---

## 8. Risks

| Risk | Why hard | Mitigation |
|---|---|---|
| Ill-posed inverse | 3 channels, many free vars | Region constraints + low-order M + iterate |
| Dark skin false makeup | Residual model bias | Fitzpatrick-aware thresholds (R14 ITA); bare-skin corpus |
| White paint / flash | Clipping, no chromophore fit | Density mode; highlight exclude |
| Hairline / lace front | Texture confuses residual | Hair mask, lace blend (A3) exclude from α |
| Double with R9 failure | Bad shading → bad S | Fallback path; spike measures both |
| Scope creep to full P1 | Months | Hard cap: R7 prior only for v1 |

---

## 9. Acceptance (product)

1. Recipe-reachable + test asserts reachability (MASTER_PLAN ground rule #7).  
2. strength=0 byte-identical.  
3. Synthetic S1–S5 green.  
4. Visual QA on cosplay white-makeup + bare control.  
5. No regression on `natural` / non-makeup portraits (auto α near 0).

---

## 10. Relationship to other frontier items

| Item | Relation |
|---|---|
| **P1 SSS** | Stronger prior; **not** blocking |
| **P2 CSF** | Orthogonal plastic fix; can pair later |
| **R7/R9/R10/R11** | Substrate — already in tree |
| **A4 neural paint seg** | Only if classical α fails NO-GO |
| **Per-face recipe** | Independent product; ship first |

---

## 11. Decision record (2026-07-14)

1. **Bootstrap prior = R7**, not wait P1.  
2. **Default formation = alpha composite in linear/float**; density mode as flag if white-paint fails.  
3. **Unmix on albedo when R9 available**, else on I.  
4. **Spike before any engine wire** — same bar as R9 spike doc.  
5. **Product order:** per-face recipe Slice 1 → P4 spike → (if GO) P4.1–P4.3.  
6. **Do not** mark P4 ✅ until recipe/CLI reachability + visual QA (no more shipped-unwired islands).

---

## 12. First code file skeleton (for implementer — do not invent extras)

```python
# retouch/makeup_unmix.py
"""Skin↔makeup layer unmix (P4). Classical alpha composite + R7 residual prior."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

def unmix_makeup(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    *,
    mole_mask: Optional[np.ndarray] = None,
    user_alpha: Optional[np.ndarray] = None,
    use_intrinsic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (skin_bgr, makeup_bgr, alpha HxW float32)."""
    ...

def recompose(skin: np.ndarray, makeup: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    ...

def even_coverage(alpha: np.ndarray, strength: float, guide: np.ndarray) -> np.ndarray:
    ...

def cake_reduce(alpha: np.ndarray, strength: float) -> np.ndarray:
    ...
```

Spike owns the `...` until GO; product wire only after synthetic + real exit criteria.

---

## 13. Research findings (code audit 2026-07-14)

### 13.1 What already ships (substrate)

| Module | Status | Gap for P4 |
|---|---|---|
| `chromophore.decompose_chromophores` | ✅ R7 live | **No** `reconstruct` / `I_hat` export — residual needs ~15 LOC forward map from mel/Hb + optional ambient `c` |
| `intrinsic.decompose_intrinsic` / `even_albedo` | ✅ R9 | Unmix should prefer albedo; recomposes `A*S` documented in module |
| `skin_chromophore.facepaint_crack_repair` | ✅ R11 | Requires **caller-supplied** `paint_mask` |
| `skin_chromophore.paint_coverage_even` | ✅ R11 | Same — oracle mask only |
| R11 tests | `tests/test_chromophore_suite.py:206–224` | Synthetic white paint + full mask — **good seed for spike fixtures** |
| High-chroma exclusion | `skin.py:1289–1295` unify_hue_line | Soft makeup/tattoo gate on chroma — **hint signal**, not α |
| Body paint exclude | `engine.py` body path ~3184 | High-chroma body zones excluded — pattern to copy |
| `makeup.py` / `makeup_v2.py` | Synthesis only | Do **not** put unmix here |
| `cosplay_moat` | Edge blend / stockings / WB lock | Orthogonal |
| Auto `paint_mask` | **Missing** | No BiSeNet “makeup” class used as paint; R11 never auto-builds mask |

### 13.2 R7 as residual prior — soft GO

Forward model today:

```
L = -log(rgb) ≈ M @ [mel, hb] - c
```

Decompose solves for mel/hb after blue-differencing (cancels `c`).  
**Reconstruct:** pick `c` from median of bare-skin pixels (or 0 trial),  
`log_rgb_hat = M @ conc + c`, `rgb_hat = exp(-log_rgb_hat)`.

Expected:

- Bare skin → low residual  
- White stage paint / vivid makeup → high residual → makeup_hint  
- Dark skin: mel high but still on manifold → residual should stay low **if** M basis OK; **must** test Fitzpatrick V–VI in spike (NO-GO if systematic false makeup)

`even_albedo` on painted faces will treat paint as albedo blotch — **do not** run heavy albedo_even before unmix; order matters (§7).

### 13.3 Call-site order (face core) — refined

From `_process_face_core` structure (frequency → skin ops → makeup_v2 → …):

```
parse regions
→ [P4] unmix → (S, M, α)          # NEW early
→ frequency / blotch / chromophore ops on S (existing)
→ recompose I' = (1-α')S' + α'M'
→ makeup_v2 synthesis             # virtual makeup after real-paint handle
→ lips / eyes / hair
```

R11 `paint_coverage_even(I, paint_mask=α)` can become thin wrapper once α exists.

### 13.4 Auto paint_mask — cheapest classical path (no neural)

```text
skin_mask ∩ ¬ lips ¬ eyes ¬ brows ¬ hair
∩ ( residual_R7 > t  OR  C_oklch > t_c  OR  L very high + C low )  # white paint
∩ ¬ mole_mask (R10)
→ guided filter / morph close → α_init
```

Then closed-form α refine (§3 Stage U2). User brush overrides all.

### 13.5 Missing code inventory (spike checklist)

1. `reconstruct_from_chromophores(mel, hb, c=...)` in `chromophore.py` or unmix module  
2. `unmix_makeup` / `recompose` / `even_coverage` / `cake_reduce`  
3. Synthetic composite fixture builder (reuse test_chromophore_suite patterns)  
4. Dark-skin residual regression test  
5. Engine wire **only after** GO  

### 13.6 Soft verdict (agent wave 2 — R7 fidelity)

| Question | Answer |
|---|---|
| Substrate enough for spike? | **Yes** — R7+R9+R11+tests |
| Residual-only prior | **NO-GO** — unconstrained LS residual ~0; white paint/foundation on-manifold |
| Multi-cue α_init | **Required** — chroma ∨ (L↑∧C↓) ∨ nonneg-clamp residual ∨ spatial |
| Blocked on P1 SSS? | **No** |
| Blocked on A4 neural? | **No** for spike; only if classical NO-GO |
| Biggest unknown | Dark-skin FP + white-paint + foundation |
| Effort to first synthetic GO | **2–3 d** if focused |
| Full synthesis | `RESEARCH_PER_FACE_AND_P4.md` |

### 13.7 Ready-to-start ranking (frontier, code-backed)

| Rank | Item | Ready? | Why |
|---|---|---|---|
| 1 | **Per-face recipe Slice 1** | ✅ ready | Pool/ctx already per-payload; ~400 LOC |
| 2 | **P4 unmix spike** | ✅ soft GO | multi-cue α locked; reconstruct ~15 LOC |
| 3 | P2 CSF budget | partial | R13/M metrics exist; no CSF curve |
| 4 | P7 cross-region | partial | S1 body skin exists; joint tone not closed |
| 5 | P6 preference | blocked | needs owner edit pairs + F9 features dump |
| 6 | P1 SSS | research only | R7 linear ≠ layered transport |
| 7 | P5 temporal | blocked | no burst pipeline / embeddings |
| 8 | P8 aging vector | blocked | needs M/I/P1 stack |

**Do next when coding:** per-face Slice 1 → P4 spike script.

---

## 14. Dark-skin audit (2026-07-15) — VERDICT: concern is REAL, measured, mechanism identified

Audit of the **shipped** solver (`retouch/makeup_unmix.py`, wired at
`perf_optimizations.py:307-330` via `skin.makeup_coverage_even` /
`skin.makeup_cake_reduce` recipe keys). Evidence is empirical —
`scripts/spike_p4_darkskin_probe.py` runs the shipped functions across a
Fitzpatrick I–VI palette; numbers below reproduce deterministically (seed 42).

### 14.1 What the shipped solver actually does (constraints inventory)

- `estimate_makeup_alpha` = multi-cue α_init: three cues OR-combined, **all with
  hardcoded absolute thresholds**: `cue_chroma = smoothstep(0.10, 0.20, C_oklch)`,
  `cue_white = smoothstep(0.85, 0.95, L) * (1 - smoothstep(0.02, 0.08, C))`,
  `cue_resid = smoothstep(0.04, 0.12, |I - I_hat|_linear/255)` — then 5×5
  `MORPH_CLOSE` (dilates sparse suprathreshold noise into regions).
- `unmix_makeup` seeds M = mean color of `α_init > 0.35` pixels (pale-pink
  literal `[220,200,210]` fallback), S = mean of `α_init < 0.15` pixels, then
  3-round IRLS with `closed_form_alpha` (projection of I−S onto M−S) and
  re-seeding at `α > 0.3`. No specular exclusion anywhere (despite §7's R12 row).

### 14.2 Measured failure table (probe output, 2026-07-15)

| Tone | bare α_init | bare α e2e | foundation α recovered (true 0.5) | phantom α outside makeup | coverage_even max damage on bare px (specular case) |
|---|---|---|---|---|---|
| I   | 0.362 | 0.009 | **0.899** | 0.045 | 124/255 |
| II  | 0.406 | 0.009 | 0.043 | 0.058 | 121/255 |
| III | 0.477 | 0.032 | 0.345 | 0.095 | 103/255 |
| IV  | 0.514 | 0.029 | 0.063 | 0.106 | 101/255 |
| V   | 0.544 | 0.057 | **0.000** | 0.154 | 63/255 |
| VI  | 0.560 | 0.096 | **0.000** | **0.214** | 45/255 |

Three distinct failure modes, two of them tone-dependent:

1. **Detection collapse on dark skin (feature silently inert).** Tone-matched
   foundation at true α=0.5 recovers α≈0.9 on Fitzpatrick I, **0.000 on V–VI**
   (and is already fragile at II/IV). Mechanism: every α_init cue measures
   signal in a space that scales with base reflectance — a fixed makeup optical
   density produces a linear-RGB delta ∝ base intensity (dI = I·Δdensity), and
   OKLab chroma likewise compresses at low L — so absolute thresholds tuned on
   light skin never fire on dark skin. The hint then never localizes the
   makeup, so the M seed collapses to ≈ mean skin (M≈S), making
   `closed_form_alpha`'s denominator degenerate. This trips the plan's own
   NO-GO framing in reverse: not *over*-estimation, but systematic
   *non-detection* — the shipped `makeup_coverage_even` does nothing on V–VI.
2. **False-positive α rises monotonically with darkness.** Bare noisy skin:
   end-to-end α mean 0.009 (I) → 0.096 (VI), 10×; phantom α on the bare region
   of a made-up face 0.045 → 0.214, ~5×. Mechanism attributed per-cue: on flat
   patches all cues are 0 for every tone; add σ=4 sensor noise and `cue_resid`
   mean goes 0.092 (I) → 0.164 (VI). The chromophore-reconstruct residual floor
   from noise alone (0.036–0.045) sits **inside** the hardcoded 0.04–0.12
   smoothstep band — the cue is effectively a noise detector, and noise couples
   to residual more strongly at high melanin (nonneg concentration clamp +
   single scalar ambient `c` leave more unexplained at low reflectance). The
   5×5 MORPH_CLOSE then amplifies sparse hits into the 0.36–0.56 α_init means.
3. **Specular → coverage_even paints skin (all tones, worst on light).** A
   small desaturated highlight seeds M from highlight pixels; `even_coverage_alpha`
   then blurs that α outward and recompose blends surrounding bare skin toward
   the highlight color — max bare-pixel change 45–124/255 at strength 0.7.
   Not tone-specific in direction, but it is the most *destructive* mode and
   §7 already prescribed the fix (exclude specular pixels from α fit) — never
   implemented. Bonus finding: `cue_white`'s absolute L>0.85 band fires on
   *bare* Fitzpatrick-I skin (flat patch L=0.883 → cue 0.12) — the absolute
   thresholds fail at both ends of the tone range.

### 14.3 Test-coverage verdict

`tests/test_makeup_unmix.py` (7 tests) uses only gray/pale synthetics
(S=140/160/128 gray, white 245, one light-skin BGR (150,165,200)). **Zero dark
tones.** Of the plan's §5 S1–S5 gates: S1 ≈ `test_closed_form_alpha_disk`
(oracle S/M only — never exercises seeding), S3 exists but with a 0.35 bound vs
the plan's 0.05, S2/S4 (edge IoU, mole protect) absent. §13.5 item 4
("dark-skin residual regression test") confirmed missing — the suite could not
have answered this question; RESEARCH_PER_FACE_AND_P4's concern (2) confirmed.

### 14.4 Recommendation (specific, ordered)

The region-anchored closed-form strategy is **salvageable — the ill-posedness
is not the culprit; the absolute-threshold prior is.** No learned prior needed
for v1. Fix set, in order of leverage:

1. **Measure `cue_resid` in optical-density space, not linear RGB** —
   `|log I − log I_hat|` (or divide the linear residual by max(local mean I,
   floor)). Makes the threshold reflectance-invariant by construction; directly
   removes the monotone tone bias in failure modes 1 and 2. ~5 LOC in
   `estimate_makeup_alpha`.
2. **Make chroma/L cues relative to per-face bare-skin statistics** —
   e.g. fire on `C > median_skin_C + k·MAD` computed over the low-hint region,
   same for L (fixes both the dark-skin non-detection and the Fitzpatrick-I
   `cue_white` misfire). This is exactly the "Fitzpatrick-aware thresholds
   (R14 ITA)" mitigation §8 already names; R14's ITA classifier exists per
   RESEARCH_PER_FACE_AND_P4 §B ("✅ lib, ❌ residual thresholds").
3. **Exclude specular pixels from hint and M-seeding** (top-percentile L
   within skin, or R12's specular map) — kills the worst destructive mode for
   every tone.
4. **Guard the degenerate seed:** in `unmix_makeup`, if `‖M−S‖` is small,
   fall back to α_init instead of running closed-form/IRLS on a near-zero
   denominator (currently produces junk that then re-seeds M at `α>0.3`).
5. **Promote the probe to pytest:** parametrize the existing synthetic tests
   over the 6-tone palette; gates = bare-skin e2e α < 0.05 *for all tones* and
   foundation-disk recovery error < 0.15 *for all tones*. Land the tests
   red-flagged (xfail) with the current solver, flip to strict with fix 1–4.

**Not done in this audit, deliberately:** fixes 1–4 change the rendered output
of a shipped, recipe-reachable feature — per ground rule (§11.6 / MASTER_PLAN
rule 7 + visual-QA discipline) they must land with the tone-parametrized tests
*and* a visual QA pass (bare dark-skin control portrait must stay untouched),
not as a drive-by threshold swap. Until then, treat `makeup_coverage_even` as
**light-skin-only and specular-unsafe**; do not enable it by default in any
recipe.

---

## 15. Second instance of the same bug class, fixed: `specular.py` intensity gate (2026-07-15)

The §14 audit's underlying pattern — **an absolute luminance/reflectance
threshold applied to a signal whose baseline scales with skin tone** — is not
unique to `makeup_unmix.py`. A quick verification pass across the other
R9-R13 skin/chromophore modules (prompted directly by the §14 finding, not a
full re-audit) found the same bug class live in `retouch/specular.py`'s R12
specular-finish path (`extract_specular`, used by `skin.py`'s
`apply_specular_finish`, i.e. the `specular_finish_strength` /
`specular_recolor` / dewy / glass_skin / powder finish modes).

**Finding:** `extract_specular`'s highlight gate was
`intensity_gate = clip((I - 170.0) / 70.0, 0, 1)` — a fixed absolute max-channel
level. Since a specular reflection is additive on top of the diffuse base
(dichromatic model, `I = I_diffuse + I_specular`), the same physical highlight
crosses a fixed absolute line far later on darker skin. Measured with a
same-swatch, same-boost sweep (`+15` to `+120` additive highlight) across
Fitzpatrick I–VI synthetic swatches, comparing the shipped gate before/after:

| tone | +15 before→after | +45 before→after | +90 before→after | +120 before→after |
|---|---|---|---|---|
| I   | 128 → 0*  | 180 → 86  | 242 → 118 | 242 → 118 |
| III | 0 → 0     | 25 → 20   | 115 → 115 | 183 → 183 |
| VI  | 0 → 0     | **0 → 11**    | **0 → 62**    | **39 → 91**    |

(*I's +15 "before" value looks anomalously high relative to its own +45 column
because the old gate's fixed 170 threshold sat unusually close to that
particular swatch's baseline; not the point of the comparison — the point is
the VI column: **the old gate read exactly 0.000 for Fitzpatrick V–VI at every
boost up to +90**, i.e. the R12 specular finish was silently inert on dark
skin for realistic highlight strengths, while lighter tones already showed
strong detection at the same physical boost.)

**Reachability at time of fix:** engine-callable via `apply_specular_finish`
(`skin.py:1616-1622`) and CLI/GUI-reachable via `specular_finish_strength` /
`specular_recolor` / `mode` params, but **no shipped recipe currently sets
`specular_finish_strength` above its no-op-equivalent default or enables
`specular_recolor`/non-matte modes** (recipes reference an unrelated
`specular_bloom` lens-glow param instead) — same "latent, not actively
shipping harm" status as the P4 finding, not worse.

**Fix landed** (`retouch/specular.py::extract_specular`): replaced the
absolute gate with a margin-above-baseline gate, `baseline` = median
max-channel intensity over `skin_mask` if provided, else over the whole crop
(the shipped `skin.py` call site passes no mask, so the whole-crop median is
the load-bearing path in production today — it's noisier than a mask-scoped
median since it can include hair/background/eyes, but skin dominates a
typical face crop so it stays stable in practice, confirmed by the no-op
tests below). `skin_mask` is an additive, backward-compatible optional
parameter.

**Verified:**
- No-op invariant preserved: realistic post-shine-removal skin (flat + small
  natural noise) reads < 0.5 mean specular at all six tones (was, and remains,
  the basis for `apply_specular_finish`'s documented byte-identical default).
- Detection floor no longer collapses on dark skin: a +90 boost (moderate
  highlight) now reads > 20 mean specular intensity at every tone, including
  Fitzpatrick VI (previously exactly 0.0).
- Visual check: same +90 highlight rendered through `render_finish(..., "dewy",
  0.6)` across all six tones — dewy bloom now visibly appears at every tone;
  flat/no-highlight swatches show no false-positive bloom introduced by the
  fix, at any tone.
- All 8 pre-existing `tests/test_specular_finish.py` tests still pass
  unmodified; 13 new tests added
  (`TestExtractSpecularToneInvariance`) parametrizing the flat-skin no-op
  check and the +90-boost detection check over Fitzpatrick I–VI, plus a
  direct VI regression test. 21/21 pass.

**Known gap, explicitly not fixed here (scope discipline):** the mask-aware
baseline (median over `regions.skin`, passed from `skin.py`) is strictly
better than the whole-crop fallback and is one line away — `skin.py:1621`'s
`extract_specular(work)` call would need `skin_mask=regions.skin` added. Left
as a follow-up since it touches a call site outside `specular.py` itself.

**Naming the pattern for future audits:** watch for `> <absolute constant>`,
`< <absolute constant>`, or `smoothstep(<abs>, <abs>, ...)` gates applied to
raw luminance/intensity/reflectance channels anywhere a signal is expected to
scale with skin tone (melanin/reflectance-dependent). Two confirmed instances
so far: `makeup_unmix.py`'s multi-cue α (§14) and `specular.py`'s highlight
gate (this section). Grep starting point: `grep -n "[<>]=\? *[0-9]\+\.[0-9]\|smoothstep(0\." retouch/*.py` — cross-check any hit against whether the input is skin-tone-dependent before trusting it. Candidates not yet
checked: other R9-R13 modules (`intrinsic.py`, `skin_chromophore.py`) had a
quick threshold grep during this pass with no clear hits, but were not put
through the same empirical tone-sweep as the two confirmed instances — treat
as unverified, not clean.

---

## 16. Targeted sweep for the same anti-pattern (2026-07-15) — third instance found + fixed, fourth found + documented (not fixed)

Following §14/§15, a targeted grep + empirical-check sweep was run across
other skin/face-adjacent modules for the same anti-pattern: an absolute
luminance/chroma/intensity threshold gating a signal whose baseline scales
with skin tone. Grep targets: `retouch/skin.py`, `retouch/eyes.py`,
`retouch/eye_enhancement.py`, `retouch/hair.py`, `retouch/hairwork.py`,
`retouch/lips.py`, `retouch/undereye.py`, `retouch/skin_chromophore.py`,
`retouch/intrinsic.py`. Method: grep for `smoothstep(<abs>, <abs>, ...)`,
`clip((X - <abs>) / <abs>, ...)`, and bare `> <abs>` / `< <abs>` comparisons
against luminance/chroma channels, then a quick reasoned check per hit, with
a full empirical tone-sweep only where the reasoned check flagged a plausible
real effect.

### 16.1 Third instance, FIXED: `retouch/lips.py` `_add_lip_gloss` specular floor

**Finding:** `specular_mask = ((l_chan > specular_thresh) & (l_chan > 130.0) & ...)`
— `specular_thresh` (`lip_median + 1.5*lip_std`) is already a correct,
per-face adaptive threshold, but it was ANDed with a redundant **absolute**
floor of `130.0`. For light lips (median L ~146+) the floor never binds. For
darker-toned lips the floor becomes the *sole* binding constraint, silently
vetoing detections the adaptive math already correctly identified.

**Measured (same +25 L-channel gloss highlight, real API, 4-tone swatch set):**

| tone | max delta in highlight spot (0 = gloss not applied) |
|---|---|
| I   | 45.0 |
| III | 15.0 |
| V   | **0.0** |
| VI  | **0.0** |

Lip gloss finish was silently inert on Fitzpatrick V–VI lips for a
realistic highlight strength.

**Fix landed** (`retouch/lips.py`): replaced the absolute `130.0` floor with
`rel_floor = lip_median + 12.0` — a margin above the lip's own median,
matching the specular.py fix pattern. The `+12.0` margin (not `+8.0`, tried
first) was chosen empirically: at `+8.0` a Fitzpatrick-VI-specific false-
positive noise sensitivity appeared (RGB→LAB's cube-root nonlinearity
stretches ±3 sensor-noise-σ excursions slightly more at low absolute
luminance — measured max noise excursion 10.0 vs 9.0 for lighter tones on a
flat patch); `+12.0` clears that false-positive rate to 0/20 seeds at every
tone while still detecting the full +25 highlight (197/197 spot pixels) at
every tone.

**Verified:**
- Detection: same +25 highlight now reads > 20 max delta at every tone
  (I=45, III=68, V=84, VI=93 post-fix) — was 0.0 at V/VI pre-fix.
- No-op: flat lips (natural noise only, no real highlight) trigger 0/20
  seeds at every tone, post-fix (pre-fix: I was 10/10, since I's median
  already exceeded the old absolute 130 floor and the floor did nothing to
  suppress noise there — the old code was *simultaneously* under-detecting
  real highlights on dark lips and over-triggering on noise for light lips).
- All pre-existing `tests/test_lips.py` (`TestAddLipGloss`, 4 tests) pass
  unmodified. 9 new tests added (`TestLipGlossToneInvariance`): per-tone
  detection check, per-tone no-op check (20 seeds each), direct VI
  regression. Full file: 35/35 pass.

### 16.2 Fourth instance, FOUND, documented, NOT fixed: `retouch/skin.py` `SkinProcessor.whiten()` shadow-protection gate

**Finding:** `whiten()` (the "Adaptive Rosy Foundation" skin whitening /
rosy-tone op) gates its lift with `shadow_protection = clip((l_val - 80.0) /
40.0, 0, 1)` (positive-strength branch) and `shadow_decay = clip((l_val -
10.0) / 20.0, 0, 1)` (negative-strength/shadow-deepen branch) — both absolute
L thresholds on the skin's own luminance channel, same bug shape as §14/§15/
16.1.

**Measured** (flat noisy skin swatches, `strength=30/60/90`, `tone="rosy"`,
`hue_stable=False`, mean abs delta over the whole image):

| tone | strength=30 | strength=60 | strength=90 |
|---|---|---|---|
| I   | 2.33 | 5.07 | 7.90 |
| III | 5.28 | 10.30 | 15.50 |
| V   | 3.34 | 7.05 | 10.65 |
| **VI**  | **1.06** | **1.06** | **1.06** |

VI's effect is flat regardless of slider strength — the `shadow_protection`
mask sits near 0 across nearly the whole face for L in the 30-60 range
typical of Fitzpatrick VI skin, capping the whitening effect near-zero no
matter how far the user pushes the strength slider. (Caveat, flagged
honestly: this was measured on a flat synthetic patch, which — per the same
lesson learned from the Fitzpatrick-I clipping artifact in §14 — likely
*overstates* "fully inert": a real face has shading, so some regions of a
real Fitzpatrick-VI face will sit above L=80 and get partial effect. The
*direction* of the bug — weaker/inconsistent effect on darker skin at the
same slider setting — is not in doubt; the exact magnitude on real photos is
unmeasured.)

**Why this was NOT fixed in this pass, unlike the other three:**

1. **Reachability/severity is categorically different.** `whiten` is
   `params.py`'s `skin.rosy` recipe key with **default=10** (not 0) and is
   called unconditionally in the main per-face pipeline
   (`perf_optimizations.py:550`) and the body-skin path (`engine.py:3465`).
   Unlike the P4 (§14) and specular (§15) findings — both reachable but not
   activated by any shipped recipe — `whiten` runs, at some nonzero
   strength, on every processed image today. That makes it higher-impact to
   fix *and* higher-risk to change: a relative-baseline rewrite would alter
   output for the existing, already-tuned light-skin userbase, not just
   extend correctness to dark skin.
2. **Wide, uncharacterized blast radius.** `grep -rln "whiten" tests/`
   returns 18+ test files (`test_skin.py`, `test_whiten_hue_stable.py`,
   `test_recipe_integration.py`, `test_c4_finish_pack.py`,
   `test_qa_backoff.py`, golden/integration suites, etc.) — none of which
   were individually triaged for what they assert about current absolute
   behavior. A same-day fix here would be a design task (re-anchor to a
   face-median baseline while reproducing today's light-skin behavior
   within tolerance), not a 5-line threshold swap.
3. **Prior history of exactly this function being delicate.** `git log`
   shows a previous regression fix on this same area (`1640eeb` "Fix
   skin.py: equalize blend strength scaling") and the project's own memory
   notes flag an "equalize pale-face regression" as a known landmine
   requiring careful bisection, not a drive-by change.

**Recommended next step, if/when this is picked up:** (a) triage the ~18
test files to establish which ones assert exact absolute-behavior numbers
vs. qualitative direction-only checks; (b) design the relative baseline to
reproduce current light-skin output within tolerance (e.g. anchor
`shadow_protection`'s ramp to something like `face_median_L - k` rather than
a fixed `80.0`, so a "typically lit light face" reproduces today's near-1.0
protection while a "typically lit dark face" gets equivalent relative
protection instead of a near-zero absolute one); (c) re-run the tone-sweep
methodology from this section plus the full existing whiten-adjacent test
suite before landing.

### 16.3 Checked, no bug: other candidates

- **`retouch/eyes.py:206`, `bright_sclera = (L > 100.0)`** (sclera
  whitening gate). Not flagged as a skin-tone bug: sclera (eye white) color
  does not vary by Fitzpatrick tone the way skin/lip reflectance does — this
  gate operates on the sclera's own luminance, not a signal that scales with
  the subject's skin melanin. Checked, no bug.
- **`retouch/eyes.py:117` / iris catchlight specular boost, `clip((l_chan -
  220.0) / 20.0, ...)`** — operates on the iris/catchlight region, not skin;
  iris/catchlight brightness is dominated by the physical light source
  reflection (near-white, ~220+), not skin melanin. Lower-priority than the
  skin/lip instances; not empirically tone-swept in this pass, but reasoned
  as low-risk since it's gated to `iris_m`, not a skin-reflectance-scaled
  region.
- **`retouch/hair.py:134`, `thresh = max(mean_val + std_val*0.8, 120.0)`**
  — same *shape* as the lips.py bug (adaptive threshold ANDed/maxed with an
  absolute floor), but the tone axis here is **hair color**, not
  Fitzpatrick skin tone — hair color varies independently of skin melanin
  (e.g. dark hair on light skin, light hair on dark skin are both common),
  so this is not the same fairness-relevant bug class. Flagged as a
  legitimate but lower-priority candidate for a general "absolute floor
  defeats adaptive threshold" cleanup pass, not a skin-tone audit item.
- **`retouch/skin.py:318`, shadow-band gate `(L < local_median - 15.0) & (L
  > 40.0) & (L < 200.0)`** — the `local_median - 15.0` term is already
  relative; the `40.0`/`200.0` bounds are a sanity clamp (reject near-black
  and near-white outliers) rather than the primary detector. Reasoned as
  low-risk, not empirically tone-swept.
- **`retouch/intrinsic.py`, `retouch/skin_chromophore.py`** — grepped for
  the same shapes; no absolute luminance/chroma gate comparable to the
  above found. Not exhaustively tone-swept (no candidate strong enough to
  justify it), but no clear hit either — treat as checked, not confirmed
  clean at the same confidence level as 16.3's sclera/iris items above.

### 16.4 Sweep verdict

Three confirmed instances of the pattern now exist across the codebase
(`makeup_unmix.py` §14, `specular.py` §15, `lips.py` §16.1), two fixed
(specular.py, lips.py) and one parked pending a fuller solver rework
(makeup_unmix.py). A fourth, higher-severity instance was found in
`skin.py::whiten()` (§16.2) but deliberately left unfixed pending a properly
scoped design pass, given it is an always-on-by-default, wide-blast-radius
path rather than a latent/parked feature. This is now a confirmed *pattern*,
not a one-off — worth a standing check ("does this absolute threshold's
trigger point shift with skin tone, and if so is that intentional") whenever
touching luminance/chroma/reflectance-gated code in any skin/lip/lash-
adjacent module.
