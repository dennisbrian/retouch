# P4 — Skin ↔ Makeup Unmixing (Research + Implementation Plan)

**Status:** 📋 RESEARCH LOCKED (design 2026-07-14) · **Parent:** `PLAN_SKIN_PROMAX.md` §P4 · MASTER_PLAN research backlog  
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
