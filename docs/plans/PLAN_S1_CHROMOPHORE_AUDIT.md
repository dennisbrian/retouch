# PLAN S1 — Melanin/Hemoglobin Chromophore Decomposition: Calibration Audit

**Status:** Research spike complete. Decision-ready.
**Date:** 2026-07-17
**Author:** Fable (S1 spike)
**Scope:** Audit only — no production code changed. Evidence: `scripts/spike_s1_chromophore_audit.py` (seeded, re-runnable), panels `test_output/spike_s1_*.png`.
**Modules audited:** `retouch/chromophore.py`, `retouch/skin_chromophore.py` (read-only; owned by R7/R10 agents).

---

## TL;DR / Recommendation

The decomposition is **a useful within-face redness/pigment heuristic, not a chromophore separation.** It is **two fixed log-ratios of the RGB channels** with no sRGB linearization, no white-balance handling, and a hard clamp at 0. It is **not** trustworthy as an absolute or cross-image melanin/hemoglobin estimate, and it is **not tone-invariant** — it violates CLAUDE.md's tone-fairness rule via an implicit clamp gate that zeros the hemoglobin signal for light-to-mid skin while leaving it live for dark skin.

**Verdict per consumer:**
- **Leave as-is:** `hemoglobin_guided_smooth` (needs only an edge guide), `undereye.attenuate_hemoglobin` (uses a tone-adaptive local baseline).
- **Targeted recalibration (specific, below):** `vein_attenuate` (magnitude-sensitive: raw hb × fixed gain 60), `blemish_vs_mole` / `mole_protect` (needs true mel-vs-hb attribution it does not deliver).
- **The one systemic fix worth doing regardless:** add sRGB linearization + a documented WB assumption, and replace the `clip(...,0,None)` floor with a tone-relative baseline before any absolute use.

Do **not** promote this to a "physical chromophore model" claim in any user-facing or downstream-contract text — the docstring's Beer–Lambert framing is physically void (see §1).

---

## 0. Production consumers (who depends on this, and on what property)

`decompose_chromophores` / `reconstruct_from_chromophores` are consumed by:

| Consumer | Call site | Product feature | Property it relies on |
|---|---|---|---|
| `hemoglobin_guided_smooth` | `skin.py:1846` (via `apply_hemoglobin_smooth`); recipe `skin.hemoglobin_smooth` (e.g. 0.18) | Skin smoothing that respects freckle/mole edges | hb map has **edges** at colour boundaries (guide only) |
| `blemish_vs_mole` → `mole_protect` | `skin.py:1883`; recipe `skin.mole_protect` (0.80 in a recipe) | Protect moles / heal only vascular blemishes | **True mel-vs-hb attribution** per spot |
| `vein_attenuate` | `skin.py:1918`; recipe `skin.vein_attenuate` (0.12) | Reduce blue-green subsurface veins | **Absolute hb magnitude** (raw hb × gain 60, `skin_chromophore.py:244`) |
| `undereye` `attenuate_hemoglobin` | `undereye.py:287,304` | Under-eye darkness/redness reduction | hb **relative to a local cheek-ring baseline** (median + MAD, `undereye.py:294-298`) |
| `makeup_unmix` | `makeup_unmix.py:126` (out of edit scope; separate agent) | Foundation/blush unmix + reconstruct round-trip | forward/inverse round-trip fidelity |

**Pruned false consumer:** `skin.py:redness_even` (`skin.py:1416`) does **not** use the decomposition — it is a LAB a*/b* Difference-of-Gaussians band-pass. Redness-evening is therefore *not* at risk from any chromophore finding here.

---

## 1. What model is it actually implementing?

**Claim (docstring, `chromophore.py:1-25`):** Tsumura/Stamatas log-RGB Beer–Lambert unmixing, `L = -log(rgb) = M·[mel,hb] - c`, solved by pinv of the blue-differenced 3×2 matrix `M`.

**What it actually is (verified numerically, spike §A, matches production to 2e-6):** the 3×2 pinv + blue-differencing collapses to **two fixed scalar log-ratios**:

```
_M       = [[0.1,-0.5],[0.2,0.25],[0.3,0.25]]      # chromophore.py:47-51
_M_PINV  = [[ 0, -10,  0 ],                         # chromophore.py:56
            [-1.3333, 2.6667, 0]]

melanin    = clip( 10·(Lb − Lg), 0 )        = clip( 10·log(G/B),  0 )
hemoglobin = clip( (−1.333·Lr + 2.667·Lg − 1.333·Lb), 0 )
           = clip( 1.333·(log R + log B) − 2.667·log G, 0 )   # weights [-1.333, 2.667, -1.333] on (Lr,Lg,Lb)
```
where `Lx = −log(x)`. **Melanin never consults the red channel** — it is purely `log(G/B)`. Hemoglobin is a green-vs-(red,blue) contrast.

**Magic numbers and their implied (and mostly unmet) assumptions:**
- `_K_MELANIN=[0.1,0.2,0.3]`, `_K_HEMOGLOBIN=[-0.5,0.25,0.25]` (`chromophore.py:47-48`): **hand-set, not fitted.** The docstring (`chromophore.py:40-46`) explicitly states the hemoglobin vector is *deliberately non-physical* to "correlate with perceived skin redness." So this is a redness heuristic by author's own admission.
- **No sRGB linearization** (`_bgr_to_log_rgb`, `chromophore.py:62-71`): `−log` is taken on gamma-encoded sRGB, not linear light. The Beer–Lambert framing is void from line 1. This is also the *mechanism* for tone-dependent behaviour: the gamma curve's local slope differs where light vs dark skin sit, so the log-ratios respond differently by tone (confirmed §2).
- **"Shared constant c cancels via blue subtraction" (`chromophore.py:16-18,94-97`)** holds only for a **scalar** exposure change. A real warm/cool illuminant is a **per-channel** gain = per-channel additive log offset that does **not** cancel → WB leaks straight into both maps (confirmed §2, headline failure).
- **`clip(...,0,None)` (`chromophore.py:104-105`)** discards the entire negative tail. Combined with the tone-dependent sign of the ratios, this becomes an *implicit absolute gate* that zeros hb for lighter skin — a tone-fairness violation in spirit even though no literal `I>170` threshold appears.

---

## 2. Tone-range validity & crosstalk (fairness-critical)

**Method (spike §C):** probes built from an **independent** skin model — Monk Skin Tone sRGB swatches (10-step light→dark ladder) × hemoglobin perturbation applied as a **Lab a\*** boost (0/10/25) × white-balance as per-channel gain (neutral / warm+ / cool+). Crucially the probes are **not** generated by `reconstruct_from_chromophores` — §B shows that doing so returns the identity (mel corr 1.0) because crosstalk is diagonal *by construction for the module's own basis*; that would be a tautology. The a\* reddening is a physically-motivated perturbation *outside* the module's basis, which is what exposes real behaviour.

### 2a. Crosstalk — reddening (hb↑) contaminates melanin heavily (neutral WB)

| Monk tone | ΔMelanin for Δa\*=+25 | ΔHemoglobin for Δa\*=+25 |
|---|---|---|
| 1 (light) | **−0.39** | +0.25 |
| 3 | **−0.80** | +0.16 |
| 6 | **−1.66** | +0.51 |
| 7 | **−2.60** | +0.98 |
| 8 | **−2.23** | +1.53 |
| 10 (dark) | **−1.50** | +2.90 |

**Finding:** the idealized "diagonal crosstalk" holds only for the module's own basis vector. A realistic redness change **drives the melanin map strongly downward** (up to −2.6). So on a genuinely flushed/rosy face the engine will *under-read* melanin — melanin and hemoglobin are entangled, not separated. This directly breaks `blemish_vs_mole`: a spot that is both pigmented and vascular will not cleanly split.

### 2b. White-balance confound — the headline failure (Δa\*=0, pigment fixed)

| Monk tone | mel neutral | mel warm+ | mel cool+ |
|---|---|---|---|
| 1 | 0.39 | **2.01** | 0.00 |
| 4 | 1.64 | **3.27** | 0.24 |
| 6 | 3.82 | **5.44** | 2.42 |
| 10 | 1.50 | **3.12** | 0.10 |

**Finding:** with pigment and blood held constant, a warm illuminant roughly **doubles to 5×** the melanin reading and a cool illuminant **crushes it toward 0**. Any absolute or cross-image melanin comparison is meaningless. (The invariant-to-reddening-but-not-to-WB signature is inverted here because the module's melanin ignores red entirely: it's `log(G/B)`, maximally exposed to the exact B and G gains a WB shift moves.)

### 2c. Tone stability — inconsistent per-tone operating point

Raw hemoglobin at neutral WB, Δa\*=0:

| tone | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hb | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.048 | **0.222** | 0.000 | **0.006** |

**Finding:** the `clip(...,0)` floor swallows the baseline hemoglobin signal for **8 of 10 swatches** (it reads exactly 0.0), and is non-zero only at tones 7–8. The module does **no** local-baseline subtraction, so hb here is the raw absolute per-pixel decomposition value, which lands slightly negative → clamped to 0 for most tones; where a swatch's hue sits relative to the hb=0 hyperplane is hue/tone-dependent and **erratic, not monotone** (note tones 9–10, the darkest, are back at ~0). The defensible claim is therefore: the absolute clamp floor zeroes baseline hb for most tones, so only *reddening above a non-uniform, tone-dependent floor* survives — the **operating point is not consistent across tones.** The same absolute hb value means different things across tones, and `vein_attenuate`'s fixed `gain=60` lands differently by tone. This is the CLAUDE.md tone-invariance concern, arising from an *implicit* clamp gate rather than an explicit threshold — the failure mode (inconsistent behaviour across Fitzpatrick/Monk levels) is exactly the one the rule exists to prevent.

---

## 3. Real-photo behaviour (visual — panels read, not just numeric)

Panels `test_output/spike_s1_DSCF455{0,1,2}.png` (orig | melanin=inferno | hemoglobin=jet), ≤1600px. **I read them.**

- **Hemoglobin map:** *within the face*, lips and cheeks correctly read hotter than surrounding skin — the **relative** ordering on a single uniformly-lit face is defensible (this is why the baseline-relative undereye consumer is safe). **But** the single hottest region in-frame is the **warm-lit hand** and warm background/floor lighting — the map lights up "warm skin under warm light" and warm ambient indiscriminately. It is a redness-vs-blue map contaminated by illuminant, not a blood-concentration map.
- **Melanin map:** essentially a **"not-blue / warm-yellowness" map.** Near-black over the (blue) wig and costume; only lights up on warm floor lighting, gold trim, and warm background. It does **not** track tan/pigment as such; it tracks `log(G/B)` = colour temperature.
- **Luminance leak:** `corr(mel,lum)` ≈ −0.03…−0.11, `corr(hb,lum)` ≈ −0.17…−0.21 (spike §D). Low direct luminance correlation is good, but the maps instead leak **chroma/white-balance**, which is worse for a chromophore claim: they are stable to brightness but swing with colour temperature.

**Generality caveat:** all three panels are one shoot (same subject, booth, blue costume, mixed warm-ambient lighting). The "melanin = `log(G/B)` warmth map" conclusion is established from the closed-form math (§1) and *corroborated* on this one scene, not proven across subjects/lighting; note "melanin is black over the blue costume" is partly tautological given melanin = `10·log(G/B)`. The within-face relative-redness finding is the robust visual claim.

**Plausibility verdict:** physically plausible *only* as a within-face relative redness ranking under uniform light. Implausible as any absolute or scene-robust chromophore estimate.

---

## 4. Downstream risk ranking (most → least at risk)

Discriminator: does the operator need **true attribution / absolute magnitude**, or just a **monotone within-face proxy**?

1. **`blemish_vs_mole` / `mole_protect` — HIGHEST RISK.** Needs real mel-vs-hb *attribution* to decide "protect (mole) vs heal (blemish)." §2a shows the two maps are entangled: reddening drags melanin down by up to −2.6, so a vascular blemish can leak into the melanin map (→ wrongly protected) or a pigmented mole with any redness can leak into hemoglobin (→ wrongly healed). This is the consumer whose product promise most directly depends on physics the model lacks. Recipe uses it at 0.80.
2. **`vein_attenuate` — HIGH RISK.** Multiplies **raw** hb by fixed `gain=60` (`skin_chromophore.py:244`). §2b/§2c show raw hb magnitude swings with WB and has a tone-dependent operating point, so the same recipe strength (0.12) removes a different amount of green/blue on a warm vs cool photo and on light vs dark skin. Visible over-/under-correction risk.
3. **`undereye.attenuate_hemoglobin` — MODERATE/LOW.** Uses hb **relative to a local cheek-ring median + MAD** (`undereye.py:294-298`) and restores source lightness (`undereye.py:314-317`). Tone- and illuminant-adaptive by construction — the within-face relative ordering (which §3 confirms is sound) is exactly what it consumes. Robust to the WB/clamp failures.
4. **`hemoglobin_guided_smooth` — LOWEST.** Only needs the guide to have **edges** at colour boundaries; absolute values and WB drift are irrelevant to an edge-aware guided filter. Fine as-is.

(`makeup_unmix` round-trip is a separate agent's audit; flag only that its reconstruction inherits the same non-physical basis.)

---

## 5. Recommendation: go / no-go

### Leave as-is
`hemoglobin_guided_smooth`, `undereye.attenuate_hemoglobin`. They consume the one property the model actually has (within-face relative structure). No change.

### Targeted recalibration (scoped, with predicted effect + test strategy)
Apply **one shared front-end fix** in `chromophore.py`, which de-risks the two magnitude/attribution consumers without a model rewrite:

1. **sRGB linearization before `−log`** (`_bgr_to_log_rgb`, `chromophore.py:62-71`): apply the sRGB→linear transfer, then `−log`. *Predicted effect:* removes the gamma-slope tone dependence behind §2c; hb operating point becomes more uniform across Monk tones. *Test:* re-run spike §2c; success = baseline hb no longer clamps to exactly 0 for tones 1–6 while non-zero for 7–8 (spread of baseline hb across tones shrinks materially).
2. **Replace the absolute `clip(...,0)` floor with a tone-relative baseline** for any consumer that reads magnitude: subtract the per-face skin-mask median (à la `specular.py`/`undereye.py`) *before* clamping, so "0" means "at this face's baseline," not "at a global sRGB constant." *Predicted effect:* neutralizes the inconsistent per-tone operating point for `vein_attenuate`. *Test:* spike §2c tone table on baseline-subtracted maps (success = spread of baseline hb across tones shrinks materially, per-tone baseline ≈ 0 by construction); `vein_attenuate` output delta becomes tone-stable.
3. **Document the WB assumption + optionally gray-world normalize** the crop before decomposition for `vein_attenuate` and `blemish_vs_mole`. *Predicted effect:* collapses the §2b 2–5× melanin swing. *Test:* spike §2b table on WB-normalized input → mel neutral/warm/cool columns converge.

These are calibration/front-end changes; the `_M` basis stays. Estimated blast radius: `chromophore.py` `_bgr_to_log_rgb` + a new `baseline`-aware helper; consumers opt in. Guard with a golden-value test on the closed-form ratios (spike §A) so the reduction identity is locked.

### Needs a stronger model (only if attribution is promised)
`blemish_vs_mole` / `mole_protect` is the sole consumer whose *product promise* needs genuine mel/hb attribution. The front-end fixes reduce but do **not** eliminate the §2a entanglement (two hand-set basis vectors cannot separate overlapping pigment+vascular spots). If mole-protect quality is a priority, scope a **fitted or physically-grounded basis** (linearized reflectance + a least-squares fit of `_M` to measured skin swatches, or the merged oxy/deoxy extinction triple the docstring already flags at `chromophore.py:46` — "ponytail: swap K_HEMOGLOBIN…"). This is a design task, not a constant tweak; recommend gating it behind whether mole-protect is on the roadmap. Interim: keep `mole_protect` conservative and lean on the compact-spot area heuristic (`_compact_spots`) rather than pure mel/hb attribution.

---

## Appendix — reproduce

```bash
python3 scripts/spike_s1_chromophore_audit.py
# prints §A closed-form check, §B circularity guard, §C crosstalk/tone matrices;
# writes test_output/spike_s1_DSCF455{0,1,2}.png (orig | melanin | hemoglobin)
```
Seed `20260717`. Independent probes (Monk swatches + Lab a\* reddening + per-channel WB gain) — never `reconstruct_from_chromophores` (see §B for why that would be a tautology).
