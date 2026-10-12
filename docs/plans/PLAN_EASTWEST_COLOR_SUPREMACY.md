# Tier C Plan — East ✕ West Skin & Color Supremacy (Stages C1–C6)

**Date:** 2026-07-02
**Type:** Research + implementation documentation (NO code — per owner instruction)
**Question:** which algorithms — combining the Chinese/Japanese/Korean retouch-and-color traditions with Western color science — would make skin and color processing genuinely more powerful than Photoshop and 像素蛋糕 (PixCake)?
**Builds on:** `PLAN_SKIN_PRO.md` (S1–S6), `PLAN_SKIN_ADVANTAGE.md` (A1–A5), `MASTER_PLAN.md`. This doc adds the **color-science layer** those plans don't cover; it deliberately does NOT re-plan anything already staged there.

---

## Part 1 — Research: what the two traditions actually do

### 1a. The Eastern stack (影楼 / PixCake / JP–KR aesthetics), decomposed algorithmically

| Practice | What it really is, algorithmically | Do we have it? |
|---|---|---|
| **中性灰 / 双曲线** (neutral-gray / dual-curve D&B) — the core of every high-end CN studio retouch and PixCake's flagship "像素级中性灰磨皮" | Local luminance add/subtract painted on a 50%-gray soft-light layer. Two distinct frequency jobs hide inside it: **(i)** evening blotches at the pore-cluster↔shading band, **(ii)** *reshaping facial form light/shadow* (立体感 — T-zone highlight, cheekbone shadow, nose ridge). PixCake automates both with a generative network that "reshapes skin light and shadow while preserving pores" | (i) = **S2** (planned). (ii) = only crude fixed-zone `dodge_burn` (`skin.py:178`) → **C2** |
| **肤色统一** (skin-tone unification) | Pull every skin pixel's hue/chroma toward a single target *along constant-hue lines* in a Lab-family space, luminance untouched — different math from our `equalize` (median a/b pull, which desaturates) and `unify_tone` (anime-strength) | Partially → **C1** |
| **记忆色 / preferred color** (the JP camera/printer tradition — the actual science behind Fuji skin) | Skin is corrected toward the *preferred* reproduction locus (slightly more pink-orange, lighter, less chromatic than real skin), moving along a constant-preferred-hue line in an appearance space (CIECAM-class), never toward the colorimetric truth | No → **C1** |
| **日系透明感** ("transparency") | Luminance-priority skin: raise L without raising C (hue-stable), lifted-black fade toe, slight cyan drift in highlights, *low mid-frequency contrast + intact micro-texture* (the "clean but not plastic" signature) | Pieces exist (whiten, fade in some presets) but not as a coordinated, hue-safe op → **C4** |
| **空气感** ("airiness") | Highlight-scoped haze/negative-clarity: soft glow injected only above an L threshold, plus slight desaturation of far background | `glow`/orton is global, not L-scoped → **C4** |
| **韩系牛奶皮 / 水光肌** (milk/water-glow skin) | High mean L, *low chroma variance* (uniformity, not just brightness), plus a **preserved-and-shaped specular** — highlights compressed then re-added as tight gloss. Removing shine then re-lighting is exactly S4 + `specular_bloom`; the missing piece is the chroma-variance target | S4 + bloom + **C1's variance target** |
| **三庭五眼 structural analysis** | Facial-proportion measurement driving where D&B/contour lands (not how much) | Landmarks exist; used for zones, not proportions → folded into **C2** |

### 1b. The Western stack, decomposed

| Practice | Algorithmic core | Do we have it? |
|---|---|---|
| Frequency separation, micro D&B, texture preservation | (shipped / S2) | Yes |
| **Film-density color** (Dehancer/Filmbox-class, beyond LUTs) | *Subtractive* model: RGB→density, per-dye density curves, **channel-crosstalk matrix** (the thing LUTs bake in but can't expose), halation (red-leaking highlight bloom), gate-weave-free grain placed in density domain | LUT/Fuji presets only — fixed, not parametric → **C3** |
| **Hue-preserving tone mapping** | Per-channel curves (every PS curve, every LUT) skew hue as they roll off — the "notorious six" drift; modern renderers (ACES 2.0-class, AgX) tone-map luminance and re-attach chroma with controlled, *chosen* skew | No — our curves are per-channel → **C3/C6** |
| **Perceptually uniform ops (OKLab/OKLCh, CAM16-UCS)** | Chroma/hue edits that behave identically across the L range; fixes CIELAB's blue-shift and skin-hue bend | All our color ops are CIELAB/LCH(ab) → **C6** |
| **Color harmonization** (Cohen-Or harmonic templates) | Measure hue histogram → snap background hues to a harmonic template (analogous/complementary…) *anchored on skin hue*, subject protected | No → **C5** |
| **ABPN blend-pyramid architecture** (CVPR 2022; basis of ModelScope's skin retouch, PixCake-class) | Retouch computed at low res; what gets upsampled is not the image but **blend layers** applied to the native-res original — texture is physically incapable of being lost | **Architecture already adopted** by F8/P2 ("downsample-compute / full-res-apply"). Validation, not new work — noted in Part 3 |

### 1c. The synthesis thesis

Photoshop gives manual versions of the Western column. PixCake automates the Eastern column but: cloud/credit-locked, face-first (weak body skin), fixed aesthetics (its looks are its looks), and no exposed color *engine* — you get presets, not parameters. **Nobody ships the two columns unified in one parametric, local, scriptable pipeline.** The engine already has the skeleton (LAB/LCH plumbing, masks, frequency bands, recipes-as-code). Tier C is the color-science flesh.

---

## Part 2 — Implementation documentation (Stages C1–C6)

### Stage C1 — Preferred-Skin-Color Core: hue-line unification + memory-color targeting (~1.5 weeks) 🔴 the color counterpart of S2

The single highest-value color upgrade. Replaces "pull toward median" with "pull toward *preferred*".

1. **Skin color state = (h̄, C̄, L̄, σ_C)** measured in LCh over the skin mask (per face + body when S1 lands). σ_C — chroma variance — is the 牛奶皮 uniformity number and becomes a first-class metric (add to A1's benchmark alongside the S2 blotch metric).
2. **Preferred-locus table** (`retouch/color_science.py`, new): per skin-tone class (measured h̄/L̄ decides the class — fair/tan/deep — NOT ethnicity guessing), a target hue line + chroma window from the preferred-reproduction literature: skin hue ≈ 40–50° in LCh(ab), preferred slightly lighter & less chromatic than measured, hue moved toward the pink-orange side. Ship as data (JSON), user-overridable — a recipe can carry its own locus (`moonlight_porcelain` wants a cooler, paler locus than `natural`).
3. **Correction = constant-hue-line pull:** each skin pixel's (a,b) is projected toward the target hue line — *rotate hue, then compress chroma spread toward C̄_target* — leaving L to S2. This is 肤色统一 done right: kills the olive/red patchwork WITHOUT the global desaturation `equalize`'s median-pull causes. Strength-bounded (Δh ≤ 8°, ΔC ≤ 12).
4. **Luminance-priority whitening** (the 透明感 fix to `whiten`): raising L in CIELAB while keeping (a,b) *increases* perceived chroma shift on skin (chalkiness at high strength). Do the raise in LCh holding C constant and h locked — hue-stable brightening. Refactor path for `whiten`'s three tone directions; keep old behavior as the default until visually A/B'd.
5. Params: `skin_hue_unify` (0–100), `skin_chroma_even` (0–100, the σ_C compressor), `skin_locus` (recipe-level target override). Composition order: **C1 runs where `equalize` runs today** (after S2 slot, before whiten); `equalize` stays for global work, defaults lowered once C1 exists.

**QA:** freckle/mole chroma preserved (protected-feature mask from A2); tattoo/body-paint exclusion (S1's high-chroma zone); lips excluded; ΔE bound per pixel; hue histogram of skin post-op must be unimodal (the objective "unified" test); dark-skin corpus mandatory (locus table per class — the failure mode of every CN tool is fair-skin bias, and beating PixCake on deep skin tones is a real win).

**Files:** `retouch/color_science.py` (new), `retouch/skin.py`, `retouch/params.py`, `retouch/recipes.py`, `benchmark.py` (σ_C metric), tests.

### Stage C2 — Structural Light-Shadow (自动中性灰 v2 / 立体感) (~1.5 weeks) 🔴 the PixCake flagship, classically

S2 evens the *blotch* band; C2 sculpts the *form* band — together they equal the full manual 中性灰 workflow (and PixCake's headline feature).

1. **Face shading model:** approximate a normal/shading map from the 478 landmarks (coarse face geometry is enough — fit the canonical MediaPipe face mesh, take per-triangle normals, Lambert-shade with a light direction *estimated from the existing L distribution*, reusing the relight stage's light-direction logic).
2. **Target shading = measured shading, smoothed and slightly exaggerated** along the model: correction = `(target_shading − measured_lowband_L) × strength × skin_mask`, applied to the low band only (form frequency; S2's band and the pore band untouched). This deepens cheekbone/nose-ridge/jaw modeling and *removes lighting accidents* (flat frontal flash faces gain structure back) — what retouchers spend hours painting.
3. **三庭五眼-informed placement:** proportion measurements (face thirds, eye spacing from landmarks) scale the highlight/shadow *placement* per face rather than fixed zones — replaces `dodge_burn`'s hardcoded regions.
4. Params: `sculpt` (0–100, default 0; portrait recipes ~20), `sculpt_light_angle` (auto by default). Runs adjacent to `relight` (they share the light estimate; document order in `PIPELINE_FLOW.md`: C2 shapes reflectance, relight shapes illumination).

**QA:** yaw gating like T2 makeup (fade beyond ~35°); no double-shadowing with relight active (compose test at both = 100); flat-field invariance; identity preservation (landmark positions unchanged — this is shading, not warping).

**Files:** `retouch/skin.py` or new `retouch/sculpt.py`, `retouch/relight.py` (shared light estimate), `retouch/params.py`, `gui.py`, tests.

### Stage C3 — Parametric Film-Density Engine (~2 weeks) 🟠 beyond LUTs

Make the Fuji-class looks *parametric* instead of baked, using the subtractive model:

1. `retouch/film.py` (new): RGB → log-density; per-dye **density curves** (shoulder/toe per channel, the S-curve that IS "film contrast"); **3×3 crosstalk matrix** (dye impurity — the warmth-in-shadows / cyan-skies behavior LUTs bake in); density-domain **grain** (σ scales inversely with density = shadows grainier, like real stock); **halation** (threshold highlights → red-orange-weighted bloom, distinct from `glow`).
2. **Hue-preserving master tone-map option:** tone-map on a luminance norm and re-attach chroma with a single controllable "skew" parameter (0 = fully hue-preserving, 1 = classic per-channel skew). This one slider spans "digital-clean" ↔ "filmic-drift" — a control Photoshop simply does not have.
3. Existing Fuji presets get re-expressed as parameter sets over this engine where feasible (keep the LUT path; film engine is a new grading block, `"film": {...}` recipe key). Immediate payoff: infinite in-between stocks, and F6 (look-from-reference) gains a much better fitting target than raw curves.

**QA:** round-trip neutrality at all-zero params (byte-identical); grain determinism (seeded); halation energy bound; A/B against the shipped Fuji LUTs (the parametric re-expression should land within ΔE ~3 of the LUT on the corpus — proves the model).

**Files:** `retouch/film.py` (new), `retouch/grading.py` (block hook), `retouch/params.py`, `retouch/recipes.py`, `gui.py`, tests. **Wants F1 (float32) underneath** — density math in uint8 bands.

### Stage C4 — 透明感 / 空气感 Finish Pack (~1 week) 🟠 quick wins, high demand

The coordinated JP-look primitives, each ~30–60 lines on existing machinery:

1. **`fade_toe`:** lifted-black with a *hue-locked* toe (fade in L only; classic RGB fade shifts shadows blue-green unintentionally — offer both, intentional is a choice).
2. **`highlight_drift`:** bounded hue rotation of highlights toward cyan (the 透明感 signature), scoped by L, skin protected above a chroma floor via C1's skin state.
3. **`airy_haze`:** glow computed only from pixels above L threshold, screen-blended with distance falloff (uses person_mask: background gets more air than subject).
4. **`clarity_split`:** negative clarity on the form band + positive micro-contrast on the texture band simultaneously — the "soft but detailed" JP portrait finish; trivially composable now that bands are shared infrastructure.
5. One recipe proves the pack: `jp_transparent_v1` (fade_toe + highlight_drift + airy_haze + C1 luminance-priority whiten + low S2).

**QA:** banding check on fade toe (F11 detector when it lands); skin-hue invariance under highlight_drift; texture-energy floor under clarity_split.

**Files:** `retouch/grading.py`, `retouch/params.py`, `retouch/recipes.py`, tests.

### Stage C5 — Skin-Anchored Color Harmonization (~1 week) 🟡

Auto-grade the *environment* to flatter the (now-corrected) skin: measure the post-C1 skin hue anchor → choose/score a harmonic hue template (analogous, complementary, split-complementary) → pull background hues (person_mask-excluded, chroma-weighted, bounded rotation ≤20°) toward the nearest template arm. Param `harmonize_background` (0–100) + template override. This is the algorithmic version of what colorists do by hand and no retouch tool automates. Rides T1's background stage if it exists; standalone Phase-3 op otherwise.

**QA:** subject untouched (containment); no hue tearing at mask boundary (feather + gradient continuity); template-choice determinism per image.

### Stage C6 — Appearance-Space Substrate (OKLab/CAM16-UCS) (~1 week, infrastructural) 🟡

Add `oklab`/`oklch` converters beside the LAB helpers in `utils.py`; migrate **C1's skin math** to OKLCh from day one (skin-hue lines are straighter there — CIELAB bends skin hue with L, which is exactly the error C1 can't afford); other ops migrate opportunistically, never as a big-bang (each migration behind an A/B test). Optional later: CAM16 viewing-condition parameter for print-vs-screen output targets. **Do C6's converter work FIRST inside C1's implementation** — it's ~150 lines and C1 is its first consumer; listed separately only so the substrate is documented as reusable.

---

## Part 3 — Architecture validation from the research (no action needed)

**ABPN (CVPR 2022)** — the architecture behind ModelScope/PixCake-class ultra-high-res retouch — computes edits at low resolution and upsamples *blend layers*, not results, precisely to keep native texture. This is independently the same conclusion F8/P2 reached ("downsample-compute / full-res-apply", detail reinjection). Treat this as external confirmation that the planned F8 path is the industry-correct one; if A4's neural boosters ever happen, ABPN's blend-layer formulation is the integration pattern to copy.

---

## Sequencing & master placement (adopted 2026-07-02 — reflected in `MASTER_PLAN.md`)

```
C1+C6 core (1.5wk) ──► C2 sculpt (1.5wk) ──► C4 finish pack (1wk) ──► C3 film engine (2wk, wants F1) ──► C5 harmonize (1wk)
```

- **C1 belongs in Phase 2 ("Skin Supremacy") with S2/S3** — S2 evens luminance, S3 evens local color casts, C1 sets the *target* color; they are three axes of one job and should tune together (A2's competitive A/B covers all three).
- C2 right after — with S2+C1+C2 shipped, the engine covers the complete manual 中性灰 workflow, i.e., PixCake's headline claim, locally and parametrically.
- C4 anytime after C1 (cheap, visible); C3 after F1; C5 floats (pairs naturally with T1).
- No changes to Phase 0/1 (P1 ✅ done, A1, F8+P2, F1, F11 unchanged).

## Verification (end-to-end)
1. A1 competitive corpus re-scored after C1/C2 — including a **PixCake trial column** (add it to A1's tool list; it's the East-bar the way Evoto is the West-bar).
2. C1 acceptance: patchy-skin corpus → unimodal skin-hue histogram, σ_C halved, freckles/tattoos untouched, deep-skin class visually approved.
3. C2 acceptance: flat-flash portrait regains modeling; blind A/B vs manual 中性灰 edit on ≥5/10.
4. C3 acceptance: parametric re-expression of one Fuji preset within ΔE ~3 of its LUT.
5. Full pytest per stage; new params under the dead-key guard pattern; F11 detectors green once available.

## Open items
- PixCake trial access for A1 benchmarking (CN-market product — verify trial availability outside CN app stores).
- Preferred-locus table needs a small perceptual validation pass (owner eyes on 3 skin-tone classes × 3 recipes) before defaults ship.
- `anime_crystal_void` dead keys — unchanged, fix vehicle remains T1.

## Sources
- [像素蛋糕 PixCake — official](https://www.pixcakeai.com/) · [feature description (中性灰磨皮 via generative network)](https://www.onetts.com/pixcake) · [PixCake business coverage](https://news.qq.com/rain/a/20250716A09B8B00)
- [中性灰与双曲线磨皮原理 — 知乎](https://zhuanlan.zhihu.com/p/60750232) · [高低频/中性灰/双曲线解析 — 知乎](https://zhuanlan.zhihu.com/p/40304487) · [全自动中性灰/双曲线 — 知乎](https://zhuanlan.zhihu.com/p/36724024) · [影楼精修-中性灰磨皮算法解析 — CSDN](https://blog.csdn.net/Trent1985/article/details/147400225) · [影楼精修-肤色统一算法解析 — CSDN](https://blog.csdn.net/Trent1985/article/details/148014647)
- [Preferred skin tones under different CCTs — Vision Research](https://www.sciencedirect.com/science/article/pii/S0042698922000669) · [A New Method for Skin Color Enhancement](https://www.researchgate.net/publication/258713114_A_New_Method_for_Skin_Color_Enhancement) · [Skin tone memory color study](https://www.researchgate.net/publication/368730913_Understanding_Color_Memory_A_Study_of_Skin_Tone_Perception_in_Hue_Intensity_and_Chroma) · [Memory-color modification patent (constant-preferred-hue-line method)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8761505)
- [日系透明感讨论 — 知乎](https://www.zhihu.com/question/276529679) · [奶油肌调色 — 知乎](https://zhuanlan.zhihu.com/p/363965696) · [空气感调色教程 — bilibili](https://www.bilibili.com/video/BV1fvmsBHExq/)
- [ABPN: Adaptive Blend Pyramid Network, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Lei_ABPN_Adaptive_Blend_Pyramid_Network_for_Real-Time_Local_Retouching_of_CVPR_2022_paper.html) · [StyleRetoucher (GAN-prior retouching)](https://arxiv.org/pdf/2312.14389)
