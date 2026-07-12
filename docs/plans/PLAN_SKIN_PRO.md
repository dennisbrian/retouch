# Tier S Plan — Pro-Grade Skin Retouch (Stages S1–S6)

**Date:** 2026-07-02
**Type:** Research + implementation documentation (no code yet)
**Question:** what separates the current skin stack from high-end professional retouching (manual pro workflow, Retouch4me/Evoto-class automation), and how do we close it?

---

## Part 1 — Research: current skin stack inventory

Verified by code reading (`retouch/skin.py`, `frequency.py`, `blemish.py`, `undereye.py`, `engine.py`):

| Capability | Implementation | Grade |
|---|---|---|
| Frequency separation | 3-band (low/mid/high), bilateral+gaussian hybrid, `mid_reduction`, `texture_opacity` | Solid |
| Micro-texture restore | `restore_micro_texture` — re-injects detail in nose/cheek/under-eye zones | Solid |
| Pore synthesis | Synthetic noise-based (`pore_synthesis`) | Basic (see S6) |
| Tone equalize | `equalize` (`skin.py:125`) — **global** median a/b pull + CLAHE on L | Basic (see S3) |
| Whiten / porcelain | 3 tone directions | Solid |
| Blemish removal | Contrast-anomaly detect + Telea inpaint | Solid |
| Under-eye, neck harmonize, zone dodge&burn | Present | Solid |
| Anime primitives | flatten / quantize / unify / glow (Stages A–B) | New, solid |
| Relight, specular bloom | Present (additive only) | See S4 |

## Part 2 — Research: the gaps (what pros do that the engine doesn't)

1. **🔴 Body skin is never touched.** All skin work runs inside the per-face ROI crop (`_process_one_face`, `engine.py:1384`); masks come from BiSeNet *face* parsing. Legs, arms, shoulders, décolletage, hands get **zero** retouch. In the reference cosplay images the legs are the second-most prominent skin surface — currently the face gets porcelain treatment and the legs keep raw texture/tone/veins/bruises → visible mismatch. No mainstream auto-retouch tool does body skin well; this is a differentiator, and the ingredients already exist (`person_mask` every run + `skin_mask_lch()` LCH skin detection in `utils.py`).
2. **🔴 No micro dodge & burn.** The #1 technique of high-end retouchers: even out *luminance blotchiness* at the transition frequency (between pores and shading) by locally brightening/darkening — never blurring. Smoothing removes texture; micro-D&B removes *unevenness*. This is precisely what Retouch4me "Dodge & Burn" automates and why its results look expensive. The engine has all the machinery (frequency bands) but no op.
3. **🟠 Redness/color blotches handled only globally.** `equalize` pulls every skin pixel toward the median a/b — it flattens *global* tone but can't kill a *local* red patch without also desaturating healthy color (that's why `anime_v2` runs `unify` on top). Pros treat color blotches at the same mid-frequency as luminance blotches.
4. **🟠 No shine/oil removal.** `specular_bloom` only *adds* glow. Hot T-zone shine, sweaty highlights under con-hall lighting, glare on foreheads — needs the inverse: compress specular L, reconstruct chroma underneath.
5. **🟡 No wrinkle/line softening.** Nasolabial folds, forehead lines, crow's feet are *ridge-shaped* (directional), so isotropic smoothing either misses them or kills everything around them. Needs ridge-aware attenuation, partial by design (depth-reduced, not erased).
6. **🟡 Pore synthesis is synthetic.** Noise-based pores read as artificial at 100% zoom. Pros clone texture from clean regions of the *same* face.

## Part 3 — Implementation documentation (Stages S1–S6)

### Stage S1 — Body Skin Retouch (~2 weeks) 🔴 flagship

**Mask construction (the hard part, all pieces exist):**
`body_skin = person_mask ∩ skin_mask_lch(image) − (face ROIs ∪ hair_mask)`, cleaned by morphological open/close, feathered via `feather_mask()`. Face ROIs excluded because they're already handled (and better). Computed once in Phase 1 alongside existing masks.

**New Phase-3 stage `_stage_body_skin`** (before global grade; after face compositing), driven by a `"body_skin": {...}` recipe block with its own ParamSpecs (`body_smooth`, `body_equalize`, `body_whiten`, `body_match_face`):
1. **Smooth:** guided-filter smoothing (P1's shared `guided_filter`) at body scale — radius from *person* size, not face size. Milder default than face (body texture reads as natural; over-smoothing legs looks plastic fast).
2. **Tone match to face** (`body_match_face`, the killer feature): measure mean LAB of *retouched* face skin (already have `acc_skin`) → pull body skin a/b/L toward it, bounded (max ±8 L, ±6 a/b). Face and legs finally match — porcelain everywhere. Reuses `harmonize_neck`'s statistical approach at body scale.
3. **Blemish pass on body:** run `BlemishRemover` with the body mask (bruises, insect bites, small marks) — size threshold scaled to person, conservative default.
4. Anime recipes extend naturally: `unify_tone` accepts any mask — run it with `body_skin` for cel-uniform legs/arms in `anime_v2`-class recipes.

**QA / artifact risks (each becomes a test):**
- **False positives are the top risk:** skin-toned fabric, wood furniture, leather. Mitigations: LCH detection ∩ person_mask only; chroma/luminance tightening; per-region contiguity check (body skin connects to neck/face regions — drop disconnected islands, protects background hands/props); GUI toggle + strength 0 default in `natural`.
- Tattoos/body paint (cosplay!): high-chroma exclusion zone inside body mask — tattoos survive equalize/whiten.
- Costume boundary seams: feather + containment test (no effect leakage past mask edge).
- Proxy consistency: body ops must run identically in preview and full-res paths (F8-aware).

**Files:** `retouch/engine.py` (mask build + stage), `retouch/skin.py` (body variants where face assumptions exist), `retouch/params.py`, `retouch/recipes.py`, `gui.py` ("🦵 Body Skin" accordion), tests.

### Stage S2 — Auto Micro Dodge & Burn (~1 week) 🔴 the pro-quality unlock

**Method (Retouch4me-class, classical implementation):**
1. Band-pass L channel at the *blotch* scale: `DoG = gauss(L, σ_small) − gauss(L, σ_large)` with σ tied to face width (σ_small ≈ pore-cluster scale ~ face_width/40, σ_large ≈ shading scale ~ face_width/12). This isolates unevenness *between* texture and form — exactly what pros paint on.
2. Correction = `−DoG × strength × skin_mask × edge_protect` added to L. Edge_protect = inverse gradient magnitude (don't touch feature lines: nostrils, lips, eye edges) × `_get_highlight_protection` (`skin.py:709`).
3. Pores unchanged (below band), face shading/relight unchanged (above band). Zero blur anywhere — this is evening, not smoothing.
4. Param `micro_dodge_burn` 0–100 (default ~25 in portrait recipes, 0 in `natural`); runs after frequency combine, before equalize. Reuses `FrequencySeparator`'s gaussian machinery; ~80 lines.

**QA:** flat-gradient invariance (no banding introduced — DoG of a linear ramp is ~0); pore-preservation (high-band energy unchanged ±2%); before/after blotch metric (std of band-passed L within skin must drop); halo check at jawline.

### Stage S3 — Color-Blotch Evening (~3 days, shares S2 infra)

Same band-pass, applied to **a-channel** (redness) and b-channel: `a ← a − DoG_a × strength × skin_mask`. Kills local red patches (nose sides, acne scars, irritation) while global tone and healthy color survive — precisely what `equalize`'s global median pull can't do. Param `redness_even`; keep `equalize` for global work (they compose: S3 local first, equalize global second). QA: lip/blush protection (exclude lips mask, blush zones applied *after*), overall chroma-shift bound.

### Stage S4 — Shine / Oil Removal (~1 week)

1. Detect: within skin, `L > adaptive threshold` ∧ low chroma (specular = desaturated) → shine mask, feathered.
2. Reconstruct: chroma inpainted from surrounding skin (guided filter of a/b with shine mask as unknown region); L compressed toward local non-shine median (soft rolloff, reuse `highlight.py:recover_highlights` logic at skin scale).
3. Param `shine_removal` 0–100. **Composition rule:** runs *before* `specular_bloom`/`relight` so intentional added glow isn't removed; document the order in `PIPELINE_FLOW.md`.
4. QA: catchlight/eye exclusion (eyes mask), jewelry exclusion (non-skin), "matte face" over-removal guard (cap L reduction), natural nose-tip highlight retains ≥30%.

### Stage S5 — Wrinkle & Line Softening (~1.5 weeks)

1. Ridge detection on mid-band L: black-hat morphology at 2–3 orientations (cheaper than full Frangi/Hessian, sufficient for lines) at wrinkle scale, restricted to **landmark zones**: forehead band, nasolabial polygons, crow's-feet fans, neck bands (landmark geometry from existing `FaceRegions`).
2. Attenuate: subtract detected ridge signal from mid band, `depth` capped at 60% by design (a 40-year-old with zero nasolabial fold reads as AI-fake; partial reduction reads as great skin).
3. Params `wrinkle_soften` + per-zone fine control later if needed. QA: hair strand/eyebrow exclusion, dimple/smile-crease preservation at low strength, no ghost-edges (over-subtraction ringing).

### Stage S6 — Texture Transplant (pore realism v2) (~1 week)

Replace synthetic-noise `pore_synthesis` fallback with **same-face texture cloning**: sample high-band patches from the cleanest skin region (lowest blotch metric from S2's band-pass — cheeks usually), build a small tileable texture, blend into over-smoothed/inpainted zones (blemish sites, S4 shine reconstructions) with rotation jitter and luminance matching. Synthetic path stays as fallback when no clean donor region exists. QA: no visible tiling (autocorrelation test), donor region unchanged, texture direction coherence at patch seams.

---

## Sequencing & master placement

```
S2 (micro D&B) ──► S3 (color evening) ──► S1 (body skin) ──► S4 (shine) ──► S5 (wrinkles) ──► S6 (texture v2)
   1 wk               3 days                2 wks              1 wk            1.5 wks           1 wk
```
- S2 first: biggest face-quality jump per line of code, and S3 rides its infrastructure 3 days later.
- S1 after: flagship value but the largest new surface (mask QA); benefits from S2/S3 existing (body gets them too via masks).
- **Master sequence placement:** Tier S needs **P1** (guided filter — S1 body smoothing at full-res demands it) and ideally lands after **F11** (self-QA detectors guard every S-stage). Recommended slot: `… F1 → F11 → [Tier S: S2→S3→S1] → P3 → F2/F3 → … → [S4–S6 with Tier 2] → …` — i.e., S2/S3/S1 jump ahead of the workflow tiers (they're pure output-quality, your core product), S4–S6 interleave with Tier 2 tools. Update the authoritative sequence in `PLAN_TIERP_PERF_ARCH_SHIP.md` when you commit to this.

## Verification (end-to-end)
1. Corpus A/B at 100% zoom per stage: cheek blotches (S2), nose-side redness (S3), face-vs-leg tone match on the 爻一爻-class shots (S1), forehead shine (S4), nasolabial depth (S5), healed-region texture (S6).
2. F11 detectors (once built): plastic-skin, halo, seam — all green on every recipe with new params at defaults and at 100.
3. Blotch metric (band-passed L std in skin) added to `benchmark.py` as a *quality* metric tracked over time — the first objective skin-quality number in the project.
4. Full pytest suite per stage; new params covered by the dead-key guard pattern from round 1.

## Open items
- `anime_crystal_void` dead keys — unchanged, fix vehicle remains Tier 3 T1.
- ~~Decision for you: adopt the recommended jump-ahead placement of S2/S3/S1 (quality before workflow), or keep Tier S after Tier Q.~~ **Resolved 2026-07-02: jump-ahead adopted — S2/S3/S1/S4 are Phase 2 ("Skin Supremacy") in `MASTER_PLAN.md`, immediately after the quality floor; S5/S6 land in Phase 4.**
