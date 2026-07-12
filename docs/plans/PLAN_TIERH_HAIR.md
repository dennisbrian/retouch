# Tier H Plan — Hair & Wig Retouch (Stages H1–H4)

**Date:** 2026-07-03
**Type:** Research + stage definitions (no code)
**Question:** the engine treats hair with a single `hair.shine` slider, yet wigs occupy a third of every cosplay frame and are the #2 manual-retouch time sink after skin. What does a pro-grade hair stack look like?
**Fits:** complements Tier S (skin) and A3 (cosplay moat — A3.2 wig-lace blend stays there; H stages are the hair *surface* work).

---

## Part 1 — Research

### What the field does
- **Retouch4me Stray Hairs / Evoto stray-hair remover:** neural nets trained on pro retouches; both decompose the job into (a) flyaways along the hair silhouette, (b) hairs crossing the face, (c) an editable mask. That decomposition is the product spec, regardless of implementation.
- **Classical evidence:** dermoscopy hair-removal literature achieves clean automated hair removal with *classical* thin-structure detection (Hough/morphological line detection) + harmonic inpainting — i.e., a curvilinear-structure detector plus our existing Telea inpaint machinery covers a big share of the neural result, exactly the S2-vs-Retouch4me story again.
- **Manual pro workflow (CN tutorials):** surface-blur + clone stamp along the silhouette band; shine painted as a coherent band ("天使环" angel ring) perpendicular to strand flow; wig-specific problems called out: synthetic high-temp fiber (高温丝) has a harsh, broad, plasticky specular; wig color is uniform-flat (no root/tip variation) and often casts wrong under venue light.
- **Rendering theory:** hair specular is anisotropic — highlight runs *perpendicular to strand direction* (Kajiya-Kay; Marschner adds primary/secondary lobes + transmission). For retouching this gives the correct generative model: estimate per-pixel strand direction, then shape/add speculars as smooth bands orthogonal to flow — never as isotropic blobs (which is what current `hair.shine` risks).

### Engine assets already in place
`hair_mask` from BiSeNet every run; `hair.shine` op; Telea inpaint (`blemish.py`, F4 heal plans); guided filter; person mask; frequency bands. Missing primitive: **strand flow field** (structure tensor) — one new ~40-line helper powers H1–H3.

## Part 2 — Stages

### Stage H0 — Strand flow field (shared primitive, ~3 days)
`hair_flow(img, hair_mask) -> (orientation, coherence)` via structure tensor: per-pixel gradient outer product, Gaussian-smoothed, principal eigenvector = strand direction; coherence = eigenvalue anisotropy (low coherence = matte/fuzzy region). Compute at ≤1200px, upsample. Unit tests on synthetic stripe patterns.

### Stage H1 — Flyaway & stray hair removal (~1.5 weeks) 🔴 highest demand
1. **Silhouette flyaways:** band = dilate(hair_mask ∪ person_mask boundary) − erode(...). Within the band, detect thin curvilinear structures: black-hat/top-hat at 2–4px scales along multiple orientations (or Frangi vesselness on L), gated by *low coherence with the main flow* (a flyaway disagrees with local strand direction; a smooth edge strand agrees). Remove = inpaint (Telea, small radius) or background-color fill where the background is smooth; strength slider = detection threshold.
2. **Face-crossing hairs:** same detector run *inside* skin_mask (thin dark/light curvilinear structures over skin) → inpaint. This is also the stray-hair-on-skin item from A4 — classical first, neural only if A1 evidence demands.
3. **Guards:** never remove inside eyebrow/eyelash regions; strand-thickness cap (≤4px — thicker = intentional hair lock); QA corpus with fine intentional wisps (two-side error metric: missed flyaways vs eaten wisps).
**Params:** `flyaway_remove` 0–100, `face_hair_remove` 0–100. **Files:** `retouch/hairwork.py` (new), tests.

### Stage H2 — Wig shine shaping (~1 week) 🔴 the cosplay differentiator
The two-sided op the market lacks: **tame** synthetic-fiber glare and **re-add** a coherent anisotropic band.
1. Compress existing specular within hair_mask (L > adaptive threshold → soft rolloff — S4's shine logic at hair scale).
2. Add "angel ring": band position from head geometry (top-of-head landmark arc), profile = Gaussian across the band, modulated by `cos²` of angle between band tangent and strand flow (Kajiya-Kay-flavored), tinted slightly toward hair chroma (not pure white).
3. Params: `hair_deglare`, `hair_ring` (strength), `hair_ring_position` (0–1 vertical). Reuses the existing `hair.shine` param as the ring strength (upgrade, not duplicate).
**QA:** ring must break at parting/occlusion (coherence gate), no ring on very short/fuzzy wigs (coherence floor), deglare preserves dimensionality (mid-band energy floor).

### Stage H3 — Hair color unify & tint (~4 days)
LCh pull toward the wig's median hue within hair_mask (kills venue-light color cast on white/silver wigs — the DSCF corpus case: warm café light turning silver wigs orange in shadows), plus optional target-tint override (recipe key, e.g. anime look wants lavender-white). Chroma-only by default (L untouched — preserves shading); flow-aware feather at the hairline. Params: `hair_unify`, `hair_tint_hue/strength`. Shares C1's hue-line math (`color_science.py`).

### Stage H4 — Depth dodge & burn along flow (~4 days, optional finisher)
Wig flatness fix: darken interior/under-layers, lift crown/edge locks — D&B field generated from distance-to-silhouette + flow coherence, applied on low band. Param `hair_depth`. (Manual pros do this on every wig shot; zero tools automate it.)

## Sequencing & placement
```
H0 (flow field) ──► H1 (flyaways) ──► H2 (shine) ──► H3 (color) ──► H4 (depth)
```
Proposed MASTER_PLAN slot: **Phase 4** (with the manual tools — H1 pairs naturally with F4 heal since both are inpaint-based), except **H3 can ride Phase 2/3 cheaply** (it's a mask + C1-math reuse). A3's wig-lace blend (hairline) stays in A3; H stages assume it.
**Adopted into MASTER_PLAN.md 2026-07-03:** H3 → Phase 3 (row 14b, rides C1 math); H0+H1 → Phase 4 (row 18b, pairs with F4); H2(+H4) → Phase 4 (row 18c).

## Verification
- H1: two-sided corpus metric (flyaways removed ≥80%, intentional wisps eaten = 0 on labeled 20-image set); before/after at 100% on silhouette bands.
- H2: ring coherence visual A/B vs manual angel-ring edit; no isotropic blob highlights.
- H3: silver-wig venue-cast corpus — hue std within hair_mask halved, L histogram unchanged.
- Full pytest per stage; params under dead-key guard.

## Sources
- [Retouch4me Stray Hairs](https://retouch4.me/stray-hairs) · [ePHOTOzine coverage](https://www.ephotozine.com/article/retouch4me-released-ai-powered-plugin-to-streamline-stray-hair-removal-37453) · [Evoto stray-hair remover](https://www.evoto.ai/features/stray-hair-remover)
- [Dermoscopy hair removal via Hough + harmonic inpainting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9777124/)
- [头发渲染: Kajiya-Kay → Marschner — 知乎](https://zhuanlan.zhihu.com/p/330259306) · [各向异性高光 — CSDN](https://blog.csdn.net/qjh5606/article/details/118117176)
- [碎发精修手法 — Envato Tuts+](https://photography.tutsplus.com/zh-hans/tutorials/3-ways-to-retouch-fly-away-hair--cms-20373) · [Cosplay假发材质（高温丝/卡丝）— 知乎](https://zhuanlan.zhihu.com/p/166477541)
