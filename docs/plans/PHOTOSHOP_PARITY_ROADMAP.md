# Photoshop-Parity Roadmap — Gap Analysis & Prioritized Plan

**Date:** 2026-07-02
**Status:** Exploration/planning document (post-V1, post-`moonlight_porcelain`)
**Question answered:** "What do we build next to make this as powerful as Photoshop — or better — for portrait/cosplay retouching?"

---

## 1. Where we already BEAT Photoshop

Don't rebuild these — they're the moat. Photoshop makes the user do all of this manually:

| Capability | Ours | Photoshop equivalent |
|---|---|---|
| One-click looks (recipes + film sims + presets) | 25+ recipes, JSON, shareable, inheritable (`extends`) | Actions/LUTs — no parameter inheritance, no semantic awareness |
| Semantic face masking | BiSeNet + MediaPipe: skin, lips, eyes, nose bridge, under-eye, cheek highlights — automatic, per face | Select Subject + manual channel work |
| Skin-protected color grading | `skin_protect` wraps every color op | Manual masking every time |
| Frequency separation | Automatic 3-band (low/mid/high) with per-band controls | Famous 8-step manual recipe |
| Anime/cel skin (flatten, quantize, unify, glow) | First-class primitives | Hours of hand-painting |
| Batch with recipes | Folder → ZIP, per-image face detection | Actions (fragile, no face awareness) |
| Test coverage | 1600+ tests on the imaging math | N/A |

**Strategic frame:** we are not chasing Photoshop-the-generalist (text, vectors, compositing, 3D). We are chasing **"everything a portrait/cosplay retoucher opens Photoshop for"** — that's a much smaller, winnable surface.

## 2. The honest gap list (what retouchers still need Photoshop for)

1. **Non-destructive editing** — layers, adjustment layers, opacity, per-layer masks, history/undo. *We are a single feed-forward pipeline with no undo.* Biggest architectural gap.
2. **Local manual adjustments** — brush/radial/linear masks driving any adjustment. We only have automatic semantic masks.
3. **Manual healing/cloning** — click-to-remove a stray hair, wrinkle, sensor dust, background object. Our blemish removal is automatic-only.
4. **Liquify / face-aware reshape** — eye size, nose width, jaw, forehead, smile. We only have cheek `slimming` (`geometry.py:FaceReshaper` is the seed).
5. **16-bit end-to-end** — PS works in 16/32-bit. Our engine hands off **uint8 between stages (17 `astype(np.uint8)` sites in `engine.py`)** despite `precision.py` float helpers existing. This caps quality: banding in exactly the dark-gradient looks we just shipped (`moonlight_porcelain`).
6. **Interactive curves** — a draggable curve editor. Ours live only in preset JSON.
7. **AI denoise / super-resolution** — LR/PS have it; cosplay shooters crop heavily and shoot high-ISO indoor sets (see the reference images: smoke + dark set = high ISO).
8. **Generative fill / object removal at PS Firefly level** — the one gap we should NOT chase head-on; LaMa-class inpainting gets 80% for the retouch use case.
9. **RAW develop module** — we decode RAW via rawpy defaults, then work in 8-bit sRGB. No linear-space exposure/WB/highlight-recovery.
10. **Dodge & burn brush** — manual D&B is the core of high-end skin work; ours is automatic-only.

## 3. Prioritized plan

### Tier 1 — Foundation that multiplies everything else

#### 1.1 Float32 end-to-end pipeline (quality ceiling raise) — ~1-2 weeks
The single highest-leverage engineering task. `precision.py` + `_F_*` grader variants already exist; the gap is the **17 uint8 round-trips between engine stages**.
- Route the full `process()` chain in float32 [0,1]; convert to uint8 once at output.
- Add 16-bit TIFF/PNG export in `io.py` (PS-parity claim needs 16-bit out).
- RAW inputs: request 16-bit output from rawpy (`output_bps=16`) so RAW shooters get real dynamic range into the float pipeline.
- **Direct payoff:** kills the banding risk documented in `PLAN_MOONLIGHT_PORCELAIN.md` QA matrix at the root, instead of dithering around it with grain.
- Risk: cv2 LAB/HSV conversions have different ranges in float — needs a careful stage-by-stage migration with per-stage regression tests (SSIM against uint8 baseline).

#### 1.2 Edit-stack / session model (the "layers" answer) — ~2 weeks
Photoshop's power is *revisability*, not the pixels. We don't need raster layers — the pipeline **is** a layer stack; make it first-class:
- **Session file**: serialize {recipe, all param overrides, masks, crop} to JSON. Load = exact reproduction. (Roadmap v3 "workspaces" — pull it forward.)
- **Undo/redo** in GUI: a bounded history of session states (cheap — sessions are tiny dicts; re-render on undo).
- **Per-stage opacity/bypass**: expose "stage mixer" (skin block, relight, grade, effects) each with 0–100% blend using existing `regions.py` blend modes. This is "adjustment layer opacity" without a layer engine.
- **Snapshots/versions**: named session saves per image, A/B toggle in GUI.

#### 1.3 Brush-painted local masks (adjustment brush) — ~2 weeks
Gradio ≥4 ships `gr.ImageEditor` with brush layers — we can capture painted masks today.
- New GUI accordion "Local Adjustments": paint a mask → choose an op (exposure, warmth, saturation, clarity, smooth, dodge, burn) → strength slider.
- Backend: painted mask → `feather_mask()` (`regions.py`) → route through the same `apply_to_region`-style path as semantic masks.
- Add radial + linear gradient mask generators (two clicks, no painting) — Lightroom-parity.
- **Killer combo nobody else has:** painted mask **intersected with** semantic masks ("this brushstroke, but only on skin") via `combine_masks()`.

### Tier 2 — The manual tools that close the PS habit loop

#### 2.1 Spot heal / object removal (click-to-fix) — ~1-2 weeks
- v0: click/lasso → OpenCV `inpaint` (Telea) for small spots (stray hairs, dust, small blemishes). Ships fast, no model download.
- v1: LaMa ONNX (~200MB, runs fine on CPU/CoreML) for large-region removal — background clutter, wig lace, support stands. This covers the #1 reason cosplay retouchers round-trip to PS.
- GUI: same `gr.ImageEditor` paint surface as 1.3, different routing.

#### 2.2 Face-aware liquify sliders — ~2-3 weeks
Extend `geometry.py:FaceReshaper` (already does landmark-driven warping for `slimming`):
- Sliders: eye size, eye distance, nose width, nose length, jaw width, chin length, forehead, mouth size, smile lift.
- Same mechanism as slimming: landmark-anchored thin-plate-spline / moving-least-squares mesh warp, per-face.
- This is PS "Face-Aware Liquify" parity, and it's automatic-per-face where PS makes you select faces.
- Artifact guard: warp field must be masked to face + feathered into background (straight lines in backgrounds are the classic liquify tell); add a background-line-preservation test.

#### 2.3 Interactive curve editor + curve-based match-to-reference — ~1 week
- GUI: draggable L / R / G / B curve points (Gradio `gr.LinePlot` won't do it — use a small custom JS component or `gr.HTML` canvas; timebox it).
- We already have `color_transfer` — add "extract curves from reference" so a user drops a finished photo (like the 爻一爻 references) and gets an editable starting curve + split-tone instead of an opaque transfer.

#### 2.4 AI denoise + super-resolution — ~1-2 weeks
- ONNX models, same runtime we already ship (BiSeNet/RetinaFace precedent): SCUNet or NAFNet for denoise, Real-ESRGAN x2/x4 for upscale.
- Run denoise **before** the pipeline (high-ISO smoke sets), upscale at export.
- Auto-tile large images to bound memory; tests for tile-seam artifacts.

### Tier 3 — Differentiators beyond Photoshop

- **Manual dodge & burn brush** (uses 1.3 infrastructure; luminosity-mask-aware strokes — better than PS's crude tool).
- **Background replace/relight:** we already have person segmentation + `relight` — composite onto new backdrop with `light_wrap` + ambient color match. (Also the natural moment to wire the 7 dead `anime_crystal_void` keys — still unfixed, see bug report in `PLAN_MOONLIGHT_PORCELAIN.md`.)
- **Linear RAW develop:** exposure/WB/highlight-recovery in linear space before the display-referred pipeline (depends on 1.1).
- **Recipe marketplace/cookbook + plugin API** (roadmap v3/v4 — unchanged).
- **Skip/defer:** full generative fill (Firefly parity is a foundation-model product, not a feature), text/vector/compositing, video.

## 4. Suggested execution order

> ⚠️ **Superseded (2026-07-02):** the current master sequence for all tiers is in `PLAN_TIERP_PERF_ARCH_SHIP.md` — P1 (guided filter) and F8 (full-res) now come before 1.1/F1. The order below is kept for historical context.

```
1.1 float32 pipeline ──► 1.2 session/undo ──► 1.3 brush masks ──► 2.1 spot heal
                                                      │
                                                      └──► Tier 2.2–2.4 in any order
```

Rationale: 1.1 is invisible but raises the ceiling for everything (and de-risks the dark-look recipes just shipped). 1.2 + 1.3 change the product category from "filter app" to "editor." 2.1 removes the most common reason to open Photoshop at all.

## 5. Known bugs still open (report-only, from previous cycle)

- `anime_crystal_void`: 7 dead recipe keys (`retouch/recipes.py:378-384`) — unchanged; would be revived by Tier 3 background work.
- Engine inter-stage uint8 truncation (this doc, 1.1) — quality bug class, not a logic bug.

## 6. Success criteria ("as powerful as Photoshop" — measurable)

- A retoucher can complete a full cosplay edit (heal, reshape, local D&B, grade, export 16-bit) **without opening Photoshop once**.
- Zero visible banding at 100% in dark-gradient looks (`moonlight_porcelain` class) on 16-bit export.
- Any edit reproducible from a session JSON; any slider undoable.
- Painted-mask local adjustment round-trips in < 2s preview at 800px.

## 7. Competitor research — Adobe Lightroom Web (2026-07-14)

Source: user walkthrough of lightroom.adobe.com (web app), doc: [Adobe Lightroom Web — Retouch System Reference](https://docs.google.com/document/d/1Jb0S-QssfRVJlIvarqOK24vLl8iFQHDpjI6-a9Jy1eE/edit?usp=sharing).

**Gaps this surfaces that PS-parity doesn't already cover:**
- **People-mask granularity** (LR's Masking → People): separate masks per attribute — Entire Person, Facial Skin, Body Skin, Eyebrows, **Eye Sclera**, **Iris and Pupil**, Lips, Hair, Clothes, with a "create separate masks" option. Our BiSeNet classes should be checked against this list — sclera/iris as distinct semantic masks may not exist yet and would sharpen eye-enhancement precision beyond a single "eyes" class.
- **Generative Remove/Heal** — maps directly to Tier 2.1 (spot heal/object removal) above; LR ships both a "Detect objects" auto mode and a People/Blemishes one-click distraction-removal category. Confirms 2.1 v0 (OpenCV inpaint) + v1 (LaMa) prioritization is the right shape.
- **Lens Blur (simulated depth of field)**: Blur Amount, bokeh shape presets, Cat Eye, Bokeh Boost, Focus Range (by subject or point) with a "Visualize Depth" toggle. Worth comparing against our relight/lens-effects stage for bokeh-shape and focus-by-point controls specifically.
- **Content Credentials / AI edit provenance** (C2PA-style tagging of AI-edit steps, "AI edit status" badge on auto-versions) — not in our roadmap anywhere; flagged as a possible future compliance/trust feature, not prioritized yet.
- **Versions (non-destructive branching)** — LR's "Create new version" is a lighter-weight variant of our Tier 1.2 session/undo model; no new gap, just corroborates 1.2's direction.

**Where we still clearly exceed LR (no action needed):** LR's "Skin" quick action is a single Refine + Soften slider pair — far coarser than our frequency-separation + component-enhancement pipeline. Their preset system (Community/Recommended/Yours) has no parameter inheritance (`extends`) like ours.

**Follow-up resolved (2026-07-15):** audited `retouch/parsing.py` — iris and sclera are already separable, distinct `FaceRegions` attributes (`left_iris`/`right_iris`/`left_sclera`/`right_sclera`), computed in `_add_landmark_subregions()` on *every* parse path (BiSeNet-success and landmark-fallback alike): iris is a landmark-derived ellipse, sclera is `eye_mask - iris_mask`. Both are actively consumed downstream in `eyes.py` and `eye_enhancement.py` for iris sculpting and sclera whitening — not a dead stub. **The people-mask-granularity gap flagged above does not apply to iris/sclera; it's already at LR parity there.** Any remaining granularity gap vs. LR's People-mask list (Body Skin, Clothes as separate maskable targets, "create separate masks" UI toggle) is a GUI/UX exposure question, not a missing semantic-mask capability.
