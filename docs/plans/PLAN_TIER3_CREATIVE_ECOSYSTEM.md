# Tier 3 Plan — Creative Expansion & Ecosystem (Stages T1–T5)

## Context

Fifth planning round — the last unplanned territory. Previous docs: looks (`PLAN_MOONLIGHT_PORCELAIN.md`), quality+workflow (`PLAN_TIER1_FOUNDATION.md`), manual tools (`PLAN_TIER2_MANUAL_TOOLS.md`), fidelity+intelligence (`PLAN_TIERQ_FIDELITY_INTELLIGENCE.md`), perf/arch/ship (`PLAN_TIERP_PERF_ARCH_SHIP.md`). Tier 3 covers what makes the system *creatively* exceed Photoshop for cosplay/portrait work — scene control, full makeup, body reshape — and what makes it an *ecosystem* rather than an app: plugins and a recipe cookbook.

**Exploration findings this round:**
- `MakeupEngine` (`retouch/makeup.py`) contains only `apply_blush` — eyeshadow/liner/contour/brows are greenfield, but the pattern (landmark regions + LAB tint + feathered blend) is established by blush/lips/eyes modules.
- `recipe_loader.py` already has the full ecosystem substrate: `import_recipe`/`export_recipe`/`write_recipe_json`/user-recipe dir with flat↔engine conversion and schema validation (`recipe_schema.py`). A cookbook is mostly UI, not plumbing.
- `register_presets_dir()` (`grading.py:57`) is an existing plugin-shaped hook.
- **No pose detection exists** — body reshape (T3) needs MediaPipe Pose (new model, same runtime family as existing `.task` files).
- Person + hair masks already produced every run (`engine.py` Phase 1) — background replacement's hard part (segmentation) is already paid for.
- Expression editing from the old v2 roadmap is **already covered by F5's** smile/eye warp sliders (Tier 2) — dropped from this tier to avoid double-planning.

---

## Stage T1 — Background Replace & Scene Relight (~2 weeks)

**Goal:** swap/darken/restyle the environment — studio-in-software. **This finally wires the 7 dead `anime_crystal_void` keys** (`recipes.py:378-384`), closing the oldest open bug properly instead of deleting the recipe.

### Steps
1. **New `_stage_background`** (Phase 3, before grade): consumes `person_mask` + `hair_mask`. Ops, each an independent ParamSpec: `background_blur`, `background_desaturation`, `background_exposure`, `matte_black` (lifted-black floor), `blue_shadow_grade` / `cyan_midtone_grade` (background-scoped split-tone via existing `split_tone_mask` arg in `grade()`), `subject_sharpen` (mask-scoped sharpen — reuse existing sharpen path).
2. **Edge quality — the whole game:** trimap from person_mask (erode=FG, dilate band=unknown) → guided-filter alpha matting (reuse shared `guided_filter` from P1) for hair-wisp-friendly alpha. Cache per image.
3. **`light_wrap`:** blurred background bled into the subject's edge band (alpha-gradient region), screen-blended (`regions.py` blend modes). Sells any background change.
4. **Backdrop replacement v1:** user image or generated backdrop (solid/gradient/vignetted studio color) + ambient match (mean LAB of new backdrop nudges subject shadows via existing split-tone machinery) + light_wrap over the new plate.
5. Register the 7 keys as ParamSpecs (dead-key guard test flips from "documents the bug" to "verifies the fix"); `anime_crystal_void` becomes functional as advertised; add GUI "🎭 Background" accordion.

### QA / tests
Halo/seam detector (F11) along mask boundary; hair-wisp visual corpus (the 爻一爻 images are perfect test cases — smoke + flyaway wig hair); alpha matting unit tests on synthetic trimaps; `anime_crystal_void` end-to-end now asserts visible background effect.

**Files:** `retouch/engine.py` (or `stages.py` post-P3), `retouch/regions.py` (matting), `retouch/params.py`, `gui.py`, tests.

---

## Stage T2 — Makeup Engine v2 (~2–3 weeks)

**Goal:** full cosplay makeup stack — the anime looks you're building lean hard on makeup that `blush` alone can't deliver.

### Steps (one primitive per commit, mirroring the Stage A/B pattern)
1. `eyeshadow(color, strength, style)` — lid region from eye landmarks extended upward, gradient falloff, LCH tint; styles: soft / smoky / anime-gradient.
2. `eyeliner(strength, wing)` — polyline along lash line landmarks, tapered stroke, slight gaussian.
3. `contour_highlight(strength)` — cheekbone shadow + nose-side shading + brow/nose-bridge/cupid highlights; reuses the dodge&burn region math (`SkinProcessor.dodge_burn`) with fixed cosmetic placement.
4. `brows(strength, color)` — brow region fill + edge crispening (region exists in BiSeNet `FaceRegions`).
5. `lip_liner` / ombre-lip mode — extends existing `lips.py` tint machinery (Korean-gradient lip = center-weighted tint falloff).
6. Recipe block `"makeup": {"eyeshadow": {...}, "eyeliner": 0.3, ...}` + ParamSpecs + GUI "💄 Makeup" accordion; add tasteful amounts to `anime_v2`/`moonlight_porcelain` variants.

### QA / tests
**Yaw gating is the critical guard:** landmark-painted makeup smears on profile faces — compute face yaw from landmarks, fade all makeup strength to 0 beyond ~35°, test at 0/20/45°. Occlusion: hair-over-face (BiSeNet hair mask subtracts from makeup regions). Per-primitive region-containment tests (pigment never outside its mask).

**Files:** `retouch/makeup.py`, `retouch/lips.py`, `retouch/params.py`, `retouch/recipes.py`, `gui.py`, tests.

---

## Stage T3 — Body Reshape (~2–3 weeks, highest risk — last)

**Goal:** waist/shoulders/legs/arms — the other half of cosplay retouch requests.

### Steps
1. **MediaPipe Pose** (`pose_landmarker.task`, download-on-first-use via P4's `model_fetch`): 33 body landmarks, same API family as face landmarker in `detection.py`.
2. Warp sets on the shared warp engine (F5's `_apply_warps`): `waist` (bilateral inward at hip-waist midpoints), `shoulders` (±width), `legs_length` (vertical stretch band between hip line and frame bottom — seam-blended), `arms_slim` (paired inward warps along upper-arm segments).
3. **Background protection (the artifact that kills every competitor):** warp displacement weighted by `person_mask` so pixels outside the subject move minimally; F5's straight-line preservation test extended to body warps (walls/furniture behind the waist are the classic tell — the reference images' chair/table edges are exact test material).
4. Confidence gating: pose visibility scores below threshold → sliders inert + GUI notice. Multi-person: apply only to the largest/selected person (person_mask intersection).
5. ParamSpecs (`body_waist` −50..+50 etc., default 0, **never** in recipes' defaults) + GUI "🧍 Body" accordion.

### QA / tests
Straight-line guard behind each warp zone; displacement cap (≤ torso_width × 0.05 at strength 100); no-pose no-op; person_mask weighting verified on synthetic grid backgrounds.

**Files:** `retouch/detection.py` (pose), `retouch/geometry.py` (body warp sets), `retouch/params.py`, `gui.py`, `models/manifest.json`, tests.

---

## Stage T4 — Plugin API v0 + Recipe Cookbook (~2 weeks, requires P3)

**Goal:** third parties (or future-you) extend the system without touching core; recipes become shareable artifacts with a browsable home.

### Steps
1. **Plugin contract:** a plugin is a Python package exposing a `retouch_plugin` entry point returning a manifest: `{stages: [Stage], param_specs: [ParamSpec], preset_dirs: [path], lut_dirs: [path], recipes: {…}}`. Stages slot into P3's registry at declared phase anchors; `register_presets_dir()` (`grading.py:57`) already handles preset dirs — replicate for `luts/`.
2. **Loading:** `~/.promax_retouch/plugins/` + entry-point discovery; failures isolate (plugin error disables plugin, never crashes app); `--no-plugins` escape hatch. Document the contract in `PLUGIN_API.md` with a worked example (e.g., a "duotone stage" sample plugin in `docs/`).
3. **Recipe cookbook UI:** gallery tab rendering each recipe (built-in + user + plugin) as a thumbnail — apply recipe to a bundled sample image at 400px (cache renders), click → load recipe. Import/export buttons wire the existing `import_recipe`/`export_recipe` (currently CLI-only plumbing).
4. **Recipe provenance:** exported JSON gains `author`/`description`/`version` metadata (schema already validates unknown keys — extend `recipe_schema.py`).

### QA / tests
Plugin lifecycle (load/disable/fail-isolation), malformed-manifest rejection, cookbook thumbnail cache invalidation, import of an exported recipe round-trips byte-identical.

**Files:** `retouch/plugins.py` (new), `retouch/stages.py` (anchors), `retouch/recipe_loader.py`, `retouch/recipe_schema.py`, `gui.py`, `PLUGIN_API.md` (new), sample plugin under `docs/`, tests.

---

## Stage T5 — Linear RAW Develop (~2 weeks, requires F1)

**Goal:** RAW files get true exposure/WB/highlight-recovery in linear light before the display-referred pipeline — Lightroom's core advantage over "open JPEG in Photoshop."

### Steps
1. `io.py`: rawpy postprocess with `output_bps=16, gamma=(1,1), no_auto_bright=True` → linear 16-bit; keep current path for non-RAW.
2. **`retouch/develop.py` (new), linear-domain ops:** exposure (pure multiply — artifact-free in linear), WB (channel gains, ties into F9's measured cast), highlight recovery (reconstruct clipped channels from unclipped neighbors — the classic trick that only works pre-tone-curve), then filmic tone-map into the display-referred float pipeline (F1's entry point).
3. GUI: "RAW Develop" accordion that appears only for RAW inputs (exposure EV, WB, highlight recovery, shadow lift).
4. Sessions (F2) store develop settings; batch supports them.

### QA / tests
Linear correctness (exposure +1EV exactly doubles linear values), highlight-recovery on synthetic clipped gradients, RAW→JPEG visual parity corpus, non-RAW paths byte-identical (regression).

**Files:** `retouch/io.py`, `retouch/develop.py` (new), `gui.py`, `retouch/session.py`, tests.

---

## Sequencing within Tier 3 & master placement

```
T1 (background) ──► T4 (plugins+cookbook) ──► T2 (makeup) ──► T5 (RAW) ──► T3 (body)
```
- T1 first: wires the long-standing dead keys, reuses masks already computed, and feeds the moody-set looks you shoot.
- T4 second: needs P3 (done by then per master sequence); unlocks community recipes early.
- T3 last: new model + highest artifact risk — benefits from every guard built before it.
- **Master sequence:** Tier 3 slots after F9/F10 and before/alongside P4 (P4's model-fetch must precede T3; T5 needs F1; T4 needs P3): `P1 → F8+P2 → F1 → F11 → P3 → F2/F3 → F4–F7 → F9/F10 → T1–T5 → P4 (ship)`.

## Verification (end-to-end)
1. Full pytest + F11 QA-corpus green after each stage.
2. T1 acceptance: replace background on the 爻一爻-style test shot — hair wisps intact at 100%, `anime_crystal_void` visibly works.
3. T2 acceptance: anime makeup on frontal + 20° face, zero smear at 45° (gated).
4. T3 acceptance: waist −20 on a shot with straight furniture lines behind subject — lines stay straight.
5. T4 acceptance: sample plugin installs, adds a stage + preset, uninstalls cleanly; exported recipe imports on a second machine.
6. T5 acceptance: underexposed RAW recovered +2EV without banding; clipped sky/highlight detail partially reconstructed.

## Open items
- With T1 planned, the `anime_crystal_void` dead-keys bug finally has a designated fix vehicle (was report-only since round 1).
- After Tier 3, the plan set covers the entire roadmap through "ship" — remaining unplanned ideas (mobile, video) stay explicitly out of scope per `ROADMAP.md`.
