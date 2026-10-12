# Next Feature Improvements — Productization-First Plan

**Date:** 2026-08-12
**Status:** Tranche 1 core implementation landed; pinned CPU face-aware runtime verified; P0 safety audit implemented; human visual certification remains pending
**Owner direction:** Document the next improvements before changing the engine
**Related:** `MASTER_PLAN.md`, `PHOTOSHOP_PARITY_ROADMAP.md`,
`PLAN_FEATURE_FRONTIER.md`, `PLAN_VIDEO_FACE_RETOUCH.md`, `VISUAL_QA.md`

## 1. Outcome

The next tranche should make existing engine capability visible, trustworthy,
and easy to operate before adding more recipes or isolated image algorithms.

The recommended first implementation is an **Advanced Retouch workspace** that
exposes the existing manual-heal, local-adjustment, face-aware reshape, session,
and semantic-mask infrastructure through the GUI.

This plan deliberately separates:

- **productization work** — making shipped engine capability usable;
- **capability-truth work** — completing or clearly labeling model fallbacks;
- **quality certification** — proving results on real face-aware renders; and
- **new feature work** — adding teeth, lip, color, and video capabilities.

## 2. Current-state evidence

Snapshot taken from the repository on 2026-08-12:

- `retouch/params.py` contains approximately 218 processing parameter
  declarations.
- `retouch/recipes.py` defines 128 unique recipe names.
- `gui.py` declares 127 `gr.State` objects. Some are ordinary UI/session state,
  but many carry processing parameters that have no visible control.
- Nine global face-aware reshape parameters and the left/right reshape variants
  are represented as hidden state rather than normal GUI sliders.
- `RetouchEngine.process()` already accepts manual heal entries and local
  adjustments. `_stage_local_adjustments()` supports exposure, warmth, and
  other operations with optional semantic-mask intersection.
- Per-face recipe selection, session save/load, snapshots, undo/redo, batch
  jobs, recipe browsing, and diagnostics already have GUI footholds.
- NAFNet denoising has a real ONNX model, while LaMa and Real-ESRGAN remain
  unbundled placeholders. The neural stray-hair and defect boosters are parked
  implementations that return empty masks.
- EXR is registered as a file extension but has no float32 codec path.
- Display P3 and Rec.2020 gamut-boundary functions exist but are not selected by
  the live export pipeline.
- The current `test_output` tree contains 54 recipe manifests, including 39
  marked `global_only: false`. This is useful historical evidence, but it is
  not a complete certification of the present 128-recipe/218-parameter surface.

## 3. Product principles

1. **Productize before expanding.** Do not add another large recipe family
   while major implemented controls remain hidden or API-only.
2. **No misleading controls.** A control labeled AI, LaMa, neural, or
   super-resolution must run the named model, clearly report a fallback, or be
   marked experimental/unavailable.
3. **Keep defaults stable.** New controls remain no-ops at their defaults and
   must not alter existing recipe output unless a recipe explicitly opts in.
4. **Face-aware claims require face-aware evidence.** A headless
   `--global-only` run validates global finishing only. It does not validate
   detection, parsing, skin work, eyes, teeth, lips, hair, or per-face controls.
5. **Visual quality outranks feature count.** Every visual-critical change must
   pass the applicable gates in `docs/VISUAL_QA.md` on real portraits.
6. **Expose curated controls.** Do not dump all 218 parameters into one panel.
   Group high-value controls into understandable workflows, with expert
   controls collapsed by default.

## 4. Prioritized execution order

| Order | Tranche | Outcome | Estimated effort | Status |
|---|---|---|---|---|
| 0 | Certification baseline and recipe curation | Trusted reference corpus, Core recipe set, opt-in preflight | 3–5 d plus human review | Proposed |
| 1 | Advanced Retouch workspace | Brush/local edits, heal, visible liquify, semantic intersection | 1–2 wk | **IMPLEMENTED — visual QA pending** |
| 2 | Model and feature truth | Real model acquisition or honest unavailable states | 1–2 wk plus model/legal review | **Truth slice implemented; model acquisition remains pending** |
| 3 | Professional color and RAW export | GUI develop controls, P3/Rec.2020, real EXR/float path | 1–2 wk | Proposed |
| 4 | Facial realism frontier | Natural teeth and optical lip finishing | 2–3 wk | Proposed |
| 5 | Stable video V1 | Temporally stable single-face short-clip export | After V0 evidence | Deferred by gate |

Tranche 0 should run alongside Tranche 1 where it does not block GUI work.

## 5. Tranche 0 — Certification baseline and recipe curation

### Scope

- Select a small **Core production recipe set** from the 128 recipes. Target
  approximately 10–15 recipes; choose names only after reviewing usage and
  visual evidence.
- Label all recipes as `core`, `specialized`, `experimental`, or `legacy`.
- Add favorites/recent recipes and a single recipe-strength control to reduce
  browsing friction.
- Define a consented local portrait corpus covering:
  - light, medium, and dark skin;
  - neutral studio, high-key, low-key, mixed light, and venue light;
  - glasses, facial hair, occluding hands, wigs/flyaways, and multiple faces;
  - white garments and dark hair for clipping/crushing checks.
- Expose `run_preflight_checks()` and throughput measurement as an explicit
  diagnostics action. Do not run preflight during every engine construction.

### Acceptance

- Every Core recipe has a full face-aware render on the agreed corpus.
- Each run writes non-empty outputs, a contact sheet, and a manifest with
  `global_only: false`.
- Automatic texture, halo, clipping, shadow, seam, and plastic-skin checks are
  recorded where applicable.
- Natural Output, Skin Tone Uniformity, and geometry gates receive human review;
  they are never inferred from unit tests alone.
- Recipe labels and Core-set membership are machine-readable and tested.

## 6. Tranche 1 — Advanced Retouch workspace

### Goal

Turn existing API-level manual tools into a coherent editor workflow without
building a second imaging engine.

### UI scope

Add an **Advanced Retouch** panel to the Single Photo Editor:

1. **Mask editor**
   - painted brush mask;
   - visible clear/invert/feather controls;
   - overlay visibility and opacity.
2. **Local adjustments**
   - exposure, dodge, burn, warmth, saturation, clarity, and smooth;
   - positive and negative strength;
   - optional intersection with skin, hair, face, lips, eyes, clothing, person,
     or background masks.
3. **Heal / remove**
   - paint a small repair mask;
   - Telea and PatchMatch choices;
   - LaMa only when a real model is available;
   - clear fallback/status messaging.
4. **Face-aware reshape**
   - visible eye size/distance, nose width/length, jaw width, chin length,
     mouth size, smile, and forehead controls;
   - conservative ranges and one-click reset;
   - apply globally or to the selected detected face;
   - advanced left/right asymmetry controls collapsed by default.
5. **Editing state**
   - every edit participates in undo/redo;
   - snapshots include masks and per-face settings;
   - saved sessions reproduce the same local edits;
   - before/after and mask-overlay views remain available.

### Engineering boundaries

- Reuse `retouch/heal.py`, `retouch/regions.py`,
  `RetouchEngine._stage_local_adjustments()`, `retouch/geometry.py`, and the
  current session/history model.
- Do not duplicate local-adjustment math inside `gui.py`.
- Store masks in a compact, versioned session representation. Validate image
  dimensions when a session is loaded against a different source.
- Keep full-resolution render separate from preview rendering.
- Preserve the existing named GUI input/output mapping guards.

### Acceptance

- A user can paint a mask, change local exposure, undo it, redo it, save the
  session, reload it, and reproduce the edit.
- Semantic intersection visibly confines a broad brush stroke to its selected
  subject region.
- A painted heal changes only the intended masked area and preserves output
  shape, dtype, ICC/EXIF handling, and export behavior.
- The nine primary reshape controls plus left/right asymmetry controls are
  visible, resettable, and selectable per face.
- Face-aware geometry passes No Edge Tearing, No Halo, background-line, and
  Natural Output gates on real tilted and frontal portraits.
- Defaults remain byte-identical when no local edit or reshape is active.

Current implementation note: the browser editor uses an 8-bit preview canvas,
while the processed recipe result remains available as the Advanced Retouch
source. Full-resolution export uses the shared ICC/EXIF writer and offers
PNG-16/TIFF-16; face-aware visual certification and native-resolution visual
QA remain separate release gates.

## 7. Tranche 2 — Model and feature truth

### Scope

- Replace placeholder model-host URLs with real controlled release URLs.
- Require filename, size, SHA-256, license, provenance, and runtime contract for
  every downloadable model.
- Acquire and validate a dynamic-shape Real-ESRGAN x4 ONNX model, or expose
  Lanczos honestly as standard resize rather than AI super-resolution.
- Acquire and validate a LaMa ONNX model, or keep large-region removal disabled
  and expose Telea/PatchMatch only.
- Keep NAFNet denoising as the verified AI-denoise path.
- Either implement the parked neural stray-hair/defect segmenters or remove
  their user-facing controls and recipe keys from the production surface.

### Acceptance

- A clean install can fetch every advertised optional model from a valid URL.
- Downloaded files pass size and checksum validation before use.
- Offline behavior is explicit and does not silently change the feature name.
- Model-unavailable state is visible in diagnostics and beside the relevant UI
  control.
- Real-photo before/after evidence demonstrates that each model-backed control
  changes the intended region without identity or texture damage.

## 8. Tranche 3 — Professional color and RAW export

### Scope

- Add GUI RAW-develop controls for exposure, white balance, tint, contrast, and
  highlight recovery.
- Add an explicit output color-space selector: sRGB, Display P3, and Rec.2020.
- Wire target-gamut selection into the existing P3/Rec.2020 gamut-compression
  functions.
- Implement real float32 EXR read/write using a selected dependency and a
  documented linear-light channel contract.
- Preserve or intentionally transform ICC/EXIF metadata in a single export
  write.
- Treat HDR10/PQ as a separate opt-in export target after the float export path
  is proven. Do not imply that ordinary EXR is necessarily PQ encoded.

### Acceptance

- Round-trip tests cover float values outside the 8-bit range and define the
  accepted numerical tolerance.
- P3/Rec.2020 output carries the correct profile and uses the corresponding
  target-gamut compression.
- A real RAW reference set confirms highlight recovery, neutral balance, and
  no unexpected color shift.
- GUI and CLI export the same selected target when given equivalent settings.

## 9. Tranche 4 — Facial realism frontier

### Recommended feature order

1. **Mouth-region separation** — distinguish teeth, gums, oral shadow, and lip
   interior before stronger whitening.
2. **Natural teeth targeting** — move teeth toward a realistic shade locus,
   preserve gum color, inter-tooth shadow, incisal translucency, and enamel
   highlights.
3. **Optical lip finish** — matte/satin/gloss/wet control using diffuse and
   specular separation instead of flat tint plus bright-spot boosting.
4. **Targeted lip texture repair** — attenuate flakes and strong vertical
   fissures while retaining natural lip texture.
5. **Light coherence** — reuse the inferred scene light direction so eye, lip,
   and skin highlights agree.

### Acceptance

- Teeth never whiten gums, the oral cavity, or inter-tooth gaps.
- Whitening retains natural tooth hue and internal shading at maximum supported
  strength.
- Lip finish changes specular character without shifting surrounding skin or
  creating a pasted highlight.
- All new controls default to off and pass isolated region-difference tests plus
  human Natural Output review.

## 10. Tranche 5 — Video V1 gate

Do not start product video export until V0 has:

- a consented local corpus;
- measured track coverage and loss behavior;
- cut detection and track-reset evidence;
- flicker, texture-energy, color, mask-edge, and displacement thresholds; and
- a reviewed naive baseline demonstrating the failure being fixed.

After that gate, V1 remains a small target: one face, 720p, short offline clip,
stable masks, audio preserved, and safe fade/reset when tracking fails.

## 11. Explicit non-goals for this tranche

- Adding another large recipe family before recipe curation.
- Exposing all engine parameters in one flat expert panel.
- Auto-classifying people by gender or age for retouch strength.
- Re-attempting whole-engine gigapixel tiling without a per-stage tile-safety
  design; the previous whole-tile approach produced visible seams.
- Calling `--global-only` output face-retouched.
- Shipping generative identity modification without a separate consent,
  provenance, and quality policy.

## 12. Verification contract for every implementation tranche

1. Run focused unit and integration tests for the changed modules.
2. Run GUI wiring/import tests for any control-map change.
3. Run `git diff --check` and inspect the intended diff only.
4. Produce a headless/global-only smoke result when relevant, labeled as such.
5. For facial or semantic-mask work, run a GUI-attached full face-aware recipe
   or feature validation.
6. Require non-empty outputs/contact sheets and inspect the manifest for
   `global_only: false` before claiming face-aware success.
7. Record automatic and human visual gates separately.
8. Do not commit or ship a visual-critical change with a failed or falsely
   reported gate.

## 13. Decision checkpoint

The document was reviewed and implementation was explicitly authorized on
2026-08-12. The authorization covers the staged follow-on sequence in §14,
subject to each gate remaining truthful.

> Finish and verify Advanced Retouch first, then proceed through the staged
> quality, model-truth, workflow, export, facial-quality, and video gates.

## 14. Authorized follow-on priority sequence

The implementation authorization now also covers the following sequence after
the Advanced Retouch visual gate:

1. **P0 safety audit** — **IMPLEMENTED 2026-08-12**: automatic face treatment
   selection is neutral; continuous CIELAB/ITA measurements and uncertainty
   are exposed only for QA parity. Human tone-parity review remains pending.
2. **P1 capture fidelity** — **FOUNDATION IMPLEMENTED 2026-08-12**: EXIF
   camera/lens identification, keyed sensor-calibration profiles, RAW-mosaic
   dark/flat/hot-pixel correction primitives, and truthful optional Lensfun
   availability. Lensfun correction execution, camera profiles, and the
   ColorChecker wizard remain pending.
3. **P2 confidence-aware Safe Auto** — **CONTRACT IMPLEMENTED 2026-08-12**:
   `retouch/safe_auto.py` provides apply/dampen/skip/review decisions,
   confidence, reason, evidence, and byte-identical skip behavior. Wiring each
   automatic stage and measuring risk-versus-coverage remain pending.
4. **P3 burst portrait fusion** — **SHOOT INTELLIGENCE WORKFLOW SLICE
   IMPLEMENTED 2026-08-12**: Shoot Intelligence now provides explainable burst
   grouping, non-destructive candidate ranking, a persistent dependency/status
   graph, resumable watch-folder ingestion, GUI scan/watch controls,
   subject-linked project profiles, and multi-reference Look Boards. Actual
   aligned fusion, motion/expression exclusion, and hero-frame generation
   remain pending.
5. **P4 personal style learning** — **INSTRUMENTATION IMPLEMENTED
   2026-08-12**: sessions can record suggested/final parameters, outcome, and
   scene features without storing face pixels by default. The inspectable
   parameter-delta learner remains pending.
6. **P5 delivery soft proof** — sRGB/P3/print/HDR-to-SDR previews, gamut flags,
   social JPEG simulation, and thumbnail inspection.

These are downstream work items, not permission to bypass the current
Advanced Retouch face-aware QA gate. Each must retain separate automatic and
human review evidence.

## 15. Additional authorized decisions

- **P6 — Identity & Likeness Lock: GO.** Combine preserve-mark survival,
  inter-eye-normalized landmark drift, face-region proportion budgets,
  skin-color/texture budgets, and optional embedding warnings. Stages may be
  attenuated or rolled back and must explain the reason.
- **P7 — Output-Conditioned Master: GO.** One non-destructive master produces
  linked social, phone/4K, portfolio P3, and print derivatives with
  destination-specific texture, sharpening, gamut, grain, and compression
  parameters.
- **P1.1 — Sensor Calibration Pack: add under P1.** Cover camera/ISO/shutter
  dark frames, bad/hot-pixel maps, camera/lens/aperture flat fields, dust
  detection, and RAW-domain correction before demosaicing.
- **Metamorphic Robustness Lab: add as a cross-cutting QA gate.** Exercise
  orientation, resize/crop, JPEG quality, exposure, white balance, bit depth,
  gamut, and proxy/native variants; compare masks, suggestions, decisions,
  geometry, and perceptual output after undoing the transform.
- **Intrinsic portrait relighting: deferred experimental spike.** No
  production slot until identity, tone-parity, and shadow-boundary gates pass;
  no generative face replacement.
