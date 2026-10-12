# PLAN - Cosplay Portrait Polish Pipeline

**Status:** First QA-and-recipe slice implemented (2026-07-17); body-parity
rewrite remains pending calibrated visual evidence.  
**North star:** owner-supplied editorial cosplay portrait in
`~/Downloads/Screenshot_20260717_001633_com_twitter_android_MainActivity.jpg`.
**Goal:** produce a polished cosplay portrait from a real photograph while
preserving the subject's identity, makeup, costume detail, and real lighting.

This is not an AI-image imitation feature. The reference has generative-image
advantages (idealized skin, lighting, anatomy, and lens behaviour) that a
retouch engine must not fabricate. The production target is instead the
credible photographic equivalent: luminous, even skin; consistent face-to-body
finish; deliberate eye and lip detail; controlled highlight bloom; and no
plastic blur, seams, or erased cosplay styling.

## 1. Visual contract

The pipeline must deliver all of the following together:

| Property | What it means in a real cosplay portrait | Must not become |
|---|---|---|
| Porcelain skin | Reduced blotch and distracting texture while retaining form shading | White, flat, or airbrushed plastic |
| Face-body continuity | Face, neck, chest, arms, hands, and visible legs share compatible texture and colour treatment | A highly filtered face on untreated body skin |
| Intentional styling | Preserve wig fibres, drawn makeup, lashes, brows, beauty marks, jewelry, lace, tattoos, and costume edges | Skin defects or smoothing targets |
| Dimensional light | Keep nose, cheeks, collarbones, limbs, and existing specular direction readable | Global exposure lift or erased shadows |
| Editorial colour | Harmonize skin with wig/costume palette and scene shadows | A global LUT that contaminates skin or whites |
| Natural focal hierarchy | Eyes/lips slightly clearer than surrounding skin; subject separated from background | Fake catchlights, excessive sharpening, or artificial bokeh |

## 2. Existing foundations and gaps

| Capability | Current state | Gap for this plan |
|---|---|---|
| Skin frequency, albedo, specular, and texture tools | Implemented across R9-R13 | Need a single conservative portrait orchestrator, not independent sliders |
| Face lighting estimate | `retouch/lighting.py` foundation exists | No visual consumer yet |
| Face regions / parser | Face, eyes, brows, lips, hair, neck masks exist | Need a shared exposed-body portrait mask and boundary confidence |
| Body skin stage | Reachable | Single guided filter; lacks face-quality texture parity and mark policy |
| Per-mark classification | S1 tone-safe freckle prerequisite complete | S2 unified `MarkRecord` / policy is still required |
| Eye and lip tools | Existing conservative enhancers | Need coordinated, light-aware finishing at portrait strength |
| QA / backoff | `qa_detectors.py` and `QABackoff` exist | No face-body texture/seam objective or portrait-specific visual review suite |
| Recipes | Recipe loader and batch QA exist | Need a deliberate `cosplay_porcelain` recipe after the controls are safe |

## 3. Pipeline architecture

Add a leaf orchestrator, proposed as `retouch/portrait_polish.py`, called from
the existing engine after face parsing and before final global grading.
It should compose existing leaf operators; it must not duplicate their maths.

```text
semantic masks + light estimate
    -> preserve masks (makeup / marks / hair / costume / tattoos)
    -> face finish and body finish using compatible texture targets
    -> local feature finish (eyes, lips, under-eye)
    -> palette-aware editorial grade + highlight bloom
    -> portrait harmony QA / backoff
```

All stages must use float32 internally and return the input byte-identically
when `portrait_polish_strength == 0` or when no confident face is present.

### 3.1 First implementation slice (2026-07-17)

Implemented the safe prerequisite, without yet rewriting the body-skin effect:

- `retouch/harmony.py` provides read-only face/body texture-parity, specular-
  parity, and differential-banding measurements.
- Its body mask is face-anchored from the subject's own Lab chroma locus and
  used for QA only. It does not replace the production body mask yet.
- `qa_detectors.run_all` records the harmony result. `flagged` deliberately
  remains false until thresholds are calibrated on a diverse real-photo set;
  `QABackoff` therefore cannot silently alter portrait settings.
- `cosplay_portrait_polish_v1` is GUI/CLI-visible. It composes existing
  texture-preserving face polish, restrained eye/lip finish, body tone match,
  directional body support, and subtle editorial finish. P4 full-face makeup
  unmix is explicitly absent.

Native visual QA was run on the owner-provided editorial reference and
`DSCF7600.jpg`; comparison panels are in
`test_output/cosplay_portrait_polish_demo/`. The reference is already highly
polished, so its change is intentionally subtle. The real-photo panel is the
more useful initial recipe demonstration.

### 3.2 Continuation (2026-07-17, second slice)

`harmony.py` completed to the full H1/H3/H4/H5 spec (mark retention +
differential banding), its body-mask contiguity bug fixed, and the gates
re-baselined through real `engine.process()` renders. Key outcome for THIS
plan: **S2 face-body parity is now additionally blocked on an engine bug** —
`_stage_body_skin` can silently no-op when its exclusion stack severs the
face-contiguity corridor (root-caused on DSCF4503), and the current body op
moves mostly low-band tone, so a texture-parity auto-tune has no gradient to
descend. Fix the body stage's contiguity + add mid/high-band body treatment
(this plan's S2 three-band design) before any auto-tune. Full evidence:
`PLAN_BLEMISH_POLICY_HARMONY.md` §5.2 and
`test_output/harmony_engine_baseline/`. The S0 baseline-suite requirement
gains a corollary: the suite needs portraits with real exposed chest/arms —
the current cosplay set is mostly clothed.

## 4. Staged implementation

### S0 - Baseline suite and no-regression harness

**Files:** `scripts/visual_qa_portrait_polish.py` (new),
`tests/test_portrait_polish.py` (new), `test_output/portrait_polish/`.

Build a fixed, inspectable set of at least eight portraits: light and dark
skin tones, clean beauty portrait, hard flash, mixed warm/cool lighting,
blue/pink wig cosplay, exposed arms/chest, tattoos or body paint, and a
low-light portrait. For each input, save original, polished, 4x delta,
face/body masks, preserve masks, and a contact sheet.

**Gate:** no recipe is considered visual-pass based on a single flattering
portrait. The suite must include the owner's cosplay images but not depend only
on them.

### S1 - Semantic preservation and portrait masks

**Dependencies:** `PLAN_BLEMISH_POLICY_HARMONY.md` S2.

Create a `PortraitRegions` value object from existing parser masks and body
masking. It must expose confident masks for:

- face skin, neck, exposed body skin, hands
- eyes, brows, lips, mouth interior, hair/wig
- costume/jewelry boundary exclusion
- preserved marks, tattoos, and drawn makeup

Do not invent a generic exposed-skin detector from fixed RGB/LAB thresholds.
Body skin must be anchored to the parsed face-skin locus with a conservative
fallback to the existing body mask.

**Gate:** no smoothing or light edit outside confident skin masks; all
preserve-mask pixels are byte-identical before any optional feature-specific
operation explicitly owns them.

### S2 - Face-body texture and tonal parity

**Dependencies:** S1, `PLAN_BLEMISH_POLICY_HARMONY.md` S5/S6.

Replace body skin's single guided-filter treatment with the same three-band
policy used for face finishing:

1. Low band: reduce blotch and uneven albedo, respecting face-relative skin
   colour and preserving broad limb/collarbone shading.
2. Mid band: attenuate distracting texture by strength, never erase it.
3. High band: retain or restore compatible microtexture with
   `texture_transplant`; do not copy facial pores literally onto the body.

Match *texture statistics* and local tone between adjacent face/neck/chest
zones, rather than forcing all skin to one LAB median. Use a seam band around
the neck and shoulder boundary to feather only the correction difference.

**QA gates:**

- Face/body pore-spectrum distance and local DeltaE stay within calibrated
  ranges on the baseline suite.
- Neck seam score improves or remains neutral.
- No banding on smooth chest, arm, thigh, or cheek ramps.
- Mandatory visual montage review; a small delta is not evidence of safety.

### S3 - Light-aware porcelain finish

**Dependencies:** S1, face light-direction estimate.

Add a conservative `portrait_finish` light pass driven only by existing scene
evidence:

- attenuate only *excess* specular hotspots, never all highlights
- protect narrow highlight ridges on nose bridge, cheeks, shoulders, and
  collarbones when they match the inferred key-light direction
- use local dodge/burn at low frequency to restore form after albedo evening
- add optional soft bloom only around existing clipped or near-clipped
  highlights, masked away from eyes, lashes, hair fibres, lace, and jewelry

No synthetic catchlights, no geometry changes, no global whitening, and no
light direction inference with low confidence. A low-confidence estimate must
skip this stage.

**QA gates:** source highlight direction is unchanged; highlight clipping does
not increase; skin-tone DeltaE remains bounded; disabled/low-confidence output
is byte-identical.

### S4 - Feature focal pass

**Dependencies:** S1 preservation masks, S3 light context.

Coordinate existing small-region operators under one capped strength:

- under-eye: local baseline-relative vascular attenuation, then minimal L lift
- eyes: preserve lashes and existing catchlights; improve iris/limbal local
  contrast only when iris confidence is high
- lips: retain texture and shape; optional satin/gloss enhancement only on
  existing specular evidence

This stage should make eyes and lips the focal points without making them look
generated. Do not synthesize catchlights in v1.

**QA gates:** iris, pupil, lash, brow, and lip-boundary masks remain sharp;
no eye-whitening halo; no lip-edge colour bleed; mirror-symmetry does not get
worse.

### S5 - Palette-aware editorial grade

**Dependencies:** S1 masks; separate from white balance.

Implement a constrained grade that samples non-skin palette anchors from the
wig, costume, and background. It may steer shadows and highlight tint with a
small split-tone adjustment, while skin is held near its corrected source
locus. For the supplied pink-wig reference, the intended family is cool
blue-gray shadows, warm-neutral skin, and restrained pink support - not a
blanket magenta filter.

Add optional subtle subject-preserving vignette/background lens blur only when
the subject mask is confident. This is a composition finish, not a background
replacement.

**QA gates:** skin DeltaE is bounded independently from scene grade; neutral
costume whites remain neutral enough; no hue discontinuity at hair/skin or
skin/costume boundaries.

### S6 - Recipe, controls, and production QA

**Dependencies:** S0-S5.

Expose one master strength plus advanced sub-controls:

```text
portrait_polish_strength
  skin_finish_strength
  body_parity_strength
  feature_focal_strength
  editorial_grade_strength
  highlight_bloom_strength
  preserve_policy
```

Add JSON recipes through the recipe loader, not `recipes.py`:

- `cosplay_porcelain.json`: luminous, polished, conservative body parity
- `cosplay_editorial.json`: lower skin reduction, stronger palette/light finish
- `natural_editorial.json`: restrained version for non-cosplay portraits

Wire the controls to GUI/CLI only after the visual suite passes. Recipe sweep
output must include side-by-side originals, results, deltas, masks, and a
manifest of the active stages.

**Release gate:** all S0 portraits visually reviewed; QA backoff has a tested
response for plastic texture, face/body mismatch, hue drift, banding, and
halo detections; no default recipe enables P4 full-face makeup unmixing.

## 5. Algorithm guardrails

- Use within-face or face-anchored relative measurements. Do not add absolute
  skin-luminance, chroma, melanin, or hemoglobin thresholds.
- Default unknown marks to preserve. Do not heal a detail merely because it is
  small or high contrast.
- Do not alter anatomy, eye direction, breast/body shape, or clothing shape in
  this pipeline. Geometry tools remain separate, explicit creative controls.
- Keep color correction, light finishing, and texture finishing independently
  disableable for diagnosis and QA.
- Never let a background/wig/costume pixel enter a skin smooth, bloom, or
  colour-matching mask.
- Keep output float32 until the final write; test long gradients for banding.

## 6. Recommended build order

1. Complete S2 `MarkRecord` policy in the blemish/harmony plan.
2. Build S0 baseline harness and S1 portrait masks in parallel.
3. Implement S2 face-body texture parity, then S3 light-aware finish.
4. Add S4 feature focal pass and S5 palette grade only after the base skin
   finish passes visual QA.
5. Package S6 recipes and GUI/CLI controls last.

**Highest immediate ROI:** face-body parity. In polished cosplay portraits,
the giveaway failure is a smooth face ending abruptly at an untreated neck,
chest, arm, or hand. This must be solved before adding stronger bloom, grading,
or eye effects.

## 7. Explicit non-goals for v1

- Full-face foundation separation or coverage evening. P4 Track B is a
  separate research investment.
- Generative replacement of pores, makeup, catchlights, anatomy, or costume.
- Global beauty filters that brighten/whiten every subject toward one skin tone.
- Automated body reshape or liquify.
