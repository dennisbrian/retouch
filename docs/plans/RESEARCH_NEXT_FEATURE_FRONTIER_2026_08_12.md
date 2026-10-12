# Research — Next Feature Frontier After Advanced Retouch

**Date:** 2026-08-12
**Status:** Research complete; implementation not authorized by this document
**Scope:** New product and workflow features beyond
`PLAN_NEXT_FEATURE_IMPROVEMENTS_2026_08_12.md`
**Research method:** Current repository audit plus official product and platform
documentation reviewed on 2026-08-12

## 1. Executive finding

After Advanced Retouch, the strongest product opportunity is not another face
slider. It is a **Project / Shoot Intelligence workflow**:

1. explainable assisted culling;
2. manual or opt-in subject linking across a shoot;
3. subject-consistent retouch settings and protected-mark memory;
4. watch-folder/tether-style ingest and automatic processing; and
5. edit-dependency status so masks and model-backed edits cannot silently go
   stale.

This moves the engine from “process this image” to “finish this portrait shoot.”
The repository already has batch jobs, per-face settings, recipe/session state,
quality detectors, look extraction, and shoot-consistency primitives. The
missing layer is project-level orchestration and identity-safe reuse.

## 2. Current competitor signals

The following are product signals, not instructions to copy implementations.

- Adobe Lightroom Assisted Culling scores subject sharpness, eye sharpness,
  eyes-open state, exposure issues, and misfires, while preserving manual
  select/reject overrides and showing why a score was assigned.
  [Official Lightroom documentation](https://helpx.adobe.com/lightroom/desktop/organize-photos/assisted-culling.html)
- Lightroom also exposes AI Edit Status and a defined operation order so users
  can see which model-backed edits need recomputation after upstream changes.
  [Official Lightroom AI-edit documentation](https://helpx.adobe.com/uk/lightroom/web/edit-photos/manage-ai-edits.html)
- Capture One recalculates subject, person, and clothing masks for each image
  when styles or next-capture adjustments are applied. It also exposes those
  semantic areas as separate editable masks.
  [Official Capture One masking documentation](https://support.captureone.com/hc/en-us/articles/14055231933853-AI-Masking)
- Capture One Match Look supports multiple reference images, similarity-based
  weighting, adjustment-group selection, and an impact control rather than
  returning only a flattened color transfer.
  [Official Capture One Match Look documentation](https://support.captureone.com/hc/en-us/articles/22188770298269-Match-Look-Tool)
- Evoto supports per-person editing and synchronizing those adjustments to the
  same person across a project. Its demographic tagging is not required for the
  useful workflow; subject linking and manual override are the relevant parts.
  [Official Evoto portrait-retouch documentation](https://support.evoto.ai/portrait-retouching-feature-module/)
- Capture One's Next Capture Adjustments can apply all or selected adjustments
  from the last, primary, or clipboard image to future captures.
  [Official Capture One tether workflow documentation](https://support.captureone.com/hc/en-us/articles/360002556677-Adding-adjustments-to-captured-images)
- Retouch4me can export dodge-and-burn work as a neutral Soft Light layer,
  showing continued demand for non-destructive handoff to Photoshop rather
  than only a flattened result.
  [Official Retouch4me Dodge and Burn documentation](https://global.retouch4.me/dodgeburn)
- Topaz positions face recovery, super focus, object removal, RAW denoise,
  sharpening, and upscaling as separate, inspectable enhancement operations.
  [Official Topaz enhancement documentation](https://docs.topazlabs.com/topaz-photo/enhancements)
- C2PA provides current implementation guidance and open SDK options for
  signing, reading, and verifying Content Credentials.
  [C2PA developer guidance](https://c2pa.ai/for-developers)

## 3. Repository boundary discovered during research

Do not plan these as new features because the repository already has them or
has active work for them:

- Advanced Retouch has an uncommitted worktree implementation in progress;
  this research does not edit or duplicate it.
- Single-image per-face recipe selection already exists.
- Shoot-consistency white-balance logic already exists, but not a persistent
  subject-linked project profile.
- Look extraction exists, but not a multi-reference weighted look board.
- C2PA/JUMBF detection exists, but writing, re-embedding, signing, and export
  integration are incomplete.
- Face-aware smile reshaping exists; expression generation is not proposed.
- Flyaway cleanup, backdrop cleanup, fabric smoothing, body reshape, semantic
  clothing masks, NAFNet denoise, and QA detectors already exist.
- Video has a separate V0–V4 plan and remains gated by temporal evidence.

## 4. Candidate feature ranking

| Rank | ID | Feature | User value | Repository fit | Effort/risk | Verdict |
|---|---|---|---|---|---|---|
| 1 | W1 | Explainable assisted culling and burst grouping | Removes the slowest pre-edit step | Reuses detection, QA, jobs, and gallery UI | Medium | **GO** |
| 2 | W2 | Subject-linked project profiles | Consistent person-specific edits across a shoot | Extends per-face params and shoot consistency | High; privacy-sensitive | **GO after W1** |
| 3 | W3 | Watch Folder / Next Capture workflow | Studio and event automation | Reuses jobs and batch processor | Medium | **GO** |
| 4 | W4 | Edit dependency and status graph | Prevents stale masks and wrong operation order | Fits sessions, stages, models, diagnostics | Medium | **GO** |
| 5 | W5 | Layer/mask handoff export | Keeps professionals in a non-destructive workflow | Reuses semantic masks and stage outputs | Medium–high | **GO** |
| 6 | W6 | Multi-reference Look Board | Better color consistency than one-reference transfer | Extends LookExtractor and style library | Medium | **GO** |
| 7 | T1 | C2PA write/sign/verify export | Trust, attribution, and honest AI disclosure | Reader already exists | Medium; signing design | **GO** |
| 8 | Q1 | Source-only Best Face rescue | Saves group photos with blinks or poor expression | Depends on W1 burst grouping and face alignment | High | **SPIKE** |
| 9 | Q2 | Glasses reflection attenuation | Frequent portrait pain point not solved by current eye tools | Can reuse specular, parsing, heal, and light models | High; reconstruction risk | **SPIKE** |
| 10 | Q3 | Portrait perspective correction | High-wow correction for close wide-angle portraits | Existing face geometry is a starting point | Very high | **MOONSHOT** |
| 11 | Q4 | Identity-safe low-resolution face recovery | Helps distant group faces and old photos | Needs a real model and identity QA | High; hallucination risk | **PARK until model gate** |
| 12 | Q5 | Wardrobe cleanup suite | Strong cosplay/studio value: lint, moire, loose threads | Clothing mask and fabric stage already exist | Medium–high | **SPIKE after workspace** |

## 5. W1 — Explainable assisted culling and burst grouping

### Product workflow

- Point the project at a folder without moving or deleting source files.
- Group visually similar frames into bursts using capture time plus perceptual
  similarity.
- Score each frame for:
  - face/subject sharpness;
  - per-face eye sharpness;
  - eyes open / closed / uncertain;
  - exposure clipping and shadow failure;
  - severe motion blur;
  - duplicate similarity;
  - face coverage and detection confidence;
  - optional composition heuristics.
- Recommend a hero frame per burst and show the reasons and scores.
- Let the user override every recommendation.
- Export ratings/labels or a selection manifest; never delete originals by
  default.

### Technical fit

- Reuse `qa_detectors.py`, `image_analyzer.py`, face detection, the Job
  Dashboard, and gallery components.
- Add image-level and per-face `CullScore` records with versioned scoring rules.
- Start with deterministic classical metrics. A learned ranker can be added
  later only if a consented preference dataset shows a measurable advantage.

### Acceptance

- A labeled local burst corpus measures top-1 hero agreement and false-reject
  rate.
- Every score is inspectable and has an `uncertain` state.
- Manual overrides survive rescoring.
- No source deletion occurs without a separate explicit action and confirmation.

## 6. W2 — Subject-linked project profiles

### Product workflow

- Let the user manually link detected faces that represent the same person.
- Optionally offer local face-embedding suggestions, with confirmation before
  linking.
- Store per-subject:
  - preferred recipe and strength overrides;
  - protected mole/freckle/beauty-mark registry;
  - skin locus and texture targets;
  - eye/teeth/makeup preferences;
  - approved reshape settings;
  - QA back-off history.
- Apply the profile across selected images while recomputing masks for each
  image.

### Privacy boundary

- Do not auto-classify gender, age, ethnicity, or attractiveness.
- Embeddings remain local, project-scoped, encrypted or protected by normal
  project filesystem permissions, and deletable.
- Manual linking must work without embeddings.
- Never reuse a subject identity across projects unless the user explicitly
  exports/imports that profile.

### Acceptance

- Same-person edits remain consistent across pose and lighting changes.
- Different people are never merged silently.
- Protected marks follow the correct subject and do not transfer to another
  face.
- The user can inspect, unlink, delete, and rebuild every subject profile.

## 7. W3 — Watch Folder / Next Capture workflow

### Product workflow

- Watch an input folder for new RAW/JPEG files.
- Wait until a file is stable and completely written before reading it.
- Apply a selected recipe, smart settings, or settings copied from the last or
  nominated hero image.
- Recompute semantic masks per image.
- Queue preview and final output jobs separately.
- Optionally generate contact sheets and copy approved results to an output
  folder.

### Scope guard

Start with a filesystem watch folder. Direct camera tether control requires
vendor SDKs and should be a later adapter, not part of the first slice.

### Acceptance

- Partial files, temporary files, duplicates, and camera filename reuse do not
  trigger corrupt or repeated jobs.
- Restarting the app resumes safely without reprocessing completed inputs.
- Failures appear in the Job Dashboard and can be retried independently.

## 8. W4 — Edit dependency and status graph

### Problem

The engine has a meaningful operation order: RAW/decode, denoise, heal,
detection/parsing, geometry, face processing, global grade, local adjustments,
finish, resize, and export. Changing an upstream operation can invalidate masks,
face contexts, local edits, or model outputs.

### Product workflow

- Show each model-backed or cached edit as `current`, `needs update`,
  `unavailable`, `fallback`, or `failed`.
- Record source hash, source dimensions, pipeline version, model hash, parameter
  hash, mask version, and upstream dependency hashes.
- Provide `Update selected` and `Update all` actions.
- Explain why an edit became stale.

### Acceptance

- Crop, resize, source replacement, model update, and relevant upstream-setting
  changes invalidate only the correct descendants.
- A saved session never silently applies a stale mask to different dimensions.
- Recomputing in documented order produces the same output as a clean render.

## 9. W5 — Layer and mask handoff export

### Product workflow

Offer a professional handoff package containing:

- flattened retouched TIFF/PNG/JPEG;
- source or source reference;
- semantic masks as 16-bit grayscale PNG/TIFF;
- local-adjustment and heal masks;
- dodge-and-burn neutral-gray or delta layer;
- stage deltas for skin, relight, grade, and finish where mathematically valid;
- session JSON and a human-readable manifest.

Layered PSD/PSB is a later compatibility target. The first slice can be a
well-defined folder/ZIP package that Photoshop can import without inventing a
fragile PSD writer.

### Acceptance

- Recombining exported base plus documented layers reproduces the flattened
  output within a stated tolerance.
- Masks retain dimensions, bit depth, naming, and subject/face identity.
- The package declares operations that cannot be represented as independent
  layers rather than pretending full reversibility.

## 10. W6 — Multi-reference Look Board

### Product workflow

- Add several reference images to a named look board.
- Analyze each reference and the target image.
- Weight references by scene/portrait similarity or allow manual weights.
- Let the user enable groups independently:
  - normalization / white balance;
  - light and contrast;
  - color relationships;
  - split tone and finish;
  - grain/halation character.
- Apply with an impact control and save as an editable style.

### Acceptance

- Multiple references reduce target-to-target variation compared with a single
  arbitrary reference on a mixed-light set.
- Skin protection remains active and neutral normalization is separated from
  creative impact.
- The generated style contains editable parameters, not only a flattened LUT.

## 11. T1 — Complete C2PA export provenance

### Scope

- Preserve valid source Content Credentials where the standard permits a new
  derived-work assertion.
- Add a new signed action describing deterministic retouch operations and any
  model-backed operations actually used.
- Embed or attach the resulting manifest during export.
- Validate exported files using an independent C2PA verifier.

### Honesty boundary

- Do not claim an unchanged capture after pixels are edited.
- Do not label a path non-AI when NAFNet, LaMa, Real-ESRGAN, generative fill, or
  another learned model actually ran.
- Keep cloud publication optional; local embedding should remain available.

## 12. Quality-feature spikes

### Q1 — Source-only Best Face rescue

Use an adjacent user-selected burst frame as the donor for a blink or poor
expression in a group image. Align in face-local coordinates, transfer only the
approved face region, match exposure/color, and preserve hair/occlusion edges.
No synthesized identity or expression is allowed in the first version.

**GO gate:** W1 burst groups plus a corpus of real adjacent-frame donor pairs;
identity similarity, seam, gaze, lighting, and human Natural Output gates pass.

### Q2 — Glasses reflection attenuation

Detect lens polygons from landmarks plus frame edges, estimate reflection
confidence, and attenuate only where both eyes retain recoverable source detail.
Use the opposite eye or nearby burst frame only as an explicitly selected donor.
If the reflection is opaque, report `not recoverable` instead of inventing an
eye.

**GO gate:** reflection coverage and recoverability can be estimated reliably;
no iris duplication, frame damage, or eye asymmetry on the validation corpus.

### Q3 — Portrait perspective correction

Estimate close-camera distortion from a coarse 3D face fit and render a bounded
virtual-focal-length correction. This is not ordinary face slimming: the goal
is to reduce near-camera nose/forehead magnification while preserving identity
and background geometry.

**GO gate:** a synthetic camera-distance benchmark plus real close/wide and
longer-focal reference pairs show a consistent improvement without uncanny
shape drift.

### Q4 — Identity-safe face recovery

Restore blur or low-resolution detail only behind an explicit advanced control.
Always show the source and recovered face at 100%, cap strength by face size,
and include an identity-drift detector. Park until a verified model, provenance,
license, and real identity-preservation corpus are available.

### Q5 — Wardrobe cleanup suite

Build on the clothing mask to target lint, sensor dust on garments, moire,
loose threads, small compression marks, and fabric wrinkles while preserving
logos, seams, lace, embroidery, and intentional texture.

**GO gate:** labeled cosplay/studio garment crops demonstrate a low rate of
intentional-detail removal across dark, bright, patterned, and reflective
costumes.

## 13. Recommended delivery sequence

### Wave A — Small infrastructure with immediate value

1. W4 edit dependency/status graph design and session metadata.
2. W5 handoff package v0: masks, session, manifest, neutral D&B layer.
3. T1 C2PA writer/signing spike.
4. W1 culling metric and gallery prototype.

### Wave B — Project / Shoot mode

1. W1 production assisted culling and burst grouping.
2. W3 watch-folder ingest integrated with jobs.
3. W2 manual subject linking and project profiles.
4. W6 multi-reference Look Board.

### Wave C — High-risk quality differentiators

1. Q1 source-only Best Face.
2. Q2 glasses reflection attenuation.
3. Q5 wardrobe cleanup.
4. Q3 perspective correction.
5. Q4 face recovery only after the model/provenance gate.

## 14. Overall recommendation

After Advanced Retouch is verified, authorize a short **W1 Assisted Culling
spike** and a **W4 Edit Status design**. Together they establish the data model
for a Project / Shoot mode without immediately taking on biometric subject
linking or high-risk pixel synthesis.

Do not start all candidates together. Each feature above has an explicit gate
because the project already has a large algorithm surface; the next advantage
should come from coherent workflow and proof, not raw feature count.
